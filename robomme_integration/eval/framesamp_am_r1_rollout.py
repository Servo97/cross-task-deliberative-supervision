"""Receipt-only per-cut FrameSamp-AM oracle rollout for the FS-R1 screen.

Every replan captures the teacher on the actual on-policy FrameSamp history,
fits and seals all 18 compact layers, consumes the authenticated stack once,
and removes the query/K/V/compact payloads before the simulator advances.  The
only durable product is a small outcome receipt; this module does not publish
scores or authorize a cloud submission.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from robomme_integration.eval.framesamp_am_oracle_server import (
    MMEVLAOraclePolicyInvoker,
    derive_teacher_tap_stack_sha256,
)
from robomme_integration.training.framesamp_am_artifact import QuantizationParityThresholds
from robomme_integration.training.framesamp_am_index import (
    create_framesamp_am_trusted_index,
    load_framesamp_am_trusted_index,
)
from robomme_integration.training.framesamp_am_oracle_route import (
    OfflineFrameSampAMLayerPin,
    OfflineFrameSampAMStackRequest,
    create_offline_framesamp_am_stack_manifest,
    load_offline_framesamp_am_stack_manifest,
    resolve_offline_framesamp_am_oracle_inputs,
)
from robomme_integration.training.framesamp_am_query_bank import (
    fit_framesamp_am_from_query_banks,
    seal_bound_framesamp_am_artifact,
)
from robomme_integration.training.framesamp_am_r0_transition import (
    _array_identity,
    _canonical_json,
    _history_at_current_cut,
    _raw_observation,
    _sha256_file,
)
from robomme_integration.training.framesamp_am_teacher_producer import (
    RELEASED_CHECKPOINT_SHA256,
    FrameSampTeacherCaptureIdentity,
    SourcePinnedFrameSampTeacherProvider,
    build_framesamp_teacher_query_plan,
    derive_ordered_teacher_tap_stack_sha256,
    prepare_framesamp_teacher_forward,
    produce_framesamp_teacher_stack,
    verify_source_pinned_released_teacher,
)

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_oracle_rollout"
ALLOWED_TASKS = frozenset({"VideoPlaceButton", "RouteStick"})
BUDGETS = frozenset({64, 128, 256})
ACTION_SHAPE = (20, 8)
BFLOAT16_PARITY_THRESHOLDS = QuantizationParityThresholds(
    output_rmse_increase=0.05,
    output_relative_l2_increase=0.05,
    log_mass_rmse_increase=0.05,
    relative_mass_rmse_increase=0.05,
)


def _wait_for_worker_file(
    path: Path,
    worker: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        returncode = worker.poll()
        if returncode is not None:
            stdout, stderr = worker.communicate()
            raise RuntimeError(
                f"FS-R1 simulator worker exited {returncode} before {path.name}: "
                f"stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FS-R1 simulator worker timed out before {path.name}")
        time.sleep(0.1)


def _write_action_request(
    exchange: Path,
    *,
    replan: int,
    causal_cut_step: int,
    actions: np.ndarray,
) -> dict[str, object]:
    actions = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    if actions.shape != ACTION_SHAPE or not np.isfinite(actions).all():
        raise ValueError("FS-R1 action request must be finite shape (20,8)")
    stem = f"request-{replan:03d}"
    destination = exchange / f"{stem}.npy"
    temporary = exchange / f"{stem}.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, actions, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    manifest = {
        "actions_npy_sha256": _sha256_file(destination),
        "causal_cut_step": causal_cut_step,
        "shape": list(ACTION_SHAPE),
    }
    manifest_path = exchange / f"{stem}.json"
    temporary_manifest = exchange / f"{stem}.json.tmp"
    temporary_manifest.write_bytes(_canonical_json(manifest))
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _load_response(
    exchange: Path,
    *,
    replan: int,
    causal_cut_step: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    stem = f"response-{replan:03d}"
    manifest_path = exchange / f"{stem}.json"
    payload_path = exchange / f"{stem}.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "causal_cut_step",
        "executed_actions",
        "executed_actions_total",
        "outcome",
        "response_npz_sha256",
        "status",
        "terminal",
    }
    if set(manifest) != expected_fields:
        raise ValueError(f"FS-R1 response {replan} fields mismatch")
    if manifest["causal_cut_step"] != causal_cut_step:
        raise ValueError(f"FS-R1 response {replan} causal cut mismatch")
    if not payload_path.is_file() or _sha256_file(payload_path) != manifest["response_npz_sha256"]:
        raise ValueError(f"FS-R1 response {replan} SHA mismatch")
    executed = manifest["executed_actions"]
    if isinstance(executed, bool) or not isinstance(executed, int) or not 1 <= executed <= 16:
        raise ValueError(f"FS-R1 response {replan} executed-action count is invalid")
    terminal = manifest["terminal"]
    expected_status = "episode_terminal" if terminal else "replan_ready"
    if not isinstance(terminal, bool) or manifest["status"] != expected_status:
        raise ValueError(f"FS-R1 response {replan} terminal status is inconsistent")
    with np.load(payload_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {
            "current_image",
            "current_wrist",
            "current_state",
            "execution_images",
            "execution_states",
        }:
            raise ValueError(f"FS-R1 response {replan} array fields mismatch")
        payload = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    if len(payload["execution_images"]) != executed or len(payload["execution_states"]) != executed:
        raise ValueError(f"FS-R1 response {replan} execution arrays are unaligned")
    if not np.isfinite(payload["current_state"]).all() or not np.isfinite(payload["execution_states"]).all():
        raise ValueError(f"FS-R1 response {replan} state arrays are non-finite")
    return manifest, payload


def _compact_action_at_cut(
    *,
    policy: Any,
    invoker: MMEVLAOraclePolicyInvoker,
    raw_observation: dict[str, Any],
    task: str,
    episode_id: str,
    budget: int,
    fit_mass: bool,
    overlay_manifest_sha256: str,
    teacher_code_sha: str,
    device: Any,
    preprocess_observation: Any,
    make_attn_mask: Any,
    sample_actions: Any,
) -> tuple[np.ndarray, dict[str, object]]:
    observation, history, memory_steps = _history_at_current_cut(policy, raw_observation)
    causal_cut = int(policy.step_idx)
    identity = FrameSampTeacherCaptureIdentity.from_history(
        history,
        teacher_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
        teacher_code_sha=teacher_code_sha,
        policy_overlay_manifest_sha256=overlay_manifest_sha256,
        task_id=task,
        episode_id=episode_id,
        causal_cut_step=causal_cut,
    )
    prepared = prepare_framesamp_teacher_forward(
        policy._model,  # noqa: SLF001 - exact released policy graph.
        observation,
        history,
        identity,
        preprocess_observation=preprocess_observation,
        make_attn_mask=make_attn_mask,
    )
    provider = SourcePinnedFrameSampTeacherProvider(
        prepared,
        observation,
        sample_actions=sample_actions,
        verify_first_suffix_parity=True,
    )
    plan = build_framesamp_teacher_query_plan(
        identity,
        diffusion_timesteps=(0.5,),
        fit_split_seed=101,
        heldout_split_seed=211,
        fit_noise_samples_per_timestep=1,
        heldout_noise_samples_per_timestep=1,
    )
    teacher_stack = produce_framesamp_teacher_stack(plan, history, provider)
    layer_receipts: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="robomme-fs-r1-cut-", dir="/tmp") as scratch_text:
        scratch = Path(scratch_text)
        bundles = scratch / "bundles"
        bundles.mkdir()
        bundle_paths: list[Path] = []
        for layer, (banks, taps) in enumerate(zip(teacher_stack.banks, teacher_stack.teacher_taps, strict=True)):
            bundle = bundles / f"layer-{layer:02d}"
            bound = fit_framesamp_am_from_query_banks(
                history,
                banks,
                taps,
                budget,
                fit_mass=fit_mass,
                mass_ridge=0.0,
                value_ridge=0.0,
            )
            manifest = seal_bound_framesamp_am_artifact(
                bundle,
                bound,
                history,
                storage_dtype="bfloat16",
                parity_thresholds=BFLOAT16_PARITY_THRESHOLDS,
            )
            bundle_paths.append(bundle)
            layer_receipts.append(
                {
                    "layer_index": layer,
                    "manifest_sha256": _sha256_file(bundle / "manifest.json"),
                    "teacher_tap_sha256": manifest.teacher_tap_sha256,
                    "stored_heldout_output_relative_l2": (manifest.stored_heldout_metrics.output_relative_l2),
                    "stored_heldout_relative_mass_rmse": (manifest.stored_heldout_metrics.relative_mass_rmse),
                }
            )
        index_path = scratch / "trusted_index.json"
        index_sha = create_framesamp_am_trusted_index(index_path, bundle_paths)
        trusted = load_framesamp_am_trusted_index(
            index_path,
            expected_sha256=index_sha,
            verify_artifacts=True,
        )
        pins = tuple(
            OfflineFrameSampAMLayerPin(
                layer_index=record.layer_index,
                manifest_sha256=record.manifest_sha256,
            )
            for record in sorted(trusted.index.records, key=lambda row: row.layer_index)
        )
        stack_path = scratch / "stack.json"
        stack_sha = create_offline_framesamp_am_stack_manifest(
            stack_path,
            index_path,
            OfflineFrameSampAMStackRequest(
                trusted_index_sha256=index_sha,
                teacher_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
                teacher_code_sha=teacher_code_sha,
                task_id=task,
                episode_id=episode_id,
                causal_cut_step=causal_cut,
                requested_budget=budget,
                storage_dtype="bfloat16",
                layer_pins=pins,
            ),
        )
        stack_manifest = load_offline_framesamp_am_stack_manifest(
            stack_path,
            expected_sha256=stack_sha,
        )
        oracle = resolve_offline_framesamp_am_oracle_inputs(
            stack_path,
            expected_stack_manifest_sha256=stack_sha,
            trusted_index_path=index_path,
            expected_trusted_index_sha256=index_sha,
            active_policy_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
            active_model_dtype="bfloat16",
            expected_device_platform="gpu",
            device_or_sharding=device,
        )
        oracle.assert_request_identity(
            task_id=task,
            episode_id=episode_id,
            causal_cut_step=causal_cut,
        )
        result = invoker.infer(raw_observation, oracle.sample_actions_dynamic_inputs())
        actions = np.ascontiguousarray(np.asarray(result["actions"], dtype=np.float32))
        if actions.shape != ACTION_SHAPE or not np.isfinite(actions).all():
            raise RuntimeError("FS-R1 compact action plan is not finite shape (20,8)")
        ordered_tap_sha = derive_teacher_tap_stack_sha256(
            index_path,
            expected_trusted_index_sha256=index_sha,
            stack=stack_manifest,
        )
        producer_tap_sha = derive_ordered_teacher_tap_stack_sha256(
            identity,
            layer_manifest_sha256s=[pin.manifest_sha256 for pin in pins],
            per_layer_teacher_tap_sha256s=teacher_stack.receipt.per_layer_teacher_tap_sha256s,
        )
        if ordered_tap_sha != producer_tap_sha:
            raise RuntimeError("producer and runtime ordered teacher-tap identities disagree")
        cut_receipt = {
            "causal_cut_step": causal_cut,
            "selected_frame_indices": memory_steps.tolist(),
            "valid_framesamp_tokens": int(history.token_mask.sum()),
            "history_identity_sha256": identity.history_sha256,
            "teacher_capture_receipt_sha256": teacher_stack.receipt.sha256(),
            "trusted_index_sha256": index_sha,
            "stack_manifest_sha256": stack_sha,
            "ordered_teacher_tap_stack_sha256": ordered_tap_sha,
            "action_plan": _array_identity(actions),
            "layers": layer_receipts,
        }
        del oracle, result, teacher_stack, prepared, provider
    if Path(scratch_text).exists():
        raise RuntimeError("FS-R1 per-cut oracle payload survived cleanup")
    return actions, cut_receipt


def run_rollout(
    *,
    task: str,
    episode: int,
    budget: int,
    fit_mass: bool,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    official_checkout: Path,
    checkpoint: Path,
    runtime_root: Path,
    simulator_cuda_device: int,
    expected_jax_device_kind: str,
    expected_simulator_gpu_name: str,
    max_replans: int = 82,
    canary_replans: int | None = None,
) -> dict[str, object]:
    if task not in ALLOWED_TASKS or episode < 0 or budget not in BUDGETS:
        raise ValueError("FS-R1 task/episode/budget is outside the sealed screen")
    if (
        simulator_cuda_device < 0
        or max_replans <= 0
        or (canary_replans is not None and not 1 <= canary_replans <= max_replans)
    ):
        raise ValueError("FS-R1 device and max-replans must be nonnegative/positive")
    provenance = verify_source_pinned_released_teacher(
        policy_overlay=policy_overlay,
        expected_policy_overlay_manifest_sha256=overlay_manifest_sha256,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
    )
    overlay_src = policy_overlay.resolve(strict=True) / "src"
    sys.path.insert(0, str(overlay_src))

    import jax
    import orbax.checkpoint as ocp
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, make_attn_mask
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from openpi.shared import nnx_utils

    policy_runtime = {
        "source": "upstream_uv_lock",
        "jax": jax.__version__,
        "orbax_checkpoint": ocp.__version__,
    }
    if policy_runtime != {
        "source": "upstream_uv_lock",
        "jax": "0.5.3",
        "orbax_checkpoint": "0.11.13",
    }:
        raise RuntimeError(f"FS-R1 released-checkpoint policy runtime drifted: {policy_runtime}")

    history_pi0_source = Path(inspect.getfile(HistoryPi0)).resolve(strict=True)
    try:
        history_pi0_source.relative_to(overlay_src)
    except ValueError as error:
        raise RuntimeError("FS-R1 policy graph was not imported overlay-first") from error
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or devices[0].device_kind != expected_jax_device_kind:
        raise RuntimeError(f"FS-R1 requires exactly one {expected_jax_device_kind!r} JAX GPU; got {devices}")
    previous_cwd = Path.cwd()
    os.chdir(policy_overlay)
    try:
        policy = policy_config.create_trained_policy(
            config.get_config("mme_vla_suite"),
            checkpoint,
            seed=7,
            default_prompt=None,
        )
    finally:
        os.chdir(previous_cwd)
    invoker = MMEVLAOraclePolicyInvoker(policy, verified_policy_overlay_root=policy_overlay)
    sample_actions = nnx_utils.module_jit(policy._model.sample_actions)  # noqa: SLF001

    examples_root = official_checkout / "examples/robomme"
    simulator_paths = (
        examples_root,
        runtime_root / "ManiSkill-07be6fbc",
        runtime_root / "robomme-benchmark-f2b540e6/src",
    )
    simulator_python = runtime_root / "env-v0.4.0/bin/python"
    for path in (*simulator_paths, simulator_python):
        if not path.exists():
            raise FileNotFoundError(f"FS-R1 simulator runtime path is absent: {path}")
    simulation_temp = tempfile.TemporaryDirectory(prefix="robomme-fs-r1-sim-", dir="/tmp")
    exchange = Path(simulation_temp.name)
    mpl_cache = Path("/tmp/robomme-fs-r1-mpl-cache")
    mpl_cache.mkdir(exist_ok=True)
    worker_environment = os.environ.copy()
    worker_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(simulator_cuda_device),
            "MPLCONFIGDIR": str(mpl_cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(path) for path in simulator_paths),
        }
    )
    worker_script = Path(__file__).with_name("framesamp_r1_sim_worker.py")
    worker_command = [
        str(simulator_python),
        str(worker_script),
        "--task",
        task,
        "--episode",
        str(episode),
        "--exchange",
        str(exchange),
        "--max-replans",
        str(max_replans),
    ]
    if canary_replans is not None:
        worker_command.extend(["--stop-after-replans", str(canary_replans)])
    worker = subprocess.Popen(
        worker_command,
        cwd=examples_root,
        env=worker_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    try:
        initial_manifest_path = exchange / "initial.json"
        _wait_for_worker_file(initial_manifest_path, worker, timeout_seconds=300.0)
        initial_manifest = json.loads(initial_manifest_path.read_text(encoding="utf-8"))
        initial_payload = exchange / "initial.npz"
        if (
            set(initial_manifest)
            != {
                "difficulty",
                "episode",
                "initial_npz_sha256",
                "seed",
                "simulator_runtime",
                "task",
                "task_goal",
            }
            or initial_manifest["task"] != task
            or initial_manifest["episode"] != episode
            or _sha256_file(initial_payload) != initial_manifest["initial_npz_sha256"]
        ):
            raise RuntimeError("FS-R1 simulator initial receipt mismatch")
        simulator_runtime = initial_manifest["simulator_runtime"]
        if (
            not isinstance(simulator_runtime, dict)
            or simulator_runtime.get("cuda_visible_devices") != str(simulator_cuda_device)
            or simulator_runtime.get("gpu_name") != expected_simulator_gpu_name
            or simulator_runtime.get("torch_cuda_version") != "12.8"
        ):
            raise RuntimeError("FS-R1 simulator runtime identity mismatch")
        with np.load(initial_payload, allow_pickle=False) as loaded:
            if set(loaded.files) != {"images", "states", "wrist_images"}:
                raise RuntimeError("FS-R1 initial payload fields mismatch")
            images = np.asarray(loaded["images"], dtype=np.uint8)
            wrists = np.asarray(loaded["wrist_images"], dtype=np.uint8)
            states = np.asarray(loaded["states"], dtype=np.float32)
        if not images.size or not (len(images) == len(wrists) == len(states)):
            raise RuntimeError("FS-R1 initial demonstration is empty or unaligned")
        task_goal = str(initial_manifest["task_goal"])
        policy.reset()
        policy.add_buffer(
            {
                "images": images[:, None],
                "state": states,
                "exec_start_idx": len(images) - 1,
            }
        )
        current = _raw_observation(images[-1], wrists[-1], states[-1], task_goal)
        episode_id = f"{task}-test-{episode:03d}"
        cut_receipts: list[dict[str, object]] = []
        executed_total = 0
        outcome: str | None = None
        for replan in range(max_replans):
            causal_cut = int(policy.step_idx)
            actions, cut_receipt = _compact_action_at_cut(
                policy=policy,
                invoker=invoker,
                raw_observation=current,
                task=task,
                episode_id=episode_id,
                budget=budget,
                fit_mass=fit_mass,
                overlay_manifest_sha256=overlay_manifest_sha256,
                teacher_code_sha=str(provenance["teacher_code_sha"]),
                device=devices[0],
                preprocess_observation=preprocess_observation,
                make_attn_mask=make_attn_mask,
                sample_actions=sample_actions,
            )
            if cut_receipt["causal_cut_step"] != causal_cut:
                raise RuntimeError("FS-R1 compact receipt causal cut changed during fitting")
            request = _write_action_request(
                exchange,
                replan=replan,
                causal_cut_step=causal_cut,
                actions=actions,
            )
            response_path = exchange / f"response-{replan:03d}.json"
            _wait_for_worker_file(response_path, worker, timeout_seconds=900.0)
            response, payload = _load_response(
                exchange,
                replan=replan,
                causal_cut_step=causal_cut,
            )
            executed = int(response["executed_actions"])
            executed_total += executed
            if response["executed_actions_total"] != executed_total:
                raise RuntimeError("FS-R1 cumulative executed-action count mismatch")
            cut_receipt.update(
                {
                    "replan": replan,
                    "request_sha256": hashlib.sha256(_canonical_json(request)).hexdigest(),
                    "executed_actions": executed,
                    "executed_actions_total": executed_total,
                    "terminal": response["terminal"],
                    "outcome": response["outcome"],
                    "oracle_payload_deleted_before_simulator_response": True,
                }
            )
            cut_receipts.append(cut_receipt)
            if response["terminal"]:
                outcome = str(response["outcome"])
                break
            if canary_replans is not None and replan + 1 == canary_replans:
                outcome = "canary_nonterminal"
                break
            policy.add_buffer(
                {
                    "images": payload["execution_images"][:, None],
                    "state": np.asarray(payload["execution_states"], dtype=np.float32),
                    "exec_start_idx": 0,
                }
            )
            current = _raw_observation(
                payload["current_image"],
                payload["current_wrist"],
                payload["current_state"],
                task_goal,
            )
        if outcome is None:
            raise RuntimeError("FS-R1 rollout exhausted max replans without terminal outcome")
        stdout, stderr = worker.communicate(timeout=30.0)
        if worker.returncode != 0:
            raise RuntimeError(
                f"FS-R1 simulator worker exited {worker.returncode}: "
                f"stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
            )
        is_canary = canary_replans is not None
        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_canary" if is_canary else KIND,
            "status": "HARD_GREEN" if is_canary else "COMPLETE",
            "scope": (
                "fs_r1_runtime_canary_not_scored_evidence"
                if is_canary
                else "fs_r1_screen_receipt_only_not_statistical_evidence"
            ),
            "task": task,
            "episode": episode,
            "episode_id": episode_id,
            "seed": int(initial_manifest["seed"]),
            "difficulty": initial_manifest["difficulty"],
            "success": None if is_canary else outcome == "success",
            "outcome": outcome,
            "budget": budget,
            "fit_mass": fit_mass,
            "mass_ridge": 0.0,
            "value_ridge": 0.0,
            "storage_dtype": "bfloat16",
            "bfloat16_quantization_parity_max_metric_increase": 0.05,
            "initial_demo_frames": len(images),
            "replans": len(cut_receipts),
            "canary_replans": canary_replans,
            "executed_actions": executed_total,
            "cuts": cut_receipts,
            "fresh_attested_stack_fraction": 1.0,
            "persistent_oracle_payloads": False,
            "ephemeral_artifacts_deleted_before_receipt": True,
            "policy_overlay_manifest_sha256": overlay_manifest_sha256,
            "patched_history_pi0_import": str(history_pi0_source),
            "teacher_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
            "teacher_code_sha": provenance["teacher_code_sha"],
            "policy_runtime": policy_runtime,
            "device": {
                "count": 1,
                "platform": devices[0].platform,
                "device_kind": devices[0].device_kind,
            },
            "simulator_runtime": simulator_runtime,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
        return receipt
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10.0)
        simulation_temp.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(ALLOWED_TASKS), required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=sorted(BUDGETS), required=True)
    parser.add_argument("--fit-mass", action="store_true")
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--simulator-cuda-device", type=int, required=True)
    parser.add_argument("--expected-jax-device-kind", required=True)
    parser.add_argument("--expected-simulator-gpu-name", required=True)
    parser.add_argument("--max-replans", type=int, default=82)
    parser.add_argument("--canary-replans", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    receipt = run_rollout(
        task=args.task,
        episode=args.episode,
        budget=args.budget,
        fit_mass=args.fit_mass,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        runtime_root=args.runtime_root,
        simulator_cuda_device=args.simulator_cuda_device,
        expected_jax_device_kind=args.expected_jax_device_kind,
        expected_simulator_gpu_name=args.expected_simulator_gpu_name,
        max_replans=args.max_replans,
        canary_replans=args.canary_replans,
    )
    with args.output.open("xb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"receipt_sha256": receipt["receipt_sha256"], "success": receipt["success"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
