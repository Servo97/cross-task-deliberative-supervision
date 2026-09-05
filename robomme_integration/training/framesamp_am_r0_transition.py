"""One real later-cut FrameSamp-AM compact-all transition.

This is the FS-R0 correctness gate, not a scored evaluation.  It rolls the
released FrameSamp policy for one 16-action chunk in a real RoboMME episode,
captures the resulting on-policy FrameSamp history, produces and consumes one
all-18-layer compact-all stack, proves that a stale history is rejected, and
deletes every query/K/V/artifact payload before writing a small sealed receipt.
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

from robomme_integration.amkv.driver import framesamp_history_from_observation
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
from robomme_integration.training.upstream_framesamp_data import FrameSampHistory

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r0_real_later_cut_transition"
EXECUTION_HORIZON = 16
ALLOWED_TASKS = frozenset({"VideoPlaceButton", "RouteStick"})
VALUE_RIDGE_GRID = (0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
BFLOAT16_PARITY_THRESHOLDS = QuantizationParityThresholds(
    output_rmse_increase=0.05,
    output_relative_l2_increase=0.05,
    log_mass_rmse_increase=0.05,
    relative_mass_rmse_increase=0.05,
)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "raw_bytes_sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


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
                f"FS-R0 simulator worker exited {returncode} before {path.name}: "
                f"stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FS-R0 simulator worker timed out before {path.name}")
        time.sleep(0.1)


def _write_action_request(exchange: Path, actions: np.ndarray) -> None:
    actions = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    temporary = exchange / "actions.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, actions, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    actions_path = exchange / "actions.npy"
    os.replace(temporary, actions_path)
    manifest = {
        "actions_npy_sha256": _sha256_file(actions_path),
        "shape": list(actions.shape),
    }
    temporary_manifest = exchange / "actions.json.tmp"
    temporary_manifest.write_bytes(_canonical_json(manifest))
    os.replace(temporary_manifest, exchange / "actions.json")


def _raw_observation(image: np.ndarray, wrist: np.ndarray, state: np.ndarray, prompt: str) -> dict[str, Any]:
    return {
        "observation/image": np.asarray(image),
        "observation/wrist_image": np.asarray(wrist),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


def _history_at_current_cut(policy: Any, raw: dict[str, Any]) -> tuple[Any, FrameSampHistory, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    inputs = policy._prepare_history(dict(raw))  # noqa: SLF001 - exact released infer path.
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    observation = HistAugObservation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs))
    memory_steps = np.asarray(
        policy.mem_buffer.get_frame_sampling_indices(
            policy.step_idx,
            policy.config.budget,
            policy.config.token_per_image,
        ),
        dtype=np.int64,
    )
    history = framesamp_history_from_observation(observation, memory_steps)
    if int(history.frame_indices[history.frame_mask][-1]) != int(policy.step_idx):
        raise RuntimeError("actual FrameSamp history does not terminate at the current policy cut")
    return observation, history, memory_steps


def _stale_history(history: FrameSampHistory) -> FrameSampHistory:
    image = np.array(history.image, copy=True)
    first = int(np.flatnonzero(history.token_mask)[0])
    image[first, 0] = np.nextafter(image[first, 0], np.float32(np.inf))
    stale = FrameSampHistory(
        image=image,
        position=np.array(history.position, copy=True),
        token_mask=np.array(history.token_mask, copy=True),
        frame_indices=np.array(history.frame_indices, copy=True),
        frame_mask=np.array(history.frame_mask, copy=True),
    )
    stale.validate()
    return stale


def run_transition(
    *,
    task: str,
    episode: int,
    budget: int,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    official_checkout: Path,
    checkpoint: Path,
    runtime_root: Path,
    simulator_cuda_device: int,
) -> dict[str, object]:
    if task not in ALLOWED_TASKS:
        raise ValueError(f"FS-R0 task must be one of {sorted(ALLOWED_TASKS)}")
    if episode < 0 or budget not in {64, 128, 256}:
        raise ValueError("episode must be nonnegative and budget must be 64, 128, or 256")
    if simulator_cuda_device < 0:
        raise ValueError("simulator CUDA device must be nonnegative")
    provenance = verify_source_pinned_released_teacher(
        policy_overlay=policy_overlay,
        expected_policy_overlay_manifest_sha256=overlay_manifest_sha256,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
    )

    overlay_src = policy_overlay.resolve(strict=True) / "src"
    sys.path.insert(0, str(overlay_src))

    import jax
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, make_attn_mask
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from openpi.shared import nnx_utils

    history_pi0_source = Path(inspect.getfile(HistoryPi0)).resolve(strict=True)
    try:
        history_pi0_source.relative_to(overlay_src)
    except ValueError as error:
        raise RuntimeError(
            "FS-R0 policy graph was not imported overlay-first; launch with the "
            "verified policy overlay src first on PYTHONPATH"
        ) from error

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("FS-R0 requires exactly one visible GPU")
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

    examples_root = official_checkout / "examples/robomme"
    simulator_paths = (
        examples_root,
        runtime_root / "ManiSkill-07be6fbc",
        runtime_root / "robomme-benchmark-f2b540e6/src",
    )
    simulator_python = runtime_root / "env-v0.4.0/bin/python"
    for path in (*simulator_paths, simulator_python):
        if not path.exists():
            raise FileNotFoundError(f"FS-R0 simulator runtime path is absent: {path}")
    simulation_temp = tempfile.TemporaryDirectory(prefix="robomme-fs-r0-sim-", dir="/tmp")
    exchange = Path(simulation_temp.name)
    simulator_mpl_cache = Path("/tmp/robomme-fs-r0-mpl-cache")
    simulator_mpl_cache.mkdir(exist_ok=True)
    worker_environment = os.environ.copy()
    worker_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(simulator_cuda_device),
            "MPLCONFIGDIR": str(simulator_mpl_cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(path) for path in simulator_paths),
        }
    )
    worker_script = Path(__file__).resolve().parents[1] / "eval/framesamp_r0_sim_worker.py"
    worker = subprocess.Popen(
        [
            str(simulator_python),
            str(worker_script),
            "--task",
            task,
            "--episode",
            str(episode),
            "--exchange",
            str(exchange),
        ],
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
        if _sha256_file(initial_payload) != initial_manifest["initial_npz_sha256"]:
            raise RuntimeError("FS-R0 simulator initial payload SHA mismatch")
        with np.load(initial_payload, allow_pickle=False) as payload:
            images = np.asarray(payload["images"])
            wrists = np.asarray(payload["wrist_images"])
            states = np.asarray(payload["states"], dtype=np.float32)
        if not images.size or not (len(images) == len(wrists) == len(states)):
            raise RuntimeError("FS-R0 initial demonstration is empty or unaligned")
        seed = int(initial_manifest["seed"])
        difficulty = initial_manifest["difficulty"]
        simulator_runtime = initial_manifest["simulator_runtime"]
        if (
            not isinstance(simulator_runtime, dict)
            or "RTX 5090" not in str(simulator_runtime.get("gpu_name", ""))
            or simulator_runtime.get("cuda_visible_devices") != str(simulator_cuda_device)
            or simulator_runtime.get("torch_cuda_version") != "12.8"
        ):
            raise RuntimeError("FS-R0 simulator runtime identity mismatch")
        task_goal = str(initial_manifest["task_goal"])
        policy.reset()
        policy.add_buffer(
            {
                "images": images.astype(np.uint8)[:, None],
                "state": states,
                "exec_start_idx": len(images) - 1,
            }
        )
        current = _raw_observation(images[-1], wrists[-1], states[-1], task_goal)
        baseline = policy.infer(current)
        baseline_actions = np.asarray(baseline["actions"])
        if baseline_actions.shape != (20, 8) or not np.isfinite(baseline_actions).all():
            raise RuntimeError("released first action plan is not finite shape (20,8)")
        _write_action_request(exchange, baseline_actions)
        later_manifest_path = exchange / "later.json"
        _wait_for_worker_file(later_manifest_path, worker, timeout_seconds=300.0)
        stdout, stderr = worker.communicate(timeout=30.0)
        if worker.returncode != 0:
            raise RuntimeError(
                f"FS-R0 simulator worker exited {worker.returncode}: "
                f"stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
            )
        later_manifest = json.loads(later_manifest_path.read_text(encoding="utf-8"))
        later_payload = exchange / "later.npz"
        if (
            later_manifest.get("status") != "later_cut_ready"
            or later_manifest.get("executed_actions") != EXECUTION_HORIZON
            or _sha256_file(later_payload) != later_manifest.get("later_npz_sha256")
        ):
            raise RuntimeError("FS-R0 simulator later-cut receipt mismatch")
        with np.load(later_payload, allow_pickle=False) as payload:
            execution_images = np.asarray(payload["execution_images"], dtype=np.uint8)
            execution_states = np.asarray(payload["execution_states"], dtype=np.float32)
            current = _raw_observation(
                np.asarray(payload["current_image"], dtype=np.uint8),
                np.asarray(payload["current_wrist"], dtype=np.uint8),
                np.asarray(payload["current_state"], dtype=np.float32),
                task_goal,
            )
        policy.add_buffer(
            {
                "images": execution_images[:, None],
                "state": execution_states,
                "exec_start_idx": 0,
            }
        )
        observation, history, memory_steps = _history_at_current_cut(policy, current)
        episode_id = f"{task}-test-{episode:03d}"
        causal_cut = int(policy.step_idx)
        identity = FrameSampTeacherCaptureIdentity.from_history(
            history,
            teacher_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
            teacher_code_sha=str(provenance["teacher_code_sha"]),
            policy_overlay_manifest_sha256=overlay_manifest_sha256,
            task_id=task,
            episode_id=episode_id,
            causal_cut_step=causal_cut,
        )
        model = policy._model  # noqa: SLF001
        prepared = prepare_framesamp_teacher_forward(
            model,
            observation,
            history,
            identity,
            preprocess_observation=preprocess_observation,
            make_attn_mask=make_attn_mask,
        )
        provider = SourcePinnedFrameSampTeacherProvider(
            prepared,
            observation,
            sample_actions=nnx_utils.module_jit(model.sample_actions),
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
        stack = produce_framesamp_teacher_stack(plan, history, provider)

        stale_rejected = False
        try:
            stack.validate(_stale_history(history))
        except ValueError as error:
            stale_rejected = "current on-policy FrameSamp history differs" in str(error)
        if not stale_rejected:
            raise RuntimeError("FS-R0 failed to reject a stale/mutated later-cut history")

        ephemeral_parent: str
        layer_summaries: list[dict[str, object]] = []
        compact_action_identity: dict[str, object]
        with tempfile.TemporaryDirectory(prefix="robomme-fs-r0-", dir="/tmp") as scratch_text:
            scratch = Path(scratch_text)
            ephemeral_parent = scratch_text
            bundles_root = scratch / "bundles"
            bundles_root.mkdir()
            bundle_paths = []
            for layer, (banks, taps) in enumerate(zip(stack.banks, stack.teacher_taps, strict=True)):
                bundle = bundles_root / f"layer-{layer:02d}"
                quantization_failures = []
                manifest = None
                for value_ridge in VALUE_RIDGE_GRID:
                    bound = fit_framesamp_am_from_query_banks(
                        history,
                        banks,
                        taps,
                        budget,
                        fit_mass=True,
                        mass_ridge=0.0,
                        value_ridge=value_ridge,
                    )
                    try:
                        manifest = seal_bound_framesamp_am_artifact(
                            bundle,
                            bound,
                            history,
                            storage_dtype="bfloat16",
                            parity_thresholds=BFLOAT16_PARITY_THRESHOLDS,
                        )
                    except ValueError as error:
                        if "storage-quantization parity gate failed" not in str(error):
                            raise
                        quantization_failures.append({"value_ridge": value_ridge, "failure": str(error)})
                        continue
                    break
                if manifest is None:
                    raise RuntimeError(
                        f"layer {layer} failed strict bfloat16 parity across the sealed value-ridge grid: "
                        f"{quantization_failures}"
                    )
                manifest_sha = _sha256_file(bundle / "manifest.json")
                bundle_paths.append(bundle)
                layer_summaries.append(
                    {
                        "layer_index": layer,
                        "manifest_sha256": manifest_sha,
                        "teacher_tap_sha256": manifest.teacher_tap_sha256,
                        "stored_heldout_output_relative_l2": (manifest.stored_heldout_metrics.output_relative_l2),
                        "stored_heldout_relative_mass_rmse": (manifest.stored_heldout_metrics.relative_mass_rmse),
                        "mass_ridge": manifest.mass_ridge,
                        "value_ridge": manifest.value_ridge,
                        "smaller_value_ridge_quantization_failures": quantization_failures,
                    }
                )

            index_path = scratch / "trusted_index.json"
            index_sha = create_framesamp_am_trusted_index(index_path, bundle_paths)
            trusted = load_framesamp_am_trusted_index(
                index_path,
                expected_sha256=index_sha,
                verify_artifacts=True,
            )
            records = sorted(trusted.index.records, key=lambda record: record.layer_index)
            pins = tuple(
                OfflineFrameSampAMLayerPin(
                    layer_index=record.layer_index,
                    manifest_sha256=record.manifest_sha256,
                )
                for record in records
            )
            stack_path = scratch / "stack.json"
            stack_sha = create_offline_framesamp_am_stack_manifest(
                stack_path,
                index_path,
                OfflineFrameSampAMStackRequest(
                    trusted_index_sha256=index_sha,
                    teacher_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
                    teacher_code_sha=str(provenance["teacher_code_sha"]),
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
                device_or_sharding=devices[0],
            )
            oracle.assert_request_identity(
                task_id=task,
                episode_id=episode_id,
                causal_cut_step=causal_cut,
            )
            invoker = MMEVLAOraclePolicyInvoker(
                policy,
                verified_policy_overlay_root=policy_overlay,
            )
            compact_result = invoker.infer(
                current,
                oracle.sample_actions_dynamic_inputs(),
            )
            compact_actions = np.asarray(compact_result["actions"])
            if compact_actions.shape != (20, 8) or not np.isfinite(compact_actions).all():
                raise RuntimeError("FS-R0 compact action plan is not finite shape (20,8)")
            compact_action_identity = _array_identity(compact_actions)
            ordered_tap_sha = derive_teacher_tap_stack_sha256(
                index_path,
                expected_trusted_index_sha256=index_sha,
                stack=stack_manifest,
            )
            producer_ordered_tap_sha = derive_ordered_teacher_tap_stack_sha256(
                identity,
                layer_manifest_sha256s=[pin.manifest_sha256 for pin in pins],
                per_layer_teacher_tap_sha256s=stack.receipt.per_layer_teacher_tap_sha256s,
            )
            if ordered_tap_sha != producer_ordered_tap_sha:
                raise RuntimeError("producer and runtime ordered teacher-tap identities disagree")
            route_summary = {
                "trusted_index_sha256": index_sha,
                "stack_manifest_sha256": stack_sha,
                "ordered_teacher_tap_stack_sha256": ordered_tap_sha,
            }

        if Path(ephemeral_parent).exists():
            raise RuntimeError("FS-R0 ephemeral oracle artifacts survived cleanup")
        elapsed = time.monotonic() - started
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "HARD_GREEN",
            "scope": "correctness_only_not_scored_evaluation",
            "task": task,
            "episode": episode,
            "episode_id": episode_id,
            "seed": int(seed),
            "difficulty": difficulty,
            "initial_demo_frames": len(images),
            "executed_policy_actions": EXECUTION_HORIZON,
            "causal_cut_step": causal_cut,
            "selected_frame_indices": memory_steps.tolist(),
            "valid_framesamp_tokens": int(history.token_mask.sum()),
            "history_identity_sha256": identity.history_sha256,
            "baseline_action_plan": _array_identity(baseline_actions),
            "compact_action_plan": compact_action_identity,
            "requested_budget": budget,
            "fit_mass": True,
            "storage_dtype": "bfloat16",
            "bfloat16_quantization_parity_max_metric_increase": 0.05,
            "layer_count": len(layer_summaries),
            "layers": layer_summaries,
            "teacher_capture_receipt_sha256": stack.receipt.sha256(),
            "stale_history_rejected": True,
            "ephemeral_artifacts_deleted_before_receipt": True,
            "persistent_payloads": "receipt_only_no_query_kv_or_compact_arrays",
            "route": route_summary,
            "policy_overlay_manifest_sha256": overlay_manifest_sha256,
            "patched_history_pi0_import": str(history_pi0_source),
            "teacher_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
            "teacher_code_sha": provenance["teacher_code_sha"],
            "device": {
                "count": 1,
                "platform": devices[0].platform,
                "device_kind": devices[0].device_kind,
            },
            "simulator_runtime": simulator_runtime,
            "elapsed_seconds": float(elapsed),
        }
        record["receipt_sha256"] = hashlib.sha256(_canonical_json(record)).hexdigest()
        return record
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
    parser.add_argument("--task", choices=sorted(ALLOWED_TASKS), default="VideoPlaceButton")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--budget", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--simulator-cuda-device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {args.output.parent}")
    record = run_transition(
        task=args.task,
        episode=args.episode,
        budget=args.budget,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        runtime_root=args.runtime_root,
        simulator_cuda_device=args.simulator_cuda_device,
    )
    with args.output.open("xb") as stream:
        stream.write(_canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    args.output.chmod(0o444)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
