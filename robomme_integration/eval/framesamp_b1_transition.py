"""One real unscored transition through the FrameSamp B1 policy control."""

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

from robomme_integration.training.framesamp_b1_data import (
    B1_PARTITION_KIND,
    select_framesamp_b1,
)
from robomme_integration.training.framesamp_b1_policy_overlay import (
    PATCHED_MEM_BUFFER_SHA256,
    PATCHED_POLICY_SHA256,
    verify_framesamp_b1_policy_overlay,
)

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_b1_real_transition"
SCOPE = "receipt_only_not_scored_evidence"
EXECUTION_HORIZON = 16
H100_NAME = "NVIDIA H100 80GB HBM3"
RELEASED_CHECKPOINT_SHA256 = "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
RELEASED_CHECKPOINT_STEP = "79999"
RELEASED_HISTORY_CONFIG = "perceptual-framesamp-modul.yaml"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_file(path: Path) -> str:
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
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _raw_observation(image: np.ndarray, wrist: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    return {
        "observation/image": np.asarray(image, dtype=np.uint8),
        "observation/wrist_image": np.asarray(wrist, dtype=np.uint8),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


def _wait(path: Path, worker: subprocess.Popen[str], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if worker.poll() is not None:
            stdout, stderr = worker.communicate()
            raise RuntimeError(
                f"B1 simulator exited {worker.returncode}: stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"B1 simulator timed out before {path.name}")
        time.sleep(0.1)


def _write_actions(exchange: Path, actions: np.ndarray) -> None:
    actions = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    if actions.shape != (20, 8) or not np.isfinite(actions).all():
        raise ValueError("B1 action request must be finite shape (20,8)")
    temporary = exchange / "actions.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, actions, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    target = exchange / "actions.npy"
    os.replace(temporary, target)
    manifest = {"actions_npy_sha256": _sha_file(target), "shape": [20, 8]}
    temporary_manifest = exchange / "actions.json.tmp"
    temporary_manifest.write_bytes(_canonical(manifest))
    os.replace(temporary_manifest, exchange / "actions.json")


def _verify_checkpoint(checkpoint: Path) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.name != RELEASED_CHECKPOINT_STEP:
        raise ValueError("B1 requires released FrameSamp checkpoint step 79999")
    if not (checkpoint / "params").is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise FileNotFoundError("released FrameSamp checkpoint tree is incomplete")
    if not (checkpoint.parent / f".EXTRACTED-{RELEASED_CHECKPOINT_SHA256}").is_file():
        raise FileNotFoundError("released FrameSamp semantic extraction marker is absent")
    history = checkpoint.parent / "history_config.txt"
    if not history.is_file() or history.read_text(encoding="utf-8").strip() != RELEASED_HISTORY_CONFIG:
        raise ValueError("released FrameSamp history configuration drifted")


def _cut_record(policy: Any, raw: dict[str, Any]) -> tuple[dict[str, object], np.ndarray]:
    selection = select_framesamp_b1(policy.step_idx, policy.exec_start_idx)
    runtime_demo, runtime_live = policy.mem_buffer.get_frame_sampling_b1_indices(
        policy.step_idx, policy.exec_start_idx
    )
    if tuple(runtime_demo) != selection.demo_indices or tuple(runtime_live) != selection.live_indices:
        raise RuntimeError("staged B1 runtime selection disagrees with the sealed pure contract")
    prepared = policy._prepare_history(dict(raw))  # noqa: SLF001 - exact released infer seam.
    mask = np.asarray(prepared["static_mask"], dtype=np.bool_)
    expected_mask = np.repeat(selection.frame_mask, 16)
    if mask.shape != (512,) or not np.array_equal(mask, expected_mask):
        raise RuntimeError("B1 prepared policy mask disagrees with fixed demo/live slots")
    result = policy.infer(raw)
    actions = np.ascontiguousarray(np.asarray(result["actions"], dtype=np.float32))
    if actions.shape != (20, 8) or not np.isfinite(actions).all():
        raise RuntimeError("B1 policy produced invalid actions")
    record = {
        "step_idx": int(policy.step_idx),
        "exec_start_idx": int(policy.exec_start_idx),
        "demo_indices": list(selection.demo_indices),
        "live_indices": list(selection.live_indices),
        "frame_indices": selection.frame_indices.tolist(),
        "frame_mask_sha256": _array_identity(selection.frame_mask)["sha256"],
        "token_mask_sha256": _array_identity(expected_mask)["sha256"],
        "valid_frames": int(selection.frame_mask.sum()),
        "valid_tokens": int(expected_mask.sum()),
        "actions": _array_identity(actions),
    }
    return record, actions


def validate_receipt(receipt: dict[str, object]) -> None:
    seal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if seal != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("B1 transition receipt seal mismatch")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("kind") != KIND
        or receipt.get("scope") != SCOPE
        or receipt.get("status") != "HARD_GREEN"
        or receipt.get("representation_policy") != B1_PARTITION_KIND
        or receipt.get("released_checkpoint_sha256") != RELEASED_CHECKPOINT_SHA256
        or receipt.get("patched_policy_sha256") != PATCHED_POLICY_SHA256
        or receipt.get("patched_mem_buffer_sha256") != PATCHED_MEM_BUFFER_SHA256
        or receipt.get("executed_actions") != EXECUTION_HORIZON
        or receipt.get("scored_evidence") is not False
        or receipt.get("cloud_publication") is not False
    ):
        raise ValueError("B1 transition receipt semantics mismatch")
    cuts = receipt.get("cuts")
    if not isinstance(cuts, list) or len(cuts) != 2:
        raise ValueError("B1 receipt must bind initial and later cuts")
    initial, later = cuts
    if (
        not isinstance(initial, dict)
        or not isinstance(later, dict)
        or initial.get("live_indices") != []
        or later.get("exec_start_idx") != initial.get("exec_start_idx")
        or later.get("demo_indices") != initial.get("demo_indices")
        or len(later.get("live_indices", [])) != EXECUTION_HORIZON
        or later.get("valid_tokens") != 512
    ):
        raise ValueError("B1 receipt did not prove a frozen demo and full raw-live transition")
    for cut in cuts:
        actions = cut.get("actions")
        if (
            not isinstance(actions, dict)
            or actions.get("dtype") != "float32"
            or actions.get("shape") != [20, 8]
            or not isinstance(actions.get("sha256"), str)
            or len(actions["sha256"]) != 64
        ):
            raise ValueError("B1 transition action identity is malformed")


def run_transition(
    *,
    task: str,
    episode: int,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    official_checkout: Path,
    checkpoint: Path,
    runtime_root: Path,
    simulator_cuda_device: int,
) -> dict[str, object]:
    overlay_manifest = verify_framesamp_b1_policy_overlay(
        policy_overlay, expected_manifest_sha256=overlay_manifest_sha256
    )
    _verify_checkpoint(checkpoint)
    overlay_src = policy_overlay.resolve(strict=True) / "src"
    sys.path.insert(0, str(overlay_src))

    import jax
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or str(devices[0].device_kind) != H100_NAME:
        raise RuntimeError("B1 transition requires exactly one visible H100 80GB HBM3")
    previous_cwd = Path.cwd()
    os.chdir(policy_overlay)
    try:
        policy = policy_config.create_trained_policy(
            config.get_config("mme_vla_suite"), checkpoint, seed=7, default_prompt=None
        )
    finally:
        os.chdir(previous_cwd)
    policy_source = Path(inspect.getfile(policy.__class__)).resolve(strict=True)
    if policy_source != overlay_src.joinpath("mme_vla_suite/policies/policy.py"):
        raise RuntimeError("B1 policy class was not imported from the authenticated overlay")

    examples_root = official_checkout / "examples/robomme"
    simulator_paths = (
        examples_root,
        runtime_root / "ManiSkill-07be6fbc",
        runtime_root / "robomme-benchmark-f2b540e6/src",
    )
    simulator_python = runtime_root / "env-v0.4.0/bin/python"
    for path in (*simulator_paths, simulator_python):
        if not path.exists():
            raise FileNotFoundError(f"B1 simulator dependency is absent: {path}")
    simulation = tempfile.TemporaryDirectory(prefix="robomme-fs-b1-sim-", dir="/tmp")
    exchange = Path(simulation.name)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(simulator_cuda_device),
            "MPLCONFIGDIR": "/tmp/robomme-fs-b1-mpl-cache",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(path) for path in simulator_paths),
        }
    )
    Path(environment["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    worker = subprocess.Popen(
        [
            str(simulator_python),
            str(Path(__file__).with_name("framesamp_r0_sim_worker.py")),
            "--task",
            task,
            "--episode",
            str(episode),
            "--exchange",
            str(exchange),
        ],
        cwd=examples_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        _wait(exchange / "initial.json", worker, timeout=300.0)
        initial_manifest = json.loads((exchange / "initial.json").read_text(encoding="utf-8"))
        initial_path = exchange / "initial.npz"
        if _sha_file(initial_path) != initial_manifest.get("initial_npz_sha256"):
            raise RuntimeError("B1 simulator initial payload SHA mismatch")
        simulator_runtime = initial_manifest.get("simulator_runtime")
        if (
            not isinstance(simulator_runtime, dict)
            or simulator_runtime.get("gpu_name") != H100_NAME
            or simulator_runtime.get("torch_cuda_version") != "12.8"
        ):
            raise RuntimeError("B1 simulator did not use the exact H100 runtime")
        with np.load(initial_path, allow_pickle=False) as payload:
            images = np.asarray(payload["images"], dtype=np.uint8)
            wrists = np.asarray(payload["wrist_images"], dtype=np.uint8)
            states = np.asarray(payload["states"], dtype=np.float32)
        if not images.size or not (len(images) == len(wrists) == len(states)):
            raise RuntimeError("B1 initial demonstration is empty or unaligned")
        policy.reset()
        policy.add_buffer({"images": images[:, None], "state": states, "exec_start_idx": len(images) - 1})
        prompt = str(initial_manifest["task_goal"])
        initial_raw = _raw_observation(images[-1], wrists[-1], states[-1], prompt)
        initial_cut, initial_actions = _cut_record(policy, initial_raw)
        _write_actions(exchange, initial_actions)

        _wait(exchange / "later.json", worker, timeout=600.0)
        stdout, stderr = worker.communicate(timeout=30.0)
        if worker.returncode != 0:
            raise RuntimeError(
                f"B1 simulator exited {worker.returncode}: stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
            )
        later_manifest = json.loads((exchange / "later.json").read_text(encoding="utf-8"))
        later_path = exchange / "later.npz"
        if (
            later_manifest.get("status") != "later_cut_ready"
            or later_manifest.get("executed_actions") != EXECUTION_HORIZON
            or _sha_file(later_path) != later_manifest.get("later_npz_sha256")
        ):
            raise RuntimeError("B1 simulator later-cut receipt mismatch")
        with np.load(later_path, allow_pickle=False) as payload:
            execution_images = np.asarray(payload["execution_images"], dtype=np.uint8)
            execution_states = np.asarray(payload["execution_states"], dtype=np.float32)
            later_raw = _raw_observation(
                payload["current_image"], payload["current_wrist"], payload["current_state"], prompt
            )
        policy.add_buffer({"images": execution_images[:, None], "state": execution_states, "exec_start_idx": 0})
        later_cut, _ = _cut_record(policy, later_raw)
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, 15)
            try:
                worker.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, 9)
                worker.wait(timeout=10.0)
        simulation.cleanup()

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scope": SCOPE,
        "status": "HARD_GREEN",
        "task": task,
        "episode": episode,
        "representation_policy": B1_PARTITION_KIND,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "overlay_source_tree_sha256": overlay_manifest["source_tree_sha256"],
        "patched_policy_sha256": PATCHED_POLICY_SHA256,
        "patched_mem_buffer_sha256": PATCHED_MEM_BUFFER_SHA256,
        "released_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
        "device": {"platform": "gpu", "device_kind": H100_NAME, "count": 1},
        "simulator_runtime": simulator_runtime,
        "cuts": [initial_cut, later_cut],
        "executed_actions": EXECUTION_HORIZON,
        "elapsed_seconds": time.monotonic() - started,
        "scored_evidence": False,
        "cloud_publication": False,
    }
    receipt = dict(unsigned)
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    validate_receipt(receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--simulator-cuda-device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("B1 transition output must be a fresh file in an existing directory")
    receipt = run_transition(
        task=args.task,
        episode=args.episode,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        runtime_root=args.runtime_root,
        simulator_cuda_device=args.simulator_cuda_device,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(_canonical(receipt))
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
