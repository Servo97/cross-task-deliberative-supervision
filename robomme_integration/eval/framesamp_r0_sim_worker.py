"""GPU-rendered one-chunk RoboMME simulator worker for the FS-R0 gate.

The released policy runs in a different process/GPU.  This worker uses the
pinned RoboMME v0.4 runtime (including its CUDA-12.8 Torch build) for CPU
physics and a single isolated GPU for SAPIEN rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

EXECUTION_HORIZON = 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _wait_for(path: Path, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.isdigit():
        raise RuntimeError("FS-R0 simulator worker requires exactly one numeric CUDA device")
    if not args.exchange.is_dir() or any(args.exchange.iterdir()):
        raise ValueError("FS-R0 exchange directory must exist and be empty")

    import torch
    from env_runner import EnvRunner

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("FS-R0 simulator worker requires exactly one visible CUDA GPU")
    runtime_identity = {
        "cuda_visible_devices": visible_device,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
    }

    runner = EnvRunner(args.task, "", max_steps=1_300)
    try:
        seed, difficulty = runner.env_builder.resolve_episode(args.episode)
        runner.make_env(args.episode)
        initial = runner.get_init_obs()
        images = np.stack(initial["images"]).astype(np.uint8)
        wrists = np.stack(initial["wrist_images"]).astype(np.uint8)
        states = np.stack(initial["states"]).astype(np.float32)
        if not images.size or images.shape[0] != wrists.shape[0] or images.shape[0] != states.shape[0]:
            raise RuntimeError("simulator initial demonstration is empty or unaligned")
        initial_payload = args.exchange / "initial.npz"
        _atomic_npz(initial_payload, images=images, wrist_images=wrists, states=states)
        _atomic_json(
            args.exchange / "initial.json",
            {
                "difficulty": difficulty,
                "episode": args.episode,
                "initial_npz_sha256": _sha256_file(initial_payload),
                "seed": int(seed),
                "simulator_runtime": runtime_identity,
                "task": args.task,
                "task_goal": initial["task_goal"],
            },
        )

        actions_path = args.exchange / "actions.npy"
        actions_manifest_path = args.exchange / "actions.json"
        _wait_for(actions_manifest_path, timeout_seconds=args.timeout_seconds)
        manifest = json.loads(actions_manifest_path.read_text(encoding="utf-8"))
        if set(manifest) != {"actions_npy_sha256", "shape"}:
            raise ValueError("FS-R0 action request fields mismatch")
        if _sha256_file(actions_path) != manifest["actions_npy_sha256"]:
            raise ValueError("FS-R0 action request SHA mismatch")
        actions = np.load(actions_path, allow_pickle=False)
        if actions.shape != (20, 8) or manifest["shape"] != [20, 8] or not np.isfinite(actions).all():
            raise ValueError("FS-R0 action request must be finite shape (20,8)")

        execution_images = []
        execution_states = []
        current_image = current_wrist = current_state = None
        for index, action in enumerate(actions[:EXECUTION_HORIZON]):
            (current_image, current_wrist, current_state), stop, outcome = runner.step(action)
            if stop:
                raise RuntimeError(f"episode stopped before later cut at action {index}: {outcome}")
            execution_images.append(np.array(current_image, copy=True))
            execution_states.append(np.array(current_state, copy=True))
        assert current_image is not None and current_wrist is not None and current_state is not None
        later_payload = args.exchange / "later.npz"
        _atomic_npz(
            later_payload,
            current_image=np.asarray(current_image, dtype=np.uint8),
            current_wrist=np.asarray(current_wrist, dtype=np.uint8),
            current_state=np.asarray(current_state, dtype=np.float32),
            execution_images=np.stack(execution_images).astype(np.uint8),
            execution_states=np.stack(execution_states).astype(np.float32),
        )
        _atomic_json(
            args.exchange / "later.json",
            {
                "executed_actions": EXECUTION_HORIZON,
                "later_npz_sha256": _sha256_file(later_payload),
                "status": "later_cut_ready",
            },
        )
        return 0
    finally:
        runner.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
