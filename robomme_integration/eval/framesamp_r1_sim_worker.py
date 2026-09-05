"""Pinned-runtime multi-replan RoboMME simulator worker for FS-R1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

EXECUTION_HORIZON = 16
ACTION_HORIZON = 20
ACTION_DIM = 8


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


def _load_action_request(exchange: Path, replan: int, expected_cut: int) -> np.ndarray:
    stem = f"request-{replan:03d}"
    manifest_path = exchange / f"{stem}.json"
    actions_path = exchange / f"{stem}.npy"
    _wait_for(manifest_path, timeout_seconds=900.0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"actions_npy_sha256", "causal_cut_step", "shape"}:
        raise ValueError(f"FS-R1 action request {replan} fields mismatch")
    if manifest["causal_cut_step"] != expected_cut:
        raise ValueError(
            f"FS-R1 request {replan} causal cut mismatch: {manifest['causal_cut_step']} != {expected_cut}"
        )
    if not actions_path.is_file() or _sha256_file(actions_path) != manifest["actions_npy_sha256"]:
        raise ValueError(f"FS-R1 action request {replan} SHA mismatch")
    actions = np.load(actions_path, allow_pickle=False)
    if (
        actions.shape != (ACTION_HORIZON, ACTION_DIM)
        or manifest["shape"] != [ACTION_HORIZON, ACTION_DIM]
        or not np.isfinite(actions).all()
    ):
        raise ValueError(f"FS-R1 action request {replan} must be finite shape (20,8)")
    return np.asarray(actions, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--max-replans", type=int, default=82)
    parser.add_argument("--stop-after-replans", type=int)
    args = parser.parse_args(argv)
    if (
        args.episode < 0
        or args.max_replans <= 0
        or (args.stop_after_replans is not None and not 1 <= args.stop_after_replans <= args.max_replans)
    ):
        raise ValueError("episode must be nonnegative and max-replans positive")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.isdigit():
        raise RuntimeError("FS-R1 simulator worker requires exactly one numeric CUDA device")
    if not args.exchange.is_dir() or any(args.exchange.iterdir()):
        raise ValueError("FS-R1 exchange directory must exist and be empty")

    import torch
    from env_runner import EnvRunner

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("FS-R1 simulator worker requires exactly one visible CUDA GPU")
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
        if not images.size or not (len(images) == len(wrists) == len(states)):
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

        executed_total = 0
        for replan in range(args.max_replans):
            expected_cut = len(images) - 1 + executed_total
            actions = _load_action_request(args.exchange, replan, expected_cut)
            execution_images: list[np.ndarray] = []
            execution_states: list[np.ndarray] = []
            terminal = False
            outcome = "unknown"
            current_image = current_wrist = current_state = None
            for action in actions[:EXECUTION_HORIZON]:
                (current_image, current_wrist, current_state), terminal, outcome = runner.step(action)
                if outcome == "error" or current_image is None:
                    raise RuntimeError(f"simulator error during FS-R1 replan {replan}")
                execution_images.append(np.array(current_image, copy=True))
                execution_states.append(np.array(current_state, copy=True))
                if terminal:
                    break
            executed = len(execution_images)
            if not executed:
                raise RuntimeError(f"FS-R1 replan {replan} executed zero actions")
            executed_total += executed
            response_payload = args.exchange / f"response-{replan:03d}.npz"
            _atomic_npz(
                response_payload,
                current_image=np.asarray(current_image, dtype=np.uint8),
                current_wrist=np.asarray(current_wrist, dtype=np.uint8),
                current_state=np.asarray(current_state, dtype=np.float32),
                execution_images=np.stack(execution_images).astype(np.uint8),
                execution_states=np.stack(execution_states).astype(np.float32),
            )
            _atomic_json(
                args.exchange / f"response-{replan:03d}.json",
                {
                    "causal_cut_step": expected_cut,
                    "executed_actions": executed,
                    "executed_actions_total": executed_total,
                    "outcome": outcome,
                    "response_npz_sha256": _sha256_file(response_payload),
                    "status": "episode_terminal" if terminal else "replan_ready",
                    "terminal": bool(terminal),
                },
            )
            if terminal:
                return 0
            if args.stop_after_replans is not None and replan + 1 == args.stop_after_replans:
                return 0
        raise RuntimeError("FS-R1 simulator exhausted max replans before terminal outcome")
    finally:
        runner.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
