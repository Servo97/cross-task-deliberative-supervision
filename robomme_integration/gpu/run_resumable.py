#!/usr/bin/env python3
"""Restore, train, and continuously publish bounded RoboMME GPU checkpoints."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .checkpoint_transport import S3Transport


def checkpoint_dir(base: Path, arm: str, experiment: str) -> Path:
    return base / f"pi05_robomme_{arm}" / experiment


def terminate_process_group(leader_pid: int, timeout_seconds: float = 15.0) -> None:
    try:
        os.killpg(leader_pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(leader_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(leader_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    try:
        from ..training.arms import ARM_IDS
    except ImportError:  # SageMaker stages robomme_integration/ contents as top-level packages.
        from training.arms import ARM_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARM_IDS)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--checkpoint-base", type=Path, required=True)
    parser.add_argument("--s3-run-root", required=True)
    parser.add_argument("--final-step", type=int, default=19999)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--watcher-stop-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.watcher_stop_timeout_seconds < 60:
        raise SystemExit("--watcher-stop-timeout-seconds must be at least 60")

    experiment = os.environ.get("WSM_EXP_NAME", f"pi05_robomme_{args.arm}")
    local_dir = checkpoint_dir(args.checkpoint_base, args.arm, experiment)
    restored = S3Transport(args.s3_run_root).restore_latest(local_dir)
    print(
        f"[gpu-supervisor] restored_step={restored} local_dir={local_dir} s3_run_root={args.s3_run_root}",
        flush=True,
    )

    environment = os.environ.copy()
    environment["WSM_CKPT_BASE"] = str(args.checkpoint_base)
    environment["WSM_EXP_NAME"] = experiment
    environment["WSM_RESUME"] = "1" if restored is not None else "0"
    environment["WSM_FINAL_ONLY_CHECKPOINTS"] = "0"
    environment.setdefault("WSM_SAVE_INTERVAL", "5000")
    state_path = args.checkpoint_base / f".{experiment}.s3-upload-state.json"
    watcher_command = [
        str(args.train_python),
        "-m",
        "gpu.checkpoint_transport",
        "--local-root",
        str(local_dir),
        "--s3-run-root",
        args.s3_run_root,
        "--state",
        str(state_path),
        "--final-step",
        str(args.final_step),
        "--poll-seconds",
        str(args.poll_seconds),
        "--milestones",
        os.environ.get("ROBOMME_CHECKPOINT_MILESTONES", "10000"),
    ]
    success_milestones = os.environ.get("ROBOMME_SUCCESS_CHECKPOINT_MILESTONES", "").strip()
    if success_milestones:
        watcher_command.extend(["--success-milestones", success_milestones])
    watcher: subprocess.Popen | None = None
    trainer: subprocess.Popen | None = None
    received_signal: int | None = None

    def forward(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        if trainer is not None and trainer.poll() is None:
            try:
                os.killpg(trainer.pid, signum)
            except ProcessLookupError:
                pass

    # Install the handler before either private child group exists.  Without this ordering, a
    # platform termination in the narrow watcher/trainer startup window could kill this supervisor
    # and leave a private child group alive after the outer campaign considered the cell drained.
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    watcher = subprocess.Popen(watcher_command, env=environment, start_new_session=True)
    if received_signal is not None:
        watcher.send_signal(signal.SIGTERM)
        watcher.wait(timeout=args.watcher_stop_timeout_seconds)
        return 128 + received_signal
    try:
        trainer = subprocess.Popen(
            [str(args.train_python), "-m", "training.train", "--arm", args.arm],
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        watcher.terminate()
        watcher.wait(timeout=60)
        raise
    if received_signal is not None and trainer.poll() is None:
        try:
            os.killpg(trainer.pid, received_signal)
        except ProcessLookupError:
            pass
    train_returncode = 1
    try:
        while trainer.poll() is None:
            if watcher.poll() is not None:
                terminate_process_group(trainer.pid)
                raise RuntimeError(f"checkpoint watcher exited early with {watcher.returncode}")
            time.sleep(5)
        train_returncode = int(trainer.returncode or 0)
    finally:
        terminate_process_group(trainer.pid)
        if watcher.poll() is None:
            watcher.send_signal(signal.SIGTERM)
            try:
                watcher.wait(timeout=args.watcher_stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                terminate_process_group(watcher.pid)
                watcher.wait()

    # OpenPI drains its async checkpoint manager before exiting.  Make that final tree remotely
    # authoritative synchronously; only a successful trainer is allowed to prune recovery history.
    final_command = [*watcher_command, "--once"]
    final_sync = subprocess.run(final_command, env=environment, check=False)
    if final_sync.returncode and train_returncode == 0:
        return final_sync.returncode
    if train_returncode == 0:
        finalized = subprocess.run([*watcher_command, "--finalize-success"], env=environment, check=False)
        if finalized.returncode:
            return finalized.returncode
    if received_signal is not None and train_returncode == 0:
        return 128 + received_signal
    return train_returncode


if __name__ == "__main__":
    raise SystemExit(main())
