#!/usr/bin/env python3
"""Disk-lean reproduction of RoboMME's released-checkpoint evaluation loop.

The policy-facing protocol is paper-exact: reset once per episode, prefill the complete initial
history, append causal execution observations, predict 20 actions, and execute the first 16.  This
adapter deliberately omits MP4 rendering and every VLM subgoal dependency; neither affects the
plain π0.5 or perceptual FrameSamp+Modul policy inputs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# These imports intentionally resolve from the pinned official examples/robomme tree.  The runner
# sets CWD/PYTHONPATH so this repository's unrelated top-level ``utils`` package cannot shadow it.
from env_runner import EnvRunner
from openpi_client import websocket_client_policy

from utils import TASK_NAME_LIST, pack_buffer

OBS_HORIZON = 16
MAX_STEPS = 1_300
EPISODES_PER_TASK = 50
SUITES = {
    "counting_temporal": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "permanence_spatial": ("VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"),
    "reference_object": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "imitation_procedural": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def _wait_flag(response: dict, key: str) -> None:
    while not response.get(key, False):
        time.sleep(0.1)


def _episode(*, task: str, episode_id: int, host: str, port: int, use_history: bool) -> bool:
    runner = EnvRunner(task, "", max_steps=MAX_STEPS)
    runner.make_env(episode_id)
    try:
        client = websocket_client_policy.MMEVLAWebsocketClientPolicy(host, port)
        _wait_flag(client.reset(), "reset_finished")
        initial = runner.get_init_obs()
        images = list(initial["images"])
        states = list(initial["states"])
        if not images or len(images) != len(states):
            raise RuntimeError("initial demonstration/history is empty or image/state unaligned")
        current_image = images[-1]
        current_wrist = initial["wrist_images"][-1]
        current_state = states[-1]
        exec_start_idx = len(images) - 1
        action_plan: collections.deque = collections.deque()
        count = 0

        while True:
            if not action_plan:
                if use_history:
                    response = client.add_buffer(pack_buffer(images, states, exec_start_idx))
                    _wait_flag(response, "add_buffer_finished")
                result = client.infer(
                    {
                        "observation/image": current_image,
                        "observation/wrist_image": current_wrist,
                        "observation/state": current_state,
                        "prompt": initial["task_goal"],
                    }
                )
                actions = np.asarray(result.get("actions"))
                if actions.shape != (20, 8) or not np.isfinite(actions).all():
                    raise RuntimeError(
                        f"policy action contract violation: expected finite (20, 8), got {actions.shape}"
                    )
                action_plan.extend(actions[:OBS_HORIZON])
                images.clear()
                states.clear()
                exec_start_idx = 0

            (current_image, current_wrist, current_state), stop, outcome = runner.step(action_plan.popleft())
            count += 1
            if count > MAX_STEPS:
                outcome, stop = "timeout", True
            if stop:
                if outcome in {"unknown", "error"}:
                    raise RuntimeError(f"simulator terminated with non-scoring outcome {outcome!r}")
                return outcome == "success"
            images.append(current_image.copy())
            states.append(current_state.copy())
    finally:
        runner.close_env()


def _score(results: dict[str, dict[str, bool]]) -> dict:
    task_rates = {}
    for task in TASK_NAME_LIST:
        outcomes = results.get(task, {})
        if sorted(map(int, outcomes)) != list(range(EPISODES_PER_TASK)):
            raise RuntimeError(f"{task} does not contain fixed episode indices 0..49")
        values = [outcomes[str(index)] for index in range(EPISODES_PER_TASK)]
        if any(not isinstance(value, bool) for value in values):
            raise RuntimeError(f"{task} contains a non-boolean outcome")
        task_rates[task] = sum(values) / EPISODES_PER_TASK
    suite_rates = {suite: sum(task_rates[task] for task in tasks) / len(tasks) for suite, tasks in SUITES.items()}
    successes = sum(
        sum(bool(results[task][str(index)]) for index in range(EPISODES_PER_TASK)) for task in TASK_NAME_LIST
    )
    return {
        "schema_version": 1,
        "episodes": 800,
        "successes": successes,
        "result_scale": "fraction_0_1",
        "task_success_rate": task_rates,
        "suite_success_rate": suite_rates,
        "overall_success_rate": sum(task_rates.values()) / len(task_rates),
        "overall_wilson_95": _wilson(successes, 800),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--use-history", action="store_true")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--policy-source-commit", required=True)
    parser.add_argument("--benchmark-source-commit", required=True)
    parser.add_argument("--maniskill-source-commit", required=True)
    args = parser.parse_args()
    manifest = {
        "schema_version": 1,
        "estimand": "released_step79999_seed7_positive_control",
        "method": args.method,
        "checkpoint_sha256": args.checkpoint_sha256,
        "policy_source_commit": args.policy_source_commit,
        "benchmark_source_commit": args.benchmark_source_commit,
        "maniskill_source_commit": args.maniskill_source_commit,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_seed": 7,
        "use_history": args.use_history,
        "action_horizon": 20,
        "execution_horizon": 16,
        "max_steps": MAX_STEPS,
        "episodes_per_task": EPISODES_PER_TASK,
        "tasks": list(TASK_NAME_LIST),
    }
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError("refusing to mix progress from a different official-reference contract")
    else:
        _atomic_json(manifest_path, manifest)
    progress_path = args.output / "progress.json"
    results = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    for task in TASK_NAME_LIST:
        task_results = results.setdefault(task, {})
        for episode_id in range(EPISODES_PER_TASK):
            key = str(episode_id)
            if isinstance(task_results.get(key), bool):
                continue
            print(
                f"OFFICIAL_REFERENCE_EPISODE method={args.method} task={task} "
                f"episode={episode_id} history={int(args.use_history)}",
                flush=True,
            )
            task_results[key] = _episode(
                task=task,
                episode_id=episode_id,
                host=args.host,
                port=args.port,
                use_history=args.use_history,
            )
            _atomic_json(progress_path, results)
    score = _score(results)
    score.update(
        method=args.method,
        model_seed=7,
        action_horizon=20,
        execution_horizon=16,
        checkpoint_sha256=args.checkpoint_sha256,
        policy_source_commit=args.policy_source_commit,
        benchmark_source_commit=args.benchmark_source_commit,
        maniskill_source_commit=args.maniskill_source_commit,
        estimand="released_step79999_seed7_positive_control",
    )
    _atomic_json(args.output / "scorecard.json", score)
    print(json.dumps(score, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
