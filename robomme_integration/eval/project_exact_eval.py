#!/usr/bin/env python3
"""Evaluate project S0/Q0/A6 under RoboMME's pinned paper rollout contract.

The environment loop intentionally reuses constants and scoring from
``official_reference_eval.py`` while retaining a distinct evaluator hash, estimand, and result
directory.  The active released-checkpoint campaign is never imported as a server and its source
file is not modified.  This evaluator adds only explicit task/episode identity to ``reset`` so the
project server can reproduce the same per-arm diffusion seed without guessing from prompts or
filesystem progress.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
from openpi_client import msgpack_numpy, websocket_client_policy

from robomme_integration.eval import official_reference_eval as reference
from robomme_integration.eval.project_exact_server import (
    ACTION_HORIZON,
    EXECUTION_HORIZON,
    PROTOCOL_ID,
    Q2_BLOCKER,
    validate_project_arm,
    validate_server_metadata,
)
from robomme_integration.eval.project_exact_source_audit import (
    BENCHMARK_SOURCE_COMMIT,
    MANISKILL_SOURCE_COMMIT,
    POLICY_SOURCE_COMMIT,
    REFERENCE_EVALUATOR_SHA256,
    audit_imported_git_source,
    require_file_in_source,
    require_pinned_commit,
    require_pinned_file,
)


def _audit_imported_runtime_sources(
    *,
    policy_source_commit: str,
    benchmark_source_commit: str,
    maniskill_source_commit: str,
) -> dict[str, object]:
    """Bind manifest commit labels to the modules this evaluator actually imported."""
    require_pinned_commit("policy source", policy_source_commit, POLICY_SOURCE_COMMIT)
    require_pinned_commit("benchmark source", benchmark_source_commit, BENCHMARK_SOURCE_COMMIT)
    require_pinned_commit("ManiSkill source", maniskill_source_commit, MANISKILL_SOURCE_COMMIT)
    require_pinned_file(
        "official_reference_eval.py",
        reference.__file__,
        REFERENCE_EVALUATOR_SHA256,
    )

    policy = audit_imported_git_source(
        "RoboMME policy/evaluator",
        inspect.getfile(reference.EnvRunner),
        POLICY_SOURCE_COMMIT,
    )
    # The wire client is another imported component of the policy repository.  Verify it resolves
    # into the same audited worktree instead of a coincidentally compatible site-package copy.
    require_file_in_source(policy, "openpi_client.websocket_client_policy", websocket_client_policy.__file__)

    benchmark_module = importlib.import_module("robomme.env_record_wrapper")
    benchmark = audit_imported_git_source(
        "RoboMME benchmark",
        inspect.getfile(benchmark_module),
        BENCHMARK_SOURCE_COMMIT,
    )
    maniskill_module = importlib.import_module("mani_skill")
    maniskill = audit_imported_git_source(
        "ManiSkill",
        inspect.getfile(maniskill_module),
        MANISKILL_SOURCE_COMMIT,
    )
    roots = {policy.root, benchmark.root, maniskill.root}
    if len(roots) != 3:
        raise RuntimeError("policy, benchmark, and ManiSkill must resolve from three distinct Git trees")
    return {
        "policy": policy.manifest_record(),
        "benchmark": benchmark.manifest_record(),
        "maniskill": maniskill.manifest_record(),
        "reference_evaluator_sha256": REFERENCE_EVALUATOR_SHA256,
    }


class ProjectExactClient(websocket_client_policy.MMEVLAWebsocketClientPolicy):
    def reset_episode(self, payload: dict) -> dict:
        request = dict(payload)
        request["reset"] = True
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in project exact server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


def _episode(
    *,
    task: str,
    episode_id: int,
    host: str,
    port: int,
    server_identity: dict,
) -> bool:
    runner = reference.EnvRunner(task, "", max_steps=reference.MAX_STEPS)
    runner.make_env(episode_id)
    client: ProjectExactClient | None = None
    try:
        client = ProjectExactClient(host, port)
        validate_server_metadata(client.get_server_metadata(), server_identity)
        reset_payload = {
            **server_identity,
            "task_name": task,
            "episode_idx": episode_id,
        }
        reference._wait_flag(client.reset_episode(reset_payload), "reset_finished")
        initial = runner.get_init_obs()
        images = list(initial["images"])
        states = list(initial["states"])
        if not images or len(images) != len(states):
            raise RuntimeError("initial demonstration/history is empty or image/state unaligned")
        current_image = images[-1]
        current_wrist = initial["wrist_images"][-1]
        current_state = states[-1]
        action_plan: collections.deque = collections.deque()
        count = 0

        while True:
            if not action_plan:
                result = client.infer(
                    {
                        "observation/image": current_image,
                        "observation/wrist_image": current_wrist,
                        "observation/state": current_state,
                        "prompt": initial["task_goal"],
                    }
                )
                actions = np.asarray(result.get("actions"))
                if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
                    raise RuntimeError(
                        f"policy action contract violation: expected finite (20, 8), got {actions.shape}"
                    )
                action_plan.extend(actions[:EXECUTION_HORIZON])

            (current_image, current_wrist, current_state), stop, outcome = runner.step(action_plan.popleft())
            count += 1
            if count > reference.MAX_STEPS:
                outcome, stop = "timeout", True
            if stop:
                if outcome in {"unknown", "error"}:
                    raise RuntimeError(f"simulator terminated with non-scoring outcome {outcome!r}")
                return outcome == "success"
    finally:
        if client is not None:
            client.close()
        runner.close_env()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--arm", required=True, choices=("s0", "q0", "a6", "q2", "v4_s0"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--project-source-sha256", required=True)
    parser.add_argument("--openpi-source-sha256", required=True)
    parser.add_argument("--server-source-sha256", required=True)
    parser.add_argument("--policy-source-commit", required=True)
    parser.add_argument("--benchmark-source-commit", required=True)
    parser.add_argument("--maniskill-source-commit", required=True)
    args = parser.parse_args()
    try:
        arm = validate_project_arm(args.arm)
    except ValueError as error:
        parser.error(str(error))
    if args.arm == "q2":  # defensive: validate_project_arm already rejects this path.
        parser.error(Q2_BLOCKER)

    # This gate runs before output directory creation or progress loading.  Commit-shaped CLI text
    # alone is not provenance: every module must resolve from the expected clean worktree.
    runtime_source_audit = _audit_imported_runtime_sources(
        policy_source_commit=args.policy_source_commit,
        benchmark_source_commit=args.benchmark_source_commit,
        maniskill_source_commit=args.maniskill_source_commit,
    )

    evaluator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reference_sha256 = hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()
    server_identity = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "checkpoint_sha256": args.checkpoint_sha256,
        "project_source_sha256": args.project_source_sha256,
        "openpi_source_sha256": args.openpi_source_sha256,
        "server_source_sha256": args.server_source_sha256,
        "model_seed": 7,
        "action_horizon": ACTION_HORIZON,
        "execution_horizon": EXECUTION_HORIZON,
        "history_mode": "forbidden_execution_only",
        "q2_supported": False,
    }
    manifest = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "estimand": "project_multitask_checkpoint_single_seed_exact_paper_protocol",
        "method": args.method,
        "arm": arm,
        "checkpoint_sha256": args.checkpoint_sha256,
        "project_source_sha256": args.project_source_sha256,
        "openpi_source_sha256": args.openpi_source_sha256,
        "server_source_sha256": args.server_source_sha256,
        "policy_source_commit": args.policy_source_commit,
        "benchmark_source_commit": args.benchmark_source_commit,
        "maniskill_source_commit": args.maniskill_source_commit,
        "runtime_source_audit": runtime_source_audit,
        "evaluator_sha256": evaluator_sha256,
        "reference_evaluator_sha256": reference_sha256,
        "model_seed": 7,
        "use_history": False,
        "action_horizon": ACTION_HORIZON,
        "execution_horizon": EXECUTION_HORIZON,
        "max_steps": reference.MAX_STEPS,
        "episodes_per_task": reference.EPISODES_PER_TASK,
        "tasks": list(reference.TASK_NAME_LIST),
    }
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError("refusing to mix progress from a different project-exact contract")
    else:
        reference._atomic_json(manifest_path, manifest)

    progress_path = args.output / "progress.json"
    results = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    for task in reference.TASK_NAME_LIST:
        task_results = results.setdefault(task, {})
        for episode_id in range(reference.EPISODES_PER_TASK):
            key = str(episode_id)
            if isinstance(task_results.get(key), bool):
                continue
            print(
                f"PROJECT_EXACT_EPISODE method={args.method} arm={arm} task={task} episode={episode_id}",
                flush=True,
            )
            task_results[key] = _episode(
                task=task,
                episode_id=episode_id,
                host=args.host,
                port=args.port,
                server_identity=server_identity,
            )
            reference._atomic_json(progress_path, results)
    score = reference._score(results)
    score.update(
        protocol_id=PROTOCOL_ID,
        method=args.method,
        arm=arm,
        model_seed=7,
        action_horizon=ACTION_HORIZON,
        execution_horizon=EXECUTION_HORIZON,
        checkpoint_sha256=args.checkpoint_sha256,
        project_source_sha256=args.project_source_sha256,
        openpi_source_sha256=args.openpi_source_sha256,
        server_source_sha256=args.server_source_sha256,
        policy_source_commit=args.policy_source_commit,
        benchmark_source_commit=args.benchmark_source_commit,
        maniskill_source_commit=args.maniskill_source_commit,
        runtime_source_audit=runtime_source_audit,
        evaluator_sha256=evaluator_sha256,
        reference_evaluator_sha256=reference_sha256,
        estimand="project_multitask_checkpoint_single_seed_exact_paper_protocol",
    )
    reference._atomic_json(args.output / "scorecard.json", score)
    print(json.dumps(score, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
