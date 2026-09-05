#!/usr/bin/env python3
"""Build auditable RoboMME per-task, suite, and challenge scorecards.

The official challenge evaluates one model on 50 episodes for each of 16 tasks.  A collection of
single-task specialists is useful mechanistic evidence, but is deliberately never labelled as a
challenge-comparable result here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path

from robomme_integration.training.single_task import TASK_ORDER

SUITES = {
    "counting_temporal": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "permanence_spatial": (
        "VideoUnmask",
        "ButtonUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmaskSwap",
    ),
    "reference_object": (
        "PickHighlight",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
    ),
    "imitation_procedural": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
MATERIALIZED_BENCHMARKS = {
    "RoboMMEOfficialHistoryBenchmark_counting": "counting_temporal",
    "RoboMMEOfficialHistoryBenchmark_permanence": "permanence_spatial",
    "RoboMMEOfficialHistoryBenchmark_reference": "reference_object",
    "RoboMMEOfficialHistoryBenchmark_imitation": "imitation_procedural",
}
CHALLENGE_TASK_ORDER = tuple(task for tasks in SUITES.values() for task in tasks)
if set(CHALLENGE_TASK_ORDER) != set(TASK_ORDER) or len(CHALLENGE_TASK_ORDER) != 16:
    raise AssertionError("RoboMME challenge suite registry does not exactly cover the 16 tasks")


def _wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    if episodes == 0:
        return [0.0, 0.0]
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _score(successes: int, episodes: int) -> dict:
    rate = successes / episodes if episodes else 0.0
    return {
        "successes": successes,
        "episodes": episodes,
        "rate": rate,
        "percent": round(100.0 * rate, 12),
        "wilson95": _wilson(successes, episodes),
    }


def _load_claim(value: str | Path | dict) -> dict:
    claim = value if isinstance(value, dict) else json.loads(Path(value).read_text())
    if claim.get("kind") != "robomme_fixed50_complete":
        raise ValueError(f"unsupported result claim kind {claim.get('kind')!r}")
    task = claim.get("task")
    if task not in CHALLENGE_TASK_ORDER:
        raise ValueError(f"unknown RoboMME task in result claim: {task!r}")
    for field in ("run_id", "eval_id", "arm", "checkpoint_uri"):
        if not isinstance(claim.get(field), str) or not claim[field]:
            raise ValueError(f"RoboMME result claim has no valid {field}")
    episodes, successes = claim.get("episodes"), claim.get("successes")
    if not isinstance(episodes, int) or not isinstance(successes, int):
        raise ValueError("episodes and successes must be integers")
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValueError(f"invalid score {successes}/{episodes} for {task}")
    return dict(claim)


def aggregate_claims(claims: Iterable[str | Path | dict], *, method_id: str, training_scope: str) -> dict:
    if training_scope not in {"single_task", "multitask"}:
        raise ValueError("training_scope must be single_task or multitask")
    loaded = [_load_claim(claim) for claim in claims]
    by_task: dict[str, dict] = {}
    for claim in loaded:
        task = claim["task"]
        if task in by_task:
            raise ValueError(f"duplicate result claim for {task}")
        by_task[task] = claim
    arms = {claim["arm"] for claim in loaded}
    if len(arms) > 1:
        raise ValueError(f"one scorecard cannot mix arms: {sorted(arms)}")

    task_scores = {
        task: _score(by_task[task]["successes"], by_task[task]["episodes"])
        for task in CHALLENGE_TASK_ORDER
        if task in by_task
    }
    suite_scores = {}
    for suite, tasks in SUITES.items():
        present = [task for task in tasks if task in task_scores]
        successes = sum(task_scores[task]["successes"] for task in present)
        episodes = sum(task_scores[task]["episodes"] for task in present)
        suite_scores[suite] = {
            **_score(successes, episodes),
            "tasks_present": present,
            "complete": len(present) == 4 and all(task_scores[t]["episodes"] == 50 for t in tasks),
        }

    failures = []
    missing = sorted(set(CHALLENGE_TASK_ORDER) - set(by_task))
    if missing:
        failures.append(f"missing tasks: {','.join(missing)}")
    wrong_counts = {task: score["episodes"] for task, score in task_scores.items() if score["episodes"] != 50}
    if wrong_counts:
        failures.append(f"non-50 episode tasks: {wrong_counts}")
    run_ids = sorted({claim["run_id"] for claim in loaded})
    checkpoints = sorted({claim["checkpoint_uri"] for claim in loaded})
    if training_scope != "multitask":
        failures.append("single-task specialists are diagnostic, not one challenge model")
    if len(run_ids) != 1:
        failures.append(f"expected one run_id, found {len(run_ids)}")
    if len(checkpoints) != 1:
        failures.append(f"expected one checkpoint, found {len(checkpoints)}")
    challenge_comparable = not failures

    total_successes = sum(score["successes"] for score in task_scores.values())
    total_episodes = sum(score["episodes"] for score in task_scores.values())
    return {
        "schema_version": 1,
        "kind": "robomme_challenge_scorecard",
        "method_id": method_id,
        "arm": next(iter(arms)) if arms else None,
        "training_scope": training_scope,
        "challenge_comparable": challenge_comparable,
        "qualification_failures": failures,
        "task_order": list(CHALLENGE_TASK_ORDER),
        "suites": {suite: list(tasks) for suite, tasks in SUITES.items()},
        "task_scores": task_scores,
        "suite_scores": suite_scores,
        "overall": _score(total_successes, total_episodes),
        "provenance": {
            "run_ids": run_ids,
            "checkpoint_uris": checkpoints,
            "eval_ids": sorted({claim["eval_id"] for claim in loaded}),
        },
    }


def aggregate_materialized(
    aggregates: Iterable[str | Path | dict],
    *,
    method_id: str,
    training_scope: str,
    run_id: str,
    arm: str,
    checkpoint_uri: str,
) -> dict:
    """Validate official vla-eval aggregates and convert them into one scorecard.

    This closes the reporting boundary between the four materialized benchmark JSON files emitted
    by the evaluator and the compact fixed-50 claims consumed by :func:`aggregate_claims`.
    """
    loaded = [value if isinstance(value, dict) else json.loads(Path(value).read_text()) for value in aggregates]
    if len(loaded) != len(MATERIALIZED_BENCHMARKS):
        raise ValueError(f"expected {len(MATERIALIZED_BENCHMARKS)} materialized suites, found {len(loaded)}")

    claims = []
    seen_benchmarks: set[str] = set()
    seen_tasks: set[str] = set()
    for aggregate in loaded:
        benchmark = aggregate.get("benchmark")
        if benchmark not in MATERIALIZED_BENCHMARKS:
            raise ValueError(f"unknown materialized RoboMME benchmark: {benchmark!r}")
        if benchmark in seen_benchmarks:
            raise ValueError(f"duplicate materialized RoboMME benchmark: {benchmark}")
        seen_benchmarks.add(benchmark)
        suite = MATERIALIZED_BENCHMARKS[benchmark]
        eval_id = aggregate.get("eval_id")
        expected_eval_id = f"{run_id}-{benchmark}"
        if eval_id != expected_eval_id:
            raise ValueError(f"aggregate eval_id {eval_id!r} != expected {expected_eval_id!r}")

        tasks = aggregate.get("tasks")
        if not isinstance(tasks, list) or {task.get("task") for task in tasks} != set(SUITES[suite]):
            raise ValueError(f"materialized task set does not match suite {suite}")
        suite_successes = 0
        suite_episodes = 0
        for task_result in tasks:
            task = task_result["task"]
            if task in seen_tasks:
                raise ValueError(f"duplicate materialized task: {task}")
            seen_tasks.add(task)
            episodes = task_result.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != 50:
                raise ValueError(f"materialized task {task} does not contain exactly 50 episodes")
            episode_indices = [episode.get("episode_idx") for episode in episodes]
            if sorted(episode_indices) != list(range(50)):
                raise ValueError(f"materialized task {task} does not contain episode_idx 0..49")
            outcomes = [episode.get("metrics", {}).get("success") for episode in episodes]
            if any(not isinstance(outcome, bool) for outcome in outcomes):
                raise ValueError(f"materialized task {task} has a non-boolean success outcome")
            successes = sum(outcomes)
            if task_result.get("num_episodes") != 50 or not math.isclose(
                task_result.get("mean_success", -1), successes / 50, abs_tol=1e-12
            ):
                raise ValueError(f"materialized task summary disagrees with episodes for {task}")
            suite_successes += successes
            suite_episodes += 50
            claims.append(
                {
                    "kind": "robomme_fixed50_complete",
                    "run_id": run_id,
                    "eval_id": eval_id,
                    "task": task,
                    "arm": arm,
                    "episodes": 50,
                    "successes": successes,
                    "checkpoint_uri": checkpoint_uri,
                }
            )
        if aggregate.get("num_episodes_total") != suite_episodes or not math.isclose(
            aggregate.get("mean_success", -1), suite_successes / suite_episodes, abs_tol=1e-12
        ):
            raise ValueError(f"materialized suite summary disagrees with episodes for {benchmark}")

    if seen_benchmarks != set(MATERIALIZED_BENCHMARKS) or seen_tasks != set(CHALLENGE_TASK_ORDER):
        raise ValueError("materialized aggregates do not exactly cover the RoboMME challenge")
    return aggregate_claims(claims, method_id=method_id, training_scope=training_scope)


def _flat_row(scorecard: dict) -> dict:
    row = {
        "method_id": scorecard["method_id"],
        "arm": scorecard.get("arm"),
        "training_scope": scorecard["training_scope"],
        "challenge_comparable": scorecard["challenge_comparable"],
    }
    for task in CHALLENGE_TASK_ORDER:
        score = scorecard["task_scores"].get(task)
        row[task] = "" if score is None else score["percent"]
    for suite in SUITES:
        score = scorecard["suite_scores"].get(suite)
        row[f"suite:{suite}"] = "" if not score or not score["complete"] else score["percent"]
    row["overall"] = scorecard["overall"]["percent"]
    row["episodes"] = scorecard["overall"]["episodes"]
    return row


def write_matrix(scorecards: Iterable[dict], output: Path) -> None:
    rows = [_flat_row(scorecard) for scorecard in scorecards]
    fields = [
        "method_id",
        "arm",
        "training_scope",
        "challenge_comparable",
        *CHALLENGE_TASK_ORDER,
        *(f"suite:{suite}" for suite in SUITES),
        "overall",
        "episodes",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--claim", action="append", required=True)
    aggregate.add_argument("--method-id", required=True)
    aggregate.add_argument("--training-scope", choices=("single_task", "multitask"), required=True)
    aggregate.add_argument("--output-json", type=Path, required=True)
    aggregate.add_argument("--output-csv", type=Path)
    aggregate.add_argument("--require-challenge-comparable", action="store_true")
    materialized = subparsers.add_parser("materialized")
    materialized.add_argument("--aggregate", action="append", required=True)
    materialized.add_argument("--method-id", required=True)
    materialized.add_argument("--training-scope", choices=("single_task", "multitask"), required=True)
    materialized.add_argument("--run-id", required=True)
    materialized.add_argument("--arm", required=True)
    materialized.add_argument("--checkpoint-uri", required=True)
    materialized.add_argument("--output-json", type=Path, required=True)
    materialized.add_argument("--output-csv", type=Path)
    materialized.add_argument("--require-challenge-comparable", action="store_true")
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--scorecard", action="append", required=True)
    matrix.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if args.command in {"aggregate", "materialized"}:
        if args.command == "aggregate":
            scorecard = aggregate_claims(args.claim, method_id=args.method_id, training_scope=args.training_scope)
        else:
            scorecard = aggregate_materialized(
                args.aggregate,
                method_id=args.method_id,
                training_scope=args.training_scope,
                run_id=args.run_id,
                arm=args.arm,
                checkpoint_uri=args.checkpoint_uri,
            )
        if args.require_challenge_comparable and not scorecard["challenge_comparable"]:
            raise SystemExit(
                "scorecard is not challenge-comparable: " + "; ".join(scorecard["qualification_failures"])
            )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
        if args.output_csv:
            write_matrix([scorecard], args.output_csv)
        return

    scorecards = [json.loads(Path(path).read_text()) for path in args.scorecard]
    if any(value.get("kind") != "robomme_challenge_scorecard" for value in scorecards):
        raise SystemExit("matrix input is not a RoboMME challenge scorecard")
    write_matrix(scorecards, args.output_csv)


if __name__ == "__main__":
    main()
