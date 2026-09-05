#!/usr/bin/env python3
"""Paired comparison for two complete exact-manifest RoboCasa evaluations.

The decisive protocol uses the same reset/noise manifest for every arm.  This tool refuses legacy,
partial, or differently manifested results, joins outcomes by the full immutable episode identity,
and reports task-macro baseline/candidate scores, paired deltas, relative lift, discordant episode
counts, and deterministic task-cluster bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Iterable

from vla_training.eval.eval_manifest import (
    episode_identity,
    load_episode_manifest,
    validate_evaluation_provenance,
)

DEFAULT_SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
SPLIT_LABELS = {"atomic_seen": "A", "composite_seen": "CS", "composite_unseen": "CUS"}
DECISIVE_SPLIT_COUNTS = {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16}
RELATIVE_LIFT_TARGET = 0.30
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return math.fsum(values) / len(values)


def _exact_int(value, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _rate(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0,1], got {value!r}")
    return value


def _assert_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field}={actual!r}; exact records imply {expected!r}")


def _identity_sha256(identities) -> str:
    raw = json.dumps(sorted(identities), separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_protocol_manifest(path: Path, expected_tasks: int, episodes_per_task: int) -> dict:
    manifest = load_episode_manifest(path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("split") != "target":
        raise ValueError(f"{path}: decisive comparison requires split=target")
    if set(manifest.get("task_sets", [])) != set(DEFAULT_SPLITS):
        raise ValueError(f"{path}: task_sets must be exactly A/CS/CUS")
    if manifest.get("episodes_per_task") != episodes_per_task:
        raise ValueError(f"{path}: expected episodes_per_task={episodes_per_task}")
    records, task_split, task_horizon = {}, {}, {}
    for index, episode in enumerate(manifest["episodes"]):
        task, split = episode["task"], episode["split_set"]
        if not isinstance(task, str) or not task or "/" in task or "\x00" in task:
            raise ValueError(f"{path}: unsafe task name at episode {index}: {task!r}")
        if split not in DEFAULT_SPLITS or episode["reset"].get("kind") != "heldout_demo":
            raise ValueError(f"{path}: episode {index} is not a decisive A/CS/CUS heldout reset")
        if task in task_split and task_split[task] != split:
            raise ValueError(f"{path}: task {task!r} appears in multiple splits")
        if task in task_horizon and task_horizon[task] != episode["horizon"]:
            raise ValueError(f"{path}: task {task!r} has inconsistent horizons")
        task_split[task], task_horizon[task] = split, int(episode["horizon"])
        records.setdefault(task, []).append(episode)
    if len(records) != expected_tasks:
        raise ValueError(f"{path}: expected {expected_tasks} tasks, found {len(records)}")
    bad = {task: len(rows) for task, rows in records.items() if len(rows) != episodes_per_task}
    if bad:
        raise ValueError(f"{path}: inexact per-task episode counts {bad}")
    split_tasks = {split: {task for task in records if task_split[task] == split} for split in DEFAULT_SPLITS}
    if any(not tasks for tasks in split_tasks.values()):
        raise ValueError(f"{path}: every A/CS/CUS split must be nonempty")
    if expected_tasks == 50 and episodes_per_task == 100:
        observed = {split: len(tasks) for split, tasks in split_tasks.items()}
        if observed != DECISIVE_SPLIT_COUNTS:
            raise ValueError(f"{path}: expected split counts {DECISIVE_SPLIT_COUNTS}, got {observed}")
    return {
        "manifest": manifest,
        "manifest_file_sha256": file_sha256,
        "records": records,
        "identities": {task: [episode_identity(row) for row in rows] for task, rows in records.items()},
        "task_split": task_split,
        "task_horizon": task_horizon,
        "split_tasks": split_tasks,
        "expected_tasks": expected_tasks,
        "episodes_per_task": episodes_per_task,
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must be in [0,1], got {q}")
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _ci(values: list[float], *, level: float) -> list[float]:
    values = sorted(values)
    alpha = (1.0 - level) / 2.0
    return [_percentile(values, alpha), _percentile(values, 1.0 - alpha)]


def task_cluster_ci(per_task_values: dict[str, float], *, samples: int, seed: int, level: float = 0.95) -> list[float]:
    """Percentile CI from resampling whole task clusters with replacement."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be in (0,1)")
    task_values = [float(per_task_values[task]) for task in sorted(per_task_values)]
    if not task_values:
        raise ValueError("task-cluster bootstrap needs at least one task")
    rng = random.Random(int(seed))
    n = len(task_values)
    draws = sorted(_mean(task_values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    alpha = (1.0 - level) / 2.0
    return [_percentile(draws, alpha), _percentile(draws, 1.0 - alpha)]


def _validate_results(path: Path, protocol: dict) -> dict:
    result = _load_json(path)
    if result.get("protocol") != "exact_manifest":
        raise ValueError(f"{path}: paired comparison requires protocol=exact_manifest")
    if result.get("complete") is not True:
        raise ValueError(f"{path}: evaluation is not complete")
    digest = result.get("manifest_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{path}: missing/invalid manifest_sha256")
    if digest != protocol["manifest"]["manifest_sha256"]:
        raise ValueError(f"{path}: result uses a different episode manifest")
    if result.get("split") != "target":
        raise ValueError(f"{path}: result split must be target")
    expected_tasks = protocol["expected_tasks"]
    episodes_per_task = protocol["episodes_per_task"]
    if _exact_int(result.get("num_trials"), f"{path}: num_trials", 1) != episodes_per_task:
        raise ValueError(f"{path}: num_trials must be {episodes_per_task}")
    if _exact_int(result.get("n_tasks_done"), f"{path}: n_tasks_done") != expected_tasks:
        raise ValueError(f"{path}: expected {expected_tasks} completed tasks")
    if _exact_int(result.get("n_episodes_done"), f"{path}: n_episodes_done") != expected_tasks * episodes_per_task:
        raise ValueError(f"{path}: incomplete episode count")
    if not isinstance(result.get("model"), str) or not result["model"]:
        raise ValueError(f"{path}: model must be a nonempty string")
    step = _exact_int(result.get("step"), f"{path}: step")
    provenance = validate_evaluation_provenance(
        result.get("evaluation_provenance"),
        episode_manifest_sha256=protocol["manifest"]["manifest_sha256"],
    )
    if provenance["episode_manifest_file_sha256"] != protocol["manifest_file_sha256"]:
        raise ValueError(f"{path}: provenance does not bind the supplied episode-manifest file")
    if step != provenance["checkpoint_step"]:
        raise ValueError(
            f"{path}: result step={step} differs from provenance checkpoint_step={provenance['checkpoint_step']}"
        )
    splits = result.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(DEFAULT_SPLITS):
        raise ValueError(f"{path}: split summaries must be exactly A/CS/CUS")
    all_rates = []
    for split_name in DEFAULT_SPLITS:
        split = splits[split_name]
        expected = protocol["split_tasks"][split_name]
        per_task, counts = split.get("per_task"), split.get("per_task_num_episodes")
        if not isinstance(per_task, dict) or set(per_task) != expected:
            raise ValueError(f"{path}: split {split_name!r} has an inexact task set")
        if not isinstance(counts, dict) or set(counts) != expected:
            raise ValueError(f"{path}: split {split_name!r} has an inexact episode-count map")
        if _exact_int(split.get("n_tasks_done"), f"{path}: {split_name}.n_tasks_done") != len(expected):
            raise ValueError(f"{path}: split {split_name!r} task completion mismatch")
        if _exact_int(split.get("n_tasks_expected"), f"{path}: {split_name}.n_tasks_expected") != len(expected):
            raise ValueError(f"{path}: split {split_name!r} expected-task mismatch")
        if (
            _exact_int(split.get("n_episodes_done"), f"{path}: {split_name}.n_episodes_done")
            != len(expected) * episodes_per_task
        ):
            raise ValueError(f"{path}: split {split_name!r} episode completion mismatch")
        rates = []
        for task in sorted(expected):
            rates.append(_rate(per_task[task], f"{path}: {split_name}/{task}"))
            if _exact_int(counts[task], f"{path}: {split_name}/{task}.episodes") != episodes_per_task:
                raise ValueError(f"{path}: {split_name}/{task} is incomplete")
        _assert_close(
            _rate(split.get("mean"), f"{path}: {split_name}.mean"), _mean(rates), f"{path}: {split_name}.mean"
        )
        all_rates.extend(rates)
    _assert_close(
        _rate(result.get("avg_task_weighted"), f"{path}: avg_task_weighted"),
        _mean(all_rates),
        f"{path}: avg_task_weighted",
    )
    return result


def _load_task_stats(
    root: Path,
    split: str,
    task: str,
    protocol: dict,
    summary_rate: float,
    evaluation_provenance: dict,
) -> dict:
    path = root / split / task / "stats.json"
    stats = _load_json(path)
    expected = {
        "task": task,
        "split_set": split,
        "split": "target",
        "manifest_sha256": protocol["manifest"]["manifest_sha256"],
        "horizon": protocol["task_horizon"][task],
        "seed": protocol["manifest"].get("base_seed"),
    }
    for field, value in expected.items():
        if stats.get(field) != value:
            raise ValueError(f"{path}: {field}={stats.get(field)!r}; expected {value!r}")
    stats_provenance = validate_evaluation_provenance(
        stats.get("evaluation_provenance"),
        episode_manifest_sha256=protocol["manifest"]["manifest_sha256"],
    )
    if stats_provenance != evaluation_provenance:
        raise ValueError(f"{path}: evaluation_provenance differs from this root's results.json")
    episode_count = protocol["episodes_per_task"]
    if _exact_int(stats.get("num_episodes"), f"{path}: num_episodes") != episode_count:
        raise ValueError(f"{path}: expected exactly {episode_count} episodes")
    _exact_int(stats.get("num_episode_shards"), f"{path}: num_episode_shards", 1)
    episodes = stats.get("per_episode")
    if not isinstance(episodes, list) or len(episodes) != episode_count:
        raise ValueError(f"{path}: exact per_episode must contain {episode_count} records")
    actual: dict[tuple[str, int, str, int], bool] = {}
    ordered_identities = []
    lengths = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or type(episode.get("success")) is not bool:
            raise ValueError(f"{path}: episode {index} has invalid success")
        length = _exact_int(episode.get("episode_length"), f"{path}: episode {index}.episode_length")
        try:
            identity = episode_identity(episode)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: episode {index} has invalid identity") from exc
        if identity in actual:
            raise ValueError(f"{path}: duplicate episode identity {identity}")
        ordered_identities.append(identity)
        actual[identity] = episode["success"]
        lengths.append(length)
    if ordered_identities != protocol["identities"][task]:
        raise ValueError(f"{path}: episode identities/order differ from canonical manifest")
    successes = [actual[identity] for identity in ordered_identities]
    if stats.get("successes") != successes or stats.get("episode_lengths") != lengths:
        raise ValueError(f"{path}: summary vectors do not match exact records")
    observed_rate = _mean(float(value) for value in actual.values())
    _assert_close(_rate(stats.get("success_rate"), f"{path}: success_rate"), observed_rate, f"{path}: success_rate")
    _assert_close(summary_rate, observed_rate, f"{path}: results.json per_task rate")
    return {
        "path": str(path),
        "outcomes": actual,
        "rate": observed_rate,
        "identity_sha256": _identity_sha256(ordered_identities),
    }


def _comparison_block(task_rows: dict[str, dict], *, samples: int, seed: int, label: str) -> dict:
    baseline = {task: row["baseline_rate"] for task, row in task_rows.items()}
    candidate = {task: row["candidate_rate"] for task, row in task_rows.items()}
    delta = {task: candidate[task] - baseline[task] for task in task_rows}
    baseline_mean = _mean(baseline.values())
    candidate_mean = _mean(candidate.values())
    delta_mean = _mean(delta.values())
    relative = delta_mean / baseline_mean if baseline_mean != 0.0 else None
    rng = random.Random(seed)
    tasks = sorted(task_rows)
    base_draws, cand_draws, delta_draws, relative_draws = [], [], [], []
    for _ in range(samples):
        sampled = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        base_draw = _mean(baseline[task] for task in sampled)
        cand_draw = _mean(candidate[task] for task in sampled)
        base_draws.append(base_draw)
        cand_draws.append(cand_draw)
        delta_draws.append(cand_draw - base_draw)
        if base_draw != 0.0:
            relative_draws.append((cand_draw - base_draw) / base_draw)
    relative_ci = _ci(relative_draws, level=0.95) if relative_draws else None
    all_relative_defined = len(relative_draws) == samples
    return {
        "label": label,
        "n_tasks": len(task_rows),
        "n_episodes": sum(row["n_episodes"] for row in task_rows.values()),
        "baseline_task_macro": baseline_mean,
        "candidate_task_macro": candidate_mean,
        "paired_delta_absolute": delta_mean,
        "relative_lift": relative,
        "relative_lift_target": RELATIVE_LIFT_TARGET,
        "relative_lift_target_met": relative >= RELATIVE_LIFT_TARGET if relative is not None else None,
        "relative_lift_target_margin": relative - RELATIVE_LIFT_TARGET if relative is not None else None,
        "baseline_task_cluster_ci95": _ci(base_draws, level=0.95),
        "candidate_task_cluster_ci95": _ci(cand_draws, level=0.95),
        "paired_delta_task_cluster_ci95": _ci(delta_draws, level=0.95),
        "relative_lift_task_cluster_ci95": relative_ci,
        "relative_lift_bootstrap_defined_samples": len(relative_draws),
        "relative_lift_ci95_supports_target": relative_ci[0] >= RELATIVE_LIFT_TARGET if all_relative_defined else None,
        "candidate_better_episodes": sum(row["candidate_only_success"] for row in task_rows.values()),
        "baseline_better_episodes": sum(row["baseline_only_success"] for row in task_rows.values()),
        "both_success_episodes": sum(row["both_success"] for row in task_rows.values()),
        "both_failure_episodes": sum(row["both_failure"] for row in task_rows.values()),
        "per_task": task_rows,
    }


def compare_exact_roots(
    baseline_root: str | os.PathLike,
    candidate_root: str | os.PathLike,
    *,
    episode_manifest: str | os.PathLike,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    expected_tasks: int = 50,
    expected_episodes_per_task: int = 100,
) -> dict:
    _exact_int(bootstrap_samples, "bootstrap_samples", 1)
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    baseline_root = Path(baseline_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    if baseline_root == candidate_root:
        raise ValueError("baseline and candidate result roots must be different")
    manifest_path = Path(episode_manifest).resolve()
    protocol = _load_protocol_manifest(manifest_path, expected_tasks, expected_episodes_per_task)
    baseline_results_path = baseline_root / "results.json"
    candidate_results_path = candidate_root / "results.json"
    baseline_results = _validate_results(baseline_results_path, protocol)
    candidate_results = _validate_results(candidate_results_path, protocol)
    baseline_provenance = baseline_results["evaluation_provenance"]
    candidate_provenance = candidate_results["evaluation_provenance"]
    if (
        baseline_provenance["eval_run_id"] == candidate_provenance["eval_run_id"]
        or baseline_provenance["eval_manifest_sha256"] == candidate_provenance["eval_manifest_sha256"]
    ):
        raise ValueError("baseline and candidate carry the same immutable evaluation identity")
    manifest_sha256 = protocol["manifest"]["manifest_sha256"]
    expected_paths = {
        (root / protocol["task_split"][task] / task / "stats.json").resolve()
        for root in (baseline_root, candidate_root)
        for task in protocol["records"]
    }
    actual_paths = {
        path.resolve()
        for root in (baseline_root, candidate_root)
        for path in root.glob("*/*/stats.json")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "evaluation roots contain an inexact/mixed canonical stats set: "
            f"missing={len(expected_paths - actual_paths)}, extra={len(actual_paths - expected_paths)}"
        )

    split_rows: dict[str, dict[str, dict]] = {}
    all_rows: dict[str, dict] = {}
    for split in DEFAULT_SPLITS:
        baseline_tasks = protocol["split_tasks"][split]
        rows = {}
        for task in sorted(baseline_tasks):
            base = _load_task_stats(
                baseline_root,
                split,
                task,
                protocol,
                _rate(baseline_results["splits"][split]["per_task"][task], f"baseline {split}/{task}"),
                baseline_provenance,
            )
            cand = _load_task_stats(
                candidate_root,
                split,
                task,
                protocol,
                _rate(candidate_results["splits"][split]["per_task"][task], f"candidate {split}/{task}"),
                candidate_provenance,
            )
            if set(base["outcomes"]) != set(cand["outcomes"]):
                raise ValueError(f"{split}/{task}: episode identities differ across arms")
            paired = [
                (bool(base["outcomes"][identity]), bool(cand["outcomes"][identity]))
                for identity in protocol["identities"][task]
            ]
            row = {
                "n_episodes": len(paired),
                "episode_identity_sha256": base["identity_sha256"],
                "baseline_rate": base["rate"],
                "candidate_rate": cand["rate"],
                "paired_delta": cand["rate"] - base["rate"],
                "candidate_only_success": sum((not b) and c for b, c in paired),
                "baseline_only_success": sum(b and (not c) for b, c in paired),
                "both_success": sum(b and c for b, c in paired),
                "both_failure": sum((not b) and (not c) for b, c in paired),
            }
            rows[task] = row
            if task in all_rows:
                raise ValueError(f"task {task!r} appears in multiple split sets")
            all_rows[task] = row
        split_rows[split] = rows

    output = {
        "schema_version": 2,
        "kind": "robocasa_exact_paired_comparison",
        "manifest_sha256": manifest_sha256,
        "manifest": {
            "path": str(manifest_path),
            "tasks": expected_tasks,
            "episodes_per_task": expected_episodes_per_task,
            "episodes": expected_tasks * expected_episodes_per_task,
            "split_task_counts": {split: len(protocol["split_tasks"][split]) for split in DEFAULT_SPLITS},
        },
        "baseline": {
            "root": str(baseline_root),
            "model": baseline_results.get("model"),
            "step": baseline_results.get("step"),
            "evaluation_provenance": baseline_provenance,
        },
        "candidate": {
            "root": str(candidate_root),
            "model": candidate_results.get("model"),
            "step": candidate_results.get("step"),
            "evaluation_provenance": candidate_provenance,
        },
        "bootstrap": {
            "unit": "paired_task_cluster",
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "confidence": 0.95,
            "interval": "percentile",
            "coupling": "same resampled task indices for baseline and candidate",
        },
        "overall": _comparison_block(all_rows, samples=bootstrap_samples, seed=bootstrap_seed + 1000, label="Overall"),
        "splits": {
            split: _comparison_block(
                rows,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 2000 + index * 100,
                label=SPLIT_LABELS[split],
            )
            for index, (split, rows) in enumerate(sorted(split_rows.items()))
        },
    }
    output["relative_lift_target"] = {
        "scope": "overall_task_macro",
        "threshold": RELATIVE_LIFT_TARGET,
        "point_estimate_met": output["overall"]["relative_lift_target_met"],
        "point_estimate_margin": output["overall"]["relative_lift_target_margin"],
        "ci95_lower_bound_supports_target": output["overall"]["relative_lift_ci95_supports_target"],
    }
    canonical = json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output["comparison_sha256"] = hashlib.sha256(canonical).hexdigest()
    return output


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results-dir", required=True)
    parser.add_argument("--candidate-results-dir", required=True)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    comparison = compare_exact_roots(
        args.baseline_results_dir,
        args.candidate_results_dir,
        episode_manifest=args.episode_manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_json_atomic(Path(args.out), comparison)
    overall = comparison["overall"]
    relative_text = "undefined" if overall["relative_lift"] is None else f"{100 * overall['relative_lift']:+.2f}%"
    print(
        f"baseline={100 * overall['baseline_task_macro']:.2f}% "
        f"candidate={100 * overall['candidate_task_macro']:.2f}% "
        f"delta={100 * overall['paired_delta_absolute']:+.2f}pp "
        f"relative={relative_text}"
    )
    print(
        "paired delta task-cluster 95% CI "
        f"[{100 * overall['paired_delta_task_cluster_ci95'][0]:+.2f}, "
        f"{100 * overall['paired_delta_task_cluster_ci95'][1]:+.2f}]pp"
    )
    target = comparison["relative_lift_target"]
    print(
        f"relative-lift target={100 * target['threshold']:.1f}% "
        f"point_met={target['point_estimate_met']} "
        f"ci95_supported={target['ci95_lower_bound_supports_target']}"
    )


if __name__ == "__main__":
    main()
