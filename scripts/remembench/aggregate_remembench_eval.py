#!/usr/bin/env python3
"""Roll ReMemBench per-task shard stats up to the 4 memory categories + overall.

The reported unit is the memory CATEGORY, not the variant: six of the thirteen variants are
corner/side permutations of two underlying spatial tasks, so a flat 13-task mean would
triple-count the spatial condition. Overall = unweighted mean over the 4 category means,
matching ``metric: category_weighted_avg_success`` in pi05_remembench_eval.yaml.

  python scripts/remembench/aggregate_remembench_eval.py \
      --results-dir /data/work/remembench_evals/<arm> --arm <arm> --step 14999
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vla_training.eval.remembench_tasks import (  # noqa: E402
    REMEMBENCH_CATEGORIES,
    REMEMBENCH_TASKS,
    summarize_by_category,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--checkpoint-uri", default=None)
    ap.add_argument("--manifest-sha256", default=None)
    ap.add_argument("--out", default=None, help="default <results-dir>/results.json")
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all 13 variants reported",
    )
    args = ap.parse_args()

    per_task = {}
    shard_shas = set()
    total_rollouts = 0
    total_success = 0
    wall = 0.0
    for path in sorted(glob.glob(os.path.join(args.results_dir, "remembench", "*", "stats_w*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        if not payload.get("complete"):
            raise SystemExit(f"incomplete shard: {path}")
        task = payload["task"]
        if task in per_task:
            raise SystemExit(f"duplicate stats for {task} ({path})")
        shard_shas.add(payload["manifest_sha256"])
        episodes = payload["per_episode"]
        successes = sum(1 for e in episodes if e["success"])
        total_rollouts += len(episodes)
        total_success += successes
        wall += float(payload.get("wall_seconds", 0.0))
        # Per-episode rate first, then mean over episodes: every held-out episode gets equal
        # weight regardless of how many rollouts survived, and matches the x3 protocol.
        by_episode = {}
        for e in episodes:
            by_episode.setdefault(int(e["episode_index"]), []).append(bool(e["success"]))
        episode_rates = {k: sum(v) / len(v) for k, v in sorted(by_episode.items())}
        per_task[task] = {
            "category": payload["category"],
            "num_episodes": len(by_episode),
            "num_rollouts": len(episodes),
            "num_success": successes,
            # Rollout-level rate (the headline): successes / total rollouts.
            "success_rate": successes / len(episodes),
            "success_rate_episode_mean": sum(episode_rates.values()) / len(episode_rates),
            "n_hard_fail": sum(1 for e in episodes if e.get("failed_task")),
            "mean_episode_length": sum(e["episode_length"] for e in episodes) / len(episodes),
            "per_episode_rate": episode_rates,
        }

    if not per_task:
        raise SystemExit(f"no stats found under {args.results_dir}")
    if len(shard_shas) != 1:
        raise SystemExit(f"shards disagree on manifest_sha256: {sorted(shard_shas)}")
    manifest_sha = shard_shas.pop()
    if args.manifest_sha256 and manifest_sha != args.manifest_sha256:
        raise SystemExit(f"manifest sha mismatch: {manifest_sha} != {args.manifest_sha256}")
    missing = [t for t in REMEMBENCH_TASKS if t not in per_task]
    if missing and args.require_complete:
        raise SystemExit(f"missing tasks: {missing}")

    rates = {t: v["success_rate"] for t, v in per_task.items()}
    by_category = summarize_by_category(rates)
    cat_means = [by_category[c]["mean"] for c in REMEMBENCH_CATEGORIES if c in by_category]
    overall = sum(cat_means) / len(cat_means)

    for cat, block in by_category.items():
        block["num_rollouts"] = sum(per_task[t]["num_rollouts"] for t in block["per_task"])
        block["num_episodes"] = sum(per_task[t]["num_episodes"] for t in block["per_task"])

    out = {
        "benchmark": "ReMemBench",
        "arm": args.arm,
        "step": args.step,
        "checkpoint_uri": args.checkpoint_uri,
        "manifest_sha256": manifest_sha,
        "metric": "category_weighted_avg_success",
        "overall_category_weighted": overall,
        "overall_rollout_pooled": total_success / total_rollouts,
        "num_tasks_done": len(per_task),
        "num_tasks_expected": len(REMEMBENCH_TASKS),
        "missing_tasks": missing,
        "total_episodes": sum(v["num_episodes"] for v in per_task.values()),
        "total_rollouts": total_rollouts,
        "total_success": total_success,
        "wall_seconds": round(wall, 1),
        "by_category": by_category,
        "by_task": per_task,
    }
    out_path = args.out or os.path.join(args.results_dir, "results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)

    print(f"== ReMemBench {args.arm} step {args.step} ==")
    print(f"  {'category':22s} {'succ':>7s}  {'tasks':>7s} {'eps':>5s} {'rollouts':>9s}")
    for cat in REMEMBENCH_CATEGORIES:
        if cat not in by_category:
            continue
        b = by_category[cat]
        print(
            f"  {cat:22s} {b['mean'] * 100:6.1f}%  "
            f"{b['n_tasks_done']}/{b['n_tasks_expected']:<5d} "
            f"{b['num_episodes']:5d} {b['num_rollouts']:9d}"
        )
    print(f"  {'OVERALL (cat-wtd)':22s} {overall * 100:6.1f}%")
    print(
        f"  {'overall (pooled)':22s} {out['overall_rollout_pooled'] * 100:6.1f}%  "
        f"({total_success}/{total_rollouts} rollouts)"
    )
    if missing:
        print(f"  MISSING TASKS: {missing}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
