"""Aggregate per-task stats.json -> results.json for one (model, step) eval.

Metric: task-weighted mean over the 50-task target set (= mean over all per-task success rates),
alongside per-split means. results.json is the completion marker (uploaded last). Default scene
split = 'target' (foundation-model-learning; the reference's 'pretrain' default was the legacy bug).

Ported from internal_planning_and_todos/ported_raw/reference_code/aggregate_eval.py.
"""

import argparse
import json
import os
import time

from vla_training.eval.remembench_tasks import (
    REMEMBENCH_TASK_SETS,
    is_remembench_task_set,
    summarize_by_category,
)

DEFAULT_TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")


def _expected_tasks(task_set):
    """Task list for one task set. ReMemBench sets resolve locally; everything else comes
    from RoboCasa's registry (imported lazily so a ReMemBench-only aggregation does not
    require the RoboCasa checkout)."""
    if is_remembench_task_set(task_set):
        return list(REMEMBENCH_TASK_SETS[task_set])
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    return list(TASK_SET_REGISTRY[task_set])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--split", default="target", choices=["pretrain", "target"])
    ap.add_argument("--num-trials", type=int, required=True)
    ap.add_argument("--task-sets", default=",".join(DEFAULT_TASK_SETS))
    args = ap.parse_args()

    splits = {}
    all_rates = []
    remembench_rates = {}
    for ts in args.task_sets.split(","):
        expected = _expected_tasks(ts)
        per_task = {}
        for t in expected:
            p = os.path.join(args.results_dir, ts, t, "stats.json")
            if os.path.exists(p):
                with open(p) as f:
                    per_task[t] = json.load(f)["success_rate"]
        rates = list(per_task.values())
        splits[ts] = {
            "mean": sum(rates) / len(rates) if rates else None,
            "n_tasks_done": len(per_task),
            "n_tasks_expected": len(expected),
            "per_task": per_task,
        }
        all_rates.extend(rates)
        if is_remembench_task_set(ts):
            remembench_rates.update(per_task)

    # ReMemBench reports by memory CATEGORY, not by variant: six of the thirteen variants
    # are corner/side permutations of two spatial tasks, so a flat 13-task mean would
    # triple-count the spatial condition. Emitted alongside (never instead of) `splits`,
    # and absent entirely from a RoboCasa-only run.
    by_category = summarize_by_category(remembench_rates) if remembench_rates else None

    complete = all(s["n_tasks_done"] == s["n_tasks_expected"] for s in splits.values())
    out = {
        "model": args.model,
        "step": args.step,
        "split": args.split,
        "num_trials": args.num_trials,
        "splits": splits,
        "avg_task_weighted": sum(all_rates) / len(all_rates) if all_rates else None,
        "n_tasks_done": len(all_rates),
        "complete": complete,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if by_category is not None:
        out["remembench_by_category"] = by_category
        cat_means = [c["mean"] for c in by_category.values()]
        out["remembench_avg_category_weighted"] = sum(cat_means) / len(cat_means)
    path = os.path.join(args.results_dir, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"== {args.model} step {args.step} ({args.split}, {args.num_trials} trials) ==")
    for ts, s in splits.items():
        mean = f"{s['mean'] * 100:.1f}%" if s["mean"] is not None else "n/a"
        print(f"  {ts:18s} {mean:>7s}  ({s['n_tasks_done']}/{s['n_tasks_expected']} tasks)")
    avg = out["avg_task_weighted"]
    print(f"  {'avg (task-wtd)':18s} {avg * 100:.1f}%" if avg is not None else "  no tasks completed")
    if by_category is not None:
        print("  -- ReMemBench by memory category --")
        for category, c in by_category.items():
            print(f"  {category:18s} {c['mean'] * 100:6.1f}%  ({c['n_tasks_done']}/{c['n_tasks_expected']} variants)")
        print(f"  {'avg (cat-wtd)':18s} {out['remembench_avg_category_weighted'] * 100:6.1f}%")
    print(f"  complete={complete} -> {path}")


if __name__ == "__main__":
    main()
