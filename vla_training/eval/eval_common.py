"""Shared helpers for RoboCasa365 eval runners (sim venv, torch-free).

Task universe + metric: the 50-task target set (atomic_seen 18 / composite_seen 16 /
composite_unseen 16); the reported "Average" is the task-weighted mean over all 50 tasks (= mean
over per-task success rates). The gym SCENE split is chosen by the runner (gym.make(..., split=)):
foundation-model-learning evals on the `target` split (held-out scenes/objects); the older
multitask metric used `pretrain` scenes. These helpers are split-agnostic.

Ported verbatim from internal_planning_and_todos/ported_raw/reference_code/eval_common.py.
"""

import os

DEFAULT_TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")


def list_tasks(task_sets, only=None):
    """[{task, split_set, horizon}] over the given task sets, deduped (first set wins).

    `only` (iterable of task names) -> restrict to that subset, for fast small-task POC evals (keeps each
    task's correct split_set label). Fail loud on a requested task absent from the given sets.

    ReMemBench task sets (`remembench`, `remembench_<category>`) resolve through
    vla_training/eval/remembench_tasks.py instead of RoboCasa's registry — their tasks live in a
    separate RoboCasa fork and are absent from ATOMIC/COMPOSITE_TASK_DATASETS. Entries from those
    sets carry an extra `category` field; RoboCasa entries are unchanged."""
    from vla_training.eval.remembench_tasks import is_remembench_task_set, list_remembench_tasks

    only = set(only) if only else None
    seen, out = set(), []
    for ts in task_sets:
        if is_remembench_task_set(ts):
            for entry in list_remembench_tasks(ts):
                if entry["task"] in seen or (only is not None and entry["task"] not in only):
                    continue
                seen.add(entry["task"])
                out.append(entry)
            continue
        # Keep merge/manifest validation usable in images that do not install the simulator.
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        for t in TASK_SET_REGISTRY[ts]:
            if t in seen or (only is not None and t not in only):
                continue
            seen.add(t)
            out.append({"task": t, "split_set": ts, "horizon": get_task_horizon(t)})
    if only is not None:
        missing = only - {e["task"] for e in out}
        if missing:
            raise SystemExit(f"--tasks not found in task_sets {list(task_sets)}: {sorted(missing)}")
    return out


# Measured per-task wall-clock priors (median minutes per 10-trial block across 3 completed 8-way evals,
# 2026-07-08 H3 analysis). Duration does NOT track the registry horizon (WashFruitColander 32min vs
# PrepareCoffee 17min) — LPT-packing on these priors repairs an 8-12% makespan loss vs horizon-snake.
# Unlisted tasks fall back to horizon-proportional weight.
TASK_MINUTES = {
    "WashFruitColander": 32.1,
    "WeighIngredients": 21.8,
    "ArrangeTea": 19.8,
    "PreSoakPan": 18.8,
    "CategorizeCondiments": 18.6,
    "PrepareCoffee": 16.6,
    "PanTransfer": 15.8,
    "LoadDishwasher": 14.7,
    "KettleBoiling": 10.9,
    "WashLettuce": 10.9,
    "CuttingToolSelection": 10.2,
    "PickPlaceCounterToCabinet": 6.7,
    "PickPlaceCounterToStove": 4.8,
    "TurnOnElectricKettle": 3.5,
    "SlideDishwasherRack": 3.4,
}


def _weight(e) -> float:
    return TASK_MINUTES.get(e["task"], e["horizon"] / 60.0)


def shard_tasks(tasks, worker_idx, num_workers):
    """Greedy LPT-pack whole tasks onto workers by MEASURED duration priors (horizon fallback) so
    per-worker wall-clocks balance; EXECUTE each worker's tasks shortest-first so cheap atomic tasks
    land their diagnostic signal early. Deterministic (stable sort + fixed tie-break); tasks stay
    atomic (no per-task stats merging). Episode-level sharding was evaluated and rejected — ceiling
    1.15-1.19x before env-build costs, a wash after (H3, 2026-07-08)."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if not 0 <= worker_idx < num_workers:
        raise ValueError(f"worker_idx must be in [0, {num_workers}), got {worker_idx}")
    ordered = sorted(tasks, key=lambda e: (-_weight(e), e["task"]))
    loads = [0.0] * num_workers
    assign = [[] for _ in range(num_workers)]
    for e in ordered:
        w = min(range(num_workers), key=lambda i: (loads[i], i))
        loads[w] += _weight(e)
        assign[w].append(e)
    return sorted(assign[worker_idx], key=lambda e: (_weight(e), e["task"]))


def stats_path(out_dir, split_set, task):
    return os.path.join(out_dir, split_set, task, "stats.json")


def write_stats(path, stats):
    from vla_training.eval.eval_manifest import write_json_atomic

    write_json_atomic(path, stats)
