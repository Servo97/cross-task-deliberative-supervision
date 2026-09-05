"""ReMemBench task registry — the 13 ``Mem*`` variants and their 4 memory categories.

Why this file exists instead of an entry in RoboCasa's ``TASK_SET_REGISTRY``: ReMemBench
is a *separate* RoboCasa v0.2 fork (~/Research/TRI/ReMemBench). Its tasks are not in
``ATOMIC_TASK_DATASETS`` / ``COMPOSITE_TASK_DATASETS``, so ``get_task_horizon`` raises on
them, and we do not modify the upstream robocasa checkout. Everything ReMemBench needs is
therefore resolved here, and every consumer branches on
:func:`is_remembench_task_set` so the RoboCasa eval path is untouched.

The category — not the variant — is the scientific unit. Six of the thirteen variants are
just corner/side permutations of two underlying spatial tasks; reporting them as thirteen
independent numbers would triple-count the spatial condition. Aggregate by category.

Horizons are mirrored from ``robocasa/wrappers/gym_wrapper.py:TASK_HORIZONS`` in the
ReMemBench checkout (see that file for the full derivation: the larger of an analytic
deadline bound and 1.15x the longest teleoperated demo in the task family). The mirror is
here so manifests can be built in an image without the simulator installed;
``tests/test_remembench_tasks.py`` asserts the two tables agree whenever ReMemBench is
importable.
"""

import math

#: Variant -> memory category.
REMEMBENCH_TASK_CATEGORIES = {
    # Spatial: recall where an object/landmark was seen during an exploration phase.
    "MemFruitInSinkLeftFar": "spatial",
    "MemFruitInSinkRightFar": "spatial",
    "MemRetrieveOilsFromCounterLL": "spatial",
    "MemRetrieveOilsFromCounterLR": "spatial",
    "MemRetrieveOilsFromCounterRL": "spatial",
    "MemRetrieveOilsFromCounterRR": "spatial",
    # Prospective: act at a future deadline (cook N minutes, then turn the stove off).
    "MemHeatPot": "prospective",
    "MemHeatPotMultiple": "prospective",
    # Object-associative: remember where an object came from and put it back there.
    "MemWashAndReturnLeft": "object_associative",
    "MemWashAndReturnRight": "object_associative",
    "MemWashAndReturnSameLocation": "object_associative",
    # Object-set: remember *how many* objects; success needs the exact count + closed door.
    "MemPutKBreadInMicrowave": "object_set",
    "MemPutKBowlInCabinet": "object_set",
}

#: Stable category order for reporting.
REMEMBENCH_CATEGORIES = ("spatial", "prospective", "object_associative", "object_set")

#: Category -> sorted variants.
REMEMBENCH_CATEGORY_TASKS = {
    category: sorted(task for task, cat in REMEMBENCH_TASK_CATEGORIES.items() if cat == category)
    for category in REMEMBENCH_CATEGORIES
}

#: All 13 variants, in category order then alphabetical — the canonical eval order.
REMEMBENCH_TASKS = tuple(task for category in REMEMBENCH_CATEGORIES for task in REMEMBENCH_CATEGORY_TASKS[category])

#: Evaluation horizon in 20 Hz control steps. Mirrors ReMemBench's gym_wrapper.TASK_HORIZONS.
REMEMBENCH_HORIZONS = {
    "MemFruitInSinkLeftFar": 1400,
    "MemFruitInSinkRightFar": 1400,
    "MemRetrieveOilsFromCounterLL": 1100,
    "MemRetrieveOilsFromCounterLR": 1100,
    "MemRetrieveOilsFromCounterRL": 1100,
    "MemRetrieveOilsFromCounterRR": 1100,
    "MemHeatPot": 2600,
    "MemHeatPotMultiple": 3200,
    "MemWashAndReturnLeft": 1000,
    "MemWashAndReturnRight": 1000,
    "MemWashAndReturnSameLocation": 1000,
    "MemPutKBreadInMicrowave": 2200,
    "MemPutKBowlInCabinet": 2400,
}

#: Task-set names this module owns. ``remembench`` is the full 13-variant set; the
#: per-category sets exist so a cheap single-category probe can be run without inventing
#: a second selection mechanism.
REMEMBENCH_TASK_SETS = {
    "remembench": list(REMEMBENCH_TASKS),
    **{f"remembench_{category}": list(REMEMBENCH_CATEGORY_TASKS[category]) for category in REMEMBENCH_CATEGORIES},
}


def is_remembench_task_set(name) -> bool:
    """True for any task-set name owned by this module (the knob everything gates on)."""
    return name in REMEMBENCH_TASK_SETS


def is_remembench_task(task) -> bool:
    return task in REMEMBENCH_TASK_CATEGORIES


def get_remembench_horizon(task) -> int:
    try:
        return REMEMBENCH_HORIZONS[task]
    except KeyError:
        raise ValueError(f"not a ReMemBench task: {task!r}") from None


def get_remembench_category(task) -> str:
    try:
        return REMEMBENCH_TASK_CATEGORIES[task]
    except KeyError:
        raise ValueError(f"not a ReMemBench task: {task!r}") from None


def list_remembench_tasks(task_set):
    """``[{task, split_set, horizon, category}]`` for one ReMemBench task-set.

    ``split_set`` is the task-set name (matching ``eval_common.list_tasks``' contract, and
    therefore the ``<out_dir>/<split_set>/<task>/stats.json`` layout); ``category`` is the
    extra field ReMemBench adds, used for the per-category rollup in ``aggregate_eval``.
    """
    if not is_remembench_task_set(task_set):
        raise ValueError(f"not a ReMemBench task set: {task_set!r}")
    return [
        {
            "task": task,
            "split_set": task_set,
            "horizon": get_remembench_horizon(task),
            "category": get_remembench_category(task),
        }
        for task in REMEMBENCH_TASK_SETS[task_set]
    ]


def summarize_by_category(per_task_rates):
    """Roll per-task success rates up to the 4 memory categories.

    ``per_task_rates``: ``{task_name: success_rate}``. Returns an ordered dict of
    ``{category: {mean, n_tasks_done, n_tasks_expected, per_task}}``, skipping categories
    with no ReMemBench tasks present. Unweighted mean over the variants in the category —
    each variant contributes equally, which is the intended reading given that the
    variants are permutations of one condition.
    """
    out = {}
    for category in REMEMBENCH_CATEGORIES:
        expected = REMEMBENCH_CATEGORY_TASKS[category]
        present = {t: per_task_rates[t] for t in expected if t in per_task_rates}
        if not present:
            continue
        rates = list(present.values())
        out[category] = {
            "mean": sum(rates) / len(rates),
            "n_tasks_done": len(present),
            "n_tasks_expected": len(expected),
            "per_task": present,
        }
    return out


def heldout_split(demo_indices, fraction=0.2, minimum=3):
    """Split one task's ordered demo indices into (train, heldout).

    Held-out = the LAST ``ceil(fraction * n)`` demos by demo index, floored at ``minimum``.
    Taking the tail (rather than a hash-random subset) keeps the split trivially
    reproducible and auditable from the hdf5 alone, and keeps whole collection sessions
    from being split down the middle more than once.
    """
    ordered = list(demo_indices)
    n = len(ordered)
    if n == 0:
        return [], []
    n_heldout = min(n, max(minimum, math.ceil(fraction * n)))
    return ordered[: n - n_heldout], ordered[n - n_heldout :]
