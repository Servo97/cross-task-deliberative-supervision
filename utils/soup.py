"""Bridge to the RoboCasa dataset registry — the framework-agnostic data substrate.

This is the ONLY module that imports ``robocasa``. Everything downstream (balancing,
the backbone adapters) operates on plain ``list[ds_meta]`` dicts, so it stays importable
from BOTH the pi0.5 (JAX) venv and the GR00T (PyTorch) venv. It must never import jax/torch.

A *soup* is a ``list[ds_meta]``; each ``ds_meta`` is the dict produced by
``robocasa.utils.dataset_registry_utils.get_ds_meta`` with at least these keys:
``path`` (absolute lerobot dir), ``task``, ``split``, ``source`` ("human"|"mg"),
``horizon``, ``filter_key`` (e.g. "100_demos"/"10000_demos"/"150_demos").

The three balancing groups (see ``utils.balancing``) are derived purely from registry
membership — NO path parsing:
  * ``mg_atomic``        — source is MimicGen and task is atomic
  * ``human_atomic``     — source is human and task is atomic
  * ``human_composite``  — source is human and task is composite
(MimicGen is atomic-only in RoboCasa365, so ``mg_composite`` cannot occur.)

A SECOND substrate lives here too: ``remembench_soup`` (soup name ``remembench13``), a robocasa-free
glob over the FLAT ``<root>/<Task>/<date>/lerobot`` ReMemBench export. See its docstring for the env
knobs (``WSM_REMEMBENCH_ROOT`` / ``WSM_REMEMBENCH_TASK_GLOB`` / ``WSM_REMEMBENCH_GROUP``); nothing
about the RoboCasa ``target50`` path changes.
"""

from __future__ import annotations

import copy
import os
from collections import OrderedDict

# robocasa is imported LAZILY (inside the functions that need it) so this module imports cleanly in a
# venv WITHOUT robocasa — e.g. the GR00T finetune node, which resolves the target soup from a dir glob
# (WSM_SOUP_FROM_DIRS) instead of the registry, to avoid the fragile robocasa/robosuite/numba overlay.
# See _glob_target_soup. When WSM_SOUP_FROM_DIRS is unset, behavior is identical to the registry path.

GROUPS = ("mg_atomic", "human_atomic", "human_composite")

_MG_SOURCES = {"mg", "mg_5x5", "mg_5x1"}
_HUMAN_SOURCES = {"human", "human_cotraining_cams"}


def _check_dataset_base_path() -> None:
    """ds_meta['path'] is built against macros.DATASET_BASE_PATH at registry import time.
    If it is unset the soup paths fall back to the robocasa package dir, which is almost
    never what we want on a training node — surface it early."""
    import robocasa.macros as _macros

    if getattr(_macros, "DATASET_BASE_PATH", None) is None:
        raise RuntimeError(
            "robocasa.macros.DATASET_BASE_PATH is None — soup paths will be wrong. "
            "Set it (macros_private.py or the env) to the dir that holds v1.0/{pretrain,target}."
        )


def resolve_soup(
    name: str | None = None,
    *,
    split: str | None = None,
    task_set: str | None = None,
    source: str | None = None,
    demo_fraction: float = 1.0,
) -> list[dict]:
    """Resolve a soup either by registered NAME or by (split, task_set, source).

    Returns a deep COPY so callers may safely mutate per-entry fields (e.g. ``filter_key``
    for a demo-fraction override) without corrupting the registry's cached soups.
    """
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_ds_soup

    _check_dataset_base_path()
    if name is not None:
        if name not in DATASET_SOUP_REGISTRY:
            raise KeyError(f"soup '{name}' not in DATASET_SOUP_REGISTRY. Available: {sorted(DATASET_SOUP_REGISTRY)}")
        return copy.deepcopy(DATASET_SOUP_REGISTRY[name])
    if not (split and task_set and source):
        raise ValueError("resolve_soup needs either name=, or all of split/task_set/source")
    return copy.deepcopy(get_ds_soup(split=split, task_set=task_set, source=source, demo_fraction=demo_fraction))


def combined_target_soup(demo_fraction: float = 0.30) -> list[dict]:
    """The combined finetune soup: atomic_seen + composite_seen + composite_unseen at a
    single demo_fraction (default 0.30, the plan's combined-30% finetune).

    For 0.10/0.30 this reuses the pre-registered ``target_*_{10,30}p`` soups; otherwise it
    builds them on the fly. NOTE: the demo_fraction is realized natively by pi0.5 (via each
    ds_meta's ``filter_key``); GR00T ignores ``filter_key`` and needs explicit episode
    selection at the adapter layer (Phase 2 — see plan)."""
    base = os.environ.get("WSM_SOUP_FROM_DIRS")
    if base:  # robocasa-free path (GR00T FT node): build from the on-disk target layout
        soup = _glob_target_soup(base, demo_fraction)
    else:
        from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

        pct = int(round(demo_fraction * 100))
        sets = ("atomic_seen", "composite_seen", "composite_unseen")
        soup = []
        for ts in sets:
            reg_name = f"target_{ts}" if demo_fraction >= 1.0 else f"target_{ts}_{pct}p"
            if reg_name in DATASET_SOUP_REGISTRY:
                soup += resolve_soup(reg_name)
            else:
                soup += resolve_soup(split="target", task_set=ts, source="human", demo_fraction=demo_fraction)
    return _filter_soup_tasks(soup)


def _filter_soup_tasks(soup: list[dict]) -> list[dict]:
    """Optional WSM_TASKS (comma-separated task names) -> restrict the combined soup to those tasks, for
    fast small-task POC runs (any N tasks). Unset -> full 50-task soup (unchanged). Single chokepoint: this
    also restricts the seed-0 subsample (uniform_num_demos) and the w_t coverage check, since both build off
    combined_target_soup. Fail loud on a typo / missing task so a POC never silently trains on the wrong set."""
    spec = os.environ.get("WSM_TASKS", "").strip()
    if not spec:
        return soup
    want = [t.strip() for t in spec.split(",") if t.strip()]
    have = {m["task"] for m in soup}
    missing = [t for t in want if t not in have]
    if missing:
        raise RuntimeError(
            f"WSM_TASKS requested tasks not in the target soup: {missing}. Available ({len(have)}): {sorted(have)}"
        )
    out = [m for m in soup if m["task"] in set(want)]
    print(f"[soup] WSM_TASKS filter -> {len(out)} dirs across {len(want)} tasks: {want}", flush=True)
    return out


def _glob_target_soup(base: str, demo_fraction: float) -> list[dict]:
    """robocasa-FREE combined-target soup, built from the on-disk layout
    ``<base>/v1.0/target/{atomic,composite}/<Task>/<date>/lerobot``. Used on the GR00T finetune node
    (no robocasa/robosuite). The ``filter_key`` encodes the demo COUNT (seed-0), matching the registry
    convention (target ~500 demos/task, so 0.30 -> '150_demos'); ``uniform_num_demos`` + the seed-0
    episode subsample then stay byte-identical to pi0.5's selection and the WSM label keep-set. Each
    meta carries an explicit ``group`` so source_group_of needs no registry lookup."""
    from glob import glob
    from pathlib import Path

    count = int(round(demo_fraction * 500))  # target tasks ~500 demos; 0.30 -> 150 (registry convention)
    filt = None if demo_fraction >= 1.0 else f"{count}_demos"
    soup: list[dict] = []
    for sub, group in (("atomic", "human_atomic"), ("composite", "human_composite")):
        pattern = os.path.join(base, "v1.0", "target", sub, "*", "*", "lerobot")
        for lerobot in sorted(glob(pattern)):
            task = Path(lerobot).parents[1].name  # .../target/<sub>/<Task>/<date>/lerobot
            soup.append(
                {
                    "path": lerobot,
                    "task": task,
                    "split": "target",
                    "source": "human",
                    "filter_key": filt,
                    "group": group,
                    "horizon": None,
                }
            )
    if not soup:
        raise RuntimeError(
            f"WSM_SOUP_FROM_DIRS={base!r}: no target lerobot dirs under {base}/v1.0/target/"
            "{atomic,composite}/*/*/lerobot"
        )
    return soup


# --------------------------------------------------------------------------------------------
# ReMemBench (flat layout) — a second, robocasa-FREE dataset substrate.
# --------------------------------------------------------------------------------------------
# RoboCasa nests target tasks as ``<root>/{atomic,composite}/<Task>/<date>/lerobot``; the ReMemBench
# v02 export is FLAT: ``<root>/<Task>/<date>/lerobot``. Everything else (task name = parents[1].name)
# is identical, so only the glob and the base dir differ.
#
# BASE DIR. ``WSM_REMEMBENCH_ROOT`` is the absolute dir that DIRECTLY contains ``<Task>/<date>/
# lerobot`` — i.e. exactly the ``--target-root`` the Stage-S validators receive, so the soup and the
# validators can never disagree about which tree is being trained on. A dedicated variable (rather
# than a layout flag on ``WSM_SOUP_FROM_DIRS``) is used because the two variables mean different
# things: WSM_SOUP_FROM_DIRS is a dataset BASE under which ``v1.0/target/...`` is appended, while
# this one is the task-dir parent itself. ``WSM_SOUP_FROM_DIRS`` is still honoured as a fallback so
# a node that only exports the older variable keeps working (it is then used verbatim, not with
# ``v1.0/target`` appended).
#
# MASS. ``filter_key`` is None => robocasa's ``get_subset_demos_filter_key`` keeps EVERY episode, so
# the soup is the native full 323-demo set with no seed-0 subsampling anywhere in the stack (and
# ``utils.subsample.uniform_num_demos`` must never be called on it — it parses filter_key).
#
# GROUP. Each meta carries an explicit ``group`` so ``source_group_of`` never touches the robocasa
# registry (which has no ReMemBench tasks). The default is ``human_composite`` — ReMemBench tasks are
# multi-stage — and it is only ever observable in the summary/partition, since balancing is OFF for
# these arms (pi05_weights None, one "all" GR00T spec).
REMEMBENCH13_SOUP = "remembench13"
_REMEMBENCH_ROOT_ENV = "WSM_REMEMBENCH_ROOT"
_REMEMBENCH_GLOB_ENV = "WSM_REMEMBENCH_TASK_GLOB"
_REMEMBENCH_GROUP_ENV = "WSM_REMEMBENCH_GROUP"
REMEMBENCH_DEFAULT_TASK_GLOB = "*/*/lerobot"
REMEMBENCH_DEFAULT_GROUP = "human_composite"


def remembench_soup(
    root: str | None = None,
    *,
    task_dir_glob: str | None = None,
    group: str | None = None,
) -> list[dict]:
    """The ReMemBench finetune soup: ALL demos of every task under a FLAT ``<Task>/<date>/lerobot``.

    ``root`` defaults to ``$WSM_REMEMBENCH_ROOT`` (fallback ``$WSM_SOUP_FROM_DIRS``); the glob and
    the group default to ``$WSM_REMEMBENCH_TASK_GLOB`` / ``$WSM_REMEMBENCH_GROUP``. No robocasa
    import, no subsampling (``filter_key=None`` => native full mass), and ``WSM_TASKS`` still
    restricts the set for small POC runs, exactly as it does for the combined target soup.
    """
    from glob import glob
    from pathlib import Path

    base = root or os.environ.get(_REMEMBENCH_ROOT_ENV) or os.environ.get("WSM_SOUP_FROM_DIRS")
    if not base:
        raise RuntimeError(
            f"soup '{REMEMBENCH13_SOUP}' needs {_REMEMBENCH_ROOT_ENV} (or WSM_SOUP_FROM_DIRS) set to "
            "the dir that directly contains <Task>/<date>/lerobot"
        )
    pattern_tail = task_dir_glob or os.environ.get(_REMEMBENCH_GLOB_ENV) or REMEMBENCH_DEFAULT_TASK_GLOB
    grp = group or os.environ.get(_REMEMBENCH_GROUP_ENV) or REMEMBENCH_DEFAULT_GROUP
    if grp not in GROUPS:
        raise ValueError(f"remembench group {grp!r} must be one of {list(GROUPS)}")
    soup: list[dict] = []
    seen: set[str] = set()
    for lerobot in sorted(glob(os.path.join(base, pattern_tail))):
        task = Path(lerobot).parents[1].name  # .../<Task>/<date>/lerobot
        if task in seen:
            raise RuntimeError(f"remembench soup has duplicate task dir for {task!r} under {base}")
        seen.add(task)
        soup.append(
            {
                "path": lerobot,
                "task": task,
                "split": "target",
                "source": "human",
                "filter_key": None,
                "group": grp,
                "horizon": None,
            }
        )
    if not soup:
        raise RuntimeError(f"{_REMEMBENCH_ROOT_ENV}={base!r}: no lerobot dirs under {base}/{pattern_tail}")
    return _filter_soup_tasks(soup)


def source_group_of(meta: dict) -> str:
    """Classify a ds_meta into one of GROUPS using registry membership only (no path parsing).
    A glob-built meta (robocasa-free node) carries an explicit ``group`` — use it directly."""
    if meta.get("group") in GROUPS:
        return meta["group"]
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS

    src, task = meta["source"], meta["task"]
    is_mg = src in _MG_SOURCES
    is_human = src in _HUMAN_SOURCES
    is_atomic = task in ATOMIC_TASK_DATASETS
    is_composite = task in COMPOSITE_TASK_DATASETS
    if not (is_mg or is_human):
        raise ValueError(f"unknown source {src!r} for task {task!r}")
    if not (is_atomic or is_composite):
        raise ValueError(f"task {task!r} is in neither ATOMIC nor COMPOSITE registry")
    if is_mg and is_composite:
        raise ValueError(f"unexpected MimicGen-composite dataset for {task!r}; MimicGen is atomic-only")
    if is_mg:
        return "mg_atomic"
    return "human_atomic" if is_atomic else "human_composite"


def partition_by_group(soup: list[dict]) -> "OrderedDict[str, list[dict]]":
    """Group a soup by source-group, preserving soup order within each group.
    Returns an OrderedDict over GROUPS; absent groups map to []."""
    out: "OrderedDict[str, list[dict]]" = OrderedDict((g, []) for g in GROUPS)
    for meta in soup:
        out[source_group_of(meta)].append(meta)
    return out


def dirs_for_group(metas: list[dict]) -> list[str]:
    """The concrete lerobot dir paths for a list of ds_meta (GR00T SingleDatasetConfig needs these)."""
    return [m["path"] for m in metas]
