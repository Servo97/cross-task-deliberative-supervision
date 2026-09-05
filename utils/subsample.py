"""Deterministic episode subsampling — byte-identical to robocasa ``get_subset_demos_filter_key``.

Pure-python (stdlib ``random`` + ``json`` over ``meta/episodes.jsonl``); framework-agnostic, so it
imports in any venv. Used to make GR00T's finetune episodes IDENTICAL to pi0.5's ``filter_key``
selection (pi0.5 gets it natively; GR00T ignores ``filter_key`` and applies this via a loader patch).

Parity contract (must match robocasa groot_dataset.get_subset_demos_filter_key exactly):
  * read the ``episode_index`` VALUES from episodes.jsonl in file order;
  * shuffle with stdlib Mersenne-Twister seeded by ``seed`` (default 0) — NOT numpy;
  * keep the first ``num_demos`` (a COUNT from the filter_key, e.g. "150_demos" -> 150), or ALL if
    num_demos >= total. ``random.Random(seed).shuffle`` is byte-identical to robocasa's
    ``random.seed(seed); random.shuffle(...)`` (same MT stream).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def num_demos_from_filter_key(filter_key: str) -> int:
    """'150_demos' -> 150 (the same parse robocasa's loader does)."""
    return int(str(filter_key).split("_")[0])


def episode_index_keep_set(dataset_path: str | Path, num_demos: int, seed: int = 0) -> set[int] | None:
    """The ``episode_index`` VALUES to keep, or None to keep all (num_demos >= total)."""
    ep_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    ids: list[int] = []
    with open(ep_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["episode_index"])
    if num_demos >= len(ids):
        return None
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    return set(ids[:num_demos])


def uniform_num_demos(soup: list[dict]) -> int:
    """The single num_demos shared by every ds_meta's filter_key in a soup (asserts uniformity).
    The combined target finetune soup is all '150_demos', so this is 150.

    A ``filter_key=None`` soup (native full mass — e.g. ``remembench_soup``, or
    ``combined_target_soup(1.0)``) has no count to parse and must never reach this function; the
    guard turns an opaque ``int('None')`` ValueError into the actual contract violation. The pi0.5
    finetune path never calls this at all (filter_key is consumed natively by robocasa's loader);
    only the GR00T adapters do, to mirror pi's selection."""
    unfiltered = [m["task"] for m in soup if m.get("filter_key") is None]
    if unfiltered:
        raise ValueError(
            "uniform_num_demos requires a filter_key on every ds_meta, but "
            f"{len(unfiltered)} entries have filter_key=None (native full mass, no subsample) — "
            f"e.g. {unfiltered[:3]}. Such a soup uses ALL demos; there is no count to derive."
        )
    counts = {num_demos_from_filter_key(m["filter_key"]) for m in soup}
    if len(counts) != 1:
        raise ValueError(f"non-uniform filter_key counts across soup: {sorted(counts)}")
    return counts.pop()
