"""Pure fixed-shape sequence helpers shared by RoboMME training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def uniformly_sample_prefix(
    indices: Sequence[int], count: int, *, pad_index: int
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return a fixed-width, chronological prefix and its validity mask.

    RoboMME's neural-memory recipe uniformly samples a long demonstration before execution. A
    fixed width keeps JAX shapes static. Short prefixes are *left* padded, matching the paper's
    history-buffer convention, and padding is explicitly invalid so it can never update fast
    weights. ``pad_index`` must belong to the same episode.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got {values.shape}")
    if values.size:
        if np.any(np.diff(values) <= 0):
            raise ValueError("prefix indices must be strictly increasing")
        if values.size > count:
            # Integer linspace includes both endpoints, preserves chronology, and has no duplicates
            # when values.size > count.
            positions = np.linspace(0, values.size - 1, count, dtype=np.int64)
            selected = values[positions]
            valid = np.ones((count,), dtype=bool)
        else:
            padding = count - values.size
            selected = np.concatenate([np.full((padding,), int(pad_index), dtype=np.int64), values])
            valid = np.concatenate(
                [
                    np.zeros((padding,), dtype=bool),
                    np.ones((values.size,), dtype=bool),
                ]
            )
    else:
        selected = np.full((count,), int(pad_index), dtype=np.int64)
        valid = np.zeros((count,), dtype=bool)
    return tuple(int(value) for value in selected), valid
