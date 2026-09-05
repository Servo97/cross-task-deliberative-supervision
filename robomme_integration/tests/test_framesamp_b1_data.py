from __future__ import annotations

import numpy as np
import pytest

from robomme_integration.training.framesamp_b1_data import (
    B1_DEMO_FRAMES,
    B1_LIVE_FRAMES,
    assemble_framesamp_b1_history,
    select_framesamp_b1,
)
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKENS_PER_FRAME,
    even_sampling_indices,
)


def _episode(steps: int) -> np.ndarray:
    return np.broadcast_to(
        np.arange(steps, dtype=np.float32)[:, None, None, None],
        (steps, 1, 64, 1),
    ).copy()


def test_initial_cut_freezes_uniform_demo_and_masks_live_partition():
    selection = select_framesamp_b1(step_idx=99, exec_start_idx=99)
    assert selection.demo_indices == tuple(int(value) for value in np.linspace(0, 99, B1_DEMO_FRAMES, dtype=np.int32))
    assert selection.live_indices == ()
    assert selection.demo_mask[:B1_DEMO_FRAMES].all()
    assert not selection.frame_mask[B1_DEMO_FRAMES:].any()
    assert selection.frame_indices[B1_DEMO_FRAMES:].tolist() == [-1] * B1_LIVE_FRAMES


def test_later_cut_keeps_demo_fixed_and_live_is_exact_recent_suffix():
    initial = select_framesamp_b1(step_idx=99, exec_start_idx=99)
    filled = select_framesamp_b1(step_idx=140, exec_start_idx=99)
    assert filled.demo_indices == initial.demo_indices
    assert filled.live_indices == tuple(range(125, 141))
    assert filled.frame_indices[:B1_DEMO_FRAMES].tolist() == list(initial.demo_indices)
    assert filled.frame_indices[B1_DEMO_FRAMES:].tolist() == list(range(125, 141))
    assert filled.frame_mask.all()


def test_live_partition_fills_left_to_right_before_sliding():
    first = select_framesamp_b1(step_idx=100, exec_start_idx=99)
    assert first.live_indices == (100,)
    assert first.frame_indices[B1_DEMO_FRAMES] == 100
    assert first.frame_indices[B1_DEMO_FRAMES + 1 :].tolist() == [-1] * 15

    partial = select_framesamp_b1(step_idx=105, exec_start_idx=99)
    assert partial.live_indices == tuple(range(100, 106))
    assert partial.live_mask[B1_DEMO_FRAMES : B1_DEMO_FRAMES + 6].all()
    assert not partial.live_mask[B1_DEMO_FRAMES + 6 :].any()


def test_b1_is_distinct_from_official_uniform_resampling_at_later_cut():
    selection = select_framesamp_b1(step_idx=140, exec_start_idx=99)
    official = even_sampling_indices(140)
    assert tuple(selection.frame_indices) != official
    assert selection.demo_indices == tuple(int(value) for value in np.linspace(0, 99, 16, dtype=np.int32))
    assert official[-1] == selection.live_indices[-1] == 140


def test_assembler_preserves_fixed_physical_slots_and_episode_global_features():
    history = assemble_framesamp_b1_history(_episode(160), 105, 99)
    history.validate()
    selection = history.selection
    frames = history.image.reshape(MAX_FRAMES, TOKENS_PER_FRAME, 1)[:, 0, 0]
    assert tuple(frames[:B1_DEMO_FRAMES].astype(int)) == selection.demo_indices
    assert tuple(frames[B1_DEMO_FRAMES : B1_DEMO_FRAMES + 6].astype(int)) == tuple(range(100, 106))
    assert np.count_nonzero(frames[B1_DEMO_FRAMES + 6 :]) == 0
    assert history.token_mask[: B1_DEMO_FRAMES * TOKENS_PER_FRAME].all()
    assert history.token_mask[B1_DEMO_FRAMES * TOKENS_PER_FRAME : (B1_DEMO_FRAMES + 6) * TOKENS_PER_FRAME].all()
    assert not history.token_mask[(B1_DEMO_FRAMES + 6) * TOKENS_PER_FRAME :].any()


def test_short_demo_keeps_live_in_second_physical_partition():
    history = assemble_framesamp_b1_history(_episode(20), 8, 3)
    history.validate()
    assert history.selection.demo_indices == (0, 1, 2, 3)
    assert history.selection.live_indices == (4, 5, 6, 7, 8)
    frames = history.image.reshape(MAX_FRAMES, TOKENS_PER_FRAME, 1)[:, 0, 0]
    assert frames[:4].tolist() == [0, 1, 2, 3]
    assert np.count_nonzero(frames[4:16]) == 0
    assert frames[16:21].tolist() == [4, 5, 6, 7, 8]


@pytest.mark.parametrize(
    ("step_idx", "exec_start_idx", "error"),
    [(-1, 0, "requires"), (4, 5, "requires"), (True, 0, "integer")],
)
def test_selection_rejects_invalid_causal_boundaries(step_idx, exec_start_idx, error):
    with pytest.raises((TypeError, ValueError), match=error):
        select_framesamp_b1(step_idx, exec_start_idx)


def test_assembler_never_reads_future_source_frames():
    source = _episode(40)
    history = assemble_framesamp_b1_history(source, 24, 9)
    assert history.image[history.token_mask].max() == 24
    with pytest.raises(IndexError, match="outside"):
        assemble_framesamp_b1_history(source, 40, 9)
