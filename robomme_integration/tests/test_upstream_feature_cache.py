from __future__ import annotations

import io
import json
import zipfile

import ml_dtypes
import numpy as np
import pytest

from robomme_integration.training.upstream_feature_cache import (
    consolidate_episode,
    recurrent_history_indices,
    sha256_file,
)
from robomme_integration.training.upstream_ttt_data import DeterministicResumeSampler, assemble_recurrent_history
from robomme_integration.training.workspace_supervision_cache import chronological_events, grounded_patch_id


def _payload(step: int) -> bytes:
    buffer = io.BytesIO()
    np.save(
        buffer,
        {
            "image_emb_8x8": np.full((1, 64, 2048), step, dtype=ml_dtypes.bfloat16),
            "pos_emb_8x8": np.full((1, 64, 768), step, dtype=np.float32),
            "state_emb": np.arange(8, dtype=np.float64) + step,
        },
        allow_pickle=True,
    )
    return buffer.getvalue()


def _archive(path, episode: int, steps: int, *, gap: int | None = None):
    with zipfile.ZipFile(path, "w") as archive:
        for step in range(steps):
            if step != gap:
                archive.writestr(f"episode_{episode}/token_emb_{step}.npy", _payload(step))


def test_recurrent_indices_match_upstream_boundaries():
    assert recurrent_history_indices(7, 0) == (7,)
    assert recurrent_history_indices(32, 0) == (0, 16, 32)
    assert recurrent_history_indices(20, 12) == (0, 8, 20)
    assert recurrent_history_indices(80, 40) == (0, 16, 32, 48, 64, 80)
    result = recurrent_history_indices(2000, 1000)
    assert len(result) <= 64 and result[-1] <= 2000
    with pytest.raises(ValueError, match="at/after"):
        recurrent_history_indices(5, 6)


def test_deterministic_sampler_resumes_at_exact_optimizer_batch():
    uninterrupted = DeterministicResumeSampler(11, 4, seed=7)
    epoch_zero = list(iter(uninterrupted))
    epoch_one = list(iter(uninterrupted))
    assert len(epoch_zero) == len(epoch_one) == 8

    # Two batches consume epoch zero; optimizer step three is epoch one's second batch.
    resumed = DeterministicResumeSampler(11, 4, seed=7, start_step=3)
    assert list(iter(resumed)) == epoch_one[4:]
    assert list(iter(resumed)) == list(iter(uninterrupted))


def test_consolidate_episode_is_compact_exact_and_resumable(tmp_path):
    archive = tmp_path / "episode_7.zip"
    _archive(archive, 7, 3)
    positions: dict[int, np.ndarray] = {}
    first = consolidate_episode(archive, tmp_path / "compact", 7, positions)
    image_path = tmp_path / "compact" / "episode_7" / "image_bf16_bits.npy"
    state_path = tmp_path / "compact" / "episode_7" / "state_f64.npy"
    image = np.load(image_path, mmap_mode="r")
    state = np.load(state_path, mmap_mode="r")
    assert image.shape == (3, 1, 64, 2048) and image.dtype == np.uint16
    assert state.shape == (3, 8) and state.dtype == np.float64
    assert image[2].view(ml_dtypes.bfloat16).astype(np.float32).mean() == 2
    assert np.array_equal(positions[2], np.full((1, 64, 768), 2, dtype=np.float32))
    assert first["image_sha256"] == sha256_file(image_path)
    assert json.loads((image_path.parent / "COMPLETE.json").read_text()) == first

    # A completed, hash-matching episode is reused without rewriting it.
    assert consolidate_episode(archive, tmp_path / "compact", 7, {}) == first


def test_consolidate_episode_fails_closed_on_gap(tmp_path):
    archive = tmp_path / "episode_8.zip"
    _archive(archive, 8, 3, gap=1)
    with pytest.raises(ValueError, match="not contiguous"):
        consolidate_episode(archive, tmp_path / "compact", 8, {})


def test_assemble_recurrent_history_exact_bfloat_bits_and_left_padding():
    image = np.stack([np.full((1, 64, 2048), value, dtype=ml_dtypes.bfloat16).view(np.uint16) for value in range(4)])
    position = np.stack([np.full((1, 64, 768), value, dtype=np.float32) for value in range(4)])
    state = np.stack([np.arange(8, dtype=np.float64) + value for value in range(4)])
    out_image, out_position, out_state, mask = assemble_recurrent_history(
        image,
        position,
        state,
        (1, 3),
        max_recur_steps=4,
    )
    assert out_image.dtype == out_position.dtype == out_state.dtype == np.float32
    assert out_image.shape == (4, 1, 64, 2048)
    assert mask.tolist() == [False, False, True, True]
    assert out_image[-2].mean() == 1 and out_image[-1].mean() == 3
    assert out_position[-1].mean() == 3
    assert np.array_equal(out_state[-2], np.arange(8, dtype=np.float32) + 1)


def test_grounded_points_map_to_8x8_patches_and_fail_closed():
    assert grounded_patch_id("pick at <0, 0>") == 0
    assert grounded_patch_id("pick at <255, 255>") == 63
    assert grounded_patch_id("pick at <89, 122>") == 26
    with pytest.raises(ValueError, match="exactly one point"):
        grounded_patch_id("no point")
    with pytest.raises(ValueError, match="outside"):
        grounded_patch_id("pick at <256, 10>")


def test_chronological_event_targets_are_anchored_once_per_segment():
    simple = ["pick one", "pick one", "put down", "put down", "pick two"]
    grounded = [
        "pick at <10, 20>",
        "pick at <10, 20>",
        "put down",
        "put down",
        "pick at <200, 220>",
    ]
    events = chronological_events(simple, grounded)
    assert [event["anchor_step"] for event in events] == [0, 4]
    assert [event["patch_id"] for event in events] == [0, 54]
    assert all(len(event["simple_subgoal_sha256"]) == 64 for event in events)
