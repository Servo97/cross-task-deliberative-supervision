"""Driver pieces that can be checked without the released checkpoint."""

from __future__ import annotations

import inspect
import types

import jax.numpy as jnp
import numpy as np
import pytest

from robomme_integration.amkv import driver
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    even_sampling_indices,
)

HEAD_DIM = 8
LAYERS = 2


def test_flow_schedule_matches_the_official_while_loop_accumulation():
    """``sample_actions`` carries a float32 time and adds dt each iteration."""

    for num_steps in (4, 10):
        dt = -1.0 / num_steps
        expected = []
        value = np.float32(1.0)
        for _ in range(num_steps):
            expected.append(float(value))
            value = np.float32(value + np.float32(dt))
        assert driver.flow_schedule(num_steps) == tuple(expected)
    schedule = driver.flow_schedule(10)
    assert schedule[0] == 1.0
    assert len(schedule) == 10
    assert schedule[-1] == pytest.approx(0.1, abs=1e-6)


def test_flow_schedule_rejects_a_degenerate_horizon():
    with pytest.raises(ValueError, match="num_steps must be positive"):
        driver.flow_schedule(0)


def test_identity_pack_serves_every_token_with_zero_mass_correction():
    keys = np.random.default_rng(0).normal(size=(LAYERS, TOKEN_BUDGET, HEAD_DIM)).astype(np.float32)
    values = np.random.default_rng(1).normal(size=(LAYERS, TOKEN_BUDGET, HEAD_DIM)).astype(np.float32)
    pack = driver.identity_am_pack(keys, values, dtype=jnp.float32)
    assert pack.compact_keys.shape == (LAYERS, 1, TOKEN_BUDGET, 1, HEAD_DIM)
    assert pack.recent_keys.shape == (LAYERS, 1, 0, 1, HEAD_DIM)
    assert pack.recent_token_mask.shape == (LAYERS, 1, 0)
    assert not np.asarray(pack.compact_beta_am).any()
    assert np.array_equal(np.asarray(pack.compact_keys)[:, 0, :, 0, :], keys)


def test_pack_from_arrays_preserves_the_scan_layout():
    arrays = {
        "compact_keys": np.zeros((LAYERS, 1, 4, 1, HEAD_DIM), np.float32),
        "compact_values": np.zeros((LAYERS, 1, 4, 1, HEAD_DIM), np.float32),
        "compact_beta_am": np.zeros((LAYERS, 1, 4), np.float32),
        "recent_keys": np.zeros((LAYERS, 1, 2, 1, HEAD_DIM), np.float32),
        "recent_values": np.zeros((LAYERS, 1, 2, 1, HEAD_DIM), np.float32),
        "recent_token_mask": np.ones((LAYERS, 1, 2), np.bool_),
        "selected_indices": np.arange(4),
    }
    pack = driver.am_pack_from_arrays(arrays, dtype=jnp.float32)
    assert pack.compact_beta_am.dtype == jnp.float32
    assert pack.recent_token_mask.dtype == jnp.bool_
    assert pack.recent_keys.shape == (LAYERS, 1, 2, 1, HEAD_DIM)


def _record(step_idx: int, memory_steps) -> types.SimpleNamespace:
    frames = len(memory_steps)
    return types.SimpleNamespace(
        fixture_id="ep000001-t00063",
        step_idx=step_idx,
        exec_start_idx=0,
        memory_step_indices=np.asarray(memory_steps, dtype=np.int64),
        memory_images=np.zeros((frames, 8, 8, 3), np.uint8),
        memory_states=np.zeros((frames, 8), np.float32),
        obs_image=np.zeros((8, 8, 3), np.uint8),
        obs_wrist_image=np.zeros((8, 8, 3), np.uint8),
        obs_state=np.zeros(8, np.float32),
        prompt="do the thing",
    )


def test_build_observation_refuses_frames_that_are_not_the_official_selection():
    wrong = list(even_sampling_indices(63))
    wrong[5] += 1
    with pytest.raises(ValueError, match="official FrameSamp selection"):
        driver.build_observation(object(), _record(63, wrong))


def test_build_observation_requires_every_fixture_field():
    record = _record(63, even_sampling_indices(63))
    del record.memory_states
    with pytest.raises(AttributeError, match="memory_states"):
        driver.build_observation(object(), record)


def test_framesamp_history_from_observation_recovers_masks_and_indices():
    steps = np.asarray(even_sampling_indices(63), dtype=np.int64)
    observation = types.SimpleNamespace(
        static_image_emb=np.zeros((1, TOKEN_BUDGET, 4), np.float32),
        static_pos_emb=np.zeros((1, TOKEN_BUDGET, 2), np.float32),
        static_mask=np.ones((1, TOKEN_BUDGET), np.bool_),
    )
    history = driver.framesamp_history_from_observation(observation, steps)
    assert history.frame_mask.all()
    assert np.array_equal(history.frame_indices, steps.astype(np.int32))
    assert history.token_mask.sum() == MAX_FRAMES * TOKENS_PER_FRAME


def test_short_prefix_history_is_right_padded():
    steps = np.asarray(even_sampling_indices(15), dtype=np.int64)
    mask = np.zeros((1, TOKEN_BUDGET), np.bool_)
    mask[0, : len(steps) * TOKENS_PER_FRAME] = True
    observation = types.SimpleNamespace(
        static_image_emb=np.ones((1, TOKEN_BUDGET, 4), np.float32),
        static_pos_emb=np.ones((1, TOKEN_BUDGET, 2), np.float32),
        static_mask=mask,
    )
    history = driver.framesamp_history_from_observation(observation, steps)
    assert int(history.frame_mask.sum()) == len(steps)
    assert (history.frame_indices[len(steps) :] == -1).all()
    assert not history.image[~history.token_mask].any()


def test_timed_reports_a_blocked_wall_clock():
    calls = []

    def work():
        calls.append(1)
        return jnp.ones((4,))

    report = driver.timed(work, warmup=1, repeats=2)
    assert len(calls) == 3
    assert report["repeats"] == 2
    assert report["seconds_min"] <= report["seconds_mean"] <= report["seconds_max"]


def test_timed_rejects_a_zero_repeat_measurement():
    with pytest.raises(ValueError, match="repeats positive"):
        driver.timed(lambda: jnp.ones(()), repeats=0)


def test_sample_noise_is_seeded_and_shaped_like_the_action_chunk():
    first = driver.sample_noise(3, batch=1, action_horizon=20, action_dim=8)
    again = driver.sample_noise(3, batch=1, action_horizon=20, action_dim=8)
    other = driver.sample_noise(4, batch=1, action_horizon=20, action_dim=8)
    assert first.shape == (1, 20, 8)
    assert np.array_equal(np.asarray(first), np.asarray(again))
    assert not np.array_equal(np.asarray(first), np.asarray(other))


def test_memory_branch_kwargs_omits_the_patch_only_keyword():
    """Regression: E0 attempt 2 died passing am_pack=None to the OFFICIAL module.

    The official history module has no ``am_pack`` parameter, so a denoiser that
    always passes it cannot wrap the unpatched baseline -- which is precisely
    the graph the identity gate must compare against.
    """

    assert driver.memory_branch_kwargs(None) == {}
    pack = driver.identity_am_pack(
        np.zeros((1, 4, HEAD_DIM), np.float32), np.zeros((1, 4, HEAD_DIM), np.float32), dtype=jnp.float32
    )
    assert driver.memory_branch_kwargs(pack) == {"am_pack": pack}


def test_driver_never_hardcodes_the_am_pack_keyword_in_the_llm_call():
    source = inspect.getsource(driver.Denoiser)
    assert "memory_branch_kwargs(am_pack)" in source
    assert "am_pack=am_pack" not in source
