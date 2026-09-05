from __future__ import annotations

import numpy as np
import pytest

from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    assemble_framesamp_history,
    even_sampling_indices,
    pool_front_8x8_to_4x4,
    position_embedding_3d,
)


def _upstream_reference(step_idx: int) -> tuple[int, ...]:
    if step_idx < 32:
        return tuple(range(step_idx + 1))
    return tuple(np.linspace(0, step_idx, 32, dtype=np.int32).tolist())


@pytest.mark.parametrize("step_idx", [0, 1, 30, 31, 32, 33, 100, 10_000])
def test_even_sampling_indices_match_official_reference(step_idx):
    actual = even_sampling_indices(step_idx)
    assert actual == _upstream_reference(step_idx)
    assert actual[0] == 0 and actual[-1] == step_idx
    assert len(actual) == min(step_idx + 1, MAX_FRAMES)


def test_pooling_is_exact_nonoverlapping_mean_and_front_view_only():
    front = np.arange(64, dtype=np.float32).reshape(8, 8)
    wrist = np.full((8, 8), 1_000_000, dtype=np.float32)
    source = np.stack([front, wrist], axis=0).reshape(1, 2, 64, 1)
    pooled = pool_front_8x8_to_4x4(source)
    expected = front.reshape(4, 2, 4, 2).mean(axis=(1, 3)).reshape(1, 16, 1)
    assert pooled.dtype == np.float32
    assert np.array_equal(pooled, expected)
    assert pooled.max() < wrist.min()


def test_assembler_includes_demo_and_causal_history_then_right_pads():
    steps = 48
    demo_steps = 12
    image = np.empty((steps, 2, 64, 2), dtype=np.float32)
    for step in range(steps):
        image[step, 0] = step
        image[step, 1] = 10_000 + step

    current_step = 20
    result = assemble_framesamp_history(image, current_step)
    result.validate()
    valid = current_step + 1
    assert result.image.shape == (TOKEN_BUDGET, 2)
    assert result.position.shape == (TOKEN_BUDGET, 768)
    assert result.frame_indices[:valid].tolist() == list(range(valid))
    assert result.frame_indices[valid:].tolist() == [-1] * (MAX_FRAMES - valid)
    assert result.frame_indices[demo_steps - 1] < demo_steps
    assert result.frame_indices[demo_steps] == demo_steps
    assert result.frame_indices[valid - 1] == current_step
    assert result.token_mask.tolist() == [True] * (valid * TOKENS_PER_FRAME) + [False] * (
        TOKEN_BUDGET - valid * TOKENS_PER_FRAME
    )
    per_frame = result.image.reshape(MAX_FRAMES, TOKENS_PER_FRAME, 2)
    assert np.all(per_frame[:valid, :, 0] == np.arange(valid)[:, None])
    assert np.all(per_frame[:valid, :, 1] == np.arange(valid)[:, None])
    assert np.count_nonzero(per_frame[valid:]) == 0
    # Future steps carry distinct sentinels and must never appear.
    assert result.image[result.token_mask, 0].max() == current_step


def test_long_prefix_is_uniform_inclusive_and_never_reads_the_future():
    steps = 80
    image = np.broadcast_to(
        np.arange(steps, dtype=np.float32)[:, None, None, None],
        (steps, 1, 64, 1),
    ).copy()
    current_step = 52
    result = assemble_framesamp_history(image, current_step)
    expected = _upstream_reference(current_step)
    assert tuple(result.frame_indices) == expected
    assert result.frame_mask.all() and result.token_mask.all()
    frame_values = result.image.reshape(MAX_FRAMES, TOKENS_PER_FRAME, 1)[:, 0, 0]
    assert tuple(frame_values.astype(int)) == expected
    assert frame_values.max() == current_step < steps - 1


def test_compact_bfloat16_payload_is_decoded_exactly():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    values = np.arange(2, dtype=np.float32)[:, None, None, None]
    image = np.broadcast_to(values, (2, 1, 64, 1)).astype(ml_dtypes.bfloat16).view(np.uint16)
    result = assemble_framesamp_history(image, 1)
    assert result.image[0, 0] == 0
    assert result.image[TOKENS_PER_FRAME, 0] == 1
    direct = np.broadcast_to(values, (2, 1, 64, 1)).astype(ml_dtypes.bfloat16)
    assert np.array_equal(assemble_framesamp_history(direct, 1).image, result.image)


def test_bfloat16_pooling_rounds_before_the_float32_output_cast_like_upstream():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    source = np.zeros((1, 1, 8, 8, 1), dtype=np.float32)
    patch = np.array(
        [[0.5546875, 0.2177734375], [0.828125, 0.828125]],
        dtype=np.float32,
    ).astype(ml_dtypes.bfloat16)
    source[0, 0, :2, :2, 0] = patch
    direct = source.astype(ml_dtypes.bfloat16).reshape(1, 1, 64, 1)
    payload = direct.view(np.uint16)

    expected = float(np.asarray(patch.mean(), dtype=np.float32))
    predecoded_float32_mean = float(patch.astype(np.float32).mean())
    assert expected == 0.609375
    assert predecoded_float32_mean == 0.607177734375
    assert pool_front_8x8_to_4x4(direct)[0, 0, 0] == expected
    assert pool_front_8x8_to_4x4(payload)[0, 0, 0] == expected


def test_assembler_rejects_out_of_episode_steps():
    source = np.zeros((2, 1, 64, 1), dtype=np.float32)
    with pytest.raises(IndexError, match="outside"):
        assemble_framesamp_history(source, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        even_sampling_indices(-1)


def test_position_embedding_matches_upstream_temporal_and_spatial_formulas():
    steps = np.array([0, 7, 31], dtype=np.int32)
    actual = position_embedding_3d(steps)
    width = 768 // 6
    omega = np.arange(width, dtype=np.float32) / np.float32(width - 1)
    temporal_omega = np.float32(1.0) / np.power(np.float32(10_000), omega)
    spatial_omega = np.float32(1.0) / np.power(np.float32(1_000), omega)

    expected_temporal_angles = np.einsum("m,d->md", steps.astype(np.float32), temporal_omega)
    expected_temporal = np.concatenate([np.sin(expected_temporal_angles), np.cos(expected_temporal_angles)], axis=-1)
    y, x = np.mgrid[:4, :4]
    y = (4 * y + 2).reshape(-1).astype(np.float32)
    x = (4 * x + 2).reshape(-1).astype(np.float32)
    expected_spatial = np.concatenate(
        [
            np.sin(np.einsum("m,d->md", y, spatial_omega)),
            np.cos(np.einsum("m,d->md", y, spatial_omega)),
            np.sin(np.einsum("m,d->md", x, spatial_omega)),
            np.cos(np.einsum("m,d->md", x, spatial_omega)),
        ],
        axis=-1,
    )
    assert actual.shape == (3, 16, 768) and actual.dtype == np.float32
    assert np.allclose(actual[:, :, : 2 * width], expected_temporal[:, None, :], atol=2e-7)
    assert np.allclose(actual[:, :, 2 * width :], expected_spatial[None, :, :], atol=2e-7)
    # Upstream 4x4 positions are direct cell-center evaluations, not averages
    # of neighboring 8x8 sinusoidal positions.
    fine_centers = np.array([1, 3], dtype=np.float32)
    pooled_first_y_sin = np.sin(fine_centers[:, None] * spatial_omega).mean(axis=0)
    assert not np.allclose(actual[0, 0, 2 * width : 3 * width], pooled_first_y_sin)


def test_assembler_accepts_only_an_exact_4x4_position_table():
    image = np.zeros((3, 1, 64, 2), dtype=np.float32)
    exact = position_embedding_3d(np.arange(3, dtype=np.int32))
    result = assemble_framesamp_history(image, 2, exact_position_features=exact)
    assert np.array_equal(result.position[: 3 * TOKENS_PER_FRAME], exact.reshape(-1, 768))
    with pytest.raises(ValueError, match=r"\[steps, 16, D\]"):
        assemble_framesamp_history(
            image,
            2,
            exact_position_features=np.zeros((3, 1, 64, 768), dtype=np.float32),
        )
