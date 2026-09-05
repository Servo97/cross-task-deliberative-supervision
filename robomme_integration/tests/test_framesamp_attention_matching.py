from __future__ import annotations

import numpy as np

from robomme_integration.training.attention_matching import fit_attention_matching
from robomme_integration.training.framesamp_attention_matching import (
    fit_framesamp_attention_matching,
)
from robomme_integration.training.upstream_framesamp_data import (
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    assemble_framesamp_history,
)


def _history(step_idx: int):
    image = np.zeros((step_idx + 1, 1, 64, 1), dtype=np.float32)
    return assemble_framesamp_history(image, step_idx)


def test_early_prefix_filters_physical_padding_and_preserves_logical_positions():
    rng = np.random.default_rng(9)
    history = _history(16)  # 17 frames x 16 patches = 272 real rows in a physical 512 rows.
    valid_count = 17 * TOKENS_PER_FRAME
    keys = rng.normal(size=(TOKEN_BUDGET, 7))
    values = rng.normal(size=(TOKEN_BUDGET, 5))
    # These extreme padded sentinels would dominate RMS selection if masking were forgotten.
    keys[valid_count:] = 1_000_000
    values[valid_count:] = -1_000_000
    queries = rng.normal(size=(4, 8, 7))

    result = fit_framesamp_attention_matching(history, queries, keys, values, 64, layer_index=3)
    direct, direct_metrics = fit_attention_matching(
        queries.reshape(-1, 7),
        keys[:valid_count],
        values[:valid_count],
        64,
    )
    result.validate(history)
    assert result.artifact.source_size == valid_count
    assert result.effective_target_size == 64
    assert np.array_equal(result.artifact.selected_indices, direct.selected_indices)
    assert np.array_equal(result.artifact.keys, direct.keys)
    assert np.allclose(result.artifact.beta_am, direct.beta_am)
    assert np.allclose(result.artifact.values, direct.values)
    assert result.metrics == direct_metrics
    assert np.all(result.selected_physical_indices < valid_count)
    assert np.array_equal(
        result.selected_step_indices,
        result.selected_physical_indices // TOKENS_PER_FRAME,
    )
    assert np.array_equal(
        result.selected_patch_indices,
        result.selected_physical_indices % TOKENS_PER_FRAME,
    )
    assert np.array_equal(result.action_query_positions(4), np.arange(512, 516))
    assert result.layer_index == 3 and result.query_head_count == 4


def test_requested_budget_larger_than_an_early_prefix_retains_every_real_token():
    rng = np.random.default_rng(12)
    history = _history(0)
    keys = rng.normal(size=(TOKEN_BUDGET, 4))
    values = rng.normal(size=(TOKEN_BUDGET, 3))
    result = fit_framesamp_attention_matching(
        history,
        rng.normal(size=(4, 3, 4)),
        keys,
        values,
        256,
        layer_index=0,
    )
    assert result.requested_target_size == 256
    assert result.artifact.source_size == TOKENS_PER_FRAME
    assert result.effective_target_size == TOKENS_PER_FRAME
    assert np.array_equal(result.artifact.selected_indices, np.arange(TOKENS_PER_FRAME))
    assert np.array_equal(result.artifact.beta_am, np.zeros(TOKENS_PER_FRAME))
    # Identity/compaction size never shifts the teacher action-query RoPE origin.
    assert result.query_position_offset == TOKEN_BUDGET
    assert result.action_query_positions(2).tolist() == [512, 513]
