"""Compaction plans, fits, and packs on real FrameSamp geometry."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from robomme_integration.amkv import compaction
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    FrameSampHistory,
)

HEAD_DIM = 16
LAYERS = 2


def _history(frames: int = MAX_FRAMES) -> FrameSampHistory:
    generator = np.random.default_rng(0)
    frame_mask = np.zeros(MAX_FRAMES, dtype=np.bool_)
    frame_mask[:frames] = True
    token_mask = np.repeat(frame_mask, TOKENS_PER_FRAME)
    frame_indices = np.full(MAX_FRAMES, -1, dtype=np.int32)
    frame_indices[:frames] = np.linspace(0, 100, frames, dtype=np.int32)
    image = generator.normal(size=(TOKEN_BUDGET, 8)).astype(np.float32) * token_mask[:, None]
    position = generator.normal(size=(TOKEN_BUDGET, 4)).astype(np.float32) * token_mask[:, None]
    history = FrameSampHistory(
        image=image,
        position=position,
        token_mask=token_mask,
        frame_indices=frame_indices,
        frame_mask=frame_mask,
    )
    history.validate()
    return history


def _taps(seed: int = 1):
    generator = np.random.default_rng(seed)
    keys = generator.normal(size=(LAYERS, TOKEN_BUDGET, HEAD_DIM)).astype(np.float32)
    values = generator.normal(size=(LAYERS, TOKEN_BUDGET, HEAD_DIM)).astype(np.float32)
    return keys, values


def _queries(seed: int, samples: int = 12):
    generator = np.random.default_rng(seed)
    return generator.normal(size=(LAYERS, 4, samples, HEAD_DIM)).astype(np.float32)


def test_plan_splits_the_budget_between_compact_and_exact_recent():
    history = _history()
    plain = compaction.plan_compaction(history, 8.0)
    assert (plain.valid_tokens, plain.served_tokens, plain.compact_size, plain.recent_size) == (512, 64, 64, 0)
    kept = compaction.plan_compaction(history, 8.0, exact_recent_frames=1)
    assert (kept.served_tokens, kept.compact_size, kept.recent_size) == (64, 48, 16)
    assert kept.effective_ratio == pytest.approx(8.0)
    assert plain.label()["method"] == compaction.COMPACTION_METHOD


def test_plan_rejects_budgets_that_cannot_compress():
    history = _history()
    with pytest.raises(ValueError, match="ratio must exceed 1"):
        compaction.plan_compaction(history, 1.0)
    with pytest.raises(ValueError, match="no room for compact tokens"):
        compaction.plan_compaction(history, 8.0, exact_recent_frames=4)


def test_plan_follows_a_short_prefix_rather_than_the_physical_buffer():
    plan = compaction.plan_compaction(_history(frames=16), 4.0)
    assert (plan.valid_tokens, plan.served_tokens) == (256, 64)


def test_old_history_view_masks_exactly_the_recent_frames():
    history = _history()
    plan = compaction.plan_compaction(history, 8.0, exact_recent_frames=2)
    view = compaction.old_history_view(history, plan)
    assert int(view.frame_mask.sum()) == MAX_FRAMES - 2
    assert view.token_mask.sum() == plan.valid_tokens - plan.recent_size
    assert np.array_equal(view.image[view.token_mask], history.image[view.token_mask])
    assert not view.image[~view.token_mask].any()
    assert compaction.recent_token_slice(plan) == slice(480, 512)


def test_fit_scores_on_held_out_queries_and_keeps_framesamp_provenance():
    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 8.0)
    fit = compaction.fit_layer_compaction(
        history,
        plan,
        layer_index=0,
        fit_queries=_queries(2)[0],
        heldout_queries=_queries(3)[0],
        keys_post_rope=keys[0],
        values_post_projection=values[0],
    )
    assert fit.artifact.target_size == plan.compact_size
    assert fit.artifact.source_size == plan.valid_tokens
    assert 0.0 <= fit.fit_metrics.output_relative_l2 < 1.0
    assert np.isfinite(fit.heldout_output_relative_l2)
    # Held-out error may not beat the fit error; that gap is the reported signal.
    assert fit.heldout_output_relative_l2 >= 0.0
    label = fit.label()
    assert len(label["selected_step_indices"]) == plan.compact_size
    assert label["artifact_sha256"]


def test_exact_recent_block_is_scored_under_one_denominator():
    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 8.0, exact_recent_frames=1)
    fit = compaction.fit_layer_compaction(
        history,
        plan,
        layer_index=3,
        fit_queries=_queries(2)[0],
        heldout_queries=_queries(3)[0],
        keys_post_rope=keys[0],
        values_post_projection=values[0],
    )
    assert fit.artifact.target_size == plan.compact_size
    # The artifact only ever indexes the compactible prefix.
    assert fit.result.selected_physical_indices.max() < plan.valid_tokens - plan.recent_size


def test_stacked_pack_has_the_scan_layout_and_one_cast():
    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 4.0, exact_recent_frames=1)
    fits = compaction.fit_all_layers(
        history,
        plan,
        fit_queries=_queries(2),
        heldout_queries=_queries(3),
        keys_post_rope=keys,
        values_post_projection=values,
    )
    assert len(fits) == LAYERS
    arrays = compaction.stack_am_pack_arrays(
        fits, plan, keys_post_rope=keys, values_post_projection=values, runtime_dtype=np.float32
    )
    assert arrays["compact_keys"].shape == (LAYERS, 1, plan.compact_size, 1, HEAD_DIM)
    assert arrays["recent_keys"].shape == (LAYERS, 1, plan.recent_size, 1, HEAD_DIM)
    assert arrays["compact_beta_am"].shape == (LAYERS, 1, plan.compact_size)
    assert arrays["compact_beta_am"].dtype == np.float32
    assert arrays["recent_token_mask"].all()
    assert np.array_equal(arrays["recent_keys"][:, 0, :, 0, :], keys[:, compaction.recent_token_slice(plan), :])


def test_random_subset_control_needs_no_fit_and_shares_one_draw():
    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 8.0)
    arrays = compaction.random_subset_pack_arrays(
        plan, keys_post_rope=keys, values_post_projection=values, seed=0, runtime_dtype=np.float32
    )
    chosen = arrays["selected_indices"]
    assert chosen.size == plan.compact_size
    assert np.unique(chosen).size == chosen.size
    assert not arrays["compact_beta_am"].any()
    for layer in range(LAYERS):
        assert np.array_equal(arrays["compact_keys"][layer, 0, :, 0, :], keys[layer, chosen, :])


def test_destroyed_control_keeps_keys_and_zeroes_values():
    keys, values = _taps()
    arrays = compaction.destroyed_pack_arrays(
        keys_post_rope=keys, values_post_projection=values, runtime_dtype=np.float32
    )
    assert arrays["compact_keys"].shape == (LAYERS, 1, TOKEN_BUDGET, 1, HEAD_DIM)
    assert not arrays["compact_values"].any()
    assert arrays["recent_keys"].shape[2] == 0


def test_kv_bytes_report_the_served_reduction():
    plan = compaction.plan_compaction(_history(), 8.0)
    report = compaction.served_kv_bytes(plan, layers=18, head_dim=256, itemsize=2)
    assert report["full_kv_bytes"] == 18 * 512 * 2 * 256 * 2
    assert report["kv_bytes_ratio"] == pytest.approx(8.0)


def test_layer_count_mismatch_is_rejected():
    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 8.0)
    with pytest.raises(ValueError, match="different layer counts"):
        compaction.fit_all_layers(
            history,
            plan,
            fit_queries=_queries(2)[:1],
            heldout_queries=_queries(3),
            keys_post_rope=keys,
            values_post_projection=values,
        )


def test_payload_quantization_is_reported_as_its_own_number():
    """Gate 3: the served cast must be isolated, not folded into the AM error."""

    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 8.0)
    fits = compaction.fit_all_layers(
        history,
        plan,
        fit_queries=_queries(2),
        heldout_queries=_queries(3),
        keys_post_rope=keys,
        values_post_projection=values,
    )
    report = compaction.payload_quantization_parity(
        fits,
        plan,
        heldout_queries=_queries(3),
        keys_post_rope=keys,
        values_post_projection=values,
        runtime_dtype=np.dtype(ml_dtypes.bfloat16),
    )
    assert report["payload_dtype"] == "bfloat16"
    assert len(report["layers"]) == LAYERS
    assert report["quantization_only_relative_l2_mean"] > 0.0
    # bfloat16 carries 8 mantissa bits: the cast alone must stay near its ULP.
    assert report["quantization_only_relative_l2_max"] < 0.05
    for row in report["layers"]:
        assert row["heldout_relative_l2_quantized"] >= 0.0
        assert np.isfinite(row["heldout_relative_l2_float64"])


def test_float32_payload_quantization_is_smaller_than_bfloat16():
    history = _history()
    keys, values = _taps()
    plan = compaction.plan_compaction(history, 8.0)
    fits = compaction.fit_all_layers(
        history,
        plan,
        fit_queries=_queries(2),
        heldout_queries=_queries(3),
        keys_post_rope=keys,
        values_post_projection=values,
    )
    common = {
        "heldout_queries": _queries(3),
        "keys_post_rope": keys,
        "values_post_projection": values,
    }
    wide = compaction.payload_quantization_parity(fits, plan, runtime_dtype=np.float32, **common)
    narrow = compaction.payload_quantization_parity(fits, plan, runtime_dtype=np.dtype(ml_dtypes.bfloat16), **common)
    assert wide["quantization_only_relative_l2_mean"] < narrow["quantization_only_relative_l2_mean"]
