"""Velocity metrics: relative, per flow time, float64 accumulation."""

from __future__ import annotations

import numpy as np
import pytest

from robomme_integration.amkv import metrics

FLOW_TIMES = (1.0, 0.9, 0.8, 0.7)
TOKENS = 5
DIM = 3


def _trace(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(len(FLOW_TIMES), TOKENS, DIM)).astype(np.float32)


def test_relative_error_is_zero_for_an_identical_trace():
    full = _trace(0)
    assert np.allclose(metrics.relative_velocity_error(full, full.copy()), 0.0)
    assert np.allclose(metrics.velocity_cosine(full, full.copy()), 1.0)


def test_relative_error_matches_a_hand_computed_scaling():
    full = np.ones((2, 1, 1), dtype=np.float32)
    compact = full * 1.25
    assert np.allclose(metrics.relative_velocity_error(full, compact), 0.25)
    assert np.allclose(metrics.velocity_cosine(full, compact), 1.0)


def test_error_is_reported_per_flow_time_not_pooled():
    full = _trace(0)
    compact = full.copy()
    compact[2] += 1.0
    errors = metrics.relative_velocity_error(full, compact)
    assert errors.shape == (len(FLOW_TIMES),)
    assert errors[2] > 0 and np.allclose(errors[[0, 1, 3]], 0.0)


def test_comparison_carries_its_labels_and_action_delta():
    full = _trace(0)
    compact = full * 1.01
    comparison = metrics.compare_velocity_traces(
        fixture_id="ep000001-t00063",
        arm_id="am8_f0",
        flow_times=FLOW_TIMES,
        full_velocity=full,
        compact_velocity=compact,
        full_actions=np.ones((TOKENS, DIM), dtype=np.float32),
        compact_actions=np.ones((TOKENS, DIM), dtype=np.float32) * 1.02,
    )
    payload = comparison.to_dict()
    assert payload["arm_id"] == "am8_f0"
    assert payload["fixture_id"] == "ep000001-t00063"
    assert len(payload["relative_velocity_error"]) == len(FLOW_TIMES)
    assert payload["relative_action_error"] == pytest.approx(0.02, rel=1e-5)
    assert payload["metric"] == metrics.VELOCITY_METRIC


def test_aggregate_reports_mean_p95_and_the_worst_flow_time():
    comparisons = []
    for index in range(4):
        full = _trace(index)
        compact = full.copy()
        compact[1] += 0.5 * (index + 1)
        comparisons.append(
            metrics.compare_velocity_traces(
                fixture_id=f"ep{index}",
                arm_id="am4_f0",
                flow_times=FLOW_TIMES,
                full_velocity=full,
                compact_velocity=compact,
                full_actions=np.ones((TOKENS, DIM), dtype=np.float32),
                compact_actions=np.ones((TOKENS, DIM), dtype=np.float32),
            )
        )
    summary = metrics.aggregate_comparisons(tuple(comparisons))
    assert summary["episode_count"] == 4
    assert summary["distinct_fixtures"] == 4
    assert summary["worst_flow_time"] == FLOW_TIMES[1]
    assert summary["relative_velocity_error_max"][1] >= summary["relative_velocity_error_mean"][1]
    assert summary["relative_velocity_error_mean"][0] == pytest.approx(0.0, abs=1e-12)


def test_mixed_arms_or_schedules_cannot_be_pooled():
    full = _trace(0)
    first = metrics.compare_velocity_traces(
        fixture_id="a",
        arm_id="am4_f0",
        flow_times=FLOW_TIMES,
        full_velocity=full,
        compact_velocity=full,
        full_actions=np.ones((2, 2), np.float32),
        compact_actions=np.ones((2, 2), np.float32),
    )
    second = metrics.compare_velocity_traces(
        fixture_id="b",
        arm_id="am8_f0",
        flow_times=FLOW_TIMES,
        full_velocity=full,
        compact_velocity=full,
        full_actions=np.ones((2, 2), np.float32),
        compact_actions=np.ones((2, 2), np.float32),
    )
    with pytest.raises(ValueError, match="one arm at a time"):
        metrics.aggregate_comparisons((first, second))


def test_non_finite_and_shape_drift_are_rejected():
    full = _trace(0)
    broken = full.copy()
    broken[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        metrics.relative_velocity_error(full, broken)
    with pytest.raises(ValueError, match="differ in shape"):
        metrics.relative_velocity_error(full, full[:, :-1])


def test_zero_velocity_denominator_is_refused_rather_than_reported_as_zero():
    full = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="relative error is undefined"):
        metrics.relative_velocity_error(full, full + 1.0)


def test_float32_inputs_are_accumulated_in_float64():
    tiny = np.full((1, 1, 3), 1e-4, dtype=np.float32)
    error = metrics.relative_velocity_error(tiny, (tiny * np.float32(1.001)).astype(np.float32))
    assert error.dtype == np.float64
    assert error[0] == pytest.approx(1e-3, rel=1e-3)
