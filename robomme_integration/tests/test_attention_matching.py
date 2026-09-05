from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from robomme_integration.training.attention_matching import (
    AttentionMatchingArtifact,
    attend_compact_old_and_recent,
    fit_attention_matching,
    fit_beta_am,
    rms_highest_attention_indices,
    scaled_dot_product_attention,
)


def test_full_size_fit_is_an_exact_identity_artifact():
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(19, 5))
    keys = rng.normal(size=(8, 5))
    values = rng.normal(size=(8, 3))
    artifact, metrics = fit_attention_matching(queries, keys, values, target_size=8)
    assert np.array_equal(artifact.selected_indices, np.arange(8))
    assert np.array_equal(artifact.keys, keys)
    assert np.array_equal(artifact.values, values)
    assert np.array_equal(artifact.beta_am, np.zeros(8))
    assert metrics.output_rmse < 1e-14
    assert metrics.log_mass_rmse < 1e-14
    assert len(artifact.sha256()) == 64
    assert artifact.sha256() == artifact.sha256()


def test_rms_selection_is_deterministic_and_returns_source_order():
    queries = np.array([[3.0, 0.0], [-2.0, 0.0], [1.0, 0.0]])
    keys = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    selected, scores = rms_highest_attention_indices(queries, keys, 2, scale=1.0)
    ranking = np.lexsort((np.arange(4), -scores))[:2]
    assert np.array_equal(selected, np.sort(ranking))
    assert np.all(np.diff(selected) > 0)


def test_beta_am_improves_output_and_mass_over_zero_bias():
    # Two retained representatives stand in for groups with unequal exact
    # multiplicities.  The correct beta_am values are log(6) and log(2).
    keys = np.concatenate(
        [np.repeat([[1.0, 0.0]], 6, axis=0), np.repeat([[-1.0, 0.0]], 2, axis=0)],
        axis=0,
    )
    values = np.concatenate(
        [np.repeat([[1.0, 0.0]], 6, axis=0), np.repeat([[0.0, 1.0]], 2, axis=0)],
        axis=0,
    )
    queries = np.stack([np.linspace(-2.5, 2.5, 31), np.zeros(31)], axis=1)
    fitted, fitted_metrics = fit_attention_matching(
        queries,
        keys,
        values,
        target_size=2,
        selected_indices=[0, 6],
        scale=1.0,
        fit_mass=True,
    )
    zero, zero_metrics = fit_attention_matching(
        queries,
        keys,
        values,
        target_size=2,
        selected_indices=[0, 6],
        scale=1.0,
        fit_mass=False,
    )
    assert np.allclose(fitted.beta_am, np.log([6.0, 2.0]), atol=1e-8)
    assert np.array_equal(zero.beta_am, np.zeros(2))
    assert fitted_metrics.output_rmse < zero_metrics.output_rmse
    assert fitted_metrics.log_mass_rmse < zero_metrics.log_mass_rmse
    assert fitted_metrics.output_rmse < 1e-10
    assert fitted_metrics.log_mass_rmse < 1e-10


def test_beta_am_matches_reference_unnormalized_mass_objective_not_relative_mass():
    """Regression for the paper/reference row-stabilized NNLS objective.

    Normalizing every row by its full denominator before NNLS silently
    reweights the queries and produces a different beta on this fixture.
    The unconstrained reference solution is strictly positive, so ordinary
    least squares and NNLS have the same unique optimum here.
    """

    full_logits = np.array([[0.0, 0.0], [0.0, np.log(9.0)]])
    compact_logits = np.zeros((2, 1))
    actual_weight = float(np.exp(fit_beta_am(full_logits, compact_logits)[0]))
    # Common-full-row-max objective: M=[1, 1/9], y=[2, 10/9].
    assert actual_weight == pytest.approx(86.0 / 41.0)
    # The rejected log-Z-normalized objective would yield 30/13 instead.
    assert actual_weight != pytest.approx(30.0 / 13.0)


def test_compact_old_and_recent_share_exactly_one_softmax_denominator():
    rng = np.random.default_rng(11)
    queries = rng.normal(size=(13, 4))
    old_keys = rng.normal(size=(7, 4))
    old_values = rng.normal(size=(7, 3))
    artifact, _ = fit_attention_matching(queries, old_keys, old_values, target_size=3)
    recent_keys = rng.normal(size=(2, 4))
    recent_values = rng.normal(size=(2, 3))
    actual_output, actual_log_mass = attend_compact_old_and_recent(
        queries,
        artifact,
        recent_keys,
        recent_values,
    )
    expected_output, expected_log_mass = scaled_dot_product_attention(
        queries,
        np.concatenate([artifact.keys, recent_keys]),
        np.concatenate([artifact.values, recent_values]),
        beta_am=np.concatenate([artifact.beta_am, np.zeros(2)]),
        scale=artifact.scale,
    )
    assert np.allclose(actual_output, expected_output, atol=1e-13)
    assert np.allclose(actual_log_mass, expected_log_mass, atol=1e-13)


def test_exact_old_compression_preserves_full_old_plus_recent_attention():
    old_keys = np.concatenate([np.repeat([[1.0, 0.0]], 5, axis=0), np.repeat([[-1.0, 0.0]], 3, axis=0)])
    old_values = np.concatenate([np.repeat([[2.0, -1.0]], 5, axis=0), np.repeat([[-0.5, 3.0]], 3, axis=0)])
    queries = np.stack([np.linspace(-3.0, 3.0, 41), np.linspace(1.0, -1.0, 41)], axis=1)
    compact_old, old_metrics = fit_attention_matching(
        queries,
        old_keys,
        old_values,
        target_size=2,
        selected_indices=[0, 5],
        scale=1.0,
    )
    assert old_metrics.output_rmse < 1e-10 and old_metrics.log_mass_rmse < 1e-10
    recent_keys = np.array([[0.0, 1.0], [0.5, -0.5]])
    recent_values = np.array([[4.0, 2.0], [-3.0, 0.25]])
    compact_output, compact_log_mass = attend_compact_old_and_recent(
        queries,
        compact_old,
        recent_keys,
        recent_values,
    )
    full_output, full_log_mass = scaled_dot_product_attention(
        queries,
        np.concatenate([old_keys, recent_keys]),
        np.concatenate([old_values, recent_values]),
        scale=1.0,
    )
    assert np.allclose(compact_output, full_output, atol=1e-9)
    assert np.allclose(compact_log_mass, full_log_mass, atol=1e-9)


def test_artifact_validation_fails_closed_on_schema_and_index_corruption():
    artifact = AttentionMatchingArtifact(
        source_size=4,
        selected_indices=np.array([0, 2]),
        keys=np.ones((2, 3)),
        values=np.ones((2, 5)),
        beta_am=np.zeros(2),
        scale=0.5,
    )
    artifact.validate(expected_source_size=4, expected_target_size=2)
    with pytest.raises(ValueError, match="unsupported"):
        dataclasses.replace(artifact, schema_version=999).validate()
    with pytest.raises(ValueError, match="positive integer"):
        dataclasses.replace(artifact, source_size=4.5).validate()
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(artifact, selected_indices=np.array([2, 2])).validate()
    with pytest.raises(ValueError, match="mismatch"):
        artifact.validate(expected_target_size=3)
