"""Framework-free attention matching for compact RoboMME memories.

This module implements the post-hoc teacher-compression stage used by the
WSM-AM/GDN-v2 recipe.  It deliberately depends only on NumPy so artifacts can
be fitted and audited without importing either the JAX policy or Torch-based
RoboMME evaluator.

For frozen teacher queries ``Q``, memory keys ``K`` and values ``V`` it:

1. retains the keys with highest RMS normalized attention over the queries;
2. fits a per-key log mass correction ``beta_am`` to preserve the teacher
   softmax denominator; and
3. fits replacement values by ordinary least squares to preserve the teacher
   attention output.

``beta_am`` is always named explicitly.  It is an attention-mass correction,
not the Gated DeltaNet write-rate beta and not a learned workspace token.
At inference, compressed old memory and exact recent tokens must be concatenated
*before* one softmax; :func:`attend_compact_old_and_recent` is the reference
implementation of that same-denominator contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_METHOD = "rms_keys_beta_am_ols_values_v1"
MASS_SOLVER = "projected_fista_nnls_common_full_row_max_v1"
MASS_SOLVER_DISABLED = "disabled_zero_beta_am"
VALUE_SOLVER = "numpy_lstsq_normalized_attention_v1"


def _as_finite_matrix(value: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{label} must be a nonempty rank-2 matrix, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _resolve_scale(scale: float | None, key_dim: int) -> float:
    resolved = 1.0 / math.sqrt(key_dim) if scale is None else float(scale)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"attention scale must be finite and positive, got {scale}")
    return resolved


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))


def attention_logits(
    queries: np.ndarray,
    keys: np.ndarray,
    *,
    scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return float64 scaled dot-product logits and the resolved scale."""

    queries = _as_finite_matrix(queries, label="queries")
    keys = _as_finite_matrix(keys, label="keys")
    if queries.shape[1] != keys.shape[1]:
        raise ValueError(f"query/key dimensions differ: {queries.shape[1]} != {keys.shape[1]}")
    resolved_scale = _resolve_scale(scale, keys.shape[1])
    logits = (queries @ keys.T) * resolved_scale
    if not np.isfinite(logits).all():
        raise ValueError("scaled attention logits are non-finite")
    return logits, resolved_scale


def attention_from_logits(
    logits: np.ndarray,
    values: np.ndarray,
    *,
    beta_am: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attention output and log denominator from precomputed logits."""

    logits = _as_finite_matrix(logits, label="logits")
    values = _as_finite_matrix(values, label="values")
    if logits.shape[1] != values.shape[0]:
        raise ValueError(f"logit key count and value count differ: {logits.shape[1]} != {values.shape[0]}")
    if beta_am is not None:
        beta_am = np.asarray(beta_am, dtype=np.float64)
        if beta_am.shape != (logits.shape[1],) or not np.isfinite(beta_am).all():
            raise ValueError(f"beta_am must be finite with shape {(logits.shape[1],)}, got {beta_am.shape}")
        logits = logits + beta_am[None, :]
    log_mass = _logsumexp(logits)
    probabilities = np.exp(logits - log_mass[:, None])
    output = probabilities @ values
    return output, log_mass


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    beta_am: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attention output and log denominator with stable normalization."""

    logits, _ = attention_logits(queries, keys, scale=scale)
    return attention_from_logits(logits, values, beta_am=beta_am)


def rms_highest_attention_indices(
    queries: np.ndarray,
    keys: np.ndarray,
    target_size: int,
    *,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the keys with highest RMS normalized teacher attention.

    Returned indices are chronological/source ordered rather than score ordered;
    the selected set is identical, while stable ordering makes artifacts easier
    to inspect and compose with recent tokens.  Equal scores break ties by the
    lower source index.
    """

    logits, _ = attention_logits(queries, keys, scale=scale)
    if isinstance(target_size, bool) or not isinstance(target_size, (int, np.integer)):
        raise TypeError("target_size must be an integer")
    target_size = int(target_size)
    if not 1 <= target_size <= logits.shape[1]:
        raise ValueError(f"target_size must be in [1, {logits.shape[1]}], got {target_size}")
    log_mass = _logsumexp(logits)
    probabilities = np.exp(logits - log_mass[:, None])
    scores = np.sqrt(np.mean(np.square(probabilities), axis=0))
    ranking = np.lexsort((np.arange(scores.size), -scores))
    selected = np.sort(ranking[:target_size]).astype(np.int64, copy=False)
    return selected, scores


def _nonnegative_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float = 0.0,
    max_iterations: int = 10_000,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Small dependency-free NNLS solver using projected accelerated descent.

    Work is performed on the Gram matrix and every iteration is one vectorized
    matrix-vector product.  The optional ridge is centered on one, which is the
    no-correction multiplicative mass prior.
    """

    design = _as_finite_matrix(design, label="mass design")
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (design.shape[0],) or not np.isfinite(target).all():
        raise ValueError(f"mass target must be finite with shape {(design.shape[0],)}")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")

    gram = design.T @ design
    linear = design.T @ target
    if ridge:
        gram.flat[:: gram.shape[0] + 1] += ridge
        linear += ridge
    lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 0.0)
    if lipschitz <= np.finfo(np.float64).tiny:
        return np.zeros(design.shape[1], dtype=np.float64)
    # Starting at beta_am=0 gives every selected key unit multiplicity.  FISTA
    # avoids a Python loop over up to 256 compact keys per iteration.
    weights = np.ones(design.shape[1], dtype=np.float64)
    extrapolated = weights.copy()
    acceleration = 1.0
    for iteration in range(max_iterations):
        candidate = np.maximum(
            0.0,
            extrapolated - (gram @ extrapolated - linear) / lipschitz,
        )
        # Check the projected-gradient mapping periodically.  Unlike the raw
        # accelerated step delta, it is a valid stationarity test under FISTA's
        # momentum and does not spuriously reject an already-optimal solution.
        if iteration % 25 == 0:
            projected = np.maximum(
                0.0,
                candidate - (gram @ candidate - linear) / lipschitz,
            )
            if np.max(np.abs(projected - candidate)) <= tolerance * (1.0 + float(np.max(candidate))):
                weights = candidate
                break
        next_acceleration = (1.0 + math.sqrt(1.0 + 4.0 * acceleration**2)) / 2.0
        extrapolated = candidate + ((acceleration - 1.0) / next_acceleration) * (candidate - weights)
        weights = candidate
        acceleration = next_acceleration
    else:
        # The iterate remains a valid nonnegative approximation; downstream
        # output/mass metrics make any poor fit explicit rather than hiding it.
        weights = candidate
    if not np.isfinite(weights).all():
        raise FloatingPointError("NNLS mass fit produced non-finite weights")
    if np.any(weights < 0):  # pragma: no cover - guarded by every projection.
        raise RuntimeError("NNLS mass fit violated nonnegativity")
    return weights


def fit_beta_am(
    full_logits: np.ndarray,
    compact_logits: np.ndarray,
    *,
    ridge: float = 0.0,
    minimum_weight: float = 1e-12,
) -> np.ndarray:
    """Fit per-key log multiplicities that preserve the full softmax mass.

    This follows the paper/reference implementation's row-stabilized
    unnormalized-mass NNLS objective.  The same maximum over each *full* logit
    row is subtracted from both sides before solving::

        exp(L_compact - row_max) @ exp(beta)
            ~= sum(exp(L_full - row_max), axis=-1)

    Dividing each row by its full denominator and regressing toward one is a
    different, relative-mass-weighted objective and must not be substituted
    here.  The shared row maximum changes neither row's exact solution nor the
    resulting ``beta_am`` interpretation, while avoiding exponent overflow.
    """

    full_logits = _as_finite_matrix(full_logits, label="full_logits")
    compact_logits = _as_finite_matrix(compact_logits, label="compact_logits")
    if full_logits.shape[0] != compact_logits.shape[0]:
        raise ValueError("full and compact logits must describe the same queries")
    minimum_weight = float(minimum_weight)
    if not math.isfinite(minimum_weight) or minimum_weight <= 0:
        raise ValueError("minimum_weight must be finite and positive")
    row_max = full_logits.max(axis=1, keepdims=True)
    full_exp = np.exp(full_logits - row_max)
    design = np.exp(compact_logits - row_max)
    target = full_exp.sum(axis=1)
    weights = _nonnegative_least_squares(
        design,
        target,
        ridge=ridge,
    )
    return np.log(np.maximum(weights, minimum_weight))


def fit_ols_values(
    compact_logits: np.ndarray,
    beta_am: np.ndarray,
    teacher_output: np.ndarray,
    *,
    ridge: float = 0.0,
) -> np.ndarray:
    """Fit compact values to the teacher's normalized attention output."""

    compact_logits = _as_finite_matrix(compact_logits, label="compact_logits")
    teacher_output = _as_finite_matrix(teacher_output, label="teacher_output")
    if compact_logits.shape[0] != teacher_output.shape[0]:
        raise ValueError("compact logits and teacher output must describe the same queries")
    beta_am = np.asarray(beta_am, dtype=np.float64)
    if beta_am.shape != (compact_logits.shape[1],) or not np.isfinite(beta_am).all():
        raise ValueError(f"beta_am must be finite with shape {(compact_logits.shape[1],)}, got {beta_am.shape}")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    adjusted = compact_logits + beta_am[None, :]
    log_mass = _logsumexp(adjusted)
    probabilities = np.exp(adjusted - log_mass[:, None])
    if ridge:
        count = probabilities.shape[1]
        design = np.concatenate(
            [probabilities, math.sqrt(ridge) * np.eye(count, dtype=np.float64)],
            axis=0,
        )
        target = np.concatenate(
            [teacher_output, np.zeros((count, teacher_output.shape[1]), dtype=np.float64)],
            axis=0,
        )
    else:
        design = probabilities
        target = teacher_output
    values, *_ = np.linalg.lstsq(design, target, rcond=None)
    return values


@dataclasses.dataclass(frozen=True)
class AttentionMatchingMetrics:
    """Numerically comparable teacher-versus-artifact errors."""

    output_rmse: float
    output_relative_l2: float
    log_mass_rmse: float
    relative_mass_rmse: float

    def validate(self) -> None:
        values = dataclasses.astuple(self)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"attention-matching metrics must be finite and nonnegative: {values}")


@dataclasses.dataclass(frozen=True)
class AttentionMatchingArtifact:
    """In-memory float64 result of post-hoc attention matching.

    ``sha256()`` covers the numeric payload and local schema only; it is not a
    scientific identity.  A producer must wrap this object in a manifest that
    seals teacher/code/query/tap/mask/logical-position provenance before an
    artifact is staged or consumed.  Runtime serialization to float32/native
    dtype also requires held-out attention-error parity against this fit result.
    """

    source_size: int
    selected_indices: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    beta_am: np.ndarray
    scale: float
    method: str = ARTIFACT_METHOD
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @property
    def target_size(self) -> int:
        return int(self.keys.shape[0])

    def validate(
        self,
        *,
        expected_source_size: int | None = None,
        expected_target_size: int | None = None,
        expected_key_dim: int | None = None,
        expected_value_dim: int | None = None,
    ) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION or self.method != ARTIFACT_METHOD:
            raise ValueError(f"unsupported attention-matching artifact {self.method!r} v{self.schema_version}")
        if (
            isinstance(self.source_size, bool)
            or not isinstance(self.source_size, (int, np.integer))
            or int(self.source_size) <= 0
        ):
            raise ValueError("source_size must be a positive integer")
        selected = np.asarray(self.selected_indices)
        keys = np.asarray(self.keys)
        values = np.asarray(self.values)
        beta_am = np.asarray(self.beta_am)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("selected_indices must be a rank-1 integer array")
        count = selected.size
        if not count or keys.ndim != 2 or values.ndim != 2:
            raise ValueError("artifact keys/values must be nonempty rank-2 matrices")
        if keys.shape[0] != count or values.shape[0] != count or beta_am.shape != (count,):
            raise ValueError("selected_indices, keys, values, and beta_am counts disagree")
        if count > int(self.source_size):
            raise ValueError("artifact target size exceeds its source size")
        if np.any(selected < 0) or np.any(selected >= int(self.source_size)):
            raise ValueError("selected_indices fall outside the source memory")
        if count > 1 and np.any(np.diff(selected.astype(np.int64, copy=False)) <= 0):
            raise ValueError("selected_indices must be unique and source ordered")
        if not np.isfinite(keys).all() or not np.isfinite(values).all() or not np.isfinite(beta_am).all():
            raise ValueError("artifact arrays contain non-finite values")
        _resolve_scale(self.scale, keys.shape[1])
        expected = {
            "source size": (expected_source_size, int(self.source_size)),
            "target size": (expected_target_size, count),
            "key dimension": (expected_key_dim, keys.shape[1]),
            "value dimension": (expected_value_dim, values.shape[1]),
        }
        for label, (wanted, actual) in expected.items():
            if wanted is not None and int(wanted) != actual:
                raise ValueError(f"artifact {label} mismatch: expected {wanted}, got {actual}")

    def sha256(self) -> str:
        """Return a stable checksum over schema, metadata, and exact array bytes."""

        self.validate()
        digest = hashlib.sha256()
        metadata = {
            "method": self.method,
            "scale": float(self.scale),
            "schema_version": int(self.schema_version),
            "source_size": int(self.source_size),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for value in (self.selected_indices, self.keys, self.values, self.beta_am):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()


def _validated_selected_indices(
    selected_indices: Sequence[int] | np.ndarray,
    *,
    source_size: int,
    target_size: int,
) -> np.ndarray:
    selected = np.asarray(selected_indices)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected_indices must be a rank-1 integer sequence")
    selected = selected.astype(np.int64, copy=False)
    if selected.size != target_size:
        raise ValueError(f"selected_indices has {selected.size} entries, expected target_size={target_size}")
    if np.any(selected < 0) or np.any(selected >= source_size):
        raise ValueError("selected_indices fall outside the source memory")
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_indices contain duplicates")
    return np.sort(selected)


def evaluate_attention_matching(
    queries: np.ndarray,
    full_keys: np.ndarray,
    full_values: np.ndarray,
    artifact: AttentionMatchingArtifact,
) -> AttentionMatchingMetrics:
    """Measure output and denominator preservation against the full teacher."""

    artifact.validate(
        expected_source_size=np.asarray(full_keys).shape[0],
        expected_key_dim=np.asarray(full_keys).shape[1],
        expected_value_dim=np.asarray(full_values).shape[1],
    )
    teacher_output, teacher_log_mass = scaled_dot_product_attention(
        queries,
        full_keys,
        full_values,
        scale=artifact.scale,
    )
    compact_output, compact_log_mass = scaled_dot_product_attention(
        queries,
        artifact.keys,
        artifact.values,
        beta_am=artifact.beta_am,
        scale=artifact.scale,
    )
    difference = compact_output - teacher_output
    output_rmse = float(np.sqrt(np.mean(np.square(difference))))
    denominator = max(float(np.linalg.norm(teacher_output)), np.finfo(np.float64).tiny)
    output_relative_l2 = float(np.linalg.norm(difference) / denominator)
    log_mass_difference = compact_log_mass - teacher_log_mass
    metrics = AttentionMatchingMetrics(
        output_rmse=output_rmse,
        output_relative_l2=output_relative_l2,
        log_mass_rmse=float(np.sqrt(np.mean(np.square(log_mass_difference)))),
        relative_mass_rmse=float(np.sqrt(np.mean(np.square(np.expm1(log_mass_difference))))),
    )
    metrics.validate()
    return metrics


def fit_attention_matching(
    queries: np.ndarray,
    full_keys: np.ndarray,
    full_values: np.ndarray,
    target_size: int,
    *,
    selected_indices: Sequence[int] | np.ndarray | None = None,
    scale: float | None = None,
    fit_mass: bool = True,
    mass_ridge: float = 0.0,
    value_ridge: float = 0.0,
) -> tuple[AttentionMatchingArtifact, AttentionMatchingMetrics]:
    """Fit and validate one compact attention-matching artifact."""

    queries = _as_finite_matrix(queries, label="queries")
    full_keys = _as_finite_matrix(full_keys, label="full_keys")
    full_values = _as_finite_matrix(full_values, label="full_values")
    if queries.shape[1] != full_keys.shape[1]:
        raise ValueError("query and key dimensions differ")
    if full_keys.shape[0] != full_values.shape[0]:
        raise ValueError("full key and value counts differ")
    if isinstance(target_size, bool) or not isinstance(target_size, (int, np.integer)):
        raise TypeError("target_size must be an integer")
    target_size = int(target_size)
    source_size = full_keys.shape[0]
    if not 1 <= target_size <= source_size:
        raise ValueError(f"target_size must be in [1, {source_size}], got {target_size}")
    full_logits, resolved_scale = attention_logits(queries, full_keys, scale=scale)

    if selected_indices is None:
        if target_size == source_size:
            selected = np.arange(source_size, dtype=np.int64)
        else:
            selected, _ = rms_highest_attention_indices(
                queries,
                full_keys,
                target_size,
                scale=resolved_scale,
            )
    else:
        selected = _validated_selected_indices(
            selected_indices,
            source_size=source_size,
            target_size=target_size,
        )

    selected_keys = full_keys[selected].copy()
    # Exact identity is a useful parity gate and avoids rank-sensitive OLS noise.
    if target_size == source_size and np.array_equal(selected, np.arange(source_size)):
        artifact = AttentionMatchingArtifact(
            source_size=source_size,
            selected_indices=selected,
            keys=selected_keys,
            values=full_values.copy(),
            beta_am=np.zeros(source_size, dtype=np.float64),
            scale=resolved_scale,
        )
    else:
        compact_logits = full_logits[:, selected]
        beta_am = (
            fit_beta_am(full_logits, compact_logits, ridge=mass_ridge)
            if fit_mass
            else np.zeros(target_size, dtype=np.float64)
        )
        teacher_output, _ = attention_from_logits(full_logits, full_values)
        compact_values = fit_ols_values(
            compact_logits,
            beta_am,
            teacher_output,
            ridge=value_ridge,
        )
        artifact = AttentionMatchingArtifact(
            source_size=source_size,
            selected_indices=selected,
            keys=selected_keys,
            values=compact_values,
            beta_am=beta_am,
            scale=resolved_scale,
        )
    artifact.validate(
        expected_source_size=source_size,
        expected_target_size=target_size,
        expected_key_dim=full_keys.shape[1],
        expected_value_dim=full_values.shape[1],
    )
    return artifact, evaluate_attention_matching(queries, full_keys, full_values, artifact)


def attend_compact_old_and_recent(
    queries: np.ndarray,
    compact_old: AttentionMatchingArtifact,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Attend to compressed old and exact recent memory under one denominator.

    A zero-length recent block is valid for the post-hoc oracle that compacts the
    entire teacher memory.  Its feature widths remain explicit, so this does not
    weaken the dimensionality or finiteness checks used by the regular attention
    paths.
    """

    recent_keys = np.asarray(recent_keys, dtype=np.float64)
    recent_values = np.asarray(recent_values, dtype=np.float64)
    if recent_keys.ndim != 2 or not recent_keys.shape[1]:
        raise ValueError(f"recent_keys must be rank-2 with nonzero width, got {recent_keys.shape}")
    if recent_values.ndim != 2 or not recent_values.shape[1]:
        raise ValueError(f"recent_values must be rank-2 with nonzero width, got {recent_values.shape}")
    if not np.isfinite(recent_keys).all() or not np.isfinite(recent_values).all():
        raise ValueError("recent K/V contain non-finite values")
    compact_old.validate(
        expected_key_dim=recent_keys.shape[1],
        expected_value_dim=recent_values.shape[1],
    )
    if recent_keys.shape[0] != recent_values.shape[0]:
        raise ValueError("recent key and value counts differ")
    combined_keys = np.concatenate([compact_old.keys, recent_keys], axis=0)
    combined_values = np.concatenate([compact_old.values, recent_values], axis=0)
    combined_beta_am = np.concatenate([compact_old.beta_am, np.zeros(recent_keys.shape[0], dtype=np.float64)])
    return scaled_dot_product_attention(
        queries,
        combined_keys,
        combined_values,
        beta_am=combined_beta_am,
        scale=compact_old.scale,
    )
