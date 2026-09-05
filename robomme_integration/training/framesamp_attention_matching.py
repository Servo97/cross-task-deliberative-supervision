"""RoboMME boundary between fixed-width FrameSamp buffers and Attention Matching.

The generic Attention Matching kernel operates only on real K/V rows.  Official
FrameSamp, however, presents a physical 512-token tensor with right-padding and
a Boolean mask.  This adapter removes padding *after the teacher K/V projection*
and carries the physical and logical source positions needed for audits.

Never infer padding from a zero feature value: a projected padded token may be
nonzero because of learned biases, while a real token may legitimately contain
zeros.  ``FrameSampHistory.token_mask`` is the sole authority.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from robomme_integration.training.attention_matching import (
    MASS_SOLVER,
    MASS_SOLVER_DISABLED,
    VALUE_SOLVER,
    AttentionMatchingArtifact,
    AttentionMatchingMetrics,
    fit_attention_matching,
)
from robomme_integration.training.upstream_framesamp_data import (
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    FrameSampHistory,
)


@dataclasses.dataclass(frozen=True)
class FrameSampAttentionMatchingResult:
    """One compact artifact plus its mapping back to causal FrameSamp tokens."""

    artifact: AttentionMatchingArtifact
    metrics: AttentionMatchingMetrics
    requested_target_size: int
    layer_index: int
    kv_head_index: int
    query_head_count: int
    teacher_key_position_span: int
    query_position_offset: int
    query_tap_stage: str
    key_tap_stage: str
    value_tap_stage: str
    fit_mass: bool
    mass_solver: str
    value_solver: str
    mass_ridge: float
    value_ridge: float
    source_physical_indices: np.ndarray
    selected_physical_indices: np.ndarray
    selected_step_indices: np.ndarray
    selected_patch_indices: np.ndarray

    @property
    def effective_target_size(self) -> int:
        return self.artifact.target_size

    def action_query_positions(self, token_count: int) -> np.ndarray:
        """Return the teacher's RoPE positions, independent of compact size."""

        if isinstance(token_count, bool) or not isinstance(token_count, (int, np.integer)):
            raise TypeError("token_count must be an integer")
        token_count = int(token_count)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        return np.arange(
            self.query_position_offset,
            self.query_position_offset + token_count,
            dtype=np.int32,
        )

    def validate(self, history: FrameSampHistory) -> None:
        history.validate()
        valid = np.flatnonzero(history.token_mask).astype(np.int64, copy=False)
        if self.requested_target_size <= 0:
            raise ValueError("requested_target_size must be positive")
        if self.layer_index < 0 or self.kv_head_index != 0 or self.query_head_count != 4:
            raise ValueError("official MemoryAttention requires layer>=0, one KV head, and four Q heads")
        if self.teacher_key_position_span != TOKEN_BUDGET or self.query_position_offset != TOKEN_BUDGET:
            raise ValueError("compaction changed the official 512-token logical RoPE span")
        expected_stages = (
            "post_rope_pre_scale",
            "post_rope",
            "post_projection",
        )
        if (self.query_tap_stage, self.key_tap_stage, self.value_tap_stage) != expected_stages:
            raise ValueError("Q/K/V tap-stage provenance is incompatible with official MemoryAttention")
        expected_mass_solver = MASS_SOLVER if self.fit_mass else MASS_SOLVER_DISABLED
        if self.mass_solver != expected_mass_solver or self.value_solver != VALUE_SOLVER:
            raise ValueError("attention-matching solver provenance is incompatible")
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in (self.mass_ridge, self.value_ridge)):
            raise ValueError("attention-matching ridge values must be finite and nonnegative")
        if not np.array_equal(self.source_physical_indices, valid):
            raise ValueError("FrameSamp source-position map disagrees with token_mask")
        if self.artifact.source_size != valid.size:
            raise ValueError("attention artifact source_size does not equal the number of valid FrameSamp tokens")
        expected_target = min(self.requested_target_size, valid.size)
        self.artifact.validate(expected_source_size=valid.size, expected_target_size=expected_target)
        selected = self.artifact.selected_indices.astype(np.int64, copy=False)
        expected_physical = valid[selected]
        if not np.array_equal(self.selected_physical_indices, expected_physical):
            raise ValueError("selected physical positions disagree with the compact artifact")
        frame_slots = expected_physical // TOKENS_PER_FRAME
        expected_steps = history.frame_indices[frame_slots]
        expected_patches = expected_physical % TOKENS_PER_FRAME
        if not np.array_equal(self.selected_step_indices, expected_steps):
            raise ValueError("selected logical step positions disagree with FrameSamp frame indices")
        if not np.array_equal(self.selected_patch_indices, expected_patches):
            raise ValueError("selected 4x4 patch positions disagree with physical token positions")


def fit_framesamp_attention_matching(
    history: FrameSampHistory,
    teacher_rope_queries_pre_scale: np.ndarray,
    teacher_rope_keys: np.ndarray,
    teacher_values: np.ndarray,
    target_size: int,
    *,
    layer_index: int,
    kv_head_index: int = 0,
    fit_mass: bool = True,
    mass_ridge: float = 0.0,
    value_ridge: float = 0.0,
) -> FrameSampAttentionMatchingResult:
    """Fit AM on valid teacher K/V rows and retain causal source metadata.

    ``teacher_rope_queries_pre_scale`` must be ``[4, samples, head_dim]``: all
    four action query heads after RoPE but before the official in-place
    ``1/sqrt(head_dim)`` scale. ``teacher_rope_keys`` is the one KV head after
    RoPE and ``teacher_values`` is after V projection.  The generic fitter then
    applies the scale exactly once.  Compact keys are consequently already
    RoPE-applied and must not pass through the original K projection/RoPE again.

    Masking happens here, after teacher projection, exactly as it does in the
    teacher logits. If an early prefix contains fewer real tokens than the
    requested budget, all real tokens are retained and the downstream
    fixed-width adapter may right-pad the compact artifact separately.  Runtime
    action-query RoPE still begins at logical position 512, not at compact size
    M; changing that offset invalidates the fitted query/key geometry.
    """

    history.validate()
    teacher_rope_queries_pre_scale = np.asarray(teacher_rope_queries_pre_scale)
    teacher_rope_keys = np.asarray(teacher_rope_keys)
    teacher_values = np.asarray(teacher_values)
    if teacher_rope_queries_pre_scale.ndim != 3 or teacher_rope_queries_pre_scale.shape[0] != 4:
        raise ValueError("official query tap must be [4 query heads, samples, head_dim]")
    if teacher_rope_keys.ndim != 2 or teacher_values.ndim != 2:
        raise ValueError("teacher K/V taps must be rank-2 per-layer/head matrices")
    if teacher_rope_keys.shape[0] != TOKEN_BUDGET or teacher_values.shape[0] != TOKEN_BUDGET:
        raise ValueError(f"teacher K/V taps must each contain the physical {TOKEN_BUDGET}-token FrameSamp buffer")
    if teacher_rope_queries_pre_scale.shape[-1] != teacher_rope_keys.shape[-1]:
        raise ValueError("teacher Q/K head dimensions differ")
    if isinstance(layer_index, bool) or not isinstance(layer_index, (int, np.integer)):
        raise TypeError("layer_index must be an integer")
    layer_index = int(layer_index)
    if layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    if isinstance(kv_head_index, bool) or int(kv_head_index) != 0:
        raise ValueError("official MemoryAttention has exactly one KV head at index 0")
    if isinstance(target_size, bool) or not isinstance(target_size, (int, np.integer)):
        raise TypeError("target_size must be an integer")
    requested_target_size = int(target_size)
    if requested_target_size <= 0:
        raise ValueError("target_size must be positive")

    source_physical_indices = np.flatnonzero(history.token_mask).astype(np.int64, copy=False)
    if not source_physical_indices.size:
        raise ValueError("FrameSamp history contains no valid source tokens")
    effective_target_size = min(requested_target_size, source_physical_indices.size)
    artifact, metrics = fit_attention_matching(
        teacher_rope_queries_pre_scale.reshape(-1, teacher_rope_queries_pre_scale.shape[-1]),
        teacher_rope_keys[source_physical_indices],
        teacher_values[source_physical_indices],
        effective_target_size,
        fit_mass=fit_mass,
        mass_ridge=mass_ridge,
        value_ridge=value_ridge,
    )
    selected_physical_indices = source_physical_indices[artifact.selected_indices]
    frame_slots = selected_physical_indices // TOKENS_PER_FRAME
    result = FrameSampAttentionMatchingResult(
        artifact=artifact,
        metrics=metrics,
        requested_target_size=requested_target_size,
        layer_index=layer_index,
        kv_head_index=0,
        query_head_count=4,
        teacher_key_position_span=TOKEN_BUDGET,
        query_position_offset=TOKEN_BUDGET,
        query_tap_stage="post_rope_pre_scale",
        key_tap_stage="post_rope",
        value_tap_stage="post_projection",
        fit_mass=bool(fit_mass),
        mass_solver=MASS_SOLVER if fit_mass else MASS_SOLVER_DISABLED,
        value_solver=VALUE_SOLVER,
        mass_ridge=float(mass_ridge),
        value_ridge=float(value_ridge),
        source_physical_indices=source_physical_indices,
        selected_physical_indices=selected_physical_indices,
        selected_step_indices=history.frame_indices[frame_slots].astype(np.int32, copy=True),
        selected_patch_indices=(selected_physical_indices % TOKENS_PER_FRAME).astype(np.int16, copy=False),
    )
    result.validate(history)
    return result
