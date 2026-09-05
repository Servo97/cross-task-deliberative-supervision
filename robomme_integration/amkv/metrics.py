"""Velocity-field comparison for full versus compacted memory.

E0's primary estimand is the policy's own control signal: at every flow time in
the denoise schedule, how far does the action velocity ``v(x_t, t)`` move when
history K/V is replaced by an AM artifact while ``x_t`` is held to the *same
full-teacher denoising state* in both calls?  Everything is *relative*
(``||dv|| / ||v||``) because absolute scales differ across flow times and the
study's serve-numerics lesson says only relative, within-run paired contrasts
are clean.

All reductions accumulate in float64 regardless of the serve dtype.
"""

from __future__ import annotations

import dataclasses

import numpy as np

VELOCITY_METRIC = "teacher_forced_relative_frobenius_velocity_delta_v2"
ACTION_METRIC = "closed_loop_relative_action_chunk_delta_v1"


def _as_trace(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or not array.shape[0] or not array.shape[1] or not array.shape[2]:
        raise ValueError(
            f"{label} must be a nonempty [flow_steps, action_tokens, action_dim] trace, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def relative_velocity_error(full: np.ndarray, compact: np.ndarray) -> np.ndarray:
    """Per-flow-time ``||v_compact - v_full||_F / ||v_full||_F``."""

    full = _as_trace(full, label="full velocity trace")
    compact = _as_trace(compact, label="compact velocity trace")
    if full.shape != compact.shape:
        raise ValueError(f"velocity traces differ in shape: {full.shape} vs {compact.shape}")
    difference = compact - full
    numerator = np.sqrt(np.sum(np.square(difference), axis=(1, 2)))
    denominator = np.sqrt(np.sum(np.square(full), axis=(1, 2)))
    if np.any(denominator <= 0):
        raise ValueError("full velocity has zero norm at some flow time; a relative error is undefined")
    return numerator / denominator


def velocity_cosine(full: np.ndarray, compact: np.ndarray) -> np.ndarray:
    """Per-flow-time cosine similarity of the flattened velocity fields."""

    full = _as_trace(full, label="full velocity trace")
    compact = _as_trace(compact, label="compact velocity trace")
    if full.shape != compact.shape:
        raise ValueError("velocity traces differ in shape")
    left = full.reshape(full.shape[0], -1)
    right = compact.reshape(compact.shape[0], -1)
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(norms <= 0):
        raise ValueError("cosine similarity is undefined for a zero velocity field")
    return np.sum(left * right, axis=1) / norms


@dataclasses.dataclass(frozen=True)
class VelocityComparison:
    """One episode-prefix, one arm: full versus compacted denoising."""

    fixture_id: str
    arm_id: str
    flow_times: tuple[float, ...]
    relative_velocity_error: tuple[float, ...]
    velocity_cosine: tuple[float, ...]
    relative_action_error: float
    full_velocity_norm: tuple[float, ...]

    def validate(self) -> None:
        if not self.fixture_id or not self.arm_id:
            raise ValueError("a velocity comparison must name its fixture and arm")
        count = len(self.flow_times)
        if not count:
            raise ValueError("a velocity comparison needs at least one flow time")
        for name in ("relative_velocity_error", "velocity_cosine", "full_velocity_norm"):
            values = getattr(self, name)
            if len(values) != count:
                raise ValueError(f"{name} has {len(values)} entries but there are {count} flow times")
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"{name} contains non-finite values")
        if not np.isfinite(self.relative_action_error) or self.relative_action_error < 0:
            raise ValueError("relative_action_error must be finite and nonnegative")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "fixture_id": self.fixture_id,
            "arm_id": self.arm_id,
            "metric": VELOCITY_METRIC,
            "velocity_evaluation": "same_full_teacher_x_t_at_each_flow_time",
            "flow_times": [float(value) for value in self.flow_times],
            "relative_velocity_error": [float(value) for value in self.relative_velocity_error],
            "velocity_cosine": [float(value) for value in self.velocity_cosine],
            "full_velocity_norm": [float(value) for value in self.full_velocity_norm],
            "relative_action_error": float(self.relative_action_error),
            "action_metric": ACTION_METRIC,
            "action_evaluation": "independent_closed_loop_denoising_trajectories",
        }


def compare_velocity_traces(
    *,
    fixture_id: str,
    arm_id: str,
    flow_times: tuple[float, ...],
    full_velocity: np.ndarray,
    compact_velocity: np.ndarray,
    full_actions: np.ndarray,
    compact_actions: np.ndarray,
) -> VelocityComparison:
    full_velocity = _as_trace(full_velocity, label="full velocity trace")
    if len(flow_times) != full_velocity.shape[0]:
        raise ValueError("flow_times does not describe the traced flow steps")
    actions_full = np.asarray(full_actions, dtype=np.float64)
    actions_compact = np.asarray(compact_actions, dtype=np.float64)
    if actions_full.shape != actions_compact.shape or actions_full.size == 0:
        raise ValueError("final action chunks differ in shape or are empty")
    denominator = float(np.linalg.norm(actions_full))
    if denominator <= 0:
        raise ValueError("full action chunk has zero norm; a relative error is undefined")
    comparison = VelocityComparison(
        fixture_id=fixture_id,
        arm_id=arm_id,
        flow_times=tuple(float(value) for value in flow_times),
        relative_velocity_error=tuple(
            float(value) for value in relative_velocity_error(full_velocity, compact_velocity)
        ),
        velocity_cosine=tuple(float(value) for value in velocity_cosine(full_velocity, compact_velocity)),
        relative_action_error=float(np.linalg.norm(actions_compact - actions_full) / denominator),
        full_velocity_norm=tuple(float(value) for value in np.sqrt(np.sum(np.square(full_velocity), axis=(1, 2)))),
    )
    comparison.validate()
    return comparison


def aggregate_comparisons(comparisons: tuple[VelocityComparison, ...]) -> dict[str, object]:
    """Per-flow-time mean/p95/max over episode prefixes, plus the action delta."""

    if not comparisons:
        raise ValueError("no comparisons to aggregate")
    arms = {comparison.arm_id for comparison in comparisons}
    if len(arms) != 1:
        raise ValueError(f"aggregate one arm at a time, got {sorted(arms)}")
    flow_times = comparisons[0].flow_times
    for comparison in comparisons:
        comparison.validate()
        if comparison.flow_times != flow_times:
            raise ValueError("comparisons use different flow schedules and cannot be pooled")
    errors = np.asarray([comparison.relative_velocity_error for comparison in comparisons], dtype=np.float64)
    cosines = np.asarray([comparison.velocity_cosine for comparison in comparisons], dtype=np.float64)
    actions = np.asarray([comparison.relative_action_error for comparison in comparisons], dtype=np.float64)
    fixtures = sorted({comparison.fixture_id for comparison in comparisons})
    return {
        "arm_id": comparisons[0].arm_id,
        "metric": VELOCITY_METRIC,
        "velocity_evaluation": "same_full_teacher_x_t_at_each_flow_time",
        "action_metric": ACTION_METRIC,
        "action_evaluation": "independent_closed_loop_denoising_trajectories",
        "episode_count": len(comparisons),
        "distinct_fixtures": len(fixtures),
        "flow_times": [float(value) for value in flow_times],
        "relative_velocity_error_mean": [float(value) for value in errors.mean(axis=0)],
        "relative_velocity_error_p95": [float(value) for value in np.percentile(errors, 95, axis=0)],
        "relative_velocity_error_max": [float(value) for value in errors.max(axis=0)],
        "velocity_cosine_mean": [float(value) for value in cosines.mean(axis=0)],
        "worst_flow_time": float(flow_times[int(np.argmax(errors.mean(axis=0)))]),
        "relative_velocity_error_pooled_mean": float(errors.mean()),
        "relative_action_error_mean": float(actions.mean()),
        "relative_action_error_p95": float(np.percentile(actions, 95)),
        "relative_action_error_max": float(actions.max()),
    }
