"""Deterministic action-query banks for RoboMME FrameSamp Attention Matching.

This module owns sampling *identity*, not policy instrumentation.  It creates a
domain-separated list of diffusion/noise/action requests and accepts Q tensors
only through :class:`CapturedActionQueries`, whose request order and tap stage
are checked exactly.  The policy-specific hook that captures the four action-Q
heads remains a separate seam.

Fit results are created through :func:`fit_framesamp_am_from_query_banks` so the
actual query bytes and their structured sampling specifications cannot be
substituted between fitting and bundle sealing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from robomme_integration.training.framesamp_am_artifact import (
    KEY_TAP_STAGE,
    QUERY_TAP_STAGE,
    VALUE_TAP_STAGE,
    FrameSampAMManifest,
    QuantizationParityThresholds,
    array_bundle_sha256,
    seal_framesamp_am_artifact,
)
from robomme_integration.training.framesamp_attention_matching import (
    FrameSampAttentionMatchingResult,
    fit_framesamp_attention_matching,
)
from robomme_integration.training.upstream_framesamp_data import TOKEN_BUDGET, FrameSampHistory

QUERY_BANK_SCHEMA_VERSION = 1
QUERY_BANK_SEED_DERIVATION = "sha256_domain_separated_uint63_v1"
QUERY_CAPTURE_CONTRACT = "official_memory_attention_action_q_post_rope_pre_scale_v1"
OFFICIAL_ACTION_TOKENS_PER_SAMPLE = 20
FIT_SPLIT = "fit"
HELDOUT_SPLIT = "heldout"
_SPLITS = frozenset({FIT_SPLIT, HELDOUT_SPLIT})
_HEX = frozenset("0123456789abcdef")


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _require_sha(value: object, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
    value = _require_nonempty(value, label=label)
    if len(value) not in lengths or any(character not in _HEX for character in value):
        raise ValueError(f"{label} must be a lowercase {'/'.join(map(str, lengths))}-hex SHA")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class ActionQuerySamplingSpec:
    """Complete route and stochastic sampling identity for one Q-bank split."""

    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    layer_index: int
    split: str
    split_seed: int
    diffusion_schedule_id: str
    diffusion_timesteps: tuple[float, ...]
    noise_distribution: str
    noise_samples_per_timestep: int
    action_sampler_id: str
    action_samples_per_noise: int
    action_tokens_per_sample: int
    kv_head_index: int = 0
    query_head_count: int = 4
    query_tap_stage: str = QUERY_TAP_STAGE
    capture_contract: str = QUERY_CAPTURE_CONTRACT
    seed_derivation: str = QUERY_BANK_SEED_DERIVATION
    schema_version: int = QUERY_BANK_SCHEMA_VERSION

    def validate(self) -> None:
        if _require_int(self.schema_version, label="schema_version") != QUERY_BANK_SCHEMA_VERSION:
            raise ValueError(f"unsupported query-bank schema {self.schema_version}")
        _require_sha(self.teacher_checkpoint_sha256, label="teacher_checkpoint_sha256")
        _require_sha(self.teacher_code_sha, label="teacher_code_sha", lengths=(40, 64))
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_int(self.layer_index, label="layer_index")
        if _require_int(self.kv_head_index, label="kv_head_index") != 0:
            raise ValueError("official MemoryAttention has exactly one KV head at index 0")
        if _require_int(self.query_head_count, label="query_head_count", minimum=1) != 4:
            raise ValueError("official MemoryAttention has exactly four action-query heads")
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}")
        _require_int(self.split_seed, label="split_seed")
        _require_nonempty(self.diffusion_schedule_id, label="diffusion_schedule_id")
        if not isinstance(self.diffusion_timesteps, tuple) or not self.diffusion_timesteps:
            raise ValueError("diffusion_timesteps must be a nonempty immutable tuple")
        timesteps = tuple(float(value) for value in self.diffusion_timesteps)
        if not all(math.isfinite(value) and value >= 0 for value in timesteps):
            raise ValueError("diffusion_timesteps must be finite and nonnegative")
        if len(set(timesteps)) != len(timesteps):
            raise ValueError("diffusion_timesteps must be unique")
        _require_nonempty(self.noise_distribution, label="noise_distribution")
        _require_int(self.noise_samples_per_timestep, label="noise_samples_per_timestep", minimum=1)
        _require_nonempty(self.action_sampler_id, label="action_sampler_id")
        _require_int(self.action_samples_per_noise, label="action_samples_per_noise", minimum=1)
        if (
            _require_int(self.action_tokens_per_sample, label="action_tokens_per_sample", minimum=1)
            != OFFICIAL_ACTION_TOKENS_PER_SAMPLE
        ):
            raise ValueError("official RoboMME pi0.5 query banks require the full 20-token action horizon")
        if self.query_tap_stage != QUERY_TAP_STAGE:
            raise ValueError(f"query_tap_stage must be {QUERY_TAP_STAGE!r}")
        if self.capture_contract != QUERY_CAPTURE_CONTRACT:
            raise ValueError("unsupported teacher action-query capture contract")
        if self.seed_derivation != QUERY_BANK_SEED_DERIVATION:
            raise ValueError("unsupported query-bank seed derivation")

    @property
    def request_count(self) -> int:
        self.validate()
        return len(self.diffusion_timesteps) * self.noise_samples_per_timestep * self.action_samples_per_noise

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "teacher_code_sha": self.teacher_code_sha,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "causal_cut_step": self.causal_cut_step,
            "layer_index": self.layer_index,
            "kv_head_index": self.kv_head_index,
            "query_head_count": self.query_head_count,
            "split": self.split,
            "split_seed": self.split_seed,
            "diffusion_schedule_id": self.diffusion_schedule_id,
            "diffusion_timesteps": [float(value) for value in self.diffusion_timesteps],
            "noise_distribution": self.noise_distribution,
            "noise_samples_per_timestep": self.noise_samples_per_timestep,
            "action_sampler_id": self.action_sampler_id,
            "action_samples_per_noise": self.action_samples_per_noise,
            "action_tokens_per_sample": self.action_tokens_per_sample,
            "query_tap_stage": self.query_tap_stage,
            "capture_contract": self.capture_contract,
            "seed_derivation": self.seed_derivation,
        }

    def sha256(self) -> str:
        return _json_sha256(self.to_dict())

    def routing_identity(self) -> tuple[object, ...]:
        """Fields that fit and held-out banks must share exactly."""

        self.validate()
        return (
            self.teacher_checkpoint_sha256,
            self.teacher_code_sha,
            self.task_id,
            self.episode_id,
            self.causal_cut_step,
            self.layer_index,
            self.kv_head_index,
            self.query_head_count,
            self.diffusion_schedule_id,
            self.noise_distribution,
            self.action_sampler_id,
            self.action_tokens_per_sample,
            self.query_tap_stage,
            self.capture_contract,
            self.seed_derivation,
        )

    def stochastic_identity(self) -> dict[str, object]:
        """Random-draw identity shared across teacher layers/checkpoints.

        Layer and teacher identity still enter each request SHA and bank
        provenance, but must not change the underlying diffusion noise/action
        seeds.  This lets every layer observe the same sampled trajectory and
        makes checkpoint comparisons paired rather than confounded by new RNG.
        """

        self.validate()
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "causal_cut_step": self.causal_cut_step,
            "split": self.split,
            "split_seed": self.split_seed,
            "diffusion_schedule_id": self.diffusion_schedule_id,
            "diffusion_timesteps": [float(value) for value in self.diffusion_timesteps],
            "noise_distribution": self.noise_distribution,
            "noise_samples_per_timestep": self.noise_samples_per_timestep,
            "action_sampler_id": self.action_sampler_id,
            "action_samples_per_noise": self.action_samples_per_noise,
            "action_tokens_per_sample": self.action_tokens_per_sample,
            "seed_derivation": self.seed_derivation,
        }


def _derive_uint63(stochastic_sha256: str, coordinates: tuple[int, int, int], *, domain: str) -> int:
    payload = {
        "coordinates": list(coordinates),
        "domain": domain,
        "stochastic_sampling_sha256": stochastic_sha256,
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(payload).encode()).digest()[:8], "big") & ((1 << 63) - 1)


@dataclasses.dataclass(frozen=True)
class ActionQueryCaptureRequest:
    """One deterministic teacher invocation within a query bank."""

    ordinal: int
    timestep_index: int
    diffusion_timestep: float
    noise_sample_index: int
    action_sample_index: int
    noise_seed: int
    action_seed: int
    request_sha256: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def action_query_capture_requests(spec: ActionQuerySamplingSpec) -> tuple[ActionQueryCaptureRequest, ...]:
    """Create a stable, domain-separated request schedule for one split."""

    spec.validate()
    spec_sha = spec.sha256()
    stochastic_sha = _json_sha256(spec.stochastic_identity())
    requests: list[ActionQueryCaptureRequest] = []
    ordinal = 0
    for timestep_index, timestep in enumerate(spec.diffusion_timesteps):
        for noise_index in range(spec.noise_samples_per_timestep):
            for action_index in range(spec.action_samples_per_noise):
                coordinates = (timestep_index, noise_index, action_index)
                body = {
                    "action_sample_index": action_index,
                    "action_seed": _derive_uint63(stochastic_sha, coordinates, domain="action"),
                    "diffusion_timestep": float(timestep),
                    "noise_sample_index": noise_index,
                    "noise_seed": _derive_uint63(stochastic_sha, coordinates, domain="noise"),
                    "ordinal": ordinal,
                    "sampling_spec_sha256": spec_sha,
                    "timestep_index": timestep_index,
                }
                requests.append(
                    ActionQueryCaptureRequest(
                        ordinal=ordinal,
                        timestep_index=timestep_index,
                        diffusion_timestep=float(timestep),
                        noise_sample_index=noise_index,
                        action_sample_index=action_index,
                        noise_seed=body["noise_seed"],
                        action_seed=body["action_seed"],
                        request_sha256=_json_sha256(body),
                    )
                )
                ordinal += 1
    if len({request.request_sha256 for request in requests}) != len(requests):  # pragma: no cover - SHA collision.
        raise RuntimeError("query-bank request SHA collision")
    return tuple(requests)


@dataclasses.dataclass(frozen=True)
class CapturedActionQueries:
    """Strict policy-instrumentation output accepted by the framework-free producer.

    ``queries_by_request`` is ``[request, 4 Q heads, action token, head dim]``.
    The external capture hook must seed its diffusion noise and action sampling
    from the supplied requests; this boundary rejects reordered or stale taps.
    """

    sampling_spec_sha256: str
    request_sha256s: tuple[str, ...]
    queries_by_request: np.ndarray
    query_tap_stage: str

    def validate(
        self,
        spec: ActionQuerySamplingSpec,
        requests: tuple[ActionQueryCaptureRequest, ...],
    ) -> None:
        spec.validate()
        if self.sampling_spec_sha256 != spec.sha256():
            raise ValueError("captured Q tensor sampling-spec SHA mismatch")
        expected_requests = tuple(request.request_sha256 for request in requests)
        if self.request_sha256s != expected_requests:
            raise ValueError("captured Q tensor request identities/order mismatch")
        if self.query_tap_stage != QUERY_TAP_STAGE:
            raise ValueError("captured action Q must be post-RoPE and pre-scale")
        queries = np.asarray(self.queries_by_request)
        expected_prefix = (len(requests), spec.query_head_count, spec.action_tokens_per_sample)
        if queries.ndim != 4 or queries.shape[:3] != expected_prefix or not queries.shape[-1]:
            raise ValueError(
                "captured Q tensor must be [request, 4 heads, action tokens, head dim]; "
                f"expected prefix {expected_prefix}, got {queries.shape}"
            )
        if not np.issubdtype(queries.dtype, np.floating):
            raise ValueError("captured Q tensor must have a floating dtype")
        if not np.isfinite(queries.astype(np.float64)).all():
            raise ValueError("captured Q tensor contains non-finite values")


@dataclasses.dataclass(frozen=True)
class ActionQueryBank:
    """Content-addressed flattened action-query bank accepted by the AM fitter."""

    spec: ActionQuerySamplingSpec
    requests: tuple[ActionQueryCaptureRequest, ...]
    queries_post_rope_pre_scale: np.ndarray
    sampling_spec_sha256: str
    requests_sha256: str
    queries_sha256: str
    bank_sha256: str

    def validate(self) -> None:
        self.spec.validate()
        expected_requests = action_query_capture_requests(self.spec)
        if self.requests != expected_requests:
            raise ValueError("query bank request schedule disagrees with its sampling spec")
        queries = np.asarray(self.queries_post_rope_pre_scale)
        expected_samples = len(self.requests) * self.spec.action_tokens_per_sample
        if queries.ndim != 3 or queries.shape[:2] != (4, expected_samples) or not queries.shape[-1]:
            raise ValueError("flattened query bank must be [4 heads, request*action tokens, head dim]")
        if not np.issubdtype(queries.dtype, np.floating) or not np.isfinite(queries.astype(np.float64)).all():
            raise ValueError("flattened query bank must contain finite floating values")
        expected_spec_sha = self.spec.sha256()
        expected_requests_sha = _json_sha256([request.to_dict() for request in self.requests])
        expected_queries_sha = array_bundle_sha256(queries_post_rope_pre_scale=queries)
        identity = {
            "dtype": str(queries.dtype),
            "queries_sha256": expected_queries_sha,
            "query_shape": list(queries.shape),
            "requests_sha256": expected_requests_sha,
            "sampling_spec_sha256": expected_spec_sha,
            "schema_version": QUERY_BANK_SCHEMA_VERSION,
        }
        expected_bank_sha = _json_sha256(identity)
        for label, actual, expected in (
            ("sampling_spec_sha256", self.sampling_spec_sha256, expected_spec_sha),
            ("requests_sha256", self.requests_sha256, expected_requests_sha),
            ("queries_sha256", self.queries_sha256, expected_queries_sha),
            ("bank_sha256", self.bank_sha256, expected_bank_sha),
        ):
            _require_sha(actual, label=label)
            if actual != expected:
                raise ValueError(f"query bank {label} mismatch")

    def binding_json(self) -> str:
        """Canonical manifest field binding both the sampling spec and actual Q bytes."""

        self.validate()
        return _canonical_json(
            {
                "bank_sha256": self.bank_sha256,
                "queries_sha256": self.queries_sha256,
                "requests_sha256": self.requests_sha256,
                "sampling_spec": self.spec.to_dict(),
                "sampling_spec_sha256": self.sampling_spec_sha256,
            }
        )


CaptureProvider = Callable[
    [ActionQuerySamplingSpec, tuple[ActionQueryCaptureRequest, ...]],
    CapturedActionQueries,
]


def produce_action_query_bank(spec: ActionQuerySamplingSpec, capture_provider: CaptureProvider) -> ActionQueryBank:
    """Plan requests, invoke the strict tap seam, and content-address its Q array."""

    if not callable(capture_provider):
        raise TypeError("capture_provider must be callable")
    requests = action_query_capture_requests(spec)
    captured = capture_provider(spec, requests)
    if not isinstance(captured, CapturedActionQueries):
        raise TypeError("capture_provider must return CapturedActionQueries")
    captured.validate(spec, requests)
    raw = np.asarray(captured.queries_by_request)
    queries = np.ascontiguousarray(raw.transpose(1, 0, 2, 3).reshape(4, -1, raw.shape[-1]))
    spec_sha = spec.sha256()
    requests_sha = _json_sha256([request.to_dict() for request in requests])
    queries_sha = array_bundle_sha256(queries_post_rope_pre_scale=queries)
    bank_identity = {
        "dtype": str(queries.dtype),
        "queries_sha256": queries_sha,
        "query_shape": list(queries.shape),
        "requests_sha256": requests_sha,
        "sampling_spec_sha256": spec_sha,
        "schema_version": QUERY_BANK_SCHEMA_VERSION,
    }
    result = ActionQueryBank(
        spec=spec,
        requests=requests,
        queries_post_rope_pre_scale=queries,
        sampling_spec_sha256=spec_sha,
        requests_sha256=requests_sha,
        queries_sha256=queries_sha,
        bank_sha256=_json_sha256(bank_identity),
    )
    result.validate()
    return result


@dataclasses.dataclass(frozen=True)
class ActionQueryBankPair:
    fit: ActionQueryBank
    heldout: ActionQueryBank

    def validate(self) -> None:
        self.fit.validate()
        self.heldout.validate()
        if self.fit.spec.split != FIT_SPLIT or self.heldout.spec.split != HELDOUT_SPLIT:
            raise ValueError("query-bank pair must contain fit then heldout splits")
        if self.fit.spec.routing_identity() != self.heldout.spec.routing_identity():
            raise ValueError("fit and heldout query banks have different teacher/routing identities")
        if self.fit.spec.diffusion_timesteps != self.heldout.spec.diffusion_timesteps:
            raise ValueError("fit and heldout query banks must cover the same diffusion timesteps")
        if self.fit.spec.split_seed == self.heldout.spec.split_seed:
            raise ValueError("fit and heldout query banks must use different split seeds")
        fit_requests = {request.request_sha256 for request in self.fit.requests}
        heldout_requests = {request.request_sha256 for request in self.heldout.requests}
        if fit_requests & heldout_requests:  # pragma: no cover - domain-separated SHA collision.
            raise ValueError("fit and heldout query-bank request identities overlap")
        if self.fit.queries_sha256 == self.heldout.queries_sha256:
            raise ValueError("fit and heldout query arrays must be content-distinct")


def produce_fit_heldout_query_banks(
    fit_spec: ActionQuerySamplingSpec,
    heldout_spec: ActionQuerySamplingSpec,
    capture_provider: CaptureProvider,
) -> ActionQueryBankPair:
    """Produce and prove disjoint deterministic fit/held-out Q banks."""

    result = ActionQueryBankPair(
        fit=produce_action_query_bank(fit_spec, capture_provider),
        heldout=produce_action_query_bank(heldout_spec, capture_provider),
    )
    result.validate()
    return result


@dataclasses.dataclass(frozen=True)
class CapturedTeacherMemoryTaps:
    """Strict K/V capture boundary; policy-layer instrumentation lives elsewhere."""

    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    layer_index: int
    kv_head_index: int
    keys_post_rope: np.ndarray
    values_post_projection: np.ndarray
    key_tap_stage: str = KEY_TAP_STAGE
    value_tap_stage: str = VALUE_TAP_STAGE

    def validate(self, spec: ActionQuerySamplingSpec) -> None:
        spec.validate()
        identity = (
            self.teacher_checkpoint_sha256,
            self.teacher_code_sha,
            self.task_id,
            self.episode_id,
            self.causal_cut_step,
            self.layer_index,
            self.kv_head_index,
        )
        if identity != spec.routing_identity()[:7]:
            raise ValueError("teacher K/V taps have a different checkpoint/routing identity than the query banks")
        if self.key_tap_stage != KEY_TAP_STAGE or self.value_tap_stage != VALUE_TAP_STAGE:
            raise ValueError("teacher K/V must be captured post-RoPE/post-projection")
        keys = np.asarray(self.keys_post_rope)
        values = np.asarray(self.values_post_projection)
        if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != TOKEN_BUDGET or values.shape[0] != TOKEN_BUDGET:
            raise ValueError(f"teacher K/V taps must retain the physical {TOKEN_BUDGET}-token buffer")
        if keys.shape[1] != values.shape[1]:
            raise ValueError("official MemoryAttention requires equal K/V head dimensions")
        if (
            not keys.shape[1]
            or not values.shape[1]
            or not all(
                np.issubdtype(value.dtype, np.floating) and np.isfinite(value.astype(np.float64)).all()
                for value in (keys, values)
            )
        ):
            raise ValueError("teacher K/V taps must contain finite floating values")


@dataclasses.dataclass(frozen=True)
class BoundFrameSampAMFit:
    """Fit result carrying the exact banks and teacher taps that created it."""

    result: FrameSampAttentionMatchingResult
    banks: ActionQueryBankPair
    teacher_taps: CapturedTeacherMemoryTaps

    def validate(self, history: FrameSampHistory) -> None:
        self.banks.validate()
        self.teacher_taps.validate(self.banks.fit.spec)
        self.result.validate(history)
        if self.result.layer_index != self.banks.fit.spec.layer_index:
            raise ValueError("fit result layer differs from query-bank route")
        if self.result.requested_target_size <= 0:
            raise ValueError("fit result has no requested target budget")
        if self.banks.fit.queries_post_rope_pre_scale.shape[-1] != self.teacher_taps.keys_post_rope.shape[-1]:
            raise ValueError("query-bank and teacher-key head dimensions differ")


def fit_framesamp_am_from_query_banks(
    history: FrameSampHistory,
    banks: ActionQueryBankPair,
    teacher_taps: CapturedTeacherMemoryTaps,
    target_size: int,
    *,
    fit_mass: bool = True,
    mass_ridge: float = 0.0,
    value_ridge: float = 0.0,
) -> BoundFrameSampAMFit:
    """Fit AM while retaining an immutable binding to actual Q bytes/specs."""

    banks.validate()
    teacher_taps.validate(banks.fit.spec)
    result = fit_framesamp_attention_matching(
        history,
        banks.fit.queries_post_rope_pre_scale,
        teacher_taps.keys_post_rope,
        teacher_taps.values_post_projection,
        target_size,
        layer_index=banks.fit.spec.layer_index,
        kv_head_index=banks.fit.spec.kv_head_index,
        fit_mass=fit_mass,
        mass_ridge=mass_ridge,
        value_ridge=value_ridge,
    )
    bound = BoundFrameSampAMFit(result=result, banks=banks, teacher_taps=teacher_taps)
    bound.validate(history)
    return bound


def seal_bound_framesamp_am_artifact(
    destination: str | Path,
    bound: BoundFrameSampAMFit,
    history: FrameSampHistory,
    *,
    storage_dtype: str = "float32",
    parity_thresholds: QuantizationParityThresholds = QuantizationParityThresholds(),
) -> FrameSampAMManifest:
    """Seal an existing bound fit through the repository's trusted bundle contract."""

    bound.validate(history)
    spec = bound.banks.fit.spec
    manifest = seal_framesamp_am_artifact(
        destination,
        bound.result,
        history,
        bound.banks.fit.queries_post_rope_pre_scale,
        bound.banks.heldout.queries_post_rope_pre_scale,
        bound.teacher_taps.keys_post_rope,
        bound.teacher_taps.values_post_projection,
        teacher_checkpoint_sha256=spec.teacher_checkpoint_sha256,
        teacher_code_sha=spec.teacher_code_sha,
        task_id=spec.task_id,
        episode_id=spec.episode_id,
        causal_cut_step=spec.causal_cut_step,
        fit_query_bank_spec=bound.banks.fit.binding_json(),
        heldout_query_bank_spec=bound.banks.heldout.binding_json(),
        storage_dtype=storage_dtype,
        parity_thresholds=parity_thresholds,
    )
    if manifest.fit_query_bank_sha256 != bound.banks.fit.queries_sha256:
        raise RuntimeError("sealed fit-query SHA disagrees with its bound query bank")
    if manifest.heldout_query_bank_sha256 != bound.banks.heldout.queries_sha256:
        raise RuntimeError("sealed heldout-query SHA disagrees with its bound query bank")
    if manifest.fit_query_bank_spec != bound.banks.fit.binding_json():
        raise RuntimeError("sealed fit-query spec disagrees with its bound query bank")
    if manifest.heldout_query_bank_spec != bound.banks.heldout.binding_json():
        raise RuntimeError("sealed heldout-query spec disagrees with its bound query bank")
    return manifest
