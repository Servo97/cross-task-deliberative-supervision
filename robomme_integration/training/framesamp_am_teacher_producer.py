"""Source-pinned FrameSamp teacher taps and all-layer AM query banks.

This is the durable producer boundary for the E1 Attention-Matching path.  It
does not edit the released RoboMME checkout or checkpoint and it does not run a
simulator.  Instead it:

* binds a capture to the *actual* 512-token causal history bytes, not merely a
  task/episode/step label;
* captures every action-expert layer in one teacher invocation schedule;
* fans those captures into the existing per-layer query-bank and K/V APIs; and
* rejects stale captures when an on-policy rollout reaches the same labelled
  cut with different history bytes.

The reviewed module overlay already exposes the required tensors through the
``framesamp_am_taps`` Linen collection.  The released policy's
``sample_actions`` method does not return that collection, so this module also
provides a narrow extractor and an explicit-suffix forward helper.  A full
released-checkpoint action-parity smoke remains a mandatory gate before this
path may be called evaluation-ready.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robomme_integration.training.framesamp_am_artifact import (
    KEY_TAP_STAGE,
    QUERY_TAP_STAGE,
    VALUE_TAP_STAGE,
    array_bundle_sha256,
)
from robomme_integration.training.framesamp_am_flax_overlay import (
    OFFICIAL_POLICY_GIT_SHA,
    PATCHED_HISTORY_GEMMA_SHA256,
)
from robomme_integration.training.framesamp_am_policy_overlay import (
    PATCHED_HISTORY_PI0_SHA256,
    verify_framesamp_am_policy_overlay,
)
from robomme_integration.training.framesamp_am_query_bank import (
    FIT_SPLIT,
    HELDOUT_SPLIT,
    ActionQueryBankPair,
    ActionQueryCaptureRequest,
    ActionQuerySamplingSpec,
    CapturedActionQueries,
    CapturedTeacherMemoryTaps,
    action_query_capture_requests,
    produce_action_query_bank,
)
from robomme_integration.training.upstream_framesamp_data import (
    TOKEN_BUDGET,
    FrameSampHistory,
)

CAPTURE_SCHEMA_VERSION = 1
ACTION_EXPERT_DEPTH = 18
ACTION_QUERY_HEADS = 4
ACTION_TOKENS = 20
MEMORY_KV_HEADS = 1
MEMORY_HEAD_DIM = 256
MEMORY_WIDTH = 1024
RELEASED_CHECKPOINT_SHA256 = "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
RELEASED_CHECKPOINT_STEP = "79999"
RELEASED_HISTORY_CONFIG = "perceptual-framesamp-modul.yaml"
NOISE_DISTRIBUTION = "standard_normal_v1"
ACTION_SAMPLER_ID = "pi05_teacher_sample_actions_10step_v1"
DIFFUSION_SCHEDULE_ID = "explicit_flow_times_v1"
_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _require_sha(value: object, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(character not in _HEX for character in value):
        allowed = "/".join(str(length) for length in lengths)
        raise ValueError(f"{label} must be a lowercase {allowed}-hex SHA")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return int(value)


def framesamp_history_sha256(history: FrameSampHistory) -> str:
    """Hash every byte that can affect the released perceptual memory encoder."""

    history.validate()
    return array_bundle_sha256(
        image=np.asarray(history.image),
        position=np.asarray(history.position),
        token_mask=np.asarray(history.token_mask),
        frame_indices=np.asarray(history.frame_indices),
        frame_mask=np.asarray(history.frame_mask),
    )


def framesamp_token_mask_sha256(history: FrameSampHistory) -> str:
    history.validate()
    return array_bundle_sha256(token_mask=np.asarray(history.token_mask))


def framesamp_frame_map_sha256(history: FrameSampHistory) -> str:
    history.validate()
    return array_bundle_sha256(
        frame_indices=np.asarray(history.frame_indices),
        frame_mask=np.asarray(history.frame_mask),
    )


@dataclasses.dataclass(frozen=True)
class FrameSampTeacherCaptureIdentity:
    """Scientific identity of one on-policy history at one causal replan cut."""

    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    policy_overlay_manifest_sha256: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    history_sha256: str
    token_mask_sha256: str
    frame_map_sha256: str
    valid_source_tokens: int
    patched_history_gemma_sha256: str = PATCHED_HISTORY_GEMMA_SHA256
    patched_history_pi0_sha256: str = PATCHED_HISTORY_PI0_SHA256
    schema_version: int = CAPTURE_SCHEMA_VERSION

    @classmethod
    def from_history(
        cls,
        history: FrameSampHistory,
        *,
        teacher_checkpoint_sha256: str,
        teacher_code_sha: str,
        policy_overlay_manifest_sha256: str,
        task_id: str,
        episode_id: str,
        causal_cut_step: int,
    ) -> "FrameSampTeacherCaptureIdentity":
        history.validate()
        result = cls(
            teacher_checkpoint_sha256=teacher_checkpoint_sha256,
            teacher_code_sha=teacher_code_sha,
            policy_overlay_manifest_sha256=policy_overlay_manifest_sha256,
            task_id=task_id,
            episode_id=episode_id,
            causal_cut_step=causal_cut_step,
            history_sha256=framesamp_history_sha256(history),
            token_mask_sha256=framesamp_token_mask_sha256(history),
            frame_map_sha256=framesamp_frame_map_sha256(history),
            valid_source_tokens=int(history.token_mask.sum()),
        )
        result.validate(history)
        return result

    def validate(self, history: FrameSampHistory | None = None) -> None:
        if _require_int(self.schema_version, label="schema_version") != CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported teacher-capture schema {self.schema_version}")
        _require_sha(self.teacher_checkpoint_sha256, label="teacher_checkpoint_sha256")
        _require_sha(self.teacher_code_sha, label="teacher_code_sha", lengths=(40, 64))
        _require_sha(self.policy_overlay_manifest_sha256, label="policy_overlay_manifest_sha256")
        _require_sha(self.patched_history_gemma_sha256, label="patched_history_gemma_sha256")
        _require_sha(self.patched_history_pi0_sha256, label="patched_history_pi0_sha256")
        if self.patched_history_gemma_sha256 != PATCHED_HISTORY_GEMMA_SHA256:
            raise ValueError("teacher capture is not pinned to the reviewed HistoryGemma overlay")
        if self.patched_history_pi0_sha256 != PATCHED_HISTORY_PI0_SHA256:
            raise ValueError("teacher capture is not pinned to the reviewed HistoryPi0 overlay")
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_sha(self.history_sha256, label="history_sha256")
        _require_sha(self.token_mask_sha256, label="token_mask_sha256")
        _require_sha(self.frame_map_sha256, label="frame_map_sha256")
        _require_int(self.valid_source_tokens, label="valid_source_tokens", minimum=1)
        if self.valid_source_tokens > TOKEN_BUDGET:
            raise ValueError(f"valid_source_tokens cannot exceed {TOKEN_BUDGET}")
        if history is not None:
            history.validate()
            actual = (
                framesamp_history_sha256(history),
                framesamp_token_mask_sha256(history),
                framesamp_frame_map_sha256(history),
                int(history.token_mask.sum()),
            )
            expected = (
                self.history_sha256,
                self.token_mask_sha256,
                self.frame_map_sha256,
                self.valid_source_tokens,
            )
            if actual != expected:
                raise ValueError(
                    "current on-policy FrameSamp history differs from the captured history; "
                    "produce a fresh artifact for this replan"
                )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        return _json_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "FrameSampTeacherCaptureIdentity":
        if not isinstance(value, dict) or set(value) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("teacher capture identity fields mismatch")
        result = cls(**value)
        result.validate()
        return result


@dataclasses.dataclass(frozen=True)
class FrameSampTeacherQueryPlan:
    """Layer-complete deterministic fit/held-out query schedule."""

    identity: FrameSampTeacherCaptureIdentity
    fit_specs: tuple[ActionQuerySamplingSpec, ...]
    heldout_specs: tuple[ActionQuerySamplingSpec, ...]

    def validate(self, history: FrameSampHistory | None = None) -> None:
        self.identity.validate(history)
        expected_layers = tuple(range(ACTION_EXPERT_DEPTH))
        for split, specs in ((FIT_SPLIT, self.fit_specs), (HELDOUT_SPLIT, self.heldout_specs)):
            if not isinstance(specs, tuple) or len(specs) != ACTION_EXPERT_DEPTH:
                raise ValueError(f"{split} query plan must contain exactly {ACTION_EXPERT_DEPTH} layer specs")
            if tuple(spec.layer_index for spec in specs) != expected_layers:
                raise ValueError(f"{split} query specs must be in exact layer 0..17 order")
            stochastic = None
            for spec in specs:
                spec.validate()
                if spec.split != split:
                    raise ValueError(f"{split} query plan contains a {spec.split} spec")
                expected_route = (
                    self.identity.teacher_checkpoint_sha256,
                    self.identity.teacher_code_sha,
                    self.identity.task_id,
                    self.identity.episode_id,
                    self.identity.causal_cut_step,
                )
                if spec.routing_identity()[:5] != expected_route:
                    raise ValueError(f"{split} query spec disagrees with capture identity")
                if spec.action_tokens_per_sample != ACTION_TOKENS:
                    raise ValueError("teacher query plan must retain all 20 action tokens")
                if spec.query_head_count != ACTION_QUERY_HEADS or spec.kv_head_index != 0:
                    raise ValueError("teacher query plan has wrong MemoryAttention head geometry")
                if spec.noise_distribution != NOISE_DISTRIBUTION:
                    raise ValueError(f"teacher query plan must use {NOISE_DISTRIBUTION}")
                if spec.action_sampler_id != ACTION_SAMPLER_ID:
                    raise ValueError(f"teacher query plan must use {ACTION_SAMPLER_ID}")
                if not all(0.0 <= float(value) <= 1.0 for value in spec.diffusion_timesteps):
                    raise ValueError("explicit diffusion timesteps must lie in [0,1]")
                current = spec.stochastic_identity()
                if stochastic is None:
                    stochastic = current
                elif current != stochastic:
                    raise ValueError(f"{split} layers do not share one paired stochastic schedule")
        if self.fit_specs[0].diffusion_timesteps != self.heldout_specs[0].diffusion_timesteps:
            raise ValueError("fit and held-out plans must cover the same diffusion timesteps")
        if self.fit_specs[0].split_seed == self.heldout_specs[0].split_seed:
            raise ValueError("fit and held-out plans must use different split seeds")


def build_framesamp_teacher_query_plan(
    identity: FrameSampTeacherCaptureIdentity,
    *,
    diffusion_timesteps: Sequence[float],
    fit_split_seed: int,
    heldout_split_seed: int,
    fit_noise_samples_per_timestep: int,
    heldout_noise_samples_per_timestep: int,
    action_samples_per_noise: int = 1,
) -> FrameSampTeacherQueryPlan:
    """Build the only supported E1 sampling plan for all 18 teacher layers."""

    identity.validate()
    times = tuple(float(value) for value in diffusion_timesteps)

    def specs(split: str, seed: int, noise_samples: int) -> tuple[ActionQuerySamplingSpec, ...]:
        return tuple(
            ActionQuerySamplingSpec(
                teacher_checkpoint_sha256=identity.teacher_checkpoint_sha256,
                teacher_code_sha=identity.teacher_code_sha,
                task_id=identity.task_id,
                episode_id=identity.episode_id,
                causal_cut_step=identity.causal_cut_step,
                layer_index=layer,
                split=split,
                split_seed=seed,
                diffusion_schedule_id=DIFFUSION_SCHEDULE_ID,
                diffusion_timesteps=times,
                noise_distribution=NOISE_DISTRIBUTION,
                noise_samples_per_timestep=noise_samples,
                action_sampler_id=ACTION_SAMPLER_ID,
                action_samples_per_noise=action_samples_per_noise,
                action_tokens_per_sample=ACTION_TOKENS,
            )
            for layer in range(ACTION_EXPERT_DEPTH)
        )

    result = FrameSampTeacherQueryPlan(
        identity=identity,
        fit_specs=specs(FIT_SPLIT, fit_split_seed, fit_noise_samples_per_timestep),
        heldout_specs=specs(HELDOUT_SPLIT, heldout_split_seed, heldout_noise_samples_per_timestep),
    )
    result.validate()
    return result


@dataclasses.dataclass(frozen=True)
class CapturedAllLayerTeacherTaps:
    """One split captured from the same history and stochastic request schedule.

    Queries are ``[request, layer, 4, 20, 256]``.  K/V retain the physical
    right-padded teacher buffer as ``[layer, 512, 1, 256]``; downstream fitting
    selects valid positions exclusively through ``FrameSampHistory.token_mask``.
    """

    capture_identity_sha256: str
    split: str
    canonical_spec_sha256: str
    canonical_request_sha256s: tuple[str, ...]
    queries_post_rope_pre_scale: np.ndarray
    keys_post_rope: np.ndarray
    values_post_projection: np.ndarray
    query_tap_stage: str = QUERY_TAP_STAGE
    key_tap_stage: str = KEY_TAP_STAGE
    value_tap_stage: str = VALUE_TAP_STAGE

    def validate(
        self,
        plan: FrameSampTeacherQueryPlan,
        history: FrameSampHistory,
    ) -> None:
        plan.validate(history)
        if self.split not in {FIT_SPLIT, HELDOUT_SPLIT}:
            raise ValueError("captured teacher split must be fit or heldout")
        spec = (plan.fit_specs if self.split == FIT_SPLIT else plan.heldout_specs)[0]
        requests = action_query_capture_requests(spec)
        if self.capture_identity_sha256 != plan.identity.sha256():
            raise ValueError("captured teacher tensors have the wrong history/provenance identity")
        if self.canonical_spec_sha256 != spec.sha256():
            raise ValueError("captured teacher tensors have the wrong canonical sampling spec")
        expected_requests = tuple(request.request_sha256 for request in requests)
        if self.canonical_request_sha256s != expected_requests:
            raise ValueError("captured teacher requests are stale, reordered, or incomplete")
        if (self.query_tap_stage, self.key_tap_stage, self.value_tap_stage) != (
            QUERY_TAP_STAGE,
            KEY_TAP_STAGE,
            VALUE_TAP_STAGE,
        ):
            raise ValueError("captured teacher Q/K/V tap stages are wrong")
        queries = np.asarray(self.queries_post_rope_pre_scale)
        keys = np.asarray(self.keys_post_rope)
        values = np.asarray(self.values_post_projection)
        expected_q = (len(requests), ACTION_EXPERT_DEPTH, ACTION_QUERY_HEADS, ACTION_TOKENS, MEMORY_HEAD_DIM)
        expected_kv = (ACTION_EXPERT_DEPTH, TOKEN_BUDGET, MEMORY_KV_HEADS, MEMORY_HEAD_DIM)
        if queries.shape != expected_q:
            raise ValueError(f"captured Q must be {expected_q}, got {queries.shape}")
        if keys.shape != expected_kv or values.shape != expected_kv:
            raise ValueError(f"captured K/V must each be {expected_kv}")
        for name, value in (("Q", queries), ("K", keys), ("V", values)):
            if not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f"captured {name} must use a NumPy floating storage dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"captured {name} contains non-finite values")

    def teacher_tap_sha256(self) -> str:
        return array_bundle_sha256(
            keys_post_rope=np.asarray(self.keys_post_rope),
            values_post_projection=np.asarray(self.values_post_projection),
        )


class AllLayerTeacherTapProvider(Protocol):
    def __call__(
        self,
        identity: FrameSampTeacherCaptureIdentity,
        canonical_spec: ActionQuerySamplingSpec,
        requests: tuple[ActionQueryCaptureRequest, ...],
    ) -> CapturedAllLayerTeacherTaps: ...


@dataclasses.dataclass(frozen=True)
class FrameSampTeacherCaptureReceipt:
    """Content identity of ephemeral Q/K/V arrays used to fit a layer stack."""

    capture_identity: FrameSampTeacherCaptureIdentity
    fit_bank_sha256s: tuple[str, ...]
    heldout_bank_sha256s: tuple[str, ...]
    per_layer_teacher_tap_sha256s: tuple[str, ...]
    teacher_tap_sha256: str
    fit_query_stack_sha256: str
    heldout_query_stack_sha256: str
    schema_version: int = CAPTURE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != CAPTURE_SCHEMA_VERSION:
            raise ValueError("unsupported teacher capture receipt schema")
        self.capture_identity.validate()
        for label, values in (
            ("fit_bank_sha256s", self.fit_bank_sha256s),
            ("heldout_bank_sha256s", self.heldout_bank_sha256s),
            ("per_layer_teacher_tap_sha256s", self.per_layer_teacher_tap_sha256s),
        ):
            if not isinstance(values, tuple) or len(values) != ACTION_EXPERT_DEPTH:
                raise ValueError(f"{label} must contain one SHA for every layer")
            for value in values:
                _require_sha(value, label=label)
        _require_sha(self.teacher_tap_sha256, label="teacher_tap_sha256")
        _require_sha(self.fit_query_stack_sha256, label="fit_query_stack_sha256")
        _require_sha(self.heldout_query_stack_sha256, label="heldout_query_stack_sha256")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = dataclasses.asdict(self)
        result["capture_identity"] = self.capture_identity.to_dict()
        result["fit_bank_sha256s"] = list(self.fit_bank_sha256s)
        result["heldout_bank_sha256s"] = list(self.heldout_bank_sha256s)
        result["per_layer_teacher_tap_sha256s"] = list(self.per_layer_teacher_tap_sha256s)
        return result

    def sha256(self) -> str:
        return _json_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "FrameSampTeacherCaptureReceipt":
        if not isinstance(value, dict) or set(value) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("teacher capture receipt fields mismatch")
        decoded = dict(value)
        decoded["capture_identity"] = FrameSampTeacherCaptureIdentity.from_dict(decoded["capture_identity"])
        for name in (
            "fit_bank_sha256s",
            "heldout_bank_sha256s",
            "per_layer_teacher_tap_sha256s",
        ):
            if not isinstance(decoded[name], list):
                raise ValueError(f"teacher capture receipt {name} must be a JSON list")
            decoded[name] = tuple(decoded[name])
        result = cls(**decoded)
        result.validate()
        return result


@dataclasses.dataclass(frozen=True)
class ProducedFrameSampTeacherStack:
    """Existing per-layer fitter inputs plus a full-history freshness guard."""

    plan: FrameSampTeacherQueryPlan
    banks: tuple[ActionQueryBankPair, ...]
    teacher_taps: tuple[CapturedTeacherMemoryTaps, ...]
    receipt: FrameSampTeacherCaptureReceipt

    def validate(self, current_history: FrameSampHistory) -> None:
        self.plan.validate(current_history)
        self.receipt.validate()
        if self.receipt.capture_identity != self.plan.identity:
            raise ValueError("teacher capture receipt and query plan identities differ")
        if len(self.banks) != ACTION_EXPERT_DEPTH or len(self.teacher_taps) != ACTION_EXPERT_DEPTH:
            raise ValueError("teacher stack must contain exactly 18 layers")
        for layer, (banks, taps) in enumerate(zip(self.banks, self.teacher_taps, strict=True)):
            banks.validate()
            taps.validate(banks.fit.spec)
            if banks.fit.spec.layer_index != layer or taps.layer_index != layer:
                raise ValueError(f"teacher stack layer order mismatch at {layer}")
        stack_tap_hash = array_bundle_sha256(
            keys_post_rope=np.stack([tap.keys_post_rope for tap in self.teacher_taps]),
            values_post_projection=np.stack([tap.values_post_projection for tap in self.teacher_taps]),
        )
        if stack_tap_hash != self.receipt.teacher_tap_sha256:
            raise ValueError("teacher stack K/V bytes differ from its capture receipt")
        per_layer_tap_hashes = tuple(
            array_bundle_sha256(
                keys_post_rope=np.asarray(tap.keys_post_rope),
                values_post_projection=np.asarray(tap.values_post_projection),
            )
            for tap in self.teacher_taps
        )
        if per_layer_tap_hashes != self.receipt.per_layer_teacher_tap_sha256s:
            raise ValueError("teacher stack per-layer K/V bytes differ from its capture receipt")
        if tuple(bank.fit.bank_sha256 for bank in self.banks) != self.receipt.fit_bank_sha256s:
            raise ValueError("teacher stack fit banks differ from its capture receipt")
        if tuple(bank.heldout.bank_sha256 for bank in self.banks) != self.receipt.heldout_bank_sha256s:
            raise ValueError("teacher stack held-out banks differ from its capture receipt")

    def layer(
        self,
        layer_index: int,
        *,
        current_history: FrameSampHistory,
    ) -> tuple[ActionQueryBankPair, CapturedTeacherMemoryTaps]:
        """Resolve a layer only after rechecking the current on-policy history."""

        self.validate(current_history)
        layer = _require_int(layer_index, label="layer_index")
        if layer >= ACTION_EXPERT_DEPTH:
            raise IndexError(f"layer_index must lie in 0..{ACTION_EXPERT_DEPTH - 1}")
        return self.banks[layer], self.teacher_taps[layer]

    def valid_memory_taps(
        self,
        layer_index: int,
        *,
        current_history: FrameSampHistory,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return only valid K/V and their original physical 0..511 positions."""

        _banks, taps = self.layer(layer_index, current_history=current_history)
        physical = np.flatnonzero(current_history.token_mask).astype(np.int32, copy=False)
        return (
            physical,
            np.ascontiguousarray(np.asarray(taps.keys_post_rope)[physical]),
            np.ascontiguousarray(np.asarray(taps.values_post_projection)[physical]),
        )


def _cached_layer_capture(
    spec: ActionQuerySamplingSpec,
    requests: tuple[ActionQueryCaptureRequest, ...],
    *,
    captured: CapturedAllLayerTeacherTaps,
    layer_index: int,
) -> CapturedActionQueries:
    expected_spec = spec
    expected_requests = action_query_capture_requests(expected_spec)
    if requests != expected_requests:
        raise ValueError("per-layer query producer received a different request schedule")
    return CapturedActionQueries(
        sampling_spec_sha256=spec.sha256(),
        request_sha256s=tuple(request.request_sha256 for request in requests),
        queries_by_request=np.ascontiguousarray(np.asarray(captured.queries_post_rope_pre_scale)[:, layer_index]),
        query_tap_stage=QUERY_TAP_STAGE,
    )


def produce_framesamp_teacher_stack(
    plan: FrameSampTeacherQueryPlan,
    history: FrameSampHistory,
    capture_provider: AllLayerTeacherTapProvider,
) -> ProducedFrameSampTeacherStack:
    """Capture each stochastic request once, then build all 18 layer banks."""

    plan.validate(history)
    if not callable(capture_provider):
        raise TypeError("capture_provider must be callable")
    fit_spec = plan.fit_specs[0]
    heldout_spec = plan.heldout_specs[0]
    fit_requests = action_query_capture_requests(fit_spec)
    heldout_requests = action_query_capture_requests(heldout_spec)
    fit_capture = capture_provider(plan.identity, fit_spec, fit_requests)
    heldout_capture = capture_provider(plan.identity, heldout_spec, heldout_requests)
    if not isinstance(fit_capture, CapturedAllLayerTeacherTaps) or not isinstance(
        heldout_capture, CapturedAllLayerTeacherTaps
    ):
        raise TypeError("all-layer capture provider returned the wrong type")
    fit_capture.validate(plan, history)
    heldout_capture.validate(plan, history)
    if fit_capture.teacher_tap_sha256() != heldout_capture.teacher_tap_sha256():
        raise ValueError(
            "teacher K/V changed between fit and held-out capture; observation/history mutated during production"
        )
    if not np.array_equal(fit_capture.keys_post_rope, heldout_capture.keys_post_rope) or not np.array_equal(
        fit_capture.values_post_projection, heldout_capture.values_post_projection
    ):
        raise ValueError("teacher K/V must be bit-identical across query-bank splits")

    pairs: list[ActionQueryBankPair] = []
    taps: list[CapturedTeacherMemoryTaps] = []
    for layer in range(ACTION_EXPERT_DEPTH):
        fit_bank = produce_action_query_bank(
            plan.fit_specs[layer],
            lambda spec, requests, layer=layer: _cached_layer_capture(
                spec, requests, captured=fit_capture, layer_index=layer
            ),
        )
        heldout_bank = produce_action_query_bank(
            plan.heldout_specs[layer],
            lambda spec, requests, layer=layer: _cached_layer_capture(
                spec, requests, captured=heldout_capture, layer_index=layer
            ),
        )
        pair = ActionQueryBankPair(fit=fit_bank, heldout=heldout_bank)
        pair.validate()
        pairs.append(pair)
        taps.append(
            CapturedTeacherMemoryTaps(
                teacher_checkpoint_sha256=plan.identity.teacher_checkpoint_sha256,
                teacher_code_sha=plan.identity.teacher_code_sha,
                task_id=plan.identity.task_id,
                episode_id=plan.identity.episode_id,
                causal_cut_step=plan.identity.causal_cut_step,
                layer_index=layer,
                kv_head_index=0,
                keys_post_rope=np.ascontiguousarray(fit_capture.keys_post_rope[layer, :, 0]),
                values_post_projection=np.ascontiguousarray(fit_capture.values_post_projection[layer, :, 0]),
            )
        )

    banks = tuple(pairs)
    teacher_taps = tuple(taps)
    receipt = FrameSampTeacherCaptureReceipt(
        capture_identity=plan.identity,
        fit_bank_sha256s=tuple(pair.fit.bank_sha256 for pair in banks),
        heldout_bank_sha256s=tuple(pair.heldout.bank_sha256 for pair in banks),
        per_layer_teacher_tap_sha256s=tuple(
            array_bundle_sha256(
                keys_post_rope=np.asarray(tap.keys_post_rope),
                values_post_projection=np.asarray(tap.values_post_projection),
            )
            for tap in teacher_taps
        ),
        teacher_tap_sha256=array_bundle_sha256(
            keys_post_rope=np.asarray(fit_capture.keys_post_rope)[:, :, 0],
            values_post_projection=np.asarray(fit_capture.values_post_projection)[:, :, 0],
        ),
        fit_query_stack_sha256=array_bundle_sha256(
            queries_post_rope_pre_scale=np.asarray(fit_capture.queries_post_rope_pre_scale)
        ),
        heldout_query_stack_sha256=array_bundle_sha256(
            queries_post_rope_pre_scale=np.asarray(heldout_capture.queries_post_rope_pre_scale)
        ),
    )
    result = ProducedFrameSampTeacherStack(
        plan=plan,
        banks=banks,
        teacher_taps=teacher_taps,
        receipt=receipt,
    )
    result.validate(history)
    return result


def derive_ordered_teacher_tap_stack_sha256(
    identity: FrameSampTeacherCaptureIdentity,
    *,
    layer_manifest_sha256s: Sequence[str],
    per_layer_teacher_tap_sha256s: Sequence[str],
) -> str:
    """Match the authenticated oracle server's ordered tap-stack digest.

    Artifact manifests do not exist until after fitting/sealing, so the
    producer emits per-layer tap SHAs first and combines them with the final
    manifest SHAs here.  An online history attestor must recapture current K/V,
    compute the same per-layer SHAs, and derive this digest for every replan.
    """

    identity.validate()
    manifests = tuple(layer_manifest_sha256s)
    taps = tuple(per_layer_teacher_tap_sha256s)
    if len(manifests) != ACTION_EXPERT_DEPTH or len(taps) != ACTION_EXPERT_DEPTH:
        raise ValueError("ordered teacher-tap stack requires exactly 18 manifest and tap SHAs")
    layers: list[dict[str, object]] = []
    for layer, (manifest_sha, tap_sha) in enumerate(zip(manifests, taps, strict=True)):
        layers.append(
            {
                "layer_index": layer,
                "manifest_sha256": _require_sha(manifest_sha, label="layer manifest SHA"),
                "teacher_tap_sha256": _require_sha(tap_sha, label="layer teacher-tap SHA"),
            }
        )
    value = {
        "kind": "robomme_framesamp_am_ordered_teacher_tap_stack_v1",
        "teacher_checkpoint_sha256": identity.teacher_checkpoint_sha256,
        "teacher_code_sha": identity.teacher_code_sha,
        "task_id": identity.task_id,
        "episode_id": identity.episode_id,
        "causal_cut_step": identity.causal_cut_step,
        "layers": layers,
    }
    # The authenticated server's canonical serializer includes one trailing
    # newline; retain it exactly so independently computed digests agree.
    return hashlib.sha256((_canonical_json(value) + "\n").encode()).hexdigest()


def write_teacher_capture_receipt(
    destination: str | Path,
    stack: ProducedFrameSampTeacherStack,
    *,
    current_history: FrameSampHistory,
) -> str:
    """Create one immutable, small receipt; large ephemeral Q banks are not copied."""

    stack.validate(current_history)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace teacher-capture receipt: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"teacher-capture receipt parent does not exist: {destination.parent}")
    payload = (_canonical_json(stack.receipt.to_dict()) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def load_teacher_capture_receipt(
    source: str | Path,
    *,
    expected_sha256: str,
) -> FrameSampTeacherCaptureReceipt:
    """Reopen an immutable receipt only through its exact file SHA."""

    expected_sha256 = _require_sha(expected_sha256, label="expected receipt SHA")
    source = Path(source).resolve(strict=True)
    payload = source.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"teacher-capture receipt SHA mismatch: expected {expected_sha256}, got {actual}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher-capture receipt is not valid UTF-8 JSON") from error
    return FrameSampTeacherCaptureReceipt.from_dict(value)


@dataclasses.dataclass(frozen=True)
class ExtractedTeacherForwardTaps:
    """Normalized arrays from one reviewed Linen ``framesamp_am_taps`` update."""

    queries_post_rope_pre_scale: np.ndarray  # [L,B,20,4,256]
    keys_post_rope: np.ndarray  # [L,B,512,1,256]
    values_post_projection: np.ndarray  # [L,B,512,1,256]

    def validate(self, *, expected_layers: int = ACTION_EXPERT_DEPTH) -> None:
        q = np.asarray(self.queries_post_rope_pre_scale)
        k = np.asarray(self.keys_post_rope)
        v = np.asarray(self.values_post_projection)
        if q.shape != (expected_layers, 1, ACTION_TOKENS, ACTION_QUERY_HEADS, MEMORY_HEAD_DIM):
            raise ValueError("overlay Q tap has wrong layer/batch/token/head geometry")
        if k.shape != (expected_layers, 1, TOKEN_BUDGET, MEMORY_KV_HEADS, MEMORY_HEAD_DIM):
            raise ValueError("overlay K tap has wrong layer/batch/memory/head geometry")
        if v.shape != k.shape:
            raise ValueError("overlay V tap must match overlay K tap")
        for label, value in (("Q", q), ("K", k), ("V", v)):
            as_float32 = value.astype(np.float32)
            if not np.isfinite(as_float32).all():
                raise ValueError(f"overlay {label} tap contains non-finite values")

    def normalized_float32(self) -> "ExtractedTeacherForwardTaps":
        return ExtractedTeacherForwardTaps(
            queries_post_rope_pre_scale=np.ascontiguousarray(
                np.asarray(self.queries_post_rope_pre_scale).astype(np.float32)
            ),
            keys_post_rope=np.ascontiguousarray(np.asarray(self.keys_post_rope).astype(np.float32)),
            values_post_projection=np.ascontiguousarray(np.asarray(self.values_post_projection).astype(np.float32)),
        )


_TAP_KEYS = frozenset(
    {
        "q_post_rope_pre_scale",
        "recent_k_post_rope",
        "recent_v_post_projection",
    }
)


def _unwrap_single_sow(value: object, *, label: str) -> np.ndarray:
    if not isinstance(value, tuple) or len(value) != 1:
        raise ValueError(f"{label} must be a fresh one-write Linen sow tuple; accumulated tap state is rejected")
    item = value[0]
    item = getattr(item, "value", item)
    return np.asarray(item)


def _find_tap_mapping(value: object, *, path: str = "root") -> Mapping[str, object]:
    if isinstance(value, Mapping):
        if _TAP_KEYS.issubset(value):
            return value
        matches: list[Mapping[str, object]] = []
        for key, child in value.items():
            try:
                matches.append(_find_tap_mapping(child, path=f"{path}.{key}"))
            except LookupError:
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("tap collection contains more than one MemoryAttention tap mapping")
    raise LookupError(f"no FrameSamp MemoryAttention taps below {path}")


def extract_scanned_framesamp_teacher_taps(
    mutable_collection: Mapping[str, object],
    *,
    expected_layers: int = ACTION_EXPERT_DEPTH,
) -> ExtractedTeacherForwardTaps:
    """Extract the current overlay's scan-stacked Q/K/V collection fail-closed."""

    if not isinstance(mutable_collection, Mapping):
        raise TypeError("mutable_collection must be a mapping returned by Linen apply")
    try:
        taps = _find_tap_mapping(mutable_collection)
    except LookupError as error:
        raise ValueError("mutable collection has no FrameSamp MemoryAttention taps") from error
    result = ExtractedTeacherForwardTaps(
        queries_post_rope_pre_scale=_unwrap_single_sow(taps["q_post_rope_pre_scale"], label="Q tap"),
        keys_post_rope=_unwrap_single_sow(taps["recent_k_post_rope"], label="K tap"),
        values_post_projection=_unwrap_single_sow(taps["recent_v_post_projection"], label="V tap"),
    )
    result.validate(expected_layers=expected_layers)
    return result


@dataclasses.dataclass(frozen=True)
class PreparedFrameSampTeacherForward:
    """Immutable prefill/memory context for explicit teacher suffix forwards."""

    model: Any
    observation: Any
    history: FrameSampHistory
    capture_identity: FrameSampTeacherCaptureIdentity
    prefix_mask: Any
    kv_cache: Any
    mem_seq: Any
    mem_mask: Any
    linen_variables: Mapping[str, object]
    make_attn_mask: Callable[..., Any]

    def validate(self) -> None:
        self.capture_identity.validate(self.history)
        if getattr(self.model, "integration_type", None) != "modulation":
            raise ValueError("teacher tap capture requires modulation integration")
        if getattr(self.model, "representation_type", None) != "perceptual":
            raise ValueError("teacher tap capture requires perceptual FrameSamp memory")
        if int(getattr(self.model, "action_horizon", -1)) != ACTION_TOKENS:
            raise ValueError("teacher tap capture requires the full 20-token action horizon")
        if np.asarray(self.prefix_mask).shape[0] != 1:
            raise ValueError("released teacher tap capture is restricted to B=1")
        if np.asarray(self.mem_seq).shape != (1, TOKEN_BUDGET, MEMORY_WIDTH):
            raise ValueError("prepared FrameSamp memory must be [1,512,1024]")
        if np.asarray(self.mem_mask).shape != (1, TOKEN_BUDGET):
            raise ValueError("prepared FrameSamp mask must be [1,512]")
        if not np.array_equal(np.asarray(self.mem_mask[0], dtype=np.bool_), self.history.token_mask):
            raise ValueError("prepared model memory mask differs from the bound on-policy history")
        if not isinstance(self.linen_variables, Mapping) or "params" not in self.linen_variables:
            raise ValueError("prepared teacher has no Linen parameter collection")
        if "framesamp_am_taps" in self.linen_variables:
            raise ValueError("prepared Linen variables contain stale tap state")
        if not callable(self.make_attn_mask):
            raise TypeError("make_attn_mask must be callable")


def _observation_matches_history(observation: Any, history: FrameSampHistory) -> None:
    """Prove that the exact current observation, not a teacher trajectory, is bound."""

    history.validate()
    required = ("static_image_emb", "static_pos_emb", "static_mask")
    if any(not hasattr(observation, name) for name in required):
        raise ValueError("observation lacks the released perceptual FrameSamp fields")
    image = np.asarray(observation.static_image_emb)
    position = np.asarray(observation.static_pos_emb)
    mask = np.asarray(observation.static_mask)
    if image.shape[0] != 1 or position.shape[0] != 1 or mask.shape != (1, TOKEN_BUDGET):
        raise ValueError("teacher observation must contain one fixed 512-token FrameSamp buffer")
    if not np.array_equal(image[0].astype(np.float32), history.image.astype(np.float32)):
        raise ValueError("observation image-history bytes differ from the current on-policy history")
    if not np.array_equal(position[0].astype(np.float32), history.position.astype(np.float32)):
        raise ValueError("observation position-history bytes differ from the current on-policy history")
    if not np.array_equal(mask[0].astype(np.bool_), history.token_mask):
        raise ValueError("observation token mask differs from the current on-policy history")


def prepare_framesamp_teacher_forward(
    model: Any,
    observation: Any,
    history: FrameSampHistory,
    capture_identity: FrameSampTeacherCaptureIdentity,
    *,
    preprocess_observation: Callable[..., Any],
    make_attn_mask: Callable[..., Any],
) -> PreparedFrameSampTeacherForward:
    """Prepare the exact released modulation prefill without mutating tap state.

    ``preprocess_observation`` and ``make_attn_mask`` must come from the staged
    policy overlay imported before the checkpoint is loaded.  They are
    explicit arguments so an official-checkout module cannot be substituted
    accidentally in a long-lived process.
    """

    capture_identity.validate(history)
    _observation_matches_history(observation, history)
    if not callable(preprocess_observation) or not callable(make_attn_mask):
        raise TypeError("staged preprocess_observation and make_attn_mask callables are required")
    if (
        getattr(model, "integration_type", None) != "modulation"
        or getattr(model, "representation_type", None) != "perceptual"
    ):
        raise ValueError("released teacher must be perceptual FrameSamp + modulation")
    if int(getattr(model, "action_horizon", -1)) != ACTION_TOKENS:
        raise ValueError("released teacher must use action horizon 20")

    import jax.numpy as jnp
    from flax.nnx.bridge import variables as bridge_variables

    prepared_observation = preprocess_observation(None, observation, train=False)
    _observation_matches_history(prepared_observation, history)
    prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = model.embed_prefix(prepared_observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )
    mem_seq, mem_mask, _, _, _ = model.embed_memory(prepared_observation)

    llm = model.PaliGemma.llm
    if not hasattr(llm, "module") or not hasattr(llm, "linen_attributes"):
        raise TypeError("released teacher LLM is not the expected Linen-to-NNX bridge")
    nnx_attrs = {name: getattr(llm, name) for name in llm.linen_attributes}
    variables = bridge_variables.nnx_attrs_to_linen_vars(nnx_attrs)
    # Mutable sow collections are never accepted as model state.  Captures use
    # pure Linen apply and receive a fresh collection on every invocation.
    variables = {name: value for name, value in variables.items() if name != "framesamp_am_taps"}
    result = PreparedFrameSampTeacherForward(
        model=model,
        observation=prepared_observation,
        history=history,
        capture_identity=capture_identity,
        prefix_mask=prefix_mask,
        kv_cache=kv_cache,
        mem_seq=mem_seq,
        mem_mask=mem_mask,
        linen_variables=variables,
        make_attn_mask=make_attn_mask,
    )
    result.validate()
    return result


def capture_prepared_framesamp_teacher_forward(
    prepared: PreparedFrameSampTeacherForward,
    noisy_actions: Any,
    diffusion_timestep: float,
    *,
    verify_capture_does_not_change_output: bool = False,
) -> tuple[Any, ExtractedTeacherForwardTaps]:
    """Run one explicit diffusion suffix and return velocity plus all Q/K/V.

    This helper deliberately bypasses ``sample_actions``'s opaque while-loop;
    the caller chooses the diffusion time and noisy action tensor.  It applies
    the staged Linen module functionally, so repeated captures cannot append to
    or contaminate the loaded checkpoint's NNX state.
    """

    prepared.validate()
    timestep = float(diffusion_timestep)
    if not np.isfinite(timestep) or not 0.0 <= timestep <= 1.0:
        raise ValueError("diffusion_timestep must be finite and lie in [0,1]")

    import jax.numpy as jnp

    model = prepared.model
    noisy_actions = jnp.asarray(noisy_actions)
    expected_actions = (1, ACTION_TOKENS, int(model.action_dim))
    if noisy_actions.shape != expected_actions:
        raise ValueError(f"noisy_actions must be {expected_actions}, got {noisy_actions.shape}")
    suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = model.embed_suffix(
        prepared.observation,
        noisy_actions,
        jnp.full((1,), timestep, dtype=jnp.float32),
    )
    suffix_attn_mask = prepared.make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = jnp.broadcast_to(
        prepared.prefix_mask[:, None, :],
        (1, suffix_tokens.shape[1], prepared.prefix_mask.shape[1]),
    )
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prepared.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    call_kwargs = {
        "mask": full_attn_mask,
        "positions": positions,
        "kv_cache": prepared.kv_cache,
        "adarms_cond": [None, adarms_cond],
        "mem_seq": [None, prepared.mem_seq],
        "mem_mask": [None, prepared.mem_mask],
    }
    llm = model.PaliGemma.llm
    captured, mutable = llm.module.apply(
        prepared.linen_variables,
        [None, suffix_tokens],
        **call_kwargs,
        capture_framesamp_am_taps=True,
        mutable=["framesamp_am_taps"],
    )
    captured_outputs, _ = captured
    suffix_out = captured_outputs[1]
    if suffix_out is None:
        raise RuntimeError("captured teacher suffix unexpectedly returned None")
    if verify_capture_does_not_change_output:
        baseline_outputs, _ = llm.module.apply(
            prepared.linen_variables,
            [None, suffix_tokens],
            **call_kwargs,
        )
        baseline_suffix = baseline_outputs[1]
        if not np.array_equal(np.asarray(baseline_suffix), np.asarray(suffix_out)):
            raise ValueError("enabling teacher taps changed the explicit suffix output")
    velocity = model.action_out_proj(suffix_out[:, -ACTION_TOKENS:])
    taps = extract_scanned_framesamp_teacher_taps(mutable)
    return velocity, taps


class SourcePinnedFrameSampTeacherProvider:
    """Reference, fail-closed provider for released-checkpoint smoke captures.

    It intentionally favors inspectability over throughput: each request first
    draws an action with the supplied 10-step teacher sampler, mixes it with an
    independently seeded Gaussian at the explicit diffusion time, then runs
    one tapped suffix.  A production batch/vmap optimization must preserve this
    exact request and receipt contract.
    """

    def __init__(
        self,
        prepared: PreparedFrameSampTeacherForward,
        raw_observation: Any,
        *,
        sample_actions: Callable[..., Any],
        verify_first_suffix_parity: bool = True,
    ) -> None:
        prepared.validate()
        if not callable(sample_actions):
            raise TypeError("sample_actions must be the staged, preferably module-jitted teacher sampler")
        self._prepared = prepared
        self._raw_observation = raw_observation
        self._sample_actions = sample_actions
        self._verify_first_suffix_parity = bool(verify_first_suffix_parity)

    def __call__(
        self,
        identity: FrameSampTeacherCaptureIdentity,
        canonical_spec: ActionQuerySamplingSpec,
        requests: tuple[ActionQueryCaptureRequest, ...],
    ) -> CapturedAllLayerTeacherTaps:
        identity.validate(self._prepared.history)
        if identity != self._prepared.capture_identity:
            raise ValueError("capture provider identity differs from its prepared on-policy history")
        canonical_spec.validate()
        if canonical_spec.layer_index != 0:
            raise ValueError("all-layer capture must be scheduled from canonical layer 0")
        if canonical_spec.noise_distribution != NOISE_DISTRIBUTION:
            raise ValueError("unsupported teacher noise distribution")
        if canonical_spec.action_sampler_id != ACTION_SAMPLER_ID:
            raise ValueError("unsupported teacher action sampler")
        if requests != action_query_capture_requests(canonical_spec):
            raise ValueError("teacher capture request schedule was changed")

        import jax
        import jax.numpy as jnp

        queries: list[np.ndarray] = []
        stable_k: np.ndarray | None = None
        stable_v: np.ndarray | None = None
        for ordinal, request in enumerate(requests):
            sampled_action = self._sample_actions(
                uint63_to_jax_key(request.action_seed),
                self._raw_observation,
                num_steps=10,
            )
            sampled_action = jnp.asarray(sampled_action)
            expected_shape = (1, ACTION_TOKENS, int(self._prepared.model.action_dim))
            if sampled_action.shape != expected_shape:
                raise ValueError(f"teacher action sampler returned {sampled_action.shape}, expected {expected_shape}")
            noise = jax.random.normal(
                uint63_to_jax_key(request.noise_seed),
                sampled_action.shape,
                dtype=sampled_action.dtype,
            )
            timestep = jnp.asarray(request.diffusion_timestep, dtype=sampled_action.dtype)
            noisy_action = timestep * noise + (1 - timestep) * sampled_action
            _velocity, tapped = capture_prepared_framesamp_teacher_forward(
                self._prepared,
                noisy_action,
                request.diffusion_timestep,
                verify_capture_does_not_change_output=(self._verify_first_suffix_parity and ordinal == 0),
            )
            tapped = tapped.normalized_float32()
            # [L,1,20,4,H] -> [L,4,20,H]
            queries.append(np.ascontiguousarray(np.transpose(tapped.queries_post_rope_pre_scale[:, 0], (0, 2, 1, 3))))
            current_k = np.ascontiguousarray(tapped.keys_post_rope[:, 0])
            current_v = np.ascontiguousarray(tapped.values_post_projection[:, 0])
            if stable_k is None:
                stable_k, stable_v = current_k, current_v
            elif not np.array_equal(stable_k, current_k) or not np.array_equal(stable_v, current_v):
                raise ValueError("teacher K/V changed across explicit diffusion requests")
        if stable_k is None or stable_v is None:  # Spec validation already requires requests.
            raise RuntimeError("teacher capture produced no requests")
        result = CapturedAllLayerTeacherTaps(
            capture_identity_sha256=identity.sha256(),
            split=canonical_spec.split,
            canonical_spec_sha256=canonical_spec.sha256(),
            canonical_request_sha256s=tuple(request.request_sha256 for request in requests),
            queries_post_rope_pre_scale=np.ascontiguousarray(np.stack(queries)),
            keys_post_rope=stable_k,
            values_post_projection=stable_v,
        )
        return result


def verify_source_pinned_released_teacher(
    *,
    policy_overlay: str | Path,
    expected_policy_overlay_manifest_sha256: str,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str = RELEASED_CHECKPOINT_SHA256,
) -> dict[str, object]:
    """Read-only verification for the exact released teacher and overlay."""

    _require_sha(expected_policy_overlay_manifest_sha256, label="expected overlay manifest SHA")
    _require_sha(expected_checkpoint_sha256, label="expected checkpoint SHA")
    if expected_checkpoint_sha256 != RELEASED_CHECKPOINT_SHA256:
        raise ValueError("E1 teacher taps are restricted to released perceptual-framesamp-modul 79999")
    overlay_manifest = verify_framesamp_am_policy_overlay(
        policy_overlay,
        expected_manifest_sha256=expected_policy_overlay_manifest_sha256,
    )
    if overlay_manifest["official_policy_git_sha"] != OFFICIAL_POLICY_GIT_SHA:
        raise ValueError("policy overlay is not based on the audited official Git commit")
    checkpoint = Path(checkpoint).resolve(strict=True)
    if checkpoint.name != RELEASED_CHECKPOINT_STEP:
        raise ValueError(f"released FrameSamp teacher checkpoint must be step {RELEASED_CHECKPOINT_STEP}")
    if not (checkpoint / "params").is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise FileNotFoundError("released FrameSamp teacher checkpoint tree is incomplete")
    parent = checkpoint.parent
    marker = parent / f".EXTRACTED-{expected_checkpoint_sha256}"
    if not marker.is_file():
        raise FileNotFoundError(f"released FrameSamp checkpoint extraction marker is absent: {marker}")
    history_config = parent / "history_config.txt"
    if not history_config.is_file() or history_config.read_text(encoding="utf-8").strip() != RELEASED_HISTORY_CONFIG:
        raise ValueError("released FrameSamp checkpoint history_config.txt is absent or wrong")
    return {
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": expected_checkpoint_sha256,
        "teacher_code_sha": OFFICIAL_POLICY_GIT_SHA,
        "policy_overlay": str(Path(policy_overlay).resolve(strict=True)),
        "policy_overlay_manifest_sha256": expected_policy_overlay_manifest_sha256,
        "patched_history_gemma_sha256": PATCHED_HISTORY_GEMMA_SHA256,
        "patched_history_pi0_sha256": PATCHED_HISTORY_PI0_SHA256,
        "status": "source_and_checkpoint_verified_capture_and_full_action_parity_not_run",
    }


def uint63_to_jax_key(seed: int) -> Any:
    """Map every deterministic uint63 request seed to a two-word JAX key."""

    seed = _require_int(seed, label="seed")
    if seed >= 1 << 63:
        raise ValueError("seed must fit in uint63")
    import jax

    low = np.uint32(seed & 0xFFFFFFFF)
    high = np.uint32(seed >> 32)
    return jax.random.fold_in(jax.random.PRNGKey(low), high)


def assert_full_action_parity(
    official_actions: np.ndarray,
    overlay_actions: np.ndarray,
) -> None:
    """Mandatory final smoke gate: released and overlaid samplers must be exact."""

    reference = np.asarray(official_actions)
    candidate = np.asarray(overlay_actions)
    if reference.shape != candidate.shape:
        raise ValueError("full action parity shape mismatch")
    if reference.dtype != candidate.dtype:
        raise ValueError("full action parity dtype mismatch")
    if not np.array_equal(reference, candidate):
        max_abs = float(np.max(np.abs(reference.astype(np.float64) - candidate.astype(np.float64))))
        raise ValueError(f"full released-checkpoint action parity failed (max_abs={max_abs:.6g})")


def released_checkpoint_smoke_command(
    *,
    policy_overlay: str | Path,
    overlay_manifest_sha256: str,
    fixture: str | Path,
    output: str | Path,
) -> str:
    """Return the explicit no-simulator smoke command expected from the runner.

    The runner itself is intentionally not claimed here: it needs a sealed
    on-policy observation fixture and an official reference action generated
    from that exact fixture.  This command makes those two missing inputs
    explicit rather than silently substituting an initial demonstration.
    """

    _require_sha(overlay_manifest_sha256, label="overlay_manifest_sha256")
    return " ".join(
        [
            "JAX_PLATFORMS=cuda",
            "python",
            "-m",
            "robomme_integration.training.framesamp_am_teacher_smoke",
            "--policy-overlay",
            str(Path(policy_overlay)),
            "--overlay-manifest-sha256",
            overlay_manifest_sha256,
            "--checkpoint",
            str(
                Path(
                    "/home/sarveshp/Research/TRI/robomme_eval/official_reference/checkpoints/"
                    "perceptual-framesamp-modul/79999"
                )
            ),
            "--fixture",
            str(Path(fixture)),
            "--output",
            str(Path(output)),
            "--require-full-action-parity",
        ]
    )


def capture_provider_from_arrays(
    captures: Mapping[str, CapturedAllLayerTeacherTaps],
) -> AllLayerTeacherTapProvider:
    """Small adapter useful for replaying already captured, content-bound taps."""

    def provider(
        identity: FrameSampTeacherCaptureIdentity,
        canonical_spec: ActionQuerySamplingSpec,
        requests: tuple[ActionQueryCaptureRequest, ...],
    ) -> CapturedAllLayerTeacherTaps:
        del identity, requests
        try:
            return captures[canonical_spec.split]
        except KeyError as error:
            raise ValueError(f"no captured teacher tensors for split {canonical_spec.split}") from error

    return provider
