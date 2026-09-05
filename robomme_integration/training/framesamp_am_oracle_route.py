"""Atomic offline routing for an 18-layer RoboMME FrameSamp-AM oracle.

This module is a host-side trust and placement boundary, not a policy patch.
One create-once stack receipt pins an externally trusted artifact index and the
exact manifest SHA for every action-expert layer in order 0..17.  Resolution
reopens all bundles through that index and emits only dynamic JAX arrays for an
explicit ``sample_actions`` call; it never writes arrays onto model attributes.

Schema v1 of this route intentionally supports only the artifact schema-v2
partition ``compact_all_valid_framesamp_tokens_no_recent_v1``.  Every layer
must have effective M == requested M and the raw-recent block is genuinely
empty.  Because FrameSamp memory changes as a rollout advances, a receipt is
specific to one task, episode, and causal replan cut and must never be reused at
a later cut.  Compact-demo + raw-live memory requires a future disjoint-
partition artifact schema.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from robomme_integration.training.attention_matching import MASS_SOLVER, MASS_SOLVER_DISABLED, VALUE_SOLVER
from robomme_integration.training.framesamp_am_artifact import (
    KEY_TAP_STAGE,
    MEMORY_PARTITION_KIND,
    QUERY_TAP_STAGE,
    VALUE_TAP_STAGE,
    ExpectedFrameSampAMIdentity,
    LoadedFrameSampAMArtifact,
)
from robomme_integration.training.framesamp_am_index import (
    LoadedFrameSampAMTrustedIndex,
    load_framesamp_am_trusted_index,
)
from robomme_integration.training.framesamp_am_jax import prepare_framesamp_am_layer

STACK_SCHEMA_VERSION = 1
STACK_KIND = "robomme_framesamp_am_offline_compact_all_stack"
ACTION_EXPERT_DEPTH = 18
RUNTIME_BATCH_SIZE = 1
MEMORY_KV_HEADS = 1
MEMORY_HEAD_DIM = 256
MEMORY_WIDTH = 1024
_HEX = frozenset("0123456789abcdef")


def _require_sha(value: object, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(character not in _HEX for character in value):
        raise ValueError(f"{label} must be a lowercase {'/'.join(map(str, lengths))}-hex SHA")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _target_devices(device_or_sharding: Any) -> frozenset[jax.Device]:
    if isinstance(device_or_sharding, jax.Device):
        return frozenset({device_or_sharding})
    devices = getattr(device_or_sharding, "device_set", None)
    if not devices:
        raise ValueError("an explicit JAX Device or Sharding with a nonempty device_set is required")
    return frozenset(devices)


def _array_devices(array: jax.Array) -> frozenset[jax.Device]:
    devices = getattr(getattr(array, "sharding", None), "device_set", None)
    if not devices:
        raise ValueError("resolved oracle array has no explicit JAX placement")
    return frozenset(devices)


@dataclasses.dataclass(frozen=True)
class OfflineFrameSampAMLayerPin:
    layer_index: int
    manifest_sha256: str

    def validate(self) -> None:
        layer = _require_int(self.layer_index, label="layer_index")
        if layer >= ACTION_EXPERT_DEPTH:
            raise ValueError(f"layer_index must be in 0..{ACTION_EXPERT_DEPTH - 1}")
        _require_sha(self.manifest_sha256, label="manifest_sha256")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "OfflineFrameSampAMLayerPin":
        if not isinstance(value, dict) or set(value) != {"layer_index", "manifest_sha256"}:
            raise ValueError("offline oracle layer pin fields mismatch")
        result = cls(**value)
        result.validate()
        return result


def _ordered_layer_pins(
    pins: tuple[OfflineFrameSampAMLayerPin, ...],
) -> tuple[OfflineFrameSampAMLayerPin, ...]:
    if not isinstance(pins, tuple):
        raise ValueError("layer pins must be an immutable tuple")
    for pin in pins:
        if not isinstance(pin, OfflineFrameSampAMLayerPin):
            raise ValueError("layer pins must contain OfflineFrameSampAMLayerPin values")
        pin.validate()
    layers = [pin.layer_index for pin in pins]
    if len(layers) != len(set(layers)):
        raise ValueError("offline oracle stack contains duplicate layer indices")
    missing = sorted(set(range(ACTION_EXPERT_DEPTH)) - set(layers))
    unexpected = sorted(set(layers) - set(range(ACTION_EXPERT_DEPTH)))
    if missing or unexpected or len(layers) != ACTION_EXPERT_DEPTH:
        raise ValueError(
            f"offline oracle stack must contain exactly layers 0..17; missing={missing}, extra={unexpected}"
        )
    manifests = [pin.manifest_sha256 for pin in pins]
    if len(manifests) != len(set(manifests)):
        raise ValueError("offline oracle stack contains duplicate layer manifest SHAs")
    return tuple(sorted(pins, key=lambda pin: pin.layer_index))


@dataclasses.dataclass(frozen=True)
class OfflineFrameSampAMStackRequest:
    """Externally supplied exact route used to create one stack receipt."""

    trusted_index_sha256: str
    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    requested_budget: int
    storage_dtype: str
    layer_pins: tuple[OfflineFrameSampAMLayerPin, ...]

    def validate(self) -> None:
        _require_sha(self.trusted_index_sha256, label="trusted_index_sha256")
        _require_sha(self.teacher_checkpoint_sha256, label="teacher_checkpoint_sha256")
        _require_sha(self.teacher_code_sha, label="teacher_code_sha", lengths=(40, 64))
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_int(self.requested_budget, label="requested_budget", minimum=1)
        if self.storage_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("storage_dtype must be bfloat16, float16, or float32")
        _ordered_layer_pins(self.layer_pins)


@dataclasses.dataclass(frozen=True)
class OfflineFrameSampAMStackManifest:
    """Create-once receipt for one complete causal-cut/layer stack."""

    trusted_index_sha256: str
    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    requested_budget: int
    storage_dtype: str
    resolved_attention_scale: float
    memory_partition_kind: str
    artifact_method: str
    fit_mass: bool
    mass_solver: str
    value_solver: str
    mass_ridge: float
    value_ridge: float
    fit_queries_per_head: int
    heldout_queries_per_head: int
    payload_encoding: str
    query_tap_stage: str
    key_tap_stage: str
    value_tap_stage: str
    token_mask_sha256: str
    frame_map_sha256: str
    valid_source_tokens: int
    layer_pins: tuple[OfflineFrameSampAMLayerPin, ...]
    schema_version: int = STACK_SCHEMA_VERSION
    kind: str = STACK_KIND

    def validate(self) -> None:
        if _require_int(self.schema_version, label="schema_version") != STACK_SCHEMA_VERSION:
            raise ValueError(f"unsupported offline oracle stack schema {self.schema_version}")
        if self.kind != STACK_KIND:
            raise ValueError("unsupported offline oracle stack kind")
        request = OfflineFrameSampAMStackRequest(
            trusted_index_sha256=self.trusted_index_sha256,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            teacher_code_sha=self.teacher_code_sha,
            task_id=self.task_id,
            episode_id=self.episode_id,
            causal_cut_step=self.causal_cut_step,
            requested_budget=self.requested_budget,
            storage_dtype=self.storage_dtype,
            layer_pins=self.layer_pins,
        )
        request.validate()
        if self.layer_pins != _ordered_layer_pins(self.layer_pins):
            raise ValueError("offline oracle receipt must list manifest SHAs in exact layer 0..17 order")
        if self.memory_partition_kind != MEMORY_PARTITION_KIND:
            raise ValueError("offline oracle receipt is not the schema-v2 compact-all/no-recent partition")
        _require_nonempty(self.artifact_method, label="artifact_method")
        if not isinstance(self.fit_mass, bool):
            raise ValueError("fit_mass must be Boolean")
        _require_nonempty(self.mass_solver, label="mass_solver")
        _require_nonempty(self.value_solver, label="value_solver")
        expected_mass_solver = MASS_SOLVER if self.fit_mass else MASS_SOLVER_DISABLED
        if self.mass_solver != expected_mass_solver or self.value_solver != VALUE_SOLVER:
            raise ValueError("offline oracle receipt solver identity disagrees with fit_mass")
        if not math.isfinite(self.mass_ridge) or self.mass_ridge < 0:
            raise ValueError("mass_ridge must be finite and nonnegative")
        if not math.isfinite(self.value_ridge) or self.value_ridge < 0:
            raise ValueError("value_ridge must be finite and nonnegative")
        _require_int(self.fit_queries_per_head, label="fit_queries_per_head", minimum=1)
        _require_int(self.heldout_queries_per_head, label="heldout_queries_per_head", minimum=1)
        _require_nonempty(self.payload_encoding, label="payload_encoding")
        if (self.query_tap_stage, self.key_tap_stage, self.value_tap_stage) != (
            QUERY_TAP_STAGE,
            KEY_TAP_STAGE,
            VALUE_TAP_STAGE,
        ):
            raise ValueError("offline oracle receipt Q/K/V tap contract mismatch")
        expected_scale = MEMORY_HEAD_DIM**-0.5
        if not math.isclose(self.resolved_attention_scale, expected_scale, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("offline oracle receipt attention scale is not 256**-0.5")
        _require_sha(self.token_mask_sha256, label="token_mask_sha256")
        _require_sha(self.frame_map_sha256, label="frame_map_sha256")
        _require_int(self.valid_source_tokens, label="valid_source_tokens", minimum=1)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = dataclasses.asdict(self)
        result["layer_pins"] = [pin.to_dict() for pin in self.layer_pins]
        return result

    @classmethod
    def from_dict(cls, value: object) -> "OfflineFrameSampAMStackManifest":
        if not isinstance(value, dict):
            raise ValueError("offline oracle stack manifest must be an object")
        names = {field.name for field in dataclasses.fields(cls)}
        if set(value) != names:
            raise ValueError("offline oracle stack manifest fields mismatch")
        decoded = dict(value)
        raw_pins = decoded["layer_pins"]
        if not isinstance(raw_pins, list):
            raise ValueError("offline oracle layer_pins must be a list")
        decoded["layer_pins"] = tuple(OfflineFrameSampAMLayerPin.from_dict(pin) for pin in raw_pins)
        result = cls(**decoded)
        result.validate()
        return result


def _expected_identity(
    route: OfflineFrameSampAMStackRequest | OfflineFrameSampAMStackManifest,
    pin: OfflineFrameSampAMLayerPin,
) -> ExpectedFrameSampAMIdentity:
    return ExpectedFrameSampAMIdentity(
        teacher_checkpoint_sha256=route.teacher_checkpoint_sha256,
        teacher_code_sha=route.teacher_code_sha,
        task_id=route.task_id,
        episode_id=route.episode_id,
        causal_cut_step=route.causal_cut_step,
        layer_index=pin.layer_index,
        kv_head_index=0,
        requested_budget=route.requested_budget,
        manifest_sha256=pin.manifest_sha256,
    )


def _resolve_layers(
    trusted: LoadedFrameSampAMTrustedIndex,
    route: OfflineFrameSampAMStackRequest | OfflineFrameSampAMStackManifest,
) -> tuple[LoadedFrameSampAMArtifact, ...]:
    pins = _ordered_layer_pins(route.layer_pins)
    loaded = tuple(trusted.resolve(_expected_identity(route, pin)) for pin in pins)
    first = loaded[0].manifest
    common = (
        first.teacher_checkpoint_sha256,
        first.teacher_code_sha,
        first.task_id,
        first.episode_id,
        first.causal_cut_step,
        first.requested_budget,
        first.storage_dtype,
        first.resolved_attention_scale,
        first.memory_partition_kind,
        first.artifact_method,
        first.fit_mass,
        first.mass_solver,
        first.value_solver,
        first.mass_ridge,
        first.value_ridge,
        first.fit_queries_per_head,
        first.heldout_queries_per_head,
        first.payload_encoding,
        first.query_tap_stage,
        first.key_tap_stage,
        first.value_tap_stage,
        first.token_mask_sha256,
        first.frame_map_sha256,
        first.valid_source_tokens,
    )
    expected_common = (
        route.teacher_checkpoint_sha256,
        route.teacher_code_sha,
        route.task_id,
        route.episode_id,
        route.causal_cut_step,
        route.requested_budget,
        route.storage_dtype,
        first.resolved_attention_scale,
        MEMORY_PARTITION_KIND,
        first.artifact_method,
        first.fit_mass,
        first.mass_solver,
        first.value_solver,
        first.mass_ridge,
        first.value_ridge,
        first.fit_queries_per_head,
        first.heldout_queries_per_head,
        first.payload_encoding,
        QUERY_TAP_STAGE,
        KEY_TAP_STAGE,
        VALUE_TAP_STAGE,
        first.token_mask_sha256,
        first.frame_map_sha256,
        first.valid_source_tokens,
    )
    if common != expected_common:
        raise ValueError("layer 0 artifact disagrees with the requested offline oracle route")
    source_map = loaded[0].source_physical_indices
    for layer_index, artifact in enumerate(loaded):
        manifest = artifact.manifest
        actual_common = (
            manifest.teacher_checkpoint_sha256,
            manifest.teacher_code_sha,
            manifest.task_id,
            manifest.episode_id,
            manifest.causal_cut_step,
            manifest.requested_budget,
            manifest.storage_dtype,
            manifest.resolved_attention_scale,
            manifest.memory_partition_kind,
            manifest.artifact_method,
            manifest.fit_mass,
            manifest.mass_solver,
            manifest.value_solver,
            manifest.mass_ridge,
            manifest.value_ridge,
            manifest.fit_queries_per_head,
            manifest.heldout_queries_per_head,
            manifest.payload_encoding,
            manifest.query_tap_stage,
            manifest.key_tap_stage,
            manifest.value_tap_stage,
            manifest.token_mask_sha256,
            manifest.frame_map_sha256,
            manifest.valid_source_tokens,
        )
        if actual_common != common:
            raise ValueError(f"offline oracle layer {layer_index} disagrees with the common stack contract")
        if manifest.layer_index != layer_index:
            raise ValueError(f"offline oracle layer order mismatch at index {layer_index}")
        if manifest.effective_budget != manifest.requested_budget:
            raise ValueError("schema-v2 offline oracle requires effective M == requested M; padding is not allowed")
        if manifest.key_dim != MEMORY_HEAD_DIM or manifest.value_dim != MEMORY_HEAD_DIM:
            raise ValueError("offline oracle compact K/V must each have official head dimension 256")
        if not np.array_equal(artifact.source_physical_indices, source_map):
            raise ValueError("offline oracle layers disagree on the causal FrameSamp source token map")
    return loaded


def create_offline_framesamp_am_stack_manifest(
    destination: str | Path,
    trusted_index_path: str | Path,
    request: OfflineFrameSampAMStackRequest,
) -> str:
    """Verify an exact 18-layer route and publish a non-replaceable receipt."""

    request.validate()
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace offline FrameSamp-AM stack manifest: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"offline stack-manifest parent does not exist: {destination.parent}")
    trusted = load_framesamp_am_trusted_index(
        trusted_index_path,
        expected_sha256=request.trusted_index_sha256,
        verify_artifacts=False,
    )
    loaded = _resolve_layers(trusted, request)
    first = loaded[0].manifest
    manifest = OfflineFrameSampAMStackManifest(
        trusted_index_sha256=request.trusted_index_sha256,
        teacher_checkpoint_sha256=request.teacher_checkpoint_sha256,
        teacher_code_sha=request.teacher_code_sha,
        task_id=request.task_id,
        episode_id=request.episode_id,
        causal_cut_step=request.causal_cut_step,
        requested_budget=request.requested_budget,
        storage_dtype=request.storage_dtype,
        resolved_attention_scale=first.resolved_attention_scale,
        memory_partition_kind=first.memory_partition_kind,
        artifact_method=first.artifact_method,
        fit_mass=first.fit_mass,
        mass_solver=first.mass_solver,
        value_solver=first.value_solver,
        mass_ridge=first.mass_ridge,
        value_ridge=first.value_ridge,
        fit_queries_per_head=first.fit_queries_per_head,
        heldout_queries_per_head=first.heldout_queries_per_head,
        payload_encoding=first.payload_encoding,
        query_tap_stage=first.query_tap_stage,
        key_tap_stage=first.key_tap_stage,
        value_tap_stage=first.value_tap_stage,
        token_mask_sha256=first.token_mask_sha256,
        frame_map_sha256=first.frame_map_sha256,
        valid_source_tokens=first.valid_source_tokens,
        layer_pins=_ordered_layer_pins(request.layer_pins),
    )
    payload = _canonical_json(manifest.to_dict())
    digest = _sha256_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def load_offline_framesamp_am_stack_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
) -> OfflineFrameSampAMStackManifest:
    """Load a stack receipt only under its external expected SHA."""

    expected_sha256 = _require_sha(expected_sha256, label="offline stack manifest SHA256")
    payload = Path(path).read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"offline FrameSamp-AM stack SHA256 mismatch: {actual} != {expected_sha256}")
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("offline FrameSamp-AM stack is not valid UTF-8 JSON") from error
    return OfflineFrameSampAMStackManifest.from_dict(raw)


@dataclasses.dataclass(frozen=True)
class OfflineFrameSampAMOracleInputs:
    """Host-verified dynamic arrays for one explicit ``sample_actions`` call."""

    stack_manifest_sha256: str
    trusted_index_sha256: str
    teacher_checkpoint_sha256: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    requested_budget: int
    model_dtype: str
    device_platform: str
    framesamp_am_compact_k: jax.Array
    framesamp_am_compact_v: jax.Array
    framesamp_am_compact_beta: jax.Array
    framesamp_am_compact_mask: jax.Array
    framesamp_am_recent_positions: jax.Array
    framesamp_am_recent_mem_seq: jax.Array
    framesamp_am_recent_mem_mask: jax.Array

    def sample_actions_dynamic_inputs(self) -> Mapping[str, jax.Array]:
        """Return arrays to thread as explicit kwargs/observation fields."""

        return {
            "framesamp_am_compact_k": self.framesamp_am_compact_k,
            "framesamp_am_compact_v": self.framesamp_am_compact_v,
            "framesamp_am_compact_beta": self.framesamp_am_compact_beta,
            "framesamp_am_compact_mask": self.framesamp_am_compact_mask,
            "framesamp_am_recent_positions": self.framesamp_am_recent_positions,
            "framesamp_am_recent_mem_seq": self.framesamp_am_recent_mem_seq,
            "framesamp_am_recent_mem_mask": self.framesamp_am_recent_mem_mask,
        }

    def assert_request_identity(self, *, task_id: str, episode_id: str, causal_cut_step: int) -> None:
        """Prevent reuse after the causal memory has advanced."""

        actual = (task_id, episode_id, causal_cut_step)
        expected = (self.task_id, self.episode_id, self.causal_cut_step)
        if actual != expected:
            raise ValueError(f"offline FrameSamp-AM request route mismatch: expected {expected!r}, got {actual!r}")

    def validate_runtime_binding(
        self,
        *,
        active_policy_checkpoint_sha256: str,
        active_model_dtype: str,
        device_or_sharding: Any,
    ) -> None:
        _require_sha(active_policy_checkpoint_sha256, label="active_policy_checkpoint_sha256")
        if active_policy_checkpoint_sha256 != self.teacher_checkpoint_sha256:
            raise ValueError("active policy checkpoint does not match the offline AM teacher checkpoint")
        if jnp.dtype(active_model_dtype).name != self.model_dtype:
            raise ValueError("active model dtype does not match the offline AM stack dtype")
        m = self.requested_budget
        if self.framesamp_am_compact_k.shape != (
            ACTION_EXPERT_DEPTH,
            RUNTIME_BATCH_SIZE,
            m,
            MEMORY_KV_HEADS,
            MEMORY_HEAD_DIM,
        ):
            raise ValueError("offline compact K does not have fixed [18,1,M,1,256] shape")
        if self.framesamp_am_compact_v.shape != self.framesamp_am_compact_k.shape:
            raise ValueError("offline compact V does not match compact K")
        if self.framesamp_am_compact_beta.shape != (ACTION_EXPERT_DEPTH, RUNTIME_BATCH_SIZE, m):
            raise ValueError("offline compact beta does not have fixed [18,1,M] shape")
        if self.framesamp_am_compact_beta.dtype != jnp.float32:
            raise ValueError("offline compact beta must be float32")
        if self.framesamp_am_compact_mask.shape != self.framesamp_am_compact_beta.shape:
            raise ValueError("offline compact mask does not have fixed [18,1,M] shape")
        if self.framesamp_am_compact_mask.dtype != jnp.bool_ or not bool(
            np.asarray(self.framesamp_am_compact_mask).all()
        ):
            raise ValueError("schema-v2 offline compact mask must be Boolean and all true")
        expected_empty = {
            "positions": ((RUNTIME_BATCH_SIZE, 0), jnp.int32),
            "mem_seq": ((RUNTIME_BATCH_SIZE, 0, MEMORY_WIDTH), jnp.dtype(self.model_dtype)),
            "mem_mask": ((RUNTIME_BATCH_SIZE, 0), jnp.bool_),
        }
        actual_empty = {
            "positions": self.framesamp_am_recent_positions,
            "mem_seq": self.framesamp_am_recent_mem_seq,
            "mem_mask": self.framesamp_am_recent_mem_mask,
        }
        for name, (shape, dtype) in expected_empty.items():
            value = actual_empty[name]
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"schema-v2 recent {name} must be a genuine R=0 array with shape {shape}")
        if self.framesamp_am_compact_k.dtype != jnp.dtype(self.model_dtype):
            raise ValueError("offline compact K dtype does not match the active model")
        if self.framesamp_am_compact_v.dtype != jnp.dtype(self.model_dtype):
            raise ValueError("offline compact V dtype does not match the active model")
        expected_devices = _target_devices(device_or_sharding)
        arrays = self.sample_actions_dynamic_inputs().values()
        if any(_array_devices(array) != expected_devices for array in arrays):
            raise ValueError("offline AM dynamic arrays are not resident on the required JAX device set")
        platforms = {device.platform for device in expected_devices}
        if platforms != {self.device_platform}:
            raise ValueError("offline AM device platform disagrees with its resolved runtime contract")


def resolve_offline_framesamp_am_oracle_inputs(
    stack_manifest_path: str | Path,
    *,
    expected_stack_manifest_sha256: str,
    trusted_index_path: str | Path,
    expected_trusted_index_sha256: str,
    active_policy_checkpoint_sha256: str,
    active_model_dtype: str,
    expected_device_platform: str,
    device_or_sharding: Any,
) -> OfflineFrameSampAMOracleInputs:
    """Resolve, verify, and place one fixed-shape batch-1 compact-all stack."""

    manifest = load_offline_framesamp_am_stack_manifest(
        stack_manifest_path,
        expected_sha256=expected_stack_manifest_sha256,
    )
    expected_trusted_index_sha256 = _require_sha(
        expected_trusted_index_sha256,
        label="expected_trusted_index_sha256",
    )
    if manifest.trusted_index_sha256 != expected_trusted_index_sha256:
        raise ValueError("offline stack receipt does not name the externally expected trusted index SHA")
    _require_sha(active_policy_checkpoint_sha256, label="active_policy_checkpoint_sha256")
    if active_policy_checkpoint_sha256 != manifest.teacher_checkpoint_sha256:
        raise ValueError("active policy checkpoint does not match the offline AM teacher checkpoint")
    model_dtype = jnp.dtype(active_model_dtype).name
    if model_dtype != manifest.storage_dtype:
        raise ValueError("active model dtype does not match the sealed offline AM storage dtype")
    target_devices = _target_devices(device_or_sharding)
    platforms = {device.platform for device in target_devices}
    if not isinstance(expected_device_platform, str) or platforms != {expected_device_platform}:
        raise ValueError(
            f"offline AM target device platform mismatch: expected {expected_device_platform!r}, got {sorted(platforms)}"
        )

    trusted = load_framesamp_am_trusted_index(
        trusted_index_path,
        expected_sha256=expected_trusted_index_sha256,
        verify_artifacts=False,
    )
    loaded = _resolve_layers(trusted, manifest)
    prepared = tuple(
        prepare_framesamp_am_layer(
            artifact,
            expected_layer_index=layer_index,
            runtime_dtype=model_dtype,
            device_or_sharding=device_or_sharding,
        )
        for layer_index, artifact in enumerate(loaded)
    )

    def place(value: jax.Array) -> jax.Array:
        return jax.device_put(value, device_or_sharding)

    compact_k = place(jnp.concatenate([layer.compact_keys_post_rope[None] for layer in prepared], axis=0))
    compact_v = place(jnp.concatenate([layer.compact_values_post_projection[None] for layer in prepared], axis=0))
    compact_beta = place(jnp.concatenate([layer.compact_beta_am[None] for layer in prepared], axis=0))
    compact_mask = place(jnp.ones(compact_beta.shape, dtype=jnp.bool_))
    recent_positions = place(jnp.empty((RUNTIME_BATCH_SIZE, 0), dtype=jnp.int32))
    recent_mem_seq = place(jnp.empty((RUNTIME_BATCH_SIZE, 0, MEMORY_WIDTH), dtype=jnp.dtype(model_dtype)))
    recent_mem_mask = place(jnp.empty((RUNTIME_BATCH_SIZE, 0), dtype=jnp.bool_))
    result = OfflineFrameSampAMOracleInputs(
        stack_manifest_sha256=expected_stack_manifest_sha256,
        trusted_index_sha256=expected_trusted_index_sha256,
        teacher_checkpoint_sha256=manifest.teacher_checkpoint_sha256,
        task_id=manifest.task_id,
        episode_id=manifest.episode_id,
        causal_cut_step=manifest.causal_cut_step,
        requested_budget=manifest.requested_budget,
        model_dtype=model_dtype,
        device_platform=expected_device_platform,
        framesamp_am_compact_k=compact_k,
        framesamp_am_compact_v=compact_v,
        framesamp_am_compact_beta=compact_beta,
        framesamp_am_compact_mask=compact_mask,
        framesamp_am_recent_positions=recent_positions,
        framesamp_am_recent_mem_seq=recent_mem_seq,
        framesamp_am_recent_mem_mask=recent_mem_mask,
    )
    result.validate_runtime_binding(
        active_policy_checkpoint_sha256=active_policy_checkpoint_sha256,
        active_model_dtype=model_dtype,
        device_or_sharding=device_or_sharding,
    )
    return result
