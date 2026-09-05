"""Fail-closed JAX consumer for sealed FrameSamp Attention-Matching memory.

This is the numerical seam used by the reviewed isolated Flax source overlay in
``framesamp_am_flax_overlay``.  It is deliberately not presented as an
end-to-end policy integration: the overlay patches the scanned model module,
but pinned ``history_pi0.py`` remains unchanged until an authoritative dynamic
task/episode/cut artifact route exists at the jitted policy boundary.

1. In each scanned action-expert layer, tap Q immediately after RoPE and before
   the single ``head_dim**-0.5`` scale.
2. Project the exact teacher memory with that layer's existing ``kv_einsum``;
   tap V immediately after projection and K immediately after applying RoPE at
   the tokens' *original 0..511 physical teacher positions*.
3. Thread the sealed compact K/V/beta arrays with scan ``in_axes=0`` so layer
   ``l`` can consume only an artifact whose manifest says ``layer_index=l``.
4. Pass those taps to :func:`memory_attention_am_core`.  Compact K has already
   been projected and RoPE-applied: never pass it through RMSNorm, kv_einsum,
   or RoPE again.  Schema v2 seals compact-all/R=0; the masked recent block in
   the algebra is reserved for a future disjoint old/recent schema.
5. Generate action-query RoPE positions from the fixed logical offset 512,
   never from compact length M or recent length R.  Keep the existing
   ``out_einsum_mem`` and modulation path unchanged.

The pinned, unmodified source contract is ``robomme_policy_learning`` commit
``ecf086c...`` and ``history_gemma.py`` SHA256 ``a488208...``.  The reviewed
overlay hash is recorded below, while the separate policy-routing gate remains
closed.  This prevents model-module parity from being mistaken for a working
policy integration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from robomme_integration.training.framesamp_am_artifact import (
    COMPACT_KEY_OPERATION,
    KEY_TAP_STAGE,
    MEMORY_PARTITION_KIND,
    QUERY_TAP_STAGE,
    RECENT_MEMORY_KIND,
    VALUE_TAP_STAGE,
    LoadedFrameSampAMArtifact,
)
from robomme_integration.training.upstream_framesamp_data import TOKEN_BUDGET

OFFICIAL_POLICY_GIT_SHA = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
OFFICIAL_HISTORY_GEMMA_SHA256 = "a4882087a74b52b08a7a002a2a8bf7d64324af3ff05daf99c15a17f30bab60d1"
OFFICIAL_MEMORY_ATTENTION_CLASS = "mme_vla_suite.models.integration.history_gemma.MemoryAttention"

# Exact output of the SHA-gated isolated overlay.  This pins the reviewed model
# patch only; it does not authorize an AM evaluation while history_pi0 routing
# remains deliberately absent.
REVIEWED_PATCHED_HISTORY_GEMMA_SHA256: str | None = "8d5084e92374296af2bcf9dcff27195df7f02884ca3b3399e3a6289147ce270e"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class OfficialMemoryAttentionAMPatchContract:
    """Pinned source and tensor contract for the isolated Flax overlay."""

    policy_git_sha: str = OFFICIAL_POLICY_GIT_SHA
    base_source_sha256: str = OFFICIAL_HISTORY_GEMMA_SHA256
    module_class: str = OFFICIAL_MEMORY_ATTENTION_CLASS
    action_query_heads: int = 4
    memory_kv_heads: int = 1
    query_position_offset: int = TOKEN_BUDGET
    query_tap_stage: str = QUERY_TAP_STAGE
    key_tap_stage: str = KEY_TAP_STAGE
    value_tap_stage: str = VALUE_TAP_STAGE
    compact_key_operation: str = COMPACT_KEY_OPERATION
    recent_memory_kind: str = RECENT_MEMORY_KIND

    def validate_unmodified_source(self, source_file: str | Path) -> None:
        """Require the exact source against which the patch contract was read."""

        source_file = Path(source_file)
        if not source_file.is_file():
            raise FileNotFoundError(f"official MemoryAttention source is missing: {source_file}")
        actual = _sha256_file(source_file)
        if actual != self.base_source_sha256:
            raise ValueError(
                "official MemoryAttention source drifted from the audited base: "
                f"expected {self.base_source_sha256}, got {actual}"
            )


PATCH_CONTRACT = OfficialMemoryAttentionAMPatchContract()


def require_reviewed_model_patch(source_file: str | Path) -> None:
    """Fail until an exact reviewed model patch is installed and pinned."""

    if REVIEWED_PATCHED_HISTORY_GEMMA_SHA256 is None:
        raise RuntimeError(
            "FrameSamp-AM JAX math is validated, but the scanned MemoryAttention source patch is not installed; "
            "do not label a policy or evaluation as AM-enabled"
        )
    actual = _sha256_file(Path(source_file))
    if actual != REVIEWED_PATCHED_HISTORY_GEMMA_SHA256:
        raise ValueError(
            "installed MemoryAttention patch hash mismatch: "
            f"expected {REVIEWED_PATCHED_HISTORY_GEMMA_SHA256}, got {actual}"
        )


def _dtype_name(value: Any) -> str:
    return jnp.dtype(value).name


def _device_set(array: jax.Array) -> frozenset[jax.Device]:
    sharding = getattr(array, "sharding", None)
    devices = getattr(sharding, "device_set", None)
    if devices is None:
        raise ValueError("AM runtime array has no explicit JAX device/sharding placement")
    return frozenset(devices)


@dataclasses.dataclass(frozen=True)
class PreparedFrameSampAMLayer:
    """One verified layer artifact, resident on its intended JAX placement."""

    layer_index: int
    manifest_sha256: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    query_position_offset: int
    scale: float
    runtime_dtype: str
    requested_budget: int
    memory_partition_kind: str
    compact_keys_post_rope: jax.Array
    compact_values_post_projection: jax.Array
    compact_beta_am: jax.Array

    @property
    def device_set(self) -> frozenset[jax.Device]:
        return _device_set(self.compact_keys_post_rope)

    def validate(self) -> None:
        if self.query_position_offset != TOKEN_BUDGET:
            raise ValueError("prepared action-query offset must remain 512")
        if self.memory_partition_kind != MEMORY_PARTITION_KIND:
            raise ValueError("prepared JAX artifact has an unsupported memory partition")
        keys = self.compact_keys_post_rope
        values = self.compact_values_post_projection
        beta = self.compact_beta_am
        if keys.ndim != 4 or keys.shape[0] != 1 or keys.shape[2] != 1 or not keys.shape[1]:
            raise ValueError("prepared compact keys must be [1, M, 1 KV head, H]")
        if values.ndim != 4 or values.shape[:3] != keys.shape[:3]:
            raise ValueError("prepared compact values must be [1, M, 1 KV head, V]")
        if values.shape[-1] != keys.shape[-1]:
            raise ValueError("official MemoryAttention requires equal 256-wide K/V heads")
        if beta.shape != (1, keys.shape[1]) or beta.dtype != jnp.float32:
            raise ValueError("prepared beta_AM must be float32 [1, M]")
        if keys.shape[1] != self.requested_budget:
            raise ValueError(
                "JIT AM oracle requires effective compact tokens == requested budget; "
                "wait for enough valid history or implement a sealed compact-token mask"
            )
        expected_scale = 1.0 / math.sqrt(keys.shape[-1])
        if not math.isclose(self.scale, expected_scale, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("prepared attention scale is not the official single pre-scale")
        if _dtype_name(keys.dtype) != self.runtime_dtype or _dtype_name(values.dtype) != self.runtime_dtype:
            raise ValueError("prepared compact K/V dtype disagrees with the declared runtime dtype")
        placements = {_device_set(keys), _device_set(values), _device_set(beta)}
        if len(placements) != 1:
            raise ValueError("prepared compact K/V/beta do not share one JAX placement")


def prepare_framesamp_am_layer(
    loaded: LoadedFrameSampAMArtifact,
    *,
    expected_layer_index: int,
    runtime_dtype: str,
    device_or_sharding: Any | None = None,
) -> PreparedFrameSampAMLayer:
    """Verify and place one sealed layer artifact without hidden dtype casts.

    RoboMME's official action expert runs in bfloat16.  Consequently production
    artifacts must be sealed as bfloat16 (including their held-out quantization
    gate) before this function will place them beside bfloat16 runtime taps.
    Float32 remains supported for CPU/reference parity tests.
    """

    loaded.validate()
    manifest = loaded.manifest
    if manifest.layer_index != expected_layer_index:
        raise ValueError(f"AM artifact layer mismatch: expected {expected_layer_index}, got {manifest.layer_index}")
    runtime_dtype = _dtype_name(runtime_dtype)
    if runtime_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("AM runtime dtype must be bfloat16, float16, or float32")
    if manifest.storage_dtype != runtime_dtype:
        raise ValueError(
            "AM artifact storage dtype must equal runtime Q/K/V dtype so no unvalidated quantization occurs: "
            f"stored {manifest.storage_dtype}, runtime {runtime_dtype}"
        )
    if manifest.effective_budget != manifest.requested_budget:
        raise ValueError(
            "JIT AM oracle requires effective compact tokens == requested budget; "
            "variable early-cut shapes are not accepted"
        )

    def place(value: np.ndarray, *, dtype: Any | None = None) -> jax.Array:
        array = jnp.asarray(value, dtype=dtype)
        return jax.device_put(array, device_or_sharding) if device_or_sharding is not None else jax.device_put(array)

    prepared = PreparedFrameSampAMLayer(
        layer_index=manifest.layer_index,
        manifest_sha256=manifest.scientific_sha256(),
        task_id=manifest.task_id,
        episode_id=manifest.episode_id,
        causal_cut_step=manifest.causal_cut_step,
        query_position_offset=manifest.query_position_offset,
        scale=manifest.resolved_attention_scale,
        runtime_dtype=runtime_dtype,
        requested_budget=manifest.requested_budget,
        memory_partition_kind=manifest.memory_partition_kind,
        compact_keys_post_rope=place(loaded.artifact.keys)[None, :, None, :],
        compact_values_post_projection=place(loaded.artifact.values)[None, :, None, :],
        # beta is added to float32 logits.  This cast is exact from any sealed
        # storage dtype and is not another payload quantization.
        compact_beta_am=place(loaded.artifact.beta_am, dtype=jnp.float32)[None, :],
    )
    prepared.validate()
    return prepared


def _validate_core_contract(
    queries: jax.Array,
    compact_keys: jax.Array,
    compact_values: jax.Array,
    compact_beta: jax.Array,
    recent_keys: jax.Array,
    recent_values: jax.Array,
    recent_mask: jax.Array,
    *,
    scale: float,
    query_position_offset: int,
    query_tap_stage: str,
    compact_key_operation: str,
    recent_key_tap_stage: str,
    recent_value_tap_stage: str,
    recent_memory_kind: str,
) -> None:
    if query_position_offset != TOKEN_BUDGET:
        raise ValueError("action-query RoPE offset must remain 512 independent of M and R")
    if query_tap_stage != QUERY_TAP_STAGE:
        raise ValueError("queries must be tapped post-RoPE and before the single attention scale")
    if compact_key_operation != COMPACT_KEY_OPERATION:
        raise ValueError("compact keys must be consumed without projection or RoPE")
    if (recent_key_tap_stage, recent_value_tap_stage) != (KEY_TAP_STAGE, VALUE_TAP_STAGE):
        raise ValueError("recent keys/values must be post-RoPE/post-projection")
    if recent_memory_kind != RECENT_MEMORY_KIND:
        raise ValueError("recent memory must be raw and uncompressed")

    if queries.ndim != 4 or queries.shape[2] != 4:
        raise ValueError("queries must be [B, T, 4 Q heads, H]")
    if compact_keys.ndim != 4 or compact_keys.shape[2] != 1 or not compact_keys.shape[1]:
        raise ValueError("compact keys must be [B, M, 1 KV head, H]")
    if recent_keys.ndim != 4 or recent_keys.shape[2] != 1 or not recent_keys.shape[-1]:
        raise ValueError("recent keys must be [B, R, 1 KV head, H] with nonzero head width")
    if compact_values.ndim != 4 or recent_values.ndim != 4:
        raise ValueError("compact/recent values must be rank-4 projected tensors")
    if queries.shape[0] != compact_keys.shape[0] or queries.shape[0] != recent_keys.shape[0]:
        raise ValueError("AM runtime batch dimensions differ")
    if compact_values.shape[:3] != compact_keys.shape[:3] or recent_values.shape[:3] != recent_keys.shape[:3]:
        raise ValueError("AM K/V token or KV-head dimensions differ")
    if queries.shape[-1] != compact_keys.shape[-1] or queries.shape[-1] != recent_keys.shape[-1]:
        raise ValueError("AM Q/K head dimensions differ")
    if compact_values.shape[-1] != recent_values.shape[-1]:
        raise ValueError("AM compact/recent value dimensions differ")
    if compact_values.shape[-1] != queries.shape[-1]:
        raise ValueError("official MemoryAttention requires equal Q/K/V head dimensions")
    if compact_beta.shape != compact_keys.shape[:2] or compact_beta.dtype != jnp.float32:
        raise ValueError("compact beta_AM must be float32 [B, M]")
    if recent_mask.shape != recent_keys.shape[:2] or recent_mask.dtype != jnp.bool_:
        raise ValueError("recent token mask must be Boolean [B, R]")
    dtypes = {queries.dtype, compact_keys.dtype, compact_values.dtype, recent_keys.dtype, recent_values.dtype}
    if len(dtypes) != 1 or _dtype_name(queries.dtype) not in {"bfloat16", "float16", "float32"}:
        raise ValueError("AM Q/K/V must share one supported runtime dtype")
    expected_scale = 1.0 / math.sqrt(queries.shape[-1])
    if not math.isclose(float(scale), expected_scale, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("AM scale must be exactly one head_dim**-0.5 pre-scale")


def memory_attention_am_core(
    queries_post_rope_pre_scale: jax.Array,
    compact_keys_post_rope: jax.Array,
    compact_values_post_projection: jax.Array,
    compact_beta_am: jax.Array,
    recent_keys_post_rope: jax.Array,
    recent_values_post_projection: jax.Array,
    recent_token_mask: jax.Array,
    *,
    scale: float,
    query_position_offset: int,
    query_tap_stage: str = QUERY_TAP_STAGE,
    compact_key_operation: str = COMPACT_KEY_OPERATION,
    recent_key_tap_stage: str = KEY_TAP_STAGE,
    recent_value_tap_stage: str = VALUE_TAP_STAGE,
    recent_memory_kind: str = RECENT_MEMORY_KIND,
) -> tuple[jax.Array, jax.Array]:
    """Official-layout AM attention: compact old + raw recent, one softmax.

    Inputs use RoboMME's native ``[B,T,N,H]`` / ``[B,S,K,H]`` layouts.  The
    output is ``[B,T,4,V]`` ready for the existing ``out_einsum_mem``; returned
    log mass is float32 ``[B,T,4]`` for diagnostics.
    """

    _validate_core_contract(
        queries_post_rope_pre_scale,
        compact_keys_post_rope,
        compact_values_post_projection,
        compact_beta_am,
        recent_keys_post_rope,
        recent_values_post_projection,
        recent_token_mask,
        scale=scale,
        query_position_offset=query_position_offset,
        query_tap_stage=query_tap_stage,
        compact_key_operation=compact_key_operation,
        recent_key_tap_stage=recent_key_tap_stage,
        recent_value_tap_stage=recent_value_tap_stage,
        recent_memory_kind=recent_memory_kind,
    )
    keys = jnp.concatenate([compact_keys_post_rope, recent_keys_post_rope], axis=1)
    values = jnp.concatenate([compact_values_post_projection, recent_values_post_projection], axis=1)
    beta = jnp.concatenate(
        [compact_beta_am, jnp.zeros(recent_keys_post_rope.shape[:2], dtype=jnp.float32)],
        axis=1,
    )

    # Match official MemoryAttention exactly: scale Q once in model dtype, use
    # float32 accumulation/logits, cast probabilities back to model dtype, and
    # contract one KV head against four grouped query heads.
    scaled_queries = queries_post_rope_pre_scale * jnp.asarray(scale, dtype=queries_post_rope_pre_scale.dtype)
    grouped_queries = scaled_queries[:, :, None, :, :]
    logits = jnp.einsum(
        "BTKGH,BSKH->BKGTS",
        grouped_queries,
        keys,
        preferred_element_type=jnp.float32,
    )
    logits = logits + beta[:, None, None, None, :]
    combined_mask = jnp.concatenate(
        [jnp.ones(compact_keys_post_rope.shape[:2], dtype=jnp.bool_), recent_token_mask],
        axis=1,
    )
    masked_logits = jnp.where(combined_mask[:, None, None, None, :], logits, -2.3819763e38)
    log_mass = jax.nn.logsumexp(masked_logits, axis=-1)
    probabilities = jax.nn.softmax(masked_logits, axis=-1).astype(queries_post_rope_pre_scale.dtype)
    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probabilities, values)
    output = encoded.reshape(
        queries_post_rope_pre_scale.shape[0],
        queries_post_rope_pre_scale.shape[1],
        4,
        values.shape[-1],
    )
    return output, jnp.transpose(log_mass[:, 0], (0, 2, 1))


def attend_prepared_framesamp_am_layer(
    prepared: PreparedFrameSampAMLayer,
    queries_post_rope_pre_scale: jax.Array,
    query_positions: np.ndarray,
    recent_keys_post_rope: jax.Array,
    recent_values_post_projection: jax.Array,
    recent_physical_positions: np.ndarray,
    recent_token_mask: jax.Array,
    *,
    expected_layer_index: int,
) -> tuple[jax.Array, jax.Array]:
    """Eager preflight plus JAX execution for one sealed B=1 artifact.

    The production Flax patch should call :func:`memory_attention_am_core`
    directly after its static/source preflight; this wrapper intentionally does
    a host check of the actual query positions and array placements.
    """

    prepared.validate()
    if prepared.layer_index != expected_layer_index:
        raise ValueError(f"prepared AM layer mismatch: expected {expected_layer_index}, got {prepared.layer_index}")
    queries = jnp.asarray(queries_post_rope_pre_scale)
    recent_keys = jnp.asarray(recent_keys_post_rope)
    recent_values = jnp.asarray(recent_values_post_projection)
    recent_mask = jnp.asarray(recent_token_mask)
    if prepared.memory_partition_kind == MEMORY_PARTITION_KIND and recent_keys.shape[1] != 0:
        raise ValueError("v1 prepared artifact compacts all valid FrameSamp tokens and requires R=0")
    if queries.shape[0] != 1:
        raise ValueError("one sealed episode artifact may only be consumed with runtime batch B=1")
    positions = np.asarray(query_positions)
    expected_positions = np.arange(TOKEN_BUDGET, TOKEN_BUDGET + queries.shape[1], dtype=np.int64)[None, :]
    if positions.shape != expected_positions.shape or not np.array_equal(
        positions.astype(np.int64, copy=False), expected_positions
    ):
        raise ValueError("runtime query positions must be the exact 512-offset action sequence")
    if _dtype_name(queries.dtype) != prepared.runtime_dtype:
        raise ValueError("runtime query dtype disagrees with the sealed artifact dtype")
    physical_positions = np.asarray(recent_physical_positions)
    host_recent_mask = np.asarray(recent_token_mask)
    if physical_positions.shape != recent_keys.shape[:2]:
        raise ValueError("runtime recent positions must be [B, R]")
    if not np.issubdtype(physical_positions.dtype, np.integer):
        raise ValueError("runtime recent positions must be integer teacher slots")
    if host_recent_mask.shape != recent_keys.shape[:2] or host_recent_mask.dtype != np.bool_:
        raise ValueError("runtime recent token mask must be Boolean [B, R]")
    if host_recent_mask.shape[1] and np.any(np.diff(host_recent_mask.astype(np.int8), axis=1) > 0):
        raise ValueError("runtime recent mask must be a valid prefix followed by right padding")
    physical_positions_i64 = physical_positions.astype(np.int64, copy=False)
    for batch_positions, batch_mask in zip(physical_positions_i64, host_recent_mask, strict=True):
        valid_positions = batch_positions[batch_mask]
        if valid_positions.size and (
            np.any(valid_positions < 0)
            or np.any(valid_positions >= TOKEN_BUDGET)
            or np.any(np.diff(valid_positions) <= 0)
        ):
            raise ValueError("valid recent positions must be unique increasing teacher slots in 0..511")
        if np.any(batch_positions[~batch_mask] != 0):
            raise ValueError("right-padded recent positions must use canonical safe sentinel 0")
    placements = {
        prepared.device_set,
        _device_set(queries),
        _device_set(recent_keys),
        _device_set(recent_values),
        _device_set(recent_mask),
    }
    if len(placements) != 1:
        raise ValueError("runtime Q/recent K/V and compact artifact do not share one JAX placement")
    return memory_attention_am_core(
        queries,
        prepared.compact_keys_post_rope,
        prepared.compact_values_post_projection,
        prepared.compact_beta_am,
        recent_keys,
        recent_values,
        recent_mask,
        scale=prepared.scale,
        query_position_offset=prepared.query_position_offset,
    )
