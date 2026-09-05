from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robomme_integration.training.attention_matching import scaled_dot_product_attention
from robomme_integration.training.framesamp_am_artifact import (
    COMPACT_KEY_OPERATION,
    KEY_TAP_STAGE,
    QUERY_TAP_STAGE,
    RECENT_MEMORY_KIND,
    VALUE_TAP_STAGE,
    ExpectedFrameSampAMIdentity,
    FrameSampAMRuntimeInputs,
    attend_framesamp_am_runtime,
    load_framesamp_am_artifact,
    seal_framesamp_am_artifact,
)
from robomme_integration.training.framesamp_am_jax import (
    PATCH_CONTRACT,
    attend_prepared_framesamp_am_layer,
    memory_attention_am_core,
    prepare_framesamp_am_layer,
    require_reviewed_model_patch,
)
from robomme_integration.training.framesamp_attention_matching import fit_framesamp_attention_matching
from robomme_integration.training.upstream_framesamp_data import TOKEN_BUDGET, assemble_framesamp_history

CHECKPOINT_SHA = "c" * 64
CODE_SHA = "d" * 40


def test_jitted_official_layout_core_matches_numpy_shared_denominator_reference():
    rng = np.random.default_rng(2207)
    batch, action_tokens, q_heads, head_dim, value_dim = 1, 5, 4, 8, 8
    compact_tokens, recent_tokens = 7, 3
    queries = rng.normal(size=(batch, action_tokens, q_heads, head_dim)).astype(np.float32)
    compact_keys = rng.normal(size=(batch, compact_tokens, 1, head_dim)).astype(np.float32)
    compact_values = rng.normal(size=(batch, compact_tokens, 1, value_dim)).astype(np.float32)
    compact_beta = rng.normal(scale=0.3, size=(batch, compact_tokens)).astype(np.float32)
    recent_keys = rng.normal(size=(batch, recent_tokens, 1, head_dim)).astype(np.float32)
    recent_values = rng.normal(size=(batch, recent_tokens, 1, value_dim)).astype(np.float32)
    recent_mask = np.array([[True, False, False]])
    scale = head_dim**-0.5

    compiled = jax.jit(
        functools.partial(
            memory_attention_am_core,
            scale=scale,
            query_position_offset=TOKEN_BUDGET,
        )
    )
    actual_output, actual_mass = compiled(
        jnp.asarray(queries),
        jnp.asarray(compact_keys),
        jnp.asarray(compact_values),
        jnp.asarray(compact_beta),
        jnp.asarray(recent_keys),
        jnp.asarray(recent_values),
        jnp.asarray(recent_mask),
    )

    expected_output, expected_mass = scaled_dot_product_attention(
        queries.reshape(-1, head_dim),
        np.concatenate([compact_keys[0, :, 0], recent_keys[0, recent_mask[0], 0]], axis=0),
        np.concatenate([compact_values[0, :, 0], recent_values[0, recent_mask[0], 0]], axis=0),
        beta_am=np.concatenate([compact_beta[0], np.zeros(int(recent_mask.sum()), dtype=np.float32)]),
        scale=scale,
    )
    assert np.allclose(np.asarray(actual_output).reshape(-1, value_dim), expected_output, atol=2e-6)
    assert np.allclose(np.asarray(actual_mass).reshape(-1), expected_mass, atol=2e-6)

    bf16_output, bf16_mass = compiled(
        jnp.asarray(queries, dtype=jnp.bfloat16),
        jnp.asarray(compact_keys, dtype=jnp.bfloat16),
        jnp.asarray(compact_values, dtype=jnp.bfloat16),
        jnp.asarray(compact_beta, dtype=jnp.float32),
        jnp.asarray(recent_keys, dtype=jnp.bfloat16),
        jnp.asarray(recent_values, dtype=jnp.bfloat16),
        jnp.asarray(recent_mask),
    )
    assert bf16_output.dtype == jnp.bfloat16
    assert bf16_mass.dtype == jnp.float32

    # The pure post-hoc oracle compresses the entire old memory and therefore
    # has no raw-recent block.  Preserve the same core and denominator contract
    # with explicit zero-row K/V arrays instead of inventing a dummy token.
    compact_only_output, compact_only_mass = compiled(
        jnp.asarray(queries),
        jnp.asarray(compact_keys),
        jnp.asarray(compact_values),
        jnp.asarray(compact_beta),
        jnp.empty((batch, 0, 1, head_dim), dtype=jnp.float32),
        jnp.empty((batch, 0, 1, value_dim), dtype=jnp.float32),
        jnp.empty((batch, 0), dtype=jnp.bool_),
    )
    expected_compact_only_output, expected_compact_only_mass = scaled_dot_product_attention(
        queries.reshape(-1, head_dim),
        compact_keys[0, :, 0],
        compact_values[0, :, 0],
        beta_am=compact_beta[0],
        scale=scale,
    )
    assert np.allclose(
        np.asarray(compact_only_output).reshape(-1, value_dim),
        expected_compact_only_output,
        atol=2e-6,
    )
    assert np.allclose(
        np.asarray(compact_only_mass).reshape(-1),
        expected_compact_only_mass,
        atol=2e-6,
    )

    # A separate compact-only softmax followed by any merge cannot reproduce
    # this reference; reject the most dangerous integration mistakes directly.
    with pytest.raises(ValueError, match="without projection or RoPE"):
        memory_attention_am_core(
            jnp.asarray(queries),
            jnp.asarray(compact_keys),
            jnp.asarray(compact_values),
            jnp.asarray(compact_beta),
            jnp.asarray(recent_keys),
            jnp.asarray(recent_values),
            jnp.asarray(recent_mask),
            scale=scale,
            query_position_offset=TOKEN_BUDGET,
            compact_key_operation="project_and_apply_rope",
        )
    with pytest.raises(ValueError, match="offset must remain 512"):
        memory_attention_am_core(
            jnp.asarray(queries),
            jnp.asarray(compact_keys),
            jnp.asarray(compact_values),
            jnp.asarray(compact_beta),
            jnp.asarray(recent_keys),
            jnp.asarray(recent_values),
            jnp.asarray(recent_mask),
            scale=scale,
            query_position_offset=compact_tokens + recent_tokens,
        )


def test_sealed_layer_reaches_jax_with_identity_dtype_position_and_source_gates(tmp_path):
    rng = np.random.default_rng(933)
    step_idx = 2
    history = assemble_framesamp_history(
        rng.normal(size=(step_idx + 1, 1, 64, 3)).astype(np.float32),
        step_idx,
    )
    keys = rng.normal(size=(TOKEN_BUDGET, 8)).astype(np.float32)
    values = rng.normal(size=(TOKEN_BUDGET, 8)).astype(np.float32)
    fit_queries = rng.normal(size=(4, 6, 8)).astype(np.float32)
    heldout_queries = rng.normal(size=(4, 4, 8)).astype(np.float32)
    result = fit_framesamp_attention_matching(
        history,
        fit_queries,
        keys,
        values,
        12,
        layer_index=3,
        mass_ridge=1e-8,
        value_ridge=1e-8,
    )
    bundle = tmp_path / "layer_03_kv_00"
    manifest = seal_framesamp_am_artifact(
        bundle,
        result,
        history,
        fit_queries,
        heldout_queries,
        keys,
        values,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id="task_03",
        episode_id="episode_09",
        causal_cut_step=step_idx,
        fit_query_bank_spec="seed=3;samples=6",
        heldout_query_bank_spec="seed=9;samples=4",
        storage_dtype="float32",
    )
    expected = ExpectedFrameSampAMIdentity(
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id="task_03",
        episode_id="episode_09",
        causal_cut_step=step_idx,
        layer_index=3,
        kv_head_index=0,
        requested_budget=12,
        manifest_sha256=manifest.scientific_sha256(),
    )
    loaded = load_framesamp_am_artifact(bundle, expected=expected)
    prepared = prepare_framesamp_am_layer(loaded, expected_layer_index=3, runtime_dtype="float32")

    queries_h_t = rng.normal(size=(4, 3, 8)).astype(np.float32)
    recent_keys = np.empty((0, 8), dtype=np.float32)
    recent_values = np.empty((0, 8), dtype=np.float32)
    numpy_inputs = FrameSampAMRuntimeInputs(
        queries_post_rope_pre_scale=queries_h_t,
        query_positions=np.arange(512, 515, dtype=np.int32),
        recent_keys_post_rope=recent_keys,
        recent_values_post_projection=recent_values,
        recent_physical_positions=np.empty(0, dtype=np.int32),
        recent_token_mask=np.empty(0, dtype=bool),
        query_tap_stage=QUERY_TAP_STAGE,
        recent_key_tap_stage=KEY_TAP_STAGE,
        recent_value_tap_stage=VALUE_TAP_STAGE,
        compact_key_operation=COMPACT_KEY_OPERATION,
        recent_memory_kind=RECENT_MEMORY_KIND,
    )
    expected_output, expected_mass = attend_framesamp_am_runtime(loaded, numpy_inputs)
    actual_output, actual_mass = attend_prepared_framesamp_am_layer(
        prepared,
        jnp.asarray(np.transpose(queries_h_t, (1, 0, 2))[None]),
        np.arange(512, 515, dtype=np.int32)[None],
        jnp.asarray(recent_keys[None, :, None, :]),
        jnp.asarray(recent_values[None, :, None, :]),
        np.empty((1, 0), dtype=np.int32),
        jnp.empty((1, 0), dtype=jnp.bool_),
        expected_layer_index=3,
    )
    assert np.allclose(np.asarray(actual_output)[0], np.transpose(expected_output, (1, 0, 2)), atol=2e-6)
    assert np.allclose(np.asarray(actual_mass)[0], np.transpose(expected_mass, (1, 0)), atol=2e-6)

    with pytest.raises(ValueError, match="requires R=0"):
        attend_prepared_framesamp_am_layer(
            prepared,
            jnp.asarray(np.transpose(queries_h_t, (1, 0, 2))[None]),
            np.arange(512, 515, dtype=np.int32)[None],
            jnp.zeros((1, 1, 1, 8), dtype=jnp.float32),
            jnp.zeros((1, 1, 1, 8), dtype=jnp.float32),
            np.zeros((1, 1), dtype=np.int32),
            jnp.ones((1, 1), dtype=jnp.bool_),
            expected_layer_index=3,
        )

    with pytest.raises(ValueError, match="storage dtype must equal runtime"):
        prepare_framesamp_am_layer(loaded, expected_layer_index=3, runtime_dtype="bfloat16")
    with pytest.raises(ValueError, match="exact 512-offset"):
        attend_prepared_framesamp_am_layer(
            prepared,
            jnp.asarray(np.transpose(queries_h_t, (1, 0, 2))[None]),
            np.arange(12, 15, dtype=np.int32)[None],
            jnp.asarray(recent_keys[None, :, None, :]),
            jnp.asarray(recent_values[None, :, None, :]),
            np.empty((1, 0), dtype=np.int32),
            jnp.empty((1, 0), dtype=jnp.bool_),
            expected_layer_index=3,
        )
    with pytest.raises(FileNotFoundError):
        require_reviewed_model_patch(tmp_path / "not_installed.py")
    drifted_source = tmp_path / "history_gemma.py"
    drifted_source.write_text("class MemoryAttention: pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source drifted"):
        PATCH_CONTRACT.validate_unmodified_source(drifted_source)
