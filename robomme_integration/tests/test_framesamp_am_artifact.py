from __future__ import annotations

import dataclasses

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
    QuantizationParityThresholds,
    attend_framesamp_am_runtime,
    load_framesamp_am_artifact,
    seal_framesamp_am_artifact,
)
from robomme_integration.training.framesamp_attention_matching import (
    fit_framesamp_attention_matching,
)
from robomme_integration.training.upstream_framesamp_data import (
    TOKEN_BUDGET,
    assemble_framesamp_history,
)

CHECKPOINT_SHA = "a" * 64
CODE_SHA = "b" * 40


def _case(seed: int = 33):
    rng = np.random.default_rng(seed)
    step_idx = 4
    history = assemble_framesamp_history(rng.normal(size=(step_idx + 1, 1, 64, 2)).astype(np.float32), step_idx)
    keys = rng.normal(size=(TOKEN_BUDGET, 6)).astype(np.float32)
    values = rng.normal(size=(TOKEN_BUDGET, 5)).astype(np.float32)
    fit_queries = rng.normal(size=(4, 7, 6)).astype(np.float32)
    heldout_queries = rng.normal(size=(4, 5, 6)).astype(np.float32)
    result = fit_framesamp_attention_matching(
        history,
        fit_queries,
        keys,
        values,
        16,
        layer_index=2,
        mass_ridge=1e-8,
        value_ridge=1e-8,
    )
    return rng, history, keys, values, fit_queries, heldout_queries, result


def _identity(manifest) -> ExpectedFrameSampAMIdentity:
    return ExpectedFrameSampAMIdentity(
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id="task_07",
        episode_id="seed_7_episode_3",
        causal_cut_step=4,
        layer_index=2,
        kv_head_index=0,
        requested_budget=16,
        manifest_sha256=manifest.scientific_sha256(),
    )


def _seal(path, *, storage_dtype="float32", parity_thresholds=QuantizationParityThresholds()):
    rng, history, keys, values, fit_queries, heldout_queries, result = _case()
    manifest = seal_framesamp_am_artifact(
        path,
        result,
        history,
        fit_queries,
        heldout_queries,
        keys,
        values,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id="task_07",
        episode_id="seed_7_episode_3",
        causal_cut_step=4,
        fit_query_bank_spec="diffusion_seed=11;action_samples=7",
        heldout_query_bank_spec="diffusion_seed=29;action_samples=5",
        storage_dtype=storage_dtype,
        parity_thresholds=parity_thresholds,
    )
    return rng, history, keys, values, manifest


def test_scientific_bundle_roundtrip_and_runtime_preserve_the_attention_contract(tmp_path):
    bundle = tmp_path / "layer_02_kv_00"
    rng, history, keys, values, manifest = _seal(bundle)
    loaded = load_framesamp_am_artifact(bundle, expected=_identity(manifest))

    assert manifest.scientific_sha256() == loaded.manifest.scientific_sha256()
    assert manifest.valid_source_tokens == int(history.token_mask.sum())
    assert manifest.physical_source_tokens == TOKEN_BUDGET
    assert manifest.logical_key_position_span == TOKEN_BUDGET
    assert manifest.query_position_offset == TOKEN_BUDGET
    assert manifest.query_tap_stage == QUERY_TAP_STAGE
    assert manifest.key_tap_stage == KEY_TAP_STAGE
    assert manifest.value_tap_stage == VALUE_TAP_STAGE
    assert manifest.storage_dtype == "float32"
    assert loaded.artifact.keys.dtype == np.float32
    assert loaded.artifact.values.dtype == np.float32
    assert loaded.artifact.beta_am.dtype == np.float32

    queries = rng.normal(size=(4, 3, 6)).astype(np.float32)
    recent_keys = np.empty((0, 6), dtype=np.float32)
    recent_values = np.empty((0, 5), dtype=np.float32)
    inputs = FrameSampAMRuntimeInputs(
        queries_post_rope_pre_scale=queries,
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
    actual_output, actual_mass = attend_framesamp_am_runtime(loaded, inputs)
    expected_output, expected_mass = scaled_dot_product_attention(
        queries.reshape(-1, 6),
        loaded.artifact.keys,
        loaded.artifact.values,
        beta_am=loaded.artifact.beta_am,
        scale=loaded.artifact.scale,
    )
    assert np.allclose(actual_output.reshape(-1, 5), expected_output)
    assert np.allclose(actual_mass.reshape(-1), expected_mass)

    compact_only = dataclasses.replace(
        inputs,
        recent_keys_post_rope=np.empty((0, 6), dtype=np.float32),
        recent_values_post_projection=np.empty((0, 5), dtype=np.float32),
        recent_physical_positions=np.empty(0, dtype=np.int32),
        recent_token_mask=np.empty(0, dtype=bool),
    )
    compact_only_output, compact_only_mass = attend_framesamp_am_runtime(loaded, compact_only)
    expected_compact_only_output, expected_compact_only_mass = scaled_dot_product_attention(
        queries.reshape(-1, 6),
        loaded.artifact.keys,
        loaded.artifact.values,
        beta_am=loaded.artifact.beta_am,
        scale=loaded.artifact.scale,
    )
    assert np.allclose(compact_only_output.reshape(-1, 5), expected_compact_only_output)
    assert np.allclose(compact_only_mass.reshape(-1), expected_compact_only_mass)

    with pytest.raises(ValueError, match="requires R=0"):
        attend_framesamp_am_runtime(
            loaded,
            dataclasses.replace(
                inputs,
                recent_keys_post_rope=np.zeros((1, 6), dtype=np.float32),
                recent_values_post_projection=np.zeros((1, 5), dtype=np.float32),
                recent_physical_positions=np.zeros(1, dtype=np.int32),
                recent_token_mask=np.ones(1, dtype=bool),
            ),
        )

    with pytest.raises(ValueError, match="must not be projected or RoPE"):
        attend_framesamp_am_runtime(
            loaded,
            dataclasses.replace(inputs, compact_key_operation="project_and_apply_rope"),
        )
    with pytest.raises(ValueError, match="offset 512"):
        attend_framesamp_am_runtime(
            loaded,
            dataclasses.replace(inputs, query_positions=np.arange(16, 19)),
        )
    with pytest.raises(ValueError, match="post-projection/RoPE"):
        attend_framesamp_am_runtime(
            loaded,
            dataclasses.replace(inputs, recent_key_tap_stage="pre_rope"),
        )


def test_native_bfloat16_payload_roundtrips_via_portable_uint16_bits(tmp_path):
    pytest.importorskip("ml_dtypes")
    bundle = tmp_path / "bf16"
    manifest = _seal(
        bundle,
        storage_dtype="bfloat16",
        parity_thresholds=QuantizationParityThresholds(
            output_rmse_increase=1,
            output_relative_l2_increase=1,
            log_mass_rmse_increase=1,
            relative_mass_rmse_increase=1,
        ),
    )[-1]
    loaded = load_framesamp_am_artifact(bundle, expected=_identity(manifest))
    assert loaded.manifest.payload_encoding == "uint16_bfloat16_bits"
    assert loaded.artifact.keys.dtype.name == "bfloat16"


def test_storage_and_loading_fail_closed_on_quantization_identity_and_payload_drift(tmp_path):
    rng, history, keys, values, fit_queries, heldout_queries, _ = _case(seed=71)
    # Identity compaction makes the float64 reference error exactly zero, so a
    # float16 payload cannot pass a zero-degradation held-out gate.
    identity_result = fit_framesamp_attention_matching(
        history,
        fit_queries,
        keys,
        values,
        TOKEN_BUDGET,
        layer_index=2,
    )
    with pytest.raises(ValueError, match="storage-quantization parity gate failed"):
        seal_framesamp_am_artifact(
            tmp_path / "rejected_float16",
            identity_result,
            history,
            fit_queries,
            heldout_queries,
            keys,
            values,
            teacher_checkpoint_sha256=CHECKPOINT_SHA,
            teacher_code_sha=CODE_SHA,
            task_id="task_07",
            episode_id="seed_7_episode_3",
            causal_cut_step=4,
            fit_query_bank_spec="diffusion_seed=11;action_samples=7",
            heldout_query_bank_spec="diffusion_seed=29;action_samples=5",
            storage_dtype="float16",
            parity_thresholds=QuantizationParityThresholds(
                output_rmse_increase=0,
                output_relative_l2_increase=0,
                log_mass_rmse_increase=0,
                relative_mass_rmse_increase=0,
            ),
        )

    bundle = tmp_path / "sealed"
    manifest = _seal(bundle)[-1]
    with pytest.raises(ValueError, match="identity mismatch for layer_index"):
        load_framesamp_am_artifact(
            bundle,
            expected=dataclasses.replace(_identity(manifest), layer_index=1),
        )

    payload = bundle / "payload.npz"
    encoded = bytearray(payload.read_bytes())
    encoded[len(encoded) // 2] ^= 1
    payload.write_bytes(encoded)
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        load_framesamp_am_artifact(bundle, expected=_identity(manifest))
