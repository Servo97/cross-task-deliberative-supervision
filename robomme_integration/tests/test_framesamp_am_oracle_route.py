from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from robomme_integration.training.framesamp_am_artifact import (
    QuantizationParityThresholds,
    seal_framesamp_am_artifact,
)
from robomme_integration.training.framesamp_am_index import create_framesamp_am_trusted_index
from robomme_integration.training.framesamp_am_oracle_route import (
    ACTION_EXPERT_DEPTH,
    OfflineFrameSampAMLayerPin,
    OfflineFrameSampAMStackRequest,
    create_offline_framesamp_am_stack_manifest,
    load_offline_framesamp_am_stack_manifest,
    resolve_offline_framesamp_am_oracle_inputs,
)
from robomme_integration.training.framesamp_attention_matching import fit_framesamp_attention_matching
from robomme_integration.training.upstream_framesamp_data import TOKEN_BUDGET, assemble_framesamp_history

CHECKPOINT_SHA = "7" * 64
CODE_SHA = "8" * 40
TASK_ID = "task_11"
EPISODE_ID = "seed_3_episode_5"
CAUSAL_CUT = 2
REQUESTED_BUDGET = 8


@pytest.fixture(scope="module")
def sealed_stack(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline_am_stack")
    rng = np.random.default_rng(8461)
    history = assemble_framesamp_history(
        rng.normal(size=(CAUSAL_CUT + 1, 1, 64, 3)).astype(np.float32),
        CAUSAL_CUT,
    )
    bundles = []
    manifests = []
    final_layer_taps = None
    thresholds = QuantizationParityThresholds(
        output_rmse_increase=1,
        output_relative_l2_increase=1,
        log_mass_rmse_increase=1,
        relative_mass_rmse_increase=1,
    )
    for layer in range(ACTION_EXPERT_DEPTH):
        keys = rng.normal(size=(TOKEN_BUDGET, 256)).astype(np.float32)
        values = rng.normal(size=(TOKEN_BUDGET, 256)).astype(np.float32)
        fit_queries = rng.normal(size=(4, 2, 256)).astype(np.float32)
        heldout_queries = rng.normal(size=(4, 1, 256)).astype(np.float32)
        result = fit_framesamp_attention_matching(
            history,
            fit_queries,
            keys,
            values,
            REQUESTED_BUDGET,
            layer_index=layer,
            fit_mass=False,
            value_ridge=1e-6,
        )
        bundle = root / f"layer_{layer:02d}"
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
            task_id=TASK_ID,
            episode_id=EPISODE_ID,
            causal_cut_step=CAUSAL_CUT,
            fit_query_bank_spec=f"split=fit;layer={layer};seed=101",
            heldout_query_bank_spec=f"split=heldout;layer={layer};seed=211",
            storage_dtype="float32",
            parity_thresholds=thresholds,
        )
        bundles.append(bundle)
        manifests.append(manifest)
        if layer == ACTION_EXPERT_DEPTH - 1:
            final_layer_taps = (keys, values, fit_queries, heldout_queries)

    # A valid but scientifically different layer-17 producer variant lives in
    # the same trusted index. Exact manifest pins permit it to coexist, while
    # the atomic stack contract must reject mixing it with output-only layers.
    assert final_layer_taps is not None
    keys, values, fit_queries, heldout_queries = final_layer_taps
    mixed_result = fit_framesamp_attention_matching(
        history,
        fit_queries,
        keys,
        values,
        REQUESTED_BUDGET,
        layer_index=ACTION_EXPERT_DEPTH - 1,
        fit_mass=True,
        mass_ridge=1e-6,
        value_ridge=1e-6,
    )
    mixed_bundle = root / "layer_17_output_plus_mass"
    mixed_manifest = seal_framesamp_am_artifact(
        mixed_bundle,
        mixed_result,
        history,
        fit_queries,
        heldout_queries,
        keys,
        values,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id=TASK_ID,
        episode_id=EPISODE_ID,
        causal_cut_step=CAUSAL_CUT,
        fit_query_bank_spec="split=fit;layer=17;seed=101",
        heldout_query_bank_spec="split=heldout;layer=17;seed=211",
        storage_dtype="float32",
        parity_thresholds=thresholds,
    )
    bundles.append(mixed_bundle)

    index_path = root / "trusted_index.json"
    index_sha = create_framesamp_am_trusted_index(index_path, bundles)
    # Deliberately reverse the request: receipt creation must resolve by the
    # explicit layer index and publish canonical layer0..17 order.
    pins = tuple(
        OfflineFrameSampAMLayerPin(layer_index=layer, manifest_sha256=manifests[layer].scientific_sha256())
        for layer in reversed(range(ACTION_EXPERT_DEPTH))
    )
    request = OfflineFrameSampAMStackRequest(
        trusted_index_sha256=index_sha,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id=TASK_ID,
        episode_id=EPISODE_ID,
        causal_cut_step=CAUSAL_CUT,
        requested_budget=REQUESTED_BUDGET,
        storage_dtype="float32",
        layer_pins=pins,
    )
    stack_path = root / "stack_manifest.json"
    stack_sha = create_offline_framesamp_am_stack_manifest(stack_path, index_path, request)
    return root, index_path, index_sha, stack_path, stack_sha, request, mixed_manifest


def test_create_once_stack_resolves_exact_dynamic_module_jit_inputs(sealed_stack):
    _, index_path, index_sha, stack_path, stack_sha, request, _ = sealed_stack
    stack = load_offline_framesamp_am_stack_manifest(stack_path, expected_sha256=stack_sha)
    assert [pin.layer_index for pin in stack.layer_pins] == list(range(ACTION_EXPERT_DEPTH))
    assert stack.fit_mass is False
    assert stack.fit_queries_per_head == 2
    assert stack.heldout_queries_per_head == 1
    assert stack.trusted_index_sha256 == index_sha

    cpu = jax.devices("cpu")[0]
    oracle = resolve_offline_framesamp_am_oracle_inputs(
        stack_path,
        expected_stack_manifest_sha256=stack_sha,
        trusted_index_path=index_path,
        expected_trusted_index_sha256=index_sha,
        active_policy_checkpoint_sha256=CHECKPOINT_SHA,
        active_model_dtype="float32",
        expected_device_platform="cpu",
        device_or_sharding=cpu,
    )
    dynamic = oracle.sample_actions_dynamic_inputs()
    assert set(dynamic) == {
        "framesamp_am_compact_k",
        "framesamp_am_compact_v",
        "framesamp_am_compact_beta",
        "framesamp_am_compact_mask",
        "framesamp_am_recent_positions",
        "framesamp_am_recent_mem_seq",
        "framesamp_am_recent_mem_mask",
    }
    assert dynamic["framesamp_am_compact_k"].shape == (18, 1, REQUESTED_BUDGET, 1, 256)
    assert dynamic["framesamp_am_compact_v"].shape == (18, 1, REQUESTED_BUDGET, 1, 256)
    assert dynamic["framesamp_am_compact_beta"].shape == (18, 1, REQUESTED_BUDGET)
    assert np.asarray(dynamic["framesamp_am_compact_mask"]).all()
    assert dynamic["framesamp_am_recent_positions"].shape == (1, 0)
    assert dynamic["framesamp_am_recent_mem_seq"].shape == (1, 0, 1024)
    assert dynamic["framesamp_am_recent_mem_mask"].shape == (1, 0)

    # All seven values remain dynamic JAX arguments; no model attribute or
    # Python route object must enter a compiled call.
    @jax.jit
    def compiled_probe(**values):
        return (
            values["framesamp_am_compact_k"][0, 0].sum()
            + values["framesamp_am_compact_v"][17, 0].sum()
            + values["framesamp_am_compact_beta"].sum()
            + values["framesamp_am_recent_mem_seq"].sum()
        )

    assert np.isfinite(np.asarray(compiled_probe(**dynamic)))
    oracle.assert_request_identity(task_id=TASK_ID, episode_id=EPISODE_ID, causal_cut_step=CAUSAL_CUT)
    with pytest.raises(ValueError, match="request route mismatch"):
        oracle.assert_request_identity(task_id=TASK_ID, episode_id=EPISODE_ID, causal_cut_step=CAUSAL_CUT + 1)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        create_offline_framesamp_am_stack_manifest(stack_path, index_path, request)


def test_stack_and_runtime_fail_closed_on_layer_trust_checkpoint_dtype_and_device(sealed_stack):
    root, index_path, index_sha, stack_path, stack_sha, request, mixed_manifest = sealed_stack
    with pytest.raises(ValueError, match="duplicate layer indices"):
        dataclasses.replace(
            request,
            layer_pins=request.layer_pins[:-1] + (request.layer_pins[0],),
        ).validate()
    with pytest.raises(ValueError, match="duplicate layer manifest SHAs"):
        ordered = tuple(sorted(request.layer_pins, key=lambda pin: pin.layer_index))
        dataclasses.replace(
            request,
            layer_pins=ordered[:-1] + (dataclasses.replace(ordered[-1], manifest_sha256=ordered[0].manifest_sha256),),
        ).validate()
    mixed_pins = tuple(
        dataclasses.replace(pin, manifest_sha256=mixed_manifest.scientific_sha256())
        if pin.layer_index == ACTION_EXPERT_DEPTH - 1
        else pin
        for pin in request.layer_pins
    )
    with pytest.raises(ValueError, match="common stack contract"):
        create_offline_framesamp_am_stack_manifest(
            root / "mixed_variant_stack.json",
            index_path,
            dataclasses.replace(request, layer_pins=mixed_pins),
        )
    with pytest.raises(ValueError, match="stack SHA256 mismatch"):
        load_offline_framesamp_am_stack_manifest(stack_path, expected_sha256="f" * 64)

    cpu = jax.devices("cpu")[0]
    common = dict(
        stack_manifest_path=stack_path,
        expected_stack_manifest_sha256=stack_sha,
        trusted_index_path=index_path,
        expected_trusted_index_sha256=index_sha,
        active_policy_checkpoint_sha256=CHECKPOINT_SHA,
        active_model_dtype="float32",
        expected_device_platform="cpu",
        device_or_sharding=cpu,
    )
    with pytest.raises(ValueError, match="active policy checkpoint"):
        resolve_offline_framesamp_am_oracle_inputs(**(common | {"active_policy_checkpoint_sha256": "e" * 64}))
    with pytest.raises(ValueError, match="model dtype"):
        resolve_offline_framesamp_am_oracle_inputs(**(common | {"active_model_dtype": "bfloat16"}))
    with pytest.raises(ValueError, match="device platform mismatch"):
        resolve_offline_framesamp_am_oracle_inputs(**(common | {"expected_device_platform": "tpu"}))
    with pytest.raises(ValueError, match="externally expected trusted index"):
        resolve_offline_framesamp_am_oracle_inputs(**(common | {"expected_trusted_index_sha256": "a" * 64}))
