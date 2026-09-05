from __future__ import annotations

import dataclasses
import json
import shutil

import numpy as np
import pytest

from robomme_integration.training.framesamp_am_artifact import (
    PAYLOAD_FILENAME,
    QUERY_TAP_STAGE,
    ExpectedFrameSampAMIdentity,
    QuantizationParityThresholds,
)
from robomme_integration.training.framesamp_am_index import (
    create_framesamp_am_trusted_index,
    load_framesamp_am_trusted_index,
)
from robomme_integration.training.framesamp_am_query_bank import (
    FIT_SPLIT,
    HELDOUT_SPLIT,
    ActionQuerySamplingSpec,
    CapturedActionQueries,
    CapturedTeacherMemoryTaps,
    action_query_capture_requests,
    fit_framesamp_am_from_query_banks,
    produce_fit_heldout_query_banks,
    seal_bound_framesamp_am_artifact,
)
from robomme_integration.training.upstream_framesamp_data import TOKEN_BUDGET, assemble_framesamp_history

CHECKPOINT_SHA = "a" * 64
CODE_SHA = "b" * 40


def _spec(split: str, seed: int) -> ActionQuerySamplingSpec:
    return ActionQuerySamplingSpec(
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id="task_07",
        episode_id="seed_7_episode_3",
        causal_cut_step=4,
        layer_index=2,
        split=split,
        split_seed=seed,
        diffusion_schedule_id="explicit_flow_times_v1",
        diffusion_timesteps=(0.2, 0.8),
        noise_distribution="standard_normal_v1",
        noise_samples_per_timestep=2 if split == FIT_SPLIT else 1,
        action_sampler_id="pi05_denoising_action_sample_v1",
        action_samples_per_noise=1,
        action_tokens_per_sample=20,
    )


def _deterministic_capture(spec, requests) -> CapturedActionQueries:
    query_sets = []
    for request in requests:
        # A policy tap implementation receives these same domain-separated
        # seeds.  The test seam uses them directly to prove replay stability.
        seed = request.noise_seed ^ request.action_seed
        query_sets.append(np.random.default_rng(seed).normal(size=(4, spec.action_tokens_per_sample, 6)))
    return CapturedActionQueries(
        sampling_spec_sha256=spec.sha256(),
        request_sha256s=tuple(request.request_sha256 for request in requests),
        queries_by_request=np.asarray(query_sets, dtype=np.float32),
        query_tap_stage=QUERY_TAP_STAGE,
    )


def _pair(*, fit_seed: int = 101, heldout_seed: int = 211):
    return produce_fit_heldout_query_banks(
        _spec(FIT_SPLIT, fit_seed),
        _spec(HELDOUT_SPLIT, heldout_seed),
        _deterministic_capture,
    )


def _seal(
    tmp_path,
    name: str,
    *,
    fit_seed: int = 101,
    heldout_seed: int = 211,
    tap_seed: int = 33,
):
    rng = np.random.default_rng(tap_seed)
    history = assemble_framesamp_history(rng.normal(size=(5, 1, 64, 2)).astype(np.float32), 4)
    pair = _pair(fit_seed=fit_seed, heldout_seed=heldout_seed)
    taps = CapturedTeacherMemoryTaps(
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=CODE_SHA,
        task_id="task_07",
        episode_id="seed_7_episode_3",
        causal_cut_step=4,
        layer_index=2,
        kv_head_index=0,
        keys_post_rope=rng.normal(size=(TOKEN_BUDGET, 6)).astype(np.float32),
        values_post_projection=rng.normal(size=(TOKEN_BUDGET, 6)).astype(np.float32),
    )
    bound = fit_framesamp_am_from_query_banks(history, pair, taps, 16, mass_ridge=1e-8, value_ridge=1e-8)
    destination = tmp_path / name
    manifest = seal_bound_framesamp_am_artifact(
        destination,
        bound,
        history,
        parity_thresholds=QuantizationParityThresholds(
            output_rmse_increase=1,
            output_relative_l2_increase=1,
            log_mass_rmse_increase=1,
            relative_mass_rmse_increase=1,
        ),
    )
    return destination, pair, manifest


def _identity(manifest) -> ExpectedFrameSampAMIdentity:
    return ExpectedFrameSampAMIdentity(
        teacher_checkpoint_sha256=manifest.teacher_checkpoint_sha256,
        teacher_code_sha=manifest.teacher_code_sha,
        task_id=manifest.task_id,
        episode_id=manifest.episode_id,
        causal_cut_step=manifest.causal_cut_step,
        layer_index=manifest.layer_index,
        kv_head_index=manifest.kv_head_index,
        requested_budget=manifest.requested_budget,
        manifest_sha256=manifest.scientific_sha256(),
    )


def test_query_bank_splits_are_disjoint_and_bit_reproducible():
    first = _pair()
    replay = _pair()

    assert first.fit.bank_sha256 == replay.fit.bank_sha256
    assert first.heldout.bank_sha256 == replay.heldout.bank_sha256
    assert np.array_equal(first.fit.queries_post_rope_pre_scale, replay.fit.queries_post_rope_pre_scale)
    assert np.array_equal(first.heldout.queries_post_rope_pre_scale, replay.heldout.queries_post_rope_pre_scale)
    assert {request.request_sha256 for request in first.fit.requests}.isdisjoint(
        request.request_sha256 for request in first.heldout.requests
    )
    assert first.fit.queries_sha256 != first.heldout.queries_sha256

    other_layer = action_query_capture_requests(dataclasses.replace(first.fit.spec, layer_index=9))
    assert [(request.noise_seed, request.action_seed) for request in first.fit.requests] == [
        (request.noise_seed, request.action_seed) for request in other_layer
    ]
    assert [request.request_sha256 for request in first.fit.requests] != [
        request.request_sha256 for request in other_layer
    ]

    with pytest.raises(ValueError, match="different split seeds"):
        produce_fit_heldout_query_banks(
            _spec(FIT_SPLIT, 5),
            _spec(HELDOUT_SPLIT, 5),
            _deterministic_capture,
        )
    with pytest.raises(ValueError, match="same diffusion timesteps"):
        produce_fit_heldout_query_banks(
            _spec(FIT_SPLIT, 5),
            dataclasses.replace(_spec(HELDOUT_SPLIT, 7), diffusion_timesteps=(0.1, 0.9)),
            _deterministic_capture,
        )
    with pytest.raises(ValueError, match="full 20-token action horizon"):
        dataclasses.replace(first.fit.spec, action_tokens_per_sample=16).validate()


def test_bound_fit_seals_actual_query_arrays_and_resolves_only_through_pinned_index(tmp_path):
    bundle, pair, manifest = _seal(tmp_path, "layer_02_budget_016")

    fit_binding = json.loads(manifest.fit_query_bank_spec)
    heldout_binding = json.loads(manifest.heldout_query_bank_spec)
    assert manifest.fit_query_bank_sha256 == pair.fit.queries_sha256
    assert manifest.heldout_query_bank_sha256 == pair.heldout.queries_sha256
    assert fit_binding["bank_sha256"] == pair.fit.bank_sha256
    assert heldout_binding["sampling_spec"] == pair.heldout.spec.to_dict()

    index_path = tmp_path / "trusted_index.json"
    index_sha = create_framesamp_am_trusted_index(index_path, [bundle])
    trusted = load_framesamp_am_trusted_index(index_path, expected_sha256=index_sha)
    loaded = trusted.resolve(_identity(manifest))
    assert loaded.manifest.scientific_sha256() == manifest.scientific_sha256()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        create_framesamp_am_trusted_index(index_path, [bundle])


def test_index_supports_explicit_variants_and_rejects_true_route_collisions_and_corruption(tmp_path):
    first_bundle, _, first_manifest = _seal(tmp_path, "first", fit_seed=101, heldout_seed=211)
    second_bundle, _, second_manifest = _seal(tmp_path, "second", fit_seed=103, heldout_seed=223)

    # Different query-bank variants at the same task/layer/budget are intentional
    # ablations and must coexist in one consolidated index.
    variants_path = tmp_path / "variants.json"
    variants_sha = create_framesamp_am_trusted_index(variants_path, [first_bundle, second_bundle])
    variants = load_framesamp_am_trusted_index(variants_path, expected_sha256=variants_sha)
    assert len(variants.index.records) == 2
    assert variants.resolve(_identity(first_manifest)).manifest == first_manifest
    assert variants.resolve(_identity(second_manifest)).manifest == second_manifest

    collision_bundle, _, _ = _seal(
        tmp_path,
        "collision",
        fit_seed=101,
        heldout_seed=211,
        tap_seed=44,
    )
    with pytest.raises(ValueError, match="routing collision"):
        create_framesamp_am_trusted_index(tmp_path / "collision.json", [first_bundle, collision_bundle])

    index_path = tmp_path / "trusted.json"
    index_sha = create_framesamp_am_trusted_index(index_path, [first_bundle])
    trusted = load_framesamp_am_trusted_index(index_path, expected_sha256=index_sha)
    with pytest.raises(ValueError, match="wrong routing identity"):
        trusted.resolve(dataclasses.replace(_identity(first_manifest), task_id="different_task"))

    corrupted = tmp_path / "corrupted.json"
    shutil.copyfile(index_path, corrupted)
    data = bytearray(corrupted.read_bytes())
    data[len(data) // 2] ^= 1
    corrupted.write_bytes(data)
    with pytest.raises(ValueError, match="index SHA256 mismatch"):
        load_framesamp_am_trusted_index(corrupted, expected_sha256=index_sha)

    (first_bundle / PAYLOAD_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing payload.npz"):
        load_framesamp_am_trusted_index(index_path, expected_sha256=index_sha)
