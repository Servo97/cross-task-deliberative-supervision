from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openpi.models.gemma import Config

from robomme_integration.eval.framesamp_am_oracle_server import (
    derive_teacher_tap_stack_sha256 as server_tap_stack_sha256,
)
from robomme_integration.training.framesamp_am_flax_overlay import (
    HISTORY_GEMMA_RELATIVE_PATH,
)
from robomme_integration.training.framesamp_am_policy_overlay import (
    stage_framesamp_am_policy_overlay,
)
from robomme_integration.training.framesamp_am_teacher_fixture_export import (
    ATTESTATION_SCOPE,
    seal_teacher_smoke_fixture,
)
from robomme_integration.training.framesamp_am_teacher_producer import (
    ACTION_EXPERT_DEPTH,
    FIT_SPLIT,
    HELDOUT_SPLIT,
    CapturedAllLayerTeacherTaps,
    FrameSampTeacherCaptureIdentity,
    build_framesamp_teacher_query_plan,
    capture_provider_from_arrays,
    derive_ordered_teacher_tap_stack_sha256,
    extract_scanned_framesamp_teacher_taps,
    framesamp_history_sha256,
    load_teacher_capture_receipt,
    produce_framesamp_teacher_stack,
    uint63_to_jax_key,
    write_teacher_capture_receipt,
)
from robomme_integration.training.framesamp_am_teacher_smoke import (
    PHASE_LOG_KIND,
    _load_observation_fixture,
    _log_phase,
)
from robomme_integration.training.upstream_framesamp_data import (
    TOKEN_BUDGET,
    assemble_framesamp_history,
)
from wsm_settings import ROBOMME_EVAL_ROOT

OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "ROBOMME_OFFICIAL_POLICY_ROOT",
        str(ROBOMME_EVAL_ROOT / "official_reference" / "robomme_policy_learning"),
    )
)
CHECKPOINT_SHA = "a" * 64
OVERLAY_SHA = "b" * 64


def _history(seed: int = 1):
    rng = np.random.default_rng(seed)
    return assemble_framesamp_history(
        rng.normal(size=(6, 1, 64, 3)).astype(np.float32),
        5,
    )


def _identity(history):
    return FrameSampTeacherCaptureIdentity.from_history(
        history,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha="c" * 40,
        policy_overlay_manifest_sha256=OVERLAY_SHA,
        task_id="task_03",
        episode_id="episode_07",
        causal_cut_step=5,
    )


def _captured(plan, split: str, *, q_seed: int):
    spec = (plan.fit_specs if split == FIT_SPLIT else plan.heldout_specs)[0]
    from robomme_integration.training.framesamp_am_query_bank import action_query_capture_requests

    requests = action_query_capture_requests(spec)
    rng = np.random.default_rng(q_seed)
    # K/V deliberately depend only on the bound history, not on split/request.
    kv_rng = np.random.default_rng(91)
    keys = kv_rng.normal(size=(ACTION_EXPERT_DEPTH, TOKEN_BUDGET, 1, 256)).astype(np.float32)
    values = kv_rng.normal(size=(ACTION_EXPERT_DEPTH, TOKEN_BUDGET, 1, 256)).astype(np.float32)
    return CapturedAllLayerTeacherTaps(
        capture_identity_sha256=plan.identity.sha256(),
        split=split,
        canonical_spec_sha256=spec.sha256(),
        canonical_request_sha256s=tuple(request.request_sha256 for request in requests),
        queries_post_rope_pre_scale=rng.normal(size=(len(requests), ACTION_EXPERT_DEPTH, 4, 20, 256)).astype(
            np.float32
        ),
        keys_post_rope=keys,
        values_post_projection=values,
    )


def test_all_layer_producer_binds_actual_history_and_reuses_existing_bank_api(tmp_path):
    history = _history()
    identity = _identity(history)
    plan = build_framesamp_teacher_query_plan(
        identity,
        diffusion_timesteps=(0.25, 0.75),
        fit_split_seed=101,
        heldout_split_seed=211,
        fit_noise_samples_per_timestep=1,
        heldout_noise_samples_per_timestep=1,
    )
    fit = _captured(plan, FIT_SPLIT, q_seed=3)
    heldout = _captured(plan, HELDOUT_SPLIT, q_seed=5)
    stack = produce_framesamp_teacher_stack(
        plan,
        history,
        capture_provider_from_arrays({FIT_SPLIT: fit, HELDOUT_SPLIT: heldout}),
    )

    assert len(stack.banks) == len(stack.teacher_taps) == ACTION_EXPERT_DEPTH
    assert stack.banks[0].fit.queries_post_rope_pre_scale.shape == (4, 40, 256)
    assert stack.teacher_taps[17].keys_post_rope.shape == (TOKEN_BUDGET, 256)
    assert len(stack.receipt.per_layer_teacher_tap_sha256s) == ACTION_EXPERT_DEPTH
    assert stack.layer(7, current_history=history)[0].fit.spec.layer_index == 7
    physical, valid_k, valid_v = stack.valid_memory_taps(7, current_history=history)
    assert physical.shape == (int(history.token_mask.sum()),)
    assert valid_k.shape == valid_v.shape == (int(history.token_mask.sum()), 256)

    receipt = tmp_path / "capture.receipt"
    digest = write_teacher_capture_receipt(receipt, stack, current_history=history)
    assert len(digest) == 64 and receipt.stat().st_size > 100
    assert load_teacher_capture_receipt(receipt, expected_sha256=digest) == stack.receipt
    with pytest.raises(ValueError, match="receipt SHA mismatch"):
        load_teacher_capture_receipt(receipt, expected_sha256="f" * 64)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_teacher_capture_receipt(receipt, stack, current_history=history)

    # Same task/episode/cut and even the same mask/frame map is insufficient:
    # an on-policy visual-history divergence must force a fresh capture.
    changed = dataclasses.replace(history, image=history.image.copy())
    changed.image[0, 0] += 1
    changed.validate()
    assert framesamp_history_sha256(changed) != identity.history_sha256
    with pytest.raises(ValueError, match="fresh artifact"):
        stack.layer(0, current_history=changed)


def test_fit_and_heldout_capture_must_share_exact_current_teacher_kv():
    history = _history()
    plan = build_framesamp_teacher_query_plan(
        _identity(history),
        diffusion_timesteps=(0.5,),
        fit_split_seed=3,
        heldout_split_seed=7,
        fit_noise_samples_per_timestep=1,
        heldout_noise_samples_per_timestep=1,
    )
    fit = _captured(plan, FIT_SPLIT, q_seed=1)
    heldout = _captured(plan, HELDOUT_SPLIT, q_seed=2)
    heldout.values_post_projection[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="K/V changed"):
        produce_framesamp_teacher_stack(
            plan,
            history,
            capture_provider_from_arrays({FIT_SPLIT: fit, HELDOUT_SPLIT: heldout}),
        )


def test_ordered_tap_digest_matches_authenticated_server_canonicalization(monkeypatch):
    history = _history()
    identity = _identity(history)
    manifests = tuple(hashlib.sha256(f"manifest-{i}".encode()).hexdigest() for i in range(18))
    taps = tuple(hashlib.sha256(f"tap-{i}".encode()).hexdigest() for i in range(18))
    expected = derive_ordered_teacher_tap_stack_sha256(
        identity,
        layer_manifest_sha256s=manifests,
        per_layer_teacher_tap_sha256s=taps,
    )

    # Exercise the server helper's independently implemented canonicalization
    # without constructing real artifacts.
    layer_pins = tuple(
        dataclasses.make_dataclass("Pin", [("layer_index", int), ("manifest_sha256", str)])(i, manifests[i])
        for i in range(18)
    )
    stack = dataclasses.make_dataclass(
        "Stack",
        [
            ("teacher_checkpoint_sha256", str),
            ("teacher_code_sha", str),
            ("task_id", str),
            ("episode_id", str),
            ("causal_cut_step", int),
            ("requested_budget", int),
            ("layer_pins", tuple),
        ],
    )(
        identity.teacher_checkpoint_sha256,
        identity.teacher_code_sha,
        identity.task_id,
        identity.episode_id,
        identity.causal_cut_step,
        64,
        layer_pins,
    )

    class Loaded:
        def __init__(self, index):
            self.manifest = dataclasses.make_dataclass("Manifest", [("teacher_tap_sha256", str)])(taps[index])

    class Trusted:
        def resolve(self, requested):
            return Loaded(requested.layer_index)

    monkeypatch.setattr(
        "robomme_integration.eval.framesamp_am_oracle_server.load_framesamp_am_trusted_index",
        lambda *args, **kwargs: Trusted(),
    )
    assert (
        server_tap_stack_sha256(
            "/unused",
            expected_trusted_index_sha256="d" * 64,
            stack=stack,
        )
        == expected
    )


def test_reviewed_overlay_collection_exposes_scan_stacked_qkv_to_adapter(tmp_path, monkeypatch):
    if not (OFFICIAL_CHECKOUT / ".git").is_dir():
        pytest.skip("pinned RoboMME policy checkout is unavailable")
    destination = tmp_path / "policy-overlay"
    stage_framesamp_am_policy_overlay(OFFICIAL_CHECKOUT, destination)
    module_path = destination.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts)
    spec = importlib.util.spec_from_file_location("producer_test_history_gemma", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PALIGEMMA_VOCAB_SIZE", 8)
    config = Config(
        width=1024,
        depth=2,
        mlp_dim=32,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    )

    class Harness(nn.Module):
        capture: bool = False

        @nn.compact
        def __call__(self, embedded, positions, mask, mem_seq, mem_mask):
            return module.Module(
                configs=(config, config),
                embed_dtype="bfloat16",
                integration_type="modulation",
                name="llm",
            )(
                embedded,
                positions,
                mask,
                mem_seq=mem_seq,
                mem_mask=mem_mask,
                capture_framesamp_am_taps=self.capture,
            )

    embedded = [
        jnp.zeros((1, 2, 1024), dtype=jnp.bfloat16),
        jnp.zeros((1, 20, 1024), dtype=jnp.bfloat16),
    ]
    positions = jnp.arange(22, dtype=jnp.int32)[None]
    mask = jnp.ones((1, 22, 22), dtype=jnp.bool_)
    memory = jnp.ones((1, TOKEN_BUDGET, 1024), dtype=jnp.bfloat16)
    memory_mask = jnp.arange(TOKEN_BUDGET)[None] < 96
    harness = Harness()
    variables = harness.init(
        jax.random.key(2),
        embedded,
        positions,
        mask,
        [None, memory],
        [None, memory_mask],
    )
    _, mutable = Harness(capture=True).apply(
        variables,
        embedded,
        positions,
        mask,
        [None, memory],
        [None, memory_mask],
        mutable=["framesamp_am_taps"],
    )
    taps = extract_scanned_framesamp_teacher_taps(mutable, expected_layers=2)
    assert taps.queries_post_rope_pre_scale.shape == (2, 1, 20, 4, 256)
    assert taps.keys_post_rope.shape == (2, 1, TOKEN_BUDGET, 1, 256)

    # Multiple sow writes indicate state accumulation and are rejected rather
    # than silently selecting a stale or latest value.
    bad = {
        "q_post_rope_pre_scale": (np.zeros((2, 1, 20, 4, 256)),) * 2,
        "recent_k_post_rope": (np.zeros((2, 1, TOKEN_BUDGET, 1, 256)),),
        "recent_v_post_projection": (np.zeros((2, 1, TOKEN_BUDGET, 1, 256)),),
    }
    with pytest.raises(ValueError, match="fresh one-write"):
        extract_scanned_framesamp_teacher_taps(bad, expected_layers=2)


def test_uint63_key_uses_both_words():
    low_only = np.asarray(uint63_to_jax_key(5))
    high_word = np.asarray(uint63_to_jax_key((1 << 32) + 5))
    assert not np.array_equal(low_only, high_word)


def test_teacher_smoke_phase_log_is_flushed_bounded_json():
    class TrackingStream(io.StringIO):
        flush_count = 0

        def flush(self):
            self.flush_count += 1
            super().flush()

    stream = TrackingStream()
    _log_phase(
        "first_jit_sample_start",
        started_at=10.0,
        stream=stream,
        clock=lambda: 12.3456,
        device_count=1,
        platforms="gpu",
    )
    assert stream.flush_count == 1
    record = json.loads(stream.getvalue())
    assert record == {
        "device_count": 1,
        "elapsed_seconds": 2.346,
        "kind": PHASE_LOG_KIND,
        "phase": "first_jit_sample_start",
        "platforms": "gpu",
    }


def test_parity_fixture_export_is_create_once_and_explicitly_not_online_attestation(tmp_path):
    history = _history()

    class Observation:
        def to_dict(self):
            return {
                "image": {
                    "base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
                    "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
                },
                "image_mask": {
                    "base_0_rgb": np.ones((1,), dtype=np.bool_),
                    "left_wrist_0_rgb": np.ones((1,), dtype=np.bool_),
                },
                "state": np.zeros((1, 8), dtype=np.float32),
                "tokenized_prompt": np.ones((1, 7), dtype=np.int32),
                "tokenized_prompt_mask": np.ones((1, 7), dtype=np.bool_),
                "token_ar_mask": None,
                "token_loss_mask": None,
                "static_image_emb": history.image[None],
                "static_mask": history.token_mask[None],
                "static_pos_emb": history.position[None],
                "static_state_emb": np.zeros((1, TOKEN_BUDGET, 8), dtype=np.float32),
                "recur_image_emb": None,
                "recur_mask": None,
                "recur_pos_emb": None,
                "recur_state_emb": None,
                "symbolic_tokenized_prompt": None,
                "symbolic_tokenized_prompt_mask": None,
            }

    destination = tmp_path / "teacher-parity-fixture"
    manifest = seal_teacher_smoke_fixture(
        destination,
        observation=Observation(),
        history=history,
        official_actions=np.zeros((1, 20, 32), dtype=np.float32),
        sampler_noise=np.ones((1, 20, 32), dtype=np.float32),
        task_id="robomme_task_003",
        episode_id="ep000076",
        causal_cut_step=120,
        source_fixture_id="ep000076-t00120",
        source_chunk_role="fit_chunk",
        source_bundle_manifest_sha256="d" * 64,
        source_bundle_content_sha256="e" * 64,
        source_bundle_payload_sha256="f" * 64,
        source_record_metadata={"fixture_id": "ep000076-t00120"},
    )
    assert manifest["attestation_scope"] == ATTESTATION_SCOPE
    loaded, observation, loaded_history, actions, noise = _load_observation_fixture(destination / "manifest.json")
    assert loaded["source_fixture_id"] == "ep000076-t00120"
    assert set(observation.images) == {"base_0_rgb", "left_wrist_0_rgb"}
    assert framesamp_history_sha256(loaded_history) == framesamp_history_sha256(history)
    assert actions.shape == noise.shape == (1, 20, 32)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        seal_teacher_smoke_fixture(
            destination,
            observation=Observation(),
            history=history,
            official_actions=np.zeros((1, 20, 32), dtype=np.float32),
            sampler_noise=np.ones((1, 20, 32), dtype=np.float32),
            task_id="robomme_task_003",
            episode_id="ep000076",
            causal_cut_step=120,
            source_fixture_id="ep000076-t00120",
            source_chunk_role="fit_chunk",
            source_bundle_manifest_sha256="d" * 64,
            source_bundle_content_sha256="e" * 64,
            source_bundle_payload_sha256="f" * 64,
            source_record_metadata={"fixture_id": "ep000076-t00120"},
        )
