"""End-to-end orchestration of E0 against a fake policy.

The AM kernel, the plans, the query banks, the metrics and the aggregation are
all real here; only the released 3B policy is replaced.  A wiring bug in the
runner therefore fails locally in a second instead of on a p5e node.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import types

import numpy as np
import pytest

from robomme_integration.amkv import driver, e0_run, query_bank, stage_e0
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    even_sampling_indices,
)

LAYERS = 2
HEAD_DIM = 16
ACTION_TOKENS = 6
ACTION_DIM = 4
FLOW_STEPS = 3
EPISODES = 2


def _record(episode: int, role: str, step: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        fixture_id=f"ep{episode:06d}-t{step:05d}",
        pair_id=f"ep{episode:06d}",
        chunk_role=role,
        step_idx=step,
        exec_start_idx=0,
        memory_step_indices=np.asarray(even_sampling_indices(step), dtype=np.int64),
    )


def _observation(seed: int):
    generator = np.random.default_rng(seed)
    return types.SimpleNamespace(
        static_image_emb=generator.normal(size=(1, TOKEN_BUDGET, 4)).astype(np.float32),
        static_pos_emb=generator.normal(size=(1, TOKEN_BUDGET, 2)).astype(np.float32),
        static_mask=np.ones((1, TOKEN_BUDGET), dtype=np.bool_),
        state=np.zeros((1, ACTION_DIM), dtype=np.float32),
    )


class _FakeDenoiser:
    """Velocities that respond to the memory the way a live policy would."""

    def __init__(self, model, *, capture: bool = False):
        self.model = model
        self.capture = capture

    def __call__(self, observation, *, noise, num_steps=FLOW_STEPS, am_pack=None, teacher_states=None):
        seed = int(abs(float(np.asarray(observation.static_image_emb).sum())) * 1000) % 9973
        generator = np.random.default_rng(seed)
        velocities = generator.normal(size=(num_steps, ACTION_TOKENS, ACTION_DIM)).astype(np.float32) + 1.0
        queries = memory_keys = memory_values = None
        if self.capture:
            queries = generator.normal(
                size=(num_steps, LAYERS, query_bank.QUERY_HEAD_COUNT, ACTION_TOKENS, HEAD_DIM)
            ).astype(np.float32)
            memory_keys = generator.normal(size=(LAYERS, TOKEN_BUDGET, HEAD_DIM)).astype(np.float32)
            memory_values = generator.normal(size=(LAYERS, TOKEN_BUDGET, HEAD_DIM)).astype(np.float32)
        if am_pack is not None:
            served = int(am_pack.compact_keys.shape[2]) + int(am_pack.recent_keys.shape[2])
            magnitude = float(np.abs(np.asarray(am_pack.compact_values, dtype=np.float32)).mean())
            # respond to BOTH the token budget and the served values, so the
            # runner's 'pack is actually consumed' probe is exercised
            if served != TOKEN_BUDGET or magnitude == 0.0:
                velocities = velocities * (
                    1.0 + 0.01 * (TOKEN_BUDGET - served) / TOKEN_BUDGET + 0.02 * (0.8 - magnitude)
                )
        if teacher_states is not None:
            # a forced pass evaluates at the teacher's states: same shape, no drift term
            velocities = velocities + 0.001 * np.asarray(teacher_states, dtype=np.float32)
        return driver.DenoiseTrace(
            flow_times=driver.flow_schedule(num_steps),
            denoise_states=(
                np.asarray(teacher_states, dtype=np.float32)
                if teacher_states is not None
                else np.cumsum(velocities, axis=0)
            ),
            velocities=velocities,
            actions=velocities.sum(axis=0),
            queries=queries,
            memory_keys=memory_keys,
            memory_values=memory_values,
            memory_kv_recomputed_per_step=True if self.capture else None,
            teacher_forced=teacher_states is not None,
        )


@pytest.fixture
def patched(monkeypatch):
    records = []
    for episode in range(EPISODES):
        records.append(_record(episode, query_bank.FIT_CHUNK, 47))
        records.append(_record(episode, query_bank.EVAL_CHUNK, 63))

    model = types.SimpleNamespace(action_horizon=ACTION_TOKENS, action_dim=ACTION_DIM)
    policy = types.SimpleNamespace(_model=model)

    monkeypatch.setattr(
        e0_run,
        "require_reviewed_amkv_patch",
        lambda root: types.SimpleNamespace(
            policy_git_sha="d" * 40,
            policy_tree_sha1="e" * 40,
            to_dict=lambda: {"patched_module_sha256": "f" * 64},
        ),
    )
    monkeypatch.setattr(
        e0_run,
        "_validated_result_identity",
        lambda args, patch: {
            "run_id": args.run_id,
            "run_manifest_sha256": args.run_manifest_sha256,
            "evidence_input_identity_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        "robomme_integration.amkv.episodes.load_fixture_bundle", lambda path: tuple(records), raising=False
    )
    # the fake policy has a 2-layer / 6-token geometry, not the released one
    monkeypatch.setattr(e0_run, "STRICT_OFFICIAL_GEOMETRY", False)
    monkeypatch.setattr(driver, "load_policy", lambda *a, **k: policy)
    monkeypatch.setattr(driver, "Denoiser", _FakeDenoiser)
    monkeypatch.setattr(driver, "installed_patch", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(
        driver,
        "selftest_matches_official_sampler",
        lambda *a, **k: {"driver_matches_official_sampler": True, "bitwise": True},
    )

    counter = {"n": 0}

    def build_observation(_policy, record):
        counter["n"] += 1
        observation = _observation(record.step_idx)
        history = driver.framesamp_history_from_observation(observation, record.memory_step_indices)
        return observation, history, {"fixture_id": record.fixture_id, "step_idx": record.step_idx}

    monkeypatch.setattr(driver, "build_observation", build_observation)
    return records


def _args(**overrides) -> argparse.Namespace:
    values = {
        "fixtures": "unused",
        "checkpoint": "unused",
        "policy_source": "unused",
        "out": "unused",
        "ratios": (4.0, 8.0),
        "num_steps": FLOW_STEPS,
        "model_seed": 7,
        "noise_seed": 0,
        "runtime_dtype": "bfloat16",
        "limit": 0,
        "minimum_episodes": EPISODES,
        "timing_repeats": 1,
        "run_manifest": "unused",
        "run_id": "amkv-e0-test",
        "run_manifest_sha256": "1" * 64,
        "code_source_tree_sha256": "2" * 64,
        "policy_source_archive_sha256": "3" * 64,
        "policy_source_receipt": "unused",
        "policy_source_receipt_sha256": "8" * 64,
        "policy_git_sha": "d" * 40,
        "policy_tree_sha1": "e" * 40,
        "checkpoint_inventory_sha256": "4" * 64,
        "fixtures_manifest_sha256": "5" * 64,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_e0_runs_every_arm_and_labels_the_result(patched):
    results = e0_run.run(_args())
    assert results["schema_version"] == e0_run.RESULT_SCHEMA_VERSION
    assert results["kind"] == e0_run.RESULT_KIND
    assert set(results["aggregates"]) == {
        "am4_f0",
        "am8_f0",
        "am8_f1",
        "am8_f0_stale",
        "drop4_random",
        "drop8_random",
        "memory_destroyed",
    }
    for arm, summary in results["aggregates"].items():
        assert summary["episode_count"] == EPISODES, arm
        assert len(summary["relative_velocity_error_mean"]) == FLOW_STEPS
        assert len(summary["relative_velocity_error_p95"]) == FLOW_STEPS
    labels = results["labels"]
    assert labels["gradients"] == "none_serve_path_cache_transform"
    assert labels["ratios"] == [4.0, 8.0]
    assert set(labels["kernel_sha256"]) == set(e0_run.KERNEL_FILES)
    assert labels["noise_sha256"]
    assert results["selftest"]["driver_matches_official_sampler"]
    assert results["identity"]["evidence_input_identity_sha256"] == "9" * 64
    assert set(results["payload_quantization"]) == {
        "am4_f0",
        "am8_f0",
        "am8_f1",
        "am8_f0_stale",
    }


def test_every_episode_proves_its_query_banks_are_disjoint(patched):
    results = e0_run.run(_args())
    assert len(results["per_episode"]) == EPISODES
    for episode in results["per_episode"]:
        banks = episode["query_banks"]
        assert banks["shared_query_rows"] == 0
        assert banks["fit"]["chunk_role"] == query_bank.FIT_CHUNK
        assert banks["heldout"]["chunk_role"] == query_bank.EVAL_CHUNK
        assert banks["fit"]["step_idx"] != banks["heldout"]["step_idx"]


def test_plans_report_the_served_budget_per_arm(patched):
    plans = e0_run.run(_args())["plans"]
    assert plans["am8_f0"]["served_tokens"] == TOKEN_BUDGET // 8
    assert plans["am8_f1"]["recent_exact_tokens"] == TOKENS_PER_FRAME
    assert plans["am8_f1"]["served_tokens"] == TOKEN_BUDGET // 8
    assert plans["am4_f0"]["served_tokens"] == TOKEN_BUDGET // 4
    assert plans["memory_destroyed"]["method"] == "memory_values_zeroed"


def test_identity_arm_is_recorded_as_the_parity_gate(patched):
    results = e0_run.run(_args())
    assert len(results["identity_parity"]) == EPISODES
    for row in results["identity_parity"]:
        assert row["arm_id"] == e0_run.IDENTITY_ARM
        assert row["memory_kv_recomputed_per_flow_step"] is True
        assert set(row["fields"]) == {"actions", "velocities", "denoise_states"}
        assert all(field["bitwise"] for field in row["fields"].values())


def test_timing_is_explicitly_nonclaimable_and_uses_real_disjoint_banks(patched):
    walls = e0_run.run(_args())["tracing_microbenchmark"]
    for key in ("full_denoise", "identity_denoise", "am4_f0_denoise", "am8_f0_denoise"):
        assert walls[key]["seconds_mean"] >= 0.0, key
    assert walls["am8_f0_kv_bytes"]["kv_bytes_ratio"] == pytest.approx(8.0)
    assert walls["am4_f0_fit_seconds"] >= 0.0
    assert walls["layers"] == LAYERS
    assert walls["flow_steps"] == FLOW_STEPS
    assert walls["measurement_kind"] == "python_unrolled_denoiser_microbenchmark_v1"
    assert walls["speedup_claim_permitted"] is False
    assert walls["query_banks_disjoint"] is True
    assert walls["fit_query_bank_id"] != walls["heldout_query_bank_id"]


def test_layer_diagnostics_exist_only_for_fitted_arms(patched):
    diagnostics = e0_run.run(_args())["layer_diagnostics"]
    assert "am8_f0" in diagnostics and "memory_destroyed" not in diagnostics
    layers = diagnostics["am8_f0"][0]["layers"]
    assert len(layers) == LAYERS
    assert layers[0]["compact_tokens"] == TOKEN_BUDGET // 8
    assert layers[0]["source_tokens"] == MAX_FRAMES * TOKENS_PER_FRAME
    assert 0.0 <= layers[0]["heldout_output_relative_l2"] < 10.0
    assert layers[0]["served_storage_dtype"] == "bfloat16"
    assert layers[0]["served_payload_finite_after_quantization"] is True
    assert 0.0 <= layers[0]["served_heldout_output_relative_l2"] < 10.0


def test_too_few_complete_pairs_is_refused(patched):
    with pytest.raises(SystemExit, match="at least"):
        e0_run.run(_args(minimum_episodes=EPISODES + 1))


def test_evidence_runner_refuses_nonofficial_runtime_precision(patched):
    with pytest.raises(ValueError, match="official bfloat16"):
        e0_run.run(_args(runtime_dtype="float32"))


def test_capture_refuses_a_right_padded_memory_in_the_full512_evidence_lane(monkeypatch):
    record = _record(0, query_bank.EVAL_CHUNK, 63)
    observation = _observation(63)
    monkeypatch.setattr(
        driver,
        "build_observation",
        lambda policy, item: (
            observation,
            types.SimpleNamespace(),
            {"fixture_id": item.fixture_id, "valid_memory_tokens": TOKEN_BUDGET - TOKENS_PER_FRAME},
        ),
    )
    monkeypatch.setattr(driver, "validate_official_capture", lambda *args, **kwargs: None)
    monkeypatch.setattr(e0_run, "STRICT_OFFICIAL_GEOMETRY", True)
    denoisers = {"capture": _FakeDenoiser(types.SimpleNamespace(), capture=True)}
    with pytest.raises(ValueError, match="fully populated 512-token"):
        e0_run._capture_chunk(
            denoisers,
            types.SimpleNamespace(),
            record,
            role=query_bank.EVAL_CHUNK,
            noise=np.zeros((1, ACTION_TOKENS, ACTION_DIM), dtype=np.float32),
            num_steps=FLOW_STEPS,
            noise_sha="0" * 64,
        )


def test_identity_trace_gate_fails_closed_on_any_velocity_drift():
    array = np.ones((FLOW_STEPS, ACTION_TOKENS, ACTION_DIM), dtype=np.float32)
    reference = types.SimpleNamespace(actions=array[0], velocities=array, denoise_states=array)
    candidate = types.SimpleNamespace(actions=array[0], velocities=array.copy(), denoise_states=array)
    candidate.velocities[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="identity failed"):
        e0_run._identity_trace_gate(reference, candidate, fixture_id="fixture")


def test_result_identity_is_cross_checked_against_the_sealed_manifest(tmp_path, monkeypatch):
    args = _args()
    monkeypatch.setattr(stage_e0, "PINNED_POLICY_GIT_SHA", args.policy_git_sha)
    monkeypatch.setattr(stage_e0, "PINNED_POLICY_TREE_SHA1", args.policy_tree_sha1)
    source = tmp_path / "policy"
    source.mkdir()
    (source / "pyproject.toml").write_text("project", encoding="utf-8")
    extracted = stage_e0.source_tree_identity(source)
    receipt_document = {
        "schema_version": stage_e0.SOURCE_RECEIPT_SCHEMA_VERSION,
        "kind": stage_e0.SOURCE_RECEIPT_KIND,
        "component": stage_e0.POLICY_SOURCE_COMPONENT,
        "git": {
            "git_sha": args.policy_git_sha,
            "git_tree_sha1": args.policy_tree_sha1,
            "worktree_status": "clean_including_untracked_and_submodules",
        },
        "archive": {
            "uri": stage_e0.policy_source_uri(args.policy_source_archive_sha256),
            "sha256": args.policy_source_archive_sha256,
            "bytes": 123,
        },
        "extracted_tree": extracted,
    }
    receipt = tmp_path / "source-receipt.json"
    receipt.write_text(stage_e0.canonical_json(receipt_document) + "\n", encoding="utf-8")
    args.policy_source = str(source)
    args.policy_source_receipt = str(receipt)
    args.policy_source_receipt_sha256 = stage_e0.sha256_file(receipt)
    scientific = {
        "code": {"sanitized_source_tree_sha256": args.code_source_tree_sha256},
        "policy_source": {
            "uri": receipt_document["archive"]["uri"],
            "sha256": args.policy_source_archive_sha256,
            "git_sha": args.policy_git_sha,
            "git_tree_sha1": args.policy_tree_sha1,
            "receipt_uri": stage_e0.source_receipt_uri(args.policy_source_receipt_sha256),
            "receipt_sha256": args.policy_source_receipt_sha256,
            "extracted_tree_sha256": extracted["tree_sha256"],
            "extracted_tree_objects": extracted["totals"]["objects"],
            "extracted_tree_bytes": extracted["totals"]["bytes"],
        },
        "checkpoint": {"inventory_sha256": args.checkpoint_inventory_sha256},
        "fixtures": {"manifest_sha256": args.fixtures_manifest_sha256},
        "ratios": list(args.ratios),
        "evaluation": {
            "runtime_dtype": args.runtime_dtype,
            "num_flow_steps": args.num_steps,
            "model_seed": args.model_seed,
            "noise_seed": args.noise_seed,
            "minimum_episodes": args.minimum_episodes,
            "timing_repeats": args.timing_repeats,
        },
    }
    scientific_sha = hashlib.sha256(e0_run._canonical_json(scientific).encode()).hexdigest()
    args.run_id = f"amkv-e0-{scientific_sha[:16]}"
    document = {
        "schema_version": 1,
        "kind": e0_run.RUN_MANIFEST_KIND,
        "run_id": args.run_id,
        "scientific_spec_sha256": scientific_sha,
        "scientific": scientific,
    }
    document["manifest_sha256"] = hashlib.sha256(e0_run._canonical_json(document).encode()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    args.run_manifest = str(manifest)
    args.run_manifest_sha256 = document["manifest_sha256"]
    patch = types.SimpleNamespace(
        policy_git_sha=args.policy_git_sha,
        policy_tree_sha1=args.policy_tree_sha1,
        official_source_sha256="6" * 64,
        patched_module_sha256="7" * 64,
    )
    identity = e0_run._validated_result_identity(args, patch)
    assert identity["run_id"] == args.run_id
    assert len(identity["evidence_input_identity_sha256"]) == 64
    assert identity["policy_source_receipt_sha256"] == args.policy_source_receipt_sha256
    assert identity["policy_source_extracted_tree_sha256"] == extracted["tree_sha256"]

    def write_manifest(value: dict) -> None:
        value = dict(value)
        value.pop("manifest_sha256", None)
        value["manifest_sha256"] = hashlib.sha256(e0_run._canonical_json(value).encode()).hexdigest()
        manifest.write_text(json.dumps(value), encoding="utf-8")
        args.run_manifest_sha256 = value["manifest_sha256"]

    original_run_id = args.run_id
    write_manifest({**document, "kind": "wrong_kind"})
    with pytest.raises(ValueError, match="manifest kind"):
        e0_run._validated_result_identity(args, patch)

    drifted_run_id = "amkv-e0-ffffffffffffffff"
    args.run_id = drifted_run_id
    write_manifest({**document, "run_id": drifted_run_id})
    with pytest.raises(ValueError, match="not derived"):
        e0_run._validated_result_identity(args, patch)

    args.run_id = original_run_id
    write_manifest(document)
    args.noise_seed += 1
    with pytest.raises(ValueError, match="evaluation config"):
        e0_run._validated_result_identity(args, patch)
