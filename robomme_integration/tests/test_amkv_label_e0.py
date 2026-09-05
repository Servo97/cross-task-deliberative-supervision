"""Sealed-proof quarantine and the pre-registered E0 numerical threshold."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from robomme_integration.amkv import label_e0, stage_e0

FLOW_TIMES = [1.0, 0.9, 0.8]
RESULT_SHA = "a" * 64


def _aggregate(arm_id: str, mean: list[float], action: float = 0.02) -> dict:
    return {
        "arm_id": arm_id,
        "flow_times": FLOW_TIMES,
        "episode_count": 32,
        "relative_velocity_error_mean": mean,
        "relative_velocity_error_p95": [value * 1.5 for value in mean],
        "relative_velocity_error_max": [value * 2 for value in mean],
        "relative_velocity_error_pooled_mean": sum(mean) / len(mean),
        "relative_action_error_mean": action,
    }


def _threshold_results(am8: list[float], destroyed: list[float], random_drop: list[float]) -> dict:
    return {
        "aggregates": {
            "am8_f0": _aggregate("am8_f0", am8),
            "memory_destroyed": _aggregate("memory_destroyed", destroyed, action=0.2),
            "drop8_random": _aggregate("drop8_random", random_drop, action=0.06),
        }
    }


def _bitwise(names: tuple[str, ...]) -> dict:
    return {name: {"bitwise": True, "max_abs_delta": 0.0} for name in names}


def _valid_results() -> dict:
    identity = {
        "run_id": "amkv-e0-0123456789abcdef",
        "run_manifest_sha256": "1" * 64,
        "code_source_tree_sha256": "2" * 64,
        "policy_source_archive_sha256": "3" * 64,
        "policy_source_receipt_sha256": "4" * 64,
        "policy_source_extracted_tree_sha256": "5" * 64,
        "policy_source_extracted_tree_objects": 123,
        "policy_git_sha": stage_e0.PINNED_POLICY_GIT_SHA,
        "policy_tree_sha1": stage_e0.PINNED_POLICY_TREE_SHA1,
        "checkpoint_inventory_sha256": "6" * 64,
        "fixtures_manifest_sha256": "7" * 64,
        "official_history_gemma_sha256": "8" * 64,
        "amkv_patched_module_sha256": "9" * 64,
    }
    identity["evidence_input_identity_sha256"] = hashlib.sha256(
        label_e0._canonical_json(identity).encode()
    ).hexdigest()
    plans = {
        arm: {
            "method": "framesamp_history_am_v1",
            "valid_frames": 32,
            "valid_tokens": 512,
        }
        for arm in ("am8_f0", "drop8_random")
    }
    plans["memory_destroyed"] = {"method": "memory_values_zeroed"}
    episodes = []
    for index in range(32):
        episodes.append(
            {
                "pair_id": f"ep{index:06d}",
                "fit_chunk": {"valid_memory_tokens": 512},
                "eval_chunk": {"valid_memory_tokens": 512},
                "query_banks": {
                    "fit": {
                        "bank_id": f"fit-{index}",
                        "chunk_role": "fit_chunk",
                        "step_idx": 47,
                    },
                    "heldout": {
                        "bank_id": f"heldout-{index}",
                        "chunk_role": "eval_chunk",
                        "step_idx": 63,
                    },
                    "shared_query_rows": 0,
                    "fit_query_rows": 240,
                    "heldout_query_rows": 240,
                    "disjointness_proof": "sha256_per_query_row_intersection_empty_v1",
                },
            }
        )
    identity_fields = ("actions", "velocities", "denoise_states")
    baseline_fields = (*identity_fields, "plain_actions", "plain_velocities", "plain_denoise_states")
    payload_rows = [
        {
            "layer_index": index,
            "heldout_relative_l2_float64": 0.01,
            "heldout_relative_l2_quantized": 0.011,
            "quantization_only_relative_l2": 0.001,
        }
        for index in range(18)
    ]
    results = {
        "schema_version": 3,
        "kind": "amkv_e0_velocity_matching_result",
        "experiment": "H10_E0_offline_velocity_matching",
        "identity": identity,
        "labels": {"runtime_dtype": "bfloat16", "provenance": identity},
        "selftest": {
            "restored_checkpoint_patched_baseline_identity": _bitwise(baseline_fields),
            "restored_checkpoint_full_kv_beta0_identity": _bitwise(identity_fields),
            "official_geometry": {
                "memory_tokens": 512,
                "layers": 18,
                "query_heads": 4,
                "kv_heads": 1,
                "head_dim": 256,
                "action_horizon": 20,
            },
        },
        "fixture_proof": {
            "complete_pair_count": 32,
            "fixture_record_count": 64,
            "required_pair_count": 32,
            "exact_pair_count": True,
            "full_memory_tokens": 512,
        },
        "per_episode": episodes,
        "identity_parity": [
            {
                "fixture_id": f"ep{index:06d}",
                "arm_id": "identity",
                "fields": _bitwise(identity_fields),
                "memory_kv_recomputed_per_flow_step": True,
            }
            for index in range(32)
        ],
        "plans": plans,
        "payload_quantization": {
            "am8_f0": {
                "payload_dtype": "bfloat16",
                "beta_dtype": "float32_exact_from_any_sealed_payload",
                "layers": payload_rows,
                "quantization_only_relative_l2_mean": 0.001,
                "quantization_only_relative_l2_max": 0.001,
            }
        },
        "tracing_microbenchmark": {
            "measurement_kind": "python_unrolled_denoiser_microbenchmark_v1",
            "speedup_claim_permitted": False,
            "timing_note": "Development only; not comparable end-to-end serve latency.",
            "full_denoise": {"seconds_mean": 0.5, "seconds_min": 0.48},
            "am8_f0_kv_bytes": {
                "full_kv_bytes": 9437184,
                "compact_kv_bytes": 1179648,
                "kv_bytes_ratio": 8.0,
            },
        },
        **_threshold_results(
            [0.006, 0.007, 0.008],
            [0.04, 0.08, 0.09],
            [0.014, 0.025, 0.03],
        ),
    }
    return results


def _completion(results: dict, *, result_sha: str = RESULT_SHA) -> dict:
    return {
        "schema_version": 1,
        "kind": "amkv_e0_velocity_matching_complete",
        "run_id": results["identity"]["run_id"],
        "run_manifest_sha256": results["identity"]["run_manifest_sha256"],
        "results_sha256": result_sha,
        "training_job_name": "sarvesh-amkv-e0-test",
    }


def _label(results: dict) -> dict:
    return label_e0.label(results, result_sha256=RESULT_SHA, completion=_completion(results))


def test_pass_requires_the_bound_at_every_flow_time_and_both_controls():
    evaluation = label_e0.evaluate_thresholds(
        _threshold_results([0.006, 0.007, 0.008], [0.04, 0.08, 0.09], [0.014, 0.025, 0.03])
    )
    assert evaluation["verdict"] == "PASS"
    assert evaluation["checks"]["mean_rel_dv_within_threshold_at_every_flow_time"]
    assert evaluation["checks"]["destroyed_scale_sufficient"]
    assert evaluation["checks"]["beats_naive_random_drop"]


def test_one_flow_time_over_the_bound_fails_the_whole_cell():
    evaluation = label_e0.evaluate_thresholds(
        _threshold_results([0.006, 0.045, 0.008], [0.4, 0.5, 0.6], [0.2, 0.3, 0.4])
    )
    assert evaluation["verdict"] == "FAIL"
    assert evaluation["checks"]["worst_flow_time_mean_rel_dv"] == pytest.approx(0.045)


def test_not_beating_random_dropping_fails():
    evaluation = label_e0.evaluate_thresholds(
        _threshold_results([0.02, 0.02, 0.02], [0.3, 0.3, 0.3], [0.005, 0.005, 0.005])
    )
    assert evaluation["verdict"] == "FAIL"
    assert evaluation["checks"]["beats_naive_random_drop"] is False


def test_a_weak_memory_makes_the_cell_indeterminate_not_a_pass():
    evaluation = label_e0.evaluate_thresholds(
        _threshold_results([0.006, 0.006, 0.006], [0.007, 0.007, 0.007], [0.01, 0.01, 0.01])
    )
    assert evaluation["verdict"] == "INDETERMINATE"
    assert "barely matters" in evaluation["reason"]


def test_schema_v3_sealed_full_contract_releases_quarantine():
    results = _valid_results()
    block = _label(results)
    release = block["quarantine_release"]
    assert release["released"] is True
    assert release["evidence_proof"]["verdict"] == "VERIFIED"
    assert all(release["evidence_proof"]["checks"].values())
    assert block["threshold_evaluation"]["verdict"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda result: result.update(schema_version=2), "schema_v3_result_kind"),
        (lambda result: result["identity"].pop("policy_source_receipt_sha256"), "required_provenance_present"),
        (lambda result: result["payload_quantization"].clear(), "bf16_payload_quantization_proof"),
        (
            lambda result: result["identity_parity"][0]["fields"]["actions"].update(bitwise=False),
            "all_fixture_full_kv_beta0_bitwise_identity",
        ),
        (
            lambda result: result["per_episode"][0]["query_banks"].update(shared_query_rows=1),
            "fit_heldout_query_banks_disjoint",
        ),
        (lambda result: result.update(wall_clock={"legacy": True}), "legacy_wall_clock_absent"),
    ],
)
def test_any_missing_scientific_proof_keeps_the_result_quarantined(mutation, failed_check):
    results = _valid_results()
    mutation(results)
    block = _label(results)
    proof = block["quarantine_release"]["evidence_proof"]
    assert block["quarantine_release"]["released"] is False
    assert proof["checks"][failed_check] is False
    assert block["threshold_evaluation"]["verdict"] == "INDETERMINATE"


def test_wrong_or_missing_completion_marker_cannot_release_exact_result_bytes():
    results = _valid_results()
    assert label_e0.label(results)["quarantine_release"]["released"] is False
    block = label_e0.label(
        results,
        result_sha256=RESULT_SHA,
        completion=_completion(results, result_sha="b" * 64),
    )
    assert block["quarantine_release"]["released"] is False
    assert not block["quarantine_release"]["evidence_proof"]["checks"]["completion_marker_seals_exact_result_bytes"]


def test_render_uses_only_explicitly_nonclaimable_timing():
    results = _valid_results()
    block = _label(results)
    timing = label_e0.nonclaimable_timing_table(results)
    assert "No latency or speedup claim" in timing
    assert "8.00x storage" in timing
    rendered = label_e0.render(results, block)
    assert "Non-claimable tracing microbenchmark" in rendered
    assert "RELEASED" in rendered and "PASS" in rendered
    assert json.loads(rendered.split("```json")[1].split("```")[0])


def test_mutating_the_provenance_after_its_self_seal_is_rejected():
    results = copy.deepcopy(_valid_results())
    results["identity"]["checkpoint_inventory_sha256"] = "f" * 64
    results["labels"]["provenance"] = results["identity"]
    proof = _label(results)["quarantine_release"]["evidence_proof"]
    assert proof["checks"]["evidence_input_identity_self_seal"] is False
