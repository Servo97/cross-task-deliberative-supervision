#!/usr/bin/env python3
"""Verify, label, and render a sealed AMKV E0 result.

The numerical threshold is useful only after the evidence contract is proven.
This module therefore keeps a result quarantined unless its exact schema-v3
bytes are sealed by the SageMaker completion marker and the result itself
proves the official BF16/512-token/32-pair geometry, both bitwise identity
gates, disjoint fit/held-out banks, quantized payload diagnostics, and all
content-addressed provenance (including the clean-source stage receipt).

The emitted timing block is explicitly non-claimable.  It is a Python-unrolled
development microbenchmark, not policy-server or simulator latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from robomme_integration.amkv import stage_e0

RESULT_SCHEMA_VERSION = 3
RESULT_KIND = "amkv_e0_velocity_matching_result"
COMPLETE_SCHEMA_VERSION = 1
COMPLETE_KIND = "amkv_e0_velocity_matching_complete"
EVIDENCE_PAIRS = 32
MEMORY_TOKENS = 512
MEMORY_FRAMES = 32
LAYERS = 18
QUERY_HEADS = 4
KV_HEADS = 1
HEAD_DIM = 256
ACTION_HORIZON = 20
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^amkv-e0-[0-9a-f]{16}$")

ORACLE_DISCLAIMER = {
    "claim_class": "oracle_diagnostic_of_compressibility_not_a_deployable_method",
    "oracle_affordance": (
        "the per-flow-time velocity delta is TEACHER FORCED: the compact run is evaluated at the "
        "full-memory run's own x_t at each flow time; that isolates the function change and is not "
        "available to a deployed policy"
    ),
    "closed_loop_number": (
        "relative_action_error comes from separate free-running passes and is the only closed-loop quantity reported"
    ),
    "query_bank_is_causal": (
        "the fit bank is the preceding chunk (t) and is scored on the following chunk (t+16); no "
        "future action queries are used"
    ),
    "artifact_status": (
        "ephemeral per-layer packs; not durable E1 receipts and never routed through the E1 oracle server"
    ),
}

# Pre-registered before the result is inspected.
THRESHOLDS = {
    "primary_ratio": 8.0,
    "max_mean_relative_velocity_error": 0.03,
    "min_destroyed_over_am": 3.0,
    "min_random_over_am": 1.0,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _arm(results: dict, arm_id: str) -> dict | None:
    return results.get("aggregates", {}).get(arm_id)


def evaluate_thresholds(results: dict, *, thresholds: dict | None = None) -> dict:
    """Apply the pre-registered numerical gate, independent of evidence release."""

    thresholds = dict(thresholds or THRESHOLDS)
    ratio = float(thresholds["primary_ratio"])
    primary_id = f"am{ratio:g}_f0"
    primary = _arm(results, primary_id)
    destroyed = _arm(results, "memory_destroyed")
    random_drop = _arm(results, f"drop{ratio:g}_random")
    if primary is None:
        return {
            "verdict": "INDETERMINATE",
            "reason": f"primary arm {primary_id} is absent from the results",
            "thresholds": thresholds,
        }
    try:
        per_time = [float(value) for value in primary["relative_velocity_error_mean"]]
        pooled = float(primary["relative_velocity_error_pooled_mean"])
    except (KeyError, TypeError, ValueError):
        return {
            "verdict": "INDETERMINATE",
            "reason": f"primary arm {primary_id} has malformed velocity metrics",
            "thresholds": thresholds,
        }
    if not per_time or not all(math.isfinite(value) and value >= 0 for value in [*per_time, pooled]):
        return {
            "verdict": "INDETERMINATE",
            "reason": f"primary arm {primary_id} has empty or non-finite velocity metrics",
            "thresholds": thresholds,
        }
    checks = {
        "mean_rel_dv_within_threshold_at_every_flow_time": all(
            value <= thresholds["max_mean_relative_velocity_error"] for value in per_time
        ),
        "worst_flow_time_mean_rel_dv": max(per_time),
    }
    if destroyed is None or random_drop is None:
        checks["controls_present"] = False
        verdict = "INDETERMINATE"
        reason = "a control arm is missing; an AM error is not interpretable without its scale"
    else:
        try:
            destroyed_pooled = float(destroyed["relative_velocity_error_pooled_mean"])
            random_pooled = float(random_drop["relative_velocity_error_pooled_mean"])
        except (KeyError, TypeError, ValueError):
            destroyed_pooled = random_pooled = float("nan")
        if not all(math.isfinite(value) and value >= 0 for value in (destroyed_pooled, random_pooled)):
            checks["controls_present"] = False
            verdict = "INDETERMINATE"
            reason = "a control arm has malformed or non-finite metrics"
        else:
            checks.update(
                {
                    "controls_present": True,
                    "destroyed_over_am": destroyed_pooled / pooled if pooled else float("inf"),
                    "random_over_am": random_pooled / pooled if pooled else float("inf"),
                }
            )
            checks["destroyed_scale_sufficient"] = checks["destroyed_over_am"] >= thresholds["min_destroyed_over_am"]
            checks["beats_naive_random_drop"] = checks["random_over_am"] >= thresholds["min_random_over_am"]
            if not checks["destroyed_scale_sufficient"]:
                verdict = "INDETERMINATE"
                reason = (
                    "the memory contributes too little to the velocity for this cell to separate "
                    "'compression works' from 'the memory barely matters'"
                )
            elif checks["mean_rel_dv_within_threshold_at_every_flow_time"] and checks["beats_naive_random_drop"]:
                verdict = "PASS"
                reason = "rel-dv is within the pre-registered bound at every flow time and beats both controls"
            else:
                verdict = "FAIL"
                reason = "rel-dv exceeds the pre-registered bound or does not beat naive random dropping"
    return {
        "verdict": verdict,
        "reason": reason,
        "primary_arm": primary_id,
        "pooled_mean_relative_velocity_error": pooled,
        "checks": checks,
        "thresholds": thresholds,
    }


def _bitwise_fields(block: object, names: set[str]) -> bool:
    return (
        isinstance(block, dict)
        and set(block) == names
        and all(
            isinstance(block[name], dict)
            and block[name].get("bitwise") is True
            and float(block[name].get("max_abs_delta", float("nan"))) == 0.0
            for name in names
        )
    )


def audit_result_proof(
    results: dict,
    *,
    result_sha256: str | None,
    completion: dict | None,
) -> dict:
    """Dynamically verify every prerequisite for leaving quarantine."""

    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, condition: object, failure: str) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            failures.append(failure)

    identity = results.get("identity")
    provenance = results.get("labels", {}).get("provenance")
    check(
        "schema_v3_result_kind",
        results.get("schema_version") == RESULT_SCHEMA_VERSION and results.get("kind") == RESULT_KIND,
        "result is not the required schema-v3 AMKV E0 result kind",
    )
    check("legacy_wall_clock_absent", "wall_clock" not in results, "legacy wall_clock block is not admissible")

    marker_ok = (
        isinstance(completion, dict) and isinstance(result_sha256, str) and bool(HEX64.fullmatch(result_sha256))
    )
    if marker_ok:
        marker_ok = (
            completion.get("schema_version") == COMPLETE_SCHEMA_VERSION
            and completion.get("kind") == COMPLETE_KIND
            and completion.get("results_sha256") == result_sha256
            and completion.get("run_id") == (identity or {}).get("run_id")
            and completion.get("run_manifest_sha256") == (identity or {}).get("run_manifest_sha256")
            and isinstance(completion.get("training_job_name"), str)
            and bool(completion["training_job_name"].strip())
        )
    check(
        "completion_marker_seals_exact_result_bytes",
        marker_ok,
        "a valid SageMaker completion marker does not seal these exact result bytes",
    )

    required_identity = {
        "run_id",
        "run_manifest_sha256",
        "code_source_tree_sha256",
        "policy_source_archive_sha256",
        "policy_source_receipt_sha256",
        "policy_source_extracted_tree_sha256",
        "policy_source_extracted_tree_objects",
        "policy_git_sha",
        "policy_tree_sha1",
        "checkpoint_inventory_sha256",
        "fixtures_manifest_sha256",
        "official_history_gemma_sha256",
        "amkv_patched_module_sha256",
        "evidence_input_identity_sha256",
    }
    identity_shape = isinstance(identity, dict) and set(identity) == required_identity
    digests_ok = identity_shape and all(
        isinstance(identity[name], str) and bool(HEX64.fullmatch(identity[name]))
        for name in required_identity
        if name.endswith("sha256")
    )
    check(
        "required_provenance_present",
        identity_shape
        and digests_ok
        and provenance == identity
        and bool(RUN_ID.fullmatch(str(identity.get("run_id", ""))))
        and identity.get("policy_git_sha") == stage_e0.PINNED_POLICY_GIT_SHA
        and identity.get("policy_tree_sha1") == stage_e0.PINNED_POLICY_TREE_SHA1
        and isinstance(identity.get("policy_source_extracted_tree_objects"), int)
        and identity["policy_source_extracted_tree_objects"] > 0,
        "required sealed provenance, source receipt, or pinned Git identity is absent/inconsistent",
    )
    identity_hash_ok = False
    if identity_shape:
        unsealed = dict(identity)
        claimed_identity_sha = unsealed.pop("evidence_input_identity_sha256")
        identity_hash_ok = hashlib.sha256(_canonical_json(unsealed).encode()).hexdigest() == claimed_identity_sha
    check(
        "evidence_input_identity_self_seal",
        identity_hash_ok,
        "evidence_input_identity_sha256 does not seal the full provenance block",
    )

    labels = results.get("labels", {})
    payload = results.get("payload_quantization")
    fitted_arms = {
        arm_id
        for arm_id, plan in results.get("plans", {}).items()
        if arm_id.startswith("am") and isinstance(plan, dict) and plan.get("method") == "framesamp_history_am_v1"
    }
    payload_ok = isinstance(payload, dict) and bool(fitted_arms) and set(payload) == fitted_arms
    if payload_ok:
        for report in payload.values():
            rows = report.get("layers") if isinstance(report, dict) else None
            payload_ok = payload_ok and (
                report.get("payload_dtype") == "bfloat16"
                and report.get("beta_dtype") == "float32_exact_from_any_sealed_payload"
                and isinstance(rows, list)
                and len(rows) == LAYERS
                and all(
                    row.get("layer_index") == index
                    and all(
                        math.isfinite(float(row.get(metric, float("nan")))) and float(row[metric]) >= 0
                        for metric in (
                            "heldout_relative_l2_float64",
                            "heldout_relative_l2_quantized",
                            "quantization_only_relative_l2",
                        )
                    )
                    for index, row in enumerate(rows)
                )
            )
    check(
        "bf16_payload_quantization_proof",
        labels.get("runtime_dtype") == "bfloat16" and payload_ok,
        "BF16 served-payload quantization proof is absent or incomplete for a fitted arm",
    )

    geometry = results.get("selftest", {}).get("official_geometry")
    expected_geometry = {
        "memory_tokens": MEMORY_TOKENS,
        "layers": LAYERS,
        "query_heads": QUERY_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "action_horizon": ACTION_HORIZON,
    }
    fixture = results.get("fixture_proof")
    episodes = results.get("per_episode")
    exact_fixture = (
        geometry == expected_geometry
        and fixture
        == {
            "complete_pair_count": EVIDENCE_PAIRS,
            "fixture_record_count": 2 * EVIDENCE_PAIRS,
            "required_pair_count": EVIDENCE_PAIRS,
            "exact_pair_count": True,
            "full_memory_tokens": MEMORY_TOKENS,
        }
        and isinstance(episodes, list)
        and len(episodes) == EVIDENCE_PAIRS
        and all(
            episode.get("fit_chunk", {}).get("valid_memory_tokens") == MEMORY_TOKENS
            and episode.get("eval_chunk", {}).get("valid_memory_tokens") == MEMORY_TOKENS
            for episode in episodes
        )
        and all(
            plan.get("valid_frames") == MEMORY_FRAMES and plan.get("valid_tokens") == MEMORY_TOKENS
            for arm_id, plan in results.get("plans", {}).items()
            if arm_id != "memory_destroyed"
        )
    )
    check(
        "official_full512_exact32_fixture_geometry",
        exact_fixture,
        "result does not prove exactly 32 complete pairs with full 32-frame/512-token official geometry",
    )

    selftest = results.get("selftest", {})
    baseline_names = {
        "actions",
        "velocities",
        "denoise_states",
        "plain_actions",
        "plain_velocities",
        "plain_denoise_states",
    }
    identity_names = {"actions", "velocities", "denoise_states"}
    check(
        "restored_checkpoint_patched_baseline_bitwise_identity",
        _bitwise_fields(selftest.get("restored_checkpoint_patched_baseline_identity"), baseline_names),
        "restored-checkpoint patched baseline is not bitwise identical in every required field",
    )
    check(
        "restored_checkpoint_full_kv_beta0_bitwise_identity",
        _bitwise_fields(selftest.get("restored_checkpoint_full_kv_beta0_identity"), identity_names),
        "restored-checkpoint full-KV beta-zero route is not bitwise identical",
    )
    parity = results.get("identity_parity")
    parity_ok = (
        isinstance(parity, list)
        and len(parity) == EVIDENCE_PAIRS
        and all(
            row.get("arm_id") == "identity"
            and row.get("memory_kv_recomputed_per_flow_step") is True
            and _bitwise_fields(row.get("fields"), identity_names)
            for row in parity
        )
    )
    check(
        "all_fixture_full_kv_beta0_bitwise_identity",
        parity_ok,
        "full-KV beta-zero identity is not bitwise on every fixture pair",
    )

    disjoint = isinstance(episodes, list) and len(episodes) == EVIDENCE_PAIRS
    if disjoint:
        for episode in episodes:
            banks = episode.get("query_banks", {})
            fit = banks.get("fit", {})
            heldout = banks.get("heldout", {})
            disjoint = disjoint and (
                banks.get("shared_query_rows") == 0
                and banks.get("disjointness_proof") == "sha256_per_query_row_intersection_empty_v1"
                and int(banks.get("fit_query_rows", 0)) > 0
                and int(banks.get("heldout_query_rows", 0)) > 0
                and fit.get("chunk_role") == "fit_chunk"
                and heldout.get("chunk_role") == "eval_chunk"
                and fit.get("bank_id") != heldout.get("bank_id")
                and fit.get("step_idx") != heldout.get("step_idx")
            )
    check(
        "fit_heldout_query_banks_disjoint",
        disjoint,
        "fit/held-out query-bank disjointness is not proven for every fixture pair",
    )

    timing = results.get("tracing_microbenchmark")
    timing_ok = (
        isinstance(timing, dict)
        and timing.get("measurement_kind") == "python_unrolled_denoiser_microbenchmark_v1"
        and timing.get("speedup_claim_permitted") is False
        and isinstance(timing.get("timing_note"), str)
        and "not comparable end-to-end" in timing["timing_note"]
    )
    check(
        "timing_explicitly_nonclaimable",
        timing_ok,
        "timing is missing or is not explicitly marked non-claimable",
    )

    return {
        "verdict": "VERIFIED" if not failures else "REJECTED",
        "checks": checks,
        "failures": failures,
        "result_sha256": result_sha256,
    }


def label(
    results: dict,
    *,
    result_sha256: str | None = None,
    completion: dict | None = None,
) -> dict:
    proof = audit_result_proof(results, result_sha256=result_sha256, completion=completion)
    released = proof["verdict"] == "VERIFIED"
    threshold = (
        evaluate_thresholds(results)
        if released
        else {
            "verdict": "INDETERMINATE",
            "reason": "evidence remains quarantined: " + "; ".join(proof["failures"]),
            "thresholds": dict(THRESHOLDS),
        }
    )
    return {
        "quarantine_release": {
            "released": released,
            "evidence_proof": proof,
            "oracle_disclaimer": ORACLE_DISCLAIMER,
        },
        "threshold_evaluation": threshold,
    }


def _fmt(value: float) -> str:
    return f"{100 * value:.2f}%"


def velocity_table(results: dict) -> str:
    aggregates = results.get("aggregates", {})
    if not aggregates:
        return "_no aggregates in results_"
    any_arm = next(iter(aggregates.values()))
    times = [float(value) for value in any_arm["flow_times"]]
    header = "| arm | " + " | ".join(f"t={value:g}" for value in times) + " | pooled | action Δ |"
    rule = "|" + "---|" * (len(times) + 3)
    lines = [header, rule]
    for arm_id in sorted(aggregates):
        summary = aggregates[arm_id]
        cells = [
            f"{_fmt(mean)} / {_fmt(p95)}"
            for mean, p95 in zip(
                summary["relative_velocity_error_mean"],
                summary["relative_velocity_error_p95"],
                strict=True,
            )
        ]
        lines.append(
            f"| `{arm_id}` | "
            + " | ".join(cells)
            + f" | {_fmt(summary['relative_velocity_error_pooled_mean'])}"
            + f" | {_fmt(summary['relative_action_error_mean'])} |"
        )
    lines.extend(("", "cells are mean / p95; action Δ is free-running (not teacher forced)"))
    return "\n".join(lines)


def nonclaimable_timing_table(results: dict) -> str:
    timing = results.get("tracing_microbenchmark", {})
    if not timing:
        return "_no tracing microbenchmark in results_"
    lines = [
        "> Development diagnostic only. No latency or speedup claim is permitted from these rows.",
        "",
        "| measurement | diagnostic |",
        "|---|---|",
    ]
    for key in sorted(timing):
        value = timing[key]
        if isinstance(value, dict) and {"seconds_mean", "seconds_min"} <= set(value):
            lines.append(f"| `{key}` | {value['seconds_mean']:.4f} s mean; {value['seconds_min']:.4f} s min |")
        elif isinstance(value, dict) and {"full_kv_bytes", "compact_kv_bytes", "kv_bytes_ratio"} <= set(value):
            lines.append(
                f"| `{key}` | {value['full_kv_bytes'] / 2**20:.2f} MiB -> "
                f"{value['compact_kv_bytes'] / 2**20:.2f} MiB ({value['kv_bytes_ratio']:.2f}x storage) |"
            )
    return "\n".join(lines)


def render(results: dict, block: dict) -> str:
    evaluation = block["threshold_evaluation"]
    release = block["quarantine_release"]
    return "\n".join(
        (
            f"# E0 cell — {results.get('experiment', 'unknown')}",
            "",
            f"run_id `{results.get('identity', {}).get('run_id')}` · evidence "
            f"**{'RELEASED' if release['released'] else 'QUARANTINED'}** · "
            f"threshold **{evaluation['verdict']}** ({evaluation['reason']})",
            "",
            "## rel-Δv per flow time",
            "",
            velocity_table(results),
            "",
            "## Non-claimable tracing microbenchmark",
            "",
            nonclaimable_timing_table(results),
            "",
            "## Evidence labels",
            "",
            "```json",
            json.dumps(block, indent=2, sort_keys=True),
            "```",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="path to the original emitted e0_results.json")
    parser.add_argument("--complete-marker", required=True, help="matching *.complete.json proof")
    parser.add_argument("--out", help="write the labelled results JSON here")
    parser.add_argument("--markdown", help="write the rendered cell here")
    args = parser.parse_args(argv)

    result_path = Path(args.results)
    raw = result_path.read_bytes()
    results = json.loads(raw)
    completion = json.loads(Path(args.complete_marker).read_text(encoding="utf-8"))
    block = label(
        results,
        result_sha256=hashlib.sha256(raw).hexdigest(),
        completion=completion,
    )
    labelled = {**results, **block}
    if args.out:
        Path(args.out).write_text(json.dumps(labelled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered = render(results, block)
    if args.markdown:
        Path(args.markdown).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
