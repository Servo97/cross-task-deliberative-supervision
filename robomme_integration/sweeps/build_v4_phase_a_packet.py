#!/usr/bin/env python3
"""Build the audited, no-submit RoboMME v4 Phase-A training packet.

The review archive digest and SageMaker sanitized source-tree digest are deliberately separate
identities.  This builder refuses any source or workspace placeholder and emits complete launcher
arguments for PickXtimes (all implemented v4 arms) and MoveCube (the three non-workspace arms that
are currently executable).  MoveCube workspace cells remain explicitly blocked until their own
task-bound workspace artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from robomme_integration import launch
from robomme_integration.training.arms import V4_ARM_IDS, WORKSPACE_ARMS
from scripts.launch.launch_guardrails import prepared_source_bundle, source_tree_sha256

REVIEW_ARCHIVE_SHA256 = ""
FROZEN_SOURCE_TREE_SHA256 = ""
PICK_WORKSPACE = {
    "encoder_id": "dd5a17e4929537f9b0472f374081618a5dff5deed31af1969324ded291bd9c15",
    "s3": (
        f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/PickXtimes/"
        "dd5a17e4929537f9b0472f374081618a5dff5deed31af1969324ded291bd9c15/omega"
    ),
    "manifest_sha256": "4ea7ca3f9759c4cc435a7c6e754f6a3227ed9f043df780eb851e37a3989a5775",
}
MOVE_READY_ARMS = ("v4_s0", "v4_q0", "v4_q2")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["packet_sha256"] = hashlib.sha256(_canonical(result).encode()).hexdigest()
    return result


def _verify_source(source_dir: Path, archive: Path) -> None:
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha != REVIEW_ARCHIVE_SHA256:
        raise SystemExit(f"review archive drifted: {archive_sha} != {REVIEW_ARCHIVE_SHA256}")
    with prepared_source_bundle(source_dir, launch.ENTRY, {"SAGEMAKER_PROGRAM": launch.ENTRY}) as (
        staged,
        _entry,
        _environment,
    ):
        tree_sha = source_tree_sha256(staged)
    if tree_sha != FROZEN_SOURCE_TREE_SHA256:
        raise SystemExit(f"frozen sanitized source tree drifted: {tree_sha} != {FROZEN_SOURCE_TREE_SHA256}")


def _arguments(source_dir: Path, task: str, arm: str) -> list[str]:
    arguments = [
        "--scope",
        "single_task",
        "--task",
        task,
        "--arm",
        arm,
        "--source-dir",
        str(source_dir),
        "--hardware",
        "p5",
        "--queue",
        launch.QUEUE,
        "--priority",
        str(launch.PRIORITY),
        "--max-run-seconds",
        str(launch.MAX_RUN_SECONDS),
        "--volume-size-gb",
        "400",
        "--attempt-index",
        "1",
    ]
    if arm in WORKSPACE_ARMS:
        if task != "PickXtimes":
            raise ValueError(f"{task}/{arm} has no task-bound workspace receipt")
        arguments.extend(
            [
                "--workspace-encoder-id",
                PICK_WORKSPACE["encoder_id"],
                "--workspace-s3",
                PICK_WORKSPACE["s3"],
                "--workspace-manifest-sha256",
                PICK_WORKSPACE["manifest_sha256"],
            ]
        )
    return arguments


def _cell(source_dir: Path, task: str, arm: str) -> dict[str, Any]:
    arguments = _arguments(source_dir, task, arm)
    parsed = launch.parser().parse_args([*arguments, "--dry-run"])
    plan = launch.build_plan(parsed, source_dir)
    if plan["source_sha"] != FROZEN_SOURCE_TREE_SHA256:
        raise AssertionError(f"{task}/{arm} plan did not bind the frozen submitted tree")
    scientific = plan["manifest"]["scientific"]
    if scientific["sources"]["robomme_integration"]["sanitized_source_tree_sha256"] != FROZEN_SOURCE_TREE_SHA256:
        raise AssertionError(f"{task}/{arm} scientific source receipt drifted")
    return {
        "task": task,
        "arm": arm,
        "status": "ready_after_canary_and_capacity_gate",
        "run_id": plan["run_id"],
        "attempt_id": plan["attempt_id"],
        "scientific_spec_sha256": plan["manifest"]["scientific_spec_sha256"],
        "manifest_sha256": plan["manifest"]["manifest_sha256"],
        "manifest_s3": plan["manifest_s3"],
        "completion_claim_s3": plan["manifest"]["claims"]["completion"],
        "output_s3": plan["output"],
        "prepared_source_tree_sha256": plan["source_sha"],
        "workspace": scientific["workspace_representation"],
        "training": scientific["training"],
        "mechanism": scientific["mechanism"],
        "launcher_argv": ["python3", "-B", "-m", "robomme_integration.launch", *arguments],
        "launcher_command": shlex.join(["python3", "-B", "-m", "robomme_integration.launch", *arguments]),
        "sealed_manifest": plan["manifest"],
    }


def build(source_dir: Path, archive: Path) -> dict[str, Any]:
    source_dir, archive = source_dir.resolve(), archive.resolve()
    _verify_source(source_dir, archive)
    cells = [_cell(source_dir, "PickXtimes", arm) for arm in sorted(V4_ARM_IDS)]
    cells.extend(_cell(source_dir, "MoveCube", arm) for arm in MOVE_READY_ARMS)
    blocked = [
        {
            "task": "MoveCube",
            "arm": arm,
            "status": "blocked_missing_v4_visreg_task_bound_workspace_artifact",
            "required_action": (
                "port workspace producer to dense/multi-point v2 plus VISReg, canary it, "
                "materialize and publish MoveCube omega, then rebuild packet"
            ),
            "legacy_pair4_disposition": (
                "do_not_run: legacy SIGReg objective and single-point grounding already failed "
                "MoveCube's two-point instruction"
            ),
            "required_move_workspace_chain": [
                "independently audited receipt-only Move dense-v2 p5/H100 task-bound canary",
                "real Move data/supervision/sampler with historical two-point regression",
                "two optimizer steps plus EMA save/cold restore and one-batch materialization",
                "audited producer launch and validated immutable completion receipt",
            ],
        }
        for arm in sorted(V4_ARM_IDS & WORKSPACE_ARMS)
    ]
    return _seal(
        {
            "schema_version": 1,
            "kind": "robomme_v4_phase_a_training_launch_packet",
            "cloud_action": False,
            "claim": "launch_packet_not_training_or_scientific_evidence",
            "source_identity": {
                "review_archive_uri": str(archive),
                "review_archive_bytes_sha256": REVIEW_ARCHIVE_SHA256,
                "submitted_prepared_source_tree_sha256": FROZEN_SOURCE_TREE_SHA256,
                "semantics": "archive byte digest and submitted sanitized source-tree digest are distinct",
                "source_dir": str(source_dir),
            },
            "hardware": {
                "queue": launch.QUEUE,
                "training_plan_arn": None,
                "instance_type": "ml.p5.48xlarge",
                "topology": "1x8 NVIDIA H100 80GB HBM3",
                "sm_use_reserved_capacity": "1",
                "submit_gate": {
                    "v4_policy_p5_h100_canary": {
                        "state": "required_pending_independent_audit_and_runtime_receipt",
                        "receipt": None,
                        "old_p5e_h200_receipt_authorizes_p5": False,
                    },
                    "ordinary_p5_capacity": {
                        "service_quota_code": "L-82E1C851",
                        "service_quota_name": "ml.p5.48xlarge for training job usage",
                        "global_usage": "sum every paginated InProgress SageMaker p5 InstanceCount",
                        "minimum_available_instance_count": 1,
                        "training_plan_capacity_used": False,
                        "target_queue_waiting_statuses_required_zero": [
                            "SUBMITTED",
                            "PENDING",
                            "RUNNABLE",
                            "SCHEDULED",
                            "STARTING",
                        ],
                    },
                    "duplicate_scan": {
                        "batch_service_job_statuses": [
                            "SUBMITTED",
                            "PENDING",
                            "RUNNABLE",
                            "SCHEDULED",
                            "STARTING",
                            "RUNNING",
                        ],
                        "batch_pagination": "must_reach_end_for_every_status",
                        "sagemaker_training_job_pagination": "must_reach_end_for_InProgress",
                        "identity": "run_id_or_canary_id_must_be_absent",
                    },
                    "publication_namespace": "must_be_empty_via_authoritative_paginated_S3_listing",
                    "failure_policy": "any_unknown_or_drift_is_HARD_RED_no_submit",
                },
            },
            "pick_workspace_receipt": PICK_WORKSPACE,
            "ready_cells": cells,
            "blocked_cells": blocked,
        }
    )


def main() -> None:
    global REVIEW_ARCHIVE_SHA256, FROZEN_SOURCE_TREE_SHA256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--review-archive", type=Path, required=True)
    parser.add_argument("--review-archive-sha256", required=True)
    parser.add_argument("--prepared-source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for label, value in (
        ("review archive SHA", args.review_archive_sha256),
        ("prepared source-tree SHA", args.prepared_source_tree_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SystemExit(f"{label} must be 64 lowercase hexadecimal characters")
    REVIEW_ARCHIVE_SHA256 = args.review_archive_sha256
    FROZEN_SOURCE_TREE_SHA256 = args.prepared_source_tree_sha256
    packet = build(args.source_dir, args.review_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"packet={args.output} sha256={packet['packet_sha256']} "
        f"ready={len(packet['ready_cells'])} blocked={len(packet['blocked_cells'])} submit=false"
    )


if __name__ == "__main__":
    main()
