#!/usr/bin/env python3
"""Build or submit the canary-gated FS-R1 twelve-cell p5 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/launch"))
from launch_guardrails import (  # noqa: E402
    OWNER_EMAIL,
    PROJECT_TAG,
    ROLE_ARN,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
)

from robomme_integration import launch  # noqa: E402
from robomme_integration.cloud_admission_p5 import (  # noqa: E402
    collect_canary_admission,
    collect_canary_backlog_admission,
)
from robomme_integration.eval import framesamp_am_r1_screen_cloud as contract  # noqa: E402
from robomme_integration.eval.framesamp_am_r1_cloud import (  # noqa: E402
    validate_canary_receipt,
)
from robomme_integration.eval.framesamp_am_r1_cloud import (  # noqa: E402
    validate_manifest as validate_canary_manifest,
)
from robomme_integration.eval.framesamp_am_r1_launch import stage_checkpoint  # noqa: E402
from robomme_integration.eval.framesamp_am_r1_packet import validate as validate_packet  # noqa: E402

ENTRY = contract.ENTRY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_tree(source_dir: Path) -> str:
    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}) as (staged, _, _):
        return source_tree_sha256(staged)


def build(
    *,
    source_dir: Path,
    packet_path: Path,
    canary_plan_path: Path,
    canary_receipt_path: Path,
) -> dict:
    source_dir = source_dir.resolve(strict=True)
    packet_path = packet_path.resolve(strict=True)
    canary_plan_path = canary_plan_path.resolve(strict=True)
    canary_receipt_path = canary_receipt_path.resolve(strict=True)
    packet = _load(packet_path)
    validate_packet(packet)
    canary_plan = _load(canary_plan_path)
    canary_manifest = canary_plan.get("manifest")
    if not isinstance(canary_manifest, dict):
        raise ValueError("FS-R1 canary plan omitted its manifest")
    validate_canary_manifest(canary_manifest)
    canary_receipt = _load(canary_receipt_path)
    validate_canary_receipt(canary_receipt, canary_manifest)
    packet_file_sha = _sha256_file(packet_path)
    if (
        canary_manifest["identity"]["packet_file_sha256"] != packet_file_sha
        or canary_manifest["identity"]["packet_sha256"] != packet["packet_sha256"]
        or canary_manifest["identity"]["source_files"] != packet["oracle_cells"][0]["identity"]["source_files"]
    ):
        raise ValueError("FS-R1 screen packet is not the packet exercised by the H100 canary")
    source_sha = _source_tree(source_dir)
    entry_sha = _sha256_file(source_dir / ENTRY)
    canary_receipt_file_sha = _sha256_file(canary_receipt_path)
    canary_receipt_s3 = canary_manifest["publication"]["receipt_s3"]
    identity = {
        "packet_file_sha256": packet_file_sha,
        "packet_sha256": packet["packet_sha256"],
        "prepared_source_tree_sha256": source_sha,
        "canary_id": canary_manifest["canary_id"],
        "canary_manifest_sha256": canary_manifest["manifest_sha256"],
        "canary_receipt_sha256": canary_receipt["receipt_sha256"],
        "canary_receipt_file_sha256": canary_receipt_file_sha,
        "canary_receipt_s3": canary_receipt_s3,
        "episode_indices": list(range(8)),
        "scientific_cells": 12,
    }
    identity_sha = contract.sha_json(identity)
    screen_id = f"fs-r1-screen-{identity_sha[:20]}"
    namespace = f"{launch.STUDY_ROOT}/evaluations/framesamp_am_r1/{screen_id}"
    results = [
        {
            "cell_id": cell["cell_id"],
            "result_s3": f"{namespace}/cells/{cell['cell_id']}/result.complete.json",
        }
        for cell in packet["oracle_cells"]
    ]
    assets = dict(canary_manifest["assets"])
    manifest = contract.seal(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.KIND,
            "screen_id": screen_id,
            "identity": identity,
            "identity_sha256": identity_sha,
            "source": {
                "prepared_source_tree_sha256": source_sha,
                "entry": ENTRY,
                "entry_bytes_sha256": entry_sha,
                "submitted_entry_mode": "0o755",
                "sagemaker_runtime_entry_mode": "0o777",
                "staged_files_excluded_from_tree_sha256": sorted(contract.STAGED_FILES),
            },
            "assets": assets,
            "infrastructure": {
                "queue": contract.QUEUE,
                "instance_type": contract.INSTANCE_TYPE,
                "topology": "1x8 NVIDIA H100 80GB HBM3",
                "priority": 100,
                "max_run_seconds": 86_400,
                "volume_size_gb": 300,
                "training_plan_arn": None,
                "sm_use_reserved_capacity": "1",
                "cell_schedule": "12 sequential cells x 8 concurrent episode lanes",
            },
            "publication": {
                "namespace_s3": namespace,
                "results": results,
                "completion_s3": f"{namespace}/screen.complete.json",
                "create_once": True,
                "scored_evidence": True,
                "statistical_evidence": False,
            },
        },
        "manifest_sha256",
    )
    contract.validate_manifest(manifest)
    environment = {
        "FS_R1_SCREEN_ID": screen_id,
        "FS_R1_SCREEN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "FS_R1_SCREEN_NAMESPACE_S3": namespace,
        "FS_R1_SCREEN_COMPLETION_S3": manifest["publication"]["completion_s3"],
        "FS_R1_EXPECTED_SOURCE_SHA256": source_sha,
        "FS_R1_PACKET_FILE_SHA256": packet_file_sha,
        "FS_R1_PACKET_SHA256": packet["packet_sha256"],
        "FS_R1_CANARY_ID": canary_manifest["canary_id"],
        "FS_R1_CANARY_MANIFEST_SHA256": canary_manifest["manifest_sha256"],
        "FS_R1_CANARY_RECEIPT_SHA256": canary_receipt["receipt_sha256"],
        "FS_R1_CANARY_RECEIPT_FILE_SHA256": canary_receipt_file_sha,
        "FS_R1_CANARY_RECEIPT_S3": canary_receipt_s3,
        "FS_R1_CHECKPOINT_S3": assets["checkpoint_archive"]["uri"],
        "FS_R1_CHECKPOINT_ARCHIVE_SHA256": assets["checkpoint_archive"]["sha256"],
        "FS_R1_CHECKPOINT_SEMANTIC_SHA256": assets["checkpoint_semantic_sha256"],
        "ROBOMME_EVAL_RUNTIME_S3": assets["eval_runtime"]["uri"],
        "ROBOMME_EVAL_RUNTIME_SHA256": assets["eval_runtime"]["sha256"],
        "ROBOMME_EVAL_VISION_S3": assets["vision"]["uri"],
        "ROBOMME_EVAL_VISION_SHA256": assets["vision"]["sha256"],
        "ROBOMME_EVAL_UPSTREAM_REPO": assets["upstream_repo"],
        "ROBOMME_EVAL_UPSTREAM_COMMIT": assets["upstream_commit"],
        "FS_R1_OVERLAY_MANIFEST_SHA256": assets["policy_overlay_manifest_sha256"],
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    contract.validate_environment(manifest, environment)
    contract.validate_staged_inputs(
        manifest=manifest,
        packet=packet,
        packet_file_sha256=packet_file_sha,
        canary_manifest=canary_manifest,
        canary_receipt=canary_receipt,
        canary_receipt_file_sha256=canary_receipt_file_sha,
    )
    return {
        "screen_id": screen_id,
        "job_name": f"sarvesh-rmme-fs-screen-{identity_sha[:20]}",
        "manifest": manifest,
        "environment": environment,
        "source_tree_sha256": source_sha,
        "staged_source_files": {
            contract.STAGED_MANIFEST: contract.canonical(manifest) + "\n",
            contract.STAGED_PACKET: packet_path.read_text(encoding="utf-8"),
            contract.STAGED_CANARY_MANIFEST: contract.canonical(canary_manifest) + "\n",
            contract.STAGED_CANARY_RECEIPT: canary_receipt_path.read_text(encoding="utf-8"),
        },
        "checkpoint_archive": canary_plan["checkpoint_archive"],
        "checkpoint_s3": canary_plan["checkpoint_s3"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument(
        "--allow-backlog",
        action="store_true",
        help="Queue this exact screen even with existing p5 waiting work or zero free nodes.",
    )
    args = parser.parse_args(argv)
    plan = build(
        source_dir=args.source_dir,
        packet_path=args.packet,
        canary_plan_path=args.canary_plan,
        canary_receipt_path=args.canary_receipt,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "screen_id": plan["screen_id"],
                "job_name": plan["job_name"],
                "manifest_sha256": plan["manifest"]["manifest_sha256"],
                "prepared_source_tree_sha256": plan["source_tree_sha256"],
                "completion_s3": plan["manifest"]["publication"]["completion_s3"],
                "cloud_action": False,
            },
            sort_keys=True,
            indent=2,
        )
    )
    if not args.confirm_submit:
        print("DRY RUN ONLY — no p5 submission")
        return 0
    stage_checkpoint(plan)
    admission_collector = collect_canary_backlog_admission if args.allow_backlog else collect_canary_admission
    admission = admission_collector(
        canary_id=plan["screen_id"],
        job_name=plan["job_name"],
        namespace_s3=plan["manifest"]["publication"]["namespace_s3"],
    )
    print(json.dumps(admission, sort_keys=True, indent=2))
    result = submit_training_job(
        entry=ENTRY,
        source_dir=args.source_dir.resolve(),
        environment=plan["environment"],
        image_uri=launch.IMAGE,
        instance_type=contract.INSTANCE_TYPE,
        volume_size=300,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.kind", "Value": "framesamp-am-r1-screen"},
            {"Key": "wsm.screen_id", "Value": plan["screen_id"]},
        ],
        retry_config={"attempts": 1},
        job_name=plan["job_name"],
        queue=contract.QUEUE,
        role=ROLE_ARN,
        priority=100,
        max_run_seconds=86_400,
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_tree_sha256"],
        staged_source_files=plan["staged_source_files"],
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
