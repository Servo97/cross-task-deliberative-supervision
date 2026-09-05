#!/usr/bin/env python3
"""Build or submit the receipt-only FrameSamp B1 p5 canary."""

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
from robomme_integration.eval import framesamp_b1_cloud as contract  # noqa: E402
from robomme_integration.eval.framesamp_am_r1_launch import stage_checkpoint  # noqa: E402
from robomme_integration.eval.framesamp_b1_canary import (  # noqa: E402
    LANES,
    validate_canary_packet,
)

ENTRY = contract.ENTRY


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree(source_dir: Path) -> str:
    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}) as (
        staged,
        _,
        _,
    ):
        return source_tree_sha256(staged)


def build(
    *,
    source_dir: Path,
    packet_path: Path,
    overlay_root: Path,
    checkpoint_archive: Path,
) -> dict:
    source_dir = source_dir.resolve(strict=True)
    packet_path = packet_path.resolve(strict=True)
    overlay_root = overlay_root.resolve(strict=True)
    checkpoint_archive = checkpoint_archive.resolve(strict=True)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("B1 packet must be a JSON object")
    validate_canary_packet(packet, source_root=source_dir.parent, overlay_root=overlay_root)
    packet_file_sha = _sha_file(packet_path)
    if packet_file_sha != contract.PACKET_FILE_SHA256:
        raise ValueError("B1 packet file SHA differs from the cloud contract")
    if _sha_file(checkpoint_archive) != contract.CHECKPOINT_ARCHIVE_SHA256:
        raise ValueError("B1 checkpoint archive SHA differs from the released asset")
    source_sha = _source_tree(source_dir)
    entry_sha = _sha_file(source_dir / ENTRY)
    identity = {
        "packet_file_sha256": packet_file_sha,
        "packet_sha256": packet["packet_sha256"],
        "prepared_source_tree_sha256": source_sha,
        "overlay_manifest_sha256": contract.OVERLAY_MANIFEST_SHA256,
        "overlay_source_tree_sha256": contract.OVERLAY_SOURCE_TREE_SHA256,
        "patched_policy_sha256": packet["source"]["patched_policy_sha256"],
        "patched_mem_buffer_sha256": packet["source"]["patched_mem_buffer_sha256"],
        "canary_lanes": [
            {"lane": lane, "task": task, "episode": episode} for lane, (task, episode) in enumerate(LANES)
        ],
    }
    identity_sha = contract.sha_json(identity)
    canary_id = f"fs-b1-p5-canary-{identity_sha[:20]}"
    namespace = f"{launch.STUDY_ROOT}/manifests/canaries/framesamp_b1/{canary_id}"
    manifest = contract.seal(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.KIND,
            "canary_id": canary_id,
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
            "assets": {
                "checkpoint_archive": {
                    "uri": contract.CHECKPOINT_ARCHIVE_S3,
                    "sha256": contract.CHECKPOINT_ARCHIVE_SHA256,
                },
                "checkpoint_semantic_sha256": contract.CHECKPOINT_SEMANTIC_SHA256,
                "eval_runtime": {
                    "uri": contract.EVAL_RUNTIME_S3,
                    "sha256": contract.EVAL_RUNTIME_SHA256,
                },
                "policy_runtime": {
                    "source": "upstream_uv_lock",
                    "jax": "0.5.3",
                    "orbax_checkpoint": "0.11.13",
                },
                "upstream_repo": contract.UPSTREAM_REPO,
                "upstream_commit": contract.UPSTREAM_COMMIT,
                "vision": {"uri": contract.VISION_S3, "sha256": contract.VISION_SHA256},
            },
            "infrastructure": {
                "queue": contract.QUEUE,
                "instance_type": contract.INSTANCE_TYPE,
                "topology": "1x8 NVIDIA H100 80GB HBM3",
                "priority": 100,
                "max_run_seconds": 14_400,
                "volume_size_gb": 250,
                "training_plan_arn": None,
                "sm_use_reserved_capacity": "1",
            },
            "publication": {
                "namespace_s3": namespace,
                "receipt_s3": f"{namespace}/canary.complete.json",
                "create_once": True,
                "only_allowed_object": "canary.complete.json",
                "scored_evidence": False,
            },
        },
        "manifest_sha256",
    )
    contract.validate_manifest(manifest)
    environment = {
        "FS_B1_CANARY_ID": canary_id,
        "FS_B1_MANIFEST_SHA256": manifest["manifest_sha256"],
        "FS_B1_NAMESPACE_S3": namespace,
        "FS_B1_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "FS_B1_EXPECTED_SOURCE_SHA256": source_sha,
        "FS_B1_PACKET_FILE_SHA256": packet_file_sha,
        "FS_B1_PACKET_SHA256": packet["packet_sha256"],
        "FS_B1_OVERLAY_MANIFEST_SHA256": contract.OVERLAY_MANIFEST_SHA256,
        "FS_B1_OVERLAY_SOURCE_TREE_SHA256": contract.OVERLAY_SOURCE_TREE_SHA256,
        "FS_B1_CHECKPOINT_S3": contract.CHECKPOINT_ARCHIVE_S3,
        "FS_B1_CHECKPOINT_ARCHIVE_SHA256": contract.CHECKPOINT_ARCHIVE_SHA256,
        "FS_B1_CHECKPOINT_SEMANTIC_SHA256": contract.CHECKPOINT_SEMANTIC_SHA256,
        "ROBOMME_EVAL_RUNTIME_S3": contract.EVAL_RUNTIME_S3,
        "ROBOMME_EVAL_RUNTIME_SHA256": contract.EVAL_RUNTIME_SHA256,
        "ROBOMME_EVAL_VISION_S3": contract.VISION_S3,
        "ROBOMME_EVAL_VISION_SHA256": contract.VISION_SHA256,
        "ROBOMME_EVAL_UPSTREAM_REPO": contract.UPSTREAM_REPO,
        "ROBOMME_EVAL_UPSTREAM_COMMIT": contract.UPSTREAM_COMMIT,
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    contract.validate_environment(manifest, environment)
    return {
        "canary_id": canary_id,
        "job_name": f"sarvesh-rmme-fs-b1-{identity_sha[:20]}",
        "manifest": manifest,
        "environment": environment,
        "source_tree_sha256": source_sha,
        "checkpoint_archive": str(checkpoint_archive),
        "checkpoint_s3": contract.CHECKPOINT_ARCHIVE_S3,
        "staged_source_files": {
            contract.STAGED_MANIFEST: contract.canonical(manifest) + "\n",
            contract.STAGED_PACKET: packet_path.read_text(encoding="utf-8"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--allow-backlog", action="store_true")
    args = parser.parse_args(argv)
    plan = build(
        source_dir=args.source_dir,
        packet_path=args.packet,
        overlay_root=args.overlay_root,
        checkpoint_archive=args.checkpoint_archive,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "canary_id": plan["canary_id"],
                "job_name": plan["job_name"],
                "manifest_sha256": plan["manifest"]["manifest_sha256"],
                "prepared_source_tree_sha256": plan["source_tree_sha256"],
                "receipt_s3": plan["manifest"]["publication"]["receipt_s3"],
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
    collector = collect_canary_backlog_admission if args.allow_backlog else collect_canary_admission
    admission = collector(
        canary_id=plan["canary_id"],
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
        volume_size=250,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.kind", "Value": "framesamp-b1-canary"},
            {"Key": "wsm.canary_id", "Value": plan["canary_id"]},
        ],
        retry_config={"attempts": 1},
        job_name=plan["job_name"],
        queue=contract.QUEUE,
        role=ROLE_ARN,
        priority=100,
        max_run_seconds=14_400,
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
