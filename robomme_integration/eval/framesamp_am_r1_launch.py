#!/usr/bin/env python3
"""Build, stage assets for, or submit the isolated FS-R1 p5 runtime canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from robomme_integration.eval import framesamp_am_r1_cloud as contract  # noqa: E402
from robomme_integration.eval.launch_p5_campaign import (  # noqa: E402
    RUNTIME_S3,
    RUNTIME_SHA,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    VISION_S3,
    VISION_SHA256,
)

ENTRY = contract.ENTRY
CHECKPOINT_SEMANTIC_SHA256 = "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
CHECKPOINT_ROOT = f"{launch.STUDY_ROOT}/artifacts/robomme/released_checkpoints/perceptual-framesamp-modul"
OVERLAY_MANIFEST_SHA256 = "5344b2a6898fe36cfec2565c2c2bcbab894244cfb30744a357c8585b33b8a85f"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree(source_dir: Path) -> str:
    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}) as (staged, _, _):
        return source_tree_sha256(staged)


def _load_packet(path: Path) -> dict:
    from robomme_integration.eval.framesamp_am_r1_packet import validate

    packet = json.loads(path.read_text(encoding="utf-8"))
    validate(packet)
    if packet["executor"]["state"] != "IMPLEMENTED_PENDING_H100_CANARY":
        raise ValueError("FS-R1 packet is not ready for its H100 canary")
    return packet


def build(*, source_dir: Path, packet_path: Path, checkpoint_archive: Path) -> dict:
    source_dir = source_dir.resolve(strict=True)
    packet_path = packet_path.resolve(strict=True)
    checkpoint_archive = checkpoint_archive.resolve(strict=True)
    packet = _load_packet(packet_path)
    source_sha = _source_tree(source_dir)
    entry_sha = _sha256_file(source_dir / ENTRY)
    archive_sha = _sha256_file(checkpoint_archive)
    checkpoint_s3 = f"{CHECKPOINT_ROOT}/{CHECKPOINT_SEMANTIC_SHA256}/{archive_sha}.tgz"
    source_files = packet["oracle_cells"][0]["identity"]["source_files"]
    identity = {
        "packet_file_sha256": _sha256_file(packet_path),
        "packet_sha256": packet["packet_sha256"],
        "prepared_source_tree_sha256": source_sha,
        "source_files": source_files,
        "checkpoint_archive_sha256": archive_sha,
        "checkpoint_semantic_sha256": CHECKPOINT_SEMANTIC_SHA256,
        "canary_lanes": packet["executor"]["canary_lanes"],
    }
    identity_sha = contract.sha_json(identity)
    canary_id = f"fs-r1-p5-canary-{identity_sha[:20]}"
    namespace = f"{launch.STUDY_ROOT}/manifests/canaries/framesamp_am_r1/{canary_id}"
    manifest = contract.seal(
        {
            "schema_version": 1,
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
            },
            "assets": {
                "checkpoint_archive": {"uri": checkpoint_s3, "sha256": archive_sha},
                "checkpoint_semantic_sha256": CHECKPOINT_SEMANTIC_SHA256,
                "eval_runtime": {"uri": RUNTIME_S3, "sha256": RUNTIME_SHA},
                "policy_runtime": {
                    "source": "upstream_uv_lock",
                    "jax": "0.5.3",
                    "orbax_checkpoint": "0.11.13",
                },
                "policy_overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_repo": UPSTREAM_REPO,
                "vision": {"uri": VISION_S3, "sha256": VISION_SHA256},
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
        "FS_R1_CANARY_ID": canary_id,
        "FS_R1_MANIFEST_SHA256": manifest["manifest_sha256"],
        "FS_R1_NAMESPACE_S3": namespace,
        "FS_R1_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "FS_R1_EXPECTED_SOURCE_SHA256": source_sha,
        "FS_R1_PACKET_FILE_SHA256": identity["packet_file_sha256"],
        "FS_R1_PACKET_SHA256": identity["packet_sha256"],
        "FS_R1_CHECKPOINT_S3": checkpoint_s3,
        "FS_R1_CHECKPOINT_ARCHIVE_SHA256": archive_sha,
        "FS_R1_CHECKPOINT_SEMANTIC_SHA256": CHECKPOINT_SEMANTIC_SHA256,
        "ROBOMME_EVAL_RUNTIME_S3": RUNTIME_S3,
        "ROBOMME_EVAL_RUNTIME_SHA256": RUNTIME_SHA,
        "ROBOMME_EVAL_VISION_S3": VISION_S3,
        "ROBOMME_EVAL_VISION_SHA256": VISION_SHA256,
        "ROBOMME_EVAL_UPSTREAM_REPO": UPSTREAM_REPO,
        "ROBOMME_EVAL_UPSTREAM_COMMIT": UPSTREAM_COMMIT,
        "FS_R1_OVERLAY_MANIFEST_SHA256": OVERLAY_MANIFEST_SHA256,
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    contract.validate_environment(manifest, environment)
    return {
        "canary_id": canary_id,
        "job_name": f"sarvesh-rmme-fs-r1-{identity_sha[:20]}",
        "manifest": manifest,
        "environment": environment,
        "source_tree_sha256": source_sha,
        "checkpoint_archive": str(checkpoint_archive),
        "checkpoint_s3": checkpoint_s3,
        "staged_source_files": {contract.STAGED_MANIFEST: contract.canonical(manifest) + "\n"},
    }


def _aws(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["aws", *arguments, "--region", "us-west-2", "--no-cli-pager"],
        check=False,
        capture_output=True,
        text=True,
    )


def stage_checkpoint(plan: dict) -> None:
    uri = plan["checkpoint_s3"]
    location = uri.removeprefix("s3://")
    bucket, key = location.split("/", 1)
    head = _aws("s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json")
    local = Path(plan["checkpoint_archive"])
    expected_size = local.stat().st_size
    expected_sha = plan["manifest"]["assets"]["checkpoint_archive"]["sha256"]
    if head.returncode == 0:
        value = json.loads(head.stdout)
        if value.get("ContentLength") != expected_size or value.get("Metadata", {}).get("sha256") != expected_sha:
            raise RuntimeError("different checkpoint archive already occupies canonical S3 key")
        return
    upload = _aws(
        "s3",
        "cp",
        str(local),
        uri,
        "--only-show-errors",
        "--metadata",
        f"sha256={expected_sha}",
    )
    if upload.returncode:
        raise RuntimeError(f"checkpoint upload failed: {upload.stderr[-1000:]}")
    head = _aws("s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json")
    if head.returncode:
        raise RuntimeError("checkpoint archive HEAD failed after upload")
    value = json.loads(head.stdout)
    if value.get("ContentLength") != expected_size or value.get("Metadata", {}).get("sha256") != expected_sha:
        raise RuntimeError("checkpoint archive S3 metadata failed read-after-write verification")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage-checkpoint", action="store_true")
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument(
        "--allow-backlog",
        action="store_true",
        help="Queue this exact canary even when p5 has waiting work or zero immediately free nodes.",
    )
    args = parser.parse_args(argv)
    plan = build(
        source_dir=args.source_dir,
        packet_path=args.packet,
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
                "checkpoint_s3": plan["checkpoint_s3"],
                "receipt_s3": plan["manifest"]["publication"]["receipt_s3"],
                "cloud_action": False,
            },
            sort_keys=True,
            indent=2,
        )
    )
    if args.stage_checkpoint:
        stage_checkpoint(plan)
        print("CHECKPOINT_STAGED_AND_VERIFIED")
    if not args.confirm_submit:
        print("DRY RUN ONLY — no p5 submission")
        return 0
    stage_checkpoint(plan)
    admission_collector = collect_canary_backlog_admission if args.allow_backlog else collect_canary_admission
    admission = admission_collector(
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
            {"Key": "wsm.kind", "Value": "framesamp-am-r1-runtime-canary"},
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
