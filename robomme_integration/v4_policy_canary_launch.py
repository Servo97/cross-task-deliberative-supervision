#!/usr/bin/env python3
"""Build or submit the isolated v4 GDN8+JEPA+VISReg B64 policy canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robomme_integration import launch
from robomme_integration.cloud_admission_p5 import collect_canary_admission
from robomme_integration.sweeps.build_v4_phase_a_packet import PICK_WORKSPACE
from robomme_integration.training import v4_policy_canary as contract
from scripts.launch.launch_guardrails import OWNER_EMAIL, PROJECT_TAG, prepared_source_bundle, source_tree_sha256

ENTRY = contract.ENTRY
MAX_RUN_SECONDS = 10_800
VOLUME_SIZE_GB = 400


def _source_tree(source_dir: Path) -> str:
    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}) as (staged, _entry, _environment):
        return source_tree_sha256(staged)


def _reference(source_dir: Path) -> dict:
    arguments = [
        "--scope",
        "single_task",
        "--task",
        "PickXtimes",
        "--arm",
        contract.ARM,
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
        "--workspace-encoder-id",
        PICK_WORKSPACE["encoder_id"],
        "--workspace-s3",
        PICK_WORKSPACE["s3"],
        "--workspace-manifest-sha256",
        PICK_WORKSPACE["manifest_sha256"],
        "--dry-run",
    ]
    args = launch.parser().parse_args(arguments)
    return launch.build_plan(args, source_dir)["manifest"]


def build(source_dir: Path) -> dict:
    source_dir = source_dir.resolve()
    tree_sha = _source_tree(source_dir)
    entry_sha = hashlib.sha256((source_dir / ENTRY).read_bytes()).hexdigest()
    reference = _reference(source_dir)
    if reference["scientific"]["sources"]["robomme_integration"]["sanitized_source_tree_sha256"] != tree_sha:
        raise ValueError("canary and production reference prepared-source trees differ")
    identity = {
        "prepared_source_tree_sha256": tree_sha,
        "entry_bytes_sha256": entry_sha,
        "reference_run_id": reference["run_id"],
        "reference_scientific_spec_sha256": reference["scientific_spec_sha256"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "task": "PickXtimes",
        "arm": contract.ARM,
        "batch_size": 64,
        "optimizer_steps": 2,
        "dtype": "bfloat16",
    }
    identity_sha = contract.sha_json(identity)
    canary_id = f"v4-policy-canary-{identity_sha[:20]}"
    namespace = f"{launch.STUDY_ROOT}/manifests/canaries/policy_training/{canary_id}"
    source = {
        "prepared_source_tree_sha256": tree_sha,
        "entry": ENTRY,
        "entry_bytes_sha256": entry_sha,
        "submitted_entry_mode": "0o755",
        "sagemaker_runtime_entry_mode": "0o777",
    }
    manifest = contract.seal(
        {
            "schema_version": 1,
            "kind": contract.KIND,
            "claim": contract.CLAIM,
            "contract": contract.CONTRACT,
            "canary_id": canary_id,
            "identity": identity,
            "identity_sha256": identity_sha,
            "arm": contract.ARM,
            "task": "PickXtimes",
            "execution": {
                "optimizer_steps": 2,
                "final_local_checkpoint_step": 1,
                "batch_size": 64,
                "dtype": "bfloat16",
                "history_dropout": 0.0,
                "checkpoint_scope": "node_local_ephemeral_only",
                "cold_restore": True,
            },
            "source": source,
            "reference_training_manifest": reference,
            "infrastructure": {
                "queue": launch.QUEUE,
                "training_plan_arn": None,
                "instance_type": "ml.p5.48xlarge",
                "topology": "1x8 NVIDIA H100 80GB HBM3",
                "priority": launch.PRIORITY,
                "max_run_seconds": MAX_RUN_SECONDS,
                "volume_size_gb": VOLUME_SIZE_GB,
            },
            "publication": {
                "namespace_s3": namespace,
                "receipt_s3": f"{namespace}/training_canary.complete.json",
                "create_once": True,
                "only_allowed_object": "training_canary.complete.json",
                "production_checkpoint_or_deploy_publication": False,
            },
        },
        "manifest_sha256",
    )
    contract.validate_manifest(manifest)
    scientific = reference["scientific"]
    environment = {
        "ROBOMME_CANARY_KIND": contract.KIND,
        "ROBOMME_CANARY_CLAIM": contract.CLAIM,
        "ROBOMME_CANARY_ID": canary_id,
        "ROBOMME_CANARY_MANIFEST_SHA256": manifest["manifest_sha256"],
        "ROBOMME_CANARY_NAMESPACE_S3": namespace,
        "ROBOMME_CANARY_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "ROBOMME_TASK": "PickXtimes",
        "ROBOMME_ARM": contract.ARM,
        "ROBOMME_DATA_S3": scientific["data"]["dataset_s3"],
        "ROBOMME_DATA_PARENT_INVENTORY_S3": scientific["data"]["parent_inventory_uri"],
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": scientific["data"]["parent_inventory_sha256"],
        "ROBOMME_DATA_DERIVED_INVENTORY_SHA256": scientific["data"]["derived_task_inventory_sha256"],
        "INIT_S3": scientific["initialization"]["checkpoint_s3"],
        "INIT_INVENTORY_S3": scientific["initialization"]["inventory_uri"],
        "INIT_INVENTORY_SHA256": scientific["initialization"]["inventory_sha256"],
        "OPENPI_FORK_S3": scientific["sources"]["openpi"]["uri"],
        "OPENPI_REQUIRED_SENTINEL": "_WSM_V4_ADVANCED",
        "PALIGEMMA_TOKENIZER_S3": scientific["sources"]["tokenizer"]["uri"],
        "PALIGEMMA_TOKENIZER_SHA256": scientific["sources"]["tokenizer"]["sha256"],
        "ROBOMME_WORKSPACE_S3": scientific["workspace_representation"]["omega"]["uri"],
        "ROBOMME_WORKSPACE_ENCODER_ID": scientific["workspace_representation"]["encoder_id"],
        "ROBOMME_WORKSPACE_MANIFEST_SHA256": scientific["workspace_representation"]["omega"]["manifest_sha256"],
        "WSM_MAX_STEPS": "2",
        "WSM_SAVE_INTERVAL": "2",
        "WSM_WARMUP_STEPS": "1",
        "WSM_PEAK_LR": "5e-5",
        "WSM_DECAY_STEPS": "2",
        "WSM_DECAY_LR": "5e-6",
        "WSM_SEED": "0",
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    forbidden = {
        "OUTPUT_S3",
        "RUN_MANIFEST_SOURCE",
        "RUN_MANIFEST_SHA256",
        "RUN_MANIFEST_S3",
        "PRODUCER_CLAIM_S3",
        "COMPLETION_CLAIM_S3",
        "CHECKPOINT_TREE_MANIFEST_ROOT",
        "ROBOMME_SCIENTIFIC_SPEC_SHA256",
        "ROBOMME_FINAL_STEP",
        "ROBOMME_RUN_ID",
    }
    if forbidden & environment.keys():
        raise ValueError("production publication environment leaked into canary")
    return {
        "canary_id": canary_id,
        "job_name": f"sarvesh-rmme-v4-canary-{identity_sha[:20]}",
        "manifest": manifest,
        "environment": environment,
        "source_tree_sha256": tree_sha,
        "staged_source_files": {contract.STAGED_MANIFEST: contract.canonical(manifest) + "\n"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-submit", action="store_true")
    args = parser.parse_args()
    plan = build(args.source_dir)
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
        print("DRY RUN ONLY — no cloud write")
        return
    admission = collect_canary_admission(
        canary_id=plan["canary_id"],
        job_name=plan["job_name"],
        namespace_s3=plan["manifest"]["publication"]["namespace_s3"],
    )
    print(json.dumps(admission, sort_keys=True, indent=2))
    result = launch.submit_training_job(
        entry=ENTRY,
        source_dir=args.source_dir.resolve(),
        environment=plan["environment"],
        image_uri=launch.IMAGE,
        instance_type="ml.p5.48xlarge",
        volume_size=VOLUME_SIZE_GB,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.kind", "Value": "v4-policy-training-canary"},
            {"Key": "wsm.canary_id", "Value": plan["canary_id"]},
        ],
        retry_config={"attempts": 1},
        job_name=plan["job_name"],
        queue=launch.QUEUE,
        role=launch.ROLE_ARN,
        priority=launch.PRIORITY,
        max_run_seconds=MAX_RUN_SECONDS,
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_tree_sha256"],
        staged_source_files=plan["staged_source_files"],
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
