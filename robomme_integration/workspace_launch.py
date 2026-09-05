#!/usr/bin/env python3
"""Approval-gated p5e/p5 launcher for all-16 RoboMME workspace artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
LAUNCH_UTILS = REPO_ROOT / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_UTILS))
from launch_guardrails import (  # noqa: E402
    DEFAULT_RESULTS_BUCKET,
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    QUEUE,
    ROLE_ARN,
    STUDY_OWNER,
    TRAINING_PLAN_QUEUE,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
    training_plan_arn,
)

from robomme_integration.fleet.task_inventory import (  # noqa: E402
    CANONICAL_PARENT_SHA256,
    CANONICAL_TASK_DERIVED_SHA256,
)
from robomme_integration.launch import IMAGE, OPENPI, OPENPI_SHA  # noqa: E402
from robomme_integration.training.single_task import TASK_ORDER, task_manifest_sha256  # noqa: E402

ENTRY = "workspace_gpu_entry.sh"
STAGED_MANIFEST = "_robomme_workspace_pair_manifest.json"
OWNER = STUDY_OWNER
STUDY = "long_context_v1"
CAMPAIGN = "uniform_gpu_v1"
UPSTREAM_REPO_ID = "Yinpei/robomme_preprocessed_data"
UPSTREAM_REVISION = "ddf0baf55b633cc6657dcd53ac0e089a273de612"
P5E_PRIORITY = 400
P5_PRIORITY = 1
MAX_RUN_SECONDS = 24 * 3600
VOLUME_SIZE_GB = 400
STUDY_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/studies/{STUDY}"
DATA_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/datasets/robomme/v1/lerobot_all16"
DATA_INVENTORY = f"{STUDY_ROOT}/manifests/inventories/data/{CANONICAL_PARENT_SHA256}.json"
ARTIFACT_ROOT = f"{STUDY_ROOT}/artifacts/robomme/workspace"
PLAN_ARN = training_plan_arn(TRAINING_PLAN_QUEUE)
HARDWARE = {
    "p5e": {
        "queue": TRAINING_PLAN_QUEUE,
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "priority": P5E_PRIORITY,
        "reserved_capacity": "0",
        "training_plan_arn": PLAN_ARN,
    },
    "p5": {
        "queue": QUEUE,
        "instance_type": "ml.p5.48xlarge",
        "accelerator": "8xH100",
        "priority": P5_PRIORITY,
        "reserved_capacity": "1",
        "training_plan_arn": None,
    },
}
PAIRS = tuple(tuple(TASK_ORDER[index : index + 2]) for index in range(0, len(TASK_ORDER), 2))


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _source_dir(value: str | None) -> Path:
    root = Path(value or Path(__file__).resolve().parent).resolve()
    required = (root / ENTRY, root / "training/workspace_gpu_producer.py")
    if not all(path.is_file() for path in required):
        raise SystemExit(f"invalid isolated workspace producer source: {root}")
    return root


def _validate(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.confirm_submit:
        raise SystemExit("submission blocked: obtain explicit user approval, then pass --confirm-submit")
    hardware = HARDWARE[args.hardware]
    if args.queue != hardware["queue"] or args.role != ROLE_ARN:
        raise SystemExit(f"workspace artifact production must use the cam-robotics {args.hardware} queue")
    if args.priority != hardware["priority"] or args.max_run_seconds != MAX_RUN_SECONDS:
        raise SystemExit(
            f"workspace {args.hardware} production requires priority {hardware['priority']} and an exact 24-hour cap"
        )
    if args.volume_size_gb != VOLUME_SIZE_GB:
        raise SystemExit("workspace artifact production requires the measured-safe 400 GiB ephemeral volume")
    if args.attempt_index < 1:
        raise SystemExit("--attempt-index must be positive")
    if args.hardware == "p5e" and PLAN_ARN is None:
        raise SystemExit("p5e queue lost its required training-plan ARN")


def build_plan(args: argparse.Namespace, source_dir: Path) -> dict:
    _validate(args)
    hardware = HARDWARE[args.hardware]
    tasks = PAIRS[args.pair_index]
    with prepared_source_bundle(source_dir, ENTRY, {}, args.secrets_manager_arn) as (staged, _entry, _env):
        source_sha = source_tree_sha256(staged)
    task_records = []
    for task in tasks:
        scientific = {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "benchmark": "RoboMME",
            "task": task,
            "task_manifest_sha256": task_manifest_sha256(task),
            "task_inventory_sha256": CANONICAL_TASK_DERIVED_SHA256[task],
            "data_parent_inventory_sha256": CANONICAL_PARENT_SHA256,
            "upstream": {"repo_id": UPSTREAM_REPO_ID, "revision": UPSTREAM_REVISION},
            "producer": {
                "hardware": args.hardware,
                "accelerator": hardware["accelerator"],
                "devices": 4,
            },
            "representation": {
                "steps": 10_000,
                "batch_size": 32,
                "devices": 4,
                "seed": 0,
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "min_lag": 40,
                "future_delta": 20,
                "history_stride": 10,
                "max_history": 128,
                "loss_weights": {"occ": 0.1, "jepa": 0.1, "sigreg": 0.05},
                "omega_dim": 512,
            },
            "sources": {
                "robomme_integration_sha256": source_sha,
                "openpi": {"uri": OPENPI, "sha256": OPENPI_SHA},
                "image": IMAGE,
            },
        }
        scientific_sha = _digest(scientific)
        run_id = f"wrep-v1-{task.lower()}-seed0-{scientific_sha[:16]}"
        claim_s3 = f"{STUDY_ROOT}/manifests/claims/workspace/{CAMPAIGN}/{task}/{run_id}.complete.json"
        task_records.append(
            {
                "task": task,
                "run_id": run_id,
                "scientific_spec_sha256": scientific_sha,
                "task_inventory_sha256": CANONICAL_TASK_DERIVED_SHA256[task],
                "claim_s3": claim_s3,
                "scientific": scientific,
            }
        )
    pair_scientific = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "pair_index": args.pair_index,
        "tasks": [{"task": item["task"], "run_id": item["run_id"]} for item in task_records],
        "topology": {
            "hardware": args.hardware,
            "node": hardware["accelerator"],
            "lanes": 2,
            "devices_per_lane": 4,
        },
        "source_tree_sha256": source_sha,
    }
    pair_id = f"wrep-pair{args.pair_index}-{_digest(pair_scientific)[:16]}"
    attempt_id = f"{pair_id}-attempt{args.attempt_index}"
    pair_claim = f"{STUDY_ROOT}/manifests/claims/workspace/{CAMPAIGN}/pairs/{pair_id}.complete.json"
    manifest_s3 = f"{STUDY_ROOT}/manifests/runs/workspace/{pair_id}/{attempt_id}.json"
    producer_claim = f"{STUDY_ROOT}/manifests/claims/workspace/{CAMPAIGN}/pairs/{pair_id}/producers/{attempt_id}.json"
    manifest = {
        "schema_version": 1,
        "kind": "robomme_all16_workspace_pair_attempt",
        "campaign": CAMPAIGN,
        "pair_id": pair_id,
        "attempt_id": attempt_id,
        "source_tree_sha256": source_sha,
        "pair_scientific": pair_scientific,
        "tasks": task_records,
        "infrastructure": {
            "provider": "aws_sagemaker",
            "account": EXECUTION_ACCOUNT,
            "queue": hardware["queue"],
            "training_plan_arn": hardware["training_plan_arn"],
            "instance_type": hardware["instance_type"],
            "priority": hardware["priority"],
            "max_run_seconds": MAX_RUN_SECONDS,
            "volume_size_gb": VOLUME_SIZE_GB,
            "attempt_index": args.attempt_index,
        },
        "manifest_s3": manifest_s3,
        "claims": {"producer": producer_claim, "completion": pair_claim},
    }
    manifest_json = _canonical(manifest) + "\n"
    manifest_sha = hashlib.sha256(manifest_json.encode()).hexdigest()
    environment = {
        "SM_USE_RESERVED_CAPACITY": hardware["reserved_capacity"],
        "ROBOMME_WORKSPACE_PAIR_ID": pair_id,
        "ROBOMME_WORKSPACE_SOURCE_SHA256": source_sha,
        "ROBOMME_WORKSPACE_PAIR_MANIFEST_SOURCE": STAGED_MANIFEST,
        "ROBOMME_WORKSPACE_PAIR_MANIFEST_SHA256": manifest_sha,
        "ROBOMME_WORKSPACE_PAIR_MANIFEST_S3": manifest_s3,
        "ROBOMME_WORKSPACE_PAIR_PRODUCER_CLAIM_S3": producer_claim,
        "ROBOMME_WORKSPACE_PAIR_COMPLETION_CLAIM_S3": pair_claim,
        "ROBOMME_WORKSPACE_ARTIFACT_ROOT_S3": ARTIFACT_ROOT,
        "OPENPI_FORK_S3": OPENPI,
        "ROBOMME_DATA_S3": DATA_ROOT,
        "ROBOMME_DATA_PARENT_INVENTORY_S3": DATA_INVENTORY,
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": CANONICAL_PARENT_SHA256,
    }
    oversized = {key: len(value.encode()) for key, value in environment.items() if len(value.encode()) > 512}
    if oversized:
        raise SystemExit(f"SageMaker environment values exceed 512 bytes: {oversized}")
    return {
        "pair_id": pair_id,
        "attempt_id": attempt_id,
        "tasks": tasks,
        "manifest_s3": manifest_s3,
        "manifest_json": manifest_json,
        "manifest_sha256": manifest_sha,
        "environment": environment,
        "source_sha": source_sha,
        "pair_claim": pair_claim,
        "hardware": hardware,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pair-index", type=int, choices=range(len(PAIRS)), required=True)
    value.add_argument("--hardware", choices=tuple(HARDWARE), default="p5e")
    value.add_argument("--source-dir")
    value.add_argument("--queue")
    value.add_argument("--role", default=ROLE_ARN)
    value.add_argument("--priority", type=int)
    value.add_argument("--max-run-seconds", type=int, default=MAX_RUN_SECONDS)
    value.add_argument("--volume-size-gb", type=int, default=VOLUME_SIZE_GB)
    value.add_argument("--attempt-index", type=int, default=1)
    value.add_argument("--secrets-manager-arn")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--confirm-submit", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    args.queue = args.queue or HARDWARE[args.hardware]["queue"]
    args.priority = args.priority if args.priority is not None else HARDWARE[args.hardware]["priority"]
    source_dir = _source_dir(args.source_dir)
    plan = build_plan(args, source_dir)
    print(
        f"pair={args.pair_index} tasks={','.join(plan['tasks'])} pair_id={plan['pair_id']} "
        f"attempt={plan['attempt_id']} source={plan['source_sha']}\n"
        f"  manifest={plan['manifest_s3']} sha256={plan['manifest_sha256']}\n"
        f"  completion={plan['pair_claim']}\n"
        f"  hardware={args.hardware} queue={args.queue} "
        f"plan={plan['hardware']['training_plan_arn']} priority={args.priority} "
        f"max_run={args.max_run_seconds}s volume={args.volume_size_gb}GiB dry={args.dry_run}"
    )
    if args.dry_run:
        print(plan["manifest_json"], end="")
        print("DRY RUN ONLY — no AWS SDK loaded and no cloud write performed")
        return
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    job_name = f"sarvesh-rmme-wrep-p{args.pair_index}-{plan['pair_id'][-16:]}-{stamp}"[:63]
    if not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", job_name):
        raise SystemExit(f"invalid SageMaker job name: {job_name}")
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=IMAGE,
        instance_type=plan["hardware"]["instance_type"],
        volume_size=args.volume_size_gb,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.benchmark", "Value": "RoboMME"},
            {"Key": "wsm.artifact_campaign", "Value": CAMPAIGN},
            {"Key": "wsm.pair_id", "Value": plan["pair_id"]},
        ],
        retry_config={"attempts": 1},
        job_name=job_name,
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=args.secrets_manager_arn,
        confirmed=args.confirm_submit,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_sha"],
        staged_source_files={STAGED_MANIFEST: plan["manifest_json"]},
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
