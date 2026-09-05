#!/usr/bin/env python3
"""Approval-gated Stage-S representation PRODUCER launcher (R2).

One node job, phase-gated (features -> encoder -> omega), that turns the sealed inputs (init tap
checkpoint, target50, teacher labels, canonical task prompts) into the canonical-terse source
features, the D0 WSM encoder, and the exact 7,500/7,500 omega cache + content-addressed manifest.

Targets ``robocasa_stage_s_features_entry.sh``. Pins both source archives + the ECR image by digest,
pins the frozen tap checkpoint by its init object-inventory id, stages the immutable D0 run-config,
and writes a deterministic run manifest. ``--dry-run`` is fully offline; real submission needs prior
user approval and ``--confirm-submit``. Account-141 cam-robotics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from datetime import datetime, timezone

from launch_guardrails import (
    DEFAULT_RESULTS_BUCKET,
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    REGION,
    STORAGE_ACCOUNT,
    STUDY_OWNER,
    WSM_ROBOCASA_S3,
    add_guardrail_arguments,
    submit_training_job,
    validate_and_confirm,
)

ENTRY = "robocasa_stage_s_features_entry.sh"
INSTANCE_TYPE = "ml.p5.48xlarge"
STUDY = "long_context_v1"
DEFAULT_OWNER = STUDY_OWNER
INIT_S3 = f"{WSM_ROBOCASA_S3}/pretrain150k/pi05/mg60_bal33/run/149999"
TARGET_DATA_S3 = f"{WSM_ROBOCASA_S3}/datasets/v1.0/target"
LABELS_S3 = f"{WSM_ROBOCASA_S3}/wsm_labels"
STAGED_RUN_CONFIG = "_stage_s_producer_run_config.json"
STAGED_MANIFEST_NAME = "_stage_s_producer_run_manifest.json"
NUM_GPUS = 8
RETRY = {"attempts": 1}

# The immutable D0 encoder objective (packet 03). Its SHA is pinned in the run manifest and, via the
# trainer, in the encoder provenance -> encoder_id. lambda_align=1.0 because D0's declared objective
# INCLUDES the SigReg alignment (the historical lambda_align=0 node entry was the drift TODO 1 fixes).
D0_RUN_CONFIG = {
    "lambda_align": 1.0,
    "target_mode": "next",
    "dropout": 1.0,
    "steps": 100000,
    "batch_size": 8,
    "lr": 3e-4,
    "warmup_steps": 1000,
    "min_lr_frac": 0.1,
    "input_norm": False,
    "val_frac": 0.1,
    "seed_split": 20260722,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMAGE = re.compile(
    rf"^{EXECUTION_ACCOUNT}\.dkr\.ecr\.{REGION}\.amazonaws\.com/"
    r"[a-z0-9][a-z0-9._/-]*@sha256:([0-9a-f]{64})$"
)


def study_root(owner: str) -> str:
    if not _OWNER.fullmatch(owner):
        raise SystemExit(f"invalid --user storage owner {owner!r}")
    return f"s3://{DEFAULT_RESULTS_BUCKET}/{owner}/wsm_robocasa/studies/{STUDY}"


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_addressed_archive(uri: str, *, component: str, root: str) -> str:
    match = re.fullmatch(rf"{re.escape(root)}/code/{re.escape(component)}/([0-9a-f]{{64}})\.tgz", uri)
    if match is None:
        raise SystemExit(f"--{component}-source-s3 must be {root}/code/{component}/<64hex>.tgz; got {uri}")
    return match.group(1)


def image_digest(image_uri: str) -> str:
    match = _IMAGE.fullmatch(image_uri)
    if match is None:
        raise SystemExit("--image-uri must be an account-141 ECR URI pinned as <repo>@sha256:<64hex>")
    return match.group(1)


def _seal_manifest(value: dict) -> tuple[dict, str]:
    sealed = dict(value)
    sealed.pop("manifest_sha256", None)
    checksum = hashlib.sha256(_canonical_json(sealed).encode("utf-8")).hexdigest()
    sealed["manifest_sha256"] = checksum
    return sealed, _canonical_json(sealed)


def resolve_source_dir(value: str | None) -> pathlib.Path:
    path = pathlib.Path(value or pathlib.Path(__file__).resolve().parents[3] / "internal_training").resolve()
    if not (path / ENTRY).is_file():
        raise SystemExit(f"source-dir {path} is missing {ENTRY}")
    return path


def build_plan(args: argparse.Namespace) -> dict:
    if type(args.attempt_index) is not int or args.attempt_index < 1:
        raise SystemExit("--attempt-index must be a positive integer")
    root = study_root(args.user)
    wsmv2_sha = content_addressed_archive(args.wsmv2_source_s3, component="wsmv2", root=root)
    openpi_sha = content_addressed_archive(args.openpi_source_s3, component="openpi", root=root)
    container_sha = image_digest(args.image_uri)
    if not _HEX64.fullmatch(args.feature_source_inventory_id):
        raise SystemExit("--feature-source-inventory-id must be 64 lowercase hex characters")
    if not _HEX64.fullmatch(args.task_prompt_manifest_sha256):
        raise SystemExit("--task-prompt-manifest-sha256 must be 64 lowercase hex characters")
    expected_prompt = (
        f"{root}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{args.task_prompt_manifest_sha256}.json"
    )
    if args.task_prompt_manifest_s3 != expected_prompt:
        raise SystemExit(f"--task-prompt-manifest-s3 must be {expected_prompt}")

    run_config_json = _canonical_json(D0_RUN_CONFIG)
    # Hash EXACTLY the staged file bytes (the submit tail appends "\n" when writing
    # STAGED_RUN_CONFIG) — the node entry sha256sums the FILE, and hashing the newline-less string
    # made the encoder-phase verification fail deterministically (exit 58, producer attempt-3).
    run_config_sha = hashlib.sha256((run_config_json + "\n").encode("utf-8")).hexdigest()
    phases = args.phases
    smoke = phases == "features" and bool(args.features_smoke_tasks)
    run_kind = "producer_smoke" if smoke else "producer"

    output_s3 = f"{root}/producer/{run_kind}/{{run_id}}"  # run_id filled after spec hash
    spec = {
        "schema_version": 1,
        "study": STUDY,
        "run_kind": run_kind,
        "phases": phases,
        "backbone": "pi0.5",
        "feature_source": {"tap_checkpoint_s3": INIT_S3, "inventory_id": args.feature_source_inventory_id},
        "data": {
            "target_data_s3": TARGET_DATA_S3,
            "labels_s3": LABELS_S3,
            "tasks": 50,
            "demos_per_task": 150,
            "episode_subsample_seed": 0,
        },
        "task_prompt_manifest": {"uri": expected_prompt, "sha256": args.task_prompt_manifest_sha256},
        "run_config": {**D0_RUN_CONFIG, "sha256": run_config_sha},
        "sources": {
            "wsmv2": {"uri": args.wsmv2_source_s3, "sha256": wsmv2_sha},
            "openpi": {"uri": args.openpi_source_s3, "sha256": openpi_sha},
            "image": {"uri": args.image_uri, "sha256": container_sha},
        },
        "infrastructure": {
            "execution_account": EXECUTION_ACCOUNT,
            "storage_account": STORAGE_ACCOUNT,
            "queue": args.queue,
            "role": args.role,
            "instance_type": INSTANCE_TYPE,
            "priority": args.priority,
            "max_run_seconds": args.max_run_seconds,
            "num_gpus": NUM_GPUS,
            "attempts": RETRY["attempts"],
            "attempt_index": args.attempt_index,
        },
        "features_smoke_tasks": args.features_smoke_tasks or None,
    }
    spec_sha = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
    run_id = f"{run_kind}-{spec_sha[:16]}"
    output_s3 = f"{root}/producer/{run_kind}/{run_id}"
    manifest_s3 = f"{root}/manifests/runs/{run_kind}/{run_id}.json"
    manifest, manifest_json = _seal_manifest(
        {**spec, "run_id": run_id, "spec_sha256": spec_sha, "output_s3": output_s3, "manifest_s3": manifest_s3}
    )

    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "WSM_REPO_S3": args.wsmv2_source_s3,
        "OPENPI_FORK_S3": args.openpi_source_s3,
        "TAP_CKPT_S3": INIT_S3,
        "FEATURE_SOURCE_INVENTORY_ID": args.feature_source_inventory_id,
        "TARGET_DATA_S3": TARGET_DATA_S3,
        "LABELS_S3": LABELS_S3,
        "TASK_PROMPT_MANIFEST_S3": expected_prompt,
        "TASK_PROMPT_MANIFEST_SHA256": args.task_prompt_manifest_sha256,
        "STUDY_ROOT": root,
        "OUTPUT_S3": output_s3,
        "STAGE_S_PRODUCER_PHASES": phases,
        # Resume from a prior attempt's checkpoint-synced outputs (same wsmv2 archive required —
        # the source manifest re-validates every restored file by content, so this is fail-closed).
        **({"RESUME_PRODUCER_OUTPUT_S3": args.resume_output_s3} if args.resume_output_s3 else {}),
        "RUN_CONFIG_SOURCE": STAGED_RUN_CONFIG,
        "RUN_CONFIG_SHA256": run_config_sha,
        "NUM_GPUS": str(NUM_GPUS),
        "RUN_MANIFEST_SOURCE": STAGED_MANIFEST_NAME,
        "RUN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "RUN_MANIFEST_S3": manifest_s3,
        "STAGE_S_RUN_ID": run_id,
        "WANDB_PROJECT": "wsm-robocasa",
        "WANDB_RUN_GROUP": f"long-context-{run_kind}",
    }
    if args.features_smoke_tasks:
        environment["FEATURES_SMOKE_TASKS"] = args.features_smoke_tasks
    oversized = {k: len(v.encode()) for k, v in environment.items() if len(v.encode()) > 512}
    if oversized:
        raise AssertionError(f"SageMaker environment value exceeds 512 bytes: {oversized}")
    return {
        "run_id": run_id,
        "run_kind": run_kind,
        "output_s3": output_s3,
        "manifest_s3": manifest_s3,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "environment": environment,
        "run_config_json": run_config_json,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=DEFAULT_OWNER)
    # Third instance of the §22.5 bug: --user is the frozen S3 storage prefix (sarvesh.patil, which
    # every content address in the study is minted under and which can never move), while the live
    # submitting identity is sarvesh.patil.pi@tri.global. Deriving the SCP tag from --user tags a
    # DEACTIVATED address and org SCP p-ahpdy5vv denies the submit.
    parser.add_argument(
        "--owner-email",
        default=OWNER_EMAIL,
        help="value of the required tri.owner.email SCP tag; independent of --user",
    )
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--wsmv2-source-s3", required=True)
    parser.add_argument("--openpi-source-s3", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument(
        "--feature-source-inventory-id",
        required=True,
        help="init object-inventory id pinning the frozen H300+MG tap checkpoint",
    )
    parser.add_argument("--task-prompt-manifest-s3", required=True)
    parser.add_argument("--task-prompt-manifest-sha256", required=True)
    parser.add_argument(
        "--resume-output-s3",
        default=None,
        help="prior attempt's OUTPUT_S3; restores its checkpoint-synced outputs so "
        "extract+tap are skipped (features phase still re-validates strictly)",
    )
    parser.add_argument("--phases", default="features,encoder,omega", help="comma subset of features,encoder,omega")
    parser.add_argument(
        "--features-smoke-tasks", default="", help="run only the features phase over these tasks (a producer smoke)"
    )
    parser.add_argument("--attempt-index", type=int, default=1)
    add_guardrail_arguments(parser)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    validate_and_confirm(args)
    source_dir = resolve_source_dir(args.source_dir)
    plan = build_plan(args)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    owner = args.user.replace(".", "-")
    job_name = f"{owner}-stage-s-{plan['run_kind']}-{plan['run_id'][-16:]}-{stamp}"[:63].rstrip("-")
    print(
        f"run_kind={plan['run_kind']} phases={args.phases} run_id={plan['run_id']}\n"
        f"  image={args.image_uri}\n  wsmv2={args.wsmv2_source_s3}\n  openpi={args.openpi_source_s3}\n"
        f"  tap_ckpt={INIT_S3}\n  prompt_manifest={args.task_prompt_manifest_s3}\n"
        f"  output={plan['output_s3']}\n"
        f"  manifest={plan['manifest_s3']} sha256={plan['manifest']['manifest_sha256']}\n"
        f"  queue={args.queue} priority={args.priority} max_run={args.max_run_seconds}s dry={args.dry_run}"
    )
    if args.dry_run:
        print("  [DRY RUN: offline; no AWS SDK load or submission]")
        print(json.dumps(plan["manifest"], sort_keys=True, indent=2))
        print("  SUBMISSION READY only after explicit approval and --confirm-submit")
        return
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=args.image_uri,
        instance_type=INSTANCE_TYPE,
        volume_size=1000,  # org SCP caps EBS; S0's 1000GB submits fine, and ~400GB covers the cache
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": args.owner_email},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.arm", "Value": plan["run_kind"]},
            {"Key": "wsm.run_kind", "Value": plan["run_kind"]},
            {"Key": "wsm.run_id", "Value": plan["run_id"]},
        ],
        retry_config=RETRY,
        job_name=job_name,
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=args.secrets_manager_arn,
        confirmed=args.confirm_submit,
        disable_profiler=True,
        staged_source_files={
            STAGED_MANIFEST_NAME: plan["manifest_json"] + "\n",
            STAGED_RUN_CONFIG: plan["run_config_json"] + "\n",
        },
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
