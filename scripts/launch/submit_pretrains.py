#!/usr/bin/env python
"""Submit the wsmv2 RoboCasa365 PRETRAIN runs (150k steps, ckpt every 20k) to the TRI SageMaker
Batch TrainingQueue. Reuses the PROVEN internal_training node entries (resume + orbax->S3 sync +
the PyOpenGL 10h-hang fix) with the human300+MimicGen60 configs.

Execution is pinned to account 141's cam-robotics queue/image/role. Durable checkpoints remain under
the canonical account-124 bucket at s3://sagemaker-us-west-2-124224456861/<user>/wsm_robocasa/;
cross-account bucket policy must grant the execution role access. Every submission requires explicit
`--confirm-submit`; dry-run is non-mutating.

  # auto-derive your user from AWS STS, dry-run first:
  /path/to/sm_launch/bin/python scripts/launch/submit_pretrains.py --dry-run
  # submit all 4:
  /path/to/sm_launch/bin/python scripts/launch/submit_pretrains.py --confirm-submit
  # subset:
  ... submit_pretrains.py --only pi05_off,groot_off --confirm-submit
"""

from __future__ import annotations

import argparse
import os
import pathlib
from datetime import datetime, timezone

from launch_guardrails import (
    DEFAULT_RESULTS_BUCKET,
    EXECUTION_ACCOUNT,
    IMAGE_OWNER,
    REGION,
    add_guardrail_arguments,
    load_aws_sdk,
    prepared_source_bundle,
    resolve_user,
    validate_and_confirm,
    validate_caller_account,
)

INSTANCE_TYPE = "ml.p5.48xlarge"  # 8x H100
DEFAULT_IMAGE_USER = IMAGE_OWNER  # the shared thin DexJoCo ECR image owner (read across the account)

PI05_ENTRY = "robocasa_pi05_train_entry.sh"
GROOT_ENTRY = "robocasa_groot_train_entry.sh"
RETRY = {
    "attempts": 2,
    "evaluateOnExit": [
        {
            "action": "RETRY",
            "onStatusReason": "Received status from SageMaker:InternalServerError: We encountered an internal error. Please try again.",
        },
        {"action": "EXIT", "onStatusReason": "*"},
    ],
}


def resolve_source_dir(arg: str | None) -> pathlib.Path:
    """--source-dir, else $WSM_SOURCE_DIR, else <repo-parent>/internal_training (the proven entries+configs)."""
    p = pathlib.Path(
        arg or os.environ.get("WSM_SOURCE_DIR") or pathlib.Path(__file__).resolve().parents[3] / "internal_training"
    )
    if not (p / PI05_ENTRY).exists():
        raise SystemExit(f"source-dir {p} missing {PI05_ENTRY} — pass --source-dir <internal_training>")
    return p


def build_jobs(output_root: str) -> dict:
    def pi05(cfg, tag):
        return dict(
            entry=PI05_ENTRY,
            env={
                "WSM_CONFIG": cfg,
                "WSM_NUM_STEPS": "150000",
                "WSM_SAVE_INTERVAL": "20000",
                "WSM_VISION_LR_SCALE": "1.0",
                "WSM_BATCH_SIZE": "256",
                "WSM_PEAK_LR": "5e-5",
                "WSM_EXP_NAME": "run",
                "WANDB_RUN_GROUP": "wsm-pretrain150k-pi05",
                "OUTPUT_S3": f"{output_root}/pi05/{tag}/run",
            },
        )

    def groot(exp, tag, balance):
        env = {
            "MAX_STEPS": "150000",
            "SAVE_STEPS": "20000",
            "LEARNING_RATE": "3e-5",
            "GLOBAL_BATCH": "64",
            "GRAD_ACCUM": "4",
            "NUM_GPUS": "8",
            "SAVE_TOTAL_LIMIT": "10",
            "GROOT_EXP": exp,
            "WANDB_RUN_GROUP": "wsm-pretrain150k-groot",
            "OUTPUT_S3": f"{output_root}/groot/{tag}",
        }
        if balance:
            env["GROOT_BALANCE"] = "1"  # entry builds 3 source-group specs (mg/h-atomic/h-composite) @ 1/3
        return dict(entry=GROOT_ENTRY, env=env)

    return {
        "pi05_off": pi05("pi05_rc_mg60", "mg60_off"),
        "pi05_on": pi05("pi05_rc_mg60_bal33", "mg60_bal33"),
        "groot_off": groot("groot-mg60-off", "mg60_off", balance=False),
        "groot_on": groot("groot-mg60-bal33", "mg60_bal33", balance=True),
    }


def submit(key, spec, *, user, image_uri, source_dir, stamp, dry, guardrails):
    name = f"{user.replace('.', '-')}-wsm-pt150k-{key.replace('_', '-')}-{stamp}"[:63].rstrip("-")
    validate_and_confirm(guardrails)
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": spec["entry"],
        "WANDB_PROJECT": "wsm-robocasa",
        **spec["env"],
    }
    print(
        f"\n=== {key} -> {name}\n    entry={spec['entry']}  out={spec['env']['OUTPUT_S3']}"
        f"\n    queue={guardrails.queue} priority={guardrails.priority} "
        f"max_run={guardrails.max_run_seconds}s"
    )
    if dry:
        print("    [DRY RUN: no source upload and no AWS submission] env:", {k: v for k, v in spec["env"].items()})
        return

    boto3, sagemaker, Estimator, TrainingQueue = load_aws_sdk()
    validate_caller_account(boto3)
    with prepared_source_bundle(source_dir, spec["entry"], environment, guardrails.secrets_manager_arn) as (
        safe_source_dir,
        safe_entry,
        safe_environment,
    ):
        est = Estimator(
            entry_point=safe_entry,
            source_dir=str(safe_source_dir),
            sagemaker_session=sagemaker.Session(boto_session=boto3.session.Session(region_name=REGION)),
            role=guardrails.role,
            image_uri=image_uri,
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            input_mode="FastFile",
            max_run=guardrails.max_run_seconds,
            environment=safe_environment,
            volume_size=1000,
            keep_alive_period_in_seconds=300,
            tags=[
                {"Key": "tri.project", "Value": "GROOT-DEXJOCO"},
                {"Key": "tri.owner.email", "Value": f"{user}@tri.global"},
            ],
        )
        q = TrainingQueue(queue_name=guardrails.queue)
        res = q.map(
            est,
            inputs=[None],
            job_names=[name],
            priority=guardrails.priority,
            share_identifier="default",
            retry_config=RETRY,
            timeout={"attemptDurationSeconds": guardrails.max_run_seconds},
        )
    print(f"    QUEUED ✓  arn={getattr(res[0], 'job_arn', '?') if res else '?'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="checkpoint S3 prefix owner (default: auto from AWS STS)")
    ap.add_argument(
        "--image-user", default=DEFAULT_IMAGE_USER, help=f"ECR image owner (shared; default {DEFAULT_IMAGE_USER})"
    )
    ap.add_argument(
        "--source-dir", default=None, help="internal_training dir (default: <repo-parent>/internal_training)"
    )
    ap.add_argument("--only", default="pi05_off,pi05_on,groot_off,groot_on")
    add_guardrail_arguments(ap)
    args = ap.parse_args()

    validate_and_confirm(args)

    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir)
    image_uri = f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    output_root = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa/pretrain150k"
    jobs = build_jobs(output_root)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")

    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    print(
        f"user={user}  image={image_uri}\n  source_dir={source_dir}\n  output_root={output_root}\n"
        f"  queue={args.queue} instance={INSTANCE_TYPE} dry={args.dry_run}  jobs={keys}"
    )
    for k in keys:
        if k not in jobs:
            raise SystemExit(f"unknown job {k}; valid: {list(jobs)}")
        submit(
            k,
            jobs[k],
            user=user,
            image_uri=image_uri,
            source_dir=source_dir,
            stamp=stamp,
            dry=args.dry_run,
            guardrails=args,
        )


if __name__ == "__main__":
    main()
