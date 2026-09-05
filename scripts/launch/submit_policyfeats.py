#!/usr/bin/env python
"""Submit the WSM POLICY-FEATURE precompute (frozen WSM encoder -> per-demo w_t) to the TRI SageMaker
Batch queue. Runs the FROZEN WSM encoder over the cached frozen-backbone features (no VLM re-run) and
writes per-demo workspace latents w_t to S3, ready for the WSM-conditioned VLA post-train to read.

Cheap: fans out generate_policy_features across the node's GPUs (one task-shard each); the cost is the
one-time feature-cache S3 sync. Mirrors submit_wsm.py.

  SM=~/Research/envs/sm_launch/bin/python
  # repackage the wsmv2 tarball first so the node gets the new precompute code:
  bash scripts/launch/package_wsmv2_to_s3.sh
  $SM scripts/launch/submit_policyfeats.py --dry-run
  $SM scripts/launch/submit_policyfeats.py --wsm-step 65000 --confirm-submit
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
    resolve_user,
    submit_training_job,
    validate_and_confirm,
)

INSTANCE_TYPE = "ml.p5.48xlarge"  # 8xH100; precompute shards one task-set per GPU
DEFAULT_IMAGE_USER = IMAGE_OWNER
ENTRY = "robocasa_policyfeat_entry.sh"
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


def resolve_source_dir(arg):
    p = pathlib.Path(
        arg or os.environ.get("WSM_SOURCE_DIR") or pathlib.Path(__file__).resolve().parents[3] / "internal_training"
    )
    if not (p / ENTRY).exists():
        raise SystemExit(f"source-dir {p} missing {ENTRY} — pass --source-dir <internal_training>")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--image-user", default=DEFAULT_IMAGE_USER)
    ap.add_argument("--source-dir", default=None)
    add_guardrail_arguments(ap, default_max_run_seconds=86400)
    ap.add_argument("--backbone", default="groot", choices=["groot", "pi"])
    ap.add_argument("--wsm-exp", default="groot_wsm_base", help="WSM run name under wsm_runs/")
    ap.add_argument("--wsm-step", type=int, default=65000, help="WSM encoder checkpoint step to use")
    ap.add_argument("--num-gpus", type=int, default=8)
    args = ap.parse_args()

    validate_and_confirm(args)

    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir)
    image_uri = f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    base = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa"
    out_s3 = f"{base}/wsm_policy_feats/{args.backbone}_step{args.wsm_step}"
    env = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "WSM_REPO_S3": f"{base}/code/wsmv2.tgz",
        "CACHE_S3": f"{base}/wsm_cache",
        "WSM_CKPT_S3": f"{base}/wsm_runs/{args.wsm_exp}/wsm_step{args.wsm_step}.pt",
        "OUTPUT_S3": out_s3,
        "POLFEAT_BACKBONE": args.backbone,
        "NUM_GPUS": str(args.num_gpus),
    }
    name = f"{user.replace('.', '-')}-wsm-polfeat-{args.backbone}-{args.wsm_step}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"[
        :63
    ].rstrip("-")
    print(
        f"user={user} image={image_uri}\n  source_dir={source_dir}\n  wsm_ckpt={env['WSM_CKPT_S3']}\n"
        f"  out={out_s3}\n  backbone={args.backbone} gpus={args.num_gpus} "
        f"queue={args.queue} priority={args.priority} instance={INSTANCE_TYPE} dry={args.dry_run}"
    )
    if args.dry_run:
        print("  [DRY RUN] env:", {k: v for k, v in env.items() if k != "SAGEMAKER_PROGRAM"})
        return
    res = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=env,
        image_uri=image_uri,
        instance_type=INSTANCE_TYPE,
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": "GROOT-DEXJOCO"},
            {"Key": "tri.owner.email", "Value": f"{user}@tri.global"},
        ],
        retry_config=RETRY,
        job_name=name,
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=args.secrets_manager_arn,
        confirmed=args.confirm_submit,
        disable_profiler=True,
    )
    print(f"  QUEUED ✓ arn={getattr(res[0], 'job_arn', '?') if res else '?'}")


if __name__ == "__main__":
    main()
