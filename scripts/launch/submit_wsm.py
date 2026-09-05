#!/usr/bin/env python
"""Submit the WSM-ENCODER training (groot backbone features) to the TRI SageMaker Batch queue.

Step 2a of the disciplined plan: train the WorkspaceModel encoder+decoder on the cached frozen
groot_on features (7422 demos, 50 target tasks) with recon/occupancy loss ONLY (lambda_align=0 —
cross-demo alignment deferred). Pure-PyTorch trainer; the entry builds a minimal torch env and pulls
the cache+labels+node-manifest from S3.

  SM=~/Research/envs/sm_launch/bin/python
  $SM scripts/launch/submit_wsm.py --dry-run
  $SM scripts/launch/submit_wsm.py --steps 100000 --save-every 25000 --confirm-submit

Requires the cache uploaded (scripts/launch/upload_cache_to_s3.sh) + the wsmv2 tarball
(scripts/launch/package_wsmv2_to_s3.sh).
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

INSTANCE_TYPE = "ml.p5.48xlarge"  # 8xH100; the WSM trainer uses 1 GPU (single-process)
DEFAULT_IMAGE_USER = IMAGE_OWNER
ENTRY = "robocasa_wsm_train_entry.sh"
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
    add_guardrail_arguments(ap)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--save-every", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=16, help="DataLoader workers (p5 node has 192 vCPUs)")
    ap.add_argument("--prefetch-factor", type=int, default=4, help="batches prefetched per worker")
    ap.add_argument(
        "--backbone",
        choices=["groot", "pi"],
        default="groot",
        help="groot -> wsm_cache + train_wsm_from_groot_17; pi -> wsm_cache_pi + train_wsm_from_pi_05",
    )
    ap.add_argument("--exp", default=None, help="run name under wsm_runs/ (default <backbone>_wsm_base)")
    ap.add_argument("--resume", action="store_true", help="resume from the latest wsm_step*.pt in OUTPUT_S3 (auto)")
    args = ap.parse_args()

    validate_and_confirm(args)

    # backbone-specific S3 cache + trainer module (both trainers share _wsm_train_common + the entry)
    cache_sub = "wsm_cache" if args.backbone == "groot" else "wsm_cache_pi"
    trainer_module = (
        "workspace_models.train.train_wsm_base.train_wsm_from_groot_17"
        if args.backbone == "groot"
        else "workspace_models.train.train_wsm_base.train_wsm_from_pi_05"
    )
    exp = args.exp or f"{args.backbone}_wsm_base"

    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir)
    image_uri = f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    base = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa"
    env = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "WSM_REPO_S3": f"{base}/code/wsmv2.tgz",
        "CACHE_S3": f"{base}/{cache_sub}",
        "LABELS_S3": f"{base}/wsm_labels",
        "MANIFEST_S3": f"{base}/{cache_sub}/manifest_node.parquet",
        "OUTPUT_S3": f"{base}/wsm_runs/{exp}",
        "WSM_STEPS": str(args.steps),
        "WSM_SAVE_EVERY": str(args.save_every),
        "WSM_BATCH": str(args.batch_size),
        "WANDB_PROJECT": "wsm-robocasa",
        "WSM_NUM_WORKERS": str(args.num_workers),
        "WSM_PREFETCH": str(args.prefetch_factor),
        "WSM_TRAINER_MODULE": trainer_module,
        "WANDB_RUN_GROUP": f"wsm-encoder-{args.backbone}",
        "WSM_RESUME": "auto" if args.resume else "",
    }
    name = f"{user.replace('.', '-')}-wsm-enc-{exp.replace('_', '-')}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"[
        :63
    ].rstrip("-")
    print(
        f"user={user} image={image_uri} backbone={args.backbone}\n  source_dir={source_dir}\n  out={env['OUTPUT_S3']}\n"
        f"  cache={env['CACHE_S3']} trainer={trainer_module}\n"
        f"  steps={args.steps} save_every={args.save_every} batch={args.batch_size} "
        f"workers={args.num_workers} prefetch={args.prefetch_factor} lambda_align=0\n"
        f"  queue={args.queue} instance={INSTANCE_TYPE} dry={args.dry_run}"
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
