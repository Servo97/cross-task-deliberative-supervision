#!/usr/bin/env python
"""Submit the WSM-CONDITIONED GR00T canary target-finetune (Step 2b) to the TRI SageMaker queue.

Same recipe as the normal GR00T target-FT (inits from the SAME bal33 @150k pretrain ckpt, combined-50
soup, seed-0 150-demo subsample) PLUS the workspace latents w_t injected via a zero-init TokenModulator.
Eval2 is compared to the normal GR00T Eval1 (30.5%) as a clean per-category WSM delta.

--combined (default): ONE launch does precompute (w_t from the frozen encoder over the feature cache)
THEN the post-train, reading w_t from local disk — avoids a second scarce-node queue wait. Without it,
the job assumes w_t already exists in S3 (POLICY_FEATS_S3) from a separate submit_policyfeats run.

  bash scripts/launch/package_wsmv2_to_s3.sh          # ship the WSM launcher + networks + precompute
  SM=~/Research/envs/sm_launch/bin/python
  $SM scripts/launch/submit_wsm_canary.py --dry-run
  $SM scripts/launch/submit_wsm_canary.py --combined --wsm-step 60000 \
      --priority 600 --max-run-seconds 432000 --confirm-submit
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

INSTANCE_TYPE = "ml.p5.48xlarge"  # 8x H100
DEFAULT_IMAGE_USER = IMAGE_OWNER
ENTRY = "robocasa_groot_wsm_finetune_entry.sh"  # post-train only (w_t precomputed in S3)
COMBINED_ENTRY = "robocasa_groot_wsm_canary_combined_entry.sh"  # precompute + post-train in one launch
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


def resolve_source_dir(arg, entry):
    p = pathlib.Path(
        arg or os.environ.get("WSM_SOURCE_DIR") or pathlib.Path(__file__).resolve().parents[3] / "internal_training"
    )
    if not (p / entry).exists():
        raise SystemExit(f"source-dir {p} missing {entry} — pass --source-dir <internal_training>")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--image-user", default=DEFAULT_IMAGE_USER)
    ap.add_argument("--source-dir", default=None)
    ap.add_argument(
        "--repo-s3", default=None, help="wsmv2 tarball (default: s3://.../<user>/wsm_robocasa/code/wsmv2.tgz)"
    )
    add_guardrail_arguments(ap)
    ap.add_argument("--backbone", default="groot", choices=["groot"], help="pi later (no pi WSM encoder yet)")
    ap.add_argument(
        "--wsm-step", type=int, default=60000, help="WSM encoder step whose w_t to use (60k: avoid 100k overfit)"
    )
    ap.add_argument("--k-window", type=int, default=2)
    ap.add_argument("--wsm-exp", default="groot_wsm_base", help="WSM encoder run name under wsm_runs/")
    ap.add_argument("--combined", action="store_true", help="precompute + post-train in ONE launch (default path)")
    ap.add_argument("--config", default="scripts/configs/train/groot17_wsm_canary_finetune.yaml")
    args = ap.parse_args()

    validate_and_confirm(args)

    entry = COMBINED_ENTRY if args.combined else ENTRY
    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir, entry)
    image_uri = f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    base = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa"
    repo_s3 = args.repo_s3 or f"{base}/code/wsmv2.tgz"
    feats_s3 = f"{base}/wsm_policy_feats/{args.backbone}_step{args.wsm_step}"
    env = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": entry,
        "INIT_S3": f"{base}/pretrain150k/groot/mg60_bal33/groot-mg60-bal33/checkpoint-150000",
        "WSM_REPO_S3": repo_s3,
        "POLICY_FEATS_S3": feats_s3,  # combined: w_t backup target; separate: required precomputed input
        "WSM_FT_CONFIG": args.config,
        "WSM_K_WINDOW": str(args.k_window),
        "OUTPUT_S3": f"{base}/target_ft_wsm/{args.backbone}_step{args.wsm_step}_k{args.k_window}",
        "WANDB_PROJECT": "wsm-robocasa",
        "WANDB_RUN_GROUP": "wsm-canary-groot",
    }
    if args.combined:
        env["CACHE_S3"] = f"{base}/wsm_cache"
        env["WSM_CKPT_S3"] = f"{base}/wsm_runs/{args.wsm_exp}/wsm_step{args.wsm_step}.pt"
        env["POLFEAT_BACKBONE"] = args.backbone

    tag = "combined" if args.combined else "ft"
    name = f"{user.replace('.', '-')}-wsm-canary-{tag}-{args.backbone}-{args.wsm_step}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"[
        :63
    ].rstrip("-")
    print(
        f"user={user} image={image_uri}\n  entry={entry}\n  init={env['INIT_S3']}\n"
        + (
            f"  encoder_ckpt={env['WSM_CKPT_S3']}\n  cache={env['CACHE_S3']}\n"
            if args.combined
            else f"  policy_feats(in)={feats_s3}\n"
        )
        + f"  out={env['OUTPUT_S3']}\n  config={args.config} k_window={args.k_window} "
        f"queue={args.queue} priority={args.priority} combined={args.combined} dry={args.dry_run}"
    )
    if args.dry_run:
        print("  [DRY RUN] env:", {k: v for k, v in env.items() if k != "SAGEMAKER_PROGRAM"})
        return
    res = submit_training_job(
        entry=entry,
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
    )
    print(f"  QUEUED ✓ arn={getattr(res[0], 'job_arn', '?') if res else '?'}")


if __name__ == "__main__":
    main()
