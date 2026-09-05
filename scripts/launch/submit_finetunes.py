#!/usr/bin/env python
"""Submit the wsmv2 RoboCasa365 PHASE-2 TARGET-FINETUNE runs (Step-0 base post-train: 50 target
tasks, combined-30% = 150 demos/task, balancing OFF) to the TRI SageMaker Batch TrainingQueue.

These are the BASELINE policies the WSM work is measured against — one per BALANCED pretrain arm:
  * pi05_on   <- pretrain150k/pi05/mg60_bal33/run/149999          (the 'horse before the cart' bet)
  * groot_on  <- pretrain150k/groot/mg60_bal33/.../checkpoint-150000

Architecture (user-chosen, 2026-06-19): ship the wsmv2 repo as an S3 tarball and run its VALIDATED
finetune launchers (vla_training/train/train_base/finetune_{pi05,groot}_17.py) on-node via the
robocasa_{pi05,groot}_finetune_entry.sh entries (which reuse the proven pretrain bootstrap). The
combined-50 soup + the seed-0 '150_demos' selection live in the wsmv2 recipe, IDENTICAL across
backbones and identical to the WSM label/feature keep-set.

  # package the repo first (uploads wsmv2.tgz to S3):  bash scripts/launch/package_wsmv2_to_s3.sh
  # dry-run (prints env, no submit):
  /path/to/sm_launch/bin/python scripts/launch/submit_finetunes.py --dry-run
  # submit both:
  /path/to/sm_launch/bin/python scripts/launch/submit_finetunes.py --confirm-submit
  # subset:
  ... submit_finetunes.py --only pi05_on --confirm-submit
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
DEFAULT_IMAGE_USER = IMAGE_OWNER

PI05_ENTRY = "robocasa_pi05_finetune_entry.sh"
GROOT_ENTRY = "robocasa_groot_finetune_entry.sh"
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
    """internal_training dir (ships the FT entry scripts). Default: <repo-parent>/internal_training."""
    p = pathlib.Path(
        arg or os.environ.get("WSM_SOURCE_DIR") or pathlib.Path(__file__).resolve().parents[3] / "internal_training"
    )
    if not (p / PI05_ENTRY).exists():
        raise SystemExit(f"source-dir {p} missing {PI05_ENTRY} — pass --source-dir <internal_training>")
    return p


def build_jobs(output_root: str, pretrain_root: str, repo_s3: str) -> dict:
    """One FT job per BALANCED pretrain arm. The wsmv2 launcher reads steps/LR/batch from the YAML;
    submit only wires the ckpt init (INIT_S3), the repo tarball, and the output location."""

    def pi05_on():
        return dict(
            entry=PI05_ENTRY,
            env={
                "INIT_S3": f"{pretrain_root}/pi05/mg60_bal33/run/149999",
                "WSM_REPO_S3": repo_s3,
                "WSM_FT_CONFIG": "scripts/configs/train/pi05_target_finetune.yaml",
                "OUTPUT_S3": f"{output_root}/pi05_bal33",
                "WANDB_RUN_GROUP": "wsm-targetft-pi05",
            },
        )

    def groot_on():
        return dict(
            entry=GROOT_ENTRY,
            env={
                "INIT_S3": f"{pretrain_root}/groot/mg60_bal33/groot-mg60-bal33/checkpoint-150000",
                "WSM_REPO_S3": repo_s3,
                "WSM_FT_CONFIG": "scripts/configs/train/groot17_target_finetune.yaml",
                "OUTPUT_S3": f"{output_root}/groot_bal33",
                "WANDB_RUN_GROUP": "wsm-targetft-groot",
            },
        )

    return {"pi05_on": pi05_on(), "groot_on": groot_on()}


def submit(key, spec, *, user, image_uri, source_dir, stamp, dry, guardrails):
    name = f"{user.replace('.', '-')}-wsm-ft-{key.replace('_', '-')}-{stamp}"[:63].rstrip("-")
    validate_and_confirm(guardrails)
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": spec["entry"],
        "WANDB_PROJECT": "wsm-robocasa",
        **spec["env"],
    }
    print(
        f"\n=== {key} -> {name}\n    entry={spec['entry']}  init={spec['env']['INIT_S3']}\n"
        f"    out={spec['env']['OUTPUT_S3']}\n    queue={guardrails.queue} "
        f"priority={guardrails.priority} max_run={guardrails.max_run_seconds}s"
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
    ap.add_argument("--image-user", default=DEFAULT_IMAGE_USER, help="ECR image owner (shared)")
    ap.add_argument("--source-dir", default=None, help="internal_training dir (FT entry scripts)")
    ap.add_argument(
        "--repo-s3", default=None, help="wsmv2 repo tarball (default: s3://.../<user>/wsm_robocasa/code/wsmv2.tgz)"
    )
    ap.add_argument("--only", default="pi05_on,groot_on")
    add_guardrail_arguments(ap)
    # --- small-task POC knobs (any N tasks, modest post-train) ---
    ap.add_argument("--tasks", default="", help="comma-list of task names to restrict to (WSM_TASKS; empty = all 50)")
    ap.add_argument("--run-tag", default="", help="suffix for OUTPUT_S3 (e.g. poc15)")
    ap.add_argument("--max-steps", type=int, default=0, help="override max_steps (0 = YAML default)")
    ap.add_argument("--save-interval", type=int, default=0, help="override save_interval (0 = YAML default 5000)")
    ap.add_argument(
        "--eval-during-train", action="store_true", help="co-located per-ckpt eval phase after training (doc 14)"
    )
    ap.add_argument("--eval-trials", type=int, default=10, help="rollouts/task for the per-ckpt curve")
    # --- WSM-conditioned pi finetune (online-w_t modulator; gated on POLICY_FEATS_S3 in the entry) ---
    ap.add_argument("--wsm", action="store_true", help="pi WSM-conditioned finetune (one job per --k-window)")
    ap.add_argument("--wsm-cfg", action="store_true", help="pi WSM-CFG finetune (model-free, doc 12; one job)")
    ap.add_argument("--p-drop", type=float, default=0.2, help="WSM-CFG per-component dropout (--wsm-cfg)")
    ap.add_argument("--with-future", action="store_true", help="pi demo-CFG (doc 15): slot -2 = precomputed z")
    ap.add_argument("--z-windows-s3", default=None, help="z4.npz root on S3 (default: wsm_z_windows/pi_100k_matched)")
    ap.add_argument("--k-window", type=int, default=2)
    ap.add_argument("--wsm-step", type=int, default=100000, help="WSM encoder step whose w_t to use")
    # --- co-located CFG eval cadence (doc 14): base every ckpt + guidance sweep every Nth ckpt ---
    ap.add_argument("--encoder-exp", default=None, help="WSM encoder run under wsm_runs/ (default pi_wsm_v1)")
    ap.add_argument(
        "--sweep-scales",
        default="0.1,0.5,1.5,2.0,4.0",
        help="CFG guidance sweep scales (--wsm-cfg + --eval-during-train)",
    )
    ap.add_argument("--sweep-every", type=int, default=4, help="run the guidance sweep at every Nth checkpoint")
    ap.add_argument("--base-scale", type=float, default=1.0, help="base CFG guidance scale evaluated at every ckpt")
    args = ap.parse_args()

    validate_and_confirm(args)

    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir)
    image_uri = f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    base = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa"
    pretrain_root = f"{base}/pretrain150k"
    output_root = f"{base}/target_ft"
    repo_s3 = args.repo_s3 or f"{base}/code/wsmv2.tgz"
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")

    def _poc_overrides() -> dict:
        """Small-task POC env shared by the WSM pi branches (tasks / steps / save / run-tag suffix)."""
        ov = {}
        if args.tasks.strip():
            ov["WSM_TASKS"] = ",".join(t.strip() for t in args.tasks.split(",") if t.strip())
        if args.max_steps:
            ov["WSM_MAX_STEPS"] = str(args.max_steps)
        if args.save_interval:
            ov["WSM_SAVE_INTERVAL"] = str(args.save_interval)
        return ov

    if args.wsm:  # pi WSM finetune: same recipe as pi05_on + the precomputed pi w_t + modulator-only train
        K = args.k_window
        env = {
            "INIT_S3": f"{pretrain_root}/pi05/mg60_bal33/run/149999",
            "WSM_REPO_S3": repo_s3,
            "WSM_FT_CONFIG": "scripts/configs/train/pi05_target_finetune.yaml",
            "POLICY_FEATS_S3": f"{base}/wsm_policy_feats/pi_step{args.wsm_step}",
            "WSM_K_WINDOW": str(K),
            "OUTPUT_S3": f"{base}/target_ft_wsm/pi_step{args.wsm_step}_k{K}",
            "WANDB_RUN_GROUP": "wsm-ft-pi-wsm",
        }
        env.update(_poc_overrides())
        if args.run_tag:
            env["OUTPUT_S3"] += f"_{args.run_tag}"
        spec = dict(entry=PI05_ENTRY, env=env)
        print(
            f"user={user} pi WSM FT k={K} wsm_step={args.wsm_step}\n  init={env['INIT_S3']}\n"
            f"  feats={env['POLICY_FEATS_S3']}\n  out={env['OUTPUT_S3']} dry={args.dry_run}"
        )
        submit(
            f"pi05_wsm_k{K}",
            spec,
            user=user,
            image_uri=image_uri,
            source_dir=source_dir,
            stamp=stamp,
            dry=args.dry_run,
            guardrails=args,
        )
        return

    if args.wsm_cfg:  # pi WSM-CFG finetune (doc 12): same recipe as pi05_on + precomputed pi w_t + zero-init
        K = args.k_window  # conditioner; WSM_CFG=1 selects CFG in the entry.
        env = {
            "INIT_S3": f"{pretrain_root}/pi05/mg60_bal33/run/149999",
            "WSM_REPO_S3": repo_s3,
            "WSM_CFG": "1",
            "WSM_CFG_P_DROP": str(args.p_drop),
            "WSM_FT_CONFIG": "scripts/configs/train/pi05_wsm_cfg_finetune.yaml",
            "POLICY_FEATS_S3": f"{base}/wsm_policy_feats/pi_step{args.wsm_step}",
            "WSM_K_WINDOW": str(K),
            "OUTPUT_S3": f"{base}/target_ft_wsm_cfg/pi_step{args.wsm_step}_k{K}",
            "WANDB_RUN_GROUP": "wsm-ft-pi-cfg",
        }
        if args.with_future:  # pi demo-CFG (doc 15): slot -2 = precomputed fused z (frozen fusion).
            # NOTE this wiring was MISSING on the first demo launch (0713): --with-future parsed but never
            # exported, so the entry's WSM_CFG_WITH_FUTURE gate stayed closed and the job silently trained
            # a plain w_t-CFG replicate. The distinct pi_demo_ output prefix also keeps demo runs from
            # colliding with w_t-CFG runs of the same step/K.
            env.update(
                {
                    "WSM_CFG_WITH_FUTURE": "1",
                    "Z_WINDOWS_S3": args.z_windows_s3 or f"{base}/wsm_z_windows/pi_100k_matched",
                    "OUTPUT_S3": f"{base}/target_ft_wsm_cfg/pi_demo_step{args.wsm_step}_k{K}",
                    "WANDB_RUN_GROUP": "wsm-demo-cfg-pi",
                }
            )
        env.update(_poc_overrides())
        if args.run_tag:
            env["OUTPUT_S3"] += f"_{args.run_tag}"
        if args.eval_during_train:  # co-located CFG eval on the same node after training (doc 14)
            wsm_exp = args.encoder_exp or "pi_wsm_v1"
            env.update(
                {
                    "WSM_EVAL_DURING_TRAIN": "1",
                    "WSM_CFG_EVAL": "1",
                    "EVAL_TRIALS": str(args.eval_trials),
                    "ENCODER_CKPT_S3": f"{base}/wsm_runs/{wsm_exp}/wsm_step{args.wsm_step}.pt",
                    "TASK_LANG_TABLE_S3": f"{base}/wsm_policy_feats/pi_step{args.wsm_step}/task_lang_table.npz",
                    "WSM_SWEEP_SCALES": args.sweep_scales,
                    "WSM_SWEEP_EVERY": str(args.sweep_every),
                    "WSM_BASE_SCALE": str(args.base_scale),
                }
            )
            if args.with_future:  # demo serve at eval: frozen fusion + registry demos (eval entry pi branch)
                env.update(
                    {
                        "WSM_DEMO_CFG": "1",
                        "DEMO_TOKENS_S3": f"{base}/wsm_demo_tokens/pi_100k_matched",
                        "FUSION_CKPT_S3": f"{base}/wsm2_runs/pi_100k_matched/wsm2_step20000.pt",
                    }
                )
        spec = dict(entry=PI05_ENTRY, env=env)
        print(
            f"user={user} pi WSM-CFG FT k={K} wsm_step={args.wsm_step} p_drop={args.p_drop} "
            f"with_future={args.with_future} eval_during_train={args.eval_during_train}\n"
            f"  init={env['INIT_S3']}\n  feats={env['POLICY_FEATS_S3']}\n  out={env['OUTPUT_S3']} "
            f"poc={_poc_overrides()} dry={args.dry_run}"
        )
        submit(
            f"pi05_{'demo_' if args.with_future else ''}cfg_k{K}",
            spec,
            user=user,
            image_uri=image_uri,
            source_dir=source_dir,
            stamp=stamp,
            dry=args.dry_run,
            guardrails=args,
        )
        return

    jobs = build_jobs(output_root, pretrain_root, repo_s3)
    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    # small-task POC overrides (apply to every selected job; defaults leave full-50 behavior unchanged)
    poc = {}
    if args.tasks.strip():
        poc["WSM_TASKS"] = ",".join(t.strip() for t in args.tasks.split(",") if t.strip())
    if args.max_steps:
        poc["WSM_MAX_STEPS"] = str(args.max_steps)
    if args.save_interval:
        poc["WSM_SAVE_INTERVAL"] = str(args.save_interval)
    if args.eval_during_train:  # co-located per-ckpt curve (baseline: WSM_CFG_EVAL=0)
        poc["WSM_EVAL_DURING_TRAIN"] = "1"
        poc["EVAL_TRIALS"] = str(args.eval_trials)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    print(
        f"user={user}  image={image_uri}\n  source_dir={source_dir}\n  repo_s3={repo_s3}\n"
        f"  pretrain_root={pretrain_root}\n  output_root={output_root}\n  poc={poc} tag={tag!r}\n"
        f"  queue={args.queue} instance={INSTANCE_TYPE} dry={args.dry_run}  jobs={keys}"
    )
    for k in keys:
        if k not in jobs:
            raise SystemExit(f"unknown job {k}; valid: {list(jobs)}")
        jobs[k]["env"].update(poc)
        if tag:
            jobs[k]["env"]["OUTPUT_S3"] += tag
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
