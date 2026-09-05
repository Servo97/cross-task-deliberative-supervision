#!/usr/bin/env python
"""Submit the CFG-CONDITIONED GR00T target-finetune (the model-free POC, doc 12) to the TRI SageMaker queue.

Same recipe as the normal GR00T target-FT (inits from the bal33@150k pretrain, combined-50 soup, seed-0
150-demo subsample) EXCEPT it conditions the action denoiser on the workspace latent via AdaLN, trained with
classifier-free-guidance dropout (the VLM backbone is untouched). Reuses the EXISTING precomputed w_t at
wsm_policy_feats/groot_step<S> (no re-precompute), runs finetune_groot_17_with_wsm_cfg.py via the (now
script-selectable) WSM finetune entry, and writes to a DISTINCT output dir so it never collides with the
injection canary runs.

  bash scripts/launch/package_wsmv2_to_s3.sh           # ship the CFG code first
  SM=~/Research/envs/sm_launch/bin/python
  $SM scripts/launch/submit_wsm_cfg.py --dry-run
  $SM scripts/launch/submit_wsm_cfg.py --wsm-step 50000 --wsm-exp groot_wsm_v2 --confirm-submit
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
ENTRY = "robocasa_groot_wsm_finetune_entry.sh"  # non-combined: w_t already precomputed in S3
CFG_SCRIPT = "vla_training/train/train_base/finetune_groot_17_with_wsm_cfg.py"
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
    ap.add_argument(
        "--repo-s3", default=None, help="wsmv2 tarball (default: s3://.../<user>/wsm_robocasa/code/wsmv2.tgz)"
    )
    add_guardrail_arguments(ap)
    ap.add_argument("--backbone", default="groot", choices=["groot"])
    ap.add_argument("--demo", action="store_true", help="WSMv2 DEMO-CFG (doc 15): frozen fusion + registry demos")
    ap.add_argument("--fusion-s3", default=None, help="wsm2 fusion ckpt on S3 (demo mode)")
    ap.add_argument("--demo-tokens-s3", default=None, help="d.npz root on S3 (demo mode)")
    ap.add_argument("--wsm-step", type=int, default=50000, help="WSM encoder step whose precomputed w_t to use")
    ap.add_argument(
        "--wsm-exp",
        default="groot_wsm_v2",
        help="encoder run name under wsm_runs/ (decode-best later: groot_wsm_base/65k)",
    )
    ap.add_argument("--p-drop", type=float, default=0.2, help="per-component CFG dropout")
    ap.add_argument("--with-future", action="store_true", help="ICL: also condition on w_{t+1} (POC leaves it off)")
    ap.add_argument("--diag-every", type=int, default=100, help="cond-vs-uncond flow-loss diagnostic interval")
    ap.add_argument(
        "--tasks", default="", help="comma-list of task names to restrict to (small-task POC; empty = all 50)"
    )
    ap.add_argument("--run-tag", default="", help="suffix for the output dir + job name (e.g. poc15)")
    ap.add_argument(
        "--max-steps", type=int, default=0, help="override max_steps for a MODEST post-train (0 = YAML default)"
    )
    ap.add_argument(
        "--save-interval", type=int, default=0, help="override checkpoint interval (0 = YAML default 5000)"
    )
    ap.add_argument(
        "--eval-during-train", action="store_true", help="co-located per-ckpt eval phase after training (doc 14)"
    )
    ap.add_argument("--eval-trials", type=int, default=10, help="rollouts/task for the per-ckpt curve")
    ap.add_argument("--base-scale", type=float, default=1.0, help="base CFG guidance scale evaluated at EVERY ckpt")
    ap.add_argument(
        "--sweep-scales", default="0.1,0.5,1.5,2.0,4.0", help="CFG guidance sweep scales (run at every Nth ckpt)"
    )
    ap.add_argument("--sweep-every", type=int, default=4, help="run the guidance sweep at every Nth checkpoint")
    ap.add_argument(
        "--eval-init-robot-base-ref",
        default="",
        help="pin robot spawn fixture for no-nav eval (empty = per-task default)",
    )
    ap.add_argument("--config", default="scripts/configs/train/groot17_wsm_cfg_finetune.yaml")
    args = ap.parse_args()

    validate_and_confirm(args)

    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir)
    image_uri = f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    base = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa"
    repo_s3 = args.repo_s3 or f"{base}/code/wsmv2.tgz"
    feats_s3 = f"{base}/wsm_policy_feats/{args.backbone}_step{args.wsm_step}"  # EXISTING precompute (v2/50k)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_s3 = f"{base}/target_ft_wsm/{args.backbone}_cfg_step{args.wsm_step}{tag}"  # DISTINCT from injection/full runs
    env = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "WSM_FT_SCRIPT": CFG_SCRIPT,  # entry runs the CFG launcher
        "WSM_FT_CONFIG": args.config,
        "INIT_S3": f"{base}/pretrain150k/groot/mg60_bal33/groot-mg60-bal33/checkpoint-150000",
        "WSM_REPO_S3": repo_s3,
        "POLICY_FEATS_S3": feats_s3,
        "OUTPUT_S3": out_s3,
        # CFG knobs read by finetune_groot_17_with_wsm_cfg.py
        "WSM_P_DROP": str(args.p_drop),
        "WSM_WITH_FUTURE": "1" if args.with_future else "0",
        "WSM_DIAG_EVERY": str(args.diag_every),
        "WANDB_PROJECT": "wsm-robocasa",
        "WANDB_RUN_GROUP": "wsm-cfg-groot",
    }
    if args.tasks.strip():  # small-task POC (any N tasks)
        env["WSM_TASKS"] = ",".join(t.strip() for t in args.tasks.split(",") if t.strip())
    if args.max_steps:  # modest post-train
        env["WSM_MAX_STEPS"] = str(args.max_steps)
    if args.save_interval:
        env["WSM_SAVE_INTERVAL"] = str(args.save_interval)
    if args.demo:  # WSMv2 demo-CFG (doc 15)
        env["WSM_DEMO_CFG"] = "1"
        env["WSM_VERIFY_2PASS"] = "1"  # H1 self-check at first eval step
        # demo mode rides the 65k arm: history/targets w.npz + lang table live under groot_65k
        env["POLICY_FEATS_S3"] = f"{base}/wsm_policy_feats/groot_65k"
        env["WSM_FT_SCRIPT"] = "vla_training/train/train_base/finetune_groot_17_with_wsm_demo_cfg.py"
        env["WSM_FT_CONFIG"] = "scripts/configs/train/groot17_wsm_demo_cfg_finetune.yaml"
        env["DEMO_TOKENS_S3"] = args.demo_tokens_s3 or f"{base}/wsm_demo_tokens/orig_65k_matched"
        env["FUSION_CKPT_S3"] = args.fusion_s3 or f"{base}/wsm2_runs/orig_65k_matched/wsm2_step20000.pt"
        env["OUTPUT_S3"] = f"{base}/target_ft_wsm/{args.backbone}_demo_cfg_step{args.wsm_step}{tag}"
        env["WANDB_RUN_GROUP"] = "wsm-demo-cfg-groot"
    if args.eval_during_train:  # co-located per-ckpt curve (doc 14)
        env.update(
            {
                "WSM_EVAL_DURING_TRAIN": "1",
                "WSM_CFG_EVAL": "1",
                "EVAL_TRIALS": str(args.eval_trials),
                "WSM_BASE_SCALE": str(args.base_scale),  # base eval at every ckpt
                "WSM_SWEEP_SCALES": args.sweep_scales,
                "WSM_SWEEP_EVERY": str(args.sweep_every),  # sweep every Nth
                "ENCODER_CKPT_S3": f"{base}/wsm_runs/{args.wsm_exp}/wsm_step{args.wsm_step}.pt",
                "TASK_LANG_TABLE_S3": f"{base}/wsm_policy_feats/{args.backbone}_step{args.wsm_step}/task_lang_table.npz",
            }
        )
        if args.eval_init_robot_base_ref:
            env["INIT_ROBOT_BASE_REF"] = args.eval_init_robot_base_ref
        if args.demo:  # demo artifacts live under groot_65k; must override the eval block's step-derived path
            env["TASK_LANG_TABLE_S3"] = f"{base}/wsm_policy_feats/groot_65k/task_lang_table.npz"
    name = f"{user.replace('.', '-')}-wsm-cfg-{args.backbone}{('-' + args.run_tag) if args.run_tag else ''}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"[
        :63
    ].rstrip("-")
    print(
        f"user={user} image={image_uri}\n  entry={ENTRY}  script={CFG_SCRIPT}\n  init={env['INIT_S3']}\n"
        f"  policy_feats(in)={feats_s3}\n  out={out_s3}\n  tasks={env.get('WSM_TASKS', 'ALL-50')}\n"
        f"  config={args.config} p_drop={args.p_drop} with_future={args.with_future} "
        f"queue={args.queue} priority={args.priority} dry={args.dry_run}"
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
    )
    print(f"  QUEUED ✓ arn={getattr(res[0], 'job_arn', '?') if res else '?'}")


if __name__ == "__main__":
    main()
