#!/usr/bin/env python
"""Submit RoboCasa365 TARGET-split evals for the wsmv2 runs to the TRI SageMaker Batch queue.

Mirrors submit_pretrains.py (per-user, reuses the proven internal_training eval entry
`robocasa_eval_entry.sh` = per-GPU policy-server + torch-free sim venv + aggregate). Defaults to
the `target` split (foundation-model-learning protocol; the proven launcher's `pretrain` default
was the legacy bug). Per arm it auto-resolves the LATEST checkpoint step under the run's S3 output.

  SM=~/Research/envs/sm_launch/bin/python
  $SM scripts/launch/submit_evals.py --dry-run                      # show what would run
  # Submission is fail-closed: pass --confirm-submit only after the user approves the run.
  $SM scripts/launch/submit_evals.py --num-trials 50 --confirm-submit
  $SM scripts/launch/submit_evals.py --only pi05_off --step 60000 --confirm-submit
  $SM scripts/launch/submit_evals.py --ckpt-s3 s3://.../run/150000 --only pi05_off --confirm-submit

NOTE: protocol = eval the POST-TRAINED checkpoint; evaluating a pretrain checkpoint here gives the
"pretrain-only" baseline. Pass --ckpt-root / --ckpt-s3 to point at post-train outputs once they exist.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
from datetime import datetime, timezone

from launch_guardrails import (
    DEFAULT_RESULTS_BUCKET,
    EXECUTION_ACCOUNT,
    IMAGE_OWNER,
    MAX_RUN_SECONDS,
    MULTI_DAY_PRIORITY,
    QUEUE,
    REGION,
    ROLE_ARN,
    add_guardrail_arguments,
    load_aws_sdk,
    prepared_source_bundle,
    require_submission_confirmation,
    resolve_user,
    validate_and_confirm,
    validate_caller_account,
    validate_launch_contract,
)

INSTANCE_TYPE = "ml.p5.48xlarge"
DEFAULT_IMAGE_USER = IMAGE_OWNER
EVAL_ENTRY = "robocasa_eval_entry.sh"
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

# arm -> (MODEL, pretrain-output subpath under <root>, checkpoint-dir layout).
# pi05 orbax: <run>/<step>/ ;  groot HF: <tag>/checkpoint-<step>/
ARMS = {
    "pi05_off": dict(model="pi05", sub="pi05/mg60_off/run", layout="step"),
    "pi05_on": dict(model="pi05", sub="pi05/mg60_bal33/run", layout="step"),
    "groot_off": dict(model="groot", sub="groot/mg60_off", layout="checkpoint"),
    "groot_on": dict(model="groot", sub="groot/mg60_bal33", layout="checkpoint"),
}


def resolve_source_dir(arg):
    p = pathlib.Path(
        arg or os.environ.get("WSM_SOURCE_DIR") or pathlib.Path(__file__).resolve().parents[3] / "internal_training"
    )
    if not (p / EVAL_ENTRY).exists():
        raise SystemExit(f"source-dir {p} missing {EVAL_ENTRY} — pass --source-dir <internal_training>")
    return p


def _ls_dirs(prefix):
    """Immediate sub-'directories' (CommonPrefixes) under an s3 prefix."""
    out = subprocess.run(
        ["aws", "s3", "ls", prefix.rstrip("/") + "/", "--region", REGION], capture_output=True, text=True
    )
    return [ln.split("PRE ", 1)[1].strip().rstrip("/") for ln in out.stdout.splitlines() if "PRE " in ln]


def resolve_ckpt(arm_root, layout, step):
    """Return (ckpt_s3, step). layout 'step' -> <root>/<N>; 'checkpoint' -> <root>/checkpoint-<N>.
    step None -> the largest numeric step present."""
    dirs = _ls_dirs(arm_root)
    if layout == "step":
        steps = sorted(int(d) for d in dirs if d.isdigit())
        if step is None:
            if not steps:
                return None, None
            step = steps[-1]
        return f"{arm_root}/{step}", step
    else:  # checkpoint-<N>, written by groot at <arm_root>/<run-tag>/checkpoint-<N>
        if not any(d.startswith("checkpoint-") for d in dirs):
            tags = [d for d in dirs if not d.startswith("checkpoint-")]
            if len(tags) == 1:  # descend through the single run-tag dir
                arm_root = f"{arm_root}/{tags[0]}"
                dirs = _ls_dirs(arm_root)
        steps = sorted(
            int(d.split("-")[-1]) for d in dirs if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
        )
        if step is None:
            if not steps:
                return None, None
            step = steps[-1]
        return f"{arm_root}/checkpoint-{step}", step


def submit(
    arm,
    model,
    ckpt_s3,
    step,
    *,
    user,
    image_uri,
    source_dir,
    num_trials,
    video,
    dry,
    queue=QUEUE,
    role=ROLE_ARN,
    results_bucket=DEFAULT_RESULTS_BUCKET,
    instance_type=INSTANCE_TYPE,
    groot_arch="",
    reserved_capacity="1",
    results_subdir="pretrain150k_evals",
    wsm=None,
    priority=MULTI_DAY_PRIORITY,
    heldout=None,
    max_run_seconds=MAX_RUN_SECONDS,
    confirmed=False,
    secrets_manager_arn=None,
):
    validate_launch_contract(
        queue=queue,
        role=role,
        priority=priority,
        max_run_seconds=max_run_seconds,
        secrets_manager_arn=secrets_manager_arn,
    )
    require_submission_confirmation(dry_run=dry, confirmed=confirmed)
    results_s3 = f"s3://{results_bucket}/{user}/wsm_robocasa/{results_subdir}/{arm}/step-{step}/trials-{num_trials}"
    env = {
        "SM_USE_RESERVED_CAPACITY": reserved_capacity,
        "SAGEMAKER_PROGRAM": EVAL_ENTRY,
        "NVIDIA_DRIVER_CAPABILITIES": "all",  # EGL render in container
        "MODEL": model,
        "CKPT_S3": ckpt_s3,
        "STEP": str(step),
        "RESULTS_S3": results_s3,
        "NUM_TRIALS": str(num_trials),
        "EVAL_SPLIT": "target",  # foundation-model protocol
        "TASK_SETS": "atomic_seen,composite_seen,composite_unseen",
        "NUM_WORKERS": "8",
        "VIDEO": video,
        "REPLAN_STEPS": "5",
        "EXEC_STEPS": "8",
        "SEED": "7",
    }
    if heldout:  # held-out-reset protocol (jul_14): reset to every held-out demo x N rollouts
        env.update(
            {"WSM_HELDOUT": "1", "WSM_ROLLOUTS_PER_DEMO": str(heldout["rollouts"]), "WSM_TASKS": heldout["tasks"]}
        )
        if heldout.get("envs_per_gpu", 1) > 1:
            env["WSM_ENVS_PER_GPU"] = str(heldout["envs_per_gpu"])
    if wsm:  # WSM-conditioned eval: entry serves serve_{groot,pi_05}_wsm[_cfg].py (online w_t), not baseline.
        env.update(
            {
                "WSM_REPO_S3": wsm["repo"],
                "ENCODER_CKPT_S3": wsm["encoder"],
                "TASK_LANG_TABLE_S3": wsm["lang_table"],
                "K_WINDOW": str(wsm["k_window"]),
                # BACKBONE selects the WSM serve in the entry (pi -> serve_pi_05_wsm.py JAX server,
                # groot -> serve_groot_wsm[_cfg].py). MODEL routes the base bootstrap; BACKBONE the serve.
                "BACKBONE": wsm["backbone"],
            }
        )
        if wsm.get("cfg"):  # CFG serve branch (serve_groot_wsm_cfg.py): guidance scale, optional task subset
            env["WSM_CFG"] = "1"
            env["GUIDANCE_SCALE"] = str(wsm.get("guidance", 0.0))
            if wsm.get("demo"):
                env["WSM_DEMO_CFG"] = "1"
                env["DEMO_TOKENS_S3"] = wsm["dtok"]
                env["FUSION_CKPT_S3"] = wsm["fusion"]
                env["WSM_CONTROL"] = wsm.get("control", "none")
                env["WSM_DEMO_INDEX"] = str(wsm.get("demo_index", 0))
        else:  # injection serve (serve_*_wsm.py): online-w modulator
            env["WSM_EVAL"] = "1"
            # WSM_TAP_PROMPT: 'expanded' = per-task Qwen prompt to the backbone tap (matches the cache);
            # 'terse' = the env string (A/B arm).
            env["WSM_TAP_PROMPT"] = wsm.get("tap_prompt", "expanded")
        if wsm.get("tasks"):  # small-task POC subset (eval only these)
            env["WSM_TASKS"] = wsm["tasks"]
    if groot_arch:  # 'blackwell' -> entry uses groot_pyproject_blackwell.toml (sdpa, B200-safe)
        env["GROOT_ARCH"] = groot_arch
    name = f"{user.replace('.', '-')}-wsm-eval-{arm.replace('_', '-')}-{step}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"[
        :63
    ].rstrip("-")
    print(
        f"\n=== {arm} ({model}) step {step}\n    ckpt={ckpt_s3}\n    results={results_s3}"
        f"\n    queue={queue}\n    role={role}\n    priority={priority} max_run={max_run_seconds}s"
    )
    if dry:
        print("    [DRY RUN: no source upload and no AWS submission]")
        return

    boto3, sagemaker, Estimator, TrainingQueue = load_aws_sdk()
    validate_caller_account(boto3)

    with prepared_source_bundle(source_dir, EVAL_ENTRY, env, secrets_manager_arn) as (
        safe_source_dir,
        safe_entry,
        safe_env,
    ):
        est = Estimator(
            entry_point=safe_entry,
            source_dir=str(safe_source_dir),
            sagemaker_session=sagemaker.Session(boto_session=boto3.session.Session(region_name=REGION)),
            role=role,
            image_uri=image_uri,
            instance_count=1,
            instance_type=instance_type,
            input_mode="FastFile",
            max_run=max_run_seconds,
            environment=safe_env,
            volume_size=300,
            keep_alive_period_in_seconds=300,
            disable_profiler=True,
            tags=[
                {"Key": "tri.project", "Value": "GROOT-DEXJOCO"},
                {"Key": "tri.owner.email", "Value": f"{user}@tri.global"},
                {"Key": "tri.eval.key", "Value": f"{arm}-{step}"},
            ],
        )
        q = TrainingQueue(queue_name=queue)
        res = q.map(
            est,
            inputs=[None],
            job_names=[name],
            priority=priority,
            share_identifier="default",
            retry_config=RETRY,
            timeout={"attemptDurationSeconds": max_run_seconds},
        )
    print(f"    QUEUED ✓ arn={getattr(res[0], 'job_arn', '?') if res else '?'}")


def _heldout(args):
    """--heldout -> the env spec dict (requires --tasks: the entry builds per-task manifests)."""
    if not getattr(args, "heldout", False):
        return None
    if not args.tasks.strip():
        raise SystemExit("--heldout requires --tasks (the held-out manifests are per task)")
    return {"rollouts": args.rollouts_per_demo, "tasks": args.tasks, "envs_per_gpu": args.envs_per_gpu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--image-user", default=DEFAULT_IMAGE_USER)
    ap.add_argument("--source-dir", default=None)
    ap.add_argument(
        "--ckpt-root", default=None, help="S3 root holding the arm subpaths (default: <user>'s pretrain150k)"
    )
    ap.add_argument("--ckpt-s3", default=None, help="explicit checkpoint dir (single --only arm)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest present)")
    ap.add_argument("--num-trials", type=int, default=50)
    ap.add_argument(
        "--image-uri", default=None, help="full image URI (default: <image-user>'s repo in account 141 ECR)"
    )
    ap.add_argument(
        "--results-bucket", default=DEFAULT_RESULTS_BUCKET, help="canonical account-124 bucket for eval results"
    )
    ap.add_argument(
        "--results-subdir",
        default="pretrain150k_evals",
        help="S3 subdir under <user>/wsm_robocasa/ (e.g. target_ft_evals for post-train evals)",
    )
    ap.add_argument("--instance-type", default=INSTANCE_TYPE, help="e.g. ml.p6-b200.48xlarge for the b200 queue")
    ap.add_argument("--groot-arch", default="", help="'blackwell' for B200 (GR00T drops flash-attn -> sdpa)")
    ap.add_argument("--reserved-capacity", default="1", help="SM_USE_RESERVED_CAPACITY; set 0 for spot queues")
    ap.add_argument("--video", default="first", choices=["none", "first", "all"])
    ap.add_argument("--only", default=",".join(ARMS))
    add_guardrail_arguments(ap)
    # --- WSM-conditioned eval (online w_t via serve_groot_wsm.py) ---
    ap.add_argument("--wsm", action="store_true", help="WSM-conditioned eval of a target_ft_wsm checkpoint")
    ap.add_argument(
        "--backbone",
        default="groot",
        choices=["groot", "pi"],
        help="groot -> serve_groot_wsm.py (HF/torch); pi -> serve_pi_05_wsm.py (openpi/JAX)",
    )
    ap.add_argument("--k-window", type=int, default=2, help="WSM K (selects the target_ft_wsm/<bb>_step<N>_k<K> run)")
    ap.add_argument(
        "--wsm-step", type=int, default=100000, help="WSM encoder step used for w_t (names the run + encoder ckpt)"
    )
    ap.add_argument(
        "--wsm-exp",
        default=None,
        help="encoder run name under wsm_runs/ (default <backbone>_wsm_v1; "
        "use groot_wsm_v2 for the stabilized v2 encoder)",
    )
    ap.add_argument(
        "--tap-prompt",
        default="expanded",
        choices=["expanded", "terse"],
        help="backbone-tap prompt for the WSM encoder inputs: 'expanded' (per-task Qwen, matches "
        "cache) or 'terse' (env string). Tags the results dir so an A/B doesn't collide.",
    )
    # --- CFG-conditioned eval (serve_groot_wsm_cfg.py): eval a specific ckpt at a guidance scale ---
    ap.add_argument("--cfg", action="store_true", help="CFG eval (serve_groot_wsm_cfg.py); needs --ckpt-s3")
    ap.add_argument("--demo", action="store_true", help="demo-CFG eval (doc 15); adds fusion/dtok/control env")
    ap.add_argument("--control", default="none", help="none|fusion_null|wrong_task|shuffle|frozen_phase")
    ap.add_argument("--demo-index", type=int, default=0)
    ap.add_argument("--fusion-s3", default=None)
    ap.add_argument("--demo-tokens-s3", default=None)
    ap.add_argument("--guidance-scale", type=float, default=1.0, help="CFG guidance s (0=learned-uncond~baseline)")
    ap.add_argument("--tasks", default="", help="comma-list task subset for the eval (small-task POC)")
    ap.add_argument("--run-tag", default="curve", help="results-dir tag for --cfg (e.g. poc15)")
    ap.add_argument(
        "--arm-name", default="", help="override the results-dir arm label (e.g. groot_base for a --ckpt-s3 eval)"
    )
    ap.add_argument(
        "--envs-per-gpu",
        type=int,
        default=1,
        help=">1: batched server + K env runners per GPU (validated 4.7x at K=8; doc jul_14/01)",
    )
    ap.add_argument(
        "--heldout",
        action="store_true",
        help="held-out-reset protocol: every held-out demo's exact reset x --rollouts-per-demo",
    )
    ap.add_argument("--rollouts-per-demo", type=int, default=3)
    args = ap.parse_args()

    validate_and_confirm(args)

    user = resolve_user(args.user)
    source_dir = resolve_source_dir(args.source_dir)
    image_uri = (
        args.image_uri or f"{EXECUTION_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{args.image_user}-groot-dexjoco:latest"
    )
    base = f"s3://{DEFAULT_RESULTS_BUCKET}/{user}/wsm_robocasa"

    if args.wsm and args.cfg:  # CFG-conditioned eval of ONE explicit ckpt (serve_{groot,pi_05}_wsm_cfg.py)
        if not args.ckpt_s3:
            raise SystemExit("--cfg eval needs --ckpt-s3 <S3 checkpoint dir>")
        bb = args.backbone  # 'groot' or 'pi'
        model, step = ("pi05" if bb == "pi" else "groot"), (args.step or 0)
        arm = f"{bb}_cfg_{args.run_tag}_s{str(args.guidance_scale).replace('.', 'p')}"  # no '.' (job-name rule)
        wsm_exp = args.wsm_exp or ("groot_wsm_v2" if bb == "groot" else "pi_wsm_v1")
        wsm = {
            "repo": f"{base}/code/wsmv2.tgz",
            "encoder": f"{base}/wsm_runs/{wsm_exp}/wsm_step{args.wsm_step}.pt",
            "lang_table": f"{base}/wsm_policy_feats/{bb}_step{args.wsm_step}/task_lang_table.npz",
            "k_window": args.k_window,
            "backbone": bb,
            "cfg": True,
            "guidance": args.guidance_scale,
            "tasks": args.tasks,
            "demo": args.demo,
            "control": args.control,
            "demo_index": args.demo_index,
            "fusion": args.fusion_s3
            or f"{base}/wsm2_runs/{'pi_100k_matched' if bb == 'pi' else 'orig_65k_matched'}/wsm2_step20000.pt",
            "dtok": args.demo_tokens_s3
            or f"{base}/wsm_demo_tokens/{'pi_100k_matched' if bb == 'pi' else 'orig_65k_matched'}",
        }
        print(
            f"user={user} CFG eval arm={arm} ({model}) step={step}\n  ckpt={args.ckpt_s3}\n  encoder={wsm['encoder']}\n"
            f"  tasks={args.tasks or 'ALL'} s={args.guidance_scale} dry={args.dry_run}"
        )
        submit(
            arm,
            model,
            args.ckpt_s3,
            step,
            user=user,
            image_uri=image_uri,
            source_dir=source_dir,
            num_trials=args.num_trials,
            video=args.video,
            dry=args.dry_run,
            queue=args.queue,
            role=args.role,
            results_bucket=args.results_bucket,
            instance_type=args.instance_type,
            groot_arch=args.groot_arch,
            reserved_capacity=args.reserved_capacity,
            results_subdir="target_ft_wsm_evals",
            wsm=wsm,
            priority=args.priority,
            heldout=_heldout(args),
            max_run_seconds=args.max_run_seconds,
            confirmed=args.confirm_submit,
            secrets_manager_arn=args.secrets_manager_arn,
        )
        return

    if args.wsm:  # WSM-conditioned eval: finetune ckpt under target_ft_wsm + online-w_t server env
        # arm carries the tap-prompt mode so an expanded-vs-terse A/B writes to DISTINCT results dirs.
        arm = f"{args.backbone}_wsm_k{args.k_window}_{args.tap_prompt}"
        ft_root = f"{base}/target_ft_wsm/{args.backbone}_step{args.wsm_step}_k{args.k_window}"
        # ckpt layout differs by backbone: groot writes HF checkpoint-<N>; pi writes orbax bare-number
        # step dirs (submit_finetunes pi WSM -> OUTPUT_S3=target_ft_wsm/pi_step{S}_k{K}; the entry syncs
        # $CKPT_DIR's step dirs flat under it). MODEL routes the eval entry's base bootstrap.
        layout = "step" if args.backbone == "pi" else "checkpoint"
        model = "pi05" if args.backbone == "pi" else "groot"
        ckpt_s3, step = (args.ckpt_s3, args.step or 0) if args.ckpt_s3 else resolve_ckpt(ft_root, layout, args.step)
        if not ckpt_s3:
            raise SystemExit(f"no WSM finetune checkpoint under {ft_root} — has it trained?")
        wsm_exp = args.wsm_exp or f"{args.backbone}_wsm_v1"
        wsm = {
            "repo": f"{base}/code/wsmv2.tgz",
            "encoder": f"{base}/wsm_runs/{wsm_exp}/wsm_step{args.wsm_step}.pt",
            "lang_table": f"{base}/wsm_policy_feats/{args.backbone}_step{args.wsm_step}/task_lang_table.npz",
            "k_window": args.k_window,
            "backbone": args.backbone,
            "tap_prompt": args.tap_prompt,
        }
        print(
            f"user={user} WSM eval arm={arm} ({model}) step={step}\n  ckpt={ckpt_s3}\n  encoder={wsm['encoder']}\n"
            f"  lang_table={wsm['lang_table']} K={args.k_window} dry={args.dry_run}"
        )
        submit(
            arm,
            model,
            ckpt_s3,
            step,
            user=user,
            image_uri=image_uri,
            source_dir=source_dir,
            num_trials=args.num_trials,
            video=args.video,
            dry=args.dry_run,
            queue=args.queue,
            role=args.role,
            results_bucket=args.results_bucket,
            instance_type=args.instance_type,
            groot_arch=args.groot_arch,
            reserved_capacity=args.reserved_capacity,
            results_subdir="target_ft_wsm_evals",
            wsm=wsm,
            priority=args.priority,
            heldout=_heldout(args),
            max_run_seconds=args.max_run_seconds,
            confirmed=args.confirm_submit,
            secrets_manager_arn=args.secrets_manager_arn,
        )
        return

    ckpt_root = args.ckpt_root or f"{base}/pretrain150k"
    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    print(
        f"user={user} split=target trials={args.num_trials}\n  ckpt_root={ckpt_root}\n  arms={keys} dry={args.dry_run}"
    )

    for k in keys:
        if k not in ARMS:
            raise SystemExit(f"unknown arm {k}; valid: {list(ARMS)}")
        a = ARMS[k]
        if args.ckpt_s3 and len(keys) == 1:
            ckpt_s3, step = args.ckpt_s3, (args.step or 0)
        else:
            ckpt_s3, step = resolve_ckpt(f"{ckpt_root}/{a['sub']}", a["layout"], args.step)
        if not ckpt_s3:
            print(f"\n=== {k}: no checkpoint found under {ckpt_root}/{a['sub']} (pretrain not far enough?) — skip")
            continue
        submit(
            args.arm_name or k,
            a["model"],
            ckpt_s3,
            step,
            user=user,
            image_uri=image_uri,
            source_dir=source_dir,
            num_trials=args.num_trials,
            video=args.video,
            dry=args.dry_run,
            queue=args.queue,
            role=args.role,
            results_bucket=args.results_bucket,
            instance_type=args.instance_type,
            groot_arch=args.groot_arch,
            reserved_capacity=args.reserved_capacity,
            results_subdir=args.results_subdir,
            priority=args.priority,
            heldout=_heldout(args),
            max_run_seconds=args.max_run_seconds,
            confirmed=args.confirm_submit,
            secrets_manager_arn=args.secrets_manager_arn,
        )


if __name__ == "__main__":
    main()
