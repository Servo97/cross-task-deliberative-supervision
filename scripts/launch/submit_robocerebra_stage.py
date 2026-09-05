#!/usr/bin/env python3
"""Approval-gated launcher for the RoboCerebra stage job (tap / omega / parity).

Thin by design: queues, priority caps, role, the sanitized source bundle and the SCP tags are all
delegated to `launch_guardrails`. This file only decides what the node is told to do.

Phases (see `robocerebra_stage_entry.sh`):

  tap     994 episodes -> the `wsm_pooled` pooled-token store Stage E consumes. Runs as soon as the
          dataset and checkpoints exist; it does NOT wait for labels, because the tap is an INPUT to
          Stage E, not an output.
  omega   re-export ω from an existing Stage-E checkpoint (the normal path exports ω inside the
          Stage-E job itself via `train_stage_e --export-omega`).
  parity  the D7 expert-replay oracle. GATES R1/R2: the serve-side incremental producer must
          reproduce the shipped ω frame-exactly, and the §25.3 language conventions are measured.

`--dry-run` is fully offline apart from building the source archive: no AWS SDK, no upload, no
submit. Submission additionally requires prior explicit approval and `--confirm-submit`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from launch_guardrails import (  # noqa: E402
    OWNER_EMAIL,
    PROJECT_TAG,
    STUDY_OWNER,
    WSM_ROBOCASA_S3,
    add_guardrail_arguments,
    normalize_queue,
    submit_training_job,
    validate_and_confirm,
)

STUDY = "long_context_v1"
ENTRY = "robocerebra_stage_entry.sh"
DEFAULT_OWNER = STUDY_OWNER
#: Independent of --user on purpose (§22.5 / §24.9 defect 4): --user is the frozen S3 storage
#: prefix every content address in this study is minted under and can never move; the live
#: submitting identity is different, and SCP p-ahpdy5vv denies a submit tagged with the dead one.
DEFAULT_OWNER_EMAIL = OWNER_EMAIL

S3 = WSM_ROBOCASA_S3
STUDY_ROOT = f"{S3}/studies/{STUDY}"

#: The sealed H12 artifacts, content-addressed. Both verified byte-identical to the local trees
#: (§24.1): the LeRobot tarball's sha256 equals its own key, and the 994-episode tree matches the
#: `robocerebra_train_v1` manifest on 2,988/2,988 objects.
RCB_DATA_SHA = "8ce6785b6f57ef3e34d6ca55fd0e3f30be8e19255869886838727635ffc0aa29"
RCB_DATA_S3 = f"{STUDY_ROOT}/robocerebra/data/lerobot/{RCB_DATA_SHA}.tar"
PI05_LIBERO_S3 = f"{STUDY_ROOT}/robocerebra/init/1cfbc327805272daf2d1512faaeaef733edc2ac4b2873f41f471d6896a5d4211.tar"
POOL_SHA = "18c26a7d54d48058302d9dc0fc155a27da66cf35559e5104e954b93390532e30"
POOL_S3 = f"{STUDY_ROOT}/artifacts/workspace/pool/{POOL_SHA}.pt"

#: Queue -> instance family. Each queue's service environment is bound to one family, so the
#: instance type is a function of --queue, never a free parameter.
QUEUE_INSTANCE_TYPES = {
    "fss-tri-cam-robotics-p5-48xlarge-us-west-2": "ml.p5.48xlarge",
    "fss-tri-cam-robotics-p5e-48xlarge-us-west-2-training-plan": "ml.p5e.48xlarge",
}
INSTANCE_TYPE = "ml.p5.48xlarge"
GPUS = 8
RETRY = {"attempts": 1}

#: max_run, sized from the MEASURED tap rate rather than assumed (A6).
#: 114,800 grid frames / 5.97 frames/s/GPU (measured, pad_batch 16, §24.2) / 8 GPUs = 2,385 s.
#: x2.5 headroom = 5,963 s, plus 3,600 s of startup (uv sync + a 12 GB checkpoint download) and
#: rounded up: the 5090 rate is a floor on H100, so this is conservative in the safe direction.
TAP_ESTIMATE_S = 2385
DEFAULT_MAX_RUN = 10_800


def build_environment(args) -> dict:
    env = {
        "RCB_PHASES": args.phases,
        "OPENPI_FORK_S3": args.openpi_source_s3,
        "RCB_DATA_S3": args.rcb_data_s3,
        "RCB_DATA_SHA256": args.rcb_data_sha256,
        "TAP_CKPT_S3": args.tap_ckpt_s3,
        "POOL_CKPT_S3": args.pool_ckpt_s3,
        "POOL_CKPT_SHA256": args.pool_ckpt_sha256,
        "OUTPUT_S3": args.output_s3,
        "NUM_GPUS": str(GPUS),
        "SM_USE_RESERVED_CAPACITY": "0" if "training-plan" in args.queue else "1",
        "SAGEMAKER_PROGRAM": ENTRY,
    }
    if "omega" in args.phases or "parity" in args.phases:
        if not args.encoder_ckpt_uri:
            raise SystemExit("--encoder-ckpt-uri is required for the omega/parity phases")
        env["ENCODER_CKPT_URI"] = args.encoder_ckpt_uri
        env["OMEGA_LANG_MODE"] = args.lang_mode
        env["PARITY_DEMOS"] = str(args.parity_demos)
        if args.stage_e_labels_dir:
            env["STAGE_E_LABELS_DIR"] = args.stage_e_labels_dir
        if args.omega_root_override:
            env["OMEGA_ROOT_OVERRIDE"] = args.omega_root_override
    return env


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_guardrail_arguments(ap, default_max_run_seconds=DEFAULT_MAX_RUN)
    ap.add_argument("--user", default=DEFAULT_OWNER, help="S3 storage owner prefix")
    ap.add_argument(
        "--owner-email",
        default=DEFAULT_OWNER_EMAIL,
        help="value of the required tri.owner.email SCP tag; independent of --user",
    )
    ap.add_argument("--phases", default="tap", help="comma list of tap,omega,parity")
    ap.add_argument("--source-dir", default=None)
    ap.add_argument("--openpi-source-s3", required=True)
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--rcb-data-s3", default=RCB_DATA_S3)
    ap.add_argument("--rcb-data-sha256", default=RCB_DATA_SHA)
    ap.add_argument("--tap-ckpt-s3", default=PI05_LIBERO_S3)
    ap.add_argument("--pool-ckpt-s3", default=POOL_S3)
    ap.add_argument("--pool-ckpt-sha256", default=POOL_SHA)
    ap.add_argument("--output-s3", default=f"{STUDY_ROOT}/robocerebra/stage")
    ap.add_argument(
        "--encoder-ckpt-uri",
        default="",
        help="the FINAL Stage-E checkpoint (encoder.pt) that exported the ω store -- "
        "NOT encoder_best.pt, which is the best-eval step and is a different "
        "model whenever best != final (§41). Parity now refuses a mismatch.",
    )
    ap.add_argument("--stage-e-labels-dir", default="")
    ap.add_argument("--omega-root-override", default="")
    ap.add_argument(
        "--lang-mode",
        default="per_frame",
        choices=("per_frame", "task_line", "demo"),
        help="the §25.3 conditioning contract the ω store and the D7 gate use",
    )
    ap.add_argument("--parity-demos", type=int, default=20)
    ap.add_argument("--instance-type", default="", help="defaults to the family bound to --queue")
    return ap


def main() -> None:
    args = parser().parse_args()
    args.queue = normalize_queue(args.queue)
    inst = args.instance_type or QUEUE_INSTANCE_TYPES.get(args.queue, INSTANCE_TYPE)
    for phase in args.phases.split(","):
        if phase.strip() not in ("tap", "omega", "parity"):
            raise SystemExit(f"unknown phase {phase!r}")
    validate_and_confirm(args)

    env = build_environment(args)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    phase_tag = args.phases.replace(",", "-")
    job_name = f"rcb-stage-{phase_tag}-{stamp}"[:63].rstrip("-")

    print(
        f"phases={args.phases} lang_mode={args.lang_mode}\n"
        f"  wsmv2 = the sanitized source bundle (no separate tarball)\n"
        f"  data={args.rcb_data_s3}\n  tap_ckpt={args.tap_ckpt_s3}\n  pool={args.pool_ckpt_s3}\n"
        f"  output={args.output_s3}\n"
        f"  queue={args.queue} instance={inst} priority={args.priority} "
        f"max_run={args.max_run_seconds}s "
        f"dry={args.dry_run}\n"
        f"  tap estimate {TAP_ESTIMATE_S}s measured x2.5 + startup"
    )
    if args.dry_run:
        print("  [DRY RUN: offline; nothing submitted]")
        print(json.dumps(env, indent=1, sort_keys=True))
        print("  SUBMISSION READY only after explicit approval and --confirm-submit")
        return

    source_dir = args.source_dir or str(pathlib.Path(__file__).resolve().parents[2])
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=env,
        image_uri=args.image_uri,
        instance_type=inst,
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": args.owner_email},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.campaign", "Value": "robocerebra_h14"},
            {"Key": "wsm.phases", "Value": args.phases},
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
    )
    print(json.dumps({"job_name": job_name, "phases": args.phases, "result": str(result)}, indent=1))


if __name__ == "__main__":
    main()
