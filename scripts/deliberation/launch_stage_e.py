#!/usr/bin/env python3
"""H14 Stage-E — approval-gated launcher for the 8-cell encoder funnel as ONE p5 job.

Shape (A4): eight encoder cells, one per GPU, one node, one submission. The cells share only their
staged inputs — no TP, no NCCL — because the funnel's job is to "screen MANY encoders on gates and
graduate FEW to policy training". A single job also matches the standing lean-ops rule (nodes are
scarce; chain everything into one launch) and keeps ONE content-addressed run id over the whole
attribution tree, so no cell can drift from its siblings' corpus or label artifact.

Every guardrail is inherited verbatim from `scripts/launch/launch_guardrails.py`: fixed role,
allowed queues, priority/timeout contract, sanitized source bundle with the sensitive-file
exclusion, `debugger_hook_config=False`, and the sanitized-sha bundle check at submit time. The
node entry stages each input prefix with a ZERO-FILES-FATAL check and S3-syncs every cell's
checkpoint and gates JSON as they land, not at exit.

`--max-run-seconds` is never allowed to default: it is derived from a MEASURED local canary
(`--measured-json`) at 2.5x the extrapolated wall plus startup (amendment A6's rule, applied to
training rather than inference). Priority is 400 on p5; >=600 is refused here regardless.

    # plan only, no AWS calls, no credentials needed
    python scripts/deliberation/launch_stage_e.py --dry-run \
        --labels-s3 s3://.../stage_e_labels/<label_id> \
        --measured-json ~/Research/TRI/wsm_data/deliberation/stage_e_measurements.json \
        --plan-out ~/Research/TRI/wsm_data/deliberation/pE_stage_e_plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "launch"))

from launch_guardrails import (  # noqa: E402
    DEXJOCO_IMAGE_REPO,
    LONG_CONTEXT_STUDY_S3,
    OWNER_EMAIL,
    PROJECT_TAG,
    QUEUE,
    ROLE_ARN,
    WSM_ROBOCASA_S3,
    normalize_queue,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
)

PRIORITY = 400
ALLOWED_PRIORITIES = (100, 400)
ENTRY = "stage_e_entry.sh"
#: Same digest-pinned ECR image the deliberation stages run on (a tag can move, a digest cannot).
IMAGE_SHA = "798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"
IMAGE = f"{DEXJOCO_IMAGE_REPO}@sha256:{IMAGE_SHA}"
INSTANCE_TYPE = "ml.p5.48xlarge"
GPUS_PER_NODE = 8
VOLUME_GB = 500
STUDY_ROOT = f"{LONG_CONTEXT_STUDY_S3}/artifacts/deliberation"

MAX_RUN_HEADROOM = 2.5
HARD_CAP_SECONDS = 24 * 3600  # cam-robotics shared-queue rule for priority < 600

#: The job, in the coordinator's execution order. Exactly GPUS_PER_NODE entries. A cell spec is
#: `name` or `name:seed`.
#:
#: REPACKAGED 2026-08-28 for A14: the funnel's attribution questions are answered locally
#: (deliberative cells 8-16x chance, lambda_del=0 cells 0-1.4x, ctrl-S/ctrl-T below chance), and the
#: ONE question it could not answer is whether Qwen positives beat embedding positives — ctrl-E's
#: lift sat inside E1's own same-config seed spread. So the node now spends its eight GPUs on the
#: pre-registered SEED REPLICATION of that contrast on v2 labels, not on re-running settled cells:
#:
#:    E1b x 3 seeds        Qwen positives, binding-aware hard negatives      (primary arm)
#:    ctrl-Eb x 3 seeds    embedding positives, THE SAME hard negatives      (primary control)
#:    E1b-analog05 x 2     ANALOGOUS positives at 0.5                        (secondary)
#:
#: `ctrl-1D` is dropped despite this venue being the only one that can stage all three taps: with
#: n=1 seed it would be read against a spread it cannot resolve, which is exactly the mistake the
#: replication exists to stop repeating. It returns when the other two taps exist AND it can be run
#: paired.
CELLS = (
    "E1b:20260828",
    "ctrl-Eb:20260828",
    "E1b-analog05:20260828",
    "E1b:20260829",
    "ctrl-Eb:20260829",
    "E1b-analog05:20260829",
    "E1b:20260830",
    "ctrl-Eb:20260830",
)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def code_shas() -> dict:
    def sha(relative: str) -> str:
        return hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()[:16]

    return {
        "trainer_sha": sha("workspace_models/train/train_wsm_base/train_stage_e.py"),
        "objectives_sha": sha("workspace_models/networks/omega_objectives.py"),
        "encoder_sha": sha("workspace_models/networks/stage_e_encoder.py"),
        "label_builder_sha": sha("scripts/deliberation/build_edge_labels.py"),
        "entry_sha": sha(ENTRY),
        "git_head": git_head(),
    }


def derive_max_run_seconds(args, n_cells: int) -> tuple[int, dict]:
    """A6 discipline: sized from a MEASURED canary, never assumed, never 86400-by-default.

    The cells run CONCURRENTLY (one per GPU), so the node's wall is the slowest cell, not their
    sum — but the measured rate comes from a single 5090, so the extrapolation is deliberately
    NOT discounted for the H100's speed. Being early is free; being cut off is not.
    """
    if args.max_run_seconds:
        if args.max_run_seconds > HARD_CAP_SECONDS:
            raise SystemExit(
                f"--max-run-seconds {args.max_run_seconds} > {HARD_CAP_SECONDS}; "
                "a longer job needs priority >=600, which needs explicit approval"
            )
        return args.max_run_seconds, {"source": "explicit flag"}
    if not args.measured_json:
        raise SystemExit(
            "refusing to guess a timeout: pass --measured-json (the local canary "
            "measurement) or an explicit --max-run-seconds"
        )
    m = json.loads(Path(args.measured_json).expanduser().read_text())
    rate = float(m["seconds_per_step"])
    fixed = float(m.get("fixed_overhead_seconds", 0.0))
    per_cell = rate * args.steps * (args.batch_episodes / float(m["batch_episodes"])) + fixed
    total = int(per_cell * MAX_RUN_HEADROOM) + args.startup_seconds
    if total > HARD_CAP_SECONDS:
        raise SystemExit(
            f"measured estimate {per_cell / 3600:.2f} h x{MAX_RUN_HEADROOM} = {total / 3600:.2f} h "
            f"exceeds the {HARD_CAP_SECONDS / 3600:.0f} h cap for priority {PRIORITY}. "
            "Shorten --steps or split the funnel; do NOT raise the priority silently."
        )
    return total, {
        "source": "measured local canary",
        "measured_seconds_per_step": rate,
        "measured_batch_episodes": m["batch_episodes"],
        "fixed_overhead_seconds": fixed,
        "per_cell_estimate_s": round(per_cell),
        "concurrent_cells": n_cells,
        "headroom": MAX_RUN_HEADROOM,
        "startup_s": args.startup_seconds,
        "note": "cells run concurrently, one per GPU; wall = slowest cell, not their sum",
    }


def build_plan(args) -> dict:
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    # A18: the entry runs cells in TWO WAVES -- everything without a `-4tap` suffix first, then the
    # 4-tap cells -- so the limit is per WAVE, not per job. Checking the total would reject the
    # pre-registered 10-cell run even though neither wave exceeds 8.
    _w2 = [c for c in cells if c.split(":")[0].endswith("-4tap")]
    _w1 = [c for c in cells if not c.split(":")[0].endswith("-4tap")]
    for _label, _w in (("wave1 (3-tap)", _w1), ("wave2 (4-tap)", _w2)):
        if len(_w) > GPUS_PER_NODE:
            raise SystemExit(f"{_label}: {len(_w)} cells > {GPUS_PER_NODE} GPUs on one node")
    if _w2 and not args.tap4_s3:
        raise SystemExit("-4tap cells listed but no --tap4-s3 given; wave 2 would have no taps")
    for spec in cells:  # `name` or `name:<int seed>`; the node splits the same way
        name, sep, seed = spec.partition(":")
        if not name or (sep and not seed.isdigit()):
            raise SystemExit(f"bad cell spec {spec!r}; expected `cell` or `cell:seed`")
    if len(set(cells)) != len(cells):
        raise SystemExit(
            f"duplicate cell specs in {cells}: a repeated cell needs a distinct seed, "
            "or two GPUs compute the identical encoder_id and race the same run dir"
        )
    taps = {}
    for entry in args.tap_s3:
        name, _, uri = entry.partition("=")
        if not uri.startswith("s3://"):
            raise SystemExit(f"--tap-s3 {entry!r} must be domain=s3://...")
        taps[name] = uri
    if not taps:
        raise SystemExit("at least one --tap-s3 domain=s3://... is required")

    shas = code_shas()
    key = {
        "cells": cells,
        "labels_s3": args.labels_s3,
        "taps": taps,
        "steps": args.steps,
        "batch_episodes": args.batch_episodes,
        "min_edges": args.min_edges,
        "warmup": args.warmup,
        "lr": args.lr,
        "lambda_sigreg": args.lambda_sigreg,
        "sigreg_rank_cap": args.sigreg_rank_cap,
        "lambda_xdom": args.lambda_xdom,
        "contrast_weight": args.contrast_weight,
        **shas,
    }
    run_id = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    max_run, why = derive_max_run_seconds(args, len(cells))

    out_prefix = f"{STUDY_ROOT}/stage_e/{run_id}"
    env = {
        "WSM_E_RUN_ID": run_id,
        "WSM_E_S3_OUT": out_prefix,
        "WSM_E_LABELS_S3": args.labels_s3,
        "WSM_E_TAPS": ";".join(f"{k}={v}" for k, v in sorted(taps.items())),
        "WSM_E_CELLS": ",".join(cells),
        "WSM_E_STEPS": str(args.steps),
        "WSM_E_BATCH_EPISODES": str(args.batch_episodes),
        "WSM_E_MIN_EDGES": str(args.min_edges),
        "WSM_E_WARMUP": str(args.warmup),
        "WSM_E_EVAL_EVERY": str(args.eval_every),
        "WSM_E_LR": str(args.lr),
        "WSM_E_LAMBDA_SIGREG": str(args.lambda_sigreg),
        "WSM_E_SIGREG_RANK_CAP": str(args.sigreg_rank_cap),
        "WSM_E_LAMBDA_XDOM": str(args.lambda_xdom),
        "WSM_E_CONTRAST_WEIGHT": str(args.contrast_weight),
        "WSM_E_LABEL_STORE_S3": args.keyframe_labels_s3,
        # §27 serve-consistent conditioning. 'serve' = per-domain: task_mean for
        # robocasa/remembench/robomme, per_frame for robocerebra.
        "WSM_E_LANG_MODE": args.lang_mode,
        "WSM_E_RAW_TAP_ERANK_S3": args.raw_tap_erank_s3,
        # A18: tap set for the -4tap cells, which run as a SECOND WAVE after the pre-registered
        # eight so their one-cell-per-GPU profile is untouched.
        "WSM_E_TAPS_4TAP": ";".join(
            f"{k}={v}" for k, v in sorted((e.split("=", 1) for e in args.tap4_s3), key=lambda kv: kv[0])
        ),
        "WSM_E_TASK_LANG_TABLES": ";".join(args.task_lang_table_s3 or []),
        "WSM_E_EXPORT_OMEGA": "1" if args.export_omega else "",
        "WSM_E_GIT_HEAD": shas["git_head"],
    }
    queue = normalize_queue(args.queue)
    if queue == QUEUE and args.priority not in ALLOWED_PRIORITIES:
        raise SystemExit(
            f"priority {args.priority} not in {ALLOWED_PRIORITIES} for {QUEUE}; "
            "anything above 400 needs explicit user say-so"
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage_e_funnel",
        "run_id": run_id,
        "cells": cells,
        "gpus_per_node": GPUS_PER_NODE,
        "queue": queue,
        "role": args.role,
        "priority": args.priority,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "volume_gb": VOLUME_GB,
        "image": IMAGE,
        "entry": ENTRY,
        "max_run_seconds": max_run,
        "max_run_rationale": why,
        "sagemaker_timeout_seconds": max_run,
        "batch_timeout_seconds": max_run,
        "debugger_hook_config": False,
        "disable_profiler": True,
        "s3_out": out_prefix,
        "staged_inputs": {"labels": args.labels_s3, "taps": taps, "keyframe_labels": args.keyframe_labels_s3},
        "code_shas": shas,
        "environment": env,
        "resume": {
            "mode": "content-addressed re-run",
            "sync": "per-cell s3 sync of *.json + encoder{,_best}.pt every 120 s, plus a final sync",
            "zero_files_fatal": "every staged prefix must land >=1 file or the entry exits 3",
        },
        "preconditions": [
            "aws sts get-caller-identity succeeds AND the identity is permitted to submit "
            "(a fresh SSO login on 2026-08-28 still returned SCP deny p-ahpdy5vv)",
            "every --tap-s3 prefix exists and is non-empty (the node fails closed if not). "
            "The pooled RoboCasa tap has only ever been verified LOCALLY "
            "(~/Research/TRI/wsm_data/wsm_pooled/pi_100k, 221 MB for the 13 A10 tasks); "
            "upload it before or with the submit: "
            "aws s3 sync ~/Research/TRI/wsm_data/wsm_pooled/pi_100k <tap-s3>",
            "the edge-label artifact at --labels-s3 is uploaded and is the one the local canary "
            "consumed (label_id must match)",
            "cells naming a domain whose tap is not staged will train on the domains that ARE "
            "staged: ctrl-1D is only a real control when >=2 taps are present",
            "no coordinator HOLD",
        ],
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=QUEUE, type=normalize_queue)
    ap.add_argument("--role", default=ROLE_ARN)
    ap.add_argument("--priority", type=int, default=PRIORITY)
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument(
        "--lang-mode",
        default="serve",
        help="conditioning contract (§27). 'serve' = the serve-consistent per-domain "
        "default; also accepts one mode for all, or 'dom=mode,...'. The sealed "
        "'episode_mean' is NOT serveable and is opt-in only.",
    )
    ap.add_argument(
        "--tap4-s3",
        action="append",
        default=[],
        help="domain=s3://... tap for the -4tap second-wave cells; repeat once per "
        "domain (normally the three 3-tap roots PLUS robomme)",
    )
    ap.add_argument(
        "--raw-tap-erank-s3",
        default="",
        help="s3://...json of {domain: raw-tap effective rank}; sets the per-domain "
        "G1b bar (0.8x) for THIS run without touching the sealed module defaults",
    )
    ap.add_argument(
        "--task-lang-table-s3",
        action="append",
        default=[],
        help="domain=s3://...npz, staged on the node and used VERBATIM for that domain's task_mean. Repeatable.",
    )
    ap.add_argument("--labels-s3", required=True, help="s3 prefix of the content-addressed edge-label artifact")
    ap.add_argument("--tap-s3", action="append", default=[], help="domain=s3://... pooled tap store, repeatable")
    ap.add_argument(
        "--keyframe-labels-s3",
        default=f"{WSM_ROBOCASA_S3}/wsm_labels",
        help="RoboCasa pi-geometry keyframe labels for the decode-grounding gate",
    )
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-episodes", type=int, default=48)
    ap.add_argument("--min-edges", type=int, default=36)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-sigreg", type=float, default=0.05)
    ap.add_argument("--sigreg-rank-cap", type=float, default=15.0)
    ap.add_argument("--lambda-xdom", type=float, default=0.5)
    ap.add_argument("--contrast-weight", type=float, default=2.0)
    ap.add_argument("--export-omega", action="store_true", default=True)
    ap.add_argument("--measured-json", default="")
    ap.add_argument("--max-run-seconds", type=int, default=0)
    ap.add_argument("--startup-seconds", type=int, default=1800)
    ap.add_argument("--plan-out", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-submit", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
    plan = build_plan(args)
    # Batch REJECTS a job on a reserved-capacity queue that does not carry this variable at all
    # ("Missing SM_USE_RESERVED_CAPACITY environment variable for reserved capacity queue"), and
    # the guardrail only validates the VALUE when one is present -- so omitting it passed every
    # local check and failed instantly at submit. Set it on both branches, as the reference
    # launchers do (submit_pi_stage_s.py:1147, submit_robocerebra.py:380).
    plan["environment"]["SM_USE_RESERVED_CAPACITY"] = "0" if "training-plan" in plan["queue"] else "1"
    if args.plan_out:
        path = Path(args.plan_out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=1))
        print(f"[plan] wrote {path}")
    if args.dry_run or not args.confirm_submit:
        print(json.dumps(plan, indent=1))
        print("\nDRY RUN — nothing submitted. Submission requires:")
        print("  1. live SSO creds AND a role permitted to submit (the current identity is denied by SCP p-ahpdy5vv)")
        print("  2. every staged S3 prefix present and non-empty")
        print("  3. explicit user approval, then --confirm-submit")
        return

    with prepared_source_bundle(str(REPO_ROOT), ENTRY, plan["environment"], None) as (staged, _e, _v):
        sanitized_sha = source_tree_sha256(staged)
    result = submit_training_job(
        entry=ENTRY,
        source_dir=str(REPO_ROOT),
        environment=plan["environment"],
        image_uri=plan["image"],
        instance_type=plan["instance_type"],
        volume_size=plan["volume_gb"],
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "study", "Value": "h14-deliberation"},
            {"Key": "stage", "Value": "stage_e_funnel"},
            {"Key": "run_id", "Value": plan["run_id"]},
        ],
        retry_config=None,
        job_name=f"h14-stage-e-{plan['run_id']}-{int(time.time())}",
        queue=plan["queue"],
        role=plan["role"],
        priority=plan["priority"],
        max_run_seconds=plan["max_run_seconds"],
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
        expected_source_tree_sha256=sanitized_sha,
    )
    print(json.dumps({"submitted": True, "plan": plan, "result": str(result)}, indent=1))


if __name__ == "__main__":
    main()
