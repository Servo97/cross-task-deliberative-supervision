#!/usr/bin/env python3
"""H14 P1 — approval-gated launcher for the deliberation loop on p5/p5e.

SHAPE IS AMENDMENT A7, NOT PLAN §7. The original plan chained pass1 -> embed -> pass2 into ONE 24 h
job. The panel replaced that with **three separately submittable, separately resumable stages**,
because (a) this role cannot terminate Batch jobs -- timeouts are the only kill switch -- and (b)
p5 has shown 3-day RUNNABLE waits, so a single long job is a single long hostage. Each stage:

  stage 1  pass1   descriptors   N shards x 1 GPU, one vLLM replica per GPU
  stage 2  embed   text embeddings over the merged descriptor store (1 GPU, minutes)
  stage 3  pass2   bucketed deliberation, N shards

Every stage S3-syncs each COMPLETED SHARD FILE as it lands (not at exit), and every stage resumes
structurally: on restart it re-reads what is already in S3 and re-validates it by PARSING, never by
existence (`caption_segments.validate_existing` / `pass2_deliberate.validate_bucket_file`). A
preempted or timed-out stage therefore costs only its in-flight shard.

Guardrails inherited verbatim from `scripts/launch/launch_guardrails.py`: fixed role, allowed
queues, priority/timeout contract, sanitized source bundle, sensitive-file exclusion,
`debugger_hook_config=False`. Priority is **100** (the established low/sweep class for
batch-inference-shaped work on p5, per `robomme_integration/eval/launch_p5_preflight.py`).

`--max-run-seconds` is NOT allowed to default to 86400 (A6): it must be derived from a MEASURED
pilot throughput via `--measured-json`, at 2.5x the measured estimate. Submitting without a
measurement is refused.

    # dry run (no AWS calls, no credentials needed)
    python scripts/deliberation/launch_deliberation.py --stage pass1 --dry-run \
        --measured-json ~/Research/TRI/wsm_data/deliberation/pilot_measurements.json

    # submission (requires live SSO + explicit user approval)
    python scripts/deliberation/launch_deliberation.py --stage pass1 --confirm-submit ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    TRAINING_PLAN_QUEUE,
    WSM_ROBOCASA_S3,
    normalize_queue,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
)

from scripts.deliberation import pass2_deliberate as P2D  # noqa: E402
from scripts.deliberation import pass2_prompt as P2  # noqa: E402
from workspace_models.labels import caption_segments as CS  # noqa: E402

# Priority 400 = the DEFAULT class, chosen by the coordinator 2026-08-22 after a live queue probe
# (p5: 4 RUNNABLE across 8 nodes; p5e: 4 across 2) and because recent history shows priority-100 work
# starving. 100 remains legal for a genuine background sweep. >=600 still needs explicit user say-so
# and is refused here regardless (MEMORY: SageMaker priority classes).
PRIORITY = 400
ALLOWED_PRIORITIES = (100, 400)
MAX_PRIORITY_WITHOUT_APPROVAL = 400
ENTRY = "deliberation_entry.sh"
# Our own ECR image, digest-pinned, and the one robomme_integration already runs on p5 (verified
# present in ECR, tag `latest`, pushed 2026-07-21). Deliberately NOT a public DLC tag: a tag can move,
# a digest cannot, and this one has a track record on this queue.
IMAGE_SHA = "798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"
IMAGE = f"{DEXJOCO_IMAGE_REPO}@sha256:{IMAGE_SHA}"
INSTANCE_TYPE = "ml.p5.48xlarge"
GPUS_PER_NODE = 8
VOLUME_GB = 500
STUDY_ROOT = f"{LONG_CONTEXT_STUDY_S3}/artifacts/deliberation"

STAGES = ("pass1", "embed", "pass2")

# A6: never 86400-by-default. Headroom multiplier over the MEASURED estimate.
MAX_RUN_HEADROOM = 2.5
HARD_CAP_SECONDS = 24 * 3600  # cam-robotics shared-queue rule for priority < 600
# Coordinator rule (2026-08-22): size pass 2 from stage-1's MEASURED on-node FP8/H100 rate, not from
# the local NVFP4 extrapolation. If the re-derived SINGLE-NODE wall exceeds this, split into two
# parallel p5 jobs @400 over disjoint shard ranges; at or under it, one job.
SPLIT_THRESHOLD_SECONDS = 20 * 3600


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def derived_shas() -> dict:
    """Everything that identifies WHAT this job computes. Any change re-addresses the outputs."""
    return {
        "pass1_prompt_sha": CS.prompt_sha("descriptor"),
        "pass1_schema_sha": CS.schema_sha("descriptor"),
        "pass1_max_tokens": CS.DESCRIPTOR_MAX_TOKENS,
        "pass2_prompt_sha": P2.prompt_sha(),
        "pass2_schema_sha": P2.schema_sha(),
        "pass2_quotas": P2D.QUOTAS,
        "pass2_k_per_bucket": P2D.K_PER_BUCKET,
        "pass2_quota_floors": P2D.QUOTA_FLOORS,
        "mining_seed": P2D.MINING_SEED,
        "caption_segments_code_sha": CS.code_sha(),
        "edge_schema_md_sha": hashlib.sha256(
            (REPO_ROOT / "scripts" / "deliberation" / "edge_schema.md").read_bytes()
        ).hexdigest()[:16],
        "git_head": git_head(),
    }


def run_id(args, shas: dict) -> str:
    # The shard RANGE is deliberately absent: the two jobs of a split are one logical stage and
    # must land in ONE content-addressed store, so their outputs merge and resume across each other.
    # `domain` and the RoboCerebra input sha are part of WHAT the job computes: without them a
    # robocerebra pass1 and a robocasa pass1 with the same corpus string collide on one run_id and
    # therefore one S3 prefix. Absent for every non-pass1 stage so the frozen embed/pass2 ids that
    # already exist are not re-addressed by this fix.
    key = {
        "stage": args.stage,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "corpus": args.corpus,
        "num_shards": args.num_shards,
        **shas,
    }
    if args.stage == "pass1":
        key["domain"] = args.domain
        if args.domain == "robocerebra":
            key["rcb_data_sha256"] = args.rcb_data_sha256
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def derive_max_run_seconds(args) -> tuple[int, dict]:
    """A6: sized from MEASURED pilot throughput, never assumed.

    pass1: measured seconds/segment/GPU from the pilot's SUMMARY line.
    pass2: measured seconds/anchor/GPU from the 200-anchor pilot.
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
            "refusing to guess a timeout: pass --measured-json (the pilot's measurements) or an "
            "explicit --max-run-seconds. A6 forbids defaulting to 86400."
        )
    m = json.loads(Path(args.measured_json).expanduser().read_text())
    gpus = args.num_shards

    def rate_of(stage_key: str, field: str) -> float:
        v = (m.get(stage_key) or {}).get(field)
        if v is None:
            raise SystemExit(
                f"{args.measured_json}: {stage_key}.{field} is null -- that stage has not been "
                f"piloted yet. Measure it first (A6); do not submit against a guessed rate."
            )
        return float(v)

    if args.stage == "pass1":
        est_s = (
            float(m["corpus"]["total_segments"])
            / max(rate_of("pass1", "segments_per_min_per_gpu"), 1e-9)
            / gpus
            * 60.0
        )
    elif args.stage == "pass2":
        # only THIS job's slice of the anchor space (see --shard-offset / --shard-count)
        frac = args.shard_count / max(args.num_shards, 1)
        est_s = (
            float(m["corpus"]["total_anchors"])
            * frac
            / max(rate_of("pass2", "anchors_per_min_per_gpu"), 1e-9)
            / args.shard_count
            * 60.0
        )
    else:
        est_s = float(m.get("embed", {}).get("estimated_seconds", 1800))
    total = int(est_s * MAX_RUN_HEADROOM) + args.startup_seconds
    if total > HARD_CAP_SECONDS:
        raise SystemExit(
            f"measured estimate {est_s / 3600:.1f} h x{MAX_RUN_HEADROOM} = {total / 3600:.1f} h "
            f"exceeds the {HARD_CAP_SECONDS / 3600:.0f} h cap for priority {PRIORITY}. "
            "Split the stage across more shards or more nodes; do NOT raise the priority silently."
        )
    return total, {
        "source": "measured",
        "measured_estimate_s": round(est_s),
        "headroom": MAX_RUN_HEADROOM,
        "startup_s": args.startup_seconds,
        "shards": gpus,
    }


def build_environment(args, shas: dict, rid: str) -> dict:
    """Node env. Every value is a string (SageMaker requirement) and none is a secret."""
    out_prefix = f"{STUDY_ROOT}/{args.stage}/{rid}"
    env = {
        "WSM_DELIB_STAGE": args.stage,
        "WSM_DELIB_RUN_ID": rid,
        "WSM_DELIB_S3_OUT": out_prefix,
        "WSM_DELIB_S3_SYNC_EVERY_SHARD": "1",  # A7: sync each completed file, not at exit
        "WSM_DELIB_NUM_SHARDS": str(args.num_shards),
        "WSM_DELIB_SHARD_OFFSET": str(args.shard_offset),
        "WSM_DELIB_SHARD_COUNT": str(args.shard_count),
        "WSM_DELIB_GPUS_PER_NODE": str(GPUS_PER_NODE),
        "WSM_DELIB_MODEL": args.model,
        "WSM_DELIB_REASONING_EFFORT": args.reasoning_effort,
        "WSM_DELIB_CORPUS": args.corpus,
        "WSM_DELIB_MAX_TOKENS": str(
            (args.pass1_max_tokens or CS.DESCRIPTOR_MAX_TOKENS) if args.stage == "pass1" else args.pass2_max_tokens
        ),
        # pilot-only: cap the work so an effort A/B costs minutes, not a corpus
        "WSM_DELIB_LIMIT_SEGMENTS": str(args.limit_segments or ""),
        "WSM_DELIB_CONCURRENCY": str(args.concurrency),
        "WSM_DELIB_STRUCTURED_OUTPUT": "1",
        "WSM_DELIB_PROMPT_SHA": shas["pass1_prompt_sha"] if args.stage == "pass1" else shas["pass2_prompt_sha"],
        "WSM_DELIB_SCHEMA_SHA": shas["pass1_schema_sha"] if args.stage == "pass1" else shas["pass2_schema_sha"],
        # cross-stage structural resume: pass2 reads pass1's store from S3 and re-validates it
        "WSM_DELIB_PASS1_S3_IN": args.pass1_s3_in or f"{STUDY_ROOT}/pass1",
        "WSM_DELIB_EMBED_S3_IN": args.embed_s3_in or f"{STUDY_ROOT}/embed",
        "WSM_DELIB_EMBED_MODEL": args.embed_model,
        # embed: extra pass-1 stores merged into the same domain-nested tree (a newly added
        # domain lands under its own run_id prefix, not inside the frozen store)
        "WSM_DELIB_PASS1_EXTRA_S3_IN": args.pass1_extra_s3_in,
        # embed/mine: restrict ANCHORS to these domains (delta mining); candidates stay the corpus
        "WSM_DELIB_ANCHOR_DOMAINS": args.anchor_domains,
        # Post-stage corpus assertion (§38): the job may not report success unless the store holds
        # the episodes/segments the corpus inventory says it should.
        "WSM_DELIB_EXPECT_ANCHORS": str(args.expect_anchors or ""),
        "WSM_DELIB_EXPECT_DOMAIN_ANCHORS": args.expect_domain_anchors,
        "WSM_DELIB_EXPECT_EPISODES": str(args.expect_episodes or ""),
        "WSM_DELIB_EXPECT_SEGMENTS": str(args.expect_segments or ""),
        "WSM_DELIB_VERIFY_RESUME": "1",
        "WSM_DELIB_GIT_HEAD": shas["git_head"],
        "WSM_TASKS": args.tasks,
        # inputs the node stages; a sync landing ZERO files is fatal on the node
        "WSM_DELIB_DATASET_S3": args.dataset_s3,
        "WSM_DELIB_LABELS_S3": args.labels_s3,
        "WSM_DELIB_CAPTIONS_S3": args.captions_s3,
        # RoboCerebra pass 1: ONE content-addressed tarball of the sealed `robocerebra_train_v1`
        # LeRobot tree (the object the H12 arms trained from). No keyframe labels and no caption
        # hints exist for it, and none are needed -- segmentation is the official `subtask_index`
        # column and the hint is the per-segment subtask string, both inside the dataset.
        "WSM_DELIB_DOMAIN": args.domain,
        "WSM_DELIB_RCB_DATA_S3": args.rcb_data_s3,
        "WSM_DELIB_RCB_DATA_SHA256": args.rcb_data_sha256,
        "WSM_MAX_MODEL_LEN": str(args.max_model_len),
        # 2026-09-01: the FIRST real node run of this entry died with the EngineCore child exiting
        # during init ("Failed core proc(s): {}" = a sentinel fired, i.e. the child exited). The one
        # UNFORCED difference from the only configuration that has ever served this model is here:
        # every local run of the 19,636-segment corpus used enforce_eager=1, and "p5 needs no
        # --enforce-eager" was an assumption written before any node had run. Qwen3.8-27B is a
        # HYBRID model (48 linear/GDN + 16 full-attention layers, head_dim 256), and CUDA-graph
        # capture over a hybrid stack is exactly the fragile part of startup that eager mode skips.
        # Revert to the proven setting; re-enable as an OPTIMISATION once a node log proves it safe.
        "WSM_ENFORCE_EAGER": "1",
        # §34: the FP8 block-scale GEMM picks a FlashInfer sm90-only cubin kernel on every Hopper
        # node and dies loading it. 0 short-circuits the gate; recorded in the manifest so the
        # kernel choice is part of the run's provenance rather than a node-local accident.
        "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0",
        "VLLM_USE_DEEP_GEMM": "0",
    }
    return env


def build_plan(args) -> dict:
    if not args.shard_count:
        args.shard_count = args.num_shards
    if args.shard_offset < 0 or args.shard_offset + args.shard_count > args.num_shards:
        raise SystemExit(
            f"shard range [{args.shard_offset}, {args.shard_offset + args.shard_count}) escapes "
            f"--num-shards {args.num_shards}; the split must PARTITION the anchor space, and an "
            "overlap would have two jobs judging the same anchors under one edge_store_id"
        )
    shas = derived_shas()
    rid = run_id(args, shas)
    max_run, max_run_why = derive_max_run_seconds(args)
    env = build_environment(args, shas, rid)
    queue = normalize_queue(args.queue)
    if queue not in (QUEUE, TRAINING_PLAN_QUEUE):
        raise SystemExit(f"queue {queue!r} is not an allowed cam-robotics queue")
    if queue == QUEUE and args.priority not in ALLOWED_PRIORITIES:
        raise SystemExit(
            f"priority {args.priority} is not in {ALLOWED_PRIORITIES} for {QUEUE}. "
            f"Anything above {MAX_PRIORITY_WITHOUT_APPROVAL} needs explicit user say-so."
        )
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "run_id": rid,
        "queue": queue,
        "role": args.role,
        "priority": args.priority,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "gpus_per_node": GPUS_PER_NODE,
        "num_shards": args.num_shards,
        "shard_range": [args.shard_offset, args.shard_offset + args.shard_count],
        "volume_gb": VOLUME_GB,
        "image": IMAGE,
        "entry": ENTRY,
        "max_run_seconds": max_run,
        "max_run_rationale": max_run_why,
        "sagemaker_timeout_seconds": max_run,  # both timeouts, same value
        "batch_timeout_seconds": max_run,
        "debugger_hook_config": False,  # guardrails sets this; recorded for the manifest
        "disable_profiler": True,
        "s3_out": env["WSM_DELIB_S3_OUT"],
        "derived_shas": shas,
        "environment": env,
        "resume": {
            "mode": "structural",
            "pass1_gate": "caption_segments.validate_existing_descriptors (re-parse + shape check)",
            "pass2_gate": "pass2_deliberate.validate_bucket_file "
            "(candidates match, len(verdicts)==len(candidates), "
            "finish_reason!='length', every verdict schema-valid)",
            "sync": "per completed shard file, s3 cp on write",
        },
    }
    return plan


def plan_pass2_layout(args) -> dict:
    """Apply the coordinator's sizing rule and PRINT the exact commands to run.

    Rule: derive the single-node wall from the MEASURED pass-2 rate. If it exceeds
    SPLIT_THRESHOLD_SECONDS, split into two parallel p5 jobs @400 over disjoint shard ranges;
    otherwise one job. The two halves share a run_id by construction, so they write one edge store
    and each half's structural resume also covers anything the other half already finished.
    """
    m = json.loads(Path(args.measured_json).expanduser().read_text())
    rate = (m.get("pass2") or {}).get("anchors_per_min_per_gpu")
    if rate is None:
        raise SystemExit(
            "pass2.anchors_per_min_per_gpu is null -- size this from stage-1's "
            "ON-NODE FP8/H100 measurement, not from the local NVFP4 extrapolation."
        )
    anchors = float(m["corpus"]["total_anchors"])
    single = anchors / float(rate) / GPUS_PER_NODE * 60.0
    # TWO independent constraints, and the earlier version conflated them:
    #  (a) the coordinator's rule -- a single-node wall over SPLIT_THRESHOLD_SECONDS gets split;
    #  (b) the HARD 24 h queue cap, which applies to max_run = wall*2.5 + startup, so the real
    #      per-job wall ceiling is (86400 - startup) / 2.5 -- about 9.4 h, well under 20 h.
    # Taking only (a) produced "1 job -- fits one node" alongside a 36.1 h max_run. Take the max.
    wall_ceiling = (HARD_CAP_SECONDS - args.startup_seconds) / MAX_RUN_HEADROOM
    n_by_rule = 2 if single > SPLIT_THRESHOLD_SECONDS else 1
    n_by_cap = math.ceil(single / wall_ceiling) if wall_ceiling > 0 else 99
    n_jobs = max(n_by_rule, n_by_cap, 1)
    per_job = single / n_jobs
    total = int(per_job * MAX_RUN_HEADROOM) + args.startup_seconds
    base = f"python scripts/deliberation/launch_deliberation.py --stage pass2 --measured-json {args.measured_json}"
    if n_jobs == 1:
        cmds = [f"{base} --num-shards {GPUS_PER_NODE} --confirm-submit"]
    else:
        total_shards = GPUS_PER_NODE * n_jobs
        cmds = [
            f"{base} --num-shards {total_shards} --shard-offset {i * GPUS_PER_NODE} "
            f"--shard-count {GPUS_PER_NODE} --confirm-submit"
            for i in range(n_jobs)
        ]
    out = {
        "measured_anchors_per_min_per_gpu": rate,
        "total_anchors": anchors,
        "single_node_wall_h": round(single / 3600, 2),
        "split_threshold_h": SPLIT_THRESHOLD_SECONDS / 3600,
        "decision": f"{n_jobs} job(s)",
        "n_jobs_by_20h_rule": n_by_rule,
        "n_jobs_by_24h_cap": n_by_cap,
        "per_job_wall_ceiling_h": round(wall_ceiling / 3600, 2),
        "per_job_wall_h": round(per_job / 3600, 2),
        "per_job_max_run_seconds": total,
        "fits_24h_cap": total <= HARD_CAP_SECONDS,
        "commands": cmds,
    }
    print(json.dumps(out, indent=1))
    if not out["fits_24h_cap"]:
        print(
            "\nWARNING: even split, a job exceeds the 24 h cap. Raise --num-shards further "
            "(more nodes); do NOT raise the priority."
        )
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--queue", default=QUEUE, type=normalize_queue)
    ap.add_argument("--role", default=ROLE_ARN)
    ap.add_argument("--priority", type=int, default=PRIORITY)
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--instance-count", type=int, default=1)
    ap.add_argument(
        "--num-shards", type=int, default=GPUS_PER_NODE, help="GLOBAL shard count across every job in the stage"
    )
    ap.add_argument(
        "--shard-offset",
        type=int,
        default=0,
        help="first global shard index this job owns (2-job split: 0 and --shard-count)",
    )
    ap.add_argument(
        "--shard-count",
        type=int,
        default=0,
        help="how many shards this job owns; 0 = all of --num-shards (single job)",
    )
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "xhigh"))
    # A10: 9 headline + 4 annex = 13 tasks x 150 demos. The 3 dropped tasks (WashLettuce and
    # RinseSinkBasin ceilinged, GetToastedBread floored) are out of the corpus too -- a task that
    # cannot be measured cannot contribute a testable cross-task edge.
    ap.add_argument("--corpus", default="robocasa_mem13_a10+remembench13+robomme16")
    ap.add_argument(
        "--tasks",
        default=(
            "ScrubCuttingBoard,KettleBoiling,SearingMeat,GatherTableware,PanTransfer,"
            "HeatKebabSandwich,StirVegetables,RecycleBottlesByType,CategorizeCondiments,"
            "PackIdenticalLunches,CuttingToolSelection,PortionHotDogs,SeparateFreezerRack"
        ),
        help="A10 headline+annex; shipped to the node as WSM_TASKS",
    )
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    # 12288, not 4096. Measured on this exact prompt across three completed judge shards:
    # 5,299-5,474 completion tokens per anchor, all run at 12,288 with ZERO truncations. At 4,096
    # most buckets finish_reason == "length" and validate_bucket_file rejects every one of them, so
    # the job burns a node and produces a store the resume gate calls invalid (§44.3).
    ap.add_argument("--pass2-max-tokens", type=int, default=12288)
    ap.add_argument(
        "--pass1-max-tokens",
        type=int,
        default=0,
        help="override the frozen DESCRIPTOR_MAX_TOKENS (pilots at higher effort need "
        "more room); 0 keeps the frozen value",
    )
    ap.add_argument(
        "--limit-segments", type=int, default=0, help="pass1 pilot: stop after roughly this many SEGMENTS per shard"
    )
    ap.add_argument("--concurrency", type=int, default=64)
    S3 = WSM_ROBOCASA_S3
    ap.add_argument(
        "--dataset-s3",
        default=f"{S3}/datasets/v1.0/target",
        help="RoboCasa lerobot mp4s; pass 1 decodes frames straight from them",
    )
    ap.add_argument(
        "--labels-s3", default=f"{S3}/wsm_labels", help="FROZEN keyframe label store -- the segmentation authority"
    )
    ap.add_argument(
        "--captions-s3", default=f"{S3}/wsm_labels_captions", help="H13 caption store, used only as a per-segment hint"
    )
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument(
        "--domain",
        default="robocasa",
        choices=("robocasa", "remembench", "robomme", "robocerebra"),
        help="pass1 only: which frame source the node stages and which caption_segments --domain branch runs",
    )
    # NB the sealed H12 tarball lives under the STUDY root, not under this module's STUDY_ROOT
    # (which is already the .../artifacts/deliberation subtree).
    ap.add_argument(
        "--rcb-data-s3",
        default=(
            f"{LONG_CONTEXT_STUDY_S3}/"
            "robocerebra/data/lerobot/"
            "8ce6785b6f57ef3e34d6ca55fd0e3f30be8e19255869886838727635ffc0aa29.tar"
        ),
    )
    ap.add_argument("--rcb-data-sha256", default="8ce6785b6f57ef3e34d6ca55fd0e3f30be8e19255869886838727635ffc0aa29")
    ap.add_argument(
        "--pass1-extra-s3-in",
        default="",
        help="embed: comma-separated extra pass-1 store prefixes merged into the same "
        "domain-nested tree (e.g. a new domain's own run_id prefix)",
    )
    ap.add_argument(
        "--anchor-domains",
        default="",
        help="embed/mine: restrict ANCHORS to these domains; candidates stay the whole "
        "corpus. This is what makes a pass-2 run a DELTA.",
    )
    ap.add_argument(
        "--expect-anchors",
        type=int,
        default=0,
        help="pass2: assert this many anchors were judged before success (0 disables)",
    )
    ap.add_argument(
        "--expect-domain-anchors",
        default="",
        help="pass2: JSON dict of per-domain anchor counts to assert, e.g. "
        '\'{"robocasa":9708,"remembench":1333,"robomme":8812,"robocerebra":8869}\'',
    )
    ap.add_argument(
        "--expect-episodes",
        type=int,
        default=0,
        help="pass1: assert this many episode files exist for --domain before the job may report success (0 disables)",
    )
    ap.add_argument(
        "--expect-segments", type=int, default=0, help="pass1: assert this total segment count across those files"
    )
    ap.add_argument("--pass1-s3-in", default="")
    ap.add_argument("--embed-s3-in", default="")
    ap.add_argument(
        "--measured-json", default="", help="pilot measurements; REQUIRED unless --max-run-seconds is explicit (A6)"
    )
    ap.add_argument("--max-run-seconds", type=int, default=0)
    ap.add_argument(
        "--startup-seconds",
        type=int,
        default=1800,
        help="image pull + model download + vLLM load, added on top of the estimate",
    )
    ap.add_argument("--plan-out", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-submit", action="store_true")
    ap.add_argument(
        "--plan-pass2-layout",
        action="store_true",
        help="apply the sizing rule to the MEASURED pass-2 rate and print the "
        "1-job or 2-job command set; submits nothing",
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.plan_pass2_layout:
        plan_pass2_layout(args)
        return
    plan = build_plan(args)
    # Plan-backed queue: the guardrail pins ResourceConfig.TrainingPlanArn, and an explicitly
    # pinned plan REPLACES the implicit reserved-capacity request -- so the sealed env must
    # carry the opt-out or the job sits SCHEDULED forever.
    # Batch REJECTS a job on a reserved-capacity queue that does not carry this variable at all
    # ("Missing SM_USE_RESERVED_CAPACITY environment variable for reserved capacity queue"), and
    # the guardrail only validates the VALUE when one is present -- so omitting it passed every
    # local check and failed instantly at submit. Set it on both branches, as the reference
    # launchers do (submit_pi_stage_s.py:1147, submit_robocerebra.py:380).
    plan["environment"]["SM_USE_RESERVED_CAPACITY"] = "0" if "training-plan" in plan["queue"] else "1"

    if args.plan_out:
        p = Path(args.plan_out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(plan, indent=1))
        print(f"[plan] wrote {p}")

    if args.dry_run or not args.confirm_submit:
        print(json.dumps(plan, indent=1))
        print("\nDRY RUN — nothing submitted. Submission requires:")
        print("  1. live SSO creds (`aws sso login`; the current token is expired)")
        print("  2. a 10-minute queue-depth + no-op probe on BOTH p5 and p5e (A7) before choosing the venue")
        print("  3. explicit user approval, then --confirm-submit")
        return

    src = REPO_ROOT
    # Hash the SANITIZED staged copy, exactly like submit_pi_stage_s.py:575 — the raw tree includes
    # docs/markers the sanitizer strips, so a raw-tree hash can never match the in-submit check.
    with prepared_source_bundle(str(src), ENTRY, plan["environment"], None) as (_staged, _e, _v):
        sanitized_sha = source_tree_sha256(_staged)
    result = submit_training_job(
        entry=ENTRY,
        source_dir=str(src),
        environment=plan["environment"],
        image_uri=plan["image"],
        instance_type=plan["instance_type"],
        volume_size=plan["volume_gb"],
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "study", "Value": "h14-deliberation"},
            {"Key": "stage", "Value": plan["stage"]},
            {"Key": "run_id", "Value": plan["run_id"]},
        ],
        retry_config=None,
        job_name=f"h14-delib-{plan['stage']}-{plan['run_id']}-{int(time.time())}",
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
