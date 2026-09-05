"""H14 pass 1, run LOCALLY on the 2x5090s -- sharded, resumable, built to hand off to p5.

The cluster path is frozen (org SCP denies batch:SubmitServiceJob for this identity on p5 and p5e).
Pass 1 runs here so the campaign keeps moving, and so that if cluster access returns mid-run a p5
job can pick up the REMAINING shards with no coordination and no lost work.

THE HANDOFF CONTRACT (why this file exists rather than a bare for-loop):

  1. The shard layout is the CLUSTER's, not the local one. `--num-shards 8` even though there are
     at most 2 GPUs, because build_jobs partitions with jobs[shard::num_shards] and a p5 node fans
     out global_shard = node_rank*8 + local_gpu. Same modulus => the same 8 disjoint episode sets in
     both venues. A local `--num-shards 2` would partition differently and a takeover would have to
     reason about overlap.
  2. ONE store, written by both venues: same root, same ep_%06d.descriptors.json names.
  3. Resume is structural and per-episode. validate_existing_descriptors re-parses and shape-checks
     every candidate, so a p5 shard re-running finished work skips it, and a shard killed mid-episode
     loses only that episode (writes are atomic renames). Correctness therefore does NOT depend on
     point 1 -- matching the layout just avoids duplicated in-flight work.
  4. Per-shard provenance: caption_segments already writes _provenance/run_shard<N>_<ts>.json and
     usage_shard<N>_<ts>.json; both venues append to the same _provenance/ directory.
  5. Quantization differs across venues and is RECORDED, not hidden. Local replicas are NVFP4 (an
     H100 cannot run NVFP4 cutlass kernels; p5 would be FP8). Every output file carries its own
     `model` and qa_descriptors reports the model histogram, so a mixed-provenance store shows up in
     QA rather than being discovered later.

GPU ETIQUETTE (the RoboCerebra v3 close-out has priority):
  * A GPU is claimed only after --idle-polls CONSECUTIVE clean polls. One idle sample is not idle.
  * While held, every --yield-check-seconds the supervisor re-checks; if a foreign process appears
    on a held GPU that GPU's work stops and the GPU is released -- structural resume makes that free.
  * "Foreign" = any compute PID on the device this supervisor did not start.

  python scripts/deliberation/local_pass1.py --gpus 0 --num-shards 8
  python scripts/deliberation/local_pass1.py --gpus 0,1 --num-shards 8   # once GPU1 drains
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsm_settings import ENVS_ROOT  # noqa: E402

A10_TASKS = (
    "ScrubCuttingBoard,KettleBoiling,SearingMeat,GatherTableware,PanTransfer,"
    "HeatKebabSandwich,StirVegetables,RecycleBottlesByType,CategorizeCondiments,"
    "PackIdenticalLunches,CuttingToolSelection,PortionHotDogs,SeparateFreezerRack"
)

RMB_TASKS = (
    "MemFruitInSinkLeftFar,MemFruitInSinkRightFar,MemHeatPot,MemHeatPotMultiple,"
    "MemPutKBowlInCabinet,MemPutKBreadInMicrowave,MemRetrieveOilsFromCounterLL,"
    "MemRetrieveOilsFromCounterLR,MemRetrieveOilsFromCounterRL,"
    "MemRetrieveOilsFromCounterRR,MemWashAndReturnLeft,MemWashAndReturnRight,"
    "MemWashAndReturnSameLocation"
)

ROBOMME_TASKS = (
    "PatternLock,ButtonUnmaskSwap,ButtonUnmask,VideoPlaceButton,VideoUnmaskSwap,"
    "PickXtimes,StopCube,SwingXtimes,PickHighlight,MoveCube,InsertPeg,RouteStick,"
    "BinFill,VideoPlaceOrder,VideoRepick,VideoUnmask"
)

ROBOMME_ROOT = (
    "~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
    "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20"
)

# Order is the coordinator's: RoboCasa (already running) -> RoboMME -> ReMemBench, then one QA over
# the whole store. Each domain is a separate subdirectory and a separate --domain code path, but
# they share the store root so a single QA pass covers all three.
#
# ReMemBench needed NO new frame source: `remembench_v02` on S3 is a LeRobot tree with the same 3
# views at 256 px and the same video_path template as RoboCasa, which is exactly the geometry
# causal_v1's keyframes were built against. Only RoboMME needed a reader (parquet-embedded frames,
# 2 views, RLE subgoal segmentation).
DOMAIN_SPECS = {
    "robocasa": dict(
        domain="robocasa",
        subdir="robocasa",
        tasks=A10_TASKS,
        dataset_root="~/Research/robocasa/datasets/v1.0/target",
        labels_root="~/Research/TRI/wsm_data/wsm_labels_pi_mirror",
        hints_root="~/Research/TRI/wsm_data/wsm_labels_captions",
    ),
    "robomme": dict(
        domain="robomme",
        subdir="robomme",
        tasks=ROBOMME_TASKS,
        dataset_root=ROBOMME_ROOT,
        labels_root="",
        hints_root="",
    ),
    "remembench": dict(
        domain="remembench",
        subdir="remembench",
        tasks=RMB_TASKS,
        dataset_root="~/Research/TRI/wsm_data/remembench_v02",
        labels_root="~/Research/TRI/wsm_data/wsm_labels_causal_v1/remembench13",
        hints_root="",
    ),
    # RoboCerebra enumerates its own tasks (947 BDDL stems over 994 episodes), so `tasks` is
    # empty = "every task in the index" rather than a pinned roster like the other three.
    "robocerebra": dict(
        domain="robocerebra",
        subdir="robocerebra",
        tasks="",
        dataset_root=("~/Research/TRI/wsm_data/robocerebra/lerobot_home/wsmv2/robocerebra_train"),
        labels_root="",
        hints_root="",
    ),
}


def gpu_uuid_map() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout
    m = {}
    for line in out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        m[int(idx)] = uuid
    return m


def compute_pids_on(uuid: str) -> set:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout
    pids = set()
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, u = [x.strip() for x in line.split(",")]
        if u == uuid:
            pids.add(int(pid))
    return pids


def log(msg: str, path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with path.open("a") as f:
        f.write(line + "\n")


class GpuWorker:
    """One vLLM replica plus a sequence of shard clients on a single GPU."""

    def __init__(self, gpu: int, args, logf: Path):
        self.gpu = gpu
        self.args = args
        self.logf = logf
        self.port = 8100 + gpu
        self.server = None
        self.client = None
        self.owned = set()
        self.current_shard = None

    def foreign_pids(self, uuid: str) -> set:
        return compute_pids_on(uuid) - self.owned

    def wait_sustained_idle(self, uuid: str, budget: float | None = None) -> bool:
        clean = 0
        deadline = time.time() + (self.args.claim_timeout if budget is None else budget)
        while time.time() < deadline:
            fp = self.foreign_pids(uuid)
            if fp:
                if clean:
                    log(f"[gpu{self.gpu}] idle streak broken by {sorted(fp)}; resetting", self.logf)
                clean = 0
            else:
                clean += 1
                log(f"[gpu{self.gpu}] idle poll {clean}/{self.args.idle_polls}", self.logf)
                if clean >= self.args.idle_polls:
                    return True
            time.sleep(self.args.idle_poll_seconds)
        return False

    def server_up(self) -> bool:
        import urllib.request

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=3)
            return True
        except Exception:
            return False

    def start_server(self) -> bool:
        if self.server_up():
            log(f"[gpu{self.gpu}] reusing the vLLM replica already on :{self.port}", self.logf)
            self.owned |= compute_pids_on(gpu_uuid_map()[self.gpu])
            return True
        env = dict(os.environ, WSM_GPU=str(self.gpu), WSM_PORT=str(self.port), WSM_MODEL=self.args.model)
        slog = Path(self.args.store).expanduser() / "_logs" / f"vllm_gpu{self.gpu}.log"
        slog.parent.mkdir(parents=True, exist_ok=True)
        log(f"[gpu{self.gpu}] starting vLLM -> {slog}", self.logf)
        with slog.open("a") as f:
            self.server = subprocess.Popen(
                [str(REPO / "scripts" / "deliberation" / "serve_vllm.sh")],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(REPO),
                preexec_fn=os.setsid,
            )
        for _ in range(180):
            if self.server_up():
                self.owned |= compute_pids_on(gpu_uuid_map()[self.gpu])
                log(f"[gpu{self.gpu}] vLLM ready on :{self.port}", self.logf)
                return True
            if self.server.poll() is not None:
                log(f"[gpu{self.gpu}] vLLM died during startup; see {slog}", self.logf)
                return False
            time.sleep(10)
        log(f"[gpu{self.gpu}] vLLM never became ready", self.logf)
        return False

    def start_shard(self, item) -> None:
        dom, shard = item
        spec = DOMAIN_SPECS[dom]
        store = Path(self.args.store).expanduser()
        clog = store / "_logs" / f"{dom}_shard{shard}_gpu{self.gpu}.log"
        clog.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.args.python,
            "-m",
            "workspace_models.labels.caption_segments",
            "--domain",
            spec["domain"],
            "--spec",
            "descriptor",
            "--backend",
            "vllm",
            "--model",
            self.args.model,
            "--vllm-base-url",
            f"http://127.0.0.1:{self.port}/v1",
            "--tasks",
            spec["tasks"],
            "--dataset-root",
            spec["dataset_root"],
            "--labels-root",
            spec["labels_root"],
            "--caption-hints-root",
            spec["hints_root"],
            "--out",
            str(store / spec["subdir"]),
            "--shard",
            str(shard),
            "--num-shards",
            str(self.args.num_shards),
            *(["--episodes", self.args.episodes] if getattr(self.args, "episodes", "") else []),
            "--concurrency",
            str(self.args.concurrency),
            "--reasoning-effort",
            "low",
            "--log-every",
            "20",
        ]
        self.current_shard = item
        log(f"[gpu{self.gpu}] {dom} shard {shard}/{self.args.num_shards} -> {clog}", self.logf)
        with clog.open("a") as f:
            self.client = subprocess.Popen(
                cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
            )

    def stop(self, why: str) -> None:
        log(f"[gpu{self.gpu}] releasing: {why}", self.logf)
        for proc, name in ((self.client, "client"), (self.server, "server")):
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                for _ in range(30):
                    if proc.poll() is not None:
                        break
                    time.sleep(1)
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                log(f"[gpu{self.gpu}] {name} stopped", self.logf)
        self.client = None
        self.server = None


def remaining_shards(args, domains: list) -> list:
    """[(domain, shard, episodes_remaining)] using build_jobs, so the count reflects the REAL
    resume gate (re-parse + shape check) rather than a directory listing."""
    import types

    from workspace_models.labels.caption_segments import build_jobs

    store = Path(args.store).expanduser()
    todo = []
    for dom in domains:
        spec = DOMAIN_SPECS[dom]
        for s in range(args.num_shards):
            ns = types.SimpleNamespace(
                domain=spec["domain"],
                labels_root=spec["labels_root"],
                dataset_root=spec["dataset_root"],
                out=str(store / spec["subdir"]),
                spec="descriptor",
                caption_hints_root=spec["hints_root"],
                force=False,
                frames_per_segment=3,
                max_images=72,
                shard=s,
                num_shards=args.num_shards,
                limit=0,
                limit_segments=0,
                stratify_tasks=False,
                episodes=getattr(args, "episodes", ""),
            )
            n = len(build_jobs(ns, [t for t in spec["tasks"].split(",") if t]))
            if n:
                todo.append((dom, s, n))
    return todo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--num-shards", type=int, default=8, help="MUST match the cluster layout (8 = one p5 node's GPUs)")
    ap.add_argument("--store", default="~/Research/TRI/wsm_data/deliberation/pass1_store")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    ap.add_argument(
        "--domains",
        default="robocasa,robomme,remembench",
        help="run order; each is a separate --domain code path sharing the store root",
    )
    ap.add_argument("--python", default=str(ENVS_ROOT / "vlm_labeler" / "bin" / "python"))
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--idle-polls", type=int, default=3)
    ap.add_argument("--idle-poll-seconds", type=int, default=60)
    ap.add_argument("--yield-check-seconds", type=int, default=1800)
    ap.add_argument("--claim-timeout", type=int, default=6 * 3600)
    ap.add_argument("--sweeps", type=int, default=4, help="re-plan and sweep again while progress is being made")
    ap.add_argument(
        "--episodes",
        default="",
        help="TOP-UP scope, passed through to caption_segments: a comma-separated "
        "episode list or a path to a JSON receipt with `topup_episodes`. "
        "Requires --num-shards 1 (see caption_segments.episode_allowlist).",
    )
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    store = Path(args.store).expanduser()
    (store / "_logs").mkdir(parents=True, exist_ok=True)
    logf = store / "_logs" / "supervisor.log"

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    todo = remaining_shards(args, domains)
    total = sum(n for _, _, n in todo)
    by_dom = {}
    for d, s_, n in todo:
        by_dom[d] = by_dom.get(d, 0) + n
    log(
        f"SHARD PLAN num_shards={args.num_shards} (cluster layout) domains={domains} "
        f"episodes_remaining={by_dom} total={total}",
        logf,
    )
    (store / "shard_plan.json").write_text(
        json.dumps(
            {
                "num_shards": args.num_shards,
                "layout": "cluster (global_shard = node_rank*8 + local_gpu)",
                "domain_order": domains,
                "remaining": [{"domain": d, "shard": s_, "episodes": n} for d, s_, n in todo],
                "episodes_remaining_by_domain": by_dom,
                "episodes_total": total,
                "model": args.model,
                "store": str(store),
                "handoff": (
                    "a p5 job with the same --num-shards writes this same store; "
                    "validate_existing_descriptors skips finished work"
                ),
            },
            indent=1,
        )
    )
    if args.plan_only or not todo:
        log("plan-only (or nothing to do); exiting", logf)
        return

    uuids = gpu_uuid_map()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    workers = {}
    queue = [(d, s_) for d, s_, _ in todo]
    prev_left = total + 1
    idle_streak: dict = {}

    def try_claim(g: int, budget: float) -> bool:
        """Attempt to claim one GPU within `budget` seconds. Never blocks the fleet: a GPU that is
        busy is simply left alone and retried on the next tick."""
        w = GpuWorker(g, args, logf)
        # Adopt a replica this campaign already left running on this GPU BEFORE the idle gate.
        # Otherwise our own server counts as a foreign process, the streak never accumulates, and
        # the gate blocks forever on a GPU that is in fact ours.
        if w.server_up():
            w.owned |= compute_pids_on(uuids[g])
            log(f"[gpu{g}] adopted our existing replica on :{w.port} (pids {sorted(w.owned)})", logf)
        if not w.wait_sustained_idle(uuids[g], budget):
            fp = w.foreign_pids(uuids[g])
            log(f"[gpu{g}] not sustained-idle (busy: {sorted(fp)}); leaving it alone", logf)
            return False
        if not w.start_server():
            w.stop("server failed")
            return False
        workers[g] = w
        return True

    # The first GPU may wait; ADDITIONAL GPUs get only one gate-length budget, because blocking the
    # whole fleet for hours on a GPU someone else is using is worse than running on fewer GPUs.
    # Un-held GPUs are retried on every yield tick, so one that frees later is picked up then.
    gate_len = args.idle_polls * args.idle_poll_seconds + 5
    for i, g in enumerate(gpus):
        log(
            f"[gpu{g}] claiming: need {args.idle_polls} consecutive idle polls ({args.idle_poll_seconds}s apart)", logf
        )
        try_claim(g, args.claim_timeout if i == 0 and not workers else gate_len)

    if not workers:
        log("no GPU could be claimed; exiting without doing work", logf)
        return

    last_yield_check = time.time()
    sweep = 1
    try:
        while True:
            while queue or any(w.client and w.client.poll() is None for w in workers.values()):
                for g, w in list(workers.items()):
                    # The replica is the fragile part: it can be killed out from under us (observed --
                    # a harness-owned background task was reaped mid-shard and every request then failed
                    # `Connection refused`). Own it, watch it, restart it. A shard interrupted this way
                    # is requeued and costs nothing, because resume is structural.
                    if not w.server_up():
                        log(f"[gpu{g}] replica on :{w.port} is DOWN", logf)
                        if w.current_shard is not None and w.client is not None:
                            if w.current_shard not in queue:
                                queue.insert(0, w.current_shard)
                            log(f"[gpu{g}] requeued shard {w.current_shard}", logf)
                        w.stop("replica down; restarting")
                        if not w.start_server():
                            log(f"[gpu{g}] replica would not restart -- dropping this GPU", logf)
                            workers.pop(g, None)
                            continue
                    if w.client is None or w.client.poll() is not None:
                        if w.client is not None:
                            rc = w.client.returncode
                            log(f"[gpu{g}] shard {w.current_shard} client exited rc={rc}", logf)
                            if rc != 0 and w.current_shard is not None and w.current_shard not in queue:
                                queue.insert(0, w.current_shard)
                                log(f"[gpu{g}] requeued shard {w.current_shard} (nonzero rc)", logf)
                            w.client = None
                        if queue:
                            w.start_shard(queue.pop(0))
                if time.time() - last_yield_check >= args.yield_check_seconds:
                    last_yield_check = time.time()
                    for g, w in list(workers.items()):
                        fp = w.foreign_pids(uuids[g])
                        if fp:
                            log(f"[gpu{g}] FOREIGN PROCESS {sorted(fp)} -- yielding this GPU", logf)
                            if w.current_shard is not None and w.client and w.client.poll() is None:
                                queue.insert(0, w.current_shard)
                            w.stop("yielded to a higher-priority job")
                            workers.pop(g, None)
                    # Opportunistic reclaim of GPUs we do NOT hold. Deliberately non-blocking:
                    # one probe per tick, with the idle streak carried ACROSS ticks. An earlier
                    # version called the blocking gate here, which would have stalled shard
                    # assignment for minutes each tick -- and in fact it never ran at all, because
                    # the patch that added it was a no-op (the replacement string matched nothing,
                    # so the assert passed and the file was unchanged). Verified present this time.
                    for g in gpus:
                        if g in workers or not queue:
                            continue
                        probe = GpuWorker(g, args, logf)
                        if probe.server_up():
                            probe.owned |= compute_pids_on(uuids[g])
                        fp = probe.foreign_pids(uuids[g])
                        if fp:
                            idle_streak[g] = 0
                            continue
                        idle_streak[g] = idle_streak.get(g, 0) + 1
                        log(f"[gpu{g}] free, idle tick {idle_streak[g]}/{args.idle_polls}", logf)
                        if idle_streak[g] >= args.idle_polls and probe.start_server():
                            workers[g] = probe
                            idle_streak[g] = 0
                            log(f"[gpu{g}] CLAIMED -- second replica joining the fleet", logf)
                    if not workers:
                        log(
                            "all GPUs yielded; remaining shards stay for the next window "
                            "(resume is structural, nothing is lost)",
                            logf,
                        )
                        break
                time.sleep(15)

            # A shard exiting rc=0 means its FIXED episode set is done, but a per-episode failure can
            # still leave gaps. Re-plan and sweep again until nothing is left or progress stops --
            # cheap insurance, and it converges because the partition is now stable.
            left = remaining_shards(args, domains)
            n_left = sum(n for _, _, n in left)
            if not left:
                log(f"sweep {sweep}: corpus COMPLETE", logf)
                break
            if sweep >= args.sweeps:
                log(f"sweep {sweep}: {n_left} episodes still missing, sweep budget exhausted", logf)
                break
            if n_left >= prev_left:
                log(
                    f"sweep {sweep}: no progress ({n_left} left, was {prev_left}) -- stopping to avoid "
                    "spinning on episodes that fail deterministically",
                    logf,
                )
                break
            prev_left = n_left
            sweep += 1
            queue = [(d, s_) for d, s_, _ in left]
            log(f"sweep {sweep}: {n_left} episodes remain; re-queued {len(queue)} shards", logf)
    except KeyboardInterrupt:
        log("interrupted", logf)
    finally:
        for w in workers.values():
            w.stop("run finished")
    log("supervisor done", logf)


if __name__ == "__main__":
    main()
