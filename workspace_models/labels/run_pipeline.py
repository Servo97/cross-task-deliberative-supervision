"""Orchestrate the WSM label pipeline over many tasks across the local GPUs.

TASK-LEVEL parallelism: each task runs extract -> qwen -> molmo -> build as a subprocess pipeline
pinned to one GPU; up to (len(gpus) * per_gpu) tasks run concurrently. MolmoPoint pointing is
sequential per process (its vendored code is NOT batch-safe — batching silently mis-shapes the
point mask and would yield wrong coordinates), so throughput comes from running multiple
task-pipelines at once, ~1 model (~16 GB) resident per worker.

  python -m workspace_models.labels.run_pipeline --tasks OpenDrawer,PrepareCoffee \
      --episodes 0-49 --out ~/Research/TRI/wsm_data/wsm_vlm_rc --gpus 0,1 --per-gpu 1 --qc
  python -m workspace_models.labels.run_pipeline --tasks target50 --episodes 0-49 \
      --out ~/Research/TRI/wsm_data/wsm_vlm_rc --gpus 0,1 --per-gpu 2

--tasks accepts a comma list of task names OR one of: target50 / atomic_seen / composite_seen /
composite_unseen (resolved via utils.soup). ReMemBench's 13 Mem* tasks have no soup shorthand here
(utils.soup.remembench_soup is a separate, robocasa-free glob) — pass them as an explicit list.
Per-task logs land in <out>/_logs/<task>.log.

Label SPECS (`--spec`, see qwen_subgoals.py): `salient` (default, the original) or `causal_v1`
(manipulated object + goal slot only). A non-default spec must carry `--tag` so its intermediates
sit beside the salient ones instead of overwriting them, and `--labels-out` so its npz land in
their own label root (the trainer's npz filename is fixed, so specs separate by ROOT):

  python -m workspace_models.labels.run_pipeline --tasks MemHeatPot,MemWashAndReturnLeft \\
      --out /data/work/rmb/frames --labels-out /data/work/causal/labels_causal_v1 \\
      --spec causal_v1 --tag _causal_v1 --geom pi --skip-extract --gpus 0,1,2,3 --per-gpu 3
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPECIAL = {"target50", "atomic_seen", "composite_seen", "composite_unseen"}


def resolve_tasks(spec: str) -> list[str]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and parts[0] in SPECIAL:
        from utils.soup import combined_target_soup, resolve_soup

        soup = (
            combined_target_soup(demo_fraction=1.0)
            if parts[0] == "target50"
            else resolve_soup(split="target", task_set=parts[0], source="human", demo_fraction=1.0)
        )
        out: list[str] = []
        for m in soup:
            if m["task"] not in out:
                out.append(m["task"])
        return out
    return parts


def resolve_episodes(spec: str) -> str:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return ",".join(str(i) for i in range(int(a), int(b) + 1))
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks", required=True, help="comma list, or target50/atomic_seen/composite_seen/composite_unseen"
    )
    ap.add_argument("--episodes", default="0-49", help="'0-49' range or '0,1,2' list (ignored if --num-demos)")
    ap.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="select the seed-0 filter_key keep-set per task (IDENTICAL to the policy "
        "finetune's 150_demos selection); overrides --episodes",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for the --num-demos keep-set (policy uses 0)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--per-gpu", type=int, default=1, help="task-pipelines per GPU (2 needs ~32GB)")
    ap.add_argument("--max-frames", type=int, default=14)
    ap.add_argument("--geom", choices=["groot", "pi"], default="groot", help="patch geometry for the build stage")
    ap.add_argument(
        "--spec",
        choices=["salient", "causal_v1"],
        default="salient",
        help="label spec (Qwen system prompt + object schema); see qwen_subgoals.py",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="artifact suffix that namespaces a non-default spec's intermediates, e.g. "
        "'_causal_v1'. REQUIRED with --spec causal_v1 so the salient artifacts in "
        "the same frames dir are never overwritten.",
    )
    ap.add_argument(
        "--labels-out",
        default="",
        help="label root for the built npz (default: --out). The trainer's filename is "
        "fixed, so specs are separated by ROOT.",
    )
    ap.add_argument("--qc", action="store_true", help="write geometry-QC overlays")
    ap.add_argument("--skip-extract", action="store_true", help="reuse existing frames (regeneration)")
    ap.add_argument(
        "--sim-python", default="~/Research/envs/robocasa_env/bin/python", help="env with robocasa+lerobot (extract)"
    )
    ap.add_argument(
        "--vlm-python", default="~/Research/envs/vlm_labeler/bin/python", help="env with Qwen/Molmo (qwen+molmo+build)"
    )
    args = ap.parse_args()

    if args.spec != "salient" and not args.tag:
        raise SystemExit(
            f"--spec {args.spec} requires --tag (e.g. --tag _{args.spec}); an untagged "
            "non-default spec would overwrite the salient intermediates in --out"
        )
    tasks = resolve_tasks(args.tasks)
    eps = resolve_episodes(args.episodes)
    out = Path(args.out).expanduser()
    logdir = out / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    sim_py = os.path.expanduser(args.sim_python)
    vlm_py = os.path.expanduser(args.vlm_python)
    slots: "queue.Queue[str]" = queue.Queue()
    for g in (g.strip() for g in args.gpus.split(",")):
        for _ in range(args.per_gpu):
            slots.put(g)
    nslots = slots.qsize()
    sel = f"seed-{args.seed} keep-set num_demos={args.num_demos}" if args.num_demos is not None else f"episodes {eps}"
    print(
        f"[run] {len(tasks)} tasks x {sel} | {nslots} concurrent slots (gpus={args.gpus} x{args.per_gpu}) | out={out}",
        flush=True,
    )

    def run(name: str, cmd: list[str], log: Path) -> int:
        with open(log, "a") as f:
            f.write(f"\n===== {name} =====\n$ {' '.join(cmd)}\n")
            f.flush()
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO)).returncode

    def pipeline(task: str) -> tuple[str, str]:
        g = slots.get()
        log = logdir / f"{task}.log"
        dev = f"cuda:{g}"
        steps = []
        if not args.skip_extract:
            extract_cmd = [
                sim_py,
                "-m",
                "workspace_models.labels.extract_frames",
                "--task",
                task,
                "--stride",
                str(args.stride),
                "--out",
                str(out),
            ]
            if args.num_demos is not None:
                extract_cmd += ["--num-demos", str(args.num_demos), "--seed", str(args.seed)]
            else:
                extract_cmd += ["--episodes", eps]
            steps.append(("extract", extract_cmd))
        steps += [
            (
                "qwen",
                [
                    vlm_py,
                    "-m",
                    "workspace_models.labels.qwen_subgoals",
                    "--task",
                    task,
                    "--in",
                    str(out),
                    "--device",
                    dev,
                    "--max-frames",
                    str(args.max_frames),
                    "--spec",
                    args.spec,
                    "--tag",
                    args.tag,
                ],
            ),
            (
                "molmo",
                [
                    vlm_py,
                    "-m",
                    "workspace_models.labels.molmo_points",
                    "--task",
                    task,
                    "--in",
                    str(out),
                    "--device",
                    dev,
                    "--tag",
                    args.tag,
                ],
            ),
            (
                "build",
                [
                    vlm_py,
                    "-m",
                    "workspace_models.labels.build_salient_sets",
                    "--task",
                    task,
                    "--in",
                    str(out),
                    "--geom",
                    args.geom,
                    "--tag",
                    args.tag,
                ]
                + (["--out", str(Path(args.labels_out).expanduser())] if args.labels_out else [])
                + (["--qc-dir", str(out / "_qc")] if args.qc else []),
            ),
        ]
        try:
            for name, cmd in steps:
                rc = run(name, cmd, log)
                if rc != 0:
                    print(f"[run] {task}: FAIL @ {name} (rc={rc}) — see {log}", flush=True)
                    return task, f"FAIL@{name}"
            print(f"[run] {task}: ok", flush=True)
            return task, "ok"
        finally:
            slots.put(g)

    with ThreadPoolExecutor(max_workers=nslots) as ex:
        results = list(ex.map(pipeline, tasks))

    ok = [t for t, s in results if s == "ok"]
    bad = [(t, s) for t, s in results if s != "ok"]
    print(f"\n[run] DONE  ok={len(ok)}/{len(tasks)}" + (f"  failed={bad}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
