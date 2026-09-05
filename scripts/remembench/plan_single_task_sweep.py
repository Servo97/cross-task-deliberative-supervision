#!/usr/bin/env python3
"""Balance the 24 single-task ReMemBench eval cells across the box's 4 GPUs.

One cell = (task, arm) = one checkpoint = one policy server + one rollout client. Cells run
SEQUENTIALLY within a GPU (never two servers on one GPU: a workspace serve holds two openpi
models plus a torch encoder, and the memory does not fit twice).

Balancing by cell COUNT would be badly wrong here — cells differ ~8x in cost because the
per-task horizon and held-out episode count differ (MemHeatPot: 2600 steps x 10 episodes x3
= 78k env steps; MemRetrieveOilsFromCounterRL: 1100 x 3 x3 = 9.9k). Greedy
longest-processing-time on worst-case env steps gets the four GPUs within ~0.1% of each other.

  python scripts/remembench/plan_single_task_sweep.py --matrix <tsv> --manifest <heldout.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Serve path per arm. The JEPA aux (jw01k16) is a TRAIN-time target whose head params are
# dropped at serve load, so it deploys through the plain base interface exactly like base.
ARM_SERVE = {
    "base": "base",
    "jw01k16": "base",
    "tanh": "workspace",
    "dnw2": "workspace",
    "dnw8": "workspace",
    "dnw16": "workspace",
    "combo-k16": "workspace",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="TSV: short<TAB>task<TAB>arm<TAB>run_id")
    ap.add_argument("--manifest", required=True, help="sealed 88-episode heldout manifest")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--out-dir", required=True, help="where to write gpu<N>.tsv worklists")
    args = ap.parse_args()

    manifest = json.loads(open(args.manifest).read())
    per_task = collections.defaultdict(list)
    for record in manifest["episodes"]:
        per_task[record["task"]].append(record)

    cells = []
    with open(args.matrix) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            short, task, arm, run_id = line.split("\t")
            if arm not in ARM_SERVE:
                raise SystemExit(f"unknown arm {arm!r}")
            records = per_task.get(task)
            if not records:
                raise SystemExit(f"task {task!r} absent from the manifest")
            cost = sum(int(r["horizon"]) for r in records) * args.rollouts
            cells.append(
                {
                    "short": short,
                    "task": task,
                    "arm": arm,
                    "run_id": run_id,
                    "serve": ARM_SERVE[arm],
                    "episodes": len(records),
                    "rollouts": len(records) * args.rollouts,
                    "cost": cost,
                }
            )
    if len(cells) != len({c["run_id"] for c in cells}):
        raise SystemExit("duplicate run_id in the matrix")

    bins = [[] for _ in range(args.num_gpus)]
    loads = [0] * args.num_gpus
    for cell in sorted(cells, key=lambda c: (-c["cost"], c["run_id"])):
        target = min(range(args.num_gpus), key=lambda i: (loads[i], i))
        bins[target].append(cell)
        loads[target] += cell["cost"]

    os.makedirs(args.out_dir, exist_ok=True)
    total_rollouts = sum(c["rollouts"] for c in cells)
    print(f"{len(cells)} cells, {total_rollouts} rollouts, {sum(loads)} worst-case env steps")
    for gpu in range(args.num_gpus):
        path = os.path.join(args.out_dir, f"gpu{gpu}.tsv")
        with open(path, "w") as handle:
            for cell in bins[gpu]:
                handle.write(
                    "\t".join(
                        [
                            cell["short"],
                            cell["task"],
                            cell["arm"],
                            cell["run_id"],
                            cell["serve"],
                            str(cell["rollouts"]),
                        ]
                    )
                    + "\n"
                )
        summary = ", ".join(f"{c['short']}/{c['arm']}" for c in bins[gpu])
        print(
            f"  gpu{gpu}: {len(bins[gpu])} cells, {loads[gpu]:>7d} steps, "
            f"{sum(c['rollouts'] for c in bins[gpu]):>3d} rollouts -> {summary}"
        )
    spread = (max(loads) - min(loads)) / max(loads) * 100
    print(f"imbalance: {spread:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
