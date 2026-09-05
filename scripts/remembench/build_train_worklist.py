#!/usr/bin/env python3
"""Emit the ReMemBench TRAIN-split worklist that drives the 256px re-render.

The split is not recomputed here in any new way: it reuses ``build_remembench_episode_manifest
.scan_demos`` (ordering by ``(session, demo_index)``) and ``remembench_tasks.heldout_split``
(tail fraction 0.2, floor 3), so train == exactly the complement of the sealed held-out
manifest. The worklist is the small artifact that gets shipped to the render box; the box
never needs wsmv2 or the held-out manifest.

Per-task ``episode_index`` is assigned 0..n-1 over the ordered train demos, and becomes the
LeRobot ``episode_index`` -- which is also the key the omega cache uses
(``omega/<task>/demo_<episode_index:06d>/w.npz``), so it must stay stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "launch"))

from build_remembench_episode_manifest import HELDOUT_FRACTION, HELDOUT_MINIMUM, scan_demos  # noqa: E402

from vla_training.eval.remembench_tasks import (  # noqa: E402
    REMEMBENCH_TASKS,
    get_remembench_category,
    heldout_split,
)


def build(data_root: str | Path) -> dict:
    per_task = scan_demos(data_root)
    tasks = []
    total = 0
    for task in REMEMBENCH_TASKS:
        demos = per_task[task]
        train, heldout = heldout_split(range(len(demos)), fraction=HELDOUT_FRACTION, minimum=HELDOUT_MINIMUM)
        episodes = []
        for episode_index, position in enumerate(train):
            demo = demos[position]
            episodes.append(
                {
                    "episode_index": episode_index,
                    "session": demo["session"],
                    "demo_key": demo["demo_key"],
                    "demo_index": demo["demo_index"],
                    "length": demo["length"],
                    # ep_meta lang is episode-specific (object names vary); the render keeps the
                    # per-episode string as the LeRobot task_description. The task-level TEMPLATE
                    # form lives in the separate task-lang table.
                    "lang": demo["ep_meta"]["lang"],
                }
            )
        total += len(episodes)
        tasks.append(
            {
                "task": task,
                "category": get_remembench_category(task),
                "n_demos": len(demos),
                "n_train": len(train),
                "n_heldout": len(heldout),
                "episodes": episodes,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "remembench_train_worklist",
        "selection": {
            "kind": "remembench_tail_fraction_complement",
            "fraction": HELDOUT_FRACTION,
            "minimum": HELDOUT_MINIMUM,
        },
        "demo_filename": "demo_im128_notp.hdf5",
        "total_train_episodes": total,
        "tasks": tasks,
    }
    return payload


def canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(args.data_root)
    data = canonical_bytes(payload)
    Path(args.output).write_bytes(data)
    print(f"path={args.output}")
    print(f"sha256={hashlib.sha256(data).hexdigest()}")
    print(f"train_episodes={payload['total_train_episodes']}")
    for task in payload["tasks"]:
        print(f"  {task['task']:32s} n={task['n_demos']:3d} train={task['n_train']:3d} heldout={task['n_heldout']:3d}")


if __name__ == "__main__":
    main()
