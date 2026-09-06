#!/usr/bin/env python3
"""A19 checkpoint-maturity curve for a RoboMME fixed-50 milestone campaign.

Reads every ``cells/*/result.complete.json`` under a synced campaign directory and prints, per
milestone step: cells landed, pooled successes / episodes, mean success with a Wilson 95 % interval,
and the per-task success grid. Then applies the pre-registered A19 selection rule on the BASE curve:
the reported step s* is the earliest milestone after which no later milestone improves the pooled
success by more than ``--mde`` percentage points. Selection-lane numbers (execute-10) are internally
consistent but NOT comparable to the paper-protocol anchor; the chosen step is re-scored under the
paper protocol separately.

Usage:
  python scripts/analysis/a19_rmme_curve.py <campaign dir> [--mde 5.0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def load_cells(root: Path) -> list[dict]:
    cells = []
    for path in sorted(root.glob("cells/*/result.complete.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("kind") != "robomme_fixed50_complete":
            continue
        cells.append(value)
    return cells


def curve(cells: list[dict]) -> dict[int, dict]:
    by_step: dict[int, dict] = defaultdict(lambda: {"tasks": {}, "successes": 0, "episodes": 0})
    for cell in cells:
        step = int(cell["checkpoint_step"])
        task = cell["task"]
        row = by_step[step]
        row["tasks"][task] = (int(cell["successes"]), int(cell["episodes"]))
        row["successes"] += int(cell["successes"])
        row["episodes"] += int(cell["episodes"])
    return dict(sorted(by_step.items()))


def select_step(points: dict[int, dict], mde_pp: float, n_tasks_expected: int | None) -> int | None:
    """Earliest milestone after which no later milestone improves pooled success by > mde_pp."""
    complete = [s for s, row in points.items() if n_tasks_expected is None or len(row["tasks"]) >= n_tasks_expected]
    if not complete:
        return None
    rate = {s: 100.0 * points[s]["successes"] / points[s]["episodes"] for s in complete}
    for s in complete:
        later = [rate[t] for t in complete if t > s]
        if all(r - rate[s] <= mde_pp for r in later):
            return s
    return complete[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--mde", type=float, default=5.0, help="selection tolerance in percentage points")
    parser.add_argument("--tasks", type=int, default=16, help="tasks per milestone for a complete point")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cells = load_cells(args.campaign_dir)
    points = curve(cells)
    tasks = sorted({t for row in points.values() for t in row["tasks"]})
    print(f"{len(cells)} cells landed under {args.campaign_dir}")
    print(f"{'step':>7} {'cells':>5} {'succ/eps':>10} {'mean %':>7} {'wilson95':>16}")
    for step, row in points.items():
        lo, hi = wilson(row["successes"], row["episodes"])
        mean = 100.0 * row["successes"] / row["episodes"] if row["episodes"] else float("nan")
        print(
            f"{step:>7} {len(row['tasks']):>5} {row['successes']:>4}/{row['episodes']:<5} {mean:>7.2f} [{100 * lo:6.2f}, {100 * hi:6.2f}]"
        )
    if tasks:
        print("\nper-task successes / 50 (columns = milestones)")
        steps = list(points)
        print(f"{'task':<24}" + "".join(f"{s:>8}" for s in steps))
        for task in tasks:
            cells_row = [points[s]["tasks"].get(task) for s in steps]
            print(f"{task:<24}" + "".join(f"{(c[0] if c else '-'):>8}" for c in cells_row))
    chosen = select_step(points, args.mde, args.tasks)
    complete_steps = [s for s, row in points.items() if len(row["tasks"]) >= args.tasks]
    print(f"\ncomplete milestones ({args.tasks} tasks): {complete_steps}")
    print(f"A19 selection (mde {args.mde} pp on the pooled base curve): s* = {chosen}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "campaign_dir": str(args.campaign_dir),
                    "cells": len(cells),
                    "points": {
                        str(s): {
                            "successes": row["successes"],
                            "episodes": row["episodes"],
                            "tasks": {t: list(v) for t, v in row["tasks"].items()},
                        }
                        for s, row in points.items()
                    },
                    "mde_pp": args.mde,
                    "selected_step": chosen,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
