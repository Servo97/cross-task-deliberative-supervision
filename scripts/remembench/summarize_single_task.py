#!/usr/bin/env python3
"""Single-task ReMemBench sweep: 4 tasks x 6 arms table + the memory-vs-disambiguation read.

The read this sweep exists for: every other arm in the study is multi-task, so a memory win
could be (a) memory helping DISAMBIGUATE tasks inside a soup or (b) memory helping WITHIN a
task. Only (b) supports the representation claim. So for each task we compare
Delta(deltanet - base) measured SINGLE-task against the same Delta measured in the 13-task
soup (from the multi-task results.json per-task rates). Delta surviving single-task =>
memory-within-task; Delta collapsing => the multi-task win was task disambiguation.

Absolute rates are NOT comparable across tasks (uniform 4k steps means a 9-demo task is seen
far more often than a 40-demo one) — only the arm ORDERING within a task is.
"""

from __future__ import annotations

import argparse
import json
import os

TASKS = [
    ("MemRetrieveOilsFromCounterRL", "oilsRL", "spatial"),
    ("MemFruitInSinkRightFar", "fruitRF", "spatial"),
    ("MemFruitInSinkLeftFar", "fruitLF", "spatial"),
    ("MemHeatPot", "heatpot", "prospective"),
    ("MemPutKBreadInMicrowave", "kbread", "object_set"),
    ("MemWashAndReturnLeft", "washL", "object_assoc"),
]
ARMS = ["base", "tanh", "dnw2", "dnw8", "dnw16", "jw01k16", "combo-k16"]
# Multi-task (13-task soup) arms whose per-task rates form the comparison baseline.
MULTI = [
    ("base", "s0-9e47bc75062b23e9"),
    ("tanh", "s1-bff55c66cffc3360"),
    ("dnw8", "s1-be5d198305786f3e"),
    ("jw01k16", "s3-5e942af9f0718e3a"),
    ("gdn+jepa", "s1-9b508f6799c3d128"),
    ("dnw2", "s1-3b9f9229b3ea51b2"),
    ("dnw16", "s1-9b28670a6f0c57d9"),
    ("dnw32", "s1-8edacfb5b7739576"),
    ("combo-k16", "s1-a781d6e251d1e87a"),
]


def load_cells(root, matrix_path):
    cells = {}
    with open(matrix_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            _short, task, arm, run_id = line.rstrip("\n").split("\t")
            path = os.path.join(root, task, run_id, "results.json")
            if not os.path.isfile(path):
                continue
            data = json.load(open(path))
            block = data["by_task"].get(task)
            if block is None:
                continue
            cells[(task, arm)] = {
                "run_id": run_id,
                "rate": block["success_rate"],
                "n": block["num_rollouts"],
                "succ": block["num_success"],
            }
    return cells


def load_multi(multi_root):
    out = {}
    for arm, run_id in MULTI:
        path = os.path.join(multi_root, run_id, "results.json")
        if not os.path.isfile(path):
            continue
        data = json.load(open(path))
        for task, block in data["by_task"].items():
            out[(task, arm)] = {
                "rate": block["success_rate"],
                "n": block["num_rollouts"],
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="single_task results root")
    ap.add_argument("--multi-root", required=True, help="multi-task results root")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = load_cells(args.root, args.matrix)
    multi = load_multi(args.multi_root)
    if not cells:
        print("no single-task results found yet")
        return 0

    def pct(x):
        return f"{x * 100:.1f}%"

    lines = [
        "# ReMemBench single-task sweep — 4 tasks x 6 arms (box tier)",
        "",
        "Each cell is an independent single-task finetune (step-3999, uniform 4k steps) evaluated",
        "on the sealed held-out manifest FILTERED to that task, x3 diffusion-noise rollouts,",
        "replan_steps=8, stable per-worker env identity.",
        "",
        "**Absolute rates are not comparable across tasks** — uniform 4k steps means the 9-demo",
        "oilsRL cell sees its demos far more often than the 40-demo heatpot cell. Only the arm",
        "ORDERING within a task is meaningful, and the claim is that ordering repeating across",
        "the four independent tasks.",
        "",
        "| Task (category) | " + " | ".join(ARMS) + " | n/cell |",
        "|---|" + "---|" * (len(ARMS) + 1),
    ]
    for task, short, cat in TASKS:
        row, n_seen = [], None
        for arm in ARMS:
            cell = cells.get((task, arm))
            if cell is None:
                row.append("—")
            else:
                row.append(pct(cell["rate"]))
                n_seen = cell["n"]
        lines.append(f"| {short} ({cat}) | " + " | ".join(row) + f" | {n_seen if n_seen else '—'} |")

    # The read: does the deltanet-over-base gap survive when the soup is removed?
    lines += [
        "",
        "## The read: memory-within-task vs task-disambiguation",
        "",
        "Delta = deltanet-w8 minus base, measured single-task, against the same Delta in the",
        "13-task soup (per-task rates from the multi-task results).",
        "",
        "| Task | single base | single dnw8 | Delta single | soup base | soup dnw8 | Delta soup | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for task, short, _cat in TASKS:
        sb, sd = cells.get((task, "base")), cells.get((task, "dnw8"))
        mb, md = multi.get((task, "base")), multi.get((task, "dnw8"))
        if not (sb and sd):
            lines.append(f"| {short} | — | — | — | — | — | — | (incomplete) |")
            continue
        d_single = sd["rate"] - sb["rate"]
        cols = [short, pct(sb["rate"]), pct(sd["rate"]), f"{d_single * 100:+.1f}pt"]
        if mb and md:
            d_soup = md["rate"] - mb["rate"]
            # Saturation guard, checked BEFORE the survival logic. When the single-task cell
            # ceilings (base already at 100%), Delta_single is mechanically 0 and the naive
            # rule would score it "collapses" — but the gap closed because BASE improved, not
            # because memory stopped helping. That is a ceiling artifact of training 4k steps
            # on one task, and it is evidence about nothing. oilsRL does exactly this.
            if sb["rate"] >= 1.0 and sd["rate"] >= 1.0:
                verdict = "uninformative (saturated)"
            elif d_soup <= 0:
                verdict = "no soup gap"
            elif d_single >= 0.5 * d_soup:
                # Same sign and at least half the soup gap retained.
                verdict = "survives"
            elif d_single <= 0:
                verdict = "collapses"
            else:
                verdict = "partial"
            cols += [pct(mb["rate"]), pct(md["rate"]), f"{d_soup * 100:+.1f}pt", verdict]
            verdicts.append(verdict)
        else:
            cols += ["—", "—", "—", "(no soup ref)"]
        lines.append("| " + " | ".join(cols) + " |")

    if verdicts:
        n_surv = sum(1 for v in verdicts if v == "survives")
        n_coll = sum(1 for v in verdicts if v == "collapses")
        n_sat = sum(1 for v in verdicts if v.startswith("uninformative"))
        n_gap = sum(1 for v in verdicts if v == "no soup gap")
        lines += [
            "",
            f"Across {len(verdicts)} tasks: **{n_surv} survive, {n_coll} collapse**, "
            f"{n_sat} saturated/uninformative, {n_gap} with no soup gap to test.",
            "",
            "Only tasks with a real, unsaturated soup gap can answer the survival question.",
            "Per-cell n is 9-36 rollouts (~±15pt), so no single row is decisive.",
        ]

    # Sign test on the dnw8 > base streak across every informative (non-saturated) task.
    signs = []
    for task, short, _cat in TASKS:
        sb, sd = cells.get((task, "base")), cells.get((task, "dnw8"))
        if not (sb and sd) or (sb["rate"] >= 1.0 and sd["rate"] >= 1.0):
            continue
        signs.append((short, sd["rate"] - sb["rate"]))
    if signs:
        n_pos = sum(1 for _s, d in signs if d > 0)
        n_tie = sum(1 for _s, d in signs if d == 0)
        n_tot = len(signs)
        # Two-sided exact binomial p under H0: P(+) = 1/2, ignoring ties.
        eff = n_tot - n_tie
        from math import comb

        p = min(1.0, 2 * sum(comb(eff, k) for k in range(n_pos, eff + 1)) / (2**eff)) if eff else float("nan")
        lines += [
            "",
            "### Sign test: dnw8 > base, within-task (saturated tasks excluded)",
            "",
            "| task | Delta(dnw8 - base) |",
            "|---|---|",
        ]
        for short, delta in signs:
            lines.append(f"| {short} | {delta * 100:+.1f}pt |")
        lines += [
            "",
            f"**{n_pos}/{eff} positive** ({n_tie} tie{'s' if n_tie != 1 else ''} excluded), "
            f"two-sided exact binomial p = {p:.3f}.",
        ]

    lines += ["", "## Cell provenance", "", "| task | arm | run_id | rollouts | successes |", "|---|---|---|---|---|"]
    for task, short, _cat in TASKS:
        for arm in ARMS:
            cell = cells.get((task, arm))
            if cell:
                lines.append(f"| {short} | {arm} | `{cell['run_id']}` | {cell['n']} | {cell['succ']} |")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {args.out}  ({len(cells)}/24 cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
