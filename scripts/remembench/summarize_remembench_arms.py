#!/usr/bin/env python3
"""Render the combined ReMemBench arms x memory-categories table to summary.md."""

from __future__ import annotations

import argparse
import json
import os

# Display order + human labels. Keys are the study run ids.
ARMS = [
    ("s0-9e47bc75062b23e9", "s0 base", "baseline finetune, base serve"),
    ("s1-bff55c66cffc3360", "s1 tanh", "workspace read, tanh MLP"),
    ("s1-be5d198305786f3e", "s1 deltanet-w8", "workspace read, gated DeltaNet window=8"),
    ("s3-5e942af9f0718e3a", "s3 jw01k16", "JEPA aux (train-only), base serve"),
    ("s1-9b508f6799c3d128", "s1 GDN+JEPA", "gated DeltaNet window=8 + JEPA aux (train-only)"),
    ("s1-3b9f9229b3ea51b2", "s1 deltanet-w2", "workspace read, gated DeltaNet window=2"),
    ("s1-9b28670a6f0c57d9", "s1 deltanet-w16", "workspace read, gated DeltaNet window=16"),
    ("s1-8edacfb5b7739576", "s1 deltanet-w32", "workspace read, gated DeltaNet window=32"),
    ("s1-a781d6e251d1e87a", "s1 combo-k16", "gated DeltaNet + JEPA aux, k16"),
]
CATS = [
    ("spatial", "Spatial"),
    ("prospective", "Prospective"),
    ("object_associative", "Object-Assoc"),
    ("object_set", "Object-Set"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None, help="default <root>/summary.md")
    args = ap.parse_args()

    loaded = []
    for run_id, label, note in ARMS:
        path = os.path.join(args.root, run_id, "results.json")
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            loaded.append((run_id, label, note, json.load(handle)))
    if not loaded:
        print("no arm results found yet")
        return 0

    def pct(x):
        return f"{x * 100:.1f}%"

    lines = [
        "# ReMemBench held-out eval — pi0.5 long_context_v1 arms",
        "",
        "Box tier (nagababa, 4x RTX PRO 6000). Protocol: 88 held-out episodes across the 13 Mem\\*",
        "variants x3 rollouts = 264 rollouts/arm; resets pinned by ep_meta+seed (both required);",
        "per-task horizons from the manifest; `failed_task` (blown prospective deadline) is a hard",
        "failure. Rollouts of one episode share the pinned reset and differ only in pi diffusion",
        "noise. replan_steps=8. Overall = unweighted mean over the 4 category means, so the six",
        "spatial permutation variants do not triple-count.",
        "",
        f"Episode manifest sha256 `{loaded[0][3]['manifest_sha256'][:16]}...` (all arms agree).",
        "",
        "| Arm | " + " | ".join(label for _, label in CATS) + " | **Overall** | pooled |",
        "|---|" + "---|" * (len(CATS) + 2),
    ]
    for _run_id, label, _note, data in loaded:
        by_cat = data["by_category"]
        cells = []
        for key, _ in CATS:
            cells.append(pct(by_cat[key]["mean"]) if key in by_cat else "—")
        lines.append(
            f"| {label} | " + " | ".join(cells) + " | "
            f"**{pct(data['overall_category_weighted'])}** | "
            f"{pct(data['overall_rollout_pooled'])} |"
        )

    ref = loaded[0][3]["by_category"]
    lines += [
        "",
        "n per category (identical across arms — same sealed manifest):",
        "",
        "| Category | variants | episodes | rollouts |",
        "|---|---|---|---|",
    ]
    for key, label in CATS:
        if key in ref:
            block = ref[key]
            lines.append(
                f"| {label} | {block['n_tasks_expected']} | {block['num_episodes']} | {block['num_rollouts']} |"
            )
    total_eps = sum(ref[k]["num_episodes"] for k, _ in CATS if k in ref)
    total_roll = sum(ref[k]["num_rollouts"] for k, _ in CATS if k in ref)
    lines.append(f"| **total** | 13 | {total_eps} | {total_roll} |")

    lines += ["", "Arms:", ""]
    for run_id, label, note, _ in loaded:
        lines.append(f"- `{run_id}` — {label}: {note}")
    missing = [r for r, _, _, in_ in ((a, b, c, None) for a, b, c in ARMS) if r not in {x[0] for x in loaded}]
    if missing:
        lines += ["", "Not yet collected: " + ", ".join(f"`{m}`" for m in missing)]
    lines.append("")

    out_path = args.out or os.path.join(args.root, "summary.md")
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
