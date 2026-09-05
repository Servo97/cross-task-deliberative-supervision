"""H14 — regenerate `pilot_measurements.json` from the stores, so A6's timeout derivation is always
recomputed from evidence rather than typed in by hand.

`launch_deliberation.py` refuses to size `--max-run-seconds` without this file, and refuses when a
stage's rate is null. This script is the only thing that should ever write it.

Sources, all measured:
  corpus.robocasa_mem16_segments   the caption store (16 x 150 episodes), counted
  corpus.robomme16_segments        robomme_subgoal_audit.json, mean seg/ep x 1600
  pass1.*                          the descriptor store's per-request usage records + wall time
  pass2.*                          the judge stage's _provenance summaries

  python scripts/deliberation/refresh_measurements.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# A10 (2026-08-22): re-ranked on the sealed per-task anchors. WashLettuce/RinseSinkBasin are
# ceilinged (base 85/77), GetToastedBread is floored (base 0) -- all three are out of the corpus.
HEADLINE = (
    "ScrubCuttingBoard",
    "KettleBoiling",
    "SearingMeat",
    "GatherTableware",
    "PanTransfer",
    "HeatKebabSandwich",
    "StirVegetables",
    "RecycleBottlesByType",
    "CategorizeCondiments",
)
ANNEX = ("PackIdenticalLunches", "CuttingToolSelection", "PortionHotDogs", "SeparateFreezerRack")
MEM16 = HEADLINE + ANNEX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="~/Research/TRI/wsm_data/wsm_labels_captions")
    ap.add_argument("--descriptors", default="~/Research/TRI/wsm_data/deliberation/descriptors/robocasa")
    ap.add_argument("--robomme-audit", default="~/Research/TRI/wsm_data/deliberation/robomme_subgoal_audit.json")
    ap.add_argument("--edges", default="~/Research/TRI/wsm_data/deliberation/pass2_pilot/edges")
    ap.add_argument("--pass2-max-tokens", type=int, default=12288)
    ap.add_argument(
        "--remembench-segments",
        type=int,
        default=1260,
        help="aug_06 pilot basis; no local rmb keyframe store to count",
    )
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/pilot_measurements.json")
    args = ap.parse_args()

    # ---- corpus ------------------------------------------------------------------------------
    cap = Path(args.captions).expanduser()
    mem16, eps = 0, 0
    per_task = {}
    for t in MEM16:
        n = 0
        files = sorted((cap / t).glob("ep_*.captions.json"))
        for p in files:
            n += len(json.loads(p.read_text())["segments"])
        per_task[t] = {"episodes": len(files), "segments": n, "seg_per_ep": round(n / max(len(files), 1), 2)}
        mem16 += n
        eps += len(files)

    robomme = 0
    ra = Path(args.robomme_audit).expanduser()
    if ra.is_file():
        a = json.loads(ra.read_text())
        robomme = int(a["overall"]["simple_subgoal"]["extrapolated_segments_1600_eps"])

    total = mem16 + args.remembench_segments + robomme

    # ---- pass 1 ------------------------------------------------------------------------------
    pass1: dict = {"segments_per_min_per_gpu": None}
    dr = Path(args.descriptors).expanduser()
    prov = sorted((dr / "_provenance").glob("usage_shard*.json")) if dr.is_dir() else []
    if prov:
        s = json.loads(prov[-1].read_text())["summary"]
        pass1 = {
            "segments_per_min_per_gpu": s["segments_per_min"],
            "prompt_tokens_per_segment": s["prompt_tokens_per_segment"],
            "completion_tokens_per_segment": s["completion_tokens_per_segment"],
            "out_tok_per_s": s["out_tok_per_s"],
            "truncated_requests": s["truncated_requests"],
            "n_requests": s["n_requests"],
            "truncation_rate": round(s["truncated_requests"] / max(s["n_requests"], 1), 5),
            "concurrency": s["concurrency"],
            "reasoning_effort": s["reasoning_effort"],
            "model": s["model"],
            "source": str(prov[-1]),
        }

    # ---- pass 2 ------------------------------------------------------------------------------
    pass2: dict = {"anchors_per_min_per_gpu": None, "_pending": "run the judge pilot first"}
    ed = Path(args.edges).expanduser()
    sums = sorted(ed.rglob("_provenance/judge_shard*.json")) if ed.is_dir() else []
    if sums:
        # Only runs at the cap we intend to SHIP, and only runs that lost no buckets. A run whose
        # buckets truncated is a measurement of the wrong configuration -- averaging it in would
        # bake the 4,096-token failure (14/20 buckets lost) into the P1 timeout.
        rates, tin, tout, trunc, oks, used = [], [], [], 0, 0, []
        for p in sums:
            s = json.loads(p.read_text())["summary"]
            if not s["buckets_ok"]:
                continue
            if args.pass2_max_tokens and s.get("max_tokens") != args.pass2_max_tokens:
                continue
            if s.get("buckets_failed"):
                continue
            rates.append(s["anchors_per_min"])
            tin.append(s["tokens_in_per_anchor"])
            tout.append(s["tokens_out_per_anchor"])
            trunc += s["truncated"]
            oks += s["buckets_ok"]
            used.append(str(p))
        if rates:
            pass2 = {
                "anchors_per_min_per_gpu": round(statistics.mean(rates), 3),
                "tokens_in_per_anchor": round(statistics.mean(tin), 1),
                "tokens_out_per_anchor": round(statistics.mean(tout), 1),
                "max_tokens": args.pass2_max_tokens,
                "buckets_ok": oks,
                "truncated": trunc,
                "truncation_rate": round(trunc / max(oks, 1), 5),
                "n_shards": len(rates),
                "source": used[-3:],
            }

    m = {
        "_generated_by": "scripts/deliberation/refresh_measurements.py",
        "_warning": "Every rate here is MEASURED on the hardware named in it. Re-measure before "
        "sizing a job on different hardware; launch_deliberation.py multiplies by 2.5 "
        "for headroom but cannot correct a wrong platform.",
        "corpus": {
            "robocasa_mem13_a10_segments": mem16,
            "robocasa_mem13_a10_episodes": eps,
            "headline_tasks": list(HEADLINE),
            "annex_tasks": list(ANNEX),
            "remembench13_segments": args.remembench_segments,
            "robomme16_segments": robomme,
            "total_segments": total,
            "total_anchors": total,
            "robocasa_per_task": per_task,
        },
        "pass1": pass1,
        "pass2": pass2,
        "embed": {
            "estimated_seconds": 1800,
            "basis": "Qwen3-Embedding-0.6B measured at 307 texts / 31.8 s on CPU "
            "=> ~37 min for 21k on CPU, minutes on one GPU",
        },
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1))
    print(
        json.dumps(
            {
                "corpus": {
                    k: v
                    for k, v in m["corpus"].items()
                    if k not in ("robocasa_per_task", "headline_tasks", "annex_tasks")
                },
                "pass1": pass1,
                "pass2": pass2,
            },
            indent=1,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
