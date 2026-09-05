#!/usr/bin/env python3
"""H14 Stage-E — collapse every cell's gates.json into the A4 attribution table.

Ranking is on the pre-registered go/no-go (amendment A2): the FRAME-LEVEL cross-task retrieval lift
on the A1d disagreement subset. `between_episode_variance_fraction` is printed but never ranked on —
the canary showed the λ_del=0 control scores HIGHER on it, so selecting on it would pick the control.

    python scripts/deliberation/summarize_stage_e.py --runs ~/Research/TRI/wsm_data/deliberation/stage_e_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.runs).expanduser().glob("*/gates.json")):
        gates = json.loads(path.read_text())
        # Cells that ran before `contrast_weight` was recorded in gates.json still carry it in
        # run_config.json; backfill rather than print None and let a reader guess.
        if gates.get("edges", {}).get("contrast_weight") is None:
            config = json.loads((path.parent / "run_config.json").read_text())
            weight = float(config.get("contrast_weight", float("nan")))
            gates["edges"]["contrast_weight"] = weight
            gates["edges"]["consumed_contrast_as_hard_negative"] = bool(
                gates["edges"].get("n_contrast", 0) > 0 and weight > 1.0 and float(config.get("lambda_del", 0)) > 0
            )
        final = gates.get("final") or {}
        domains = final.get("g1b_per_domain") or {}
        primary = next(iter(domains.values()), {})
        retrieval = final.get("retrieval_gate") or {}
        train = final.get("train") or {}
        rows.append(
            {
                "cell": gates["cell"],
                "encoder_id": gates["encoder_id"],
                "edges": gates["edges"]["set"],
                "n_pos": gates["edges"]["n_positive"],
                "n_contrast": gates["edges"].get("n_contrast"),
                "contrast_w": gates["edges"].get("contrast_weight"),
                "hard_neg": gates["edges"].get("consumed_contrast_as_hard_negative"),
                "g1b": (primary.get("g1b") or {}).get("verdict"),
                "coh": round(primary.get("temporal_coherence_gap", float("nan")), 3),
                "erank": round(primary.get("effective_rank", float("nan")), 2),
                "bevf": round(primary.get("between_episode_variance_fraction", float("nan")), 3),
                "d_bevf": (gates.get("delta_vs_untrained") or {})
                .get(next(iter(domains), ""), {})
                .get("between_episode_variance_fraction"),
                "retr_top1": retrieval.get("top1"),
                "retr_chance": retrieval.get("chance"),
                "retr_lift": retrieval.get("lift"),
                "retr_wilson_lo": (retrieval.get("wilson95") or [None])[0],
                "beats_chance": retrieval.get("beats_chance"),
                "del_lift": round((train.get("del_discriminative") or {}).get("lift", float("nan")), 2),
                "decode_lift": (gates.get("decode_grounding") or {}).get("lift"),
                "collapse_ctrl_fails": gates.get("collapse_control_trips_fail"),
                "steps": final.get("step"),
                "minutes": final.get("minutes"),
            }
        )
    rows.sort(key=lambda r: -(r["retr_lift"] or 0))
    header = [
        "cell",
        "g1b",
        "coh",
        "erank",
        "bevf",
        "d_bevf",
        "retr_top1",
        "retr_chance",
        "retr_lift",
        "retr_wilson_lo",
        "beats_chance",
        "del_lift",
        "decode_lift",
        "n_contrast",
        "contrast_w",
        "hard_neg",
        "minutes",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(r.get(h)) for h in header) + " |")
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(rows, indent=1))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
