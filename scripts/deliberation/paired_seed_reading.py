#!/usr/bin/env python3
"""H14 Stage-E — the A14 PAIRED-BY-SEED reading of the primary and secondary contrasts.

Pre-registered before the replication ran (plan A14):

  primary    delta(E1b - ctrl-Eb)        paired by seed, on the retrieval-lift WILSON LOWER BOUND
  secondary  delta(E1b-analog05 - E1b)   likewise

The statistic is the Wilson-95 LOWER BOUND of top-1 on the A1d disagreement subset, not the point
lift: the lower bound already carries the anchor count, so a cell that retrieves well on a thin
subset cannot outrank one that retrieves nearly as well on a thick one.

The criterion is SIGN AGREEMENT across the three seeds plus the mean, and the per-arm seed SD is
printed next to it so the n behind any MDE is stated rather than implied. n=3 does not license a
p-value here and none is printed.

    python scripts/deliberation/paired_seed_reading.py \
        --runs ~/Research/TRI/wsm_data/deliberation/stage_e_runs \
        --arms E1b,ctrl-Eb,E1b-analog05 --seeds 20260828,20260829,20260830
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def collect(runs: Path, label_id: str = "") -> dict:
    """(cell, seed) -> the gate row.

    `label_id` is not cosmetic: `E1b` exists BOTH as the sealed v2 funnel cell (label
    `ab38d9efc0c649a3`) and as the seed-replication cell (label `adc1c7575dd70fa3` = v2 + the
    ctrl-Eb edge file, `edges_E1b.npz` byte-identical). Without the filter the two collide on
    (cell, seed) and which one survives depends on directory sort order.
    """
    out = {}
    for gates_path in sorted(runs.glob("*/gates.json")):
        gates = json.loads(gates_path.read_text())
        config = json.loads((gates_path.parent / "run_config.json").read_text())
        if label_id and gates.get("label_id") != label_id:
            continue
        final = gates.get("final") or {}
        domains = final.get("g1b_per_domain") or {}
        primary = next(iter(domains.values()), {})
        retrieval = final.get("retrieval_gate") or {}
        wilson = retrieval.get("wilson95") or [None, None]
        key = (gates["cell"], int(config["seed"]))
        out[key] = {
            "cell": gates["cell"],
            "seed": int(config["seed"]),
            "encoder_id": gates["encoder_id"],
            "label_id": gates["label_id"],
            "run_dir": gates_path.parent.name,
            "n_pos": gates["edges"]["n_positive"],
            "n_neg": gates["edges"].get("n_contrast"),
            "retr_top1": retrieval.get("top1"),
            "chance": retrieval.get("chance"),
            "lift": retrieval.get("lift"),
            "wilson_lo": wilson[0],
            "beats_chance": retrieval.get("beats_chance"),
            "coh": primary.get("temporal_coherence_gap"),
            "erank": primary.get("effective_rank"),
            "bevf": primary.get("between_episode_variance_fraction"),
            "g1b": (primary.get("g1b") or {}).get("verdict"),
            "decode": (gates.get("decode_grounding") or {}).get("lift"),
            "del_lift": ((final.get("train") or {}).get("del_discriminative") or {}).get("lift"),
            "minutes": final.get("minutes"),
        }
    return out


def paired(rows: dict, a: str, b: str, seeds: list, field: str = "wilson_lo") -> dict:
    deltas, missing = [], []
    for s in seeds:
        if (a, s) in rows and (b, s) in rows:
            deltas.append((s, rows[(a, s)][field] - rows[(b, s)][field]))
        else:
            missing.append(s)
    values = [d for _, d in deltas]
    signs = {d > 0 for d in values}
    # MDE sizing, stated so the null is falsifiable rather than merely reported. For the PAIRED
    # design the relevant dispersion is the SD of the per-seed DIFFERENCES, not either arm's own
    # seed SD (the arms are correlated through the seed, which is the whole point of pairing).
    # n per arm for 80% power, two-sided alpha .05, to detect an effect the size of the observed
    # mean delta:   n = (z_.975 + z_.80)^2 * sd_delta^2 / delta^2 = 7.849 * (sd/delta)^2.
    # The normal approximation is optimistic at these n; it is a floor on the seeds needed.
    sd_delta = statistics.stdev(values) if len(values) > 1 else None
    mean_delta = statistics.fmean(values) if values else None
    n_for_mde = None
    if sd_delta is not None and mean_delta not in (None, 0.0):
        n_for_mde = math.ceil(7.849 * (sd_delta / abs(mean_delta)) ** 2)
    return {
        "contrast": f"{a} - {b}",
        "field": field,
        "per_seed": [{"seed": s, "delta": round(d, 5)} for s, d in deltas],
        "n_pairs": len(values),
        "missing_seeds": missing,
        "mean": round(statistics.fmean(values), 5) if values else None,
        "all_same_sign": len(signs) == 1 if values else None,
        "sign": (
            "positive"
            if values and all(d > 0 for d in values)
            else "negative"
            if values and all(d < 0 for d in values)
            else "mixed"
        ),
        "sd_a": round(statistics.stdev([rows[(a, s)][field] for s in seeds if (a, s) in rows]), 5)
        if sum((a, s) in rows for s in seeds) > 1
        else None,
        "sd_b": round(statistics.stdev([rows[(b, s)][field] for s in seeds if (b, s) in rows]), 5)
        if sum((b, s) in rows for s in seeds) > 1
        else None,
        "sd_delta": round(sd_delta, 5) if sd_delta is not None else None,
        "n_per_arm_for_mde_at_80pct_power": n_for_mde,
        "mde_note": (
            "n per arm to detect an effect the size of the observed mean delta at 80% "
            "power, two-sided alpha .05, paired: 7.849*(sd_delta/mean_delta)^2"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arms", default="E1b,ctrl-Eb,E1b-analog05")
    ap.add_argument("--seeds", default="20260828,20260829,20260830")
    ap.add_argument(
        "--label-id", default="", help="only read cells trained on this label artifact (disambiguates E1b)"
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    rows = collect(Path(args.runs).expanduser(), args.label_id)

    header = [
        "cell",
        "seed",
        "n_pos",
        "n_neg",
        "retr_top1",
        "lift",
        "wilson_lo",
        "beats_chance",
        "coh",
        "erank",
        "bevf",
        "decode",
        "del_lift",
        "g1b",
        "minutes",
        "encoder_id",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    fmt = {
        "retr_top1": 4,
        "lift": 2,
        "wilson_lo": 4,
        "coh": 3,
        "erank": 2,
        "bevf": 3,
        "decode": 2,
        "del_lift": 2,
        "minutes": 1,
    }
    for arm in arms:
        for seed in seeds:
            r = rows.get((arm, seed))
            if not r:
                continue
            cells = [(round(r[h], fmt[h]) if h in fmt and r[h] is not None else r[h]) for h in header]
            print("| " + " | ".join(str(c) for c in cells) + " |")

    readings = {}
    if len(arms) >= 2:
        readings["primary"] = paired(rows, arms[0], arms[1], seeds)
    if len(arms) >= 3:
        readings["secondary"] = paired(rows, arms[2], arms[0], seeds)
    for arm in arms:
        vals = [rows[(arm, s)]["wilson_lo"] for s in seeds if (arm, s) in rows]
        lifts = [rows[(arm, s)]["lift"] for s in seeds if (arm, s) in rows]
        readings.setdefault("per_arm", {})[arm] = {
            "n_seeds": len(vals),
            "wilson_lo_mean": round(statistics.fmean(vals), 5) if vals else None,
            "wilson_lo_sd": round(statistics.stdev(vals), 5) if len(vals) > 1 else None,
            "lift_mean": round(statistics.fmean(lifts), 3) if lifts else None,
            "lift_sd": round(statistics.stdev(lifts), 3) if len(lifts) > 1 else None,
        }
    print()
    print(json.dumps(readings, indent=1))
    if args.out:
        Path(args.out).expanduser().write_text(
            json.dumps({"rows": [rows[k] for k in sorted(rows)], "readings": readings}, indent=1)
        )
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
