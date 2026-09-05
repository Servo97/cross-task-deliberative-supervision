#!/usr/bin/env python3
"""A17 decision harness: score a max-effort (xhigh) pilot against the pre-registered rules.

Four subcommands, one per rule, all offline:

  regrade3  R1 - three-way blind re-grade of the 40 low/medium-disagreement pairs.
                 Ground truth is `a9_effort_regrade/mylabels.json`, which was committed during the
                 §15.3 round -- BEFORE the xhigh run existed. That ordering is what makes this blind
                 for max: the labels cannot have been anchored to verdicts that did not yet exist.
  probes    R2 - planted-CONTRAST recovery (F3) on the same 45 probes.
  pass1     R3 - side-by-side low-vs-xhigh descriptor sheet, 30 segments, for blind grading.
  costs     re-project the full redo from MEASURED xhigh throughput.

Nothing here fires a job or reads the network.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import Counter

DELIB = pathlib.Path("~/Research/TRI/wsm_data/deliberation").expanduser()
POSITIVE = {"EQUIVALENT", "ANALOGOUS"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def sign_test_one_sided(wins: int, losses: int) -> float:
    """P(X >= wins) under Binomial(wins+losses, 0.5): does max beat low on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def bucket_index(store_root: pathlib.Path) -> dict:
    """(anchor, candidate) -> verdict, from every bucket file under an edge store."""
    out = {}
    for f in store_root.rglob("*.bucket.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        anchor, cands = d.get("anchor"), d.get("candidates") or []
        for v in d.get("verdicts") or []:
            if not isinstance(v, dict):
                continue
            # The store keys each verdict by anchor_id/candidate_id and names the label `type`.
            # Zipping verdicts to candidates by position would silently mis-pair whenever a bucket
            # dropped or reordered one, so join on the ids the record carries.
            cid = v.get("candidate_id")
            if cid is None:
                i = v.get("candidate")
                cid = cands[i] if isinstance(i, int) and 0 <= i < len(cands) else None
            aid = v.get("anchor_id") or anchor
            if cid is not None:
                out[(aid, cid)] = v.get("type")
    return out


def cmd_regrade3(args) -> None:
    key = json.loads((DELIB / "a9_effort_regrade" / "key.json").read_text())
    mine = json.loads((DELIB / "a9_effort_regrade" / "mylabels.json").read_text())
    xh = bucket_index(pathlib.Path(args.xhigh_store).expanduser())

    rows, missing = [], 0
    tally = Counter()
    for p in key["pairs"]:
        truth = (mine.get(str(p["index"])) or {}).get("label")
        v = xh.get((p["anchor"], p["candidate"]))
        if v is None:
            missing += 1
        rows.append(
            {
                **{k: p[k] for k in ("index", "anchor", "candidate", "stratum", "low", "medium")},
                "xhigh": v,
                "truth": truth,
                "low_ok": p["low"] == truth,
                "medium_ok": p["medium"] == truth,
                "xhigh_ok": v == truth,
            }
        )
        tally["low"] += p["low"] == truth
        tally["medium"] += p["medium"] == truth
        tally["xhigh"] += v == truth
    n = len(rows)
    wins = sum(1 for r in rows if r["xhigh_ok"] and not r["low_ok"])
    losses = sum(1 for r in rows if r["low_ok"] and not r["xhigh_ok"])
    p_sign = sign_test_one_sided(wins, losses)
    delta = tally["xhigh"] - tally["low"]
    rule = {
        "xhigh_correct_rate": round(tally["xhigh"] / n, 4) if n else None,
        "rate_ge_0.60": tally["xhigh"] / n >= 0.60 if n else False,
        "delta_vs_low_pairs": delta,
        "delta_ge_10": delta >= 10,
        "discordant_wins_over_low": wins,
        "discordant_losses": losses,
        "sign_test_p_one_sided": round(p_sign, 5),
        "p_lt_0.05": p_sign < 0.05,
    }
    rule["R1_PASS"] = bool(rule["rate_ge_0.60"] and rule["delta_ge_10"] and rule["p_lt_0.05"])
    out = {
        "n": n,
        "missing_xhigh_verdicts": missing,
        "correct": dict(tally),
        "rates": {k: round(v / n, 4) for k, v in tally.items()} if n else {},
        "wilson95": {k: wilson(v, n) for k, v in tally.items()},
        "R1": rule,
        "rows": rows,
    }
    pathlib.Path(args.out).expanduser().write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n", "missing_xhigh_verdicts", "correct", "rates", "R1")}, indent=1))


def cmd_probes(args) -> None:
    prev = json.loads((DELIB / "qa_pass2_probe_recovery.json").read_text())
    probes = prev["probes"] if isinstance(prev.get("probes"), list) else json.loads(prev["probes"])
    xh = bucket_index(pathlib.Path(args.xhigh_store).expanduser())
    known = [p for p in probes if p.get("ground_truth") == "CONTRAST"]
    strict = loose = graded = 0
    rows = []
    for p in known:
        v = xh.get((p["anchor"], p["candidate"]))
        if v is None:
            rows.append({**{k: p.get(k) for k in ("probe_id", "family")}, "xhigh": None})
            continue
        graded += 1
        strict += v == "CONTRAST"
        loose += v in ("CONTRAST", "UNRELATED")
        rows.append({**{k: p.get(k) for k in ("probe_id", "family")}, "xhigh": v})
    rule = {
        "n_known_contrast": len(known),
        "n_graded": graded,
        "strict_recovery": round(strict / graded, 4) if graded else None,
        "loose_recovery": round(loose / graded, 4) if graded else None,
        "strict_wilson95": wilson(strict, graded),
        "loose_wilson95": wilson(loose, graded),
        "baseline_low": {"strict": 0.3556, "loose": 0.5333},
        "strict_ge_0.60": (strict / graded >= 0.60) if graded else False,
        "loose_ge_0.75": (loose / graded >= 0.75) if graded else False,
    }
    rule["R2_PASS"] = bool(rule["strict_ge_0.60"] and rule["loose_ge_0.75"])
    out = {"R2": rule, "by_family": {}, "rows": rows}
    fam = {}
    for p, r in zip(known, [r for r in rows if "xhigh" in r]):
        fam.setdefault(p.get("family"), Counter())[r["xhigh"]] += 1
    out["by_family"] = {k: dict(v) for k, v in fam.items()}
    pathlib.Path(args.out).expanduser().write_text(json.dumps(out, indent=2))
    print(json.dumps(out["R2"], indent=1))


def cmd_pass1(args) -> None:
    """Render a blind-gradeable low-vs-xhigh sheet. Sides are SHUFFLED per item and the mapping is
    written to a separate key file, so the grader cannot tell which column is max."""
    lo_root = pathlib.Path(args.low_descriptors).expanduser()
    hi_root = pathlib.Path(args.xhigh_descriptors).expanduser()
    rng = random.Random(args.seed)
    items = []
    for hi in sorted(hi_root.rglob("ep_*.descriptors.json")):
        lo = lo_root / hi.relative_to(hi_root)
        if not lo.is_file():
            continue
        H, L = json.loads(hi.read_text()), json.loads(lo.read_text())
        for a, b in zip(H.get("descriptors", []), L.get("descriptors", [])):
            if a.get("segment") != b.get("segment"):
                continue
            items.append(
                {
                    "task": H.get("task"),
                    "episode": H.get("episode_id"),
                    "segment": a.get("segment"),
                    "xhigh": a.get("descriptor"),
                    "low": b.get("descriptor"),
                }
            )
    rng.shuffle(items)
    items = items[: args.n]
    sheet, key = [], []
    for i, it in enumerate(items):
        flip = rng.random() < 0.5
        A, B = (it["low"], it["xhigh"]) if flip else (it["xhigh"], it["low"])
        key.append(
            {
                "index": i,
                "A_is": "low" if flip else "xhigh",
                "task": it["task"],
                "episode": it["episode"],
                "segment": it["segment"],
            }
        )
        sheet.append(
            {
                "index": i,
                "task": it["task"],
                "episode": it["episode"],
                "segment": it["segment"],
                "A": A,
                "B": B,
                "grade": {
                    "states_completion_condition": {"A": None, "B": None},
                    "states_bound_variable": {"A": None, "B": None},
                    "note": "",
                },
            }
        )
    outdir = pathlib.Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sheet.json").write_text(json.dumps(sheet, indent=2))
    (outdir / "key.json").write_text(json.dumps(key, indent=2))
    tok = {}
    for name, root in (("xhigh", hi_root), ("low", lo_root)):
        n = t = trunc = 0
        for f in root.rglob("ep_*.descriptors.json"):
            for d in json.loads(f.read_text()).get("descriptors", []):
                u = d.get("usage") or {}
                n += 1
                t += int(u.get("completion_tokens") or 0)
                trunc += u.get("finish_reason") == "length"
        tok[name] = {
            "segments": n,
            "mean_completion_tokens": round(t / max(n, 1), 1),
            "truncated": trunc,
            "truncation_rate": round(trunc / max(n, 1), 4),
        }
    (outdir / "tokens.json").write_text(json.dumps(tok, indent=2))
    print(
        json.dumps(
            {
                "sheet_items": len(sheet),
                "out": str(outdir),
                "tokens": tok,
                "R3_rule": ">=50% of graded items must show xhigh newly stating a completion "
                "condition or bound variable the low descriptor omitted",
            },
            indent=1,
        )
    )


def cmd_costs(args) -> None:
    SEG = ANCH = args.corpus
    rows = []
    for nodes in (1, 2):
        p1 = SEG / args.pass1_seg_per_min_per_gpu / 8 / 60 / nodes if args.pass1_seg_per_min_per_gpu else None
        p2 = ANCH / args.pass2_anchors_per_min_per_gpu / 8 / 60 / nodes
        rows.append(
            {
                "nodes": nodes,
                "pass1_h": round(p1, 1) if p1 else None,
                "pass2_h": round(p2, 1),
                "total_h": round((p1 or 0) + p2, 1),
                "days": round(((p1 or 0) + p2) / 24, 2),
            }
        )
    out = {
        "corpus": SEG,
        "measured": {
            "pass2_anchors_per_min_per_gpu": args.pass2_anchors_per_min_per_gpu,
            "pass1_seg_per_min_per_gpu": args.pass1_seg_per_min_per_gpu,
            "tokens_out_per_anchor": args.tokens_out_per_anchor,
            "truncation_rate": args.truncation_rate,
        },
        "projection_p5_H100": rows,
        "note": "measured on H100 at xhigh; replaces the 4/5/8x envelope in §45.6",
    }
    print(json.dumps(out, indent=1))
    if args.out:
        pathlib.Path(args.out).expanduser().write_text(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("regrade3")
    a.set_defaults(fn=cmd_regrade3)
    a.add_argument("--xhigh-store", required=True, help="edges/<825e4c29…> dir of the PILOT-2 output")
    a.add_argument("--out", default=str(DELIB / "a17_regrade3.json"))

    b = sub.add_parser("probes")
    b.set_defaults(fn=cmd_probes)
    b.add_argument("--xhigh-store", required=True)
    b.add_argument("--out", default=str(DELIB / "a17_probes_xhigh.json"))

    c = sub.add_parser("pass1")
    c.set_defaults(fn=cmd_pass1)
    c.add_argument("--low-descriptors", default=str(DELIB / "pass1_store/robocasa"))
    c.add_argument("--xhigh-descriptors", required=True)
    c.add_argument("--n", type=int, default=30)
    c.add_argument("--seed", type=int, default=20260902)
    c.add_argument("--out", default=str(DELIB / "a17_pass1_sidebyside"))

    d = sub.add_parser("costs")
    d.set_defaults(fn=cmd_costs)
    d.add_argument("--corpus", type=int, default=28722)
    d.add_argument("--pass2-anchors-per-min-per-gpu", type=float, required=True)
    d.add_argument("--pass1-seg-per-min-per-gpu", type=float, default=0.0)
    d.add_argument("--tokens-out-per-anchor", type=float, default=0.0)
    d.add_argument("--truncation-rate", type=float, default=0.0)
    d.add_argument("--out", default=str(DELIB / "a17_cost_reprojection.json"))

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
