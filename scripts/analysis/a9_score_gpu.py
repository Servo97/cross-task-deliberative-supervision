"""A9 scorers for the two GPU runs: planted-probe recovery (F3) and low-vs-medium effort (F4).

python scripts/analysis/a9_score_gpu.py --stage probes
python scripts/analysis/a9_score_gpu.py --stage effort
python scripts/analysis/a9_score_gpu.py --stage effort-blind   # render disagreeing pairs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.a9_sample_edges import render_pair  # noqa: E402

LIVE = "~/Research/TRI/wsm_data/deliberation/pass2_store"
LIVE_ESID = "fb22b06bb8e74fc849be75e8a3f619908031767723e7e48a5b87ca4095371815"
POS = {"EQUIVALENT", "ANALOGOUS"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_verdicts(store: Path, esid: str | None = None) -> dict:
    edges = store / "edges"
    root = edges / esid if esid else next(p for p in sorted(edges.iterdir()) if p.is_dir())
    out = {}
    for p in (root / "buckets").rglob("*.bucket.json"):
        d = json.loads(p.read_text())
        for v in d["verdicts"]:
            out[frozenset((v["anchor_id"], v["candidate_id"]))] = v
    return out, root


def stage_probes(args) -> None:
    store = Path(args.probe_store).expanduser()
    probes = json.loads((store / "probes.json").read_text())["probes"]
    verd, root = load_verdicts(store)
    rows, missing = [], 0
    for p in probes:
        v = verd.get(frozenset((p["anchor"], p["candidate"])))
        if v is None:
            missing += 1
            continue
        rows.append({**p, "verdict": v["type"], "confidence": v["confidence"], "rationale": v["rationale"]})

    con = [r for r in rows if r["ground_truth"] == "CONTRAST"]
    pos = [r for r in rows if r["ground_truth"] == "EQUIVALENT"]
    rec = sum(1 for r in con if r["verdict"] in ("CONTRAST", "UNRELATED"))
    strict = sum(1 for r in con if r["verdict"] == "CONTRAST")
    eqfail = sum(1 for r in con if r["verdict"] == "EQUIVALENT")
    posok = sum(1 for r in pos if r["verdict"] in POS)

    by_fam = {}
    for f in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == f]
        gt = sel[0]["ground_truth"]
        ok = (
            sum(1 for r in sel if (r["verdict"] in ("CONTRAST", "UNRELATED")) if gt == "CONTRAST")
            if gt == "CONTRAST"
            else sum(1 for r in sel if r["verdict"] in POS)
        )
        by_fam[f] = {
            "ground_truth": gt,
            "n": len(sel),
            "ok": ok,
            "verdicts": {
                t: sum(1 for r in sel if r["verdict"] == t)
                for t in ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")
            },
        }

    out = {
        "edge_store_id": root.name,
        "n_probes_judged": len(rows),
        "n_missing": missing,
        "known_CONTRAST": {
            "n": len(con),
            "recovery_CONTRAST_or_UNRELATED": round(rec / max(len(con), 1), 4),
            "wilson95": [round(x, 4) for x in wilson(rec, len(con))],
            "strict_CONTRAST_only": round(strict / max(len(con), 1), 4),
            "wilson95_strict": [round(x, 4) for x in wilson(strict, len(con))],
            "hard_failure_EQUIVALENT": eqfail,
            "verdict_histogram": {
                t: sum(1 for r in con if r["verdict"] == t)
                for t in ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")
            },
        },
        "sanity_positives": {
            "n": len(pos),
            "kept_positive": posok,
            "rate": round(posok / max(len(pos), 1), 4),
            "verdict_histogram": {
                t: sum(1 for r in pos if r["verdict"] == t)
                for t in ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")
            },
        },
        "by_family": by_fam,
        "probes": rows,
    }
    Path(args.out_probes).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "probes"}, indent=1))


def _kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _paired(args):
    live, _ = load_verdicts(Path(args.live_store).expanduser(), LIVE_ESID)
    med, root = load_verdicts(Path(args.medium_store).expanduser())
    pairs = []
    for key, mv in med.items():
        lv = live.get(key)
        if lv is None:
            continue
        pairs.append(
            {
                "anchor": mv["anchor_id"],
                "candidate": mv["candidate_id"],
                "stratum": mv["stratum"],
                "low": lv["type"],
                "medium": mv["type"],
                "low_conf": lv["confidence"],
                "medium_conf": mv["confidence"],
                "low_rationale": lv["rationale"],
                "medium_rationale": mv["rationale"],
            }
        )
    return pairs, root


def stage_effort(args) -> None:
    pairs, root = _paired(args)
    n = len(pairs)
    four = sum(1 for p in pairs if p["low"] == p["medium"])
    lb = [1 if p["low"] in POS else 0 for p in pairs]
    mb = [1 if p["medium"] in POS else 0 for p in pairs]
    binagree = sum(1 for x, y in zip(lb, mb) if x == y)
    k = _kappa(lb, mb)
    conf = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        conf[p["low"]][p["medium"]] += 1
    T = ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")
    out = {
        "medium_edge_store_id": root.name,
        "n_paired_verdicts": n,
        "n_buckets_medium": len({p["anchor"] for p in pairs}),
        "four_way_agreement": round(four / max(n, 1), 4),
        "wilson95_four_way": [round(x, 4) for x in wilson(four, n)],
        "binary_agreement": round(binagree / max(n, 1), 4),
        "wilson95_binary": [round(x, 4) for x in wilson(binagree, n)],
        "cohens_kappa_binary": round(k, 4),
        "positive_rate": {"low": round(sum(lb) / max(n, 1), 4), "medium": round(sum(mb) / max(n, 1), 4)},
        "type_rates": {
            e: {t: round(sum(1 for p in pairs if p[e] == t) / max(n, 1), 4) for t in T} for e in ("low", "medium")
        },
        "confusion_low_rows_x_medium_cols": {a: {b: conf[a][b] for b in T} for a in T},
        "by_stratum": {
            s: {
                "n": sum(1 for p in pairs if p["stratum"] == s),
                "four_way": round(
                    sum(1 for p in pairs if p["stratum"] == s and p["low"] == p["medium"])
                    / max(sum(1 for p in pairs if p["stratum"] == s), 1),
                    4,
                ),
                "low_EQUIV": round(
                    sum(1 for p in pairs if p["stratum"] == s and p["low"] == "EQUIVALENT")
                    / max(sum(1 for p in pairs if p["stratum"] == s), 1),
                    4,
                ),
                "medium_EQUIV": round(
                    sum(1 for p in pairs if p["stratum"] == s and p["medium"] == "EQUIVALENT")
                    / max(sum(1 for p in pairs if p["stratum"] == s), 1),
                    4,
                ),
            }
            for s in sorted({p["stratum"] for p in pairs})
        },
        "pairs": pairs,
    }
    Path(args.out_effort).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "pairs"}, indent=1))


def stage_effort_blind(args) -> None:
    """Render N disagreeing paired edges for the same blind adjudication as task 1."""
    pairs, _ = _paired(args)
    dis = [p for p in pairs if p["low"] != p["medium"]]
    rng = random.Random(args.seed)
    rng.shuffle(dis)
    dis = dis[: args.n_blind]
    rows = {}
    for line in (Path(args.live_store).expanduser() / "index" / "segments.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["seg_id"]] = r
    out = Path(args.sheet).expanduser()
    (out / "blind").mkdir(parents=True, exist_ok=True)
    txt, key = [], []
    for i, p in enumerate(dis):
        txt.append(f"=== DPAIR {i} ===")
        txt.append(
            render_pair(
                rows[p["anchor"]]["descriptor"],
                rows[p["candidate"]]["descriptor"],
                p["anchor"],
                p["candidate"],
                p["stratum"],
            )
        )
        txt.append("")
        key.append({"index": i, **p})
    (out / "blind" / "disagreements.txt").write_text("\n".join(txt))
    (out / "key.json").write_text(json.dumps({"seed": args.seed, "n": len(key), "pairs": key}, indent=1))
    print(
        json.dumps(
            {
                "n_disagreeing_total": len([p for p in pairs if p["low"] != p["medium"]]),
                "n_rendered": len(key),
                "out": str(out),
            },
            indent=1,
        )
    )


def stage_effort_score(args) -> None:
    out = Path(args.sheet).expanduser()
    key = json.loads((out / "key.json").read_text())
    mine = json.loads((out / "mylabels.json").read_text())
    rows = []
    for p in key["pairs"]:
        m = mine[str(p["index"])]
        rows.append(
            {
                **p,
                "adjudicator": m["label"],
                "adjudicator_rationale": m["why"],
                "low_right": m["label"] == p["low"],
                "medium_right": m["label"] == p["medium"],
            }
        )
    graded = [r for r in rows if r["adjudicator"] != "AMBIGUOUS"]
    lo_r = sum(1 for r in graded if r["low_right"])
    me_r = sum(1 for r in graded if r["medium_right"])
    res = {
        "n_rendered": len(rows),
        "n_graded": len(graded),
        "low_correct": lo_r,
        "medium_correct": me_r,
        "neither_correct": sum(1 for r in graded if not r["low_right"] and not r["medium_right"]),
        "low_correct_rate": round(lo_r / max(len(graded), 1), 4),
        "medium_correct_rate": round(me_r / max(len(graded), 1), 4),
        "wilson95_low": [round(x, 4) for x in wilson(lo_r, len(graded))],
        "wilson95_medium": [round(x, 4) for x in wilson(me_r, len(graded))],
        "rows": rows,
    }
    (out / "regrade_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("probes", "effort", "effort-blind", "effort-score"))
    ap.add_argument("--live-store", default=LIVE)
    ap.add_argument("--probe-store", default="~/Research/TRI/wsm_data/deliberation/a9_probe_store")
    ap.add_argument("--medium-store", default="~/Research/TRI/wsm_data/deliberation/a9_medium_store")
    ap.add_argument("--sheet", default="~/Research/TRI/wsm_data/deliberation/a9_effort_regrade")
    ap.add_argument("--out-probes", default="~/Research/TRI/wsm_data/deliberation/qa_pass2_probe_recovery.json")
    ap.add_argument("--out-effort", default="~/Research/TRI/wsm_data/deliberation/qa_pass2_effort_ab.json")
    ap.add_argument("--n-blind", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    {
        "probes": stage_probes,
        "effort": stage_effort,
        "effort-blind": stage_effort_blind,
        "effort-score": stage_effort_score,
    }[args.stage](args)


if __name__ == "__main__":
    main()
