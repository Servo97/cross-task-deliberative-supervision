"""A9 accuracy sheet scorer: adjudicator labels (committed blind) vs the live Qwen verdicts.

Precision is computed with the Qwen verdict as the "prediction" and the adjudicator label as the
reference, per class, with Wilson 95% bounds. AMBIGUOUS adjudications are excluded from the
denominators and reported separately. A lenient variant collapses EQUIVALENT/ANALOGOUS (both are
SupCon positives) to show how much of any shortfall is the EQ/AN boundary rather than a real error.

  python scripts/analysis/a9_score_sheet.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TYPES = ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="~/Research/TRI/wsm_data/deliberation/a9_sheet")
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/qa_pass2_accuracy_sheet.json")
    args = ap.parse_args()

    sheet = Path(args.sheet).expanduser()
    key = json.loads((sheet / "key.json").read_text())
    mine = {}
    for p in sorted((sheet / "mylabels").glob("chunk*.json")):
        mine.update(json.loads(p.read_text()))

    rows = []
    for e in key["edges"]:
        m = mine.get(str(e["edge_index"]))
        if m is None:
            raise SystemExit(f"missing adjudication for edge {e['edge_index']}")
        rows.append(
            {
                **e,
                "adjudicator": m["label"],
                "adjudicator_rationale": m["why"],
                "qwen": e["type"],
                "agree": (m["label"] == e["type"]) if m["label"] != "AMBIGUOUS" else None,
            }
        )

    conf = defaultdict(lambda: defaultdict(int))
    for r in rows:
        conf[r["adjudicator"]][r["qwen"]] += 1

    per_class = {}
    for t in TYPES:
        sel = [r for r in rows if r["qwen"] == t]
        graded = [r for r in sel if r["adjudicator"] != "AMBIGUOUS"]
        k = sum(1 for r in graded if r["adjudicator"] == t)
        lo, hi = wilson(k, len(graded))
        # lenient: EQUIVALENT/ANALOGOUS are interchangeable (both SupCon positives)
        pos = {"EQUIVALENT", "ANALOGOUS"}
        kl = sum(1 for r in graded if r["adjudicator"] == t or (t in pos and r["adjudicator"] in pos))
        llo, lhi = wilson(kl, len(graded))
        per_class[t] = {
            "n_sampled": len(sel),
            "n_graded": len(graded),
            "n_ambiguous": len(sel) - len(graded),
            "precision": round(k / max(len(graded), 1), 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "precision_positive_collapsed": round(kl / max(len(graded), 1), 4),
            "wilson95_positive_collapsed": [round(llo, 4), round(lhi, 4)],
            "adjudicator_labels": {
                u: sum(1 for r in sel if r["adjudicator"] == u) for u in list(TYPES) + ["AMBIGUOUS"]
            },
        }

    by_stratum = {}
    for s in sorted({r["stratum"] for r in rows}):
        sel = [r for r in rows if r["stratum"] == s and r["adjudicator"] != "AMBIGUOUS"]
        k = sum(1 for r in sel if r["agree"])
        lo, hi = wilson(k, len(sel))
        by_stratum[s] = {
            "n": len(sel),
            "agreement": round(k / max(len(sel), 1), 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
        }

    graded_all = [r for r in rows if r["adjudicator"] != "AMBIGUOUS"]
    k_all = sum(1 for r in graded_all if r["agree"])
    lo, hi = wilson(k_all, len(graded_all))

    out = {
        "generated": "2026-08-28",
        "procedure": (
            "stratified sample drawn by scripts/analysis/a9_sample_edges.py "
            "(seed recorded below); the adjudicator read only the two rendered "
            "descriptors + task names + failure_lookalikes, committed a label per edge to "
            "a9_sheet/mylabels/chunk*.json, and only then compared to the Qwen verdict"
        ),
        "seed": key["seed"],
        "n": len(rows),
        "edge_store_id": key["edge_store_id"],
        "n_per_class_drawn": key["n_per_class_drawn"],
        "overall_agreement": {
            "n_graded": len(graded_all),
            "agreement": round(k_all / max(len(graded_all), 1), 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "n_ambiguous": len(rows) - len(graded_all),
            "ambiguous_rate": round((len(rows) - len(graded_all)) / len(rows), 4),
        },
        "per_class_precision": per_class,
        "confusion_adjudicator_rows_x_qwen_cols": {a: dict(c) for a, c in conf.items()},
        "agreement_by_stratum": by_stratum,
        "edges": rows,
    }
    Path(args.out).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "edges"}, indent=1))


if __name__ == "__main__":
    main()
