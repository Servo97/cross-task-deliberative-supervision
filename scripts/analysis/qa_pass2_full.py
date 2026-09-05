#!/usr/bin/env python
"""Full QA of the H14 pass-2 edge store (plan §3 QA + amendments A1a/A1c) + low-vs-medium
reasoning-effort agreement against the medium pilot store.

Read-only w.r.t. the edge stores; writes one JSON to the deliberation data root.

  python scripts/analysis/qa_pass2_full.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.deliberation import pass2_deliberate as P2D  # noqa: E402
from scripts.deliberation import pass2_prompt as P2  # noqa: E402

ROOT = Path("~/Research/TRI/wsm_data/deliberation").expanduser()
LIVE_STORE = ROOT / "pass2_store"
LIVE_ESID = "fb22b06bb8e74fc849be75e8a3f619908031767723e7e48a5b87ca4095371815"
PILOT_STORE = ROOT / "pass2_pilot"
PILOT_ESIDS = [
    "240604a3170871745f04fd1bcd707244dbffb97efd2d92f3475654ea8b164d95",
    "2218498968f358436d5e00df29e1b4a94dedc7c81925990b54be54ebc98c636c",
]
OUT = ROOT / "qa_pass2_full.json"

POS = ("EQUIVALENT", "ANALOGOUS")
TYPES = list(P2.EDGE_TYPES)


def load_rows(store: Path) -> dict:
    return {r["seg_id"]: r for r in P2D.load_index(store)}


def load_buckets(store: Path) -> dict:
    out = {}
    p = store / "mine" / "buckets.jsonl"
    for line in p.read_text().splitlines():
        if line.strip():
            b = json.loads(line)
            out[b["anchor"]] = b
    return out


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n, 2) for c in counter.values() if c)


def read_store(store: Path, esid: str):
    """-> (list of verdict dicts, per-anchor verdict lists, validation report)."""
    rows = load_rows(store)
    mined = load_buckets(store)
    edges_root = store / "edges" / esid
    files = sorted((edges_root / "buckets").rglob("*.bucket.json"))
    verdicts, per_anchor, invalid = [], {}, []
    missing_from_mine = 0
    for p in files:
        d = json.loads(p.read_text())
        anchor = d["anchor"]
        b = mined.get(anchor)
        if b is None:
            missing_from_mine += 1
            ok = None
        else:
            ok = P2D.validate_bucket_file(p, b["candidates"])
        if ok is False:
            reasons = []
            if d.get("candidates") != b["candidates"]:
                reasons.append("candidate list mismatch")
            if (d.get("usage") or {}).get("finish_reason") == "length":
                reasons.append("truncated")
            vs = d.get("verdicts")
            if not isinstance(vs, list) or len(vs) != len(b["candidates"]):
                reasons.append(f"verdict count {len(vs) if isinstance(vs, list) else 'NA'}")
            else:
                for i, v in enumerate(vs):
                    e = P2.validate_verdict(v, i)
                    if e:
                        reasons.append(f"v{i}: {e}")
                    elif v.get("candidate_id") != b["candidates"][i]:
                        reasons.append(f"v{i}: candidate_id mismatch")
            invalid.append({"path": str(p.relative_to(edges_root)), "reasons": reasons[:5]})
        per_anchor[anchor] = d["verdicts"]
        verdicts.extend(d["verdicts"])
    rep = {
        "n_bucket_files": len(files),
        "n_invalid": len(invalid),
        "n_anchors_missing_from_mine": missing_from_mine,
        "n_mined_buckets": len(mined),
        "n_mined_without_bucket_file": len(set(mined) - set(per_anchor)),
        "invalid_examples": invalid[:20],
    }
    return rows, verdicts, per_anchor, rep


def analyse(rows, verdicts, per_anchor):
    out = {}
    by_type = Counter(v["type"] for v in verdicts)
    out["n_verdicts"] = len(verdicts)
    out["type_histogram"] = {t: by_type.get(t, 0) for t in TYPES}
    out["type_histogram_frac"] = {t: round(by_type.get(t, 0) / max(len(verdicts), 1), 4) for t in TYPES}
    out["confidence_histogram"] = {c: sum(1 for v in verdicts if v["confidence"] == c) for c in P2.CONFIDENCES}
    out["memory_relation_histogram"] = dict(Counter(v.get("memory_relation") for v in verdicts))

    # per stratum
    strat = defaultdict(Counter)
    for v in verdicts:
        strat[v["stratum"]][v["type"]] += 1
    out["type_by_stratum"] = {
        s: {
            **{t: c.get(t, 0) for t in TYPES},
            "n": sum(c.values()),
            **{f"frac_{t}": round(c.get(t, 0) / max(sum(c.values()), 1), 4) for t in TYPES},
        }
        for s, c in sorted(strat.items())
    }

    # per domain-pair (anchor domain, candidate domain), unordered
    dp = defaultdict(Counter)
    for v in verdicts:
        a, c = rows[v["anchor_id"]]["domain"], rows[v["candidate_id"]]["domain"]
        dp["|".join(sorted((a, c)))][v["type"]] += 1
    out["type_by_domain_pair"] = {
        k: {
            **{t: c.get(t, 0) for t in TYPES},
            "n": sum(c.values()),
            **{f"frac_{t}": round(c.get(t, 0) / max(sum(c.values()), 1), 4) for t in TYPES},
        }
        for k, c in sorted(dp.items())
    }

    # ---- degenerate judge
    allsame = [a for a, vs in per_anchor.items() if len({v["type"] for v in vs}) == 1]
    allsame_by_type = Counter(per_anchor[a][0]["type"] for a in allsame)
    ent = [entropy(Counter(v["type"] for v in vs)) for vs in per_anchor.values()]
    out["degenerate"] = {
        "n_anchors": len(per_anchor),
        "n_anchors_all_same_verdict": len(allsame),
        "frac_anchors_all_same_verdict": round(len(allsame) / max(len(per_anchor), 1), 4),
        "all_same_by_type": dict(allsame_by_type),
        "mean_within_anchor_type_entropy_bits": round(sum(ent) / max(len(ent), 1), 4),
        "median_within_anchor_type_entropy_bits": round(sorted(ent)[len(ent) // 2], 4) if ent else 0,
        "max_entropy_bits": 2.0,
    }
    # per-task verdict entropy (over anchors of that task)
    task_c = defaultdict(Counter)
    for v in verdicts:
        task_c[rows[v["anchor_id"]]["task"]][v["type"]] += 1
    per_task = {
        t: {"n": sum(c.values()), "entropy_bits": round(entropy(c), 4), **{tt: c.get(tt, 0) for tt in TYPES}}
        for t, c in sorted(task_c.items())
    }
    out["per_task_entropy"] = per_task
    # position bias: does the verdict depend on the slot index rather than the content?
    posc = defaultdict(Counter)
    for vs in per_anchor.values():
        for v in vs:
            posc[int(v["candidate"])][v["type"]] += 1
    out["degenerate"]["type_by_candidate_position"] = {
        str(i): {
            "n": sum(posc[i].values()),
            **{t: round(posc[i].get(t, 0) / max(sum(posc[i].values()), 1), 4) for t in TYPES},
        }
        for i in sorted(posc)
    }
    eqr = [posc[i].get("EQUIVALENT", 0) / max(sum(posc[i].values()), 1) for i in sorted(posc)]
    out["degenerate"]["EQUIVALENT_rate_position_spread"] = round(max(eqr) - min(eqr), 4)

    lows = sorted(per_task.items(), key=lambda kv: kv[1]["entropy_bits"])[:10]
    out["degenerate"]["lowest_entropy_tasks"] = {k: v["entropy_bits"] for k, v in lows}

    # ---- A1c quota floors on positives
    pos = [v for v in verdicts if v["type"] in POS]
    n_pos = len(pos)

    def frac(pred):
        return round(sum(1 for v in pos if pred(v)) / max(n_pos, 1), 4)

    stratum_ct = frac(lambda v: v["stratum"] in ("cross_task", "cross_domain"))
    stratum_cd = frac(lambda v: v["stratum"] == "cross_domain")
    actual_ct = frac(lambda v: rows[v["anchor_id"]]["task"] != rows[v["candidate_id"]]["task"])
    actual_cd = frac(lambda v: rows[v["anchor_id"]]["domain"] != rows[v["candidate_id"]]["domain"])
    out["A1c_quota_floors"] = {
        "n_positives": n_pos,
        "positive_rate": round(n_pos / max(len(verdicts), 1), 4),
        "by_mining_stratum": {
            "cross_task_or_domain_frac": stratum_ct,
            "floor": 0.40,
            "PASS_cross_task": stratum_ct >= 0.40,
            "cross_domain_frac": stratum_cd,
            "cross_domain_floor": 0.15,
            "PASS_cross_domain": stratum_cd >= 0.15,
        },
        "by_actual_relation": {
            "cross_task_frac": actual_ct,
            "floor": 0.40,
            "PASS_cross_task": actual_ct >= 0.40,
            "cross_domain_frac": actual_cd,
            "cross_domain_floor": 0.15,
            "PASS_cross_domain": actual_cd >= 0.15,
        },
        "positives_by_type": {t: sum(1 for v in pos if v["type"] == t) for t in POS},
    }

    # ---- unusable anchors (no positive edge at all -> no SupCon pair from this bucket)
    zero_pos = [a for a, vs in per_anchor.items() if not any(v["type"] in POS for v in vs)]
    zero_ct_pos = [
        a
        for a, vs in per_anchor.items()
        if not any(v["type"] in POS and rows[v["anchor_id"]]["task"] != rows[v["candidate_id"]]["task"] for v in vs)
    ]
    zero_eq = [a for a, vs in per_anchor.items() if not any(v["type"] == "EQUIVALENT" for v in vs)]
    out["unusable_anchors"] = {
        "n_anchors": len(per_anchor),
        "zero_positive": len(zero_pos),
        "zero_positive_frac": round(len(zero_pos) / max(len(per_anchor), 1), 4),
        "zero_cross_task_positive": len(zero_ct_pos),
        "zero_cross_task_positive_frac": round(len(zero_ct_pos) / max(len(per_anchor), 1), 4),
        "zero_EQUIVALENT": len(zero_eq),
        "zero_positive_by_task": dict(Counter(rows[a]["task"] for a in zero_pos).most_common(15)),
        "zero_positive_by_domain": dict(Counter(rows[a]["domain"] for a in zero_pos)),
    }

    # ---- A1a circularity gate (same computation as stage_qa)
    ec = [v for v in verdicts if v["type"] in ("EQUIVALENT", "CONTRAST")]
    auc = P2D._auc([v["cosine"] for v in ec], [1 if v["type"] == "EQUIVALENT" else 0 for v in ec])
    pa = [v for v in verdicts if v["type"] in POS]
    na = [v for v in verdicts if v["type"] in ("CONTRAST", "UNRELATED")]
    auc_pn = P2D._auc([v["cosine"] for v in pa + na], [1] * len(pa) + [0] * len(na))
    out["A1a_cosine_auc"] = {
        "n_equivalent_vs_contrast": len(ec),
        "auc_EQUIVALENT_vs_CONTRAST": None if auc != auc else round(auc, 4),
        "hold_threshold": 0.90,
        "VERDICT": ("HOLD" if auc == auc and auc >= 0.90 else "PROCEED"),
        "aux_auc_positive_vs_negative": None if auc_pn != auc_pn else round(auc_pn, 4),
        "mean_cosine_by_type": {
            t: round(
                sum(v["cosine"] for v in verdicts if v["type"] == t)
                / max(sum(1 for v in verdicts if v["type"] == t), 1),
                4,
            )
            for t in TYPES
        },
    }
    # per-stratum AUC (the forced strata are where cosine has least room)
    per_s = {}
    for s in sorted(strat):
        e = [v for v in verdicts if v["type"] in ("EQUIVALENT", "CONTRAST") and v["stratum"] == s]
        a = P2D._auc([v["cosine"] for v in e], [1 if v["type"] == "EQUIVALENT" else 0 for v in e])
        per_s[s] = {"n": len(e), "auc": None if a != a else round(a, 4)}
    out["A1a_cosine_auc"]["by_stratum"] = per_s

    # coverage
    tasks = sorted({r["task"] for r in rows.values()})
    contributing = set()
    for v in verdicts:
        if v["type"] == "EQUIVALENT" and v["stratum"] in ("cross_task", "cross_domain"):
            contributing.add(rows[v["anchor_id"]]["task"])
            contributing.add(rows[v["candidate_id"]]["task"])
    out["coverage"] = {
        "n_tasks": len(tasks),
        "tasks_with_cross_task_EQUIVALENT": len(contributing & set(tasks)),
        "isolated_tasks": sorted(set(tasks) - contributing),
    }
    out["wilson_95_EQUIVALENT_rate"] = [round(x, 4) for x in P2D._wilson(by_type.get("EQUIVALENT", 0), len(verdicts))]
    return out


def kappa(pairs, labels):
    """Cohen's kappa on a list of (a, b) label pairs."""
    n = len(pairs)
    if n == 0:
        return None
    obs = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    exp = sum((ca.get(lab, 0) / n) * (cb.get(lab, 0) / n) for lab in labels)
    if exp >= 1.0:
        return None
    return round((obs - exp) / (1 - exp), 4)


def effort_agreement(live_rows, live_per_anchor, pilot_stores):
    """Match (anchor, candidate) pairs present in both the medium pilot and the low live store.

    The two stores were mined over DIFFERENT corpora (pilot: 494 robocasa segments; live: 19,636
    segments over 3 domains), so the mined candidate sets barely intersect. Directed and
    undirected overlap are both reported, plus a distribution-level (unpaired) comparison on the
    comparable subpopulation: robocasa-anchor / robocasa-candidate verdicts, per stratum.
    """
    live_map = {}
    live_sym = {}
    for a, vs in live_per_anchor.items():
        for v in vs:
            live_map[(a, v["candidate_id"])] = v["type"]
            live_sym.setdefault(frozenset((a, v["candidate_id"])), v["type"])

    res = {}
    for esid, store in pilot_stores:
        edges_root = store / "edges" / esid
        files = sorted((edges_root / "buckets").rglob("*.bucket.json"))
        pil_map, pil_anchors = {}, set()
        for p in files:
            d = json.loads(p.read_text())
            pil_anchors.add(d["anchor"])
            for v in d["verdicts"]:
                pil_map[(d["anchor"], v["candidate_id"])] = v["type"]
        pil_sym = {}
        for (a, c), t in pil_map.items():
            pil_sym.setdefault(frozenset((a, c)), t)
        both_sym = sorted(set(pil_sym) & set(live_sym), key=lambda s: sorted(s))
        sym_pairs = [(pil_sym[k], live_sym[k]) for k in both_sym]

        both = sorted(set(pil_map) & set(live_map))
        pairs = [(pil_map[k], live_map[k]) for k in both]
        bin_pairs = [(("POS" if a in POS else "NEG"), ("POS" if b in POS else "NEG")) for a, b in pairs]
        conf = defaultdict(Counter)
        for a, b in pairs:
            conf[a][b] += 1
        res[esid] = {
            "n_pilot_buckets": len(files),
            "n_pilot_anchors_also_in_live": len(pil_anchors & set(live_per_anchor)),
            "n_pilot_pairs": len(pil_map),
            "n_overlapping_pairs": len(both),
            "exact4_agreement": round(sum(1 for a, b in pairs if a == b) / max(len(pairs), 1), 4),
            "kappa_exact4": kappa(pairs, TYPES),
            "binary_pos_agreement": round(sum(1 for a, b in bin_pairs if a == b) / max(len(bin_pairs), 1), 4),
            "kappa_binary": kappa(bin_pairs, ["POS", "NEG"]),
            "confusion_medium_rows_low_cols": {a: {b: conf[a].get(b, 0) for b in TYPES} for a in TYPES},
            "marginals_medium": {t: sum(1 for a, _ in pairs if a == t) for t in TYPES},
            "marginals_low": {t: sum(1 for _, b in pairs if b == t) for t in TYPES},
            "undirected": {
                "n_overlapping_unordered_pairs": len(both_sym),
                "exact4_agreement": round(sum(1 for a, b in sym_pairs if a == b) / max(len(sym_pairs), 1), 4),
                "kappa_exact4": kappa(sym_pairs, TYPES),
                "binary_pos_agreement": round(
                    sum(1 for a, b in sym_pairs if (a in POS) == (b in POS)) / max(len(sym_pairs), 1), 4
                ),
                "pairs": [{"medium": a, "low": b} for a, b in sym_pairs],
            },
            "PAIRED_COMPARISON_COMPUTABLE": len(both) >= 30 or len(both_sym) >= 30,
        }
    return res


def effort_distribution_compare(live_rows, live_per_anchor, pilot_store, pilot_esid):
    """Unpaired, distribution-level low-vs-medium comparison on the comparable subpopulation:
    robocasa anchor + robocasa candidate, per mining stratum. Different candidate sets, so this
    bounds systematic verdict drift, it is not an agreement measure."""
    med = defaultdict(Counter)
    med_anchors = set()
    pil_rows = load_rows(pilot_store)
    for p in sorted((pilot_store / "edges" / pilot_esid / "buckets").rglob("*.bucket.json")):
        d = json.loads(p.read_text())
        med_anchors.add(d["anchor"])
        for v in d["verdicts"]:
            if (
                pil_rows[v["anchor_id"]]["domain"] == "robocasa"
                and pil_rows[v["candidate_id"]]["domain"] == "robocasa"
            ):
                med[v["stratum"]][v["type"]] += 1
                med["ALL"][v["type"]] += 1
    low = defaultdict(Counter)
    low_shared = defaultdict(Counter)
    for a, vs in live_per_anchor.items():
        if live_rows[a]["domain"] != "robocasa":
            continue
        for v in vs:
            if live_rows[v["candidate_id"]]["domain"] != "robocasa":
                continue
            low[v["stratum"]][v["type"]] += 1
            low["ALL"][v["type"]] += 1
            if a in med_anchors:
                low_shared[v["stratum"]][v["type"]] += 1
                low_shared["ALL"][v["type"]] += 1

    def rates(c):
        n = sum(c.values())
        return {"n": n, **{t: round(c.get(t, 0) / max(n, 1), 4) for t in TYPES}}

    strata = sorted(set(med) | set(low))
    return {
        "note": "robocasa-anchor x robocasa-candidate only; unpaired (candidate sets differ)",
        "n_medium_anchors": len(med_anchors),
        "n_medium_anchors_present_in_live": len(med_anchors & set(live_per_anchor)),
        "medium_rates": {s: rates(med[s]) for s in strata},
        "low_rates_all_robocasa": {s: rates(low[s]) for s in strata},
        "low_rates_shared_anchors_only": {s: rates(low_shared[s]) for s in strata},
        "positive_rate_medium": round(sum(med["ALL"][t] for t in POS) / max(sum(med["ALL"].values()), 1), 4),
        "positive_rate_low_all": round(sum(low["ALL"][t] for t in POS) / max(sum(low["ALL"].values()), 1), 4),
        "positive_rate_low_shared_anchors": round(
            sum(low_shared["ALL"][t] for t in POS) / max(sum(low_shared["ALL"].values()), 1), 4
        ),
    }


def main():
    rows, verdicts, per_anchor, rep = read_store(LIVE_STORE, LIVE_ESID)
    result = {
        "generated": "2026-08-28",
        "edge_store": {"store": str(LIVE_STORE), "edge_store_id": LIVE_ESID},
        "validation": rep,
    }
    result.update(analyse(rows, verdicts, per_anchor))
    result["low_vs_medium_effort"] = effort_agreement(rows, per_anchor, [(e, PILOT_STORE) for e in PILOT_ESIDS])
    result["low_vs_medium_distribution"] = effort_distribution_compare(rows, per_anchor, PILOT_STORE, PILOT_ESIDS[0])
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_task_entropy"}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
