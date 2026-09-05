"""A9 follow-up: re-adjudicate CONTRAST verdicts + planted probes under the INTENT rule.

The frozen schema's EQUIVALENT clause ("bound objects may differ in colour or instance") makes the
letter-rule adjudication treat instance/side/count swaps as EQUIVALENT. For a MEMORY campaign that
is the wrong reading: if the differing variable is *memory-bound* in either segment, pulling the
pair together as a SupCon positive erases the bound variable.

INTENT RULE (pre-registered by the coordinator before any re-labelling):
  CONTRAST if the two segments share subskill/goal but differ in a variable that is memory-bound in
  at least one of them (per its descriptor's memory_dependency kinds/targets), such that executing
  one's completion in the other's context would fail;
  EQUIVALENT if the differing variable is not memory-bound.

Stages:
  render   -> blind pair renders (descriptors only, no prior label, no Qwen verdict)
  score    -> join committed intent labels with the letter-rule sheet + Qwen verdicts

  python scripts/analysis/a9_readjudicate_intent.py render --out <dir>
  python scripts/analysis/a9_readjudicate_intent.py score  --labels <dir>/mylabels
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def memory_kinds_of(desc: dict) -> list:
    """Vendored from workspace_models.labels.caption_segments (avoids a numpy import here)."""
    md = (desc or {}).get("memory_dependency") or {}
    ks = md.get("kinds")
    if isinstance(ks, list) and ks:
        return [str(k) for k in ks]
    k = md.get("kind")
    return [str(k)] if k else ["none"]


def render_segment(desc: dict, *, header: str) -> str:
    """Byte-identical to scripts/deliberation/pass2_prompt.render_segment."""
    t = desc.get("target_object", {}) or {}
    attrs = ", ".join(t.get("attributes") or []) or "-"
    md = desc.get("memory_dependency", {}) or {}
    kinds = memory_kinds_of(desc)
    kind_str = "+".join(kinds)
    return (
        f"{header}\n"
        f"  subskill: {desc.get('subskill', '')}\n"
        f"  verb_frame: {desc.get('verb_frame', '')}\n"
        f"  object: {t.get('class', '')} [{attrs}]\n"
        f"  object_state: {t.get('state_before', '')} -> {t.get('state_after', '')}\n"
        f"  spatial: {desc.get('spatial_relation', '')}\n"
        f"  preconditions: {'; '.join(desc.get('preconditions') or [])}\n"
        f"  postconditions: {'; '.join(desc.get('postconditions') or [])}\n"
        f"  memory_dependency: {kind_str}"
        f"{' -- ' + str(md.get('evidence')) if kinds != ['none'] else ''}\n"
    )


DELIB = Path("~/Research/TRI/wsm_data/deliberation").expanduser()
STORE = DELIB / "pass2_store"
SHEET = DELIB / "qa_pass2_accuracy_sheet.json"
PROBES = DELIB / "qa_pass2_probe_recovery.json"


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_rows() -> dict:
    rows = {}
    for line in (STORE / "index" / "segments.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["seg_id"]] = r
    return rows


def render_pair(a: dict, b: dict, aid: str, bid: str, tag: str) -> str:
    ad, at, aep, aseg = aid.split("/")
    bd, bt, bep, bseg = bid.split("/")
    return "\n".join(
        [
            f"[{tag}]",
            f"A  ({ad}/{at}  ep {aep} seg {aseg})",
            render_segment(a, header="").rstrip(),
            f"   failure_lookalikes: {'; '.join(a.get('failure_lookalikes') or []) or '-'}",
            f"B  ({bd}/{bt}  ep {bep} seg {bseg})",
            render_segment(b, header="").rstrip(),
            f"   failure_lookalikes: {'; '.join(b.get('failure_lookalikes') or []) or '-'}",
        ]
    )


def stage_render(args) -> None:
    rows = load_rows()
    sheet = json.loads(SHEET.read_text())
    probe = json.loads(PROBES.read_text())

    items = []
    for e in sheet["edges"]:
        if e["qwen"] != "CONTRAST":
            continue
        items.append(
            {
                "uid": f"S{e['edge_index']:03d}",
                "kind": "sheet",
                "anchor": e["anchor"],
                "candidate": e["candidate"],
                "stratum": e["stratum"],
            }
        )
    for p in probe["probes"]:
        if p["ground_truth"] != "CONTRAST":
            continue
        items.append(
            {
                "uid": f"P{p['probe_id'][:8]}",
                "kind": "probe",
                "anchor": p["anchor"],
                "candidate": p["candidate"],
                "stratum": "probe",
            }
        )

    out = Path(args.out).expanduser()
    (out / "blind").mkdir(parents=True, exist_ok=True)
    chunk = args.chunk
    chunks = [items[i : i + chunk] for i in range(0, len(items), chunk)]
    for ci, ch in enumerate(chunks):
        txt = []
        for it in ch:
            txt.append(f"=== ITEM {it['uid']} ===")
            txt.append(
                render_pair(
                    rows[it["anchor"]]["descriptor"],
                    rows[it["candidate"]]["descriptor"],
                    it["anchor"],
                    it["candidate"],
                    it["stratum"],
                )
            )
            txt.append("")
        (out / "blind" / f"chunk{ci:02d}.txt").write_text("\n".join(txt))
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print(
        json.dumps(
            {
                "n_items": len(items),
                "n_sheet": sum(1 for i in items if i["kind"] == "sheet"),
                "n_probe": sum(1 for i in items if i["kind"] == "probe"),
                "chunks": len(chunks),
                "out": str(out),
            },
            indent=1,
        )
    )


def stage_score(args) -> None:
    root = Path(args.out).expanduser()
    _items = {i["uid"]: i for i in json.loads((root / "items.json").read_text())}  # read kept: fails fast if absent
    labels = {}
    for f in sorted((root / "mylabels").glob("*.json")):
        labels.update(json.loads(f.read_text()))
    sheet = json.loads(SHEET.read_text())
    probe = json.loads(PROBES.read_text())
    by_idx = {f"S{e['edge_index']:03d}": e for e in sheet["edges"] if e["qwen"] == "CONTRAST"}
    by_probe = {f"P{p['probe_id'][:8]}": p for p in probe["probes"] if p["ground_truth"] == "CONTRAST"}

    POS = {"EQUIVALENT", "ANALOGOUS"}
    res = {
        "rule": (
            "CONTRAST if the two segments share subskill/goal but differ in a variable that "
            "is memory-bound in at least one of them (per its descriptor's memory_dependency "
            "kinds/targets), such that executing one's completion in the other's context "
            "would fail; EQUIVALENT if the differing variable is not memory-bound."
        ),
        "procedure": (
            "blind: pairs re-rendered from pass-1 descriptors only "
            "(scripts/analysis/a9_readjudicate_intent.py render); the intent-rule label "
            "was committed per item to mylabels/ BEFORE the letter-rule label or the "
            "Qwen verdict was joined in (this script)."
        ),
        "edge_store_id": sheet["edge_store_id"],
    }

    # ---- sheet CONTRAST edges ----
    rec = []
    for uid, e in by_idx.items():
        L = labels[uid]
        rec.append(
            {
                "uid": uid,
                "anchor": e["anchor"],
                "candidate": e["candidate"],
                "stratum": e["stratum"],
                "qwen": e["qwen"],
                "letter_rule": e["adjudicator"],
                "intent_rule": L["label"],
                "intent_rationale": L.get("why", ""),
                "bound_var": L.get("bound_var", ""),
                "letter_correct": e["adjudicator"] == "CONTRAST",
                "intent_correct": L["label"] == "CONTRAST",
            }
        )
    # the letter-rule sheet left 2 of the 60 AMBIGUOUS; headline numbers use the same 58
    graded = [r for r in rec if r["letter_rule"] != "AMBIGUOUS"]
    n_all = len(rec)
    n = len(graded)
    k_int = sum(r["intent_correct"] for r in graded)
    k_let = sum(r["letter_correct"] for r in graded)
    k_int_all = sum(r["intent_correct"] for r in rec)
    flips = [r for r in rec if r["letter_rule"] in POS and r["intent_rule"] == "CONTRAST"]
    stay_pos = [r for r in rec if r["letter_rule"] in POS and r["intent_rule"] in POS]
    residual = [r for r in rec if not r["intent_correct"] and not r["letter_correct"]]
    res["sheet_contrast"] = {
        "n_sampled": n_all,
        "n_graded": n,
        "n_ambiguous_under_letter_rule": n_all - n,
        "letter_rule_precision": round(k_let / n, 4),
        "letter_wilson95": wilson(k_let, n),
        "intent_rule_precision": round(k_int / n, 4),
        "intent_wilson95": wilson(k_int, n),
        "intent_rule_precision_all60": round(k_int_all / n_all, 4),
        "intent_wilson95_all60": wilson(k_int_all, n_all),
        "n_letter_positive_adjudications": sum(1 for r in rec if r["letter_rule"] in POS),
        "n_flipped_positive_to_CONTRAST": len(flips),
        "n_stayed_positive": len(stay_pos),
        "intent_label_hist": dict(Counter(r["intent_rule"] for r in rec)),
        "by_stratum": {
            s: {
                "n": sum(1 for r in rec if r["stratum"] == s),
                "letter": round(
                    sum(r["letter_correct"] for r in rec if r["stratum"] == s)
                    / max(1, sum(1 for r in rec if r["stratum"] == s)),
                    4,
                ),
                "intent": round(
                    sum(r["intent_correct"] for r in rec if r["stratum"] == s)
                    / max(1, sum(1 for r in rec if r["stratum"] == s)),
                    4,
                ),
            }
            for s in sorted({r["stratum"] for r in rec})
        },
        "residual_wrong_under_both": {
            "n": len(residual),
            "rate": round(len(residual) / n_all, 4),
            "items": [
                {
                    "uid": r["uid"],
                    "anchor": r["anchor"],
                    "candidate": r["candidate"],
                    "letter": r["letter_rule"],
                    "intent": r["intent_rule"],
                    "why": r["intent_rationale"],
                }
                for r in residual
            ],
        },
        "edges": rec,
    }

    # ---- probes ----
    prec = []
    for uid, p in by_probe.items():
        L = labels[uid]
        v = p["verdict"]
        prec.append(
            {
                "uid": uid,
                "probe_id": p["probe_id"],
                "family": p["family"],
                "anchor": p["anchor"],
                "candidate": p["candidate"],
                "qwen": v,
                "intent_rule": L["label"],
                "why": L.get("why", ""),
                "descriptor_visible": L.get("descriptor_visible", None),
                "recovered_letter": v in ("CONTRAST", "UNRELATED"),
                "intent_confirms_gt": L["label"] == "CONTRAST",
                "qwen_matches_intent": (v in ("CONTRAST", "UNRELATED")) if L["label"] == "CONTRAST" else (v in POS),
            }
        )
    np_ = len(prec)
    conf = sum(p["intent_confirms_gt"] for p in prec)
    rec_letter = sum(p["recovered_letter"] for p in prec)
    match_intent = sum(p["qwen_matches_intent"] for p in prec)
    # recovery restricted to probes the intent rule still calls CONTRAST
    sub = [p for p in prec if p["intent_confirms_gt"]]
    rec_sub = sum(p["recovered_letter"] for p in sub)
    fam = defaultdict(lambda: {"n": 0, "intent_CONTRAST": 0, "recovered": 0})
    for p in prec:
        f = fam[p["family"]]
        f["n"] += 1
        f["intent_CONTRAST"] += int(p["intent_confirms_gt"])
        f["recovered"] += int(p["recovered_letter"])
    res["probes"] = {
        "n": np_,
        "letter_recovery": round(rec_letter / np_, 4),
        "letter_wilson95": wilson(rec_letter, np_),
        "intent_confirms_ground_truth": conf,
        "intent_confirm_rate": round(conf / np_, 4),
        "intent_confirm_wilson95": wilson(conf, np_),
        "recovery_on_intent_confirmed": (round(rec_sub / len(sub), 4) if sub else None),
        "recovery_on_intent_confirmed_wilson95": wilson(rec_sub, len(sub)) if sub else None,
        "qwen_agrees_with_intent_label": round(match_intent / np_, 4),
        "qwen_agrees_wilson95": wilson(match_intent, np_),
        "by_family": {k: dict(v) for k, v in sorted(fam.items())},
        "by_descriptor_visibility": {
            ("visible" if vis else "check_success_only"): {
                "n": len(S),
                "intent_CONTRAST": sum(p["intent_confirms_gt"] for p in S),
                "letter_recovery": round(sum(p["recovered_letter"] for p in S) / max(1, len(S)), 4),
                "qwen_matches_intent": round(sum(p["qwen_matches_intent"] for p in S) / max(1, len(S)), 4),
            }
            for vis, S in (
                (True, [p for p in prec if p["descriptor_visible"]]),
                (False, [p for p in prec if not p["descriptor_visible"]]),
            )
        },
        "probes": prec,
    }
    outp = Path(args.write).expanduser()
    outp.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res["sheet_contrast"].items() if k != "edges"}, indent=1))
    print(json.dumps({k: v for k, v in res["probes"].items() if k != "probes"}, indent=1))
    print(f"[write] {outp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "score"])
    ap.add_argument("--out", default=str(DELIB / "a9_intent_readjud"))
    ap.add_argument("--chunk", type=int, default=15)
    ap.add_argument("--write", default=str(DELIB / "qa_pass2_contrast_readjudication.json"))
    main_args = ap.parse_args()
    {"render": stage_render, "score": stage_score}[main_args.stage](main_args)


if __name__ == "__main__":
    main()
