"""A9 accuracy sheet: draw a stratified sample of live pass-2 edges and render a BLIND sheet.

Blind protocol: `blind/*.txt` carries the two descriptors + adjudicator-only context (task names,
failure_lookalikes) and NEVER the Qwen verdict; `key.json` carries the verdicts. The adjudicator
reads the blind chunks, commits labels to `mylabels.json`, and only then runs a9_score_sheet.py.

  python scripts/analysis/a9_sample_edges.py --n-per-class 60 --seed 20260828
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.deliberation import pass2_prompt as P2  # noqa: E402

TYPES = ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")
STRATA = ("within_task", "cross_task", "cross_domain", "mined_hard_neg")
DOMAINS = ("robocasa", "robomme", "remembench")


def render_pair(a: dict, b: dict, aid: str, bid: str, stratum: str) -> str:
    ad, at, aep, aseg = aid.split("/")
    bd, bt, bep, bseg = bid.split("/")
    out = [
        f"[stratum: {stratum}]",
        f"A  ({ad}/{at}  ep {aep} seg {aseg})",
        P2.render_segment(a, header="").rstrip(),
        f"   failure_lookalikes: {'; '.join(a.get('failure_lookalikes') or []) or '-'}",
        f"B  ({bd}/{bt}  ep {bep} seg {bseg})",
        P2.render_segment(b, header="").rstrip(),
        f"   failure_lookalikes: {'; '.join(b.get('failure_lookalikes') or []) or '-'}",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/Research/TRI/wsm_data/deliberation/pass2_store")
    ap.add_argument("--edge-store-id", default="fb22b06bb8e74fc849be75e8a3f619908031767723e7e48a5b87ca4095371815")
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/a9_sheet")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--chunk", type=int, default=30)
    args = ap.parse_args()

    store = Path(args.store).expanduser()
    rows = {}
    for line in (store / "index" / "segments.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["seg_id"]] = r
    edges_root = store / "edges" / args.edge_store_id
    files = sorted((edges_root / "buckets").rglob("*.bucket.json"))
    print(f"[sample] {len(files)} buckets, {len(rows)} segments", flush=True)

    # cells: (type, stratum, anchor_domain) -> list of edge refs
    cells = defaultdict(list)
    for p in files:
        d = json.loads(p.read_text())
        for v in d["verdicts"]:
            a, c = v["anchor_id"], v["candidate_id"]
            cells[(v["type"], v["stratum"], a.split("/")[0])].append(
                {
                    "anchor": a,
                    "candidate": c,
                    "type": v["type"],
                    "stratum": v["stratum"],
                    "confidence": v["confidence"],
                    "rationale": v["rationale"],
                    "memory_relation": v["memory_relation"],
                    "cosine": v["cosine"],
                }
            )

    rng = random.Random(args.seed)
    picked, seen = [], set()

    def take(cell_key, k):
        pool = cells.get(cell_key, [])
        if not pool:
            return 0
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        got = 0
        for i in idx:
            if got >= k:
                break
            e = pool[i]
            key = frozenset((e["anchor"], e["candidate"]))
            if key in seen:
                continue
            seen.add(key)
            picked.append(e)
            got += 1
        return got

    per_class = {}
    for t in TYPES:
        want = args.n_per_class
        got_t = 0
        # 15 per stratum, spread 5 per anchor-domain
        for s in STRATA:
            per_stratum = want // len(STRATA)
            got_s = 0
            for dom in DOMAINS:
                got_s += take((t, s, dom), per_stratum // len(DOMAINS))
            # backfill inside the stratum from any domain
            for dom in DOMAINS:
                if got_s >= per_stratum:
                    break
                got_s += take((t, s, dom), per_stratum - got_s)
            got_t += got_s
        # backfill inside the class from any stratum/domain
        for s in STRATA:
            for dom in DOMAINS:
                if got_t >= want:
                    break
                got_t += take((t, s, dom), want - got_t)
        per_class[t] = got_t

    rng.shuffle(picked)
    for i, e in enumerate(picked):
        e["edge_index"] = i

    out = Path(args.out).expanduser()
    (out / "blind").mkdir(parents=True, exist_ok=True)
    chunks = [picked[i : i + args.chunk] for i in range(0, len(picked), args.chunk)]
    for ci, ch in enumerate(chunks):
        txt = []
        for e in ch:
            txt.append(f"=== EDGE {e['edge_index']} ===")
            txt.append(
                render_pair(
                    rows[e["anchor"]]["descriptor"],
                    rows[e["candidate"]]["descriptor"],
                    e["anchor"],
                    e["candidate"],
                    e["stratum"],
                )
            )
            txt.append("")
        (out / "blind" / f"chunk{ci:02d}.txt").write_text("\n".join(txt))
    (out / "key.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_per_class_requested": args.n_per_class,
                "n_per_class_drawn": per_class,
                "n": len(picked),
                "edge_store_id": args.edge_store_id,
                "edges": picked,
            },
            indent=1,
        )
    )
    print(json.dumps({"n": len(picked), "per_class": per_class, "chunks": len(chunks), "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
