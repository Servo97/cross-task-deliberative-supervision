#!/usr/bin/env python3
"""Union a FROZEN pass-2 edge store with a DELTA store under one edge_store_id.

Step 0 of the §42.5 <V2C> rebuild. The §21 delta contract is that frozen anchors keep their frozen
buckets and the delta only adds NEW anchors, so the union is a per-FILE symlink merge and any
filename collision means that contract was violated -- which is fatal, not something to resolve by
picking a winner.

Granularity is per bucket FILE (not per domain or per task dir): two stores routinely share a domain
(the robomme top-up did) and even a task, while holding disjoint anchors inside it.

  python scripts/deliberation/merge_pass2_stores.py \
      --frozen ~/Research/TRI/wsm_data/deliberation/pass2_store \
      --delta  ~/Research/TRI/wsm_data/deliberation/pass2_delta_store \
      --out    ~/Research/TRI/wsm_data/deliberation/pass2_merged_v2c
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from collections import Counter


def edge_root(store: pathlib.Path) -> pathlib.Path:
    roots = sorted(p for p in (store / "edges").iterdir() if p.is_dir())
    if len(roots) != 1:
        raise SystemExit(f"{store}: expected exactly one edge store id, found {[p.name for p in roots]}")
    return roots[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--edge-store-id",
        default="",
        help="id for the merged store; defaults to the FROZEN store's id so downstream "
        "content addresses stay anchored to the frozen corpus",
    )
    args = ap.parse_args()

    frozen, delta = (pathlib.Path(p).expanduser() for p in (args.frozen, args.delta))
    out = pathlib.Path(args.out).expanduser()
    fr, dl = edge_root(frozen), edge_root(delta)
    esid = args.edge_store_id or fr.name

    # index/embed/mine come from the DELTA: it is the run that saw the whole 4-domain corpus.
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("index", "embed", "mine"):
        src = delta / sub
        if not src.exists():
            raise SystemExit(f"delta store has no {sub}/ — it cannot supply the merged index")
        link = out / sub
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src, target_is_directory=True)

    buckets_out = out / "edges" / esid / "buckets"
    buckets_out.mkdir(parents=True, exist_ok=True)
    per_source, collisions, seen = Counter(), [], {}
    for label, root in (("frozen", fr), ("delta", dl)):
        for f in sorted((root / "buckets").rglob("*.bucket.json")):
            rel = f.relative_to(root / "buckets")
            dest = buckets_out / rel
            key = str(rel)
            if key in seen:
                collisions.append({"bucket": key, "first": seen[key], "second": label})
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            os.symlink(f, dest)
            seen[key] = label
            per_source[label] += 1

    if collisions:
        raise SystemExit(
            f"[merge] FATAL: {len(collisions)} bucket(s) exist in BOTH stores, e.g. "
            f"{collisions[:3]}. The §21 delta contract is that frozen anchors keep their frozen "
            "buckets and the delta adds only NEW anchors; a collision means that was violated and "
            "the union would silently prefer one judgement over another."
        )

    def dir_sha(root: pathlib.Path) -> str:
        names = sorted(str(p.relative_to(root)) for p in root.rglob("*.bucket.json"))
        return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]

    per_domain = Counter(pathlib.Path(k).parts[0] for k in seen)
    meta = {
        "merged_edge_store_id": esid,
        "parents": {
            "frozen": {
                "store": str(frozen),
                "edge_store_id": fr.name,
                "buckets": per_source["frozen"],
                "bucket_list_sha16": dir_sha(fr / "buckets"),
            },
            "delta": {
                "store": str(delta),
                "edge_store_id": dl.name,
                "buckets": per_source["delta"],
                "bucket_list_sha16": dir_sha(dl / "buckets"),
            },
        },
        "index_embed_mine_from": str(delta),
        "buckets_total": sum(per_source.values()),
        "buckets_per_domain": dict(sorted(per_domain.items())),
        "collisions": 0,
        "note": "per-file symlink union; frozen anchors keep frozen buckets (§21)",
    }
    (out / "_merge.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
