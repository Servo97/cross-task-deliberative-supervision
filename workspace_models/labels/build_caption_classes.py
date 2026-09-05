#!/usr/bin/env python3
"""R3b: emit caption CLASS ids alongside the frozen embeddings.

The R3/R4 head aligned `w` to a frozen text embedding with InfoNCE and collapsed to the caption
centroid (distinct captions have mean pairwise cosine 0.841, so "predict the average" nearly
satisfied it). Cross-entropy over the caption vocabulary cannot be satisfied that way: a centroid
gives you the prior, not the label.

Ids are the index of the caption string in the SORTED unique vocabulary — deterministic from the
caption text alone, so the map is reproducible from the label store without this cache. The
embeddings are copied through unchanged, for TELEMETRY only (never a loss).

Out: <out>/<Task>/ep_%06d.capcls.npz {seg_t0, seg_t1, seg_id, emb} + <out>/vocab.json (content-addressed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from build_caption_embeddings import CAPTION_GLOB, load_episode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captions-root", required=True)
    ap.add_argument("--emb-root", required=True, help="existing capemb cache (embeddings reused as-is)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.captions_root)
    episodes = [
        load_episode(path)
        for task in sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
        for path in sorted((root / task).glob(CAPTION_GLOB))
    ]
    vocab = sorted({t for ep in episodes for t in ep["texts"]})
    index = {t: i for i, t in enumerate(vocab)}
    vocab_json = json.dumps(vocab, sort_keys=False, separators=(",", ":"), ensure_ascii=True)
    vocab_sha = hashlib.sha256(vocab_json.encode()).hexdigest()

    out = Path(args.out)
    digest = hashlib.sha256()
    for ep in episodes:
        src = Path(args.emb_root) / ep["task"] / f"ep_{ep['episode_id']:06d}.capemb.npz"
        with np.load(src, allow_pickle=False) as a:
            t0, t1, emb = a["seg_t0"], a["seg_t1"], a["emb"]
        assert np.array_equal(t0, ep["t0"]) and np.array_equal(t1, ep["t1"]), src
        ids = np.asarray([index[t] for t in ep["texts"]], dtype=np.int32)
        dest = out / ep["task"]
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / f"ep_{ep['episode_id']:06d}.capcls.npz", seg_t0=t0, seg_t1=t1, seg_id=ids, emb=emb)
        digest.update(f"{ep['task']}/{ep['episode_id']}".encode())
        digest.update(ids.tobytes())
    (out / "vocab.json").write_text(vocab_json + "\n")
    (out / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "h13_caption_class_ids",
                "episodes": len(episodes),
                "vocab_size": len(vocab),
                "vocab_sha256": vocab_sha,
                "id_definition": "index into the sorted unique caption strings",
                "embeddings": "copied from the capemb cache; TELEMETRY ONLY, never a loss",
                "content_sha256": digest.hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"[capcls] {len(episodes)} episodes, V={len(vocab)}, vocab_sha={vocab_sha}")
    print(f"[capcls] content_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
