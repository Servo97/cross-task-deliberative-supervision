#!/usr/bin/env python3
"""RoboCerebra corpus inventory + FREE segmentation for the H14 deliberation pipeline.

Segmentation is native and exact: the sealed `robocerebra_train_v1` LeRobot tree carries a
per-frame `subtask_index` column (and a per-frame `task_index` giving the episode-local subtask
ordinal, plus `global_task_index` into `meta/tasks.jsonl`). A segment is a maximal contiguous run
of a constant subtask ordinal. No keyframe pipeline, no VLM, nothing to tune -- the same "FREE
from official columns" route RoboMME took (`robomme_source.py`).

Emits one JSON: episodes x segments, with the frame triple pass 1 renders per segment (t0, mid,
t1-1 -- the RoboCasa/rmb convention in `pass1_store`).

  python scripts/deliberation/robocerebra_corpus.py --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter

import numpy as np
import pandas as pd

DEFAULT_ROOT = pathlib.Path("~/Research/TRI/wsm_data/robocerebra/lerobot_home/wsmv2/robocerebra_train").expanduser()


def load_meta(root: pathlib.Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {
        int(r["task_index"]): r["task"]
        for r in (
            json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()
        )
    }
    prov = {
        int(r["episode_index"]): r
        for r in (
            json.loads(line)
            for line in (root / "meta" / "episode_provenance.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    eps = {
        int(r["episode_index"]): r
        for r in (
            json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()
        )
    }
    return info, tasks, prov, eps


def segments_for(df: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    """-> [(ordinal, t0, t1_exclusive, task_index)] from contiguous runs of `subtask_index`.

    Column semantics, established by inspection rather than assumed (three int columns, all
    plausible-looking, and two of them are traps):
      * `subtask_index`      episode-LOCAL ordinal 0..n-1  -> THE segmentation
      * `task_index`         LeRobot global id of the SUBTASK string -> segment text
      * `global_task_index`  LeRobot global id of the EPISODE task line (constant per episode)
    Segmenting on `task_index` merges two adjacent subtasks that happen to share a string, and
    `meta/episodes.jsonl["tasks"]` is a de-duplicated set, NOT temporal order.
    """
    ordinal = df["subtask_index"].to_numpy()
    gti = df["task_index"].to_numpy()
    cuts = np.flatnonzero(np.diff(ordinal) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(ordinal)]])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.append((int(ordinal[a]), int(a), int(b), int(gti[a])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    root = args.root.expanduser()
    info, tasks, prov, eps = load_meta(root)
    chunks_size = int(info.get("chunks_size", 1000))

    records, seg_total, monotone_bad, len_mismatch = [], 0, 0, 0
    seglen = []
    for ep in sorted(eps):
        chunk = ep // chunks_size
        df = pd.read_parquet(root / info["data_path"].format(episode_chunk=chunk, episode_index=ep))
        segs = segments_for(df)
        p = prov[ep]
        if len(df) != int(p["length"]):
            len_mismatch += 1
        # The real contract is "strictly increasing, never revisited", NOT "starts at 0 and covers
        # every declared subtask": 8 episodes are demos truncated short of their case definition
        # (ep 983 even starts at subtask 1), which costs 18 of the 8,887 declared segments and is
        # data, not corruption. Checking against range(n) flags those 8 as anomalies; checking
        # monotonicity is what actually catches a broken segmentation.
        ordinals = [s[0] for s in segs]
        if any(b <= a for a, b in zip(ordinals, ordinals[1:])):
            monotone_bad += 1
        seg_recs = []
        for k, (ordinal, t0, t1, gti) in enumerate(segs):
            mid = t0 + (t1 - t0) // 2
            frames = sorted({t0, mid, t1 - 1})
            seg_recs.append(
                {
                    "segment": k,
                    "t0": t0,
                    "t1": t1,
                    "ordinal": ordinal,
                    "global_task_index": gti,
                    "text": tasks.get(gti, ""),
                    "frames": frames,
                }
            )
            seglen.append(t1 - t0)
        seg_total += len(seg_recs)
        records.append(
            {
                "episode_index": ep,
                "scene": p["scene"],
                "case": p["case"],
                "bddl_file": p["bddl_file"],
                "distractor": p.get("distractor"),
                "task_line": p["task_line"],
                "n_frames": int(len(df)),
                "num_subtasks_declared": int(p["num_subtasks"]),
                "prompt": p["task_line"],
                "segments": seg_recs,
            }
        )

    payload = {
        "domain": "robocerebra",
        "dataset_artifact": "robocerebra_train_v1",
        "dataset_root": str(root),
        "segmentation": "native per-frame task_index runs (LeRobot columns); FREE, no keyframe pipeline",
        "n_episodes": len(records),
        "n_segments": seg_total,
        "n_frames": sum(r["n_frames"] for r in records),
        "n_unique_subtask_strings": len(tasks),
        "scenes": dict(Counter(r["scene"] for r in records)),
        "segment_len": {
            "min": int(min(seglen)),
            "max": int(max(seglen)),
            "mean": float(np.mean(seglen)),
            "median": float(np.median(seglen)),
        },
        "checks": {
            "episodes_with_nonmonotone_ordinals": monotone_bad,
            "episodes_with_length_mismatch_vs_provenance": len_mismatch,
            "declared_vs_derived_segment_total": [sum(r["num_subtasks_declared"] for r in records), seg_total],
        },
        "episodes": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=1, sort_keys=True)
    args.out.write_text(blob)
    payload_sha = hashlib.sha256(blob.encode()).hexdigest()
    summary = {k: v for k, v in payload.items() if k != "episodes"}
    summary["corpus_sha256"] = payload_sha
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
