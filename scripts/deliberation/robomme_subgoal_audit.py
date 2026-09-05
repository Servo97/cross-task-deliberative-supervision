"""H14 P0 — RoboMME subgoal-coverage audit (s4 G7's cheap audit; plan §7 P0, amendment A5(i)).

The plan ASSUMES the official per-step `simple_subgoal` / `grounded_subgoal` columns can be
run-length-encoded into pass-1 segments, giving 1,600 demos of segmentation for free (s3 §4 row
"RoboMME per-step subgoal strings"). This script VERIFIES that assumption instead of inheriting it,
and answers the A3 view-count question by reading the store rather than the two disagreeing scouts.

CPU only. Reads parquet directly from the pinned HF snapshot; touches no GPU and no network.

  python scripts/deliberation/robomme_subgoal_audit.py --episodes-per-task 5 \
      --out ~/Research/TRI/wsm_data/deliberation/robomme_subgoal_audit.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_ROOT = (
    "~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
    "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20"
)

TASK_ORDER = (
    "PatternLock",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoPlaceButton",
    "VideoUnmaskSwap",
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "PickHighlight",
    "MoveCube",
    "InsertPeg",
    "RouteStick",
    "BinFill",
    "VideoPlaceOrder",
    "VideoRepick",
    "VideoUnmask",
)
EPISODES_PER_TASK = 100

SUITE = {
    "BinFill": "counting",
    "PickXtimes": "counting",
    "SwingXtimes": "counting",
    "StopCube": "counting",
    "VideoUnmask": "permanence",
    "ButtonUnmask": "permanence",
    "VideoUnmaskSwap": "permanence",
    "ButtonUnmaskSwap": "permanence",
    "PickHighlight": "reference",
    "VideoRepick": "reference",
    "VideoPlaceButton": "reference",
    "VideoPlaceOrder": "reference",
    "MoveCube": "imitation",
    "InsertPeg": "imitation",
    "PatternLock": "imitation",
    "RouteStick": "imitation",
}

SUBGOAL_COLS = ("simple_subgoal", "grounded_subgoal", "simple_subgoal_online", "grounded_subgoal_online")


def episode_path(root: Path, ep: int) -> Path:
    return root / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"


def rle(values: list[str]) -> list[tuple[int, int, str]]:
    """Per-step strings -> [(t0, t1, text)) segments. This IS the proposed segmentation."""
    if not values:
        return []
    out, t0 = [], 0
    for t in range(1, len(values) + 1):
        if t == len(values) or values[t] != values[t0]:
            out.append((t0, t, values[t0]))
            t0 = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--episodes-per-task", type=int, default=5)
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/robomme_subgoal_audit.json")
    ap.add_argument("--eyeball", type=int, default=10, help="segments to print verbatim")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    root = Path(args.root).expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    image_cols = sorted(k for k, v in feats.items() if v.get("dtype") in ("image", "video"))
    report: dict = {
        "root": str(root),
        "store_facts": {
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fps": info.get("fps"),
            "total_videos": info.get("total_videos"),
            "image_columns": image_cols,  # <- the A3 view-count question, answered by the store
            "n_image_columns": len(image_cols),
            "image_shapes": {k: feats[k].get("shape") for k in image_cols},
            "subgoal_columns_present": [c for c in SUBGOAL_COLS if c in feats],
            "has_is_demo": "is_demo" in feats,
            "has_exec_start_idx": "exec_start_idx" in feats,
        },
        "tasks": {},
    }

    eyeball: list[dict] = []
    all_counts: dict[str, list[int]] = {c: [] for c in SUBGOAL_COLS}
    all_lens: dict[str, list[int]] = {c: [] for c in SUBGOAL_COLS}
    all_empty = {c: 0 for c in SUBGOAL_COLS}
    all_words: dict[str, list[int]] = {c: [] for c in SUBGOAL_COLS}
    vocab: dict[str, set] = {c: set() for c in SUBGOAL_COLS}
    grand_segments = {c: 0 for c in SUBGOAL_COLS}
    grand_frames = 0

    for ti, task in enumerate(TASK_ORDER):
        base = ti * EPISODES_PER_TASK
        eps = list(range(base, base + args.episodes_per_task))
        trow: dict = {"suite": SUITE[task], "episodes": eps, "per_column": {}}
        per_col: dict[str, dict] = {
            c: {"segments": [], "seg_lens": [], "empty": 0, "monotonic_ok": 0, "n_ep": 0} for c in SUBGOAL_COLS
        }
        ep_lens = []
        demo_prefix = []
        for ep in eps:
            p = episode_path(root, ep)
            if not p.is_file():
                trow.setdefault("missing", []).append(ep)
                continue
            cols = [c for c in SUBGOAL_COLS if c in feats] + [c for c in ("is_demo", "exec_start_idx") if c in feats]
            tbl = pq.read_table(p, columns=cols)
            n = tbl.num_rows
            ep_lens.append(n)
            grand_frames += n
            if "exec_start_idx" in cols:
                v = tbl.column("exec_start_idx").to_pylist()
                demo_prefix.append(int(v[0]) if v else 0)
            for c in SUBGOAL_COLS:
                if c not in cols:
                    continue
                vals = ["" if v is None else str(v) for v in tbl.column(c).to_pylist()]
                segs = rle(vals)
                per_col[c]["n_ep"] += 1
                per_col[c]["segments"].append(len(segs))
                per_col[c]["seg_lens"].extend(b - a for a, b, _ in segs)
                per_col[c]["empty"] += sum(1 for _, _, t in segs if not t.strip())
                # the contract pass 1 needs: contiguous, covering, non-empty
                ok = (
                    bool(segs)
                    and segs[0][0] == 0
                    and segs[-1][1] == n
                    and all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
                )
                per_col[c]["monotonic_ok"] += int(ok)
                all_counts[c].append(len(segs))
                all_lens[c].extend(b - a for a, b, _ in segs)
                all_empty[c] += sum(1 for _, _, t in segs if not t.strip())
                all_words[c].extend(len(t.split()) for _, _, t in segs if t.strip())
                vocab[c].update(t for _, _, t in segs if t.strip())
                grand_segments[c] += len(segs)
                if c == "simple_subgoal" and len(eyeball) < args.eyeball and segs:
                    mid = segs[len(segs) // 2]
                    eyeball.append(
                        {
                            "task": task,
                            "episode": ep,
                            "n_frames": n,
                            "n_segments": len(segs),
                            "segment": {"t0": mid[0], "t1": mid[1], "simple": mid[2]},
                            "all_simple": [t for _, _, t in segs][:12],
                        }
                    )
        for c, d in per_col.items():
            if not d["segments"]:
                continue
            trow["per_column"][c] = {
                "n_ep": d["n_ep"],
                "segments_per_ep_mean": round(statistics.mean(d["segments"]), 2),
                "segments_per_ep_min": min(d["segments"]),
                "segments_per_ep_max": max(d["segments"]),
                "seg_len_median": int(statistics.median(d["seg_lens"])) if d["seg_lens"] else 0,
                "empty_segments": d["empty"],
                "contiguous_cover_ok": f"{d['monotonic_ok']}/{d['n_ep']}",
            }
        trow["ep_len_median"] = int(statistics.median(ep_lens)) if ep_lens else 0
        trow["exec_start_idx_median"] = int(statistics.median(demo_prefix)) if demo_prefix else None
        report["tasks"][task] = trow

    report["overall"] = {}
    for c in SUBGOAL_COLS:
        if not all_counts[c]:
            continue
        counts = sorted(all_counts[c])
        report["overall"][c] = {
            "episodes_audited": len(counts),
            "total_segments": grand_segments[c],
            "segments_per_ep": {
                "mean": round(statistics.mean(counts), 2),
                "median": counts[len(counts) // 2],
                "p10": counts[int(0.10 * (len(counts) - 1))],
                "p90": counts[int(0.90 * (len(counts) - 1))],
                "min": counts[0],
                "max": counts[-1],
            },
            "seg_len_frames_median": int(statistics.median(all_lens[c])) if all_lens[c] else 0,
            "seg_len_frames_p10": sorted(all_lens[c])[int(0.10 * (len(all_lens[c]) - 1))],
            "empty_segments": all_empty[c],
            "empty_frac": round(all_empty[c] / max(grand_segments[c], 1), 4),
            "words_per_segment_mean": round(statistics.mean(all_words[c]), 2) if all_words[c] else 0,
            "distinct_strings": len(vocab[c]),
            "extrapolated_segments_1600_eps": round(statistics.mean(counts) * 1600),
        }
    report["overall"]["frames_audited"] = grand_frames
    report["eyeball"] = eyeball

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"store_facts": report["store_facts"], "overall": report["overall"]}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
