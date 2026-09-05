"""WSM label stage A0 for RoboCerebra (LIBERO, 2 views): one frames npz per SUBTASK SEGMENT.

Sibling of extract_frames.py (RoboCasa365), not a replacement — the two datasets differ in the
only two places this stage touches: where episodes come from and what a "unit to label" is.

  * RoboCasa resolves its lerobot dir through utils.soup + a filter_key keep-set. RoboCerebra is one
    flat LeRobot v2.1 dir, selected by explicit --episodes (and this pilot picks them by scene from
    meta/episode_provenance.jsonl).
  * A RoboCasa episode is ONE atomic task, so Qwen decomposes the whole episode. A RoboCerebra
    episode is a ~900-frame, 8-11 step composite, and the dataset already ships GROUND-TRUTH
    segmentation: the per-frame `subtask_index` column plus `task_index` -> meta/tasks.jsonl gives
    each segment's own instruction. Throwing that away and asking Qwen for 4-6 subgoals over 900
    frames would be strictly worse labels. So each SEGMENT is emitted as its own
    `ep{seg:03d}_frames.npz` with `prompt` = the subtask instruction, and stages A/B/C then run
    UNCHANGED (they see short single-task "episodes" exactly like RoboCasa's). `frame_indices` and
    the keyframes downstream stay EPISODE-GLOBAL, so labels re-attach to the source episode.

Decoding is direct PyAV over the mp4s, reusing extract_frames.decode_frames_at verbatim (see that
module's docstring for why not lerobot's video stack). The episode is decoded ONCE per view and
sliced into segments.

TWO ENVS, because no single local env has both readers: the segment table needs pyarrow (parquet)
and the decode needs av. Run the manifest pass in the numpy/pyarrow env with --no-decode, then the
decode pass in the vlm env with --segments-json:

  ~/miniconda3/envs/ogpo2/bin/python -m workspace_models.labels.extract_frames_robocerebra \
      --lerobot-dir <root> --episodes 89,875,946 --out <OUT> --no-decode
  ~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.extract_frames_robocerebra \
      --lerobot-dir <root> --episodes 89,875,946 --out <OUT> --stride 4 \
      --segments-json <OUT>/_segments

Output per (episode, segment): <out>/<scene>_<case>_ep<ep>/ep{seg:03d}_frames.npz
  frames_agentview / frames_eye_in_hand  [K,256,256,3] uint8
  frame_indices  [K] int64   EPISODE-GLOBAL frame indices
  n_frames       int64       EPISODE length (not the segment's)
  prompt         str         the SUBTASK instruction
  views          str (json list)
  episode_index, subtask_index, seg_start, seg_end, scene, case, task_line
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from workspace_models.labels.extract_frames import episode_video_path

GEOM_MODULES = {"groot": "geometry", "pi": "pi_geometry", "pi_libero": "pi_geometry_libero"}


def case_tag(prov: dict) -> str:
    """Per-episode output dir name; doubles as the --task argument of stages A/B/C."""
    return f"{prov['scene']}_{prov['case']}_ep{int(prov['episode_index']):03d}"


def read_meta(root: Path) -> tuple[dict, dict[int, str], dict[int, dict]]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = {}
    with open(root / "meta" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                tasks[int(rec["task_index"])] = str(rec["task"])
    prov = {}
    with open(root / "meta" / "episode_provenance.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                prov[int(rec["episode_index"])] = rec
    return info, tasks, prov


def segments_from_parquet(root: Path, info: dict, ep: int, tasks: dict[int, str]) -> list[dict]:
    """Contiguous runs of `subtask_index` -> [{seg, subtask_index, start, end, prompt}] (needs pyarrow).

    Fails loud if a run spans more than one task_index: that would mean the segmentation and the
    instruction disagree, and every label in that segment would carry the wrong prompt.
    """
    import pyarrow.parquet as pq

    path = root / info["data_path"].format(episode_chunk=ep // int(info.get("chunks_size", 1000)), episode_index=ep)
    tbl = pq.read_table(path, columns=["subtask_index", "task_index"])
    st = np.asarray(tbl.column("subtask_index").to_numpy(), dtype=np.int64).reshape(-1)
    ti = np.asarray(tbl.column("task_index").to_numpy(), dtype=np.int64).reshape(-1)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(st)) + 1, [len(st)]])
    segs = []
    for s in range(len(bounds) - 1):
        a, b = int(bounds[s]), int(bounds[s + 1])
        uniq = np.unique(ti[a:b])
        if len(uniq) != 1:
            raise SystemExit(
                f"ep{ep}: subtask run [{a},{b}) spans task_index {uniq.tolist()} — segmentation and "
                "instruction disagree; refusing to label it with an arbitrary prompt"
            )
        segs.append(
            {
                "seg": s,
                "subtask_index": int(st[a]),
                "start": a,
                "end": b - 1,
                "task_index": int(uniq[0]),
                "prompt": tasks[int(uniq[0])],
            }
        )
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-dir", required=True)
    ap.add_argument("--episodes", required=True, help="comma list of episode_index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="frame stride WITHIN a segment")
    ap.add_argument("--geom", choices=sorted(GEOM_MODULES), default="pi_libero")
    ap.add_argument(
        "--segments-json",
        default="",
        help="dir holding ep{idx:06d}.json segment manifests (from a --no-decode pass); "
        "if empty the parquet is read directly (needs pyarrow)",
    )
    ap.add_argument(
        "--no-decode", action="store_true", help="only write the segment manifests (env with pyarrow, no av)"
    )
    args = ap.parse_args()

    G = importlib.import_module("workspace_models.labels." + GEOM_MODULES[args.geom])
    VIEWS, KEY = G.VIEWS, G.VIEW_LEROBOT_KEY
    root = Path(args.lerobot_dir).expanduser()
    out = Path(args.out).expanduser()
    info, tasks, prov = read_meta(root)
    for v in VIEWS:
        if KEY[v] not in info.get("features", {}):
            raise SystemExit(
                f"{root}: missing video key {KEY[v]!r}; has "
                f"{sorted(k for k, f in info['features'].items() if f.get('dtype') == 'video')}"
            )

    seg_dir = Path(args.segments_json).expanduser() if args.segments_json else (out / "_segments")
    seg_dir.mkdir(parents=True, exist_ok=True)
    eps = [int(e) for e in args.episodes.split(",") if e.strip()]

    for ep in eps:
        p = prov.get(ep)
        if p is None:
            raise SystemExit(f"episode {ep} not in meta/episode_provenance.jsonl")
        tag = case_tag(p)
        seg_path = seg_dir / f"ep{ep:06d}.json"
        if args.segments_json and seg_path.exists():
            segs = json.loads(seg_path.read_text())["segments"]
        else:
            segs = segments_from_parquet(root, info, ep, tasks)
            seg_path.write_text(
                json.dumps(
                    {
                        "episode_index": ep,
                        "case_tag": tag,
                        "length": int(p["length"]),
                        "scene": p["scene"],
                        "case": p["case"],
                        "task_line": p["task_line"],
                        "segments": segs,
                    },
                    indent=2,
                )
            )
        n = int(p["length"])
        if segs[-1]["end"] != n - 1:
            raise SystemExit(f"ep{ep}: segments end at {segs[-1]['end']} but length is {n}")
        print(f"[ep{ep:03d}] {tag}: {len(segs)} segments, {n} frames", flush=True)
        if args.no_decode:
            continue

        # frames to keep, episode-global; the LAST frame of a segment is always kept (it is the
        # moment of completion Qwen is asked to point at).
        keep_per_seg = []
        for s in segs:
            sel = np.arange(s["start"], s["end"] + 1, args.stride, dtype=np.int64)
            if len(sel) == 0 or sel[-1] != s["end"]:
                sel = np.append(sel, s["end"])
            keep_per_seg.append(sel)
        wanted = np.unique(np.concatenate(keep_per_seg))

        from workspace_models.labels.extract_frames import decode_frames_at

        decoded = {}
        for view in VIEWS:
            vpath = episode_video_path(root, info, KEY[view], ep)
            frames = decode_frames_at(vpath, wanted, n)
            decoded[view] = {int(i): np.asarray(f, dtype=np.uint8) for i, f in zip(wanted, frames)}

        ep_out = out / tag
        ep_out.mkdir(parents=True, exist_ok=True)
        for s, sel in zip(segs, keep_per_seg):
            np.savez_compressed(
                ep_out / f"ep{s['seg']:03d}_frames.npz",
                frame_indices=sel,
                n_frames=np.int64(n),
                prompt=str(s["prompt"]),
                views=json.dumps(list(VIEWS)),
                episode_index=np.int64(ep),
                subtask_index=np.int64(s["subtask_index"]),
                seg_start=np.int64(s["start"]),
                seg_end=np.int64(s["end"]),
                scene=str(p["scene"]),
                case=str(p["case"]),
                task_line=str(p["task_line"]),
                **{f"frames_{v}": np.stack([decoded[v][int(i)] for i in sel]) for v in VIEWS},
            )
            print(
                f"  seg{s['seg']:02d} [{s['start']}..{s['end']}] {len(sel)} frames prompt={s['prompt']!r}", flush=True
            )


if __name__ == "__main__":
    main()
