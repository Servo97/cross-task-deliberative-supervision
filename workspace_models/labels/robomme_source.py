"""RoboMME frame source for pass 1 — the A5(i) unlock, ~8,740 segments (45% of the corpus).

RoboMME's LeRobot snapshot has **no MP4s** (`total_videos: 0`): frames are encoded image bytes
embedded in the parquet as `{"bytes": ..., "path": ...}` cells, so `caption_segments`' PyAV path
does not apply. This module supplies the two things that path provided — job enumeration and frame
decode — and nothing else changes.

Decisions, each grounded in the 2026-08-22 subgoal audit (`robomme_subgoal_audit.json`):

* **Segmentation = RLE of `simple_subgoal`.** Verified contiguous and covering `[0, T)` on 80/80
  episodes with 0 empty segments — the same contract `segments_from_keyframes` guarantees for
  RoboCasa. Mean 5.46 segments/episode.
* **Hint = `grounded_subgoal`, not `simple_subgoal`.** Same boundaries, but 310 distinct strings vs
  81: `grounded` carries the object binding, `simple` often does not.
* **2 views, not 3.** The store has exactly `image` (front) and `wrist_image`, both [256,256,3] —
  resolved by reading the store rather than trusting either scout. 256 px is exactly Qwen's
  `shortest_edge`, so frames are passed through without resize, same as RoboCasa.
* **`PatternLock` is flagged `low_confidence_language`** downstream (pass2_deliberate): its
  per-step subgoals degenerate to bare directions ("move right") with no object binding, so its
  RLE gives usable BOUNDARIES but an unusable hint.

Task identity is the pinned 100-episodes-per-task block order from
`robomme_integration/training/single_task.py`; it is deliberately not inferred from language,
because several tasks share instruction text.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

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
FPS = 10.0

VIEWS = ("front", "wrist")
VIEW_COLUMN = {"front": "image", "wrist": "wrist_image"}
CAPTION = {"front": "front", "wrist": "wrist"}

SEG_COLUMN = "simple_subgoal"  # boundaries
HINT_COLUMN = "grounded_subgoal"  # object binding


def task_of(ep: int) -> str:
    return TASK_ORDER[ep // EPISODES_PER_TASK]


def episodes_of(task: str) -> range:
    i = TASK_ORDER.index(task)
    return range(i * EPISODES_PER_TASK, (i + 1) * EPISODES_PER_TASK)


def episode_path(root: Path, ep: int) -> Path:
    return root / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"


def rle(values: list) -> list:
    """Per-step strings -> [(t0, t1)) segments. THIS is the segmentation."""
    if not values:
        return []
    out, t0 = [], 0
    for t in range(1, len(values) + 1):
        if t == len(values) or values[t] != values[t0]:
            out.append((t0, t))
            t0 = t
    return out


def _instruction(root: Path, ep: int, cache: dict) -> str:
    if "ep_task" not in cache:
        tasks = {}
        tp = root / "meta" / "tasks.jsonl"
        if tp.is_file():
            for line in tp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    tasks[int(r["task_index"])] = str(r["task"])
        ep_task = {}
        epp = root / "meta" / "episodes.jsonl"
        if epp.is_file():
            for line in epp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    t = r.get("tasks") or []
                    ep_task[int(r["episode_index"])] = str(t[0]) if t else ""
        cache["ep_task"] = ep_task
        cache["tasks"] = tasks
    return cache["ep_task"].get(ep, "")


def build_index(root: Path, tasks: list, cache_path: Path) -> dict:
    """Per-episode {n_frames, segments, hints, instruction}, cached.

    Built once and reused: eight shards each re-reading 1,600 parquet files would be pure waste,
    and the index is the only thing job construction needs (frames are read lazily at decode time).
    """
    import pyarrow.parquet as pq

    cached = {}
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = {}
    meta_cache: dict = {}
    unreadable: list = []
    changed = False
    for task in tasks:
        for ep in episodes_of(task):
            key = str(ep)
            if key in cached:
                continue
            p = episode_path(root, ep)
            if not p.is_file():
                continue
            try:
                tbl = pq.read_table(p, columns=[SEG_COLUMN, HINT_COLUMN])
            except Exception as e:  # noqa: BLE001
                # 33 of the 1,600 snapshot files are unreadable ("Couldn't deserialize thrift"),
                # all inside the ButtonUnmaskSwap block. A corrupt episode must never take a shard
                # down: record it and carry on, so the damage is a countable gap in the manifest
                # rather than a crashed run.
                unreadable.append(
                    {
                        "episode": ep,
                        "task": task,
                        "bytes": p.stat().st_size,
                        "error": f"{type(e).__name__}: {str(e)[:120]}",
                    }
                )
                continue
            simple = ["" if v is None else str(v) for v in tbl.column(SEG_COLUMN).to_pylist()]
            grounded = ["" if v is None else str(v) for v in tbl.column(HINT_COLUMN).to_pylist()]
            segs = rle(simple)
            cached[key] = {
                "task": task,
                "n_frames": len(simple),
                "segments": [[a, b] for a, b in segs],
                "hints": [grounded[a] if a < len(grounded) else "" for a, _ in segs],
                "instruction": _instruction(root, ep, meta_cache),
            }
            changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached))
    if unreadable:
        rep = cache_path.parent / "_robomme_unreadable.json"
        rep.write_text(json.dumps(unreadable, indent=1))
        print(f"[robomme] {len(unreadable)} UNREADABLE parquet episodes skipped (recorded in {rep})", flush=True)
    return cached


def decode_views(job) -> object:
    """Fill job.frames for the planned frames, decoding the parquet's embedded image bytes."""
    import pyarrow.parquet as pq
    from PIL import Image

    wanted = sorted({int(f) for p in job.plan for f in p})
    try:
        tbl = pq.read_table(job.root, columns=[VIEW_COLUMN[v] for v in VIEWS])
        n = tbl.num_rows
        for view in VIEWS:
            col = tbl.column(VIEW_COLUMN[view])
            got = {}
            for f in wanted:
                if f >= n:
                    raise RuntimeError(f"frame {f} beyond {n} rows in {job.root.name}")
                cell = col[f].as_py()
                raw = cell["bytes"] if isinstance(cell, dict) else cell
                got[f] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            job.frames[view] = got
    except Exception as e:  # noqa: BLE001
        job.error = f"decode: {e}"
    return job
