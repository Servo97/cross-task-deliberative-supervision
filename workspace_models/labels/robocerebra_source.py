"""RoboCerebra frame source for pass 1 — the fourth domain, +8,869 segments (~31% of the corpus).

RoboCerebra's sealed `robocerebra_train_v1` LeRobot tree is an ordinary MP4 tree (unlike RoboMME,
whose frames are parquet-embedded bytes), so `caption_segments.decode_views`' PyAV path applies
almost unchanged. What this module supplies is job enumeration, the segmentation, and a 2-view
decode that reads `job.views` instead of the module-level 3-view RoboCasa constant.

Decisions, each read off the dataset rather than assumed:

* **Segmentation = runs of the per-frame `subtask_index` column.** FREE and exact — the same
  "official column" route RoboMME took. Three int columns are present and two of them are traps:
    - `subtask_index`      episode-LOCAL ordinal 0..n-1   -> THE segmentation
    - `task_index`         LeRobot global id of the SUBTASK string  -> the per-segment hint
    - `global_task_index`  LeRobot global id of the EPISODE task line (constant per episode)
  Segmenting on `task_index` silently merges two adjacent subtasks that share a string, and
  `meta/episodes.jsonl["tasks"]` is a de-duplicated SET, not temporal order. Verified over all
  994 episodes: runs are strictly increasing with no revisits, and cover `[0, T)`.
* **8,869 segments, not the 8,887 the case definitions declare.** Eight episodes are demos
  truncated short of their declared subtask list (ep 983 even starts at subtask 1). Nothing is
  corrupt — `dropped_subtasks` is 0 everywhere and simply does not measure this — so the segments
  are taken from what is actually in the episode. Listed in `TRUNCATED_EPISODES` so the count is
  never re-litigated.
* **Hint = the per-segment subtask instruction** (`tasks[task_index]`), which is exactly the string
  the policy is served when the harness re-pins that subtask. `instruction` = the episode task line.
* **2 views** (`image`, `wrist_image`), both [256,256,3] at 20 fps — LIBERO geometry. 256 px is
  Qwen's `shortest_edge`, so frames pass through unresized, as for RoboCasa and RoboMME. The 2-view
  prompt sha is therefore shared with RoboMME by construction, not by coincidence.

Task identity = the BDDL stem. RoboCerebra's 994 training episodes cover 947 distinct BDDL files:
it is a one-episode-per-task corpus, not RoboCasa's 13 x 150. That is the honest label, and it is
also the consequential one — it makes pass 2's `within_task` stratum nearly empty and `cross_task`
the whole domain, which HELPS the G-E quota floors (cross-task-or-domain >= 0.40) rather than
gaming them. Collapsing to the 3 scenes would relabel genuinely different tasks as "within task".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = "~/Research/TRI/wsm_data/robocerebra/lerobot_home/wsmv2/robocerebra_train"

VIEWS = ("front", "wrist")
VIEW_LEROBOT_KEY = {"front": "image", "wrist": "wrist_image"}
CAPTION = {"front": "front", "wrist": "wrist"}
FPS = 20.0

#: Demos that stop before their case's last declared subtask (or, for 983, start at subtask 1).
#: Together they account for the 8,887 - 8,869 = 18 segment difference. Data, not corruption.
TRUNCATED_EPISODES = (18, 138, 277, 428, 575, 635, 983, 990)


def task_name(bddl_file: str) -> str:
    """`<BDDL stem>` — already filesystem-safe (uppercase scene prefix + underscores)."""
    return bddl_file[:-5] if bddl_file.endswith(".bddl") else bddl_file


def episode_path(root: Path, ep: int, info: dict) -> Path:
    chunk = ep // int(info.get("chunks_size", 1000))
    return root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)


def video_path(root: Path, ep: int, view: str, info: dict) -> Path:
    chunk = ep // int(info.get("chunks_size", 1000))
    return root / info["video_path"].format(episode_chunk=chunk, video_key=VIEW_LEROBOT_KEY[view], episode_index=ep)


def runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of a per-frame integer column -> [(t0, t1)). THIS is the segmentation."""
    if len(values) == 0:
        return []
    cuts = np.flatnonzero(np.diff(values) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(values)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def build_index(root: Path, cache_path: Path) -> dict:
    """Per-episode {task, n_frames, segments, hints, instruction}, cached.

    Built once and reused: eight shards each re-reading 994 parquet files would be pure waste, and
    the index is all job construction needs (frames are decoded lazily).
    """
    import pandas as pd

    root = Path(root).expanduser()
    if cache_path is not None and Path(cache_path).is_file():
        try:
            blob = json.loads(Path(cache_path).read_text())
            if blob.get("_root") == str(root):
                return {k: v for k, v in blob.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001 — a bad cache is rebuilt, never fatal
            pass

    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {
        int(r["task_index"]): str(r["task"])
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

    index = {}
    for ep in sorted(prov):
        df = pd.read_parquet(episode_path(root, ep, info), columns=["subtask_index", "task_index"])
        sti = df["subtask_index"].to_numpy()
        ti = df["task_index"].to_numpy()
        segs = runs(sti)
        index[str(ep)] = {
            "task": task_name(prov[ep]["bddl_file"]),
            "n_frames": int(len(df)),
            "segments": [[a, b] for a, b in segs],
            "hints": [tasks.get(int(ti[a]), "") for a, _ in segs],
            "instruction": str(prov[ep]["task_line"]),
            "scene": prov[ep]["scene"],
            "case": prov[ep]["case"],
        }
    if cache_path is not None:
        # ATOMIC. All 8 shard clients start together and each misses the cache, so all 8 build the
        # index and all 8 write this same path concurrently. A plain write_text is not atomic: a
        # reader can observe a truncated file and every later shard would then fail to parse it.
        # Write a per-writer temp beside it and os.replace, which is atomic on POSIX -- concurrent
        # writers then produce byte-identical content and last-one-wins is harmless.
        import os

        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({**index, "_root": str(root)}))
        os.replace(tmp, cp)
    return index


def decode_views(job) -> object:
    """Fill job.frames for the planned frames from the 2 MP4 views.

    Deliberately not delegated to `caption_segments.decode_views`: that function iterates the
    module-level 3-view RoboCasa `VIEWS` and its `VIEW_LEROBOT_KEY`, so a 2-view LIBERO episode
    would send it looking for a third stream that does not exist.
    """
    import av

    wanted = sorted({int(f) for p in job.plan for f in p})
    try:
        info = json.loads((job.root / "meta" / "info.json").read_text())
        stop = max(wanted)
        for view in job.views or VIEWS:
            path = video_path(job.root, job.ep, view, info)
            got, need = {}, set(wanted)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                pos = 0
                for frame in container.decode(stream):
                    if pos in need:
                        got[pos] = frame.to_ndarray(format="rgb24")
                        need.discard(pos)
                        if not need:
                            break
                    if pos > stop:
                        break
                    pos += 1
            if need:
                raise RuntimeError(f"{path.name}: frames {sorted(need)[:5]} not decoded")
            job.frames[view] = got
    except Exception as e:  # noqa: BLE001
        job.error = f"decode: {e}"
    return job
