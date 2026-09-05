#!/usr/bin/env python3
"""Assemble replay payloads into one LeRobot v2.1 dataset in the openpi-LIBERO flavour.

``replay_shard.py`` writes per-case payloads (2 mp4 + arrays.npz + episode.json). This script
stitches them into a single dataset directory whose column names are exactly the ones
``openpi.training.config.LeRobotLiberoDataConfig`` repacks:

    image -> observation/image      wrist_image -> observation/wrist_image
    state -> observation/state      actions     -> actions      (+ prompt from task_index)

so the released ``pi05_libero`` checkpoint can be post-trained on RoboCerebra with **no**
transform edits: 2 cameras, 8-d state, 7-d action, ``prompt_from_task=True``.

Two RoboCerebra-specific columns ride along because our mechanisms need them and nothing
downstream in openpi touches unknown columns:

* ``subtask_index``       -- which annotated subtask each frame belongs to (omega-history,
                             segment-conditioned targets, per-subtask eval slicing).
* ``source_frame_index``  -- index back into the unfiltered raw demo, so no-op filtering is
                             reversible and frame ranges stay comparable with the authors'
                             ``[start, end]`` annotations.

Per-frame ``task_index`` points at the **subtask** instruction (this is what the benchmark's
own protocol feeds the low-level policy: the planner emits one subtask string at a time).
``global_task_index`` points at the episode-level ``Task:`` line in the same tasks table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

FPS = 20
CAMERAS = ("image", "wrist_image")
IMAGE_STAT_SAMPLES = 100
STATE_DIM = 8
ACTION_DIM = 7


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def video_feature(size: int) -> dict:
    return {
        "dtype": "video",
        "shape": [size, size, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(FPS),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
        "info": {
            "video.height": size,
            "video.width": size,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": float(FPS),
            "video.channels": 3,
            "has_audio": False,
        },
    }


def build_info(*, total_episodes: int, total_frames: int, total_tasks: int, size: int) -> dict:
    features = {camera: video_feature(size) for camera in CAMERAS}
    features.update(
        {
            "state": {"dtype": "float32", "shape": [STATE_DIM], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [ACTION_DIM], "names": ["actions"]},
            "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
            "global_task_index": {"dtype": "int64", "shape": [1], "names": None},
            "source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": (total_episodes + 999) // 1000,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def column_stats(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def video_stats(path: Path) -> dict:
    """Per-channel stats over <=100 uniformly sampled frames, normalised to [0, 1]."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"unreadable video: {path}")
    wanted = set(np.linspace(0, total - 1, min(IMAGE_STAT_SAMPLES, total)).round().astype(int).tolist())
    frames, position = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if position in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        position += 1
    capture.release()
    stack = np.stack(frames).astype(np.float64) / 255.0
    reshape = lambda a: a.reshape(3, 1, 1).tolist()  # noqa: E731
    return {
        "min": reshape(stack.min(axis=(0, 1, 2))),
        "max": reshape(stack.max(axis=(0, 1, 2))),
        "mean": reshape(stack.mean(axis=(0, 1, 2))),
        "std": reshape(stack.std(axis=(0, 1, 2))),
        "count": [len(frames)],
    }


def write_parquet(path: Path, columns: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    length = len(columns["frame_index"])
    table = pa.table(
        {
            "state": pa.FixedSizeListArray.from_arrays(
                pa.array(columns["state"].reshape(-1), type=pa.float32()), STATE_DIM
            ),
            "actions": pa.FixedSizeListArray.from_arrays(
                pa.array(columns["actions"].reshape(-1), type=pa.float32()), ACTION_DIM
            ),
            "subtask_index": pa.array(columns["subtask_index"], type=pa.int64()),
            "global_task_index": pa.array(columns["global_task_index"], type=pa.int64()),
            "source_frame_index": pa.array(columns["source_frame_index"], type=pa.int64()),
            "timestamp": pa.array((np.arange(length) / FPS).astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "index": pa.array(columns["index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payloads", required=True, help="root written by replay_shard.py")
    parser.add_argument("--out", required=True, help="LeRobot dataset directory to create")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--move-videos", action="store_true", help="move instead of copy the payload mp4s (saves a full data copy)"
    )
    args = parser.parse_args()

    payload_root = Path(args.payloads)
    episodes = sorted(payload_root.glob("*/*/_episode/episode.json"))
    if not episodes:
        raise SystemExit(f"no payloads under {payload_root}")
    out = Path(args.out)
    for sub in ("data", "videos", "meta"):
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}

    def task_id(text: str) -> int:
        return tasks.setdefault(text, len(tasks))

    episodes_jsonl, episodes_stats_jsonl, provenance = [], [], []
    running_index = 0
    total_frames = 0

    for episode_index, meta_path in enumerate(episodes):
        payload = meta_path.parent
        meta = json.loads(meta_path.read_text())
        arrays = np.load(payload / "arrays.npz")
        state, actions = arrays["state"], arrays["actions"]
        subtask_index, source_frame_index = arrays["subtask_index"], arrays["source_frame_index"]
        length = len(state)
        if not (len(actions) == len(subtask_index) == len(source_frame_index) == length):
            raise ValueError(f"{payload}: ragged arrays")

        subtask_texts = [entry["text"] for entry in meta["subtasks"]]
        subtask_ids = np.array([task_id(text) for text in subtask_texts], dtype=np.int64)
        per_frame_task = subtask_ids[subtask_index]
        global_id = task_id(meta["task_line"] or meta["language_instruction"])

        chunk = episode_index // 1000
        write_parquet(
            out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet",
            {
                "state": state,
                "actions": actions,
                "subtask_index": subtask_index.astype(np.int64),
                "global_task_index": np.full(length, global_id, dtype=np.int64),
                "source_frame_index": source_frame_index.astype(np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(running_index, running_index + length, dtype=np.int64),
                "task_index": per_frame_task,
            },
        )

        stats = {
            "state": column_stats(state),
            "actions": column_stats(actions),
            "subtask_index": column_stats(subtask_index),
            "timestamp": column_stats(np.arange(length) / FPS),
            "frame_index": column_stats(np.arange(length)),
            "episode_index": column_stats(np.full(length, episode_index)),
            "index": column_stats(np.arange(running_index, running_index + length)),
            "task_index": column_stats(per_frame_task),
        }
        for camera in CAMERAS:
            destination = out / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{episode_index:06d}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = payload / f"{camera}.mp4"
            (shutil.move if args.move_videos else shutil.copy2)(str(source), str(destination))
            stats[camera] = video_stats(destination)

        episodes_jsonl.append(
            {
                "episode_index": episode_index,
                "tasks": sorted(set(subtask_texts)),
                "length": length,
            }
        )
        episodes_stats_jsonl.append({"episode_index": episode_index, "stats": stats})
        provenance.append(
            {
                "episode_index": episode_index,
                "scene": meta["scene"],
                "case": meta["case"],
                "bddl_file": meta["bddl_file"],
                "distractor": meta["distractor"],
                "task_line": meta["task_line"],
                "num_subtasks": len(subtask_texts),
                "length": length,
                "dropped_subtasks": meta.get("dropped_subtasks", 0),
                "num_raw_frames": meta["num_raw_frames"],
                "dropped_noop_frames": meta["dropped_noop_frames"],
                "actions_out_of_unit_range": meta["actions_out_of_unit_range"],
                "actions_clipped": meta["actions_clipped"],
            }
        )
        running_index += length
        total_frames += length
        if (episode_index + 1) % 50 == 0:
            log(f"{episode_index + 1}/{len(episodes)} episodes assembled")

    meta_dir = out / "meta"
    with (meta_dir / "episodes.jsonl").open("w") as stream:
        for entry in episodes_jsonl:
            stream.write(json.dumps(entry) + "\n")
    with (meta_dir / "episodes_stats.jsonl").open("w") as stream:
        for entry in episodes_stats_jsonl:
            stream.write(json.dumps(entry) + "\n")
    with (meta_dir / "tasks.jsonl").open("w") as stream:
        for text, index in sorted(tasks.items(), key=lambda kv: kv[1]):
            stream.write(json.dumps({"task_index": index, "task": text}) + "\n")
    (meta_dir / "episode_provenance.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in provenance))

    # Dataset-level stats: exact for the numeric columns (re-read from the per-episode blocks
    # via count-weighted pooling), sampled for the video columns.
    global_stats: dict[str, dict] = {}
    for key in list(episodes_stats_jsonl[0]["stats"]):
        counts = np.array([e["stats"][key]["count"][0] for e in episodes_stats_jsonl], dtype=np.float64)
        means = np.array([e["stats"][key]["mean"] for e in episodes_stats_jsonl], dtype=np.float64)
        stds = np.array([e["stats"][key]["std"] for e in episodes_stats_jsonl], dtype=np.float64)
        weights = (counts / counts.sum()).reshape(-1, *([1] * (means.ndim - 1)))
        mean = (means * weights).sum(axis=0)
        var = ((stds**2 + (means - mean) ** 2) * weights).sum(axis=0)
        global_stats[key] = {
            "min": np.min([e["stats"][key]["min"] for e in episodes_stats_jsonl], axis=0).tolist(),
            "max": np.max([e["stats"][key]["max"] for e in episodes_stats_jsonl], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(counts.sum())],
        }
    # Exact q01/q99 for state/actions -- openpi normalises with quantiles, and pooled moments
    # cannot recover them, so re-read the two cheap numeric columns in full.
    import pyarrow.parquet as pq

    pooled = {"state": [], "actions": []}
    for episode_index in range(len(episodes)):
        chunk = episode_index // 1000
        table = pq.read_table(
            out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet", columns=["state", "actions"]
        )
        for key in pooled:
            pooled[key].append(np.stack(table[key].to_numpy(zero_copy_only=False)))
    for key, blocks in pooled.items():
        stacked = np.concatenate(blocks).astype(np.float64)
        global_stats[key]["q01"] = np.quantile(stacked, 0.01, axis=0).tolist()
        global_stats[key]["q99"] = np.quantile(stacked, 0.99, axis=0).tolist()
    (meta_dir / "stats.json").write_text(json.dumps(global_stats, indent=4))

    info = build_info(
        total_episodes=len(episodes), total_frames=total_frames, total_tasks=len(tasks), size=args.image_size
    )
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    log(f"wrote {len(episodes)} episodes / {total_frames} frames / {len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
