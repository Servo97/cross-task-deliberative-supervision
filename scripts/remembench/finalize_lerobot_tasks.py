#!/usr/bin/env python3
"""Assemble per-task LeRobot v2.1 datasets from the sharded ReMemBench re-render payloads.

Reads ``<render-root>/<task>/_episodes/<ep:06d>/{arrays.npz,episode.json,<cam>.mp4}`` and emits
``<out>/<task>/<capture>/lerobot/{meta,data,videos}`` byte-compatible with the RoboCasa
``datasets/v1.0/target/**/lerobot`` trees (verified field-by-field against
``atomic/TurnOnMicrowave/20250813``).

Two things must be assembled centrally rather than per shard:

* ``index`` is a dataset-global running row counter (episode 1 of the reference dataset starts at
  115, right after episode 0's 115 rows), so it can only be assigned once every episode length is
  known.
* ``tasks.jsonl`` maps instruction strings to ``task_index``. RoboCasa target tasks have exactly
  one instruction per task; ReMemBench instructions are episode-specific (object names vary), so
  the unique instructions are enumerated in episode order as indices ``0..K-1`` and the task name
  is appended at index ``K`` -- which degenerates to the reference layout (instruction at 0, task
  name at 1) whenever a task has a single instruction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

CAMERAS = ("robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right")
FPS = 20
CHUNK = 0
IMAGE_STAT_SAMPLES = 100  # LeRobot samples at most 100 frames per episode for image stats

HF_SCHEMA_METADATA = json.dumps(
    {
        "info": {
            "features": {
                "annotation.human.task_description": {"dtype": "int64", "_type": "Value"},
                "annotation.human.task_name": {"dtype": "int64", "_type": "Value"},
                "observation.state": {
                    "feature": {"dtype": "float64", "_type": "Value"},
                    "length": 16,
                    "_type": "Sequence",
                },
                "action": {
                    "feature": {"dtype": "float64", "_type": "Value"},
                    "length": 12,
                    "_type": "Sequence",
                },
                "next.reward": {"dtype": "float32", "_type": "Value"},
                "next.done": {"dtype": "bool", "_type": "Value"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }
        }
    },
    separators=(", ", ": "),
)


def video_feature(size: int) -> dict:
    info = {
        "video.height": size,
        "video.width": size,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": FPS,
        "video.channels": 3,
        "has_audio": False,
    }
    return {
        "dtype": "video",
        "shape": [size, size, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": FPS,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
        "info": info,
    }


def build_info(*, total_episodes: int, total_frames: int, total_tasks: int, size: int) -> dict:
    features = {f"observation.images.{camera}": video_feature(size) for camera in CAMERAS}
    features.update(
        {
            "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
            "annotation.human.task_name": {"dtype": "int64", "shape": [1]},
            "observation.state": {"dtype": "float64", "shape": [16]},
            "action": {"dtype": "float64", "shape": [12]},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def column_stats(values: np.ndarray) -> dict:
    """min/max/mean/std/count for one non-image column, in LeRobot's per-episode layout."""
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
    """Per-channel image stats over <=100 uniformly sampled frames, normalised to [0, 1]."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"unreadable video: {path}")
    count = min(IMAGE_STAT_SAMPLES, total)
    wanted = set(np.linspace(0, total - 1, count).round().astype(int).tolist())
    # Sequential decode, keeping the wanted frames. Seeking with CAP_PROP_POS_FRAMES re-decodes
    # from the nearest keyframe on every call, which on the 2600-3200 frame MemHeatPot* episodes
    # costs more than decoding the whole clip once.
    frames = []
    position = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if position in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        position += 1
    capture.release()
    if not frames:
        raise ValueError(f"decoded no frames from {path}")
    stack = np.stack(frames).astype(np.float64) / 255.0  # [N,H,W,3]
    axes = (0, 1, 2)
    reshape = lambda a: a.reshape(3, 1, 1).tolist()  # noqa: E731
    return {
        "min": reshape(stack.min(axis=axes)),
        "max": reshape(stack.max(axis=axes)),
        "mean": reshape(stack.mean(axis=axes)),
        "std": reshape(stack.std(axis=axes)),
        "count": [len(frames)],
    }


def write_parquet(path: Path, columns: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    length = len(columns["frame_index"])
    table = pa.table(
        {
            "annotation.human.task_description": pa.array(columns["task_description"], type=pa.int64()),
            "annotation.human.task_name": pa.array(columns["task_name"], type=pa.int64()),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(columns["state"].reshape(-1), type=pa.float64()), 16
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(columns["action"].reshape(-1), type=pa.float64()), 12
            ),
            "next.reward": pa.array(columns["rewards"], type=pa.float32()),
            "next.done": pa.array(columns["dones"], type=pa.bool_()),
            "timestamp": pa.array((np.arange(length, dtype=np.float64) / FPS).astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "index": pa.array(columns["index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
        }
    )
    table = table.replace_schema_metadata({"huggingface": HF_SCHEMA_METADATA})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def finalize_task(
    task: str,
    render_root: Path,
    out_root: Path,
    capture: str,
    expected: list,
    size: int,
    modality: dict,
    embodiment: dict,
) -> dict:
    episodes_dir = render_root / task / "_episodes"
    records = []
    for entry in sorted(episodes_dir.iterdir()):
        meta_path = entry / "episode.json"
        if not meta_path.is_file():
            continue
        records.append(json.loads(meta_path.read_text()))
    records.sort(key=lambda record: record["episode_index"])
    indices = [record["episode_index"] for record in records]
    if indices != list(range(len(expected))):
        raise ValueError(f"{task}: episode indices {indices[:5]}... do not cover 0..{len(expected) - 1}")
    by_index = {record["episode_index"]: record for record in expected}
    for record in records:
        want = by_index[record["episode_index"]]
        if record["length"] != want["length"]:
            raise ValueError(
                f"{task} ep {record['episode_index']}: length {record['length']} != worklist {want['length']}"
            )
        if record["demo_key"] != want["demo_key"]:
            raise ValueError(f"{task} ep {record['episode_index']}: demo_key mismatch")

    # instruction -> task_index, in first-appearance order; task name last.
    instructions: list[str] = []
    for record in records:
        if record["lang"] not in instructions:
            instructions.append(record["lang"])
    task_name_index = len(instructions)

    root = out_root / task / capture / "lerobot"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    running = 0
    episodes_jsonl = []
    episodes_stats = []
    pooled: dict[str, list[np.ndarray]] = {}
    for record in records:
        episode_index = record["episode_index"]
        source = episodes_dir / f"{episode_index:06d}"
        arrays = np.load(source / "arrays.npz")
        length = record["length"]
        instruction_index = instructions.index(record["lang"])
        columns = {
            "state": arrays["state"],
            "action": arrays["action"],
            "rewards": arrays["rewards"],
            "dones": arrays["dones"],
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(running, running + length, dtype=np.int64),
            "task_index": np.full(length, instruction_index, dtype=np.int64),
            "task_description": np.full(length, instruction_index, dtype=np.int64),
            "task_name": np.full(length, task_name_index, dtype=np.int64),
        }
        running += length
        parquet_path = root / "data" / f"chunk-{CHUNK:03d}" / f"episode_{episode_index:06d}.parquet"
        write_parquet(parquet_path, columns)

        stats = {}
        for camera in CAMERAS:
            destination = (
                root
                / "videos"
                / f"chunk-{CHUNK:03d}"
                / f"observation.images.{camera}"
                / f"episode_{episode_index:06d}.mp4"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / f"{camera}.mp4", destination)
            stats[f"observation.images.{camera}"] = video_stats(destination)
        timestamps = (np.arange(length, dtype=np.float64) / FPS).astype(np.float32)
        scalar_columns = {
            "annotation.human.task_description": columns["task_description"],
            "annotation.human.task_name": columns["task_name"],
            "observation.state": columns["state"],
            "action": columns["action"],
            "next.reward": columns["rewards"],
            "next.done": columns["dones"],
            "timestamp": timestamps,
            "frame_index": columns["frame_index"],
            "episode_index": columns["episode_index"],
            "index": columns["index"],
            "task_index": columns["task_index"],
        }
        for name, values in scalar_columns.items():
            stats[name] = column_stats(values)
            array = np.asarray(values, dtype=np.float64)
            pooled.setdefault(name, []).append(array if array.ndim == 2 else array[:, None])
        episodes_stats.append({"episode_index": episode_index, "stats": stats})
        episodes_jsonl.append({"episode_index": episode_index, "tasks": [record["lang"]], "length": length})

    meta = root / "meta"
    with (meta / "episodes.jsonl").open("w") as stream:
        for entry in episodes_jsonl:
            stream.write(json.dumps(entry) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as stream:
        for entry in episodes_stats:
            stream.write(json.dumps(entry) + "\n")
    with (meta / "tasks.jsonl").open("w") as stream:
        for position, instruction in enumerate(instructions):
            stream.write(json.dumps({"task_index": position, "task": instruction}) + "\n")
        stream.write(json.dumps({"task_index": task_name_index, "task": task}) + "\n")

    global_stats = {}
    for name, chunks in pooled.items():
        stacked = np.concatenate(chunks, axis=0)
        global_stats[name] = {
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
            "min": stacked.min(axis=0).tolist(),
            "max": stacked.max(axis=0).tolist(),
            "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
            "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
        }
    (meta / "stats.json").write_text(json.dumps(global_stats, indent=4))
    info = build_info(
        total_episodes=len(records),
        total_frames=running,
        total_tasks=len(instructions),
        size=size,
    )
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))
    (meta / "embodiment.json").write_text(json.dumps(embodiment, indent=4))
    return {
        "task": task,
        "episodes": len(records),
        "frames": running,
        "instructions": len(instructions),
        "root": str(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--capture", default="20260803")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--modality", required=True, help="reference PandaOmron modality.json")
    parser.add_argument("--embodiment", required=True, help="reference PandaOmron embodiment.json")
    args = parser.parse_args()

    worklist = json.loads(Path(args.worklist).read_text())
    modality = json.loads(Path(args.modality).read_text())
    embodiment = json.loads(Path(args.embodiment).read_text())
    summary = []
    for task in worklist["tasks"]:
        result = finalize_task(
            task["task"],
            Path(args.render_root),
            Path(args.out),
            args.capture,
            task["episodes"],
            args.camera_size,
            modality,
            embodiment,
        )
        summary.append(result)
        print(
            f"{result['task']:32s} episodes={result['episodes']:3d} frames={result['frames']:6d} "
            f"instructions={result['instructions']}",
            flush=True,
        )
    total_episodes = sum(item["episodes"] for item in summary)
    total_frames = sum(item["frames"] for item in summary)
    print(f"TOTAL tasks={len(summary)} episodes={total_episodes} frames={total_frames}")
    if total_episodes != worklist["total_train_episodes"]:
        print("MISMATCH against worklist total", file=sys.stderr)
        sys.exit(1)
    Path(args.out, "_finalize_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
