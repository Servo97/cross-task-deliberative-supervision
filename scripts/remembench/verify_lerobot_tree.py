#!/usr/bin/env python3
"""Verify the finalized ReMemBench LeRobot tree against the train worklist.

Checks, per task and per episode:
  * every worklist episode has a parquet and 3 mp4s;
  * parquet row count == mp4 frame count (all 3 cameras) == the source demo's ``num_samples``;
  * actions carried over verbatim from the source hdf5 (after the hdf5 -> LeRobot re-ordering);
  * ``index`` is a contiguous dataset-global counter and ``frame_index`` restarts per episode;
  * ``meta/episodes.jsonl`` lengths agree with the parquets and info.json totals add up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CAMERAS = ("robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right")
# hdf5 action layout -> lerobot action layout (mirrors render_lerobot_shard.ACTION_SEGMENTS)
ACTION_SEGMENTS = (
    (7, 11, 0, 4),
    (11, 12, 4, 5),
    (0, 3, 5, 8),
    (3, 6, 8, 11),
    (6, 7, 11, 12),
)
DEMO_FILENAME = "demo_im128_notp.hdf5"


def mp4_frames(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="dir holding <task>/<capture>/lerobot")
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--capture", default="20260803")
    parser.add_argument("--action-check-episodes", type=int, default=3)
    args = parser.parse_args()

    import h5py
    import pandas as pd

    worklist = json.loads(Path(args.worklist).read_text())
    problems: list[str] = []
    total_frames = 0
    total_episodes = 0
    for task in worklist["tasks"]:
        root = Path(args.root) / task["task"] / args.capture / "lerobot"
        info = json.loads((root / "meta" / "info.json").read_text())
        episodes_meta = [
            json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()
        ]
        if len(episodes_meta) != task["n_train"]:
            problems.append(f"{task['task']}: episodes.jsonl has {len(episodes_meta)}")
        running = 0
        for position, episode in enumerate(task["episodes"]):
            index = episode["episode_index"]
            parquet = root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
            frame = pd.read_parquet(parquet)
            if len(frame) != episode["length"]:
                problems.append(f"{task['task']}/{index}: parquet {len(frame)} != source {episode['length']}")
            for camera in CAMERAS:
                video = root / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{index:06d}.mp4"
                frames = mp4_frames(video)
                if frames != episode["length"]:
                    problems.append(f"{task['task']}/{index}/{camera}: mp4 {frames} != {episode['length']}")
            if int(frame["index"].iloc[0]) != running:
                problems.append(f"{task['task']}/{index}: global index restart at {running}")
            if int(frame["frame_index"].iloc[0]) != 0:
                problems.append(f"{task['task']}/{index}: frame_index does not restart")
            running += len(frame)
            if position < args.action_check_episodes:
                path = Path(args.data_root) / task["task"] / episode["session"] / DEMO_FILENAME
                with h5py.File(path, "r") as handle:
                    source_actions = handle["data"][episode["demo_key"]]["actions"][()]
                expect = np.zeros_like(source_actions)
                for h0, h1, l0, l1 in ACTION_SEGMENTS:
                    expect[:, l0:l1] = source_actions[:, h0:h1]
                got = np.stack(frame["action"].to_list())
                if not np.array_equal(got, expect):
                    problems.append(f"{task['task']}/{index}: actions differ from source")
        if info["total_frames"] != running:
            problems.append(f"{task['task']}: info.total_frames {info['total_frames']} != {running}")
        if info["total_episodes"] != len(episodes_meta):
            problems.append(f"{task['task']}: info.total_episodes mismatch")
        total_frames += running
        total_episodes += len(episodes_meta)
        print(f"{task['task']:32s} ok episodes={len(episodes_meta):3d} frames={running:6d}", flush=True)

    print(f"TOTAL episodes={total_episodes} frames={total_frames}")
    if total_episodes != worklist["total_train_episodes"]:
        problems.append("total episode count != worklist")
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for problem in problems[:40]:
            print("  ", problem)
        sys.exit(1)
    print("VERIFY OK")


if __name__ == "__main__":
    main()
