#!/usr/bin/env python3
"""Video renderer: replay a recorded state trajectory and composite every camera view.

Runs as its own pass over ``traj.npz`` / the expert ``states.npz``, never inside the
rollout, so (a) rollout throughput is not charged for pixels, (b) resolution and camera set
are free parameters, and (c) a policy rollout and its expert reference are rendered by
identical code into identical frames.

Layout: the three cameras the POLICY actually sees (agentview_left, agentview_right,
eye_in_hand) plus agentview_center as a human-readable overview, tiled 2x2 at 640x480 each
(1280x960 total, i.e. comfortably >=480p per view). A caption bar carries task / arm /
reset / outcome so a video is self-identifying once it has left this directory.

Timing: the env runs at 20 Hz. Frames are kept every ``--stride`` control steps and written
at ``20 / stride`` fps, so playback is always REAL TIME regardless of stride.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_env  # noqa: E402
from fm_common import VIDEO_CAMERAS, VIDEO_H, VIDEO_W, cell_name  # noqa: E402

ENV_HZ = 20


def caption(frame_width: int, lines, height: int = 56):
    import cv2

    bar = np.zeros((height, frame_width, 3), dtype=np.uint8)
    for index, text in enumerate(lines[:2]):
        cv2.putText(
            bar,
            text,
            (12, 24 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return bar


def tile(frames, cameras, cell_w: int, cell_h: int):
    import cv2

    tiles = []
    for image, name in zip(frames, cameras):
        cell = image
        if cell.shape[0] != cell_h or cell.shape[1] != cell_w:
            cell = cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cell = np.ascontiguousarray(cell)
        cv2.putText(
            cell,
            name.replace("robot0_", ""),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 40),
            1,
            cv2.LINE_AA,
        )
        tiles.append(cell)
    while len(tiles) % 2:
        tiles.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
    rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)]
    return np.concatenate(rows, axis=0)


def encode(frames, path: str, fps: float, crf: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    tmp = f"{path}.tmp.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        tmp,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    os.replace(tmp, path)


def render_one(env, core, bench, extras_dir, seed, states, out_path, lines, stride, crf, cameras, cell_w, cell_h):
    fm_env.reset_to_demo(env, extras_dir, seed=int(seed), bench=bench)
    present = fm_env.available_cameras(core, cameras)
    if not present:
        raise RuntimeError("no requested camera exists in this model")
    frames = []
    for index in range(0, len(states), stride):
        fm_env.set_sim_state(core, states[index])
        views = fm_env.render_views(core, present, cell_w, cell_h)
        composite = tile(views, present, cell_w, cell_h)
        frames.append(np.concatenate([caption(composite.shape[1], lines), composite], axis=0))
    if not frames:
        raise RuntimeError("no frames rendered")
    encode(frames, out_path, fps=ENV_HZ / stride, crf=crf)
    return len(frames), present


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bench", choices=["remembench", "robocasa"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument(
        "--ckpt-label",
        required=True,
        help="checkpoint label, or the literal 'expert' to render the demonstrations",
    )
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--crf", type=int, default=26)
    parser.add_argument("--cell-w", type=int, default=VIDEO_W)
    parser.add_argument("--cell-h", type=int, default=VIDEO_H)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rollout-idx", type=int, default=0)
    args = parser.parse_args()

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    episodes = [e for e in manifest["episodes"] if e["task"] == args.task]
    episodes.sort(key=lambda e: (e["split"] != "heldout", int(e["episode_index"])))
    if args.limit:
        episodes = episodes[: args.limit]
    episodes = episodes[args.shard_idx :: args.num_shards]

    import robocasa  # noqa: F401

    env = fm_env.make_env(args.bench, args.task, seed=int(manifest["base_seed"]))
    core = env.unwrapped.env
    is_expert = args.ckpt_label == "expert"
    written = 0
    skipped = 0

    try:
        for episode in episodes:
            # Draw 0 keeps the bare reset id; extra draws get a __r<k> suffix. The expert is
            # per-RESET, not per-draw, so it is rendered once and shared.
            rid = cell_name(episode["reset_id"], 0 if is_expert else args.rollout_idx)
            out_path = os.path.join(args.video_root, args.bench, args.task, args.ckpt_label, f"{rid}.mp4")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 4096:
                skipped += 1
                continue
            if is_expert:
                states = np.load(episode["expert"]["states"])["states"]
                outcome = "EXPERT DEMONSTRATION"
            else:
                cell = os.path.join(args.out_root, "raw", args.bench, args.task, args.ckpt_label, rid)
                traj = os.path.join(cell, "traj.npz")
                result_path = os.path.join(cell, "result.json")
                if not (os.path.exists(traj) and os.path.exists(result_path)):
                    print(f"[render] no rollout for {rid}, skip", flush=True)
                    continue
                with np.load(traj) as data:
                    states = data["states"]
                with open(result_path) as handle:
                    result = json.load(handle)
                flag = "SUCCESS" if result.get("rollout__success") else "FAIL"
                if result.get("rollout__failed_task"):
                    flag = "FAIL(deadline)"
                outcome = f"{flag}  len={result.get('rollout__episode_length')}"
            lines = [
                f"{args.task}  |  {args.ckpt_label}  |  {rid} ({episode['split']})  |  {outcome}",
                (episode["expert"].get("lang") or "")[:130],
            ]
            count, present = render_one(
                env,
                core,
                args.bench,
                episode["reset"]["extras_dir"],
                episode["seed"],
                states,
                out_path,
                lines,
                args.stride,
                args.crf,
                list(VIDEO_CAMERAS),
                args.cell_w,
                args.cell_h,
            )
            written += 1
            print(
                f"[render] {args.task}/{args.ckpt_label}/{rid} frames={count} "
                f"cams={len(present)} size={os.path.getsize(out_path) / 1e6:.1f}MB",
                flush=True,
            )
    finally:
        env.close()
    print(f"[render] done: {written} written, {skipped} already present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
