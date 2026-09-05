"""WSM label stage A0: dump subsampled RoboCasa target-episode frames (3 views).

Native 256x256 LeRobot frames — the same geometry the frozen GR00T backbone tokenizes
(see geometry.py). Ported from Isaac-GR00T/wsm/vlm_label/extract_frames.py (DexJoCo
front+wrist) to RoboCasa365 agentview_left/right + eye_in_hand. No sim privilege — frames
come straight from the dataset videos.

DECODING IS DIRECT PyAV over the episode mp4s — deliberately NOT lerobot's video stack. On the
SageMaker node BOTH lerobot backends are unusable: torchcodec imports but cannot dlopen FFmpeg
(no system libs, no apt egress), and the pyav backend routes through torchvision.io.VideoReader,
which this torchvision build removed. openpi pins av==14.2.0 exactly because its manylinux wheel
bundles FFmpeg, so `av` decodes self-contained; frame CONTENT is decoder-independent (same H.264
stream through the same FFmpeg codecs). Metadata (episode lengths, prompts, layout) comes straight
from the lerobot meta/ JSON files — no lerobot import at all.

Resolve the task's lerobot dir via utils.soup (needs robocasa.macros.DATASET_BASE_PATH set to
the dir holding v1.0/{pretrain,target}), or pass --lerobot-dir explicitly.

Run in an env with robocasa + av (the openpi env):
  python -m workspace_models.labels.extract_frames --task <Task> \
      --episodes 0,1,2,3,4 --stride 4 --out ~/Research/TRI/wsm_data/wsm_vlm_rc_v0
  python -m workspace_models.labels.extract_frames --lerobot-dir <abs lerobot dir> \
      --task <Task> --episodes 0,1,2,3,4 --out ~/Research/TRI/wsm_data/wsm_vlm_rc_v0

Output per episode: <out>/<task>/ep{idx:03d}_frames.npz
  frames_<view>  [K,256,256,3] uint8   for view in agentview_left/right, eye_in_hand
  frame_indices  [K] int64             (frame index within the episode)
  n_frames       int64                 (episode length)
  prompt         str                   (LeRobot task string)
  views          str (json list)       (the view order, for downstream consumers)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.subsample import episode_index_keep_set
from workspace_models.labels.geometry import VIEW_LEROBOT_KEY, VIEWS


def resolve_lerobot_dir(task: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    from utils.soup import combined_target_soup  # imports robocasa; needs DATASET_BASE_PATH

    soup = combined_target_soup(demo_fraction=1.0)
    metas = [m for m in soup if m["task"] == task]
    if not metas:
        raise SystemExit(f"task {task!r} not in the target soup. Available: {sorted({m['task'] for m in soup})}")
    return Path(metas[0]["path"]).expanduser()


def load_episode_meta(root: Path) -> tuple[dict[int, dict], dict]:
    """episodes.jsonl + info.json -> ({episode_index: {length, prompt}}, info)."""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes: dict[int, dict] = {}
    with open(root / "meta" / "episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tasks = rec.get("tasks") or [""]
            episodes[int(rec["episode_index"])] = {
                "length": int(rec["length"]),
                "prompt": str(tasks[0]),
            }
    return episodes, info


def episode_video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
    template = info["video_path"]  # e.g. videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
    chunk = episode_index // int(info.get("chunks_size", 1000))
    return root / template.format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)


def decode_frames_at(path: Path, indices: np.ndarray, expected_len: int) -> list[np.ndarray]:
    """Sequentially decode one mp4 with PyAV and return RGB frames at the requested indices.

    Sequential full decode (no seeking): exact frame indices with zero keyframe-snapping risk, and
    these episodes are short (a few hundred frames at 256x256). Fails loud if the stream is shorter
    than the metadata's episode length.
    """
    import av

    wanted = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        position = 0
        for frame in container.decode(stream):
            if position in wanted:
                out[position] = frame.to_ndarray(format="rgb24")
            position += 1
    if position < expected_len:
        raise SystemExit(
            f"{path}: decoded only {position} frames, metadata says {expected_len} — refusing silent truncation"
        )
    missing = sorted(wanted - set(out))
    if missing:
        raise SystemExit(f"{path}: frames {missing[:5]}... not produced by the decoder")
    return [out[int(i)] for i in indices]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--lerobot-dir", default=None, help="explicit lerobot root (else resolved via utils.soup)")
    ap.add_argument("--episodes", default="0,1,2,3,4", help="episode_index list (ignored if --num-demos)")
    ap.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="select the seed-0 filter_key keep-set (IDENTICAL to the policy finetune's "
        "150_demos selection) instead of --episodes",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for the --num-demos keep-set (policy uses 0)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = resolve_lerobot_dir(args.task, args.lerobot_dir)
    episodes_meta, info = load_episode_meta(root)
    for view in VIEWS:
        if VIEW_LEROBOT_KEY[view] not in info.get("features", {}):
            raise SystemExit(
                f"{root}: missing video key {VIEW_LEROBOT_KEY[view]}; "
                f"has {sorted(k for k in info.get('features', {}) if 'images' in k)}"
            )

    if args.num_demos is not None:
        keep = episode_index_keep_set(root, args.num_demos, args.seed)
        eps = sorted(int(e) for e in (keep if keep is not None else episodes_meta))
        print(
            f"[{args.task}] seed-{args.seed} filter_key keep-set: {len(eps)} episodes "
            f"(num_demos={args.num_demos}) — matches the policy finetune",
            flush=True,
        )
    else:
        eps = [int(e) for e in args.episodes.split(",")]
    out_dir = Path(args.out).expanduser() / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in eps:
        meta = episodes_meta.get(ep)
        if meta is None:
            print(f"[{args.task}] ep{ep:03d}: NOT FOUND — skip", flush=True)
            continue
        n = meta["length"]
        sel = np.arange(0, n, args.stride, dtype=np.int64)
        if sel[-1] != n - 1:
            sel = np.append(sel, n - 1)  # always include the final frame
        frames = {}
        for view in VIEWS:
            path = episode_video_path(root, info, VIEW_LEROBOT_KEY[view], ep)
            decoded = decode_frames_at(path, sel, n)
            frames[view] = np.stack([np.asarray(f, dtype=np.uint8) for f in decoded])
        np.savez_compressed(
            out_dir / f"ep{ep:03d}_frames.npz",
            frame_indices=sel,
            n_frames=np.int64(n),
            prompt=str(meta["prompt"]),
            views=json.dumps(list(VIEWS)),
            **{f"frames_{v}": frames[v] for v in VIEWS},
        )
        print(
            f"[{args.task}] ep{ep:03d}: {len(sel)}/{n} frames @stride{args.stride} prompt={meta['prompt']!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
