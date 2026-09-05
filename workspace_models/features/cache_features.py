"""Feature stage: cache frozen GR00T backbone features per demo (the raw-patch-token cache).

Reuses the extracted 3-view frames (labels/extract_frames.py) + the Qwen `expanded_prompt`
(labels/qwen_subgoals.py) and runs the FROZEN backbone once per subsampled frame. Writes a
self-contained, mmap-friendly per-demo record (doc 07 schema):

  <cache_root>/<task>/demo_<ep>/
    patch_tokens.npy  [F,192,2048] fp16   (mmap; the heavy part)
    feats.npz         state_emb [F,1536] fp16, lang_global [2048] fp16,
                      subgoal_embs [n_sub,2048] fp16, frame_indices [F] int64,
                      keyframes [K] int64, expanded_prompt (str)
    .done_features

Recon TARGET features are NOT pre-stored: the data loader gathers them at train time from
patch_tokens[keyframe_pos, global_patch_id] (the labels' salient_global ids), so the cache stays
minimal + flexible. NEEDS the GR00T env + a GR00T checkpoint (frozen pretrain). On Blackwell GPUs
use a flash-attn-free GR00T install (sdpa) — see [[blackwell-b200-eval-path]].

  python -m workspace_models.features.cache_features --task OpenDrawer \
      --frames-dir ~/Research/TRI/wsm_data/wsm_vlm_rc --ckpt <groot ckpt-150000 dir> \
      --cache-root ~/Research/TRI/wsm_data/wsm_cache --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workspace_models.labels.geometry import VIEWS


def _state_at(lerobot_dir: Path, ep: int, frame_indices: np.ndarray) -> np.ndarray:
    """Read observation.state for one episode and select the subsampled frame_indices -> [F, D]."""
    import pandas as pd

    df = pd.read_parquet(lerobot_dir / f"data/chunk-000/episode_{ep:06d}.parquet")
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # [T_full, D]
    return state[frame_indices]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--frames-dir", required=True, help="extract_frames/qwen output root (<root>/<task>/ep*)")
    ap.add_argument("--lerobot-dir", default=None, help="LeRobot task dir (else resolved via utils.soup)")
    ap.add_argument("--ckpt", required=True, help="frozen GR00T checkpoint dir (e.g. .../checkpoint-150000)")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16, help="frames per backbone forward")
    args = ap.parse_args()

    import sys

    import torch

    # Blackwell (sm_100 B200 / sm_120 RTX 5090): the prebuilt flash-attn 2.7.4 wheel has no kernels
    # for these arches and crashes at launch; block its import BEFORE gr00t loads so GR00T's
    # qwen3_backbone falls back to torch sdpa (Blackwell-native). No-op on Hopper/Ampere.
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
        sys.modules["flash_attn"] = None
        print("[cache_features] Blackwell GPU detected -> flash_attn disabled (sdpa)", flush=True)

    from workspace_models.features.backbone_tap import load_tap

    if args.lerobot_dir:
        lerobot_dir = Path(args.lerobot_dir).expanduser()
    else:
        from utils.soup import combined_target_soup

        metas = [m for m in combined_target_soup(demo_fraction=1.0) if m["task"] == args.task]
        if not metas:
            raise SystemExit(f"task {args.task!r} not in target soup")
        lerobot_dir = Path(metas[0]["path"]).expanduser()

    tap = load_tap(args.ckpt, embodiment_tag=args.embodiment_tag, device=args.device)
    # GR00T modality uses 'robot0_<view>' keys; our frames are saved under geometry's short VIEWS.
    # Map each (ordered) modality video key -> the short view; the order MUST match geometry.VIEW_OFFSETS
    # (so cached patch_tokens [...,0:64]=left, 64:128=right, 128:192=eye_in_hand align with the labels).
    video_keys = list(tap.policy.modality_configs["video"].modality_keys)
    vk_to_view = {vk: next(v for v in VIEWS if vk.endswith(v)) for vk in video_keys}
    if [vk_to_view[vk] for vk in video_keys] != list(VIEWS):
        raise SystemExit(
            f"view order mismatch: modality {video_keys} vs geometry VIEWS {VIEWS} "
            "(patch_tokens layout would not match the salient_global ids)"
        )
    print(f"[cache_features] video_keys={video_keys}", flush=True)
    task_frames = Path(args.frames_dir).expanduser() / args.task
    out_task = Path(args.cache_root).expanduser() / args.task

    for fp in sorted(task_frames.glob("ep*_frames.npz")):
        ep = int(fp.name[2:5])
        out_dir = out_task / f"demo_{ep:06d}"
        if (out_dir / ".done_features").exists():
            continue
        sj = fp.with_name(fp.name.replace("_frames.npz", "_subgoals.json"))
        if not sj.exists():
            print(f"[{args.task}] ep{ep:03d}: no subgoals.json — skip (run qwen first)", flush=True)
            continue
        d = np.load(fp, allow_pickle=True)
        framesets = {v: d[f"frames_{v}"] for v in VIEWS}
        fidx = d["frame_indices"].astype(np.int64)
        sub = json.loads(sj.read_text())
        expanded = str(sub.get("expanded_prompt", d["prompt"]))
        subgoals = sub.get("subgoals", [])
        state = _state_at(lerobot_dir, ep, fidx)  # [F, D_state]
        F = len(fidx)

        # --- per-frame backbone features (batched) with the EXPANDED prompt as text ---
        patches, state_embs, lang_global = [], [], None
        for lo in range(0, F, args.batch_size):
            hi = min(lo + args.batch_size, F)
            images = {vk: framesets[vk_to_view[vk]][lo:hi] for vk in video_keys}
            r = tap.tap(tap.obs_from_frames(images, state[lo:hi], expanded))
            patches.append(r.patch_tokens.to(torch.float16).cpu().numpy())
            state_embs.append(r.state_emb[:, 0].to(torch.float16).cpu().numpy())
            if lang_global is None:
                lang_global = r.lang_emb[0].to(torch.float16).cpu().numpy()
        patch_tokens = np.concatenate(patches)  # [F,192,2048] fp16
        state_emb = np.concatenate(state_embs)  # [F,1536] fp16

        # --- per-subgoal language emb (frame at the subgoal keyframe + the subgoal name) ---
        subgoal_embs = []
        for sg in subgoals:
            kf = int(sg.get("completion_frame", fidx[0]))
            pos = int(np.argmin(np.abs(fidx - kf)))
            images = {vk: framesets[vk_to_view[vk]][pos : pos + 1] for vk in video_keys}
            r = tap.tap(tap.obs_from_frames(images, state[pos : pos + 1], str(sg.get("name", ""))))
            subgoal_embs.append(r.lang_emb[0].to(torch.float16).cpu().numpy())
        subgoal_embs = np.stack(subgoal_embs) if subgoal_embs else np.zeros((0, patch_tokens.shape[-1]), np.float16)
        keyframes = np.asarray([int(sg.get("completion_frame", 0)) for sg in subgoals], dtype=np.int64)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "patch_tokens.npy", patch_tokens)  # mmap target
        np.savez_compressed(
            out_dir / "feats.npz",
            state_emb=state_emb,
            lang_global=lang_global,
            subgoal_embs=subgoal_embs,
            frame_indices=fidx,
            keyframes=keyframes,
            expanded_prompt=np.str_(expanded),
        )
        (out_dir / ".done_features").write_text("ok")
        print(
            f"[{args.task}] ep{ep:03d}: F={F} patch_tokens={patch_tokens.shape} "
            f"n_subgoals={len(subgoal_embs)} -> {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
