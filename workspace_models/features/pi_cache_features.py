"""Feature stage (pi0.5): cache frozen pi0.5 backbone features per demo for the pi WSM.

Mirrors cache_features.py (groot) but uses the pi05 tap (pi_backbone_tap.Pi05BackboneTap) and the pi
schema. pi0.5 discretizes the robot STATE into the prompt, so there is NO separate state_emb — the
per-frame language embedding (lang_per_frame) carries the per-frame state and serves as the WSM fuse
input (the proprio-replacement). Reuses the SAME seed-0 extracted frames + Qwen subgoals as groot.

  <cache_root>/<task>/demo_<ep>/
    patch_tokens.npy  [F,192,2048] fp16   (bin-averaged SigLIP grids, LABEL view order via the tap reorder)
    feats.npz         lang_per_frame [F,2048] fp16  (per-frame expanded-prompt+state -> WSM fuse input),
                      lang_global [2048] fp16, subgoal_embs [n,2048] fp16,
                      frame_indices [F] int64, keyframes [K] int64, expanded_prompt (str)
    .done_features

Run in the openpi-jax-latest env (jax + openpi + robocasa). See pi_backbone_tap.py for the tap + the
critical model-slot->label view reorder.

  DATASET_BASE_PATH=.../robocasa/datasets WSM_SOUP_FROM_DIRS=.../robocasa/datasets PYTHONPATH=<repo> \
    CUDA_VISIBLE_DEVICES=0 ~/Research/envs/openpi-jax-latest/bin/python -m \
    workspace_models.features.pi_cache_features --task OpenDrawer --frames-dir ~/Research/TRI/wsm_data/wsm_vlm_rc \
    --ckpt ~/Research/TRI/wsm_data/wsm_ckpts/pi05_on/149999 --cache-root ~/Research/TRI/wsm_data/wsm_cache_pi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workspace_models.labels.geometry import VIEWS


def _state_at(lerobot_dir: Path, ep: int, frame_indices: np.ndarray) -> np.ndarray:
    """observation.state for one episode at the subsampled frame_indices -> [F, D] (same as groot)."""
    import pandas as pd

    df = pd.read_parquet(lerobot_dir / f"data/chunk-000/episode_{ep:06d}.parquet")
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    return state[frame_indices]


def cache_task(tap, task: str, frames_dir: str, cache_root: str, lerobot_dir, batch_size: int, limit: int) -> None:
    if lerobot_dir is None:
        from utils.soup import combined_target_soup

        metas = [m for m in combined_target_soup(demo_fraction=1.0) if m["task"] == task]
        if not metas:
            print(f"[{task}] not in target soup — skip", flush=True)
            return
        lerobot_dir = Path(metas[0]["path"]).expanduser()
    task_frames = Path(frames_dir).expanduser() / task
    out_task = Path(cache_root).expanduser() / task

    fps = sorted(task_frames.glob("ep*_frames.npz"))
    if limit:
        fps = fps[:limit]
    for fp in fps:
        ep = int(fp.name[2:5])
        out_dir = out_task / f"demo_{ep:06d}"
        if (out_dir / ".done_features").exists():
            continue
        sj = fp.with_name(fp.name.replace("_frames.npz", "_subgoals.json"))
        if not sj.exists():
            print(f"[{task}] ep{ep:03d}: no subgoals.json — skip", flush=True)
            continue
        d = np.load(fp, allow_pickle=True)
        framesets = {v: d[f"frames_{v}"] for v in VIEWS}
        fidx = d["frame_indices"].astype(np.int64)
        sub = json.loads(sj.read_text())
        expanded = str(sub.get("expanded_prompt", d["prompt"]))
        subgoals = sub.get("subgoals", [])
        state = _state_at(lerobot_dir, ep, fidx)  # [F, D_state] (tap pads to action_dim)
        F = len(fidx)

        # --- per-frame patch tokens + lang (EXPANDED prompt; lang carries the per-frame state) ---
        patches, langs = [], []
        for lo in range(0, F, batch_size):
            hi = min(lo + batch_size, F)
            frames = {v: framesets[v][lo:hi] for v in VIEWS}
            r = tap.tap(frames, state[lo:hi], expanded)
            patches.append(r.patch_tokens)  # [b,192,2048] fp16, LABEL view order
            langs.append(r.lang_emb)  # [b,2048] fp16
        patch_tokens = np.concatenate(patches)  # [F,192,2048]
        lang_per_frame = np.concatenate(langs)  # [F,2048]
        lang_global = lang_per_frame.astype(np.float32).mean(0).astype(np.float16)  # [2048] global fallback

        # --- per-subgoal language emb (subgoal-keyframe frame + the subgoal name) ---
        subgoal_embs = []
        for sg in subgoals:
            kf = int(sg.get("completion_frame", fidx[0]))
            pos = int(np.argmin(np.abs(fidx - kf)))
            frames = {v: framesets[v][pos : pos + 1] for v in VIEWS}
            r = tap.tap(frames, state[pos : pos + 1], str(sg.get("name", "")))
            subgoal_embs.append(r.lang_emb[0])
        subgoal_embs = np.stack(subgoal_embs) if subgoal_embs else np.zeros((0, patch_tokens.shape[-1]), np.float16)
        keyframes = np.asarray([int(sg.get("completion_frame", 0)) for sg in subgoals], dtype=np.int64)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "patch_tokens.npy", patch_tokens)
        np.savez_compressed(
            out_dir / "feats.npz",
            lang_per_frame=lang_per_frame,
            lang_global=lang_global,
            subgoal_embs=subgoal_embs,
            frame_indices=fidx,
            keyframes=keyframes,
            expanded_prompt=np.str_(expanded),
        )
        (out_dir / ".done_features").write_text("ok")
        print(
            f"[{task}] ep{ep:03d}: F={F} patch_tokens={patch_tokens.shape} "
            f"lang_per_frame={lang_per_frame.shape} n_subgoals={len(subgoal_embs)} -> {out_dir}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated task names (one model load for all)")
    ap.add_argument("--frames-dir", required=True, help="extract_frames/qwen output root (<root>/<task>/ep*)")
    ap.add_argument("--lerobot-dir", default=None, help="LeRobot task dir (single-task only; else via utils.soup)")
    ap.add_argument("--ckpt", required=True, help="frozen pi05 checkpoint dir (params+assets)")
    ap.add_argument("--config", default="pi05_rc_mg60_bal33")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--batch-size", type=int, default=16, help="frames per backbone forward")
    ap.add_argument("--limit", type=int, default=0, help="cap demos per task (0 = all; for smoke tests)")
    args = ap.parse_args()

    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tap = Pi05BackboneTap(args.ckpt, args.config)  # load the 2B model ONCE for all tasks
    for task in tasks:
        ld = Path(args.lerobot_dir).expanduser() if (args.lerobot_dir and len(tasks) == 1) else None
        cache_task(tap, task, args.frames_dir, args.cache_root, ld, args.batch_size, args.limit)
        print(f"[pi-cache] {task}: done", flush=True)


if __name__ == "__main__":
    main()
