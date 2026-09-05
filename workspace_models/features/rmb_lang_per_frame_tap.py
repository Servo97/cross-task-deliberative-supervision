#!/usr/bin/env python3
"""Recover the PER-FRAME tap language embedding for the ReMemBench corpus.

Why this exists. `pi_pooled_tap` computes `lang_per_frame [F,2048]` on its way to `p.npz` and then
throws it away, keeping only the episode mean (`lang_global`). That was the right call at the time —
the whole point of the fused tap was to keep intermediates off disk — but it makes serve convention
(b) (a CAUSAL RUNNING MEAN of the per-frame language, `lang_t = mean(lang_tap[0..t])`) untestable
against the shipped ω, because the running mean cannot be reconstructed from its own final value.
No ReMemBench per-frame store exists anywhere: 0 `feats.npz` under the study prefix in S3.

This re-runs the SAME frozen tap on the SAME frame grid and writes ONLY the language stream. It does
not re-pool (the `p` in `wsm_pooled/rmb_pi_100k` is already the sanctioned artefact and is reused
untouched), so nothing here can perturb an existing store — it writes to its own root.

The correctness claim this file must earn: `mean(lang_per_frame)` under the RoboCasa order
(`float32 mean -> fp16`) must reproduce the `lang_global` already stored in `p.npz`, and the frame
grid must match `frame_indices` exactly. Both are asserted per episode, and a mismatch is fatal
rather than warned — if the re-tap is not the original tap, convention (b) would be measured against
the wrong reference.

  WSM_CONFIGS_DIR=~/Research/TRI/internal_training/robocasa PYTHONPATH=<repo> \
  CUDA_VISIBLE_DEVICES=0 ~/Research/envs/openpi-jax-latest/bin/python -m \
    workspace_models.features.rmb_lang_per_frame_tap \
      --dataset-root ~/Research/TRI/wsm_data/remembench_v02/train \
      --pooled-root  ~/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k \
      --ckpt         ~/Research/TRI/wsm_data/local_ckpts/pi05_on_149999 \
      --out-root     ~/Research/TRI/wsm_data/wsm_pooled/rmb_lang_pf
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from workspace_models.features.pi_pooled_tap import (
    decode_frames_at,
    episode_video_path,
    frame_selection,
    load_episode_meta,
    resolve_lerobot_dir,
    state_at,
)
from workspace_models.labels.geometry import VIEW_LEROBOT_KEY, VIEWS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument(
        "--pooled-root",
        required=True,
        help="existing rmb_pi_100k store; defines the episode set AND the reference "
        "frame grid / lang_global every episode is checked against",
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_rc_mg60_bal33")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

    dataset_root = Path(args.dataset_root).expanduser()
    pooled_root = Path(args.pooled_root).expanduser()
    out_root = Path(args.out_root).expanduser()

    want = [t.strip() for t in args.tasks.split(",") if t.strip()]
    jobs = []
    for task_dir in sorted(pooled_root.iterdir()):
        if not task_dir.is_dir() or (want and task_dir.name not in want):
            continue
        root = resolve_lerobot_dir(dataset_root, task_dir.name)
        if root is None:
            raise SystemExit(f"[tap] cannot resolve LeRobot dir for {task_dir.name}")
        ep_meta, info = load_episode_meta(root)
        demos = sorted(d for d in task_dir.iterdir() if (d / "p.npz").exists())
        if args.limit:
            demos = demos[: args.limit]
        for demo in demos:
            ep = int(demo.name.split("_")[-1])
            meta = ep_meta.get(ep)
            if meta is None:
                raise SystemExit(f"[tap] {task_dir.name}/ep{ep} present in pooled store, absent from episode meta")
            jobs.append((task_dir.name, ep, root, info, meta, demo / "p.npz"))

    todo = [j for j in jobs if not (out_root / j[0] / f"demo_{j[1]:06d}" / ".done_lang").exists()]
    print(f"[tap] {len(todo)} episodes to do / {len(jobs) - len(todo)} already done", flush=True)
    if not todo:
        return

    tap = Pi05BackboneTap(str(Path(args.ckpt).expanduser()), args.config)
    print(f"[tap] backbone={Path(args.ckpt).expanduser().name} batch={args.batch_size}", flush=True)

    t_start, n_frames_total = time.time(), 0
    for i, (task, ep, root, info, meta, p_path) in enumerate(todo):
        n = int(meta["length"])
        sel = frame_selection(n)
        ref = np.load(p_path)
        if not np.array_equal(sel.astype(np.int64), np.asarray(ref["frame_indices"], np.int64)):
            raise SystemExit(
                f"[tap] FRAME GRID MISMATCH for {task}/ep{ep}: re-tap selected "
                f"{len(sel)} frames, stored grid has {len(ref['frame_indices'])}"
            )
        prompt = str(meta["prompt"])
        frames = {v: decode_frames_at(episode_video_path(root, info, VIEW_LEROBOT_KEY[v], ep), sel, n) for v in VIEWS}
        state = state_at(root, info, ep, sel)

        F = len(sel)
        langs = []
        for lo in range(0, F, args.batch_size):
            hi = min(lo + args.batch_size, F)
            pad = args.batch_size - (hi - lo)  # keep ONE compiled XLA shape, as the pooled tap does
            if pad:
                sl = {v: np.concatenate([frames[v][lo:hi], np.repeat(frames[v][hi - 1 : hi], pad, 0)]) for v in VIEWS}
                st = np.concatenate([state[lo:hi], np.repeat(state[hi - 1 : hi], pad, 0)])
            else:
                sl, st = {v: frames[v][lo:hi] for v in VIEWS}, state[lo:hi]
            res = tap.tap(sl, st, prompt)
            langs.append(res.lang_emb[: hi - lo] if pad else res.lang_emb)

        lang_pf = np.concatenate(langs)  # [F,2048] fp16
        if not np.isfinite(lang_pf.astype(np.float32)).all():
            raise SystemExit(f"[tap] NON-FINITE lang for {task}/ep{ep}")

        # The claim that makes this store usable: the re-tap IS the original tap.
        mine = lang_pf.astype(np.float32).mean(0).astype(np.float16).astype(np.float32)
        stored = np.asarray(ref["lang_global"], dtype=np.float32)
        if not np.array_equal(mine, stored):
            delta = float(np.abs(mine - stored).max())
            cos = float((mine @ stored) / max(np.linalg.norm(mine) * np.linalg.norm(stored), 1e-12))
            raise SystemExit(
                f"[tap] LANG_GLOBAL MISMATCH for {task}/ep{ep}: max|Δ|={delta:.3e} "
                f"cos={cos:.8f} — the re-tap does not reproduce the stored tap; "
                f"refusing to write a reference convention (b) would be scored against"
            )

        out_dir = out_root / task / f"demo_{ep:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "lang.npz", lang_per_frame=lang_pf, frame_indices=sel.astype(np.int64), lang_global=stored)
        (out_dir / ".done_lang").touch()
        n_frames_total += F
        if i % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t_start
            print(
                f"[tap] {i + 1}/{len(todo)} {task}/ep{ep} F={F} | cum {n_frames_total} frames, "
                f"{n_frames_total / max(el, 1e-9):.1f} fr/s, eta "
                f"{(len(todo) - i - 1) / max((i + 1) / max(el, 1e-9), 1e-9) / 60:.1f} min",
                flush=True,
            )

    el = time.time() - t_start
    print(f"[tap] COMPLETE: {len(todo)} episodes, {n_frames_total} frames in {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
