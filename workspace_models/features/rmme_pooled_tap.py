#!/usr/bin/env python3
"""rmme_pooled_tap — FUSED frozen pi0.5 tap + frozen WSMv1 pool for RoboMME (4th Stage-E domain).

The RoboMME sibling of `pi_pooled_tap.py` (RoboCasa/ReMemBench) and `rcb_pooled_tap.py`
(RoboCerebra). Same product — a byte-compatible `wsm_pooled` `p.npz` store that `train_stage_e.py`
loads as one more `--tap` — plus TWO additive fields that only RoboMME needs (`is_demo`,
`exec_start_idx`), because a RoboMME episode is `[demonstration video ; execution]` in ONE frame
stream and every downstream demo-vs-live decision has to be able to recover that split.

Four things about RoboMME force a separate module rather than another `--domain` branch:

1. **No MP4s.** `meta/info.json` says `total_videos: 0`; frames are encoded image bytes embedded in
   the parquet as `{"bytes": ..., "path": ...}` cells (`workspace_models/labels/robomme_source.py`
   established this for pass 1). The PyAV `decode_frames_at` path in `pi_pooled_tap` does not apply,
   so decode goes through pyarrow + PIL on the same selected indices.

2. **Backbone — the SAME frozen network as robocasa/remembench, on purpose.** Every project RoboMME
   arm initialises from the RoboCasa H300+MG pretrain `pi05_on/149999`
   (`robomme_integration/launch.py:INIT_ROOT` = `pretrain150k/pi05/mg60_bal33/run/149999`;
   `ROBOMME_ALIGNMENT.md` "initialization | RoboCasa H300+MG step 149999"). Tapping that checkpoint
   means RoboMME does NOT introduce a third frozen network the way RoboCerebra's `pi05_libero` did:
   the A3 domain bridge has strictly less work to do here than it already passed for RoboCerebra.
   `Pi05BackboneTap` is imported rather than reimplemented, so RoboCasa/ReMemBench/RoboMME pooled
   tokens come from ONE tap definition (same 14x14 -> 8x8 binning, same masked-mean language pool).

3. **Geometry — RoboMME's exact policy input tree, three slots with the right wrist zeroed.**
   `robomme_integration/training/data.py::RoboMMEInputs` maps
   `observation/image -> base_0_rgb`, `observation/wrist_image -> left_wrist_0_rgb`, and fills
   `right_wrist_0_rgb` with `np.zeros_like(wrist_image)`. This tap reproduces that tree exactly, so
   the tapped tokens are the tokens the policy's own backbone sees, and the geometry stays
   RoboCasa's 3 x 64 = 192 bin-averaged patches -> the A3 audit is apples-to-apples against the
   other two pi05_on taps rather than against a 128-token store.
   APPROXIMATE-PARITY CELL, recorded rather than hidden: `RoboMMEInputs` sets
   `image_mask[right_wrist_0_rgb] = False` while `RobocasaInputs` (the tap's transform) sets all
   three True. The zero view is therefore attended here and masked in the policy. This is internally
   consistent — the SAME producer runs at train and at serve, which is the invariant that governs
   the omega contract — but it is not bitwise the policy's prefix.

4. **Prompt — one instruction per episode, constant.** RoboMME's LeRobot metadata carries 116
   instruction variants over 16 environment tasks and the instruction does not change within an
   episode (unlike RoboCerebra's per-subtask re-pin), so `lang_per_frame` is constant and
   `lang_global == lang_task_line` by construction. Both are still written so the schema matches the
   other taps field-for-field and `--lang-mode {episode_mean,task_mean,per_frame,stored}` all
   resolve to the same vector for this domain.

Frame grid — the `wsm_pooled` convention (`pi_pooled_tap.frame_selection`): stride 8 from 0 with the
final frame always appended, over the WHOLE episode. `step_idx` is episode-global and the
demonstration is the episode's own leading `exec_start_idx` rows, so ONE grid over `[0, len)` covers
demo and execution in one causal stream and no separate demo store is needed:

    demo omega  = omega[frame_indices <  exec_start_idx]
    live omega  = omega[frame_indices >= exec_start_idx]

768,897 frames -> 98,215 tapped (MEASURED over all 1,600 episodes: 36,907 demo + 61,308 live).

Contract, byte-compatible with `wsm_pooled/pi_100k/<Task>/demo_%06d/p.npz`:

    p              [F,512]  fp16    frozen pool over the 192 bin-averaged pi0.5 patches
    frame_indices  [F]      int64   stride 8, final frame always included
    lang_global    [2048]   float32 mean over frames of the masked-mean language embedding
    encoder_id     ()       str     "wsm_pool:<first 16 hex of the pool ckpt sha256>" (§32.4/32.7)
    pool_sha256    ()       str     full sha256 of the frozen pool checkpoint
  + backbone_id, prompt_source, lang_per_frame [F,2048] f16, lang_task_line [2048] f32
  + is_demo [F] bool, exec_start_idx (), n_frames_episode ()          <- RoboMME-only, additive
  + `.done_pooled` marker

`<Task>` is the pinned 100-episodes-per-task block name from
`robomme_integration/training/single_task.py::TASK_ORDER`, and the episode id is the GLOBAL
0..1,599 index — identical to the pass-1 descriptor keys
(`pass1_store/robomme/<Task>/ep_%06d.descriptors.json`), which is the join key
`train_stage_e.Corpus` uses (`<tap_root>/<task>/demo_%06d/p.npz`).

Run in the openpi-jax env (jax + openpi + torch + pyarrow + pillow):

  WSM_CONFIGS_DIR=~/Research/TRI/internal_training/robocasa PYTHONPATH=<repo> \
  CUDA_VISIBLE_DEVICES=0 ~/Research/envs/openpi-jax-latest/bin/python -m \
    workspace_models.features.rmme_pooled_tap \
      --dataset-root ~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/snapshots/1510653c... \
      --ckpt         ~/Research/TRI/wsm_data/local_ckpts/pi05_on_149999 \
      --pool-ckpt    ~/Research/TRI/wsm_data/wsm_runs/pi_wsm_v1/wsm_step100000.pt \
      --out-root     ~/Research/TRI/wsm_data/wsm_pooled/rmme_pi_100k

`--plan-only` runs every data path (meta, shard, resume gate, parquet read, frame selection, image
decode, demo split) and writes NOTHING and loads NO model — the CPU dry run for the node's argv.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.features.pi_pooled_tap import STRIDE, frame_selection  # noqa: E402
from workspace_models.features.rmme_demo_prefix import serve_aligned_grid  # noqa: E402
from workspace_models.labels.robomme_source import (  # noqa: E402
    TASK_ORDER,
    episode_path,
    episodes_of,
)

VIEW_COLUMNS = ("image", "wrist_image")  # RoboMME is 2-view; slot 3 is the policy's zero view
STATE_COLUMN = "state"  # [8] joint angles (RoboMME `action_space: joint_angle`)
DEFAULT_DATASET = (
    "~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
    "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20"
)
DEFAULT_TAP_CKPT = "~/Research/TRI/wsm_data/local_ckpts/pi05_on_149999"
TAP_CONFIG = "pi05_rc_mg60_bal33"
PAD_BATCH = 16  # one XLA kernel shape across every tap in this study


def load_meta(root: Path) -> tuple[dict, dict]:
    """(episode_index -> instruction, info.json)."""
    info = json.loads((root / "meta" / "info.json").read_text())
    ep_task: dict[int, str] = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            tasks = rec.get("tasks") or [""]
            ep_task[int(rec["episode_index"])] = str(tasks[0])
    return ep_task, info


def read_episode(path: Path, columns: tuple[str, ...]):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=list(columns))


def decode_rows(table, column: str, rows: np.ndarray) -> np.ndarray:
    """Decode the parquet's embedded image bytes at `rows` -> [K,H,W,3] uint8."""
    from PIL import Image

    col = table.column(column)
    n = table.num_rows
    out = []
    for index in rows:
        index = int(index)
        if index >= n:
            raise RuntimeError(f"frame {index} beyond {n} rows")
        cell = col[index].as_py()
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(f"RoboMME frame {index} is not HxWx3: {image.shape}")
        out.append(image)
    return np.stack(out)


def _scalar_column(table, name: str) -> np.ndarray:
    """RoboMME stores several scalars as length-1 lists; flatten to [T]."""
    values = table.column(name).to_pylist()
    flat = [v[0] if isinstance(v, (list, tuple)) else v for v in values]
    return np.asarray(flat)


GRID_MODES = ("wsm_pooled", "serve_aligned")


def episode_plan(table, grid_mode: str = "wsm_pooled") -> dict:
    """Frame grid + the demo/live split for one episode, from the parquet alone.

    `wsm_pooled`    stride 8 from 0 + the final frame — the cross-domain REPRESENTATION corpus
                    convention. Use for the Stage-E encoder tap (A3 comparability).
    `serve_aligned` arange(0,D,8) U arange(D,n,16), no trailing frame — the POLICY store grid.
                    Required because a serve frame is `exec_start_idx + 16k` and 16 == 0 (mod 8),
                    so on the 81.4 % of demo episodes whose `exec_start_idx % 8 != 0` NOT ONE live
                    frame lands on the stride-8 grid (measured over all 900). See
                    `rmme_demo_prefix.serve_aligned_grid`.
    """
    if grid_mode not in GRID_MODES:
        raise ValueError(f"grid_mode must be one of {GRID_MODES}, got {grid_mode!r}")
    n = table.num_rows
    if n < 1:
        raise RuntimeError("empty RoboMME episode")
    is_demo = _scalar_column(table, "is_demo").astype(bool)
    exec_start = _scalar_column(table, "exec_start_idx").astype(np.int64)
    if is_demo.shape != (n,) or exec_start.shape != (n,):
        raise RuntimeError(f"RoboMME scalar columns are not [{n}]")
    start = int(exec_start[0])
    sel = frame_selection(n) if grid_mode == "wsm_pooled" else serve_aligned_grid(n, start)
    if int(is_demo.sum()) != start:
        raise RuntimeError(f"is_demo count {int(is_demo.sum())} != exec_start_idx {start}")
    if start and not (is_demo[:start].all() and not is_demo[start:].any()):
        raise RuntimeError("RoboMME demo rows are not a contiguous leading prefix")
    return {
        "n_frames": n,
        "frame_indices": sel,
        "is_demo": is_demo[sel],
        "exec_start_idx": start,
        "grid_mode": grid_mode,
    }


def build_jobs(
    root: Path, tasks: tuple[str, ...], *, worker_idx: int, num_workers: int, limit: int
) -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    for task in tasks:
        eps = list(episodes_of(task))
        if limit:
            eps = eps[:limit]
        jobs.extend((task, ep) for ep in eps)
    jobs.sort()
    return jobs[worker_idx::num_workers]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--ckpt", default=DEFAULT_TAP_CKPT, help="frozen pi05_on/149999 checkpoint dir")
    ap.add_argument("--config", default=TAP_CONFIG)
    ap.add_argument("--pool-ckpt", default="", help="frozen WSMv1 ckpt (required unless --plan-only)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tasks", default="", help="comma list; default = all 16 in pinned order")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes per task (0 = all)")
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument(
        "--grid",
        default="wsm_pooled",
        choices=GRID_MODES,
        help="wsm_pooled = the Stage-E representation corpus (stride 8 + final frame); "
        "serve_aligned = the POLICY omega store grid the online producer can "
        "reproduce exactly",
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="exercise every data path with the node's exact argv; load no model, write no store",
    )
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip()) or TASK_ORDER
    unknown = [t for t in tasks if t not in TASK_ORDER]
    if unknown:
        raise SystemExit(f"unknown RoboMME task(s) {unknown}; expected {list(TASK_ORDER)}")
    if args.num_workers < 1 or not 0 <= args.worker_idx < args.num_workers:
        raise SystemExit("invalid shard: need 0 <= --worker-idx < --num-workers")

    ep_task, info = load_meta(root)
    jobs = build_jobs(root, tasks, worker_idx=args.worker_idx, num_workers=args.num_workers, limit=args.limit)
    todo = [j for j in jobs if not (out_root / j[0] / f"demo_{j[1]:06d}" / ".done_pooled").exists()]
    print(
        f"[rmme-tap] {len(todo)} episodes to do / {len(jobs) - len(todo)} already done "
        f"(shard {args.worker_idx}/{args.num_workers}, {len(tasks)} tasks)",
        flush=True,
    )
    if not todo:
        # FATAL on zero work for a full-corpus shard: a silent no-op that exits 0 is the failure
        # class that cost this study three node cycles (h14 §37.3 / §38.2).
        if not jobs:
            raise SystemExit("[rmme-tap] FATAL: shard enumerated zero episodes")
        return

    if args.plan_only:
        t0 = time.time()
        frames_total = demo_total = 0
        for i, (task, ep) in enumerate(todo):
            table = read_episode(episode_path(root, ep), (*VIEW_COLUMNS, STATE_COLUMN, "is_demo", "exec_start_idx"))
            plan = episode_plan(table, args.grid)
            sel = plan["frame_indices"]
            images = decode_rows(table, VIEW_COLUMNS[0], sel[:2])
            wrist = decode_rows(table, VIEW_COLUMNS[1], sel[:2])
            state = np.stack(table.column(STATE_COLUMN).to_pylist()).astype(np.float32)
            if images.shape[1:] != (256, 256, 3) or wrist.shape[1:] != (256, 256, 3):
                raise SystemExit(f"[rmme-tap] FATAL: {task}/ep{ep} view geometry {images.shape}/{wrist.shape}")
            if state.shape != (plan["n_frames"], 8) or not np.isfinite(state).all():
                raise SystemExit(f"[rmme-tap] FATAL: {task}/ep{ep} state {state.shape}")
            if not ep_task.get(ep):
                raise SystemExit(f"[rmme-tap] FATAL: no instruction for ep{ep}")
            frames_total += len(sel)
            demo_total += int(plan["is_demo"].sum())
            if i % 100 == 0 or i == len(todo) - 1:
                print(
                    f"[rmme-tap:plan] {i + 1}/{len(todo)} {task}/ep{ep} "
                    f"n={plan['n_frames']} F={len(sel)} demo_tok={int(plan['is_demo'].sum())} "
                    f"exec_start={plan['exec_start_idx']}",
                    flush=True,
                )
        print(
            f"[rmme-tap:plan] OK {len(todo)} episodes, {frames_total} tapped frames "
            f"({demo_total} demo, {frames_total - demo_total} live) on grid={args.grid} in "
            f"{time.time() - t0:.1f}s — NOTHING written, no model loaded",
            flush=True,
        )
        return

    if not args.pool_ckpt:
        raise SystemExit("--pool-ckpt is required unless --plan-only")

    import torch

    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap
    from workspace_models.features.pi_pooled_tap import load_pool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm, pool, encoder_id, pool_sha256 = load_pool(Path(args.pool_ckpt), device)
    tap = Pi05BackboneTap(str(Path(args.ckpt).expanduser()), args.config)
    backbone_id = Path(args.ckpt).expanduser().name
    print(
        f"[rmme-tap] backbone={backbone_id} config={args.config} pool={encoder_id} "
        f"device={device} patch_in_norm={'yes' if norm is not None else 'absent'} "
        f"grid={args.grid} stride={STRIDE}",
        flush=True,
    )

    t_start, n_frames_total = time.time(), 0
    for i, (task, ep) in enumerate(todo):
        t0 = time.time()
        table = read_episode(episode_path(root, ep), (*VIEW_COLUMNS, STATE_COLUMN, "is_demo", "exec_start_idx"))
        plan = episode_plan(table, args.grid)
        sel = plan["frame_indices"]
        base = decode_rows(table, VIEW_COLUMNS[0], sel)
        wrist = decode_rows(table, VIEW_COLUMNS[1], sel)
        states = np.stack(table.column(STATE_COLUMN).to_pylist()).astype(np.float32)[sel]
        prompt = ep_task[ep]
        if not prompt:
            raise SystemExit(f"[rmme-tap] FATAL: episode {ep} has no instruction")

        F = len(sel)
        pooled = torch.empty(F, pool.query.shape[-1], dtype=torch.float16)
        langs = []
        for lo in range(0, F, PAD_BATCH):
            hi = min(lo + PAD_BATCH, F)
            frames = {
                "agentview_left": base[lo:hi],
                "eye_in_hand": wrist[lo:hi],
                # The policy's zero-filled third slot (RoboMMEInputs), reproduced exactly.
                "agentview_right": np.zeros_like(wrist[lo:hi]),
            }
            result = tap.tap(frames, states[lo:hi], prompt)
            langs.append(np.asarray(result.lang_emb, dtype=np.float32))
            x = torch.from_numpy(np.asarray(result.patch_tokens, dtype=np.float32)).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                if norm is not None:
                    x = norm(x)
                p = pool(x[None])[0]  # PatchPool: [B,T,P,D] -> [B,T,512]
            pooled[lo:hi] = p.float().half().cpu()

        lang_per_frame = np.concatenate(langs).astype(np.float32)  # [F,2048]
        lang_global = lang_per_frame.mean(0)
        # RoboMME's instruction is constant within an episode, so the task line IS the per-frame
        # prompt. Recomputing it would be a second identical forward; take frame 0's vector so the
        # field exists with the same meaning as in the other taps.
        lang_task_line = lang_per_frame[0].copy()

        if not torch.isfinite(pooled.float()).all():
            raise SystemExit(f"[rmme-tap] NON-FINITE pooled tokens for {task}/ep{ep}")
        if pooled.float().abs().max() > 60000:
            raise SystemExit(f"[rmme-tap] fp16 overflow risk for {task}/ep{ep}")
        if not np.isfinite(lang_per_frame).all():
            raise SystemExit(f"[rmme-tap] NON-FINITE language embedding for {task}/ep{ep}")

        out_dir = out_root / task / f"demo_{ep:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "p.npz",
            p=pooled.numpy(),
            frame_indices=sel.astype(np.int64),
            lang_global=lang_global.astype(np.float32),
            lang_per_frame=lang_per_frame.astype(np.float16),
            lang_task_line=lang_task_line.astype(np.float32),
            encoder_id=np.array(encoder_id),
            pool_sha256=np.array(pool_sha256),
            backbone_id=np.array(backbone_id),
            prompt_source=np.array("lerobot_episode_instruction"),
            grid_mode=np.array(plan["grid_mode"]),
            is_demo=plan["is_demo"].astype(np.bool_),
            exec_start_idx=np.array(plan["exec_start_idx"], dtype=np.int64),
            n_frames_episode=np.array(plan["n_frames"], dtype=np.int64),
        )
        (out_dir / ".done_pooled").touch()
        n_frames_total += F
        if i % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t_start
            print(
                f"[rmme-tap] {i + 1}/{len(todo)} {task}/ep{ep} F={F} ({plan['n_frames']}) "
                f"demo_tok={int(plan['is_demo'].sum())} {time.time() - t0:.1f}s | "
                f"cum {n_frames_total} fr, {n_frames_total / max(el, 1e-9):.2f} fr/s, "
                f"{(i + 1) / max(el, 1e-9) * 3600:.0f} ep/h",
                flush=True,
            )

    el = time.time() - t_start
    print(
        f"[rmme-tap] COMPLETE: {len(todo)} episodes, {n_frames_total} frames in {el / 60:.1f} min "
        f"({n_frames_total / max(el, 1e-9):.2f} frames/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
