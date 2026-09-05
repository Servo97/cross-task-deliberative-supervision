#!/usr/bin/env python3
"""rcb_pooled_tap — FUSED frozen pi05_libero tap + frozen WSMv1 pool for RoboCerebra.

The RoboCerebra sibling of `pi_pooled_tap.py` (RoboCasa/ReMemBench). Same product — a
byte-compatible `wsm_pooled` `p.npz` store that `train_stage_e.py` can load as one more `--tap` —
but three things about RoboCerebra force a separate module rather than another `--domain` branch:

1. **Backbone.** RoboCasa and ReMemBench are tapped from the frozen RoboCasa pretrain
   `pi05_on/149999`. RoboCerebra's sealed H12 arms init from the RELEASED `pi05_libero`, and the
   RoboCerebra eval server taps that same released checkpoint (`serve_pi05_libero_wsm.py`
   `DEFAULT_TAP_CHECKPOINT`, `--tap-source frozen` is the default). Tapping anything else here would
   put the encoder's train inputs and its serve inputs on two different networks — the exact
   train/serve split that has already invalidated one eval in this study. So: pi05_libero.
   The price is that RoboCerebra's tap is a THIRD frozen network; A3 (`tap_stats_audit.py`) is the
   audit that says whether the per-domain input adapter can reconcile it.

2. **Geometry.** LIBERO is 2-view (`image`, `wrist_image`) => 128 bin-averaged patch tokens, not
   RoboCasa's 3-view 192. The frozen WSMv1 `PatchPool` is token-count agnostic by construction (a
   single learned query attends over the patch axis; no positional embedding over patches, no
   P-dependent weight), so 128 pools exactly as 192 does. `patch_in_norm` is absent from the pi
   encoder (`input_norm=False`), so the tokens go straight to the pool.

3. **Prompt.** RoboCerebra's policy is served the CURRENT SUBTASK instruction, re-pinned at each
   subtask boundary — not one episode-level goal. The per-frame `task_index` column names it, and
   `omega_tap.py` already tapped it that way. The prefix is bidirectional (the prompt reaches the
   patch tokens), so the tap must use the same string the policy gets.

The tap itself is NOT reimplemented: `Pi05Tap` is imported from the eval server, so the tokens this
store is built from and the tokens the server will feed the encoder come from one definition,
including the fp16 round-trip and the `pad_batch=16` kernel pin.

Frame grid — a DEVIATION from the H12 store, taken deliberately. The H12 ω tap used
`linspace(0, len-1, 64)`, a 64-frame grid whose serve-side reconstruction needs the episode length
up front (`grid_stride_for_episode`). This store uses the `wsm_pooled` convention instead — stride 8
from 0 with the final frame always appended — because (a) Stage-E's length/positional conventions
are calibrated on it and the A3 audit is only apples-to-apples against the other two taps on it, and
(b) a fixed stride is causal and needs no episode length at serve. 907,875 frames -> 114,800 tapped.

Contract, byte-compatible with `wsm_pooled/pi_100k/<Task>/demo_%06d/p.npz`:

    p              [F,512]  fp16    frozen pool over the 128 bin-averaged SigLIP patches
    frame_indices  [F]      int64   stride 8, final frame always included
    lang_global    [2048]   float32 mean over frames of the masked-mean language embedding
    encoder_id     ()       str     "wsm_pool:<first 16 hex of the pool ckpt sha256>" (§32.4)
    pool_sha256    ()       str     full sha256 of the frozen pool checkpoint
  + backbone_id, prompt_source, subtask_index [F], task_index [F]   (additive; readers ignore)
  + `.done_pooled` marker

Task naming. RoboCerebra's 994 training episodes cover 947 distinct BDDL task files — it is a
one-episode-per-task corpus, not RoboCasa's 13 tasks x 150 demos. `<Task>` is therefore the BDDL
stem, which is the honest label: it makes pass-2's `within_task` stratum nearly empty and its
`cross_task` stratum the whole domain, which is what the data actually is. Collapsing to the 3
scenes would relabel genuinely different tasks as "within task".

Run in the openpi-jax env:

  PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES=1 ~/Research/envs/openpi-jax-latest/bin/python -m \
    workspace_models.features.rcb_pooled_tap \
      --dataset-root ~/Research/TRI/wsm_data/robocerebra/lerobot_home/wsmv2/robocerebra_train \
      --pool-ckpt    ~/Research/TRI/wsm_data/wsm_runs/pi_wsm_v1/wsm_step100000.pt \
      --out-root     ~/Research/TRI/wsm_data/wsm_pooled/rcb_pi_libero
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.features.pi_pooled_tap import (  # noqa: E402
    STRIDE,
    decode_frames_at,
    frame_selection,
)

VIEW_KEYS = ("image", "wrist_image")  # LIBERO 2-view; slot order = base, wrist
DEFAULT_TAP_CKPT = "~/Research/TRI/wsm_data/robocerebra/openpi_assets/openpi-assets/checkpoints/pi05_libero"
DEFAULT_DATASET = "~/Research/TRI/wsm_data/robocerebra/lerobot_home/wsmv2/robocerebra_train"
PAD_BATCH = 16  # the serve tap's pin; keeping it here means one XLA kernel across train and serve


def task_name(bddl_file: str) -> str:
    """`<BDDL stem>` — filesystem-safe already (uppercase scene prefix + underscores)."""
    return bddl_file[:-5] if bddl_file.endswith(".bddl") else bddl_file


def load_meta(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {
        int(r["task_index"]): r["task"]
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
    return info, tasks, prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--ckpt", default=DEFAULT_TAP_CKPT, help="frozen pi05_libero checkpoint dir")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--pool-ckpt", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()

    import pandas as pd
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "robocerebra"))
    from serve_pi05_libero_wsm import Pi05Tap  # noqa: E402  (one tap definition, train and serve)

    from workspace_models.features.pi_pooled_tap import load_pool

    root = Path(args.dataset_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info, tasks, prov = load_meta(root)
    chunks_size = int(info.get("chunks_size", 1000))

    eps = sorted(prov) if args.episodes is None else list(args.episodes)
    if args.limit:
        eps = eps[: args.limit]
    jobs = [(task_name(prov[e]["bddl_file"]), e) for e in eps]
    jobs.sort()
    jobs = jobs[args.worker_idx :: args.num_workers]
    todo = [j for j in jobs if not (out_root / j[0] / f"demo_{j[1]:06d}" / ".done_pooled").exists()]
    print(
        f"[rcb-tap] {len(todo)} episodes to do / {len(jobs) - len(todo)} already done "
        f"(shard {args.worker_idx}/{args.num_workers})",
        flush=True,
    )
    if not todo:
        return

    norm, pool, encoder_id, pool_sha256 = load_pool(Path(args.pool_ckpt), device)
    tap = Pi05Tap.from_checkpoint(Path(args.ckpt).expanduser(), args.config, pad_batch=PAD_BATCH, tokens_fp16=True)
    backbone_id = Path(args.ckpt).expanduser().name
    print(
        f"[rcb-tap] backbone={backbone_id} pool={encoder_id} device={device} "
        f"patch_in_norm={'yes' if norm is not None else 'absent'} stride={STRIDE}",
        flush=True,
    )

    t_start, n_frames_total = time.time(), 0
    for i, (task, ep) in enumerate(todo):
        t0 = time.time()
        chunk = ep // chunks_size
        df = pd.read_parquet(root / info["data_path"].format(episode_chunk=chunk, episode_index=ep))
        n = len(df)
        sel = frame_selection(n)
        states = np.stack(df["state"].to_numpy()).astype(np.float32)
        ti = df["task_index"].to_numpy()
        sti = df["subtask_index"].to_numpy()

        frames = {}
        for v in VIEW_KEYS:
            path = root / info["video_path"].format(episode_chunk=chunk, video_key=v, episode_index=ep)
            frames[v] = decode_frames_at(path, sel, n)

        examples = [
            {
                "observation/image": frames["image"][k],
                "observation/wrist_image": frames["wrist_image"][k],
                "observation/state": states[int(t)],
                "prompt": tasks[int(ti[int(t)])],
            }
            for k, t in enumerate(sel)
        ]

        F = len(sel)
        pooled = torch.empty(F, pool.query.shape[-1], dtype=torch.float16)
        langs = []
        for lo in range(0, F, PAD_BATCH):
            hi = min(lo + PAD_BATCH, F)
            tokens, _pooled_img, pooled_lang = tap.embed(examples[lo:hi])
            langs.append(pooled_lang)
            x = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                if norm is not None:
                    x = norm(x)
                p = pool(x[None])[0]  # PatchPool: [B,T,P,D] -> [B,T,512]
            pooled[lo:hi] = p.float().half().cpu()

        lang_per_frame = np.concatenate(langs).astype(np.float32)  # [F,2048]
        lang_global = lang_per_frame.mean(0)

        # ONE extra 1-frame forward whose prompt is the EPISODE task line rather than the current
        # subtask. Costs ~0.2 s/episode and exists because of §25.3: Stage-E conditions on a single
        # lang vector per episode, but RoboCerebra's prompt CHANGES at every re-pin, so an episode
        # mean is a blend of ~9 different instructions that no causal serve path can reproduce.
        # Storing all three candidates now keeps the conditioning contract an open decision; the tap
        # is a node-job and re-running it to add a field later costs a p5 node.
        #   lang_per_frame  the current subtask's vector      -> causal, train==serve exactly
        #   lang_task_line  the episode goal, constant        -> causal (known at reset), Stage-E-shaped
        #   lang_global     mean over frames                  -> what the other taps store; NOT serveable
        task_line_prompt = str(prov[ep]["task_line"])
        tl_example = dict(examples[0])
        tl_example["prompt"] = task_line_prompt
        _tk, _pi, tl_lang = tap.embed([tl_example])
        lang_task_line = np.asarray(tl_lang, dtype=np.float32)[0]
        if not torch.isfinite(pooled.float()).all():
            raise SystemExit(f"[rcb-tap] NON-FINITE pooled tokens for {task}/ep{ep}")
        if pooled.float().abs().max() > 60000:
            raise SystemExit(f"[rcb-tap] fp16 overflow risk for {task}/ep{ep}")

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
            prompt_source=np.array("lerobot_subtask_instruction"),
            subtask_index=sti[sel].astype(np.int64),
            task_index=ti[sel].astype(np.int64),
        )
        (out_dir / ".done_pooled").touch()
        n_frames_total += F
        if i % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t_start
            print(
                f"[rcb-tap] {i + 1}/{len(todo)} {task[:40]}/ep{ep} F={F} ({n}) "
                f"{time.time() - t0:.1f}s | cum {n_frames_total} fr, "
                f"{n_frames_total / max(el, 1e-9):.2f} fr/s, "
                f"{(i + 1) / max(el, 1e-9) * 3600:.0f} ep/h",
                flush=True,
            )

    el = time.time() - t_start
    print(
        f"[rcb-tap] COMPLETE: {len(todo)} episodes, {n_frames_total} frames in {el / 60:.1f} min "
        f"({n_frames_total / max(el, 1e-9):.2f} frames/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
