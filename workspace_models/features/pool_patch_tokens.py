#!/usr/bin/env python3
"""pool_patch_tokens — one-shot offline pooling of the raw VLM patch cache through the FROZEN WSMv1
patch_in_norm + PatchPool -> p.npz per demo (doc 15 D2).

Turns the ~300GB patch cache into ~460MB of pooled per-frame tokens so the ENTIRE WSMv2 encoder phase
becomes a small-tensor job (p.npz + w.npz + lang only — no VLM, no policy, local GPUs). The pool runs
PRE-proprio in WorkspaceEncoder, so the pooled tokens are vision+language-only — the proprio-free demo2
contract for the human-video interface. patch_in_norm MUST be applied (it sits before the pool in
workspace_latent.py; omitting it puts tokens off-distribution — feasibility critique).

  <cache-root>/<task>/demo_<ep>/patch_tokens.npy [F,192,2048] fp16  +  feats.npz (frame_indices, lang)
    -> <same dir>/p.npz { p [F,512] fp16, frame_indices [F] int64, lang_global [2048] fp32 }  + .done_pooled

Perf: mmap + chunked GPU forwards (256 frames/chunk, bf16 autocast), resume via .done markers, shardable
(--worker-idx/--num-workers stable ordering split) — mirrors generate_policy_features' idioms.

  python workspace_models/features/pool_patch_tokens.py \
      --cache-root ~/Research/TRI/wsm_data/wsm_cache --encoder-ckpt ~/Research/TRI/wsm_data/wsm_ckpts/groot_wsm/wsm_step65000.pt \
      --device cuda [--tasks A,B] [--worker-idx 0 --num-workers 4] [--out-root ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_pool(ckpt_path: str, device: str, proprio_dim: int = 1536):
    """(patch_in_norm | None, PatchPool) from a frozen WSMv1 ckpt, eval-mode on device, plus the ckpt id."""
    from workspace_models.features.generate_policy_features import load_wsm

    model, meta = load_wsm(ckpt_path, device, proprio_dim=proprio_dim)
    enc = model.encoder
    norm = getattr(enc, "patch_in_norm", None)
    pool = enc.pool
    for m in filter(None, (norm, pool)):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return norm, pool, meta


@torch.no_grad()
def pool_demo(demo_dir: Path, norm, pool, device: str, chunk: int = 256) -> dict:
    patch = np.load(demo_dir / "patch_tokens.npy", mmap_mode="r")  # [F,192,2048] fp16
    f = np.load(demo_dir / "feats.npz", allow_pickle=True)
    frame_indices = f["frame_indices"].astype(np.int64)
    lang = np.asarray(f["lang_global"], dtype=np.float32)
    F = patch.shape[0]
    out = torch.empty(F, pool.query.shape[-1], dtype=torch.float16)
    for s in range(0, F, chunk):
        x = torch.from_numpy(np.asarray(patch[s : s + chunk], dtype=np.float32)).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            if norm is not None:
                x = norm(x)
            p = pool(x[None])[0]  # [chunk,512] (pool takes [B,T,P,D])
        out[s : s + chunk] = p.float().half().cpu()
    if not torch.isfinite(out.float()).all():
        raise RuntimeError(f"[pool] NON-FINITE pooled tokens for {demo_dir}")
    if out.float().abs().max() > 60000:
        raise RuntimeError(f"[pool] fp16 overflow risk (absmax {out.float().abs().max():.0f}) for {demo_dir}")
    return {"p": out.numpy(), "frame_indices": frame_indices, "lang_global": lang}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--encoder-ckpt", required=True, help="frozen WSMv1 ckpt (decode-selected; doc 15 D11)")
    ap.add_argument("--out-root", default=None, help="default: write p.npz next to patch_tokens.npy")
    ap.add_argument("--tasks", default="", help="comma-list; default all task dirs")
    ap.add_argument("--proprio-dim", type=int, default=1536, help="1536 groot / 2048 pi (loader arg only)")
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(args.cache_root).expanduser()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    demos = [d for t in tasks for d in sorted((root / t).glob("demo_*")) if (d / "patch_tokens.npy").exists()]
    demos = demos[args.worker_idx :: args.num_workers]
    norm, pool, meta = load_pool(args.encoder_ckpt, args.device, args.proprio_dim)
    enc_id = f"{Path(args.encoder_ckpt).parent.name}/{Path(args.encoder_ckpt).name}"
    print(f"[pool] {len(demos)} demos (shard {args.worker_idx}/{args.num_workers}) encoder={enc_id}", flush=True)

    done = skip = 0
    for d in demos:
        out_dir = (Path(args.out_root).expanduser() / d.relative_to(root)) if args.out_root else d
        marker = out_dir / ".done_pooled"
        if marker.exists():
            skip += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        data = pool_demo(d, norm, pool, args.device, args.chunk)
        np.savez(out_dir / "p.npz", **data, encoder_id=np.array(enc_id))
        marker.touch()
        done += 1
        if done % 200 == 0:
            print(f"[pool] {done} done / {skip} skipped", flush=True)
    print(f"[pool] COMPLETE: {done} pooled, {skip} already done", flush=True)


if __name__ == "__main__":
    main()
