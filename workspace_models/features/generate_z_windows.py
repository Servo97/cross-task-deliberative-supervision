#!/usr/bin/env python3
"""generate_z_windows — precomputed fused demo latents z for the pi (JAX) post-train (doc 15 / pi port).

pi trains in JAX but the HistoryDemoFusion is torch. Since the fusion is FROZEN at post-train (D17), z is
a deterministic function of (demo1 grid step, partner, tau) — so we precompute it offline (jitter=0,
proportional tau) and the pi dataloader ships wsm_w_window = [z, w_t] (slot -2 = z consumed by
wsm_cfg_with_future; slot -1 = w_t). No JAX port, zero pi model changes beyond the with_future read.

  <w-root>/<task>/demo_<ep>/w.npz + <dtok-root>/.../d.npz + partner manifest (matched, registry-excluded)
    -> <out-root>/<task>/demo_<ep>/z4.npz { z [P,F,512] fp16, partner_eps [P] int64, _meta } + .done_z

Perf: per demo, ALL (partner, grid-step) pairs batch through the fusion in chunks — one GPU pass per demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vla_training.train.train_base._groot_wsm_demo_cfg_common import build_partner_manifest  # noqa: E402
from workspace_models.features.wsm_align import demo_window_at  # noqa: E402
from workspace_models.networks.demo_fusion import HistoryDemoFusion  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w-root", required=True)
    ap.add_argument("--dtok-root", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--fusion-ckpt", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lang-table", required=True, help="task_lang_table.npz (task-mean lang, serve parity)")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    w_root, dtok, out_root = (Path(p).expanduser() for p in (args.w_root, args.dtok_root, args.out_root))
    dev = args.device
    ck = torch.load(Path(args.fusion_ckpt).expanduser(), map_location="cpu", weights_only=False)
    fus = HistoryDemoFusion()
    fus.load_state_dict(ck["fusion"])
    fus = fus.to(dev).eval()
    K, W = fus.k_hist, args.window
    registry = json.loads(Path(args.registry).expanduser().read_text())
    lt = np.load(Path(args.lang_table).expanduser(), allow_pickle=True)
    lang_by_task = {str(t): v for t, v in zip(lt["tasks"], lt["lang"])} if "tasks" in lt.files else None
    tasks = sorted(t for t in registry if (w_root / t).is_dir())
    manifest = build_partner_manifest(w_root, dtok, set(tasks), registry)
    meta = json.dumps(
        {"fusion": str(args.fusion_ckpt), "registry_sha": ck.get("registry_sha", "?"), "window": W, "jitter": 0}
    )
    print(f"[z4] {len(tasks)} tasks  K={K} W={W}  fusion={args.fusion_ckpt}", flush=True)

    done = skip = 0
    for task in tasks:
        lang = torch.from_numpy(
            np.asarray(lang_by_task[task] if lang_by_task else np.zeros(2048), dtype=np.float32)
        ).to(dev)
        for ddir in sorted((w_root / task).glob("demo_*")):
            ep = int(ddir.name.split("_")[1])
            out_dir = out_root / task / ddir.name
            if (out_dir / ".done_z").exists():
                skip += 1
                continue
            partners = manifest.get(task, {}).get(ep)
            if not partners:
                continue
            wz = np.load(ddir / "w.npz")
            w = torch.from_numpy(wz["w"].astype(np.float32)).to(dev)  # [F,512]
            F = w.shape[0]
            ar = torch.arange(F, device=dev)
            hidx = (ar[:, None] - (K - 1) + torch.arange(K, device=dev)[None]).clamp_min(0)
            hist = w[hidx]  # [F,K,512]
            zs = []
            for p_ep in partners:
                dz = np.load(dtok / task / f"demo_{p_ep:06d}" / "d.npz")
                d = torch.from_numpy(dz["d"].astype(np.float32)).to(dev)  # [M,512]
                M = d.shape[0]
                taus = np.round(np.arange(F) / max(F - 1, 1) * (M - 1)).astype(np.int64)
                idxs, offs, masks = zip(*(demo_window_at(M, int(t), W) for t in taus))
                widx = torch.from_numpy(np.stack(idxs)).to(dev)  # [F,41]
                woff = torch.from_numpy(np.stack(offs)).to(dev)
                wmask = torch.from_numpy(np.stack(masks)).to(dev)
                dw = d[widx]  # [F,41,512]
                z_parts = []
                for s in range(0, F, args.chunk):
                    e = min(F, s + args.chunk)
                    o = fus(hist[s:e], dw[s:e], woff[s:e], wmask[s:e], lang[None].expand(e - s, -1))
                    z_parts.append(o["z"].half().cpu())
                zs.append(torch.cat(z_parts))
            z = torch.stack(zs)  # [P,F,512]
            if not torch.isfinite(z.float()).all():
                raise RuntimeError(f"[z4] non-finite z for {task}/demo_{ep:06d}")
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_dir / "z4.npz",
                z=z.numpy(),
                partner_eps=np.asarray(partners, dtype=np.int64),
                frame_indices=wz["frame_indices"].astype(np.int64),
                _meta=np.array(meta),
            )
            (out_dir / ".done_z").touch()
            done += 1
            if done % 500 == 0:
                print(f"[z4] {done} done / {skip} skipped", flush=True)
    print(f"[z4] COMPLETE: {done} written, {skip} skipped", flush=True)


if __name__ == "__main__":
    main()
