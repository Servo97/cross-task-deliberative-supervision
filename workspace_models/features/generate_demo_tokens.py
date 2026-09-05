#!/usr/bin/env python3
"""generate_demo_tokens — frozen WSMv2 DemoEncoder over each demo's pooled tokens -> d.npz (doc 15 D17).

Run AFTER the WSMv2 encoder phase freezes: encodes every demo's p.npz (bidirectional, full-length) into
the demo-token sequence the post-train dataloader windows into the fusion. One shot, local, resumable.

  <p-root>/<task>/demo_<ep>/p.npz -> <out-root>/<task>/demo_<ep>/d.npz
      { d [M,512] fp16, frame_indices [M] int64, _meta json: wsm2 ckpt id + registry sha }  + .done_demo_tokens

Provenance (D19): _meta carries the WSMv2 ckpt id + registry sha; the serve asserts d.npz meta == the
finetune ckpt's stamp — the NaN-encoder class of bug dies here.

  PYTHONPATH=. python workspace_models/features/generate_demo_tokens.py \
      --p-root ~/Research/TRI/wsm_data/wsm_pooled/orig_65k --wsm2-ckpt ~/Research/TRI/wsm_data/wsm2_runs/orig_65k_mixed/wsm2_step20000.pt \
      --out-root ~/Research/TRI/wsm_data/wsm_demo_tokens/orig_65k_mixed [--tasks A,B] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.networks.demo_encoder import DemoEncoder  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-root", required=True)
    ap.add_argument("--wsm2-ckpt", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    p_root, out_root = Path(args.p_root).expanduser(), Path(args.out_root).expanduser()
    ck = torch.load(Path(args.wsm2_ckpt).expanduser(), map_location="cpu", weights_only=False)
    enc = DemoEncoder().to(args.device).eval()
    enc.load_state_dict(ck["demo_encoder"])
    ckpt_id = f"{Path(args.wsm2_ckpt).parent.name}/{Path(args.wsm2_ckpt).name}"
    meta = json.dumps(
        {"wsm2_ckpt": ckpt_id, "registry_sha": ck.get("registry_sha", "?"), "step": int(ck.get("step", -1))}
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or sorted(
        d.name for d in p_root.iterdir() if d.is_dir()
    )
    demos = [d for t in tasks for d in sorted((p_root / t).glob("demo_*")) if (d / "p.npz").exists()]
    print(f"[dtok] {len(demos)} demos  encoder={ckpt_id}", flush=True)

    done = skip = 0
    for d in demos:
        out_dir = out_root / d.relative_to(p_root)
        marker = out_dir / ".done_demo_tokens"
        if marker.exists():
            skip += 1
            continue
        z = np.load(d / "p.npz", allow_pickle=True)
        toks = torch.from_numpy(z["p"].astype(np.float32))[None].to(args.device)
        lang = torch.from_numpy(np.asarray(z["lang_global"], dtype=np.float32))[None].to(args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            dt = enc(toks, lang)[0].float()
        if not torch.isfinite(dt).all():
            raise RuntimeError(f"[dtok] NON-FINITE demo tokens for {d}")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "d.npz",
            d=dt.half().cpu().numpy(),
            frame_indices=z["frame_indices"].astype(np.int64),
            _meta=np.array(meta),
        )
        marker.touch()
        done += 1
        if done % 500 == 0:
            print(f"[dtok] {done} done / {skip} skipped", flush=True)
    print(f"[dtok] COMPLETE: {done} encoded, {skip} skipped", flush=True)


if __name__ == "__main__":
    main()
