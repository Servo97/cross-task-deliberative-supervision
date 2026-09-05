#!/usr/bin/env python3
"""DEMO-CFG GR00T N1.7 target finetune (WSMv2, doc 15) — condition the action denoiser on a DEMO VIDEO.

Identical to finetune_groot_17_with_wsm_cfg.py (combined soup, seed-0 subsample, init from the Phase-1
pretrain, VLM untouched, zero-init conditioner => step 0 == baseline) EXCEPT slot2 of the conditioner is
LIVE: a FROZEN HistoryDemoFusion turns (causal frozen-w history, ±W window of a matched partner demo's
frozen DemoEncoder tokens) into z, which rides the proven w_next pathway into the DiT temb. Per-component
CFG dropout trains the null modes; at eval the serve guides with s and can null the demo slot (C1b).

Env (set by the entry / submit):
  WSM_POLICY_FEATS_ROOT   precomputed frozen-w dir  (65k arm: wsm_policy_feats/groot_65k)
  WSM_DEMO_TOKENS_ROOT    precomputed d.npz dir     (wsm_demo_tokens/orig_65k_matched)
  WSM_REGISTRY            registry_eval.json        (same dir; sha stamped + asserted)
  WSM_FUSION_CKPT         wsm2_stepN.pt             (fusion weights + registry sha)
  WSM_P_DROP / WSM_DIAG_EVERY / WSM_K_HIST / WSM_WINDOW / WSM_JITTER

  python vla_training/train/train_base/finetune_groot_17_with_wsm_demo_cfg.py \
      --config scripts/configs/train/groot17_wsm_demo_cfg_finetune.yaml   [--dry-run]
"""

from __future__ import annotations

import argparse
import os

from utils.subsample import uniform_num_demos
from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "groot_17", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/train/groot17_wsm_demo_cfg_finetune.yaml")
    ap.add_argument("--dry-run", action="store_true", help="soup + config only; no gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    num_demos = uniform_num_demos(gs.soup)

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    dtok_root = os.environ.get("WSM_DEMO_TOKENS_ROOT")
    registry = os.environ.get("WSM_REGISTRY")
    fusion_ckpt = os.environ.get("WSM_FUSION_CKPT")
    w_dim = int(os.environ.get("WSM_W_DIM", "512"))
    p_drop = float(os.environ.get("WSM_P_DROP", "0.2"))
    diag_every = int(os.environ.get("WSM_DIAG_EVERY", "100"))
    k_hist = int(os.environ.get("WSM_K_HIST", "16"))
    window = int(os.environ.get("WSM_WINDOW", "20"))
    jitter = int(os.environ.get("WSM_JITTER", "5"))
    demo_fraction = float((cfg.data.raw.get("subsample") or {}).get("target_fraction", 0.30))
    print(
        f"[ft-demo-cfg] subsample {num_demos}/dir | p_drop={p_drop} K={k_hist} W={window} J={jitter}\n"
        f"  feats={feats_root}\n  dtok={dtok_root}\n  registry={registry}\n  fusion={fusion_ckpt}"
    )
    missing = [
        n
        for n, v in (
            ("WSM_POLICY_FEATS_ROOT", feats_root),
            ("WSM_DEMO_TOKENS_ROOT", dtok_root),
            ("WSM_REGISTRY", registry),
            ("WSM_FUSION_CKPT", fusion_ckpt),
        )
        if not v
    ]
    if missing and not args.dry_run:
        raise ValueError(f"demo-CFG finetune requires env: {missing}")

    if args.dry_run:
        print("[ft-demo-cfg] dry-run OK (soup + config; skipping gr00t).")
        return

    from vla_training.train.train_base._groot_common import build_and_run_groot
    from vla_training.train.train_base._groot_wsm_cfg_common import assert_wt_coverage
    from vla_training.train.train_base._groot_wsm_demo_cfg_common import (
        install_wsm_demo_cfg_action_head,
        install_wsm_demo_cfg_dataset,
    )

    install_wsm_demo_cfg_dataset(
        feats_root, dtok_root, registry, demo_fraction=demo_fraction, k_hist=k_hist, window=window, jitter=jitter
    )
    install_wsm_demo_cfg_action_head(fusion_ckpt, w_dim=w_dim, p_drop=p_drop, diag_every=diag_every)
    assert_wt_coverage(feats_root, demo_fraction=demo_fraction, num_demos=num_demos, seed=0)

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("demo-CFG finetune requires train.init_from = the Phase-1 pretrain checkpoint")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,
    )


if __name__ == "__main__":
    main()
