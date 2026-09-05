#!/usr/bin/env python3
"""CFG-CONDITIONED GR00T N1.7 target finetune (doc 12) — the model-free POC.

Identical to finetune_groot_17.py (same combined-target soup, seed-0 30% subsample, init-from the Phase-1
pretrain ckpt, freeze projector+DiT, VLM frozen) EXCEPT it conditions the action DENOISER on the workspace
latent w_t via AdaLN (added to the DiT temb), trained with classifier-free-guidance dropout. The VLM
backbone + perception tokens are NEVER touched. The net-new trainable module is a small zero-init
WSMCfgConditioner (so step 0 == the Eval1 baseline). At eval the serve extrapolates the velocity with a
guidance scale; with a human demo's w it becomes on-the-fly ICL (future work — the w_next slot is built but
left dropped here).

  WSM_POLICY_FEATS_ROOT=~/Research/TRI/wsm_data/wsm_policy_feats/groot_step50000 \
    WSM_P_DROP=0.2 WSM_DIAG_EVERY=100 \
    python vla_training/train/train_base/finetune_groot_17_with_wsm_cfg.py \
      --config scripts/configs/train/groot17_wsm_cfg_finetune.yaml
  python ... --dry-run     # build/log soup + subsample + CFG config, no gr00t
"""

from __future__ import annotations

import argparse
import os

from utils.subsample import uniform_num_demos
from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "groot_17", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="scripts/configs/train/groot17_wsm_cfg_finetune.yaml",
        help="CFG-conditioned WSM train YAML",
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log soup + subsample + CFG config; no gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    num_demos = uniform_num_demos(gs.soup)

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    w_dim = int(os.environ.get("WSM_W_DIM", "512"))
    p_drop = float(os.environ.get("WSM_P_DROP", "0.2"))
    with_future = os.environ.get("WSM_WITH_FUTURE", "0") == "1"  # POC: 0 (w_next slot dropped); ICL: 1
    diag_every = int(os.environ.get("WSM_DIAG_EVERY", "100"))
    demo_fraction = float((cfg.data.raw.get("subsample") or {}).get("target_fraction", 0.30))
    print(
        f"[finetune_groot_17_with_wsm_cfg] subsample first {num_demos}/dir (seed 0) | w_dim={w_dim} "
        f"p_drop={p_drop} with_future={with_future} diag_every={diag_every} feats_root={feats_root}"
    )
    if not feats_root and not args.dry_run:
        raise ValueError("set WSM_POLICY_FEATS_ROOT to the precomputed w_t dir (generate_policy_features)")

    if args.dry_run:
        print("[finetune_groot_17_with_wsm_cfg] dry-run OK (soup + subsample + CFG config; skipping gr00t).")
        return

    from vla_training.train.train_base._groot_common import build_and_run_groot
    from vla_training.train.train_base._groot_wsm_cfg_common import (
        assert_wt_coverage,
        install_wsm_cfg_action_head,
        install_wsm_cfg_dataset,
    )

    install_wsm_cfg_dataset(feats_root, demo_fraction=demo_fraction, with_future=with_future)
    install_wsm_cfg_action_head(w_dim=w_dim, p_drop=p_drop, with_future=with_future, diag_every=diag_every)
    assert_wt_coverage(feats_root, demo_fraction=demo_fraction, num_demos=num_demos, seed=0)

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("CFG WSM finetune requires train.init_from = the Phase-1 pretrain checkpoint")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,
    )


if __name__ == "__main__":
    main()
