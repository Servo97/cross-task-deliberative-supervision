#!/usr/bin/env python3
"""MODEL-FREE WSM GR00T N1.7 target finetune (doc 12) — the alternative to the injection canary.

Identical to finetune_groot_17.py (same combined-target soup, seed-0 30% subsample, init-from the
Phase-1 pretrain ckpt, freeze projector+DiT) EXCEPT it adds a JEPA+SIGReg AUX LOSS at the action head's
penultimate layer: the penultimate features are aligned to the next workspace latent w_{t+1} (precomputed,
frozen encoder) and SIGReg-regularized. w is a TRAINING TARGET ONLY — never injected, never in the
inference graph — so the injection recipe's eval-OOD failure mode is structurally impossible. The net-new
trainable module is a small JEPAPredictor (the DiT/projector train as in the normal finetune).

  WSM_POLICY_FEATS_ROOT=~/Research/TRI/wsm_data/wsm_policy_feats/groot_step65000 \
    WSM_JEPA_WEIGHT=1.0 WSM_SIGREG_WEIGHT=0.05 \
    python vla_training/train/train_base/finetune_groot_17_with_wsm_jepa.py \
      --config scripts/configs/train/groot17_wsm_jepa_finetune.yaml
  python ... --dry-run     # build/log soup + subsample + JEPA config, no torch/gr00t
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
        default="scripts/configs/train/groot17_wsm_jepa_finetune.yaml",
        help="model-free WSM (JEPA) train YAML",
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log soup + subsample + JEPA config; no gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    # SOUP MASS — mirrors finetune_groot_17.py: target50 subsamples seed-0 keep-first-N;
    # remembench13 is native full mass (filter_key None everywhere) and uniform_num_demos raises
    # by contract on such a soup, so it must not be called.
    full_mass = all(m.get("filter_key") is None for m in gs.soup)
    num_demos = None if full_mass else uniform_num_demos(gs.soup)

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    w_dim = int(os.environ.get("WSM_W_DIM", "512"))
    jepa_weight = float(os.environ.get("WSM_JEPA_WEIGHT", "1.0"))
    sigreg_weight = float(os.environ.get("WSM_SIGREG_WEIGHT", "0.05"))
    direct = os.environ.get("WSM_JEPA_DIRECT", "0") == "1"
    # k: the head predicts grid rows +1..+k. Exported to the dataset patch AND the head from the
    # SAME variable so the loader's k and the model's k can never disagree (a mismatch is a hard
    # shape error in wsm_jepa_sigreg_loss, never a silent broadcast).
    num_futures = int(os.environ.get("WSM_JEPA_NUM_FUTURES", "1"))
    demo_fraction = float((cfg.data.raw.get("subsample") or {}).get("target_fraction", 0.30))
    print(
        f"[finetune_groot_17_with_wsm_jepa] subsample first {num_demos}/dir (seed 0) | "
        f"w_dim={w_dim} jepa_w={jepa_weight} sigreg_w={sigreg_weight} direct={direct} "
        f"k={num_futures} feats_root={feats_root}"
    )
    if not feats_root and not args.dry_run:
        raise ValueError("set WSM_POLICY_FEATS_ROOT to the precomputed w_t dir (generate_policy_features)")

    if args.dry_run:
        print("[finetune_groot_17_with_wsm_jepa] dry-run OK (soup + subsample + JEPA config; skipping gr00t).")
        return

    from vla_training.train.train_base._groot_common import build_and_run_groot
    from vla_training.train.train_base._groot_wsm_jepa_common import (
        assert_wt_coverage,
        install_wsm_jepa_action_head,
        install_wsm_jepa_dataset,
    )

    install_wsm_jepa_dataset(feats_root, demo_fraction=demo_fraction, num_futures=num_futures, soup=gs.soup)
    install_wsm_jepa_action_head(
        w_dim=w_dim, jepa_weight=jepa_weight, sigreg_weight=sigreg_weight, direct=direct, num_futures=num_futures
    )
    # Same fail-fast coverage contract as the injection canary: every sampled demo must have w.npz.
    if not full_mass:
        assert_wt_coverage(feats_root, demo_fraction=demo_fraction, num_demos=num_demos, seed=0)

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("model-free WSM finetune requires train.init_from = the Phase-1 pretrain checkpoint")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,
    )


if __name__ == "__main__":
    main()
