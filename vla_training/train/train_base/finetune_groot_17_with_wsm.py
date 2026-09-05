#!/usr/bin/env python3
"""WSM-CONDITIONED GR00T N1.7 target finetune — the canary (Step 2b).

Identical to finetune_groot_17.py (same combined-target soup, same seed-0 30% episode subsample, same
init-from the Phase-1 pretrain checkpoint, same freeze = projector + DiT) EXCEPT it installs the WSM
hooks before dispatch: each sample is conditioned on a causal window of K precomputed workspace latents
w_t (generate_policy_features) via a ZERO-INIT TokenModulator in the action head. Zero-init => step 0
is the EXACT Eval1 baseline, so any Eval2 gain is attributable to w_t alone (apples-to-apples vs the
30.5% GR00T Eval1).

The ONLY differences vs the normal finetune are the two install_wsm_* monkeypatches; everything else
(optimizer, batch 128, DDP, 60k steps) is unchanged, so the modulator is the sole net-new trainable
module. Requires the precomputed policy features on the node (WSM_POLICY_FEATS_ROOT).

  WSM_POLICY_FEATS_ROOT=~/Research/TRI/wsm_data/wsm_policy_feats/groot_step65000 \
    python vla_training/train/train_base/finetune_groot_17_with_wsm.py \
      --config scripts/configs/train/groot17_wsm_canary_finetune.yaml
  python ... --dry-run     # build/log soup + subsample + WSM config, no torch/gr00t
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
        "--config", default="scripts/configs/train/groot17_wsm_canary_finetune.yaml", help="WSM canary train YAML"
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log soup + subsample + WSM config; no gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    num_demos = uniform_num_demos(gs.soup)

    # WSM canary config (env wins, matching the WSM_* idiom; YAML wsm block is provenance).
    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    k_window = int(os.environ.get("WSM_K_WINDOW", "2"))
    w_dim = int(os.environ.get("WSM_W_DIM", "512"))
    demo_fraction = float((cfg.data.raw.get("subsample") or {}).get("target_fraction", 0.30))
    print(
        f"[finetune_groot_17_with_wsm] subsample first {num_demos}/dir (seed 0) | "
        f"k_window={k_window} w_dim={w_dim} feats_root={feats_root}"
    )
    if not feats_root and not args.dry_run:
        raise ValueError("set WSM_POLICY_FEATS_ROOT to the precomputed w_t dir (generate_policy_features)")

    if args.dry_run:
        print("[finetune_groot_17_with_wsm] dry-run OK (soup + subsample + WSM config; skipping gr00t).")
        return

    from vla_training.train.train_base._groot_common import build_and_run_groot
    from vla_training.train.train_base._groot_wsm_common import (
        assert_wt_coverage,
        install_wsm_action_head,
        install_wsm_dataset,
    )

    # Install BEFORE build_and_run_groot (which patches the trainer, then run()s).
    install_wsm_dataset(feats_root, k_window=k_window, demo_fraction=demo_fraction)
    install_wsm_action_head(w_dim=w_dim, k_window=k_window, lang_dim=2048)
    # FAIL-FAST coverage contract: every demo in the seed-0 first-`num_demos` keep-set MUST have w_t.
    # The cache is complete (7500/7500), so we train the FULL keep-set via the normal episode subsample
    # (episode_subsample_num_demos=num_demos) — NOT a w_t-filtered subset, which previously biased the
    # composite splits. Asserts up front so any gap fails at startup, not mid-dataloading.
    assert_wt_coverage(feats_root, demo_fraction=demo_fraction, num_demos=num_demos, seed=0)

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("WSM finetune requires train.init_from = the Phase-1 pretrain checkpoint (same as Eval1)")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,  # normal seed-0 first-num_demos subsample (coverage asserted)
    )


if __name__ == "__main__":
    main()
