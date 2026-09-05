#!/usr/bin/env python3
"""Gated-DeltaNet workspace-conditioner GR00T N1.7 target finetune — the torch twin of the pi0.5
`s1/deltanet` arm, for the ReMemBench sync validation.

Identical to finetune_groot_17.py (same combined-target soup, seed-0 30% subsample, init-from the
Phase-1 pretrain ckpt, same freeze/optim) EXCEPT the DiT's global AdaLN conditioning bus `temb` gets
an additive read of the causal omega window: a gated delta-rule linear-attention recurrence over the
K window positions, read out at the newest position and gated by tanh(alpha). The recurrence is
STATELESS across policy calls, so this is the steering axis, not the RoboTTT fast-weight axis.

The conditioner is a parity-verified port of the JAX `WSMGatedDeltaNetConditioner`
(tests/test_groot_wsm_deltanet.py pins it against a fixture extracted from the JAX module).

  WSM_POLICY_FEATS_ROOT=<the study's shared omega cache> WSM_DN_WINDOW=8 \
    python vla_training/train/train_base/finetune_groot_17_with_wsm_deltanet.py \
      --config scripts/configs/train/groot17_wsm_deltanet_finetune.yaml
  python ... --dry-run     # build/log soup + subsample + conditioner config, no torch/gr00t
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
        default="scripts/configs/train/groot17_wsm_deltanet_finetune.yaml",
        help="gated-deltanet conditioner train YAML",
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log soup + subsample + conditioner config; no gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    # SOUP MASS — mirrors finetune_groot_17.py exactly (selected by the soup, never by a flag):
    #   * target50: filter_key set => deterministic seed-0 keep-first-N episode subsample.
    #   * remembench13: filter_key None on every meta => NATIVE FULL MASS. uniform_num_demos raises
    #     by contract on such a soup, so it must not be called and the subsample patch must not be
    #     installed.
    full_mass = all(m.get("filter_key") is None for m in gs.soup)
    num_demos = None if full_mass else uniform_num_demos(gs.soup)

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    w_dim = int(os.environ.get("WSM_W_DIM", "512"))
    window_len = int(os.environ.get("WSM_DN_WINDOW", "8"))
    num_heads = int(os.environ.get("WSM_DN_HEADS", "2"))
    head_dim = int(os.environ.get("WSM_DN_HEAD_DIM", "256"))
    gate_init = float(os.environ.get("WSM_DN_GATE_INIT", "1e-3"))
    history_dropout = float(os.environ.get("WSM_DN_HISTORY_DROPOUT", "0.0"))
    # cond_dim defaults to the DiT inner width (1536 on the released N1.7 ckpt), resolved at attach.
    cond_dim_env = os.environ.get("WSM_DN_COND_DIM")
    cond_dim = int(cond_dim_env) if cond_dim_env else None
    demo_fraction = float((cfg.data.raw.get("subsample") or {}).get("target_fraction", 0.30))
    print(
        f"[finetune_groot_17_with_wsm_deltanet] subsample first {num_demos}/dir (seed 0) | "
        f"w_dim={w_dim} window={window_len} heads={num_heads}x{head_dim} "
        f"gate_init={gate_init} history_dropout={history_dropout} cond_dim={cond_dim or 'auto'} "
        f"feats_root={feats_root}"
    )
    if not feats_root and not args.dry_run:
        raise ValueError("set WSM_POLICY_FEATS_ROOT to the precomputed omega cache dir")

    if args.dry_run:
        print(
            "[finetune_groot_17_with_wsm_deltanet] dry-run OK (soup + subsample + conditioner config; skipping gr00t)."
        )
        return

    from vla_training.train.train_base._groot_common import build_and_run_groot
    from vla_training.train.train_base._groot_wsm_deltanet_common import (
        assert_wt_coverage,
        install_wsm_deltanet_action_head,
        install_wsm_deltanet_dataset,
    )

    install_wsm_deltanet_dataset(feats_root, window_len=window_len, demo_fraction=demo_fraction, soup=gs.soup)
    install_wsm_deltanet_action_head(
        w_dim=w_dim,
        cond_dim=cond_dim,
        window_len=window_len,
        num_heads=num_heads,
        head_dim=head_dim,
        gate_init=gate_init,
        history_dropout=history_dropout,
    )
    # Same fail-fast coverage contract as every other WSM arm: each sampled demo must have w.npz.
    # On full-mass soups there is no subsample, so coverage is checked over every demo.
    if not full_mass:
        assert_wt_coverage(feats_root, demo_fraction=demo_fraction, num_demos=num_demos, seed=0)

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("deltanet finetune requires train.init_from = the Phase-1 pretrain checkpoint")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,
    )


if __name__ == "__main__":
    main()
