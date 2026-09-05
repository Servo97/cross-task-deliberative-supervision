#!/usr/bin/env python3
"""Phase-2 base-VLA TARGET-FINETUNE of pi0.5 on the 50 RoboCasa365 target tasks (openpi jax-latest).

ONE combined finetune over atomic_seen + composite_seen + composite_unseen at the configured demo
fraction (default 30% via data.subsample.target_fraction; composite_unseen first appears here).
pi0.5 realizes the 30% NATIVELY — each combined-target ds_meta carries filter_key='150_demos', which
robocasa's loader (get_subset_demos_filter_key, seed 0) consumes at load time. Single human source =>
balancing OFF. Inits from the Phase-1 pretrain checkpoint (train.init_from).

Same adapter machinery as pretrain_pi_05; only the YAML differs (combined-30% soup, ckpt init,
balancing off). See internal_planning_and_todos/01_robocasa_protocol_and_recipes.md ("Phase 2").

  python vla_training/train/train_base/finetune_pi_05.py --config scripts/configs/train/pi05_target_finetune.yaml
  python ... --dry-run     # build + log the combined-30% GroupedSoup WITHOUT importing jax/openpi
"""

from __future__ import annotations

import argparse

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "pi05", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=None, help="train YAML (default: scripts/configs/train/pi05_target_finetune.yaml)"
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log GroupedSoup only; no jax/openpi")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    if args.dry_run:
        print("[finetune_pi_05] dry-run OK (combined-target GroupedSoup built; skipping openpi build + dispatch).")
        return

    from vla_training.train.train_base._pi05_common import build_and_run_pi05

    build_and_run_pi05(cfg, gs)


if __name__ == "__main__":
    main()
