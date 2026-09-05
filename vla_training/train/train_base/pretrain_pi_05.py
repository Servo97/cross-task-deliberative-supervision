#!/usr/bin/env python3
"""Phase-1 base-VLA PRETRAIN of pi0.5 on RoboCasa365 ``pretrain_human300_mg60`` (openpi jax-latest).

Adapter (backbone-specific feeder): loads the YAML recipe + the shared ``GroupedSoup`` (soup +
per-source balancing weights), then delegates the openpi build/dispatch to ``_pi05_common``.

Recipe = the official 39.6%-atomic_seen pi05 config extended to human+MimicGen: freeze NOTHING
(prefix Gemma trains), full-LR vision, EMA 0.99, cosine peak 2.5e-5 (->5e-5 @ bs256). See
internal_planning_and_todos/01_robocasa_protocol_and_recipes.md ("pi0.5 recipe").

Runs in the pi05 venv with robocasa_openpi (@ jax-latest) + robocasa importable. The balancing-ON
arm relies on the fork's GrootOpenpiMultiDataset ``dataset_weights`` fix (groot_openpi_dataset.py).

  python vla_training/train/train_base/pretrain_pi_05.py --config scripts/configs/train/pi05_pretrain.yaml
  python ... --dry-run     # build + log the GroupedSoup/recipe WITHOUT importing jax/openpi
"""

from __future__ import annotations

import argparse

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "pi05", "pretrain"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="train YAML (default: scripts/configs/train/pi05_pretrain.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="build/log GroupedSoup only; no jax/openpi")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    if args.dry_run:
        print("[pretrain_pi_05] dry-run OK (GroupedSoup built; skipping openpi build + dispatch).")
        return

    from vla_training.train.train_base._pi05_common import build_and_run_pi05

    build_and_run_pi05(cfg, gs)


if __name__ == "__main__":
    main()
