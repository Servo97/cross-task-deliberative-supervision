#!/usr/bin/env python3
"""Phase-1 base-VLA PRETRAIN of GR00T N1.7 on RoboCasa365 ``pretrain_human300_mg60`` (Isaac-GR00T).

Adapter (backbone-specific feeder): loads the YAML recipe + shared ``GroupedSoup``, then hands the
GR00T-specific build/dispatch to ``_groot_common.build_and_run_groot``. Balancing-ON yields 3
``SingleDatasetConfig`` groups (mg_atomic / human_atomic / human_composite) each at mix_ratio 1/3;
balancing-OFF yields a single size-proportional spec (GR00T's native default).

Recipe (the proven N1.5 43%-atomic_seen recipe ported to N1.7): freeze LLM + vision, train
projector + DiT only, LR 3e-5, action_horizon 16, color-jitter on. See
internal_planning_and_todos/01_robocasa_protocol_and_recipes.md ("GR00T recipe").

Runs in the groot venv with Isaac-GR00T + robocasa importable. Base model nvidia/GR00T-N1.7-3B (gated).

  python vla_training/train/train_base/pretrain_groot_17.py --config scripts/configs/train/groot17_pretrain.yaml
  python ... --dry-run     # build + log the GroupedSoup/recipe WITHOUT importing torch/gr00t
"""

from __future__ import annotations

import argparse

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "groot_17", "pretrain"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="train YAML (default: scripts/configs/train/groot17_pretrain.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="build/log GroupedSoup only; no torch/gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    if args.dry_run:
        print("[pretrain_groot_17] dry-run OK (GroupedSoup built; skipping gr00t build + dispatch).")
        return

    # Heavy gr00t/torch imports happen inside _groot_common (groot venv only).
    from vla_training.train.train_base._groot_common import build_and_run_groot

    base_model = str(cfg.model.get("base_model", "nvidia/GR00T-N1.7-3B"))
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=base_model,
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
    )


if __name__ == "__main__":
    main()
