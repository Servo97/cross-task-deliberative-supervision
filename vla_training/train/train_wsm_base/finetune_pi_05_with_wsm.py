#!/usr/bin/env python3
"""WSM-base finetune of pi0.5: action loss + salient-keyframe-patch aux head (jointly trained).

Finetunes the base VLA while a WSM decoder head (from workspace_models/) predicts the R1
GROOT+MolmoPoint salient-keyframe-patch labels from the backbone's workspace latent ``w_t``,
coupled in via AdaLN-Zero (zero-init = exact base-VLA behavior). This file owns the *action
policy*; the head module/networks live in workspace_models. Runs on robocasa_openpi @ jax-latest.
Governed by internal_planning_and_todos/04_wsm_roadmap.md ("WSM base").

TODO
----
- [ ] Load the YAML recipe (default: scripts/configs/train/wsm/pi05_wsm_base.yaml; configs TBD).
- [ ] Start from the R0 pi0.5 target-finetune (or pretrain) orbax checkpoint as the base VLA.
- [ ] Tap ``w_t`` from the backbone hidden states (workspace_models.networks.workspace_latent).
- [ ] Attach the keyframe-patch head (workspace_models.networks.keyframe_patch_head) via AdaLN-Zero.
- [ ] Add the joint loss = action_loss + lambda * keyframe_patch_loss; expose lambda in the config.
- [ ] Load the R1 keyframe-patch labels alongside the RoboCasa demos in the data pipeline.
- [ ] Reuse the verified pi0.5 recipe (01) for the action branch; eval on the target split afterwards.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WsmBaseConfig:
    """Typed view of the YAML recipe (fields TODO — base pi0.5 recipe + WSM head knobs)."""

    base_checkpoint: str = "ckpts/base/pi_05/target_finetune"
    wsm_loss_weight: float = 1.0
    keyframe_labels_dir: str = "data/wsm_labels/keyframe_patches"
    output_dir: str = "ckpts/wsm_base/pi_05"
    head: dict = field(default_factory=dict)


def load_config(path: str | Path) -> WsmBaseConfig:
    """TODO: parse YAML -> WsmBaseConfig (base recipe + WSM head/loss knobs)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return WsmBaseConfig(**raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/train/wsm/pi05_wsm_base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[finetune_pi_05_with_wsm] resolved config: {cfg}")
    raise NotImplementedError("TODO: attach keyframe-patch WSM head to pi0.5 and train jointly")


if __name__ == "__main__":
    main()
