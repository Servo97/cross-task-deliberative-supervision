#!/usr/bin/env python3
"""WSM-v2 finetune of GR00T N1.7: action loss + 3D-flow (and SigReg) aux heads (jointly trained).

Extends WSM-base with a *second* decoder that predicts 3D tracks (DynaFLIP recipe) from the same
workspace latent ``w_t``, optionally with the SigReg/LeJEPA regularizer. Heads couple in via
AdaLN-Zero (zero-init = exact base-VLA behavior). This file owns the *action policy*; the head
modules/networks live in workspace_models. Governed by internal_planning_and_todos/04_wsm_roadmap.md
("R3 = E2 — 3D flow (WSM v2)" + "E3 — SigReg / LeJEPA").

TODO
----
- [ ] Load the YAML recipe (default: scripts/configs/train/wsm_v2/groot_wsm_v2.yaml; configs TBD).
- [ ] Start from the WSM-base (or R0) GR00T checkpoint as the base VLA.
- [ ] Tap ``w_t`` (workspace_models.networks.workspace_latent); attach flow_head (+ optional sigreg_loss).
- [ ] Optionally keep the keyframe-patch head from WSM base; select head set via config.
- [ ] Joint loss = action_loss + lambda_flow * flow_loss (+ lambda_sigreg * sigreg_loss); expose weights.
- [ ] Load the DynaFLIP 3D-track labels (SpatialTrackerV2 path) alongside the RoboCasa demos.
- [ ] Reuse the verified GR00T recipe (01) for the action branch; eval on the target split afterwards.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WsmV2Config:
    """Typed view of the YAML recipe (fields TODO — base GR00T recipe + flow/sigreg head knobs)."""

    base_checkpoint: str = "ckpts/wsm_base/groot_17"
    flow_loss_weight: float = 1.0
    sigreg_loss_weight: float = 0.0
    flow_labels_dir: str = "data/wsm_labels/flow3d"
    output_dir: str = "ckpts/wsm_v2/groot_17"
    heads: dict = field(default_factory=dict)


def load_config(path: str | Path) -> WsmV2Config:
    """TODO: parse YAML -> WsmV2Config (base recipe + 3D-flow/sigreg head/loss knobs)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return WsmV2Config(**raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/train/wsm_v2/groot_wsm_v2.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[finetune_groot_17_with_wsm/v2] resolved config: {cfg}")
    raise NotImplementedError("TODO: attach 3D-flow (+sigreg) WSM heads to GR00T N1.7 and train jointly")


if __name__ == "__main__":
    main()
