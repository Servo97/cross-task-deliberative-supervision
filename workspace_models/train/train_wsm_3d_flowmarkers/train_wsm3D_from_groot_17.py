#!/usr/bin/env python3
"""Train the WSM-v2 3D-flow predictor on top of GR00T N1.7's workspace latent ``w_t`` (DynaFLIP).

Trains the *workspace model itself*: a second decoder that predicts 3D tracks from GR00T N1.7's
``w_t``, using the DynaFLIP preprocessing recipe (grid tracks + depth/pose + unprojection with
camera motion canceled; SpatialTrackerV2 single-tool path, fallback CoTracker3 + VGGT + TAPIP3D).
Networks in workspace_models/networks/. Governed by
internal_planning_and_todos/04_wsm_roadmap.md ("R3 = E2 — 3D flow (WSM v2)").

TODO
----
- [ ] Load the YAML recipe (default: scripts/configs/train/wsm_v2/groot_wsm_v2.yaml; configs TBD).
- [ ] Extract ``w_t`` from a (frozen) GR00T N1.7 backbone (workspace_models.networks.workspace_latent).
- [ ] Instantiate the 3D-flow head (workspace_models.networks.flow_head).
- [ ] Build the DynaFLIP 3D-track label dataset (SpatialTrackerV2 path; cancel camera motion).
- [ ] Train the track-prediction objective; log track endpoint error; checkpoint the WSM module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Wsm3DTrainConfig:
    """Typed view of the YAML recipe (fields TODO — backbone tap + flow head + label-tool knobs)."""

    backbone_checkpoint: str = "ckpts/base/groot_17/target_finetune"
    flow_labels_dir: str = "data/wsm_labels/flow3d"
    tracker: str = "spatialtracker_v2"
    num_train_steps: int = 50_000
    output_dir: str = "ckpts/wsm/groot_17/flow3d"
    head: dict = field(default_factory=dict)


def load_config(path: str | Path) -> Wsm3DTrainConfig:
    """TODO: parse YAML -> Wsm3DTrainConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Wsm3DTrainConfig(**raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="scripts/configs/train/wsm_v2/groot_wsm_v2.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[train_wsm3D_from_groot_17] resolved config: {cfg}")
    raise NotImplementedError("TODO: train the 3D-flow WSM head on GR00T N1.7 w_t (DynaFLIP)")


if __name__ == "__main__":
    main()
