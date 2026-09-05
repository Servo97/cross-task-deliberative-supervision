"""WSM-base frozen-probe trainer (pi0.5): Hungarian salient-patch recon + occupancy + cross-demo
workspace-latent alignment — the pi0.5 sibling of train_wsm_from_groot_17.py.

Trains ONLY the WorkspaceModel on the cached frozen pi0.5 features (wsm_cache_pi). Thin wrapper over
_wsm_train_common.train with the two pi-specific knobs: load_demo_pi (pi's per-frame language embedding
`lang_per_frame` fills the encoder's proprio slot, since pi0.5 puts robot state in the prompt) and
proprio_dim=2048 (vs GR00T's state_emb 1536). Salient targets use the pi-geometry labels
(vlm_episode_pi_*.npz; pi SigLIP 14x14->bin-8x8 maps points to DIFFERENT patch ids than GR00T's 256-crop
8x8 — see pi_geometry.py). Everything else (loop/loss/IO/--resume/--profile) is shared.

  python -m workspace_models.train.train_wsm_base.train_wsm_from_pi_05 \
      --manifest ~/Research/TRI/wsm_data/wsm_cache_pi/manifest.parquet --steps 2000 --batch-size 8 \
      --num-workers 16 --lambda-align 0 --out ~/Research/TRI/wsm_data/wsm_runs/pi_wsm_base

See internal_planning_and_todos/07_wsm_preprocessing_and_revised_plan.md + 04_wsm_roadmap.md.
"""

from __future__ import annotations

from workspace_models.train.train_wsm_base._wsm_train_common import train
from workspace_models.train.train_wsm_base.data import load_demo_pi
from wsm_settings import WSM_DATA_ROOT


def main() -> None:
    train(
        load_demo_pi,
        proprio_dim=2048,
        backbone="pi",
        default_out=str(WSM_DATA_ROOT / "wsm_runs" / "pi_wsm_base"),
    )


if __name__ == "__main__":
    main()
