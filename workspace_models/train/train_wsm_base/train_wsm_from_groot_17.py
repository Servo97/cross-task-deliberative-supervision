"""WSM-base frozen-probe trainer (GR00T N1.7): Hungarian salient-patch recon + occupancy +
cross-demo workspace-latent alignment (the core contribution).

Trains ONLY the WorkspaceModel (encoder + decoder) on the cached frozen GR00T features — the backbone
never updates. Thin wrapper over _wsm_train_common.train: GR00T's per-demo proprio is the state
embedding (state_emb [F,1536]). See _wsm_train_common for the loop/loss/IO; the pi0.5 sibling
(train_wsm_from_pi_05.py) is the same call with load_demo_pi + proprio_dim 2048.

  python -m workspace_models.train.train_wsm_base.train_wsm_from_groot_17 \
      --manifest ~/Research/TRI/wsm_data/wsm_cache/manifest.parquet --steps 2000 --batch-size 8 \
      --num-workers 16 --lambda-align 0 --out ~/Research/TRI/wsm_data/wsm_runs/groot_wsm_base

See internal_planning_and_todos/07_wsm_preprocessing_and_revised_plan.md.
"""

from __future__ import annotations

from workspace_models.train.train_wsm_base._wsm_train_common import train
from workspace_models.train.train_wsm_base.data import load_demo
from wsm_settings import WSM_DATA_ROOT


def main() -> None:
    train(
        load_demo,
        proprio_dim=1536,
        backbone="groot",
        default_out=str(WSM_DATA_ROOT / "wsm_runs" / "groot_wsm_base"),
    )


if __name__ == "__main__":
    main()
