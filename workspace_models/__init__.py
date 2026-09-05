"""Workspace Model (WSM) — the research object: the auxiliary world-/workspace-prediction module.

The WSM predicts *where action-relevant change will happen* (salient 2D keyframe patches, then 3D
flow) from the backbone's workspace latent ``w_t``. This package defines the WSM module, its NN
heads (``networks/``), and the drivers that train it from each backbone (``train/``). Governed by
internal_planning_and_todos/04_wsm_roadmap.md.
"""
