# workspace_models/ — the Workspace Model (the research object)

The **WSM** is an auxiliary world-/workspace-prediction objective: it predicts *where
action-relevant change will happen* from the backbone's workspace latent `w_t` — first as salient
2D keyframe patches (WSM base), then as 3D flow (WSM v2), with a SigReg/LeJEPA regularized variant.
The output of every driver here is the **WSM predictor module** itself.

```
workspace_models/
  train/
    train_wsm_base/            # keyframe-patch predictor from w_t  (per backbone)
      train_wsm_from_{groot_17,pi_05}.py
    train_wsm_3d_flowmarkers/  # 3D-flow / track predictor (DynaFLIP)
      train_wsm3D_from_{groot_17,pi_05}.py
    train_wsm_3d_sigreg/       # SigReg / LeJEPA variant
      train_wsm3Dsigreg_from_{groot_17,pi_05}.py
  networks/                    # the WSM NN modules (see networks/README.md)
```

## vla_training vs workspace_models (the `train_wsm_*` split)

These two packages both contain `train_wsm_*` drivers, but they train **different objects**:

- **`workspace_models/train/*`** (here) trains/defines the **WSM module + networks** — the
  *predictor* is the output. Typically reads `w_t` from a frozen backbone.
- **`vla_training/train/train_wsm_*`** trains the **action policy** with a WSM head attached — the
  *policy* is the output, and it consumes the head defined here.

Final integration (roadmap stage R5) couples the chosen head into the policy via the **AdaLN-Zero**
pathway (`networks/adaln_zero.py`), whose zero-init gates make the WSM-augmented policy *exactly*
the base VLA at init. The headline eval column is **`composite_unseen`**. Governed by
`internal_planning_and_todos/04_wsm_roadmap.md`.
