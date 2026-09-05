# vla_training/ — the VLA (action-policy) training + eval code

Trains and evaluates the **action policy** (GR00T N1.7 / pi0.5) on RoboCasa365. The output of every
driver here is a *policy checkpoint*. Organized by regime:

```
vla_training/
  train/
    train_base/          # R0 base VLA: official pretrain -> target-finetune, no WSM
      {pretrain,finetune}_{groot_17,pi_05}.py
    train_wsm_base/      # base VLA + salient-keyframe-patch WSM aux head (joint loss)
      finetune_{groot_17,pi_05}_with_wsm.py
    train_wsm_v2/        # base VLA + 3D-flow (and SigReg) WSM aux heads (joint loss)
      finetune_{groot_17,pi_05}_with_wsm.py
  eval/                  # eval on the TARGET split (50-task protocol)
    eval_{groot_17,pi_05}.py
```

## vla_training vs workspace_models (the `train_wsm_*` split)

The WSM work has **two sides** and they live in different packages:

- **`vla_training/train/train_wsm_*`** trains the **VLA** with a WSM auxiliary head attached — the
  output is the **action policy**. It *consumes* the WSM head from `workspace_models`.
- **`workspace_models/train/*`** trains/defines the **WSM module + its networks** — the output is the
  **workspace predictor** itself (the research object).

Both can share the same head networks (`workspace_models/networks/`); the difference is which object
is the training target. Per the README naming discipline, plain RoboCasa finetuning is **base-VLA
training** (`train_base/`), *not* WSM. Recipes are governed by
`internal_planning_and_todos/01_robocasa_protocol_and_recipes.md` and `04_wsm_roadmap.md`.
