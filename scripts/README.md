# scripts/ — orchestration (shell) + recipes (yaml)

One entry point per training regime plus a single eval entry. Each `train.sh` selects **pretrain vs
target-finetune** via `--phase` and routes to the matching backbone driver in `vla_training/`. The
shell scripts only resolve the plan (driver + config) and dispatch; the SageMaker-Batch submit is a
TODO (see `internal_planning_and_todos/03_infra_and_sagemaker.md`).

```
scripts/
  train/
    vla_base/train.sh       # regime R0: official RoboCasa protocol, no WSM
    vla_wsm/train.sh        # WSM base: + salient-keyframe-patch aux head      (configs TBD)
    vla_wsm_v2/train.sh     # WSM v2:   + 3D-flow / SigReg aux heads            (configs TBD)
  eval/eval.sh              # eval on the TARGET split (50-task protocol)
  configs/{train,eval}/     # yaml recipes — DEFAULT to the official RoboCasa pretrain/finetune recipe
```

Usage:

```bash
scripts/train/vla_base/train.sh --backbone groot_17 --phase pretrain \
    --config scripts/configs/train/groot_pretrain.yaml
scripts/train/vla_base/train.sh --backbone pi_05    --phase target_finetune \
    --config scripts/configs/train/pi05_finetune.yaml
scripts/eval/eval.sh --backbone groot_17 --config scripts/configs/eval/groot_eval.yaml
```

`--backbone` is `groot_17` (GR00T N1.7) or `pi_05` (pi0.5). Eval always runs on the `target` split
(`atomic_seen` 18 / `composite_seen` 16 / `composite_unseen` 16); the reported metric is the
task-weighted average success %, and **`composite_unseen` is the headline column**. Verified recipes
live in `internal_planning_and_todos/01_robocasa_protocol_and_recipes.md`.
