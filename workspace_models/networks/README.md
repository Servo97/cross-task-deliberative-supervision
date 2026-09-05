# workspace_models/networks/ — WSM neural-network modules

The building blocks every WSM head shares, per
`internal_planning_and_todos/04_wsm_roadmap.md` ("Networks to build"). All heads read the **same**
workspace latent `w_t` so head variants stay comparable. Current files are **placeholder
skeletons** (intent docstring + class skeleton + TODO); framework is `torch.nn` (the GR00T side), a
jax/nnx mirror for pi0.5 is a TODO.

| Module | Role |
|---|---|
| `workspace_latent.py` (`WorkspaceLatentExtractor`) | Extract the shared `w_t` from a backbone's hidden states. |
| `adaln_zero.py` (`AdaLNZero`) | Zero-init modulation pathway; at init = exact base-VLA behavior (R5 integration). |
| `keyframe_patch_head.py` (`KeyframePatchHead`) | WSM-base head: predict salient keyframe patches from `w_t`. |
| `flow_head.py` (`Flow3DHead`) | WSM-v2 head: predict 3D flow / tracks from `w_t` (DynaFLIP target). |
| `sigreg_loss.py` (`SigRegLoss`) | E3 SigReg / LeJEPA regularized-representation loss over `w_t`. |

Data flow: backbone hidden states -> `WorkspaceLatentExtractor` -> `w_t` -> {`KeyframePatchHead`,
`Flow3DHead`, `SigRegLoss`}; for policy integration, the head signal re-enters the VLA through
`AdaLNZero`. The drivers in `workspace_models/train/` and `vla_training/train/train_wsm_*` both
import from here.
