"""WSM neural-network modules (per internal_planning_and_todos/04_wsm_roadmap.md "Networks to build").

- workspace_latent : WorkspaceEncoder (+ PatchPool) — per-step (pooled VLM + proprio + lang) -> causal w_t.
- adaln_zero       : AdaLN-Zero transformer blocks (zero-init gates); self-attn + cross-attn variants.
- keyframe_patch_head : SalientPatchDecoder — w_t -> 32 slots (presence + frozen-VLM-feature recon).
- wsm_model        : WSMConfig + WorkspaceModel (encoder + decoder; the WSM-base frozen-probe object).
- flow_head        : WSM-v2 head predicting 3D flow / tracks from ``w_t`` (DynaFLIP target). [stub]
- sigreg_loss      : SigReg / LeJEPA regularized-representation loss over ``w_t``. [stub]

torch.nn (the GR00T side); a jax/nnx mirror for pi0.5 is TODO.
"""

from workspace_models.networks.adaln_zero import AdaLNZeroBlock, AdaLNZeroCrossBlock
from workspace_models.networks.flow_head import Flow3DHead
from workspace_models.networks.keyframe_patch_head import SalientPatchDecoder
from workspace_models.networks.sigreg_loss import SigRegLoss
from workspace_models.networks.workspace_latent import PatchPool, WorkspaceEncoder
from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig

__all__ = [
    "AdaLNZeroBlock",
    "AdaLNZeroCrossBlock",
    "Flow3DHead",
    "PatchPool",
    "SalientPatchDecoder",
    "SigRegLoss",
    "WorkspaceEncoder",
    "WorkspaceModel",
    "WSMConfig",
]
