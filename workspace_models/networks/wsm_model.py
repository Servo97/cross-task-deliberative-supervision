"""WorkspaceModel — the WSM-base module: workspace encoder + salient-patch decoder.

The frozen-probe object (roadmap Stage-1, locked 2026-06-18): trained on cached features from the
FROZEN GR00T backbone; the base VLA never updates. Per-step inputs (pooled frozen-VLM patches +
proprio + language) -> causal AdaLN-Zero stream -> omega_t (WorkspaceEncoder); each omega_t -> 32 slots ->
presence + frozen-VLM-feature recon (SalientPatchDecoder). The policy-integration modulator
(AdaLN-Zero into the GR00T action head) is a later stage and lives elsewhere.

GR00T N1.7 dims (from backbone_tap): backbone_dim=2048 (Qwen3-VL patch tokens), proprio_dim=1536
(CategorySpecificMLP state-encoder emb), lang_dim=2048 (masked-mean text emb).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from workspace_models.networks.keyframe_patch_head import SalientPatchDecoder
from workspace_models.networks.workspace_latent import WorkspaceEncoder


@dataclass
class WSMConfig:
    dim: int = 512
    n_layers: int = 4  # temporal encoder blocks
    n_dec_layers: int = 2  # slot-decoder blocks
    n_heads: int = 8
    k_slots: int = 32  # salient-patch slots (locked: 32)
    backbone_dim: int = 2048  # frozen-VLM patch-token dim (pool input + recon target)
    proprio_dim: int = 1536  # GR00T state-encoder emb (pi0.5 would use its raw state dim)
    lang_dim: int = 2048  # language embedding dim (backbone text space)
    c_horizon: int = 1000  # causal attention window (steps)
    max_t: int = 1200  # timestep-embedding table (longest RoboCasa episode + margin)
    mlp_ratio: float = 4.0
    input_norm: bool = False  # LayerNorm the raw (patch/proprio/lang) inputs before projection.
    # OFF by default so pre-existing encoders load unchanged; the stabilized
    # groot retrain sets it True (GR00T patch RMS ~4.3 raw -> forward overflow).


class WorkspaceModel(nn.Module):
    def __init__(self, cfg: WSMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WSMConfig()
        self.encoder = WorkspaceEncoder(self.cfg)
        self.decoder = SalientPatchDecoder(self.cfg)

    def encode(self, patches: torch.Tensor, proprio: torch.Tensor, cond_lang: torch.Tensor) -> torch.Tensor:
        """Raw frame inputs -> workspace tokens omega [B,T,dim]."""
        return self.encoder(patches, proprio, cond_lang)

    def fuse_inputs(
        self,
        patches: torch.Tensor,
        proprio: torch.Tensor,
        cond_lang: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project raw per-frame inputs into cacheable fused tokens and AdaLN conditions."""
        return self.encoder.fuse_inputs(patches, proprio, cond_lang)

    def encode_fused(self, fused: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run the causal temporal encoder over cached per-frame fused inputs."""
        return self.encoder.encode_fused(fused, cond)

    def decode(self, w_flat: torch.Tensor, lang_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """w_flat [N,dim] (sampled supervised timesteps), lang_flat [N,lang_dim] ->
        recon [N,k,backbone_dim], occ [N,k]."""
        return self.decoder(w_flat, lang_flat)
