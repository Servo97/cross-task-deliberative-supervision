"""Salient-patch decoder (WSM-base head): w_t -> a set of salient patches.

DETR-style set prediction (the 2026-06-18 locked design): a fixed number of learned slot queries
cross-attend the workspace latent w_t and each emits (a) a presence logit `occ` (is this slot
used?) and (b) a reconstruction of the FROZEN-VLM patch feature at a salient patch. The decoder is
LANGUAGE-CONDITIONED (AdaLN), since saliency is task-dependent. Trained with Hungarian matching
against the keyframe salient-patch labels (workspace_models/labels/), recon in frozen-VLM-feature
space (backbone_dim). w_t is a single token per timestep -> a hard bottleneck the slots must unpack.
Governed by internal_planning_and_todos/04_wsm_roadmap.md + [[wsm-recipe-first-cut]].
"""

from __future__ import annotations

import torch
from torch import nn

from workspace_models.networks.adaln_zero import AdaLNZeroCrossBlock


class SalientPatchDecoder(nn.Module):
    """w_t [N,dim] + language [N,lang_dim] -> recon [N,k,backbone_dim], occ [N,k] (presence logit)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(cfg.k_slots, cfg.dim) * 0.02)
        self.lang_proj = nn.Linear(cfg.lang_dim, cfg.dim)
        self.blocks = nn.ModuleList(
            AdaLNZeroCrossBlock(cfg.dim, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_dec_layers)
        )
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.recon_head = nn.Linear(cfg.dim, cfg.backbone_dim)  # -> frozen-VLM patch-feature space
        self.occ_head = nn.Linear(cfg.dim, 1)

    def forward(self, w: torch.Tensor, lang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # w [N,dim] (flattened supervised timesteps); lang [N,lang_dim].
        n = w.shape[0]
        slots = self.slots.unsqueeze(0).expand(n, -1, -1)  # [N,k,dim]
        cond = self.lang_proj(lang).unsqueeze(1)  # [N,1,dim] (broadcast over slots)
        memory = w.unsqueeze(1)  # [N,1,dim] (the single w_t token)
        for blk in self.blocks:
            slots = blk(slots, memory, cond)
        slots = self.out_norm(slots)
        return self.recon_head(slots), self.occ_head(slots).squeeze(-1)
