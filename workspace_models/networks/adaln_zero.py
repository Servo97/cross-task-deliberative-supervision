"""AdaLN-Zero transformer blocks (zero-init gates) — the shared building block of the WSM.

AdaLN-Zero produces (shift, scale, gate) from a conditioning vector; the gate is zero-initialized
so each residual branch starts inert (block == identity at init). For the workspace encoder/decoder
this is a benign init; for the future policy-integration modulator it is what makes the WSM-augmented
policy *exactly* the base VLA at step 0. Governed by internal_planning_and_todos/04_wsm_roadmap.md.

Mirrors the proven reference (Isaac-GR00T/wsm/head_v2.py: AdaLNZeroBlock), generalized to take an
explicit conditioning tensor so we can condition BOTH the temporal encoder and the slot decoder on
language (the 2026-06-18 locked decision).
"""

from __future__ import annotations

import torch
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class AdaLNZeroBlock(nn.Module):
    """Pre-LN self-attention + MLP block; affine + gates generated from a per-token condition.

    cond is broadcast/added over the 6 (shift, scale, gate) x (attn, mlp) parameters; gate is
    zero-init so at init ``forward(x, cond, mask) == x`` (identity residual)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x [B,L,D]; cond [B,L,D] (per token); attn_mask [L,L] bool, True = MASKED (torch MHA).
        sa_shift, sa_scale, sa_gate, mlp_shift, mlp_scale, mlp_gate = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), sa_shift, sa_scale)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + sa_gate * attn_out
        x = x + mlp_gate * self.mlp(modulate(self.norm2(x), mlp_shift, mlp_scale))
        return x


class AdaLNZeroCrossBlock(nn.Module):
    """Decoder block: slot self-attention + cross-attention to memory + MLP, all AdaLN-Zero
    conditioned (used by the salient-patch decoder; condition = language)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # slots [N,k,D]; memory [N,M,D]; cond [N,1,D] (broadcast over slots).
        # Optional (all default to the original behavior — the salient-patch decoder is unchanged):
        #   self_attn_mask [k,k] bool (True = masked) -> CAUSAL self-attn for the demo-fusion history;
        #   memory_key_padding_mask [N,M] bool (True = pad) -> masks invalid demo-window tokens;
        #   need_weights -> also return the cross-attn weights [N,k,M] (eval-time logging only).
        (sa_sh, sa_sc, sa_g, ca_sh, ca_sc, ca_g, m_sh, m_sc, m_g) = self.ada(cond).chunk(9, dim=-1)
        h = modulate(self.norm_q(slots), sa_sh, sa_sc)
        sa, _ = self.self_attn(h, h, h, attn_mask=self_attn_mask, need_weights=False)
        slots = slots + sa_g * sa
        h = modulate(self.norm_x(slots), ca_sh, ca_sc)
        mem = self.norm_m(memory)
        ca, ca_w = self.cross_attn(h, mem, mem, key_padding_mask=memory_key_padding_mask, need_weights=need_weights)
        slots = slots + ca_g * ca
        slots = slots + m_g * self.mlp(modulate(self.norm_x(slots), m_sh, m_sc))
        return (slots, ca_w) if need_weights else slots
