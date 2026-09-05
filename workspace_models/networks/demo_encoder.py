"""DemoEncoder — bidirectional, PROPRIO-FREE encoder over a demo video's pooled frozen-VLM tokens.

The demo2 branch of WSMv2 (doc 15, D1–D4): consumes per-grid-frame POOLED tokens (p.npz, produced offline
by features/pool_patch_tokens.py with the frozen WSMv1 patch_in_norm + PatchPool — vision + language only,
NO proprio, so a human video drops in later with zero interface change) and re-encodes them with FULL
bidirectional attention (the demo is a completely observed reference video, at train and at eval).

D4 (critique-hardened): NO absolute or normalized-progress positional encoding anywhere in this module.
The fusion's window is centered proportionally to the rollout's progress, so any absolute-phase channel in
the demo tokens would leak `t/T1` into z — a shortcut that (a) inflates the Δ-gates and (b) SURVIVES the
temporally-shuffled control (shuffling permutes tokens but not the PE *set*). Temporal ORDER enters only
through a distance-bucketed RELATIVE attention bias (Toeplitz: bias[i,j] = f(i-j), shared across heads,
built per forward) — order without phase.

Perf (perf-first mandate): everything batched, no python loops over batch/time; the bias matrix is O(M^2)
ints + one embedding gather; bf16-autocast and torch.compile friendly (no data-dependent control flow).
"""

from __future__ import annotations

import torch
from torch import nn

from workspace_models.networks.adaln_zero import AdaLNZeroBlock


class DemoEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        lang_dim: int = 2048,
        n_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        max_rel: int = 64,
    ) -> None:
        super().__init__()
        self.dim, self.n_heads, self.max_rel = dim, n_heads, max_rel
        # Input LayerNorm: pooled-token SCALE varies wildly by frozen encoder (orig_65k RMS ~5300 — no
        # patch_in_norm in that ckpt — vs v2_50k RMS ~14). v1 absorbs this via pre-LN inside its blocks;
        # here the consumer must be encoder-agnostic (and human-video-agnostic), so normalize per token.
        self.in_norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)  # pooled tokens are already dim-d; light adapt
        self.lang_proj = nn.Linear(lang_dim, dim)
        self.null_lang = nn.Parameter(torch.zeros(dim))  # lang-dropout target (D13: p≈0.5 at train)
        # relative attention bias: one scalar per clamped signed distance, shared across heads/layers.
        self.rel_bias = nn.Embedding(2 * max_rel + 1, 1)
        nn.init.zeros_(self.rel_bias.weight)
        self.blocks = nn.ModuleList(AdaLNZeroBlock(dim, n_heads, mlp_ratio) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim)

    def _bias(self, m: int, device: torch.device) -> torch.Tensor:
        """[m,m] float relative-distance bias (Toeplitz by construction)."""
        i = torch.arange(m, device=device)
        rel = (i[None, :] - i[:, None]).clamp(-self.max_rel, self.max_rel) + self.max_rel
        return self.rel_bias(rel).squeeze(-1)  # [m,m]

    def forward(
        self,
        tokens: torch.Tensor,
        lang: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lang_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens [B,M,dim] pooled frame tokens; lang [B,lang_dim]; pad_mask [B,M] bool True=PAD;
        lang_keep [B] bool (False -> the learned null_lang; train-time lang-dropout). Returns d [B,M,dim].
        Padded positions produce garbage outputs — mask them downstream (they are key-masked here, so
        they never contaminate valid tokens)."""
        b, m, _ = tokens.shape
        x = self.in_proj(self.in_norm(tokens))
        cond = self.lang_proj(lang)  # [B,dim]
        if lang_keep is not None:
            cond = torch.where(lang_keep[:, None], cond, self.null_lang.expand(b, -1).to(cond.dtype))
        cond = cond[:, None, :].expand(b, m, -1)  # per-token broadcast (AdaLNZeroBlock API)

        # additive float mask: relative bias everywhere + large-negative on padded KEYS. Padded queries
        # still see valid keys (no all-masked softmax rows -> no NaN); their outputs are discarded.
        attn = self._bias(m, x.device)[None, None].expand(b, self.n_heads, m, m)
        if pad_mask is not None:
            attn = attn + pad_mask[:, None, None, :].float() * torch.finfo(x.dtype).min / 2
        attn = attn.reshape(b * self.n_heads, m, m)

        for blk in self.blocks:
            x = blk(x, cond, attn_mask=attn)
        return self.out_norm(x)
