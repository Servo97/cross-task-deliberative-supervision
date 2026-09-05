"""Classifier-free-guidance (CFG) workspace conditioner for the VLA action head.

Turns the workspace latents [w_t, w_{t+1}] into an ADDITIVE AdaLN conditioning vector that rides on the
DiT's timestep embedding (`temb`) — so it steers ONLY the action denoiser, never the VLM/perception
tokens. Each component is independently dropped to a LEARNED NULL during training (CFG); at eval the
serve runs a conditional + an unconditional pass and extrapolates the velocity:
    v_guided = v_uncond + s * (v_cond - v_uncond).

POC: only `w_t` is fed (w_next=None -> its null), with per-component dropout p_drop on w_t. The `w_next`
slot (params + null) is BUILT NOW so demo-ICL — which conditions on current AND future demo tokens — is a
finetune-on swap, not a new architecture. Output projections are ZERO-INIT, so at step 0 the conditioning
is exactly 0 -> `temb` unchanged -> the policy is byte-identical to the baseline finetune (and the
conditional/unconditional paths only diverge as training proceeds). See doc 12 + [[wsm-jepa-penultimate-model-free-design]].
"""

from __future__ import annotations

import torch
from torch import nn


class WSMCfgConditioner(nn.Module):
    def __init__(self, w_dim: int = 512, cond_dim: int = 512, p_drop: float = 0.2, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or cond_dim
        self.w_dim, self.cond_dim, self.p_drop = w_dim, cond_dim, p_drop
        self.null_t = nn.Parameter(torch.zeros(w_dim))  # learned null embeddings (the "uncond" input)
        self.null_next = nn.Parameter(torch.zeros(w_dim))
        self.proj_t = nn.Sequential(nn.Linear(w_dim, h), nn.SiLU(), nn.Linear(h, cond_dim))
        self.proj_next = nn.Sequential(nn.Linear(w_dim, h), nn.SiLU(), nn.Linear(h, cond_dim))
        for proj in (self.proj_t, self.proj_next):  # AdaLN-Zero: cond==0 at init -> baseline
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def _component(self, w, null, keep, B, dev, dt):
        """w [B,w_dim] or None -> projected component [B,cond_dim]; rows with keep=False use `null`."""
        nullB = null.to(device=dev, dtype=dt).unsqueeze(0).expand(B, -1)
        x = nullB if w is None else torch.where(keep.unsqueeze(-1), w.to(device=dev, dtype=dt), nullB)
        return x

    def forward(self, w_t, w_next, *, training: bool, force_uncond: bool = False) -> torch.Tensor:
        """w_t / w_next: [B, w_dim] or None. Returns cond [B, cond_dim] to ADD to the DiT temb.
        training -> independent per-component Bernoulli(p_drop) drop-to-null; force_uncond -> all null."""
        ref = w_t if w_t is not None else w_next
        if ref is None:
            raise ValueError("WSMCfgConditioner.forward needs at least one of w_t / w_next (or a batch ref)")
        B = ref.shape[0]
        # Use the conditioner's OWN param device/dtype as authoritative (inputs get moved to it). At train
        # the collator already puts w on the batch device; at eval w may arrive on the encoder device — this
        # makes the proj_t/proj_next Linear (on the param device) safe regardless. (review fix: device mismatch)
        p = next(self.parameters())
        dev, dt = p.device, p.dtype
        ones = torch.ones(B, dtype=torch.bool, device=dev)
        zeros = torch.zeros(B, dtype=torch.bool, device=dev)

        def keep_mask(w):
            if w is None or force_uncond:
                return zeros  # absent or forced-uncond -> use null
            if training:
                return torch.rand(B, device=dev) >= self.p_drop  # CFG dropout
            return ones  # eval conditional -> keep

        ct = self.proj_t(self._component(w_t, self.null_t, keep_mask(w_t), B, dev, dt))
        cn = self.proj_next(self._component(w_next, self.null_next, keep_mask(w_next), B, dev, dt))
        return ct + cn
