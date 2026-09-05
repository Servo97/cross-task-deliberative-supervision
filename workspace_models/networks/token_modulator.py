"""TokenModulator — the WSM->policy interface (AdaLN-Zero modulation of the policy's backbone tokens).

History rewrites the present: from a short CAUSAL window of K workspace latents w_{t-..t} (+ the global
language embedding) we generate per-step (alpha, beta, gamma) over the backbone token dim and apply

    tokens' = tokens + gamma * (alpha * LN(tokens) + beta)

to every backbone token of the current step. The generator's last layer is zero-init, so gamma == 0 at
init => tokens' == tokens EXACTLY => the WSM-conditioned post-train starts as the unmodified Eval1
baseline and can only depart from it if the optimizer finds w useful ("can only help if the signal is
real"). The action head consumes tokens' in place of the raw tokens.

Generalized from the single-w reference Isaac-GR00T/wsm/head_v2.py:98-116 to a K-token window
(K=1 recovers the reference / the dossier's causal last-neighbor; K=2-4 adds recent-dynamics context
for the action chunk). The causal window itself is gathered upstream (workspace_models.features.wsm_align).
"""

from __future__ import annotations

import torch
from torch import nn


class TokenModulator(nn.Module):
    def __init__(self, w_dim: int = 512, lang_dim: int = 2048, token_dim: int = 2048, k_window: int = 2) -> None:
        """w_dim = WorkspaceEncoder latent dim; token_dim = policy backbone token dim (GR00T 2048);
        k_window = #workspace latents conditioned on (current + k-1 past grid tokens)."""
        super().__init__()
        self.k_window = k_window
        self.lang_proj = nn.Linear(lang_dim, w_dim)
        self.norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.gen = nn.Sequential(nn.SiLU(), nn.Linear(k_window * w_dim + w_dim, 3 * token_dim))
        nn.init.zeros_(self.gen[-1].weight)  # zero-init => exact identity at step 0
        nn.init.zeros_(self.gen[-1].bias)

    def forward(self, tokens: torch.Tensor, w_window: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """tokens [..., P, token_dim]; w_window [..., K, w_dim] (oldest..newest, causal);
        lang [..., lang_dim] (broadcast over P). dtype-robust: GR00T mixes bf16 backbone tokens with
        fp32 trainable params, so compute the modulation in the modulator's param dtype and add the
        delta back in the token dtype (zero-init => delta is exactly 0 => bitwise identity at step 0)."""
        *lead, k, dw = w_window.shape
        assert k == self.k_window, f"w_window K={k} != configured k_window={self.k_window}"
        dt = self.gen[-1].weight.dtype
        cond = torch.cat([w_window.reshape(*lead, k * dw).to(dt), self.lang_proj(lang.to(dt))], dim=-1)
        alpha, beta, gamma = self.gen(cond).chunk(3, dim=-1)  # each [..., token_dim]
        alpha, beta, gamma = (x.unsqueeze(-2) for x in (alpha, beta, gamma))  # broadcast over P
        # (1 + alpha): the scale is offset by 1 so the inner modulation is IDENTITY (not zero) at init.
        # Without it, gen is fully zero-init => alpha=beta=gamma=0 => delta=gamma*(alpha*LN+beta) is a
        # product of zeros whose gradient w.r.t. every factor is 0 => DEAD gradient, the modulator can
        # never leave zero-init (verified: gen.weight.grad==0). With (1+alpha), d(delta)/d(gamma)=LN(tokens)
        # at init => gamma (the gate) gets a real gradient, opens, then alpha/beta follow. delta is STILL
        # exactly 0 at init (gamma=0) so the WSM post-train still starts as the exact Eval1 baseline.
        delta = gamma * ((1.0 + alpha) * self.norm(tokens.to(dt)) + beta)
        return tokens + delta.to(tokens.dtype)
