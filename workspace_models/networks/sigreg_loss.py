"""SigReg / LeJEPA (VICReg-style) regularized alignment over the workspace latent ``w_t``.

The cross-demo alignment (WSM core contribution) cannot be a bare MSE between matched same-task
latents: that is minimized by COLLAPSE (all latents -> one vector), and indeed at init the workspace
latents of different demos are already ~collinear (cosine ~0.99), so MSE alone gives no useful signal.
This is the standard VICReg/SigReg remedy: an INVARIANCE term that pulls matched positives together,
plus VARIANCE (hinge keeping each dim's std >= 1) and COVARIANCE (decorrelate dims) terms that
actively prevent collapse. Governed by internal_planning_and_todos/{04,07}.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def sigreg_epps_pulley(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 64,
    t_max: float = 5.0,
    num_points: int = 17,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SIGReg toward N(0,I) via the sketched Epps-Pulley statistic (LeJEPA, Algorithm 1).

    The TRUE SIGReg (vs the VICReg `SigRegLoss` class below). `x` [B,D] are PRE-normalization
    embeddings; directions are resampled every step and SHARED across ranks (seed = global_step) so
    the empirical CF is consistent under DDP all-reduce. DO NOT standardize the projections — SIGReg
    must see RAW projections to drive mean->0 AND var->1; standardizing hides collapse (the failure
    mode it exists to catch). Ported from DynaFLIP/dynaflip/losses/sigreg.py; see
    [[leworldmodel-and-lejepa]] + doc 12."""
    B, D = x.shape
    g = torch.Generator(device=x.device).manual_seed(int(global_step))
    A = torch.randn(D, num_slices, generator=g, device=x.device, dtype=x.dtype)
    A = A / (A.norm(dim=0, keepdim=True) + eps)  # unit directions on S^{D-1}
    P = x @ A  # [B, num_slices] RAW projections
    t = torch.linspace(-t_max, t_max, num_points, device=x.device, dtype=x.dtype)
    ang = P.unsqueeze(-1) * t.view(1, 1, -1)  # [B, S, T]
    re = torch.cos(ang).sum(0)
    im = torch.sin(ang).sum(0)
    n = torch.tensor([float(B)], device=x.device, dtype=x.dtype)
    if dist.is_available() and dist.is_initialized():
        # Plain in-place all_reduce sees global statistics forward but loses cross-rank gradients.
        from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce

        re = differentiable_all_reduce(re, op=dist.ReduceOp.SUM)
        im = differentiable_all_reduce(im, op=dist.ReduceOp.SUM)
        dist.all_reduce(n)
    re, im = re / n, im / n  # empirical CF, global over the DDP batch
    phiN = torch.exp(-0.5 * t**2)  # CF of N(0,1); also the Epps-Pulley weight
    integrand = ((re - phiN) ** 2 + im**2) * phiN
    stat = torch.trapz(integrand, t, dim=-1) * n  # quadrature over t, scaled by N (Alg. 1)
    return stat.mean()  # average over slices


class SigRegLoss(nn.Module):
    """VICReg/SigReg over matched workspace-latent positive pairs (same task, different demo).

    forward(w, w_pair): w, w_pair are [M, D] aligned positives (matched keyframes across two demos).
    Returns (total_loss, {inv, var, cov}). Variance/covariance need M >= 2."""

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        inv_weight: float = 25.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.inv_weight = inv_weight
        self.gamma = gamma  # target per-dim std
        self.eps = eps

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(self.gamma - std))

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        m, d = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (m - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        return off.pow(2).sum() / d

    def forward(self, w: torch.Tensor, w_pair: torch.Tensor) -> tuple[torch.Tensor, dict]:
        inv = F.mse_loss(w, w_pair)
        if w.shape[0] < 2:  # need >=2 samples for var/cov
            total = self.inv_weight * inv
            return total, {"inv": float(inv), "var": 0.0, "cov": 0.0}
        var = self._variance(w) + self._variance(w_pair)
        cov = self._covariance(w) + self._covariance(w_pair)
        total = self.inv_weight * inv + self.var_weight * var + self.cov_weight * cov
        return total, {"inv": float(inv), "var": float(var), "cov": float(cov)}
