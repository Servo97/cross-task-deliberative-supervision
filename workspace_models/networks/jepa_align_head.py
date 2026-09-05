"""Model-free WSM aux head: JEPA-align the action-head PENULTIMATE features to the next workspace
latent `w_{t+1}`, + SIGReg isotropy on those features. The frozen encoder's `w` is a *training
target only* (precomputed) — never injected, never in the inference graph — so the eval-OOD failure
mode of the injection recipe is structurally impossible. See internal_planning_and_todos/12 +
[[wsm-jepa-penultimate-model-free-design]].

Pure-torch + testable: no gr00t/jax imports. The GR00T action-head wiring lives in
vla_training/train/train_base/_groot_wsm_jepa_common.py.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from workspace_models.networks.sigreg_loss import sigreg_epps_pulley


class JEPAPredictor(nn.Module):
    """penult (pooled) -> predicted next workspace latent(s). BYOL/JEPA-style predictor decouples
    "be predictive of w_{t+1}" from "be decodable to actions". `direct=True` = no predictor
    (penult IS the prediction; a bare Linear only if dims differ) — the ablation variant.

    `num_futures` (k) = how many consecutive future grid latents the head predicts. k > 1 keeps the
    SHARED trunk (Linear -> GELU) and widens ONLY the output projection to k*w_dim, reshaped to
    [..., k, w_dim] — a multi-head readout off one representation, not k predictors. At the default
    k == 1 every parameter SHAPE is unchanged, so existing checkpoints load byte-identically.
    Mirrors the JAX twin `openpi/models/wsm_jepa.py::WSMJepaHead`.
    """

    def __init__(
        self, in_dim: int, w_dim: int, hidden: int | None = None, direct: bool = False, num_futures: int = 1
    ) -> None:
        super().__init__()
        if num_futures < 1:
            raise ValueError(f"num_futures must be >= 1, got {num_futures}")
        if direct and num_futures > 1:
            # "penult IS the prediction" has no meaning for k targets; a widened direct head would
            # silently be a plain Linear, i.e. not the direct ablation at all.
            raise ValueError(f"direct=True is incompatible with num_futures={num_futures} (>1)")
        self.w_dim = int(w_dim)
        self.num_futures = int(num_futures)
        out_dim = self.w_dim * self.num_futures
        if direct:
            self.net = nn.Identity() if in_dim == w_dim else nn.Linear(in_dim, w_dim)
        else:
            h = hidden or w_dim
            self.net = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.num_futures == 1:
            return out
        return out.reshape(*out.shape[:-1], self.num_futures, self.w_dim)


def wsm_jepa_sigreg_loss(
    penult_act: torch.Tensor,  # [B, H, Dp] per-action-token penultimate features (with grad)
    w_next: torch.Tensor,  # [B, Dw] (k=1) or [B, k, Dw] frozen future target(s)
    predictor: JEPAPredictor,
    *,
    jepa_weight: float = 1.0,
    sigreg_weight: float = 0.05,
    global_step: int = 0,
    sigreg_per_token: bool = True,
    target_valid: torch.Tensor | None = None,
    num_futures: int = 1,
) -> tuple[torch.Tensor, dict]:
    """L = jepa_weight·(1 - cos(predictor(mean_H penult), w_next)) + sigreg_weight·SIGReg(penult).

    JEPA target is a precomputed constant => natural stop-grad. SIGReg runs on the RAW (pre-norm)
    penult so isotropy is enforced, not divided out. The SIGReg term is normalized by the row count
    (see the `/n` comment at the call site below), so `sigreg_weight` is on the same footing as
    `jepa_weight` and as `--lam-sig` in the torch WSM trainer: it does NOT scale with B or the action
    horizon. Returns (total, metrics).

    `num_futures` (k) is the head's prediction horizon. The arithmetic below is rank-agnostic: with
    k > 1 the head emits [B, k, Dw], the cosine is per (sample, future) -> [B, k], and the mask is
    [B, k], so the FLATTENED masked mean is the mean over BOTH B and k. That keeps `jepa_weight`
    invariant to k (and to batch size) — an n-scaled aux term is the exact bug class that collapsed
    the first s3 run (see the /n note below). The argument is also a SHAPE GUARD: without it a
    k-mismatched loader would broadcast [B, Dw] against [B, k, Dw] into a silently wrong loss
    instead of failing.
    """
    if num_futures < 1:
        raise ValueError(f"num_futures must be >= 1, got {num_futures}")
    want = (
        (penult_act.shape[0], w_next.shape[-1])
        if num_futures == 1
        else (penult_act.shape[0], num_futures, w_next.shape[-1])
    )
    if tuple(w_next.shape) != want:
        raise ValueError(
            f"num_futures={num_futures} expects w_next {want}, got {tuple(w_next.shape)} — "
            "the loader's k must match the model's"
        )
    if target_valid is not None and tuple(target_valid.shape) != want[:-1]:
        raise ValueError(
            f"num_futures={num_futures} expects target_valid {want[:-1]}, got {tuple(target_valid.shape)}"
        )
    pooled = penult_act.mean(dim=1)  # [B, Dp] (keeps penult dtype)
    pp = next(predictor.parameters(), None)  # predictor may be bf16 under the trainer
    pdtype = pp.dtype if pp is not None else pooled.dtype  # (direct=Identity has no params)
    pred = predictor(pooled.to(pdtype)).float()  # [B, Dw] (k=1) or [B, k, Dw]
    tgt = w_next.float()
    cos_each = F.cosine_similarity(pred, tgt, dim=-1)  # [B] or [B, k]
    cos_flat = cos_each.reshape(-1)
    if target_valid is not None:
        valid = target_valid.to(device=cos_flat.device, dtype=torch.bool).reshape(-1)
        if valid.numel() != cos_flat.numel():
            raise ValueError(f"target_valid has {valid.numel()} rows, expected {cos_flat.numel()}")
        if bool(valid.any()):
            cos = cos_flat[valid].mean()  # masked MEAN over B and k => k-invariant
            jepa = 1.0 - cos
        else:
            cos = cos_flat.sum() * 0.0
            jepa = cos  # graph-connected zero; SIGReg still trains the representation
    else:
        cos = cos_flat.mean()
        jepa = 1.0 - cos

    sig_in = penult_act.reshape(-1, penult_act.shape[-1]) if sigreg_per_token else pooled
    # /n: SAMPLE-COUNT-INVARIANT statistic. `sigreg_epps_pulley` returns the LeJEPA Alg.-1 form,
    # deliberately scaled by n for its chi-square asymptotics under H0; every trainer call site
    # divides it straight back out (`train_wsm2_icl.py:362  ... / args.batch  # /B: batch-invariant`,
    # and :364 dividing by the ROW count, which is the general rule). This call site did not, and the
    # JAX port inherited the omission: with per-token features n = B*H the configured
    # sigreg_weight=0.05 acted as ~160, the aux was 99% of the step-0 gradient for 60k steps, and the
    # pi/s3 arm evaluated at 12.4% vs a 55.8% base. Full autopsy:
    # internal_planning_and_todos/jul_31/s3_collapse_forensics.md (the JAX twin was fixed 2026-07-31
    # by folding /n INSIDE the function; here the function is shared with train_wsm2_icl.py, which
    # already divides at its own call sites, so the fix belongs here instead of in sigreg_loss.py).
    # Under DDP the function all-reduces (re, im, n), so the n it used is the GLOBAL row count.
    n_rows = sig_in.shape[0]
    if dist.is_available() and dist.is_initialized():
        n_rows *= dist.get_world_size()
    sig = sigreg_epps_pulley(sig_in.float(), global_step) / n_rows

    total = jepa_weight * jepa + sigreg_weight * sig
    return total, {
        "jepa": float(jepa.detach()),
        "cos": float(cos.detach()),
        "sigreg": float(sig.detach()),
        "jepa_valid": int(target_valid.sum().detach()) if target_valid is not None else int(cos_flat.numel()),
        "jepa_total": int(cos_flat.numel()),
        "jepa_w": jepa_weight,
        "sigreg_w": sigreg_weight,
        "num_futures": int(num_futures),
    }
