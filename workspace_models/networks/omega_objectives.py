"""The ω-training objective family, extracted from the RoboCerebra a2 retrain (s4 §2.2, G5).

Provenance: every term below is lifted from `scripts/robocerebra/train_omega_retrain_lite.py`
(lines 191-237 of the sealed a2 run) rather than re-derived, so the H14 Stage-E encoder inherits
the *measured* recipe, not a paraphrase of it. a2 is the only configuration on record that moved
`between_episode_variance_fraction` off frozen levels (0.042 -> 0.559); the terms it needs are:

  jepa_loss          EMA-target prediction (MSE + 1-cos). The only term asking ω to be predictive
                     of the future, i.e. a workspace state rather than a frame embedding.
  sigreg_term        Epps-Pulley anti-collapse, with the a2 RANK CAP (attempt 1 whitened to
                     eff-rank 84 against an in-domain reference of 10.9 — clearing the floor by
                     overshooting it while the representation stayed useless).
  supcon             Weighted supervised contrastive over frames. `supcon_episode` is the a2 term
                     with positives = same-episode frames; `supcon_deliberative` (NEW, H14) is the
                     same kernel with positives = frames whose SEGMENTS carry a Qwen
                     EQUIVALENT/ANALOGOUS edge and with CONTRAST-linked frames upweighted in the
                     denominator (amendment A2: FRAME level, never a mean-pooled z_seg — a w16 GDN
                     window spans ~1.1 segments, so a segment-pooled objective is invariant to the
                     content the read actually consumes).

The -inf x 0 = NaN trap of the original (:231-233) is preserved verbatim: self entries are
masked-filled to 0.0 in log-space BEFORE the masked sum, never left to a boolean mask.

`supcon_discriminative_stat` is the gate-G4 telemetry the H13 lesson forces (aux-gate-g4:
a contrastive aux whose normalised discriminative term never beats chance by end-of-canary is a
HOLD, even when every flow-parity/rising/grad-norm signal passes). It is computed under no_grad
next to the loss so the canary can watch the term beat chance rather than merely descend.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from workspace_models.networks.sigreg_loss import sigreg_epps_pulley

__all__ = [
    "jepa_loss",
    "batch_effective_rank",
    "sigreg_term",
    "supcon",
    "supcon_episode",
    "supcon_deliberative",
    "supcon_discriminative_stat",
]


def jepa_loss(predicted: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """a2 :198-199 — cosine supplies direction (the axis G1 found collapsed), MSE supplies scale."""
    return F.mse_loss(predicted, wanted) + (1 - F.cosine_similarity(predicted, wanted, dim=-1)).mean()


@torch.no_grad()
def batch_effective_rank(sample: torch.Tensor) -> float:
    """Participation ratio of the sample covariance. a2 :211-215, verbatim."""
    centered = sample - sample.mean(0, keepdim=True)
    cov = (centered.T @ centered / max(len(centered) - 1, 1)).float()
    eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0)
    return float(eigenvalues.sum() ** 2 / eigenvalues.pow(2).sum().clamp_min(1e-12))


def sigreg_term(
    sample: torch.Tensor, step: int, lambda_sigreg: float, rank_cap: float
) -> tuple[torch.Tensor, float, float]:
    """SIGReg with the a2 rank cap. Returns (loss, effective lambda, measured batch rank).

    Once the batch is already spread past the cap, SIGReg has done its job and is only destroying
    structure, so stop paying it (a2 :204-217).
    """
    value = sigreg_epps_pulley(sample, step)
    rank = float("nan")
    effective = lambda_sigreg
    if rank_cap > 0:
        rank = batch_effective_rank(sample)
        if rank > rank_cap:
            effective = 0.0
    return value, effective, rank


def supcon(
    features: torch.Tensor,
    positive: torch.Tensor,
    *,
    tau: float = 0.1,
    positive_weight: torch.Tensor | None = None,
    negative_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted supervised-contrastive loss over already-flattened frame features.

    features         [N, D] — normalised internally, exactly as a2 :224 does.
    positive         [N, N] bool — True where j is a positive of i. The diagonal is cleared here.
    positive_weight  [N, N] float or None — per-positive weight (edge type x confidence). None = 1.
    negative_weight  [N, N] float or None — per-negative multiplier inside the denominator; this is
                     how CONTRAST becomes a *hard* negative rather than one negative among many.
                     None = 1. Entries <= 0 remove that pair from the denominator entirely.

    Returns a scalar; rows with no positive contribute nothing (and cannot make the mean NaN).
    """
    z = F.normalize(features, dim=-1)
    logits = (z @ z.T) / tau
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)

    if negative_weight is not None:
        # log-domain reweighting of the denominator: logsumexp(logit + log w).
        bias = torch.log(negative_weight.clamp_min(1e-12))
        bias = bias.masked_fill(negative_weight <= 0, float("-inf"))
        logits_for_denominator = logits + bias
    else:
        logits_for_denominator = logits
    logits_for_denominator = logits_for_denominator.masked_fill(self_mask, float("-inf"))

    positive = positive & ~self_mask
    log_prob = logits - torch.logsumexp(logits_for_denominator, dim=1, keepdim=True)
    # a2 :231-233 — the self entries are -inf; -inf * 0.0 is NaN, so zero them in log-space before
    # the masked sum rather than relying on the boolean mask to suppress them.
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    weight = positive.float() if positive_weight is None else positive_weight * positive.float()
    denominator = weight.sum(1)
    rows = denominator > 0
    if not bool(rows.any()):
        return features.sum() * 0.0
    per_row = -(log_prob * weight).sum(1)[rows] / denominator[rows]
    return per_row.mean()


def supcon_episode(omega: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """a2 :219-235 — positives are frames of the same episode. omega is [B, T, D]."""
    b, t, d = omega.shape
    flat = omega.reshape(-1, d)
    episode_of = torch.arange(b, device=omega.device).repeat_interleave(t)
    return supcon(flat, episode_of[:, None] == episode_of[None, :], tau=tau)


def supcon_deliberative(
    features: torch.Tensor,
    segment_of: torch.Tensor,
    positive_weight_by_segment: torch.Tensor,
    contrast_by_segment: torch.Tensor,
    *,
    tau: float = 0.1,
    contrast_weight: float = 2.0,
    negative_multiplier_by_segment: torch.Tensor | None = None,
) -> torch.Tensor:
    """H14 SupCon_deliberative, FRAME level (amendment A2).

    features                    [N, D] frame features (already flattened, padding removed).
    segment_of                  [N] int64 — the global segment key of each frame.
    positive_weight_by_segment  [S, S] float — edge weight between segment keys present in this
                                batch (0 = not a positive). Built by the sampler from the
                                content-addressed label artifact.
    contrast_by_segment         [S, S] bool — CONTRAST edges between those segment keys.

    Segment keys are *batch-local* indices into the two [S, S] tables, so the tables stay small
    (S = segments realised in this batch) instead of |corpus|^2.

    `negative_multiplier_by_segment` (label artifact v2) supplies a PER-PAIR denominator multiplier
    already in multiplier space, overriding the single `contrast_weight` scalar. It exists because
    v2 grades hard negatives by evidence: a binding-corroborated CONTRAST is repulsive at full
    strength, a Qwen-only CONTRAST at half. A multiplier of 1.0 means "an ordinary negative", never
    "excluded" — excluding a pair is a different intervention, and the two must not be conflated.
    """
    pw = positive_weight_by_segment[segment_of][:, segment_of]
    positive = pw > 0
    if negative_multiplier_by_segment is not None:
        negative_weight = negative_multiplier_by_segment[segment_of][:, segment_of].clamp_min(1e-6)
    else:
        negative_weight = torch.ones_like(pw)
        hard = contrast_by_segment[segment_of][:, segment_of]
        negative_weight = torch.where(hard, torch.full_like(negative_weight, contrast_weight), negative_weight)
    return supcon(features, positive, tau=tau, positive_weight=pw, negative_weight=negative_weight)


@torch.no_grad()
def supcon_discriminative_stat(
    features: torch.Tensor,
    positive: torch.Tensor,
    candidate: torch.Tensor | None = None,
) -> dict:
    """Gate-G4 telemetry: does the contrastive term actually DISCRIMINATE, or only descend?

    For every row that has at least one positive and at least one non-positive candidate, ask
    whether the nearest candidate is a positive. `chance` is the per-row positive density over the
    candidate set, averaged the same way — so `lift` is 1.0 for a representation that carries no
    information at all, whatever the loss value is doing.

    candidate  [N, N] bool or None — which columns are eligible (e.g. cross-task frames only).
               None = every column except self.
    """
    z = F.normalize(features, dim=-1)
    similarity = z @ z.T
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    eligible = (~self_mask) if candidate is None else (candidate & ~self_mask)
    positive = positive & eligible
    rows = (positive.sum(1) > 0) & ((eligible & ~positive).sum(1) > 0)
    if not bool(rows.any()):
        return {"top1": float("nan"), "chance": float("nan"), "lift": float("nan"), "n_rows": 0}
    similarity = similarity.masked_fill(~eligible, float("-inf"))
    top1 = similarity.argmax(1)
    hit = positive[torch.arange(len(z), device=z.device), top1].float()[rows]
    density = (positive.sum(1).float() / eligible.sum(1).clamp_min(1).float())[rows]
    top1_mean, chance = float(hit.mean()), float(density.mean())
    return {
        "top1": top1_mean,
        "chance": chance,
        "lift": float(top1_mean / max(chance, 1e-9)),
        "n_rows": int(rows.sum()),
    }
