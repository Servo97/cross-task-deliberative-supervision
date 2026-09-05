#!/usr/bin/env python3
"""Label-free domain-adaptation retrain of the workspace encoder for RoboCerebra.

Why this exists: the canonical robocasa-trained ω encoder collapses on RoboCerebra (G1 — temporal
coherence gap 0.030 vs 0.785 in-domain, effective rank 6.2 vs 10.9). Every mechanism arm consumes
ω, so the arms are blocked until ω carries signal in *this* domain.

Objective (no VLM labels this round):

* **JEPA**: a predictor maps ω_t to ω_{t+k} of an **EMA target encoder** (stop-grad). This is the
  only term that asks ω to be predictive of the future, which is what makes it a workspace state
  rather than a frame embedding.
* **SIGReg** (`sigreg_loss.sigreg_epps_pulley`, Epps-Pulley toward N(0,I)): the anti-collapse
  term. JEPA alone is minimised by a constant ω — exactly the failure mode G1 measured — so
  SIGReg is not optional decoration here, it is the thing that has to work. Its statistic is
  logged every eval so we can watch it hold the representation open.

Architecture and dims are identical to the canonical encoder (512-d ω, 4 layers, 8 heads,
backbone_dim/proprio_dim/lang_dim 2048, input_norm False), so every mechanism arm consumes the
result unchanged. Tap backbone is **pi05_libero** — ω must live in the space the arms train and
serve against. Convention: 2 views, agentview + eye-in-hand, 128 tokens.

Split is at the **episode** level: between-episode variance is only meaningful on unseen episodes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(0, keepdims=True)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(centered, rowvar=False)), 0, None)
    return float(eigenvalues.sum() ** 2 / max((eigenvalues**2).sum(), 1e-12))


class Predictor(nn.Module):
    """ω_t -> predicted ω_{t+k}. Deliberately small: the encoder must do the work, not this."""

    def __init__(self, dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * hidden_mult), nn.GELU(), nn.Linear(dim * hidden_mult, dim)
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.net(w)


class EpisodeCache:
    """Lazily-mmapped tap npz files, one per episode."""

    def __init__(self, paths: list[pathlib.Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(self.paths[index])
        return (
            data["tokens"].astype(np.float32),
            data["pooled_img"].astype(np.float32),
            data["pooled_lang"].astype(np.float32),
        )


def batch_from(cache: EpisodeCache, indices: np.ndarray, device) -> tuple[torch.Tensor, ...]:
    tokens, proprio, lang = zip(*(cache.get(int(i)) for i in indices))
    length = min(len(t) for t in tokens)
    stack = lambda xs: torch.from_numpy(np.stack([x[:length] for x in xs])).to(device)  # noqa: E731
    return stack(tokens), stack(proprio), stack(lang)


@torch.no_grad()
def evaluate(encoder, cache: EpisodeCache, indices: np.ndarray, device, step: int) -> dict:
    """The G1b metrics, computed exactly as the G1 table computed them."""
    from workspace_models.networks.sigreg_loss import sigreg_epps_pulley

    episodes = []
    for i in indices:
        tokens, proprio, lang = cache.get(int(i))
        omega = encoder(
            torch.from_numpy(tokens).unsqueeze(0).to(device),
            torch.from_numpy(proprio).unsqueeze(0).to(device),
            torch.from_numpy(lang).unsqueeze(0).to(device),
        )[0]
        episodes.append(omega.float().cpu().numpy())
    omega = np.concatenate(episodes)
    within = np.mean([e.var(0).mean() for e in episodes])
    between = np.stack([e.mean(0) for e in episodes]).var(0).mean()
    unit = omega / np.maximum(np.linalg.norm(omega, axis=1, keepdims=True), 1e-9)
    adjacent = np.concatenate(
        [
            (u[:-1] * u[1:]).sum(1)
            for u in [e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9) for e in episodes]
        ]
    )
    rng = np.random.default_rng(0)
    random_pairs = (unit[rng.integers(0, len(unit), 20000)] * unit[rng.integers(0, len(unit), 20000)]).sum(1)
    sigreg = float(sigreg_epps_pulley(torch.from_numpy(omega[:512]).to(device), step))
    return {
        "n_frames": int(len(omega)),
        "finite": bool(np.isfinite(omega).all()),
        "omega_rms": float(np.sqrt((omega**2).mean())),
        "effective_rank": effective_rank(omega),
        "between_episode_variance_fraction": float(between / max(between + within, 1e-12)),
        "cos_adjacent_mean": float(adjacent.mean()),
        "cos_random_mean": float(random_pairs.mean()),
        "temporal_coherence_gap": float(adjacent.mean() - random_pairs.mean()),
        "sigreg_stat": sigreg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tap", required=True)
    parser.add_argument(
        "--init-from", default=None, help="canonical encoder.pt to warm-start from (same dims); omit for scratch"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--predict-k", type=int, default=1, help="JEPA horizon in sampled-frame units")
    parser.add_argument("--lambda-sigreg", type=float, default=1.0)
    parser.add_argument("--ema", type=float, default=0.996)
    # --- attempt-2 additions (defaults off, so attempt-1 stays exactly reproducible) ---
    parser.add_argument(
        "--lambda-contrast",
        type=float,
        default=0.0,
        help="SupCon across episodes in the batch; targets between-episode variance, the metric attempt 1 failed",
    )
    parser.add_argument("--contrast-tau", type=float, default=0.1)
    parser.add_argument(
        "--sigreg-rank-cap",
        type=float,
        default=0.0,
        help="if >0, disable SIGReg on steps where batch effective rank already "
        "exceeds this (attempt 1 whitened to rank 84 vs in-domain 10.9)",
    )
    parser.add_argument("--heldout-frac", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from workspace_models.networks.sigreg_loss import sigreg_epps_pulley
    from workspace_models.networks.workspace_latent import WorkspaceEncoder

    device = torch.device("cuda")
    paths = sorted(pathlib.Path(args.tap).glob("episode_*.npz"))
    if not paths:
        raise SystemExit(f"no tap files under {args.tap}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    n_heldout = max(8, int(round(args.heldout_frac * len(paths))))
    heldout, train_idx = order[:n_heldout], order[n_heldout:]
    cache = EpisodeCache(paths)
    print(f"episodes: {len(paths)} ({len(train_idx)} train / {len(heldout)} heldout)", flush=True)

    cfg = SimpleNamespace(
        dim=512,
        n_layers=4,
        n_dec_layers=2,
        n_heads=8,
        k_slots=32,
        backbone_dim=2048,
        proprio_dim=2048,
        lang_dim=2048,
        c_horizon=1000,
        max_t=1200,
        mlp_ratio=4.0,
        input_norm=False,
    )
    encoder = WorkspaceEncoder(cfg).to(device)
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu")
        state = {k[len("encoder.") :]: v for k, v in blob["model"].items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"warm-start from {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    target = WorkspaceEncoder(cfg).to(device)
    target.load_state_dict(encoder.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = Predictor(cfg.dim).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()

    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        for group in opt.param_groups:
            group["lr"] = lr
        picked = rng.choice(train_idx, size=args.batch_episodes, replace=False)
        tokens, proprio, lang = batch_from(cache, picked, device)

        omega = encoder(tokens, proprio, lang)  # [B,T,512]
        with torch.no_grad():
            omega_target = target(tokens, proprio, lang)
        k = args.predict_k
        predicted = predictor(omega[:, :-k].reshape(-1, cfg.dim))
        wanted = omega_target[:, k:].reshape(-1, cfg.dim).detach()
        # Cosine + MSE: cosine supplies direction (the axis G1 found collapsed), MSE supplies scale.
        jepa = F.mse_loss(predicted, wanted) + (1 - F.cosine_similarity(predicted, wanted, dim=-1)).mean()
        flat = omega.reshape(-1, cfg.dim)
        sample = flat[torch.randperm(flat.shape[0], device=device)[:512]]
        sigreg = sigreg_epps_pulley(sample, step)

        # SIGReg rank cap. Attempt 1 whitened to effective rank 84 against an in-domain reference
        # of 10.9, clearing the eff-rank floor by overshooting it while the representation stayed
        # useless. Once the batch is already spread past the cap, SIGReg has done its job and is
        # only destroying structure, so stop paying it.
        lambda_sigreg = args.lambda_sigreg
        batch_rank = float("nan")
        if args.sigreg_rank_cap > 0:
            with torch.no_grad():
                centered = sample - sample.mean(0, keepdim=True)
                eigenvalues = torch.linalg.eigvalsh(
                    (centered.T @ centered / max(len(centered) - 1, 1)).float()
                ).clamp_min(0)
                batch_rank = float(eigenvalues.sum() ** 2 / eigenvalues.pow(2).sum().clamp_min(1e-12))
            if batch_rank > args.sigreg_rank_cap:
                lambda_sigreg = 0.0

        # Episode-discrimination (SupCon): positives are frames of the same episode, negatives are
        # every frame of every other episode in the batch. This is the only term that directly asks
        # ω to encode *which episode you are in* — the metric attempt 1 never moved off frozen levels.
        contrast = omega.sum() * 0.0
        if args.lambda_contrast > 0:
            z = F.normalize(flat, dim=-1)
            episode_of = torch.arange(omega.shape[0], device=device).repeat_interleave(omega.shape[1])
            logits = (z @ z.T) / args.contrast_tau
            self_mask = torch.eye(len(z), dtype=torch.bool, device=device)
            logits = logits.masked_fill(self_mask, float("-inf"))
            positive = (episode_of[:, None] == episode_of[None, :]) & ~self_mask
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            # The self entries are -inf; -inf * 0.0 is NaN, so zero them before the masked sum
            # rather than relying on the boolean mask to suppress them.
            log_prob = log_prob.masked_fill(self_mask, 0.0)
            contrast = -(log_prob * positive.float()).sum(1).div(positive.sum(1).clamp_min(1)).mean()

        loss = jepa + lambda_sigreg * sigreg + args.lambda_contrast * contrast

        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        else:
            print(f"[warn] non-finite loss at step {step} — skipped", flush=True)
        with torch.no_grad():
            for tp, ep in zip(target.parameters(), encoder.parameters()):
                tp.mul_(args.ema).add_(ep, alpha=1 - args.ema)

        if step % 50 == 0:
            print(
                f"step {step} loss {float(loss):.4f} jepa {float(jepa):.4f} "
                f"sigreg {float(sigreg):.4f} (lam {lambda_sigreg:.3f} rank {batch_rank:.1f}) "
                f"contrast {float(contrast):.4f} lr {lr:.2e}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            encoder.eval()
            metrics = evaluate(encoder, cache, heldout, device, step)
            encoder.train()
            metrics["step"] = step
            metrics["minutes"] = round((time.time() - started) / 60, 2)
            history.append(metrics)
            print(
                "[eval] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}),
                flush=True,
            )
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            payload = {
                "model": {f"encoder.{k}": v for k, v in encoder.state_dict().items()},
                "cfg": vars(cfg),
                "step": step,
                "feat_scale": 1.0,
                "heldout_episodes": [str(paths[i].name) for i in heldout],
                "train_args": vars(args),
                "eval": metrics,
            }
            torch.save(payload, out_dir / "encoder.pt")
            # Select on the PRIMARY discriminator (temporal coherence gap), not on the last step:
            # the canary showed the gap can peak and then decay as SIGReg keeps whitening.
            best = max(history, key=lambda h: h["temporal_coherence_gap"])
            if best["step"] == step:
                torch.save(payload, out_dir / "encoder_best.pt")
                print(f"[best] step {step} gap {metrics['temporal_coherence_gap']:.4f}", flush=True)

    print(f"done in {(time.time() - started) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
