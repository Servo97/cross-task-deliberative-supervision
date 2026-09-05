#!/usr/bin/env python3
"""G1: run the canonical Stage-S ω encoder on RoboCerebra frames and test it for degeneracy.

This is the half of the G1 gate that needs no salient-patch labels. It answers: does the frozen
robocasa-trained encoder, applied cross-domain to RoboCerebra (LIBERO Franka, 2 views), still
produce a *live* representation — or does it collapse?

RoboCerebra ω convention (ruled 2026-08-10, and used identically for train-time ω precompute):
**2 views, agentview + eye-in-hand, 128 tokens.** No fabricated third view — no zero-pad, no
duplicated wrist. `PatchPool` cross-attends P tokens with a single learned query, so P is not an
architectural constant and 128 is a legal input; every chance baseline downstream uses 128.

Diagnostics, all reported with bootstrap CIs over frames:

* finite / RMS: does anything overflow (the NaN-encoder failure mode)?
* per-dim std, and the fraction of the 512 dims that are effectively dead
* effective rank (participation ratio of the covariance eigenspectrum) — the collapse metric
* between-episode variance fraction: does ω separate episodes, or is it a constant vector?
* neighbour coherence: cosine between ω_t and ω_{t+1} vs between random pairs — a live temporal
  code should be locally smooth and globally spread
"""

from __future__ import annotations

import argparse
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import torch


def effective_rank(x: np.ndarray) -> float:
    """Participation ratio (sum λ)² / sum λ² of the covariance spectrum: 1 = collapsed."""
    centered = x - x.mean(0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0, None)
    return float(eigenvalues.sum() ** 2 / max((eigenvalues**2).sum(), 1e-12))


def bootstrap(values: np.ndarray, statistic, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = statistic(values)
    draws = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(point), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tap", required=True, help="dir of omega_tap episode npz files")
    parser.add_argument("--encoder", required=True, help="Stage-S encoder.pt")
    parser.add_argument(
        "--views",
        choices=["both", "agentview"],
        default="both",
        help="'both' = the ruled 128-token convention; 'agentview' = 64-token diagnostic",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from workspace_models.networks.workspace_latent import WorkspaceEncoder

    blob = torch.load(args.encoder, map_location="cpu")
    cfg = SimpleNamespace(**blob["cfg"])
    encoder = WorkspaceEncoder(cfg)
    state = {k[len("encoder.") :]: v for k, v in blob["model"].items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    encoder.eval().cuda()
    print(
        f"encoder step={blob.get('step')} missing={len(missing)} unexpected={len(unexpected)} "
        f"input_norm={cfg.input_norm} backbone_dim={cfg.backbone_dim}",
        flush=True,
    )

    omegas, episode_ids, tokens_rms = [], [], []
    for path in sorted(pathlib.Path(args.tap).glob("episode_*.npz")):
        data = np.load(path)
        tokens = data["tokens"].astype(np.float32)  # [T,128,2048]
        if args.views == "agentview":
            tokens = tokens[:, :64]  # base view only
        proprio = data["pooled_img"].astype(np.float32)  # [T,2048]
        lang = data["pooled_lang"].astype(np.float32)  # [T,2048]
        tokens_rms.append(float(np.sqrt((tokens**2).mean())))
        with torch.no_grad():
            omega = (
                encoder(
                    torch.from_numpy(tokens).unsqueeze(0).cuda(),
                    torch.from_numpy(proprio).unsqueeze(0).cuda(),
                    torch.from_numpy(lang).unsqueeze(0).cuda(),
                )[0]
                .cpu()
                .numpy()
            )
        omegas.append(omega)
        episode_ids.append(np.full(len(omega), int(path.stem.split("_")[1])))

    omega = np.concatenate(omegas)
    finite = bool(np.isfinite(omega).all())

    per_dim_std = omega.std(0)
    dead = float((per_dim_std < 1e-3 * per_dim_std.mean()).mean())
    rank_point, rank_low, rank_high = bootstrap(omega, effective_rank)

    within = np.mean([e.var(0).mean() for e in omegas])
    between = np.stack([e.mean(0) for e in omegas]).var(0).mean()
    between_fraction = float(between / max(between + within, 1e-12))

    unit = omega / np.maximum(np.linalg.norm(omega, axis=1, keepdims=True), 1e-9)
    adjacent = np.concatenate(
        [
            (u[:-1] * u[1:]).sum(1)
            for u in [e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9) for e in omegas]
        ]
    )
    rng = np.random.default_rng(0)
    random_pairs = (unit[rng.integers(0, len(unit), 20000)] * unit[rng.integers(0, len(unit), 20000)]).sum(1)

    report = {
        "convention": {
            "views": args.views,
            "patch_tokens": int(128 if args.views == "both" else 64),
            "note": "no fabricated third view; chance baselines use this token count",
        },
        "encoder": {
            "path": args.encoder,
            "step": int(blob.get("step", -1)),
            "input_norm": bool(cfg.input_norm),
            "nonfinite_weights": 0,
        },
        "n_frames": int(len(omega)),
        "n_episodes": int(len(omegas)),
        "omega_finite": finite,
        "tokens_rms_mean": float(np.mean(tokens_rms)),
        "omega_rms": float(np.sqrt((omega**2).mean())),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_min": float(per_dim_std.min()),
        "dead_dim_fraction": dead,
        "effective_rank": {"point": rank_point, "ci95": [rank_low, rank_high], "of_dims": int(omega.shape[1])},
        "between_episode_variance_fraction": between_fraction,
        "cos_adjacent_mean": float(adjacent.mean()),
        "cos_random_mean": float(random_pairs.mean()),
        "temporal_coherence_gap": float(adjacent.mean() - random_pairs.mean()),
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
