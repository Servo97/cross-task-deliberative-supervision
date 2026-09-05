"""H14 P0 (amendment A3) — cross-domain TOKEN-STATISTICS audit across the frozen taps.

A3: "the domain bridge is a measured decision, not an assumption." The three domains are tapped from
DIFFERENT frozen networks (pi0.5 backbone tokens for RoboCasa/rmb, a frozen SigLIP for RoboMME,
pi05-libero for RoboCerebra). If their RMS / per-dim scale / subspace geometry are far apart, feeding
them to one shared trunk is a NaN hazard first and a silent-domination hazard second; the design
default is then per-domain input adapters (LayerNorm + affine), and if the stats are irreconcilable
RoboMME drops out of the JOINT ENCODER while staying in the deliberation corpus.

Statistics, reusing `scripts/robocerebra/g1_encoder_sanity.py`'s definitions verbatim so the numbers
are comparable to every G1/G1b bar we have already registered:

  finite fraction · RMS · per-dim std (mean, and dead-dim fraction) · effective rank (participation
  ratio) · linear CKA between domains on a matched token sample

CKA needs a shared dimensionality. Taps that disagree on width are compared only on the
width-agnostic statistics, and that is REPORTED rather than papered over with a projection.

  python scripts/deliberation/tap_stats_audit.py \
      --tap robocerebra=~/Research/TRI/wsm_data/robocerebra/omega_tap_full:tokens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def effective_rank(x: np.ndarray) -> float:
    """Participation ratio (sum L)^2 / sum L^2 of the covariance spectrum: 1 = collapsed."""
    centered = x - x.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / max((ev**2).sum(), 1e-12))


def bootstrap(values: np.ndarray, statistic, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    point = statistic(values)
    draws = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [N, D] samples with the SAME N (paired rows)."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    num = np.linalg.norm(x.T @ y, ord="fro") ** 2
    den = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(num / max(den, 1e-12))


def load_tap(root: Path, key: str, max_files: int, max_rows: int, seed: int, stratify: bool = False) -> np.ndarray:
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise SystemExit(f"no npz under {root}")
    # A per-EPISODE store sorts by task, so the plain head is 24 episodes of whichever task sorts
    # first — a within-task sample masquerading as a domain sample. `stratify` takes an even sweep
    # across the whole sorted list instead. Off by default so earlier audits reproduce byte-exactly.
    files = (
        list(np.asarray(files)[np.linspace(0, len(files) - 1, min(max_files, len(files))).astype(int)])
        if stratify
        else files[:max_files]
    )
    rng = np.random.default_rng(seed)
    rows = []
    for p in files:
        d = np.load(p)
        if key not in d.files:
            raise SystemExit(f"{p} has no '{key}' (has {d.files})")
        a = np.asarray(d[key], dtype=np.float32)
        a = a.reshape(-1, a.shape[-1])  # [.., D] -> [N, D]
        take = min(len(a), max(1, max_rows // len(files)))
        rows.append(a[rng.choice(len(a), take, replace=False)])
    return np.concatenate(rows, 0)


def stats_for(x: np.ndarray) -> dict:
    finite = np.isfinite(x)
    xf = np.where(finite, x, 0.0)
    per_dim_std = xf.std(0)
    er, er_lo, er_hi = bootstrap(xf[: min(len(xf), 4000)], effective_rank)
    return {
        "n_rows": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "finite_frac": round(float(finite.mean()), 6),
        "rms": round(float(np.sqrt((xf**2).mean())), 5),
        "abs_max": round(float(np.abs(xf).max()), 4),
        "mean": round(float(xf.mean()), 5),
        "per_dim_std_mean": round(float(per_dim_std.mean()), 5),
        "per_dim_std_p95_over_p05": round(
            float(np.percentile(per_dim_std, 95) / max(np.percentile(per_dim_std, 5), 1e-9)), 3
        ),
        "dead_dim_frac": round(float((per_dim_std < 1e-4).mean()), 5),
        "effective_rank": round(er, 3),
        "effective_rank_ci95": [round(er_lo, 3), round(er_hi, 3)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tap", action="append", default=[], help="name=path[:key]  (key defaults to 'tokens'); repeatable"
    )
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument(
        "--stratify-files",
        action="store_true",
        help="sample files evenly across the sorted store instead of taking the head "
        "(required for per-episode stores, which sort by task)",
    )
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/tap_stats_audit.json")
    args = ap.parse_args()
    if not args.tap:
        raise SystemExit("pass at least one --tap name=path[:key]")

    samples: dict[str, np.ndarray] = {}
    report: dict = {"taps": {}, "missing": []}
    for spec in args.tap:
        name, _, rest = spec.partition("=")
        path, _, key = rest.partition(":")
        key = key or "tokens"
        root = Path(path).expanduser()
        if not root.exists():
            report["missing"].append({"tap": name, "path": str(root), "reason": "not present locally"})
            continue
        x = load_tap(root, key, args.max_files, args.max_rows, args.seed, stratify=args.stratify_files)
        samples[name] = x
        report["taps"][name] = {"root": str(root), "key": key, **stats_for(x)}

    names = sorted(samples)
    report["pairwise"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            xa, xb = samples[a], samples[b]
            entry: dict = {
                "dim_a": int(xa.shape[1]),
                "dim_b": int(xb.shape[1]),
                "rms_ratio": round(float(np.sqrt((xa**2).mean()) / max(np.sqrt((xb**2).mean()), 1e-12)), 4),
            }
            if xa.shape[1] == xb.shape[1]:
                n = min(len(xa), len(xb), 4000)
                entry["linear_cka"] = round(linear_cka(xa[:n], xb[:n]), 4)
            else:
                entry["linear_cka"] = None
                entry["note"] = (
                    "widths differ; CKA needs a shared dimensionality. Reported as "
                    "null rather than projected -- a projection would be a design "
                    "choice smuggled into a measurement."
                )
            report["pairwise"][f"{a}|{b}"] = entry

    report["reading"] = {
        "rms_spread": (
            round(
                max(t["rms"] for t in report["taps"].values())
                / max(min(t["rms"] for t in report["taps"].values()), 1e-12),
                3,
            )
            if report["taps"]
            else None
        ),
        "adapter_needed_if": "rms_spread >> 1 or per_dim_std_p95_over_p05 differs by orders; "
        "A3's design default is per-domain LayerNorm+affine into the shared trunk",
        "complete": len(report["taps"]) >= 3,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    if not report["reading"]["complete"]:
        print(
            "\nINCOMPLETE: fewer than 3 taps available locally. A3's cross-domain decision "
            "cannot be made from this run; missing taps are listed under 'missing'."
        )


if __name__ == "__main__":
    main()
