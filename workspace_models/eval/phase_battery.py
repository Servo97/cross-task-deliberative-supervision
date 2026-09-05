"""phase_battery — stress battery for the demo-conditioning phase estimator (doc 15 gate G6).

At eval a rollout must be mapped to a grid position ("phase" tau) inside a reference demo so the
±W demo-token window (doc 15 D8) can be sliced around the corresponding moment. Rollouts pace
differently than demos (slower, stalls, faster); this battery measures how badly simple estimators
mis-place the window under simulated pacing mismatch, BEFORE any cloud run. The estimator only
needs ±W accuracy (W=20 grid steps) for the window to contain the right content.

Setup: for each task in the eval registry, each held-out demo r is replayed as a simulated
"rollout" (through its precomputed w latents) under 5 pacing WARPS, against the FIRST other
registry demo d as the reference. Ground truth tau* is proportional in COMPLETED FRACTION of the
ORIGINAL r. Three estimators produce tau_hat at every warped step; we report |tau_hat - tau*|
stats and the W=20 window-miss rate.

Run from the repo root:  PYTHONPATH=. python workspace_models/eval/phase_battery.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wsm_settings import WSM_DATA_ROOT  # noqa: E402

FEAT_ROOT = WSM_DATA_ROOT / "wsm_policy_feats" / "groot_65k"
REGISTRY = WSM_DATA_ROOT / "wsm_demo_tokens" / "orig_65k_matched" / "registry_eval.json"
OUT_JSON = Path(
    "/tmp/claude-22992/-home-sarveshp-Research-TRI-wsmv2/a6f4dfc9-1e7d-4b78-af61-779a05b0e2cc"
    "/scratchpad/phase_battery.json"
)

W_WINDOW = 20  # ±W token window (grid steps): |err| > W = the window misses the moment
STALL_LEN = 30  # mid-stall warp: same frame repeated this many grid steps
CLAMP_PROG = 0.95  # clamp_frac counts tau_hat == M_d-1 while true_progress < this

WARPS = ("none", "slow1.5x", "slow2x", "midstall", "fast0.75x")
ESTIMATORS = ("prop-clamp", "prop-scaled", "nn-monotonic")


# ---------------------------------------------------------------------------------------------
# Warps: map r's original grid-frame sequence 0..F_r-1 to a warped replay (array of ORIGINAL
# grid indices, one per warped rollout step). true_progress = orig_idx / (F_r - 1).
# ---------------------------------------------------------------------------------------------
def warp_indices(f_r: int, warp: str) -> np.ndarray:
    orig = np.arange(f_r, dtype=np.int64)
    if warp == "none":
        return orig
    if warp == "slow1.5x":  # each frame appears 1-2x (avg 1.5x)
        n = int(round(f_r * 1.5))
        return np.minimum(np.floor(np.arange(n) / 1.5).astype(np.int64), f_r - 1)
    if warp == "slow2x":  # each frame repeated 2x
        return np.repeat(orig, 2)
    if warp == "midstall":  # normal -> 30-step stall at 40% -> normal
        split = int(round(0.4 * (f_r - 1)))
        return np.concatenate([orig[: split + 1], np.full(STALL_LEN, split, dtype=np.int64), orig[split + 1 :]])
    if warp == "fast0.75x":  # skip every 4th frame (keep the final frame)
        keep = orig[(orig % 4) != 3]
        if keep[-1] != f_r - 1:
            keep = np.concatenate([keep, [f_r - 1]])
        return keep
    raise ValueError(f"unknown warp {warp!r}")


# ---------------------------------------------------------------------------------------------
# Estimators: tau_hat [T] at every warped step. m_d = reference demo grid length.
# ---------------------------------------------------------------------------------------------
def est_prop_clamp(t_steps: np.ndarray, m_d: int) -> np.ndarray:
    """Assume 1:1 pacing with the reference demo: tau = elapsed grid steps, clamped."""
    return np.minimum(t_steps, m_d - 1)


def est_prop_scaled(t_steps: np.ndarray, m_d: int, t_expected: int) -> np.ndarray:
    """Oracle-ish length prior: assume the rollout paces like the ORIGINAL demo (T_expected=F_r)."""
    return np.clip(np.round(t_steps / t_expected * (m_d - 1)).astype(np.int64), 0, m_d - 1)


def est_nn_monotonic(sim: np.ndarray, m_d: int, back: int = 2, fwd: int = 3) -> np.ndarray:
    """Nearest-neighbor in w-space with limited retract: tau_t = argmax over
    [tau_{t-1}-back, tau_{t-1}+fwd] of cosine(w_rollout_t, w_d_tau); tau_0 = 0.
    `sim` is the precomputed [T, M_d] cosine matrix (rollout step x demo grid)."""
    t_len = sim.shape[0]
    tau = np.zeros(t_len, dtype=np.int64)
    for t in range(1, t_len):
        lo = max(0, int(tau[t - 1]) - back)
        hi = min(m_d - 1, int(tau[t - 1]) + fwd)
        tau[t] = lo + int(np.argmax(sim[t, lo : hi + 1]))
    return tau


# ---------------------------------------------------------------------------------------------
def load_w(task: str, ep: int) -> np.ndarray:
    with np.load(FEAT_ROOT / task / f"demo_{ep:06d}" / "w.npz") as z:
        return z["w"].astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def metrics(tau_hat: np.ndarray, tau_star: np.ndarray, progress: np.ndarray, m_d: int) -> dict:
    err = np.abs(tau_hat - tau_star)
    return {
        "mean_err": float(err.mean()),
        "p90_err": float(np.percentile(err, 90)),
        "miss_rate": float((err > W_WINDOW).mean()),
        "clamp_frac": float(((tau_hat == m_d - 1) & (progress < CLAMP_PROG)).mean()),
    }


def main() -> None:
    t0 = time.time()
    registry: dict[str, list[int]] = json.loads(REGISTRY.read_text())
    per_task: dict[str, dict] = {}
    # agg[warp][est][metric] -> list of per-task means
    metric_keys = ("mean_err", "p90_err", "miss_rate", "clamp_frac")

    for task, eps in sorted(registry.items()):
        w_cache = {ep: load_w(task, ep) for ep in eps}
        # task-level accumulator: (warp, est) -> list of per-pair metric dicts
        acc: dict[tuple[str, str], list[dict]] = {}
        for r_ep in eps:
            d_ep = next(e for e in eps if e != r_ep)  # FIRST other registry demo
            w_r, w_d = w_cache[r_ep], w_cache[d_ep]
            f_r, m_d = len(w_r), len(w_d)
            w_d_unit = _unit(w_d)
            for warp in WARPS:
                oidx = warp_indices(f_r, warp)  # [T] original r indices
                progress = oidx / max(f_r - 1, 1)  # true completed fraction
                tau_star = np.round(progress * (m_d - 1)).astype(np.int64)
                t_steps = np.arange(len(oidx), dtype=np.int64)
                sim = _unit(w_r[oidx]) @ w_d_unit.T  # [T, M_d] cosine, vectorized
                hats = {
                    "prop-clamp": est_prop_clamp(t_steps, m_d),
                    "prop-scaled": est_prop_scaled(t_steps, m_d, f_r),
                    "nn-monotonic": est_nn_monotonic(sim, m_d),
                }
                for est, tau_hat in hats.items():
                    acc.setdefault((warp, est), []).append(metrics(tau_hat, tau_star, progress, m_d))
        per_task[task] = {
            warp: {
                est: {k: float(np.mean([p[k] for p in acc[(warp, est)]])) for k in metric_keys} for est in ESTIMATORS
            }
            for warp in WARPS
        }

    # aggregate over tasks (mean of per-task means, each task mean already over its pairs)
    summary = {
        warp: {
            est: {k: float(np.mean([per_task[t][warp][est][k] for t in per_task])) for k in metric_keys}
            for est in ESTIMATORS
        }
        for warp in WARPS
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "per_task": per_task}, indent=1))

    n_tasks = len(per_task)
    print(
        f"phase_battery: {n_tasks} tasks x 5 pairs x {len(WARPS)} warps x {len(ESTIMATORS)} "
        f"estimators  (W={W_WINDOW}, {time.time() - t0:.1f}s)"
    )
    hdr = f"{'warp':<11} {'estimator':<13} {'mean_err':>8} {'p90_err':>8} {'miss@20':>8} {'clamp':>7}"
    print(hdr)
    print("-" * len(hdr))
    for warp in WARPS:
        for est in ESTIMATORS:
            s = summary[warp][est]
            print(
                f"{warp:<11} {est:<13} {s['mean_err']:>8.2f} {s['p90_err']:>8.2f} "
                f"{s['miss_rate']:>8.3f} {s['clamp_frac']:>7.3f}"
            )
        print("-" * len(hdr))
    print(f"full per-task results -> {OUT_JSON}")


if __name__ == "__main__":
    main()
