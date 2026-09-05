"""wsm_align — map the WSM workspace latents (computed on the stride-8 cache grid) to a native policy
timestep, with a CAUSAL window. Shared by BOTH the training dataloader transform and the online eval
wrapper so the two produce w identically (train/eval consistency).

The WSM encoder emits w only at the cached frames (`frame_indices`, stride-8). To condition the policy
at an arbitrary native step t we take the last K grid frames with frame_indices <= t (never a future
frame), left-padded by repeating the earliest available when fewer than K exist. Newest = the most
recent grid frame at or before t. At eval the policy decides every 8 env steps (= the grid), so t lands
on a grid point and the newest w is exact; at train t is uniform, so the newest w may be up to stride-1
steps stale — a benign, consistent lag (eval is a subset of the training w-distribution).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def causal_window_indices(frame_indices: np.ndarray, t: int, k: int) -> np.ndarray:
    """Indices into the grid axis for the K most recent frames with frame_indices <= t.
    Returns [K] int64, ordered oldest..newest (newest = most recent grid frame <= t)."""
    valid = np.nonzero(np.asarray(frame_indices) <= int(t))[0]
    if len(valid) == 0:  # t before the first grid frame -> clamp to frame 0
        valid = np.array([0], dtype=np.int64)
    win = valid[-k:]
    if len(win) < k:  # left-pad by repeating the earliest available
        win = np.concatenate([np.full(k - len(win), win[0], dtype=np.int64), win])
    return win.astype(np.int64)


def load_w(feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a demo's precomputed workspace latents: (w [F,Dw] fp16, frame_indices [F] int64)."""
    d = np.load(Path(feature_dir) / "w.npz")
    return d["w"], d["frame_indices"].astype(np.int64)


def load_w_and_lang(feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(w [F,Dw], frame_indices [F], lang_global [lang_dim] | None) — lang for the policy modulator."""
    d = np.load(Path(feature_dir) / "w.npz")
    lang = d["lang_global"] if "lang_global" in d.files else None
    return d["w"], d["frame_indices"].astype(np.int64), lang


def window_at(w: np.ndarray, frame_indices: np.ndarray, t: int, k: int) -> np.ndarray:
    """Convenience: the [K, Dw] causal workspace-latent window conditioning native step t."""
    return w[causal_window_indices(frame_indices, t, k)]


# --------------------------------------------------------------------------------------------------
# FUTURE-targeting (model-free JEPA aux): the next grid-frame latent is a *training target*, not an
# injected input — so unlike the causal window above it MAY reference a frame after t. Only ever used
# offline at train time against the precomputed (frozen) w; never in the inference graph (doc 12).
# --------------------------------------------------------------------------------------------------
def next_index(frame_indices: np.ndarray, t: int) -> int:
    """Grid index of the FIRST frame strictly after native step t (clamped to the last frame).
    ~one control cycle ahead (grid stride 8) = the 't+1 workspace token' target."""
    fi = np.asarray(frame_indices)
    nxt = np.nonzero(fi > int(t))[0]
    return int(nxt[0]) if len(nxt) else int(len(fi) - 1)


def next_at(w: np.ndarray, frame_indices: np.ndarray, t: int) -> np.ndarray:
    """The [Dw] next-grid-frame workspace latent w_{t+1} (JEPA target for native step t)."""
    return w[next_index(frame_indices, t)]


def next_at_with_valid(w: np.ndarray, frame_indices: np.ndarray, t: int) -> tuple[np.ndarray, bool]:
    """Return (w_next, valid); valid is false when no strictly-future grid frame exists.

    next_at intentionally clamps for callers that need a total function. JEPA must use this variant
    so terminal samples are not trained to predict the final representation as if it were future.
    """
    fi = np.asarray(frame_indices)
    valid = bool(np.any(fi > int(t)))
    return w[next_index(fi, t)], valid


def future_window_at(w: np.ndarray, frame_indices: np.ndarray, t: int, h: int) -> np.ndarray:
    """[h, Dw] per-action-step future targets: for action token j (0..h-1) the grid latent nearest the
    native step t+1+j (clamped). Used by the optional per-step JEPA mode (doc 12 D4 v2)."""
    fi = np.asarray(frame_indices)
    out = np.empty((h, w.shape[-1]), dtype=w.dtype)
    for j in range(h):
        out[j] = w[next_index(fi, int(t) + j)]
    return out


# --------------------------------------------------------------------------------------------------
# WSMv2 demo-conditioning helpers (doc 15). GRID-index domain (demo tokens live on the stride-8 grid).
# --------------------------------------------------------------------------------------------------
def base_grid_index(frame_indices: np.ndarray, t: int) -> int:
    """Grid index of the newest frame <= native step t (the causal window's newest; clamped to 0)."""
    return int(causal_window_indices(frame_indices, t, 1)[-1])


def next_k_at(
    w: np.ndarray,
    frame_indices: np.ndarray,
    t: int,
    ks: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-horizon JEPA targets (doc 15 D10): for each k, the frozen latent k GRID frames ahead of the
    newest grid frame <= t. Returns (targets [len(ks), Dw], valid [len(ks)] bool). valid=False where the
    horizon runs off the demo end (index clamped to the last frame) — mask those loss terms rather than
    train on a repeated final target."""
    base = base_grid_index(frame_indices, t)
    last = len(np.asarray(frame_indices)) - 1
    idx = np.minimum(base + np.asarray(ks, dtype=np.int64), last)
    return w[idx], (base + np.asarray(ks)) <= last


def demo_window_at(m: int, tau: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ±window demo-token slice around grid index tau of a demo with m grid frames.
    Returns (idx [2W+1] int64 clamped to [0, m-1], off [2W+1] int64 = −W..+W, mask [2W+1] bool = the
    offset lands in-bounds). Clamped out-of-range positions are key-masked by the fusion, never attended
    — the mask (not edge-repeat content) is what the model sees at demo edges (doc 15 D8)."""
    off = np.arange(-window, window + 1, dtype=np.int64)
    raw = int(tau) + off
    mask = (raw >= 0) & (raw < int(m))
    return np.clip(raw, 0, int(m) - 1), off, mask


def proportional_tau(
    t: int,
    t1_len: int,
    m2: int,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
) -> int:
    """Train-time window center (doc 15 D8): proportional map of native step t in a demo of length t1_len
    onto a partner demo with m2 grid frames, plus optional uniform jitter (trains the cross-attention to
    LOCATE content within the window rather than trust the center)."""
    tau = int(round((min(int(t), t1_len - 1) / max(t1_len - 1, 1)) * (m2 - 1)))
    if jitter and rng is not None:
        tau += int(rng.integers(-jitter, jitter + 1))
    return int(np.clip(tau, 0, m2 - 1))
