#!/usr/bin/env python3
"""Shared primitives for the failure-mode study (videos + trajectory-difference metrics).

Design notes that matter for reading the numbers later:

* One rollout per (task, checkpoint, reset). The 20 resets per task are IDENTICAL across
  every checkpoint of that benchmark, so every contrast is paired.
* Rollout index 0 is used everywhere, so the pi diffusion-noise stream is the sealed one
  (``rollout_noise_base(seed, 0) == seed``); a ReMemBench held-out reset therefore replays
  bit-identically to its sealed eval cell.
* Trajectories are stored as raw MuJoCo state vectors plus a small set of derived
  quantities. Video is rendered in a SEPARATE pass by replaying those states into a
  render-configured environment, so rollout throughput is never paid for pixels and the
  video resolution/camera set is a free parameter.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

# --------------------------------------------------------------------------------------
# study constants
# --------------------------------------------------------------------------------------
RESETS_PER_TASK = 20

RMB_TASKS = ("MemFruitInSinkRightFar", "MemHeatPot", "MemWashAndReturnLeft")
RC_TASKS = ("KettleBoiling", "ScrubCuttingBoard", "SearingMeat")

#: ``build_remembench_episode_manifest.BASE_SEED`` — reused so top-up seeds are drawn from
#: the same stream as the sealed held-out ones.
RMB_BASE_SEED = 20260803
#: ``build_heldout_eval_manifest.BASE_SEED``.
RC_BASE_SEED = 20260723

#: Episode-index namespace for the training-demo top-ups. Sealed held-out episodes occupy
#: 0..n_heldout-1; starting the top-ups at 1000 makes a seed collision impossible and makes
#: the split visible in the id itself.
TRAIN_TOPUP_INDEX_BASE = 1000

#: First-divergence detector. EE world-position deviation from the expert, in metres, that
#: must persist for K consecutive control steps.
DIVERGENCE_THRESHOLD_M = 0.10
DIVERGENCE_CONSECUTIVE_K = 10

#: Target-commitment approach radius (metres) between the end-effector and a candidate
#: target's world position.
APPROACH_RADIUS_M = 0.15
#: A candidate counts as approached only if the EE stays inside the radius this long.
APPROACH_CONSECUTIVE_K = 5

VIDEO_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_center",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
VIDEO_H = 480
VIDEO_W = 640
VIDEO_FPS = 20
VIDEO_STRIDE = 2  # env runs at ~20 Hz control; keep every 2nd frame at 20 fps => 2x speed


# --------------------------------------------------------------------------------------
# seeds / ids
# --------------------------------------------------------------------------------------
def seed_for(base_seed: int, task: str, episode_index: int) -> int:
    """Vendored from ``build_remembench_episode_manifest._seed_for`` (identical bytes)."""
    raw = f"{int(base_seed)}\0{task}\0{int(episode_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def policy_noise_seed(episode_seed: int, env_step: int) -> int:
    """Vendored from ``run_remembench_eval.policy_noise_seed`` (identical bytes)."""
    if not 0 <= int(episode_seed) <= 0xFFFFFFFF:
        raise ValueError(f"episode_seed is outside uint32: {episode_seed}")
    raw = f"pi_diffusion_sha256_v1\0{int(episode_seed)}\0{int(env_step)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


NOISE_KIND = "remembench_rollout_v1"


def rollout_noise_base(episode_seed: int, rollout_idx: int) -> int:
    """Per-rollout uint32 noise-stream root. Vendored byte-for-byte from
    ``run_remembench_eval.rollout_noise_base`` so draw *k* here is draw *k* there.

    Rollout 0 reuses the episode seed unchanged, which is why the single-draw cells in this
    study replay the sealed protocol's rollout 0 exactly.
    """
    if rollout_idx == 0:
        return int(episode_seed)
    raw = f"{NOISE_KIND}\0{int(episode_seed)}\0{int(rollout_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def reset_id(split: str, episode_index: int) -> str:
    """Stable, human-sortable reset identifier, e.g. ``heldout_003`` / ``train_1007``."""
    return f"{split}_{int(episode_index):03d}"


def cell_name(reset_id_value: str, rollout_idx: int = 0) -> str:
    """On-disk name for one rollout of one reset.

    Draw 0 keeps the bare reset id so every artifact produced before multi-draw support was
    added stays valid and is never recomputed; extra draws get a ``__r<k>`` suffix.
    """
    return reset_id_value if int(rollout_idx) == 0 else f"{reset_id_value}__r{int(rollout_idx)}"


def write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: str) -> str:
    """Content digest of a checkpoint tree: sorted (relpath, size, sha256) triples."""
    entries = []
    for base, _dirs, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root)
            entries.append(f"{rel}\0{os.path.getsize(full)}\0{sha256_file(full)}")
    return hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def quat_to_mat(quat) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation matrix (MuJoCo convention)."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_geodesic(q_a, q_b) -> float:
    """Angle in radians between two unit quaternions, sign-invariant."""
    a = np.asarray(q_a, dtype=np.float64)
    b = np.asarray(q_b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


#: DTW is computed on curves decimated to at most this many points. The recurrence is
#: inherently sequential in Python, so full-resolution DTW over ~3200-step episodes costs
#: seconds per rollout and minutes per benchmark. An end-effector path sampled at 20 Hz is
#: massively oversampled for a *shape* distance -- decimating to <=400 points (>=5 Hz on the
#: longest episodes here) changes the value negligibly and makes the whole metrics pass
#: ~25x cheaper. Both curves are decimated by the same rule, so the comparison stays fair.
DTW_MAX_POINTS = 400


def _decimate(curve: np.ndarray, limit: int = DTW_MAX_POINTS) -> np.ndarray:
    if len(curve) <= limit:
        return curve
    return curve[:: int(np.ceil(len(curve) / limit))]


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 100) -> float:
    """Sakoe-Chiba-banded DTW over two (T, D) position curves; mean cost per aligned pair.

    Banded because an unbanded O(T^2) fill is both slow and meaningless once the warp
    exceeds a few hundred control steps. The returned value is the accumulated cost divided
    by the alignment path length, so it is comparable across episodes of different length
    (units: metres). Inputs are decimated first -- see :data:`DTW_MAX_POINTS`.
    """
    a = _decimate(np.asarray(a, dtype=np.float64))
    b = _decimate(np.asarray(b, dtype=np.float64))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    band = max(band, abs(n - m) + 1)
    inf = np.inf
    cost = np.full((m + 1,), inf)
    steps = np.full((m + 1,), 0.0)
    cost[0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        prev_cost, prev_steps = cost.copy(), steps.copy()
        cost[:] = inf
        steps[:] = 0.0
        if lo > 1:
            cost[lo - 1] = inf
        for j in range(lo, hi + 1):
            local = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            options = (
                (prev_cost[j], prev_steps[j]),  # insertion
                (cost[j - 1], steps[j - 1]),  # deletion
                (prev_cost[j - 1], prev_steps[j - 1]),  # match
            )
            best_cost, best_steps = min(options, key=lambda item: item[0])
            cost[j] = best_cost + local
            steps[j] = best_steps + 1.0
    if not np.isfinite(cost[m]) or steps[m] == 0:
        return float("nan")
    return float(cost[m] / steps[m])


def first_divergence_step(
    deviation: np.ndarray,
    threshold: float = DIVERGENCE_THRESHOLD_M,
    consecutive: int = DIVERGENCE_CONSECUTIVE_K,
) -> int:
    """First t whose deviation exceeds ``threshold`` for ``consecutive`` steps. -1 if never."""
    over = np.asarray(deviation, dtype=np.float64) > threshold
    if len(over) < consecutive:
        return -1
    window = np.convolve(over.astype(np.int32), np.ones(consecutive, dtype=np.int32), "valid")
    hits = np.flatnonzero(window == consecutive)
    return int(hits[0]) if len(hits) else -1


def sustained_entry_step(
    distance: np.ndarray,
    radius: float = APPROACH_RADIUS_M,
    consecutive: int = APPROACH_CONSECUTIVE_K,
) -> int:
    """First t where ``distance`` stays under ``radius`` for ``consecutive`` steps. -1 if never."""
    under = np.asarray(distance, dtype=np.float64) < radius
    if len(under) < consecutive:
        return -1
    window = np.convolve(under.astype(np.int32), np.ones(consecutive, dtype=np.int32), "valid")
    hits = np.flatnonzero(window == consecutive)
    return int(hits[0]) if len(hits) else -1
