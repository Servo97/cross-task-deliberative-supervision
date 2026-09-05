#!/usr/bin/env python3
"""Shared ω-window construction for RoboCerebra: one definition for train and serve.

Why this file exists: the single most expensive bug available to this campaign is a train/serve
ω mismatch (cf. the GR00T eval that served a different encoder than it trained on and read 0%).
So the *window semantics* live here, in one place, and both the training transform and the
policy-server wrapper import them. If this file is wrong, it is wrong identically on both sides
and shows up as a bad number rather than as a silent skew.

Feature-store layout (verified on disk, 994 episodes / 63,616 vectors):

    <root>/episode_%06d/w.npz
        w              [N, 512] float32   -- ω at the sampled frames
        frame_indices  [N]      int64     -- the frames those ω belong to, ascending
        lang_global    [N, 2048] float16
        subtask_index  [N]      int64

**ω is sparse in time.** The tap sampled ``N = 64`` uniformly-spaced frames per episode, so for a
~910-frame episode the ω grid has stride ≈14-16 frames (≈0.7-0.8 s at 20 fps). It is not, and
cannot cheaply be, per-frame: a dense tap over all 907,875 frames would cost ~14x the 2.5 h the
sparse tap took. The window is therefore defined on the **ω grid**, not on env steps:

    window(t, w) = the w most recent ω vectors whose frame_index <= t,
                   left-padded by repeating the oldest available vector.

Consequences, stated because they are load-bearing for the eval protocol:

* A window of w=8 spans ~8 x 15 = 120 env steps of history, i.e. slightly under one 150-step
  RoboCerebra subtask. w=16 spans ~240 steps, i.e. one subtask plus the one before it. That is a
  sensible scale for a "workspace" summary and is why the robocasa-tuned w=8/w=16 pair transfers
  here as a *comparable* contrast rather than an arbitrary one.
* Serve side must sample ω on the same cadence. ``grid_stride_for`` returns the stride an episode
  was tapped at, and ``ServeWindow`` re-encodes every ``stride`` env steps so the served window
  has the same temporal footprint as the trained one.
* Left-padding by repetition (not by zeros) keeps the window's statistics in-distribution at
  episode start; a zero vector is far outside the ω distribution (‖ω‖ ≈ 1 per dim RMS) and would
  be a distinct, unintended token.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np

OMEGA_DIM = 512
DEFAULT_FRAMES_PER_EPISODE = 64


class OmegaStore:
    """Read-only view over the precomputed per-episode ω store, with an LRU cache.

    Cheap enough to instantiate per dataloader worker: each episode is a ~350 KB npz and the
    cache holds the decompressed arrays for the hot episodes only.
    """

    def __init__(self, root: str | pathlib.Path, cache_size: int = 256) -> None:
        self.root = pathlib.Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"omega store not found: {self.root}")
        self._load = functools.lru_cache(maxsize=cache_size)(self._load_uncached)

    def _load_uncached(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / f"episode_{episode:06d}" / "w.npz"
        if not path.is_file():
            raise FileNotFoundError(f"no omega for episode {episode}: {path}")
        blob = np.load(path)
        w = np.asarray(blob["w"], dtype=np.float32)
        idx = np.asarray(blob["frame_indices"], dtype=np.int64)
        if w.ndim != 2 or w.shape[0] != idx.shape[0]:
            raise ValueError(f"corrupt omega for episode {episode}: w{w.shape} idx{idx.shape}")
        if not np.all(np.diff(idx) > 0):
            raise ValueError(f"omega frame_indices not strictly ascending for episode {episode}")
        if not np.isfinite(w).all():
            raise ValueError(f"non-finite omega for episode {episode}")
        return w, idx

    def episodes(self) -> list[int]:
        return sorted(int(p.name.split("_")[1]) for p in self.root.glob("episode_*") if p.is_dir())

    def grid_stride_for(self, episode: int) -> float:
        _, idx = self._load(episode)
        return float(np.diff(idx).mean()) if len(idx) > 1 else float("inf")

    def window(self, episode: int, frame: int, w: int) -> np.ndarray:
        """[w, 512] float32 — the w most recent ω at or before ``frame``, oldest first.

        Left-padded by repeating the oldest available vector, so the array is always [w, 512].
        """
        if w <= 0:
            raise ValueError(f"window length must be positive, got {w}")
        vecs, idx = self._load(episode)
        # number of ω strictly available at this frame; searchsorted 'right' => idx <= frame
        n_avail = int(np.searchsorted(idx, frame, side="right"))
        if n_avail == 0:
            # frame precedes the first tapped frame (only possible if frame < idx[0] == 0 normally)
            n_avail = 1
        lo = max(0, n_avail - w)
        chunk = vecs[lo:n_avail]
        if len(chunk) < w:
            pad = np.repeat(chunk[:1], w - len(chunk), axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        return np.ascontiguousarray(chunk, dtype=np.float32)


class ServeWindow:
    """Serve-side ring buffer holding the last ``w`` ω vectors, refreshed every ``stride`` steps.

    Mirrors :meth:`OmegaStore.window` exactly: same length, same oldest-first order, same
    repeat-padding at episode start. ``reset`` must be called at every episode start *and* at every
    resume re-pin, because the sim state jumps and stale ω would describe a workspace the robot is
    no longer in.
    """

    def __init__(self, w: int, stride: int, dim: int = OMEGA_DIM) -> None:
        if w <= 0 or stride <= 0:
            raise ValueError(f"bad window config w={w} stride={stride}")
        self.w, self.stride, self.dim = w, stride, dim
        self._buf: list[np.ndarray] = []
        self._last_encoded_step: int | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._last_encoded_step = None

    def due(self, step: int) -> bool:
        """True when this env step should be encoded (first step, then every ``stride``)."""
        return self._last_encoded_step is None or (step - self._last_encoded_step) >= self.stride

    def push(self, omega: np.ndarray, step: int) -> None:
        omega = np.asarray(omega, dtype=np.float32).reshape(-1)
        if omega.shape[0] != self.dim:
            raise ValueError(f"omega dim {omega.shape[0]} != {self.dim}")
        if not np.isfinite(omega).all():
            raise ValueError("non-finite omega at serve time")
        self._buf.append(omega)
        if len(self._buf) > self.w:
            del self._buf[: len(self._buf) - self.w]
        self._last_encoded_step = step

    def get(self) -> np.ndarray:
        if not self._buf:
            raise RuntimeError("ServeWindow.get() before any push(); encode the first frame first")
        chunk = np.stack(self._buf)
        if len(chunk) < self.w:
            chunk = np.concatenate([np.repeat(chunk[:1], self.w - len(chunk), axis=0), chunk])
        return np.ascontiguousarray(chunk, dtype=np.float32)


def _self_test() -> None:
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from wsm_settings import WSM_DATA_ROOT

    root = str(WSM_DATA_ROOT / "robocerebra" / "omega_features")
    store = OmegaStore(root)
    eps = store.episodes()
    print(f"episodes: {len(eps)}  first={eps[0]} last={eps[-1]}")
    vecs, idx = store._load(eps[0])
    print(f"ep0: w{vecs.shape} idx[0..3]={idx[:4].tolist()} stride={store.grid_stride_for(eps[0]):.1f}")

    # window at a frame past the end of the grid -> the last w vectors
    last = int(idx[-1])
    win = store.window(eps[0], last + 500, 8)
    assert win.shape == (8, OMEGA_DIM), win.shape
    assert np.allclose(win, vecs[-8:]), "tail window must equal the last 8 ω"

    # window at frame 0 -> all-padding with the first vector
    win0 = store.window(eps[0], 0, 8)
    assert np.allclose(win0, np.repeat(vecs[:1], 8, axis=0)), "start window must repeat ω[0]"

    # a mid window is contiguous and ends at the most recent ω <= frame
    mid_frame = int(idx[20]) + 3
    win_m = store.window(eps[0], mid_frame, 4)
    assert np.allclose(win_m, vecs[17:21]), "mid window misaligned"

    # ServeWindow reproduces OmegaStore.window when fed the same vectors in order
    sw = ServeWindow(w=4, stride=int(round(store.grid_stride_for(eps[0]))))
    for k in range(21):
        sw.push(vecs[k], step=int(idx[k]))
    assert np.allclose(sw.get(), win_m), "serve window != train window"

    # padding parity at episode start
    sw2 = ServeWindow(w=8, stride=15)
    sw2.push(vecs[0], step=0)
    assert np.allclose(sw2.get(), win0), "serve padding != train padding"

    # cadence
    sw3 = ServeWindow(w=8, stride=15)
    assert sw3.due(0)
    sw3.push(vecs[0], step=0)
    assert not sw3.due(14) and sw3.due(15), "stride cadence wrong"

    strides = [store.grid_stride_for(e) for e in eps[:50]]
    print(f"grid stride over 50 eps: min={min(strides):.1f} mean={np.mean(strides):.1f} max={max(strides):.1f}")
    print("omega_window self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
