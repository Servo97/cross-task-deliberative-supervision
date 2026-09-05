"""Correctness tests for the online-eval omega_t conditioner (vla_training/eval/_groot_wsm_eval.py).

The load-bearing property: at eval we encode a GROWING causal prefix each step and take newest omega_t;
because the WorkspaceEncoder is causal, newest omega_t MUST equal the token from encoding the FULL
sequence (what precompute/training used). If this holds, train/eval omega match by construction. CPU-only,
small frame count; uses the real WorkspaceModel so the encode signature/dims are exercised.

Run:  PYTHONPATH=. <torch-python> tests/test_wsm_eval.py
"""

from __future__ import annotations

import numpy as np
import torch

from vla_training.eval._groot_wsm_eval import WSMEvalConditioner
from workspace_models.features.wsm_align import causal_window_indices
from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig

P, DB, DP, DL, DW = 192, 2048, 1536, 2048, 512


def _encoder(seed=0):
    torch.manual_seed(seed)
    m = WorkspaceModel(WSMConfig()).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _synthetic(F, seed=1):
    rng = np.random.default_rng(seed)
    patches = rng.standard_normal((F, P, DB)).astype(np.float32)
    proprio = rng.standard_normal((F, DP)).astype(np.float32)
    lang = rng.standard_normal(DL).astype(np.float32)
    return patches, proprio, lang


def test_window_shape_and_growth():
    cond = WSMEvalConditioner(_encoder(), k_window=4, stride=8, device="cpu")
    patches, proprio, lang = _synthetic(6)
    cond.reset(lang)
    for t in range(6):
        w_win, lg = cond.step(patches[t], proprio[t])
        assert w_win.shape == (4, DW), f"t={t}: {w_win.shape}"  # K always 4 (left-padded early)
        assert lg.shape == (DL,)
    print("  PASS test_window_shape_and_growth")


def test_reset_clears_buffer():
    cond = WSMEvalConditioner(_encoder(), k_window=2, stride=8, device="cpu")
    patches, proprio, lang = _synthetic(4)
    cond.reset(lang)
    for t in range(3):
        cond.step(patches[t], proprio[t])
    assert len(cond._fused) == len(cond._conds) == 3
    assert not hasattr(cond, "_patches") and not hasattr(cond, "_proprio")
    cond.reset(lang)
    assert len(cond._fused) == len(cond._conds) == 0
    cond.step(patches[0], proprio[0])
    assert len(cond._fused) == len(cond._conds) == 1
    assert cond._fused[0].shape == cond._conds[0].shape == (DW,)
    print("  PASS test_reset_clears_buffer")


def test_step_before_reset_raises():
    cond = WSMEvalConditioner(_encoder(), k_window=2, device="cpu")
    p, pr, _ = _synthetic(1)
    try:
        cond.step(p[0], pr[0])
        assert False, "expected RuntimeError"
    except RuntimeError:
        print("  PASS test_step_before_reset_raises")


def test_streaming_matches_full_causal():
    """THE key property: streaming newest-omega_t == full-sequence omega_t (causal encoder)."""
    enc = _encoder(seed=2)
    F = 5
    patches, proprio, lang = _synthetic(F, seed=3)
    # full-sequence encode (what precompute does): w_full[F, DW]
    pt = torch.from_numpy(patches)[None]
    pr = torch.from_numpy(proprio)[None]
    cl = torch.from_numpy(lang).view(1, 1, -1).expand(1, F, -1)
    with torch.no_grad():
        w_full = enc.encode(pt, pr, cl)[0]  # [F, DW]
    # streaming: newest element of each step's window == w at current frame t
    cond = WSMEvalConditioner(enc, k_window=3, stride=8, device="cpu")
    cond.reset(lang)
    for t in range(F):
        w_win, _ = cond.step(patches[t], proprio[t])
        newest = w_win[-1]  # newest = omega_t
        assert torch.allclose(newest, w_full[t], atol=1e-4), (
            f"t={t}: streaming omega_t != full-sequence omega_t (max {(newest - w_full[t]).abs().max():.2e})"
        )
    print("  PASS test_streaming_matches_full_causal (train/eval omega consistency holds)")


def test_window_matches_wsm_align():
    """The eval window must equal the SHARED train helper window_at over the same grid."""
    enc = _encoder(seed=4)
    F = 6
    patches, proprio, lang = _synthetic(F, seed=5)
    pt = torch.from_numpy(patches)[None]
    pr = torch.from_numpy(proprio)[None]
    cl = torch.from_numpy(lang).view(1, 1, -1).expand(1, F, -1)
    with torch.no_grad():
        w_full = enc.encode(pt, pr, cl)[0].numpy()
    k, stride = 4, 8
    cond = WSMEvalConditioner(enc, k_window=k, stride=stride, device="cpu")
    cond.reset(lang)
    for t in range(F):
        w_win, _ = cond.step(patches[t], proprio[t])
        fi = np.arange(t + 1) * stride
        idx = causal_window_indices(fi, int(fi[-1]), k)
        assert np.allclose(w_win.numpy(), w_full[idx], atol=1e-4), f"t={t}: window != wsm_align window_at"
    print("  PASS test_window_matches_wsm_align")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all eval-conditioner tests passed")
