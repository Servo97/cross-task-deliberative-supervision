"""Correctness tests for the WSM->policy interface: TokenModulator zero-init identity + causal windowing.

Run:  PYTHONPATH=. <torch-python> -m pytest tests/test_wsm_policy.py -q
  or: PYTHONPATH=. <torch-python> tests/test_wsm_policy.py   (plain asserts, no pytest needed)
"""

from __future__ import annotations

import numpy as np
import torch

from workspace_models.features.wsm_align import causal_window_indices, window_at
from workspace_models.networks.token_modulator import TokenModulator


def test_zero_init_is_exact_identity():
    """At init (gen zero-init) the modulator must be a bitwise no-op for ANY K."""
    torch.manual_seed(0)
    for k in (1, 2, 4):
        mod = TokenModulator(w_dim=512, lang_dim=2048, token_dim=2048, k_window=k).eval()
        tokens = torch.randn(3, 192, 2048)  # [B, P, token_dim]
        w_win = torch.randn(3, k, 512)
        lang = torch.randn(3, 2048)
        out = mod(tokens, w_win, lang)
        assert torch.equal(out, tokens), f"K={k}: zero-init modulator changed the tokens"


def test_nonzero_after_perturbation():
    """Once the generator is non-zero, modulation actually moves the tokens (it can learn)."""
    mod = TokenModulator(k_window=2).eval()
    with torch.no_grad():
        mod.gen[-1].weight.normal_(std=0.1)
        mod.gen[-1].bias.normal_(std=0.1)
    tokens = torch.randn(2, 192, 2048)
    out = mod(tokens, torch.randn(2, 2, 512), torch.randn(2, 2048))
    assert not torch.allclose(out, tokens), "perturbed modulator left tokens unchanged"
    assert out.shape == tokens.shape


def test_time_axis_broadcast():
    """Modulator must also work with a leading time axis [B,T,P,Db] / [B,T,K,Dw] / [B,T,lang]."""
    mod = TokenModulator(k_window=2).eval()
    tokens = torch.randn(2, 5, 192, 2048)
    out = mod(tokens, torch.randn(2, 5, 2, 512), torch.randn(2, 5, 2048))
    assert out.shape == tokens.shape and torch.equal(out, tokens)  # still zero-init identity


def test_causal_window_basic():
    fi = np.array([0, 8, 16, 24, 32], dtype=np.int64)
    # t on a later grid point: last 2 grid frames <= 20 are 8 and 16 -> indices 1,2
    assert causal_window_indices(fi, 20, 2).tolist() == [1, 2]
    # newest must never be a FUTURE frame
    assert fi[causal_window_indices(fi, 20, 2)][-1] <= 20
    # exact grid hit
    assert causal_window_indices(fi, 16, 2).tolist() == [1, 2]
    # t at end
    assert causal_window_indices(fi, 40, 2).tolist() == [3, 4]


def test_causal_window_left_pad():
    fi = np.array([0, 8, 16, 24, 32], dtype=np.int64)
    # before the first stride point: only frame 0 valid -> left-pad to K
    assert causal_window_indices(fi, 3, 2).tolist() == [0, 0]
    assert causal_window_indices(fi, 0, 4).tolist() == [0, 0, 0, 0]
    # K larger than available history
    assert causal_window_indices(fi, 8, 4).tolist() == [0, 0, 0, 1]


def test_masked_modulation_seam():
    """Mirror WSMConditionedActionHead.process_backbone_output: modulate-all then where(image_mask).
    Zero-init => identity everywhere; perturbed => changed at image positions, untouched elsewhere."""
    mod = TokenModulator(k_window=2).eval()
    B, S = 2, 200
    feats = torch.randn(B, S, 2048)
    img_mask = torch.zeros(B, S, dtype=torch.bool)
    img_mask[:, :192] = True  # first 192 = vision patch tokens
    w_win, lang = torch.randn(B, 2, 512), torch.randn(B, 2048)

    def seam(m):
        out = m(feats, w_win, lang)
        return torch.where(img_mask.unsqueeze(-1), out, feats)

    assert torch.equal(seam(mod), feats), "zero-init seam must be identity everywhere"
    with torch.no_grad():
        mod.gen[-1].weight.normal_(std=0.1)
        mod.gen[-1].bias.normal_(std=0.1)
    res = seam(mod)
    assert not torch.allclose(res[:, :192], feats[:, :192]), "image tokens must change once trained"
    assert torch.equal(res[:, 192:], feats[:, 192:]), "non-image tokens must be untouched"


def test_window_at_gathers_rows():
    fi = np.array([0, 8, 16, 24], dtype=np.int64)
    w = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)  # [F=4, Dw=3]
    win = window_at(w, fi, t=20, k=2)  # frames 8,16 -> rows 1,2
    assert win.shape == (2, 3)
    assert np.array_equal(win, w[[1, 2]])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
