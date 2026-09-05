"""Unit suite for the WSMv2 demo-conditioning modules (doc 15) — every architectural invariant the design
depends on, CPU-runnable. Run: <torch-env>/bin/python tests/test_wsm_demo_cfg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workspace_models.features.wsm_align import (  # noqa: E402
    demo_window_at,
    next_k_at,
    proportional_tau,
)
from workspace_models.networks.demo_encoder import DemoEncoder  # noqa: E402
from workspace_models.networks.demo_fusion import HistoryDemoFusion  # noqa: E402

torch.manual_seed(0)
B, M, K, W, D, LD = 3, 30, 16, 20, 64, 96  # small dims for speed; window 2W+1 = 41 > M exercises masking


def _enc(depth=2):
    return DemoEncoder(dim=D, lang_dim=LD, n_heads=4, depth=depth, max_rel=8)


def _fus(**kw):
    return HistoryDemoFusion(dim=D, lang_dim=LD, n_heads=4, depth=2, k_hist=K, window=W, **kw)


def _fus_inputs(fus, seed=1):
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(B, K, D, generator=g)
    demo_win = torch.randn(B, 2 * W + 1, D, generator=g)
    off = torch.arange(-W, W + 1)[None].expand(B, -1).clone()
    mask = torch.ones(B, 2 * W + 1, dtype=torch.bool)
    mask[:, -3:] = False  # a few invalid window slots (demo edge)
    lang = torch.randn(B, LD, generator=g)
    return hist, demo_win, off, mask, lang


def _randomize_gates(m):
    """Open the zero-init gates so information actually flows (post-init behavior probes)."""
    for name, p in m.named_parameters():
        if "ada" in name or name.endswith(("rel_emb", "pnf_emb", "hist_time_emb", "branch_emb")):
            torch.nn.init.normal_(p, std=0.05)


def test_demo_encoder_bidirectional():
    enc = _enc()
    _randomize_gates(enc)
    tok = torch.randn(B, M, D)
    lang = torch.randn(B, LD)
    base = enc(tok, lang)
    tok2 = tok.clone()
    tok2[:, -1] += torch.randn(D)  # perturb the LAST input token (non-uniform: in_norm kills constant shifts)
    out2 = enc(tok2, lang)
    assert (out2[:, 0] - base[:, 0]).abs().max() > 1e-6, "late token must influence early output (bidirectional)"
    tok3 = tok.clone()
    tok3[:, 0] += torch.randn(D)  # and the reverse
    out3 = enc(tok3, lang)
    assert (out3[:, -1] - base[:, -1]).abs().max() > 1e-6
    print("ok bidirectional demo encoder")


def test_demo_encoder_relative_bias_toeplitz_and_no_abs_pe():
    enc = _enc()
    bias = enc._bias(M, torch.device("cpu"))
    i, j = 5, 9
    assert torch.allclose(bias[i, j], bias[i + 7, j + 7]), "bias must depend on distance only (Toeplitz)"
    names = [n for n, _ in enc.named_parameters()]
    assert not any("time_emb" in n or "pos_emb" in n for n in names), "no absolute PE params allowed (D4)"
    print("ok relative-only positional structure (no absolute PE)")


def test_demo_encoder_pad_isolation():
    enc = _enc()
    _randomize_gates(enc)
    tok = torch.randn(B, M, D)
    lang = torch.randn(B, LD)
    pad = torch.zeros(B, M, dtype=torch.bool)
    pad[:, -5:] = True
    base = enc(tok, lang, pad_mask=pad)
    tok2 = tok.clone()
    tok2[:, -1] += 10 * torch.randn(D)  # perturb a PADDED token
    out2 = enc(tok2, lang, pad_mask=pad)
    assert torch.allclose(out2[:, :-5], base[:, :-5], atol=1e-5), "padded tokens must not leak into valid outputs"
    print("ok pad isolation")


def test_fusion_zero_init_identity():
    fus = _fus()
    hist, dw, off, mask, lang = _fus_inputs(fus)
    out = fus(hist, dw, off, mask, lang)
    expect = fus.out_norm(hist[:, -1])
    assert torch.allclose(out["z"], expect, atol=1e-6), "step-0 z must equal out_norm(hist[-1]) exactly (D6)"
    out_null = fus(hist, dw, off, mask, lang, drop_demo=torch.ones(B, dtype=torch.bool))
    assert torch.allclose(out_null["z"], expect, atol=1e-6)
    print("ok zero-init identity (with and without demo)")


def test_fusion_history_causality():
    fus = _fus()
    _randomize_gates(fus)
    hist, dw, off, mask, lang = _fus_inputs(fus)
    base = fus(hist, dw, off, mask, lang)["h_all"]
    j = K - 2
    hist2 = hist.clone()
    hist2[:, j] += 1.0  # perturb a LATE history token
    out2 = fus(hist2, dw, off, mask, lang)["h_all"]
    assert torch.allclose(out2[:, :j], base[:, :j], atol=1e-5), "history self-attn must be strictly causal"
    assert (out2[:, j:] - base[:, j:]).abs().max() > 1e-6
    print("ok strict history causality")


def test_fusion_null_demo_sees_no_demo_values():
    fus = _fus()
    _randomize_gates(fus)
    hist, dw, off, mask, lang = _fus_inputs(fus)
    drop = torch.ones(B, dtype=torch.bool)
    a = fus(hist, dw, off, mask, lang, drop_demo=drop)["z"]
    b = fus(hist, dw + 5.0, off, mask, lang, drop_demo=drop)["z"]  # totally different demo content
    assert torch.allclose(a, b, atol=1e-6), "drop_demo must be blind to demo values (C1b control soundness)"
    c = fus(hist, dw, off, mask, lang)["z"]
    assert (a - c).abs().max() > 1e-6, "with open gates, demo-on vs demo-null must differ"
    print("ok null-window isolation")


def test_fusion_window_mask_isolation():
    fus = _fus()
    _randomize_gates(fus)
    hist, dw, off, mask, lang = _fus_inputs(fus)
    base = fus(hist, dw, off, mask, lang)["z"]
    dw2 = dw.clone()
    dw2[:, -1] += 10.0  # perturb an INVALID (masked) window slot
    out2 = fus(hist, dw2, off, mask, lang)["z"]
    assert torch.allclose(base, out2, atol=1e-5), "masked window slots must not affect z"
    print("ok window pad-mask isolation")


def test_phase_head_no_grad_boundary():
    fus = _fus()
    _randomize_gates(fus)
    hist, dw, off, mask, lang = _fus_inputs(fus)
    hist = hist.requires_grad_(True)
    loss = ((fus.predict_phase(hist) - 0.5) ** 2).sum()
    loss.backward()
    for n, p in fus.named_parameters():
        if not n.startswith("phase_head"):
            assert p.grad is None or p.grad.abs().max() == 0, f"phase loss leaked into {n} (F1 boundary)"
    assert hist.grad is None or hist.grad.abs().max() == 0, "phase head must not backprop into history"
    print("ok phase-head no-grad boundary")


def test_align_helpers():
    idx, off, mask = demo_window_at(m=10, tau=0, window=3)
    assert list(off) == [-3, -2, -1, 0, 1, 2, 3] and mask.tolist() == [False] * 3 + [True] * 4
    assert idx.min() >= 0 and idx.max() <= 9
    idx, _, mask = demo_window_at(m=10, tau=9, window=3)
    assert mask.tolist() == [True] * 4 + [False] * 3 and idx.max() == 9
    w = np.arange(20, dtype=np.float32)[:, None].repeat(4, 1)  # F=20 grid frames
    fi = np.arange(20) * 8  # stride-8 grid
    tgt, valid = next_k_at(w, fi, t=40, ks=(1, 2, 4, 8))  # base grid idx = 5
    assert tgt[0, 0] == 6 and tgt[2, 0] == 9 and valid.all()
    tgt, valid = next_k_at(w, fi, t=150, ks=(1, 2, 4, 8))  # base = 18; k=2.. run off the end
    assert valid.tolist() == [True, False, False, False]
    assert proportional_tau(0, 100, 50) == 0 and proportional_tau(99, 100, 50) == 49
    rng = np.random.default_rng(0)
    taus = {proportional_tau(50, 100, 50, jitter=5, rng=rng) for _ in range(50)}
    assert len(taus) > 1 and all(0 <= x <= 49 for x in taus)
    print("ok align helpers (window edges, next_k validity, proportional tau)")


def test_grads_flow_to_everything_trainable():
    # Invariant 1 (wake-up guarantee): at EXACT zero-init the demo pathway contributes 0 to z, so the
    # encoder legitimately gets no grad THROUGH the fusion — but the AdaLN gates must get a LIVE grad
    # (single-multiplicative, unlike the dead doubly-multiplicative TokenModulator trap), so the pathway
    # opens at step 1.
    fus = _fus()
    enc = _enc()
    hist, dw, off, mask, lang = _fus_inputs(fus)
    d = enc(torch.randn(B, 2 * W + 1, D), lang)
    out = fus(hist, d, off, mask, lang)
    preds = fus.predict_future(out["z"])
    loss = sum((1 - torch.cosine_similarity(p, torch.randn_like(p))).mean() for p in preds.values())
    loss.backward()
    gate_grads = [p.grad for n, p in fus.named_parameters() if ".ada." in n and p.grad is not None]
    assert gate_grads and any(g.abs().max() > 0 for g in gate_grads), (
        "AdaLN gate params must receive a live gradient at zero-init (wake-up guarantee)"
    )
    # Invariant 2: once gates are open, gradients reach the encoder end-to-end.
    fus2 = _fus()
    enc2 = _enc()
    _randomize_gates(fus2)
    d2 = enc2(torch.randn(B, 2 * W + 1, D), lang)
    out2 = fus2(hist, d2, off, mask, lang)
    preds2 = fus2.predict_future(out2["z"])
    loss2 = sum((1 - torch.cosine_similarity(p, torch.randn_like(p))).mean() for p in preds2.values())
    loss2.backward()
    got = any(p.grad is not None and p.grad.abs().max() > 0 for n, p in enc2.named_parameters() if "null_" not in n)
    assert got, "no gradient reached the encoder with open gates (dead pathway)"
    print("ok gate wake-up grad at init + end-to-end encoder grads with open gates")


if __name__ == "__main__":
    test_demo_encoder_bidirectional()
    test_demo_encoder_relative_bias_toeplitz_and_no_abs_pe()
    test_demo_encoder_pad_isolation()
    test_fusion_zero_init_identity()
    test_fusion_history_causality()
    test_fusion_null_demo_sees_no_demo_values()
    test_fusion_window_mask_isolation()
    test_phase_head_no_grad_boundary()
    test_align_helpers()
    test_grads_flow_to_everything_trainable()
    print("\nALL WSMv2 UNIT TESTS PASS")
