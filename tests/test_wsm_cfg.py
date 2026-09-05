"""Smoke tests for the CFG workspace conditioner + seam (pure torch, no gr00t). doc 12.

Covers: zero-init => cond==0 at init (policy == baseline at step 0); shapes; per-component dropout
extremes; w_next=None uses its null; grads; the temb-add seam (TEWrapper behavior); and the CFG
velocity-combination formula v = v_u + s(v_c - v_u). Run: python tests/test_wsm_cfg.py
"""

import torch
import torch.nn as nn

from workspace_models.networks.wsm_cfg_cond import WSMCfgConditioner


def test_zero_init_is_baseline():
    c = WSMCfgConditioner(w_dim=512, cond_dim=512, p_drop=0.2)
    w_t = torch.randn(4, 512)
    cond = c(w_t, None, training=False, force_uncond=False)
    unc = c(w_t, None, training=False, force_uncond=True)
    assert cond.shape == (4, 512)
    assert torch.allclose(cond, torch.zeros_like(cond)), "zero-init must give cond==0 (baseline at step 0)"
    assert torch.allclose(cond, unc), "at init conditional==uncond (both 0)"
    print("[ok] zero-init => cond==0 (baseline at init), cond==uncond")


def _unzero(c):  # break the AdaLN-zero so the conditioner actually does something (simulate a trained head)
    with torch.no_grad():
        for proj in (c.proj_t, c.proj_next):
            proj[-1].weight.normal_(0, 0.1)
            proj[-1].bias.normal_(0, 0.1)
        c.null_t.normal_(0, 1)
        c.null_next.normal_(0, 1)
    return c


def test_dropout_extremes_and_null():
    c = _unzero(WSMCfgConditioner(512, 512, p_drop=0.2))
    w_t, w_next = torch.randn(8, 512), torch.randn(8, 512)
    # p_drop=1.0 in training -> everything dropped to null -> equals force_uncond
    c.p_drop = 1.0
    drop_all = c(w_t, w_next, training=True)
    unc = c(w_t, w_next, training=False, force_uncond=True)
    assert torch.allclose(drop_all, unc, atol=1e-5), "p_drop=1.0 (train) must equal forced-uncond"
    # p_drop=0.0 -> conditional keeps w_t; differs from uncond
    c.p_drop = 0.0
    keep = c(w_t, w_next, training=True)
    assert not torch.allclose(keep, unc, atol=1e-3), "p_drop=0 conditional must differ from uncond"
    # w_next=None uses its null (no crash, deterministic)
    only_t = c(w_t, None, training=False)
    only_t2 = c(w_t, None, training=False)
    assert torch.allclose(only_t, only_t2), "w_next=None path must be deterministic at eval"
    print("[ok] dropout extremes (p=1 == uncond, p=0 != uncond) + w_next=None null path")


def test_grads():
    c = _unzero(WSMCfgConditioner(512, 512, p_drop=0.2))
    w_t = torch.randn(4, 512, requires_grad=True)
    loss = c(w_t, None, training=False).pow(2).mean()
    loss.backward()
    assert c.proj_t[-1].weight.grad is not None and torch.isfinite(c.proj_t[-1].weight.grad).all()
    assert c.proj_next[-1].weight.grad is not None  # exercised via null_next
    print("[ok] grads flow to proj_t and proj_next")


def test_temb_seam():
    # mirrors TEWrapper: temb + cond when set, passthrough when None
    class FakeTE(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(1, 8)

        def forward(self, ts):
            return self.lin(ts.float().unsqueeze(-1))

    class TEWrapper(nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig
            self.cond = None

        def forward(self, ts):
            temb = self.orig(ts)
            return temb if self.cond is None else temb + self.cond.to(temb.dtype)

    te = TEWrapper(FakeTE())
    ts = torch.zeros(4, dtype=torch.long)
    base = te(ts).clone()
    te.cond = torch.ones(4, 8)
    assert torch.allclose(te(ts), base + 1.0), "temb must get + cond"
    te.cond = None
    assert torch.allclose(te(ts), base), "cond=None must pass through (exact baseline)"
    print("[ok] temb seam: temb += cond, passthrough when None")


def test_cfg_velocity_formula():
    v_c, v_u = torch.randn(4, 16, 7), torch.randn(4, 16, 7)
    for s, expect in [(0.0, v_u), (1.0, v_c)]:
        v = v_u + s * (v_c - v_u)
        assert torch.allclose(v, expect, atol=1e-6), f"s={s} guided velocity wrong"
    # s>1 amplifies beyond conditional
    v2 = v_u + 2.0 * (v_c - v_u)
    assert torch.allclose(v2, 2 * v_c - v_u, atol=1e-6)
    print("[ok] CFG velocity: s=0->uncond, s=1->cond, s=2->2c-u")


if __name__ == "__main__":
    test_zero_init_is_baseline()
    test_dropout_extremes_and_null()
    test_grads()
    test_temb_seam()
    test_cfg_velocity_formula()
    print("\nALL WSM-CFG SMOKE TESTS PASSED")
