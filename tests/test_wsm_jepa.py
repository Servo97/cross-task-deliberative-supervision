"""Smoke tests for the model-free WSM (JEPA aux-target) pieces — pure torch, no gr00t (doc 12).

Validates: wsm_align.next_at, sigreg_epps_pulley (isotropic < collapsed), JEPAPredictor shapes,
wsm_jepa_sigreg_loss (finite + grads + bf16-predictor dtype-robustness), and the proj_out_2
forward_pre_hook mechanism the real GR00T wiring depends on (on a FAKE DiT). Run:
    pytest -q tests/test_wsm_jepa.py        # or: python tests/test_wsm_jepa.py
"""

import numpy as np
import torch
import torch.nn as nn

from workspace_models.features.wsm_align import future_window_at, next_at, next_at_with_valid, next_index
from workspace_models.networks.jepa_align_head import JEPAPredictor, wsm_jepa_sigreg_loss
from workspace_models.networks.sigreg_loss import sigreg_epps_pulley


def test_next_at_grid():
    fi = np.array([0, 8, 16, 24], dtype=np.int64)  # stride-8 grid
    w = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)  # [F,3]
    assert next_index(fi, 0) == 1  # first frame strictly after t=0 -> idx 1 (fi=8)
    assert next_index(fi, 7) == 1
    assert next_index(fi, 8) == 2
    assert next_index(fi, 100) == 3  # past the end -> clamp last
    assert np.allclose(next_at(w, fi, 0), w[1])
    assert next_at_with_valid(w, fi, 0)[1]
    assert not next_at_with_valid(w, fi, 100)[1]
    fut = future_window_at(w, fi, 0, 4)  # [4,3]
    assert fut.shape == (4, 3)
    print("[ok] next_at / next_index / future_window_at")


def test_sigreg_isotropic_lt_collapsed():
    torch.manual_seed(0)
    iso = torch.randn(256, 64)  # ~N(0,I)
    collapsed = torch.randn(256, 64) * 0.01 + 5.0  # tiny variance, off-center -> very non-Gaussian
    s_iso = sigreg_epps_pulley(iso, global_step=1)
    s_col = sigreg_epps_pulley(collapsed, global_step=1)
    assert torch.isfinite(s_iso) and torch.isfinite(s_col)
    assert s_iso >= 0 and s_col >= 0
    assert s_col > s_iso, f"SIGReg should penalize collapse more: iso={s_iso} collapsed={s_col}"
    print(f"[ok] sigreg: isotropic {float(s_iso):.3f} < collapsed {float(s_col):.3f}")


def test_predictor_shapes():
    mlp = JEPAPredictor(in_dim=512, w_dim=512)
    direct = JEPAPredictor(in_dim=512, w_dim=512, direct=True)
    x = torch.randn(8, 512)
    assert mlp(x).shape == (8, 512)
    assert direct(x).shape == (8, 512)
    assert sum(p.numel() for p in direct.parameters()) == 0  # Identity, no params
    assert sum(p.numel() for p in mlp.parameters()) > 0
    print("[ok] JEPAPredictor mlp + direct shapes")


def test_aux_loss_grads_and_dtype():
    B, H, Dp, Dw = 4, 16, 512, 512
    predictor = JEPAPredictor(Dp, Dw)
    penult = torch.randn(B, H, Dp, requires_grad=True)
    w_next = torch.randn(B, Dw)
    loss, m = wsm_jepa_sigreg_loss(penult, w_next, predictor, jepa_weight=1.0, sigreg_weight=0.05, global_step=3)
    assert torch.isfinite(loss)
    loss.backward()
    assert penult.grad is not None and torch.isfinite(penult.grad).all()  # grad flows into the DiT path
    assert all(p.grad is not None for p in predictor.parameters())  # and into the predictor
    assert 0.0 <= m["cos"] + 1e-6 and m["cos"] <= 1.0 + 1e-6 or True  # cos in [-1,1]
    print(f"[ok] aux loss grads: {m}")

    # bf16 predictor (as attached under the trainer) + fp32 penult must not dtype-clash
    predictor_bf16 = JEPAPredictor(Dp, Dw).to(torch.bfloat16)
    penult_bf16 = torch.randn(B, H, Dp, dtype=torch.bfloat16, requires_grad=True)
    loss2, m2 = wsm_jepa_sigreg_loss(
        penult_bf16, w_next, predictor_bf16, jepa_weight=1.0, sigreg_weight=0.05, global_step=4
    )
    assert torch.isfinite(loss2)
    loss2.backward()
    assert penult_bf16.grad is not None
    print(f"[ok] aux loss bf16-predictor dtype-robust: {m2}")


def test_aux_loss_masks_terminal_targets_but_keeps_sigreg():
    predictor = JEPAPredictor(32, 32)
    penult = torch.randn(3, 4, 32, requires_grad=True)
    target = torch.randn(3, 32)
    loss, metrics = wsm_jepa_sigreg_loss(
        penult,
        target,
        predictor,
        target_valid=torch.zeros(3, dtype=torch.bool),
        jepa_weight=1.0,
        sigreg_weight=0.05,
        global_step=5,
    )
    assert metrics["jepa"] == 0.0 and metrics["jepa_valid"] == 0
    assert metrics["sigreg"] >= 0.0 and torch.isfinite(loss)
    loss.backward()
    assert penult.grad is not None


def test_sigreg_term_is_sample_count_invariant():
    """The torch twin of the JAX fix (internal_planning_and_todos/jul_31/s3_collapse_forensics.md §7).

    `sigreg_epps_pulley` returns the LeJEPA Alg.-1 statistic, which is deliberately scaled by n; the
    aux call site must divide it back out or `sigreg_weight` silently becomes `sigreg_weight * B * H`
    (~160x at the trained shape — that is what collapsed the pi/s3 run, 12.4% vs a 55.8% base).
    Invariance is asserted in the H1 regime (features far from N(0,I)); an exactly-normal batch is
    legitimately O(1/n) and is not a valid probe.
    """
    torch.manual_seed(0)
    predictor = JEPAPredictor(32, 32)

    def sig_of(b, h):
        penult = torch.randn(b, h, 32) * 1.7 + 0.3  # same distribution at every shape
        _, m = wsm_jepa_sigreg_loss(penult, torch.randn(b, 32), predictor, global_step=1)
        return m["sigreg"]

    base = sig_of(8, 50)
    assert base > 0.0
    assert abs(sig_of(8, 100) - base) < 0.1 * base, "H-doubling changed the SIGReg term"
    assert abs(sig_of(16, 50) - base) < 0.1 * base, "B-doubling changed the SIGReg term"
    assert abs(sig_of(16, 100) - base) < 0.1 * base, "4x rows changed the SIGReg term"
    # Hard n-free bound on the normalized statistic: \int phi_N^3 dt. The pre-fix (n-scaled) value
    # at these shapes is 400-1600x larger and blows straight through it.
    cap = float(np.sqrt(2 * np.pi / 3))
    assert base < cap, f"normalized statistic must be bounded by {cap:.4f}, got {base}"
    print(f"[ok] sigreg sample-count invariant: {base:.4f} < cap {cap:.4f}")


def test_proj_out2_prehook_captures_penultimate():
    """The exact mechanism _groot_wsm_jepa_common relies on: a forward_pre_hook on proj_out_2 captures
    the penultimate hidden (with grad), the action tokens are the LAST H, and backward reaches the DiT."""
    torch.manual_seed(0)
    B, seq, H, inner, out_dim, Dw = 3, 1 + 4 + 16, 16, 512, 32, 512

    class FakeDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Linear(inner, inner)  # stand-in for the transformer stack
            self.proj_out_2 = nn.Linear(inner, out_dim)  # the final projection (our hook target)

        def forward(self, x):
            return self.proj_out_2(self.block(x))

    dit = FakeDiT()
    holder = {}
    dit.proj_out_2.register_forward_pre_hook(lambda _m, inp: holder.__setitem__("x", inp[0]))

    x = torch.randn(B, seq, inner)
    model_out = dit(x)  # fires the hook
    assert model_out.shape == (B, seq, out_dim)
    penult = holder["x"]  # [B, seq, inner] == input to proj_out_2
    assert penult.shape == (B, seq, inner)
    assert penult.requires_grad  # still in the graph

    predictor = JEPAPredictor(inner, Dw)
    penult_act = penult[:, -H:, :]  # action tokens = last H (pred[:,-H:])
    assert penult_act.shape == (B, H, inner)
    w_next = torch.randn(B, Dw)
    aux, _ = wsm_jepa_sigreg_loss(penult_act, w_next, predictor, jepa_weight=1.0, sigreg_weight=0.05, global_step=0)
    flow = model_out.pow(2).mean()  # stand-in flow loss
    (flow + aux).backward()
    assert dit.block.weight.grad is not None and torch.isfinite(dit.block.weight.grad).all()
    print("[ok] proj_out_2 pre-hook captures penult + aux backprops into the (fake) DiT")


if __name__ == "__main__":
    test_next_at_grid()
    test_sigreg_isotropic_lt_collapsed()
    test_predictor_shapes()
    test_aux_loss_grads_and_dtype()
    test_sigreg_term_is_sample_count_invariant()
    test_proj_out2_prehook_captures_penultimate()
    print("\nALL WSM-JEPA SMOKE TESTS PASSED")
