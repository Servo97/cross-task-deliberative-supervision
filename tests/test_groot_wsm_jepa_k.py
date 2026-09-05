"""GR00T JEPA aux: multi-future (k) targets and the loss invariants that a bug here would break.

`test_target_matches_pi_semantics` is the load-bearing one. The omega-target pipeline is the exact
place a previous bug cost a 12.4-avg collapse, so the groot target selection is pinned against a
fixture produced by CALLING THE SHIPPED pi loader
(`openpi.groot_utils.groot_openpi_dataset._wsm_jepa_target`, driven by
`tests/fixtures/extract_jepa_target_pi_semantics.py`) rather than against a reimplementation of it,
which would be free to agree with a bug.

The other invariant class is SAMPLE-COUNT / k INVARIANCE of the aux terms. The s3 collapse was an
n-scaled aux term: with per-token features the configured lambda acted as ~160x, the aux was 99% of
the step-0 gradient for 60k steps, and the arm evaluated at 12.4% against a 55.8% base. Both the
JEPA term (masked MEAN over B and k) and the SIGReg term (`/n` at the call site) must therefore be
invariant to B, to the action horizon, and to k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from workspace_models.features.wsm_align import next_at_with_valid, next_k_at  # noqa: E402
from workspace_models.networks.jepa_align_head import (  # noqa: E402
    JEPAPredictor,
    wsm_jepa_sigreg_loss,
)

FIXTURE = Path(__file__).parent / "fixtures" / "jepa_target_pi_semantics.npz"


def test_target_matches_pi_semantics():
    """groot `next_k_at` must select exactly what the shipped pi loader selects, for every k."""
    f = np.load(FIXTURE)
    n_cases = int(f["num_cases"])
    checked = 0
    for c in range(n_cases):
        w, fi, frames = f[f"case{c}_w"], f[f"case{c}_fi"], f[f"case{c}_frames"]
        for k in [int(x) for x in f["ks"]]:
            ref_t, ref_v = f[f"case{c}_k{k}_target"], f[f"case{c}_k{k}_valid"]
            for row, frame in enumerate(frames):
                if k == 1:
                    got_t, got_v = next_at_with_valid(w, fi, int(frame))
                else:
                    got_t, got_v = next_k_at(w, fi, int(frame), ks=tuple(range(1, k + 1)))
                np.testing.assert_array_equal(
                    np.asarray(got_t, dtype=np.float32),
                    ref_t[row],
                    err_msg=f"case={c} k={k} frame={frame}: target rows differ from pi",
                )
                np.testing.assert_array_equal(
                    np.asarray(got_v, dtype=np.bool_),
                    ref_v[row],
                    err_msg=f"case={c} k={k} frame={frame}: valid mask differs from pi",
                )
                checked += 1
    assert checked > 0


def test_predictor_shapes_and_k1_is_unchanged():
    """k=1 parameter shapes must be untouched so existing `jepa_predictor` checkpoints still load."""
    p1 = JEPAPredictor(in_dim=1536, w_dim=512, num_futures=1)
    p16 = JEPAPredictor(in_dim=1536, w_dim=512, num_futures=16)
    assert p1(torch.randn(4, 1536)).shape == (4, 512)
    assert p16(torch.randn(4, 1536)).shape == (4, 16, 512)
    s1 = {k: tuple(v.shape) for k, v in p1.state_dict().items()}
    assert s1 == {"net.0.weight": (512, 1536), "net.0.bias": (512,), "net.2.weight": (512, 512), "net.2.bias": (512,)}
    # k>1 widens ONLY the output projection; the trunk is shared, not duplicated.
    s16 = {k: tuple(v.shape) for k, v in p16.state_dict().items()}
    assert s16["net.0.weight"] == s1["net.0.weight"]
    assert s16["net.2.weight"] == (512 * 16, 512)


def test_direct_rejects_multi_future():
    with pytest.raises(ValueError, match="direct"):
        JEPAPredictor(in_dim=512, w_dim=512, direct=True, num_futures=16)


def test_loss_shape_guard_catches_k_mismatch():
    """A k-mismatched loader must hard-fail, never silently broadcast into a wrong loss."""
    pred = JEPAPredictor(in_dim=64, w_dim=32, num_futures=4)
    penult = torch.randn(3, 16, 64)
    with pytest.raises(ValueError, match="num_futures"):
        wsm_jepa_sigreg_loss(
            penult, torch.randn(3, 32), pred, num_futures=4, target_valid=torch.ones(3, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="num_futures"):
        wsm_jepa_sigreg_loss(
            penult, torch.randn(3, 4, 32), pred, num_futures=4, target_valid=torch.ones(3, dtype=torch.bool)
        )


def test_jepa_term_is_invariant_to_k():
    """Identical per-(sample,future) cosines must give the same JEPA term at k=1 and k=16.

    If the masked reduction were a SUM (or divided by B rather than by B*k), the term -- and hence
    the effective lambda -- would scale with k. That is the s3 bug class.
    """
    torch.manual_seed(0)
    penult = torch.randn(6, 16, 64)
    p1 = JEPAPredictor(in_dim=64, w_dim=32, num_futures=1)
    with torch.no_grad():  # make every future target identical => identical cosines
        base = p1(penult.mean(1))
    tgt1 = base
    tgt16 = base[:, None, :].expand(6, 16, 32).contiguous()

    p16 = JEPAPredictor(in_dim=64, w_dim=32, num_futures=16)
    with torch.no_grad():  # force the k=16 head to emit the same vector on every future slot
        p16.net[0].load_state_dict(p1.net[0].state_dict())
        p16.net[2].weight.copy_(p1.net[2].weight.repeat(16, 1))
        p16.net[2].bias.copy_(p1.net[2].bias.repeat(16))

    _, m1 = wsm_jepa_sigreg_loss(penult, tgt1, p1, num_futures=1, target_valid=torch.ones(6, dtype=torch.bool))
    _, m16 = wsm_jepa_sigreg_loss(penult, tgt16, p16, num_futures=16, target_valid=torch.ones(6, 16, dtype=torch.bool))
    assert m1["jepa"] == pytest.approx(m16["jepa"], abs=1e-5)
    assert m16["jepa_total"] == 6 * 16 and m16["jepa_valid"] == 6 * 16


def test_sigreg_is_sample_count_invariant():
    """SIGReg must not scale with B or the action horizon — the `/n` fix (s3 collapse).

    The features are deliberately NOT standard normal (scaled and shifted), so the statistic has a
    real O(1) population value rather than sitting at the null where it would be pure O(1/n) MC
    noise and would shrink with n for reasons that have nothing to do with the bug. Under the
    pre-fix `* n` scaling, 4x the rows would multiply the value by ~4x.
    """
    torch.manual_seed(0)
    pred = JEPAPredictor(in_dim=64, w_dim=32)
    small = torch.randn(8, 16, 64) * 3.0 + 1.0
    large = torch.randn(32, 16, 64) * 3.0 + 1.0
    _, ms = wsm_jepa_sigreg_loss(
        small, torch.randn(8, 32), pred, target_valid=torch.ones(8, dtype=torch.bool), global_step=0
    )
    _, ml = wsm_jepa_sigreg_loss(
        large, torch.randn(32, 32), pred, target_valid=torch.ones(32, dtype=torch.bool), global_step=0
    )
    assert ms["sigreg"] > 0.01  # a real discrepancy, not the null
    assert ms["sigreg"] == pytest.approx(ml["sigreg"], rel=0.25)


def test_sigreg_scales_with_neither_batch_nor_horizon_independently():
    """Doubling the ACTION HORIZON alone must not move the statistic either (n = B*H)."""
    torch.manual_seed(0)
    pred = JEPAPredictor(in_dim=64, w_dim=32)
    h16 = torch.randn(8, 16, 64) * 3.0 + 1.0
    h32 = torch.randn(8, 32, 64) * 3.0 + 1.0
    _, m16 = wsm_jepa_sigreg_loss(
        h16, torch.randn(8, 32), pred, target_valid=torch.ones(8, dtype=torch.bool), global_step=0
    )
    _, m32 = wsm_jepa_sigreg_loss(
        h32, torch.randn(8, 32), pred, target_valid=torch.ones(8, dtype=torch.bool), global_step=0
    )
    assert m16["sigreg"] == pytest.approx(m32["sigreg"], rel=0.25)


def test_all_invalid_targets_keep_a_live_gradient():
    """A batch whose futures are all terminal must contribute a graph-connected zero, not a NaN."""
    pred = JEPAPredictor(in_dim=64, w_dim=32, num_futures=4)
    penult = torch.randn(3, 16, 64, requires_grad=True)
    total, m = wsm_jepa_sigreg_loss(
        penult, torch.randn(3, 4, 32), pred, num_futures=4, target_valid=torch.zeros(3, 4, dtype=torch.bool)
    )
    assert m["jepa"] == pytest.approx(0.0, abs=1e-6) and m["jepa_valid"] == 0
    total.backward()
    assert penult.grad is not None and torch.isfinite(penult.grad).all()
    assert penult.grad.abs().sum() > 0  # SIGReg still trains the representation


def test_masked_mean_ignores_padded_rows():
    """Padded (invalid) future rows must not enter the JEPA mean at all."""
    torch.manual_seed(0)
    pred = JEPAPredictor(in_dim=64, w_dim=32, num_futures=4)
    penult = torch.randn(5, 16, 64)
    tgt = torch.randn(5, 4, 32)
    valid = torch.ones(5, 4, dtype=torch.bool)
    valid[:, 2:] = False
    _, m_masked = wsm_jepa_sigreg_loss(penult, tgt, pred, num_futures=4, target_valid=valid)
    # Corrupting only the masked slots must not move the loss.
    tgt2 = tgt.clone()
    tgt2[:, 2:] = 1e3
    _, m_corrupt = wsm_jepa_sigreg_loss(penult, tgt2, pred, num_futures=4, target_valid=valid)
    assert m_masked["jepa"] == pytest.approx(m_corrupt["jepa"], abs=1e-6)
    assert m_masked["jepa_valid"] == 10
