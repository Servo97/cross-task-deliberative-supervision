"""Torch `WSMGatedDeltaNetConditioner` (GR00T seam) vs the JAX original it was ported from.

The load-bearing test is `test_jax_parity`: the same parameters and the same omega window must give
the same conditioning vector in torch as in JAX, within fp tolerance. The fixture in
`tests/fixtures/deltanet_jax_parity.npz` was extracted from
`robocasa_openpi/src/openpi/models/wsm_current_cond.py::WSMGatedDeltaNetConditioner` by
`tests/fixtures/extract_deltanet_jax_parity.py` (run under the openpi jax env). Parameters there are
drawn from a seeded RNG rather than left at init precisely so that no term is inert: `pos_decay_bias`
is non-zero (otherwise the positional decay prior would not be exercised) and `alpha` is away from
1e-3 (otherwise tanh(alpha) ~ 0 would mask any readout bug).

The remaining tests pin the invariants that the port inherited and that a well-meaning refactor would
otherwise erase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from workspace_models.networks.wsm_gated_deltanet import (  # noqa: E402
    WSMGatedDeltaNetConditioner,
    current_workspace_token,
)

FIXTURE = Path(__file__).parent / "fixtures" / "deltanet_jax_parity.npz"


def _from_fixture():
    f = np.load(FIXTURE)
    module = WSMGatedDeltaNetConditioner(
        w_dim=int(f["w_dim"]),
        cond_dim=int(f["cond_dim"]),
        window_len=int(f["window_len"]),
        num_heads=int(f["num_heads"]),
        head_dim=int(f["head_dim"]),
    )
    module.load_jax_params({k: f[k] for k in f.files})
    module.eval()
    return module, f


def test_jax_parity():
    """Identical omega window in => JAX and torch outputs agree within fp tolerance."""
    module, f = _from_fixture()
    with torch.no_grad():
        got = module(torch.as_tensor(f["omega_window"], dtype=torch.float32)).numpy()
    ref = f["out"]
    assert got.shape == ref.shape
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)


def test_parity_holds_per_sample():
    """The recurrence must not leak across the batch: row i alone == row i inside the batch."""
    module, f = _from_fixture()
    window = torch.as_tensor(f["omega_window"], dtype=torch.float32)
    with torch.no_grad():
        batched = module(window)
        single = module(window[1:2])
    np.testing.assert_allclose(single.numpy(), batched[1:2].numpy(), atol=1e-6, rtol=1e-5)


def test_window_length_mismatch_is_a_hard_error():
    """A checkpoint trained at K=8 fed a K=1 window must fail loudly, never silently reshape."""
    module, f = _from_fixture()
    bad = torch.zeros(2, int(f["window_len"]) - 1, int(f["w_dim"]))
    with pytest.raises(ValueError, match="window"):
        module(bad)


def test_pos_decay_bias_shape_encodes_the_trained_window():
    """Serve auto-detects the trained window from this parameter's leading axis — keep it [K, H]."""
    module = WSMGatedDeltaNetConditioner(w_dim=8, cond_dim=6, window_len=8, num_heads=2, head_dim=4)
    assert tuple(module.pos_decay_bias.shape) == (8, 2)
    assert torch.all(module.pos_decay_bias == 0.0)  # exact no-op at init


def test_gate_is_near_silent_at_init_but_every_param_has_a_live_gradient():
    """Init discipline: near-silence comes ONLY from tanh(alpha); the readout is NOT zero-init."""
    module = WSMGatedDeltaNetConditioner(w_dim=8, cond_dim=6, window_len=4, num_heads=2, head_dim=4, gate_init=1e-3)
    window = torch.randn(3, 4, 8)
    out = module(window)
    assert out.abs().max().item() < 0.05  # gated to near-silence at step 0
    out.sum().backward()
    dead = [n for n, p in module.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"parameters with no gradient at step 0: {dead}"


def test_output_is_finite_under_a_large_window_input():
    """l2 on k keeps k k^T O(1); a large-magnitude window must not blow the state up."""
    module = WSMGatedDeltaNetConditioner(w_dim=16, cond_dim=8, window_len=8, num_heads=2, head_dim=8)
    out = module(torch.randn(4, 8, 16) * 50.0)
    assert torch.isfinite(out).all()


def test_history_dropout_is_train_only():
    """An eval path handed a dropout generator is a silent distribution shift — refuse it."""
    module = WSMGatedDeltaNetConditioner(
        w_dim=8, cond_dim=6, window_len=4, num_heads=2, head_dim=4, history_dropout=0.5
    )
    module.eval()
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="train-only"):
        module(torch.randn(2, 4, 8), train=False, generator=g)


def test_history_dropout_never_deletes_the_current_timestep():
    """The arm must become "less history", never "no current observation"."""
    module = WSMGatedDeltaNetConditioner(
        w_dim=8, cond_dim=6, window_len=4, num_heads=2, head_dim=4, history_dropout=0.9
    )
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        keep = module._history_keep_mask((5,), 4, torch.device("cpu"), g)
        assert bool(keep[..., -1].all())


def test_history_dropout_off_is_the_pre_knob_computation():
    """history_dropout == 0.0 must not perturb the traced computation at all."""
    module, f = _from_fixture()
    window = torch.as_tensor(f["omega_window"], dtype=torch.float32)
    with torch.no_grad():
        module.train()
        train_out = module(window)
        module.eval()
        eval_out = module(window)
    np.testing.assert_allclose(train_out.numpy(), eval_out.numpy(), atol=0, rtol=0)


def test_current_workspace_token_selects_the_newest_row():
    window = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    np.testing.assert_array_equal(current_workspace_token(window).numpy(), window[:, -1, :].numpy())
