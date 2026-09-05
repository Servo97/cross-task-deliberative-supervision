"""RoboTTT fast weights (reduced form), torch/GR00T port — the update math and its invariants.

The load-bearing checks are the ones that unit tests have historically MISSED on this mechanism:

* `test_stability_at_production_dims` — the D-9 regression. With a per-token SUM inner loss the
  inner-GD curvature scales with d; at production geometry (d=256, h=128) eta=0.1 sits ~10x past the
  2/lambda stability boundary and W diverges to NaN within 5 commits. The toy dims (d=16, h=8) are
  stable, which is exactly why the original tests passed while production diverged. This test runs at
  PRODUCTION dims for 32 commits.
* `test_maml_gradient_reaches_w0_through_the_commit` — if the inner update were detached, W_0 would
  silently stop being meta-learned and the arm would degrade to a fixed init with no error anywhere.
* `test_gelu_matches_jax_default` — `jax.nn.gelu` defaults to the tanh approximation, `F.gelu` to
  exact erf. Getting this wrong makes the torch arm a quietly different model.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from workspace_models.networks.robottt_fast_weights import (  # noqa: E402
    RoboTTTConfig,
    RoboTTTFastWeights,
    _gelu,
)

TOY = dict(token_dim=16, fast_hidden=8, num_registers=4, cond_dim=12, state_dim=6, action_dim=5, action_horizon=4)


def _mk(**over):
    cfg = RoboTTTConfig(**{**TOY, **over})
    torch.manual_seed(0)
    return RoboTTTFastWeights(cfg), cfg


def _batch(cfg, b=3):
    return (torch.randn(b, cfg.state_dim), torch.randn(b, cfg.action_horizon, cfg.action_dim))


def test_gelu_matches_jax_default():
    """jax.nn.gelu's default is approximate='tanh'; exact erf differs by ~1e-3, far above tolerance."""
    x = torch.linspace(-3, 3, 101, dtype=torch.float64)
    tanh_form = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    torch.testing.assert_close(_gelu(x), tanh_form, atol=1e-6, rtol=1e-6)
    # and it is genuinely NOT the exact form, so the choice is observable
    assert (_gelu(x) - torch.nn.functional.gelu(x, approximate="none")).abs().max() > 1e-4


def test_output_is_exactly_zero_at_init():
    """W_0's second layer and the readout are both zero-init => O_t == 0 before any training."""
    m, cfg = _mk()
    state, _ = _batch(cfg)
    o = m.condition(m.init_state(state.shape[0]), state)
    assert torch.equal(o, torch.zeros_like(o))


def test_alpha_scale_zero_is_exact_parity_with_the_base_policy():
    """alpha_scale=0 must zero the contribution EXACTLY (the off/on parity control)."""
    m, cfg = _mk(alpha_scale=0.0)
    with torch.no_grad():  # perturb so only alpha_scale can be responsible for the zero
        m.readout.weight.normal_()
        m.readout.bias.normal_()
        m.w0_w2.normal_()
    state, actions = _batch(cfg)
    w = m.commit(m.init_state(state.shape[0]), state, actions)
    o = m.condition(w, state)
    assert torch.equal(o, torch.zeros_like(o))


def test_commit_decreases_the_inner_loss():
    """Eq 1 is a descent step: one commit must reduce L_FW on the same (K, V)."""
    m, cfg = _mk()
    with torch.no_grad():
        m.w0_w2.normal_(std=0.1)  # move off the zero init so there is something to descend
    state, actions = _batch(cfg)
    w0 = m.init_state(state.shape[0])
    tokens = m._tokens_from(state, actions)
    k, v = m.proj_k(tokens), m.proj_v(tokens)
    before = m._inner_loss(w0, k, v)
    w1 = m.commit(w0, state, actions)
    after = m._inner_loss({n: t.detach() for n, t in w1.items()}, k, v)
    assert (after < before).all(), f"inner loss did not decrease: {before} -> {after}"


def test_commit_is_per_sample_independent():
    """Sample i's fast weights must depend ONLY on sample i — no cross-batch leakage."""
    m, cfg = _mk()
    with torch.no_grad():
        m.w0_w2.normal_(std=0.1)
    state, actions = _batch(cfg, b=4)
    full = m.commit(m.init_state(4), state, actions)
    single = m.commit(m.init_state(1), state[1:2], actions[1:2])
    for name in ("w1", "b1", "w2", "b2"):
        torch.testing.assert_close(full[name][1:2], single[name], atol=1e-6, rtol=1e-5)


def test_maml_gradient_reaches_w0_through_the_commit():
    """The outer loss must backprop THROUGH the inner update into W_0 (grads-of-grads)."""
    m, cfg = _mk()
    m.train()
    with torch.no_grad():
        m.readout.weight.normal_(std=0.1)  # else O_t is identically zero and every grad is zero
        m.w0_w2.normal_(std=0.1)
    state, actions = _batch(cfg)
    w = m.commit(m.init_state(state.shape[0]), state, actions)  # W_0 -> W_1
    o = m.condition(w, state)  # readout uses the UPDATED W
    o.pow(2).mean().backward()
    for name in ("w0_w1", "w0_b1", "w0_w2", "w0_b2"):
        g = getattr(m, name).grad
        assert g is not None, f"{name} received no gradient — W_0 is not being meta-learned"
        assert torch.isfinite(g).all()
    assert m.w0_w1.grad.abs().sum() > 0, "W_0's first layer got only a zero gradient"
    # the learnable inner LR is on the meta path too (it scales the update)
    assert m.log_inner_lr.grad is not None and m.log_inner_lr.grad.abs() > 0


def test_tbptt_changes_gradients_but_not_forward_values():
    """TBPTT truncates gradients only; the rolled-out O_seq must be bitwise identical."""
    m, cfg = _mk(window_len=8, tbptt_segment=8)
    m.train()
    with torch.no_grad():
        m.readout.weight.normal_(std=0.1)
        m.w0_w2.normal_(std=0.1)
    b, L = 2, 8
    state_seq = torch.randn(b, L, cfg.state_dim)
    action_seq = torch.randn(b, L, cfg.action_horizon, cfg.action_dim)

    o_full, _ = m.run_sequence(state_seq, action_seq, tbptt_segment=99)
    o_trunc, _ = m.run_sequence(state_seq, action_seq, tbptt_segment=2)
    torch.testing.assert_close(o_full, o_trunc, atol=0, rtol=0)  # forward values identical

    def grad_of(seg):
        m.zero_grad(set_to_none=True)
        o, _ = m.run_sequence(state_seq, action_seq, tbptt_segment=seg)
        o.pow(2).mean().backward()
        return m.w0_w1.grad.clone()

    g_full, g_trunc = grad_of(99), grad_of(2)
    assert torch.isfinite(g_full).all() and torch.isfinite(g_trunc).all()
    # Compare RELATIVELY: the raw magnitudes here are ~1e-11 (zero-init readout keeps the whole
    # meta-path small), so torch.allclose's absolute tolerance would call any two of them equal and
    # the test would pass even if truncation were a no-op.
    rel = (g_full - g_trunc).norm() / g_full.norm()
    assert rel > 0.1, f"TBPTT did not actually truncate the gradient (relative change {rel:.3g})"
    # truncating strictly drops contributions, so the full-BPTT gradient should be the larger one
    assert g_full.norm() > g_trunc.norm()


def test_w0_still_gets_gradient_through_the_first_segment():
    """Truncation must not orphan W_0 — it is applied at t=0 and is the base of every commit."""
    m, cfg = _mk()
    m.train()
    with torch.no_grad():
        m.readout.weight.normal_(std=0.1)
        m.w0_w2.normal_(std=0.1)
    state_seq = torch.randn(2, 8, cfg.state_dim)
    action_seq = torch.randn(2, 8, cfg.action_horizon, cfg.action_dim)
    o, _ = m.run_sequence(state_seq, action_seq, tbptt_segment=2)
    o.pow(2).mean().backward()
    assert m.w0_w1.grad is not None and m.w0_w1.grad.abs().sum() > 0


def test_commit_works_under_no_grad_for_serving():
    """The inner update is a gradient step; it must survive the no_grad that wraps the serve path."""
    m, cfg = _mk()
    m.eval()
    state, actions = _batch(cfg)
    with torch.no_grad():
        w = m.commit(m.init_state(state.shape[0]), state, actions)
        o = m.condition(w, state)
    assert all(not t.requires_grad for t in w.values()), "serve-time W must be detached"
    assert torch.isfinite(o).all()


def test_stability_at_production_dims():
    """D-9 regression: 32 commits at d=256/h=128 with eta=0.1 must stay finite and bounded.

    A per-token SUM inner loss diverges to NaN within ~5 commits here while remaining stable at toy
    dims. This test exists because the toy-dims-only suite is what let that bug reach production.
    """
    cfg = RoboTTTConfig(
        token_dim=256, fast_hidden=128, num_registers=16, cond_dim=1536, state_dim=64, action_dim=32, action_horizon=16
    )
    torch.manual_seed(0)
    m = RoboTTTFastWeights(cfg)
    m.eval()
    with torch.no_grad():
        m.w0_w2.normal_(std=0.1)  # non-degenerate start so the update actually moves
    b = 4
    w = m.init_state(b)
    norms, losses = [], []
    with torch.no_grad():
        for step in range(32):
            state = torch.randn(b, cfg.state_dim)
            actions = torch.randn(b, cfg.action_horizon, cfg.action_dim)
            tokens = m._tokens_from(state, actions)
            losses.append(float(m._inner_loss(w, m.proj_k(tokens), m.proj_v(tokens)).mean()))
            w = m.commit(w, state, actions)
            total = math.sqrt(sum(float((t.detach() ** 2).sum()) for t in w.values()))
            assert all(torch.isfinite(t).all() for t in w.values()), f"W went non-finite at commit {step}"
            norms.append(total)
    assert norms[-1] < 10.0 * norms[0], f"|W| grew unboundedly: {norms[0]:.3f} -> {norms[-1]:.3f}"
    assert losses[-1] < losses[0], f"online inner loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_inner_loss_scale_is_dimension_independent():
    """Mean-normalization over tokens AND features is what makes (eta x curvature) d-independent."""
    vals = []
    for d in (32, 256):
        cfg = RoboTTTConfig(
            token_dim=d, fast_hidden=d // 2, num_registers=4, cond_dim=8, state_dim=6, action_dim=5, action_horizon=4
        )
        torch.manual_seed(0)
        m = RoboTTTFastWeights(cfg)
        b = 2
        w = m.init_state(b)
        k = torch.randn(b, 9, d)
        v = torch.randn(b, 9, d)
        vals.append(float(m._inner_loss(w, k, v).mean()))
    # an 8x change in d must not change the loss SCALE (a per-token sum would make it ~8x)
    assert vals[1] == pytest.approx(vals[0], rel=0.3), f"inner loss scales with d: {vals}"


def test_inner_lr_is_learnable_and_starts_at_base():
    m, cfg = _mk()
    assert float(m.inner_lr()) == pytest.approx(cfg.base_inner_lr, rel=1e-5)
    m2, _ = _mk(learn_inner_lr=False)
    assert float(m2.inner_lr()) == pytest.approx(cfg.base_inner_lr, rel=1e-5)


def test_token_bundle_shapes():
    """Query bundle = N+1 (no future action); update bundle = N+1+H."""
    m, cfg = _mk()
    state, actions = _batch(cfg)
    assert m._tokens_from(state, None).shape == (3, cfg.num_registers + 1, cfg.token_dim)
    assert m._tokens_from(state, actions).shape == (3, cfg.num_registers + 1 + cfg.action_horizon, cfg.token_dim)
