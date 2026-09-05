"""Unit smoke for the pi0.5 WSM-CFG conditioner (JAX/nnx) — the pi twin of tests/test_wsm_cfg.py.

Runs on a CPU jax (no GPU / no PaliGemma weights needed): it exercises the SELF-CONTAINED conditioner
(workspace_models-equivalent openpi.models.wsm_cfg_cond.WSMCfgConditioner) plus the CFG velocity formula,
which are the parts most likely to be silently wrong. The full Pi0 seam (compute_loss inject, sample_actions
two-pass) is exercised on the node where the backbone weights exist; here we assert the invariants the seam
relies on:

  1. ZERO-INIT == BASELINE: a freshly-built conditioner outputs exactly 0 (so adarms_cond is unchanged and
     the policy is byte-identical to the baseline finetune at step 0), for both the conditional and the
     unconditional (force_uncond) pass.
  2. DROPOUT EXTREMES: p_drop=1 => every row is the null (cond pass == uncond pass); p_drop=0 => no row is
     dropped (deterministic given w).
  3. LIVE GRAD THROUGH ZERO-INIT: d(loss)/d(proj_t_out.kernel) is non-zero at init (the additive path is NOT
     dead, unlike the multiplicative TokenModulator — see wsm_modulator.py's note).
  4. RNG THREADING: same rng => same dropout mask; different rng => (statistically) different.
  5. CFG VELOCITY: v = v_u + s*(v_c - v_u) reduces to v_u at s=0 and v_c at s=1.

Skips cleanly at module scope if jax/flax is not importable.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


def _skip(msg):
    pytest.skip(msg, allow_module_level=True)


try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
except Exception as e:  # noqa: BLE001
    _skip(f"jax/flax not importable ({e}) — run on the openpi venv")

from wsm_settings import ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))
try:
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.wsm_cfg_cond import WSMCfgConditioner
    from openpi.models.wsm_current_cond import (
        WSMCurrentCfgConditioner,
        WSMTanhConditioner,
        cfg_velocity,
        current_workspace_token,
    )
    from openpi.policies.policy import _noise_from_seed, _pop_policy_noise_seed
except Exception as e:  # noqa: BLE001
    _skip(f"OpenPI workspace conditioners not importable ({e})")

W_DIM, COND_DIM, B = 512, 1024, 4


def _mk(p_drop=0.2):
    return WSMCfgConditioner(w_dim=W_DIM, cond_dim=COND_DIM, p_drop=p_drop, rngs=nnx.Rngs(0))


def _w(seed=1):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((B, W_DIM)), dtype=jnp.float32)


def test_zero_init_is_baseline():
    cond = _mk()
    w = _w()
    out_cond = cond(w, None, train=False, rng=None, force_uncond=False)
    out_unc = cond(w, None, train=False, rng=None, force_uncond=True)
    assert out_cond.shape == (B, COND_DIM), out_cond.shape
    assert float(jnp.max(jnp.abs(out_cond))) == 0.0, "zero-init cond pass must be exactly 0"
    assert float(jnp.max(jnp.abs(out_unc))) == 0.0, "zero-init uncond pass must be exactly 0"
    print("✓ zero-init == baseline (cond & uncond both exactly 0)")


def _detrain(cond):
    """Force proj_t_out away from zero so dropout/grad effects are observable (mimics a trained conditioner)."""
    k = cond.proj_t_out.kernel.value
    cond.proj_t_out.kernel.value = k + 0.1 * jax.random.normal(jax.random.key(7), k.shape)
    b = cond.proj_t_out.bias.value
    cond.proj_t_out.bias.value = b + 0.1


def test_dropout_extremes():
    # p_drop=1 -> every row dropped to null -> cond pass identical to the uncond pass.
    c1 = _mk(p_drop=1.0)
    _detrain(c1)
    w = _w()
    train_all_drop = c1(w, None, train=True, rng=jax.random.key(0), force_uncond=False)
    uncond = c1(w, None, train=False, rng=None, force_uncond=True)
    assert jnp.allclose(train_all_drop, uncond), "p_drop=1 must equal the unconditional (all-null) pass"
    # p_drop=0 -> no row dropped -> train == eval-conditional (deterministic given w).
    c0 = _mk(p_drop=0.0)
    _detrain(c0)
    train_no_drop = c0(w, None, train=True, rng=jax.random.key(0), force_uncond=False)
    eval_cond = c0(w, None, train=False, rng=None, force_uncond=False)
    assert jnp.allclose(train_no_drop, eval_cond), "p_drop=0 must equal the eval-conditional pass"
    print("✓ dropout extremes: p_drop=1 -> uncond, p_drop=0 -> conditional")


def test_live_grad_through_zero_init():
    cond = _mk()
    w = _w()
    graphdef, params = nnx.split(cond, nnx.Param)

    def loss_fn(params):
        m = nnx.merge(graphdef, params)
        out = m(w, None, train=False, rng=None, force_uncond=False)  # 0 at init, but grad must be live
        return jnp.sum(out**2) + jnp.sum(out)  # +sum so grad of an all-zero output is still non-trivial

    grads = jax.grad(loss_fn)(params)
    flat = {"/".join(str(k) for k in path): v for path, v in nnx.to_flat_state(grads)}
    out_kernel_g = next(v.value for k, v in flat.items() if "proj_t_out" in k and "kernel" in k)
    g_l2 = float(jnp.linalg.norm(out_kernel_g))
    assert g_l2 > 0.0, f"proj_t_out.kernel grad is dead ({g_l2}) — additive zero-init should be LIVE"
    print(f"✓ live grad through zero-init proj_t_out (||g||={g_l2:.4g} > 0)")


def test_rng_threading():
    cond = _mk(p_drop=0.5)
    _detrain(cond)
    w = _w()
    a = cond(w, None, train=True, rng=jax.random.key(123), force_uncond=False)
    a2 = cond(w, None, train=True, rng=jax.random.key(123), force_uncond=False)
    b = cond(w, None, train=True, rng=jax.random.key(999), force_uncond=False)
    assert jnp.allclose(a, a2), "same rng must give the same dropout mask"
    assert not jnp.allclose(a, b), "different rng should (statistically) give a different mask"
    print("✓ rng threading deterministic per-key, varies across keys")


def test_cfg_velocity_formula():
    v_u = jnp.asarray(np.random.default_rng(2).standard_normal((B, 16, 32)), jnp.float32)
    v_c = jnp.asarray(np.random.default_rng(3).standard_normal((B, 16, 32)), jnp.float32)
    for s, ref in [(0.0, v_u), (1.0, v_c)]:
        v = v_u + s * (v_c - v_u)
        # atol loosened: v_u + 1*(v_c - v_u) is not bit-identical to v_c in float32 (round-off near 0).
        assert jnp.allclose(v, ref, atol=1e-5, rtol=1e-4), f"CFG at s={s} should reduce to the right endpoint"
    # monotone interpolation/extrapolation sanity at s=2
    v2 = v_u + 2.0 * (v_c - v_u)
    assert jnp.allclose(v2, 2 * v_c - v_u, atol=1e-5, rtol=1e-4)
    print("✓ CFG velocity v = v_u + s*(v_c - v_u): s=0->uncond, s=1->cond, s=2 extrapolates")


def test_current_only_cfg2_tree_and_causality():
    cond = WSMCurrentCfgConditioner(w_dim=W_DIM, cond_dim=COND_DIM, p_drop=0.2, rngs=nnx.Rngs(11))
    paths = ["/".join(str(k) for k in path) for path, _ in nnx.state(cond, nnx.Param).flat_state()]
    assert paths
    assert not any("next" in path or "future" in path for path in paths), paths
    signature_names = inspect.signature(cond.__call__).parameters
    assert not any("next" in name or "future" in name for name in signature_names)

    window = jnp.asarray(np.random.default_rng(12).standard_normal((B, 4, W_DIM)), dtype=jnp.float32)
    changed_past = window.at[:, :-1, :].set(1000.0)
    omega = current_workspace_token(window)
    omega_changed = current_workspace_token(changed_past)
    assert jnp.array_equal(omega, window[:, -1, :])
    assert jnp.array_equal(omega, omega_changed), "earlier window slots must not affect omega_t"

    # Make the zero-init output observable, then verify identical omega_t gives identical conditioning.
    k = cond.proj_t_out.kernel.value
    cond.proj_t_out.kernel.value = k + 0.1 * jax.random.normal(jax.random.key(13), k.shape)
    out = cond(omega, train=False, rng=None)
    out_changed = cond(omega_changed, train=False, rng=None)
    assert jnp.allclose(out, out_changed)
    print("✓ CFG2 has only P_t/null_t and reads only causal omega_t (newest window slot)")


def test_current_only_cfg2_zero_init_and_live_grad():
    cond = WSMCurrentCfgConditioner(w_dim=W_DIM, cond_dim=COND_DIM, p_drop=0.2, rngs=nnx.Rngs(14))
    omega = _w(15)
    out = cond(omega, train=False, rng=None)
    unc = cond(omega, train=False, rng=None, force_uncond=True)
    assert float(jnp.max(jnp.abs(out))) == 0.0
    assert float(jnp.max(jnp.abs(unc))) == 0.0

    graphdef, params = nnx.split(cond, nnx.Param)

    def loss_fn(p):
        model = nnx.merge(graphdef, p)
        y = model(omega, train=False, rng=None)
        return jnp.sum(y)

    grads = jax.grad(loss_fn)(params)
    flat = {"/".join(str(k) for k in path): v for path, v in nnx.to_flat_state(grads)}
    out_kernel_g = next(v.value for k, v in flat.items() if "proj_t_out" in k and "kernel" in k)
    assert float(jnp.linalg.norm(out_kernel_g)) > 0.0
    print("✓ CFG2 zero-init is exact baseline and its output projection has live gradient")


def test_tanh_zero_gate_and_initialization():
    cond = WSMTanhConditioner(w_dim=W_DIM, cond_dim=COND_DIM, gate_init=1e-3, rngs=nnx.Rngs(16))
    omega = _w(17)
    assert jnp.allclose(cond.alpha.value, 1e-3)
    near_silent = cond(omega)
    assert bool(jnp.isfinite(near_silent).all())
    assert float(jnp.max(jnp.abs(near_silent))) > 0.0

    cond.alpha.value = jnp.zeros_like(cond.alpha.value)
    bypass = cond(omega)
    assert float(jnp.max(jnp.abs(bypass))) == 0.0, "alpha=0 must be an exact additive no-op"

    try:
        WSMTanhConditioner(w_dim=W_DIM, cond_dim=COND_DIM, gate_init=float("nan"), rngs=nnx.Rngs(20))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite tanh gate initialization must fail loudly")
    print("✓ tanh alpha starts at 1e-3; alpha=0 is an exact zero-gate bypass")


def test_cfg_endpoint_fast_paths():
    v_c = np.random.default_rng(18).standard_normal((B, 3))
    v_u = np.random.default_rng(19).standard_normal((B, 3))

    def run(scale):
        calls = []

        def velocity(which):
            calls.append(which)
            return v_c if which == "cond" else v_u

        return cfg_velocity(velocity, "cond", "uncond", scale), calls

    out0, calls0 = run(0.0)
    out1, calls1 = run(1.0)
    out_half, calls_half = run(0.5)
    assert calls0 == ["uncond"] and np.array_equal(out0, v_u)
    assert calls1 == ["cond"] and np.array_equal(out1, v_c)
    assert calls_half == ["uncond", "cond"]
    assert np.allclose(out_half, v_u + 0.5 * (v_c - v_u))
    print("✓ CFG endpoints use one suffix call: s=0 learned-null, s=1 conditional")


def test_cfg_rng_preserves_stock_streams():
    from openpi.models.pi0 import Pi0

    loss_source = inspect.getsource(Pi0.compute_loss)
    assert "jax.random.split(rng, 3)" in loss_source
    assert "jax.random.split(rng, 4)" not in loss_source
    assert "jax.random.fold_in(rng, _WSM_CFG_RNG_DOMAIN)" in loss_source
    print("✓ CFG dropout is domain-separated without perturbing stock augmentation/noise/time RNG streams")


def test_serve_requires_explicit_interface_and_pinned_tap():
    root = Path(__file__).resolve().parents[1]
    source = (root / "vla_training/eval/serve_pi_05_wsm_cfg.py").read_text()

    tap_pos = source.index('"--tap-ckpt"')
    tap_block = source[tap_pos : tap_pos + 300]
    assert "required=True" in tap_block
    assert "Pi05BackboneTap(args.tap_ckpt" in source
    assert "Pi05BackboneTap(args.finetune_ckpt" not in source

    interface_pos = source.index('"--interface"')
    interface_block = source[interface_pos : interface_pos + 300]
    assert "required=True" in interface_block
    assert "default=" not in interface_block
    assert 'require_wsm_prompt=args.interface != "legacy_cfg"' in source
    assert "forbids historical demo-expanded tap prompts" in source
    print("✓ serving requires an explicit interface and a separately pinned feature-source tap checkpoint")


def test_modes_are_exclusive_and_prefix_is_untouched():
    try:
        Pi0Config(pi05=True, wsm_cfg2=True, wsm_tanh=True)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("multiple workspace interfaces must fail loudly")

    from openpi.models.pi0 import Pi0

    prefix_source = inspect.getsource(Pi0.embed_prefix)
    loss_source = inspect.getsource(Pi0.compute_loss)
    sample_source = inspect.getsource(Pi0.sample_actions)
    assert "wsm_cfg2" not in prefix_source and "wsm_tanh" not in prefix_source
    assert "adarms_cond = adarms_cond + workspace_vec" in loss_source
    assert "cfg_velocity(velocity, cfg_cond, cfg_uncond" in sample_source
    print("✓ S1/S2 are exclusive, bypass embed_prefix, and share the additive adarms_cond seam")


def test_policy_noise_seed_is_request_order_independent():
    obs = {"observation/state": np.zeros(3), "policy_noise_seed": np.uint32(123)}
    clean, seed = _pop_policy_noise_seed(obs)
    assert seed == 123
    assert "policy_noise_seed" not in clean
    assert "policy_noise_seed" in obs, "reserved-key extraction must not mutate caller input"

    first = _noise_from_seed(seed, 5, 7)
    interleaved = _noise_from_seed(999, 5, 7)
    second = _noise_from_seed(seed, 5, 7)
    assert jnp.array_equal(first, second)
    assert not jnp.array_equal(first, interleaved)

    for bad in (-1, 2**32, np.array([1, 2])):
        try:
            _pop_policy_noise_seed({"policy_noise_seed": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid policy_noise_seed must fail: {bad!r}")
    print("✓ policy_noise_seed pins π diffusion noise independently of request/batch order")


if __name__ == "__main__":
    test_zero_init_is_baseline()
    test_dropout_extremes()
    test_live_grad_through_zero_init()
    test_rng_threading()
    test_cfg_velocity_formula()
    test_current_only_cfg2_tree_and_causality()
    test_current_only_cfg2_zero_init_and_live_grad()
    test_tanh_zero_gate_and_initialization()
    test_cfg_endpoint_fast_paths()
    test_cfg_rng_preserves_stock_streams()
    test_serve_requires_explicit_interface_and_pinned_tap()
    test_modes_are_exclusive_and_prefix_is_untouched()
    test_policy_noise_seed_is_request_order_independent()
    print("\n[test_pi_wsm_cfg] ALL PASS")
