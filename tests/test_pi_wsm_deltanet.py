"""Unit suite for the gated-DeltaNet STEERING conditioner (the s1 `cond_type: gated_deltanet` variant).

CPU-only (no PaliGemma weights): it exercises the self-contained conditioner, the Pi0Config selection
knobs, and the serve-side checkpoint auto-detection. The invariants asserted here are the ones that
would silently produce a garbage eval if they broke:

  1. DEFAULT IS BYTE-IDENTICAL: cond_type defaults to "tanh", and the default conditioner's parameter
     tree (structure AND tensor values, bitwise) is unchanged by the existence of this variant.
  2. SAME SUBTREE, DISTINGUISHING LEAVES: the deltanet registers under `wsm_tanh_cond` (so the 22-site
     interface map / missing_regex backfill / eval gates stay untouched) but carries leaf names no
     tanh checkpoint has.
  3. NEAR-IDENTITY AT INIT: output magnitude is gated by tanh(alpha=1e-3); alpha=0 is an exact no-op.
  4. LIVE GRADIENT TO EVERY PARAM at init — the zero-grad trap documented in wsm_modulator.py.
  5. RECURRENCE SANITY: full-forget decay => only the newest window token matters; beta=0 => the read
     is exactly the gated readout of a zero state.
  6. WINDOW CONTRACT: a K mismatch fails loudly, and the serve-side buffer's early-episode padding is
     bit-identical to the train loader's.
  7. JIT: the whole thing compiles.

Skips cleanly at module scope if jax/flax is not importable.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

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

from wsm_settings import ROBOCASA_OPENPI_SRC, ROBOCASA_ROOT, ROBOSUITE_ROOT

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))
try:
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.wsm_current_cond import (
        WSMGatedDeltaNetConditioner,
        WSMTanhConditioner,
    )
except Exception as e:  # noqa: BLE001
    _skip(f"OpenPI workspace conditioners not importable ({e})")

from vla_training.eval.serve_pi_05_wsm_cfg import (
    WSM_COND_SUBTREE,
    classify_wsm_cond_subtree,
)

W_DIM, COND_DIM, B, K = 512, 1024, 3, 4


def _mk(window_len=K, num_heads=2, head_dim=32, gate_init=1e-3, seed=0):
    return WSMGatedDeltaNetConditioner(
        w_dim=W_DIM,
        cond_dim=COND_DIM,
        window_len=window_len,
        num_heads=num_heads,
        head_dim=head_dim,
        gate_init=gate_init,
        rngs=nnx.Rngs(seed),
    )


def _window(seed=1, window_len=K, batch=B):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((batch, window_len, W_DIM)), dtype=jnp.float32)


def _flat(module):
    return {"/".join(str(k) for k in path): v.value for path, v in nnx.state(module, nnx.Param).flat_state()}


def test_default_cond_type_leaves_the_tanh_tree_byte_identical():
    assert Pi0Config(pi05=True).wsm_cond_type == "tanh"
    assert Pi0Config(pi05=True, wsm_tanh=True).wsm_cond_type == "tanh"

    a = WSMTanhConditioner(w_dim=W_DIM, cond_dim=COND_DIM, gate_init=1e-3, rngs=nnx.Rngs(7))
    b = WSMTanhConditioner(w_dim=W_DIM, cond_dim=COND_DIM, gate_init=1e-3, rngs=nnx.Rngs(7))
    flat_a, flat_b = _flat(a), _flat(b)
    assert (
        set(flat_a)
        == set(flat_b)
        == {
            "proj_t_in/kernel",
            "proj_t_in/bias",
            "proj_t_out/kernel",
            "proj_t_out/bias",
            "alpha",
        }
    )
    for name in flat_a:
        assert jnp.array_equal(flat_a[name], flat_b[name]), name
    omega = _window(2)[:, -1, :]
    assert jnp.array_equal(a(omega), b(omega))

    # gated_deltanet is only meaningful behind the tanh interface flag.
    with pytest.raises(ValueError, match="only meaningful with wsm_tanh"):
        Pi0Config(pi05=True, wsm_cond_type="gated_deltanet")
    with pytest.raises(ValueError, match="wsm_cond_type must be"):
        Pi0Config(pi05=True, wsm_tanh=True, wsm_cond_type="deltanet")
    print("✓ cond_type defaults to tanh; the tanh conditioner's param tree is unchanged")


def test_deltanet_subtree_name_and_distinguishing_leaves():
    config = Pi0Config(pi05=True, wsm_tanh=True, wsm_cond_type="gated_deltanet", wsm_cond_window=K)
    # The checkpoint subtree name is the Pi0 ATTRIBUTE name, which stays wsm_tanh_cond for both
    # variants (that is what keeps the interface map and missing_regex backfill untouched).
    import inspect

    from openpi.models.pi0 import Pi0

    source = inspect.getsource(Pi0.__init__)
    assert "self.wsm_tanh_cond = WSMGatedDeltaNetConditioner(" in source
    assert "wsm_deltanet_cond" not in source
    assert config.wsm_cond_type == "gated_deltanet"

    names = set(_flat(_mk()))
    assert {
        "proj_q/kernel",
        "proj_k/kernel",
        "proj_v/kernel",
        "proj_beta/kernel",
        "proj_decay/kernel",
        "proj_readout/kernel",
        "pos_decay_bias",
        "alpha",
    } <= names
    assert not any(name.startswith(("proj_t_in", "proj_t_out")) for name in names)
    assert _flat(_mk())["pos_decay_bias"].shape == (K, 2)
    print(f"✓ deltanet fills {WSM_COND_SUBTREE} with leaves no tanh checkpoint has")


def test_pi0_param_tree_registers_under_wsm_tanh_cond():
    """The abstract Pi0 tree (what BaseModelConfig.load matches against) keeps the s1 subtree name."""

    def subtree(cond_type, window):
        config = Pi0Config(
            pi05=True,
            wsm_tanh=True,
            wsm_cond_type=cond_type,
            wsm_cond_window=window,
            wsm_cond_head_dim=32,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
        )
        model = nnx.eval_shape(config.create, jax.random.key(0))
        pure = nnx.split(model)[1].to_pure_dict()
        assert "wsm_tanh_cond" in pure, sorted(pure)
        assert "wsm_deltanet_cond" not in pure
        return {
            "/".join(str(k) for k in path) for path, _ in nnx.to_flat_state(nnx.state(model.wsm_tanh_cond, nnx.Param))
        }

    tanh_leaves = subtree("tanh", 1)
    delta_leaves = subtree("gated_deltanet", 4)
    assert tanh_leaves == {
        "proj_t_in/kernel",
        "proj_t_in/bias",
        "proj_t_out/kernel",
        "proj_t_out/bias",
        "alpha",
    }
    assert "pos_decay_bias" in delta_leaves and not (tanh_leaves & delta_leaves - {"alpha"})
    print("✓ both variants materialize inside the Pi0 tree under wsm_tanh_cond")


def test_parameter_budget_stays_within_2x_of_tanh():
    tanh = WSMTanhConditioner(w_dim=W_DIM, cond_dim=COND_DIM, rngs=nnx.Rngs(3))
    deltanet = _mk(window_len=8, num_heads=2, head_dim=256)
    n_tanh = sum(int(v.size) for v in _flat(tanh).values())
    n_delta = sum(int(v.size) for v in _flat(deltanet).values())
    assert n_delta <= 2 * n_tanh, (n_delta, n_tanh)
    print(f"✓ deltanet {n_delta} params vs tanh {n_tanh} ({n_delta / n_tanh:.2f}x)")


def test_near_identity_at_init_and_exact_bypass():
    cond = _mk(gate_init=1e-3, seed=4)
    window = _window(5)
    assert jnp.allclose(cond.alpha.value, 1e-3)
    out = cond(window)
    assert bool(jnp.isfinite(out).all())
    assert float(jnp.max(jnp.abs(out))) > 0.0
    # The gate, not a zero-init readout, is what makes step 0 the S0 policy: the readout kernel is
    # normally initialized, and the output is that projection scaled by tanh(1e-3) ~ 1e-3.
    assert float(jnp.linalg.norm(cond.proj_readout.kernel.value)) > 0.0
    cond.alpha.value = jnp.full_like(cond.alpha.value, jnp.arctanh(1.0 - 1e-7))
    ungated = float(jnp.max(jnp.abs(cond(window))))
    cond.alpha.value = jnp.full_like(cond.alpha.value, 1e-3)
    assert abs(float(jnp.max(jnp.abs(cond(window)))) / ungated - float(jnp.tanh(1e-3))) < 1e-6

    cond.alpha.value = jnp.zeros_like(cond.alpha.value)
    assert float(jnp.max(jnp.abs(cond(window)))) == 0.0

    with pytest.raises(ValueError, match="finite"):
        _mk(gate_init=float("nan"))
    print("✓ deltanet is tanh(alpha=1e-3)-gated at init and alpha=0 is an exact additive no-op")


def test_gradient_reaches_every_parameter_at_init():
    cond = _mk(window_len=K, seed=6)
    window = _window(7)
    graphdef, params = nnx.split(cond, nnx.Param)

    def loss_fn(p):
        return jnp.sum(nnx.merge(graphdef, p)(window) ** 2) + jnp.sum(nnx.merge(graphdef, p)(window))

    grads = jax.grad(loss_fn)(params)
    flat = {"/".join(str(k) for k in path): v.value for path, v in nnx.to_flat_state(grads)}
    dead = sorted(name for name, g in flat.items() if float(jnp.linalg.norm(g)) == 0.0)
    assert not dead, f"dead gradient at init for {dead}"
    print(f"✓ all {len(flat)} deltanet parameter tensors have a live gradient at init")


def test_recurrence_full_forget_reads_only_the_newest_token():
    cond = _mk(window_len=K, seed=8)
    # Force a huge decay logit => gamma = exp(-softplus(large)) == 0 => S_i = beta_i v_i k_i^T.
    cond.proj_decay.kernel.value = jnp.zeros_like(cond.proj_decay.kernel.value)
    cond.proj_decay.bias.value = jnp.full_like(cond.proj_decay.bias.value, 60.0)
    window = _window(9)
    clobbered = window.at[:, :-1, :].set(1000.0)
    assert jnp.allclose(cond(window), cond(clobbered), atol=0.0, rtol=0.0)

    # beta == 0 (sigmoid(-large)) => the state never leaves zero => the read is the gated readout of 0.
    cond2 = _mk(window_len=K, seed=8)
    cond2.proj_beta.kernel.value = jnp.zeros_like(cond2.proj_beta.kernel.value)
    cond2.proj_beta.bias.value = jnp.full_like(cond2.proj_beta.bias.value, -60.0)
    zero_read = cond2.proj_readout(jnp.zeros((B, cond2.num_heads * cond2.head_dim), jnp.float32))
    expected = jnp.tanh(cond2.alpha.value) * zero_read
    assert jnp.allclose(cond2(window), expected, atol=1e-6)
    print("✓ full-forget decay reads only omega_t; beta=0 reads an exactly zero state")


def test_window_length_mismatch_fails_loudly_and_jits():
    cond = _mk(window_len=K, seed=10)
    with pytest.raises(ValueError, match="trained window_len"):
        cond(_window(11, window_len=K + 1))

    graphdef, state = nnx.split(cond)

    @jax.jit
    def run(state, window):
        return nnx.merge(graphdef, state)(window)

    window = _window(12)
    out = run(state, window)
    assert out.shape == (B, COND_DIM)
    assert jnp.allclose(out, cond(window), atol=1e-6)
    print("✓ K mismatch raises; the scan-based recurrence jit-compiles")


# ---------------------------------------------------------------------------------------------
# Serve-side auto-detection (pure function over synthetic param trees)
# ---------------------------------------------------------------------------------------------
def _tanh_tree():
    return {
        "proj_t_in/kernel": (512, 1024),
        "proj_t_in/bias": (1024,),
        "proj_t_out/kernel": (1024, 1024),
        "proj_t_out/bias": (1024,),
        "alpha": (1024,),
    }


def _deltanet_tree(window=8, heads=2, head_dim=256):
    inner = heads * head_dim
    return {
        "proj_q/kernel": (512, inner),
        "proj_q/bias": (inner,),
        "proj_k/kernel": (512, inner),
        "proj_k/bias": (inner,),
        "proj_v/kernel": (512, inner),
        "proj_v/bias": (inner,),
        "proj_beta/kernel": (512, heads),
        "proj_beta/bias": (heads,),
        "proj_decay/kernel": (512, heads),
        "proj_decay/bias": (heads,),
        "pos_decay_bias": (window, heads),
        "proj_readout/kernel": (inner, 1024),
        "proj_readout/bias": (1024,),
        "alpha": (1024,),
    }


def test_serve_autodetect_on_synthetic_trees():
    tanh = classify_wsm_cond_subtree(_tanh_tree())
    assert (tanh.cond_type, tanh.window) == ("tanh", 1)

    delta = classify_wsm_cond_subtree(_deltanet_tree(window=8))
    assert (delta.cond_type, delta.window, delta.num_heads, delta.head_dim) == ("gated_deltanet", 8, 2, 256)
    assert classify_wsm_cond_subtree(_deltanet_tree(window=3, heads=1, head_dim=128)).window == 3

    # Explicit override must agree with the tree.
    assert classify_wsm_cond_subtree(_tanh_tree(), requested="tanh").cond_type == "tanh"
    with pytest.raises(RuntimeError, match="Refusing to serve a mismatched conditioner"):
        classify_wsm_cond_subtree(_tanh_tree(), requested="gated_deltanet")
    with pytest.raises(RuntimeError, match="Refusing to serve a mismatched conditioner"):
        classify_wsm_cond_subtree(_deltanet_tree(), requested="tanh")

    # Ambiguous / broken trees never fall back to a guess.
    with pytest.raises(RuntimeError, match="mixes tanh and gated_deltanet"):
        classify_wsm_cond_subtree({**_tanh_tree(), **_deltanet_tree()})
    with pytest.raises(RuntimeError, match="no known conditioner"):
        classify_wsm_cond_subtree({"alpha": (1024,)})
    with pytest.raises(RuntimeError, match="missing"):
        classify_wsm_cond_subtree({k: v for k, v in _deltanet_tree().items() if not k.startswith("proj_v")})
    with pytest.raises(RuntimeError, match="no wsm_tanh_cond subtree"):
        classify_wsm_cond_subtree({})
    print("✓ serve auto-detect resolves both trees and fails loudly on every ambiguity")


def test_detected_geometry_reconstructs_the_trained_module():
    """The detected spec must rebuild a module whose param tree matches the trained one exactly."""
    trained = _mk(window_len=6, num_heads=2, head_dim=64, seed=13)
    shapes = {name: tuple(v.shape) for name, v in _flat(trained).items()}
    spec = classify_wsm_cond_subtree(shapes)
    rebuilt = WSMGatedDeltaNetConditioner(
        w_dim=W_DIM,
        cond_dim=COND_DIM,
        window_len=spec.window,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        rngs=nnx.Rngs(99),
    )
    assert {n: tuple(v.shape) for n, v in _flat(rebuilt).items()} == shapes
    print("✓ the checkpoint-detected geometry rebuilds an identically shaped conditioner")


# ---------------------------------------------------------------------------------------------
# PTRM (H9): the deltanet tree PLUS a recursive core. Serve must recover the recursion DEPTH from
# `step_bias` exactly as it recovers the window from `pos_decay_bias`, and it must never quietly
# serve such a checkpoint as its deltanet parent (that would drop the whole recursion).
# ---------------------------------------------------------------------------------------------
def _ptrm_tree(window=8, heads=2, head_dim=256, steps=4, cond_dim=COND_DIM):
    return {
        **_deltanet_tree(window=window, heads=heads, head_dim=head_dim),
        "z0": (cond_dim,),
        "core_in/kernel": (cond_dim, 2 * cond_dim),
        "core_in/bias": (2 * cond_dim,),
        "core_out/kernel": (2 * cond_dim, cond_dim),
        "core_out/bias": (cond_dim,),
        "step_bias": (steps, cond_dim),
        "rms/scale": (cond_dim,),
        "proj_out/kernel": (cond_dim, cond_dim),
        "proj_out/bias": (cond_dim,),
        "q_head/kernel": (cond_dim, 1),
        "q_head/bias": (1,),
    }


def test_serve_autodetects_the_ptrm_recursion_depth():
    from vla_training.eval.serve_pi_05_wsm_cfg import _PTRM_LEAF_PREFIXES

    spec = classify_wsm_cond_subtree(_ptrm_tree(window=8, steps=4))
    assert (spec.cond_type, spec.window, spec.num_heads, spec.head_dim, spec.ptrm_steps) == (
        "gated_deltanet_ptrm",
        8,
        2,
        256,
        4,
    )
    assert classify_wsm_cond_subtree(_ptrm_tree(steps=7)).ptrm_steps == 7
    assert "steps=7" in classify_wsm_cond_subtree(_ptrm_tree(steps=7)).describe()

    # A plain deltanet checkpoint must NOT acquire a depth, and must still report its own type.
    plain = classify_wsm_cond_subtree(_deltanet_tree(window=8))
    assert plain.cond_type == "gated_deltanet" and plain.ptrm_steps is None

    # Explicit overrides must agree with the tree, in both directions.
    assert classify_wsm_cond_subtree(_ptrm_tree(), requested="gated_deltanet_ptrm").cond_type == "gated_deltanet_ptrm"
    with pytest.raises(RuntimeError, match="Refusing to serve a mismatched conditioner"):
        classify_wsm_cond_subtree(_ptrm_tree(), requested="gated_deltanet")
    with pytest.raises(RuntimeError, match="Refusing to serve a mismatched conditioner"):
        classify_wsm_cond_subtree(_deltanet_tree(), requested="gated_deltanet_ptrm")

    # A HALF-present core is a corrupt PTRM tree, never a deltanet checkpoint with extras.
    for dropped in _PTRM_LEAF_PREFIXES:
        truncated = {name: shape for name, shape in _ptrm_tree().items() if not name.split("/", 1)[0] == dropped}
        with pytest.raises(RuntimeError, match="PTRM .* is missing"):
            classify_wsm_cond_subtree(truncated)
    with pytest.raises(RuntimeError, match=r"step_bias must be \[steps, cond_dim\]"):
        classify_wsm_cond_subtree({**_ptrm_tree(), "step_bias": (4,)})
    print("✓ serve recovers the PTRM depth structurally and refuses every partial tree")


def test_ptrm_eval_knobs_are_parsed_from_the_environment_and_fail_loud():
    from vla_training.eval.serve_pi_05_wsm_cfg import PTRMEvalKnobs, ptrm_eval_knobs_from_env

    default = ptrm_eval_knobs_from_env({})
    assert default == PTRMEvalKnobs(k=1, sigma=0.0, select="q")
    assert default.deterministic  # unset env == the honest PTRM-off control

    swept = ptrm_eval_knobs_from_env(
        {"WSM_PTRM_EVAL_K": "32", "WSM_PTRM_EVAL_SIGMA": "0.3", "WSM_PTRM_EVAL_SELECT": "random"}
    )
    assert swept == PTRMEvalKnobs(k=32, sigma=0.3, select="random")
    assert not swept.deterministic

    # Garbage must fail HERE, not silently fall back to the default and mislabel the eval cell.
    for env in (
        {"WSM_PTRM_EVAL_K": "many"},
        {"WSM_PTRM_EVAL_K": "0"},
        {"WSM_PTRM_EVAL_SIGMA": "loud"},
        {"WSM_PTRM_EVAL_SIGMA": "-0.1"},
        {"WSM_PTRM_EVAL_SIGMA": "nan"},
        {"WSM_PTRM_EVAL_SELECT": "argmax"},
    ):
        with pytest.raises(RuntimeError, match="WSM_PTRM_EVAL_"):
            ptrm_eval_knobs_from_env(env)
    print("✓ PTRM eval knobs are parsed before any model is built and reject garbage")


def test_serve_threads_the_ptrm_knobs_through_the_same_config_path_as_the_geometry():
    """The from_pretrained-bypass lesson: the knobs must reach Pi0Config, not a post-build object."""
    import inspect

    from vla_training.eval import serve_pi_05_wsm_cfg as serve

    source = inspect.getsource(serve.build_stage_s_workspace_policy)
    # Everything PTRM-related enters through cond_kwargs, which is splatted into Pi0Config(...).
    assert "cond_kwargs.update(" in source
    assert '"wsm_ptrm_eval_k": knobs.k' in source
    assert '"wsm_ptrm_eval_sigma": knobs.sigma' in source
    assert '"wsm_ptrm_eval_select": knobs.select' in source
    assert '"wsm_ptrm_steps": cond_spec.ptrm_steps' in source
    assert "**cond_kwargs," in source
    # ...and nothing sets a ptrm attribute on the built policy/model afterwards.
    assert "policy._model.wsm_ptrm" not in source

    main_source = inspect.getsource(serve.main)
    assert "ptrm_eval_knobs_from_env()" in main_source
    # The windowed conditioners share one serve-window branch, so PTRM cannot be served at K=1.
    assert "cond_spec.cond_type in WSM_WINDOWED_COND_TYPES" in main_source
    assert serve.WSM_WINDOWED_COND_TYPES == ("gated_deltanet", "gated_deltanet_ptrm")
    print("✓ PTRM eval knobs ride the Pi0Config path the deltanet geometry already uses")


# ---------------------------------------------------------------------------------------------
# Train-loader <-> serve-buffer window padding parity
# ---------------------------------------------------------------------------------------------
def _train_loader_window(frame_indices, t, k):
    """Run the loader's OWN _wsm_causal_window source (importing the module needs robocasa)."""
    import ast

    source = (ROBOCASA_OPENPI_SRC / "openpi" / "groot_utils" / "groot_openpi_dataset.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_wsm_causal_window")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<loader>", "exec"), namespace)  # noqa: S102
    return np.asarray(namespace["_wsm_causal_window"](np.asarray(frame_indices), int(t), int(k)))


def _serve_buffer_window(frames, k, stride=8):
    """What WSMEvalConditioner.step_many builds after `frames` grid steps of one episode."""
    from workspace_models.features.wsm_align import causal_window_indices

    frame_indices = np.arange(frames, dtype=np.int64) * stride
    return np.asarray(causal_window_indices(frame_indices, int(frame_indices[-1]), int(k)))


def test_episode_start_padding_parity_between_loader_and_serve_buffer():
    k, stride = 8, 8
    for frames in range(1, 12):
        serve = _serve_buffer_window(frames, k, stride)
        # The train loader at the frame of the newest grid row of the same partial episode.
        grid = np.arange(frames, dtype=np.int64) * stride
        train = _train_loader_window(grid, int(grid[-1]), k)
        assert serve.tolist() == train.tolist(), (frames, serve, train)
        assert len(serve) == k
        if frames < k:  # left-pad by REPEATING the oldest available grid row (never zeros)
            assert serve[: k - frames].tolist() == [0] * (k - frames)
            assert serve[k - frames :].tolist() == list(range(frames))
        assert serve[-1] == frames - 1, "newest slot is always the current grid row"
    print("✓ serve buffer and train loader agree bit-for-bit, including early-episode left padding")


_BUILD_PROBE = r"""
import json, os, sys, yaml
sys.path.insert(0, {wsmv2!r})
sys.path.insert(0, {openpi!r})
from utils.config_schema import TrainConfigView
from vla_training.train.train_base._pi05_common import build_train_config

class _Soup:
    soup = []
    pi05_weights = None
    balancing_enabled = False

cfg = TrainConfigView.from_yaml(yaml.safe_load(open(sys.argv[1])))
model = build_train_config(cfg, _Soup()).model
print(json.dumps({{
    "cond_type": model.wsm_cond_type,
    "cond_window": model.wsm_cond_window,
    "heads": model.wsm_cond_num_heads,
    "head_dim": model.wsm_cond_head_dim,
    "k_window": model.wsm_k_window,
    "env_k_window": os.environ["WSM_K_WINDOW"],
    "history_dropout": model.wsm_cond_history_dropout,
    "env_history_dropout": os.environ.get("WSM_COND_HISTORY_DROPOUT"),
    "ptrm_steps": model.wsm_ptrm_steps,
    "ptrm_q_weight": model.wsm_ptrm_q_weight,
    "ptrm_eval": [model.wsm_ptrm_eval_k, model.wsm_ptrm_eval_sigma, model.wsm_ptrm_eval_select],
    "tanh": model.wsm_tanh,
}}))
"""


def _build_probe(config_name):
    """Build the openpi TrainConfig for one recipe in a FRESH process (the loader freezes
    WSM_K_WINDOW at import, so import order is part of the contract under test)."""
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    openpi = str(ROBOCASA_OPENPI_SRC)
    env = {
        **os.environ,
        "WSM_POLICY_FEATS_ROOT": "/tmp/wsm-feats-probe",
        "WSM_TANH": "1",
        "WSM_K_WINDOW": "1",  # what submit_pi_stage_s.py pins for every workspace arm
        "PYTHONPATH": ":".join(
            [
                str(ROBOCASA_ROOT),
                str(ROBOSUITE_ROOT),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _BUILD_PROBE.format(wsmv2=str(root), openpi=openpi),
            str(root / "scripts/configs/train" / config_name),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        if "No module named 'robocasa'" in result.stderr or "No module named 'robosuite'" in result.stderr:
            pytest.skip("robocasa/robosuite not importable in this env — loader build probe skipped")
        raise AssertionError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_recipe_drives_cond_type_and_the_loader_window():
    baseline = _build_probe("pi05_workspace_finetune.yaml")
    assert baseline["tanh"] is True
    assert baseline["cond_type"] == "tanh"
    # The shipped S1 arm is untouched: the launcher's WSM_K_WINDOW=1 still wins end to end.
    assert baseline["k_window"] == 1 and baseline["env_k_window"] == "1"

    deltanet = _build_probe("pi05_stage_s1_deltanet_finetune.yaml")
    assert deltanet["tanh"] is True, "the deltanet arm is still interface tanh (subtree wsm_tanh_cond)"
    assert deltanet["cond_type"] == "gated_deltanet"
    assert deltanet["cond_window"] == 8
    assert (deltanet["heads"], deltanet["head_dim"]) == (2, 256)
    # The recipe, not the launcher env, is authoritative for the deltanet window, and it reaches the
    # dataloader (which froze WSM_K_WINDOW at import) as well as the conditioner.
    assert deltanet["k_window"] == 8 and deltanet["env_k_window"] == "8"
    print("✓ cond_type/cond_window flow from the recipe into both the model and the dataloader")


def test_history_dropout_recipe_reaches_the_model_and_leaves_parents_at_zero():
    """The three causal-confusion arms must resolve to 0.5; every parent arm stays at 0.0."""
    for recipe, window in (
        ("pi05_rmb_deltanet_drop_finetune.yaml", 8),
        ("pi05_rmb_deltanet_w16_drop_finetune.yaml", 16),
        ("pi05_rmb_deltanet_w32_drop_finetune.yaml", 32),
    ):
        built = _build_probe(recipe)
        assert built["cond_type"] == "gated_deltanet"
        assert built["cond_window"] == window and built["k_window"] == window
        assert built["history_dropout"] == 0.5, recipe
        assert built["env_history_dropout"] == "0.5", recipe
    for parent in (
        "pi05_rmb_deltanet_finetune.yaml",
        "pi05_rmb_deltanet_w16_finetune.yaml",
        "pi05_rmb_deltanet_w32_finetune.yaml",
        "pi05_stage_s1_deltanet_finetune.yaml",
    ):
        assert _build_probe(parent)["history_dropout"] == 0.0, parent
    assert _build_probe("pi05_workspace_finetune.yaml")["history_dropout"] == 0.0


def test_ptrm_recipe_reaches_the_model_and_leaves_every_parent_arm_at_the_defaults():
    """The PTRM arm resolves to the recursive conditioner at depth 4; parents never do."""
    built = _build_probe("pi05_norm_s1_ptrm_finetune.yaml")
    assert built["tanh"] is True, "PTRM is still interface tanh (subtree wsm_tanh_cond)"
    assert built["cond_type"] == "gated_deltanet_ptrm"
    # It inherits the deltanet w=8 geometry exactly, loader window included.
    assert built["cond_window"] == 8 and built["k_window"] == 8 and built["env_k_window"] == "8"
    assert (built["heads"], built["head_dim"]) == (2, 256)
    assert built["ptrm_steps"] == 4 and built["ptrm_q_weight"] == 0.1
    # Eval knobs are NOT a train-time recipe knob: the trained checkpoint always carries the
    # deterministic control, and serve is what sweeps K/sigma/selection.
    assert built["ptrm_eval"] == [1, 0.0, "q"]
    # No confounders rode along.
    assert built["history_dropout"] == 0.0

    parent = _build_probe("pi05_norm_s1_deltanet_finetune.yaml")
    assert parent["cond_type"] == "gated_deltanet"
    assert parent["cond_window"] == 8 and parent["k_window"] == 8
    # The ptrm fields exist on every config but are inert off the arm, so no parent moved.
    assert parent["ptrm_steps"] == 4 and parent["ptrm_q_weight"] == 0.1
    print("✓ the PTRM recipe reaches the model; the deltanet parent is unchanged")
    print("✓ cond_history_dropout=0.5 reaches the model on the drop arms and 0.0 on every parent")


def test_episode_reset_restarts_the_window():
    """wsm_t == 0 drops the per-env buffer, so a fresh episode re-pads from one frame."""
    k = 8
    assert _serve_buffer_window(1, k).tolist() == [0] * k
    assert _serve_buffer_window(5, k).tolist() == [0, 0, 0, 0, 1, 2, 3, 4]
    import inspect

    from vla_training.eval._groot_wsm_eval import WSMEvalConditioner

    reset_source = inspect.getsource(WSMEvalConditioner.reset)
    assert "self._fused: list" in reset_source and "self._conds: list" in reset_source
    print("✓ episode reset clears the causal buffer; the window re-pads from the first grid frame")


# ---------------------------------------------------------------------------------------------
# History-intervention dropout (causal-confusion wave). Train-only deletion of HISTORICAL window
# elements; the newest element is never deleted; nothing is ever deleted at inference.
# ---------------------------------------------------------------------------------------------
def _copy_projections(src, dst):
    """Share every learned tensor between two conditioners of different window_len."""
    for name in ("proj_q", "proj_k", "proj_v", "proj_beta", "proj_decay", "proj_readout"):
        s, d = getattr(src, name), getattr(dst, name)
        d.kernel.value = s.kernel.value
        d.bias.value = s.bias.value
    dst.alpha.value = src.alpha.value
    # pos_decay_bias is position-indexed, so the two windows can only be compared with it zeroed.
    src.pos_decay_bias.value = jnp.zeros_like(src.pos_decay_bias.value)
    dst.pos_decay_bias.value = jnp.zeros_like(dst.pos_decay_bias.value)


def test_history_dropout_defaults_off_and_is_byte_identical():
    """The knob at its default must not perturb ANY existing arm — the study depends on this."""
    assert _mk().history_dropout == 0.0
    assert (
        Pi0Config(pi05=True, wsm_tanh=True, wsm_cond_type="gated_deltanet", wsm_cond_window=K).wsm_cond_history_dropout
        == 0.0
    )

    baseline = _mk(window_len=K, seed=31)
    knobbed = WSMGatedDeltaNetConditioner(
        w_dim=W_DIM,
        cond_dim=COND_DIM,
        window_len=K,
        num_heads=2,
        head_dim=32,
        history_dropout=0.0,
        rngs=nnx.Rngs(31),
    )
    assert set(_flat(baseline)) == set(_flat(knobbed)), "the knob must add no parameters"
    window = _window(32)
    ref = baseline(window)
    # Same value from the pre-knob call signature, from an explicit train=False, and — critically —
    # from a TRAINING step: at 0.0 there is no masked branch at all.
    assert jnp.array_equal(ref, knobbed(window))
    assert jnp.array_equal(ref, knobbed(window, train=False))
    assert jnp.array_equal(ref, knobbed(window, train=True))
    assert jnp.array_equal(ref, knobbed(window, train=True, rng=jax.random.key(0)))
    print("✓ history_dropout=0.0 adds no params and is bitwise identical to the pre-knob module")


def test_history_dropout_mask_never_deletes_the_newest_element():
    cond = _mk(window_len=8, seed=33)
    cond.history_dropout = 0.9
    keeps = [
        np.asarray(cond._history_keep_mask(jax.random.key(seed), (64,), 8))  # noqa: SLF001
        for seed in range(24)
    ]
    stacked = np.stack(keeps)
    assert stacked[..., -1].all(), "the current-timestep element was deleted"
    historical_keep_rate = float(stacked[..., :-1].mean())
    assert 0.05 < historical_keep_rate < 0.16, historical_keep_rate  # ~1 - 0.9
    # Fresh masks every call: two different keys must not give the same mask.
    assert not np.array_equal(keeps[0], keeps[1])
    # Deterministic under a fixed key.
    assert np.array_equal(
        np.asarray(cond._history_keep_mask(jax.random.key(7), (64,), 8)),  # noqa: SLF001
        np.asarray(cond._history_keep_mask(jax.random.key(7), (64,), 8)),  # noqa: SLF001
    )
    # A window of one is all-newest, so the intervention is a structural no-op there.
    assert np.asarray(cond._history_keep_mask(jax.random.key(1), (4,), 1)).all()  # noqa: SLF001
    print(f"✓ newest slot always kept; historical keep rate {historical_keep_rate:.3f} at p=0.9")


def test_deleted_elements_are_dropped_exactly_as_if_absent(monkeypatch):
    """A masked K=4 pass must equal an UNMASKED pass over just the surviving elements."""
    wide = _mk(window_len=4, num_heads=2, head_dim=32, seed=21)
    wide.history_dropout = 0.5
    narrow = _mk(window_len=2, num_heads=2, head_dim=32, seed=21)
    _copy_projections(wide, narrow)

    keep = jnp.asarray([[False, True, False, True]])  # index 3 == newest, always kept
    monkeypatch.setattr(
        WSMGatedDeltaNetConditioner,
        "_history_keep_mask",
        lambda self, rng, lead, k_len: jnp.broadcast_to(keep, (*lead, k_len)),
    )
    window = _window(22, window_len=4, batch=1)
    masked = wide(window, train=True, rng=jax.random.key(0))
    surviving = narrow(window[:, jnp.asarray([1, 3]), :])
    assert jnp.allclose(masked, surviving, atol=1e-5), float(jnp.max(jnp.abs(masked - surviving)))
    # ...and it is NOT simply the unmasked read (the intervention actually changes the recurrence).
    assert not jnp.allclose(masked, wide(window), atol=1e-4)
    print("✓ a deleted element leaves the recurrence exactly as if it had never been in the window")


def test_history_dropout_is_train_only_and_fails_loud_off_train():
    cond = _mk(window_len=K, seed=35)
    cond.history_dropout = 0.9
    window = _window(36)
    clean = _mk(window_len=K, seed=35)(window)  # same seed => same params, knob off
    # Eval/serve: no masking, ever.
    assert jnp.array_equal(cond(window), clean)
    assert jnp.array_equal(cond(window, train=False), clean)
    # A dropout key on a non-training call is a hard error, not a silent serve-time intervention.
    with pytest.raises(ValueError, match="train-only"):
        cond(window, train=False, rng=jax.random.key(0))
    # Training with the knob on but no key is also a hard error (never a silently-unmasked step).
    with pytest.raises(ValueError, match="requires an rng"):
        cond(window, train=True)
    # A masked training step differs from the clean read.
    assert not jnp.allclose(cond(window, train=True, rng=jax.random.key(2)), clean, atol=1e-4)
    print("✓ masking is train-only; eval is bit-identical and a serve-side key fails loudly")


def test_history_dropout_config_validation():
    for bad in (-0.1, 0.95, 1.0):
        with pytest.raises(ValueError, match=r"wsm_cond_history_dropout must be in"):
            Pi0Config(
                pi05=True,
                wsm_tanh=True,
                wsm_cond_type="gated_deltanet",
                wsm_cond_window=K,
                wsm_cond_history_dropout=bad,
            )
    # Meaningless on the plain tanh read (there is no window to intervene on).
    with pytest.raises(ValueError, match="gated-DeltaNet omega window"):
        Pi0Config(pi05=True, wsm_tanh=True, wsm_cond_history_dropout=0.5)
    ok = Pi0Config(
        pi05=True, wsm_tanh=True, wsm_cond_type="gated_deltanet", wsm_cond_window=8, wsm_cond_history_dropout=0.5
    )
    assert ok.wsm_cond_history_dropout == 0.5
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"history_dropout must be in"):
            WSMGatedDeltaNetConditioner(
                w_dim=W_DIM,
                cond_dim=COND_DIM,
                window_len=K,
                num_heads=2,
                head_dim=32,
                history_dropout=bad,
                rngs=nnx.Rngs(0),
            )
    print("✓ [0.0, 0.9] range enforced and the knob is refused outside the deltanet read")


def test_pi0_wiring_never_masks_outside_training():
    """Exercise Pi0's own methods (unbound, on a stub) — the eval path must not mask."""
    from openpi.models.pi0 import Pi0

    cond = _mk(window_len=K, seed=37)
    cond.history_dropout = 0.9
    clean = _mk(window_len=K, seed=37)
    window = _window(38)

    class _Stub:
        wsm_cfg2 = False
        wsm_tanh = True
        wsm_cond_type = "gated_deltanet"
        wsm_cond_history_dropout = 0.9
        wsm_tanh_cond = cond
        # The stub borrows Pi0's OWN gate predicates, so this exercises the shipped logic.
        _history_dropout_active = Pi0._history_dropout_active
        _ptrm_active = Pi0._ptrm_active

    stub, obs = _Stub(), SimpleNamespace(wsm_w_window=window)
    assert Pi0._history_dropout_active(stub, train=False) is False
    assert Pi0._history_dropout_active(stub, train=True) is True

    off = SimpleNamespace(
        wsm_cfg2=False, wsm_tanh=True, wsm_cond_type="gated_deltanet", wsm_cond_history_dropout=0.0, wsm_tanh_cond=cond
    )
    assert Pi0._history_dropout_active(off, train=True) is False
    plain = SimpleNamespace(
        wsm_cfg2=False, wsm_tanh=True, wsm_cond_type="tanh", wsm_cond_history_dropout=0.5, wsm_tanh_cond=cond
    )
    assert Pi0._history_dropout_active(plain, train=True) is False

    # sample_actions' call shape (train=False, no hist key) reproduces the clean read exactly.
    served = Pi0._current_workspace_vec(stub, obs, train=False, rng=None, force_uncond=False)
    assert jnp.array_equal(served, clean(window))
    # compute_loss' call shape masks.
    trained = Pi0._current_workspace_vec(
        stub, obs, train=True, rng=None, force_uncond=False, hist_rng=jax.random.key(3)
    )
    assert not jnp.allclose(trained, served, atol=1e-4)
    # A dropout key handed to an inactive path is refused rather than ignored.
    with pytest.raises(ValueError, match="intervention is inactive"):
        Pi0._current_workspace_vec(stub, obs, train=False, rng=None, force_uncond=False, hist_rng=jax.random.key(3))
    print("✓ Pi0 masks only on a training step of a knob-on deltanet arm")


def test_masked_recurrence_jits():
    cond = _mk(window_len=K, seed=39)
    cond.history_dropout = 0.5
    graphdef, state = nnx.split(cond)

    @jax.jit
    def run(state, window, key):
        return nnx.merge(graphdef, state)(window, train=True, rng=key)

    window = _window(40)
    out = run(state, window, jax.random.key(5))
    assert out.shape == (B, COND_DIM) and bool(jnp.isfinite(out).all())
    assert jnp.allclose(out, cond(window, train=True, rng=jax.random.key(5)), atol=1e-6)
    print("✓ the masked scan jit-compiles and matches the eager result")
