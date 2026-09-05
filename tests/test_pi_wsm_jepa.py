"""Packet-08 S3 (pi JEPA+SigReg) load-bearing tests — the six invariants from the packet.

1  Gradients: the aux produces nonzero grads to the head AND the backbone; ZERO gradient reaches
   the omega target (it is data — stop-gradient inside the loss).
2  Parameter accounting: the S3 tree == the S0 tree + the `wsm_jepa_head` subtree, exactly.
3  Terminal-target masking parity with the groot prototype (all-invalid => sigreg-only, graph-connected).
4  No inference dependency: compute_loss(train=False) and the loss path with identical shared
   weights match S0 exactly; the train=True difference is ONE scalar broadcast over [B, H].
5  Global-batch SigReg: the Epps-Pulley statistic is NOT per-shard decomposable (averaging shard
   values != the global value), and jit == eager; under the fork's single-process jit/GSPMD
   training the in-loss reduction IS the global statistic.
6  RNG discipline: JEPA draws only from a fold-in domain — the flow-matching part of the loss is
   bit-identical to S0 under the same rng (pinned by test 4's constant-offset structure).

Added 2026-07-31 after the s3 collapse (internal_planning_and_todos/jul_31/s3_collapse_forensics.md):

7  SCALE INVARIANCE: the SigReg statistic (and therefore the aux loss at fixed lambda) is invariant
   to doubling B and to doubling action_horizon. The shipped port returned the LeJEPA Alg.-1
   n-scaled statistic, so lambda_sigreg=0.05 acted as 0.05*n = ~20 on [B*H, D] per-token features.
8  BUDGET: at lambda_sigreg=0.05 the SigReg contribution is BELOW a typical step-0 flow-matching
   loss (0.168, measured on S1/S2/A6), not ~200x above it.

Added 2026-07-31 with the VISReg arm (arXiv:2606.02572, internal_planning_and_todos/jul_31/visreg_arm.md):

9   VISReg is sample-count invariant too (2xB, 2xH, 4x rows) — the s3 bug class must not reappear
    in the replacement regularizer.
10  BUDGET at the yaml weight (0.05) and yaml slice count (128), same shape and same flow reference
    as test 8.
11  DETERMINISM: bit-identical under the same rng, different under a different rng, jit == eager.
12  REGRESSION: the default (sigreg) path is bit-unchanged — identical to the pre-VISReg formula and
    completely insensitive to the visreg knobs.
13  COLLAPSE GRADIENT (the paper's core claim, verified numerically): on a degenerate z the VISReg
    gradient is O(1) while the Epps-Pulley gradient is ~0 — and EXACTLY 0 at z == 0.

Added 2026-08-01 with the prediction-horizon knob (`wsm_jepa_num_futures` = k, grid offsets +1..+k):

14  k-INVARIANT LAMBDA: replicating one future k times leaves the aux unchanged — the JEPA term is a
    masked MEAN over (sample, future), and the regularizer never sees the target at all.
15  PER-FUTURE MASKING + the shape guard: padded future slots contribute EXACTLY zero, all-invalid is
    still a graph-connected zero, and a loader/model k mismatch raises instead of broadcasting.
16  k > 1 is train-time only: the aux is still one scalar over [B, H] and sample_actions is
    byte-identical to S0. (Test 2 additionally pins that k only WIDENS proj_out — no new subtree, and
    unchanged param shapes at the default k == 1, so existing checkpoints load.)

Run: PYTHONPATH=. ~/Research/envs/openpi-jax-latest/bin/python -m pytest -q \
     tests/test_pi_wsm_jepa.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsm_settings import ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx", reason="flax nnx required")

from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models.wsm_jepa import (  # noqa: E402
    WSMJepaHead,
    sigreg_epps_pulley,
    visreg_loss,
    wsm_jepa_aux_loss,
)
from openpi.shared import array_typing as at  # noqa: E402

B, H, DW = 3, 10, 512


@pytest.fixture(autouse=True)
def _release_jax_memory():
    yield
    jax.clear_caches()


def _cfg(jepa: bool, regularizer: str = "sigreg", num_futures: int = 1) -> Pi0Config:
    return Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=H,
        max_token_len=48,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        wsm_jepa=jepa,
        wsm_jepa_w_dim=DW,
        wsm_jepa_regularizer=regularizer,
        wsm_jepa_num_futures=num_futures,
    )


def _obs_act(cfg, *, targets: bool, seed=5):
    obs_spec, act_spec = cfg.inputs_spec(batch_size=B)
    kx = jax.random.split(jax.random.key(seed), 64)
    idx = [0]

    def rnd(spec):
        idx[0] += 1
        k = kx[idx[0] % 64]
        if spec.dtype == jnp.float32:
            return jax.random.normal(k, spec.shape, spec.dtype)
        if spec.dtype == bool:
            return jnp.ones(spec.shape, bool)
        return jnp.zeros(spec.shape, spec.dtype)

    obs = jax.tree.map(rnd, obs_spec)
    obs = dataclasses.replace(
        obs,
        tokenized_prompt=jnp.ones((B, cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, cfg.max_token_len), bool),
    )
    if targets:
        # k == 1 keeps the legacy [B, Dw] / [B] shapes; k > 1 adds the future axis on both.
        k = cfg.wsm_jepa_num_futures
        flags = jnp.asarray([True, True, False])
        obs = dataclasses.replace(
            obs,
            wsm_w_target=jax.random.normal(jax.random.key(seed + 1), (B, DW) if k == 1 else (B, k, DW)),
            wsm_w_target_valid=flags if k == 1 else jnp.broadcast_to(flags[:, None], (B, k)),
        )
    act = jax.random.normal(jax.random.key(seed + 2), act_spec.shape, act_spec.dtype)
    return obs, act


# ------------------------- 1: gradient routing (head+backbone yes, target no) -------------------------


def test_01_aux_grads_reach_head_and_backbone_never_the_target():
    head = WSMJepaHead(16, DW, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    w_tgt = jax.random.normal(jax.random.key(2), (B, DW))
    valid = jnp.asarray([True, True, True])
    graphdef, state = nnx.split(head)

    def loss_wrt_all(st, pen, tgt):
        return wsm_jepa_aux_loss(nnx.merge(graphdef, st), pen, tgt, valid, jax.random.key(3))

    g_head, g_pen, g_tgt = jax.grad(loss_wrt_all, argnums=(0, 1, 2))(state, penult, w_tgt)
    head_leaves = [np.asarray(x) for x in jax.tree.leaves(nnx.state(nnx.merge(graphdef, g_head)))]
    assert any(np.abs(leaf).max() > 0 for leaf in head_leaves), "aux must train the head"
    assert float(jnp.abs(g_pen).max()) > 0, "aux must backprop into the penultimate (the backbone)"
    assert float(jnp.abs(g_tgt).max()) == 0.0, "the omega target is DATA — zero gradient (stop-grad)"


# ------------------------- 2: parameter accounting -------------------------


def test_02_s3_param_tree_is_s0_plus_wsm_jepa_head_exactly():
    with at.disable_typechecking():
        m0 = _cfg(jepa=False).create(jax.random.key(0))
        m3 = _cfg(jepa=True).create(jax.random.key(0))
    p0 = {".".join(str(x) for x in k) for k, _ in nnx.to_flat_state(nnx.state(m0))}
    p3 = {".".join(str(x) for x in k) for k, _ in nnx.to_flat_state(nnx.state(m3))}
    extra = sorted(p3 - p0)
    assert p0 <= p3, f"S3 must contain the full S0 tree; missing {sorted(p0 - p3)[:5]}"
    assert extra and all("wsm_jepa_head" in path for path in extra), (
        f"S3 extras must be exactly the wsm_jepa_head subtree, got {extra[:8]}"
    )
    # A longer prediction horizon must not add a subtree either — it only WIDENS proj_out, so the
    # extra paths are the same set and only the proj_out kernel's last axis grows (k * w_dim). At
    # k == 1 both head params keep their shipped shapes, which is what lets old ckpts load.
    with at.disable_typechecking():
        mk = _cfg(jepa=True, num_futures=4).create(jax.random.key(0))
    flat3 = {".".join(str(x) for x in k): v for k, v in nnx.to_flat_state(nnx.state(m3))}
    flatk = {".".join(str(x) for x in k): v for k, v in nnx.to_flat_state(nnx.state(mk))}
    assert set(flatk) == p3, "k > 1 must not introduce new parameters, only reshape proj_out"
    assert flat3["wsm_jepa_head.proj_out.kernel"][...].shape == (DW, DW)
    assert flatk["wsm_jepa_head.proj_out.kernel"][...].shape == (DW, 4 * DW)
    assert flat3["wsm_jepa_head.proj_in.kernel"][...].shape == flatk["wsm_jepa_head.proj_in.kernel"][...].shape, (
        "the trunk is SHARED across futures"
    )


# ------------------------- 3: terminal-target masking -------------------------


def test_03_terminal_masking_matches_prototype_semantics():
    head = WSMJepaHead(16, DW, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    w_tgt = jax.random.normal(jax.random.key(2), (B, DW))
    rng = jax.random.key(3)
    none_valid = jnp.zeros((B,), bool)
    all_valid = jnp.ones((B,), bool)
    # All-invalid: the JEPA term is a graph-connected zero => the loss is EXACTLY the sigreg term.
    sig_only = float(
        sigreg_epps_pulley(
            penult.reshape(-1, 16).astype(jnp.float32),
            jax.random.fold_in(rng, 0x57534A50),
        )
    )
    total_none = float(wsm_jepa_aux_loss(head, penult, w_tgt, none_valid, rng, sigreg_weight=0.05))
    assert total_none == pytest.approx(0.05 * sig_only, rel=1e-6)
    # Mixed: masking row 2 must equal computing over rows {0,1} only.
    mixed = jnp.asarray([True, True, False])
    t_mixed = float(wsm_jepa_aux_loss(head, penult, w_tgt, mixed, rng, sigreg_weight=0.0))
    t_sub = float(wsm_jepa_aux_loss(head, penult[:2], w_tgt[:2], all_valid[:2], rng, sigreg_weight=0.0))
    assert t_mixed == pytest.approx(t_sub, rel=1e-6)
    # And the all-invalid JEPA zero still lets gradients flow via sigreg (graph-connected).
    graphdef, state = nnx.split(head)

    def f(pen):
        return wsm_jepa_aux_loss(nnx.merge(graphdef, state), pen, w_tgt, none_valid, rng)

    assert float(jnp.abs(jax.grad(f)(penult)).max()) > 0


# ------------------------- 4 + 6: no inference dependency; scalar-offset structure -------------------------


def test_04_s3_loss_is_s0_plus_one_scalar_and_eval_path_is_identical():
    with at.disable_typechecking():
        m0 = _cfg(jepa=False).create(jax.random.key(0))
        m3 = _cfg(jepa=True).create(jax.random.key(0))
        nnx.update(m3, nnx.state(m0))  # share every S0 parameter; head params stay as initialized
        cfg = _cfg(jepa=True)
        obs_t, act = _obs_act(cfg, targets=True)
        obs_p, _ = _obs_act(cfg, targets=False)

        rng = jax.random.key(11)
        l0 = np.asarray(m0.compute_loss(rng, obs_p, act, train=True))
        l3 = np.asarray(m3.compute_loss(rng, obs_t, act, train=True))
        diff = l3 - l0
        # One scalar broadcast over [B, H]: the flow part is bit-identical (rng discipline — JEPA
        # only folds a domain tag), the aux is a single added constant.
        assert diff.shape == l0.shape
        assert float(np.ptp(diff)) < 1e-5, f"aux must be one scalar over [B,H]; spread={np.ptp(diff)}"
        assert float(diff.mean()) > 0.0, "the aux (1-cos at init + sigreg) should be positive here"

        # train=False: the aux is OFF and the losses match exactly even with targets present.
        l0e = np.asarray(m0.compute_loss(rng, obs_p, act, train=False))
        l3e = np.asarray(m3.compute_loss(rng, obs_t, act, train=False))
        assert np.array_equal(l0e, l3e), "eval-path loss must be byte-identical to S0"

        # And enabling jepa without targets on a TRAIN batch fails loud (never silently trains S0).
        with pytest.raises(ValueError, match="wsm_w_target"):
            m3.compute_loss(rng, obs_p, act, train=True)


def test_04b_sample_actions_identical_to_s0_with_shared_weights():
    with at.disable_typechecking():
        m0 = _cfg(jepa=False).create(jax.random.key(0))
        m3 = _cfg(jepa=True).create(jax.random.key(0))
        nnx.update(m3, nnx.state(m0))
        cfg = _cfg(jepa=True)
        obs, _ = _obs_act(cfg, targets=False)  # inference carries NO wsm fields
        a0 = np.asarray(m0.sample_actions(jax.random.key(7), obs, num_steps=2))
        a3 = np.asarray(m3.sample_actions(jax.random.key(7), obs, num_steps=2))
    assert np.array_equal(a0, a3), "S3 inference must be byte-identical to S0 (D5)"


# ------------------------- 5: SigReg is a global-batch statistic -------------------------


def test_05_sigreg_not_shard_decomposable_and_jit_matches_eager():
    rng = jax.random.key(9)
    x = jax.random.normal(jax.random.key(10), (64, 32)) * 1.7 + 0.3  # deliberately non-N(0,I)
    full = float(sigreg_epps_pulley(x, rng))
    half_mean = 0.5 * (float(sigreg_epps_pulley(x[:32], rng)) + float(sigreg_epps_pulley(x[32:], rng)))
    # Per-shard averaging is NOT the same statistic — this is why the loss must see the global
    # batch (the fork's single-process jit/GSPMD reduction provides exactly that).
    assert abs(full - half_mean) > 1e-4
    jitted = float(jax.jit(lambda a, r: sigreg_epps_pulley(a, r))(x, rng))
    assert jitted == pytest.approx(full, rel=1e-6)
    # Sanity: a genuinely standard-normal batch scores much lower than the shifted/scaled one.
    z = jax.random.normal(jax.random.key(12), (4096, 32))
    assert float(sigreg_epps_pulley(z, rng)) < full


# ------------------------- 7: SigReg scale invariance (the s3 collapse) -------------------------

# Step-0 flow-matching loss of the three identical-recipe healthy peers (S1 0.1678, S2 0.1676,
# A6 0.1434) — the budget any aux term has to live inside. s3 logged 33.73.
FLOW_STEP0 = 0.168
# Per-slice cap of the normalized Epps-Pulley integrand for features arbitrarily far from N(0,1):
# re, im -> 0 leaves \int phi_N^3 dt = sqrt(2*pi/3).
SIGREG_CAP = float(np.sqrt(2 * np.pi / 3))  # 1.4472


def _far_from_normal(shape, seed):
    """Features deliberately far from N(0,I) — the regime where SigReg carries real signal.

    Invariance is a statement about the H1 (non-normal) regime: there the normalized statistic
    estimates the population CF discrepancy, an n-free quantity. On an exactly-N(0,I) batch the
    statistic is pure MC noise and IS O(1/n) by construction — that is the n-scaling the LeJEPA
    chi-square asymptotics exist for, and precisely why the raw statistic must not be used as a loss.
    """
    return jax.random.normal(jax.random.key(seed), shape) * 1.7 + 0.3


def test_07_sigreg_statistic_invariant_to_doubling_b_and_h():
    rng = jax.random.key(21)
    d = 32
    b, h = 8, 50  # the trained shape: per-device batch 8, action_horizon 50 => n = 400
    pool = _far_from_normal((b, 2 * h, d), 0)  # same distribution everywhere
    base = pool[:, :h]
    double_h = pool  # 2H at fixed B
    double_b = jnp.concatenate([base, _far_from_normal((b, h, d), 1)], axis=0)  # 2B at fixed H

    def stat(z):
        return float(sigreg_epps_pulley(z.reshape(-1, d).astype(jnp.float32), rng))

    s_base, s_h, s_b = stat(base), stat(double_h), stat(double_b)
    assert s_base == pytest.approx(s_h, rel=0.1), f"H-doubling changed the statistic: {s_base} -> {s_h}"
    assert s_base == pytest.approx(s_b, rel=0.1), f"B-doubling changed the statistic: {s_base} -> {s_b}"
    # 4x the rows (2B x 2H), still the same number (the pre-fix code returned 4x the value).
    assert s_base == pytest.approx(stat(_far_from_normal((2 * b, 2 * h, d), 6)), rel=0.1)
    # Hard bound independent of n: the normalized statistic cannot exceed the phi_N^3 integral.
    assert 0.0 < s_base < SIGREG_CAP
    extreme = float(sigreg_epps_pulley(jax.random.normal(jax.random.key(2), (b * h, d)) * 8.0, rng))
    assert extreme < SIGREG_CAP, f"statistic must be bounded by {SIGREG_CAP}, got {extreme}"

    # Same invariance at the LOSS level, which is what lambda_sigreg multiplies.
    head = WSMJepaHead(d, DW, rngs=nnx.Rngs(0))

    def aux(z):
        n = z.shape[0]
        tgt = jax.random.normal(jax.random.key(3), (n, DW))
        return float(wsm_jepa_aux_loss(head, z, tgt, jnp.ones((n,), bool), rng, jepa_weight=0.0, sigreg_weight=0.05))

    assert aux(base) == pytest.approx(aux(double_h), rel=0.1)
    assert aux(base) == pytest.approx(aux(double_b), rel=0.1)


# ------------------------- 8: the aux fits inside the flow-loss budget -------------------------


def test_08_aux_magnitude_at_init_is_same_order_as_the_flow_loss():
    rng = jax.random.key(22)
    # The shape the s3 run actually fed the loss: under the fork's single-process jit/GSPMD the
    # reshape sees the GLOBAL batch (64), not the per-device 8 — so n was 64*50 = 3200 and
    # lambda_sigreg=0.05 acted as ~160. d = pi05 action-expert width.
    b, h, d = 64, 50, 1024
    penult = _far_from_normal((b, h, d), 4)
    head = WSMJepaHead(d, DW, rngs=nnx.Rngs(0))
    tgt = jax.random.normal(jax.random.key(5), (b, DW))
    valid = jnp.ones((b,), bool)

    sig = float(sigreg_epps_pulley(penult.reshape(-1, d).astype(jnp.float32), jax.random.fold_in(rng, 0x57534A50)))
    sigreg_term = 0.05 * sig
    total = float(wsm_jepa_aux_loss(head, penult, tgt, valid, rng, jepa_weight=1.0, sigreg_weight=0.05))
    jepa_term = total - sigreg_term

    # THE regression assertion: at lambda_sigreg=0.05 the SigReg contribution is below one step-0
    # flow loss. Pre-fix (statistic x n) the same batch gives >100x the flow loss — the s3 run
    # logged aux ~33.6 against a flow term of 0.168.
    assert sigreg_term / FLOW_STEP0 < 1.0, f"sigreg/flow = {sigreg_term / FLOW_STEP0:.1f} (want < 1)"
    pre_fix = 0.05 * (sig * b * h)  # what the shipped port computed
    assert pre_fix / FLOW_STEP0 > 100.0, f"pre-fix scale was {pre_fix / FLOW_STEP0:.0f}x flow"

    # The whole aux stays the same ORDER as the flow loss. The (1 - cos) term is ~1.0 at a
    # random-init predictor and lambda_jepa is 1.0 BY DESIGN (packet 08), so the total sits a few x
    # above 0.168 and decays as cos rises — that is the intended aux, not the 33.73 that was logged.
    assert jepa_term == pytest.approx(1.0, abs=0.5), f"1-cos at init should be ~1, got {jepa_term}"
    assert total / FLOW_STEP0 < 10.0, f"aux/flow = {total / FLOW_STEP0:.1f} — aux must not dominate"
    assert total < 1.0 * 2.0 + 0.05 * SIGREG_CAP, "aux exceeds its analytic ceiling"


# ======================= VISReg (arXiv:2606.02572) — tests 9-13 =======================

# The yaml knobs of scripts/configs/train/pi05_stage_s3_visreg_finetune.yaml. Kept here as the
# single source the budget test asserts against, exactly as test 8 does for lambda_sigreg.
VISREG_WEIGHT = 0.05
VISREG_SLICES = 128


def _skewed(shape, seed):
    """Non-Gaussian MARGINALS (lognormal), i.e. the H1 regime for the shape term.

    VISReg's shape term matches the 1-D marginal of the STANDARDIZED features against standard-
    Gaussian quantiles, so a rescaled/shifted Gaussian (test 7's `_far_from_normal`) is standardized
    straight back onto the target and leaves L_shape ~ 0. That would make the invariance assertion
    vacuous for the term whose n-scaling is at issue (a sorted-vs-quantiles comparison is the one
    piece that could plausibly grow with n — Eq. 5 in the paper literally does). A skewed marginal
    keeps L_shape O(1), so the assertion has teeth.
    """
    g = jax.random.normal(jax.random.key(seed), shape)
    return jnp.exp(0.7 * g) * 1.3 - 0.4


def test_09_visreg_invariant_to_doubling_b_and_h():
    rng = jax.random.key(31)
    d = 32
    b, h = 8, 50  # the trained shape; n = B*H
    base = _skewed((b, h, d), 0)
    double_h = _skewed((b, 2 * h, d), 0)
    double_b = _skewed((2 * b, h, d), 0)
    quad = _skewed((2 * b, 2 * h, d), 0)

    def stat(z):
        return float(visreg_loss(z.reshape(-1, d).astype(jnp.float32), rng, num_slices=VISREG_SLICES))

    s_base = stat(base)
    assert s_base > 0.5, f"the fixture must be far from N(0,I) for this to be a real test: {s_base}"
    assert s_base == pytest.approx(stat(double_h), rel=0.1), "H-doubling changed VISReg"
    assert s_base == pytest.approx(stat(double_b), rel=0.1), "B-doubling changed VISReg"
    assert s_base == pytest.approx(stat(quad), rel=0.1), "4x rows changed VISReg"
    # An exactly-N(0,I) batch scores ~0 — VISReg is measuring the right target distribution.
    iso = jax.random.normal(jax.random.key(33), (4096, d))
    assert float(visreg_loss(iso, rng, num_slices=VISREG_SLICES)) < 0.01

    # Same invariance at the LOSS level, which is what visreg_weight multiplies.
    head = WSMJepaHead(d, DW, rngs=nnx.Rngs(0))

    def aux(z):
        n = z.shape[0]
        tgt = jax.random.normal(jax.random.key(3), (n, DW))
        return float(
            wsm_jepa_aux_loss(
                head,
                z,
                tgt,
                jnp.ones((n,), bool),
                rng,
                jepa_weight=0.0,
                regularizer="visreg",
                visreg_weight=VISREG_WEIGHT,
                visreg_num_slices=VISREG_SLICES,
            )
        )

    assert aux(base) == pytest.approx(aux(double_h), rel=0.1)
    assert aux(base) == pytest.approx(aux(double_b), rel=0.1)


def test_10_visreg_budget_at_the_yaml_weight():
    """Mirror of test 8 at the VISReg yaml weight: the aux must fit inside the flow-loss budget."""
    rng = jax.random.key(32)
    b, h, d = 64, 50, 1024  # the shape the loss actually sees (global batch under jit/GSPMD)
    penult = _far_from_normal((b, h, d), 4)
    head = WSMJepaHead(d, DW, rngs=nnx.Rngs(0))
    tgt = jax.random.normal(jax.random.key(5), (b, DW))
    valid = jnp.ones((b,), bool)

    vis = float(
        visreg_loss(
            penult.reshape(-1, d).astype(jnp.float32),
            jax.random.fold_in(rng, 0x56495352),
            num_slices=VISREG_SLICES,
        )
    )
    reg_term = VISREG_WEIGHT * vis
    total = float(
        wsm_jepa_aux_loss(
            head,
            penult,
            tgt,
            valid,
            rng,
            jepa_weight=1.0,
            regularizer="visreg",
            visreg_weight=VISREG_WEIGHT,
            visreg_num_slices=VISREG_SLICES,
        )
    )
    assert reg_term / FLOW_STEP0 < 1.0, f"visreg/flow = {reg_term / FLOW_STEP0:.2f} (want < 1)"
    assert total - reg_term == pytest.approx(1.0, abs=0.5), "1-cos at init should be ~1"
    assert total / FLOW_STEP0 < 10.0, f"aux/flow = {total / FLOW_STEP0:.1f} — aux must not dominate"
    # The n-scaled reading of the paper's Eq. 5 (sum over rows instead of mean) is the s3 bug class:
    # it would be >1000x the flow loss at this shape. Pinned so nobody "restores fidelity" to it.
    assert VISREG_WEIGHT * (vis * b * h) / FLOW_STEP0 > 100.0


def test_11_visreg_is_deterministic_in_the_rng_and_jit_safe():
    x = _skewed((400, 32), 7).astype(jnp.float32)
    a = float(visreg_loss(x, jax.random.key(1), num_slices=VISREG_SLICES))
    b = float(visreg_loss(x, jax.random.key(1), num_slices=VISREG_SLICES))
    c = float(visreg_loss(x, jax.random.key(2), num_slices=VISREG_SLICES))
    assert a == b, "same rng must give a bit-identical value (the slice sketch is its only randomness)"
    assert a != c, "a different rng must draw different slices"
    # ... but only by MC noise: the two keys estimate the SAME population quantity.
    assert a == pytest.approx(c, rel=0.05)
    jitted = float(jax.jit(lambda z, r: visreg_loss(z, r, num_slices=VISREG_SLICES))(x, jax.random.key(1)))
    assert jitted == pytest.approx(a, rel=1e-6)
    # The aux threads the rng through a fold-in, so it is deterministic in the STEP rng too.
    head = WSMJepaHead(32, DW, rngs=nnx.Rngs(0))
    z = x.reshape(8, 50, 32)
    tgt = jax.random.normal(jax.random.key(4), (8, DW))
    valid = jnp.ones((8,), bool)

    def call(rng):
        return float(wsm_jepa_aux_loss(head, z, tgt, valid, rng, regularizer="visreg"))

    assert call(jax.random.key(6)) == call(jax.random.key(6))
    assert call(jax.random.key(6)) != call(jax.random.key(7))


def test_12_default_regularizer_is_sigreg_and_that_path_is_bit_unchanged():
    head = WSMJepaHead(16, DW, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    w_tgt = jax.random.normal(jax.random.key(2), (B, DW))
    valid = jnp.asarray([True, True, False])
    rng = jax.random.key(3)

    default = float(wsm_jepa_aux_loss(head, penult, w_tgt, valid, rng))
    explicit = float(wsm_jepa_aux_loss(head, penult, w_tgt, valid, rng, regularizer="sigreg"))
    assert default == explicit, "the default must still be sigreg"
    # Bit-identical to the pre-VISReg formula, recomputed from its parts.
    sig = float(sigreg_epps_pulley(penult.reshape(-1, 16).astype(jnp.float32), jax.random.fold_in(rng, 0x57534A50)))
    jepa_only = float(wsm_jepa_aux_loss(head, penult, w_tgt, valid, rng, sigreg_weight=0.0))
    assert default == pytest.approx(jepa_only + 0.05 * sig, rel=1e-6)
    # The visreg knobs are dead code on the sigreg branch — absurd values change nothing.
    assert default == float(
        wsm_jepa_aux_loss(
            head, penult, w_tgt, valid, rng, visreg_weight=1e6, visreg_num_slices=4, visreg_scale_weight=1e3
        )
    )
    with pytest.raises(ValueError, match="regularizer"):
        wsm_jepa_aux_loss(head, penult, w_tgt, valid, rng, regularizer="vicreg")

    # And the config surface: default Pi0Config is sigreg, the flag is validated, and selecting
    # visreg actually changes the trained loss (i.e. it is really plumbed through pi0.py).
    assert Pi0Config().wsm_jepa_regularizer == "sigreg"
    with pytest.raises(ValueError, match="wsm_jepa_regularizer"):
        _cfg(jepa=True, regularizer="vicreg")
    with at.disable_typechecking():
        m_sig = _cfg(jepa=True).create(jax.random.key(0))
        m_vis = _cfg(jepa=True, regularizer="visreg").create(jax.random.key(0))
        nnx.update(m_vis, nnx.state(m_sig))
        obs, act = _obs_act(_cfg(jepa=True), targets=True)
        k = jax.random.key(11)
        l_sig = np.asarray(m_sig.compute_loss(k, obs, act, train=True))
        l_vis = np.asarray(m_vis.compute_loss(k, obs, act, train=True))
        assert not np.array_equal(l_sig, l_vis), "visreg selection must reach the loss"
        # Still exactly one scalar over [B, H] — the flow part is untouched by the swap.
        assert float(np.ptp(l_vis - l_sig)) < 1e-5
        # And inference is still byte-identical to S0.
        obs_p, _ = _obs_act(_cfg(jepa=True), targets=False)
        assert np.array_equal(
            np.asarray(m_sig.sample_actions(jax.random.key(7), obs_p, num_steps=2)),
            np.asarray(m_vis.sample_actions(jax.random.key(7), obs_p, num_steps=2)),
        )


def test_13_visreg_gradient_survives_collapse_where_sigreg_vanishes():
    """The paper's core claim, verified numerically on this port.

    "the gradient of the Epps-Pulley test diminishes as the embedding collapses (Figure 2),
    eventually vanishing entirely — precisely when a strong corrective signal is needed most."
    Degenerate z = every row identical: the batch carries zero information and the regularizer is the
    only thing that can push it apart.
    """
    rng = jax.random.key(34)
    n, d = 400, 32
    c = jax.random.normal(jax.random.key(9), (d,))

    def grads(z):
        gv = jax.grad(lambda x: visreg_loss(x, rng, num_slices=64))(z)
        gs = jax.grad(lambda x: sigreg_epps_pulley(x, rng))(z)
        return float(jnp.linalg.norm(gv)), float(jnp.linalg.norm(gs)), gv

    # (a) total collapse AT the origin: Epps-Pulley's gradient is EXACTLY zero (re == 1 == phi_N and
    # im == 0 at every t, so the integrand sits at a stationary point), VISReg's is O(1).
    z0 = jnp.zeros((n, d))
    gv0, gs0, gvec0 = grads(z0)
    assert gs0 == 0.0, "sigreg gradient at z==0 should be exactly zero (the vanishing claim)"
    assert gv0 > 1e-3, f"visreg gradient must survive total collapse, got {gv0}"
    assert bool(jnp.isfinite(gvec0).all()), "no NaN at zero variance (the eps-in-sqrt choice)"

    # (b) collapse AWAY from the origin, at several magnitudes: sigreg is damped by its exp(-t^2/2)
    # weight, visreg is not.
    for scale in (1.0, 3.0):
        z = jnp.broadcast_to(c * scale, (n, d))
        gv, gs, gvec = grads(z)
        assert bool(jnp.isfinite(gvec).all())
        assert gv > 1e-3, f"visreg gradient vanished at collapse scale {scale}: {gv}"
        assert gv > 50.0 * gs, f"scale {scale}: visreg |g|={gv:.2e} not >> sigreg |g|={gs:.2e}"
        # The LOSS also reports the collapse loudly (sigreg saturates at sqrt(2*pi/3) = 1.447).
        assert float(visreg_loss(z, rng, num_slices=64)) > 1.0

    # (c) near-collapse (tiny but nonzero variance) — the regime a real run actually enters.
    z = jnp.broadcast_to(c * 3.0, (n, d)) + 1e-3 * jax.random.normal(jax.random.key(11), (n, d))
    gv, gs, _ = grads(z)
    assert gv > 50.0 * gs, f"near-collapse: visreg |g|={gv:.2e} vs sigreg |g|={gs:.2e}"


# ======================= multi-future horizon (wsm_jepa_num_futures = k) — tests 14-16 =======================


class _TiledHead:
    """A head whose k output slots are IDENTICAL, so the only thing test 14 measures is the loss's
    normalization over the future axis (a real head's k slots differ by construction)."""

    def __init__(self, inner: WSMJepaHead, num_futures: int):
        self.inner, self.num_futures = inner, num_futures

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = self.inner(x)
        return jnp.broadcast_to(out[..., None, :], (*out.shape[:-1], self.num_futures, out.shape[-1]))


def test_14_jepa_term_is_the_masked_mean_over_futures_so_lambda_is_k_invariant():
    """Replicate one future k times: the aux must be UNCHANGED. The JEPA term is a masked MEAN over
    (sample, future), not a sum — a k-scaled aux would silently multiply lambda_jepa by k, which is
    the s3 collapse's bug class transplanted onto the horizon knob."""
    inner = WSMJepaHead(16, DW, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    tgt = jax.random.normal(jax.random.key(2), (B, DW))
    valid = jnp.asarray([True, True, False])
    rng = jax.random.key(3)

    one = float(wsm_jepa_aux_loss(inner, penult, tgt, valid, rng, sigreg_weight=0.05))
    for k in (2, 4, 16):
        rep = float(
            wsm_jepa_aux_loss(
                _TiledHead(inner, k),
                penult,
                jnp.broadcast_to(tgt[:, None], (B, k, DW)),
                jnp.broadcast_to(valid[:, None], (B, k)),
                rng,
                sigreg_weight=0.05,
                num_futures=k,
            )
        )
        assert rep == pytest.approx(one, rel=1e-6), f"k={k} changed the aux scale: {one} -> {rep}"
    # The regularizer input is the per-token penultimate and never sees the target, so the sigreg
    # term is k-free by construction — pinned here so a future refactor cannot route targets into it.
    sig = float(sigreg_epps_pulley(penult.reshape(-1, 16).astype(jnp.float32), jax.random.fold_in(rng, 0x57534A50)))
    jepa_only = float(
        wsm_jepa_aux_loss(
            _TiledHead(inner, 4),
            penult,
            jnp.broadcast_to(tgt[:, None], (B, 4, DW)),
            jnp.broadcast_to(valid[:, None], (B, 4)),
            rng,
            sigreg_weight=0.0,
            num_futures=4,
        )
    )
    assert one == pytest.approx(jepa_only + 0.05 * sig, rel=1e-6)


def test_15_per_future_masking_and_the_shape_guard():
    """Distinct futures: masking a future slot must equal dropping it, and a k mismatch between the
    loader and the model must RAISE (the shapes would otherwise broadcast into a wrong loss)."""
    inner = WSMJepaHead(16, DW, rngs=nnx.Rngs(0))
    head4 = _TiledHead(inner, 4)
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    tgt = jax.random.normal(jax.random.key(2), (B, 4, DW))
    rng = jax.random.key(3)
    kw = {"sigreg_weight": 0.0, "num_futures": 4}

    # The padded tail (valid=False on the last futures) must contribute exactly zero: poisoning the
    # masked rows' CONTENT cannot move the loss.
    mask = jnp.asarray([[True, True, False, False]] * B)
    ref = float(wsm_jepa_aux_loss(head4, penult, tgt, mask, rng, **kw))
    poisoned = jnp.where(mask[..., None], tgt, 987.0)
    assert float(wsm_jepa_aux_loss(head4, penult, poisoned, mask, rng, **kw)) == ref
    # All-invalid stays a graph-connected zero at k > 1 as well (sigreg still trains the backbone).
    none_valid = jnp.zeros((B, 4), bool)
    assert float(wsm_jepa_aux_loss(head4, penult, tgt, none_valid, rng, **kw)) == 0.0
    graphdef, state = nnx.split(inner)

    def f(pen):
        return wsm_jepa_aux_loss(
            _TiledHead(nnx.merge(graphdef, state), 4),
            pen,
            tgt,
            none_valid,
            rng,
            num_futures=4,
            sigreg_weight=0.05,
        )

    assert float(jnp.abs(jax.grad(f)(penult)).max()) > 0

    # Shape guard, both directions.
    with pytest.raises(ValueError, match="num_futures"):
        wsm_jepa_aux_loss(head4, penult, tgt[:, 0], mask[:, 0], rng, **kw)
    with pytest.raises(ValueError, match="num_futures"):
        wsm_jepa_aux_loss(inner, penult, tgt, mask, rng, sigreg_weight=0.0)
    with pytest.raises(ValueError, match="wsm_jepa_num_futures"):
        _cfg(jepa=True, num_futures=0)
    assert Pi0Config().wsm_jepa_num_futures == 1, "the default horizon must stay 1"


def test_16_k_gt_1_trains_end_to_end_and_leaves_inference_identical():
    """The k > 1 config must run the full loss path (loader shapes -> head -> aux) and STILL be one
    scalar over [B, H], with sample_actions byte-identical to S0."""
    with at.disable_typechecking():
        m0 = _cfg(jepa=False).create(jax.random.key(0))
        mk = _cfg(jepa=True, num_futures=4).create(jax.random.key(0))
        nnx.update(mk, nnx.state(m0))
        cfg = _cfg(jepa=True, num_futures=4)
        obs_t, act = _obs_act(cfg, targets=True)
        obs_p, _ = _obs_act(cfg, targets=False)
        assert obs_t.wsm_w_target.shape == (B, 4, DW)
        assert obs_t.wsm_w_target_valid.shape == (B, 4)

        rng = jax.random.key(11)
        l0 = np.asarray(m0.compute_loss(rng, obs_p, act, train=True))
        lk = np.asarray(mk.compute_loss(rng, obs_t, act, train=True))
        assert float(np.ptp(lk - l0)) < 1e-5, "the aux is still ONE scalar over [B, H]"
        assert float((lk - l0).mean()) > 0.0
        assert np.array_equal(
            np.asarray(m0.sample_actions(jax.random.key(7), obs_p, num_steps=2)),
            np.asarray(mk.sample_actions(jax.random.key(7), obs_p, num_steps=2)),
        ), "k > 1 is train-time only; inference stays byte-identical to S0"
        # And a k mismatch (k=1 loader targets under a k=4 model) fails loud rather than broadcasting.
        obs_1, _ = _obs_act(_cfg(jepa=True), targets=True)
        with pytest.raises(ValueError, match="num_futures"):
            mk.compute_loss(rng, obs_1, act, train=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} PASSED")
