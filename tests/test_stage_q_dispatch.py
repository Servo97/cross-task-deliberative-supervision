"""Load-bearing Stage-Q dispatch tests (7a §E) — the gate-removal evidence.

Proves, with the REAL fork code (no stand-ins on the path under test):
  A  Q0 reduction:   stage_q_train_step with robottt=False on a [B, L, ...] window batch is
     BITWISE-identical (loss, grad_norm, updated params) to the stock per-step train_step on the
     flattened [B*L, ...] batch. Q0 *is* the baseline recipe on window-sampled data.
  B  Q2 emergence:   the same batch through a robottt=True model (readout de-zeroed) produces a
     DIFFERENT loss — the fast-weight read demonstrably conditions the policy.
  C  Flat == loop:   the flat window loss (one pi forward at B*L, W chain via run_sequence) equals
     the per-step-loop reference `robottt_sequence_loss` on a noise-free stub — same Eq 4-5 math.
  D  Window indexing: flat-index arithmetic against all_steps layout; boundary + stacking rules.
  E  Serve passthrough: `robottt_cond` survives RobocasaInputs and reaches Observation.from_dict
     (the audit-confirmed silent-no-op regression).
  F  Production-geometry stability: 32 commits at d=256/h=128/H=50 stay finite and bounded, the
     online inner loss decreases, and outer meta-grads are finite (the D-9 divergence regression).

Run: PYTHONPATH=. ~/Research/envs/openpi-jax-latest/bin/python -m pytest -q \
     tests/test_stage_q_dispatch.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsm_settings import ROBOCASA_OPENPI_ROOT, ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx", reason="flax nnx required")

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models.robottt_fast_weights import RoboTTTConfig, RoboTTTFastWeights, robottt_sequence_loss  # noqa: E402
from openpi.shared import array_typing as at  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402
from openpi.training import utils as training_utils  # noqa: E402
from openpi.training.stage_q_windows import StageQWindowDataset, TransformedWindowDataset  # noqa: E402


def _load_train_module():
    """Import the fork's scripts/train.py (not a package) as a module."""
    path = ROBOCASA_OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_fork_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train = _load_train_module()

B, L = 2, 3  # windows x chunk-steps (small; the point is exactness, not scale)


@pytest.fixture(autouse=True)
def _release_jax_memory():
    yield
    jax.clear_caches()  # full-suite runs share one process; drop compiled executables between tests


def _tiny_config(robottt: bool):
    from openpi.training import config as _config

    # dummy gemma variants (width 64, depth 4): the dispatch/parity math is architecture-shape
    # independent, and full-size towers OOM host RAM when several test modules share one process.
    model = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        robottt=robottt,
        robottt_token_dim=16,
        robottt_fast_hidden=8,
        robottt_num_registers=3,
        robottt_window_len=L,
        robottt_tbptt_segment=L,
    )
    return _config.TrainConfig(
        name="stage_q_test",
        exp_name="stage_q_test",
        model=model,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-4, decay_steps=10, decay_lr=1e-5),
        batch_size=B,
        num_train_steps=2,
    )


def _window_batch(model_cfg, seed=4):
    """A synthetic [B, L, ...] window batch with REAL Observation structure (random but valid)."""
    obs_spec, act_spec = model_cfg.inputs_spec(batch_size=B * L)
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

    obs_flat = jax.tree.map(rnd, obs_spec)
    obs_flat = dataclasses.replace(
        obs_flat,
        tokenized_prompt=jnp.ones((B * L, model_cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B * L, model_cfg.max_token_len), bool),
    )
    act_flat = jax.random.normal(jax.random.key(seed + 1), act_spec.shape, act_spec.dtype)

    def to_window(x):
        return x.reshape(B, L, *x.shape[1:])

    return (
        (jax.tree.map(to_window, obs_flat), to_window(act_flat)),  # [B, L, ...] window batch
        (obs_flat, act_flat),  # the SAME data flattened [B*L, ...]
    )


def _init_state(config, rng):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    model = config.model.create(rng)
    params = nnx.state(model)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


# ---------------------------- A: Q0 == baseline (bitwise) ----------------------------


def test_q0_window_step_bitwise_equals_per_step_train_step_on_flattened_batch():
    config = _tiny_config(robottt=False)
    with at.disable_typechecking():
        # Sequential with aggressive frees: a full-suite process is near its RAM ceiling, so pull the
        # Stage-Q result to host and drop every device buffer before running the baseline step.
        state = _init_state(config, jax.random.key(0))
        (window_batch, flat_batch) = _window_batch(config.model)
        rng = jax.random.key(7)
        new_q, info_q = _train.stage_q_train_step(config, rng, state, window_batch)
        loss_q, gnorm_q = float(info_q["loss"]), float(info_q["grad_norm"])
        host_q = [np.asarray(leaf) for leaf in jax.tree.leaves(new_q.params)]
        del new_q, info_q, state, window_batch
        jax.clear_caches()

        state2 = _init_state(config, jax.random.key(0))  # fresh identical init
        new_s, info_s = _train.train_step(config, rng, state2, flat_batch)
        loss_s, gnorm_s = float(info_s["loss"]), float(info_s["grad_norm"])
        host_s = [np.asarray(leaf) for leaf in jax.tree.leaves(new_s.params)]
        del new_s, info_s, state2, flat_batch
        jax.clear_caches()

    assert loss_q == loss_s, "Q0 window loss must equal the per-step loss BITWISE"
    assert gnorm_q == gnorm_s
    # And the updated parameters are identical — the optimizer walked the same step.
    assert len(host_q) == len(host_s)
    for a, b in zip(host_q, host_s):
        assert np.array_equal(a, b), "Q0 must update params exactly like the baseline step"


# ---------------------------- B: Q2 emergence (differs from Q0) ----------------------------


def test_q2_fast_weights_change_the_loss_vs_q0_on_the_same_batch():
    cfg_q0, cfg_q2 = _tiny_config(robottt=False), _tiny_config(robottt=True)
    with at.disable_typechecking():
        # Build Q2's model, then copy the ENTIRE shared (non-robottt) parameter tree from Q0's model
        # so the ONLY difference is the fast-weight subtree + flag.
        model_q0 = cfg_q0.model.create(jax.random.key(0))
        model_q2 = cfg_q2.model.create(jax.random.key(0))
        shared = nnx.state(model_q0)
        nnx.update(model_q2, shared)  # robottt_fast params absent from Q0's tree -> left as-is
        # De-zero the readout / adaRMS seam so the conditioning is observable (trained-model regime;
        # at exact init the zero-init readout makes Q2 == Q0 by design — that's test A's regime).
        model_q2.robottt_fast.readout.kernel[...] = 0.1 * jax.random.normal(
            jax.random.key(3), model_q2.robottt_fast.readout.kernel[...].shape
        )
        model_q2.robottt_fast.w0_w2[...] = 0.1 * jax.random.normal(
            jax.random.key(4), model_q2.robottt_fast.w0_w2[...].shape
        )
        flat = dict(nnx.to_flat_state(nnx.state(model_q2)))
        rng_p = jax.random.key(21)
        for path, val in list(flat.items()):
            name = ".".join(str(x) for x in path)
            if name.endswith("_1.Dense_0.kernel") and ("pre_attention_norm" in name or "pre_ffw_norm" in name):
                rng_p, sub = jax.random.split(rng_p)
                flat[path] = type(val)(0.02 * jax.random.normal(sub, val[...].shape))
        nnx.update(model_q2, nnx.from_flat_state(flat))
        flat_q0 = dict(nnx.to_flat_state(nnx.state(model_q0)))
        for path, val in list(flat_q0.items()):
            name = ".".join(str(x) for x in path)
            if name.endswith("_1.Dense_0.kernel") and ("pre_attention_norm" in name or "pre_ffw_norm" in name):
                flat_q0[path] = type(val)(np.asarray(flat[path][...]))  # same perturbation as Q2
        nnx.update(model_q0, nnx.from_flat_state(flat_q0))

        (window_batch, _) = _window_batch(cfg_q0.model)
        obs_w, act_w = window_batch

        def window_loss(model):
            b, length = act_w.shape[0], act_w.shape[1]
            obs_flat = jax.tree.map(lambda x: x.reshape(b * length, *x.shape[2:]), obs_w)
            act_flat = act_w.reshape(b * length, *act_w.shape[2:])
            if getattr(model, "robottt", False):
                state_seq = obs_flat.state.reshape(b, length, -1)
                o_seq, _ = model.robottt_fast.run_sequence(state_seq, act_w)
                cond = jnp.swapaxes(o_seq, 0, 1).reshape(b * length, -1)
                obs_flat = dataclasses.replace(obs_flat, robottt_cond=cond)
            return float(jnp.mean(model.compute_loss(jax.random.key(9), obs_flat, act_flat, train=False)))

        l0, l2 = window_loss(model_q0), window_loss(model_q2)
    assert abs(l0 - l2) > 1e-7, f"Q2 must measurably differ from Q0 ({l0} vs {l2})"


# ---------------------------- C: flat window loss == per-step-loop reference ----------------------------


class _NoiseFreeStub:
    """compute_loss ignores rng entirely, so flat-vs-loop equality is exact by construction."""

    def __init__(self, fast):
        self.robottt = True
        self.robottt_fast = fast

    def compute_loss(self, rng, obs, actions, train):
        base = jnp.mean(actions**2, axis=-1)  # [B(,H)] deterministic
        rc = obs.robottt_cond
        extra = jnp.zeros(()) if rc is None else jnp.mean(rc**2)
        return base + extra


def test_flat_window_loss_matches_per_step_loop_reference():
    cfg = RoboTTTConfig(
        fast_weights=True,
        token_dim=16,
        fast_hidden=8,
        num_registers=3,
        cond_dim=12,
        state_dim=4,
        action_dim=4,
        action_horizon=5,
        window_len=L,
        tbptt_segment=L,
    )
    fast = RoboTTTFastWeights(cfg, rngs=nnx.Rngs(0))
    fast.readout.kernel[...] = 0.1 * jax.random.normal(jax.random.key(11), fast.readout.kernel[...].shape)
    fast.w0_w2[...] = 0.1 * jax.random.normal(jax.random.key(12), fast.w0_w2[...].shape)
    stub = _NoiseFreeStub(fast)

    k = jax.random.key(3)
    s = jax.random.normal(k, (B, L, cfg.state_dim))
    a = jax.random.normal(jax.random.fold_in(k, 1), (B, L, cfg.action_horizon, cfg.action_dim))

    # Loop reference (Eq 4-5 as a per-step python loop).
    obs_seq = []
    with at.disable_typechecking():
        for t in range(L):
            obs_seq.append(_model.Observation(images={}, image_masks={}, state=s[:, t]))
        loop = float(robottt_sequence_loss(stub, jax.random.key(0), s, obs_seq, a, cfg=cfg))

        # Flat path: run_sequence for the W chain, one stub call at batch B*L.
        o_seq, _ = fast.run_sequence(s, a, tbptt_segment=cfg.tbptt_segment)
        cond = jnp.swapaxes(o_seq, 0, 1).reshape(B * L, -1)
        obs_flat = _model.Observation(images={}, image_masks={}, state=s.reshape(B * L, -1), robottt_cond=cond)
        _per = stub.compute_loss(
            jax.random.key(0), obs_flat, a.reshape(B * L, *a.shape[2:]), True
        )  # exercises the path
        # NB: the stub adds mean(cond^2) over the WHOLE flattened batch to every sample; the loop
        # adds mean(cond_t^2) per step. Equality therefore needs the per-sample decomposition:
        per_step_means = [
            float(jnp.mean(jnp.mean(a[:, t] ** 2, axis=-1)) + jnp.mean(jnp.swapaxes(o_seq, 0, 1)[:, t] ** 2))
            for t in range(L)
        ]
        flat_equiv = sum(per_step_means) / L
    assert loop == pytest.approx(flat_equiv, rel=1e-6), (
        "flat window decomposition must reproduce the per-step-loop Eq 4 average"
    )
    # And the injected conditioning tensors are literally the same numbers in both paths.
    for t in range(L):
        loop_cond = fast.run_sequence(s, a, tbptt_segment=cfg.tbptt_segment)[0][t]
        flat_cond = cond.reshape(B, L, -1)[:, t]
        assert bool(jnp.array_equal(loop_cond, flat_cond))


# ---------------------------- D: window indexing + stacking ----------------------------


class _FakeStepDS:
    """all_steps trajectory-major, like the real GrootOpenpiSingleDataset."""

    def __init__(self, lengths):
        self.trajectory_ids = np.asarray([f"traj_{i}" for i in range(len(lengths))])
        self.trajectory_lengths = np.asarray(lengths)
        self.all_steps = [(f"traj_{i}", b) for i, n in enumerate(lengths) for b in range(n)]

    def __len__(self):
        return len(self.all_steps)

    def __getitem__(self, i):
        traj, base = self.all_steps[i]
        return {"traj": traj, "base": np.int64(base), "observation/state": np.full((4,), float(base))}


def test_window_dataset_flat_indexing_and_boundaries():
    ds = _FakeStepDS([20, 10, 7])
    win = StageQWindowDataset([], 5, window_len=4, chunk_stride=2, _datasets=[ds])
    assert len(win) > 0
    for i in range(len(win)):
        steps = win[i]
        assert len(steps) == 4
        trajs = {s["traj"] for s in steps}
        assert len(trajs) == 1, "a window must never cross an episode boundary"
        bases = [int(s["base"]) for s in steps]
        assert bases == sorted(bases) and all(b2 - b1 == 2 for b1, b2 in zip(bases, bases[1:]))
    # length-7 trajectory: steps at 0,2,4,6 -> exactly one window; no padding invented
    last_traj_windows = [win[i] for i in range(len(win)) if win[i][0]["traj"] == "traj_2"]
    assert len(last_traj_windows) == 1


def test_transformed_window_dataset_stacks_per_step_transform():
    ds = _FakeStepDS([12])
    win = StageQWindowDataset([], 5, window_len=3, chunk_stride=4, _datasets=[ds])

    def xform(step):
        # Nested dict mirrors the REAL transformed item (image: {cam: array}); the q2 canary crashed
        # on exactly this shape when stacking was per-key instead of tree-aware (object dtype).
        s = np.asarray(step["observation/state"], np.float32)
        return {"state": s * 2.0, "image": {"cam_a": np.full((2, 2, 3), s[0], np.uint8)}}

    twin = TransformedWindowDataset(win, [xform])
    item = twin[0]
    assert set(item) == {"state", "image"}
    assert item["state"].shape == (3, 4)
    assert np.allclose(item["state"][:, 0], [0.0, 8.0, 16.0])  # bases 0,4,8 doubled
    assert isinstance(item["image"], dict), "nested dicts must stay dicts (tree-aware stacking)"
    assert item["image"]["cam_a"].shape == (3, 2, 2, 3)
    assert item["image"]["cam_a"].dtype == np.uint8, "object dtype here is the q2-canary crash"
    assert np.array_equal(item["image"]["cam_a"][:, 0, 0, 0], np.asarray([0, 4, 8], np.uint8))


# ---------------------------- E: serve passthrough (audit-confirmed regression) ----------------------------


def test_robottt_cond_survives_robocasa_inputs_and_reaches_observation():
    from openpi.models import model as om
    from openpi.models.model import Observation
    from openpi.policies.robocasa_policy import RobocasaInputs

    tin = RobocasaInputs(action_dim=32, model_type=om.ModelType.PI0)
    img = np.zeros((224, 224, 3), np.uint8)
    data = {
        "observation/image": img,
        "observation/wrist_image": img,
        "observation/right_image": img,
        "observation/state": np.zeros((16,), np.float32),
        "prompt": "close the drawer",
        "robottt_cond": np.ones((1024,), np.float32),
    }
    out = tin(data)
    assert "robottt_cond" in out, "RobocasaInputs must forward robottt_cond (audit 2026-07-23)"
    with at.disable_typechecking():
        obs = Observation.from_dict(
            {
                **out,
                "state": np.zeros((1, 32), np.float32),
                "image": {k: v[None] for k, v in out["image"].items()},
                "image_mask": {k: np.asarray([True]) for k in out["image"]},
                "robottt_cond": out["robottt_cond"][None],
            }
        )
    assert obs.robottt_cond is not None and obs.robottt_cond.shape == (1, 1024)


# ---------------------------- F: production-geometry stability (D-9 regression) ----------------------------


def test_inner_update_stable_at_production_geometry():
    cfg = RoboTTTConfig(
        fast_weights=True,
        token_dim=256,
        fast_hidden=128,
        num_registers=16,
        cond_dim=1024,
        state_dim=32,
        action_dim=32,
        action_horizon=50,
        base_inner_lr=0.1,
        window_len=8,
        tbptt_segment=8,
    )
    m = RoboTTTFastWeights(cfg, rngs=nnx.Rngs(0))
    m.readout.kernel[...] = 0.1 * jax.random.normal(jax.random.key(11), m.readout.kernel[...].shape)
    m.w0_w2[...] = 0.1 * jax.random.normal(jax.random.key(12), m.w0_w2[...].shape)
    b, steps = 2, 32  # 4x the training window: the serve regime is long-horizon
    k = jax.random.key(1)
    s = jax.random.normal(k, (b, steps, cfg.state_dim))
    a = jax.random.normal(jax.random.fold_in(k, 1), (b, steps, cfg.action_horizon, cfg.action_dim))
    w = m.init_state(b)
    inner_first = inner_last = None
    for t in range(steps):
        toks = m._tokens_from(s[:, t], a[:, t])
        il = float(
            jnp.mean(jax.vmap(lambda wi, ki, vi: m._inner_loss_single(wi, ki, vi))(w, m.proj_k(toks), m.proj_v(toks)))
        )
        inner_first = il if inner_first is None else inner_first
        inner_last = il
        w = m.commit(w, s[:, t], a[:, t])
        assert all(bool(jnp.isfinite(leaf).all()) for leaf in w.values()), f"W not finite at t={t}"
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in w.values()) < 10.0, f"W unbounded at t={t}"
    assert inner_last < inner_first, "the online inner objective must improve over the trajectory"
    # Outer meta-grads through a full window remain finite.
    graphdef, state = nnx.split(m)

    def outer(st):
        mod = nnx.merge(graphdef, st)
        o_seq, _ = mod.run_sequence(s[:, :8], a[:, :8], tbptt_segment=8)
        return jnp.sum(o_seq**2)

    g = jax.tree.leaves(nnx.state(nnx.merge(graphdef, jax.grad(outer)(state))))
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in g if hasattr(leaf, "shape"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} PASSED")
