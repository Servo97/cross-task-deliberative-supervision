"""Load-bearing tests for paper-faithful RoboTTT fast weights (packet 07 §7d).

Two families:
  * MODULE / PI (JAX): the fast-weight mechanism in `openpi.models.robottt_fast_weights` and its
    pi0.5 seam. Skips cleanly if jax/openpi is not importable.
  * SERVING (pure fakes, no JAX): the per-env fast-weight lifecycle in `WSMPiInferWrapper`
    (`vla_training/eval/serve_pi_05_wsm.py`), mirroring `test_pi_wsm_stateful_batching.py`.

Run: PYTHONPATH=. ~/Research/envs/openpi-jax-latest/bin/python -m pytest -q \
     tests/test_robottt_fast_weights.py

The 10 tests map 1:1 to packet 07 §7d:
  1  alpha_W=0 / updates-off  => exact parity with Q0
  2  exactly one commit per executed chunk
  3  no update during Euler/CFG branching (W held; condition once, commit once)
  4  reset / interleaved-client / gathered-batch / eviction isolation
  5  nonzero outer grads to W_0/registers/meta-params; zero to non-intended
  6  untruncated vs long-TBPTT forward equality; gradient cut exactly at the boundary
  7  action forcing cannot read future ground-truth at inference (structural)
  8  checkpoint/resume restores slow/meta params; online W starts from the episode init
  9  tiny synthetic: inner update improves its declared inner objective and changes outputs
  10 one-batch pi integration: Q2 (RoboTTT) output != tanh-workspace-read output

Q3 combined-serve (tanh_robottt) additions:
  17 combined wrapper: omega injection + held-W robottt_cond on the SAME chunk, prompt contract,
     one commit per executed chunk, norm rows stripped
  18 combined runner parity (test-16 mirror with the tanh subtree present) + the tanh modulator
     is loaded and modulates
  19 from_pretrained-bypass guards: refuse alpha==gate_init for BOTH trained subtrees
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsm_settings import ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

# ---------------------------------------------------------------------------------------------
# Serving-side fakes (no JAX). Mirror tests/test_pi_wsm_stateful_batching.py.
# ---------------------------------------------------------------------------------------------
from vla_training.eval.serve_pi_05_wsm import WSMPiInferWrapper  # noqa: E402


class _Tap:
    def tap(self, frames, state, prompts):
        from types import SimpleNamespace

        marker = np.asarray(frames["agentview_left"], dtype=np.float32)[:, 0, 0, 0]
        return SimpleNamespace(patch_tokens=marker[:, None, None], lang_emb=(marker + 0.5)[:, None])


class _Conditioner:
    """Minimal omega conditioner so the inherited omega path runs alongside RoboTTT."""

    def __init__(self, k=1):
        self.k = int(k)
        self.history = []
        self.lang = None

    def reset(self, lang):
        self.history = []
        self.lang = np.asarray(lang, dtype=np.float32)

    def _result(self):
        vals = list(np.cumsum(self.history, dtype=np.float32))[-self.k :]
        vals = [vals[0]] * (self.k - len(vals)) + vals
        return np.asarray(vals, dtype=np.float32)[:, None], self.lang.copy()

    def step(self, patch, proprio):
        self.history.append(float(np.asarray(patch).reshape(-1)[0]))
        return self._result()

    @classmethod
    def step_many(cls, conditioners, patches, proprio):
        for c, p in zip(conditioners, patches):
            c.history.append(float(np.asarray(p).reshape(-1)[0]))
        return [c._result() for c in conditioners]


class _Policy:
    """Fake policy that records the robottt_cond it saw and emits an action chunk to commit on."""

    metadata = {"fake": True}

    def __init__(self):
        self.seen_cond = []
        self.action_value = 2.0

    def _result(self, obs):
        for key in ("wsm_env_id", "wsm_task", "wsm_demo_episode", "wsm_t", "wsm_prompt"):
            assert key not in obs
        assert "actions" not in obs and "action" not in obs  # test 7: no labels reach the policy
        cond = obs.get("robottt_cond")
        self.seen_cond.append(None if cond is None else np.asarray(cond).copy())
        return {
            "request_id": obs["request_id"],
            "robottt_cond_seen": None if cond is None else np.asarray(cond).copy(),
            "actions": np.full((3, 2), self.action_value + 1.0, dtype=np.float32),
            # Model-space rows (expose_norm_actions=True contract): the commit consumes THESE, never
            # the unnormalized client-facing "actions" above (deliberately offset by +1 so a test
            # committing on the wrong key shows a wrong mean).
            "norm_state": np.full((4,), self.action_value, dtype=np.float32),
            "norm_actions": np.full((3, 2), self.action_value, dtype=np.float32),
        }

    def infer(self, obs, **kwargs):
        return self._result(obs)

    def infer_batch(self, obs_list, **kwargs):
        return [self._result(o) for o in obs_list]


class _FakeRoboTTTRunner:
    """Scalar-state stand-in for the real fast-weight runner. State h is a pure function of history.

    condition() reads the HELD state (never mutates); commit() advances h by the executed chunk mean.
    Distinct from omega: condition ignores wsm_w_window entirely.
    """

    COND_DIM = 4

    def __init__(self):
        self.init_calls = 0
        self.condition_calls = []  # (id(w), h_seen)
        self.commit_calls = []  # (h_before, chunk_mean)

    def init_state(self):
        self.init_calls += 1
        return {"h": np.float32(0.0)}

    def condition(self, w, obs):
        self.condition_calls.append((id(w), float(w["h"])))
        return np.full(self.COND_DIM, w["h"], dtype=np.float32)

    def commit(self, w, norm_state, norm_actions):
        chunk_mean = float(np.asarray(norm_actions).mean())
        self.commit_calls.append((float(w["h"]), chunk_mean))
        return {"h": np.float32(w["h"] + chunk_mean)}

    def is_finite(self, w):
        return bool(np.isfinite(w["h"]))


_TABLE = {
    "task_a": np.asarray([100.0], dtype=np.float32),
    "task_b": np.asarray([200.0], dtype=np.float32),
}


def _obs(env, task, demo, t, marker):
    image = np.full((2, 2, 3), marker, dtype=np.uint8)
    return {
        "request_id": f"{env}:{task}:{demo}:{t}",
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/right_image": image,
        "observation/state": np.asarray([marker], dtype=np.float32),
        "prompt": f"terse {task}",
        "wsm_env_id": env,
        "wsm_task": task,
        "wsm_demo_episode": demo,
        "wsm_t": t,
        "policy_noise_seed": int(marker),
    }


def _wrapper(max_envs=2, max_grid_frames=32):
    policy = _Policy()
    runner = _FakeRoboTTTRunner()
    wrapper = WSMPiInferWrapper(
        policy,
        _Tap(),
        _Conditioner(),
        _TABLE,
        stride=8,
        max_envs=max_envs,
        max_grid_frames=max_grid_frames,
        conditioner_factory=_Conditioner,
        robottt_runner=runner,
    )
    return wrapper, policy, runner


# ============================== SERVING TESTS (tests 2, 3, 4, 7) ==============================


def test_02_exactly_one_commit_per_executed_chunk():
    wrapper, policy, runner = _wrapper(max_envs=1)
    # Reset (t=0) then two more chunks at stride-aligned grids: 3 executed chunks total.
    wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    wrapper.infer(_obs("env_a", "task_a", 1, 8, 11))
    wrapper.infer(_obs("env_a", "task_a", 1, 16, 12))
    assert runner.init_calls == 1  # one episode
    assert len(runner.commit_calls) == 3  # exactly one commit per executed chunk
    assert wrapper._states["env_a"].robottt_commits == 3
    # Each commit consumed the policy's OWN executed chunk (mean of the 2.0 action fill).
    assert [round(m, 3) for _, m in runner.commit_calls] == [2.0, 2.0, 2.0]


def test_03_no_update_during_euler_cfg_branching():
    wrapper, policy, runner = _wrapper(max_envs=1)
    wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    # First chunk enters with h=0; the policy sees exactly that held state, and exactly ONE condition
    # call served the whole chunk (all Euler/CFG passes share it). The single commit ran AFTER.
    assert len(runner.condition_calls) == 1
    assert runner.condition_calls[0][1] == 0.0  # conditioned on the ENTERING (pre-commit) W
    assert np.array_equal(policy.seen_cond[0], np.zeros(runner.COND_DIM, np.float32))
    assert len(runner.commit_calls) == 1
    # Second chunk: h advanced to 2.0 by the one prior commit; policy sees the new held state, still
    # one condition + one commit (no mid-chunk updates).
    wrapper.infer(_obs("env_a", "task_a", 1, 8, 11))
    assert len(runner.condition_calls) == 2
    assert runner.condition_calls[1][1] == 2.0
    assert np.array_equal(policy.seen_cond[1], np.full(runner.COND_DIM, 2.0, np.float32))
    assert len(runner.commit_calls) == 2


def test_04_reset_interleaved_gathered_batch_and_eviction_isolation():
    wrapper, policy, runner = _wrapper(max_envs=2)
    # Gathered batch of two independent envs: each conditions on its own private W (both h=0 at reset).
    out = wrapper.infer_batch([_obs("env_a", "task_a", 1, 0, 10), _obs("env_b", "task_b", 2, 0, 20)])
    assert np.array_equal(out[0]["robottt_cond_seen"], np.zeros(runner.COND_DIM, np.float32))
    assert np.array_equal(out[1]["robottt_cond_seen"], np.zeros(runner.COND_DIM, np.float32))
    # After one commit each, both advance to h=2.0 but in SEPARATE state objects.
    wa, wb = wrapper._states["env_a"].robottt_w, wrapper._states["env_b"].robottt_w
    assert wa is not wb and wa["h"] == 2.0 and wb["h"] == 2.0
    # Interleaved next step (order swapped) keeps per-env identity.
    wrapper.infer_batch([_obs("env_b", "task_b", 2, 8, 21), _obs("env_a", "task_a", 1, 8, 11)])
    assert wrapper._states["env_a"].robottt_w["h"] == 4.0
    assert wrapper._states["env_b"].robottt_w["h"] == 4.0
    # Reset env_a only: its W re-inits to h=0 (proven by init_calls++ and the reset chunk conditioning
    # on h=0, NOT the pre-reset 4.0). That reset chunk then executes and commits once -> h=2.0. env_b
    # is untouched.
    reset_init_calls = runner.init_calls
    reset_cond_calls = len(runner.condition_calls)
    wrapper.infer(_obs("env_a", "task_a", 3, 0, 30))
    assert runner.init_calls == reset_init_calls + 1
    assert runner.condition_calls[reset_cond_calls][1] == 0.0  # conditioned on the re-initialized W
    assert wrapper._states["env_a"].robottt_w["h"] == 2.0  # 0 (reset) + one commit
    assert wrapper._states["env_b"].robottt_w["h"] == 4.0
    # Eviction: a new env over the bound fails loud rather than silently sharing/evicting live W.
    with pytest.raises(RuntimeError, match="active env-state bound exceeded"):
        wrapper.infer(_obs("env_c", "task_a", 9, 0, 40))


def test_07_action_forcing_cannot_read_future_ground_truth_at_inference():
    wrapper, policy, runner = _wrapper(max_envs=1)
    out = wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    # Structural: the injected obs the policy saw carried NO label/target key (asserted inside _Policy),
    # and the commit consumed the policy's OWN produced chunk, never a ground-truth action.
    _h_before, chunk_mean = runner.commit_calls[0]
    assert chunk_mean == policy.action_value  # the executed (own) chunk, not a dataset label
    assert "robottt_cond" not in _TABLE  # sanity: table holds omega, not labels
    assert out["actions"].shape == (3, 2)


def test_13_commit_consumes_model_space_rows_and_strips_them():
    wrapper, policy, runner = _wrapper(max_envs=1)
    out = wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    # The commit consumed norm_actions (2.0), NOT the unnormalized client "actions" (3.0).
    assert runner.commit_calls[0][1] == 2.0
    assert float(out["actions"].mean()) == 3.0
    # Model-space rows never leak to the client / persisted shards.
    assert "norm_state" not in out and "norm_actions" not in out


def test_14_workspace_free_mode_serves_robottt_without_tap_or_omega():
    policy = _Policy()
    runner = _FakeRoboTTTRunner()
    wrapper = WSMPiInferWrapper(policy, None, None, None, stride=8, max_envs=2, robottt_runner=runner)
    assert wrapper.metadata["wsm_workspace"] is False
    assert wrapper.metadata["wsm_robottt"] is True
    wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    wrapper.infer(_obs("env_a", "task_a", 1, 8, 11))
    # Full RoboTTT lifecycle without any workspace machinery.
    assert runner.init_calls == 1 and len(runner.commit_calls) == 2
    # The policy saw robottt_cond but NO omega injection.
    assert all(cond is not None for cond in policy.seen_cond)
    injected_keys = set()  # verify via a probe request
    probe = {}

    class _SpyPolicy(_Policy):
        def _result(self, obs):
            probe.update({k: True for k in obs})
            return super()._result(obs)

    spy = _SpyPolicy()
    wrapper2 = WSMPiInferWrapper(spy, None, None, None, stride=8, max_envs=1, robottt_runner=_FakeRoboTTTRunner())
    wrapper2.infer(_obs("env_b", "task_unknown_ok", 5, 0, 12))  # no table => any task accepted
    injected_keys = set(probe)
    assert "wsm_w_window" not in injected_keys and "wsm_lang" not in injected_keys
    assert "robottt_cond" in injected_keys
    # Ordering/isolation validation is unchanged: a replayed (non-advancing) t still refuses.
    wrapper2.infer(_obs("env_b", "task_unknown_ok", 5, 8, 13))
    with pytest.raises(RuntimeError, match="out-of-order"):
        wrapper2.infer(_obs("env_b", "task_unknown_ok", 5, 8, 14))


def test_15_workspace_free_mode_requires_a_runner():
    with pytest.raises(ValueError, match="requires a RoboTTT runner"):
        WSMPiInferWrapper(_Policy(), None, None, None, stride=8, max_envs=1)
    with pytest.raises(ValueError, match="both"):
        WSMPiInferWrapper(
            _Policy(),
            _Tap(),
            None,
            None,
            stride=8,
            max_envs=1,
            robottt_runner=_FakeRoboTTTRunner(),
        )


def test_17_combined_interface_serves_workspace_and_fast_weights_together():
    """Q3 (tanh_robottt) serving contract: the SAME chunk carries BOTH the omega injection
    (identical to tanh, incl. the required private wsm_prompt) and the held-W robottt_cond, with
    exactly one commit per executed chunk and the model-space rows stripped from responses."""
    probe_keys: list[set] = []

    class _SpyPolicy(_Policy):
        def _result(self, obs):
            probe_keys.append(set(obs))
            return super()._result(obs)

    policy = _SpyPolicy()
    runner = _FakeRoboTTTRunner()
    wrapper = WSMPiInferWrapper(
        policy,
        _Tap(),
        _Conditioner(),
        _TABLE,
        stride=8,
        max_envs=2,
        conditioner_factory=_Conditioner,
        require_wsm_prompt=True,
        robottt_runner=runner,
    )
    assert wrapper.metadata["wsm_workspace"] is True
    assert wrapper.metadata["wsm_robottt"] is True
    assert wrapper.metadata["wsm_required_signal_fields"] == ["wsm_prompt"]

    # The tanh half's private prompt contract holds unchanged under the combined interface.
    with pytest.raises(RuntimeError, match="wsm_prompt"):
        wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))

    def obs_with_prompt(env, task, demo, t, marker):
        obs = _obs(env, task, demo, t, marker)
        obs["wsm_prompt"] = f"canonical {task}"
        return obs

    out = wrapper.infer(obs_with_prompt("env_a", "task_a", 1, 0, 10))
    wrapper.infer(obs_with_prompt("env_a", "task_a", 1, 8, 11))
    # Both halves reached the policy on every chunk; the protocol signal keys never did.
    for keys in probe_keys:
        assert {"wsm_w_window", "wsm_lang", "robottt_cond"} <= keys
        assert not keys & {"wsm_env_id", "wsm_task", "wsm_demo_episode", "wsm_t", "wsm_prompt"}
    # Fast-weight lifecycle identical to Q2: one init at reset, one commit per executed chunk.
    assert runner.init_calls == 1
    assert len(runner.commit_calls) == 2
    assert wrapper._states["env_a"].robottt_commits == 2
    # Model-space rows fed the commit and never leaked to the client.
    assert runner.commit_calls[0][1] == policy.action_value
    assert "norm_state" not in out and "norm_actions" not in out


# ============================== MODULE / PI TESTS (JAX) ==============================

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx", reason="flax nnx required")
try:
    from openpi.models.robottt_fast_weights import (
        RoboTTTConfig,
        RoboTTTFastWeights,
        robottt_sequence_loss,
    )
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"openpi robottt module not importable ({exc})", allow_module_level=True)


def _cfg(**kw):
    base = dict(
        fast_weights=True,
        token_dim=16,
        fast_hidden=8,
        num_registers=3,
        cond_dim=12,
        state_dim=4,
        action_dim=4,
        action_horizon=5,
        window_len=4,
        tbptt_segment=2,
    )
    base.update(kw)
    return RoboTTTConfig(**base)


def _module(seed=0, **kw):
    return RoboTTTFastWeights(_cfg(**kw), rngs=nnx.Rngs(seed))


def _detrain(m):
    """Move the zero-init readout / second fast layer off zero so the whole path is observable."""
    m.readout.kernel[...] = 0.1 * jax.random.normal(jax.random.key(11), m.readout.kernel[...].shape)
    m.readout.bias[...] = 0.05 * jnp.ones_like(m.readout.bias[...])
    m.w0_w2[...] = 0.1 * jax.random.normal(jax.random.key(12), m.w0_w2[...].shape)


def _seq(B, cfg, seed=3):
    k = jax.random.key(seed)
    s = jax.random.normal(k, (B, cfg.window_len, cfg.state_dim))
    a = jax.random.normal(jax.random.fold_in(k, 1), (B, cfg.window_len, cfg.action_horizon, cfg.action_dim))
    return s, a


# ----- test 1: alpha_W=0 / updates-off parity with Q0 -----


class _SeqStub:
    """Pi-shaped stub: compute_loss reads only obs.robottt_cond so we can prove exact Q0 parity."""

    def __init__(self, fastmod):
        self.robottt = True
        self.robottt_fast = _SpyFast(fastmod)

    def compute_loss(self, rng, obs, actions, train):
        base = jnp.ones((actions.shape[0], actions.shape[1]))  # [B, H]
        rc = obs.robottt_cond
        extra = jnp.zeros(()) if rc is None else jnp.sum(rc)
        return base + extra


class _SpyFast:
    def __init__(self, real):
        self._real = real
        self.n_condition = 0
        self.n_commit = 0

    def init_state(self, b):
        return self._real.init_state(b)

    def condition(self, w, s):
        self.n_condition += 1
        return self._real.condition(w, s)

    def commit(self, w, s, a):
        self.n_commit += 1
        return self._real.commit(w, s, a)


def _mk_obs_seq(B, cfg):
    from openpi.models import model as _model
    from openpi.shared import array_typing as at

    obs_seq = []
    with at.disable_typechecking():
        for _ in range(cfg.window_len):
            obs_seq.append(_model.Observation(images={}, image_masks={}, state=jnp.zeros((B, cfg.state_dim))))
    return obs_seq


def test_01_alpha_zero_and_updates_off_match_q0():
    B = 2
    # (a) alpha_scale=0 => the RoboTTT contribution O_t is exactly the zero vector, so adding it to
    #     adarms_cond is a no-op => byte parity with Q0.
    m0 = _module(alpha_scale=0.0)
    _detrain(m0)
    w = m0.init_state(B)
    s, a = _seq(B, m0.cfg)
    w = m0.commit(w, s[:, 0], a[:, 0])
    o = m0.condition(w, s[:, 1])
    assert float(jnp.max(jnp.abs(o))) == 0.0, "alpha_scale=0 must zero the RoboTTT contribution"

    # (b) fast_weights OFF in the sequence loss => condition/commit are never called and the loss is
    #     exactly the per-step baseline (Q0). fast_weights ON with alpha_scale=0 injects only zeros =>
    #     identical loss, proving the graph reduces to Q0.
    cfg_off = _cfg(fast_weights=False)
    cfg_on0 = _cfg(fast_weights=True, alpha_scale=0.0)
    s, a = _seq(B, cfg_off)
    obs_seq = _mk_obs_seq(B, cfg_off)

    stub_off = _SeqStub(_module())
    loss_off = robottt_sequence_loss(stub_off, jax.random.key(0), s, obs_seq, a, cfg=cfg_off)
    assert stub_off.robottt_fast.n_condition == 0 and stub_off.robottt_fast.n_commit == 0

    stub_on0 = _SeqStub(_module(alpha_scale=0.0))
    _detrain(stub_on0.robottt_fast._real)
    loss_on0 = robottt_sequence_loss(stub_on0, jax.random.key(0), s, obs_seq, a, cfg=cfg_on0)
    assert stub_on0.robottt_fast.n_commit == cfg_on0.window_len
    assert float(loss_off) == float(loss_on0), "alpha_scale=0 must reproduce the Q0 loss exactly"


# ----- test 5: outer grads route to intended params, zero to non-intended -----


def test_05_outer_grads_reach_meta_params_and_skip_unused():
    B = 2
    m = _module()
    _detrain(m)
    s, a = _seq(B, m.cfg)
    graphdef, state = nnx.split(m)

    def full_loss(st):
        mod = nnx.merge(graphdef, st)
        o_seq, _ = mod.run_sequence(s, a, tbptt_segment=m.cfg.window_len)
        return jnp.sum(o_seq**2)

    def apply_only_loss(st):
        mod = nnx.merge(graphdef, st)
        w = mod.init_state(B)
        o = mod.condition(w, s[:, 0])  # apply path only: no commit => proj_k/proj_v/eta unused
        return jnp.sum(o**2)

    gfull = nnx.state(nnx.merge(graphdef, jax.grad(full_loss)(state)))
    gapply = nnx.state(nnx.merge(graphdef, jax.grad(apply_only_loss)(state)))

    def leaf(g, *path):
        node = g
        for p in path:
            node = getattr(node, p)
        return np.asarray(node[...] if hasattr(node, "__getitem__") else node)

    def nz(g, *path):
        return float(np.max(np.abs(leaf(g, *path)))) > 0.0

    # Intended meta-params all receive gradient under the full sequence objective.
    for path in [
        ("w0_w1",),
        ("w0_w2",),
        ("registers",),
        ("alpha",),
        ("log_inner_lr",),
        ("proj_q", "kernel"),
        ("proj_k", "kernel"),
        ("proj_v", "kernel"),
        ("readout", "kernel"),
    ]:
        assert nz(gfull, *path), f"expected nonzero full-loss grad at {path}"

    # Apply-only objective never invokes commit => proj_k/proj_v/log_inner_lr grads are EXACTLY zero
    # (zero to non-intended), while the apply-path params still receive gradient.
    assert float(np.max(np.abs(leaf(gapply, "proj_k", "kernel")))) == 0.0
    assert float(np.max(np.abs(leaf(gapply, "proj_v", "kernel")))) == 0.0
    assert float(np.abs(leaf(gapply, "log_inner_lr"))) == 0.0
    assert nz(gapply, "proj_q", "kernel") and nz(gapply, "readout", "kernel") and nz(gapply, "registers")


# ----- test 6: TBPTT forward-equality + gradient cut at boundary -----


def test_06_tbptt_forward_equal_and_gradient_cut():
    B = 2
    m = _module()
    _detrain(m)
    s, a = _seq(B, m.cfg)
    L = m.cfg.window_len

    # Forward VALUES are independent of the TBPTT segment length (stop_gradient is identity forward).
    o_full, _ = m.run_sequence(s, a, tbptt_segment=L)
    o_cut, _ = m.run_sequence(s, a, tbptt_segment=1)
    assert bool(jnp.array_equal(o_full, o_cut)), "TBPTT must not change the forward pass"

    graphdef, state = nnx.split(m)

    def last_step_loss(state_seq, segment):
        mod = nnx.merge(graphdef, state)
        o_seq, _ = mod.run_sequence(state_seq, a, tbptt_segment=segment)
        return jnp.sum(o_seq[-1] ** 2)  # loss on the LAST chunk-step only

    g_full = np.asarray(jax.grad(lambda ss: last_step_loss(ss, L))(s))
    g_cut = np.asarray(jax.grad(lambda ss: last_step_loss(ss, 1))(s))

    # Full BPTT: the last step's loss reaches the FIRST step's input through the W chain.
    assert float(np.max(np.abs(g_full[:, 0]))) > 0.0
    # TBPTT=1: the carried W is detached at every boundary, so the gradient to the first step is cut
    # exactly to zero, while the within-segment (last-step) input still receives gradient.
    assert float(np.max(np.abs(g_cut[:, 0]))) == 0.0
    assert float(np.max(np.abs(g_cut[:, L - 1]))) > 0.0


# ----- test 8: checkpoint round-trip; online W is episode-initialized from W_0 -----


def test_08_checkpoint_restores_meta_params_and_online_w_starts_from_init():
    m = _module(seed=1)
    _detrain(m)
    saved = nnx.state(m)
    saved_w1 = np.asarray(m.w0_w1[...]).copy()

    fresh = _module(seed=999)  # different init
    assert not np.allclose(np.asarray(fresh.w0_w1[...]), saved_w1)
    nnx.update(fresh, saved)
    assert np.allclose(np.asarray(fresh.w0_w1[...]), saved_w1)
    assert np.allclose(np.asarray(fresh.alpha[...]), np.asarray(m.alpha[...]))
    assert np.allclose(np.asarray(fresh.readout.kernel[...]), np.asarray(m.readout.kernel[...]))

    # Online W is a pure function of the (restored) W_0 at the episode boundary — not carried across.
    B = 3
    w_init = m.init_state(B)
    assert np.allclose(np.asarray(w_init["w1"][0]), saved_w1)  # broadcast of the learned init
    s, a = _seq(B, m.cfg)
    w_after = m.commit(w_init, s[:, 0], a[:, 0])
    w_reinit = m.init_state(B)  # a fresh episode ignores any prior online update
    assert np.allclose(np.asarray(w_reinit["w1"]), np.asarray(w_init["w1"]))
    assert not np.allclose(np.asarray(w_after["w2"]), np.asarray(w_init["w2"]))


# ----- test 9: tiny synthetic inner update improves the inner objective and changes outputs -----


def test_09_inner_update_improves_inner_objective_and_changes_outputs():
    d, h, m_tok = 6, 5, 4
    key = jax.random.key(7)
    ks = jax.random.split(key, 6)
    W = {
        "w1": jax.random.normal(ks[0], (d, h)) * 0.3,
        "b1": jax.random.normal(ks[1], (h,)) * 0.1,
        "w2": jax.random.normal(ks[2], (h, d)) * 0.3,
        "b2": jax.random.normal(ks[3], (d,)) * 0.1,
    }
    K = jax.random.normal(ks[4], (m_tok, d))
    V = jax.random.normal(ks[5], (m_tok, d))
    Q = jax.random.normal(jax.random.fold_in(key, 9), (m_tok, d))
    eta = jnp.asarray(0.1)

    l0 = float(RoboTTTFastWeights._inner_loss_single(W, K, V))
    W1 = RoboTTTFastWeights._commit_single(W, K, V, eta)
    l1 = float(RoboTTTFastWeights._inner_loss_single(W1, K, V))
    assert l1 < l0, f"one inner step must reduce L_FW: {l0} -> {l1}"

    o0 = np.asarray(RoboTTTFastWeights._apply_single(W, Q))
    o1 = np.asarray(RoboTTTFastWeights._apply_single(W1, Q))
    assert float(np.max(np.abs(o1 - o0))) > 0.0, "the apply output must change after the inner update"


def test_16_serve_runner_matches_training_run_sequence():
    """Train/serve parity for the REAL runner: RoboTTTServeRunner's per-chunk condition/commit
    sequence over an episode reproduces run_sequence's O_seq (the exact chain training saw when
    conditioning), given the same normalized rows."""
    from types import SimpleNamespace

    from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner

    m = _module()
    _detrain(m)
    cfg = m.cfg
    s, a = _seq(1, cfg, seed=5)  # [1, L, S], [1, L, H, A]
    o_seq, _w_final = m.run_sequence(s, a)  # [L, 1, C]

    # Minimal policy facade: the runner touches only _model.{robottt, robottt_fast} and
    # _input_transform (identity here — the test feeds pre-normalized model rows directly).
    policy = SimpleNamespace(
        _model=SimpleNamespace(robottt=True, robottt_fast=m),
        _input_transform=lambda x: x,
    )
    runner = RoboTTTServeRunner(policy)

    w = runner.init_state()
    conds = []
    for t in range(cfg.window_len):
        conds.append(runner.condition(w, {"state": np.asarray(s[0, t])}))
        w = runner.commit(w, np.asarray(s[0, t]), np.asarray(a[0, t]))  # teacher rows == training chain
        assert runner.is_finite(w)
    np.testing.assert_allclose(np.stack(conds), np.asarray(o_seq[:, 0, :]), rtol=1e-5, atol=1e-6)


def test_18_q3_combined_runner_matches_training_and_tanh_modulator_modulates():
    """Q3 parity, mirroring test_16 for the fast-weights half: with the tanh workspace subtree
    PRESENT on the served model, RoboTTTServeRunner's per-chunk condition/commit chain still
    reproduces run_sequence's O_seq exactly (the workspace read must not perturb the fast-weight
    chain). The workspace half is covered by asserting the tanh modulator is loaded and actually
    modulates: omega-dependent, deterministic, and gated off near-zero at gate_init."""
    from types import SimpleNamespace

    from openpi.models.wsm_current_cond import WSMTanhConditioner

    from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner

    m = _module()
    _detrain(m)
    cfg = m.cfg
    s, a = _seq(1, cfg, seed=5)
    o_seq, _w_final = m.run_sequence(s, a)  # the exact chain training saw

    tanh_cond = WSMTanhConditioner(w_dim=8, cond_dim=cfg.cond_dim, gate_init=1e-3, rngs=nnx.Rngs(2))
    policy = SimpleNamespace(
        _model=SimpleNamespace(robottt=True, robottt_fast=m, wsm_tanh=True, wsm_tanh_cond=tanh_cond),
        _input_transform=lambda x: x,
    )
    runner = RoboTTTServeRunner(policy)

    w = runner.init_state()
    conds = []
    for t in range(cfg.window_len):
        conds.append(runner.condition(w, {"state": np.asarray(s[0, t])}))
        w = runner.commit(w, np.asarray(s[0, t]), np.asarray(a[0, t]))
        assert runner.is_finite(w)
    np.testing.assert_allclose(np.stack(conds), np.asarray(o_seq[:, 0, :]), rtol=1e-5, atol=1e-6)

    # Workspace half: the tanh modulator is loaded and modulates. At gate_init the read is gated
    # near zero (|tanh(1e-3)| bound); a trained gate produces a genuine omega-dependent vector.
    omega_a = jnp.asarray(np.random.default_rng(0).normal(size=(2, 8)), jnp.float32)
    omega_b = omega_a + 1.0
    v_init = np.asarray(tanh_cond(omega_a))
    assert float(np.max(np.abs(v_init))) <= float(np.tanh(1e-3)) * (
        float(np.max(np.abs(np.asarray(tanh_cond.proj_t_out(jax.nn.silu(tanh_cond.proj_t_in(omega_a))))))) + 1e-6
    )
    tanh_cond.alpha[...] = tanh_cond.alpha[...] + 0.5  # trained-style gate
    v_a = np.asarray(tanh_cond(omega_a))
    v_b = np.asarray(tanh_cond(omega_b))
    assert float(np.max(np.abs(v_a))) > 0.0
    assert float(np.max(np.abs(v_a - v_b))) > 0.0  # modulates with omega
    assert bool(np.array_equal(np.asarray(tanh_cond(omega_a)), v_a))  # deterministic read
    # And the fast-weight chain is untouched by the workspace read: same runner, same parity.
    w2 = runner.init_state()
    np.testing.assert_allclose(runner.condition(w2, {"state": np.asarray(s[0, 0])}), conds[0], rtol=0, atol=0)


def test_19_combined_serve_refuses_unloaded_trained_subtrees():
    """The from_pretrained-bypass trap, both halves: serving refuses when either trained subtree
    (robottt_fast alpha or wsm_tanh_cond alpha) is still exactly its constant init — i.e. the
    checkpoint restore silently missed it — and accepts once the params depart from init."""
    from types import SimpleNamespace

    from openpi.models.wsm_current_cond import WSMTanhConditioner

    from vla_training.eval.serve_pi_05_robottt import assert_robottt_loaded
    from vla_training.eval.serve_pi_05_wsm_cfg import assert_tanh_cond_trained

    m = _module()
    policy = SimpleNamespace(_model=SimpleNamespace(robottt=True, robottt_fast=m))
    with pytest.raises(RuntimeError, match="NOT restored"):
        assert_robottt_loaded(policy)  # fresh alpha == gate_init everywhere
    m.alpha[...] = m.alpha[...] + 0.05
    assert_robottt_loaded(policy)  # trained-style alpha passes

    cond = WSMTanhConditioner(w_dim=8, cond_dim=12, gate_init=1e-3, rngs=nnx.Rngs(0))
    tanh_policy = SimpleNamespace(_model=SimpleNamespace(wsm_tanh=True, wsm_tanh_cond=cond))
    with pytest.raises(RuntimeError, match="NOT restored"):
        assert_tanh_cond_trained(tanh_policy, 1e-3)
    cond.alpha[...] = cond.alpha[...] + 0.02
    assert_tanh_cond_trained(tanh_policy, 1e-3)

    # Wrong-config direction: a model without the subtree refuses loudly, never no-ops.
    with pytest.raises(RuntimeError, match="no wsm_tanh_cond"):
        assert_tanh_cond_trained(SimpleNamespace(_model=SimpleNamespace(wsm_tanh=False)), 1e-3)
    with pytest.raises(RuntimeError, match="no robottt_fast"):
        assert_robottt_loaded(SimpleNamespace(_model=SimpleNamespace(robottt=False)))


# ----- test 10: one-batch pi integration — Q2 (RoboTTT) output != tanh-workspace-read output -----


def test_10_pi_integration_q2_differs_from_tanh_workspace_read():
    import dataclasses

    from openpi.models.pi0_config import Pi0Config
    from openpi.models.wsm_current_cond import WSMTanhConditioner

    B, C = 2, 1024  # action-expert width for gemma_300m
    cfg = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        robottt=True,
        robottt_token_dim=16,
        robottt_fast_hidden=8,
        robottt_num_registers=3,
    )
    model = cfg.create(jax.random.key(0))

    # The adaRMS modulation Dense is zero-init (identity at step 0), so ANY adarms_cond addition is a
    # no-op at init. Perturb the action-expert modulation kernels so the seam is observable, matching a
    # trained model. This is the only way a one-batch pi forward can distinguish the two reads.
    flat = dict(nnx.to_flat_state(nnx.state(model)))
    rng_p = jax.random.key(21)
    for path, val in list(flat.items()):
        name = ".".join(str(x) for x in path)
        if name.endswith("_1.Dense_0.kernel") and ("pre_attention_norm" in name or "pre_ffw_norm" in name):
            rng_p, sub = jax.random.split(rng_p)
            flat[path] = type(val)(0.02 * jax.random.normal(sub, val[...].shape))
    nnx.update(model, nnx.from_flat_state(flat))

    # Non-degenerate inputs (adaRMS scales normalized activations; zero inputs are insensitive).
    obs, act = cfg.inputs_spec(batch_size=B)
    kx = jax.random.split(jax.random.key(5), 16)
    idx = [0]

    def rnd(spec):
        idx[0] += 1
        if spec.dtype == jnp.float32:
            return jax.random.normal(kx[idx[0] % 16], spec.shape, spec.dtype)
        if spec.dtype == bool:
            return jnp.ones(spec.shape, bool)
        return jnp.zeros(spec.shape, spec.dtype)

    obs = jax.tree.map(rnd, obs)
    obs = dataclasses.replace(
        obs,
        tokenized_prompt=jnp.ones((B, cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, cfg.max_token_len), bool),
    )
    act = jax.random.normal(jax.random.key(9), act.shape, act.dtype)

    # RoboTTT read (Q2): O_t from the recurrent fast weights after a commit. Force the readout off zero
    # so O_t is a genuine (nonzero) conditioning vector.
    model.robottt_fast.readout.kernel[...] = 0.1 * jax.random.normal(
        jax.random.key(3), model.robottt_fast.readout.kernel[...].shape
    )
    model.robottt_fast.w0_w2[...] = 0.1 * jax.random.normal(jax.random.key(4), model.robottt_fast.w0_w2[...].shape)
    proprio = jax.random.normal(jax.random.key(6), (B, cfg.action_dim))
    w0 = model.robottt_fast.init_state(B)
    w1 = model.robottt_fast.commit(w0, proprio, jnp.ones((B, cfg.action_horizon, cfg.action_dim)))
    o_robottt = model.robottt_fast.condition(w1, proprio)  # [B, C]
    assert float(jnp.max(jnp.abs(o_robottt))) > 0.0

    # Tanh workspace read (Q1): tanh(alpha) * P(omega_t) — a STATIC function of omega, no recurrence.
    tanh_cond = WSMTanhConditioner(w_dim=512, cond_dim=C, gate_init=1.0, rngs=nnx.Rngs(2))
    tanh_cond.proj_t_out.kernel[...] = 0.1 * jax.random.normal(
        jax.random.key(8), tanh_cond.proj_t_out.kernel[...].shape
    )
    omega = jax.random.normal(jax.random.key(10), (B, 512))
    v_tanh = tanh_cond(omega)  # [B, C]

    # Both ride the SAME adarms_cond seam; feed each vector through the robottt_cond input and compare
    # the resulting pi loss. Different conditioning vectors => different pi output => Q2 != tanh read.
    loss_robottt = model.compute_loss(jax.random.key(1), dataclasses.replace(obs, robottt_cond=o_robottt), act)
    loss_tanh = model.compute_loss(jax.random.key(1), dataclasses.replace(obs, robottt_cond=v_tanh), act)
    assert float(jnp.max(jnp.abs(o_robottt - v_tanh))) > 0.0, "the two reads must produce different vectors"
    assert abs(float(jnp.mean(loss_robottt)) - float(jnp.mean(loss_tanh))) > 1e-6, (
        "Q2 (RoboTTT fast-weight read) must not coincide with the tanh workspace read"
    )

    # The load-bearing distinction: RoboTTT's read is RECURRENT (changes after a commit even with the
    # same query input); the tanh read is a pure function of omega and cannot do this.
    o_before = model.robottt_fast.condition(w0, proprio)
    o_after = model.robottt_fast.condition(w1, proprio)
    assert float(jnp.max(jnp.abs(o_after - o_before))) > 0.0
    assert bool(jnp.array_equal(tanh_cond(omega), v_tanh))  # tanh read is stateless/deterministic


# ----- deliverable-3 guards: 2x2-from-one-dataclass + contiguous-episode windowing -----


def test_11_stage_q_arms_derive_only_two_flags_from_one_dataclass():
    from openpi.models.robottt_fast_weights import ALL_STAGE_Q_ARMS, StageQArms

    assert len(ALL_STAGE_Q_ARMS) == 4
    names = {a.name for a in ALL_STAGE_Q_ARMS}
    assert names == {"q0", "q1", "q2", "q3"}
    by = {a.name: a for a in ALL_STAGE_Q_ARMS}
    # Q2 is vanilla RoboTTT: fast weights on, workspace off — a config point.
    assert by["q2"] == StageQArms(fast_weights=True, workspace=False)
    # The ONLY per-arm knobs are the two Pi0Config flags; the whole 2x2 is the two-bool product.
    assert {tuple(sorted(a.pi0_flags().items())) for a in ALL_STAGE_Q_ARMS} == {
        (("robottt", fw), ("wsm_tanh", ws)) for fw in (False, True) for ws in (False, True)
    }
    for a in ALL_STAGE_Q_ARMS:
        assert set(a.pi0_flags()) == {"robottt", "wsm_tanh"}  # nothing else varies
    # Back-fill regex names the stable robottt_fast subtree exactly when fast weights are on.
    assert "robottt_fast" in by["q2"].missing_regex() and "wsm_tanh_cond" not in by["q2"].missing_regex()
    assert "robottt_fast" not in by["q0"].missing_regex() and "robottt_fast" not in by["q1"].missing_regex()
    assert "robottt_fast" in by["q3"].missing_regex() and "wsm_tanh_cond" in by["q3"].missing_regex()


def test_12_contiguous_episode_windows_never_cross_boundaries():
    gd = pytest.importorskip("openpi.groot_utils.groot_openpi_dataset", reason="gr00t/lerobot deps required")
    # Two episodes of length 20 and 10; window_len=4, chunk_stride=2 => chunk-steps at 0,2,4,...
    windows = gd.contiguous_episode_windows([20, 10], window_len=4, chunk_stride=2)
    for traj_index, base_indices in windows:
        assert len(base_indices) == 4
        assert all(bi < (20 if traj_index == 0 else 10) for bi in base_indices)  # within its episode
        assert base_indices == sorted(base_indices)  # contiguous, increasing
        assert base_indices[1] - base_indices[0] == 2  # respects chunk_stride
    # Every window is drawn from exactly one trajectory (never crosses a boundary).
    assert all(t in (0, 1) for t, _ in windows)
    # Too-short leftovers are dropped (fail-closed, no cross-episode padding).
    assert gd.contiguous_episode_windows([3], window_len=4, chunk_stride=1) == []


# ==================== SERVE-ONLY ABLATION KNOBS + A0 PROBE (tests 20-31) ====================
# The decisive Q0-vs-Q2 result (-3.7 pts) is attributed by perturbing ONLY the serve-time W
# lifecycle. Contract: with no ablation set, the serve path must be bit-identical to the code the
# decisive eval ran (test 26); every knob must do exactly the arithmetic it claims (22-25); and no
# ablated arm may run unacknowledged or unlabeled (21, 28, 29).

from vla_training.eval._robottt_ablation import (  # noqa: E402
    ProbeLogger,
    RoboTTTAblation,
    ablation_from_env,
    apply_post_commit,
    delta_norm,
    lerp_tree,
    parse_ablation,
)


def _ablated_wrapper(spec, *, probe=None, max_envs=1):
    """Workspace-free (Q2-shaped) wrapper carrying one parsed ablation spec."""
    policy = _Policy()
    runner = _FakeRoboTTTRunner()
    wrapper = WSMPiInferWrapper(
        policy,
        None,
        None,
        None,
        stride=8,
        max_envs=max_envs,
        robottt_runner=runner,
        robottt_ablation=parse_ablation(spec),
        robottt_probe=probe,
    )
    return wrapper, policy, runner


def _run_chunks(wrapper, n, env="env_a"):
    """n stride-aligned executed chunks on one episode; returns the per-chunk W scalar after each."""
    hs = []
    for i in range(n):
        wrapper.infer(_obs(env, "task_a", 1, 8 * i, 10 + i))
        hs.append(float(wrapper._states[env].robottt_w["h"]))
    return hs


def test_20_ablation_parsing_accepts_the_ladder_and_rejects_garbage():
    inert = parse_ablation("")
    assert (inert.active, inert.spec, inert.eta_scale, inert.changes_w) == (False, "", 1.0, False)
    assert parse_ablation(None).active is False and parse_ablation("   ").active is False

    assert parse_ablation("freeze").freeze is True
    assert parse_ablation("reset:7").reset_every == 7
    assert parse_ablation("decay:0.8").decay == 0.8
    assert parse_ablation("eta:0.3").eta_scale == 0.3
    assert parse_ablation("commitfirst").commit_first is True
    # commitfirst can never change the W trajectory (documented no-op at serve).
    assert parse_ablation("commitfirst").changes_w is False
    # Canonical spec is order-independent and is what lands in the metadata / probe lines.
    combo = parse_ablation(" eta:0.3 , decay:0.5 ,commitfirst ")
    assert combo.spec == "decay:0.5,eta:0.3,commitfirst"
    assert (combo.decay, combo.eta_scale, combo.commit_first) == (0.5, 0.3, True)
    assert combo.as_metadata()["spec"] == combo.spec
    # eta:0 is legal on purpose: the "commit ran but moved nothing" control (distinct from freeze).
    assert parse_ablation("eta:0").eta_scale == 0.0

    for bad in (
        "frezee",  # unknown token
        "reset",  # missing argument
        "reset:0",  # N must be >= 1
        "reset:-3",
        "reset:abc",
        "reset:2.5",
        "decay:1.0",  # G must be strictly inside (0, 1)
        "decay:0",
        "decay:-0.5",
        "decay:half",
        "eta:-1",
        "eta:nan",
        "eta",
        "freeze:1",  # takes no argument
        "commitfirst:yes",
        "freeze,reset:2",  # contradictory: no commit runs under freeze
        "freeze,decay:0.5",
        "freeze,eta:0.5",
        "reset:2,reset:3",  # duplicate
        "freeze,,decay:0.5",  # empty token
        "decay:0.5;eta:0.3",  # wrong separator => unknown token
    ):
        with pytest.raises(ValueError):
            parse_ablation(bad)


def test_21_ablation_requires_the_smoke_ack_and_is_inert_without_the_env():
    assert ablation_from_env({}).active is False
    assert ablation_from_env({"ROBOTTT_ABLATION": ""}).active is False  # the control arm
    # An ACK with no ablation stays the unmodified path.
    assert ablation_from_env({"ROBOTTT_ABLATION_ACK": "smoke"}).active is False
    for env in (
        {"ROBOTTT_ABLATION": "freeze"},
        {"ROBOTTT_ABLATION": "decay:0.5", "ROBOTTT_ABLATION_ACK": ""},
        {"ROBOTTT_ABLATION": "eta:0.3", "ROBOTTT_ABLATION_ACK": "yes"},
    ):
        with pytest.raises(RuntimeError, match="ROBOTTT_ABLATION_ACK"):
            ablation_from_env(env)
    ok = ablation_from_env({"ROBOTTT_ABLATION": "reset:7", "ROBOTTT_ABLATION_ACK": "smoke"})
    assert ok.reset_every == 7 and ok.spec == "reset:7"
    # Garbage still fails hard even with the ACK present.
    with pytest.raises(ValueError):
        ablation_from_env({"ROBOTTT_ABLATION": "nope", "ROBOTTT_ABLATION_ACK": "smoke"})


def test_22_freeze_holds_w_at_the_episode_init_all_episode():
    wrapper, policy, runner = _ablated_wrapper("freeze")
    hs = _run_chunks(wrapper, 4)
    assert hs == [0.0, 0.0, 0.0, 0.0]  # W never leaves its per-episode init
    assert runner.commit_calls == []  # freeze never even calls commit
    assert runner.init_calls == 1
    assert wrapper._states["env_a"].robottt_commits == 4  # executed chunks still counted
    # The policy is still conditioned every chunk -- on the CONSTANT W_0 read.
    assert len(policy.seen_cond) == 4
    assert all(float(np.asarray(c).max()) == 0.0 for c in policy.seen_cond)
    # Model-space rows are still consumed/stripped under freeze (contract is mode-independent).
    out = wrapper.infer(_obs("env_a", "task_a", 1, 32, 20))
    assert "norm_state" not in out and "norm_actions" not in out
    assert wrapper.metadata["robottt_ablation"] == "freeze"


def test_23_reset_n_returns_w_to_a_fresh_init_every_n_commits():
    wrapper, policy, runner = _ablated_wrapper("reset:2")
    # fake commit adds +2.0 per chunk: 2 -> (4 -> reset 0) -> 2 -> (4 -> reset 0)
    assert _run_chunks(wrapper, 4) == [2.0, 0.0, 2.0, 0.0]
    fresh = runner.init_state()
    assert wrapper._states["env_a"].robottt_w["h"] == fresh["h"]  # equals a FRESH init
    assert len(runner.commit_calls) == 4  # the commit itself still runs every chunk
    # Conditioning sees the reset immediately on the following chunk.
    assert [float(np.asarray(c).max()) for c in policy.seen_cond] == [0.0, 2.0, 0.0, 2.0]
    # reset:1 == "one commit of memory then back to init"
    w1, _p1, _r1 = _ablated_wrapper("reset:1")
    assert _run_chunks(w1, 3) == [0.0, 0.0, 0.0]


def test_24_decay_moves_w_halfway_back_toward_the_episode_init():
    wrapper, _policy, _runner = _ablated_wrapper("decay:0.5")
    # w0=0; commit adds +2.0, then W <- w0 + 0.5*(W_committed - w0):
    #   (0+2)*0.5 = 1.0 ; (1+2)*0.5 = 1.5 ; (1.5+2)*0.5 = 1.75
    assert _run_chunks(wrapper, 3) == [1.0, 1.5, 1.75]

    # Same arithmetic, checked directly on a nested toy pytree against manual math.
    w0 = {"a": np.asarray([1.0, -2.0], dtype=np.float32), "b": {"c": np.asarray([4.0], np.float32)}}
    committed = {
        "a": np.asarray([3.0, 2.0], dtype=np.float32),
        "b": {"c": np.asarray([0.0], np.float32)},
    }
    got, ops = apply_post_commit(parse_ablation("decay:0.5"), w0, committed, w0, 1)
    assert ops == ("decay:0.5",)
    np.testing.assert_allclose(np.asarray(got["a"]), [2.0, 0.0])  # 1+0.5*2, -2+0.5*4
    np.testing.assert_allclose(np.asarray(got["b"]["c"]), [2.0])  # 4+0.5*(0-4)
    # G -> 1 is the identity; G -> 0 would be a full snap-back (rejected by the parser as degenerate).
    near_one, _ = apply_post_commit(parse_ablation("decay:0.999999"), w0, committed, w0, 1)
    np.testing.assert_allclose(np.asarray(near_one["a"]), np.asarray(committed["a"]), atol=1e-5)
    # lerp_tree is the single primitive behind both eta and decay.
    np.testing.assert_allclose(np.asarray(lerp_tree(w0, committed, 0.25)["a"]), [1.5, -1.0])


def test_25_eta_scale_rescales_the_step_and_eta_zero_is_a_no_op():
    zero, _policy, runner = _ablated_wrapper("eta:0")
    assert _run_chunks(zero, 3) == [0.0, 0.0, 0.0]  # commit moves W nowhere
    assert len(runner.commit_calls) == 3  # ... but the commit DID run (unlike freeze)

    half, _policy_h, _runner_h = _ablated_wrapper("eta:0.5")
    # entering W + 0.5*(committed - entering): 0+0.5*2=1 ; 1+0.5*2=2 ; 2+0.5*2=3
    assert _run_chunks(half, 3) == [1.0, 2.0, 3.0]

    # Toy pytree: the scale multiplies the STEP, measured from the ENTERING weights.
    entering = {"a": np.asarray([1.0, 1.0], dtype=np.float32)}
    committed = {"a": np.asarray([3.0, -1.0], dtype=np.float32)}
    got, ops = apply_post_commit(parse_ablation("eta:0.25"), entering, committed, entering, 1)
    assert ops == ("eta:0.25",)
    np.testing.assert_allclose(np.asarray(got["a"]), [1.5, 0.5])


def test_29_ablation_is_refused_without_a_runner_and_defaults_to_inert():
    with pytest.raises(ValueError, match="no RoboTTT runner"):
        WSMPiInferWrapper(
            _Policy(),
            _Tap(),
            _Conditioner(),
            _TABLE,
            stride=8,
            max_envs=1,
            conditioner_factory=_Conditioner,
            robottt_ablation=parse_ablation("freeze"),
        )
    # Default construction (every pre-existing caller) is inert and says so in its metadata.
    plain = WSMPiInferWrapper(_Policy(), None, None, None, stride=8, max_envs=1, robottt_runner=_FakeRoboTTTRunner())
    assert plain.metadata["robottt_ablation"] == ""
    assert plain.metadata["robottt_ablation_detail"]["freeze"] is False
    assert plain.metadata["robottt_probe_log"] is None
    assert RoboTTTAblation().active is False


def test_28_commitfirst_is_a_documented_no_op_and_is_still_self_describing():
    """A5/G7 cannot apply at serve: _commit_robottt runs at the END of the infer call that produced
    the chunk, strictly before the next request is conditioned in _prepare_batch, so there is never
    a pending uncommitted chunk to hoist. The flag is accepted, announced, and inert."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wrapper, _policy, runner = _ablated_wrapper("commitfirst")
    printed = buf.getvalue()
    assert "commitfirst REQUESTED BUT IS A NO-OP" in printed
    hs = _run_chunks(wrapper, 3)
    baseline, _bp, _br = _ablated_wrapper("")
    assert hs == _run_chunks(baseline, 3) == [2.0, 4.0, 6.0]
    assert len(runner.commit_calls) == 3
    assert wrapper.metadata["robottt_ablation"] == "commitfirst"
    assert wrapper.metadata["robottt_ablation_detail"]["commit_first"] is True


def test_27_probe_log_emits_one_json_line_per_reset_and_per_commit():
    import json
    import tempfile

    path = os.path.join(tempfile.mkdtemp(prefix="robottt_probe_"), "probe.jsonl")
    probe = ProbeLogger(path)
    wrapper, _policy, _runner = _ablated_wrapper("", probe=probe)
    _run_chunks(wrapper, 3)  # one reset + three commits
    wrapper.infer(_obs("env_a", "task_a", 1, 0, 99))  # second episode on the same env slot
    _run_chunks(wrapper, 0)
    probe.close()

    lines = [json.loads(ln) for ln in open(path) if ln.strip()]
    kinds = [rec["kind"] for rec in lines]
    assert kinds == ["reset", "commit", "commit", "commit", "reset", "commit"]
    required = {
        "kind",
        "env_id",
        "commit_idx",
        "w_delta_norm",
        "o_norm",
        "eta_effective",
        "wall_ms",
        "ts",
    }
    assert all(required <= set(rec) for rec in lines)
    commits = [rec for rec in lines if rec["kind"] == "commit"]
    assert [rec["commit_idx"] for rec in commits] == [1, 2, 3, 1]  # per-episode, 1-based
    # ||W - W_episode_init|| grows by the fake commit step (+2.0) and restarts with the episode.
    assert [rec["w_delta_norm"] for rec in commits] == [2.0, 4.0, 6.0, 2.0]
    # o_norm is the L2 of the O_t the wrapper ALREADY holds for the just-conditioned step
    # (fake runner: O_t = [h]*4 => ||O_t|| = 2|h|), so it costs no extra forward pass.
    assert [rec["o_norm"] for rec in commits] == [0.0, 4.0, 8.0, 0.0]
    assert all(rec["env_id"] == "env_a" and rec["ablation"] == "" for rec in lines)
    assert all(rec["wall_ms"] >= 0.0 for rec in lines)
    assert lines[0]["commit_idx"] == 0 and lines[0]["w_delta_norm"] == 0.0
    # The fake runner exposes no inner_lr, so eta_effective is null rather than invented.
    assert all(rec["eta_effective"] is None for rec in lines)
    # No probe configured => no probe cost and no file handle.
    quiet, _qp, _qr = _ablated_wrapper("")
    assert quiet.metadata["robottt_probe_log"] is None

    # delta_norm is a real global L2 over the whole pytree.
    a = {"x": np.asarray([3.0, 0.0], np.float32), "y": {"z": np.asarray([4.0], np.float32)}}
    b = {"x": np.asarray([0.0, 0.0], np.float32), "y": {"z": np.asarray([0.0], np.float32)}}
    assert abs(delta_norm(a, b) - 5.0) < 1e-6


# ------------------------- real-runner regression + eta identity (JAX) -------------------------


class _RowPolicy:
    """Policy facade returning FIXED model-space rows so the whole commit chain is determined."""

    metadata = {}

    def __init__(self, module, states, actions):
        from types import SimpleNamespace

        self._model = SimpleNamespace(robottt=True, robottt_fast=module)
        self._input_transform = lambda x: {"state": np.asarray(x["observation/state"], dtype=np.float32)}
        self._states = states
        self._actions = actions
        self.n = 0
        self.seen_cond = []

    def infer(self, obs, **kwargs):
        i = self.n
        self.n += 1
        self.seen_cond.append(np.asarray(obs["robottt_cond"]).copy())
        return {
            "actions": np.zeros((1, 1), dtype=np.float32),
            "norm_state": self._states[i],
            "norm_actions": self._actions[i],
        }


def _row_obs(env, t, state_row):
    return {
        "request_id": f"{env}:{t}",
        "observation/state": np.asarray(state_row, dtype=np.float32),
        "wsm_env_id": env,
        "wsm_task": "task_a",
        "wsm_demo_episode": 1,
        "wsm_t": int(t),
    }


def test_26_default_path_w_trajectory_is_bit_identical_to_the_unablated_runner():
    """Regression vs the decisive-eval behavior: with no ablation (None, "" or the no-op
    commitfirst), the wrapper's per-chunk condition/commit chain reproduces the bare
    RoboTTTServeRunner sequence LEAF-FOR-LEAF, bitwise. Same fixtures as the train/serve parity
    test 16, so the whole chain from run_sequence through serve stays pinned."""
    from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner

    m = _module()
    _detrain(m)
    cfg = m.cfg
    s, a = _seq(1, cfg, seed=5)
    states = [np.asarray(s[0, t]) for t in range(cfg.window_len)]
    actions = [np.asarray(a[0, t]) for t in range(cfg.window_len)]

    # Reference: the runner alone, exactly as test 16 drives it (no wrapper, no ablation code).
    ref_runner = RoboTTTServeRunner(_RowPolicy(m, states, actions))
    w_ref = ref_runner.init_state()
    conds_ref = []
    for t in range(cfg.window_len):
        conds_ref.append(ref_runner.condition(w_ref, _row_obs("env_a", 8 * t, states[t])))
        w_ref = ref_runner.commit(w_ref, states[t], actions[t])

    for spec in (None, "", "commitfirst"):
        policy = _RowPolicy(m, states, actions)
        wrapper = WSMPiInferWrapper(
            policy,
            None,
            None,
            None,
            stride=8,
            max_envs=1,
            robottt_runner=RoboTTTServeRunner(policy),
            robottt_ablation=None if spec is None else parse_ablation(spec),
        )
        for t in range(cfg.window_len):
            wrapper.infer(_row_obs("env_a", 8 * t, states[t]))
        w_got = wrapper._states["env_a"].robottt_w
        assert set(w_got) == set(w_ref)
        for name in w_ref:
            np.testing.assert_array_equal(
                np.asarray(w_got[name]),
                np.asarray(w_ref[name]),
                err_msg=f"W leaf {name!r} drifted under ablation spec {spec!r}",
            )
        for t, (got, ref) in enumerate(zip(policy.seen_cond, conds_ref)):
            np.testing.assert_array_equal(got, np.asarray(ref), err_msg=f"O_t drift at chunk {t}")


def test_30_eta_scale_equals_a_genuinely_smaller_inner_lr():
    """`eta:X` must be the SAME arithmetic as training with X-times the inner LR — not an
    approximation. One inner GD step is W - eta*grad(W), so W + X*(commit(W) - W) is exact. Checked
    against a module whose cfg.base_inner_lr really is scaled (identical params, same seeds)."""
    scale = 0.3
    m = _module(seed=0)
    _detrain(m)
    m_scaled = _module(seed=0, base_inner_lr=_cfg().base_inner_lr * scale)
    _detrain(m_scaled)
    assert float(m_scaled.inner_lr()) == pytest.approx(float(m.inner_lr()) * scale, rel=1e-6)

    s, a = _seq(1, m.cfg, seed=5)
    w0 = m.init_state(1)
    w_full = m.commit(w0, s[:, 0], a[:, 0])
    w_ref = m_scaled.commit(w0, s[:, 0], a[:, 0])  # a genuinely smaller inner LR
    w_eta, ops = apply_post_commit(parse_ablation(f"eta:{scale}"), w0, w_full, w0, 1)
    assert ops == (f"eta:{scale:g}",)
    for name in w_ref:
        np.testing.assert_allclose(
            np.asarray(w_eta[name]),
            np.asarray(w_ref[name]),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"eta-scaled leaf {name!r} != a genuinely scaled inner LR",
        )
    # eta:0 is exactly the entering W (no step at all), leaf for leaf.
    w_zero, _ = apply_post_commit(parse_ablation("eta:0"), w0, w_full, w0, 1)
    for name in w_zero:
        np.testing.assert_array_equal(np.asarray(w_zero[name]), np.asarray(w0[name]))
    # ... and it is NOT the same as the full step (the fixture is not degenerate).
    assert float(np.max(np.abs(np.asarray(w_full["w1"]) - np.asarray(w0["w1"])))) > 0.0


def test_31_runner_reports_the_trained_scalars_for_the_startup_probe():
    """A0 startup line: eta and the tanh-gate magnitude answer 'did the model train its updates
    toward zero?'. Read-only — reading them must not perturb the module."""
    from types import SimpleNamespace

    from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner

    m = _module()
    _detrain(m)
    runner = RoboTTTServeRunner(
        SimpleNamespace(_model=SimpleNamespace(robottt=True, robottt_fast=m), _input_transform=lambda x: x)
    )
    scalars = runner.trained_scalars()
    assert set(scalars) >= {
        "inner_lr_eta",
        "base_inner_lr",
        "softplus_log_inner_lr",
        "alpha_max_abs_tanh",
        "alpha_mean_abs_tanh",
        "alpha_dim",
    }
    assert scalars["inner_lr_eta"] == pytest.approx(float(m.inner_lr()))
    assert runner.inner_lr() == pytest.approx(float(m.inner_lr()))
    assert scalars["alpha_dim"] == m.cfg.cond_dim
    expected_gate = float(jnp.max(jnp.abs(jnp.tanh(m.alpha[...]))))
    assert scalars["alpha_max_abs_tanh"] == pytest.approx(expected_gate)
    assert 0.0 <= scalars["alpha_mean_abs_tanh"] <= scalars["alpha_max_abs_tanh"]
    # Untrained init: the gate is exactly gate_init everywhere (the "trained toward zero?" baseline).
    assert scalars["alpha_max_abs_tanh"] == pytest.approx(float(np.tanh(m.cfg.gate_init)), rel=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} PASSED")
