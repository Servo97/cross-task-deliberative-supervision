"""Serve-side RoboTTT fast-weight runner (Q2/Q3): the normalized-space online update.

Train/serve parity is the whole game. During Stage-Q training the fast-weight chain consumed the
MODEL-SPACE rows — `observation.state` ([32], normalized) and the H=50 action chunk ([50, 32],
normalized) — so the serve-side condition/commit must consume byte-for-byte the same
representation, never the raw robot-space request/response:

- condition(W, obs): the ENTERING state is produced by running the policy's OWN input-transform
  chain (repack -> robocasa inputs -> Normalize -> model transforms) over the raw request, then
  reading `state`. Same functions the policy applies internally one call later.
- commit(W, norm_state, norm_actions): the executed chunk arrives as the policy result's
  `norm_state`/`norm_actions` rows (Policy(expose_norm_actions=True) captures them before
  Unnormalize). The wrapper extracts them fail-closed; this runner never sees robot-space actions.

W stays a per-env JAX pytree between calls (no host round-trip); the wrapper owns its lifecycle
(init at t=0, one condition per chunk, one commit per executed chunk).
"""

from __future__ import annotations

import numpy as np


class RoboTTTServeRunner:
    """Wraps the served model's `robottt_fast` module for the WSMPiInferWrapper protocol."""

    # Stripped before the input-transform pass: server-protocol signal keys the policy transform
    # chain never sees (the wrapper strips the same set before policy.infer).
    _SIGNAL_KEYS = ("wsm_env_id", "wsm_task", "wsm_demo_episode", "wsm_t", "wsm_prompt")

    def __init__(self, policy):
        import jax
        import jax.numpy as jnp
        from openpi.shared import nnx_utils

        self._jax = jax
        self._jnp = jnp
        model = policy._model
        if not getattr(model, "robottt", False) or not hasattr(model, "robottt_fast"):
            raise RuntimeError(
                "[robottt-serve] served model has no robottt_fast subtree; build the policy config "
                "with robottt=True (Q2/Q3 checkpoint)"
            )
        self._module = model.robottt_fast
        self._input_transform = policy._input_transform
        # jit the two per-chunk calls; shapes are fixed ([1, 32] / [1, H, 32]) so each compiles once.
        self._condition_fn = nnx_utils.module_jit(self._module.condition)
        self._commit_fn = nnx_utils.module_jit(self._module.commit)

    def init_state(self):
        """Fresh per-episode W from the meta-learned init (batch of one)."""
        return self._module.init_state(1)

    def _model_state(self, obs: dict) -> "np.ndarray":
        clean = {k: v for k, v in obs.items() if k not in self._SIGNAL_KEYS}
        inputs = self._input_transform(self._jax.tree.map(lambda x: x, clean))
        return np.asarray(inputs["state"], dtype=np.float32)

    def condition(self, w, obs: dict) -> np.ndarray:
        """O_t from the HELD entering W and the current normalized state; [cond_dim] numpy row."""
        state = self._jnp.asarray(self._model_state(obs))[None]
        return np.asarray(self._condition_fn(w, state)[0], dtype=np.float32)

    def commit(self, w, norm_state: np.ndarray, norm_actions: np.ndarray):
        """One inner-GD step from the executed chunk's model-space rows; returns the next W."""
        state = self._jnp.asarray(np.asarray(norm_state, dtype=np.float32))[None]
        actions = self._jnp.asarray(np.asarray(norm_actions, dtype=np.float32))[None]
        return self._commit_fn(w, state, actions)

    def is_finite(self, w) -> bool:
        return all(bool(self._jnp.isfinite(leaf).all()) for leaf in self._jax.tree.leaves(w))

    # ---------------------------------- read-only instrumentation ----------------------------------
    def inner_lr(self) -> float:
        """The TRAINED effective inner-GD step eta = base_inner_lr * softplus(log_inner_lr).

        Read-only: nothing here (or in the ablation layer) writes a trained parameter. Serve-time
        eta scaling is applied as a pytree recombination of the entering/committed W — see
        `_robottt_ablation.apply_post_commit` for the exact identity.
        """
        return float(self._module.inner_lr())

    def trained_scalars(self) -> dict:
        """The two numbers that answer 'did the model train its updates toward zero?'.

        `inner_lr_eta` is the learned step size; `alpha_max_abs_tanh` is the largest gate magnitude
        over the cond_dim subtree (the multiplier on O_t, Eq 3). Both are logged at startup so every
        probe file is self-describing about the checkpoint it came from.
        """
        jnp = self._jnp
        module = self._module
        alpha = jnp.asarray(module.alpha[...])
        gate = jnp.abs(jnp.tanh(alpha))
        cfg = module.cfg
        return {
            "inner_lr_eta": float(module.inner_lr()),
            "base_inner_lr": float(cfg.base_inner_lr),
            "softplus_log_inner_lr": float(module.inner_lr()) / float(cfg.base_inner_lr)
            if float(cfg.base_inner_lr)
            else float("nan"),
            "alpha_max_abs_tanh": float(jnp.max(gate)),
            "alpha_mean_abs_tanh": float(jnp.mean(gate)),
            "alpha_dim": int(alpha.size),
            "alpha_scale": float(cfg.alpha_scale),
        }
