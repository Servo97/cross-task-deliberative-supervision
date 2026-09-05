"""Execution-only RoboTTT lifecycle for the RoboMME evaluation server.

This module is intentionally isolated from the RoboCasa serving stack.  It keeps one fast-weight
state ``W`` per RoboMME session, conditions on the entering state, and commits only the ten actions
that the synchronous harness has finished executing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np


def audit_execution_robottt_checkpoint(policy) -> dict[str, Any]:
    """Fail closed unless an execution-only Q2 checkpoint restored trained finite fast weights."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    fast = getattr(model, "robottt_fast", None)
    if not getattr(model, "robottt", False) or fast is None:
        raise RuntimeError("Q2 policy has no robottt_fast subtree")
    if any(not hasattr(fast, name) for name in ("condition", "commit", "init_state")):
        raise RuntimeError("Q2 policy does not expose the execution-only fast-weight contract")

    flat = list(nnx.state(fast, nnx.Param).flat_state())
    arrays = [(path, jnp.asarray(variable[...])) for path, variable in flat]
    bad = [path for path, value in arrays if not bool(jnp.isfinite(value).all())]
    if bad:
        raise RuntimeError(f"Q2 robottt_fast has {len(bad)} non-finite parameter tensors")

    alpha = jnp.asarray(fast.alpha[...])
    gate_init = float(fast.cfg.gate_init)
    if bool(jnp.all(alpha == gate_init)):
        raise RuntimeError(
            "Q2 robottt_fast.alpha is still exactly at initialization; the trained subtree was not restored"
        )
    summary = {
        "param_tensors": len(arrays),
        "inner_lr": float(fast.inner_lr()),
        "gate_max_abs_tanh": float(jnp.max(jnp.abs(jnp.tanh(alpha)))),
        "gate_mean_abs_tanh": float(jnp.mean(jnp.abs(jnp.tanh(alpha)))),
    }
    if not np.isfinite(np.asarray(list(summary.values()), dtype=np.float64)).all():
        raise RuntimeError(f"Q2 checkpoint audit produced non-finite summary: {summary}")
    return summary


@dataclasses.dataclass(frozen=True)
class PendingExecutionCommit:
    """Normalized model-space rows held until their action block has completed."""

    state: np.ndarray
    actions: np.ndarray


class ExecutionRoboTTTRunner:
    """Batched condition/commit operations over independent per-episode fast weights."""

    def __init__(self, policy, *, execution_commit_steps: int = 10):
        import jax
        import jax.numpy as jnp
        from openpi.shared import nnx_utils

        model = policy._model
        fast = getattr(model, "robottt_fast", None)
        if fast is None or any(not hasattr(fast, name) for name in ("condition", "commit", "init_state")):
            raise RuntimeError("checkpoint/model lacks execution-only robottt_fast methods")
        if not 1 <= execution_commit_steps <= model.action_horizon:
            raise ValueError(
                "execution_commit_steps must lie within the policy action horizon, got "
                f"{execution_commit_steps} for horizon {model.action_horizon}"
            )

        self._jax = jax
        self._jnp = jnp
        self._fast = fast
        self._input_transform = policy._input_transform
        self.execution_commit_steps = int(execution_commit_steps)
        self._condition = nnx_utils.module_jit(fast.condition)
        self._commit = nnx_utils.module_jit(fast.commit)

    @staticmethod
    def _bucket_size(rows: int) -> int:
        if not 1 <= rows <= 8:
            raise ValueError(f"RoboMME Q2 batch must contain 1--8 rows, got {rows}")
        return 4 if rows <= 4 else 8

    def init_state(self):
        return self._fast.init_state(1)

    def _model_state(self, raw: dict) -> np.ndarray:
        transformed = self._input_transform(self._jax.tree.map(lambda value: value, raw))
        return np.asarray(transformed["state"], dtype=np.float32)

    def condition_many(self, states: Sequence, raw_observations: Sequence[dict]) -> np.ndarray:
        """Return one condition vector per row using the held entering ``W``."""
        if not states or len(states) != len(raw_observations):
            raise ValueError("states and raw_observations must have the same nonzero length")
        rows = len(states)
        bucket = self._bucket_size(rows)
        padded_states = [*states, *([states[-1]] * (bucket - rows))]
        padded_raw = [*raw_observations, *([raw_observations[-1]] * (bucket - rows))]
        w_batch = self._jax.tree.map(
            lambda *values: self._jnp.concatenate(values, axis=0),
            *padded_states,
        )
        model_states = self._jnp.asarray(np.stack([self._model_state(raw) for raw in padded_raw]))
        return np.asarray(self._condition(w_batch, model_states), dtype=np.float32)[:rows]

    def commit_many(
        self,
        states: Sequence,
        pending: Sequence[PendingExecutionCommit],
    ) -> list:
        """Commit completed chunks together, returning independent batch-one pytrees."""
        if not states or len(states) != len(pending):
            raise ValueError("states and pending commits must have the same nonzero length")
        rows = len(states)
        bucket = self._bucket_size(rows)
        padded_states = [*states, *([states[-1]] * (bucket - rows))]
        padded_pending = [*pending, *([pending[-1]] * (bucket - rows))]
        w_batch = self._jax.tree.map(
            lambda *values: self._jnp.concatenate(values, axis=0),
            *padded_states,
        )
        model_states = self._jnp.asarray(
            np.stack([np.asarray(item.state, dtype=np.float32) for item in padded_pending])
        )
        actions = self._jnp.asarray(np.stack([np.asarray(item.actions, dtype=np.float32) for item in padded_pending]))
        if actions.shape[1] != self.execution_commit_steps:
            raise ValueError(f"Q2 commit expected {self.execution_commit_steps} executed rows, got {actions.shape[1]}")
        next_batch = self._commit(w_batch, model_states, actions)
        if not all(bool(self._jnp.isfinite(leaf).all()) for leaf in self._jax.tree.leaves(next_batch)):
            raise FloatingPointError("Q2 fast weights became non-finite")
        return [
            self._jax.tree.map(lambda value, index=index: value[index : index + 1], next_batch)
            for index in range(rows)
        ]

    def delta_norm(self, state, initial_state) -> float:
        total = sum(
            self._jnp.sum(self._jnp.square(current - initial))
            for current, initial in zip(
                self._jax.tree.leaves(state),
                self._jax.tree.leaves(initial_state),
                strict=True,
            )
        )
        return float(self._jnp.sqrt(total))
