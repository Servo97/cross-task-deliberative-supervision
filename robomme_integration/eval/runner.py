"""Per-session demo-prefill and execution lifecycle for Q2V."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from robomme_integration.sequence import uniformly_sample_prefix
except ModuleNotFoundError as error:
    if error.name != "robomme_integration":
        raise
    from sequence import uniformly_sample_prefix


def audit_demo_robottt_checkpoint(policy) -> dict[str, Any]:
    """Fail closed unless the restored Q2V subtree is trained, complete, and finite."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    fast = getattr(model, "robottt_fast", None)
    if not getattr(model, "robottt", False) or fast is None:
        raise RuntimeError("Q2V policy has no robottt_fast subtree")
    required_methods = ("condition_visual", "commit_context", "commit_visual")
    if any(not hasattr(fast, name) for name in required_methods):
        raise RuntimeError("Q2V policy restored the execution-only fast-weight class")

    flat = list(nnx.state(fast, nnx.Param).flat_state())
    arrays = [(path, jnp.asarray(variable[...])) for path, variable in flat]
    bad = [path for path, value in arrays if not bool(jnp.isfinite(value).all())]
    if bad:
        raise RuntimeError(f"Q2V robottt_fast has {len(bad)} non-finite parameter tensors")
    roots = {str(path[0]) for path, _ in arrays if path}
    required_roots = {
        "context_in",
        "context_q",
        "context_k",
        "context_v",
        "context_out",
    }
    missing = sorted(required_roots - roots)
    if missing:
        raise RuntimeError(f"Q2V checkpoint lacks context parameter roots: {missing}")

    alpha = jnp.asarray(fast.alpha[...])
    gate_init = float(fast.cfg.gate_init)
    if bool(jnp.all(alpha == gate_init)):
        raise RuntimeError(
            "Q2V robottt_fast.alpha is still exactly at initialization; the trained subtree was not restored"
        )
    eta = float(fast.inner_lr())
    summary = {
        "param_tensors": len(arrays),
        "inner_lr": eta,
        "gate_max_abs_tanh": float(jnp.max(jnp.abs(jnp.tanh(alpha)))),
        "gate_mean_abs_tanh": float(jnp.mean(jnp.abs(jnp.tanh(alpha)))),
    }
    if not np.isfinite(np.asarray(list(summary.values()), dtype=np.float64)).all():
        raise RuntimeError(f"Q2V checkpoint audit produced non-finite summary: {summary}")
    return summary


class DemoRoboTTTRunner:
    """Run model-space Q2V updates while keeping ``W`` outside the slow model."""

    def __init__(
        self,
        policy,
        *,
        demo_frames: int = 16,
        execution_commit_steps: int = 10,
    ):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.shared import nnx_utils

        model = policy._model
        fast = getattr(model, "robottt_fast", None)
        required = (
            "condition_visual",
            "commit_context",
            "commit_visual",
        )
        if fast is None or any(not hasattr(fast, name) for name in required):
            raise RuntimeError("checkpoint/model lacks demo-capable robottt_fast methods")
        if not hasattr(model, "robottt_context_tokens"):
            raise RuntimeError("Pi model lacks the demo-context side tap")
        if demo_frames < 1:
            raise ValueError("demo_frames must be positive")
        if not 1 <= execution_commit_steps <= model.action_horizon:
            raise ValueError(
                "execution_commit_steps must lie in [1, action horizon], got "
                f"{execution_commit_steps} for horizon {model.action_horizon}"
            )

        self._jax = jax
        self._jnp = jnp
        self._model_api = _model
        self._policy = policy
        self._fast = fast
        self._input_transform = policy._input_transform
        self.demo_frames = int(demo_frames)
        self.execution_commit_steps = int(execution_commit_steps)
        self._encode = nnx_utils.module_jit(model.robottt_context_tokens)
        self._condition = nnx_utils.module_jit(fast.condition_visual)
        self._commit_context = nnx_utils.module_jit(fast.commit_context)
        self._commit_visual = nnx_utils.module_jit(fast.commit_visual)

    def init_state(self):
        return self._fast.init_state(1)

    def _transform_one(self, raw: dict) -> dict:
        return self._input_transform(self._jax.tree.map(lambda value: value, raw))

    def _observation_batch(self, rows: Sequence[dict]):
        transformed = [self._transform_one(row) for row in rows]
        batch = self._jax.tree.map(
            lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
            *transformed,
        )
        return self._model_api.Observation.from_dict(batch), batch

    @staticmethod
    def _raw_observation(
        image,
        state,
        prompt: str,
        wrist_image=None,
    ) -> dict:
        image = np.asarray(image, dtype=np.uint8)
        wrist = np.zeros_like(image) if wrist_image is None else np.asarray(wrist_image, dtype=np.uint8)
        return {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(prompt).lower(),
        }

    def prefill(
        self,
        w,
        *,
        video_history: Sequence,
        state_history: Sequence,
        prompt: str,
        wrist_history: Sequence | None = None,
    ):
        """Uniformly sample and ingest the observation-only demonstration prefix."""
        if len(video_history) != len(state_history):
            raise ValueError(f"video/state history lengths differ: {len(video_history)} vs {len(state_history)}")
        if wrist_history is not None and len(wrist_history) != len(video_history):
            raise ValueError(f"wrist/video history lengths differ: {len(wrist_history)} vs {len(video_history)}")
        if not video_history:
            return w

        selected, valid = uniformly_sample_prefix(
            range(len(video_history)),
            self.demo_frames,
            pad_index=0,
        )
        rows = [
            self._raw_observation(
                video_history[index],
                state_history[index],
                prompt,
                None if wrist_history is None else wrist_history[index],
            )
            for index in selected
        ]
        observation, transformed = self._observation_batch(rows)
        tokens, token_mask = self._encode(observation)
        states = self._jnp.asarray(transformed["state"])
        for index, is_valid in enumerate(valid):
            if is_valid:
                w = self._commit_context(
                    w,
                    states[index : index + 1],
                    tokens[index : index + 1],
                    token_mask[index : index + 1],
                )
        return w

    def infer_batch(self, states: Sequence, raw_observations: Sequence[dict]):
        """Batch the expensive Pi read while preserving one independent ``W`` per row."""
        if not states or len(states) != len(raw_observations):
            raise ValueError("states and raw_observations must have the same nonzero length")
        rows = len(states)
        bucket = 4 if rows <= 4 else 8
        if rows > bucket:
            raise ValueError(f"Q2V inference batch exceeds 8 rows: {rows}")
        # Match OpenPI's infer_batch buckets so the context encoder and fast read compile only for
        # batch 4/8, rather than once for every opportunistic server batch size.
        padded_observations = [
            *raw_observations,
            *([raw_observations[-1]] * (bucket - rows)),
        ]
        padded_states = [*states, *([states[-1]] * (bucket - rows))]
        observation, transformed = self._observation_batch(padded_observations)
        tokens, token_mask = self._encode(observation)
        state = self._jnp.asarray(transformed["state"])
        w_batch = self._jax.tree.map(
            lambda *values: self._jnp.concatenate(values, axis=0),
            *padded_states,
        )
        condition = np.asarray(
            self._condition(w_batch, state, tokens, token_mask),
            dtype=np.float32,
        )[:rows]

        requests = []
        for raw, cond in zip(raw_observations, condition, strict=True):
            request = dict(raw)
            request["robottt_cond"] = cond
            requests.append(request)
        results = self._policy.infer_batch(requests)

        public_results = []
        pending_commits = []
        for index, result in enumerate(results):
            if "norm_state" not in result or "norm_actions" not in result:
                raise RuntimeError("Q2V policy must be created with expose_norm_actions=True")
            norm_state = self._jnp.asarray(np.asarray(result["norm_state"], dtype=np.float32))[None]
            norm_actions = self._jnp.asarray(np.asarray(result["norm_actions"], dtype=np.float32))[
                None, : self.execution_commit_steps
            ]
            pending_commits.append(
                PendingFastCommit(
                    state=norm_state,
                    actions=norm_actions,
                    context_tokens=tokens[index : index + 1],
                    context_token_mask=token_mask[index : index + 1],
                )
            )
            public_results.append(
                {key: value for key, value in result.items() if key not in {"norm_state", "norm_actions"}}
            )
        return public_results, pending_commits

    def infer(self, w, raw_observation: dict):
        """Single-row wrapper used by local probes and max-batch-one serving."""
        results, pending = self.infer_batch([w], [raw_observation])
        return results[0], pending[0]

    def commit(self, w, pending: PendingFastCommit):
        """Commit a block only when the server has observed that the next replan was reached."""
        w_next = self._commit_visual(
            w,
            pending.state,
            pending.actions,
            pending.context_tokens,
            pending.context_token_mask,
        )
        if not all(bool(self._jnp.isfinite(leaf).all()) for leaf in self._jax.tree.leaves(w_next)):
            raise FloatingPointError("Q2V fast weights became non-finite")
        return w_next

    def delta_norm(self, w, w0) -> float:
        total = sum(
            self._jnp.sum(self._jnp.square(a - b))
            for a, b in zip(
                self._jax.tree.leaves(w),
                self._jax.tree.leaves(w0),
                strict=True,
            )
        )
        return float(self._jnp.sqrt(total))


@dataclasses.dataclass
class PendingFastCommit:
    """Model-space proposal held until its action block has actually completed."""

    state: object
    actions: object
    context_tokens: object
    context_token_mask: object


@dataclasses.dataclass
class EpisodeFastState:
    """Observable per-session state owned by the harness model server."""

    w: object
    w0: object
    pending: PendingFastCommit | None = None
    prefills: int = 0
    execution_commits: int = 0
