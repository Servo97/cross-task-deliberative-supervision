"""Session-isolated, batched inference for the pinned official RoboMME recurrent TTT."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import numpy as np


@dataclasses.dataclass
class UpstreamTTTSession:
    """All mutable recurrent state owned by one RoboMME episode."""

    task_name: str
    episode_idx: int
    pending_images: list[np.ndarray] = dataclasses.field(default_factory=list)
    pending_states: list[np.ndarray] = dataclasses.field(default_factory=list)
    pending_exec_start_idx: int | None = None
    memory: Any | None = None
    step_idx: int = -1
    exec_start_idx: int = 0
    plans: int = 0
    dense_observations: int = 0
    saw_video_history: bool = False


def _raw_observation(obs: dict[str, Any], state_dim: int) -> dict[str, Any]:
    images = obs.get("images", {})
    if not isinstance(images, dict) or not images:
        raise ValueError("RoboMME T1 observation has no image dictionary")
    front = images.get("agentview", next(iter(images.values())))
    front = np.asarray(front, dtype=np.uint8)
    wrist = images.get("wrist")
    wrist = np.zeros_like(front) if wrist is None else np.asarray(wrist, dtype=np.uint8)
    raw_state = obs.get("states", obs.get("state"))
    if raw_state is None:
        raise ValueError("RoboMME T1 observation has no proprioceptive state")
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.size < state_dim:
        raise ValueError(f"RoboMME T1 state has {state.size} values, expected at least {state_dim}")
    return {
        "observation/image": front,
        "observation/wrist_image": wrist,
        "observation/state": state[:state_dim],
        "prompt": str(obs.get("task_description", "")).lower(),
    }


def capture_dense_observation(
    session: UpstreamTTTSession,
    obs: dict[str, Any],
    *,
    state_dim: int,
) -> dict[str, Any]:
    """Capture every environment observation, including chunk-buffered steps.

    The first call prepends the official conditioning video and its paired proprioception.  The
    current observation is then appended exactly once.  Later calls accumulate the dense execution
    observations that ``PredictModelServer`` would otherwise discard while serving an action chunk.
    """

    raw = _raw_observation(obs, state_dim)
    video = obs.get("video_history")
    if video:
        if session.saw_video_history:
            raise RuntimeError("RoboMME supplied T1 video history more than once in one episode")
        states = obs.get("video_state_history")
        if states is None:
            raise ValueError("T1 video_history requires paired video_state_history")
        if len(video) != len(states):
            raise ValueError(f"T1 video/state history lengths differ: {len(video)} vs {len(states)}")
        for image, state in zip(video, states, strict=True):
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if state.size < state_dim:
                raise ValueError(f"T1 video state has {state.size} values, expected at least {state_dim}")
            session.pending_images.append(np.asarray(image, dtype=np.uint8))
            session.pending_states.append(state[:state_dim])
        session.pending_exec_start_idx = len(video)
        session.saw_video_history = True

    session.pending_images.append(raw["observation/image"])
    session.pending_states.append(raw["observation/state"])
    session.dense_observations += 1
    return raw


def _noise_seed(model_seed: int, session: UpstreamTTTSession) -> int:
    payload = f"{model_seed}:{session.task_name}:{session.episode_idx}:{session.plans}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


class UpstreamTTTEvalRunner:
    """Share one trained model while keeping the official memory buffer private per session."""

    def __init__(self, policy, *, model_seed: int, state_dim: int = 8):
        import jax
        from mme_vla_suite.shared.mem_buffer import MemoryBufferRecurrent

        config = policy._model.history_config
        if config is None:
            raise RuntimeError("official T1 checkpoint loaded without history_config")
        if (
            config.representation_type != "recurrent"
            or config.recurrent_memory.type != "ttt"
            or config.recurrent_memory.max_recur_steps != 64
            or policy._model.integration_type != "modulation"
        ):
            raise RuntimeError("checkpoint is not official recurrent-TTT + modulation")
        if config.num_views != 1 or config.token_per_image != 64:
            raise RuntimeError("official T1 evaluation expects one view and 64 tokens per history frame")

        self._jax = jax
        self._policy = policy
        self._model = policy._model
        self._memory_type = MemoryBufferRecurrent
        self.model_seed = int(model_seed)
        self.state_dim = int(state_dim)
        self._config = config

    def _new_memory(self):
        config = self._config
        return self._memory_type(
            num_views=config.num_views,
            img_emb_dim=config.memory_feature.img.input_dim,
            pos_emb_dim=config.memory_feature.pos.input_dim,
            state_emb_dim=config.memory_feature.state.input_dim,
            input_obs_horizon=config.streaming_obs_horizon,
            max_recur_steps=config.recurrent_memory.max_recur_steps,
            max_video_steps=config.recurrent_memory.max_pretraj_steps,
            prepare_buffer=True,
            vision_enc_fn=self._policy._vision_encode,
        )

    def _materialize_pending(self, session: UpstreamTTTSession) -> None:
        if not session.pending_images or len(session.pending_images) != len(session.pending_states):
            raise RuntimeError("T1 inference requires a nonempty paired dense-observation buffer")
        if session.memory is None:
            session.memory = self._new_memory()
        images = np.stack(session.pending_images, axis=0)[:, np.newaxis]
        states = np.stack(session.pending_states, axis=0).astype(np.float32)
        begin = session.step_idx + 1
        indices = list(range(begin, begin + len(images)))
        session.memory.add_buffer(images, states, indices)
        session.step_idx = indices[-1]
        if session.pending_exec_start_idx is not None:
            session.exec_start_idx = session.pending_exec_start_idx
        session.pending_images.clear()
        session.pending_states.clear()
        session.pending_exec_start_idx = None

    def _transform(self, session: UpstreamTTTSession, raw: dict[str, Any]) -> dict[str, Any]:
        self._materialize_pending(session)
        gather = session.memory.default_history_feats_gather_fn
        image, position, state, mask = session.memory.prepare_token_recurrent(
            session.step_idx,
            session.exec_start_idx,
            gather,
        )
        inputs = dict(raw)
        inputs.update(
            recur_image_emb=image,
            recur_pos_emb=position,
            recur_state_emb=self._policy._normalize_state(state),
            recur_mask=mask,
        )
        return self._policy._input_transform(self._jax.tree.map(lambda value: value, inputs))

    def infer_batch(
        self,
        sessions: list[UpstreamTTTSession],
        raw_observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Batch only the expensive policy read; recurrent buffers remain episode-private."""

        import jax.numpy as jnp
        from mme_vla_suite.models.integration.history_observation import HistAugObservation

        if not sessions or len(sessions) != len(raw_observations):
            raise ValueError("T1 sessions and observations must have equal nonzero length")
        rows = len(sessions)
        bucket = 4 if rows <= 4 else 8
        if rows > bucket:
            raise ValueError(f"T1 inference batch exceeds 8 rows: {rows}")

        transformed = [self._transform(session, raw) for session, raw in zip(sessions, raw_observations, strict=True)]
        padded = [*transformed, *([transformed[-1]] * (bucket - rows))]
        batch = self._jax.tree.map(
            lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
            *padded,
        )
        observation = HistAugObservation.from_dict(batch)

        noises = [
            self._jax.random.normal(
                self._jax.random.key(_noise_seed(self.model_seed, session)),
                (self._model.action_horizon, self._model.action_dim),
            )
            for session in sessions
        ]
        noises.extend([noises[-1]] * (bucket - rows))
        actions = self._policy._sample_actions(
            self._jax.random.key(0),
            observation,
            noise=jnp.stack(noises),
        )
        state, actions = self._jax.device_get((observation.state[:rows], actions[:rows]))

        results = []
        for index, session in enumerate(sessions):
            output = self._policy._output_transform({"state": state[index], "actions": actions[index]})
            results.append(output)
            session.plans += 1
        return results

    @staticmethod
    def diagnostics(session: UpstreamTTTSession) -> dict[str, int | bool]:
        return {
            "plans": session.plans,
            "dense_observations": session.dense_observations,
            "memory_steps": session.step_idx + 1,
            "exec_start_idx": session.exec_start_idx,
            "saw_video_history": session.saw_video_history,
            "pending_observations_dropped": len(session.pending_images),
        }
