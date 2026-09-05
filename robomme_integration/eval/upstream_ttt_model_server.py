"""Batched, session-isolated model server for official RoboMME recurrent TTT+Modul."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

from robomme_integration.eval.upstream_ttt_runner import (
    UpstreamTTTEvalRunner,
    UpstreamTTTSession,
    _raw_observation,
    capture_dense_observation,
)

logger = logging.getLogger(__name__)


def _create_official_ttt_policy(
    checkpoint: Path,
    *,
    upstream_root: Path,
    task_name: str,
    model_seed: int,
):
    """Load the pinned model without upstream's fragile history_config sidecar override."""

    import jax.numpy as jnp
    import openpi.models.model as model_module
    import openpi.training.checkpoints as checkpoints
    import openpi.transforms as transforms
    from mme_vla_suite.policies.policy import MME_VLA_Policy

    from robomme_integration.training.upstream_ttt_train import (
        _install_flax_param_leaf_compat,
        build_config,
        verify_upstream_source,
    )

    updates = {
        "ROBOMME_UPSTREAM_ROOT": str(upstream_root),
        "ROBOMME_TASK": task_name,
        "ROBOMME_DATA_ROOT": str(checkpoint),
        "ROBOMME_ASSETS_ROOT": str(checkpoint / "assets"),
        "WSM_INIT_FROM": str(checkpoint / "params"),
        # Evaluation never opens the training feature cache, but build_config deliberately requires
        # every path to be explicit and local.  The checkpoint is an existing harmless sentinel.
        "ROBOMME_UPSTREAM_CACHE_ROOT": str(checkpoint),
        "WANDB_MODE": "disabled",
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        root = verify_upstream_source(upstream_root)
        config = build_config(root)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    _install_flax_param_leaf_compat()
    model = config.model.load(model_module.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise RuntimeError("official T1 data config has no normalization asset identity")
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    return MME_VLA_Policy(
        model,
        seed=model_seed,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )


class RoboMMEUpstreamTTTModelServer(PredictModelServer):
    """Official TTT+Modul with dense history and one recurrent buffer per session."""

    def __init__(
        self,
        checkpoint: str,
        *,
        upstream_root: str | None = None,
        task_name: str = "PickXtimes",
        model_seed: int = 7,
        state_dim: int = 8,
        chunk_size: int = 10,
        max_batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        if chunk_size != 10:
            raise ValueError("official T1 Gate-A evaluation uses an exact decision stride of 10")
        if not 1 <= max_batch_size <= 8:
            raise ValueError("official T1 max_batch_size must lie in [1, 8]")
        super().__init__(chunk_size=chunk_size, max_batch_size=max_batch_size, **kwargs)
        self.checkpoint = str(checkpoint)
        self.upstream_root = upstream_root or os.environ.get("ROBOMME_UPSTREAM_ROOT", "")
        self.task_name = str(task_name)
        self.model_seed = int(model_seed)
        self.state_dim = int(state_dim)
        self._policy = None
        self._runner: UpstreamTTTEvalRunner | None = None
        self._sessions: dict[str, UpstreamTTTSession] = {}
        self._batch_calls = 0

    def _load_model(self) -> None:
        if self._policy is not None:
            return
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        upstream_root = Path(self.upstream_root).expanduser().resolve()
        if not (checkpoint / "params").is_dir() or not (checkpoint / "assets").is_dir():
            raise FileNotFoundError(f"official T1 checkpoint lacks params/assets: {checkpoint}")
        if not upstream_root.is_dir():
            raise FileNotFoundError(f"pinned official RoboMME source is missing: {upstream_root}")
        self._policy = _create_official_ttt_policy(
            checkpoint,
            upstream_root=upstream_root,
            task_name=self.task_name,
            model_seed=self.model_seed,
        )
        self._runner = UpstreamTTTEvalRunner(
            self._policy,
            model_seed=self.model_seed,
            state_dim=self.state_dim,
        )
        logger.info("Loaded official T1 checkpoint=%s task=%s", checkpoint, self.task_name)

    @staticmethod
    def _task_identity(config: dict[str, Any]) -> tuple[str, int]:
        task = config.get("task", {})
        if not isinstance(task, dict):
            raise ValueError("RoboMME EPISODE_START payload has no task dictionary")
        name = str(task.get("name", task.get("env_id", "")))
        if not name:
            raise ValueError("RoboMME EPISODE_START task has no name/env_id")
        return name, int(task.get("episode_idx", 0))

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        task_name, episode_idx = self._task_identity(config)
        if task_name != self.task_name:
            raise ValueError(f"single-task T1 checkpoint is bound to {self.task_name}, got {task_name}")
        self._sessions[ctx.session_id] = UpstreamTTTSession(task_name, episode_idx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        session = self._sessions.pop(ctx.session_id, None)
        if session is not None:
            logger.info(
                "T1 episode_end session=%s diagnostics=%s result=%s",
                ctx.session_id[:8],
                UpstreamTTTEvalRunner.diagnostics(session),
                result,
            )
        await super().on_episode_end(result, ctx)

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        try:
            session = self._sessions[ctx.session_id]
        except KeyError as error:
            raise RuntimeError("T1 observation arrived before a valid EPISODE_START") from error
        capture_dense_observation(session, obs, state_dim=self.state_dim)
        await super().on_observation(obs, ctx)

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[Action]:
        if not obs_batch or len(obs_batch) != len(ctx_batch):
            raise ValueError("T1 observations and contexts must have equal nonzero length")
        self._load_model()
        assert self._runner is not None
        sessions = []
        raws = []
        for obs, ctx in zip(obs_batch, ctx_batch, strict=True):
            try:
                session = self._sessions[ctx.session_id]
            except KeyError as error:
                raise RuntimeError("T1 predict arrived before a valid EPISODE_START") from error
            sessions.append(session)
            raws.append(_raw_observation(obs, self.state_dim))
        results = self._runner.infer_batch(sessions, raws)
        self._batch_calls += 1
        if self._batch_calls <= 10 or self._batch_calls % 50 == 0:
            logger.info("T1 inference batch_call=%d rows=%d", self._batch_calls, len(results))
        return results

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self.predict_batch([obs], [ctx])[0]

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": True,
            "send_state": True,
            "send_video_history": True,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "agentview": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "state": RAW,
            "language": LANGUAGE,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"action": RAW}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RoboMMEUpstreamTTTModelServer)
