"""Official-history RoboMME model server for the demo-capable Q2V checkpoint."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

try:
    from robomme_integration.eval.runner import (
        DemoRoboTTTRunner,
        EpisodeFastState,
        audit_demo_robottt_checkpoint,
    )
except ModuleNotFoundError as error:
    if error.name != "robomme_integration":
        raise
    from eval.runner import (
        DemoRoboTTTRunner,
        EpisodeFastState,
        audit_demo_robottt_checkpoint,
    )

logger = logging.getLogger(__name__)


class RoboMMEDemoRoboTTTModelServer(PredictModelServer):
    """One Q2V policy with isolated ``W`` per concurrent RoboMME session."""

    def __init__(
        self,
        checkpoint: str,
        *,
        model_seed: int = 7,
        demo_frames: int = 16,
        chunk_size: int = 10,
        max_batch_size: int = 8,
        **kwargs: Any,
    ):
        if not 1 <= max_batch_size <= 8:
            raise ValueError("Q2V max_batch_size must lie in [1, 8]")
        super().__init__(
            chunk_size=chunk_size,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        self.checkpoint = str(checkpoint)
        self.model_seed = int(model_seed)
        self.demo_frames = int(demo_frames)
        self._policy = None
        self._runner: DemoRoboTTTRunner | None = None
        self._sessions: dict[str, EpisodeFastState] = {}

    def _load_model(self) -> None:
        if self._policy is not None:
            return
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"Q2V checkpoint has no params/: {checkpoint}")
        if not (checkpoint / "assets").is_dir():
            raise FileNotFoundError(f"Q2V checkpoint has no assets/: {checkpoint}")

        from openpi.policies import policy_config

        try:
            from robomme_integration.training.config import build_train_config
            from robomme_integration.training.demo_robottt import (
                install_demo_robottt_patch,
            )
        except ModuleNotFoundError as error:
            if error.name != "robomme_integration":
                raise
            from training.config import build_train_config
            from training.demo_robottt import install_demo_robottt_patch

        install_demo_robottt_patch()
        # Config construction is intentionally node-local and read-only.  The dataset root is not
        # opened by policy creation; the checkpoint assets provide the exact norm statistics.
        os.environ["ROBOMME_DATA_ROOT"] = str(checkpoint)
        os.environ["ROBOMME_ASSETS_ROOT"] = str(checkpoint / "assets")
        os.environ["WSM_INIT_FROM"] = str(checkpoint / "params")
        config = build_train_config("q2v")
        self._policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            expose_norm_actions=True,
        )

        import jax

        self._policy._rng = jax.random.key(self.model_seed)
        audit = audit_demo_robottt_checkpoint(self._policy)
        self._runner = DemoRoboTTTRunner(
            self._policy,
            demo_frames=self.demo_frames,
            execution_commit_steps=self.chunk_size,
        )
        logger.info(
            "Loaded Q2V checkpoint=%s model_seed=%d demo_frames=%d audit=%s",
            checkpoint,
            self.model_seed,
            self.demo_frames,
            audit,
        )

    @staticmethod
    def _current_raw(obs: Observation) -> dict:
        images = obs.get("images", {})
        if not isinstance(images, dict) or not images:
            raise ValueError("RoboMME observation has no image dictionary")
        front = images.get("agentview", next(iter(images.values())))
        wrist = images.get("wrist")
        if wrist is None:
            wrist = np.zeros_like(front)
        state = obs.get("states", obs.get("state"))
        if state is None:
            raise ValueError("RoboMME observation has no proprioceptive state")
        return DemoRoboTTTRunner._raw_observation(
            front,
            state,
            obs.get("task_description", ""),
            wrist,
        )

    async def on_episode_start(
        self,
        config: dict[str, Any],
        ctx: SessionContext,
    ) -> None:
        await super().on_episode_start(config, ctx)
        self._sessions.pop(ctx.session_id, None)

    async def on_episode_end(
        self,
        result: dict[str, Any],
        ctx: SessionContext,
    ) -> None:
        session = self._sessions.pop(ctx.session_id, None)
        if session is not None and self._runner is not None:
            logger.info(
                "Q2V episode_end session=%s prefills=%d execution_commits=%d delta_norm=%.6f result=%s",
                ctx.session_id[:8],
                session.prefills,
                session.execution_commits,
                self._runner.delta_norm(session.w, session.w0),
                {**result, "pending_action_block_dropped": session.pending is not None},
            )
        await super().on_episode_end(result, ctx)

    def _prepare_session(
        self,
        obs: Observation,
        ctx: SessionContext,
    ) -> tuple[EpisodeFastState, dict]:
        self._load_model()
        assert self._runner is not None
        session = self._sessions.get(ctx.session_id)
        if session is None:
            w0 = self._runner.init_state()
            session = EpisodeFastState(w=w0, w0=w0)
            self._sessions[ctx.session_id] = session

        # ``predict`` is called only when the harness's 10-action buffer is exhausted.  Therefore a
        # pending block is known to have completed exactly here; if the episode ended early, this
        # branch is never reached and the proposal is correctly discarded at on_episode_end.
        if session.pending is not None:
            session.w = self._runner.commit(session.w, session.pending)
            session.pending = None
            session.execution_commits += 1

        history = obs.get("video_history")
        if history:
            if session.prefills:
                raise RuntimeError("RoboMME demonstration was supplied more than once")
            states = obs.get("video_state_history")
            if states is None:
                raise ValueError(
                    "video_history requires paired video_state_history; use RoboMMEOfficialHistoryBenchmark"
                )
            session.w = self._runner.prefill(
                session.w,
                video_history=history,
                state_history=states,
                wrist_history=obs.get("wrist_video_history"),
                prompt=obs.get("task_description", ""),
            )
            session.prefills += 1
            logger.debug(
                "Q2V prefill session=%s frames=%d delta_norm=%.5f",
                ctx.session_id[:8],
                len(history),
                self._runner.delta_norm(session.w, session.w0),
            )
        return session, self._current_raw(obs)

    def _predict_many(
        self,
        observations: list[Observation],
        contexts: list[SessionContext],
    ) -> list[Action]:
        if not observations or len(observations) != len(contexts):
            raise ValueError("observations and contexts must have equal nonzero length")
        if len(observations) > 8:
            raise ValueError(f"Q2V inference batch exceeds 8 rows: {len(observations)}")
        prepared = [self._prepare_session(obs, ctx) for obs, ctx in zip(observations, contexts, strict=True)]
        sessions = [item[0] for item in prepared]
        assert self._runner is not None
        results, pending = self._runner.infer_batch(
            [session.w for session in sessions],
            [item[1] for item in prepared],
        )
        for session, update in zip(sessions, pending, strict=True):
            session.pending = update
        return results

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self._predict_many([obs], [ctx])[0]

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[Action]:
        return self._predict_many(obs_batch, ctx_batch)

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

    run_server(RoboMMEDemoRoboTTTModelServer)
