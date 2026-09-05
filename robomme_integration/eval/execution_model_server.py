"""Batched pi0.5 server for single-task specialists and all-16 RoboMME models."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

try:
    from robomme_integration.eval.execution_runner import (
        ExecutionRoboTTTRunner,
        PendingExecutionCommit,
        audit_execution_robottt_checkpoint,
    )
except ModuleNotFoundError as error:
    if error.name != "robomme_integration":
        raise
    from eval.execution_runner import (
        ExecutionRoboTTTRunner,
        PendingExecutionCommit,
        audit_execution_robottt_checkpoint,
    )

try:
    from robomme_integration.eval.workspace_runner import (
        TRAIN_ONLY_HEADS,
        WORKSPACE_STEERING_ARMS,
        OnlineWorkspaceRunner,
        TaskRoutedOnlineWorkspaceRunner,
        WorkspaceSession,
        capture_workspace_observation,
        serving_model_arm,
        workspace_window_for_arm,
    )
except ModuleNotFoundError as error:
    if error.name != "robomme_integration":
        raise
    from eval.workspace_runner import (
        TRAIN_ONLY_HEADS,
        WORKSPACE_STEERING_ARMS,
        OnlineWorkspaceRunner,
        TaskRoutedOnlineWorkspaceRunner,
        WorkspaceSession,
        capture_workspace_observation,
        serving_model_arm,
        workspace_window_for_arm,
    )

logger = logging.getLogger(__name__)
V4_CFG_EVAL_SCALES = frozenset({0.5, 1.0, 1.5, 2.0})

try:
    from robomme_integration.training.single_task import TASK_EPISODES
except ModuleNotFoundError as error:
    if error.name != "robomme_integration":
        raise
    from training.single_task import TASK_EPISODES


@dataclasses.dataclass
class _EpisodeState:
    task_name: str
    episode_idx: int
    w: object | None = None
    w0: object | None = None
    pending: PendingExecutionCommit | None = None
    commits: int = 0
    saw_video_history: bool = False
    workspace: WorkspaceSession | None = None


@dataclasses.dataclass(frozen=True)
class _LoadedModelState:
    """One fully constructed server state, published only after every dependency is ready."""

    policy: Any
    runner: ExecutionRoboTTTRunner | None
    workspace_runner: OnlineWorkspaceRunner | TaskRoutedOnlineWorkspaceRunner | None
    audit: dict[str, Any]


FAST_WEIGHT_ARMS = frozenset({"q2", "q3", "q2_noforce", "v4_q2"})
EXECUTION_ARMS = frozenset(
    {
        "s0",
        "q0",
        "q0_noforce",
        "a6",
        "q2",
        "q2_noforce",
        "q1",
        "q3",
        "wsm_cfg",
        "wsm_tanh",
        "wsm_d8",
        "wsm_d8_drop05",
        "wsm_d16",
        "wsm_d16_drop05",
        "gdn8_jepa_l01_k1",
        "ptrm",
        *TRAIN_ONLY_HEADS,
        "v4_s0",
        "v4_q0",
        "v4_q2",
        "v4_wsm_tanh",
        "v4_wsm_cfg",
        "v4_wsm_gdn8_drop00",
        "v4_wsm_gdn8_drop02",
        "v4_wsm_gdn16_drop00",
        "v4_wsm_gdn16_drop02",
        "v4_ptrm",
    }
)


def _variable_array(value):
    return value.value if hasattr(value, "value") else value[...]


def audit_workspace_conditioner(policy, arm: str, guidance_scale: float) -> dict[str, Any]:
    """Refuse to score a workspace arm if its trained conditioner was not restored."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    interface = "cfg" if arm in {"wsm_cfg", "v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"} else "tanh"
    flag, attribute = ("wsm_cfg2", "wsm_cfg2_cond") if interface == "cfg" else ("wsm_tanh", "wsm_tanh_cond")
    if not getattr(model, flag, False) or not hasattr(model, attribute):
        raise RuntimeError(f"{arm} served model is missing required {attribute} subtree")
    if interface == "cfg" and float(getattr(model, "guidance_scale", -1.0)) != float(guidance_scale):
        raise RuntimeError("WSM-CFG trace-time guidance scale mismatch")
    conditioner = getattr(model, attribute)
    arrays = [jnp.asarray(_variable_array(value)) for _path, value in nnx.state(conditioner, nnx.Param).flat_state()]
    if not arrays or any(not bool(jnp.isfinite(value).all()) for value in arrays):
        raise RuntimeError(f"{attribute} parameters are missing or non-finite")
    readout_name = next(
        (name for name in ("proj_t_out", "proj_readout") if hasattr(conditioner, name)),
        None,
    )
    if readout_name is None:
        raise RuntimeError(f"{attribute} has no recognized output projection")
    readout = getattr(conditioner, readout_name)
    readout_l2 = float(jnp.linalg.norm(jnp.asarray(_variable_array(readout.kernel))))
    if readout_l2 == 0.0:
        raise RuntimeError(f"{attribute}.{readout_name} is zero-init; trained conditioner was not restored")
    summary: dict[str, Any] = {
        "interface": interface,
        "param_tensors": len(arrays),
        "readout_l2": readout_l2,
    }
    if interface == "tanh":
        alpha = jnp.asarray(_variable_array(conditioner.alpha))
        if bool(jnp.all(alpha == 0.001)):
            raise RuntimeError("wsm_tanh_cond.alpha is still exactly at gate initialization")
        summary["gate_max_abs_tanh"] = float(jnp.max(jnp.abs(jnp.tanh(alpha))))
    deltanet_windows = {
        "wsm_d8": 8,
        "wsm_d8_drop05": 8,
        "wsm_d16": 16,
        "wsm_d16_drop05": 16,
        "gdn8_jepa_l01_k1": 8,
        "ptrm": 8,
        "v4_wsm_gdn8_drop00": 8,
        "v4_wsm_gdn8_drop02": 8,
        "v4_wsm_gdn16_drop00": 16,
        "v4_wsm_gdn16_drop02": 16,
        "v4_gdn8_jepa_visreg_l01_k1": 8,
        "v4_ptrm": 8,
    }
    if arm in deltanet_windows:
        if not hasattr(conditioner, "pos_decay_bias"):
            raise RuntimeError(f"{arm} checkpoint restored no gated-DeltaNet positional decay")
        marker = tuple(int(value) for value in np.asarray(_variable_array(conditioner.pos_decay_bias)).shape)
        expected = (deltanet_windows[arm], 2)
        if marker != expected:
            raise RuntimeError(f"{arm} positional decay shape is {marker}, expected {expected}")
        summary["window"] = deltanet_windows[arm]
        summary["heads"] = 2
    if arm in {"ptrm", "v4_ptrm"}:
        if not hasattr(conditioner, "step_bias") or not hasattr(conditioner, "q_head"):
            raise RuntimeError("PTRM checkpoint restored no recursive-depth marker or Q head")
        step_shape = tuple(int(value) for value in np.asarray(_variable_array(conditioner.step_bias)).shape)
        if step_shape != (4, 1024):
            raise RuntimeError(f"PTRM step_bias shape is {step_shape}, expected (4,1024)")
        summary["ptrm_steps"] = 4
        summary["q_head"] = True
    return summary


def _checkpoint_param_roots(checkpoint: Path) -> set[str]:
    """Read checkpoint parameter roots without constructing a potentially mismatched model."""
    import jax
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(checkpoint / "params")
    tree = getattr(metadata, "item_metadata", metadata)["params"]
    roots = set()
    for path, _leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if path:
            roots.add(str(getattr(path[0], "key", getattr(path[0], "name", path[0]))))
    return roots


def audit_train_only_checkpoint(checkpoint: Path, arm: str) -> dict[str, Any]:
    expected = TRAIN_ONLY_HEADS.get(arm)
    if expected is None:
        return {}
    roots = _checkpoint_param_roots(checkpoint)
    missing = [name for name in expected if name not in roots]
    if missing:
        raise RuntimeError(f"{arm} checkpoint is missing its trained auxiliary subtree(s): {missing}")
    return {
        "checkpoint_auxiliary_subtrees": list(expected),
        "served_as": serving_model_arm(arm),
    }


class RoboMMEExecutionModelServer(PredictModelServer):
    """Serve controls, workspace steering, auxiliary-head controls, and project RoboTTT.

    Only the explicit workspace steering arms consume the official conditioning video. JEPA and
    salient heads supervise representation learning during post-training but do not alter the
    inference graph; those checkpoints are deliberately restored into the base Pi0.5 model tree.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        arm: str = "s0",
        workspace_checkpoint: str | None = None,
        workspace_index: str | None = None,
        upstream_root: str | None = None,
        vision_encoder_home: str | None = None,
        cfg_guidance_scale: float = 1.0,
        ptrm_eval_k: int = 1,
        ptrm_eval_sigma: float = 0.0,
        ptrm_eval_select: str = "q",
        task_name: str = "PickXtimes",
        model_seed: int = 7,
        chunk_size: int = 10,
        max_batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        if arm not in EXECUTION_ARMS:
            raise ValueError(f"unsupported execution server arm: {arm!r}")
        if not 1 <= max_batch_size <= 8:
            raise ValueError("RoboMME pi0.5 max_batch_size must lie in [1, 8]")
        if chunk_size != 10:
            raise ValueError("RoboMME single-task checkpoints were trained with an exact stride of 10")
        super().__init__(chunk_size=chunk_size, max_batch_size=max_batch_size, **kwargs)
        self.checkpoint = str(checkpoint)
        self.arm = arm
        self.workspace_checkpoint = workspace_checkpoint or ""
        self.workspace_index = workspace_index or ""
        self.upstream_root = upstream_root or os.environ.get("ROBOMME_UPSTREAM_ROOT", "")
        self.vision_encoder_home = vision_encoder_home or os.environ.get("OPENPI_DATA_HOME", "")
        self.cfg_guidance_scale = float(cfg_guidance_scale)
        cfg_arms = {"wsm_cfg", "v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}
        if arm not in cfg_arms and self.cfg_guidance_scale != 1.0:
            raise ValueError("CFG guidance scale is defined only for wsm_cfg; all other arms require 1.0")
        if arm in cfg_arms and not np.isfinite(self.cfg_guidance_scale):
            raise ValueError("WSM-CFG guidance scale must be finite")
        if arm in {"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"} and (self.cfg_guidance_scale not in V4_CFG_EVAL_SCALES):
            raise ValueError("RoboMME v4 CFG scale must be one of {0.5, 1.0, 1.5, 2.0}")
        self.ptrm_eval_k = int(ptrm_eval_k)
        self.ptrm_eval_sigma = float(ptrm_eval_sigma)
        self.ptrm_eval_select = str(ptrm_eval_select)
        if arm in {"ptrm", "v4_ptrm"}:
            if (self.ptrm_eval_k, self.ptrm_eval_sigma, self.ptrm_eval_select) != (1, 0.0, "q"):
                raise ValueError("RoboMME PTRM is preregistered as E0 only: K=1, sigma=0, select=q")
        elif (self.ptrm_eval_k, self.ptrm_eval_sigma, self.ptrm_eval_select) != (1, 0.0, "q"):
            raise ValueError("PTRM evaluation knobs are valid only for the ptrm arm")
        self.task_name = str(task_name)
        if self.task_name == "all16" and arm in WORKSPACE_STEERING_ARMS and not self.workspace_index:
            raise ValueError(f"multitask {arm} serving requires a sealed workspace index")
        if self.task_name != "all16" and self.workspace_index:
            raise ValueError("single-task serving forbids an all-16 workspace index")
        self.model_seed = int(model_seed)
        self._policy = None
        self._runner: ExecutionRoboTTTRunner | None = None
        self._workspace_runner: OnlineWorkspaceRunner | TaskRoutedOnlineWorkspaceRunner | None = None
        self._load_lock = threading.Lock()
        self._episodes: dict[str, _EpisodeState] = {}
        self._batch_calls = 0
        # ModelServer advertises readiness as soon as __init__ returns.  Restore and validate the
        # policy plus every arm-specific runner before that can happen; _load_model remains
        # idempotent for direct callers and protects against concurrent first-use regressions.
        self._load_model()

    def _initialize_model(self) -> _LoadedModelState:
        """Construct a complete state locally without exposing partially initialized members."""
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"{self.arm} checkpoint has no params/: {checkpoint}")
        if not (checkpoint / "assets").is_dir():
            raise FileNotFoundError(f"{self.arm} checkpoint has no assets/: {checkpoint}")

        auxiliary_audit = audit_train_only_checkpoint(checkpoint, self.arm)
        if self.arm in WORKSPACE_STEERING_ARMS:
            workspace_checkpoint = Path(self.workspace_checkpoint).expanduser().resolve()
            workspace_index = Path(self.workspace_index).expanduser().resolve() if self.workspace_index else None
            upstream_root = Path(self.upstream_root).expanduser().resolve()
            vision_home = Path(self.vision_encoder_home).expanduser().resolve()
            if not workspace_checkpoint.is_dir():
                raise FileNotFoundError(f"{self.arm} requires a local workspace checkpoint: {workspace_checkpoint}")
            if self.task_name == "all16" and (workspace_index is None or not workspace_index.is_file()):
                raise FileNotFoundError(f"{self.arm} requires a local all-16 workspace index: {workspace_index}")
            if not upstream_root.is_dir() or not vision_home.is_dir():
                raise FileNotFoundError("workspace evaluation requires pinned upstream source and vision encoder home")

        from openpi.policies import policy_config

        try:
            from robomme_integration.training.config import build_train_config
        except ModuleNotFoundError as error:
            if error.name != "robomme_integration":
                raise
            from training.config import build_train_config

        updates = {
            "ROBOMME_DATA_ROOT": str(checkpoint),
            "ROBOMME_ASSETS_ROOT": str(checkpoint / "assets"),
            "WSM_INIT_FROM": str(checkpoint / "params"),
        }
        if self.task_name != "all16":
            updates["ROBOMME_TASK"] = self.task_name
        if self.arm in WORKSPACE_STEERING_ARMS:
            updates["ROBOMME_WORKSPACE_ROOT"] = self.workspace_checkpoint
            if self.task_name == "all16":
                updates["ROBOMME_WORKSPACE_INDEX"] = self.workspace_index
        previous = {name: os.environ.get(name) for name in updates}
        os.environ.update(updates)
        try:
            config = build_train_config(serving_model_arm(self.arm))
            if self.arm in {"wsm_cfg", "v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}:
                config = dataclasses.replace(
                    config,
                    model=dataclasses.replace(
                        config.model,
                        wsm_cfg_guidance_scale=self.cfg_guidance_scale,
                    ),
                )
            elif self.arm in {"ptrm", "v4_ptrm"}:
                config = dataclasses.replace(
                    config,
                    model=dataclasses.replace(
                        config.model,
                        wsm_ptrm_eval_k=self.ptrm_eval_k,
                        wsm_ptrm_eval_sigma=self.ptrm_eval_sigma,
                        wsm_ptrm_eval_select=self.ptrm_eval_select,
                    ),
                )
            policy = policy_config.create_trained_policy(
                config,
                checkpoint,
                expose_norm_actions=self.arm in FAST_WEIGHT_ARMS,
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        import jax

        policy._rng = jax.random.key(self.model_seed)
        audit: dict[str, Any] = dict(auxiliary_audit)
        if self.arm in TRAIN_ONLY_HEADS:
            leaked = [name for name in TRAIN_ONLY_HEADS[self.arm] if hasattr(policy._model, name)]
            if leaked:
                raise RuntimeError(f"train-only auxiliary subtrees leaked into served policy: {leaked}")
        workspace_runner: OnlineWorkspaceRunner | TaskRoutedOnlineWorkspaceRunner | None = None
        if self.arm in WORKSPACE_STEERING_ARMS:
            audit["workspace_conditioner"] = audit_workspace_conditioner(
                policy,
                self.arm,
                self.cfg_guidance_scale,
            )
            if self.task_name == "all16":
                workspace_runner = TaskRoutedOnlineWorkspaceRunner.from_index(
                    self.workspace_index,
                    self.workspace_checkpoint,
                    upstream_root=self.upstream_root,
                    vision_encoder_home=self.vision_encoder_home,
                )
            else:
                workspace_runner = OnlineWorkspaceRunner.from_checkpoint(
                    self.workspace_checkpoint,
                    task_name=self.task_name,
                    upstream_root=self.upstream_root,
                    vision_encoder_home=self.vision_encoder_home,
                )
        runner: ExecutionRoboTTTRunner | None = None
        if self.arm in FAST_WEIGHT_ARMS:
            audit["fast_weights"] = audit_execution_robottt_checkpoint(policy)
            runner = ExecutionRoboTTTRunner(
                policy,
                execution_commit_steps=self.chunk_size,
            )
        return _LoadedModelState(
            policy=policy,
            runner=runner,
            workspace_runner=workspace_runner,
            audit=audit,
        )

    def _require_complete_model(self) -> None:
        if self._policy is None:
            raise RuntimeError("policy model is not initialized")
        if self.arm in WORKSPACE_STEERING_ARMS and self._workspace_runner is None:
            raise RuntimeError(f"{self.arm} policy was published without its workspace runner")
        if self.arm in FAST_WEIGHT_ARMS and self._runner is None:
            raise RuntimeError(f"{self.arm} policy was published without its fast-weight runner")

    def _load_model(self) -> None:
        # _policy is deliberately the final publication write.  Seeing it non-None therefore
        # means every required runner was already installed.
        if self._policy is not None:
            self._require_complete_model()
            return
        with self._load_lock:
            if self._policy is not None:
                self._require_complete_model()
                return
            try:
                loaded = self._initialize_model()
                if self.arm in WORKSPACE_STEERING_ARMS and loaded.workspace_runner is None:
                    raise RuntimeError(f"{self.arm} initialization produced no workspace runner")
                if self.arm in FAST_WEIGHT_ARMS and loaded.runner is None:
                    raise RuntimeError(f"{self.arm} initialization produced no fast-weight runner")
            except BaseException:
                # A failed restore must leave no state that a later request can mistake for ready.
                self._policy = None
                self._runner = None
                self._workspace_runner = None
                raise
            self._runner = loaded.runner
            self._workspace_runner = loaded.workspace_runner
            self._policy = loaded.policy
            self._require_complete_model()
        logger.info(
            "Loaded arm=%s checkpoint=%s serving_model=%s audit=%s",
            self.arm,
            Path(self.checkpoint).expanduser().resolve(),
            serving_model_arm(self.arm),
            loaded.audit,
        )

    @staticmethod
    def _current_raw(obs: Observation) -> dict[str, Any]:
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
        return {
            "observation/image": np.asarray(front, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist, dtype=np.uint8),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(obs.get("task_description", "")).lower(),
        }

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
        if self.task_name == "all16" and task_name not in TASK_EPISODES:
            raise ValueError(f"multitask checkpoint received unknown RoboMME task {task_name}")
        if self.task_name != "all16" and task_name != self.task_name:
            raise ValueError(f"single-task checkpoint is bound to {self.task_name}, received episode for {task_name}")
        workspace = (
            WorkspaceSession(task_name=task_name, episode_idx=episode_idx)
            if self.arm in WORKSPACE_STEERING_ARMS
            else None
        )
        self._episodes[ctx.session_id] = _EpisodeState(
            task_name=task_name,
            episode_idx=episode_idx,
            workspace=workspace,
        )

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        episode = self._episodes.pop(ctx.session_id, None)
        if episode is not None:
            diagnostics: dict[str, Any] = {}
            if episode.workspace is not None:
                diagnostics["workspace"] = OnlineWorkspaceRunner.diagnostics(episode.workspace)
            if self.arm in FAST_WEIGHT_ARMS and episode.w is not None and self._runner is not None:
                diagnostics["fast_weights"] = {
                    "commits": episode.commits,
                    "delta_norm": self._runner.delta_norm(episode.w, episode.w0),
                    "pending_action_block_dropped": episode.pending is not None,
                }
            logger.info(
                "%s episode_end session=%s task=%s episode_idx=%d diagnostics=%s result=%s",
                self.arm,
                ctx.session_id[:8],
                episode.task_name,
                episode.episode_idx,
                diagnostics,
                result,
            )
        await super().on_episode_end(result, ctx)

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        episode = self._episode(ctx)
        if episode.workspace is not None:
            capture_workspace_observation(episode.workspace, obs)
        await super().on_observation(obs, ctx)

    def _episode(self, ctx: SessionContext) -> _EpisodeState:
        try:
            return self._episodes[ctx.session_id]
        except KeyError as error:
            raise RuntimeError("predict called before a valid RoboMME EPISODE_START") from error

    def _noise_seed(self, ctx: SessionContext, episode: _EpisodeState) -> int:
        payload = f"{self.model_seed}:{episode.task_name}:{episode.episode_idx}:{ctx.step}".encode()
        return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")

    def _prepare_raw(self, obs: Observation, ctx: SessionContext) -> tuple[_EpisodeState, dict[str, Any]]:
        episode = self._episode(ctx)
        video_history = obs.get("video_history")
        if (
            self.arm not in WORKSPACE_STEERING_ARMS
            and video_history is not None
            and len(video_history) > 0
            and not episode.saw_video_history
        ):
            episode.saw_video_history = True
            logger.info(
                "%s intentionally ignoring conditioning video for execution-only arm task=%s episode_idx=%d",
                self.arm,
                episode.task_name,
                episode.episode_idx,
            )
        raw = self._current_raw(obs)
        raw["policy_noise_seed"] = self._noise_seed(ctx, episode)
        return episode, raw

    @staticmethod
    def _public_result(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key not in {"norm_state", "norm_actions"}}

    def _predict_many(
        self,
        observations: list[Observation],
        contexts: list[SessionContext],
    ) -> list[Action]:
        if not observations or len(observations) != len(contexts):
            raise ValueError("observations and contexts must have equal nonzero length")
        if len(observations) > 8:
            raise ValueError(f"RoboMME inference batch exceeds 8 rows: {len(observations)}")
        self._load_model()
        self._require_complete_model()
        policy = self._policy
        if policy is None:  # Narrow the type after the fail-closed runtime check above.
            raise RuntimeError("policy model disappeared after initialization")

        prepared = [self._prepare_raw(obs, ctx) for obs, ctx in zip(observations, contexts, strict=True)]
        episodes = [item[0] for item in prepared]
        raw = [item[1] for item in prepared]
        started = time.monotonic()

        if self.arm in WORKSPACE_STEERING_ARMS:
            workspace_runner = self._workspace_runner
            if workspace_runner is None:
                raise RuntimeError(f"{self.arm} inference has no workspace runner")
            workspace_sessions = [episode.workspace for episode in episodes]
            if any(session is None for session in workspace_sessions):
                raise RuntimeError("workspace arm has an uninitialized episode representation")
            windows = workspace_runner.windows(
                workspace_sessions,
                window=workspace_window_for_arm(self.arm),
                steering_stride=self.chunk_size,
            )
            for request, omega in zip(raw, windows, strict=True):
                request["wsm_w_window"] = omega

        if self.arm in FAST_WEIGHT_ARMS:
            runner = self._runner
            if runner is None:
                raise RuntimeError(f"{self.arm} inference has no fast-weight runner")
            for episode in episodes:
                if episode.w is None:
                    episode.w = runner.init_state()
                    episode.w0 = episode.w
            commit_indices = [index for index, episode in enumerate(episodes) if episode.pending is not None]
            if commit_indices:
                committed = runner.commit_many(
                    [episodes[index].w for index in commit_indices],
                    [episodes[index].pending for index in commit_indices],
                )
                for index, state in zip(commit_indices, committed, strict=True):
                    episodes[index].w = state
                    episodes[index].pending = None
                    episodes[index].commits += 1

            conditions = runner.condition_many([episode.w for episode in episodes], raw)
            requests = []
            for item, condition in zip(raw, conditions, strict=True):
                request = dict(item)
                request["robottt_cond"] = condition
                requests.append(request)
            results = policy.infer_batch(requests)
            if len(results) != len(episodes):
                raise RuntimeError(f"policy returned {len(results)} rows for {len(episodes)} requests")
            for episode, result in zip(episodes, results, strict=True):
                if "norm_state" not in result or "norm_actions" not in result:
                    raise RuntimeError("fast-weight policy must be created with expose_norm_actions=True")
                episode.pending = PendingExecutionCommit(
                    state=np.asarray(result["norm_state"], dtype=np.float32),
                    actions=np.asarray(result["norm_actions"], dtype=np.float32)[: self.chunk_size],
                )
            self._batch_calls += 1
            if self._batch_calls <= 10 or self._batch_calls % 50 == 0:
                logger.info(
                    "inference arm=%s batch_call=%d rows=%d committed_rows=%d elapsed_s=%.3f",
                    self.arm,
                    self._batch_calls,
                    len(episodes),
                    len(commit_indices),
                    time.monotonic() - started,
                )
            return [self._public_result(result) for result in results]

        results = policy.infer_batch(raw)
        if len(results) != len(episodes):
            raise RuntimeError(f"policy returned {len(results)} rows for {len(episodes)} requests")
        self._batch_calls += 1
        if self._batch_calls <= 10 or self._batch_calls % 50 == 0:
            logger.info(
                "inference arm=%s batch_call=%d rows=%d committed_rows=0 elapsed_s=%.3f",
                self.arm,
                self._batch_calls,
                len(episodes),
                time.monotonic() - started,
            )
        return [self._public_result(result) for result in results]

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

    run_server(RoboMMEExecutionModelServer)
