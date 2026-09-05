"""Checkpoint-faithful online ``omega_t`` construction for RoboMME workspace arms.

The policy checkpoints were trained against a materialized cache, but evaluation must construct
the same representation causally from the official conditioning video and dense execution stream.
This module keeps that state private to one harness session and separates pure history assembly
from the expensive frozen SigLIP and recurrent-workspace implementations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_DIM = 2048
STATE_DIM = 8
OMEGA_DIM = 512
INPUT_DIM = FEATURE_DIM + STATE_DIM

VISION_REPO_ID = "Yinpei/pi05_vision_encoder"
VISION_REVISION = "59bd9ff4d58ea0638064bda851fd7d477ee9708c"
VISION_PARAMS_BYTES = 1_659_216_368
VISION_PARAMS_SHA256 = "f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293"
# The first PickXtimes workspace producer predates GPU admission in the otherwise identical
# deliberative trainer.  Its archived source is content-addressed under the local eval runtime.
# Compatibility is intentionally limited to this one reviewed full-file pair.  At runtime we
# reconstruct the archived TPU-only source bytes from the current file and require its exact
# SHA-256; this proves that no encoder, initializer, loss, or inference math changed.
LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256 = "103f93f8571e0bc55a5cde56dcc931c5638b5e8dcaf1d012698ba7ed8d282570"
GPU_ENABLED_WORKSPACE_TRAINER_SHA256 = "a5ed0fb47c234b4cadc71d9a8662fe6d5703e5b69f56915502b6cca272edbeb5"
#: 2026-08-12 refactor of workspace_deliberative.py (reviewed 2026-09-05 against the Aug-10 bundle
#: sarvesh-rmme-PickXtimes-wsm-d16-2c36d9519c880876-0810-182116/source): 23 lines out / 117 in, all
#: training-loop plumbing — pluggable loss_function / sampler_class / run_config_builder /
#: init_params_function, an EMA train step, checkpoint_completion_payload(). The encoder definition,
#: loss math, normalisation and ω layout are unchanged, so checkpoints produced by either earlier
#: trainer serve identically under this file. Status §56.7.
REFACTORED_WORKSPACE_TRAINER_SHA256 = "b8d66d26e43af41f6e9937e1d0fa77b161839fc2079a7ad004db7119d8577a63"
_REVIEWED_PRODUCERS_FOR_REFACTORED = frozenset(
    {LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256, GPU_ENABLED_WORKSPACE_TRAINER_SHA256}
)
_GPU_ENABLED_DEVICE_GUARD = b"""    platforms = {device.platform for device in devices}\n    if not args.cpu_smoke and platforms not in ({"tpu"}, {"gpu"}):\n        raise SystemExit("WSM deliberative production training requires homogeneous TPU or GPU devices")\n"""
_LEGACY_TPU_ONLY_DEVICE_GUARD = b"""    if not args.cpu_smoke and {device.platform for device in devices} != {"tpu"}:\n        raise SystemExit("WSM deliberative production training requires TPU devices")\n"""
UPSTREAM_PREPROCESS_SHA256 = {
    "src/mme_vla_suite/shared/siglip_tokenizer.py": (
        "72fb842327467a4d7cb0f770a514278d67b20721c84d59c82e6cae25f4ce0858"
    ),
    "src/mme_vla_suite/shared/data_utils.py": ("dda1583743528403aa97a4bde8c0305deacfb5a618c9c61937703e59ae76d27a"),
}

WORKSPACE_WINDOWS = {
    "q1": 1,
    "q3": 1,
    "wsm_cfg": 1,
    "wsm_tanh": 1,
    "wsm_d8": 8,
    "wsm_d8_drop05": 8,
    "wsm_d16": 16,
    "wsm_d16_drop05": 16,
    "gdn8_jepa_l01_k1": 8,
    "ptrm": 8,
    "v4_wsm_tanh": 1,
    "v4_wsm_cfg": 1,
    "v4_wsm_gdn8_drop00": 8,
    "v4_wsm_gdn8_drop02": 8,
    "v4_wsm_gdn16_drop00": 16,
    "v4_wsm_gdn16_drop02": 16,
    "v4_gdn8_jepa_visreg_l01_k1": 8,
    "v4_cfg_jepa_visreg_l01_k1": 1,
    "v4_ptrm": 8,
}
WORKSPACE_STEERING_ARMS = frozenset(WORKSPACE_WINDOWS)
TRAIN_ONLY_HEADS = {
    "jepa_l01_k1": ("wsm_jepa_head",),
    "jepa_l1_k32": ("wsm_jepa_head",),
    "jepa_l01_k16": ("wsm_jepa_head",),
    "salient": ("wsm_jepa_head", "wsm_salient_head"),
    "causal_v1": ("wsm_jepa_head", "wsm_salient_head"),
    "gdn8_jepa_l01_k1": ("wsm_jepa_head",),
    "v4_jepa_visreg_l01_k1": ("wsm_jepa_head",),
    "v4_gdn8_jepa_visreg_l01_k1": ("wsm_jepa_head",),
    "v4_cfg_jepa_visreg_l01_k1": ("wsm_jepa_head",),
}


def sha256_file(path: str | Path, *, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_workspace_trainer_implementation(producer_sha256: object, trainer: Path) -> str:
    """Validate exact trainer identity or the sole reviewed TPU-to-GPU admission delta.

    A matching full-file digest remains the normal path.  A mismatch is accepted only when the
    checkpoint names the archived TPU-only producer, the live trainer is the exact reviewed
    GPU-enabled file, and replacing its one device-admission hunk reconstructs the producer's
    full-file digest.  Any unknown pair or any one-byte drift therefore fails closed.
    """
    trainer_bytes = trainer.read_bytes()
    current_sha256 = hashlib.sha256(trainer_bytes).hexdigest()
    if producer_sha256 == current_sha256:
        return "exact_full_source_sha256"
    if current_sha256 == REFACTORED_WORKSPACE_TRAINER_SHA256 and producer_sha256 in _REVIEWED_PRODUCERS_FOR_REFACTORED:
        return "reviewed_training_loop_refactor_2026_08_12_v1"
    if (
        producer_sha256 != LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256
        or current_sha256 != GPU_ENABLED_WORKSPACE_TRAINER_SHA256
        or trainer_bytes.count(_GPU_ENABLED_DEVICE_GUARD) != 1
    ):
        raise ValueError("workspace checkpoint trainer implementation differs from evaluation source")
    reconstructed = trainer_bytes.replace(
        _GPU_ENABLED_DEVICE_GUARD,
        _LEGACY_TPU_ONLY_DEVICE_GUARD,
        1,
    )
    if hashlib.sha256(reconstructed).hexdigest() != LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256:
        raise ValueError("workspace checkpoint trainer implementation differs from evaluation source")
    return "reviewed_tpu_to_gpu_device_admission_only_v1"


def serving_model_arm(arm: str) -> str:
    """Return the model tree to build; auxiliary JEPA/salient heads are train-only."""
    if arm == "q0_noforce":
        return "q0"
    if arm == "q2_noforce":
        return "q2"
    if arm == "gdn8_jepa_l01_k1":
        return "wsm_d8"
    if arm == "v4_gdn8_jepa_visreg_l01_k1":
        return "v4_wsm_gdn8_drop00"
    if arm == "v4_cfg_jepa_visreg_l01_k1":
        return "v4_wsm_cfg"
    if arm == "v4_jepa_visreg_l01_k1":
        return "v4_s0"
    return "s0" if arm in TRAIN_ONLY_HEADS else arm


def workspace_window_for_arm(arm: str) -> int:
    if arm not in WORKSPACE_STEERING_ARMS:
        raise ValueError(f"{arm!r} does not consume workspace tokens at inference")
    return WORKSPACE_WINDOWS[arm]


def _state8(value: Any, *, label: str) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.size < STATE_DIM or not np.isfinite(state[:STATE_DIM]).all():
        raise ValueError(f"{label} must contain at least {STATE_DIM} finite values, got {state.shape}")
    return state[:STATE_DIM].copy()


def _has_rows(value: Any) -> bool:
    return value is not None and len(value) > 0


@dataclasses.dataclass
class WorkspaceSession:
    """All mutable representation state owned by one RoboMME episode."""

    task_name: str
    episode_idx: int
    pending_images: list[np.ndarray] = dataclasses.field(default_factory=list)
    pending_states: list[np.ndarray] = dataclasses.field(default_factory=list)
    frame_features: list[np.ndarray] = dataclasses.field(default_factory=list)
    dense_states: list[np.ndarray] = dataclasses.field(default_factory=list)
    omega_by_step: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    saw_video_history: bool = False
    dense_observations: int = 0
    plans: int = 0


def capture_workspace_observation(session: WorkspaceSession, obs: dict[str, Any]) -> None:
    """Capture demo history once and every dense execution observation exactly once."""
    images = obs.get("images", {})
    if not isinstance(images, dict) or not images:
        raise ValueError("RoboMME workspace observation has no image dictionary")
    front = np.asarray(images.get("agentview", next(iter(images.values()))), dtype=np.uint8)
    if front.ndim != 3 or front.shape[-1] != 3:
        raise ValueError(f"RoboMME workspace agentview must be HxWx3, got {front.shape}")
    state = obs.get("states", obs.get("state"))
    if state is None:
        raise ValueError("RoboMME workspace observation has no proprioceptive state")

    video_present = "video_history" in obs
    state_history_present = "video_state_history" in obs
    episode_restart = obs.get("episode_restart") is True
    if episode_restart:
        if session.saw_video_history or session.pending_images or session.frame_features:
            raise RuntimeError("RoboMME supplied workspace video history after episode history began")
        if not video_present or not state_history_present:
            raise ValueError(
                "workspace episode_restart requires explicit paired video_history and video_state_history"
            )
        video = obs["video_history"]
        video_states = obs["video_state_history"]
        if len(video) != len(video_states):
            raise ValueError(f"workspace video/state history lengths differ: {len(video)} vs {len(video_states)}")
        for index, (image, history_state) in enumerate(zip(video, video_states, strict=True)):
            image = np.asarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"workspace video frame {index} must be HxWx3, got {image.shape}")
            session.pending_images.append(image)
            session.pending_states.append(_state8(history_state, label=f"workspace video state {index}"))
        session.saw_video_history = True
    elif session.dense_observations == 0:
        raise ValueError("first workspace observation requires an explicit official episode_restart history envelope")
    elif video_present or state_history_present:
        raise RuntimeError("RoboMME supplied workspace video history after episode history began")

    session.pending_images.append(front)
    session.pending_states.append(_state8(state, label="workspace current state"))
    session.dense_observations += 1


def causal_history_indices(decision: int, *, stride: int, max_history: int) -> tuple[int, ...]:
    """Match ``WorkspaceBatchSampler.history`` exactly, including modulo alignment."""
    if decision < 0 or stride < 1 or max_history < 1:
        raise ValueError("invalid workspace history geometry")
    start = decision % stride
    return tuple(range(start, decision + 1, stride))[-max_history:]


def requested_omega_steps(decision: int, *, window: int, stride: int) -> tuple[int, ...]:
    """Match the training loader's oldest-to-newest causal window with clamp-to-zero."""
    if decision < 0 or window < 1 or stride < 1:
        raise ValueError("invalid workspace steering window geometry")
    return tuple(max(decision - stride * offset, 0) for offset in range(window - 1, -1, -1))


class OnlineWorkspaceRunner:
    """Batch online features/representations while preserving independent session histories."""

    def __init__(
        self,
        frame_encoder: Callable[[np.ndarray], np.ndarray],
        omega_encoder: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        state_mean: Sequence[float],
        state_std: Sequence[float],
        history_stride: int,
        max_history: int,
        require_video_history: bool = True,
    ) -> None:
        self.frame_encoder = frame_encoder
        self.omega_encoder = omega_encoder
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        if self.state_mean.shape != (STATE_DIM,) or self.state_std.shape != (STATE_DIM,):
            raise ValueError("workspace checkpoint state statistics must each have shape [8]")
        if not np.isfinite(self.state_mean).all() or not np.isfinite(self.state_std).all():
            raise ValueError("workspace checkpoint state statistics are non-finite")
        if np.any(self.state_std <= 0):
            raise ValueError("workspace checkpoint state_std must be strictly positive")
        if history_stride < 1 or max_history < 1:
            raise ValueError("invalid workspace checkpoint history geometry")
        self.history_stride = int(history_stride)
        self.max_history = int(max_history)
        self.require_video_history = bool(require_video_history)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        task_name: str,
        upstream_root: str | Path,
        vision_encoder_home: str | Path,
    ) -> "OnlineWorkspaceRunner":
        omega = CheckpointWorkspaceEncoder(checkpoint, task_name=task_name)
        vision = FrozenSigLipFrameEncoder(
            upstream_root=upstream_root,
            vision_encoder_home=vision_encoder_home,
        )
        config = omega.run_config
        return cls(
            vision,
            omega,
            state_mean=config["state_mean"],
            state_std=config["state_std"],
            history_stride=int(config["history_stride"]),
            max_history=int(config["max_history"]),
            require_video_history=True,
        )

    def _materialize_frames(self, sessions: Sequence[WorkspaceSession]) -> None:
        pending: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        begin = 0
        for session in sessions:
            if len(session.pending_images) != len(session.pending_states):
                raise RuntimeError("workspace session has unpaired pending images/states")
            end = begin + len(session.pending_images)
            spans.append((begin, end))
            pending.extend(session.pending_images)
            begin = end
        if not pending:
            raise RuntimeError("workspace prediction has no newly captured dense observations")
        features = np.asarray(self.frame_encoder(np.stack(pending)), dtype=np.float16)
        if features.shape != (len(pending), FEATURE_DIM) or not np.isfinite(features).all():
            raise RuntimeError(f"frozen frame encoder returned invalid values/shape: {features.shape}")
        for session, (begin, end) in zip(sessions, spans, strict=True):
            session.frame_features.extend(features[begin:end])
            session.dense_states.extend(session.pending_states)
            session.pending_images.clear()
            session.pending_states.clear()
            if len(session.frame_features) != len(session.dense_states):
                raise RuntimeError("workspace materialized feature/state stream became unpaired")

    def _history(self, session: WorkspaceSession, decision: int) -> tuple[np.ndarray, np.ndarray]:
        if decision >= len(session.frame_features):
            raise IndexError(f"workspace decision {decision} exceeds dense history {len(session.frame_features)}")
        indices = causal_history_indices(
            decision,
            stride=self.history_stride,
            max_history=self.max_history,
        )
        history = np.zeros((self.max_history, INPUT_DIM), dtype=np.float32)
        mask = np.zeros((self.max_history,), dtype=np.bool_)
        count = len(indices)
        history[:count, :FEATURE_DIM] = np.asarray(
            [session.frame_features[index] for index in indices], dtype=np.float32
        )
        states = np.asarray([session.dense_states[index] for index in indices], dtype=np.float32)
        history[:count, FEATURE_DIM:] = (states - self.state_mean) / self.state_std
        mask[:count] = True
        return history, mask

    def windows(
        self,
        sessions: Sequence[WorkspaceSession],
        *,
        window: int,
        steering_stride: int = 10,
    ) -> list[np.ndarray]:
        if not sessions or len({id(session) for session in sessions}) != len(sessions):
            raise ValueError("workspace inference requires distinct nonempty session objects")
        if self.require_video_history and any(not session.saw_video_history for session in sessions):
            raise RuntimeError("workspace evaluation requires official conditioning video history")
        self._materialize_frames(sessions)

        requests: list[tuple[int, ...]] = []
        missing: list[tuple[WorkspaceSession, int]] = []
        for session in sessions:
            decision = len(session.frame_features) - 1
            steps = requested_omega_steps(decision, window=window, stride=steering_stride)
            requests.append(steps)
            for step in dict.fromkeys(steps):
                if step not in session.omega_by_step:
                    missing.append((session, step))

        if missing:
            histories, masks = zip(
                *(self._history(session, step) for session, step in missing),
                strict=True,
            )
            encoded = np.asarray(
                self.omega_encoder(np.stack(histories), np.stack(masks)),
                dtype=np.float32,
            )
            if encoded.shape != (len(missing), OMEGA_DIM) or not np.isfinite(encoded).all():
                raise RuntimeError(f"workspace encoder returned invalid values/shape: {encoded.shape}")
            for (session, step), value in zip(missing, encoded, strict=True):
                session.omega_by_step[step] = value.copy()

        outputs = []
        for session, steps in zip(sessions, requests, strict=True):
            value = np.stack([session.omega_by_step[step] for step in steps]).astype(np.float32)
            outputs.append(value)
            session.plans += 1
        return outputs

    @staticmethod
    def diagnostics(session: WorkspaceSession) -> dict[str, int | bool]:
        return {
            "plans": session.plans,
            "dense_observations": session.dense_observations,
            "history_steps": len(session.frame_features),
            "cached_omega_steps": len(session.omega_by_step),
            "saw_video_history": session.saw_video_history,
            "pending_observations_dropped": len(session.pending_images),
        }


class TaskRoutedOnlineWorkspaceRunner:
    """Route all-16 sessions to task-specific encoders while sharing one frozen vision model."""

    def __init__(
        self,
        frame_encoder: Callable[[np.ndarray], np.ndarray],
        checkpoints: dict[str, Path],
    ) -> None:
        try:
            from robomme_integration.training.single_task import TASK_ORDER
        except ModuleNotFoundError as error:
            if error.name != "robomme_integration":
                raise
            from training.single_task import TASK_ORDER

        if tuple(checkpoints) != TASK_ORDER:
            raise ValueError("multitask workspace checkpoints must cover all 16 tasks in canonical order")
        self.frame_encoder = frame_encoder
        self.checkpoints = dict(checkpoints)
        self._runners: dict[str, OnlineWorkspaceRunner] = {}

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        checkpoint_root: str | Path,
        *,
        upstream_root: str | Path,
        vision_encoder_home: str | Path,
    ) -> "TaskRoutedOnlineWorkspaceRunner":
        try:
            from robomme_integration.training.workspace_index import load_workspace_index
        except ModuleNotFoundError as error:
            if error.name != "robomme_integration":
                raise
            from training.workspace_index import load_workspace_index

        index = load_workspace_index(index_path)
        root = Path(checkpoint_root).expanduser().resolve()
        checkpoints: dict[str, Path] = {}
        for task, record in index["tasks"].items():
            representation = record["representation"]
            checkpoint = root / task / str(representation["step"])
            complete = checkpoint / "WSM_GENERATION_COMPLETE.json"
            if not complete.is_file():
                raise ValueError(f"multitask workspace checkpoint is incomplete for {task}: {complete}")
            actual = sha256_file(complete)
            if actual != representation["completion_sha256"]:
                raise ValueError(
                    f"multitask workspace completion/index SHA mismatch for {task}: "
                    f"{actual} != {representation['completion_sha256']}"
                )
            checkpoints[task] = checkpoint
        vision = FrozenSigLipFrameEncoder(
            upstream_root=upstream_root,
            vision_encoder_home=vision_encoder_home,
        )
        return cls(vision, checkpoints)

    def _runner(self, task_name: str) -> OnlineWorkspaceRunner:
        runner = self._runners.get(task_name)
        if runner is not None:
            return runner
        try:
            checkpoint = self.checkpoints[task_name]
        except KeyError as error:
            raise ValueError(f"unknown task for multitask workspace router: {task_name}") from error
        omega = CheckpointWorkspaceEncoder(checkpoint, task_name=task_name)
        config = omega.run_config
        runner = OnlineWorkspaceRunner(
            self.frame_encoder,
            omega,
            state_mean=config["state_mean"],
            state_std=config["state_std"],
            history_stride=int(config["history_stride"]),
            max_history=int(config["max_history"]),
            require_video_history=True,
        )
        self._runners[task_name] = runner
        return runner

    def windows(
        self,
        sessions: Sequence[WorkspaceSession],
        *,
        window: int,
        steering_stride: int = 10,
    ) -> list[np.ndarray]:
        if not sessions or len({id(session) for session in sessions}) != len(sessions):
            raise ValueError("workspace inference requires distinct nonempty session objects")
        groups: dict[str, list[tuple[int, WorkspaceSession]]] = {}
        for index, session in enumerate(sessions):
            groups.setdefault(session.task_name, []).append((index, session))
        outputs: list[np.ndarray | None] = [None] * len(sessions)
        for task_name, group in groups.items():
            values = self._runner(task_name).windows(
                [session for _index, session in group],
                window=window,
                steering_stride=steering_stride,
            )
            for (index, _session), value in zip(group, values, strict=True):
                outputs[index] = value
        if any(value is None for value in outputs):
            raise RuntimeError("multitask workspace router failed to return every row")
        return [value for value in outputs if value is not None]

    @staticmethod
    def diagnostics(session: WorkspaceSession) -> dict[str, int | bool]:
        return OnlineWorkspaceRunner.diagnostics(session)


class FrozenSigLipFrameEncoder:
    """Official pinned Pi0.5 SigLIP preprocessing through the training-time fp16 frame mean."""

    def __init__(
        self,
        *,
        upstream_root: str | Path,
        vision_encoder_home: str | Path,
        max_batch_size: int = 64,
    ) -> None:
        upstream_root = Path(upstream_root).expanduser().resolve()
        for relative, expected in UPSTREAM_PREPROCESS_SHA256.items():
            path = upstream_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"pinned official preprocessing source drifted or is missing: {path}")
        home = Path(vision_encoder_home).expanduser().resolve()
        params = home / "pi05_vision_encoder" / "siglip_params.pkl"
        if not params.is_file() or params.stat().st_size != VISION_PARAMS_BYTES:
            raise ValueError(f"pinned {VISION_REPO_ID}@{VISION_REVISION} asset missing/size mismatch: {params}")
        if sha256_file(params) != VISION_PARAMS_SHA256:
            raise ValueError(f"pinned {VISION_REPO_ID}@{VISION_REVISION} SHA-256 mismatch: {params}")
        if not 1 <= max_batch_size <= 64:
            raise ValueError("SigLIP max_batch_size must lie in [1,64]")

        import jax
        from mme_vla_suite.shared.data_utils import pool_tokens_to_size
        from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer
        from openpi.shared import image_tools

        tokenizer_path = Path(__import__("mme_vla_suite.shared.siglip_tokenizer", fromlist=["x"]).__file__).resolve()
        if tokenizer_path != upstream_root / "src/mme_vla_suite/shared/siglip_tokenizer.py":
            raise ValueError(f"SigLipTokenizer imported outside the pinned official source: {tokenizer_path}")
        prior_home = os.environ.get("OPENPI_DATA_HOME")
        os.environ["OPENPI_DATA_HOME"] = str(home)
        try:
            tokenizer = SigLipTokenizer(inference_batch_size=max_batch_size)
        finally:
            if prior_home is None:
                os.environ.pop("OPENPI_DATA_HOME", None)
            else:
                os.environ["OPENPI_DATA_HOME"] = prior_home

        def encode(images):
            images = images.astype(np.float32) / 255.0 * 2.0 - 1.0
            images = image_tools.resize_with_pad(images, 224, 224)
            tokens = tokenizer(images[:, None])
            return pool_tokens_to_size(tokens, 64)

        self._jax = jax
        self._encode = jax.jit(encode)
        self.max_batch_size = int(max_batch_size)

    @staticmethod
    def _bucket(rows: int, maximum: int) -> int:
        return min(maximum, 1 << (rows - 1).bit_length())

    def __call__(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=np.uint8)
        if images.ndim != 4 or images.shape[-1] != 3 or not len(images):
            raise ValueError(f"SigLIP input must be nonempty [N,H,W,3] uint8, got {images.shape}")
        pieces = []
        for begin in range(0, len(images), self.max_batch_size):
            chunk = images[begin : begin + self.max_batch_size]
            rows = len(chunk)
            bucket = self._bucket(rows, self.max_batch_size)
            if rows < bucket:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], bucket - rows, axis=0)])
            pooled = np.asarray(self._jax.device_get(self._encode(chunk)))[:rows]
            # Offline cache construction converts the bfloat16 8x8 tokens to float32, averages the
            # 64 patches, then persists float16. Preserve that quantization boundary exactly.
            pieces.append(pooled.astype(np.float32)[:, 0].mean(axis=1).astype(np.float16))
        return np.concatenate(pieces)


class CheckpointWorkspaceEncoder:
    """Restore one embedded, provenance-checked recurrent workspace checkpoint."""

    def __init__(self, checkpoint: str | Path, *, task_name: str) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        config_path = checkpoint / "WSM_RUN_CONFIG.json"
        best_path = checkpoint / "WSM_BEST.json"
        complete_path = checkpoint / "WSM_GENERATION_COMPLETE.json"
        for path in (config_path, best_path, complete_path):
            if not path.is_file():
                raise ValueError(f"workspace checkpoint is missing embedded provenance: {path}")
        try:
            step = int(checkpoint.name)
        except ValueError as error:
            raise ValueError(f"workspace checkpoint directory must be its numeric step: {checkpoint}") from error

        config_bytes = config_path.read_bytes()
        best_bytes = best_path.read_bytes()
        run_config = json.loads(config_bytes)
        best = json.loads(best_bytes)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        run_sha = hashlib.sha256(config_bytes).hexdigest()
        embedded = complete.get("embedded_sha256", {})
        expected = {
            "WSM_RUN_CONFIG.json": hashlib.sha256(config_bytes).hexdigest(),
            "WSM_BEST.json": hashlib.sha256(best_bytes).hexdigest(),
        }
        if complete.get("schema_version") != 1 or int(complete.get("step", -1)) != step:
            raise ValueError("workspace generation completion marker has wrong schema/step")
        if complete.get("run_config_sha256") != run_sha or embedded != expected:
            raise ValueError("workspace generation embedded provenance hashes do not match")
        if best.get("run_config_sha256") != run_sha:
            raise ValueError("workspace BEST/config identity mismatch")
        if run_config.get("task") != task_name or run_config.get("omega_dim") != OMEGA_DIM:
            raise ValueError("workspace checkpoint task or omega dimension mismatch")
        trainer = Path(__file__).resolve().parents[1] / "training/workspace_deliberative.py"
        self.trainer_compatibility = validate_workspace_trainer_implementation(
            run_config.get("implementation_sha256"),
            trainer,
        )
        if int(run_config.get("history_stride", -1)) < 1 or int(run_config.get("max_history", -1)) < 1:
            raise ValueError("workspace checkpoint history geometry is invalid")

        import jax
        import optax

        from robomme_integration.training.workspace_deliberative import _manager, encode, init_params

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(run_config["learning_rate"]),
            warmup_steps=int(run_config.get("warmup_steps", 500)),
            decay_steps=int(run_config["steps"]),
            end_value=float(run_config["learning_rate"]) * 0.1,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(run_config.get("clip_gradient_norm", 1.0))),
            optax.adamw(schedule, weight_decay=float(run_config["weight_decay"])),
        )
        initial_params = init_params(jax.random.key(int(run_config["seed"])))
        template = {
            "params": initial_params,
            "opt_state": optimizer.init(initial_params),
            "step": np.asarray(0, dtype=np.int64),
        }
        manager = _manager(checkpoint.parent, max_to_keep=int(run_config.get("max_checkpoints", 10)))
        try:
            if step not in manager.all_steps():
                raise ValueError(f"workspace checkpoint step {step} is not a sealed Orbax generation")
            restored = manager.restore(step, items={"state": template})["state"]
        finally:
            manager.close()
        if int(restored["step"]) != step:
            raise ValueError("restored workspace train-state step mismatch")
        self.run_config = run_config
        self.checkpoint_step = step
        self._jax = jax
        self._params = jax.device_put(restored["params"])
        self._encode = jax.jit(encode)

    @staticmethod
    def _bucket(rows: int) -> int:
        if not 1 <= rows <= 64:
            raise ValueError(f"workspace encoder batch must contain 1--64 rows, got {rows}")
        return 1 << (rows - 1).bit_length()

    def __call__(self, history: np.ndarray, mask: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.bool_)
        rows = len(history)
        if history.shape != (rows, int(self.run_config["max_history"]), INPUT_DIM):
            raise ValueError(f"workspace history shape mismatch: {history.shape}")
        if mask.shape != history.shape[:2]:
            raise ValueError(f"workspace history mask shape mismatch: {mask.shape}")
        bucket = self._bucket(rows)
        if rows < bucket:
            history = np.concatenate([history, np.repeat(history[-1:], bucket - rows, axis=0)])
            mask = np.concatenate([mask, np.repeat(mask[-1:], bucket - rows, axis=0)])
        value = self._encode(self._params, history, mask)
        return np.asarray(self._jax.device_get(value), dtype=np.float32)[:rows]
