"""RoboMME-only LeRobot adapter and sequence windows.

This module deliberately monkey-patches only the process running a RoboMME job. It makes no source
change to the shared OpenPI fork used by the concurrent RoboCasa steelman.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import pathlib
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.stage_q_windows as _stage_q_windows
import openpi.transforms as _transforms
from typing_extensions import override

try:
    from robomme_integration.sequence import uniformly_sample_prefix
except ModuleNotFoundError as error:
    if error.name != "robomme_integration":
        raise
    # SageMaker stages the contents of robomme_integration/ as /opt/ml/code.
    from sequence import uniformly_sample_prefix


OFFICIAL_RECIPE_POLICY_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
_ORIGINAL_PREPROCESS_OBSERVATION = _model.preprocess_observation
_OFFICIAL_RECIPE_TWO_VIEW_PATCHED = False


def _official_recipe_preprocess_observation(
    rng,
    observation,
    *,
    train: bool = False,
    image_keys=OFFICIAL_RECIPE_POLICY_IMAGE_KEYS,
    image_resolution=_model.IMAGE_RESOLUTION,
):
    return _ORIGINAL_PREPROCESS_OBSERVATION(
        rng,
        observation,
        train=train,
        image_keys=image_keys,
        image_resolution=image_resolution,
    )


def install_official_recipe_two_view_patch() -> None:
    """Make only the diagnostic process use RoboMME's exact two-view policy input."""
    global _OFFICIAL_RECIPE_TWO_VIEW_PATCHED
    if _OFFICIAL_RECIPE_TWO_VIEW_PATCHED:
        if _model.preprocess_observation is not _official_recipe_preprocess_observation:
            raise RuntimeError("official two-view preprocessing patch was overwritten")
        return
    parameters = inspect.signature(_ORIGINAL_PREPROCESS_OBSERVATION).parameters
    if not {"train", "image_keys", "image_resolution"}.issubset(parameters):
        raise RuntimeError("current OpenPI preprocessing cannot express an explicit two-view policy")
    expected_current = {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    if set(_model.IMAGE_KEYS) != expected_current:
        raise RuntimeError(f"unreviewed OpenPI image-key contract: {_model.IMAGE_KEYS}; refusing recipe approximation")
    import openpi.models.pi0 as pi0_model

    for method_name in ("compute_loss", "sample_actions"):
        method_source = inspect.getsource(getattr(pi0_model.Pi0, method_name))
        if "_model.preprocess_observation(" not in method_source:
            raise RuntimeError(f"current Pi0.{method_name} bypasses the patchable preprocessing seam")
    if _model.preprocess_observation is not _ORIGINAL_PREPROCESS_OBSERVATION:
        raise RuntimeError("OpenPI preprocessing was already patched by an unknown integration")
    _model.preprocess_observation = _official_recipe_preprocess_observation
    _OFFICIAL_RECIPE_TWO_VIEW_PATCHED = True


def validate_official_recipe_two_view_patch() -> None:
    if (
        not _OFFICIAL_RECIPE_TWO_VIEW_PATCHED
        or _model.preprocess_observation is not _official_recipe_preprocess_observation
    ):
        raise RuntimeError("official_recipe_lerobot exact two-view preprocessing is not installed")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.abs(image).max(initial=0.0) > 1.0:
            raise ValueError("floating-point RoboMME images must lie in [0, 1]")
        image = (255.0 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D RoboMME image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected an RGB RoboMME image, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType
    include_masked_right_view: bool = True

    def __call__(self, data: dict) -> dict:
        _ = self.model_type
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
        }
        if self.include_masked_right_view:
            # Historical project arms keep pi0.5's static three-camera pytree.  The diagnostic
            # official-recipe arm deliberately omits this key to match RoboMME's two-view adapter.
            images["right_wrist_0_rgb"] = np.zeros_like(wrist_image)
            image_masks["right_wrist_0_rgb"] = np.False_
        result = {
            "state": np.asarray(data["observation/state"]),
            "image": images,
            "image_mask": image_masks,
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "robottt_cond" in data:
            # Serve-only additive conditioning computed from the held per-episode W.  It is already
            # in model space and must pass through transforms unchanged.
            result["robottt_cond"] = np.asarray(data["robottt_cond"])
        if "wsm_w_window" in data:
            # Steering receives a causal [K, 512] window (K=1 for CFG/tanh, K=8 for DeltaNet).
            # The newest token is last and every row is at or before the sampled decision.
            omega = np.asarray(data["wsm_w_window"], dtype=np.float32)
            if omega.ndim != 2 or omega.shape[0] < 1 or omega.shape[1] != 512 or not np.isfinite(omega).all():
                raise ValueError(f"invalid RoboMME workspace token shape/value: {omega.shape}")
            result["wsm_w_window"] = omega
        if "wsm_w_target" in data:
            target = np.asarray(data["wsm_w_target"], dtype=np.float32)
            valid = np.asarray(data["wsm_w_target_valid"], dtype=np.bool_)
            if target.shape == (512,):
                expected_valid = ()
            elif target.ndim == 2 and target.shape[0] >= 1 and target.shape[1] == 512:
                expected_valid = (target.shape[0],)
            else:
                raise ValueError(f"invalid RoboMME JEPA target shape: {target.shape}")
            if valid.shape != expected_valid or not np.isfinite(target).all():
                raise ValueError(f"invalid RoboMME JEPA target mask/value: {valid.shape}/{target.shape}")
            result["wsm_w_target"] = target
            result["wsm_w_target_valid"] = valid
        if "wsm_salient_target" in data:
            target = np.asarray(data["wsm_salient_target"], dtype=np.float32)
            valid = np.asarray(data["wsm_salient_valid"], dtype=np.bool_)
            if target.ndim != 1 or target.shape[0] < 1 or valid.shape != () or not np.isfinite(target).all():
                raise ValueError(f"invalid RoboMME salient target/mask: {target.shape}/{valid.shape}")
            result["wsm_salient_target"] = target
            result["wsm_salient_valid"] = valid
        return result


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclasses.dataclass(frozen=True)
class RoboMMEDataConfig(_config.DataConfig):
    lerobot_root: str = ""
    task_name: str | None = None
    execution_only: bool = True
    stage_q_window_len: int = 0
    stage_q_chunk_stride: int = 10
    stage_q_demo_frames: int = 0
    stage_q_iid_steps: bool = False
    workspace_root: str | None = None
    workspace_index: str | None = None
    workspace_window: int = 1
    workspace_stride: int = 10
    workspace_jepa_futures: int = 0
    workspace_jepa_with_window: bool = False
    salient_root: str | None = None
    official_recipe_lerobot: bool = False


@dataclasses.dataclass(frozen=True)
class RoboMMEDataConfigFactory(_config.DataConfigFactory):
    lerobot_root: str | None = None
    task_name: str | None = None
    execution_only: bool = True
    stage_q_window_len: int = 0
    stage_q_chunk_stride: int = 10
    stage_q_demo_frames: int = 0
    stage_q_iid_steps: bool = False
    workspace_root: str | None = None
    workspace_index: str | None = None
    workspace_window: int = 1
    workspace_stride: int = 10
    workspace_jepa_futures: int = 0
    workspace_jepa_with_window: bool = False
    salient_root: str | None = None
    official_recipe_lerobot: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RoboMMEDataConfig:
        if not self.lerobot_root:
            raise ValueError("RoboMME requires an inventory-verified local lerobot_root")
        if self.stage_q_window_len < 0 or self.stage_q_chunk_stride < 1 or self.stage_q_demo_frames < 0:
            raise ValueError("invalid RoboMME sequence geometry")
        if self.stage_q_demo_frames and not self.stage_q_window_len:
            raise ValueError("RoboMME demo prefixes require sequence-window training")
        if self.stage_q_iid_steps and not self.stage_q_window_len:
            raise ValueError("RoboMME A6 iid regrouping requires sequence-window support")
        if self.workspace_window < 1 or self.workspace_stride < 1 or self.workspace_jepa_futures < 0:
            raise ValueError("invalid RoboMME workspace window/stride/JEPA geometry")
        if self.workspace_jepa_with_window:
            if not self.workspace_jepa_futures or self.workspace_window <= 1:
                raise ValueError("JEPA+steering requires an explicit future target and a nontrivial causal window")
        elif self.workspace_jepa_futures and self.workspace_window != 1:
            raise ValueError("JEPA targets do not also expose a steering window without the combo gate")
        if self.official_recipe_lerobot:
            if self.task_name is not None or not self.execution_only:
                raise ValueError("official_recipe_lerobot requires the execution-only all16 dataset")
            if self.stage_q_window_len or self.stage_q_demo_frames or self.stage_q_iid_steps:
                raise ValueError("official_recipe_lerobot forbids sequence-window sampling")
            if self.workspace_root is not None or self.workspace_index is not None or self.salient_root is not None:
                raise ValueError("official_recipe_lerobot forbids workspace/supervision inputs")
            if not bool(getattr(model_config, "pi05", False)) or bool(
                getattr(model_config, "discrete_state_input", True)
            ):
                raise ValueError("official_recipe_lerobot requires pi0.5 with no proprioceptive prompt input")
        base = self.create_base_config(assets_dirs, model_config)
        repack_structure = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "task",
        }
        if self.workspace_root is not None:
            if (self.task_name is None) != bool(self.workspace_index):
                raise ValueError(
                    "RoboMME workspace caches require exactly one routing identity: task_name or all-16 index"
                )
            if self.stage_q_window_len and (
                self.workspace_window != 1 or self.workspace_jepa_futures or self.salient_root is not None
            ):
                raise ValueError("sequence-plus-workspace is reserved for current-only q3 conditioning")
            if self.workspace_jepa_futures:
                repack_structure["wsm_w_target"] = "wsm_w_target"
                repack_structure["wsm_w_target_valid"] = "wsm_w_target_valid"
            if not self.workspace_jepa_futures or self.workspace_jepa_with_window:
                repack_structure["wsm_w_window"] = "wsm_w_window"
        if self.salient_root is not None:
            if self.workspace_root is None or not self.workspace_jepa_futures:
                raise ValueError("salient supervision rides a JEPA workspace-target arm")
            repack_structure["wsm_salient_target"] = "wsm_salient_target"
            repack_structure["wsm_salient_valid"] = "wsm_salient_valid"
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        delta_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                RoboMMEInputs(
                    model_type=model_config.model_type,
                    include_masked_right_view=not self.official_recipe_lerobot,
                ),
                _transforms.DeltaActions(delta_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_mask),
                RoboMMEOutputs(),
            ],
        )
        values = {field.name: getattr(base, field.name) for field in dataclasses.fields(_config.DataConfig)}
        values.update(
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("actions",),
            prompt_from_task=False,
            lerobot_root=self.lerobot_root,
            task_name=self.task_name,
            execution_only=self.execution_only,
            stage_q_window_len=self.stage_q_window_len,
            stage_q_chunk_stride=self.stage_q_chunk_stride,
            stage_q_demo_frames=self.stage_q_demo_frames,
            stage_q_iid_steps=self.stage_q_iid_steps,
            workspace_root=self.workspace_root,
            workspace_index=self.workspace_index,
            workspace_window=self.workspace_window,
            workspace_stride=self.workspace_stride,
            workspace_jepa_futures=self.workspace_jepa_futures,
            workspace_jepa_with_window=self.workspace_jepa_with_window,
            salient_root=self.salient_root,
            official_recipe_lerobot=self.official_recipe_lerobot,
        )
        return RoboMMEDataConfig(**values)


def _sha256_file(path: pathlib.Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


class RoboMMEWorkspaceExecutionDataset:
    """Execution rows augmented by a provenance-locked causal workspace token omega_t."""

    _CAUSAL_CONTRACT = {
        "source_frames": "at_or_before_step_idx",
        "uses_future_execution_frames": False,
        "video_prefix": "benchmark_provided_only",
    }

    def __init__(
        self,
        dataset,
        *,
        root: str,
        task_name: str | None,
        episodes: Sequence[int] | None,
        workspace_index: str | None = None,
        window: int = 1,
        stride: int = 10,
        jepa_futures: int = 0,
        jepa_with_window: bool = False,
        salient_root: str | None = None,
        execution_index_view: bool = True,
    ):
        from .single_task import TASK_EPISODES, TASK_ORDER, task_manifest_sha256
        from .workspace_index import load_workspace_index

        self._dataset = dataset
        if window < 1 or stride < 1 or jepa_futures < 0:
            raise ValueError("invalid workspace window/stride/JEPA geometry")
        if jepa_with_window:
            if not jepa_futures or window <= 1:
                raise ValueError("JEPA+steering requires an explicit future target and a nontrivial causal window")
        elif jepa_futures and window != 1:
            raise ValueError("JEPA target mode cannot also expose a steering window without the combo gate")
        self._window = int(window)
        self._stride = int(stride)
        self._jepa_futures = int(jepa_futures)
        self._jepa_with_window = bool(jepa_with_window)
        root_path = pathlib.Path(root)
        if task_name is None:
            if episodes is not None:
                raise ValueError("all-16 workspace routing derives episodes only from the pinned task map")
            if not workspace_index:
                raise ValueError("all-16 workspace routing requires a sealed local index")
            index = load_workspace_index(
                workspace_index,
                require_supervision=salient_root is not None,
            )
            task_episodes = {task: tuple(TASK_EPISODES[task]) for task in TASK_ORDER}
        else:
            if workspace_index:
                raise ValueError("single-task workspace routing forbids an all-16 index")
            if episodes is None:
                raise ValueError("single-task workspace routing requires an episode manifest")
            task_episodes = {task_name: tuple(int(episode) for episode in episodes)}
            index = None

        self._paths: dict[int, pathlib.Path] = {}
        self._steps: dict[int, int] = {}
        verify_hashes = os.environ.get("WSM_VERIFY_WORKSPACE_HASHES", "1") == "1"
        for routed_task, routed_episodes in task_episodes.items():
            task_root = root_path / routed_task
            manifest_path = task_root / "MANIFEST.json"
            if not manifest_path.is_file():
                raise ValueError(f"workspace cache is incomplete: {manifest_path} is missing")
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            expected = {
                "schema_version": 1,
                "task_name": routed_task,
                "task_manifest_sha256": task_manifest_sha256(routed_task),
                "omega_dim": 512,
                "episodes": list(routed_episodes),
                "causal_contract": self._CAUSAL_CONTRACT,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError(
                        f"workspace manifest {routed_task}.{key} mismatch: {manifest.get(key)!r} != {value!r}"
                    )
            encoder_id = manifest.get("encoder_id")
            if not isinstance(encoder_id, str) or len(encoder_id) != 64:
                raise ValueError(f"workspace manifest for {routed_task} requires a 64-hex encoder_id")
            try:
                int(encoder_id, 16)
            except ValueError as error:
                raise ValueError(f"workspace encoder_id for {routed_task} is not hexadecimal") from error
            if index is not None:
                indexed = index["tasks"][routed_task]
                if encoder_id != indexed["encoder_id"]:
                    raise ValueError(f"workspace encoder/index mismatch for {routed_task}")
                actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
                if actual_manifest_sha != indexed["omega"]["manifest_sha256"]:
                    raise ValueError(f"workspace manifest/index SHA mismatch for {routed_task}")
            records = {int(record["episode"]): record for record in manifest.get("records", [])}
            if tuple(sorted(records)) != routed_episodes:
                raise ValueError(f"workspace manifest episode set mismatch for {routed_task}")
            for episode in routed_episodes:
                if episode in self._paths:
                    raise ValueError(f"workspace episode {episode} is routed by multiple tasks")
                record = records[episode]
                relative = f"episode_{episode}/omega_f16.npy"
                if record.get("path") != relative:
                    raise ValueError(f"workspace path drift for episode {episode}")
                path = task_root / relative
                if not path.is_file():
                    raise ValueError(f"missing workspace token file: {path}")
                if verify_hashes and _sha256_file(path) != record.get("sha256"):
                    raise ValueError(f"workspace token hash mismatch for episode {episode}")
                array = np.load(path, mmap_mode="r")
                steps = int(record.get("steps", -1))
                if array.shape != (steps, 512) or array.dtype != np.float16:
                    raise ValueError(
                        f"invalid workspace token array for episode {episode}: {array.shape}/{array.dtype}"
                    )
                self._paths[episode] = path
                self._steps[episode] = steps

        hf = dataset.hf_dataset
        self._episode = np.asarray(hf["episode_index"], dtype=np.int64)
        self._step = np.asarray(hf["step_idx"], dtype=np.int64)
        is_demo = np.asarray(hf["is_demo"], dtype=bool)
        self.hf_dataset = hf
        self._is_demo = is_demo
        self._execution_index_view = bool(execution_index_view)
        self._indices = np.flatnonzero(~is_demo).astype(np.int64, copy=False)
        if not len(self._indices):
            raise ValueError("RoboMME workspace task has no execution frames")
        self._arrays: dict[int, np.ndarray] = {}
        self._salient_paths: dict[int, pathlib.Path] = {}
        self._salient_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if salient_root is not None:
            from .workspace_deliberative import verify_supervision_manifest

            verify_salient = os.environ.get("WSM_VERIFY_SALIENT_HASHES", "1") == "1"
            for routed_task, routed_episodes in task_episodes.items():
                salient_task_root, salient_manifest = verify_supervision_manifest(
                    salient_root,
                    routed_task,
                    verify_hashes=verify_salient,
                )
                salient_records = {int(record["episode"]): record for record in salient_manifest["records"]}
                if tuple(sorted(salient_records)) != routed_episodes:
                    raise ValueError(f"salient supervision episode set mismatch for {routed_task}")
                if index is not None:
                    indexed = index["tasks"][routed_task]["supervision"]
                    manifest_path = pathlib.Path(salient_root) / routed_task / "MANIFEST.json"
                    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != indexed["manifest_sha256"]:
                        raise ValueError(f"salient manifest/index SHA mismatch for {routed_task}")
                self._salient_paths.update(
                    {episode: salient_task_root / salient_records[episode]["path"] for episode in routed_episodes}
                )

    def __len__(self) -> int:
        return len(self._indices) if self._execution_index_view else len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        local_index = int(self._indices[int(index)]) if self._execution_index_view else int(index)
        if self._is_demo[local_index]:
            raise IndexError("workspace tokens are defined only on RoboMME execution rows")
        episode = int(self._episode[local_index])
        step = int(self._step[local_index])
        if not 0 <= step < self._steps[episode]:
            raise IndexError(f"workspace step {step} outside episode {episode}")
        omega = self._arrays.get(episode)
        if omega is None:
            omega = np.load(self._paths[episode], mmap_mode="r")
            self._arrays[episode] = omega
        if self._jepa_futures:
            requested = step + self._stride * np.arange(1, self._jepa_futures + 1, dtype=np.int64)
            valid = requested < self._steps[episode]
            selected = np.minimum(requested, self._steps[episode] - 1)
            value = np.asarray(omega[selected], dtype=np.float32)
            if self._jepa_futures == 1:
                value, valid = value[0], np.asarray(bool(valid[0]), dtype=np.bool_)
            if self._jepa_with_window:
                window_requested = step - self._stride * np.arange(self._window - 1, -1, -1, dtype=np.int64)
                window_selected = np.maximum(window_requested, 0)
                window_value = np.asarray(omega[window_selected], dtype=np.float32)
        else:
            requested = step - self._stride * np.arange(self._window - 1, -1, -1, dtype=np.int64)
            selected = np.maximum(requested, 0)
            value = np.asarray(omega[selected], dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite workspace value for episode={episode} step={step}")
        if self._jepa_with_window and not np.isfinite(window_value).all():
            raise ValueError(f"non-finite workspace window for episode={episode} step={step}")
        data = self._dataset[local_index]
        if self._jepa_futures:
            data["wsm_w_target"] = value
            data["wsm_w_target_valid"] = valid
            if self._jepa_with_window:
                data["wsm_w_window"] = window_value
        else:
            data["wsm_w_window"] = value
        if self._salient_paths:
            arrays = self._salient_arrays.get(episode)
            if arrays is None:
                with np.load(self._salient_paths[episode]) as source:
                    arrays = (
                        np.asarray(source["event_anchor_i32"], dtype=np.int64),
                        np.asarray(source["event_patch_id_i16"], dtype=np.int64),
                    )
                self._salient_arrays[episode] = arrays
            anchors, patch_ids = arrays
            next_event = int(np.searchsorted(anchors, step, side="right"))
            target = np.zeros((64,), dtype=np.float32)
            salient_valid = next_event < len(anchors)
            if salient_valid:
                patch = int(patch_ids[next_event])
                if not 0 <= patch < 64:
                    raise ValueError(f"salient patch id outside 8x8 base view: {patch}")
                target[patch] = 1.0
            data["wsm_salient_target"] = target
            data["wsm_salient_valid"] = np.asarray(salient_valid, dtype=np.bool_)
        return data


def _repair_selected_episode_data_index(dataset, episodes: Sequence[int]) -> None:
    """Map global episode IDs to local Arrow row spans for LeRobot v2.1 subsets.

    LeRobot 0.3 builds ``episode_data_index`` as a dense length-``len(episodes)`` tensor, but
    ``__getitem__`` indexes it with the original global ``episode_index`` stored in each row.  A
    subset such as 500..599 therefore indexes 500 into a 100-entry tensor.  Keep global identities
    intact (they are part of our task manifest) and expand only this lookup table.
    """
    import torch

    selected = tuple(int(ep) for ep in episodes)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected RoboMME episodes must be nonempty and unique")
    total = int(dataset.meta.total_episodes)
    if min(selected) < 0 or max(selected) >= total:
        raise ValueError(f"selected episode outside [0, {total}): {selected[:3]}...{selected[-3:]}")
    starts = torch.full((total,), -1, dtype=torch.long)
    ends = torch.full((total,), -1, dtype=torch.long)
    cursor = 0
    for episode in selected:
        length = int(dataset.meta.episodes[episode]["length"])
        if length < 1:
            raise ValueError(f"episode {episode} has invalid length {length}")
        starts[episode] = cursor
        cursor += length
        ends[episode] = cursor
    if cursor != len(dataset.hf_dataset):
        raise RuntimeError(
            f"selected episode lengths sum to {cursor}, but Arrow subset has {len(dataset.hf_dataset)} rows"
        )
    dataset.episode_data_index = {"from": starts, "to": ends}


class RoboMMEExecutionDataset:
    """Index view containing only execution frames, never the demo-video prefix."""

    def __init__(self, dataset):
        self._dataset = dataset
        is_demo = np.asarray(dataset.hf_dataset["is_demo"], dtype=bool)
        if is_demo.shape != (len(dataset),):
            raise RuntimeError(f"is_demo shape {is_demo.shape} != {(len(dataset),)}")
        self._indices = np.flatnonzero(~is_demo).astype(np.int64, copy=False)
        if not len(self._indices):
            raise ValueError("RoboMME dataset has no execution samples")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        return self._dataset[int(self._indices[int(index)])]


class RoboMMEWindowDataset:
    """Episode-safe execution windows built from scalar Arrow metadata only."""

    def __init__(self, dataset, *, window_len: int, chunk_stride: int):
        if window_len < 1 or chunk_stride < 1:
            raise ValueError("window_len and chunk_stride must be positive")
        self._dataset = dataset
        self.window_len = int(window_len)
        self.chunk_stride = int(chunk_stride)
        hf = dataset.hf_dataset
        episode = np.asarray(hf["episode_index"], dtype=np.int64)
        step = np.asarray(hf["step_idx"], dtype=np.int64)
        is_demo = np.asarray(hf["is_demo"], dtype=bool)
        expected = (len(dataset),)
        if any(column.shape != expected for column in (episode, step, is_demo)):
            raise RuntimeError("RoboMME scalar metadata columns do not match dataset length")

        self._windows: list[tuple[int, ...]] = []
        begin = 0
        while begin < len(dataset):
            end = begin + 1
            while end < len(dataset) and episode[end] == episode[begin]:
                end += 1
            execution = np.flatnonzero(~is_demo[begin:end]) + begin
            by_step = {int(step[index]): int(index) for index in execution}
            if len(by_step) != len(execution):
                raise RuntimeError(f"duplicate step_idx in episode {int(episode[begin])}")
            if execution.size:
                first, last = int(step[execution[0]]), int(step[execution[-1]])
                span = (self.window_len - 1) * self.chunk_stride
                for base in range(first, last - span + 1, self.chunk_stride):
                    window = tuple(by_step.get(base + i * self.chunk_stride, -1) for i in range(self.window_len))
                    if -1 not in window:
                        self._windows.append(window)
            begin = end
        if not self._windows:
            raise ValueError(f"no RoboMME execution windows fit L={self.window_len}, stride={self.chunk_stride}")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> list[dict]:
        return [self._dataset[item] for item in self._windows[int(index)]]


class RoboMMEIidStepDataset(RoboMMEWindowDataset):
    """A6: identical Stage-Q step multiset, seeded iid regrouping, no support drift."""

    def __init__(self, *args, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        pool = np.asarray([step for window in self._windows for step in window], dtype=np.int64)
        expected = len(self._windows) * self.window_len
        if pool.shape != (expected,):
            raise RuntimeError(f"RoboMME A6 window pool is ragged: {pool.shape} != {(expected,)}")
        self._groups = np.random.default_rng(int(seed)).permutation(pool).reshape(len(self._windows), self.window_len)

    def __getitem__(self, index: int) -> list[dict]:
        return [self._dataset[int(step)] for step in self._groups[int(index)]]


class RoboMMEDemoWindow(NamedTuple):
    """One fixed-shape video-context sequence before per-step transforms."""

    steps: list[dict]
    context_mask: np.ndarray
    loss_mask: np.ndarray


class RoboMMEDemoWindowDataset:
    """Episode-safe ``[uniform demo prefix, execution window]`` sequences.

    The first ``demo_frames`` slots are observation-only context.  The final ``window_len`` slots
    are execution decisions spaced by ``chunk_stride`` and carry the flow-matching loss.  Demo
    slots never require action labels: their action arrays merely keep the transform pytree static
    and are ignored by the demo-capable train step.
    """

    def __init__(
        self,
        dataset,
        *,
        window_len: int,
        chunk_stride: int,
        demo_frames: int,
    ):
        if window_len < 1 or chunk_stride < 1 or demo_frames < 1:
            raise ValueError("window_len, chunk_stride, and demo_frames must be positive")
        self._dataset = dataset
        self.window_len = int(window_len)
        self.chunk_stride = int(chunk_stride)
        self.demo_frames = int(demo_frames)

        hf = dataset.hf_dataset
        episode = np.asarray(hf["episode_index"], dtype=np.int64)
        step = np.asarray(hf["step_idx"], dtype=np.int64)
        is_demo = np.asarray(hf["is_demo"], dtype=bool)
        expected = (len(dataset),)
        if any(column.shape != expected for column in (episode, step, is_demo)):
            raise RuntimeError("RoboMME scalar metadata columns do not match dataset length")

        # (fixed prefix indices, prefix validity, execution indices)
        self._windows: list[tuple[tuple[int, ...], np.ndarray, tuple[int, ...]]] = []
        begin = 0
        while begin < len(dataset):
            end = begin + 1
            while end < len(dataset) and episode[end] == episode[begin]:
                end += 1
            demo = np.flatnonzero(is_demo[begin:end]) + begin
            execution = np.flatnonzero(~is_demo[begin:end]) + begin
            if not execution.size:
                begin = end
                continue

            by_step = {int(step[index]): int(index) for index in execution}
            if len(by_step) != len(execution):
                raise RuntimeError(f"duplicate step_idx in episode {int(episode[begin])}")
            prefix, prefix_valid = uniformly_sample_prefix(
                demo,
                self.demo_frames,
                pad_index=int(execution[0]),
            )
            first, last = int(step[execution[0]]), int(step[execution[-1]])
            span = (self.window_len - 1) * self.chunk_stride
            for base in range(first, last - span + 1, self.chunk_stride):
                window = tuple(by_step.get(base + i * self.chunk_stride, -1) for i in range(self.window_len))
                if -1 not in window:
                    self._windows.append((prefix, prefix_valid.copy(), window))
            begin = end
        if not self._windows:
            raise ValueError(
                f"no RoboMME demo+execution windows fit demo={demo_frames}, L={window_len}, stride={chunk_stride}"
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> RoboMMEDemoWindow:
        prefix, prefix_valid, execution = self._windows[int(index)]
        indices = (*prefix, *execution)
        context_mask = np.concatenate([prefix_valid, np.zeros((self.window_len,), dtype=bool)])
        loss_mask = np.concatenate(
            [
                np.zeros((self.demo_frames,), dtype=np.float32),
                np.ones((self.window_len,), dtype=np.float32),
            ]
        )
        return RoboMMEDemoWindow(
            steps=[self._dataset[item] for item in indices],
            context_mask=context_mask,
            loss_mask=loss_mask,
        )


class RoboMMETransformedDemoWindowDataset:
    """Apply the stock transform chain and retain the two sequence-role masks."""

    def __init__(
        self,
        dataset: RoboMMEDemoWindowDataset,
        transforms: Sequence[_transforms.DataTransformFn],
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        import jax

        window = self._dataset[index]
        steps = [self._transform(step) for step in window.steps]
        keys = set(steps[0])
        for step_index, step in enumerate(steps[1:], 1):
            if set(step) != keys:
                raise RuntimeError(
                    f"demo window step {step_index} produced keys {sorted(step)} != step 0 keys {sorted(keys)}"
                )
        stacked = jax.tree.map(
            lambda *xs: np.stack([np.asarray(value) for value in xs], axis=0),
            *steps,
        )
        stacked["robottt_context_mask"] = window.context_mask
        stacked["robottt_loss_mask"] = window.loss_mask
        return stacked


_PATCHED = False


def install_data_loader_patch() -> None:
    """Install the RoboMME dispatch in this Python process only."""
    global _PATCHED
    if _PATCHED:
        return
    original_create = _data_loader.create_torch_dataset
    original_transform = _data_loader.transform_dataset

    def create_torch_dataset(data_config, action_horizon, model_config, *, seed=0):
        if not isinstance(data_config, RoboMMEDataConfig):
            return original_create(data_config, action_horizon, model_config, seed=seed)
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        episodes = None
        if data_config.task_name is not None:
            from .single_task import select_task_episodes

            episodes = list(select_task_episodes(data_config.lerobot_root, data_config.task_name))

        metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=data_config.lerobot_root)
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=data_config.lerobot_root,
            episodes=episodes,
            delta_timestamps={
                key: [step / metadata.fps for step in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        if episodes is not None:
            _repair_selected_episode_data_index(dataset, episodes)
        if data_config.stage_q_window_len:
            if data_config.stage_q_demo_frames:
                if data_config.execution_only:
                    raise ValueError("demo-prefix sequence training cannot be execution-only")
                return RoboMMEDemoWindowDataset(
                    dataset,
                    window_len=data_config.stage_q_window_len,
                    chunk_stride=data_config.stage_q_chunk_stride,
                    demo_frames=data_config.stage_q_demo_frames,
                )
            if not data_config.execution_only:
                raise ValueError("non-demo RoboMME sequence training must be execution-only")
            step_dataset = dataset
            if data_config.workspace_root is not None:
                if data_config.task_name is not None and episodes is None:
                    raise ValueError("sequence-plus-workspace single-task dispatch lost its episodes")
                step_dataset = RoboMMEWorkspaceExecutionDataset(
                    dataset,
                    root=data_config.workspace_root,
                    task_name=data_config.task_name,
                    episodes=episodes,
                    workspace_index=data_config.workspace_index,
                    window=data_config.workspace_window,
                    stride=data_config.workspace_stride,
                    execution_index_view=False,
                )
            dataset_type = RoboMMEIidStepDataset if data_config.stage_q_iid_steps else RoboMMEWindowDataset
            return dataset_type(
                step_dataset,
                window_len=data_config.stage_q_window_len,
                chunk_stride=data_config.stage_q_chunk_stride,
                **({"seed": seed} if data_config.stage_q_iid_steps else {}),
            )
        if data_config.workspace_root is not None:
            if data_config.task_name is not None and episodes is None:
                raise ValueError("workspace single-task dispatch lost its selected episodes")
            if not data_config.execution_only:
                raise ValueError("workspace cache dispatch requires execution-only training")
            return RoboMMEWorkspaceExecutionDataset(
                dataset,
                root=data_config.workspace_root,
                task_name=data_config.task_name,
                episodes=episodes,
                workspace_index=data_config.workspace_index,
                window=data_config.workspace_window,
                stride=data_config.workspace_stride,
                jepa_futures=data_config.workspace_jepa_futures,
                jepa_with_window=data_config.workspace_jepa_with_window,
                salient_root=data_config.salient_root,
            )
        return RoboMMEExecutionDataset(dataset) if data_config.execution_only else dataset

    def transform_dataset(dataset, data_config, *, skip_norm_stats=False):
        if not isinstance(dataset, (RoboMMEWindowDataset, RoboMMEDemoWindowDataset)):
            return original_transform(dataset, data_config, skip_norm_stats=skip_norm_stats)
        if skip_norm_stats:
            norm_stats = {}
        elif data_config.norm_stats is None:
            raise ValueError("RoboMME normalization stats are required")
        else:
            norm_stats = data_config.norm_stats
        transforms: Sequence[_transforms.DataTransformFn] = [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
        if isinstance(dataset, RoboMMEDemoWindowDataset):
            return RoboMMETransformedDemoWindowDataset(dataset, transforms)
        return _stage_q_windows.TransformedWindowDataset(dataset, transforms)

    original_loader_iter = _data_loader.DataLoaderImpl.__iter__

    def data_loader_iter(self):
        # The stock iterator intentionally drops keys that Observation does not know.  Demo arms
        # carry role masks beside Observation instead, so the custom Stage-Q step can distinguish
        # loss-masked context from execution without overloading any model input field.
        if not isinstance(self.data_config(), RoboMMEDataConfig):
            yield from original_loader_iter(self)
            return
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if "robottt_context_mask" in batch or "robottt_loss_mask" in batch:
                if not {
                    "robottt_context_mask",
                    "robottt_loss_mask",
                }.issubset(batch):
                    raise RuntimeError("RoboMME demo batch has only one sequence-role mask")
                yield (
                    observation,
                    batch["actions"],
                    batch["robottt_loss_mask"],
                    batch["robottt_context_mask"],
                )
            else:
                yield observation, batch["actions"]

    _data_loader.create_torch_dataset = create_torch_dataset
    _data_loader.transform_dataset = transform_dataset
    _data_loader.DataLoaderImpl.__iter__ = data_loader_iter
    _PATCHED = True
