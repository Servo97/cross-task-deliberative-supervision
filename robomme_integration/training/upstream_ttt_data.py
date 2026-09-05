"""LeRobot adapter for the official RoboMME recurrent-TTT representation.

This keeps the official model/representation implementation unchanged while replacing its
small-file pickle loader with the pinned LeRobot snapshot plus the compact official feature cache.
The history sampling rule, left padding, feature values, and masks match upstream.  Padding is
explicit float32 instead of upstream NumPy's accidental float64 default; JAX x64 is disabled and
the official feature projection is float32, so this removes host traffic without changing the
model-space computation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .single_task import select_task_episodes, task_manifest_sha256
from .upstream_feature_cache import UPSTREAM_REPO_ID, UPSTREAM_REVISION, recurrent_history_indices


class DeterministicResumeSampler:
    """Epoch-shuffled sampler whose next batch is a pure function of optimizer step.

    PyTorch's default random sampler owns mutable generator state and can prefetch ahead of the
    batch consumed by JAX. Reconstructing that state after TPU preemption would require replaying
    every prior sample. This sampler seeds each epoch independently and starts at the batch implied
    by ``start_step``, so worker prefetch cannot change resume identity.
    """

    def __init__(self, dataset_size: int, batch_size: int, *, seed: int, start_step: int = 0):
        if dataset_size < batch_size or batch_size < 1 or seed < 0 or start_step < 0:
            raise ValueError("invalid deterministic resume-sampler geometry")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.batches_per_epoch = self.dataset_size // self.batch_size
        self.usable_samples = self.batches_per_epoch * self.batch_size
        self.reset(start_step)

    def reset(self, start_step: int) -> None:
        if start_step < 0:
            raise ValueError("start_step must be nonnegative")
        self._next_epoch, batch_in_epoch = divmod(int(start_step), self.batches_per_epoch)
        self._first_sample = batch_in_epoch * self.batch_size

    def __iter__(self):
        import torch

        epoch = self._next_epoch
        self._next_epoch += 1
        generator = torch.Generator()
        generator.manual_seed((self.seed + 1_000_003 * epoch) % (2**63 - 1))
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        begin = self._first_sample
        self._first_sample = 0
        return iter(indices[begin : self.usable_samples])

    def __len__(self) -> int:
        return self.usable_samples


def _sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _repair_episode_index(dataset, episodes: tuple[int, ...]) -> None:
    import torch

    total = int(dataset.meta.total_episodes)
    starts = torch.full((total,), -1, dtype=torch.long)
    ends = torch.full((total,), -1, dtype=torch.long)
    cursor = 0
    for episode in episodes:
        length = int(dataset.meta.episodes[episode]["length"])
        starts[episode] = cursor
        cursor += length
        ends[episode] = cursor
    if cursor != len(dataset.hf_dataset):
        raise RuntimeError(f"selected episode rows {cursor} != Arrow rows {len(dataset.hf_dataset)}")
    dataset.episode_data_index = {"from": starts, "to": ends}


def verify_compact_manifest(
    cache_root: str | Path,
    task_name: str,
    episodes: tuple[int, ...],
    *,
    verify_hashes: bool,
) -> tuple[Path, dict]:
    root = Path(cache_root) / task_name
    path = root / "MANIFEST.json"
    if not path.is_file():
        raise ValueError(f"official recurrent-TTT cache is incomplete: {path} is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored_manifest_sha = manifest.pop("manifest_sha256", None)
    unhashed = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    actual_manifest_sha = hashlib.sha256(unhashed).hexdigest()
    manifest["manifest_sha256"] = stored_manifest_sha
    expected = {
        "schema_version": 1,
        "task_name": task_name,
        "task_manifest_sha256": task_manifest_sha256(task_name),
        "episodes": list(episodes),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"compact-cache manifest {key} mismatch: {manifest.get(key)!r} != {value!r}")
    if manifest.get("upstream") != {"repo_id": UPSTREAM_REPO_ID, "revision": UPSTREAM_REVISION}:
        raise ValueError("compact-cache upstream identity mismatch")
    if stored_manifest_sha != actual_manifest_sha:
        raise ValueError(f"compact-cache manifest hash mismatch: {stored_manifest_sha} != {actual_manifest_sha}")

    compact = root / "compact"
    position_path = compact / "position_f32.npy"
    if not position_path.is_file():
        raise ValueError(f"missing compact position table: {position_path}")
    if verify_hashes and _sha256_file(position_path) != manifest.get("position_sha256"):
        raise ValueError("compact position-table sha256 mismatch")
    records = {int(record["episode"]): record for record in manifest.get("records", [])}
    if tuple(sorted(records)) != episodes:
        raise ValueError("compact-cache record episode set mismatch")
    for episode in episodes:
        episode_root = compact / f"episode_{episode}"
        image = episode_root / "image_bf16_bits.npy"
        state = episode_root / "state_f64.npy"
        if not image.is_file() or not state.is_file():
            raise ValueError(f"missing compact tensors for episode {episode}")
        if verify_hashes:
            if _sha256_file(image) != records[episode].get("image_sha256"):
                raise ValueError(f"compact image sha256 mismatch for episode {episode}")
            if _sha256_file(state) != records[episode].get("state_sha256"):
                raise ValueError(f"compact state sha256 mismatch for episode {episode}")
    return compact, manifest


def assemble_recurrent_history(
    image_bits: np.ndarray,
    positions: np.ndarray,
    states: np.ndarray,
    selected: tuple[int, ...],
    *,
    max_recur_steps: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather and left-pad exact official features into the model's fixed-shape contract."""
    import ml_dtypes

    if not selected or len(selected) > max_recur_steps:
        raise ValueError("selected history must contain 1..max_recur_steps frames")
    indices = np.asarray(selected, dtype=np.int64)
    if indices.min() < 0 or indices.max() >= image_bits.shape[0] or indices.max() >= positions.shape[0]:
        raise IndexError("history index is outside the compact feature cache")
    if indices.max() >= states.shape[0]:
        raise IndexError("history state index is outside the compact feature cache")
    length = len(indices)
    begin = max_recur_steps - length
    image = np.zeros((max_recur_steps, 1, 64, 2048), dtype=np.float32)
    position = np.zeros((max_recur_steps, 1, 64, 768), dtype=np.float32)
    state = np.zeros((max_recur_steps, 8), dtype=np.float32)
    mask = np.zeros((max_recur_steps,), dtype=np.bool_)
    image[begin:] = image_bits[indices].view(ml_dtypes.bfloat16).astype(np.float32)
    position[begin:] = positions[indices]
    state[begin:] = states[indices].astype(np.float32)
    mask[begin:] = True
    return image, position, state, mask


class OfficialTTTLeRobotDataset:
    """Execution-frame dataset with exact upstream recurrent history attached."""

    def __init__(
        self,
        dataset,
        *,
        task_name: str,
        episodes: tuple[int, ...],
        cache_root: str | Path,
        data_config,
        verify_hashes: bool = True,
    ):
        self.dataset = dataset
        self.task_name = task_name
        self.episodes = episodes
        self.compact_root, self.manifest = verify_compact_manifest(
            cache_root,
            task_name,
            episodes,
            verify_hashes=verify_hashes,
        )
        self.positions = np.load(self.compact_root / "position_f32.npy", mmap_mode="r")
        self._episode_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._use_quantiles = bool(data_config.use_quantile_norm)
        if data_config.norm_stats is None or "state" not in data_config.norm_stats:
            raise ValueError("official recurrent-TTT data requires state normalization stats")
        self._state_stats = data_config.norm_stats["state"]

        hf = dataset.hf_dataset
        self._episode = np.asarray(hf["episode_index"], dtype=np.int64)
        self._step = np.asarray(hf["step_idx"], dtype=np.int64)
        self._is_demo = np.asarray(hf["is_demo"], dtype=bool)
        expected = (len(dataset),)
        if any(value.shape != expected for value in (self._episode, self._step, self._is_demo)):
            raise RuntimeError("RoboMME scalar columns do not match selected Arrow length")
        self._execution_indices = np.flatnonzero(~self._is_demo).astype(np.int64, copy=False)
        if not len(self._execution_indices):
            raise ValueError("selected task has no execution frames")
        self._exec_start: dict[int, int] = {}
        for episode in episodes:
            match = np.flatnonzero((self._episode == episode) & ~self._is_demo)
            if not len(match):
                raise ValueError(f"episode {episode} has no execution frame")
            self._exec_start[episode] = int(self._step[match[0]])

    def __len__(self) -> int:
        return len(self._execution_indices)

    def _arrays(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._episode_arrays.get(episode)
        if cached is None:
            root = self.compact_root / f"episode_{episode}"
            cached = (
                np.load(root / "image_bf16_bits.npy", mmap_mode="r"),
                np.load(root / "state_f64.npy", mmap_mode="r"),
            )
            expected_steps = next(
                int(record["steps"]) for record in self.manifest["records"] if int(record["episode"]) == episode
            )
            if cached[0].shape != (expected_steps, 1, 64, 2048) or cached[0].dtype != np.uint16:
                raise ValueError(f"invalid compact image tensor for episode {episode}")
            if cached[1].shape != (expected_steps, 8) or cached[1].dtype != np.float64:
                raise ValueError(f"invalid compact state tensor for episode {episode}")
            self._episode_arrays[episode] = cached
        return cached

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self._use_quantiles:
            stats = self._state_stats
            return (state - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
        return (state - self._state_stats.mean) / (self._state_stats.std + 1e-6)

    def __getitem__(self, index: int) -> dict:
        local_index = int(self._execution_indices[int(index)])
        data = self.dataset[local_index]
        episode = int(self._episode[local_index])
        step = int(self._step[local_index])
        selected = recurrent_history_indices(step, self._exec_start[episode])
        image_bits, states = self._arrays(episode)
        image, position, state, mask = assemble_recurrent_history(
            image_bits,
            self.positions,
            states,
            selected,
        )
        state = self._normalize_state(state)
        result = {
            "image": data["image"],
            "wrist_image": data["wrist_image"],
            "state": data["state"],
            "actions": data["actions"],
            "prompt": data["task"],
            "recur_image_emb": image,
            "recur_pos_emb": position,
            "recur_state_emb": state,
            "recur_mask": mask,
            "static_image_emb": None,
            "static_pos_emb": None,
            "static_state_emb": None,
            "static_mask": None,
            "simple_subgoal": data.get("simple_subgoal", ""),
            "grounded_subgoal": data.get("grounded_subgoal", ""),
        }
        return result


def create_official_ttt_data_loader(
    *,
    data_root: str | Path,
    cache_root: str | Path,
    task_name: str,
    repo_id: str,
    data_config,
    action_horizon: int,
    batch_size: int,
    sharding,
    num_workers: int,
    seed: int,
    verify_hashes: bool,
):
    """Create the upstream transform/batch stack around the compact LeRobot dataset."""
    import jax
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from openpi.training.data_loader import TorchDataLoader, transform_dataset

    episodes = select_task_episodes(data_root, task_name)
    metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=data_root,
        episodes=list(episodes),
        delta_timestamps={"actions": [step / metadata.fps for step in range(action_horizon)]},
    )
    _repair_episode_index(dataset, episodes)
    dataset = OfficialTTTLeRobotDataset(
        dataset,
        task_name=task_name,
        episodes=episodes,
        cache_root=cache_root,
        data_config=data_config,
        verify_hashes=verify_hashes,
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=False)
    local_batch_size = batch_size // jax.process_count()
    torch_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        sampler=DeterministicResumeSampler(
            len(dataset),
            local_batch_size,
            seed=seed,
        ),
        num_workers=num_workers,
        framework="jax",
    )

    class _Loader:
        def __init__(self):
            self._data_config = data_config
            self._data_loader = torch_loader

        def data_config(self):
            return self._data_config

        def set_start_step(self, start_step: int) -> None:
            self._data_loader.torch_loader.sampler.reset(start_step)

        def __iter__(self):
            for batch in self._data_loader:
                yield HistAugObservation.from_dict(batch), batch["actions"]

    # Fail closed if the supplied history config drifted away from the official recurrent recipe.
    history = get_history_config(
        str(
            Path(os.environ["ROBOMME_UPSTREAM_ROOT"])
            / "src/mme_vla_suite/models/config/robomme/recurrent-ttt-modul.yaml"
        )
    )
    if (
        history.representation_type != "recurrent"
        or history.recurrent_memory.type != "ttt"
        or history.recurrent_memory.max_recur_steps != 64
        or history.token_per_image != 64
    ):
        raise ValueError("upstream recurrent-TTT config identity drifted")
    return _Loader()
