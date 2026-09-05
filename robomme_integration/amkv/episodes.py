"""Simulator-free episode prefix fixtures for the official FrameSamp memory.

The official ``perceptual-framesamp-modul`` serve path never shows the policy a
whole episode.  ``MemoryBuffer.get_frame_sampling_indices`` reduces the 512-token
budget to ``512 // (16 patches * 1 view) == 32`` frames and asks
``even_sampling_indices`` for the inclusive causal prefix ``[0, t]``: every frame
when ``t < 32``, otherwise ``np.linspace(0, t, 32, dtype=np.int32)``.  Only those
frames -- FRONT view pixels plus their per-frame state -- ever enter memory.

A fixture therefore needs 32 frames, not a prefix of up to 1411 frames.  This
module turns the local RoboMME LeRobot snapshot into such fixtures so that
Attention-Matching work can drive the official policy offline, with no
simulator, no GPU, and no LeRobot dependency.

Deliberate constraints:

* pure NumPy plus stdlib; parquet is read through whichever of pyarrow/polars is
  installed, and PNG payloads through Pillow;
* nothing here imports JAX, Torch, LeRobot, or the simulator;
* sampling indices are never re-derived -- they come from
  ``robomme_integration.training.upstream_framesamp_data.even_sampling_indices``
  so the fixture cannot silently drift from the official rule;
* every artifact is validated on write and re-validated on load, and the payload
  is sealed by both a file digest (``payload_sha256``) and a rebuild-stable
  array digest (``content_sha256``).

Dataset facts this module relies on (verified against the local snapshot):

* ``image``/``wrist_image`` are HF image structs ``{bytes, path}`` holding PNG
  bytes, 256x256 RGB;
* ``step_idx == frame_index == arange(len(episode))`` inside every episode;
* ``exec_start_idx`` is constant per episode and equals the number of leading
  demonstration frames (``is_demo``); it is *not* 48 everywhere -- observed
  values range from 0 to 1064 -- so it is recorded per fixture rather than
  assumed.  Cuts are therefore execution-relative -- ``max(31, exec_start_idx +
  exec_offset)`` and 16 steps later -- because a fixed step number would land
  mid-demonstration for roughly half the dataset and query the policy at a
  state the online evaluator never reaches;
* 34 of the 1600 local shards are corrupt (zeroed ``PAR1`` header, blob hash
  != LFS name).  They are screened out before selection and named in the
  manifest, so a bundle's episode set does not depend on which parquet engine
  happens to be installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    even_sampling_indices,
)

SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
PAYLOAD_FILENAME = "fixtures.npz"

DEFAULT_DATASET_ROOT = Path(
    "/home/sarveshp/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot"
    "/snapshots/1510653cccb4d9e5165fb3141c06d88053decc20"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
STATE_DIM = 8

# The official execution horizon: the policy emits 16 actions per inference.
EXECUTION_HORIZON = 16
# Cuts are execution-relative: the online evaluator only ever queries the policy
# after the demonstration prefill, so a fixture taken inside the demo video would
# probe a state the policy never sees.
DEFAULT_EXEC_OFFSET = 16
MIN_EPISODE_LENGTH = 80
DEFAULT_EPISODE_COUNT = 32
# ``even_sampling_indices`` only returns a full 32-frame memory once t >= 31.
MIN_CAUSAL_STEP = MAX_FRAMES - 1
# The shortest episode any execution-relative cut can fit in.
MIN_FEASIBLE_LENGTH = MIN_CAUSAL_STEP + EXECUTION_HORIZON + 1

FIT_ROLE = "fit_chunk"
EVAL_ROLE = "eval_chunk"
CHUNK_ROLES = (FIT_ROLE, EVAL_ROLE)

FRONT_IMAGE_COLUMN = "image"
WRIST_IMAGE_COLUMN = "wrist_image"
# Scalars are read for the whole episode (a few tens of KB) so the cut can be
# placed before any PNG payload is touched; pixels are then read only up to the
# eval cut.
_SCALAR_COLUMNS = (
    "exec_start_idx",
    "is_demo",
    "step_idx",
    "frame_index",
    "episode_index",
    "task_index",
)
_PIXEL_COLUMNS = (
    FRONT_IMAGE_COLUMN,
    WRIST_IMAGE_COLUMN,
    "state",
)
_RECORD_ARRAY_FIELDS = (
    "memory_step_indices",
    "memory_images",
    "memory_states",
    "obs_image",
    "obs_wrist_image",
    "obs_state",
)
_RECORD_METADATA_FIELDS = (
    "fixture_id",
    "pair_id",
    "chunk_role",
    "episode_index",
    "task_index",
    "task",
    "episode_length",
    "step_idx",
    "exec_start_idx",
    "execution_step",
    "fit_step",
    "eval_step",
    "memory_step_indices",
)


class EpisodeCutUnavailable(ValueError):
    """An episode cannot host both execution-relative cuts; skip it.

    Distinct from the plain ``ValueError`` raised for dataset drift: this one is
    an expected property of short or late-starting episodes and makes the
    builder walk to the next candidate, whereas drift always fails the build.
    """


# --------------------------------------------------------------------------
# Defensive scalar helpers (same contract as the sealed AM artifact module).
# --------------------------------------------------------------------------


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_sha(value: object, *, label: str) -> str:
    value = _require_nonempty(value, label=label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash named arrays including name, dtype, shape, and exact bytes.

    Unlike a digest of the ``.npz`` file this is stable across rebuilds: zip
    containers embed local timestamps, raw array bytes do not.
    """

    if not arrays:
        raise ValueError("at least one array is required for a bundle hash")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if not name:
            raise ValueError("array hash names must be nonempty")
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def bundle_content_sha256(arrays: dict[str, np.ndarray], metadata: object) -> str:
    """Seal payload arrays *and* their manifest metadata under one digest.

    The metadata half matters: fields such as ``exec_start_idx`` or ``task``
    live only in the manifest, so hashing pixels alone would leave the
    scientific labelling of a bundle editable after the fact.
    """

    digest = hashlib.sha256()
    digest.update(array_bundle_sha256(arrays).encode())
    digest.update(b"\0")
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DatasetSource:
    """Identity of the LeRobot snapshot a fixture bundle was cut from.

    ``unreadable_episodes`` records the local integrity screen.  The pinned
    snapshot on this machine contains 34 shards whose blobs do not hash to their
    LFS names and whose ``PAR1`` header has been zeroed; which episodes are
    damaged is a property of the local copy, not of the pinned revision, so it
    belongs in the bundle's provenance rather than in a comment.
    """

    root: str
    info_sha256: str
    episodes_sha256: str
    tasks_sha256: str
    total_episodes: int
    total_tasks: int
    chunks_size: int
    unreadable_episodes: tuple[int, ...]

    def validate(self) -> None:
        _require_nonempty(self.root, label="dataset root")
        for label in ("info_sha256", "episodes_sha256", "tasks_sha256"):
            _require_sha(getattr(self, label), label=label)
        total = _require_int(self.total_episodes, label="total_episodes", minimum=1)
        _require_int(self.total_tasks, label="total_tasks", minimum=1)
        _require_int(self.chunks_size, label="chunks_size", minimum=1)
        if not isinstance(self.unreadable_episodes, tuple):
            raise ValueError("unreadable_episodes must be a tuple")
        indices = [_require_int(value, label="unreadable episode") for value in self.unreadable_episodes]
        if indices != sorted(set(indices)):
            raise ValueError("unreadable_episodes must be sorted and unique")
        if any(index >= total for index in indices):
            raise ValueError("unreadable_episodes references an episode outside the dataset")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = dataclasses.asdict(self)
        result["unreadable_episodes"] = [int(value) for value in self.unreadable_episodes]
        return result

    @classmethod
    def from_dict(cls, value: object) -> "DatasetSource":
        decoded = _require_exact_fields(value, cls, label="dataset_source")
        indices = decoded["unreadable_episodes"]
        if not isinstance(indices, (list, tuple)):
            raise ValueError("dataset_source.unreadable_episodes must be a list")
        decoded["unreadable_episodes"] = tuple(int(item) for item in indices)
        result = cls(**decoded)
        result.validate()
        return result


@dataclasses.dataclass(frozen=True)
class SelectionSpec:
    """Every knob that determines which episodes and cuts end up in a bundle.

    The cuts are not fixed step numbers.  ``exec_start_idx`` varies from 0 to
    1064 across this dataset, so a constant cut would land mid-demonstration for
    roughly half the episodes and query the policy at a state the online
    evaluator never produces.  Instead each episode is cut ``exec_offset`` steps
    into its own execution phase, floored at ``MIN_CAUSAL_STEP`` so all 32
    memory frames remain real.
    """

    seed: int
    episode_count: int
    min_episode_length: int
    exec_offset: int

    def validate(self) -> None:
        _require_int(self.seed, label="seed")
        _require_int(self.episode_count, label="episode_count", minimum=1)
        _require_int(self.exec_offset, label="exec_offset")
        if _require_int(self.min_episode_length, label="min_episode_length", minimum=1) < MIN_FEASIBLE_LENGTH:
            raise ValueError(f"min_episode_length must be at least {MIN_FEASIBLE_LENGTH} to admit any cut")

    def cuts_for(self, *, exec_start_idx: int, episode_length: int) -> tuple[int, int]:
        """Place this episode's ``(fit_step, eval_step)`` or refuse it."""

        self.validate()
        exec_start_idx = _require_int(exec_start_idx, label="exec_start_idx")
        episode_length = _require_int(episode_length, label="episode_length", minimum=1)
        if episode_length < self.min_episode_length:
            raise EpisodeCutUnavailable(
                f"episode length {episode_length} is below the {self.min_episode_length}-step floor"
            )
        fit = max(MIN_CAUSAL_STEP, exec_start_idx + self.exec_offset)
        evaluation = fit + EXECUTION_HORIZON
        if fit < exec_start_idx:
            raise EpisodeCutUnavailable(f"fit cut {fit} precedes execution start {exec_start_idx}")
        if evaluation > episode_length - 1:
            raise EpisodeCutUnavailable(
                f"eval cut {evaluation} exceeds the last step {episode_length - 1} (exec_start_idx={exec_start_idx})"
            )
        return fit, evaluation

    def rule(self) -> str:
        self.validate()
        return (
            "round-robin over task_index ascending; within each task the eligible episodes are "
            f"shuffled by random.Random(f'{{seed}}:{{task_index}}') with seed={self.seed}; "
            f"eligible means episode_length >= {self.min_episode_length} and the shard passes the "
            "PAR1 integrity screen; cuts are execution-relative per episode, "
            f"fit_step = max({MIN_CAUSAL_STEP}, exec_start_idx + {self.exec_offset}) and "
            f"eval_step = fit_step + {EXECUTION_HORIZON}, requiring exec_start_idx <= fit_step and "
            "eval_step <= episode_length - 1; candidates whose cuts do not fit are skipped, and the "
            f"first {self.episode_count} that do each contribute {FIT_ROLE} at fit_step and "
            f"{EVAL_ROLE} at eval_step"
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "SelectionSpec":
        result = cls(**_require_exact_fields(value, cls, label="selection"))
        result.validate()
        return result


def _require_exact_fields(value: object, cls: type, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    names = {field.name for field in dataclasses.fields(cls)}
    if set(value) != names:
        raise ValueError(
            f"{label} field mismatch: missing={sorted(names - set(value))}, unexpected={sorted(set(value) - names)}"
        )
    return dict(value)


# --------------------------------------------------------------------------
# Fixture record
# --------------------------------------------------------------------------


def fixture_id(episode_index: int, step_idx: int) -> str:
    episode_index = _require_int(episode_index, label="episode_index")
    step_idx = _require_int(step_idx, label="step_idx")
    return f"ep{episode_index:06d}-t{step_idx:05d}"


def pair_id(episode_index: int) -> str:
    return f"ep{_require_int(episode_index, label='episode_index'):06d}"


@dataclasses.dataclass(frozen=True)
class FixtureRecord:
    """One causal cut: exactly the frames the official memory would ingest."""

    source: DatasetSource
    selection: SelectionSpec
    fixture_id: str
    pair_id: str
    chunk_role: str
    episode_index: int
    task_index: int
    task: str
    episode_length: int
    step_idx: int
    exec_start_idx: int
    fit_step: int
    eval_step: int
    memory_step_indices: np.ndarray
    memory_images: np.ndarray
    memory_states: np.ndarray
    obs_image: np.ndarray
    obs_wrist_image: np.ndarray
    obs_state: np.ndarray

    @property
    def prompt(self) -> str:
        """Alias for ``task``: the instruction string the policy is given."""

        return self.task

    @property
    def execution_step(self) -> int:
        """Steps elapsed since the demonstration prefill ended; always >= 0."""

        return int(self.step_idx) - int(self.exec_start_idx)

    def validate(self) -> None:
        self.source.validate()
        self.selection.validate()
        episode_index = _require_int(self.episode_index, label="episode_index")
        _require_int(self.task_index, label="task_index")
        _require_nonempty(self.task, label="task")
        length = _require_int(self.episode_length, label="episode_length", minimum=1)
        step = _require_int(self.step_idx, label="step_idx", minimum=MIN_CAUSAL_STEP)
        exec_start = _require_int(self.exec_start_idx, label="exec_start_idx")
        if step > length - 1:
            raise ValueError(f"step_idx {step} is outside an episode of length {length}")
        if exec_start > length:
            raise ValueError(f"exec_start_idx {exec_start} exceeds episode length {length}")
        if self.chunk_role not in CHUNK_ROLES:
            raise ValueError(f"chunk_role must be one of {CHUNK_ROLES}, got {self.chunk_role!r}")
        fit = _require_int(self.fit_step, label="fit_step", minimum=MIN_CAUSAL_STEP)
        evaluation = _require_int(self.eval_step, label="eval_step", minimum=MIN_CAUSAL_STEP)
        if evaluation - fit != EXECUTION_HORIZON:
            raise ValueError(f"a record's cuts must be {EXECUTION_HORIZON} steps apart, got {fit}/{evaluation}")
        expected_cuts = self.selection.cuts_for(
            exec_start_idx=exec_start,
            episode_length=length,
        )
        if (fit, evaluation) != expected_cuts:
            raise ValueError(f"recorded cuts {(fit, evaluation)} disagree with the selection rule {expected_cuts}")
        expected_step = fit if self.chunk_role == FIT_ROLE else evaluation
        if step != expected_step:
            raise ValueError(f"{self.chunk_role} must sit at t={expected_step}, got t={step}")
        if exec_start > fit:
            raise ValueError(f"fit cut {fit} precedes execution start {exec_start}")
        if self.execution_step < 0:
            raise ValueError(f"execution_step must be nonnegative, got {self.execution_step}")
        if self.fixture_id != fixture_id(episode_index, step):
            raise ValueError(f"fixture_id {self.fixture_id!r} does not encode ep{episode_index}/t{step}")
        if self.pair_id != pair_id(episode_index):
            raise ValueError(f"pair_id {self.pair_id!r} does not encode episode {episode_index}")

        indices = self.memory_step_indices
        if indices.shape != (MAX_FRAMES,) or indices.dtype != np.int32:
            raise ValueError(f"memory_step_indices must be int32[{MAX_FRAMES}], got {indices.shape}/{indices.dtype}")
        official = np.asarray(even_sampling_indices(step), dtype=np.int32)
        if official.shape != (MAX_FRAMES,):
            raise ValueError("official sampling did not return a full 32-frame memory")
        if not np.array_equal(indices, official):
            raise ValueError("memory_step_indices disagree with the official FrameSamp rule")
        if int(indices[0]) != 0 or int(indices[-1]) != step:
            raise ValueError("memory must start at frame 0 and end at the causal cut")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("memory_step_indices must be strictly increasing and unique")
        if np.any(indices >= length):
            raise ValueError("memory_step_indices leaked past the end of the episode")

        _validate_uint8_images(
            self.memory_images,
            shape=(MAX_FRAMES, *IMAGE_SHAPE),
            label="memory_images",
        )
        _validate_uint8_images(self.obs_image, shape=IMAGE_SHAPE, label="obs_image")
        _validate_uint8_images(self.obs_wrist_image, shape=IMAGE_SHAPE, label="obs_wrist_image")
        _validate_float32(self.memory_states, shape=(MAX_FRAMES, STATE_DIM), label="memory_states")
        _validate_float32(self.obs_state, shape=(STATE_DIM,), label="obs_state")

        # The last sampled memory frame *is* the current observation frame.
        if not np.array_equal(self.memory_images[-1], self.obs_image):
            raise ValueError("final memory frame does not match the current front observation")
        if not np.array_equal(self.memory_states[-1], self.obs_state):
            raise ValueError("final memory state does not match the current state")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "fixture_id": self.fixture_id,
            "pair_id": self.pair_id,
            "chunk_role": self.chunk_role,
            "episode_index": int(self.episode_index),
            "task_index": int(self.task_index),
            "task": self.task,
            "episode_length": int(self.episode_length),
            "step_idx": int(self.step_idx),
            "exec_start_idx": int(self.exec_start_idx),
            "execution_step": int(self.execution_step),
            "fit_step": int(self.fit_step),
            "eval_step": int(self.eval_step),
            "memory_step_indices": [int(value) for value in self.memory_step_indices],
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in _RECORD_ARRAY_FIELDS}


def _validate_uint8_images(value: object, *, shape: tuple[int, ...], label: str) -> None:
    array = value
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} must be a NumPy array")
    if array.dtype != np.uint8:
        raise ValueError(f"{label} must be uint8, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")


def _validate_float32(value: object, *, shape: tuple[int, ...], label: str) -> None:
    array = value
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} must be a NumPy array")
    if array.dtype != np.float32:
        raise ValueError(f"{label} must be float32, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")


def validate_fixture_records(records: object) -> tuple[FixtureRecord, ...]:
    """Validate every record plus the pair structure that ties them together."""

    if isinstance(records, FixtureRecord) or not isinstance(records, (tuple, list)):
        raise ValueError("records must be a sequence of FixtureRecord")
    result = tuple(records)
    if not result:
        raise ValueError("a fixture bundle must contain at least one record")
    if any(not isinstance(record, FixtureRecord) for record in result):
        raise ValueError("records must be a sequence of FixtureRecord")
    for record in result:
        record.validate()
    if len(result) % 2:
        raise ValueError("fixture records must come in fit/eval pairs")
    source = result[0].source
    selection = result[0].selection
    if any(record.source != source or record.selection != selection for record in result):
        raise ValueError("all records in a bundle must share one dataset source and selection")
    identifiers = [record.fixture_id for record in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("fixture ids must be unique within a bundle")
    pairs = []
    for index in range(0, len(result), 2):
        fit, evaluation = result[index], result[index + 1]
        if (fit.chunk_role, evaluation.chunk_role) != (FIT_ROLE, EVAL_ROLE):
            raise ValueError("records must be ordered as consecutive (fit_chunk, eval_chunk) pairs")
        if fit.pair_id != evaluation.pair_id:
            raise ValueError("paired records must share one pair_id")
        for label in (
            "episode_index",
            "task_index",
            "task",
            "episode_length",
            "exec_start_idx",
            "fit_step",
            "eval_step",
        ):
            if getattr(fit, label) != getattr(evaluation, label):
                raise ValueError(f"paired records disagree on {label}")
        if evaluation.step_idx - fit.step_idx != EXECUTION_HORIZON:
            raise ValueError(f"a pair must be {EXECUTION_HORIZON} steps apart")
        if fit.step_idx != fit.fit_step or evaluation.step_idx != evaluation.eval_step:
            raise ValueError("paired records are not seated at their own cuts")
        pairs.append(fit.pair_id)
    if len(set(pairs)) != len(pairs):
        raise ValueError("each episode may contribute exactly one fit/eval pair")
    if len(pairs) != selection.episode_count:
        raise ValueError(f"bundle holds {len(pairs)} episodes but the selection requested {selection.episode_count}")
    return result


# --------------------------------------------------------------------------
# Dataset reading
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeMeta:
    """One ``meta/episodes.jsonl`` row joined with ``meta/tasks.jsonl``."""

    episode_index: int
    task_index: int
    task: str
    length: int

    def validate(self) -> None:
        _require_int(self.episode_index, label="episode_index")
        _require_int(self.task_index, label="task_index")
        _require_nonempty(self.task, label="task")
        _require_int(self.length, label="length", minimum=1)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {number} is not valid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def read_dataset_meta(dataset_root: str | Path) -> tuple[DatasetSource, tuple[EpisodeMeta, ...], dict]:
    """Load and cross-check ``meta/info.json``, ``episodes.jsonl``, ``tasks.jsonl``."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"
    for path in (info_path, episodes_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset metadata file: {path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError("meta/info.json must be a JSON object")

    task_rows = _read_jsonl(tasks_path)
    tasks: dict[str, int] = {}
    for row in task_rows:
        index = _require_int(row.get("task_index"), label="tasks.jsonl task_index")
        text = _require_nonempty(row.get("task"), label="tasks.jsonl task")
        if text in tasks:
            raise ValueError(f"duplicate task instruction in tasks.jsonl: {text!r}")
        tasks[text] = index
    if len(set(tasks.values())) != len(tasks):
        raise ValueError("tasks.jsonl contains duplicate task_index values")

    episode_rows = _read_jsonl(episodes_path)
    episodes: list[EpisodeMeta] = []
    for position, row in enumerate(episode_rows):
        index = _require_int(row.get("episode_index"), label="episodes.jsonl episode_index")
        if index != position:
            raise ValueError(f"episodes.jsonl is not contiguous at position {position}")
        instructions = row.get("tasks")
        if not isinstance(instructions, list) or len(instructions) != 1:
            raise ValueError(f"episode {index} must declare exactly one task instruction")
        text = _require_nonempty(instructions[0], label="episodes.jsonl task")
        if text not in tasks:
            raise ValueError(f"episode {index} references an instruction absent from tasks.jsonl")
        meta = EpisodeMeta(
            episode_index=index,
            task_index=tasks[text],
            task=text,
            length=_require_int(row.get("length"), label="episodes.jsonl length", minimum=1),
        )
        meta.validate()
        episodes.append(meta)

    total_episodes = _require_int(info.get("total_episodes"), label="info.total_episodes", minimum=1)
    total_tasks = _require_int(info.get("total_tasks"), label="info.total_tasks", minimum=1)
    chunks_size = _require_int(info.get("chunks_size"), label="info.chunks_size", minimum=1)
    if total_episodes != len(episodes):
        raise ValueError("info.total_episodes disagrees with episodes.jsonl")
    if total_tasks != len(tasks):
        raise ValueError("info.total_tasks disagrees with tasks.jsonl")

    unreadable = tuple(
        meta.episode_index
        for meta in episodes
        if parquet_integrity_problem(episode_parquet_path(root, meta.episode_index, chunks_size=chunks_size))
    )
    source = DatasetSource(
        root=str(root),
        info_sha256=_sha256_file(info_path),
        episodes_sha256=_sha256_file(episodes_path),
        tasks_sha256=_sha256_file(tasks_path),
        total_episodes=total_episodes,
        total_tasks=total_tasks,
        chunks_size=chunks_size,
        unreadable_episodes=unreadable,
    )
    source.validate()
    return source, tuple(episodes), info


def episode_parquet_path(dataset_root: str | Path, episode_index: int, *, chunks_size: int) -> Path:
    """Resolve ``data/chunk-{chunk:03d}/episode_{episode:06d}.parquet``."""

    episode_index = _require_int(episode_index, label="episode_index")
    chunks_size = _require_int(chunks_size, label="chunks_size", minimum=1)
    chunk = episode_index // chunks_size
    return Path(dataset_root) / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


PARQUET_MAGIC = b"PAR1"


def parquet_integrity_problem(path: str | Path) -> str | None:
    """Cheap, engine-independent shard screen: existence plus both magics.

    pyarrow and polars disagree about damaged shards -- polars refuses a zeroed
    header outright -- so selection must not depend on which one is installed.
    Screening the container here keeps a bundle's episode set identical on every
    machine, and the failures it catches are exactly the locally corrupted
    Hugging Face blobs.
    """

    path = Path(path)
    if not path.is_file():
        return "missing"
    try:
        size = path.stat().st_size
        if size < 2 * len(PARQUET_MAGIC):
            return f"truncated to {size} bytes"
        with path.open("rb") as stream:
            header = stream.read(len(PARQUET_MAGIC))
            stream.seek(-len(PARQUET_MAGIC), os.SEEK_END)
            footer = stream.read(len(PARQUET_MAGIC))
    except OSError as error:
        return f"unreadable: {error}"
    if header != PARQUET_MAGIC:
        return f"header magic is {header!r}, not {PARQUET_MAGIC!r}"
    if footer != PARQUET_MAGIC:
        return f"footer magic is {footer!r}, not {PARQUET_MAGIC!r}"
    return None


def _read_parquet_prefix(
    path: Path,
    columns: tuple[str, ...],
    rows: int | None,
) -> tuple[dict[str, list], int]:
    """Read the first ``rows`` rows of ``columns`` (all rows if ``None``).

    The row count of the whole shard is returned alongside.  Prefixing matters
    for the pixel pass: episodes reach 1411 frames of PNG payload and a fixture
    never looks beyond its causal cut.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"episode parquet is missing: {path}")
    if rows is not None:
        rows = _require_int(rows, label="rows", minimum=1)
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        pq = None
    if pq is not None:
        handle = pq.ParquetFile(path)
        total = int(handle.metadata.num_rows)
        available = set(handle.schema_arrow.names)
        missing = [name for name in columns if name not in available]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        wanted = total if rows is None else rows
        batches = []
        collected = 0
        for batch in handle.iter_batches(batch_size=max(wanted, 1), columns=list(columns)):
            batches.append(batch)
            collected += batch.num_rows
            if collected >= wanted:
                break
        if not batches:
            raise ValueError(f"{path} is empty")
        import pyarrow as pa

        table = pa.Table.from_batches(batches).slice(0, wanted)
        return {name: table.column(name).to_pylist() for name in columns}, total
    try:
        import polars as pl
    except ModuleNotFoundError as error:
        raise RuntimeError("reading RoboMME parquet requires either pyarrow or polars") from error
    frame = pl.scan_parquet(path)
    available = set(frame.collect_schema().names())
    missing = [name for name in columns if name not in available]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    total = int(frame.select(pl.len()).collect().item())
    selected = frame.select(list(columns))
    prefix = (selected if rows is None else selected.head(rows)).collect()
    return {name: prefix[name].to_list() for name in columns}, total


def _decode_image(value: object, *, label: str) -> np.ndarray:
    """Decode one HF image cell (PNG bytes struct, raw bytes, or an array)."""

    payload = value
    if isinstance(payload, dict):
        if "bytes" not in payload:
            raise ValueError(f"{label} struct has no 'bytes' member")
        payload = payload["bytes"]
    if isinstance(payload, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image
        except ModuleNotFoundError as error:  # pragma: no cover - Pillow ships with the policy venv.
            raise RuntimeError(f"decoding {label} requires Pillow") from error
        with Image.open(io.BytesIO(bytes(payload))) as image:
            if image.mode != "RGB":
                raise ValueError(f"{label} must be an RGB image, got mode {image.mode}")
            array = np.asarray(image, dtype=np.uint8)
    else:
        array = np.asarray(payload)
        if array.dtype != np.uint8:
            raise ValueError(f"{label} must be uint8, got {array.dtype}")
    array = np.ascontiguousarray(array, dtype=np.uint8)
    if array.shape != IMAGE_SHAPE:
        raise ValueError(f"{label} must have shape {IMAGE_SHAPE}, got {array.shape}")
    return array


def _decode_state(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (STATE_DIM,):
        raise ValueError(f"{label} must have shape ({STATE_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def _scalar_int(value: object, *, label: str) -> int:
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"{label} must hold exactly one value")
        value = array.reshape(-1)[0]
    return _require_int(value, label=label)


# --------------------------------------------------------------------------
# Selection + build
# --------------------------------------------------------------------------


def candidate_episode_order(
    episodes: tuple[EpisodeMeta, ...],
    selection: SelectionSpec,
    *,
    exclude: frozenset[int] | set[int] = frozenset(),
) -> tuple[EpisodeMeta, ...]:
    """Round-robin over tasks, seeded shuffle within each task.

    Every eligible episode appears exactly once, so callers can walk further
    down the ordering if a candidate turns out to be unusable.  ``exclude``
    drops episodes that the integrity screen already rejected.
    """

    selection.validate()
    excluded = {_require_int(value, label="excluded episode") for value in exclude}
    # Length is the only cut-feasibility test available from metadata alone;
    # ``exec_start_idx`` lives in the shard, so late-starting episodes are
    # filtered later by ``SelectionSpec.cuts_for``.
    minimum_length = max(selection.min_episode_length, MIN_FEASIBLE_LENGTH)
    buckets: dict[int, list[EpisodeMeta]] = {}
    for episode in episodes:
        episode.validate()
        if episode.length < minimum_length or episode.episode_index in excluded:
            continue
        buckets.setdefault(episode.task_index, []).append(episode)
    for task_index in sorted(buckets):
        bucket = buckets[task_index]
        bucket.sort(key=lambda item: item.episode_index)
        random.Random(f"{selection.seed}:{task_index}").shuffle(bucket)
    ordered: list[EpisodeMeta] = []
    depth = 0
    remaining = True
    while remaining:
        remaining = False
        for task_index in sorted(buckets):
            bucket = buckets[task_index]
            if depth < len(bucket):
                ordered.append(bucket[depth])
                remaining = remaining or depth + 1 < len(bucket)
        depth += 1
    return tuple(ordered)


def _build_episode_pair(
    dataset_root: Path,
    source: DatasetSource,
    selection: SelectionSpec,
    episode: EpisodeMeta,
) -> tuple[FixtureRecord, FixtureRecord]:
    """Cut the fit/eval pair for one episode, failing closed on any drift.

    Two passes: episode-wide scalars decide where the execution-relative cut
    lands, then only the pixels up to that cut are decoded.
    """

    path = episode_parquet_path(dataset_root, episode.episode_index, chunks_size=source.chunks_size)
    problem = parquet_integrity_problem(path)
    if problem:
        raise ValueError(f"episode {episode.episode_index}: parquet integrity screen failed ({problem})")
    scalars, total_rows = _read_parquet_prefix(path, _SCALAR_COLUMNS, None)
    if total_rows != episode.length:
        raise ValueError(
            f"episode {episode.episode_index}: parquet has {total_rows} rows but "
            f"episodes.jsonl declares {episode.length}"
        )
    if len(scalars["step_idx"]) != total_rows:
        raise ValueError(f"episode {episode.episode_index}: scalar columns are short of the row count")
    step_column = np.asarray([_scalar_int(value, label="step_idx") for value in scalars["step_idx"]])
    frame_column = np.asarray([_scalar_int(value, label="frame_index") for value in scalars["frame_index"]])
    expected_steps = np.arange(total_rows)
    if not np.array_equal(step_column, expected_steps) or not np.array_equal(frame_column, expected_steps):
        raise ValueError(f"episode {episode.episode_index}: step_idx/frame_index are not contiguous from 0")
    episode_column = {_scalar_int(value, label="episode_index") for value in scalars["episode_index"]}
    if episode_column != {episode.episode_index}:
        raise ValueError(f"episode {episode.episode_index}: parquet episode_index disagrees with the path")
    task_column = {_scalar_int(value, label="task_index") for value in scalars["task_index"]}
    if task_column != {episode.task_index}:
        raise ValueError(
            f"episode {episode.episode_index}: parquet task_index {sorted(task_column)} "
            f"disagrees with meta task_index {episode.task_index}"
        )
    exec_column = {_scalar_int(value, label="exec_start_idx") for value in scalars["exec_start_idx"]}
    if len(exec_column) != 1:
        raise ValueError(f"episode {episode.episode_index}: exec_start_idx is not constant")
    exec_start_idx = exec_column.pop()
    if exec_start_idx > episode.length:
        raise ValueError(f"episode {episode.episode_index}: exec_start_idx exceeds the episode length")
    demo_flags = np.asarray([bool(np.asarray(value).reshape(-1)[0]) for value in scalars["is_demo"]])
    if not np.array_equal(demo_flags, expected_steps < exec_start_idx):
        raise ValueError(f"episode {episode.episode_index}: is_demo disagrees with exec_start_idx {exec_start_idx}")

    fit_step, eval_step = selection.cuts_for(
        exec_start_idx=exec_start_idx,
        episode_length=episode.length,
    )
    steps = {FIT_ROLE: fit_step, EVAL_ROLE: eval_step}
    needed_rows = eval_step + 1
    columns, pixel_rows = _read_parquet_prefix(path, _PIXEL_COLUMNS, needed_rows)
    if pixel_rows != total_rows or len(columns["state"]) != needed_rows:
        raise ValueError(f"episode {episode.episode_index}: pixel pass disagrees with the scalar pass")

    wanted = sorted({int(index) for step in steps.values() for index in even_sampling_indices(step)})
    front = {index: _decode_image(columns[FRONT_IMAGE_COLUMN][index], label="image") for index in wanted}
    states = {index: _decode_state(columns["state"][index], label="state") for index in wanted}

    records = []
    for role in CHUNK_ROLES:
        step = steps[role]
        indices = np.asarray(even_sampling_indices(step), dtype=np.int32)
        memory_images = np.stack([front[int(index)] for index in indices], axis=0)
        memory_states = np.stack([states[int(index)] for index in indices], axis=0)
        record = FixtureRecord(
            source=source,
            selection=selection,
            fixture_id=fixture_id(episode.episode_index, step),
            pair_id=pair_id(episode.episode_index),
            chunk_role=role,
            episode_index=episode.episode_index,
            task_index=episode.task_index,
            task=episode.task,
            episode_length=episode.length,
            step_idx=step,
            exec_start_idx=exec_start_idx,
            fit_step=fit_step,
            eval_step=eval_step,
            memory_step_indices=indices,
            memory_images=np.ascontiguousarray(memory_images, dtype=np.uint8),
            memory_states=np.ascontiguousarray(memory_states, dtype=np.float32),
            obs_image=front[step],
            obs_wrist_image=_decode_image(columns[WRIST_IMAGE_COLUMN][step], label="wrist_image"),
            obs_state=states[step],
        )
        record.validate()
        records.append(record)
    return records[0], records[1]


def build_fixture_records(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    episodes: int = DEFAULT_EPISODE_COUNT,
    seed: int = 0,
    exec_offset: int = DEFAULT_EXEC_OFFSET,
    min_episode_length: int = MIN_EPISODE_LENGTH,
) -> tuple[FixtureRecord, ...]:
    """Select episodes and cut two execution-relative chunks each.

    Candidates whose demonstration runs too late to host both cuts are skipped
    in favour of the next candidate; dataset drift is never skipped.
    """

    root = Path(dataset_root)
    selection = SelectionSpec(
        seed=_require_int(seed, label="seed"),
        episode_count=_require_int(episodes, label="episodes", minimum=1),
        min_episode_length=_require_int(min_episode_length, label="min_episode_length", minimum=1),
        exec_offset=exec_offset,
    )
    selection.validate()
    source, episode_meta, _ = read_dataset_meta(root)
    ordering = candidate_episode_order(
        episode_meta,
        selection,
        exclude=frozenset(source.unreadable_episodes),
    )
    if len(ordering) < selection.episode_count:
        raise ValueError(
            f"only {len(ordering)} episodes satisfy the selection rule, {selection.episode_count} requested"
        )
    records: list[FixtureRecord] = []
    skipped: list[str] = []
    for candidate in ordering:
        if len(records) == 2 * selection.episode_count:
            break
        try:
            fit, evaluation = _build_episode_pair(root, source, selection, candidate)
        except EpisodeCutUnavailable as error:
            skipped.append(f"episode {candidate.episode_index}: {error}")
            continue
        records.extend((fit, evaluation))
    if len(records) != 2 * selection.episode_count:
        raise ValueError(
            "ran out of candidate episodes before the bundle was complete; "
            f"{len(skipped)} candidates could not host both cuts"
        )
    return validate_fixture_records(records)


# --------------------------------------------------------------------------
# Bundle manifest, write, load
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FixtureBundleManifest:
    """Everything about a bundle except the pixels."""

    schema_version: int
    dataset_source: DatasetSource
    selection: SelectionSpec
    selection_rule: str
    memory_frames: int
    image_shape: tuple[int, int, int]
    state_dim: int
    execution_horizon: int
    record_count: int
    payload_filename: str
    payload_sha256: str
    content_sha256: str
    records: tuple[dict, ...]

    def validate(self) -> None:
        if _require_int(self.schema_version, label="schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported fixture bundle schema {self.schema_version}")
        self.dataset_source.validate()
        self.selection.validate()
        if self.selection_rule != self.selection.rule():
            raise ValueError("selection_rule does not describe the recorded selection")
        if _require_int(self.memory_frames, label="memory_frames", minimum=1) != MAX_FRAMES:
            raise ValueError("official FrameSamp memory must hold exactly 32 frames")
        if tuple(self.image_shape) != IMAGE_SHAPE:
            raise ValueError(f"image_shape must be {IMAGE_SHAPE}, got {tuple(self.image_shape)}")
        if _require_int(self.state_dim, label="state_dim", minimum=1) != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}")
        if _require_int(self.execution_horizon, label="execution_horizon", minimum=1) != EXECUTION_HORIZON:
            raise ValueError(f"execution_horizon must be {EXECUTION_HORIZON}")
        count = _require_int(self.record_count, label="record_count", minimum=1)
        if count != 2 * self.selection.episode_count:
            raise ValueError("record_count must be two chunks per selected episode")
        if self.payload_filename != PAYLOAD_FILENAME:
            raise ValueError(f"payload_filename must be {PAYLOAD_FILENAME!r}")
        _require_sha(self.payload_sha256, label="payload_sha256")
        _require_sha(self.content_sha256, label="content_sha256")
        if not isinstance(self.records, tuple) or len(self.records) != count:
            raise ValueError("manifest records must be a tuple with record_count entries")
        seen = set()
        for entry in self.records:
            if not isinstance(entry, dict) or set(entry) != set(_RECORD_METADATA_FIELDS):
                raise ValueError(f"manifest record fields must be exactly {sorted(_RECORD_METADATA_FIELDS)}")
            identifier = _require_nonempty(entry["fixture_id"], label="record fixture_id")
            if identifier in seen:
                raise ValueError(f"duplicate manifest record {identifier}")
            seen.add(identifier)
            if entry["chunk_role"] not in CHUNK_ROLES:
                raise ValueError(f"record {identifier} has an unknown chunk_role")
            indices = entry["memory_step_indices"]
            if not isinstance(indices, list) or len(indices) != MAX_FRAMES:
                raise ValueError(f"record {identifier} must list {MAX_FRAMES} memory step indices")
            step = _require_int(entry["step_idx"], label="record step_idx", minimum=MIN_CAUSAL_STEP)
            if [int(value) for value in indices] != list(even_sampling_indices(step)):
                raise ValueError(f"record {identifier} manifest indices disagree with the official rule")
            fit = _require_int(entry["fit_step"], label="record fit_step", minimum=MIN_CAUSAL_STEP)
            evaluation = _require_int(entry["eval_step"], label="record eval_step", minimum=MIN_CAUSAL_STEP)
            if evaluation - fit != EXECUTION_HORIZON:
                raise ValueError(f"record {identifier} cuts are not {EXECUTION_HORIZON} steps apart")
            if step != (fit if entry["chunk_role"] == FIT_ROLE else evaluation):
                raise ValueError(f"record {identifier} is not seated at its own cut")
            exec_start = _require_int(entry["exec_start_idx"], label="record exec_start_idx")
            execution_step = _require_int(entry["execution_step"], label="record execution_step")
            if execution_step != step - exec_start:
                raise ValueError(f"record {identifier} execution_step disagrees with its cut")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": int(self.schema_version),
            "dataset_source": self.dataset_source.to_dict(),
            "selection": self.selection.to_dict(),
            "selection_rule": self.selection_rule,
            "memory_frames": int(self.memory_frames),
            "image_shape": [int(value) for value in self.image_shape],
            "state_dim": int(self.state_dim),
            "execution_horizon": int(self.execution_horizon),
            "record_count": int(self.record_count),
            "payload_filename": self.payload_filename,
            "payload_sha256": self.payload_sha256,
            "content_sha256": self.content_sha256,
            "records": [dict(entry) for entry in self.records],
        }

    @classmethod
    def from_dict(cls, value: object) -> "FixtureBundleManifest":
        decoded = _require_exact_fields(value, cls, label="manifest")
        decoded["dataset_source"] = DatasetSource.from_dict(decoded["dataset_source"])
        decoded["selection"] = SelectionSpec.from_dict(decoded["selection"])
        shape = decoded["image_shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError("manifest image_shape must be a 3-element sequence")
        decoded["image_shape"] = tuple(int(item) for item in shape)
        entries = decoded["records"]
        if not isinstance(entries, (list, tuple)):
            raise ValueError("manifest records must be a list")
        decoded["records"] = tuple(dict(entry) if isinstance(entry, dict) else entry for entry in entries)
        result = cls(**decoded)
        result.validate()
        return result


def _payload_arrays(records: tuple[FixtureRecord, ...]) -> dict[str, np.ndarray]:
    """Flatten every record into ``"{fixture_id}.{field}"`` payload keys.

    One flat ``fixtures.npz`` is used rather than a directory of per-record
    files: it keeps the bundle to exactly two files, which is what the loader,
    the digest, and S3 staging all want.
    """

    payload: dict[str, np.ndarray] = {}
    for record in records:
        for name, array in record.arrays().items():
            key = f"{record.fixture_id}.{name}"
            if key in payload:
                raise ValueError(f"duplicate payload key {key}")
            payload[key] = np.ascontiguousarray(array)
    return payload


def write_fixture_bundle(records: object, out_dir: str | Path) -> FixtureBundleManifest:
    """Atomically seal ``manifest.json`` + ``fixtures.npz`` into ``out_dir``."""

    validated = validate_fixture_records(records)
    destination = Path(out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to replace an existing fixture bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload_arrays(validated)
    selection = validated[0].selection
    metadata = [record.metadata() for record in validated]
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        payload_path = staging / PAYLOAD_FILENAME
        with payload_path.open("wb") as stream:
            np.savez(stream, **payload)
        manifest = FixtureBundleManifest(
            schema_version=SCHEMA_VERSION,
            dataset_source=validated[0].source,
            selection=selection,
            selection_rule=selection.rule(),
            memory_frames=MAX_FRAMES,
            image_shape=IMAGE_SHAPE,
            state_dim=STATE_DIM,
            execution_horizon=EXECUTION_HORIZON,
            record_count=len(validated),
            payload_filename=PAYLOAD_FILENAME,
            payload_sha256=_sha256_file(payload_path),
            content_sha256=bundle_content_sha256(payload, metadata),
            records=tuple(metadata),
        )
        manifest.validate()
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # mkdtemp is 0700; a staged fixture bundle is meant to be readable.
        staging.chmod(0o755)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def read_fixture_manifest(bundle: str | Path) -> FixtureBundleManifest:
    bundle = Path(bundle)
    if not bundle.is_dir():
        raise FileNotFoundError(f"fixture bundle is not a directory: {bundle}")
    files = {path.name for path in bundle.iterdir()}
    if files != {MANIFEST_FILENAME, PAYLOAD_FILENAME}:
        raise ValueError(f"fixture bundle file set mismatch: {sorted(files)}")
    try:
        raw = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("fixture manifest is not valid UTF-8 JSON") from error
    return FixtureBundleManifest.from_dict(raw)


def load_fixture_bundle(bundle: str | Path) -> tuple[FixtureRecord, ...]:
    """Load a bundle after re-verifying digests and every invariant."""

    bundle = Path(bundle)
    manifest = read_fixture_manifest(bundle)
    payload_path = bundle / PAYLOAD_FILENAME
    if _sha256_file(payload_path) != manifest.payload_sha256:
        raise ValueError("fixture payload SHA256 mismatch")
    expected_keys = {f"{entry['fixture_id']}.{name}" for entry in manifest.records for name in _RECORD_ARRAY_FIELDS}
    records: list[FixtureRecord] = []
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError("fixture payload array set mismatch")
        arrays = {key: np.ascontiguousarray(payload[key]) for key in sorted(payload.files)}
    if bundle_content_sha256(arrays, [dict(entry) for entry in manifest.records]) != manifest.content_sha256:
        raise ValueError("fixture bundle content SHA256 mismatch")
    for entry in manifest.records:
        identifier = entry["fixture_id"]
        record = FixtureRecord(
            source=manifest.dataset_source,
            selection=manifest.selection,
            fixture_id=identifier,
            pair_id=_require_nonempty(entry["pair_id"], label="pair_id"),
            chunk_role=_require_nonempty(entry["chunk_role"], label="chunk_role"),
            episode_index=_require_int(entry["episode_index"], label="episode_index"),
            task_index=_require_int(entry["task_index"], label="task_index"),
            task=_require_nonempty(entry["task"], label="task"),
            episode_length=_require_int(entry["episode_length"], label="episode_length", minimum=1),
            step_idx=_require_int(entry["step_idx"], label="step_idx", minimum=MIN_CAUSAL_STEP),
            exec_start_idx=_require_int(entry["exec_start_idx"], label="exec_start_idx"),
            fit_step=_require_int(entry["fit_step"], label="fit_step", minimum=MIN_CAUSAL_STEP),
            eval_step=_require_int(entry["eval_step"], label="eval_step", minimum=MIN_CAUSAL_STEP),
            **{name: arrays[f"{identifier}.{name}"] for name in _RECORD_ARRAY_FIELDS},
        )
        record.validate()
        if record.metadata() != dict(entry):
            raise ValueError(f"manifest metadata for {identifier} disagrees with the payload")
        records.append(record)
    return validate_fixture_records(records)


def bundle_size_report(bundle: str | Path) -> dict[str, object]:
    """Payload/manifest byte accounting, for staging estimates."""

    bundle = Path(bundle)
    manifest = read_fixture_manifest(bundle)
    payload_bytes = (bundle / PAYLOAD_FILENAME).stat().st_size
    manifest_bytes = (bundle / MANIFEST_FILENAME).stat().st_size
    return {
        "bundle": str(bundle),
        "record_count": manifest.record_count,
        "episode_count": manifest.selection.episode_count,
        "episode_indices": sorted({int(entry["episode_index"]) for entry in manifest.records}),
        "task_indices": sorted({int(entry["task_index"]) for entry in manifest.records}),
        "exec_start_range": [
            min(int(entry["exec_start_idx"]) for entry in manifest.records),
            max(int(entry["exec_start_idx"]) for entry in manifest.records),
        ],
        "execution_steps": sorted({int(entry["execution_step"]) for entry in manifest.records}),
        "unreadable_source_episodes": len(manifest.dataset_source.unreadable_episodes),
        "payload_bytes": int(payload_bytes),
        "manifest_bytes": int(manifest_bytes),
        "total_bytes": int(payload_bytes + manifest_bytes),
        "bytes_per_record": int(payload_bytes // manifest.record_count),
        "payload_sha256": manifest.payload_sha256,
        "content_sha256": manifest.content_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="bundle directory to create")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODE_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--exec-offset",
        type=int,
        default=DEFAULT_EXEC_OFFSET,
        help="steps into each episode's own execution phase to place the fit cut",
    )
    args = parser.parse_args(argv)

    records = build_fixture_records(
        args.dataset_root,
        episodes=args.episodes,
        seed=args.seed,
        exec_offset=args.exec_offset,
    )
    write_fixture_bundle(records, args.out)
    reloaded = load_fixture_bundle(args.out)
    if len(reloaded) != len(records):
        raise RuntimeError("reloaded bundle does not match the written bundle")
    print(json.dumps(bundle_size_report(args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
