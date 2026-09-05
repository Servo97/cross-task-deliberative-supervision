#!/usr/bin/env python3
"""Validate an immutable Stage-S omega cache against RoboCasa's exact seed-0 T30 keep-set.

The feature manifest schema is intentionally strict::

    {
      "schema_version": 1,
      "artifact": "pi05_workspace_omega",
      "encoder_id": "<64 lowercase hex>",
      "encoder_provenance": {
        "encoder_checkpoint": {"uri": "s3://...", "sha256": "<64hex>"},
        "workspace_model": {"architecture": "...", "config": {...}, "step": 123},
        "frozen_pi_feature_source": {"checkpoint_inventory_id": "<64hex>"},
        "source_features": {"manifest_id": "<64hex>"},
        "conditioning": {
          "subgoal_dropout": 1.0,
          "global_language_mode": "canonical_terse_task_instruction",
          "canonical_task_prompt_manifest_id": "<64hex>",
          "canonical_task_prompt_manifest_uri": "s3://.../<64hex>.json",
          "task_lang_table_manifest_sha256": "<64hex>"
        },
        "producing_code": {"sha256": "<64hex>"}
      },
      "dataset": {
        "name": "robocasa_target50_t30",
        "episode_subsample_seed": 0,
        "demos_per_task": 150
      },
      "tasks": [
        {
          "task": "<RoboCasa task directory>",
          "episodes": [
            {
              "episode_index": 123,
              "path": "<task>/demo_000123/w.npz",
              "size_bytes": 456,
              "sha256": "<64 lowercase hex>"
            }
          ]
        }
      ]
    }

RoboCasa production uses exactly 50 tasks and 150 seed-0 episodes per task. Every declared file is
checked by size and SHA-256 in parallel, and the cache may contain no undeclared ``w.npz`` files.

Encoder ID is the SHA-256 of canonical JSON for encoder_provenance. The encoder training features
and omega generation must both use the same SHA-pinned, demo-independent canonical terse per-task
prompt mapping. Historical per-demo Qwen-expanded encoders/caches, including the 7422/7500 cache
that skipped demos without subgoal JSON, are incompatible.

SECOND-DATASET PARAMETRIZATION (added 2026-08-03 for the ReMemBench chain). Three knobs, each
defaulting to the RoboCasa contract so the target50 path is byte-identical:

  * ``dataset_name``   — the ``dataset.name`` string (default ``robocasa_target50_t30``).
  * ``task_dir_globs`` — how task directories are found under ``target_root``. RoboCasa nests tasks
    under ``atomic``/``composite``; ReMemBench is flat ``<task>/<date>/lerobot``. The task name is
    ``path.parents[1].name`` in both layouts, so only the glob differs.
  * ``demos_per_task`` — an ``int`` (uniform, RoboCasa) OR a ``{task: count}`` mapping. ReMemBench
    keeps every demo of every task and the counts differ per task (9..44), so a uniform count cannot
    express it. The mapping is stored verbatim in ``dataset.demos_per_task``.

A non-default ``dataset_name`` ALSO requires a seventh ``dataset`` key inside ``encoder_provenance``
(see :func:`provenance_includes_dataset`): a second dataset conditioned on the same encoder weights
must not collide with the first dataset's ``encoder_id``. RoboCasa's already-published provenance
predates that key and keeps its exact six-key shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT = "pi05_workspace_omega"
DATASET_NAME = "robocasa_target50_t30"
GLOBAL_LANGUAGE_MODE = "canonical_terse_task_instruction"
OMEGA_DIM = 512
LANGUAGE_DIM = 2048
OMEGA_KEYS = frozenset({"w", "frame_indices", "lang_global"})
#: RoboCasa's two-level task layout. The task name is ``path.parents[1].name`` for every supported
#: layout, so a second dataset only overrides the glob (ReMemBench is flat: ``*/*/lerobot``).
DEFAULT_TASK_DIR_GLOBS = ("atomic/*/*/lerobot", "composite/*/*/lerobot")
#: The six-key ``encoder_provenance`` shape shipped for RoboCasa target50.
BASE_PROVENANCE_KEYS = frozenset(
    {
        "encoder_checkpoint",
        "workspace_model",
        "frozen_pi_feature_source",
        "source_features",
        "conditioning",
        "producing_code",
    }
)


def _fail(message: str) -> None:
    raise ValueError(message)


def provenance_includes_dataset(dataset_name: str) -> bool:
    """Whether ``encoder_provenance`` must carry a ``dataset`` block.

    The same frozen encoder can be run over a second dataset; without the dataset block in the
    provenance both caches would hash to the SAME ``encoder_id`` and overwrite each other's
    create-once S3 prefix. RoboCasa's target50 provenance is already published with six keys, so it
    keeps its exact shape and every other dataset commits its dataset block into the identity.
    """
    return dataset_name != DATASET_NAME


def demos_for_task(demos_per_task: int | Mapping[str, int], task: str) -> int:
    """Resolve the per-task demo count from an int (uniform) or a ``{task: count}`` mapping."""
    if isinstance(demos_per_task, Mapping):
        if task not in demos_per_task:
            _fail(f"demos_per_task mapping has no entry for task {task!r}")
        count = demos_per_task[task]
    else:
        count = demos_per_task
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        _fail(f"demos_per_task for task {task!r} must be a positive int; got {count!r}")
    return count


def canonical_demos_per_task(demos_per_task: int | Mapping[str, int]) -> int | dict[str, int]:
    """The exact JSON value stored in ``dataset.demos_per_task`` (int, or a plain sorted dict)."""
    if isinstance(demos_per_task, Mapping):
        return {task: int(demos_per_task[task]) for task in sorted(demos_per_task)}
    return int(demos_per_task)


def load_demos_per_task_map(path: str | Path) -> dict[str, int]:
    """Load a ``{task: count}`` JSON mapping (the ``--demos-per-task-map`` flag's payload)."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        _fail(f"demos-per-task map {path} must be a non-empty JSON object")
    out: dict[str, int] = {}
    for task, count in value.items():
        if not isinstance(task, str) or not task or "/" in task:
            _fail(f"demos-per-task map {path} has an invalid task key {task!r}")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            _fail(f"demos-per-task map {path} has an invalid count for {task!r}: {count!r}")
        out[task] = count
    return out


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        _fail(f"encoder provenance is not canonical JSON: {error}")


def _exact_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        _fail(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _provenance_sha256(
    provenance: object,
    *,
    expected_feature_source_inventory_id: str,
    expected_task_prompt_manifest_id: str,
    expected_task_prompt_manifest_uri: str,
    expected_dataset: dict | None = None,
) -> str:
    expected_provenance_keys = set(BASE_PROVENANCE_KEYS)
    if expected_dataset is not None:
        expected_provenance_keys.add("dataset")
    provenance = _exact_keys(provenance, expected_provenance_keys, "encoder_provenance")
    if expected_dataset is not None and provenance["dataset"] != expected_dataset:
        _fail(
            "encoder_provenance dataset block does not match the declared dataset contract: "
            f"{provenance['dataset']!r} != {expected_dataset!r}"
        )
    checkpoint = _exact_keys(provenance["encoder_checkpoint"], {"uri", "sha256"}, "encoder_checkpoint")
    if not isinstance(checkpoint["uri"], str) or not checkpoint["uri"].startswith("s3://"):
        _fail("encoder_checkpoint.uri must be an s3:// URI")
    if not isinstance(checkpoint["sha256"], str) or not HEX64.fullmatch(checkpoint["sha256"]):
        _fail("encoder_checkpoint.sha256 must be 64 lowercase hex characters")

    model = _exact_keys(provenance["workspace_model"], {"architecture", "config", "step"}, "workspace_model")
    if not isinstance(model["architecture"], str) or not model["architecture"].strip():
        _fail("workspace_model.architecture must be a non-empty string")
    if not isinstance(model["config"], dict) or not model["config"]:
        _fail("workspace_model.config must be a non-empty JSON object")
    if not isinstance(model["step"], int) or isinstance(model["step"], bool) or model["step"] < 0:
        _fail("workspace_model.step must be a non-negative integer")

    frozen = _exact_keys(
        provenance["frozen_pi_feature_source"],
        {"checkpoint_inventory_id"},
        "frozen_pi_feature_source",
    )
    inventory_id = frozen["checkpoint_inventory_id"]
    if inventory_id != expected_feature_source_inventory_id:
        _fail(
            "frozen_pi_feature_source.checkpoint_inventory_id does not match the Stage-S "
            f"initialization inventory: {inventory_id!r} != {expected_feature_source_inventory_id!r}"
        )

    source_features = _exact_keys(provenance["source_features"], {"manifest_id"}, "source_features")
    if not isinstance(source_features["manifest_id"], str) or not HEX64.fullmatch(source_features["manifest_id"]):
        _fail("source_features.manifest_id must be 64 lowercase hex characters")

    conditioning = _exact_keys(
        provenance["conditioning"],
        {
            "subgoal_dropout",
            "global_language_mode",
            "canonical_task_prompt_manifest_id",
            "canonical_task_prompt_manifest_uri",
            "task_lang_table_manifest_sha256",
        },
        "conditioning",
    )
    if not isinstance(conditioning["task_lang_table_manifest_sha256"], str) or not HEX64.fullmatch(
        conditioning["task_lang_table_manifest_sha256"]
    ):
        _fail("conditioning.task_lang_table_manifest_sha256 must be 64 lowercase hex characters")
    dropout = conditioning["subgoal_dropout"]
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not math.isfinite(dropout)
        or float(dropout) != 1.0
    ):
        _fail("conditioning.subgoal_dropout must be exactly 1.0 for Stage-S")
    if conditioning["global_language_mode"] != GLOBAL_LANGUAGE_MODE:
        _fail(
            "conditioning.global_language_mode must be "
            f"{GLOBAL_LANGUAGE_MODE!r}; per-demo teacher/expanded prompts are forbidden"
        )
    if conditioning["canonical_task_prompt_manifest_id"] != expected_task_prompt_manifest_id:
        _fail("encoder provenance task-prompt manifest ID does not match Stage-S")
    if conditioning["canonical_task_prompt_manifest_uri"] != expected_task_prompt_manifest_uri:
        _fail("encoder provenance task-prompt manifest URI does not match Stage-S")

    code = _exact_keys(provenance["producing_code"], {"sha256"}, "producing_code")
    if not isinstance(code["sha256"], str) or not HEX64.fullmatch(code["sha256"]):
        _fail("producing_code.sha256 must be 64 lowercase hex characters")
    return hashlib.sha256(_canonical_json(provenance).encode("utf-8")).hexdigest()


def add_dataset_shape_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared dataset-shape flags. Defaults reproduce the RoboCasa target50 contract.

    Every Stage-S CLI that touches a dataset layout uses these EXACT flags, so a second dataset is
    described in one place instead of drifting between producer and validator.
    """
    group = parser.add_argument_group("dataset shape (defaults = RoboCasa target50)")
    group.add_argument("--dataset-name", default=DATASET_NAME)
    group.add_argument(
        "--task-dir-glob",
        action="append",
        default=None,
        dest="task_dir_glob",
        help=(
            "task-directory glob under --target-root, repeatable "
            f"(default: {' and '.join(DEFAULT_TASK_DIR_GLOBS)}); task name is parents[1].name"
        ),
    )
    group.add_argument("--expected-tasks", type=int, default=50)
    group.add_argument("--demos-per-task", type=int, default=150)
    group.add_argument(
        "--demos-per-task-map",
        default=None,
        help="path to a {task: count} JSON mapping; overrides --demos-per-task when given",
    )
    group.add_argument("--seed", type=int, default=0)


def dataset_shape_kwargs(args: argparse.Namespace) -> dict:
    """Namespace -> the dataset-shape keyword arguments shared by every Stage-S entry point."""
    return {
        "dataset_name": args.dataset_name,
        "task_dir_globs": tuple(args.task_dir_glob or DEFAULT_TASK_DIR_GLOBS),
        "expected_tasks": args.expected_tasks,
        "demos_per_task": (
            load_demos_per_task_map(args.demos_per_task_map) if args.demos_per_task_map else args.demos_per_task
        ),
        "seed": args.seed,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_omega_archive(path: Path) -> None:
    """Validate the tensor contract consumed by the Stage-S OpenPI dataset.

    A content hash proves identity, not that the producer wrote a usable artifact. Validate every
    archive once during node setup so corrupt shapes/dtypes or a noncausal time grid cannot surface
    nondeterministically inside one of the persistent DataLoader workers.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != OMEGA_KEYS:
                _fail(f"omega archive keys for {path} must be exactly {sorted(OMEGA_KEYS)}; got {sorted(keys)}")
            omega = archive["w"]
            frame_indices = archive["frame_indices"]
            language = archive["lang_global"]
    except (OSError, EOFError, KeyError, ValueError) as error:
        _fail(f"invalid omega npz archive {path}: {error}")

    if omega.dtype != np.dtype(np.float16):
        _fail(f"omega tensor dtype for {path} must be float16; got {omega.dtype}")
    if omega.ndim != 2 or omega.shape[0] < 1 or omega.shape[1] != OMEGA_DIM:
        _fail(f"omega tensor shape for {path} must be [F,{OMEGA_DIM}] with F>=1; got {omega.shape}")
    if frame_indices.dtype != np.dtype(np.int64):
        _fail(f"frame_indices dtype for {path} must be int64; got {frame_indices.dtype}")
    if frame_indices.shape != (omega.shape[0],):
        _fail(f"frame_indices shape for {path} must be ({omega.shape[0]},); got {frame_indices.shape}")
    if language.dtype != np.dtype(np.float16):
        _fail(f"lang_global dtype for {path} must be float16; got {language.dtype}")
    if language.shape != (LANGUAGE_DIM,):
        _fail(f"lang_global shape for {path} must be ({LANGUAGE_DIM},); got {language.shape}")
    if not np.isfinite(omega).all():
        _fail(f"omega tensor for {path} contains NaN or infinity")
    if not np.isfinite(language).all():
        _fail(f"lang_global for {path} contains NaN or infinity")
    if frame_indices[0] < 0 or np.any(np.diff(frame_indices) <= 0):
        _fail(f"frame_indices for {path} must be nonnegative and strictly increasing")


def _dataset_task_dirs(
    target_root: Path, *, task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS
) -> dict[str, Path]:
    paths: list[Path] = []
    for pattern in task_dir_globs:
        paths += sorted(target_root.glob(pattern))
    tasks: dict[str, Path] = {}
    for path in paths:
        task = path.parents[1].name
        if task in tasks:
            _fail(f"dataset contains multiple lerobot directories for task {task!r}")
        tasks[task] = path
    return tasks


def _expected_episode_ids(dataset_dir: Path, *, demos_per_task: int, seed: int) -> set[int]:
    metadata = dataset_dir / "meta" / "episodes.jsonl"
    if not metadata.is_file():
        _fail(f"dataset episode metadata missing: {metadata}")
    episode_ids: list[int] = []
    with metadata.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = record.get("episode_index")
            if not isinstance(episode_index, int) or isinstance(episode_index, bool):
                _fail(f"{metadata}:{line_number} has invalid episode_index={episode_index!r}")
            episode_ids.append(episode_index)
    if len(episode_ids) != len(set(episode_ids)):
        _fail(f"dataset has duplicate episode_index values: {metadata}")
    if len(episode_ids) < demos_per_task:
        _fail(f"dataset task {dataset_dir} has {len(episode_ids)} episodes; need {demos_per_task}")
    random.Random(seed).shuffle(episode_ids)
    return set(episode_ids[:demos_per_task])


def _safe_declared_path(task: str, episode_index: int, value: object) -> str:
    expected = f"{task}/demo_{episode_index:06d}/w.npz"
    if value != expected:
        _fail(f"feature path must be exactly {expected!r}; got {value!r}")
    path = PurePosixPath(expected)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"unsafe feature path {expected!r}")
    return expected


def validate_manifest(
    *,
    features_root: str | Path,
    target_root: str | Path,
    manifest_path: str | Path,
    encoder_id: str,
    expected_feature_source_inventory_id: str,
    expected_task_prompt_manifest_id: str,
    expected_task_prompt_manifest_uri: str,
    expected_tasks: int = 50,
    demos_per_task: int | Mapping[str, int] = 150,
    seed: int = 0,
    workers: int = 32,
    dataset_name: str = DATASET_NAME,
    task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS,
) -> dict[str, int]:
    features_root = Path(features_root).resolve()
    target_root = Path(target_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not HEX64.fullmatch(encoder_id):
        _fail("expected encoder_id must be 64 lowercase hex characters")
    if not HEX64.fullmatch(expected_feature_source_inventory_id):
        _fail("expected feature-source inventory ID must be 64 lowercase hex characters")
    if not HEX64.fullmatch(expected_task_prompt_manifest_id):
        _fail("expected task-prompt manifest ID must be 64 lowercase hex characters")
    if not expected_task_prompt_manifest_uri.startswith("s3://"):
        _fail("expected task-prompt manifest URI must be an s3:// URI")
    if expected_tasks <= 0 or workers <= 0:
        _fail("expected_tasks and workers must be positive")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        _fail("dataset_name must be a non-empty string")
    expected_dataset = {
        "name": dataset_name,
        "episode_subsample_seed": seed,
        "demos_per_task": canonical_demos_per_task(demos_per_task),
    }

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        _fail("feature manifest schema_version must be 1")
    if manifest.get("artifact") != ARTIFACT:
        _fail(f"feature manifest artifact must be {ARTIFACT!r}")
    derived_encoder_id = _provenance_sha256(
        manifest.get("encoder_provenance"),
        expected_feature_source_inventory_id=expected_feature_source_inventory_id,
        expected_task_prompt_manifest_id=expected_task_prompt_manifest_id,
        expected_task_prompt_manifest_uri=expected_task_prompt_manifest_uri,
        expected_dataset=expected_dataset if provenance_includes_dataset(dataset_name) else None,
    )
    if derived_encoder_id != encoder_id:
        _fail(
            "encoder_id must equal canonical SHA-256 of encoder_provenance: "
            f"derived={derived_encoder_id} expected={encoder_id}"
        )
    if manifest.get("encoder_id") != encoder_id:
        _fail(f"feature manifest encoder_id={manifest.get('encoder_id')!r} does not match {encoder_id}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        _fail("feature manifest dataset must be an object")
    if dataset != expected_dataset:
        _fail(f"feature manifest dataset contract mismatch: {dataset!r} != {expected_dataset!r}")

    dataset_tasks = _dataset_task_dirs(target_root, task_dir_globs=task_dir_globs)
    if len(dataset_tasks) != expected_tasks:
        _fail(f"dataset must contain exactly {expected_tasks} unique tasks; got {len(dataset_tasks)}")
    task_records = manifest.get("tasks")
    if not isinstance(task_records, list) or len(task_records) != expected_tasks:
        _fail(f"feature manifest must enumerate exactly {expected_tasks} task records")

    declared_paths: set[str] = set()
    manifest_tasks: set[str] = set()
    file_contracts: list[tuple[Path, int, str]] = []
    for task_record in task_records:
        if not isinstance(task_record, dict):
            _fail("each feature manifest task record must be an object")
        task = task_record.get("task")
        if not isinstance(task, str) or task not in dataset_tasks:
            _fail(f"feature manifest has unknown task {task!r}")
        if task in manifest_tasks:
            _fail(f"feature manifest duplicates task {task!r}")
        manifest_tasks.add(task)
        task_demos = demos_for_task(demos_per_task, task)
        episodes = task_record.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != task_demos:
            _fail(f"task {task!r} must enumerate exactly {task_demos} episodes")

        expected_ids = _expected_episode_ids(dataset_tasks[task], demos_per_task=task_demos, seed=seed)
        declared_ids: set[int] = set()
        for episode in episodes:
            if not isinstance(episode, dict):
                _fail(f"task {task!r} has a non-object episode record")
            episode_index = episode.get("episode_index")
            if not isinstance(episode_index, int) or isinstance(episode_index, bool):
                _fail(f"task {task!r} has invalid episode_index={episode_index!r}")
            if episode_index in declared_ids:
                _fail(f"task {task!r} duplicates episode_index={episode_index}")
            declared_ids.add(episode_index)
            relative = _safe_declared_path(task, episode_index, episode.get("path"))
            if relative in declared_paths:
                _fail(f"feature manifest duplicates path {relative!r}")
            declared_paths.add(relative)
            size = episode.get("size_bytes")
            checksum = episode.get("sha256")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                _fail(f"feature manifest has invalid size_bytes for {relative}")
            if not isinstance(checksum, str) or not HEX64.fullmatch(checksum):
                _fail(f"feature manifest has invalid sha256 for {relative}")
            file_contracts.append((features_root / relative, size, checksum))
        if declared_ids != expected_ids:
            missing = sorted(expected_ids - declared_ids)[:10]
            extra = sorted(declared_ids - expected_ids)[:10]
            _fail(f"task {task!r} is not the seed-{seed} T30 keep-set; missing={missing} extra={extra}")

    if manifest_tasks != set(dataset_tasks):
        _fail(
            f"feature manifest task set differs from {dataset_name}; "
            f"missing={sorted(set(dataset_tasks) - manifest_tasks)} "
            f"extra={sorted(manifest_tasks - set(dataset_tasks))}"
        )
    expected_files = sum(demos_for_task(demos_per_task, task) for task in dataset_tasks)
    if len(declared_paths) != expected_files:
        _fail(f"feature manifest must declare exactly {expected_files} unique w.npz paths")
    actual_paths = {path.relative_to(features_root).as_posix() for path in features_root.rglob("w.npz")}
    if actual_paths != declared_paths:
        _fail(
            "omega cache file set differs from manifest; "
            f"missing={sorted(declared_paths - actual_paths)[:10]} "
            f"undeclared={sorted(actual_paths - declared_paths)[:10]}"
        )

    def check_file(contract: tuple[Path, int, str]) -> None:
        path, expected_size, expected_sha = contract
        if not path.is_file():
            _fail(f"declared feature file is missing: {path}")
        stat = path.stat()
        if stat.st_size != expected_size:
            _fail(f"feature size mismatch for {path}: expected={expected_size} actual={stat.st_size}")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            _fail(f"feature sha256 mismatch for {path}: expected={expected_sha} actual={actual_sha}")
        _validate_omega_archive(path)

    with ThreadPoolExecutor(max_workers=min(workers, len(file_contracts))) as executor:
        list(executor.map(check_file, file_contracts))
    return {"tasks": len(manifest_tasks), "episodes": len(declared_paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--feature-source-inventory-id", required=True)
    parser.add_argument("--task-prompt-manifest-id", required=True)
    parser.add_argument("--task-prompt-manifest-uri", required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    add_dataset_shape_args(parser)
    args = parser.parse_args()
    summary = validate_manifest(
        features_root=args.features_root,
        target_root=args.target_root,
        manifest_path=args.manifest,
        encoder_id=args.encoder_id,
        expected_feature_source_inventory_id=args.feature_source_inventory_id,
        expected_task_prompt_manifest_id=args.task_prompt_manifest_id,
        expected_task_prompt_manifest_uri=args.task_prompt_manifest_uri,
        workers=args.workers,
        **dataset_shape_kwargs(args),
    )
    print(
        f"[stage-s-features] verified encoder_id={args.encoder_id} "
        f"dataset={args.dataset_name} tasks={summary['tasks']} episodes={summary['episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
