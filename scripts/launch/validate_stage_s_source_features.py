#!/usr/bin/env python3
"""Validate the immutable Stage-S canonical-terse frozen-feature cache + its source manifest.

This is the FIRST artifact in the Stage-S representation chain: frozen pi0.5 tap features cached
with the demo-independent canonical terse prompt (never the Qwen-expanded prompt). It replaces the
historical ``pi_cache_features`` cache, whose per-demo expanded language is Stage-S-incompatible.

Manifest schema (strict; extra keys are rejected)::

    {
      "schema_version": 1,
      "artifact": "pi05_stage_s_source_features",
      "global_language_mode": "canonical_terse_task_instruction",
      "dataset": {"name": "robocasa_target50_t30", "episode_subsample_seed": 0, "demos_per_task": 150},
      "task_prompt_manifest": {"id": "<64hex>", "uri": "s3://.../<id>.json"},
      "feature_source": {"checkpoint_inventory_id": "<64hex>"},
      "frame_grid": {"stride": 8, "include_last": true, "image_hw": [256, 256]},
      "producing_code": {"sha256": "<64hex>"},
      "tasks": [{"task": "<RoboCasa task>", "episodes": [{
          "episode_index": 123,
          "patch_tokens": {"path": "<task>/demo_000123/patch_tokens.npy",
                            "size_bytes": 1, "sha256": "<64hex>", "shape": [F,192,2048], "dtype": "float16"},
          "feats": {"path": "<task>/demo_000123/feats.npz", "size_bytes": 1, "sha256": "<64hex>"},
          "frame_count": F,
          "teacher_labels": {"available": true, "path": "<task>/vlm_episode_pi_000123.npz", "sha256": "<64hex>"}
      }]}]
    }

The keep-set (seed-0 T30, 150 demos/task) is derived exactly as the omega validator derives it, by
importing its helpers — the two artifacts MUST select the same demonstrations. Feature availability
is separated from teacher-label availability: a demo with no Qwen label still produces frozen
features (its ``teacher_labels.available`` is false), so a missing label can never silently shrink
the frozen-feature coverage below 7,500/7,500.

The ``dataset_name`` / ``task_dir_globs`` / ``demos_per_task`` knobs (int or ``{task: count}``) are
the ones documented in ``validate_stage_s_policy_features`` and are imported from it, so the source
cache and the omega cache can never disagree about which demos a dataset contains. Defaults are the
RoboCasa target50 contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

try:
    from .stage_s_provenance import canonical_json_sha256, sha256_file
    from .validate_stage_s_policy_features import (
        DEFAULT_TASK_DIR_GLOBS,
        _dataset_task_dirs,
        _expected_episode_ids,
        add_dataset_shape_args,
        canonical_demos_per_task,
        dataset_shape_kwargs,
        demos_for_task,
    )
except ImportError:  # direct-script / pytest sys.path execution
    from stage_s_provenance import canonical_json_sha256, sha256_file
    from validate_stage_s_policy_features import (
        DEFAULT_TASK_DIR_GLOBS,
        _dataset_task_dirs,
        _expected_episode_ids,
        add_dataset_shape_args,
        canonical_demos_per_task,
        dataset_shape_kwargs,
        demos_for_task,
    )

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT = "pi05_stage_s_source_features"
DATASET_NAME = "robocasa_target50_t30"
GLOBAL_LANGUAGE_MODE = "canonical_terse_task_instruction"
PATCH_GRID = 192
BACKBONE_DIM = 2048
LANGUAGE_DIM = 2048
FRAME_STRIDE = 8
FEATS_KEYS = frozenset({"lang_per_frame", "frame_indices"})
TOP_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "global_language_mode",
        "dataset",
        "task_prompt_manifest",
        "feature_source",
        "frame_grid",
        "producing_code",
        "tasks",
    }
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _exact_keys(value: object, expected: frozenset[str] | set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _fail(f"{label} must be 64 lowercase hex characters")
    return value  # type: ignore[return-value]


def source_manifest_id(manifest: dict) -> str:
    """The manifest identity = canonical-JSON SHA-256 of the whole object (no embedded id field)."""
    return canonical_json_sha256(manifest)


def _validate_frame_grid(frame_indices: np.ndarray, path: Path) -> None:
    if frame_indices.dtype != np.dtype(np.int64):
        _fail(f"frame_indices dtype for {path} must be int64; got {frame_indices.dtype}")
    if frame_indices.ndim != 1 or frame_indices.shape[0] < 1:
        _fail(f"frame_indices for {path} must be 1-D with at least one frame")
    if int(frame_indices[0]) != 0:
        _fail(f"frame_indices for {path} must start at 0")
    if frame_indices.shape[0] > 1:
        diffs = np.diff(frame_indices)
        if np.any(diffs <= 0):
            _fail(f"frame_indices for {path} must be strictly increasing")
        if np.any(diffs[:-1] != FRAME_STRIDE):
            _fail(f"frame_indices for {path} must step by {FRAME_STRIDE} before the final frame")
        if int(diffs[-1]) < 1 or int(diffs[-1]) > FRAME_STRIDE:
            _fail(f"frame_indices for {path} final step must be within [1,{FRAME_STRIDE}]")


def _validate_feats(path: Path, frame_count: int) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != FEATS_KEYS:
                _fail(f"feats keys for {path} must be exactly {sorted(FEATS_KEYS)}; got {sorted(keys)}")
            lang = archive["lang_per_frame"]
            frame_indices = archive["frame_indices"]
    except (OSError, EOFError, KeyError, ValueError) as error:
        _fail(f"invalid feats npz {path}: {error}")
    if lang.dtype != np.dtype(np.float16):
        _fail(f"lang_per_frame dtype for {path} must be float16; got {lang.dtype}")
    if lang.ndim != 2 or lang.shape[1] != LANGUAGE_DIM:
        _fail(f"lang_per_frame shape for {path} must be [F,{LANGUAGE_DIM}]; got {lang.shape}")
    if lang.shape[0] != frame_count:
        _fail(f"lang_per_frame frame count for {path} ({lang.shape[0]}) != declared {frame_count}")
    if not np.isfinite(lang).all():
        _fail(f"lang_per_frame for {path} contains NaN or infinity")
    _validate_frame_grid(frame_indices, path)
    if frame_indices.shape[0] != frame_count:
        _fail(f"frame_indices count for {path} ({frame_indices.shape[0]}) != declared {frame_count}")


def _validate_patch_header(path: Path, frame_count: int) -> None:
    """Cheap header-only shape/dtype check (mmap reads no payload); bytes are pinned by sha256."""
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, EOFError, ValueError) as error:
        _fail(f"invalid patch_tokens npy {path}: {error}")
    if array.dtype != np.dtype(np.float16):
        _fail(f"patch_tokens dtype for {path} must be float16; got {array.dtype}")
    if array.shape != (frame_count, PATCH_GRID, BACKBONE_DIM):
        _fail(f"patch_tokens shape for {path} must be ({frame_count},{PATCH_GRID},{BACKBONE_DIM}); got {array.shape}")


def _safe_rel(task: str, episode_index: int, suffix: str, value: object) -> str:
    expected = f"{task}/demo_{episode_index:06d}/{suffix}"
    if value != expected:
        _fail(f"feature path must be exactly {expected!r}; got {value!r}")
    path = PurePosixPath(expected)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"unsafe feature path {expected!r}")
    return expected


def _label_rel(task: str, episode_index: int, value: object) -> str:
    expected = f"{task}/vlm_episode_pi_{episode_index:06d}.npz"
    if value != expected:
        _fail(f"teacher label path must be exactly {expected!r}; got {value!r}")
    return expected


def validate_source_manifest(
    *,
    features_root: str | Path,
    target_root: str | Path,
    manifest_path: str | Path,
    labels_root: str | Path,
    expected_task_prompt_manifest_id: str,
    expected_feature_source_inventory_id: str,
    expected_tasks: int = 50,
    demos_per_task: int | Mapping[str, int] = 150,
    seed: int = 0,
    workers: int = 32,
    dataset_name: str = DATASET_NAME,
    task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS,
) -> dict[str, int]:
    features_root = Path(features_root).resolve()
    target_root = Path(target_root).resolve()
    labels_root = Path(labels_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    _hex64(expected_task_prompt_manifest_id, "expected task-prompt manifest id")
    _hex64(expected_feature_source_inventory_id, "expected feature-source inventory id")
    if expected_tasks <= 0 or workers <= 0:
        _fail("expected_tasks and workers must be positive")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        _fail("dataset_name must be a non-empty string")

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    _exact_keys(manifest, TOP_KEYS, "source-feature manifest")
    if manifest["schema_version"] != 1:
        _fail("source-feature manifest schema_version must be 1")
    if manifest["artifact"] != ARTIFACT:
        _fail(f"source-feature manifest artifact must be {ARTIFACT!r}")
    if manifest["global_language_mode"] != GLOBAL_LANGUAGE_MODE:
        _fail(
            "source-feature manifest global_language_mode must be "
            f"{GLOBAL_LANGUAGE_MODE!r}; expanded/teacher prompts are forbidden"
        )
    dataset = _exact_keys(manifest["dataset"], {"name", "episode_subsample_seed", "demos_per_task"}, "dataset")
    expected_dataset = {
        "name": dataset_name,
        "episode_subsample_seed": seed,
        "demos_per_task": canonical_demos_per_task(demos_per_task),
    }
    if dataset != expected_dataset:
        _fail(f"source-feature manifest dataset contract mismatch: {dataset!r} != {expected_dataset!r}")
    prompt_manifest = _exact_keys(manifest["task_prompt_manifest"], {"id", "uri"}, "task_prompt_manifest")
    if prompt_manifest["id"] != expected_task_prompt_manifest_id:
        _fail("source-feature manifest task_prompt_manifest.id does not match Stage-S")
    if not isinstance(prompt_manifest["uri"], str) or not prompt_manifest["uri"].startswith("s3://"):
        _fail("task_prompt_manifest.uri must be an s3:// URI")
    feature_source = _exact_keys(manifest["feature_source"], {"checkpoint_inventory_id"}, "feature_source")
    if feature_source["checkpoint_inventory_id"] != expected_feature_source_inventory_id:
        _fail("source-feature manifest feature_source.checkpoint_inventory_id does not match Stage-S")
    frame_grid = _exact_keys(manifest["frame_grid"], {"stride", "include_last", "image_hw"}, "frame_grid")
    if frame_grid != {"stride": FRAME_STRIDE, "include_last": True, "image_hw": [256, 256]}:
        _fail(f"source-feature manifest frame_grid contract mismatch: {frame_grid!r}")
    _hex64(_exact_keys(manifest["producing_code"], {"sha256"}, "producing_code")["sha256"], "producing_code.sha256")

    dataset_tasks = _dataset_task_dirs(target_root, task_dir_globs=task_dir_globs)
    if len(dataset_tasks) != expected_tasks:
        _fail(f"dataset must contain exactly {expected_tasks} unique tasks; got {len(dataset_tasks)}")
    task_records = manifest["tasks"]
    if not isinstance(task_records, list) or len(task_records) != expected_tasks:
        _fail(f"source-feature manifest must enumerate exactly {expected_tasks} task records")

    declared_patch: set[str] = set()
    declared_feats: set[str] = set()
    manifest_tasks: set[str] = set()
    file_contracts: list[tuple[Path, int, str]] = []
    labeled = 0
    for task_record in task_records:
        task = _exact_keys(task_record, {"task", "episodes"}, "task record")["task"]
        if not isinstance(task, str) or task not in dataset_tasks:
            _fail(f"source-feature manifest has unknown task {task!r}")
        if task in manifest_tasks:
            _fail(f"source-feature manifest duplicates task {task!r}")
        manifest_tasks.add(task)
        task_demos = demos_for_task(demos_per_task, task)
        episodes = task_record["episodes"]
        if not isinstance(episodes, list) or len(episodes) != task_demos:
            _fail(f"task {task!r} must enumerate exactly {task_demos} episodes")
        expected_ids = _expected_episode_ids(dataset_tasks[task], demos_per_task=task_demos, seed=seed)
        declared_ids: set[int] = set()
        for episode in episodes:
            _exact_keys(
                episode,
                {"episode_index", "patch_tokens", "feats", "frame_count", "teacher_labels"},
                "episode record",
            )
            episode_index = episode["episode_index"]
            if not isinstance(episode_index, int) or isinstance(episode_index, bool):
                _fail(f"task {task!r} has invalid episode_index={episode_index!r}")
            if episode_index in declared_ids:
                _fail(f"task {task!r} duplicates episode_index={episode_index}")
            declared_ids.add(episode_index)
            frame_count = episode["frame_count"]
            if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
                _fail(f"task {task!r} demo {episode_index} has invalid frame_count")

            patch = _exact_keys(
                episode["patch_tokens"],
                {"path", "size_bytes", "sha256", "shape", "dtype"},
                "patch_tokens",
            )
            patch_rel = _safe_rel(task, episode_index, "patch_tokens.npy", patch["path"])
            if patch_rel in declared_patch:
                _fail(f"source-feature manifest duplicates patch path {patch_rel!r}")
            declared_patch.add(patch_rel)
            if patch["dtype"] != "float16" or patch["shape"] != [frame_count, PATCH_GRID, BACKBONE_DIM]:
                _fail(f"patch_tokens declared shape/dtype for {patch_rel} is not canonical")
            _size = patch["size_bytes"]
            if not isinstance(_size, int) or isinstance(_size, bool) or _size <= 0:
                _fail(f"patch_tokens size_bytes for {patch_rel} must be a positive int")
            file_contracts.append((features_root / patch_rel, _size, _hex64(patch["sha256"], "patch sha")))

            feats = _exact_keys(episode["feats"], {"path", "size_bytes", "sha256"}, "feats")
            feats_rel = _safe_rel(task, episode_index, "feats.npz", feats["path"])
            if feats_rel in declared_feats:
                _fail(f"source-feature manifest duplicates feats path {feats_rel!r}")
            declared_feats.add(feats_rel)
            _fsize = feats["size_bytes"]
            if not isinstance(_fsize, int) or isinstance(_fsize, bool) or _fsize <= 0:
                _fail(f"feats size_bytes for {feats_rel} must be a positive int")
            file_contracts.append((features_root / feats_rel, _fsize, _hex64(feats["sha256"], "feats sha")))

            labels = _exact_keys(episode["teacher_labels"], {"available", "path", "sha256"}, "teacher_labels")
            available = labels["available"]
            if not isinstance(available, bool):
                _fail(f"teacher_labels.available for {task}/{episode_index} must be a bool")
            label_rel = _label_rel(task, episode_index, labels["path"])
            label_path = labels_root / label_rel
            if available:
                labeled += 1
                if not label_path.is_file():
                    _fail(f"teacher_labels.available=true but label file is missing: {label_path}")
                if sha256_file(label_path) != _hex64(labels["sha256"], "teacher label sha"):
                    _fail(f"teacher label sha256 mismatch for {label_path}")
            else:
                if labels["sha256"] is not None:
                    _fail(f"teacher_labels.available=false must have sha256=null for {label_rel}")
                if label_path.is_file():
                    _fail(f"teacher_labels.available=false but a label file exists: {label_path}")
        if declared_ids != expected_ids:
            missing = sorted(expected_ids - declared_ids)[:10]
            extra = sorted(declared_ids - expected_ids)[:10]
            _fail(f"task {task!r} is not the seed-{seed} T30 keep-set; missing={missing} extra={extra}")

    if manifest_tasks != set(dataset_tasks):
        _fail(
            f"source-feature manifest task set differs from {dataset_name}; "
            f"missing={sorted(set(dataset_tasks) - manifest_tasks)} "
            f"extra={sorted(manifest_tasks - set(dataset_tasks))}"
        )
    expected_files = sum(demos_for_task(demos_per_task, task) for task in dataset_tasks)
    if len(declared_patch) != expected_files or len(declared_feats) != expected_files:
        _fail(f"source-feature manifest must declare exactly {expected_files} patch and feats files")
    on_disk_patch = {p.relative_to(features_root).as_posix() for p in features_root.rglob("patch_tokens.npy")}
    on_disk_feats = {p.relative_to(features_root).as_posix() for p in features_root.rglob("feats.npz")}
    if on_disk_patch != declared_patch:
        _fail(
            "patch cache file set differs from manifest; "
            f"missing={sorted(declared_patch - on_disk_patch)[:10]} "
            f"undeclared={sorted(on_disk_patch - declared_patch)[:10]}"
        )
    if on_disk_feats != declared_feats:
        _fail(
            "feats cache file set differs from manifest; "
            f"missing={sorted(declared_feats - on_disk_feats)[:10]} "
            f"undeclared={sorted(on_disk_feats - declared_feats)[:10]}"
        )

    frame_by_feats = {
        f"{r['task']}/demo_{e['episode_index']:06d}/feats.npz": e["frame_count"]
        for r in task_records
        for e in r["episodes"]
    }

    def check_file(contract: tuple[Path, int, str]) -> None:
        path, expected_size, expected_sha = contract
        if not path.is_file():
            _fail(f"declared feature file is missing: {path}")
        if path.stat().st_size != expected_size:
            _fail(f"feature size mismatch for {path}: expected={expected_size} actual={path.stat().st_size}")
        if sha256_file(path) != expected_sha:
            _fail(f"feature sha256 mismatch for {path}")
        rel = path.relative_to(features_root).as_posix()
        if path.name == "feats.npz":
            _validate_feats(path, frame_by_feats[rel])
        else:
            _validate_patch_header(path, frame_by_feats[rel.replace("patch_tokens.npy", "feats.npz")])

    with ThreadPoolExecutor(max_workers=min(workers, len(file_contracts))) as executor:
        list(executor.map(check_file, file_contracts))
    return {"tasks": len(manifest_tasks), "episodes": expected_files, "labeled": labeled}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-prompt-manifest-id", required=True)
    parser.add_argument("--feature-source-inventory-id", required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    add_dataset_shape_args(parser)
    args = parser.parse_args()
    summary = validate_source_manifest(
        features_root=args.features_root,
        target_root=args.target_root,
        labels_root=args.labels_root,
        manifest_path=args.manifest,
        expected_task_prompt_manifest_id=args.task_prompt_manifest_id,
        expected_feature_source_inventory_id=args.feature_source_inventory_id,
        workers=args.workers,
        **dataset_shape_kwargs(args),
    )
    print(
        f"[stage-s-source-features] verified dataset={args.dataset_name} tasks={summary['tasks']} "
        f"episodes={summary['episodes']} labeled={summary['labeled']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
