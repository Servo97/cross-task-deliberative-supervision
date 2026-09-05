#!/usr/bin/env python3
"""Materialize a Stage-S inventory by exact S3 VersionId, never by mutable-prefix sync.

The destination must be empty. Files are fetched concurrently, size/VersionId and optional ETag or
S3 SHA-256 checksum are verified, and each file is atomically renamed from ``.incomplete``. The
final local file set must equal the manifest exactly. For target50, the selected episode IDs are
independently re-derived from LeRobot metadata and unselected parquet/video payloads are forbidden.

Dataset SHAPE is parametrized, with defaults that are exactly the RoboCasa target50 contract:
  * ``--task-dir-glob``  (repeatable) — how task dirs are found under the selection root. RoboCasa
    nests ``{atomic,composite}/<Task>/<date>/lerobot`` (the two defaults); the ReMemBench export is
    FLAT, ``<Task>/<date>/lerobot``. Task name is ``path.parents[1].name`` in both layouts.
  * ``--selection-root`` — the dir the task-dir globs are applied to. Defaults to the download
    destination; ReMemBench object keys carry a leading ``train/`` component, so its selection root
    is ``<destination>/train``.
The per-artifact selection CHECK differs too: target50 re-derives a seed-0 uniform subsample, while
remembench_train13 asserts the metadata episode set equals the manifest's enumerated set exactly
(every demo is used — there is no subsample to re-derive).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

if __package__:
    from .validate_stage_s_inventory import REMEMBENCH_TRAIN_ARTIFACT, ROBOCASA_TARGET_ARTIFACT, validate_inventory
else:
    from validate_stage_s_inventory import REMEMBENCH_TRAIN_ARTIFACT, ROBOCASA_TARGET_ARTIFACT, validate_inventory


EPISODE_FILE = re.compile(r"^episode_(\d+)\.(parquet|mp4)$")
# RoboCasa target50's two-level {atomic,composite} nesting — the historical, unchanged default.
DEFAULT_TASK_DIR_GLOBS = ("atomic/*/*/lerobot", "composite/*/*/lerobot")


def _fail(message: str) -> None:
    raise ValueError(message)


def _parse_root(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        _fail(f"invalid S3 root URI {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _target_task_dirs(target_root: Path, task_dir_globs: "Sequence[str]" = DEFAULT_TASK_DIR_GLOBS) -> dict[str, Path]:
    paths: list[Path] = []
    for pattern in task_dir_globs:
        paths += sorted(target_root.glob(pattern))
    tasks: dict[str, Path] = {}
    for path in paths:
        task = path.parents[1].name
        if task in tasks:
            _fail(f"materialized target has duplicate task directory {task!r}")
        tasks[task] = path
    return tasks


def _metadata_ids(dataset_dir: Path) -> list[int]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    if not path.is_file():
        _fail(f"materialized target metadata missing: {path}")
    values: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get("episode_index")
            if not isinstance(value, int) or isinstance(value, bool):
                _fail(f"{path}:{line_number} has invalid episode_index={value!r}")
            values.append(value)
    if len(values) != len(set(values)):
        _fail(f"materialized target metadata has duplicate episode IDs: {path}")
    return values


def _match_task_dirs_to_selection(
    target_root: Path, selection: dict, task_dir_globs: Sequence[str]
) -> tuple[dict[str, Path], dict[str, set[int]]]:
    task_dirs = _target_task_dirs(target_root, task_dir_globs)
    selected = {record["task"]: set(record["episode_indices"]) for record in selection["tasks"]}
    if set(task_dirs) != set(selected):
        _fail(
            "materialized target task set differs from selection; "
            f"missing={sorted(set(selected) - set(task_dirs))} "
            f"extra={sorted(set(task_dirs) - set(selected))}"
        )
    return task_dirs, selected


def _check_episode_payloads(task: str, dataset_dir: Path, expected: set[int]) -> None:
    """Every selected episode has both payloads, and NO unselected episode payload was staged."""
    payloads: dict[int, set[str]] = {episode: set() for episode in expected}
    for path in dataset_dir.rglob("episode_*.*"):
        match = EPISODE_FILE.fullmatch(path.name)
        if match is None:
            continue
        episode = int(match.group(1))
        kind = match.group(2)
        if episode not in expected:
            _fail(f"unselected episode payload was staged: {path}")
        payloads[episode].add(kind)
    incomplete = sorted(episode for episode, kinds in payloads.items() if not {"parquet", "mp4"}.issubset(kinds))
    if incomplete:
        _fail(f"selected target task {task!r} lacks parquet/video payloads for episodes {incomplete[:10]}")


def validate_selected_target(
    target_root: str | Path,
    selection: dict,
    *,
    task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS,
) -> None:
    """Prove the local target payload is exactly the manifest's seed-0 50x150 selection."""
    target_root = Path(target_root)
    task_dirs, selected = _match_task_dirs_to_selection(target_root, selection, task_dir_globs)
    for task, dataset_dir in task_dirs.items():
        episode_ids = _metadata_ids(dataset_dir)
        shuffled = list(episode_ids)
        random.Random(selection["episode_subsample_seed"]).shuffle(shuffled)
        expected = set(shuffled[: selection["demos_per_task"]])
        if expected != selected[task]:
            _fail(f"target selection for {task!r} is not the metadata-derived seed-0 T30 set")
        _check_episode_payloads(task, dataset_dir, expected)


def validate_selected_remembench(
    target_root: str | Path,
    selection: dict,
    *,
    task_dir_globs: Sequence[str] = ("*/*/lerobot",),
) -> None:
    """Prove the local ReMemBench payload is exactly the manifest's enumerated train split.

    There is no seed or uniform per-task count to re-derive here — the train split is the complement
    of a held-out tail — so the independent check is that the staged LeRobot metadata enumerates
    EXACTLY the manifest's episode IDs (every demo used, none extra), plus the same
    both-payloads-present / no-unselected-payload rule target50 enforces.
    """
    target_root = Path(target_root)
    task_dirs, selected = _match_task_dirs_to_selection(target_root, selection, task_dir_globs)
    for task, dataset_dir in task_dirs.items():
        expected = selected[task]
        metadata = set(_metadata_ids(dataset_dir))
        if metadata != expected:
            _fail(
                f"remembench selection for {task!r} differs from staged metadata; "
                f"missing={sorted(expected - metadata)[:10]} extra={sorted(metadata - expected)[:10]}"
            )
        _check_episode_payloads(task, dataset_dir, expected)


def materialize_inventory(
    manifest_path: str | Path,
    *,
    expected_artifact: str,
    expected_root_s3: str,
    destination: str | Path,
    workers: int = 64,
    s3_client=None,
    selection_root: str | Path | None = None,
    task_dir_globs: Sequence[str] | None = None,
) -> dict[str, int]:
    summary = validate_inventory(
        manifest_path,
        expected_artifact=expected_artifact,
        expected_root_s3=expected_root_s3,
    )
    if workers <= 0:
        _fail("workers must be positive")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        _fail(f"inventory destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _parse_root(expected_root_s3)

    if s3_client is None:
        import boto3
        from botocore.config import Config

        s3_client = boto3.client(
            "s3",
            config=Config(
                max_pool_connections=workers,
                retries={"mode": "adaptive", "max_attempts": 10},
            ),
        )

    def fetch(record: dict) -> None:
        relative = record["key"]
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_name(destination_path.name + ".incomplete")
        source_key = f"{prefix}/{relative}" if prefix else relative
        request = {"Bucket": bucket, "Key": source_key}
        # version mode pins the exact S3 version; etag mode reads the current object and gates on
        # IfMatch so a mutated object fails (precondition) rather than silently materializing.
        if "version_id" in record:
            request["VersionId"] = record["version_id"]
        if "etag" in record:
            request["IfMatch"] = record["etag"]
        if "checksum_sha256" in record:
            request["ChecksumMode"] = "ENABLED"
        response = s3_client.get_object(**request)
        if "version_id" in record and response.get("VersionId") != record["version_id"]:
            _fail(f"S3 returned wrong VersionId for {relative}")
        if response.get("ContentLength") != record["size_bytes"]:
            _fail(f"S3 returned wrong ContentLength for {relative}")
        if "etag" in record and response.get("ETag") != record["etag"]:
            _fail(f"S3 returned wrong ETag for {relative}")
        if "checksum_sha256" in record:
            returned = response.get("ChecksumSHA256")
            expected = base64.b64encode(bytes.fromhex(record["checksum_sha256"])).decode("ascii")
            if returned != expected:
                _fail(f"S3 returned wrong SHA-256 checksum for {relative}")

        written = 0
        body = response["Body"]
        try:
            with temporary.open("xb") as stream:
                while True:
                    block = body.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    written += len(block)
        finally:
            body.close()
        if written != record["size_bytes"]:
            temporary.unlink(missing_ok=True)
            _fail(f"downloaded size mismatch for {relative}: {written}")
        temporary.replace(destination_path)

    with ThreadPoolExecutor(max_workers=min(workers, len(manifest["objects"]))) as pool:
        list(pool.map(fetch, manifest["objects"]))

    expected_files = {record["key"] for record in manifest["objects"]}
    actual_files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        _fail(
            "materialized inventory file set mismatch; "
            f"missing={sorted(expected_files - actual_files)[:10]} "
            f"extra={sorted(actual_files - expected_files)[:10]}"
        )
    # The selection root defaults to the download destination, which is exactly the historical
    # behavior; a dataset whose object keys carry a leading path component (ReMemBench: "train/")
    # points it at that subdirectory instead.
    selection_root = destination if selection_root is None else Path(selection_root).resolve()
    if expected_artifact == ROBOCASA_TARGET_ARTIFACT:
        validate_selected_target(
            selection_root,
            manifest["selection"],
            **({"task_dir_globs": tuple(task_dir_globs)} if task_dir_globs else {}),
        )
    elif expected_artifact == REMEMBENCH_TRAIN_ARTIFACT:
        validate_selected_remembench(
            selection_root,
            manifest["selection"],
            **({"task_dir_globs": tuple(task_dir_globs)} if task_dir_globs else {}),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--root-s3", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument(
        "--selection-root",
        default=None,
        help="dir the task-dir globs apply to (default: --destination)",
    )
    parser.add_argument(
        "--task-dir-glob",
        action="append",
        default=None,
        dest="task_dir_glob",
        help=f"repeatable task-directory glob (default: {' and '.join(DEFAULT_TASK_DIR_GLOBS)})",
    )
    args = parser.parse_args()
    summary = materialize_inventory(
        args.manifest,
        expected_artifact=args.artifact,
        expected_root_s3=args.root_s3,
        destination=args.destination,
        workers=args.workers,
        selection_root=args.selection_root,
        task_dir_globs=tuple(args.task_dir_glob) if args.task_dir_glob else None,
    )
    print(
        f"[stage-s-materialize] exact VersionIds verified artifact={args.artifact} "
        f"objects={summary['objects']} bytes={summary['bytes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
