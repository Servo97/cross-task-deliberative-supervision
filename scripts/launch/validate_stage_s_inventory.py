#!/usr/bin/env python3
"""Validate a SHA-pinned Stage-S S3 object-inventory manifest.

The manifest bytes are content-addressed by the launcher and checked by the node entrypoint.  This
validator checks the semantic contract without reading or hashing the dataset/checkpoint payloads::

    {
      "schema_version": 1,
      "artifact": "pi05_h300_mg_init" | "robocasa_target50" | "remembench_train13",
      "root_s3": "s3://bucket/immutable/prefix",
      "selection": null | {
        "name": "seed0_t30",
        "episode_subsample_seed": 0,
        "demos_per_task": 150,
        "tasks": [{"task": "...", "episode_indices": [0, ...]}]
      } | {
        "name": "remembench_train_tail02",
        "kind": "remembench_tail_fraction_complement",
        "fraction": 0.2,
        "minimum": 3,
        "tasks": [{"task": "...", "episode_indices": [0, ...]}]
      },
      "objects": [
        {
          "key": "relative/object/key",
          "size_bytes": 123,
          "version_id": "required-non-null-S3-VersionId",
          "checksum_sha256": "optional-64-lowercase-hex",
          "etag": "optional-S3-ETag"
        }
      ]
    }

Every object requires an exact, non-null S3 VersionId. Payloads are materialized with GetObject at
that version; checksums and ETags are optional defense-in-depth metadata. The raw manifest SHA-256
is the inventory ID.

Per-artifact selection contracts (``_validate_selection``):
  * ``pi05_h300_mg_init``  — selection MUST be null (a checkpoint has no episode selection).
  * ``robocasa_target50``  — exactly 50 x 150 seed-0 T30 episode IDs.
  * ``remembench_train13`` — the 13-task ReMemBench train split. Its selection is a TAIL-FRACTION
    COMPLEMENT (the held-out eval tail is the last ``fraction`` of each task, at least ``minimum``
    demos; train is everything else), so per-task counts legitimately DIFFER (9..44) and no seed or
    uniform demos_per_task exists. Each task therefore only has to enumerate a non-empty, unique,
    nonnegative episode-ID list.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROBOCASA_TARGET_ARTIFACT = "robocasa_target50"
REMEMBENCH_TRAIN_ARTIFACT = "remembench_train13"
REMEMBENCH_SELECTION_KIND = "remembench_tail_fraction_complement"
REMEMBENCH_EXPECTED_TASKS = 13
ARTIFACTS = frozenset({"pi05_h300_mg_init", ROBOCASA_TARGET_ARTIFACT, REMEMBENCH_TRAIN_ARTIFACT})
TOP_LEVEL_KEYS = frozenset({"schema_version", "artifact", "root_s3", "content_addressing", "selection", "objects"})
CONTENT_ADDRESSING = frozenset({"version_id", "etag"})
# The immutability anchor is EITHER the S3 VersionId (on a versioned bucket) OR the object ETag
# (on an unversioned bucket that the study writes create-once). Exactly one is the per-object anchor;
# checksum_sha256 is always optional extra evidence.
OBJECT_REQUIRED_BY_MODE = {
    "version_id": frozenset({"key", "size_bytes", "version_id"}),
    "etag": frozenset({"key", "size_bytes", "etag"}),
}
OBJECT_ALLOWED_BY_MODE = {
    "version_id": frozenset({"key", "size_bytes", "version_id", "checksum_sha256", "etag"}),
    "etag": frozenset({"key", "size_bytes", "etag", "checksum_sha256"}),
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _safe_key(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(character) < 32 for character in value):
        _fail(f"invalid inventory object key {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value.endswith("/") or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"inventory object key must be a safe relative file key: {value!r}")
    return value


def _episode_indices(task: str, episodes: object, *, expected: int | None) -> None:
    """Shared episode-ID list contract: unique nonnegative ints, and (optionally) an exact count."""
    if (
        not isinstance(episodes, list)
        or not episodes
        or (expected is not None and len(episodes) != expected)
        or len(set(episodes)) != len(episodes)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in episodes)
    ):
        _fail(
            f"selection task {task!r} must have "
            + (f"{expected} " if expected is not None else "a non-empty set of ")
            + "unique nonnegative episode IDs"
        )


def _validate_remembench_selection(selection: object) -> None:
    """The 13-task ReMemBench train split: a tail-fraction complement, so counts vary per task."""
    expected_keys = {"name", "kind", "fraction", "minimum", "tasks"}
    if not isinstance(selection, dict) or set(selection) != expected_keys:
        _fail(f"remembench inventory selection keys must be exactly {sorted(expected_keys)}")
    if selection["kind"] != REMEMBENCH_SELECTION_KIND:
        _fail(f"remembench inventory selection kind must be {REMEMBENCH_SELECTION_KIND!r}")
    name = selection["name"]
    if not isinstance(name, str) or not name:
        _fail(f"remembench inventory selection name must be a non-empty string, got {name!r}")
    fraction = selection["fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        _fail(f"remembench selection fraction must be strictly between 0 and 1, got {fraction!r}")
    minimum = selection["minimum"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        _fail(f"remembench selection minimum must be a positive integer, got {minimum!r}")
    tasks = selection["tasks"]
    if not isinstance(tasks, list) or len(tasks) != REMEMBENCH_EXPECTED_TASKS:
        _fail(f"remembench inventory selection must enumerate exactly {REMEMBENCH_EXPECTED_TASKS} tasks")
    names: set[str] = set()
    for record in tasks:
        if not isinstance(record, dict) or set(record) != {"task", "episode_indices"}:
            _fail("each remembench selection task must contain task and episode_indices exactly")
        task = record["task"]
        if not isinstance(task, str) or not task or "/" in task or task in names:
            _fail(f"invalid or duplicate remembench selection task {task!r}")
        names.add(task)
        # Per-task counts are NOT uniform (the eval tail is a fraction with a floor), so only the
        # structural contract is checkable here; the exact IDs are re-derived at materialization.
        _episode_indices(task, record["episode_indices"], expected=None)


def _validate_selection(artifact: str, selection: object) -> None:
    if artifact == "pi05_h300_mg_init":
        if selection is not None:
            _fail("initialization inventory selection must be null")
        return
    if artifact == REMEMBENCH_TRAIN_ARTIFACT:
        _validate_remembench_selection(selection)
        return
    expected_keys = {"name", "episode_subsample_seed", "demos_per_task", "tasks"}
    if not isinstance(selection, dict) or set(selection) != expected_keys:
        _fail(f"target inventory selection keys must be exactly {sorted(expected_keys)}")
    if selection["name"] != "seed0_t30" or selection["episode_subsample_seed"] != 0:
        _fail("target inventory selection must be seed0_t30 with seed 0")
    if selection["demos_per_task"] != 150:
        _fail("target inventory selection must contain 150 demos per task")
    tasks = selection["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 50:
        _fail("target inventory selection must enumerate exactly 50 tasks")
    names: set[str] = set()
    for record in tasks:
        if not isinstance(record, dict) or set(record) != {"task", "episode_indices"}:
            _fail("each target selection task must contain task and episode_indices exactly")
        task = record["task"]
        episodes = record["episode_indices"]
        if not isinstance(task, str) or not task or "/" in task or task in names:
            _fail(f"invalid or duplicate target selection task {task!r}")
        names.add(task)
        if (
            not isinstance(episodes, list)
            or len(episodes) != 150
            or len(set(episodes)) != 150
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in episodes)
        ):
            _fail(f"target selection task {task!r} must have 150 unique nonnegative episode IDs")


def validate_inventory(
    manifest_path: str | Path,
    *,
    expected_artifact: str,
    expected_root_s3: str,
) -> dict[str, int]:
    if expected_artifact not in ARTIFACTS:
        _fail(f"unsupported expected artifact {expected_artifact!r}")
    if not expected_root_s3.startswith("s3://") or expected_root_s3.endswith("/"):
        _fail("expected root_s3 must be an s3:// URI without a trailing slash")
    with Path(manifest_path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or set(manifest) != TOP_LEVEL_KEYS:
        _fail(f"inventory top-level keys must be exactly {sorted(TOP_LEVEL_KEYS)}")
    if manifest["schema_version"] != 1:
        _fail("inventory schema_version must be 1")
    if manifest["artifact"] != expected_artifact:
        _fail(f"inventory artifact={manifest['artifact']!r} does not match {expected_artifact!r}")
    if manifest["root_s3"] != expected_root_s3:
        _fail(f"inventory root_s3={manifest['root_s3']!r} does not match {expected_root_s3!r}")
    mode = manifest["content_addressing"]
    if mode not in CONTENT_ADDRESSING:
        _fail(f"inventory content_addressing must be one of {sorted(CONTENT_ADDRESSING)}")
    required_keys = OBJECT_REQUIRED_BY_MODE[mode]
    allowed_keys = OBJECT_ALLOWED_BY_MODE[mode]
    _validate_selection(expected_artifact, manifest["selection"])
    objects = manifest["objects"]
    if not isinstance(objects, list) or not objects:
        _fail("inventory objects must be a non-empty list")

    keys: set[str] = set()
    total_bytes = 0
    for index, record in enumerate(objects):
        if not isinstance(record, dict):
            _fail(f"inventory object {index} must be an object")
        record_keys = set(record)
        if not required_keys.issubset(record_keys) or not record_keys.issubset(allowed_keys):
            _fail(
                f"inventory object {index} keys must contain {sorted(required_keys)} "
                f"and use only {sorted(allowed_keys)} for content_addressing={mode!r}"
            )
        key = _safe_key(record["key"])
        if key in keys:
            _fail(f"inventory duplicates object key {key!r}")
        keys.add(key)
        size = record["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _fail(f"inventory object {key!r} has invalid size_bytes={size!r}")
        total_bytes += size

        if mode == "version_id":
            version_id = record["version_id"]
            if not isinstance(version_id, str) or not version_id or version_id == "null" or len(version_id) > 1024:
                _fail(f"inventory object {key!r} requires a non-null S3 version_id")
        else:  # etag mode: the ETag is the immutability anchor
            etag = record["etag"]
            if not isinstance(etag, str) or not etag.strip() or len(etag) > 1024:
                _fail(f"inventory object {key!r} requires a non-empty ETag in etag mode")
        if "checksum_sha256" in record:
            value = record["checksum_sha256"]
            if not isinstance(value, str) or not HEX64.fullmatch(value):
                _fail(f"inventory object {key!r} has invalid checksum_sha256")
        if "etag" in record:
            value = record["etag"]
            if not isinstance(value, str) or not value.strip() or len(value) > 1024:
                _fail(f"inventory object {key!r} has invalid etag")
    return {"objects": len(keys), "bytes": total_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True, choices=sorted(ARTIFACTS))
    parser.add_argument("--root-s3", required=True)
    args = parser.parse_args()
    summary = validate_inventory(
        args.manifest,
        expected_artifact=args.artifact,
        expected_root_s3=args.root_s3,
    )
    print(
        f"[stage-s-inventory] verified artifact={args.artifact} objects={summary['objects']} bytes={summary['bytes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
