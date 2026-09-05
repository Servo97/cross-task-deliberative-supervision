#!/usr/bin/env python3
"""Build immutable Stage-S S3 VersionId inventories without downloading payloads.

The builder requires an explicitly versioned bucket, records the exact current VersionId of every
selected object, and writes canonical JSON under a content-addressed filename. Target selection is
re-derived from each task's pinned ``meta/episodes.jsonl`` using seed 0 and includes all static task
metadata plus only the selected parquet/video episode payloads.

This command performs S3 reads. The CLI therefore fails before importing boto3 unless
``--confirm-read`` is supplied after explicit user approval.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

try:
    from .validate_stage_s_inventory import validate_inventory
except ImportError:
    from validate_stage_s_inventory import validate_inventory


ARTIFACT_NAMESPACES = {
    "pi05_h300_mg_init": "init",
    "robocasa_target50": "data",
}
EPISODE_PAYLOAD = re.compile(r"^episode_(\d+)\.(parquet|mp4)$")


def _fail(message: str) -> None:
    raise ValueError(message)


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _parse_s3_root(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or uri.endswith("/"):
        _fail("root_s3 must be an s3:// URI without a trailing slash")
    return parsed.netloc, parsed.path.lstrip("/")


def _relative_key(key: str, prefix: str) -> str:
    marker = f"{prefix}/" if prefix else ""
    if not key.startswith(marker) or key == prefix:
        _fail(f"S3 key {key!r} is outside inventory root prefix {prefix!r}")
    return key[len(marker) :]


def _list_current_keys(client, *, bucket: str, prefix: str) -> list[str]:
    request = {"Bucket": bucket, "Prefix": f"{prefix}/" if prefix else ""}
    keys: list[str] = []
    while True:
        response = client.list_objects_v2(**request)
        keys.extend(record["Key"] for record in response.get("Contents", ()))
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            _fail("truncated S3 listing omitted NextContinuationToken")
        request["ContinuationToken"] = token
    if len(keys) != len(set(keys)):
        _fail("S3 current-object listing contains duplicate keys")
    return sorted(keys)


def _head_record(client, *, bucket: str, key: str, prefix: str, mode: str) -> dict:
    response = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    size = response.get("ContentLength")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail(f"object has invalid ContentLength: s3://{bucket}/{key}")
    etag = response.get("ETag")
    record = {
        "key": _relative_key(key, prefix),
        "size_bytes": size,
    }
    if mode == "version_id":
        version_id = response.get("VersionId")
        if not isinstance(version_id, str) or not version_id or version_id == "null":
            _fail(f"object lacks a non-null S3 VersionId: s3://{bucket}/{key}")
        record["version_id"] = version_id
        if isinstance(etag, str) and etag:
            record["etag"] = etag
    else:  # etag mode: the ETag is the immutability anchor (unversioned bucket, create-once)
        if not isinstance(etag, str) or not etag.strip():
            _fail(f"object lacks an ETag (needed in etag mode): s3://{bucket}/{key}")
        record["etag"] = etag
    checksum = response.get("ChecksumSHA256")
    if checksum:
        try:
            decoded = base64.b64decode(checksum, validate=True)
        except ValueError as error:
            raise ValueError(f"invalid base64 ChecksumSHA256 for s3://{bucket}/{key}") from error
        if len(decoded) != hashlib.sha256().digest_size:
            _fail(f"invalid ChecksumSHA256 length for s3://{bucket}/{key}")
        record["checksum_sha256"] = decoded.hex()
    return record


def snapshot_current_objects(
    client, *, root_s3: str, workers: int = 64, allow_unversioned: bool = False
) -> tuple[str, str, list[dict], str]:
    if workers <= 0:
        _fail("workers must be positive")
    bucket, prefix = _parse_s3_root(root_s3)
    status = client.get_bucket_versioning(Bucket=bucket).get("Status")
    if status == "Enabled":
        mode = "version_id"
    elif allow_unversioned:
        mode = "etag"  # unversioned bucket: pin ETag + size (study writes this prefix create-once)
    else:
        _fail(
            f"bucket versioning is {status!r} (not Enabled) for {bucket}; pass allow_unversioned "
            "to pin ETags instead of VersionIds"
        )
    keys = _list_current_keys(client, bucket=bucket, prefix=prefix)
    if not keys:
        _fail(f"inventory root has no current objects: {root_s3}")

    def head(key: str) -> dict:
        return _head_record(client, bucket=bucket, key=key, prefix=prefix, mode=mode)

    with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as pool:
        records = list(pool.map(head, keys))
    if keys != _list_current_keys(client, bucket=bucket, prefix=prefix):
        _fail("S3 key set changed while inventory was being built; retry from a stable prefix")
    return bucket, prefix, sorted(records, key=lambda item: item["key"]), mode


def _read_jsonl_episode_ids(client, *, bucket: str, prefix: str, record: dict) -> list[int]:
    key = f"{prefix}/{record['key']}" if prefix else record["key"]
    request = {"Bucket": bucket, "Key": key}
    if "version_id" in record:  # version mode pins the exact version; etag mode reads current + IfMatch
        request["VersionId"] = record["version_id"]
    else:
        request["IfMatch"] = record["etag"]
    response = client.get_object(**request)
    body = response["Body"]
    try:
        raw = body.read()
    finally:
        body.close()
    if len(raw) != record["size_bytes"]:
        _fail(f"pinned episodes metadata size mismatch: {record['key']}")
    values: list[int] = []
    for line_number, line in enumerate(io.BytesIO(raw), 1):
        if not line.strip():
            continue
        value = json.loads(line).get("episode_index")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _fail(f"{record['key']}:{line_number} has invalid episode_index={value!r}")
        values.append(value)
    if len(values) != len(set(values)):
        _fail(f"pinned episodes metadata has duplicate IDs: {record['key']}")
    return values


def _target_dataset_prefix(metadata_key: str) -> tuple[str, str]:
    parts = metadata_key.split("/")
    if len(parts) < 6 or parts[-3:] != ["lerobot", "meta", "episodes.jsonl"]:
        _fail(f"unexpected target metadata layout: {metadata_key}")
    if parts[0] not in {"atomic", "composite"}:
        _fail(f"unexpected target family in {metadata_key}")
    # Layout is {atomic|composite}/{task}/{capture}/lerobot/meta/episodes.jsonl.
    task = parts[-5]
    dataset_prefix = "/".join(parts[:-2])
    return task, dataset_prefix


def select_target_records(
    client,
    *,
    bucket: str,
    prefix: str,
    records: list[dict],
    demos_per_task: int = 150,
    expected_tasks: int = 50,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    metadata = [record for record in records if record["key"].endswith("/meta/episodes.jsonl")]
    if len(metadata) != expected_tasks:
        _fail(f"target inventory requires {expected_tasks} task metadata files; found {len(metadata)}")
    selected_records: list[dict] = []
    selection_tasks: list[dict] = []
    seen_tasks: set[str] = set()
    for metadata_record in sorted(metadata, key=lambda item: item["key"]):
        task, dataset_prefix = _target_dataset_prefix(metadata_record["key"])
        if task in seen_tasks:
            _fail(f"duplicate target task {task!r}")
        seen_tasks.add(task)
        episode_ids = _read_jsonl_episode_ids(client, bucket=bucket, prefix=prefix, record=metadata_record)
        if len(episode_ids) < demos_per_task:
            _fail(f"target task {task!r} has only {len(episode_ids)} episodes")
        shuffled = list(episode_ids)
        random.Random(seed).shuffle(shuffled)
        selected_ids = set(shuffled[:demos_per_task])
        selection_tasks.append({"task": task, "episode_indices": sorted(selected_ids)})

        task_records = [record for record in records if record["key"].startswith(f"{dataset_prefix}/")]
        found_payloads = {episode: set() for episode in selected_ids}
        for record in task_records:
            relative = record["key"][len(dataset_prefix) + 1 :]
            if relative.startswith(("data/", "videos/")):
                match = EPISODE_PAYLOAD.fullmatch(relative.rsplit("/", 1)[-1])
                if match is None:
                    _fail(f"unexpected episode payload name: {record['key']}")
                episode = int(match.group(1))
                if episode not in selected_ids:
                    continue
                found_payloads[episode].add(match.group(2))
            selected_records.append(record)
        incomplete = sorted(
            episode for episode, kinds in found_payloads.items() if not {"parquet", "mp4"}.issubset(kinds)
        )
        if incomplete:
            _fail(f"target task {task!r} has incomplete selected payloads: {incomplete[:10]}")

    selection = {
        "name": "seed0_t30",
        "episode_subsample_seed": seed,
        "demos_per_task": demos_per_task,
        "tasks": sorted(selection_tasks, key=lambda item: item["task"]),
    }
    return sorted(selected_records, key=lambda item: item["key"]), selection


def _select_init_records(records: list[dict]) -> list[dict]:
    selected = [record for record in records if record["key"].startswith(("params/", "assets/"))]
    if not any(record["key"].startswith("params/") for record in selected):
        _fail("initialization inventory has no params objects")
    if not any(record["key"].startswith("assets/") for record in selected):
        _fail("initialization inventory has no assets objects")
    return selected


def build_inventory(
    *,
    artifact: str,
    root_s3: str,
    study_root: str,
    output_dir: str | Path,
    workers: int = 64,
    allow_unversioned: bool = False,
    s3_client=None,
) -> tuple[Path, str, str, dict]:
    if artifact not in ARTIFACT_NAMESPACES:
        _fail(f"unsupported artifact {artifact!r}")
    if not study_root.startswith("s3://") or study_root.endswith("/"):
        _fail("study_root must be an s3:// URI without a trailing slash")
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
    bucket, prefix, records, mode = snapshot_current_objects(
        s3_client, root_s3=root_s3, workers=workers, allow_unversioned=allow_unversioned
    )
    if artifact == "pi05_h300_mg_init":
        objects = _select_init_records(records)
        selection = None
    else:
        objects, selection = select_target_records(s3_client, bucket=bucket, prefix=prefix, records=records)
    manifest = {
        "schema_version": 1,
        "artifact": artifact,
        "root_s3": root_s3,
        "content_addressing": mode,
        "selection": selection,
        "objects": objects,
    }
    data = _canonical_bytes(manifest)
    digest = hashlib.sha256(data).hexdigest()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{digest}.json"
    if destination.exists() and destination.read_bytes() != data:
        _fail(f"content-addressed inventory collision: {destination}")
    temporary = output / f".{digest}.json.incomplete"
    temporary.write_bytes(data)
    temporary.replace(destination)
    validate_inventory(destination, expected_artifact=artifact, expected_root_s3=root_s3)
    namespace = ARTIFACT_NAMESPACES[artifact]
    uri = f"{study_root}/manifests/inventories/{namespace}/{digest}.json"
    return destination, digest, uri, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=tuple(ARTIFACT_NAMESPACES), required=True)
    parser.add_argument("--root-s3", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help="pin ETag+size instead of S3 VersionId (for an unversioned bucket the study writes "
        "create-once); default requires bucket versioning Enabled",
    )
    parser.add_argument(
        "--confirm-read",
        action="store_true",
        help="required only after explicit user approval for this S3 inventory read",
    )
    args = parser.parse_args()
    if not args.confirm_read:
        raise SystemExit("S3 inventory read blocked: obtain explicit user approval, then pass --confirm-read")
    path, digest, uri, manifest = build_inventory(
        artifact=args.artifact,
        root_s3=args.root_s3,
        study_root=args.study_root,
        output_dir=args.output_dir,
        workers=args.workers,
        allow_unversioned=args.allow_unversioned,
    )
    print(f"path={path}")
    print(f"sha256={digest}")
    print(f"canonical_uri={uri}")
    print(f"objects={len(manifest['objects'])}")
    print("upload=false")


if __name__ == "__main__":
    main()
