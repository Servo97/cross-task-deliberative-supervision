#!/usr/bin/env python3
"""Derive and materialize one RoboMME task from the sealed all-16 inventory.

The S3 dataset remains consolidated once.  A single-task job downloads the shared metadata and
only its exact 100 episode Parquets, while every selected object retains the strong checksums and
immutable anchors from the canonical parent inventory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .inventory import DATA_ARTIFACT, validate_inventory

try:
    from ..training.single_task import TASK_EPISODES, task_manifest_sha256
except ImportError:  # SageMaker stages robomme_integration/ contents as top-level packages.
    from training.single_task import TASK_EPISODES, task_manifest_sha256


CANONICAL_PARENT_SHA256 = "e77968b4c72c7589d92c1e85b1c6f7bf81aa49dd74472fb88dcead4277b5dad2"
# Filled from the canonical parent inventory.  A parent-object mutation or task-range change must
# deliberately update these values and therefore the submitted source identity.
CANONICAL_TASK_DERIVED_SHA256: dict[str, str] = {
    "PatternLock": "06da20e4d0e11cf895a90235beeb7034b4ed83fa8dc2878b4888cd0bafa29cbc",
    "ButtonUnmaskSwap": "aea557d5423dbd0fe0c13ca8328391d6f885b2d2e2f0ed5bb8ab66e4319e7f32",
    "ButtonUnmask": "0df8ce002d5019d354415218c2139eb98ed15d50e13beca206898502750a1583",
    "VideoPlaceButton": "ebfb1facb3d08eb534c27dc7f5dec7fca898740b3c5c34e43e4171d670289b1a",
    "VideoUnmaskSwap": "ee88b73d34cb9d3bdb779800ab7f94d6b488a871e8a7787f4e8d13a7f5016ef9",
    "PickXtimes": "b4849edc217869715b03529b3a2cf904dbf68465bab84781b4cc82ad26a39837",
    "StopCube": "4dfd62cf676d70277aed3bfc8bb924c73a6a3214b3204902b7b285ab35836be3",
    "SwingXtimes": "cf5f14d979d58f07f163b0dfba07d1dbd1ce1e5f338b70977f7d85b748241f34",
    "PickHighlight": "2f959ff1f630493f66ffa3330db827c47c9c61e4329c79461db0c981aeb26281",
    "MoveCube": "0fbfc217fd24aff9e2f1020d092ce57d4ac20980d9f11335066c6f035066d509",
    "InsertPeg": "0c48b549b51c57161cfa1994a46633b353ec4797066dc9e63aeb8f2dcd78b1fc",
    "RouteStick": "e8b6513f6ce4df43aeae2d54b706f364e2a242ba78e48b475e38e81a150e5812",
    "BinFill": "95d379b7ccfbf572d6ae24bc9c8ec1b3cecf5fe089975958d1484aff2e6a72eb",
    "VideoPlaceOrder": "db9ced6ec48dd4aa6921772b29f2ea9b386c355174508f6af171cc774a0ca093",
    "VideoRepick": "e51adedbea97bab10f12956190a22d5318c8d7794be24b6550860e00165c77b9",
    "VideoUnmask": "f1fc53e2854c4a922209fca4ccdd88c6f2b2416541829998e2864b3d17776659",
}


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_key(episode: int) -> str:
    return f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def derive(parent_path: str | Path, task: str, *, root_s3: str) -> tuple[dict, str]:
    """Return a deterministic task view and its SHA without writing another dataset copy."""
    if task not in TASK_EPISODES:
        raise ValueError(f"unknown RoboMME task {task!r}")
    parent_path = Path(parent_path)
    actual_parent_sha = _sha256_file(parent_path)
    if actual_parent_sha != CANONICAL_PARENT_SHA256:
        raise ValueError(f"single-task derivation requires parent {CANONICAL_PARENT_SHA256}; got {actual_parent_sha}")
    validate_inventory(parent_path, artifact=DATA_ARTIFACT, root_s3=root_s3)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    selected_keys = {_episode_key(episode) for episode in TASK_EPISODES[task]}
    shared = [record for record in parent["objects"] if not record["key"].startswith("data/")]
    episode_records = [record for record in parent["objects"] if record["key"] in selected_keys]
    found = {record["key"] for record in episode_records}
    if found != selected_keys:
        raise ValueError(f"parent inventory lacks task Parquets: {sorted(selected_keys - found)[:5]}")
    if any("source_sha256" not in record or "checksum_crc64nvme" not in record for record in episode_records):
        raise ValueError("single-task Parquets must retain source SHA-256 and S3 CRC64NVME")
    derived = {
        "schema_version": 1,
        "artifact": "robomme_lerobot_single_task_view",
        "root_s3": root_s3,
        "parent_inventory_sha256": CANONICAL_PARENT_SHA256,
        "task_name": task,
        "task_manifest_sha256": task_manifest_sha256(task),
        "episodes": list(TASK_EPISODES[task]),
        "objects": sorted([*shared, *episode_records], key=lambda record: record["key"]),
    }
    data = _canonical_bytes(derived)
    return derived, hashlib.sha256(data).hexdigest()


def validate_derived(manifest: dict, expected_sha256: str) -> dict[str, int]:
    if hashlib.sha256(_canonical_bytes(manifest)).hexdigest() != expected_sha256:
        raise ValueError("derived single-task inventory SHA mismatch")
    task = manifest.get("task_name")
    if task not in TASK_EPISODES:
        raise ValueError("derived inventory has an unknown task")
    if manifest.get("parent_inventory_sha256") != CANONICAL_PARENT_SHA256:
        raise ValueError("derived inventory has the wrong parent")
    if manifest.get("task_manifest_sha256") != task_manifest_sha256(task):
        raise ValueError("derived inventory task manifest drifted")
    expected_episodes = list(TASK_EPISODES[task])
    if manifest.get("episodes") != expected_episodes:
        raise ValueError("derived inventory episode list drifted")
    records = manifest.get("objects")
    if not isinstance(records, list):
        raise ValueError("derived inventory objects must be a list")
    keys = [record.get("key") for record in records if isinstance(record, dict)]
    if len(keys) != len(records) or len(keys) != len(set(keys)):
        raise ValueError("derived inventory keys must be unique strings")
    data_keys = {key for key in keys if isinstance(key, str) and key.startswith("data/")}
    expected_keys = {_episode_key(episode) for episode in expected_episodes}
    if data_keys != expected_keys:
        raise ValueError("derived inventory does not contain exactly the task Parquets")
    return {
        "objects": len(records),
        "bytes": sum(int(record["size_bytes"]) for record in records),
        "episodes": len(expected_episodes),
    }


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def materialize(
    parent_path: str | Path,
    task: str,
    *,
    root_s3: str,
    destination: str | Path,
    expected_derived_sha256: str,
    workers: int = 32,
) -> dict[str, int]:
    """Download a verified task view directly from the consolidated S3 dataset."""
    derived, digest = derive(parent_path, task, root_s3=root_s3)
    if digest != expected_derived_sha256:
        raise ValueError(f"derived inventory mismatch: {digest} != {expected_derived_sha256}")
    canonical = CANONICAL_TASK_DERIVED_SHA256.get(task)
    if canonical != digest:
        raise ValueError(f"unregistered canonical task inventory for {task}: {digest} != {canonical}")
    summary = validate_derived(derived, digest)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _parse_s3(root_s3)

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        config=Config(max_pool_connections=workers, retries={"mode": "adaptive", "max_attempts": 10}),
    )

    def fetch(record: dict) -> None:
        relative = record["key"]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".incomplete")
        key = f"{prefix}/{relative}" if prefix else relative
        request = {"Bucket": bucket, "Key": key}
        if "version_id" in record:
            request["VersionId"] = record["version_id"]
        if "etag" in record:
            request["IfMatch"] = record["etag"]
        if "checksum_sha256" in record or "checksum_crc64nvme" in record:
            request["ChecksumMode"] = "ENABLED"
        response = client.get_object(**request)
        if response.get("ContentLength") != record["size_bytes"]:
            raise ValueError(f"ContentLength mismatch for {relative}")
        if "version_id" in record and response.get("VersionId") != record["version_id"]:
            raise ValueError(f"VersionId mismatch for {relative}")
        if "etag" in record and response.get("ETag") != record["etag"]:
            raise ValueError(f"ETag mismatch for {relative}")
        if "checksum_sha256" in record:
            expected = base64.b64encode(bytes.fromhex(record["checksum_sha256"])).decode()
            if response.get("ChecksumSHA256") != expected:
                raise ValueError(f"S3 SHA-256 mismatch for {relative}")
        if "checksum_crc64nvme" in record and response.get("ChecksumCRC64NVME") != record["checksum_crc64nvme"]:
            raise ValueError(f"S3 CRC64NVME mismatch for {relative}")
        source_digest = hashlib.sha256() if "source_sha256" in record else None
        written = 0
        body = response["Body"]
        try:
            with temporary.open("xb") as stream:
                while block := body.read(8 * 1024 * 1024):
                    stream.write(block)
                    written += len(block)
                    if source_digest is not None:
                        source_digest.update(block)
        finally:
            body.close()
        if written != record["size_bytes"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"download size mismatch for {relative}")
        if source_digest is not None and source_digest.hexdigest() != record["source_sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"source SHA-256 mismatch for {relative}")
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=min(workers, len(derived["objects"]))) as pool:
        list(pool.map(fetch, derived["objects"]))
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    expected = {record["key"] for record in derived["objects"]}
    if actual != expected:
        raise ValueError(
            f"materialized task file set mismatch missing={sorted(expected - actual)[:5]} "
            f"extra={sorted(actual - expected)[:5]}"
        )

    # Verify the exact metadata bytes and task mapping after materialization.
    try:
        from ..training.single_task import select_task_episodes
    except ImportError:
        from training.single_task import select_task_episodes

    if tuple(select_task_episodes(destination, task)) != TASK_EPISODES[task]:
        raise ValueError("materialized single-task metadata selected the wrong episodes")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASK_EPISODES))
    parser.add_argument("--root-s3", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--expected-derived-sha256")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--derive-only", action="store_true")
    args = parser.parse_args()
    derived, digest = derive(args.parent_manifest, args.task, root_s3=args.root_s3)
    summary = validate_derived(derived, digest)
    if args.derive_only:
        print(json.dumps({"sha256": digest, **summary}, sort_keys=True))
        return
    if not args.destination or not args.expected_derived_sha256:
        raise SystemExit("materialization requires --destination and --expected-derived-sha256")
    summary = materialize(
        args.parent_manifest,
        args.task,
        root_s3=args.root_s3,
        destination=args.destination,
        expected_derived_sha256=args.expected_derived_sha256,
        workers=args.workers,
    )
    print(json.dumps({"sha256": digest, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
