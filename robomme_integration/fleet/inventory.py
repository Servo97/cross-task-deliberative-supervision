#!/usr/bin/env python3
"""Validate and materialize immutable RoboMME/init S3 inventories."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

DATA_ARTIFACT = "robomme_lerobot_all16"
INIT_ARTIFACT = "pi05_h300_mg_init"
PI05_BASE_INIT_ARTIFACT = "pi05_base_init"
INIT_ARTIFACTS = frozenset({INIT_ARTIFACT, PI05_BASE_INIT_ARTIFACT})
ARTIFACTS = frozenset({DATA_ARTIFACT, *INIT_ARTIFACTS})
HF_REPO = "Yinpei/robomme_data_lerobot"
HF_REVISION = "1510653cccb4d9e5165fb3141c06d88053decc20"
NORM_STATS_SHA256 = "f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5"
BENCHMARK_TASKS = 16
TASK_INSTRUCTION_VARIANTS = 116
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL = {
    "schema_version",
    "artifact",
    "root_s3",
    "content_addressing",
    "selection",
    "objects",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _safe_key(value) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"unsafe inventory key {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value.endswith("/") or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"unsafe inventory key {value!r}")
    return value


def _validate_selection(artifact: str, value) -> None:
    if artifact in INIT_ARTIFACTS:
        if value is not None:
            _fail("initialization inventory selection must be null")
        return
    expected = {
        "name",
        "hf_repo_id",
        "hf_revision",
        "episodes",
        "benchmark_tasks",
        "task_instruction_variants",
        "training_scope",
        "norm_stats_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        _fail(f"RoboMME selection keys must be exactly {sorted(expected)}")
    required = {
        "name": "all16_v1",
        "hf_repo_id": HF_REPO,
        "hf_revision": HF_REVISION,
        "episodes": 1600,
        "benchmark_tasks": BENCHMARK_TASKS,
        "task_instruction_variants": TASK_INSTRUCTION_VARIANTS,
        "training_scope": "execution_frames_only",
        "norm_stats_sha256": NORM_STATS_SHA256,
    }
    if value != required:
        _fail(f"RoboMME selection differs from the registered dataset contract: {value}")


def validate_inventory(path, *, artifact: str, root_s3: str) -> dict[str, int]:
    if artifact not in ARTIFACTS:
        _fail(f"unsupported artifact {artifact!r}")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != TOP_LEVEL:
        _fail(f"inventory keys must be exactly {sorted(TOP_LEVEL)}")
    schema_version = manifest["schema_version"]
    if schema_version not in {1, 2} or manifest["artifact"] != artifact:
        _fail("inventory schema/artifact mismatch")
    if manifest["root_s3"] != root_s3:
        _fail("inventory root_s3 mismatch")
    mode = manifest["content_addressing"]
    if mode not in {"version_id", "etag"}:
        _fail("inventory content_addressing must be version_id or etag")
    _validate_selection(artifact, manifest["selection"])

    objects = manifest["objects"]
    if not isinstance(objects, list) or not objects:
        _fail("inventory must contain objects")
    keys: set[str] = set()
    total = 0
    for record in objects:
        required = {"key", "size_bytes", mode}
        allowed = required | {
            "checksum_sha256",
            "checksum_crc64nvme",
            "source_sha256",
            "etag",
        }
        if not isinstance(record, dict) or not required.issubset(record) or not set(record).issubset(allowed):
            _fail(f"invalid inventory record keys: {record}")
        key = _safe_key(record["key"])
        if key in keys:
            _fail(f"duplicate inventory key {key}")
        keys.add(key)
        size = record["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _fail(f"invalid size for {key}")
        total += size
        anchor = record[mode]
        if not isinstance(anchor, str) or not anchor or anchor == "null":
            _fail(f"invalid {mode} for {key}")
        if "checksum_sha256" in record and not HEX64.fullmatch(record["checksum_sha256"]):
            _fail(f"invalid SHA-256 for {key}")
        if "source_sha256" in record and not HEX64.fullmatch(record["source_sha256"]):
            _fail(f"invalid source SHA-256 for {key}")
        if "checksum_crc64nvme" in record:
            try:
                crc64 = base64.b64decode(record["checksum_crc64nvme"], validate=True)
            except (TypeError, ValueError) as error:
                _fail(f"invalid CRC64NVME for {key}: {error}")
            if len(crc64) != 8:
                _fail(f"invalid CRC64NVME length for {key}")

    if artifact in INIT_ARTIFACTS:
        if not any(key.startswith("params/") for key in keys):
            _fail("initialization inventory has no params")
        if not any(key.startswith("assets/") for key in keys):
            _fail("initialization inventory has no assets")
    else:
        required_keys = {
            "_robomme_source.json",
            "meta/info.json",
            "meta/episodes.jsonl",
            "meta/episodes_stats.jsonl",
            "meta/tasks.jsonl",
            "assets/robomme/norm_stats.json",
        }
        if not required_keys.issubset(keys):
            _fail(f"RoboMME inventory lacks {sorted(required_keys - keys)}")
        parquet = [key for key in keys if key.endswith(".parquet") and key.startswith("data/")]
        if len(parquet) != 1600:
            _fail(f"RoboMME inventory requires 1600 episode parquet files; found {len(parquet)}")
        if schema_version == 2:
            parquet_records = [
                record
                for record in objects
                if record["key"].endswith(".parquet") and record["key"].startswith("data/")
            ]
            if any("source_sha256" not in record or "checksum_crc64nvme" not in record for record in parquet_records):
                _fail("RoboMME v2 inventory requires source SHA-256 and CRC64NVME per Parquet")
    return {"objects": len(keys), "bytes": total}


def validate_dataset_root(root) -> dict[str, int]:
    root = Path(root)
    source = json.loads((root / "_robomme_source.json").read_text(encoding="utf-8"))
    if source != {
        "schema_version": 1,
        "hf_repo_id": HF_REPO,
        "hf_revision": HF_REVISION,
    }:
        _fail(f"RoboMME source marker mismatch: {source}")
    norm_path = root / "assets" / "robomme" / "norm_stats.json"
    if hashlib.sha256(norm_path.read_bytes()).hexdigest() != NORM_STATS_SHA256:
        _fail("RoboMME norm_stats checksum mismatch")
    episodes: list[int] = []
    with (root / "meta" / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                episodes.append(json.loads(line)["episode_index"])
    if len(episodes) != 1600 or len(set(episodes)) != 1600:
        _fail("RoboMME must contain 1600 unique episodes")
    with (root / "meta" / "tasks.jsonl").open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    task_indices = [task.get("task_index") for task in tasks]
    task_text = [task.get("task") for task in tasks]
    if (
        len(tasks) != TASK_INSTRUCTION_VARIANTS
        or sorted(task_indices) != list(range(TASK_INSTRUCTION_VARIANTS))
        or len(set(task_text)) != TASK_INSTRUCTION_VARIANTS
    ):
        _fail(f"RoboMME must contain 116 unique, contiguous instruction variants; found {len(tasks)} records")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if (
        info.get("total_episodes") != 1600
        or info.get("total_tasks") != TASK_INSTRUCTION_VARIANTS
        or info.get("total_chunks") != 2
    ):
        _fail("RoboMME info.json counts differ from the pinned dataset contract")
    parquet = list((root / "data").rglob("*.parquet"))
    if len(parquet) != 1600:
        _fail(f"RoboMME must materialize 1600 parquet files; found {len(parquet)}")
    return {
        "episodes": len(episodes),
        "benchmark_tasks": BENCHMARK_TASKS,
        "task_instruction_variants": len(tasks),
        "parquet": len(parquet),
    }


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        _fail(f"invalid S3 URI {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def materialize(
    manifest_path,
    *,
    artifact: str,
    root_s3: str,
    destination,
    workers: int = 64,
) -> dict[str, int]:
    summary = validate_inventory(manifest_path, artifact=artifact, root_s3=root_s3)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        _fail(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _parse_s3(root_s3)

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        config=Config(max_pool_connections=workers, retries={"mode": "adaptive", "max_attempts": 10}),
    )

    def fetch(record):
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
            _fail(f"ContentLength mismatch for {relative}")
        if "version_id" in record and response.get("VersionId") != record["version_id"]:
            _fail(f"VersionId mismatch for {relative}")
        if "etag" in record and response.get("ETag") != record["etag"]:
            _fail(f"ETag mismatch for {relative}")
        if "checksum_sha256" in record:
            expected = base64.b64encode(bytes.fromhex(record["checksum_sha256"])).decode()
            if response.get("ChecksumSHA256") != expected:
                _fail(f"checksum mismatch for {relative}")
        if "checksum_crc64nvme" in record and response.get("ChecksumCRC64NVME") != record["checksum_crc64nvme"]:
            _fail(f"CRC64NVME mismatch for {relative}")
        written = 0
        source_sha256 = hashlib.sha256() if "source_sha256" in record else None
        body = response["Body"]
        try:
            with temporary.open("xb") as stream:
                while block := body.read(8 * 1024 * 1024):
                    stream.write(block)
                    written += len(block)
                    if source_sha256 is not None:
                        source_sha256.update(block)
        finally:
            body.close()
        if written != record["size_bytes"]:
            temporary.unlink(missing_ok=True)
            _fail(f"download size mismatch for {relative}")
        if source_sha256 is not None and source_sha256.hexdigest() != record["source_sha256"]:
            temporary.unlink(missing_ok=True)
            _fail(f"source SHA-256 mismatch for {relative}")
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=min(workers, len(manifest["objects"]))) as pool:
        list(pool.map(fetch, manifest["objects"]))
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    expected = {record["key"] for record in manifest["objects"]}
    if actual != expected:
        _fail(
            f"materialized file set mismatch missing={sorted(expected - actual)[:10]} "
            f"extra={sorted(actual - expected)[:10]}"
        )
    if artifact == DATA_ARTIFACT:
        validate_dataset_root(destination)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True, choices=sorted(ARTIFACTS))
    parser.add_argument("--root-s3", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    result = materialize(
        args.manifest,
        artifact=args.artifact,
        root_s3=args.root_s3,
        destination=args.destination,
        workers=args.workers,
    )
    print(f"[robomme-inventory] verified objects={result['objects']} bytes={result['bytes']}")


if __name__ == "__main__":
    main()
