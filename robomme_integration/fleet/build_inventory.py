#!/usr/bin/env python3
"""Build a content-addressed S3 inventory for RoboMME data or a registered init artifact."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .inventory import (
    BENCHMARK_TASKS,
    DATA_ARTIFACT,
    HF_REPO,
    HF_REVISION,
    INIT_ARTIFACTS,
    NORM_STATS_SHA256,
    PI05_BASE_INIT_ARTIFACT,
    TASK_INSTRUCTION_VARIANTS,
    validate_inventory,
)


def _parse(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or uri.endswith("/"):
        raise ValueError("root must be an S3 URI without trailing slash")
    return parsed.netloc, parsed.path.lstrip("/")


def _source_hashes(path, *, root_s3: str) -> dict[str, str]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        report.get("hf_repo_id") != HF_REPO
        or report.get("hf_revision") != HF_REVISION
        or report.get("root_s3") != root_s3
        or report.get("summary", {}).get("s3_source_mismatches") != 0
    ):
        raise ValueError("source audit is not a successful pinned RoboMME audit")
    records = report.get("records", [])
    hashes = {
        record["key"]: record["expected_source_sha256"] for record in records if record.get("source_verified_on_s3")
    }
    if len(records) != 1600 or len(hashes) != 1600:
        raise ValueError("source audit must verify exactly 1600 Parquet objects")
    return hashes


def build(
    artifact,
    root_s3,
    study_root,
    output_dir,
    *,
    workers=64,
    allow_unversioned=False,
    source_audit=None,
):
    import boto3
    from botocore.config import Config

    bucket, prefix = _parse(root_s3)
    client = boto3.client(
        "s3",
        config=Config(max_pool_connections=workers, retries={"mode": "adaptive", "max_attempts": 10}),
    )
    status = client.get_bucket_versioning(Bucket=bucket).get("Status")
    if status == "Enabled":
        mode = "version_id"
    elif allow_unversioned:
        mode = "etag"
    else:
        raise ValueError(f"bucket versioning is not enabled: {status!r}")

    request = {"Bucket": bucket, "Prefix": prefix + "/"}
    keys = []
    while True:
        response = client.list_objects_v2(**request)
        keys.extend(item["Key"] for item in response.get("Contents", []))
        if not response.get("IsTruncated"):
            break
        request["ContinuationToken"] = response["NextContinuationToken"]
    keys = sorted(keys)
    if not keys:
        raise ValueError(f"no objects under {root_s3}")

    def head(key):
        response = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        record = {
            "key": key[len(prefix) + 1 :],
            "size_bytes": response["ContentLength"],
        }
        if mode == "version_id":
            version = response.get("VersionId")
            if not version or version == "null":
                raise ValueError(f"missing VersionId for {key}")
            record["version_id"] = version
            if response.get("ETag"):
                record["etag"] = response["ETag"]
        else:
            record["etag"] = response["ETag"]
        if response.get("ChecksumSHA256"):
            raw = base64.b64decode(response["ChecksumSHA256"], validate=True)
            record["checksum_sha256"] = raw.hex()
        if response.get("ChecksumCRC64NVME"):
            raw = base64.b64decode(response["ChecksumCRC64NVME"], validate=True)
            if len(raw) != 8:
                raise ValueError(f"invalid S3 CRC64NVME for {key}")
            record["checksum_crc64nvme"] = response["ChecksumCRC64NVME"]
        return record

    with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as pool:
        objects = sorted(pool.map(head, keys), key=lambda item: item["key"])
    if artifact == DATA_ARTIFACT:
        if source_audit is None:
            schema_version = 1
        else:
            schema_version = 2
            hashes = _source_hashes(source_audit, root_s3=root_s3)
            for record in objects:
                if record["key"] in hashes:
                    record["source_sha256"] = hashes[record["key"]]
            if any(
                "checksum_crc64nvme" not in record or "source_sha256" not in record
                for record in objects
                if record["key"].startswith("data/") and record["key"].endswith(".parquet")
            ):
                raise ValueError("S3/source audit lacks strong checksums for a Parquet object")
        selection = {
            "name": "all16_v1",
            "hf_repo_id": HF_REPO,
            "hf_revision": HF_REVISION,
            "episodes": 1600,
            "benchmark_tasks": BENCHMARK_TASKS,
            "task_instruction_variants": TASK_INSTRUCTION_VARIANTS,
            "training_scope": "execution_frames_only",
            "norm_stats_sha256": NORM_STATS_SHA256,
        }
        namespace = "data"
    elif artifact in INIT_ARTIFACTS:
        if source_audit is not None:
            raise ValueError("source audit is only valid for the RoboMME data artifact")
        schema_version = 1
        selection = None
        namespace = "init/pi05_base" if artifact == PI05_BASE_INIT_ARTIFACT else "init"
    else:
        raise ValueError(f"unknown artifact {artifact}")
    manifest = {
        "schema_version": schema_version,
        "artifact": artifact,
        "root_s3": root_s3,
        "content_addressing": mode,
        "selection": selection,
        "objects": objects,
    }
    data = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    destination = Path(output_dir).resolve() / f"{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".incomplete")
    temporary.write_bytes(data)
    temporary.replace(destination)
    validate_inventory(destination, artifact=artifact, root_s3=root_s3)
    uri = f"{study_root}/manifests/inventories/{namespace}/{digest}.json"
    return destination, digest, uri, len(objects), sum(item["size_bytes"] for item in objects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, choices=(DATA_ARTIFACT, *sorted(INIT_ARTIFACTS)))
    parser.add_argument("--root-s3", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--allow-unversioned", action="store_true")
    parser.add_argument("--source-audit")
    parser.add_argument("--confirm-read", action="store_true")
    args = parser.parse_args()
    if not args.confirm_read:
        raise SystemExit("S3 read blocked: obtain explicit user approval, then pass --confirm-read")
    path, digest, uri, objects, size = build(
        args.artifact,
        args.root_s3,
        args.study_root,
        args.output_dir,
        workers=args.workers,
        allow_unversioned=args.allow_unversioned,
        source_audit=args.source_audit,
    )
    print(f"path={path}\nsha256={digest}\ncanonical_uri={uri}\nobjects={objects}\nbytes={size}\nupload=false")


if __name__ == "__main__":
    main()
