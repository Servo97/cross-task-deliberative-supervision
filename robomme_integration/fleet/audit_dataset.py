#!/usr/bin/env python3
"""Audit staged RoboMME Parquet objects without downloading them from S3.

The pinned Hugging Face snapshot stores each LFS object behind a symlink whose target filename is
the expected SHA-256. S3 exposes a full-object CRC64NVME for every object uploaded by AWS CLI v2.
Reading each local shard once therefore proves both the local source content and its equality with
the S3 object while using only metadata requests against S3.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from .inventory import HF_REPO, HF_REVISION

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or uri.endswith("/"):
        raise ValueError(f"invalid S3 root {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _expected_sha256(path: Path) -> str:
    if not path.is_symlink():
        raise ValueError(f"pinned LFS path is not a symlink: {path}")
    digest = path.resolve(strict=True).name
    if not HEX64.fullmatch(digest):
        raise ValueError(f"LFS target does not encode SHA-256: {path} -> {digest}")
    return digest


def _local_digests(path: Path, crc64nvme) -> tuple[int, str, str]:
    sha256 = hashlib.sha256()
    crc64 = 0
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            size += len(block)
            sha256.update(block)
            crc64 = crc64nvme(block, crc64)
    encoded_crc = base64.b64encode(crc64.to_bytes(8, "big")).decode()
    return size, sha256.hexdigest(), encoded_crc


def audit(snapshot, root_s3: str, output, *, workers: int = 16, overlay=None) -> dict:
    import boto3
    from awscrt.checksums import crc64nvme
    from botocore.config import Config

    snapshot = Path(snapshot).resolve()
    overlay = Path(overlay).resolve() if overlay else None
    if snapshot.name != HF_REVISION:
        raise ValueError(f"snapshot is not pinned at {HF_REVISION}: {snapshot}")
    paths = sorted((snapshot / "data").rglob("*.parquet"))
    if len(paths) != 1600:
        raise ValueError(f"expected 1600 Parquet shards, found {len(paths)}")

    bucket, prefix = _parse_s3(root_s3)
    client = boto3.client(
        "s3",
        config=Config(max_pool_connections=workers, retries={"mode": "adaptive", "max_attempts": 10}),
    )

    def inspect(path: Path) -> dict:
        relative = path.relative_to(snapshot).as_posix()
        expected_sha256 = _expected_sha256(path)
        local_path = overlay / relative if overlay and (overlay / relative).is_file() else path
        size, local_sha256, local_crc64 = _local_digests(local_path, crc64nvme)
        response = client.head_object(
            Bucket=bucket,
            Key=f"{prefix}/{relative}",
            ChecksumMode="ENABLED",
        )
        s3_size = response["ContentLength"]
        s3_crc64 = response.get("ChecksumCRC64NVME")
        if response.get("ChecksumType") != "FULL_OBJECT" or not s3_crc64:
            raise ValueError(f"S3 object lacks a full-object CRC64NVME: {relative}")
        local_source_match = local_sha256 == expected_sha256
        s3_local_match = size == s3_size and local_crc64 == s3_crc64
        return {
            "key": relative,
            "size_bytes": size,
            "expected_source_sha256": expected_sha256,
            "local_sha256": local_sha256,
            "local_checksum_crc64nvme": local_crc64,
            "s3_size_bytes": s3_size,
            "s3_checksum_crc64nvme": s3_crc64,
            "s3_etag": response["ETag"],
            "overlay_used": local_path != path,
            "local_source_match": local_source_match,
            "s3_local_match": s3_local_match,
            "source_verified_on_s3": local_source_match and s3_local_match,
        }

    records = []
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        futures = {pool.submit(inspect, path): path for path in paths}
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if index % 100 == 0 or index == len(paths):
                print(f"[audit] checked {index}/{len(paths)}", flush=True)
    records.sort(key=lambda item: item["key"])

    summary = {
        "objects": len(records),
        "bytes": sum(record["size_bytes"] for record in records),
        "local_source_mismatches": sum(not record["local_source_match"] for record in records),
        "s3_local_mismatches": sum(not record["s3_local_match"] for record in records),
        "s3_source_mismatches": sum(not record["source_verified_on_s3"] for record in records),
    }
    report = {
        "schema_version": 1,
        "hf_repo_id": HF_REPO,
        "hf_revision": HF_REVISION,
        "snapshot": str(snapshot),
        "overlay": str(overlay) if overlay else None,
        "root_s3": root_s3,
        "method": "single_pass_local_sha256_crc64nvme_plus_s3_head",
        "summary": summary,
        "records": records,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".incomplete")
    temporary.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--root-s3", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--confirm-read", action="store_true")
    args = parser.parse_args()
    if not args.confirm_read:
        raise SystemExit("audit blocked: obtain explicit approval and pass --confirm-read")
    report = audit(
        args.snapshot,
        args.root_s3,
        args.output,
        workers=args.workers,
        overlay=args.overlay,
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
