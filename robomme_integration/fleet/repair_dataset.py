#!/usr/bin/env python3
"""Repair only source-mismatched RoboMME Parquet objects in the canonical S3 prefix."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .inventory import HF_REPO, HF_REVISION


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or uri.endswith("/"):
        raise ValueError(f"invalid S3 root {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _digests(path: Path, crc64nvme) -> tuple[int, str, str]:
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


def _decode_parquet(path: Path, expected_rows: int) -> int:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    rows = 0
    for index in range(source.metadata.num_row_groups):
        rows += source.read_row_group(index).num_rows
    if rows != expected_rows or source.metadata.num_rows != expected_rows:
        raise ValueError(f"row count mismatch for {path}: {rows} != {expected_rows}")
    return rows


def _download(record: dict, destination: Path, crc64nvme, expected_rows: int) -> dict:
    target = destination / record["key"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = record["expected_source_sha256"]
    expected_size = record["size_bytes"]

    if target.is_file():
        size, sha256, crc64 = _digests(target, crc64nvme)
        if size == expected_size and sha256 == expected_sha256:
            rows = _decode_parquet(target, expected_rows)
            return {
                "key": record["key"],
                "path": str(target),
                "size_bytes": size,
                "source_sha256": sha256,
                "checksum_crc64nvme": crc64,
                "rows": rows,
                "downloaded": False,
            }
        target.unlink()

    temporary = target.with_name(target.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{quote(record['key'], safe='/')}"
    request = Request(url, headers={"User-Agent": "robomme-integrity-repair/1"})
    sha256 = hashlib.sha256()
    crc64 = 0
    size = 0
    try:
        with urlopen(request, timeout=300) as response, temporary.open("xb") as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                size += len(block)
                sha256.update(block)
                crc64 = crc64nvme(block, crc64)
        actual_sha256 = sha256.hexdigest()
        if size != expected_size or actual_sha256 != expected_sha256:
            raise ValueError(
                f"source mismatch for {record['key']}: "
                f"size={size}/{expected_size} sha={actual_sha256}/{expected_sha256}"
            )
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    encoded_crc = base64.b64encode(crc64.to_bytes(8, "big")).decode()
    rows = _decode_parquet(target, expected_rows)
    return {
        "key": record["key"],
        "path": str(target),
        "size_bytes": size,
        "source_sha256": actual_sha256,
        "checksum_crc64nvme": encoded_crc,
        "rows": rows,
        "downloaded": True,
    }


def repair(
    audit_report,
    episodes_metadata,
    destination,
    output,
    *,
    workers: int = 8,
    expected_count: int,
) -> dict:
    import boto3
    from awscrt.checksums import crc64nvme
    from botocore.config import Config

    audit = json.loads(Path(audit_report).read_text(encoding="utf-8"))
    if (
        audit.get("hf_repo_id") != HF_REPO
        or audit.get("hf_revision") != HF_REVISION
        or audit.get("method") != "single_pass_local_sha256_crc64nvme_plus_s3_head"
    ):
        raise ValueError("audit report does not match the pinned RoboMME contract")
    mismatches = [record for record in audit["records"] if not record["source_verified_on_s3"]]
    if len(mismatches) != expected_count:
        raise ValueError(f"expected {expected_count} repairs, audit requires {len(mismatches)}")

    episode_rows = {}
    with Path(episodes_metadata).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                episode_rows[record["episode_index"]] = record["length"]
    if len(episode_rows) != 1600:
        raise ValueError(f"expected 1600 episode metadata records, found {len(episode_rows)}")

    destination = Path(destination).resolve()

    def download(record: dict) -> dict:
        episode_index = int(Path(record["key"]).stem.removeprefix("episode_"))
        return _download(
            record,
            destination,
            crc64nvme,
            episode_rows[episode_index],
        )

    sources = []
    with ThreadPoolExecutor(max_workers=min(workers, len(mismatches))) as pool:
        futures = {pool.submit(download, record): record for record in mismatches}
        for index, future in enumerate(as_completed(futures), start=1):
            sources.append(future.result())
            print(f"[repair] source-verified {index}/{len(mismatches)}", flush=True)
    sources.sort(key=lambda item: item["key"])

    bucket, prefix = _parse_s3(audit["root_s3"])
    client = boto3.client(
        "s3",
        config=Config(max_pool_connections=workers, retries={"mode": "adaptive", "max_attempts": 10}),
    )
    old_records = {record["key"]: record for record in mismatches}

    def upload(source: dict) -> dict:
        key = f"{prefix}/{source['key']}"
        before = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if (
            before["ContentLength"] == source["size_bytes"]
            and before.get("ChecksumCRC64NVME") == source["checksum_crc64nvme"]
        ):
            return {
                **source,
                "uploaded": False,
                "old_etag": before["ETag"],
                "new_etag": before["ETag"],
            }
        old = old_records[source["key"]]
        if (
            before["ContentLength"] != old["s3_size_bytes"]
            or before.get("ChecksumCRC64NVME") != old["s3_checksum_crc64nvme"]
        ):
            raise ValueError(f"concurrent or unknown S3 mutation at {source['key']}")
        with Path(source["path"]).open("rb") as stream:
            response = client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                ContentLength=source["size_bytes"],
                ContentType="binary/octet-stream",
                IfMatch=before["ETag"],
                ChecksumAlgorithm="CRC64NVME",
                ChecksumCRC64NVME=source["checksum_crc64nvme"],
                ServerSideEncryption="AES256",
                Metadata={"hf-lfs-sha256": source["source_sha256"]},
            )
        after = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if (
            after["ContentLength"] != source["size_bytes"]
            or after.get("ChecksumCRC64NVME") != source["checksum_crc64nvme"]
        ):
            raise ValueError(f"post-repair S3 checksum mismatch at {source['key']}")
        return {
            **source,
            "uploaded": True,
            "old_etag": before["ETag"],
            "new_etag": after["ETag"],
            "put_checksum_crc64nvme": response.get("ChecksumCRC64NVME"),
        }

    repairs = []
    with ThreadPoolExecutor(max_workers=min(workers, len(sources))) as pool:
        futures = {pool.submit(upload, source): source for source in sources}
        for index, future in enumerate(as_completed(futures), start=1):
            repairs.append(future.result())
            print(f"[repair] s3-verified {index}/{len(sources)}", flush=True)
    repairs.sort(key=lambda item: item["key"])

    report = {
        "schema_version": 1,
        "hf_repo_id": HF_REPO,
        "hf_revision": HF_REVISION,
        "root_s3": audit["root_s3"],
        "source_audit_sha256": hashlib.sha256(Path(audit_report).read_bytes()).hexdigest(),
        "summary": {
            "objects": len(repairs),
            "bytes": sum(item["size_bytes"] for item in repairs),
            "downloaded": sum(item["downloaded"] for item in repairs),
            "uploaded": sum(item["uploaded"] for item in repairs),
            "decoded_rows": sum(item["rows"] for item in repairs),
        },
        "repairs": repairs,
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
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--episodes-metadata", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--confirm-repair", action="store_true")
    args = parser.parse_args()
    if not args.confirm_repair:
        raise SystemExit("repair blocked: obtain explicit approval and pass --confirm-repair")
    report = repair(
        args.audit_report,
        args.episodes_metadata,
        args.destination,
        args.output,
        workers=args.workers,
        expected_count=args.expected_count,
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
