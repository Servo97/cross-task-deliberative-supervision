#!/usr/bin/env python3
"""Create-once upload of the ReMemBench LeRobot tree, plus its content-addressed inventory.

Every object is written with ``put_object(IfNoneMatch="*", ChecksumSHA256=...)``, so a re-run
after a partial upload is safe: an existing key returns 412 and is verified byte-for-byte
(size + sha256) against the local file instead of being overwritten.

The inventory mirrors ``scripts/launch/build_stage_s_inventory.py``'s schema -- ``schema_version``,
``artifact``, ``root_s3``, ``content_addressing``, ``selection``, ``objects`` with
``{key, size_bytes, etag, checksum_sha256}`` records sorted by key, canonical JSON + trailing
newline, filename = sha256 of those bytes. It differs in exactly one respect: ``artifact`` is
``remembench_train13`` and the ``selection`` block describes the held-out-tail complement rather
than a seed-0 150-of-500 subsample, because ReMemBench has 13 tasks with 9-44 train demos each
and the RoboCasa selection contract is hard-coded to 50 tasks x 150 demos.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

ARTIFACT = "remembench_train13"


def canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.digest()


def parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or uri.endswith("/"):
        raise SystemExit("s3 root must be an s3:// URI without a trailing slash")
    return parsed.netloc, parsed.path.lstrip("/")


def put_once(client, *, bucket: str, key: str, path: Path) -> dict:
    """Write one object create-once; on collision verify the existing bytes match."""
    from botocore.exceptions import ClientError

    digest = sha256_file(path)
    size = path.stat().st_size
    try:
        with path.open("rb") as stream:
            response = client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                IfNoneMatch="*",
                ChecksumSHA256=base64.b64encode(digest).decode("ascii"),
            )
        etag = response["ETag"]
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code not in {"PreconditionFailed", "ConditionalRequestConflict"}:
            raise
        head = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if head["ContentLength"] != size:
            raise SystemExit(f"create-once conflict with different size: s3://{bucket}/{key}")
        existing = head.get("ChecksumSHA256")
        if existing and base64.b64decode(existing) != digest:
            raise SystemExit(f"create-once conflict with different sha256: s3://{bucket}/{key}")
        etag = head["ETag"]
    return {
        "key": key,
        "size_bytes": size,
        "etag": etag,
        "checksum_sha256": digest.hex(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, help="dir whose tree becomes <root>/train/...")
    parser.add_argument("--root-s3", required=True, help="e.g. s3://.../datasets/remembench_v02")
    parser.add_argument("--prefix", default="train")
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--confirm-upload", action="store_true")
    args = parser.parse_args()
    if not args.confirm_upload:
        raise SystemExit("S3 write blocked: pass --confirm-upload after explicit approval")

    import boto3
    from botocore.config import Config

    bucket, root_prefix = parse_s3(args.root_s3)
    client = boto3.client(
        "s3",
        config=Config(
            max_pool_connections=args.workers * 2,
            retries={"mode": "adaptive", "max_attempts": 10},
        ),
    )

    local_root = Path(args.local_root).resolve()
    files = sorted(p for p in local_root.rglob("*") if p.is_file() and not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no files under {local_root}")

    def upload(path: Path) -> dict:
        relative = f"{args.prefix}/{path.relative_to(local_root).as_posix()}"
        key = f"{root_prefix}/{relative}" if root_prefix else relative
        record = put_once(client, bucket=bucket, key=key, path=path)
        record["key"] = relative  # inventory keys are relative to root_s3
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(upload, files))
    records.sort(key=lambda record: record["key"])
    if len({record["key"] for record in records}) != len(records):
        raise SystemExit("duplicate inventory keys")

    worklist = json.loads(Path(args.worklist).read_text())
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "root_s3": args.root_s3,
        "content_addressing": "etag",
        "selection": {
            "name": "remembench_train_tail02",
            "kind": "remembench_tail_fraction_complement",
            "fraction": worklist["selection"]["fraction"],
            "minimum": worklist["selection"]["minimum"],
            "tasks": sorted(
                (
                    {
                        "task": task["task"],
                        "episode_indices": [episode["episode_index"] for episode in task["episodes"]],
                    }
                    for task in worklist["tasks"]
                ),
                key=lambda item: item["task"],
            ),
        },
        "objects": records,
    }
    data = canonical_bytes(manifest)
    digest = hashlib.sha256(data).hexdigest()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{digest}.json"
    if destination.exists() and destination.read_bytes() != data:
        raise SystemExit(f"content-addressed inventory collision: {destination}")
    destination.write_bytes(data)
    print(f"objects={len(records)}")
    print(f"bytes={sum(record['size_bytes'] for record in records)}")
    print(f"inventory_path={destination}")
    print(f"inventory_sha256={digest}")


if __name__ == "__main__":
    main()
