#!/usr/bin/env python3
"""Build a deterministic content manifest for a deploy-only checkpoint tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def deploy_receipt_identity(value: dict) -> dict:
    """Return immutable scientific identity, excluding attempt provenance.

    A preempted retry may finish sealing a checkpoint whose marker was created by an earlier
    attempt.  ``attempt_id`` records that creator, while ``run_manifest_sha256`` seals that
    attempt's infrastructure manifest; both necessarily change when ``attempt_index`` changes.
    They remain required provenance, but are not part of the checkpoint's scientific identity.
    Every other parsed JSON field must match exactly.
    """
    if not isinstance(value, dict):
        raise ValueError("deploy receipt must be a JSON object")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("deploy receipt requires nonempty run_id scientific identity")
    attempt_id = value.get("attempt_id")
    attempt_prefix = f"{run_id}-attempt"
    attempt_index = (
        attempt_id[len(attempt_prefix) :]
        if isinstance(attempt_id, str) and attempt_id.startswith(attempt_prefix)
        else ""
    )
    if not attempt_index.isdigit() or int(attempt_index) < 1:
        raise ValueError("deploy receipt requires a run-scoped positive attempt_id provenance")
    run_manifest_sha256 = value.get("run_manifest_sha256")
    if (
        not isinstance(run_manifest_sha256, str)
        or len(run_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in run_manifest_sha256)
    ):
        raise ValueError("deploy receipt requires a lowercase SHA-256 run-manifest provenance")
    identity = dict(value)
    identity.pop("attempt_id")
    identity.pop("run_manifest_sha256")
    return identity


def deploy_receipts_equivalent(left: dict, right: dict) -> bool:
    """Whether two deploy markers name the same immutable scientific checkpoint."""
    return deploy_receipt_identity(left) == deploy_receipt_identity(right)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root, uri: str, output, *, workers: int = 32, require_finalized: bool = True) -> str:
    root = Path(root).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    for required in ("params", "assets"):
        if not (root / required).is_dir():
            raise ValueError(f"checkpoint is missing required {required}/ tree")
    if require_finalized and not (root / "_CHECKPOINT_METADATA").is_file():
        raise ValueError("checkpoint lacks Orbax _CHECKPOINT_METADATA")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint tree contains symlink: {path}")
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.relative_to(root).parts[0] in {"params", "assets"}
    )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in paths
        if "orbax-checkpoint-tmp" in path.as_posix() or path.name.endswith(".tmp")
    ]
    if forbidden:
        raise ValueError(f"checkpoint contains temporary deploy files: {forbidden[:10]}")

    def describe(path: Path) -> dict:
        return {
            "key": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(paths) or 1)) as pool:
        records = list(pool.map(describe, paths))
    if not records or not any(record["key"].startswith("params/") for record in records):
        raise ValueError("checkpoint manifest contains no params")
    manifest = {"schema_version": 1, "checkpoint_uri": uri, "objects": records}
    data = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    output = Path(output)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_bytes(data)
    temporary.replace(output)
    return digest


def verify(root, manifest_path, *, expected_uri: str | None = None) -> str:
    """Rebuild and compare a deployment tree manifest without trusting S3 listing order."""
    manifest_path = Path(manifest_path)
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if expected_uri is not None and manifest.get("checkpoint_uri") != expected_uri:
        raise ValueError("checkpoint manifest URI mismatch")
    temporary = manifest_path.with_name(".rebuilt-checkpoint-manifest.json")
    try:
        digest = build(
            root,
            manifest["checkpoint_uri"],
            temporary,
            require_finalized=False,
        )
        if temporary.read_bytes() != payload:
            raise ValueError("local checkpoint tree differs from its sealed manifest")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="build a params/assets manifest after optimizer state and root Orbax metadata were pruned",
    )
    parser.add_argument("--expected-uri")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.verify:
        print(verify(args.root, args.output, expected_uri=args.expected_uri or args.uri))
    else:
        print(
            build(
                args.root,
                args.uri,
                args.output,
                workers=args.workers,
                require_finalized=not args.deploy_only,
            )
        )


if __name__ == "__main__":
    main()
