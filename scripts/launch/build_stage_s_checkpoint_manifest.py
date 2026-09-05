#!/usr/bin/env python3
"""Build the immutable deploy-tree manifest for one completed Stage-S OpenPI checkpoint.

Stage-S does not resume.  The published policy artifact therefore contains only `params/` and
`assets/`; optimizer `train_state/` remains node-local and is neither hashed nor uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    checkpoint_root: str | Path,
    *,
    checkpoint_uri: str,
    workers: int = 32,
) -> tuple[dict, bytes, str]:
    root = Path(checkpoint_root).resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not root.is_dir():
        raise ValueError(f"checkpoint directory is missing: {root}")
    for required in ("params", "assets"):
        if not (root / required).is_dir():
            raise ValueError(f"checkpoint is missing required {required}/ tree")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint tree contains symlink: {path}")
    deploy_roots = {"params", "assets"}
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.relative_to(root).parts[0] in deploy_roots
    )
    if not paths:
        raise ValueError("checkpoint tree contains no files")
    relative = [path.relative_to(root).as_posix() for path in paths]
    forbidden = [name for name in relative if "orbax-checkpoint-tmp" in name or name.endswith(".tmp")]
    if forbidden:
        raise ValueError(f"checkpoint tree contains temporary files: {forbidden[:10]}")
    commit_markers = (
        "commit_success.txt",
        "_CHECKPOINT_METADATA",
        "manifest.ocdbt",
        "_METADATA",
    )
    if not any(name.endswith(commit_markers) for name in relative):
        raise ValueError("checkpoint tree has no recognized Orbax completion metadata")

    def describe(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        files = list(pool.map(describe, paths))
    files.sort(key=lambda record: record["path"])
    manifest = {
        "schema_version": 1,
        "kind": "wsm_artifact_tree_manifest",
        "artifact_uri": checkpoint_uri.rstrip("/"),
        "files": files,
    }
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    return manifest, payload, hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    _manifest, payload, digest = build_manifest(
        args.checkpoint_root,
        checkpoint_uri=args.checkpoint_uri,
        workers=args.workers,
    )
    output = Path(args.output)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_bytes(payload)
    temporary.replace(output)
    print(digest)


if __name__ == "__main__":
    main()
