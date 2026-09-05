#!/usr/bin/env python3
"""Validate an exact local artifact tree against a content-addressed JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def validate_artifact_tree(
    root: str | Path,
    manifest_path: str | Path,
    *,
    manifest_sha256: str,
    artifact_uri: str,
    workers: int = 16,
    require_prefix: str | None = None,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    expected_manifest_sha = _sha(manifest_sha256, "manifest_sha256")
    root = Path(root)
    manifest_path = Path(manifest_path)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(f"tree manifest sha256={actual_manifest_sha}, expected={expected_manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "kind", "artifact_uri", "files"}:
        raise ValueError(f"tree manifest has unexpected keys {sorted(set(manifest))}")
    if manifest["schema_version"] != 1:
        raise ValueError(f"tree manifest schema_version={manifest['schema_version']!r}")
    if manifest["kind"] != "wsm_artifact_tree_manifest":
        raise ValueError(f"tree manifest kind={manifest['kind']!r}")
    if manifest["artifact_uri"].rstrip("/") != artifact_uri.rstrip("/"):
        raise ValueError(f"tree artifact_uri={manifest['artifact_uri']!r}, expected={artifact_uri!r}")
    records = manifest["files"]
    if not isinstance(records, list) or not records:
        raise ValueError("tree manifest files must be a non-empty list")

    expected: dict[str, dict] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError(f"files[{index}] must contain path,size,sha256 exactly")
        relative = PurePosixPath(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe tree path {record['path']!r}")
        name = relative.as_posix()
        if name in expected:
            raise ValueError(f"duplicate tree path {name}")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"tree file {name} has invalid size")
        _sha(record["sha256"], f"tree file {name} sha256")
        expected[name] = record
    if require_prefix and not any(name.startswith(require_prefix) for name in expected):
        raise ValueError(f"tree manifest contains no files under {require_prefix!r}")

    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains symlink {path}")
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"artifact file-set mismatch: missing={missing[:10]} extra={extra[:10]}")

    def verify(item: tuple[str, Path]) -> None:
        name, path = item
        descriptor = expected[name]
        size = path.stat().st_size
        if size != descriptor["size"]:
            raise ValueError(f"artifact {name} size={size}, expected={descriptor['size']}")
        digest = sha256_file(path)
        if digest != descriptor["sha256"]:
            raise ValueError(f"artifact {name} sha256={digest}, expected={descriptor['sha256']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify, sorted(actual.items())))
    return {"num_files": len(actual), "manifest_sha256": actual_manifest_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--require-prefix", default=None)
    args = parser.parse_args()
    result = validate_artifact_tree(
        args.root,
        args.manifest,
        manifest_sha256=args.manifest_sha256,
        artifact_uri=args.artifact_uri,
        workers=args.workers,
        require_prefix=args.require_prefix,
    )
    print(
        f"[artifact-tree] verified files={result['num_files']} manifest_sha256={result['manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
