#!/usr/bin/env python3
"""Content-addressed, read-only node cache for serial RoboMME campaign cells."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath

from . import inventory, task_inventory

MARKER_SUFFIX = ".robomme-shared-complete.json"


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_cache_root(root: Path) -> Path:
    root = root.resolve()
    if not root.is_absolute() or root == Path("/"):
        raise ValueError("shared artifact root must be a non-root absolute path")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _target(root: Path, category: str, digest: str) -> Path:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("shared artifact key must be a lowercase SHA-256")
    target = root / category / digest
    if not target.resolve().is_relative_to(root):
        raise ValueError("shared artifact target escapes cache root")
    return target


def _marker(target: Path) -> Path:
    return target.with_name(target.name + MARKER_SUFFIX)


def _receipt(root: Path, *, identity: dict) -> dict:
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"shared artifact contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "key": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not files:
        raise ValueError(f"shared artifact is empty: {root}")
    return {"schema_version": 1, "identity": identity, "files": files}


def _verify(root: Path, marker: Path, *, identity: dict) -> None:
    if not root.is_dir() or not marker.is_file():
        raise ValueError(f"shared artifact is incomplete: {root}")
    expected = json.loads(marker.read_text(encoding="utf-8"))
    if expected.get("identity") != identity:
        raise ValueError(f"shared artifact identity drifted: {root}")
    actual = _receipt(root, identity=identity)
    if actual != expected:
        raise ValueError(f"shared artifact bytes drifted: {root}")


def _install(directory: Path, target: Path, *, identity: dict) -> Path:
    marker = _marker(target)
    if target.exists() or marker.exists():
        raise ValueError(f"shared artifact destination raced or is partial: {target}")
    receipt = _receipt(directory, identity=identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    directory.replace(target)
    temporary = marker.with_name(marker.name + ".incomplete")
    temporary.write_bytes(_canonical(receipt))
    os.replace(temporary, marker)
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    _verify(target, marker, identity=identity)
    return target


def _prepare_directory(
    cache_root: Path,
    category: str,
    digest: str,
    *,
    identity: dict,
    materialize,
) -> Path:
    cache_root = _safe_cache_root(cache_root)
    target = _target(cache_root, category, digest)
    marker = _marker(target)
    if target.is_dir() and marker.is_file():
        _verify(target, marker, identity=identity)
        return target
    if target.exists() or marker.exists():
        raise ValueError(f"partial shared artifact requires manual diagnosis: {target}")
    temporary = target.parent / f".{target.name}.incomplete-{uuid.uuid4().hex}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        materialize(temporary)
        return _install(temporary, target, identity=identity)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_inventory(
    manifest_path: Path,
    *,
    artifact: str,
    root_s3: str,
    manifest_sha256: str,
    cache_root: Path,
    workers: int,
) -> Path:
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("shared inventory manifest SHA mismatch")
    inventory.validate_inventory(manifest_path, artifact=artifact, root_s3=root_s3)
    identity = {
        "kind": "inventory",
        "artifact": artifact,
        "root_s3": root_s3,
        "manifest_sha256": manifest_sha256,
    }
    return _prepare_directory(
        cache_root,
        "inventory",
        manifest_sha256,
        identity=identity,
        materialize=lambda destination: inventory.materialize(
            manifest_path,
            artifact=artifact,
            root_s3=root_s3,
            destination=destination,
            workers=workers,
        ),
    )


def prepare_task(
    parent_manifest: Path,
    *,
    task: str,
    root_s3: str,
    derived_sha256: str,
    cache_root: Path,
    workers: int,
) -> Path:
    derived, actual = task_inventory.derive(parent_manifest, task, root_s3=root_s3)
    if actual != derived_sha256:
        raise ValueError(f"shared task inventory mismatch: {actual} != {derived_sha256}")
    task_inventory.validate_derived(derived, derived_sha256)
    identity = {
        "kind": "single_task_inventory",
        "task": task,
        "root_s3": root_s3,
        "parent_manifest_sha256": task_inventory.CANONICAL_PARENT_SHA256,
        "derived_sha256": derived_sha256,
    }
    return _prepare_directory(
        cache_root,
        "task",
        derived_sha256,
        identity=identity,
        materialize=lambda destination: task_inventory.materialize(
            parent_manifest,
            task,
            root_s3=root_s3,
            destination=destination,
            expected_derived_sha256=derived_sha256,
            workers=workers,
        ),
    )


def prepare_blob(uri: str, sha256: str, cache_root: Path, *, name: str) -> Path:
    cache_root = _safe_cache_root(cache_root)
    target = _target(cache_root, "blob", sha256) / name
    if target.is_file():
        if _sha256(target) != sha256:
            raise ValueError(f"shared blob bytes drifted: {target}")
        return target
    if target.parent.exists() and any(target.parent.iterdir()):
        raise ValueError(f"partial shared blob cache: {target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".incomplete-{uuid.uuid4().hex}")
    result = subprocess.run(
        ["aws", "s3", "cp", uri, str(temporary), "--only-show-errors", "--region", "us-west-2"],
        check=False,
    )
    if result.returncode or _sha256(temporary) != sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to stage exact shared blob {uri}")
    temporary.replace(target)
    target.chmod(0o444)
    return target


def materialize_blob(
    source: Path,
    sha256: str,
    cache_home: Path,
    *,
    relative_path: str,
) -> Path:
    """Expose an exact shared blob as a real file inside a consumer cache.

    OpenPI resolves cached files before checking that they remain below ``OPENPI_DATA_HOME``.
    A symlink from a cell-local cache to the campaign artifact cache therefore escapes that
    containment check.  A hard link keeps one physical copy while giving every cell a genuine
    in-cache path; filesystems that cannot hard-link across the two roots fall back to a verified
    copy without redownloading the object.
    """
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("materialized blob key must be a lowercase SHA-256")
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"shared blob source must be a regular non-symlink file: {source_path}")
    source_path = source_path.resolve()
    if _sha256(source_path) != sha256:
        raise ValueError(f"shared blob source bytes drifted: {source_path}")

    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe consumer-cache blob path: {relative_path!r}")
    home = _safe_cache_root(cache_home)
    parent = home
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError(f"consumer-cache parent is a symlink: {parent}")
        parent.mkdir(exist_ok=True)
    target = parent / relative.name
    if not target.parent.resolve().is_relative_to(home):
        raise ValueError("consumer-cache blob target escapes cache home")
    if target.is_symlink():
        raise ValueError(f"consumer-cache blob target is a symlink: {target}")
    if target.exists():
        if not target.is_file() or _sha256(target) != sha256:
            raise ValueError(f"consumer-cache blob bytes drifted: {target}")
        return target

    temporary = target.with_name(target.name + f".incomplete-{uuid.uuid4().hex}")
    try:
        try:
            os.link(source_path, temporary)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                raise
            shutil.copyfile(source_path, temporary)
            temporary.chmod(0o444)
        if _sha256(temporary) != sha256:
            raise RuntimeError(f"consumer-cache blob materialization drifted: {target}")
        try:
            # Linking the verified temporary into place is create-once and never overwrites a
            # concurrently materialized cache entry.
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or _sha256(target) != sha256:
                raise ValueError(f"consumer-cache blob destination raced or drifted: {target}")
    finally:
        temporary.unlink(missing_ok=True)
    if target.resolve().parent != target.parent.resolve() or not target.resolve().is_relative_to(home):
        raise ValueError(f"materialized blob escaped consumer cache: {target}")
    return target


def prepare_s3_tree(
    uri: str,
    manifest_sha256: str,
    cache_root: Path,
    *,
    category: str,
) -> Path:
    """Cache a manifest-sealed S3 prefix and verify every local byte on each reuse."""
    if not category.replace("_", "").isalnum():
        raise ValueError(f"unsafe shared tree category {category!r}")
    identity = {
        "kind": "s3_manifest_tree",
        "uri": uri.rstrip("/"),
        "manifest_sha256": manifest_sha256,
        "category": category,
    }

    def materialize(destination: Path) -> None:
        result = subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                uri,
                str(destination),
                "--only-show-errors",
                "--region",
                "us-west-2",
            ],
            check=False,
        )
        manifest = destination / "MANIFEST.json"
        if result.returncode or not manifest.is_file() or _sha256(manifest) != manifest_sha256:
            raise RuntimeError(f"failed to stage exact manifest tree {uri}")

    return _prepare_directory(
        _safe_cache_root(cache_root),
        category,
        manifest_sha256,
        identity=identity,
        materialize=materialize,
    )


def source_receipt(root: Path) -> str:
    """Hash immutable OpenPI source while excluding the environment and bytecode caches."""
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if ".venv" not in path.relative_to(root).parts
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix != ".pyc"
            and path.name != ".ROBOMME_OPENPI_CACHE.json"
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        if path.is_dir():
            digest.update(b"d")
        elif path.is_file():
            digest.update(b"f")
            digest.update(bytes.fromhex(_sha256(path)))
        else:
            raise ValueError(f"unsupported OpenPI cache entry {path}")
    return digest.hexdigest()


def verify_openpi(root: Path, archive_sha256: str) -> None:
    marker = root / ".ROBOMME_OPENPI_CACHE.json"
    if not root.is_dir() or not marker.is_file() or not (root / ".venv/bin/python").is_file():
        raise ValueError(f"shared OpenPI environment is incomplete: {root}")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value != {
        "schema_version": 1,
        "archive_sha256": archive_sha256,
        "source_receipt_sha256": source_receipt(root),
    }:
        raise ValueError(f"shared OpenPI environment/source drifted: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    blob = subparsers.add_parser("blob")
    blob.add_argument("--uri", required=True)
    blob.add_argument("--sha256", required=True)
    blob.add_argument("--cache-root", type=Path, required=True)
    blob.add_argument("--name", required=True)
    materialized_blob = subparsers.add_parser("materialize-blob")
    materialized_blob.add_argument("--source", type=Path, required=True)
    materialized_blob.add_argument("--sha256", required=True)
    materialized_blob.add_argument("--cache-home", type=Path, required=True)
    materialized_blob.add_argument("--relative-path", required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--manifest", type=Path, required=True)
    inventory_parser.add_argument("--artifact", choices=tuple(inventory.ARTIFACTS), required=True)
    inventory_parser.add_argument("--root-s3", required=True)
    inventory_parser.add_argument("--manifest-sha256", required=True)
    inventory_parser.add_argument("--cache-root", type=Path, required=True)
    inventory_parser.add_argument("--workers", type=int, default=48)
    task = subparsers.add_parser("task")
    task.add_argument("--parent-manifest", type=Path, required=True)
    task.add_argument("--task", required=True)
    task.add_argument("--root-s3", required=True)
    task.add_argument("--derived-sha256", required=True)
    task.add_argument("--cache-root", type=Path, required=True)
    task.add_argument("--workers", type=int, default=48)
    tree = subparsers.add_parser("tree")
    tree.add_argument("--uri", required=True)
    tree.add_argument("--manifest-sha256", required=True)
    tree.add_argument("--cache-root", type=Path, required=True)
    tree.add_argument("--category", required=True)
    verify = subparsers.add_parser("verify-openpi")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    if args.command == "blob":
        result = prepare_blob(args.uri, args.sha256, args.cache_root, name=args.name)
    elif args.command == "materialize-blob":
        result = materialize_blob(
            args.source,
            args.sha256,
            args.cache_home,
            relative_path=args.relative_path,
        )
    elif args.command == "inventory":
        result = prepare_inventory(
            args.manifest,
            artifact=args.artifact,
            root_s3=args.root_s3,
            manifest_sha256=args.manifest_sha256,
            cache_root=args.cache_root,
            workers=args.workers,
        )
    elif args.command == "task":
        result = prepare_task(
            args.parent_manifest,
            task=args.task,
            root_s3=args.root_s3,
            derived_sha256=args.derived_sha256,
            cache_root=args.cache_root,
            workers=args.workers,
        )
    elif args.command == "tree":
        result = prepare_s3_tree(
            args.uri,
            args.manifest_sha256,
            args.cache_root,
            category=args.category,
        )
    else:
        verify_openpi(args.root, args.archive_sha256)
        result = args.root
    print(result)


if __name__ == "__main__":
    main()
