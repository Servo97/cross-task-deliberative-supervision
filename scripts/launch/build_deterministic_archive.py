#!/usr/bin/env python3
"""Build a deterministic, content-addressed source archive without uploading it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path

try:
    from .launch_guardrails import _ignore_sensitive_source_files
except ImportError:
    from launch_guardrails import _ignore_sensitive_source_files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(
    source_dir: str | Path,
    *,
    output_dir: str | Path,
    component: str,
    study_root: str,
) -> tuple[Path, str, str]:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"source directory is missing: {source}")
    if not component or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in component):
        raise ValueError("component must use lowercase letters, digits, hyphen, or underscore")
    if not study_root.startswith("s3://"):
        raise ValueError("study_root must be an s3:// URI")
    if output == source or output.is_relative_to(source):
        raise ValueError("output_dir must be outside source_dir")
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{component}-archive-") as temporary_dir:
        staged = Path(temporary_dir) / "source"
        shutil.copytree(
            source,
            staged,
            symlinks=True,
            ignore=_ignore_sensitive_source_files,
        )
        incomplete = output / f".{component}.tgz.incomplete"
        incomplete.unlink(missing_ok=True)
        with incomplete.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for path in sorted(staged.rglob("*"), key=lambda item: item.relative_to(staged).as_posix()):
                        relative = path.relative_to(staged).as_posix()
                        mode = path.lstat().st_mode
                        info = tarfile.TarInfo(relative)
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        if path.is_symlink():
                            info.type = tarfile.SYMTYPE
                            info.mode = 0o777
                            info.linkname = os.readlink(path)
                            archive.addfile(info)
                        elif path.is_dir():
                            info.type = tarfile.DIRTYPE
                            info.mode = 0o755
                            archive.addfile(info)
                        elif path.is_file():
                            info.type = tarfile.REGTYPE
                            info.mode = 0o755 if stat.S_IMODE(mode) & 0o111 else 0o644
                            info.size = path.stat().st_size
                            with path.open("rb") as stream:
                                archive.addfile(info, stream)
                        else:
                            raise ValueError(f"unsupported source entry type: {path}")
        digest = _sha256(incomplete)
        destination = output / f"{digest}.tgz"
        if destination.exists():
            if _sha256(destination) != digest:
                raise ValueError(f"content-addressed archive collision: {destination}")
            incomplete.unlink()
        else:
            incomplete.replace(destination)
    uri = f"{study_root.rstrip('/')}/code/{component}/{digest}.tgz"
    return destination, digest, uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--study-root", required=True)
    args = parser.parse_args()
    path, digest, uri = build_archive(
        args.source_dir,
        output_dir=args.output_dir,
        component=args.component,
        study_root=args.study_root,
    )
    print(f"path={path}")
    print(f"sha256={digest}")
    print(f"canonical_uri={uri}")
    print("upload=false")


if __name__ == "__main__":
    main()
