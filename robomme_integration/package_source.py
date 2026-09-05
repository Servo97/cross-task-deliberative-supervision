#!/usr/bin/env python3
"""Build a deterministic archive containing only the isolated RoboMME integration package."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "launch"))
from build_deterministic_archive import build_archive  # noqa: E402
from launch_guardrails import _ignore_sensitive_source_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/robomme-source-archives")
    parser.add_argument("--study-root", required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="robomme-package-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        shutil.copytree(
            package,
            source / "robomme_integration",
            symlinks=True,
            ignore=_ignore_sensitive_source_files,
        )
        path, digest, uri = build_archive(
            source,
            output_dir=args.output_dir,
            component="robomme_integration",
            study_root=args.study_root,
        )
    print(f"path={path}\nsha256={digest}\ncanonical_uri={uri}\nupload=false")


if __name__ == "__main__":
    main()
