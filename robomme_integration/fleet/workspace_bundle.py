"""Materialize one sealed all-16 workspace index onto a training node."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import subprocess
from pathlib import Path

try:
    from ..training.workspace_index import load_workspace_index
except ImportError:  # SageMaker stages robomme_integration/ contents as top-level packages.
    from training.workspace_index import load_workspace_index


def _sync(uri: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            uri,
            str(destination),
            "--only-show-errors",
            "--no-follow-symlinks",
            "--region",
            "us-west-2",
        ],
        check=True,
    )


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} seal is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} seal SHA-256 mismatch: {actual} != {expected}")


def materialize(args: argparse.Namespace) -> None:
    index = load_workspace_index(
        args.index,
        expected_sha256=args.index_sha256,
        require_supervision=args.require_supervision,
    )
    workspace_root = Path(args.workspace_root)
    supervision_root = Path(args.supervision_root) if args.supervision_root else None
    work: list[tuple[str, str, Path, str]] = []
    for task, record in index["tasks"].items():
        work.append(
            (
                task,
                "omega",
                workspace_root / task,
                record["omega"]["uri"],
            )
        )
        if args.require_supervision:
            if supervision_root is None:
                raise ValueError("--require-supervision requires --supervision-root")
            work.append(
                (
                    task,
                    "supervision",
                    supervision_root / task,
                    record["supervision"]["uri"],
                )
            )

    def transfer(item: tuple[str, str, Path, str]) -> tuple[str, str, Path]:
        task, kind, destination, uri = item
        _sync(uri, destination)
        return task, kind, destination

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(work))) as executor:
        completed = list(executor.map(transfer, work))
    for task, kind, destination in completed:
        record = index["tasks"][task][kind]
        _verify(destination / "MANIFEST.json", record["manifest_sha256"], f"{task}/{kind}")
    print(
        f"ROBOMME ALL16 WORKSPACE MATERIALIZED tasks={len(index['tasks'])} supervision={args.require_supervision}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--supervision-root")
    parser.add_argument("--require-supervision", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must lie in [1,16]")
    materialize(args)


if __name__ == "__main__":
    main()
