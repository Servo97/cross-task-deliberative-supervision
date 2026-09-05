"""Fail-closed source identity checks for the project paper-protocol evaluator.

The evaluator imports code from three independent repositories.  A commit written in a result
manifest is evidence only when the imported module is inside that exact, clean Git worktree.  These
helpers are dependency-free so their failure modes can be unit tested without importing RoboMME.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

POLICY_SOURCE_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
BENCHMARK_SOURCE_COMMIT = "856bc3a189d4172f3f47dbee4424d585f8d78db3"
MANISKILL_SOURCE_COMMIT = "07be6fbc66350ddca200abfb0a11b692f078f7fd"
REFERENCE_EVALUATOR_SHA256 = "e82019b40e474036a1892a265a8ddf736165b331deac565e6b8f6ee323a2175d"

GitReader = Callable[[Path, Sequence[str]], str]


@dataclasses.dataclass(frozen=True)
class GitSourceAudit:
    label: str
    root: Path
    commit: str

    def manifest_record(self) -> dict[str, object]:
        # Host-specific paths are intentionally excluded so a byte-identical run can resume after
        # restaging at a different absolute location.
        return {"commit": self.commit, "tracked_tree_clean": True}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pinned_commit(label: str, supplied: str, expected: str) -> None:
    if supplied != expected:
        raise RuntimeError(f"{label} must be pinned to {expected}, got {supplied!r}")


def require_pinned_file(label: str, path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 drifted: {actual} != {expected_sha256}")


def _read_git(cwd: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"git {' '.join(arguments)} failed for {cwd}: {detail}")
    return result.stdout


def audit_imported_git_source(
    label: str,
    imported_file: str | Path,
    expected_commit: str,
    *,
    git_reader: GitReader = _read_git,
) -> GitSourceAudit:
    """Prove an imported file belongs to ``expected_commit`` with no tracked edits.

    Untracked files are deliberately ignored: they cannot change an imported tracked module, and
    treating caches or logs as source drift would make resumable evaluation needlessly brittle.
    A Git-less source export is rejected because its directory name is not commit evidence.
    """
    anchor = Path(imported_file).resolve()
    if not anchor.is_file():
        raise RuntimeError(f"{label} imported source file is missing: {anchor}")
    root_text = git_reader(anchor.parent, ("rev-parse", "--show-toplevel")).strip()
    if not root_text:
        raise RuntimeError(f"{label} Git root query returned an empty path")
    root = Path(root_text).resolve()
    if not anchor.is_relative_to(root):
        raise RuntimeError(f"{label} imported file {anchor} is outside claimed Git root {root}")
    relative = anchor.relative_to(root).as_posix()
    tracked = git_reader(root, ("ls-files", "--error-unmatch", "--", relative)).strip()
    if tracked != relative:
        raise RuntimeError(f"{label} imported file is not the tracked source at {relative!r}")
    head = git_reader(root, ("rev-parse", "HEAD")).strip().lower()
    if head != expected_commit:
        raise RuntimeError(f"{label} imported Git HEAD drifted: {head} != {expected_commit}")
    tracked_status = git_reader(
        root,
        ("status", "--porcelain=v1", "--untracked-files=no"),
    ).strip()
    if tracked_status:
        raise RuntimeError(f"{label} imported Git tree has tracked changes: {tracked_status}")
    return GitSourceAudit(label=label, root=root, commit=head)


def require_file_in_source(
    audit: GitSourceAudit,
    label: str,
    imported_file: str | Path,
    *,
    git_reader: GitReader = _read_git,
) -> None:
    path = Path(imported_file).resolve()
    if not path.is_file() or not path.is_relative_to(audit.root):
        raise RuntimeError(f"{label} imported file {path} is outside {audit.label} Git root {audit.root}")
    relative = path.relative_to(audit.root).as_posix()
    tracked = git_reader(audit.root, ("ls-files", "--error-unmatch", "--", relative)).strip()
    if tracked != relative:
        raise RuntimeError(f"{label} imported file is not the tracked source at {relative!r}")
