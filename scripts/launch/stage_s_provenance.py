#!/usr/bin/env python3
"""Shared Stage-S provenance helpers: canonical JSON + content-addressed code hashing.

Every Stage-S artifact (task prompts, task-language table, source features, omega cache) records a
``producing_code.sha256`` so that changing ANY line of the code that produced it changes the
artifact's derived identity. This module is the single definition of "the code that produced X",
so the builders, the omega producer, and the tests all agree byte-for-byte.

Offline only: imports stdlib + nothing else. Never touches AWS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_json(value: object) -> str:
    """The one canonical JSON encoding used across every Stage-S manifest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_s_code_sha256(paths: list[str | Path]) -> str:
    """Content hash of a fixed, ordered set of source files.

    The digest is over ``<repo-relative-posix-path>\\n<sha256-of-bytes>\\n`` per file, in the exact
    order given (order is part of the contract — callers pass a deterministic list). A file that is
    absent is a hard error: a provenance hash must never silently omit an input.
    """
    lines: list[str] = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"provenance source file is missing: {path}")
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.name
        lines.append(f"{rel}\n{sha256_file(path)}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
