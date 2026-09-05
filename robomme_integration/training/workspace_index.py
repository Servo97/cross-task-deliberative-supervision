"""Sealed all-16 RoboMME task-to-workspace artifact index.

The policy sees only ``omega_t``.  This index is infrastructure/provenance: it binds every
environment task to the task-specific deliberative encoder, causal omega cache, and optional
salient supervision used to produce that token.  Episode ranges remain pinned by
``single_task.py`` and are never inferred from language strings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .single_task import EPISODES_PER_TASK, TASK_ORDER, task_manifest_sha256

HEX64 = frozenset("0123456789abcdef")


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in HEX64 for character in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _artifact(value: Any, label: str, *, seal_field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an artifact dictionary")
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri.startswith("s3://") or uri.endswith("/"):
        raise ValueError(f"{label}.uri must be a canonical S3 object/prefix without a trailing slash")
    seal = value.get(seal_field)
    if not isinstance(seal, str):
        raise ValueError(f"{label}.{seal_field} is required")
    _sha256(seal, f"{label}.{seal_field}")
    return value


def load_workspace_index(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_supervision: bool = False,
) -> dict:
    """Load and fail-closed validate one complete all-16 workspace index."""
    path = Path(path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != _sha256(expected_sha256, "workspace index SHA-256"):
        raise ValueError(f"workspace index SHA-256 mismatch: {actual} != {expected_sha256}")
    value = json.loads(payload)
    if value.get("schema_version") != 1 or value.get("benchmark") != "RoboMME":
        raise ValueError("workspace index has an unsupported schema or benchmark")
    if value.get("scope") != "all16" or value.get("task_order") != list(TASK_ORDER):
        raise ValueError("workspace index is not pinned to the canonical all-16 task order")
    tasks = value.get("tasks")
    if not isinstance(tasks, dict) or tuple(tasks) != TASK_ORDER:
        raise ValueError("workspace index task map must contain all 16 tasks in canonical order")

    for task in TASK_ORDER:
        record = tasks[task]
        if not isinstance(record, dict):
            raise ValueError(f"workspace index record for {task} is not a dictionary")
        if record.get("task_manifest_sha256") != task_manifest_sha256(task):
            raise ValueError(f"workspace index task manifest mismatch for {task}")
        encoder_id = record.get("encoder_id")
        if not isinstance(encoder_id, str):
            raise ValueError(f"workspace index encoder_id missing for {task}")
        _sha256(encoder_id, f"{task}.encoder_id")
        _artifact(record.get("omega"), f"{task}.omega", seal_field="manifest_sha256")
        representation = _artifact(
            record.get("representation"),
            f"{task}.representation",
            seal_field="completion_sha256",
        )
        step = representation.get("step")
        if not isinstance(step, int) or step < 1:
            raise ValueError(f"{task}.representation.step must be a positive integer")
        supervision = record.get("supervision")
        if require_supervision and supervision is None:
            raise ValueError(f"salient all-16 training requires supervision for {task}")
        if supervision is not None:
            _artifact(supervision, f"{task}.supervision", seal_field="manifest_sha256")
    return value


def task_for_episode(episode: int) -> str:
    """Return the immutable environment task for a global RoboMME training episode."""
    total = len(TASK_ORDER) * EPISODES_PER_TASK
    if not isinstance(episode, int) or not 0 <= episode < total:
        raise ValueError(f"RoboMME episode must lie in [0, {total}), got {episode!r}")
    return TASK_ORDER[episode // EPISODES_PER_TASK]
