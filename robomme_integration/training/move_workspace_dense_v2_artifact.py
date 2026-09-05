"""Fail-closed validation for the MoveCube dense/multi-point VISReg-v2 completion claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .single_task import task_manifest_sha256
from .workspace_deliberative_dense_v2 import PROTOCOL
from .workspace_gpu_producer_dense_v2 import (
    ARTIFACT_ROOT,
    CAMPAIGN,
    TASK,
    validate_manifest,
)
from .workspace_supervision_dense_v2 import ARTIFACT, TARGET_SEMANTICS


def _hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be lowercase hex64")
    return value


def _artifact(value: object, label: str, seal_field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an artifact dictionary")
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri.startswith("s3://") or uri.endswith("/"):
        raise ValueError(f"{label}.uri is not canonical S3")
    _hex64(value.get(seal_field), f"{label}.{seal_field}")
    return value


def load_completion_claim(
    path: str | Path,
    *,
    expected_manifest: dict,
    expected_manifest_sha256: str,
    expected_sha256: str | None = None,
) -> dict:
    validate_manifest(
        expected_manifest,
        manifest_sha256=expected_manifest_sha256,
        expected_source_tree_sha256=expected_manifest["source"]["source_tree_sha256"],
        expected_entry_sha256=expected_manifest["source"]["entry_sha256"],
    )
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != _hex64(expected_sha256, "claim SHA-256"):
        raise ValueError("dense v2 claim SHA-256 mismatch")
    claim = json.loads(payload)
    if (
        claim.get("schema_version") != 2
        or claim.get("kind") != "robomme_move_workspace_dense_v2_complete"
        or claim.get("campaign") != CAMPAIGN
        or claim.get("task") != TASK
        or claim.get("run_id") != expected_manifest["identity"]["run_id"]
        or claim.get("scientific_spec_sha256") != expected_manifest["identity"]["scientific_spec_sha256"]
        or claim.get("source_tree_sha256") != expected_manifest["source"]["source_tree_sha256"]
        or claim.get("task_manifest_sha256") != task_manifest_sha256(TASK)
        or claim.get("protocol") != PROTOCOL
        or claim.get("supervision_artifact") != ARTIFACT
        or claim.get("target_semantics") != TARGET_SEMANTICS
        or claim.get("regularizer") != {"name": "visreg", "weight": 0.05, "sigreg_weight": 0.0}
        or claim.get("ema_decay") != 0.999
    ):
        raise ValueError("dense v2 completion claim protocol/identity mismatch")
    encoder_id = _hex64(claim.get("encoder_id"), "encoder id")
    artifact_root = f"{ARTIFACT_ROOT}/{TASK}/{encoder_id}"
    omega = _artifact(claim.get("omega"), "omega", "manifest_sha256")
    supervision = _artifact(claim.get("supervision"), "supervision", "manifest_sha256")
    representation = _artifact(claim.get("representation"), "representation", "completion_sha256")
    if not isinstance(representation.get("step"), int) or representation["step"] < 1:
        raise ValueError("dense v2 representation step is invalid")
    if omega["uri"] != f"{artifact_root}/omega":
        raise ValueError("dense v2 omega namespace mismatch")
    if supervision["uri"] != f"{artifact_root}/supervision":
        raise ValueError("dense v2 supervision namespace mismatch")
    if representation["uri"] != f"{artifact_root}/representation/step-{representation['step']}":
        raise ValueError("dense v2 representation namespace mismatch")
    return claim
