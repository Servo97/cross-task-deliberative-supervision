"""Immutable p5 cloud contract for the FrameSamp B1 transition canary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robomme_integration.eval.framesamp_am_r1_canary import H100_NAME
from robomme_integration.eval.framesamp_am_r1_cloud import (
    INSTANCE_TYPE,
    QUEUE,
)
from robomme_integration.eval.framesamp_b1_canary import (
    KIND as RECEIPT_KIND,
)
from robomme_integration.eval.framesamp_b1_canary import (
    LANES,
    validate_canary_packet,
    validate_canary_receipt,
)
from robomme_integration.training.framesamp_b1_policy_overlay import (
    PATCHED_MEM_BUFFER_SHA256,
    PATCHED_POLICY_SHA256,
)

CHECKPOINT_ARCHIVE_SHA256 = "092a9c38b36650e086f9d327f8061a1b4a93ba784961ba7469d2d4d1e4a3a151"
CHECKPOINT_SEMANTIC_SHA256 = "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
CHECKPOINT_ARCHIVE_S3 = (
    "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/"
    "long_context_v1/artifacts/robomme/released_checkpoints/perceptual-framesamp-modul/"
    f"{CHECKPOINT_SEMANTIC_SHA256}/{CHECKPOINT_ARCHIVE_SHA256}.tgz"
)
EVAL_RUNTIME_SHA256 = "60da89c378241f75b3244be408c845989ac79f06831e63e81191851c3e3803f2"
EVAL_RUNTIME_S3 = (
    "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/"
    f"long_context_v1/artifacts/robomme/eval_runtime/v0.4.0/{EVAL_RUNTIME_SHA256}.tgz"
)
VISION_SHA256 = "f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293"
VISION_S3 = (
    "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/"
    "long_context_v1/artifacts/vision_encoders/pi05/59bd9ff4d58ea0638064bda851fd7d477ee9708c/"
    "siglip_params.pkl"
)
UPSTREAM_REPO = "https://github.com/RoboMME/robomme_policy_learning.git"
UPSTREAM_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_b1_p5_canary_manifest"
ENTRY = "gpu_framesamp_b1_entry.sh"
STAGED_MANIFEST = "_robomme_framesamp_b1_manifest.json"
STAGED_PACKET = "_robomme_framesamp_b1_packet.json"
STAGED_FILES = frozenset({STAGED_MANIFEST, STAGED_PACKET})
PACKET_FILE_SHA256 = "d0ea02857671b7c0d26dcee3341c217e87def9c45a943dc349e4ce88bb672be8"
PACKET_SHA256 = "ec6a197bc2f9476a1f2e04c32d2a349901643638153d88dc8752904d43e06acf"
OVERLAY_MANIFEST_SHA256 = "09bc3d6e04570d200056c499b45c360315e3cf497903c07cd02d8a203bf6c916"
OVERLAY_SOURCE_TREE_SHA256 = "dd20edbaba508000513a6673091fcbc5ded4a3d8df5845359e3171cb3085ab74"
TRANSITION_RUNNER_SHA256 = "1cab88afd7811f0137f1180f1dc3634c3113dc736222c3f55eae0c1cf4e17227"
SIM_WORKER_SHA256 = "92630a323d27c5b6aed804836fd7d0234767a999378c9cb2cd7bb2902a7e5874"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_json(value: object) -> str:
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise ValueError(f"refusing to reseal B1 field {field}")
    result = dict(value)
    result[field] = sha_json(value)
    return result


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    claimed = _sha(manifest.get("manifest_sha256"), "B1 manifest SHA")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if sha_json(unsigned) != claimed:
        raise ValueError("B1 cloud manifest seal mismatch")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != KIND
        or not isinstance(manifest.get("canary_id"), str)
        or not manifest["canary_id"].startswith("fs-b1-p5-canary-")
    ):
        raise ValueError("B1 cloud manifest identity mismatch")
    identity = manifest.get("identity")
    source = manifest.get("source")
    assets = manifest.get("assets")
    infrastructure = manifest.get("infrastructure")
    publication = manifest.get("publication")
    if not all(isinstance(value, dict) for value in (identity, source, assets, infrastructure, publication)):
        raise ValueError("B1 cloud manifest sections are malformed")
    identity_sha = _sha(manifest.get("identity_sha256"), "B1 identity SHA")
    if sha_json(identity) != identity_sha or manifest["canary_id"] != f"fs-b1-p5-canary-{identity_sha[:20]}":
        raise ValueError("B1 canary ID is not derived from its identity")
    if identity != {
        "packet_file_sha256": PACKET_FILE_SHA256,
        "packet_sha256": PACKET_SHA256,
        "prepared_source_tree_sha256": source.get("prepared_source_tree_sha256"),
        "overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
        "overlay_source_tree_sha256": OVERLAY_SOURCE_TREE_SHA256,
        "patched_policy_sha256": PATCHED_POLICY_SHA256,
        "patched_mem_buffer_sha256": PATCHED_MEM_BUFFER_SHA256,
        "canary_lanes": [
            {"lane": lane, "task": task, "episode": episode} for lane, (task, episode) in enumerate(LANES)
        ],
    }:
        raise ValueError("B1 cloud scientific identity drifted")
    if (
        source.get("entry") != ENTRY
        or source.get("submitted_entry_mode") != "0o755"
        or source.get("sagemaker_runtime_entry_mode") != "0o777"
        or source.get("staged_files_excluded_from_tree_sha256") != sorted(STAGED_FILES)
    ):
        raise ValueError("B1 cloud source contract drifted")
    _sha(source.get("prepared_source_tree_sha256"), "B1 prepared source SHA")
    _sha(source.get("entry_bytes_sha256"), "B1 entry SHA")
    expected_assets = {
        "checkpoint_archive": {
            "uri": CHECKPOINT_ARCHIVE_S3,
            "sha256": CHECKPOINT_ARCHIVE_SHA256,
        },
        "checkpoint_semantic_sha256": CHECKPOINT_SEMANTIC_SHA256,
        "eval_runtime": {"uri": EVAL_RUNTIME_S3, "sha256": EVAL_RUNTIME_SHA256},
        "policy_runtime": {
            "source": "upstream_uv_lock",
            "jax": "0.5.3",
            "orbax_checkpoint": "0.11.13",
        },
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "vision": {"uri": VISION_S3, "sha256": VISION_SHA256},
    }
    if assets != expected_assets:
        raise ValueError("B1 cloud asset identity drifted")
    if infrastructure != {
        "queue": QUEUE,
        "instance_type": INSTANCE_TYPE,
        "topology": f"1x8 {H100_NAME}",
        "priority": 100,
        "max_run_seconds": 14_400,
        "volume_size_gb": 250,
        "training_plan_arn": None,
        "sm_use_reserved_capacity": "1",
    }:
        raise ValueError("B1 cloud infrastructure drifted")
    namespace = publication.get("namespace_s3")
    if (
        not isinstance(namespace, str)
        or not namespace.endswith("/" + manifest["canary_id"])
        or publication
        != {
            "namespace_s3": namespace,
            "receipt_s3": f"{namespace}/canary.complete.json",
            "create_once": True,
            "only_allowed_object": "canary.complete.json",
            "scored_evidence": False,
        }
    ):
        raise ValueError("B1 cloud publication contract drifted")


def validate_environment(manifest: dict[str, Any], environment: dict[str, str]) -> None:
    validate_manifest(manifest)
    assets = manifest["assets"]
    expected = {
        "FS_B1_CANARY_ID": manifest["canary_id"],
        "FS_B1_MANIFEST_SHA256": manifest["manifest_sha256"],
        "FS_B1_NAMESPACE_S3": manifest["publication"]["namespace_s3"],
        "FS_B1_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "FS_B1_EXPECTED_SOURCE_SHA256": manifest["source"]["prepared_source_tree_sha256"],
        "FS_B1_PACKET_FILE_SHA256": PACKET_FILE_SHA256,
        "FS_B1_PACKET_SHA256": PACKET_SHA256,
        "FS_B1_OVERLAY_MANIFEST_SHA256": OVERLAY_MANIFEST_SHA256,
        "FS_B1_OVERLAY_SOURCE_TREE_SHA256": OVERLAY_SOURCE_TREE_SHA256,
        "FS_B1_CHECKPOINT_S3": assets["checkpoint_archive"]["uri"],
        "FS_B1_CHECKPOINT_ARCHIVE_SHA256": assets["checkpoint_archive"]["sha256"],
        "FS_B1_CHECKPOINT_SEMANTIC_SHA256": assets["checkpoint_semantic_sha256"],
        "ROBOMME_EVAL_RUNTIME_S3": assets["eval_runtime"]["uri"],
        "ROBOMME_EVAL_RUNTIME_SHA256": assets["eval_runtime"]["sha256"],
        "ROBOMME_EVAL_VISION_S3": assets["vision"]["uri"],
        "ROBOMME_EVAL_VISION_SHA256": assets["vision"]["sha256"],
        "ROBOMME_EVAL_UPSTREAM_REPO": assets["upstream_repo"],
        "ROBOMME_EVAL_UPSTREAM_COMMIT": assets["upstream_commit"],
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    if environment != expected:
        raise ValueError("B1 cloud environment differs from its sealed manifest")


def validate_staged_packet(
    *,
    manifest: dict[str, Any],
    packet: dict[str, Any],
    packet_file_sha256: str,
    source_root: Path,
    overlay_root: Path,
) -> None:
    validate_manifest(manifest)
    if packet_file_sha256 != PACKET_FILE_SHA256 or packet.get("packet_sha256") != PACKET_SHA256:
        raise ValueError("B1 staged packet identity mismatch")
    validate_canary_packet(packet, source_root=source_root, overlay_root=overlay_root)


def bind_cloud_receipt(mechanism_receipt: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Bind the runtime proof to this exact cloud identity and namespace."""

    validate_manifest(manifest)
    validate_canary_receipt(mechanism_receipt)
    if (
        mechanism_receipt.get("kind") != RECEIPT_KIND
        or mechanism_receipt.get("overlay_manifest_sha256") != OVERLAY_MANIFEST_SHA256
        or mechanism_receipt.get("runner_sha256") != TRANSITION_RUNNER_SHA256
        or mechanism_receipt.get("sim_worker_sha256") != SIM_WORKER_SHA256
    ):
        raise ValueError("B1 cloud receipt runtime identity drifted")
    return seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "robomme_framesamp_b1_cloud_receipt",
            "status": "HARD_GREEN",
            "canary_id": manifest["canary_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "prepared_source_tree_sha256": manifest["source"]["prepared_source_tree_sha256"],
            "publication_receipt_s3": manifest["publication"]["receipt_s3"],
            "mechanism_receipt_sha256": mechanism_receipt["receipt_sha256"],
            "mechanism_receipt": mechanism_receipt,
            "scored_evidence": False,
            "deployable": False,
        },
        "cloud_receipt_sha256",
    )


def validate_receipt(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    claimed = _sha(receipt.get("cloud_receipt_sha256"), "B1 cloud receipt SHA")
    unsigned = dict(receipt)
    unsigned.pop("cloud_receipt_sha256", None)
    if sha_json(unsigned) != claimed:
        raise ValueError("B1 cloud receipt seal mismatch")
    mechanism_receipt = receipt.get("mechanism_receipt")
    if not isinstance(mechanism_receipt, dict):
        raise ValueError("B1 cloud receipt mechanism proof is malformed")
    if receipt != bind_cloud_receipt(mechanism_receipt, manifest):
        raise ValueError("B1 cloud receipt identity drifted")


def load_and_validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("B1 manifest must be a JSON object")
    validate_manifest(value)
    return value
