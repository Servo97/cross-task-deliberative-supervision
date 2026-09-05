"""Immutable cloud contract for the FS-R1 twelve-cell p5 screen."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from robomme_integration.eval.framesamp_am_r1_canary import H100_NAME
from robomme_integration.eval.framesamp_am_r1_cloud import (
    INSTANCE_TYPE,
    QUEUE,
    validate_canary_receipt,
)
from robomme_integration.eval.framesamp_am_r1_cloud import (
    validate_manifest as validate_canary_manifest,
)
from robomme_integration.eval.framesamp_am_r1_packet import validate as validate_packet

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_p5_screen_manifest"
ENTRY = "gpu_framesamp_am_r1_screen_entry.sh"
STAGED_MANIFEST = "_robomme_framesamp_am_r1_screen_manifest.json"
STAGED_PACKET = "_robomme_framesamp_am_r1_screen_packet.json"
STAGED_CANARY_MANIFEST = "_robomme_framesamp_am_r1_canary_manifest.json"
STAGED_CANARY_RECEIPT = "_robomme_framesamp_am_r1_canary_receipt.json"
STAGED_FILES = frozenset({STAGED_MANIFEST, STAGED_PACKET, STAGED_CANARY_MANIFEST, STAGED_CANARY_RECEIPT})
PACKET_FILE_SHA256 = "adcb2258fac42ccb29e6b4740085365945515a83c4c641bbca815dc8a69be960"
PACKET_SHA256 = "3c8d5c8fad587b813f3c48269a14f550e71a6349480642775a4f96b8a2c07bfa"
CANARY_ID = "fs-r1-p5-canary-732b43ca9cbc299b3a7d"
CANARY_MANIFEST_SHA256 = "2b45820c48b8d966a9b42f98cd67665b9febce6cca0ae9963ed35de41b355ee9"
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


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_json(value: object) -> str:
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise ValueError(f"refusing to reseal FS-R1 screen field {field}")
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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    claimed = _sha(manifest.get("manifest_sha256"), "screen manifest SHA")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if sha_json(unsigned) != claimed:
        raise ValueError("FS-R1 screen manifest seal mismatch")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != KIND
        or not isinstance(manifest.get("screen_id"), str)
        or not manifest["screen_id"].startswith("fs-r1-screen-")
    ):
        raise ValueError("FS-R1 screen manifest identity mismatch")
    identity = manifest.get("identity")
    source = manifest.get("source")
    assets = manifest.get("assets")
    infrastructure = manifest.get("infrastructure")
    publication = manifest.get("publication")
    if not all(isinstance(value, dict) for value in (identity, source, assets, infrastructure, publication)):
        raise ValueError("FS-R1 screen manifest sections are malformed")
    identity_sha = _sha(manifest.get("identity_sha256"), "screen identity SHA")
    if sha_json(identity) != identity_sha or manifest["screen_id"] != f"fs-r1-screen-{identity_sha[:20]}":
        raise ValueError("FS-R1 screen ID is not derived from identity")
    for key in (
        "packet_file_sha256",
        "packet_sha256",
        "prepared_source_tree_sha256",
        "canary_manifest_sha256",
        "canary_receipt_sha256",
        "canary_receipt_file_sha256",
    ):
        _sha(identity.get(key), f"screen identity {key}")
    if (
        identity.get("packet_file_sha256") != PACKET_FILE_SHA256
        or identity.get("packet_sha256") != PACKET_SHA256
        or identity.get("canary_id") != CANARY_ID
        or identity.get("canary_manifest_sha256") != CANARY_MANIFEST_SHA256
        or identity.get("canary_receipt_s3")
        != (
            "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/"
            f"long_context_v1/manifests/canaries/framesamp_am_r1/{CANARY_ID}/"
            "canary.complete.json"
        )
        or identity.get("episode_indices") != list(range(8))
        or identity.get("scientific_cells") != 12
    ):
        raise ValueError("FS-R1 screen identity matrix drifted")
    if (
        source.get("entry") != ENTRY
        or source.get("submitted_entry_mode") != "0o755"
        or source.get("sagemaker_runtime_entry_mode") != "0o777"
        or source.get("staged_files_excluded_from_tree_sha256") != sorted(STAGED_FILES)
    ):
        raise ValueError("FS-R1 screen source contract drifted")
    for key in ("prepared_source_tree_sha256", "entry_bytes_sha256"):
        _sha(source.get(key), f"screen source {key}")
    expected_asset_keys = {
        "checkpoint_archive",
        "checkpoint_semantic_sha256",
        "eval_runtime",
        "policy_runtime",
        "policy_overlay_manifest_sha256",
        "upstream_commit",
        "upstream_repo",
        "vision",
    }
    if set(assets) != expected_asset_keys:
        raise ValueError("FS-R1 screen asset fields mismatch")
    for name in ("checkpoint_archive", "eval_runtime", "vision"):
        asset = assets[name]
        if (
            not isinstance(asset, dict)
            or not str(asset.get("uri", "")).startswith("s3://")
            or _sha(asset.get("sha256"), f"screen {name} SHA") != asset["sha256"]
        ):
            raise ValueError(f"FS-R1 screen asset {name} is malformed")
    if assets["policy_runtime"] != {
        "source": "upstream_uv_lock",
        "jax": "0.5.3",
        "orbax_checkpoint": "0.11.13",
    }:
        raise ValueError("FS-R1 screen released-checkpoint runtime drifted")
    if (
        assets["checkpoint_archive"]
        != {
            "uri": CHECKPOINT_ARCHIVE_S3,
            "sha256": CHECKPOINT_ARCHIVE_SHA256,
        }
        or assets["checkpoint_semantic_sha256"] != CHECKPOINT_SEMANTIC_SHA256
    ):
        raise ValueError("FS-R1 screen checkpoint identity drifted")
    if assets["eval_runtime"] != {"uri": EVAL_RUNTIME_S3, "sha256": EVAL_RUNTIME_SHA256}:
        raise ValueError("FS-R1 screen evaluation runtime drifted")
    if assets["vision"] != {"uri": VISION_S3, "sha256": VISION_SHA256}:
        raise ValueError("FS-R1 screen vision asset drifted")
    if assets["policy_overlay_manifest_sha256"] != "5344b2a6898fe36cfec2565c2c2bcbab894244cfb30744a357c8585b33b8a85f":
        raise ValueError("FS-R1 screen overlay identity drifted")
    if assets["upstream_repo"] != UPSTREAM_REPO or assets["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("FS-R1 screen upstream commit drifted")
    if infrastructure != {
        "queue": QUEUE,
        "instance_type": INSTANCE_TYPE,
        "topology": f"1x8 {H100_NAME}",
        "priority": 100,
        "max_run_seconds": 86_400,
        "volume_size_gb": 300,
        "training_plan_arn": None,
        "sm_use_reserved_capacity": "1",
        "cell_schedule": "12 sequential cells x 8 concurrent episode lanes",
    }:
        raise ValueError("FS-R1 screen infrastructure contract drifted")
    namespace = publication.get("namespace_s3")
    results = publication.get("results")
    if (
        not isinstance(namespace, str)
        or not namespace.endswith("/" + manifest["screen_id"])
        or not isinstance(results, list)
        or len(results) != 12
        or len({row.get("cell_id") for row in results if isinstance(row, dict)}) != 12
        or any(
            not isinstance(row, dict)
            or row.get("result_s3") != f"{namespace}/cells/{row.get('cell_id')}/result.complete.json"
            for row in results
        )
        or publication.get("completion_s3") != f"{namespace}/screen.complete.json"
        or publication.get("create_once") is not True
        or publication.get("scored_evidence") is not True
        or publication.get("statistical_evidence") is not False
    ):
        raise ValueError("FS-R1 screen publication contract drifted")


def validate_environment(manifest: dict[str, Any], environment: dict[str, str]) -> None:
    validate_manifest(manifest)
    assets = manifest["assets"]
    expected = {
        "FS_R1_SCREEN_ID": manifest["screen_id"],
        "FS_R1_SCREEN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "FS_R1_SCREEN_NAMESPACE_S3": manifest["publication"]["namespace_s3"],
        "FS_R1_SCREEN_COMPLETION_S3": manifest["publication"]["completion_s3"],
        "FS_R1_EXPECTED_SOURCE_SHA256": manifest["source"]["prepared_source_tree_sha256"],
        "FS_R1_PACKET_FILE_SHA256": manifest["identity"]["packet_file_sha256"],
        "FS_R1_PACKET_SHA256": manifest["identity"]["packet_sha256"],
        "FS_R1_CANARY_ID": manifest["identity"]["canary_id"],
        "FS_R1_CANARY_MANIFEST_SHA256": manifest["identity"]["canary_manifest_sha256"],
        "FS_R1_CANARY_RECEIPT_SHA256": manifest["identity"]["canary_receipt_sha256"],
        "FS_R1_CANARY_RECEIPT_FILE_SHA256": manifest["identity"]["canary_receipt_file_sha256"],
        "FS_R1_CANARY_RECEIPT_S3": manifest["identity"]["canary_receipt_s3"],
        "FS_R1_CHECKPOINT_S3": assets["checkpoint_archive"]["uri"],
        "FS_R1_CHECKPOINT_ARCHIVE_SHA256": assets["checkpoint_archive"]["sha256"],
        "FS_R1_CHECKPOINT_SEMANTIC_SHA256": assets["checkpoint_semantic_sha256"],
        "ROBOMME_EVAL_RUNTIME_S3": assets["eval_runtime"]["uri"],
        "ROBOMME_EVAL_RUNTIME_SHA256": assets["eval_runtime"]["sha256"],
        "ROBOMME_EVAL_VISION_S3": assets["vision"]["uri"],
        "ROBOMME_EVAL_VISION_SHA256": assets["vision"]["sha256"],
        "ROBOMME_EVAL_UPSTREAM_REPO": assets["upstream_repo"],
        "ROBOMME_EVAL_UPSTREAM_COMMIT": assets["upstream_commit"],
        "FS_R1_OVERLAY_MANIFEST_SHA256": assets["policy_overlay_manifest_sha256"],
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    if environment != expected:
        raise ValueError("FS-R1 screen environment does not exactly match its manifest")


def validate_staged_inputs(
    *,
    manifest: dict[str, Any],
    packet: dict[str, Any],
    packet_file_sha256: str,
    canary_manifest: dict[str, Any],
    canary_receipt: dict[str, Any],
    canary_receipt_file_sha256: str,
) -> None:
    validate_manifest(manifest)
    validate_packet(packet)
    validate_canary_manifest(canary_manifest)
    validate_canary_receipt(canary_receipt, canary_manifest)
    identity = manifest["identity"]
    if (
        packet_file_sha256 != identity["packet_file_sha256"]
        or packet["packet_sha256"] != identity["packet_sha256"]
        or canary_manifest["manifest_sha256"] != identity["canary_manifest_sha256"]
        or canary_manifest["canary_id"] != identity["canary_id"]
        or canary_manifest["identity"]["packet_file_sha256"] != packet_file_sha256
        or canary_manifest["identity"]["packet_sha256"] != packet["packet_sha256"]
        or canary_receipt["receipt_sha256"] != identity["canary_receipt_sha256"]
        or canary_receipt_file_sha256 != identity["canary_receipt_file_sha256"]
    ):
        raise ValueError("FS-R1 screen staged packet/canary lineage mismatch")
    cell_ids = [cell["cell_id"] for cell in packet["oracle_cells"]]
    if [row["cell_id"] for row in manifest["publication"]["results"]] != cell_ids:
        raise ValueError("FS-R1 screen result ordering differs from the sealed packet")


def validate_completion(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    claimed = _sha(receipt.get("receipt_sha256"), "screen completion receipt SHA")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if sha_json(unsigned) != claimed:
        raise ValueError("FS-R1 screen completion seal mismatch")
    cells = receipt.get("cells")
    topology = receipt.get("topology")
    expected_results = manifest["publication"]["results"]
    if (
        receipt.get("kind") != "robomme_framesamp_am_r1_oracle_screen_completion"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("scope") != "n8_regression_screen_not_statistical_evidence"
        or receipt.get("packet_file_sha256") != manifest["identity"]["packet_file_sha256"]
        or receipt.get("packet_sha256") != manifest["identity"]["packet_sha256"]
        or receipt.get("cell_count") != 12
        or receipt.get("valid_episodes") != 96
        or receipt.get("harness_failures") != 0
        or receipt.get("all_cells_complete") is not True
        or not isinstance(cells, list)
        or [row.get("cell_id") for row in cells] != [row["cell_id"] for row in expected_results]
        or not isinstance(topology, list)
        or len(topology) != 8
        or [row.get("index") for row in topology] != list(range(8))
        or len({row.get("uuid") for row in topology}) != 8
        or any(
            row.get("name") != H100_NAME or not 80_000 <= int(row.get("memory_total_mib", 0)) <= 90_000
            for row in topology
        )
    ):
        raise ValueError("FS-R1 screen completion semantics mismatch")
    for record, destination in zip(cells, expected_results, strict=True):
        successes = record.get("successes")
        elapsed = record.get("cell_elapsed_seconds")
        if (
            set(record)
            != {
                "cell_id",
                "claim_sha256",
                "claim_file_sha256",
                "successes",
                "valid_episodes",
                "promote_to_paired_fixed50",
                "result_s3",
            }
            or record.get("valid_episodes") != 8
            or isinstance(successes, bool)
            or not isinstance(successes, int)
            or not 0 <= successes <= 8
            or not isinstance(record.get("promote_to_paired_fixed50"), bool)
            or record.get("result_s3") != destination["result_s3"]
            or _sha(record.get("claim_sha256"), "cell claim SHA") != record["claim_sha256"]
            or _sha(record.get("claim_file_sha256"), "cell claim file SHA") != record["claim_file_sha256"]
        ):
            raise ValueError("FS-R1 screen completion cell semantics mismatch")
        if elapsed is not None and (
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
        ):
            raise ValueError("FS-R1 screen completion cell duration is non-finite")


def load_and_validate(path: Path) -> dict[str, Any]:
    value = _load(path)
    validate_manifest(value)
    return value
