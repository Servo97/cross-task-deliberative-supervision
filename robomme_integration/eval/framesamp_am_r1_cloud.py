"""Immutable cloud contract for the isolated FS-R1 p5 runtime canary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robomme_integration.eval.framesamp_am_r1_canary import (
    H100_NAME,
    LANES,
    validate_lane_receipt,
)

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_p5_canary_manifest"
ENTRY = "gpu_framesamp_am_r1_entry.sh"
STAGED_MANIFEST = "_robomme_framesamp_am_r1_canary_manifest.json"
QUEUE = "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
INSTANCE_TYPE = "ml.p5.48xlarge"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_json(value: object) -> str:
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise ValueError(f"refusing to reseal FS-R1 field {field}")
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
    claimed = _sha(manifest.get("manifest_sha256"), "manifest SHA")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if sha_json(unsigned) != claimed:
        raise ValueError("FS-R1 cloud manifest seal mismatch")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != KIND
        or not isinstance(manifest.get("canary_id"), str)
        or not manifest["canary_id"].startswith("fs-r1-p5-canary-")
    ):
        raise ValueError("FS-R1 cloud manifest identity mismatch")
    identity = manifest.get("identity")
    source = manifest.get("source")
    assets = manifest.get("assets")
    infrastructure = manifest.get("infrastructure")
    publication = manifest.get("publication")
    if not all(isinstance(value, dict) for value in (identity, source, assets, infrastructure, publication)):
        raise ValueError("FS-R1 cloud manifest sections are malformed")
    identity_sha = _sha(manifest.get("identity_sha256"), "identity SHA")
    if sha_json(identity) != identity_sha or manifest["canary_id"] != f"fs-r1-p5-canary-{identity_sha[:20]}":
        raise ValueError("FS-R1 cloud canary ID is not derived from identity")
    for key in ("packet_file_sha256", "packet_sha256"):
        _sha(identity.get(key), f"identity {key}")
    if identity.get("canary_lanes") != [
        {
            "lane": lane,
            "task": task,
            "episode": episode,
            "budget": budget,
            "fit_mass": fit_mass,
        }
        for lane, (task, episode, budget, fit_mass) in enumerate(LANES)
    ]:
        raise ValueError("FS-R1 cloud canary lane matrix drifted")
    if (
        source.get("entry") != ENTRY
        or source.get("submitted_entry_mode") != "0o755"
        or source.get("sagemaker_runtime_entry_mode") != "0o777"
    ):
        raise ValueError("FS-R1 cloud source entry contract drifted")
    for key in ("prepared_source_tree_sha256", "entry_bytes_sha256"):
        _sha(source.get(key), f"source {key}")
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
        raise ValueError("FS-R1 cloud asset fields mismatch")
    for name in ("checkpoint_archive", "eval_runtime", "vision"):
        asset = assets[name]
        if (
            not isinstance(asset, dict)
            or not str(asset.get("uri", "")).startswith("s3://")
            or not asset.get("uri")
            or _sha(asset.get("sha256"), f"{name} SHA") != asset["sha256"]
        ):
            raise ValueError(f"FS-R1 cloud asset {name} is malformed")
    if assets["policy_runtime"] != {
        "source": "upstream_uv_lock",
        "jax": "0.5.3",
        "orbax_checkpoint": "0.11.13",
    }:
        raise ValueError("FS-R1 released-checkpoint policy runtime drifted")
    if assets["checkpoint_semantic_sha256"] != "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62":
        raise ValueError("FS-R1 cloud checkpoint is not the released FrameSamp checkpoint")
    if assets["policy_overlay_manifest_sha256"] != "5344b2a6898fe36cfec2565c2c2bcbab894244cfb30744a357c8585b33b8a85f":
        raise ValueError("FS-R1 cloud policy overlay identity drifted")
    if assets["upstream_commit"] != "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b":
        raise ValueError("FS-R1 cloud upstream commit drifted")
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
        raise ValueError("FS-R1 cloud infrastructure contract drifted")
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
        raise ValueError("FS-R1 cloud publication contract drifted")


def validate_environment(manifest: dict[str, Any], environment: dict[str, str]) -> None:
    validate_manifest(manifest)
    assets = manifest["assets"]
    expected = {
        "FS_R1_CANARY_ID": manifest["canary_id"],
        "FS_R1_MANIFEST_SHA256": manifest["manifest_sha256"],
        "FS_R1_NAMESPACE_S3": manifest["publication"]["namespace_s3"],
        "FS_R1_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "FS_R1_EXPECTED_SOURCE_SHA256": manifest["source"]["prepared_source_tree_sha256"],
        "FS_R1_PACKET_FILE_SHA256": manifest["identity"]["packet_file_sha256"],
        "FS_R1_PACKET_SHA256": manifest["identity"]["packet_sha256"],
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
        raise ValueError("FS-R1 cloud environment does not exactly match the manifest")


def validate_canary_receipt(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    seal_value = _sha(receipt.get("receipt_sha256"), "canary receipt SHA")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if sha_json(unsigned) != seal_value:
        raise ValueError("FS-R1 canary aggregate receipt seal mismatch")
    if (
        receipt.get("kind") != "robomme_framesamp_am_r1_8xh100_runtime_canary"
        or receipt.get("status") != "HARD_GREEN"
        or receipt.get("scope") != "runtime_canary_only_not_scored_evidence"
        or receipt.get("lane_count") != 8
        or receipt.get("authenticated_cuts") != 8
        or receipt.get("executed_simulator_actions") != 128
        or receipt.get("cloud_publication") is not False
        or len(receipt.get("topology", [])) != 8
        or len(receipt.get("lanes", [])) != 8
    ):
        raise ValueError("FS-R1 aggregate canary semantics mismatch")
    for lane, (task, episode, budget, fit_mass) in enumerate(LANES):
        lane_record = receipt["lanes"][lane]
        if (
            lane_record.get("lane") != lane
            or lane_record.get("task") != task
            or lane_record.get("episode") != episode
            or lane_record.get("budget") != budget
            or lane_record.get("fit_mass") is not fit_mass
            or lane_record.get("receipt_sha256") != lane_record.get("receipt", {}).get("receipt_sha256")
        ):
            raise ValueError("FS-R1 aggregate canary lane binding mismatch")
        validate_lane_receipt(
            lane_record["receipt"],
            task=task,
            episode=episode,
            budget=budget,
            fit_mass=fit_mass,
        )
    expected_sources = manifest["identity"]["source_files"]
    if (
        receipt.get("runner_sha256") != expected_sources["per_cut_rollout"]
        or receipt.get("sim_worker_sha256") != expected_sources["multi_replan_sim_worker"]
    ):
        raise ValueError("FS-R1 canary runtime source differs from the sealed packet")


def load_and_validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("FS-R1 cloud manifest must be an object")
    validate_manifest(value)
    return value
