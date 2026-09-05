from __future__ import annotations

from copy import deepcopy

import pytest

from robomme_integration.eval import framesamp_b1_cloud as cloud
from robomme_integration.eval.framesamp_b1_canary import LANES
from robomme_integration.training.framesamp_b1_policy_overlay import (
    PATCHED_MEM_BUFFER_SHA256,
    PATCHED_POLICY_SHA256,
)


def _manifest() -> dict:
    source = {
        "prepared_source_tree_sha256": "3" * 64,
        "entry": cloud.ENTRY,
        "entry_bytes_sha256": "4" * 64,
        "submitted_entry_mode": "0o755",
        "sagemaker_runtime_entry_mode": "0o777",
        "staged_files_excluded_from_tree_sha256": sorted(cloud.STAGED_FILES),
    }
    identity = {
        "packet_file_sha256": cloud.PACKET_FILE_SHA256,
        "packet_sha256": cloud.PACKET_SHA256,
        "prepared_source_tree_sha256": source["prepared_source_tree_sha256"],
        "overlay_manifest_sha256": cloud.OVERLAY_MANIFEST_SHA256,
        "overlay_source_tree_sha256": cloud.OVERLAY_SOURCE_TREE_SHA256,
        "patched_policy_sha256": PATCHED_POLICY_SHA256,
        "patched_mem_buffer_sha256": PATCHED_MEM_BUFFER_SHA256,
        "canary_lanes": [
            {"lane": lane, "task": task, "episode": episode} for lane, (task, episode) in enumerate(LANES)
        ],
    }
    identity_sha = cloud.sha_json(identity)
    canary_id = f"fs-b1-p5-canary-{identity_sha[:20]}"
    namespace = f"s3://example/manifests/canaries/framesamp_b1/{canary_id}"
    return cloud.seal(
        {
            "schema_version": cloud.SCHEMA_VERSION,
            "kind": cloud.KIND,
            "canary_id": canary_id,
            "identity": identity,
            "identity_sha256": identity_sha,
            "source": source,
            "assets": {
                "checkpoint_archive": {
                    "uri": cloud.CHECKPOINT_ARCHIVE_S3,
                    "sha256": cloud.CHECKPOINT_ARCHIVE_SHA256,
                },
                "checkpoint_semantic_sha256": cloud.CHECKPOINT_SEMANTIC_SHA256,
                "eval_runtime": {
                    "uri": cloud.EVAL_RUNTIME_S3,
                    "sha256": cloud.EVAL_RUNTIME_SHA256,
                },
                "policy_runtime": {
                    "source": "upstream_uv_lock",
                    "jax": "0.5.3",
                    "orbax_checkpoint": "0.11.13",
                },
                "upstream_repo": cloud.UPSTREAM_REPO,
                "upstream_commit": cloud.UPSTREAM_COMMIT,
                "vision": {"uri": cloud.VISION_S3, "sha256": cloud.VISION_SHA256},
            },
            "infrastructure": {
                "queue": cloud.QUEUE,
                "instance_type": cloud.INSTANCE_TYPE,
                "topology": "1x8 NVIDIA H100 80GB HBM3",
                "priority": 100,
                "max_run_seconds": 14_400,
                "volume_size_gb": 250,
                "training_plan_arn": None,
                "sm_use_reserved_capacity": "1",
            },
            "publication": {
                "namespace_s3": namespace,
                "receipt_s3": f"{namespace}/canary.complete.json",
                "create_once": True,
                "only_allowed_object": "canary.complete.json",
                "scored_evidence": False,
            },
        },
        "manifest_sha256",
    )


def _reseal_manifest(value: dict) -> dict:
    unsigned = deepcopy(value)
    unsigned.pop("manifest_sha256", None)
    return cloud.seal(unsigned, "manifest_sha256")


def _reseal_receipt(value: dict) -> dict:
    unsigned = deepcopy(value)
    unsigned.pop("cloud_receipt_sha256", None)
    return cloud.seal(unsigned, "cloud_receipt_sha256")


def test_manifest_rejects_resealed_identity_infrastructure_and_publication_drift():
    manifest = _manifest()
    cloud.validate_manifest(manifest)
    for section, key, value in (
        ("identity", "packet_sha256", "0" * 64),
        ("infrastructure", "instance_type", "ml.p5e.48xlarge"),
        ("publication", "only_allowed_object", "score.json"),
    ):
        changed = deepcopy(manifest)
        changed[section][key] = value
        with pytest.raises(ValueError):
            cloud.validate_manifest(_reseal_manifest(changed))


def test_cloud_receipt_binds_canary_manifest_source_and_namespace(monkeypatch):
    manifest = _manifest()
    mechanism = {
        "kind": cloud.RECEIPT_KIND,
        "receipt_sha256": "a" * 64,
        "overlay_manifest_sha256": cloud.OVERLAY_MANIFEST_SHA256,
        "runner_sha256": cloud.TRANSITION_RUNNER_SHA256,
        "sim_worker_sha256": cloud.SIM_WORKER_SHA256,
    }
    monkeypatch.setattr(cloud, "validate_canary_receipt", lambda _: None)
    receipt = cloud.bind_cloud_receipt(mechanism, manifest)
    cloud.validate_receipt(receipt, manifest)
    for key, value in (
        ("canary_id", "fs-b1-p5-canary-wrong"),
        ("manifest_sha256", "0" * 64),
        ("prepared_source_tree_sha256", "1" * 64),
        ("publication_receipt_s3", "s3://example/wrong.json"),
    ):
        changed = deepcopy(receipt)
        changed[key] = value
        with pytest.raises(ValueError, match="identity drifted"):
            cloud.validate_receipt(_reseal_receipt(changed), manifest)


def test_cloud_receipt_rejects_wrong_runtime_proof(monkeypatch):
    manifest = _manifest()
    mechanism = {
        "kind": cloud.RECEIPT_KIND,
        "receipt_sha256": "a" * 64,
        "overlay_manifest_sha256": cloud.OVERLAY_MANIFEST_SHA256,
        "runner_sha256": "0" * 64,
        "sim_worker_sha256": cloud.SIM_WORKER_SHA256,
    }
    monkeypatch.setattr(cloud, "validate_canary_receipt", lambda _: None)
    with pytest.raises(ValueError, match="runtime identity"):
        cloud.bind_cloud_receipt(mechanism, manifest)
