#!/usr/bin/env python3
"""Build the review-only MoveCube dense/multi-point VISReg-v2 producer packet.

This module intentionally has no submission path.  A separate independent review must bind the
final source tree, manifest and GPU-canary receipt before any cloud launcher may consume the packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "launch"))

from launch_guardrails import (  # noqa: E402
    DEFAULT_RESULTS_BUCKET,
    EXECUTION_ACCOUNT,
    ROLE_ARN,
    STUDY_OWNER,
    prepared_source_bundle,
    source_tree_sha256,
)

from robomme_integration.fleet.task_inventory import (  # noqa: E402
    CANONICAL_PARENT_SHA256,
    CANONICAL_TASK_DERIVED_SHA256,
)
from robomme_integration.launch import IMAGE, OPENPI, OPENPI_SHA  # noqa: E402
from robomme_integration.training.single_task import task_manifest_sha256  # noqa: E402
from robomme_integration.training.workspace_deliberative_dense_v2 import PROTOCOL  # noqa: E402
from robomme_integration.training.workspace_gpu_producer_dense_v2 import CAMPAIGN, TASK  # noqa: E402
from robomme_integration.training.workspace_supervision_dense_v2 import (  # noqa: E402
    ARTIFACT,
    TARGET_SEMANTICS,
)

ENTRY = "gpu_move_workspace_dense_v2_entry.sh"
STAGED_MANIFEST = "_robomme_move_workspace_dense_v2_manifest.json"
OWNER = STUDY_OWNER
STUDY = "long_context_v1"
STUDY_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/studies/{STUDY}"
DATA_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/datasets/robomme/v1/lerobot_all16"
DATA_INVENTORY = f"{STUDY_ROOT}/manifests/inventories/data/{CANONICAL_PARENT_SHA256}.json"
ARTIFACT_ROOT = f"{STUDY_ROOT}/artifacts/robomme/workspace_dense_multipoint_visreg_v2"


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def source_identity(source_dir: Path) -> str:
    required = [
        source_dir / ENTRY,
        source_dir / "training/workspace_supervision_dense_v2.py",
        source_dir / "training/workspace_deliberative_dense_v2.py",
        source_dir / "training/workspace_materialize_dense_v2.py",
        source_dir / "training/workspace_gpu_producer_dense_v2.py",
    ]
    if not all(path.is_file() for path in required):
        raise ValueError(f"dense v2 isolated source is incomplete: {required}")
    with prepared_source_bundle(source_dir, ENTRY, {}, None) as (staged, _entry, _environment):
        return source_tree_sha256(staged)


def build_plan(source_dir: Path, *, attempt_index: int = 1) -> dict:
    if attempt_index < 1:
        raise ValueError("attempt_index must be positive")
    source_sha = source_identity(source_dir)
    scientific = {
        "schema_version": 2,
        "benchmark": "RoboMME",
        "protocol": PROTOCOL,
        "supervision": {
            "artifact": ARTIFACT,
            "target_semantics": TARGET_SEMANTICS,
            "accepts_all_grounded_points": True,
            "role_order": "left_to_right_coordinate_occurrence_in_grounded_text",
            "grounding_in_encoder_inputs": False,
        },
        "representation": {
            "steps": 10_000,
            "batch_size": 64,
            "devices": 8,
            "seed": 0,
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "warmup_steps": 500,
            "global_gradient_clip": 10.0,
            "ema_decay": 0.999,
            "min_lag": 40,
            "future_delta": 20,
            "history_stride": 10,
            "max_history": 128,
            "history_mask_probability": 0.2,
            "omega_dim": 512,
            "loss_weights": {
                "dense_attention": 0.1,
                "occupancy": 0.1,
                "jepa": 0.1,
                "sigreg": 0.0,
                "visreg": 0.05,
            },
            "visreg": {"slices": 128, "scale": 1.0, "shape": 1.0, "center": 1.0},
        },
        "upstream": {
            "repo_id": "Yinpei/robomme_preprocessed_data",
            "revision": "ddf0baf55b633cc6657dcd53ac0e089a273de612",
        },
    }
    scientific_sha = _digest(scientific)
    run_identity_sha = _digest(
        {
            "scientific_spec_sha256": scientific_sha,
            "source_tree_sha256": source_sha,
        }
    )
    run_id = f"move-wrep-v2-visreg-seed0-{run_identity_sha[:16]}"
    attempt_id = f"{run_id}-attempt{attempt_index}"
    manifest_s3 = f"{STUDY_ROOT}/manifests/runs/workspace_dense_v2/{run_id}/{attempt_id}.json"
    producer_claim = f"{STUDY_ROOT}/manifests/claims/workspace_dense_v2/{run_id}/producers/{attempt_id}.json"
    completion_claim = f"{STUDY_ROOT}/manifests/claims/workspace_dense_v2/{run_id}.complete.json"
    manifest = {
        "schema_version": 2,
        "kind": "robomme_move_workspace_dense_v2_attempt",
        "identity": {
            "campaign": CAMPAIGN,
            "task": TASK,
            "task_manifest_sha256": task_manifest_sha256(TASK),
            "task_inventory_sha256": CANONICAL_TASK_DERIVED_SHA256[TASK],
            "scientific_spec_sha256": scientific_sha,
            "run_id": run_id,
            "attempt_id": attempt_id,
        },
        "scientific": scientific,
        "source": {
            "source_tree_sha256": source_sha,
            "entry": ENTRY,
            "entry_sha256": hashlib.sha256((source_dir / ENTRY).read_bytes()).hexdigest(),
            "openpi": {"uri": OPENPI, "sha256": OPENPI_SHA},
            "image": IMAGE,
        },
        "infrastructure": {
            "provider": "aws_sagemaker",
            "account": EXECUTION_ACCOUNT,
            "role": ROLE_ARN,
            "queue": "fss-tri-cam-robotics-p5-48xlarge-us-west-2",
            "training_plan_arn": None,
            "instance_type": "ml.p5.48xlarge",
            "accelerator": "8xH100-80GB-HBM3",
            "priority": 400,
            "max_run_seconds": 86_400,
            "volume_size_gb": 400,
        },
        "claims": {"producer": producer_claim, "completion": completion_claim},
        "manifest_s3": manifest_s3,
        "submission_gate": {
            "state": "blocked_pending_independent_review_and_gpu_canary",
            "required_receipts": [
                "independent_source_protocol_review",
                "task_bound_8xH100_gpu_canary",
            ],
            "this_packet_authorizes_submission": False,
        },
    }
    manifest_json = _canonical(manifest) + "\n"
    manifest_sha = hashlib.sha256(manifest_json.encode()).hexdigest()
    environment = {
        "ROBOMME_MOVE_DENSE_V2_RUN_ID": run_id,
        "ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256": source_sha,
        "ROBOMME_MOVE_DENSE_V2_MANIFEST_SOURCE": STAGED_MANIFEST,
        "ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256": manifest_sha,
        "ROBOMME_MOVE_DENSE_V2_MANIFEST_S3": manifest_s3,
        "ROBOMME_MOVE_DENSE_V2_PRODUCER_CLAIM_S3": producer_claim,
        "ROBOMME_MOVE_DENSE_V2_COMPLETION_CLAIM_S3": completion_claim,
        "ROBOMME_MOVE_DENSE_V2_ARTIFACT_ROOT_S3": ARTIFACT_ROOT,
        "OPENPI_FORK_S3": OPENPI,
        "ROBOMME_DATA_S3": DATA_ROOT,
        "ROBOMME_DATA_PARENT_INVENTORY_S3": DATA_INVENTORY,
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": CANONICAL_PARENT_SHA256,
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    if any(len(value.encode()) > 512 for value in environment.values()):
        raise ValueError("SageMaker environment value exceeds 512 bytes")
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "source_tree_sha256": source_sha,
        "scientific_spec_sha256": scientific_sha,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "manifest_sha256": manifest_sha,
        "environment": environment,
        "staged_manifest": STAGED_MANIFEST,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--attempt-index", type=int, default=1)
    args = parser.parse_args()
    plan = build_plan(Path(args.source_dir).resolve(), attempt_index=args.attempt_index)
    print(plan["manifest_json"], end="")
    print(f"REVIEW ONLY source={plan['source_tree_sha256']} manifest={plan['manifest_sha256']} submission=false")


if __name__ == "__main__":
    main()
