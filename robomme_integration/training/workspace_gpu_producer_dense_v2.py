"""Produce one immutable MoveCube dense/multi-point VISReg-v2 workspace artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from .single_task import task_manifest_sha256
from .workspace_deliberative_dense_v2 import PROTOCOL
from .workspace_gpu_producer import _compute_python, _put_once, _run, _sync_tree
from .workspace_supervision_dense_v2 import ARTIFACT, TARGET_SEMANTICS

CAMPAIGN = "move_workspace_dense_multipoint_visreg_v2"
TASK = "MoveCube"
ENTRY = "gpu_move_workspace_dense_v2_entry.sh"
ACCOUNT = "141701954645"
ROLE = "arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2"
QUEUE = "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
IMAGE = (
    "141701954645.dkr.ecr.us-west-2.amazonaws.com/"
    "sarvesh.patil-groot-dexjoco@sha256:798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"
)
OPENPI_SHA = "ed923b2c27d2f608d62cc4b5ca89d5b80c14739dba1ab81d6f53d8013bcb66ad"
STUDY_ROOT = "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"
OPENPI = f"{STUDY_ROOT}/code/openpi/{OPENPI_SHA}.tgz"
PARENT_INVENTORY_SHA = "e77968b4c72c7589d92c1e85b1c6f7bf81aa49dd74472fb88dcead4277b5dad2"
TASK_INVENTORY_SHA = "0fbfc217fd24aff9e2f1020d092ce57d4ac20980d9f11335066c6f035066d509"
DATA_ROOT = "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/datasets/robomme/v1/lerobot_all16"
DATA_INVENTORY = f"{STUDY_ROOT}/manifests/inventories/data/{PARENT_INVENTORY_SHA}.json"
ARTIFACT_ROOT = f"{STUDY_ROOT}/artifacts/robomme/workspace_dense_multipoint_visreg_v2"
HEX64 = frozenset("0123456789abcdef")


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX64 for char in value):
        raise ValueError(f"{label} must be lowercase hex64")
    return value


def validate_manifest(
    manifest: dict,
    *,
    manifest_sha256: str,
    expected_source_tree_sha256: str,
    expected_entry_sha256: str,
) -> dict:
    actual = hashlib.sha256(_canonical(manifest)).hexdigest()
    if actual != manifest_sha256:
        raise ValueError("dense v2 producer manifest SHA-256 mismatch")
    scientific = manifest.get("scientific", {})
    expected_scientific = {
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
    if scientific != expected_scientific:
        raise ValueError("dense v2 scientific contract drifted")
    scientific_sha = hashlib.sha256(_canonical(scientific)[:-1]).hexdigest()
    source = manifest.get("source", {})
    source_sha = _hex64(source.get("source_tree_sha256"), "source tree")
    if source_sha != _hex64(expected_source_tree_sha256, "expected source tree"):
        raise ValueError("dense v2 source identity differs from trusted expected source")
    entry_sha = _hex64(source.get("entry_sha256"), "entry SHA-256")
    if entry_sha != _hex64(expected_entry_sha256, "expected entry SHA-256"):
        raise ValueError("dense v2 entry SHA differs from trusted expected entry")
    if source != {
        "source_tree_sha256": source_sha,
        "entry": ENTRY,
        "entry_sha256": entry_sha,
        "openpi": {"uri": OPENPI, "sha256": OPENPI_SHA},
        "image": IMAGE,
    }:
        raise ValueError("dense v2 source contract drifted")
    run_identity_sha = hashlib.sha256(
        _canonical(
            {
                "scientific_spec_sha256": scientific_sha,
                "source_tree_sha256": source_sha,
            }
        )[:-1]
    ).hexdigest()
    identity = manifest.get("identity", {})
    run_id = f"move-wrep-v2-visreg-seed0-{run_identity_sha[:16]}"
    attempt_id = identity.get("attempt_id")
    match = re.fullmatch(re.escape(run_id) + r"-attempt(?P<index>[1-9][0-9]*)", str(attempt_id))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "robomme_move_workspace_dense_v2_attempt"
        or identity.get("campaign") != CAMPAIGN
        or identity.get("task") != TASK
        or identity.get("task_manifest_sha256") != task_manifest_sha256(TASK)
        or identity.get("task_inventory_sha256") != TASK_INVENTORY_SHA
        or identity.get("scientific_spec_sha256") != scientific_sha
        or identity.get("run_id") != run_id
        or match is None
    ):
        raise ValueError("dense v2 producer identity drifted")
    if manifest.get("infrastructure") != {
        "provider": "aws_sagemaker",
        "account": ACCOUNT,
        "role": ROLE,
        "queue": QUEUE,
        "training_plan_arn": None,
        "instance_type": "ml.p5.48xlarge",
        "accelerator": "8xH100-80GB-HBM3",
        "priority": 400,
        "max_run_seconds": 86_400,
        "volume_size_gb": 400,
    }:
        raise ValueError("dense v2 infrastructure contract drifted")
    producer_claim = f"{STUDY_ROOT}/manifests/claims/workspace_dense_v2/{run_id}/producers/{attempt_id}.json"
    completion_claim = f"{STUDY_ROOT}/manifests/claims/workspace_dense_v2/{run_id}.complete.json"
    manifest_s3 = f"{STUDY_ROOT}/manifests/runs/workspace_dense_v2/{run_id}/{attempt_id}.json"
    if manifest.get("claims") != {"producer": producer_claim, "completion": completion_claim}:
        raise ValueError("dense v2 claim namespaces drifted")
    if manifest.get("manifest_s3") != manifest_s3:
        raise ValueError("dense v2 manifest namespace drifted")
    if manifest.get("submission_gate") != {
        "state": "blocked_pending_independent_review_and_gpu_canary",
        "required_receipts": [
            "independent_source_protocol_review",
            "task_bound_8xH100_gpu_canary",
        ],
        "this_packet_authorizes_submission": False,
    }:
        raise ValueError("dense v2 submission gate drifted")
    return manifest


def expected_runtime_environment(
    manifest: dict,
    *,
    manifest_sha256: str,
    expected_entry_sha256: str,
) -> dict[str, str]:
    validate_manifest(
        manifest,
        manifest_sha256=manifest_sha256,
        expected_source_tree_sha256=manifest["source"]["source_tree_sha256"],
        expected_entry_sha256=expected_entry_sha256,
    )
    identity, claims = manifest["identity"], manifest["claims"]
    return {
        "ROBOMME_MOVE_DENSE_V2_RUN_ID": identity["run_id"],
        "ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256": manifest["source"]["source_tree_sha256"],
        "ROBOMME_MOVE_DENSE_V2_MANIFEST_SOURCE": "_robomme_move_workspace_dense_v2_manifest.json",
        "ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256": manifest_sha256,
        "ROBOMME_MOVE_DENSE_V2_MANIFEST_S3": manifest["manifest_s3"],
        "ROBOMME_MOVE_DENSE_V2_PRODUCER_CLAIM_S3": claims["producer"],
        "ROBOMME_MOVE_DENSE_V2_COMPLETION_CLAIM_S3": claims["completion"],
        "ROBOMME_MOVE_DENSE_V2_ARTIFACT_ROOT_S3": ARTIFACT_ROOT,
        "OPENPI_FORK_S3": OPENPI,
        "ROBOMME_DATA_S3": DATA_ROOT,
        "ROBOMME_DATA_PARENT_INVENTORY_S3": DATA_INVENTORY,
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": PARENT_INVENTORY_SHA,
        "SM_USE_RESERVED_CAPACITY": "1",
    }


def validate_runtime_environment(
    manifest: dict,
    *,
    manifest_sha256: str,
    environment: dict[str, str],
    expected_entry_sha256: str | None = None,
) -> None:
    if expected_entry_sha256 is None:
        expected_entry_sha256 = hashlib.sha256((Path(__file__).resolve().parents[1] / ENTRY).read_bytes()).hexdigest()
    validate_manifest(
        manifest,
        manifest_sha256=manifest_sha256,
        expected_source_tree_sha256=environment.get("ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256", ""),
        expected_entry_sha256=expected_entry_sha256,
    )
    expected = expected_runtime_environment(
        manifest,
        manifest_sha256=manifest_sha256,
        expected_entry_sha256=expected_entry_sha256,
    )
    drift = {key: (environment.get(key), value) for key, value in expected.items() if environment.get(key) != value}
    if drift:
        raise ValueError(f"dense v2 runtime environment drifted: {drift}")


def _verified_local_outputs(task_root: Path, manifest: dict) -> dict:
    supervision = json.loads((task_root / "supervision" / TASK / "MANIFEST.json").read_text(encoding="utf-8"))
    run_config = json.loads((task_root / "representation" / TASK / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    omega = json.loads((task_root / "omega" / TASK / "MANIFEST.json").read_text(encoding="utf-8"))
    if (
        supervision.get("artifact") != ARTIFACT
        or supervision.get("target_semantics", {}).get("name") != TARGET_SEMANTICS
        or run_config.get("protocol") != PROTOCOL
        or omega.get("protocol") != PROTOCOL
    ):
        raise RuntimeError("dense v2 local artifact protocol chain drifted")
    weights = run_config.get("loss_weights", {})
    if (
        weights.get("attention") != 0.1
        or weights.get("occ") != 0.1
        or weights.get("jepa") != 0.1
        or weights.get("sigreg") != 0.0
        or weights.get("visreg") != 0.05
        or weights.get("visreg_slices") != 128
        or run_config.get("ema_decay") != 0.999
    ):
        raise RuntimeError("dense v2 local artifact is not VISReg-only")
    if run_config.get("supervision_manifest_sha256") != supervision.get("manifest_sha256"):
        raise RuntimeError("dense v2 trainer/supervision provenance mismatch")
    if (
        omega.get("encoder_identity", {}).get("run_config_sha256")
        != hashlib.sha256((task_root / "representation" / TASK / "RUN_CONFIG.json").read_bytes()).hexdigest()
    ):
        raise RuntimeError("dense v2 omega/trainer provenance mismatch")
    if omega.get("encoder_identity", {}).get("parameter_source") != "ema":
        raise RuntimeError("dense v2 omega was not materialized from EMA parameters")
    return {"supervision": supervision, "run_config": run_config, "omega": omega}


def produce(args: argparse.Namespace) -> dict:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        manifest_sha256=args.manifest_sha256,
        expected_source_tree_sha256=os.environ.get("ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256", ""),
        expected_entry_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / ENTRY).read_bytes()).hexdigest(),
    )
    validate_runtime_environment(
        manifest,
        manifest_sha256=args.manifest_sha256,
        environment=os.environ,
    )
    if manifest["identity"]["run_id"] != args.run_id:
        raise ValueError("dense v2 run identity mismatch")
    task_root = Path(args.work_root) / TASK
    data = task_root / "data"
    upstream = task_root / "upstream"
    supervision = task_root / "supervision"
    representation = task_root / "representation"
    omega = task_root / "omega"
    task_root.mkdir(parents=True, exist_ok=True)
    control_python = sys.executable
    compute_python = _compute_python()
    _run(
        [
            control_python,
            "-m",
            "fleet.task_inventory",
            "--parent-manifest",
            args.parent_manifest,
            "--task",
            TASK,
            "--root-s3",
            args.data_s3,
            "--destination",
            str(data),
            "--expected-derived-sha256",
            manifest["identity"]["task_inventory_sha256"],
            "--workers",
            "32",
        ]
    )
    _run(
        [
            compute_python,
            "-m",
            "training.upstream_feature_cache",
            "--task",
            TASK,
            "--lerobot-root",
            str(data),
            "--output-root",
            str(upstream),
            "--download-workers",
            "8",
        ]
    )
    _run(
        [
            compute_python,
            "-m",
            "training.workspace_supervision_dense_v2",
            "--task",
            TASK,
            "--lerobot-root",
            str(data),
            "--upstream-cache-root",
            str(upstream),
            "--output-root",
            str(supervision),
        ]
    )
    environment = os.environ.copy()
    environment["WSM_MOVE_DENSE_V2_REP_ALLOW_RUN"] = "1"
    environment["WSM_MOVE_DENSE_V2_MATERIALIZE_ALLOW_RUN"] = "1"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    _run(
        [
            compute_python,
            "-m",
            "training.workspace_deliberative_dense_v2",
            "--task",
            TASK,
            "--supervision-root",
            str(supervision),
            "--output-root",
            str(representation),
            "--steps",
            "10000",
            "--batch-size",
            "64",
            "--expected-devices",
            "8",
        ],
        environment=environment,
    )
    _run(
        [
            compute_python,
            "-m",
            "training.workspace_materialize_dense_v2",
            "--task",
            TASK,
            "--supervision-root",
            str(supervision),
            "--train-root",
            str(representation),
            "--output-root",
            str(omega),
            "--batch-size",
            "64",
            "--expected-devices",
            "8",
        ],
        environment=environment,
    )
    outputs = _verified_local_outputs(task_root, manifest)
    best = json.loads((representation / TASK / "BEST.json").read_text(encoding="utf-8"))
    step = int(best["best_step"])
    completion = json.loads(
        (representation / TASK / "checkpoints" / str(step) / "WSM_GENERATION_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    if completion.get("parameter_source") != "ema" or completion.get("ema_decay") != 0.999:
        raise RuntimeError("dense v2 selected checkpoint is not EMA-bound")
    encoder_id = outputs["omega"]["encoder_id"]
    artifact_root = f"{args.artifact_root_s3}/{TASK}/{encoder_id}"
    uris = {
        "omega": f"{artifact_root}/omega",
        "supervision": f"{artifact_root}/supervision",
        "representation": f"{artifact_root}/representation/step-{step}",
    }
    seals = {
        "omega": _sync_tree(omega / TASK, uris["omega"], seal="MANIFEST.json"),
        "supervision": _sync_tree(supervision / TASK, uris["supervision"], seal="MANIFEST.json"),
        "representation": _sync_tree(
            representation / TASK / "checkpoints" / str(step),
            uris["representation"],
            seal="WSM_GENERATION_COMPLETE.json",
        ),
    }
    claim = {
        "schema_version": 2,
        "kind": "robomme_move_workspace_dense_v2_complete",
        "campaign": CAMPAIGN,
        "task": TASK,
        "run_id": args.run_id,
        "scientific_spec_sha256": manifest["identity"]["scientific_spec_sha256"],
        "source_tree_sha256": manifest["source"]["source_tree_sha256"],
        "task_manifest_sha256": task_manifest_sha256(TASK),
        "protocol": PROTOCOL,
        "supervision_artifact": ARTIFACT,
        "target_semantics": TARGET_SEMANTICS,
        "regularizer": {"name": "visreg", "weight": 0.05, "sigreg_weight": 0.0},
        "ema_decay": 0.999,
        "encoder_id": encoder_id,
        "omega": {"uri": uris["omega"], "manifest_sha256": seals["omega"]},
        "supervision": {
            "uri": uris["supervision"],
            "manifest_sha256": seals["supervision"],
        },
        "representation": {
            "uri": uris["representation"],
            "step": step,
            "completion_sha256": seals["representation"],
        },
    }
    _put_once(_canonical(claim), args.claim_s3)
    shutil.rmtree(task_root)
    print(f"MOVE WORKSPACE DENSE V2 COMPLETE run_id={args.run_id} encoder={encoder_id}", flush=True)
    return claim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--claim-s3", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--data-s3", required=True)
    parser.add_argument("--artifact-root-s3", required=True)
    parser.add_argument("--work-root", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
