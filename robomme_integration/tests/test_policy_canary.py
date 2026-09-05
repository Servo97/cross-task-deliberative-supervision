from __future__ import annotations

import hashlib
import json
import tarfile
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robomme_integration import launch
from robomme_integration import policy_canary_launch as canary_launch
from robomme_integration.training import policy_canary


def _sealed(value: dict) -> dict:
    value = dict(value)
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return value


def _reference_manifest() -> dict:
    scientific = {
        "schema_version": 2,
        "study": launch.STUDY,
        "benchmark": "RoboMME",
        "scope": "single_task_v1",
        "task": {
            "name": "PickXtimes",
            "episodes": list(range(500, 600)),
            "task_manifest_sha256": ("a35942200731c1e4a33d526335adcca1e844e8bace055d69eb0e66aeca3de761"),
        },
        "arm": canary_launch.ARM,
        "mechanism": launch._arm_spec(canary_launch.ARM),
        "initialization": {
            "checkpoint_s3": launch.INIT_ROOT,
            "inventory_uri": launch.INIT_INVENTORY,
            "inventory_sha256": launch.INIT_INVENTORY_SHA,
            "recipe": "RoboCasa H300+MG balanced",
        },
        "data": {
            "action_dim": 8,
            "dataset_s3": launch.DATA_ROOT,
            "download_policy": "shared_metadata_plus_exact_100_task_parquets",
            "parent_inventory_uri": launch.DATA_INVENTORY,
            "parent_inventory_sha256": launch.CANONICAL_PARENT_SHA256,
            "derived_task_inventory_sha256": launch.CANONICAL_TASK_DERIVED_SHA256["PickXtimes"],
            "repo_id": "Yinpei/robomme_data_lerobot",
            "revision": "1510653cccb4d9e5165fb3141c06d88053decc20",
            "state_dim": 8,
            "views": ["base_0_rgb", "left_wrist_0_rgb"],
        },
        "training": {
            "steps": 20_000,
            "seed": 0,
            "batch_size": 64,
            "batch_unit": "steps",
            "effective_per_step_batch": 64,
            "action_horizon": 20,
            "full_finetune": True,
            "optimizer": "AdamW",
            "peak_lr": 5e-5,
            "decay_lr": 5e-6,
            "ema_decay": 0.99,
            "jax_devices": 8,
            "jax_processes": 1,
            "fsdp_devices": 1,
            "checkpoint_policy": {
                "remote_resume": True,
                "save_interval": 5000,
                "success_retention": [19999],
                "training_retention": "newest_plus_step10000",
                "upload_only_finalized_orbax": True,
            },
            "chunk_stride": None,
            "decay_steps": 20_000,
            "num_workers": 32,
            "warmup_steps": 1000,
            "window_len": None,
        },
        "workspace_representation": {
            "encoder_id": "dd5a17e4929537f9b0472f374081618a5dff5deed31af1969324ded291bd9c15",
            "omega": {
                "uri": (
                    f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/PickXtimes/"
                    "dd5a17e4929537f9b0472f374081618a5dff5deed31af1969324ded291bd9c15/omega"
                ),
                "manifest_sha256": ("4ea7ca3f9759c4cc435a7c6e754f6a3227ed9f043df780eb851e37a3989a5775"),
            },
            "supervision": None,
            "task_bound": True,
            "omega_symbol": "omega_t",
        },
        "sources": {
            "robomme_integration": {
                "sanitized_source_tree_sha256": canary_launch.PRODUCTION_SOURCE_SHA256,
                "entry": "gpu_train_entry.sh",
                "entry_sha256": canary_launch.PRODUCTION_ENTRY_SHA256,
            },
            "openpi": {"uri": launch.OPENPI, "sha256": launch.OPENPI_SHA},
            "tokenizer": {"uri": launch.TOKENIZER, "sha256": launch.TOKENIZER_SHA},
            "image": {"uri": launch.IMAGE, "sha256": launch.IMAGE_SHA},
        },
    }
    assert canary_launch._sha256_json(scientific) == canary_launch.REFERENCE_SCIENTIFIC_SPEC_SHA256
    return {
        "schema_version": 2,
        "kind": "robomme_gpu_training_attempt",
        "run_id": canary_launch.REFERENCE_RUN_ID,
        "manifest_s3": canary_launch.REFERENCE_MANIFEST_S3,
        "manifest_sha256": canary_launch.REFERENCE_MANIFEST_SHA256,
        "scientific_spec_sha256": canary_launch.REFERENCE_SCIENTIFIC_SPEC_SHA256,
        "scientific": scientific,
    }


def _canary_manifest() -> dict:
    reference = _reference_manifest()
    identity = {
        "canary_source_tree_sha256": "a" * 64,
        "canary_runtime_source_tree_sha256": "c" * 64,
        "reference_run_id": canary_launch.REFERENCE_RUN_ID,
        "reference_scientific_spec_sha256": canary_launch.REFERENCE_SCIENTIFIC_SPEC_SHA256,
        "reference_manifest_sha256": canary_launch.REFERENCE_MANIFEST_SHA256,
        "production_source_tree_sha256": canary_launch.PRODUCTION_SOURCE_SHA256,
        "task": reference["scientific"]["task"],
        "arm": canary_launch.ARM,
        "mechanism": reference["scientific"]["mechanism"],
        "data": reference["scientific"]["data"],
        "initialization": reference["scientific"]["initialization"],
        "workspace_representation": reference["scientific"]["workspace_representation"],
        "sources": reference["scientific"]["sources"],
        "model_training_identity": {
            name: reference["scientific"]["training"][name]
            for name in (
                "seed",
                "batch_size",
                "batch_unit",
                "effective_per_step_batch",
                "action_horizon",
                "full_finetune",
                "optimizer",
                "peak_lr",
                "ema_decay",
                "fsdp_devices",
                "jax_devices",
                "jax_processes",
            )
        },
    }
    identity_sha = canary_launch._sha256_json(identity)
    canary_id = f"policy-canary-v1-pickxtimes-{canary_launch.ARM}-{identity_sha[:16]}"
    namespace = f"{launch.STUDY_ROOT}/manifests/canaries/policy_training/{canary_id}"
    job_name = f"sarvesh-rmme-policy-canary-{identity_sha[:20]}"
    return _sealed(
        {
            "schema_version": 1,
            "kind": canary_launch.KIND,
            "claim": canary_launch.CLAIM,
            "contract_version": canary_launch.CONTRACT_VERSION,
            "canary_id": canary_id,
            "identity_sha256": identity_sha,
            "identity": identity,
            "canary_execution": {
                "optimizer_steps": 2,
                "final_local_checkpoint_step": 1,
                "seed": 0,
                "batch_size": 64,
                "effective_per_step_batch": 64,
                "warmup_steps": 1,
                "decay_steps": 2,
                "peak_lr": 5e-5,
                "decay_lr": 5e-6,
                "diagnostics": [
                    "total_loss",
                    "action_loss",
                    "jepa_loss",
                    "sigreg_loss",
                    "gradient_norm",
                    "parameter_update_norm",
                    "parameters_finite",
                ],
                "checkpoint_scope": "node_local_ephemeral_only",
                "restore_smoke": "mandatory_local_save_and_restore",
            },
            "source": {
                "canary_source_tree_sha256": "a" * 64,
                "canary_runtime_source_tree_sha256": "c" * 64,
                "entry": canary_launch.ENTRY,
                "entry_sha256": canary_launch.CANARY_ENTRY_SHA256,
                "submitted_entry_mode": canary_launch.SUBMITTED_ENTRY_MODE,
                "sagemaker_runtime_entry_mode": canary_launch.SAGEMAKER_RUNTIME_ENTRY_MODE,
                "baseline_source_tree_sha256": canary_launch.PRODUCTION_SOURCE_SHA256,
                "production_entry_sha256": canary_launch.PRODUCTION_ENTRY_SHA256,
                "reviewed_delta_paths": sorted(canary_launch.ALLOWED_SOURCE_DELTAS),
            },
            "infrastructure": {
                "provider": "aws_sagemaker",
                "execution_account": launch.EXECUTION_ACCOUNT,
                "queue": launch.TRAINING_PLAN_QUEUE,
                "training_plan_arn": launch.training_plan_arn(launch.TRAINING_PLAN_QUEUE),
                "role": launch.ROLE_ARN,
                "instance_type": "ml.p5e.48xlarge",
                "accelerator": "8xH200",
                "priority": 400,
                "max_run_seconds": 10_800,
                "volume_size_gb": 250,
                "attempts_in_job": 1,
                "deterministic_job_name": job_name,
            },
            "publication": {
                "namespace_s3": namespace,
                "receipt_s3": f"{namespace}/training_canary.complete.json",
                "create_once": True,
                "only_allowed_object": "training_canary.complete.json",
                "production_checkpoint_or_deploy_publication": False,
            },
        }
    )


def _environment(manifest: dict) -> dict[str, str]:
    identity = manifest["identity"]
    return {
        "ROBOMME_CANARY_KIND": canary_launch.KIND,
        "ROBOMME_CANARY_CLAIM": canary_launch.CLAIM,
        "ROBOMME_CANARY_ID": manifest["canary_id"],
        "ROBOMME_CANARY_STEPS": "2",
        "ROBOMME_CANARY_MANIFEST_SHA256": manifest["manifest_sha256"],
        "ROBOMME_CANARY_NAMESPACE_S3": manifest["publication"]["namespace_s3"],
        "ROBOMME_CANARY_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "ROBOMME_REFERENCE_RUN_ID": identity["reference_run_id"],
        "ROBOMME_REFERENCE_SCIENTIFIC_SPEC_SHA256": identity["reference_scientific_spec_sha256"],
        "ROBOMME_REFERENCE_SOURCE_SHA256": identity["production_source_tree_sha256"],
        "ROBOMME_ARM": identity["arm"],
        "ROBOMME_TASK": identity["task"]["name"],
        "ROBOMME_SCOPE": "single_task_canary",
        "ROBOMME_DATA_S3": identity["data"]["dataset_s3"],
        "ROBOMME_DATA_PARENT_INVENTORY_S3": identity["data"]["parent_inventory_uri"],
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": identity["data"]["parent_inventory_sha256"],
        "ROBOMME_DATA_DERIVED_INVENTORY_SHA256": identity["data"]["derived_task_inventory_sha256"],
        "INIT_S3": identity["initialization"]["checkpoint_s3"],
        "INIT_INVENTORY_S3": identity["initialization"]["inventory_uri"],
        "INIT_INVENTORY_SHA256": identity["initialization"]["inventory_sha256"],
        "OPENPI_FORK_S3": identity["sources"]["openpi"]["uri"],
        "PALIGEMMA_TOKENIZER_S3": identity["sources"]["tokenizer"]["uri"],
        "PALIGEMMA_TOKENIZER_SHA256": identity["sources"]["tokenizer"]["sha256"],
        "ROBOMME_WORKSPACE_S3": identity["workspace_representation"]["omega"]["uri"],
        "ROBOMME_WORKSPACE_MANIFEST_SHA256": identity["workspace_representation"]["omega"]["manifest_sha256"],
        "WSM_MAX_STEPS": "2",
        "WSM_SAVE_INTERVAL": "2",
        "WSM_WARMUP_STEPS": "1",
        "WSM_DECAY_STEPS": "2",
        "WSM_SEED": "0",
    }


def _records() -> list[dict]:
    return [
        {
            "step": step,
            "total_loss": 1.205,
            "diagnostic_total_loss": 1.205,
            "production_loss_parity_error": 0.0,
            "action_loss": 1.0,
            "jepa_loss": 2.0,
            "jepa_weighted": 0.2,
            "sigreg_loss": 0.1,
            "sigreg_weighted": 0.005,
            "valid_target_count": 60.0,
            "grad_norm": 1.0,
            "gdn_grad_norm": 0.2,
            "jepa_head_grad_norm": 0.3,
            "update_norm": 0.01,
            "parameter_delta_norm": 0.01,
            "parameters_finite": True,
            "optimizer_finite": True,
        }
        for step in range(2)
    ]


def _test_flatten_with_path(tree):
    return [((f"leaf-{index}",), leaf) for index, leaf in enumerate(tree)], (
        "test-list",
        len(tree),
    )


def _state_fingerprints(values: tuple[int, int, int] = (1, 2, 3)) -> dict:
    state = SimpleNamespace(
        params=[np.asarray([values[0]], dtype=np.float32)],
        ema_params=[np.asarray([values[1]], dtype=np.float32)],
        opt_state=[np.asarray([values[2]], dtype=np.int32)],
    )
    return policy_canary.fingerprint_train_state_components(
        state,
        _flatten_with_path=_test_flatten_with_path,
        _device_get=lambda leaf: leaf,
    )


def _restore_fingerprint_proof() -> dict:
    saved = _state_fingerprints()
    restored = json.loads(json.dumps(saved))
    return policy_canary.build_restore_fingerprint_proof(saved, restored)


def test_reference_manifest_pins_exact_frozen_gdn_jepa_identity():
    reference = _reference_manifest()
    scientific = canary_launch.validate_reference_manifest(reference, task="PickXtimes", arm=canary_launch.ARM)
    assert scientific["training"]["effective_per_step_batch"] == 64
    assert scientific["training"]["jax_devices"] == 8
    assert scientific["mechanism"]["steering"] == "gated_deltanet_k8"
    assert scientific["mechanism"]["jepa"] == {
        "lambda": 0.1,
        "futures": 1,
        "sigreg": 0.05,
    }
    assert (
        scientific["sources"]["robomme_integration"]["sanitized_source_tree_sha256"]
        == canary_launch.PRODUCTION_SOURCE_SHA256
    )
    wrong_data = json.loads(json.dumps(reference))
    wrong_data["scientific"]["data"]["dataset_s3"] = "s3://arbitrary/wrong-data"
    wrong_data["scientific_spec_sha256"] = canary_launch._sha256_json(wrong_data["scientific"])
    with pytest.raises(SystemExit, match="exact frozen Wave2 cell"):
        canary_launch.validate_reference_manifest(wrong_data, task="PickXtimes", arm=canary_launch.ARM)


def test_standalone_manifest_contract_rejects_resealed_semantic_drift():
    mutations = (
        lambda value: value["identity"]["data"].update({"dataset_s3": "s3://arbitrary/wrong-data"}),
        lambda value: value.update({"canary_id": "policy-canary-v1-wrong"}),
        lambda value: value["source"].update({"entry_sha256": "f" * 64}),
        lambda value: value["source"].update({"sagemaker_runtime_entry_mode": 0o755}),
        lambda value: value["canary_execution"].update({"optimizer_steps": 1}),
        lambda value: value["infrastructure"].update({"accelerator": "8xH100"}),
    )
    for mutate in mutations:
        manifest = json.loads(json.dumps(_canary_manifest()))
        mutate(manifest)
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = policy_canary._sha256_json(manifest)
        with pytest.raises(ValueError):
            policy_canary.validate_manifest_contract(manifest)


def test_manifest_environment_is_canary_only_and_fails_on_production_leak():
    manifest = _canary_manifest()
    environment = _environment(manifest)
    policy_canary.validate_manifest_environment(manifest, environment)
    environment["OUTPUT_S3"] = f"{launch.STUDY_ROOT}/checkpoints/robomme/pi05/bad"
    with pytest.raises(ValueError, match="production output flags leaked"):
        policy_canary.validate_manifest_environment(manifest, environment)


def test_node_gate_requires_exactly_eight_h200s_in_one_jax_process():
    policy_canary.validate_node_topology(
        device_count=8,
        process_count=1,
        gpu_names=["NVIDIA H200" for _ in range(8)],
    )
    with pytest.raises(ValueError, match="8 H200"):
        policy_canary.validate_node_topology(
            device_count=8,
            process_count=1,
            gpu_names=["NVIDIA H100" for _ in range(8)],
        )
    with pytest.raises(ValueError, match="eight devices"):
        policy_canary.validate_node_topology(
            device_count=4,
            process_count=1,
            gpu_names=["NVIDIA H200" for _ in range(4)],
        )


def test_cloud_gate_requires_available_one_empty_namespace_and_no_duplicate():
    plan = {
        "TrainingPlanArn": launch.training_plan_arn(launch.TRAINING_PLAN_QUEUE),
        "Status": "Active",
        "TotalInstanceCount": 2,
        "InUseInstanceCount": 1,
        "AvailableInstanceCount": 1,
        "UnhealthyInstanceCount": 0,
    }
    canary_launch.validate_cloud_snapshot(
        account=launch.EXECUTION_ACCOUNT,
        training_plan=plan,
        namespace_objects=[],
        active_batch_jobs=[{"jobName": "wave2"}],
        active_training_jobs=[],
        job_name="policy-canary",
    )
    with pytest.raises(SystemExit, match="namespace is not empty"):
        canary_launch.validate_cloud_snapshot(
            account=launch.EXECUTION_ACCOUNT,
            training_plan=plan,
            namespace_objects=[{"Key": "collision"}],
            active_batch_jobs=[],
            active_training_jobs=[],
            job_name="policy-canary",
        )
    with pytest.raises(SystemExit, match="active duplicate"):
        canary_launch.validate_cloud_snapshot(
            account=launch.EXECUTION_ACCOUNT,
            training_plan=plan,
            namespace_objects=[],
            active_batch_jobs=[{"jobName": "policy-canary"}],
            active_training_jobs=[],
            job_name="policy-canary",
        )
    saturated = {**plan, "InUseInstanceCount": 2, "AvailableInstanceCount": 0}
    with pytest.raises(SystemExit, match="Available1"):
        canary_launch.validate_cloud_snapshot(
            account=launch.EXECUTION_ACCOUNT,
            training_plan=saturated,
            namespace_objects=[],
            active_batch_jobs=[],
            active_training_jobs=[],
            job_name="policy-canary",
        )


def test_service_job_duplicate_scan_uses_correct_api_and_paginates(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_aws_json(*arguments: str) -> dict:
        calls.append(arguments)
        assert arguments[:2] == ("batch", "list-service-jobs")
        status = arguments[arguments.index("--job-status") + 1]
        token = arguments[arguments.index("--next-token") + 1] if "--next-token" in arguments else None
        if status == "SUBMITTED" and token is None:
            return {"jobSummaryList": [{"jobName": "first"}], "nextToken": "second-page"}
        if status == "SUBMITTED" and token == "second-page":
            return {"jobSummaryList": [{"jobName": "policy-canary"}]}
        return {"jobSummaryList": []}

    monkeypatch.setattr(canary_launch, "_aws_json", fake_aws_json)
    jobs = canary_launch._list_active_service_jobs()
    assert [job["jobName"] for job in jobs] == ["first", "policy-canary"]
    assert len(calls) == len(canary_launch.ACTIVE_SERVICE_JOB_STATUSES) + 1
    assert all("--max-results" in call for call in calls)
    assert all("list-jobs" not in call for call in calls)
    scanned_statuses = {call[call.index("--job-status") + 1] for call in calls}
    assert scanned_statuses == set(canary_launch.ACTIVE_SERVICE_JOB_STATUSES)
    assert {"PENDING", "SCHEDULED"} <= scanned_statuses


def test_sagemaker_scans_page_two_for_plans_and_active_duplicate(monkeypatch):
    canary_id = "policy-canary-v1-page-two"

    def fake_aws_json(*arguments: str) -> dict:
        operation = arguments[1]
        token = arguments[arguments.index("--next-token") + 1] if "--next-token" in arguments else None
        if operation == "list-training-plans":
            if token is None:
                return {"TrainingPlanSummaries": [], "NextToken": "plans-page-two"}
            assert token == "plans-page-two"
            return {
                "TrainingPlanSummaries": [
                    {
                        "TrainingPlanName": "cam-robotics-tp",
                        "TrainingPlanArn": launch.training_plan_arn(launch.TRAINING_PLAN_QUEUE),
                    }
                ]
            }
        if operation == "list-training-jobs":
            if token is None:
                return {
                    "TrainingJobSummaries": [{"TrainingJobName": "unrelated"}],
                    "NextToken": "jobs-page-two",
                }
            assert token == "jobs-page-two"
            return {"TrainingJobSummaries": [{"TrainingJobName": "page-two-duplicate"}]}
        assert operation == "describe-training-job"
        name = arguments[-1]
        return {"Environment": {"ROBOMME_CANARY_ID": canary_id if name == "page-two-duplicate" else "other"}}

    monkeypatch.setattr(canary_launch, "_aws_json", fake_aws_json)
    plans = canary_launch._list_active_training_plans()
    assert [plan["TrainingPlanName"] for plan in plans] == ["cam-robotics-tp"]
    jobs = canary_launch._list_in_progress_training_jobs(canary_id=canary_id)
    assert jobs[-1]["CanaryIdMatch"] is True
    plan = {
        **plans[0],
        "Status": "Active",
        "TotalInstanceCount": 2,
        "InUseInstanceCount": 1,
        "AvailableInstanceCount": 1,
        "UnhealthyInstanceCount": 0,
    }
    with pytest.raises(SystemExit, match="active duplicate"):
        canary_launch.validate_cloud_snapshot(
            account=launch.EXECUTION_ACCOUNT,
            training_plan=plan,
            namespace_objects=[],
            active_batch_jobs=[],
            active_training_jobs=jobs,
            job_name="different-job-name",
        )


def test_receipt_requires_finite_components_nonzero_subtree_grads_and_step1_update():
    manifest = _canary_manifest()
    checkpoint = {
        "scope": "node_local_ephemeral_only",
        "step": 1,
        "saved_state_step": 2,
        "restored_state_step": 2,
        "restore_smoke": True,
        "restore_fingerprints": _restore_fingerprint_proof(),
        "parameters_finite": True,
        "ema_parameters_finite": True,
        "optimizer_finite": True,
        "published": False,
        "tree": {"files": 4},
    }
    checkpoint["tree"] = {
        "files": 4,
        "bytes": 4096,
        "layout_sha256": "1" * 64,
        "checkpoint_metadata_sha256": "2" * 64,
        "content_sha256": "3" * 64,
    }
    runtime = {
        "source": {
            "canary_source_tree_sha256": manifest["source"]["canary_source_tree_sha256"],
            "canary_runtime_source_tree_sha256": manifest["source"]["canary_runtime_source_tree_sha256"],
            "entry": canary_launch.ENTRY,
            "entry_sha256": manifest["source"]["entry_sha256"],
            "submitted_entry_mode": canary_launch.SUBMITTED_ENTRY_MODE,
            "sagemaker_runtime_entry_mode": canary_launch.SAGEMAKER_RUNTIME_ENTRY_MODE,
            "excluded_staged_file": canary_launch.STAGED_MANIFEST,
        },
        "infrastructure": {
            "instance_type": "ml.p5e.48xlarge",
            "accelerator": "8xH200",
            "jax_process_count": 1,
        },
        "gpu_inventory": [{"index": index, "uuid": f"GPU-{index}", "name": "NVIDIA H200"} for index in range(8)],
        "jax_inventory": [{"id": index, "platform": "gpu", "device_kind": "NVIDIA H200"} for index in range(8)],
        "openpi_archive_sha256": manifest["identity"]["sources"]["openpi"]["sha256"],
        "openpi_overlay": {
            "overlay_manifest_sha256": policy_canary.GDN_JEPA_OVERLAY_MANIFEST_SHA256,
            "model_math_changed": False,
            "allowed_workspace_pair": ["jepa_aux_target", "tanh"],
            "loaded_pi0": "/tmp/overlay/src/openpi/models/pi0.py",
            "loaded_pi0_config": "/tmp/overlay/src/openpi/models/pi0_config.py",
        },
    }
    receipt = policy_canary.build_completion_receipt(
        manifest,
        records=_records(),
        checkpoint=checkpoint,
        training_job_name="AWSBatch-policy-canary",
        runtime=runtime,
    )
    policy_canary.validate_completion_receipt(receipt)
    assert receipt["claim"] == "not_scientific_training_evidence"
    assert receipt["scientific_evidence"] is False
    assert receipt["deployable"] is False
    assert receipt["evaluation_eligible"] is False
    assert receipt["checkpoint"]["published"] is not True
    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["sealed_manifest"] == manifest
    tampered = json.loads(json.dumps(receipt))
    tampered["scientific_evidence"] = True
    with pytest.raises(ValueError, match="self-seal mismatch"):
        policy_canary.validate_completion_receipt(tampered)
    resealed_bad_runtime = json.loads(json.dumps(receipt))
    resealed_bad_runtime["runtime"]["openpi_archive_sha256"] = "f" * 64
    resealed_bad_runtime.pop("receipt_sha256")
    resealed_bad_runtime["receipt_sha256"] = policy_canary._sha256_json(resealed_bad_runtime)
    with pytest.raises(ValueError, match="runtime OpenPI archive differs"):
        policy_canary.validate_completion_receipt(resealed_bad_runtime)
    resealed_bad_source = json.loads(json.dumps(receipt))
    resealed_bad_source["runtime"]["source"]["canary_runtime_source_tree_sha256"] = "f" * 64
    resealed_bad_source.pop("receipt_sha256")
    resealed_bad_source["receipt_sha256"] = policy_canary._sha256_json(resealed_bad_source)
    with pytest.raises(ValueError, match="runtime source proof differs"):
        policy_canary.validate_completion_receipt(resealed_bad_source)
    resealed_bad_mode = json.loads(json.dumps(receipt))
    resealed_bad_mode["runtime"]["source"]["sagemaker_runtime_entry_mode"] = 0o755
    resealed_bad_mode.pop("receipt_sha256")
    resealed_bad_mode["receipt_sha256"] = policy_canary._sha256_json(resealed_bad_mode)
    with pytest.raises(ValueError, match="entry/mode/exclusion proof drifted"):
        policy_canary.validate_completion_receipt(resealed_bad_mode)
    wrong_overlay = json.loads(json.dumps(receipt))
    wrong_overlay["runtime"]["openpi_overlay"]["overlay_manifest_sha256"] = "f" * 64
    wrong_overlay.pop("receipt_sha256")
    wrong_overlay["receipt_sha256"] = policy_canary._sha256_json(wrong_overlay)
    with pytest.raises(ValueError, match="overlay manifest identity drifted"):
        policy_canary.validate_completion_receipt(wrong_overlay)
    no_op_cold_restore = json.loads(json.dumps(receipt))
    no_op_cold_restore["checkpoint"]["restored_state_step"] = 0
    no_op_cold_restore.pop("receipt_sha256")
    no_op_cold_restore["receipt_sha256"] = policy_canary._sha256_json(no_op_cold_restore)
    with pytest.raises(ValueError, match="checkpoint/restore proof drifted"):
        policy_canary.validate_completion_receipt(no_op_cold_restore)
    no_op_saved_state = json.loads(json.dumps(receipt))
    no_op_saved_state["checkpoint"]["saved_state_step"] = 0
    no_op_saved_state.pop("receipt_sha256")
    no_op_saved_state["receipt_sha256"] = policy_canary._sha256_json(no_op_saved_state)
    with pytest.raises(ValueError, match="checkpoint/restore proof drifted"):
        policy_canary.validate_completion_receipt(no_op_saved_state)

    for component in policy_canary.TREE_FINGERPRINT_COMPONENTS:
        mutated_restore = json.loads(json.dumps(receipt))
        fingerprint = mutated_restore["checkpoint"]["restore_fingerprints"]["restored"][component]
        fingerprint["content_sha256"] = "f" * 64
        fingerprint_core = dict(fingerprint)
        fingerprint_core.pop("fingerprint_sha256")
        fingerprint["fingerprint_sha256"] = policy_canary._sha256_json(fingerprint_core)
        mutated_restore.pop("receipt_sha256")
        mutated_restore["receipt_sha256"] = policy_canary._sha256_json(mutated_restore)
        with pytest.raises(ValueError, match="differs from exact saved host-byte"):
            policy_canary.validate_completion_receipt(mutated_restore)

    resealed_bad_manifest = json.loads(json.dumps(receipt))
    embedded = resealed_bad_manifest["sealed_manifest"]
    embedded["identity"]["data"]["dataset_s3"] = "s3://arbitrary/wrong-data"
    embedded["identity_sha256"] = policy_canary._sha256_json(embedded["identity"])
    embedded["canary_id"] = f"policy-canary-v1-pickxtimes-{canary_launch.ARM}-{embedded['identity_sha256'][:16]}"
    embedded["manifest_sha256"] = policy_canary._sha256_json(
        {name: value for name, value in embedded.items() if name != "manifest_sha256"}
    )
    resealed_bad_manifest["manifest_sha256"] = embedded["manifest_sha256"]
    resealed_bad_manifest["identity_sha256"] = embedded["identity_sha256"]
    resealed_bad_manifest["canary_id"] = embedded["canary_id"]
    resealed_bad_manifest.pop("receipt_sha256")
    resealed_bad_manifest["receipt_sha256"] = policy_canary._sha256_json(resealed_bad_manifest)
    with pytest.raises(ValueError, match="identity drifted"):
        policy_canary.validate_completion_receipt(resealed_bad_manifest)

    for step in (0, 1):
        zero_update = _records()
        zero_update[step]["update_norm"] = 0.0
        with pytest.raises(ValueError, match=f"optimizer step {step}"):
            policy_canary.validate_diagnostic_records(zero_update)
    parity_drift = _records()
    parity_drift[0]["diagnostic_total_loss"] = 1.3
    parity_drift[0]["production_loss_parity_error"] = 0.095
    with pytest.raises(ValueError, match="production compute_loss differs"):
        policy_canary.validate_diagnostic_records(parity_drift)
    zero_jepa_grad = _records()
    zero_jepa_grad[0]["jepa_head_grad_norm"] = 0.0
    with pytest.raises(ValueError, match="both GDN and JEPA"):
        policy_canary.validate_diagnostic_records(zero_jepa_grad)


def test_entry_has_one_create_once_write_and_no_production_completion_path():
    entry = (Path(canary_launch.__file__).resolve().parent / canary_launch.ENTRY).read_text()
    assert entry.count("aws s3api put-object") == 1
    assert "--if-none-match '*'" in entry
    assert "gpu.run_resumable" not in entry
    assert "robomme_gpu_checkpoint_complete" not in entry
    assert "_DEPLOY_COMPLETE" not in entry
    assert "CHECKPOINT_URI=" not in entry
    assert "training_canary.complete.json" in entry
    assert "training.gdn_jepa_overlay stage" in entry
    assert "training.gdn_jepa_overlay validate-loaded" in entry
    runner = (Path(policy_canary.__file__)).read_text()
    assert "model.compute_loss(step_rng, observation, actions, train=True)" in runner
    assert "tree_delta_norm" not in runner
    saved_release = runner.index("saved_fingerprints = fingerprint_train_state_components")
    saved_delete = runner.index("del train_state", saved_release)
    cold_create = runner.index("cold_restore_template, cold_restore_sharding")
    restore_call = runner.index("restored = checkpoints.restore_state", cold_create)
    template_delete = runner.index("del cold_restore_template, cold_restore_sharding", restore_call)
    restored_fingerprint = runner.index("restored_fingerprints = fingerprint_train_state_components", template_delete)
    assert saved_release < saved_delete < cold_create < restore_call < template_delete
    assert template_delete < restored_fingerprint
    assert "jax.clear_caches()" in runner[saved_delete - 1000 : cold_create]
    restore_tail = runner.split("restored = checkpoints.restore_state", 1)[1]
    assert "optax.global_norm" not in restore_tail
    assert "jax.tree.map" not in restore_tail


def test_exact_tree_fingerprint_is_sequential_bounded_and_rejects_mutated_subtrees():
    ml_dtypes = pytest.importorskip("ml_dtypes")

    class LazyLeaf:
        def __init__(self, value: int):
            self.value = value

    host_lifetimes: list[weakref.ReferenceType[np.ndarray]] = []

    def materialize(leaf: LazyLeaf) -> np.ndarray:
        # The previous 16 MiB host leaf must be gone before the next device leaf is requested.
        assert not host_lifetimes or host_lifetimes[-1]() is None
        host = np.full(4 * 1024 * 1024, leaf.value, dtype=np.float32)
        host_lifetimes.append(weakref.ref(host))
        return host

    fingerprint = policy_canary.exact_tree_fingerprint(
        [LazyLeaf(1), LazyLeaf(2), LazyLeaf(3)],
        _flatten_with_path=_test_flatten_with_path,
        _device_get=materialize,
    )
    assert len(host_lifetimes) == 3
    assert all(reference() is None for reference in host_lifetimes)
    assert fingerprint["leaf_count"] == 3
    assert fingerprint["largest_leaf_bytes"] == 16 * 1024 * 1024
    assert fingerprint["total_bytes"] == 48 * 1024 * 1024
    assert fingerprint["all_finite"] is True
    scalar = policy_canary.exact_tree_fingerprint(
        [np.asarray(2, dtype=np.int32)],
        _flatten_with_path=_test_flatten_with_path,
        _device_get=lambda leaf: leaf,
    )
    assert scalar["leaf_count"] == 1
    assert scalar["total_bytes"] == 4
    bfloat_scalar = policy_canary.exact_tree_fingerprint(
        [np.asarray(2, dtype=ml_dtypes.bfloat16)],
        _flatten_with_path=_test_flatten_with_path,
        _device_get=lambda leaf: leaf,
    )
    assert bfloat_scalar["total_bytes"] == 2
    nonfinite = policy_canary.exact_tree_fingerprint(
        [np.asarray([np.nan], dtype=ml_dtypes.bfloat16)],
        _flatten_with_path=_test_flatten_with_path,
        _device_get=lambda leaf: leaf,
    )
    assert nonfinite["all_finite"] is False

    saved = _state_fingerprints()
    policy_canary.build_restore_fingerprint_proof(saved, json.loads(json.dumps(saved)))
    for index, component in enumerate(policy_canary.TREE_FINGERPRINT_COMPONENTS):
        values = [1, 2, 3]
        values[index] += 1
        restored = _state_fingerprints(tuple(values))
        with pytest.raises(ValueError, match=component):
            policy_canary.build_restore_fingerprint_proof(saved, restored)


def test_reviewed_source_delta_rejects_any_unlisted_production_change(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    baseline.mkdir()
    current.mkdir()
    production = "#!/usr/bin/env bash\nexit 0\n"
    (baseline / "gpu_train_entry.sh").write_text(production)
    (current / "gpu_train_entry.sh").write_text(production)
    (current / "gpu_policy_canary_entry.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (baseline / "gpu_train_entry.sh").chmod(0o755)
    (current / "gpu_train_entry.sh").chmod(0o755)
    (current / "gpu_policy_canary_entry.sh").chmod(0o755)
    with launch.prepared_source_bundle(
        baseline,
        "gpu_train_entry.sh",
        {"SAGEMAKER_PROGRAM": "gpu_train_entry.sh"},
    ) as (staged, _entry, _environment):
        baseline_sha = launch.source_tree_sha256(staged)
    monkeypatch.setattr(canary_launch, "PRODUCTION_SOURCE_SHA256", baseline_sha)
    monkeypatch.setattr(
        canary_launch,
        "PRODUCTION_ENTRY_SHA256",
        hashlib.sha256(production.encode()).hexdigest(),
    )
    result = canary_launch.validate_reviewed_source_delta(
        current,
        baseline,
        allowed=frozenset({"gpu_policy_canary_entry.sh"}),
    )
    assert result["reviewed_delta_paths"] == ["gpu_policy_canary_entry.sh"]
    (current / "gpu_train_entry.sh").write_text("#!/usr/bin/env bash\nexit 9\n")
    with pytest.raises(SystemExit, match="unreviewed source delta"):
        canary_launch.validate_reviewed_source_delta(
            current,
            baseline,
            allowed=frozenset({"gpu_policy_canary_entry.sh"}),
        )


def test_packaged_source_verification_accepts_only_toolkit_entry_chmod(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    entry = code / canary_launch.ENTRY
    entry.write_text("#!/usr/bin/env bash\nexit 0\n")
    entry.chmod(canary_launch.SUBMITTED_ENTRY_MODE)
    module = code / "module.py"
    module.write_text("VALUE = 1\n")
    expected_submitted_tree = policy_canary.source_tree_sha256(code)
    entry.chmod(canary_launch.SAGEMAKER_RUNTIME_ENTRY_MODE)
    expected_runtime_tree = policy_canary.source_tree_sha256(code)
    assert expected_runtime_tree != expected_submitted_tree
    manifest = {
        "source": {
            "canary_source_tree_sha256": expected_submitted_tree,
            "canary_runtime_source_tree_sha256": expected_runtime_tree,
            "entry": canary_launch.ENTRY,
            "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
            "submitted_entry_mode": canary_launch.SUBMITTED_ENTRY_MODE,
            "sagemaker_runtime_entry_mode": canary_launch.SAGEMAKER_RUNTIME_ENTRY_MODE,
        }
    }
    staged = code / canary_launch.STAGED_MANIFEST
    staged.write_text("{}\n")
    proof = policy_canary.verify_packaged_source(
        code,
        manifest,
        staged_manifest=canary_launch.STAGED_MANIFEST,
    )
    assert proof["canary_source_tree_sha256"] == expected_submitted_tree
    assert proof["canary_runtime_source_tree_sha256"] == expected_runtime_tree
    assert proof["submitted_entry_mode"] == 0o755
    assert proof["sagemaker_runtime_entry_mode"] == 0o777

    entry.chmod(0o755)
    with pytest.raises(ValueError, match="entry mode contract drifted"):
        policy_canary.verify_packaged_source(
            code,
            manifest,
            staged_manifest=canary_launch.STAGED_MANIFEST,
        )
    entry.chmod(0o777)

    module.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="packaged canary source drifted"):
        policy_canary.verify_packaged_source(
            code,
            manifest,
            staged_manifest=canary_launch.STAGED_MANIFEST,
        )
    module.write_text("VALUE = 1\n")

    module.chmod(0o600)
    with pytest.raises(ValueError, match="packaged canary source drifted"):
        policy_canary.verify_packaged_source(
            code,
            manifest,
            staged_manifest=canary_launch.STAGED_MANIFEST,
        )
    module.chmod(0o644)

    unexpected = code / "unexpected.py"
    unexpected.write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="packaged canary source drifted"):
        policy_canary.verify_packaged_source(
            code,
            manifest,
            staged_manifest=canary_launch.STAGED_MANIFEST,
        )
    unexpected.unlink()

    cache = code / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"runtime-cache")
    with pytest.raises(ValueError, match="packaged canary source drifted"):
        policy_canary.verify_packaged_source(
            code,
            manifest,
            staged_manifest=canary_launch.STAGED_MANIFEST,
        )


def test_failed_node_hash_is_exactly_the_toolkit_entry_chmod(tmp_path):
    archive = Path("/tmp/robomme-policy-canary-b4b983ba73de5624.sourcedir.tar.gz")
    if not archive.is_file():
        pytest.skip("exact first-attempt source archive is not staged")
    code = tmp_path / "submitted-source"
    code.mkdir()
    with tarfile.open(archive, "r:gz") as stream:
        # This is the exact locally retained source archive uploaded by the audited launcher.
        # Preserve its permission bits because modes are part of the source identity.
        stream.extractall(code, filter="fully_trusted")
    entry = code / canary_launch.ENTRY
    assert entry.stat().st_mode & 0o777 == 0o755
    excluded = frozenset({canary_launch.STAGED_MANIFEST})
    submitted = policy_canary.source_tree_sha256(code, excluded=excluded)
    assert submitted == "79da8f380d2ee757c2d039aa743e272b941aebc5355d28d20cb8bb444fe609a5"
    entry.chmod(0o777)
    runtime = policy_canary.source_tree_sha256(code, excluded=excluded)
    normalized = policy_canary.source_tree_sha256(
        code,
        excluded=excluded,
        mode_overrides={canary_launch.ENTRY: 0o755},
    )
    assert runtime == "3cf561f41edd43a6747af3ff8a8c4cd975bb25347aa5e9d908337c0d622e9016"
    assert normalized == submitted


def test_strong_checkpoint_tree_binds_all_local_bytes(tmp_path):
    checkpoint = tmp_path / "1"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "array").write_bytes(b"parameter-bytes")
    (checkpoint / "_CHECKPOINT_METADATA").write_bytes(b"metadata")
    before = policy_canary.strong_checkpoint_tree(checkpoint)
    assert before["files"] == 2
    assert before["bytes"] == len(b"parameter-bytesmetadata")
    assert len(before["content_sha256"]) == 64
    (checkpoint / "params" / "array").write_bytes(b"changed-content")
    after = policy_canary.strong_checkpoint_tree(checkpoint)
    assert before["content_sha256"] != after["content_sha256"]


def test_production_policy_training_files_still_match_frozen_528d_tree():
    baseline = Path("/tmp/robomme-wave2-bc3c-clean")
    if not baseline.is_dir():
        pytest.skip("frozen 528d source tree not staged")
    current = Path(canary_launch.__file__).resolve().parent
    for relative in (
        "gpu_train_entry.sh",
        "launch.py",
        "training/train.py",
        "training/config.py",
        "training/gdn_jepa_overlay.py",
        "gpu/run_resumable.py",
    ):
        assert (current / relative).read_bytes() == (baseline / relative).read_bytes(), relative
