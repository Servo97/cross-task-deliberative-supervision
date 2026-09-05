from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from robomme_integration import v4_policy_canary_launch
from robomme_integration.training import v4_policy_canary as contract
from scripts.launch.launch_guardrails import prepared_source_bundle, write_staged_source_files

SOURCE = Path(__file__).resolve().parents[1]


def _fingerprints():
    def fingerprint(marker):
        core = {
            "algorithm": "jax-tree-path-shape-dtype-c-order-raw-bytes-sha256-v1",
            "leaf_count": 1,
            "total_bytes": 4,
            "largest_leaf_bytes": 4,
            "all_finite": True,
            "structure_sha256": marker * 64,
            "metadata_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "leaf_records_sha256": "c" * 64,
        }
        return {**core, "fingerprint_sha256": contract.sha_json(core)}

    saved = {
        "params": fingerprint("1"),
        "ema_params": fingerprint("2"),
        "optimizer_state": fingerprint("3"),
    }
    return {
        "algorithm": "jax-tree-path-shape-dtype-c-order-raw-bytes-sha256-v1",
        "max_simultaneous_host_leaves": 1,
        "saved": saved,
        "restored": copy.deepcopy(saved),
        "exact_match": {"params": True, "ema_params": True, "optimizer_state": True},
    }


def _receipt(plan):
    records = []
    for step in range(2):
        records.append(
            {
                "step": step,
                "total_loss": 1.15,
                "action_loss": 1.0,
                "jepa_loss": 1.0,
                "jepa_weighted": 0.1,
                "visreg_loss": 1.0,
                "visreg_weighted": 0.05,
                "grad_norm": 2.0,
                "backbone_grad_norm": 1.0,
                "gdn_grad_norm": 1.0,
                "jepa_head_grad_norm": 1.0,
                "update_norm": 1.0,
                "parameter_delta_norm": 1.0,
                "backbone_update_norm": 1.0,
                "gdn_update_norm": 1.0,
                "jepa_head_update_norm": 1.0,
                "ema_delta_norm": 0.001,
                "ema_decay": 0.999,
                "ema_formula_residual": 0.0,
                "backbone_applied_lr": 2.5e-5,
                "new_module_applied_lr": 1.5e-4,
                "applied_lr_ratio": 6.0,
                "parameters_finite": True,
                "optimizer_finite": True,
                "valid_target_count": 64.0,
                "production_loss_parity_error": 0.0,
            }
        )
    manifest = plan["manifest"]
    return contract.seal(
        {
            "schema_version": 1,
            "kind": contract.RECEIPT_KIND,
            "claim": contract.CLAIM,
            "contract": contract.CONTRACT,
            "canary_id": manifest["canary_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "identity": manifest["identity"],
            "identity_sha256": manifest["identity_sha256"],
            "source": {
                "prepared_source_tree_sha256": manifest["identity"]["prepared_source_tree_sha256"],
                "entry": contract.ENTRY,
                "entry_bytes_sha256": manifest["identity"]["entry_bytes_sha256"],
                "runtime_entry_mode": "0o777",
                "staged_manifest_excluded_from_source_identity": contract.STAGED_MANIFEST,
                "submitted_entry_mode": "0o755",
                "sagemaker_runtime_entry_mode": "0o777",
            },
            "arm": contract.ARM,
            "task": "PickXtimes",
            "training_job_name": "exact-job",
            "runtime": {
                "instance_type": "ml.p5.48xlarge",
                "jax_processes": 1,
                "dtype": "bfloat16",
                "jax_inventory": [
                    {"id": i, "platform": "gpu", "device_kind": "NVIDIA H100 80GB HBM3"} for i in range(8)
                ],
                "gpu_names": ["NVIDIA H100 80GB HBM3"] * 8,
                "gpu_uuids": [f"uuid-{i}" for i in range(8)],
                "gpu_memory_mib": [81559] * 8,
            },
            "optimizer_recipe": {
                "schema_version": 1,
                "protocol": "robomme_v4",
                "arm": contract.ARM,
                "global_clip_before_parameter_groups": 10.0,
                "weight_decay": 1e-6,
                "backbone_lr": {"peak": 5e-5, "decay": 5e-6},
                "new_module_lr": {"peak": 3e-4, "decay": 3e-5},
                "new_parameter_subtrees": ["wsm_tanh_cond", "wsm_jepa_head"],
                "new_parameter_group_active": True,
            },
            "optimizer_semantics_probe": {
                "forced_raw_gradient_norm": 20.0,
                "forced_post_global_clip_norm": 10.0,
                "global_clip_threshold": 10.0,
                "zero_gradient_nonzero_parameter_weight_decay": {
                    "backbone_update_abs": 1e-10,
                    "gdn_update_abs": 6e-10,
                    "jepa_update_abs": 6e-10,
                    "new_to_backbone_ratio": 6.0,
                },
            },
            "parameter_routing": {
                "labels": {
                    "wsm_tanh_cond": "new_module",
                    "wsm_jepa_head": "new_module",
                    "all_other_parameter_paths": "backbone",
                },
                "leaf_counts": {"backbone": 10, "new_module": 4},
            },
            "ema_decay": 0.999,
            "records": records,
            "checkpoint": {
                "scope": "node_local_ephemeral_only",
                "step": 1,
                "saved_state_step": 2,
                "restored_state_step": 2,
                "tree": {"files": 1},
                "restore_fingerprints": _fingerprints(),
                "published": False,
            },
            "production_checkpoint_or_deploy_publication": False,
        },
        "receipt_sha256",
    )


def _reseal_manifest(value: dict) -> dict:
    clean = copy.deepcopy(value)
    clean.pop("manifest_sha256", None)
    return contract.seal(clean, "manifest_sha256")


def test_exact_v4_canary_manifest_builds_and_validates():
    plan = v4_policy_canary_launch.build(SOURCE)
    contract.validate_manifest(plan["manifest"])
    assert plan["manifest"]["reference_training_manifest"]["scientific"]["mechanism"]["jepa"]["sigreg_weight"] == 0.0
    assert plan["manifest"]["source"]["prepared_source_tree_sha256"] == plan["source_tree_sha256"]
    assert not {"OUTPUT_S3", "RUN_MANIFEST_S3", "COMPLETION_CLAIM_S3", "ROBOMME_RUN_ID"} & plan["environment"].keys()
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "1"
    assert plan["manifest"]["infrastructure"]["training_plan_arn"] is None


def test_exact_p5_h100_topology_rejects_fake_or_other_accelerators():
    names = ["NVIDIA H100 80GB HBM3"] * 8
    uuids = [f"uuid-{index}" for index in range(8)]
    memory = [81559] * 8
    contract.validate_h100_node_topology(
        device_count=8,
        process_count=1,
        gpu_names=names,
        gpu_uuids=uuids,
        gpu_memory_mib=memory,
    )
    for bad_name in ("NVIDIA H100 FAKE", "NVIDIA H100", "NVIDIA A100-SXM4-80GB", "NVIDIA H200"):
        bad = list(names)
        bad[0] = bad_name
        with pytest.raises(ValueError, match="H100 80GB"):
            contract.validate_h100_node_topology(
                device_count=8,
                process_count=1,
                gpu_names=bad,
                gpu_uuids=uuids,
                gpu_memory_mib=memory,
            )
    with pytest.raises(ValueError, match="UUID"):
        contract.validate_h100_node_topology(
            device_count=8,
            process_count=1,
            gpu_names=names,
            gpu_uuids=["same"] * 8,
            gpu_memory_mib=memory,
        )
    with pytest.raises(ValueError, match="memory"):
        contract.validate_h100_node_topology(
            device_count=8,
            process_count=1,
            gpu_names=names,
            gpu_uuids=uuids,
            gpu_memory_mib=[40_000] * 8,
        )


def test_verify_source_shaped_proof_is_accepted_by_receipt_validator():
    plan = v4_policy_canary_launch.build(SOURCE)
    with prepared_source_bundle(SOURCE, contract.ENTRY, {"SAGEMAKER_PROGRAM": contract.ENTRY}) as (
        staged,
        _entry,
        _environment,
    ):
        write_staged_source_files(staged, plan["staged_source_files"])
        (staged / contract.ENTRY).chmod(0o777)
        proof = contract.verify_source(staged, plan["manifest"])
        assert set(proof) == {
            "prepared_source_tree_sha256",
            "entry",
            "entry_bytes_sha256",
            "runtime_entry_mode",
            "staged_manifest_excluded_from_source_identity",
            "submitted_entry_mode",
            "sagemaker_runtime_entry_mode",
        }
        receipt = _receipt(plan)
        receipt["source"] = proof
        receipt = contract.seal(receipt, "receipt_sha256")
        contract.validate_receipt(receipt, plan["manifest"])


def test_resealed_wrong_workspace_uri_is_rejected():
    manifest = v4_policy_canary_launch.build(SOURCE)["manifest"]
    mutated = copy.deepcopy(manifest)
    mutated["reference_training_manifest"]["scientific"]["workspace_representation"]["omega"]["uri"] += "-wrong"
    scientific = mutated["reference_training_manifest"]["scientific"]
    mutated["reference_training_manifest"]["scientific_spec_sha256"] = contract.sha_json(scientific)
    reference = dict(mutated["reference_training_manifest"])
    reference.pop("manifest_sha256")
    mutated["reference_training_manifest"]["manifest_sha256"] = contract.sha_json(reference)
    mutated = _reseal_manifest(mutated)
    with pytest.raises(ValueError, match="reference|workspace"):
        contract.validate_manifest(mutated)


def test_resealed_sigreg_or_source_tree_mutations_are_rejected():
    manifest = v4_policy_canary_launch.build(SOURCE)["manifest"]
    for mutate in ("sigreg", "source"):
        value = copy.deepcopy(manifest)
        if mutate == "sigreg":
            scientific = value["reference_training_manifest"]["scientific"]
            scientific["mechanism"]["jepa"]["sigreg_weight"] = 0.05
            value["reference_training_manifest"]["scientific_spec_sha256"] = contract.sha_json(scientific)
            reference = dict(value["reference_training_manifest"])
            reference.pop("manifest_sha256")
            value["reference_training_manifest"]["manifest_sha256"] = contract.sha_json(reference)
        else:
            value["source"]["prepared_source_tree_sha256"] = "0" * 64
        value = _reseal_manifest(value)
        with pytest.raises(ValueError):
            contract.validate_manifest(value)


def test_shell_validate_receipt_cli_matches_module_parser(monkeypatch, tmp_path):
    script = (SOURCE / "gpu_v4_policy_canary_entry.sh").read_text(encoding="utf-8")
    assert "validate-receipt" in script
    assert '--manifest "$MANIFEST" --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256"' in script
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v4_policy_canary",
            "validate-receipt",
            "--receipt",
            str(receipt),
            "--manifest",
            str(manifest),
            "--sha256",
            "0" * 64,
        ],
    )
    with pytest.raises((ValueError, KeyError)) as error:
        contract.main()
    assert "unrecognized arguments" not in str(error.value)


def test_openpi_runtime_probe_uses_the_sealed_robocasa_compatibility_surface():
    script = (SOURCE / "gpu_v4_policy_canary_entry.sh").read_text(encoding="utf-8")
    compat_guard = '[[ -f "$COMPAT/robocasa/utils/groot_utils/embodiment_tags.py" ]]'
    probe = 'PYTHONPATH="$COMPAT:$CODE_DIR:$OPENPI/src" "$PY" - <<\'PY\''
    assert compat_guard in script
    assert probe in script
    assert script.index(compat_guard) < script.index("from openpi.models import pi0_config")
    assert 'assert "wsm_cond_history_dropout" in parameters' in script
    assert 'assert getattr(wsm_current_cond, "_WSM_PTRM", False)' in script
    assert 'assert getattr(pi0_config, "_WORKSPACE_COMBO", None)' in script
    assert "from openpi.training import config as c" not in script
    assert 'PYTHONPATH="$OPENPI/src" "$PY" - <<\'PY\'' not in script


@pytest.mark.parametrize(
    "name",
    [
        "ROBOMME_DATA_S3",
        "ROBOMME_DATA_PARENT_INVENTORY_S3",
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256",
        "ROBOMME_DATA_DERIVED_INVENTORY_SHA256",
        "INIT_S3",
        "INIT_INVENTORY_S3",
        "INIT_INVENTORY_SHA256",
        "OPENPI_FORK_S3",
        "PALIGEMMA_TOKENIZER_S3",
        "PALIGEMMA_TOKENIZER_SHA256",
        "ROBOMME_WORKSPACE_ENCODER_ID",
        "ROBOMME_WORKSPACE_S3",
        "ROBOMME_WORKSPACE_MANIFEST_SHA256",
        "ROBOMME_TASK",
        "ROBOMME_ARM",
        "WSM_MAX_STEPS",
        "WSM_WARMUP_STEPS",
        "WSM_PEAK_LR",
        "WSM_DECAY_STEPS",
        "WSM_DECAY_LR",
        "SM_USE_RESERVED_CAPACITY",
    ],
)
def test_every_duplicate_environment_input_is_exactly_bound(name):
    plan = v4_policy_canary_launch.build(SOURCE)
    contract.validate_environment(plan["manifest"], plan["environment"])
    mutated = dict(plan["environment"])
    mutated[name] += "-wrong"
    with pytest.raises(ValueError, match="environment differs"):
        contract.validate_environment(plan["manifest"], mutated)


def test_resealed_identity_and_publication_mutations_are_rejected():
    manifest = v4_policy_canary_launch.build(SOURCE)["manifest"]
    for mutation in ("identity", "namespace", "infrastructure", "only_object"):
        value = copy.deepcopy(manifest)
        if mutation == "identity":
            value["identity"]["reference_run_id"] += "-wrong"
            value["identity_sha256"] = contract.sha_json(value["identity"])
            value["canary_id"] = f"v4-policy-canary-{value['identity_sha256'][:20]}"
            value["publication"]["namespace_s3"] = (
                contract.STUDY_ROOT + "/manifests/canaries/policy_training/" + value["canary_id"]
            )
            value["publication"]["receipt_s3"] = (
                value["publication"]["namespace_s3"] + "/training_canary.complete.json"
            )
        elif mutation == "namespace":
            value["publication"]["namespace_s3"] += "-wrong"
            value["publication"]["receipt_s3"] = (
                value["publication"]["namespace_s3"] + "/training_canary.complete.json"
            )
        elif mutation == "infrastructure":
            value["infrastructure"]["volume_size_gb"] = 300
        else:
            value["publication"]["only_allowed_object"] = "anything.json"
        value = _reseal_manifest(value)
        with pytest.raises(ValueError):
            contract.validate_manifest(value)


@pytest.mark.parametrize(
    "path,value",
    [
        (("identity", "task"), "MoveCube"),
        (("identity", "reference_manifest_sha256"), "0" * 64),
        (("identity", "prepared_source_tree_sha256"), "0" * 64),
        (("identity", "optimizer_steps"), 999),
        (
            ("publication", "namespace_s3"),
            contract.STUDY_ROOT + "/manifests/canaries/policy_training/v4-policy-canary-evil",
        ),
        (("publication", "only_allowed_object"), "evil.bin"),
        (("infrastructure", "instance_type"), "ml.g5.48xlarge"),
    ],
)
def test_exact_adversarial_resealed_manifest_mutations_are_rejected(path, value):
    manifest = copy.deepcopy(v4_policy_canary_launch.build(SOURCE)["manifest"])
    cursor = manifest
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    if path == ("publication", "namespace_s3"):
        manifest["publication"]["receipt_s3"] = value + "/training_canary.complete.json"
    if path[0] == "identity":
        manifest["identity_sha256"] = contract.sha_json(manifest["identity"])
        manifest["canary_id"] = f"v4-policy-canary-{manifest['identity_sha256'][:20]}"
    manifest = _reseal_manifest(manifest)
    with pytest.raises(ValueError):
        contract.validate_manifest(manifest)


@pytest.mark.parametrize(
    "path,value",
    [
        (("arm",), "v4_s0"),
        (("training_job_name",), ""),
        (("runtime", "dtype"), "float32"),
        (("runtime", "instance_type"), "ml.g5.48xlarge"),
        (("source", "runtime_entry_mode"), "0o755"),
        (("optimizer_recipe", "weight_decay"), 1e-5),
        (("ema_decay",), 0.99),
        (("checkpoint", "restored_state_step"), 1),
        (("production_checkpoint_or_deploy_publication",), True),
    ],
)
def test_resealed_receipt_mutations_are_rejected(path, value):
    plan = v4_policy_canary_launch.build(SOURCE)
    receipt = _receipt(plan)
    contract.validate_receipt(receipt, plan["manifest"])
    mutated = copy.deepcopy(receipt)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    mutated = contract.seal(mutated, "receipt_sha256")
    with pytest.raises(ValueError):
        contract.validate_receipt(mutated, plan["manifest"])


def test_step_records_accept_float32_transport_of_ema_0999_but_reject_recipe_drift():
    records = copy.deepcopy(_receipt(v4_policy_canary_launch.build(SOURCE))["records"])
    for record in records:
        record["ema_decay"] = 0.9990000128746033
    contract._validate_records(records)

    records[0]["ema_decay"] = 0.99
    with pytest.raises(ValueError, match="EMA 0.999"):
        contract._validate_records(records)
