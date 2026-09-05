#!/usr/bin/env python3
"""Two-step v4 GDN8+JEPA+VISReg policy canary (never scientific evidence)."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from .policy_canary import (
    build_restore_fingerprint_proof,
    fingerprint_train_state_components,
    source_tree_sha256,
    strong_checkpoint_tree,
)

KIND = "robomme_v4_policy_training_canary_attempt"
RECEIPT_KIND = "robomme_v4_policy_training_canary_complete"
CLAIM = "runtime_evidence_only_not_scientific_training_evidence"
CONTRACT = "robomme-v4-policy-canary-v1"
ARM = "v4_gdn8_jepa_visreg_l01_k1"
ENTRY = "gpu_v4_policy_canary_entry.sh"
STAGED_MANIFEST = "_robomme_v4_policy_canary_manifest.json"
STEPS = 2
FINAL_STEP = 1
HEX = frozenset("0123456789abcdef")
STUDY_ROOT = "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"


H100_DEVICE_NAMES = frozenset({"NVIDIA H100 80GB HBM3"})


def is_h100_device_name(value: object) -> bool:
    """Accept only the exact H100 model reported by ml.p5.48xlarge nodes."""
    return isinstance(value, str) and value.strip() in H100_DEVICE_NAMES


def validate_h100_node_topology(
    *,
    device_count: int,
    process_count: int,
    gpu_names: list[str],
    gpu_uuids: list[str],
    gpu_memory_mib: list[int],
) -> None:
    if device_count != 8 or process_count != 1:
        raise ValueError(
            f"p5 canary requires eight devices in one process; got devices={device_count} processes={process_count}"
        )
    if len(gpu_names) != 8 or any(not is_h100_device_name(name) for name in gpu_names):
        raise ValueError(f"p5 canary requires exact 8x H100 80GB HBM3; got {gpu_names}")
    if len(gpu_uuids) != 8 or any(not value.strip() for value in gpu_uuids) or len(set(gpu_uuids)) != 8:
        raise ValueError("p5 canary requires eight distinct nonempty GPU UUIDs")
    if len(gpu_memory_mib) != 8 or any(value < 80_000 or value > 90_000 for value in gpu_memory_mib):
        raise ValueError(f"p5 canary requires eight H100 80GB memory reports; got {gpu_memory_mib}")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = sha_json(result)
    return result


def validate_manifest(value: dict[str, Any]) -> None:
    claimed = require_sha(value.get("manifest_sha256"), "manifest_sha256")
    clean = dict(value)
    clean.pop("manifest_sha256")
    if sha_json(clean) != claimed:
        raise ValueError("v4 canary manifest seal mismatch")
    if (
        value.get("kind") != KIND
        or value.get("claim") != CLAIM
        or value.get("contract") != CONTRACT
        or value.get("arm") != ARM
        or value.get("task") != "PickXtimes"
    ):
        raise ValueError("v4 canary identity drifted")
    identity = value.get("identity")
    if not isinstance(identity, dict) or sha_json(identity) != value.get("identity_sha256"):
        raise ValueError("v4 canary identity seal mismatch")
    execution = value.get("execution", {})
    required_execution = {
        "optimizer_steps": 2,
        "final_local_checkpoint_step": 1,
        "batch_size": 64,
        "dtype": "bfloat16",
        "history_dropout": 0.0,
        "checkpoint_scope": "node_local_ephemeral_only",
        "cold_restore": True,
    }
    if execution != required_execution:
        raise ValueError(f"v4 canary execution drifted: {execution}")
    source = value.get("source", {})
    if (
        set(source)
        != {
            "prepared_source_tree_sha256",
            "entry",
            "entry_bytes_sha256",
            "submitted_entry_mode",
            "sagemaker_runtime_entry_mode",
        }
        or source.get("entry") != ENTRY
        or source.get("submitted_entry_mode") != "0o755"
        or source.get("sagemaker_runtime_entry_mode") != "0o777"
    ):
        raise ValueError("v4 canary source contract drifted")
    require_sha(source.get("prepared_source_tree_sha256"), "prepared source tree")
    require_sha(source.get("entry_bytes_sha256"), "entry bytes")
    reference = value.get("reference_training_manifest")
    if not isinstance(reference, dict):
        raise ValueError("v4 canary lacks exact production reference manifest")
    reference_clean = dict(reference)
    reference_claim = require_sha(reference_clean.pop("manifest_sha256", None), "reference manifest")
    if sha_json(reference_clean) != reference_claim:
        raise ValueError("production reference manifest seal mismatch")
    scientific = reference.get("scientific")
    if (
        not isinstance(scientific, dict)
        or reference.get("scientific_spec_sha256") != sha_json(scientific)
        or scientific.get("arm") != ARM
        or scientific.get("task", {}).get("name") != "PickXtimes"
        or scientific.get("scope") != "single_task_v4"
    ):
        raise ValueError("v4 canary reference is not exact PickXtimes GDN8+JEPA")
    expected_run_id = f"st-v4-pickxtimes-{ARM}-seed0-{reference['scientific_spec_sha256'][:16]}"
    if (
        reference.get("run_id") != expected_run_id
        or reference.get("attempt_id") != f"{expected_run_id}-attempt1"
        or not str(reference.get("manifest_s3", "")).endswith(
            f"/manifests/runs/train/{expected_run_id}/{expected_run_id}-attempt1.json"
        )
    ):
        raise ValueError("v4 canary reference run/attempt/manifest namespace is not spec-derived")
    training = scientific.get("training", {})
    if (
        training.get("steps") != 20_000
        or training.get("warmup_steps") != 1_000
        or training.get("decay_steps") != 20_000
        or training.get("batch_size") != 64
        or training.get("ema_decay") != 0.999
        or training.get("peak_lr") != 5e-5
        or training.get("decay_lr") != 5e-6
        or training.get("optimizer", {}).get("weight_decay") != 1e-6
        or training.get("optimizer", {}).get("clip_gradient_norm") != 10.0
        or training.get("parameter_groups", {}).get("new_modules")
        != {
            "active": True,
            "peak_lr": 3e-4,
            "decay_lr": 3e-5,
            "subtrees": ["wsm_tanh_cond", "wsm_jepa_head"],
        }
        or training.get("parameter_groups", {}).get("pretrained_backbone_and_action_model", {}).get("peak_lr") != 5e-5
    ):
        raise ValueError("v4 canary reference optimizer/training contract drifted")
    mechanism = scientific.get("mechanism", {})
    jepa = mechanism.get("jepa") or {}
    if (
        mechanism.get("steering") != "gated_deltanet_k8"
        or mechanism.get("train_history_dropout") != 0.0
        or jepa.get("regularizer") != "visreg"
        or jepa.get("sigreg_weight") != 0.0
        or jepa.get("visreg_weight") != 0.05
        or jepa.get("visreg_slices") != 128
        or jepa.get("visreg_components") != {"scale": 1.0, "shape": 1.0, "center": 1.0}
    ):
        raise ValueError("v4 canary reference mechanism is not GDN8+JEPA+VISReg")
    expected_source_tree = value.get("source", {}).get("prepared_source_tree_sha256")
    sources = scientific.get("sources", {})
    workspace = scientific.get("workspace_representation", {})
    if (
        sources.get("robomme_integration", {}).get("sanitized_source_tree_sha256") != expected_source_tree
        or sources.get("openpi", {}).get("sha256")
        != "24bd889d3c0b95a7b01cd6ad30a91fdc266fa115fb2ef5ec89fe45c9c5260900"
        or workspace.get("encoder_id") != "dd5a17e4929537f9b0472f374081618a5dff5deed31af1969324ded291bd9c15"
        or workspace.get("omega", {}).get("manifest_sha256")
        != "4ea7ca3f9759c4cc435a7c6e754f6a3227ed9f043df780eb851e37a3989a5775"
        or workspace.get("omega", {}).get("uri")
        != (
            "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/"
            "long_context_v1/artifacts/robomme/workspace/PickXtimes/"
            "dd5a17e4929537f9b0472f374081618a5dff5deed31af1969324ded291bd9c15/omega"
        )
    ):
        raise ValueError("v4 canary source/OpenPI/Pick workspace identity drifted")
    expected_identity = {
        "prepared_source_tree_sha256": source["prepared_source_tree_sha256"],
        "entry_bytes_sha256": source["entry_bytes_sha256"],
        "reference_run_id": reference["run_id"],
        "reference_scientific_spec_sha256": reference["scientific_spec_sha256"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "task": value["task"],
        "arm": value["arm"],
        "batch_size": execution["batch_size"],
        "optimizer_steps": execution["optimizer_steps"],
        "dtype": execution["dtype"],
    }
    if identity != expected_identity:
        raise ValueError("v4 canary identity is not exactly reconstructed from sealed fields")
    expected_canary_id = f"v4-policy-canary-{value['identity_sha256'][:20]}"
    if value.get("canary_id") != expected_canary_id:
        raise ValueError("v4 canary id is not derived from its complete identity")
    infrastructure = value.get("infrastructure")
    if infrastructure != {
        "queue": "fss-tri-cam-robotics-p5-48xlarge-us-west-2",
        "training_plan_arn": None,
        "instance_type": "ml.p5.48xlarge",
        "topology": "1x8 NVIDIA H100 80GB HBM3",
        "priority": 400,
        "max_run_seconds": 10_800,
        "volume_size_gb": 400,
    }:
        raise ValueError("v4 canary infrastructure contract drifted")
    publication = value.get("publication", {})
    namespace = publication.get("namespace_s3", "")
    expected_namespace = f"{STUDY_ROOT}/manifests/canaries/policy_training/{expected_canary_id}"
    if (
        set(publication)
        != {
            "namespace_s3",
            "receipt_s3",
            "create_once",
            "only_allowed_object",
            "production_checkpoint_or_deploy_publication",
        }
        or namespace != expected_namespace
        or publication.get("receipt_s3") != namespace.rstrip("/") + "/training_canary.complete.json"
        or publication.get("create_once") is not True
        or publication.get("only_allowed_object") != "training_canary.complete.json"
        or publication.get("production_checkpoint_or_deploy_publication") is not False
    ):
        raise ValueError("v4 canary publication escaped isolated create-once namespace")


def load_manifest(path: Path, expected_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    if value["manifest_sha256"] != expected_sha:
        raise ValueError("staged v4 canary manifest differs from environment seal")
    return value


def validate_environment(manifest: dict[str, Any], environment: dict[str, str]) -> None:
    """Bind every duplicated download/runtime input to the exact sealed production reference."""

    validate_manifest(manifest)
    scientific = manifest["reference_training_manifest"]["scientific"]
    workspace = scientific["workspace_representation"]
    expected = {
        "ROBOMME_CANARY_KIND": KIND,
        "ROBOMME_CANARY_CLAIM": CLAIM,
        "ROBOMME_CANARY_ID": manifest["canary_id"],
        "ROBOMME_CANARY_MANIFEST_SHA256": manifest["manifest_sha256"],
        "ROBOMME_CANARY_NAMESPACE_S3": manifest["publication"]["namespace_s3"],
        "ROBOMME_CANARY_RECEIPT_S3": manifest["publication"]["receipt_s3"],
        "ROBOMME_TASK": "PickXtimes",
        "ROBOMME_ARM": ARM,
        "ROBOMME_DATA_S3": scientific["data"]["dataset_s3"],
        "ROBOMME_DATA_PARENT_INVENTORY_S3": scientific["data"]["parent_inventory_uri"],
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": scientific["data"]["parent_inventory_sha256"],
        "ROBOMME_DATA_DERIVED_INVENTORY_SHA256": scientific["data"]["derived_task_inventory_sha256"],
        "INIT_S3": scientific["initialization"]["checkpoint_s3"],
        "INIT_INVENTORY_S3": scientific["initialization"]["inventory_uri"],
        "INIT_INVENTORY_SHA256": scientific["initialization"]["inventory_sha256"],
        "OPENPI_FORK_S3": scientific["sources"]["openpi"]["uri"],
        "OPENPI_REQUIRED_SENTINEL": "_WSM_V4_ADVANCED",
        "PALIGEMMA_TOKENIZER_S3": scientific["sources"]["tokenizer"]["uri"],
        "PALIGEMMA_TOKENIZER_SHA256": scientific["sources"]["tokenizer"]["sha256"],
        "ROBOMME_WORKSPACE_ENCODER_ID": workspace["encoder_id"],
        "ROBOMME_WORKSPACE_S3": workspace["omega"]["uri"],
        "ROBOMME_WORKSPACE_MANIFEST_SHA256": workspace["omega"]["manifest_sha256"],
        "WSM_MAX_STEPS": "2",
        "WSM_SAVE_INTERVAL": "2",
        "WSM_WARMUP_STEPS": "1",
        "WSM_PEAK_LR": "5e-5",
        "WSM_DECAY_STEPS": "2",
        "WSM_DECAY_LR": "5e-6",
        "WSM_SEED": "0",
        "SM_USE_RESERVED_CAPACITY": "1",
    }
    drift = {
        name: (environment.get(name), wanted) for name, wanted in expected.items() if environment.get(name) != wanted
    }
    if drift:
        raise ValueError(f"v4 canary environment differs from sealed reference: {drift}")
    forbidden = {
        "OUTPUT_S3",
        "RUN_MANIFEST_SOURCE",
        "RUN_MANIFEST_SHA256",
        "RUN_MANIFEST_S3",
        "PRODUCER_CLAIM_S3",
        "COMPLETION_CLAIM_S3",
        "CHECKPOINT_TREE_MANIFEST_ROOT",
        "ROBOMME_SCIENTIFIC_SPEC_SHA256",
        "ROBOMME_FINAL_STEP",
        "ROBOMME_RUN_ID",
    }
    if forbidden & environment.keys():
        raise ValueError("production publication environment leaked into v4 canary")


def verify_source(code_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    entry = code_dir / ENTRY
    actual_entry = hashlib.sha256(entry.read_bytes()).hexdigest()
    if actual_entry != source["entry_bytes_sha256"]:
        raise ValueError("v4 canary entry bytes drifted")
    actual_mode = stat.S_IMODE(entry.stat().st_mode)
    if (
        source.get("submitted_entry_mode") != "0o755"
        or source.get("sagemaker_runtime_entry_mode") != "0o777"
        or actual_mode != 0o777
    ):
        raise ValueError("v4 canary SageMaker entry mode contract drifted")
    actual_tree = source_tree_sha256(
        code_dir,
        excluded=frozenset({STAGED_MANIFEST}),
        mode_overrides={ENTRY: 0o755},
    )
    if actual_tree != source["prepared_source_tree_sha256"]:
        raise ValueError("on-node v4 canary source differs from exact prepared source")
    return {
        "prepared_source_tree_sha256": actual_tree,
        "entry": ENTRY,
        "entry_bytes_sha256": actual_entry,
        "runtime_entry_mode": oct(actual_mode),
        "staged_manifest_excluded_from_source_identity": STAGED_MANIFEST,
        "submitted_entry_mode": "0o755",
        "sagemaker_runtime_entry_mode": "0o777",
    }


def _finite(tree: Any, jax: Any, jnp: Any) -> Any:
    leaves = [
        jnp.all(jnp.isfinite(jnp.asarray(x)))
        for x in jax.tree_util.tree_leaves(tree)
        if hasattr(x, "dtype") and jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)
    ]
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.asarray(True)


def _subtree_norm(tree: Any, name: str, jax: Any, jnp: Any, optax: Any) -> Any:
    def select(path: Any, value: Any) -> Any:
        head = getattr(path[0], "key", None) if path else None
        return value if head == name else jnp.zeros_like(value)

    return optax.global_norm(jax.tree_util.tree_map_with_path(select, tree))


def _validate_records(records: list[dict[str, Any]]) -> None:
    if [record.get("step") for record in records] != [0, 1]:
        raise ValueError("v4 canary did not execute exactly steps [0,1]")
    positive = (
        "total_loss",
        "action_loss",
        "jepa_loss",
        "jepa_weighted",
        "visreg_loss",
        "visreg_weighted",
        "grad_norm",
        "backbone_grad_norm",
        "gdn_grad_norm",
        "jepa_head_grad_norm",
        "update_norm",
        "parameter_delta_norm",
        "backbone_update_norm",
        "gdn_update_norm",
        "jepa_head_update_norm",
        "ema_delta_norm",
        "backbone_applied_lr",
        "new_module_applied_lr",
        "applied_lr_ratio",
    )
    for record in records:
        for name in positive:
            value = float(record.get(name, math.nan))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"step {record.get('step')} has invalid {name}={value}")
        if record.get("parameters_finite") is not True or record.get("optimizer_finite") is not True:
            raise ValueError("v4 canary produced non-finite state")
        # This diagnostic is emitted from a JAX scalar.  In the production BF16 run the decay is
        # intentionally represented as float32, where 0.999 round-trips as
        # 0.9990000128746033.  Keep the exact EMA-formula residual check below as the strong
        # semantic proof, while allowing only the expected float32 transport error here.
        if not math.isclose(record.get("ema_decay", math.nan), 0.999, rel_tol=0, abs_tol=5e-8):
            raise ValueError("v4 canary did not apply EMA 0.999")
        if float(record.get("ema_formula_residual", math.inf)) > 1e-8:
            raise ValueError("v4 canary EMA differs from its exact 0.999 update formula")
        if not math.isclose(record["applied_lr_ratio"], 6.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("v4 canary did not apply exact 6x new-module LR ratio")
        if float(record.get("valid_target_count", 0)) <= 0:
            raise ValueError("v4 canary did not exercise a valid JEPA target")
        expected = record["action_loss"] + record["jepa_weighted"] + record["visreg_weighted"]
        if not math.isclose(record["total_loss"], expected, rel_tol=2e-4, abs_tol=2e-5):
            raise ValueError("v4 canary loss does not decompose as action+JEPA+VISReg")
        if record["production_loss_parity_error"] > max(2e-5, 2e-4 * abs(record["total_loss"])):
            raise ValueError("production and diagnostic v4 losses differ")


def run(
    manifest: dict[str, Any],
    proof: Path,
    source_proof: dict[str, Any],
    gpu_names: list[str],
    gpu_uuids: list[str],
    gpu_memory_mib: list[int],
) -> None:
    import functools

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import openpi.training.checkpoints as checkpoints
    import openpi.training.data_loader as data_loader_api
    import openpi.training.sharding as sharding
    import optax
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.wsm_jepa import wsm_jepa_aux_loss

    from .config import build_train_config, validate_train_config
    from .data import install_data_loader_patch
    from .differential_lr import install_parameter_group_optimizer
    from .train import _openpi_train_module

    validate_h100_node_topology(
        device_count=jax.device_count(),
        process_count=jax.process_count(),
        gpu_names=gpu_names,
        gpu_uuids=gpu_uuids,
        gpu_memory_mib=gpu_memory_mib,
    )
    jax_inventory = [
        {"id": int(device.id), "platform": device.platform, "device_kind": device.device_kind}
        for device in jax.devices()
    ]
    if any(x["platform"] != "gpu" or not is_h100_device_name(x["device_kind"]) for x in jax_inventory):
        raise ValueError(f"JAX did not bind exact 8-H100 topology: {jax_inventory}")
    install_data_loader_patch()
    config = build_train_config(ARM)
    validate_train_config(config, ARM)
    if (
        config.num_train_steps != 2
        or config.save_interval != 2
        or config.batch_size != 64
        or str(config.model.dtype) != "bfloat16"
        or config.model.wsm_cond_type != "gated_deltanet"
        or config.model.wsm_cond_window != 8
        or getattr(config.model, "wsm_cond_history_dropout", 0.0) != 0.0
        or config.model.wsm_jepa_regularizer != "visreg"
        or config.model.wsm_jepa_sigreg_weight != 0.0
        or config.model.wsm_jepa_visreg_weight != 0.05
    ):
        raise ValueError("runtime v4 canary config drifted")
    optimizer_recipe = install_parameter_group_optimizer(ARM)
    train_module = _openpi_train_module()
    train_module.init_logging()

    def diagnostic_forward(model: Any, rng: Any, observation: Any, actions: Any):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_api.preprocess_observation(preprocess_rng, observation, train=True)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        workspace_vec = model._current_workspace_vec(observation, train=True, rng=None, force_uncond=False)
        adarms_cond = adarms_cond + workspace_vec.astype(adarms_cond.dtype)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        action = jnp.mean(jnp.mean(jnp.square(v_t - u_t), axis=-1))
        common = dict(
            head=model.wsm_jepa_head,
            penult_act=suffix_out[:, -model.action_horizon :],
            w_target=observation.wsm_w_target,
            target_valid=observation.wsm_w_target_valid,
            rng=rng,
            num_futures=model.wsm_jepa_num_futures,
            regularizer="visreg",
            visreg_num_slices=model.wsm_jepa_visreg_slices,
            visreg_scale_weight=model.wsm_jepa_visreg_scale_weight,
            visreg_shape_weight=model.wsm_jepa_visreg_shape_weight,
            visreg_center_weight=model.wsm_jepa_visreg_center_weight,
        )
        jepa = wsm_jepa_aux_loss(**common, jepa_weight=1.0, sigreg_weight=0.0, visreg_weight=0.0)
        visreg = wsm_jepa_aux_loss(**common, jepa_weight=0.0, sigreg_weight=0.0, visreg_weight=1.0)
        total = action + model.wsm_jepa_weight * jepa + model.wsm_jepa_visreg_weight * visreg
        return total, {
            "total_loss": total,
            "action_loss": action,
            "jepa_loss": jepa,
            "jepa_weighted": model.wsm_jepa_weight * jepa,
            "visreg_loss": visreg,
            "visreg_weighted": model.wsm_jepa_visreg_weight * visreg,
            "valid_target_count": observation.wsm_w_target_valid.astype(jnp.float32).sum(),
        }

    def train_step(config: Any, rng: Any, state: Any, batch: Any):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model: Any, step_rng: Any, observation: Any, actions: Any):
            production = jnp.mean(model.compute_loss(step_rng, observation, actions, train=True))
            diagnostic, values = diagnostic_forward(model, step_rng, observation, actions)
            return production, {
                **values,
                "total_loss": production,
                "production_loss_parity_error": jnp.abs(production - diagnostic),
            }

        step_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, config.trainable_filter)
        (_loss, values), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, step_rng, observation, actions
        )
        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        updated = optax.apply_updates(params, updates)
        delta = jax.tree.map(lambda old, new: new - old, params, updated)
        old_ema = state.ema_params
        nnx.update(model, updated)
        new_params = nnx.state(model)
        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
        new_module_sq = (
            _subtree_norm(grads, "wsm_tanh_cond", jax, jnp, optax) ** 2
            + _subtree_norm(grads, "wsm_jepa_head", jax, jnp, optax) ** 2
        )
        global_sq = optax.global_norm(grads) ** 2
        backbone_lr = config.lr_schedule.create()(state.step)
        new_module_lr = optax.warmup_cosine_decay_schedule(
            init_value=3e-4 / (config.lr_schedule.warmup_steps + 1),
            peak_value=3e-4,
            warmup_steps=config.lr_schedule.warmup_steps,
            decay_steps=config.lr_schedule.decay_steps,
            end_value=3e-5,
        )(state.step)
        return new_state, {
            **values,
            "grad_norm": optax.global_norm(grads),
            "backbone_grad_norm": jnp.sqrt(jnp.maximum(global_sq - new_module_sq, 0)),
            "gdn_grad_norm": _subtree_norm(grads, "wsm_tanh_cond", jax, jnp, optax),
            "jepa_head_grad_norm": _subtree_norm(grads, "wsm_jepa_head", jax, jnp, optax),
            "update_norm": optax.global_norm(updates),
            "parameter_delta_norm": optax.global_norm(delta),
            "backbone_update_norm": jnp.sqrt(
                jnp.maximum(
                    optax.global_norm(updates) ** 2
                    - _subtree_norm(updates, "wsm_tanh_cond", jax, jnp, optax) ** 2
                    - _subtree_norm(updates, "wsm_jepa_head", jax, jnp, optax) ** 2,
                    0,
                )
            ),
            "gdn_update_norm": _subtree_norm(updates, "wsm_tanh_cond", jax, jnp, optax),
            "jepa_head_update_norm": _subtree_norm(updates, "wsm_jepa_head", jax, jnp, optax),
            "ema_delta_norm": optax.global_norm(
                jax.tree.map(lambda old, new: new - old, old_ema, new_state.ema_params)
            ),
            "ema_decay": jnp.asarray(state.ema_decay),
            "ema_formula_residual": optax.global_norm(
                jax.tree.map(
                    lambda actual, old, new: actual - (state.ema_decay * old + (1 - state.ema_decay) * new),
                    new_state.ema_params,
                    old_ema,
                    new_params,
                )
            ),
            "backbone_applied_lr": backbone_lr,
            "new_module_applied_lr": new_module_lr,
            "applied_lr_ratio": new_module_lr / backbone_lr,
            "parameters_finite": _finite(new_params, jax, jnp),
            "optimizer_finite": _finite(new_opt_state, jax, jnp),
            "state_step": new_state.step,
        }

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=None, overwrite=False, resume=False
    )
    if resuming:
        raise ValueError("v4 canary unexpectedly entered resume mode")
    loader = data_loader_api.create_data_loader(config, sharding=data_sharding, shuffle=True)
    iterator = iter(loader)
    state, state_sharding = train_module.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    parameter_route_counts = {"backbone": 0, "new_module": 0}
    observed_new_subtrees: set[str] = set()
    for path, _leaf in jax.tree_util.tree_flatten_with_path(state.params.filter(config.trainable_filter))[0]:
        head = str(getattr(path[0], "key", path[0]))
        route = "new_module" if head in {"wsm_tanh_cond", "wsm_jepa_head"} else "backbone"
        parameter_route_counts[route] += 1
        if route == "new_module":
            observed_new_subtrees.add(head)
    if observed_new_subtrees != {"wsm_tanh_cond", "wsm_jepa_head"} or not all(parameter_route_counts.values()):
        raise ValueError(f"v4 differential-LR path routing drifted: {observed_new_subtrees}/{parameter_route_counts}")
    raw_probe = {"x": jnp.asarray([12.0, 16.0])}
    clip_tx = optax.clip_by_global_norm(10.0)
    clipped_probe, _ = clip_tx.update(raw_probe, clip_tx.init(raw_probe))
    import openpi.training.optimizer as optimizer_module

    grouped_tx = optimizer_module.create_optimizer(config.optimizer, config.lr_schedule)
    probe_params = {
        "backbone_probe": jnp.asarray(1.0),
        "wsm_tanh_cond": jnp.asarray(1.0),
        "wsm_jepa_head": jnp.asarray(1.0),
    }
    zero_grads = jax.tree.map(jnp.zeros_like, probe_params)
    wd_updates, _ = grouped_tx.update(zero_grads, grouped_tx.init(probe_params), probe_params)
    wd_backbone = abs(float(wd_updates["backbone_probe"]))
    wd_gdn = abs(float(wd_updates["wsm_tanh_cond"]))
    wd_jepa = abs(float(wd_updates["wsm_jepa_head"]))
    optimizer_semantics_probe = {
        "forced_raw_gradient_norm": float(optax.global_norm(raw_probe)),
        "forced_post_global_clip_norm": float(optax.global_norm(clipped_probe)),
        "global_clip_threshold": 10.0,
        "zero_gradient_nonzero_parameter_weight_decay": {
            "backbone_update_abs": wd_backbone,
            "gdn_update_abs": wd_gdn,
            "jepa_update_abs": wd_jepa,
            "new_to_backbone_ratio": wd_gdn / wd_backbone,
        },
    }
    if (
        not math.isclose(optimizer_semantics_probe["forced_raw_gradient_norm"], 20.0, rel_tol=1e-6)
        or not math.isclose(optimizer_semantics_probe["forced_post_global_clip_norm"], 10.0, rel_tol=1e-6)
        or min(wd_backbone, wd_gdn, wd_jepa) <= 0
        or not math.isclose(wd_gdn / wd_backbone, 6.0, rel_tol=1e-5)
        or not math.isclose(wd_jepa / wd_backbone, 6.0, rel_tol=1e-5)
    ):
        raise ValueError(f"v4 optimizer executable semantics probe failed: {optimizer_semantics_probe}")
    pstep = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    records = []
    for step in range(STEPS):
        batch = next(iterator)
        with sharding.set_mesh(mesh):
            state, info = pstep(train_rng, state, batch)
        reduced = jax.device_get(info)
        if int(reduced.pop("state_step")) != step + 1:
            raise ValueError("optimizer step did not advance")
        records.append(
            {
                "step": step,
                **{name: bool(value) if name.endswith("_finite") else float(value) for name, value in reduced.items()},
            }
        )
    _validate_records(records)
    checkpoints.save_state(manager, state, loader, FINAL_STEP)
    manager.wait_until_finished()
    checkpoint_dir = Path(config.checkpoint_dir) / str(FINAL_STEP)
    checkpoint_tree = strong_checkpoint_tree(checkpoint_dir)
    saved_step = int(jax.device_get(state.step))
    del pstep, iterator, batch, info, reduced, state_sharding, data_sharding, replicated
    gc.collect()
    jax.clear_caches()
    gc.collect()
    saved_fingerprints = fingerprint_train_state_components(state)
    del state
    gc.collect()
    template, _restore_sharding = train_module.init_train_state(config, init_rng, mesh, resume=True)
    restored = checkpoints.restore_state(manager, template, loader, step=FINAL_STEP)
    # The restore call has materialized its result; release the full template before blocking or
    # fingerprinting the restored state, preserving the one-full-state device-memory contract.
    del template, _restore_sharding
    gc.collect()
    jax.block_until_ready(restored)
    del manager, loader
    gc.collect()
    restored_step = int(jax.device_get(restored.step))
    restored_fingerprints = fingerprint_train_state_components(restored)
    restore_proof = build_restore_fingerprint_proof(saved_fingerprints, restored_fingerprints)
    if saved_step != STEPS or restored_step != STEPS or restored.ema_params is None:
        raise ValueError("v4 canary save/cold-restore state drifted")
    training_environment: dict[str, Any] = {}
    if os.environ.get("SM_TRAINING_ENV"):
        training_environment = json.loads(os.environ["SM_TRAINING_ENV"])
    instance_type = os.environ.get("SM_CURRENT_INSTANCE_TYPE") or training_environment.get("resource_config", {}).get(
        "current_instance_type"
    )
    receipt = seal(
        {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "claim": CLAIM,
            "contract": CONTRACT,
            "canary_id": manifest["canary_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "identity": manifest["identity"],
            "identity_sha256": manifest["identity_sha256"],
            "source": source_proof,
            "arm": ARM,
            "task": "PickXtimes",
            "training_job_name": os.environ.get("TRAINING_JOB_NAME") or os.environ.get("SM_TRAINING_JOB_NAME"),
            "runtime": {
                "instance_type": instance_type,
                "jax_processes": jax.process_count(),
                "jax_inventory": jax_inventory,
                "gpu_names": gpu_names,
                "gpu_uuids": gpu_uuids,
                "gpu_memory_mib": gpu_memory_mib,
                "dtype": "bfloat16",
            },
            "optimizer_recipe": optimizer_recipe,
            "optimizer_semantics_probe": optimizer_semantics_probe,
            "parameter_routing": {
                "labels": {
                    "wsm_tanh_cond": "new_module",
                    "wsm_jepa_head": "new_module",
                    "all_other_parameter_paths": "backbone",
                },
                "leaf_counts": parameter_route_counts,
            },
            "ema_decay": 0.999,
            "records": records,
            "checkpoint": {
                "scope": "node_local_ephemeral_only",
                "step": FINAL_STEP,
                "saved_state_step": saved_step,
                "restored_state_step": restored_step,
                "tree": checkpoint_tree,
                "restore_fingerprints": restore_proof,
                "published": False,
            },
            "production_checkpoint_or_deploy_publication": False,
        },
        "receipt_sha256",
    )
    proof.write_text(canonical(receipt) + "\n", encoding="utf-8")


def validate_receipt(value: dict[str, Any], manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    claimed = require_sha(value.get("receipt_sha256"), "receipt_sha256")
    clean = dict(value)
    clean.pop("receipt_sha256")
    if (
        sha_json(clean) != claimed
        or value.get("kind") != RECEIPT_KIND
        or value.get("claim") != CLAIM
        or value.get("contract") != CONTRACT
        or value.get("arm") != ARM
        or value.get("task") != "PickXtimes"
    ):
        raise ValueError("v4 canary receipt contract/seal mismatch")
    identity = value.get("identity")
    identity_sha = value.get("identity_sha256")
    if not isinstance(identity, dict) or sha_json(identity) != identity_sha:
        raise ValueError("v4 canary receipt identity seal mismatch")
    if value.get("canary_id") != f"v4-policy-canary-{identity_sha[:20]}":
        raise ValueError("v4 canary receipt id is not identity-derived")
    if value.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("v4 canary receipt manifest identity drifted")
    if identity != manifest.get("identity") or identity_sha != manifest.get("identity_sha256"):
        raise ValueError("v4 canary receipt does not bind the exact staged manifest identity")
    source = value.get("source", {})
    expected_source_proof = {
        "prepared_source_tree_sha256": identity["prepared_source_tree_sha256"],
        "entry": ENTRY,
        "entry_bytes_sha256": identity["entry_bytes_sha256"],
        "runtime_entry_mode": "0o777",
        "staged_manifest_excluded_from_source_identity": STAGED_MANIFEST,
        "submitted_entry_mode": "0o755",
        "sagemaker_runtime_entry_mode": "0o777",
    }
    if source != expected_source_proof:
        raise ValueError("v4 canary receipt source identity drifted")
    runtime = value.get("runtime", {})
    inventory = runtime.get("jax_inventory")
    if (
        runtime.get("instance_type") != "ml.p5.48xlarge"
        or runtime.get("jax_processes") != 1
        or runtime.get("dtype") != "bfloat16"
        or not isinstance(inventory, list)
        or len(inventory) != 8
        or any(x.get("platform") != "gpu" or not is_h100_device_name(x.get("device_kind")) for x in inventory)
        or len(runtime.get("gpu_names", [])) != 8
        or any(not is_h100_device_name(name) for name in runtime.get("gpu_names", []))
        or len(set(runtime.get("gpu_uuids", []))) != 8
        or len(runtime.get("gpu_memory_mib", [])) != 8
        or any(value < 80_000 or value > 90_000 for value in runtime.get("gpu_memory_mib", []))
        or not value.get("training_job_name")
    ):
        raise ValueError("v4 canary receipt topology/BF16 drifted")
    expected_recipe = {
        "schema_version": 1,
        "protocol": "robomme_v4",
        "arm": ARM,
        "global_clip_before_parameter_groups": 10.0,
        "weight_decay": 1e-6,
        "backbone_lr": {"peak": 5e-5, "decay": 5e-6},
        "new_module_lr": {"peak": 3e-4, "decay": 3e-5},
        "new_parameter_subtrees": ["wsm_tanh_cond", "wsm_jepa_head"],
        "new_parameter_group_active": True,
    }
    if value.get("optimizer_recipe") != expected_recipe or value.get("ema_decay") != 0.999:
        raise ValueError("v4 canary receipt optimizer/EMA recipe drifted")
    semantics = value.get("optimizer_semantics_probe", {})
    wd = semantics.get("zero_gradient_nonzero_parameter_weight_decay", {})
    if (
        not math.isclose(semantics.get("forced_raw_gradient_norm", math.nan), 20.0, rel_tol=1e-6)
        or not math.isclose(semantics.get("forced_post_global_clip_norm", math.nan), 10.0, rel_tol=1e-6)
        or semantics.get("global_clip_threshold") != 10.0
        or min(
            wd.get("backbone_update_abs", 0),
            wd.get("gdn_update_abs", 0),
            wd.get("jepa_update_abs", 0),
        )
        <= 0
        or not math.isclose(wd.get("new_to_backbone_ratio", math.nan), 6.0, rel_tol=1e-5)
    ):
        raise ValueError("v4 canary receipt optimizer executable semantics probe drifted")
    routing = value.get("parameter_routing", {})
    if routing.get("labels") != {
        "wsm_tanh_cond": "new_module",
        "wsm_jepa_head": "new_module",
        "all_other_parameter_paths": "backbone",
    } or any(int(routing.get("leaf_counts", {}).get(name, 0)) < 1 for name in ("backbone", "new_module")):
        raise ValueError("v4 canary receipt parameter routing drifted")
    _validate_records(value.get("records", []))
    checkpoint = value.get("checkpoint", {})
    if (
        checkpoint.get("scope") != "node_local_ephemeral_only"
        or checkpoint.get("step") != FINAL_STEP
        or checkpoint.get("saved_state_step") != STEPS
        or checkpoint.get("restored_state_step") != STEPS
        or checkpoint.get("published") is not False
        or value.get("production_checkpoint_or_deploy_publication") is not False
    ):
        raise ValueError("v4 canary receipt checkpoint/nonpublication contract drifted")
    restore = checkpoint.get("restore_fingerprints")
    from .policy_canary import validate_restore_fingerprint_proof

    validate_restore_fingerprint_proof(restore)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--sha256", required=True)
    validate_environment_parser = sub.add_parser("validate-environment")
    validate_environment_parser.add_argument("--manifest", type=Path, required=True)
    validate_environment_parser.add_argument("--sha256", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--sha256", required=True)
    run_parser.add_argument("--code-dir", type=Path, required=True)
    run_parser.add_argument("--proof", type=Path, required=True)
    run_parser.add_argument("--gpu-name", action="append", default=[])
    run_parser.add_argument("--gpu-uuid", action="append", default=[])
    run_parser.add_argument("--gpu-memory-mib", action="append", type=int, default=[])
    receipt_parser = sub.add_parser("validate-receipt")
    receipt_parser.add_argument("--receipt", type=Path, required=True)
    receipt_parser.add_argument("--manifest", type=Path, required=True)
    receipt_parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "validate-environment":
        manifest = load_manifest(args.manifest, args.sha256)
        validate_environment(manifest, dict(os.environ))
        return
    if args.command == "validate-receipt":
        manifest = load_manifest(args.manifest, args.sha256)
        validate_receipt(json.loads(args.receipt.read_text(encoding="utf-8")), manifest)
        return
    manifest = load_manifest(args.manifest, args.sha256)
    if args.command == "validate-manifest":
        return
    source_proof = verify_source(args.code_dir.resolve(), manifest)
    run(manifest, args.proof, source_proof, args.gpu_name, args.gpu_uuid, args.gpu_memory_mib)
    validate_receipt(json.loads(args.proof.read_text(encoding="utf-8")), manifest)


if __name__ == "__main__":
    main()
