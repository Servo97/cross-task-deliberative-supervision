"""Task-bound source/GPU canary and receipt validator for MoveCube workspace v2.

The receipt is operational evidence only, never a RoboMME score or a production workspace claim.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from robomme_integration.move_workspace_dense_v2_launch import source_identity

from .workspace_deliberative import _train_step_ema
from .workspace_deliberative_dense_v2 import (
    FEATURE_DIM,
    INPUT_DIM,
    MAX_EVENTS,
    PROTOCOL,
    dense_v2_loss_and_metrics,
    init_params_dense_v2,
    visreg_loss,
)
from .workspace_gpu_producer_dense_v2 import CAMPAIGN, TASK, validate_manifest
from .workspace_supervision_dense_v2 import (
    ARTIFACT,
    GroundedPoint,
    chronological_dense_events,
    dense_patch_distribution,
)

KIND = "robomme_move_workspace_dense_v2_canary_receipt"
H100_DEVICE_PATTERN = re.compile(r"NVIDIA H100 80GB HBM3")


def is_h100_device_name(value: object) -> bool:
    return isinstance(value, str) and H100_DEVICE_PATTERN.fullmatch(value.strip()) is not None


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _semantic_proof() -> dict:
    fixtures = [
        "pick at <10, 20>",
        "move from <9, 7> to <220, 201>",
        "visit <1, 2>, then <33, 44>, finally <255, 255>",
    ]
    counts = []
    fingerprints = []
    for text in fixtures:
        event = chronological_dense_events(["fixture"], [text])[0]
        counts.append(event["target_count"])
        matrix = np.stack(
            [
                dense_patch_distribution(
                    GroundedPoint(
                        order=role["role_index"],
                        x=role["point_xy"][0],
                        y=role["point_xy"][1],
                    )
                )
                for role in event["roles"]
            ]
        )
        if not np.all(matrix > 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("canary dense target proof failed")
        fingerprints.append(hashlib.sha256(matrix.tobytes()).hexdigest())
    forward = chronological_dense_events(["move"], [fixtures[1]])[0]
    reverse = chronological_dense_events(["move"], ["move from <220, 201> to <9, 7>"])[0]
    if [role["point_xy"] for role in forward["roles"]] == [role["point_xy"] for role in reverse["roles"]]:
        raise RuntimeError("canary ordering proof failed")
    return {
        "point_counts": counts,
        "dense_target_sha256": fingerprints,
        "wrong_order_changes_roles": True,
        "all_grid_weights_positive": True,
        "all_grid_weights_normalized": True,
    }


def run_canary(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    source_dir: Path,
    expected_devices: int,
    required_platform: str,
) -> dict:
    import jax
    import jax.numpy as jnp
    import optax

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_source_sha = source_identity(source_dir)
    validate_manifest(
        manifest,
        manifest_sha256=manifest_sha256,
        expected_source_tree_sha256=actual_source_sha,
        expected_entry_sha256=hashlib.sha256(
            (source_dir / "gpu_move_workspace_dense_v2_entry.sh").read_bytes()
        ).hexdigest(),
    )
    devices = jax.devices()
    platforms = {device.platform for device in devices}
    device_kinds = [str(getattr(device, "device_kind", "unknown")) for device in devices]
    if (
        expected_devices != 8
        or required_platform != "gpu"
        or len(devices) != 8
        or platforms != {"gpu"}
        or any(not is_h100_device_name(kind) for kind in device_kinds)
    ):
        raise SystemExit(f"canary requires exact 8xH100 GPU runtime, got devices={devices} kinds={device_kinds}")
    features = jax.random.normal(jax.random.key(11), (64, 32))

    def objective(value):
        return visreg_loss(value, jax.random.key(12), num_slices=128)

    value, gradient = jax.value_and_grad(objective)(features)
    doubled = jnp.concatenate([features, features], axis=0)
    doubled_value = objective(doubled)
    gradient_norm = jnp.linalg.norm(gradient)
    if not (
        np.isfinite(float(value))
        and np.isfinite(float(gradient_norm))
        and float(gradient_norm) > 0
        and np.isclose(float(value), float(doubled_value), rtol=0.08, atol=0.0)
    ):
        raise RuntimeError("VISReg canary numerical proof failed")
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(3e-4, weight_decay=1e-6),
    )
    params = init_params_dense_v2(jax.random.key(21))
    ema_params = jax.tree.map(lambda item: jnp.array(item), params)
    opt_state = optimizer.init(params)
    weights = {
        "occ": 0.1,
        "attention": 0.1,
        "jepa": 0.1,
        "sigreg": 0.0,
        "visreg": 0.05,
        "visreg_slices": 128,
        "visreg_scale": 1.0,
        "visreg_shape": 1.0,
        "visreg_center": 1.0,
    }
    optimizer_step = jax.pmap(
        functools.partial(
            _train_step_ema,
            optimizer=optimizer,
            weights=weights,
            loss_function=dense_v2_loss_and_metrics,
            ema_decay=0.999,
        ),
        axis_name="devices",
        in_axes=(None, None, None, 0),
        out_axes=(None, None, None, 0),
    )
    batch_rng = np.random.default_rng(22)
    per_device = 64 // len(devices)
    event_attention = batch_rng.random((len(devices), per_device, MAX_EVENTS, 64), dtype=np.float32)
    event_attention /= event_attention.sum(axis=-1, keepdims=True)
    event_presence = np.zeros((len(devices), per_device, MAX_EVENTS), dtype=np.float32)
    event_presence[..., :3] = 1.0
    batch = {
        "history": batch_rng.standard_normal((len(devices), per_device, 4, INPUT_DIM), dtype=np.float32),
        "history_mask": np.ones((len(devices), per_device, 4), dtype=np.bool_),
        "event_target": batch_rng.standard_normal(
            (len(devices), per_device, MAX_EVENTS, FEATURE_DIM), dtype=np.float32
        ),
        "event_attention": event_attention,
        "event_presence": event_presence,
        "future_target": batch_rng.standard_normal((len(devices), per_device, FEATURE_DIM), dtype=np.float32),
    }
    initial_params = jax.tree.map(lambda item: jnp.array(item), params)
    metrics = None
    for _ in range(2):
        params, opt_state, ema_params, metrics = optimizer_step(params, opt_state, ema_params, batch)

    def tree_delta_norm(left, right):
        leaves = jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum(jnp.square(a - b)), left, right))
        return float(jnp.sqrt(jnp.sum(jnp.stack(leaves))))

    raw_update_norm = tree_delta_norm(params, initial_params)
    ema_update_norm = tree_delta_norm(ema_params, initial_params)
    ema_raw_delta = tree_delta_norm(ema_params, params)
    attention_update_norm = float(jnp.linalg.norm(params["dense_attention_w"] - initial_params["dense_attention_w"]))
    input_update_norm = float(jnp.linalg.norm(params["input_w"] - initial_params["input_w"]))
    metrics = {key: float(np.asarray(item[0])) for key, item in metrics.items()}
    if not (
        np.isfinite(raw_update_norm)
        and np.isfinite(ema_update_norm)
        and raw_update_norm > 0
        and 0 < ema_update_norm < raw_update_norm
        and ema_raw_delta > 0
        and attention_update_norm > 0
        and input_update_norm > 0
        and all(np.isfinite(metric) for metric in metrics.values())
        and metrics["grad_norm"] > 0
        and metrics["attention"] > 0
    ):
        raise RuntimeError("actual dense-v2 optimizer/EMA canary proof failed")
    receipt = {
        "schema_version": 2,
        "kind": KIND,
        "evidence_class": "operational_canary_not_scientific_evidence",
        "identity": {
            "campaign": CAMPAIGN,
            "task": TASK,
            "run_id": manifest["identity"]["run_id"],
            "attempt_id": manifest["identity"]["attempt_id"],
            "manifest_sha256": manifest_sha256,
            "source_tree_sha256": actual_source_sha,
        },
        "protocol": {
            "name": PROTOCOL,
            "supervision_artifact": ARTIFACT,
            "regularizer": "visreg",
            "visreg_weight": 0.05,
            "visreg_slices": 128,
            "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
            "sigreg_weight": 0.0,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 3e-4,
                "weight_decay": 1e-6,
                "global_gradient_clip": 10.0,
                "ema_decay": 0.999,
            },
        },
        "runtime": {
            "platform": "gpu",
            "device_count": len(devices),
            "device_kinds": device_kinds,
        },
        "proof": {
            "grounding": _semantic_proof(),
            "visreg_value": float(value),
            "visreg_doubled_value": float(doubled_value),
            "visreg_gradient_norm": float(gradient_norm),
            "sample_count_invariant": True,
            "sigreg_executed": False,
            "optimizer_steps": 2,
            "actual_dense_v2_batch_size": 64,
            "raw_update_norm": raw_update_norm,
            "ema_update_norm": ema_update_norm,
            "ema_raw_delta": ema_raw_delta,
            "dense_attention_update_norm": attention_update_norm,
            "encoder_input_update_norm": input_update_norm,
            "step2_metrics": metrics,
            "checkpoint_written": False,
            "cloud_artifact_published": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def validate_receipt(receipt: dict, *, manifest: dict, manifest_sha256: str) -> None:
    actual_manifest_sha = hashlib.sha256(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    if actual_manifest_sha != manifest_sha256:
        raise ValueError("dense v2 canary manifest argument SHA-256 mismatch")
    validate_manifest(
        manifest,
        manifest_sha256=manifest_sha256,
        expected_source_tree_sha256=manifest["source"]["source_tree_sha256"],
        expected_entry_sha256=manifest["source"]["entry_sha256"],
    )
    value = dict(receipt)
    claimed_sha = value.pop("receipt_sha256", None)
    if claimed_sha != hashlib.sha256(_canonical(value)).hexdigest():
        raise ValueError("dense v2 canary receipt SHA-256 mismatch")
    identity = receipt.get("identity", {})
    if (
        receipt.get("schema_version") != 2
        or receipt.get("kind") != KIND
        or receipt.get("evidence_class") != "operational_canary_not_scientific_evidence"
        or identity.get("campaign") != CAMPAIGN
        or identity.get("task") != TASK
        or identity.get("run_id") != manifest["identity"]["run_id"]
        or identity.get("attempt_id") != manifest["identity"]["attempt_id"]
        or identity.get("manifest_sha256") != manifest_sha256
        or identity.get("source_tree_sha256") != manifest["source"]["source_tree_sha256"]
    ):
        raise ValueError("dense v2 canary receipt identity mismatch")
    runtime = receipt.get("runtime", {})
    kinds = runtime.get("device_kinds")
    if (
        runtime.get("platform") != "gpu"
        or runtime.get("device_count") != 8
        or not isinstance(kinds, list)
        or len(kinds) != 8
        or any(not is_h100_device_name(kind) for kind in kinds)
    ):
        raise ValueError("dense v2 canary receipt is not exact 8xH100")
    if receipt.get("protocol") != {
        "name": PROTOCOL,
        "supervision_artifact": ARTIFACT,
        "regularizer": "visreg",
        "visreg_weight": 0.05,
        "visreg_slices": 128,
        "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
        "sigreg_weight": 0.0,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "global_gradient_clip": 10.0,
            "ema_decay": 0.999,
        },
    }:
        raise ValueError("dense v2 canary receipt protocol mismatch")
    proof = receipt.get("proof", {})
    grounding = proof.get("grounding", {})
    target_hashes = grounding.get("dense_target_sha256")
    finite_fields = (
        "visreg_value",
        "visreg_doubled_value",
        "visreg_gradient_norm",
        "raw_update_norm",
        "ema_update_norm",
        "ema_raw_delta",
        "dense_attention_update_norm",
        "encoder_input_update_norm",
    )
    finite = all(np.isfinite(float(proof.get(field, float("nan")))) for field in finite_fields)
    metrics = proof.get("step2_metrics")
    metrics_finite = (
        isinstance(metrics, dict) and metrics and all(np.isfinite(float(value)) for value in metrics.values())
    )
    values_invariant = finite and np.isclose(
        float(proof["visreg_value"]),
        float(proof["visreg_doubled_value"]),
        rtol=0.08,
        atol=0.0,
    )
    if (
        grounding.get("point_counts") != [1, 2, 3]
        or not isinstance(target_hashes, list)
        or len(target_hashes) != 3
        or any(
            not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in target_hashes
        )
        or grounding.get("wrong_order_changes_roles") is not True
        or grounding.get("all_grid_weights_positive") is not True
        or grounding.get("all_grid_weights_normalized") is not True
        or proof.get("sample_count_invariant") is not True
        or not values_invariant
        or not finite
        or proof.get("sigreg_executed") is not False
        or proof.get("optimizer_steps") != 2
        or proof.get("actual_dense_v2_batch_size") != 64
        or not 0 < float(proof.get("ema_update_norm", 0)) < float(proof.get("raw_update_norm", 0))
        or float(proof.get("ema_raw_delta", 0)) <= 0
        or float(proof.get("dense_attention_update_norm", 0)) <= 0
        or float(proof.get("encoder_input_update_norm", 0)) <= 0
        or not metrics_finite
        or float(metrics.get("grad_norm", 0)) <= 0
        or float(metrics.get("attention", 0)) <= 0
        or proof.get("checkpoint_written") is not False
        or proof.get("cloud_artifact_published") is not False
        or float(proof["visreg_gradient_norm"]) <= 0
    ):
        raise ValueError("dense v2 canary receipt proof mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--required-platform", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    receipt = run_canary(
        manifest_path=Path(args.manifest),
        manifest_sha256=args.manifest_sha256,
        source_dir=Path(args.source_dir),
        expected_devices=args.expected_devices,
        required_platform=args.required_platform,
    )
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_receipt(receipt, manifest=manifest, manifest_sha256=args.manifest_sha256)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
