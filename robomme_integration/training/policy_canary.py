#!/usr/bin/env python3
"""Two-step, non-scientific policy-training canary for the frozen RoboMME stack.

The implementation is intentionally separate from ``training.train``.  It uses the same data
adapter, model, initialization, optimizer, and OpenPI primitives, while returning loss components
as JAX auxiliary values so the scalar differentiated by the canary remains the production scalar.
No callback or leaked tracer is used to recover diagnostics.
"""

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

KIND = "robomme_policy_training_canary_attempt"
RECEIPT_KIND = "robomme_policy_training_canary_complete"
CLAIM = "not_scientific_training_evidence"
ARM = "gdn8_jepa_l01_k1"
CONTRACT_VERSION = "robomme-policy-training-canary-v1"
STEPS = 2
FINAL_STEP = 1
SUBMITTED_ENTRY_MODE = 0o755
SAGEMAKER_RUNTIME_ENTRY_MODE = 0o777
PRODUCTION_SOURCE_SHA256 = "528d2c580dfd43a300fce4046fec7bf8bd39a73f700170bcc81dd4316260de3d"
PRODUCTION_ENTRY_SHA256 = "d8a157dcbb9f8092dcf85bc700dbd4b6eae93c4f5c1f809caf1ddbc2bb34b7ca"
CANARY_ENTRY_SHA256 = "292edebfd1fee4e6a23138be9ad1115e5a50de84f98afd1daf82a002eda41dfc"
REFERENCE_RUN_ID = "st-v1-pickxtimes-gdn8_jepa_l01_k1-seed0-64dee36adb843a2f"
REFERENCE_SCIENTIFIC_SPEC_SHA256 = "64dee36adb843a2ffc548543e39c3c908609044d8c513363025e43d884db109c"
REFERENCE_MANIFEST_SHA256 = "294ea48d79ac664c2efc14089308d0f64bc436deee86fccb3bbab810376f0e8d"
STUDY_ROOT = "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"
REVIEWED_SOURCE_DELTAS = frozenset(
    {
        "gpu_policy_canary_entry.sh",
        "policy_canary_launch.py",
        "tests/test_policy_canary.py",
        "training/policy_canary.py",
    }
)
IDENTITY_COMPONENT_SHA256 = {
    "task": "e60c3e3d70563ef5c985e07d3530bdf0531dda3dd673085a9ad8f2df9f26e953",
    "mechanism": "8e89d35bdd03a3d180228b81524d0dded22fd2ce27eb78a3970055a2a65ab971",
    "data": "23afce0a0de715cf1a434c316ecb735feb552a21910fcba03fcd125a2b482cdc",
    "initialization": "eefa495466a5aa005fc7a4a4ff8b654ab0f5acdded407862b78c4a24c0fb955d",
    "workspace_representation": ("82c853a949c1fc49240485651d95ffae8a3294772aead35c2371f99aadd3c02b"),
    "sources": "a4481eb92ec53f587e663759d8123cb171c510def4db07bfa15f4e98455763af",
    "model_training_identity": ("9263cf6e63e401092eb854c12d3682b8cc76f8eace827346255f05360bf6feb5"),
}
GDN_JEPA_OVERLAY_MANIFEST_SHA256 = "6f885cb5a4f8b266fd444b0736e4ca525250929d06c60c3ca0e383212aa9bbfc"
TREE_FINGERPRINT_ALGORITHM = "jax-tree-path-shape-dtype-c-order-raw-bytes-sha256-v1"
TREE_FINGERPRINT_COMPONENTS = ("params", "ema_params", "optimizer_state")
TREE_FINGERPRINT_MAX_SIMULTANEOUS_HOST_LEAVES = 1
TREE_FINGERPRINT_HASH_CHUNK_BYTES = 8 * 1024 * 1024
TREE_FINGERPRINT_FINITE_CHUNK_BYTES = 8 * 1024 * 1024
FORBIDDEN_ENVIRONMENT_KEYS = frozenset(
    {
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
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _framed_hash_update(digest: Any, value: str | bytes) -> None:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _canonical_tree_path(path: tuple[Any, ...]) -> list[dict[str, str]]:
    """Represent a JAX key path without relying on ambiguous string concatenation."""

    result: list[dict[str, str]] = []
    for key in path:
        value = key
        for attribute in ("key", "idx", "name"):
            if hasattr(key, attribute):
                value = getattr(key, attribute)
                break
        result.append(
            {
                "key_type": f"{type(key).__module__}.{type(key).__qualname__}",
                "value_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value_repr": repr(value),
            }
        )
    return result


def exact_tree_fingerprint(
    tree: Any,
    *,
    _flatten_with_path: Any | None = None,
    _device_get: Any | None = None,
) -> dict[str, Any]:
    """Fingerprint a device tree exactly while materializing at most one host leaf.

    The function deliberately performs no JAX arithmetic or whole-tree transfer. Each leaf is
    copied to the host independently, scanned for finiteness in bounded chunks, hashed in logical
    C order, and released before the next device leaf is requested. The injectable callables are
    only for a dependency-light regression that proves the sequential lifetime contract.
    """

    import numpy as np

    if _flatten_with_path is None or _device_get is None:
        import jax

        if _flatten_with_path is None:
            _flatten_with_path = jax.tree_util.tree_flatten_with_path
        if _device_get is None:
            _device_get = jax.device_get

    path_leaves, tree_definition = _flatten_with_path(tree)
    structure_sha = hashlib.sha256(repr(tree_definition).encode("utf-8")).hexdigest()
    metadata_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    leaf_records_digest = hashlib.sha256()
    total_bytes = 0
    largest_leaf_bytes = 0
    all_finite = True

    for index, (path, leaf) in enumerate(path_leaves):
        host_value = _device_get(leaf)
        host_array = np.asarray(host_value)
        if host_array.dtype.hasobject:
            raise ValueError(f"tree leaf {index} has unsupported object dtype")
        if not host_array.flags.c_contiguous:
            raise ValueError(f"tree leaf {index} is not a single C-contiguous host materialization")
        contiguous = host_array
        # Viewing as uint8 avoids both a copy and NumPy's unsupported PEP-3118 buffer format for
        # ml_dtypes.bfloat16 (the dominant parameter dtype in the pinned OpenPI model).
        raw_byte_array = contiguous.reshape(-1).view(np.uint8).reshape(-1)
        byte_view = memoryview(raw_byte_array)
        leaf_content_digest = hashlib.sha256()
        for offset in range(0, byte_view.nbytes, TREE_FINGERPRINT_HASH_CHUNK_BYTES):
            leaf_content_digest.update(byte_view[offset : offset + TREE_FINGERPRINT_HASH_CHUNK_BYTES])
        leaf_content_sha = leaf_content_digest.hexdigest()
        path_record = _canonical_tree_path(tuple(path))
        metadata = {
            "index": index,
            "path": path_record,
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "bytes": byte_view.nbytes,
        }
        content = {
            "index": index,
            "path": path_record,
            "raw_bytes_sha256": leaf_content_sha,
        }
        leaf_record = {**metadata, "raw_bytes_sha256": leaf_content_sha}
        _framed_hash_update(metadata_digest, _canonical_json(metadata))
        _framed_hash_update(content_digest, _canonical_json(content))
        _framed_hash_update(leaf_records_digest, _canonical_json(leaf_record))
        total_bytes += byte_view.nbytes
        largest_leaf_bytes = max(largest_leaf_bytes, byte_view.nbytes)

        dtype_name = str(contiguous.dtype).lower()
        if contiguous.dtype.kind in "fc" or "float" in dtype_name or "complex" in dtype_name:
            flat_view = contiguous.reshape(-1)
            chunk_elements = max(
                1,
                TREE_FINGERPRINT_FINITE_CHUNK_BYTES // max(1, contiguous.dtype.itemsize),
            )
            for offset in range(0, flat_view.size, chunk_elements):
                finite_chunk = np.isfinite(flat_view[offset : offset + chunk_elements])
                if not bool(finite_chunk.all()):
                    all_finite = False
                del finite_chunk
            del flat_view

        # These explicit releases are part of the memory contract: the next device_get cannot
        # overlap the lifetime of this host materialization or its bounded finite-scan buffer.
        del byte_view, raw_byte_array, contiguous, host_array, host_value

    core = {
        "algorithm": TREE_FINGERPRINT_ALGORITHM,
        "leaf_count": len(path_leaves),
        "total_bytes": total_bytes,
        "largest_leaf_bytes": largest_leaf_bytes,
        "all_finite": all_finite,
        "structure_sha256": structure_sha,
        "metadata_sha256": metadata_digest.hexdigest(),
        "content_sha256": content_digest.hexdigest(),
        "leaf_records_sha256": leaf_records_digest.hexdigest(),
    }
    del path_leaves
    gc.collect()
    return {**core, "fingerprint_sha256": _sha256_json(core)}


def fingerprint_train_state_components(
    state: Any,
    *,
    _flatten_with_path: Any | None = None,
    _device_get: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Fingerprint params, EMA params, and optimizer state strictly one tree at a time."""

    if state.ema_params is None:
        raise ValueError("GDN+JEPA canary train state has no EMA parameters")
    return {
        "params": exact_tree_fingerprint(
            state.params,
            _flatten_with_path=_flatten_with_path,
            _device_get=_device_get,
        ),
        "ema_params": exact_tree_fingerprint(
            state.ema_params,
            _flatten_with_path=_flatten_with_path,
            _device_get=_device_get,
        ),
        "optimizer_state": exact_tree_fingerprint(
            state.opt_state,
            _flatten_with_path=_flatten_with_path,
            _device_get=_device_get,
        ),
    }


def _validate_tree_fingerprint(value: Any, name: str) -> None:
    expected_fields = {
        "algorithm",
        "leaf_count",
        "total_bytes",
        "largest_leaf_bytes",
        "all_finite",
        "structure_sha256",
        "metadata_sha256",
        "content_sha256",
        "leaf_records_sha256",
        "fingerprint_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{name} fingerprint fields drifted")
    if value.get("algorithm") != TREE_FINGERPRINT_ALGORITHM:
        raise ValueError(f"{name} fingerprint algorithm drifted")
    for field in ("leaf_count", "total_bytes", "largest_leaf_bytes"):
        if not isinstance(value.get(field), int) or isinstance(value.get(field), bool):
            raise ValueError(f"{name} fingerprint {field} is not an integer")
    if (
        value["leaf_count"] < 1
        or value["total_bytes"] < 1
        or value["largest_leaf_bytes"] < 1
        or value["largest_leaf_bytes"] > value["total_bytes"]
    ):
        raise ValueError(f"{name} fingerprint has invalid leaf/byte counts")
    if value.get("all_finite") is not True:
        raise ValueError(f"{name} fingerprint contains non-finite state")
    for field in (
        "structure_sha256",
        "metadata_sha256",
        "content_sha256",
        "leaf_records_sha256",
        "fingerprint_sha256",
    ):
        _require_sha(value.get(field), f"{name} fingerprint {field}")
    core = dict(value)
    claimed = core.pop("fingerprint_sha256")
    if _sha256_json(core) != claimed:
        raise ValueError(f"{name} fingerprint self-seal mismatch")


def build_restore_fingerprint_proof(
    saved: dict[str, dict[str, Any]], restored: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Require exact host-byte equality for every cold-restored train-state component."""

    exact_match = {
        component: saved.get(component) == restored.get(component) for component in TREE_FINGERPRINT_COMPONENTS
    }
    proof = {
        "algorithm": TREE_FINGERPRINT_ALGORITHM,
        "max_simultaneous_host_leaves": TREE_FINGERPRINT_MAX_SIMULTANEOUS_HOST_LEAVES,
        "saved": saved,
        "restored": restored,
        "exact_match": exact_match,
    }
    validate_restore_fingerprint_proof(proof)
    return proof


def validate_restore_fingerprint_proof(proof: Any) -> None:
    expected_fields = {
        "algorithm",
        "max_simultaneous_host_leaves",
        "saved",
        "restored",
        "exact_match",
    }
    if not isinstance(proof, dict) or set(proof) != expected_fields:
        raise ValueError("cold-restore fingerprint proof fields drifted")
    if (
        proof.get("algorithm") != TREE_FINGERPRINT_ALGORITHM
        or proof.get("max_simultaneous_host_leaves") != TREE_FINGERPRINT_MAX_SIMULTANEOUS_HOST_LEAVES
    ):
        raise ValueError("cold-restore fingerprint algorithm/memory bound drifted")
    saved = proof.get("saved")
    restored = proof.get("restored")
    exact_match = proof.get("exact_match")
    component_set = set(TREE_FINGERPRINT_COMPONENTS)
    if (
        not isinstance(saved, dict)
        or set(saved) != component_set
        or not isinstance(restored, dict)
        or set(restored) != component_set
        or not isinstance(exact_match, dict)
        or set(exact_match) != component_set
    ):
        raise ValueError("cold-restore fingerprint components drifted")
    mismatches: list[str] = []
    for component in TREE_FINGERPRINT_COMPONENTS:
        _validate_tree_fingerprint(saved[component], f"saved {component}")
        _validate_tree_fingerprint(restored[component], f"restored {component}")
        actual_match = saved[component] == restored[component]
        if exact_match[component] is not actual_match or not actual_match:
            mismatches.append(component)
    if mismatches:
        raise ValueError(f"cold-restored train state differs from exact saved host-byte fingerprints: {mismatches}")


def source_tree_sha256(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset(),
    mode_overrides: dict[str, int] | None = None,
) -> str:
    """Recompute the launch guardrail identity with explicit runtime-only normalization."""

    root = root.resolve()
    digest = hashlib.sha256()

    def field(value: str | bytes) -> None:
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        mode = path.lstat().st_mode
        field(relative)
        effective_mode = (mode_overrides or {}).get(relative, stat.S_IMODE(mode))
        field(oct(effective_mode))
        if path.is_symlink():
            field("symlink")
            field(os.readlink(path))
        elif path.is_dir():
            field("directory")
        elif path.is_file():
            field("file")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    field(block)
        else:
            raise ValueError(f"unsupported source entry type: {path}")
    return digest.hexdigest()


def verify_packaged_source(
    code_dir: Path,
    manifest: dict[str, Any],
    *,
    staged_manifest: str,
) -> dict[str, Any]:
    """Bind the on-node unpacked bytes to the launch-time canary source receipt."""

    code_dir = code_dir.resolve()
    relative = Path(staged_manifest)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"unsafe staged canary manifest path: {staged_manifest!r}")
    manifest_path = code_dir / relative
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"staged canary manifest absent from packaged source: {manifest_path}")
    source = manifest.get("source", {})
    expected_submitted_tree = _require_sha(source.get("canary_source_tree_sha256"), "canary source_tree_sha256")
    expected_runtime_tree = _require_sha(
        source.get("canary_runtime_source_tree_sha256"),
        "canary runtime source_tree_sha256",
    )
    expected_entry = _require_sha(source.get("entry_sha256"), "canary entry_sha256")
    entry_name = source.get("entry")
    if entry_name != "gpu_policy_canary_entry.sh":
        raise ValueError(f"unexpected canary entry name: {entry_name!r}")
    entry = code_dir / entry_name
    if not entry.is_file() or entry.is_symlink():
        raise ValueError(f"packaged canary entry absent or unsafe: {entry}")
    actual_entry_mode = stat.S_IMODE(entry.lstat().st_mode)
    if (
        source.get("submitted_entry_mode") != SUBMITTED_ENTRY_MODE
        or source.get("sagemaker_runtime_entry_mode") != SAGEMAKER_RUNTIME_ENTRY_MODE
        or actual_entry_mode != SAGEMAKER_RUNTIME_ENTRY_MODE
    ):
        raise ValueError(
            "SageMaker canary entry mode contract drifted: "
            f"submitted={source.get('submitted_entry_mode')!r} "
            f"sealed_runtime={source.get('sagemaker_runtime_entry_mode')!r} "
            f"actual_runtime={oct(actual_entry_mode)}"
        )
    actual_entry = hashlib.sha256(entry.read_bytes()).hexdigest()
    if actual_entry != expected_entry:
        raise ValueError(f"packaged canary entry drifted: {actual_entry} != {expected_entry}")
    actual_runtime_tree = source_tree_sha256(
        code_dir,
        excluded=frozenset({staged_manifest}),
    )
    actual_submitted_tree = source_tree_sha256(
        code_dir,
        excluded=frozenset({staged_manifest}),
        mode_overrides={entry_name: SUBMITTED_ENTRY_MODE},
    )
    if actual_runtime_tree != expected_runtime_tree or actual_submitted_tree != expected_submitted_tree:
        raise ValueError(
            "packaged canary source drifted after exact SageMaker mode normalization: "
            f"runtime={actual_runtime_tree}/{expected_runtime_tree} "
            f"submitted={actual_submitted_tree}/{expected_submitted_tree}"
        )
    return {
        "canary_source_tree_sha256": actual_submitted_tree,
        "canary_runtime_source_tree_sha256": actual_runtime_tree,
        "entry": entry_name,
        "entry_sha256": actual_entry,
        "submitted_entry_mode": SUBMITTED_ENTRY_MODE,
        "sagemaker_runtime_entry_mode": actual_entry_mode,
        "excluded_staged_file": staged_manifest,
    }


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canary manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("canary manifest must be a JSON object")
    claimed = value.get("manifest_sha256")
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    actual = _sha256_json(clean)
    if claimed != expected_sha256 or actual != expected_sha256:
        raise ValueError(
            f"canary manifest seal mismatch: claimed={claimed} actual={actual} expected={expected_sha256}"
        )
    validate_manifest_contract(value)
    return value


def validate_manifest_contract(manifest: dict[str, Any]) -> None:
    """Authenticate the complete standalone launch contract without trusting its self-seal."""

    expected_top_level = {
        "schema_version",
        "kind",
        "claim",
        "contract_version",
        "canary_id",
        "identity_sha256",
        "identity",
        "canary_execution",
        "source",
        "infrastructure",
        "publication",
        "manifest_sha256",
    }
    if set(manifest) != expected_top_level:
        raise ValueError(f"canary manifest fields drifted: {sorted(set(manifest) ^ expected_top_level)}")
    claimed_manifest_sha = _require_sha(manifest.get("manifest_sha256"), "manifest_sha256")
    clean_manifest = dict(manifest)
    clean_manifest.pop("manifest_sha256")
    if _sha256_json(clean_manifest) != claimed_manifest_sha:
        raise ValueError("canary manifest self-seal mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != KIND
        or manifest.get("claim") != CLAIM
        or manifest.get("contract_version") != CONTRACT_VERSION
    ):
        raise ValueError("canary kind/claim/version contract drifted")

    identity = manifest.get("identity")
    expected_identity_fields = {
        "canary_source_tree_sha256",
        "canary_runtime_source_tree_sha256",
        "reference_run_id",
        "reference_scientific_spec_sha256",
        "reference_manifest_sha256",
        "production_source_tree_sha256",
        "task",
        "arm",
        "mechanism",
        "data",
        "initialization",
        "workspace_representation",
        "sources",
        "model_training_identity",
    }
    if not isinstance(identity, dict) or set(identity) != expected_identity_fields:
        raise ValueError("canary identity fields drifted")
    identity_sha = _require_sha(manifest.get("identity_sha256"), "identity_sha256")
    if _sha256_json(identity) != identity_sha:
        raise ValueError("canary identity seal mismatch")
    exact_lineage = {
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_scientific_spec_sha256": REFERENCE_SCIENTIFIC_SPEC_SHA256,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "production_source_tree_sha256": PRODUCTION_SOURCE_SHA256,
        "arm": ARM,
    }
    lineage_drift = {
        name: (identity.get(name), expected)
        for name, expected in exact_lineage.items()
        if identity.get(name) != expected
    }
    if lineage_drift:
        raise ValueError(f"frozen production lineage drifted: {lineage_drift}")
    _require_sha(identity.get("canary_source_tree_sha256"), "identity canary source SHA")
    _require_sha(
        identity.get("canary_runtime_source_tree_sha256"),
        "identity canary runtime source SHA",
    )
    component_drift = {
        name: _sha256_json(identity.get(name))
        for name, expected in IDENTITY_COMPONENT_SHA256.items()
        if _sha256_json(identity.get(name)) != expected
    }
    if component_drift:
        raise ValueError(f"frozen task/model/data/init/workspace/source identity drifted: {component_drift}")

    task_name = identity["task"]["name"]
    expected_canary_id = f"policy-canary-v1-{task_name.lower()}-{ARM}-{identity_sha[:16]}"
    if manifest.get("canary_id") != expected_canary_id:
        raise ValueError("canary ID is not canonically derived from its sealed identity")

    execution = manifest.get("canary_execution", {})
    expected_execution = {
        "optimizer_steps": STEPS,
        "final_local_checkpoint_step": FINAL_STEP,
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
    }
    if execution != expected_execution:
        raise ValueError("canary exact two-step execution/restore contract drifted")

    source = manifest.get("source", {})
    expected_source_fields = {
        "canary_source_tree_sha256",
        "canary_runtime_source_tree_sha256",
        "entry",
        "entry_sha256",
        "submitted_entry_mode",
        "sagemaker_runtime_entry_mode",
        "baseline_source_tree_sha256",
        "production_entry_sha256",
        "reviewed_delta_paths",
    }
    if not isinstance(source, dict) or set(source) != expected_source_fields:
        raise ValueError("canary source proof fields drifted")
    if (
        source.get("canary_source_tree_sha256") != identity["canary_source_tree_sha256"]
        or source.get("canary_runtime_source_tree_sha256") != identity["canary_runtime_source_tree_sha256"]
        or source.get("entry") != "gpu_policy_canary_entry.sh"
        or source.get("submitted_entry_mode") != SUBMITTED_ENTRY_MODE
        or source.get("sagemaker_runtime_entry_mode") != SAGEMAKER_RUNTIME_ENTRY_MODE
        or source.get("baseline_source_tree_sha256") != PRODUCTION_SOURCE_SHA256
        or source.get("production_entry_sha256") != PRODUCTION_ENTRY_SHA256
        or set(source.get("reviewed_delta_paths", [])) != REVIEWED_SOURCE_DELTAS
        or len(source.get("reviewed_delta_paths", [])) != len(REVIEWED_SOURCE_DELTAS)
    ):
        raise ValueError("canary source/baseline/reviewed-delta contract drifted")
    if _require_sha(source.get("entry_sha256"), "canary entry_sha256") != CANARY_ENTRY_SHA256:
        raise ValueError("canary entry byte identity drifted")
    _require_sha(
        source.get("canary_runtime_source_tree_sha256"),
        "canary runtime source_tree_sha256",
    )

    expected_job_name = f"sarvesh-rmme-policy-canary-{identity_sha[:20]}"
    infrastructure = manifest.get("infrastructure", {})
    expected_infrastructure = {
        "provider": "aws_sagemaker",
        "execution_account": "141701954645",
        "queue": "fss-tri-cam-robotics-p5e-48xlarge-us-west-2-training-plan",
        "training_plan_arn": ("arn:aws:sagemaker:us-west-2:141701954645:training-plan/cam-robotics-tp"),
        "role": ("arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2"),
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "priority": 400,
        "max_run_seconds": 10_800,
        "volume_size_gb": 250,
        "attempts_in_job": 1,
        "deterministic_job_name": expected_job_name,
    }
    if infrastructure != expected_infrastructure:
        raise ValueError("canary exact p5e infrastructure contract drifted")

    publication = manifest.get("publication", {})
    namespace = f"{STUDY_ROOT}/manifests/canaries/policy_training/{expected_canary_id}"
    receipt = publication.get("receipt_s3", "")
    if (
        set(publication)
        != {
            "namespace_s3",
            "receipt_s3",
            "create_once",
            "only_allowed_object",
            "production_checkpoint_or_deploy_publication",
        }
        or publication.get("namespace_s3") != namespace
        or receipt != f"{namespace}/training_canary.complete.json"
        or publication.get("create_once") is not True
        or publication.get("only_allowed_object") != "training_canary.complete.json"
        or publication.get("production_checkpoint_or_deploy_publication") is not False
    ):
        raise ValueError("canary publication contract drifted")
    forbidden_fragments = (
        "/checkpoints/robomme/pi05/",
        "/manifests/runs/train/",
        "/manifests/claims/train/",
        "/manifests/artifacts/checkpoints/",
        "/deploy/",
        "_DEPLOY_COMPLETE",
    )
    manifest_text = _canonical_json(manifest)
    leaked_fragments = [fragment for fragment in forbidden_fragments if fragment in manifest_text]
    if leaked_fragments:
        raise ValueError(f"production output namespace leaked into canary manifest: {leaked_fragments}")


def validate_manifest_environment(manifest: dict[str, Any], environment: dict[str, str]) -> None:
    """Fail closed before downloads, accelerator initialization, or any S3 write."""

    validate_manifest_contract(manifest)
    publication = manifest["publication"]
    receipt = publication["receipt_s3"]
    namespace = publication["namespace_s3"]
    leaked_keys = sorted(FORBIDDEN_ENVIRONMENT_KEYS & environment.keys())
    if leaked_keys:
        raise ValueError(f"production output flags leaked into canary environment: {leaked_keys}")
    expected_environment = {
        "ROBOMME_CANARY_KIND": KIND,
        "ROBOMME_CANARY_CLAIM": CLAIM,
        "ROBOMME_CANARY_ID": manifest.get("canary_id"),
        "ROBOMME_CANARY_STEPS": "2",
        "ROBOMME_CANARY_MANIFEST_SHA256": manifest.get("manifest_sha256"),
        "ROBOMME_CANARY_NAMESPACE_S3": namespace,
        "ROBOMME_CANARY_RECEIPT_S3": receipt,
        "ROBOMME_REFERENCE_RUN_ID": manifest.get("identity", {}).get("reference_run_id"),
        "ROBOMME_REFERENCE_SCIENTIFIC_SPEC_SHA256": manifest.get("identity", {}).get(
            "reference_scientific_spec_sha256"
        ),
        "ROBOMME_REFERENCE_SOURCE_SHA256": manifest.get("identity", {}).get("production_source_tree_sha256"),
        "ROBOMME_ARM": manifest.get("identity", {}).get("arm"),
        "ROBOMME_TASK": manifest.get("identity", {}).get("task", {}).get("name"),
        "ROBOMME_SCOPE": "single_task_canary",
        "WSM_MAX_STEPS": "2",
        "WSM_SAVE_INTERVAL": "2",
        "WSM_WARMUP_STEPS": "1",
        "WSM_DECAY_STEPS": "2",
        "WSM_SEED": "0",
    }
    drift = {
        name: (environment.get(name), expected)
        for name, expected in expected_environment.items()
        if environment.get(name) != expected
    }
    if drift:
        raise ValueError(f"canary environment differs from its sealed manifest: {drift}")
    if environment.get("ROBOMME_ARM") != ARM:
        raise ValueError(f"this audited canary runner requires {ARM}")
    if environment.get("ROBOMME_CANARY_RECEIPT_S3", "").endswith("/training_canary.complete.json") is False:
        raise ValueError("canary receipt basename drifted")
    identity = manifest.get("identity", {})
    scientific_sources = identity.get("sources", {})
    expected_artifacts = {
        "ROBOMME_DATA_S3": identity.get("data", {}).get("dataset_s3"),
        "ROBOMME_DATA_PARENT_INVENTORY_S3": identity.get("data", {}).get("parent_inventory_uri"),
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": identity.get("data", {}).get("parent_inventory_sha256"),
        "ROBOMME_DATA_DERIVED_INVENTORY_SHA256": identity.get("data", {}).get("derived_task_inventory_sha256"),
        "INIT_S3": identity.get("initialization", {}).get("checkpoint_s3"),
        "INIT_INVENTORY_S3": identity.get("initialization", {}).get("inventory_uri"),
        "INIT_INVENTORY_SHA256": identity.get("initialization", {}).get("inventory_sha256"),
        "OPENPI_FORK_S3": scientific_sources.get("openpi", {}).get("uri"),
        "PALIGEMMA_TOKENIZER_S3": scientific_sources.get("tokenizer", {}).get("uri"),
        "PALIGEMMA_TOKENIZER_SHA256": scientific_sources.get("tokenizer", {}).get("sha256"),
        "ROBOMME_WORKSPACE_S3": identity.get("workspace_representation", {}).get("omega", {}).get("uri"),
        "ROBOMME_WORKSPACE_MANIFEST_SHA256": identity.get("workspace_representation", {})
        .get("omega", {})
        .get("manifest_sha256"),
    }
    artifact_drift = {
        name: (environment.get(name), expected)
        for name, expected in expected_artifacts.items()
        if not expected or environment.get(name) != expected
    }
    if artifact_drift:
        raise ValueError(f"canary artifact identity drifted: {artifact_drift}")


def validate_node_topology(*, device_count: int, process_count: int, gpu_names: list[str]) -> None:
    if device_count != 8 or process_count != 1:
        raise ValueError(
            f"canary requires eight devices in one process; got devices={device_count} processes={process_count}"
        )
    if len(gpu_names) != 8 or any("H200" not in name.upper() for name in gpu_names):
        raise ValueError(f"canary requires exactly 8 H200 GPUs; got {gpu_names}")


def validate_diagnostic_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if [record.get("step") for record in records] != [0, 1]:
        raise ValueError(f"canary must produce exactly optimizer steps [0, 1], got {records}")
    numeric = (
        "total_loss",
        "diagnostic_total_loss",
        "production_loss_parity_error",
        "action_loss",
        "jepa_loss",
        "jepa_weighted",
        "sigreg_loss",
        "sigreg_weighted",
        "grad_norm",
        "gdn_grad_norm",
        "jepa_head_grad_norm",
        "update_norm",
        "parameter_delta_norm",
    )
    clean: list[dict[str, Any]] = []
    expected_fields = {
        "step",
        *numeric,
        "valid_target_count",
        "parameters_finite",
        "optimizer_finite",
    }
    for record in records:
        if set(record) != expected_fields:
            raise ValueError(
                f"step {record.get('step')} diagnostic fields drifted: {sorted(set(record) ^ expected_fields)}"
            )
        normalized = dict(record)
        for name in numeric:
            try:
                value = float(record[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"step {record.get('step')} missing numeric diagnostic {name}") from error
            if not math.isfinite(value):
                raise ValueError(f"step {record.get('step')} has non-finite {name}={value}")
            normalized[name] = value
        if float(record.get("valid_target_count", 0.0)) <= 0:
            raise ValueError(f"step {record.get('step')} did not exercise a valid JEPA target")
        if normalized["grad_norm"] <= 0:
            raise ValueError(f"step {record.get('step')} has a zero gradient")
        if normalized["gdn_grad_norm"] <= 0 or normalized["jepa_head_grad_norm"] <= 0:
            raise ValueError(f"step {record.get('step')} did not train both GDN and JEPA subtrees")
        if record.get("parameters_finite") is not True or record.get("optimizer_finite") is not True:
            raise ValueError(f"step {record.get('step')} produced non-finite params/optimizer state")
        expected = normalized["action_loss"] + normalized["jepa_weighted"] + normalized["sigreg_weighted"]
        if not math.isclose(normalized["total_loss"], expected, rel_tol=2e-4, abs_tol=2e-5):
            raise ValueError(
                f"step {record.get('step')} total loss does not decompose into action+JEPA+SigReg: "
                f"{normalized['total_loss']} != {expected}"
            )
        if not math.isclose(
            normalized["total_loss"],
            normalized["diagnostic_total_loss"],
            rel_tol=2e-4,
            abs_tol=2e-5,
        ) or normalized["production_loss_parity_error"] > max(2e-5, 2e-4 * abs(normalized["total_loss"])):
            raise ValueError(f"step {record.get('step')} production compute_loss differs from the diagnostic forward")
        for name in ("action_loss", "jepa_loss", "jepa_weighted", "sigreg_loss", "sigreg_weighted"):
            if normalized[name] < -1e-5:
                raise ValueError(f"step {record.get('step')} has unexpectedly negative {name}={normalized[name]}")
        clean.append(normalized)
    # The pinned schedule initializes at peak_lr/(warmup_steps+1), so both steps must mutate params.
    for record in clean:
        if record["update_norm"] <= 0 or record["parameter_delta_norm"] <= 0:
            raise ValueError(f"optimizer step {record['step']} did not produce a nonzero parameter update")
    return clean


def build_completion_receipt(
    manifest: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    training_job_name: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    validate_manifest_contract(manifest)
    records = validate_diagnostic_records(records)
    if not training_job_name:
        raise ValueError("SageMaker training job name unavailable")
    required_checkpoint = {
        "scope": "node_local_ephemeral_only",
        "step": FINAL_STEP,
        "saved_state_step": STEPS,
        "restored_state_step": STEPS,
        "restore_smoke": True,
        "parameters_finite": True,
        "ema_parameters_finite": True,
        "optimizer_finite": True,
        "published": False,
    }
    drift = {
        name: (checkpoint.get(name), expected)
        for name, expected in required_checkpoint.items()
        if checkpoint.get(name) != expected
    }
    if drift:
        raise ValueError(f"local checkpoint/restore proof drifted: {drift}")
    expected_checkpoint_fields = {
        *required_checkpoint,
        "tree",
        "restore_fingerprints",
    }
    if set(checkpoint) != expected_checkpoint_fields:
        raise ValueError(f"checkpoint proof fields drifted: {sorted(set(checkpoint) ^ expected_checkpoint_fields)}")
    validate_restore_fingerprint_proof(checkpoint.get("restore_fingerprints"))
    tree = checkpoint.get("tree", {})
    expected_tree_fields = {
        "files",
        "bytes",
        "layout_sha256",
        "checkpoint_metadata_sha256",
        "content_sha256",
    }
    if set(tree) != expected_tree_fields:
        raise ValueError(f"checkpoint tree fields drifted: {sorted(set(tree) ^ expected_tree_fields)}")
    if (
        not isinstance(tree.get("files"), int)
        or tree["files"] < 1
        or not isinstance(tree.get("bytes"), int)
        or tree["bytes"] < 1
    ):
        raise ValueError(f"checkpoint tree has invalid file/byte counts: {tree}")
    for name in ("layout_sha256", "checkpoint_metadata_sha256", "content_sha256"):
        _require_sha(tree.get(name), f"checkpoint tree {name}")
    source = runtime.get("source", {})
    expected_runtime_fields = {
        "source",
        "infrastructure",
        "gpu_inventory",
        "jax_inventory",
        "openpi_archive_sha256",
        "openpi_overlay",
    }
    if set(runtime) != expected_runtime_fields:
        raise ValueError(f"runtime proof fields drifted: {sorted(set(runtime) ^ expected_runtime_fields)}")
    if set(source) != {
        "canary_source_tree_sha256",
        "canary_runtime_source_tree_sha256",
        "entry",
        "entry_sha256",
        "submitted_entry_mode",
        "sagemaker_runtime_entry_mode",
        "excluded_staged_file",
    }:
        raise ValueError("runtime source proof fields drifted")
    manifest_source = manifest.get("source", {})
    if (
        source.get("canary_source_tree_sha256") != manifest_source.get("canary_source_tree_sha256")
        or source.get("canary_runtime_source_tree_sha256") != manifest_source.get("canary_runtime_source_tree_sha256")
        or source.get("entry_sha256") != manifest_source.get("entry_sha256")
    ):
        raise ValueError("runtime source proof differs from the sealed canary source")
    if (
        source.get("entry") != manifest_source.get("entry")
        or source.get("submitted_entry_mode") != SUBMITTED_ENTRY_MODE
        or source.get("sagemaker_runtime_entry_mode") != SAGEMAKER_RUNTIME_ENTRY_MODE
        or source.get("submitted_entry_mode") != manifest_source.get("submitted_entry_mode")
        or source.get("sagemaker_runtime_entry_mode") != manifest_source.get("sagemaker_runtime_entry_mode")
        or source.get("excluded_staged_file") != "_robomme_policy_canary_manifest.json"
    ):
        raise ValueError("runtime source entry/mode/exclusion proof drifted")
    infrastructure = runtime.get("infrastructure", {})
    if set(infrastructure) != {"instance_type", "accelerator", "jax_process_count"}:
        raise ValueError("runtime infrastructure proof fields drifted")
    if infrastructure.get("instance_type") != "ml.p5e.48xlarge":
        raise ValueError(f"runtime instance type is not p5e: {infrastructure}")
    if infrastructure.get("accelerator") != "8xH200" or infrastructure.get("jax_process_count") != 1:
        raise ValueError(f"runtime infrastructure is not one 8-H200 process: {infrastructure}")
    gpu_inventory = runtime.get("gpu_inventory")
    jax_inventory = runtime.get("jax_inventory")
    if (
        not isinstance(gpu_inventory, list)
        or len(gpu_inventory) != 8
        or any("H200" not in record.get("name", "").upper() for record in gpu_inventory)
        or any(not record.get("uuid") for record in gpu_inventory)
        or len({record.get("uuid") for record in gpu_inventory}) != 8
        or not isinstance(jax_inventory, list)
        or len(jax_inventory) != 8
        or any("H200" not in record.get("device_kind", "").upper() for record in jax_inventory)
        or any(record.get("platform") != "gpu" for record in jax_inventory)
        or len({record.get("id") for record in jax_inventory}) != 8
        or any(set(record) != {"index", "uuid", "name"} for record in gpu_inventory)
        or any(set(record) != {"id", "platform", "device_kind"} for record in jax_inventory)
    ):
        raise ValueError("runtime receipt lacks exact 8-H200 NVIDIA/JAX inventories")
    expected_openpi = manifest.get("identity", {}).get("sources", {}).get("openpi", {}).get("sha256")
    if runtime.get("openpi_archive_sha256") != expected_openpi:
        raise ValueError("runtime OpenPI archive differs from the sealed production identity")
    overlay = runtime.get("openpi_overlay", {})
    if (
        set(overlay)
        != {
            "overlay_manifest_sha256",
            "loaded_pi0_config",
            "loaded_pi0",
            "allowed_workspace_pair",
            "model_math_changed",
        }
        or overlay.get("model_math_changed") is not False
        or overlay.get("allowed_workspace_pair") != ["jepa_aux_target", "tanh"]
        or not overlay.get("loaded_pi0", "").endswith("/src/openpi/models/pi0.py")
        or not overlay.get("loaded_pi0_config", "").endswith("/src/openpi/models/pi0_config.py")
    ):
        raise ValueError(f"runtime GDN+JEPA overlay proof is incomplete: {overlay}")
    if (
        _require_sha(overlay.get("overlay_manifest_sha256"), "runtime overlay_manifest_sha256")
        != GDN_JEPA_OVERLAY_MANIFEST_SHA256
    ):
        raise ValueError("runtime GDN+JEPA overlay manifest identity drifted")
    receipt = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "claim": CLAIM,
        "canary_id": manifest["canary_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "identity_sha256": manifest["identity_sha256"],
        "reference_run_id": manifest["identity"]["reference_run_id"],
        "reference_scientific_spec_sha256": manifest["identity"]["reference_scientific_spec_sha256"],
        "training_job_name": training_job_name,
        "optimizer_steps": STEPS,
        "effective_per_step_batch": 64,
        "diagnostics": records,
        "checkpoint": checkpoint,
        "runtime": runtime,
        # The only published object is intentionally standalone: it embeds the already self-sealed
        # launch manifest, including source, data, init, workspace, OpenPI, image, and infrastructure.
        "sealed_manifest": manifest,
        "scientific_evidence": False,
        "deployable": False,
        "evaluation_eligible": False,
        "published_production_checkpoint": False,
        "published_deploy_artifact": False,
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    return receipt


def validate_completion_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("kind") != RECEIPT_KIND or receipt.get("claim") != CLAIM:
        raise ValueError("completion receipt kind/claim drifted")
    claimed = _require_sha(receipt.get("receipt_sha256"), "receipt_sha256")
    clean = dict(receipt)
    clean.pop("receipt_sha256")
    if _sha256_json(clean) != claimed:
        raise ValueError("completion receipt self-seal mismatch")
    manifest = receipt.get("sealed_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("standalone completion receipt omitted its sealed manifest")
    manifest_clean = dict(manifest)
    manifest_claimed = _require_sha(manifest_clean.pop("manifest_sha256", None), "embedded manifest_sha256")
    if _sha256_json(manifest_clean) != manifest_claimed:
        raise ValueError("embedded canary manifest self-seal mismatch")
    if receipt.get("manifest_sha256") != manifest_claimed:
        raise ValueError("receipt does not bind its embedded manifest")
    validate_manifest_contract(manifest)
    rebuilt = build_completion_receipt(
        manifest,
        records=receipt.get("diagnostics", []),
        checkpoint=receipt.get("checkpoint", {}),
        training_job_name=receipt.get("training_job_name", ""),
        runtime=receipt.get("runtime", {}),
    )
    if rebuilt != receipt:
        raise ValueError("completion receipt fields differ from the canonical standalone contract")


def strong_checkpoint_tree(root: Path) -> dict[str, Any]:
    """Hash every finalized local checkpoint byte for a standalone, non-deploy receipt."""

    try:
        from ..gpu.checkpoint_transport import tree_summary
    except ImportError:  # SageMaker stages robomme_integration contents as top-level packages.
        from gpu.checkpoint_transport import tree_summary

    summary = tree_summary(root)
    digest = hashlib.sha256()
    counted_files = 0
    counted_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or "orbax-checkpoint-tmp" in path.as_posix():
            continue
        relative = path.relative_to(root).as_posix()
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        counted_files += 1
        counted_bytes += size
    if counted_files != summary["files"] or counted_bytes != summary["bytes"]:
        raise ValueError(
            "checkpoint changed between structural and content receipts: "
            f"files={counted_files}/{summary['files']} bytes={counted_bytes}/{summary['bytes']}"
        )
    return {**summary, "content_sha256": digest.hexdigest()}


def _run(
    manifest: dict[str, Any],
    proof_path: Path,
    *,
    source_proof: dict[str, Any],
    gpu_names: list[str],
    gpu_uuids: list[str],
) -> None:
    """Execute the exact two-step GDN+JEPA model/optimizer path and a local restore smoke."""

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
    from .gdn_jepa_overlay import validate_loaded_overlay
    from .train import _openpi_train_module

    validate_node_topology(
        device_count=jax.device_count(),
        process_count=jax.process_count(),
        gpu_names=gpu_names,
    )
    if len(gpu_uuids) != 8 or len(set(gpu_uuids)) != 8 or any(not value for value in gpu_uuids):
        raise ValueError(f"canary requires eight distinct NVIDIA GPU UUIDs, got {gpu_uuids}")
    jax_inventory = [
        {
            "id": int(device.id),
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    if any(record["platform"] != "gpu" or "H200" not in record["device_kind"].upper() for record in jax_inventory):
        raise ValueError(f"JAX did not bind the exact H200 topology: {jax_inventory}")
    overlay_root = Path(os.environ["ROBOMME_GDN_JEPA_OVERLAY_ROOT"])
    overlay_proof = validate_loaded_overlay(overlay_root)
    install_data_loader_patch()
    config = build_train_config(ARM)
    validate_train_config(config, ARM)
    expected_model = manifest["identity"]["model_training_identity"]
    actual_model = {
        "seed": config.seed,
        "batch_size": config.batch_size,
        "batch_unit": "steps",
        "effective_per_step_batch": config.batch_size,
        "action_horizon": config.model.action_horizon,
        "full_finetune": True,
        "optimizer": "AdamW",
        "peak_lr": config.lr_schedule.peak_lr,
        "ema_decay": config.ema_decay,
        "fsdp_devices": config.fsdp_devices,
        "jax_devices": jax.device_count(),
        "jax_processes": jax.process_count(),
    }
    if actual_model != expected_model:
        raise ValueError(
            f"runtime model/training identity differs from production reference: {actual_model} != {expected_model}"
        )
    if (
        config.num_train_steps != STEPS
        or config.save_interval != STEPS
        or config.model.wsm_cond_type != "gated_deltanet"
        or config.model.wsm_cond_window != 8
        or not config.model.wsm_tanh
        or not config.model.wsm_jepa
        or config.model.wsm_jepa_weight != 0.1
        or config.model.wsm_jepa_sigreg_weight != 0.05
        or config.model.wsm_jepa_num_futures != 1
        or config.batch_size != 64
    ):
        raise ValueError("runtime canary config lost the exact GDN8+JEPA model identity")
    if config.staged is not None:
        raise ValueError(
            "GDN8+JEPA production config unexpectedly acquired a staged recipe; "
            "the canary refuses to omit staged grad/update masks"
        )

    train_module = _openpi_train_module()
    train_module.init_logging()

    def compute_loss_with_diagnostics(model, rng, observation, actions):
        """Exact GDN+JEPA production forward, with diagnostics returned as auxiliary values."""

        if not (
            model.pi05
            and model.wsm_tanh
            and model.wsm_cond_type == "gated_deltanet"
            and model.wsm_jepa
            and not model.wsm_cfg
            and not model.wsm_cfg2
            and not model.robottt
        ):
            raise ValueError("canary loss is only valid for the reviewed GDN8+JEPA model")
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_api.preprocess_observation(preprocess_rng, observation, train=True)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        if observation.wsm_w_window is None:
            raise ValueError("GDN canary batch has no causal workspace window")
        workspace_vec = model._current_workspace_vec(observation, train=True, rng=None, force_uncond=False)
        adarms_cond = adarms_cond + workspace_vec.astype(adarms_cond.dtype)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        action_chunks = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if observation.wsm_w_target is None or observation.wsm_w_target_valid is None:
            raise ValueError("JEPA canary batch has no future workspace target/valid mask")
        common = dict(
            head=model.wsm_jepa_head,
            penult_act=suffix_out[:, -model.action_horizon :],
            w_target=observation.wsm_w_target,
            target_valid=observation.wsm_w_target_valid,
            rng=rng,
            num_futures=model.wsm_jepa_num_futures,
            regularizer=model.wsm_jepa_regularizer,
            visreg_weight=model.wsm_jepa_visreg_weight,
            visreg_num_slices=model.wsm_jepa_visreg_slices,
            visreg_scale_weight=model.wsm_jepa_visreg_scale_weight,
            visreg_shape_weight=model.wsm_jepa_visreg_shape_weight,
            visreg_center_weight=model.wsm_jepa_visreg_center_weight,
        )
        aux_total = wsm_jepa_aux_loss(
            **common,
            jepa_weight=model.wsm_jepa_weight,
            sigreg_weight=model.wsm_jepa_sigreg_weight,
        )
        jepa_raw = wsm_jepa_aux_loss(**common, jepa_weight=1.0, sigreg_weight=0.0)
        sigreg_raw = wsm_jepa_aux_loss(**common, jepa_weight=0.0, sigreg_weight=1.0)
        total = jnp.mean(action_chunks + aux_total.astype(action_chunks.dtype))
        action = jnp.mean(action_chunks)
        diagnostics = {
            "total_loss": total,
            "action_loss": action,
            "jepa_loss": jepa_raw,
            "jepa_weighted": model.wsm_jepa_weight * jepa_raw,
            "sigreg_loss": sigreg_raw,
            "sigreg_weighted": model.wsm_jepa_sigreg_weight * sigreg_raw,
            "valid_target_count": observation.wsm_w_target_valid.astype(jnp.float32).sum(),
        }
        return total, diagnostics

    def tree_finite(tree):
        flags = [
            jnp.all(jnp.isfinite(jnp.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(tree)
            if hasattr(leaf, "dtype") and jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
        ]
        return jnp.all(jnp.stack(flags)) if flags else jnp.asarray(True)

    def subtree_norm(tree, name: str):
        def select(path, value):
            head = getattr(path[0], "key", None) if path else None
            return value if head == name else jnp.zeros_like(value)

        return optax.global_norm(jax.tree_util.tree_map_with_path(select, tree))

    def training_parameter_delta_norm(before, after):
        delta = jax.tree.map(lambda old, new: new - old, before, after)
        return optax.global_norm(delta)

    def canary_train_step(config, rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, step_rng, observation, actions):
            # Differentiate the pinned production callable itself. The second same-RNG forward is
            # auxiliary only and proves the diagnostic decomposition remains exactly equivalent.
            production_total = jnp.mean(model.compute_loss(step_rng, observation, actions, train=True))
            diagnostic_total, diagnostics = compute_loss_with_diagnostics(model, step_rng, observation, actions)
            return production_total, {
                **diagnostics,
                "total_loss": production_total,
                "diagnostic_total_loss": diagnostic_total,
                "production_loss_parity_error": jnp.abs(production_total - diagnostic_total),
            }

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, config.trainable_filter)
        (loss, diagnostics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        updated_trainable = optax.apply_updates(params, updates)
        parameter_delta = training_parameter_delta_norm(params, updated_trainable)
        nnx.update(model, updated_trainable)
        new_params = nnx.state(model)
        new_state = dataclasses.replace(
            state,
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )
        return new_state, {
            **diagnostics,
            "grad_norm": optax.global_norm(grads),
            "gdn_grad_norm": subtree_norm(grads, "wsm_tanh_cond"),
            "jepa_head_grad_norm": subtree_norm(grads, "wsm_jepa_head"),
            "update_norm": optax.global_norm(updates),
            "parameter_delta_norm": parameter_delta,
            "parameters_finite": tree_finite(new_params),
            "optimizer_finite": tree_finite(new_opt_state),
            "state_step_after": new_state.step,
        }

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=None,
        overwrite=False,
        resume=False,
    )
    if resuming:
        raise ValueError("canary checkpoint directory unexpectedly entered resume mode")
    data_loader = data_loader_api.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iterator = iter(data_loader)
    batch = next(data_iterator)
    train_state, state_sharding = train_module.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    ptrain_step = jax.jit(
        functools.partial(canary_train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    records: list[dict[str, Any]] = []
    for step in range(STEPS):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        reduced = jax.device_get(info)
        record = {
            "step": step,
            **{
                name: bool(value) if name in {"parameters_finite", "optimizer_finite"} else float(value)
                for name, value in reduced.items()
                if name != "state_step_after"
            },
        }
        if int(reduced["state_step_after"]) != step + 1:
            raise ValueError(f"optimizer state step did not advance: {reduced['state_step_after']} != {step + 1}")
        records.append(record)
        if step + 1 < STEPS:
            batch = next(data_iterator)
    records = validate_diagnostic_records(records)
    checkpoints.save_state(checkpoint_manager, train_state, data_loader, FINAL_STEP)
    checkpoint_manager.wait_until_finished()
    if tuple(checkpoint_manager.all_steps()) != (FINAL_STEP,):
        raise ValueError(f"local checkpoint manager steps drifted: {checkpoint_manager.all_steps()}")
    saved_state_step = int(jax.device_get(train_state.step))
    if saved_state_step != STEPS:
        raise ValueError(f"saved optimizer step drifted: {saved_state_step} != {STEPS}")
    checkpoint_dir = Path(config.checkpoint_dir) / str(FINAL_STEP)
    if (
        not (checkpoint_dir / "params").is_dir()
        or not (checkpoint_dir / "train_state").is_dir()
        or not (checkpoint_dir / "_CHECKPOINT_METADATA").is_file()
    ):
        raise ValueError(f"final local Orbax checkpoint is incomplete: {checkpoint_dir}")
    checkpoint_tree = strong_checkpoint_tree(checkpoint_dir)

    # Release every training/compilation/data reference not required by Orbax before constructing a
    # cold template. In particular, clear JAX's compiled executable cache before either host
    # fingerprinting or restore, so the trained state cannot contend with the two-forward train step.
    del (
        ptrain_step,
        batch,
        info,
        reduced,
        data_iterator,
        state_sharding,
        data_sharding,
        replicated_sharding,
    )
    gc.collect()
    jax.clear_caches()
    gc.collect()

    # Fingerprint and release the saved state before even constructing the cold resume template.
    # This prevents the saved, template, and restored model trees from ever being resident together.
    saved_fingerprints = fingerprint_train_state_components(train_state)
    del train_state
    gc.collect()

    cold_restore_template, cold_restore_sharding = train_module.init_train_state(config, init_rng, mesh, resume=True)
    restored = checkpoints.restore_state(checkpoint_manager, cold_restore_template, data_loader, step=FINAL_STEP)
    jax.block_until_ready(restored)
    del cold_restore_template, cold_restore_sharding
    gc.collect()
    restored_state_step = int(jax.device_get(restored.step))
    if restored_state_step != STEPS:
        raise ValueError(f"restored optimizer step drifted: {restored_state_step} != {STEPS}")
    if restored.ema_params is None:
        raise ValueError("GDN+JEPA canary lost its required EMA state during restore")
    del checkpoint_manager, data_loader
    gc.collect()

    # No JAX arithmetic or device-side comparison is performed. Shape, dtype, tree path, and every
    # raw byte must match exactly, with at most one extra host leaf resident at any instant.
    restored_fingerprints = fingerprint_train_state_components(restored)
    restore_fingerprints = build_restore_fingerprint_proof(saved_fingerprints, restored_fingerprints)
    del restored
    gc.collect()

    checkpoint_proof = {
        "scope": "node_local_ephemeral_only",
        "step": FINAL_STEP,
        "tree": checkpoint_tree,
        "saved_state_step": saved_state_step,
        "restored_state_step": restored_state_step,
        "restore_smoke": True,
        "restore_fingerprints": restore_fingerprints,
        "parameters_finite": restored_fingerprints["params"]["all_finite"],
        "ema_parameters_finite": restored_fingerprints["ema_params"]["all_finite"],
        "optimizer_finite": restored_fingerprints["optimizer_state"]["all_finite"],
        "published": False,
    }
    training_environment: dict[str, Any] = {}
    if os.environ.get("SM_TRAINING_ENV"):
        try:
            training_environment = json.loads(os.environ["SM_TRAINING_ENV"])
        except json.JSONDecodeError as error:
            raise ValueError("SM_TRAINING_ENV is malformed") from error
    instance_type = os.environ.get("SM_CURRENT_INSTANCE_TYPE") or training_environment.get("resource_config", {}).get(
        "current_instance_type"
    )
    if instance_type != "ml.p5e.48xlarge":
        raise ValueError(f"canary runtime is not ml.p5e.48xlarge: {instance_type!r}")
    runtime_proof = {
        "source": source_proof,
        "infrastructure": {
            "instance_type": instance_type,
            "accelerator": "8xH200",
            "jax_process_count": jax.process_count(),
        },
        "gpu_inventory": [
            {"index": index, "uuid": uuid, "name": name}
            for index, (uuid, name) in enumerate(zip(gpu_uuids, gpu_names, strict=True))
        ],
        "jax_inventory": jax_inventory,
        "openpi_archive_sha256": manifest["identity"]["sources"]["openpi"]["sha256"],
        "openpi_overlay": overlay_proof,
    }
    receipt = build_completion_receipt(
        manifest,
        records=records,
        checkpoint=checkpoint_proof,
        training_job_name=os.environ.get("SM_TRAINING_JOB_NAME") or os.environ.get("TRAINING_JOB_NAME", ""),
        runtime=runtime_proof,
    )
    validate_completion_receipt(receipt)
    temporary = proof_path.with_name("." + proof_path.name + ".incomplete")
    temporary.write_text(_canonical_json(receipt) + "\n", encoding="utf-8")
    temporary.replace(proof_path)
    print(_canonical_json(receipt), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--sha256", required=True)
    validate.add_argument("--code-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--sha256", required=True)
    run.add_argument("--code-dir", type=Path, required=True)
    run.add_argument("--proof", type=Path, required=True)
    run.add_argument("--gpu-name", action="append", default=[])
    run.add_argument("--gpu-uuid", action="append", default=[])
    validate_receipt = commands.add_parser("validate-receipt")
    validate_receipt.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-receipt":
        try:
            receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read completion receipt: {error}") from error
        validate_completion_receipt(receipt)
        print(
            _canonical_json(
                {
                    "kind": receipt["kind"],
                    "claim": receipt["claim"],
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
        )
        return
    manifest = load_manifest(args.manifest, args.sha256)
    validate_manifest_environment(manifest, dict(os.environ))
    source_proof = verify_packaged_source(
        args.code_dir,
        manifest,
        staged_manifest=args.manifest.name,
    )
    if args.command == "validate-manifest":
        print(
            _canonical_json(
                {
                    "kind": manifest["kind"],
                    "claim": manifest["claim"],
                    "canary_id": manifest["canary_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "source": source_proof,
                }
            )
        )
        return
    if args.proof.exists() or args.proof.is_symlink():
        raise SystemExit(f"refusing to overwrite existing canary proof: {args.proof}")
    _run(
        manifest,
        args.proof,
        source_proof=source_proof,
        gpu_names=args.gpu_name,
        gpu_uuids=args.gpu_uuid,
    )


if __name__ == "__main__":
    main()
