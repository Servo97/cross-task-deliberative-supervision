#!/usr/bin/env python3
"""Run a sealed series of single-task RoboMME fixed-50 evaluations on one p5 node.

This module is deliberately *not* a cloud launcher.  It is a node-local, approval-gated
supervisor for checkpoints which already have immutable training completion claims.  A queue is
accepted only when it pins both a successful native-render p5 preflight claim and a separately
sealed staged-runtime receipt.  Unknown failures are terminal for one cell, not retryable and not
allowed to advance the remaining queue.

The old ``run_local_fixed50_queue`` remains the authority for fixed-50 result auditing and evidence
selection; ``launch_gpu_fleet`` remains the authority for policy/simulator process topology.  This
file adds queue identity, workspace staging, narrowly classified retries, atomic resume state, and
bounded cleanup around those two components.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol
from urllib.parse import urlparse

from robomme_integration.eval import launch_gpu_fleet
from robomme_integration.eval import run_local_fixed50_queue as fixed50
from robomme_integration.fleet.checkpoint import verify as verify_checkpoint_manifest
from robomme_integration.training.arms import OFFICIAL_RECIPE_LEROBOT_ARM
from robomme_integration.training.single_task import TASK_EPISODES

QUEUE_KIND = "robomme_single_task_fixed50_eval_series"
# A19 checkpoint-maturity sweep (2026-09-02): the same fixed-50 lanes evaluate ONE multitask v4_70k
# checkpoint, per task, at each retained milestone (deploy/<step>).  A milestone queue is a distinct
# kind so it can never be mistaken for, or pooled with, the sealed single-task series.
MILESTONE_QUEUE_KIND = "robomme_multitask_milestone_fixed50_eval_series"
QUEUE_KINDS = frozenset({QUEUE_KIND, MILESTONE_QUEUE_KIND})
MILESTONE_COMPLETION_KIND = "robomme_gpu_milestone_checkpoint_set_complete"
MILESTONE_TRAINING_STEPS = frozenset({70_000})
MILESTONE_RUN_ID_PREFIX = "mt-v4-70k-all16-"
# What these lanes actually run (launch_gpu_fleet -> execution_model_server): predict 20, EXECUTE 10
# (execution_model_server refuses any other chunk_size), model_seed 7, the 50 fixed test indices of
# each task, max 1,300 steps.  That is the execute-10 ledger.  The paper protocol (h20/e16,
# robomme-paper856-h20-e16-fixed50-project-v1) is implemented only by the local project_exact runner
# (CAMPAIGNS.md W4: universes are never pooled).  Sealed into every milestone queue for that reason.
MILESTONE_EVAL_PROTOCOL = {
    "universe": "p5_fixed50_vla_eval_predict20_execute10_v1",
    "evaluator": "robomme_integration.eval.launch_gpu_fleet+execution_model_server",
    "model_seed": 7,
    "action_horizon": 20,
    "execute_steps": 10,
    "max_steps": 1300,
    "episodes_per_task": 50,
    "dataset": "test",
    "paper_protocol_id": "robomme-paper856-h20-e16-fixed50-project-v1",
    "paper_protocol_matched": False,
}
RUNTIME_KIND = "robomme_p5_native_eval_runtime_receipt"
PREFLIGHT_KIND = "robomme_p5_native_eval_preflight"
QUEUE_COMPLETION_KIND = "robomme_single_task_fixed50_eval_series_complete"
CELL_FAILURE_KIND = "robomme_single_task_fixed50_terminal_failure"
CLASSIFIER_VERSION = "robomme-eval-transients-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,159}$")
WORKSPACE_EVAL_ARMS = launch_gpu_fleet.WORKSPACE_ARMS
EVALUABLE_ARMS = frozenset(launch_gpu_fleet.EVALUABLE_ARMS)
TRAINING_COMPLETION_CURRENT = "scientific_and_manifest_v1"
TRAINING_COMPLETION_LEGACY = "legacy_manifest_derived_scientific_v1"
TRAINING_COMPLETION_BINDINGS = frozenset({TRAINING_COMPLETION_CURRENT, TRAINING_COMPLETION_LEGACY})
WORKSPACE_PROVENANCE_CLAIM = "producer_claim_v1"
WORKSPACE_PROVENANCE_LEGACY = "omega_manifest_checkpoint_tree_v1"

# These strings intentionally form a short allowlist.  A generic nonzero return code, policy-server
# exit, OOM, import error, missing asset, or incomplete fixed-50 is *not* transient.
TRANSIENT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vulkan_device_lost", ("vk_error_device_lost", "device lost (vulkan)")),
    (
        "harness_transport_reset",
        (
            "connection reset by peer",
            "websocket connection is closed",
            "websocket connection closed unexpectedly",
            "server disconnected without sending a response",
        ),
    ),
    (
        "harness_worker_interrupted",
        (
            "brokenprocesspool",
            "worker process terminated unexpectedly",
            "evaluation worker exited unexpectedly",
        ),
    ),
)

# These failures indicate that the sealed resource/runtime contract is unsafe for every active
# lane.  They are terminal (never retried), and the parallel supervisor cancels peer lanes without
# publishing fabricated failures for those interrupted cells.
SYSTEMIC_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "gpu_resource_exhausted",
        (
            "cuda_error_out_of_memory",
            "resource_exhausted: [0] failed to load in-memory cubin",
            "failed to load in-memory cubin",
        ),
    ),
)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _seal_digest(value: dict, field: str) -> str:
    clean = dict(value)
    clean.pop(field, None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def seal_document(value: dict, *, field: str) -> dict:
    """Return a shallow copy with a canonical self-seal (useful to queue builders/tests)."""
    result = dict(value)
    result.pop(field, None)
    result[field] = _seal_digest(result, field)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_milestone_queue(queue: dict) -> bool:
    return queue.get("kind") == MILESTONE_QUEUE_KIND


def is_milestone_cell(cell: dict) -> bool:
    """Milestone cells carry the run's deployed milestone set; validate_queue forbids it elsewhere."""
    return "deployed_milestones" in cell


def checkpoint_step(cell: dict) -> int:
    """The deployed step a cell evaluates: ``checkpoint_step`` for milestone cells, else the final step."""
    step = cell.get("checkpoint_step", cell["final_step"])
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"{cell.get('cell_id')} has an invalid checkpoint step {step!r}")
    return step


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_s3(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected an S3 URI string")
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI {value!r}")
    return value.rstrip("/")


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("relative path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path {value!r}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json_bytes(payload: bytes | None, *, label: str) -> dict | None:
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def verify_training_receipt_identity(
    receipt: dict,
    *,
    expected: dict,
    scientific_spec_sha256: str,
    expected_binding: str | None,
    label: str,
) -> str:
    """Verify a current or legacy checkpoint receipt without trusting an arm/task label.

    Legacy receipts predate the redundant ``scientific_spec_sha256`` field.  They remain
    authenticated only when they bind the exact sealed run manifest from which that digest is
    derived.  If the redundant field is present it is always required to match; a completion and
    its deploy receipt must also advertise the same binding mode.
    """
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} must be one JSON object")
    drift = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if drift:
        raise ValueError(f"{label} identity drift: {drift}")
    if "scientific_spec_sha256" in receipt:
        actual = _require_sha(
            receipt.get("scientific_spec_sha256"),
            f"{label} scientific digest",
        )
        if actual != scientific_spec_sha256:
            raise ValueError(f"{label} scientific digest differs from the sealed training manifest")
        binding = TRAINING_COMPLETION_CURRENT
    else:
        binding = TRAINING_COMPLETION_LEGACY
    if expected_binding is not None and binding != expected_binding:
        raise ValueError(f"{label} completion-binding mode drift: {binding} != {expected_binding}")
    return binding


def verify_checkpoint_tree_identity(
    payload: bytes | None,
    *,
    expected_sha256: str,
    checkpoint_uri: str,
    label: str,
) -> dict:
    """Authenticate a content-addressed checkpoint inventory before staging its objects."""
    if payload is None or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} is absent or corrupt")
    tree = _json_bytes(payload, label=label)
    assert tree is not None
    if tree.get("schema_version") != 1 or tree.get("checkpoint_uri") != checkpoint_uri:
        raise ValueError(f"{label} has the wrong checkpoint identity")
    objects = tree.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"{label} has no checkpoint objects")
    keys: set[str] = set()
    for record in objects:
        if not isinstance(record, dict) or set(record) != {"key", "sha256", "size_bytes"}:
            raise ValueError(f"{label} contains a malformed object record")
        key = _safe_relative(record.get("key"))
        if key in keys:
            raise ValueError(f"{label} contains duplicate object key {key!r}")
        keys.add(key)
        _require_sha(record.get("sha256"), f"{label} object digest")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"{label} object size is invalid")
    return tree


def _legacy_workspace_tree_sha256(root: Path, *, expected_step: int) -> str:
    """Reproduce the producer's pre-upload checkpoint-tree digest.

    ``workspace_materialize._tree_sha256`` ran on the finalized Orbax generation, after the
    three ``WSM_*`` receipts had been embedded.  The TPU transport subsequently added exactly
    one infrastructure sentinel, ``_UPLOAD_COMPLETE.json``; ``GcsRestore`` removes that sentinel
    before exposing the checkpoint to a consumer.  Authenticate its narrow transport contract,
    then exclude only that sentinel from the historical digest.
    """
    expected_entries = {
        "WSM_BEST.json",
        "WSM_GENERATION_COMPLETE.json",
        "WSM_RUN_CONFIG.json",
        "_CHECKPOINT_METADATA",
        "_UPLOAD_COMPLETE.json",
        "state",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_entries:
        raise ValueError("legacy workspace checkpoint has an unexpected root layout")
    for name in expected_entries - {"state"}:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError(f"legacy workspace checkpoint entry {name!r} is not a regular file")
    state = root / "state"
    if not state.is_dir() or state.is_symlink():
        raise ValueError("legacy workspace checkpoint state is not a regular directory")
    upload = _json_bytes(
        (root / "_UPLOAD_COMPLETE.json").read_bytes(),
        label="legacy workspace upload marker",
    )
    if upload is None or set(upload) != {
        "schema_version",
        "step",
        "completed_at",
        "source_marker",
    }:
        raise ValueError("legacy workspace upload marker has an unexpected schema")
    if (
        upload["schema_version"] != 1
        or upload["step"] != expected_step
        or upload["source_marker"] != "_CHECKPOINT_METADATA"
        or not isinstance(upload["completed_at"], str)
        or not upload["completed_at"]
    ):
        raise ValueError("legacy workspace upload marker identity drift")

    digest = hashlib.sha256()
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"legacy workspace checkpoint contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"legacy workspace checkpoint contains a non-file: {relative}")
        if relative.as_posix() == "_UPLOAD_COMPLETE.json":
            continue
        if any("orbax-checkpoint-tmp" in part or part.endswith(".incomplete") for part in relative.parts):
            raise ValueError(f"legacy workspace checkpoint contains a temporary file: {relative}")
        files.append(path)
    if not files:
        raise ValueError(f"empty legacy workspace checkpoint tree: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def verify_legacy_workspace_metadata(
    workspace: dict,
    *,
    task: str,
    omega_payload: bytes | None,
    completion_payload: bytes | None,
    run_config_payload: bytes | None,
    best_payload: bytes | None,
) -> None:
    """Verify the old producer's claim-equivalent, content-addressed metadata chain."""
    payloads = {
        "omega manifest": (omega_payload, workspace["omega_manifest_sha256"]),
        "representation completion": (completion_payload, workspace["completion_sha256"]),
        "workspace run config": (run_config_payload, workspace["run_config_sha256"]),
        "workspace best marker": (best_payload, workspace["best_sha256"]),
    }
    for label, (payload, expected_sha) in payloads.items():
        if payload is None or hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"legacy {label} is absent or corrupt")
    omega = _json_bytes(omega_payload, label="legacy omega manifest")
    completion = _json_bytes(completion_payload, label="legacy workspace completion")
    run_config = _json_bytes(run_config_payload, label="legacy workspace run config")
    best = _json_bytes(best_payload, label="legacy workspace best marker")
    assert all(value is not None for value in (omega, completion, run_config, best))
    identity = omega.get("encoder_identity")
    if not isinstance(identity, dict) or set(identity) != {
        "schema_version",
        "run_config_sha256",
        "checkpoint_step",
        "checkpoint_tree_sha256",
        "materializer_sha256",
    }:
        raise ValueError("legacy omega manifest has no exact encoder identity")
    derived_encoder = hashlib.sha256(
        (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    expected_omega = {
        "schema_version": 1,
        "task_name": task,
        "task_manifest_sha256": workspace["task_manifest_sha256"],
        "encoder_id": workspace["encoder_id"],
    }
    if any(omega.get(key) != value for key, value in expected_omega.items()):
        raise ValueError("legacy omega manifest task/encoder identity drift")
    representation = omega.get("representation")
    if not isinstance(representation, dict) or representation.get("uses_labels_at_inference") is not False:
        raise ValueError("legacy omega manifest does not forbid labels at inference")
    if derived_encoder != workspace["encoder_id"]:
        raise ValueError("legacy omega encoder identity digest mismatch")
    expected_identity = {
        "schema_version": 1,
        "run_config_sha256": workspace["run_config_sha256"],
        "checkpoint_step": workspace["step"],
        "checkpoint_tree_sha256": workspace["checkpoint_tree_sha256"],
        "materializer_sha256": workspace["materializer_sha256"],
    }
    if identity != expected_identity:
        raise ValueError("legacy omega encoder identity fields drifted")
    internal = dict(omega)
    claimed_internal = _require_sha(internal.pop("manifest_sha256", None), "legacy omega internal digest")
    internal_payload = json.dumps(internal, indent=2, sort_keys=True).encode() + b"\n"
    if hashlib.sha256(internal_payload).hexdigest() != claimed_internal:
        raise ValueError("legacy omega internal manifest seal mismatch")
    if completion != {
        "schema_version": 1,
        "step": workspace["step"],
        "run_config_sha256": workspace["run_config_sha256"],
        "embedded_sha256": {
            "WSM_BEST.json": workspace["best_sha256"],
            "WSM_RUN_CONFIG.json": workspace["run_config_sha256"],
        },
    }:
        raise ValueError("legacy workspace generation completion identity drift")
    if (
        run_config.get("schema_version") != 1
        or run_config.get("task") != task
        or run_config.get("task_manifest_sha256") != workspace["task_manifest_sha256"]
        or run_config.get("steps") != workspace["step"]
        or run_config.get("seed") != 0
        or run_config.get("omega_dim") != 512
    ):
        raise ValueError("legacy workspace run config scientific identity drift")
    best_score = best.get("best_score")
    if best != {
        "best_score": best.get("best_score"),
        "best_step": workspace["step"],
        "latest_step": workspace["step"],
        "run_config_sha256": workspace["run_config_sha256"],
    } or (
        not isinstance(best_score, (int, float))
        or isinstance(best_score, bool)
        or not math.isfinite(float(best_score))
    ):
        raise ValueError("legacy workspace best marker identity drift")


def validate_queue(value: dict, *, source_root: Path | None = None) -> dict:
    """Validate a self-sealed queue without reading cloud state."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported RoboMME eval queue schema")
    if value.get("kind") not in QUEUE_KINDS:
        raise ValueError(f"queue kind must be one of {sorted(QUEUE_KINDS)}")
    milestone = value.get("kind") == MILESTONE_QUEUE_KIND
    claimed = _require_sha(value.get("queue_manifest_sha256"), "queue manifest seal")
    if _seal_digest(value, "queue_manifest_sha256") != claimed:
        raise ValueError("queue manifest seal mismatch")
    queue_id = value.get("queue_id")
    if not isinstance(queue_id, str) or not SAFE_ID.fullmatch(queue_id):
        raise ValueError("unsafe queue_id")

    publish_root = _safe_s3(value.get("publish_root_s3"))
    if f"/{queue_id}" not in publish_root:
        raise ValueError("publish_root_s3 must be queue-specific")
    claims = value.get("claims")
    if not isinstance(claims, dict) or set(claims) != {"manifest", "completion"}:
        raise ValueError("queue claims must contain manifest and completion")
    if _safe_s3(claims["manifest"]) != f"{publish_root}/manifest.json":
        raise ValueError("queue manifest claim is not canonical")
    if _safe_s3(claims["completion"]) != f"{publish_root}/complete.json":
        raise ValueError("queue completion claim is not canonical")

    gates = value.get("gates")
    if not isinstance(gates, dict) or set(gates) != {"native_preflight", "runtime_receipt"}:
        raise ValueError("queue must pin native_preflight and runtime_receipt gates")
    preflight = gates["native_preflight"]
    runtime = gates["runtime_receipt"]
    if not isinstance(preflight, dict) or not isinstance(runtime, dict):
        raise ValueError("queue gates must be objects")
    _require_sha(preflight.get("claim_sha256"), "preflight claim digest")
    _require_sha(preflight.get("source_tree_sha256"), "preflight source-tree digest")
    if not isinstance(preflight.get("preflight_id"), str) or not SAFE_ID.fullmatch(preflight["preflight_id"]):
        raise ValueError("invalid preflight_id")
    _require_sha(runtime.get("receipt_sha256"), "runtime receipt file digest")
    _require_sha(runtime.get("runtime_artifact_sha256"), "runtime artifact digest")
    _require_sha(runtime.get("openpi_sha256"), "runtime OpenPI digest")

    topology = value.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("queue topology must be an object")
    execution_mode = topology.get("execution_mode")
    if execution_mode not in {None, "parallel_fixed50_lanes_v1"}:
        raise ValueError("queue topology has an unknown execution mode")
    if execution_mode == "parallel_fixed50_lanes_v1":
        if topology.get("schema_version") != 1 or not isinstance(topology.get("lanes"), list):
            raise ValueError("parallel queue topology has an invalid schema")
        _require_sha(topology.get("parallel_topology_sha256"), "parallel topology self-seal")
    gpus = topology.get("policy_gpus")
    simulator_gpus = topology.get("simulator_gpus")
    if (
        not isinstance(gpus, list)
        or not gpus
        or len(gpus) != len(set(gpus))
        or any(not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0 for gpu in gpus)
    ):
        raise ValueError("policy_gpus must be distinct nonnegative integers")
    if (
        not isinstance(simulator_gpus, list)
        or not simulator_gpus
        or any(gpu not in range(8) for gpu in simulator_gpus)
    ):
        raise ValueError("simulator_gpus must select p5 GPU IDs 0..7")
    shards = topology.get("simulator_shards")
    if not isinstance(shards, int) or isinstance(shards, bool) or not len(gpus) <= shards <= 32:
        raise ValueError("simulator_shards must lie in [number of policy GPUs, 32]")
    if not isinstance(topology.get("cpu_range"), str) or not re.fullmatch(r"\d+-\d+", topology["cpu_range"]):
        raise ValueError("cpu_range must be an inclusive integer range")
    if not isinstance(topology.get("base_port"), int) or not 1024 <= topology["base_port"] <= 65527:
        raise ValueError("invalid campaign base_port")
    fraction = topology.get("xla_memory_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0.1 <= fraction <= 0.95:
        raise ValueError("xla_memory_fraction must lie in [0.1, 0.95]")

    retry = value.get("retry")
    if not isinstance(retry, dict) or retry.get("classifier_version") != CLASSIFIER_VERSION:
        raise ValueError("queue retry classifier is absent or unrecognized")
    max_attempts = retry.get("max_attempts")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 3:
        raise ValueError("retry.max_attempts must lie in [1, 3]")
    if set(retry) != {"classifier_version", "max_attempts"}:
        raise ValueError("queue may not inject custom retry markers")

    limits = value.get("limits")
    if not isinstance(limits, dict) or set(limits) != {
        "max_run_seconds",
        "runtime_reserve_seconds",
        "estimated_cell_seconds",
        "minimum_free_bytes",
    }:
        raise ValueError("queue must seal runtime and disk limits")
    maximum = limits["max_run_seconds"]
    reserve = limits["runtime_reserve_seconds"]
    estimate = limits["estimated_cell_seconds"]
    minimum_free = limits["minimum_free_bytes"]
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 86_400:
        raise ValueError("eval max_run_seconds must lie in [1, 86400]")
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 600:
        raise ValueError("eval runtime_reserve_seconds must be at least 600")
    if not isinstance(estimate, int) or isinstance(estimate, bool) or estimate < 1:
        raise ValueError("eval estimated_cell_seconds must be positive")
    if not isinstance(minimum_free, int) or isinstance(minimum_free, bool) or minimum_free < 1:
        raise ValueError("eval minimum_free_bytes must be positive")

    comparability = value.get("comparability")
    comparability_fields = {
        "serving_openpi",
        "task_benchmark_configs",
        "task_common_training_nuisance",
        "sequence_geometry_policy",
    }
    if milestone:
        comparability_fields = comparability_fields | {"eval_protocol"}
    if not isinstance(comparability, dict) or set(comparability) != comparability_fields:
        raise ValueError("queue must seal benchmark and training comparability identities")
    if milestone and comparability["eval_protocol"] != MILESTONE_EVAL_PROTOCOL:
        raise ValueError("milestone queue must declare the exact execute-10 fixed-50 evaluation protocol universe")
    if comparability["sequence_geometry_policy"] != "manifest_verified_per_cell_not_assumed_common":
        raise ValueError("queue must expose rather than hide sequence-geometry differences")
    task_configs = comparability["task_benchmark_configs"]
    common_nuisance = comparability["task_common_training_nuisance"]
    if not isinstance(task_configs, dict) or not isinstance(common_nuisance, dict):
        raise ValueError("task comparability records must be objects")
    serving_openpi = comparability["serving_openpi"]
    if not isinstance(serving_openpi, dict) or set(serving_openpi) != {"uri", "sha256"}:
        raise ValueError("queue must pin one training-matched OpenPI serving artifact")
    _safe_s3(serving_openpi["uri"])
    _require_sha(serving_openpi["sha256"], "serving OpenPI digest")
    if serving_openpi["sha256"] not in serving_openpi["uri"]:
        raise ValueError("serving OpenPI URI must be content addressed")

    cells = value.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("eval queue has no cells")
    identities: set[tuple] = set()
    run_id_uses: dict[str, list[tuple]] = {}
    represented_tasks: set[str] = set()
    for ordinal, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("ordinal") != ordinal:
            raise ValueError("queue cell ordinals must be contiguous and zero based")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not SAFE_ID.fullmatch(cell_id):
            raise ValueError(f"unsafe cell_id at ordinal {ordinal}")
        task, arm = cell.get("task"), cell.get("arm")
        if task not in TASK_EPISODES:
            raise ValueError(f"unsupported single-task RoboMME task {task!r}")
        if arm not in EVALUABLE_ARMS or arm == OFFICIAL_RECIPE_LEROBOT_ARM:
            raise ValueError(f"unsupported fixed-50 arm {arm!r}")
        guidance = cell.get("cfg_guidance_scale", 1.0)
        cfg_identity = float(guidance) if arm in {"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"} else None
        run_id = cell.get("run_id")
        if milestone:
            # One multitask checkpoint is evaluated per task at several retained milestones, so a
            # run_id legitimately repeats; the eval identity is (task, arm, cfg, run, step).
            if not isinstance(run_id, str) or not run_id.startswith(MILESTONE_RUN_ID_PREFIX):
                raise ValueError(f"invalid multitask v4_70k run_id for {cell_id}")
            final_step = cell.get("final_step")
            deployed = cell.get("deployed_milestones")
            if (
                not isinstance(final_step, int)
                or isinstance(final_step, bool)
                or not isinstance(deployed, list)
                or not deployed
                or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in deployed)
                or deployed != sorted(set(deployed))
                or deployed[-1] != final_step
            ):
                raise ValueError(f"{cell_id} must carry the run's deployed milestone set ending at its final step")
            step = checkpoint_step(cell)
            if step not in deployed:
                raise ValueError(
                    f"{cell_id} evaluates step {step}, which is not in the run's deployed milestone set {deployed}"
                )
            identity = (task, arm, cfg_identity, run_id, step)
            if identity in identities:
                raise ValueError(f"duplicate task/arm/eval/run/step identity {identity}")
            identities.add(identity)
            represented_tasks.add(task)
            run_id_uses.setdefault(run_id, []).append(identity)
            completion_step = final_step
        else:
            if "checkpoint_step" in cell or "deployed_milestones" in cell:
                raise ValueError(f"{cell_id} single-task cells may not carry milestone fields")
            identity = (task, arm, cfg_identity)
            if identity in identities:
                raise ValueError(f"duplicate task/arm/eval identity {identity}")
            identities.add(identity)
            represented_tasks.add(task)
            if not isinstance(run_id, str) or not run_id.startswith(("st-v1-", "st-v4-")):
                raise ValueError(f"invalid single-task run_id for {cell_id}")
            prior_uses = run_id_uses.setdefault(run_id, [])
            if prior_uses and (
                arm not in {"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}
                or any(previous[:2] != (task, arm) for previous in prior_uses)
            ):
                raise ValueError(f"duplicate run_id is valid only for one v4 CFG scale sweep: {cell_id}")
            prior_uses.append(identity)
            if cell.get("final_step") != 19_999:
                raise ValueError(f"{cell_id} must evaluate final step 19999")
            step = completion_step = 19_999
        _require_sha(cell.get("scientific_spec_sha256"), f"{cell_id} scientific digest")
        _require_sha(cell.get("run_manifest_sha256"), f"{cell_id} run-manifest digest")
        if cell.get("training_openpi") != serving_openpi:
            raise ValueError(f"{cell_id} training/serving OpenPI identity differs")
        manifest_uri = _safe_s3(cell.get("training_run_manifest_s3"))
        manifest_pattern = (
            rf"/manifests/runs/train/{re.escape(run_id)}/"
            rf"{re.escape(run_id)}-attempt\d+\.json$"
        )
        if not re.search(manifest_pattern, manifest_uri):
            raise ValueError(f"{cell_id} training manifest URI does not bind its run/attempt")
        output = _safe_s3(cell.get("training_output_s3"))
        expected_tail = f"/all16/{arm}/seed0/{run_id}" if milestone else f"/{task}/{arm}/seed0/{run_id}"
        if not output.endswith(expected_tail):
            raise ValueError(f"training output does not bind {task}/{arm}/{run_id}")
        completion = _safe_s3(cell.get("training_completion_claim_s3"))
        if not completion.endswith(f"/train/{run_id}/step-{completion_step}.complete.json"):
            raise ValueError(f"training completion claim does not bind {run_id}")
        if cell.get("training_completion_binding") not in TRAINING_COMPLETION_BINDINGS:
            raise ValueError(f"{cell_id} has an unrecognized training completion binding")
        config = _safe_relative(cell.get("benchmark_config"))
        config_sha = _require_sha(cell.get("benchmark_config_sha256"), f"{cell_id} config digest")
        expected_config = task_configs.get(task)
        if expected_config != {"path": config, "sha256": config_sha}:
            raise ValueError(f"{cell_id} differs from the task-common benchmark config")
        nuisance = cell.get("training_nuisance")
        nuisance_fields = {
            "data_parent_inventory_sha256",
            "data_task_inventory_sha256",
            "initialization_inventory_sha256",
            "initialization_checkpoint_s3",
            "seed",
            "steps",
            "action_horizon",
            "window_len",
            "chunk_stride",
        }
        if not isinstance(nuisance, dict) or set(nuisance) != nuisance_fields:
            raise ValueError(f"{cell_id} must pin the exact training nuisance recipe")
        sha_fields = [
            "data_parent_inventory_sha256",
            "data_task_inventory_sha256",
            "initialization_inventory_sha256",
        ]
        if milestone:
            # all16 training consumes the parent inventory directly; there is no derived task view.
            sha_fields.remove("data_task_inventory_sha256")
            if nuisance.get("data_task_inventory_sha256") is not None:
                raise ValueError(f"{cell_id} multitask all16 training has no derived task inventory")
        for name in sha_fields:
            _require_sha(nuisance.get(name), f"{cell_id} {name}")
        _safe_s3(nuisance.get("initialization_checkpoint_s3"))
        if milestone:
            if nuisance.get("steps") not in MILESTONE_TRAINING_STEPS or (
                nuisance.get("seed"),
                nuisance.get("action_horizon"),
            ) != (0, 20):
                raise ValueError(f"{cell_id} is outside the matched seed0/70k/horizon20 multitask maturity study")
            if nuisance["steps"] - 1 != completion_step:
                raise ValueError(f"{cell_id} final_step does not match its training step count")
        elif (nuisance.get("seed"), nuisance.get("steps"), nuisance.get("action_horizon")) != (0, 20_000, 20):
            raise ValueError(f"{cell_id} is outside the matched seed0/20k/horizon20 study")
        geometry = (nuisance.get("window_len"), nuisance.get("chunk_stride"))
        if geometry not in {(None, None), (8, 10)}:
            raise ValueError(f"{cell_id} has unrecognized action-sequence geometry {geometry}")
        common_fields = nuisance_fields - {"window_len", "chunk_stride"}
        expected_common = common_nuisance.get(task)
        actual_common = {name: nuisance[name] for name in sorted(common_fields)}
        if expected_common != actual_common:
            raise ValueError(f"{cell_id} differs from the task-common training nuisance recipe")
        scale_token = f"-cfgs{int(round(float(guidance) * 100)):03d}" if cfg_identity is not None else ""
        if scale_token and not cell_id.endswith(scale_token):
            raise ValueError(f"{cell_id} v4 CFG cell_id does not bind guidance scale {guidance}")
        expected_eval_id = (
            f"{run_id}-s{step}-{task.lower()}-{queue_id}{scale_token}"
            if milestone
            else f"{run_id}-fixed50-{queue_id}{scale_token}"
        )
        if cell.get("eval_id") != expected_eval_id or not SAFE_ID.fullmatch(expected_eval_id):
            raise ValueError(f"{cell_id} eval_id is not queue/run exact")
        expected_result = f"{publish_root}/cells/{cell_id}/result.complete.json"
        if _safe_s3(cell.get("result_claim_s3")) != expected_result:
            raise ValueError(f"{cell_id} result claim is not queue-specific")
        workspace = cell.get("workspace")
        if arm in WORKSPACE_EVAL_ARMS:
            if not isinstance(workspace, dict):
                raise ValueError(f"{cell_id} requires exact workspace representation inputs")
            provenance = workspace.get("provenance_mode")
            common_workspace = {
                "provenance_mode",
                "encoder_id",
                "representation_s3",
                "completion_sha256",
                "step",
            }
            if provenance == WORKSPACE_PROVENANCE_CLAIM:
                required_workspace = common_workspace | {"claim_s3", "claim_sha256"}
            elif provenance == WORKSPACE_PROVENANCE_LEGACY:
                required_workspace = common_workspace | {
                    "omega_manifest_s3",
                    "omega_manifest_sha256",
                    "task_manifest_sha256",
                    "checkpoint_tree_sha256",
                    "run_config_sha256",
                    "best_sha256",
                    "materializer_sha256",
                }
            else:
                raise ValueError(f"{cell_id} workspace provenance mode is unrecognized")
            if set(workspace) != required_workspace:
                raise ValueError(f"{cell_id} workspace fields are not exact")
            _safe_s3(workspace["representation_s3"])
            _require_sha(workspace["encoder_id"], f"{cell_id} workspace encoder identity")
            _require_sha(workspace["completion_sha256"], f"{cell_id} workspace completion digest")
            if not isinstance(workspace["step"], int) or isinstance(workspace["step"], bool) or workspace["step"] < 1:
                raise ValueError(f"{cell_id} workspace step must be positive")
            if provenance == WORKSPACE_PROVENANCE_CLAIM:
                _safe_s3(workspace["claim_s3"])
                _require_sha(workspace["claim_sha256"], f"{cell_id} workspace claim digest")
            else:
                _safe_s3(workspace["omega_manifest_s3"])
                for name in (
                    "omega_manifest_sha256",
                    "task_manifest_sha256",
                    "checkpoint_tree_sha256",
                    "run_config_sha256",
                    "best_sha256",
                    "materializer_sha256",
                ):
                    _require_sha(workspace[name], f"{cell_id} legacy workspace {name}")
                artifact_root = workspace["omega_manifest_s3"].removesuffix("/omega/MANIFEST.json")
                expected_root = f"/artifacts/robomme/workspace/{task}/{workspace['encoder_id']}"
                if not artifact_root.endswith(expected_root):
                    raise ValueError(f"{cell_id} legacy workspace artifact root is not canonical")
                if workspace["representation_s3"] != (f"{artifact_root}/representation/step-{workspace['step']}"):
                    raise ValueError(f"{cell_id} legacy representation URI is not encoder/step bound")
        elif workspace is not None:
            raise ValueError(f"{cell_id} non-workspace serving arm forbids workspace inputs")
        if not isinstance(guidance, (int, float)) or isinstance(guidance, bool) or not math.isfinite(guidance):
            raise ValueError(f"{cell_id} CFG scale must be finite")
        cfg_arms = {"wsm_cfg", "v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}
        if arm not in cfg_arms and float(guidance) != 1.0:
            raise ValueError(f"{cell_id} CFG scale is valid only for wsm_cfg")
        if arm in {"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"} and float(guidance) not in {
            0.5,
            1.0,
            1.5,
            2.0,
        }:
            raise ValueError(f"{cell_id} RoboMME v4 CFG scale is outside the sealed sweep")
        ptrm = cell.get("ptrm")
        if arm in {"ptrm", "v4_ptrm"}:
            if not isinstance(ptrm, dict) or set(ptrm) != {"eval_k", "eval_sigma", "eval_select"}:
                raise ValueError(f"{cell_id} requires exact PTRM inference knobs")
            if (ptrm["eval_k"], ptrm["eval_sigma"], ptrm["eval_select"]) != (1, 0.0, "q"):
                raise ValueError(f"{cell_id} PTRM is E0-only: K=1, sigma=0, select=q")
        elif ptrm is not None:
            raise ValueError(f"{cell_id} non-PTRM arm forbids PTRM inference knobs")

        if source_root is not None:
            root = source_root.resolve()
            path = (root / config).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"{cell_id} benchmark config is absent")
            if _sha256(path) != config_sha:
                raise ValueError(f"{cell_id} benchmark config digest mismatch")
    if set(task_configs) != represented_tasks or set(common_nuisance) != represented_tasks:
        raise ValueError("comparability maps must cover exactly the queue tasks")
    if not milestone:
        for run_id, uses in run_id_uses.items():
            if len(uses) > 1 and {use[2] for use in uses} != {0.5, 1.0, 1.5, 2.0}:
                raise ValueError(f"{run_id} repeated v4 CFG checkpoint must cover the exact four-scale sweep")
    return value


@dataclass(frozen=True)
class Runtime:
    receipt_sha256: str
    preflight_claim_sha256: str
    policy_python: Path
    vla_eval: Path
    harness_src: Path
    robomme_src: Path
    maniskill_src: Path
    openpi_src: Path
    policy_site: Path
    simulator_site: Path
    upstream_root: Path
    vision_encoder_home: Path
    render_environment: dict[str, str]


def policy_pythonpath_entries(
    *,
    source_root: Path,
    harness_src: Path,
    openpi_src: Path,
    policy_site: Path,
    upstream_root: Path,
    simulator_site: Path,
) -> tuple[Path, ...]:
    """Exact policy-server import order; the simulator site is fallback-only."""
    return (
        source_root,
        harness_src,
        openpi_src,
        policy_site,
        upstream_root / "src",
        simulator_site,
    )


def verify_gates(
    queue: dict,
    *,
    preflight_claim: Path,
    runtime_receipt: Path,
    require_runtime_paths: bool = True,
) -> Runtime:
    """Authenticate the rendered-reset claim and the exact staged runtime it certified."""
    preflight_bytes = preflight_claim.read_bytes()
    runtime_bytes = runtime_receipt.read_bytes()
    expected_preflight = queue["gates"]["native_preflight"]
    expected_runtime = queue["gates"]["runtime_receipt"]
    if hashlib.sha256(preflight_bytes).hexdigest() != expected_preflight["claim_sha256"]:
        raise ValueError("native-render preflight claim file digest mismatch")
    if hashlib.sha256(runtime_bytes).hexdigest() != expected_runtime["receipt_sha256"]:
        raise ValueError("native runtime receipt file digest mismatch")
    preflight = _json_bytes(preflight_bytes, label="native-render preflight claim")
    receipt = _json_bytes(runtime_bytes, label="native runtime receipt")
    assert preflight is not None and receipt is not None
    if preflight.get("preflight_id") != expected_preflight["preflight_id"]:
        raise ValueError("native-render preflight identity mismatch")
    if preflight.get("source_tree_sha256") != expected_preflight["source_tree_sha256"]:
        raise ValueError("native-render preflight source tree mismatch")
    if preflight.get("vla_eval_entrypoint") != {
        "kind": "python_module_wrapper",
        "module": "vla_eval.cli.main",
    }:
        raise ValueError("preflight claim did not exercise the relocatable vla-eval wrapper")
    infrastructure = preflight.get("infrastructure", {})
    if infrastructure.get("instance_type") != "ml.p5.48xlarge" or infrastructure.get("accelerator") != "8xH100":
        raise ValueError("native-render preflight was not run on the p5/H100 evaluation target")
    p5_parallel_action_mode = (
        queue["topology"].get("execution_mode") == "parallel_fixed50_lanes_v1"
        and queue["topology"].get("topology_id") == "p5-8xh100-fixed50-v1"
    )
    if p5_parallel_action_mode:
        from robomme_integration.eval import p5_parallel_action_preflight as action_canary
        from robomme_integration.launch import IMAGE, IMAGE_SHA

        action_canary.validate_success_claim(
            preflight,
            source_sha256=expected_preflight["source_tree_sha256"],
            expected_openpi=queue["comparability"]["serving_openpi"],
            expected_topology=queue["topology"],
            expected_image={"uri": IMAGE, "sha256": IMAGE_SHA},
            expected_vision=receipt.get("vision"),
            expected_upstream=receipt.get("upstream"),
            expected_infrastructure={
                "instance_type": "ml.p5.48xlarge",
                "accelerator": "8xH100",
                "priority": 100,
            },
        )
    else:
        if preflight.get("kind") != PREFLIGHT_KIND or preflight.get("status") != "native_render_reset_passed":
            raise ValueError("p5 native-render preflight did not pass")
        expected_probe = {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
        }
        if preflight.get("probe") != expected_probe:
            raise ValueError("preflight claim lacks the exact paired video/state demonstration rendered reset")
        # The preflight publisher seals the manifest before appending status.
        sealed_preflight = dict(preflight)
        manifest_sha = _require_sha(sealed_preflight.pop("manifest_sha256", None), "preflight manifest seal")
        sealed_preflight.pop("status", None)
        if _seal_digest(sealed_preflight, "manifest_sha256") != manifest_sha:
            raise ValueError("native-render preflight manifest seal mismatch")

    if receipt.get("kind") != RUNTIME_KIND or receipt.get("status") != "staged_and_verified":
        raise ValueError("native runtime receipt is not a verified staged runtime")
    if _seal_digest(receipt, "receipt_sha256") != _require_sha(
        receipt.get("receipt_sha256"), "runtime receipt self-seal"
    ):
        raise ValueError("native runtime receipt self-seal mismatch")
    if receipt.get("preflight_claim_sha256") != expected_preflight["claim_sha256"]:
        raise ValueError("runtime receipt is not bound to the supplied rendered-reset claim")
    if receipt.get("runtime") != preflight.get("runtime") or receipt.get("openpi") != preflight.get("openpi"):
        raise ValueError("runtime receipt artifacts disagree with rendered-reset preflight")
    if receipt.get("runtime", {}).get("sha256") != expected_runtime["runtime_artifact_sha256"]:
        raise ValueError("runtime artifact digest disagrees with queue")
    if receipt.get("openpi", {}).get("sha256") != expected_runtime["openpi_sha256"]:
        raise ValueError("OpenPI artifact digest disagrees with queue")
    if receipt.get("openpi") != queue["comparability"]["serving_openpi"]:
        raise ValueError("runtime OpenPI differs from the training-matched queue source")

    paths = receipt.get("paths")
    expected_paths = {
        "policy_python",
        "vla_eval",
        "harness_src",
        "robomme_src",
        "maniskill_src",
        "openpi_src",
        "policy_site",
        "simulator_site",
        "upstream_root",
        "vision_encoder_home",
    }
    if not isinstance(paths, dict) or set(paths) != expected_paths:
        raise ValueError("runtime receipt paths are incomplete or expanded")
    resolved: dict[str, Path] = {}
    for name in expected_paths:
        path = Path(paths[name]) if isinstance(paths[name], str) else Path("")
        if not path.is_absolute():
            raise ValueError(f"runtime path {name} must be absolute")
        # Preserve lexical executable paths.  In particular, resolving ``.venv/bin/python`` to
        # its base interpreter target disables Python's venv discovery and changes the installed
        # distribution set.  Directory identities may still be canonicalized.
        resolved[name] = path if name in {"policy_python", "vla_eval"} else path.resolve()
    if require_runtime_paths:
        for name in ("policy_python", "vla_eval"):
            if not resolved[name].is_file() or not os.access(resolved[name], os.X_OK):
                raise ValueError(f"runtime executable {name} is absent")
        for name in expected_paths - {"policy_python", "vla_eval"}:
            if not resolved[name].is_dir():
                raise ValueError(f"runtime directory {name} is absent")
        if not (resolved["vision_encoder_home"] / "pi05_vision_encoder/siglip_params.pkl").is_file():
            raise ValueError("runtime receipt lacks pinned pi0.5 vision weights")
        wrapper = receipt.get("vla_eval_wrapper")
        if (
            not isinstance(wrapper, dict)
            or set(wrapper) != {"kind", "module", "sha256"}
            or wrapper.get("kind") != "python_module_wrapper"
            or wrapper.get("module") != "vla_eval.cli.main"
        ):
            raise ValueError("runtime receipt lacks the exact relocatable vla-eval wrapper")
        _require_sha(wrapper["sha256"], "vla-eval wrapper digest")
        if _sha256(resolved["vla_eval"]) != wrapper["sha256"]:
            raise ValueError("relocatable vla-eval wrapper digest mismatch")
    render_environment = receipt.get("render_environment")
    # 2026-09-05 (preflight 43a532a7…): "auto" probes the native Vulkan path in a child process with a
    # hard-coded 15 s watchdog (vla_eval.benchmarks.robomme.benchmark.native_render_path_works). With
    # 8 lanes × 4 shards probing at once next to 8 JIT-compiling policy servers, 1–2 probes per lane
    # timed out, the harness declared the native path "hung" and switched to lavapipe, whose ICD the
    # image does not ship → RuntimeError at reset. The native path rendered in 24/32 shards on the
    # same node, so the p5 native lanes pin it; a genuinely affected host now fails loudly at the
    # 1200 s stall watchdog instead of flapping.
    if render_environment != {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "ROBOMME_USE_LAVAPIPE": "0",
    }:
        raise ValueError("runtime render environment is not the sealed EGL/native-preflight contract")
    return Runtime(
        receipt_sha256=expected_runtime["receipt_sha256"],
        preflight_claim_sha256=expected_preflight["claim_sha256"],
        policy_python=resolved["policy_python"],
        vla_eval=resolved["vla_eval"],
        harness_src=resolved["harness_src"],
        robomme_src=resolved["robomme_src"],
        maniskill_src=resolved["maniskill_src"],
        openpi_src=resolved["openpi_src"],
        policy_site=resolved["policy_site"],
        simulator_site=resolved["simulator_site"],
        upstream_root=resolved["upstream_root"],
        vision_encoder_home=resolved["vision_encoder_home"],
        render_environment=dict(render_environment),
    )


class ObjectStore(Protocol):
    def read_bytes(self, uri: str) -> bytes | None: ...

    def exists(self, uri: str) -> bool: ...

    def put_bytes_once(self, payload: bytes, uri: str) -> None: ...

    def put_file_once(self, path: Path, uri: str) -> None: ...


class AwsCliStore:
    def __init__(
        self,
        *,
        region: str = "us-west-2",
        cancel_event: threading.Event | None = None,
        cancel_poll_seconds: float = 0.5,
    ) -> None:
        self.region = region
        self.cancel_event = cancel_event
        self.cancel_poll_seconds = cancel_poll_seconds

    def _run(
        self,
        command: list[str],
        *,
        stdout: int | None,
        stderr: int | None,
    ) -> subprocess.CompletedProcess:
        """Run one AWS CLI call in an isolated, optionally cancellable process group."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise EvaluatorCancelled("parallel campaign cancelled before S3 object access")
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            if self.cancel_event is None:
                output, error = process.communicate()
            else:
                while True:
                    if self.cancel_event.is_set():
                        raise EvaluatorCancelled("parallel campaign cancelled during S3 object access")
                    try:
                        output, error = process.communicate(timeout=self.cancel_poll_seconds)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        except BaseException as original:
            try:
                _terminate_evaluator_process(process)
                process.communicate()
            except Exception as cleanup_error:
                failure = EvaluatorCleanupFailure(
                    f"S3 object-access process-group cleanup failed after {type(original).__name__}: {original}"
                )
                failure.add_note(f"cleanup error: {type(cleanup_error).__name__}: {cleanup_error}")
                raise failure from cleanup_error
            raise
        return subprocess.CompletedProcess(command, process.returncode, output, error)

    def read_bytes(self, uri: str) -> bytes | None:
        result = self._run(
            ["aws", "s3", "cp", uri, "-", "--only-show-errors", "--region", self.region],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return result.stdout
        detail = result.stderr.decode(errors="replace").casefold()
        if any(marker in detail for marker in ("404", "not found", "nosuchkey", "does not exist")):
            return None
        raise RuntimeError(f"S3 read failed for {uri}: {detail[:500]}")

    def exists(self, uri: str) -> bool:
        parsed = urlparse(_safe_s3(uri))
        result = self._run(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                parsed.netloc,
                "--key",
                parsed.path.lstrip("/"),
                "--region",
                self.region,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return True
        detail = result.stderr.decode(errors="replace").casefold()
        if any(marker in detail for marker in ("404", "not found", "nosuchkey")):
            return False
        raise RuntimeError(f"S3 head failed for {uri}: {detail[:500]}")

    def put_bytes_once(self, payload: bytes, uri: str) -> None:
        fd, name = tempfile.mkstemp(prefix="robomme-eval-campaign-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            self.put_file_once(Path(name), uri)
        finally:
            Path(name).unlink(missing_ok=True)

    def put_file_once(self, path: Path, uri: str) -> None:
        parsed = urlparse(_safe_s3(uri))
        result = self._run(
            [
                "aws",
                "s3api",
                "put-object",
                "--bucket",
                parsed.netloc,
                "--key",
                parsed.path.lstrip("/"),
                "--body",
                str(path),
                "--if-none-match",
                "*",
                "--region",
                self.region,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return
        existing = self.read_bytes(uri)
        if existing is None or existing != path.read_bytes():
            detail = result.stderr.decode(errors="replace")[:500]
            raise RuntimeError(f"immutable S3 collision at {uri}: {detail}")


class Stager(Protocol):
    def stage_checkpoint(self, cell: dict, destination: Path) -> str: ...

    def stage_workspace(self, cell: dict, destination: Path) -> Path | None: ...


class AwsStager:
    def __init__(
        self,
        store: ObjectStore,
        *,
        region: str = "us-west-2",
        cancel_event: threading.Event | None = None,
        cancel_poll_seconds: float = 0.5,
    ) -> None:
        self.store = store
        self.region = region
        self.cancel_event = cancel_event
        self.cancel_poll_seconds = cancel_poll_seconds

    def _sync(self, uri: str, destination: Path, *, checkpoint: bool = False) -> None:
        destination.mkdir(parents=True)
        command = ["aws", "s3", "sync", uri, str(destination), "--only-show-errors"]
        if checkpoint:
            command.extend(["--exclude", "*", "--include", "params/*", "--include", "assets/*"])
        command.extend(["--region", self.region])
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise EvaluatorCancelled("parallel campaign cancelled before S3 staging")
        process = subprocess.Popen(command, start_new_session=True)
        try:
            if self.cancel_event is None:
                returncode = process.wait()
            else:
                while True:
                    if self.cancel_event.is_set():
                        raise EvaluatorCancelled("parallel campaign cancelled during S3 staging")
                    try:
                        returncode = process.wait(timeout=self.cancel_poll_seconds)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        except BaseException as error:
            try:
                _terminate_evaluator_process(process)
            except Exception as cleanup_error:
                failure = EvaluatorCleanupFailure(
                    f"S3 staging process-group cleanup failed after {type(error).__name__}: {error}"
                )
                failure.add_note(f"cleanup error: {type(cleanup_error).__name__}: {cleanup_error}")
                raise failure from cleanup_error
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    def _verify_training_manifest(self, cell: dict) -> dict:
        uri = cell["training_run_manifest_s3"]
        manifest = _json_bytes(self.store.read_bytes(uri), label=uri)
        if manifest is None:
            raise ValueError(f"training run manifest is absent for {cell['cell_id']}")
        manifest_sha = _require_sha(manifest.get("manifest_sha256"), "training manifest seal")
        if manifest_sha != cell["run_manifest_sha256"]:
            raise ValueError("training run manifest digest differs from the eval cell")
        if _seal_digest(manifest, "manifest_sha256") != manifest_sha:
            raise ValueError("training run manifest self-seal mismatch")
        scientific = manifest.get("scientific")
        if not isinstance(scientific, dict):
            raise ValueError("training run manifest has no scientific specification")
        scientific_sha = hashlib.sha256(
            json.dumps(
                scientific,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        expected_identity = {
            "schema_version": 2,
            "kind": "robomme_gpu_training_attempt",
            "run_id": cell["run_id"],
            "scientific_spec_sha256": cell["scientific_spec_sha256"],
            "manifest_s3": uri,
            "output_s3": cell["training_output_s3"],
        }
        drift = {
            key: (manifest.get(key), expected)
            for key, expected in expected_identity.items()
            if manifest.get(key) != expected
        }
        if scientific_sha != cell["scientific_spec_sha256"]:
            drift["scientific_payload_sha256"] = (
                scientific_sha,
                cell["scientific_spec_sha256"],
            )
        claims = manifest.get("claims")
        if not isinstance(claims, dict) or claims.get("completion") != cell["training_completion_claim_s3"]:
            drift["completion_claim"] = (
                claims.get("completion") if isinstance(claims, dict) else None,
                cell["training_completion_claim_s3"],
            )
        task = scientific.get("task")
        if is_milestone_cell(cell):
            # A multitask (all16) checkpoint is evaluated one task at a time; the cell's task must
            # be one the run actually trained on.
            trained = task.get("tasks") if isinstance(task, dict) else None
            if (
                not isinstance(task, dict)
                or task.get("name") != "all16"
                or not isinstance(trained, list)
                or cell["task"] not in trained
            ):
                drift["task"] = (
                    task.get("name") if isinstance(task, dict) else None,
                    f"all16 including {cell['task']}",
                )
        elif not isinstance(task, dict) or task.get("name") != cell["task"]:
            drift["task"] = (task.get("name") if isinstance(task, dict) else None, cell["task"])
        if scientific.get("arm") != cell["arm"]:
            drift["arm"] = (scientific.get("arm"), cell["arm"])
        if scientific.get("sources", {}).get("openpi") != cell["training_openpi"]:
            drift["training_openpi"] = (
                scientific.get("sources", {}).get("openpi"),
                cell["training_openpi"],
            )
        if drift:
            raise ValueError(f"training run manifest identity drift for {cell['cell_id']}: {drift}")

        data = scientific.get("data")
        initialization = scientific.get("initialization")
        training = scientific.get("training")
        if not all(isinstance(value, dict) for value in (data, initialization, training)):
            raise ValueError("training run manifest lacks data/init/training recipe objects")
        actual_nuisance = {
            "data_parent_inventory_sha256": data.get("parent_inventory_sha256"),
            "data_task_inventory_sha256": data.get("derived_task_inventory_sha256"),
            "initialization_inventory_sha256": initialization.get("inventory_sha256"),
            "initialization_checkpoint_s3": initialization.get("checkpoint_s3"),
            "seed": training.get("seed"),
            "steps": training.get("steps"),
            "action_horizon": training.get("action_horizon"),
            "window_len": training.get("window_len"),
            "chunk_stride": training.get("chunk_stride"),
        }
        if actual_nuisance != cell["training_nuisance"]:
            differences = {
                key: (actual_nuisance.get(key), expected)
                for key, expected in cell["training_nuisance"].items()
                if actual_nuisance.get(key) != expected
            }
            raise ValueError(f"training nuisance recipe drift for {cell['cell_id']}: {differences}")
        attempt_id = manifest.get("attempt_id")
        if (
            not isinstance(attempt_id, str)
            or not re.fullmatch(rf"{re.escape(cell['run_id'])}-attempt[1-9][0-9]*", attempt_id)
            or not uri.endswith(f"/{attempt_id}.json")
        ):
            raise ValueError(f"training attempt identity drift for {cell['cell_id']}")
        tree_root = _safe_s3(manifest.get("checkpoint_tree_manifest_root"))
        expected_root_tail = (
            f"/{cell['run_id']}/milestones"
            if is_milestone_cell(cell)
            else f"/{cell['run_id']}/step-{cell['final_step']}"
        )
        if not tree_root.endswith(expected_root_tail):
            raise ValueError(f"checkpoint tree root does not bind {cell['cell_id']}")
        return manifest

    def stage_checkpoint(self, cell: dict, destination: Path) -> str:
        manifest = self._verify_training_manifest(cell)
        uri = cell["training_completion_claim_s3"]
        claim = _json_bytes(self.store.read_bytes(uri), label=uri)
        if claim is None:
            raise ValueError(f"training completion claim is absent for {cell['cell_id']}")
        step = checkpoint_step(cell)
        checkpoint_uri = f"{cell['training_output_s3']}/deploy/{step}"
        tree_root = _safe_s3(manifest["checkpoint_tree_manifest_root"])
        if is_milestone_cell(cell):
            # The v4_70k completion claim enumerates every deployed milestone.  A cell may only
            # evaluate a step that claim lists, at the exact deploy/<step> address it records.
            deployed = list(cell["deployed_milestones"])
            expected = {
                "schema_version": 1,
                "kind": MILESTONE_COMPLETION_KIND,
                "run_id": cell["run_id"],
                "attempt_id": manifest["attempt_id"],
                "final_step": cell["final_step"],
                "steps": deployed,
                "run_manifest_sha256": cell["run_manifest_sha256"],
            }
            binding = verify_training_receipt_identity(
                claim,
                expected=expected,
                scientific_spec_sha256=cell["scientific_spec_sha256"],
                expected_binding=cell["training_completion_binding"],
                label=f"milestone training completion for {cell['cell_id']}",
            )
            records = claim.get("checkpoints")
            if (
                not isinstance(records, list)
                or [record.get("step") if isinstance(record, dict) else None for record in records] != deployed
            ):
                raise ValueError(
                    f"milestone completion claim does not enumerate the deployed steps for {cell['cell_id']}"
                )
            if step not in deployed:
                raise ValueError(f"{cell['cell_id']} evaluates step {step}, which the run never deployed: {deployed}")
            record = records[deployed.index(step)]
            if record.get("checkpoint_uri") != checkpoint_uri:
                raise ValueError(f"milestone record for step {step} does not address deploy/{step}")
            tree_sha = _require_sha(record.get("tree_manifest_sha256"), "milestone checkpoint tree digest")
            tree_uri = _safe_s3(record.get("tree_manifest_uri"))
            expected_tree_uri = f"{tree_root}/step-{step}/{tree_sha}.json"
            deploy_expected = {
                "schema_version": 1,
                "kind": "robomme_gpu_deploy_checkpoint_complete",
                "run_id": cell["run_id"],
                "attempt_id": manifest["attempt_id"],
                "step": step,
                "checkpoint_uri": checkpoint_uri,
                "tree_manifest_sha256": tree_sha,
                "run_manifest_sha256": cell["run_manifest_sha256"],
            }
        else:
            expected = {
                "schema_version": 1,
                "kind": "robomme_gpu_checkpoint_complete",
                "run_id": cell["run_id"],
                "attempt_id": manifest["attempt_id"],
                "step": cell["final_step"],
                "checkpoint_uri": checkpoint_uri,
                "run_manifest_sha256": cell["run_manifest_sha256"],
            }
            binding = verify_training_receipt_identity(
                claim,
                expected=expected,
                scientific_spec_sha256=cell["scientific_spec_sha256"],
                expected_binding=cell["training_completion_binding"],
                label=f"training completion for {cell['cell_id']}",
            )
            tree_sha = _require_sha(claim.get("tree_manifest_sha256"), "checkpoint tree digest")
            tree_uri = _safe_s3(claim.get("tree_manifest_uri"))
            expected_tree_uri = f"{tree_root}/{tree_sha}.json"
            deploy_expected = {
                **expected,
                "kind": "robomme_gpu_deploy_checkpoint_complete",
                "tree_manifest_sha256": tree_sha,
            }
        if tree_uri != expected_tree_uri:
            raise ValueError("checkpoint tree manifest is not the manifest-bound content address")
        tree_bytes = self.store.read_bytes(tree_uri)
        verify_checkpoint_tree_identity(
            tree_bytes,
            expected_sha256=tree_sha,
            checkpoint_uri=checkpoint_uri,
            label=f"checkpoint tree for {cell['cell_id']}",
        )
        deploy_uri = f"{checkpoint_uri}/_DEPLOY_COMPLETE.json"
        deploy = _json_bytes(self.store.read_bytes(deploy_uri), label=deploy_uri)
        if deploy is None:
            raise ValueError("checkpoint deploy receipt is absent")
        verify_training_receipt_identity(
            deploy,
            expected=deploy_expected,
            scientific_spec_sha256=cell["scientific_spec_sha256"],
            expected_binding=binding,
            label=f"checkpoint deploy receipt for {cell['cell_id']}",
        )
        self._sync(checkpoint_uri, destination, checkpoint=True)
        tree_path = destination.parent / "checkpoint-tree.json"
        tree_path.write_bytes(tree_bytes)
        if verify_checkpoint_manifest(destination, tree_path, expected_uri=checkpoint_uri) != tree_sha:
            raise ValueError("staged checkpoint failed exact tree verification")
        return checkpoint_uri

    def stage_workspace(self, cell: dict, destination: Path) -> Path | None:
        expected = cell.get("workspace")
        if expected is None:
            return None
        # CheckpointWorkspaceEncoder treats the leaf directory name as the checkpoint step.
        # Keep the S3 step payload under that numeric leaf instead of flattening it into the
        # campaign's generic ``workspace`` directory.
        destination = destination / str(expected["step"])
        provenance = expected["provenance_mode"]
        if provenance == WORKSPACE_PROVENANCE_CLAIM:
            claim_bytes = self.store.read_bytes(expected["claim_s3"])
            if claim_bytes is None or hashlib.sha256(claim_bytes).hexdigest() != expected["claim_sha256"]:
                raise ValueError("workspace producer claim is absent or has the wrong digest")
            claim = _json_bytes(claim_bytes, label=expected["claim_s3"])
            assert claim is not None
            exact = {
                "kind": "robomme_all16_workspace_task_complete",
                "task": cell["task"],
                "encoder_id": expected["encoder_id"],
            }
            if any(claim.get(key) != value for key, value in exact.items()):
                raise ValueError("workspace producer claim identity mismatch")
            representation = claim.get("representation", {})
            if representation != {
                "uri": expected["representation_s3"],
                "step": expected["step"],
                "completion_sha256": expected["completion_sha256"],
            }:
                raise ValueError("workspace representation identity mismatch")
            omega_payload = None
        elif provenance == WORKSPACE_PROVENANCE_LEGACY:
            omega_payload = self.store.read_bytes(expected["omega_manifest_s3"])
            if omega_payload is None:
                raise ValueError("legacy workspace omega manifest is absent")
        else:  # validate_queue rejects this before staging; retain a local fail-closed guard.
            raise ValueError("workspace provenance mode is unrecognized")
        self._sync(expected["representation_s3"], destination)
        seal_path = destination / "WSM_GENERATION_COMPLETE.json"
        if not seal_path.is_file() or _sha256(seal_path) != expected["completion_sha256"]:
            raise ValueError("staged workspace completion seal mismatch")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal.get("step") != expected["step"]:
            raise ValueError("staged workspace step mismatch")
        embedded = seal.get("embedded_sha256")
        for name in ("WSM_RUN_CONFIG.json", "WSM_BEST.json"):
            path = destination / name
            if not isinstance(embedded, dict) or not path.is_file() or _sha256(path) != embedded.get(name):
                raise ValueError(f"staged workspace embedded seal mismatch for {name}")
        if provenance == WORKSPACE_PROVENANCE_LEGACY:
            verify_legacy_workspace_metadata(
                expected,
                task=cell["task"],
                omega_payload=omega_payload,
                completion_payload=seal_path.read_bytes(),
                run_config_payload=(destination / "WSM_RUN_CONFIG.json").read_bytes(),
                best_payload=(destination / "WSM_BEST.json").read_bytes(),
            )
            actual_tree = _legacy_workspace_tree_sha256(destination, expected_step=expected["step"])
            if actual_tree != expected["checkpoint_tree_sha256"]:
                raise ValueError("staged legacy workspace checkpoint tree identity mismatch")
        return destination


def build_launch_command(
    queue: dict,
    cell: dict,
    *,
    source_root: Path,
    runtime: Runtime,
    checkpoint: Path,
    workspace: Path | None,
    output: Path,
) -> list[str]:
    topology = queue["topology"]
    if topology.get("execution_mode") == "parallel_fixed50_lanes_v1":
        raise ValueError("parallel fixed50 queues require lane-specific topology; use ParallelCampaignRunner")
    command = [
        str(runtime.policy_python),
        "-m",
        "robomme_integration.eval.launch_gpu_fleet",
        "--source-root",
        str(source_root),
        "--checkpoint",
        str(checkpoint),
        "--arm",
        cell["arm"],
        "--task",
        cell["task"],
        "--benchmark-config",
        str(source_root / cell["benchmark_config"]),
        "--vla-eval",
        str(runtime.vla_eval),
        "--output-root",
        str(output),
        "--eval-id",
        cell["eval_id"],
        "--gpus",
        ",".join(map(str, topology["policy_gpus"])),
        "--base-port",
        str(topology["base_port"]),
        "--shards",
        str(topology["simulator_shards"]),
        "--cpu-range",
        topology["cpu_range"],
        "--xla-memory-fraction",
        str(topology["xla_memory_fraction"]),
        "--native-simulator",
        "--simulator-gpus",
        ",".join(map(str, topology["simulator_gpus"])),
        "--pin-native-cpus",
    ]
    for path in (
        source_root,
        runtime.harness_src,
        runtime.robomme_src,
        runtime.maniskill_src,
        runtime.simulator_site,
    ):
        command.extend(["--simulator-pythonpath", str(path)])
    if workspace is not None:
        command.extend(
            [
                "--workspace-checkpoint",
                str(workspace),
                "--upstream-root",
                str(runtime.upstream_root),
                "--vision-encoder-home",
                str(runtime.vision_encoder_home),
                "--cfg-guidance-scale",
                str(float(cell.get("cfg_guidance_scale", 1.0))),
            ]
        )
    if cell["arm"] == "ptrm":
        command.extend(
            [
                "--ptrm-eval-k",
                str(cell["ptrm"]["eval_k"]),
                "--ptrm-eval-sigma",
                str(cell["ptrm"]["eval_sigma"]),
                "--ptrm-eval-select",
                cell["ptrm"]["eval_select"],
            ]
        )
    prewarm = topology.get("native_shard_prewarm_seconds")
    stagger = topology.get("native_shard_stagger_seconds")
    if prewarm is not None:
        command.extend(["--native-shard-prewarm-seconds", str(float(prewarm))])
    if stagger is not None:
        command.extend(["--native-shard-stagger-seconds", str(float(stagger))])
    if topology.get("execution_mode") == "parallel_fixed50_lane_v1":
        # Pin the supervisor before it creates the policy server and native launcher.  Every
        # descendant therefore stays inside the lane's sealed CPU range; launch_sharded further
        # partitions that same range among the four simulator workers.
        command = ["taskset", "--cpu-list", topology["cpu_range"], *command]
    return command


class Evaluator(Protocol):
    def run(
        self,
        queue: dict,
        cell: dict,
        *,
        checkpoint: Path,
        workspace: Path | None,
        output: Path,
    ) -> int: ...


class EvaluatorCancelled(BaseException):
    """Cooperative parallel-lane cancellation which must never mint a failure claim."""


class EvaluatorCleanupFailure(BaseException):
    """Cancellation could not prove that the evaluator process tree was drained."""


class EvaluatorDeadlineExceeded(BaseException):
    """The node budget expired; partial work is restartable and must never become a claim."""


@dataclass
class SubprocessEvaluator:
    source_root: Path
    runtime: Runtime
    topology_override: dict | None = None
    cancel_event: threading.Event | None = None
    cancel_poll_seconds: float = 0.5
    deadline_monotonic: float | None = None

    def run(
        self,
        queue: dict,
        cell: dict,
        *,
        checkpoint: Path,
        workspace: Path | None,
        output: Path,
    ) -> int:
        launch_queue = queue if self.topology_override is None else {**queue, "topology": dict(self.topology_override)}
        command = build_launch_command(
            launch_queue,
            cell,
            source_root=self.source_root,
            runtime=self.runtime,
            checkpoint=checkpoint,
            workspace=workspace,
            output=output,
        )
        environment = os.environ.copy()
        environment.update(self.runtime.render_environment)
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path)
            for path in policy_pythonpath_entries(
                source_root=self.source_root,
                harness_src=self.runtime.harness_src,
                openpi_src=self.runtime.openpi_src,
                policy_site=self.runtime.policy_site,
                upstream_root=self.runtime.upstream_root,
                # Sealed runtime-only fallback for pure harness dependencies (for example anyio).
                # Policy OpenPI/JAX/NumPy/Torch remain ahead of it and are audited by preflight.
                simulator_site=self.runtime.simulator_site,
            )
        )
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise EvaluatorCancelled("parallel campaign cancelled before evaluator launch")
        if self.deadline_monotonic is not None and self.deadline_monotonic <= time.monotonic():
            raise EvaluatorDeadlineExceeded("parallel campaign node deadline expired before evaluator launch")
        print("[eval-campaign] exec " + " ".join(command), flush=True)
        process = subprocess.Popen(
            command,
            cwd=self.source_root,
            env=environment,
            start_new_session=True,
        )
        try:
            if self.cancel_event is None:
                return int(process.wait())
            while True:
                if self.cancel_event.is_set():
                    raise EvaluatorCancelled("parallel campaign cancelled this evaluator lane")
                remaining = None if self.deadline_monotonic is None else self.deadline_monotonic - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise EvaluatorDeadlineExceeded("parallel campaign node deadline expired during evaluation")
                poll_seconds = self.cancel_poll_seconds
                if remaining is not None:
                    poll_seconds = min(poll_seconds, remaining)
                try:
                    return int(process.wait(timeout=poll_seconds))
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as error:
            try:
                _terminate_evaluator_process(process)
            except Exception as cleanup_error:
                # This remains a BaseException so the cell transaction cannot mint a permanent
                # scientific claim.  It must not be swallowed as ordinary cooperative
                # cancellation: the supervisor has lost proof that its GPU resources are clean.
                failure = EvaluatorCleanupFailure(
                    f"evaluator process-group cleanup failed after {type(error).__name__}: {error}"
                )
                failure.add_note(f"cleanup error: {type(cleanup_error).__name__}: {cleanup_error}")
                raise failure from cleanup_error
            raise


def _descendant_process_groups(root_pid: int) -> set[int]:
    """Snapshot child process groups before an interrupted fleet can orphan GPU workers."""
    pending = [root_pid]
    seen: set[int] = set()
    groups: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            pending.extend(int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
            groups.add(os.getpgid(pid))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    groups.discard(os.getpgrp())
    return groups


def _terminate_evaluator_process(
    process: subprocess.Popen,
    *,
    grace_seconds: int = 60,
) -> None:
    """Ask the fleet supervisor to drain, then kill snapshotted child groups if it cannot."""
    groups = _descendant_process_groups(process.pid)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        groups.update(_descendant_process_groups(process.pid))
    # Even a clean supervisor exit is not proof that every descendant exited.  Sweep the exact
    # process groups snapshotted before SIGTERM so an orphaned server/shard cannot retain a GPU or
    # be mistaken for the next lane's process.
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("interrupted evaluator process group did not terminate") from error


def _bounded_failure_text(path: Path, *, limit: int = 4 * 1024 * 1024) -> str:
    """Read bounded head/tail evidence so late failures in verbose logs are never skipped."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size <= limit:
            payload = stream.read(limit + 1)
        else:
            head_bytes = min(64 * 1024, limit // 4)
            head = stream.read(head_bytes)
            stream.seek(max(head_bytes, size - (limit - head_bytes)))
            payload = head + b"\n...[bounded middle omitted]...\n" + stream.read(limit - head_bytes)
    return payload.decode("utf-8", errors="replace")


def _failure_haystack(output: Path, detail: str) -> str:
    fragments = [detail]
    if output.exists():
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".json", ".jsonl", ".log", ".txt"} and path.name not in {
                "FAILED",
                "COMPLETED",
            }:
                continue
            fragments.append(_bounded_failure_text(path))
    return "\n".join(fragments).casefold()


def classify_transient(output: Path, detail: str) -> str | None:
    """Return an allowlisted transient class; absence means terminal/unknown."""
    haystack = _failure_haystack(output, detail)
    for failure_class, markers in TRANSIENT_MARKERS:
        if any(marker in haystack for marker in markers):
            return failure_class
    return None


def classify_systemic(output: Path, detail: str) -> str | None:
    """Return a fail-closed campaign-wide resource class, or ``None`` for a lane-local failure."""
    haystack = _failure_haystack(output, detail)
    for failure_class, markers in SYSTEMIC_MARKERS:
        if any(marker in haystack for marker in markers):
            return failure_class
    return None


class Artifacts(Protocol):
    def publish_success(
        self,
        queue: dict,
        cell: dict,
        *,
        output: Path,
        cell_work: Path,
        checkpoint_uri: str,
        store: ObjectStore,
        runtime: Runtime,
    ) -> dict: ...

    def publish_failure(
        self,
        queue: dict,
        cell: dict,
        *,
        cell_work: Path,
        failure: dict,
        store: ObjectStore,
    ) -> dict: ...


class Fixed50Artifacts:
    def publish_success(
        self,
        queue: dict,
        cell: dict,
        *,
        output: Path,
        cell_work: Path,
        checkpoint_uri: str,
        store: ObjectStore,
        runtime: Runtime,
    ) -> dict:
        legacy_job = {"run_id": cell["run_id"], "task": cell["task"], "arm": cell["arm"]}
        try:
            result_path = fixed50._result_claim(output, legacy_job, cell["eval_id"], checkpoint_uri)
        except Exception as error:
            raise EvaluationFailure(f"fixed-50 audit failed: {type(error).__name__}: {error}", output) from error
        launch = json.loads((output / "eval/launch_manifest.json").read_text(encoding="utf-8"))
        if launch.get("config_sha256") != cell["benchmark_config_sha256"]:
            raise EvaluationFailure("fixed-50 launch used the wrong benchmark config", output)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update(
            queue_id=queue["queue_id"],
            queue_manifest_sha256=queue["queue_manifest_sha256"],
            cell_id=cell["cell_id"],
            scientific_spec_sha256=cell["scientific_spec_sha256"],
            run_manifest_sha256=cell["run_manifest_sha256"],
            benchmark_config_sha256=cell["benchmark_config_sha256"],
            native_preflight_claim_sha256=runtime.preflight_claim_sha256,
            runtime_receipt_sha256=runtime.receipt_sha256,
        )
        if is_milestone_cell(cell):
            result.update(
                training_scope="multitask_v4",
                checkpoint_step=checkpoint_step(cell),
                final_step=cell["final_step"],
                deployed_milestones=list(cell["deployed_milestones"]),
                eval_protocol=queue["comparability"]["eval_protocol"],
            )
        _atomic_json(result_path, result)
        archive, archive_sha = fixed50._archive_evidence(cell_work, output, result_path, cell["eval_id"])
        evidence_uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/evidence/{archive_sha}.tgz"
        store.put_file_once(archive, evidence_uri)
        result.update(evidence_archive_sha256=archive_sha, evidence_archive_uri=evidence_uri)
        _atomic_json(result_path, result)
        store.put_file_once(result_path, cell["result_claim_s3"])
        return result

    def publish_failure(
        self,
        queue: dict,
        cell: dict,
        *,
        cell_work: Path,
        failure: dict,
        store: ObjectStore,
    ) -> dict:
        failure_path = cell_work / "terminal-failure.json"
        payload = {
            "schema_version": 1,
            "kind": CELL_FAILURE_KIND,
            "queue_id": queue["queue_id"],
            "queue_manifest_sha256": queue["queue_manifest_sha256"],
            "cell_id": cell["cell_id"],
            "run_id": cell["run_id"],
            "task": cell["task"],
            "arm": cell["arm"],
            **failure,
        }
        _atomic_json(failure_path, payload)
        evidence_root = cell_work / "terminal-evidence"
        evidence_root.mkdir()
        shutil.copy2(failure_path, evidence_root / failure_path.name)
        large_files: list[dict] = []
        for attempt in sorted((cell_work / "attempts").glob("attempt-*")):
            for path in sorted(attempt.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix not in {".json", ".jsonl", ".log", ".txt"} and path.name != "FAILED":
                    continue
                relative = path.relative_to(attempt)
                size = path.stat().st_size
                if size > 4 * 1024 * 1024:
                    head_bytes = 64 * 1024
                    tail_bytes = 4 * 1024 * 1024 - head_bytes
                    with path.open("rb") as source:
                        head = source.read(head_bytes)
                        source.seek(max(len(head), size - tail_bytes))
                        tail = source.read(tail_bytes)
                    record = {
                        "attempt": attempt.name,
                        "path": relative.as_posix(),
                        "bytes": size,
                        "sha256": _sha256(path),
                        "captured_head_bytes": len(head),
                        "captured_tail_bytes": len(tail),
                    }
                    target = (
                        evidence_root
                        / attempt.name
                        / "bounded-large-files"
                        / relative.parent
                        / f"{relative.name}.head-tail.txt"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(
                        _canonical(record)
                        + b"--- original head ---\n"
                        + head
                        + b"\n--- omitted middle; original tail follows ---\n"
                        + tail
                    )
                    large_files.append(record)
                    continue
                target = evidence_root / attempt.name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        if large_files:
            _atomic_json(
                evidence_root / "bounded-large-file-manifest.json",
                {"schema_version": 1, "files": large_files},
            )
        archive = cell_work / "terminal-evidence.tgz"
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as stream:
                    for path in sorted(evidence_root.rglob("*")):
                        if path.is_file():
                            info = stream.gettarinfo(str(path), arcname=path.relative_to(cell_work).as_posix())
                            info.mtime = 0
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            with path.open("rb") as source:
                                stream.addfile(info, source)
        archive_sha = _sha256(archive)
        archive_uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/failures/{archive_sha}.tgz"
        store.put_file_once(archive, archive_uri)
        payload.update(evidence_archive_sha256=archive_sha, evidence_archive_uri=archive_uri)
        _atomic_json(failure_path, payload)
        failure_uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/failure.complete.json"
        store.put_file_once(failure_path, failure_uri)
        return {**payload, "failure_claim_s3": failure_uri}


class EvaluationFailure(RuntimeError):
    def __init__(self, detail: str, output: Path) -> None:
        super().__init__(detail)
        self.output = output


def _verify_existing_result(queue: dict, cell: dict, store: ObjectStore, runtime: Runtime) -> dict | None:
    uri = cell["result_claim_s3"]
    result = _json_bytes(store.read_bytes(uri), label=uri)
    if result is None:
        return None
    expected = {
        "kind": "robomme_fixed50_complete",
        "queue_id": queue["queue_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "cell_id": cell["cell_id"],
        "run_id": cell["run_id"],
        "task": cell["task"],
        "arm": cell["arm"],
        "eval_id": cell["eval_id"],
        "episodes": 50,
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
        "run_manifest_sha256": cell["run_manifest_sha256"],
        "benchmark_config_sha256": cell["benchmark_config_sha256"],
        "native_preflight_claim_sha256": runtime.preflight_claim_sha256,
        "runtime_receipt_sha256": runtime.receipt_sha256,
    }
    drift = {key: (result.get(key), value) for key, value in expected.items() if result.get(key) != value}
    if drift:
        raise ValueError(f"existing fixed-50 result identity drift for {cell['cell_id']}: {drift}")
    successes = result.get("successes")
    if not isinstance(successes, int) or isinstance(successes, bool) or not 0 <= successes <= 50:
        raise ValueError("existing fixed-50 result has invalid success count")
    archive_sha = _require_sha(result.get("evidence_archive_sha256"), "evidence archive digest")
    archive_uri = _safe_s3(result.get("evidence_archive_uri"))
    if not archive_uri.endswith(f"/{archive_sha}.tgz") or not store.exists(archive_uri):
        raise ValueError("existing fixed-50 result evidence is absent or not content addressed")
    return result


def _verify_existing_failure(queue: dict, cell: dict, store: ObjectStore) -> dict | None:
    uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/failure.complete.json"
    failure = _json_bytes(store.read_bytes(uri), label=uri)
    if failure is None:
        return None
    expected = {
        "kind": CELL_FAILURE_KIND,
        "queue_id": queue["queue_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "cell_id": cell["cell_id"],
        "run_id": cell["run_id"],
        "task": cell["task"],
        "arm": cell["arm"],
        "status": "terminal_failure",
    }
    drift = {key: (failure.get(key), value) for key, value in expected.items() if failure.get(key) != value}
    if drift:
        raise ValueError(f"existing terminal failure identity drift for {cell['cell_id']}: {drift}")
    archive_sha = _require_sha(failure.get("evidence_archive_sha256"), "failure evidence digest")
    archive_uri = _safe_s3(failure.get("evidence_archive_uri"))
    if not archive_uri.endswith(f"/{archive_sha}.tgz") or not store.exists(archive_uri):
        raise ValueError("existing terminal-failure evidence is absent or not content addressed")
    return {**failure, "failure_claim_s3": uri}


def _verify_queue_completion(queue: dict, completion: dict, store: ObjectStore, runtime: Runtime) -> str:
    expected = {
        "kind": QUEUE_COMPLETION_KIND,
        "queue_id": queue["queue_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "native_preflight_claim_sha256": runtime.preflight_claim_sha256,
        "runtime_receipt_sha256": runtime.receipt_sha256,
    }
    drift = {key: (completion.get(key), value) for key, value in expected.items() if completion.get(key) != value}
    if drift:
        raise ValueError(f"existing queue completion receipt identity drift: {drift}")
    status = completion.get("status")
    if status not in {"complete", "complete_with_terminal_failures"}:
        raise ValueError("existing queue completion receipt has an invalid status")
    records = completion.get("records")
    if not isinstance(records, list) or [record.get("cell_id") for record in records] != [
        cell["cell_id"] for cell in queue["cells"]
    ]:
        raise ValueError("existing queue completion receipt does not cover cells in exact order")
    all_complete = True
    for cell, record in zip(queue["cells"], records, strict=True):
        record_status = record.get("status")
        if record_status in {"complete", "skipped_exact_complete"}:
            if _verify_existing_result(queue, cell, store, runtime) is None:
                raise ValueError(f"queue completion names absent result for {cell['cell_id']}")
            continue
        all_complete = False
        if record_status not in {"terminal_failure", "skipped_terminal_failure"}:
            raise ValueError(f"queue completion has invalid cell status for {cell['cell_id']}")
        failure_uri = _safe_s3(record.get("failure_claim_s3"))
        evidence_uri = _safe_s3(record.get("evidence_archive_uri"))
        if not failure_uri.startswith(f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/"):
            raise ValueError(f"queue completion failure claim is not cell-specific for {cell['cell_id']}")
        if not evidence_uri.startswith(f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/"):
            raise ValueError(f"queue completion failure evidence is not cell-specific for {cell['cell_id']}")
        if not store.exists(failure_uri) or not store.exists(evidence_uri):
            raise ValueError(f"queue completion failure evidence is absent for {cell['cell_id']}")
    expected_status = "complete" if all_complete else "complete_with_terminal_failures"
    if status != expected_status:
        raise ValueError("queue completion aggregate status disagrees with its cell records")
    return status


def _safe_rmtree(path: Path, root: Path) -> None:
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root) or not path.name.startswith("cell-"):
        raise ValueError(f"refusing unsafe eval-campaign cleanup: {path}")
    if path.exists():
        shutil.rmtree(path)


@dataclass
class CampaignRunner:
    queue: dict
    source_root: Path
    work_root: Path
    runtime: Runtime
    store: ObjectStore
    stager: Stager
    evaluator: Evaluator
    artifacts: Artifacts

    def __post_init__(self) -> None:
        self.source_root = self.source_root.resolve()
        self.work_root = self.work_root.resolve()
        validate_queue(self.queue, source_root=self.source_root)
        if self.work_root == Path("/") or not self.work_root.is_absolute():
            raise ValueError("eval campaign work root must be an absolute non-root path")
        (self.work_root / "cells").mkdir(parents=True, exist_ok=True)
        (self.work_root / "state" / "cells").mkdir(parents=True, exist_ok=True)

    @property
    def campaign_state_path(self) -> Path:
        return self.work_root / "state" / "campaign.json"

    def _write_campaign_state(self, records: list[dict], *, status: str) -> None:
        _atomic_json(
            self.campaign_state_path,
            {
                "schema_version": 1,
                "queue_id": self.queue["queue_id"],
                "queue_manifest_sha256": self.queue["queue_manifest_sha256"],
                "status": status,
                "records": records,
                "updated_utc": _utc(),
            },
        )

    def _write_cell_state(self, cell: dict, value: dict) -> None:
        _atomic_json(
            self.work_root / "state" / "cells" / f"{cell['cell_id']}.json",
            {
                "schema_version": 1,
                "queue_id": self.queue["queue_id"],
                "queue_manifest_sha256": self.queue["queue_manifest_sha256"],
                "cell_id": cell["cell_id"],
                "run_id": cell["run_id"],
                **value,
                "updated_utc": _utc(),
            },
        )

    def _halt_after_terminal(
        self,
        records: list[dict],
        cell: dict,
        record: dict,
    ) -> int:
        """Persist one terminal cell and stop before staging any later cell."""
        records.append(record)
        self._write_cell_state(cell, record)
        self._write_campaign_state(records, status="halted_terminal_failure")
        return 2

    def run_cell_transaction(
        self,
        cell: dict,
        *,
        evaluator: Evaluator | None = None,
        systemic_callback: Callable[[str], bool] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        """Stage, evaluate, audit, publish, and clean one independently claimable cell.

        The sequential and parallel supervisors deliberately share this transaction.  In
        particular, there is still exactly one fixed-50 audit and one content-addressed result or
        failure claim per original queue cell; parallel scheduling never merges observations or
        constructs a derived scientific result.
        """
        evaluator = evaluator or self.evaluator
        cell_work = self.work_root / "cells" / f"cell-{cell['ordinal']:03d}-{cell['cell_id']}"
        _safe_rmtree(cell_work, self.work_root / "cells")
        cell_work.mkdir(parents=True)
        attempts: list[dict] = []
        record: dict
        originated_systemic_failure = False
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise EvaluatorCancelled("parallel campaign cancelled before cell staging")
            self._write_cell_state(cell, {"status": "staging", "attempts": attempts})
            checkpoint = cell_work / "checkpoint" / str(checkpoint_step(cell))
            checkpoint_uri = self.stager.stage_checkpoint(cell, checkpoint)
            if cancel_event is not None and cancel_event.is_set():
                raise EvaluatorCancelled("parallel campaign cancelled during checkpoint staging")
            workspace = self.stager.stage_workspace(cell, cell_work / "workspace")
            if cancel_event is not None and cancel_event.is_set():
                raise EvaluatorCancelled("parallel campaign cancelled during workspace staging")
            max_attempts = self.queue["retry"]["max_attempts"]
            result = None
            terminal_detail = "evaluation did not start"
            terminal_class = "unknown"
            for attempt in range(1, max_attempts + 1):
                output = cell_work / "attempts" / f"attempt-{attempt}" / "output"
                output.parent.mkdir(parents=True)
                self._write_cell_state(
                    cell,
                    {"status": "evaluating", "attempt": attempt, "attempts": attempts},
                )
                try:
                    if cancel_event is not None and cancel_event.is_set():
                        raise EvaluatorCancelled("parallel campaign cancelled before evaluator launch")
                    returncode = evaluator.run(
                        self.queue,
                        cell,
                        checkpoint=checkpoint,
                        workspace=workspace,
                        output=output,
                    )
                    if returncode != 0:
                        raise EvaluationFailure(f"evaluation fleet returned {returncode}", output)
                    if cancel_event is not None and cancel_event.is_set():
                        raise EvaluatorCancelled("parallel campaign cancelled before success publication")
                    result = self.artifacts.publish_success(
                        self.queue,
                        cell,
                        output=output,
                        cell_work=cell_work,
                        checkpoint_uri=checkpoint_uri,
                        store=self.store,
                        runtime=self.runtime,
                    )
                    attempts.append({"attempt": attempt, "status": "complete"})
                    break
                except EvaluationFailure as error:
                    systemic_class = classify_systemic(error.output, str(error))
                    if cancel_event is not None and cancel_event.is_set() and systemic_class is None:
                        raise EvaluatorCancelled("parallel campaign cancelled while evaluator was failing") from error
                    # A policy-server OOM commonly tears down the WebSocket too.  The secondary
                    # transport-reset marker must never downgrade a sealed-resource failure into
                    # a retryable harness transient.
                    retry_class = None if systemic_class is not None else classify_transient(error.output, str(error))
                    failure_class = systemic_class or retry_class or "unclassified"
                    if systemic_class is not None:
                        originated_systemic_failure = True
                        if systemic_callback is not None and not systemic_callback(systemic_class):
                            raise EvaluatorCancelled("another lane owns the systemic failure claim") from error
                    attempts.append(
                        {
                            "attempt": attempt,
                            "status": "transient_failure" if retry_class else "terminal_failure",
                            "failure_class": failure_class,
                            "detail": str(error)[:1000],
                        }
                    )
                    terminal_detail = str(error)[:1000]
                    terminal_class = failure_class
                    self._write_cell_state(
                        cell,
                        {"status": attempts[-1]["status"], "attempts": attempts},
                    )
                    if retry_class is None or attempt == max_attempts:
                        break
                    print(
                        f"[eval-campaign] RETRY {cell['cell_id']} class={retry_class} "
                        f"attempt={attempt + 1}/{max_attempts}",
                        flush=True,
                    )
            if result is not None:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "complete",
                    "attempts": attempts,
                    "successes": result["successes"],
                    "result_claim_s3": cell["result_claim_s3"],
                    "evidence_archive_uri": result["evidence_archive_uri"],
                }
            else:
                if cancel_event is not None and cancel_event.is_set() and not originated_systemic_failure:
                    raise EvaluatorCancelled("parallel campaign cancelled before terminal failure publication")
                failure = self.artifacts.publish_failure(
                    self.queue,
                    cell,
                    cell_work=cell_work,
                    failure={
                        "status": "terminal_failure",
                        "attempts": attempts,
                        "failure_class": terminal_class,
                        "detail": terminal_detail,
                    },
                    store=self.store,
                )
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "terminal_failure",
                    "attempts": attempts,
                    "failure_class": terminal_class,
                    "failure_claim_s3": failure["failure_claim_s3"],
                    "evidence_archive_uri": failure["evidence_archive_uri"],
                }
        except Exception as error:
            # Staging, identity, and publication errors are never silently classified transient.
            # Process-control exceptions (KeyboardInterrupt/SystemExit) deliberately bypass this
            # block: they are not scientific cell failures and must never mint an immutable
            # terminal claim that poisons an otherwise restartable queue identity.
            if cancel_event is not None and cancel_event.is_set():
                raise EvaluatorCancelled("parallel campaign cancelled during cell staging or publication") from error
            if systemic_callback is not None and not systemic_callback("control_plane_or_identity"):
                raise EvaluatorCancelled("another lane owns the control-plane failure claim") from error
            failure = self.artifacts.publish_failure(
                self.queue,
                cell,
                cell_work=cell_work,
                failure={
                    "status": "terminal_failure",
                    "attempts": attempts,
                    "failure_class": "control_plane_or_identity",
                    "detail": f"{type(error).__name__}: {error}"[:1000],
                },
                store=self.store,
            )
            record = {
                "cell_id": cell["cell_id"],
                "status": "terminal_failure",
                "attempts": attempts,
                "failure_class": "control_plane_or_identity",
                "failure_claim_s3": failure["failure_claim_s3"],
                "evidence_archive_uri": failure["evidence_archive_uri"],
            }
        finally:
            # Checkpoint, workspace, simulator output, and packaged evidence are all ephemeral;
            # their exact durable receipts were published before a successful record was made.
            _safe_rmtree(cell_work, self.work_root / "cells")
        self._write_cell_state(cell, record)
        return record

    def run(self) -> int:
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.queue["limits"]["max_run_seconds"]
        self.store.put_bytes_once(_canonical(self.queue), self.queue["claims"]["manifest"])
        existing_completion = _json_bytes(
            self.store.read_bytes(self.queue["claims"]["completion"]),
            label=self.queue["claims"]["completion"],
        )
        if existing_completion is not None:
            status = _verify_queue_completion(self.queue, existing_completion, self.store, self.runtime)
            return 0 if status == "complete" else 2

        records: list[dict] = []
        self._write_campaign_state(records, status="running")
        for cell in self.queue["cells"]:
            remaining = deadline - time.monotonic()
            required = self.queue["limits"]["estimated_cell_seconds"] + self.queue["limits"]["runtime_reserve_seconds"]
            if remaining < required:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "deferred_runtime_budget",
                    "remaining_seconds": max(0, int(remaining)),
                    "required_seconds": required,
                }
                records.append(record)
                self._write_cell_state(cell, record)
                self._write_campaign_state(records, status="deferred_runtime_budget")
                return 0
            minimum_free = self.queue["limits"]["minimum_free_bytes"]
            if shutil.disk_usage(self.work_root).free < minimum_free:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "blocked_disk_floor",
                    "minimum_free_bytes": minimum_free,
                }
                records.append(record)
                self._write_cell_state(cell, record)
                self._write_campaign_state(records, status="blocked_disk_floor")
                return 3
            existing = _verify_existing_result(self.queue, cell, self.store, self.runtime)
            if existing is not None:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "skipped_exact_complete",
                    "successes": existing["successes"],
                    "result_claim_s3": cell["result_claim_s3"],
                    "evidence_archive_uri": existing["evidence_archive_uri"],
                }
                records.append(record)
                self._write_cell_state(cell, record)
                self._write_campaign_state(records, status="running")
                continue
            existing_failure = _verify_existing_failure(self.queue, cell, self.store)
            if existing_failure is not None:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "skipped_terminal_failure",
                    "failure_class": existing_failure.get("failure_class", "unclassified"),
                    "failure_claim_s3": existing_failure["failure_claim_s3"],
                    "evidence_archive_uri": existing_failure["evidence_archive_uri"],
                }
                return self._halt_after_terminal(records, cell, record)

            record = self.run_cell_transaction(cell)
            if record["status"] == "terminal_failure":
                return self._halt_after_terminal(records, cell, record)
            records.append(record)
            self._write_campaign_state(records, status="running")

        all_complete = all(record["status"] in {"complete", "skipped_exact_complete"} for record in records)
        completion = {
            "schema_version": 1,
            "kind": QUEUE_COMPLETION_KIND,
            "queue_id": self.queue["queue_id"],
            "queue_manifest_sha256": self.queue["queue_manifest_sha256"],
            "native_preflight_claim_sha256": self.runtime.preflight_claim_sha256,
            "runtime_receipt_sha256": self.runtime.receipt_sha256,
            "status": "complete" if all_complete else "complete_with_terminal_failures",
            "records": records,
        }
        self.store.put_bytes_once(_canonical(completion), self.queue["claims"]["completion"])
        self._write_campaign_state(records, status=completion["status"])
        return 0 if all_complete else 2


def _dry_run_payload(queue: dict, source_root: Path, runtime: Runtime, work_root: Path) -> dict:
    cells = []
    for cell in queue["cells"]:
        cell_root = work_root / "cells" / f"cell-{cell['ordinal']:03d}-{cell['cell_id']}"
        workspace = (
            cell_root / "workspace" / str(cell["workspace"]["step"]) if cell["arm"] in WORKSPACE_EVAL_ARMS else None
        )
        command = build_launch_command(
            queue,
            cell,
            source_root=source_root,
            runtime=runtime,
            checkpoint=cell_root / "checkpoint" / str(checkpoint_step(cell)),
            workspace=workspace,
            output=cell_root / "attempts/attempt-1/output",
        )
        cells.append(
            {
                "cell_id": cell["cell_id"],
                "task": cell["task"],
                "arm": cell["arm"],
                "max_attempts": queue["retry"]["max_attempts"],
                "launch_command": command,
            }
        )
    return {
        "schema_version": 1,
        "dry_run": True,
        "queue_id": queue["queue_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "native_preflight_claim_sha256": runtime.preflight_claim_sha256,
        "runtime_receipt_sha256": runtime.receipt_sha256,
        "cells": cells,
        "note": "no cloud submission, S3 access, simulator, or policy process was started",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--native-preflight-claim", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/opt/ml/robomme-eval-campaign"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="run locally and publish evidence; never submits a cloud job",
    )
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    source_root = args.source_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    validate_queue(queue, source_root=source_root)
    runtime = verify_gates(
        queue,
        preflight_claim=args.native_preflight_claim.expanduser().resolve(),
        runtime_receipt=args.runtime_receipt.expanduser().resolve(),
    )
    parallel_mode = queue["topology"].get("execution_mode") == "parallel_fixed50_lanes_v1"
    if args.dry_run:
        if parallel_mode:
            from robomme_integration.eval import parallel_campaign

            payload = parallel_campaign.dry_run_payload(queue, source_root, runtime, work_root)
        else:
            payload = _dry_run_payload(queue, source_root, runtime, work_root)
        print(json.dumps(payload, indent=2))
        return 0
    if not args.confirm_run:
        raise SystemExit("execution blocked: obtain explicit approval, then pass --confirm-run")
    store = AwsCliStore()
    if parallel_mode:
        from robomme_integration.eval import parallel_campaign

        return parallel_campaign.ParallelCampaignRunner(
            queue=queue,
            source_root=source_root,
            work_root=work_root,
            runtime=runtime,
            store=store,
            stager=AwsStager(store),
            artifacts=Fixed50Artifacts(),
        ).run()
    return CampaignRunner(
        queue=queue,
        source_root=source_root,
        work_root=work_root,
        runtime=runtime,
        store=store,
        stager=AwsStager(store),
        evaluator=SubprocessEvaluator(source_root, runtime),
        artifacts=Fixed50Artifacts(),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
