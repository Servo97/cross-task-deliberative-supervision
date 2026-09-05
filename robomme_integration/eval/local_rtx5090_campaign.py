#!/usr/bin/env python3
"""Approval-gated local RTX 5090 adapter for the sealed RoboMME eval campaign.

This module deliberately reuses :mod:`robomme_integration.eval.campaign` for checkpoint
authentication, fixed-50 auditing, narrowly classified retries, per-cell continuation, evidence
publication, and resume.  It only replaces the p5-specific runtime gate and topology with a
content-sealed local two-RTX-5090 gate.  With no ``--confirm-run`` it performs no S3 operation and
starts no policy or simulator process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
from pathlib import Path

from robomme_integration.eval import (
    build_existing_pick_button_queue as pick_button,
)
from robomme_integration.eval import campaign, parallel_campaign
from robomme_integration.eval.launch_p5_campaign import (
    UPSTREAM_COMMIT,
    UPSTREAM_CRITICAL_SHA256,
    VISION_BYTES,
    VISION_REVISION,
    VISION_S3,
    VISION_SHA256,
)
from robomme_integration.eval.launch_p5_preflight import RUNTIME_S3, RUNTIME_SHA
from robomme_integration.launch import OPENPI, OPENPI_SHA
from wsm_settings import ROBOMME_EVAL_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_ROOT = ROBOMME_EVAL_ROOT
LOCAL_PREFLIGHT_KIND = "robomme_local_rtx5090_native_eval_preflight"
LOCAL_RECEIPT_KIND = "robomme_local_rtx5090_eval_runtime_receipt"
LOCAL_ACCELERATOR = "2xNVIDIA GeForce RTX 5090"
LOCAL_RETRY_QUEUE_ID = "pick-button-representation-fixed50-local5090-v2"
LOCAL_PARALLEL_QUEUE_ID = "pick-button-representation-fixed50-local5090-v3r"
LOCAL_WORKSPACE_ACTION_CONFIG_SHA256 = "a244b62e1e7e66714b4305cf04f2b3f50449500cc00d0379426af82e5dc91c9f"
LOCAL_PARALLEL_WORKSPACE_ACTION_CONFIG_SHA256 = "a0c2e32f9d84c15fa1c57449f7523cadce851b7cc6d2174876958c0105b36039"
POLICY_IMPORT_VERSIONS = {
    "jax": "0.10.1",
    "numpy": "2.2.5",
    "torch": "2.11.0+cu128",
    "anyio": "4.14.2",
    "vla_eval": "0.4.0",
}
SIMULATOR_IMPORT_VERSIONS = {
    "numpy": "1.26.4",
    "torch": "2.9.1+cu128",
    "mplib": "0.1.1",
    "sapien": "3.0.3",
    "vla_eval": "0.4.0",
}
LOCAL_RENDER_ENVIRONMENT = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "ROBOMME_USE_LAVAPIPE": "0",
    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
    "PYTHONDONTWRITEBYTECODE": "1",
}
LOCAL_TOPOLOGY = {
    "policy_gpus": [0, 1],
    "simulator_gpus": [0, 1],
    "simulator_shards": 8,
    "cpu_range": "0-127",
    "base_port": 18100,
    "xla_memory_fraction": 0.65,
}
CONTRACT_KEYS = {
    "runtime",
    "openpi",
    "base_environment",
    "upstream",
    "vision",
    "paths",
    "critical_file_sha256",
    "render_environment",
    "import_contract",
    "executables",
}
PATH_KEYS = {
    "runtime_archive",
    "openpi_archive",
    "base_openpi_archive",
    "policy_python",
    "simulator_python",
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
EXECUTABLE_PATH_KEYS = frozenset({"policy_python", "simulator_python", "vla_eval"})
SOURCE_EXCLUDED_NAMES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "wandb"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_path(name: str, raw: str | os.PathLike[str]) -> Path:
    """Normalize contract paths without collapsing virtual-environment executables."""
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"local runtime path {name} must be absolute")
    if ".." in path.parts:
        raise ValueError(f"local runtime path {name} contains parent traversal")
    if name in EXECUTABLE_PATH_KEYS:
        return Path(os.path.abspath(path))
    return path.resolve()


def _executable_identity(path: Path) -> dict[str, object]:
    lexical = Path(os.path.abspath(path))
    if not lexical.is_file() or not os.access(lexical, os.X_OK):
        raise ValueError(f"sealed executable is absent or not executable: {lexical}")
    resolved = lexical.resolve(strict=True)
    return {
        "path": str(lexical),
        "symlink_target": os.readlink(lexical) if lexical.is_symlink() else None,
        "resolved_path": str(resolved),
        "resolved_sha256": _sha256(resolved),
        "mode": oct(stat.S_IMODE(lexical.lstat().st_mode)),
    }


def _python_identity(path: Path) -> dict[str, object]:
    lexical = Path(os.path.abspath(path))
    result = subprocess.run(
        [
            str(lexical),
            "-I",
            "-B",
            "-c",
            (
                "import json,sys,sysconfig; "
                "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
                "'purelib':sysconfig.get_paths()['purelib']},sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(result.stdout)
    if not isinstance(runtime, dict) or set(runtime) != {"executable", "prefix", "purelib"}:
        raise ValueError(f"Python runtime identity is malformed: {lexical}")
    return {**_executable_identity(lexical), "runtime": runtime}


def _vla_eval_identity(path: Path, simulator_python: Path) -> dict[str, object]:
    lexical = Path(os.path.abspath(path))
    first_line = lexical.read_text(encoding="utf-8").splitlines()[0]
    expected = f"#!{Path(os.path.abspath(simulator_python))}"
    if first_line != expected:
        raise ValueError(f"vla-eval shebang differs from sealed simulator Python: {first_line!r}")
    return {**_executable_identity(lexical), "shebang": first_line}


def _verify_python_identity(
    expected: object,
    path: Path,
    *,
    sealed_root: Path,
    expected_prefix: Path,
    expected_purelib: Path,
    label: str,
) -> None:
    lexical = Path(os.path.abspath(path))
    if not lexical.is_relative_to(sealed_root):
        raise ValueError(f"{label} lexical path escaped its sealed root")
    actual = _python_identity(lexical)
    if expected != actual:
        raise ValueError(f"{label} executable identity drift")
    runtime = actual["runtime"]
    if runtime != {
        "executable": str(lexical),
        "prefix": str(expected_prefix),
        "purelib": str(expected_purelib),
    }:
        raise ValueError(f"{label} virtual-environment identity drift")


def _verify_vla_eval_identity(expected: object, path: Path, simulator_python: Path, *, sealed_root: Path) -> None:
    lexical = Path(os.path.abspath(path))
    if not lexical.is_relative_to(sealed_root):
        raise ValueError("vla-eval lexical path escaped its sealed runtime")
    if expected != _vla_eval_identity(lexical, simulator_python):
        raise ValueError("vla-eval executable identity drift")


def _distribution_inventory(python: Path) -> bytes:
    """Return a canonical, pip-independent inventory of one isolated virtual environment."""
    script = r"""
import importlib.metadata as metadata
import json
import re

records = []
seen = set()
for distribution in metadata.distributions():
    raw_name = distribution.metadata.get("Name")
    if not raw_name:
        raise RuntimeError("installed distribution has no Name metadata")
    name = re.sub(r"[-_.]+", "-", raw_name).lower()
    if name in seen:
        raise RuntimeError(f"duplicate normalized distribution name: {name}")
    seen.add(name)
    direct_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_text) if direct_text else None
    records.append({"name": name, "version": distribution.version, "direct_url": direct_url})
records.sort(key=lambda record: record["name"])
print(json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
"""
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("isolated distribution inventory is not JSON") from error
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or set(record) != {"name", "version", "direct_url"} for record in records
    ):
        raise ValueError("isolated distribution inventory is malformed")
    names = [record["name"] for record in records]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("isolated distribution inventory has an invalid normalized name")
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("isolated distribution inventory has duplicate or unsorted normalized names")
    return (json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sanitized_source_sha256(source_root: Path) -> str:
    """Hash executable integration source while excluding caches and local artifacts."""
    root = (source_root / "robomme_integration").resolve()
    if not root.is_dir():
        raise ValueError(f"source root lacks robomme_integration: {source_root}")
    digest = hashlib.sha256()

    def field(value: str | bytes) -> None:
        payload = value if isinstance(value, bytes) else value.encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    paths = [
        path
        for path in root.rglob("*")
        if not any(part in SOURCE_EXCLUDED_NAMES for part in path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".log"}
    ]
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        field(relative)
        field(oct(stat.S_IMODE(mode)))
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
            raise ValueError(f"unsupported source entry: {path}")
    return digest.hexdigest()


def _json_file(path: Path, *, label: str) -> tuple[dict, str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _seal_preflight(value: dict) -> str:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    clean.pop("status", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _validate_artifact(value: object, *, uri: str, sha256: str, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"uri", "sha256", "local_sha256"}:
        raise ValueError(f"{label} artifact must contain uri, sha256, and local_sha256")
    if value != {"uri": uri, "sha256": sha256, "local_sha256": sha256}:
        raise ValueError(f"{label} artifact differs from the canonical content-addressed object")
    return value


def _validate_demo_history_configs(queue: dict, source_root: Path) -> None:
    """Fail closed on the official fixed-50 test/demo-history protocol.

    This intentionally checks the human-readable YAML in addition to its SHA.  The campaign's
    normal validator authenticates the bytes; these checks ensure a newly built local queue cannot
    faithfully hash the wrong benchmark semantics.
    """
    for task, record in queue["comparability"]["task_benchmark_configs"].items():
        text = (source_root / record["path"]).read_text(encoding="utf-8")
        required = (
            "episodes_per_task: 50",
            "dataset: test",
            "send_video_history: true",
            f"tasks: [{task}]",
        )
        missing = [fragment for fragment in required if fragment not in text]
        if missing:
            raise ValueError(f"{task} config does not implement fixed-50 test/demo-history: {missing}")


def _validate_workspace_action_probe(preflight: dict, queue: dict) -> None:
    """Require the real two-client Q3 action probe and bind it to this queue/runtime."""
    probe = preflight.get("probe")
    action = probe.get("workspace_action") if isinstance(probe, dict) else None
    if queue.get("queue_id") == LOCAL_PARALLEL_QUEUE_ID:
        topology = parallel_campaign.local_2x5090_topology().as_queue_topology()
        expected_outcome = {
            "unscored": True,
            "arm": "q3",
            "task": "PickXtimes",
            "execution_mode": parallel_campaign.PARALLEL_EXECUTION_MODE,
            "parallel_topology_sha256": topology["parallel_topology_sha256"],
            "parallel_lanes": 2,
            "policy_servers": 2,
            "native_shards_per_lane": 4,
            "concurrent_native_shards": 8,
            "xla_memory_fraction": 0.55,
            "shard_prewarm_seconds": 180.0,
            "shard_stagger_seconds": 30.0,
            "episodes": 8,
            "actions_executed": 8,
            "episode_indices": [0, 0, 1, 1, 2, 2, 3, 3],
            "harness_failures": 0,
            "load_completed_before_readiness": True,
        }
    else:
        expected_outcome = {
            "unscored": True,
            "arm": "q3",
            "task": "PickXtimes",
            "policy_servers": 1,
            "concurrent_native_shards": 2,
            "episodes": 2,
            "actions_executed": 2,
            "episode_indices": [0, 1],
            "harness_failures": 0,
            "load_completed_before_readiness": True,
        }
    if not isinstance(action, dict) or any(
        action.get(name) != expected for name, expected in expected_outcome.items()
    ):
        qualifier = "exact-topology" if queue.get("queue_id") == LOCAL_PARALLEL_QUEUE_ID else "concurrent"
        raise ValueError(f"local preflight has no successful {qualifier} Q3 workspace action probe")
    parallel_probe = queue.get("queue_id") == LOCAL_PARALLEL_QUEUE_ID
    probe_prefix = "local5090-unscored-q3-parallel-v1-" if parallel_probe else "local5090-unscored-q3-action-v1-"
    probe_id = action.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id.startswith(probe_prefix):
        raise ValueError("local workspace action probe has the wrong identity")
    for name in (
        "supervisor_sha256",
        "launch_manifest_sha256",
        "server_log_sha256",
        "launcher_log_sha256",
        "probe_identity_sha256",
        "runtime_fingerprint_sha256",
    ):
        _require_sha(action.get(name), f"workspace action {name}")
    results = action.get("materialized_results")
    expected_results = 2 if parallel_probe else 1
    if not isinstance(results, list) or len(results) != expected_results:
        raise ValueError("local workspace action probe has the wrong aggregate evidence count")
    for result in results:
        if (
            not isinstance(result, dict)
            or set(result) != {"path", "bytes", "sha256"}
            or not isinstance(result.get("path"), str)
            or not Path(result["path"]).is_absolute()
            or not isinstance(result.get("bytes"), int)
            or isinstance(result.get("bytes"), bool)
            or result["bytes"] <= 0
        ):
            raise ValueError("local workspace action aggregate identity is malformed")
        _require_sha(result.get("sha256"), "workspace action aggregate digest")
    if parallel_probe:
        server_logs = action.get("server_logs")
        if (
            not isinstance(server_logs, list)
            or len(server_logs) != 2
            or any(
                not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or not isinstance(record.get("path"), str)
                or not Path(record["path"]).is_absolute()
                or not isinstance(record.get("bytes"), int)
                or isinstance(record.get("bytes"), bool)
                or record["bytes"] <= 0
                for record in server_logs
            )
        ):
            raise ValueError("parallel workspace action server-log evidence is malformed")
        for record in server_logs:
            _require_sha(record.get("sha256"), "parallel workspace server-log digest")
        if [Path(record["path"]).name for record in server_logs] != [
            "server-gpu0-port18100.log",
            "server-gpu1-port18101.log",
        ]:
            raise ValueError("parallel workspace server logs differ from sealed lane ports")
        combined = hashlib.sha256(json.dumps(server_logs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if combined != action["server_log_sha256"]:
            raise ValueError("parallel workspace server-log aggregate digest mismatch")

    inputs = action.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "lineage",
        "policy_checkpoint",
        "workspace_checkpoint",
    }:
        raise ValueError("local workspace action inputs are incomplete")
    cells = [
        cell
        for cell in queue.get("cells", [])
        if isinstance(cell, dict) and cell.get("task") == "PickXtimes" and cell.get("arm") == "q3"
    ]
    if len(cells) != 1:
        raise ValueError("local queue has no unique PickXtimes/Q3 probe lineage")
    cell = cells[0]
    lineage = inputs["lineage"]
    workspace_names = (
        "step",
        "checkpoint_tree_sha256",
        "completion_sha256",
        "run_config_sha256",
        "best_sha256",
        "representation_s3",
    )
    expected_lineage = {
        "queue_id": queue["queue_id"],
        "cell_id": cell["cell_id"],
        "run_id": cell["run_id"],
        "final_step": cell["final_step"],
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
        "run_manifest_sha256": cell["run_manifest_sha256"],
        "training_output_s3": cell["training_output_s3"],
        "training_completion_claim_s3": cell["training_completion_claim_s3"],
        "workspace": {name: cell["workspace"][name] for name in workspace_names},
    }
    if (
        not isinstance(lineage, dict)
        or set(lineage) != {*expected_lineage, "queue_template_file_sha256"}
        or any(lineage.get(name) != expected for name, expected in expected_lineage.items())
    ):
        raise ValueError("local workspace action lineage differs from the scored Q3 queue cell")
    _require_sha(lineage.get("queue_template_file_sha256"), "workspace action queue template digest")

    policy = inputs["policy_checkpoint"]
    workspace = inputs["workspace_checkpoint"]
    if not isinstance(policy, dict) or not isinstance(workspace, dict):
        raise ValueError("local workspace action checkpoint identities are malformed")
    if set(policy) != {
        "path",
        "step",
        "files",
        "bytes",
        "local_tree_sha256",
        "deploy_checkpoint_uri",
        "tree_manifest_path",
        "deploy_tree_manifest_sha256",
    } or set(workspace) != {
        "path",
        "step",
        "files",
        "bytes",
        "local_tree_sha256",
        "producer_tree_sha256",
        "seals",
    }:
        raise ValueError("local workspace action checkpoint identity schema drift")
    policy_path = Path(str(policy.get("path", "")))
    workspace_path = Path(str(workspace.get("path", "")))
    expected_policy_uri = f"{cell['training_output_s3']}/deploy/{cell['final_step']}"
    if (
        not policy_path.is_absolute()
        or not policy_path.name.isdigit()
        or policy.get("step") != cell["final_step"]
        or policy.get("deploy_checkpoint_uri") != expected_policy_uri
        or policy.get("tree_manifest_path") != str(policy_path.parent / "checkpoint-tree.json")
        or not workspace_path.is_absolute()
        or not workspace_path.name.isdigit()
        or workspace.get("step") != cell["workspace"]["step"]
        or workspace.get("producer_tree_sha256") != cell["workspace"]["checkpoint_tree_sha256"]
        or workspace.get("seals")
        != {name: cell["workspace"][name] for name in ("completion_sha256", "run_config_sha256", "best_sha256")}
    ):
        raise ValueError("local workspace action used a checkpoint outside the scored Q3 lineage")
    for label, identity in (("policy", policy), ("workspace", workspace)):
        _require_sha(identity.get("local_tree_sha256"), f"workspace action {label} tree digest")
        if (
            not isinstance(identity.get("bytes"), int)
            or isinstance(identity.get("bytes"), bool)
            or identity["bytes"] <= 0
            or not isinstance(identity.get("files"), list)
            or not identity["files"]
        ):
            raise ValueError(f"workspace action {label} byte inventory is malformed")
        records = identity["files"]
        if any(
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] < 0
            for record in records
        ):
            raise ValueError(f"workspace action {label} file inventory is malformed")
        for record in records:
            _require_sha(record.get("sha256"), f"workspace action {label} file digest")
        if [record["path"] for record in records] != sorted({record["path"] for record in records}) or sum(
            record["bytes"] for record in records
        ) != identity["bytes"]:
            raise ValueError(f"workspace action {label} file inventory is duplicate or inconsistent")
        local_manifest = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if hashlib.sha256(local_manifest).hexdigest() != identity["local_tree_sha256"]:
            raise ValueError(f"workspace action {label} local tree digest mismatch")
    _require_sha(
        policy.get("deploy_tree_manifest_sha256"),
        "workspace action policy deploy-tree digest",
    )

    contract = preflight.get("runtime_contract")
    infrastructure = preflight.get("infrastructure")
    if not isinstance(contract, dict) or not isinstance(infrastructure, dict):
        raise ValueError("local workspace action has no sealed runtime fingerprint inputs")
    fingerprint = {
        "source_tree_sha256": preflight["source_tree_sha256"],
        "gpu_inventory": infrastructure.get("gpu_inventory"),
        "runtime_archive_sha256": contract["runtime"]["local_sha256"],
        "openpi_archive_sha256": contract["openpi"]["local_sha256"],
        "openpi_distribution_inventory_sha256": contract["base_environment"]["distribution_inventory_sha256"],
        "python_version": contract["base_environment"]["python_version"],
        "executables": contract["executables"],
        "critical_file_sha256": contract["critical_file_sha256"],
        "render_environment": contract["render_environment"],
        "policy_import_contract": contract["import_contract"]["policy"],
        "simulator_import_contract": contract["import_contract"]["simulator"],
        "upstream": contract["upstream"],
        "vision": {name: contract["vision"][name] for name in ("revision", "sha256", "bytes")},
    }
    fingerprint_sha = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if action["runtime_fingerprint_sha256"] != fingerprint_sha:
        raise ValueError("local workspace action runtime fingerprint differs from the receipt")
    identity = {
        "schema_version": 1,
        "source_tree_sha256": preflight["source_tree_sha256"],
        "arm": "q3",
        "task": "PickXtimes",
        "episodes": 8 if parallel_probe else 2,
        "lineage": lineage,
        "policy_checkpoint": policy,
        "workspace_checkpoint": workspace,
        "config_sha256": (
            LOCAL_PARALLEL_WORKSPACE_ACTION_CONFIG_SHA256 if parallel_probe else LOCAL_WORKSPACE_ACTION_CONFIG_SHA256
        ),
        "runtime_fingerprint": fingerprint,
    }
    if parallel_probe:
        identity["parallel_topology"] = parallel_campaign.local_2x5090_topology().as_queue_topology()
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if action["probe_identity_sha256"] != identity_sha or probe_id != (f"{probe_prefix}{identity_sha[:20]}"):
        raise ValueError("local workspace action probe identity is not content-derived")


def _validate_exact_panel(
    template: dict,
    *,
    expected_queue_id: str = LOCAL_RETRY_QUEUE_ID,
) -> None:
    expected = {(task, arm) for task in pick_button.TASKS for arm in pick_button.ARMS}
    cells = template.get("cells")
    actual = (
        {(cell.get("task"), cell.get("arm")) for cell in cells if isinstance(cell, dict)}
        if isinstance(cells, list)
        else set()
    )
    if len(cells or []) != 16 or actual != expected:
        raise ValueError(
            "local representation queue must be the exact 16 completed PickXtimes/"
            "ButtonUnmaskSwap cells; rebuild from immutable S3 claims"
        )
    queue_id = template.get("queue_id")
    if queue_id != expected_queue_id:
        raise ValueError(f"this retry must use the fresh queue_id {expected_queue_id!r}")


def finalize_queue(
    template: dict,
    *,
    template_file_sha256: str,
    preflight: dict,
    preflight_file_sha256: str,
    receipt: dict,
    receipt_file_sha256: str,
    source_root: Path,
    topology: dict | None = None,
) -> dict:
    """Bind an unsealed S3-resolved 16-cell template to one exact local runtime."""
    if "gates" in template or "queue_manifest_sha256" in template:
        raise ValueError("local adapter requires an unsealed queue template, not a prior attempt")
    parallel_topology = parallel_campaign.local_2x5090_topology().as_queue_topology()
    if topology is None:
        expected_queue_id = LOCAL_RETRY_QUEUE_ID
        selected_topology = dict(LOCAL_TOPOLOGY)
    else:
        if topology != parallel_topology:
            raise ValueError("local parallel queue requires the exact reviewed 2xRTX5090 topology")
        expected_queue_id = LOCAL_PARALLEL_QUEUE_ID
        selected_topology = parallel_topology
    _validate_exact_panel(template, expected_queue_id=expected_queue_id)
    _require_sha(template_file_sha256, "local queue template digest")
    queue = dict(template)
    _validate_workspace_action_probe(preflight, queue)
    action_template_sha = preflight["probe"]["workspace_action"]["inputs"]["lineage"]["queue_template_file_sha256"]
    if action_template_sha != template_file_sha256:
        raise ValueError("local workspace action probe used different queue-template bytes")
    queue["topology"] = selected_topology
    queue["gates"] = {
        "native_preflight": {
            "preflight_id": preflight["preflight_id"],
            "claim_sha256": preflight_file_sha256,
            "source_tree_sha256": preflight["source_tree_sha256"],
            "queue_template_file_sha256": template_file_sha256,
        },
        "runtime_receipt": {
            "receipt_sha256": receipt_file_sha256,
            "runtime_artifact_sha256": RUNTIME_SHA,
            "openpi_sha256": OPENPI_SHA,
        },
    }
    queue = campaign.seal_document(queue, field="queue_manifest_sha256")
    campaign.validate_queue(queue, source_root=source_root)
    _validate_demo_history_configs(queue, source_root)
    return queue


def _critical_path_sha(paths: dict[str, Path], relative: str) -> str:
    root_name, child = relative.split("/", 1)
    root = paths[root_name]
    target = (root / child).resolve()
    if not target.is_file() or not target.is_relative_to(root.resolve()):
        raise ValueError(f"runtime critical file is absent or unsafe: {relative}")
    return _sha256(target)


def _gpu_inventory() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError(f"malformed nvidia-smi inventory line: {line!r}")
        records.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2]})
    return records


def _verify_materialized_contract(contract: dict, paths: dict[str, Path]) -> None:
    for name in (
        "runtime_archive",
        "openpi_archive",
        "base_openpi_archive",
        "policy_python",
        "simulator_python",
        "vla_eval",
    ):
        if not paths[name].is_file():
            raise ValueError(f"local runtime path is absent: {name}={paths[name]}")
    for name in (
        "harness_src",
        "robomme_src",
        "maniskill_src",
        "openpi_src",
        "policy_site",
        "simulator_site",
        "upstream_root",
        "vision_encoder_home",
    ):
        if not paths[name].is_dir():
            raise ValueError(f"local runtime directory is absent: {name}={paths[name]}")
    if _sha256(paths["runtime_archive"]) != RUNTIME_SHA:
        raise ValueError("local RoboMME runtime archive digest drift")
    if _sha256(paths["openpi_archive"]) != OPENPI_SHA:
        raise ValueError("local standard-ed923 OpenPI archive digest drift")
    if _sha256(paths["base_openpi_archive"]) != OPENPI_SHA:
        raise ValueError("local base OpenPI environment archive digest drift")
    executables = contract.get("executables")
    if not isinstance(executables, dict) or set(executables) != EXECUTABLE_PATH_KEYS:
        raise ValueError("local runtime executable contract is incomplete")
    openpi_root = paths["openpi_src"].parent
    runtime_root = Path(
        os.path.commonpath(
            (
                paths["harness_src"],
                paths["robomme_src"],
                paths["maniskill_src"],
                paths["simulator_site"],
            )
        )
    )
    _verify_python_identity(
        executables["policy_python"],
        paths["policy_python"],
        sealed_root=openpi_root,
        expected_prefix=openpi_root / ".venv",
        expected_purelib=paths["policy_site"],
        label="policy Python",
    )
    simulator_prefix = paths["simulator_site"].parents[2]
    _verify_python_identity(
        executables["simulator_python"],
        paths["simulator_python"],
        sealed_root=runtime_root,
        expected_prefix=simulator_prefix,
        expected_purelib=paths["simulator_site"],
        label="simulator Python",
    )
    _verify_vla_eval_identity(
        executables["vla_eval"],
        paths["vla_eval"],
        paths["simulator_python"],
        sealed_root=runtime_root,
    )
    imports = contract.get("import_contract")
    if not isinstance(imports, dict) or set(imports) != {"policy", "simulator"}:
        raise ValueError("local runtime import contract is missing")
    policy_imports = imports["policy"]
    simulator_imports = imports["simulator"]
    if not isinstance(policy_imports, dict) or not isinstance(simulator_imports, dict):
        raise ValueError("local runtime import contract is malformed")
    policy_roots = {
        "openpi": paths["openpi_src"],
        "jax": paths["policy_site"],
        "numpy": paths["policy_site"],
        "torch": paths["policy_site"],
        "vla_eval": paths["harness_src"],
        "execution_model_server": REPO_ROOT,
        "anyio": paths["simulator_site"],
    }
    for module, expected_root in policy_roots.items():
        imported = policy_imports.get(module)
        if not isinstance(imported, str) or not Path(imported).resolve().is_relative_to(expected_root):
            raise ValueError(f"local policy import contract drift: {module}")
    if policy_imports.get("versions") != POLICY_IMPORT_VERSIONS:
        raise ValueError("local policy import-version contract drift")
    if simulator_imports.get("openpi") is not None:
        raise ValueError("local simulator import contract leaked OpenPI")
    simulator_roots = {
        "numpy": paths["simulator_site"],
        "torch": paths["simulator_site"],
        "mplib": paths["simulator_site"],
        "sapien": paths["simulator_site"],
        "vla_eval": paths["harness_src"],
        "robomme": paths["robomme_src"],
        "mani_skill": paths["maniskill_src"],
    }
    for module, expected_root in simulator_roots.items():
        imported = simulator_imports.get(module)
        if not isinstance(imported, str) or not Path(imported).resolve().is_relative_to(expected_root):
            raise ValueError(f"local simulator import contract drift: {module}")
    if simulator_imports.get("versions") != SIMULATOR_IMPORT_VERSIONS:
        raise ValueError("local simulator import-version contract drift")
    base = contract["base_environment"]
    required_base = {
        "uri",
        "sha256",
        "python_version",
        "uv_lock_sha256",
        "distribution_inventory_sha256",
    }
    if not isinstance(base, dict) or set(base) != required_base:
        raise ValueError("local base OpenPI environment receipt is incomplete")
    uv_lock = paths["openpi_src"].parent / "uv.lock"
    if not uv_lock.is_file() or _sha256(uv_lock) != base["uv_lock_sha256"]:
        raise ValueError("local standard-ed923 uv.lock drift")
    python_version = subprocess.run(
        [str(paths["policy_python"]), "-c", "import platform; print(platform.python_version())"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inventory_sha = hashlib.sha256(_distribution_inventory(paths["policy_python"])).hexdigest()
    if python_version != base["python_version"] or inventory_sha != base["distribution_inventory_sha256"]:
        raise ValueError("local standard-ed923 Python environment drift")
    vision = paths["vision_encoder_home"] / "pi05_vision_encoder/siglip_params.pkl"
    if not vision.is_file() or vision.stat().st_size != VISION_BYTES or _sha256(vision) != VISION_SHA256:
        raise ValueError("local pi0.5 vision weights are absent or corrupt")

    critical = contract.get("critical_file_sha256")
    if not isinstance(critical, dict) or not critical:
        raise ValueError("local runtime receipt has no critical-file identity map")
    actual = {name: _critical_path_sha(paths, name) for name in critical}
    if actual != critical:
        drift = {name: (actual.get(name), digest) for name, digest in critical.items() if actual.get(name) != digest}
        raise ValueError(f"local runtime critical-file drift: {drift}")

    upstream = paths["upstream_root"]
    head = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT or dirty:
        raise ValueError("local upstream policy source is not the exact clean pinned commit")
    for relative, digest in UPSTREAM_CRITICAL_SHA256.items():
        if _sha256(upstream / relative) != digest:
            raise ValueError(f"local upstream critical source drift: {relative}")

    pi0 = paths["openpi_src"] / "openpi/models/pi0.py"
    if "self.wsm_jepa and train" not in pi0.read_text(encoding="utf-8"):
        raise ValueError("local standard-ed923 OpenPI lacks JEPA checkpoint audit support")


def verify_local_gates(
    queue: dict,
    *,
    preflight_claim: Path,
    runtime_receipt: Path,
    source_root: Path,
    verify_materialized: bool = True,
    verify_gpu: bool = False,
) -> campaign.Runtime:
    """Authenticate a local rendered-reset claim and return campaign runtime paths."""
    preflight, preflight_sha = _json_file(preflight_claim, label="local preflight claim")
    receipt, receipt_sha = _json_file(runtime_receipt, label="local runtime receipt")
    gates = queue["gates"]
    if preflight_sha != gates["native_preflight"]["claim_sha256"]:
        raise ValueError("local preflight file digest differs from the queue")
    if receipt_sha != gates["runtime_receipt"]["receipt_sha256"]:
        raise ValueError("local runtime receipt file digest differs from the queue")
    if preflight.get("kind") != LOCAL_PREFLIGHT_KIND or preflight.get("status") != "native_render_reset_passed":
        raise ValueError("local native rendered-reset preflight has not passed")
    if preflight.get("manifest_sha256") != _seal_preflight(preflight):
        raise ValueError("local preflight self-seal mismatch")
    if preflight.get("preflight_id") != gates["native_preflight"]["preflight_id"]:
        raise ValueError("local preflight identity differs from the queue")
    source_sha = gates["native_preflight"]["source_tree_sha256"]
    if preflight.get("source_tree_sha256") != source_sha:
        raise ValueError("local preflight source identity differs from the queue")
    if sanitized_source_sha256(source_root) != source_sha:
        raise ValueError("current sanitized source differs from the successful local preflight")
    probe = preflight.get("probe")
    expected_probe = {
        "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
        "task": "MoveCube",
        "dataset": "test",
        "episode_idx": 0,
        "rendered_reset": True,
        "require_demo_history": True,
        "require_demo_state_history": True,
    }
    if not isinstance(probe, dict) or any(probe.get(key) != value for key, value in expected_probe.items()):
        raise ValueError("local preflight did not certify the exact test/demo-history rendered reset")
    frames = probe.get("observed_demo_frames") if isinstance(probe, dict) else None
    states = probe.get("observed_demo_states") if isinstance(probe, dict) else None
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError("local preflight observed no demonstration-history frames")
    if not isinstance(states, int) or isinstance(states, bool) or states != frames:
        raise ValueError("local preflight observed missing/unpaired demonstration-state history")
    _validate_workspace_action_probe(preflight, queue)
    action_template_sha = preflight["probe"]["workspace_action"]["inputs"]["lineage"]["queue_template_file_sha256"]
    if gates["native_preflight"].get("queue_template_file_sha256") != action_template_sha:
        raise ValueError("local queue gate is not bound to the workspace-action template bytes")
    infrastructure = preflight.get("infrastructure")
    if not isinstance(infrastructure, dict) or infrastructure.get("provider") != "local_workstation":
        raise ValueError("local preflight has the wrong infrastructure provider")
    if infrastructure.get("accelerator") != LOCAL_ACCELERATOR:
        raise ValueError("local preflight was not performed on the two-RTX-5090 target")
    gpu_inventory = infrastructure.get("gpu_inventory")
    if (
        not isinstance(gpu_inventory, list)
        or len(gpu_inventory) != 2
        or [record.get("index") for record in gpu_inventory if isinstance(record, dict)] != [0, 1]
        or any(record.get("name") != "NVIDIA GeForce RTX 5090" for record in gpu_inventory)
    ):
        raise ValueError("local preflight GPU inventory is not exact two-RTX-5090")

    if receipt.get("kind") != LOCAL_RECEIPT_KIND or receipt.get("status") != "staged_and_verified":
        raise ValueError("local runtime receipt is not staged and verified")
    if campaign._seal_digest(receipt, "receipt_sha256") != _require_sha(
        receipt.get("receipt_sha256"), "local receipt seal"
    ):
        raise ValueError("local runtime receipt self-seal mismatch")
    if receipt.get("preflight_claim_sha256") != preflight_sha:
        raise ValueError("local runtime receipt is not bound to the supplied preflight")
    if receipt.get("source_tree_sha256") != source_sha:
        raise ValueError("local runtime receipt source identity drift")
    contract = receipt.get("runtime_contract")
    if (
        contract != preflight.get("runtime_contract")
        or not isinstance(contract, dict)
        or set(contract) != CONTRACT_KEYS
    ):
        raise ValueError("local runtime contract differs from the rendered-reset preflight")
    _validate_artifact(contract["runtime"], uri=RUNTIME_S3, sha256=RUNTIME_SHA, label="runtime")
    _validate_artifact(contract["openpi"], uri=OPENPI, sha256=OPENPI_SHA, label="OpenPI")
    base = contract["base_environment"]
    if not isinstance(base, dict) or base.get("uri") != OPENPI or base.get("sha256") != OPENPI_SHA:
        raise ValueError("local base OpenPI environment identity drift")
    if contract["upstream"] != {
        "commit": UPSTREAM_COMMIT,
        "critical_sha256": UPSTREAM_CRITICAL_SHA256,
    }:
        raise ValueError("local upstream source identity drift")
    if contract["vision"] != {
        "uri": VISION_S3,
        "revision": VISION_REVISION,
        "sha256": VISION_SHA256,
        "bytes": VISION_BYTES,
    }:
        raise ValueError("local vision asset identity drift")
    if contract["render_environment"] != LOCAL_RENDER_ENVIRONMENT:
        raise ValueError("local evaluator is not sealed to native NVIDIA EGL (lavapipe disabled)")
    raw_paths = contract["paths"]
    if not isinstance(raw_paths, dict) or set(raw_paths) != PATH_KEYS:
        raise ValueError("local runtime path contract is incomplete or expanded")
    paths = {}
    for name, raw in raw_paths.items():
        if not isinstance(raw, str):
            raise ValueError(f"local runtime path {name} must be a string")
        paths[name] = _contract_path(name, raw)
    if verify_materialized:
        _verify_materialized_contract(contract, paths)
    if verify_gpu and _gpu_inventory() != gpu_inventory:
        raise ValueError("current GPU inventory differs from the successful local preflight")
    return campaign.Runtime(
        receipt_sha256=receipt_sha,
        preflight_claim_sha256=preflight_sha,
        policy_python=paths["policy_python"],
        vla_eval=paths["vla_eval"],
        harness_src=paths["harness_src"],
        robomme_src=paths["robomme_src"],
        maniskill_src=paths["maniskill_src"],
        openpi_src=paths["openpi_src"],
        policy_site=paths["policy_site"],
        simulator_site=paths["simulator_site"],
        upstream_root=paths["upstream_root"],
        vision_encoder_home=paths["vision_encoder_home"],
        render_environment=dict(LOCAL_RENDER_ENVIRONMENT),
    )


def _gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"local GPUs already have compute processes: {result.stdout.strip()}")


def _ports_free(base_port: int, count: int) -> None:
    for port in range(base_port, base_port + count):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"local policy port {port} is occupied") from error


def _topology_ports_free(topology: dict) -> None:
    if topology.get("execution_mode") == parallel_campaign.PARALLEL_EXECUTION_MODE:
        for lane in parallel_campaign.ParallelTopology.from_queue_topology(topology).lanes:
            _ports_free(lane.port, 1)
        return
    _ports_free(topology["base_port"], len(topology["policy_gpus"]))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--queue-template", type=Path, required=True)
    value.add_argument("--native-preflight-claim", type=Path, required=True)
    value.add_argument("--runtime-receipt", type=Path, required=True)
    value.add_argument("--source-root", type=Path, default=REPO_ROOT)
    value.add_argument("--work-root", type=Path, default=None)
    value.add_argument("--sealed-queue-output", type=Path)
    value.add_argument(
        "--parallel-fixed50",
        action="store_true",
        help="seal the fresh queue for two disjoint one-cell-per-RTX-5090 lanes",
    )
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "--confirm-run",
        action="store_true",
        help="start scored local evaluation only after explicit user approval",
    )
    return value


def build_plan(args: argparse.Namespace, *, verify_materialized: bool = True) -> dict:
    source_root = args.source_root.expanduser().resolve()
    preflight_path = args.native_preflight_claim.expanduser().resolve()
    receipt_path = args.runtime_receipt.expanduser().resolve()
    template_payload = args.queue_template.expanduser().resolve().read_bytes()
    template = json.loads(template_payload)
    if not isinstance(template, dict):
        raise ValueError("local eval queue template must contain one JSON object")
    preflight, preflight_sha = _json_file(preflight_path, label="local preflight claim")
    receipt, receipt_sha = _json_file(receipt_path, label="local runtime receipt")
    queue = finalize_queue(
        template,
        template_file_sha256=hashlib.sha256(template_payload).hexdigest(),
        preflight=preflight,
        preflight_file_sha256=preflight_sha,
        receipt=receipt,
        receipt_file_sha256=receipt_sha,
        source_root=source_root,
        topology=(
            parallel_campaign.local_2x5090_topology().as_queue_topology()
            if getattr(args, "parallel_fixed50", False)
            else None
        ),
    )
    runtime = verify_local_gates(
        queue,
        preflight_claim=preflight_path,
        runtime_receipt=receipt_path,
        source_root=source_root,
        verify_materialized=verify_materialized,
        verify_gpu=False,
    )
    work_root = (
        args.work_root.expanduser().resolve()
        if args.work_root is not None
        else (DEFAULT_LOCAL_ROOT / "campaigns" / queue["queue_id"]).resolve()
    )
    return {"queue": queue, "runtime": runtime, "source_root": source_root, "work_root": work_root}


def main() -> int:
    args = parser().parse_args()
    if args.dry_run and args.confirm_run:
        raise SystemExit("choose --dry-run or --confirm-run, not both")
    plan = build_plan(args)
    queue = plan["queue"]
    payload = campaign._canonical(queue)
    if args.sealed_queue_output is not None:
        _atomic_write(args.sealed_queue_output.expanduser().resolve(), payload)
    summary = {
        "queue_id": queue["queue_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "cells": len(queue["cells"]),
        "topology": queue["topology"],
        "work_root": str(plan["work_root"]),
        "protocol": "fixed50/test/send_video_history=true",
        "rendering": "native NVIDIA EGL; lavapipe disabled",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.confirm_run:
        print("DRY RUN ONLY — no S3 access and no policy/simulator process started")
        return 0

    # Recheck the machine only at the explicit execution boundary.  The receipt is already exact;
    # these guards prevent running it on a different workstation or trampling an active local job.
    runtime = verify_local_gates(
        queue,
        preflight_claim=args.native_preflight_claim.expanduser().resolve(),
        runtime_receipt=args.runtime_receipt.expanduser().resolve(),
        source_root=plan["source_root"],
        verify_materialized=True,
        verify_gpu=True,
    )
    _gpu_idle()
    _topology_ports_free(queue["topology"])
    store = campaign.AwsCliStore()
    if queue["topology"].get("execution_mode") == parallel_campaign.PARALLEL_EXECUTION_MODE:
        return parallel_campaign.ParallelCampaignRunner(
            queue=queue,
            source_root=plan["source_root"],
            work_root=plan["work_root"],
            runtime=runtime,
            store=store,
            stager=campaign.AwsStager(store),
            artifacts=campaign.Fixed50Artifacts(),
        ).run()
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=plan["source_root"],
        work_root=plan["work_root"],
        runtime=runtime,
        store=store,
        stager=campaign.AwsStager(store),
        evaluator=campaign.SubprocessEvaluator(plan["source_root"], runtime),
        artifacts=campaign.Fixed50Artifacts(),
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
