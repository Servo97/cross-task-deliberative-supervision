from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from robomme_integration.eval import campaign
from robomme_integration.eval import local_rtx5090_campaign as local
from robomme_integration.eval import local_rtx5090_preflight as preflight
from robomme_integration.eval.launch_p5_campaign import (
    UPSTREAM_COMMIT,
    UPSTREAM_CRITICAL_SHA256,
    VISION_BYTES,
    VISION_REVISION,
    VISION_S3,
    VISION_SHA256,
)
from robomme_integration.eval.launch_p5_preflight import RUNTIME_S3, RUNTIME_SHA
from robomme_integration.fleet.checkpoint import build as build_checkpoint_manifest
from robomme_integration.launch import OPENPI, OPENPI_SHA


def _write(path: Path, value: dict) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _probe_cell() -> dict:
    return {
        "cell_id": "000-pickxtimes-q3",
        "task": "PickXtimes",
        "arm": "q3",
        "run_id": "st-v1-pickxtimes-q3-seed0-test",
        "final_step": 19999,
        "scientific_spec_sha256": "1" * 64,
        "run_manifest_sha256": "2" * 64,
        "training_output_s3": "s3://bucket/pickxtimes/q3",
        "training_completion_claim_s3": "s3://bucket/pickxtimes/q3/complete.json",
        "workspace": {
            "step": 10000,
            "checkpoint_tree_sha256": "3" * 64,
            "completion_sha256": "4" * 64,
            "run_config_sha256": "5" * 64,
            "best_sha256": "6" * 64,
            "representation_s3": "s3://bucket/workspace/PickXtimes/step-10000",
        },
    }


def _workspace_action(
    preflight: dict,
    cell: dict,
    *,
    template_sha: str,
    parallel: bool = False,
) -> dict:
    lineage = {
        "queue_id": (local.LOCAL_PARALLEL_QUEUE_ID if parallel else local.LOCAL_RETRY_QUEUE_ID),
        "queue_template_file_sha256": template_sha,
        "cell_id": cell["cell_id"],
        "run_id": cell["run_id"],
        "final_step": cell["final_step"],
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
        "run_manifest_sha256": cell["run_manifest_sha256"],
        "training_output_s3": cell["training_output_s3"],
        "training_completion_claim_s3": cell["training_completion_claim_s3"],
        "workspace": dict(cell["workspace"]),
    }
    policy_path = Path("/tmp/probe/checkpoint/19999")
    workspace_path = Path("/tmp/probe/workspace/10000")
    policy_files = [{"path": "params/mock", "bytes": 1, "sha256": "7" * 64}]
    workspace_files = [{"path": "state/mock", "bytes": 1, "sha256": "9" * 64}]
    policy_local_sha = hashlib.sha256(
        (json.dumps(policy_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    workspace_local_sha = hashlib.sha256(
        (json.dumps(workspace_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    policy = {
        "path": str(policy_path),
        "step": 19999,
        "files": policy_files,
        "bytes": 1,
        "local_tree_sha256": policy_local_sha,
        "deploy_checkpoint_uri": f"{cell['training_output_s3']}/deploy/19999",
        "tree_manifest_path": str(policy_path.parent / "checkpoint-tree.json"),
        "deploy_tree_manifest_sha256": "7" * 64,
    }
    workspace = {
        "path": str(workspace_path),
        "step": 10000,
        "files": workspace_files,
        "bytes": 1,
        "local_tree_sha256": workspace_local_sha,
        "producer_tree_sha256": cell["workspace"]["checkpoint_tree_sha256"],
        "seals": {name: cell["workspace"][name] for name in ("completion_sha256", "run_config_sha256", "best_sha256")},
    }
    contract = preflight["runtime_contract"]
    fingerprint = {
        "source_tree_sha256": preflight["source_tree_sha256"],
        "gpu_inventory": preflight["infrastructure"]["gpu_inventory"],
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
    inputs = {
        "lineage": lineage,
        "policy_checkpoint": policy,
        "workspace_checkpoint": workspace,
    }
    topology = local.parallel_campaign.local_2x5090_topology().as_queue_topology()
    identity = {
        "schema_version": 1,
        "source_tree_sha256": preflight["source_tree_sha256"],
        "arm": "q3",
        "task": "PickXtimes",
        "episodes": 8 if parallel else 2,
        "lineage": lineage,
        "policy_checkpoint": policy,
        "workspace_checkpoint": workspace,
        "config_sha256": (
            local.LOCAL_PARALLEL_WORKSPACE_ACTION_CONFIG_SHA256
            if parallel
            else local.LOCAL_WORKSPACE_ACTION_CONFIG_SHA256
        ),
        "runtime_fingerprint": fingerprint,
    }
    if parallel:
        identity["parallel_topology"] = topology
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    fingerprint_sha = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    server_logs = [
        {
            "path": f"/tmp/probe/server-gpu{gpu}-port{18100 + gpu}.log",
            "bytes": 1,
            "sha256": str(gpu + 1) * 64,
        }
        for gpu in range(2)
    ]
    server_log_sha = (
        hashlib.sha256(json.dumps(server_logs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if parallel
        else "d" * 64
    )
    value = {
        "unscored": True,
        "inputs": inputs,
        "probe_id": (
            f"local5090-unscored-q3-parallel-v1-{identity_sha[:20]}"
            if parallel
            else f"local5090-unscored-q3-action-v1-{identity_sha[:20]}"
        ),
        "arm": "q3",
        "task": "PickXtimes",
        "policy_servers": 2 if parallel else 1,
        "concurrent_native_shards": 8 if parallel else 2,
        "episodes": 8 if parallel else 2,
        "actions_executed": 8 if parallel else 2,
        "episode_indices": [0, 0, 1, 1, 2, 2, 3, 3] if parallel else [0, 1],
        "harness_failures": 0,
        "load_completed_before_readiness": True,
        "supervisor_sha256": "b" * 64,
        "launch_manifest_sha256": "c" * 64,
        "server_log_sha256": server_log_sha,
        "launcher_log_sha256": "e" * 64,
        "probe_identity_sha256": identity_sha,
        "runtime_fingerprint_sha256": fingerprint_sha,
        "materialized_results": [
            {
                "path": f"/tmp/probe/aggregate-{index}.json",
                "bytes": 1,
                "sha256": "f" * 64,
            }
            for index in range(2 if parallel else 1)
        ],
    }
    if parallel:
        value.update(
            execution_mode=local.parallel_campaign.PARALLEL_EXECUTION_MODE,
            parallel_topology_sha256=topology["parallel_topology_sha256"],
            parallel_lanes=2,
            native_shards_per_lane=4,
            xla_memory_fraction=0.55,
            shard_prewarm_seconds=180.0,
            shard_stagger_seconds=30.0,
            server_logs=server_logs,
        )
    return value


def _gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: int = 8,
    states: int | None = None,
):
    source_sha = "a" * 64
    monkeypatch.setattr(local, "sanitized_source_sha256", lambda _root: source_sha)
    paths = {name: str((tmp_path / name).resolve()) for name in local.PATH_KEYS}
    contract = {
        "runtime": {"uri": RUNTIME_S3, "sha256": RUNTIME_SHA, "local_sha256": RUNTIME_SHA},
        "openpi": {"uri": OPENPI, "sha256": OPENPI_SHA, "local_sha256": OPENPI_SHA},
        "base_environment": {
            "uri": OPENPI,
            "sha256": OPENPI_SHA,
            "python_version": "3.11.14",
            "distribution_inventory_sha256": "0" * 64,
        },
        "upstream": {"commit": UPSTREAM_COMMIT, "critical_sha256": UPSTREAM_CRITICAL_SHA256},
        "vision": {
            "uri": VISION_S3,
            "revision": VISION_REVISION,
            "sha256": VISION_SHA256,
            "bytes": VISION_BYTES,
        },
        "paths": paths,
        "critical_file_sha256": {"openpi_src/openpi/models/pi0.py": "b" * 64},
        "render_environment": dict(local.LOCAL_RENDER_ENVIRONMENT),
        "executables": {name: {"test": name} for name in local.EXECUTABLE_PATH_KEYS},
        "import_contract": {
            "policy": {
                "devices": 2,
                "openpi": str((tmp_path / "openpi_src/openpi/__init__.py").resolve()),
                "jax": str((tmp_path / "policy_site/jax/__init__.py").resolve()),
                "numpy": str((tmp_path / "policy_site/numpy/__init__.py").resolve()),
                "torch": str((tmp_path / "policy_site/torch/__init__.py").resolve()),
                "vla_eval": str((tmp_path / "harness_src/vla_eval/__init__.py").resolve()),
                "execution_model_server": str(
                    (local.REPO_ROOT / "robomme_integration/eval/execution_model_server.py").resolve()
                ),
                "anyio": str((tmp_path / "simulator_site/anyio/__init__.py").resolve()),
                "versions": dict(local.POLICY_IMPORT_VERSIONS),
                "server_help_sha256": "c" * 64,
            },
            "simulator": {
                "openpi": None,
                "numpy": str((tmp_path / "simulator_site/numpy/__init__.py").resolve()),
                "torch": str((tmp_path / "simulator_site/torch/__init__.py").resolve()),
                "mplib": str((tmp_path / "simulator_site/mplib/__init__.py").resolve()),
                "sapien": str((tmp_path / "simulator_site/sapien/__init__.py").resolve()),
                "vla_eval": str((tmp_path / "harness_src/vla_eval/__init__.py").resolve()),
                "robomme": str((tmp_path / "robomme_src/robomme/__init__.py").resolve()),
                "mani_skill": str((tmp_path / "maniskill_src/mani_skill/__init__.py").resolve()),
                "versions": dict(local.SIMULATOR_IMPORT_VERSIONS),
            },
        },
    }
    preflight = {
        "schema_version": 1,
        "kind": local.LOCAL_PREFLIGHT_KIND,
        "preflight_id": "local5090-native-eval-v1-test",
        "source_tree_sha256": source_sha,
        "probe": {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
            "observed_demo_frames": frames,
            "observed_demo_states": frames if states is None else states,
        },
        "runtime_contract": contract,
        "infrastructure": {
            "provider": "local_workstation",
            "accelerator": local.LOCAL_ACCELERATOR,
            "gpu_inventory": [
                {"index": 0, "name": "NVIDIA GeForce RTX 5090", "uuid": "GPU-a"},
                {"index": 1, "name": "NVIDIA GeForce RTX 5090", "uuid": "GPU-b"},
            ],
        },
    }
    cell = _probe_cell()
    template_sha = "d" * 64
    preflight["probe"]["workspace_action"] = _workspace_action(
        preflight,
        cell,
        template_sha=template_sha,
    )
    preflight["manifest_sha256"] = local._seal_preflight(preflight)
    preflight["status"] = "native_render_reset_passed"
    preflight_path = tmp_path / "preflight.json"
    preflight_sha = _write(preflight_path, preflight)
    receipt = campaign.seal_document(
        {
            "schema_version": 1,
            "kind": local.LOCAL_RECEIPT_KIND,
            "status": "staged_and_verified",
            "preflight_claim_sha256": preflight_sha,
            "source_tree_sha256": source_sha,
            "runtime_contract": contract,
        },
        field="receipt_sha256",
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_sha = _write(receipt_path, receipt)
    queue = {
        "queue_id": local.LOCAL_RETRY_QUEUE_ID,
        "cells": [cell],
        "gates": {
            "native_preflight": {
                "preflight_id": preflight["preflight_id"],
                "claim_sha256": preflight_sha,
                "source_tree_sha256": source_sha,
                "queue_template_file_sha256": template_sha,
            },
            "runtime_receipt": {
                "receipt_sha256": receipt_sha,
                "runtime_artifact_sha256": RUNTIME_SHA,
                "openpi_sha256": OPENPI_SHA,
            },
        },
    }
    return queue, preflight_path, receipt_path


def test_local_gate_is_standard_ed923_native_egl_and_demo_history_exact(tmp_path, monkeypatch):
    queue, preflight, receipt = _gates(tmp_path, monkeypatch)
    runtime = local.verify_local_gates(
        queue,
        preflight_claim=preflight,
        runtime_receipt=receipt,
        source_root=tmp_path,
        verify_materialized=False,
    )
    assert runtime.render_environment["ROBOMME_USE_LAVAPIPE"] == "0"
    assert runtime.openpi_src == (tmp_path / "openpi_src").resolve()
    assert queue["gates"]["runtime_receipt"]["openpi_sha256"] == OPENPI_SHA


def test_local_gate_rejects_empty_demo_history(tmp_path, monkeypatch):
    queue, preflight, receipt = _gates(tmp_path, monkeypatch, frames=0)
    with pytest.raises(ValueError, match="no demonstration-history frames"):
        local.verify_local_gates(
            queue,
            preflight_claim=preflight,
            runtime_receipt=receipt,
            source_root=tmp_path,
            verify_materialized=False,
        )


def test_local_gate_rejects_unpaired_demo_state_history(tmp_path, monkeypatch):
    queue, preflight, receipt = _gates(tmp_path, monkeypatch, frames=8, states=7)
    with pytest.raises(ValueError, match="unpaired demonstration-state history"):
        local.verify_local_gates(
            queue,
            preflight_claim=preflight,
            runtime_receipt=receipt,
            source_root=tmp_path,
            verify_materialized=False,
        )


def _rewrite_preflight_gate(queue: dict, preflight_path: Path, receipt_path: Path, mutate) -> None:
    claim = json.loads(preflight_path.read_text(encoding="utf-8"))
    mutate(claim)
    claim["manifest_sha256"] = local._seal_preflight(claim)
    claim_sha = _write(preflight_path, claim)
    queue["gates"]["native_preflight"]["claim_sha256"] = claim_sha
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["preflight_claim_sha256"] = claim_sha
    receipt = campaign.seal_document(receipt, field="receipt_sha256")
    queue["gates"]["runtime_receipt"]["receipt_sha256"] = _write(receipt_path, receipt)


def test_local_scored_gate_rejects_missing_workspace_action_probe(tmp_path, monkeypatch):
    queue, preflight_path, receipt_path = _gates(tmp_path, monkeypatch)
    _rewrite_preflight_gate(
        queue,
        preflight_path,
        receipt_path,
        lambda claim: claim["probe"].pop("workspace_action"),
    )
    with pytest.raises(ValueError, match="no successful concurrent Q3 workspace action probe"):
        local.verify_local_gates(
            queue,
            preflight_claim=preflight_path,
            runtime_receipt=receipt_path,
            source_root=tmp_path,
            verify_materialized=False,
        )


def test_local_scored_gate_rejects_workspace_action_template_mismatch(tmp_path, monkeypatch):
    queue, preflight_path, receipt_path = _gates(tmp_path, monkeypatch)
    queue["gates"]["native_preflight"]["queue_template_file_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="not bound to the workspace-action template bytes"):
        local.verify_local_gates(
            queue,
            preflight_claim=preflight_path,
            runtime_receipt=receipt_path,
            source_root=tmp_path,
            verify_materialized=False,
        )


def test_local_finalizer_requires_exact_16_and_replaces_p5_topology(monkeypatch, tmp_path):
    cells = [{"task": task, "arm": arm} for task in local.pick_button.TASKS for arm in local.pick_button.ARMS]
    q3 = _probe_cell()
    cells = [q3 if (cell["task"], cell["arm"]) == ("PickXtimes", "q3") else cell for cell in cells]
    template = {"queue_id": local.LOCAL_RETRY_QUEUE_ID, "cells": cells}
    _gate_queue, preflight_path, _receipt_path = _gates(tmp_path, monkeypatch)
    preflight_claim = json.loads(preflight_path.read_text(encoding="utf-8"))
    template_sha = "d" * 64
    monkeypatch.setattr(local.campaign, "validate_queue", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(local, "_validate_demo_history_configs", lambda *_args: None)
    queue = local.finalize_queue(
        template,
        template_file_sha256=template_sha,
        preflight=preflight_claim,
        preflight_file_sha256="b" * 64,
        receipt={},
        receipt_file_sha256="c" * 64,
        source_root=tmp_path,
    )
    assert queue["topology"] == local.LOCAL_TOPOLOGY
    assert queue["gates"]["runtime_receipt"]["openpi_sha256"] == OPENPI_SHA
    assert campaign._seal_digest(queue, "queue_manifest_sha256") == queue["queue_manifest_sha256"]

    parallel_preflight = json.loads(json.dumps(preflight_claim))
    parallel_preflight["probe"]["workspace_action"] = _workspace_action(
        parallel_preflight,
        q3,
        template_sha=template_sha,
        parallel=True,
    )
    parallel_queue = local.finalize_queue(
        {**template, "queue_id": local.LOCAL_PARALLEL_QUEUE_ID},
        template_file_sha256=template_sha,
        preflight=parallel_preflight,
        preflight_file_sha256="b" * 64,
        receipt={},
        receipt_file_sha256="c" * 64,
        source_root=tmp_path,
        topology=local.parallel_campaign.local_2x5090_topology().as_queue_topology(),
    )
    assert parallel_queue["topology"]["execution_mode"] == "parallel_fixed50_lanes_v1"
    assert parallel_queue["queue_manifest_sha256"] != queue["queue_manifest_sha256"]

    # v2 preflight lineage can never be repurposed for the parallel queue identity.
    with pytest.raises(ValueError, match="local5090-v3"):
        local.finalize_queue(
            template,
            template_file_sha256=template_sha,
            preflight=preflight_claim,
            preflight_file_sha256="b" * 64,
            receipt={},
            receipt_file_sha256="c" * 64,
            source_root=tmp_path,
            topology=local.parallel_campaign.local_2x5090_topology().as_queue_topology(),
        )
    with pytest.raises(ValueError, match="exact-topology"):
        local.finalize_queue(
            {**template, "queue_id": local.LOCAL_PARALLEL_QUEUE_ID},
            template_file_sha256=template_sha,
            preflight=preflight_claim,
            preflight_file_sha256="b" * 64,
            receipt={},
            receipt_file_sha256="c" * 64,
            source_root=tmp_path,
            topology=local.parallel_campaign.local_2x5090_topology().as_queue_topology(),
        )

    with pytest.raises(ValueError, match="exact 16"):
        local.finalize_queue(
            {**template, "cells": cells[:-1]},
            template_file_sha256=template_sha,
            preflight=preflight_claim,
            preflight_file_sha256="b" * 64,
            receipt={},
            receipt_file_sha256="c" * 64,
            source_root=tmp_path,
        )


def test_local_entry_is_approval_gated_by_default():
    args = local.parser().parse_args(
        [
            "--queue-template",
            "queue.json",
            "--native-preflight-claim",
            "preflight.json",
            "--runtime-receipt",
            "receipt.json",
        ]
    )
    assert args.confirm_run is False
    assert args.parallel_fixed50 is False


def test_preflight_separates_policy_and_simulator_import_stacks(tmp_path):
    paths = {
        name: tmp_path / name
        for name in (
            "harness_src",
            "robomme_src",
            "maniskill_src",
            "openpi_src",
            "policy_site",
            "simulator_site",
            "upstream_root",
        )
    }
    sealed = paths["openpi_src"] / "openpi"
    reference = paths["upstream_root"] / "src/openpi"
    sealed.mkdir(parents=True)
    reference.mkdir(parents=True)
    (sealed / "__init__.py").write_text("SOURCE = 'sealed'\n", encoding="utf-8")
    (reference / "__init__.py").write_text("SOURCE = 'reference'\n", encoding="utf-8")
    for module in ("numpy", "torch"):
        package = paths["policy_site"] / module
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("SOURCE = 'policy'\n", encoding="utf-8")
    for module in ("numpy", "torch", "mplib", "anyio"):
        package = paths["simulator_site"] / module
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("SOURCE = 'simulator'\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = preflight._pythonpath(paths, tmp_path / "source_root")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import anyio,numpy,openpi,torch; "
                "print(openpi.SOURCE,numpy.SOURCE,torch.SOURCE,anyio.SOURCE); print(openpi.__file__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "sealed policy policy simulator"
    assert Path(lines[1]).is_relative_to(paths["openpi_src"])
    policy_paths = preflight._pythonpath(paths, tmp_path / "source_root").split(os.pathsep)
    assert policy_paths == [
        str(tmp_path / "source_root"),
        str(paths["harness_src"]),
        str(paths["openpi_src"]),
        str(paths["policy_site"]),
        str(paths["upstream_root"] / "src"),
        str(paths["simulator_site"]),
    ]

    simulator_path = preflight._simulator_pythonpath(paths, tmp_path / "source_root")
    assert str(paths["openpi_src"]) not in simulator_path.split(os.pathsep)
    assert str(paths["policy_site"]) not in simulator_path.split(os.pathsep)
    assert str(paths["upstream_root"] / "src") not in simulator_path.split(os.pathsep)
    assert simulator_path.split(os.pathsep) == [
        str(tmp_path / "source_root"),
        str(paths["harness_src"]),
        str(paths["robomme_src"]),
        str(paths["maniskill_src"]),
        str(paths["simulator_site"]),
    ]
    environment["PYTHONPATH"] = simulator_path
    simulator = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.util,mplib,numpy,torch; "
                "print(numpy.SOURCE,torch.SOURCE,mplib.SOURCE); "
                "print(importlib.util.find_spec('openpi'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert simulator.stdout.splitlines() == ["simulator simulator simulator", "None"]


def test_preflight_rejects_source_drift_around_rendered_probe(tmp_path):
    package = tmp_path / "robomme_integration"
    package.mkdir()
    source = package / "adapter.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    before = local.sanitized_source_sha256(tmp_path)
    preflight._require_source_unchanged(tmp_path, before)
    source.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed during the rendered-reset preflight"):
        preflight._require_source_unchanged(tmp_path, before)


def _workspace_probe_checkpoints(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    policy = tmp_path / "policy/19999"
    workspace = tmp_path / "workspace/10000"
    for path, payload in {
        policy / "assets/robomme/norm_stats.json": "{}\n",
        policy / "params/_METADATA": "metadata\n",
        policy / "params/_sharding": "sharding\n",
        policy / "params/manifest.ocdbt": "manifest\n",
        policy / "params/d/chunk": "weights\n",
        workspace / "WSM_RUN_CONFIG.json": json.dumps({"task": "PickXtimes", "steps": 10000}),
        workspace / "WSM_BEST.json": json.dumps({"best_step": 10000}),
        workspace / "WSM_GENERATION_COMPLETE.json": json.dumps({"step": 10000}),
        workspace / "_CHECKPOINT_METADATA": "metadata\n",
        workspace / "_UPLOAD_COMPLETE.json": json.dumps(
            {
                "schema_version": 1,
                "step": 10000,
                "completed_at": "2026-08-11T00:00:00+00:00",
                "source_marker": "_CHECKPOINT_METADATA",
            }
        ),
        workspace / "state/_METADATA": "metadata\n",
        workspace / "state/manifest.ocdbt": "manifest\n",
        workspace / "state/d/chunk": "representation\n",
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    checkpoint_uri = "s3://bucket/pickxtimes/q3/deploy/19999"
    policy_tree = policy.parent / "checkpoint-tree.json"
    build_checkpoint_manifest(
        policy,
        checkpoint_uri,
        policy_tree,
        workers=1,
        require_finalized=False,
    )
    workspace_identity = campaign._legacy_workspace_tree_sha256(workspace, expected_step=10000)
    q3_cell = {
        "cell_id": "000-pickxtimes-q3",
        "task": "PickXtimes",
        "arm": "q3",
        "run_id": "st-v1-pickxtimes-q3-seed0-test",
        "final_step": 19999,
        "scientific_spec_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "training_output_s3": "s3://bucket/pickxtimes/q3",
        "training_completion_claim_s3": "s3://bucket/pickxtimes/q3/complete.json",
        "workspace": {
            "step": 10000,
            "checkpoint_tree_sha256": workspace_identity,
            "completion_sha256": preflight._sha(workspace / "WSM_GENERATION_COMPLETE.json"),
            "run_config_sha256": preflight._sha(workspace / "WSM_RUN_CONFIG.json"),
            "best_sha256": preflight._sha(workspace / "WSM_BEST.json"),
            "representation_s3": "s3://bucket/workspace/PickXtimes/step-10000",
        },
    }
    cells = [
        (q3_cell if (task, arm) == ("PickXtimes", "q3") else {"task": task, "arm": arm})
        for task in local.pick_button.TASKS
        for arm in local.pick_button.ARMS
    ]
    queue = {
        "queue_id": local.LOCAL_RETRY_QUEUE_ID,
        "cells": cells,
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    return policy, workspace, queue_path, policy_tree


def test_workspace_action_probe_binds_every_checkpoint_byte_and_numeric_step(tmp_path):
    policy, workspace, queue, policy_tree = _workspace_probe_checkpoints(tmp_path)
    first = preflight._workspace_probe_inputs(
        policy,
        workspace,
        queue_template=queue,
        policy_tree_manifest=policy_tree,
    )
    assert first["policy_checkpoint"]["step"] == 19999
    assert first["workspace_checkpoint"]["step"] == 10000
    assert first["workspace_checkpoint"]["bytes"] > 0

    (workspace / "state/d/chunk").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="representation tree differs"):
        preflight._workspace_probe_inputs(
            policy,
            workspace,
            queue_template=queue,
            policy_tree_manifest=policy_tree,
        )

    nonnumeric = tmp_path / "workspace-root"
    workspace.rename(nonnumeric)
    with pytest.raises(ValueError, match="numeric step"):
        preflight._workspace_probe_inputs(
            policy,
            nonnumeric,
            queue_template=queue,
            policy_tree_manifest=policy_tree,
        )


def test_workspace_action_probe_command_is_one_policy_server_two_native_shards(tmp_path):
    paths = {
        name: tmp_path / name
        for name in (
            "policy_python",
            "vla_eval",
            "upstream_root",
            "vision_encoder_home",
            "harness_src",
            "robomme_src",
            "maniskill_src",
            "simulator_site",
        )
    }
    policy, workspace, _queue, _policy_tree = _workspace_probe_checkpoints(tmp_path)
    command = preflight._workspace_probe_command(
        source_root=tmp_path / "source",
        paths=paths,
        policy_checkpoint=policy,
        workspace_checkpoint=workspace,
        config=tmp_path / "probe.yaml",
        output=tmp_path / "output",
        probe_id="probe-id",
    )
    assert command[:3] == [
        str(paths["policy_python"]),
        "-m",
        "robomme_integration.eval.launch_gpu_fleet",
    ]
    assert command[command.index("--gpus") + 1] == "0"
    assert command[command.index("--simulator-gpus") + 1] == "1"
    assert command[command.index("--shards") + 1] == "2"
    assert command[command.index("--workspace-checkpoint") + 1] == str(workspace)
    assert "--native-simulator" in command
    assert "--pin-native-cpus" in command


def test_parallel_workspace_probe_commands_match_exact_disjoint_lane_topology(tmp_path):
    paths = {
        name: tmp_path / name
        for name in (
            "policy_python",
            "vla_eval",
            "upstream_root",
            "vision_encoder_home",
            "harness_src",
            "robomme_src",
            "maniskill_src",
            "simulator_site",
        )
    }
    policy, workspace, _queue, _policy_tree = _workspace_probe_checkpoints(tmp_path)
    topology = local.parallel_campaign.local_2x5090_topology()

    rendered = []
    for lane in topology.lanes:
        command = preflight._workspace_probe_command(
            source_root=tmp_path / "source",
            paths=paths,
            policy_checkpoint=policy,
            workspace_checkpoint=workspace,
            config=tmp_path / "probe.yaml",
            output=tmp_path / lane.lane_id,
            probe_id=f"probe-{lane.lane_id}",
            parallel_lane=lane,
        )

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        rendered.append(
            (
                value("--gpus"),
                value("--simulator-gpus"),
                value("--base-port"),
                value("--cpu-range"),
            )
        )
        assert value("--gpus") == str(lane.policy_gpu)
        assert value("--simulator-gpus") == str(lane.simulator_gpu)
        assert value("--base-port") == str(lane.port)
        assert value("--cpu-range") == lane.cpu_range
        assert value("--shards") == "4"
        assert value("--xla-memory-fraction") == str(lane.xla_memory_fraction)
        assert value("--native-shard-prewarm-seconds") == str(lane.shard_prewarm_seconds)
        assert value("--native-shard-stagger-seconds") == str(lane.shard_stagger_seconds)
        assert "--native-simulator" in command
        assert "--pin-native-cpus" in command

    assert rendered == [
        ("0", "0", "18100", "0-63"),
        ("1", "1", "18101", "64-127"),
    ]


def _stub_parallel_workspace_probe(monkeypatch, tmp_path):
    inputs = {
        "lineage": {"queue_id": local.LOCAL_PARALLEL_QUEUE_ID},
        "policy_checkpoint": {"path": str(tmp_path / "policy/19999")},
        "workspace_checkpoint": {"path": str(tmp_path / "workspace/10000")},
    }
    monkeypatch.setattr(
        preflight,
        "_workspace_probe_inputs",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        preflight,
        "_workspace_probe_command",
        lambda **kwargs: ["probe", kwargs["probe_id"]],
    )
    monkeypatch.setattr(preflight, "_environment", lambda *_args: {})
    return inputs


def _run_stub_parallel_workspace_probe(tmp_path):
    return preflight._run_workspace_action_probe(
        tmp_path / "root",
        tmp_path / "source",
        {},
        policy_checkpoint=tmp_path / "policy/19999",
        workspace_checkpoint=tmp_path / "workspace/10000",
        queue_template=tmp_path / "queue.json",
        policy_tree_manifest=tmp_path / "checkpoint-tree.json",
        source_sha256="a" * 64,
        runtime_fingerprint={},
    )


def test_parallel_workspace_probe_spawn_failure_cleans_already_started_lane(monkeypatch, tmp_path):
    _stub_parallel_workspace_probe(monkeypatch, tmp_path)

    class Process:
        pid = 41001
        returncode = None

        def poll(self):
            return self.returncode

    started = Process()
    popen_calls = []

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        if len(popen_calls) == 1:
            return started
        raise OSError("second lane spawn failed")

    terminated = []
    requested = []

    def request_termination(process):
        requested.append(process)
        return set()

    def terminate(process):
        terminated.append(process)
        process.returncode = -15
        return "", ""

    monkeypatch.setattr(preflight.subprocess, "Popen", popen)
    monkeypatch.setattr(preflight, "_request_probe_termination", request_termination)
    monkeypatch.setattr(preflight, "_terminate_probe_process", terminate)

    with pytest.raises(OSError, match="second lane spawn failed"):
        _run_stub_parallel_workspace_probe(tmp_path)

    assert len(popen_calls) == 2
    assert all(kwargs["start_new_session"] is True for _command, kwargs in popen_calls)
    assert requested == [started]
    assert terminated == [started]


@pytest.mark.parametrize("interrupt_kind", ["keyboard", "sigterm"])
def test_parallel_workspace_probe_interrupt_signals_peer_before_executor_wait(monkeypatch, tmp_path, interrupt_kind):
    _stub_parallel_workspace_probe(monkeypatch, tmp_path)
    events = []
    previous_sigterm = preflight.signal.getsignal(preflight.signal.SIGTERM)

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return self.returncode

    processes = [Process(42001), Process(42002)]
    pending_processes = list(processes)
    monkeypatch.setattr(
        preflight.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pending_processes.pop(0),
    )

    submitted = []

    class Future:
        def __init__(self, process):
            self.process = process

        def result(self):
            events.append(("owner-interrupted", self.process.pid))
            if interrupt_kind == "keyboard":
                self.process.returncode = -2
                raise KeyboardInterrupt
            handler = preflight.signal.getsignal(preflight.signal.SIGTERM)
            assert callable(handler)
            handler(preflight.signal.SIGTERM, None)
            raise AssertionError("the installed SIGTERM handler must interrupt the probe")

        def cancel(self):
            events.append(("future-cancel", self.process.pid))
            self._cancelled = True
            return True

        def cancelled(self):
            return getattr(self, "_cancelled", False)

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def submit(self, function, process, *, timeout_seconds):
            assert function is preflight._communicate_probe_process
            assert timeout_seconds == 1800
            future = Future(process)
            submitted.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures=False):
            events.append(("executor-shutdown", wait, cancel_futures))

    monkeypatch.setattr(preflight.concurrent.futures, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        preflight.concurrent.futures,
        "as_completed",
        lambda _futures: [submitted[0]],
    )

    def request_termination(process):
        events.append(("peer-sigterm", process.pid))
        process.returncode = -15
        return set()

    def terminate(process):
        events.append(("outer-terminate", process.pid))
        process.returncode = -15
        return "", ""

    monkeypatch.setattr(preflight, "_request_probe_termination", request_termination)
    monkeypatch.setattr(preflight, "_terminate_probe_process", terminate)

    expected_error = KeyboardInterrupt if interrupt_kind == "keyboard" else preflight.WorkspaceProbeSignal
    with pytest.raises(expected_error):
        _run_stub_parallel_workspace_probe(tmp_path)

    peer_signal_index = events.index(("peer-sigterm", 42002))
    first_shutdown_index = next(index for index, event in enumerate(events) if event[0] == "executor-shutdown")
    assert peer_signal_index < first_shutdown_index
    assert events[first_shutdown_index] == ("executor-shutdown", False, True)
    assert {event[1] for event in events if event[0] == "outer-terminate"} == {process.pid for process in processes}
    assert preflight.signal.getsignal(preflight.signal.SIGTERM) is previous_sigterm


def test_request_probe_termination_only_signals_process_group(monkeypatch):
    class Process:
        pid = 43001
        returncode = None

        def poll(self):
            return None

        def communicate(self, *_args, **_kwargs):
            raise AssertionError("peer signalling must not concurrently call communicate")

    signals = []
    monkeypatch.setattr(
        preflight,
        "_descendant_process_groups",
        lambda pid: ({43101, 43102} if pid == 43001 else set()),
    )
    monkeypatch.setattr(
        preflight.os,
        "killpg",
        lambda process_group, sent_signal: signals.append((process_group, sent_signal)),
    )
    groups = preflight._request_probe_termination(Process())
    assert signals == [(43001, preflight.signal.SIGTERM)]
    assert groups == {43101, 43102}


def test_workspace_probe_timeout_and_operator_interrupt_both_trigger_teardown(monkeypatch):
    class Process:
        def __init__(self, error):
            self.error = error

        def communicate(self, *, timeout):
            raise self.error

    stopped = []
    monkeypatch.setattr(
        preflight,
        "_terminate_probe_process",
        lambda process: (stopped.append(process) or ("partial-out", "partial-err")),
    )
    timeout = Process(subprocess.TimeoutExpired(["probe"], 1))
    assert preflight._communicate_probe_process(timeout, timeout_seconds=1) == (
        "partial-out",
        "partial-err",
        True,
    )
    interrupted = Process(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        preflight._communicate_probe_process(interrupted, timeout_seconds=1)
    assert stopped == [timeout, interrupted]

    def broken_cleanup(_process):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(preflight, "_terminate_probe_process", broken_cleanup)
    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        preflight._communicate_probe_process(Process(original), timeout_seconds=1)
    assert caught.value is original
    assert caught.value.__notes__ == ["workspace probe cleanup also failed: cleanup failed"]


def test_workspace_action_probe_audit_requires_real_actions_and_zero_harness_errors(tmp_path):
    policy, workspace, _queue, _policy_tree = _workspace_probe_checkpoints(tmp_path)
    output = tmp_path / "output"
    evaluation = output / "eval"
    evaluation.mkdir(parents=True)
    (output / "COMPLETED").write_text("done\n", encoding="utf-8")
    probe_id = "probe-id"
    supervisor = {
        "launcher_returncode": 0,
        "failure": None,
        "arm": "q3",
        "task": "PickXtimes",
        "checkpoint": str(policy.resolve()),
        "gpus": [0],
        "ports": [18200],
        "shards": 2,
        "server_commands": [
            [
                "server",
                "--args.checkpoint",
                str(policy.resolve()),
                "--args.workspace_checkpoint",
                str(workspace.resolve()),
            ]
        ],
    }
    (output / "supervisor.json").write_text(json.dumps(supervisor), encoding="utf-8")
    result = evaluation / "workspace_probe_aggregate.json"
    result.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "PickXtimes",
                        "episodes": [
                            {
                                "episode_id": 0,
                                "episode_idx": 0,
                                "steps": 1,
                                "metrics": {"success": False},
                            },
                            {
                                "episode_id": 1,
                                "episode_idx": 1,
                                "steps": 1,
                                "metrics": {"success": False},
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "eval_id": probe_id,
        "config_sha256": hashlib.sha256(preflight.WORKSPACE_PROBE_CONFIG).hexdigest(),
        "shards": 2,
        "recording_mode": "sqlite",
        "server_urls": ["ws://127.0.0.1:18200"],
        "returncodes": [0, 0],
        "native": {"gpus": [1], "gpu_by_shard": [1, 1]},
        "commands": [
            ["vla-eval", "--server-url", "ws://127.0.0.1:18200"],
            ["vla-eval", "--server-url", "ws://127.0.0.1:18200"],
        ],
        "episode_audit": {"episodes": 2, "harness_failures": []},
        "materialized_results": [
            {"path": str(result), "sha256": preflight._sha(result), "bytes": result.stat().st_size}
        ],
    }
    (evaluation / "launch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "server-gpu0-port18200.log").write_text(
        "Loaded arm=q3\nStarting server on ws://127.0.0.1:18200\ninference arm=q3 batch_call=1 rows=2\n",
        encoding="utf-8",
    )

    audit = preflight._audit_workspace_action_probe(
        output,
        policy_checkpoint=policy,
        workspace_checkpoint=workspace,
        probe_id=probe_id,
    )
    assert audit["actions_executed"] == 2
    assert audit["concurrent_native_shards"] == 2
    assert audit["load_completed_before_readiness"] is True

    manifest["episode_audit"]["harness_failures"] = [{"reason": "exception"}]
    (evaluation / "launch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="harness contract failed"):
        preflight._audit_workspace_action_probe(
            output,
            policy_checkpoint=policy,
            workspace_checkpoint=workspace,
            probe_id=probe_id,
        )


def test_distribution_inventory_rejects_duplicate_normalized_names(monkeypatch, tmp_path):
    duplicate = [
        {"name": "same-name", "version": "1", "direct_url": None},
        {"name": "same-name", "version": "2", "direct_url": None},
    ]
    monkeypatch.setattr(
        local.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(duplicate), stderr=""
        ),
    )
    with pytest.raises(ValueError, match="duplicate or unsorted normalized names"):
        local._distribution_inventory(tmp_path / "python")


def test_runtime_contract_preserves_virtual_environment_executable_symlink(tmp_path):
    target = tmp_path / "base/python3.11"
    target.parent.mkdir()
    target.write_text("interpreter\n", encoding="utf-8")
    launcher = tmp_path / "openpi/.venv/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(target)
    assert local._contract_path("policy_python", launcher) == launcher.absolute()
    assert local._contract_path("policy_python", launcher) != launcher.resolve()
    assert local._contract_path("openpi_src", launcher) == target.resolve()


def test_python_identity_executes_through_lexical_virtual_environment(tmp_path):
    environment = tmp_path / "sealed/.venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin/python"
    identity = local._python_identity(python)
    assert identity["path"] == str(python)
    assert identity["runtime"]["executable"] == str(python)
    assert identity["runtime"]["prefix"] == str(environment)
    assert Path(identity["runtime"]["purelib"]).is_relative_to(environment)
