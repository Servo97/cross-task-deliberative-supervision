from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

from robomme_integration.eval import campaign


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gates(tmp_path: Path) -> tuple[Path, Path, dict]:
    runtime_artifact = {"uri": "s3://bucket/runtime.tgz", "sha256": "1" * 64}
    openpi = {"uri": f"s3://bucket/openpi/{'2' * 64}.tgz", "sha256": "2" * 64}
    preflight = {
        "schema_version": 1,
        "kind": campaign.PREFLIGHT_KIND,
        "preflight_id": "p5-native-eval-v1-test",
        "runtime": runtime_artifact,
        "openpi": openpi,
        "vla_eval_entrypoint": {
            "kind": "python_module_wrapper",
            "module": "vla_eval.cli.main",
        },
        "image": {"uri": "image", "sha256": "3" * 64},
        "probe": {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
        },
        "source_tree_sha256": "4" * 64,
        "claim_s3": "s3://bucket/preflight.json",
        "infrastructure": {"instance_type": "ml.p5.48xlarge", "accelerator": "8xH100"},
    }
    preflight = campaign.seal_document(preflight, field="manifest_sha256")
    preflight["status"] = "native_render_reset_passed"
    preflight_path = tmp_path / "preflight.json"
    preflight_sha = _write_json(preflight_path, preflight)

    runtime_root = tmp_path / "runtime"
    paths = {
        "policy_python": runtime_root / "openpi/.venv/bin/python",
        "vla_eval": runtime_root / "eval/bin/vla-eval",
        "harness_src": runtime_root / "harness/src",
        "robomme_src": runtime_root / "robomme/src",
        "maniskill_src": runtime_root / "maniskill",
        "openpi_src": runtime_root / "openpi/src",
        "policy_site": runtime_root / "openpi/site-packages",
        "simulator_site": runtime_root / "eval/site-packages",
        "upstream_root": runtime_root / "upstream",
        "vision_encoder_home": runtime_root / "vision",
    }
    for name, path in paths.items():
        if name in {"policy_python", "vla_eval"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)
        else:
            path.mkdir(parents=True, exist_ok=True)
    vision = paths["vision_encoder_home"] / "pi05_vision_encoder/siglip_params.pkl"
    vision.parent.mkdir(parents=True)
    vision.write_bytes(b"pinned")
    receipt = campaign.seal_document(
        {
            "schema_version": 1,
            "kind": campaign.RUNTIME_KIND,
            "status": "staged_and_verified",
            "preflight_claim_sha256": preflight_sha,
            "runtime": runtime_artifact,
            "openpi": openpi,
            "vla_eval_wrapper": {
                "kind": "python_module_wrapper",
                "module": "vla_eval.cli.main",
                "sha256": hashlib.sha256(paths["vla_eval"].read_bytes()).hexdigest(),
            },
            "paths": {name: str(path) for name, path in paths.items()},
            "render_environment": {
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "ROBOMME_USE_LAVAPIPE": "0",
            },
        },
        field="receipt_sha256",
    )
    receipt_path = tmp_path / "runtime-receipt.json"
    receipt_file_sha = _write_json(receipt_path, receipt)
    gates = {
        "native_preflight": {
            "preflight_id": preflight["preflight_id"],
            "claim_sha256": preflight_sha,
            "source_tree_sha256": preflight["source_tree_sha256"],
        },
        "runtime_receipt": {
            "receipt_sha256": receipt_file_sha,
            "runtime_artifact_sha256": runtime_artifact["sha256"],
            "openpi_sha256": openpi["sha256"],
        },
    }
    return preflight_path, receipt_path, gates


def _cell(source: Path, queue_id: str, ordinal: int, arm: str, *, workspace: bool = False) -> dict:
    task = "PickXtimes"
    run_id = f"st-v1-pickxtimes-{arm}-seed0-{'a' * 16}"
    config = source / "robomme_integration/eval/configs/pickxtimes.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("benchmark: fixed50\n", encoding="utf-8")
    publish = f"s3://bucket/eval-campaigns/{queue_id}"
    value = {
        "ordinal": ordinal,
        "cell_id": f"{ordinal:03d}-pickxtimes-{arm}",
        "task": task,
        "arm": arm,
        "run_id": run_id,
        "final_step": 19_999,
        "scientific_spec_sha256": "5" * 64,
        "run_manifest_sha256": "6" * 64,
        "training_openpi": {"uri": f"s3://bucket/openpi/{'2' * 64}.tgz", "sha256": "2" * 64},
        "training_run_manifest_s3": (f"s3://bucket/manifests/runs/train/{run_id}/{run_id}-attempt1.json"),
        "training_output_s3": f"s3://bucket/checkpoints/{task}/{arm}/seed0/{run_id}",
        "training_completion_claim_s3": f"s3://bucket/manifests/claims/train/{run_id}/step-19999.complete.json",
        "training_completion_binding": campaign.TRAINING_COMPLETION_CURRENT,
        "benchmark_config": "robomme_integration/eval/configs/pickxtimes.yaml",
        "benchmark_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "training_nuisance": {
            "data_parent_inventory_sha256": "a" * 64,
            "data_task_inventory_sha256": "b" * 64,
            "initialization_inventory_sha256": "c" * 64,
            "initialization_checkpoint_s3": "s3://bucket/init/149999",
            "seed": 0,
            "steps": 20_000,
            "action_horizon": 20,
            "window_len": 8 if arm in {"q0", "q1", "q2", "q3"} else None,
            "chunk_stride": 10 if arm in {"q0", "q1", "q2", "q3"} else None,
        },
        "eval_id": f"{run_id}-fixed50-{queue_id}",
        "result_claim_s3": f"{publish}/cells/{ordinal:03d}-pickxtimes-{arm}/result.complete.json",
        "cfg_guidance_scale": 1.0,
        "workspace": None,
        "ptrm": None,
    }
    if workspace:
        value["workspace"] = {
            "provenance_mode": campaign.WORKSPACE_PROVENANCE_CLAIM,
            "claim_s3": f"s3://bucket/workspace/{task}/claim.json",
            "claim_sha256": "7" * 64,
            "encoder_id": "8" * 64,
            "representation_s3": f"s3://bucket/workspace/{task}/representation/step-10000",
            "completion_sha256": "9" * 64,
            "step": 10_000,
        }
    if arm == "ptrm":
        value["ptrm"] = {"eval_k": 1, "eval_sigma": 0.0, "eval_select": "q"}
    return value


def _queue(source: Path, gates: dict, arms: tuple[str, ...]) -> dict:
    queue_id = "fixed50-two-day-v1"
    publish = f"s3://bucket/eval-campaigns/{queue_id}"
    cells = [
        _cell(source, queue_id, ordinal, arm, workspace=arm in campaign.WORKSPACE_EVAL_ARMS)
        for ordinal, arm in enumerate(arms)
    ]
    config = cells[0]["benchmark_config"]
    config_sha = cells[0]["benchmark_config_sha256"]
    common_nuisance = {
        key: value for key, value in cells[0]["training_nuisance"].items() if key not in {"window_len", "chunk_stride"}
    }
    value = {
        "schema_version": 1,
        "kind": campaign.QUEUE_KIND,
        "queue_id": queue_id,
        "publish_root_s3": publish,
        "claims": {"manifest": f"{publish}/manifest.json", "completion": f"{publish}/complete.json"},
        "gates": gates,
        "topology": {
            "policy_gpus": [0, 1, 2, 3],
            "simulator_gpus": [4, 5, 6, 7],
            "simulator_shards": 16,
            "cpu_range": "0-191",
            "base_port": 18100,
            "xla_memory_fraction": 0.65,
        },
        "retry": {"classifier_version": campaign.CLASSIFIER_VERSION, "max_attempts": 2},
        "limits": {
            "max_run_seconds": 86_400,
            "runtime_reserve_seconds": 1_800,
            "estimated_cell_seconds": 1_800,
            "minimum_free_bytes": 1,
        },
        "comparability": {
            "serving_openpi": cells[0]["training_openpi"],
            "task_benchmark_configs": {
                "PickXtimes": {"path": config, "sha256": config_sha},
            },
            "task_common_training_nuisance": {"PickXtimes": common_nuisance},
            "sequence_geometry_policy": "manifest_verified_per_cell_not_assumed_common",
        },
        "cells": cells,
    }
    return campaign.seal_document(value, field="queue_manifest_sha256")


def test_v4_cfg_four_scales_share_one_checkpoint_but_have_distinct_eval_identities(tmp_path):
    _preflight, _receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("v4_wsm_cfg",))
    base = queue["cells"][0]
    old_run = base["run_id"]
    new_run = old_run.replace("st-v1-", "st-v4-", 1)
    cells = []
    for ordinal, scale in enumerate((0.5, 1.0, 1.5, 2.0)):
        token = f"cfgs{int(scale * 100):03d}"
        cell = deepcopy(base)
        cell["ordinal"] = ordinal
        cell["cell_id"] = f"{ordinal:03d}-pickxtimes-v4_wsm_cfg-{token}"
        cell["run_id"] = new_run
        for field in (
            "training_run_manifest_s3",
            "training_output_s3",
            "training_completion_claim_s3",
        ):
            cell[field] = cell[field].replace(old_run, new_run)
        cell["cfg_guidance_scale"] = scale
        cell["eval_id"] = f"{new_run}-fixed50-{queue['queue_id']}-{token}"
        cell["result_claim_s3"] = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/result.complete.json"
        cells.append(cell)
    queue["cells"] = cells
    queue.pop("queue_manifest_sha256")
    queue = campaign.seal_document(queue, field="queue_manifest_sha256")
    campaign.validate_queue(queue, source_root=source)

    invalid = deepcopy(queue)
    invalid["cells"][0]["cell_id"] = "000-pickxtimes-v4_wsm_cfg"
    invalid.pop("queue_manifest_sha256")
    invalid = campaign.seal_document(invalid, field="queue_manifest_sha256")
    with pytest.raises(ValueError, match="cell_id does not bind guidance scale"):
        campaign.validate_queue(invalid, source_root=source)


def test_queue_and_runtime_are_exactly_sealed_and_fail_closed(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("ptrm",))
    campaign.validate_queue(queue, source_root=source)
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    assert runtime.policy_python.is_file()
    assert runtime.preflight_claim_sha256 == gates["native_preflight"]["claim_sha256"]

    wrapper = runtime.vla_eval
    original_wrapper = wrapper.read_bytes()
    wrapper.write_bytes(original_wrapper + b"# drift\n")
    with pytest.raises(ValueError, match="wrapper digest mismatch"):
        campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    wrapper.write_bytes(original_wrapper)

    tampered = dict(queue)
    tampered["retry"] = {"classifier_version": campaign.CLASSIFIER_VERSION, "max_attempts": 3}
    with pytest.raises(ValueError, match="seal mismatch"):
        campaign.validate_queue(tampered, source_root=source)

    preflight.write_text(preflight.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="claim file digest"):
        campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)


def test_ptrm_launch_routes_workspace_and_exact_inference_knobs(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("ptrm",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    cell = queue["cells"][0]
    command = campaign.build_launch_command(
        queue,
        cell,
        source_root=source,
        runtime=runtime,
        checkpoint=tmp_path / "checkpoint",
        workspace=tmp_path / "workspace",
        output=tmp_path / "output",
    )
    assert command[command.index("--workspace-checkpoint") + 1] == str(tmp_path / "workspace")
    assert command[command.index("--upstream-root") + 1] == str(runtime.upstream_root)
    assert command[command.index("--ptrm-eval-k") + 1] == "1"
    assert command[command.index("--ptrm-eval-sigma") + 1] == "0.0"
    assert command[command.index("--ptrm-eval-select") + 1] == "q"
    assert command.count("--simulator-pythonpath") == 5
    simulator_paths = [command[index + 1] for index, value in enumerate(command) if value == "--simulator-pythonpath"]
    assert simulator_paths == [
        str(source),
        str(runtime.harness_src),
        str(runtime.robomme_src),
        str(runtime.maniskill_src),
        str(runtime.simulator_site),
    ]
    assert str(runtime.openpi_src) not in simulator_paths
    assert str(runtime.policy_site) not in simulator_paths
    assert str(runtime.upstream_root / "src") not in simulator_paths
    assert "--native-simulator" in command and "--pin-native-cpus" in command


def test_scored_policy_environment_exactly_matches_preflight_order(monkeypatch, tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    observed = {}

    class FakeProcess:
        pid = 1234

        @staticmethod
        def wait(*, timeout=None):
            assert timeout is None
            return 0

    def fake_popen(command, *, cwd, env, start_new_session):
        observed.update(
            command=command,
            cwd=cwd,
            env=env,
            start_new_session=start_new_session,
        )
        return FakeProcess()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    result = campaign.SubprocessEvaluator(source, runtime).run(
        queue,
        queue["cells"][0],
        checkpoint=tmp_path / "checkpoint",
        workspace=None,
        output=tmp_path / "output",
    )
    assert result == 0
    assert observed["start_new_session"] is True
    assert observed["env"]["PYTHONPATH"].split(os.pathsep) == [
        str(path)
        for path in campaign.policy_pythonpath_entries(
            source_root=source,
            harness_src=runtime.harness_src,
            openpi_src=runtime.openpi_src,
            policy_site=runtime.policy_site,
            upstream_root=runtime.upstream_root,
            simulator_site=runtime.simulator_site,
        )
    ]


def test_scored_evaluator_interrupt_drains_process_group_and_reraises(monkeypatch, tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    observed = {"drained": False}

    class InterruptingProcess:
        pid = 4321

        @staticmethod
        def wait(*, timeout=None):
            assert timeout is None
            raise KeyboardInterrupt

    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *_args, **kwargs: (
            observed.update(start_new_session=kwargs["start_new_session"]),
            InterruptingProcess(),
        )[1],
    )

    def failed_cleanup(process):
        observed.update(drained=process.pid == 4321)
        raise RuntimeError("synthetic drain failure")

    monkeypatch.setattr(campaign, "_terminate_evaluator_process", failed_cleanup)

    with pytest.raises(campaign.EvaluatorCleanupFailure) as captured:
        campaign.SubprocessEvaluator(source, runtime).run(
            queue,
            queue["cells"][0],
            checkpoint=tmp_path / "checkpoint",
            workspace=None,
            output=tmp_path / "output",
        )
    assert observed == {"drained": True, "start_new_session": True}
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "synthetic drain failure" in captured.value.__notes__[0]


class MemoryStore:
    def __init__(self):
        self.values: dict[str, bytes] = {}

    def read_bytes(self, uri: str) -> bytes | None:
        return self.values.get(uri)

    def exists(self, uri: str) -> bool:
        return uri in self.values

    def put_bytes_once(self, payload: bytes, uri: str) -> None:
        existing = self.values.get(uri)
        if existing is not None and existing != payload:
            raise RuntimeError(f"collision {uri}")
        self.values[uri] = payload

    def put_file_once(self, path: Path, uri: str) -> None:
        self.put_bytes_once(path.read_bytes(), uri)


def _training_manifest(cell: dict) -> dict:
    nuisance = cell["training_nuisance"]
    scientific = {
        "scope": "single_task_v1",
        "task": {"name": cell["task"]},
        "arm": cell["arm"],
        "sources": {"openpi": cell["training_openpi"]},
        "initialization": {
            "checkpoint_s3": nuisance["initialization_checkpoint_s3"],
            "inventory_sha256": nuisance["initialization_inventory_sha256"],
        },
        "data": {
            "parent_inventory_sha256": nuisance["data_parent_inventory_sha256"],
            "derived_task_inventory_sha256": nuisance["data_task_inventory_sha256"],
        },
        "training": {
            "seed": nuisance["seed"],
            "steps": nuisance["steps"],
            "action_horizon": nuisance["action_horizon"],
            "window_len": nuisance["window_len"],
            "chunk_stride": nuisance["chunk_stride"],
        },
    }
    cell["scientific_spec_sha256"] = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attempt_id = f"{cell['run_id']}-attempt1"
    manifest = {
        "schema_version": 2,
        "kind": "robomme_gpu_training_attempt",
        "run_id": cell["run_id"],
        "attempt_id": attempt_id,
        "manifest_s3": cell["training_run_manifest_s3"],
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
        "scientific": scientific,
        "output_s3": cell["training_output_s3"],
        "checkpoint_tree_manifest_root": (f"s3://bucket/manifests/artifacts/checkpoints/{cell['run_id']}/step-19999"),
        "claims": {"completion": cell["training_completion_claim_s3"]},
    }
    manifest = campaign.seal_document(manifest, field="manifest_sha256")
    cell["run_manifest_sha256"] = manifest["manifest_sha256"]
    return manifest


def test_training_manifest_is_opened_and_nuisance_recipe_is_verified(tmp_path):
    _preflight, _receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q0",))
    cell = queue["cells"][0]
    store = MemoryStore()
    manifest = _training_manifest(cell)
    store.values[cell["training_run_manifest_s3"]] = campaign._canonical(manifest)
    stager = campaign.AwsStager(store)
    stager._verify_training_manifest(cell)

    manifest["scientific"]["training"]["chunk_stride"] = 1
    manifest = campaign.seal_document(manifest, field="manifest_sha256")
    cell["run_manifest_sha256"] = manifest["manifest_sha256"]
    cell["scientific_spec_sha256"] = hashlib.sha256(
        json.dumps(manifest["scientific"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["scientific_spec_sha256"] = cell["scientific_spec_sha256"]
    manifest = campaign.seal_document(manifest, field="manifest_sha256")
    cell["run_manifest_sha256"] = manifest["manifest_sha256"]
    store.values[cell["training_run_manifest_s3"]] = campaign._canonical(manifest)
    with pytest.raises(ValueError, match="nuisance recipe drift"):
        stager._verify_training_manifest(cell)


def test_runtime_stager_accepts_only_manifest_bound_legacy_receipt_chain(tmp_path):
    _preflight, _receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q0",))
    cell = queue["cells"][0]
    cell["training_completion_binding"] = campaign.TRAINING_COMPLETION_LEGACY
    manifest = _training_manifest(cell)
    checkpoint_uri = f"{cell['training_output_s3']}/deploy/19999"
    checkpoint_payload = b"x"
    tree = {
        "schema_version": 1,
        "checkpoint_uri": checkpoint_uri,
        "objects": [
            {
                "key": "params/mock",
                "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
                "size_bytes": len(checkpoint_payload),
            }
        ],
    }
    tree_bytes = campaign._canonical(tree)
    tree_sha = hashlib.sha256(tree_bytes).hexdigest()
    tree_uri = f"{manifest['checkpoint_tree_manifest_root']}/{tree_sha}.json"
    expected = {
        "schema_version": 1,
        "run_id": cell["run_id"],
        "attempt_id": manifest["attempt_id"],
        "step": 19_999,
        "checkpoint_uri": checkpoint_uri,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "tree_manifest_sha256": tree_sha,
    }
    completion = {
        **expected,
        "kind": "robomme_gpu_checkpoint_complete",
        "tree_manifest_uri": tree_uri,
    }
    deploy = {**expected, "kind": "robomme_gpu_deploy_checkpoint_complete"}
    store = MemoryStore()
    store.values[cell["training_run_manifest_s3"]] = campaign._canonical(manifest)
    store.values[cell["training_completion_claim_s3"]] = campaign._canonical(completion)
    store.values[tree_uri] = tree_bytes
    store.values[f"{checkpoint_uri}/_DEPLOY_COMPLETE.json"] = campaign._canonical(deploy)

    class LocalStager(campaign.AwsStager):
        def _sync(self, uri: str, destination: Path, *, checkpoint: bool = False) -> None:
            assert uri == checkpoint_uri and checkpoint
            (destination / "params").mkdir(parents=True)
            (destination / "assets").mkdir()
            (destination / "params/mock").write_bytes(checkpoint_payload)

    destination = tmp_path / "stage/checkpoint/19999"
    assert LocalStager(store).stage_checkpoint(cell, destination) == checkpoint_uri

    completion["scientific_spec_sha256"] = cell["scientific_spec_sha256"]
    store.values[cell["training_completion_claim_s3"]] = campaign._canonical(completion)
    with pytest.raises(ValueError, match="completion-binding mode drift"):
        LocalStager(store).stage_checkpoint(cell, tmp_path / "stage2/checkpoint/19999")


def test_legacy_workspace_queue_and_runtime_verify_omega_to_checkpoint_tree(tmp_path):
    _preflight, _receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q3",))
    cell = queue["cells"][0]
    task_manifest_sha = "d" * 64
    representation = tmp_path / "legacy-representation"
    representation.mkdir()
    run_config = {
        "schema_version": 1,
        "task": cell["task"],
        "task_manifest_sha256": task_manifest_sha,
        "steps": 10_000,
        "seed": 0,
        "omega_dim": 512,
    }
    run_config_path = representation / "WSM_RUN_CONFIG.json"
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    run_config_sha = hashlib.sha256(run_config_path.read_bytes()).hexdigest()
    best = {
        "best_score": 0.1,
        "best_step": 10_000,
        "latest_step": 10_000,
        "run_config_sha256": run_config_sha,
    }
    best_path = representation / "WSM_BEST.json"
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    best_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    completion = {
        "schema_version": 1,
        "step": 10_000,
        "run_config_sha256": run_config_sha,
        "embedded_sha256": {
            "WSM_BEST.json": best_sha,
            "WSM_RUN_CONFIG.json": run_config_sha,
        },
    }
    completion_path = representation / "WSM_GENERATION_COMPLETE.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    (representation / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    (representation / "state").mkdir()
    (representation / "state/mock").write_bytes(b"workspace-state")
    (representation / "_UPLOAD_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "step": 10_000,
                "completed_at": "2026-08-04T01:26:04+00:00",
                "source_marker": "_CHECKPOINT_METADATA",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tree_sha = campaign._legacy_workspace_tree_sha256(representation, expected_step=10_000)
    materializer_sha = "e" * 64
    identity = {
        "schema_version": 1,
        "run_config_sha256": run_config_sha,
        "checkpoint_step": 10_000,
        "checkpoint_tree_sha256": tree_sha,
        "materializer_sha256": materializer_sha,
    }
    encoder_id = hashlib.sha256(
        (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    omega = {
        "schema_version": 1,
        "task_name": cell["task"],
        "task_manifest_sha256": task_manifest_sha,
        "encoder_id": encoder_id,
        "encoder_identity": identity,
        "representation": {"uses_labels_at_inference": False},
    }
    omega["manifest_sha256"] = hashlib.sha256(json.dumps(omega, indent=2, sort_keys=True).encode() + b"\n").hexdigest()
    omega_payload = json.dumps(omega, indent=2, sort_keys=True).encode() + b"\n"
    artifact_root = f"s3://bucket/artifacts/robomme/workspace/{cell['task']}/{encoder_id}"
    workspace = {
        "provenance_mode": campaign.WORKSPACE_PROVENANCE_LEGACY,
        "encoder_id": encoder_id,
        "omega_manifest_s3": f"{artifact_root}/omega/MANIFEST.json",
        "omega_manifest_sha256": hashlib.sha256(omega_payload).hexdigest(),
        "task_manifest_sha256": task_manifest_sha,
        "representation_s3": f"{artifact_root}/representation/step-10000",
        "completion_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "step": 10_000,
        "checkpoint_tree_sha256": tree_sha,
        "run_config_sha256": run_config_sha,
        "best_sha256": best_sha,
        "materializer_sha256": materializer_sha,
    }
    cell["workspace"] = workspace
    queue = campaign.seal_document(queue, field="queue_manifest_sha256")
    campaign.validate_queue(queue, source_root=source)
    store = MemoryStore()
    store.values[workspace["omega_manifest_s3"]] = omega_payload

    class LocalWorkspaceStager(campaign.AwsStager):
        def _sync(self, uri: str, destination: Path, *, checkpoint: bool = False) -> None:
            assert uri == workspace["representation_s3"] and not checkpoint
            shutil.copytree(representation, destination)

    destination = tmp_path / "staged-workspace"
    numeric_destination = destination / "10000"
    assert LocalWorkspaceStager(store).stage_workspace(cell, destination) == numeric_destination
    assert (numeric_destination / "WSM_GENERATION_COMPLETE.json").is_file()
    # The post-sync sentinel is intentionally outside the producer's tree hash, but its exact
    # schema and step are still authenticated before accepting the tree.
    upload = json.loads((representation / "_UPLOAD_COMPLETE.json").read_text())
    upload["step"] = 9_999
    (representation / "_UPLOAD_COMPLETE.json").write_text(json.dumps(upload) + "\n")
    with pytest.raises(ValueError, match="upload marker identity drift"):
        LocalWorkspaceStager(store).stage_workspace(cell, tmp_path / "bad-upload-workspace")
    upload["step"] = 10_000
    (representation / "_UPLOAD_COMPLETE.json").write_text(json.dumps(upload) + "\n")
    (representation / "state/mock").write_bytes(b"tampered-state")
    with pytest.raises(ValueError, match="checkpoint tree identity mismatch"):
        LocalWorkspaceStager(store).stage_workspace(cell, tmp_path / "tampered-workspace")

    best["best_score"] = float("inf")
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    workspace["best_sha256"] = hashlib.sha256(best_path.read_bytes()).hexdigest()
    completion["embedded_sha256"]["WSM_BEST.json"] = workspace["best_sha256"]
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    workspace["completion_sha256"] = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="best marker identity drift"):
        LocalWorkspaceStager(store).stage_workspace(cell, tmp_path / "nonfinite-best-workspace")


class FakeStager:
    def __init__(self):
        self.cells: list[str] = []
        self.workspace_destinations: dict[str, Path] = {}

    def stage_checkpoint(self, cell: dict, destination: Path) -> str:
        self.cells.append(cell["cell_id"])
        (destination / "params").mkdir(parents=True)
        (destination / "assets").mkdir()
        return f"{cell['training_output_s3']}/deploy/19999"

    def stage_workspace(self, cell: dict, destination: Path) -> Path | None:
        if cell["workspace"] is None:
            return None
        destination = destination / str(cell["workspace"]["step"])
        destination.mkdir(parents=True)
        self.workspace_destinations[cell["cell_id"]] = destination
        return destination


class ScriptedEvaluator:
    def __init__(self, outcomes: dict[str, list[str]]):
        self.outcomes = {key: list(value) for key, value in outcomes.items()}
        self.calls: dict[str, int] = {}
        self.workspaces: dict[str, list[Path | None]] = {}

    def run(self, queue, cell, *, checkpoint, workspace, output) -> int:
        self.calls[cell["cell_id"]] = self.calls.get(cell["cell_id"], 0) + 1
        self.workspaces.setdefault(cell["cell_id"], []).append(workspace)
        output.mkdir(parents=True)
        outcome = self.outcomes[cell["cell_id"]].pop(0)
        (output / "shard.log").write_text(outcome + "\n", encoding="utf-8")
        return 0 if outcome == "success" else 1


class FakeArtifacts:
    def publish_success(self, queue, cell, *, output, cell_work, checkpoint_uri, store, runtime):
        archive = cell_work / "success.tgz"
        archive.write_bytes(b"success-evidence")
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive_uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/evidence/{archive_sha}.tgz"
        store.put_file_once(archive, archive_uri)
        result = {
            "schema_version": 2,
            "kind": "robomme_fixed50_complete",
            "queue_id": queue["queue_id"],
            "queue_manifest_sha256": queue["queue_manifest_sha256"],
            "cell_id": cell["cell_id"],
            "run_id": cell["run_id"],
            "task": cell["task"],
            "arm": cell["arm"],
            "eval_id": cell["eval_id"],
            "episodes": 50,
            "successes": 17,
            "scientific_spec_sha256": cell["scientific_spec_sha256"],
            "run_manifest_sha256": cell["run_manifest_sha256"],
            "benchmark_config_sha256": cell["benchmark_config_sha256"],
            "native_preflight_claim_sha256": runtime.preflight_claim_sha256,
            "runtime_receipt_sha256": runtime.receipt_sha256,
            "evidence_archive_sha256": archive_sha,
            "evidence_archive_uri": archive_uri,
        }
        store.put_bytes_once(campaign._canonical(result), cell["result_claim_s3"])
        return result

    def publish_failure(self, queue, cell, *, cell_work, failure, store):
        evidence = cell_work / "failure.tgz"
        evidence.write_bytes(json.dumps(failure, sort_keys=True).encode())
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/failures/{digest}.tgz"
        claim_uri = f"{queue['publish_root_s3']}/cells/{cell['cell_id']}/failure.complete.json"
        store.put_file_once(evidence, uri)
        payload = {
            "schema_version": 1,
            "kind": campaign.CELL_FAILURE_KIND,
            "queue_id": queue["queue_id"],
            "queue_manifest_sha256": queue["queue_manifest_sha256"],
            "cell_id": cell["cell_id"],
            "run_id": cell["run_id"],
            "task": cell["task"],
            "arm": cell["arm"],
            **failure,
            "evidence_archive_sha256": digest,
            "evidence_archive_uri": uri,
        }
        store.put_bytes_once(campaign._canonical(payload), claim_uri)
        return {**payload, "failure_claim_s3": claim_uri}


def test_runner_retries_only_allowlisted_failures_and_passes_numeric_workspace(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q3", "s0"))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    store = MemoryStore()
    stager = FakeStager()
    first, second = (cell["cell_id"] for cell in queue["cells"])
    evaluator = ScriptedEvaluator(
        {
            first: ["VK_ERROR_DEVICE_LOST", "success"],
            second: ["success"],
        }
    )
    work = tmp_path / "work"
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=source,
        work_root=work,
        runtime=runtime,
        store=store,
        stager=stager,
        evaluator=evaluator,
        artifacts=FakeArtifacts(),
    )
    assert runner.run() == 0
    assert evaluator.calls == {first: 2, second: 1}
    expected_workspace = work / "cells" / f"cell-000-{first}" / "workspace" / "10000"
    assert stager.workspace_destinations == {first: expected_workspace}
    assert evaluator.workspaces == {first: [expected_workspace, expected_workspace], second: [None]}
    assert not list((work / "cells").iterdir())
    completion = json.loads(store.values[queue["claims"]["completion"]])
    assert completion["status"] == "complete"
    assert [record["status"] for record in completion["records"]] == ["complete", "complete"]
    assert completion["records"][0]["attempts"][0]["failure_class"] == "vulkan_device_lost"
    assert json.loads(runner.campaign_state_path.read_text())["status"] == "complete"
    assert all((work / "state/cells" / f"{cell['cell_id']}.json").is_file() for cell in queue["cells"])


def test_runner_halts_after_harness_failure_before_staging_next_cell(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("s0", "q0"))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    store = MemoryStore()
    stager = FakeStager()
    first, second = (cell["cell_id"] for cell in queue["cells"])
    evaluator = ScriptedEvaluator(
        {
            first: ["merged results contain 50 harness/environment failures"],
            second: ["should-not-run"],
        }
    )
    work = tmp_path / "work"
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=source,
        work_root=work,
        runtime=runtime,
        store=store,
        stager=stager,
        evaluator=evaluator,
        artifacts=FakeArtifacts(),
    )

    assert runner.run() == 2
    assert stager.cells == [first]
    assert evaluator.calls == {first: 1}
    assert queue["claims"]["completion"] not in store.values
    assert not list((work / "cells").iterdir())
    state = json.loads(runner.campaign_state_path.read_text())
    assert state["status"] == "halted_terminal_failure"
    assert [record["cell_id"] for record in state["records"]] == [first]
    assert state["records"][0]["status"] == "terminal_failure"
    assert (work / "state/cells" / f"{first}.json").is_file()
    assert not (work / "state/cells" / f"{second}.json").exists()


def test_operator_interrupt_does_not_publish_terminal_claim_or_stage_next_cell(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("s0", "q0"))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    store = MemoryStore()
    stager = FakeStager()
    first, second = (cell["cell_id"] for cell in queue["cells"])

    class InterruptingEvaluator:
        def __init__(self):
            self.calls: list[str] = []

        def run(self, _queue, cell, **_kwargs):
            self.calls.append(cell["cell_id"])
            raise KeyboardInterrupt

    evaluator = InterruptingEvaluator()
    work = tmp_path / "work"
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=source,
        work_root=work,
        runtime=runtime,
        store=store,
        stager=stager,
        evaluator=evaluator,
        artifacts=FakeArtifacts(),
    )

    with pytest.raises(KeyboardInterrupt):
        runner.run()

    assert stager.cells == [first]
    assert evaluator.calls == [first]
    assert queue["claims"]["completion"] not in store.values
    assert not any(key.endswith("/failure.complete.json") for key in store.values)
    assert not list((work / "cells").iterdir())
    assert json.loads(runner.campaign_state_path.read_text())["status"] == "running"
    assert not (work / "state/cells" / f"{second}.json").exists()


def test_operator_interrupt_during_fixed50_audit_is_not_reclassified(tmp_path, monkeypatch):
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.fixed50, "_result_claim", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.Fixed50Artifacts().publish_success(
            {},
            {"run_id": "run", "task": "PickXtimes", "arm": "q3", "eval_id": "eval"},
            output=tmp_path / "output",
            cell_work=tmp_path / "cell",
            checkpoint_uri="s3://bucket/checkpoint",
            store=MemoryStore(),
            runtime=None,
        )


def test_exact_remote_result_skips_staging_on_fresh_local_state(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("s0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    store = MemoryStore()
    cell = queue["cells"][0]
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    result = FakeArtifacts().publish_success(
        queue,
        cell,
        output=seed_root,
        cell_work=seed_root,
        checkpoint_uri="unused",
        store=store,
        runtime=runtime,
    )
    assert result["successes"] == 17
    stager = FakeStager()
    evaluator = ScriptedEvaluator({cell["cell_id"]: ["should-not-run"]})
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=source,
        work_root=tmp_path / "work",
        runtime=runtime,
        store=store,
        stager=stager,
        evaluator=evaluator,
        artifacts=FakeArtifacts(),
    )
    assert runner.run() == 0
    assert not stager.cells and not evaluator.calls
    completion = json.loads(store.values[queue["claims"]["completion"]])
    assert completion["records"][0]["status"] == "skipped_exact_complete"


def test_exact_remote_terminal_failure_skips_rerun_after_preemption(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("s0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    store = MemoryStore()
    cell = queue["cells"][0]
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    FakeArtifacts().publish_failure(
        queue,
        cell,
        cell_work=seed_root,
        failure={
            "status": "terminal_failure",
            "attempts": [{"attempt": 1, "status": "terminal_failure"}],
            "failure_class": "unclassified",
            "detail": "sealed before worker preemption",
        },
        store=store,
    )
    stager = FakeStager()
    evaluator = ScriptedEvaluator({cell["cell_id"]: ["should-not-run"]})
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=source,
        work_root=tmp_path / "work",
        runtime=runtime,
        store=store,
        stager=stager,
        evaluator=evaluator,
        artifacts=FakeArtifacts(),
    )
    assert runner.run() == 2
    assert not stager.cells and not evaluator.calls
    assert queue["claims"]["completion"] not in store.values
    state = json.loads(runner.campaign_state_path.read_text())
    assert state["status"] == "halted_terminal_failure"
    assert state["records"][0]["status"] == "skipped_terminal_failure"


def test_eval_campaign_defers_before_likely_partial_cell(tmp_path, monkeypatch):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("s0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    ticks = iter((0.0, 86_000.0))
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(ticks))
    stager = FakeStager()
    evaluator = ScriptedEvaluator({queue["cells"][0]["cell_id"]: ["should-not-run"]})
    runner = campaign.CampaignRunner(
        queue=queue,
        source_root=source,
        work_root=tmp_path / "work",
        runtime=runtime,
        store=MemoryStore(),
        stager=stager,
        evaluator=evaluator,
        artifacts=FakeArtifacts(),
    )
    assert runner.run() == 0
    assert not stager.cells and not evaluator.calls
    state = json.loads(runner.campaign_state_path.read_text())
    assert state["status"] == "deferred_runtime_budget"
    assert state["records"][0]["status"] == "deferred_runtime_budget"


def test_large_failure_log_archive_preserves_bounded_tail_and_original_identity(tmp_path):
    _preflight, _receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _queue(source, gates, ("q0",))
    cell = queue["cells"][0]
    cell_work = tmp_path / "cell"
    log = cell_work / "attempts/attempt-1/output/server.log"
    log.parent.mkdir(parents=True)
    payload = (
        b"verbose-start\n"
        + b"x" * (4 * 1024 * 1024 + 256)
        + b"\nFailed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY\n"
    )
    log.write_bytes(payload)
    store = MemoryStore()
    failure = campaign.Fixed50Artifacts().publish_failure(
        queue,
        cell,
        cell_work=cell_work,
        failure={
            "status": "terminal_failure",
            "attempts": [],
            "failure_class": "gpu_resource_exhausted",
            "detail": "evaluation fleet returned 1",
        },
        store=store,
    )
    archive = store.values[failure["evidence_archive_uri"]]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
        names = stream.getnames()
        manifest_name = next(name for name in names if name.endswith("bounded-large-file-manifest.json"))
        bounded_name = next(name for name in names if name.endswith("server.log.head-tail.txt"))
        manifest = json.load(stream.extractfile(manifest_name))
        bounded = stream.extractfile(bounded_name).read()
    assert manifest["files"] == [
        {
            "attempt": "attempt-1",
            "path": "output/server.log",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "captured_head_bytes": 64 * 1024,
            "captured_tail_bytes": 4 * 1024 * 1024 - 64 * 1024,
        }
    ]
    assert b"CUDA_ERROR_OUT_OF_MEMORY" in bounded
