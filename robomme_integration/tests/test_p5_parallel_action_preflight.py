from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from robomme_integration.eval import (
    campaign,
    launch_p5_preflight,
    parallel_campaign,
)
from robomme_integration.eval import (
    p5_parallel_action_preflight as action,
)


def _manifest() -> dict:
    source = launch_p5_preflight.REPO_ROOT / "robomme_integration"
    args = launch_p5_preflight.parser().parse_args(["--dry-run", "--parallel-action-canary"])
    return launch_p5_preflight.build_plan(args, source)["manifest"]


def _runtime(tmp_path: Path, *, policy_python: Path | None = None) -> campaign.Runtime:
    return campaign.Runtime(
        receipt_sha256="0" * 64,
        preflight_claim_sha256="1" * 64,
        policy_python=policy_python or tmp_path / "openpi/.venv/bin/python",
        vla_eval=tmp_path / "links/vla-eval",
        harness_src=tmp_path / "runtime/harness",
        robomme_src=tmp_path / "runtime/robomme",
        maniskill_src=tmp_path / "runtime/maniskill",
        openpi_src=tmp_path / "openpi/src",
        policy_site=tmp_path / "openpi/site",
        simulator_site=tmp_path / "runtime/site",
        upstream_root=tmp_path / "upstream",
        vision_encoder_home=tmp_path / "vision",
        render_environment={
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "ROBOMME_USE_LAVAPIPE": "auto",
        },
    )


def _records(tmp_path: Path, manifest: dict) -> tuple[list[dict], Path, Path]:
    checkpoint = tmp_path / "inputs/checkpoint/19999"
    workspace = tmp_path / "inputs/workspace"
    checkpoint.mkdir(parents=True)
    workspace.mkdir(parents=True)
    config = tmp_path / "evidence/action-canary.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(action.CONFIG)
    records = action.build_lane_commands(
        manifest,
        source_root=tmp_path / "source",
        runtime=_runtime(tmp_path),
        checkpoint=checkpoint,
        workspace=workspace,
        config=config,
        work_root=tmp_path / "evidence",
    )
    return records, checkpoint, workspace


def _materialize_lane(record: dict, checkpoint: Path, workspace: Path) -> None:
    lane = record["lane"]
    output = record["output"]
    eval_root = output / "eval"
    eval_root.mkdir(parents=True)
    (output / "COMPLETED").write_text("ok\n", encoding="utf-8")
    record["launcher_log"].write_text("returncode=0\n", encoding="utf-8")
    server_log = output / f"server-gpu{lane.policy_gpu}-port{lane.port}.log"
    server_log.write_text(
        f"Loaded arm=q1\nStarting server on ws://127.0.0.1:{lane.port}\ninference arm=q1\n",
        encoding="utf-8",
    )
    supervisor = {
        "launcher_returncode": 0,
        "failure": None,
        "task": action.CANARY_TASK,
        "arm": action.CANARY_ARM,
        "checkpoint": str(checkpoint.resolve()),
        "gpus": [lane.policy_gpu],
        "simulator_gpus": [lane.simulator_gpu],
        "ports": [lane.port],
        "shards": 4,
        "cpu_range": lane.cpu_range,
        "pin_native_cpus": True,
        "xla_memory_fraction": lane.xla_memory_fraction,
        "native_shard_prewarm_seconds": lane.shard_prewarm_seconds,
        "native_shard_stagger_seconds": lane.shard_stagger_seconds,
        "server_commands": [
            [
                "policy",
                str(checkpoint.resolve()),
                "--args.workspace_checkpoint",
                str(workspace.resolve()),
            ]
        ],
    }
    (output / "supervisor.json").write_text(json.dumps(supervisor), encoding="utf-8")
    aggregate = {
        "tasks": [
            {
                "task": action.CANARY_TASK,
                "episodes": [
                    {
                        "episode_idx": index,
                        "episode_id": index,
                        "failure_reason": None,
                        "steps": 1,
                        "metrics": {"success": bool(index % 2)},
                    }
                    for index in range(4)
                ],
            }
        ]
    }
    aggregate_path = eval_root / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    partitions = action._expected_cpu_partitions(lane.cpu_range)
    url = f"ws://127.0.0.1:{lane.port}"
    launch = {
        "eval_id": record["eval_id"],
        "config_sha256": action.CONFIG_SHA256,
        "shards": 4,
        "recording_mode": "sqlite",
        "server_urls": [url],
        "returncodes": [0, 0, 0, 0],
        "native": {
            "gpus": [lane.simulator_gpu],
            "gpu_by_shard": [lane.simulator_gpu] * 4,
            "cpu_partitions": partitions,
            "launch_waves": [[0], [1], [2], [3]],
            "shard_prewarm_seconds": lane.shard_prewarm_seconds,
            "shard_stagger_seconds": lane.shard_stagger_seconds,
            "prewarm_marker": "inference arm=q1",
            "prewarm_logs": ["server.log"],
        },
        "commands": [["taskset", "--cpu-list", partition, "vla-eval", url] for partition in partitions],
        "episode_audit": {"episodes": 4, "harness_failures": []},
        "materialized_results": [
            {
                "path": str(aggregate_path),
                "bytes": aggregate_path.stat().st_size,
                "sha256": campaign._sha256(aggregate_path),
            }
        ],
    }
    (eval_root / "launch_manifest.json").write_text(json.dumps(launch), encoding="utf-8")


def test_action_manifest_and_commands_bind_exact_disjoint_topology(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert action.load_and_validate_manifest(manifest_path) == manifest
    records, _checkpoint, _workspace = _records(tmp_path, manifest)
    assert len(records) == 8
    assert {record["lane"].policy_gpu for record in records} == set(range(8))
    assert {record["lane"].port for record in records} == set(range(18100, 18108))
    assert {record["lane"].cpu_range for record in records} == {f"{gpu * 24}-{gpu * 24 + 23}" for gpu in range(8)}
    for record in records:
        lane = record["lane"]
        command = record["command"]
        assert command[:3] == ["taskset", "--cpu-list", lane.cpu_range]
        assert command[command.index("--gpus") + 1] == str(lane.policy_gpu)
        assert command[command.index("--simulator-gpus") + 1] == str(lane.simulator_gpu)
        assert command[command.index("--base-port") + 1] == str(lane.port)
        assert command[command.index("--shards") + 1] == "4"
        config = Path(command[command.index("--benchmark-config") + 1])
        assert config.read_bytes() == action.CONFIG


def test_action_runtime_preserves_lexical_venv_python_symlink(tmp_path):
    lexical = tmp_path / "openpi/.venv/bin/python"
    lexical.parent.mkdir(parents=True)
    lexical.symlink_to(sys.executable)
    args = argparse.Namespace(
        policy_python=lexical,
        vla_eval=tmp_path / "links/vla-eval",
        harness_src=tmp_path / "harness",
        robomme_src=tmp_path / "robomme",
        maniskill_src=tmp_path / "maniskill",
        openpi_src=tmp_path / "openpi/src",
        policy_site=tmp_path / "openpi/site",
        simulator_site=tmp_path / "simulator-site",
        upstream_root=tmp_path / "upstream",
        vision_encoder_home=tmp_path / "vision",
    )
    runtime = action._runtime_from_args(args, _manifest())
    assert runtime.policy_python == lexical.absolute()
    assert runtime.policy_python != lexical.resolve()
    config = tmp_path / "canary.yaml"
    config.write_bytes(action.CONFIG)
    checkpoint = tmp_path / "checkpoint"
    workspace = tmp_path / "workspace"
    checkpoint.mkdir()
    workspace.mkdir()
    command = action.build_lane_commands(
        _manifest(),
        source_root=tmp_path / "source",
        runtime=runtime,
        checkpoint=checkpoint,
        workspace=workspace,
        config=config,
        work_root=tmp_path / "work",
    )[0]["command"]
    assert str(lexical.absolute()) in command
    assert str(lexical.resolve()) not in command


def test_audit_and_claim_require_8_servers_32_valid_actions_and_zero_scores(tmp_path):
    manifest = _manifest()
    records, checkpoint, workspace = _records(tmp_path, manifest)
    for record in records:
        _materialize_lane(record, checkpoint, workspace)
    topology = parallel_campaign.p5_8xh100_topology()
    observed = action.audit_canary(
        records,
        checkpoint=checkpoint,
        workspace=workspace,
        topology=topology,
    )
    observed.update(all_lane_gpus_idle=True, all_lane_ports_free=True)
    evidence_sha = "a" * 64
    evidence = {
        "uri": f"{manifest['evidence_root_s3']}/{evidence_sha}.tgz",
        "sha256": evidence_sha,
        "bytes": 123,
    }
    claim = action.build_success_claim(manifest, observed=observed, evidence=evidence)
    assert claim["score_publication"] == {
        "performed": False,
        "result_claim_uris": [],
    }
    assert "result_claim_s3" not in claim["cell"]
    assert (
        action.validate_success_claim(
            claim,
            source_sha256=manifest["source_tree_sha256"],
            expected_openpi=manifest["openpi"],
            expected_topology=manifest["topology"],
            expected_image=manifest["image"],
            expected_vision=manifest["vision"],
            expected_upstream=manifest["upstream"],
            expected_infrastructure=manifest["infrastructure"],
        )
        == claim
    )

    publishable = tmp_path / "publishable"
    config = tmp_path / "evidence/action-canary.yaml"
    action._build_publishable_evidence(
        manifest,
        observed=observed,
        config=config,
        destination=publishable,
    )
    archive = tmp_path / "publishable.tgz"
    action._deterministic_archive(publishable, archive)
    with tarfile.open(archive, "r:gz") as stream:
        assert stream.getnames() == ["action-canary.yaml", "attestation.json"]
        attestation = stream.extractfile("attestation.json").read()
    assert b'"success_values_published": false' in attestation
    assert b'"metrics"' not in attestation
    assert b"aggregate.json" not in archive.read_bytes()
    assert b"launcher.log" not in archive.read_bytes()

    drift = dict(observed)
    drift["actions_executed"] = 31
    with pytest.raises(ValueError, match="complete valid evidence"):
        action.build_success_claim(manifest, observed=drift, evidence=evidence)
    tampered = json.loads(json.dumps(claim))
    tampered["score_publication"]["performed"] = True
    with pytest.raises(ValueError, match="must not publish scores"):
        action.validate_success_claim(
            tampered,
            source_sha256=manifest["source_tree_sha256"],
            expected_openpi=manifest["openpi"],
            expected_topology=manifest["topology"],
        )
    missing_idle = dict(observed)
    missing_idle.pop("all_lane_gpus_idle")
    with pytest.raises(ValueError, match="complete valid evidence"):
        action.build_success_claim(manifest, observed=missing_idle, evidence=evidence)
    false_idle = dict(observed)
    false_idle["all_lane_ports_free"] = False
    with pytest.raises(ValueError, match="complete valid evidence"):
        action.build_success_claim(manifest, observed=false_idle, evidence=evidence)


def test_config_drift_is_rejected_and_evidence_archive_is_deterministic(tmp_path):
    manifest = _manifest()
    drift = json.loads(json.dumps(manifest))
    drift["cell"]["benchmark_config_sha256"] = "f" * 64
    drift.pop("manifest_sha256")
    drift = campaign.seal_document(drift, field="manifest_sha256")
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(drift), encoding="utf-8")
    with pytest.raises(ValueError, match="source-matched PickXtimes/Q1"):
        action.load_and_validate_manifest(path)

    namespace_drift = json.loads(json.dumps(manifest))
    namespace_drift["claim_s3"] = f"s3://caller-bucket/preflight/{namespace_drift['preflight_id']}.json"
    namespace_drift.pop("manifest_sha256")
    namespace_drift = campaign.seal_document(namespace_drift, field="manifest_sha256")
    path.write_text(json.dumps(namespace_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="claim URI"):
        action.load_and_validate_manifest(path)

    evidence_drift = json.loads(json.dumps(manifest))
    evidence_drift["evidence_root_s3"] = f"s3://caller-bucket/{evidence_drift['preflight_id']}/evidence"
    evidence_drift.pop("manifest_sha256")
    evidence_drift = campaign.seal_document(evidence_drift, field="manifest_sha256")
    path.write_text(json.dumps(evidence_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence root"):
        action.load_and_validate_manifest(path)

    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "lane").mkdir(parents=True)
        (root / "lane/log.txt").write_text("same evidence\n", encoding="utf-8")
        os.utime(root / "lane/log.txt", (1 if root == first else 999, 1 if root == first else 999))
    first_archive = tmp_path / "first.tgz"
    second_archive = tmp_path / "second.tgz"
    assert action._deterministic_archive(first, first_archive) == (
        action._deterministic_archive(second, second_archive)
    )
    assert first_archive.read_bytes() == second_archive.read_bytes()


def test_lane_failure_requests_cleanup_for_every_process_group(tmp_path, monkeypatch):
    created = []
    terminated = []
    killed = []

    class Process:
        def __init__(self, command):
            self.command = command
            self.index = len(created)
            self.returncode = 9 if self.index == 0 else 0
            created.append(self)

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process(command)

    def communicate(process, *, timeout_seconds):
        assert timeout_seconds == 12
        return (f"stdout-{process.index}", "", False)

    def request_termination(process):
        terminated.append(process.index)
        return {10_000 + process.index}

    def kill_groups(groups):
        killed.append(set(groups))

    monkeypatch.setattr(action.subprocess, "Popen", popen)
    monkeypatch.setattr(action, "_communicate_process", communicate)
    monkeypatch.setattr(action, "_request_process_termination", request_termination)
    monkeypatch.setattr(action, "_kill_process_groups", kill_groups)
    records = [
        {
            "command": ["lane", str(index)],
            "lane_id": f"lane-{index}",
            "output": tmp_path / f"output-{index}",
            "launcher_log": tmp_path / f"lane-{index}.log",
        }
        for index in range(8)
    ]
    with pytest.raises(RuntimeError, match="lane-0 failed with 9") as caught:
        action._run_lane_processes(records, timeout_seconds=12)
    assert sorted(terminated) == list(range(8))
    assert set().union(*killed) == {10_000 + index for index in range(8)}
    assert caught.value.__notes__ == ["all p5 action-canary process groups received cleanup"]


def test_resource_drain_waits_for_pids_and_ports_then_requires_idle(monkeypatch):
    topology = parallel_campaign.p5_8xh100_topology()
    attempts = {"count": 0}

    def states():
        attempts["count"] += 1
        busy = attempts["count"] == 1
        return {
            lane.policy_gpu: parallel_campaign.GpuState(
                index=lane.policy_gpu,
                name="NVIDIA H100",
                uuid=f"gpu-{lane.policy_gpu}",
                total_bytes=80 * 1024**3,
                free_bytes=80 * 1024**3,
                utilization_percent=0,
                compute_pids=(1234,) if busy and lane.policy_gpu == 0 else (),
            )
            for lane in topology.lanes
        }

    monkeypatch.setattr(parallel_campaign, "query_gpu_states", states)
    monkeypatch.setattr(parallel_campaign, "_port_available", lambda _port: True)
    monkeypatch.setattr(action.time, "sleep", lambda _seconds: None)
    assert action._audit_resources_drained(topology, timeout_seconds=1) == {
        "all_lane_gpus_idle": True,
        "all_lane_ports_free": True,
    }
    assert attempts["count"] == 2

    monkeypatch.setattr(parallel_campaign, "query_gpu_states", lambda: {})
    with pytest.raises(RuntimeError, match="resources did not drain"):
        action._audit_resources_drained(topology, timeout_seconds=0)


def test_physical_canary_yaml_must_equal_embedded_bytes(tmp_path, monkeypatch):
    config = tmp_path / action.CANARY_CONFIG_PATH
    config.parent.mkdir(parents=True)
    config.write_text("drift: true\n", encoding="utf-8")
    monkeypatch.setattr(launch_p5_preflight, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="physical p5 action-canary YAML"):
        launch_p5_preflight._load_action_template(launch_p5_preflight.ACTION_TEMPLATE)


def test_action_manifest_accepts_the_standard_priority_class_and_rejects_others(tmp_path):
    """2026-09-05: preflight 9747ba48… (fired at 400 on the lead's instruction) died on the node with
    "wrong accelerator or priority" because this module pinned 100 while the launcher already allowed
    {100, 400}. The node-side set must stay in step with launch_p5_preflight.ALLOWED_PRIORITIES."""
    from robomme_integration.eval import launch_p5_preflight

    assert action.ALLOWED_PRIORITIES == launch_p5_preflight.ALLOWED_PRIORITIES == (100, 400)
    for priority in (100, 400):
        manifest = json.loads(json.dumps(_manifest()))
        manifest["infrastructure"]["priority"] = priority
        manifest.pop("manifest_sha256")
        manifest = campaign.seal_document(manifest, field="manifest_sha256")
        path = tmp_path / f"p{priority}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        assert action.load_and_validate_manifest(path) == manifest
    rejected = json.loads(json.dumps(_manifest()))
    rejected["infrastructure"]["priority"] = 200
    rejected.pop("manifest_sha256")
    rejected = campaign.seal_document(rejected, field="manifest_sha256")
    path = tmp_path / "p200.json"
    path.write_text(json.dumps(rejected), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong accelerator or priority"):
        action.load_and_validate_manifest(path)
