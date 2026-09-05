from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from robomme_integration.eval import campaign
from robomme_integration.eval import parallel_campaign as parallel
from robomme_integration.eval.launch_gpu_fleet import main as gpu_fleet_main
from robomme_integration.eval.launch_sharded import (
    _native_launch_waves,
    _wait_for_prewarm_markers,
)
from robomme_integration.tests.test_eval_campaign import (
    FakeArtifacts,
    FakeStager,
    MemoryStore,
    _gates,
    _queue,
)


def _parallel_queue(source: Path, gates: dict, arms: tuple[str, ...]) -> dict:
    queue = dict(_queue(source, gates, arms))
    queue.pop("queue_manifest_sha256")
    queue["topology"] = parallel.local_2x5090_topology().as_queue_topology()
    return campaign.seal_document(queue, field="queue_manifest_sha256")


def _runner(
    tmp_path: Path,
    arms: tuple[str, ...],
    evaluator_factory,
):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _parallel_queue(source, gates, arms)
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    store = MemoryStore()
    stager = FakeStager()
    runner = parallel.ParallelCampaignRunner(
        queue=queue,
        source_root=source,
        work_root=tmp_path / "work",
        runtime=runtime,
        store=store,
        stager=stager,
        artifacts=FakeArtifacts(),
        evaluator_factory=evaluator_factory,
        resource_admission=lambda _queue, topology, _work: {
            "status": "pass",
            "parallel_topology_sha256": topology.as_queue_topology()["parallel_topology_sha256"],
        },
        lane_admission=lambda lane: {"lane_id": lane.lane_id},
        lane_release_wait=lambda lane: {"lane_id": lane.lane_id},
        disk_free=lambda _path: 1 << 60,
    )
    return runner, queue, store, stager


def test_parallel_topology_presets_are_self_sealed_and_disjoint():
    local = parallel.local_2x5090_topology()
    assert [lane.policy_gpu for lane in local.lanes] == [0, 1]
    assert [lane.port for lane in local.lanes] == [18100, 18101]
    assert [lane.cpu_range for lane in local.lanes] == ["0-63", "64-127"]
    assert all(lane.simulator_shards == 4 for lane in local.lanes)
    assert all(lane.shard_prewarm_seconds > lane.shard_stagger_seconds > 0 for lane in local.lanes)
    assert parallel.ParallelTopology.from_queue_topology(local.as_queue_topology()) == local

    p5 = parallel.p5_8xh100_topology()
    assert len(p5.lanes) == 8
    assert [lane.policy_gpu for lane in p5.lanes] == list(range(8))
    assert [lane.port for lane in p5.lanes] == list(range(18100, 18108))
    assert p5.as_queue_topology()["simulator_shards"] == 32
    assert set().union(*(parallel._cpu_set(lane.cpu_range) for lane in p5.lanes)) == set(range(192))

    drift = local.as_queue_topology()
    drift["lanes"][0]["port"] += 10
    with pytest.raises(ValueError, match="self-seal drift"):
        parallel.ParallelTopology.from_queue_topology(drift)


def test_lane_port_release_wait_accepts_only_bounded_transient_drain():
    lane = parallel.local_2x5090_topology().lanes[0]
    availability = iter((False, False, True))
    clock = iter((0.0, 0.1, 0.2, 0.3))
    sleeps: list[float] = []

    result = parallel.wait_for_lane_port_release(
        lane,
        port_available=lambda _port: next(availability),
        port_has_listener=lambda _port: False,
        monotonic=lambda: next(clock),
        sleep=sleeps.append,
    )

    assert result["polls"] == 2
    assert result["port"] == lane.port
    assert sleeps == [0.25, 0.25]


def test_lane_port_release_wait_rejects_listener_and_timeout():
    lane = parallel.local_2x5090_topology().lanes[0]
    with pytest.raises(parallel.ResourceAdmissionError, match="live listener"):
        parallel.wait_for_lane_port_release(
            lane,
            port_available=lambda _port: False,
            port_has_listener=lambda _port: True,
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("external listener must fail without waiting")
            ),
        )

    clock = iter((0.0, 30.0))
    with pytest.raises(parallel.ResourceAdmissionError, match="did not release"):
        parallel.wait_for_lane_port_release(
            lane,
            port_available=lambda _port: False,
            port_has_listener=lambda _port: False,
            monotonic=lambda: next(clock),
            sleep=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("expired release deadline must fail without another sleep")
            ),
        )


def test_parallel_topology_rejects_overlap_and_resource_admission_is_fail_closed(tmp_path, monkeypatch):
    local = parallel.local_2x5090_topology()
    lane0, lane1 = local.lanes
    with pytest.raises(ValueError, match="CPU ranges overlap"):
        parallel.ParallelTopology(
            topology_id="overlap",
            lanes=(lane0, parallel.LaneSpec(**{**lane1.as_dict(), "cpu_range": "32-95"})),
            minimum_free_disk_bytes=1,
        )
    topology = parallel.ParallelTopology(
        topology_id=local.topology_id,
        lanes=local.lanes,
        minimum_free_disk_bytes=1,
    )
    queue = {
        "limits": {"minimum_free_bytes": 1},
    }
    states = {
        gpu: parallel.GpuState(
            index=gpu,
            name="NVIDIA GeForce RTX 5090",
            uuid=f"GPU-{gpu}",
            total_bytes=32 * parallel.GIB,
            free_bytes=32 * parallel.GIB,
            utilization_percent=0,
        )
        for gpu in range(2)
    }
    monkeypatch.setattr(
        parallel.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=100 * parallel.GIB),
    )
    admitted = parallel.admit_parallel_resources(
        queue,
        topology,
        tmp_path,
        gpu_states=states,
        allowed_cpus=set(range(128)),
        port_available=lambda _port: True,
    )
    assert len(admitted["lanes"]) == 2
    states[0] = parallel.GpuState(**{**states[0].as_dict(), "compute_pids": (1234,)})
    with pytest.raises(parallel.ResourceAdmissionError, match="occupied"):
        parallel.admit_parallel_resources(
            queue,
            topology,
            tmp_path,
            gpu_states=states,
            allowed_cpus=set(range(128)),
            port_available=lambda _port: True,
        )


def test_runner_blocks_before_manifest_or_staging_when_campaign_admission_fails(tmp_path):
    calls: list[str] = []

    class NeverEvaluator:
        def run(self, _queue, cell, **_kwargs):
            calls.append(cell["cell_id"])
            return 0

    runner, queue, store, stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda _lane, _cancel: NeverEvaluator(),
    )

    def reject(_queue, _topology, _work):
        raise parallel.ResourceAdmissionError("synthetic insufficient GPU headroom")

    runner.resource_admission = reject
    assert runner.run() == 3
    assert not calls and not stager.cells and not store.values
    assert queue["claims"]["manifest"] not in store.values
    assert json.loads(runner._transaction.campaign_state_path.read_text())["status"] == ("blocked_resource_admission")


def test_native_shards_launch_in_one_per_gpu_waves_and_lane_command_is_exact(tmp_path):
    assert _native_launch_waves(4, (0,)) == [[0], [1], [2], [3]]
    assert _native_launch_waves(8, (0, 1)) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert _native_launch_waves(4, ()) == [[0, 1, 2, 3]]

    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _parallel_queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    lane = parallel.local_2x5090_topology().lanes[1]
    launch_queue = {**queue, "topology": lane.launch_topology()}
    command = campaign.build_launch_command(
        launch_queue,
        queue["cells"][0],
        source_root=source,
        runtime=runtime,
        checkpoint=tmp_path / "checkpoint",
        workspace=None,
        output=tmp_path / "output",
    )
    assert command[:3] == ["taskset", "--cpu-list", "64-127"]
    assert command[command.index("--gpus") + 1] == "1"
    assert command[command.index("--simulator-gpus") + 1] == "1"
    assert command[command.index("--shards") + 1] == "4"
    assert command[command.index("--cpu-range") + 1] == "64-127"
    assert command[command.index("--base-port") + 1] == "18101"
    assert command[command.index("--native-shard-prewarm-seconds") + 1] == "180.0"
    assert command[command.index("--native-shard-stagger-seconds") + 1] == "30.0"
    with pytest.raises(ValueError, match="ParallelCampaignRunner"):
        campaign.build_launch_command(
            queue,
            queue["cells"][0],
            source_root=source,
            runtime=runtime,
            checkpoint=tmp_path / "checkpoint",
            workspace=None,
            output=tmp_path / "output",
        )
    dry = parallel.dry_run_payload(queue, source, runtime, tmp_path / "work")
    assert dry["execution_mode"] == parallel.PARALLEL_EXECUTION_MODE
    assert dry["cells"][0]["lane_id"] == "local5090-gpu0"
    assert dry["note"].startswith("no S3 access")


def test_gpu_fleet_forwards_native_prewarm_and_stagger(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    server = source / "robomme_integration/eval/execution_model_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# server\n", encoding="utf-8")
    (source / "robomme_integration/compat/robocasa").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    config = tmp_path / "pick.yaml"
    config.write_text("benchmark: fixed50\n", encoding="utf-8")
    executable = tmp_path / "eval/bin/vla-eval"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    help_text = " ".join(
        f"--args.{name}" for name in ("checkpoint", "arm", "task_name", "model_seed", "chunk_size", "max_batch_size")
    )
    monkeypatch.setattr(
        "robomme_integration.eval.launch_gpu_fleet.subprocess.run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=help_text),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_gpu_fleet.py",
            "--source-root",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--arm",
            "s0",
            "--task",
            "PickXtimes",
            "--benchmark-config",
            str(config),
            "--vla-eval",
            str(executable),
            "--output-root",
            str(tmp_path / "output"),
            "--eval-id",
            "parallel-prewarm",
            "--gpus",
            "0",
            "--base-port",
            "18100",
            "--shards",
            "4",
            "--native-simulator",
            "--simulator-gpus",
            "0",
            "--native-shard-prewarm-seconds",
            "180",
            "--native-shard-stagger-seconds",
            "30",
            "--dry-run",
        ],
    )
    assert gpu_fleet_main() == 0
    payload = json.loads(capsys.readouterr().out)
    launcher = payload["launcher_command"]
    assert launcher[launcher.index("--native-shard-prewarm-seconds") + 1] == "180.0"
    assert launcher[launcher.index("--native-shard-stagger-seconds") + 1] == "30.0"
    assert launcher[launcher.index("--native-prewarm-marker") + 1] == "inference arm=s0"
    assert launcher[launcher.index("--native-prewarm-log") + 1].endswith("server-gpu0-port18100.log")


def test_two_lanes_run_cells_concurrently_and_publish_one_ordered_completion(tmp_path):
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    counters = {"active": 0, "maximum": 0}

    class ConcurrentEvaluator:
        def __init__(self):
            self.first = True

        def run(self, _queue, _cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            with lock:
                counters["active"] += 1
                counters["maximum"] = max(counters["maximum"], counters["active"])
            try:
                if self.first:
                    self.first = False
                    barrier.wait(timeout=5)
                time.sleep(0.01)
                (output / "shard.log").write_text("success\n", encoding="utf-8")
                return 0
            finally:
                with lock:
                    counters["active"] -= 1

    runner, queue, store, stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda _lane, _cancel: ConcurrentEvaluator(),
    )
    assert runner.run() == 0
    assert counters["maximum"] == 2
    assert set(stager.cells) == {cell["cell_id"] for cell in queue["cells"]}
    completion = json.loads(store.values[queue["claims"]["completion"]])
    assert completion["status"] == "complete"
    assert [record["cell_id"] for record in completion["records"]] == [cell["cell_id"] for cell in queue["cells"]]
    assert all(record["successes"] == 17 for record in completion["records"])


def test_parallel_resume_skips_exact_remote_cell_and_runs_only_missing_claims(tmp_path):
    calls: list[str] = []
    lock = threading.Lock()

    class RecordingEvaluator:
        def run(self, _queue, cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            with lock:
                calls.append(cell["cell_id"])
            (output / "shard.log").write_text("success\n", encoding="utf-8")
            return 0

    runner, queue, store, stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda _lane, _cancel: RecordingEvaluator(),
    )
    seeded = queue["cells"][0]
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    FakeArtifacts().publish_success(
        queue,
        seeded,
        output=seed_root,
        cell_work=seed_root,
        checkpoint_uri="unused",
        store=store,
        runtime=runner.runtime,
    )

    assert runner.run() == 0
    assert seeded["cell_id"] not in calls
    assert seeded["cell_id"] not in stager.cells
    assert set(calls) == {cell["cell_id"] for cell in queue["cells"][1:]}
    completion = json.loads(store.values[queue["claims"]["completion"]])
    assert completion["records"][0]["status"] == "skipped_exact_complete"
    assert [record["cell_id"] for record in completion["records"]] == [cell["cell_id"] for cell in queue["cells"]]


def test_parallel_resume_rejects_conflicting_success_and_failure_claims(tmp_path):
    class NeverEvaluator:
        def run(self, *_args, **_kwargs):
            raise AssertionError("dual terminal claims must reject before evaluation")

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0",),
        lambda _lane, _cancel: NeverEvaluator(),
    )
    cell = queue["cells"][0]
    success_root = tmp_path / "dual-success"
    success_root.mkdir()
    FakeArtifacts().publish_success(
        queue,
        cell,
        output=success_root,
        cell_work=success_root,
        checkpoint_uri="unused",
        store=store,
        runtime=runner.runtime,
    )
    failure_root = tmp_path / "dual-failure"
    failure_root.mkdir()
    FakeArtifacts().publish_failure(
        queue,
        cell,
        cell_work=failure_root,
        failure={
            "status": "terminal_failure",
            "attempts": [],
            "failure_class": "unclassified",
            "detail": "conflicting terminal outcome",
        },
        store=store,
    )

    with pytest.raises(ValueError, match="conflicting immutable success and failure"):
        runner.run()


def test_lane_terminal_failure_stops_only_its_lane(tmp_path):
    calls: list[str] = []
    lock = threading.Lock()

    class RoutedEvaluator:
        def run(self, _queue, cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            with lock:
                calls.append(cell["cell_id"])
            if cell["ordinal"] == 0:
                (output / "shard.log").write_text("unknown harness failure\n", encoding="utf-8")
                return 1
            (output / "shard.log").write_text("success\n", encoding="utf-8")
            return 0

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda _lane, _cancel: RoutedEvaluator(),
    )
    assert runner.run() == 2
    assert queue["cells"][0]["cell_id"] in calls
    assert queue["cells"][2]["cell_id"] not in calls
    assert queue["cells"][1]["cell_id"] in calls
    assert queue["cells"][3]["cell_id"] in calls
    assert queue["claims"]["completion"] not in store.values
    assert any(key.endswith("/failure.complete.json") for key in store.values)
    assert json.loads(runner._transaction.campaign_state_path.read_text())["status"] == (
        "halted_lane_terminal_failure"
    )


def test_gpu_oom_is_systemic_and_cancels_peer_without_false_failure_claim(tmp_path):
    barrier = threading.Barrier(2)

    class SystemicEvaluator:
        def __init__(self, lane, cancel):
            self.lane = lane
            self.cancel = cancel

        def run(self, _queue, _cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            barrier.wait(timeout=5)
            if self.lane.policy_gpu == 0:
                (output / "server.log").write_text(
                    "RESOURCE_EXHAUSTED: [0] Failed to load in-memory CUBIN: "
                    "CUDA_ERROR_OUT_OF_MEMORY\n"
                    "websocket connection closed unexpectedly\n",
                    encoding="utf-8",
                )
                return 1
            assert self.cancel.wait(timeout=5)
            raise campaign.EvaluatorCancelled("peer lane requested systemic cancellation")

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda lane, cancel: SystemicEvaluator(lane, cancel),
    )

    class CancelAwareArtifacts(FakeArtifacts):
        def publish_failure(self, *args, **kwargs):
            assert runner._cancel.is_set(), "systemic cancellation must precede evidence upload"
            return super().publish_failure(*args, **kwargs)

    runner._transaction.artifacts = CancelAwareArtifacts()
    assert runner.run() == 2
    state = json.loads(runner._transaction.campaign_state_path.read_text())
    assert state["status"] == "halted_systemic_failure"
    failure_keys = [key for key in store.values if key.endswith("/failure.complete.json")]
    assert failure_keys == [f"{queue['publish_root_s3']}/cells/{queue['cells'][0]['cell_id']}/failure.complete.json"]
    failure = json.loads(store.values[failure_keys[0]])
    assert failure["failure_class"] == "gpu_resource_exhausted"
    assert len(failure["attempts"]) == 1
    assert not list((runner.work_root / "cells").iterdir())


def test_simultaneous_gpu_ooms_publish_exactly_one_systemic_failure_claim(tmp_path):
    barrier = threading.Barrier(2)

    class OomEvaluator:
        def run(self, _queue, _cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            barrier.wait(timeout=5)
            (output / "server.log").write_text(
                "Failed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY\n",
                encoding="utf-8",
            )
            return 1

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda _lane, _cancel: OomEvaluator(),
    )
    assert runner.run() == 2
    failures = [key for key in store.values if key.endswith("/failure.complete.json")]
    assert len(failures) == 1
    assert runner._systemic_cell in {cell["cell_id"] for cell in queue["cells"]}
    assert json.loads(store.values[failures[0]])["failure_class"] == ("gpu_resource_exhausted")


def test_operator_interrupt_cancels_peer_and_never_mints_terminal_claim(tmp_path):
    barrier = threading.Barrier(2)

    class InterruptEvaluator:
        def __init__(self, lane, cancel):
            self.lane = lane
            self.cancel = cancel

        def run(self, _queue, _cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            barrier.wait(timeout=5)
            if self.lane.policy_gpu == 0:
                raise KeyboardInterrupt
            assert self.cancel.wait(timeout=5)
            raise campaign.EvaluatorCancelled("operator interrupted peer lane")

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda lane, cancel: InterruptEvaluator(lane, cancel),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    assert queue["claims"]["completion"] not in store.values
    assert not any(key.endswith("/failure.complete.json") for key in store.values)
    assert not list((runner.work_root / "cells").iterdir())
    assert json.loads(runner._transaction.campaign_state_path.read_text())["status"] == ("interrupted_parallel")


def test_cancellable_subprocess_evaluator_drains_its_process_group(monkeypatch, tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _parallel_queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    cancel = threading.Event()
    observed = {"drained": False}

    class FakeProcess:
        pid = 4321

        @staticmethod
        def wait(*, timeout=None):
            if timeout is None:
                raise AssertionError("cancellable evaluator must poll")
            cancel.set()
            raise subprocess.TimeoutExpired("fake", timeout)

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(
        campaign,
        "_terminate_evaluator_process",
        lambda process: observed.update(drained=process.pid == 4321),
    )
    evaluator = campaign.SubprocessEvaluator(
        source,
        runtime,
        topology_override=parallel.local_2x5090_topology().lanes[0].launch_topology(),
        cancel_event=cancel,
        cancel_poll_seconds=0.01,
    )
    with pytest.raises(campaign.EvaluatorCancelled):
        evaluator.run(
            queue,
            queue["cells"][0],
            checkpoint=tmp_path / "checkpoint",
            workspace=None,
            output=tmp_path / "output",
        )
    assert observed["drained"] is True


def test_cancellable_subprocess_cleanup_failure_is_never_swallowed(monkeypatch, tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _parallel_queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    cancel = threading.Event()

    class FakeProcess:
        pid = 4321

        @staticmethod
        def wait(*, timeout=None):
            cancel.set()
            raise subprocess.TimeoutExpired("fake", timeout)

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(
        campaign,
        "_terminate_evaluator_process",
        lambda _process: (_ for _ in ()).throw(RuntimeError("process tree remains live")),
    )
    evaluator = campaign.SubprocessEvaluator(
        source,
        runtime,
        topology_override=parallel.local_2x5090_topology().lanes[0].launch_topology(),
        cancel_event=cancel,
        cancel_poll_seconds=0.01,
    )
    with pytest.raises(campaign.EvaluatorCleanupFailure, match="cleanup failed"):
        evaluator.run(
            queue,
            queue["cells"][0],
            checkpoint=tmp_path / "checkpoint",
            workspace=None,
            output=tmp_path / "output",
        )


def test_aws_staging_cancellation_terminates_sync_process_group(monkeypatch, tmp_path):
    cancel = threading.Event()
    observed = {"drained": False}

    class SyncProcess:
        pid = 5321
        returncode = None

        @staticmethod
        def wait(*, timeout=None):
            assert timeout == 0.01
            cancel.set()
            raise subprocess.TimeoutExpired("aws s3 sync", timeout)

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_args, **_kwargs: SyncProcess())
    monkeypatch.setattr(
        campaign,
        "_terminate_evaluator_process",
        lambda process: observed.update(drained=process.pid == 5321),
    )
    stager = campaign.AwsStager(MemoryStore(), cancel_event=cancel, cancel_poll_seconds=0.01)
    with pytest.raises(campaign.EvaluatorCancelled, match="S3 staging"):
        stager._sync("s3://bucket/checkpoint", tmp_path / "checkpoint")
    assert observed["drained"] is True


def test_aws_object_read_cancellation_terminates_cli_process_group(monkeypatch):
    cancel = threading.Event()
    observed = {"drained": False}

    class ReadProcess:
        pid = 6321
        returncode = None

        @staticmethod
        def communicate(*, timeout=None):
            if timeout is None:
                return b"", b""
            assert timeout == 0.01
            cancel.set()
            raise subprocess.TimeoutExpired("aws s3 cp", timeout)

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_args, **_kwargs: ReadProcess())
    monkeypatch.setattr(
        campaign,
        "_terminate_evaluator_process",
        lambda process: observed.update(drained=process.pid == 6321),
    )
    store = campaign.AwsCliStore(cancel_event=cancel, cancel_poll_seconds=0.01)
    with pytest.raises(campaign.EvaluatorCancelled, match="S3 object access"):
        store.read_bytes("s3://bucket/claim.json")
    assert observed["drained"] is True


def test_parallel_runner_separates_cancellable_staging_reads_from_commit_store(tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _parallel_queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    commit_store = campaign.AwsCliStore()
    stager = campaign.AwsStager(commit_store)
    runner = parallel.ParallelCampaignRunner(
        queue=queue,
        source_root=source,
        work_root=tmp_path / "work",
        runtime=runtime,
        store=commit_store,
        stager=stager,
        artifacts=FakeArtifacts(),
    )
    assert runner.store is commit_store
    assert commit_store.cancel_event is None
    assert runner.stager.store is not commit_store
    assert runner.stager.store.cancel_event is runner._cancel
    assert runner.stager.cancel_event is runner._cancel


def test_staging_error_after_cancellation_never_mints_terminal_claim(tmp_path):
    class CancelledStager:
        def __init__(self, cancel):
            self.cancel = cancel

        def stage_checkpoint(self, _cell, _destination):
            self.cancel.set()
            raise subprocess.CalledProcessError(143, ["aws", "s3", "sync"])

        def stage_workspace(self, *_args, **_kwargs):
            raise AssertionError("workspace staging must not continue")

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("q0",),
        lambda _lane, _cancel: (_ for _ in ()).throw(
            AssertionError("evaluator must not launch after cancelled staging")
        ),
    )
    runner._transaction.stager = CancelledStager(runner._cancel)
    with pytest.raises(campaign.EvaluatorCancelled, match="staging or publication"):
        runner._transaction.run_cell_transaction(
            queue["cells"][0],
            cancel_event=runner._cancel,
        )
    assert not any(key.endswith("/failure.complete.json") for key in store.values)
    assert not list((runner.work_root / "cells").iterdir())


def test_large_log_tail_oom_is_systemic_and_dominates_transport_marker(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "server.log").write_bytes(
        b"connection reset by peer\n"
        + b"x" * (4 * 1024 * 1024 + 128)
        + b"\nFailed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY\n"
    )
    assert campaign.classify_transient(output, "fleet failed") == "harness_transport_reset"
    assert campaign.classify_systemic(output, "fleet failed") == "gpu_resource_exhausted"


def test_p5_admission_accounts_for_actual_xla_allocator_fraction():
    lane = parallel.p5_8xh100_topology().lanes[0]
    state = parallel.GpuState(
        index=0,
        name="NVIDIA H100 80GB HBM3",
        uuid="GPU-H100",
        total_bytes=80 * parallel.GIB,
        free_bytes=60 * parallel.GIB,
        utilization_percent=0,
    )
    with pytest.raises(parallel.ResourceAdmissionError, match="free GPU memory"):
        parallel._admit_lane_snapshot(
            lane,
            state,
            allowed_cpus=set(range(24)),
            port_available=lambda _port: True,
        )
    admitted = parallel._admit_lane_snapshot(
        lane,
        parallel.GpuState(**{**state.as_dict(), "free_bytes": 80 * parallel.GIB}),
        allowed_cpus=set(range(24)),
        port_available=lambda _port: True,
    )
    assert admitted["xla_allocator_reservation_bytes"] == 52 * parallel.GIB
    assert admitted["required_free_gpu_bytes"] == 66 * parallel.GIB


def test_host_gpu_and_port_leases_are_exclusive_and_releasable(tmp_path):
    topology = parallel.local_2x5090_topology()
    lane = topology.lanes[0]
    states = {
        0: parallel.GpuState(
            index=0,
            name="NVIDIA GeForce RTX 5090",
            uuid="GPU-EXCLUSIVE-0",
            total_bytes=32 * parallel.GIB,
            free_bytes=32 * parallel.GIB,
            utilization_percent=0,
        )
    }
    first = parallel.acquire_host_resource_leases(
        topology,
        (lane,),
        lease_root=tmp_path / "leases",
        gpu_states=states,
    )
    try:
        with pytest.raises(parallel.ResourceAdmissionError, match="already held"):
            parallel.acquire_host_resource_leases(
                topology,
                (lane,),
                lease_root=tmp_path / "leases",
                gpu_states=states,
            )
    finally:
        first.close()
    released = parallel.acquire_host_resource_leases(
        topology,
        (lane,),
        lease_root=tmp_path / "leases",
        gpu_states=states,
    )
    released.close()


def test_completion_fast_path_requires_exact_manifest_object(tmp_path):
    class SuccessfulEvaluator:
        def run(self, _queue, _cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            (output / "shard.log").write_text("success\n", encoding="utf-8")
            return 0

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda _lane, _cancel: SuccessfulEvaluator(),
    )
    assert runner.run() == 0
    store.values.pop(queue["claims"]["manifest"])
    with pytest.raises(ValueError, match="no exact canonical manifest"):
        runner.run()


def test_resume_leases_and_admits_only_lane_with_missing_cells(tmp_path):
    calls: list[str] = []

    class SuccessfulEvaluator:
        def run(self, _queue, cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            calls.append(cell["cell_id"])
            (output / "shard.log").write_text("success\n", encoding="utf-8")
            return 0

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda _lane, _cancel: SuccessfulEvaluator(),
    )
    for cell in queue["cells"][0::2]:
        seed_root = tmp_path / f"seed-{cell['ordinal']}"
        seed_root.mkdir()
        FakeArtifacts().publish_success(
            queue,
            cell,
            output=seed_root,
            cell_work=seed_root,
            checkpoint_uri="unused",
            store=store,
            runtime=runner.runtime,
        )
    leased: list[str] = []

    def lease_factory(_topology, lanes):
        leased.extend(lane.lane_id for lane in lanes)
        return parallel.HostResourceLeases([], ())

    runner.lease_factory = lease_factory
    assert runner.run() == 0
    assert leased == ["local5090-gpu1"]
    assert set(calls) == {cell["cell_id"] for cell in queue["cells"][1::2]}


def test_sigterm_cancels_active_lanes_and_never_mints_cell_claim(tmp_path):
    barrier = threading.Barrier(3)

    class BlockingEvaluator:
        def __init__(self, cancel):
            self.cancel = cancel

        def run(self, _queue, _cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            barrier.wait(timeout=5)
            assert self.cancel.wait(timeout=5)
            raise campaign.EvaluatorCancelled("SIGTERM cancelled active lane")

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda _lane, cancel: BlockingEvaluator(cancel),
    )

    def terminate_campaign():
        barrier.wait(timeout=5)
        os.kill(os.getpid(), signal.SIGTERM)

    trigger = threading.Thread(target=terminate_campaign)
    trigger.start()
    with pytest.raises(parallel.ParallelCampaignSignal):
        runner.run()
    trigger.join(timeout=5)
    assert not trigger.is_alive()
    assert queue["claims"]["completion"] not in store.values
    assert not any(key.endswith("/failure.complete.json") for key in store.values)
    assert json.loads(runner._transaction.campaign_state_path.read_text())["status"] == ("interrupted_parallel")


def test_evaluator_teardown_sweeps_descendant_groups_after_supervisor_exit(
    monkeypatch,
):
    killed: list[tuple[int, signal.Signals]] = []

    class ExitedProcess:
        pid = 4321

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def wait(*, timeout=None):
            return 0

    monkeypatch.setattr(campaign, "_descendant_process_groups", lambda _pid: {7001, 7002})
    monkeypatch.setattr(campaign.os, "killpg", lambda group, sig: killed.append((group, sig)))
    campaign._terminate_evaluator_process(ExitedProcess(), grace_seconds=0)
    assert killed == [(7001, signal.SIGKILL), (7002, signal.SIGKILL)]


def test_policy_jit_prewarm_requires_real_inference_marker(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("Loaded arm=q3\ninference arm=q3 batch_call=1\n", encoding="utf-8")

    class RunningProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    _wait_for_prewarm_markers([log], "inference arm=q3", 0.1, [RunningProcess()])
    with pytest.raises(RuntimeError, match="did not appear"):
        _wait_for_prewarm_markers(
            [tmp_path / "missing.log"],
            "inference arm=q3",
            0.001,
            [RunningProcess()],
        )


def test_known_systemic_failure_claim_halts_before_any_resource_admission(tmp_path):
    calls: list[str] = []

    class NeverEvaluator:
        def run(self, _queue, cell, **_kwargs):
            calls.append(cell["cell_id"])
            return 0

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda _lane, _cancel: NeverEvaluator(),
    )
    cell = queue["cells"][0]
    evidence = tmp_path / "seed-systemic"
    evidence.mkdir()
    FakeArtifacts().publish_failure(
        queue,
        cell,
        cell_work=evidence,
        failure={
            "status": "terminal_failure",
            "attempts": [],
            "failure_class": "gpu_resource_exhausted",
            "detail": "seeded exact OOM claim",
        },
        store=store,
    )

    def admission_must_not_run(*_args, **_kwargs):
        raise AssertionError("known systemic claim must halt before resource admission")

    runner.resource_admission = admission_must_not_run
    assert runner.run() == 2
    assert not calls
    assert json.loads(runner._transaction.campaign_state_path.read_text())["status"] == ("halted_systemic_failure")


def test_all_exact_resume_finishes_without_disk_or_gpu_admission(tmp_path, monkeypatch):
    class NeverEvaluator:
        def run(self, *_args, **_kwargs):
            raise AssertionError("all exact cells must be resumed, not evaluated")

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda _lane, _cancel: NeverEvaluator(),
    )
    for cell in queue["cells"]:
        evidence = tmp_path / f"seed-exact-{cell['ordinal']}"
        evidence.mkdir()
        FakeArtifacts().publish_success(
            queue,
            cell,
            output=evidence,
            cell_work=evidence,
            checkpoint_uri="unused",
            store=store,
            runtime=runner.runtime,
        )

    def admission_must_not_run(*_args, **_kwargs):
        raise AssertionError("all exact resume must not require GPU admission")

    runner.resource_admission = admission_must_not_run
    runner.disk_free = lambda _path: (_ for _ in ()).throw(AssertionError("no disk admission required"))
    assert runner.run() == 0
    completion = json.loads(store.values[queue["claims"]["completion"]])
    assert all(record["status"] == "skipped_exact_complete" for record in completion["records"])


def test_midrun_shared_disk_floor_cancels_all_lanes_before_staging(tmp_path, monkeypatch):
    calls: list[str] = []

    class NeverEvaluator:
        def run(self, _queue, cell, **_kwargs):
            calls.append(cell["cell_id"])
            raise AssertionError("a depleted shared filesystem must stop every lane")

    runner, queue, store, stager = _runner(
        tmp_path,
        ("s0", "q0"),
        lambda _lane, _cancel: NeverEvaluator(),
    )
    floor = runner._topology.minimum_free_disk_bytes
    assert queue["limits"]["minimum_free_bytes"] < floor
    runner.disk_free = lambda _path: floor - 1

    assert runner.run() == 3
    assert not calls and not stager.cells
    assert queue["claims"]["completion"] not in store.values
    assert not any(key.endswith("/failure.complete.json") for key in store.values)
    state = json.loads(runner._transaction.campaign_state_path.read_text())
    assert state["status"] == "blocked_disk_floor"
    assert all(record["minimum_free_bytes"] == floor for record in state["records"])


def test_midrun_lane_admission_block_cancels_peers_without_failure_claim(tmp_path):
    calls: list[str] = []
    admissions: dict[str, int] = {}

    class SuccessfulEvaluator:
        def run(self, _queue, cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            calls.append(cell["cell_id"])
            (output / "shard.log").write_text("success\n", encoding="utf-8")
            return 0

    runner, queue, store, _stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda _lane, _cancel: SuccessfulEvaluator(),
    )

    def lane_admission(lane):
        admissions[lane.lane_id] = admissions.get(lane.lane_id, 0) + 1
        if lane.policy_gpu == 0 and admissions[lane.lane_id] == 2:
            raise parallel.ResourceAdmissionError("synthetic leaked GPU allocation")
        return {"lane_id": lane.lane_id}

    runner.lane_admission = lane_admission
    assert runner.run() == 3
    assert queue["cells"][0]["cell_id"] in calls
    assert queue["cells"][2]["cell_id"] not in calls
    assert queue["claims"]["completion"] not in store.values
    assert not any(key.endswith("/failure.complete.json") for key in store.values)


def test_successful_lane_waits_for_port_release_before_next_admission(tmp_path):
    events: dict[str, list[str]] = {}

    class SuccessfulEvaluator:
        def __init__(self, lane_id: str):
            self.lane_id = lane_id

        def run(self, _queue, cell, *, output, **_kwargs):
            output.mkdir(parents=True)
            events.setdefault(self.lane_id, []).append(f"eval:{cell['cell_id']}")
            (output / "shard.log").write_text("success\n", encoding="utf-8")
            return 0

    runner, queue, _store, _stager = _runner(
        tmp_path,
        ("s0", "q0", "q1", "q2"),
        lambda lane, _cancel: SuccessfulEvaluator(lane.lane_id),
    )
    runner.lane_admission = lambda lane: events.setdefault(lane.lane_id, []).append("admit") or {}
    runner.lane_release_wait = lambda lane: events.setdefault(lane.lane_id, []).append("release") or {}

    assert runner.run() == 0
    for lane_index, lane in enumerate(runner._topology.lanes):
        lane_cells = queue["cells"][lane_index :: len(runner._topology.lanes)]
        assert events[lane.lane_id] == [
            "admit",
            f"eval:{lane_cells[0]['cell_id']}",
            "release",
            "admit",
            f"eval:{lane_cells[1]['cell_id']}",
        ]


def test_expired_deadline_stops_before_spawning_evaluator(monkeypatch, tmp_path):
    preflight, receipt, gates = _gates(tmp_path)
    source = tmp_path / "source"
    queue = _parallel_queue(source, gates, ("q0",))
    runtime = campaign.verify_gates(queue, preflight_claim=preflight, runtime_receipt=receipt)
    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("expired evaluator must not spawn")),
    )
    evaluator = campaign.SubprocessEvaluator(
        source,
        runtime,
        topology_override=parallel.local_2x5090_topology().lanes[0].launch_topology(),
        cancel_event=threading.Event(),
        deadline_monotonic=time.monotonic() - 1,
    )
    with pytest.raises(campaign.EvaluatorDeadlineExceeded):
        evaluator.run(
            queue,
            queue["cells"][0],
            checkpoint=tmp_path / "checkpoint",
            workspace=None,
            output=tmp_path / "output",
        )
