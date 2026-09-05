#!/usr/bin/env python3
"""Run and attest the unscored 8-H100/32-action RoboMME workspace canary.

The canary deliberately does not use :class:`Fixed50Artifacts` and has no result-claim URI.
It stages one authenticated Q1 checkpoint and workspace representation once, restores that same
read-only input on eight disjoint H100 lanes, and asks four native EGL simulator shards per lane
for exactly one action.  Only an immutable preflight claim and its content-addressed evidence
archive are published.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

from robomme_integration.eval import campaign, parallel_campaign
from robomme_integration.launch import STUDY_ROOT

CANARY_KIND = "robomme_p5_native_eval_preflight"
CANARY_MODE = "p5_parallel_workspace_action_v1"
CANARY_STATUS = "native_parallel_action_passed"
CANARY_TEMPLATE_KIND = "robomme_p5_parallel_action_canary_template"
CANARY_ARM = "q1"
CANARY_TASK = "PickXtimes"
#: Node-side mirror of launch_p5_preflight.ALLOWED_PRIORITIES (100 sweep class, 400 standard class,
#: 2026-09-05). Kept literal here because this module runs on the node without the launcher.
ALLOWED_PRIORITIES = (100, 400)
CANARY_CONFIG_PATH = "robomme_integration/eval/configs/p5_parallel_action_canary.yaml"
CANONICAL_CLAIM_ROOT = f"{STUDY_ROOT}/manifests/claims/preflight"
CANONICAL_EVIDENCE_ROOT = f"{STUDY_ROOT}/artifacts/robomme/eval_preflight"
EPISODES_PER_LANE = 4
EXPECTED_ACTIONS = 8 * EPISODES_PER_LANE
CONFIG = b"""server:
  url: "ws://127.0.0.1:18100"
  timeout: 1200

output_dir: "./results/p5_parallel_action_preflight"

benchmarks:
  - benchmark: "robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"
    subname: counting-pickxtimes-p5-parallel-action-preflight
    episodes_per_task: 4
    max_steps: 1
    params:
      tasks: [PickXtimes]
      action_space: joint_angle
      dataset: test
      send_wrist_image: true
      send_state: true
      send_video_history: true
"""
CONFIG_SHA256 = hashlib.sha256(CONFIG).hexdigest()


class CanarySignal(BaseException):
    """SIGTERM converted into the canary's cooperative cleanup path."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"p5 parallel action canary received signal {signum}")
        self.signum = signum


def _pretty(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha(path: Path) -> str:
    return campaign._sha256(path)


def load_and_validate_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported p5 parallel action-preflight manifest")
    if value.get("kind") != CANARY_KIND or value.get("preflight_mode") != CANARY_MODE:
        raise ValueError("manifest is not the exact p5 parallel workspace-action canary")
    claimed = campaign._require_sha(value.get("manifest_sha256"), "action-preflight input manifest seal")
    if campaign._seal_digest(value, "manifest_sha256") != claimed:
        raise ValueError("action-preflight input manifest self-seal mismatch")
    topology = parallel_campaign.ParallelTopology.from_queue_topology(value.get("topology"))
    if topology != parallel_campaign.p5_8xh100_topology():
        raise ValueError("action preflight does not bind the exact 8-H100 topology")
    cell = value.get("cell")
    if not isinstance(cell, dict):
        raise ValueError("action preflight has no Q1 cell")
    if (
        cell.get("task") != CANARY_TASK
        or cell.get("arm") != CANARY_ARM
        or cell.get("final_step") != 19_999
        or cell.get("benchmark_config") != CANARY_CONFIG_PATH
        or cell.get("benchmark_config_sha256") != CONFIG_SHA256
        or cell.get("training_openpi") != value.get("openpi")
        or not isinstance(cell.get("workspace"), dict)
        or cell["workspace"].get("provenance_mode") != campaign.WORKSPACE_PROVENANCE_LEGACY
    ):
        raise ValueError("action preflight cell is not the source-matched PickXtimes/Q1 probe")
    if "result_claim_s3" in cell:
        raise ValueError("unscored action-preflight cell must not contain a score claim URI")
    probe = value.get("probe")
    if probe != {
        "actions_expected": EXPECTED_ACTIONS,
        "arm": CANARY_ARM,
        "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
        "benchmark_config_sha256": CONFIG_SHA256,
        "dataset": "test",
        "episodes_per_lane": EPISODES_PER_LANE,
        "max_steps": 1,
        "native_egl": True,
        "score_publication": False,
        "task": CANARY_TASK,
    }:
        raise ValueError("action-preflight scientific probe contract drift")
    infrastructure = value.get("infrastructure")
    if not isinstance(infrastructure, dict) or infrastructure.get("instance_type") != "ml.p5.48xlarge":
        raise ValueError("action preflight is not assigned to p5")
    if infrastructure.get("accelerator") != "8xH100" or infrastructure.get("priority") not in ALLOWED_PRIORITIES:
        raise ValueError("action preflight has the wrong accelerator or priority")
    claim_s3 = campaign._safe_s3(value.get("claim_s3"))
    preflight_id = value.get("preflight_id")
    if not isinstance(preflight_id, str) or claim_s3 != f"{CANONICAL_CLAIM_ROOT}/{preflight_id}.json":
        raise ValueError("action-preflight claim URI does not bind its identity")
    evidence_root = campaign._safe_s3(value.get("evidence_root_s3"))
    if evidence_root != f"{CANONICAL_EVIDENCE_ROOT}/{preflight_id}/evidence":
        raise ValueError("action-preflight evidence root does not bind its identity")
    return value


def validate_success_claim(
    claim: object,
    *,
    source_sha256: str,
    expected_openpi: dict,
    expected_topology: dict,
    expected_image: dict | None = None,
    expected_vision: dict | None = None,
    expected_upstream: dict | None = None,
    expected_infrastructure: dict | None = None,
) -> dict:
    """Fail closed unless *claim* proves the exact unscored 8-lane action contract."""
    if not isinstance(claim, dict):
        raise ValueError("p5 parallel action-preflight claim must be one JSON object")
    if (
        claim.get("kind") != CANARY_KIND
        or claim.get("status") != CANARY_STATUS
        or claim.get("preflight_mode") != CANARY_MODE
    ):
        raise ValueError("p5 parallel action-preflight did not pass")
    if claim.get("source_tree_sha256") != source_sha256:
        raise ValueError("p5 parallel action-preflight source tree mismatch")
    if claim.get("openpi") != expected_openpi:
        raise ValueError("p5 parallel action-preflight OpenPI mismatch")
    if claim.get("topology") != expected_topology:
        raise ValueError("p5 parallel action-preflight topology mismatch")
    # Parsing independently checks the aggregate topology, each lane, and the topology self-seal.
    topology = parallel_campaign.ParallelTopology.from_queue_topology(claim["topology"])
    if topology != parallel_campaign.p5_8xh100_topology():
        raise ValueError("p5 parallel action-preflight is not the exact 8-H100 topology")
    if expected_image is not None and claim.get("image") != expected_image:
        raise ValueError("p5 parallel action-preflight image mismatch")
    if expected_vision is not None and claim.get("vision") != expected_vision:
        raise ValueError("p5 parallel action-preflight vision artifact mismatch")
    if expected_upstream is not None and claim.get("upstream") != expected_upstream:
        raise ValueError("p5 parallel action-preflight upstream source mismatch")
    infrastructure = claim.get("infrastructure")
    if not isinstance(infrastructure, dict):
        raise ValueError("p5 parallel action-preflight infrastructure is absent")
    if expected_infrastructure is not None and any(
        infrastructure.get(key) != value for key, value in expected_infrastructure.items()
    ):
        raise ValueError("p5 parallel action-preflight infrastructure mismatch")
    cell = claim.get("cell")
    if (
        not isinstance(cell, dict)
        or cell.get("task") != CANARY_TASK
        or cell.get("arm") != CANARY_ARM
        or cell.get("final_step") != 19_999
        or cell.get("benchmark_config") != CANARY_CONFIG_PATH
        or cell.get("benchmark_config_sha256") != CONFIG_SHA256
        or cell.get("training_openpi") != expected_openpi
        or not isinstance(cell.get("workspace"), dict)
        or cell["workspace"].get("provenance_mode") != campaign.WORKSPACE_PROVENANCE_LEGACY
        or "result_claim_s3" in cell
    ):
        raise ValueError("p5 parallel action-preflight cell provenance mismatch")
    if claim.get("probe") != {
        "actions_expected": EXPECTED_ACTIONS,
        "arm": CANARY_ARM,
        "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
        "benchmark_config_sha256": CONFIG_SHA256,
        "dataset": "test",
        "episodes_per_lane": EPISODES_PER_LANE,
        "max_steps": 1,
        "native_egl": True,
        "score_publication": False,
        "task": CANARY_TASK,
    }:
        raise ValueError("p5 parallel action-preflight probe contract mismatch")
    if claim.get("score_publication") != {
        "performed": False,
        "result_claim_uris": [],
    }:
        raise ValueError("p5 parallel action-preflight must not publish scores")
    observed = claim.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("p5 parallel action-preflight observations are absent")
    expected_observed = {
        "execution_mode": parallel_campaign.PARALLEL_EXECUTION_MODE,
        "parallel_topology_sha256": expected_topology["parallel_topology_sha256"],
        "parallel_lanes": 8,
        "policy_servers": 8,
        "native_shards_per_lane": EPISODES_PER_LANE,
        "native_shards_total": EXPECTED_ACTIONS,
        "episodes": EXPECTED_ACTIONS,
        "actions_executed": EXPECTED_ACTIONS,
        "harness_failures": 0,
        "load_completed_before_readiness": True,
        "all_lane_gpus_idle": True,
        "all_lane_ports_free": True,
    }
    if set(observed) != {*expected_observed, "lanes"} or any(
        observed.get(key) != value for key, value in expected_observed.items()
    ):
        raise ValueError("p5 parallel action-preflight combined observations are incomplete")
    lanes = observed.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != len(topology.lanes):
        raise ValueError("p5 parallel action-preflight lane evidence is incomplete")
    digest_fields = {
        "supervisor_sha256",
        "server_log_sha256",
    }
    for lane, lane_observed in zip(topology.lanes, lanes, strict=True):
        if not isinstance(lane_observed, dict):
            raise ValueError("p5 parallel action-preflight lane evidence is malformed")
        expected_lane = {
            "lane_id": lane.lane_id,
            "gpu": lane.policy_gpu,
            "port": lane.port,
            "cpu_range": lane.cpu_range,
            "episodes": EPISODES_PER_LANE,
            "actions_executed": EPISODES_PER_LANE,
            "harness_failures": 0,
            "load_completed_before_readiness": True,
        }
        if set(lane_observed) != {*expected_lane, *digest_fields} or any(
            lane_observed.get(key) != value for key, value in expected_lane.items()
        ):
            raise ValueError("p5 parallel action-preflight lane observations mismatch")
        for key in digest_fields:
            campaign._require_sha(lane_observed.get(key), f"action-preflight lane {key}")
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"uri", "sha256", "bytes"}:
        raise ValueError("p5 parallel action-preflight evidence artifact is malformed")
    evidence_uri = campaign._safe_s3(evidence.get("uri"))
    evidence_sha = campaign._require_sha(evidence.get("sha256"), "action-preflight evidence")
    if evidence_sha not in evidence_uri or not isinstance(evidence.get("bytes"), int) or evidence["bytes"] < 1:
        raise ValueError("p5 parallel action-preflight evidence is not content addressed")
    if not evidence_uri.startswith(f"{campaign._safe_s3(claim.get('evidence_root_s3'))}/"):
        raise ValueError("p5 parallel action-preflight evidence escaped its namespace")
    preflight_id = claim.get("preflight_id")
    claim_uri = campaign._safe_s3(claim.get("claim_s3"))
    if (
        not isinstance(preflight_id, str)
        or claim_uri != f"{CANONICAL_CLAIM_ROOT}/{preflight_id}.json"
        or campaign._safe_s3(claim.get("evidence_root_s3")) != f"{CANONICAL_EVIDENCE_ROOT}/{preflight_id}/evidence"
    ):
        raise ValueError("p5 parallel action-preflight claim URI does not bind its identity")

    # Authenticate both the launch input and the post-run claim.  The publisher seals the claim
    # before appending status, matching the original rendered-reset preflight convention.
    input_manifest = dict(claim)
    for key in (
        "status",
        "manifest_sha256",
        "input_manifest_sha256",
        "observed",
        "evidence",
        "score_publication",
    ):
        input_manifest.pop(key, None)
    input_seal = campaign._require_sha(claim.get("input_manifest_sha256"), "action-preflight input manifest seal")
    if campaign._seal_digest(input_manifest, "manifest_sha256") != input_seal:
        raise ValueError("p5 parallel action-preflight input manifest seal mismatch")
    sealed_claim = dict(claim)
    manifest_sha = campaign._require_sha(sealed_claim.pop("manifest_sha256", None), "action-preflight claim seal")
    sealed_claim.pop("status", None)
    if campaign._seal_digest(sealed_claim, "manifest_sha256") != manifest_sha:
        raise ValueError("p5 parallel action-preflight claim seal mismatch")
    return claim


def _verify_runtime_assets(manifest: dict, runtime: campaign.Runtime) -> None:
    vision = manifest.get("vision")
    if not isinstance(vision, dict):
        raise RuntimeError("p5 action-canary vision provenance is absent")
    vision_path = runtime.vision_encoder_home / "pi05_vision_encoder/siglip_params.pkl"
    if (
        not vision_path.is_file()
        or vision_path.stat().st_size != vision.get("bytes")
        or _sha(vision_path) != vision.get("sha256")
    ):
        raise RuntimeError("p5 action-canary staged vision artifact identity mismatch")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict) or not isinstance(upstream.get("critical_sha256"), dict):
        raise RuntimeError("p5 action-canary upstream provenance is absent")
    try:
        commit = subprocess.run(
            ["git", "-C", str(runtime.upstream_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"p5 action-canary upstream identity probe failed: {error}") from error
    if commit != upstream.get("commit"):
        raise RuntimeError("p5 action-canary upstream commit mismatch")
    for relative, expected_sha in sorted(upstream["critical_sha256"].items()):
        path = (runtime.upstream_root / relative).resolve()
        if not path.is_relative_to(runtime.upstream_root) or not path.is_file():
            raise RuntimeError(f"p5 action-canary upstream critical file is absent: {relative}")
        if _sha(path) != expected_sha:
            raise RuntimeError(f"p5 action-canary upstream critical file drift: {relative}")


def _runtime_from_args(args: argparse.Namespace, manifest: dict) -> campaign.Runtime:
    def lexical_absolute(path: Path) -> Path:
        # Do not resolve the OpenPI venv's ``bin/python`` symlink: invoking its base-interpreter
        # target directly changes Python's venv discovery and therefore the installed package set.
        return Path(os.path.abspath(path.expanduser()))

    return campaign.Runtime(
        receipt_sha256="0" * 64,
        preflight_claim_sha256="0" * 64,
        policy_python=lexical_absolute(args.policy_python),
        vla_eval=lexical_absolute(args.vla_eval),
        harness_src=args.harness_src.resolve(),
        robomme_src=args.robomme_src.resolve(),
        maniskill_src=args.maniskill_src.resolve(),
        openpi_src=args.openpi_src.resolve(),
        policy_site=args.policy_site.resolve(),
        simulator_site=args.simulator_site.resolve(),
        upstream_root=args.upstream_root.resolve(),
        vision_encoder_home=args.vision_encoder_home.resolve(),
        render_environment={
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "ROBOMME_USE_LAVAPIPE": "auto",
        },
    )


def _descendant_process_groups(root_pid: int) -> set[int]:
    """Snapshot descendant process groups so timeout cleanup cannot orphan GPU children."""
    pending = [root_pid]
    seen: set[int] = set()
    groups: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(int(child) for child in children)
            groups.add(os.getpgid(pid))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    groups.discard(os.getpgrp())
    return groups


def _kill_process_groups(groups: set[int]) -> None:
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _request_process_termination(process: subprocess.Popen) -> set[int]:
    if process.returncode is not None:
        return set()
    groups = _descendant_process_groups(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return groups


def _terminate_process(process: subprocess.Popen, *, grace_seconds: int = 60) -> tuple[str, str]:
    groups = _descendant_process_groups(process.pid)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        result = process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        groups.update(_descendant_process_groups(process.pid))
        result = None
    _kill_process_groups(groups)
    return process.communicate() if result is None else result


def _communicate_process(process: subprocess.Popen, *, timeout_seconds: int) -> tuple[str, str, bool]:
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process(process)
        return stdout, stderr, True
    except BaseException as error:
        try:
            _terminate_process(process)
        except Exception as cleanup_error:
            error.add_note(f"p5 action-canary cleanup also failed: {cleanup_error}")
        raise


def build_lane_commands(
    manifest: dict,
    *,
    source_root: Path,
    runtime: campaign.Runtime,
    checkpoint: Path,
    workspace: Path,
    config: Path,
    work_root: Path,
) -> list[dict[str, Any]]:
    topology = parallel_campaign.ParallelTopology.from_queue_topology(manifest["topology"])
    if config.read_bytes() != CONFIG:
        raise ValueError("p5 action-canary runtime config differs from its sealed identity")
    cell = {**manifest["cell"], "benchmark_config": str(config)}
    records = []
    for lane in topology.lanes:
        lane_id = lane.lane_id
        eval_id = f"{manifest['preflight_id']}-{lane_id}"
        lane_cell = {**cell, "eval_id": eval_id}
        command = campaign.build_launch_command(
            {"topology": lane.launch_topology()},
            lane_cell,
            source_root=source_root,
            runtime=runtime,
            checkpoint=checkpoint,
            workspace=workspace,
            output=work_root / "lanes" / lane_id / "output",
        )
        records.append(
            {
                "lane": lane,
                "lane_id": lane_id,
                "eval_id": eval_id,
                "output": work_root / "lanes" / lane_id / "output",
                "launcher_log": work_root / "lanes" / lane_id / "launcher.log",
                "command": command,
            }
        )
    return records


def _run_lane_processes(records: list[dict[str, Any]], *, timeout_seconds: int) -> None:
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    futures: dict[concurrent.futures.Future, dict[str, Any]] = {}
    captured_groups: set[int] = set()
    for record in records:
        output = record["output"]
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing prior p5 action-canary evidence: {output}")

    try:
        for record in records:
            record["process"] = subprocess.Popen(
                record["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(records))
        for record in records:
            future = executor.submit(
                _communicate_process,
                record["process"],
                timeout_seconds=timeout_seconds,
            )
            record["future"] = future
            futures[future] = record
        for future in concurrent.futures.as_completed(futures):
            record = futures[future]
            stdout, stderr, timed_out = future.result()
            process = record["process"]
            record["launcher_log"].parent.mkdir(parents=True, exist_ok=True)
            record["launcher_log"].write_text(
                f"returncode={process.returncode}\ntimeout_seconds={timeout_seconds}\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
                encoding="utf-8",
            )
            if timed_out or process.returncode:
                status = "timed out" if timed_out else f"failed with {process.returncode}"
                raise RuntimeError(f"p5 action-canary lane {record['lane_id']} {status}")
        executor.shutdown(wait=True)
        executor = None
    except BaseException as error:
        for record in records:
            process = record.get("process")
            if process is not None:
                captured_groups.update(_request_process_termination(process))
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        running = [future for future in futures if not future.cancelled()]
        if running:
            _done, pending = concurrent.futures.wait(running, timeout=60)
            if pending:
                _kill_process_groups(captured_groups)
                concurrent.futures.wait(pending, timeout=60)
        _kill_process_groups(captured_groups)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        error.add_note("all p5 action-canary process groups received cleanup")
        raise


def _expected_cpu_partitions(cpu_range: str) -> list[str]:
    start, end = (int(value) for value in cpu_range.split("-", 1))
    count = end - start + 1
    base, extra = divmod(count, EPISODES_PER_LANE)
    cursor = start
    result = []
    for shard in range(EPISODES_PER_LANE):
        width = base + int(shard < extra)
        result.append(f"{cursor}-{cursor + width - 1}")
        cursor += width
    return result


def audit_lane(
    record: dict[str, Any],
    *,
    checkpoint: Path,
    workspace: Path,
) -> dict:
    lane = record["lane"]
    output = record["output"]
    supervisor_path = output / "supervisor.json"
    launch_path = output / "eval/launch_manifest.json"
    server_log_path = output / f"server-gpu{lane.policy_gpu}-port{lane.port}.log"
    if not (output / "COMPLETED").is_file():
        raise RuntimeError(f"p5 action-canary lane did not complete: {lane.lane_id}")
    if any(not path.is_file() for path in (supervisor_path, launch_path, server_log_path)):
        raise RuntimeError(f"p5 action-canary lane evidence is incomplete: {lane.lane_id}")
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    expected_checkpoint = str(checkpoint.resolve())
    expected_workspace = str(workspace.resolve())
    if (
        supervisor.get("launcher_returncode") != 0
        or supervisor.get("failure") is not None
        or supervisor.get("task") != CANARY_TASK
        or supervisor.get("arm") != CANARY_ARM
        or supervisor.get("checkpoint") != expected_checkpoint
        or supervisor.get("gpus") != [lane.policy_gpu]
        or supervisor.get("simulator_gpus") != [lane.simulator_gpu]
        or supervisor.get("ports") != [lane.port]
        or supervisor.get("shards") != EPISODES_PER_LANE
        or supervisor.get("cpu_range") != lane.cpu_range
        or supervisor.get("pin_native_cpus") is not True
        or supervisor.get("xla_memory_fraction") != lane.xla_memory_fraction
        or supervisor.get("native_shard_prewarm_seconds") != lane.shard_prewarm_seconds
        or supervisor.get("native_shard_stagger_seconds") != lane.shard_stagger_seconds
    ):
        raise RuntimeError(f"p5 action-canary supervisor contract failed: {lane.lane_id}")
    commands = supervisor.get("server_commands")
    if not isinstance(commands, list) or len(commands) != 1:
        raise RuntimeError("p5 action-canary requires exactly one policy server per lane")
    command = commands[0]
    workspace_flags = ("--args.workspace_checkpoint", "--workspace_checkpoint")
    workspace_flag = next((flag for flag in workspace_flags if flag in command), None)
    if (
        workspace_flag is None
        or command[command.index(workspace_flag) + 1] != expected_workspace
        or expected_checkpoint not in command
    ):
        raise RuntimeError("p5 action-canary server did not receive exact staged inputs")
    expected_partitions = _expected_cpu_partitions(lane.cpu_range)
    native = launch.get("native")
    launch_commands = launch.get("commands")
    audit = launch.get("episode_audit")
    expected_url = f"ws://127.0.0.1:{lane.port}"
    if (
        launch.get("eval_id") != record["eval_id"]
        or launch.get("config_sha256") != CONFIG_SHA256
        or launch.get("shards") != EPISODES_PER_LANE
        or launch.get("recording_mode") != "sqlite"
        or launch.get("server_urls") != [expected_url]
        or launch.get("returncodes") != [0] * EPISODES_PER_LANE
        or not isinstance(native, dict)
        or native.get("gpus") != [lane.simulator_gpu]
        or native.get("gpu_by_shard") != [lane.simulator_gpu] * EPISODES_PER_LANE
        or native.get("cpu_partitions") != expected_partitions
        or native.get("launch_waves") != [[0], [1], [2], [3]]
        or native.get("shard_prewarm_seconds") != lane.shard_prewarm_seconds
        or native.get("shard_stagger_seconds") != lane.shard_stagger_seconds
        or native.get("prewarm_marker") != f"inference arm={CANARY_ARM}"
        or len(native.get("prewarm_logs", [])) != 1
        or not isinstance(launch_commands, list)
        or len(launch_commands) != EPISODES_PER_LANE
        or any(
            command[:3] != ["taskset", "--cpu-list", expected_partitions[index]] or expected_url not in command
            for index, command in enumerate(launch_commands)
        )
        or not isinstance(audit, dict)
        or audit.get("episodes") != EPISODES_PER_LANE
        or audit.get("harness_failures") != []
    ):
        raise RuntimeError(f"p5 action-canary native harness contract failed: {lane.lane_id}")
    materialized = launch.get("materialized_results")
    if not isinstance(materialized, list) or len(materialized) != 1:
        raise RuntimeError("p5 action-canary lane must produce one aggregate")
    aggregate_path = Path(materialized[0].get("path", "")).resolve()
    if (
        not aggregate_path.is_relative_to((output / "eval").resolve())
        or not aggregate_path.is_file()
        or aggregate_path.stat().st_size != materialized[0].get("bytes")
        or _sha(aggregate_path) != materialized[0].get("sha256")
    ):
        raise RuntimeError("p5 action-canary aggregate identity mismatch")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    episodes = [
        episode
        for task in aggregate.get("tasks", [])
        if isinstance(task, dict) and task.get("task") == CANARY_TASK
        for episode in task.get("episodes", [])
        if isinstance(episode, dict)
    ]
    if (
        len(episodes) != EPISODES_PER_LANE
        or sorted(episode.get("episode_idx") for episode in episodes) != list(range(EPISODES_PER_LANE))
        or sorted(episode.get("episode_id") for episode in episodes) != list(range(EPISODES_PER_LANE))
        or any(
            episode.get("failure_reason")
            or episode.get("steps") != 1
            or not isinstance(episode.get("metrics", {}).get("success"), bool)
            for episode in episodes
        )
    ):
        raise RuntimeError("p5 action-canary did not execute exactly one valid action per episode")
    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    loaded = server_log.find(f"Loaded arm={CANARY_ARM}")
    serving = server_log.find("Starting server on")
    inferred = server_log.find(f"inference arm={CANARY_ARM}")
    if (
        server_log.count(f"Loaded arm={CANARY_ARM}") != 1
        or min(loaded, serving, inferred) < 0
        or not loaded < serving < inferred
        or "Traceback (most recent call last)" in server_log
        or "AssertionError" in server_log
    ):
        raise RuntimeError("p5 action-canary did not load-before-ready and produce an action")
    return {
        "lane_id": lane.lane_id,
        "gpu": lane.policy_gpu,
        "port": lane.port,
        "cpu_range": lane.cpu_range,
        "episodes": EPISODES_PER_LANE,
        "actions_executed": EPISODES_PER_LANE,
        "harness_failures": 0,
        "load_completed_before_readiness": True,
        "supervisor_sha256": _sha(supervisor_path),
        "server_log_sha256": _sha(server_log_path),
    }


def audit_canary(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    workspace: Path,
    topology: parallel_campaign.ParallelTopology,
) -> dict:
    lanes = [audit_lane(record, checkpoint=checkpoint, workspace=workspace) for record in records]
    if [item["lane_id"] for item in lanes] != [lane.lane_id for lane in topology.lanes]:
        raise RuntimeError("p5 action-canary evidence does not cover every sealed lane")
    observed = {
        "execution_mode": parallel_campaign.PARALLEL_EXECUTION_MODE,
        "parallel_topology_sha256": topology.as_queue_topology()["parallel_topology_sha256"],
        "parallel_lanes": len(lanes),
        "policy_servers": len(lanes),
        "native_shards_per_lane": EPISODES_PER_LANE,
        "native_shards_total": len(lanes) * EPISODES_PER_LANE,
        "episodes": sum(item["episodes"] for item in lanes),
        "actions_executed": sum(item["actions_executed"] for item in lanes),
        "harness_failures": sum(item["harness_failures"] for item in lanes),
        "load_completed_before_readiness": all(item["load_completed_before_readiness"] for item in lanes),
        "lanes": lanes,
    }
    if observed != {
        **observed,
        "parallel_lanes": 8,
        "policy_servers": 8,
        "native_shards_per_lane": 4,
        "native_shards_total": 32,
        "episodes": 32,
        "actions_executed": 32,
        "harness_failures": 0,
        "load_completed_before_readiness": True,
    }:
        raise RuntimeError("p5 action-canary combined contract failed")
    return observed


def _deterministic_archive(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed) as archive:
                for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
                    if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                        raise RuntimeError(f"unsupported action-canary evidence entry: {path}")
                    relative = path.relative_to(source).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        archive.addfile(info)
                    else:
                        payload = path.read_bytes()
                        info.size = len(payload)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(payload))
    return _sha(destination)


def _build_publishable_evidence(
    manifest: dict,
    *,
    observed: dict,
    config: Path,
    destination: Path,
) -> None:
    """Materialize contract evidence without publishing result records or success values."""
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing prior publishable action-canary evidence: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if config.read_bytes() != CONFIG:
        raise RuntimeError("p5 action-canary evidence config identity drift")
    shutil.copyfile(config, destination / "action-canary.yaml")
    payload = {
        "schema_version": 1,
        "kind": "robomme_p5_parallel_action_canary_redacted_evidence",
        "preflight_id": manifest["preflight_id"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "parallel_topology_sha256": manifest["topology"]["parallel_topology_sha256"],
        "cell": {
            "task": manifest["cell"]["task"],
            "arm": manifest["cell"]["arm"],
            "run_id": manifest["cell"]["run_id"],
            "scientific_spec_sha256": manifest["cell"]["scientific_spec_sha256"],
            "run_manifest_sha256": manifest["cell"]["run_manifest_sha256"],
            "workspace_checkpoint_tree_sha256": manifest["cell"]["workspace"]["checkpoint_tree_sha256"],
        },
        "observed": observed,
        "redaction": {
            "episode_result_records_published": False,
            "success_values_published": False,
            "raw_simulator_logs_published": False,
            "raw_policy_logs_published": False,
            "raw_artifact_digests_retained_in_observed_attestation": True,
        },
    }
    (destination / "attestation.json").write_bytes(_pretty(payload))


def _verify_staged_inputs_unchanged(
    manifest: dict,
    *,
    checkpoint: Path,
    checkpoint_tree: Path,
    workspace: Path,
) -> None:
    checkpoint_uri = f"{manifest['cell']['training_output_s3']}/deploy/{manifest['cell']['final_step']}"
    if not checkpoint_tree.is_file():
        raise RuntimeError("p5 action-canary staged checkpoint tree manifest is absent")
    expected_tree_sha = _sha(checkpoint_tree)
    actual_tree_sha = campaign.verify_checkpoint_manifest(checkpoint, checkpoint_tree, expected_uri=checkpoint_uri)
    if actual_tree_sha != expected_tree_sha:
        raise RuntimeError("p5 action-canary checkpoint changed during inference")
    expected_workspace_sha = manifest["cell"]["workspace"]["checkpoint_tree_sha256"]
    actual_workspace_sha = campaign._legacy_workspace_tree_sha256(
        workspace, expected_step=manifest["cell"]["workspace"]["step"]
    )
    if actual_workspace_sha != expected_workspace_sha:
        raise RuntimeError("p5 action-canary workspace representation changed during inference")


def _audit_resources_drained(
    topology: parallel_campaign.ParallelTopology,
    *,
    timeout_seconds: int = 120,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while True:
        states = parallel_campaign.query_gpu_states()
        busy_gpus = {}
        for lane in topology.lanes:
            state = states.get(lane.policy_gpu)
            if state is None:
                busy_gpus[lane.policy_gpu] = ["gpu_absent"]
            elif state.compute_pids:
                busy_gpus[lane.policy_gpu] = list(state.compute_pids)
        busy_ports = [lane.port for lane in topology.lanes if not parallel_campaign._port_available(lane.port)]
        last = {"busy_gpus": busy_gpus, "busy_ports": busy_ports}
        if not busy_gpus and not busy_ports:
            return {"all_lane_gpus_idle": True, "all_lane_ports_free": True}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"p5 action-canary resources did not drain: {last}")
        time.sleep(1)


def build_success_claim(manifest: dict, *, observed: dict, evidence: dict) -> dict:
    expected_observed_fields = {
        "execution_mode",
        "parallel_topology_sha256",
        "parallel_lanes",
        "policy_servers",
        "native_shards_per_lane",
        "native_shards_total",
        "episodes",
        "actions_executed",
        "harness_failures",
        "load_completed_before_readiness",
        "all_lane_gpus_idle",
        "all_lane_ports_free",
        "lanes",
    }
    if (
        set(observed) != expected_observed_fields
        or observed.get("actions_executed") != EXPECTED_ACTIONS
        or observed.get("harness_failures") != 0
        or observed.get("policy_servers") != 8
        or observed.get("native_shards_total") != 32
        or observed.get("load_completed_before_readiness") is not True
        or observed.get("all_lane_gpus_idle") is not True
        or observed.get("all_lane_ports_free") is not True
    ):
        raise ValueError("refusing a p5 action-preflight claim without complete valid evidence")
    input_seal = manifest["manifest_sha256"]
    claim = {
        **{key: value for key, value in manifest.items() if key != "manifest_sha256"},
        "input_manifest_sha256": input_seal,
        "observed": observed,
        "evidence": evidence,
        "score_publication": {"performed": False, "result_claim_uris": []},
    }
    claim = campaign.seal_document(claim, field="manifest_sha256")
    claim["status"] = CANARY_STATUS
    return validate_success_claim(
        claim,
        source_sha256=manifest["source_tree_sha256"],
        expected_openpi=manifest["openpi"],
        expected_topology=manifest["topology"],
        expected_image=manifest["image"],
        expected_vision=manifest["vision"],
        expected_upstream=manifest["upstream"],
        expected_infrastructure=manifest["infrastructure"],
    )


def _run(args: argparse.Namespace) -> dict:
    manifest = load_and_validate_manifest(args.manifest.resolve())
    source_root = args.source_root.resolve()
    work_root = args.work_root.resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise RuntimeError(f"refusing non-empty p5 action-canary work root: {work_root}")
    work_root.mkdir(parents=True)
    runtime = _runtime_from_args(args, manifest)
    _verify_runtime_assets(manifest, runtime)
    topology = parallel_campaign.ParallelTopology.from_queue_topology(manifest["topology"])
    parallel_campaign.admit_parallel_resources(
        {"limits": {"minimum_free_bytes": topology.minimum_free_disk_bytes}},
        topology,
        work_root,
    )
    store = campaign.AwsCliStore()
    stager = campaign.AwsStager(store)
    checkpoint = work_root / "inputs/checkpoint" / str(manifest["cell"]["final_step"])
    workspace_root = work_root / "inputs/workspace"
    stager.stage_checkpoint(manifest["cell"], checkpoint)
    checkpoint_tree = checkpoint.parent / "checkpoint-tree.json"
    workspace = stager.stage_workspace(manifest["cell"], workspace_root)
    if workspace is None:
        raise RuntimeError("p5 action canary did not stage its workspace representation")
    config = work_root / "evidence/action-canary.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(CONFIG)
    records = build_lane_commands(
        manifest,
        source_root=source_root,
        runtime=runtime,
        checkpoint=checkpoint,
        workspace=workspace,
        config=config,
        work_root=work_root / "evidence",
    )
    _run_lane_processes(records, timeout_seconds=args.timeout_seconds)
    resource_drain = _audit_resources_drained(topology)
    _verify_staged_inputs_unchanged(
        manifest,
        checkpoint=checkpoint,
        checkpoint_tree=checkpoint_tree,
        workspace=workspace,
    )
    observed = audit_canary(
        records,
        checkpoint=checkpoint,
        workspace=workspace,
        topology=topology,
    )
    observed.update(resource_drain)
    publishable = work_root / "publishable-evidence"
    _build_publishable_evidence(
        manifest,
        observed=observed,
        config=config,
        destination=publishable,
    )
    archive = work_root / "action-canary-evidence.tgz"
    evidence_sha = _deterministic_archive(publishable, archive)
    evidence_uri = f"{manifest['evidence_root_s3']}/{evidence_sha}.tgz"
    store.put_file_once(archive, evidence_uri)
    evidence = {
        "uri": evidence_uri,
        "sha256": evidence_sha,
        "bytes": archive.stat().st_size,
    }
    claim = build_success_claim(manifest, observed=observed, evidence=evidence)
    store.put_bytes_once(_pretty(claim), manifest["claim_s3"])
    return claim


def run(args: argparse.Namespace) -> dict:
    previous_sigterm = None
    previous_sigint = None

    def interrupt(signum: int, _frame) -> None:
        raise CanarySignal(signum)

    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
        previous_sigint = signal.signal(signal.SIGINT, interrupt)
    try:
        return _run(args)
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--work-root", type=Path, required=True)
    value.add_argument("--policy-python", type=Path, required=True)
    value.add_argument("--vla-eval", type=Path, required=True)
    value.add_argument("--harness-src", type=Path, required=True)
    value.add_argument("--robomme-src", type=Path, required=True)
    value.add_argument("--maniskill-src", type=Path, required=True)
    value.add_argument("--openpi-src", type=Path, required=True)
    value.add_argument("--policy-site", type=Path, required=True)
    value.add_argument("--simulator-site", type=Path, required=True)
    value.add_argument("--upstream-root", type=Path, required=True)
    value.add_argument("--vision-encoder-home", type=Path, required=True)
    value.add_argument("--timeout-seconds", type=int, default=3_600)
    value.add_argument("--confirm-run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.confirm_run:
        raise SystemExit("p5 action canary is blocked without --confirm-run")
    claim = run(args)
    print(
        f"ROBOMME_P5_PARALLEL_ACTION_PREFLIGHT_OK id={claim['preflight_id']} "
        f"actions={claim['observed']['actions_executed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
