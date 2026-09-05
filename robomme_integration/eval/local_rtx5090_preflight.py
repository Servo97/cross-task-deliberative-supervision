#!/usr/bin/env python3
"""Inspect or seal the exact local two-RTX-5090 RoboMME evaluation runtime.

The default command is read-only and reports blockers.  ``--confirm-preflight`` is required before
the rendered reset and two-episode workspace action probe are executed or receipts are written.
This command never downloads cloud objects and never runs a scored episode; exact policy and
workspace checkpoints must be staged separately after approval.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from robomme_integration.eval import campaign
from robomme_integration.eval import local_rtx5090_campaign as local
from robomme_integration.eval.launch_p5_campaign import (
    UPSTREAM_COMMIT,
    UPSTREAM_CRITICAL_SHA256,
    VISION_BYTES,
    VISION_REVISION,
    VISION_S3,
    VISION_SHA256,
)
from robomme_integration.eval.launch_p5_preflight import RUNTIME_S3, RUNTIME_SHA
from robomme_integration.fleet.checkpoint import verify as verify_checkpoint_manifest
from robomme_integration.launch import OPENPI, OPENPI_SHA

MODULE = "robomme_integration.eval.local_rtx5090_preflight"
WORKSPACE_PROBE_ARM = "q3"
WORKSPACE_PROBE_TASK = "PickXtimes"
WORKSPACE_PROBE_EPISODES = 2
WORKSPACE_PROBE_CONFIG = b"""server:
  url: "ws://127.0.0.1:18200"
  timeout: 1200

output_dir: "./results/local_workspace_action_preflight"

benchmarks:
  - benchmark: "robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"
    subname: counting-pickxtimes-workspace-action-preflight
    episodes_per_task: 2
    max_steps: 1
    params:
      tasks: [PickXtimes]
      action_space: joint_angle
      dataset: test
      send_wrist_image: true
      send_state: true
      send_video_history: true
"""
PARALLEL_WORKSPACE_PROBE_EPISODES = 8
PARALLEL_WORKSPACE_PROBE_EPISODES_PER_LANE = 4
PARALLEL_WORKSPACE_PROBE_CONFIG = b"""server:
  url: "ws://127.0.0.1:18100"
  timeout: 1200

output_dir: "./results/local_parallel_workspace_action_preflight"

benchmarks:
  - benchmark: "robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"
    subname: counting-pickxtimes-parallel-workspace-action-preflight
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


class WorkspaceProbeSignal(BaseException):
    """Convert SIGTERM into the probe's fail-closed cooperative cleanup path."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"workspace action probe received signal {signum}")
        self.signum = signum


def runtime_paths(root: Path, vision_home: Path | None = None) -> dict[str, Path]:
    runtime = root / "runtime-v0.4.0"
    openpi = root / "openpi/ed923b2c"
    values = list((runtime / "env-v0.4.0/lib").glob("python*/site-packages"))
    if len(values) != 1:
        simulator_site = runtime / "env-v0.4.0/lib/python3.11/site-packages"
    else:
        simulator_site = values[0]
    return {
        "runtime_archive": root / "artifacts/robomme-eval-runtime-v0.4.0-60da89c3.tgz",
        "openpi_archive": root / "artifacts/openpi-ed923b2c.tgz",
        "base_openpi_archive": root / "artifacts/openpi-ed923b2c.tgz",
        "policy_python": openpi / ".venv/bin/python",
        "simulator_python": runtime / "env-v0.4.0/bin/python",
        "vla_eval": runtime / "env-v0.4.0/bin/vla-eval",
        "harness_src": runtime / "robomme-v0.4.0/src",
        "robomme_src": runtime / "robomme-benchmark-f2b540e6/src",
        "maniskill_src": runtime / "ManiSkill-07be6fbc",
        "openpi_src": openpi / "src",
        "policy_site": openpi / ".venv/lib/python3.11/site-packages",
        "simulator_site": simulator_site,
        "upstream_root": root / "official_reference/robomme_policy_learning",
        "vision_encoder_home": (
            vision_home if vision_home is not None else root / f"artifacts/vision/pi05/{VISION_REVISION}"
        ),
    }


def _sha(path: Path) -> str:
    return local._sha256(path)


def _run_text(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 120) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    ).stdout.strip()


def _pythonpath(paths: dict[str, Path], source_root: Path) -> str:
    # Match SubprocessEvaluator exactly: keep policy dependencies separate from the native
    # simulator stack, and make standard-ed923 OpenPI win over the official-reference checkout.
    return os.pathsep.join(
        str(path)
        for path in campaign.policy_pythonpath_entries(
            source_root=source_root,
            harness_src=paths["harness_src"],
            openpi_src=paths["openpi_src"],
            policy_site=paths["policy_site"],
            upstream_root=paths["upstream_root"],
            simulator_site=paths["simulator_site"],
        )
    )


def _simulator_pythonpath(paths: dict[str, Path], source_root: Path) -> str:
    """Keep OpenPI/JAX policy dependencies out of native simulator children."""
    return os.pathsep.join(
        str(path)
        for path in (
            source_root,
            paths["harness_src"],
            paths["robomme_src"],
            paths["maniskill_src"],
            paths["simulator_site"],
        )
    )


def _require_source_unchanged(source_root: Path, expected_sha256: str) -> None:
    actual = local.sanitized_source_sha256(source_root)
    if actual != expected_sha256:
        raise RuntimeError(
            "robomme_integration source changed during the rendered-reset preflight: "
            f"expected {expected_sha256}, got {actual}"
        )


def inspect(root: Path, paths: dict[str, Path]) -> list[str]:
    blockers = []
    files = (
        "runtime_archive",
        "openpi_archive",
        "policy_python",
        "simulator_python",
        "vla_eval",
    )
    directories = (
        "harness_src",
        "robomme_src",
        "maniskill_src",
        "openpi_src",
        "policy_site",
        "simulator_site",
        "upstream_root",
    )
    for name in files:
        if not paths[name].is_file():
            blockers.append(f"missing {name}: {paths[name]}")
    for name in directories:
        if not paths[name].is_dir():
            blockers.append(f"missing {name}: {paths[name]}")
    vision = paths["vision_encoder_home"] / "pi05_vision_encoder/siglip_params.pkl"
    if not vision.is_file():
        blockers.append(
            f"missing pi0.5 vision weights: {vision} (stage {VISION_S3}, {VISION_BYTES} bytes, sha256={VISION_SHA256})"
        )
    if paths["runtime_archive"].is_file() and _sha(paths["runtime_archive"]) != RUNTIME_SHA:
        blockers.append("cached RoboMME runtime archive has the wrong SHA-256")
    if paths["openpi_archive"].is_file() and _sha(paths["openpi_archive"]) != OPENPI_SHA:
        blockers.append("cached standard-ed923 OpenPI archive has the wrong SHA-256")
    if vision.is_file() and (vision.stat().st_size != VISION_BYTES or _sha(vision) != VISION_SHA256):
        blockers.append("cached pi0.5 vision weights have the wrong size or SHA-256")
    return blockers


def _critical(paths: dict[str, Path]) -> dict[str, str]:
    names = (
        "openpi_src/openpi/models/pi0.py",
        "openpi_src/openpi/models/pi0_config.py",
        "openpi_src/openpi/models/wsm_current_cond.py",
        "harness_src/vla_eval/benchmarks/robomme/benchmark.py",
    )
    return {name: local._critical_path_sha(paths, name) for name in names}


def _environment(paths: dict[str, Path], source_root: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(local.LOCAL_RENDER_ENVIRONMENT)
    value["PYTHONPATH"] = _pythonpath(paths, source_root)
    value["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return value


def _simulator_environment(paths: dict[str, Path], source_root: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(local.LOCAL_RENDER_ENVIRONMENT)
    value["PYTHONPATH"] = _simulator_pythonpath(paths, source_root)
    return value


def _probe_only() -> int:
    from robomme_integration.eval.benchmark import RoboMMEOfficialHistoryBenchmark

    async def run() -> dict:
        benchmark = RoboMMEOfficialHistoryBenchmark(
            tasks=["MoveCube"],
            dataset="test",
            max_steps=1,
            send_wrist_image=True,
            send_state=True,
            send_video_history=True,
        )
        task = {"name": "MoveCube", "env_id": "MoveCube", "episode_idx": 0}
        await benchmark.start_episode(task)
        observation = await benchmark.get_observation()
        history = observation.get("video_history", [])
        state_history = observation.get("video_state_history", [])
        if (
            not observation.get("episode_restart")
            or not history
            or not state_history
            or len(history) != len(state_history)
        ):
            raise RuntimeError(
                "official-history reset returned missing/unpaired demonstration video/state "
                f"history: frames={len(history)} states={len(state_history)}"
            )
        env = getattr(benchmark, "_env", None)
        if env is not None and hasattr(env, "close"):
            env.close()
        return {
            "rendered_reset": True,
            "observed_demo_frames": len(history),
            "observed_demo_states": len(state_history),
        }

    print(json.dumps(asyncio.run(run()), sort_keys=True))
    return 0


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-addressed local receipt collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _descendant_process_groups(root_pid: int) -> set[int]:
    """Snapshot descendant process groups so a hard timeout cannot orphan GPU children."""
    pending = [root_pid]
    seen = set()
    groups = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = (Path(f"/proc/{pid}/task/{pid}/children")).read_text().split()
            pending.extend(int(child) for child in children)
            groups.add(os.getpgid(pid))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    groups.discard(os.getpgrp())
    return groups


def _terminate_probe_process(process: subprocess.Popen, *, grace_seconds: int = 60) -> tuple[str, str]:
    """Gracefully trigger fleet cleanup, then kill every snapshotted child group if needed."""
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
    _kill_probe_groups(groups)
    return process.communicate() if result is None else result


def _kill_probe_groups(groups: set[int]) -> None:
    """Sweep groups captured before a supervisor had an opportunity to orphan children."""
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _request_probe_termination(process: subprocess.Popen) -> set[int]:
    """Signal a peer and return groups for a post-drain orphan sweep.

    The peer's owning worker remains solely responsible for ``communicate()``.  Capturing groups
    before SIGTERM matters because the supervisor can exit before its detached policy server.
    """
    if process.returncode is not None:
        return set()
    groups = _descendant_process_groups(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return groups


def _communicate_probe_process(process: subprocess.Popen, *, timeout_seconds: int) -> tuple[str, str, bool]:
    """Communicate with a probe while guaranteeing cleanup on timeout or operator interruption."""
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_probe_process(process)
        return stdout, stderr, True
    except BaseException as error:
        try:
            _terminate_probe_process(process)
        except Exception as cleanup_error:
            error.add_note(f"workspace probe cleanup also failed: {cleanup_error}")
        raise


def _tree_identity(path: Path, *, label: str) -> dict[str, Any]:
    """Bind every byte of a small staged checkpoint without trusting its directory name."""
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{label} is not a directory: {root}")
    if not root.name.isdigit():
        raise ValueError(f"{label} directory must be its numeric step: {root}")
    records = []
    for child in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if child.is_symlink() or (not child.is_dir() and not child.is_file()):
            raise ValueError(f"{label} contains an unsupported filesystem entry: {child}")
        if child.is_file():
            records.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": child.stat().st_size,
                    "sha256": _sha(child),
                }
            )
    if not records:
        raise ValueError(f"{label} has no files: {root}")
    manifest = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return {
        "path": str(root),
        "step": int(root.name),
        "files": records,
        "bytes": sum(record["bytes"] for record in records),
        "local_tree_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def _workspace_probe_lineage(queue_template: Path) -> dict[str, Any]:
    payload = queue_template.expanduser().resolve().read_bytes()
    queue = json.loads(payload)
    allowed_queue_ids = {local.LOCAL_RETRY_QUEUE_ID, local.LOCAL_PARALLEL_QUEUE_ID}
    if not isinstance(queue, dict) or queue.get("queue_id") not in allowed_queue_ids:
        raise ValueError("workspace probe requires a fresh unscored allowlisted local queue template")
    if "gates" in queue or "queue_manifest_sha256" in queue:
        raise ValueError("workspace probe requires an unsealed local template, not a prior queue")
    local._validate_exact_panel(queue, expected_queue_id=queue["queue_id"])
    matches = [
        cell
        for cell in queue.get("cells", [])
        if isinstance(cell, dict)
        and cell.get("task") == WORKSPACE_PROBE_TASK
        and cell.get("arm") == WORKSPACE_PROBE_ARM
    ]
    if len(matches) != 1:
        raise ValueError("workspace probe queue must contain exactly one PickXtimes/Q3 cell")
    cell = matches[0]
    workspace = cell.get("workspace")
    required_cell = (
        "cell_id",
        "run_id",
        "final_step",
        "scientific_spec_sha256",
        "run_manifest_sha256",
        "training_output_s3",
        "training_completion_claim_s3",
    )
    if any(not cell.get(name) for name in required_cell) or not isinstance(workspace, dict):
        raise ValueError("workspace probe Q3 cell has incomplete training/workspace lineage")
    required_workspace = (
        "step",
        "checkpoint_tree_sha256",
        "completion_sha256",
        "run_config_sha256",
        "best_sha256",
        "representation_s3",
    )
    if any(not workspace.get(name) for name in required_workspace):
        raise ValueError("workspace probe Q3 cell has incomplete representation lineage")
    return {
        "queue_id": queue["queue_id"],
        "queue_template_file_sha256": hashlib.sha256(payload).hexdigest(),
        "cell_id": cell["cell_id"],
        "run_id": cell["run_id"],
        "final_step": cell["final_step"],
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
        "run_manifest_sha256": cell["run_manifest_sha256"],
        "training_output_s3": cell["training_output_s3"],
        "training_completion_claim_s3": cell["training_completion_claim_s3"],
        "workspace": {name: workspace[name] for name in required_workspace},
    }


def _workspace_probe_inputs(
    policy_checkpoint: Path,
    workspace_checkpoint: Path,
    *,
    queue_template: Path,
    policy_tree_manifest: Path,
) -> dict[str, Any]:
    lineage = _workspace_probe_lineage(queue_template)
    policy = _tree_identity(policy_checkpoint, label="workspace-probe policy checkpoint")
    workspace = _tree_identity(workspace_checkpoint, label="workspace-probe workspace checkpoint")
    policy_files = {record["path"] for record in policy["files"]}
    workspace_files = {record["path"] for record in workspace["files"]}
    required_policy = {
        "assets/robomme/norm_stats.json",
        "params/_METADATA",
        "params/_sharding",
        "params/manifest.ocdbt",
    }
    required_workspace = {
        "WSM_RUN_CONFIG.json",
        "WSM_BEST.json",
        "WSM_GENERATION_COMPLETE.json",
        "state/_METADATA",
        "state/manifest.ocdbt",
    }
    if missing := sorted(required_policy - policy_files):
        raise ValueError(f"workspace-probe policy checkpoint is incomplete: {missing}")
    if missing := sorted(required_workspace - workspace_files):
        raise ValueError(f"workspace-probe workspace checkpoint is incomplete: {missing}")
    completion = json.loads((Path(workspace["path"]) / "WSM_GENERATION_COMPLETE.json").read_text(encoding="utf-8"))
    run_config = json.loads((Path(workspace["path"]) / "WSM_RUN_CONFIG.json").read_text(encoding="utf-8"))
    if completion.get("step") != workspace["step"]:
        raise ValueError("workspace-probe completion marker step differs from its numeric directory")
    if run_config.get("task") != WORKSPACE_PROBE_TASK:
        raise ValueError(
            f"workspace-probe representation is not bound to {WORKSPACE_PROBE_TASK}: {run_config.get('task')!r}"
        )
    if policy["step"] != lineage["final_step"]:
        raise ValueError("workspace-probe policy step differs from the intended Q3 queue cell")
    if workspace["step"] != lineage["workspace"]["step"]:
        raise ValueError("workspace-probe representation step differs from the intended Q3 queue cell")
    expected_tree_path = Path(policy["path"]).parent / "checkpoint-tree.json"
    tree_path = policy_tree_manifest.expanduser().resolve()
    if tree_path != expected_tree_path.resolve() or not tree_path.is_file():
        raise ValueError(f"workspace-probe policy tree manifest must be the stager-authenticated {expected_tree_path}")
    checkpoint_uri = f"{lineage['training_output_s3']}/deploy/{lineage['final_step']}"
    policy_tree_sha = verify_checkpoint_manifest(
        Path(policy["path"]),
        tree_path,
        expected_uri=checkpoint_uri,
    )
    if (
        campaign._legacy_workspace_tree_sha256(Path(workspace["path"]), expected_step=workspace["step"])
        != lineage["workspace"]["checkpoint_tree_sha256"]
    ):
        raise ValueError("workspace-probe representation tree differs from the intended Q3 queue cell")
    seals = {
        "completion_sha256": _sha(Path(workspace["path"]) / "WSM_GENERATION_COMPLETE.json"),
        "run_config_sha256": _sha(Path(workspace["path"]) / "WSM_RUN_CONFIG.json"),
        "best_sha256": _sha(Path(workspace["path"]) / "WSM_BEST.json"),
    }
    expected_seals = {
        name: lineage["workspace"][name] for name in ("completion_sha256", "run_config_sha256", "best_sha256")
    }
    if seals != expected_seals:
        raise ValueError("workspace-probe representation seals differ from the intended Q3 queue cell")
    policy["deploy_checkpoint_uri"] = checkpoint_uri
    policy["tree_manifest_path"] = str(tree_path)
    policy["deploy_tree_manifest_sha256"] = policy_tree_sha
    workspace["producer_tree_sha256"] = lineage["workspace"]["checkpoint_tree_sha256"]
    workspace["seals"] = seals
    return {
        "lineage": lineage,
        "policy_checkpoint": policy,
        "workspace_checkpoint": workspace,
    }


def _workspace_probe_command(
    *,
    source_root: Path,
    paths: dict[str, Path],
    policy_checkpoint: Path,
    workspace_checkpoint: Path,
    config: Path,
    output: Path,
    probe_id: str,
    parallel_lane: local.parallel_campaign.LaneSpec | None = None,
) -> list[str]:
    if parallel_lane is not None:
        topology = local.parallel_campaign.local_2x5090_topology()
        if parallel_lane not in topology.lanes:
            raise ValueError("parallel workspace probe lane is outside the sealed topology")
        gpus = str(parallel_lane.policy_gpu)
        simulator_gpus = str(parallel_lane.simulator_gpu)
        base_port = parallel_lane.port
        shards = parallel_lane.simulator_shards
        cpu_range = parallel_lane.cpu_range
        xla_memory_fraction = parallel_lane.xla_memory_fraction
    else:
        topology = None
        gpus = "0"
        simulator_gpus = "1"
        base_port = 18200
        shards = WORKSPACE_PROBE_EPISODES
        cpu_range = "0-127"
        xla_memory_fraction = 0.65
    command = [
        str(paths["policy_python"]),
        "-m",
        "robomme_integration.eval.launch_gpu_fleet",
        "--source-root",
        str(source_root),
        "--checkpoint",
        str(policy_checkpoint),
        "--arm",
        WORKSPACE_PROBE_ARM,
        "--scope",
        "single_task",
        "--task",
        WORKSPACE_PROBE_TASK,
        "--benchmark-config",
        str(config),
        "--vla-eval",
        str(paths["vla_eval"]),
        "--output-root",
        str(output),
        "--eval-id",
        probe_id,
        "--gpus",
        gpus,
        "--base-port",
        str(base_port),
        "--shards",
        str(shards),
        "--cpu-range",
        cpu_range,
        "--xla-memory-fraction",
        str(xla_memory_fraction),
        "--native-simulator",
        "--simulator-gpus",
        simulator_gpus,
        "--pin-native-cpus",
        "--workspace-checkpoint",
        str(workspace_checkpoint),
        "--upstream-root",
        str(paths["upstream_root"]),
        "--vision-encoder-home",
        str(paths["vision_encoder_home"]),
        "--ready-timeout-seconds",
        "1200",
    ]
    if parallel_lane is not None:
        command.extend(
            [
                "--native-shard-prewarm-seconds",
                str(parallel_lane.shard_prewarm_seconds),
                "--native-shard-stagger-seconds",
                str(parallel_lane.shard_stagger_seconds),
            ]
        )
    for entry in _simulator_pythonpath(paths, source_root).split(os.pathsep):
        command.extend(["--simulator-pythonpath", entry])
    return command


def _audit_workspace_action_probe(
    output: Path,
    *,
    policy_checkpoint: Path,
    workspace_checkpoint: Path,
    probe_id: str,
    parallel_lane: local.parallel_campaign.LaneSpec | None = None,
) -> dict[str, Any]:
    if parallel_lane is not None:
        topology = local.parallel_campaign.local_2x5090_topology()
        if parallel_lane not in topology.lanes:
            raise ValueError("parallel workspace audit lane is outside the sealed topology")
        episodes_expected = PARALLEL_WORKSPACE_PROBE_EPISODES_PER_LANE
        gpus_expected = [parallel_lane.policy_gpu]
        ports_expected = [parallel_lane.port]
        native_gpus_expected = [parallel_lane.simulator_gpu]
        gpu_by_shard_expected = [
            native_gpus_expected[shard % len(native_gpus_expected)] for shard in range(episodes_expected)
        ]
        server_urls_expected = [f"ws://127.0.0.1:{port}" for port in ports_expected]
        config_payload = PARALLEL_WORKSPACE_PROBE_CONFIG
    else:
        topology = None
        episodes_expected = WORKSPACE_PROBE_EPISODES
        gpus_expected = [0]
        ports_expected = [18200]
        native_gpus_expected = [1]
        gpu_by_shard_expected = [1, 1]
        server_urls_expected = ["ws://127.0.0.1:18200"]
        config_payload = WORKSPACE_PROBE_CONFIG
    supervisor_path = output / "supervisor.json"
    manifest_path = output / "eval/launch_manifest.json"
    server_log_paths = [
        output / f"server-gpu{gpu}-port{port}.log" for gpu, port in zip(gpus_expected, ports_expected, strict=True)
    ]
    if not (output / "COMPLETED").is_file():
        raise RuntimeError(f"workspace action probe did not complete: {output}")
    for path in (supervisor_path, manifest_path, *server_log_paths):
        if not path.is_file():
            raise RuntimeError(f"workspace action probe is missing evidence: {path}")
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checkpoint = str(policy_checkpoint.resolve())
    expected_workspace = str(workspace_checkpoint.resolve())
    if (
        supervisor.get("launcher_returncode") != 0
        or supervisor.get("failure") is not None
        or supervisor.get("arm") != WORKSPACE_PROBE_ARM
        or supervisor.get("task") != WORKSPACE_PROBE_TASK
        or supervisor.get("checkpoint") != expected_checkpoint
        or supervisor.get("gpus") != gpus_expected
        or supervisor.get("ports") != ports_expected
        or supervisor.get("shards") != episodes_expected
    ):
        raise RuntimeError(f"workspace action supervisor contract failed: {supervisor}")
    if parallel_lane is not None and (
        supervisor.get("cpu_range") != parallel_lane.cpu_range
        or supervisor.get("simulator_gpus") != [parallel_lane.simulator_gpu]
        or supervisor.get("pin_native_cpus") is not True
        or supervisor.get("xla_memory_fraction") != parallel_lane.xla_memory_fraction
        or supervisor.get("native_shard_prewarm_seconds") != parallel_lane.shard_prewarm_seconds
        or supervisor.get("native_shard_stagger_seconds") != parallel_lane.shard_stagger_seconds
    ):
        raise RuntimeError("parallel workspace action supervisor differs from the sealed lane resources")
    server_commands = supervisor.get("server_commands")
    if not isinstance(server_commands, list) or len(server_commands) != len(gpus_expected):
        raise RuntimeError("workspace action probe policy-server count differs from topology")
    workspace_flags = ("--args.workspace_checkpoint", "--workspace_checkpoint")
    for command in server_commands:
        workspace_flag = next(
            (flag for flag in workspace_flags if isinstance(command, list) and flag in command),
            None,
        )
        if (
            workspace_flag is None
            or command[command.index(workspace_flag) + 1] != expected_workspace
            or expected_checkpoint not in command
        ):
            raise RuntimeError("workspace action probe server did not receive the exact numeric checkpoint")
    audit = manifest.get("episode_audit")
    native = manifest.get("native")
    commands = manifest.get("commands")
    if (
        manifest.get("eval_id") != probe_id
        or manifest.get("config_sha256") != hashlib.sha256(config_payload).hexdigest()
        or manifest.get("shards") != episodes_expected
        or manifest.get("recording_mode") != "sqlite"
        or manifest.get("server_urls") != server_urls_expected
        or manifest.get("returncodes") != [0] * episodes_expected
        or not isinstance(native, dict)
        or native.get("gpus") != native_gpus_expected
        or native.get("gpu_by_shard") != gpu_by_shard_expected
        or not isinstance(commands, list)
        or len(commands) != episodes_expected
        or any(
            server_urls_expected[shard % len(server_urls_expected)] not in command
            for shard, command in enumerate(commands)
        )
        or not isinstance(audit, dict)
        or audit.get("episodes") != episodes_expected
        or audit.get("harness_failures") != []
    ):
        raise RuntimeError(f"workspace action harness contract failed: {manifest}")
    materialized = manifest.get("materialized_results")
    if not isinstance(materialized, list) or len(materialized) != 1:
        raise RuntimeError("workspace action probe must produce exactly one aggregate")
    result_paths = []
    episodes = []
    for record in materialized:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("workspace action result record is malformed")
        path = Path(record["path"]).resolve()
        if (
            not path.is_relative_to((output / "eval").resolve())
            or not path.name.endswith("_aggregate.json")
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha(path) != record.get("sha256")
        ):
            raise RuntimeError(f"workspace action result identity mismatch: {path}")
        result_paths.append(path)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for task in payload.get("tasks", []):
                if isinstance(task, dict) and task.get("task") == WORKSPACE_PROBE_TASK:
                    episodes.extend(item for item in task.get("episodes", []) if isinstance(item, dict))
    episode_indices = sorted(episode.get("episode_idx") for episode in episodes)
    episode_ids = sorted(episode.get("episode_id") for episode in episodes)
    expected_episode_ids = list(range(episodes_expected))
    if (
        len(episodes) != episodes_expected
        or episode_indices != expected_episode_ids
        or episode_ids != expected_episode_ids
        or any(
            episode.get("failure_reason")
            or episode.get("steps") != 1
            or not isinstance(episode.get("metrics", {}).get("success"), bool)
            for episode in episodes
        )
    ):
        raise RuntimeError(f"workspace action probe did not execute exactly one action per episode: {episodes}")
    server_log_records = []
    for path in server_log_paths:
        server_log = path.read_text(encoding="utf-8", errors="replace")
        loaded = server_log.find(f"Loaded arm={WORKSPACE_PROBE_ARM}")
        serving = server_log.find("Starting server on")
        inferred = server_log.find(f"inference arm={WORKSPACE_PROBE_ARM}")
        if (
            server_log.count(f"Loaded arm={WORKSPACE_PROBE_ARM}") != 1
            or loaded < 0
            or serving < 0
            or inferred < 0
            or not loaded < serving < inferred
        ):
            raise RuntimeError("workspace action probe did not load-before-ready and produce an action")
        if "Traceback (most recent call last)" in server_log or "AssertionError" in server_log:
            raise RuntimeError("workspace action policy log contains an initialization/inference failure")
        server_log_records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    combined_server_log_sha = server_log_records[0]["sha256"]
    result = {
        "probe_id": probe_id,
        "arm": WORKSPACE_PROBE_ARM,
        "task": WORKSPACE_PROBE_TASK,
        "policy_servers": len(gpus_expected),
        "concurrent_native_shards": episodes_expected,
        "episodes": episodes_expected,
        "actions_executed": sum(int(episode["steps"]) for episode in episodes),
        "episode_indices": episode_indices,
        "harness_failures": 0,
        "load_completed_before_readiness": True,
        "supervisor_sha256": _sha(supervisor_path),
        "launch_manifest_sha256": _sha(manifest_path),
        "server_log_sha256": combined_server_log_sha,
        "server_logs": server_log_records,
        "materialized_results": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in result_paths
        ],
    }
    if parallel_lane is not None:
        cpu_start, cpu_end = (int(value) for value in parallel_lane.cpu_range.split("-", 1))
        cpu_count = cpu_end - cpu_start + 1
        base, extra = divmod(cpu_count, parallel_lane.simulator_shards)
        expected_partitions = []
        cursor = cpu_start
        for shard in range(parallel_lane.simulator_shards):
            width = base + int(shard < extra)
            expected_partitions.append(f"{cursor}-{cursor + width - 1}")
            cursor += width
        result.update(
            lane=parallel_lane.as_dict(),
        )
        if (
            native.get("cpu_partitions") != expected_partitions
            or native.get("launch_waves") != [[0], [1], [2], [3]]
            or native.get("shard_prewarm_seconds") != parallel_lane.shard_prewarm_seconds
            or native.get("shard_stagger_seconds") != parallel_lane.shard_stagger_seconds
            or native.get("prewarm_marker") != f"inference arm={WORKSPACE_PROBE_ARM}"
            or len(native.get("prewarm_logs", [])) != 1
            or any(
                not isinstance(command, list) or command[:3] != ["taskset", "--cpu-list", expected_partitions[shard]]
                for shard, command in enumerate(commands)
            )
        ):
            raise RuntimeError("parallel workspace action probe did not use staggered JIT prewarm")
    return result


def _run_workspace_action_probe(
    root: Path,
    source_root: Path,
    paths: dict[str, Path],
    *,
    policy_checkpoint: Path,
    workspace_checkpoint: Path,
    queue_template: Path,
    policy_tree_manifest: Path,
    source_sha256: str,
    runtime_fingerprint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _workspace_probe_inputs(
        policy_checkpoint,
        workspace_checkpoint,
        queue_template=queue_template,
        policy_tree_manifest=policy_tree_manifest,
    )
    parallel_probe = inputs["lineage"]["queue_id"] == local.LOCAL_PARALLEL_QUEUE_ID
    episodes = PARALLEL_WORKSPACE_PROBE_EPISODES if parallel_probe else WORKSPACE_PROBE_EPISODES
    config_payload = PARALLEL_WORKSPACE_PROBE_CONFIG if parallel_probe else WORKSPACE_PROBE_CONFIG
    identity = {
        "schema_version": 1,
        "source_tree_sha256": source_sha256,
        "arm": WORKSPACE_PROBE_ARM,
        "task": WORKSPACE_PROBE_TASK,
        "episodes": episodes,
        "lineage": inputs["lineage"],
        "policy_checkpoint": inputs["policy_checkpoint"],
        "workspace_checkpoint": inputs["workspace_checkpoint"],
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "runtime_fingerprint": runtime_fingerprint,
    }
    if parallel_probe:
        identity["parallel_topology"] = local.parallel_campaign.local_2x5090_topology().as_queue_topology()
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    probe_kind = "q3-parallel-v1" if parallel_probe else "q3-action-v1"
    probe_id = f"local5090-unscored-{probe_kind}-{digest[:20]}"
    work = root / "campaign-runtime/preflight-work" / probe_id
    config = work / "workspace-action.yaml"
    _write_once(config, config_payload)
    topology = local.parallel_campaign.local_2x5090_topology() if parallel_probe else None
    lanes = topology.lanes if topology is not None else (None,)
    lane_runs = []
    for lane in lanes:
        lane_id = lane.lane_id if lane is not None else "legacy"
        lane_root = work / "lanes" / lane_id if lane is not None else work
        lane_runs.append(
            {
                "lane": lane,
                "lane_id": lane_id,
                "output": lane_root / "output",
                "launcher_log": lane_root / "launcher.log",
                "eval_id": f"{probe_id}-{lane_id}" if lane is not None else probe_id,
            }
        )
    completed = [(run["output"] / "COMPLETED").is_file() for run in lane_runs]
    if any(completed) and not all(completed):
        raise RuntimeError(f"parallel workspace probe has only a partial completed lane set: {work}")
    if not all(completed):
        executor: concurrent.futures.ThreadPoolExecutor | None = None
        futures: dict[concurrent.futures.Future, dict[str, Any]] = {}
        captured_groups: set[int] = set()
        signal_seen = False

        def interrupt_probe(signum: int, _frame) -> None:
            nonlocal signal_seen
            if signal_seen:
                return
            signal_seen = True
            raise WorkspaceProbeSignal(signum)

        previous_sigterm = None
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.signal(signal.SIGTERM, interrupt_probe)
        try:
            for run in lane_runs:
                output = run["output"]
                if output.exists() and any(output.iterdir()):
                    raise RuntimeError(f"refusing to overwrite prior workspace probe evidence: {output}")
                run["command"] = _workspace_probe_command(
                    source_root=source_root,
                    paths=paths,
                    policy_checkpoint=Path(inputs["policy_checkpoint"]["path"]),
                    workspace_checkpoint=Path(inputs["workspace_checkpoint"]["path"]),
                    config=config,
                    output=output,
                    probe_id=run["eval_id"],
                    parallel_lane=run["lane"],
                )
                run["process"] = subprocess.Popen(
                    run["command"],
                    cwd=source_root,
                    env=_environment(paths, source_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(lane_runs))
            for run in lane_runs:
                future = executor.submit(
                    _communicate_probe_process,
                    run["process"],
                    timeout_seconds=1800,
                )
                run["future"] = future
                futures[future] = run
            for future in concurrent.futures.as_completed(futures):
                run = futures[future]
                stdout, stderr, timed_out = future.result()
                process = run["process"]
                launcher_log = (
                    f"returncode={process.returncode}\ntimeout_seconds=1800\n"
                    f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
                ).encode()
                _write_once(run["launcher_log"], launcher_log)
                if timed_out or process.returncode:
                    status = "timed out" if timed_out else f"failed with {process.returncode}"
                    raise RuntimeError(f"workspace action probe lane {run['lane_id']} {status}; evidence={work}")
            executor.shutdown(wait=True)
            executor = None
        except BaseException as error:
            for run in lane_runs:
                process = run.get("process")
                if process is not None:
                    captured_groups.update(_request_probe_termination(process))
            for future in futures:
                future.cancel()
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            for run in lane_runs:
                process = run.get("process")
                future = run.get("future")
                if process is None or (future is not None and not future.cancelled()):
                    continue
                try:
                    _terminate_probe_process(process)
                except Exception as cleanup_error:
                    error.add_note(f"workspace probe {run['lane_id']} direct cleanup failed: {cleanup_error}")
            running = [future for future in futures if not future.cancelled()]
            if running:
                _done, pending = concurrent.futures.wait(running, timeout=60)
                if pending:
                    _kill_probe_groups(captured_groups)
                    concurrent.futures.wait(pending, timeout=60)
            _kill_probe_groups(captured_groups)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)

    lane_audits = []
    for run in lane_runs:
        lane_audit = _audit_workspace_action_probe(
            run["output"],
            policy_checkpoint=Path(inputs["policy_checkpoint"]["path"]),
            workspace_checkpoint=Path(inputs["workspace_checkpoint"]["path"]),
            probe_id=run["eval_id"],
            parallel_lane=run["lane"],
        )
        lane_audit["launcher_log_sha256"] = _sha(run["launcher_log"])
        lane_audits.append(lane_audit)
    if parallel_probe:

        def combined_digest(field: str) -> str:
            records = [
                {"lane_id": run["lane_id"], field: lane_audit[field]}
                for run, lane_audit in zip(lane_runs, lane_audits, strict=True)
            ]
            return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        server_logs = [record for item in lane_audits for record in item["server_logs"]]
        materialized_results = [record for item in lane_audits for record in item["materialized_results"]]
        audit = {
            "probe_id": probe_id,
            "arm": WORKSPACE_PROBE_ARM,
            "task": WORKSPACE_PROBE_TASK,
            "execution_mode": local.parallel_campaign.PARALLEL_EXECUTION_MODE,
            "parallel_topology_sha256": topology.as_queue_topology()["parallel_topology_sha256"],
            "parallel_lanes": len(topology.lanes),
            "policy_servers": len(topology.lanes),
            "native_shards_per_lane": topology.lanes[0].simulator_shards,
            "concurrent_native_shards": sum(lane.simulator_shards for lane in topology.lanes),
            "xla_memory_fraction": topology.lanes[0].xla_memory_fraction,
            "shard_prewarm_seconds": topology.lanes[0].shard_prewarm_seconds,
            "shard_stagger_seconds": topology.lanes[0].shard_stagger_seconds,
            "episodes": sum(item["episodes"] for item in lane_audits),
            "actions_executed": sum(item["actions_executed"] for item in lane_audits),
            "episode_indices": sorted(episode for item in lane_audits for episode in item["episode_indices"]),
            "harness_failures": sum(item["harness_failures"] for item in lane_audits),
            "load_completed_before_readiness": all(item["load_completed_before_readiness"] for item in lane_audits),
            "supervisor_sha256": combined_digest("supervisor_sha256"),
            "launch_manifest_sha256": combined_digest("launch_manifest_sha256"),
            "server_log_sha256": hashlib.sha256(
                json.dumps(server_logs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "launcher_log_sha256": combined_digest("launcher_log_sha256"),
            "server_logs": server_logs,
            "materialized_results": materialized_results,
            "lane_evidence": lane_audits,
        }
    else:
        audit = lane_audits[0]
    after = _workspace_probe_inputs(
        policy_checkpoint,
        workspace_checkpoint,
        queue_template=queue_template,
        policy_tree_manifest=policy_tree_manifest,
    )
    if after != inputs:
        raise RuntimeError("workspace action probe checkpoint bytes changed during execution")
    audit["probe_identity_sha256"] = digest
    audit["runtime_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(runtime_fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return inputs, audit


def seal_runtime(
    root: Path,
    source_root: Path,
    paths: dict[str, Path],
    *,
    workspace_probe_policy_checkpoint: Path,
    workspace_probe_checkpoint: Path,
    workspace_probe_queue_template: Path,
    workspace_probe_policy_tree_manifest: Path,
) -> dict:
    blockers = inspect(root, paths)
    if blockers:
        raise RuntimeError("local runtime is not ready:\n- " + "\n- ".join(blockers))
    inventory = local._gpu_inventory()
    if [record["index"] for record in inventory] != [0, 1] or any(
        record["name"] != "NVIDIA GeForce RTX 5090" for record in inventory
    ):
        raise RuntimeError(f"preflight requires exact two-RTX-5090 inventory, got {inventory}")
    local._gpu_idle()

    env = _environment(paths, source_root)
    smoke = _run_text(
        [
            str(paths["policy_python"]),
            "-c",
            (
                "import anyio,json,jax,numpy,openpi,torch,vla_eval; "
                "import robomme_integration.eval.execution_model_server as server; "
                "from importlib.metadata import version; from pathlib import Path; "
                "print(json.dumps({'devices':jax.device_count(),"
                "'openpi':str(Path(openpi.__file__).resolve()),"
                "'jax':str(Path(jax.__file__).resolve()),"
                "'numpy':str(Path(numpy.__file__).resolve()),"
                "'torch':str(Path(torch.__file__).resolve()),"
                "'vla_eval':str(Path(vla_eval.__file__).resolve()),"
                "'execution_model_server':str(Path(server.__file__).resolve()),"
                "'anyio':str(Path(anyio.__file__).resolve()),"
                "'versions':{'jax':jax.__version__,'numpy':numpy.__version__,"
                "'torch':torch.__version__,'anyio':version('anyio'),"
                "'vla_eval':version('vla-eval')}}))"
            ),
        ],
        env=env,
        timeout=180,
    )
    smoke_record = json.loads(smoke.splitlines()[-1])
    if smoke_record.get("devices") != 2:
        raise RuntimeError(f"policy preflight did not see two JAX devices: {smoke_record}")
    if not Path(smoke_record["openpi"]).is_relative_to(paths["openpi_src"]):
        raise RuntimeError("policy preflight imported OpenPI outside the sealed standard-ed923 source")
    for module in ("jax", "numpy", "torch"):
        if not Path(smoke_record[module]).is_relative_to(paths["policy_site"]):
            raise RuntimeError(f"policy preflight imported {module} outside the sealed standard-ed923 environment")
    if not Path(smoke_record["vla_eval"]).is_relative_to(paths["harness_src"]):
        raise RuntimeError("policy preflight imported vla_eval outside the sealed harness source")
    if not Path(smoke_record["execution_model_server"]).is_relative_to(source_root):
        raise RuntimeError("policy preflight imported the execution server outside the sealed source")
    if not Path(smoke_record["anyio"]).is_relative_to(paths["simulator_site"]):
        raise RuntimeError("policy preflight imported anyio outside the sealed runtime fallback")
    if smoke_record.get("versions") != local.POLICY_IMPORT_VERSIONS:
        raise RuntimeError(f"policy import versions drifted: {smoke_record.get('versions')}")
    server_help = _run_text(
        [str(paths["policy_python"]), "-m", "robomme_integration.eval.execution_model_server", "--help"],
        env=env,
        timeout=180,
    )
    if "--args.checkpoint CHECKPOINT" not in server_help or "--args.arm ARM" not in server_help:
        raise RuntimeError("policy execution-server CLI is incomplete")
    smoke_record["server_help_sha256"] = hashlib.sha256((server_help + "\n").encode()).hexdigest()

    simulator_python = paths["simulator_python"]
    if not simulator_python.is_file():
        raise RuntimeError(f"sealed simulator Python is absent: {simulator_python}")
    simulator_env = _simulator_environment(paths, source_root)
    simulator_smoke = _run_text(
        [
            str(simulator_python),
            "-c",
            (
                "import importlib.util,json,mani_skill,mplib,numpy,robomme,sapien,torch,vla_eval; "
                "from importlib.metadata import version; from pathlib import Path; "
                "spec=importlib.util.find_spec('openpi'); "
                "print(json.dumps({'openpi':None if spec is None else str(Path(spec.origin).resolve()),"
                "'numpy':str(Path(numpy.__file__).resolve()),"
                "'torch':str(Path(torch.__file__).resolve()),"
                "'mplib':str(Path(mplib.__file__).resolve()),"
                "'sapien':str(Path(sapien.__file__).resolve()),"
                "'vla_eval':str(Path(vla_eval.__file__).resolve()),"
                "'robomme':str(Path(robomme.__file__).resolve()),"
                "'mani_skill':str(Path(mani_skill.__file__).resolve()),"
                "'versions':{'numpy':numpy.__version__,'torch':torch.__version__,"
                "'mplib':version('mplib'),'sapien':version('sapien'),"
                "'vla_eval':version('vla-eval')}}))"
            ),
        ],
        env=simulator_env,
        timeout=180,
    )
    simulator_record = json.loads(simulator_smoke.splitlines()[-1])
    if simulator_record.get("openpi") is not None:
        raise RuntimeError("native simulator preflight leaked the OpenPI policy package")
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
        if not Path(simulator_record[module]).is_relative_to(expected_root):
            raise RuntimeError(f"native simulator imported {module} outside its sealed simulator source")
    if simulator_record.get("versions") != local.SIMULATOR_IMPORT_VERSIONS:
        raise RuntimeError(f"native simulator import versions drifted: {simulator_record.get('versions')}")
    source_sha = local.sanitized_source_sha256(source_root)
    distribution_inventory = local._distribution_inventory(paths["policy_python"])
    python_version = _run_text(
        [str(paths["policy_python"]), "-c", "import platform; print(platform.python_version())"],
        env=env,
    )
    executables = {
        "policy_python": local._python_identity(paths["policy_python"]),
        "simulator_python": local._python_identity(paths["simulator_python"]),
        "vla_eval": local._vla_eval_identity(paths["vla_eval"], paths["simulator_python"]),
    }
    critical = _critical(paths)
    runtime_fingerprint = {
        "source_tree_sha256": source_sha,
        "gpu_inventory": inventory,
        "runtime_archive_sha256": _sha(paths["runtime_archive"]),
        "openpi_archive_sha256": _sha(paths["openpi_archive"]),
        "openpi_distribution_inventory_sha256": hashlib.sha256(distribution_inventory).hexdigest(),
        "python_version": python_version,
        "executables": executables,
        "critical_file_sha256": critical,
        "render_environment": dict(local.LOCAL_RENDER_ENVIRONMENT),
        "policy_import_contract": smoke_record,
        "simulator_import_contract": simulator_record,
        "upstream": {"commit": UPSTREAM_COMMIT, "critical_sha256": UPSTREAM_CRITICAL_SHA256},
        "vision": {
            "revision": VISION_REVISION,
            "sha256": VISION_SHA256,
            "bytes": VISION_BYTES,
        },
    }
    probe_text = _run_text(
        [str(simulator_python), "-m", MODULE, "--probe-only"],
        env=simulator_env,
        timeout=900,
    )
    _require_source_unchanged(source_root, source_sha)
    observed = json.loads(probe_text.splitlines()[-1])
    if observed != {
        "observed_demo_frames": observed.get("observed_demo_frames"),
        "observed_demo_states": observed.get("observed_demo_states"),
        "rendered_reset": True,
    }:
        raise RuntimeError(f"unexpected rendered-reset probe output: {observed}")
    if (
        not isinstance(observed["observed_demo_frames"], int)
        or isinstance(observed["observed_demo_frames"], bool)
        or observed["observed_demo_frames"] < 1
        or observed["observed_demo_states"] != observed["observed_demo_frames"]
    ):
        raise RuntimeError("rendered-reset probe observed missing/unpaired demonstration state history")
    workspace_probe_inputs, workspace_action_probe = _run_workspace_action_probe(
        root,
        source_root,
        paths,
        policy_checkpoint=workspace_probe_policy_checkpoint,
        workspace_checkpoint=workspace_probe_checkpoint,
        queue_template=workspace_probe_queue_template,
        policy_tree_manifest=workspace_probe_policy_tree_manifest,
        source_sha256=source_sha,
        runtime_fingerprint=runtime_fingerprint,
    )
    _require_source_unchanged(source_root, source_sha)

    contract = {
        "runtime": {"uri": RUNTIME_S3, "sha256": RUNTIME_SHA, "local_sha256": _sha(paths["runtime_archive"])},
        "openpi": {"uri": OPENPI, "sha256": OPENPI_SHA, "local_sha256": _sha(paths["openpi_archive"])},
        "base_environment": {
            "uri": OPENPI,
            "sha256": OPENPI_SHA,
            "python_version": python_version,
            "uv_lock_sha256": _sha(paths["openpi_src"].parent / "uv.lock"),
            "distribution_inventory_sha256": hashlib.sha256(distribution_inventory).hexdigest(),
        },
        "upstream": {"commit": UPSTREAM_COMMIT, "critical_sha256": UPSTREAM_CRITICAL_SHA256},
        "vision": {
            "uri": VISION_S3,
            "revision": VISION_REVISION,
            "sha256": VISION_SHA256,
            "bytes": VISION_BYTES,
        },
        "paths": {name: str(local._contract_path(name, path)) for name, path in paths.items()},
        "executables": executables,
        "critical_file_sha256": critical,
        "render_environment": dict(local.LOCAL_RENDER_ENVIRONMENT),
        "import_contract": {"policy": smoke_record, "simulator": simulator_record},
    }
    # Re-run the same materialized-runtime verifier used at scored execution before blessing the
    # rendered reset.  This catches a dirty upstream tree or Python environment drift now, rather
    # than publishing a receipt that can never pass the campaign gate.
    local._verify_materialized_contract(contract, paths)
    scientific = {
        "schema_version": 1,
        "kind": local.LOCAL_PREFLIGHT_KIND,
        "source_tree_sha256": source_sha,
        "probe": {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
            "observed_demo_frames": observed["observed_demo_frames"],
            "observed_demo_states": observed["observed_demo_states"],
            "workspace_action": {
                "unscored": True,
                "inputs": workspace_probe_inputs,
                **workspace_action_probe,
            },
        },
        "runtime_contract": contract,
        "infrastructure": {
            "provider": "local_workstation",
            "accelerator": local.LOCAL_ACCELERATOR,
            "gpu_inventory": inventory,
        },
    }
    scientific_sha = hashlib.sha256(json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    preflight = {**scientific, "preflight_id": f"local5090-native-eval-v1-{scientific_sha[:20]}"}
    preflight["manifest_sha256"] = local._seal_preflight(preflight)
    preflight["status"] = "native_render_reset_passed"
    preflight_payload = (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode()
    preflight_sha = hashlib.sha256(preflight_payload).hexdigest()
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
    receipt_payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    _require_source_unchanged(source_root, source_sha)
    output = root / "campaign-runtime/receipts"
    preflight_path = output / f"preflight-{preflight_sha}.json"
    receipt_path = output / f"runtime-{receipt_sha}.json"
    _write_once(preflight_path, preflight_payload)
    _write_once(receipt_path, receipt_payload)
    return {
        "preflight": str(preflight_path),
        "preflight_file_sha256": preflight_sha,
        "runtime_receipt": str(receipt_path),
        "runtime_receipt_file_sha256": receipt_sha,
        "source_tree_sha256": source_sha,
        "observed_demo_frames": observed["observed_demo_frames"],
        "observed_demo_states": observed["observed_demo_states"],
        "workspace_action_probe_id": workspace_action_probe["probe_id"],
        "workspace_action_probe_harness_failures": workspace_action_probe["harness_failures"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=local.DEFAULT_LOCAL_ROOT)
    value.add_argument("--source-root", type=Path, default=local.REPO_ROOT)
    value.add_argument("--vision-encoder-home", type=Path)
    value.add_argument(
        "--workspace-probe-policy-checkpoint",
        type=Path,
        help="exact staged numeric Q3 Pi checkpoint used only by the unscored action probe",
    )
    value.add_argument(
        "--workspace-probe-checkpoint",
        type=Path,
        help="exact staged numeric PickXtimes workspace checkpoint used only by the unscored action probe",
    )
    value.add_argument(
        "--workspace-probe-queue-template",
        type=Path,
        help="unsealed local5090 queue whose exact PickXtimes/Q3 lineage is probed",
    )
    value.add_argument(
        "--workspace-probe-policy-tree-manifest",
        type=Path,
        help="stager-authenticated checkpoint-tree.json adjacent to the numeric Q3 policy checkpoint",
    )
    value.add_argument("--confirm-preflight", action="store_true")
    value.add_argument("--probe-only", action="store_true", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.probe_only:
        return _probe_only()
    root = args.root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    vision = args.vision_encoder_home.expanduser().resolve() if args.vision_encoder_home else None
    paths = runtime_paths(root, vision)
    blockers = inspect(root, paths)
    if not args.confirm_preflight:
        print(
            json.dumps(
                {
                    "ready_for_preflight": not blockers,
                    "blockers": blockers,
                    "source_root": str(source_root),
                    "paths": {name: str(path) for name, path in paths.items()},
                    "next": "obtain approval, then pass --confirm-preflight"
                    if not blockers
                    else "stage exact missing assets first",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not blockers else 2
    required_probe_args = (
        args.workspace_probe_policy_checkpoint,
        args.workspace_probe_checkpoint,
        args.workspace_probe_queue_template,
        args.workspace_probe_policy_tree_manifest,
    )
    if any(value is None for value in required_probe_args):
        raise SystemExit(
            "--confirm-preflight requires all --workspace-probe-{policy-checkpoint,checkpoint,"
            "queue-template,policy-tree-manifest} arguments"
        )
    print(
        json.dumps(
            seal_runtime(
                root,
                source_root,
                paths,
                workspace_probe_policy_checkpoint=args.workspace_probe_policy_checkpoint,
                workspace_probe_checkpoint=args.workspace_probe_checkpoint,
                workspace_probe_queue_template=args.workspace_probe_queue_template,
                workspace_probe_policy_tree_manifest=args.workspace_probe_policy_tree_manifest,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
