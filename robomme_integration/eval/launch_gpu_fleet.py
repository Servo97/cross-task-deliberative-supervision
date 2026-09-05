#!/usr/bin/env python3
"""Run one fixed-50 RoboMME arm across several GPU policy servers and CPU simulator shards."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from robomme_integration.training.arms import ARM_IDS, OFFICIAL_RECIPE_LEROBOT_ARM
from robomme_integration.training.single_task import TASK_EPISODES

WORKSPACE_ARMS = frozenset(
    {
        "q1",
        "q3",
        "wsm_cfg",
        "wsm_tanh",
        "wsm_d8",
        "wsm_d8_drop05",
        "wsm_d16",
        "wsm_d16_drop05",
        "gdn8_jepa_l01_k1",
        "ptrm",
        "v4_wsm_tanh",
        "v4_wsm_cfg",
        "v4_wsm_gdn8_drop00",
        "v4_wsm_gdn8_drop02",
        "v4_wsm_gdn16_drop00",
        "v4_wsm_gdn16_drop02",
        "v4_gdn8_jepa_visreg_l01_k1",
        "v4_cfg_jepa_visreg_l01_k1",
        "v4_ptrm",
    }
)
V4_CFG_EVAL_SCALES = frozenset({0.5, 1.0, 1.5, 2.0})
# ``official_recipe_lerobot`` has a distinct two-view model/data contract and all-16 checkpoint
# lineage.  It must use a dedicated evaluator rather than being admitted to the generic project
# server (whose S0-shaped serving config would be scientifically ambiguous).
EVALUABLE_ARMS = tuple(arm for arm in ARM_IDS if arm not in {"q0v", "q2v", OFFICIAL_RECIPE_LEROBOT_ARM})


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _terminate_group(pid: int, timeout: float = 30.0) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _server_flag(help_text: str, name: str) -> str:
    for candidate in (f"--args.{name}", f"--{name}"):
        if candidate in help_text:
            return candidate
    raise RuntimeError(f"policy-server CLI exposes no flag for {name!r}")


def _wait_for_server(process: subprocess.Popen, port: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server on port {port} exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"policy server on port {port} was not ready: {last_error}")


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must be comma-separated integers") from error
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("GPU list must be nonempty, unique, and nonnegative")
    return gpus


def _build_server_command(args: argparse.Namespace, help_text: str, port: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robomme_integration.eval.execution_model_server",
        _server_flag(help_text, "checkpoint"),
        str(args.checkpoint),
        _server_flag(help_text, "arm"),
        args.arm,
    ]
    if args.arm in WORKSPACE_ARMS:
        command.extend(
            [
                _server_flag(help_text, "workspace_checkpoint"),
                str(args.workspace_checkpoint),
                _server_flag(help_text, "upstream_root"),
                str(args.upstream_root),
                _server_flag(help_text, "vision_encoder_home"),
                str(args.vision_encoder_home),
                _server_flag(help_text, "cfg_guidance_scale"),
                str(args.cfg_guidance_scale),
            ]
        )
        if args.scope == "multitask":
            command.extend(
                [
                    _server_flag(help_text, "workspace_index"),
                    str(args.workspace_index),
                ]
            )
        if args.arm in {"ptrm", "v4_ptrm"}:
            command.extend(
                [
                    _server_flag(help_text, "ptrm_eval_k"),
                    str(args.ptrm_eval_k),
                    _server_flag(help_text, "ptrm_eval_sigma"),
                    str(args.ptrm_eval_sigma),
                    _server_flag(help_text, "ptrm_eval_select"),
                    args.ptrm_eval_select,
                ]
            )
    command.extend(
        [
            _server_flag(help_text, "task_name"),
            args.task,
            _server_flag(help_text, "model_seed"),
            "7",
            _server_flag(help_text, "chunk_size"),
            "10",
            _server_flag(help_text, "max_batch_size"),
            "8",
            "--port",
            str(port),
        ]
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=EVALUABLE_ARMS)
    parser.add_argument("--scope", choices=("single_task", "multitask"), default="single_task")
    parser.add_argument("--task", choices=tuple(TASK_EPISODES))
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--vla-eval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--gpus", type=_parse_gpus, default=(0, 1, 2, 3))
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--cpu-range", default="0-191")
    parser.add_argument("--xla-memory-fraction", type=float, default=0.90)
    parser.add_argument("--container-image", default="wsm/robomme-cpu-runtime:ubuntu22")
    parser.add_argument(
        "--native-simulator",
        action="store_true",
        help="run vla-eval --no-docker after the p5 native-render preflight has passed",
    )
    parser.add_argument(
        "--simulator-pythonpath",
        action="append",
        type=Path,
        default=[],
        help="exact native-simulator PYTHONPATH entry; repeat to add entries",
    )
    parser.add_argument(
        "--simulator-gpus",
        type=_parse_gpus,
        default=(),
        help="GPU IDs assigned round-robin to native simulator shards",
    )
    parser.add_argument("--pin-native-cpus", action="store_true")
    parser.add_argument("--container-user", default="2001:2001")
    parser.add_argument("--workspace-checkpoint", type=Path)
    parser.add_argument("--workspace-index", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--vision-encoder-home", type=Path)
    parser.add_argument("--cfg-guidance-scale", type=float, default=1.0)
    parser.add_argument("--ptrm-eval-k", type=int, default=1)
    parser.add_argument("--ptrm-eval-sigma", type=float, default=0.0)
    parser.add_argument("--ptrm-eval-select", choices=("q", "random", "mean"), default="q")
    parser.add_argument("--ready-timeout-seconds", type=int, default=600)
    parser.add_argument("--native-shard-prewarm-seconds", type=float, default=0.0)
    parser.add_argument("--native-shard-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.scope == "single_task" and not args.task:
        raise SystemExit("single_task evaluation requires --task")
    if args.scope == "multitask" and args.task:
        raise SystemExit("multitask evaluation covers all16 and forbids --task")
    if args.scope == "multitask":
        args.task = "all16"
    if args.native_simulator:
        args.container_image = None
    elif args.simulator_pythonpath or args.simulator_gpus or args.pin_native_cpus:
        raise SystemExit("native simulator routing flags require --native-simulator")
    if not 0.0 <= args.native_shard_prewarm_seconds <= 600.0:
        raise SystemExit("--native-shard-prewarm-seconds must lie in [0, 600]")
    if not 0.0 <= args.native_shard_stagger_seconds <= 600.0:
        raise SystemExit("--native-shard-stagger-seconds must lie in [0, 600]")
    if (args.native_shard_prewarm_seconds or args.native_shard_stagger_seconds) and not args.native_simulator:
        raise SystemExit("native shard prewarm/stagger requires --native-simulator")
    if args.native_shard_prewarm_seconds and len(args.gpus) != len(args.simulator_gpus):
        raise SystemExit("native JIT prewarm requires one policy server and simulator GPU per lane")

    args.source_root = args.source_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.benchmark_config = args.benchmark_config.expanduser().resolve()
    args.vla_eval = args.vla_eval.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.simulator_pythonpath = [path.expanduser().resolve() for path in args.simulator_pythonpath]
    for name in ("workspace_checkpoint", "workspace_index", "upstream_root", "vision_encoder_home"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if not (args.source_root / "robomme_integration/eval/execution_model_server.py").is_file():
        raise SystemExit("source root lacks the RoboMME execution server")
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / "assets").is_dir():
        raise SystemExit("checkpoint must contain finalized params/ and assets/")
    if not args.benchmark_config.is_file() or not args.vla_eval.is_file():
        raise SystemExit("benchmark config or vla-eval binary is missing")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing non-empty evaluation output {args.output_root}")
    if not 1 <= args.shards <= 32 or args.shards < len(args.gpus):
        raise SystemExit("shards must lie in [number of policy GPUs, 32]")
    if not 0.10 <= args.xla_memory_fraction <= 0.95:
        raise SystemExit("--xla-memory-fraction must lie in [0.10, 0.95]")
    if not 1024 <= args.base_port <= 65535 - len(args.gpus):
        raise SystemExit("invalid policy-server port range")
    if args.ready_timeout_seconds < 60:
        raise SystemExit("model restore/compile readiness timeout must be at least 60 seconds")
    if args.arm in WORKSPACE_ARMS:
        if any(
            getattr(args, name) is None for name in ("workspace_checkpoint", "upstream_root", "vision_encoder_home")
        ):
            raise SystemExit("workspace steering requires checkpoint, pinned upstream, and vision weights")
        if args.scope == "multitask":
            if args.workspace_index is None or not args.workspace_index.is_file():
                raise SystemExit("multitask workspace evaluation requires a sealed local workspace index")
            try:
                from robomme_integration.training.workspace_index import load_workspace_index

                index = load_workspace_index(args.workspace_index)
            except (OSError, ValueError) as error:
                raise SystemExit(f"invalid multitask workspace index: {error}") from error
            for task, record in index["tasks"].items():
                checkpoint = args.workspace_checkpoint / task / str(record["representation"]["step"])
                complete = checkpoint / "WSM_GENERATION_COMPLETE.json"
                if not complete.is_file():
                    raise SystemExit(f"workspace representation checkpoint is incomplete for {task}")
                actual = hashlib.sha256(complete.read_bytes()).hexdigest()
                expected = record["representation"]["completion_sha256"]
                if actual != expected:
                    raise SystemExit(f"workspace representation/index SHA mismatch for {task}")
        else:
            if args.workspace_index is not None:
                raise SystemExit("single-task workspace evaluation forbids an all-16 workspace index")
            required = {"WSM_RUN_CONFIG.json", "WSM_BEST.json", "WSM_GENERATION_COMPLETE.json"}
            missing = [name for name in required if not (args.workspace_checkpoint / name).is_file()]
            if missing:
                raise SystemExit(f"workspace representation checkpoint is incomplete: {missing}")
        if not (args.vision_encoder_home / "pi05_vision_encoder/siglip_params.pkl").is_file():
            raise SystemExit("workspace evaluation lacks pinned pi0.5 vision weights")
    elif any(
        getattr(args, name) is not None
        for name in ("workspace_checkpoint", "workspace_index", "upstream_root", "vision_encoder_home")
    ):
        raise SystemExit("non-workspace arms forbid workspace evaluation inputs")
    cfg_arms = {"wsm_cfg", "v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}
    if args.arm not in cfg_arms and args.cfg_guidance_scale != 1.0:
        raise SystemExit("CFG guidance scale is valid only for wsm_cfg")
    if args.arm in {"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"} and (args.cfg_guidance_scale not in V4_CFG_EVAL_SCALES):
        raise SystemExit("RoboMME v4 CFG scale must be one of 0.5,1.0,1.5,2.0")
    if args.arm in {"ptrm", "v4_ptrm"}:
        if (args.ptrm_eval_k, args.ptrm_eval_sigma, args.ptrm_eval_select) != (1, 0.0, "q"):
            raise SystemExit("RoboMME PTRM is preregistered as E0 only: K=1, sigma=0, select=q")
    elif (args.ptrm_eval_k, args.ptrm_eval_sigma, args.ptrm_eval_select) != (1, 0.0, "q"):
        raise SystemExit("PTRM evaluation knobs are valid only for the ptrm arm")

    compatibility_root = args.source_root / "robomme_integration" / "compat"
    if not (compatibility_root / "robocasa").is_dir():
        raise SystemExit("source root lacks the isolated RoboCasa compatibility package")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(compatibility_root) + os.pathsep + str(args.source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    help_probe = subprocess.run(
        [sys.executable, "-m", "robomme_integration.eval.execution_model_server", "--help"],
        cwd=args.source_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if help_probe.returncode:
        raise SystemExit(f"policy-server CLI probe failed:\n{help_probe.stdout}")
    ports = tuple(args.base_port + index for index in range(len(args.gpus)))
    server_commands = [_build_server_command(args, help_probe.stdout, port) for port in ports]
    eval_dir = args.output_root / "eval"
    mounts = sorted(
        {
            args.source_root,
            args.output_root,
            args.vla_eval.parent.parent,
            args.benchmark_config.parent,
        },
        key=str,
    )
    launcher_command = [
        sys.executable,
        "-m",
        "robomme_integration.eval.launch_sharded",
        "--vla-eval",
        str(args.vla_eval),
        "--config",
        str(args.benchmark_config),
        "--output-dir",
        str(eval_dir),
        "--shards",
        str(args.shards),
        "--eval-id",
        args.eval_id,
        "--cpu-range",
        args.cpu_range,
    ]
    if args.container_image:
        launcher_command.extend(
            [
                "--container-image",
                args.container_image,
                "--container-user",
                args.container_user,
                "--container-pythonpath",
                str(args.source_root),
            ]
        )
        for mount in mounts:
            launcher_command.extend(["--container-mount", str(mount)])
    else:
        launcher_command.append("--no-docker")
        for path in args.simulator_pythonpath:
            launcher_command.extend(["--native-pythonpath", str(path)])
        if args.simulator_gpus:
            launcher_command.extend(["--native-gpus", ",".join(str(gpu) for gpu in args.simulator_gpus)])
        if args.pin_native_cpus:
            launcher_command.append("--pin-native-cpus")
        if args.native_shard_prewarm_seconds:
            launcher_command.extend(["--native-shard-prewarm-seconds", str(args.native_shard_prewarm_seconds)])
            for gpu, port in zip(args.gpus, ports, strict=True):
                launcher_command.extend(
                    [
                        "--native-prewarm-log",
                        str(args.output_root / f"server-gpu{gpu}-port{port}.log"),
                    ]
                )
            launcher_command.extend(["--native-prewarm-marker", f"inference arm={args.arm}"])
        if args.native_shard_stagger_seconds:
            launcher_command.extend(["--native-shard-stagger-seconds", str(args.native_shard_stagger_seconds)])
    for port in ports:
        launcher_command.extend(["--server-url", f"ws://127.0.0.1:{port}"])
    payload = {
        "schema_version": 1,
        "task": args.task,
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "source_root": str(args.source_root),
        "benchmark_config": str(args.benchmark_config),
        "eval_id": args.eval_id,
        "gpus": list(args.gpus),
        "ports": list(ports),
        "shards": args.shards,
        "cpu_range": args.cpu_range,
        "simulator_gpus": list(args.simulator_gpus),
        "pin_native_cpus": args.pin_native_cpus,
        "xla_memory_fraction": args.xla_memory_fraction,
        "native_shard_prewarm_seconds": args.native_shard_prewarm_seconds,
        "native_shard_stagger_seconds": args.native_shard_stagger_seconds,
        "policy_pythonpath": environment["PYTHONPATH"],
        "server_commands": server_commands,
        "launcher_command": launcher_command,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    args.output_root.mkdir(parents=True)
    (args.output_root / "supervisor.json").write_text(json.dumps(payload, indent=2) + "\n")
    processes: list[subprocess.Popen] = []
    logs = []
    returncode: int | None = None
    failure: str | None = None
    received_signal: int | None = None

    def stop(signum: int, _frame) -> None:
        nonlocal received_signal
        received_signal = signum
        raise InterruptedError(f"received signal {signum}")

    previous_sigint = signal.signal(signal.SIGINT, stop)
    previous_sigterm = signal.signal(signal.SIGTERM, stop)
    try:
        for gpu, command, port in zip(args.gpus, server_commands, ports, strict=True):
            log = (args.output_root / f"server-gpu{gpu}-port{port}.log").open("wb")
            logs.append(log)
            server_env = dict(environment)
            server_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            server_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_memory_fraction)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=args.source_root,
                    env=server_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        for process, port in zip(processes, ports, strict=True):
            _wait_for_server(process, port, args.ready_timeout_seconds)
        launcher = subprocess.run(launcher_command, cwd=args.source_root, env=environment, check=False)
        returncode = int(launcher.returncode)
    except BaseException as error:
        returncode = 128 + received_signal if received_signal is not None else 1
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        for process in processes:
            _terminate_group(process.pid)
        for log in logs:
            log.close()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        payload.update(
            finished_utc=_utc_now(),
            launcher_returncode=returncode,
            failure=failure,
            received_signal=received_signal,
        )
        (args.output_root / "supervisor.json").write_text(json.dumps(payload, indent=2) + "\n")
        if returncode == 0:
            (args.output_root / "COMPLETED").write_text(_utc_now() + "\n")
        else:
            (args.output_root / "FAILED").write_text(str(returncode) + "\n")
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
