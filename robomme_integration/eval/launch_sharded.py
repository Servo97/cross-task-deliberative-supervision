#!/usr/bin/env python3
"""Launch bounded RoboMME CPU shards against one TPU policy server and merge results."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _build_shard_command(
    executable: Path,
    config: Path,
    output_dir: Path,
    shard: int,
    shards: int,
    eval_id: str | None = None,
    no_docker: bool = False,
    server_url: str | None = None,
) -> list[str]:
    command = [
        str(executable),
        "run",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--shard-id",
        str(shard),
        "--num-shards",
        str(shards),
        "--yes",
    ]
    if eval_id is not None:
        command.extend(["--eval-id", eval_id])
    if server_url is not None:
        command.extend(["--server-url", server_url])
    if no_docker:
        command.append("--no-docker")
    return command


def _uses_sqlite_recording(executable: Path) -> bool:
    result = subprocess.run(
        [str(executable), "run", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return "--eval-id" in result.stdout


def _cpu_partitions(cpu_range: str, shards: int) -> list[str]:
    start_text, separator, end_text = cpu_range.partition("-")
    if not separator:
        raise ValueError("--cpu-range must be one contiguous inclusive range such as 0-239")
    start, end = int(start_text), int(end_text)
    if start < 0 or end < start or end - start + 1 < shards:
        raise ValueError(f"invalid CPU range {cpu_range!r} for {shards} shards")
    cpus = list(range(start, end + 1))
    base, extra = divmod(len(cpus), shards)
    partitions = []
    cursor = 0
    for shard in range(shards):
        count = base + int(shard < extra)
        selected = cpus[cursor : cursor + count]
        cursor += count
        partitions.append(f"{selected[0]}-{selected[-1]}")
    return partitions


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError("device list must be comma-separated integers") from error
    if not devices or len(devices) != len(set(devices)) or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("device list must be nonempty, unique, and nonnegative")
    return devices


def _native_launch_waves(shards: int, native_gpus: tuple[int, ...]) -> list[list[int]]:
    """Return waves containing at most one newly started simulator per physical GPU."""
    if not native_gpus:
        return [list(range(shards))]
    by_gpu = {gpu: [] for gpu in native_gpus}
    for shard in range(shards):
        by_gpu[native_gpus[shard % len(native_gpus)]].append(shard)
    return [
        [assigned[index] for assigned in by_gpu.values() if index < len(assigned)]
        for index in range(max(len(assigned) for assigned in by_gpu.values()))
    ]


def _wait_launch_delay(seconds: float, processes: list[subprocess.Popen]) -> None:
    """Interruptibly wait between native launch waves and fail on an early shard crash."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"native shard failed during launch prewarm: {failed}")
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def _wait_for_prewarm_markers(
    paths: list[Path],
    marker: str,
    timeout_seconds: float,
    processes: list[subprocess.Popen],
) -> None:
    """Fail closed until every policy lane has completed at least one real inference."""
    deadline = time.monotonic() + timeout_seconds
    marker_bytes = marker.encode("utf-8")

    def contains_marker(path: Path) -> bool:
        if not path.is_file():
            return False
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 1024 * 1024))
            return marker_bytes in stream.read()

    while time.monotonic() < deadline:
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"native shard failed during policy JIT prewarm: {failed}")
        ready = [contains_marker(path) for path in paths]
        if ready and all(ready):
            return
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    missing = [str(path) for path in paths if not contains_marker(path)]
    raise RuntimeError(f"policy JIT prewarm marker {marker!r} did not appear before timeout: {missing}")


def _audit_episode_results(paths: list[Path]) -> dict[str, object]:
    """Distinguish valid task failures from harness/environment failures."""
    episodes = 0
    failures: list[dict[str, object]] = []
    for path in paths:
        if path.suffix != ".json":
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            continue
        for task in payload.get("tasks", []):
            if not isinstance(task, dict):
                continue
            for episode in task.get("episodes", []):
                if not isinstance(episode, dict):
                    continue
                episodes += 1
                reason = episode.get("failure_reason")
                if reason:
                    failures.append(
                        {
                            "path": str(path),
                            "task": task.get("task"),
                            "episode_id": episode.get("episode_id"),
                            "reason": reason,
                            "detail": episode.get("failure_detail"),
                        }
                    )
    return {"episodes": episodes, "harness_failures": failures}


def _container_command(
    command: list[str],
    *,
    image: str,
    name: str,
    cpus: str,
    user: str,
    mounts: list[Path],
    pythonpath: list[Path] | None = None,
) -> list[str]:
    wrapped = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--security-opt",
        "seccomp=unconfined",
        "--cpuset-cpus",
        cpus,
        "--user",
        user,
        "-e",
        "HOME=/tmp",
        "-e",
        "ROBOMME_USE_LAVAPIPE=1",
        "-e",
        "ROBOMME_LAVAPIPE_ICD=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
        "-e",
        "SAPIEN_RENDER_DEVICE=cpu",
        "-e",
        "MUJOCO_GL=osmesa",
        "-e",
        "LP_NUM_THREADS=4",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
    ]
    if pythonpath:
        wrapped.extend(["-e", "PYTHONPATH=" + ":".join(str(path) for path in pythonpath)])
    for mount in mounts:
        wrapped.extend(["-v", f"{mount}:{mount}"])
    return [*wrapped, image, *command]


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 15
    while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
        time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-eval", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--merged-name", default="merged.json")
    parser.add_argument("--eval-id", default=None)
    parser.add_argument("--container-image", default=None)
    parser.add_argument("--container-mount", action="append", type=Path, default=[])
    parser.add_argument("--container-user", default="2001:2001")
    parser.add_argument("--container-pythonpath", action="append", type=Path, default=[])
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="run the benchmark natively; the caller must have certified simulator/rendering deps",
    )
    parser.add_argument(
        "--native-pythonpath",
        action="append",
        type=Path,
        default=[],
        help="exact PYTHONPATH entry for native simulator children; repeat to add entries",
    )
    parser.add_argument(
        "--native-gpus",
        type=_parse_devices,
        default=(),
        help="GPU IDs assigned round-robin to native simulator children",
    )
    parser.add_argument(
        "--pin-native-cpus",
        action="store_true",
        help="pin each native simulator child to its disjoint --cpu-range partition",
    )
    parser.add_argument(
        "--native-shard-prewarm-seconds",
        type=float,
        default=0.0,
        help="timeout for first real inference on one shard/GPU before admitting later waves",
    )
    parser.add_argument(
        "--native-shard-stagger-seconds",
        type=float,
        default=0.0,
        help="delay later one-shard-per-GPU waves to bound concurrent JIT/CUBIN loads",
    )
    parser.add_argument(
        "--native-prewarm-log",
        action="append",
        type=Path,
        default=[],
        help="policy-server log which must emit the JIT-ready marker; repeat per lane",
    )
    parser.add_argument("--native-prewarm-marker", default="inference arm=")
    parser.add_argument(
        "--server-url",
        action="append",
        default=[],
        help="one or more policy servers; shards are assigned round-robin (config URL if omitted)",
    )
    parser.add_argument("--cpu-range", default="0-239")
    args = parser.parse_args()

    executable = args.vla_eval.expanduser().resolve()
    config = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not executable.is_file():
        raise SystemExit(f"vla-eval executable does not exist: {executable}")
    if not config.is_file():
        raise SystemExit(f"evaluation config does not exist: {config}")
    if not 1 <= args.shards <= 32:
        raise SystemExit("--shards must lie in [1, 32]")
    if any(not url.startswith(("ws://127.0.0.1:", "ws://localhost:")) for url in args.server_url):
        raise SystemExit("--server-url entries must be loopback WebSocket endpoints")
    if args.container_pythonpath and not args.container_image:
        raise SystemExit("--container-pythonpath requires --container-image")
    if (args.native_pythonpath or args.native_gpus or args.pin_native_cpus) and not args.no_docker:
        raise SystemExit("native simulator routing flags require --no-docker")
    if not 0.0 <= args.native_shard_prewarm_seconds <= 600.0:
        raise SystemExit("--native-shard-prewarm-seconds must lie in [0, 600]")
    if not 0.0 <= args.native_shard_stagger_seconds <= 600.0:
        raise SystemExit("--native-shard-stagger-seconds must lie in [0, 600]")
    if (args.native_shard_prewarm_seconds or args.native_shard_stagger_seconds) and (
        not args.no_docker or not args.native_gpus
    ):
        raise SystemExit("native shard prewarm/stagger requires --no-docker and --native-gpus")
    if args.native_prewarm_log and (
        args.native_shard_prewarm_seconds <= 0
        or len(args.native_prewarm_log) != len(args.native_gpus)
        or len(args.native_prewarm_log) != len(args.server_url)
    ):
        raise SystemExit("native JIT prewarm logs require a positive timeout and one log per GPU/server")
    if not args.native_prewarm_marker:
        raise SystemExit("native JIT prewarm marker must be nonempty")
    if args.container_image and args.no_docker:
        raise SystemExit("--container-image and --no-docker are mutually exclusive")
    sqlite_recording = _uses_sqlite_recording(executable)
    eval_id = args.eval_id
    if sqlite_recording:
        eval_id = eval_id or dt.datetime.now(dt.timezone.utc).strftime("robomme-%Y%m%dT%H%M%SZ")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", eval_id):
            raise SystemExit("--eval-id may contain only letters, digits, underscore, period, and hyphen")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to mix a new evaluation with non-empty output: {output_dir}")
    mounts = [path.expanduser().resolve() for path in args.container_mount]
    container_pythonpath = [path.expanduser().resolve() for path in args.container_pythonpath]
    native_pythonpath = [path.expanduser().resolve() for path in args.native_pythonpath]
    if args.container_image:
        missing = [path for path in [*mounts, *container_pythonpath] if not path.exists()]
        if missing:
            raise SystemExit(f"container mounts do not exist: {missing}")
    else:
        missing = [path for path in native_pythonpath if not path.exists()]
        if missing:
            raise SystemExit(f"native PYTHONPATH entries do not exist: {missing}")
    if args.container_image or args.pin_native_cpus:
        try:
            cpu_partitions = _cpu_partitions(args.cpu_range, args.shards)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        cpu_partitions = []

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir()
    started = _utc_now()
    commands = [
        _build_shard_command(
            executable,
            config,
            output_dir,
            shard,
            args.shards,
            eval_id=eval_id if sqlite_recording else None,
            no_docker=bool(args.container_image) or args.no_docker,
            server_url=(args.server_url[shard % len(args.server_url)] if args.server_url else None),
        )
        for shard in range(args.shards)
    ]
    if args.container_image:
        commands = [
            _container_command(
                command,
                image=args.container_image,
                name=f"{eval_id or 'robomme'}-shard-{shard:02d}",
                cpus=cpu_partitions[shard],
                user=args.container_user,
                mounts=mounts,
                pythonpath=container_pythonpath,
            )
            for shard, command in enumerate(commands)
        ]
    elif args.pin_native_cpus:
        commands = [
            ["taskset", "--cpu-list", cpu_partitions[shard], *command] for shard, command in enumerate(commands)
        ]
    process_environments: list[dict[str, str] | None] = []
    for shard in range(args.shards):
        if args.container_image or (not native_pythonpath and not args.native_gpus):
            process_environments.append(None)
            continue
        child_environment = os.environ.copy()
        if native_pythonpath:
            child_environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in native_pythonpath)
        if args.native_gpus:
            child_environment["CUDA_VISIBLE_DEVICES"] = str(args.native_gpus[shard % len(args.native_gpus)])
        process_environments.append(child_environment)
    processes: list[subprocess.Popen] = []
    processes_by_shard: list[subprocess.Popen | None] = [None] * args.shards
    logs = []
    launch_waves = (
        _native_launch_waves(args.shards, args.native_gpus)
        if args.native_shard_prewarm_seconds or args.native_shard_stagger_seconds
        else [list(range(args.shards))]
    )

    def stop(signum=None, _frame=None):
        # Raising is essential while the supervisor is between launch waves: returning from this
        # handler would resume the loop and admit new shards after cancellation.  The enclosing
        # BaseException handler owns the one cleanup path for every child admitted so far.
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for wave_index, wave in enumerate(launch_waves):
            for shard in wave:
                log_path = log_dir / f"shard-{shard:02d}.log"
                log_stream = log_path.open("wb")
                logs.append(log_stream)
                process = subprocess.Popen(
                    commands[shard],
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    env=process_environments[shard],
                )
                processes.append(process)
                processes_by_shard[shard] = process
            if wave_index < len(launch_waves) - 1:
                if wave_index == 0 and args.native_prewarm_log:
                    _wait_for_prewarm_markers(
                        args.native_prewarm_log,
                        args.native_prewarm_marker,
                        args.native_shard_prewarm_seconds,
                        processes,
                    )
                else:
                    delay = args.native_shard_prewarm_seconds if wave_index == 0 else args.native_shard_stagger_seconds
                    _wait_launch_delay(delay, processes)

        while any(process.poll() is None for process in processes):
            failures = [process.returncode for process in processes if process.returncode not in (None, 0)]
            if failures:
                _terminate(processes)
                break
            time.sleep(1)
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for stream in logs:
            stream.close()

    if any(process is None for process in processes_by_shard):
        raise RuntimeError("not all native simulator shards were admitted")
    returncodes = [process.wait() for process in processes_by_shard if process is not None]
    shard_files = sorted(output_dir.glob(f"*_shard*of{args.shards}.json"))
    manifest = {
        "started_utc": started,
        "finished_utc": _utc_now(),
        "config": str(config),
        "config_sha256": _sha256(config),
        "shards": args.shards,
        "recording_mode": "sqlite" if sqlite_recording else "per-shard-json",
        "eval_id": eval_id,
        "container": {
            "image": args.container_image,
            "mounts": [str(path) for path in mounts],
            "user": args.container_user if args.container_image else None,
            "cpu_partitions": cpu_partitions,
            "pythonpath": [str(path) for path in container_pythonpath],
        },
        "native": {
            "pythonpath": [str(path) for path in native_pythonpath],
            "gpus": list(args.native_gpus),
            "gpu_by_shard": [
                args.native_gpus[shard % len(args.native_gpus)] if args.native_gpus else None
                for shard in range(args.shards)
            ],
            "cpu_partitions": cpu_partitions if args.pin_native_cpus else [],
            "launch_waves": launch_waves,
            "shard_prewarm_seconds": args.native_shard_prewarm_seconds,
            "shard_stagger_seconds": args.native_shard_stagger_seconds,
            "prewarm_logs": [str(path) for path in args.native_prewarm_log],
            "prewarm_marker": args.native_prewarm_marker,
        },
        "server_urls": args.server_url,
        "commands": commands,
        "returncodes": returncodes,
        "shard_files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in shard_files
        ],
    }
    manifest_path = output_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if any(code != 0 for code in returncodes):
        raise SystemExit(f"one or more evaluation shards failed: {returncodes}; see {log_dir}")
    if sqlite_recording:
        database = output_dir / f"recording-{eval_id}.sqlite"
        if not database.is_file():
            raise SystemExit(f"expected shared recording database does not exist: {database}")
        merge_command = [
            str(executable),
            "merge",
            "--db",
            str(database),
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(merge_command, check=True)
        materialized = sorted(
            path
            for pattern in ("*.json", "*.jsonl")
            for path in output_dir.glob(pattern)
            if path.name != manifest_path.name
        )
        if not materialized:
            raise SystemExit(f"SQLite merge produced no JSON/JSONL result under {output_dir}")
        manifest["database"] = {
            "path": str(database),
            "sha256": _sha256(database),
            "bytes": database.stat().st_size,
        }
        manifest["materialized_results"] = [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in materialized
        ]
        episode_audit = _audit_episode_results(materialized)
        manifest["episode_audit"] = episode_audit
        summary = manifest["materialized_results"]
    else:
        if len(shard_files) != args.shards:
            raise SystemExit(
                f"expected exactly {args.shards} shard result files, found {len(shard_files)}; see {output_dir}"
            )
        merged = output_dir / args.merged_name
        merge_command = [
            str(executable),
            "merge",
            *[str(path) for path in shard_files],
            "--output",
            str(merged),
        ]
        subprocess.run(merge_command, check=True)
        manifest["merged"] = {
            "path": str(merged),
            "sha256": _sha256(merged),
            "bytes": merged.stat().st_size,
        }
        summary = manifest["merged"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if sqlite_recording:
        if not episode_audit["episodes"]:
            raise SystemExit(f"merged results contain no episode records; see {output_dir}")
        if episode_audit["harness_failures"]:
            raise SystemExit(
                f"merged results contain {len(episode_audit['harness_failures'])} harness/environment failures; "
                f"see {manifest_path}"
            )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
