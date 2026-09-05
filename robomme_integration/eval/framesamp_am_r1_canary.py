"""Exact 8xH100 one-cut runtime canary for the FS-R1 oracle screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_8xh100_runtime_canary"
H100_NAME = "NVIDIA H100 80GB HBM3"
LANES = (
    ("VideoPlaceButton", 0, 256, False),
    ("VideoPlaceButton", 1, 128, False),
    ("VideoPlaceButton", 2, 64, False),
    ("VideoPlaceButton", 3, 256, True),
    ("VideoPlaceButton", 4, 128, True),
    ("VideoPlaceButton", 5, 64, True),
    ("RouteStick", 0, 64, False),
    ("RouteStick", 1, 64, True),
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_h100_topology(output: str) -> list[dict[str, object]]:
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError("p5 nvidia-smi row must contain index, uuid, name, memory")
        index_text, uuid, name, memory_text = fields
        if not index_text.isdigit() or not uuid.startswith("GPU-") or name != H100_NAME:
            raise ValueError("FS-R1 canary requires exact indexed H100 80GB HBM3 rows")
        try:
            memory_mib = int(memory_text.split()[0])
        except (ValueError, IndexError) as error:
            raise ValueError("FS-R1 canary GPU memory report is malformed") from error
        rows.append(
            {
                "index": int(index_text),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": memory_mib,
            }
        )
    if (
        len(rows) != 8
        or [row["index"] for row in rows] != list(range(8))
        or len({row["uuid"] for row in rows}) != 8
        or any(not 80_000 <= int(row["memory_total_mib"]) <= 90_000 for row in rows)
    ):
        raise ValueError("FS-R1 canary requires eight distinct 80--90GiB H100 GPUs")
    return rows


def validate_lane_receipt(
    receipt: dict[str, object],
    *,
    task: str,
    episode: int,
    budget: int,
    fit_mass: bool,
) -> None:
    seal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if seal != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("FS-R1 canary lane receipt seal mismatch")
    if (
        receipt.get("kind") != "robomme_framesamp_am_r1_oracle_rollout_canary"
        or receipt.get("status") != "HARD_GREEN"
        or receipt.get("scope") != "fs_r1_runtime_canary_not_scored_evidence"
        or receipt.get("task") != task
        or receipt.get("episode") != episode
        or receipt.get("budget") != budget
        or receipt.get("fit_mass") is not fit_mass
        or receipt.get("canary_replans") != 1
        or receipt.get("replans") != 1
        or receipt.get("success") is not None
        or receipt.get("fresh_attested_stack_fraction") != 1.0
        or receipt.get("persistent_oracle_payloads") is not False
        or receipt.get("ephemeral_artifacts_deleted_before_receipt") is not True
        or receipt.get("policy_runtime")
        != {
            "source": "upstream_uv_lock",
            "jax": "0.5.3",
            "orbax_checkpoint": "0.11.13",
        }
    ):
        raise ValueError("FS-R1 canary lane receipt semantic mismatch")
    device = receipt.get("device")
    simulator = receipt.get("simulator_runtime")
    if (
        not isinstance(device, dict)
        or device.get("count") != 1
        or device.get("platform") != "gpu"
        or device.get("device_kind") != H100_NAME
        or not isinstance(simulator, dict)
        or simulator.get("gpu_name") != H100_NAME
        or simulator.get("torch_cuda_version") != "12.8"
    ):
        raise ValueError("FS-R1 canary lane did not execute on the exact H100 runtime")
    cuts = receipt.get("cuts")
    if not isinstance(cuts, list) or len(cuts) != 1:
        raise ValueError("FS-R1 canary lane must contain exactly one authenticated cut")
    cut = cuts[0]
    if (
        not isinstance(cut, dict)
        or len(cut.get("layers", [])) != 18
        or cut.get("oracle_payload_deleted_before_simulator_response") is not True
        or cut.get("executed_actions") != 16
    ):
        raise ValueError("FS-R1 canary lane did not fit, consume, and execute one complete cut")


def run_canary(
    *,
    output_dir: Path,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    official_checkout: Path,
    checkpoint: Path,
    runtime_root: Path,
) -> dict[str, object]:
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise ValueError("FS-R1 canary output directory must exist and be empty")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader",
    ]
    topology = parse_h100_topology(
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30.0).stdout
    )
    runner = Path(__file__).with_name("framesamp_am_r1_rollout.py")
    processes: list[tuple[subprocess.Popen[str], object, object]] = []
    started = time.monotonic()
    try:
        for lane, (task, episode, budget, fit_mass) in enumerate(LANES):
            lane_dir = output_dir / f"lane-{lane}"
            lane_dir.mkdir()
            receipt = lane_dir / "receipt.json"
            stdout = (lane_dir / "stdout.log").open("x", encoding="utf-8")
            stderr = (lane_dir / "stderr.log").open("x", encoding="utf-8")
            lane_command = [
                sys.executable,
                str(runner),
                "--task",
                task,
                "--episode",
                str(episode),
                "--budget",
                str(budget),
                "--policy-overlay",
                str(policy_overlay),
                "--overlay-manifest-sha256",
                overlay_manifest_sha256,
                "--official-checkout",
                str(official_checkout),
                "--checkpoint",
                str(checkpoint),
                "--runtime-root",
                str(runtime_root),
                "--simulator-cuda-device",
                str(lane),
                "--expected-jax-device-kind",
                H100_NAME,
                "--expected-simulator-gpu-name",
                H100_NAME,
                "--max-replans",
                "1",
                "--canary-replans",
                "1",
                "--output",
                str(receipt),
            ]
            if fit_mass:
                lane_command.append("--fit-mass")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(lane),
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.70",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            process = subprocess.Popen(
                lane_command,
                cwd=policy_overlay,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            processes.append((process, stdout, stderr))
        failures = []
        for lane, (process, stdout, stderr) in enumerate(processes):
            returncode = process.wait(timeout=3_600.0)
            stdout.close()
            stderr.close()
            if returncode != 0:
                lane_dir = output_dir / f"lane-{lane}"
                failures.append(
                    {
                        "lane": lane,
                        "returncode": returncode,
                        "stdout_tail": (lane_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")[
                            -4_000:
                        ],
                        "stderr_tail": (lane_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")[
                            -4_000:
                        ],
                    }
                )
        if failures:
            raise RuntimeError(
                "FS-R1 H100 canary lane failures: " + json.dumps(failures, sort_keys=True, separators=(",", ":"))
            )
    finally:
        for process, stdout, stderr in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10.0)
            if not stdout.closed:
                stdout.close()
            if not stderr.closed:
                stderr.close()
    lane_receipts = []
    for lane, (task, episode, budget, fit_mass) in enumerate(LANES):
        path = output_dir / f"lane-{lane}/receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_lane_receipt(
            receipt,
            task=task,
            episode=episode,
            budget=budget,
            fit_mass=fit_mass,
        )
        lane_receipts.append(
            {
                "lane": lane,
                "task": task,
                "episode": episode,
                "budget": budget,
                "fit_mass": fit_mass,
                "receipt_file_sha256": _sha256_file(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt": receipt,
            }
        )
    aggregate: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "HARD_GREEN",
        "scope": "runtime_canary_only_not_scored_evidence",
        "topology": topology,
        "lanes": lane_receipts,
        "lane_count": 8,
        "authenticated_cuts": 8,
        "executed_simulator_actions": 128,
        "runner_sha256": _sha256_file(runner),
        "sim_worker_sha256": _sha256_file(Path(__file__).with_name("framesamp_r1_sim_worker.py")),
        "elapsed_seconds": float(time.monotonic() - started),
        "cloud_publication": False,
    }
    aggregate["receipt_sha256"] = hashlib.sha256(_canonical(aggregate)).hexdigest()
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run_canary(
        output_dir=args.output_dir,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        runtime_root=args.runtime_root,
    )
    path = args.output_dir / "canary.complete.json"
    with path.open("xb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"receipt_sha256": receipt["receipt_sha256"], "status": "HARD_GREEN"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
