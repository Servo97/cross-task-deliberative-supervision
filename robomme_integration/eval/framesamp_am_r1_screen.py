"""Run the sealed FS-R1 n=8 screen as twelve sequential eight-H100 cells."""

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
from typing import Any

from robomme_integration.eval.framesamp_am_r1_canary import (
    H100_NAME,
    parse_h100_topology,
)
from robomme_integration.eval.framesamp_am_r1_packet import validate as validate_packet
from robomme_integration.eval.framesamp_am_r1_publish import (
    build_result_claim,
    publish_s3_create_once,
    write_create_once,
)

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_oracle_screen_completion"
LANE_TIMEOUT_SECONDS = 14_400.0


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(path: Path, limit: int = 4_000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def _run_cell(
    *,
    cell: dict[str, Any],
    packet_path: Path,
    output_dir: Path,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    official_checkout: Path,
    checkpoint: Path,
    runtime_root: Path,
    result_s3: str | None,
) -> dict[str, Any]:
    identity = cell["identity"]
    cell_dir = output_dir / cell["cell_id"]
    cell_dir.mkdir()
    runner = Path(__file__).with_name("framesamp_am_r1_rollout.py")
    processes: list[tuple[subprocess.Popen[str], object, object]] = []
    failures: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        for lane, episode in enumerate(range(8)):
            lane_dir = cell_dir / f"lane-{lane}"
            lane_dir.mkdir()
            stdout_path = lane_dir / "stdout.log"
            stderr_path = lane_dir / "stderr.log"
            stdout = stdout_path.open("x", encoding="utf-8")
            stderr = stderr_path.open("x", encoding="utf-8")
            command = [
                sys.executable,
                str(runner),
                "--task",
                identity["task"],
                "--episode",
                str(episode),
                "--budget",
                str(identity["budget"]),
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
                "--output",
                str(lane_dir / "receipt.json"),
            ]
            if identity["fit_mass"]:
                command.append("--fit-mass")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(lane),
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.70",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            process = subprocess.Popen(
                command,
                cwd=policy_overlay,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            processes.append((process, stdout, stderr))
        deadline = time.monotonic() + LANE_TIMEOUT_SECONDS
        for lane, (process, stdout, stderr) in enumerate(processes):
            try:
                returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                returncode = -signal.SIGTERM
                _terminate(process)
            stdout.close()
            stderr.close()
            if returncode != 0:
                lane_dir = cell_dir / f"lane-{lane}"
                failures.append(
                    {
                        "lane": lane,
                        "returncode": returncode,
                        "stdout_tail": _tail(lane_dir / "stdout.log"),
                        "stderr_tail": _tail(lane_dir / "stderr.log"),
                    }
                )
        if failures:
            raise RuntimeError(
                "FS-R1 screen lane failures: " + json.dumps(failures, sort_keys=True, separators=(",", ":"))
            )
    finally:
        for process, stdout, stderr in processes:
            _terminate(process)
            if not stdout.closed:
                stdout.close()
            if not stderr.closed:
                stderr.close()

    receipt_paths = [cell_dir / f"lane-{lane}/receipt.json" for lane in range(8)]
    for path in receipt_paths:
        if _load_json(path).get("policy_runtime") != {
            "source": "upstream_uv_lock",
            "jax": "0.5.3",
            "orbax_checkpoint": "0.11.13",
        }:
            raise ValueError("FS-R1 screen lane used the wrong released-checkpoint runtime")
    claim = build_result_claim(
        packet_path=packet_path,
        cell_id=cell["cell_id"],
        receipt_paths=receipt_paths,
    )
    claim["cell_elapsed_seconds"] = float(time.monotonic() - started)
    # The publisher seal covers the scientific result only; runtime duration is operational data.
    claim["claim_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in claim.items() if key != "claim_sha256"})
    ).hexdigest()
    payload = json.dumps(claim, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    local_claim = cell_dir / "result.complete.json"
    write_create_once(local_claim, payload)
    if result_s3 is not None:
        publish_s3_create_once(result_s3, payload)
    return {
        "cell_id": cell["cell_id"],
        "claim_sha256": claim["claim_sha256"],
        "claim_file_sha256": _sha256_file(local_claim),
        "successes": claim["successes"],
        "valid_episodes": claim["valid_episodes"],
        "promote_to_paired_fixed50": claim["promote_to_paired_fixed50"],
        "result_s3": result_s3,
    }


def run_screen(
    *,
    packet_path: Path,
    output_dir: Path,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    official_checkout: Path,
    checkpoint: Path,
    runtime_root: Path,
    result_uris: dict[str, str] | None = None,
    completion_s3: str | None = None,
) -> dict[str, object]:
    packet = _load_json(packet_path)
    validate_packet(packet)
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise ValueError("FS-R1 screen output directory must exist and be empty")
    cell_ids = [cell["cell_id"] for cell in packet["oracle_cells"]]
    if result_uris is not None and set(result_uris) != set(cell_ids):
        raise ValueError("FS-R1 result URI mapping must cover exactly all twelve cells")
    topology = parse_h100_topology(
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        ).stdout
    )
    started = time.monotonic()
    records = []
    for cell in packet["oracle_cells"]:
        records.append(
            _run_cell(
                cell=cell,
                packet_path=packet_path,
                output_dir=output_dir,
                policy_overlay=policy_overlay,
                overlay_manifest_sha256=overlay_manifest_sha256,
                official_checkout=official_checkout,
                checkpoint=checkpoint,
                runtime_root=runtime_root,
                result_s3=None if result_uris is None else result_uris[cell["cell_id"]],
            )
        )
    completion: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "COMPLETE",
        "scope": "n8_regression_screen_not_statistical_evidence",
        "packet_file_sha256": _sha256_file(packet_path),
        "packet_sha256": packet["packet_sha256"],
        "topology": topology,
        "cells": records,
        "cell_count": 12,
        "valid_episodes": 96,
        "harness_failures": 0,
        "elapsed_seconds": float(time.monotonic() - started),
        "all_cells_complete": True,
    }
    completion["receipt_sha256"] = hashlib.sha256(_canonical(completion)).hexdigest()
    payload = json.dumps(completion, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    write_create_once(output_dir / "screen.complete.json", payload)
    if completion_s3 is not None:
        publish_s3_create_once(completion_s3, payload)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--result-uri", action="append", default=[])
    parser.add_argument("--completion-s3")
    parser.add_argument("--confirm-publish", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.result_uri or args.completion_s3) != args.confirm_publish:
        raise ValueError("screen publication requires both destinations and --confirm-publish")
    result_uris: dict[str, str] | None = None
    if args.confirm_publish:
        result_uris = {}
        for item in args.result_uri:
            cell_id, separator, uri = item.partition("=")
            if not separator or not cell_id or not uri or cell_id in result_uris:
                raise ValueError("--result-uri must be unique CELL_ID=s3://... entries")
            result_uris[cell_id] = uri
        if not args.completion_s3:
            raise ValueError("screen publication requires --completion-s3")
    result = run_screen(
        packet_path=args.packet,
        output_dir=args.output_dir,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        runtime_root=args.runtime_root,
        result_uris=result_uris,
        completion_s3=args.completion_s3,
    )
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
