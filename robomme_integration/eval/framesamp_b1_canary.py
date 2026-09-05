"""Eight-lane H100 receipt-only canary for the FrameSamp B1 control."""

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

from robomme_integration.eval.framesamp_am_r1_canary import H100_NAME, parse_h100_topology
from robomme_integration.eval.framesamp_b1_transition import (
    KIND as TRANSITION_KIND,
)
from robomme_integration.eval.framesamp_b1_transition import (
    validate_receipt as validate_transition_receipt,
)
from robomme_integration.training.framesamp_b1_data import B1_PARTITION_KIND
from robomme_integration.training.framesamp_b1_policy_overlay import (
    OFFICIAL_MEM_BUFFER_SHA256,
    OFFICIAL_POLICY_GIT_SHA,
    OFFICIAL_POLICY_SHA256,
    PATCHED_MEM_BUFFER_SHA256,
    PATCHED_POLICY_SHA256,
    verify_framesamp_b1_policy_overlay,
)

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_b1_8xh100_transition_canary"
LANES = (
    ("PickXtimes", 0),
    ("StopCube", 0),
    ("VideoUnmaskSwap", 0),
    ("ButtonUnmaskSwap", 0),
    ("PickHighlight", 0),
    ("VideoPlaceOrder", 0),
    ("MoveCube", 0),
    ("RouteStick", 0),
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_canary_packet(
    packet: dict[str, object],
    *,
    source_root: Path,
    overlay_root: Path,
) -> None:
    """Authenticate the pre-registered B1 source, representation, and canary."""

    seal = packet.get("packet_sha256")
    unsigned = dict(packet)
    unsigned.pop("packet_sha256", None)
    if seal != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("B1 canary packet seal mismatch")
    if packet.get("schema_version") != 1 or packet.get("kind") != "robomme_framesamp_b1_control_canary_packet":
        raise ValueError("B1 canary packet identity mismatch")
    science = packet.get("scientific_identity")
    source = packet.get("source")
    canary = packet.get("canary")
    gate = packet.get("scientific_gate")
    if not all(isinstance(value, dict) for value in (science, source, canary, gate)):
        raise ValueError("B1 canary packet sections are malformed")
    if science != {
        "arm": "fs_b1_control_demo16_live16",
        "representation_policy": B1_PARTITION_KIND,
        "official_equivalence": False,
        "released_anchor": "perceptual-framesamp-modul/79999",
        "released_checkpoint_semantic_sha256": ("2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"),
        "total_frames": 32,
        "tokens_per_frame": 16,
        "total_memory_tokens": 512,
        "demo_slots": [0, 15],
        "demo_selection": "uniform_inclusive_once_over_0_through_exec_start_idx",
        "live_slots": [16, 31],
        "live_selection": "most_recent_16_execution_frames_chronological",
        "padding": "each_partition_independently_right_padded",
        "time_features": "episode_global_source_step",
        "memory_attention_physical_slots": "fixed_partition_slot_0_through_31",
        "action_query_rope_offset": 512,
        "attention_denominators": 1,
        "compression": "none",
    }:
        raise ValueError("B1 scientific identity drifted")
    if (
        source.get("official_policy_git_sha") != OFFICIAL_POLICY_GIT_SHA
        or source.get("base_policy_sha256") != OFFICIAL_POLICY_SHA256
        or source.get("patched_policy_sha256") != PATCHED_POLICY_SHA256
        or source.get("base_mem_buffer_sha256") != OFFICIAL_MEM_BUFFER_SHA256
        or source.get("patched_mem_buffer_sha256") != PATCHED_MEM_BUFFER_SHA256
        or source.get("overlay_path") != "logical://framesamp-b1-overlay-v1"
    ):
        raise ValueError("B1 packet source identity drifted")
    overlay_manifest_sha = source.get("overlay_manifest_sha256")
    manifest = verify_framesamp_b1_policy_overlay(overlay_root, expected_manifest_sha256=str(overlay_manifest_sha))
    if (
        source.get("overlay_source_tree_sha256") != manifest["source_tree_sha256"]
        or source.get("overlay_source_file_count") != manifest["source_file_count"]
    ):
        raise ValueError("B1 packet overlay inventory drifted")
    runtime_files = source.get("runtime_files")
    if not isinstance(runtime_files, dict) or not runtime_files:
        raise ValueError("B1 packet runtime source list is empty")
    for relative, expected_sha in runtime_files.items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(expected_sha, str)
            or _sha_file(source_root / relative) != expected_sha
        ):
            raise ValueError(f"B1 packet runtime source drifted at {relative}")
    expected_lanes = [{"lane": lane, "task": task, "episode": episode} for lane, (task, episode) in enumerate(LANES)]
    if (
        canary.get("state") != "IMPLEMENTED_PENDING_REAL_8XH100_RECEIPT"
        or canary.get("scope") != "receipt_only_not_scored_evidence"
        or canary.get("topology") != f"1x8 {H100_NAME}"
        or canary.get("queue") != "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
        or canary.get("instance_type") != "ml.p5.48xlarge"
        or canary.get("training_plan_arn") is not None
        or canary.get("sm_use_reserved_capacity") != "1"
        or canary.get("lanes") != expected_lanes
        or canary.get("receipt_kind") != KIND
        or canary.get("receipt_required_before_scored_eval") is not True
    ):
        raise ValueError("B1 packet canary contract drifted")
    if gate != {
        "status": "HARD_RED_PENDING_CANARY_RECEIPT",
        "next_after_green": "paired_same_episode_fixed50_fs_b1_control_vs_official_framesamp",
        "no_score_or_claim_from_canary": True,
        "b1_am_blocked_until_control_score": True,
    }:
        raise ValueError("B1 scientific gate drifted")


def validate_canary_receipt(receipt: dict[str, object]) -> None:
    seal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if seal != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("B1 canary receipt seal mismatch")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("kind") != KIND
        or receipt.get("status") != "HARD_GREEN"
        or receipt.get("scope") != "runtime_canary_only_not_scored_evidence"
        or receipt.get("representation_policy") != B1_PARTITION_KIND
        or receipt.get("lane_count") != 8
        or receipt.get("transition_count") != 8
        or receipt.get("executed_simulator_actions") != 8 * 16
        or receipt.get("scored_evidence") is not False
        or receipt.get("cloud_publication") is not False
    ):
        raise ValueError("B1 aggregate canary semantics mismatch")
    topology = receipt.get("topology")
    lanes = receipt.get("lanes")
    if not isinstance(topology, list) or len(topology) != 8:
        raise ValueError("B1 aggregate topology is malformed")
    if not isinstance(lanes, list) or len(lanes) != len(LANES):
        raise ValueError("B1 aggregate lane list is malformed")
    for lane, (task, episode) in enumerate(LANES):
        row = lanes[lane]
        if (
            not isinstance(row, dict)
            or row.get("lane") != lane
            or row.get("task") != task
            or row.get("episode") != episode
            or row.get("transition_receipt_sha256") != row.get("receipt", {}).get("receipt_sha256")
            or row.get("receipt", {}).get("kind") != TRANSITION_KIND
        ):
            raise ValueError("B1 aggregate lane binding mismatch")
        validate_transition_receipt(row["receipt"])


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
        raise ValueError("B1 canary output directory must exist and be empty")
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
    runner = Path(__file__).with_name("framesamp_b1_transition.py")
    processes: list[tuple[subprocess.Popen[str], object, object]] = []
    started = time.monotonic()
    try:
        for lane, (task, episode) in enumerate(LANES):
            lane_dir = output_dir / f"lane-{lane}"
            lane_dir.mkdir()
            stdout = (lane_dir / "stdout.log").open("x", encoding="utf-8")
            stderr = (lane_dir / "stderr.log").open("x", encoding="utf-8")
            command = [
                sys.executable,
                "-B",
                str(runner),
                "--task",
                task,
                "--episode",
                str(episode),
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
                "--output",
                str(lane_dir / "receipt.json"),
            ]
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
        failures = []
        for lane, (process, stdout, stderr) in enumerate(processes):
            try:
                returncode = process.wait(timeout=3_600.0)
            except subprocess.TimeoutExpired:
                returncode = -1
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
                "B1 canary lane failure: " + json.dumps(failures, sort_keys=True, separators=(",", ":"))
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

    lanes = []
    for lane, (task, episode) in enumerate(LANES):
        path = output_dir / f"lane-{lane}/receipt.json"
        transition = json.loads(path.read_text(encoding="utf-8"))
        validate_transition_receipt(transition)
        if transition.get("task") != task or transition.get("episode") != episode:
            raise ValueError("B1 canary transition receipt task/episode mismatch")
        lanes.append(
            {
                "lane": lane,
                "task": task,
                "episode": episode,
                "transition_receipt_sha256": transition["receipt_sha256"],
                "receipt_file_sha256": _sha_file(path),
                "receipt": transition,
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "HARD_GREEN",
        "scope": "runtime_canary_only_not_scored_evidence",
        "representation_policy": B1_PARTITION_KIND,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "topology": topology,
        "lane_count": 8,
        "transition_count": 8,
        "executed_simulator_actions": 8 * 16,
        "lanes": lanes,
        "runner_sha256": _sha_file(runner),
        "sim_worker_sha256": _sha_file(Path(__file__).with_name("framesamp_r0_sim_worker.py")),
        "elapsed_seconds": time.monotonic() - started,
        "scored_evidence": False,
        "cloud_publication": False,
    }
    result = dict(unsigned)
    result["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    validate_canary_receipt(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("B1 aggregate receipt must be a fresh file")
    result = run_canary(
        output_dir=args.output_dir,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        runtime_root=args.runtime_root,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(_canonical(result))
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
