#!/usr/bin/env python3
"""Resume the terminated VideoRepick/A6 cell, then return control to the p5 dispatcher.

This is a one-cell, user-approved recovery controller. It preserves the scientific run
identity, waits for the current MoveCube/Q0 admission to clear, resumes from the sealed
step-15000 recovery checkpoint, attaches the new Batch identity to the campaign ledger,
and execs the normal one-at-a-time quota dispatcher.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from robomme_integration import launch  # noqa: E402
from robomme_integration.sweeps.replay_p5_quota_failures import (  # noqa: E402
    ADMISSION_PENDING,
    CloudAudit,
    canonical_sha,
)
from wsm_settings import (  # noqa: E402
    INTERNAL_TRAINING_ROOT,
    LONG_CONTEXT_STUDY_S3,
    RESULTS_BUCKET,
    ROBOMME_EVAL_ROOT,
)

REGION = "us-west-2"
SOURCE_ID = "6039c667-3e85-4135-a565-5d9c0ca3aeb3"
TERMINATED_JOB_ID = "026b4621-75d3-458b-9719-296454c4f1b7"
MOVE_CUBE_JOB_ID = "904d99e0-5886-4e8e-8358-6c8548657824"
RUN_ID = "st-v1-videorepick-a6-seed0-3715d6bcdfcdf0f1"
ATTEMPT_ID = f"{RUN_ID}-attempt2"
RECOVERY_STEP = 15000
SOURCE_ARCHIVE = (
    f"s3://{RESULTS_BUCKET}/sarvesh-rmme-VideoRepic-a6-3715d6bcdfcdf0f1-0806-135410/source/sourcedir.tar.gz"
)
SOURCE_ARCHIVE_SHA256 = "c28d7a0182852f4c1e9ced34e63c49f9280cfba28e11eb19e0cb4c50e6ee9639"
SOURCE_TREE_SHA256 = "398e89b11de0d70c7417722f717ebb441cdc46e9a7e27bef1323ce9cc4de5c9c"
MANIFEST_SHA256 = "2547f9a17d136e8e57fd207854f2bd626078ed302e2d8f421d6708c90f036887"
OUTPUT = f"{LONG_CONTEXT_STUDY_S3}/checkpoints/robomme/pi05/single_task_v1/VideoRepick/a6/seed0/{RUN_ID}"
TERMINATION_REASON = (
    "User-approved termination: VideoRepick A6 hung after CUDA graph error and 8-GPU "
    "rendezvous deadlock; resume from verified step 15000"
)
DEFAULT_RUNTIME = ROBOMME_EVAL_ROOT / "state"
PYTHON = INTERNAL_TRAINING_ROOT / ".venv" / "bin" / "python"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_json(*args: str) -> dict[str, Any]:
    command = ["aws", *args, "--region", REGION, "--output", "json", "--no-cli-pager"]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def describe(job_id: str) -> dict[str, Any]:
    return run_json("batch", "describe-service-job", "--job-id", job_id)


def s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise RuntimeError(f"invalid S3 URI {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def object_exists(client: Any, uri: str) -> bool:
    from botocore.exceptions import ClientError

    bucket, key = s3_parts(uri)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def read_s3_json(client: Any, uri: str) -> dict[str, Any]:
    bucket, key = s3_parts(uri)
    return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare_source(root: Path) -> Path:
    archive = root / "sourcedir.tar.gz"
    source = root / "source"
    subprocess.run(
        ["aws", "s3", "cp", SOURCE_ARCHIVE, str(archive), "--only-show-errors", "--region", REGION],
        check=True,
    )
    actual = sha256(archive)
    if actual != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(f"source archive digest mismatch: {actual}")
    source.mkdir()
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(source)], check=True)
    staged_manifest = source / launch.STAGED_MANIFEST
    if not staged_manifest.is_file():
        raise RuntimeError("attempt-1 source archive has no staged run manifest")
    staged_manifest.unlink()
    actual = launch.source_tree_sha256(source)
    if actual != SOURCE_TREE_SHA256:
        raise RuntimeError(f"recovered scientific source digest mismatch: {actual}")
    return source


def build_plan(source: Path) -> tuple[Any, dict[str, Any], list[str]]:
    cli = [
        "--scope",
        "single_task",
        "--task",
        "VideoRepick",
        "--arm",
        "a6",
        "--hardware",
        "p5",
        "--priority",
        "1",
        "--attempt-index",
        "2",
        "--source-dir",
        str(source),
        "--dry-run",
    ]
    args = launch.parser().parse_args(cli)
    args.queue = launch.HARDWARE[args.hardware]["queue"]
    plan = launch.build_plan(args, source)
    expected = {
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "output": OUTPUT,
        "source_sha": SOURCE_TREE_SHA256,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"attempt-2 plan drifted at {key}: {plan.get(key)!r}")
    if plan["manifest"]["manifest_sha256"] != MANIFEST_SHA256:
        raise RuntimeError("attempt-2 run-manifest digest drifted")
    submit_cli = [item for item in cli if item != "--dry-run"] + ["--confirm-submit"]
    return args, plan, submit_cli


def verify_recovery(cloud: CloudAudit, plan: dict[str, Any]) -> None:
    old = describe(TERMINATED_JOB_ID)
    if old.get("status") != "FAILED" or old.get("statusReason") != TERMINATION_REASON:
        raise RuntimeError("terminated A6 job no longer matches the approved failure record")
    if len(old.get("attempts", [])) != 1:
        raise RuntimeError("terminated A6 job does not have exactly one SageMaker attempt")
    latest = read_s3_json(cloud.s3, f"{OUTPUT}/LATEST.json")
    if latest != {"schema_version": 1, "step": RECOVERY_STEP}:
        raise RuntimeError(f"unexpected recovery pointer: {latest}")
    marker = read_s3_json(cloud.s3, f"{OUTPUT}/steps/{RECOVERY_STEP}/_UPLOAD_COMPLETE.json")
    if marker.get("step") != RECOVERY_STEP:
        raise RuntimeError("step-15000 upload marker is invalid")
    for label, uri in (
        ("attempt-2 run manifest", plan["manifest_s3"]),
        ("attempt-2 producer claim", plan["environment"]["PRODUCER_CLAIM_S3"]),
        ("final completion claim", plan["environment"]["COMPLETION_CLAIM_S3"]),
    ):
        if object_exists(cloud.s3, uri):
            raise RuntimeError(f"unexpected {label} already exists at {uri}")


def register_submission(
    state_path: Path,
    control_path: Path,
    plan: dict[str, Any],
    job_id: str,
) -> None:
    record = describe(job_id)
    raw_payload = record["serviceRequestPayload"]
    payload = json.loads(raw_payload)
    control = {
        "schema_version": 1,
        "job_id": job_id,
        "job_arn": record["jobArn"],
        "job_name": record["jobName"],
        "attempt_id": ATTEMPT_ID,
        "run_id": RUN_ID,
        "manifest_sha256": MANIFEST_SHA256,
        "recovery_step": RECOVERY_STEP,
        "submitted_at": utc(),
    }
    atomic_json(control_path, control)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        source = state["sources"][SOURCE_ID]
        if source["run_id"] != RUN_ID:
            raise RuntimeError("dispatcher source identity drifted")
        old = source["replays"][0]
        if old["job_id"] != TERMINATED_JOB_ID:
            raise RuntimeError("dispatcher no longer contains the expected terminated replay")
        old_record = describe(TERMINATED_JOB_ID)
        old.update(
            status=old_record["status"],
            status_reason=old_record.get("statusReason"),
            attempt_count=len(old_record.get("attempts", [])),
            observed_at=utc(),
        )
        existing = [item for item in source["replays"] if item.get("job_id") == job_id]
        if not existing:
            replay = {
                "generation": 2,
                "job_id": job_id,
                "job_arn": record["jobArn"],
                "job_name": record["jobName"],
                "training_job_name": payload["TrainingJobName"],
                "attempt_id": ATTEMPT_ID,
                "manifest_sha256": plan["manifest"]["manifest_sha256"],
                "source_payload_sha256": canonical_sha(raw_payload),
                "submitted_payload_sha256": canonical_sha(raw_payload),
                "resume_step": RECOVERY_STEP,
                "submitted_at": control["submitted_at"],
                "status": record["status"],
            }
            source["replays"].append(replay)
        status = record["status"]
        source["disposition"] = (
            "replay_running"
            if status == "RUNNING"
            else "replay_admission_pending"
            if status in ADMISSION_PENDING
            else "replay_succeeded"
            if status == "SUCCEEDED"
            else "manual_recovery_failed"
        )
        state["updated_at"] = utc()
        atomic_json(state_path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)


def resume_dispatcher(state_path: Path, log_path: Path) -> None:
    command = [
        str(PYTHON),
        "-u",
        str(REPO / "robomme_integration/sweeps/replay_p5_quota_failures.py"),
        "--confirm-submit",
        "--state",
        str(state_path),
        "--log",
        str(log_path),
        "--quota-limit",
        "10",
        "--safety-headroom",
        "1",
        "--poll-seconds",
        "120",
        "--max-admission-retries",
        "10",
    ]
    print(f"{utc()} handing control to quota dispatcher", flush=True)
    os.execv(command[0], command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_RUNTIME / "p5_priority1_retry_v1.json")
    parser.add_argument("--dispatcher-log", type=Path, default=DEFAULT_RUNTIME / "p5_priority1_retry_v1.log")
    parser.add_argument("--control", type=Path, default=DEFAULT_RUNTIME / "videorepick_a6_attempt2.json")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--confirm-submit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_submit:
        raise SystemExit("submission blocked: obtain explicit user approval, then pass --confirm-submit")
    if args.poll_seconds < 30:
        raise SystemExit("--poll-seconds must be at least 30")
    if not PYTHON.is_file():
        raise SystemExit(f"missing SageMaker Python runtime {PYTHON}")
    if args.control.exists():
        control = json.loads(args.control.read_text())
        if control.get("attempt_id") != ATTEMPT_ID:
            raise SystemExit("recovery control file belongs to a different attempt")
        register_submission(
            args.state, args.control, {"manifest": {"manifest_sha256": MANIFEST_SHA256}}, control["job_id"]
        )
        resume_dispatcher(args.state, args.dispatcher_log)

    cloud = CloudAudit()
    with tempfile.TemporaryDirectory(prefix="robomme-videorepick-a6-resume-") as temporary:
        source = prepare_source(Path(temporary))
        _args, plan, submit_cli = build_plan(source)
        verify_recovery(cloud, plan)
        print(f"{utc()} recovery preflight passed; waiting for MoveCube/Q0 admission", flush=True)
        while True:
            move = describe(MOVE_CUBE_JOB_ID)
            p5_count, names = cloud.p5_in_progress()
            safe = p5_count <= 8
            print(
                f"{utc()} move_cube={move['status']} p5_in_progress={p5_count} safe_slot={str(safe).lower()}",
                flush=True,
            )
            if move["status"] not in ADMISSION_PENDING and safe:
                break
            time.sleep(args.poll_seconds)
        verify_recovery(cloud, plan)
        command = [str(PYTHON), str(REPO / "robomme_integration/launch.py"), *submit_cli]
        result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"attempt-2 launcher failed with {result.returncode}")
        match = re.search(r"QUEUED arn=(arn:aws:batch:[^\s]+/([0-9a-f-]{36}))", result.stdout)
        if not match:
            raise RuntimeError("attempt-2 launcher returned no parseable Batch ARN")
        job_id = match.group(2)
        register_submission(args.state, args.control, plan, job_id)
        print(f"{utc()} registered VideoRepick/A6 attempt2 Batch job {job_id}", flush=True)
    resume_dispatcher(args.state, args.dispatcher_log)


if __name__ == "__main__":
    main()
