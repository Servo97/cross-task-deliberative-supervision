#!/usr/bin/env python3
"""Copy only the two compact RoboMME workspace artifacts from GCS to canonical S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from plan_p5_priority1 import DEFAULT_SPEC, _load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wsm_settings import LONG_CONTEXT_STUDY_S3, STUDY_OWNER, TRI_ROOT  # noqa: E402

GCS_ROOT = f"gs://maxlab-tpu-wsm-robocasa-us-east1/{STUDY_OWNER}/wsm_robocasa/robomme/single_task_poc/v1/workspace"
S3_STUDY_ROOT = LONG_CONTEXT_STUDY_S3
SCRATCH = TRI_ROOT / ".robomme_workspace_bridge"
EXPECTED_TOTAL_BYTES = 598_471_550


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=capture)


def _remote_bytes(uri: str) -> bytes | None:
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--only-show-errors", "--region", "us-west-2"],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout
    return None


def _publish_once(path: Path, uri: str) -> None:
    prior = _remote_bytes(uri)
    payload = path.read_bytes()
    if prior is not None:
        if prior != payload:
            raise RuntimeError(f"immutable artifact collision at {uri}")
        return
    parsed = urlparse(uri)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            parsed.netloc,
            "--key",
            parsed.path.lstrip("/"),
            "--body",
            str(path),
            "--if-none-match",
            "*",
            "--region",
            "us-west-2",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode and _remote_bytes(uri) != payload:
        raise RuntimeError(f"failed to publish immutable artifact seal at {uri}")


def _records(spec: dict) -> list[dict]:
    records = []
    for task, artifact in spec["workspace"]["tasks"].items():
        encoder = artifact["encoder_id"]
        destination = f"{S3_STUDY_ROOT}/artifacts/robomme/workspace/{task}/{encoder}"
        records.extend(
            [
                {
                    "task": task,
                    "kind": "omega",
                    "source": f"{GCS_ROOT}/omega/deliberative_v1/{task}",
                    "destination": f"{destination}/omega",
                    "seal": "MANIFEST.json",
                    "seal_sha256": artifact["omega_manifest_file_sha256"],
                },
                {
                    "task": task,
                    "kind": "supervision",
                    "source": f"{GCS_ROOT}/supervision/v1/{task}",
                    "destination": f"{destination}/supervision",
                    "seal": "MANIFEST.json",
                    "seal_sha256": artifact["supervision_manifest_file_sha256"],
                },
                {
                    "task": task,
                    "kind": "representation",
                    "source": (f"{GCS_ROOT}/representations/deliberative_v1/{task}/seed0/steps/10000"),
                    "destination": f"{destination}/representation/step-10000",
                    "seal": "WSM_GENERATION_COMPLETE.json",
                },
            ]
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--confirm-copy", action="store_true")
    args = parser.parse_args()
    records = _records(_load(args.spec))
    print(
        json.dumps(
            {
                "artifact_groups": len(records),
                "expected_total_bytes": EXPECTED_TOTAL_BYTES,
                "scratch": str(SCRATCH),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.confirm_copy:
        print("DRY PLAN ONLY — pass --confirm-copy only after explicit user approval")
        return
    if SCRATCH.exists():
        raise SystemExit(f"scratch already exists; inspect before retrying: {SCRATCH}")
    try:
        for record in records:
            local = SCRATCH / record["task"] / record["kind"]
            local.mkdir(parents=True)
            _run(["gcloud", "storage", "rsync", "--recursive", record["source"], str(local)])
            seal = local / record["seal"]
            if not seal.is_file():
                raise RuntimeError(f"source seal missing: {seal}")
            expected = record.get("seal_sha256")
            if expected and _sha256(seal) != expected:
                raise RuntimeError(f"source seal SHA mismatch: {record['source']}")
            remote_seal = f"{record['destination']}/{record['seal']}"
            prior = _remote_bytes(remote_seal)
            if prior is not None:
                if prior != seal.read_bytes():
                    raise RuntimeError(f"immutable artifact collision at {remote_seal}")
                continue
            _run(
                [
                    "aws",
                    "s3",
                    "sync",
                    str(local),
                    record["destination"],
                    "--exclude",
                    record["seal"],
                    "--only-show-errors",
                    "--no-follow-symlinks",
                    "--region",
                    "us-west-2",
                ]
            )
            _publish_once(seal, remote_seal)
            if _remote_bytes(remote_seal) != seal.read_bytes():
                raise RuntimeError(f"remote artifact seal verification failed: {remote_seal}")
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print("WORKSPACE ARTIFACT BRIDGE COMPLETE")


if __name__ == "__main__":
    main()
