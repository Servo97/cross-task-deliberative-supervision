"""Produce two uniform task-specific RoboMME workspace bundles on one 8-GPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from .single_task import TASK_EPISODES, task_manifest_sha256
except ImportError:  # SageMaker stages robomme_integration/ contents as top-level packages.
    from training.single_task import TASK_EPISODES, task_manifest_sha256


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(retries={"mode": "adaptive", "max_attempts": 10}))


def _remote_bytes(uri: str) -> bytes | None:
    bucket, key = _parse_s3(uri)
    client = _s3_client()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        return None
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()


def _put_once(payload: bytes, uri: str) -> None:
    prior = _remote_bytes(uri)
    if prior is not None:
        if prior != payload:
            raise RuntimeError(f"immutable artifact collision at {uri}")
        return
    bucket, key = _parse_s3(uri)
    client = _s3_client()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=payload, IfNoneMatch="*")
    except Exception:
        if _remote_bytes(uri) != payload:
            raise


def _sync_tree(root: Path, uri: str, *, seal: str) -> str:
    seal_path = root / seal
    if not seal_path.is_file():
        raise RuntimeError(f"artifact seal is missing: {seal_path}")
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(root),
            uri,
            "--exclude",
            seal,
            "--only-show-errors",
            "--no-follow-symlinks",
            "--region",
            "us-west-2",
        ],
        check=True,
    )
    payload = seal_path.read_bytes()
    _put_once(payload, f"{uri}/{seal}")
    return hashlib.sha256(payload).hexdigest()


def _verified_existing_claim(
    uri: str,
    *,
    task: str,
    run_id: str,
    pair_id: str,
    source_tree_sha256: str,
) -> dict | None:
    payload = _remote_bytes(uri)
    if payload is None:
        return None
    claim = json.loads(payload)
    if (
        claim.get("task") != task
        or claim.get("run_id") != run_id
        or claim.get("pair_id") != pair_id
        or claim.get("source_tree_sha256") != source_tree_sha256
    ):
        raise RuntimeError(f"workspace completion claim identity mismatch at {uri}")
    if claim.get("task_manifest_sha256") != task_manifest_sha256(task):
        raise RuntimeError(f"workspace completion claim task manifest mismatch at {uri}")
    for kind, seal_name, sha_field in (
        ("omega", "MANIFEST.json", "manifest_sha256"),
        ("supervision", "MANIFEST.json", "manifest_sha256"),
        ("representation", "WSM_GENERATION_COMPLETE.json", "completion_sha256"),
    ):
        record = claim[kind]
        remote = _remote_bytes(f"{record['uri']}/{seal_name}")
        if remote is None or hashlib.sha256(remote).hexdigest() != record[sha_field]:
            raise RuntimeError(f"workspace completion claim has a missing/corrupt {kind} seal")
    return claim


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("[workspace-producer] exec " + " ".join(command), flush=True)
    subprocess.run(command, env=environment, check=True)


def _compute_python() -> str:
    value = os.environ.get("ROBOMME_WORKSPACE_COMPUTE_PYTHON")
    if not value:
        raise RuntimeError("ROBOMME_WORKSPACE_COMPUTE_PYTHON is required")
    path = Path(value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"workspace compute interpreter is not executable: {path}")
    return str(path)


def produce_task(args: argparse.Namespace) -> dict:
    if args.task not in TASK_EPISODES:
        raise ValueError(f"unknown RoboMME task {args.task}")
    existing = _verified_existing_claim(
        args.claim_s3,
        task=args.task,
        run_id=args.run_id,
        pair_id=args.pair_id,
        source_tree_sha256=args.source_tree_sha256,
    )
    if existing is not None:
        print(f"WORKSPACE TASK ALREADY COMPLETE task={args.task} run_id={args.run_id}", flush=True)
        return existing

    task_root = Path(args.work_root) / args.task
    data = task_root / "data"
    upstream = task_root / "upstream"
    supervision = task_root / "supervision"
    representation = task_root / "representation"
    omega = task_root / "omega"
    task_root.mkdir(parents=True, exist_ok=True)
    # The SageMaker control interpreter owns boto3 and credential-chain access.  Keep the uv-locked
    # OpenPI interpreter isolated for NumPy/JAX work so the runtime site cannot shadow its ABI.
    control_python = sys.executable
    compute_python = _compute_python()
    _run(
        [
            control_python,
            "-m",
            "fleet.task_inventory",
            "--parent-manifest",
            args.parent_manifest,
            "--task",
            args.task,
            "--root-s3",
            args.data_s3,
            "--destination",
            str(data),
            "--expected-derived-sha256",
            args.task_inventory_sha256,
            "--workers",
            "32",
        ]
    )
    _run(
        [
            compute_python,
            "-m",
            "training.upstream_feature_cache",
            "--task",
            args.task,
            "--lerobot-root",
            str(data),
            "--output-root",
            str(upstream),
            "--download-workers",
            "8",
        ]
    )
    _run(
        [
            compute_python,
            "-m",
            "training.workspace_supervision_cache",
            "--task",
            args.task,
            "--lerobot-root",
            str(data),
            "--upstream-cache-root",
            str(upstream),
            "--output-root",
            str(supervision),
        ]
    )
    environment = os.environ.copy()
    environment["WSM_WSM_REP_ALLOW_RUN"] = "1"
    environment["WSM_WSM_MATERIALIZE_ALLOW_RUN"] = "1"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"
    _run(
        [
            compute_python,
            "-m",
            "training.workspace_deliberative",
            "--task",
            args.task,
            "--supervision-root",
            str(supervision),
            "--output-root",
            str(representation),
            "--steps",
            "10000",
            "--batch-size",
            "32",
            "--expected-devices",
            "4",
        ],
        environment=environment,
    )
    _run(
        [
            compute_python,
            "-m",
            "training.workspace_materialize",
            "--task",
            args.task,
            "--supervision-root",
            str(supervision),
            "--train-root",
            str(representation),
            "--output-root",
            str(omega),
            "--batch-size",
            "32",
            "--expected-devices",
            "4",
        ],
        environment=environment,
    )

    omega_root = omega / args.task
    supervision_root = supervision / args.task
    train_root = representation / args.task
    omega_manifest = json.loads((omega_root / "MANIFEST.json").read_text(encoding="utf-8"))
    encoder_id = omega_manifest["encoder_id"]
    if len(encoder_id) != 64:
        raise RuntimeError("materializer returned an invalid encoder_id")
    best = json.loads((train_root / "BEST.json").read_text(encoding="utf-8"))
    step = int(best["best_step"])
    generation = train_root / "checkpoints" / str(step)
    if not generation.is_dir():
        raise RuntimeError(f"best representation checkpoint is absent: {generation}")

    artifact_root = f"{args.artifact_root_s3}/{args.task}/{encoder_id}"
    omega_uri = f"{artifact_root}/omega"
    supervision_uri = f"{artifact_root}/supervision"
    representation_uri = f"{artifact_root}/representation/step-{step}"
    omega_sha = _sync_tree(omega_root, omega_uri, seal="MANIFEST.json")
    supervision_sha = _sync_tree(supervision_root, supervision_uri, seal="MANIFEST.json")
    representation_sha = _sync_tree(
        generation,
        representation_uri,
        seal="WSM_GENERATION_COMPLETE.json",
    )
    claim = {
        "schema_version": 1,
        "kind": "robomme_all16_workspace_task_complete",
        "campaign": "uniform_gpu_v1",
        "task": args.task,
        "run_id": args.run_id,
        "pair_id": args.pair_id,
        "task_manifest_sha256": task_manifest_sha256(args.task),
        "source_tree_sha256": args.source_tree_sha256,
        "encoder_id": encoder_id,
        "omega": {"uri": omega_uri, "manifest_sha256": omega_sha},
        "supervision": {"uri": supervision_uri, "manifest_sha256": supervision_sha},
        "representation": {
            "uri": representation_uri,
            "step": step,
            "completion_sha256": representation_sha,
        },
    }
    _put_once(_canonical(claim), args.claim_s3)
    verified = _verified_existing_claim(
        args.claim_s3,
        task=args.task,
        run_id=args.run_id,
        pair_id=args.pair_id,
        source_tree_sha256=args.source_tree_sha256,
    )
    if verified != claim:
        raise RuntimeError("workspace task completion claim failed post-publication verification")
    shutil.rmtree(task_root)
    print(
        f"WORKSPACE TASK COMPLETE task={args.task} run_id={args.run_id} encoder_id={encoder_id}",
        flush=True,
    )
    return claim


def produce_pair(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("pair_id") != args.pair_id or manifest.get("source_tree_sha256") != args.source_tree_sha256:
        raise RuntimeError("workspace pair manifest identity mismatch")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 2:
        raise RuntimeError("workspace pair manifest must contain exactly two tasks")
    processes = []
    for lane, record in enumerate(tasks):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" if lane == 0 else "4,5,6,7"
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        command = [
            sys.executable,
            "-m",
            "training.workspace_gpu_producer",
            "task",
            "--task",
            record["task"],
            "--run-id",
            record["run_id"],
            "--pair-id",
            args.pair_id,
            "--source-tree-sha256",
            args.source_tree_sha256,
            "--task-inventory-sha256",
            record["task_inventory_sha256"],
            "--claim-s3",
            record["claim_s3"],
            "--parent-manifest",
            args.parent_manifest,
            "--data-s3",
            args.data_s3,
            "--artifact-root-s3",
            args.artifact_root_s3,
            "--work-root",
            args.work_root,
        ]
        # Keep both children in the SageMaker process group so a platform SIGTERM reaches the
        # active JAX workers as well as this coordinator.  Task claims are sealed independently,
        # so a retry skips a lane that completed before the other lane failed.
        processes.append((record, subprocess.Popen(command, env=environment)))
    failures = []
    for record, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append((record["task"], returncode))
    if failures:
        raise RuntimeError(f"workspace pair lanes failed: {failures}")
    task_claims = []
    for record in tasks:
        payload = _remote_bytes(record["claim_s3"])
        if payload is None:
            raise RuntimeError(f"workspace task claim absent after successful lane: {record['task']}")
        task_claims.append(
            {
                "task": record["task"],
                "claim_s3": record["claim_s3"],
                "claim_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    pair_claim = {
        "schema_version": 1,
        "kind": "robomme_all16_workspace_pair_complete",
        "campaign": "uniform_gpu_v1",
        "pair_id": args.pair_id,
        "source_tree_sha256": args.source_tree_sha256,
        "tasks": task_claims,
    }
    _put_once(_canonical(pair_claim), args.pair_claim_s3)
    print(f"WORKSPACE PAIR COMPLETE pair_id={args.pair_id}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="mode", required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--manifest", required=True)
    pair.add_argument("--pair-id", required=True)
    pair.add_argument("--source-tree-sha256", required=True)
    pair.add_argument("--pair-claim-s3", required=True)
    pair.add_argument("--parent-manifest", required=True)
    pair.add_argument("--data-s3", required=True)
    pair.add_argument("--artifact-root-s3", required=True)
    pair.add_argument("--work-root", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--task", required=True)
    task.add_argument("--run-id", required=True)
    task.add_argument("--pair-id", required=True)
    task.add_argument("--source-tree-sha256", required=True)
    task.add_argument("--task-inventory-sha256", required=True)
    task.add_argument("--claim-s3", required=True)
    task.add_argument("--parent-manifest", required=True)
    task.add_argument("--data-s3", required=True)
    task.add_argument("--artifact-root-s3", required=True)
    task.add_argument("--work-root", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.mode == "pair":
        return produce_pair(args)
    produce_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
