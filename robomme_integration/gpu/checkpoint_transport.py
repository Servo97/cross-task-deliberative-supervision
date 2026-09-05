#!/usr/bin/env python3
"""Atomic-enough S3 transport for finalized Orbax checkpoints.

Only numeric directories with Orbax's completion marker are eligible.  The remote completion
marker is written after ``aws s3 sync`` and ``LATEST.json`` is updated last, so restore never
selects an in-flight tree.  At most the newest recovery generation (plus an optional preregistered
milestone) is retained while training; successful completion prunes to the final generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

COMPLETE_MARKER = "_CHECKPOINT_METADATA"
REMOTE_COMPLETE_MARKER = "_UPLOAD_COMPLETE.json"


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def complete_steps(root: Path) -> list[int]:
    if not root.exists():
        return []
    return sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / COMPLETE_MARKER).is_file()
    )


def tree_summary(root: Path) -> dict[str, Any]:
    """Cheap structural receipt; AWS CLI supplies transport checksums for file contents."""
    digest = hashlib.sha256()
    count = 0
    total = 0
    metadata_sha256 = None
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or "orbax-checkpoint-tmp" in path.as_posix():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == REMOTE_COMPLETE_MARKER:
            continue
        size = path.stat().st_size
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        count += 1
        total += size
        if relative == COMPLETE_MARKER:
            metadata_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if count < 1 or metadata_sha256 is None:
        raise ValueError(f"checkpoint tree is not finalized: {root}")
    return {
        "files": count,
        "bytes": total,
        "layout_sha256": digest.hexdigest(),
        "checkpoint_metadata_sha256": metadata_sha256,
    }


def retention_set(remote_steps: Sequence[int], *, latest: int, milestones: set[int], final_step: int) -> set[int]:
    keep = {latest}
    keep.update(step for step in remote_steps if step in milestones)
    if final_step in remote_steps:
        keep.add(final_step)
    return keep


class S3Transport:
    def __init__(
        self,
        run_root: str,
        *,
        max_attempts: int = 8,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        _parse_s3(run_root)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.run_root = run_root.rstrip("/")
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.runner = runner
        self.sleeper = sleeper

    def run(self, arguments: Sequence[str]) -> str:
        command = ["aws", *arguments, "--region", "us-west-2"]
        for attempt in range(1, self.max_attempts + 1):
            result = self.runner(command, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            detail = " ".join((result.stderr or result.stdout or "").split())[:500] or "<no output>"
            if attempt == self.max_attempts:
                raise RuntimeError(
                    f"AWS checkpoint operation failed after {attempt} attempts "
                    f"(returncode={result.returncode}, operation={' '.join(arguments[:2])}): {detail}"
                )
            delay = min(self.retry_base_seconds * (2 ** (attempt - 1)), self.retry_max_seconds)
            print(
                f"[gpu-checkpoint] retry={attempt}/{self.max_attempts} delay={delay:g}s detail={detail}",
                flush=True,
            )
            self.sleeper(delay)
        raise AssertionError("unreachable")

    def read_json(self, uri: str) -> dict[str, Any]:
        value = json.loads(self.run(["s3", "cp", uri, "-", "--only-show-errors"]))
        if not isinstance(value, dict):
            raise ValueError(f"{uri} did not contain a JSON object")
        return value

    @staticmethod
    def _not_found(error: BaseException) -> bool:
        message = str(error).casefold()
        return any(marker in message for marker in ("nosuchkey", "not found", "404", "does not exist"))

    def _put_json(self, value: dict, uri: str, *, create_once: bool) -> None:
        bucket, key = _parse_s3(uri)
        payload = _canonical_bytes(value)
        fd, name = tempfile.mkstemp(prefix="robomme-s3-json-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            arguments = ["s3api", "put-object", "--bucket", bucket, "--key", key, "--body", name]
            if create_once:
                arguments.extend(["--if-none-match", "*"])
            try:
                self.run(arguments)
            except RuntimeError:
                if not create_once:
                    raise
                prior = self.read_json(uri)
                if _canonical_bytes(prior) != payload:
                    raise ValueError(f"immutable S3 JSON collision at {uri}")
        finally:
            Path(name).unlink(missing_ok=True)

    def upload_step(self, local_root: Path, step: int) -> dict[str, Any]:
        source = local_root / str(step)
        summary = tree_summary(source)
        destination = f"{self.run_root}/steps/{step}"
        self.run(
            [
                "s3",
                "sync",
                str(source),
                destination,
                "--only-show-errors",
                "--no-follow-symlinks",
                "--exclude",
                "*orbax-checkpoint-tmp*",
                "--delete",
            ]
        )
        marker = {
            "schema_version": 1,
            "step": step,
            "source_marker": COMPLETE_MARKER,
            "tree": summary,
        }
        self._put_json(marker, f"{destination}/{REMOTE_COMPLETE_MARKER}", create_once=True)
        print(f"[gpu-checkpoint] uploaded step={step} bytes={summary['bytes']}", flush=True)
        return marker

    def publish_latest(self, step: int) -> None:
        self._put_json({"schema_version": 1, "step": step}, f"{self.run_root}/LATEST.json", create_once=False)

    def list_steps(self) -> set[int]:
        bucket, key = _parse_s3(f"{self.run_root}/steps")
        output = self.run(
            [
                "s3api",
                "list-objects-v2",
                "--bucket",
                bucket,
                "--prefix",
                key.rstrip("/") + "/",
                "--delimiter",
                "/",
                "--output",
                "json",
            ]
        )
        value = json.loads(output or "{}")
        candidates = set()
        for record in value.get("CommonPrefixes", []):
            candidate = str(record.get("Prefix", "")).rstrip("/").rsplit("/", 1)[-1]
            if candidate.isdigit():
                candidates.add(int(candidate))
        # A prefix can exist while ``aws s3 sync`` is still copying an interrupted generation.
        # Only the post-sync marker makes a step restorable and eligible to become LATEST.
        return {
            step for step in candidates if self.object_exists(f"{self.run_root}/steps/{step}/{REMOTE_COMPLETE_MARKER}")
        }

    def object_exists(self, uri: str) -> bool:
        bucket, key = _parse_s3(uri)
        command = [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--region",
            "us-west-2",
        ]
        for attempt in range(1, self.max_attempts + 1):
            result = self.runner(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
            detail = " ".join((result.stderr or result.stdout or "").split())[:500]
            if any(marker in detail.casefold() for marker in ("404", "not found", "nosuchkey")):
                return False
            if attempt == self.max_attempts:
                raise RuntimeError(
                    f"S3 head-object failed for {uri} after {attempt} attempts: {detail or '<no output>'}"
                )
            delay = min(self.retry_base_seconds * (2 ** (attempt - 1)), self.retry_max_seconds)
            print(
                f"[gpu-checkpoint] retry={attempt}/{self.max_attempts} "
                f"delay={delay:g}s detail={detail or '<no output>'}",
                flush=True,
            )
            self.sleeper(delay)
        raise AssertionError("unreachable")

    def delete_step(self, step: int) -> None:
        self.run(["s3", "rm", f"{self.run_root}/steps/{step}", "--recursive", "--only-show-errors"])
        print(f"[gpu-checkpoint] pruned step={step}", flush=True)

    def restore_latest(self, local_root: Path) -> int | None:
        latest_uri = f"{self.run_root}/LATEST.json"
        # A missing pointer is the expected state for a brand-new run, not a transient AWS
        # failure.  Probe once so fresh jobs do not spend the full retry budget on a 404.
        if not self.object_exists(latest_uri):
            return None
        try:
            latest = self.read_json(latest_uri)
        except RuntimeError as error:
            if self._not_found(error):
                return None
            raise
        step = int(latest["step"])
        remote = f"{self.run_root}/steps/{step}"
        marker = self.read_json(f"{remote}/{REMOTE_COMPLETE_MARKER}")
        if int(marker.get("step", -1)) != step:
            raise ValueError(f"remote completion marker step mismatch for {step}")
        destination = local_root / str(step)
        if not (destination / COMPLETE_MARKER).is_file():
            # A killed restore may leave extra bytes that do not exist remotely.  Starting from an
            # empty generation makes the structural receipt authoritative and keeps resume safe.
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            self.run(["s3", "sync", remote, str(destination), "--only-show-errors"])
            (destination / REMOTE_COMPLETE_MARKER).unlink(missing_ok=True)
        actual = tree_summary(destination)
        if actual != marker.get("tree"):
            raise ValueError(f"restored checkpoint receipt mismatch for step {step}")
        return step


class CheckpointWatcher:
    def __init__(
        self,
        local_root: Path,
        transport: S3Transport,
        state_path: Path,
        *,
        milestones: set[int],
        final_step: int,
    ) -> None:
        self.local_root = local_root
        self.transport = transport
        self.state_path = state_path
        self.milestones = milestones
        self.final_step = final_step
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "uploaded_steps": [], "latest": None}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def sync_once(self) -> bool:
        complete = complete_steps(self.local_root)
        if not complete:
            return False
        newest = complete[-1]
        remote = self.transport.list_steps()
        if newest not in remote:
            self.transport.upload_step(self.local_root, newest)
            remote.add(newest)
        self.transport.publish_latest(newest)
        keep = retention_set(sorted(remote), latest=newest, milestones=self.milestones, final_step=self.final_step)
        for old in sorted(remote - keep):
            self.transport.delete_step(old)
            remote.remove(old)
        self.state = {"schema_version": 1, "uploaded_steps": sorted(remote), "latest": newest}
        self._save()
        return True

    def finalize_success(self, *, success_milestones: set[int] | None = None) -> None:
        self.sync_once()
        remote = self.transport.list_steps()
        if self.final_step not in remote:
            raise RuntimeError(f"final step {self.final_step} was not uploaded")
        marker = self.transport.read_json(
            f"{self.transport.run_root}/steps/{self.final_step}/{REMOTE_COMPLETE_MARKER}"
        )
        if int(marker.get("step", -1)) != self.final_step:
            raise RuntimeError("final remote checkpoint marker is invalid")
        keep = {self.final_step}
        if success_milestones is not None:
            missing = success_milestones - remote
            if missing:
                raise RuntimeError(f"required scientific checkpoint milestones were not uploaded: {sorted(missing)}")
            keep.update(success_milestones)
        self.transport.publish_latest(self.final_step)
        for old in sorted(remote - keep):
            self.transport.delete_step(old)
        self.state = {
            "schema_version": 1,
            "uploaded_steps": sorted(keep),
            "latest": self.final_step,
        }
        self._save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--s3-run-root", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--milestones", default="10000")
    parser.add_argument(
        "--success-milestones",
        default="",
        help="optional milestones retained after success; all must already be finalized remotely",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--finalize-success", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise SystemExit("--poll-seconds must be at least 10")
    watcher = CheckpointWatcher(
        args.local_root,
        S3Transport(args.s3_run_root),
        args.state,
        milestones={int(value) for value in args.milestones.split(",") if value},
        final_step=args.final_step,
    )
    if args.finalize_success:
        watcher.finalize_success(
            success_milestones=(
                {int(value) for value in args.success_milestones.split(",") if value}
                if args.success_milestones
                else None
            )
        )
        return 0
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        watcher.sync_once()
        if args.once or stopping:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
