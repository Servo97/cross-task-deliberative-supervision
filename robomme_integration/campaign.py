#!/usr/bin/env python3
"""Run a sealed sequence of RoboMME single-task training cells on one GPU node.

The individual training entry remains the scientific authority for checkpoints and completion
claims.  This supervisor only amortizes queue/provisioning time: it validates every staged cell,
skips already-complete *exact* cells, runs the rest serially, and removes each cell's ephemeral
work tree after its durable claim has been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlparse

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
SAFE_ENV = re.compile(r"^[A-Z][A-Z0-9_]*$")
CHILD_GROUP_DRAIN_TIMEOUT_SECONDS = 1_800.0
CHILD_GROUP_DRAIN_POLL_SECONDS = 0.1
FORBIDDEN_ENV = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "PYTHONHOME",
        "PYTHONPATH",
    }
)
TRAIN_ENTRY = "gpu_train_entry.sh"
CAMPAIGN_KIND = "robomme_single_task_train_series"
COMPLETION_KIND = "robomme_gpu_checkpoint_complete"
DEPLOY_KIND = "robomme_gpu_deploy_checkpoint_complete"
SUPPORTED_HARDWARE = {
    "p5": "ml.p5.48xlarge",
    "p5e": "ml.p5e.48xlarge",
}


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _seal_digest(value: dict) -> str:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _safe_s3(uri: object) -> str:
    if not isinstance(uri, str):
        raise ValueError("expected an S3 URI string")
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI {uri!r}")
    return uri.rstrip("/")


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("staged source path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe staged source path {value!r}")
    return value


def _require_hex(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_environment(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("cell environment must be a nonempty object")
    result: dict[str, str] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not SAFE_ENV.fullmatch(name) or name in FORBIDDEN_ENV:
            raise ValueError(f"unsafe cell environment name {name!r}")
        if not isinstance(raw, str) or "\x00" in raw:
            raise ValueError(f"cell environment value for {name} must be a NUL-free string")
        if len(raw.encode()) > 4096:
            raise ValueError(f"cell environment value for {name} is unexpectedly large")
        if any(marker in name for marker in ("SECRET", "PASSWORD", "TOKEN", "API_KEY")) and raw:
            # The Paligemma tokenizer variables are immutable artifact metadata, not credentials.
            if name not in {"PALIGEMMA_TOKENIZER_S3", "PALIGEMMA_TOKENIZER_SHA256"}:
                raise ValueError(f"plaintext secret-like campaign environment is forbidden: {name}")
        result[name] = raw
    return result


def validate_manifest(value: dict, *, code_dir: Path | None = None) -> dict:
    """Validate and return a normalized campaign manifest without touching cloud state."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported RoboMME campaign schema")
    if value.get("kind") != CAMPAIGN_KIND:
        raise ValueError(f"campaign kind must be {CAMPAIGN_KIND}")
    claimed = _require_hex(value.get("manifest_sha256"), "campaign manifest_sha256")
    actual = _seal_digest(value)
    if claimed != actual:
        raise ValueError(f"campaign manifest seal mismatch: {claimed} != {actual}")
    campaign_id = value.get("campaign_id")
    attempt_id = value.get("attempt_id")
    if not isinstance(campaign_id, str) or not SAFE_ID.fullmatch(campaign_id):
        raise ValueError("unsafe campaign_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith(f"{campaign_id}-attempt"):
        raise ValueError("campaign attempt_id is not run-scoped")
    suffix = attempt_id.removeprefix(f"{campaign_id}-attempt")
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError("campaign attempt_id requires a positive attempt index")

    infrastructure = value.get("infrastructure")
    if not isinstance(infrastructure, dict):
        raise ValueError("campaign infrastructure must be an object")
    hardware = infrastructure.get("hardware")
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(f"unsupported campaign hardware {hardware!r}")
    if infrastructure.get("instance_type") != SUPPORTED_HARDWARE[hardware]:
        raise ValueError("campaign hardware/instance type mismatch")
    maximum = infrastructure.get("max_run_seconds")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 86_400:
        raise ValueError("campaign max_run_seconds must lie in [1, 86400]")

    if value.get("failure_policy") not in {"stop", "continue"}:
        raise ValueError("campaign failure_policy must be stop or continue")
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("mode") != "deferred":
        raise ValueError("same-node evaluation is not certified: campaigns must use evaluation.mode=deferred")
    if hardware == "p5e" and evaluation.get("required_gate") != "p5_training_only":
        raise ValueError("p5e campaign must explicitly record the training-only evaluation gate")
    if hardware == "p5" and evaluation.get("required_gate") != "p5_native_render_reset_claim":
        raise ValueError("p5 campaign must explicitly record the native-render evaluation gate")

    cleanup = value.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("remove_cell_work_after_attempt") is not True:
        raise ValueError("campaign must remove ephemeral cell work after every attempted cell")
    minimum_free = cleanup.get("minimum_free_bytes")
    if not isinstance(minimum_free, int) or isinstance(minimum_free, bool) or minimum_free < 1:
        raise ValueError("cleanup.minimum_free_bytes must be positive")

    claims = value.get("claims")
    if not isinstance(claims, dict) or set(claims) != {"manifest", "attempt_result", "completion"}:
        raise ValueError("campaign claims must contain manifest, attempt_result, and completion")
    for uri in claims.values():
        _safe_s3(uri)

    cells = value.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("campaign must contain at least one cell")
    identities: set[tuple[str, str]] = set()
    run_ids: set[str] = set()
    for ordinal, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("ordinal") != ordinal:
            raise ValueError("campaign cells must have contiguous zero-based ordinals")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not SAFE_ID.fullmatch(cell_id):
            raise ValueError(f"unsafe campaign cell_id at ordinal {ordinal}")
        task, arm = cell.get("task"), cell.get("arm")
        if not isinstance(task, str) or not task or not isinstance(arm, str) or not arm:
            raise ValueError(f"campaign cell {cell_id} lacks task/arm")
        identity = (task, arm)
        if identity in identities:
            raise ValueError(f"duplicate campaign task/arm cell {identity}")
        identities.add(identity)
        run_id = cell.get("run_id")
        if not isinstance(run_id, str) or not run_id.startswith(("st-v1-", "st-v4-")) or run_id in run_ids:
            raise ValueError(f"invalid/duplicate single-task run_id for {cell_id}")
        run_ids.add(run_id)
        _require_hex(cell.get("scientific_spec_sha256"), f"{cell_id} scientific identity")
        _require_hex(cell.get("run_manifest_sha256"), f"{cell_id} run manifest")
        if cell.get("final_step") != 19_999:
            raise ValueError(f"{cell_id} must use the single-task final step 19999")
        output = _safe_s3(cell.get("output_s3"))
        claim = _safe_s3(cell.get("completion_claim_s3"))
        environment = _validate_environment(cell.get("environment"))
        if claim != environment.get("COMPLETION_CLAIM_S3"):
            raise ValueError(f"{cell_id} completion claim/environment mismatch")
        if output != environment.get("OUTPUT_S3"):
            raise ValueError(f"{cell_id} output/environment mismatch")
        expected_environment = {
            "ROBOMME_SCOPE": "single_task",
            "ROBOMME_TASK": task,
            "ROBOMME_ARM": arm,
            "ROBOMME_RUN_ID": run_id,
            "ROBOMME_FINAL_STEP": "19999",
            "WSM_MAX_STEPS": "20000",
            "OUTPUT_S3": output,
            "COMPLETION_CLAIM_S3": claim,
            "ROBOMME_SCIENTIFIC_SPEC_SHA256": cell["scientific_spec_sha256"],
            "RUN_MANIFEST_SHA256": cell["run_manifest_sha256"],
        }
        drift = {
            name: (environment.get(name), expected)
            for name, expected in expected_environment.items()
            if environment.get(name) != expected
        }
        if drift:
            raise ValueError(f"campaign cell {cell_id} environment drifted: {drift}")
        source = _safe_relative(cell.get("run_manifest_source"))
        if environment.get("RUN_MANIFEST_SOURCE") != source:
            raise ValueError(f"{cell_id} staged run-manifest path drifted")
        estimated = cell.get("estimated_train_seconds")
        if not isinstance(estimated, int) or isinstance(estimated, bool) or estimated < 1:
            raise ValueError(f"{cell_id} estimated_train_seconds must be positive")
        if code_dir is not None:
            staged = (code_dir / source).resolve()
            root = code_dir.resolve()
            if not staged.is_relative_to(root) or not staged.is_file():
                raise ValueError(f"{cell_id} staged run manifest is absent")
            payload = json.loads(staged.read_text(encoding="utf-8"))
            if payload.get("manifest_sha256") != cell["run_manifest_sha256"]:
                raise ValueError(f"{cell_id} staged run manifest claims the wrong digest")
            if _seal_digest(payload) != cell["run_manifest_sha256"]:
                raise ValueError(f"{cell_id} staged run manifest seal is invalid")
            if (
                payload.get("run_id") != run_id
                or payload.get("scientific_spec_sha256") != cell["scientific_spec_sha256"]
            ):
                raise ValueError(f"{cell_id} staged run manifest identity drifted")

    budget = sum(cell["estimated_train_seconds"] for cell in cells)
    reserve = value.get("runtime_reserve_seconds")
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 600:
        raise ValueError("campaign runtime_reserve_seconds must be at least 600")
    if budget + reserve > maximum:
        raise ValueError(f"campaign estimated runtime {budget}+{reserve}s exceeds cap {maximum}s")
    if code_dir is not None and not (code_dir / TRAIN_ENTRY).is_file():
        raise ValueError(f"campaign source lacks {TRAIN_ENTRY}")
    return value


class ObjectStore(Protocol):
    def read_bytes(self, uri: str) -> bytes | None: ...

    def put_json_once(self, value: dict, uri: str) -> None: ...


class ChildProcessGroupNotDrained(RuntimeError):
    """The cell leader exited but another process may still be using its work tree."""


def _wait_for_process_group_drain(
    process_group: int,
    *,
    timeout_seconds: float = CHILD_GROUP_DRAIN_TIMEOUT_SECONDS,
    poll_seconds: float = CHILD_GROUP_DRAIN_POLL_SECONDS,
) -> bool:
    """Return only after no process remains in the cell's process group.

    ``gpu.run_resumable`` stays in the entry shell's process group while its trainer and uploader
    use private groups.  It exits only after those children have drained and the final checkpoint
    sync has completed.  Waiting for this outer group therefore prevents campaign cleanup from
    deleting a checkpoint tree underneath the resumable supervisor after a platform SIGTERM.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Same-user campaign children should always be observable.  Treat an unexpected
            # permission boundary as still live and preserve the work tree on timeout.
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)


class AwsCliStore:
    """Small retry-independent control path; checkpoint transport owns bulk/retry behavior."""

    def __init__(self, *, region: str = "us-west-2") -> None:
        self.region = region

    def read_bytes(self, uri: str) -> bytes | None:
        result = subprocess.run(
            ["aws", "s3", "cp", uri, "-", "--only-show-errors", "--region", self.region],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return result.stdout
        detail = result.stderr.decode(errors="replace").casefold()
        if any(marker in detail for marker in ("404", "not found", "nosuchkey", "does not exist")):
            return None
        raise RuntimeError(f"S3 read failed for {uri}: {detail[:500]}")

    def put_json_once(self, value: dict, uri: str) -> None:
        payload = _canonical_bytes(value)
        parsed = urlparse(_safe_s3(uri))
        fd, name = tempfile.mkstemp(prefix="robomme-campaign-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
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
                    name,
                    "--if-none-match",
                    "*",
                    "--region",
                    self.region,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                return
            existing = self.read_bytes(uri)
            if existing is None or existing != payload:
                detail = result.stderr.decode(errors="replace")[:500]
                raise RuntimeError(f"immutable campaign claim collision at {uri}: {detail}")
        finally:
            Path(name).unlink(missing_ok=True)


def _json_bytes(payload: bytes | None, *, uri: str) -> dict | None:
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON at {uri}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return value


def verify_completion(store: ObjectStore, cell: dict) -> dict | None:
    """Return an exact durable completion, or None when no completion claim exists."""
    claim_uri = cell["completion_claim_s3"]
    claim = _json_bytes(store.read_bytes(claim_uri), uri=claim_uri)
    if claim is None:
        return None
    expected = {
        "kind": COMPLETION_KIND,
        "run_id": cell["run_id"],
        "step": cell["final_step"],
        "checkpoint_uri": f"{cell['output_s3']}/deploy/{cell['final_step']}",
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
    }
    drift = {name: (claim.get(name), value) for name, value in expected.items() if claim.get(name) != value}
    if drift:
        raise ValueError(f"completion claim for {cell['cell_id']} is not exact: {drift}")
    tree_sha = _require_hex(claim.get("tree_manifest_sha256"), "checkpoint tree digest")
    tree_uri = _safe_s3(claim.get("tree_manifest_uri"))
    if not tree_uri.endswith(f"/{tree_sha}.json"):
        raise ValueError(f"completion tree URI for {cell['cell_id']} is not content addressed")
    tree_bytes = store.read_bytes(tree_uri)
    if tree_bytes is None or hashlib.sha256(tree_bytes).hexdigest() != tree_sha:
        raise ValueError(f"checkpoint tree manifest for {cell['cell_id']} is absent or corrupt")
    tree = _json_bytes(tree_bytes, uri=tree_uri)
    if tree.get("checkpoint_uri") != expected["checkpoint_uri"] or not tree.get("objects"):
        raise ValueError(f"checkpoint tree manifest for {cell['cell_id']} has the wrong identity")
    deploy_uri = f"{expected['checkpoint_uri']}/_DEPLOY_COMPLETE.json"
    deploy = _json_bytes(store.read_bytes(deploy_uri), uri=deploy_uri)
    if deploy is None:
        raise ValueError(f"deploy completion receipt for {cell['cell_id']} is absent")
    deploy_expected = {
        "kind": DEPLOY_KIND,
        "run_id": cell["run_id"],
        "step": cell["final_step"],
        "checkpoint_uri": expected["checkpoint_uri"],
        "tree_manifest_sha256": tree_sha,
        "scientific_spec_sha256": cell["scientific_spec_sha256"],
    }
    deploy_drift = {
        name: (deploy.get(name), value) for name, value in deploy_expected.items() if deploy.get(name) != value
    }
    if deploy_drift:
        raise ValueError(f"deploy receipt for {cell['cell_id']} is not exact: {deploy_drift}")
    return claim


def _safe_rmtree(path: Path, root: Path) -> None:
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root) or not path.name.startswith("cell-"):
        raise ValueError(f"refusing unsafe campaign cleanup: {path}")
    if path.exists():
        shutil.rmtree(path)


@dataclass
class CampaignRunner:
    manifest: dict
    code_dir: Path
    work_root: Path
    store: ObjectStore

    def __post_init__(self) -> None:
        self.code_dir = self.code_dir.resolve()
        self.work_root = self.work_root.resolve()
        validate_manifest(self.manifest, code_dir=self.code_dir)
        if self.work_root == Path("/") or not self.work_root.is_absolute():
            raise ValueError("campaign work root must be a non-root absolute path")
        self.work_root.mkdir(parents=True, exist_ok=True)
        (self.work_root / "cells").mkdir(exist_ok=True)
        (self.work_root / "cache").mkdir(exist_ok=True)
        (self.work_root / "artifacts").mkdir(exist_ok=True)
        self._active: subprocess.Popen | None = None
        self._received_signal: int | None = None

    def _run_cell(self, cell: dict, cell_work: Path) -> int:
        environment = os.environ.copy()
        environment.update(cell["environment"])
        environment["ROBOMME_WORK_ROOT"] = str(cell_work)
        environment["ROBOMME_CACHE_ROOT"] = str(self.work_root / "cache")
        environment["ROBOMME_SHARED_ARTIFACT_ROOT"] = str(self.work_root / "artifacts")
        process = subprocess.Popen(
            ["bash", str(self.code_dir / TRAIN_ENTRY)],
            cwd=self.code_dir,
            env=environment,
            start_new_session=True,
        )
        self._active = process
        # A signal can arrive in the few instructions between Popen returning and publishing the
        # active handle to `_forward`.  Replay it here so that window cannot leave a cell running
        # after the campaign has entered termination handling.
        if self._received_signal is not None and process.poll() is None:
            try:
                os.killpg(process.pid, self._received_signal)
            except ProcessLookupError:
                pass
        try:
            returncode = int(process.wait())
            if not _wait_for_process_group_drain(process.pid):
                raise ChildProcessGroupNotDrained(
                    f"cell process group {process.pid} did not drain within {CHILD_GROUP_DRAIN_TIMEOUT_SECONDS:.0f}s"
                )
            return returncode
        finally:
            self._active = None

    def _forward(self, signum: int, _frame) -> None:
        self._received_signal = signum
        process = self._active
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def _maybe_clear_cache(self) -> None:
        minimum = self.manifest["cleanup"]["minimum_free_bytes"]
        if shutil.disk_usage(self.work_root).free >= minimum:
            return
        cache = self.work_root / "cache"
        print(f"[campaign] low disk; clearing shared dependency cache {cache}", flush=True)
        shutil.rmtree(cache)
        cache.mkdir()
        if shutil.disk_usage(self.work_root).free < minimum:
            # No cell is live when this method runs.  The content-addressed artifact cache is
            # therefore safe to evict and can be reconstructed from its immutable S3 receipts.
            # This trades a restage for bounded paid disk instead of letting a long campaign
            # fail after accumulating multiple OpenPI environments or task artifacts.
            artifacts = self.work_root / "artifacts"
            print(f"[campaign] disk still low; clearing shared artifact cache {artifacts}", flush=True)
            shutil.rmtree(artifacts)
            artifacts.mkdir()
        if shutil.disk_usage(self.work_root).free < minimum:
            raise RuntimeError(f"campaign node has less than required {minimum} free bytes")

    def run(self) -> int:
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.manifest["infrastructure"]["max_run_seconds"]
        self.store.put_json_once(self.manifest, self.manifest["claims"]["manifest"])
        previous_int = signal.signal(signal.SIGINT, self._forward)
        previous_term = signal.signal(signal.SIGTERM, self._forward)
        records: list[dict] = []
        overall = 0
        deferred_runtime_budget = False
        try:
            for cell in self.manifest["cells"]:
                if self._received_signal is not None:
                    overall = 128 + self._received_signal
                    break
                exact = verify_completion(self.store, cell)
                if exact is not None:
                    print(f"[campaign] SKIP exact-complete {cell['cell_id']}", flush=True)
                    records.append({"cell_id": cell["cell_id"], "status": "skipped_exact_complete"})
                    continue
                remaining = deadline - time.monotonic()
                required = cell["estimated_train_seconds"] + self.manifest["runtime_reserve_seconds"]
                if remaining < required:
                    deferred_runtime_budget = True
                    records.append(
                        {
                            "cell_id": cell["cell_id"],
                            "status": "deferred_runtime_budget",
                            "remaining_seconds": max(0, int(remaining)),
                            "required_seconds": required,
                        }
                    )
                    print(
                        f"[campaign] DEFER {cell['cell_id']} remaining={remaining:.0f}s required={required}s",
                        flush=True,
                    )
                    break
                self._maybe_clear_cache()
                cell_work = self.work_root / "cells" / f"cell-{cell['ordinal']:03d}-{cell['cell_id']}"
                _safe_rmtree(cell_work, self.work_root / "cells")
                cell_work.mkdir(parents=True)
                print(
                    f"[campaign] START {cell['ordinal'] + 1}/{len(self.manifest['cells'])} {cell['cell_id']}",
                    flush=True,
                )
                returncode = 1
                detail = None
                preserve_cell_work = False
                try:
                    returncode = self._run_cell(cell, cell_work)
                    if returncode == 0:
                        if verify_completion(self.store, cell) is None:
                            raise RuntimeError(
                                f"training entry returned success without a durable completion "
                                f"claim for {cell['cell_id']}"
                            )
                    else:
                        detail = f"training entry returned {returncode}"
                except ChildProcessGroupNotDrained as error:
                    preserve_cell_work = True
                    returncode = 1 if self._received_signal is None else 128 + self._received_signal
                    detail = f"{type(error).__name__}: {error}"
                except BaseException as error:
                    returncode = 1 if self._received_signal is None else 128 + self._received_signal
                    detail = f"{type(error).__name__}: {error}"
                finally:
                    if preserve_cell_work:
                        print(
                            f"[campaign] PRESERVE {cell['cell_id']} work tree because its child "
                            "process group did not drain",
                            flush=True,
                        )
                    else:
                        _safe_rmtree(cell_work, self.work_root / "cells")
                if returncode == 0:
                    records.append({"cell_id": cell["cell_id"], "status": "completed"})
                    print(f"[campaign] COMPLETE {cell['cell_id']}", flush=True)
                    continue
                overall = returncode or 1
                records.append(
                    {
                        "cell_id": cell["cell_id"],
                        "status": "failed",
                        "returncode": overall,
                        "detail": detail,
                    }
                )
                print(f"[campaign] FAILED {cell['cell_id']}: {detail}", flush=True)
                if self.manifest["failure_policy"] == "stop" or self._received_signal is not None:
                    break
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)

        completed_ids = {
            record["cell_id"] for record in records if record["status"] in {"completed", "skipped_exact_complete"}
        }
        expected_ids = {cell["cell_id"] for cell in self.manifest["cells"]}
        result = {
            "schema_version": 1,
            "kind": "robomme_single_task_train_series_attempt_result",
            "campaign_id": self.manifest["campaign_id"],
            "attempt_id": self.manifest["attempt_id"],
            "campaign_manifest_sha256": self.manifest["manifest_sha256"],
            "records": records,
            "all_cells_complete": completed_ids == expected_ids,
            "deferred_runtime_budget": deferred_runtime_budget,
            "evaluation": self.manifest["evaluation"],
        }
        self.store.put_json_once(result, self.manifest["claims"]["attempt_result"])
        if result["all_cells_complete"]:
            completion = {
                "schema_version": 1,
                "kind": "robomme_single_task_train_series_complete",
                "campaign_id": self.manifest["campaign_id"],
                "campaign_manifest_sha256": self.manifest["manifest_sha256"],
                "cells": [cell["run_id"] for cell in self.manifest["cells"]],
                "evaluation_status": "deferred_by_certification_gate",
            }
            self.store.put_json_once(completion, self.manifest["claims"]["completion"])
            return 0
        if deferred_runtime_budget and overall == 0:
            # A bounded campaign ending before a likely-partial cell is an expected, resumable
            # infrastructure outcome.  The stable campaign completion remains absent.
            return 0
        return overall or 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, default=Path("/opt/ml/code"))
    parser.add_argument("--work-root", type=Path, default=Path("/opt/ml/robomme-campaign"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    return CampaignRunner(
        manifest=manifest,
        code_dir=args.code_dir,
        work_root=args.work_root,
        store=AwsCliStore(),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
