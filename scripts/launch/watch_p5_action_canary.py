#!/usr/bin/env python3
"""Quietly gate one immutable p5 action canary and hand off at most once.

This watcher is intentionally outside the submitted ``robomme_integration`` source tree.  It
performs only read-only AWS calls while waiting.  A submission handoff is possible only when an
independently produced, byte-pinned HARD_GREEN audit receipt matches the immutable ready packet,
the frozen source still has the exact reviewed delta from its baseline, and the normal p5 launcher
reports an actually free node with no waiting work.  The normal launcher repeats the live gate
immediately before its one submission attempt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

# Importing the frozen launcher must not create ``__pycache__`` entries inside the source whose
# exact identity we revalidate on every poll.  Set the interpreter flag before any frozen module
# is loaded; the submitted source itself remains byte-for-byte unchanged.
sys.dont_write_bytecode = True

PACKET_KIND = "robomme_p5_action_canary_ready_packet"
AUDIT_KIND = "robomme_p5_action_canary_independent_audit"
PACKET_STATUS = "READY_FOR_INDEPENDENT_AUDIT"
AUDIT_STATUS = "HARD_GREEN"
CANARY_SOURCE_RELATIVE = "eval/p5_q1_parallel_action_canary_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_TASK = re.compile(r"^/[a-z0-9_/-]+$")
WAITING_STATUSES = frozenset({"SUBMITTED", "PENDING", "RUNNABLE", "SCHEDULED", "STARTING"})
ACTIVE_STATUSES = WAITING_STATUSES | {"RUNNING"}
REQUIRED_AUDIT_CHECKS = frozenset(
    {
        "canary_contract_reviewed",
        "cleanup_and_score_redaction_reviewed",
        "focused_tests_green",
        "live_gate_reviewed",
        "python_compile_green",
        "ruff_green",
        "runtime_closure_isolated",
        "shell_syntax_green",
        "source_delta_exact",
    }
)
TRANSIENT_HOLD_MESSAGES = (
    "p5 queue already has committed waiting work; refusing backlog submission",
    "p5 has no genuinely available node ",
)


class WatcherContractError(RuntimeError):
    """The immutable packet, audit, source, or gate contract is not safe to use."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def seal_document(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    """Return a canonical self-sealed document (also used by the independent auditor)."""
    if field in value:
        raise ValueError(f"refusing to overwrite seal field {field}")
    sealed = dict(value)
    sealed[field] = hashlib.sha256(_canonical(value)).hexdigest()
    return sealed


def _validate_seal(value: dict[str, Any], *, field: str, label: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed):
        raise WatcherContractError(f"{label} has no valid {field}")
    unsealed = dict(value)
    unsealed.pop(field)
    actual = hashlib.sha256(_canonical(unsealed)).hexdigest()
    if actual != claimed:
        raise WatcherContractError(f"{label} self-seal mismatch")
    return claimed


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WatcherContractError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise WatcherContractError(f"{label} must be one JSON object")
    return value, payload


def _safe_delta_path(value: object) -> str:
    if not isinstance(value, str):
        raise WatcherContractError("ready-packet delta path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or path.as_posix() != value:
        raise WatcherContractError(f"unsafe ready-packet delta path: {value!r}")
    return value


def validate_ready_packet(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "status",
        "builder_task",
        "baseline_source_tree_sha256",
        "source_tree_sha256",
        "source_delta",
        "launch_guardrails_sha256",
        "preflight_id",
        "job_name",
        "manifest_sha256",
        "claim_s3",
        "evidence_root_s3",
        "canary_template_sha256",
        "canary_config_sha256",
        "parallel_topology_sha256",
        "actions_expected",
        "score_publication",
        "scored_queue_template",
        "packet_sha256",
    }
    if set(value) != required:
        raise WatcherContractError("ready packet has extra or missing fields")
    if value.get("schema_version") != 1 or value.get("kind") != PACKET_KIND or value.get("status") != PACKET_STATUS:
        raise WatcherContractError("ready packet has the wrong schema, kind, or status")
    _validate_seal(value, field="packet_sha256", label="ready packet")
    for name in (
        "baseline_source_tree_sha256",
        "source_tree_sha256",
        "launch_guardrails_sha256",
        "manifest_sha256",
        "canary_template_sha256",
        "canary_config_sha256",
        "parallel_topology_sha256",
    ):
        if not isinstance(value.get(name), str) or not HEX64.fullmatch(value[name]):
            raise WatcherContractError(f"ready packet has invalid {name}")
    builder = value.get("builder_task")
    if not isinstance(builder, str) or not SAFE_TASK.fullmatch(builder):
        raise WatcherContractError("ready packet builder task is invalid")
    preflight_id = value.get("preflight_id")
    if not isinstance(preflight_id, str) or not re.fullmatch(r"p5-native-eval-v1-[0-9a-f]{20}", preflight_id):
        raise WatcherContractError("ready packet has an invalid preflight ID")
    expected_job = f"sarvesh-rmme-p5-action-{preflight_id.rsplit('-', 1)[-1]}"
    if value.get("job_name") != expected_job:
        raise WatcherContractError("ready packet job name does not bind its preflight ID")
    if not str(value.get("claim_s3", "")).endswith(f"/preflight/{preflight_id}.json"):
        raise WatcherContractError("ready packet claim URI does not bind its preflight ID")
    if not str(value.get("evidence_root_s3", "")).endswith(f"/eval_preflight/{preflight_id}/evidence"):
        raise WatcherContractError("ready packet evidence URI does not bind its preflight ID")
    if value.get("actions_expected") != 32 or value.get("score_publication") is not False:
        raise WatcherContractError("ready packet is not the unscored 32-action canary")
    delta = value.get("source_delta")
    if not isinstance(delta, list) or not delta:
        raise WatcherContractError("ready packet source delta is absent")
    normalized: list[dict[str, str]] = []
    for record in delta:
        if not isinstance(record, dict) or set(record) != {"path", "mode", "sha256"}:
            raise WatcherContractError("ready packet source-delta record is malformed")
        path = _safe_delta_path(record["path"])
        mode = record["mode"]
        digest = record["sha256"]
        if not isinstance(mode, str) or not re.fullmatch(r"0[0-7]{3}", mode):
            raise WatcherContractError(f"ready packet has invalid mode for {path}")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise WatcherContractError(f"ready packet has invalid digest for {path}")
        normalized.append({"path": path, "mode": mode, "sha256": digest})
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise WatcherContractError("ready packet source delta is not sorted")
    if len({record["path"] for record in normalized}) != len(normalized):
        raise WatcherContractError("ready packet source delta contains duplicate paths")
    queue = value.get("scored_queue_template")
    if (
        not isinstance(queue, dict)
        or set(queue) != {"path", "sha256", "queue_id", "cells"}
        or not isinstance(queue.get("path"), str)
        or not isinstance(queue.get("sha256"), str)
        or not HEX64.fullmatch(queue["sha256"])
        or not isinstance(queue.get("queue_id"), str)
        or not isinstance(queue.get("cells"), list)
        or len(queue["cells"]) != 8
    ):
        raise WatcherContractError("ready packet scored-queue template is malformed")
    return value


def validate_audit_receipt(
    value: dict[str, Any],
    *,
    packet: dict[str, Any],
    payload: bytes,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_payload_sha256):
        raise WatcherContractError("expected audit-receipt digest is invalid")
    if hashlib.sha256(payload).hexdigest() != expected_payload_sha256:
        raise WatcherContractError("independent audit receipt bytes differ from the approved digest")
    required = {
        "schema_version",
        "kind",
        "status",
        "auditor_task",
        "packet_sha256",
        "source_tree_sha256",
        "preflight_id",
        "manifest_sha256",
        "checks",
        "audit_receipt_sha256",
    }
    if set(value) != required:
        raise WatcherContractError("independent audit receipt has extra or missing fields")
    if value.get("schema_version") != 1 or value.get("kind") != AUDIT_KIND or value.get("status") != AUDIT_STATUS:
        raise WatcherContractError("independent audit receipt is not HARD_GREEN")
    _validate_seal(value, field="audit_receipt_sha256", label="independent audit receipt")
    auditor = value.get("auditor_task")
    if not isinstance(auditor, str) or not SAFE_TASK.fullmatch(auditor) or auditor == packet["builder_task"]:
        raise WatcherContractError("audit receipt does not identify an independent auditor")
    bindings = {
        "packet_sha256": packet["packet_sha256"],
        "source_tree_sha256": packet["source_tree_sha256"],
        "preflight_id": packet["preflight_id"],
        "manifest_sha256": packet["manifest_sha256"],
    }
    if any(value.get(name) != expected for name, expected in bindings.items()):
        raise WatcherContractError("independent audit receipt does not bind the ready packet")
    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_AUDIT_CHECKS:
        raise WatcherContractError("independent audit receipt has incomplete check coverage")
    if not all(checks.values()) or any(item is not True for item in checks.values()):
        raise WatcherContractError("independent audit receipt contains a non-green check")
    return value


def source_tree_sha256(root: Path) -> str:
    """Hash exact submitted-source bytes, modes, types, and symlink targets."""
    root = root.resolve()
    digest = hashlib.sha256()

    def field(value: str | bytes) -> None:
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        field(path.relative_to(root).as_posix())
        field(oct(stat.S_IMODE(path.lstat().st_mode)))
        if path.is_symlink():
            field("symlink")
            field(os.readlink(path))
        elif path.is_dir():
            field("directory")
        elif path.is_file():
            field("file")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    field(block)
        else:
            raise WatcherContractError(f"unsupported source entry: {path}")
    return digest.hexdigest()


def _source_delta(source: Path, baseline: Path) -> list[dict[str, str]]:
    relative_paths = {path.relative_to(source) for path in source.rglob("*")} | {
        path.relative_to(baseline) for path in baseline.rglob("*")
    }
    records: list[dict[str, str]] = []
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        current = source / relative
        original = baseline / relative
        current_exists = current.exists() or current.is_symlink()
        original_exists = original.exists() or original.is_symlink()
        if not current_exists:
            raise WatcherContractError(f"frozen source removed baseline path {relative}")
        if not original_exists:
            if not current.is_file() or current.is_symlink():
                raise WatcherContractError(f"frozen source added unsupported entry {relative}")
            records.append(
                {
                    "path": relative.as_posix(),
                    "mode": f"0{stat.S_IMODE(current.lstat().st_mode):03o}",
                    "sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
                }
            )
            continue
        current_mode = stat.S_IMODE(current.lstat().st_mode)
        original_mode = stat.S_IMODE(original.lstat().st_mode)
        if stat.S_IFMT(current.lstat().st_mode) != stat.S_IFMT(original.lstat().st_mode):
            raise WatcherContractError(f"frozen source changed entry type for {relative}")
        changed = current_mode != original_mode
        if current.is_symlink():
            if os.readlink(current) != os.readlink(original):
                raise WatcherContractError(f"frozen source changed symlink {relative}")
        elif current.is_file():
            current_digest = hashlib.sha256(current.read_bytes()).hexdigest()
            original_digest = hashlib.sha256(original.read_bytes()).hexdigest()
            changed = changed or current_digest != original_digest
            if changed:
                records.append(
                    {
                        "path": relative.as_posix(),
                        "mode": f"0{current_mode:03o}",
                        "sha256": current_digest,
                    }
                )
        elif changed:
            raise WatcherContractError(f"frozen source changed directory mode for {relative}")
    return records


def build_ready_packet(
    *,
    source: Path,
    baseline: Path,
    launch_guardrails: Path,
    builder_task: str,
) -> dict[str, Any]:
    """Build the exact packet which an independent agent must review and seal green."""
    if not SAFE_TASK.fullmatch(builder_task):
        raise WatcherContractError("ready-packet builder task is invalid")
    if not launch_guardrails.is_file():
        raise WatcherContractError("launch guardrails file is absent")
    launch = load_frozen_launcher(source)
    # Build once from the frozen tree, then require the same identity through normal validation.
    args = launch.parser().parse_args(
        [
            "--source-dir",
            str(source),
            "--parallel-action-canary",
            "--action-template",
            str(source / CANARY_SOURCE_RELATIVE),
            "--dry-run",
        ]
    )
    plan = launch.build_plan(args, source)
    queue_path = source / "eval/p5_standard_wave1_fixed50_parallel_v1.json"
    queue_payload = queue_path.read_bytes()
    queue = json.loads(queue_payload)
    if not isinstance(queue, dict) or not isinstance(queue.get("cells"), list):
        raise WatcherContractError("scored queue template is malformed")
    packet = {
        "schema_version": 1,
        "kind": PACKET_KIND,
        "status": PACKET_STATUS,
        "builder_task": builder_task,
        "baseline_source_tree_sha256": source_tree_sha256(baseline),
        "source_tree_sha256": source_tree_sha256(source),
        "source_delta": _source_delta(source, baseline),
        "launch_guardrails_sha256": hashlib.sha256(launch_guardrails.read_bytes()).hexdigest(),
        "preflight_id": plan["preflight_id"],
        "job_name": f"sarvesh-rmme-p5-action-{plan['preflight_id'].rsplit('-', 1)[-1]}",
        "manifest_sha256": plan["manifest"]["manifest_sha256"],
        "claim_s3": plan["claim_s3"],
        "evidence_root_s3": plan["manifest"]["evidence_root_s3"],
        "canary_template_sha256": plan["manifest"]["canary_template_sha256"],
        "canary_config_sha256": plan["manifest"]["probe"]["benchmark_config_sha256"],
        "parallel_topology_sha256": plan["manifest"]["topology"]["parallel_topology_sha256"],
        "actions_expected": plan["manifest"]["probe"]["actions_expected"],
        "score_publication": plan["manifest"]["probe"]["score_publication"],
        "scored_queue_template": {
            "path": "eval/p5_standard_wave1_fixed50_parallel_v1.json",
            "sha256": hashlib.sha256(queue_payload).hexdigest(),
            "queue_id": queue["queue_id"],
            "cells": [
                {
                    "ordinal": cell["ordinal"],
                    "task": cell["task"],
                    "arm": cell["arm"],
                    "run_id": cell["run_id"],
                }
                for cell in queue["cells"]
            ],
        },
    }
    return seal_document(packet, field="packet_sha256")


def validate_frozen_source(packet: dict[str, Any], *, source: Path, baseline: Path) -> None:
    if source_tree_sha256(baseline) != packet["baseline_source_tree_sha256"]:
        raise WatcherContractError("pristine baseline source identity drift")
    if source_tree_sha256(source) != packet["source_tree_sha256"]:
        raise WatcherContractError("frozen p5 source identity drift")
    if _source_delta(source, baseline) != packet["source_delta"]:
        raise WatcherContractError("frozen p5 source delta differs from the reviewed packet")
    guardrail = source.parent / "scripts/launch/launch_guardrails.py"
    if (
        not guardrail.is_file()
        or hashlib.sha256(guardrail.read_bytes()).hexdigest() != packet["launch_guardrails_sha256"]
    ):
        raise WatcherContractError("frozen launch guardrails identity drift")


def load_frozen_launcher(source: Path):
    """Import the launcher only from the isolated source tree in this fresh watcher process."""
    source = source.resolve()
    if source.name != "robomme_integration":
        raise WatcherContractError("source directory must be the isolated robomme_integration root")
    if "launch_guardrails" in sys.modules or any(
        name == "robomme_integration" or name.startswith("robomme_integration.") for name in sys.modules
    ):
        raise WatcherContractError("watcher process imported mutable launch modules before freeze validation")
    sys.path.insert(0, str(source.parent / "scripts/launch"))
    sys.path.insert(0, str(source.parent))
    module = importlib.import_module("robomme_integration.eval.launch_p5_preflight")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(source):
        raise WatcherContractError("p5 launcher imported outside the frozen source tree")
    guardrail_module = sys.modules.get("launch_guardrails")
    guardrail_path = Path(getattr(guardrail_module, "__file__", "")).resolve()
    if guardrail_path != (source.parent / "scripts/launch/launch_guardrails.py").resolve():
        raise WatcherContractError("launcher guardrails imported outside the frozen packet")
    return module


def build_and_validate_plan(launch: Any, packet: dict[str, Any], *, source: Path) -> dict:
    template = source / CANARY_SOURCE_RELATIVE
    args = launch.parser().parse_args(
        [
            "--source-dir",
            str(source),
            "--parallel-action-canary",
            "--action-template",
            str(template),
            "--dry-run",
        ]
    )
    plan = launch.build_plan(args, source)
    expected = {
        "source_sha": packet["source_tree_sha256"],
        "preflight_id": packet["preflight_id"],
        "claim_s3": packet["claim_s3"],
    }
    if any(plan.get(name) != value for name, value in expected.items()):
        raise WatcherContractError("rebuilt p5 canary plan differs from the ready packet")
    manifest = plan.get("manifest")
    manifest_expected = {
        "manifest_sha256": packet["manifest_sha256"],
        "evidence_root_s3": packet["evidence_root_s3"],
        "canary_template_sha256": packet["canary_template_sha256"],
    }
    if not isinstance(manifest, dict) or any(manifest.get(name) != value for name, value in manifest_expected.items()):
        raise WatcherContractError("rebuilt p5 canary manifest differs from the ready packet")
    if (
        manifest.get("topology", {}).get("parallel_topology_sha256") != packet["parallel_topology_sha256"]
        or manifest.get("probe", {}).get("benchmark_config_sha256") != packet["canary_config_sha256"]
        or manifest.get("probe", {}).get("actions_expected") != 32
        or manifest.get("probe", {}).get("score_publication") is not False
    ):
        raise WatcherContractError("rebuilt p5 action/topology contract differs from the packet")
    return plan


def validate_go_snapshot(snapshot: dict[str, Any], *, job_name: str, account: str, p5_limit: int) -> dict[str, int]:
    if snapshot.get("account") != account:
        raise WatcherContractError("live p5 gate used the wrong AWS account")
    if snapshot.get("claim_objects") or snapshot.get("evidence_objects"):
        raise WatcherContractError("immutable canary claim/evidence namespace is not empty")
    batch = snapshot.get("batch_jobs")
    training = snapshot.get("training_jobs")
    if not isinstance(batch, list) or not isinstance(training, list):
        raise WatcherContractError("live p5 gate returned an incomplete snapshot")
    if snapshot.get("training_job_name_exists"):
        raise WatcherContractError("exact historical p5 canary job identity already exists")
    for job in batch:
        status = job.get("status")
        if status not in ACTIVE_STATUSES:
            raise WatcherContractError(f"live p5 gate returned unknown active status {status!r}")
        if job.get("jobName") == job_name:
            raise WatcherContractError("exact p5 canary Batch identity already exists")
        if status in WAITING_STATUSES:
            raise WatcherContractError(f"p5 queue still contains waiting status {status}")
    p5_instances = 0
    for job in training:
        if job.get("TrainingJobName") == job_name:
            raise WatcherContractError("exact p5 canary SageMaker identity already exists")
        if job.get("InstanceType") != "ml.p5.48xlarge":
            continue
        count = job.get("InstanceCount")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise WatcherContractError("live p5 training job has an invalid instance count")
        p5_instances += count
    available = p5_limit - p5_instances
    if p5_instances >= p5_limit or available < 1:
        raise WatcherContractError(f"p5 has no genuinely available node ({p5_instances}/{p5_limit})")
    return {"p5_instances": p5_instances, "p5_available": available}


def classify_gate_exit(error: SystemExit) -> str:
    message = str(error)
    if message == TRANSIENT_HOLD_MESSAGES[0] or message.startswith(TRANSIENT_HOLD_MESSAGES[1]):
        return message
    raise WatcherContractError(f"fatal live p5 gate failure: {message}") from error


def _event(kind: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "event": kind,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _write_handoff_state_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise WatcherContractError(
            f"handoff state already exists; refusing a second submission attempt: {path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _update_handoff_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _submission_command(args: argparse.Namespace, source: Path) -> list[str]:
    return [
        str(args.submit_python),
        str(source / "eval/launch_p5_preflight.py"),
        "--source-dir",
        str(source),
        "--parallel-action-canary",
        "--action-template",
        str(source / CANARY_SOURCE_RELATIVE),
        "--confirm-submit",
    ]


def run(
    args: argparse.Namespace,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    submit: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    packet_value, _packet_payload = _load_json(args.ready_packet, label="ready packet")
    packet = validate_ready_packet(packet_value)
    audit_value, audit_payload = _load_json(args.audit_receipt, label="independent audit receipt")
    validate_audit_receipt(
        audit_value,
        packet=packet,
        payload=audit_payload,
        expected_payload_sha256=args.expected_audit_receipt_sha256,
    )
    source = args.source_dir.resolve()
    baseline = args.baseline_source_dir.resolve()
    validate_frozen_source(packet, source=source, baseline=baseline)
    launch = load_frozen_launcher(source)
    plan = build_and_validate_plan(launch, packet, source=source)
    if args.confirm_auto_submit:
        if args.state_file is None:
            raise WatcherContractError("auto-submit mode requires an exclusive --state-file")
        submit_python = args.submit_python
        if not submit_python.is_absolute() or not submit_python.is_file() or not os.access(submit_python, os.X_OK):
            raise WatcherContractError("--submit-python must be an absolute executable file")
        if args.state_file.exists():
            raise WatcherContractError("handoff state already exists; watcher cannot be re-armed")
    deadline = monotonic() + args.max_wait_seconds
    last_hold: str | None = None
    while True:
        # Recompute source and scientific identities before every live observation.
        validate_frozen_source(packet, source=source, baseline=baseline)
        plan = build_and_validate_plan(launch, packet, source=source)
        try:
            snapshot = launch.collect_action_submission_snapshot(plan, job_name=packet["job_name"])
        except SystemExit as error:
            hold = classify_gate_exit(error)
            if hold != last_hold:
                _event("HOLD", reason=hold)
                last_hold = hold
            if args.check_once:
                return 75
            remaining = deadline - monotonic()
            if remaining <= 0:
                _event("TIMEOUT", reason=hold)
                return 75
            sleep(min(args.poll_seconds, remaining))
            continue
        admission = validate_go_snapshot(
            snapshot,
            job_name=packet["job_name"],
            account=launch.EXECUTION_ACCOUNT,
            p5_limit=launch.P5_CONCURRENCY_LIMIT,
        )
        # Close source drift between the successful read and the handoff.  The child launcher then
        # repeats source hashing, namespace/duplicate checks, and the full live capacity gate.
        validate_frozen_source(packet, source=source, baseline=baseline)
        build_and_validate_plan(launch, packet, source=source)
        _event("GO", **admission)
        if args.check_once:
            _event("READ_ONLY_COMPLETE", reason="--check-once forbids submission")
            return 0
        command = _submission_command(args, source)
        state = {
            "schema_version": 1,
            "kind": "robomme_p5_action_canary_handoff_state",
            "status": "HANDOFF_IN_PROGRESS",
            "packet_sha256": packet["packet_sha256"],
            "audit_receipt_file_sha256": args.expected_audit_receipt_sha256,
            "preflight_id": packet["preflight_id"],
            "job_name": packet["job_name"],
            "command": command,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _write_handoff_state_once(args.state_file, state)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = submit(
            command,
            cwd=source.parent,
            env=environment,
            check=False,
        )
        state.update(
            status=("HANDOFF_SUCCEEDED" if result.returncode == 0 else "HANDOFF_FAILED_REVIEW_REQUIRED"),
            returncode=int(result.returncode),
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        _update_handoff_state(args.state_file, state)
        _event("HANDOFF_COMPLETE", returncode=int(result.returncode))
        # Exactly one handoff attempt per watcher invocation, including ambiguous child failures.
        return int(result.returncode)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--ready-packet", type=Path, required=True)
    value.add_argument("--audit-receipt", type=Path, required=True)
    value.add_argument("--expected-audit-receipt-sha256", required=True)
    value.add_argument("--source-dir", type=Path, required=True)
    value.add_argument("--baseline-source-dir", type=Path, required=True)
    value.add_argument("--submit-python", type=Path, default=Path(sys.executable))
    value.add_argument("--poll-seconds", type=float, default=300.0)
    value.add_argument("--max-wait-seconds", type=float, default=172800.0)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-once", action="store_true")
    mode.add_argument(
        "--confirm-auto-submit",
        action="store_true",
        help="arm exactly one guarded handoff after explicit approval of the HARD_GREEN receipt",
    )
    value.add_argument("--state-file", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if not 10.0 <= args.poll_seconds <= 3600.0:
        raise SystemExit("--poll-seconds must lie in [10, 3600]")
    if not 1.0 <= args.max_wait_seconds <= 172800.0:
        raise SystemExit("--max-wait-seconds must lie in [1, 172800]")
    try:
        returncode = run(args)
    except (OSError, WatcherContractError) as error:
        raise SystemExit(f"P5 WATCHER FATAL: {error}") from error
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
