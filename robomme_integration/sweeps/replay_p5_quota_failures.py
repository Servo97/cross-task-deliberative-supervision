#!/usr/bin/env python3
"""Safely replay quota-rejected RoboMME p5 Batch service jobs.

Scientific fields and the immutable source archive are preserved byte-for-byte.
Invalid underscores in legacy SageMaker infrastructure names are normalized.
New Batch wrapper identities are admitted one at a time
only while the account-level p5 quota has reserved headroom.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REGION = "us-west-2"
QUEUE = "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
QUEUE_SUFFIX = f":job-queue/{QUEUE}"
INSTANCE_TYPE = "ml.p5.48xlarge"
QUOTA_REASON = "account-level service limit 'ml.p5.48xlarge for training job usage'"
SM_NAME = re.compile(r"^[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}$")
ACTIVE = {"SUBMITTED", "PENDING", "RUNNABLE", "SCHEDULED", "STARTING", "RUNNING"}
ADMISSION_PENDING = ACTIVE - {"RUNNING"}
TERMINAL = {"SUCCEEDED", "FAILED"}
DEFAULT_SPEC = Path(__file__).with_name("p5_priority1_retry_sources_v1.json")
DEFAULT_RUNTIME = Path.home() / "Research" / "TRI" / "robomme_eval" / "state"
LOG = logging.getLogger("p5_quota_retry")


class GuardrailError(RuntimeError):
    """A scientific or infrastructure invariant failed."""


class AwsCliError(RuntimeError):
    """An AWS CLI command failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardrailError(f"{path} must contain a JSON object")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def aws_cli(*arguments: str) -> dict[str, Any]:
    command = [
        "aws",
        *arguments,
        "--region",
        REGION,
        "--output",
        "json",
        "--no-cli-pager",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise AwsCliError(f"{' '.join(command[:3])} failed: {detail}")
    text = result.stdout.strip()
    return json.loads(text) if text else {}


def describe_service_job(job_id: str) -> dict[str, Any]:
    return aws_cli("batch", "describe-service-job", "--job-id", job_id)


def s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise GuardrailError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def source_archive_uri(payload: dict[str, Any]) -> str:
    raw = payload.get("HyperParameters", {}).get("sagemaker_submit_directory")
    if not isinstance(raw, str):
        raise GuardrailError("source payload lacks sagemaker_submit_directory")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = raw.strip('"')
    s3_parts(decoded)
    return decoded


def available_slots(p5_in_progress: int, quota_limit: int, safety_headroom: int) -> int:
    if quota_limit < 1 or safety_headroom < 0 or safety_headroom >= quota_limit:
        raise GuardrailError("invalid quota/headroom configuration")
    return max(0, quota_limit - safety_headroom - p5_in_progress)


def validate_source_job(
    record: dict[str, Any],
    *,
    expected_job_id: str,
) -> dict[str, Any]:
    if record.get("jobId") != expected_job_id:
        raise GuardrailError(f"Batch job identity mismatch for {expected_job_id}")
    queue = record.get("jobQueue", "")
    if queue != QUEUE and not queue.endswith(QUEUE_SUFFIX):
        raise GuardrailError(f"{expected_job_id}: unexpected queue {queue}")
    if record.get("serviceJobType") != "SAGEMAKER_TRAINING":
        raise GuardrailError(f"{expected_job_id}: not a SageMaker training service job")
    if record.get("schedulingPriority") != 1:
        raise GuardrailError(f"{expected_job_id}: expected priority 1")
    if record.get("retryStrategy", {}).get("attempts") != 1:
        raise GuardrailError(f"{expected_job_id}: source retry strategy drifted")

    raw_payload = record.get("serviceRequestPayload")
    if not isinstance(raw_payload, str):
        raise GuardrailError(f"{expected_job_id}: missing raw serviceRequestPayload")
    payload = json.loads(raw_payload)
    resource = payload.get("ResourceConfig", {})
    if (resource.get("InstanceType"), resource.get("InstanceCount")) != (INSTANCE_TYPE, 1):
        raise GuardrailError(f"{expected_job_id}: expected one {INSTANCE_TYPE}")
    if payload.get("StoppingCondition", {}).get("MaxRuntimeInSeconds") != 86400:
        raise GuardrailError(f"{expected_job_id}: max runtime drifted")
    if payload.get("TrainingJobName") != record.get("jobName"):
        raise GuardrailError(f"{expected_job_id}: Batch/SageMaker name mismatch")

    environment = payload.get("Environment", {})
    tags = record.get("tags", {})
    required = {
        "ROBOMME_RUN_ID",
        "ROBOMME_ATTEMPT_ID",
        "RUN_MANIFEST_S3",
        "PRODUCER_CLAIM_S3",
        "COMPLETION_CLAIM_S3",
        "OUTPUT_S3",
        "RUN_MANIFEST_SHA256",
    }
    missing = sorted(required - environment.keys())
    if missing:
        raise GuardrailError(f"{expected_job_id}: missing environment keys {missing}")
    if not environment["ROBOMME_ATTEMPT_ID"].endswith("-attempt1"):
        raise GuardrailError(f"{expected_job_id}: source was not attempt1")
    if tags.get("wsm.run_id") != environment["ROBOMME_RUN_ID"]:
        raise GuardrailError(f"{expected_job_id}: run-id tag mismatch")
    if tags.get("wsm.task") != environment.get("ROBOMME_TASK"):
        raise GuardrailError(f"{expected_job_id}: task tag mismatch")
    if tags.get("wsm.arm") != environment.get("ROBOMME_ARM"):
        raise GuardrailError(f"{expected_job_id}: arm tag mismatch")

    status = record.get("status")
    if status == "SUCCEEDED":
        return {
            "status": status,
            "payload": payload,
            "environment": environment,
            "raw_payload_sha256": canonical_sha(raw_payload),
        }
    if status != "FAILED":
        raise GuardrailError(f"{expected_job_id}: expected terminal status, got {status}")
    if record.get("attempts"):
        raise GuardrailError(f"{expected_job_id}: failed source unexpectedly has attempts")
    if QUOTA_REASON not in record.get("statusReason", ""):
        raise GuardrailError(f"{expected_job_id}: failure was not the approved quota rejection")
    return {
        "status": status,
        "payload": payload,
        "environment": environment,
        "raw_payload_sha256": canonical_sha(raw_payload),
    }


def replay_name(source_name: str, generation: int) -> str:
    name = f"{source_name}-qr{generation}"
    if len(name) > 128:
        name = f"{source_name[:110]}-{canonical_sha(name)[:12]}-qr{generation}"
    return name


def valid_training_name(source_name: str) -> str:
    if SM_NAME.fullmatch(source_name):
        return source_name
    normalized = re.sub(r"[^A-Za-z0-9-]+", "-", source_name)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if len(normalized) > 63:
        normalized = f"{normalized[:50].rstrip('-')}-{canonical_sha(source_name)[:12]}"
    if not SM_NAME.fullmatch(normalized):
        raise GuardrailError(f"cannot normalize SageMaker TrainingJobName {source_name!r}")
    return normalized


def scientific_payload_projection(payload: dict[str, Any]) -> dict[str, Any]:
    projected = json.loads(json.dumps(payload))
    projected.pop("TrainingJobName", None)
    projected.get("HyperParameters", {}).pop("sagemaker_job_name", None)
    return projected


def replay_service_payload(raw_payload: str) -> tuple[str, str]:
    source = json.loads(raw_payload)
    source_name = source["TrainingJobName"]
    target_name = valid_training_name(source_name)
    if target_name == source_name:
        return raw_payload, target_name
    replay = json.loads(raw_payload)
    replay["TrainingJobName"] = target_name
    replay["HyperParameters"]["sagemaker_job_name"] = json.dumps(target_name)
    if scientific_payload_projection(replay) != scientific_payload_projection(source):
        raise GuardrailError("SageMaker name repair changed scientific payload fields")
    return json.dumps(replay, sort_keys=True, separators=(",", ":")), target_name


def build_replay_request(
    record: dict[str, Any],
    generation: int,
    scheduling_priority: int | None = None,
) -> dict[str, Any]:
    if generation < 1:
        raise GuardrailError("replay generation must be positive")
    priority = record["schedulingPriority"] if scheduling_priority is None else scheduling_priority
    if priority not in {1, 400}:
        raise GuardrailError(f"unsupported replay scheduling priority {priority}")
    source_id = record["jobId"]
    name = replay_name(record["jobName"], generation)
    replay_payload, training_name = replay_service_payload(record["serviceRequestPayload"])
    tags = dict(record.get("tags", {}))
    tags.update(
        {
            "wsm.replay_of": source_id,
            "wsm.replay_generation": str(generation),
            "wsm.replay_priority": str(priority),
            "wsm.sagemaker_name_repaired": str(training_name != record["jobName"]).lower(),
        }
    )
    request: dict[str, Any] = {
        "jobName": name,
        "jobQueue": record["jobQueue"],
        "retryStrategy": record["retryStrategy"],
        "schedulingPriority": priority,
        "serviceRequestPayload": replay_payload,
        "serviceJobType": record["serviceJobType"],
        "timeoutConfig": record["timeoutConfig"],
        "tags": tags,
        "clientToken": canonical_sha(f"robomme-p5-quota-retry-v1:{source_id}:{generation}:priority{priority}"),
    }
    for key in (
        "shareIdentifier",
        "quotaShareName",
        "preemptionConfiguration",
    ):
        if record.get(key) is not None:
            request[key] = record[key]
    source_payload = json.loads(record["serviceRequestPayload"])
    submitted_payload = json.loads(request["serviceRequestPayload"])
    if scientific_payload_projection(submitted_payload) != scientific_payload_projection(source_payload):
        raise GuardrailError("scientific service request payload changed")
    return request


class CloudAudit:
    def __init__(self) -> None:
        try:
            import boto3
        except ImportError as error:
            raise GuardrailError("boto3 is required for S3/SageMaker audits") from error
        self.s3 = boto3.client("s3", region_name=REGION)
        self.sagemaker = boto3.client("sagemaker", region_name=REGION)

    def object_exists(self, uri: str) -> bool:
        from botocore.exceptions import ClientError

        bucket, key = s3_parts(uri)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code"))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def prefix_exists(self, uri: str) -> bool:
        bucket, key = s3_parts(uri)
        response = self.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=key.rstrip("/") + "/",
            MaxKeys=1,
        )
        return bool(response.get("Contents"))

    def training_job_exists(self, name: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.sagemaker.describe_training_job(TrainingJobName=name)
            return True
        except ClientError as error:
            message = error.response.get("Error", {}).get("Message", "").lower()
            if "requested resource not found" in message:
                return False
            raise

    def p5_in_progress(self) -> tuple[int, list[str]]:
        paginator = self.sagemaker.get_paginator("list_training_jobs")
        names: list[str] = []
        for page in paginator.paginate(StatusEquals="InProgress"):
            names.extend(item["TrainingJobName"] for item in page["TrainingJobSummaries"])

        def describe(name: str) -> tuple[str, int]:
            job = self.sagemaker.describe_training_job(TrainingJobName=name)
            resource = job.get("ResourceConfig", {})
            count = int(resource.get("InstanceCount", 0)) if resource.get("InstanceType") == INSTANCE_TYPE else 0
            return name, count

        p5: list[str] = []
        if names:
            with ThreadPoolExecutor(max_workers=min(12, len(names))) as pool:
                for name, count in pool.map(describe, names):
                    p5.extend([name] * count)
        return len(p5), p5


def audit_unrun_source(
    cloud: CloudAudit,
    source_record: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    environment = validated["environment"]
    source = source_archive_uri(validated["payload"])
    if not cloud.object_exists(source):
        raise GuardrailError(f"{source_record['jobId']}: immutable source archive is missing")
    for label, uri in (
        ("run manifest", environment["RUN_MANIFEST_S3"]),
        ("producer claim", environment["PRODUCER_CLAIM_S3"]),
        ("completion claim", environment["COMPLETION_CLAIM_S3"]),
    ):
        if cloud.object_exists(uri):
            raise GuardrailError(f"{source_record['jobId']}: unexpected {label} exists at {uri}")
    if cloud.prefix_exists(environment["OUTPUT_S3"]):
        raise GuardrailError(f"{source_record['jobId']}: unexpected checkpoint/output prefix is nonempty")
    name = valid_training_name(validated["payload"]["TrainingJobName"])
    if cloud.training_job_exists(name):
        raise GuardrailError(f"{source_record['jobId']}: SageMaker training job {name} already exists")
    return {
        "job_id": source_record["jobId"],
        "run_id": environment["ROBOMME_RUN_ID"],
        "attempt_id": environment["ROBOMME_ATTEMPT_ID"],
        "task": environment.get("ROBOMME_TASK"),
        "arm": environment.get("ROBOMME_ARM"),
        "manifest_sha256": environment["RUN_MANIFEST_SHA256"],
        "service_request_payload_sha256": validated["raw_payload_sha256"],
        "source_archive": source,
        "target_training_job_name": name,
    }


def read_spec(path: Path) -> dict[str, Any]:
    spec = load_json(path)
    if spec.get("schema_version") != 1:
        raise GuardrailError("unsupported retry source schema")
    if (
        spec.get("region"),
        spec.get("queue"),
        spec.get("instance_type"),
        spec.get("priority"),
    ) != (REGION, QUEUE, INSTANCE_TYPE, 1):
        raise GuardrailError("retry source infrastructure contract drifted")
    ids = spec.get("source_job_ids", [])
    if len(ids) != 44 or len(set(ids)) != 44:
        raise GuardrailError("retry source inventory must contain 44 unique jobs")
    if spec.get("expected_terminal_counts") != {"SUCCEEDED": 9, "QUOTA_FAILED": 35}:
        raise GuardrailError("retry source terminal-count contract drifted")
    return spec


def read_or_create_state(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        state = load_json(path)
        if state.get("source_inventory_sha256") != canonical_sha(
            json.dumps(spec, sort_keys=True, separators=(",", ":"))
        ):
            raise GuardrailError("state belongs to a different source inventory")
        return state
    return {
        "schema_version": 1,
        "campaign": spec["name"],
        "source_inventory_sha256": canonical_sha(json.dumps(spec, sort_keys=True, separators=(",", ":"))),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "sources": {},
    }


def describe_sources(spec: dict[str, Any]) -> list[dict[str, Any]]:
    ids = spec["source_job_ids"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(describe_service_job, ids))


def initial_audit(
    spec: dict[str, Any],
    state: dict[str, Any],
    cloud: CloudAudit,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = describe_sources(spec)
    validated = [
        validate_source_job(record, expected_job_id=job_id)
        for job_id, record in zip(spec["source_job_ids"], records, strict=True)
    ]
    counts = {
        "SUCCEEDED": sum(value["status"] == "SUCCEEDED" for value in validated),
        "QUOTA_FAILED": sum(value["status"] == "FAILED" for value in validated),
    }
    if counts != spec["expected_terminal_counts"]:
        raise GuardrailError(f"source ledger drifted: expected {spec['expected_terminal_counts']}, got {counts}")

    pending_audits: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, value in zip(records, validated, strict=True):
        source_state = state["sources"].setdefault(
            record["jobId"],
            {
                "source_job_name": record["jobName"],
                "run_id": value["environment"]["ROBOMME_RUN_ID"],
                "task": value["environment"].get("ROBOMME_TASK"),
                "arm": value["environment"].get("ROBOMME_ARM"),
                "replays": [],
            },
        )
        if value["status"] == "SUCCEEDED":
            source_state["disposition"] = "original_succeeded"
        elif not source_state["replays"]:
            pending_audits.append((record, value))

    audits: list[dict[str, Any]] = []
    if pending_audits:
        with ThreadPoolExecutor(max_workers=8) as pool:
            audits = list(
                pool.map(
                    lambda pair: audit_unrun_source(cloud, pair[0], pair[1]),
                    pending_audits,
                )
            )
    by_id = {record["jobId"]: record for record in records}
    for audit in audits:
        state["sources"][audit["job_id"]]["preflight"] = audit
        state["sources"][audit["job_id"]]["disposition"] = "verified_unrun"
    state["updated_at"] = utc_now()
    return records, {"counts": counts, "audited_unrun": len(audits), "records": by_id}


def reconcile_replays(
    state: dict[str, Any],
) -> tuple[list[str], list[str]]:
    pending: list[str] = []
    running: list[str] = []
    for source_id, source in state["sources"].items():
        replays = source.get("replays", [])
        if not replays:
            continue
        latest = replays[-1]
        record = describe_service_job(latest["job_id"])
        latest.update(
            {
                "status": record["status"],
                "status_reason": record.get("statusReason"),
                "attempt_count": len(record.get("attempts", [])),
                "observed_at": utc_now(),
            }
        )
        status = record["status"]
        if status == "SUCCEEDED":
            source["disposition"] = "replay_succeeded"
        elif status == "RUNNING":
            source["disposition"] = "replay_running"
            running.append(source_id)
        elif status in ADMISSION_PENDING:
            source["disposition"] = "replay_admission_pending"
            pending.append(source_id)
        elif status == "FAILED":
            reason = record.get("statusReason", "")
            if record.get("attempts"):
                raise GuardrailError(
                    f"replay {record['jobId']} entered SageMaker and failed; manual diagnosis required"
                )
            if QUOTA_REASON not in reason:
                raise GuardrailError(f"replay {record['jobId']} failed for a non-quota reason: {reason}")
            source["disposition"] = "quota_retryable"
        else:
            raise GuardrailError(f"replay {record['jobId']} has unknown status {status}")
    state["updated_at"] = utc_now()
    return pending, running


def submit_replay(
    source_record: dict[str, Any],
    source_state: dict[str, Any],
    *,
    max_admission_retries: int,
    scheduling_priority: int,
) -> dict[str, Any]:
    generation = len(source_state.get("replays", [])) + len(source_state.get("retired_replays", [])) + 1
    if generation > max_admission_retries:
        raise GuardrailError(f"{source_record['jobId']}: exhausted {max_admission_retries} quota admission retries")
    request = build_replay_request(source_record, generation, scheduling_priority)
    try:
        response = aws_cli(
            "batch",
            "submit-service-job",
            "--cli-input-json",
            json.dumps(request, separators=(",", ":")),
        )
    except AwsCliError:
        matches = aws_cli(
            "batch",
            "list-service-jobs",
            "--job-queue",
            source_record["jobQueue"],
            "--filters",
            json.dumps([{"name": "JOB_NAME", "values": [request["jobName"]]}]),
        ).get("jobSummaryList", [])
        exact = [item for item in matches if item.get("jobName") == request["jobName"]]
        if len(exact) != 1:
            raise
        response = exact[0]
    job_id = response.get("jobId")
    if not job_id:
        raise GuardrailError(f"submission for {source_record['jobId']} returned no Batch job ID")
    submitted_payload = json.loads(request["serviceRequestPayload"])
    replay = {
        "generation": generation,
        "job_id": job_id,
        "job_name": request["jobName"],
        "submitted_at": utc_now(),
        "client_token": request["clientToken"],
        "source_payload_sha256": canonical_sha(source_record["serviceRequestPayload"]),
        "submitted_payload_sha256": canonical_sha(request["serviceRequestPayload"]),
        "scheduling_priority": request["schedulingPriority"],
        "training_job_name": submitted_payload["TrainingJobName"],
        "sagemaker_name_repaired": (submitted_payload["TrainingJobName"] != source_record["jobName"]),
        "status": "SUBMITTED",
    }
    source_state.setdefault("replays", []).append(replay)
    source_state["disposition"] = "replay_admission_pending"
    return replay


def choose_source(
    spec: dict[str, Any],
    state: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    cloud: CloudAudit,
) -> str | None:
    for source_id in spec["source_job_ids"]:
        source = state["sources"][source_id]
        if source.get("disposition") not in {"verified_unrun", "quota_retryable"}:
            continue
        record = by_id[source_id]
        value = validate_source_job(record, expected_job_id=source_id)
        audit_unrun_source(cloud, record, value)
        return source_id
    return None


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.handlers[:] = [stream, file_handler]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    value.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_RUNTIME / "p5_priority1_retry_v1.json",
    )
    value.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_RUNTIME / "p5_priority1_retry_v1.log",
    )
    value.add_argument("--quota-limit", type=int, default=10)
    value.add_argument("--safety-headroom", type=int, default=1)
    value.add_argument("--poll-seconds", type=int, default=120)
    value.add_argument("--max-admission-retries", type=int, default=10)
    value.add_argument("--audit-only", action="store_true")
    value.add_argument("--submission-priority", type=int, choices=(1, 400), default=1)
    value.add_argument("--confirm-submit", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.poll_seconds < 30:
        raise SystemExit("--poll-seconds must be at least 30")
    if not args.audit_only and not args.confirm_submit:
        raise SystemExit("submission blocked: obtain explicit user approval, then pass --confirm-submit")
    configure_logging(args.log)
    spec = read_spec(args.spec)
    state = read_or_create_state(args.state, spec)
    cloud = CloudAudit()
    from botocore.exceptions import BotoCoreError, ClientError

    records, audit = initial_audit(spec, state, cloud)
    by_id = audit["records"]
    atomic_json(args.state, state)
    LOG.info(
        "source audit complete: original_succeeded=%d verified_unrun=%d",
        audit["counts"]["SUCCEEDED"],
        audit["audited_unrun"],
    )
    if args.audit_only:
        print(
            json.dumps(
                {
                    "source_counts": audit["counts"],
                    "verified_unrun": audit["audited_unrun"],
                    "state": str(args.state),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    while True:
        try:
            pending, running = reconcile_replays(state)
            completed = sum(
                source.get("disposition") in {"original_succeeded", "replay_succeeded"}
                for source in state["sources"].values()
            )
            atomic_json(args.state, state)
            if completed == len(spec["source_job_ids"]):
                LOG.info("campaign complete: all %d scientific cells succeeded", completed)
                return
            if pending:
                LOG.info(
                    "one replay awaits admission (%s); no additional job will be submitted",
                    pending[0],
                )
                time.sleep(args.poll_seconds)
                continue

            p5_count, names = cloud.p5_in_progress()
            slots = available_slots(p5_count, args.quota_limit, args.safety_headroom)
            LOG.info(
                "progress=%d/44 replay_running=%d p5_in_progress=%d safe_slots=%d",
                completed,
                len(running),
                p5_count,
                slots,
            )
            if not slots:
                time.sleep(args.poll_seconds)
                continue

            source_id = choose_source(spec, state, by_id, cloud)
            if source_id is None:
                LOG.info("no unsubmitted source remains; waiting for %d running replay(s)", len(running))
                time.sleep(args.poll_seconds)
                continue
            replay = submit_replay(
                by_id[source_id],
                state["sources"][source_id],
                max_admission_retries=args.max_admission_retries,
                scheduling_priority=args.submission_priority,
            )
            atomic_json(args.state, state)
            LOG.info(
                "submitted scientific-payload replay source=%s replay=%s name=%s priority=%d",
                source_id,
                replay["job_id"],
                replay["job_name"],
                replay["scheduling_priority"],
            )
            time.sleep(args.poll_seconds)
        except (AwsCliError, BotoCoreError, ClientError, OSError) as error:
            LOG.warning("transient control-plane error; retrying after poll interval: %s", error)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
