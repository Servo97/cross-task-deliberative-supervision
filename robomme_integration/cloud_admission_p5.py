#!/usr/bin/env python3
"""Fail-closed read-only admission for the ordinary RoboMME p5/H100 queue.

Unlike the p5e training-plan gate, the ordinary p5 queue exposes no authoritative free-node
counter.  Admission therefore makes no capacity claim: every committed/waiting service job must
be absent, while RUNNING jobs are scanned only for an exact duplicate identity.  SageMaker and S3
surfaces are fully paginated and every active Batch summary is expanded with DescribeServiceJob.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from typing import Any
from urllib.parse import urlparse

from scripts.launch.launch_guardrails import EXECUTION_ACCOUNT, QUEUE, REGION

WAITING_BATCH_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "SCHEDULED", "STARTING")
ACTIVE_BATCH_STATUSES = (*WAITING_BATCH_STATUSES, "RUNNING")
PAGE_SIZE = 100
MAX_PAGES = 100
P5_INSTANCE_TYPE = "ml.p5.48xlarge"
P5_QUEUE_ARN = f"arn:aws:batch:{REGION}:{EXECUTION_ACCOUNT}:job-queue/{QUEUE}"
P5_QUOTA_CODE = "L-82E1C851"
P5_QUOTA_NAME = "ml.p5.48xlarge for training job usage"


def _aws_json(*arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["aws", *arguments, "--region", REGION, "--output", "json", "--no-cli-pager"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError(f"AWS returned non-object JSON for {' '.join(arguments[:2])}")
    return value


def _pages(arguments: list[str], *, items_key: str, token_key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token: str | None = None
    observed_tokens: set[str] = set()
    for _page in range(MAX_PAGES):
        current = list(arguments)
        if token is not None:
            current.extend(["--next-token", token])
        response = _aws_json(*current)
        page = response.get(items_key)
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise RuntimeError(f"AWS pagination response omitted or malformed {items_key}")
        items.extend(page)
        next_token = response.get(token_key)
        if next_token is None:
            return items
        if not isinstance(next_token, str) or not next_token or next_token in observed_tokens:
            raise RuntimeError(f"AWS pagination returned invalid/repeated {token_key}")
        observed_tokens.add(next_token)
        token = next_token
    raise RuntimeError(f"AWS pagination exceeded {MAX_PAGES} pages for {items_key}")


def _s3_objects(uri: str) -> list[dict[str, Any]]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 namespace {uri!r}")
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    objects: list[dict[str, Any]] = []
    token: str | None = None
    observed_tokens: set[str] = set()
    for _page in range(MAX_PAGES):
        arguments = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
        ]
        if token is not None:
            arguments.extend(["--continuation-token", token])
        response = _aws_json(*arguments)
        contents = response.get("Contents", [])
        if not isinstance(contents, list) or any(not isinstance(item, dict) for item in contents):
            raise RuntimeError("S3 listing returned malformed Contents")
        objects.extend(contents)
        truncated = response.get("IsTruncated")
        if truncated is None and not contents and response.get("NextContinuationToken") is None:
            # The real CLI may omit false-valued IsTruncated on an empty result. This is safe only
            # for that exact terminal shape; any objects or token with a missing Boolean reject.
            return objects
        if truncated is False:
            return objects
        if truncated is not True:
            raise RuntimeError("S3 listing omitted a Boolean IsTruncated")
        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token in observed_tokens:
            raise RuntimeError("truncated S3 listing omitted/repeated NextContinuationToken")
        observed_tokens.add(next_token)
        token = next_token
    raise RuntimeError(f"S3 listing exceeded {MAX_PAGES} pages for {uri}")


def campaign_namespaces(manifest: dict[str, Any]) -> list[str]:
    claims = manifest["claims"]
    namespaces = [claims["manifest"], claims["attempt_result"], claims["completion"]]
    for cell in manifest["cells"]:
        environment = cell["environment"]
        namespaces.extend(
            [
                environment["RUN_MANIFEST_S3"],
                environment["PRODUCER_CLAIM_S3"],
                environment["COMPLETION_CLAIM_S3"],
                environment["OUTPUT_S3"].rstrip("/") + "/",
            ]
        )
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("campaign contains duplicate publication namespaces")
    return namespaces


def _identity_duplicates(
    *, batch_jobs: list[dict[str, Any]], training_jobs: list[dict[str, Any]], tokens: set[str]
) -> list[str]:
    duplicates: list[str] = []
    for surface, records in (("batch", batch_jobs), ("sagemaker", training_jobs)):
        for record in records:
            rendered = json.dumps(record, sort_keys=True, separators=(",", ":"))
            matched = sorted(token for token in tokens if token and token in rendered)
            if matched:
                # The described payload may contain third-party tokens. Retain only candidate
                # identity matches and a one-way record digest; never log or persist raw payloads.
                record_sha = hashlib.sha256(rendered.encode()).hexdigest()
                duplicates.append(f"{surface}:matches={matched}:record_sha256={record_sha}")
    return duplicates


def validate_snapshot(
    *,
    account: str,
    batch_jobs_by_status: dict[str, list[dict[str, Any]]],
    training_jobs: list[dict[str, Any]],
    namespace_objects: dict[str, list[dict[str, Any]]],
    quota: dict[str, Any],
    identity_tokens: set[str],
    expected_namespaces: set[str],
    identity: dict[str, str],
    allow_backlog: bool = False,
) -> dict[str, Any]:
    if account != EXECUTION_ACCOUNT:
        raise ValueError(f"admission requires AWS account {EXECUTION_ACCOUNT}; got {account}")
    if set(batch_jobs_by_status) != set(ACTIVE_BATCH_STATUSES):
        raise ValueError("admission snapshot did not cover all six Batch statuses")
    waiting_counts = {status: len(batch_jobs_by_status[status]) for status in WAITING_BATCH_STATUSES}
    if not allow_backlog and any(waiting_counts.values()):
        raise ValueError(f"ordinary p5 queue already has committed waiting work: {waiting_counts}")
    all_batch_jobs = [record for status in ACTIVE_BATCH_STATUSES for record in batch_jobs_by_status[status]]
    target_queue_running_capacity = sum(
        int(record["capacityUsage"][0]["quantity"]) for record in batch_jobs_by_status["RUNNING"]
    )
    if quota.get("QuotaCode") != P5_QUOTA_CODE or quota.get("QuotaName") != P5_QUOTA_NAME:
        raise ValueError("p5 Service Quota identity/status drifted")
    try:
        quota_value = float(quota["Value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("p5 Service Quota value missing/malformed") from error
    if quota_value <= 0 or not quota_value.is_integer():
        raise ValueError(f"p5 Service Quota is not a positive integer: {quota_value}")
    quota_units = int(quota_value)
    global_p5_usage = 0
    target_queue_training_usage: dict[str, int] = {}
    for record in batch_jobs_by_status["RUNNING"]:
        service_resource = record.get("latestAttempt", {}).get("serviceResourceId", {})
        arn = service_resource.get("value")
        if service_resource.get("name") != "TrainingJobArn" or not isinstance(arn, str) or ":training-job/" not in arn:
            raise RuntimeError(f"RUNNING Batch job omitted SageMaker TrainingJobArn for jobId={record.get('jobId')}")
        training_name = arn.rsplit("/", 1)[-1]
        if training_name in target_queue_training_usage:
            raise ValueError(f"duplicate RUNNING Batch TrainingJobArn name {training_name}")
        target_queue_training_usage[training_name] = int(record["capacityUsage"][0]["quantity"])
    global_training_usage: dict[str, int] = {}
    for record in training_jobs:
        name = record.get("TrainingJobName")
        resource = record.get("ResourceConfig")
        if not isinstance(name, str) or not name or not isinstance(resource, dict):
            raise RuntimeError("InProgress SageMaker description omitted name/resource config")
        instance_type = resource.get("InstanceType")
        raw_count = resource.get("InstanceCount")
        try:
            numeric_count = float(raw_count)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"SageMaker InstanceCount malformed for {name}") from error
        if isinstance(raw_count, bool) or numeric_count < 1 or not numeric_count.is_integer():
            raise RuntimeError(f"SageMaker InstanceCount must be a positive integer for {name}")
        count = int(numeric_count)
        if instance_type == P5_INSTANCE_TYPE:
            global_p5_usage += count
            if name in global_training_usage:
                raise ValueError(f"duplicate InProgress SageMaker training name {name}")
            global_training_usage[name] = count
    mismatched = {
        name: {"batch": expected, "sagemaker": global_training_usage.get(name)}
        for name, expected in target_queue_training_usage.items()
        if global_training_usage.get(name) != expected
    }
    if mismatched:
        raise ValueError(f"RUNNING Batch/SageMaker p5 per-job reconciliation failed: {mismatched}")
    available_capacity = quota_units - global_p5_usage
    if available_capacity > quota_units or (available_capacity < 1 and not allow_backlog):
        raise ValueError(
            f"ordinary p5 queue lacks a free node: quota={quota_units} "
            f"global_running={global_p5_usage} available={available_capacity}"
        )
    if available_capacity < 0:
        raise ValueError(
            f"global p5 usage exceeds the service quota: quota={quota_units} "
            f"global_running={global_p5_usage} available={available_capacity}"
        )
    duplicates = _identity_duplicates(batch_jobs=all_batch_jobs, training_jobs=training_jobs, tokens=identity_tokens)
    if duplicates:
        raise ValueError(f"active duplicate campaign/run identity found: {duplicates}")
    if set(namespace_objects) != expected_namespaces:
        raise ValueError("admission snapshot did not cover every exact publication namespace")
    nonempty = {uri: objects for uri, objects in namespace_objects.items() if objects}
    if nonempty:
        raise ValueError(f"create-once namespace is nonempty: {nonempty}")
    return {
        "schema_version": 1,
        "kind": "robomme_p5_ordinary_queue_live_admission",
        "cloud_action": False,
        "account": account,
        "queue": QUEUE,
        "instance_type": P5_INSTANCE_TYPE,
        "training_plan_arn": None,
        "capacity_observation": "global_paginated_inprogress_sagemaker_p5_instance_usage",
        "p5_quota_code": P5_QUOTA_CODE,
        "p5_quota_name": P5_QUOTA_NAME,
        "p5_instance_quota": quota_units,
        "target_queue_running_p5_instance_units": target_queue_running_capacity,
        "global_running_p5_instance_units": global_p5_usage,
        "available_p5_instance_units": available_capacity,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scheduling_policy": ("user_authorized_backlog" if allow_backlog else "require_immediate_capacity"),
        "waiting_batch_statuses_required_zero": ([] if allow_backlog else list(WAITING_BATCH_STATUSES)),
        "waiting_batch_counts": waiting_counts,
        "running_batch_jobs_scanned": len(batch_jobs_by_status["RUNNING"]),
        "active_training_jobs_scanned": len(training_jobs),
        "publication_namespaces_proven_empty": len(namespace_objects),
        **identity,
        "admitted": True,
    }


def _collect_batch_jobs() -> dict[str, list[dict[str, Any]]]:
    jobs_by_status: dict[str, list[dict[str, Any]]] = {}
    descriptions: dict[str, dict[str, Any]] = {}
    for status in ACTIVE_BATCH_STATUSES:
        summaries = _pages(
            [
                "batch",
                "list-service-jobs",
                "--job-queue",
                QUEUE,
                "--job-status",
                status,
                "--max-results",
                str(PAGE_SIZE),
            ],
            items_key="jobSummaryList",
            token_key="nextToken",
        )
        records = []
        for summary in summaries:
            job_id = summary.get("jobId")
            if not isinstance(job_id, str) or not job_id:
                raise RuntimeError("active Batch service-job summary omitted jobId")
            if job_id not in descriptions:
                descriptions[job_id] = _aws_json("batch", "describe-service-job", "--job-id", job_id)
            record = descriptions[job_id]
            capacity = record.get("capacityUsage")
            if (
                record.get("jobId") != job_id
                or record.get("jobName") != summary.get("jobName")
                or record.get("status") != status
                or record.get("jobQueue") not in {QUEUE, P5_QUEUE_ARN}
                or record.get("serviceJobType") != "SAGEMAKER_TRAINING"
                or not isinstance(capacity, list)
                or len(capacity) != 1
                or capacity[0].get("capacityUnit") != P5_INSTANCE_TYPE
                or capacity != summary.get("capacityUsage")
            ):
                raise RuntimeError(f"Batch summary/description binding drifted for jobId={job_id}")
            quantity = capacity[0].get("quantity")
            try:
                numeric_quantity = float(quantity)
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Batch capacity quantity malformed for jobId={job_id}") from error
            if numeric_quantity <= 0 or not numeric_quantity.is_integer():
                raise RuntimeError(f"Batch capacity quantity invalid for jobId={job_id}")
            # Keep a normalized safe surface alongside the full in-memory description. Duplicate
            # scanning still sees the payload, but receipts and failures never serialize it.
            record["capacityUsage"] = [{"capacityUnit": P5_INSTANCE_TYPE, "quantity": int(numeric_quantity)}]
            records.append(record)
        jobs_by_status[status] = records
    return jobs_by_status


def _collect_training_jobs() -> list[dict[str, Any]]:
    summaries = _pages(
        [
            "sagemaker",
            "list-training-jobs",
            "--status-equals",
            "InProgress",
            "--max-results",
            str(PAGE_SIZE),
        ],
        items_key="TrainingJobSummaries",
        token_key="NextToken",
    )
    jobs = []
    for summary in summaries:
        name = summary.get("TrainingJobName")
        if not isinstance(name, str) or not name:
            raise RuntimeError("SageMaker InProgress summary omitted TrainingJobName")
        record = _aws_json("sagemaker", "describe-training-job", "--training-job-name", name)
        if record.get("TrainingJobName") != name or record.get("TrainingJobStatus") != "InProgress":
            raise RuntimeError(f"SageMaker summary/description binding drifted for {name}")
        jobs.append(record)
    return jobs


def _collect_p5_quota() -> dict[str, Any]:
    response = _aws_json(
        "service-quotas",
        "get-service-quota",
        "--service-code",
        "sagemaker",
        "--quota-code",
        P5_QUOTA_CODE,
    )
    quota = response.get("Quota")
    if not isinstance(quota, dict):
        raise RuntimeError("Service Quotas response omitted Quota")
    return quota


def collect_and_validate(manifest: dict[str, Any], *, job_name: str) -> dict[str, Any]:
    namespaces = set(campaign_namespaces(manifest))
    tokens = {manifest["campaign_id"], manifest["attempt_id"], job_name}
    tokens.update(cell["run_id"] for cell in manifest["cells"])
    return validate_snapshot(
        account=str(_aws_json("sts", "get-caller-identity").get("Account")),
        batch_jobs_by_status=_collect_batch_jobs(),
        training_jobs=_collect_training_jobs(),
        namespace_objects={uri: _s3_objects(uri) for uri in namespaces},
        quota=_collect_p5_quota(),
        identity_tokens=tokens,
        expected_namespaces=namespaces,
        identity={
            "campaign_id": manifest["campaign_id"],
            "attempt_id": manifest["attempt_id"],
            "job_name": job_name,
        },
    )


def collect_canary_admission(*, canary_id: str, job_name: str, namespace_s3: str) -> dict[str, Any]:
    namespace = namespace_s3.rstrip("/") + "/"
    return validate_snapshot(
        account=str(_aws_json("sts", "get-caller-identity").get("Account")),
        batch_jobs_by_status=_collect_batch_jobs(),
        training_jobs=_collect_training_jobs(),
        namespace_objects={namespace: _s3_objects(namespace)},
        quota=_collect_p5_quota(),
        identity_tokens={canary_id, f"{canary_id}-attempt1", job_name},
        expected_namespaces={namespace},
        identity={
            "canary_id": canary_id,
            "attempt_id": f"{canary_id}-attempt1",
            "job_name": job_name,
            "namespace_s3": namespace_s3.rstrip("/"),
        },
    )


def collect_canary_backlog_admission(*, canary_id: str, job_name: str, namespace_s3: str) -> dict[str, Any]:
    """Validate identity/publication safety while allowing the exact canary to wait for p5."""
    namespace = namespace_s3.rstrip("/") + "/"
    return validate_snapshot(
        account=str(_aws_json("sts", "get-caller-identity").get("Account")),
        batch_jobs_by_status=_collect_batch_jobs(),
        training_jobs=_collect_training_jobs(),
        namespace_objects={namespace: _s3_objects(namespace)},
        quota=_collect_p5_quota(),
        identity_tokens={canary_id, f"{canary_id}-attempt1", job_name},
        expected_namespaces={namespace},
        identity={
            "canary_id": canary_id,
            "attempt_id": f"{canary_id}-attempt1",
            "job_name": job_name,
            "namespace_s3": namespace_s3.rstrip("/"),
        },
        allow_backlog=True,
    )
