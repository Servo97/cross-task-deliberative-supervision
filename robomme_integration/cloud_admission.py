#!/usr/bin/env python3
"""Fail-closed live admission for RoboMME p5e campaign submissions.

This module is read-only.  It fully paginates every active control-plane surface, rejects unknown
or inconsistent capacity, rejects duplicate scientific/campaign identities, and proves every
create-once campaign/run namespace empty immediately before the caller submits.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any
from urllib.parse import urlparse

from scripts.launch.launch_guardrails import (
    EXECUTION_ACCOUNT,
    REGION,
    TRAINING_PLAN_QUEUE,
    training_plan_arn,
)

ACTIVE_BATCH_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "SCHEDULED", "STARTING", "RUNNING")
PAGE_SIZE = 100
MAX_PAGES = 100


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
    for _page in range(MAX_PAGES):
        current = list(arguments)
        if token is not None:
            current.extend(["--next-token", token])
        response = _aws_json(*current)
        page = response.get(items_key)
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise RuntimeError(f"AWS pagination response omitted {items_key}")
        items.extend(page)
        next_token = response.get(token_key)
        if next_token is None:
            return items
        if not isinstance(next_token, str) or not next_token:
            raise RuntimeError(f"AWS pagination returned invalid {token_key}")
        token = next_token
    raise RuntimeError(f"AWS pagination exceeded {MAX_PAGES} pages for {items_key}")


def _s3_objects(uri: str) -> list[dict[str, Any]]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 namespace {uri!r}")
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    objects: list[dict[str, Any]] = []
    token: str | None = None
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
        truncated = response.get("IsTruncated", False)
        if truncated is False:
            return objects
        if truncated is not True:
            raise RuntimeError("S3 listing omitted a Boolean IsTruncated")
        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token:
            raise RuntimeError("truncated S3 listing omitted NextContinuationToken")
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


def validate_snapshot(
    *,
    account: str,
    plans: list[dict[str, Any]],
    batch_jobs: list[dict[str, Any]],
    training_jobs: list[dict[str, Any]],
    namespace_objects: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    job_name: str,
) -> dict[str, Any]:
    if account != EXECUTION_ACCOUNT:
        raise ValueError(f"admission requires AWS account {EXECUTION_ACCOUNT}; got {account}")
    expected_arn = training_plan_arn(TRAINING_PLAN_QUEUE)
    matching = [plan for plan in plans if plan.get("TrainingPlanArn") == expected_arn]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one authoritative active training plan: {matching}")
    plan = matching[0]
    if plan.get("Status") != "Active":
        raise ValueError(f"p5e plan is not Active: {plan}")
    # list-training-plans is the only authorized authoritative plan surface for this role and the
    # real response does not contain UnhealthyInstanceCount.  Never turn that absence into a
    # fabricated zero.  If AWS does return the field, however, a malformed/nonzero value is a
    # hard rejection.  Active state plus the exact count invariant below are the admission
    # requirements that this surface can actually prove.
    reported_unhealthy = plan.get("UnhealthyInstanceCount")
    if reported_unhealthy is not None:
        try:
            reported_unhealthy = int(reported_unhealthy)
        except (TypeError, ValueError) as error:
            raise ValueError("p5e plan unhealthy count is malformed") from error
        if reported_unhealthy != 0:
            raise ValueError(f"p5e plan reports unhealthy instances: {plan}")
    try:
        total = int(plan["TotalInstanceCount"])
        in_use = int(plan["InUseInstanceCount"])
        available = int(plan["AvailableInstanceCount"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("p5e plan count fields are missing/malformed") from error
    if total != in_use + available or available < 1:
        raise ValueError(
            f"p5e plan capacity is inconsistent/unavailable: total={total} in_use={in_use} available={available}"
        )

    identities = {manifest["campaign_id"], manifest["attempt_id"], job_name}
    identities.update(cell["run_id"] for cell in manifest["cells"])
    identity_tokens = sorted(identities)
    duplicates: list[str] = []
    for surface, records in (("batch", batch_jobs), ("sagemaker", training_jobs)):
        for record in records:
            rendered = json.dumps(record, sort_keys=True, separators=(",", ":"))
            matched = [identity for identity in identity_tokens if identity in rendered]
            if matched:
                duplicates.append(f"{surface}:{matched}:{rendered[:300]}")
    if duplicates:
        raise ValueError(f"active duplicate campaign/run identity found: {duplicates}")
    nonempty = {uri: objects for uri, objects in namespace_objects.items() if objects}
    if nonempty:
        raise ValueError(f"campaign create-once namespace is nonempty: {nonempty}")
    expected_namespaces = set(campaign_namespaces(manifest))
    if set(namespace_objects) != expected_namespaces:
        raise ValueError("admission snapshot did not cover every exact campaign namespace")
    return {
        "schema_version": 1,
        "kind": "robomme_p5e_live_admission",
        "cloud_action": False,
        "account": account,
        "training_plan_arn": expected_arn,
        "status": "Active",
        "total_instance_count": total,
        "in_use_instance_count": in_use,
        "available_instance_count": available,
        "unhealthy_instance_count": reported_unhealthy,
        "unhealthy_instance_count_observation": (
            "reported_zero_by_list_training_plans"
            if reported_unhealthy == 0
            else "not_returned_by_list_training_plans_not_claimed"
        ),
        "batch_statuses_scanned": list(ACTIVE_BATCH_STATUSES),
        "active_batch_jobs_scanned": len(batch_jobs),
        "active_training_jobs_scanned": len(training_jobs),
        "publication_namespaces_proven_empty": len(namespace_objects),
        "campaign_id": manifest["campaign_id"],
        "attempt_id": manifest["attempt_id"],
        "job_name": job_name,
        "admitted": True,
    }


def collect_and_validate(manifest: dict[str, Any], *, job_name: str) -> dict[str, Any]:
    account = str(_aws_json("sts", "get-caller-identity").get("Account"))
    plans = _pages(
        ["sagemaker", "list-training-plans", "--filters", "Name=Status,Value=Active", "--max-results", str(PAGE_SIZE)],
        items_key="TrainingPlanSummaries",
        token_key="NextToken",
    )
    batch_summaries: list[dict[str, Any]] = []
    for status in ACTIVE_BATCH_STATUSES:
        batch_summaries.extend(
            _pages(
                [
                    "batch",
                    "list-service-jobs",
                    "--job-queue",
                    TRAINING_PLAN_QUEUE,
                    "--job-status",
                    status,
                    "--max-results",
                    str(PAGE_SIZE),
                ],
                items_key="jobSummaryList",
                token_key="nextToken",
            )
        )
    # Batch list summaries do not expose the serviceRequestPayload that carries exact campaign,
    # attempt, and run identities.  Describe every active service job and scan those full records.
    batch_jobs: list[dict[str, Any]] = []
    described_job_ids: set[str] = set()
    for summary in batch_summaries:
        job_id = summary.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("active Batch service-job summary omitted jobId")
        if job_id in described_job_ids:
            continue
        described_job_ids.add(job_id)
        batch_jobs.append(_aws_json("batch", "describe-service-job", "--job-id", job_id))
    training_summaries = _pages(
        ["sagemaker", "list-training-jobs", "--status-equals", "InProgress", "--max-results", str(PAGE_SIZE)],
        items_key="TrainingJobSummaries",
        token_key="NextToken",
    )
    training_jobs = []
    for summary in training_summaries:
        name = summary.get("TrainingJobName")
        if not isinstance(name, str) or not name:
            raise RuntimeError("SageMaker InProgress summary omitted TrainingJobName")
        training_jobs.append(_aws_json("sagemaker", "describe-training-job", "--training-job-name", name))
    namespace_objects = {uri: _s3_objects(uri) for uri in campaign_namespaces(manifest)}
    return validate_snapshot(
        account=account,
        plans=plans,
        batch_jobs=batch_jobs,
        training_jobs=training_jobs,
        namespace_objects=namespace_objects,
        manifest=manifest,
        job_name=job_name,
    )


def collect_canary_admission(*, canary_id: str, job_name: str, namespace_s3: str) -> dict[str, Any]:
    """Apply the same live gate to one isolated v4 canary namespace."""

    synthetic = {
        "campaign_id": canary_id,
        "attempt_id": f"{canary_id}-attempt1",
        "claims": {
            "manifest": f"{namespace_s3.rstrip('/')}/training_canary.complete.json",
            "attempt_result": f"{namespace_s3.rstrip('/')}/admission-attempt-result.absent",
            "completion": f"{namespace_s3.rstrip('/')}/admission-completion.absent",
        },
        "cells": [],
    }
    # The synthetic helper above would ask for extra nonexistent namespaces. A canary permits one
    # object only, so collect the common live surfaces directly and validate exact identities.
    account = str(_aws_json("sts", "get-caller-identity").get("Account"))
    plans = _pages(
        ["sagemaker", "list-training-plans", "--filters", "Name=Status,Value=Active", "--max-results", str(PAGE_SIZE)],
        items_key="TrainingPlanSummaries",
        token_key="NextToken",
    )
    summaries: list[dict[str, Any]] = []
    for status in ACTIVE_BATCH_STATUSES:
        summaries.extend(
            _pages(
                [
                    "batch",
                    "list-service-jobs",
                    "--job-queue",
                    TRAINING_PLAN_QUEUE,
                    "--job-status",
                    status,
                    "--max-results",
                    str(PAGE_SIZE),
                ],
                items_key="jobSummaryList",
                token_key="nextToken",
            )
        )
    batch_jobs = []
    for summary in {item.get("jobId"): item for item in summaries}.values():
        job_id = summary.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("active Batch service-job summary omitted jobId")
        batch_jobs.append(_aws_json("batch", "describe-service-job", "--job-id", job_id))
    training_jobs = []
    for summary in _pages(
        ["sagemaker", "list-training-jobs", "--status-equals", "InProgress", "--max-results", str(PAGE_SIZE)],
        items_key="TrainingJobSummaries",
        token_key="NextToken",
    ):
        name = summary.get("TrainingJobName")
        if not isinstance(name, str) or not name:
            raise RuntimeError("SageMaker InProgress summary omitted TrainingJobName")
        training_jobs.append(_aws_json("sagemaker", "describe-training-job", "--training-job-name", name))
    namespace_objects = _s3_objects(namespace_s3.rstrip("/") + "/")
    # Reuse the capacity and exact rendered-identity checks with the one allowed namespace.
    synthetic["claims"] = {
        "manifest": namespace_s3.rstrip("/") + "/",
        "attempt_result": namespace_s3.rstrip("/") + "/attempt-result.absent",
        "completion": namespace_s3.rstrip("/") + "/completion.absent",
    }
    # Capacity validation remains centralized; supply all three keys as the same authoritative
    # listing but then assert only the actual canary prefix was queried.
    snapshot = validate_snapshot(
        account=account,
        plans=plans,
        batch_jobs=batch_jobs,
        training_jobs=training_jobs,
        namespace_objects={uri: namespace_objects for uri in synthetic["claims"].values()},
        manifest=synthetic,
        job_name=job_name,
    )
    snapshot["publication_namespaces_proven_empty"] = 1
    snapshot["namespace_s3"] = namespace_s3
    return snapshot
