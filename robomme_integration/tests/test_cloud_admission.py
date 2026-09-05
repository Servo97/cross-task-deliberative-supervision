from __future__ import annotations

import copy
import json
import sys

import pytest

from robomme_integration import campaign_launch, cloud_admission
from robomme_integration.cloud_admission import campaign_namespaces, collect_and_validate, validate_snapshot
from scripts.launch.launch_guardrails import EXECUTION_ACCOUNT, TRAINING_PLAN_QUEUE, training_plan_arn


def _manifest() -> dict:
    root = "s3://bucket/study"
    environment = {
        "RUN_MANIFEST_S3": f"{root}/runs/r1/a1.json",
        "PRODUCER_CLAIM_S3": f"{root}/claims/r1/producer.json",
        "COMPLETION_CLAIM_S3": f"{root}/claims/r1/complete.json",
        "OUTPUT_S3": f"{root}/checkpoints/r1",
    }
    return {
        "campaign_id": "campaign-abc",
        "attempt_id": "campaign-abc-attempt1",
        "claims": {
            "manifest": f"{root}/campaign/manifest.json",
            "attempt_result": f"{root}/campaign/result.json",
            "completion": f"{root}/campaign/complete.json",
        },
        "cells": [{"run_id": "r1", "environment": environment}],
    }


def _plan(**updates):
    value = {
        "TrainingPlanArn": training_plan_arn(TRAINING_PLAN_QUEUE),
        "Status": "Active",
        "TotalInstanceCount": 2,
        "InUseInstanceCount": 1,
        "AvailableInstanceCount": 1,
        "UnhealthyInstanceCount": 0,
    }
    value.update(updates)
    return value


def _validate(**updates):
    manifest = _manifest()
    arguments = {
        "account": EXECUTION_ACCOUNT,
        "plans": [_plan()],
        "batch_jobs": [],
        "training_jobs": [],
        "namespace_objects": {uri: [] for uri in campaign_namespaces(manifest)},
        "manifest": manifest,
        "job_name": "job-xyz",
    }
    arguments.update(updates)
    return validate_snapshot(**arguments)


def test_live_admission_green_is_read_only_receipt():
    receipt = _validate()
    assert receipt["admitted"] is True
    assert receipt["cloud_action"] is False
    assert receipt["available_instance_count"] == 1
    assert receipt["unhealthy_instance_count"] == 0
    assert receipt["batch_statuses_scanned"] == [
        "SUBMITTED",
        "PENDING",
        "RUNNABLE",
        "SCHEDULED",
        "STARTING",
        "RUNNING",
    ]


def test_missing_plan_health_is_explicitly_unverified_never_fabricated_zero():
    plan = _plan()
    plan.pop("UnhealthyInstanceCount")
    receipt = _validate(plans=[plan])
    assert receipt["admitted"] is True
    assert receipt["unhealthy_instance_count"] is None
    assert receipt["unhealthy_instance_count_observation"] == ("not_returned_by_list_training_plans_not_claimed")


@pytest.mark.parametrize(
    "plan",
    [
        _plan(Status="Pending"),
        _plan(UnhealthyInstanceCount=1),
        _plan(AvailableInstanceCount=0, InUseInstanceCount=2),
        _plan(TotalInstanceCount=3),
        _plan(TrainingPlanArn="arn:wrong"),
    ],
)
def test_live_admission_rejects_capacity_and_identity_drift(plan):
    with pytest.raises(ValueError):
        _validate(plans=[plan])


def test_live_admission_rejects_batch_or_sagemaker_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        _validate(batch_jobs=[{"jobName": "prefix-campaign-abc-suffix"}])
    with pytest.raises(ValueError, match="duplicate"):
        _validate(training_jobs=[{"Environment": {"ROBOMME_RUN_ID": "r1"}}])


def test_live_admission_rejects_nonempty_or_incomplete_namespace_snapshot():
    manifest = _manifest()
    objects = {uri: [] for uri in campaign_namespaces(manifest)}
    first = next(iter(objects))
    objects[first] = [{"Key": "collision"}]
    with pytest.raises(ValueError, match="nonempty"):
        _validate(namespace_objects=objects)
    incomplete = copy.deepcopy(objects)
    incomplete[first] = []
    incomplete.pop(first)
    with pytest.raises(ValueError, match="every exact"):
        _validate(namespace_objects=incomplete)


def test_collect_fully_paginates_all_surfaces_and_describes_training_jobs(monkeypatch):
    manifest = _manifest()
    calls: list[tuple[str, ...]] = []

    def fake(*args):
        calls.append(args)
        if args[:2] == ("sts", "get-caller-identity"):
            return {"Account": EXECUTION_ACCOUNT}
        if args[:2] == ("sagemaker", "list-training-plans"):
            if "--next-token" in args:
                return {"TrainingPlanSummaries": [_plan()]}
            return {"TrainingPlanSummaries": [], "NextToken": "plans-page-2"}
        if args[:2] == ("batch", "list-service-jobs"):
            status = args[args.index("--job-status") + 1]
            if status == "RUNNABLE" and "--next-token" not in args:
                return {"jobSummaryList": [], "nextToken": "batch-page-2"}
            return {"jobSummaryList": []}
        if args[:2] == ("batch", "describe-service-job"):
            return {"jobId": args[-1], "serviceRequestPayload": "{}"}
        if args[:2] == ("sagemaker", "list-training-jobs"):
            if "--next-token" in args:
                return {"TrainingJobSummaries": [{"TrainingJobName": "active-b"}]}
            return {
                "TrainingJobSummaries": [{"TrainingJobName": "active-a"}],
                "NextToken": "training-page-2",
            }
        if args[:2] == ("sagemaker", "describe-training-job"):
            return {"TrainingJobName": args[-1], "Environment": {"UNRELATED": "1"}}
        if args[:2] == ("s3api", "list-objects-v2"):
            prefix = args[args.index("--prefix") + 1]
            if prefix.endswith("campaign/manifest.json") and "--continuation-token" not in args:
                return {"Contents": [], "IsTruncated": True, "NextContinuationToken": "s3-page-2"}
            return {"Contents": [], "IsTruncated": False}
        raise AssertionError(args)

    monkeypatch.setattr(cloud_admission, "_aws_json", fake)
    receipt = collect_and_validate(manifest, job_name="job-xyz")
    assert receipt["admitted"] is True
    for status in cloud_admission.ACTIVE_BATCH_STATUSES:
        assert any(
            call[:2] == ("batch", "list-service-jobs") and call[call.index("--job-status") + 1] == status
            for call in calls
        )
    assert any("plans-page-2" in call for call in calls)
    assert any("batch-page-2" in call for call in calls)
    assert any("training-page-2" in call for call in calls)
    assert any("s3-page-2" in call for call in calls)
    assert {call[-1] for call in calls if call[:2] == ("sagemaker", "describe-training-job")} == {
        "active-a",
        "active-b",
    }


def test_collect_rejects_duplicate_visible_only_in_described_batch_payload(monkeypatch):
    manifest = _manifest()

    def fake(*args):
        if args[:2] == ("sts", "get-caller-identity"):
            return {"Account": EXECUTION_ACCOUNT}
        if args[:2] == ("sagemaker", "list-training-plans"):
            plan = _plan()
            plan.pop("UnhealthyInstanceCount")
            return {"TrainingPlanSummaries": [plan]}
        if args[:2] == ("batch", "list-service-jobs"):
            status = args[args.index("--job-status") + 1]
            if status == "RUNNABLE":
                return {"jobSummaryList": [{"jobId": "hidden-duplicate", "jobName": "opaque"}]}
            return {"jobSummaryList": []}
        if args[:2] == ("batch", "describe-service-job"):
            return {
                "jobId": args[-1],
                "jobName": "opaque",
                "serviceRequestPayload": json.dumps({"Environment": {"ROBOMME_CAMPAIGN_ID": "campaign-abc"}}),
            }
        if args[:2] == ("sagemaker", "list-training-jobs"):
            return {"TrainingJobSummaries": []}
        if args[:2] == ("s3api", "list-objects-v2"):
            return {"Contents": [], "IsTruncated": False}
        raise AssertionError(args)

    monkeypatch.setattr(cloud_admission, "_aws_json", fake)
    with pytest.raises(ValueError, match="duplicate"):
        collect_and_validate(manifest, job_name="job-xyz")


def test_collect_rejects_active_batch_summary_without_describable_id(monkeypatch):
    manifest = _manifest()

    def fake(*args):
        if args[:2] == ("sts", "get-caller-identity"):
            return {"Account": EXECUTION_ACCOUNT}
        if args[:2] == ("sagemaker", "list-training-plans"):
            return {"TrainingPlanSummaries": [_plan()]}
        if args[:2] == ("batch", "list-service-jobs"):
            status = args[args.index("--job-status") + 1]
            return {"jobSummaryList": [{}] if status == "SUBMITTED" else []}
        raise AssertionError(args)

    monkeypatch.setattr(cloud_admission, "_aws_json", fake)
    with pytest.raises(RuntimeError, match="omitted jobId"):
        collect_and_validate(manifest, job_name="job-xyz")


@pytest.mark.parametrize(
    "response",
    [
        {"items": [], "token": ""},
        {"wrong": []},
    ],
)
def test_generic_pagination_rejects_malformed_pages_and_tokens(monkeypatch, response):
    monkeypatch.setattr(cloud_admission, "_aws_json", lambda *_args: response)
    with pytest.raises(RuntimeError):
        cloud_admission._pages(["service", "op"], items_key="items", token_key="token")


def test_s3_pagination_rejects_malformed_continuation(monkeypatch):
    monkeypatch.setattr(
        cloud_admission,
        "_aws_json",
        lambda *_args: {"Contents": [], "IsTruncated": True, "NextContinuationToken": ""},
    )
    with pytest.raises(RuntimeError, match="NextContinuationToken"):
        cloud_admission._s3_objects("s3://bucket/prefix")


def test_campaign_submit_calls_admission_immediately_before_submit(monkeypatch, tmp_path, capsys):
    order: list[str] = []
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    manifest = {
        "campaign_id": "campaign-abc",
        "attempt_id": "campaign-abc-attempt1",
        "manifest_sha256": "a" * 64,
        "infrastructure": {
            "hardware": "p5e",
            "priority": 400,
            "max_run_seconds": 86400,
            "volume_size_gb": 400,
        },
        "evaluation": {"mode": "deferred"},
        "cells": [
            {
                "task": "PickXtimes",
                "arm": "v4_s0",
                "run_id": "st-v4-pick-v4_s0-seed0-abc",
                "completion_claim_s3": "s3://bucket/complete",
            }
        ],
    }
    plan = {
        "campaign_id": manifest["campaign_id"],
        "attempt_id": manifest["attempt_id"],
        "manifest": manifest,
        "entry": "gpu_campaign_entry.sh",
        "environment": {},
        "source_sha": "b" * 64,
        "staged_source_files": {},
    }
    monkeypatch.setattr(campaign_launch, "_load_spec", lambda _path: {})
    monkeypatch.setattr(campaign_launch, "build_campaign_plan", lambda _spec, _source: plan)

    def admit(_manifest, *, job_name):
        order.append("admission")
        return {"admitted": True, "job_name": job_name}

    def submit(**_kwargs):
        order.append("submit")
        return []

    monkeypatch.setattr(campaign_launch, "collect_and_validate", admit)
    monkeypatch.setattr(campaign_launch.launch, "submit_training_job", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        ["campaign_launch", "--spec", str(spec), "--source-dir", str(tmp_path), "--confirm-submit"],
    )
    campaign_launch.main()
    assert order == ["admission", "submit"]


def test_campaign_admission_failure_prevents_submit(monkeypatch, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    manifest = {
        "campaign_id": "campaign-abc",
        "attempt_id": "campaign-abc-attempt1",
        "manifest_sha256": "a" * 64,
        "infrastructure": {"hardware": "p5e", "priority": 400, "max_run_seconds": 86400, "volume_size_gb": 400},
        "evaluation": {},
        "cells": [{"task": "T", "arm": "A", "run_id": "r", "completion_claim_s3": "s3://b/c"}],
    }
    plan = {
        "campaign_id": "campaign-abc",
        "attempt_id": "campaign-abc-attempt1",
        "manifest": manifest,
        "entry": "e",
        "environment": {},
        "source_sha": "b" * 64,
        "staged_source_files": {},
    }
    monkeypatch.setattr(campaign_launch, "_load_spec", lambda _path: {})
    monkeypatch.setattr(campaign_launch, "build_campaign_plan", lambda _spec, _source: plan)
    monkeypatch.setattr(
        campaign_launch,
        "collect_and_validate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("AWS failed")),
    )
    submitted = False

    def submit(**_kwargs):
        nonlocal submitted
        submitted = True

    monkeypatch.setattr(campaign_launch.launch, "submit_training_job", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        ["campaign_launch", "--spec", str(spec), "--source-dir", str(tmp_path), "--confirm-submit"],
    )
    with pytest.raises(RuntimeError, match="AWS failed"):
        campaign_launch.main()
    assert submitted is False
