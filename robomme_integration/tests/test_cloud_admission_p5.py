from __future__ import annotations

import copy

import pytest

from robomme_integration import cloud_admission_p5 as admission
from scripts.launch.launch_guardrails import EXECUTION_ACCOUNT


def _quota(value: float = 10.0) -> dict:
    return {
        "QuotaCode": admission.P5_QUOTA_CODE,
        "QuotaName": admission.P5_QUOTA_NAME,
        "Value": value,
    }


def _training(name: str, *, instance_type: str = admission.P5_INSTANCE_TYPE, count: int = 1) -> dict:
    return {
        "TrainingJobName": name,
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": count},
    }


def _batch(name: str, *, job_id: str = "job-1", quantity: int = 1) -> dict:
    return {
        "jobId": job_id,
        "jobName": name,
        "jobQueue": admission.P5_QUEUE_ARN,
        "serviceJobType": "SAGEMAKER_TRAINING",
        "status": "RUNNING",
        "capacityUsage": [{"capacityUnit": admission.P5_INSTANCE_TYPE, "quantity": quantity}],
        "latestAttempt": {
            "serviceResourceId": {
                "name": "TrainingJobArn",
                "value": ("arn:aws:sagemaker:us-west-2:141701954645:training-job/" + name),
            }
        },
        "serviceRequestPayload": "{}",
    }


def _statuses() -> dict[str, list[dict]]:
    return {status: [] for status in admission.ACTIVE_BATCH_STATUSES}


def _validate(*, statuses=None, training=None, quota=None, allow_backlog=False):
    return admission.validate_snapshot(
        account=EXECUTION_ACCOUNT,
        batch_jobs_by_status=statuses or _statuses(),
        training_jobs=training or [],
        namespace_objects={"s3://bucket/exact/": []},
        quota=quota or _quota(),
        identity_tokens={"candidate-id"},
        expected_namespaces={"s3://bucket/exact/"},
        identity={"canary_id": "candidate-id", "job_name": "candidate-job"},
        allow_backlog=allow_backlog,
    )


def test_admits_only_when_zero_waiting_and_global_quota_has_a_slot():
    receipt = _validate(training=[_training(f"other-{i}") for i in range(9)])
    assert receipt["p5_instance_quota"] == 10
    assert receipt["global_running_p5_instance_units"] == 9
    assert receipt["available_p5_instance_units"] == 1
    assert receipt["training_plan_arn"] is None


def test_waiting_job_or_global_outside_queue_saturation_rejects():
    statuses = _statuses()
    statuses["RUNNABLE"] = [{"safe": "description"}]
    with pytest.raises(ValueError, match="waiting work"):
        _validate(statuses=statuses)
    with pytest.raises(ValueError, match="lacks a free node"):
        _validate(training=[_training(f"outside-{i}") for i in range(10)])


def test_explicit_backlog_policy_allows_waiting_and_zero_capacity_only():
    statuses = _statuses()
    statuses["RUNNABLE"] = [{"jobName": "unrelated-waiter"}]
    receipt = _validate(
        statuses=statuses,
        training=[_training(f"outside-{i}") for i in range(10)],
        allow_backlog=True,
    )
    assert receipt["admitted"] is True
    assert receipt["available_p5_instance_units"] == 0
    assert receipt["scheduling_policy"] == "user_authorized_backlog"
    assert receipt["waiting_batch_statuses_required_zero"] == []
    assert receipt["waiting_batch_counts"]["RUNNABLE"] == 1


def test_backlog_policy_keeps_duplicate_and_namespace_gates_strict():
    statuses = _statuses()
    statuses["RUNNABLE"] = [{"jobName": "candidate-id"}]
    with pytest.raises(ValueError, match="active duplicate"):
        _validate(statuses=statuses, allow_backlog=True)
    with pytest.raises(ValueError, match="namespace is nonempty"):
        admission.validate_snapshot(
            account=EXECUTION_ACCOUNT,
            batch_jobs_by_status=_statuses(),
            training_jobs=[_training(f"outside-{i}") for i in range(10)],
            namespace_objects={"s3://bucket/exact/": [{"Key": "already-there"}]},
            quota=_quota(),
            identity_tokens={"candidate-id"},
            expected_namespaces={"s3://bucket/exact/"},
            identity={"canary_id": "candidate-id", "job_name": "candidate-job"},
            allow_backlog=True,
        )


def test_target_queue_running_reconciles_to_global_sagemaker_without_double_counting():
    statuses = _statuses()
    statuses["RUNNING"] = [_batch("AWSBatchcandidate")]
    receipt = _validate(
        statuses=statuses,
        training=[_training("AWSBatchcandidate"), *[_training(f"outside-{i}") for i in range(8)]],
    )
    assert receipt["target_queue_running_p5_instance_units"] == 1
    assert receipt["global_running_p5_instance_units"] == 9
    broken = copy.deepcopy(statuses)
    broken["RUNNING"][0]["latestAttempt"] = {}
    with pytest.raises(RuntimeError, match="TrainingJobArn"):
        _validate(statuses=broken, training=[_training("AWSBatchcandidate")])


def test_quota_identity_and_value_fail_closed():
    for mutated in (
        {**_quota(), "QuotaCode": "wrong"},
        {**_quota(), "QuotaName": "wrong"},
        {**_quota(), "Value": "unknown"},
        {**_quota(), "Value": 10.5},
    ):
        with pytest.raises(ValueError, match="Quota"):
            _validate(quota=mutated)


def test_fractional_or_boolean_sagemaker_instance_count_rejects():
    for count in (1.5, True):
        with pytest.raises(RuntimeError, match="positive integer"):
            _validate(training=[_training("outside", count=count)])


def test_empty_s3_real_cli_shape_without_is_truncated_is_terminal(monkeypatch):
    monkeypatch.setattr(admission, "_aws_json", lambda *_args: {"KeyCount": 0})
    assert admission._s3_objects("s3://bucket/empty/") == []
    monkeypatch.setattr(
        admission,
        "_aws_json",
        lambda *_args: {"Contents": [{"Key": "x"}]},
    )
    with pytest.raises(RuntimeError, match="IsTruncated"):
        admission._s3_objects("s3://bucket/not-empty/")


def test_collect_batch_binds_summary_description_and_redacts_payload(monkeypatch):
    summary = {
        "jobId": "job-1",
        "jobName": "safe-name",
        "status": "RUNNING",
        "capacityUsage": [{"capacityUnit": admission.P5_INSTANCE_TYPE, "quantity": 1.0}],
    }
    described = _batch("AWSBatchsafe", job_id="job-1")
    described["jobName"] = "safe-name"
    described["serviceRequestPayload"] = '{"Environment":{"HF_TOKEN":"secret"}}'

    def fake(*args):
        if args[:2] == ("batch", "list-service-jobs"):
            status = args[args.index("--job-status") + 1]
            return {"jobSummaryList": [summary] if status == "RUNNING" else []}
        if args[:2] == ("batch", "describe-service-job"):
            return copy.deepcopy(described)
        raise AssertionError(args)

    monkeypatch.setattr(admission, "_aws_json", fake)
    assert admission._collect_batch_jobs()["RUNNING"][0]["jobId"] == "job-1"
    bad = copy.deepcopy(described)
    bad["status"] = "RUNNABLE"
    monkeypatch.setattr(
        admission,
        "_aws_json",
        lambda *args: (
            {"jobSummaryList": [summary] if args[args.index("--job-status") + 1] == "RUNNING" else []}
            if args[:2] == ("batch", "list-service-jobs")
            else bad
        ),
    )
    with pytest.raises(RuntimeError) as error:
        admission._collect_batch_jobs()
    assert "secret" not in str(error.value)
