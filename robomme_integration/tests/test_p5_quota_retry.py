from __future__ import annotations

import json

import pytest

from robomme_integration.sweeps.replay_p5_quota_failures import (
    DEFAULT_SPEC,
    QUOTA_REASON,
    GuardrailError,
    available_slots,
    build_replay_request,
    read_spec,
    scientific_payload_projection,
    valid_training_name,
    validate_source_job,
)


def source_record() -> dict:
    job_id = "11111111-2222-3333-4444-555555555555"
    name = "sarvesh-rmme-StopCube-q2-deadbeef-0806-120000"
    environment = {
        "ROBOMME_RUN_ID": "st-v1-stopcube-q2-seed0-deadbeef",
        "ROBOMME_ATTEMPT_ID": "st-v1-stopcube-q2-seed0-deadbeef-attempt1",
        "ROBOMME_TASK": "StopCube",
        "ROBOMME_ARM": "q2",
        "RUN_MANIFEST_S3": "s3://bucket/manifests/run.json",
        "PRODUCER_CLAIM_S3": "s3://bucket/claims/producer.json",
        "COMPLETION_CLAIM_S3": "s3://bucket/claims/complete.json",
        "OUTPUT_S3": "s3://bucket/checkpoints/run",
        "RUN_MANIFEST_SHA256": "a" * 64,
    }
    payload = {
        "TrainingJobName": name,
        "ResourceConfig": {
            "InstanceType": "ml.p5.48xlarge",
            "InstanceCount": 1,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "HyperParameters": {"sagemaker_submit_directory": json.dumps("s3://bucket/source/sourcedir.tar.gz")},
        "Environment": environment,
    }
    return {
        "jobId": job_id,
        "jobName": name,
        "jobQueue": ("arn:aws:batch:us-west-2:141701954645:job-queue/fss-tri-cam-robotics-p5-48xlarge-us-west-2"),
        "serviceJobType": "SAGEMAKER_TRAINING",
        "schedulingPriority": 1,
        "retryStrategy": {"attempts": 1, "evaluateOnExit": []},
        "timeoutConfig": {"attemptDurationSeconds": 86400},
        "shareIdentifier": "default",
        "serviceRequestPayload": json.dumps(payload, sort_keys=True),
        "tags": {
            "wsm.run_id": environment["ROBOMME_RUN_ID"],
            "wsm.task": environment["ROBOMME_TASK"],
            "wsm.arm": environment["ROBOMME_ARM"],
        },
        "status": "FAILED",
        "statusReason": f"Received status from SageMaker: {QUOTA_REASON} is 10 Instances.",
        "attempts": [],
    }


def test_retry_inventory_and_capacity_boundary_are_frozen():
    spec = read_spec(DEFAULT_SPEC)
    assert len(spec["source_job_ids"]) == len(set(spec["source_job_ids"])) == 44
    assert spec["expected_terminal_counts"] == {"SUCCEEDED": 9, "QUOTA_FAILED": 35}
    assert available_slots(0, 10, 1) == 9
    assert available_slots(8, 10, 1) == 1
    assert available_slots(9, 10, 1) == 0
    with pytest.raises(GuardrailError):
        available_slots(0, 10, 10)


def test_replay_preserves_scientific_payload_and_rejects_non_admission_failures():
    record = source_record()
    validated = validate_source_job(record, expected_job_id=record["jobId"])
    request = build_replay_request(record, generation=1)

    assert validated["status"] == "FAILED"
    assert request["serviceRequestPayload"] is record["serviceRequestPayload"]
    assert json.loads(request["serviceRequestPayload"]) == json.loads(record["serviceRequestPayload"])
    assert request["jobName"].endswith("-qr1")
    assert request["schedulingPriority"] == 1
    assert request["retryStrategy"] == record["retryStrategy"]
    assert request["tags"]["wsm.replay_of"] == record["jobId"]
    assert len(request["clientToken"]) == 64
    assert request == build_replay_request(record, generation=1)

    promoted = build_replay_request(record, generation=1, scheduling_priority=400)
    assert promoted["schedulingPriority"] == 400
    assert promoted["tags"]["wsm.replay_priority"] == "400"
    assert promoted["retryStrategy"] == record["retryStrategy"]
    assert json.loads(promoted["serviceRequestPayload"]) == json.loads(record["serviceRequestPayload"])
    assert promoted["clientToken"] != request["clientToken"]
    with pytest.raises(GuardrailError, match="unsupported replay scheduling priority"):
        build_replay_request(record, generation=1, scheduling_priority=600)

    attempted = {**record, "attempts": [{"serviceJobArn": "arn:attempt"}]}
    with pytest.raises(GuardrailError, match="unexpectedly has attempts"):
        validate_source_job(attempted, expected_job_id=record["jobId"])

    unrelated = {**record, "statusReason": "training container exited 1"}
    with pytest.raises(GuardrailError, match="not the approved quota rejection"):
        validate_source_job(unrelated, expected_job_id=record["jobId"])

    legacy = source_record()
    legacy_name = legacy["jobName"].replace("q2", "wsm_cfg")
    legacy["jobName"] = legacy_name
    legacy_payload = json.loads(legacy["serviceRequestPayload"])
    legacy_payload["TrainingJobName"] = legacy_name
    legacy_payload["HyperParameters"]["sagemaker_job_name"] = json.dumps(legacy_name)
    legacy["serviceRequestPayload"] = json.dumps(legacy_payload, sort_keys=True)
    repaired = build_replay_request(legacy, generation=1)
    repaired_payload = json.loads(repaired["serviceRequestPayload"])
    assert repaired_payload["TrainingJobName"] == valid_training_name(legacy_name)
    assert "_" not in repaired_payload["TrainingJobName"]
    assert scientific_payload_projection(repaired_payload) == scientific_payload_projection(legacy_payload)
