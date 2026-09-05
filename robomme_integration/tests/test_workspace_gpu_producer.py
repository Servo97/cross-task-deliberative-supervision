from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from robomme_integration import workspace_launch
from robomme_integration.training import workspace_gpu_producer
from robomme_integration.training.single_task import TASK_ORDER


def _args(pair_index: int, hardware: str = "p5e", **overrides) -> argparse.Namespace:
    resource = workspace_launch.HARDWARE[hardware]
    values = {
        "pair_index": pair_index,
        "hardware": hardware,
        "queue": resource["queue"],
        "role": workspace_launch.ROLE_ARN,
        "priority": resource["priority"],
        "max_run_seconds": 86_400,
        "volume_size_gb": 400,
        "attempt_index": 1,
        "secrets_manager_arn": None,
        "dry_run": True,
        "confirm_submit": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_all_pair_plans_cover_all16_once_with_one_frozen_source_identity() -> None:
    source = Path(workspace_launch.__file__).resolve().parent
    plans = [workspace_launch.build_plan(_args(index), source) for index in range(8)]
    assert tuple(task for plan in plans for task in plan["tasks"]) == TASK_ORDER
    assert len({plan["pair_id"] for plan in plans}) == 8
    assert len({plan["source_sha"] for plan in plans}) == 1
    for plan in plans:
        manifest = json.loads(plan["manifest_json"])
        assert manifest["infrastructure"] == {
            "provider": "aws_sagemaker",
            "account": workspace_launch.EXECUTION_ACCOUNT,
            "queue": workspace_launch.TRAINING_PLAN_QUEUE,
            "training_plan_arn": workspace_launch.PLAN_ARN,
            "instance_type": "ml.p5e.48xlarge",
            "priority": 400,
            "max_run_seconds": 86_400,
            "volume_size_gb": 400,
            "attempt_index": 1,
        }
        assert [record["task"] for record in manifest["tasks"]] == list(plan["tasks"])
        assert all(record["scientific"]["representation"]["devices"] == 4 for record in manifest["tasks"])


def test_p5_backfill_has_a_distinct_sealed_hardware_identity() -> None:
    source = Path(workspace_launch.__file__).resolve().parent
    p5e = workspace_launch.build_plan(_args(1), source)
    p5 = workspace_launch.build_plan(_args(1, hardware="p5"), source)
    assert p5["source_sha"] == p5e["source_sha"]
    assert p5["pair_id"] != p5e["pair_id"]
    manifest = json.loads(p5["manifest_json"])
    assert manifest["infrastructure"]["instance_type"] == "ml.p5.48xlarge"
    assert manifest["infrastructure"]["priority"] == 1
    assert manifest["infrastructure"]["training_plan_arn"] is None
    assert manifest["pair_scientific"]["topology"]["node"] == "8xH100"
    assert all(record["scientific"]["producer"]["hardware"] == "p5" for record in manifest["tasks"])


@pytest.mark.parametrize(
    ("field", "value"),
    (("priority", 1), ("max_run_seconds", 86_399), ("volume_size_gb", 100)),
)
def test_workspace_production_rejects_resource_contract_drift(field: str, value: object) -> None:
    with pytest.raises(SystemExit):
        workspace_launch._validate(_args(0, **{field: value}))


def test_control_and_compute_interpreters_are_explicitly_separated(monkeypatch) -> None:
    monkeypatch.delenv("ROBOMME_WORKSPACE_COMPUTE_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="COMPUTE_PYTHON is required"):
        workspace_gpu_producer._compute_python()
    monkeypatch.setenv("ROBOMME_WORKSPACE_COMPUTE_PYTHON", sys.executable)
    assert workspace_gpu_producer._compute_python() == sys.executable

    entry = Path(workspace_launch.__file__).with_name("workspace_gpu_entry.sh").read_text()
    assert '"$CONTROL_PY" -c \'import boto3, botocore' in entry
    assert '"$CONTROL_PY" -m training.workspace_gpu_producer pair' in entry
    assert 'export ROBOMME_WORKSPACE_COMPUTE_PYTHON="$PY"' in entry
