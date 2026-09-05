"""Offline tests for the shared safety invariants of every submit_*.py launcher."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_LAUNCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "launch"
sys.path.insert(0, str(_LAUNCH_DIR))

import launch_guardrails as guardrails

_LAUNCHER = _LAUNCH_DIR / "submit_evals.py"
_SPEC = importlib.util.spec_from_file_location("submit_evals_guardrails", _LAUNCHER)
assert _SPEC is not None and _SPEC.loader is not None
eval_launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_launcher)
launcher = guardrails


def test_cam_robotics_defaults_and_storage_are_separate():
    # Storage moved to account 141 on 2026-07-22 (see launch_guardrails). Storage == execution now.
    assert launcher.EXECUTION_ACCOUNT == "141701954645"
    assert launcher.STORAGE_ACCOUNT == "141701954645"
    assert launcher.QUEUE == "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
    assert launcher.ROLE_ARN == ("arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2")
    assert launcher.DEFAULT_RESULTS_BUCKET == "sagemaker-us-west-2-141701954645"
    assert launcher.MULTI_DAY_PRIORITY == 600
    assert launcher.MAX_RUN_SECONDS == 432000


def test_launch_contract_accepts_only_the_required_multiday_target():
    launcher.validate_launch_contract(
        queue=launcher.QUEUE,
        role=launcher.ROLE_ARN,
        priority=600,
        max_run_seconds=432000,
    )
    with pytest.raises(SystemExit, match="cam-robotics queue"):
        launcher.validate_launch_contract(
            queue="fss-tri-cam-humanoid-p5-48xlarge-us-west-2",
            role=launcher.ROLE_ARN,
            priority=600,
            max_run_seconds=432000,
        )
    with pytest.raises(SystemExit, match="must use execution role"):
        launcher.validate_launch_contract(
            queue=launcher.QUEUE,
            role="arn:aws:iam::124224456861:role/incorrect",
            priority=600,
            max_run_seconds=432000,
        )


def test_timeout_cap_and_multiday_priority_are_enforced():
    with pytest.raises(SystemExit, match=r"\[1, 432000\]"):
        launcher.validate_launch_contract(
            queue=launcher.QUEUE,
            role=launcher.ROLE_ARN,
            priority=600,
            max_run_seconds=432001,
        )
    with pytest.raises(SystemExit, match="must use --priority 600"):
        launcher.validate_launch_contract(
            queue=launcher.QUEUE,
            role=launcher.ROLE_ARN,
            priority=1,
            max_run_seconds=86401,
        )


def test_two_day_standard_class_is_admitted_only_at_priority_400_up_to_48h():
    # User directive 2026-09-02: 48 h runs at priority 400; anything longer still needs 600.
    assert launcher.STANDARD_PRIORITY == 400
    assert launcher.STANDARD_TWO_DAY_MAX_SECONDS == 172800
    launcher.validate_launch_contract(
        queue=launcher.QUEUE,
        role=launcher.ROLE_ARN,
        priority=400,
        max_run_seconds=172800,
    )
    with pytest.raises(SystemExit, match="must use --priority 600"):
        launcher.validate_launch_contract(
            queue=launcher.QUEUE,
            role=launcher.ROLE_ARN,
            priority=400,
            max_run_seconds=172801,
        )
    with pytest.raises(SystemExit, match="must use --priority 600"):
        launcher.validate_launch_contract(
            queue=launcher.QUEUE,
            role=launcher.ROLE_ARN,
            priority=200,
            max_run_seconds=172800,
        )
    launcher.validate_launch_contract(
        queue=launcher.QUEUE,
        role=launcher.ROLE_ARN,
        priority=1,
        max_run_seconds=86400,
    )


def test_submission_requires_confirmation_but_dry_run_does_not():
    launcher.require_submission_confirmation(dry_run=True, confirmed=False)
    with pytest.raises(SystemExit, match="explicit user approval"):
        launcher.require_submission_confirmation(dry_run=False, confirmed=False)
    launcher.require_submission_confirmation(dry_run=False, confirmed=True)


def test_training_submission_disables_the_sagemaker_debugger_hook(monkeypatch, tmp_path):
    """The profiler flag does not disable Debugger's separately injected hook config."""
    entry = "train.sh"
    (tmp_path / entry).write_text("#!/bin/sh\n", encoding="utf-8")
    captured = {}

    class FakeBotoSession:
        @staticmethod
        def Session(*, region_name):
            return {"region_name": region_name}

    class FakeBoto3:
        session = FakeBotoSession

    class FakeSageMaker:
        @staticmethod
        def Session(*, boto_session):
            return {"boto_session": boto_session}

    class FakeEstimator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeTrainingQueue:
        def __init__(self, *, queue_name):
            self.queue_name = queue_name

        def map(self, estimator, **kwargs):
            captured["training_queue"] = self.queue_name
            captured["map"] = kwargs
            return []

    monkeypatch.setattr(
        launcher,
        "load_aws_sdk",
        lambda: (FakeBoto3, FakeSageMaker, FakeEstimator, FakeTrainingQueue),
    )
    monkeypatch.setattr(launcher, "validate_caller_account", lambda _boto3: None)
    monkeypatch.setattr(launcher, "training_plan_arn", lambda _queue: None)

    launcher.submit_training_job(
        entry=entry,
        source_dir=tmp_path,
        environment={"SM_USE_RESERVED_CAPACITY": "1"},
        image_uri="example.invalid/image",
        instance_type="ml.p5.48xlarge",
        volume_size=400,
        tags=[],
        retry_config={"attempts": 1},
        job_name="debugger-hook-disabled",
        queue=launcher.QUEUE,
        role=launcher.ROLE_ARN,
        priority=400,
        max_run_seconds=10_800,
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
    )

    assert captured["disable_profiler"] is True
    assert captured["debugger_hook_config"] is False
    assert captured["training_queue"] == launcher.QUEUE


def test_sanitized_source_bundle_excludes_plaintext_credentials(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = "robocasa_eval_entry.sh"
    (source / entry).write_text("#!/bin/sh\n", encoding="utf-8")
    (source / "worker.py").write_text("pass\n", encoding="utf-8")
    (source / "secrets.env").write_text("SECRET=do-not-copy\n", encoding="utf-8")
    (source / "gsheets_sa.json").write_text('{"private_key": "do-not-copy"}\n', encoding="utf-8")
    (source / "service_account_secret.json").write_text('{"private_key": "do-not-copy"}\n', encoding="utf-8")
    (source / ".env.local").write_text("SECRET=do-not-copy\n", encoding="utf-8")
    (source / "client.pem").write_text("do-not-copy\n", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "huge.bin").write_text("do-not-copy\n", encoding="utf-8")
    (source / "wandb").mkdir()
    (source / "wandb" / "debug.log").write_text("do-not-copy\n", encoding="utf-8")

    with launcher.prepared_source_bundle(source, entry, {"SAGEMAKER_PROGRAM": entry}) as (
        staged,
        safe_entry,
        safe_environment,
    ):
        assert safe_entry == entry
        assert safe_environment["WANDB_MODE"] == "disabled"
        assert (staged / entry).is_file()
        assert (staged / "worker.py").is_file()
        assert not (staged / "secrets.env").exists()
        assert not (staged / "gsheets_sa.json").exists()
        assert not (staged / "service_account_secret.json").exists()
        assert not (staged / ".env.local").exists()
        assert not (staged / "client.pem").exists()
        assert not (staged / ".venv").exists()
        assert not (staged / "wandb").exists()


def test_secrets_manager_reference_uses_on_node_wrapper(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = "train.sh"
    (source / entry).write_text("#!/bin/sh\n", encoding="utf-8")
    (source / "secrets.env").write_text("HF_TOKEN=plaintext\n", encoding="utf-8")
    secret_arn = "arn:aws:secretsmanager:us-west-2:141701954645:secret:wsm-launch"

    with launcher.prepared_source_bundle(source, entry, {"SAGEMAKER_PROGRAM": entry}, secret_arn) as (
        staged,
        safe_entry,
        safe_environment,
    ):
        assert safe_entry == launcher.SECURE_ENTRY
        assert (staged / launcher.SECURE_ENTRY).is_file()
        assert not (staged / "secrets.env").exists()
        assert safe_environment["WSM_SECRETS_MANAGER_ARN"] == secret_arn
        assert safe_environment["WSM_ORIGINAL_ENTRY"] == entry
        assert "plaintext" not in repr(safe_environment)


def test_plaintext_sensitive_environment_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = "train.sh"
    (source / entry).write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="plaintext sensitive environment keys"):
        with launcher.prepared_source_bundle(
            source,
            entry,
            {"SAGEMAKER_PROGRAM": entry, "HF_TOKEN": "must-not-enter-job-metadata"},
        ):
            pass


def test_every_submit_launcher_uses_shared_fail_closed_policy():
    launchers = sorted(_LAUNCH_DIR.glob("submit_*.py"))
    # Tripwire: bump this ONLY after confirming the new launcher passes every assertion below.
    # 10th = submit_stage_s_producer.py (Stage-S D0 encoder + omega producer).
    # 11th = submit_groot_rmb.py (GR00T N1.7 ReMemBench baselines + mechanism arms + the combined
    #        canary). Signed off by its owner 2026-08-07: it takes the shared guardrail flags,
    #        gates on validate_and_confirm/--confirm-submit, defers the priority and 5-day timeout
    #        caps to launch_guardrails, and submits only through submit_training_job — which pins
    #        the p5e TrainingPlanArn from the queue and enforces the SM_USE_RESERVED_CAPACITY=0
    #        pairing, so no code path in it can strand a job in SCHEDULED or bypass the caps.
    assert len(launchers) == 11, [p.name for p in launchers]
    for path in launchers:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert "from launch_guardrails import" in source, path
        assert "add_guardrail_arguments(" in source, path
        assert "validate_and_confirm(args)" in source, path
        assert "load_secrets" not in source, path
        assert "cam-humanoid" not in source, path
        assert "SageMaker-SageMakerAllAccess" not in source, path
        if "resolve_user(args.user)" in source:
            assert source.index("validate_and_confirm(args)") < source.index("resolve_user(args.user)"), path


def test_legacy_bypass_paths_are_fail_closed():
    repo = _LAUNCH_DIR.parents[1]
    legacy = (repo / "temp_launch_training.py").read_text(encoding="utf-8")
    assert "DEPRECATED: unsafe historical launcher" in legacy
    assert legacy.index("raise SystemExit") < legacy.index("import boto3")

    cache_script = (_LAUNCH_DIR / "cache_all_features.sh").read_text(encoding="utf-8")
    assert "internal_training/secrets.env" not in cache_script
    assert "HF_TOKEN:?" in cache_script


def test_eval_dry_submit_never_loads_aws_sdk(monkeypatch, tmp_path):
    def fail_if_called():
        raise AssertionError("AWS SDK must not load in dry-run")

    monkeypatch.setattr(eval_launcher, "load_aws_sdk", fail_if_called)
    eval_launcher.submit(
        "pi05_guardrail_test",
        "pi05",
        "s3://example/checkpoint",
        1,
        user="offline.test",
        image_uri="example.invalid/image",
        source_dir=tmp_path,
        num_trials=1,
        video="none",
        dry=True,
    )


def test_all_launchers_complete_offline_dry_run(tmp_path):
    entry_names = {
        "robocasa_eval_entry.sh",
        "robocasa_pi05_train_entry.sh",
        "robocasa_pi05_finetune_entry.sh",
        "robocasa_groot_wsm_finetune_entry.sh",
        "robocasa_policyfeat_entry.sh",
        "robocasa_wsm_train_entry.sh",
    }
    for name in entry_names:
        (tmp_path / name).write_text("#!/bin/sh\n", encoding="utf-8")

    common = [
        "--dry-run",
        "--user",
        "offline.test",
        "--source-dir",
        str(tmp_path),
    ]
    cases = [
        (
            "submit_evals.py",
            common
            + [
                "--only",
                "pi05_off",
                "--ckpt-s3",
                "s3://offline/checkpoint",
                "--step",
                "1",
            ],
        ),
        ("submit_pretrains.py", common + ["--only", "pi05_off"]),
        ("submit_finetunes.py", common + ["--only", "pi05_on"]),
        (
            "submit_pi_stage_s.py",
            common
            + [
                "--arm",
                "s0",
                "--wsmv2-source-s3",
                "s3://sagemaker-us-west-2-141701954645/offline.test/wsm_robocasa/"
                "studies/long_context_v1/code/wsmv2/" + "a" * 64 + ".tgz",
                "--openpi-source-s3",
                "s3://sagemaker-us-west-2-141701954645/offline.test/wsm_robocasa/"
                "studies/long_context_v1/code/openpi/" + "b" * 64 + ".tgz",
                "--tokenizer-s3",
                "s3://sagemaker-us-west-2-141701954645/offline.test/wsm_robocasa/"
                "studies/long_context_v1/artifacts/tokenizers/paligemma/" + "f" * 64 + ".model",
                "--tokenizer-sha256",
                "f" * 64,
                "--init-inventory-s3",
                "s3://sagemaker-us-west-2-141701954645/offline.test/wsm_robocasa/"
                "studies/long_context_v1/manifests/inventories/init/" + "d" * 64 + ".json",
                "--init-inventory-sha256",
                "d" * 64,
                "--target-inventory-s3",
                "s3://sagemaker-us-west-2-141701954645/offline.test/wsm_robocasa/"
                "studies/long_context_v1/manifests/inventories/data/" + "e" * 64 + ".json",
                "--target-inventory-sha256",
                "e" * 64,
                "--image-uri",
                "141701954645.dkr.ecr.us-west-2.amazonaws.com/offline@sha256:" + "c" * 64,
            ],
        ),
        ("submit_wsm_cfg.py", common),
        ("submit_policyfeats.py", common),
        ("submit_wsm.py", common),
        ("submit_wsm_canary.py", common),
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo = _LAUNCH_DIR.parents[1]
    for filename, arguments in cases:
        result = subprocess.run(
            [sys.executable, str(_LAUNCH_DIR / filename), *arguments],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, (
            filename,
            result.stdout,
            result.stderr,
        )
        assert "DRY RUN" in result.stdout, filename
