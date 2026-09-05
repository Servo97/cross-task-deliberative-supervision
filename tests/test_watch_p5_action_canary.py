from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch/watch_p5_action_canary.py"
SPEC = importlib.util.spec_from_file_location("watch_p5_action_canary_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def _packet() -> dict:
    preflight_id = "p5-native-eval-v1-" + "1" * 20
    value = {
        "schema_version": 1,
        "kind": watcher.PACKET_KIND,
        "status": watcher.PACKET_STATUS,
        "builder_task": "/root/prepare_p5_parallel_eval",
        "baseline_source_tree_sha256": "2" * 64,
        "source_tree_sha256": "3" * 64,
        "source_delta": [{"path": "eval/canary.py", "mode": "0664", "sha256": "4" * 64}],
        "launch_guardrails_sha256": "5" * 64,
        "preflight_id": preflight_id,
        "job_name": "sarvesh-rmme-p5-action-" + "1" * 20,
        "manifest_sha256": "6" * 64,
        "claim_s3": f"s3://bucket/manifests/claims/preflight/{preflight_id}.json",
        "evidence_root_s3": f"s3://bucket/artifacts/robomme/eval_preflight/{preflight_id}/evidence",
        "canary_template_sha256": "7" * 64,
        "canary_config_sha256": "8" * 64,
        "parallel_topology_sha256": "9" * 64,
        "actions_expected": 32,
        "score_publication": False,
        "scored_queue_template": {
            "path": "eval/queue.json",
            "sha256": "a" * 64,
            "queue_id": "pick-wave",
            "cells": [
                {
                    "ordinal": ordinal,
                    "task": "PickXtimes",
                    "arm": f"arm{ordinal}",
                    "run_id": f"run-{ordinal}",
                }
                for ordinal in range(8)
            ],
        },
    }
    return watcher.seal_document(value, field="packet_sha256")


def _audit(packet: dict, *, auditor: str = "/root/independent_auditor") -> dict:
    value = {
        "schema_version": 1,
        "kind": watcher.AUDIT_KIND,
        "status": watcher.AUDIT_STATUS,
        "auditor_task": auditor,
        "packet_sha256": packet["packet_sha256"],
        "source_tree_sha256": packet["source_tree_sha256"],
        "preflight_id": packet["preflight_id"],
        "manifest_sha256": packet["manifest_sha256"],
        "checks": {name: True for name in watcher.REQUIRED_AUDIT_CHECKS},
    }
    return watcher.seal_document(value, field="audit_receipt_sha256")


def _audit_payload(audit: dict) -> bytes:
    return (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()


def _snapshot(*, p5_instances: int = 9, status: str = "RUNNING") -> dict:
    return {
        "account": "141701954645",
        "claim_objects": [],
        "evidence_objects": [],
        "training_job_name_exists": False,
        "batch_jobs": [{"jobName": "other", "status": status}],
        "training_jobs": [
            {
                "TrainingJobName": "other-sm",
                "InstanceType": "ml.p5.48xlarge",
                "InstanceCount": p5_instances,
            }
        ],
    }


def test_ready_packet_and_independent_receipt_are_exactly_sealed_and_byte_pinned():
    packet = watcher.validate_ready_packet(_packet())
    audit = _audit(packet)
    payload = _audit_payload(audit)
    watcher.validate_audit_receipt(
        audit,
        packet=packet,
        payload=payload,
        expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )

    forged = dict(packet)
    forged["actions_expected"] = 31
    with pytest.raises(watcher.WatcherContractError, match="self-seal mismatch"):
        watcher.validate_ready_packet(forged)
    with pytest.raises(watcher.WatcherContractError, match="approved digest"):
        watcher.validate_audit_receipt(
            audit,
            packet=packet,
            payload=payload + b" ",
            expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
    same_builder = _audit(packet, auditor=packet["builder_task"])
    same_builder_payload = _audit_payload(same_builder)
    with pytest.raises(watcher.WatcherContractError, match="independent auditor"):
        watcher.validate_audit_receipt(
            same_builder,
            packet=packet,
            payload=same_builder_payload,
            expected_payload_sha256=hashlib.sha256(same_builder_payload).hexdigest(),
        )
    red = _audit(packet)
    red["checks"]["ruff_green"] = False
    red.pop("audit_receipt_sha256")
    red = watcher.seal_document(red, field="audit_receipt_sha256")
    red_payload = _audit_payload(red)
    with pytest.raises(watcher.WatcherContractError, match="non-green"):
        watcher.validate_audit_receipt(
            red,
            packet=packet,
            payload=red_payload,
            expected_payload_sha256=hashlib.sha256(red_payload).hexdigest(),
        )


@pytest.mark.parametrize("status", sorted(watcher.WAITING_STATUSES))
def test_live_gate_rejects_every_waiting_batch_status(status):
    with pytest.raises(watcher.WatcherContractError, match=f"waiting status {status}"):
        watcher.validate_go_snapshot(
            _snapshot(status=status),
            job_name="canary",
            account="141701954645",
            p5_limit=10,
        )


def test_live_gate_requires_empty_namespaces_no_duplicate_and_a_real_free_p5_node():
    assert watcher.validate_go_snapshot(
        _snapshot(),
        job_name="canary",
        account="141701954645",
        p5_limit=10,
    ) == {"p5_instances": 9, "p5_available": 1}

    namespace = _snapshot()
    namespace["claim_objects"] = [{"Key": "claim"}]
    with pytest.raises(watcher.WatcherContractError, match="namespace is not empty"):
        watcher.validate_go_snapshot(
            namespace,
            job_name="canary",
            account="141701954645",
            p5_limit=10,
        )
    duplicate = _snapshot()
    duplicate["batch_jobs"][0]["jobName"] = "canary"
    with pytest.raises(watcher.WatcherContractError, match="Batch identity"):
        watcher.validate_go_snapshot(
            duplicate,
            job_name="canary",
            account="141701954645",
            p5_limit=10,
        )
    historical = _snapshot()
    historical["training_job_name_exists"] = True
    with pytest.raises(watcher.WatcherContractError, match="historical"):
        watcher.validate_go_snapshot(
            historical,
            job_name="canary",
            account="141701954645",
            p5_limit=10,
        )
    with pytest.raises(watcher.WatcherContractError, match=r"10/10"):
        watcher.validate_go_snapshot(
            _snapshot(p5_instances=10),
            job_name="canary",
            account="141701954645",
            p5_limit=10,
        )


def test_source_delta_binds_only_exact_changed_file_bytes_and_modes(tmp_path):
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    baseline.mkdir()
    source.mkdir()
    (baseline / "same.txt").write_text("same")
    (source / "same.txt").write_text("same")
    (baseline / "changed.txt").write_text("old")
    (source / "changed.txt").write_text("new")
    (source / "new.txt").write_text("added")
    for root in (baseline, source):
        (root / "same.txt").chmod(0o644)
    (baseline / "changed.txt").chmod(0o644)
    (source / "changed.txt").chmod(0o640)
    (source / "new.txt").chmod(0o600)
    assert watcher._source_delta(source, baseline) == [
        {
            "path": "changed.txt",
            "mode": "0640",
            "sha256": hashlib.sha256(b"new").hexdigest(),
        },
        {
            "path": "new.txt",
            "mode": "0600",
            "sha256": hashlib.sha256(b"added").hexdigest(),
        },
    ]


def test_only_capacity_and_waiter_exits_are_pollable_holds():
    assert "waiting work" in watcher.classify_gate_exit(SystemExit(watcher.TRANSIENT_HOLD_MESSAGES[0]))
    assert "9/9" in watcher.classify_gate_exit(SystemExit("p5 has no genuinely available node (9/9)"))
    with pytest.raises(watcher.WatcherContractError, match="fatal live p5 gate"):
        watcher.classify_gate_exit(SystemExit("namespace collision"))


class _FakeLaunch:
    EXECUTION_ACCOUNT = "141701954645"
    P5_CONCURRENCY_LIMIT = 10

    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)

    def collect_action_submission_snapshot(self, _plan, *, job_name):
        assert job_name == "sarvesh-rmme-p5-action-" + "1" * 20
        value = next(self.snapshots)
        if isinstance(value, BaseException):
            raise value
        return value


def _run_args(tmp_path, *, check_once: bool) -> argparse.Namespace:
    packet = _packet()
    audit = _audit(packet)
    ready = tmp_path / "packet.json"
    receipt = tmp_path / "audit.json"
    ready.write_text(json.dumps(packet))
    payload = _audit_payload(audit)
    receipt.write_bytes(payload)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return argparse.Namespace(
        ready_packet=ready,
        audit_receipt=receipt,
        expected_audit_receipt_sha256=hashlib.sha256(payload).hexdigest(),
        source_dir=tmp_path / "source",
        baseline_source_dir=tmp_path / "baseline",
        submit_python=python,
        poll_seconds=10.0,
        max_wait_seconds=100.0,
        check_once=check_once,
        confirm_auto_submit=not check_once,
        state_file=None if check_once else tmp_path / "handoff.json",
    )


def _patch_run_boundaries(monkeypatch, fake_launch):
    monkeypatch.setattr(watcher, "validate_frozen_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(watcher, "load_frozen_launcher", lambda _source: fake_launch)
    monkeypatch.setattr(watcher, "build_and_validate_plan", lambda *args, **kwargs: {"ok": True})


def test_read_only_check_never_hands_off(monkeypatch, tmp_path):
    args = _run_args(tmp_path, check_once=True)
    fake_launch = _FakeLaunch([_snapshot()])
    _patch_run_boundaries(monkeypatch, fake_launch)
    handed_off = []
    assert watcher.run(args, submit=lambda *a, **k: handed_off.append((a, k))) == 0
    assert handed_off == []


def test_auto_mode_polls_hold_then_hands_off_exactly_once(monkeypatch, tmp_path):
    args = _run_args(tmp_path, check_once=False)
    fake_launch = _FakeLaunch(
        [
            SystemExit(watcher.TRANSIENT_HOLD_MESSAGES[0]),
            _snapshot(),
        ]
    )
    _patch_run_boundaries(monkeypatch, fake_launch)
    handed_off = []

    def submit(command, **kwargs):
        handed_off.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    assert (
        watcher.run(
            args,
            sleep=lambda _seconds: None,
            monotonic=iter([0.0, 1.0]).__next__,
            submit=submit,
        )
        == 0
    )
    assert len(handed_off) == 1
    state = json.loads(args.state_file.read_text())
    assert state["status"] == "HANDOFF_SUCCEEDED"
    assert state["returncode"] == 0
    assert "--confirm-submit" in handed_off[0][0]
    assert handed_off[0][1]["check"] is False


def test_failed_child_handoff_is_not_retried(monkeypatch, tmp_path):
    args = _run_args(tmp_path, check_once=False)
    fake_launch = _FakeLaunch([_snapshot(), _snapshot()])
    _patch_run_boundaries(monkeypatch, fake_launch)
    calls = []

    def submit(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 7)

    assert watcher.run(args, submit=submit) == 7
    assert len(calls) == 1
    assert json.loads(args.state_file.read_text())["status"] == ("HANDOFF_FAILED_REVIEW_REQUIRED")


def test_unknown_live_gate_failure_is_fatal_and_never_hands_off(monkeypatch, tmp_path):
    args = _run_args(tmp_path, check_once=False)
    fake_launch = _FakeLaunch([SystemExit("claim namespace collision")])
    _patch_run_boundaries(monkeypatch, fake_launch)
    calls = []
    with pytest.raises(watcher.WatcherContractError, match="fatal live p5 gate"):
        watcher.run(args, submit=lambda *a, **k: calls.append((a, k)))
    assert calls == []
    assert not args.state_file.exists()
