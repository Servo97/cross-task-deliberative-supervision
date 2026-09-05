from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import pytest

from robomme_integration.eval import project_exact_runner as runner
from robomme_integration.eval.project_exact_server import PROTOCOL_ID
from robomme_integration.fleet.checkpoint import build as build_checkpoint_manifest


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_runner_accepts_only_the_three_execution_controls():
    assert [runner.validate_runner_arm(arm) for arm in ("s0", "q0", "a6")] == [
        "s0",
        "q0",
        "a6",
    ]
    with pytest.raises(ValueError, match="stride/commit=10"):
        runner.validate_runner_arm("q2")
    with pytest.raises(ValueError, match="distinct two-view"):
        runner.validate_runner_arm("official_recipe_lerobot")


def test_python_validation_preserves_the_virtualenv_entry_symlink(tmp_path):
    interpreter = tmp_path / "system-python"
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(interpreter)
    assert runner._require_python(venv_python, "test") == venv_python.absolute()


def test_checkpoint_verification_binds_tree_bytes_manifest_and_expected_digest(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write(checkpoint / "params" / "weights", b"weights")
    _write(checkpoint / "assets" / "norm.json", b"{}")
    manifest = tmp_path / "tree.json"
    digest = build_checkpoint_manifest(
        checkpoint,
        "s3://sealed/checkpoints/s0/steps/59999",
        manifest,
        require_finalized=False,
    )
    assert runner.verify_checkpoint_tree(
        checkpoint_root=checkpoint,
        manifest_path=manifest,
        expected_sha256=digest,
    ) == (digest, "s3://sealed/checkpoints/s0/steps/59999")

    _write(checkpoint / "params" / "weights", b"drifted")
    with pytest.raises(ValueError, match="differs from its sealed manifest"):
        runner.verify_checkpoint_tree(
            checkpoint_root=checkpoint,
            manifest_path=manifest,
            expected_sha256=digest,
        )


def test_source_staging_derives_identity_and_reextracts_the_pinned_archive(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _write(project / "robomme_integration" / "eval" / "project_exact_runner.py", b"runner")
    _write(project / "robomme_integration" / "eval" / "project_exact_server.py", b"server")
    harness = tmp_path / "harness"
    _write(harness / "src" / "vla_eval" / "model_servers" / "base.py", b"base")
    archive_tree = tmp_path / "archive-tree"
    _write(archive_tree / "src" / "openpi" / "__init__.py", b"")
    _write(
        archive_tree / "packages" / "openpi-client" / "src" / "openpi_client" / "__init__.py",
        b"",
    )
    archive = tmp_path / "openpi.tgz"
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(archive_tree.rglob("*")):
            output.add(path, arcname=path.relative_to(archive_tree), recursive=False)
    archive_sha = runner.sha256_file(archive)
    monkeypatch.setattr(runner, "OPENPI_ARCHIVE_SHA256", archive_sha)

    project_snapshot = runner.stage_project_snapshot(
        project_root=project,
        vla_eval_root=harness,
        work_root=tmp_path / "work",
    )
    openpi_snapshot = runner.stage_openpi_archive(
        archive=archive,
        work_root=tmp_path / "work",
    )
    assert project_snapshot.root.name == f"project-{project_snapshot.sha256}"
    assert openpi_snapshot.root.name == f"openpi-{archive_sha}"
    assert runner.source_tree_sha256(project_snapshot.root) == project_snapshot.sha256

    # Every invocation extracts a fresh copy and compares it with the reusable content-addressed
    # tree, so silent mutation is detected instead of trusted from a marker filename.
    _write(openpi_snapshot.root / "src" / "openpi" / "__init__.py", b"tampered")
    with pytest.raises(RuntimeError, match="was modified"):
        runner.stage_openpi_archive(archive=archive, work_root=tmp_path / "work")

    resumed = runner.load_project_snapshot(
        work_root=tmp_path / "work",
        expected_sha256=project_snapshot.sha256,
        expected_robomme_sha256=project_snapshot.robomme_sha256,
        expected_vla_eval_sha256=project_snapshot.vla_eval_sha256,
    )
    assert resumed == project_snapshot
    _write(resumed.root / "robomme_integration" / "drift.py", b"drift")
    with pytest.raises(RuntimeError, match="content differs"):
        runner.load_project_snapshot(
            work_root=tmp_path / "work",
            expected_sha256=project_snapshot.sha256,
            expected_robomme_sha256=project_snapshot.robomme_sha256,
            expected_vla_eval_sha256=project_snapshot.vla_eval_sha256,
        )


def _contract(tmp_path: Path) -> runner.RuntimeContract:
    project = tmp_path / "project-snapshot"
    _write(project / "robomme_integration" / "eval" / "project_exact_server.py", b"server")
    _write(project / "robomme_integration" / "eval" / "project_exact_eval.py", b"evaluator")
    openpi = tmp_path / "openpi-snapshot"
    (openpi / "src").mkdir(parents=True)
    policy = tmp_path / "policy"
    (policy / "examples" / "robomme").mkdir(parents=True)
    benchmark = tmp_path / "benchmark"
    maniskill = tmp_path / "maniskill"
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    return runner.RuntimeContract(
        arm="s0",
        method="project-exact-s0",
        checkpoint_root=tmp_path / "checkpoint",
        checkpoint_sha256="a" * 64,
        checkpoint_uri="s3://sealed/s0",
        project=runner.ProjectSnapshot(project, "b" * 64, "c" * 64, "d" * 64),
        openpi=runner.OpenPISnapshot(openpi, "e" * 64, "f" * 64),
        openpi_python=python,
        simulator_python=python,
        policy_root=policy,
        benchmark_root=benchmark,
        maniskill_root=maniskill,
        output=tmp_path / "output",
        port=18720,
        server_source_sha256=runner.sha256_file(project / "robomme_integration" / "eval" / "project_exact_server.py"),
        evaluator_source_sha256=runner.sha256_file(project / "robomme_integration" / "eval" / "project_exact_eval.py"),
        policy_runtime={"fingerprint_sha256": "1" * 64},
        simulator_runtime={"fingerprint_sha256": "2" * 64},
    )


def _scorecard(contract: runner.RuntimeContract) -> dict:
    return {
        "episodes": 800,
        "successes": 0,
        "result_scale": "fraction_0_1",
        "protocol_id": PROTOCOL_ID,
        "method": contract.method,
        "arm": contract.arm,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "project_source_sha256": contract.project.sha256,
        "openpi_source_sha256": contract.openpi.archive_sha256,
        "server_source_sha256": contract.server_source_sha256,
        "evaluator_sha256": contract.evaluator_source_sha256,
        "reference_evaluator_sha256": runner.REFERENCE_EVALUATOR_SHA256,
        "policy_source_commit": runner.POLICY_SOURCE_COMMIT,
        "benchmark_source_commit": runner.BENCHMARK_SOURCE_COMMIT,
        "maniskill_source_commit": runner.MANISKILL_SOURCE_COMMIT,
        "model_seed": 7,
        "action_horizon": 20,
        "execution_horizon": 16,
        "estimand": "project_multitask_checkpoint_single_seed_exact_paper_protocol",
        "task_success_rate": {f"task-{index}": 0.0 for index in range(16)},
    }


def test_exact_vulkan_retry_keeps_one_policy_process_and_resumes_atomic_eval(tmp_path, monkeypatch):
    contract = _contract(tmp_path)
    popen_calls = []
    eval_calls = []

    class PolicyProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return PolicyProcess()

    def fake_run(*args, **kwargs):
        eval_calls.append((args, kwargs))
        if len(eval_calls) == 1:
            kwargs["stdout"].write((runner.VULKAN_DRIVER_SIGNATURE + "\n").encode())
            return argparse.Namespace(returncode=1)
        scorecard = contract.output / "evaluation" / "scorecard.json"
        scorecard.parent.mkdir(parents=True, exist_ok=True)
        scorecard.write_text(json.dumps(_scorecard(contract)), encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_assert_port_free", lambda _port: None)
    monkeypatch.setattr(runner, "_wait_for_server", lambda _process, _port, _timeout: None)
    monkeypatch.setattr(runner, "_wait_for_renderer_recovery", lambda _process, _timeout: None)
    args = argparse.Namespace(
        max_renderer_restarts=2,
        policy_cuda_device="0",
        simulator_cuda_device="1",
        server_ready_timeout=1,
        renderer_recovery_timeout=1,
    )
    assert runner.run(contract, args) == 0
    assert len(popen_calls) == 1
    assert len(eval_calls) == 2
    assert (contract.output / "evaluation" / "progress.json").exists() is False
    assert (contract.output / "PROJECT_EXACT_FIXED800_COMPLETE").is_file()
    retry_state = json.loads((contract.output / "renderer_retry_state.json").read_text())
    assert retry_state == {"attempts": 2, "renderer_restarts": 1}


def test_retry_classifier_does_not_retry_generic_vulkan_or_driver_failures():
    assert runner.is_retryable_renderer_failure(runner.VULKAN_DRIVER_SIGNATURE)
    assert not runner.is_retryable_renderer_failure("Vulkan unavailable")
    assert not runner.is_retryable_renderer_failure("ErrorIncompatibleDriver")


def test_interrupted_attempt_log_is_preserved_and_resume_allocates_next_number(tmp_path, monkeypatch):
    contract = _contract(tmp_path)
    eval_calls = 0

    class PolicyProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_run(*args, **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        if eval_calls == 1:
            kwargs["stdout"].write(b"interrupted after atomic progress\n")
            raise KeyboardInterrupt
        scorecard = contract.output / "evaluation" / "scorecard.json"
        scorecard.parent.mkdir(parents=True, exist_ok=True)
        scorecard.write_text(json.dumps(_scorecard(contract)), encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: PolicyProcess())
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_assert_port_free", lambda _port: None)
    monkeypatch.setattr(runner, "_wait_for_server", lambda _process, _port, _timeout: None)
    args = argparse.Namespace(
        max_renderer_restarts=2,
        policy_cuda_device="0",
        simulator_cuda_device="1",
        server_ready_timeout=1,
        renderer_recovery_timeout=1,
    )

    with pytest.raises(KeyboardInterrupt):
        runner.run(contract, args)
    first = contract.output / "logs" / "eval-attempt-0001.log"
    assert first.read_text() == "interrupted after atomic progress\n"
    assert json.loads((contract.output / "renderer_retry_state.json").read_text())["attempts"] == 1

    assert runner.run(contract, args) == 0
    assert first.read_text() == "interrupted after atomic progress\n"
    assert (contract.output / "logs" / "eval-attempt-0002.log").is_file()
    assert json.loads((contract.output / "renderer_retry_state.json").read_text())["attempts"] == 2


def test_policy_command_and_environment_use_only_staged_sources_in_safe_order(tmp_path, monkeypatch):
    contract = _contract(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/untrusted/installed/fork")
    command = runner.server_command(contract)
    environment = runner.server_environment(contract, "0")
    staged_server = contract.project.root / "robomme_integration" / "eval" / "project_exact_server.py"
    assert command[:2] == [str(contract.openpi_python), str(staged_server)]
    assert "/untrusted/installed/fork" not in environment["PYTHONPATH"]
    assert environment["PYTHONPATH"].split(runner.os.pathsep) == [
        str(contract.project.root / "robomme_integration" / "compat"),
        str(contract.project.root),
        str(contract.openpi.root / "src"),
        str(contract.openpi.root / "packages" / "openpi-client" / "src"),
    ]
    evaluator = runner.evaluator_command(contract)
    assert evaluator.count("--maniskill-source-commit") == 1
    assert evaluator[evaluator.index("--maniskill-source-commit") + 1] == runner.MANISKILL_SOURCE_COMMIT


def test_renderer_recovery_waits_for_nvidia_without_restarting_policy(monkeypatch, capsys):
    class PolicyProcess:
        returncode = None

        def poll(self):
            return None

    probes = iter((argparse.Namespace(returncode=1), argparse.Namespace(returncode=0)))
    sleeps = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: next(probes))
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner._wait_for_renderer_recovery(PolicyProcess(), 60)
    assert sleeps == [30, 5]
    output = capsys.readouterr().out
    assert "WAITING_FOR_NVIDIA_DRIVER" in output
    assert "NVIDIA_DRIVER_RECOVERED" in output
