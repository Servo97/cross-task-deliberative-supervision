import importlib
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from robomme_integration.eval import (
    launch_p5_preflight,
    p5_parallel_action_preflight,
    parallel_campaign,
)
from robomme_integration.launch import OPENPI_SHA, PTRM_OPENPI_SHA
from robomme_integration.training import policy_canary


def _entry_identity_heredoc(entry_text: str, marker: str) -> str:
    """Return the inline python the node runs to re-hash /opt/ml/code, verbatim from the entry."""
    lines = entry_text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(marker))
    opener = next(index for index in range(start, len(lines)) if lines[index].endswith("<<'PY'"))
    closer = next(index for index in range(opener + 1, len(lines)) if lines[index] == "PY")
    return "\n".join(lines[opener + 1 : closer]) + "\n"


def _sagemaker_roundtrip(staged: Path, destination: Path) -> Path:
    """Mimic the SDK's sourcedir.tar.gz (top-level members, modes kept) and the toolkit's unpack."""
    archive = destination / "sourcedir.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for child in sorted(staged.iterdir()):
            tar.add(child, arcname=child.name)
    code = destination / "code"
    code.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(code, filter="fully_trusted")
    return code


def _node_identity_check(script: str, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-", *argv],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )


def test_p5_native_eval_preflight_is_small_low_priority_and_approval_gated():
    source = Path(launch_p5_preflight.__file__).resolve().parents[1]
    args = launch_p5_preflight.parser().parse_args(["--dry-run"])
    plan = launch_p5_preflight.build_plan(args, source)
    infra = plan["manifest"]["infrastructure"]
    assert infra["instance_type"] == "ml.p5.48xlarge"
    assert infra["accelerator"] == "8xH100"
    assert infra["priority"] == 100
    assert infra["max_run_seconds"] == 7200
    assert infra["volume_size_gb"] == 100
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "1"
    assert plan["manifest"]["runtime"]["sha256"] in plan["manifest"]["runtime"]["uri"]
    assert plan["manifest"]["openpi"]["sha256"] == OPENPI_SHA
    advanced = launch_p5_preflight.parser().parse_args(["--dry-run", "--openpi-profile", "advanced"])
    assert launch_p5_preflight.build_plan(advanced, source)["manifest"]["openpi"]["sha256"] == PTRM_OPENPI_SHA
    assert plan["manifest"]["probe"] == {
        "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
        "task": "MoveCube",
        "dataset": "test",
        "episode_idx": 0,
        "rendered_reset": True,
        "require_demo_history": True,
        "require_demo_state_history": True,
    }
    entry_path = source / launch_p5_preflight.ENTRY
    assert os.access(entry_path, os.X_OK)
    entry = entry_path.read_text()
    assert "unset PYTHONPATH PYTHONHOME" in entry
    assert entry.index("unset PYTHONPATH PYTHONHOME") < entry.index("uv sync --frozen")
    assert 'OPENPI_SITE="$WORK/openpi/.venv/lib/python3.11/site-packages"' in entry
    ordered_path = 'PYTHONPATH="$HARNESS:$ROBOMME:$MANISKILL:$OPENPI_SITE:$SITE:$CODE"'
    assert ordered_path in entry
    assert "numpy_path.is_relative_to(openpi_site)" in entry
    assert "lacks PTRM restore support" in entry
    assert "lacks JEPA checkpoint audit support" in entry
    assert '"$WORK/links/vla-eval" run --help' in entry
    assert "await benchmark.start_episode(task)" in entry
    assert "from eval.benchmark import RoboMMEOfficialHistoryBenchmark" in entry
    assert 'observation.get("video_history", [])' in entry
    assert 'observation.get("video_state_history", [])' in entry
    assert "len(history) != len(state_history)" in entry

    blocked = launch_p5_preflight.parser().parse_args([])
    with pytest.raises(SystemExit, match="explicit user approval"):
        launch_p5_preflight.build_plan(blocked, source)


def test_p5_parallel_action_preflight_seals_exact_unscored_8_lane_probe():
    source = Path(launch_p5_preflight.__file__).resolve().parents[1]
    args = launch_p5_preflight.parser().parse_args(["--dry-run", "--parallel-action-canary"])
    plan = launch_p5_preflight.build_plan(args, source)
    manifest = plan["manifest"]
    assert plan["action_mode"] is True
    assert plan["max_run_seconds"] == 4 * 3600
    assert plan["volume_size_gb"] == 200
    assert manifest["preflight_mode"] == p5_parallel_action_preflight.CANARY_MODE
    assert manifest["topology"] == parallel_campaign.p5_8xh100_topology().as_queue_topology()
    assert manifest["probe"]["actions_expected"] == 32
    assert manifest["probe"]["episodes_per_lane"] == 4
    assert manifest["probe"]["score_publication"] is False
    assert "result_claim_s3" not in manifest["cell"]
    assert manifest["cell"]["arm"] == "q1"
    assert manifest["cell"]["task"] == "PickXtimes"
    assert manifest["cell"]["workspace"]["provenance_mode"] == ("omega_manifest_checkpoint_tree_v1")
    assert manifest["vision"]["bytes"] == launch_p5_preflight.VISION_BYTES
    assert manifest["upstream"]["critical_sha256"] == (launch_p5_preflight.UPSTREAM_CRITICAL_SHA256)
    assert plan["claim_s3"].endswith(f"/{plan['preflight_id']}.json")
    assert manifest["evidence_root_s3"].endswith(f"/{plan['preflight_id']}/evidence")
    entry = (source / launch_p5_preflight.ENTRY).read_text(encoding="utf-8")
    assert "SOURCE_IDENTITY_OK" in entry
    assert "p5_parallel_action_preflight" in entry
    assert "--timeout-seconds 3600" in entry
    assert "ROBOMME_EVAL_VISION_SHA256" in entry
    assert 'git -C "$UPSTREAM" rev-parse HEAD' in entry


def test_action_submission_snapshot_requires_empty_namespace_no_waiter_and_free_p5():
    job_name = "sarvesh-rmme-p5-action-test"
    clean = {
        "account": launch_p5_preflight.EXECUTION_ACCOUNT,
        "claim_objects": [],
        "evidence_objects": [],
        "batch_jobs": [{"jobName": "other-running", "status": "RUNNING"}],
        "training_jobs": [
            {
                "TrainingJobName": f"other-{index}",
                "InstanceType": "ml.p5.48xlarge",
                "InstanceCount": 1,
            }
            for index in range(9)
        ],
        "training_job_name_exists": False,
    }
    assert launch_p5_preflight.validate_action_submission_snapshot(clean, job_name=job_name) == {
        "p5_in_progress": 9,
        "p5_available": 1,
        "waiting_jobs": 0,
        "namespace_empty": True,
        "duplicate_free": True,
    }

    cases = [
        ({**clean, "account": "000000000000"}, "wrong AWS account"),
        ({**clean, "claim_objects": [{"Key": "claim"}]}, "namespace is not empty"),
        ({**clean, "evidence_objects": [{"Key": "evidence"}]}, "namespace is not empty"),
        (
            {
                **clean,
                "batch_jobs": [{"jobName": job_name, "status": "RUNNING"}],
            },
            "duplicate job identity",
        ),
        ({**clean, "training_job_name_exists": True}, "duplicate job identity"),
        (
            {
                **clean,
                "batch_jobs": [{"jobName": "queued", "status": "PENDING"}],
            },
            "waiting work",
        ),
        (
            {
                **clean,
                "batch_jobs": [{"jobName": "scheduled", "status": "SCHEDULED"}],
            },
            "waiting work",
        ),
        (
            {
                **clean,
                "training_jobs": [
                    *clean["training_jobs"],
                    {
                        "TrainingJobName": "tenth",
                        "InstanceType": "ml.p5.48xlarge",
                        "InstanceCount": 1,
                    },
                ],
            },
            "no genuinely available node",
        ),
    ]
    for snapshot, message in cases:
        with pytest.raises(SystemExit, match=message):
            launch_p5_preflight.validate_action_submission_snapshot(snapshot, job_name=job_name)


def test_preflight_entry_identity_survives_tar_roundtrip_and_toolkit_chmod(tmp_path):
    """The node re-hash must equal the launcher's digest after SageMaker's own mutations.

    2026-09-04, job sarvesh-rmme-p5-action-4dda9bf2f82aa472cd0a: the training toolkit chmods the
    selected program 0755 -> 0777 before invoking it; an entry that hashed raw mode bits reported
    ``preflight source identity drift``.  The entry now requires that runtime mode and normalizes
    only that path; the launcher pins the submitted mode it normalizes back to.
    """
    source = Path(launch_p5_preflight.__file__).resolve().parents[1]
    args = launch_p5_preflight.parser().parse_args(["--dry-run", "--parallel-action-canary"])
    plan = launch_p5_preflight.build_plan(args, source)
    entry_name = launch_p5_preflight.ENTRY
    manifest_name = launch_p5_preflight.STAGED_MANIFEST
    assert launch_p5_preflight.SUBMITTED_ENTRY_MODE == 0o755
    assert launch_p5_preflight.SAGEMAKER_RUNTIME_ENTRY_MODE == 0o777
    with launch_p5_preflight.prepared_source_bundle(source, entry_name, {"SAGEMAKER_PROGRAM": entry_name}, None) as (
        staged,
        _,
        _,
    ):
        assert launch_p5_preflight.source_tree_sha256(staged) == plan["source_sha"]
        # submit_training_job adds the sealed manifest only after the identity is fixed (mode 0600)
        # scripts/launch is on sys.path once the launcher module above is imported.
        importlib.import_module("launch_guardrails").write_staged_source_files(
            staged, {manifest_name: plan["manifest_json"]}
        )
        code = _sagemaker_roundtrip(staged, tmp_path)
    entry = code / entry_name
    assert stat.S_IMODE(entry.lstat().st_mode) == 0o755
    assert stat.S_IMODE((code / manifest_name).lstat().st_mode) == 0o600
    script = _entry_identity_heredoc(
        (source / entry_name).read_text(encoding="utf-8"),
        "# Reproduce the launcher's sanitized source identity",
    )
    argv = [str(code), plan["source_sha"], manifest_name, entry_name]

    # The toolkit chmod is part of the contract: a tree it did not touch is refused, not silently passed.
    refused = _node_identity_check(script, argv)
    assert refused.returncode != 0
    assert "must be mode 0777" in refused.stderr

    entry.chmod(0o777)
    # The pre-fix algorithm (raw mode bits) is exactly the failed job's disagreement...
    excluded = frozenset({manifest_name})
    assert policy_canary.source_tree_sha256(code, excluded=excluded) != plan["source_sha"]
    assert (
        policy_canary.source_tree_sha256(code, excluded=excluded, mode_overrides={entry_name: 0o755})
        == plan["source_sha"]
    )
    # ...and the shipped entry's inline check now agrees with the launcher on the unpacked tree.
    accepted = _node_identity_check(script, argv)
    assert accepted.returncode == 0, accepted.stderr
    assert f"SOURCE_IDENTITY_OK sha256={plan['source_sha']}" in accepted.stdout

    # Every other mutation still fails the gate: a mode change elsewhere, a byte change, an extra file.
    other = code / "__init__.py"
    other_mode = stat.S_IMODE(other.lstat().st_mode)
    other.chmod(0o777)
    assert "preflight source identity drift" in _node_identity_check(script, argv).stderr
    other.chmod(other_mode)
    assert _node_identity_check(script, argv).returncode == 0
    other.write_bytes(other.read_bytes() + b"# tampered\n")
    assert "preflight source identity drift" in _node_identity_check(script, argv).stderr


def test_failed_preflight_node_hash_is_exactly_the_toolkit_entry_chmod(tmp_path):
    archive = Path(
        "/home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_eval_snapshot_0904/"
        "sourcedir-4dda9bf2f82aa472cd0a.tar.gz"
    )
    if not archive.is_file():
        pytest.skip("the shipped sourcedir.tar.gz of the failed 2026-09-04 preflight is not retained locally")
    code = tmp_path / "code"
    code.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        # Exact bytes SageMaker unpacked into /opt/ml/code; modes are part of the identity.
        tar.extractall(code, filter="fully_trusted")
    entry = code / launch_p5_preflight.ENTRY
    excluded = frozenset({launch_p5_preflight.STAGED_MANIFEST})
    assert stat.S_IMODE(entry.lstat().st_mode) == 0o755
    submitted = policy_canary.source_tree_sha256(code, excluded=excluded)
    assert submitted == "06b7b05b9a9d5884453258923e4c85ba737ba2112d07b7b05ab32a326510a57c"
    entry.chmod(0o777)
    runtime = policy_canary.source_tree_sha256(code, excluded=excluded)
    assert runtime == "62f894378780124d3d81352cc9f31a8a29a786fa529b948bd7d0987a777e77d0"
    normalized = policy_canary.source_tree_sha256(
        code,
        excluded=excluded,
        mode_overrides={launch_p5_preflight.ENTRY: 0o755},
    )
    assert normalized == submitted
