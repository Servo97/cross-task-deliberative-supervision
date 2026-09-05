from __future__ import annotations

import hashlib
import importlib
import json
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from robomme_integration.eval import (
    campaign,
    launch_p5_campaign,
    launch_p5_preflight,
    p5_parallel_action_preflight,
    parallel_campaign,
)
from robomme_integration.launch import (
    IMAGE,
    IMAGE_SHA,
    OPENPI,
    OPENPI_SHA,
    PTRM_OPENPI,
    PTRM_OPENPI_SHA,
)


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


def _preflight(source: Path, destination: Path) -> Path:
    with launch_p5_campaign.prepared_source_bundle(
        source,
        launch_p5_campaign.ENTRY,
        {"SAGEMAKER_PROGRAM": launch_p5_campaign.ENTRY},
        None,
    ) as (staged, _, _):
        source_sha = launch_p5_campaign.source_tree_sha256(staged)
    value = {
        "schema_version": 1,
        "kind": campaign.PREFLIGHT_KIND,
        "preflight_id": "p5-native-eval-v1-test",
        "runtime": {
            "uri": launch_p5_campaign.RUNTIME_S3,
            "sha256": launch_p5_campaign.RUNTIME_SHA,
        },
        "openpi": {"uri": PTRM_OPENPI, "sha256": PTRM_OPENPI_SHA},
        "vla_eval_entrypoint": {
            "kind": "python_module_wrapper",
            "module": "vla_eval.cli.main",
        },
        "image": {"uri": IMAGE, "sha256": IMAGE_SHA},
        "probe": {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
        },
        "source_tree_sha256": source_sha,
        "claim_s3": (f"{launch_p5_campaign.STUDY_ROOT}/manifests/claims/preflight/p5-native-eval-v1-test.json"),
        "infrastructure": {
            "queue": launch_p5_campaign.QUEUE,
            "role": launch_p5_campaign.ROLE_ARN,
            "instance_type": "ml.p5.48xlarge",
            "accelerator": "8xH100",
            "priority": 100,
        },
    }
    value = campaign.seal_document(value, field="manifest_sha256")
    value["status"] = "native_render_reset_passed"
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _queue_template(
    destination: Path,
    queue_id: str = "pick-anchor-eval-v1",
    *,
    serving_openpi: tuple[str, str] = (PTRM_OPENPI, PTRM_OPENPI_SHA),
) -> Path:
    run_id = "st-v1-pickxtimes-s0-seed0-aaaaaaaaaaaaaaaa"
    config_path = "robomme_integration/eval/configs/pickxtimes.yaml"
    config_sha = hashlib.sha256((launch_p5_campaign.REPO_ROOT / config_path).read_bytes()).hexdigest()
    nuisance = {
        "data_parent_inventory_sha256": "a" * 64,
        "data_task_inventory_sha256": "b" * 64,
        "initialization_inventory_sha256": "c" * 64,
        "initialization_checkpoint_s3": "s3://bucket/init/149999",
        "seed": 0,
        "steps": 20_000,
        "action_horizon": 20,
        "window_len": None,
        "chunk_stride": None,
    }
    common = {key: value for key, value in nuisance.items() if key not in {"window_len", "chunk_stride"}}
    publish = f"s3://bucket/eval-campaigns/{queue_id}"
    cell_id = "000-pickxtimes-s0"
    openpi_uri, openpi_sha = serving_openpi
    value = {
        "schema_version": 1,
        "kind": campaign.QUEUE_KIND,
        "queue_id": queue_id,
        "publish_root_s3": publish,
        "claims": {"manifest": f"{publish}/manifest.json", "completion": f"{publish}/complete.json"},
        "topology": {
            "policy_gpus": [0, 1, 2, 3],
            "simulator_gpus": [4, 5, 6, 7],
            "simulator_shards": 16,
            "cpu_range": "0-191",
            "base_port": 18100,
            "xla_memory_fraction": 0.65,
        },
        "retry": {"classifier_version": campaign.CLASSIFIER_VERSION, "max_attempts": 2},
        "limits": {
            "max_run_seconds": 72_000,
            "runtime_reserve_seconds": 1_800,
            "estimated_cell_seconds": 3_600,
            "minimum_free_bytes": 64 * 1024**3,
        },
        "comparability": {
            "serving_openpi": {"uri": openpi_uri, "sha256": openpi_sha},
            "task_benchmark_configs": {
                "PickXtimes": {"path": config_path, "sha256": config_sha},
            },
            "task_common_training_nuisance": {"PickXtimes": common},
            "sequence_geometry_policy": "manifest_verified_per_cell_not_assumed_common",
        },
        "cells": [
            {
                "ordinal": 0,
                "cell_id": cell_id,
                "task": "PickXtimes",
                "arm": "s0",
                "run_id": run_id,
                "final_step": 19_999,
                "scientific_spec_sha256": "d" * 64,
                "run_manifest_sha256": "e" * 64,
                "training_openpi": {"uri": openpi_uri, "sha256": openpi_sha},
                "training_run_manifest_s3": (f"s3://bucket/manifests/runs/train/{run_id}/{run_id}-attempt1.json"),
                "training_output_s3": (f"s3://bucket/checkpoints/PickXtimes/s0/seed0/{run_id}"),
                "training_completion_claim_s3": (
                    f"s3://bucket/manifests/claims/train/{run_id}/step-19999.complete.json"
                ),
                "training_completion_binding": campaign.TRAINING_COMPLETION_CURRENT,
                "benchmark_config": config_path,
                "benchmark_config_sha256": config_sha,
                "training_nuisance": nuisance,
                "eval_id": f"{run_id}-fixed50-{queue_id}",
                "result_claim_s3": f"{publish}/cells/{cell_id}/result.complete.json",
                "workspace": None,
                "ptrm": None,
                "cfg_guidance_scale": 1.0,
            }
        ],
    }
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _parallel_preflight(source: Path, destination: Path) -> Path:
    args = launch_p5_preflight.parser().parse_args(["--dry-run", "--parallel-action-canary"])
    plan = launch_p5_preflight.build_plan(args, source)
    topology = parallel_campaign.p5_8xh100_topology()
    lane_digests = {
        "supervisor_sha256": "1" * 64,
        "server_log_sha256": "3" * 64,
    }
    observed = {
        "execution_mode": parallel_campaign.PARALLEL_EXECUTION_MODE,
        "parallel_topology_sha256": topology.as_queue_topology()["parallel_topology_sha256"],
        "parallel_lanes": 8,
        "policy_servers": 8,
        "native_shards_per_lane": 4,
        "native_shards_total": 32,
        "episodes": 32,
        "actions_executed": 32,
        "harness_failures": 0,
        "load_completed_before_readiness": True,
        "all_lane_gpus_idle": True,
        "all_lane_ports_free": True,
        "lanes": [
            {
                "lane_id": lane.lane_id,
                "gpu": lane.policy_gpu,
                "port": lane.port,
                "cpu_range": lane.cpu_range,
                "episodes": 4,
                "actions_executed": 4,
                "harness_failures": 0,
                "load_completed_before_readiness": True,
                **lane_digests,
            }
            for lane in topology.lanes
        ],
    }
    evidence_payload = b"score-redacted-action-canary-evidence"
    evidence_sha = hashlib.sha256(evidence_payload).hexdigest()
    claim = p5_parallel_action_preflight.build_success_claim(
        plan["manifest"],
        observed=observed,
        evidence={
            "uri": f"{plan['manifest']['evidence_root_s3']}/{evidence_sha}.tgz",
            "sha256": evidence_sha,
            "bytes": len(evidence_payload),
        },
    )
    destination.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def test_campaign_launch_is_dry_default_low_priority_exact_and_node_resident(tmp_path):
    source = launch_p5_campaign.REPO_ROOT / "robomme_integration"
    preflight = _preflight(source, tmp_path / "preflight.json")
    queue = _queue_template(tmp_path / "queue.json")
    args = launch_p5_campaign.parser().parse_args(
        ["--queue-template", str(queue), "--native-preflight-claim", str(preflight)]
    )
    assert not args.confirm_submit and not args.dry_run
    plan = launch_p5_campaign.build_plan(args, source)
    assert plan["launch"]["infrastructure"] == {
        "provider": "aws_sagemaker",
        "execution_account": launch_p5_campaign.EXECUTION_ACCOUNT,
        "queue": launch_p5_campaign.QUEUE,
        "role": launch_p5_campaign.ROLE_ARN,
        "instance_type": "ml.p5.48xlarge",
        "accelerator": "8xH100",
        "priority": 100,
        "max_run_seconds": 86_400,
        "staging_reserve_seconds": 7_200,
        "volume_size_gb": 200,
    }
    assert plan["queue"]["gates"]["runtime_receipt"]["openpi_sha256"] == PTRM_OPENPI_SHA
    assert set(plan["staged_files"]) == set(launch_p5_campaign.GENERATED_FILES)
    assert plan["receipt"]["generated_source_files_excluded"] == list(launch_p5_campaign.GENERATED_FILES)
    assert (
        plan["receipt"]["vla_eval_wrapper"]["sha256"]
        == hashlib.sha256(launch_p5_campaign._vla_eval_wrapper()).hexdigest()
    )
    assert plan["environment"]["ROBOMME_EVAL_OPENPI_PROFILE"] == "advanced"
    entry = (source / launch_p5_campaign.ENTRY).read_text(encoding="utf-8")
    assert "SOURCE_IDENTITY_OK" in entry
    assert "RUNTIME_RECEIPT_OK" in entry
    assert "--confirm-run" in entry
    assert "robomme_integration.eval.campaign" in entry
    assert "robomme-v0.4.0/src" in entry
    assert "ManiSkill-07be6fbc" in entry
    assert 'exec "$PY" -m vla_eval.cli.main' in entry
    assert '"$LINKS/vla-eval" run --help' in entry

    parallel_queue = _queue_template(
        tmp_path / "parallel-queue.json",
        queue_id="pick-anchor-eval-parallel-v1",
        serving_openpi=(OPENPI, OPENPI_SHA),
    )
    parallel_preflight = _parallel_preflight(source, tmp_path / "parallel-preflight.json")
    parallel_args = launch_p5_campaign.parser().parse_args(
        [
            "--queue-template",
            str(parallel_queue),
            "--native-preflight-claim",
            str(parallel_preflight),
            "--parallel-fixed50",
        ]
    )
    parallel_plan = launch_p5_campaign.build_plan(parallel_args, source)
    assert parallel_plan["queue"]["topology"] == (parallel_campaign.p5_8xh100_topology().as_queue_topology())
    assert (
        parallel_plan["launch"]["parallel_topology_sha256"]
        == (parallel_plan["queue"]["topology"]["parallel_topology_sha256"])
    )
    assert parallel_plan["environment"]["ROBOMME_EVAL_OPENPI_PROFILE"] == "standard"
    receipt_path = tmp_path / "parallel-receipt.json"
    receipt_path.write_text(
        json.dumps(parallel_plan["receipt"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = campaign.verify_gates(
        parallel_plan["queue"],
        preflight_claim=parallel_preflight,
        runtime_receipt=receipt_path,
        require_runtime_paths=False,
    )
    # The runtime gate must preserve the lexical venv executable used by the receipt.
    assert runtime.policy_python == launch_p5_campaign.OPENPI_ROOT / ".venv/bin/python"
    evidence_payload = b"score-redacted-action-canary-evidence"
    launch_p5_campaign.validate_published_preflight(
        parallel_plan,
        claim_payload=parallel_plan["staged_files"][launch_p5_campaign.STAGED_PREFLIGHT],
        evidence_payload=evidence_payload,
    )
    with pytest.raises(SystemExit, match="evidence is absent or corrupt"):
        launch_p5_campaign.validate_published_preflight(
            parallel_plan,
            claim_payload=parallel_plan["staged_files"][launch_p5_campaign.STAGED_PREFLIGHT],
            evidence_payload=b"tampered",
        )
    assert (
        parallel_plan["environment"]["ROBOMME_EVAL_PREFLIGHT_CLAIM_S3"]
        == (parallel_plan["preflight_claim"]["claim_s3"])
    )
    entry = (source / launch_p5_campaign.ENTRY).read_text(encoding="utf-8")
    assert 'cmp -s "$CODE/$ROBOMME_EVAL_PREFLIGHT_SOURCE"' in entry
    assert '"$WORK/preflight-evidence.tgz"' in entry
    assert "parallel action-preflight evidence contains unapproved members" in entry


def test_campaign_launch_rejects_preflight_source_or_base_openpi_drift(tmp_path):
    source = launch_p5_campaign.REPO_ROOT / "robomme_integration"
    preflight = _preflight(source, tmp_path / "preflight.json")
    value = json.loads(preflight.read_text(encoding="utf-8"))
    value["openpi"] = {"uri": "s3://bucket/base/" + "f" * 64 + ".tgz", "sha256": "f" * 64}
    value.pop("status")
    value = campaign.seal_document(value, field="manifest_sha256")
    value["status"] = "native_render_reset_passed"
    preflight.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args = launch_p5_campaign.parser().parse_args(
        [
            "--queue-template",
            str(_queue_template(tmp_path / "queue.json")),
            "--native-preflight-claim",
            str(preflight),
        ]
    )
    with pytest.raises(SystemExit, match="training source"):
        launch_p5_campaign.build_plan(args, source)


def test_campaign_launch_rejects_claim_selected_runtime_and_noncanonical_claim(tmp_path):
    source = launch_p5_campaign.REPO_ROOT / "robomme_integration"
    preflight = _preflight(source, tmp_path / "preflight.json")
    value = json.loads(preflight.read_text(encoding="utf-8"))
    value.pop("status")
    value["runtime"] = {
        "uri": f"s3://bucket/runtime/{'f' * 64}.tgz",
        "sha256": "f" * 64,
    }
    value = campaign.seal_document(value, field="manifest_sha256")
    value["status"] = "native_render_reset_passed"
    preflight.write_text(json.dumps(value), encoding="utf-8")
    args = launch_p5_campaign.parser().parse_args(
        [
            "--queue-template",
            str(_queue_template(tmp_path / "queue.json")),
            "--native-preflight-claim",
            str(preflight),
        ]
    )
    with pytest.raises(SystemExit, match="registered native evaluator"):
        launch_p5_campaign.build_plan(args, source)

    preflight = _preflight(source, tmp_path / "preflight.json")
    value = json.loads(preflight.read_text(encoding="utf-8"))
    value.pop("status")
    value["claim_s3"] = "s3://caller-bucket/preflight.json"
    value = campaign.seal_document(value, field="manifest_sha256")
    value["status"] = "native_render_reset_passed"
    preflight.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SystemExit, match="claim URI is not canonical"):
        launch_p5_campaign.build_plan(args, source)


def test_campaign_entry_identity_survives_tar_roundtrip_and_toolkit_chmod(tmp_path):
    """Same toolkit 0755 -> 0777 entry chmod as the preflight; the campaign entry excludes exactly
    its four generated files and normalizes only its own path (see test_p5_eval_preflight)."""
    source = Path(launch_p5_campaign.__file__).resolve().parents[1]
    entry_name = launch_p5_campaign.ENTRY
    generated = list(launch_p5_campaign.GENERATED_FILES)
    assert launch_p5_campaign.SUBMITTED_ENTRY_MODE == 0o755
    assert launch_p5_campaign.SAGEMAKER_RUNTIME_ENTRY_MODE == 0o777
    with launch_p5_campaign.prepared_source_bundle(source, entry_name, {"SAGEMAKER_PROGRAM": entry_name}, None) as (
        staged,
        _,
        _,
    ):
        assert stat.S_IMODE((staged / entry_name).lstat().st_mode) == launch_p5_campaign.SUBMITTED_ENTRY_MODE
        source_sha = launch_p5_campaign.source_tree_sha256(staged)
        # scripts/launch is on sys.path once the launcher module above is imported.
        importlib.import_module("launch_guardrails").write_staged_source_files(
            staged, {name: "{}\n" for name in generated}
        )
        code = _sagemaker_roundtrip(staged, tmp_path)
    script = _entry_identity_heredoc(
        (source / entry_name).read_text(encoding="utf-8"),
        "# Reproduce launch_guardrails.source_tree_sha256 on the actual unpacked SageMaker tree",
    )
    argv = [str(code), source_sha, ",".join(generated), entry_name]
    refused = _node_identity_check(script, argv)
    assert refused.returncode != 0 and "must be mode 0777" in refused.stderr
    (code / entry_name).chmod(0o777)
    accepted = _node_identity_check(script, argv)
    assert accepted.returncode == 0, accepted.stderr
    assert f"SOURCE_IDENTITY_OK sha256={source_sha}" in accepted.stdout
    (code / "__init__.py").write_bytes(b"# tampered\n")
    assert "sanitized source identity differs from preflight" in _node_identity_check(script, argv).stderr
