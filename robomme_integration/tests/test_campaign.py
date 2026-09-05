from __future__ import annotations

import hashlib
import json
import shutil
import signal
import threading
import time
import types
from pathlib import Path

import pytest

import robomme_integration.campaign as campaign_module
from robomme_integration.campaign import CampaignRunner, validate_manifest, verify_completion
from robomme_integration.campaign_plan import build_campaign_plan
from robomme_integration.fleet import shared_artifacts


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sealed(value: dict) -> dict:
    result = dict(value)
    result["manifest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return result


class MemoryStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def read_bytes(self, uri: str) -> bytes | None:
        return self.objects.get(uri)

    def put_json_once(self, value: dict, uri: str) -> None:
        payload = _canonical(value)
        prior = self.objects.setdefault(uri, payload)
        if prior != payload:
            raise RuntimeError(f"immutable collision at {uri}")


def _cell(root: Path, ordinal: int, task: str, arm: str) -> dict:
    scientific = hashlib.sha256(f"science:{task}:{arm}".encode()).hexdigest()
    run_id = f"st-v1-{task.lower()}-{arm}-seed0-{scientific[:16]}"
    output = f"s3://bucket/study/{task}/{arm}/{run_id}"
    completion = f"s3://bucket/claims/{run_id}/step-19999.complete.json"
    run_manifest = _sealed(
        {
            "schema_version": 2,
            "kind": "robomme_gpu_training_attempt",
            "run_id": run_id,
            "attempt_id": f"{run_id}-attempt1",
            "scientific_spec_sha256": scientific,
        }
    )
    source = f"cells/{ordinal:03d}.json"
    path = root / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_manifest) + "\n")
    environment = {
        "ROBOMME_SCOPE": "single_task",
        "ROBOMME_TASK": task,
        "ROBOMME_ARM": arm,
        "ROBOMME_RUN_ID": run_id,
        "ROBOMME_ATTEMPT_ID": f"{run_id}-attempt1",
        "ROBOMME_SCIENTIFIC_SPEC_SHA256": scientific,
        "ROBOMME_FINAL_STEP": "19999",
        "WSM_MAX_STEPS": "20000",
        "OUTPUT_S3": output,
        "COMPLETION_CLAIM_S3": completion,
        "RUN_MANIFEST_SOURCE": source,
        "RUN_MANIFEST_SHA256": run_manifest["manifest_sha256"],
    }
    return {
        "ordinal": ordinal,
        "cell_id": f"{ordinal:03d}-{task.lower()}-{arm}",
        "task": task,
        "arm": arm,
        "run_id": run_id,
        "attempt_id": f"{run_id}-attempt1",
        "scientific_spec_sha256": scientific,
        "run_manifest_source": source,
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "final_step": 19999,
        "output_s3": output,
        "completion_claim_s3": completion,
        "estimated_train_seconds": 100,
        "environment": environment,
    }


def _campaign(code: Path, cells: list[dict], *, hardware: str = "p5e", policy: str = "stop") -> dict:
    campaign_id = "rmme-st-series-v1-0123456789abcdef0123"
    gate = "p5_training_only" if hardware == "p5e" else "p5_native_render_reset_claim"
    return _sealed(
        {
            "schema_version": 1,
            "kind": "robomme_single_task_train_series",
            "campaign_id": campaign_id,
            "attempt_id": f"{campaign_id}-attempt1",
            "campaign_scientific_sha256": "f" * 64,
            "infrastructure": {
                "provider": "aws_sagemaker",
                "hardware": hardware,
                "queue": "queue",
                "training_plan_arn": "plan" if hardware == "p5e" else None,
                "instance_type": "ml.p5e.48xlarge" if hardware == "p5e" else "ml.p5.48xlarge",
                "accelerator": "8xH200" if hardware == "p5e" else "8xH100",
                "priority": 400,
                "max_run_seconds": 1000,
                "volume_size_gb": 300,
            },
            "runtime_reserve_seconds": 600,
            "failure_policy": policy,
            "evaluation": {"mode": "deferred", "required_gate": gate, "reason": "test"},
            "cleanup": {
                "remove_cell_work_after_attempt": True,
                "minimum_free_bytes": 1,
                "durable_outputs": "test",
            },
            "cells": cells,
            "claims": {
                "manifest": "s3://bucket/campaign/manifest.json",
                "attempt_result": "s3://bucket/campaign/attempt.json",
                "completion": "s3://bucket/campaign/complete.json",
            },
        }
    )


def _complete(store: MemoryStore, cell: dict) -> None:
    checkpoint = f"{cell['output_s3']}/deploy/19999"
    tree = {"schema_version": 1, "checkpoint_uri": checkpoint, "objects": [{"key": "params/x"}]}
    tree_bytes = _canonical(tree)
    tree_sha = hashlib.sha256(tree_bytes).hexdigest()
    tree_uri = f"s3://bucket/trees/{tree_sha}.json"
    store.objects[tree_uri] = tree_bytes
    store.put_json_once(
        {
            "schema_version": 1,
            "kind": "robomme_gpu_deploy_checkpoint_complete",
            "run_id": cell["run_id"],
            "attempt_id": cell["attempt_id"],
            "scientific_spec_sha256": cell["scientific_spec_sha256"],
            "step": 19999,
            "checkpoint_uri": checkpoint,
            "tree_manifest_sha256": tree_sha,
            "run_manifest_sha256": cell["run_manifest_sha256"],
        },
        f"{checkpoint}/_DEPLOY_COMPLETE.json",
    )
    store.put_json_once(
        {
            "schema_version": 1,
            "kind": "robomme_gpu_checkpoint_complete",
            "run_id": cell["run_id"],
            "attempt_id": cell["attempt_id"],
            "scientific_spec_sha256": cell["scientific_spec_sha256"],
            "step": 19999,
            "checkpoint_uri": checkpoint,
            "tree_manifest_uri": tree_uri,
            "tree_manifest_sha256": tree_sha,
            "run_manifest_sha256": cell["run_manifest_sha256"],
        },
        cell["completion_claim_s3"],
    )


class StubRunner(CampaignRunner):
    def __init__(self, *args, results, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = results
        self.started: list[str] = []

    def _run_cell(self, cell: dict, cell_work: Path) -> int:
        self.started.append(cell["cell_id"])
        (cell_work / "large-ephemeral-checkpoint").write_bytes(b"temporary")
        returncode = self.results[cell["cell_id"]]
        if returncode == 0:
            _complete(self.store, cell)
        return returncode


class FalseSuccessRunner(StubRunner):
    def _run_cell(self, cell: dict, cell_work: Path) -> int:
        self.started.append(cell["cell_id"])
        return 0


def test_campaign_skips_exact_cell_runs_remaining_and_cleans_work(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\nexit 0\n")
    cells = [_cell(code, 0, "PickXtimes", "q1"), _cell(code, 1, "PickXtimes", "ptrm")]
    # Duplicate task/arm is forbidden, but the same task across different mechanisms is the
    # intended high-throughput anchor campaign.
    manifest = _campaign(code, cells)
    store = MemoryStore()
    _complete(store, cells[0])
    runner = StubRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=store,
        results={cells[1]["cell_id"]: 0},
    )
    assert runner.run() == 0
    assert runner.started == [cells[1]["cell_id"]]
    assert not any((tmp_path / "work" / "cells").iterdir())
    attempt = json.loads(store.objects[manifest["claims"]["attempt_result"]])
    assert [record["status"] for record in attempt["records"]] == [
        "skipped_exact_complete",
        "completed",
    ]
    assert manifest["claims"]["completion"] in store.objects


def test_campaign_continue_policy_runs_later_cell_but_never_seals_partial_completion(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\nexit 0\n")
    cells = [_cell(code, 0, "PickXtimes", "q1"), _cell(code, 1, "MoveCube", "ptrm")]
    manifest = _campaign(code, cells, policy="continue")
    store = MemoryStore()
    runner = StubRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=store,
        results={cells[0]["cell_id"]: 7, cells[1]["cell_id"]: 0},
    )
    assert runner.run() == 7
    assert runner.started == [cell["cell_id"] for cell in cells]
    assert manifest["claims"]["completion"] not in store.objects
    attempt = json.loads(store.objects[manifest["claims"]["attempt_result"]])
    assert [record["status"] for record in attempt["records"]] == ["failed", "completed"]


def test_zero_exit_without_durable_cell_claim_cannot_seal_campaign(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\nexit 0\n")
    cell = _cell(code, 0, "PickXtimes", "q1")
    manifest = _campaign(code, [cell])
    store = MemoryStore()
    runner = FalseSuccessRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=store,
        results={},
    )
    assert runner.run() == 1
    attempt = json.loads(store.objects[manifest["claims"]["attempt_result"]])
    assert attempt["records"][0]["status"] == "failed"
    assert "without a durable completion claim" in attempt["records"][0]["detail"]
    assert manifest["claims"]["completion"] not in store.objects


def test_sigterm_waits_for_cell_process_group_before_cleanup(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    ready = tmp_path / "child-ready"
    drained = tmp_path / "child-drained"
    child = code / "delayed_signal_child.py"
    child.write_text(
        "import os, pathlib, signal, time\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        f"drained = pathlib.Path({str(drained)!r})\n"
        "work = pathlib.Path(os.environ['ROBOMME_WORK_ROOT'])\n"
        "def stop(signum, _frame):\n"
        "    time.sleep(0.25)\n"
        "    drained.write_text('work_exists' if work.is_dir() else 'deleted')\n"
        "    raise SystemExit(128 + signum)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "ready.write_text('ready')\n"
        "while True:\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    (code / "gpu_train_entry.sh").write_text(f"#!/usr/bin/env bash\npython3 {child}\n", encoding="utf-8")
    cell = _cell(code, 0, "PickXtimes", "q1")
    manifest = _campaign(code, [cell])
    runner = CampaignRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=MemoryStore(),
    )

    def terminate_when_ready() -> None:
        deadline = time.monotonic() + 5
        while not ready.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError("cell child never became ready")
            time.sleep(0.01)
        runner._forward(signal.SIGTERM, None)

    terminator = threading.Thread(target=terminate_when_ready)
    terminator.start()
    try:
        assert runner.run() != 0
    finally:
        terminator.join(timeout=5)
    assert not terminator.is_alive()
    assert drained.read_text(encoding="utf-8") == "work_exists"
    assert not any((tmp_path / "work" / "cells").iterdir())


def test_unproven_process_group_drain_preserves_cell_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/usr/bin/env bash\nexit 7\n")
    cell = _cell(code, 0, "PickXtimes", "q1")
    manifest = _campaign(code, [cell])
    store = MemoryStore()
    monkeypatch.setattr(campaign_module, "_wait_for_process_group_drain", lambda _pgid: False)
    runner = CampaignRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=store,
    )

    assert runner.run() != 0
    preserved = list((tmp_path / "work" / "cells").iterdir())
    assert len(preserved) == 1 and preserved[0].is_dir()
    attempt = json.loads(store.objects[manifest["claims"]["attempt_result"]])
    assert attempt["records"][0]["status"] == "failed"
    assert "ChildProcessGroupNotDrained" in attempt["records"][0]["detail"]
    assert manifest["claims"]["completion"] not in store.objects


def test_completion_mismatch_fails_closed_instead_of_skipping(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\n")
    cell = _cell(code, 0, "PickXtimes", "q1")
    store = MemoryStore()
    _complete(store, cell)
    claim = json.loads(store.objects[cell["completion_claim_s3"]])
    claim["scientific_spec_sha256"] = "0" * 64
    store.objects[cell["completion_claim_s3"]] = _canonical(claim)
    with pytest.raises(ValueError, match="not exact"):
        verify_completion(store, cell)


def test_eval_is_explicitly_deferred_and_p5e_cannot_claim_native_eval(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\n")
    manifest = _campaign(code, [_cell(code, 0, "PickXtimes", "q1")])
    manifest["evaluation"] = {
        "mode": "inline",
        "required_gate": "p5_native_render_reset_claim",
    }
    manifest = _sealed({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError, match="same-node evaluation is not certified"):
        validate_manifest(manifest, code_dir=code)


def test_campaign_soft_deadline_defers_without_starting_or_claiming_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\nexit 0\n")
    cell = _cell(code, 0, "PickXtimes", "q1")
    manifest = _campaign(code, [cell])
    ticks = iter((0.0, 500.0))
    monkeypatch.setattr(campaign_module.time, "monotonic", lambda: next(ticks))
    store = MemoryStore()
    runner = StubRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=store,
        results={cell["cell_id"]: 0},
    )
    assert runner.run() == 0
    assert runner.started == []
    result = json.loads(store.objects[manifest["claims"]["attempt_result"]])
    assert result["deferred_runtime_budget"] is True
    assert result["records"][0]["status"] == "deferred_runtime_budget"
    assert manifest["claims"]["completion"] not in store.objects


def test_low_disk_evicts_reconstructible_artifacts_only_between_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpu_train_entry.sh").write_text("#!/bin/bash\nexit 0\n")
    cell = _cell(code, 0, "PickXtimes", "q1")
    manifest = _campaign(code, [cell])
    runner = StubRunner(
        manifest=manifest,
        code_dir=code,
        work_root=tmp_path / "work",
        store=MemoryStore(),
        results={cell["cell_id"]: 0},
    )
    (runner.work_root / "cache" / "dependency").write_bytes(b"cache")
    (runner.work_root / "artifacts" / "immutable").write_bytes(b"artifact")
    free = iter((0, 0, 2))
    monkeypatch.setattr(
        campaign_module.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=next(free)),
    )
    runner._maybe_clear_cache()
    assert not any((runner.work_root / "cache").iterdir())
    assert not any((runner.work_root / "artifacts").iterdir())


def test_shared_workspace_tree_reuse_rehashes_every_file(tmp_path: Path):
    cache = tmp_path / "cache"
    manifest_bytes = b'{"manifest_sha256":"fixture"}\n'
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    target = cache / "workspace" / manifest_sha
    target.mkdir(parents=True)
    (target / "MANIFEST.json").write_bytes(manifest_bytes)
    (target / "episode_0.npy").write_bytes(b"omega")
    identity = {
        "kind": "s3_manifest_tree",
        "uri": "s3://bucket/workspace",
        "manifest_sha256": manifest_sha,
        "category": "workspace",
    }
    marker = target.with_name(target.name + shared_artifacts.MARKER_SUFFIX)
    marker.write_bytes(_canonical(shared_artifacts._receipt(target, identity=identity)))
    assert (
        shared_artifacts.prepare_s3_tree("s3://bucket/workspace", manifest_sha, cache, category="workspace") == target
    )
    (target / "episode_0.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes drifted"):
        shared_artifacts.prepare_s3_tree("s3://bucket/workspace", manifest_sha, cache, category="workspace")


def test_planner_builds_one_priority400_h200_series_without_cloud_action():
    source = Path(__file__).resolve().parents[1]
    encoder = "a" * 64
    omega_sha = "b" * 64
    workspace = f"{launch_root()}/artifacts/robomme/workspace/PickXtimes/{encoder}/omega"
    spec = {
        "schema_version": 1,
        "name": "test-anchor",
        "hardware": "p5e",
        "priority": 400,
        "max_run_seconds": 86400,
        "volume_size_gb": 300,
        "estimated_train_seconds": 30000,
        "runtime_reserve_seconds": 3600,
        "failure_policy": "continue",
        "evaluation_mode": "deferred",
        "cells": [
            {
                "task": "PickXtimes",
                "arm": "q1",
                "workspace_encoder_id": encoder,
                "workspace_s3": workspace,
                "workspace_manifest_sha256": omega_sha,
            },
            {
                "task": "PickXtimes",
                "arm": "ptrm",
                "workspace_encoder_id": encoder,
                "workspace_s3": workspace,
                "workspace_manifest_sha256": omega_sha,
            },
        ],
    }
    plan = build_campaign_plan(spec, source)
    assert [cell["arm"] for cell in plan["manifest"]["cells"]] == ["q1", "ptrm"]
    assert plan["manifest"]["infrastructure"]["instance_type"] == "ml.p5e.48xlarge"
    assert plan["manifest"]["infrastructure"]["priority"] == 400
    assert plan["manifest"]["evaluation"] == {
        "mode": "deferred",
        "required_gate": "p5_training_only",
        "reason": "p5e is contractually training-only",
    }
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "0"
    assert len(plan["staged_source_files"]) == 3


def test_pick_mechanism_campaign_is_the_complete_single_node_screen():
    source = Path(__file__).resolve().parents[1]
    spec_path = source / "sweeps/p5e_pick_anchor_core_v3.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert [cell["arm"] for cell in spec["cells"]] == [
        "s0",
        "q0",
        "q1",
        "ptrm",
        "q0_noforce",
        "q2_noforce",
    ]
    plan = build_campaign_plan(spec, source)
    records = plan["manifest"]["cells"]
    by_arm = {cell["arm"]: cell for cell in records}
    assert len(by_arm) == 6
    assert plan["manifest"]["infrastructure"] == {
        "provider": "aws_sagemaker",
        "hardware": "p5e",
        "queue": "fss-tri-cam-robotics-p5e-48xlarge-us-west-2-training-plan",
        "training_plan_arn": ("arn:aws:sagemaker:us-west-2:141701954645:training-plan/cam-robotics-tp"),
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "priority": 400,
        "max_run_seconds": 86400,
        "volume_size_gb": 300,
    }
    assert by_arm["q0_noforce"]["environment"]["OPENPI_REQUIRED_SENTINEL"] == ("_ROBOMME_SEQUENCE_FORCING")
    assert by_arm["q2_noforce"]["environment"]["OPENPI_REQUIRED_SENTINEL"] == ("_ROBOMME_SEQUENCE_FORCING")
    assert by_arm["ptrm"]["environment"]["OPENPI_REQUIRED_SENTINEL"] == "_WSM_PTRM"
    assert sum(cell["estimated_train_seconds"] for cell in records) == 79_800


def test_campaign_launcher_is_dry_by_default_and_has_explicit_confirmation():
    from robomme_integration.campaign_launch import parser

    parsed = parser().parse_args(["--spec", "fixture.json"])
    assert parsed.confirm_submit is False
    assert "--confirm-submit" in parser().format_help()


def test_training_entry_reuses_a_campaign_scoped_jax_compilation_cache():
    source = Path(__file__).resolve().parents[1]
    entry = (source / "gpu_train_entry.sh").read_text(encoding="utf-8")
    assert 'JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax-compilation-cache"' in entry
    assert 'mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$JAX_COMPILATION_CACHE_DIR"' in entry
    assert entry.index('ROBOMME_COMPAT="$CODE_DIR/compat"') < entry.index(
        'TRAIN_PYTHONPATH="$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"'
    )


def test_shared_tokenizer_materializes_exactly_inside_each_serial_cell_cache(tmp_path: Path):
    payload = b"sealed-paligemma-tokenizer"
    digest = hashlib.sha256(payload).hexdigest()
    shared = tmp_path / "campaign" / "artifacts" / "blob" / digest
    shared.mkdir(parents=True)
    source = shared / "paligemma_tokenizer.model"
    source.write_bytes(payload)

    targets = []
    for ordinal in range(2):
        cache_home = tmp_path / "campaign" / "cells" / f"cell-{ordinal:03d}" / "openpi_cache"
        target = shared_artifacts.materialize_blob(
            source,
            digest,
            cache_home,
            relative_path="big_vision/paligemma_tokenizer.model",
        )
        assert target == cache_home / "big_vision" / "paligemma_tokenizer.model"
        assert target.is_file() and not target.is_symlink()
        assert target.resolve().is_relative_to(cache_home.resolve())
        assert target.read_bytes() == payload
        targets.append(target)

    # The normal same-volume campaign path consumes no second model copy.  Removing one cell's
    # ephemeral work must leave both the shared source and the later cell usable.
    assert source.stat().st_ino == targets[0].stat().st_ino == targets[1].stat().st_ino
    shutil.rmtree(targets[0].parents[2])
    assert source.read_bytes() == targets[1].read_bytes() == payload

    entry = (Path(__file__).resolve().parents[1] / "gpu_train_entry.sh").read_text(encoding="utf-8")
    assert "materialize-blob" in entry
    assert 'ln -s "$TOKENIZER_CACHE" "$TOKENIZER"' not in entry


def launch_root() -> str:
    # Local helper avoids making the test's scientific path a second independent constant.
    from robomme_integration.launch import STUDY_ROOT

    return STUDY_ROOT
