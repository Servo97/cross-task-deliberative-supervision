"""Correctness tests for immutable RoboCasa episode manifests and shard merging."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vla_training.eval.eval_manifest import (
    build_episode_manifest,
    build_heldout_episode_manifest,
    build_shard_results,
    episode_identity,
    evaluation_provenance_from_run_manifest,
    manifest_records_for_task,
    merge_task_episode_shards,
    output_lock,
    policy_noise_seed,
    sanitize_policy_timing,
    shard_episode_records,
    shard_stats_path,
    summarize_policy_performance,
    validate_evaluation_provenance,
    validate_shard_results,
    write_episode_manifest,
    write_json_atomic,
)


def _tasks(count: int = 2) -> list[dict]:
    return [
        {
            "task": f"Task{index:02d}",
            "split_set": "atomic_seen" if index % 2 == 0 else "composite_seen",
            "horizon": 100 + index,
        }
        for index in range(count)
    ]


def _result(record: dict) -> dict:
    episode_index = int(record["episode_index"])
    return {
        "task": record["task"],
        "episode_index": episode_index,
        "reset": record["reset"],
        "seed": int(record["seed"]),
        "success": episode_index % 2 == 0,
        "episode_length": 10 + episode_index,
    }


def _provenance(manifest, arm="s1"):
    # Keep this helper in lockstep with eval_manifest.validate_evaluation_provenance.
    interface = {
        "s0": "base",
        "s1": "tanh",
        "s2": "cfg2",
        "s3": "base",
        "q0": "base",
        "q1": "tanh",
        "q2": "robottt_fast",
        "q3": "tanh_robottt",
    }[arm]
    return {
        "schema_version": 1,
        "kind": "pi_stage_s_evaluation_provenance",
        "eval_run_id": f"eval-{arm}-step59999-{'a' * 16}",
        "eval_manifest_sha256": "a" * 64,
        "arm": arm,
        "interface": interface,
        "training_run_id": f"{arm}-{'b' * 16}",
        "training_manifest_sha256": "b" * 64,
        "checkpoint_uri": f"s3://test/checkpoints/{arm}/59999",
        "checkpoint_step": 59999,
        "checkpoint_tree_manifest_sha256": "c" * 64,
        "episode_manifest_sha256": manifest["manifest_sha256"],
        "episode_manifest_file_sha256": "d" * 64,
    }


@pytest.mark.parametrize(
    ("arm", "interface"),
    (
        ("s0", "base"),
        ("s1", "tanh"),
        ("s2", "cfg2"),
        ("s3", "base"),
        ("q0", "base"),
        ("q1", "tanh"),
        ("q2", "robottt_fast"),
        ("q3", "tanh_robottt"),
    ),
)
def test_every_arm_pins_exactly_one_serve_interface(arm, interface):
    manifest = {"manifest_sha256": "9" * 64}
    provenance = _provenance(manifest, arm=arm)
    assert provenance["interface"] == interface
    assert validate_evaluation_provenance(provenance) == provenance
    # Any other interface for the same arm refuses — incl. the whack-a-mole classics: a q3 shard
    # claiming the workspace-free robottt_fast serve, or a q1 shard claiming base.
    for wrong in ("base", "tanh", "cfg2", "robottt_fast", "tanh_robottt"):
        if wrong == interface:
            continue
        with pytest.raises(ValueError, match="arm/interface mismatch"):
            validate_evaluation_provenance({**provenance, "interface": wrong})


def test_unknown_arm_refuses_any_interface():
    manifest = {"manifest_sha256": "9" * 64}
    provenance = _provenance(manifest, arm="q3")
    with pytest.raises(ValueError, match="arm/interface mismatch"):
        validate_evaluation_provenance({**provenance, "arm": "q4"})


def _write_complete_shards(tmp_path, manifest, task_entry, num_shards, evaluation_provenance=None):
    task = task_entry["task"]
    records = manifest_records_for_task(manifest, task)
    for shard_idx in range(num_shards):
        assigned = shard_episode_records(records, shard_idx, num_shards)
        payload = build_shard_results(
            manifest,
            task_entry["split_set"],
            task,
            shard_idx,
            num_shards,
            [_result(record) for record in assigned],
            complete=True,
            wall_seconds=1.25,
            evaluation_provenance=evaluation_provenance,
        )
        validate_shard_results(
            payload,
            manifest,
            task_entry["split_set"],
            task,
            shard_idx,
            num_shards,
            require_complete=True,
            evaluation_provenance=evaluation_provenance,
        )
        write_json_atomic(
            shard_stats_path(
                tmp_path,
                task_entry["split_set"],
                task,
                shard_idx,
                num_shards,
            ),
            payload,
        )


def test_50_by_100_manifest_and_trial_shards_have_exact_stable_coverage():
    tasks = _tasks(50)
    first = build_episode_manifest(tasks, 100, 7, split="target", task_sets=["synthetic"])
    second = build_episode_manifest(tasks, 100, 7, split="target", task_sets=["synthetic"])
    assert first == second
    assert len(first["episodes"]) == 5000
    assert len({episode_identity(record) for record in first["episodes"]}) == 5000
    assert first["policy_noise"]["kind"] == "pi_diffusion_sha256_v1"
    assert policy_noise_seed(123, 8) == policy_noise_seed(123, 8)
    assert policy_noise_seed(123, 8) != policy_noise_seed(123, 16)

    for task in tasks:
        records = manifest_records_for_task(first, task["task"])
        shards = [shard_episode_records(records, shard_idx, 8) for shard_idx in range(8)]
        flattened = [record for shard in shards for record in shard]
        assert len(flattened) == 100
        assert {episode_identity(record) for record in flattened} == {episode_identity(record) for record in records}
        for shard_idx, shard in enumerate(shards):
            assert all(record["episode_index"] % 8 == shard_idx for record in shard)


def test_heldout_manifest_selects_exactly_100_distinct_complement_demos(tmp_path):
    task = _tasks(1)[0]
    task_root = tmp_path / task["task"]
    task_root.mkdir()
    episode_indices = list(range(200, 305))
    (task_root / "heldout.json").write_text(
        json.dumps(
            {
                "task": task["task"],
                "episodes": episode_indices,
                "num_train": 150,
                "seed": 0,
                "source": "s3://pinned-dataset/task/lerobot",
            }
        )
    )
    for episode_index in episode_indices:
        extras = task_root / "extras" / f"episode_{episode_index:06d}"
        extras.mkdir(parents=True)
        for filename in ("ep_meta.json", "model.xml.gz", "states.npz"):
            (extras / filename).touch()

    first = build_heldout_episode_manifest([task], tmp_path, 100, 7, split="target")
    second = build_heldout_episode_manifest([task], tmp_path, 100, 7, split="target")
    assert first == second
    records = manifest_records_for_task(first, task["task"])
    assert len(records) == 100
    assert len({record["episode_index"] for record in records}) == 100
    assert {record["episode_index"] for record in records} < set(episode_indices)
    assert all(record["reset"]["kind"] == "heldout_demo" for record in records)
    assert all(
        set(record["reset"]["artifacts"]) == {"ep_meta.json", "model.xml.gz", "states.npz"} for record in records
    )
    assert all(
        record["reset"]["extras_relpath"] == f"{task['task']}/extras/episode_{record['episode_index']:06d}"
        for record in records
    )
    shards = [shard_episode_records(records, index, 8) for index in range(8)]
    assert sorted(len(shard) for shard in shards) == [12] * 4 + [13] * 4
    assert {episode_identity(record) for shard in shards for record in shard} == {
        episode_identity(record) for record in records
    }


def test_exact_reset_artifacts_are_verified_before_loading(tmp_path):
    from vla_training.eval.heldout_reset import load_episode_state

    extras = tmp_path / "episode_000001"
    extras.mkdir()
    (extras / "ep_meta.json").write_text(json.dumps({"lang": "test"}))
    with gzip.open(extras / "model.xml.gz", "wt") as handle:
        handle.write("<mujoco/>")
    np.savez(extras / "states.npz", states=np.arange(6).reshape(2, 3))

    artifacts = {}
    for filename in ("ep_meta.json", "model.xml.gz", "states.npz"):
        data = (extras / filename).read_bytes()
        artifacts[filename] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
    state = load_episode_state(extras, artifacts=artifacts)
    assert state["ep_meta"] == {"lang": "test"}
    assert state["model_xml"] == "<mujoco/>"
    assert state["state0"].tolist() == [0, 1, 2]

    (extras / "ep_meta.json").write_text(json.dumps({"lang": "tampered"}))
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_episode_state(extras, artifacts=artifacts)


def test_manifest_publication_is_immutable_and_idempotent(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = build_episode_manifest(_tasks(1), 3, 7)
    write_episode_manifest(path, manifest)
    write_episode_manifest(path, manifest)
    assert json.loads(path.read_text()) == manifest

    different = build_episode_manifest(_tasks(1), 3, 8)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_episode_manifest(path, different)
    assert json.loads(path.read_text()) == manifest


def test_eval_provenance_is_derived_from_sealed_run_and_exact_episode_file(tmp_path):
    manifest = build_episode_manifest(_tasks(1), 3, 7)
    episode_path = tmp_path / "episodes.json"
    write_episode_manifest(episode_path, manifest)
    episode_file_sha = hashlib.sha256(episode_path.read_bytes()).hexdigest()
    run = {
        "schema_version": 1,
        "kind": "pi_stage_s_robocasa_eval_run",
        "arm": "s2",
        "interface": "cfg2",
        "eval_run_id": f"eval-s2-step59999-{'a' * 16}",
        "training_run": {
            "run_id": f"s2-{'b' * 16}",
            "manifest_sha256": "b" * 64,
            "checkpoint_uri": "s3://test/checkpoints/s2/run/59999",
            "checkpoint_step": 59999,
            "checkpoint_tree_manifest": {"file_sha256": "c" * 64},
        },
        "protocol": {
            "episode_manifest": {"file_sha256": episode_file_sha},
        },
    }
    run["manifest_sha256"] = hashlib.sha256(
        json.dumps(run, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    run_path = tmp_path / "eval.json"
    run_path.write_text(json.dumps(run))

    provenance = evaluation_provenance_from_run_manifest(run_path, manifest, episode_manifest_path=episode_path)
    assert provenance["arm"] == "s2"
    assert provenance["training_run_id"] == run["training_run"]["run_id"]
    assert provenance["checkpoint_tree_manifest_sha256"] == "c" * 64
    assert provenance["episode_manifest_sha256"] == manifest["manifest_sha256"]
    assert provenance["episode_manifest_file_sha256"] == episode_file_sha

    run["training_run"]["checkpoint_uri"] += "-tampered"
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="seal mismatch"):
        evaluation_provenance_from_run_manifest(run_path, manifest, episode_manifest_path=episode_path)


def test_merge_publishes_only_after_exact_coverage(tmp_path):
    task = _tasks(1)[0]
    manifest = build_episode_manifest([task], 7, 11)
    _write_complete_shards(tmp_path, manifest, task, num_shards=3)

    merged = merge_task_episode_shards(tmp_path, manifest, task["split_set"], task["task"], 3)
    assert merged["num_episodes"] == 7
    assert [item["episode_index"] for item in merged["per_episode"]] == list(range(7))
    assert merged["success_rate"] == pytest.approx(4 / 7)
    assert merged["manifest_sha256"] == manifest["manifest_sha256"]
    assert merged["wall_seconds_kind"] == "aggregate_shard_worker_seconds"
    assert merged["performance"]["rollouts_per_hour"] is None
    assert merged["performance"]["throughput_scope"] == "unavailable_without_shared_rollout_wall_clock"


def test_merge_rejects_missing_coverage_without_publishing_stats(tmp_path):
    task = _tasks(1)[0]
    manifest = build_episode_manifest([task], 7, 11)
    _write_complete_shards(tmp_path, manifest, task, num_shards=3)

    path = shard_stats_path(tmp_path, task["split_set"], task["task"], 1, 3)
    with open(path) as handle:
        payload = json.load(handle)
    payload["per_episode"].pop()
    payload["num_episodes"] -= 1
    payload["performance"] = summarize_policy_performance(payload["per_episode"], payload["wall_seconds"])
    write_json_atomic(path, payload)

    with pytest.raises(ValueError, match="inexact shard coverage"):
        merge_task_episode_shards(tmp_path, manifest, task["split_set"], task["task"], 3)
    assert not (tmp_path / task["split_set"] / task["task"] / "stats.json").exists()


def test_merge_rejects_shard_from_another_eval_or_checkpoint(tmp_path):
    task = _tasks(1)[0]
    manifest = build_episode_manifest([task], 7, 11)
    provenance = _provenance(manifest)
    _write_complete_shards(
        tmp_path,
        manifest,
        task,
        num_shards=3,
        evaluation_provenance=provenance,
    )
    path = Path(shard_stats_path(tmp_path, task["split_set"], task["task"], 1, 3))
    payload = json.loads(path.read_text())
    payload["evaluation_provenance"]["checkpoint_tree_manifest_sha256"] = "e" * 64
    write_json_atomic(path, payload)

    with pytest.raises(ValueError, match="does not match this eval run"):
        merge_task_episode_shards(
            tmp_path,
            manifest,
            task["split_set"],
            task["task"],
            3,
            evaluation_provenance=provenance,
        )
    assert not (tmp_path / task["split_set"] / task["task"] / "stats.json").exists()


def test_duplicate_or_wrong_shard_episode_is_rejected(tmp_path):
    task = _tasks(1)[0]
    manifest = build_episode_manifest([task], 5, 11)
    records = manifest_records_for_task(manifest, task["task"])
    assigned = shard_episode_records(records, 0, 2)
    payload = build_shard_results(
        manifest,
        task["split_set"],
        task["task"],
        0,
        2,
        [_result(assigned[0]), _result(assigned[0])],
        complete=False,
    )
    with pytest.raises(ValueError, match="duplicate episode result"):
        validate_shard_results(
            payload,
            manifest,
            task["split_set"],
            task["task"],
            0,
            2,
            require_complete=False,
        )

    wrong = build_shard_results(
        manifest,
        task["split_set"],
        task["task"],
        0,
        2,
        [_result(shard_episode_records(records, 1, 2)[0])],
        complete=False,
    )
    with pytest.raises(ValueError, match="not assigned"):
        validate_shard_results(
            wrong,
            manifest,
            task["split_set"],
            task["task"],
            0,
            2,
            require_complete=False,
        )


def test_output_lock_rejects_two_writers_for_same_shard(tmp_path):
    output = tmp_path / "split" / "task" / "stats_shard0of2.json"
    with output_lock(output):
        with pytest.raises(RuntimeError, match="another process owns"):
            with output_lock(output):
                pass


def test_policy_timing_sanitizer_is_closed_typed_and_prefers_explicit_metrics():
    sanitized = sanitize_policy_timing(
        {
            "infer_ms": 999.0,
            "policy_model_amortized_ms": 1.25,
            "policy_call_amortized_ms": 2,
            "wsm_tap_amortized_ms": 0.5,
            "wsm_encoder_amortized_ms": 0.75,
            "wsm_prepare_amortized_ms": 1.5,
            "wsm_end_to_end_amortized_ms": 3.0,
            "gather_ms": 4.0,
            "client_roundtrip_ms": 5.0,
            "gather_batch_n": 4,
            "policy_model_batch_n": 4.0,
            "policy_model_bucket_n": 8,
            "wsm_request_batch_n": 4,
            "wsm_new_grid_batch_n": 0,
            "typo_ms": 123,
            "server_request_ms": "6.0",
            "bad_bool_ms": True,
        }
    )
    assert "infer_ms" not in sanitized
    assert "typo_ms" not in sanitized
    assert "server_request_ms" not in sanitized
    assert "bad_bool_ms" not in sanitized
    assert sanitized["policy_model_amortized_ms"] == 1.25
    assert sanitized["policy_call_amortized_ms"] == 2.0
    assert sanitized["gather_batch_n"] == 4
    assert sanitized["policy_model_batch_n"] == 4
    assert sanitized["policy_model_bucket_n"] == 8
    assert sanitized["wsm_new_grid_batch_n"] == 0
    assert sanitized["gather_request_ms"] == 4.0
    assert "gather_ms" not in sanitized

    for bad in (True, "1", -1, float("nan"), float("inf")):
        assert sanitize_policy_timing({"client_roundtrip_ms": bad}) == {}
    for bad in (True, "1", 0, 1.5, float("nan"), float("inf")):
        assert sanitize_policy_timing({"gather_batch_n": bad}) == {}
    assert sanitize_policy_timing({"infer_ms": 7}) == {"infer_ms": 7.0}


def test_policy_performance_percentiles_and_request_weighted_batch_semantics():
    summary = summarize_policy_performance(
        [{"policy_timing_calls": [{"client_roundtrip_ms": value} for value in (1, 2, 3, 4)]}],
        wall_seconds=2.0,
    )
    assert summary["policy_calls"] == 4
    assert summary["latency_ms"]["client_roundtrip_ms"] == {
        "count": 4,
        "mean": 2.5,
        "p50": 2.5,
        "p95": 3.85,
        "max": 4.0,
    }
    assert summary["rollouts_per_hour"] == 1800.0

    # One singleton, one batch of two, and one batch of four: every response carries its realized N.
    sizes = [1, 2, 2, 4, 4, 4, 4]
    batch_summary = summarize_policy_performance(
        [{"policy_timing_calls": [{"gather_batch_n": size} for size in sizes]}],
        wall_seconds=1.0,
    )
    batching = batch_summary["batching"]
    assert batching["histogram"] == {"1": 1, "2": 2, "4": 4}
    assert batching["requests_observed"] == 7
    assert batching["request_weighted_mean"] == pytest.approx(3.0)
    assert batching["multi_request_fraction"] == pytest.approx(6 / 7, abs=1e-6)
    assert batching["effective_batch_size"] == pytest.approx(7 / 3, abs=1e-3)
    assert batching["weighting"] == "per_request"
    assert batching["by_stage"]["gather_batch_n"] == {
        key: value for key, value in batching.items() if key != "by_stage"
    }
    assert summarize_policy_performance([{"policy_timing_calls": [{}, {"typo_ms": 1}]}], 1.0)["policy_calls"] == 0


def test_shard_validation_rejects_tampered_timing_summary():
    task = _tasks(1)[0]
    manifest = build_episode_manifest([task], 1, 11)
    record = manifest_records_for_task(manifest, task["task"])[0]
    result = _result(record)
    result["policy_timing_calls"] = [{"client_roundtrip_ms": 10.0, "gather_batch_n": 1}]
    payload = build_shard_results(
        manifest,
        task["split_set"],
        task["task"],
        0,
        1,
        [result],
        complete=True,
        wall_seconds=2.25,
    )
    validate_shard_results(
        payload,
        manifest,
        task["split_set"],
        task["task"],
        0,
        1,
        require_complete=True,
    )
    payload["performance"]["policy_calls"] += 1
    with pytest.raises(ValueError, match="performance summary"):
        validate_shard_results(
            payload,
            manifest,
            task["split_set"],
            task["task"],
            0,
            1,
            require_complete=True,
        )
