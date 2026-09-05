import hashlib
import json
from pathlib import Path

import pytest

from vla_training.eval.compare_exact_evals import (
    DEFAULT_SPLITS,
    _write_json_atomic,
    compare_exact_roots,
    task_cluster_ci,
)
from vla_training.eval.eval_manifest import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    POLICY_NOISE_KIND,
    seal_episode_manifest,
)


def _reset(task, index):
    descriptor = {"sha256": "a" * 64, "size": index + 1}
    return {
        "kind": "heldout_demo",
        "extras_relpath": f"{task}/extras/episode_{index:06d}",
        "source": "s3://immutable-test-source",
        "artifacts": {
            "ep_meta.json": descriptor,
            "model.xml.gz": descriptor,
            "states.npz": descriptor,
        },
    }


def _manifest(outcomes, base_seed=17):
    counts = {len(values) for tasks in outcomes.values() for values in tasks.values()}
    assert len(counts) == 1
    episodes_per_task = counts.pop()
    episodes = []
    for split in DEFAULT_SPLITS:
        for task, values in outcomes[split].items():
            for index in range(len(values)):
                episodes.append(
                    {
                        "task": task,
                        "split_set": split,
                        "horizon": 20,
                        "episode_index": index,
                        "reset": _reset(task, index),
                        "seed": base_seed * 100_000 + sum(map(ord, task)) * 100 + index,
                    }
                )
    return seal_episode_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "split": "target",
            "task_sets": list(DEFAULT_SPLITS),
            "base_seed": base_seed,
            "policy_noise": {
                "kind": POLICY_NOISE_KIND,
                "key_fields": ["episode.seed", "env_step"],
            },
            "episodes_per_task": episodes_per_task,
            "episodes": episodes,
        }
    )


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


def _provenance(root: Path, manifest):
    baseline = root.name == "base"
    arm, interface = ("s0", "base") if baseline else ("s1", "tanh")
    identity = hashlib.sha256(root.name.encode()).hexdigest()
    return {
        "schema_version": 1,
        "kind": "pi_stage_s_evaluation_provenance",
        "eval_run_id": f"eval-{arm}-step59999-{identity[:16]}",
        "eval_manifest_sha256": identity,
        "arm": arm,
        "interface": interface,
        "training_run_id": f"{arm}-{identity[:16]}",
        "training_manifest_sha256": hashlib.sha256(f"train-{root.name}".encode()).hexdigest(),
        "checkpoint_uri": f"s3://test/checkpoints/{arm}/{identity[:16]}/59999",
        "checkpoint_step": 59999,
        "checkpoint_tree_manifest_sha256": hashlib.sha256(f"tree-{root.name}".encode()).hexdigest(),
        "episode_manifest_sha256": manifest["manifest_sha256"],
        "episode_manifest_file_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }


def _write_arm(root: Path, manifest, outcomes):
    root.mkdir()
    provenance = _provenance(root, manifest)
    records = {}
    for episode in manifest["episodes"]:
        records.setdefault(episode["task"], []).append(episode)
    splits = {}
    all_rates = []
    total_episodes = 0
    for split in DEFAULT_SPLITS:
        per_task, per_task_counts = {}, {}
        for task, values in outcomes[split].items():
            episodes = [
                {
                    "task": task,
                    "episode_index": spec["episode_index"],
                    "reset": spec["reset"],
                    "seed": spec["seed"],
                    "success": success,
                    "episode_length": 10,
                }
                for spec, success in zip(records[task], values, strict=True)
            ]
            rate = sum(values) / len(values)
            path = root / split / task / "stats.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "task": task,
                        "split_set": split,
                        "split": "target",
                        "manifest_sha256": manifest["manifest_sha256"],
                        "evaluation_provenance": provenance,
                        "horizon": 20,
                        "seed": manifest["base_seed"],
                        "num_episode_shards": 1,
                        "num_episodes": len(values),
                        "success_rate": rate,
                        "successes": values,
                        "episode_lengths": [10] * len(values),
                        "per_episode": episodes,
                    }
                )
            )
            per_task[task], per_task_counts[task] = rate, len(values)
            all_rates.append(rate)
            total_episodes += len(values)
        splits[split] = {
            "mean": sum(per_task.values()) / len(per_task),
            "n_tasks_done": len(per_task),
            "n_tasks_expected": len(per_task),
            "n_episodes_done": sum(per_task_counts.values()),
            "per_task": per_task,
            "per_task_num_episodes": per_task_counts,
        }
    (root / "results.json").write_text(
        json.dumps(
            {
                "protocol": "exact_manifest",
                "complete": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "evaluation_provenance": provenance,
                "model": root.name,
                "step": 59999,
                "split": "target",
                "num_trials": manifest["episodes_per_task"],
                "n_tasks_done": len(all_rates),
                "n_episodes_done": total_episodes,
                "avg_task_weighted": sum(all_rates) / len(all_rates),
                "splits": splits,
            }
        )
    )


def _small_outcomes():
    return {
        "atomic_seen": {"TaskA": [False, True]},
        "composite_seen": {"TaskB": [False, False]},
        "composite_unseen": {"TaskC": [True, True]},
    }


def _candidate_outcomes():
    return {
        "atomic_seen": {"TaskA": [True, True]},
        "composite_seen": {"TaskB": [False, True]},
        "composite_unseen": {"TaskC": [True, False]},
    }


def _compare(tmp_path, baseline_values=None, candidate_values=None, samples=200):
    baseline_values = baseline_values or _small_outcomes()
    candidate_values = candidate_values or _candidate_outcomes()
    manifest = _manifest(baseline_values)
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    baseline, candidate = tmp_path / "base", tmp_path / "candidate"
    _write_arm(baseline, manifest, baseline_values)
    _write_arm(candidate, manifest, candidate_values)
    result = compare_exact_roots(
        baseline,
        candidate,
        episode_manifest=manifest_path,
        bootstrap_samples=samples,
        bootstrap_seed=7,
        expected_tasks=3,
        expected_episodes_per_task=2,
    )
    return result, baseline, candidate, manifest_path


def test_paired_task_macro_splits_discordance_and_target(tmp_path):
    result, baseline, candidate, manifest_path = _compare(tmp_path)
    overall = result["overall"]
    assert overall["baseline_task_macro"] == pytest.approx(0.5)
    assert overall["candidate_task_macro"] == pytest.approx(2 / 3)
    assert overall["paired_delta_absolute"] == pytest.approx(1 / 6)
    assert overall["relative_lift"] == pytest.approx(1 / 3)
    assert overall["candidate_better_episodes"] == 2
    assert overall["baseline_better_episodes"] == 1
    assert result["relative_lift_target"]["threshold"] == 0.30
    assert result["relative_lift_target"]["point_estimate_met"] is True
    assert [result["splits"][split]["label"] for split in DEFAULT_SPLITS] == ["A", "CS", "CUS"]
    assert result["bootstrap"]["unit"] == "paired_task_cluster"
    assert len(result["comparison_sha256"]) == 64
    repeated = compare_exact_roots(
        baseline,
        candidate,
        episode_manifest=manifest_path,
        bootstrap_samples=200,
        bootstrap_seed=7,
        expected_tasks=3,
        expected_episodes_per_task=2,
    )
    assert repeated["overall"]["paired_delta_task_cluster_ci95"] == overall["paired_delta_task_cluster_ci95"]
    assert repeated["overall"]["relative_lift_task_cluster_ci95"] == overall["relative_lift_task_cluster_ci95"]


def test_decisive_50_by_100_contract(tmp_path):
    baseline, candidate = {}, {}
    for split, count, prefix in zip(DEFAULT_SPLITS, (18, 16, 16), ("A", "CS", "CUS"), strict=True):
        baseline[split], candidate[split] = {}, {}
        for index in range(count):
            task = f"{prefix}_Task_{index:02d}"
            baseline[split][task] = [episode < 25 for episode in range(100)]
            candidate[split][task] = [episode < 40 for episode in range(100)]
    manifest = _manifest(baseline)
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    base_root, cand_root = tmp_path / "base", tmp_path / "candidate"
    _write_arm(base_root, manifest, baseline)
    _write_arm(cand_root, manifest, candidate)
    result = compare_exact_roots(base_root, cand_root, episode_manifest=manifest_path, bootstrap_samples=25)
    assert result["overall"]["n_tasks"] == 50
    assert result["overall"]["n_episodes"] == 5000
    assert result["overall"]["relative_lift"] == pytest.approx(0.6)
    assert result["manifest"]["split_task_counts"] == dict(zip(DEFAULT_SPLITS, (18, 16, 16)))
    assert result["relative_lift_target"]["ci95_lower_bound_supports_target"] is True


def test_rejects_different_manifest(tmp_path):
    values = _small_outcomes()
    first, second = _manifest(values, 17), _manifest(values, 18)
    manifest_path = _write_manifest(tmp_path / "manifest.json", first)
    baseline, candidate = tmp_path / "base", tmp_path / "candidate"
    _write_arm(baseline, first, values)
    _write_arm(candidate, second, values)
    with pytest.raises(ValueError, match="different episode manifest"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )


def test_rejects_identity_not_in_manifest(tmp_path):
    _, baseline, candidate, manifest_path = _compare(tmp_path)
    path = candidate / "atomic_seen" / "TaskA" / "stats.json"
    payload = json.loads(path.read_text())
    payload["per_episode"][1]["reset"]["extras_relpath"] += "_wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identities/order differ"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )


def test_rejects_same_result_root(tmp_path):
    _, baseline, _, manifest_path = _compare(tmp_path)
    with pytest.raises(ValueError, match="must be different"):
        compare_exact_roots(
            baseline,
            baseline,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )


def test_rejects_stat_copied_from_another_checkpoint(tmp_path):
    _, baseline, candidate, manifest_path = _compare(tmp_path)
    source = baseline / "atomic_seen" / "TaskA" / "stats.json"
    destination = candidate / "atomic_seen" / "TaskA" / "stats.json"
    destination.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="evaluation_provenance differs"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )


def test_rejects_duplicate_immutable_eval_identity_across_different_roots(tmp_path):
    _, baseline, candidate, manifest_path = _compare(tmp_path)
    baseline_results = json.loads((baseline / "results.json").read_text())
    candidate_results = json.loads((candidate / "results.json").read_text())
    candidate_results["evaluation_provenance"] = baseline_results["evaluation_provenance"]
    (candidate / "results.json").write_text(json.dumps(candidate_results))
    with pytest.raises(ValueError, match="same immutable evaluation identity"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )


def test_rejects_partial_mixed_and_summary_mismatch(tmp_path):
    _, baseline, candidate, manifest_path = _compare(tmp_path)
    results = json.loads((candidate / "results.json").read_text())
    results["complete"] = False
    (candidate / "results.json").write_text(json.dumps(results))
    with pytest.raises(ValueError, match="not complete"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )

    results["complete"] = True
    results["splits"]["atomic_seen"]["per_task"]["TaskA"] = 0.25
    (candidate / "results.json").write_text(json.dumps(results))
    with pytest.raises(ValueError, match="exact records imply"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )

    results["splits"]["atomic_seen"]["per_task"]["TaskA"] = 1.0
    (candidate / "results.json").write_text(json.dumps(results))
    extra = candidate / "atomic_seen" / "StrayTask" / "stats.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}")
    with pytest.raises(ValueError, match="inexact/mixed"):
        compare_exact_roots(
            baseline,
            candidate,
            episode_manifest=manifest_path,
            bootstrap_samples=10,
            expected_tasks=3,
            expected_episodes_per_task=2,
        )


def test_cluster_bootstrap_is_deterministic():
    values = {"a": -0.1, "b": 0.2, "c": 0.4}
    assert task_cluster_ci(values, samples=1000, seed=9) == task_cluster_ci(values, samples=1000, seed=9)


def test_zero_baseline_reports_relative_as_undefined(tmp_path):
    zeros = {
        split: {task: [False, False] for task in tasks}
        for split, tasks in {
            "atomic_seen": ["TaskA"],
            "composite_seen": ["TaskB"],
            "composite_unseen": ["TaskC"],
        }.items()
    }
    result, _, _, _ = _compare(tmp_path, zeros, zeros, samples=20)
    assert result["overall"]["relative_lift"] is None
    assert result["overall"]["relative_lift_task_cluster_ci95"] is None
    assert result["relative_lift_target"]["point_estimate_met"] is None


def test_atomic_writer_replaces_complete_json(tmp_path):
    path = tmp_path / "comparison.json"
    path.write_text('{"old": true}')
    _write_json_atomic(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    assert not list(tmp_path.glob(".comparison.json.*.tmp"))
