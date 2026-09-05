"""The second-dataset knobs on the Stage-S chain (added for ReMemBench), and the target50 guard.

The chain was written for exactly one dataset: RoboCasa target50, 50 tasks nested under
``atomic``/``composite``, a uniform 150 seed-0 demos per task. ReMemBench is 13 tasks in a FLAT
``<task>/<date>/lerobot`` layout with a different demo count per task (9..44) and every demo kept.
Three knobs express that — ``dataset_name``, ``task_dir_globs``, ``demos_per_task`` (int OR
``{task: count}``) — and every default is still the target50 contract.

The load-bearing property tested here: switching ``dataset_name`` MUST move ``encoder_id``. The
same frozen encoder weights are reused across both datasets, so if the dataset did not enter
``encoder_provenance`` the two omega caches would hash to the same id and collide on the
create-once ``$STUDY/caches/<encoder_id>/omega`` prefix. RoboCasa's already-published six-key
provenance must NOT gain the block (that would retroactively rename a shipped artifact).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.launch import stage_s_provenance as prov
from scripts.launch import validate_stage_s_policy_features as omega_val
from scripts.launch import validate_stage_s_source_features as sfv
from scripts.launch import validate_stage_s_task_prompts as prompt_val

FLAT_GLOBS = ("*/*/lerobot",)
RMB_NAME = "remembench_v02_train13"


# --------------------------------------------------------------------------- unit-level knobs
def test_demos_for_task_accepts_int_and_mapping():
    assert omega_val.demos_for_task(150, "AnyTask") == 150
    assert omega_val.demos_for_task({"A": 9, "B": 44}, "B") == 44


@pytest.mark.parametrize(
    "value,task",
    [({"A": 9}, "B"), ({"A": 0}, "A"), ({"A": -1}, "A"), ({"A": True}, "A"), (0, "A")],
)
def test_demos_for_task_rejects_bad_counts(value, task):
    with pytest.raises(ValueError):
        omega_val.demos_for_task(value, task)


def test_canonical_demos_per_task_normalizes():
    assert omega_val.canonical_demos_per_task(150) == 150
    assert omega_val.canonical_demos_per_task({"B": 2, "A": 1}) == {"A": 1, "B": 2}


def test_load_demos_per_task_map(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"A": 9, "B": 44}), encoding="utf-8")
    assert omega_val.load_demos_per_task_map(path) == {"A": 9, "B": 44}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"A": "nine"}), encoding="utf-8")
    with pytest.raises(ValueError):
        omega_val.load_demos_per_task_map(bad)


def test_provenance_includes_dataset_only_for_non_target50():
    assert omega_val.provenance_includes_dataset(RMB_NAME) is True
    assert omega_val.provenance_includes_dataset(omega_val.DATASET_NAME) is False


def test_flat_and_nested_task_globs(tmp_path):
    for kind, task in (("atomic", "RcTask"), ("composite", "RcComposite")):
        (tmp_path / "rc" / kind / task / "20260101" / "lerobot").mkdir(parents=True)
    (tmp_path / "flat" / "MemHeatPot" / "20260803" / "lerobot").mkdir(parents=True)
    (tmp_path / "flat" / "MemWashAndReturnLeft" / "20260803" / "lerobot").mkdir(parents=True)

    assert set(omega_val._dataset_task_dirs(tmp_path / "rc")) == {"RcTask", "RcComposite"}
    # the default globs see nothing in a flat tree — a layout mismatch fails loud, not silently
    assert omega_val._dataset_task_dirs(tmp_path / "flat") == {}
    flat = omega_val._dataset_task_dirs(tmp_path / "flat", task_dir_globs=FLAT_GLOBS)
    assert set(flat) == {"MemHeatPot", "MemWashAndReturnLeft"}
    assert flat["MemHeatPot"].parents[1].name == "MemHeatPot"


# ---------------------------------------------------------------- a flat, ragged-count fixture
COUNTS = {"MemAlpha": 2, "MemBeta": 3, "MemGamma": 1}
FRAME_STRIDE = sfv.FRAME_STRIDE


def _flat_dataset(root: Path) -> None:
    """A flat ReMemBench-shaped target tree; every task keeps EVERY episode it has."""
    for task, count in COUNTS.items():
        dataset = root / task / "20260803" / "lerobot"
        (dataset / "meta").mkdir(parents=True)
        with (dataset / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode_index in range(count):
                stream.write(json.dumps({"episode_index": episode_index, "tasks": [f"do {task}"]}) + "\n")


def _source_cache(features_root: Path, prompt_id: str, *, dataset_name: str) -> tuple[Path, dict]:
    rng = np.random.default_rng(0)
    task_records = []
    for task in sorted(COUNTS):
        episodes = []
        for episode_index in range(COUNTS[task]):
            n = 2 + episode_index % 2
            demo_rel = f"{task}/demo_{episode_index:06d}"
            demo_dir = features_root / demo_rel
            demo_dir.mkdir(parents=True)
            patch_path = demo_dir / "patch_tokens.npy"
            np.save(
                patch_path,
                rng.standard_normal((n, sfv.PATCH_GRID, sfv.BACKBONE_DIM)).astype(np.float16),
            )
            feats_path = demo_dir / "feats.npz"
            np.savez(
                feats_path,
                lang_per_frame=rng.standard_normal((n, sfv.LANGUAGE_DIM)).astype(np.float16),
                frame_indices=(np.arange(n) * FRAME_STRIDE).astype(np.int64),
            )
            episodes.append(
                {
                    "episode_index": episode_index,
                    "patch_tokens": {
                        "path": f"{demo_rel}/patch_tokens.npy",
                        "size_bytes": patch_path.stat().st_size,
                        "sha256": prov.sha256_file(patch_path),
                        "shape": [n, sfv.PATCH_GRID, sfv.BACKBONE_DIM],
                        "dtype": "float16",
                    },
                    "feats": {
                        "path": f"{demo_rel}/feats.npz",
                        "size_bytes": feats_path.stat().st_size,
                        "sha256": prov.sha256_file(feats_path),
                    },
                    "frame_count": n,
                    # ReMemBench ships no teacher labels; the chain must still reach full coverage.
                    "teacher_labels": {
                        "available": False,
                        "path": f"{task}/vlm_episode_pi_{episode_index:06d}.npz",
                        "sha256": None,
                    },
                }
            )
        task_records.append({"task": task, "episodes": episodes})
    manifest = {
        "schema_version": 1,
        "artifact": sfv.ARTIFACT,
        "global_language_mode": sfv.GLOBAL_LANGUAGE_MODE,
        "dataset": {
            "name": dataset_name,
            "episode_subsample_seed": 0,
            "demos_per_task": omega_val.canonical_demos_per_task(COUNTS),
        },
        "task_prompt_manifest": {
            "id": prompt_id,
            "uri": f"s3://bucket/study/manifests/artifacts/workspace/task_prompts/remembench13/{prompt_id}.json",
        },
        "feature_source": {"checkpoint_inventory_id": "f" * 64},
        "frame_grid": {"stride": FRAME_STRIDE, "include_last": True, "image_hw": [256, 256]},
        "producing_code": {"sha256": "e" * 64},
        "tasks": task_records,
    }
    path = features_root / "source_manifest.json"
    path.write_text(prov.canonical_json(manifest) + "\n", encoding="utf-8")
    return path, manifest


def _prompt_manifest(path: Path) -> str:
    payload = {
        "schema_version": 1,
        "artifact": "remembench_train13_task_prompts",
        "global_language_mode": prompt_val.GLOBAL_LANGUAGE_MODE,
        "demo_derived": False,
        "tasks": [{"task": task, "prompt": f"do {task}."} for task in sorted(COUNTS)],
    }
    path.write_text(prov.canonical_json(payload) + "\n", encoding="utf-8")
    return prov.sha256_file(path)


def test_flat_ragged_source_manifest_validates(tmp_path):
    target = tmp_path / "target"
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    labels.mkdir()
    _flat_dataset(target)
    prompt_id = _prompt_manifest(tmp_path / "prompts.json")
    manifest_path, _m = _source_cache(features, prompt_id, dataset_name=RMB_NAME)

    summary = sfv.validate_source_manifest(
        features_root=features,
        target_root=target,
        labels_root=labels,
        manifest_path=manifest_path,
        expected_task_prompt_manifest_id=prompt_id,
        expected_feature_source_inventory_id="f" * 64,
        expected_tasks=len(COUNTS),
        demos_per_task=COUNTS,
        dataset_name=RMB_NAME,
        task_dir_globs=FLAT_GLOBS,
        workers=2,
    )
    assert summary == {"tasks": len(COUNTS), "episodes": sum(COUNTS.values()), "labeled": 0}


def test_ragged_counts_are_enforced_per_task(tmp_path):
    """A uniform count must NOT be silently accepted for a ragged dataset."""
    target = tmp_path / "target"
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    labels.mkdir()
    _flat_dataset(target)
    prompt_id = _prompt_manifest(tmp_path / "prompts.json")
    manifest_path, _m = _source_cache(features, prompt_id, dataset_name=RMB_NAME)
    with pytest.raises(ValueError):
        sfv.validate_source_manifest(
            features_root=features,
            target_root=target,
            labels_root=labels,
            manifest_path=manifest_path,
            expected_task_prompt_manifest_id=prompt_id,
            expected_feature_source_inventory_id="f" * 64,
            expected_tasks=len(COUNTS),
            demos_per_task=3,
            dataset_name=RMB_NAME,
            task_dir_globs=FLAT_GLOBS,
            workers=2,
        )


def test_prompt_manifest_artifact_and_layout_are_parametrized(tmp_path):
    target = tmp_path / "target"
    _flat_dataset(target)
    path = tmp_path / "prompts.json"
    _prompt_manifest(path)
    prompts = prompt_val.load_task_prompts(
        path,
        target_root=target,
        expected_tasks=len(COUNTS),
        expected_artifact="remembench_train13_task_prompts",
        task_dir_globs=FLAT_GLOBS,
    )
    assert set(prompts) == set(COUNTS)
    # the RoboCasa artifact name must still be rejected for this manifest
    with pytest.raises(ValueError):
        prompt_val.load_task_prompts(path, target_root=target, expected_tasks=len(COUNTS), task_dir_globs=FLAT_GLOBS)


# ------------------------------------------------------- the identity property (no real encoder)
def _provenance(dataset_name: str, dataset_block: dict | None) -> dict:
    provenance = {
        "encoder_checkpoint": {"uri": "s3://b/e/" + "a" * 64 + ".pt", "sha256": "a" * 64},
        "workspace_model": {"architecture": "WorkspaceModel-v1", "config": {"dim": 512}, "step": 1},
        "frozen_pi_feature_source": {"checkpoint_inventory_id": "f" * 64},
        "source_features": {"manifest_id": "b" * 64},
        "conditioning": {
            "subgoal_dropout": 1.0,
            "global_language_mode": omega_val.GLOBAL_LANGUAGE_MODE,
            "canonical_task_prompt_manifest_id": "c" * 64,
            "canonical_task_prompt_manifest_uri": "s3://b/p/" + "c" * 64 + ".json",
            "task_lang_table_manifest_sha256": "d" * 64,
        },
        "producing_code": {"sha256": "e" * 64},
    }
    if dataset_block is not None:
        provenance["dataset"] = dataset_block
    return provenance


def test_dataset_block_moves_encoder_id_and_target50_keeps_six_keys():
    rmb_block = {
        "name": RMB_NAME,
        "episode_subsample_seed": 0,
        "demos_per_task": omega_val.canonical_demos_per_task(COUNTS),
    }
    target50 = _provenance(omega_val.DATASET_NAME, None)
    remembench = _provenance(RMB_NAME, rmb_block)
    assert set(target50) == set(omega_val.BASE_PROVENANCE_KEYS)
    assert prov.canonical_json_sha256(target50) != prov.canonical_json_sha256(remembench)

    shared = dict(
        expected_feature_source_inventory_id="f" * 64,
        expected_task_prompt_manifest_id="c" * 64,
        expected_task_prompt_manifest_uri="s3://b/p/" + "c" * 64 + ".json",
    )
    assert omega_val._provenance_sha256(target50, **shared) == prov.canonical_json_sha256(target50)
    assert omega_val._provenance_sha256(
        remembench, expected_dataset=rmb_block, **shared
    ) == prov.canonical_json_sha256(remembench)
    # a target50-shaped provenance must be refused once a dataset block is demanded, and vice versa
    with pytest.raises(ValueError):
        omega_val._provenance_sha256(target50, expected_dataset=rmb_block, **shared)
    with pytest.raises(ValueError):
        omega_val._provenance_sha256(remembench, **shared)
    # and a MISMATCHED dataset block is refused
    with pytest.raises(ValueError):
        omega_val._provenance_sha256(remembench, expected_dataset={**rmb_block, "name": "other"}, **shared)
