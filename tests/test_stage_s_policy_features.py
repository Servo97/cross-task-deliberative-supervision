"""Offline tests for exact Stage-S omega-cache manifest validation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from scripts.launch import validate_stage_s_policy_features as validator

FEATURE_SOURCE_INVENTORY_ID = "f" * 64
TASK_PROMPT_MANIFEST_ID = "d" * 64
TASK_PROMPT_MANIFEST_URI = (
    f"s3://bucket/study/manifests/artifacts/workspace/task_prompts/robocasa_target50/{TASK_PROMPT_MANIFEST_ID}.json"
)
ENCODER_PROVENANCE = {
    "encoder_checkpoint": {"uri": "s3://bucket/encoder.pt", "sha256": "a" * 64},
    "workspace_model": {
        "architecture": "WorkspaceModel-v1",
        "config": {"dim": 512, "layers": 4},
        "step": 100000,
    },
    "frozen_pi_feature_source": {
        "checkpoint_inventory_id": FEATURE_SOURCE_INVENTORY_ID,
    },
    "source_features": {"manifest_id": "b" * 64},
    "conditioning": {
        "subgoal_dropout": 1.0,
        "global_language_mode": "canonical_terse_task_instruction",
        "canonical_task_prompt_manifest_id": TASK_PROMPT_MANIFEST_ID,
        "canonical_task_prompt_manifest_uri": TASK_PROMPT_MANIFEST_URI,
        "task_lang_table_manifest_sha256": "d" * 64,
    },
    "producing_code": {"sha256": "c" * 64},
}
ENCODER_ID = hashlib.sha256(
    json.dumps(
        ENCODER_PROVENANCE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
).hexdigest()


def _make_fixture(tmp_path: Path, *, tasks: int = 2, demos: int = 3):
    target = tmp_path / "target"
    features = tmp_path / "omega"
    task_records = []
    for task_index in range(tasks):
        task = f"Task{task_index:02d}"
        kind = "atomic" if task_index % 2 == 0 else "composite"
        dataset = target / kind / task / "20260101" / "lerobot"
        (dataset / "meta").mkdir(parents=True)
        all_ids = list(range(task_index * 100, task_index * 100 + demos + 2))
        with (dataset / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode_index in all_ids:
                stream.write(json.dumps({"episode_index": episode_index}) + "\n")
        shuffled = list(all_ids)
        random.Random(0).shuffle(shuffled)
        selected = sorted(shuffled[:demos])
        episode_records = []
        for episode_index in selected:
            relative = f"{task}/demo_{episode_index:06d}/w.npz"
            path = features / relative
            path.parent.mkdir(parents=True)
            np.savez(
                path,
                w=np.full((3, validator.OMEGA_DIM), episode_index, dtype=np.float16),
                frame_indices=np.array([0, 8, 16], dtype=np.int64),
                lang_global=np.full(validator.LANGUAGE_DIM, task_index, dtype=np.float16),
            )
            payload = path.read_bytes()
            episode_records.append(
                {
                    "episode_index": episode_index,
                    "path": relative,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        task_records.append({"task": task, "episodes": episode_records})
    manifest = {
        "schema_version": 1,
        "artifact": validator.ARTIFACT,
        "encoder_id": ENCODER_ID,
        "encoder_provenance": json.loads(json.dumps(ENCODER_PROVENANCE)),
        "dataset": {
            "name": validator.DATASET_NAME,
            "episode_subsample_seed": 0,
            "demos_per_task": demos,
        },
        "tasks": task_records,
    }
    manifest_path = features / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return target, features, manifest_path, manifest


def _replace_archive(path: Path, record: dict, **arrays) -> None:
    np.savez(path, **arrays)
    payload = path.read_bytes()
    record["size_bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()


def _validate(target, features, manifest_path, *, tasks=2, demos=3):
    return validator.validate_manifest(
        features_root=features,
        target_root=target,
        manifest_path=manifest_path,
        encoder_id=ENCODER_ID,
        expected_feature_source_inventory_id=FEATURE_SOURCE_INVENTORY_ID,
        expected_task_prompt_manifest_id=TASK_PROMPT_MANIFEST_ID,
        expected_task_prompt_manifest_uri=TASK_PROMPT_MANIFEST_URI,
        expected_tasks=tasks,
        demos_per_task=demos,
        seed=0,
        workers=2,
    )


def test_exact_seed0_keep_set_and_all_files_pass(tmp_path):
    target, features, manifest_path, _manifest = _make_fixture(tmp_path)
    assert _validate(target, features, manifest_path) == {"tasks": 2, "episodes": 6}


def test_undeclared_file_is_rejected(tmp_path):
    target, features, manifest_path, _manifest = _make_fixture(tmp_path)
    extra = features / "Task00" / "demo_999999" / "w.npz"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="undeclared"):
        _validate(target, features, manifest_path)


def test_wrong_episode_even_with_valid_file_is_rejected(tmp_path):
    target, features, manifest_path, manifest = _make_fixture(tmp_path)
    record = manifest["tasks"][0]["episodes"][0]
    old = features / record["path"]
    wrong_id = 999
    relative = f"Task00/demo_{wrong_id:06d}/w.npz"
    replacement = features / relative
    replacement.parent.mkdir(parents=True)
    old.replace(replacement)
    payload = replacement.read_bytes()
    record.update(
        episode_index=wrong_id,
        path=relative,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="seed-0 T30 keep-set"):
        _validate(target, features, manifest_path)


def test_corrupt_bytes_are_rejected(tmp_path):
    target, features, manifest_path, manifest = _make_fixture(tmp_path)
    path = features / manifest["tasks"][0]["episodes"][0]["path"]
    original = path.read_bytes()
    path.write_bytes(b"X" * len(original))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _validate(target, features, manifest_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"w": np.zeros((3, 511), dtype=np.float16)}, "omega tensor shape"),
        ({"w": np.zeros((3, 512), dtype=np.float32)}, "omega tensor dtype"),
        ({"frame_indices": np.array([0, 8, 8], dtype=np.int64)}, "strictly increasing"),
        ({"frame_indices": np.array([0, 8, 16], dtype=np.int32)}, "frame_indices dtype"),
        ({"lang_global": np.zeros(2047, dtype=np.float16)}, "lang_global shape"),
        ({"lang_global": np.full(2048, np.nan, dtype=np.float16)}, "NaN or infinity"),
    ],
)
def test_invalid_tensor_contract_is_rejected(tmp_path, overrides, message):
    target, features, manifest_path, manifest = _make_fixture(tmp_path)
    record = manifest["tasks"][0]["episodes"][0]
    path = features / record["path"]
    arrays = {
        "w": np.zeros((3, validator.OMEGA_DIM), dtype=np.float16),
        "frame_indices": np.array([0, 8, 16], dtype=np.int64),
        "lang_global": np.zeros(validator.LANGUAGE_DIM, dtype=np.float16),
    }
    arrays.update(overrides)
    _replace_archive(path, record, **arrays)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _validate(target, features, manifest_path)


def test_archive_key_set_is_exact(tmp_path):
    target, features, manifest_path, manifest = _make_fixture(tmp_path)
    record = manifest["tasks"][0]["episodes"][0]
    path = features / record["path"]
    _replace_archive(
        path,
        record,
        w=np.zeros((3, validator.OMEGA_DIM), dtype=np.float16),
        frame_indices=np.array([0, 8, 16], dtype=np.int64),
        lang_global=np.zeros(validator.LANGUAGE_DIM, dtype=np.float16),
        unexpected=np.zeros(1, dtype=np.float16),
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="keys .* exactly"):
        _validate(target, features, manifest_path)


def test_duplicate_task_and_encoder_mismatch_are_rejected(tmp_path):
    target, features, manifest_path, manifest = _make_fixture(tmp_path)
    manifest["tasks"][1]["task"] = manifest["tasks"][0]["task"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates task"):
        _validate(target, features, manifest_path)

    manifest["tasks"][1]["task"] = "Task01"
    manifest["encoder_id"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _validate(target, features, manifest_path)


def test_encoder_id_is_derived_and_teacher_prompt_cache_is_rejected(tmp_path):
    target, features, manifest_path, manifest = _make_fixture(tmp_path)
    manifest["encoder_provenance"]["workspace_model"]["step"] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA-256"):
        _validate(target, features, manifest_path)

    target, features, manifest_path, manifest = _make_fixture(tmp_path / "expanded")
    manifest["encoder_provenance"]["conditioning"]["global_language_mode"] = "per_demo_qwen_expanded_prompt"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="per-demo teacher/expanded prompts are forbidden"):
        _validate(target, features, manifest_path)


def test_feature_source_inventory_must_match_training_initialization(tmp_path):
    target, features, manifest_path, _manifest = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="initialization inventory"):
        validator.validate_manifest(
            features_root=features,
            target_root=target,
            manifest_path=manifest_path,
            encoder_id=ENCODER_ID,
            expected_feature_source_inventory_id="0" * 64,
            expected_task_prompt_manifest_id=TASK_PROMPT_MANIFEST_ID,
            expected_task_prompt_manifest_uri=TASK_PROMPT_MANIFEST_URI,
            expected_tasks=2,
            demos_per_task=3,
            seed=0,
            workers=2,
        )


def test_task_prompt_manifest_identity_is_part_of_encoder_provenance(tmp_path):
    target, features, manifest_path, _manifest = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="task-prompt manifest ID"):
        validator.validate_manifest(
            features_root=features,
            target_root=target,
            manifest_path=manifest_path,
            encoder_id=ENCODER_ID,
            expected_feature_source_inventory_id=FEATURE_SOURCE_INVENTORY_ID,
            expected_task_prompt_manifest_id="0" * 64,
            expected_task_prompt_manifest_uri=TASK_PROMPT_MANIFEST_URI,
            expected_tasks=2,
            demos_per_task=3,
            seed=0,
            workers=2,
        )
