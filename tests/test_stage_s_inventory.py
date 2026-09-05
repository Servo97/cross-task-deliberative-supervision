"""Offline tests for Stage-S immutable S3 object-inventory validation."""

from __future__ import annotations

import json

import pytest

from scripts.launch import validate_stage_s_inventory as validator

ROOT = "s3://bucket/canonical/root"


def _manifest():
    return {
        "schema_version": 1,
        "artifact": "pi05_h300_mg_init",
        "root_s3": ROOT,
        "content_addressing": "version_id",
        "selection": None,
        "objects": [
            {"key": "params/a", "size_bytes": 7, "version_id": "v1"},
            {
                "key": "assets/b",
                "size_bytes": 0,
                "version_id": "v2",
                "checksum_sha256": "a" * 64,
            },
            {"key": "assets/c", "size_bytes": 3, "version_id": "v3", "etag": '"etag"'},
        ],
    }


def _etag_manifest():
    """An unversioned-bucket inventory: ETag is the immutability anchor, no version_id."""
    return {
        "schema_version": 1,
        "artifact": "pi05_h300_mg_init",
        "root_s3": ROOT,
        "content_addressing": "etag",
        "selection": None,
        "objects": [
            {"key": "params/a", "size_bytes": 7, "etag": '"abc"'},
            {"key": "assets/b", "size_bytes": 0, "etag": '"def"', "checksum_sha256": "a" * 64},
        ],
    }


REMEMBENCH_TASKS = (
    ("MemFruitInSinkLeftFar", 20),
    ("MemFruitInSinkRightFar", 20),
    ("MemHeatPot", 40),
    ("MemHeatPotMultiple", 40),
    ("MemPutKBowlInCabinet", 44),
    ("MemPutKBreadInMicrowave", 40),
    ("MemRetrieveOilsFromCounterLL", 10),
    ("MemRetrieveOilsFromCounterLR", 10),
    ("MemRetrieveOilsFromCounterRL", 9),
    ("MemRetrieveOilsFromCounterRR", 10),
    ("MemWashAndReturnLeft", 20),
    ("MemWashAndReturnRight", 20),
    ("MemWashAndReturnSameLocation", 40),
)


def _remembench_manifest():
    """The 13-task ReMemBench train split: a tail-fraction COMPLEMENT, so per-task counts differ."""
    return {
        "schema_version": 1,
        "artifact": "remembench_train13",
        "root_s3": ROOT,
        "content_addressing": "etag",
        "selection": {
            "name": "remembench_train_tail02",
            "kind": "remembench_tail_fraction_complement",
            "fraction": 0.2,
            "minimum": 3,
            "tasks": [{"task": task, "episode_indices": list(range(count))} for task, count in REMEMBENCH_TASKS],
        },
        "objects": [
            {"key": f"train/{task}/20260803/lerobot/meta/episodes.jsonl", "size_bytes": 11, "etag": f'"{index:032x}"'}
            for index, (task, _count) in enumerate(REMEMBENCH_TASKS)
        ],
    }


def _write(tmp_path, manifest):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_inventory_uses_metadata_without_reading_payloads(tmp_path):
    summary = validator.validate_inventory(
        _write(tmp_path, _manifest()),
        expected_artifact="pi05_h300_mg_init",
        expected_root_s3=ROOT,
    )
    assert summary == {"objects": 3, "bytes": 10}


def test_duplicate_and_unsafe_keys_are_rejected(tmp_path):
    manifest = _manifest()
    manifest["objects"][1]["key"] = manifest["objects"][0]["key"]
    with pytest.raises(ValueError, match="duplicates"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )
    manifest["objects"][1]["key"] = "../escape"
    with pytest.raises(ValueError, match="safe relative"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )


def test_every_object_needs_snapshot_identity(tmp_path):
    manifest = _manifest()
    manifest["objects"][0] = {"key": "params/a", "size_bytes": 7}
    with pytest.raises(ValueError, match="keys must contain"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )


def test_artifact_root_and_schema_are_exact(tmp_path):
    manifest = _manifest()
    manifest["extra"] = "not allowed"
    with pytest.raises(ValueError, match="top-level keys"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )
    manifest.pop("extra")
    with pytest.raises(ValueError, match="does not match"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="robocasa_target50",
            expected_root_s3=ROOT,
        )


def test_null_version_and_noncanonical_target_selection_are_rejected(tmp_path):
    manifest = _manifest()
    manifest["objects"][0]["version_id"] = "null"
    with pytest.raises(ValueError, match="non-null S3 version_id"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )

    manifest = _manifest()
    manifest["artifact"] = "robocasa_target50"
    manifest["selection"] = {"name": "seed0_t30"}
    with pytest.raises(ValueError, match="selection keys"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="robocasa_target50",
            expected_root_s3=ROOT,
        )


def test_etag_mode_valid_inventory(tmp_path):
    summary = validator.validate_inventory(
        _write(tmp_path, _etag_manifest()),
        expected_artifact="pi05_h300_mg_init",
        expected_root_s3=ROOT,
    )
    assert summary == {"objects": 2, "bytes": 7}


def test_etag_mode_forbids_version_id(tmp_path):
    manifest = _etag_manifest()
    manifest["objects"][0]["version_id"] = "v1"  # not allowed in etag mode
    with pytest.raises(ValueError, match="use only"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )


def test_etag_mode_requires_etag(tmp_path):
    manifest = _etag_manifest()
    del manifest["objects"][0]["etag"]
    with pytest.raises(ValueError, match="must contain"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )


def test_remembench_inventory_accepts_nonuniform_per_task_counts(tmp_path):
    summary = validator.validate_inventory(
        _write(tmp_path, _remembench_manifest()),
        expected_artifact="remembench_train13",
        expected_root_s3=ROOT,
    )
    assert summary == {"objects": 13, "bytes": 143}
    # 323 train demos across 13 tasks, counts 9..44 — no uniform demos_per_task exists.
    selection = _remembench_manifest()["selection"]
    assert sum(len(t["episode_indices"]) for t in selection["tasks"]) == 323


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s.__setitem__("kind", "seed0_t30"), "selection kind"),
        (lambda s: s.__setitem__("episode_subsample_seed", 0), "selection keys"),
        (lambda s: s.pop("minimum"), "selection keys"),
        (lambda s: s.__setitem__("fraction", 1.0), "fraction must be strictly"),
        (lambda s: s.__setitem__("fraction", 0), "fraction must be strictly"),
        (lambda s: s.__setitem__("minimum", 0), "minimum must be a positive"),
        (lambda s: s.__setitem__("minimum", True), "minimum must be a positive"),
        (lambda s: s.__setitem__("name", ""), "selection name must be"),
        (lambda s: s["tasks"].pop(), "exactly 13 tasks"),
        (lambda s: s["tasks"].__setitem__(0, {"task": "A", "episode_indices": []}), "non-empty"),
        (lambda s: s["tasks"].__setitem__(0, {"task": "A", "episode_indices": [0, 0]}), "unique"),
        (lambda s: s["tasks"].__setitem__(0, {"task": "A", "episode_indices": [-1]}), "nonnegative"),
        (lambda s: s["tasks"].__setitem__(0, {"task": "A", "episode_indices": [True]}), "nonnegative"),
        (lambda s: s["tasks"].__setitem__(0, {"task": "A/B", "episode_indices": [0]}), "invalid or duplicate"),
        (lambda s: s["tasks"].__setitem__(1, s["tasks"][0]), "invalid or duplicate"),
        (lambda s: s["tasks"].__setitem__(0, {"task": "A", "episode_indices": [0], "x": 1}), "exactly"),
    ],
)
def test_remembench_selection_contract_is_enforced(tmp_path, mutate, match):
    manifest = _remembench_manifest()
    mutate(manifest["selection"])
    with pytest.raises(ValueError, match=match):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="remembench_train13",
            expected_root_s3=ROOT,
        )


def test_remembench_selection_must_not_be_null(tmp_path):
    manifest = _remembench_manifest()
    manifest["selection"] = None
    with pytest.raises(ValueError, match="remembench inventory selection keys"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="remembench_train13",
            expected_root_s3=ROOT,
        )


def test_robocasa_target50_selection_contract_is_untouched(tmp_path):
    """The remembench branch must not have loosened the 50x150 seed-0 target50 contract."""
    manifest = _remembench_manifest()
    manifest["artifact"] = "robocasa_target50"
    with pytest.raises(ValueError, match="target inventory selection keys"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="robocasa_target50",
            expected_root_s3=ROOT,
        )


def test_remembench_artifact_is_registered_and_selectable(tmp_path):
    assert "remembench_train13" in validator.ARTIFACTS
    assert validator.REMEMBENCH_TRAIN_ARTIFACT == "remembench_train13"
    assert validator.ROBOCASA_TARGET_ARTIFACT == "robocasa_target50"
    # An unregistered artifact still fails closed.
    with pytest.raises(ValueError, match="unsupported expected artifact"):
        validator.validate_inventory(
            _write(tmp_path, _remembench_manifest()),
            expected_artifact="remembench_train14",
            expected_root_s3=ROOT,
        )


def test_unknown_content_addressing_rejected(tmp_path):
    manifest = _manifest()
    manifest["content_addressing"] = "bogus"
    with pytest.raises(ValueError, match="content_addressing must be"):
        validator.validate_inventory(
            _write(tmp_path, manifest),
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
        )
