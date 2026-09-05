"""Offline producer-to-consumer contract for final Stage-S checkpoint trees."""

from __future__ import annotations

import shutil

import pytest

from scripts.launch import build_stage_s_checkpoint_manifest as builder
from scripts.launch import validate_artifact_tree as consumer

URI = "s3://bucket/study/checkpoints/pi05/s0/run/59999"


def _checkpoint(tmp_path):
    root = tmp_path / "59999"
    for relative, payload in {
        "params/manifest.ocdbt": b"commit",
        "params/_METADATA": b"metadata",
        "assets/norm_stats.json": b"{}",
        "train_state/state": b"optimizer",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def test_builder_output_is_exactly_accepted_by_shared_eval_consumer(tmp_path):
    root = _checkpoint(tmp_path)
    manifest, payload, digest = builder.build_manifest(
        root,
        checkpoint_uri=URI,
        workers=2,
    )
    assert all(record["path"].startswith(("params/", "assets/")) for record in manifest["files"])
    assert not any(record["path"].startswith("train_state/") for record in manifest["files"])
    shutil.rmtree(root / "train_state")
    assert set(manifest) == {"schema_version", "kind", "artifact_uri", "files"}
    assert manifest["kind"] == "wsm_artifact_tree_manifest"
    manifest_path = tmp_path / f"{digest}.json"
    manifest_path.write_bytes(payload)
    result = consumer.validate_artifact_tree(
        root,
        manifest_path,
        manifest_sha256=digest,
        artifact_uri=URI,
        workers=2,
        require_prefix="params/",
    )
    assert result["num_files"] == 3


def test_mutation_and_missing_orbax_completion_are_rejected(tmp_path):
    root = _checkpoint(tmp_path)
    _manifest, payload, digest = builder.build_manifest(
        root,
        checkpoint_uri=URI,
    )
    manifest_path = tmp_path / "tree.json"
    manifest_path.write_bytes(payload)
    shutil.rmtree(root / "train_state")
    (root / "params/_METADATA").write_bytes(b"METADATA")
    with pytest.raises(ValueError, match="sha256"):
        consumer.validate_artifact_tree(
            root,
            manifest_path,
            manifest_sha256=digest,
            artifact_uri=URI,
        )

    root = _checkpoint(tmp_path / "no-marker")
    (root / "params/manifest.ocdbt").unlink()
    (root / "params/_METADATA").unlink()
    with pytest.raises(ValueError, match="no recognized Orbax"):
        builder.build_manifest(root, checkpoint_uri=URI)


def test_builder_does_not_require_optimizer_state(tmp_path):
    root = _checkpoint(tmp_path)
    shutil.rmtree(root / "train_state")
    manifest, _payload, _digest = builder.build_manifest(root, checkpoint_uri=URI)
    assert {record["path"].split("/", 1)[0] for record in manifest["files"]} == {
        "params",
        "assets",
    }
