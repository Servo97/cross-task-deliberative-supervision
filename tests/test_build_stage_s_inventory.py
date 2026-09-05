"""Offline tests for immutable Stage-S S3 VersionId inventory construction."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_DIR))
SCRIPT = LAUNCH_DIR / "build_stage_s_inventory.py"
SPEC = importlib.util.spec_from_file_location("build_stage_s_inventory_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class _Body(io.BytesIO):
    pass


class FakeS3:
    def __init__(self, *, bucket: str, prefix: str, objects: dict[str, bytes]):
        self.bucket = bucket
        self.prefix = prefix
        self.objects = dict(objects)

    def get_bucket_versioning(self, *, Bucket):
        assert Bucket == self.bucket
        return {"Status": "Enabled"}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        assert Bucket == self.bucket
        assert Prefix == f"{self.prefix}/"
        assert ContinuationToken is None
        return {
            "IsTruncated": False,
            "Contents": [{"Key": key} for key in sorted(self.objects)],
        }

    def head_object(self, *, Bucket, Key, ChecksumMode):
        assert Bucket == self.bucket
        assert ChecksumMode == "ENABLED"
        value = self.objects[Key]
        return {
            "ContentLength": len(value),
            "VersionId": "v-" + hashlib.sha256(Key.encode()).hexdigest()[:12],
            "ETag": '"' + hashlib.md5(value, usedforsecurity=False).hexdigest() + '"',  # noqa: S324
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(value).digest()).decode(),
        }

    def get_object(self, *, Bucket, Key, VersionId):
        assert Bucket == self.bucket
        relative = Key[len(self.prefix) + 1 :]
        assert VersionId == "v-" + hashlib.sha256(relative.encode()).hexdigest()[:12]
        return {"Body": _Body(self.objects[Key])}


def test_build_init_inventory_is_content_addressed_and_deploy_only(tmp_path):
    bucket = "versioned-bucket"
    prefix = "study/init"
    objects = {
        f"{prefix}/params/a": b"params",
        f"{prefix}/assets/norm.json": b"assets",
        f"{prefix}/train_state/optimizer": b"must-not-ship",
    }
    client = FakeS3(bucket=bucket, prefix=prefix, objects=objects)
    result = inventory.build_inventory(
        artifact="pi05_h300_mg_init",
        root_s3=f"s3://{bucket}/{prefix}",
        study_root="s3://versioned-bucket/owner/studies/long_context_v1",
        output_dir=tmp_path,
        workers=2,
        s3_client=client,
    )
    path, digest, uri, manifest = result
    assert path.name == f"{digest}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert uri.endswith(f"/manifests/inventories/init/{digest}.json")
    assert [record["key"] for record in manifest["objects"]] == [
        "assets/norm.json",
        "params/a",
    ]
    assert all(record["version_id"].startswith("v-") for record in manifest["objects"])
    assert json.loads(path.read_text())["selection"] is None


def _episode_metadata(ids: list[int]) -> bytes:
    return b"".join((json.dumps({"episode_index": value}) + "\n").encode() for value in ids)


def _record(key: str, value: bytes) -> dict:
    return {
        "key": key,
        "size_bytes": len(value),
        "version_id": "v-" + hashlib.sha256(key.encode()).hexdigest()[:12],
    }


def test_target_selection_keeps_static_metadata_and_only_seeded_payloads():
    bucket = "bucket"
    prefix = "target"
    objects: dict[str, bytes] = {}
    records: list[dict] = []
    for family, task in (("atomic", "TaskA"), ("composite", "TaskB")):
        dataset = f"{family}/{task}/capture/lerobot"
        metadata = _episode_metadata([0, 1, 2])
        values = {
            f"{dataset}/meta/episodes.jsonl": metadata,
            f"{dataset}/meta/info.json": b"{}",
        }
        for episode in range(3):
            values[f"{dataset}/data/chunk-000/episode_{episode:06d}.parquet"] = b"p"
            values[f"{dataset}/videos/chunk-000/camera/episode_{episode:06d}.mp4"] = b"v"
        for key, value in values.items():
            objects[f"{prefix}/{key}"] = value
            records.append(_record(key, value))
    client = FakeS3(bucket=bucket, prefix=prefix, objects=objects)
    selected, selection = inventory.select_target_records(
        client,
        bucket=bucket,
        prefix=prefix,
        records=records,
        demos_per_task=2,
        expected_tasks=2,
        seed=0,
    )
    expected = [0, 2]
    shuffled = [0, 1, 2]
    random.Random(0).shuffle(shuffled)
    assert sorted(shuffled[:2]) == expected
    assert all(record["episode_indices"] == expected for record in selection["tasks"])
    keys = {record["key"] for record in selected}
    assert any(key.endswith("/meta/info.json") for key in keys)
    assert any(key.endswith("/meta/episodes.jsonl") for key in keys)
    assert not any("episode_000001" in key for key in keys)
    assert sum(key.endswith(".parquet") for key in keys) == 4
    assert sum(key.endswith(".mp4") for key in keys) == 4


def test_missing_version_and_unapproved_cli_fail_closed(tmp_path):
    client = FakeS3(
        bucket="bucket",
        prefix="root",
        objects={"root/params/a": b"x", "root/assets/a": b"y"},
    )
    original = client.head_object

    def no_version(**kwargs):
        result = original(**kwargs)
        result.pop("VersionId")
        return result

    client.head_object = no_version
    with pytest.raises(ValueError, match="VersionId"):
        inventory.snapshot_current_objects(client, root_s3="s3://bucket/root")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact",
            "pi05_h300_mg_init",
            "--root-s3",
            "s3://bucket/root",
            "--study-root",
            "s3://bucket/study",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0
    assert "obtain explicit user approval" in result.stderr


class UnversionedFakeS3(FakeS3):
    """Bucket without versioning: no VersionId; get_object gates on IfMatch ETag."""

    def get_bucket_versioning(self, *, Bucket):
        return {}  # no Status key == versioning never enabled

    def head_object(self, *, Bucket, Key, ChecksumMode):
        value = self.objects[Key]
        return {
            "ContentLength": len(value),
            "ETag": '"' + hashlib.md5(value, usedforsecurity=False).hexdigest() + '"',  # noqa: S324
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(value).digest()).decode(),
        }

    def get_object(self, *, Bucket, Key, IfMatch=None, **_kw):
        value = self.objects[Key]
        assert IfMatch == '"' + hashlib.md5(value, usedforsecurity=False).hexdigest() + '"'  # noqa: S324
        return {"Body": _Body(value)}


def test_build_etag_inventory_on_unversioned_bucket(tmp_path):
    bucket = "unversioned-bucket"
    prefix = "study/init"
    objects = {
        f"{prefix}/params/a": b"params",
        f"{prefix}/assets/norm.json": b"assets",
        f"{prefix}/train_state/optimizer": b"must-not-ship",
    }
    client = UnversionedFakeS3(bucket=bucket, prefix=prefix, objects=objects)
    path, digest, uri, manifest = inventory.build_inventory(
        artifact="pi05_h300_mg_init",
        root_s3=f"s3://{bucket}/{prefix}",
        study_root="s3://unversioned-bucket/owner/studies/long_context_v1",
        output_dir=tmp_path,
        workers=2,
        allow_unversioned=True,
        s3_client=client,
    )
    assert manifest["content_addressing"] == "etag"
    assert all("version_id" not in record and record["etag"] for record in manifest["objects"])
    assert path.name == f"{digest}.json"


def test_unversioned_bucket_without_flag_fails_closed(tmp_path):
    bucket = "unversioned-bucket"
    prefix = "study/init"
    client = UnversionedFakeS3(bucket=bucket, prefix=prefix, objects={f"{prefix}/params/a": b"p"})
    with pytest.raises(ValueError, match="not Enabled"):
        inventory.build_inventory(
            artifact="pi05_h300_mg_init",
            root_s3=f"s3://{bucket}/{prefix}",
            study_root="s3://unversioned-bucket/owner/studies/long_context_v1",
            output_dir=tmp_path,
            workers=2,
            s3_client=client,  # allow_unversioned defaults False
        )
