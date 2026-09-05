"""Offline exact-VersionId materialization tests (no AWS SDK/network)."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest

from scripts.launch import materialize_stage_s_inventory as materializer

ROOT = "s3://bucket/snapshot"


class FakeS3:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []

    def get_object(self, **request):
        self.requests.append(request)
        # version mode keys by VersionId; etag mode keys by the IfMatch ETag.
        anchor = request.get("VersionId", request.get("IfMatch"))
        payload = self.payloads[(request["Key"], anchor)]
        response = {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
        }
        if "VersionId" in request:
            response["VersionId"] = request["VersionId"]
        if "IfMatch" in request:
            response["ETag"] = request["IfMatch"]
        if request.get("ChecksumMode") == "ENABLED":
            response["ChecksumSHA256"] = base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")
        return response


def _write_manifest(tmp_path):
    one = b"params"
    two = b"assets"
    manifest = {
        "schema_version": 1,
        "artifact": "pi05_h300_mg_init",
        "root_s3": ROOT,
        "content_addressing": "version_id",
        "selection": None,
        "objects": [
            {
                "key": "params/a",
                "size_bytes": len(one),
                "version_id": "version-a",
                "checksum_sha256": hashlib.sha256(one).hexdigest(),
            },
            {
                "key": "assets/b",
                "size_bytes": len(two),
                "version_id": "version-b",
                "etag": '"etag-b"',
            },
        ],
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, {("snapshot/params/a", "version-a"): one, ("snapshot/assets/b", "version-b"): two}


def _write_etag_manifest(tmp_path):
    one = b"params"
    two = b"assets"
    manifest = {
        "schema_version": 1,
        "artifact": "pi05_h300_mg_init",
        "root_s3": ROOT,
        "content_addressing": "etag",
        "selection": None,
        "objects": [
            {
                "key": "params/a",
                "size_bytes": len(one),
                "etag": '"etag-a"',
                "checksum_sha256": hashlib.sha256(one).hexdigest(),
            },
            {"key": "assets/b", "size_bytes": len(two), "etag": '"etag-b"'},
        ],
    }
    path = tmp_path / "inventory_etag.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, {("snapshot/params/a", '"etag-a"'): one, ("snapshot/assets/b", '"etag-b"'): two}


def test_materializes_exact_versions_and_exact_file_set(tmp_path):
    manifest, payloads = _write_manifest(tmp_path)
    client = FakeS3(payloads)
    destination = tmp_path / "download"
    summary = materializer.materialize_inventory(
        manifest,
        expected_artifact="pi05_h300_mg_init",
        expected_root_s3=ROOT,
        destination=destination,
        workers=2,
        s3_client=client,
    )
    assert summary == {"objects": 2, "bytes": 12}
    assert (destination / "params/a").read_bytes() == b"params"
    assert (destination / "assets/b").read_bytes() == b"assets"
    assert {request["VersionId"] for request in client.requests} == {"version-a", "version-b"}
    assert not list(destination.rglob("*.incomplete"))


def test_nonempty_destination_and_wrong_returned_version_fail(tmp_path):
    manifest, payloads = _write_manifest(tmp_path)
    destination = tmp_path / "download"
    destination.mkdir()
    (destination / "extra").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        materializer.materialize_inventory(
            manifest,
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
            destination=destination,
            s3_client=FakeS3(payloads),
        )

    class WrongVersion(FakeS3):
        def get_object(self, **request):
            response = super().get_object(**request)
            response["VersionId"] = "latest-not-requested"
            return response

    with pytest.raises(ValueError, match="wrong VersionId"):
        materializer.materialize_inventory(
            manifest,
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
            destination=tmp_path / "fresh",
            s3_client=WrongVersion(payloads),
        )


def test_etag_mode_materializes_without_versionid(tmp_path):
    manifest, payloads = _write_etag_manifest(tmp_path)
    client = FakeS3(payloads)
    destination = tmp_path / "download_etag"
    summary = materializer.materialize_inventory(
        manifest,
        expected_artifact="pi05_h300_mg_init",
        expected_root_s3=ROOT,
        destination=destination,
        workers=2,
        s3_client=client,
    )
    assert summary == {"objects": 2, "bytes": 12}
    assert (destination / "params/a").read_bytes() == b"params"
    # etag mode gates on IfMatch, never sends VersionId
    assert all("VersionId" not in r for r in client.requests)
    assert {r["IfMatch"] for r in client.requests} == {'"etag-a"', '"etag-b"'}


REMEMBENCH_ROOT = "s3://bucket/remembench_v02"
# 13 tasks with DELIBERATELY non-uniform counts — the property target50's uniform check cannot model.
REMEMBENCH_TASKS = tuple((f"MemTask{index:02d}", 2 + index % 3) for index in range(13))
REMEMBENCH_GLOBS = ("*/*/lerobot",)


def _remembench_fixture(tmp_path, *, drop_metadata_id=False, skip_task=None):
    """A FLAT <task>/<date>/lerobot export whose object keys carry a leading 'train/' component."""
    objects, payloads = [], {}

    def add(key: str, body: bytes):
        etag = f'"{len(payloads):032x}"'
        objects.append({"key": key, "size_bytes": len(body), "etag": etag})
        payloads[(f"remembench_v02/{key}", etag)] = body

    for task, count in REMEMBENCH_TASKS:
        if task == skip_task:  # declared in the selection but never staged
            continue
        base = f"train/{task}/20260803/lerobot"
        ids = list(range(count))
        declared = ids[:-1] if drop_metadata_id else ids
        add(
            f"{base}/meta/episodes.jsonl",
            "".join(json.dumps({"episode_index": i, "tasks": [task]}) + "\n" for i in declared).encode(),
        )
        for i in ids:
            add(f"{base}/data/chunk-000/episode_{i:06d}.parquet", b"p")
            add(f"{base}/videos/chunk-000/cam/episode_{i:06d}.mp4", b"v")

    manifest = {
        "schema_version": 1,
        "artifact": "remembench_train13",
        "root_s3": REMEMBENCH_ROOT,
        "content_addressing": "etag",
        "selection": {
            "name": "remembench_train_tail02",
            "kind": "remembench_tail_fraction_complement",
            "fraction": 0.2,
            "minimum": 3,
            "tasks": [{"task": task, "episode_indices": list(range(count))} for task, count in REMEMBENCH_TASKS],
        },
        "objects": objects,
    }
    path = tmp_path / "remembench.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, payloads, manifest


def _materialize_remembench(manifest_path, payloads, destination):
    return materializer.materialize_inventory(
        manifest_path,
        expected_artifact="remembench_train13",
        expected_root_s3=REMEMBENCH_ROOT,
        destination=destination,
        selection_root=destination / "train",
        task_dir_globs=REMEMBENCH_GLOBS,
        s3_client=FakeS3(payloads),
    )


def test_remembench_flat_layout_materializes_and_verifies_every_demo(tmp_path):
    manifest_path, payloads, manifest = _remembench_fixture(tmp_path)
    destination = tmp_path / "download"
    summary = _materialize_remembench(manifest_path, payloads, destination)
    assert summary["objects"] == len(manifest["objects"])
    # The task dirs live one level BELOW the download destination (keys start with "train/").
    assert (destination / "train/MemTask00/20260803/lerobot/meta/episodes.jsonl").is_file()
    assert len(list((destination / "train").glob("*/*/lerobot"))) == 13


def test_remembench_metadata_must_enumerate_exactly_the_selected_episodes(tmp_path):
    manifest_path, payloads, _ = _remembench_fixture(tmp_path, drop_metadata_id=True)
    with pytest.raises(ValueError, match="differs from staged metadata"):
        _materialize_remembench(manifest_path, payloads, tmp_path / "download")


def test_remembench_task_set_must_match_the_selection(tmp_path):
    manifest_path, payloads, _ = _remembench_fixture(tmp_path, skip_task="MemTask07")
    with pytest.raises(ValueError, match="task set differs from selection"):
        _materialize_remembench(manifest_path, payloads, tmp_path / "download")


def test_robocasa_defaults_are_unchanged_by_the_new_parameters(tmp_path):
    """No --selection-root / --task-dir-glob => the historical nested-glob target50 behavior."""
    assert materializer.DEFAULT_TASK_DIR_GLOBS == (
        "atomic/*/*/lerobot",
        "composite/*/*/lerobot",
    )
    root = tmp_path / "target"
    (root / "atomic" / "TaskA" / "20260101" / "lerobot").mkdir(parents=True)
    (root / "composite" / "TaskB" / "20260101" / "lerobot").mkdir(parents=True)
    (root / "TaskC" / "20260101" / "lerobot").mkdir(parents=True)  # flat: invisible by default
    assert sorted(materializer._target_task_dirs(root)) == ["TaskA", "TaskB"]
    assert sorted(materializer._target_task_dirs(root, ("*/*/lerobot",))) == ["TaskC"]


def test_etag_mode_wrong_etag_precondition_fails(tmp_path):
    manifest, payloads = _write_etag_manifest(tmp_path)

    class WrongETag(FakeS3):
        def get_object(self, **request):
            response = super().get_object(**request)
            response["ETag"] = '"mutated"'
            return response

    with pytest.raises(ValueError, match="wrong ETag"):
        materializer.materialize_inventory(
            manifest,
            expected_artifact="pi05_h300_mg_init",
            expected_root_s3=ROOT,
            destination=tmp_path / "fresh_etag",
            s3_client=WrongETag(payloads),
        )
