"""Offline tests for immutable Stage-S artifact publication."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_DIR))
SCRIPT = LAUNCH_DIR / "publish_stage_s_artifact.py"
SPEC = importlib.util.spec_from_file_location("publish_stage_s_artifact_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class PreconditionFailed(Exception):
    def __init__(self):
        self.response = {"Error": {"Code": "PreconditionFailed"}}


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.puts = 0

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, ChecksumSHA256):
        assert IfNoneMatch == "*"
        identity = (Bucket, Key)
        if identity in self.objects:
            raise PreconditionFailed
        value = Body.read()
        assert ChecksumSHA256 == base64.b64encode(hashlib.sha256(value).digest()).decode()
        self.puts += 1
        self.objects[identity] = (value, "version-1")
        return {"VersionId": "version-1"}

    def head_object(self, *, Bucket, Key, VersionId, ChecksumMode):
        assert ChecksumMode == "ENABLED"
        value, version = self.objects[(Bucket, Key)]
        assert VersionId == version
        return {
            "VersionId": version,
            "ContentLength": len(value),
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(value).digest()).decode(),
        }

    def get_object(self, *, Bucket, Key, ChecksumMode):
        assert ChecksumMode == "ENABLED"
        value, version = self.objects[(Bucket, Key)]
        return {"VersionId": version, "Body": io.BytesIO(value)}


def _fixture(tmp_path):
    source = tmp_path / "artifact.tgz"
    source.write_bytes(b"immutable artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    root = "s3://bucket/owner/wsm_robocasa/studies/long_context_v1"
    destination = f"{root}/code/wsmv2/{digest}.tgz"
    return source, digest, root, destination


def test_publish_once_is_create_only_and_idempotent(tmp_path):
    source, digest, root, destination = _fixture(tmp_path)
    client = FakeS3()
    first = publisher.publish_once(source, destination_s3=destination, study_root=root, s3_client=client)
    second = publisher.publish_once(source, destination_s3=destination, study_root=root, s3_client=client)
    assert first == {
        "created": True,
        "sha256": digest,
        "version_id": "version-1",
        "uri": destination,
    }
    assert second["created"] is False
    assert second["version_id"] == "version-1"
    assert client.puts == 1


def test_noncanonical_or_digest_mismatched_destination_fails(tmp_path):
    source, _digest, root, destination = _fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        publisher.validate_publish_target(source, destination_s3="s3://other/path", study_root=root)
    wrong = destination.replace(destination.rsplit("/", 1)[-1], f"{'0' * 64}.tgz")
    with pytest.raises(ValueError, match="does not match"):
        publisher.validate_publish_target(source, destination_s3=wrong, study_root=root)


def test_cli_defaults_to_no_aws_dry_run(tmp_path):
    source, _digest, root, destination = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--destination-s3",
            destination,
            "--study-root",
            root,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
