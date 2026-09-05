"""Offline deterministic source-archive builder tests; never uploads."""

from __future__ import annotations

import os
import tarfile

from scripts.launch import build_deterministic_archive as builder


def test_archive_is_content_addressed_mtime_stable_and_secret_free(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (source / "run.sh").chmod(0o755)
    (source / "secrets.env").write_text("TOKEN=nope\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/x.pyc").write_bytes(b"cache")
    output = tmp_path / "archives"
    study = "s3://bucket/owner/studies/long_context_v1"

    first, first_sha, first_uri = builder.build_archive(
        source,
        output_dir=output,
        component="wsmv2",
        study_root=study,
    )
    os.utime(source / "code.py", (123456789, 123456789))
    second, second_sha, second_uri = builder.build_archive(
        source,
        output_dir=output,
        component="wsmv2",
        study_root=study,
    )
    assert first == second
    assert first_sha == second_sha
    assert first.name == f"{first_sha}.tgz"
    assert first_uri == second_uri == f"{study}/code/wsmv2/{first_sha}.tgz"
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert "code.py" in names
        assert "run.sh" in names
        assert "secrets.env" not in names
        assert not any("__pycache__" in name for name in names)
        assert all(member.uid == member.gid == member.mtime == 0 for member in archive)
