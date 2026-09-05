"""Node-resident bootstrap and publication gate for the FS-R1 p5 screen."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CODE = Path("/opt/ml/code")
WORK = Path(os.environ.get("FS_R1_WORK_ROOT", "/opt/ml/framesamp-r1-screen"))
ENTRY = "gpu_framesamp_am_r1_screen_entry.sh"
STAGED = {
    "_robomme_framesamp_am_r1_screen_manifest.json",
    "_robomme_framesamp_am_r1_screen_packet.json",
    "_robomme_framesamp_am_r1_canary_manifest.json",
    "_robomme_framesamp_am_r1_canary_receipt.json",
}
RUNTIME = {"source": "upstream_uv_lock", "jax": "0.5.3", "orbax_checkpoint": "0.11.13"}


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode} command={command[:3]} "
            f"stdout={completed.stdout[-4000:]!r} stderr={completed.stderr[-4000:]!r}"
        )
    return completed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(digest: Any, value: object) -> None:
    data = value if isinstance(value, bytes) else str(value).encode()
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _normalized_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(CODE.rglob("*"), key=lambda item: item.relative_to(CODE).as_posix()):
        relative = path.relative_to(CODE).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if relative in STAGED:
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"staged FS-R1 input is not a regular file: {relative}")
            continue
        if relative == ENTRY:
            if mode != 0o777:
                raise RuntimeError(f"runtime entry mode must be 0777, got {oct(mode)}")
            mode = 0o755
        _field(digest, relative)
        _field(digest, oct(mode))
        if path.is_symlink():
            _field(digest, "symlink")
            _field(digest, os.readlink(path))
        elif path.is_dir():
            _field(digest, "directory")
        elif path.is_file():
            _field(digest, "file")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    _field(digest, block)
        else:
            raise RuntimeError(f"unsupported source entry: {relative}")
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _download(uri: str, destination: Path, expected_sha256: str) -> None:
    incomplete = destination.with_name(destination.name + ".incomplete")
    _run(["aws", "s3", "cp", uri, str(incomplete), "--only-show-errors"])
    if _sha256_file(incomplete) != expected_sha256:
        raise RuntimeError(f"download SHA mismatch for {uri}")
    incomplete.replace(destination)


def _split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _require_empty_namespace(uri: str) -> None:
    bucket, prefix = _split_s3(uri.rstrip("/") + "/")
    response = _run(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1",
            "--output",
            "json",
            "--no-cli-pager",
        ]
    )
    value = json.loads(response.stdout)
    if value.get("KeyCount", 0) != 0 or value.get("Contents"):
        raise RuntimeError("FS-R1 screen namespace is not empty")


def _one_directory(root: Path, pattern: str) -> Path:
    matches = [path for path in root.rglob(pattern) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one runtime directory {pattern}, got {matches}")
    return matches[0]


def main() -> int:
    if not str(WORK).startswith("/opt/ml/") or WORK == Path("/opt/ml"):
        raise ValueError("unsafe FS-R1 screen work root")
    required = (
        "FS_R1_SCREEN_ID FS_R1_SCREEN_MANIFEST_SHA256 FS_R1_SCREEN_NAMESPACE_S3 "
        "FS_R1_SCREEN_COMPLETION_S3 FS_R1_EXPECTED_SOURCE_SHA256 FS_R1_PACKET_FILE_SHA256 "
        "FS_R1_PACKET_SHA256 FS_R1_CANARY_ID FS_R1_CANARY_MANIFEST_SHA256 "
        "FS_R1_CANARY_RECEIPT_SHA256 FS_R1_CANARY_RECEIPT_FILE_SHA256 "
        "FS_R1_CANARY_RECEIPT_S3 FS_R1_CHECKPOINT_S3 FS_R1_CHECKPOINT_ARCHIVE_SHA256 "
        "FS_R1_CHECKPOINT_SEMANTIC_SHA256 FS_R1_OVERLAY_MANIFEST_SHA256 "
        "ROBOMME_EVAL_RUNTIME_S3 ROBOMME_EVAL_RUNTIME_SHA256 ROBOMME_EVAL_VISION_S3 "
        "ROBOMME_EVAL_VISION_SHA256 ROBOMME_EVAL_UPSTREAM_REPO "
        "ROBOMME_EVAL_UPSTREAM_COMMIT SM_USE_RESERVED_CAPACITY"
    ).split()
    environment = {key: os.environ[key] for key in required}
    if environment["SM_USE_RESERVED_CAPACITY"] != "1":
        raise ValueError("p5 reserved-capacity routing flag drifted")
    for name in STAGED:
        path = CODE / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"staged FS-R1 screen input absent: {name}")
    actual_source_sha = _normalized_source_sha256()
    if actual_source_sha != environment["FS_R1_EXPECTED_SOURCE_SHA256"]:
        raise RuntimeError(
            f"FS-R1 normalized source mismatch {actual_source_sha} != {environment['FS_R1_EXPECTED_SOURCE_SHA256']}"
        )
    WORK.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "uv-cache", "jax-cache", "out", "source-parent"):
        (WORK / name).mkdir()
    package_link = WORK / "source-parent/robomme_integration"
    package_link.symlink_to(CODE, target_is_directory=True)
    if package_link.resolve() != CODE:
        raise RuntimeError("FS-R1 package reparenting failed")
    sys.path.insert(0, str(WORK / "source-parent"))
    from robomme_integration.eval.framesamp_am_r1_screen_cloud import (
        validate_environment,
        validate_staged_inputs,
    )

    manifest_path = CODE / "_robomme_framesamp_am_r1_screen_manifest.json"
    packet_path = CODE / "_robomme_framesamp_am_r1_screen_packet.json"
    canary_manifest_path = CODE / "_robomme_framesamp_am_r1_canary_manifest.json"
    canary_receipt_path = CODE / "_robomme_framesamp_am_r1_canary_receipt.json"
    manifest = _load(manifest_path)
    packet = _load(packet_path)
    canary_manifest = _load(canary_manifest_path)
    canary_receipt = _load(canary_receipt_path)
    validate_environment(manifest, environment)
    validate_staged_inputs(
        manifest=manifest,
        packet=packet,
        packet_file_sha256=_sha256_file(packet_path),
        canary_manifest=canary_manifest,
        canary_receipt=canary_receipt,
        canary_receipt_file_sha256=_sha256_file(canary_receipt_path),
    )
    _require_empty_namespace(environment["FS_R1_SCREEN_NAMESPACE_S3"])
    canonical_canary = WORK / "canonical-canary.json"
    _download(
        environment["FS_R1_CANARY_RECEIPT_S3"],
        canonical_canary,
        environment["FS_R1_CANARY_RECEIPT_FILE_SHA256"],
    )
    if canonical_canary.read_bytes() != canary_receipt_path.read_bytes():
        raise RuntimeError("canonical FS-R1 canary receipt differs from staged receipt")

    runtime_archive = WORK / "runtime.tgz"
    checkpoint_archive = WORK / "checkpoint.tgz"
    vision = WORK / "siglip_params.pkl"
    _download(
        environment["ROBOMME_EVAL_RUNTIME_S3"],
        runtime_archive,
        environment["ROBOMME_EVAL_RUNTIME_SHA256"],
    )
    _download(
        environment["FS_R1_CHECKPOINT_S3"],
        checkpoint_archive,
        environment["FS_R1_CHECKPOINT_ARCHIVE_SHA256"],
    )
    _download(
        environment["ROBOMME_EVAL_VISION_S3"],
        vision,
        environment["ROBOMME_EVAL_VISION_SHA256"],
    )
    runtime_root = WORK / "runtime"
    checkpoint_root = WORK / "checkpoint"
    upstream = WORK / "upstream/robomme_policy_learning"
    runtime_root.mkdir()
    checkpoint_root.mkdir()
    upstream.parent.mkdir()
    _run(["tar", "xzf", str(runtime_archive), "-C", str(runtime_root)])
    _run(["tar", "xzf", str(checkpoint_archive), "-C", str(checkpoint_root)])
    _run(["git", "init", "-q", str(upstream)], environment={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    _run(["git", "-C", str(upstream), "remote", "add", "origin", environment["ROBOMME_EVAL_UPSTREAM_REPO"]])
    git_environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    _run(
        [
            "git",
            "-C",
            str(upstream),
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            environment["ROBOMME_EVAL_UPSTREAM_COMMIT"],
        ],
        environment=git_environment,
    )
    _run(["git", "-C", str(upstream), "checkout", "-q", "--detach", "FETCH_HEAD"])
    if (
        _run(["git", "-C", str(upstream), "rev-parse", "HEAD"]).stdout.strip()
        != environment["ROBOMME_EVAL_UPSTREAM_COMMIT"]
    ):
        raise RuntimeError("FS-R1 upstream checkout commit drifted")
    upstream_environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(upstream / ".venv"),
        "UV_CACHE_DIR": str(WORK / "uv-cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    _run(["uv", "sync", "--frozen"], cwd=upstream, environment=upstream_environment, timeout=1_800)
    policy_python = upstream / ".venv/bin/python"
    versions = _run(
        [
            str(policy_python),
            "-c",
            "import jax, orbax.checkpoint as ocp; print(jax.__version__+':'+ocp.__version__)",
        ]
    ).stdout.strip()
    if versions != f"{RUNTIME['jax']}:{RUNTIME['orbax_checkpoint']}":
        raise RuntimeError(f"FS-R1 policy runtime drifted: {versions}")

    overlay = WORK / "policy-overlay"
    overlay_environment = {
        **upstream_environment,
        "PYTHONPATH": f"{WORK / 'source-parent'}:{upstream / 'src'}",
    }
    overlay_script = (
        "import sys; from robomme_integration.training.framesamp_am_policy_overlay import "
        "stage_framesamp_am_policy_overlay, verify_framesamp_am_policy_overlay; "
        "actual=stage_framesamp_am_policy_overlay(sys.argv[1],sys.argv[2]); "
        "assert actual==sys.argv[3], (actual,sys.argv[3]); "
        "verify_framesamp_am_policy_overlay(sys.argv[2],expected_manifest_sha256=sys.argv[3])"
    )
    _run(
        [
            str(policy_python),
            "-B",
            "-c",
            overlay_script,
            str(upstream),
            str(overlay),
            environment["FS_R1_OVERLAY_MANIFEST_SHA256"],
        ],
        environment=overlay_environment,
    )
    simulator_env = _one_directory(runtime_root, "env-v0.4.0")
    robomme_src = _one_directory(runtime_root, "robomme-benchmark-f2b540e6/src")
    maniskill = _one_directory(runtime_root, "ManiSkill-07be6fbc")
    normalized = WORK / "runtime-normalized"
    normalized.mkdir()
    (normalized / "env-v0.4.0").symlink_to(simulator_env, target_is_directory=True)
    (normalized / "ManiSkill-07be6fbc").symlink_to(maniskill, target_is_directory=True)
    (normalized / "robomme-benchmark-f2b540e6").symlink_to(robomme_src.parent, target_is_directory=True)
    vision_root = WORK / "vision/pi05_vision_encoder"
    vision_root.mkdir(parents=True)
    shutil.move(str(vision), vision_root / "siglip_params.pkl")
    checkpoint = checkpoint_root / "perceptual-framesamp-modul/79999"
    if not (checkpoint / "params").is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise RuntimeError("FS-R1 released checkpoint extraction is incomplete")

    screen_environment = {
        **upstream_environment,
        "TMPDIR": str(WORK / "tmp"),
        "JAX_COMPILATION_CACHE_DIR": str(WORK / "jax-cache"),
        "OPENPI_DATA_HOME": str(WORK / "vision"),
        "PYTHONPATH": os.pathsep.join(
            [
                str(overlay / "src"),
                str(WORK / "source-parent"),
                str(upstream / "src"),
                str(upstream / ".venv/lib/python3.11/site-packages"),
                str(simulator_env / "lib/python3.11/site-packages"),
            ]
        ),
    }
    command = [
        str(policy_python),
        "-B",
        "-m",
        "robomme_integration.eval.framesamp_am_r1_screen",
        "--packet",
        str(packet_path),
        "--output-dir",
        str(WORK / "out"),
        "--policy-overlay",
        str(overlay),
        "--overlay-manifest-sha256",
        environment["FS_R1_OVERLAY_MANIFEST_SHA256"],
        "--official-checkout",
        str(upstream),
        "--checkpoint",
        str(checkpoint),
        "--runtime-root",
        str(normalized),
        "--completion-s3",
        manifest["publication"]["completion_s3"],
        "--confirm-publish",
    ]
    for row in manifest["publication"]["results"]:
        command.extend(["--result-uri", f"{row['cell_id']}={row['result_s3']}"])
    _run(command, cwd=overlay, environment=screen_environment, timeout=85_000)
    completion = WORK / "out/screen.complete.json"
    from robomme_integration.eval.framesamp_am_r1_screen_cloud import validate_completion

    validate_completion(_load(completion), manifest)
    canonical_completion = WORK / "canonical-screen-complete.json"
    _download(
        manifest["publication"]["completion_s3"],
        canonical_completion,
        _sha256_file(completion),
    )
    if canonical_completion.read_bytes() != completion.read_bytes():
        raise RuntimeError("FS-R1 screen completion readback differs")
    print(f"FS-R1 P5 SCREEN COMPLETE {environment['FS_R1_SCREEN_ID']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
