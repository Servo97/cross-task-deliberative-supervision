"""Node-resident bootstrap and create-once publication gate for FS-B1."""

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
WORK = Path(os.environ.get("FS_B1_WORK_ROOT", "/opt/ml/framesamp-b1-canary"))
ENTRY = "gpu_framesamp_b1_entry.sh"
STAGED = {"_robomme_framesamp_b1_manifest.json", "_robomme_framesamp_b1_packet.json"}
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


def _sha_file(path: Path) -> str:
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
                raise RuntimeError(f"staged B1 input is not a regular file: {relative}")
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
            raise RuntimeError(f"unsupported B1 source entry: {relative}")
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _download(uri: str, destination: Path, expected_sha256: str) -> None:
    incomplete = destination.with_name(destination.name + ".incomplete")
    _run(["aws", "s3", "cp", uri, str(incomplete), "--only-show-errors"])
    if _sha_file(incomplete) != expected_sha256:
        raise RuntimeError(f"B1 download SHA mismatch for {uri}")
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
        raise RuntimeError("B1 canary namespace is not empty")


def _one_directory(root: Path, pattern: str) -> Path:
    matches = [path for path in root.rglob(pattern) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one B1 runtime directory {pattern}, got {matches}")
    return matches[0]


def main() -> int:
    if not str(WORK).startswith("/opt/ml/") or WORK == Path("/opt/ml"):
        raise ValueError("unsafe B1 work root")
    required = (
        "FS_B1_CANARY_ID FS_B1_MANIFEST_SHA256 FS_B1_NAMESPACE_S3 FS_B1_RECEIPT_S3 "
        "FS_B1_EXPECTED_SOURCE_SHA256 FS_B1_PACKET_FILE_SHA256 FS_B1_PACKET_SHA256 "
        "FS_B1_OVERLAY_MANIFEST_SHA256 FS_B1_OVERLAY_SOURCE_TREE_SHA256 "
        "FS_B1_CHECKPOINT_S3 FS_B1_CHECKPOINT_ARCHIVE_SHA256 "
        "FS_B1_CHECKPOINT_SEMANTIC_SHA256 ROBOMME_EVAL_RUNTIME_S3 "
        "ROBOMME_EVAL_RUNTIME_SHA256 ROBOMME_EVAL_VISION_S3 ROBOMME_EVAL_VISION_SHA256 "
        "ROBOMME_EVAL_UPSTREAM_REPO ROBOMME_EVAL_UPSTREAM_COMMIT SM_USE_RESERVED_CAPACITY"
    ).split()
    environment = {key: os.environ[key] for key in required}
    if environment["SM_USE_RESERVED_CAPACITY"] != "1":
        raise ValueError("B1 p5 reserved-capacity routing flag drifted")
    for name in STAGED:
        path = CODE / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"staged B1 input absent: {name}")
    actual_source = _normalized_source_sha256()
    if actual_source != environment["FS_B1_EXPECTED_SOURCE_SHA256"]:
        raise RuntimeError(
            f"B1 normalized source mismatch {actual_source} != {environment['FS_B1_EXPECTED_SOURCE_SHA256']}"
        )
    WORK.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "uv-cache", "jax-cache", "out", "source-parent"):
        (WORK / name).mkdir()
    package_link = WORK / "source-parent/robomme_integration"
    package_link.symlink_to(CODE, target_is_directory=True)
    if package_link.resolve() != CODE:
        raise RuntimeError("B1 source reparenting failed")
    sys.path.insert(0, str(WORK / "source-parent"))
    from robomme_integration.eval.framesamp_b1_cloud import (
        bind_cloud_receipt,
        validate_environment,
        validate_manifest,
        validate_receipt,
        validate_staged_packet,
    )

    manifest_path = CODE / "_robomme_framesamp_b1_manifest.json"
    packet_path = CODE / "_robomme_framesamp_b1_packet.json"
    manifest = _load(manifest_path)
    packet = _load(packet_path)
    validate_manifest(manifest)
    validate_environment(manifest, environment)
    if _sha_file(packet_path) != environment["FS_B1_PACKET_FILE_SHA256"]:
        raise RuntimeError("B1 staged packet file SHA mismatch")
    if packet.get("packet_sha256") != environment["FS_B1_PACKET_SHA256"]:
        raise RuntimeError("B1 staged packet internal seal mismatch")
    _require_empty_namespace(environment["FS_B1_NAMESPACE_S3"])

    runtime_archive = WORK / "runtime.tgz"
    checkpoint_archive = WORK / "checkpoint.tgz"
    vision = WORK / "siglip_params.pkl"
    _download(
        environment["ROBOMME_EVAL_RUNTIME_S3"],
        runtime_archive,
        environment["ROBOMME_EVAL_RUNTIME_SHA256"],
    )
    _download(
        environment["FS_B1_CHECKPOINT_S3"],
        checkpoint_archive,
        environment["FS_B1_CHECKPOINT_ARCHIVE_SHA256"],
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
    git_environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    _run(["git", "init", "-q", str(upstream)], environment=git_environment)
    _run(["git", "-C", str(upstream), "remote", "add", "origin", environment["ROBOMME_EVAL_UPSTREAM_REPO"]])
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
        timeout=600.0,
    )
    _run(["git", "-C", str(upstream), "checkout", "-q", "--detach", "FETCH_HEAD"])
    if (
        _run(["git", "-C", str(upstream), "rev-parse", "HEAD"]).stdout.strip()
        != environment["ROBOMME_EVAL_UPSTREAM_COMMIT"]
    ):
        raise RuntimeError("B1 upstream checkout commit drifted")
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
            "-B",
            "-c",
            "import jax, orbax.checkpoint as ocp; print(jax.__version__+':'+ocp.__version__)",
        ]
    ).stdout.strip()
    if versions != f"{RUNTIME['jax']}:{RUNTIME['orbax_checkpoint']}":
        raise RuntimeError(f"B1 policy runtime drifted: {versions}")

    overlay = WORK / "policy-overlay"
    overlay_environment = {
        **upstream_environment,
        "PYTHONPATH": f"{WORK / 'source-parent'}:{upstream / 'src'}",
    }
    overlay_script = (
        "import sys; from robomme_integration.training.framesamp_b1_policy_overlay import "
        "stage_framesamp_b1_policy_overlay, verify_framesamp_b1_policy_overlay; "
        "actual=stage_framesamp_b1_policy_overlay(sys.argv[1],sys.argv[2]); "
        "assert actual==sys.argv[3], (actual,sys.argv[3]); "
        "m=verify_framesamp_b1_policy_overlay(sys.argv[2],expected_manifest_sha256=sys.argv[3]); "
        "assert m['source_tree_sha256']==sys.argv[4], (m['source_tree_sha256'],sys.argv[4])"
    )
    _run(
        [
            str(policy_python),
            "-B",
            "-c",
            overlay_script,
            str(upstream),
            str(overlay),
            environment["FS_B1_OVERLAY_MANIFEST_SHA256"],
            environment["FS_B1_OVERLAY_SOURCE_TREE_SHA256"],
        ],
        environment=overlay_environment,
    )
    validate_staged_packet(
        manifest=manifest,
        packet=packet,
        packet_file_sha256=_sha_file(packet_path),
        source_root=WORK / "source-parent",
        overlay_root=overlay,
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
        raise RuntimeError("B1 released checkpoint extraction is incomplete")
    marker = checkpoint.parent / f".EXTRACTED-{environment['FS_B1_CHECKPOINT_SEMANTIC_SHA256']}"
    if not marker.is_file():
        raise RuntimeError("B1 checkpoint semantic extraction marker is absent")

    canary_environment = {
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
    mechanism_receipt = WORK / "out/mechanism-receipt.json"
    receipt = WORK / "out/canary.complete.json"
    _run(
        [
            str(policy_python),
            "-B",
            "-m",
            "robomme_integration.eval.framesamp_b1_canary",
            "--output-dir",
            str(WORK / "out"),
            "--policy-overlay",
            str(overlay),
            "--overlay-manifest-sha256",
            environment["FS_B1_OVERLAY_MANIFEST_SHA256"],
            "--official-checkout",
            str(upstream),
            "--checkpoint",
            str(checkpoint),
            "--runtime-root",
            str(normalized),
            "--output",
            str(mechanism_receipt),
        ],
        cwd=overlay,
        environment=canary_environment,
        timeout=13_000,
    )
    mechanism_proof = _load(mechanism_receipt)
    proof = bind_cloud_receipt(mechanism_proof, manifest)
    receipt.write_text(
        json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_receipt(proof, manifest)
    _require_empty_namespace(environment["FS_B1_NAMESPACE_S3"])
    bucket, key = _split_s3(environment["FS_B1_RECEIPT_S3"])
    _run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(receipt),
            "--content-type",
            "application/json",
            "--if-none-match",
            "*",
        ]
    )
    readback = WORK / "readback.json"
    _download(environment["FS_B1_RECEIPT_S3"], readback, _sha_file(receipt))
    if readback.read_bytes() != receipt.read_bytes():
        raise RuntimeError("B1 receipt readback differs")
    validate_receipt(_load(readback), manifest)
    print(f"FS-B1 P5 RUNTIME CANARY COMPLETE (NOT SCORED) {environment['FS_B1_CANARY_ID']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
