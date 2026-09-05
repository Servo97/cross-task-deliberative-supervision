#!/usr/bin/env python3
"""Run a project S0/Q0/A6 checkpoint under the pinned RoboMME fixed-800 protocol.

This is the only supported local entry point for ``project_exact_server.py`` plus
``project_exact_eval.py``.  It binds evaluation to checkpoint bytes, a freshly staged snapshot of
the project sources, the pinned OpenPI source archive, and clean official RoboMME source trees.
The policy process is started once; only a simulator process that fails with SAPIEN's exact known
Vulkan-driver signature is retried.

The wrapper performs no downloading, uploading, checkpoint discovery, or cloud submission.  Every
input path and the expected checkpoint manifest digest must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from robomme_integration.eval.project_exact_server import PROTOCOL_ID, validate_project_arm
from robomme_integration.eval.project_exact_source_audit import (
    BENCHMARK_SOURCE_COMMIT,
    MANISKILL_SOURCE_COMMIT,
    POLICY_SOURCE_COMMIT,
    REFERENCE_EVALUATOR_SHA256,
    audit_imported_git_source,
    sha256_file,
)
from robomme_integration.fleet.checkpoint import build as build_checkpoint_manifest

OPENPI_ARCHIVE_SHA256 = "ed923b2c27d2f608d62cc4b5ca89d5b80c14739dba1ab81d6f53d8013bcb66ad"
VULKAN_DRIVER_SIGNATURE = "vk::createInstanceUnique: ErrorIncompatibleDriver"
METHODS = {arm: f"project-exact-{arm}" for arm in ("s0", "q0", "a6", "v4_s0")}
_HEX64 = frozenset("0123456789abcdef")
_ATTEMPT_LOG_RE = re.compile(r"eval-attempt-([0-9]{4,})\.log")
_SOURCE_IGNORES = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "tests",
    }
)


def _require_sha256(label: str, value: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(character not in _HEX64 for character in value):
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def validate_runner_arm(value: str) -> str:
    """Accept only execution-only project controls and explain the two important exclusions."""
    arm = str(value).lower()
    if arm == "official_recipe_lerobot":
        raise ValueError(
            "official_recipe_lerobot has a distinct two-view recipe/evaluator contract and is "
            "forbidden in the project-exact S0/Q0/A6 runner"
        )
    return validate_project_arm(arm)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_ignore(_directory: str, names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if name in _SOURCE_IGNORES or name.endswith((".pyc", ".pyo", ".log", ".md")) or name.startswith(".env")
    ]


def source_tree_sha256(root: str | Path) -> str:
    """Hash paths, types, modes, symlink targets, and bytes without filesystem ordering."""
    root = Path(root).resolve()
    digest = hashlib.sha256()

    def field(value: bytes | str) -> None:
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        field(relative)
        field(oct(stat.S_IMODE(mode)))
        if path.is_symlink():
            field("symlink")
            field(os.readlink(path))
        elif path.is_dir():
            field("directory")
        elif path.is_file():
            field("file")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    field(block)
        else:
            raise RuntimeError(f"unsupported staged source entry: {path}")
    return digest.hexdigest()


def _reject_source_symlinks(root: Path, label: str) -> None:
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise RuntimeError(f"{label} source contains unsupported symlink: {links[0]}")


@dataclass(frozen=True)
class ProjectSnapshot:
    root: Path
    sha256: str
    robomme_sha256: str
    vla_eval_sha256: str


def stage_project_snapshot(
    *,
    project_root: str | Path,
    vla_eval_root: str | Path,
    work_root: str | Path,
) -> ProjectSnapshot:
    """Stage the sources actually put on the policy/evaluator ``PYTHONPATH`` and derive identity."""
    project_package = Path(project_root).resolve() / "robomme_integration"
    vla_eval_package = Path(vla_eval_root).resolve() / "src" / "vla_eval"
    if not (project_package / "eval" / "project_exact_runner.py").is_file():
        raise RuntimeError(f"project source is not this exact-runner tree: {project_package}")
    if not (vla_eval_package / "model_servers" / "base.py").is_file():
        raise RuntimeError(f"vla-evaluation-harness source is incomplete: {vla_eval_package}")
    _reject_source_symlinks(project_package, "RoboMME integration")
    _reject_source_symlinks(vla_eval_package, "vla-evaluation-harness")

    source_parent = Path(work_root).resolve() / "sources"
    source_parent.mkdir(parents=True, exist_ok=True)
    temporary = source_parent / f".project-{uuid.uuid4().hex}.incomplete"
    temporary.mkdir()
    try:
        shutil.copytree(project_package, temporary / "robomme_integration", ignore=_copy_ignore)
        shutil.copytree(vla_eval_package, temporary / "vla_eval", ignore=_copy_ignore)
        robomme_sha = source_tree_sha256(temporary / "robomme_integration")
        vla_eval_sha = source_tree_sha256(temporary / "vla_eval")
        combined_sha = source_tree_sha256(temporary)
        destination = source_parent / f"project-{combined_sha}"
        if destination.exists():
            if source_tree_sha256(destination) != combined_sha:
                raise RuntimeError(f"content-addressed project snapshot was modified: {destination}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return ProjectSnapshot(destination, combined_sha, robomme_sha, vla_eval_sha)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_project_snapshot(
    *,
    work_root: str | Path,
    expected_sha256: str,
    expected_robomme_sha256: str,
    expected_vla_eval_sha256: str,
) -> ProjectSnapshot:
    """Reload the exact source snapshot sealed by an interrupted long evaluation."""

    combined_sha = _require_sha256("project source SHA256", expected_sha256)
    robomme_sha = _require_sha256("robomme_integration source SHA256", expected_robomme_sha256)
    vla_eval_sha = _require_sha256("vla_eval source SHA256", expected_vla_eval_sha256)
    root = Path(work_root).resolve() / "sources" / f"project-{combined_sha}"
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"sealed project source snapshot is missing: {root}")
    _reject_source_symlinks(root, "sealed project")
    actual = ProjectSnapshot(
        root=root,
        sha256=source_tree_sha256(root),
        robomme_sha256=source_tree_sha256(root / "robomme_integration"),
        vla_eval_sha256=source_tree_sha256(root / "vla_eval"),
    )
    expected = ProjectSnapshot(
        root=root, sha256=combined_sha, robomme_sha256=robomme_sha, vla_eval_sha256=vla_eval_sha
    )
    if actual != expected:
        raise RuntimeError("sealed project source snapshot content differs from its orchestration manifest")
    return actual


def _extract_regular_tar(archive_path: Path, destination: Path) -> None:
    """Extract only ordinary files/directories; reject links, devices, and path traversal."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            pure = Path(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise RuntimeError(f"unsafe OpenPI archive member: {member.name!r}")
            target = destination.joinpath(*pure.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f"OpenPI archive member escapes extraction root: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(stat.S_IMODE(member.mode))
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read OpenPI archive member: {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                target.chmod(stat.S_IMODE(member.mode))
            else:
                raise RuntimeError(f"unsupported OpenPI archive entry type: {member.name!r}")


@dataclass(frozen=True)
class OpenPISnapshot:
    root: Path
    archive_sha256: str
    tree_sha256: str


def stage_openpi_archive(*, archive: str | Path, work_root: str | Path) -> OpenPISnapshot:
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise RuntimeError(f"pinned OpenPI archive is missing: {archive}")
    actual_archive_sha = sha256_file(archive)
    if actual_archive_sha != OPENPI_ARCHIVE_SHA256:
        raise RuntimeError(f"OpenPI archive SHA256 drifted: {actual_archive_sha} != {OPENPI_ARCHIVE_SHA256}")
    source_parent = Path(work_root).resolve() / "sources"
    source_parent.mkdir(parents=True, exist_ok=True)
    temporary = source_parent / f".openpi-{uuid.uuid4().hex}.incomplete"
    temporary.mkdir()
    try:
        _extract_regular_tar(archive, temporary)
        for required in ("src/openpi/__init__.py", "packages/openpi-client/src/openpi_client/__init__.py"):
            if not (temporary / required).is_file():
                raise RuntimeError(f"pinned OpenPI archive is missing {required}")
        tree_sha = source_tree_sha256(temporary)
        destination = source_parent / f"openpi-{OPENPI_ARCHIVE_SHA256}"
        if destination.exists():
            if source_tree_sha256(destination) != tree_sha:
                raise RuntimeError(f"extracted pinned OpenPI source was modified: {destination}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return OpenPISnapshot(destination, OPENPI_ARCHIVE_SHA256, tree_sha)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_checkpoint_tree(
    *, checkpoint_root: str | Path, manifest_path: str | Path, expected_sha256: str
) -> tuple[str, str]:
    expected = _require_sha256("expected checkpoint tree manifest SHA256", expected_sha256)
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    uri = payload.get("checkpoint_uri")
    if not isinstance(uri, str) or not uri:
        raise RuntimeError("checkpoint tree manifest has no nonempty checkpoint_uri")
    if "official_recipe_lerobot" in uri.lower():
        raise RuntimeError("official_recipe_lerobot checkpoint is forbidden in this evaluator")
    # Rebuild in a private directory so simultaneous S0/Q0 validations cannot race on a shared
    # temporary filename beside the supplied manifests.
    with tempfile.TemporaryDirectory(prefix="project-exact-checkpoint-manifest-") as directory:
        rebuilt = Path(directory) / "tree.json"
        actual = build_checkpoint_manifest(
            checkpoint_root,
            uri,
            rebuilt,
            require_finalized=False,
        )
        if rebuilt.read_bytes() != manifest_path.read_bytes():
            raise ValueError("local checkpoint tree differs from its sealed manifest")
    if actual != expected:
        raise RuntimeError(f"checkpoint tree manifest SHA256 mismatch: {actual} != {expected}")
    return actual, uri


def _verify_git_sources(*, policy_root: Path, benchmark_root: Path, maniskill_root: Path) -> None:
    records = (
        (
            "RoboMME policy/evaluator",
            policy_root / "examples" / "robomme" / "env_runner.py",
            POLICY_SOURCE_COMMIT,
        ),
        (
            "RoboMME benchmark",
            benchmark_root / "src" / "robomme" / "env_record_wrapper" / "__init__.py",
            BENCHMARK_SOURCE_COMMIT,
        ),
        ("ManiSkill", maniskill_root / "mani_skill" / "__init__.py", MANISKILL_SOURCE_COMMIT),
    )
    roots = {audit_imported_git_source(label, anchor, commit).root for label, anchor, commit in records}
    expected_roots = {policy_root.resolve(), benchmark_root.resolve(), maniskill_root.resolve()}
    if roots != expected_roots or len(roots) != 3:
        raise RuntimeError("pinned policy, benchmark, and ManiSkill roots are not three exact Git trees")


def _require_python(path: str | Path, label: str) -> Path:
    # Preserve the venv entry path. Resolving ``.venv/bin/python -> /usr/bin/python`` would invoke
    # the system interpreter and silently discard the venv's site-packages.
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"{label} Python is missing or not executable: {path}")
    return path


@dataclass(frozen=True)
class RuntimeContract:
    arm: str
    method: str
    checkpoint_root: Path
    checkpoint_sha256: str
    checkpoint_uri: str
    project: ProjectSnapshot
    openpi: OpenPISnapshot
    openpi_python: Path
    simulator_python: Path
    policy_root: Path
    benchmark_root: Path
    maniskill_root: Path
    output: Path
    port: int
    server_source_sha256: str
    evaluator_source_sha256: str
    policy_runtime: dict[str, Any]
    simulator_runtime: dict[str, Any]

    @property
    def server_pythonpath(self) -> str:
        return os.pathsep.join(
            (
                str(self.project.root / "robomme_integration" / "compat"),
                str(self.project.root),
                str(self.openpi.root / "src"),
                str(self.openpi.root / "packages" / "openpi-client" / "src"),
            )
        )

    @property
    def evaluator_pythonpath(self) -> str:
        return os.pathsep.join(
            (
                str(self.project.root),
                str(self.policy_root / "examples" / "robomme"),
                str(self.policy_root / "packages" / "openpi-client" / "src"),
                str(self.benchmark_root / "src"),
                str(self.maniskill_root),
            )
        )


def _runtime_fingerprint(
    python: Path,
    *,
    distributions: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Collect deterministic interpreter/package versions without importing heavy frameworks."""
    program = """
import hashlib, importlib.metadata, json, platform, sys
names = json.loads(sys.argv[1])
packages = {}
for name in names:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
value = {
    "implementation": sys.implementation.name,
    "python_version": platform.python_version(),
    "python_build": list(platform.python_build()),
    "packages": packages,
}
canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
value["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
print("RUNTIME_FINGERPRINT=" + json.dumps(value, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", program, json.dumps(list(distributions))],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"runtime fingerprint failed for {python}: {detail}")
    prefix = "RUNTIME_FINGERPRINT="
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"runtime fingerprint returned no unique sealed record for {python}")
    value = json.loads(lines[0][len(prefix) :])
    if set(value.get("packages", {})) != set(distributions):
        raise RuntimeError(f"runtime fingerprint package coverage drifted for {python}")
    _require_sha256("runtime fingerprint", value.get("fingerprint_sha256", ""))
    return value


def build_runtime_contract(args: argparse.Namespace, *, project_root: Path | None = None) -> RuntimeContract:
    arm = validate_runner_arm(args.arm)
    checkpoint_sha, checkpoint_uri = verify_checkpoint_tree(
        checkpoint_root=args.checkpoint_root,
        manifest_path=args.checkpoint_manifest,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    root = project_root or Path(__file__).resolve().parents[2]
    output_root = Path(args.output).resolve()
    work_root = Path(args.work_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    for label, path in (("work root", work_root), ("output root", output_root)):
        if path.is_relative_to(root.resolve()):
            raise RuntimeError(f"{label} must live outside the project source tree: {path}")
        if path.is_relative_to(checkpoint_root) or checkpoint_root.is_relative_to(path):
            raise RuntimeError(f"{label} and checkpoint root must not contain one another: {path}")
    orchestration_path = output_root / "orchestration_manifest.json"
    if orchestration_path.is_file():
        try:
            prior_orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("existing orchestration manifest is not valid UTF-8 JSON") from error
        if not isinstance(prior_orchestration, dict):
            raise RuntimeError("existing orchestration manifest must be a JSON object")
        project = load_project_snapshot(
            work_root=work_root,
            expected_sha256=prior_orchestration.get("project_source_sha256", ""),
            expected_robomme_sha256=prior_orchestration.get("robomme_integration_source_sha256", ""),
            expected_vla_eval_sha256=prior_orchestration.get("vla_eval_source_sha256", ""),
        )
    else:
        project = stage_project_snapshot(
            project_root=root,
            vla_eval_root=args.vla_eval_root,
            work_root=work_root,
        )
    openpi = stage_openpi_archive(archive=args.openpi_archive, work_root=args.work_root)
    policy_root = Path(args.policy_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    maniskill_root = Path(args.maniskill_root).resolve()
    _verify_git_sources(
        policy_root=policy_root,
        benchmark_root=benchmark_root,
        maniskill_root=maniskill_root,
    )
    port = int(args.port)
    if not 1 <= port <= 65535:
        raise ValueError("port must lie in [1,65535]")
    server_source = project.root / "robomme_integration" / "eval" / "project_exact_server.py"
    evaluator_source = project.root / "robomme_integration" / "eval" / "project_exact_eval.py"
    openpi_python = _require_python(args.openpi_python, "OpenPI policy")
    simulator_python = _require_python(args.simulator_python, "RoboMME simulator")
    contract = RuntimeContract(
        arm=arm,
        method=METHODS[arm],
        checkpoint_root=checkpoint_root,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_uri=checkpoint_uri,
        project=project,
        openpi=openpi,
        openpi_python=openpi_python,
        simulator_python=simulator_python,
        policy_root=policy_root,
        benchmark_root=benchmark_root,
        maniskill_root=maniskill_root,
        output=output_root,
        port=port,
        server_source_sha256=sha256_file(server_source),
        evaluator_source_sha256=sha256_file(evaluator_source),
        policy_runtime={},
        simulator_runtime={},
    )
    contract = replace(
        contract,
        policy_runtime=_runtime_fingerprint(
            openpi_python,
            distributions=("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy"),
            cwd=openpi.root,
            environment=server_environment(contract, ""),
        ),
        simulator_runtime=_runtime_fingerprint(
            simulator_python,
            distributions=("torch", "gymnasium", "sapien", "numpy"),
            cwd=policy_root / "examples" / "robomme",
            environment=evaluator_environment(contract, ""),
        ),
    )
    _preflight_policy_imports(contract)
    return contract


def _preflight_policy_imports(contract: RuntimeContract) -> None:
    """Prove the policy interpreter resolves every local package from the staged snapshots."""
    expected = {
        "robocasa": str(contract.project.root / "robomme_integration" / "compat" / "robocasa"),
        "robomme_integration": str(contract.project.root / "robomme_integration"),
        "vla_eval": str(contract.project.root / "vla_eval"),
        "openpi": str(contract.openpi.root / "src" / "openpi"),
        "openpi_client": str(contract.openpi.root / "packages" / "openpi-client" / "src" / "openpi_client"),
    }
    program = """
import importlib, json, pathlib, sys
expected = json.loads(sys.argv[1])
actual = {}
for name, root in expected.items():
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    root_path = pathlib.Path(root).resolve()
    if not path.is_relative_to(root_path):
        raise SystemExit(f"{name} resolved outside staged source: {path} !< {root_path}")
    actual[name] = str(path.relative_to(root_path))
print(json.dumps(actual, sort_keys=True))
"""
    result = subprocess.run(
        [str(contract.openpi_python), "-c", program, json.dumps(expected, sort_keys=True)],
        cwd=contract.openpi.root,
        env=server_environment(contract, ""),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"staged policy import preflight failed: {detail}")


def orchestration_manifest(contract: RuntimeContract) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "estimand": "project_multitask_checkpoint_single_seed_exact_paper_protocol",
        "arm": contract.arm,
        "method": contract.method,
        "checkpoint_tree_manifest_sha256": contract.checkpoint_sha256,
        "checkpoint_uri": contract.checkpoint_uri,
        "project_source_sha256": contract.project.sha256,
        "robomme_integration_source_sha256": contract.project.robomme_sha256,
        "vla_eval_source_sha256": contract.project.vla_eval_sha256,
        "openpi_archive_sha256": contract.openpi.archive_sha256,
        "openpi_extracted_tree_sha256": contract.openpi.tree_sha256,
        "server_source_sha256": contract.server_source_sha256,
        "evaluator_source_sha256": contract.evaluator_source_sha256,
        "reference_evaluator_sha256": REFERENCE_EVALUATOR_SHA256,
        "policy_runtime": contract.policy_runtime,
        "simulator_runtime": contract.simulator_runtime,
        "policy_source_commit": POLICY_SOURCE_COMMIT,
        "benchmark_source_commit": BENCHMARK_SOURCE_COMMIT,
        "maniskill_source_commit": MANISKILL_SOURCE_COMMIT,
        "model_seed": 7,
        "action_horizon": 20,
        "execution_horizon": 16,
        "episodes": 800,
    }


def _seal_or_verify_manifest(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("refusing to resume output under a different orchestration contract")
    else:
        _atomic_json(path, value)


def server_command(contract: RuntimeContract) -> list[str]:
    return [
        str(contract.openpi_python),
        str(contract.project.root / "robomme_integration" / "eval" / "project_exact_server.py"),
        "--checkpoint",
        str(contract.checkpoint_root),
        "--arm",
        contract.arm,
        "--checkpoint-sha256",
        contract.checkpoint_sha256,
        "--project-source-sha256",
        contract.project.sha256,
        "--openpi-source-sha256",
        contract.openpi.archive_sha256,
        "--model-seed",
        "7",
        "--host",
        "127.0.0.1",
        "--port",
        str(contract.port),
    ]


def evaluator_command(contract: RuntimeContract) -> list[str]:
    return [
        str(contract.simulator_python),
        str(contract.project.root / "robomme_integration" / "eval" / "project_exact_eval.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(contract.port),
        "--output",
        str(contract.output / "evaluation"),
        "--method",
        contract.method,
        "--arm",
        contract.arm,
        "--checkpoint-sha256",
        contract.checkpoint_sha256,
        "--project-source-sha256",
        contract.project.sha256,
        "--openpi-source-sha256",
        contract.openpi.archive_sha256,
        "--server-source-sha256",
        contract.server_source_sha256,
        "--policy-source-commit",
        POLICY_SOURCE_COMMIT,
        "--benchmark-source-commit",
        BENCHMARK_SOURCE_COMMIT,
        "--maniskill-source-commit",
        MANISKILL_SOURCE_COMMIT,
    ]


def server_environment(contract: RuntimeContract, cuda_device: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": contract.server_pythonpath,
            "CUDA_VISIBLE_DEVICES": cuda_device,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def evaluator_environment(contract: RuntimeContract, cuda_device: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": contract.evaluator_pythonpath,
            "CUDA_VISIBLE_DEVICES": cuda_device,
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "MPLCONFIGDIR": str(contract.output / "matplotlib"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def is_retryable_renderer_failure(log_text: str) -> bool:
    """Retry the one renderer constructor failure observed in the released campaign."""
    return VULKAN_DRIVER_SIGNATURE in log_text


def _assert_port_free(port: int) -> None:
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise RuntimeError(f"local policy port {port} is already in use") from error
    finally:
        probe.close()


def _wait_for_server(process: subprocess.Popen[Any], port: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"project policy server exited before readiness with code {return_code}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200 and response.read() == b"OK\n":
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"project policy server did not become healthy within {timeout_seconds}s")


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_renderer_recovery(process: subprocess.Popen[Any], timeout_seconds: int) -> None:
    """Avoid burning the bounded retry budget while the host NVIDIA driver is unavailable."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited while waiting for driver recovery: {process.returncode}")
        try:
            probe = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            recovered = probe.returncode == 0
        except OSError:
            recovered = False
        if recovered:
            print("PROJECT_EXACT_NVIDIA_DRIVER_RECOVERED settling_seconds=5", flush=True)
            time.sleep(5)
            return
        remaining = max(0, int(deadline - time.monotonic()))
        print(f"PROJECT_EXACT_WAITING_FOR_NVIDIA_DRIVER remaining_seconds={remaining}", flush=True)
        time.sleep(30)
    raise RuntimeError(f"NVIDIA driver did not recover within {timeout_seconds}s")


def _validate_complete_scorecard(contract: RuntimeContract) -> dict[str, Any]:
    path = contract.output / "evaluation" / "scorecard.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "episodes": 800,
        "result_scale": "fraction_0_1",
        "protocol_id": PROTOCOL_ID,
        "method": contract.method,
        "arm": contract.arm,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "project_source_sha256": contract.project.sha256,
        "openpi_source_sha256": contract.openpi.archive_sha256,
        "server_source_sha256": contract.server_source_sha256,
        "evaluator_sha256": contract.evaluator_source_sha256,
        "reference_evaluator_sha256": REFERENCE_EVALUATOR_SHA256,
        "policy_source_commit": POLICY_SOURCE_COMMIT,
        "benchmark_source_commit": BENCHMARK_SOURCE_COMMIT,
        "maniskill_source_commit": MANISKILL_SOURCE_COMMIT,
        "model_seed": 7,
        "action_horizon": 20,
        "execution_horizon": 16,
        "estimand": "project_multitask_checkpoint_single_seed_exact_paper_protocol",
    }
    bad = {
        key: {"actual": value.get(key), "expected": wanted}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if bad:
        raise RuntimeError(f"project exact scorecard contract mismatch: {bad}")
    if len(value.get("task_success_rate", {})) != 16:
        raise RuntimeError("project exact scorecard does not contain all 16 task rates")
    return value


def _highest_attempt_log(log_directory: Path) -> int:
    attempts = []
    for path in log_directory.glob("eval-attempt-*.log"):
        match = _ATTEMPT_LOG_RE.fullmatch(path.name)
        if match is not None:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0)


def run(contract: RuntimeContract, args: argparse.Namespace) -> int:
    contract.output.mkdir(parents=True, exist_ok=True)
    (contract.output / "logs").mkdir(exist_ok=True)
    (contract.output / "matplotlib").mkdir(exist_ok=True)
    _seal_or_verify_manifest(
        contract.output / "orchestration_manifest.json",
        orchestration_manifest(contract),
    )
    complete_marker = contract.output / "PROJECT_EXACT_FIXED800_COMPLETE"
    if complete_marker.is_file():
        scorecard = _validate_complete_scorecard(contract)
        print(f"PROJECT_EXACT_ALREADY_COMPLETE successes={scorecard['successes']}/800")
        return 0

    retry_state_path = contract.output / "renderer_retry_state.json"
    retry_state = (
        json.loads(retry_state_path.read_text(encoding="utf-8"))
        if retry_state_path.is_file()
        else {"renderer_restarts": 0, "attempts": 0}
    )
    renderer_restarts = int(retry_state.get("renderer_restarts", -1))
    persisted_attempt = int(retry_state.get("attempts", -1))
    attempt = max(persisted_attempt, _highest_attempt_log(contract.output / "logs"))
    if renderer_restarts < 0 or persisted_attempt < 0:
        raise RuntimeError("invalid persisted renderer retry state")
    if renderer_restarts > args.max_renderer_restarts:
        raise RuntimeError("persisted renderer restart count already exceeds the configured bound")

    _assert_port_free(contract.port)
    server_log_path = contract.output / "logs" / "server.log"
    with server_log_path.open("ab", buffering=0) as server_log:
        process = subprocess.Popen(
            server_command(contract),
            cwd=contract.openpi.root,
            env=server_environment(contract, args.policy_cuda_device),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(process, contract.port, args.server_ready_timeout)
            while True:
                attempt += 1
                attempt_log_path = contract.output / "logs" / f"eval-attempt-{attempt:04d}.log"
                if attempt_log_path.exists():
                    raise RuntimeError(f"refusing to overwrite prior attempt log: {attempt_log_path}")
                # Allocate before opening/spawning. If the node dies mid-attempt, the next wrapper
                # starts at N+1 and preserves this interrupted log while the evaluator resumes its
                # separately atomic Boolean progress.
                _atomic_json(
                    retry_state_path,
                    {"renderer_restarts": renderer_restarts, "attempts": attempt},
                )
                with attempt_log_path.open("wb") as attempt_log:
                    completed = subprocess.run(
                        evaluator_command(contract),
                        cwd=contract.policy_root / "examples" / "robomme",
                        env=evaluator_environment(contract, args.simulator_cuda_device),
                        stdout=attempt_log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode == 0:
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"policy server exited during simulator attempt with code {process.returncode}")
                log_text = attempt_log_path.read_text(encoding="utf-8", errors="replace")
                if not is_retryable_renderer_failure(log_text):
                    raise RuntimeError(
                        f"project evaluator failed with non-renderer error (exit {completed.returncode}); "
                        f"see {attempt_log_path}"
                    )
                renderer_restarts += 1
                _atomic_json(
                    retry_state_path,
                    {"renderer_restarts": renderer_restarts, "attempts": attempt},
                )
                if renderer_restarts > args.max_renderer_restarts:
                    raise RuntimeError(
                        f"renderer restart limit exceeded ({renderer_restarts}>{args.max_renderer_restarts})"
                    )
                print(
                    f"PROJECT_EXACT_RENDERER_RESTART count={renderer_restarts}/"
                    f"{args.max_renderer_restarts} attempt_log={attempt_log_path}",
                    flush=True,
                )
                _wait_for_renderer_recovery(process, args.renderer_recovery_timeout)
        finally:
            _terminate(process)

    scorecard = _validate_complete_scorecard(contract)
    marker_tmp = complete_marker.with_name(complete_marker.name + ".tmp")
    marker_tmp.write_text(
        json.dumps(
            {
                "checkpoint_sha256": contract.checkpoint_sha256,
                "project_source_sha256": contract.project.sha256,
                "successes": scorecard["successes"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, complete_marker)
    print(f"PROJECT_EXACT_FIXED800_COMPLETE successes={scorecard['successes']}/800")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--openpi-archive", type=Path, required=True)
    parser.add_argument("--openpi-python", type=Path, required=True)
    parser.add_argument("--simulator-python", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--maniskill-root", type=Path, required=True)
    parser.add_argument("--vla-eval-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18720)
    parser.add_argument("--policy-cuda-device", default="0")
    parser.add_argument("--simulator-cuda-device", default="1")
    parser.add_argument("--server-ready-timeout", type=int, default=900)
    parser.add_argument("--renderer-recovery-timeout", type=int, default=1800)
    parser.add_argument("--max-renderer-restarts", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.server_ready_timeout < 1:
        parser.error("--server-ready-timeout must be positive")
    if args.renderer_recovery_timeout < 1:
        parser.error("--renderer-recovery-timeout must be positive")
    if args.max_renderer_restarts < 0:
        parser.error("--max-renderer-restarts must be nonnegative")
    try:
        contract = build_runtime_contract(args)
        return run(contract, args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
