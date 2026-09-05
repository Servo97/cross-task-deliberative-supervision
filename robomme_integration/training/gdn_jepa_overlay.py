"""Minimal, content-recorded OpenPI overlay for the RoboMME GDN+JEPA arm.

The canonical OpenPI archive rejects every pair of workspace interfaces.  GDN+JEPA needs one
exception: the ``tanh`` read interface (used by gated DeltaNet) may coexist with the train-only
``jepa_aux_target`` interface.  This module stages a node-local copy of the 1.7 MiB OpenPI Python
runtime and changes exactly the single mutual-exclusion predicate in ``Pi0Config``.  It never
mutates the canonical/shared OpenPI tree and does not change model or loss math.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

BASE_ARCHIVE_SHA256 = "ed923b2c27d2f608d62cc4b5ca89d5b80c14739dba1ab81d6f53d8013bcb66ad"
BASE_PI0_CONFIG_SHA256 = "0685f4d99091153a6b3dfd8c15f04c201ba327f3090e3cefcfb9b013eff62224"
# These two receipts are over ``src/openpi`` and ``scripts`` with runtime bytecode excluded.
# They are populated from the canonical ed923 archive and intentionally fail closed on drift.
BASE_RUNTIME_TREE_SHA256 = "a8448ee0f2b345f36b1c5c933776c48814bb0ac4084746c50e602e532f89ea65"
PATCHED_PI0_CONFIG_SHA256 = "94fc818525b9f96ff487d3f8f15b1f1a4155e7c1a53edc45ab005c349db15b8f"
PATCHED_RUNTIME_TREE_SHA256 = "30c15b473164d966f5d0ca4936e301aa6b5c8285ef3ed50b7b31c8d098e0a6ee"
BASE_RUNTIME_FILE_COUNT = 82

OVERLAY_VERSION = "robomme-gdn-jepa-config-v1"
OVERLAY_KIND = "robomme_gdn_jepa_openpi_overlay"
OVERLAY_MANIFEST = "ROBOMME_GDN_JEPA_OVERLAY.json"
PI0_CONFIG_RELATIVE = Path("src/openpi/models/pi0_config.py")
COPIED_ROOTS = (Path("src/openpi"), Path("scripts"))
ALLOWED_PAIR = frozenset({"tanh", "jepa_aux_target"})

_OLD_PREDICATE = "        if len(enabled) > 1:\n"
_NEW_PREDICATE = '        if len(enabled) > 1 and set(enabled) != {"tanh", "jepa_aux_target"}:\n'


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _ignored(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}


def runtime_inventory(root: Path) -> list[dict[str, Any]]:
    """Return the deterministic content inventory used to bind the copied runtime."""

    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for copied_root in COPIED_ROOTS:
        source = root / copied_root
        if not source.is_dir():
            raise ValueError(f"OpenPI runtime is missing {copied_root}")
        for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root)
            if _ignored(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"OpenPI runtime contains a symlink: {relative}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"OpenPI runtime contains an unsupported entry: {relative}")
            entries.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not entries:
        raise ValueError(f"OpenPI runtime is empty: {root}")
    return entries


def runtime_tree_sha256(root: Path) -> str:
    return _sha256_bytes(_canonical(runtime_inventory(root)))


def patch_pi0_config_source(source: str) -> str:
    """Patch exactly one reviewed predicate in the canonical Pi0Config source."""

    if source.count(_OLD_PREDICATE) != 1:
        if _NEW_PREDICATE in source:
            raise ValueError("Pi0Config already contains the GDN+JEPA exception; refusing a stacked overlay")
        raise ValueError("unreviewed Pi0Config mutual-exclusion predicate; overlay not applied")
    patched = source.replace(_OLD_PREDICATE, _NEW_PREDICATE, 1)
    before = source.splitlines(keepends=True)
    after = patched.splitlines(keepends=True)
    differences = [(old, new) for old, new in itertools.zip_longest(before, after) if old != new]
    if differences != [(_OLD_PREDICATE, _NEW_PREDICATE)]:
        raise AssertionError(f"GDN+JEPA overlay changed more than the reviewed predicate: {differences!r}")
    compile(patched, str(PI0_CONFIG_RELATIVE), "exec")
    return patched


def _manifest_sha256(value: dict[str, Any]) -> str:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical(clean))


def _expected_manifest() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "kind": OVERLAY_KIND,
        "overlay_version": OVERLAY_VERSION,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "base_runtime_tree_sha256": BASE_RUNTIME_TREE_SHA256,
        "overlay_runtime_tree_sha256": PATCHED_RUNTIME_TREE_SHA256,
        "copied_roots": [path.as_posix() for path in COPIED_ROOTS],
        "changed_files": [
            {
                "path": PI0_CONFIG_RELATIVE.as_posix(),
                "before_sha256": BASE_PI0_CONFIG_SHA256,
                "after_sha256": PATCHED_PI0_CONFIG_SHA256,
            }
        ],
        "unchanged_file_count": BASE_RUNTIME_FILE_COUNT - 1,
        "allowed_workspace_pair": sorted(ALLOWED_PAIR),
        "scientific_delta": (
            "Pi0Config validation only: sanction exactly the tanh workspace read plus the train-only jepa_aux_target"
        ),
        "model_math_changed": False,
    }
    value["manifest_sha256"] = _manifest_sha256(value)
    return value


def _validate_source_runtime(source_repo: Path) -> None:
    source_config = source_repo / PI0_CONFIG_RELATIVE
    if not source_config.is_file():
        raise ValueError(f"canonical OpenPI runtime is missing {PI0_CONFIG_RELATIVE}")
    source_config_sha = _sha256_file(source_config)
    if source_config_sha != BASE_PI0_CONFIG_SHA256:
        raise ValueError(f"canonical ed923 Pi0Config drifted: {source_config_sha} != {BASE_PI0_CONFIG_SHA256}")
    inventory = runtime_inventory(source_repo)
    if len(inventory) != BASE_RUNTIME_FILE_COUNT:
        raise ValueError(f"canonical ed923 runtime file count drifted: {len(inventory)} != {BASE_RUNTIME_FILE_COUNT}")
    source_tree_sha = _sha256_bytes(_canonical(inventory))
    if source_tree_sha != BASE_RUNTIME_TREE_SHA256:
        raise ValueError(f"canonical ed923 runtime drifted: {source_tree_sha} != {BASE_RUNTIME_TREE_SHA256}")


def validate_staged_overlay(output_repo: Path, *, source_repo: Path | None = None) -> dict[str, Any]:
    """Fully verify an existing overlay before reuse or activation."""

    output_repo = output_repo.resolve()
    manifest_path = output_repo / OVERLAY_MANIFEST
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"GDN+JEPA overlay manifest is absent or malformed: {output_repo}") from error
    expected_keys = set(_expected_manifest())
    if set(value) != expected_keys:
        raise ValueError(f"GDN+JEPA overlay manifest keys drifted: {sorted(set(value) ^ expected_keys)}")
    if value.get("manifest_sha256") != _manifest_sha256(value):
        raise ValueError("GDN+JEPA overlay manifest seal drifted")
    if source_repo is not None:
        _validate_source_runtime(source_repo.resolve())
    expected = _expected_manifest()
    if value != expected:
        drift = {key: (value.get(key), expected[key]) for key in expected if value.get(key) != expected[key]}
        raise ValueError(f"GDN+JEPA overlay provenance drifted: {drift}")
    actual_tree = runtime_tree_sha256(output_repo)
    if actual_tree != PATCHED_RUNTIME_TREE_SHA256:
        raise ValueError(f"GDN+JEPA overlay runtime drifted: {actual_tree} != {PATCHED_RUNTIME_TREE_SHA256}")
    config_path = output_repo / PI0_CONFIG_RELATIVE
    if _sha256_file(config_path) != PATCHED_PI0_CONFIG_SHA256:
        raise ValueError("GDN+JEPA overlaid Pi0Config bytes drifted")
    return value


def stage_openpi_overlay(
    source_repo: Path,
    output_repo: Path,
    *,
    source_archive_sha256: str,
) -> dict[str, Any]:
    """Stage the exact ed923 runtime plus the one-line config-validation overlay."""

    if source_archive_sha256 != BASE_ARCHIVE_SHA256:
        raise ValueError(
            f"GDN+JEPA overlay requires canonical ed923 archive {BASE_ARCHIVE_SHA256}, got {source_archive_sha256}"
        )
    source_repo = source_repo.resolve()
    output_repo = output_repo.resolve()
    if output_repo == source_repo or output_repo.is_relative_to(source_repo):
        raise ValueError("overlay output must be outside the canonical OpenPI source tree")
    _validate_source_runtime(source_repo)
    if output_repo.exists():
        return validate_staged_overlay(output_repo, source_repo=source_repo)

    temporary = output_repo.parent / f".{output_repo.name}.incomplete-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        ignore = shutil.ignore_patterns("__pycache__", "*.py[co]")
        for copied_root in COPIED_ROOTS:
            destination = temporary / copied_root
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_repo / copied_root, destination, symlinks=False, ignore=ignore)
        target_config = temporary / PI0_CONFIG_RELATIVE
        original = target_config.read_text(encoding="utf-8")
        patched = patch_pi0_config_source(original)
        if _sha256_bytes(patched.encode()) != PATCHED_PI0_CONFIG_SHA256:
            raise ValueError("reviewed GDN+JEPA Pi0Config patch digest drifted")
        target_config.write_text(patched, encoding="utf-8")
        actual_tree = runtime_tree_sha256(temporary)
        if actual_tree != PATCHED_RUNTIME_TREE_SHA256:
            raise ValueError(
                f"staged GDN+JEPA runtime receipt drifted: {actual_tree} != {PATCHED_RUNTIME_TREE_SHA256}"
            )
        manifest = _expected_manifest()
        (temporary / OVERLAY_MANIFEST).write_bytes(_canonical(manifest))
        try:
            os.rename(temporary, output_repo)
        except FileExistsError:
            shutil.rmtree(temporary)
            return validate_staged_overlay(output_repo, source_repo=source_repo)
        return validate_staged_overlay(output_repo, source_repo=source_repo)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_loaded_overlay(overlay_repo: Path) -> dict[str, Any]:
    """Fail unless this process loaded the verified overlay, then exercise the exact pair rule."""

    overlay_repo = overlay_repo.resolve()
    manifest = validate_staged_overlay(overlay_repo)
    config_module = importlib.import_module("openpi.models.pi0_config")
    pi0_module = importlib.import_module("openpi.models.pi0")
    expected_config = (overlay_repo / PI0_CONFIG_RELATIVE).resolve()
    expected_pi0 = (overlay_repo / "src/openpi/models/pi0.py").resolve()
    actual_config = Path(config_module.__file__).resolve()
    actual_pi0 = Path(pi0_module.__file__).resolve()
    if actual_config != expected_config or actual_pi0 != expected_pi0:
        raise RuntimeError(
            "GDN+JEPA overlay was staged but not loaded first on PYTHONPATH: "
            f"pi0_config={actual_config} pi0={actual_pi0} expected_root={overlay_repo}"
        )
    if _sha256_file(actual_config) != PATCHED_PI0_CONFIG_SHA256:
        raise RuntimeError("loaded GDN+JEPA Pi0Config digest drifted")

    config_type = config_module.Pi0Config
    mode_flags = {
        "legacy_token_injection": "wsm",
        "legacy_cfg": "wsm_cfg",
        "current_cfg": "wsm_cfg2",
        "tanh": "wsm_tanh",
        "jepa_aux_target": "wsm_jepa",
    }
    for count in range(2, len(mode_flags) + 1):
        for enabled in itertools.combinations(mode_flags, count):
            kwargs = {mode_flags[name]: True for name in enabled}
            if frozenset(enabled) == ALLOWED_PAIR:
                config_type(pi05=True, **kwargs)
            else:
                try:
                    config_type(pi05=True, **kwargs)
                except ValueError as error:
                    if "mutually exclusive" not in str(error):
                        raise RuntimeError(f"unexpected rejection for workspace modes {enabled}: {error}") from error
                else:
                    raise RuntimeError(f"overlay incorrectly sanctioned workspace modes {enabled}")
    return {
        "overlay_manifest_sha256": manifest["manifest_sha256"],
        "loaded_pi0_config": str(actual_config),
        "loaded_pi0": str(actual_pi0),
        "allowed_workspace_pair": sorted(ALLOWED_PAIR),
        "model_math_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--source-repo", type=Path, required=True)
    stage.add_argument("--output-repo", type=Path, required=True)
    stage.add_argument("--source-archive-sha256", required=True)
    validate = commands.add_parser("validate-loaded")
    validate.add_argument("--overlay-repo", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage_openpi_overlay(
            args.source_repo,
            args.output_repo,
            source_archive_sha256=args.source_archive_sha256,
        )
    else:
        result = validate_loaded_overlay(args.overlay_repo)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
