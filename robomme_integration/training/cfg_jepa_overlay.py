"""Content-recorded OpenPI config overlay for v4 CFG + JEPA-VISReg.

The canonical runtime already implements both losses but rejects every pair of workspace modes.
This overlay changes only that validation predicate and sanctions exactly current-only CFG plus the
train-only JEPA target.  Model, loss, optimizer, and inference math are otherwise byte-identical.
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

from .gdn_jepa_overlay import (
    BASE_ARCHIVE_SHA256,
    BASE_PI0_CONFIG_SHA256,
    BASE_RUNTIME_FILE_COUNT,
    BASE_RUNTIME_TREE_SHA256,
    COPIED_ROOTS,
    PI0_CONFIG_RELATIVE,
    runtime_inventory,
    runtime_tree_sha256,
)

PATCHED_PI0_CONFIG_SHA256 = "c2ffe8c4577e1c546939e7695575c5d5efdcfbceb39ea6e3c46cc96f73607d90"
PATCHED_RUNTIME_TREE_SHA256 = "63095080ef946c1cb8ca4261899fac5e5587686780f97fceb7eec20d4ce44c30"
OVERLAY_VERSION = "robomme-v4-cfg-jepa-config-v1"
OVERLAY_KIND = "robomme_v4_cfg_jepa_openpi_overlay"
OVERLAY_MANIFEST = "ROBOMME_V4_CFG_JEPA_OVERLAY.json"
ALLOWED_PAIR = frozenset({"current_cfg", "jepa_aux_target"})

_OLD_PREDICATE = "        if len(enabled) > 1:\n"
_NEW_PREDICATE = '        if len(enabled) > 1 and set(enabled) != {"current_cfg", "jepa_aux_target"}:\n'


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def patch_pi0_config_source(source: str) -> str:
    if source.count(_OLD_PREDICATE) != 1:
        if _NEW_PREDICATE in source:
            raise ValueError("Pi0Config already contains the CFG+JEPA exception")
        raise ValueError("unreviewed Pi0Config mutual-exclusion predicate; overlay not applied")
    patched = source.replace(_OLD_PREDICATE, _NEW_PREDICATE, 1)
    differences = [
        pair
        for pair in itertools.zip_longest(source.splitlines(keepends=True), patched.splitlines(keepends=True))
        if pair[0] != pair[1]
    ]
    if differences != [(_OLD_PREDICATE, _NEW_PREDICATE)]:
        raise AssertionError(f"CFG+JEPA overlay changed unexpected source lines: {differences!r}")
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
            "Pi0Config validation only: sanction exactly current-only CFG plus the train-only JEPA-VISReg target"
        ),
        "model_math_changed": False,
    }
    value["manifest_sha256"] = _manifest_sha256(value)
    return value


def _validate_source(source_repo: Path) -> None:
    config = source_repo / PI0_CONFIG_RELATIVE
    if _sha256_file(config) != BASE_PI0_CONFIG_SHA256:
        raise ValueError("canonical OpenPI Pi0Config bytes drifted")
    inventory = runtime_inventory(source_repo)
    if len(inventory) != BASE_RUNTIME_FILE_COUNT:
        raise ValueError("canonical OpenPI runtime file count drifted")
    if _sha256_bytes(_canonical(inventory)) != BASE_RUNTIME_TREE_SHA256:
        raise ValueError("canonical OpenPI runtime tree drifted")


def validate_staged_overlay(output_repo: Path, *, source_repo: Path | None = None) -> dict:
    output_repo = output_repo.resolve()
    if source_repo is not None:
        _validate_source(source_repo.resolve())
    try:
        manifest = json.loads((output_repo / OVERLAY_MANIFEST).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("CFG+JEPA overlay manifest is absent or malformed") from error
    if manifest != _expected_manifest() or manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("CFG+JEPA overlay manifest/provenance drifted")
    if runtime_tree_sha256(output_repo) != PATCHED_RUNTIME_TREE_SHA256:
        raise ValueError("CFG+JEPA overlay runtime tree drifted")
    if _sha256_file(output_repo / PI0_CONFIG_RELATIVE) != PATCHED_PI0_CONFIG_SHA256:
        raise ValueError("CFG+JEPA overlaid Pi0Config bytes drifted")
    return manifest


def stage_openpi_overlay(source_repo: Path, output_repo: Path, *, source_archive_sha256: str) -> dict:
    if source_archive_sha256 != BASE_ARCHIVE_SHA256:
        raise ValueError("CFG+JEPA overlay requires the canonical ed923 OpenPI archive")
    source_repo, output_repo = source_repo.resolve(), output_repo.resolve()
    if output_repo == source_repo or output_repo.is_relative_to(source_repo):
        raise ValueError("overlay output must be outside the canonical OpenPI tree")
    _validate_source(source_repo)
    if output_repo.exists():
        return validate_staged_overlay(output_repo, source_repo=source_repo)
    temporary = output_repo.parent / f".{output_repo.name}.incomplete-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        ignore = shutil.ignore_patterns("__pycache__", "*.py[co]")
        for copied_root in COPIED_ROOTS:
            shutil.copytree(
                source_repo / copied_root,
                temporary / copied_root,
                symlinks=False,
                ignore=ignore,
            )
        config = temporary / PI0_CONFIG_RELATIVE
        config.write_text(patch_pi0_config_source(config.read_text(encoding="utf-8")), encoding="utf-8")
        if runtime_tree_sha256(temporary) != PATCHED_RUNTIME_TREE_SHA256:
            raise ValueError("staged CFG+JEPA runtime receipt drifted")
        (temporary / OVERLAY_MANIFEST).write_bytes(_canonical(_expected_manifest()))
        try:
            os.rename(temporary, output_repo)
        except FileExistsError:
            shutil.rmtree(temporary)
        return validate_staged_overlay(output_repo, source_repo=source_repo)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_loaded_overlay(overlay_repo: Path) -> dict:
    overlay_repo = overlay_repo.resolve()
    manifest = validate_staged_overlay(overlay_repo)
    config_module = importlib.import_module("openpi.models.pi0_config")
    actual = Path(config_module.__file__).resolve()
    expected = (overlay_repo / PI0_CONFIG_RELATIVE).resolve()
    if actual != expected or _sha256_file(actual) != PATCHED_PI0_CONFIG_SHA256:
        raise RuntimeError(f"staged CFG+JEPA overlay was not loaded first: {actual} != {expected}")
    config_type = config_module.Pi0Config
    config_type(pi05=True, wsm_cfg2=True, wsm_jepa=True)
    for kwargs in (
        {"wsm_tanh": True, "wsm_jepa": True},
        {"wsm_cfg2": True, "wsm_tanh": True},
        {"wsm_cfg2": True, "wsm_jepa": True, "wsm_tanh": True},
    ):
        try:
            config_type(pi05=True, **kwargs)
        except ValueError as error:
            if "mutually exclusive" not in str(error):
                raise RuntimeError(f"unexpected CFG+JEPA overlay rejection: {error}") from error
        else:
            raise RuntimeError(f"CFG+JEPA overlay sanctioned an unreviewed pair: {kwargs}")
    return manifest


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
        print(
            json.dumps(
                stage_openpi_overlay(
                    args.source_repo,
                    args.output_repo,
                    source_archive_sha256=args.source_archive_sha256,
                ),
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(validate_loaded_overlay(args.overlay_repo), sort_keys=True))


if __name__ == "__main__":
    main()
