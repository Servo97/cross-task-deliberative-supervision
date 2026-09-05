from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from robomme_integration.training import gdn_jepa_overlay as overlay
from wsm_settings import ROBOMME_EVAL_ROOT

CANONICAL_OPENPI = ROBOMME_EVAL_ROOT / "openpi" / "ed923b2c"


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _fixture_source(root: Path) -> Path:
    source = root / "openpi-ed923-fixture"
    _write(source / "src/openpi/__init__.py", "")
    _write(source / "src/openpi/models/__init__.py", "")
    _write(
        source / overlay.PI0_CONFIG_RELATIVE,
        """class Pi0Config:
    def __init__(
        self, *, pi05=False, wsm=False, wsm_cfg=False, wsm_cfg2=False,
        wsm_tanh=False, wsm_jepa=False,
    ):
        modes = {
            "legacy_token_injection": wsm,
            "legacy_cfg": wsm_cfg,
            "current_cfg": wsm_cfg2,
            "tanh": wsm_tanh,
            "jepa_aux_target": wsm_jepa,
        }
        enabled = [name for name, value in modes.items() if value]
        if len(enabled) > 1:
            raise ValueError(f"workspace policy interfaces are mutually exclusive, got {enabled}")
        if wsm_jepa and not pi05:
            raise ValueError("wsm_jepa requires pi05=True")
""",
    )
    _write(source / "src/openpi/models/pi0.py", "MODEL_MATH_SENTINEL = 'unchanged'\n")
    _write(source / "scripts/train.py", "# exact training entry\n")
    return source


def _bind_fixture_receipts(monkeypatch: pytest.MonkeyPatch, source: Path) -> tuple[str, str]:
    base_config = source / overlay.PI0_CONFIG_RELATIVE
    original = base_config.read_text(encoding="utf-8")
    patched = overlay.patch_pi0_config_source(original)
    monkeypatch.setattr(overlay, "BASE_PI0_CONFIG_SHA256", hashlib.sha256(original.encode()).hexdigest())
    monkeypatch.setattr(overlay, "PATCHED_PI0_CONFIG_SHA256", hashlib.sha256(patched.encode()).hexdigest())
    entries = overlay.runtime_inventory(source)
    monkeypatch.setattr(overlay, "BASE_RUNTIME_FILE_COUNT", len(entries))
    monkeypatch.setattr(overlay, "BASE_RUNTIME_TREE_SHA256", overlay.runtime_tree_sha256(source))
    for entry in entries:
        if entry["path"] == overlay.PI0_CONFIG_RELATIVE.as_posix():
            entry["sha256"] = hashlib.sha256(patched.encode()).hexdigest()
            entry["size_bytes"] = len(patched.encode())
    patched_tree = hashlib.sha256(overlay._canonical(entries)).hexdigest()
    monkeypatch.setattr(overlay, "PATCHED_RUNTIME_TREE_SHA256", patched_tree)
    return original, patched


def test_patch_changes_exactly_the_mutual_exclusion_predicate() -> None:
    source = """def validate(enabled):
        if len(enabled) > 1:
            raise ValueError(f"workspace policy interfaces are mutually exclusive, got {enabled}")
"""
    patched = overlay.patch_pi0_config_source(source)
    before = source.splitlines()
    after = patched.splitlines()
    changed = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
    assert changed == [
        (
            "        if len(enabled) > 1:",
            '        if len(enabled) > 1 and set(enabled) != {"tanh", "jepa_aux_target"}:',
        )
    ]
    with pytest.raises(ValueError, match="stacked overlay"):
        overlay.patch_pi0_config_source(patched)


def test_stage_is_content_recorded_reusable_and_preserves_model_math(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source(tmp_path)
    original, patched = _bind_fixture_receipts(monkeypatch, source)
    destination = tmp_path / "combo-overlay"

    manifest = overlay.stage_openpi_overlay(
        source,
        destination,
        source_archive_sha256=overlay.BASE_ARCHIVE_SHA256,
    )
    assert manifest == overlay.validate_staged_overlay(destination, source_repo=source)
    assert manifest["model_math_changed"] is False
    assert manifest["changed_files"] == [
        {
            "path": overlay.PI0_CONFIG_RELATIVE.as_posix(),
            "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        }
    ]
    assert (destination / overlay.PI0_CONFIG_RELATIVE).read_text(encoding="utf-8") == patched
    assert (destination / "src/openpi/models/pi0.py").read_bytes() == (
        source / "src/openpi/models/pi0.py"
    ).read_bytes()
    assert (destination / "scripts/train.py").read_bytes() == (source / "scripts/train.py").read_bytes()
    # Same-node serial campaigns may encounter the already-complete overlay; verified reuse is exact.
    assert (
        overlay.stage_openpi_overlay(
            source,
            destination,
            source_archive_sha256=overlay.BASE_ARCHIVE_SHA256,
        )
        == manifest
    )

    (destination / "src/openpi/models/pi0.py").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime drifted"):
        overlay.validate_staged_overlay(destination)


def test_stage_fails_closed_on_wrong_archive_source_or_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source(tmp_path)
    _bind_fixture_receipts(monkeypatch, source)
    with pytest.raises(ValueError, match="requires canonical ed923"):
        overlay.stage_openpi_overlay(source, tmp_path / "wrong", source_archive_sha256="0" * 64)
    with pytest.raises(ValueError, match="outside the canonical"):
        overlay.stage_openpi_overlay(
            source,
            source / "nested-overlay",
            source_archive_sha256=overlay.BASE_ARCHIVE_SHA256,
        )
    (source / "src/openpi/models/pi0.py").write_text("source drift", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime drifted"):
        overlay.stage_openpi_overlay(
            source,
            tmp_path / "source-drift",
            source_archive_sha256=overlay.BASE_ARCHIVE_SHA256,
        )


def test_loaded_validation_proves_exact_pair_and_overlay_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source(tmp_path)
    _bind_fixture_receipts(monkeypatch, source)
    destination = tmp_path / "combo-overlay"
    overlay.stage_openpi_overlay(
        source,
        destination,
        source_archive_sha256=overlay.BASE_ARCHIVE_SHA256,
    )

    saved = {name: value for name, value in sys.modules.items() if name == "openpi" or name.startswith("openpi.")}
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(destination / "src"))
    try:
        receipt = overlay.validate_loaded_overlay(destination)
        assert receipt["allowed_workspace_pair"] == ["jepa_aux_target", "tanh"]
        assert receipt["model_math_changed"] is False
        assert Path(receipt["loaded_pi0_config"]) == (destination / overlay.PI0_CONFIG_RELATIVE).resolve()
    finally:
        sys.path.remove(str(destination / "src"))
        for name in tuple(sys.modules):
            if name == "openpi" or name.startswith("openpi."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.mark.skipif(not CANONICAL_OPENPI.is_dir(), reason="canonical ed923 extraction is not local")
def test_static_receipts_match_the_canonical_ed923_runtime() -> None:
    config = CANONICAL_OPENPI / overlay.PI0_CONFIG_RELATIVE
    original = config.read_bytes()
    assert hashlib.sha256(original).hexdigest() == overlay.BASE_PI0_CONFIG_SHA256
    assert overlay.runtime_tree_sha256(CANONICAL_OPENPI) == overlay.BASE_RUNTIME_TREE_SHA256
    patched = overlay.patch_pi0_config_source(original.decode("utf-8")).encode()
    assert hashlib.sha256(patched).hexdigest() == overlay.PATCHED_PI0_CONFIG_SHA256
    entries = overlay.runtime_inventory(CANONICAL_OPENPI)
    for entry in entries:
        if entry["path"] == overlay.PI0_CONFIG_RELATIVE.as_posix():
            entry.update(sha256=hashlib.sha256(patched).hexdigest(), size_bytes=len(patched))
    assert hashlib.sha256(overlay._canonical(entries)).hexdigest() == overlay.PATCHED_RUNTIME_TREE_SHA256
