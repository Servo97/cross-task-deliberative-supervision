from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_entry_reparents_flat_sagemaker_source_for_package_import(tmp_path: Path) -> None:
    entry = Path(__file__).resolve().parents[1] / "gpu_framesamp_am_r1_entry.sh"
    text = entry.read_text(encoding="utf-8")
    assert 'ln -s "$CODE" "$PKG_PARENT/robomme_integration"' in text
    assert 'PYTHONPATH="$PKG_PARENT" python3 -B - "$MANIFEST"' in text
    assert "$OVERLAY/src:$PKG_PARENT:$UPSTREAM/src" in text
    assert 'UV_PROJECT_ENVIRONMENT="$UPSTREAM/.venv"' in text
    assert '"0.5.3:0.11.13"' in text
    assert "OPENPI_FORK_S3" not in text

    flat = tmp_path / "code"
    (flat / "eval").mkdir(parents=True)
    (flat / "__init__.py").write_text("", encoding="utf-8")
    (flat / "eval" / "__init__.py").write_text("", encoding="utf-8")
    (flat / "eval" / "sentinel.py").write_text("VALUE = 17\n", encoding="utf-8")
    parent = tmp_path / "source-parent"
    parent.mkdir()
    os.symlink(flat, parent / "robomme_integration")
    environment = {**os.environ, "PYTHONPATH": str(parent), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from robomme_integration.eval.sentinel import VALUE; assert VALUE == 17",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
