from __future__ import annotations

import subprocess
from pathlib import Path


def test_historical_fleet_entry_fails_closed():
    entry = Path(__file__).parents[1] / "fleet_entry.sh"
    result = subprocess.run([str(entry)], text=True, capture_output=True, check=False)
    assert result.returncode == 64
    assert "retired" in result.stderr
    assert "gpu_train_entry.sh" in result.stderr
