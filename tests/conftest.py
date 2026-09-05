"""Make repo-local test helpers importable as top-level modules.

An unrelated ``tests`` package exists in some site-packages, which shadows ``from tests import ...``.
Putting this directory on ``sys.path`` lets shared helpers (e.g. ``stage_s_fixtures``) import as
plain top-level modules across the sm_launch / openpi-jax-latest / pytorch test environments.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
