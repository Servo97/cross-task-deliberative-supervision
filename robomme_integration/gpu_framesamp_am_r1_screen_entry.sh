#!/usr/bin/env bash
# Isolated FS-R1 12-cell screen entry. All scientific work is in the sealed Python driver.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_TRACEBACK_FILTERING=off
exec python3 -B /opt/ml/code/eval/framesamp_am_r1_screen_entry.py
