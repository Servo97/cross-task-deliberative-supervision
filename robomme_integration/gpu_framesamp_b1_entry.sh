#!/usr/bin/env bash
# Isolated FS-B1 8-H100 transition canary. It publishes no score.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_TRACEBACK_FILTERING=off
exec python3 -B /opt/ml/code/eval/framesamp_b1_entry.py
