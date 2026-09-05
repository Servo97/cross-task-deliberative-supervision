#!/usr/bin/env bash
# Historical 60k/p5/priority-600 RoboMME entry.  Kept only as a fail-closed tombstone so an old
# launch command cannot silently start the superseded all-task, final-only recipe.
set -euo pipefail
echo "FATAL: fleet_entry.sh is retired; use launch.py -> gpu_train_entry.sh (p5e, 20k, resumable)." >&2
exit 64
