#!/bin/bash
# Fan the ReMemBench 256px re-render across the box's 4 GPUs (WORKERS_PER_GPU procs each).
# Each shard is resumable: an episode dir holding episode.json is skipped, so a re-run after a
# crash or an instance STOP only redoes what is missing.
set -uo pipefail
VENV=${VENV:-/data/work/remembench_env}
SCRIPTS=${SCRIPTS:-/data/work/rb_scripts}
WORKLIST=${WORKLIST:-$SCRIPTS/remembench_train_worklist.json}
DATA=${DATA:-/data/remembench_data}
OUT=${OUT:-/data2/rb_render/out}
LOGS=${LOGS:-/data2/logs/render}
GPUS=${GPUS:-4}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-6}
TOTAL=$((GPUS * WORKERS_PER_GPU))
mkdir -p "$OUT" "$LOGS"

for ((i = 0; i < TOTAL; i++)); do
  gpu=$((i % GPUS))
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    nohup "$VENV/bin/python" "$SCRIPTS/render_lerobot_shard.py" \
      --worklist "$WORKLIST" --data-root "$DATA" --out "$OUT" \
      --shard "$i/$TOTAL" --camera-size 256 \
      >"$LOGS/shard_$(printf '%03d' "$i").log" 2>&1 &
done
echo "launched $TOTAL shards over $GPUS GPUs; logs in $LOGS"
wait
echo "ALL SHARDS EXITED $(date -u +%FT%TZ)"
