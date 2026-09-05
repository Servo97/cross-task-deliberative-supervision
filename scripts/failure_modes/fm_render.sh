#!/bin/bash
# Server-free video pass. Replays recorded state trajectories and encodes the composited
# multi-view videos. Runs after (or alongside) the rollout cells — it needs no policy server,
# so it can occupy GPUs a serve cell is not using.
#
#   BENCH=remembench CKPT_LABEL=expert bash fm_render.sh
#   BENCH=remembench CKPT_LABEL=dnw8 CRF=23 bash fm_render.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR=/data/tmp

WORK=/data/work
FM=/data2/failure_modes
WSMV2="$WORK/wsmv2"

: "${BENCH:?set BENCH}"
: "${CKPT_LABEL:?set CKPT_LABEL}"
NUM_GPUS="${NUM_GPUS:-4}"
STRIDE="${STRIDE:-2}"
# CRF 23 everywhere, rollouts and demonstrations alike. The original plan split them (26 for
# the 540 rollouts, 23 for the 120 demos) on the assumption that the ~0.6 MB/s VPN made the
# return leg the binding constraint. Measured 2026-08-11 the link runs at ~18.7 MB/s and
# files are 2-5 MB, so the whole set moves in minutes either way and there is nothing to buy
# by degrading the rollouts -- which are the videos actually being eyeballed.
CRF="${CRF:-23}"
LIMIT="${LIMIT:-}"
ROLLOUT_IDX="${ROLLOUT_IDX:-0}"
TASKS="${TASKS:-}"

if [[ "$BENCH" == "remembench" ]]; then
  PY="$WORK/remembench_env/bin/python"
  ALL_TASKS="MemFruitInSinkRightFar MemHeatPot MemWashAndReturnLeft"
else
  PY="$WORK/simenv/bin/python"
  ALL_TASKS="KettleBoiling ScrubCuttingBoard SearingMeat"
fi
[[ -n "$TASKS" ]] && ALL_TASKS="${TASKS//,/ }"

MANIFEST="$FM/manifests/fm_${BENCH}_manifest.json"
LOGS="$FM/logs/render_${BENCH}_${CKPT_LABEL}"
mkdir -p "$LOGS" "$FM/videos"
LIMIT_ARG=(); [[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

FAILS=0
for TASK in $ALL_TASKS; do
  declare -a PIDS=()
  for ((w = 0; w < NUM_GPUS; w++)); do
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$w CUDA_VISIBLE_DEVICES=$w \
    PYTHONPATH="$WSMV2" PYTHONUNBUFFERED=1 \
    "$PY" "$WSMV2/scripts/failure_modes/fm_render.py" \
      --manifest "$MANIFEST" --bench "$BENCH" --task "$TASK" \
      --out-root "$FM" --video-root "$FM/videos" --ckpt-label "$CKPT_LABEL" \
      --shard-idx "$w" --num-shards "$NUM_GPUS" --stride "$STRIDE" --crf "$CRF" \
      --rollout-idx "$ROLLOUT_IDX" \
      "${LIMIT_ARG[@]}" >"$LOGS/${TASK}_w$w.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS + 1)); done
  echo "[render] $TASK done"
done
echo "RENDER DONE $BENCH/$CKPT_LABEL fails=$FAILS  size=$(du -sh "$FM/videos" 2>/dev/null | cut -f1)"
[[ "$FAILS" -eq 0 ]] || exit 23
