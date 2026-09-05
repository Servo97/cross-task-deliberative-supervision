#!/usr/bin/env bash
# H14 Stage-E — run the A4 encoder-cell funnel LOCALLY (coordinator ruling 2026-08-28: the p5
# submission is denied by SCP p-ahpdy5vv, so the funnel runs here while the packaged job waits).
#
# Two workers pull from ONE work queue via atomic `mkdir` claims, so a cell is never run twice and
# a worker that dies mid-cell leaves a visible claim rather than a silent gap. GPU0 starts now;
# GPU1 is claimed only once it is genuinely idle (<1 GB used) because another agent's vLLM server
# owns it until it releases.
#
#   scripts/deliberation/run_stage_e_funnel.sh <labels_dir> <out_dir> <log_dir> [steps] [batch]
set -u
REPO=/home/sarveshp/Research/TRI/wsmv2
PY=/home/sarveshp/miniconda3/envs/ogpo2/bin/python
LABELS="${1:?labels dir}"
OUT="${2:?out dir}"
LOGS="${3:?log dir}"
STEPS="${4:-12000}"
BATCH="${5:-64}"
TAP="$HOME/Research/TRI/wsm_data/wsm_pooled/pi_100k"

# Coordinator's execution order. ctrl-1D is replaced by ctrl-0-seed2: with only the RoboCasa tap
# loadable, ctrl-1D is a bit-identical rerun of E1 and contributes no attribution, whereas a paired
# second seed on ctrl-0 makes the primary E1 - ctrl-0 contrast readable against run-to-run spread.
CELLS=(E1 ctrl-0 ctrl-E ctrl-T ctrl-S ctrl-0-seed2 E1-analog05 E1-seed2)

mkdir -p "$LOGS" "$OUT" "$LOGS/claims"

run_cell() {
  local cell="$1" gpu="$2"
  echo "[$(date -Is)] gpu$gpu START $cell" >>"$LOGS/funnel.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$PY" -u \
    "$REPO/workspace_models/train/train_wsm_base/train_stage_e.py" \
    --labels "$LABELS" --tap "robocasa=$TAP" \
    --cell "$cell" --out "$OUT" \
    --steps "$STEPS" --batch-episodes "$BATCH" --min-edges-per-batch 48 \
    --warmup 1000 --eval-every 1000 \
    --export-omega "$OUT/omega/$cell" \
    >"$LOGS/$cell.log" 2>&1
  echo "[$(date -Is)] gpu$gpu END $cell exit=$?" >>"$LOGS/funnel.log"
}

worker() {
  local gpu="$1" wait_for_free="$2"
  if [ "$wait_for_free" = "1" ]; then
    # Poll until the other agent's process releases the GPU. Never place anything on a busy GPU.
    while true; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
      [ "${used:-99999}" -lt 1024 ] && break
      sleep 300
    done
    echo "[$(date -Is)] gpu$gpu released (${used} MiB) — joining the queue" >>"$LOGS/funnel.log"
  fi
  for cell in "${CELLS[@]}"; do
    if mkdir "$LOGS/claims/$cell" 2>/dev/null; then
      echo "$gpu" >"$LOGS/claims/$cell/gpu"
      run_cell "$cell" "$gpu"
    fi
  done
  echo "[$(date -Is)] gpu$gpu drained" >>"$LOGS/funnel.log"
}

worker 0 0 &
W0=$!
worker 1 1 &
W1=$!
wait $W0 $W1
echo "[$(date -Is)] FUNNEL COMPLETE" >>"$LOGS/funnel.log"
