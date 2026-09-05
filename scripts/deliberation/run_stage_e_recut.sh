#!/usr/bin/env bash
# H14 Stage-E funnel RE-CUT (coordinator, 2026-08-28, after the A9 accuracy results).
#
# CONTRAST precision adjudicated at 0.172 and planted-probe recovery at 0.533, so the contested
# CONTRAST edges get their own discriminating cell — `E1-noCONTRAST`, highest priority after the
# cells already in flight. `ctrl-0-seed2` is dropped for its slot (it was the lowest-value cell:
# paired spread, not attribution). The first funnel's remaining cells were pre-claimed so its two
# workers drain out cleanly rather than racing this queue.
set -u
REPO=/home/sarveshp/Research/TRI/wsmv2
PY=/home/sarveshp/miniconda3/envs/ogpo2/bin/python
LABELS="${1:?}"; OUT="${2:?}"; LOGS="${3:?}"; STEPS="${4:-12000}"; BATCH="${5:-64}"
TAP="$HOME/Research/TRI/wsm_data/wsm_pooled/pi_100k"
CELLS=(E1-noCONTRAST ctrl-S E1-analog05 E1-seed2)
mkdir -p "$LOGS" "$OUT" "$LOGS/claims2"

worker() {
  local gpu="$1"
  while true; do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    [ "${used:-99999}" -lt 1024 ] && break
    sleep 60
  done
  for cell in "${CELLS[@]}"; do
    mkdir "$LOGS/claims2/$cell" 2>/dev/null || continue
    echo "[$(date -Is)] gpu$gpu START $cell" >>"$LOGS/funnel.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$PY" -u \
      "$REPO/workspace_models/train/train_wsm_base/train_stage_e.py" \
      --labels "$LABELS" --tap "robocasa=$TAP" --cell "$cell" --out "$OUT" \
      --steps "$STEPS" --batch-episodes "$BATCH" --min-edges-per-batch 48 \
      --warmup 1000 --eval-every 1000 --export-omega "$OUT/omega/$cell" \
      >"$LOGS/$cell.log" 2>&1
    echo "[$(date -Is)] gpu$gpu END $cell exit=$?" >>"$LOGS/funnel.log"
  done
  echo "[$(date -Is)] gpu$gpu recut-drained" >>"$LOGS/funnel.log"
}
worker 0 & W0=$!
worker 1 & W1=$!
wait $W0 $W1
echo "[$(date -Is)] RECUT COMPLETE" >>"$LOGS/funnel.log"
