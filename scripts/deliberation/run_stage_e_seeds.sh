#!/usr/bin/env bash
# H14 Stage-E — A14 SEED REPLICATION of the primary contrast (pre-registered; 9 cells).
#
#   arms  {E1b, ctrl-Eb, E1b-analog05}  x  seeds {20260828, 20260829, 20260830}
#
# Primary reading = paired-by-seed delta(E1b - ctrl-Eb) on the retrieval-lift Wilson lower bound
# (A1d disagreement subset). Secondary = paired delta(E1b-analog05 - E1b). The queue is ordered
# SEED-MAJOR so that an interruption still leaves whole paired triples rather than a full arm.
#
# All nine cells read ONE label artifact (v2b), whose edges_E1b.npz / segments.npz / vocab.json /
# gate_pairs.npz are byte-identical to v2 ab38d9efc0c649a3 — the gate's ground truth does not move.
#
# GPU rule (coordinator): GPU0 is ours unconditionally. GPU1 is joined ONLY for a cell whose launch
# moment finds it under 1 GB — another agent's ReMemBench tap build takes it when idle and must not
# be evicted. The check is re-run immediately before each launch and the claim is RELEASED if the
# GPU went busy between the poll and the claim.
set -u
REPO=/home/sarveshp/Research/TRI/wsmv2
PY=/home/sarveshp/miniconda3/envs/ogpo2/bin/python
LABELS="${1:?labels dir}"; OUT="${2:?run dir}"; LOGS="${3:?log dir}"; STEPS="${4:-12000}"; BATCH="${5:-64}"
TAP="$HOME/Research/TRI/wsm_data/wsm_pooled/pi_100k"
CLAIMS="$LOGS/claims_seeds"
QUEUE=(
  "E1b:20260828" "ctrl-Eb:20260828" "E1b-analog05:20260828"
  "E1b:20260829" "ctrl-Eb:20260829" "E1b-analog05:20260829"
  "E1b:20260830" "ctrl-Eb:20260830" "E1b-analog05:20260830"
)
mkdir -p "$LOGS" "$OUT" "$CLAIMS"

gpu_free() {  # 0 = free (<1 GB used)
  local used
  used=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
  [ "${used:-99999}" -lt 1024 ]
}

worker() {
  local gpu="$1" entry cell seed name
  for entry in "${QUEUE[@]}"; do
    cell="${entry%%:*}"; seed="${entry##*:}"; name="${cell}_s${seed}"
    [ -d "$CLAIMS/$name" ] && continue
    if [ "$gpu" != 0 ]; then
      # Opportunistic: wait for GPU1 to be idle, but give the entry up the moment GPU0 claims it.
      while ! gpu_free "$gpu"; do
        [ -d "$CLAIMS/$name" ] && break
        sleep 60
      done
      [ -d "$CLAIMS/$name" ] && continue
    fi
    mkdir "$CLAIMS/$name" 2>/dev/null || continue
    if [ "$gpu" != 0 ] && ! gpu_free "$gpu"; then
      rmdir "$CLAIMS/$name"; continue   # went busy between poll and claim — hand it back
    fi
    echo "[$(date -Is)] gpu$gpu START $name" >>"$LOGS/seeds.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$PY" -u \
      "$REPO/workspace_models/train/train_wsm_base/train_stage_e.py" \
      --labels "$LABELS" --tap "robocasa=$TAP" --cell "$cell" --seed "$seed" --out "$OUT" \
      --steps "$STEPS" --batch-episodes "$BATCH" --min-edges-per-batch 48 \
      --warmup 1000 --eval-every 1000 --export-omega "$OUT/omega/$name" \
      >"$LOGS/$name.log" 2>&1
    # `$(date -Is)` runs BEFORE the expansion of $? and resets it — capture the status first.
    # (Observed 2026-08-28: E1b-analog05_s20260829 OOMed at step 4150 and was logged exit=0.)
    rc=$?
    echo "[$(date -Is)] gpu$gpu END $name exit=$rc" >>"$LOGS/seeds.log"
  done
  echo "[$(date -Is)] gpu$gpu seeds-drained" >>"$LOGS/seeds.log"
}

worker 0 & W0=$!
worker 1 & W1=$!
wait $W0 $W1
echo "[$(date -Is)] SEED REPLICATION COMPLETE" >>"$LOGS/seeds.log"
