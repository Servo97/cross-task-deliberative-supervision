#!/usr/bin/env bash
# H14 Stage-E — MULTI-DOMAIN funnel (RoboCasa + ReMemBench), 12 cells (coordinator, 2026-08-29).
#
#   arms  {E1b, ctrl-0b, ctrl-1Db, ctrl-Eb}  x  seeds {20260828, 20260829, 20260830}
#
# Primary readings, paired by seed, on the retrieval-lift Wilson lower bound:
#   E1b - ctrl-0b    the deliberative term, now multi-domain
#   E1b - ctrl-1Db   domain mixing (ctrl-1Db = E1b with the corpus restricted to RoboCasa)
#   E1b - ctrl-Eb    Qwen positives vs embedding positives
#
# Both taps are loaded, so `train_stage_e` applies the PRE-REGISTERED multi-domain G1b
# recalibration: effective-rank fail line = 0.80 x that domain's raw-tap effective rank
# (RoboCasa 10.16 -> 8.13, rmb 5.90 -> 4.72); coherence and bevf floors unchanged; the collapse
# control must still trip FAIL on every domain.
#
# GPU rule: BOTH 5090s are ours for this run (coordinator). Each cell is pinned to one GPU and no
# GPU ever holds two trainers — the 2026-08-28 replication lost a cell to exactly that (two
# processes on GPU1, OOM at step 4150).
#
# Queue order is SEED-MAJOR: an interruption leaves whole paired quadruples, not a full arm.
set -u
REPO=/home/sarveshp/Research/TRI/wsmv2
PY=/home/sarveshp/miniconda3/envs/ogpo2/bin/python
LABELS="${1:?labels dir}"; OUT="${2:?run dir}"; LOGS="${3:?log dir}"; STEPS="${4:-12000}"; BATCH="${5:-64}"
RC_TAP="$HOME/Research/TRI/wsm_data/wsm_pooled/pi_100k"
RMB_TAP="$HOME/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k"
CLAIMS="$LOGS/claims_multidomain"
QUEUE=(
  "E1b:20260828" "ctrl-0b:20260828" "ctrl-1Db:20260828" "ctrl-Eb:20260828"
  "E1b:20260829" "ctrl-0b:20260829" "ctrl-1Db:20260829" "ctrl-Eb:20260829"
  "E1b:20260830" "ctrl-0b:20260830" "ctrl-1Db:20260830" "ctrl-Eb:20260830"
)
mkdir -p "$LOGS" "$OUT" "$CLAIMS"

worker() {
  local gpu="$1" entry cell seed name rc
  for entry in "${QUEUE[@]}"; do
    cell="${entry%%:*}"; seed="${entry##*:}"; name="${cell}_s${seed}"
    mkdir "$CLAIMS/$name" 2>/dev/null || continue
    echo "[$(date -Is)] gpu$gpu START $name" >>"$LOGS/multidomain.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$PY" -u \
      "$REPO/workspace_models/train/train_wsm_base/train_stage_e.py" \
      --labels "$LABELS" --tap "robocasa=$RC_TAP" --tap "remembench=$RMB_TAP" \
      --cell "$cell" --seed "$seed" --out "$OUT" \
      --steps "$STEPS" --batch-episodes "$BATCH" --min-edges-per-batch 48 \
      --warmup 1000 --eval-every 1000 --export-omega "$OUT/omega/$name" \
      >"$LOGS/md_$name.log" 2>&1
    rc=$?   # capture BEFORE any command substitution resets $?
    echo "[$(date -Is)] gpu$gpu END $name exit=$rc" >>"$LOGS/multidomain.log"
  done
  echo "[$(date -Is)] gpu$gpu multidomain-drained" >>"$LOGS/multidomain.log"
}

# $6 = space-separated GPU list. A GPU can be joined LATER with a second invocation: the claim
# directories are the only coordination, so a worker started afterwards picks up the unclaimed tail.
GPUS="${6:-0 1}"
pids=()
for g in $GPUS; do worker "$g" & pids+=($!); done
wait "${pids[@]}"
echo "[$(date -Is)] gpus[$GPUS] drained" >>"$LOGS/multidomain.log"
