#!/usr/bin/env bash
# Cache frozen pi05 backbone features for ALL 50 target tasks across the 2 local 5090s (openpi-jax-latest).
# Each GPU runs pi_cache_features ONCE over its half of the tasks -> the 2B model loads once per GPU.
# Reads the SAME seed-0 labels (ep*_frames.npz) as the groot cache; per-demo .done_features makes it resumable.
#   screen -dmS wsm_cache_pi bash scripts/launch/cache_all_features_pi.sh
set -uo pipefail
REPO=/home/sarveshp/Research/TRI/wsmv2
FRAMES="$HOME/Research/TRI/wsm_data/wsm_vlm_rc"
CACHE="$HOME/Research/TRI/wsm_data/wsm_cache_pi"
CKPT="$HOME/Research/TRI/wsm_data/wsm_ckpts/pi05_on/149999"
DATA=/home/sarveshp/Research/robocasa/datasets
PY="$HOME/Research/envs/openpi-jax-latest/bin/python"
LOGD="$CACHE/_logs"; mkdir -p "$LOGD"

mapfile -t TASKS < <(for d in "$FRAMES"/*/rc_report.json; do basename "$(dirname "$d")"; done | sort)
A=(); B=(); i=0
for t in "${TASKS[@]}"; do if (( i % 2 == 0 )); then A+=("$t"); else B+=("$t"); fi; ((i++)); done
A_CSV=$(IFS=,; echo "${A[*]}"); B_CSV=$(IFS=,; echo "${B[*]}")
echo "[pi-cache] ${#TASKS[@]} tasks | gpu0: ${#A[@]} | gpu1: ${#B[@]} | ckpt=$CKPT"

run_gpu () {  # $1=physical gpu  $2=comma task list
  CUDA_VISIBLE_DEVICES="$1" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  DATASET_BASE_PATH="$DATA" WSM_SOUP_FROM_DIRS="$DATA" PYTHONPATH="$REPO" \
    "$PY" -m workspace_models.features.pi_cache_features \
    --tasks "$2" --frames-dir "$FRAMES" --ckpt "$CKPT" --cache-root "$CACHE" --batch-size 16
}
run_gpu 0 "$A_CSV" > "$LOGD/gpu0.log" 2>&1 &
run_gpu 1 "$B_CSV" > "$LOGD/gpu1.log" 2>&1 &
wait
echo "[pi-cache] ALL DONE $(date) | cached demos: $(ls -1d "$CACHE"/*/demo_* 2>/dev/null | wc -l) (expect ~7422)"
