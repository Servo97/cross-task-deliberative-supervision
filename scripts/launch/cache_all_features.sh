#!/usr/bin/env bash
# Cache frozen groot_on backbone features for ALL 50 target tasks across the 2 local 5090s.
# Reads the seed-0 labels (<frames>/<task>/ep*_frames.npz) -> per-demo patch_tokens cache. Resolves
# each task's lerobot dir robocasa-FREE via the glob bypass (WSM_SOUP_FROM_DIRS). Runs in the gr00t
# venv (Blackwell guard inside cache_features blocks flash_attn -> sdpa). Tasks split across the 2
# GPUs, run sequentially within each GPU.
#   screen -dmS wsm_cache bash scripts/launch/cache_all_features.sh
set -uo pipefail
REPO=/home/sarveshp/Research/TRI/wsmv2
FRAMES="$HOME/Research/TRI/wsm_data/wsm_vlm_rc"
CACHE="$HOME/Research/TRI/wsm_data/wsm_cache"
CKPT="$HOME/Research/TRI/wsm_data/wsm_ckpts/groot_on/checkpoint-150000"
DATA=/home/sarveshp/Research/robocasa/datasets
PY="$HOME/Research/Isaac-GR00T/.venv/bin/python"
LOGD="$CACHE/_feat_logs"; mkdir -p "$LOGD"
# Never read a repo-local secret file. The caller (or on-node Secrets Manager wrapper) must inject it.
: "${HF_TOKEN:?HF_TOKEN must already be present in the environment; use the guarded Secrets Manager path}"
export HF_TOKEN
export HF_HOME="$HOME/.cache/huggingface"

mapfile -t TASKS < <(for d in "$FRAMES"/*/rc_report.json; do basename "$(dirname "$d")"; done | sort)
echo "[cache] ${#TASKS[@]} tasks across cuda:0,cuda:1  | ckpt=$CKPT"

run_gpu () {  # $1=physical gpu  $2..=tasks
  local gpu=$1; shift
  for t in "$@"; do
    if [[ -f "$CACHE/$t/.done" ]]; then echo "[cache] gpu$gpu $t already done — skip"; continue; fi
    echo "[cache] gpu$gpu -> $t  $(date '+%H:%M:%S')"
    if WSM_SOUP_FROM_DIRS="$DATA" PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="$gpu" \
        "$PY" "$REPO/workspace_models/features/cache_features.py" \
        --task "$t" --frames-dir "$FRAMES" --ckpt "$CKPT" --cache-root "$CACHE" \
        --device cuda:0 --batch-size 16 > "$LOGD/$t.log" 2>&1; then
      touch "$CACHE/$t/.done"; echo "[cache] gpu$gpu $t OK  ($(ls -1d "$CACHE/$t"/demo_* 2>/dev/null | wc -l) demos)"
    else
      echo "[cache] gpu$gpu $t FAIL — see $LOGD/$t.log"
    fi
  done
}

A=(); B=(); i=0
for t in "${TASKS[@]}"; do if (( i % 2 == 0 )); then A+=("$t"); else B+=("$t"); fi; ((i++)); done
echo "[cache] gpu0: ${#A[@]} tasks | gpu1: ${#B[@]} tasks"
run_gpu 0 "${A[@]}" &
run_gpu 1 "${B[@]}" &
wait
echo "[cache] ALL DONE $(date)  | cached tasks: $(ls -1 "$CACHE"/*/.done 2>/dev/null | wc -l)/${#TASKS[@]}"
