#!/usr/bin/env bash
# Orchestrate RoboCasa365 eval on the TARGET split (50-task protocol) for any regime's checkpoint.
# Fans out one rollout CLIENT per GPU (the eval_${BACKBONE}.py driver) against a policy server at
# BASE_PORT+w, snake-sharded by horizon, then aggregates per-task stats.json -> results.json.
#   Drivers: vla_training/eval/eval_{groot_17,pi_05}.py   Recipe: scripts/configs/eval/<x>.yaml
#   Metric: task-weighted avg success % over atomic_seen/composite_seen/composite_unseen
#           (composite_unseen = headline). Governs: 01_robocasa_protocol_and_recipes.md.
#
#   scripts/eval/eval.sh --backbone groot_17 [--config <eval-yaml>] [--step N] [--dry-run]
#
# NOTE: the per-GPU POLICY SERVERS + the torch-free sim venv setup are the SageMaker eval ENTRY's
# job (Phase-3 infra; reference: ported_raw/reference_code/robocasa_eval_entry.sh — pi05 serves
# openpi websocket on 8000+w, groot serves run_gr00t_server.py zmq on 5600+w). This script assumes
# the servers are already up at $SERVER_HOST:BASE_PORT+w (set them up via that entry, or run a
# single worker locally against one server).
set -euo pipefail

BACKBONE=""; CONFIG=""; DRY=""; STEP="${STEP:-0}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() { echo "usage: $0 --backbone {groot_17|pi_05} [--config <eval-yaml>] [--step N] [--dry-run]"; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) BACKBONE="$2"; shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    --step)     STEP="$2";     shift 2 ;;
    --dry-run)  DRY="--dry-run"; shift ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done
[[ -n "$BACKBONE" ]] || usage

case "$BACKBONE" in
  pi_05)    BASE_PORT=8000; MODEL=pi05 ;;
  groot_17) BASE_PORT=5600; MODEL=groot ;;
  *) echo "bad --backbone: $BACKBONE"; usage ;;
esac
DRIVER="$REPO_ROOT/vla_training/eval/eval_${BACKBONE}.py"
[[ -f "$DRIVER" ]] || { echo "ERROR: driver not found: $DRIVER"; exit 3; }
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Recipe knobs from the eval YAML. (Importing utils pulls robocasa, which prints import noise to
# stdout, so emit a sentinel-prefixed line and extract just that.)
CFGVALS=$(python - "$CONFIG" "$BACKBONE" <<'PY'
import sys
from utils.config_schema import load_eval_config, default_eval_config_path
cfg = load_eval_config(sys.argv[1] or default_eval_config_path(sys.argv[2]))
print("@CFG", cfg.num_workers, cfg.output_dir or f"results/{sys.argv[2]}_eval",
      cfg.num_trials, cfg.split, ",".join(cfg.task_sets))
PY
)
read -r NUM_WORKERS OUT_DIR NUM_TRIALS SPLIT TASK_SETS <<< "$(printf '%s\n' "$CFGVALS" | sed -n 's/^@CFG //p')"
[[ -n "$NUM_WORKERS" ]] || { echo "ERROR: could not read eval config"; printf '%s\n' "$CFGVALS"; exit 4; }

echo "=========================================================="
echo " eval: backbone=$BACKBONE model=$MODEL split=$SPLIT trials=$NUM_TRIALS workers=$NUM_WORKERS"
echo "   driver  : $DRIVER"
echo "   config  : ${CONFIG:-(driver default)}"
echo "   servers : $SERVER_HOST:$BASE_PORT..$((BASE_PORT+NUM_WORKERS-1))   out: $OUT_DIR"
echo "=========================================================="

# Dry-run: just preview worker 0's shard (no servers / sim needed).
if [[ -n "$DRY" ]]; then
  python "$DRIVER" ${CONFIG:+--config "$CONFIG"} --worker-idx 0 --num-workers "$NUM_WORKERS" --dry-run
  exit 0
fi

# Fan out one rollout client per GPU (EGL render pinned to the same GPU). Servers must be up.
declare -a PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$w CUDA_VISIBLE_DEVICES=$w \
  python "$DRIVER" ${CONFIG:+--config "$CONFIG"} \
    --worker-idx "$w" --num-workers "$NUM_WORKERS" \
    --host "$SERVER_HOST" --port $((BASE_PORT + w)) --out-dir "$OUT_DIR" &
  PIDS+=($!)
done
FAILS=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS + 1)); done
echo "[eval] runners done, $FAILS/$NUM_WORKERS exited nonzero"

# Aggregate per-task stats.json -> results.json (task-weighted avg + per-split breakdown).
python "$REPO_ROOT/vla_training/eval/aggregate_eval.py" \
  --results-dir "$OUT_DIR" --model "$MODEL" --step "$STEP" \
  --split "$SPLIT" --num-trials "$NUM_TRIALS" --task-sets "$TASK_SETS"
