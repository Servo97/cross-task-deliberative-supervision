#!/bin/bash
# Failure-mode study driver for nagababa (4x RTX PRO 6000).
#
# One invocation = one (bench, checkpoint) CELL: bring up 4 policy servers (one per GPU),
# run the closed-loop rollouts and the teacher-forcing pass for all three tasks sharded
# across those 4 GPUs, tear the servers down. Videos and metrics are separate server-free
# passes (fm_render.sh / fm_metrics.py) so a GPU is never idle behind an encoder.
#
# LABEL GATE: every cell writes cell.json naming the arm, the checkpoint tree sha256, the
# encoder sha256, the serve interface and the pinned source trees. A cell with no cell.json
# is not a result. Workspace arms serve from the content-addressed trees their own training
# manifest pins, and with the encoder their sealed eval used — a mismatch there is the
# GR00T eval-2 failure and silently invalidates everything.
#
#   BENCH=remembench CKPT_LABEL=dnw8 bash fm_run.sh
#   BENCH=remembench CKPT_LABEL=base MODE=teacher_force bash fm_run.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR=/data/tmp

WORK=/data/work
FM=/data2/failure_modes
WSMV2="$WORK/wsmv2"
RBPY="$WORK/remembench_env/bin/python"
SIMPY="$WORK/simenv/bin/python"
OPENPI_VENV="$WORK/openpi/.venv/bin/python"

: "${BENCH:?set BENCH=remembench|robocasa}"
: "${CKPT_LABEL:?set CKPT_LABEL}"
MODE="${MODE:-rollout}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_PORT="${BASE_PORT:-5950}"
REPLAN="${REPLAN:-8}"
LIMIT="${LIMIT:-}"
ROLLOUT_IDX="${ROLLOUT_IDX:-0}"
TASKS="${TASKS:-}"

MANIFEST="$FM/manifests/fm_${BENCH}_manifest.json"
LOGS="$FM/logs/${BENCH}_${CKPT_LABEL}_${MODE}${ROLLOUT_IDX:+_r$ROLLOUT_IDX}"
mkdir -p "$LOGS" "$TMPDIR"

if [[ "$BENCH" == "remembench" ]]; then
  CLIENT_PY="$RBPY"
  ALL_TASKS="MemFruitInSinkRightFar MemHeatPot MemWashAndReturnLeft"
  ENC="$WORK/wsm_artifacts/rmb/encoder.pt"
  LANG_TABLE="$WORK/wsm_artifacts/rmb/task_lang_table.npz"
  TASK_PROMPTS="$WORK/wsm_artifacts/rmb/task_prompts_remembench13.json"
else
  CLIENT_PY="$SIMPY"
  ALL_TASKS="KettleBoiling ScrubCuttingBoard SearingMeat"
  ENC="$WORK/wsm_artifacts/encoder.pt"
  LANG_TABLE="$WORK/wsm_artifacts/task_lang_table.npz"
  TASK_PROMPTS="$WORK/wsm_artifacts/task_prompt_manifest.json"
fi
[[ -n "$TASKS" ]] && ALL_TASKS="${TASKS//,/ }"

# --------------------------------------------------------------------------------------
# arm table: label -> run id, checkpoint, serve interface, pinned source trees
# --------------------------------------------------------------------------------------
SERVE=base; OPENPI_SRC="$WORK/openpi"; WSMV2_SRC=""; CONFIG_NAME=pi05_robocasa_target_ft
CONFIGS_DIR=""; RUN_ID=""; CKPT=""
case "$BENCH:$CKPT_LABEL" in
  remembench:pretrain150k)
    RUN_ID=tap_149999; CKPT="$WORK/ckpts/tap_149999"
    CONFIG_NAME=wsm_pi05_robocasa_pretrain300 ;;
  remembench:base)
    RUN_ID=s0-9e47bc75062b23e9; CKPT="$WORK/ckpts/$RUN_ID/14999" ;;
  remembench:jepa_k16)
    RUN_ID=s3-5e942af9f0718e3a; CKPT="$WORK/ckpts/$RUN_ID/14999" ;;
  remembench:dnw8)
    RUN_ID=s1-be5d198305786f3e; CKPT="$WORK/ckpts/$RUN_ID/14999"
    SERVE=workspace; OPENPI_SRC="$WORK/openpi_rmb"; WSMV2_SRC="$WORK/wsmv2_rmb" ;;
  remembench:dnw16_drop)
    RUN_ID=s1-a40b147a41885d03; CKPT="$WORK/ckpts/$RUN_ID/14999"
    SERVE=workspace; OPENPI_SRC="$WORK/openpi_cc"; WSMV2_SRC="$WORK/wsmv2_b969680c" ;;
  robocasa:pretrain150k)
    RUN_ID=tap_149999; CKPT="$WORK/ckpts/tap_149999"
    CONFIG_NAME=wsm_pi05_robocasa_pretrain300 ;;
  robocasa:base)
    RUN_ID=s0-c43f076daad4a799; CKPT="$WORK/ckpts_rc/$RUN_ID/59999" ;;
  robocasa:jepa_k1)
    RUN_ID=s3-f55c188d8e13717b; CKPT="$WORK/ckpts_rc/$RUN_ID/59999" ;;
  robocasa:dnw8)
    RUN_ID=s1-f8e6400ab0e21968; CKPT="$WORK/ckpts_rc/$RUN_ID/59999"
    SERVE=workspace; OPENPI_SRC="$WORK/fm_src/openpi_ed923b2c"
    WSMV2_SRC="$WORK/fm_src/wsmv2_c8df8e88"
    CONFIG_NAME=pi05_robocasa_workspace_stage_s
    CONFIGS_DIR="$WORK/internal_training/robocasa" ;;
  *) echo "FATAL: unknown cell $BENCH:$CKPT_LABEL"; exit 20 ;;
esac
[[ -d "$CKPT/params" ]] || { echo "FATAL: $CKPT has no params/"; exit 21; }

# --------------------------------------------------------------------------------------
# label gate
# --------------------------------------------------------------------------------------
CELL="$FM/cells/${BENCH}_${CKPT_LABEL}"
mkdir -p "$CELL"
if [[ ! -f "$CELL/cell.json" ]]; then
  CKPT_SHA=$(PYTHONPATH="$WSMV2/scripts/failure_modes" "$RBPY" -c \
    "import fm_common,sys;print(fm_common.sha256_tree(sys.argv[1]))" "$CKPT")
  ENC_SHA=""; [[ "$SERVE" == "workspace" ]] && ENC_SHA=$(sha256sum "$ENC" | cut -d' ' -f1)
  cat > "$CELL/cell.json" <<EOF
{"bench":"$BENCH","ckpt_label":"$CKPT_LABEL","run_id":"$RUN_ID","ckpt":"$CKPT",
 "ckpt_tree_sha256":"$CKPT_SHA","serve":"$SERVE","config_name":"$CONFIG_NAME",
 "encoder_ckpt":"${ENC_SHA:+$ENC}","encoder_sha256":"$ENC_SHA",
 "lang_table":"${ENC_SHA:+$LANG_TABLE}","tap_ckpt":"${ENC_SHA:+$WORK/ckpts/tap_149999}",
 "openpi_src":"$OPENPI_SRC","wsmv2_src":"$WSMV2_SRC","configs_dir":"$CONFIGS_DIR",
 "replan_steps":$REPLAN,"manifest":"$MANIFEST","host":"nagababa-g7e","tier":"box",
 "reset_kind":"demo_pinned_v1","rollouts_per_reset":1,
 "labelled":"$(date -u +%FT%TZ)"}
EOF
fi
echo "[cell] $(cat "$CELL/cell.json" | tr -d '\n')"

# --------------------------------------------------------------------------------------
# servers
# --------------------------------------------------------------------------------------
declare -a SERVER_PIDS=()
start_servers() {
  for ((w = 0; w < NUM_GPUS; w++)); do
    if [[ "$SERVE" == "base" ]]; then
      ( cd "$OPENPI_SRC" && CUDA_VISIBLE_DEVICES=$w XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
        WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
        OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
        exec "$OPENPI_VENV" scripts/wsm_serve_rc.py --port $((BASE_PORT + w)) \
          policy:checkpoint --policy.config="$CONFIG_NAME" --policy.dir="$CKPT" ) \
        >"$LOGS/server_$w.log" 2>&1 &
    else
      local cdir="${CONFIGS_DIR:-$WSMV2_SRC/export/lab_handoff/infra/robocasa}"
      local extra=(); [[ "$CONFIG_NAME" != pi05_robocasa_target_ft ]] && \
        extra=(--config-name "$CONFIG_NAME")
      ( cd "$OPENPI_SRC" && CUDA_VISIBLE_DEVICES=$w XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
        WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
        OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
        PYTHONPATH="$WSMV2_SRC:$OPENPI_SRC/src" \
        exec "$OPENPI_VENV" "$WSMV2_SRC/vla_training/eval/serve_pi_05_wsm_cfg.py" \
          --interface tanh --finetune-ckpt "$CKPT" --tap-ckpt "$WORK/ckpts/tap_149999" \
          --configs-dir "$cdir" --encoder-ckpt "$ENC" --task-lang-table "$LANG_TABLE" \
          --guidance-scale 1.0 --k-window 1 --stride "$REPLAN" --tap-prompt terse \
          "${extra[@]}" --port $((BASE_PORT + w)) ) \
        >"$LOGS/server_$w.log" 2>&1 &
    fi
    SERVER_PIDS+=("$!")
  done
  for ((w = 0; w < NUM_GPUS; w++)); do
    local up=0
    for i in $(seq 1 180); do
      "$RBPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',$((BASE_PORT+w))));s.close()" 2>/dev/null && { up=1; break; }
      kill -0 "${SERVER_PIDS[$w]}" 2>/dev/null || { echo "FATAL server $w died:"; tail -40 "$LOGS/server_$w.log"; exit 22; }
      sleep 10
    done
    [[ "$up" == 1 ]] || { echo "FATAL server $w never came up"; tail -40 "$LOGS/server_$w.log"; exit 22; }
  done
  echo "[servers] $NUM_GPUS up, ports $BASE_PORT..$((BASE_PORT + NUM_GPUS - 1)) ($SERVE)"
}
cleanup() { kill "${SERVER_PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

start_servers

# --------------------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------------------
PROMPT_ARG=()
[[ "$SERVE" == "workspace" ]] && PROMPT_ARG=(--task-prompt-manifest "$TASK_PROMPTS")
LIMIT_ARG=(); [[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")
FAILS=0
for TASK in $ALL_TASKS; do
  declare -a PIDS=()
  for ((w = 0; w < NUM_GPUS; w++)); do
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$w CUDA_VISIBLE_DEVICES=$w \
    PYTHONPATH="$WSMV2" PYTHONUNBUFFERED=1 \
    "$CLIENT_PY" "$WSMV2/scripts/failure_modes/fm_rollout.py" \
      --manifest "$MANIFEST" --bench "$BENCH" --task "$TASK" \
      --ckpt-label "$CKPT_LABEL" --out-root "$FM" --mode "$MODE" \
      --host 127.0.0.1 --port $((BASE_PORT + w)) \
      --shard-idx "$w" --num-shards "$NUM_GPUS" --replan-steps "$REPLAN" \
      --rollout-idx "$ROLLOUT_IDX" \
      "${PROMPT_ARG[@]}" "${LIMIT_ARG[@]}" \
      >"$LOGS/${TASK}_w$w.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS + 1)); done
  echo "[task] $TASK done ($MODE)"
done

if [[ "$FAILS" -eq 0 ]]; then
  echo "OK $(date -u +%FT%TZ)" > "$CELL/${MODE}_r${ROLLOUT_IDX}_COMPLETED"
  echo "CELL DONE $BENCH/$CKPT_LABEL/$MODE"
else
  echo "FAILED workers=$FAILS $(date -u +%FT%TZ)" > "$CELL/${MODE}_r${ROLLOUT_IDX}_FAILED"
  echo "CELL FAILED $BENCH/$CKPT_LABEL/$MODE workers=$FAILS"
  exit 23
fi
