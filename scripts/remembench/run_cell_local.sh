#!/bin/bash
# One ReMemBench cell on the local 5090s: 2 policy servers (one per GPU) + 2 task-sharded
# rollout workers + aggregation. Same protocol/runner as the sealed box arms.
#
# DEFAULTS REPRODUCE THE SDE-STUDY CELL BYTE-FOR-BYTE (CELL=<name> STD=<sigma>): base serve of
# $R/ckpt with PI_SDE_NOISE_STD=$STD PI_SDE_SCORE_CORRECT=1, aggregated as step 14999 of
# s0-9e47bc75062b23e9. Every knob below is opt-in and inert unless set:
#   CKPT        checkpoint dir with params/ + assets/           (default $R/ckpt)
#   STEP        milestone step the ckpt is                      (default 14999)
#   CKPT_URI    <run_id>/<step> recorded by the aggregator      (default s0-9e47bc75062b23e9/14999)
#   SDE         1 = the SDE-study serve (STD required, PI_SDE_* exported); 0 = plain ODE sampling,
#               the sealed protocol, STD not required                          (default 1)
#   SERVE_KIND  base | stage_e                                  (default base)
#   stage_e (Stage-E omega arms P1'/P2'/P3', serve_pi_05_wsm_cfg.py --encoder-kind stage_e):
#     ENCODER_CKPT     Stage-E encoder.pt                             (required)
#     ENCODER_SHA256   its sha256; checked here AND by the server      (required)
#     OMEGA_ROOT       the omega store the policy trained on, .../omega/<cell>/remembench (required)
#     TASK_PROMPTS     canonical remembench13 task-prompt manifest (wsm_prompt)         (required)
#     POOL_CKPT        frozen WSMv1 pool     (default $HOME/Research/TRI/wsm_data/wsm_runs/pi_wsm_v1/wsm_step100000.pt)
#     TAP_CKPT         frozen pi tap ckpt    (default $HOME/Research/TRI/wsm_data/local_ckpts/pi05_on_149999)
#     POOLED_ROOT      wsm_pooled store for the startup parity self-test
#                                            (default $HOME/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k)
#     PARITY_DEMOS     store demos replayed before serving; D7 FAIL refuses to serve   (default 3)
#     CONFIGS_DIR      wsm_robocasa_configs.py dir for the tap (default $HOME/Research/TRI/internal_training/robocasa)
#     OPENPI_SRC / WSMV2_SRC  content-addressed trees to serve from (default $OPENPI / $WSMV2)
#     WSM_TAP_MIN_BATCH       kernel-matched tap pad; 8 reproduces the B=32 store tap   (default 8)
#     LANG_TABLE_MODE  strict | task_mean_of_store (SMOKE ONLY)   (default strict)
#   DRY         1 = print the exact server/runner/aggregate commands and exit 0 (no GPU touched)
set -uo pipefail
: "${CELL:?}"
R=$HOME/Research/TRI/wsm_data/wsmv2_scratch/sde_rmb
WSMV2=$HOME/Research/TRI/wsmv2
OPENPI=$HOME/Research/robocasa_openpi
RBPY=$HOME/Research/envs/remembench_env/bin/python
CKPT=${CKPT:-$R/ckpt}
STEP=${STEP:-14999}
CKPT_URI=${CKPT_URI:-s0-9e47bc75062b23e9/14999}
SDE=${SDE:-1}
SERVE_KIND=${SERVE_KIND:-base}
DRY=${DRY:-0}
MANIFEST=$R/remembench_heldout.json
MSHA=cb24fe49f0de284cfcb0972d432f7aa791376614a76e7e76d801cc55ad0b92f8
OUT=$R/evals/$CELL; LOGS=$OUT/logs
NW=${NW:-2}; GPU_OFFSET=${GPU_OFFSET:-0}; BASE_PORT=${BASE_PORT:-5960}; ROLLOUTS=${ROLLOUTS:-3}; REPLAN=${REPLAN:-8}

# --- SDE env: exactly the historical pair when SDE=1, nothing at all when SDE=0 ---
SDE_ENV=()
if [[ "$SDE" == "1" ]]; then
  : "${STD:?}"
  SDE_ENV=(PI_SDE_NOISE_STD="$STD" PI_SDE_SCORE_CORRECT=1)
fi

# --- server command per kind ---
RUNNER_EXTRA=()
if [[ "$SERVE_KIND" == "base" ]]; then
  [[ -d "$CKPT/params" && -d "$CKPT/assets" ]] || { echo "FATAL: $CKPT lacks params/ or assets/"; exit 21; }
  server_cmd() {  # $1 = worker index
    echo "cd $OPENPI && CUDA_VISIBLE_DEVICES=$((GPU_OFFSET+$1)) XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 PYTHONUNBUFFERED=1 PYTHONPATH=$OPENPI/src:$HOME/Research/robocasa:$HOME/Research/robosuite exec env ${SDE_ENV[*]} $OPENPI/.venv/bin/python $R/serve_rmb_base.py --checkpoint $CKPT --port $((BASE_PORT+$1))"
  }
elif [[ "$SERVE_KIND" == "stage_e" ]]; then
  : "${ENCODER_CKPT:?stage_e needs ENCODER_CKPT}"; : "${ENCODER_SHA256:?stage_e needs ENCODER_SHA256}"
  : "${OMEGA_ROOT:?stage_e needs OMEGA_ROOT (.../omega/<cell>/remembench)}"; : "${TASK_PROMPTS:?stage_e needs TASK_PROMPTS}"
  POOL_CKPT=${POOL_CKPT:-$HOME/Research/TRI/wsm_data/wsm_runs/pi_wsm_v1/wsm_step100000.pt}
  TAP_CKPT=${TAP_CKPT:-$HOME/Research/TRI/wsm_data/local_ckpts/pi05_on_149999}
  POOLED_ROOT=${POOLED_ROOT:-$HOME/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k}
  PARITY_DEMOS=${PARITY_DEMOS:-3}
  CONFIGS_DIR=${CONFIGS_DIR:-$HOME/Research/TRI/internal_training/robocasa}
  OPENPI_SRC=${OPENPI_SRC:-$OPENPI}; WSMV2_SRC=${WSMV2_SRC:-$WSMV2}
  WSM_TAP_MIN_BATCH=${WSM_TAP_MIN_BATCH:-8}
  LANG_TABLE_MODE=${LANG_TABLE_MODE:-strict}
  [[ -d "$CKPT/params" && -d "$CKPT/assets" ]] || { echo "FATAL: $CKPT lacks params/ or assets/"; exit 21; }
  [[ "$SDE" == "1" ]] && { echo "FATAL: SDE=1 is the base-serve SDE study; the Stage-E serve has no SDE knob (set SDE=0)"; exit 21; }
  for f in "$ENCODER_CKPT" "$POOL_CKPT" "$TASK_PROMPTS"; do [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 21; }; done
  [[ -d "$OMEGA_ROOT" && -d "$TAP_CKPT/params" ]] || { echo "FATAL: OMEGA_ROOT or TAP_CKPT missing"; exit 21; }
  ACTUAL_SHA=$(sha256sum "$ENCODER_CKPT" | cut -d' ' -f1)
  [[ "$ACTUAL_SHA" == "$ENCODER_SHA256" ]] || { echo "FATAL: encoder sha256 $ACTUAL_SHA != ENCODER_SHA256 $ENCODER_SHA256"; exit 21; }
  mkdir -p "$OUT"
  RUNNER_EXTRA=(--task-prompt-manifest "$TASK_PROMPTS")
  server_cmd() {
    echo "cd $OPENPI_SRC && CUDA_VISIBLE_DEVICES=$((GPU_OFFSET+$1)) XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 WSM_TAP_MIN_BATCH=$WSM_TAP_MIN_BATCH OPENPI_DATA_HOME=$HOME/Research/TRI/wsm_data/openpi_cache PYTHONUNBUFFERED=1 PYTHONPATH=$WSMV2_SRC:$OPENPI_SRC/src exec $OPENPI/.venv/bin/python $WSMV2_SRC/vla_training/eval/serve_pi_05_wsm_cfg.py --interface tanh --encoder-kind stage_e --finetune-ckpt $CKPT --tap-ckpt $TAP_CKPT --configs-dir $CONFIGS_DIR --encoder-ckpt $ENCODER_CKPT --expect-encoder-sha256 $ENCODER_SHA256 --pool-ckpt $POOL_CKPT --stage-e-omega-root $OMEGA_ROOT --stage-e-pooled-root $POOLED_ROOT --stage-e-parity-demos $PARITY_DEMOS --stage-e-lang-table-mode $LANG_TABLE_MODE --stage-e-table-out $OUT/task_lang_table.npz --guidance-scale 1.0 --k-window 1 --stride $REPLAN --tap-prompt terse --port $((BASE_PORT+$1))"
  }
else
  echo "FATAL: SERVE_KIND must be base or stage_e; got $SERVE_KIND"; exit 21
fi
runner_cmd() {  # $1 = worker index
  echo "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$((GPU_OFFSET+$1)) PYTHONPATH=$WSMV2 PYTHONUNBUFFERED=1 $RBPY $WSMV2/scripts/remembench/run_remembench_eval.py --manifest $MANIFEST --manifest-sha256 $MSHA --out-dir $OUT --host 127.0.0.1 --port $((BASE_PORT+$1)) --worker-idx $1 --num-workers $NW --rollouts $ROLLOUTS --replan-steps $REPLAN --video none --obs-image-size 224 ${TASKS:+--tasks $TASKS} ${MAX_EPS:+--max-episodes-per-task $MAX_EPS} ${RUNNER_EXTRA[*]}"
}
aggregate_cmd() {
  echo "PYTHONPATH=$WSMV2 $RBPY $WSMV2/scripts/remembench/aggregate_remembench_eval.py --results-dir $OUT --arm $CELL --step $STEP --checkpoint-uri $CKPT_URI --manifest-sha256 $MSHA"
}
if [[ "$DRY" == "1" ]]; then
  for ((w=0; w<NW; w++)); do echo "[server $w] $(server_cmd $w)"; done
  for ((w=0; w<NW; w++)); do echo "[runner $w] $(runner_cmd $w)"; done
  echo "[aggregate] $(aggregate_cmd)"
  exit 0
fi
mkdir -p "$LOGS"

declare -a SP=()
for ((w=0; w<NW; w++)); do
  if [[ "$SERVE_KIND" == "base" ]]; then
    ( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$((GPU_OFFSET+w)) XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
      WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      PYTHONUNBUFFERED=1 PYTHONPATH="$OPENPI/src:$HOME/Research/robocasa:$HOME/Research/robosuite" \
      exec env "${SDE_ENV[@]}" \
      "$OPENPI/.venv/bin/python" "$R/serve_rmb_base.py" --checkpoint "$CKPT" --port $((BASE_PORT+w)) ) \
      >"$LOGS/server_$w.log" 2>&1 &
  else
    ( cd "$OPENPI_SRC" && CUDA_VISIBLE_DEVICES=$((GPU_OFFSET+w)) XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
      WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      WSM_TAP_MIN_BATCH="$WSM_TAP_MIN_BATCH" OPENPI_DATA_HOME="$HOME/Research/TRI/wsm_data/openpi_cache" \
      PYTHONUNBUFFERED=1 PYTHONPATH="$WSMV2_SRC:$OPENPI_SRC/src" \
      exec "$OPENPI/.venv/bin/python" "$WSMV2_SRC/vla_training/eval/serve_pi_05_wsm_cfg.py" \
        --interface tanh --encoder-kind stage_e --finetune-ckpt "$CKPT" --tap-ckpt "$TAP_CKPT" \
        --configs-dir "$CONFIGS_DIR" --encoder-ckpt "$ENCODER_CKPT" --expect-encoder-sha256 "$ENCODER_SHA256" \
        --pool-ckpt "$POOL_CKPT" --stage-e-omega-root "$OMEGA_ROOT" --stage-e-pooled-root "$POOLED_ROOT" \
        --stage-e-parity-demos "$PARITY_DEMOS" --stage-e-lang-table-mode "$LANG_TABLE_MODE" \
        --stage-e-table-out "$OUT/task_lang_table.npz" \
        --guidance-scale 1.0 --k-window 1 --stride "$REPLAN" --tap-prompt terse --port $((BASE_PORT+w)) ) \
      >"$LOGS/server_$w.log" 2>&1 &
  fi
  SP+=("$!")
done
cleanup(){ kill "${SP[@]}" 2>/dev/null || true; }; trap cleanup EXIT
for ((w=0; w<NW; w++)); do
  up=0
  for i in $(seq 1 180); do
    "$RBPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',$((BASE_PORT+w))));s.close()" 2>/dev/null && { up=1; break; }
    kill -0 "${SP[$w]}" 2>/dev/null || { echo "FATAL server $w died"; tail -30 "$LOGS/server_$w.log"; exit 22; }
    sleep 10
  done
  [ "$up" = 1 ] || { echo "FATAL server $w never up"; tail -30 "$LOGS/server_$w.log"; exit 22; }
done
grep -h "sde_noise_std" "$LOGS"/server_*.log | sort -u
grep -h "stage-e-serve\] encoder_id\|VERDICT" "$LOGS"/server_*.log | sort -u
if [[ "$SERVE_KIND" == "base" && "$SDE" == "1" && "$CKPT" == "$R/ckpt" && "$STEP" == "14999" && "$CKPT_URI" == "s0-9e47bc75062b23e9/14999" ]]; then
  echo "[$CELL] servers up, std=$STD"          # the historical line, byte-for-byte, on the default path
else
  echo "[$CELL] servers up, kind=$SERVE_KIND sde=$SDE${STD:+ std=$STD} ckpt=$CKPT step=$STEP ckpt_uri=$CKPT_URI"
fi

declare -a RP=()
for ((w=0; w<NW; w++)); do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$((GPU_OFFSET+w)) \
  PYTHONPATH="$WSMV2" PYTHONUNBUFFERED=1 \
  "$RBPY" "$WSMV2/scripts/remembench/run_remembench_eval.py" \
    --manifest "$MANIFEST" --manifest-sha256 "$MSHA" \
    --out-dir "$OUT" --host 127.0.0.1 --port $((BASE_PORT+w)) \
    --worker-idx "$w" --num-workers "$NW" \
    --rollouts "$ROLLOUTS" --replan-steps "$REPLAN" --video none --obs-image-size 224 \
    ${TASKS:+--tasks "$TASKS"} ${MAX_EPS:+--max-episodes-per-task "$MAX_EPS"} \
    "${RUNNER_EXTRA[@]}" \
    >"$LOGS/runner_$w.log" 2>&1 &
  RP+=("$!")
done
F=0; for p in "${RP[@]}"; do wait "$p" || F=$((F+1)); done
echo "[$CELL] runners done, $F/$NW nonzero"
PYTHONPATH="$WSMV2" "$RBPY" "$WSMV2/scripts/remembench/aggregate_remembench_eval.py" \
  --results-dir "$OUT" --arm "$CELL" --step "$STEP" --checkpoint-uri "$CKPT_URI" \
  --manifest-sha256 "$MSHA" 2>&1 | tee "$LOGS/aggregate.log"
exit $F
