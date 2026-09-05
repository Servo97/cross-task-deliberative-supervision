#!/bin/bash
# ReMemBench 13-task held-out eval on nagababa (4x RTX PRO 6000): 4 base policy servers
# (one per GPU) + 4 task-sharded rollout workers, then per-category aggregation.
#
# The rollout client runs in remembench_env (ReMemBench fork's robocasa + openpi-client);
# the policy server runs in openpi's own venv. Results are BOX TIER (not sealed).
#
# Env: ARM (run id), CKPT (local checkpoint dir with params/ + assets/), MANIFEST,
#      ROLLOUTS (3), REPLAN (8), NUM_WORKERS (4), TASKS (optional subset),
#      MAX_EPS (optional per-task cap, preflight), OUT (results dir), VIDEO (none|first).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR=/data/tmp
WORK=/data/work
OPENPI="$WORK/openpi"
WSMV2="$WORK/wsmv2"
# The ROLLOUT RUNNER's own tree. Defaults to $WSMV2, so every pi arm is unchanged. It must be
# overridable because the runner and the groot serve adapter are a MATCHED PAIR (`--obs-image-size`
# exists only in the newer tree, and a groot serve refuses the older tree's 224 default): pointing
# the server at wsmv2_groot_mech while the runner still came out of $WSMV2 would fail at argparse,
# or worse, silently on a tree where the flag existed but the guard did not.
RUNNER_SRC="${RUNNER_SRC:-$WSMV2}"
RBPY="$WORK/remembench_env/bin/python"

: "${ARM:?set ARM}"
: "${CKPT:?set CKPT}"
MANIFEST="${MANIFEST:-$WORK/remembench_heldout.json}"
MANIFEST_SHA="${MANIFEST_SHA:-cb24fe49f0de284cfcb0972d432f7aa791376614a76e7e76d801cc55ad0b92f8}"
ROLLOUTS="${ROLLOUTS:-3}"
REPLAN="${REPLAN:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_PORT="${BASE_PORT:-5900}"
# Which physical GPU worker 0 lands on. Default 0 => every existing invocation is unchanged.
# Exists because the runner shards work BY TASK (shard_tasks_lpt), so a single-task cell has
# exactly one non-empty shard and NUM_WORKERS>1 would idle 3 GPUs. Single-task cells therefore run
# one-worker-per-cell, four CELLS in parallel, which needs each to own a different device.
GPU_OFFSET="${GPU_OFFSET:-0}"
VIDEO="${VIDEO:-none}"
OUT="${OUT:-/data/work/remembench_evals/$ARM}"
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$TMPDIR"

SERVE_KIND="${SERVE_KIND:-base}"
# Frames are sent at 224 for pi (its training resolution) and at the env's NATIVE 256 for GR00T,
# whose own processor performs the single resize it trained under — a 224-padded frame would put a
# second resample in front of it. Overridable, but the per-backbone default is the correct one.
if [[ "$SERVE_KIND" == "groot" ]]; then OBS_IMAGE_SIZE="${OBS_IMAGE_SIZE:-256}"; else OBS_IMAGE_SIZE="${OBS_IMAGE_SIZE:-224}"; fi

if [[ "$SERVE_KIND" == "groot" ]]; then
  # GR00T ships an HF checkpoint dir, not an orbax params/assets pair.
  [[ -f "$CKPT/config.json" ]] || { echo "FATAL: $CKPT has no config.json (GR00T HF ckpt dir?)"; exit 21; }
  ls "$CKPT"/model*.safetensors >/dev/null 2>&1 || { echo "FATAL: $CKPT has no model*.safetensors"; exit 21; }
else
  [[ -d "$CKPT/params" ]]  || { echo "FATAL: $CKPT has no params/";  exit 21; }
  [[ -d "$CKPT/assets" ]]  || { echo "FATAL: $CKPT has no assets/";  exit 21; }
fi
[[ -f "$MANIFEST" ]]     || { echo "FATAL: no manifest $MANIFEST"; exit 21; }

cat > "$OUT/arm.json" <<EOF
{"arm":"$ARM","ckpt":"$CKPT","benchmark":"ReMemBench","serve":"${SERVE_KIND:-base}",
 "encoder_ckpt":"${ENCODER_CKPT:-}","lang_table":"${LANG_TABLE:-}","tap_ckpt":"${TAP_CKPT:-}",
 "task_prompts":"${TASK_PROMPTS:-}","interface":"${INTERFACE:-tanh}","guidance_scale":${GUIDANCE:-1.0},
 "mechanism":"${GROOT_MECHANISM:-none}","serve_extra":"${GROOT_SERVE_EXTRA:-}","gpu_offset":$GPU_OFFSET,
 "max_episodes_per_task":"${MAX_EPS:-all}",
 "rollouts_per_episode":$ROLLOUTS,"replan_steps":$REPLAN,"workers":$NUM_WORKERS,
 "manifest_sha256":"$MANIFEST_SHA","host":"nagababa-g7e","tier":"box",
 "started":"$(date -u +%FT%TZ)"}
EOF

# ---- policy servers ----
# SERVE_KIND=base      : plain openpi serve, NO workspace arguments. Used by s0 (baseline) and
#                        by the s3 JEPA arm, whose aux head is a TRAIN-time target only and is
#                        never touched by sample_actions -> it deploys exactly like base.
# SERVE_KIND=workspace : online-omega serve. --interface tanh covers BOTH the shipped tanh MLP
#                        read and the gated-DeltaNet variant; which one a ckpt holds (and the
#                        deltanet's trained window) is AUTO-DETECTED from the params tree, so
#                        the same command serves s1-tanh and s1-deltanet-w8.
# SERVE_KIND=groot    : GR00T N1.7 backbone-generality arm. Runs the openpi-websocket ADAPTER
#                       (vla_training/eval/serve_groot_ws.py) out of the GR00T venv, so the SAME
#                       runner/manifest/reset/seeding drives both backbones — only the obs/action
#                       wire mapping differs. GROOT_ENV / GROOT_SRC point at the box's GR00T venv
#                       and Isaac-GR00T checkout.
declare -a SERVER_PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  if [[ "$SERVE_KIND" == "groot" ]]; then
    GROOT_ENV="${GROOT_ENV:-$WORK/groot_env}"
    GROOT_SRC="${GROOT_SRC:-$WORK/Isaac-GR00T}"
    WSMV2_SRC="${WSMV2_SRC:-$WSMV2}"
    [[ -x "$GROOT_ENV/bin/python" ]] || { echo "FATAL: no GR00T venv at $GROOT_ENV"; exit 21; }
    ( CUDA_VISIBLE_DEVICES=$((GPU_OFFSET + w)) PYTHONUNBUFFERED=1 \
      PYTHONPATH="$WSMV2_SRC:$GROOT_SRC" \
      exec "$GROOT_ENV/bin/python" "$WSMV2_SRC/vla_training/eval/serve_groot_ws.py" \
        --model-path "$CKPT" --port $((BASE_PORT + w)) ${GROOT_SERVE_EXTRA:-} ) \
      >"$LOGS/server_$w.log" 2>&1 &
    SERVER_PIDS+=("$!")
  elif [[ "$SERVE_KIND" == "base" ]]; then
    # OPENPI_BASE_SRC lets a base-serve arm run from the content-addressed openpi tree its own
    # train manifest pins (parity rule), while still borrowing the venv for dependencies.
    # Defaults to the box's stock checkout, which is what the earlier base arms used.
    OPENPI_BASE_SRC="${OPENPI_BASE_SRC:-$OPENPI}"
    ( cd "$OPENPI_BASE_SRC" && CUDA_VISIBLE_DEVICES=$((GPU_OFFSET + w)) XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
      WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
      PYTHONPATH="$OPENPI_BASE_SRC/src" \
      exec "$OPENPI/.venv/bin/python" "$OPENPI_BASE_SRC/scripts/wsm_serve_rc.py" --port $((BASE_PORT + w)) \
        policy:checkpoint --policy.config=pi05_robocasa_target_ft --policy.dir="$CKPT" ) \
      >"$LOGS/server_$w.log" 2>&1 &
    SERVER_PIDS+=("$!")
  else
    : "${ENCODER_CKPT:?set ENCODER_CKPT}"; : "${LANG_TABLE:?set LANG_TABLE}"; : "${TAP_CKPT:?set TAP_CKPT}"
    # The workspace arms were TRAINED against a newer openpi fork than the box's default
    # checkout: gated_deltanet needs Pi0Config.wsm_cond_type, which the old tree lacks
    # (TypeError at build). Serve from the CONTENT-ADDRESSED trees pinned in the arm's train
    # run manifest (sources.openpi / sources.wsmv2), so serve-time code matches train-time
    # code exactly. The venv is reused only for its dependency set; PYTHONPATH wins over the
    # editable install, verified by openpi.__file__.
    OPENPI_SRC="${OPENPI_SRC:-$WORK/openpi_rmb}"
    WSMV2_SRC="${WSMV2_SRC:-$WORK/wsmv2_rmb}"
    ( cd "$OPENPI_SRC" && CUDA_VISIBLE_DEVICES=$((GPU_OFFSET + w)) XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
      WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
      PYTHONPATH="$WSMV2_SRC:$OPENPI_SRC/src" \
      exec "$OPENPI/.venv/bin/python" "$WSMV2_SRC/vla_training/eval/serve_pi_05_wsm_cfg.py" \
        --interface "${INTERFACE:-tanh}" --finetune-ckpt "$CKPT" --tap-ckpt "$TAP_CKPT" \
        --configs-dir "${CONFIGS_DIR:-$WSMV2_SRC/export/lab_handoff/infra/robocasa}" \
        --encoder-ckpt "$ENCODER_CKPT" --task-lang-table "$LANG_TABLE" \
        --guidance-scale "${GUIDANCE:-1.0}" --k-window 1 --stride "$REPLAN" --tap-prompt terse \
        --port $((BASE_PORT + w)) ) \
      >"$LOGS/server_$w.log" 2>&1 &
    SERVER_PIDS+=("$!")
  fi
done
cleanup() { kill "${SERVER_PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

for ((w=0; w<NUM_WORKERS; w++)); do
  up=0
  for i in $(seq 1 150); do
    "$RBPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',$((BASE_PORT+w))));s.close()" 2>/dev/null && { up=1; break; }
    kill -0 "${SERVER_PIDS[$w]}" 2>/dev/null || { echo "FATAL server $w died:"; tail -30 "$LOGS/server_$w.log"; exit 22; }
    sleep 10
  done
  [[ "$up" == 1 ]] || { echo "FATAL server $w never came up"; tail -30 "$LOGS/server_$w.log"; exit 22; }
done
echo "[servers] $NUM_WORKERS up on ports $BASE_PORT..$((BASE_PORT+NUM_WORKERS-1)) gpus $GPU_OFFSET..$((GPU_OFFSET+NUM_WORKERS-1))"

# ---- rollout workers: one per GPU, EGL pinned to the same device ----
declare -a RUNNER_PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$((GPU_OFFSET + w)) \
  PYTHONPATH="$RUNNER_SRC" PYTHONUNBUFFERED=1 \
  "$RBPY" "$RUNNER_SRC/scripts/remembench/run_remembench_eval.py" \
    --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
    --out-dir "$OUT" --host 127.0.0.1 --port $((BASE_PORT + w)) \
    --worker-idx "$w" --num-workers "$NUM_WORKERS" \
    --rollouts "$ROLLOUTS" --replan-steps "$REPLAN" --video "$VIDEO" \
    --obs-image-size "$OBS_IMAGE_SIZE" \
    ${TASKS:+--tasks "$TASKS"} ${MAX_EPS:+--max-episodes-per-task "$MAX_EPS"} \
    ${TASK_PROMPTS:+--task-prompt-manifest "$TASK_PROMPTS"} \
    >"$LOGS/runner_$w.log" 2>&1 &
  RUNNER_PIDS+=("$!")
done

FAILS=0
for pid in "${RUNNER_PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS+1)); done
echo "[runners] done, $FAILS/$NUM_WORKERS exited nonzero"

PYTHONPATH="$RUNNER_SRC" "$RBPY" "$RUNNER_SRC/scripts/remembench/aggregate_remembench_eval.py" \
  --results-dir "$OUT" --arm "$ARM" --step "${STEP:-14999}" \
  --checkpoint-uri "${CKPT_URI:-$CKPT}" --manifest-sha256 "$MANIFEST_SHA" \
  ${REQUIRE_COMPLETE:+--require-complete} 2>&1 | tee "$LOGS/aggregate.log"
AGG=$?

if [[ "$FAILS" -eq 0 && "$AGG" -eq 0 ]]; then
  echo "OK $(date -u +%FT%TZ)" > "$OUT/COMPLETED"
  echo "REMEMBENCH EVAL DONE -> $OUT/results.json"
else
  echo "FAILED runners=$FAILS agg=$AGG $(date -u +%FT%TZ)" > "$OUT/FAILED"
  echo "REMEMBENCH EVAL FAILED (runners=$FAILS agg=$AGG)"
  exit 23
fi
