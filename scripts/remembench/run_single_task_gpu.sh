#!/bin/bash
# Run one GPU's worklist of single-task ReMemBench eval cells, strictly sequentially.
#
# One cell = (task, arm) = one checkpoint. Each cell brings up its own policy server, runs the
# held-out manifest FILTERED to that cell's single task (x3 diffusion-noise rollouts), aggregates,
# marks COMPLETED, and tears the server down before the next cell starts. Never two servers on
# one GPU — a workspace serve holds two openpi models plus a torch encoder.
#
#   GPU=0 WORKLIST=/data/work/st_plan/gpu0.tsv bash run_single_task_gpu.sh
# Worklist TSV columns: short, task, arm, run_id, serve(base|workspace), rollouts
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR=/data/tmp
WORK=/data/work
OPENPI="$WORK/openpi"
WSMV2="$WORK/wsmv2"
RBPY="$WORK/remembench_env/bin/python"

: "${GPU:?set GPU}"
: "${WORKLIST:?set WORKLIST}"
CKPT_ROOT="${CKPT_ROOT:-$WORK/ckpts_st}"
MANIFEST="${MANIFEST:-$WORK/remembench_heldout.json}"
MANIFEST_SHA="${MANIFEST_SHA:-cb24fe49f0de284cfcb0972d432f7aa791376614a76e7e76d801cc55ad0b92f8}"
ROLLOUTS="${ROLLOUTS:-3}"
REPLAN="${REPLAN:-8}"
STEP="${STEP:-3999}"
PORT=$((${BASE_PORT:-5900} + GPU))
ROOT="${ROOT:-$WORK/remembench_evals/single_task}"
# Pinned per the parity rule: every one of the 24 single-task run manifests names this pair.
OPENPI_SRC="${OPENPI_SRC:-$WORK/openpi_rmb2}"
WSMV2_SRC="${WSMV2_SRC:-$WORK/wsmv2_rmb3}"
TAP_CKPT="${TAP_CKPT:-$WORK/ckpts/tap_149999}"
ENCODER_CKPT="${ENCODER_CKPT:-$WORK/wsm_artifacts/rmb/encoder.pt}"
LANG_TABLE="${LANG_TABLE:-$WORK/wsm_artifacts/rmb/task_lang_table.npz}"
TASK_PROMPTS="${TASK_PROMPTS:-$WORK/wsm_artifacts/rmb/task_prompts_remembench13.json}"

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

while IFS=$'\t' read -r SHORT TASK ARM RID SERVE NROLL; do
  [[ -z "${RID:-}" ]] && continue
  OUT="$ROOT/$TASK/$RID"
  LOGS="$OUT/logs"
  if [[ -e "$OUT/COMPLETED" ]]; then
    echo "[gpu$GPU] skip $SHORT/$ARM ($RID): already COMPLETED"
    continue
  fi
  mkdir -p "$LOGS"
  CKPT="$CKPT_ROOT/$RID/$STEP"
  if [[ ! -d "$CKPT/params" ]]; then
    echo "[gpu$GPU] FATAL $RID: no params/ at $CKPT"; echo "missing ckpt" > "$OUT/FAILED"; continue
  fi

  cat > "$OUT/arm.json" <<EOF
{"arm":"$RID","short":"$SHORT","task":"$TASK","arm_kind":"$ARM","serve":"$SERVE",
 "benchmark":"ReMemBench-single-task","step":$STEP,"ckpt":"$CKPT",
 "rollouts_per_episode":$ROLLOUTS,"replan_steps":$REPLAN,"expected_rollouts":$NROLL,
 "manifest_sha256":"$MANIFEST_SHA","host":"nagababa-g7e","tier":"box","gpu":$GPU,
 "started":"$(date -u +%FT%TZ)"}
EOF

  echo "[gpu$GPU] === $SHORT/$ARM $RID ($SERVE serve, $NROLL rollouts) $(date -u +%FT%TZ) ==="
  if [[ "$SERVE" == "base" ]]; then
    ( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
      WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
      exec "$OPENPI/.venv/bin/python" scripts/wsm_serve_rc.py --port "$PORT" \
        policy:checkpoint --policy.config=pi05_robocasa_target_ft --policy.dir="$CKPT" ) \
      >"$LOGS/server.log" 2>&1 &
    SERVER_PID=$!
  else
    ( cd "$OPENPI_SRC" && CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 \
      WSM_SERVE_NO_DATA=1 WSM_ENVS_PER_GPU=1 PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
      PYTHONPATH="$WSMV2_SRC:$OPENPI_SRC/src" \
      exec "$OPENPI/.venv/bin/python" "$WSMV2_SRC/vla_training/eval/serve_pi_05_wsm_cfg.py" \
        --interface tanh --finetune-ckpt "$CKPT" --tap-ckpt "$TAP_CKPT" \
        --configs-dir "$WSMV2_SRC/export/lab_handoff/infra/robocasa" \
        --encoder-ckpt "$ENCODER_CKPT" --task-lang-table "$LANG_TABLE" \
        --guidance-scale 1.0 --k-window 1 --stride "$REPLAN" --tap-prompt terse \
        --port "$PORT" ) \
      >"$LOGS/server.log" 2>&1 &
    SERVER_PID=$!
  fi

  up=0
  for _ in $(seq 1 180); do
    "$RBPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',$PORT));s.close()" 2>/dev/null && { up=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 10
  done
  if [[ "$up" != 1 ]]; then
    echo "[gpu$GPU] FATAL server never came up for $RID"; tail -25 "$LOGS/server.log"
    echo "server failed" > "$OUT/FAILED"; kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; continue
  fi
  # Provenance for the workspace arms: record the conditioner the checkpoint actually holds.
  grep -E "auto-detected|server ready" "$LOGS/server.log" | head -3

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$GPU \
  PYTHONPATH="$WSMV2" PYTHONUNBUFFERED=1 \
  "$RBPY" "$WSMV2/scripts/remembench/run_remembench_eval.py" \
    --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
    --out-dir "$OUT" --host 127.0.0.1 --port "$PORT" \
    --worker-idx 0 --num-workers 1 \
    --rollouts "$ROLLOUTS" --replan-steps "$REPLAN" --video none \
    --tasks "$TASK" \
    $([[ "$SERVE" == "workspace" ]] && echo --task-prompt-manifest "$TASK_PROMPTS") \
    >"$LOGS/runner.log" 2>&1
  RC=$?
  kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
  sleep 5

  if [[ "$RC" -ne 0 ]]; then
    echo "[gpu$GPU] FAILED $RID (runner rc=$RC)"; tail -15 "$LOGS/runner.log"
    echo "runner rc=$RC" > "$OUT/FAILED"; continue
  fi
  PYTHONPATH="$WSMV2" "$RBPY" "$WSMV2/scripts/remembench/aggregate_remembench_eval.py" \
    --results-dir "$OUT" --arm "$RID" --step "$STEP" \
    --checkpoint-uri "$CKPT" --manifest-sha256 "$MANIFEST_SHA" \
    >"$LOGS/aggregate.log" 2>&1
  if [[ $? -eq 0 ]]; then
    echo "OK $(date -u +%FT%TZ)" > "$OUT/COMPLETED"
    grep -E "OVERALL|pooled" "$LOGS/aggregate.log" | head -2
    echo "[gpu$GPU] DONE $SHORT/$ARM $RID"
  else
    echo "aggregate failed" > "$OUT/FAILED"; tail -10 "$LOGS/aggregate.log"
  fi
done < "$WORKLIST"

echo "GPU$GPU WORKLIST DONE $(date -u +%FT%TZ)" > "$ROOT/.gpu${GPU}_done"
echo "[gpu$GPU] worklist complete $(date -u +%FT%TZ)"
