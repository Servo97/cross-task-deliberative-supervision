#!/bin/bash
# Single RoboTTT serve-ablation arm on nagababa (smoke tier — NO publishes, results self-labeled).
# Env params: ARM (label), CKPT (local dir), TASKS (comma), GPU, ABLATION (ROBOTTT_ABLATION value,
# empty = unmodified control), K (default 8), PORT (default 5700+GPU).
# Serve command = exact node robottt_fast branch + ablation/probe envs (see robottt_ablation_knobs.md).
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"; export TMPDIR=/data/tmp
WORK=/data/work
OPENPI="$WORK/openpi"; WSMV2="$WORK/wsmv2"; SIMPY="$WORK/simenv/bin/python"
: "${ARM:?}"; : "${CKPT:?}"; : "${TASKS:?}"; : "${GPU:?}"
K="${K:-8}"; PORT="${PORT:-$((5700 + GPU))}"; ABLATION="${ABLATION:-}"
SERVER_KIND="${SERVER_KIND:-robottt}"   # robottt | base (base = wsm_serve_rc, any plain-pi05 ckpt)
SEED=20260723
TASK_SETS="atomic_seen,composite_seen,composite_unseen"
EPISODE_MANIFEST="$WORK/canonical_episode_manifest.json"
HELD_ROOT="$WORK/heldout_ablate"
RESULTS_DIR="/data2/evals/ablate_${ARM}_$(date -u +%m%d_%H%M)"
LOGS="$RESULTS_DIR/logs"; mkdir -p "$LOGS"
echo "{\"arm\":\"$ARM\",\"ablation\":\"$ABLATION\",\"ckpt\":\"$CKPT\",\"tasks\":\"$TASKS\",\"gpu\":$GPU,\"k\":$K,\"host\":\"nagababa-g7e\"}" > "$RESULTS_DIR/arm.json"

[[ -d "$CKPT/params" ]] || { echo "FATAL $CKPT has no params/"; exit 21; }

if [[ "$SERVER_KIND" == "base" ]]; then
  [[ -z "$ABLATION" ]] || { echo "FATAL base server takes no ROBOTTT_ABLATION"; exit 25; }
  STATE_MODE=stateless_v1
  cd "$OPENPI"
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 \
  WSM_ENVS_PER_GPU=$K PI_WSM_SERVER_STATE_MODE=stateless_v1 \
  OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
  "$OPENPI/.venv/bin/python" scripts/wsm_serve_rc.py --port "$PORT" \
    policy:checkpoint --policy.config=pi05_robocasa_target_ft --policy.dir="$CKPT" \
    >"$LOGS/server.log" 2>&1 &
  SERVER_PID=$!
else
  STATE_MODE=per_env_isolated_v1
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 \
  WSM_ENVS_PER_GPU=$K PI_WSM_SERVER_STATE_MODE=per_env_isolated_v1 \
  ROBOTTT_ABLATION="$ABLATION" ROBOTTT_ABLATION_ACK=smoke ROBOTTT_PROBE_LOG="$RESULTS_DIR/probe.jsonl" \
  OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 PYTHONPATH="$WSMV2:$OPENPI/src" \
  "$OPENPI/.venv/bin/python" "$WSMV2/vla_training/eval/serve_pi_05_robottt.py" \
    --finetune-ckpt "$CKPT" --stride 8 --port "$PORT" \
    >"$LOGS/server.log" 2>&1 &
  SERVER_PID=$!
fi
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  "$SIMPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect((\"127.0.0.1\",$PORT));s.close()" 2>/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "FATAL server died:"; tail -30 "$LOGS/server.log"; exit 22; }
  sleep 10
done
echo "[server] $ARM up on :$PORT (ablation='$ABLATION')"

cd "$WSMV2"
declare -a RUNNER_PIDS=()
for ((j=0; j<K; j++)); do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$GPU PYTHONPATH="$WSMV2" \
  "$SIMPY" vla_training/eval/eval_pi_05.py \
    --config "$WSMV2/scripts/configs/eval/pi05_eval.yaml" \
    --worker-idx 0 --num-workers 1 \
    --host 127.0.0.1 --port "$PORT" --out-dir "$RESULTS_DIR" \
    --task-sets "$TASK_SETS" --num-trials 100 --video none --seed "$SEED" \
    --tasks "$TASKS" \
    --episode-manifest "$EPISODE_MANIFEST" --heldout-root "$HELD_ROOT" --rollouts-per-demo 1 \
    --replan-steps 8 \
    --episode-shard-idx "$j" --num-episode-shards "$K" \
    --server-state-mode "$STATE_MODE" \
    >"$LOGS/runner_$j.log" 2>&1 &
  RUNNER_PIDS+=("$!")
done
FAILS=0
for pid in "${RUNNER_PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS+1)); done
echo "[runners] $ARM done, FAILS=$FAILS"

"$SIMPY" - "$RESULTS_DIR" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
per_task = {}
for f in sorted(root.rglob("stats_shard*.json")):
    d = json.load(open(f)); t = f.parent.name
    s = sum(1 for e in d["per_episode"] if e.get("success"))
    a, b = per_task.get(t, (0, 0)); per_task[t] = (a + s, b + len(d["per_episode"]))
tot = succ = 0
for t, (s, n) in sorted(per_task.items()):
    print(f"{t}: {s}/{n}"); succ += s; tot += n
print(f"ARM TOTAL: {succ}/{tot} ({succ/max(tot,1):.1%})")
json.dump({t: {"succ": s, "n": n} for t, (s, n) in per_task.items()}, open(root/"arm_summary.json", "w"), indent=1)
EOF
[[ "$FAILS" -eq 0 ]] || { echo "ARM $ARM FAILED ($FAILS runners)"; exit 23; }
echo "ARM $ARM DONE"
