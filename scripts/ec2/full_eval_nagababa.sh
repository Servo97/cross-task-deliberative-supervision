#!/bin/bash
# FULL 50-task x 100-episode eval on nagababa (5000 rollouts, exact sealed protocol shape).
# 4 GPUs = 4 task-sharded workers (server per GPU + K shard clients each), mirroring the node
# entry's NUM_WORKERS layout. Results are box-tier (not sealed/claimed); provenance = arm.json.
# Env: ARM, CKPT, SERVER_KIND=base|robottt, K (default 8).
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"; export TMPDIR=/data/tmp
WORK=/data/work
OPENPI="$WORK/openpi"; WSMV2="$WORK/wsmv2"; SIMPY="$WORK/simenv/bin/python"
: "${ARM:?}"; : "${CKPT:?}"
SERVER_KIND="${SERVER_KIND:-base}"; K="${K:-8}"; NUM_WORKERS=4; BASE_PORT=5800
SEED=20260723
TASK_SETS="atomic_seen,composite_seen,composite_unseen"
EPISODE_MANIFEST="$WORK/canonical_episode_manifest.json"
HELD_ROOT="$WORK/heldout_full"
RESULTS_DIR="/data2/evals/full_${ARM}_$(date -u +%m%d_%H%M)"
LOGS="$RESULTS_DIR/logs"; mkdir -p "$LOGS"
echo "{\"arm\":\"$ARM\",\"ckpt\":\"$CKPT\",\"kind\":\"full5000\",\"server\":\"$SERVER_KIND\",\"k\":$K,\"workers\":$NUM_WORKERS,\"host\":\"nagababa-g7e\",\"seed\":$SEED}" > "$RESULTS_DIR/arm.json"
[[ -d "$CKPT/params" ]] || { echo "FATAL $CKPT has no params/"; exit 21; }
[[ -f "$HELD_ROOT/.prep_done" ]] || { echo "FATAL heldout_full not prepped"; exit 26; }

declare -a SERVER_PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  if [[ "$SERVER_KIND" == "base" ]]; then
    STATE_MODE=stateless_v1
    ( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$w XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 \
      WSM_ENVS_PER_GPU=$K PI_WSM_SERVER_STATE_MODE=stateless_v1 \
      OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
      exec "$OPENPI/.venv/bin/python" scripts/wsm_serve_rc.py --port $((BASE_PORT + w)) \
        policy:checkpoint --policy.config=pi05_robocasa_target_ft --policy.dir="$CKPT" ) \
      >"$LOGS/server_$w.log" 2>&1 &
    SERVER_PIDS+=("$!")
  else
    STATE_MODE=per_env_isolated_v1
    CUDA_VISIBLE_DEVICES=$w XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 \
    WSM_ENVS_PER_GPU=$K PI_WSM_SERVER_STATE_MODE=per_env_isolated_v1 \
    OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 PYTHONPATH="$WSMV2:$OPENPI/src" \
    "$OPENPI/.venv/bin/python" "$WSMV2/vla_training/eval/serve_pi_05_robottt.py" \
      --finetune-ckpt "$CKPT" --stride 8 --port $((BASE_PORT + w)) \
      >"$LOGS/server_$w.log" 2>&1 &
    SERVER_PIDS+=("$!")
  fi
done
trap 'kill "${SERVER_PIDS[@]}" 2>/dev/null || true' EXIT
for ((w=0; w<NUM_WORKERS; w++)); do
  for i in $(seq 1 120); do
    "$SIMPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect((\"127.0.0.1\",$((BASE_PORT + w))));s.close()" 2>/dev/null && break
    kill -0 "${SERVER_PIDS[$w]}" 2>/dev/null || { echo "FATAL server $w died:"; tail -20 "$LOGS/server_$w.log"; exit 22; }
    sleep 10
  done
done
echo "[servers] 4 up (base port $BASE_PORT)"

cd "$WSMV2"
declare -a RUNNER_PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  for ((j=0; j<K; j++)); do
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$w PYTHONPATH="$WSMV2" \
    "$SIMPY" vla_training/eval/eval_pi_05.py \
      --config "$WSMV2/scripts/configs/eval/pi05_eval.yaml" \
      --worker-idx "$w" --num-workers "$NUM_WORKERS" \
      --host 127.0.0.1 --port $((BASE_PORT + w)) --out-dir "$RESULTS_DIR" \
      --task-sets "$TASK_SETS" --num-trials 100 --video none --seed "$SEED" \
      --episode-manifest "$EPISODE_MANIFEST" --heldout-root "$HELD_ROOT" --rollouts-per-demo 1 \
      --replan-steps 8 \
      --episode-shard-idx "$j" --num-episode-shards "$K" \
      --server-state-mode "$STATE_MODE" \
      >"$LOGS/runner_${w}_$j.log" 2>&1 &
    RUNNER_PIDS+=("$!")
  done
done
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOGS/gpu_mem.csv"; sleep 300; done ) &
MEM_PID=$!; trap 'kill "${SERVER_PIDS[@]}" $MEM_PID 2>/dev/null || true' EXIT
FAILS=0
for pid in "${RUNNER_PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS+1)); done
kill $MEM_PID 2>/dev/null || true

"$SIMPY" - "$RESULTS_DIR" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
splits = {}
for f in sorted(root.rglob("stats_shard*.json")):
    d = json.load(open(f)); split, task = f.parent.parent.name, f.parent.name
    s = sum(1 for e in d["per_episode"] if e.get("success"))
    a, b = splits.setdefault(split, {}).setdefault(task, (0, 0)); splits[split][task] = (a + s, b + len(d["per_episode"]))
task_means, tot_s, tot_n = [], 0, 0
for split, tasks in sorted(splits.items()):
    ms = [s / n for s, n in tasks.values()]
    print(f"{split}: mean {sum(ms)/len(ms):.3f} over {len(ms)} tasks ({sum(n for _, n in tasks.values())} eps)")
    task_means += ms
    tot_s += sum(s for s, _ in tasks.values()); tot_n += sum(n for _, n in tasks.values())
print(f"FULL EVAL avg_task_weighted = {sum(task_means)/len(task_means):.4f}  episodes={tot_n}")
json.dump({sp: {t: {"succ": s, "n": n} for t, (s, n) in ts.items()} for sp, ts in splits.items()},
          open(root / "full_summary.json", "w"), indent=1)
EOF
[[ "$FAILS" -eq 0 ]] || { echo "FULL EVAL FAILED ($FAILS runners)"; exit 23; }
echo "FULL EVAL DONE"
