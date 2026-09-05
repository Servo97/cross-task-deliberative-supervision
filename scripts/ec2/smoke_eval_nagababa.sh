#!/bin/bash
# nagababa smoke v0 — serve+rollout MECHANICS only (no S3 publishes, no run manifest, no claims).
# Purpose: prove any entry/serve change end-to-end in <1h on 1 GPU before burning a p5 round-trip.
# Shape mirrors robocasa_eval_entry.sh EXACTLY for the s0/base arm: same server command
# (wsm_serve_rc.py policy:checkpoint pi05_robocasa_target_ft), same client command
# (eval_pi_05.py + episode-manifest shards + stateless_v1), same seed/protocol pins.
# v1 (provenance-honest --smoke namespace across launcher/validators) is a separate change.
# Run: tmux new -d -s smoke 'bash /data/smoke_eval_nagababa.sh 2>&1 | tee -a /data2/evals/smoke.log'
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"

WORK=/data/work
OPENPI="$WORK/openpi"; WSMV2="$WORK/wsmv2"; SIMPY="$WORK/simenv/bin/python"
CODE_DIR="$WORK/internal_training"
STUDY_S3="s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"

# Protocol pins (match the sealed E4/decisive manifests)
SEED=20260723
TASK_SETS="atomic_seen,composite_seen,composite_unseen"
NUM_TRIALS=100
REPLAN_STEPS=8
EPISODE_MANIFEST_SHA256="d57ab80be9ee2c14a70d0d28dd3722586200e9e5fe43207d1d11fc48d22d889a"
EPISODE_MANIFEST_S3="$STUDY_S3/manifests/artifacts/eval/heldout50/$EPISODE_MANIFEST_SHA256.json"
CKPT_S3="${CKPT_S3:-$STUDY_S3/checkpoints/pi05/s0/s0-c43f076daad4a799/59999}"

SMOKE_TASK="${SMOKE_TASK:-TurnOnMicrowave}"   # 1 task => 1 server GPU + K shard clients
K_ENVS="${K_ENVS:-8}"
GPU="${GPU:-0}"
BASE_PORT=5600
RESULTS_DIR="${RESULTS_DIR:-/data2/evals/smoke_$(date -u +%Y%m%d_%H%M%S)}"
LOGS="$RESULTS_DIR/logs"; mkdir -p "$LOGS"

# ---- inputs ----
CKPT="$WORK/ckpt/s0-59999"; mkdir -p "$CKPT"
aws s3 sync "$CKPT_S3" "$CKPT" --only-show-errors
[[ -d "$CKPT/params" ]] || { echo "FATAL ckpt has no params/ (orbax leaf missing)"; exit 21; }
EPISODE_MANIFEST="$WORK/canonical_episode_manifest.json"
aws s3 cp "$EPISODE_MANIFEST_S3" "$EPISODE_MANIFEST" --only-show-errors
echo "$EPISODE_MANIFEST_SHA256  $EPISODE_MANIFEST" | sha256sum -c -
HELD_ROOT="$WORK/heldout_smoke"
PYTHONPATH="$WSMV2" "$SIMPY" "$CODE_DIR/robocasa/prep_heldout_root.py" \
  --root "$HELD_ROOT" --tasks "$SMOKE_TASK" \
  --episode-manifest "$EPISODE_MANIFEST" --workers 16

# ---- server (exact base-branch command; mirror entry's staging block ~line 404) ----
cp "$CODE_DIR/robocasa/wsm_robocasa_configs.py" "$OPENPI/src/openpi/training/"
cp "$CODE_DIR/robocasa/wsm_serve_rc.py" "$OPENPI/scripts/"
# tokenizer pre-fetch (openpi's gs:// maybe_download is deterministically broken — entry does this)
export OPENPI_DATA_HOME="$WORK/openpi_cache"
TOKCACHE="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"; mkdir -p "$(dirname "$TOKCACHE")"
[[ -s "$TOKCACHE" ]] || curl -fsSL --retry 3 https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o "$TOKCACHE" || true
cd "$OPENPI"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 \
WSM_ENVS_PER_GPU=$K_ENVS PI_WSM_SERVER_STATE_MODE=stateless_v1 \
OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 \
"$OPENPI/.venv/bin/python" scripts/wsm_serve_rc.py --port $BASE_PORT \
  policy:checkpoint --policy.config=pi05_robocasa_target_ft --policy.dir="$CKPT" \
  >"$LOGS/server_0.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  "$SIMPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect((\"127.0.0.1\",$BASE_PORT));s.close()" 2>/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "FATAL server died:"; tail -30 "$LOGS/server_0.log"; exit 22; }
  sleep 10
done
echo "[server] up on :$BASE_PORT"

# ---- K shard clients (exact wsm_pi runner command; worker 0 of 1) ----
cd "$WSMV2"
declare -a RUNNER_PIDS=()
for ((j=0; j<K_ENVS; j++)); do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$GPU PYTHONPATH="$WSMV2" \
  "$SIMPY" vla_training/eval/eval_pi_05.py \
    --config "$WSMV2/scripts/configs/eval/pi05_eval.yaml" \
    --worker-idx 0 --num-workers 1 \
    --host 127.0.0.1 --port $BASE_PORT --out-dir "$RESULTS_DIR" \
    --task-sets "$TASK_SETS" --num-trials "$NUM_TRIALS" --video none --seed "$SEED" \
    --tasks "$SMOKE_TASK" \
    --episode-manifest "$EPISODE_MANIFEST" --heldout-root "$HELD_ROOT" --rollouts-per-demo 1 \
    --replan-steps "$REPLAN_STEPS" \
    --episode-shard-idx "$j" --num-episode-shards "$K_ENVS" \
    --server-state-mode stateless_v1 \
    >"$LOGS/runner_0_$j.log" 2>&1 &
  RUNNER_PIDS+=("$!")
done
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOGS/gpu_mem.csv"; sleep 60; done ) &
MEM_PID=$!; trap 'kill $SERVER_PID $MEM_PID 2>/dev/null || true' EXIT

FAILS=0
for pid in "${RUNNER_PIDS[@]}"; do wait "$pid" || FAILS=$((FAILS+1)); done
kill $MEM_PID 2>/dev/null || true
echo "[runners] done, FAILS=$FAILS"

# ---- summary (no publishes) ----
"$SIMPY" - "$RESULTS_DIR" <<'EOF'
import json, pathlib, statistics as st, sys
root = pathlib.Path(sys.argv[1])
total = succ = 0; times = []
for f in sorted(root.rglob("stats_shard*.json")):
    d = json.load(open(f))
    eps = d["per_episode"]  # robocasa_episode_shard_results schema
    s = sum(1 for e in eps if e.get("success"))
    total += len(eps); succ += s
    print(f"{f.relative_to(root)}: {s}/{len(eps)}")
    times += [c["policy_model_amortized_ms"] for e in eps for c in e.get("policy_timing_calls", [])[1:]]
print(f"SMOKE TOTAL: {succ}/{total} successes ({succ/max(total,1):.1%})")
if times:
    print(f"amortized model ms: median {st.median(times):.0f}, p90 {sorted(times)[int(.9*len(times))]:.0f}")
EOF
tail -1 "$LOGS/gpu_mem.csv" || true
[[ "$FAILS" -eq 0 ]] || { echo "SMOKE FAILED ($FAILS runner(s))"; exit 23; }
echo "SMOKE PASSED"
