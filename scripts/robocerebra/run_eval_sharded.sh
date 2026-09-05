#!/usr/bin/env bash
# K env-runner processes per GPU against ONE gather-batching policy server, then merge.
#
# The server must ALREADY be running with WSM_ENVS_PER_GPU=K in its environment (that variable is
# what turns openpi's gather on; setting it only here does nothing). Launch order, one GPU:
#
#   # shell 1 -- openpi venv, the server, GPU 0
#   cd /home/sarveshp/Research/robocasa_openpi
#   CUDA_VISIBLE_DEVICES=0 WSM_ENVS_PER_GPU=8 \
#   PYTHONPATH=/home/sarveshp/Research/TRI/wsmv2:/home/sarveshp/Research/robocasa:/home/sarveshp/Research/robosuite \
#     .venv/bin/python .../serve_pi05_libero_wsm.py --checkpoint <arm> --config <cfg> --port 8000
#
#   # shell 2 -- this script, sim venv, envs on GPU 0
#   scripts/robocerebra/run_eval_sharded.sh --k 8 --gpu 0 --port 8000 \
#     --arm A1_gdn_w8 --ckpt-sha <sha> --encoder-sha 09a1107d... --wsm \
#     --modes Ideal --trials 10 --out /path/results.json
#
# MUJOCO_EGL_DEVICE_ID is the render pin and CUDA_VISIBLE_DEVICES is deliberately NOT set: the
# NVIDIA driver filters EGL device enumeration by CUDA_VISIBLE_DEVICES, so setting both makes
# MUJOCO_EGL_DEVICE_ID=1 index past the end of a one-device list and eglMakeCurrent fails. The sim
# process needs no CUDA context of its own anyway.
#
# Every runner gets its own MUJOCO_EGL_DEVICE_ID, its own wsm_env_id (the ω server keys ALL per-env
# state on that string, so two runners sharing one would corrupt each other's ω window) and its own
# shard of the TRIALS. --deterministic-seeding is forced on: without it the shards do not reproduce
# the unsharded run. --trace-digest is forced on too: it costs a hash per request and turns
# "K=8 agreed with K=1" from an assertion into a value stored in the results file. Results are merged
# by merge_eval_shards.py, which recomputes every rate from the union of per-trial rows.
#
# env0..env{K-1} are REUSED across cells on purpose. The ω server never garbage-collects an env slot
# on disconnect (a closed socket does not prove the episode ended), so a fresh id set per cell would
# hit --max-envs; a t=0 request on an existing slot resets it for free.
set -uo pipefail

WSM=${WSM_DATA:-/home/sarveshp/Research/TRI/wsm_data/robocerebra}
REPO=${WSM_REPO:-/home/sarveshp/Research/TRI/wsmv2}
SIM_PY=${SIM_PY:-$WSM/venv_sim/bin/python}
HARNESS=$REPO/scripts/robocerebra/eval_robocerebra_openpi.py

K=8; GPU=0; PORT=8000; OUT=""; HOST=0.0.0.0
PASSTHRU=()
while [ $# -gt 0 ]; do
  case "$1" in
    --k) K=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ -x "$SIM_PY" ] || { echo "sim venv python not found: $SIM_PY" >&2; exit 2; }

BASE="${OUT%.json}"
mkdir -p "$(dirname "$OUT")"
echo "[sharded-eval] K=$K envs on GPU $GPU -> server $HOST:$PORT; shards -> ${BASE}.shard<i>.json"

PIDS=()
for ((i=0; i<K; i++)); do
  (
    cd "$WSM" || exit 3
    LIBERO_CONFIG_PATH=$WSM/libero_config \
    PYTHONPATH=$WSM/code/LIBERO \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=$GPU \
    "$SIM_PY" "$HARNESS" \
      --bench-root "$WSM/RoboCerebraBench" \
      --host "$HOST" --port "$PORT" \
      --num-shards "$K" --shard "$i" --deterministic-seeding --trace-digest \
      --wsm-env-id "env$i" \
      --out "${BASE}.shard${i}.json" \
      "${PASSTHRU[@]}"
  ) > "${BASE}.shard${i}.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "[sharded-eval] at least one runner FAILED; see ${BASE}.shard*.log" >&2
  echo "[sharded-eval] merging what completed (--allow-partial)" >&2
  "$SIM_PY" "$REPO/scripts/robocerebra/merge_eval_shards.py" \
    --shards "${BASE}".shard*.json --out "$OUT" --allow-partial
  exit 1
fi

"$SIM_PY" "$REPO/scripts/robocerebra/merge_eval_shards.py" \
  --shards "${BASE}".shard*.json --out "$OUT"
