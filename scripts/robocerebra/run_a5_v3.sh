#!/usr/bin/env bash
# A5 / Stage-Q "Q2" arm under protocol v3. Same two gates as the main ladder (rc + artifact).
#
# Upstream gates, all passed before this script was written:
#   * serve_pi05_libero_stageq.py --self-test -> ALL PASS (fixtures A/B/C)
#   * assert_robottt_loaded -> readout kernel non-zero, alpha != gate_init (19 tensors, finite)
#   * CRN inertness probe vs A0 on shared coordinates -> 8/8 identical env init, 0/8 identical
#     action digests, i.e. the fast-weight read demonstrably changes the actions.
#
# Standing caveat for whoever reads the result: the TRAINED gate is tiny. mean|tanh(alpha)| =
# 2.36e-4 against a 1e-3 init -- training drove the read DOWN by ~4.4x. A null here is therefore a
# property of the CHECKPOINT, not of the serving path.
set -uo pipefail
WSM=${WSM_DATA:-/home/sarveshp/Research/TRI/wsm_data/robocerebra}
REPO=${WSM_REPO:-/home/sarveshp/Research/TRI/wsmv2}
OPENPI=${OPENPI_ROOT:-/home/sarveshp/Research/robocasa_openpi}
LOGS=$WSM/logs_v3; K=${K:-8}; GPU=${GPU:-1}; PORT=${PORT:-8030}; TRIALS=${TRIALS:-10}
CKPT=$WSM/ckpts/a5_stageq_q2/14999
ALL="Ideal Observation_Mismatching Random_Disturbance Memory_Execution Memory_Exploration Mix"
MEM="Memory_Execution Memory_Exploration"
mkdir -p "$LOGS"
say(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOGS/a5.log"; }

# Bracket the first char so this pattern can never match this script's own command line.
# 6 s was NOT enough: a JAX server holds its ~18 GB for several seconds after SIGTERM, and the
# next server then OOMs on a GPU that "looks" free to a naive check. Wait for the memory to come
# back, not for a fixed sleep.
stop_server(){
  pkill -f "[s]erve_pi05_libero_stageq.py --checkpoint" 2>/dev/null
  for _ in $(seq 1 40); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d " ")
    [ -n "$free" ] && [ "$free" -ge 20000 ] && { sleep 2; return 0; }
    sleep 3
  done
  say "WARNING: GPU $GPU still below 20 GB free after 120 s"
}

stop_server
( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$GPU WSM_ENVS_PER_GPU=$K \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH=/home/sarveshp/Research/robocasa:/home/sarveshp/Research/robosuite:$REPO \
    .venv/bin/python "$REPO/scripts/robocerebra/serve_pi05_libero_stageq.py" \
      --checkpoint "$CKPT" --config pi05_robocerebra_stageq_q2 --max-envs "$K" --port "$PORT" \
      > "$LOGS/server_a5.log" 2>&1 ) &
up=0
for _ in $(seq 1 120); do
  grep -q "server listening" "$LOGS/server_a5.log" 2>/dev/null && { up=1; break; }
  sleep 5
done
[ $up -eq 1 ] || { say "SERVER FAILED"; exit 1; }
say "server up: a5_stageq_q2 $(grep -o 'max|tanh(alpha)|=[0-9.]*' "$LOGS/server_a5.log" | tail -1)"

run_cell(){  # $1=modes  $2=trials  $3=trial_start  $4=out
  local nm exp rc
  nm=$(echo "$1" | wc -w); exp=$((10 * $2 * nm))
  say "cell: modes=[$1] trials=$2 start=$3 -> $(basename "$4")"
  "$REPO/scripts/robocerebra/run_eval_sharded.sh" --k "$K" --gpu "$GPU" --port "$PORT" --out "$4" \
      --modes $1 --trials "$2" --trial-start "$3" --wsm --wsm-env-id env0 \
      --arm A5_stageq_q2 --ckpt-sha "a5_stageq_q2-6cd89319@14999" --budget-steps 15000 \
      --note v3_a5 >> "$LOGS/a5.run.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { say "RC-GATE FAIL $(basename "$4"): exit $rc"; return 1; }
  /usr/bin/python3 - "$4" "$exp" <<'PY' | tee -a "$LOGS/a5.log"
import json, sys
p, exp = sys.argv[1], int(sys.argv[2])
d = json.load(open(p)); pt = d.get("per_trial", [])
if d.get("complete") is not True or len(pt) != exp:
    print(f"ARTIFACT-GATE FAIL {p}: complete={d.get('complete')} {len(pt)}/{exp}"); sys.exit(1)
if any(r.get("protocol") != "v3" for r in pt):
    print(f"ARTIFACT-GATE FAIL {p}: non-v3 rows"); sys.exit(1)
a = sum(r["agent_subtasks"] for r in pt); q = sum(r["possible_subtasks"] for r in pt)
print(f"ARTIFACT-GATE PASS A5 {p.split('/')[-1]}: {len(pt)} trials, {a}/{q} = {100*a/q:.2f}%")
PY
  return "${PIPESTATUS[0]}"
}

run_cell "$ALL" "$TRIALS" 0 "$LOGS/v3_a5_stageq_q2_6mode.json"
run_cell "$MEM" "$TRIALS" "$TRIALS" "$LOGS/v3_a5_stageq_q2_memtopup.json"
stop_server
say "=== A5 done ==="
