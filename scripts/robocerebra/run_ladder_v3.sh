#!/usr/bin/env bash
# Full 5-arm RoboCerebra ladder under PROTOCOL v3 (corrected scoring).
#
# Order: A0 6-mode -> A3 (fast/plain server) -> A1/A2/A4 (omega server, mem 0.55)
#        -> Memory_Execution/Memory_Exploration N=20 top-ups for every arm.
#
# Gating, both kinds, because a green exit code on a run that produced half a results file is the
# failure mode that cost this campaign a withdrawn table:
#   * rc-gated      -- every stage's exit status is checked; a non-zero stops that arm.
#   * artifact-gated-- after each cell the merged json must have complete=true AND the expected
#                      per_trial count, or the arm is marked FAILED and the ladder moves on.
#
# GPU safety: refuses to start unless BOTH GPUs are idle on three consecutive polls 20 s apart.
# The SDE sigma-sweep shares this box; preempting it is not an option.
set -uo pipefail

WSM=${WSM_DATA:-/home/sarveshp/Research/TRI/wsm_data/robocerebra}
REPO=${WSM_REPO:-/home/sarveshp/Research/TRI/wsmv2}
OPENPI=${OPENPI_ROOT:-/home/sarveshp/Research/robocasa_openpi}
LOGS=$WSM/logs_v3
K=${K:-6}
GPU=${GPU:-0}
PORT=${PORT:-8010}
TRIALS=${TRIALS:-10}
ENCODER_SHA=${ENCODER_SHA:-09a1107d486ae6bfe3112e4858c3a9101e8a934297b21b8fbb13cb3118acc483}
ALL_MODES="Ideal Observation_Mismatching Random_Disturbance Memory_Execution Memory_Exploration Mix"
MEM_MODES="Memory_Execution Memory_Exploration"
mkdir -p "$LOGS"

say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOGS/ladder.log"; }

# ---- GPU idle gate (3-poll rule) -----------------------------------------------------------
# Self-excluding: our own server/shards must not count as "someone else is using the GPU".
others_on_gpu() {
  local mine
  mine=$(pgrep -f "serve_pi05_libero(_wsm)?\.py|eval_robocerebra_openpi\.py" | tr '\n' '|')
  mine="${mine%|}"
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  local n=0
  for p in $pids; do
    if [ -z "$mine" ] || ! echo "$p" | grep -qE "^(${mine})$"; then n=$((n+1)); fi
  done
  echo "$n"
}
wait_for_idle_gpus() {
  local hits=0
  while [ "$hits" -lt 3 ]; do
    if [ "$(others_on_gpu)" -eq 0 ]; then
      hits=$((hits+1)); say "GPU idle poll $hits/3"
    else
      [ "$hits" -ne 0 ] && say "GPU busy again, resetting poll counter"
      hits=0
    fi
    [ "$hits" -lt 3 ] && sleep 20
  done
  say "GPUs idle on 3 consecutive polls -- proceeding"
}

stop_server() { pkill -f "serve_pi05_libero.*--port $PORT" 2>/dev/null; sleep 8; }

start_server() {  # $1=kind(plain|omega) $2=ckpt $3=config $4=memfrac
  stop_server
  local extra=() script=$REPO/scripts/robocerebra/serve_pi05_libero.py
  if [ "$1" = omega ]; then
    script=$REPO/scripts/robocerebra/serve_pi05_libero_wsm.py
    extra=(--max-envs "$K")
  fi
  ( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$GPU WSM_ENVS_PER_GPU=$K \
      XLA_PYTHON_CLIENT_MEM_FRACTION=$4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH=/home/sarveshp/Research/robocasa:/home/sarveshp/Research/robosuite \
      .venv/bin/python "$script" --checkpoint "$2" --config "$3" --port "$PORT" "${extra[@]}" \
      > "$LOGS/server_${5}.log" 2>&1 ) &
  for _ in $(seq 1 90); do
    grep -q "server listening" "$LOGS/server_${5}.log" 2>/dev/null && { say "server up: $5"; return 0; }
    sleep 5
  done
  say "SERVER FAILED TO START: $5"; return 1
}

# artifact gate: merged json must be complete and hold the expected number of trials
check_artifact() {  # $1=json $2=expected_trials $3=label
  /usr/bin/python3 - "$1" "$2" "$3" <<'PY'
import json,sys
path,exp,label=sys.argv[1],int(sys.argv[2]),sys.argv[3]
try: d=json.load(open(path))
except Exception as e: print(f"ARTIFACT-GATE FAIL {label}: unreadable ({e})"); sys.exit(1)
pt=d.get("per_trial",[]); comp=d.get("complete")
if comp is not True: print(f"ARTIFACT-GATE FAIL {label}: complete={comp}"); sys.exit(1)
if len(pt)!=exp: print(f"ARTIFACT-GATE FAIL {label}: {len(pt)} trials, expected {exp}"); sys.exit(1)
if any(r.get("protocol")!="v3" for r in pt): print(f"ARTIFACT-GATE FAIL {label}: non-v3 rows"); sys.exit(1)
a=sum(r["agent_subtasks"] for r in pt); p=sum(r["possible_subtasks"] for r in pt)
print(f"ARTIFACT-GATE PASS {label}: {len(pt)} trials, {a}/{p} = {100*a/p:.2f}%")
PY
}

run_cell() {  # $1=arm $2=ckptsha $3=out $4=modes $5=trials $6=trial_start $7=extra...
  local arm=$1 sha=$2 out=$3 modes=$4 tr=$5 tstart=$6; shift 6
  local ncases=10 nmodes; nmodes=$(echo "$modes" | wc -w)
  local expected=$(( ncases * tr * nmodes ))
  say "cell: $arm modes=[$modes] trials=$tr start=$tstart -> $(basename "$out")"
  "$REPO/scripts/robocerebra/run_eval_sharded.sh" --k "$K" --gpu "$GPU" --port "$PORT" --out "$out" \
      --modes $modes --trials "$tr" --trial-start "$tstart" \
      --arm "$arm" --ckpt-sha "$sha" --budget-steps 15000 --note v3_ladder "$@" \
      >> "$LOGS/$(basename "${out%.json}").run.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { say "RC-GATE FAIL $arm ($(basename "$out")): exit $rc"; return 1; }
  check_artifact "$out" "$expected" "$arm/$(basename "$out")" | tee -a "$LOGS/ladder.log"
  return "${PIPESTATUS[0]}"
}

run_arm() {  # $1=arm $2=kind $3=ckpt $4=config $5=memfrac $6=tag $7...=extra harness flags
  local arm=$1 kind=$2 ckpt=$3 cfg=$4 mem=$5 tag=$6; shift 6
  start_server "$kind" "$ckpt" "$cfg" "$mem" "$tag" || { say "ARM $arm SKIPPED (server)"; return 1; }
  local ok=0
  run_cell "$arm" "$tag" "$LOGS/v3_${tag}_6mode.json" "$ALL_MODES" "$TRIALS" 0 "$@" || ok=1
  run_cell "$arm" "$tag" "$LOGS/v3_${tag}_memtopup.json" "$MEM_MODES" "$TRIALS" "$TRIALS" "$@" || ok=1
  stop_server
  [ $ok -eq 0 ] && say "ARM $arm COMPLETE" || say "ARM $arm FINISHED WITH FAILURES"
  return $ok
}

say "=== v3 ladder starting (K=$K GPU=$GPU port=$PORT trials=$TRIALS) ==="
wait_for_idle_gpus

run_arm A0_base  plain "$WSM/ckpts/a0_probe/15000"        pi05_robocerebra_base           0.42 a0_base
run_arm A3_jepa  plain "$WSM/ckpts/a3_jepa/15000"         pi05_robocerebra_jepa           0.42 a3_jepa
run_arm A1_gdn_w8 omega "$WSM/ckpts/a1_gdn_w8/15000"      pi05_robocerebra_gdn_w8         0.55 a1_gdn_w8 \
        --wsm --encoder-sha "$ENCODER_SHA"
run_arm A2_gdn_w16_hd05 omega "$WSM/ckpts/a2_gdn_w16_hd05/15000" pi05_robocerebra_gdn_w16_hd05 0.55 a2_gdn_w16_hd05 \
        --wsm --encoder-sha "$ENCODER_SHA"
run_arm A4_ptrm  omega "$WSM/ckpts/a4_ptrm/15000"         pi05_robocerebra_ptrm           0.55 a4_ptrm \
        --wsm --encoder-sha "$ENCODER_SHA"

say "=== v3 ladder done ==="
