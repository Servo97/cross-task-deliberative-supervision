#!/usr/bin/env bash
# A0-long budget curve under protocol v3: the under-training discriminator.
#
# Three SAME-RUN checkpoints of a0_base-5a2b7e82 ({15k, 30k, 45k} = 4.23 / 8.46 / 12.7 epochs at
# bs 256 over 907,875 frames). Ideal only -- the curve needs one mode, and Ideal is the cell with
# the published comparator. Deterministic seeding makes the three budgets CRN-paired by
# construction: identical (mode, case, trial) coordinates => identical env inits across budgets,
# so the per-case contrast is variance-reduced the same way the arm ladder was.
set -uo pipefail
WSM=${WSM_DATA:-/home/sarveshp/Research/TRI/wsm_data/robocerebra}
REPO=${WSM_REPO:-/home/sarveshp/Research/TRI/wsmv2}
OPENPI=${OPENPI_ROOT:-/home/sarveshp/Research/robocasa_openpi}
LOGS=$WSM/logs_v3
K=${K:-8}; GPU=${GPU:-1}; PORT=${PORT:-8020}; TRIALS=${TRIALS:-10}
mkdir -p "$LOGS"
say(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOGS/budget_curve.log"; }

for STEP in 15000 30000 44999; do
  CKPT=$WSM/ckpts/a0_long/$STEP
  [ -d "$CKPT/params" ] || { say "MISSING $CKPT/params -- skipping"; continue; }
  OUT=$LOGS/v3_a0long_${STEP}_Ideal.json
  if [ -f "$OUT" ]; then say "already done: $(basename "$OUT")"; continue; fi
  pkill -f "serve_pi05_libero.py.*--port $PORT" 2>/dev/null; sleep 6
  ( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$GPU WSM_ENVS_PER_GPU=$K \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH=/home/sarveshp/Research/robocasa:/home/sarveshp/Research/robosuite \
      .venv/bin/python "$REPO/scripts/robocerebra/serve_pi05_libero.py" \
        --checkpoint "$CKPT" --config pi05_robocerebra_base --port $PORT \
        > "$LOGS/server_a0long_$STEP.log" 2>&1 ) &
  up=0
  for _ in $(seq 1 90); do
    grep -q "server listening" "$LOGS/server_a0long_$STEP.log" 2>/dev/null && { up=1; break; }
    sleep 5
  done
  [ $up -eq 1 ] || { say "SERVER FAILED at step $STEP"; continue; }
  say "server up: a0_long/$STEP"
  "$REPO/scripts/robocerebra/run_eval_sharded.sh" --k $K --gpu $GPU --port $PORT --out "$OUT" \
      --modes Ideal --trials $TRIALS --arm "A0long_$STEP" --ckpt-sha "a0_base-5a2b7e82@$STEP" \
      --budget-steps "$STEP" --note v3_budget_curve >> "$LOGS/budget_curve.run.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { say "RC-GATE FAIL step $STEP (exit $rc)"; continue; }
  /usr/bin/python3 - "$OUT" "$((10*TRIALS))" "$STEP" <<'PY' | tee -a "$LOGS/budget_curve.log"
import json,sys
p,exp,step=sys.argv[1],int(sys.argv[2]),sys.argv[3]
d=json.load(open(p)); pt=d.get("per_trial",[])
if d.get("complete") is not True or len(pt)!=exp:
    print(f"ARTIFACT-GATE FAIL {step}: complete={d.get('complete')} trials={len(pt)}/{exp}"); sys.exit(1)
a=sum(r["agent_subtasks"] for r in pt); q=sum(r["possible_subtasks"] for r in pt)
print(f"ARTIFACT-GATE PASS a0_long@{step}: {len(pt)} trials, {a}/{q} = {100*a/q:.2f}%")
PY
done
pkill -f "serve_pi05_libero.py.*--port $PORT" 2>/dev/null
say "=== budget curve done ==="
