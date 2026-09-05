#!/bin/bash
# CFG guidance-scale sweep on nagababa for a Stage-S workspace arm (default s2 / interface cfg2).
# ONE guidance value per invocation: guidance is a TRACE-TIME constant baked into Pi0 before policy
# construction (serve_pi_05_wsm_cfg.py --guidance-scale), so each value needs its own server process.
# 4 GPUs = 4 task-sharded workers (server per GPU + K episode-shard clients each), identical layout to
# full_eval_nagababa.sh. Results are box-tier (NOT sealed/claimed); provenance = arm.json.
#
# Subset protocol: the sealed manifest is 100 eps/task and eval_pi_05 requires
# len(manifest_records(task)) == --num-trials, so a subset needs a DERIVED manifest. This script builds
# one deterministically (every-(100/TRIALS)-th record under the canonical episode_identity ordering —
# the same ordering primitive shard_episode_records uses), re-seals its manifest_sha256, and records the
# parent digest in derived_from. Every kept record's (task, episode_index, reset, seed) is byte-identical
# to the sealed manifest, so a subset run is directly comparable to the sealed 100-ep numbers.
#
# Env knobs: ARM CKPT GUIDANCE TRIALS K NUM_WORKERS TASKS RESULTS_DIR INTERFACE
# Usage:
#   GUIDANCE=1.5 bash scripts/ec2/sweep_guidance_nagababa.sh
#   GUIDANCE=1.0 TASKS=TurnOnMicrowave NUM_WORKERS=1 bash ... sweep_guidance_nagababa.sh   # smoke
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"; export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"

WORK=/data/work
OPENPI="$WORK/openpi"; WSMV2="$WORK/wsmv2"; SIMPY="$WORK/simenv/bin/python"
CODE_DIR="$WORK/internal_training"

ARM="${ARM:-s2}"
INTERFACE="${INTERFACE:-cfg2}"
CKPT="${CKPT:-$WORK/ckpts/s2_59999}"
TAP_CKPT="${TAP_CKPT:-$WORK/ckpts/tap_149999}"
ENC_CKPT="${ENC_CKPT:-$WORK/wsm_artifacts/encoder.pt}"
LANG_TABLE="${LANG_TABLE:-$WORK/wsm_artifacts/task_lang_table.npz}"
TASK_PROMPTS="${TASK_PROMPTS:-$WORK/wsm_artifacts/task_prompt_manifest.json}"
: "${GUIDANCE:?set GUIDANCE (CFG scale s)}"

TRIALS="${TRIALS:-25}"
K="${K:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_PORT="${BASE_PORT:-5900}"
SEED=20260723
REPLAN_STEPS=8            # Stage-S: WSM encoder stride is 8, EXEC_STEPS must match
TASK_SETS="atomic_seen,composite_seen,composite_unseen"
STATE_MODE=per_env_isolated_v1   # pinned for every workspace interface (entry line ~151)
CANONICAL_MANIFEST="$WORK/canonical_episode_manifest.json"
HELD_ROOT="$WORK/heldout_full"
SERVER_UP_TIMEOUT=1800    # timeout #1: server must bind within 30 min (12GB ckpt + tap + encoder load)
RUN_TIMEOUT="${RUN_TIMEOUT:-21600}"  # timeout #2: whole rollout phase capped at 6h/value

GTAG="$(echo "$GUIDANCE" | tr -d '.' | cut -c1-2)"
RESULTS_DIR="${RESULTS_DIR:-/data2/evals/gsweep_${ARM}_g${GTAG}_$(date -u +%m%d_%H%M)}"
LOGS="$RESULTS_DIR/logs"; mkdir -p "$LOGS"

[[ -d "$CKPT/params" ]] || { echo "FATAL $CKPT has no params/"; exit 21; }
[[ -d "$TAP_CKPT/params" ]] || { echo "FATAL tap $TAP_CKPT has no params/"; exit 21; }
[[ -s "$ENC_CKPT" ]] || { echo "FATAL missing encoder $ENC_CKPT"; exit 21; }
[[ -s "$LANG_TABLE" ]] || { echo "FATAL missing task lang table $LANG_TABLE"; exit 21; }
[[ -s "$TASK_PROMPTS" ]] || { echo "FATAL missing task prompt manifest $TASK_PROMPTS"; exit 21; }
[[ -f "$HELD_ROOT/.prep_done" ]] || { echo "FATAL heldout_full not prepped"; exit 26; }

# ---- derived subset manifest (idempotent; byte-identical records, re-sealed digest) ----
if [[ "$TRIALS" == "100" ]]; then
  EPISODE_MANIFEST="$CANONICAL_MANIFEST"
else
  EPISODE_MANIFEST="$WORK/subset${TRIALS}_episode_manifest.json"
  if [[ ! -s "$EPISODE_MANIFEST" ]]; then
    PYTHONPATH="$WSMV2" "$SIMPY" - "$CANONICAL_MANIFEST" "$EPISODE_MANIFEST" "$TRIALS" <<'PY'
import copy, hashlib, json, sys
from vla_training.eval.eval_manifest import (
    _canonical_bytes, _manifest_digest_payload, episode_identity,
    validate_episode_manifest, write_episode_manifest,
)
src_path, dst_path, trials = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = validate_episode_manifest(json.load(open(src_path)))
per_task = int(src["episodes_per_task"])
if per_task % trials:
    raise SystemExit(f"{per_task} episodes/task is not divisible by --num-trials {trials}")
stride = per_task // trials
by_task = {}
for record in src["episodes"]:
    by_task.setdefault(record["task"], []).append(record)
kept = []
for task in sorted(by_task):
    # SAME ordering primitive as shard_episode_records; every stride-th record spreads the subset
    # across the whole sealed episode range instead of biasing toward low episode_index.
    ordered = sorted(by_task[task], key=lambda r: episode_identity(r)[1:])
    if len(ordered) != per_task:
        raise SystemExit(f"{task} has {len(ordered)} episodes, expected {per_task}")
    picked = ordered[::stride][:trials]
    if len(picked) != trials:
        raise SystemExit(f"{task} yielded {len(picked)} picks, expected {trials}")
    kept.extend(picked)
out = copy.deepcopy(src)
out["episodes"] = kept
out["episodes_per_task"] = trials
for entry in out.get("selection", {}).get("per_task", {}).values():
    entry["selected"] = trials
out["derived_from"] = {
    "manifest_sha256": src["manifest_sha256"],
    "kind": "every_nth_subset_of_canonical_ordering",
    "stride": stride,
    "episodes_per_task": trials,
}
out.pop("manifest_sha256", None)
out["manifest_sha256"] = hashlib.sha256(_canonical_bytes(_manifest_digest_payload(out))).hexdigest()
validate_episode_manifest(out)
write_episode_manifest(dst_path, out)
print(f"[subset] wrote {dst_path}: {len(kept)} episodes, {trials}/task, "
      f"sha256={out['manifest_sha256']} (parent {src['manifest_sha256']})", flush=True)
PY
  fi
fi

echo "{\"arm\":\"$ARM\",\"interface\":\"$INTERFACE\",\"guidance\":$GUIDANCE,\"ckpt\":\"$CKPT\"," \
     "\"tap_ckpt\":\"$TAP_CKPT\",\"encoder\":\"$ENC_CKPT\",\"kind\":\"subset$((TRIALS * 50))\"," \
     "\"trials_per_task\":$TRIALS,\"episode_manifest\":\"$EPISODE_MANIFEST\",\"k\":$K," \
     "\"workers\":$NUM_WORKERS,\"state_mode\":\"$STATE_MODE\",\"replan_steps\":$REPLAN_STEPS," \
     "\"host\":\"nagababa-g7e\",\"seed\":$SEED,\"tier\":\"box\",\"sealed\":false}" \
  | tr -s ' ' > "$RESULTS_DIR/arm.json"

# ---- staging (mirror smoke_eval_nagababa.sh) ----
cp "$CODE_DIR/robocasa/wsm_robocasa_configs.py" "$OPENPI/src/openpi/training/"
cp "$CODE_DIR/robocasa/wsm_serve_rc.py" "$OPENPI/scripts/"
export OPENPI_DATA_HOME="$WORK/openpi_cache"
TOKCACHE="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"; mkdir -p "$(dirname "$TOKCACHE")"
[[ -s "$TOKCACHE" ]] || curl -fsSL --retry 3 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o "$TOKCACHE" || true
CONFIGS_DIR="$CODE_DIR/robocasa"   # the tap does `import wsm_robocasa_configs` off this dir

# ---- servers: one workspace server per GPU, guidance baked in ----
declare -a SERVER_PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  CUDA_VISIBLE_DEVICES=$w XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 WSM_SERVE_NO_DATA=1 \
  WSM_ENVS_PER_GPU=$K PI_WSM_SERVER_STATE_MODE=$STATE_MODE \
  OPENPI_DATA_HOME="$WORK/openpi_cache" PYTHONUNBUFFERED=1 PYTHONPATH="$WSMV2:$OPENPI/src" \
  "$OPENPI/.venv/bin/python" "$WSMV2/vla_training/eval/serve_pi_05_wsm_cfg.py" \
    --interface "$INTERFACE" --finetune-ckpt "$CKPT" --tap-ckpt "$TAP_CKPT" \
    --encoder-ckpt "$ENC_CKPT" --task-lang-table "$LANG_TABLE" \
    --guidance-scale "$GUIDANCE" --p-drop "${WSM_CFG_P_DROP:-0.2}" \
    --config-name pi05_robocasa_workspace_stage_s --configs-dir "$CONFIGS_DIR" \
    --k-window 1 --stride "$REPLAN_STEPS" --tap-prompt terse \
    --port $((BASE_PORT + w)) \
    >"$LOGS/server_$w.log" 2>&1 &
  SERVER_PIDS+=("$!")
done
trap 'kill "${SERVER_PIDS[@]}" 2>/dev/null || true' EXIT
for ((w=0; w<NUM_WORKERS; w++)); do
  up=0
  for i in $(seq 1 $((SERVER_UP_TIMEOUT / 10))); do
    "$SIMPY" -c "import socket;s=socket.socket();s.settimeout(2);s.connect((\"127.0.0.1\",$((BASE_PORT + w))));s.close()" 2>/dev/null && { up=1; break; }
    kill -0 "${SERVER_PIDS[$w]}" 2>/dev/null || { echo "FATAL server $w died:"; tail -40 "$LOGS/server_$w.log"; exit 22; }
    sleep 10
  done
  [[ "$up" -eq 1 ]] || { echo "FATAL server $w never bound in ${SERVER_UP_TIMEOUT}s"; tail -40 "$LOGS/server_$w.log"; exit 22; }
done
echo "[servers] $NUM_WORKERS up (base port $BASE_PORT, s=$GUIDANCE, interface=$INTERFACE)"
grep -h "serve-pi-workspace" "$LOGS/server_0.log" | tail -8 || true

# ---- K episode-shard clients per worker ----
cd "$WSMV2"
declare -a RUNNER_PIDS=()
for ((w=0; w<NUM_WORKERS; w++)); do
  for ((j=0; j<K; j++)); do
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$w PYTHONPATH="$WSMV2" \
    timeout "$RUN_TIMEOUT" \
    "$SIMPY" vla_training/eval/eval_pi_05.py \
      --config "$WSMV2/scripts/configs/eval/pi05_eval.yaml" \
      --worker-idx "$w" --num-workers "$NUM_WORKERS" \
      --host 127.0.0.1 --port $((BASE_PORT + w)) --out-dir "$RESULTS_DIR" \
      --task-sets "$TASK_SETS" --num-trials "$TRIALS" --video none --seed "$SEED" \
      ${TASKS:+--tasks "$TASKS"} \
      --episode-manifest "$EPISODE_MANIFEST" --heldout-root "$HELD_ROOT" --rollouts-per-demo 1 \
      --task-prompt-manifest "$TASK_PROMPTS" \
      --replan-steps "$REPLAN_STEPS" \
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

"$SIMPY" - "$RESULTS_DIR" "$GUIDANCE" <<'EOF'
import json, pathlib, sys
root, guidance = pathlib.Path(sys.argv[1]), sys.argv[2]
splits = {}
for f in sorted(root.rglob("stats_shard*.json")):
    d = json.load(open(f)); split, task = f.parent.parent.name, f.parent.name
    s = sum(1 for e in d["per_episode"] if e.get("success"))
    a, b = splits.setdefault(split, {}).setdefault(task, (0, 0)); splits[split][task] = (a + s, b + len(d["per_episode"]))
task_means, tot_s, tot_n = [], 0, 0
for split, tasks in sorted(splits.items()):
    ms = [s / n for s, n in tasks.values()]
    print(f"s={guidance} {split}: mean {sum(ms)/len(ms):.3f} over {len(ms)} tasks ({sum(n for _, n in tasks.values())} eps)")
    task_means += ms
    tot_s += sum(s for s, _ in tasks.values()); tot_n += sum(n for _, n in tasks.values())
if not task_means:
    print(f"s={guidance} NO SHARD STATS FOUND"); raise SystemExit(0)
print(f"GUIDANCE {guidance} avg_task_weighted = {sum(task_means)/len(task_means):.4f}  episodes={tot_n}")
json.dump({"guidance": float(guidance),
           "avg_task_weighted": sum(task_means)/len(task_means), "episodes": tot_n,
           "splits": {sp: {t: {"succ": s, "n": n} for t, (s, n) in ts.items()} for sp, ts in splits.items()}},
          open(root / "full_summary.json", "w"), indent=1)
EOF
[[ "$FAILS" -eq 0 ]] || { echo "GUIDANCE $GUIDANCE FAILED ($FAILS runners)"; exit 23; }
echo "GUIDANCE $GUIDANCE DONE -> $RESULTS_DIR"
