#!/usr/bin/env bash
# Run the released official π0.5 and FrameSamp+Modul checkpoints sequentially on all 800 episodes.
set -euo pipefail

ROOT="${ROBOMME_OFFICIAL_ROOT:-$HOME/Research/TRI/robomme_eval/official_reference}"
REPO="$ROOT/robomme_policy_learning"
BENCH_REPO="$ROOT/robomme_benchmark"
WSMV2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVALUATOR="$WSMV2_ROOT/robomme_integration/eval/official_reference_eval.py"
POLICY_PY="$REPO/.venv/bin/python"
SIM_ROOT="$HOME/Research/TRI/robomme_eval/runtime-v0.4.0"
SIM_PY="$SIM_ROOT/env-v0.4.0/bin/python"
OUT="$ROOT/results"
PORT="${ROBOMME_OFFICIAL_PORT:-18700}"
POLICY_COMMIT="ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
BENCHMARK_COMMIT="856bc3a189d4172f3f47dbee4424d585f8d78db3"
MANISKILL_COMMIT="07be6fbc66350ddca200abfb0a11b692f078f7fd"
BASELINE_SHA="b8a9e9d78e4336e04582a01767a59ee132ea1e17295c7cf98b4952500de178e5"
FRAMESAMP_SHA="2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
MANISKILL_ROOT="$SIM_ROOT/ManiSkill-07be6fbc"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export MPLCONFIGDIR="$ROOT/matplotlib"
mkdir -p "$OUT" "$MPLCONFIGDIR"

eval_pythonpath="${REPO}/examples/robomme:${REPO}/packages/openpi-client/src:${BENCH_REPO}/src:${MANISKILL_ROOT}"
ACTIVE_SERVER_PID=""

cleanup_server() {
  if [[ -n "$ACTIVE_SERVER_PID" ]]; then
    kill "$ACTIVE_SERVER_PID" 2>/dev/null || true
    wait "$ACTIVE_SERVER_PID" 2>/dev/null || true
    ACTIVE_SERVER_PID=""
  fi
}
trap cleanup_server EXIT INT TERM

wait_for_common_staging() {
  while true; do
    local policy_ready=0 benchmark_ready=0 env_ready=0
    [[ "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" == "$POLICY_COMMIT" ]] && policy_ready=1
    [[ "$(git -C "$BENCH_REPO" rev-parse HEAD 2>/dev/null || true)" == "$BENCHMARK_COMMIT" ]] && benchmark_ready=1
    if [[ -x "$POLICY_PY" ]] && \
      JAX_PLATFORMS=cpu "$POLICY_PY" -c 'import jax, mme_vla_suite, pytest; from mme_vla_suite.policies import policy' >/dev/null 2>&1; then
      env_ready=1
    fi
    if (( policy_ready && benchmark_ready && env_ready )); then
      return
    fi
    printf 'WAITING policy_source=%d benchmark_source=%d env=%d %s\n' \
      "$policy_ready" "$benchmark_ready" "$env_ready" "$(date -u +%FT%TZ)"
    sleep 30
  done
}

wait_for_checkpoint() {
  local root="$1" sha="$2" label="$3"
  while [[ ! -f "$root/.EXTRACTED-$sha" ]]; do
    echo "WAITING_CHECKPOINT method=$label marker=.EXTRACTED-$sha $(date -u +%FT%TZ)"
    sleep 30
  done
}

checkpoint_dir() {
  local root="$1"
  mapfile -t params < <(find "$root" -type d -name params -print)
  [[ "${#params[@]}" == 1 ]] || {
    echo "expected exactly one params tree under $root; found ${#params[@]}" >&2
    exit 3
  }
  local value="${params[0]%/params}"
  [[ -d "$value/assets" ]] || { echo "assets missing under $value" >&2; exit 4; }
  echo "$value"
}

wait_for_server() {
  local pid="$1"
  for _ in $(seq 1 900); do
    kill -0 "$pid" 2>/dev/null || { echo "policy server exited before readiness" >&2; return 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/healthz" | grep -qx 'OK'; then
      return
    fi
    sleep 1
  done
  echo "policy server did not bind within 15 minutes" >&2
  return 1
}

wait_for_resources() {
  while true; do
    local compute port_free
    if ! compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
      echo "WAITING_FOR_NVIDIA_SMI $(date -u +%FT%TZ)"
      sleep 30
      continue
    fi
    compute="$(printf '%s\n' "$compute" | sed '/^[[:space:]]*$/d')"
    if "$SIM_PY" - "$PORT" <<'PY' >/dev/null 2>&1
import socket, sys
probe = socket.socket()
try:
    probe.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    probe.close()
PY
    then
      port_free=1
    else
      port_free=0
    fi
    if [[ -z "$compute" && "$port_free" == 1 ]]; then
      return
    fi
    echo "WAITING_FOR_LOCAL_GPU_OR_PORT compute_pids=${compute:-none} port_free=$port_free $(date -u +%FT%TZ)"
    sleep 30
  done
}

validate_scorecard() {
  local scorecard="$1" method="$2" sha="$3"
  "$SIM_PY" - "$scorecard" "$method" "$sha" "$POLICY_COMMIT" "$BENCHMARK_COMMIT" "$MANISKILL_COMMIT" <<'PY'
import json, pathlib, sys

path, method, sha, policy_commit, benchmark_commit, maniskill_commit = sys.argv[1:]
value = json.loads(pathlib.Path(path).read_text())
expected = {
    "method": method,
    "checkpoint_sha256": sha,
    "policy_source_commit": policy_commit,
    "benchmark_source_commit": benchmark_commit,
    "maniskill_source_commit": maniskill_commit,
    "episodes": 800,
    "result_scale": "fraction_0_1",
    "model_seed": 7,
    "action_horizon": 20,
    "execution_horizon": 16,
}
bad = {key: (value.get(key), wanted) for key, wanted in expected.items() if value.get(key) != wanted}
if bad:
    raise SystemExit(f"scorecard contract mismatch: {bad}")
if len(value.get("task_success_rate", {})) != 16:
    raise SystemExit("scorecard does not contain all 16 task rates")
print(f"OFFICIAL_SCORECARD_VALID method={method} successes={value['successes']}/800")
PY
}

run_method() {
  local method="$1" config="$2" checkpoint="$3" use_history="$4" checkpoint_sha="$5"
  local method_out="$OUT/$method"
  local server_log="$method_out/server.log"
  local eval_log="$method_out/eval.log"
  local attempt_log="$method_out/eval.current-attempt.log"
  local marker="$method_out/OFFICIAL_FIXED800_COMPLETE"
  local renderer_restarts=0
  local max_renderer_restarts="${ROBOMME_OFFICIAL_MAX_RENDERER_RESTARTS:-64}"
  mkdir -p "$method_out"
  if [[ -f "$marker" ]]; then
    validate_scorecard "$method_out/evaluation/scorecard.json" "$method" "$checkpoint_sha"
    echo "SKIP_OFFICIAL_REFERENCE_COMPLETE method=$method"
    return
  fi

  wait_for_resources

  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
      "$POLICY_PY" scripts/serve_policy.py --seed=7 --port="$PORT" \
      policy:checkpoint --policy.dir="$checkpoint" --policy.config="$config"
  ) >"$server_log" 2>&1 &
  local server_pid=$!
  ACTIVE_SERVER_PID="$server_pid"
  wait_for_server "$server_pid"

  local history_flag=()
  [[ "$use_history" == 1 ]] && history_flag=(--use-history)
  while true; do
    local eval_rc=0
    printf '\nOFFICIAL_EVAL_ATTEMPT method=%s renderer_restart=%d utc=%s\n' \
      "$method" "$renderer_restarts" "$(date -u +%FT%TZ)" >"$attempt_log"
    if (
      cd "$REPO/examples/robomme"
      CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$eval_pythonpath" \
        "$SIM_PY" "$EVALUATOR" --port "$PORT" --method "$method" \
        --output "$method_out/evaluation" \
        --checkpoint-sha256 "$checkpoint_sha" \
        --policy-source-commit "$POLICY_COMMIT" \
        --benchmark-source-commit "$BENCHMARK_COMMIT" \
        --maniskill-source-commit "$MANISKILL_COMMIT" \
        "${history_flag[@]}"
    ) >>"$attempt_log" 2>&1; then
      cat "$attempt_log" >>"$eval_log"
      rm -f "$attempt_log"
      break
    else
      eval_rc=$?
    fi
    cat "$attempt_log" >>"$eval_log"
    if ! grep -q 'vk::createInstanceUnique: ErrorIncompatibleDriver' "$attempt_log"; then
      echo "OFFICIAL_EVAL_ABORT method=$method rc=$eval_rc reason=non_renderer_failure" >&2
      return "$eval_rc"
    fi
    renderer_restarts=$((renderer_restarts + 1))
    if (( renderer_restarts > max_renderer_restarts )); then
      echo "OFFICIAL_EVAL_ABORT method=$method rc=$eval_rc reason=renderer_restart_limit" >&2
      return "$eval_rc"
    fi
    echo "OFFICIAL_EVAL_RENDERER_RESTART method=$method count=$renderer_restarts/$max_renderer_restarts" \
      | tee -a "$eval_log"
    until nvidia-smi -L >/dev/null 2>&1; do
      echo "WAITING_FOR_NVIDIA_DRIVER method=$method $(date -u +%FT%TZ)" | tee -a "$eval_log"
      sleep 30
    done
    # SAPIEN/svulkan occasionally leaves a process-local renderer in a bad state after many
    # create/destroy cycles.  A fresh simulator process resumes from atomically sealed Booleans;
    # the policy process and scientific contract remain unchanged.
    sleep 5
  done

  cleanup_server
  validate_scorecard "$method_out/evaluation/scorecard.json" "$method" "$checkpoint_sha"
  printf '%s\n' "$(date -u +%FT%TZ)" > "$marker"
  echo "OFFICIAL_FIXED800_COMPLETE method=$method checkpoint=$checkpoint"
}

[[ -x "$SIM_PY" ]] || { echo "RoboMME simulator Python missing: $SIM_PY" >&2; exit 10; }
[[ -d "$MANISKILL_ROOT/mani_skill" ]] || { echo "pinned ManiSkill source missing: $MANISKILL_ROOT" >&2; exit 11; }
grep -q "$MANISKILL_COMMIT" "$BENCH_REPO/pyproject.toml" || {
  echo "benchmark source does not pin the expected ManiSkill commit" >&2
  exit 12
}
(
  cd "$REPO/examples/robomme"
  PYTHONPATH="$eval_pythonpath" "$SIM_PY" -c 'from env_runner import EnvRunner; import robomme, mani_skill'
) || {
    echo "exact paper runtime import preflight failed" >&2
    exit 13
  }

wait_for_common_staging
wait_for_checkpoint "$ROOT/checkpoints/pi05_baseline" "$BASELINE_SHA" pi05_baseline
baseline="$(checkpoint_dir "$ROOT/checkpoints/pi05_baseline")"
run_method pi05_baseline pi05_baseline "$baseline" 0 "$BASELINE_SHA"

wait_for_checkpoint "$ROOT/checkpoints/perceptual-framesamp-modul" "$FRAMESAMP_SHA" perceptual-framesamp-modul
framesamp="$(checkpoint_dir "$ROOT/checkpoints/perceptual-framesamp-modul")"
run_method perceptual-framesamp-modul mme_vla_suite "$framesamp" 1 "$FRAMESAMP_SHA"
echo "OFFICIAL_REFERENCE_CAMPAIGN_COMPLETE root=$OUT"
