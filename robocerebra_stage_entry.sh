#!/usr/bin/env bash
# ============================================================================================
# RoboCerebra Stage entry — pooled tap, Stage-E ω precompute, and the D7 serve-parity gate.
#
# Phases behind RCB_PHASES so ONE approval can run a chain and resume (the
# robocasa_stage_s_features_entry.sh pattern):
#
#   tap    : JAX/openpi env. Frozen pi05_libero tap + frozen WSMv1 pool -> wsm_pooled `p.npz`
#            store for all 994 RoboCerebra training episodes, fanned one shard per GPU.
#   omega  : torch env. A trained Stage-E encoder -> the per-episode ω store the policy arms read.
#   parity : torch env. The D7 expert-replay oracle (§25.2): the SERVE-side incremental producer
#            must reproduce the ω store frame-exactly on held-out demos, and the serve-time
#            language gap (§25.3) is MEASURED here rather than assumed away.
#
# ORDERING, because it is easy to get backwards: the tap is an INPUT to Stage E, not an output.
#   tap  ->  (launch_stage_e.py, 4-domain retrain)  ->  omega  ->  parity  ->  R1/R2
# `tap` therefore runs as soon as the dataset and checkpoints exist — it does NOT wait for labels.
# `omega`/`parity` need ENCODER_CKPT_URI and run after the Stage-E job lands.
#
# Required env (submit_robocerebra_stage.py):
#   RCB_PHASES              tap[,omega][,parity]           (default: tap)
#   OPENPI_FORK_S3          content-addressed openpi tarball (wsmv2 ships as the source bundle)
#   RCB_DATA_S3 + _SHA256   sealed robocerebra_train_v1 LeRobot tarball
#   TAP_CKPT_S3             released pi05_libero (params+assets) tarball
#   POOL_CKPT_S3 + _SHA256  frozen WSMv1 pool checkpoint (wsm_step100000.pt)
#   OUTPUT_S3               where the tap / ω / parity artifacts land
#   NUM_GPUS                default 8
# omega/parity only:
#   ENCODER_CKPT_URI        content-addressed Stage-E <sha>.pt
#   OMEGA_LANG_MODE         per_frame | task_line | episode_mean   (the §25.3 conditioning contract)
#   PARITY_DEMOS            held-out demos to score (default 20)
# ============================================================================================
set -euo pipefail
echo "======== RoboCerebra stage | $(hostname) | $(date) ========"

WORK="${WORK:-/opt/ml/work}"; mkdir -p "$WORK"
OUT="$WORK/out"; mkdir -p "$OUT"
PHASES="${RCB_PHASES:-tap}"
NG="${NUM_GPUS:-8}"
: "${OUTPUT_S3:?}"
has_phase () { [[ ",$PHASES," == *",$1,"* ]]; }

# ---- per-file S3 sync in the BACKGROUND, not at exit ------------------------------------------
# There is no terminate on this queue, so a max_run timeout is a NORMAL exit path: anything that
# only ships in a final sync is lost. (This is defect #3 from the stage_e syncer, not repeated.)
( while true; do aws s3 sync "$OUT" "$OUTPUT_S3" --only-show-errors || true; sleep 120; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; aws s3 sync "$OUT" "$OUTPUT_S3" --only-show-errors || true' EXIT

# ---- resume: pull whatever a previous attempt already produced --------------------------------
aws s3 sync "$OUTPUT_S3" "$OUT" --only-show-errors || true
echo "[entry] resumed $(find "$OUT" -type f | wc -l) files from $OUTPUT_S3"

sha_of () { python3 -c "
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for b in iter(lambda: f.read(1<<20), b''): h.update(b)
print(h.hexdigest())" "$1"; }

stage_tar_sha () {  # $1=s3 uri  $2=dest dir  $3=expected sha  $4=label
  local t="$WORK/$4.tar"
  aws s3 cp "$1" "$t" --only-show-errors
  local got; got="$(sha_of "$t")"
  [[ "$got" == "$3" ]] || { echo "[entry] FATAL: $4 sha $got != $3"; exit 3; }
  mkdir -p "$2"; tar -xf "$t" -C "$2"; rm -f "$t"
  echo "[entry] staged $4 ($(du -sh "$2" | cut -f1)), sha verified"
}

# ---- code ------------------------------------------------------------------------------------
# The entry ships INSIDE the sanitized wsmv2 source bundle (it lives at the repo root and is the
# SAGEMAKER_PROGRAM), so the tree is already on the node -- downloading a second content-addressed
# copy would be a redundant artifact that could disagree with the code actually running.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSMV2="${WSM_REPO_DIR:-$SCRIPT_DIR}"
[ -d "$WSMV2/workspace_models" ] || { echo "[entry] FATAL: no workspace_models under $WSMV2"; exit 2; }
echo "[entry] wsmv2 tree = $WSMV2"
OPENPI="$WORK/openpi"; mkdir -p "$OPENPI"
aws s3 cp "${OPENPI_FORK_S3:?}" "$WORK/openpi.tgz" --only-show-errors && tar -xzf "$WORK/openpi.tgz" -C "$OPENPI"
# openpi tarballs sometimes carry a top-level dir; point at the tree that actually holds pyproject.
[ -f "$OPENPI/pyproject.toml" ] || OPENPI="$(cd "$(dirname "$(find "$OPENPI" -maxdepth 3 -name pyproject.toml | head -1)")" && pwd)"

ROBOSUITE_SHA="${ROBOSUITE_SHA:-85abee228d1c43ab1939bce33028099945d453b4}"
ROBOCASA_SHA="${ROBOCASA_SHA:-be22d659b02db8f6d7f3a3c3edc742934fdcbaae}"
unset PYTHONPATH PYTHONHOME || true
nvidia-smi -L || true

# ============================================================================================
# PHASE tap — JAX/openpi env
# ============================================================================================
if has_phase tap; then
  echo "---- phase: tap ----"
  DATA="$WORK/robocerebra"
  stage_tar_sha "${RCB_DATA_S3:?}" "$DATA" "${RCB_DATA_SHA256:?}" "rcb_data"
  RCB_ROOT="$(cd "$(dirname "$(find "$DATA" -type f -path '*/meta/info.json' | head -1)")/.." && pwd)"
  n_par=$(find "$RCB_ROOT/data" -name '*.parquet' | wc -l)
  n_mp4=$(find "$RCB_ROOT/videos" -name '*.mp4' | wc -l)
  echo "[tap] root=$RCB_ROOT parquet=$n_par mp4=$n_mp4"
  [[ "$n_par" -eq 994 && "$n_mp4" -eq 1988 ]] || {
    echo "[tap] FATAL: expected 994 parquet / 1988 mp4, got $n_par / $n_mp4"; exit 3; }

  TAPCK="$WORK/pi05_libero"
  aws s3 cp "${TAP_CKPT_S3:?}" "$WORK/init.tar" --only-show-errors
  mkdir -p "$TAPCK"; tar -xf "$WORK/init.tar" -C "$TAPCK"; rm -f "$WORK/init.tar"
  # The tar may or may not carry a top-level dir; find the tree that actually holds params/.
  TAPCK="$(cd "$(dirname "$(find "$TAPCK" -maxdepth 3 -type d -name params | head -1)")" && pwd)"
  echo "[tap] pi05_libero at $TAPCK ($(du -sh "$TAPCK" | cut -f1))"

  POOL="$WORK/wsm_step100000.pt"
  aws s3 cp "${POOL_CKPT_S3:?}" "$POOL" --only-show-errors
  got="$(sha_of "$POOL")"
  [[ "$got" == "${POOL_CKPT_SHA256:?}" ]] || { echo "[tap] FATAL: pool sha $got"; exit 3; }

  cd "$WORK"
  [[ -d robosuite ]] || git clone -q https://github.com/ARISE-Initiative/robosuite.git
  git -C robosuite checkout -q "$ROBOSUITE_SHA"
  [[ -d robocasa ]] || git clone -q https://github.com/robocasa/robocasa.git
  git -C robocasa checkout -q "$ROBOCASA_SHA"
  cd "$OPENPI"; export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
  if [[ ! -d "$OPENPI/.venv" ]]; then
    uv sync || { echo "FATAL: uv sync"; exit 50; }
    ROBOSUITE_DIR="$WORK/robosuite" ROBOCASA_DIR="$WORK/robocasa" \
      UV_PROJECT_ENVIRONMENT="$OPENPI/.venv" bash scripts/install_robocasa_deps.sh \
      || { echo "FATAL: jax env install"; exit 50; }
    uv pip install --python "$OPENPI/.venv/bin/python" --no-deps PyOpenGL
  fi
  JPY="$OPENPI/.venv/bin/python"
  "$JPY" -c "import jax,av,pandas; print('jax',jax.__version__,'av+pandas OK')" \
    || { echo "FATAL: jax env verify"; exit 50; }

  TAP_OUT="$OUT/wsm_pooled/rcb_pi_libero"; mkdir -p "$TAP_OUT"
  TAP_PIDS=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES="$i" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH="$WSMV2" "$JPY" -m workspace_models.features.rcb_pooled_tap \
      --dataset-root "$RCB_ROOT" --ckpt "$TAPCK" --pool-ckpt "$POOL" \
      --out-root "$TAP_OUT" --worker-idx "$i" --num-workers "$NG" \
      > "$WORK/tap_gpu$i.log" 2>&1 &
    TAP_PIDS+=($!)
  done
  # Wait on the TAP pids explicitly. `jobs -p` would also return the background S3 syncer, which
  # never exits -- waiting on it hangs the job until max_run kills it.
  rc=0; for pid in "${TAP_PIDS[@]}"; do wait "$pid" || rc=$?; done
  [[ "$rc" == 0 ]] || { for i in $(seq 0 $((NG-1))); do echo "--gpu$i--"; tail -n5 "$WORK/tap_gpu$i.log"; done
                        echo "FATAL: tap shard failed (rc=$rc)"; exit "$rc"; }
  n_done=$(find "$TAP_OUT" -name .done_pooled | wc -l)
  echo "[tap] complete: $n_done / 994 episodes"
  [[ "$n_done" -eq 994 ]] || { echo "FATAL: tap produced $n_done of 994"; exit 4; }

  # A3 token-statistics audit against the other two taps -- the measurement train_stage_e now
  # REFUSES to run a multi-domain cell without (WSM_RAW_TAP_ERANK_JSON, §24.3).
  PYTHONPATH="$WSMV2" "$JPY" "$WSMV2/scripts/deliberation/tap_stats_audit.py" \
    --tap "robocerebra=$TAP_OUT:p" --out "$OUT/tap_stats_robocerebra.json" || true
fi

# ============================================================================================
# PHASES omega / parity — torch env
# ============================================================================================
if has_phase omega || has_phase parity; then
  echo "---- torch env ----"
  TPY="${TORCH_PYTHON:-python3}"  # the node image has no bare `python` (h14 §33 / Stage-E 09-04)
  "$TPY" -c "import torch;print('torch',torch.__version__,torch.cuda.device_count(),'gpus')" \
    || { echo "FATAL: torch env"; exit 51; }
  : "${ENCODER_CKPT_URI:?omega/parity need a Stage-E checkpoint}"
  ENC="$WORK/stage_e_encoder.pt"
  aws s3 cp "$ENCODER_CKPT_URI" "$ENC" --only-show-errors
  TAP_OUT="$OUT/wsm_pooled/rcb_pi_libero"
  [[ -d "$TAP_OUT" ]] || { echo "FATAL: no tap store at $TAP_OUT — run the tap phase first"; exit 5; }
  LANG_MODE="${OMEGA_LANG_MODE:-per_frame}"
fi

if has_phase omega; then
  # NOTE: ω is normally produced INSIDE the Stage-E job (`train_stage_e.py --export-omega`), which
  # is what launch_stage_e.py already runs -- so this phase exists only to re-export ω from an
  # existing checkpoint without retraining. It re-runs the trainer in export-only mode.
  echo "---- phase: omega (lang mode: $LANG_MODE) ----"
  : "${STAGE_E_LABELS_DIR:?omega re-export needs the label artifact}"
  OMEGA_OUT="$OUT/omega/robocerebra"; mkdir -p "$OMEGA_OUT"
  PYTHONPATH="$WSMV2" "$TPY" -m workspace_models.train.train_wsm_base.train_stage_e \
    --labels "$STAGE_E_LABELS_DIR" --tap "robocerebra=$TAP_OUT" \
    --lang-mode "$LANG_MODE" --steps 0 --resume "$ENC" \
    --export-omega "$OMEGA_OUT" \
    || { echo "FATAL: omega export"; exit 6; }
  echo "[omega] $(find "$OMEGA_OUT" -name 'w.npz' | wc -l) episodes exported"
fi

if has_phase parity; then
  echo "---- phase: parity (D7) ----"
  OMEGA_ROOT="${OMEGA_ROOT_OVERRIDE:-$OUT/omega/robocerebra}"
  [ -d "$OMEGA_ROOT" ] || { echo "FATAL: no ω store at $OMEGA_ROOT"; exit 5; }
  # 1) IDENTITY: the serve-side producer must reproduce the shipped ω frame-exactly under the
  #    conditioning Stage-E actually trained on. This is the D7 bar (§25.2) and it GATES R1/R2.
  PYTHONPATH="$WSMV2" "$TPY" scripts/deliberation/stage_e_omega_parity.py \
    --domain robocerebra --encoder "$ENC" \
    --pooled-root "$TAP_OUT" --omega-root "$OMEGA_ROOT" \
    --lang-mode "$LANG_MODE" --demos "${PARITY_DEMOS:-20}" \
    --out "$OUT/parity_robocerebra_${LANG_MODE}.json" \
    || { echo "FATAL: D7 identity gate FAILED under lang-mode=$LANG_MODE — do NOT submit R1/R2"; exit 7; }
  # 2) CONFOUND MEASUREMENT (§25.3): score the conditioning conventions Stage-E did NOT train on.
  #    On the rmb lane this is where a serve convention was found to cripple the CONTROL arm alone
  #    and manufacture the primary contrast. Reported, never gating -- a failure here is a design
  #    decision for the coordinator, not a node error.
  # `stored` is the cross-check: it reads the conditioning vector the ω store itself records, so it
  # must agree with per_frame for a robocerebra cell. `demo`/`task_line` are the non-served
  # conventions, measured for the §25.3 table.
  for alt in stored demo task_line; do
    m="$alt"
    [ "$m" = "$LANG_MODE" ] && continue
    PYTHONPATH="$WSMV2" "$TPY" scripts/deliberation/stage_e_omega_parity.py \
      --domain robocerebra --encoder "$ENC" \
      --pooled-root "$TAP_OUT" --omega-root "$OMEGA_ROOT" \
      --lang-mode "$m" --demos "${PARITY_DEMOS:-20}" \
      --out "$OUT/parity_robocerebra_alt_${m}.json" || true
  done
  echo "[parity] identity gate PASSED under lang-mode=$LANG_MODE; alternatives measured"
fi

echo "======== done: phases=$PHASES ========"
