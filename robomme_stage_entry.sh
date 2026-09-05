#!/usr/bin/env bash
# ============================================================================================
# RoboMME Stage entry — pooled pi0.5 tap, Stage-E ω precompute, and the D7 serve-parity gate.
#
# The RoboMME sibling of `robocerebra_stage_entry.sh`, same phase contract so ONE approval can
# run a chain and resume:
#
#   tap      : JAX/openpi env. Frozen `pi05_on/149999` tap + frozen WSMv1 pool -> `wsm_pooled`
#              `p.npz` store for all 1,600 RoboMME episodes on the CROSS-DOMAIN grid
#              (stride 8 + final frame). This is the Stage-E ENCODER corpus; A3 compares it against
#              the other three taps. Fanned one shard per GPU.
#   tapserve : the SAME tap on the SERVE-ALIGNED grid (arange(0,D,8) U arange(D,n,16), no trailing
#              frame) -> the POLICY omega store. Required, not optional: a serve frame is
#              `exec_start_idx + 16k` and 16 == 0 (mod 8), so on the 81.4 % of demo episodes whose
#              `exec_start_idx % 8 != 0` NOT ONE live frame lands on the stride-8 grid and an
#              online producer would emit zero live omegas. Measured 2026-09-02 over all 900:
#              only 167 have `exec_start_idx % 8 == 0`.
#   omega  : torch env. A trained Stage-E encoder -> the per-episode ω store the policy arms read.
#   parity : torch env. The D7 expert-replay oracle: the SERVE-side incremental producer must
#            reproduce the ω store frame-exactly on held-out episodes.
#
# ORDERING (same as robocerebra, and just as easy to get backwards): the tap is an INPUT to
# Stage E, not an output.
#   tap  ->  (launch_stage_e.py, 4-domain retrain)  ->  omega  ->  parity  ->  M1/M2/M3
# `tap` therefore runs as soon as the dataset and checkpoints exist — it does NOT wait for labels.
#
# Required env (scripts/launch/submit_robomme_stage.py):
#   RMME_PHASES             tap[,tapserve][,omega][,parity]      (default: tap)
#   OPENPI_FORK_S3          content-addressed openpi tarball
#   RMME_DATA_S3            the sealed all16 LeRobot PREFIX (data/ meta/ assets/)
#   RMME_DATA_INVENTORY_S3 + _SHA256   the sealed parent inventory (per-object source_sha256)
#   TAP_CKPT_S3             pi05_on/149999 PREFIX (params/ + assets/ + _CHECKPOINT_METADATA)
#   POOL_CKPT_S3 + _SHA256  frozen WSMv1 pool checkpoint (wsm_step100000.pt)
#   WSM_CONFIGS_S3          internal_training/robocasa tarball registering pi05_rc_mg60_bal33
#   OUTPUT_S3               where the tap / ω / parity artifacts land
#   NUM_GPUS                default 8
# omega/parity only:
#   ENCODER_CKPT_URI        content-addressed Stage-E <sha>.pt — the FINAL encoder.pt that
#                           EXPORTED the store, never encoder_best.pt (h14 §41.2)
#   OMEGA_LANG_MODE         stored | demo | taskmean | per_frame | task_line
#                           (§39.3: `stored` is the GATE mode; `taskmean` is a diagnostic that
#                            FAILS a correct encoder and must never gate)
#   PARITY_DEMOS            held-out episodes to score (default 20)
# ============================================================================================
set -euo pipefail
trap 'rc=$?; echo "[entry] DIED at line $LINENO (rc=$rc)" >&2' ERR      # h14 §37.4
echo "======== RoboMME stage | $(hostname) | $(date) ========"

WORK="${WORK:-/opt/ml/work}"; mkdir -p "$WORK"
OUT="$WORK/out"; mkdir -p "$OUT"
PHASES="${RMME_PHASES:-tap}"
NG="${NUM_GPUS:-8}"
: "${OUTPUT_S3:?}"
ATTEMPT="${SM_CURRENT_ATTEMPT:-$(date -u +%Y%m%dT%H%M%SZ)}"             # per-attempt log prefix
has_phase () { [[ ",$PHASES," == *",$1,"* ]]; }

# ---- per-file S3 sync in the BACKGROUND, not at exit ------------------------------------------
# There is no guaranteed terminate on this queue, so a max_run timeout is a NORMAL exit path:
# anything that only ships in a final sync is lost.
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

stage_tar_sha () {  # $1=s3 uri  $2=dest dir  $3=expected sha ("" to skip)  $4=label
  local t="$WORK/$4.tar"
  aws s3 cp "$1" "$t" --only-show-errors
  if [[ -n "$3" ]]; then
    local got; got="$(sha_of "$t")"
    [[ "$got" == "$3" ]] || { echo "[entry] FATAL: $4 sha $got != $3"; exit 3; }
  fi
  mkdir -p "$2"
  if [[ "$1" == *.tgz || "$1" == *.tar.gz ]]; then tar -xzf "$t" -C "$2"; else tar -xf "$t" -C "$2"; fi
  rm -f "$t"
  echo "[entry] staged $4 ($(du -sh "$2" | cut -f1))${3:+, sha verified}"
}

# ---- upload one file to the per-attempt log prefix (h14 §34.6: replica logs were shadowed) -----
ship_log () { aws s3 cp "$1" "$OUTPUT_S3/logs/$ATTEMPT/$(basename "$1")" --only-show-errors || true; }

# ---- wait ONLY on the named client pids; never a bare `wait` (h14 §36.3/§37.1) -----------------
wait_clients () {   # $@ = pids ; logs are $WORK/<LOG_PREFIX>$i.log
  local rc=0 i=0 status
  for pid in "$@"; do
    status=0; wait "$pid" || status=$?            # $?-after-simple-command under set -e (§37.1)
    if [[ "$status" != 0 ]]; then
      echo "[entry] client $i (pid $pid) exited NON-ZERO ($status)" >&2
      rc="$status"
    fi
    i=$((i + 1))
  done
  for f in "$WORK/${LOG_PREFIX:?}"*.log; do [[ -f "$f" ]] || continue; ship_log "$f"; done
  if [[ "$rc" != 0 ]]; then
    for f in "$WORK/${LOG_PREFIX}"*.log; do
      [[ -f "$f" ]] || continue; echo "---- $(basename "$f") ----"; tail -n 25 "$f"
    done
  fi
  return "$rc"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSMV2="${WSM_REPO_DIR:-$SCRIPT_DIR}"
[ -d "$WSMV2/workspace_models" ] || { echo "[entry] FATAL: no workspace_models under $WSMV2"; exit 2; }
echo "[entry] wsmv2 tree = $WSMV2"
unset PYTHONPATH PYTHONHOME || true
nvidia-smi -L || true

# ============================================================================================
# PHASE tap — JAX/openpi env
# ============================================================================================
if has_phase tap || has_phase tapserve; then
  echo "---- phase: tap (grids: $(has_phase tap && printf 'wsm_pooled ')$(has_phase tapserve && printf 'serve_aligned')) ----"

  # -- dataset: the SEALED inventory, not a bare `s3 sync`. `fleet/inventory.py::materialize`
  #    hashes every object as it streams and fails BEFORE moving it into place on a
  #    `source SHA-256 mismatch` (CAMPAIGNS §W6: the path that would have caught the 54 corrupt
  #    local parquets).
  DATA="$WORK/robomme_data"
  MANIFEST="$WORK/robomme.parent.inventory.json"
  aws s3 cp "${RMME_DATA_INVENTORY_S3:?}" "$MANIFEST" --only-show-errors
  got="$(sha_of "$MANIFEST")"
  [[ "$got" == "${RMME_DATA_INVENTORY_SHA256:?}" ]] || {
    echo "[tap] FATAL: inventory sha $got != $RMME_DATA_INVENTORY_SHA256"; exit 3; }
  PYTHONPATH="$WSMV2/robomme_integration" python3 -m fleet.inventory \
    --manifest "$MANIFEST" --artifact robomme_lerobot_all16 \
    --root-s3 "${RMME_DATA_S3:?}" --destination "$DATA" --workers 48 \
    || { echo "[tap] FATAL: dataset materialization"; exit 3; }
  RMME_ROOT="$(cd "$(dirname "$(find "$DATA" -type f -path '*/meta/info.json' | head -1)")/.." && pwd)"
  n_par=$(find "$RMME_ROOT/data" -name '*.parquet' | wc -l)
  echo "[tap] root=$RMME_ROOT parquet=$n_par"
  [[ "$n_par" -eq 1600 ]] || { echo "[tap] FATAL: expected 1600 parquet, got $n_par"; exit 3; }

  # -- frozen tap checkpoint (a PREFIX: params/ + assets/ + _CHECKPOINT_METADATA)
  TAPCK="$WORK/pi05_on_149999"; mkdir -p "$TAPCK"
  aws s3 sync "${TAP_CKPT_S3:?}" "$TAPCK" --only-show-errors
  [[ -d "$TAPCK/params" ]] || { echo "[tap] FATAL: no params/ under $TAPCK"; exit 3; }
  echo "[tap] pi05_on/149999 at $TAPCK ($(du -sh "$TAPCK" | cut -f1))"

  # -- frozen WSMv1 pool
  POOL="$WORK/wsm_step100000.pt"
  aws s3 cp "${POOL_CKPT_S3:?}" "$POOL" --only-show-errors
  got="$(sha_of "$POOL")"
  [[ "$got" == "${POOL_CKPT_SHA256:?}" ]] || { echo "[tap] FATAL: pool sha $got"; exit 3; }

  # -- the openpi config package that registers pi05_rc_mg60_bal33 (WSM_CONFIGS_DIR)
  CONFIGS="$WORK/wsm_configs"
  stage_tar_sha "${WSM_CONFIGS_S3:?}" "$CONFIGS" "${WSM_CONFIGS_SHA256:-}" "wsm_configs"
  CONFIGS="$(cd "$(dirname "$(find "$CONFIGS" -maxdepth 3 -name 'wsm_robocasa_configs*' | head -1)")" && pwd)"
  echo "[tap] WSM_CONFIGS_DIR=$CONFIGS"

  # -- openpi env
  OPENPI="$WORK/openpi"; mkdir -p "$OPENPI"
  aws s3 cp "${OPENPI_FORK_S3:?}" "$WORK/openpi.tgz" --only-show-errors
  tar -xzf "$WORK/openpi.tgz" -C "$OPENPI"; rm -f "$WORK/openpi.tgz"
  [ -f "$OPENPI/pyproject.toml" ] || OPENPI="$(cd "$(dirname "$(find "$OPENPI" -maxdepth 3 -name pyproject.toml | head -1)")" && pwd)"
  cd "$OPENPI"; export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
  if [[ ! -d "$OPENPI/.venv" ]]; then
    uv sync || { echo "FATAL: uv sync"; exit 50; }
    # CLIENT DEPS, installed and then IMPORTED for real (h14 §37.2: the node venv had no
    # pandas/pyarrow/av and a shard died in one silent second). This tap needs pyarrow + pillow;
    # it never touches PyAV, because RoboMME has no MP4s.
    uv pip install --python "$OPENPI/.venv/bin/python" pyarrow pillow
    # ROBOCASA DEPS (2026-09-03, both rmme tap jobs FAILED "No module named 'robocasa'"): the config
    # package that registers pi05_rc_mg60_bal33 imports robocasa.utils.dataset_registry at import
    # time (wsm_robocasa_configs.py:20), so the node venv needs robocasa+robosuite exactly as the
    # proven RoboCerebra stage entry installs them (same fork, same pins, same install script).
    ROBOSUITE_SHA="${ROBOSUITE_SHA:-85abee228d1c43ab1939bce33028099945d453b4}"
    ROBOCASA_SHA="${ROBOCASA_SHA:-be22d659b02db8f6d7f3a3c3edc742934fdcbaae}"
    ( cd "$WORK" \
      && { [[ -d robosuite ]] || git clone -q https://github.com/ARISE-Initiative/robosuite.git; } \
      && git -C robosuite checkout -q "$ROBOSUITE_SHA" \
      && { [[ -d robocasa ]] || git clone -q https://github.com/robocasa/robocasa.git; } \
      && git -C robocasa checkout -q "$ROBOCASA_SHA" ) || { echo "FATAL: robocasa/robosuite clone"; exit 50; }
    ( cd "$OPENPI" && ROBOSUITE_DIR="$WORK/robosuite" ROBOCASA_DIR="$WORK/robocasa" \
        UV_PROJECT_ENVIRONMENT="$OPENPI/.venv" bash scripts/install_robocasa_deps.sh ) \
      || { echo "FATAL: robocasa deps install"; exit 50; }
    uv pip install --python "$OPENPI/.venv/bin/python" --no-deps PyOpenGL
  fi
  JPY="$OPENPI/.venv/bin/python"
  "$JPY" -c "import jax, pyarrow, PIL, numpy, torch; print('jax', jax.__version__, 'pyarrow', pyarrow.__version__, 'client deps OK')" \
    || { echo "FATAL: jax/client env verify"; exit 50; }
  # Import the exact module the tap imports, with the node's WSM_CONFIGS_DIR — fail here, not
  # after the 1600-parquet materialization and model build.
  WSM_CONFIGS_DIR="$CONFIGS" PYTHONPATH="$WSMV2" "$JPY" -c "import robocasa, wsm_robocasa_configs; print('robocasa + wsm_robocasa_configs import OK')" 2>/dev/null \
    || WSM_CONFIGS_DIR="$CONFIGS" PYTHONPATH="$WSMV2:$CONFIGS" "$JPY" -c "import robocasa, wsm_robocasa_configs; print('robocasa + wsm_robocasa_configs import OK (configs on path)')" \
    || { echo "FATAL: robocasa / wsm_robocasa_configs import"; exit 50; }

  # -- CPU PREFLIGHT with the node's exact argv: every data path, no model, nothing written.
  #    Costs seconds and turns a bad dataset/shard/enumeration into an instant, named failure.
  for g in wsm_pooled serve_aligned; do
    WSM_CONFIGS_DIR="$CONFIGS" PYTHONPATH="$WSMV2" "$JPY" -m workspace_models.features.rmme_pooled_tap \
      --dataset-root "$RMME_ROOT" --out-root "$OUT/wsm_pooled/_preflight" --grid "$g" \
      --worker-idx 0 --num-workers "$NG" --limit 2 --plan-only \
      || { echo "[tap] FATAL: plan-only preflight failed for grid=$g"; exit 4; }
  done

  run_tap_grid () {   # $1 = grid mode, $2 = store name
    local grid="$1" store="$2" out="$OUT/wsm_pooled/$2"
    mkdir -p "$out"
    LOG_PREFIX="tap_${grid}_gpu"
    local pids=()
    for i in $(seq 0 $((NG - 1))); do
      CUDA_VISIBLE_DEVICES="$i" XLA_PYTHON_CLIENT_PREALLOCATE=false \
      WSM_CONFIGS_DIR="$CONFIGS" PYTHONPATH="$WSMV2" "$JPY" -m workspace_models.features.rmme_pooled_tap \
        --dataset-root "$RMME_ROOT" --ckpt "$TAPCK" --pool-ckpt "$POOL" --grid "$grid" \
        --out-root "$out" --worker-idx "$i" --num-workers "$NG" \
        > "$WORK/${LOG_PREFIX}$i.log" 2>&1 &
      pids+=($!)
    done
    wait_clients "${pids[@]}" || { echo "FATAL: $grid tap shard failed"; return 5; }
    # POST-STAGE ASSERTION (h14 §38.3): clients not crashing is not proof they did the work.
    local n_done n_npz
    n_done=$(find "$out" -name .done_pooled | wc -l)
    n_npz=$(find "$out" -name p.npz | wc -l)
    echo "[tap:$grid] complete: $n_done done markers / $n_npz p.npz of 1600 episodes"
    [[ "$n_done" -eq 1600 && "$n_npz" -eq 1600 ]] || {
      echo "FATAL: $grid tap produced $n_done/$n_npz of 1600"; return 6; }
    return 0
  }

  TAP_OUT="$OUT/wsm_pooled/rmme_pi_100k"
  if has_phase tap;      then run_tap_grid wsm_pooled    rmme_pi_100k       || exit $?; fi
  if has_phase tapserve; then run_tap_grid serve_aligned rmme_pi_100k_serve || exit $?; fi
  echo "[tap] stores: encoder-corpus=$OUT/wsm_pooled/rmme_pi_100k  policy=$OUT/wsm_pooled/rmme_pi_100k_serve"

  # -- A3 token-statistics audit on the CROSS-DOMAIN grid only (the serve-aligned store is a
  #    policy artifact, not part of the Stage-E corpus, so it is not audited against the other taps).
  #    `train_stage_e` REFUSES a multi-domain cell without a raw-tap effective rank for every loaded
  #    tap (§24.3), so this is a required OUTPUT of the tap job: robomme's G1b bar is 0.8 x it.
  has_phase tap && PYTHONPATH="$WSMV2" "$JPY" "$WSMV2/scripts/deliberation/tap_stats_audit.py" \
    --tap "robomme=$TAP_OUT:p" --stratify-files --max-files 48 --seed 20260822 \
    --out "$OUT/tap_stats_robomme.json" \
    || echo "[tap] WARNING: A3 audit failed; rerun locally before any multi-domain Stage-E cell"
  [[ -s "$OUT/tap_stats_robomme.json" ]] || echo "[tap] WARNING: empty A3 audit artifact"
fi

# ============================================================================================
# PHASES omega / parity — torch env
# ============================================================================================
if has_phase omega || has_phase parity; then
  echo "---- torch env ----"
  TPY="${TORCH_PYTHON:-python}"
  "$TPY" -c "import torch;print('torch',torch.__version__,torch.cuda.device_count(),'gpus')" \
    || { echo "FATAL: torch env"; exit 51; }
  : "${ENCODER_CKPT_URI:?omega/parity need a Stage-E checkpoint}"
  case "$ENCODER_CKPT_URI" in
    *encoder_best.pt) echo "FATAL: encoder_best.pt is the BEST-EVAL step, not the checkpoint that "\
"exported the ω store; parity refuses a step mismatch (h14 §41.2). Pass encoder.pt."; exit 52;;
  esac
  ENC="$WORK/stage_e_encoder.pt"
  aws s3 cp "$ENCODER_CKPT_URI" "$ENC" --only-show-errors
  # The POLICY omega store is exported from the SERVE-ALIGNED tap; the cross-domain store is the
  # encoder's training corpus and must never be the thing a policy arm reads.
  TAP_OUT="${OMEGA_TAP_OVERRIDE:-$OUT/wsm_pooled/rmme_pi_100k_serve}"
  [[ -d "$TAP_OUT" ]] || {
    echo "FATAL: no serve-aligned tap store at $TAP_OUT — run the tapserve phase first"; exit 5; }
  LANG_MODE="${OMEGA_LANG_MODE:-stored}"
fi

if has_phase omega; then
  # ω is normally produced INSIDE the Stage-E job (`train_stage_e.py --export-omega`). This phase
  # exists only to re-export ω from an existing checkpoint without retraining.
  echo "---- phase: omega (lang mode: $LANG_MODE) ----"
  : "${STAGE_E_LABELS_DIR:?omega re-export needs the label artifact}"
  OMEGA_OUT="$OUT/omega/robomme"; mkdir -p "$OMEGA_OUT"
  PYTHONPATH="$WSMV2" "$TPY" -m workspace_models.train.train_wsm_base.train_stage_e \
    --labels "$STAGE_E_LABELS_DIR" --tap "robomme=$TAP_OUT" \
    --lang-mode "$LANG_MODE" --steps 0 --resume "$ENC" \
    --export-omega "$OMEGA_OUT" \
    || { echo "FATAL: omega export"; exit 7; }
  n_w=$(find "$OMEGA_OUT" -name 'w.npz' | wc -l)
  echo "[omega] $n_w episodes exported"
  [[ "$n_w" -gt 0 ]] || { echo "FATAL: omega export wrote zero episodes"; exit 8; }
fi

if has_phase parity; then
  echo "---- phase: parity (D7) ----"
  OMEGA_ROOT="${OMEGA_ROOT_OVERRIDE:-$OUT/omega/robomme}"
  [ -d "$OMEGA_ROOT" ] || { echo "FATAL: no ω store at $OMEGA_ROOT"; exit 5; }
  # 1) IDENTITY. `stored` compares against the conditioning vector the ω store ITSELF recorded,
  #    which is a true identity check for any one-vector-per-episode contract (h14 §39.3). It also
  #    refuses to run when the encoder's step differs from the store's `encoder_step` (§41.2).
  PYTHONPATH="$WSMV2" "$TPY" scripts/deliberation/stage_e_omega_parity.py \
    --domain robomme --encoder "$ENC" \
    --pooled-root "$TAP_OUT" --omega-root "$OMEGA_ROOT" \
    --lang-mode "$LANG_MODE" --demos "${PARITY_DEMOS:-20}" \
    --out "$OUT/parity_robomme_${LANG_MODE}.json" \
    || { echo "FATAL: D7 identity gate FAILED under lang-mode=$LANG_MODE — do NOT submit M1/M2/M3"; exit 9; }
  # 2) CONFOUND MEASUREMENT: the conventions Stage E did NOT train on. Reported, never gating.
  for alt in demo taskmean; do
    [ "$alt" = "$LANG_MODE" ] && continue
    PYTHONPATH="$WSMV2" "$TPY" scripts/deliberation/stage_e_omega_parity.py \
      --domain robomme --encoder "$ENC" \
      --pooled-root "$TAP_OUT" --omega-root "$OMEGA_ROOT" \
      --lang-mode "$alt" --demos "${PARITY_DEMOS:-20}" \
      --out "$OUT/parity_robomme_alt_${alt}.json" || true
  done
  echo "[parity] identity gate PASSED under lang-mode=$LANG_MODE; alternatives measured"
fi

echo "======== done: phases=$PHASES ========"
