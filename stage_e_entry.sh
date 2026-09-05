#!/usr/bin/env bash
# H14 Stage-E node entry — the 8-cell encoder funnel, one cell per GPU, ONE p5 job.
#
# Submitted by scripts/deliberation/launch_stage_e.py. Shape: eight independent single-GPU trainings
# fanned out on one node (no TP, no NCCL, no cross-GPU collectives) because A4's funnel is "screen
# MANY encoders on gates, graduate FEW to policy training" — the cells share only their inputs.
#
# Contract with the launcher (all env, all non-secret):
#   WSM_E_RUN_ID          content-addressed id; also the S3 leaf
#   WSM_E_S3_OUT          s3://.../stage_e/<run_id>
#   WSM_E_LABELS_S3       the content-addressed edge-label artifact to stage
#   WSM_E_TAPS            "robocasa=s3://...;remembench=s3://...;robomme=s3://..."  (staged in order)
#   WSM_E_CELLS           comma-separated cell specs, one per GPU. A spec is `cell` or `cell:seed`;
#                         the seeded form exists for the A14 replication, where the SAME cell runs
#                         at several seeds on one node and the paired-by-seed delta is the reading.
#   WSM_E_STEPS / _BATCH_EPISODES / _MIN_EDGES / _WARMUP / _EVAL_EVERY / _LR
#   WSM_E_LAMBDA_XDOM / _CONTRAST_WEIGHT / _LAMBDA_SIGREG / _SIGREG_RANK_CAP
#   WSM_E_LABEL_STORE_S3  RoboCasa pi-geometry keyframe labels (decode-grounding gate); may be empty
#   WSM_E_EXPORT_OMEGA    1 to write the ω store per cell and sync it
#
# Zero-files-fatal: every staged prefix must land at least one file, or the job dies before it can
# spend an hour training on an empty corpus.
set -euo pipefail
trap 'rc=$?; echo "[entry] DIED at line $LINENO (rc=$rc)" >&2' ERR

WORK="${WORK:-/opt/ml/input/data/work}"
[ -d "$WORK" ] || WORK=/tmp/stage_e
mkdir -p "$WORK"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 2026-09-04: run 5fe2556ba063477a died at the first cell with "python: command not found" — the node
# image has no bare `python` (h14 §33 landmine #1). Default to python3 and prove the trainer's
# imports BEFORE any data is staged, so a wrong interpreter fails in seconds, not after the pulls.
PY="${WSM_E_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "FATAL: interpreter '$PY' not found on the node"; exit 50; }
if ! "$PY" -c "import torch, numpy" >/dev/null 2>&1; then
  # 2026-09-04 (run 431067a32ee85c0b, exit 50): the digest-pinned dexjoco image ships ONLY
  # /usr/bin/python3.10 WITHOUT torch (inspected locally with docker; §39.4's "torch image, both
  # present" note was wrong). uv is on the image, so build a venv pinned to the versions the
  # RoboCerebra nodes resolve (torch 2.11.0+cu128, numpy 2.2.5). Takes a few minutes, once per job.
  command -v uv >/dev/null 2>&1 || { echo "FATAL: '$PY' lacks torch and uv is not on the node"; exit 50; }
  VENV="$WORK/venv"; export UV_CACHE_DIR="$WORK/uv_cache"
  echo "[entry] '$PY' lacks torch — building $VENV with uv"
  uv venv --python "$(command -v "$PY")" "$VENV" || { echo "FATAL: uv venv failed"; exit 50; }
  uv pip install --python "$VENV/bin/python" \
    "torch==${WSM_E_TORCH_VERSION:-2.11.0}" "numpy==${WSM_E_NUMPY_VERSION:-2.2.5}" \
    || { echo "FATAL: uv pip install torch/numpy failed"; exit 50; }
  PY="$VENV/bin/python"
fi
"$PY" -c "import torch, numpy; print('[entry] python', __import__('sys').version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)" \
  || { echo "FATAL: '$PY' lacks torch/numpy after bootstrap"; exit 50; }
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: torch sees no CUDA device (wheel/driver mismatch) — refusing to train on CPU"; exit 50; }
RUN_ID="${WSM_E_RUN_ID:?}"
S3_OUT="${WSM_E_S3_OUT:?}"
CELLS="${WSM_E_CELLS:?}"

echo "[entry] stage_e run_id=$RUN_ID cells=$CELLS work=$WORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

stage() {                       # stage <s3-uri> <dest> ; zero files is fatal
  local uri="$1" dest="$2"
  mkdir -p "$dest"
  echo "[stage] $uri -> $dest"
  aws s3 sync "$uri" "$dest" --only-show-errors
  local n
  n=$(find "$dest" -type f | wc -l)
  echo "[stage] $dest has $n files"
  if [ "$n" -eq 0 ]; then
    echo "[fatal] staging $uri produced ZERO files — refusing to train on an empty corpus" >&2
    exit 3
  fi
}

LABELS="$WORK/labels"
stage "${WSM_E_LABELS_S3:?}" "$LABELS"

TAP_ARGS=()
IFS=';' read -ra TAP_SPECS <<<"${WSM_E_TAPS:?}"
for spec in "${TAP_SPECS[@]}"; do
  [ -n "$spec" ] || continue
  name="${spec%%=*}"; uri="${spec#*=}"
  stage "$uri" "$WORK/taps/$name"
  TAP_ARGS+=(--tap "$name=$WORK/taps/$name")
done

# A18 SECOND WAVE. The 4-tap cells load one extra domain, so they need their own tap set. They run
# as a SEPARATE WAVE rather than alongside the 8 pre-registered cells, deliberately: the 8 were
# validated at one cell per GPU, and co-scheduling two cells on a GPU would change their memory and
# contention profile -- acquiring a confound in the pre-registered comparison for no reason. The
# extra wall time is ~30 min against a 21,600 s ceiling.
TAP4_ARGS=()
if [ -n "${WSM_E_TAPS_4TAP:-}" ]; then
  IFS=';' read -ra TAP4_SPECS <<<"$WSM_E_TAPS_4TAP"
  for spec in "${TAP4_SPECS[@]}"; do
    [ -n "$spec" ] || continue
    name="${spec%%=*}"; uri="${spec#*=}"
    stage "$uri" "$WORK/taps4/$name"
    TAP4_ARGS+=(--tap "$name=$WORK/taps4/$name")
  done
fi

LABEL_STORE_ARG=()
if [ -n "${WSM_E_LABEL_STORE_S3:-}" ]; then
  stage "$WSM_E_LABEL_STORE_S3" "$WORK/keyframe_labels"
  LABEL_STORE_ARG=(--label-store "$WORK/keyframe_labels")
fi

OUT="$WORK/runs"
mkdir -p "$OUT"

# ---- per-cell background syncer: push artifacts AS THEY LAND, never only at exit ---------------
sync_loop() {
  local cell="$1"
  while true; do
    for d in "$OUT"/"$cell"_*; do
      [ -d "$d" ] || continue
      # The ω export lives under <cell>/omega/<domain>/*.npz. Without it in THIS filter it ships
      # only in the final sync, so a preemption or a max_run timeout loses the one artifact the
      # policy arms actually consume -- and there is no terminate, so a timeout is the normal exit.
      aws s3 sync "$d" "$S3_OUT/cells/$(basename "$d")" \
        --exclude "*" --include "*.json" --include "encoder_best.pt" --include "encoder.pt" \
        --include "omega/*" --include "omega/*/*" \
        --only-show-errors || true
    done
    # The per-cell TRAINING LOGS live at $OUT/<tag>.log, i.e. OUTSIDE any <cell>_* dir, so the
    # filtered per-cell sync above never touched them and they shipped only in the final sync --
    # lost on a timeout or preemption, which is the normal exit here. Ship them every cycle.
    for l in "$OUT"/*.log; do
      [ -f "$l" ] || continue
      aws s3 cp "$l" "$S3_OUT/cells/_logs/$(basename "$l")" --only-show-errors || true
    done
    sleep 120
  done
}

# Serve-consistent conditioning (§27): optional per-domain task-lang tables, staged like the taps.
TASK_LANG_ARGS=()
if [ -n "${WSM_E_TASK_LANG_TABLES:-}" ]; then
  IFS=';' read -ra TL_SPECS <<<"$WSM_E_TASK_LANG_TABLES"
  for spec in "${TL_SPECS[@]}"; do
    [ -n "$spec" ] || continue
    dom="${spec%%=*}"; uri="${spec#*=}"
    dest="$WORK/task_lang_$dom.npz"
    aws s3 cp "$uri" "$dest" --only-show-errors
    [ -s "$dest" ] || { echo "FATAL: empty task-lang table for $dom from $uri"; exit 3; }
    TASK_LANG_ARGS+=(--task-lang-table "$dom=$dest")
    echo "[entry] task-lang table $dom <- $uri"
  done
fi

# Per-domain G1b bars (§32.3). Shipped as a file rather than baked into the module so the SEALED
# defaults (rmb 5.90) stay the module's answer and sealed cells are never silently re-judged; a
# NEW multi-domain cell opts in to the stratified bars explicitly.
if [ -n "${WSM_E_RAW_TAP_ERANK_S3:-}" ]; then
  aws s3 cp "$WSM_E_RAW_TAP_ERANK_S3" "$WORK/raw_tap_erank.json" --only-show-errors
  [ -s "$WORK/raw_tap_erank.json" ] || { echo "FATAL: empty raw-tap erank json"; exit 3; }
  export WSM_RAW_TAP_ERANK_JSON="$WORK/raw_tap_erank.json"
  echo "[entry] per-domain G1b bars from $WSM_E_RAW_TAP_ERANK_S3: $(cat "$WORK/raw_tap_erank.json" | tr -d '\n')"
fi

IFS=',' read -ra CELL_LIST <<<"$CELLS"

# Wave 1 = every cell whose name does not end in -4tap; wave 2 = the rest.
WAVE1=(); WAVE2=()
for spec in "${CELL_LIST[@]}"; do
  case "${spec%%:*}" in *-4tap) WAVE2+=("$spec");; *) WAVE1+=("$spec");; esac
done
echo "[entry] wave1 (${#WAVE1[@]} cells, 3-tap): ${WAVE1[*]:-none}"
echo "[entry] wave2 (${#WAVE2[@]} cells, 4-tap): ${WAVE2[*]:-none}"
if [ "${#WAVE2[@]}" -gt 0 ] && [ "${#TAP4_ARGS[@]}" -eq 0 ]; then
  echo "[entry] FATAL: -4tap cells requested but WSM_E_TAPS_4TAP is empty" >&2; exit 3
fi

STATUS=0
run_wave () {                     # $1 = wave label, $2 = 3tap|4tap, rest = cell specs
  local label="$1" which="$2"; shift 2
  local -a CELLS_IN=("$@")
  [ "${#CELLS_IN[@]}" -gt 0 ] || return 0
  local -a WAVE_TAPS
  if [ "$which" = "4tap" ]; then WAVE_TAPS=("${TAP4_ARGS[@]}"); else WAVE_TAPS=("${TAP_ARGS[@]}"); fi
  echo "[entry] === $label: ${#CELLS_IN[@]} cells, taps: ${WAVE_TAPS[*]} ==="
  local GPU=0
  local -a PIDS=()
  for spec in "${CELLS_IN[@]}"; do
  cell="${spec%%:*}"
  SEED_ARG=()
  tag="$cell"
  if [ "$spec" != "$cell" ]; then
    SEED_ARG=(--seed "${spec#*:}")
    tag="${cell}_s${spec#*:}"
  fi
  log="$OUT/${tag}.log"
  echo "[launch] cell=$cell seed=${spec#*:} gpu=$GPU -> $log"
  (
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" "$PY" -u \
      "$REPO/workspace_models/train/train_wsm_base/train_stage_e.py" \
      --labels "$LABELS" "${WAVE_TAPS[@]}" "${LABEL_STORE_ARG[@]}" \
      --cell "$cell" "${SEED_ARG[@]}" --out "$OUT" \
      --steps "${WSM_E_STEPS:?}" \
      --batch-episodes "${WSM_E_BATCH_EPISODES:?}" \
      --min-edges-per-batch "${WSM_E_MIN_EDGES:?}" \
      --warmup "${WSM_E_WARMUP:?}" \
      --eval-every "${WSM_E_EVAL_EVERY:?}" \
      --lr "${WSM_E_LR:?}" \
      --lambda-sigreg "${WSM_E_LAMBDA_SIGREG:?}" \
      --sigreg-rank-cap "${WSM_E_SIGREG_RANK_CAP:?}" \
      --lambda-xdom "${WSM_E_LAMBDA_XDOM:?}" \
      --contrast-weight "${WSM_E_CONTRAST_WEIGHT:?}" \
      ${WSM_E_LANG_MODE:+--lang-mode "$WSM_E_LANG_MODE"} \
      "${TASK_LANG_ARGS[@]+"${TASK_LANG_ARGS[@]}"}" \
      ${WSM_E_EXPORT_OMEGA:+--export-omega "$OUT/omega/$tag"} \
      --device cuda:0
  ) >"$log" 2>&1 &
  PIDS+=($!)
  # One syncer per CELL NAME, not per spec: its glob is "$OUT/<cell>_*", which already covers every
  # seed of that cell, so a second copy would only duplicate uploads.
  case " ${SYNCED:-} " in *" $cell "*) : ;; *) sync_loop "$cell" & SYNCED="${SYNCED:-} $cell";; esac
  GPU=$((GPU + 1))
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
  done
  echo "[entry] === $label complete (status=$STATUS) ==="
}

run_wave "wave1 (pre-registered 3-tap)" 3tap "${WAVE1[@]+"${WAVE1[@]}"}"
run_wave "wave2 (A18 4-tap)"            4tap "${WAVE2[@]+"${WAVE2[@]}"}"

# Cells that exit 0 having produced nothing are the silent-success class: assert each cell left an
# encoder AND a gates json before this job may report success.
MISSING=""
for spec in "${CELL_LIST[@]}"; do
  cell="${spec%%:*}"; tag="$cell"
  [ "$spec" != "$cell" ] && tag="${cell}_s${spec#*:}"
  # The trainer names its run dir <cell>_<encoder_id> (no seed suffix); match on the cell and pick the
  # dir whose run_config.json carries this seed. 2026-09-05: run 772597789979f88a trained all 10
  # cells and then FAILED here because the old pattern "<cell>_s<seed>_*" matched nothing.
  seed="${spec#*:}"; [ "$spec" = "$cell" ] && seed=""
  d=""
  for cand in $(find "$OUT" -maxdepth 1 -type d -name "${cell}_*"); do
    if [ -z "$seed" ] || grep -q "\"seed\": *${seed}\b" "$cand/run_config.json" 2>/dev/null; then d="$cand"; break; fi
  done
  if [ -z "$d" ]; then MISSING="$MISSING $tag(no-run-dir)"; continue; fi
  [ -n "$(find "$d" -name 'encoder*.pt' -print -quit)" ] || MISSING="$MISSING $tag(no-encoder)"
  [ -n "$(find "$d" -name '*.json'      -print -quit)" ] || MISSING="$MISSING $tag(no-json)"
done
if [ -n "$MISSING" ]; then
  echo "[entry] FATAL: cells produced no output:$MISSING" >&2
  STATUS=1
fi
echo "[entry] all cells finished (status=$STATUS); final sync"
aws s3 sync "$OUT" "$S3_OUT/cells" --only-show-errors || STATUS=1
exit "$STATUS"
