#!/usr/bin/env bash
# RoboCerebra (LIBERO, 2-view) salient-patch label pilot — the exact 5 stages, timed per case.
#
# Same pipeline as RoboCasa365 (workspace_models/labels/), with three differences, all injected by
# flag rather than forked: `--geom pi_libero` (2 views / 128 patch ids, matching the omega tap's
# base->wrist concatenation), a RoboCerebra-specific stage A0 that emits ONE frames npz per
# GROUND-TRUTH SUBTASK SEGMENT, and the 2-view Qwen prompt carried by the geometry module.
#
# Two envs by necessity: the segment table needs pyarrow (ogpo2) and the video decode needs av
# (vlm_labeler); no local env has both, and neither may install anything.
#
#   scripts/labels/robocerebra_label_pilot.sh <OUT_DIR> [EPISODES] [GPU] [LEROBOT_DIR]
#
# Per-stage wall clock is appended to <OUT_DIR>/_timings.tsv (case, stage, seconds) — that file is
# the input to the full-pass cost projection.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: robocerebra_label_pilot.sh <OUT_DIR> [EPISODES] [GPU] [LEROBOT_DIR]}"
EPISODES="${2:-89,875,946}"
GPU="${3:-1}"
LEROBOT="${4:-$HOME/Research/TRI/wsm_data/robocerebra/lerobot_home/wsmv2/robocerebra_train}"
STRIDE="${STRIDE:-4}"
MAX_FRAMES="${MAX_FRAMES:-14}"

NUMPY_PY="${NUMPY_PY:-$HOME/miniconda3/envs/ogpo2/bin/python}"   # numpy + pyarrow (no av)
VLM_PY="${VLM_PY:-$HOME/Research/envs/vlm_labeler/bin/python}"    # torch + transformers + av

mkdir -p "$OUT/_logs"
TIMINGS="$OUT/_timings.tsv"
[ -f "$TIMINGS" ] || printf 'case\tstage\tseconds\n' > "$TIMINGS"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"   # every stage is pinned here; --device is then always cuda:0

run() {  # run <case> <stage> <cmd...>
  local case="$1" stage="$2"; shift 2
  local log="$OUT/_logs/${case}.${stage}.log" t0 t1
  echo "[pilot] $case / $stage ..."
  t0=$(date +%s.%N)
  "$@" > "$log" 2>&1 || { echo "[pilot] FAIL $case/$stage — see $log"; tail -20 "$log"; exit 1; }
  t1=$(date +%s.%N)
  awk -v c="$case" -v s="$stage" -v a="$t0" -v b="$t1" 'BEGIN{printf "%s\t%s\t%.1f\n", c, s, b-a}' \
    >> "$TIMINGS"
  tail -3 "$log"
}

# --- A0: segment manifests (pyarrow) then the PyAV decode, both for all episodes at once ---
run all segments "$NUMPY_PY" -m workspace_models.labels.extract_frames_robocerebra \
  --lerobot-dir "$LEROBOT" --episodes "$EPISODES" --out "$OUT" --no-decode
run all extract "$VLM_PY" -m workspace_models.labels.extract_frames_robocerebra \
  --lerobot-dir "$LEROBOT" --episodes "$EPISODES" --out "$OUT" --stride "$STRIDE" \
  --segments-json "$OUT/_segments"

# --- A/B/C per case (one case = one episode dir = <scene>_<case>_ep<idx>) ---
for ep in ${EPISODES//,/ }; do
  CASE="$("$NUMPY_PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['case_tag'])" \
          "$OUT/_segments/$(printf 'ep%06d.json' "$ep")")"
  run "$CASE" qwen "$VLM_PY" -m workspace_models.labels.qwen_subgoals \
    --task "$CASE" --in "$OUT" --geom pi_libero --device cuda:0 --max-frames "$MAX_FRAMES"
  run "$CASE" molmo "$VLM_PY" -m workspace_models.labels.molmo_points \
    --task "$CASE" --in "$OUT" --geom pi_libero --device cuda:0
  run "$CASE" build "$VLM_PY" -m workspace_models.labels.build_salient_sets \
    --task "$CASE" --in "$OUT" --geom pi_libero --qc-dir "$OUT/_qc"
done

echo "[pilot] done — timings: $TIMINGS"
