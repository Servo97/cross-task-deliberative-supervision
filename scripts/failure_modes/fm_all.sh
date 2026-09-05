#!/bin/bash
# Full fan-out for one benchmark. Every stage is content-validated resumable, so re-running
# after an interruption re-does only what is missing.
#
# Order is chosen so the numbers land before the pixels: rollouts and teacher-forcing first
# (they hold the policy servers), then the server-free renders. fm_sync.sh can be run from
# the workstation at any point — it pulls metrics immediately and videos incrementally.
#
#   BENCH=remembench bash fm_all.sh
set -uo pipefail
cd /data/work/wsmv2
FM=/data2/failure_modes
: "${BENCH:?set BENCH}"
NUM_GPUS="${NUM_GPUS:-4}"

if [[ "$BENCH" == "remembench" ]]; then
  CKPTS=(pretrain150k base jepa_k16 dnw8 dnw16_drop)
  PY=/data/work/remembench_env/bin/python
else
  CKPTS=(pretrain150k base jepa_k1 dnw8)
  PY=/data/work/simenv/bin/python
fi

stage() { echo "=== $* :: $(date -u +%FT%TZ) ==="; }

# 1. expert probe -- server-free, once per benchmark; every arm differences against it
stage "expert probe"
for TASK in $($PY -c "
import json;m=json.load(open('$FM/manifests/fm_${BENCH}_manifest.json'));print(' '.join(m['tasks']))"); do
  for ((w = 0; w < NUM_GPUS; w++)); do
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$w CUDA_VISIBLE_DEVICES=$w \
    PYTHONPATH=/data/work/wsmv2 "$PY" scripts/failure_modes/fm_rollout.py \
      --manifest "$FM/manifests/fm_${BENCH}_manifest.json" --bench "$BENCH" --task "$TASK" \
      --ckpt-label expert --out-root "$FM" --mode expert_probe \
      --shard-idx "$w" --num-shards "$NUM_GPUS" \
      >"$FM/logs/expert_${BENCH}_${TASK}_w$w.log" 2>&1 &
  done
  wait
  echo "[expert] $TASK done"
done

# 2. per-arm closed-loop rollouts, then the teacher-forced pass on the same servers
for CK in "${CKPTS[@]}"; do
  for MODE in rollout teacher_force; do
    stage "$BENCH / $CK / $MODE"
    BENCH="$BENCH" CKPT_LABEL="$CK" MODE="$MODE" NUM_GPUS="$NUM_GPUS" \
      bash scripts/failure_modes/fm_run.sh 2>&1 | grep -vE "^\[robosuite|WARN:" | tail -12
  done
done

# 3. videos -- server-free
for CK in expert "${CKPTS[@]}"; do
  stage "render $BENCH / $CK"
  BENCH="$BENCH" CKPT_LABEL="$CK" NUM_GPUS="$NUM_GPUS" \
    bash scripts/failure_modes/fm_render.sh 2>&1 | grep -E "RENDER DONE|\[render\] .* done" | tail -6
done

# 4. metrics over every arm of this benchmark (plus the other benchmark's manifest if it has
#    already run, so master.csv is always the complete picture)
stage "metrics"
MAN=(--manifest "$FM/manifests/fm_${BENCH}_manifest.json")
OTHER=remembench; [[ "$BENCH" == remembench ]] && OTHER=robocasa
[[ -f "$FM/manifests/fm_${OTHER}_manifest.json" ]] && \
  MAN+=(--manifest "$FM/manifests/fm_${OTHER}_manifest.json")
ALL=$(IFS=,; echo "pretrain150k,base,jepa_k16,jepa_k1,dnw8,dnw16_drop")
/data/work/remembench_env/bin/python scripts/failure_modes/fm_metrics.py \
  "${MAN[@]}" --root "$FM" --video-root "$FM/videos" --out-root "$FM/out" \
  --ckpts "$ALL" --csv-name master.csv 2>&1 | tail -30

/data/work/remembench_env/bin/python scripts/failure_modes/fm_readme.py \
  --out "$FM/out/README.md" --csv "$FM/out/master.csv"

echo "FM_ALL_DONE $BENCH $(date -u +%FT%TZ)"
