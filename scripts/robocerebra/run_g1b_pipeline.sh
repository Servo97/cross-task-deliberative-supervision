#!/usr/bin/env bash
# G1b end-to-end: finish the tap, retrain the encoder label-free, judge it against the
# PRE-REGISTERED bar, and (only on PASS) precompute ω over all episodes with a local manifest.
#
# Launch DETACHED so it outlives any interactive session:
#   setsid nohup scripts/robocerebra/run_g1b_pipeline.sh > <log> 2>&1 < /dev/null &
# Session-tied background jobs die with the session; that already cost one stalled tap run.
#
# Every stage is idempotent (--skip-existing / existence checks), so re-running after an
# interruption resumes rather than recomputes.
set -uo pipefail

WSM=/home/sarveshp/Research/TRI/wsm_data/robocerebra
REPO=/home/sarveshp/Research/TRI/wsmv2
OPENPI=/home/sarveshp/Research/robocasa_openpi
PY_TRAIN=/home/sarveshp/miniconda3/envs/ogpo2/bin/python
CKPT=$WSM/openpi_assets/openpi-assets/checkpoints/pi05_libero
TAP=$WSM/omega_tap_full
STATE=$WSM/g1b_pipeline_state
mkdir -p "$STATE"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# ---------------------------------------------------------------- stage 1: finish the tap
log "stage 1: tap (have $(ls "$TAP" 2>/dev/null | wc -l)/994)"
for pass in 1 2 3; do
  have=$(ls "$TAP" 2>/dev/null | wc -l)
  [ "$have" -ge 994 ] && break
  # Integrity, not just existence: a tap killed mid-write leaves a truncated npz that `os.path
  # .exists` happily accepts and `np.load` later dies on (BadZipFile). Validate and delete, so
  # the same pass that finds gaps also repairs corruption.
  MISSING=$("$PY_TRAIN" - <<PYEOF
import numpy as np, pathlib
root = pathlib.Path("$TAP")
have = set()
for p in sorted(root.glob("episode_*.npz")):
    try:
        d = np.load(p)
        if d["tokens"].shape[1:] != (128, 2048):
            raise ValueError("bad shape")
        for k in ("pooled_img", "pooled_lang", "frame_idx", "subtask_index"):
            _ = d[k].shape
        have.add(int(p.stem.split("_")[1]))
    except Exception:
        p.unlink(missing_ok=True)
print(" ".join(str(i) for i in range(994) if i not in have))
PYEOF
)
  [ -z "$MISSING" ] && break
  log "tap pass $pass: $(echo "$MISSING" | wc -w) episodes remaining"
  for g in 0 1; do
    SUB=$("$PY_TRAIN" -c "
eps='''$MISSING'''.split()
print(' '.join(eps[i] for i in range($g, len(eps), 2)))")
    [ -z "$SUB" ] && continue
    ( cd "$OPENPI" && CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
      PYTHONPATH=/home/sarveshp/Research/robocasa:/home/sarveshp/Research/robosuite \
      .venv/bin/python "$REPO/scripts/robocerebra/omega_tap.py" \
        --dataset "$WSM/lerobot_home/wsmv2/robocerebra_train" --checkpoint "$CKPT" \
        --episodes $SUB --frames-per-ep 64 --batch-size 16 --out "$TAP" \
        > "$STATE/tap_g$g.pass$pass.log" 2>&1 ) &
  done
  wait
done
log "stage 1 done: $(ls "$TAP" | wc -l)/994 episodes tapped"

# ---------------------------------------------------------------- stage 2: retrain
# 8k steps, batch 8 episodes: ~72 passes over the ~890-episode train split. Warm-started from a
# converged canonical encoder, so this is adapting an existing representation, not learning one
# from scratch (the canonical 50-100k-step budgets were scratch runs on RoboCasa365).
# --predict-k 6: the canary showed k=1 is trivially satisfiable on 64 subsampled frames
# (ω_{t+1}≈ω_t), which let SIGReg whiten away the temporal structure. k=6 sampled steps is ~84
# raw frames (~4 s), forcing ω to carry slow, episode-level content.
if [ ! -f "$STATE/train.done" ]; then
  log "stage 2: retrain-lite"
  CUDA_VISIBLE_DEVICES=0 "$PY_TRAIN" "$REPO/scripts/robocerebra/train_omega_retrain_lite.py" \
    --tap "$TAP" --init-from "$WSM/omega_encoder/stage_s_0883c9bd_encoder.pt" \
    --out "$WSM/omega_retrain_full" --steps 8000 --batch-episodes 8 --predict-k 6 \
    --lambda-sigreg 0.05 --eval-every 500 --heldout-frac 0.1 \
    > "$STATE/train.log" 2>&1 && touch "$STATE/train.done"
  log "stage 2 exit=$?"
fi

# ---------------------------------------------------------------- stage 3: verdict vs the bar
log "stage 3: verdict"
"$PY_TRAIN" "$REPO/scripts/robocerebra/g1b_verdict.py" \
  --history "$WSM/omega_retrain_full/history.json" \
  --best "$WSM/omega_retrain_full/encoder_best.pt" \
  --out "$WSM/g1b_verdict.json" > "$STATE/verdict.log" 2>&1
VERDICT=$("$PY_TRAIN" -c "import json;print(json.load(open('$WSM/g1b_verdict.json'))['verdict'])" 2>/dev/null || echo ERROR)
log "verdict: $VERDICT"

# ---------------------------------------------------------------- stage 4: ω precompute on PASS
if [ "$VERDICT" = "PASS" ]; then
  log "stage 4: omega precompute over all episodes"
  CUDA_VISIBLE_DEVICES=0 "$PY_TRAIN" "$REPO/scripts/robocerebra/precompute_omega.py" \
    --tap "$TAP" --encoder "$WSM/omega_retrain_full/encoder_best.pt" \
    --out "$WSM/omega_features" --manifests "$WSM/manifests" \
    > "$STATE/precompute.log" 2>&1
  log "stage 4 exit=$?"
else
  log "not PASS -> stopping before omega precompute, as pre-registered"
fi
touch "$STATE/PIPELINE_DONE"
log "pipeline finished (verdict=$VERDICT)"
