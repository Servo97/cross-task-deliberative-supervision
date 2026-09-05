#!/usr/bin/env bash
# Upload the WSM feature cache + salient labels + node-path manifest to S3 for SageMaker WSM training.
# Uploads ONLY what training needs: per-demo patch_tokens.npy + feats.npz (cache) and the salient
# vlm_episode_*.npz (labels). Skips the big ep*_frames.npz (~40G, caching-only) and the DexJoCo junk.
#   screen -dmS wsm_upload bash scripts/launch/upload_cache_to_s3.sh
set -uo pipefail
S3=s3://sagemaker-us-west-2-124224456861/sarvesh.patil/wsm_robocasa
# Parameterized by backbone: groot -> wsm_cache, pi -> wsm_cache_pi (env-overridable). Labels are
# SHARED (both vlm_episode_*.npz and vlm_episode_pi_*.npz live under wsm_vlm_rc -> S3/wsm_labels).
CACHE="${WSM_CACHE_DIR:-$HOME/Research/TRI/wsm_data/wsm_cache}"
SUB="${WSM_CACHE_S3SUB:-wsm_cache}"
LABELS="$HOME/Research/TRI/wsm_data/wsm_vlm_rc"

echo "[upload] $(date) — $CACHE -> $S3/$SUB ; manifest + salient labels (small) first"
aws s3 cp "$CACHE/manifest_node.parquet" "$S3/$SUB/manifest_node.parquet" --only-show-errors
aws s3 sync "$LABELS" "$S3/wsm_labels/" --exclude "*" --include "*/vlm_episode_*.npz" --only-show-errors
echo "[upload] labels done; counting -> $(aws s3 ls --recursive "$S3/wsm_labels/" | grep -c vlm_episode)"

echo "[upload] cache (fp16) — this is the long pole ..."
aws s3 sync "$CACHE" "$S3/$SUB/" \
  --exclude "click_mouse/*" --exclude "fold_glasses/*" --exclude "hammer_nail/*" \
  --exclude "pick_bucket/*" --exclude "pinch_tongs/*" --exclude "water_plant/*" \
  --exclude "_feat_logs/*" --exclude "_logs/*" --exclude "manifest.parquet" --exclude "manifest_node.parquet" \
  --only-show-errors
echo "[upload] DONE $(date)"
echo "[upload] cache objects: $(aws s3 ls --recursive "$S3/$SUB/" | grep -c patch_tokens.npy) patch_tokens.npy"
