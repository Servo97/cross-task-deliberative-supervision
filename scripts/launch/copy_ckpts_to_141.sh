#!/usr/bin/env bash
# Copy the 2 GR00T pretrain-150k checkpoints (INFERENCE files only — no optimizer/rng/scheduler,
# ~12.8 GB/arm vs ~26 GB) from the 124 SSO account into a dedicated bucket in the 141 robotics
# account, so the evals can run on the cam-robotics queue.
#
# RUN THIS YOURSELF (the agent is gated from cross-account egress of model data):
#   bash scripts/launch/copy_ckpts_to_141.sh /path/to/141-creds.env
# where 141-creds.env exports AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN for
# account 141701954645. Stage A reads 124 with your DEFAULT SSO creds; Stage B writes 141 with
# the creds file. Re-runnable (aws s3 sync is incremental).
set -euo pipefail

RC_CREDS="${1:?usage: $0 /path/to/141-creds.env}"
SRC=s3://sagemaker-us-west-2-124224456861/sarvesh.patil/wsm_robocasa/pretrain150k/groot
DST_BUCKET=sagemaker-us-west-2-141701954645
DST=s3://$DST_BUCKET/sarvesh.patil/wsm_robocasa/pretrain150k/groot
SCRATCH=${SCRATCH:-/home/sarveshp/wsm_xfer}
ARMS=("mg60_off/groot-mg60-off" "mg60_bal33/groot-mg60-bal33")
EXCL=(--exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt" --exclude "trainer_state.json")

echo "=== STAGE A: download inference files from 124 (default SSO creds) -> $SCRATCH ==="
for a in "${ARMS[@]}"; do
  env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
    aws s3 sync "$SRC/$a/checkpoint-150000/" "$SCRATCH/$a/checkpoint-150000/" "${EXCL[@]}" --region us-west-2
done

echo "=== STAGE B: create 141 bucket + upload (creds from $RC_CREDS) ==="
( set -a; source "$RC_CREDS"; set +a
  aws s3api create-bucket --bucket "$DST_BUCKET" --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2 2>/dev/null \
    || echo "  (bucket exists or already owned — continuing)"
  for a in "${ARMS[@]}"; do
    aws s3 sync "$SCRATCH/$a/checkpoint-150000/" "$DST/$a/checkpoint-150000/" --region us-west-2
  done )

echo "=== DONE. Checkpoints now at: $DST/<arm>/checkpoint-150000/ ==="
echo "Eval ckpt-root for submit_evals --ckpt-root:"
echo "  s3://$DST_BUCKET/sarvesh.patil/wsm_robocasa/pretrain150k"
