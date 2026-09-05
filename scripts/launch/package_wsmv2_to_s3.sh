#!/usr/bin/env bash
# Package the wsmv2 repo into the S3 tarball that the FT entry scripts download (WSM_REPO_S3).
# Same pattern as the openpi jax-latest fork tarball. Run this BEFORE submit_finetunes.py, and
# re-run after any change to utils/ vla_training/ workspace_models/ scripts/configs/.
#
#   bash scripts/launch/package_wsmv2_to_s3.sh                 # -> s3://.../<user>/wsm_robocasa/code/wsmv2.tgz
#   WSM_USER=jane.doe bash scripts/launch/package_wsmv2_to_s3.sh
set -euo pipefail
REGION="us-west-2"; ACCOUNT="124224456861"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_PREFIX="${WSM_USER:-$(aws sts get-caller-identity --query Arn --output text | sed 's#.*/##; s/@.*//')}"
DEST="s3://sagemaker-${REGION}-${ACCOUNT}/${USER_PREFIX}/wsm_robocasa/code/wsmv2.tgz"
TGZ="$(mktemp -d)/wsmv2.tgz"

echo "[package] repo=$REPO"
echo "[package] -> $DEST"
# Ship the import surface the launchers need; exclude vcs/caches/venvs/local ckpts/labels.
tar czf "$TGZ" -C "$REPO" \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  --exclude='*.egg-info' --exclude='.pytest_cache' --exclude='.claude' \
  utils vla_training workspace_models scripts pyproject.toml README.md
echo "[package] tarball: $(du -h "$TGZ" | cut -f1)"
aws s3 cp "$TGZ" "$DEST" --only-show-errors
echo "[package] uploaded ✓  $DEST"
rm -rf "$(dirname "$TGZ")"
