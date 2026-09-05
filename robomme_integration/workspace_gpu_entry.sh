#!/usr/bin/env bash
# Two independent four-H200 lanes produce one uniform pair of RoboMME workspace bundles.
set -euo pipefail

echo "======== RoboMME workspace producer | $(hostname) | $(date -u +%FT%TZ) ========"
CODE_DIR=/opt/ml/code
WORK=/opt/ml/workspace-producer
mkdir -p "$WORK" "$WORK/tmp" "$WORK/tasks"
cd "$WORK"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export UV_CACHE_DIR="$WORK/uv-cache"
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true
nvidia-smi -L
CONTROL_PY="$(command -v python3)"
"$CONTROL_PY" -c 'import boto3, botocore; print("WORKSPACE_CONTROL_PREFLIGHT_OK", boto3.__version__, botocore.__version__)'

required=(
  ROBOMME_WORKSPACE_PAIR_ID ROBOMME_WORKSPACE_SOURCE_SHA256
  ROBOMME_WORKSPACE_PAIR_MANIFEST_SOURCE ROBOMME_WORKSPACE_PAIR_MANIFEST_SHA256
  ROBOMME_WORKSPACE_PAIR_MANIFEST_S3 ROBOMME_WORKSPACE_PAIR_PRODUCER_CLAIM_S3
  ROBOMME_WORKSPACE_PAIR_COMPLETION_CLAIM_S3 ROBOMME_WORKSPACE_ARTIFACT_ROOT_S3
  OPENPI_FORK_S3 ROBOMME_DATA_S3 ROBOMME_DATA_PARENT_INVENTORY_S3
  ROBOMME_DATA_PARENT_INVENTORY_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 20; }
done

publish_once() {
  local source="$1" destination="$2" existing="$WORK/existing-immutable"
  rm -f "$existing"
  if aws s3 cp "$destination" "$existing" --only-show-errors 2>/dev/null; then
    cmp -s "$source" "$existing" || { echo "FATAL immutable collision $destination" >&2; return 22; }
    return 0
  fi
  local location bucket key
  location="${destination#s3://}"
  bucket="${location%%/*}"
  key="${location#*/}"
  if aws s3api put-object --bucket "$bucket" --key "$key" --body "$source" \
      --if-none-match '*' >/dev/null; then
    return 0
  fi
  aws s3 cp "$destination" "$existing" --only-show-errors
  cmp -s "$source" "$existing" || { echo "FATAL immutable collision $destination" >&2; return 22; }
}

download_hashed() {
  local uri="$1" expected="$2" destination="$3"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "FATAL invalid SHA $expected" >&2; return 23; }
  aws s3 cp "$uri" "$destination" --only-show-errors
  [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected" ]] || {
    echo "FATAL checksum mismatch $uri" >&2; return 23;
  }
}

PAIR_MANIFEST="$WORK/pair-manifest.json"
cp "$CODE_DIR/$ROBOMME_WORKSPACE_PAIR_MANIFEST_SOURCE" "$PAIR_MANIFEST"
[[ "$(sha256sum "$PAIR_MANIFEST" | awk '{print $1}')" == "$ROBOMME_WORKSPACE_PAIR_MANIFEST_SHA256" ]] || {
  echo "FATAL pair manifest SHA mismatch" >&2; exit 23;
}
publish_once "$PAIR_MANIFEST" "$ROBOMME_WORKSPACE_PAIR_MANIFEST_S3"

PRODUCER="$WORK/producer.json"
python3 - "$PRODUCER" <<'PY'
import json, os, sys
value = {
    "schema_version": 1,
    "kind": "robomme_all16_workspace_pair_producer",
    "pair_id": os.environ["ROBOMME_WORKSPACE_PAIR_ID"],
    "source_tree_sha256": os.environ["ROBOMME_WORKSPACE_SOURCE_SHA256"],
    "training_job_name": os.environ.get("TRAINING_JOB_NAME") or os.environ.get("SM_TRAINING_JOB_NAME"),
    "pair_manifest_sha256": os.environ["ROBOMME_WORKSPACE_PAIR_MANIFEST_SHA256"],
}
if not value["training_job_name"]:
    raise SystemExit("training job name unavailable")
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
publish_once "$PRODUCER" "$ROBOMME_WORKSPACE_PAIR_PRODUCER_CLAIM_S3"

PARENT="$WORK/robomme.parent.inventory.json"
download_hashed \
  "$ROBOMME_DATA_PARENT_INVENTORY_S3" "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" "$PARENT"

OPENPI_SHA="${OPENPI_FORK_S3%.tgz}"
OPENPI_SHA="${OPENPI_SHA##*/}"
download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA" "$WORK/openpi.tgz"
OPENPI="$WORK/openpi"
mkdir "$OPENPI"
tar xzf "$WORK/openpi.tgz" -C "$OPENPI"
cd "$OPENPI"
export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
uv sync --frozen
PY="$OPENPI/.venv/bin/python"
"$PY" -c 'import jax, numpy, pyarrow, optax, orbax.checkpoint; assert len(jax.devices()) == 8; assert {d.platform for d in jax.devices()} == {"gpu"}; print("WORKSPACE_GPU_PREFLIGHT_OK", jax.devices(), numpy.__version__, pyarrow.__version__)'
export ROBOMME_WORKSPACE_COMPUTE_PYTHON="$PY"

cd "$CODE_DIR"
PYTHONPATH="$CODE_DIR:$OPENPI/src" "$CONTROL_PY" -m training.workspace_gpu_producer pair \
  --manifest "$PAIR_MANIFEST" \
  --pair-id "$ROBOMME_WORKSPACE_PAIR_ID" \
  --source-tree-sha256 "$ROBOMME_WORKSPACE_SOURCE_SHA256" \
  --pair-claim-s3 "$ROBOMME_WORKSPACE_PAIR_COMPLETION_CLAIM_S3" \
  --parent-manifest "$PARENT" \
  --data-s3 "$ROBOMME_DATA_S3" \
  --artifact-root-s3 "$ROBOMME_WORKSPACE_ARTIFACT_ROOT_S3" \
  --work-root "$WORK/tasks"

echo "ROBOMME WORKSPACE GPU PAIR COMPLETE pair_id=$ROBOMME_WORKSPACE_PAIR_ID"
