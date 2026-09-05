#!/usr/bin/env bash
# Isolated 8-H200, two-step policy-training canary. Never a scientific training entry point.
set -euo pipefail

echo "======== RoboMME POLICY CANARY (NOT SCIENTIFIC EVIDENCE) | $(hostname) | $(date -u +%FT%TZ) ========"
CODE_DIR=/opt/ml/code
ROBOMME_COMPAT="$CODE_DIR/compat"
WORK=${ROBOMME_CANARY_WORK_ROOT:-/opt/ml/policy-canary}
[[ "$WORK" == /opt/ml/* && "$WORK" != /opt/ml ]] || {
  echo "FATAL unsafe canary work root $WORK" >&2; exit 20;
}
mkdir -p "$WORK" "$WORK/tmp" "$WORK/hf" "$WORK/uv-cache" "$WORK/jax-cache"
cd "$WORK"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export UV_CACHE_DIR="$WORK/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$WORK/jax-cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export JAX_TRACEBACK_FILTERING=off
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true

mapfile -t GPU_RECORDS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)
[[ "${#GPU_RECORDS[@]}" == 8 ]] || {
  echo "FATAL policy canary requires 8 GPUs, got ${#GPU_RECORDS[@]}" >&2; exit 20;
}
GPU_UUIDS=()
GPU_NAMES=()
for gpu_record in "${GPU_RECORDS[@]}"; do
  gpu_uuid="${gpu_record%%,*}"
  gpu_name="${gpu_record#*,}"
  gpu_uuid="${gpu_uuid#"${gpu_uuid%%[![:space:]]*}"}"
  gpu_name="${gpu_name#"${gpu_name%%[![:space:]]*}"}"
  [[ -n "$gpu_uuid" && -n "$gpu_name" ]] || {
    echo "FATAL malformed NVIDIA GPU inventory record: $gpu_record" >&2; exit 20;
  }
  [[ "${gpu_name^^}" == *H200* ]] || {
    echo "FATAL policy canary requires H200 GPUs, got $gpu_name" >&2; exit 20;
  }
  GPU_UUIDS+=("$gpu_uuid")
  GPU_NAMES+=("$gpu_name")
done
nvidia-smi -L
[[ -f "$ROBOMME_COMPAT/robocasa/utils/groot_utils/groot_dataset.py" ]] || {
  echo "FATAL isolated RoboMME compatibility surface absent" >&2; exit 25;
}

required=(
  ROBOMME_CANARY_KIND ROBOMME_CANARY_CLAIM ROBOMME_CANARY_ID ROBOMME_CANARY_STEPS
  ROBOMME_CANARY_MANIFEST_SOURCE ROBOMME_CANARY_MANIFEST_SHA256
  ROBOMME_CANARY_NAMESPACE_S3 ROBOMME_CANARY_RECEIPT_S3
  ROBOMME_REFERENCE_RUN_ID ROBOMME_REFERENCE_SCIENTIFIC_SPEC_SHA256
  ROBOMME_REFERENCE_SOURCE_SHA256 ROBOMME_ARM ROBOMME_SCOPE ROBOMME_TASK
  OPENPI_FORK_S3 OPENPI_REQUIRED_SENTINEL
  ROBOMME_DATA_S3 ROBOMME_DATA_PARENT_INVENTORY_S3
  ROBOMME_DATA_PARENT_INVENTORY_SHA256 ROBOMME_DATA_DERIVED_INVENTORY_SHA256
  INIT_S3 INIT_INVENTORY_S3 INIT_INVENTORY_SHA256
  PALIGEMMA_TOKENIZER_S3 PALIGEMMA_TOKENIZER_SHA256
  ROBOMME_WORKSPACE_S3 ROBOMME_WORKSPACE_MANIFEST_SHA256
  WSM_MAX_STEPS WSM_SAVE_INTERVAL WSM_WARMUP_STEPS WSM_PEAK_LR
  WSM_DECAY_STEPS WSM_DECAY_LR WSM_SEED
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 20; }
done
forbidden=(
  OUTPUT_S3 RUN_MANIFEST_SOURCE RUN_MANIFEST_SHA256 RUN_MANIFEST_S3
  PRODUCER_CLAIM_S3 COMPLETION_CLAIM_S3 CHECKPOINT_TREE_MANIFEST_ROOT
  ROBOMME_SCIENTIFIC_SPEC_SHA256 ROBOMME_FINAL_STEP ROBOMME_RUN_ID
)
for name in "${forbidden[@]}"; do
  [[ ! -v "$name" ]] || { echo "FATAL production flag leaked into canary: $name" >&2; exit 20; }
done
[[ "$ROBOMME_CANARY_KIND" == robomme_policy_training_canary_attempt ]] || {
  echo "FATAL canary kind drifted" >&2; exit 20;
}
[[ "$ROBOMME_CANARY_CLAIM" == not_scientific_training_evidence ]] || {
  echo "FATAL canary evidence claim drifted" >&2; exit 20;
}
[[ "$ROBOMME_CANARY_STEPS" == 2 && "$WSM_MAX_STEPS" == 2 && \
   "$WSM_SAVE_INTERVAL" == 2 && "$WSM_WARMUP_STEPS" == 1 && \
   "$WSM_DECAY_STEPS" == 2 && "$WSM_SEED" == 0 ]] || {
  echo "FATAL canary must be exactly two optimizer steps with the sealed smoke schedule" >&2; exit 21;
}
[[ "$ROBOMME_ARM" == gdn8_jepa_l01_k1 && "$ROBOMME_SCOPE" == single_task_canary ]] || {
  echo "FATAL this audited canary is only GDN8+JEPA single-task" >&2; exit 21;
}
[[ "$OPENPI_REQUIRED_SENTINEL" == _WSM_GDN_JEPA ]] || {
  echo "FATAL GDN+JEPA overlay sentinel absent" >&2; exit 24;
}
[[ "$ROBOMME_CANARY_RECEIPT_S3" == \
   "${ROBOMME_CANARY_NAMESPACE_S3%/}/training_canary.complete.json" ]] || {
  echo "FATAL canary receipt escaped its sealed namespace" >&2; exit 20;
}
case "$ROBOMME_CANARY_NAMESPACE_S3" in
  s3://*/manifests/canaries/policy_training/policy-canary-v1-*) ;;
  *) echo "FATAL unsafe/non-canary publication namespace" >&2; exit 20 ;;
esac

download_hashed() {
  local uri="$1" expected="$2" destination="$3"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || {
    echo "FATAL invalid expected SHA $expected" >&2; return 23;
  }
  aws s3 cp "$uri" "$destination" --only-show-errors
  [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected" ]] || {
    echo "FATAL checksum mismatch $uri" >&2; return 23;
  }
}

verify_manifest_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "FATAL missing manifest $path" >&2; return 23; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "FATAL artifact manifest mismatch $path" >&2; return 23;
  }
}

MANIFEST="$CODE_DIR/$ROBOMME_CANARY_MANIFEST_SOURCE"
[[ -f "$MANIFEST" ]] || { echo "FATAL staged canary manifest absent" >&2; exit 20; }
PYTHONPATH="$CODE_DIR" python3 -B -m training.policy_canary validate-manifest \
  --manifest "$MANIFEST" --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256" \
  --code-dir "$CODE_DIR"

receipt_location="${ROBOMME_CANARY_RECEIPT_S3#s3://}"
receipt_bucket="${receipt_location%%/*}"
receipt_key="${receipt_location#*/}"
namespace_location="${ROBOMME_CANARY_NAMESPACE_S3#s3://}"
namespace_bucket="${namespace_location%%/*}"
namespace_prefix="${namespace_location#*/}/"
[[ "$receipt_bucket" == "$namespace_bucket" && "$receipt_key" == "$namespace_prefix"* ]] || {
  echo "FATAL receipt bucket/key escaped namespace" >&2; exit 20;
}
if aws s3api head-object --bucket "$receipt_bucket" --key "$receipt_key" \
    >"$WORK/preexisting-receipt.json" 2>"$WORK/preexisting-receipt.error"; then
  echo "FATAL create-once canary receipt already exists" >&2
  exit 22
elif ! grep -Eq '\(404\)|Not Found|NoSuchKey' "$WORK/preexisting-receipt.error"; then
  echo "FATAL could not prove canary receipt absence" >&2
  cat "$WORK/preexisting-receipt.error" >&2
  exit 22
fi

DATA_PARENT="$WORK/robomme.parent.inventory.json"
INIT_INVENTORY="$WORK/init.inventory.json"
download_hashed \
  "$ROBOMME_DATA_PARENT_INVENTORY_S3" \
  "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" \
  "$DATA_PARENT"
download_hashed "$INIT_INVENTORY_S3" "$INIT_INVENTORY_SHA256" "$INIT_INVENTORY"

DATA="$WORK/robomme_data"
INIT="$WORK/init_ckpt"
PYTHONPATH="$CODE_DIR" python3 -B -m fleet.task_inventory \
  --parent-manifest "$DATA_PARENT" \
  --task "$ROBOMME_TASK" \
  --root-s3 "$ROBOMME_DATA_S3" \
  --destination "$DATA" \
  --expected-derived-sha256 "$ROBOMME_DATA_DERIVED_INVENTORY_SHA256" \
  --workers 48 &
DATA_PID=$!
PYTHONPATH="$CODE_DIR" python3 -B -m fleet.inventory \
  --manifest "$INIT_INVENTORY" --artifact pi05_h300_mg_init \
  --root-s3 "$INIT_S3" --destination "$INIT" --workers 48 &
INIT_PID=$!

OPENPI_SHA="${OPENPI_FORK_S3%.tgz}"
OPENPI_SHA="${OPENPI_SHA##*/}"
[[ "$OPENPI_SHA" =~ ^[0-9a-f]{64}$ ]] || {
  echo "FATAL non-content-addressed OpenPI URI" >&2; exit 24;
}
download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA" "$WORK/openpi.tgz"
OPENPI="$WORK/openpi"
mkdir "$OPENPI"
tar xzf "$WORK/openpi.tgz" -C "$OPENPI"
cd "$OPENPI"
export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
uv sync --frozen
PY="$OPENPI/.venv/bin/python"

GDN_JEPA_OVERLAY="$WORK/openpi-gdn-jepa-config-v1"
PYTHONPATH="$CODE_DIR:$ROBOMME_COMPAT" "$PY" -B \
  -m training.gdn_jepa_overlay stage \
  --source-repo "$OPENPI" \
  --output-repo "$GDN_JEPA_OVERLAY" \
  --source-archive-sha256 "$OPENPI_SHA" \
  >"$WORK/gdn-jepa-overlay-stage.json"
TRAIN_PYTHONPATH="$GDN_JEPA_OVERLAY/src:$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"
PYTHONPATH="$TRAIN_PYTHONPATH" "$PY" -B \
  -m training.gdn_jepa_overlay validate-loaded \
  --overlay-repo "$GDN_JEPA_OVERLAY" \
  >"$WORK/gdn-jepa-overlay-loaded.json"

wait "$DATA_PID"
wait "$INIT_PID"
export OPENPI_DATA_HOME="$WORK/openpi_cache"
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
mkdir -p "$(dirname "$TOKENIZER")"
download_hashed "$PALIGEMMA_TOKENIZER_S3" "$PALIGEMMA_TOKENIZER_SHA256" "$TOKENIZER"

mkdir -p "$WORK/workspace/$ROBOMME_TASK"
aws s3 sync "$ROBOMME_WORKSPACE_S3" "$WORK/workspace/$ROBOMME_TASK" --only-show-errors
verify_manifest_sha \
  "$WORK/workspace/$ROBOMME_TASK/MANIFEST.json" \
  "$ROBOMME_WORKSPACE_MANIFEST_SHA256"

export ROBOMME_DATA_ROOT="$DATA"
export ROBOMME_ASSETS_ROOT="$DATA/assets"
export WSM_INIT_FROM="$INIT/params"
export ROBOMME_WORKSPACE_ROOT="$WORK/workspace"
export ROBOMME_GDN_JEPA_OVERLAY_ROOT="$GDN_JEPA_OVERLAY"
export WSM_CKPT_BASE="$WORK/checkpoints"
export WSM_EXP_NAME="$ROBOMME_CANARY_ID"
export WSM_WSM_POLICY_ALLOW_RUN=1
export WSM_EXPECTED_JAX_DEVICES=8
export WSM_NUM_WORKERS=32
export WSM_FINAL_ONLY_CHECKPOINTS=1
export WSM_RESUME=0
export WANDB_MODE=disabled
[[ -d "$WSM_INIT_FROM" ]] || { echo "FATAL init params absent" >&2; exit 25; }

"$PY" -c 'import jax, flax, orbax.checkpoint as o; print("jax", jax.__version__, "devices", jax.devices(), "flax", flax.__version__, "orbax", o.__version__)'
cd "$CODE_DIR"
PROOF="$WORK/training_canary.complete.json"
RUN_ARGS=(
  -B -m training.policy_canary run
  --manifest "$MANIFEST"
  --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256"
  --code-dir "$CODE_DIR"
  --proof "$PROOF"
)
for index in "${!GPU_NAMES[@]}"; do
  RUN_ARGS+=(--gpu-uuid "${GPU_UUIDS[$index]}")
  RUN_ARGS+=(--gpu-name "${GPU_NAMES[$index]}")
done
PYTHONPATH="$TRAIN_PYTHONPATH" "$PY" "${RUN_ARGS[@]}"
[[ -f "$PROOF" && ! -L "$PROOF" ]] || {
  echo "FATAL canary proof absent after successful runner" >&2; exit 26;
}
PYTHONPATH="$CODE_DIR" python3 -B -m training.policy_canary validate-receipt \
  --receipt "$PROOF"

# Re-check the entire namespace immediately before the only allowed S3 write. A concurrent writer
# is a hard collision; this canary never accepts, replaces, or aliases an existing receipt.
namespace_count="$(aws s3api list-objects-v2 \
  --bucket "$namespace_bucket" --prefix "$namespace_prefix" --max-keys 1 \
  --query 'KeyCount' --output text)"
[[ "$namespace_count" == 0 ]] || {
  echo "FATAL canary namespace became nonempty before create-once publication" >&2; exit 22;
}
aws s3api put-object \
  --bucket "$receipt_bucket" \
  --key "$receipt_key" \
  --body "$PROOF" \
  --content-type application/json \
  --if-none-match '*' \
  >"$WORK/put-receipt.json"
aws s3 cp "$ROBOMME_CANARY_RECEIPT_S3" "$WORK/published-receipt.json" --only-show-errors
cmp -s "$PROOF" "$WORK/published-receipt.json" || {
  echo "FATAL published canary receipt bytes differ" >&2; exit 27;
}
PYTHONPATH="$CODE_DIR" python3 -B -m training.policy_canary validate-receipt \
  --receipt "$WORK/published-receipt.json"
echo "ROBOMME POLICY CANARY COMPLETE claim=$ROBOMME_CANARY_CLAIM canary_id=$ROBOMME_CANARY_ID receipt=$ROBOMME_CANARY_RECEIPT_S3"
