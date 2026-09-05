#!/usr/bin/env bash
# Isolated v4 GDN8+JEPA+VISReg 8-H100 canary. Never a scientific training entry point.
set -euo pipefail

CODE_DIR=/opt/ml/code
COMPAT="$CODE_DIR/compat"
WORK=${ROBOMME_CANARY_WORK_ROOT:-/opt/ml/v4-policy-canary}
[[ "$WORK" == /opt/ml/* && "$WORK" != /opt/ml ]] || { echo "unsafe canary root" >&2; exit 20; }
mkdir -p "$WORK" "$WORK/tmp" "$WORK/hf" "$WORK/uv-cache" "$WORK/jax-cache"
cd "$WORK"
export TMPDIR="$WORK/tmp" HF_HOME="$WORK/hf" UV_CACHE_DIR="$WORK/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$WORK/jax-cache" XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 JAX_TRACEBACK_FILTERING=off
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true

[[ -f "$COMPAT/robocasa/utils/groot_utils/embodiment_tags.py" ]] || {
  echo "isolated RoboCasa compatibility surface absent" >&2; exit 20;
}

mapfile -t GPU_RECORDS < <(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader)
[[ "${#GPU_RECORDS[@]}" == 8 ]] || { echo "requires exactly 8 GPUs" >&2; exit 20; }
GPU_UUIDS=(); GPU_NAMES=(); GPU_MEMORY_MIB=()
for record in "${GPU_RECORDS[@]}"; do
  uuid="${record%%,*}"; remainder="${record#*,}"; name="${remainder%,*}"; memory="${remainder##*,}"
  uuid="${uuid#"${uuid%%[![:space:]]*}"}"; name="${name#"${name%%[![:space:]]*}"}"
  name="${name%"${name##*[![:space:]]}"}"; memory="${memory#"${memory%%[![:space:]]*}"}"
  memory="${memory% MiB}"
  [[ -n "$uuid" && "$name" == "NVIDIA H100 80GB HBM3" && "$memory" =~ ^[0-9]+$ && \
     "$memory" -ge 80000 && "$memory" -le 90000 ]] || { echo "requires exact H100 80GB HBM3: $record" >&2; exit 20; }
  GPU_UUIDS+=("$uuid"); GPU_NAMES+=("$name"); GPU_MEMORY_MIB+=("$memory")
done
nvidia-smi -L

required=(
  ROBOMME_CANARY_KIND ROBOMME_CANARY_CLAIM ROBOMME_CANARY_ID
  ROBOMME_CANARY_MANIFEST_SHA256 ROBOMME_CANARY_RECEIPT_S3 ROBOMME_CANARY_NAMESPACE_S3
  ROBOMME_TASK ROBOMME_ARM ROBOMME_DATA_S3 ROBOMME_DATA_PARENT_INVENTORY_S3
  ROBOMME_DATA_PARENT_INVENTORY_SHA256 ROBOMME_DATA_DERIVED_INVENTORY_SHA256
  INIT_S3 INIT_INVENTORY_S3 INIT_INVENTORY_SHA256 OPENPI_FORK_S3 OPENPI_REQUIRED_SENTINEL
  PALIGEMMA_TOKENIZER_S3 PALIGEMMA_TOKENIZER_SHA256 ROBOMME_WORKSPACE_S3
  ROBOMME_WORKSPACE_ENCODER_ID ROBOMME_WORKSPACE_MANIFEST_SHA256 WSM_MAX_STEPS WSM_SAVE_INTERVAL WSM_WARMUP_STEPS
  WSM_PEAK_LR WSM_DECAY_STEPS WSM_DECAY_LR WSM_SEED
)
for key in "${required[@]}"; do [[ -n "${!key:-}" ]] || { echo "missing $key" >&2; exit 20; }; done
for key in OUTPUT_S3 RUN_MANIFEST_SOURCE RUN_MANIFEST_SHA256 RUN_MANIFEST_S3 PRODUCER_CLAIM_S3 \
  COMPLETION_CLAIM_S3 CHECKPOINT_TREE_MANIFEST_ROOT ROBOMME_SCIENTIFIC_SPEC_SHA256 \
  ROBOMME_FINAL_STEP ROBOMME_RUN_ID; do
  [[ ! -v "$key" ]] || { echo "production flag leaked: $key" >&2; exit 20; }
done
[[ "$ROBOMME_CANARY_KIND" == robomme_v4_policy_training_canary_attempt ]] || exit 21
[[ "$ROBOMME_CANARY_CLAIM" == runtime_evidence_only_not_scientific_training_evidence ]] || exit 21
[[ "$ROBOMME_TASK" == PickXtimes && "$ROBOMME_ARM" == v4_gdn8_jepa_visreg_l01_k1 ]] || exit 21
[[ "$OPENPI_REQUIRED_SENTINEL" == _WSM_V4_ADVANCED ]] || exit 21
[[ "$WSM_MAX_STEPS" == 2 && "$WSM_SAVE_INTERVAL" == 2 && "$WSM_WARMUP_STEPS" == 1 && \
   "$WSM_DECAY_STEPS" == 2 && "$WSM_PEAK_LR" == 5e-5 && "$WSM_DECAY_LR" == 5e-6 && \
   "$WSM_SEED" == 0 ]] || { echo "canary schedule drifted" >&2; exit 21; }
[[ "$ROBOMME_CANARY_RECEIPT_S3" == "${ROBOMME_CANARY_NAMESPACE_S3%/}/training_canary.complete.json" ]] || exit 21
case "$ROBOMME_CANARY_NAMESPACE_S3" in
  s3://*/manifests/canaries/policy_training/v4-policy-canary-*) ;;
  *) echo "unsafe canary namespace" >&2; exit 21 ;;
esac

MANIFEST="$CODE_DIR/_robomme_v4_policy_canary_manifest.json"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || exit 22
PYTHONPATH="$CODE_DIR" python3 -B -m training.v4_policy_canary validate-manifest \
  --manifest "$MANIFEST" --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256"
PYTHONPATH="$CODE_DIR" python3 -B -m training.v4_policy_canary validate-environment \
  --manifest "$MANIFEST" --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256"

namespace="${ROBOMME_CANARY_NAMESPACE_S3#s3://}"
bucket="${namespace%%/*}"; prefix="${namespace#*/}/"
receipt="${ROBOMME_CANARY_RECEIPT_S3#s3://}"; receipt_bucket="${receipt%%/*}"; receipt_key="${receipt#*/}"
[[ "$bucket" == "$receipt_bucket" && "$receipt_key" == "$prefix"* ]] || exit 22
count="$(aws s3api list-objects-v2 --bucket "$bucket" --prefix "$prefix" --max-keys 1 --query KeyCount --output text)"
[[ "$count" == 0 ]] || { echo "canary namespace is not empty" >&2; exit 22; }

download_hashed() {
  local uri="$1" expected="$2" destination="$3"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || return 23
  aws s3 cp "$uri" "$destination" --only-show-errors
  [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected" ]] || return 23
}
verify_manifest() {
  [[ -f "$1" && "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || return 23
}

DATA_PARENT="$WORK/data.parent.json"; INIT_INVENTORY="$WORK/init.inventory.json"
download_hashed "$ROBOMME_DATA_PARENT_INVENTORY_S3" "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" "$DATA_PARENT"
download_hashed "$INIT_INVENTORY_S3" "$INIT_INVENTORY_SHA256" "$INIT_INVENTORY"
DATA="$WORK/data"; INIT="$WORK/init"
DATA_PID=""; INIT_PID=""
cleanup_downloads() {
  for pid in "$DATA_PID" "$INIT_PID"; do
    [[ -z "$pid" ]] || kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  for pid in "$DATA_PID" "$INIT_PID"; do [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; done
}
trap cleanup_downloads EXIT INT TERM
PYTHONPATH="$CODE_DIR" python3 -B -m fleet.task_inventory \
  --parent-manifest "$DATA_PARENT" --task "$ROBOMME_TASK" --root-s3 "$ROBOMME_DATA_S3" \
  --destination "$DATA" --expected-derived-sha256 "$ROBOMME_DATA_DERIVED_INVENTORY_SHA256" --workers 48 &
DATA_PID=$!
PYTHONPATH="$CODE_DIR" python3 -B -m fleet.inventory \
  --manifest "$INIT_INVENTORY" --artifact pi05_h300_mg_init --root-s3 "$INIT_S3" \
  --destination "$INIT" --workers 48 &
INIT_PID=$!

OPENPI_SHA="${OPENPI_FORK_S3%.tgz}"; OPENPI_SHA="${OPENPI_SHA##*/}"
download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA" "$WORK/openpi.tgz"
OPENPI="$WORK/openpi"; mkdir "$OPENPI"; tar xzf "$WORK/openpi.tgz" -C "$OPENPI"
cd "$OPENPI"; export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"; uv sync --frozen
PY="$OPENPI/.venv/bin/python"
PYTHONPATH="$COMPAT:$CODE_DIR:$OPENPI/src" "$PY" - <<'PY'
import inspect
from openpi.models import pi0_config, wsm_current_cond, wsm_jepa
parameters = inspect.signature(pi0_config.Pi0Config).parameters
assert "wsm_cond_history_dropout" in parameters
assert "wsm_jepa_regularizer" in parameters
assert "visreg_weight" in inspect.signature(wsm_jepa.wsm_jepa_aux_loss).parameters
assert getattr(wsm_current_cond, "_WSM_PTRM", False)
assert getattr(pi0_config, "_WORKSPACE_COMBO", None) == {"tanh", "jepa_aux_target"}
PY
data_status=0; init_status=0
wait "$DATA_PID" || data_status=$?
DATA_PID=""
wait "$INIT_PID" || init_status=$?
INIT_PID=""
[[ "$data_status" == 0 && "$init_status" == 0 ]] || {
  echo "artifact download failed data=$data_status init=$init_status" >&2; exit 23;
}
trap - EXIT INT TERM

export OPENPI_DATA_HOME="$WORK/openpi_cache"
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"; mkdir -p "$(dirname "$TOKENIZER")"
download_hashed "$PALIGEMMA_TOKENIZER_S3" "$PALIGEMMA_TOKENIZER_SHA256" "$TOKENIZER"
mkdir -p "$WORK/workspace/$ROBOMME_TASK"
aws s3 sync "$ROBOMME_WORKSPACE_S3" "$WORK/workspace/$ROBOMME_TASK" --only-show-errors
verify_manifest "$WORK/workspace/$ROBOMME_TASK/MANIFEST.json" "$ROBOMME_WORKSPACE_MANIFEST_SHA256"

export ROBOMME_DATA_ROOT="$DATA" ROBOMME_ASSETS_ROOT="$DATA/assets" WSM_INIT_FROM="$INIT/params"
export ROBOMME_WORKSPACE_ROOT="$WORK/workspace" WSM_CKPT_BASE="$WORK/checkpoints"
export WSM_EXP_NAME="$ROBOMME_CANARY_ID" WSM_WSM_POLICY_ALLOW_RUN=1 WSM_EXPECTED_JAX_DEVICES=8
export WSM_NUM_WORKERS=32 WSM_FINAL_ONLY_CHECKPOINTS=1 WSM_RESUME=0 WANDB_MODE=disabled
TRAIN_PYTHONPATH="$OPENPI/src:$COMPAT:$CODE_DIR"
cd "$CODE_DIR"; PROOF="$WORK/training_canary.complete.json"
ARGS=(-B -m training.v4_policy_canary run --manifest "$MANIFEST" \
  --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256" --code-dir "$CODE_DIR" --proof "$PROOF")
for index in "${!GPU_NAMES[@]}"; do
  ARGS+=(--gpu-name "${GPU_NAMES[$index]}" --gpu-uuid "${GPU_UUIDS[$index]}" \
    --gpu-memory-mib "${GPU_MEMORY_MIB[$index]}")
done
PYTHONPATH="$TRAIN_PYTHONPATH" "$PY" "${ARGS[@]}"
PYTHONPATH="$CODE_DIR" python3 -B -m training.v4_policy_canary validate-receipt \
  --receipt "$PROOF" --manifest "$MANIFEST" --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256"

# Recheck isolation immediately before the only cloud write, then create once and read back bytes.
count="$(aws s3api list-objects-v2 --bucket "$bucket" --prefix "$prefix" --max-keys 1 --query KeyCount --output text)"
[[ "$count" == 0 ]] || { echo "concurrent canary namespace collision" >&2; exit 22; }
aws s3api put-object --bucket "$receipt_bucket" --key "$receipt_key" --body "$PROOF" \
  --content-type application/json --if-none-match '*' >"$WORK/put.json"
aws s3 cp "$ROBOMME_CANARY_RECEIPT_S3" "$WORK/readback.json" --only-show-errors
cmp -s "$PROOF" "$WORK/readback.json" || { echo "receipt readback mismatch" >&2; exit 27; }
PYTHONPATH="$CODE_DIR" python3 -B -m training.v4_policy_canary validate-receipt \
  --receipt "$WORK/readback.json" --manifest "$MANIFEST" --sha256 "$ROBOMME_CANARY_MANIFEST_SHA256"
echo "V4 POLICY CANARY COMPLETE (NOT SCIENTIFIC) $ROBOMME_CANARY_ID"
