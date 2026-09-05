#!/usr/bin/env bash
# Isolated 8-GPU entry for one full RoboMME training run (20k single-task; 60k or A19 70k multitask;
# 80k official-recipe diagnostic) on p5e/H200 or p5/H100.
set -euo pipefail

echo "======== RoboMME GPU train | $(hostname) | $(date -u +%FT%TZ) ========"
CODE_DIR=/opt/ml/code
ROBOMME_COMPAT="$CODE_DIR/compat"
WORK=${ROBOMME_WORK_ROOT:-/opt/ml/work}
CACHE_ROOT=${ROBOMME_CACHE_ROOT:-$WORK}
SHARED_ARTIFACT_ROOT=${ROBOMME_SHARED_ARTIFACT_ROOT:-}
[[ "$WORK" == /opt/ml/* && "$WORK" != /opt/ml ]] || {
  echo "FATAL unsafe work root $WORK" >&2; exit 20;
}
[[ "$CACHE_ROOT" == /opt/ml/* && "$CACHE_ROOT" != /opt/ml ]] || {
  echo "FATAL unsafe cache root $CACHE_ROOT" >&2; exit 20;
}
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  [[ "$SHARED_ARTIFACT_ROOT" == /opt/ml/* && "$SHARED_ARTIFACT_ROOT" != /opt/ml ]] || {
    echo "FATAL unsafe shared artifact root $SHARED_ARTIFACT_ROOT" >&2; exit 20;
  }
  mkdir -p "$SHARED_ARTIFACT_ROOT"
fi
mkdir -p "$WORK" "$WORK/tmp"
mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/uv-cache"
cd "$WORK"
export TMPDIR="$WORK/tmp"
export HF_HOME="$CACHE_ROOT/hf"
export UV_CACHE_DIR="$CACHE_ROOT/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax-compilation-cache"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$JAX_COMPILATION_CACHE_DIR"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export JAX_TRACEBACK_FILTERING=off
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true
nvidia-smi -L
[[ -f "$ROBOMME_COMPAT/robocasa/utils/groot_utils/groot_dataset.py" ]] || {
  echo "FATAL isolated RoboMME import compatibility surface absent" >&2; exit 25;
}

required=(
  ROBOMME_ARM ROBOMME_SCOPE ROBOMME_RUN_ID ROBOMME_ATTEMPT_ID
  ROBOMME_SCIENTIFIC_SPEC_SHA256 ROBOMME_FINAL_STEP
  WSM_MAX_STEPS OPENPI_FORK_S3 ROBOMME_DATA_S3 ROBOMME_DATA_PARENT_INVENTORY_S3
  ROBOMME_DATA_PARENT_INVENTORY_SHA256
  PALIGEMMA_TOKENIZER_S3
  PALIGEMMA_TOKENIZER_SHA256 OUTPUT_S3 RUN_MANIFEST_SOURCE RUN_MANIFEST_SHA256
  RUN_MANIFEST_S3 PRODUCER_CLAIM_S3 COMPLETION_CLAIM_S3 CHECKPOINT_TREE_MANIFEST_ROOT
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 20; }
done
[[ "$ROBOMME_ARM" =~ ^[a-z0-9_]+$ ]] || { echo "FATAL unsafe arm" >&2; exit 20; }
# ROBOMME_RECIPE is the explicit opt-in for the A19 checkpoint-maturity recipe (launch.py
# --multitask-train-steps 70000).  Unset means the sealed recipe for the scope/arm.
[[ -z "${ROBOMME_RECIPE:-}" || "$ROBOMME_RECIPE" == v4_70k ]] || {
  echo "FATAL unknown ROBOMME_RECIPE=$ROBOMME_RECIPE" >&2; exit 20;
}
if [[ "$ROBOMME_ARM" == official_recipe_lerobot ]]; then
  recipe_required=(
    ROBOMME_RECIPE_LABEL ROBOMME_PI05_BASE_INIT_S3
    ROBOMME_PI05_BASE_INIT_INVENTORY_S3 ROBOMME_PI05_BASE_INIT_INVENTORY_SHA256
    ROBOMME_CHECKPOINT_MILESTONES ROBOMME_SUCCESS_CHECKPOINT_MILESTONES
  )
  for name in "${recipe_required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "FATAL official recipe missing $name" >&2; exit 20; }
  done
  [[ "$ROBOMME_RECIPE_LABEL" == recipe_matched_lerobot_not_exact_source_or_data_reproduction ]] || {
    echo "FATAL official_recipe_lerobot claim label drifted" >&2; exit 20;
  }
  [[ -z "${INIT_S3:-}" && -z "${INIT_INVENTORY_S3:-}" && -z "${INIT_INVENTORY_SHA256:-}" ]] || {
    echo "FATAL official_recipe_lerobot must not alias the legacy INIT_* channel" >&2; exit 20;
  }
else
  for name in INIT_S3 INIT_INVENTORY_S3 INIT_INVENTORY_SHA256; do
    [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 20; }
  done
  [[ -z "${ROBOMME_RECIPE_LABEL:-}" && -z "${ROBOMME_PI05_BASE_INIT_S3:-}" ]] || {
    echo "FATAL project arm received official-recipe initialization metadata" >&2; exit 20;
  }
fi
case "$ROBOMME_SCOPE" in
  single_task)
    [[ "$ROBOMME_ARM" != official_recipe_lerobot ]] || {
      echo "FATAL official_recipe_lerobot is all16-only" >&2; exit 20;
    }
    [[ "${ROBOMME_TASK:-}" =~ ^[A-Za-z0-9]+$ ]] || { echo "FATAL unsafe/missing task" >&2; exit 20; }
    [[ -n "${ROBOMME_DATA_DERIVED_INVENTORY_SHA256:-}" ]] || { echo "FATAL task inventory absent" >&2; exit 20; }
    [[ -z "${ROBOMME_RECIPE:-}" ]] || { echo "FATAL single-task runs have no maturity recipe" >&2; exit 21; }
    [[ "$WSM_MAX_STEPS" == 20000 && "$ROBOMME_FINAL_STEP" == 19999 ]] || {
      echo "FATAL single-task recipe must be 20k/step-19999" >&2; exit 21;
    }
    ;;
  multitask)
    [[ -z "${ROBOMME_TASK:-}" && -z "${ROBOMME_DATA_DERIVED_INVENTORY_SHA256:-}" ]] || {
      echo "FATAL multitask run must consume the parent all16 inventory" >&2; exit 20;
    }
    if [[ "$ROBOMME_ARM" == official_recipe_lerobot ]]; then
      [[ -z "${ROBOMME_RECIPE:-}" ]] || { echo "FATAL official_recipe_lerobot has no maturity recipe" >&2; exit 21; }
      [[ "$WSM_MAX_STEPS" == 80000 && "$ROBOMME_FINAL_STEP" == 79999 ]] || {
        echo "FATAL official_recipe_lerobot must be 80k/step-79999" >&2; exit 21;
      }
    elif [[ "${ROBOMME_RECIPE:-}" == v4_70k ]]; then
      # A19 checkpoint-maturity recipe: multitask v4 arms only; every 10k milestone is retained
      # remotely during training and exported as a deploy-only tree after success.
      [[ "$ROBOMME_RUN_ID" == mt-v4-70k-all16-* ]] || {
        echo "FATAL v4_70k is defined only for multitask v4 arms (run_id prefix mt-v4-70k-all16-)" >&2; exit 21;
      }
      [[ "$WSM_MAX_STEPS" == 70000 && "$ROBOMME_FINAL_STEP" == 69999 ]] || {
        echo "FATAL v4_70k recipe must be 70k/step-69999" >&2; exit 21;
      }
      [[ "${ROBOMME_CHECKPOINT_MILESTONES:-}" == 10000,20000,30000,40000,50000,60000 && \
         "${ROBOMME_SUCCESS_CHECKPOINT_MILESTONES:-}" == 10000,20000,30000,40000,50000,60000 ]] || {
        echo "FATAL v4_70k must retain steps 10000,20000,30000,40000,50000,60000,69999" >&2; exit 21;
      }
      [[ "${WSM_WARMUP_STEPS:-}" == 3500 && "${WSM_DECAY_STEPS:-}" == 70000 ]] || {
        echo "FATAL v4_70k schedule must be warmup 3500 / cosine decay through 70000" >&2; exit 21;
      }
    else
      [[ -z "${ROBOMME_RECIPE:-}" && -z "${ROBOMME_SUCCESS_CHECKPOINT_MILESTONES:-}" ]] || {
        echo "FATAL sealed 60k multitask recipe received checkpoint-maturity metadata" >&2; exit 21;
      }
      [[ "$WSM_MAX_STEPS" == 60000 && "$ROBOMME_FINAL_STEP" == 59999 ]] || {
        echo "FATAL multitask recipe must be 60k/step-59999" >&2; exit 21;
      }
    fi
    ;;
  *) echo "FATAL invalid ROBOMME_SCOPE=$ROBOMME_SCOPE" >&2; exit 20 ;;
esac
if [[ "$ROBOMME_ARM" == official_recipe_lerobot ]]; then
  [[ "${WSM_SAVE_INTERVAL:-}" == 10000 && -z "${WSM_KEEP_PERIOD:-}" ]] || {
    echo "FATAL official_recipe_lerobot checkpoint cadence must be 10k" >&2; exit 21;
  }
  [[ "${WSM_WARMUP_STEPS:-}" == 10000 && "${WSM_PEAK_LR:-}" == 5e-5 && \
     "${WSM_DECAY_STEPS:-}" == 100000 && "${WSM_DECAY_LR:-}" == 5e-5 && \
     "${WSM_SEED:-}" == 42 ]] || {
    echo "FATAL official_recipe_lerobot optimizer schedule/seed drifted" >&2; exit 21;
  }
  [[ "$ROBOMME_CHECKPOINT_MILESTONES" == 60000,70000 && \
     "$ROBOMME_SUCCESS_CHECKPOINT_MILESTONES" == 60000,70000 ]] || {
    echo "FATAL official_recipe_lerobot must retain steps 60000,70000,79999" >&2; exit 21;
  }
else
  [[ "${WSM_SAVE_INTERVAL:-}" == 5000 ]] || { echo "FATAL save interval must be 5000" >&2; exit 21; }
fi

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

compare_attempt_receipts() {
  local current="$1" existing="$2"
  PYTHONPATH="$CODE_DIR" python3 - "$current" "$existing" <<'PY'
import json, sys
from fleet.checkpoint import deploy_receipts_equivalent
current, existing = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
if not deploy_receipts_equivalent(current, existing):
    raise SystemExit("receipt scientific identity differs")
PY
}

# Attempt provenance changes on a retry, while the run/checkpoint/tree identity must not.  This
# helper is create-once and never overwrites a prior receipt; a race is resolved by reading and
# semantically validating the winner.
publish_attempt_receipt_once() {
  local source="$1" destination="$2" existing="$WORK/existing-attempt-receipt"
  rm -f "$existing"
  if aws s3 cp "$destination" "$existing" --only-show-errors 2>/dev/null; then
    compare_attempt_receipts "$source" "$existing" || {
      echo "FATAL immutable scientific receipt collision $destination" >&2; return 27;
    }
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
  compare_attempt_receipts "$source" "$existing" || {
    echo "FATAL immutable scientific receipt collision $destination" >&2; return 27;
  }
}

download_hashed() {
  local uri="$1" expected="$2" destination="$3"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "FATAL invalid SHA $expected" >&2; return 23; }
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

RUN_MANIFEST="$WORK/run_manifest.json"
cp "$CODE_DIR/$RUN_MANIFEST_SOURCE" "$RUN_MANIFEST"
python3 - "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256" <<'PY'
import hashlib, json, sys
path, expected = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
claimed = value.pop("manifest_sha256", None)
actual = hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
).hexdigest()
if claimed != expected or actual != expected:
    raise SystemExit(f"run manifest seal mismatch claimed={claimed} actual={actual} expected={expected}")
PY
publish_once "$RUN_MANIFEST" "$RUN_MANIFEST_S3"

PRODUCER="$WORK/producer.json"
python3 - "$PRODUCER" <<'PY'
import json, os, sys
value = {
    "schema_version": 1,
    "kind": "robomme_gpu_training_producer",
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "attempt_id": os.environ["ROBOMME_ATTEMPT_ID"],
    "scientific_spec_sha256": os.environ["ROBOMME_SCIENTIFIC_SPEC_SHA256"],
    "training_job_name": os.environ.get("TRAINING_JOB_NAME") or os.environ.get("SM_TRAINING_JOB_NAME"),
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
}
if not value["training_job_name"]:
    raise SystemExit("training job name unavailable")
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
# Attempt-specific producer claims make retries resumable without permitting duplicate producers.
publish_once "$PRODUCER" "$PRODUCER_CLAIM_S3"

DATA_PARENT="$WORK/robomme.parent.inventory.json"
INIT_INVENTORY="$WORK/init.inventory.json"
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  DATA_PARENT="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts blob \
    --uri "$ROBOMME_DATA_PARENT_INVENTORY_S3" \
    --sha256 "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" \
    --cache-root "$SHARED_ARTIFACT_ROOT" --name robomme.parent.inventory.json)"
else
  download_hashed \
    "$ROBOMME_DATA_PARENT_INVENTORY_S3" "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" "$DATA_PARENT"
fi
if [[ "$ROBOMME_ARM" == official_recipe_lerobot ]]; then
  if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
    INIT_INVENTORY="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts blob \
      --uri "$ROBOMME_PI05_BASE_INIT_INVENTORY_S3" \
      --sha256 "$ROBOMME_PI05_BASE_INIT_INVENTORY_SHA256" \
      --cache-root "$SHARED_ARTIFACT_ROOT" --name init.inventory.json)"
  else
    download_hashed \
      "$ROBOMME_PI05_BASE_INIT_INVENTORY_S3" \
      "$ROBOMME_PI05_BASE_INIT_INVENTORY_SHA256" \
      "$INIT_INVENTORY"
  fi
  INIT_ROOT="$ROBOMME_PI05_BASE_INIT_S3"
  INIT_ARTIFACT=pi05_base_init
  INIT_MANIFEST_SHA="$ROBOMME_PI05_BASE_INIT_INVENTORY_SHA256"
else
  if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
    INIT_INVENTORY="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts blob \
      --uri "$INIT_INVENTORY_S3" --sha256 "$INIT_INVENTORY_SHA256" \
      --cache-root "$SHARED_ARTIFACT_ROOT" --name init.inventory.json)"
  else
    download_hashed "$INIT_INVENTORY_S3" "$INIT_INVENTORY_SHA256" "$INIT_INVENTORY"
  fi
  INIT_ROOT="$INIT_S3"
  INIT_ARTIFACT=pi05_h300_mg_init
  INIT_MANIFEST_SHA="$INIT_INVENTORY_SHA256"
fi

DATA="$WORK/robomme_data"
INIT="$WORK/init_ckpt"
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  if [[ "$ROBOMME_SCOPE" == single_task ]]; then
    PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts task \
      --parent-manifest "$DATA_PARENT" --task "$ROBOMME_TASK" \
      --root-s3 "$ROBOMME_DATA_S3" \
      --derived-sha256 "$ROBOMME_DATA_DERIVED_INVENTORY_SHA256" \
      --cache-root "$SHARED_ARTIFACT_ROOT" --workers 48 >"$WORK/data-cache-path" &
  else
    PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts inventory \
      --manifest "$DATA_PARENT" --artifact robomme_lerobot_all16 \
      --root-s3 "$ROBOMME_DATA_S3" \
      --manifest-sha256 "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" \
      --cache-root "$SHARED_ARTIFACT_ROOT" --workers 48 >"$WORK/data-cache-path" &
  fi
else
  if [[ "$ROBOMME_SCOPE" == single_task ]]; then
    PYTHONPATH="$CODE_DIR" python3 -m fleet.task_inventory \
      --parent-manifest "$DATA_PARENT" \
      --task "$ROBOMME_TASK" \
      --root-s3 "$ROBOMME_DATA_S3" \
      --destination "$DATA" \
      --expected-derived-sha256 "$ROBOMME_DATA_DERIVED_INVENTORY_SHA256" \
      --workers 48 &
  else
    PYTHONPATH="$CODE_DIR" python3 -m fleet.inventory \
      --manifest "$DATA_PARENT" --artifact robomme_lerobot_all16 \
      --root-s3 "$ROBOMME_DATA_S3" --destination "$DATA" --workers 48 &
  fi
fi
DATA_PID=$!
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts inventory \
    --manifest "$INIT_INVENTORY" --artifact "$INIT_ARTIFACT" \
    --root-s3 "$INIT_ROOT" --manifest-sha256 "$INIT_MANIFEST_SHA" \
    --cache-root "$SHARED_ARTIFACT_ROOT" --workers 48 >"$WORK/init-cache-path" &
else
  PYTHONPATH="$CODE_DIR" python3 -m fleet.inventory \
    --manifest "$INIT_INVENTORY" --artifact "$INIT_ARTIFACT" \
    --root-s3 "$INIT_ROOT" --destination "$INIT" --workers 48 &
fi
INIT_PID=$!

OPENPI_SHA="${OPENPI_FORK_S3%.tgz}"
OPENPI_SHA="${OPENPI_SHA##*/}"
[[ "$OPENPI_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "FATAL non-content-addressed OpenPI URI" >&2; exit 24; }
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  OPENPI_ARCHIVE="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts blob \
    --uri "$OPENPI_FORK_S3" --sha256 "$OPENPI_SHA" \
    --cache-root "$SHARED_ARTIFACT_ROOT" --name openpi.tgz)"
  OPENPI="$SHARED_ARTIFACT_ROOT/openpi/$OPENPI_SHA"
  if [[ -f "$OPENPI/.ROBOMME_OPENPI_CACHE.json" ]]; then
    PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts verify-openpi \
      --root "$OPENPI" --archive-sha256 "$OPENPI_SHA" >/dev/null
    # This is a metadata check/no-op for a healthy shared environment.  Frozen resolution repairs
    # no package and downloads nothing when the exact environment is intact.
    cd "$OPENPI"
    export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
    uv sync --frozen
    PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts verify-openpi \
      --root "$OPENPI" --archive-sha256 "$OPENPI_SHA" >/dev/null
  else
    if [[ -e "$OPENPI" ]]; then
      echo "FATAL partial shared OpenPI cache $OPENPI" >&2
      exit 24
    fi
    mkdir -p "$OPENPI"
    tar xzf "$OPENPI_ARCHIVE" -C "$OPENPI"
    cd "$OPENPI"
    export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
    uv sync --frozen
    PYTHONPATH="$CODE_DIR" python3 -B - "$OPENPI" "$OPENPI_SHA" <<'PY'
import json, pathlib, sys
from fleet.shared_artifacts import source_receipt
root, archive_sha = pathlib.Path(sys.argv[1]), sys.argv[2]
value = {
    "schema_version": 1,
    "archive_sha256": archive_sha,
    "source_receipt_sha256": source_receipt(root),
}
(root / ".ROBOMME_OPENPI_CACHE.json").write_text(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
)
PY
    PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts verify-openpi \
      --root "$OPENPI" --archive-sha256 "$OPENPI_SHA" >/dev/null
  fi
else
  download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA" "$WORK/openpi.tgz"
  OPENPI="$WORK/openpi"
  mkdir "$OPENPI"
  tar xzf "$WORK/openpi.tgz" -C "$OPENPI"
  cd "$OPENPI"
  export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
  uv sync --frozen
fi
TRAIN_PYTHONPATH="$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"
if [[ -n "${OPENPI_REQUIRED_SENTINEL:-}" ]]; then
  case "$OPENPI_REQUIRED_SENTINEL" in
    _WSM_PTRM)
      grep -Eq '^_WSM_PTRM[[:space:]]*=[[:space:]]*True$' \
        "$OPENPI/src/openpi/models/wsm_current_cond.py" || {
          echo "FATAL PTRM arm paired with an OpenPI archive lacking _WSM_PTRM=True" >&2; exit 24;
        }
      ;;
    _WSM_V4_ADVANCED)
      grep -Eq '^_WSM_PTRM[[:space:]]*=[[:space:]]*True$' \
        "$OPENPI/src/openpi/models/wsm_current_cond.py" && \
        grep -q 'wsm_cond_history_dropout' "$OPENPI/src/openpi/models/pi0_config.py" && \
        grep -q '_WORKSPACE_COMBO = {"tanh", "jepa_aux_target"}' \
          "$OPENPI/src/openpi/models/pi0_config.py" && \
        grep -q 'wsm_jepa_regularizer' "$OPENPI/src/openpi/models/pi0_config.py" || {
          echo "FATAL v4 advanced GDN source lacks dropout/PTRM/GDN+JEPA/VISReg" >&2; exit 24;
        }
      ;;
    _ROBOMME_SEQUENCE_FORCING)
      SEQUENCE_OVERLAY="$WORK/openpi-shared-tau-v1"
      PYTHONPATH="$CODE_DIR:$ROBOMME_COMPAT" "$OPENPI/.venv/bin/python" -B \
        -m training.sequence_forcing stage \
        --source-repo "$OPENPI" \
        --output-repo "$SEQUENCE_OVERLAY" \
        --source-archive-sha256 "$OPENPI_SHA" \
        >/dev/null
      TRAIN_PYTHONPATH="$SEQUENCE_OVERLAY/src:$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"
      ;;
    _WSM_GDN_JEPA)
      GDN_JEPA_OVERLAY="$WORK/openpi-gdn-jepa-config-v1"
      PYTHONPATH="$CODE_DIR:$ROBOMME_COMPAT" "$OPENPI/.venv/bin/python" -B \
        -m training.gdn_jepa_overlay stage \
        --source-repo "$OPENPI" \
        --output-repo "$GDN_JEPA_OVERLAY" \
        --source-archive-sha256 "$OPENPI_SHA" \
        >"$WORK/gdn-jepa-overlay-stage.json"
      TRAIN_PYTHONPATH="$GDN_JEPA_OVERLAY/src:$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"
      PYTHONPATH="$TRAIN_PYTHONPATH" "$OPENPI/.venv/bin/python" -B \
        -m training.gdn_jepa_overlay validate-loaded \
        --overlay-repo "$GDN_JEPA_OVERLAY" \
        >"$WORK/gdn-jepa-overlay-loaded.json"
      ;;
    _WSM_CFG_JEPA_V4)
      CFG_JEPA_OVERLAY="$WORK/openpi-v4-cfg-jepa-config-v1"
      PYTHONPATH="$CODE_DIR:$ROBOMME_COMPAT" "$OPENPI/.venv/bin/python" -B \
        -m training.cfg_jepa_overlay stage \
        --source-repo "$OPENPI" \
        --output-repo "$CFG_JEPA_OVERLAY" \
        --source-archive-sha256 "$OPENPI_SHA" \
        >"$WORK/cfg-jepa-overlay-stage.json"
      TRAIN_PYTHONPATH="$CFG_JEPA_OVERLAY/src:$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"
      PYTHONPATH="$TRAIN_PYTHONPATH" "$OPENPI/.venv/bin/python" -B \
        -m training.cfg_jepa_overlay validate-loaded \
        --overlay-repo "$CFG_JEPA_OVERLAY" \
        >"$WORK/cfg-jepa-overlay-loaded.json"
      ;;
    _WSM_HISTORY_DROPOUT)
      grep -q 'wsm_cond_history_dropout' "$OPENPI/src/openpi/models/pi0_config.py" && \
        grep -q '_history_dropout_active' "$OPENPI/src/openpi/models/pi0.py" || {
          echo "FATAL history-dropout arm paired with an incapable OpenPI archive" >&2; exit 24;
        }
      ;;
    _OFFICIAL_RECIPE_LEROBOT)
      grep -q 'max_token_len' "$OPENPI/src/openpi/models/pi0_config.py" && \
        grep -q 'weight_decay' "$OPENPI/src/openpi/training/optimizer.py" && \
        grep -q 'freeze_filter' "$OPENPI/src/openpi/training/config.py" || {
          echo "FATAL OpenPI archive cannot express the official-recipe diagnostic" >&2; exit 24;
        }
      ;;
    *) echo "FATAL unknown OpenPI architecture sentinel $OPENPI_REQUIRED_SENTINEL" >&2; exit 24 ;;
  esac
fi
PY="$OPENPI/.venv/bin/python"

wait "$DATA_PID"
wait "$INIT_PID"
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  DATA_CACHE="$(tail -n 1 "$WORK/data-cache-path")"
  INIT_CACHE="$(tail -n 1 "$WORK/init-cache-path")"
  [[ -d "$DATA_CACHE" && -d "$INIT_CACHE" ]] || {
    echo "FATAL shared data/init cache did not materialize" >&2; exit 25;
  }
  ln -s "$DATA_CACHE" "$DATA"
  ln -s "$INIT_CACHE" "$INIT"
fi

export OPENPI_DATA_HOME="$WORK/openpi_cache"
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
mkdir -p "$(dirname "$TOKENIZER")"
if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
  TOKENIZER_CACHE="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts blob \
    --uri "$PALIGEMMA_TOKENIZER_S3" --sha256 "$PALIGEMMA_TOKENIZER_SHA256" \
    --cache-root "$SHARED_ARTIFACT_ROOT" --name paligemma_tokenizer.model)"
  MATERIALIZED_TOKENIZER="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts \
    materialize-blob --source "$TOKENIZER_CACHE" --sha256 "$PALIGEMMA_TOKENIZER_SHA256" \
    --cache-home "$OPENPI_DATA_HOME" \
    --relative-path big_vision/paligemma_tokenizer.model)"
  [[ "$MATERIALIZED_TOKENIZER" == "$TOKENIZER" && -f "$TOKENIZER" && ! -L "$TOKENIZER" ]] || {
    echo "FATAL exact tokenizer did not materialize inside OPENPI_DATA_HOME" >&2; exit 25;
  }
else
  download_hashed "$PALIGEMMA_TOKENIZER_S3" "$PALIGEMMA_TOKENIZER_SHA256" "$TOKENIZER"
fi

export ROBOMME_DATA_ROOT="$DATA"
export ROBOMME_ASSETS_ROOT="$DATA/assets"
export WSM_INIT_FROM="$INIT/params"
export WSM_CKPT_BASE="$WORK/checkpoints"
export WSM_EXP_NAME="$ROBOMME_RUN_ID"
export WSM_STAGE_Q_ALLOW_RUN=1
export WSM_WSM_POLICY_ALLOW_RUN=1
export WSM_EXPECTED_JAX_DEVICES=8
export WSM_NUM_WORKERS=32
export WANDB_PROJECT="${WANDB_PROJECT:-wsm-robomme}"
[[ -d "$WSM_INIT_FROM" ]] || { echo "FATAL init params absent" >&2; exit 25; }
if [[ -n "${ROBOMME_WORKSPACE_INDEX_S3:-}" ]]; then
  [[ "$ROBOMME_SCOPE" == multitask ]] || { echo "FATAL all-16 workspace index on single-task run" >&2; exit 25; }
  [[ -z "${ROBOMME_WORKSPACE_S3:-}" && -z "${ROBOMME_SUPERVISION_S3:-}" ]] || {
    echo "FATAL all-16 index cannot be combined with single-task workspace artifacts" >&2; exit 25;
  }
  : "${ROBOMME_WORKSPACE_INDEX_SHA256:?}"
  INDEX="$WORK/workspace-index.json"
  download_hashed "$ROBOMME_WORKSPACE_INDEX_S3" "$ROBOMME_WORKSPACE_INDEX_SHA256" "$INDEX"
  BUNDLE_ARGS=(
    --index "$INDEX"
    --index-sha256 "$ROBOMME_WORKSPACE_INDEX_SHA256"
    --workspace-root "$WORK/workspace"
    --workers 8
  )
  if [[ "${ROBOMME_REQUIRE_SUPERVISION:-0}" == 1 ]]; then
    BUNDLE_ARGS+=(--require-supervision --supervision-root "$WORK/supervision")
    export ROBOMME_SALIENT_ROOT="$WORK/supervision"
  fi
  PYTHONPATH="$CODE_DIR" python3 -m fleet.workspace_bundle "${BUNDLE_ARGS[@]}"
  export ROBOMME_WORKSPACE_ROOT="$WORK/workspace"
  export ROBOMME_WORKSPACE_INDEX="$INDEX"
elif [[ -n "${ROBOMME_WORKSPACE_S3:-}" ]]; then
  : "${ROBOMME_WORKSPACE_MANIFEST_SHA256:?}"
  mkdir -p "$WORK/workspace/$ROBOMME_TASK"
  if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
    rmdir "$WORK/workspace/$ROBOMME_TASK"
    WORKSPACE_CACHE="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts tree \
      --uri "$ROBOMME_WORKSPACE_S3" \
      --manifest-sha256 "$ROBOMME_WORKSPACE_MANIFEST_SHA256" \
      --cache-root "$SHARED_ARTIFACT_ROOT" --category workspace)"
    ln -s "$WORKSPACE_CACHE" "$WORK/workspace/$ROBOMME_TASK"
  else
    aws s3 sync "$ROBOMME_WORKSPACE_S3" "$WORK/workspace/$ROBOMME_TASK" --only-show-errors
  fi
  verify_manifest_sha \
    "$WORK/workspace/$ROBOMME_TASK/MANIFEST.json" "$ROBOMME_WORKSPACE_MANIFEST_SHA256"
  export ROBOMME_WORKSPACE_ROOT="$WORK/workspace"
fi
if [[ -n "${ROBOMME_SUPERVISION_S3:-}" ]]; then
  : "${ROBOMME_SUPERVISION_MANIFEST_SHA256:?}"
  mkdir -p "$WORK/supervision/$ROBOMME_TASK"
  if [[ -n "$SHARED_ARTIFACT_ROOT" ]]; then
    rmdir "$WORK/supervision/$ROBOMME_TASK"
    SUPERVISION_CACHE="$(PYTHONPATH="$CODE_DIR" python3 -B -m fleet.shared_artifacts tree \
      --uri "$ROBOMME_SUPERVISION_S3" \
      --manifest-sha256 "$ROBOMME_SUPERVISION_MANIFEST_SHA256" \
      --cache-root "$SHARED_ARTIFACT_ROOT" --category supervision)"
    ln -s "$SUPERVISION_CACHE" "$WORK/supervision/$ROBOMME_TASK"
  else
    aws s3 sync "$ROBOMME_SUPERVISION_S3" "$WORK/supervision/$ROBOMME_TASK" --only-show-errors
  fi
  verify_manifest_sha \
    "$WORK/supervision/$ROBOMME_TASK/MANIFEST.json" "$ROBOMME_SUPERVISION_MANIFEST_SHA256"
  export ROBOMME_SALIENT_ROOT="$WORK/supervision"
fi

"$PY" -c 'import jax, flax, orbax.checkpoint as o; print("jax", jax.__version__, "devices", jax.devices(), "flax", flax.__version__, "orbax", o.__version__)'
cd "$CODE_DIR"
PYTHONPATH="$TRAIN_PYTHONPATH" "$PY" -m training.train --arm "$ROBOMME_ARM" --dry-run
PYTHONPATH="$TRAIN_PYTHONPATH" "$PY" -m gpu.run_resumable \
  --arm "$ROBOMME_ARM" \
  --train-python "$PY" \
  --checkpoint-base "$WSM_CKPT_BASE" \
  --s3-run-root "$OUTPUT_S3" \
  --final-step "$ROBOMME_FINAL_STEP" \
  --poll-seconds 30

# One finalized generation -> deploy-only params+assets tree + _DEPLOY_COMPLETE.json receipt +
# content-addressed tree manifest.  Shared by the official-recipe diagnostic (60000/70000/79999) and
# the A19 v4_70k checkpoint-maturity recipe (10000..60000 + 69999).  Orbax keeps only the newest
# local generation, so a milestone is restored from its retained remote generation while exporting.
RECEIPTS="$WORK/scientific-checkpoints.jsonl"
python3 - "$RECEIPTS" <<'PY'
import pathlib, sys
pathlib.Path(sys.argv[1]).write_text("", encoding="utf-8")
PY

deploy_recipe_step() {
  local step="$1"
  local source="$WSM_CKPT_BASE/pi05_robomme_${ROBOMME_ARM}/$WSM_EXP_NAME/$step"
  local checkpoint_uri="${OUTPUT_S3%/}/deploy/$step"
  local upload_marker="$WORK/upload-$step.json"
  local tree="$WORK/checkpoint-tree-$step.json"
  local tree_sha tree_s3 deploy_complete deploy_complete_s3 existing restored=0
  aws s3 cp \
    "${OUTPUT_S3%/}/steps/$step/_UPLOAD_COMPLETE.json" \
    "$upload_marker" --only-show-errors
  python3 - "$upload_marker" "$step" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if int(value.get("step", -1)) != int(sys.argv[2]):
    raise SystemExit(f"remote upload marker step mismatch: {value.get('step')} != {sys.argv[2]}")
PY
  if [[ ! -d "$source/params" || ! -d "$source/assets" || ! -f "$source/_CHECKPOINT_METADATA" ]]; then
    # Orbax keeps only the newest local generation.  Restore a milestone only while exporting its
    # deploy tree; the remote finalized generation is already the resume/scientific authority.
    source="$WORK/scientific-step-$step"
    mkdir -p "$source"
    aws s3 sync "${OUTPUT_S3%/}/steps/$step" "$source" --only-show-errors
    restored=1
  fi
  [[ -d "$source/params" && -d "$source/assets" && -f "$source/_CHECKPOINT_METADATA" ]] || {
    echo "FATAL required scientific checkpoint missing at $source" >&2; return 26;
  }
  PYTHONPATH="$CODE_DIR" python3 - "$source" "$upload_marker" <<'PY'
import json, pathlib, sys
from gpu.checkpoint_transport import tree_summary
expected = json.load(open(sys.argv[2], encoding="utf-8")).get("tree")
actual = tree_summary(pathlib.Path(sys.argv[1]))
if actual != expected:
    raise SystemExit(f"restored scientific checkpoint receipt mismatch: {actual} != {expected}")
PY
  tree_sha="$(PYTHONPATH="$CODE_DIR" python3 -m fleet.checkpoint \
    --root "$source" --uri "$checkpoint_uri" --output "$tree")"
  tree_s3="${CHECKPOINT_TREE_MANIFEST_ROOT%/}/step-$step/$tree_sha.json"
  deploy_complete="$WORK/deploy-complete-$step.json"
  python3 - "$deploy_complete" "$step" "$checkpoint_uri" "$tree_s3" "$tree_sha" <<'PY'
import json, os, sys
path, step, checkpoint_uri, tree_uri, tree_sha = sys.argv[1:]
value = {
    "schema_version": 1,
    "kind": "robomme_gpu_deploy_checkpoint_complete",
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "attempt_id": os.environ["ROBOMME_ATTEMPT_ID"],
    "scientific_spec_sha256": os.environ["ROBOMME_SCIENTIFIC_SPEC_SHA256"],
    "step": int(step),
    "checkpoint_uri": checkpoint_uri,
    "tree_manifest_uri": tree_uri,
    "tree_manifest_sha256": tree_sha,
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
}
label = os.environ.get("ROBOMME_RECIPE_LABEL")
if label:
    value["diagnostic_label"] = label
recipe = os.environ.get("ROBOMME_RECIPE")
if recipe:
    value["recipe"] = recipe
open(path, "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
  deploy_complete_s3="${checkpoint_uri%/}/_DEPLOY_COMPLETE.json"
  existing="$WORK/existing-deploy-complete-$step.json"
  if aws s3 cp "$deploy_complete_s3" "$existing" --only-show-errors 2>/dev/null; then
    compare_attempt_receipts "$deploy_complete" "$existing" || {
      echo "FATAL immutable deploy-checkpoint collision $deploy_complete_s3" >&2; return 27;
    }
  else
    aws s3 sync "$source/params" "${checkpoint_uri%/}/params" \
      --only-show-errors --no-follow-symlinks --delete
    aws s3 sync "$source/assets" "${checkpoint_uri%/}/assets" \
      --only-show-errors --no-follow-symlinks --delete
    publish_attempt_receipt_once "$deploy_complete" "$deploy_complete_s3"
  fi
  publish_once "$tree" "$tree_s3"
  python3 - "$RECEIPTS" "$step" "$checkpoint_uri" "$tree_s3" "$tree_sha" <<'PY'
import json, sys
path, step, checkpoint_uri, tree_uri, tree_sha = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "step": int(step),
        "checkpoint_uri": checkpoint_uri,
        "tree_manifest_uri": tree_uri,
        "tree_manifest_sha256": tree_sha,
    }, sort_keys=True, separators=(",", ":")) + "\n")
PY
  if [[ "$restored" == 1 ]]; then
    python3 - "$source" <<'PY'
import pathlib, shutil, sys
path = pathlib.Path(sys.argv[1])
if path.parent.name != "work" or not path.name.startswith("scientific-step-"):
    raise SystemExit(f"refusing unsafe milestone cleanup: {path}")
shutil.rmtree(path)
PY
  fi
}

if [[ "$ROBOMME_ARM" == official_recipe_lerobot ]]; then
  for step in 60000 70000 79999; do
    deploy_recipe_step "$step"
  done
  COMPLETE="$WORK/completion.json"
  python3 - "$RECEIPTS" "$COMPLETE" <<'PY'
import json, os, sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if [record["step"] for record in records] != [60000, 70000, 79999]:
    raise SystemExit(f"scientific checkpoint receipt mismatch: {records}")
value = {
    "schema_version": 2,
    "kind": "robomme_gpu_diagnostic_checkpoint_set_complete",
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "attempt_id": os.environ["ROBOMME_ATTEMPT_ID"],
    "scientific_spec_sha256": os.environ["ROBOMME_SCIENTIFIC_SPEC_SHA256"],
    "diagnostic_label": os.environ["ROBOMME_RECIPE_LABEL"],
    "claim": "RECIPE_MATCHED_NOT_EXACT_SOURCE_OR_DATA_REPRODUCTION",
    "steps": [60000, 70000, 79999],
    "checkpoints": records,
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
}
open(sys.argv[2], "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
  publish_attempt_receipt_once "$COMPLETE" "$COMPLETION_CLAIM_S3"
  # Full optimizer states are useful only for resume.  The three deploy-only parameter trees above
  # are the durable scientific artifacts, so remove recovery generations after sealing the set.
  aws s3 rm "${OUTPUT_S3%/}/steps" --recursive --only-show-errors || \
    echo "WARNING recovery-step cleanup failed after sealed completion" >&2
  aws s3 rm "${OUTPUT_S3%/}/LATEST.json" --only-show-errors || \
    echo "WARNING recovery-pointer cleanup failed after sealed completion" >&2
  echo "ROBOMME GPU TRAINING COMPLETE run_id=$ROBOMME_RUN_ID steps=60000,70000,79999"
  exit 0
fi

if [[ "${ROBOMME_RECIPE:-}" == v4_70k ]]; then
  # A19 checkpoint-maturity recipe: every retained milestone becomes an immutable deploy-only tree.
  IFS=, read -r -a MILESTONE_STEPS <<<"$ROBOMME_SUCCESS_CHECKPOINT_MILESTONES"
  MILESTONE_STEPS+=("$ROBOMME_FINAL_STEP")
  for step in "${MILESTONE_STEPS[@]}"; do
    deploy_recipe_step "$step"
  done
  COMPLETE="$WORK/completion.json"
  python3 - "$RECEIPTS" "$COMPLETE" "${MILESTONE_STEPS[@]}" <<'PY'
import json, os, sys
receipts, output, *expected = sys.argv[1:]
expected = [int(step) for step in expected]
final_step = int(os.environ["ROBOMME_FINAL_STEP"])
if expected != sorted(set(expected)) or expected[-1] != final_step or len(expected) < 2:
    raise SystemExit(f"milestone set must be strictly increasing and end at the final step: {expected}")
records = [json.loads(line) for line in open(receipts, encoding="utf-8") if line.strip()]
if [record["step"] for record in records] != expected:
    raise SystemExit(f"milestone checkpoint receipt mismatch: {records} != {expected}")
for record in records:
    if record["checkpoint_uri"] != f"{os.environ['OUTPUT_S3'].rstrip('/')}/deploy/{record['step']}":
        raise SystemExit(f"milestone receipt does not address deploy/<step>: {record}")
value = {
    "schema_version": 1,
    "kind": "robomme_gpu_milestone_checkpoint_set_complete",
    "recipe": os.environ["ROBOMME_RECIPE"],
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "attempt_id": os.environ["ROBOMME_ATTEMPT_ID"],
    "scientific_spec_sha256": os.environ["ROBOMME_SCIENTIFIC_SPEC_SHA256"],
    "final_step": final_step,
    "steps": expected,
    "checkpoints": records,
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
}
open(output, "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
  publish_attempt_receipt_once "$COMPLETE" "$COMPLETION_CLAIM_S3"
  # Every retained milestone is now a sealed deploy-only tree; the full optimizer generations were
  # only ever resume state.  They are pruned ONLY here, after every deploy and the completion claim
  # succeeded (set -e aborts earlier on any deploy failure, leaving steps/ intact for a retry).
  aws s3 rm "${OUTPUT_S3%/}/steps" --recursive --only-show-errors || \
    echo "WARNING recovery-step cleanup failed after sealed completion" >&2
  aws s3 rm "${OUTPUT_S3%/}/LATEST.json" --only-show-errors || \
    echo "WARNING recovery-pointer cleanup failed after sealed completion" >&2
  echo "ROBOMME GPU TRAINING COMPLETE run_id=$ROBOMME_RUN_ID recipe=v4_70k steps=$(IFS=,; echo "${MILESTONE_STEPS[*]}")"
  exit 0
fi

FINAL_DIR="$WSM_CKPT_BASE/pi05_robomme_${ROBOMME_ARM}/$WSM_EXP_NAME/$ROBOMME_FINAL_STEP"
[[ -d "$FINAL_DIR/params" && -f "$FINAL_DIR/_CHECKPOINT_METADATA" ]] || {
  echo "FATAL finalized params missing at $FINAL_DIR" >&2; exit 26;
}
aws s3 cp \
  "${OUTPUT_S3%/}/steps/$ROBOMME_FINAL_STEP/_UPLOAD_COMPLETE.json" \
  "$WORK/final-upload.json" --only-show-errors

# Keep full optimizer state only while the run is recoverable.  The immutable scientific output is
# a deploy-only tree; this prevents a broad single-task sweep from retaining ~32 GB of optimizer
# state per successful cell.
export CHECKPOINT_URI="${OUTPUT_S3%/}/deploy/$ROBOMME_FINAL_STEP"
TREE="$WORK/checkpoint_tree.json"
TREE_SHA="$(PYTHONPATH="$CODE_DIR" python3 -m fleet.checkpoint \
  --root "$FINAL_DIR" --uri "$CHECKPOINT_URI" --output "$TREE")"
export CHECKPOINT_TREE_MANIFEST_S3="${CHECKPOINT_TREE_MANIFEST_ROOT%/}/$TREE_SHA.json"

DEPLOY_COMPLETE="$WORK/deploy-complete.json"
python3 - "$DEPLOY_COMPLETE" "$TREE_SHA" <<'PY'
import json, os, sys
value = {
    "schema_version": 1,
    "kind": "robomme_gpu_deploy_checkpoint_complete",
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "attempt_id": os.environ["ROBOMME_ATTEMPT_ID"],
    "scientific_spec_sha256": os.environ["ROBOMME_SCIENTIFIC_SPEC_SHA256"],
    "step": int(os.environ["ROBOMME_FINAL_STEP"]),
    "checkpoint_uri": os.environ["CHECKPOINT_URI"],
    "tree_manifest_sha256": sys.argv[2],
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
}
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
DEPLOY_COMPLETE_S3="${CHECKPOINT_URI%/}/_DEPLOY_COMPLETE.json"
if aws s3 cp "$DEPLOY_COMPLETE_S3" "$WORK/existing-deploy-complete.json" \
    --only-show-errors 2>/dev/null; then
  compare_attempt_receipts "$DEPLOY_COMPLETE" "$WORK/existing-deploy-complete.json" || {
    echo "FATAL immutable deploy-checkpoint collision $DEPLOY_COMPLETE_S3" >&2; exit 27;
  }
else
  aws s3 sync "$FINAL_DIR/params" "${CHECKPOINT_URI%/}/params" \
    --only-show-errors --no-follow-symlinks --delete
  aws s3 sync "$FINAL_DIR/assets" "${CHECKPOINT_URI%/}/assets" \
    --only-show-errors --no-follow-symlinks --delete
  publish_attempt_receipt_once "$DEPLOY_COMPLETE" "$DEPLOY_COMPLETE_S3"
fi
publish_once "$TREE" "$CHECKPOINT_TREE_MANIFEST_S3"

COMPLETE="$WORK/completion.json"
python3 - "$COMPLETE" "$TREE_SHA" <<'PY'
import json, os, sys
value = {
    "schema_version": 1,
    "kind": "robomme_gpu_checkpoint_complete",
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "attempt_id": os.environ["ROBOMME_ATTEMPT_ID"],
    "scientific_spec_sha256": os.environ["ROBOMME_SCIENTIFIC_SPEC_SHA256"],
    "step": int(os.environ["ROBOMME_FINAL_STEP"]),
    "checkpoint_uri": os.environ["CHECKPOINT_URI"],
    "tree_manifest_uri": os.environ["CHECKPOINT_TREE_MANIFEST_S3"],
    "tree_manifest_sha256": sys.argv[2],
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
}
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
publish_attempt_receipt_once "$COMPLETE" "$COMPLETION_CLAIM_S3"
# The completion claim now points only at the verified deployment tree.  Recovery state is no
# longer scientifically meaningful; cleanup is deliberately best-effort because a transient S3
# delete error must not invalidate an already sealed model.
aws s3 rm "${OUTPUT_S3%/}/steps" --recursive --only-show-errors || \
  echo "WARNING recovery-step cleanup failed after sealed completion" >&2
aws s3 rm "${OUTPUT_S3%/}/LATEST.json" --only-show-errors || \
  echo "WARNING recovery-pointer cleanup failed after sealed completion" >&2
echo "ROBOMME GPU TRAINING COMPLETE run_id=$ROBOMME_RUN_ID step=$ROBOMME_FINAL_STEP"
