#!/usr/bin/env bash
# Orchestrate base-VLA (regime vla_base / R0) training: official RoboCasa protocol, no WSM.
# Selects pretrain vs target-finetune via --phase and routes to the right backbone driver.
# Drivers: vla_training/train/train_base/{pretrain,finetune}_{groot_17,pi_05}.py
# Recipes (YAML): scripts/configs/train/<x>.yaml — DEFAULT to the official RoboCasa recipe.
# Governs: internal_planning_and_todos/01_robocasa_protocol_and_recipes.md + 04_wsm_roadmap.md (R0).
#
#   scripts/train/vla_base/train.sh --backbone groot_17 --phase pretrain \
#       --config scripts/configs/train/groot_pretrain.yaml
#
# TODO (SageMaker Batch submit — see internal_planning_and_todos/03_infra_and_sagemaker.md):
#   - Submit to AWS Batch TrainingQueue (HUMANOID_QUEUE, ml.p5.48xlarge 8xH100), --priority 1.
#   - Override the baked image entry via SAGEMAKER_PROGRAM; pass recipe knobs as env vars.
#   - GR00T: checkpoint_s3_uri live-sync; pi0.5: orbax->S3 background loop inside the entry.
set -euo pipefail

REGIME="vla_base"
BACKBONE=""                                   # groot_17 | pi_05
PHASE="pretrain"                              # pretrain | target_finetune
CONFIG=""
DRY=""                                        # --dry-run -> build/log GroupedSoup only
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

usage() { echo "usage: $0 --backbone {groot_17|pi_05} --phase {pretrain|target_finetune} [--config <yaml>] [--dry-run]"; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) BACKBONE="$2"; shift 2 ;;
    --phase)    PHASE="$2";    shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    --dry-run)  DRY="--dry-run"; shift ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done

[[ -n "$BACKBONE" ]] || usage
case "$PHASE" in pretrain|target_finetune) ;; *) echo "bad --phase: $PHASE"; usage ;; esac

case "$PHASE" in
  pretrain)        DRIVER_STEM="pretrain" ;;
  target_finetune) DRIVER_STEM="finetune" ;;
esac
DRIVER="$REPO_ROOT/vla_training/train/train_base/${DRIVER_STEM}_${BACKBONE}.py"
# Default config is resolved INSIDE the driver (utils.default_config_path), so the
# pi_05<->pi05 / finetune<->target_finetune token mapping lives in exactly one place.
# Only pass --config when the user gave one explicitly.

echo "=========================================================="
echo " plan: regime=$REGIME backbone=$BACKBONE phase=$PHASE"
echo "   driver : $DRIVER"
echo "   config : ${CONFIG:-(driver default)} ${DRY}"
echo "=========================================================="

[[ -f "$DRIVER" ]] || { echo "ERROR: driver not found: $DRIVER"; exit 3; }

# Driver imports the wsmv2 packages (utils, vla_training) from the repo root.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# TODO: replace local dispatch with the SageMaker Batch submit (03_infra_and_sagemaker.md).
python "$DRIVER" ${CONFIG:+--config "$CONFIG"} ${DRY}
