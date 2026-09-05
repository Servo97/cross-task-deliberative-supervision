#!/usr/bin/env bash
# Orchestrate WSM-base (regime vla_wsm) training: action loss + salient-keyframe-patch aux head.
# Selects pretrain vs target-finetune via --phase and routes to the right backbone driver.
# Drivers: vla_training/train/train_wsm_base/finetune_{groot_17,pi_05}_with_wsm.py
# Recipes (YAML): scripts/configs/train/wsm/<x>.yaml  (NOTE: WSM configs TBD).
# Governs: internal_planning_and_todos/04_wsm_roadmap.md ("WSM base").
#
#   scripts/train/vla_wsm/train.sh --backbone groot_17 --phase target_finetune \
#       --config scripts/configs/train/wsm/groot_wsm_base.yaml
#
# TODO (SageMaker Batch submit — see internal_planning_and_todos/03_infra_and_sagemaker.md):
#   - Submit to AWS Batch TrainingQueue (HUMANOID_QUEUE, ml.p5.48xlarge 8xH100), --priority 1.
#   - Override the baked image entry via SAGEMAKER_PROGRAM; pass recipe + WSM-head knobs as env vars.
#   - GR00T: checkpoint_s3_uri live-sync; pi0.5: orbax->S3 background loop inside the entry.
set -euo pipefail

REGIME="vla_wsm"
BACKBONE=""                                   # groot_17 | pi_05
PHASE="target_finetune"                       # pretrain | target_finetune
CONFIG=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

usage() { echo "usage: $0 --backbone {groot_17|pi_05} --phase {pretrain|target_finetune} --config <yaml>"; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) BACKBONE="$2"; shift 2 ;;
    --phase)    PHASE="$2";    shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done

[[ -n "$BACKBONE" ]] || usage
case "$PHASE" in pretrain|target_finetune) ;; *) echo "bad --phase: $PHASE"; usage ;; esac

# WSM-base couples the head during (target-)finetune; one driver per backbone.
DRIVER="$REPO_ROOT/vla_training/train/train_wsm_base/finetune_${BACKBONE}_with_wsm.py"
CONFIG="${CONFIG:-$REPO_ROOT/scripts/configs/train/wsm/${BACKBONE}_wsm_base.yaml}"   # TBD

echo "=========================================================="
echo " plan: regime=$REGIME backbone=$BACKBONE phase=$PHASE"
echo "   driver : $DRIVER"
echo "   config : $CONFIG   (WSM configs TBD)"
echo "=========================================================="

[[ -f "$DRIVER" ]] || { echo "ERROR: driver not found: $DRIVER"; exit 3; }

# TODO: replace local dispatch with the SageMaker Batch submit (03_infra_and_sagemaker.md).
python "$DRIVER" --config "$CONFIG"
