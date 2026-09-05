#!/bin/bash
# Collect finished ReMemBench box-tier arm results: box -> workstation -> S3 (create-once),
# then render the combined arms x categories table.
#
#   bash scripts/remembench/collect_remembench_evals.sh [run_id ...]
#
# With no arguments it collects all four study arms. Arms whose COMPLETED marker is absent are
# skipped with a note, so this is safe to re-run while later arms are still in flight.
set -uo pipefail
PEM=${PEM:-/home/sarveshp/Research/TRI/nagababa.pem}
BOX=${BOX:-ubuntu@10.242.9.112}
SSH="ssh -i $PEM -o StrictHostKeyChecking=no"
LOCAL_ROOT=${LOCAL_ROOT:-/home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/remembench_evals}
BUCKET=sagemaker-us-west-2-141701954645
PREFIX=sarvesh.patil/wsm_robocasa/studies/long_context_v1/evals/remembench
export AWS_DEFAULT_REGION=us-west-2

ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(s0-9e47bc75062b23e9 s3-5e942af9f0718e3a s1-bff55c66cffc3360 s1-be5d198305786f3e
        s1-9b508f6799c3d128 s1-3b9f9229b3ea51b2 s1-9b28670a6f0c57d9 s1-8edacfb5b7739576 s1-a781d6e251d1e87a)
fi
mkdir -p "$LOCAL_ROOT"

for RID in "${ARMS[@]}"; do
  if ! $SSH "$BOX" "test -e /data/work/remembench_evals/$RID/COMPLETED" 2>/dev/null; then
    echo "[skip] $RID: no COMPLETED marker on the box yet"
    continue
  fi
  mkdir -p "$LOCAL_ROOT/$RID"
  for f in results.json arm.json; do
    scp -q -i "$PEM" "$BOX:/data/work/remembench_evals/$RID/$f" "$LOCAL_ROOT/$RID/$f" \
      || { echo "[warn] $RID: could not fetch $f"; continue; }
  done
  [[ -f "$LOCAL_ROOT/$RID/results.json" ]] || { echo "[warn] $RID: no results.json"; continue; }
  for f in results.json arm.json; do
    [[ -f "$LOCAL_ROOT/$RID/$f" ]] || continue
    # Create-once: a 412 PreconditionFailed means it is already published; never overwrite.
    if aws s3api put-object --bucket "$BUCKET" --key "$PREFIX/$RID/$f" \
         --body "$LOCAL_ROOT/$RID/$f" --if-none-match '*' \
         --checksum-algorithm SHA256 >/dev/null 2>&1; then
      echo "[s3 ] $RID/$f published"
    else
      echo "[s3 ] $RID/$f already present (create-once preserved)"
    fi
  done
  echo "[ok ] $RID collected"
done

python3 "$(dirname "$0")/summarize_remembench_arms.py" --root "$LOCAL_ROOT"
