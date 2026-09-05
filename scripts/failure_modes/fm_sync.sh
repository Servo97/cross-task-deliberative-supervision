#!/bin/bash
# Pull finished cells home. Runs on the WORKSTATION, incrementally — call it repeatedly while
# the box is still producing; rsync only moves what is new.
#
# Metrics/CSV/provenance are pulled first and always, so even mid-run there is a complete,
# readable set of numbers on the workstation with the videos filling in behind them.
#
# Measured 2026-08-11: the box->workstation link runs at ~18.7 MB/s, not the ~0.6 MB/s the
# plan budgeted for. The whole video set (~3-4 GB) therefore lands in minutes, not hours,
# and the CRF-26 choice for rollouts is comfort rather than necessity.
#
#   bash fm_sync.sh              # everything available so far
#   bash fm_sync.sh --no-videos  # numbers only, seconds not hours
set -uo pipefail
PEM=/home/sarveshp/Research/TRI/nagababa.pem
BOX=ubuntu@10.242.9.112
REMOTE=/data2/failure_modes
LOCAL=/home/sarveshp/Research/TRI/wsm_data/failure_modes
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $PEM"
WITH_VIDEOS=1
[[ "${1:-}" == "--no-videos" ]] && WITH_VIDEOS=0

mkdir -p "$LOCAL/_provenance"

echo "[sync] metrics + provenance"
rsync -az --info=stats1 -e "$RSH" "$BOX:$REMOTE/out/" "$LOCAL/" || exit 1
rsync -az -e "$RSH" "$BOX:$REMOTE/cells/" "$LOCAL/_provenance/cells/" || true
rsync -az -e "$RSH" --include='*/' --include='fm_*_manifest.json' --exclude='*' \
  "$BOX:$REMOTE/manifests/" "$LOCAL/_provenance/manifests/" || true

if [[ "$WITH_VIDEOS" == 1 ]]; then
  REMOTE_SIZE=$($RSH "$BOX" "du -sb $REMOTE/videos 2>/dev/null | cut -f1" || echo 0)
  echo "[sync] videos: $((REMOTE_SIZE / 1000000)) MB on the box (already-transferred files are skipped)"
  rsync -a --info=progress2 --partial -e "$RSH" "$BOX:$REMOTE/videos/" "$LOCAL/" || exit 1
fi

echo "[sync] local tree:"
du -sh "$LOCAL" 2>/dev/null
find "$LOCAL" -name '*.mp4' | wc -l | xargs echo "  videos:"
find "$LOCAL" -name '*.json' -not -path '*_provenance*' | wc -l | xargs echo "  metric records:"
