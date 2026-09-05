#!/bin/bash
# Push the laptop-only ReMemBench tri-integration working tree + the setup script to nagababa.
# The branch is never pushed to a remote, so rsync is the only transport. Assets and demo
# hdf5s are excluded: the VPN link is ~1.5MB/s, and the box downloads both from source at
# two orders of magnitude more bandwidth (see setup_remembench_nagababa.sh stages 4 and 5).
set -euo pipefail
PEM=${PEM:-/home/sarveshp/Research/TRI/nagababa.pem}
BOX=${BOX:-ubuntu@10.242.9.112}
SRC=${SRC:-/home/sarveshp/Research/TRI/ReMemBench}
SSH="ssh -i $PEM -o StrictHostKeyChecking=no"

$SSH "$BOX" 'mkdir -p /data/work /data2/logs /data/tmp'
rsync -az --info=progress2 -e "$SSH" \
  --exclude 'robocasa/models/assets/' \
  --exclude '**/__pycache__/' \
  --exclude '*.hdf5' \
  "$SRC/" "$BOX:/data/work/ReMemBench/"
# The git-TRACKED files under models/assets (arenas/*.xml, scenes/*.yaml, some fixtures) are NOT
# in the UT Box zips -- excluding the whole assets dir above drops them and every env build fails
# with a missing empty_kitchen_arena.xml. Ship exactly the tracked subset (152 small files).
git -C "$SRC" ls-files robocasa/models/assets \
  | sed 's|^robocasa/models/assets/||' > /tmp/rb_tracked_assets.txt
$SSH "$BOX" 'mkdir -p /data/work/ReMemBench/robocasa/models/assets'
rsync -az --files-from=/tmp/rb_tracked_assets.txt -e "$SSH" \
  "$SRC/robocasa/models/assets/" "$BOX:/data/work/ReMemBench/robocasa/models/assets/"

rsync -az -e "$SSH" \
  "$(dirname "$0")/setup_remembench_nagababa.sh" "$BOX:/data/setup_remembench_nagababa.sh"
echo "pushed ReMemBench source + tracked assets + setup script"
