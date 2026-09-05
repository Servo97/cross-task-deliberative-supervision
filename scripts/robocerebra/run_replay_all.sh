#!/usr/bin/env bash
# Replay every downloaded RoboCerebra training case in parallel, then assemble one LeRobot tree.
#
# Sim side runs in the LIBERO venv (robosuite 1.4.0 / mujoco 2.3.2), which cannot share a process
# with the robocasa robosuite 1.5.2 fork. The finalize + norm-stats steps run in the openpi venv.
# Measured single-process throughput on one RTX 5090: ~4.2k raw frames/s (~780 rendered fps), so
# the full ~3.0M-frame trainset is roughly 12 minutes of wall clock at 1 worker and a couple of
# minutes fanned out.
set -euo pipefail

WSM_DATA="${WSM_DATA:-/home/sarveshp/Research/TRI/wsm_data/robocerebra}"
OPENPI="${OPENPI:-/home/sarveshp/Research/robocasa_openpi}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS="${WORKERS:-8}"
GPUS="${GPUS:-0,1}"
PAYLOADS="${PAYLOADS:-$WSM_DATA/payloads}"
DATASET="${DATASET:-$WSM_DATA/lerobot_home/wsmv2/robocerebra_train}"
REPO_ID="${REPO_ID:-wsmv2/robocerebra_train}"

mapfile -t CASES < <(find "$WSM_DATA/RoboCerebra/RoboCerebra_trainset" -mindepth 2 -maxdepth 2 \
  -type d -exec test -e '{}/demo.hdf5' \; -print | sort)
echo "cases with a demo.hdf5: ${#CASES[@]}"

export WSM_DATA HERE PAYLOADS GPUS
printf '%s\n' "${CASES[@]}" | xargs -P "$WORKERS" -n 8 bash -c '
  IFS="," read -r -a gpus <<< "$GPUS"
  gpu=${gpus[$(( RANDOM % ${#gpus[@]} ))]}
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl \
  LIBERO_CONFIG_PATH="$WSM_DATA/libero_config" PYTHONPATH="$WSM_DATA/code/LIBERO" \
  "$WSM_DATA/venv_sim/bin/python" "$HERE/replay_shard.py" --skip-existing --out "$PAYLOADS" --cases "$@"
' _ \
  || { echo "one or more shards failed; see $PAYLOADS/_failures.json"; exit 1; }

"$OPENPI/.venv/bin/python" "$HERE/finalize_lerobot.py" --payloads "$PAYLOADS" --out "$DATASET" --move-videos

HF_LEROBOT_HOME="$WSM_DATA/lerobot_home" JAX_PLATFORMS=cpu \
PYTHONPATH="$HERE/_worker_shim:$HERE:/home/sarveshp/Research/robocasa:/home/sarveshp/Research/robosuite" \
"$OPENPI/.venv/bin/python" "$HERE/compute_norm_stats.py" \
  --repo-id "$REPO_ID" --assets-dir "$WSM_DATA/assets" --num-workers 8 \
  --reference "$WSM_DATA/meta/pi05_libero_norm_stats.json"

"$OPENPI/.venv/bin/python" "$HERE/build_inventory.py" --root "$DATASET" \
  --artifact robocerebra_train_v1 --out "$WSM_DATA/manifests"
