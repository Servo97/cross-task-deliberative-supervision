#!/bin/bash
# nagababa canary-tier env — mirrors robocasa_eval_entry.sh env sections (same pins/venvs).
# Idempotent: safe to re-run; a creds-less first pass banks the non-S3 work (clones + simenv),
# then re-run after ~/.aws/credentials exists to finish (assets, openpi fork, wsmv2).
# Launch on the box:  tmux new -d -s study 'bash /data/setup_env_nagababa.sh 2>&1 | tee -a /data/setup.log'
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
# MUJOCO_GL deliberately NOT set during build: with egl set, any mujoco import (e.g. inside
# install_robocasa_deps.sh's sanity import) demands PyOpenGL before it's installed.
export HF_HOME=/data/cache/hf UV_CACHE_DIR=/data/cache/uv
unset MUJOCO_GL || true
# Root disk is 8GB: wheel staging in /tmp ENOSPCs (burned pass-1) — stage on /data.
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"
WORK=/data/work; mkdir -p "$WORK"; cd "$WORK"

STUDY_S3="s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"
ASSETS_S3="${ASSETS_S3:-s3://sagemaker-us-west-2-124224456861/sarvesh.patil/wsm_robocasa/assets/models_assets}"
# Pins — keep in lockstep with internal_training/robocasa_eval_entry.sh
ROBOSUITE_SHA="85abee228d1c43ab1939bce33028099945d453b4"
ROBOCASA_SHA="be22d659b02db8f6d7f3a3c3edc742934fdcbaae"
LEROBOT_REV="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
# Current archive prefixes (filename = bytes-sha256; verified after download)
OPENPI_SHA_PREFIX="${OPENPI_SHA_PREFIX:-f3957857}"
WSMV2_SHA_PREFIX="${WSMV2_SHA_PREFIX:-d07ea867}"

# ---- 0. system libs for EGL/OSMesa rendering (same set as the node entry) ----
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libosmesa6 libglu1-mesa

# ---- 1. Sim stack (no S3 needed) ----
[[ -d robosuite ]] || git clone https://github.com/ARISE-Initiative/robosuite.git
git -C robosuite checkout -q "$ROBOSUITE_SHA"
[[ -d robocasa ]] || git clone https://github.com/robocasa/robocasa.git
git -C robocasa checkout -q "$ROBOCASA_SHA"
SIMENV="$WORK/simenv"
if [[ ! -f "$SIMENV/.install_done" ]]; then   # marker, NOT dir-exists: a half-built venv must rebuild
  rm -rf "$SIMENV"
  uv venv --python 3.11 "$SIMENV"
  # lerobot pin override + undeclared PyOpenGL dep — see node entry comments
  printf 'lerobot @ git+https://github.com/huggingface/lerobot@%s\n' "$LEROBOT_REV" > "$WORK/uv_override_rc.txt"
  uv pip install --python "$SIMENV/bin/python" --override "$WORK/uv_override_rc.txt" \
    -e "$WORK/robosuite" -e "$WORK/robocasa" \
    "imageio[ffmpeg]" msgpack msgpack-numpy pyzmq tyro PyOpenGL
  touch "$SIMENV/.install_done"
fi
SIMPY="$SIMENV/bin/python"
printf 'DATASET_BASE_PATH = "%s"\n' "$WORK/datasets" > "$WORK/robocasa/robocasa/macros_private.py"

# ---- 2. S3-gated parts (need working aws creds for BOTH the 141 and 124 buckets) ----
if ! aws sts get-caller-identity >/dev/null 2>&1 || ! aws s3 ls "$STUDY_S3/code/" >/dev/null 2>&1; then
  echo "NO USABLE AWS CREDS — non-S3 setup done; push creds and re-run to finish."; exit 0
fi

echo "[assets] syncing kitchen assets..."
aws s3 sync "$ASSETS_S3" "$WORK/robocasa/robocasa/models/assets" --only-show-errors
"$SIMPY" -c "from robocasa.utils.dataset_registry import TASK_SET_REGISTRY; print('sim env OK,', len(TASK_SET_REGISTRY['target50']), 'target tasks')"

fetch_archive() {  # component prefix dest_tgz : resolve full sha by prefix, download, verify
  local comp="$1" prefix="$2" dest="$3"
  local key; key=$(aws s3 ls "$STUDY_S3/code/$comp/" | awk '{print $4}' | grep "^$prefix" | head -1)
  [[ -n "$key" ]] || { echo "FATAL no $comp archive with prefix $prefix"; exit 3; }
  aws s3 cp "$STUDY_S3/code/$comp/$key" "$dest" --only-show-errors
  echo "${key%.tgz}  $dest" | sha256sum -c -
}

OPENPI="$WORK/openpi"
if [[ ! -f "$OPENPI/.install_done" ]]; then
  fetch_archive openpi "$OPENPI_SHA_PREFIX" "$WORK/openpi_jax_latest.tgz"
  mkdir -p "$OPENPI"; tar xzf "$WORK/openpi_jax_latest.tgz" -C "$OPENPI"
  cd "$OPENPI"; export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
  uv sync
  uv pip install --python "$OPENPI/.venv/bin/python" --no-deps PyOpenGL
  ROBOSUITE_DIR="$WORK/robosuite" ROBOCASA_DIR="$WORK/robocasa" \
    UV_PROJECT_ENVIRONMENT="$OPENPI/.venv" bash scripts/install_robocasa_deps.sh
  cd "$WORK"; touch "$OPENPI/.install_done"
fi
uv pip install --python "$SIMENV/bin/python" -e "$OPENPI/packages/openpi-client"

WSMV2="$WORK/wsmv2"
if [[ ! -d "$WSMV2/workspace_models" ]]; then
  fetch_archive wsmv2 "$WSMV2_SHA_PREFIX" "$WORK/wsmv2.tgz"
  mkdir -p "$WSMV2"; tar xzf "$WORK/wsmv2.tgz" -C "$WSMV2"
fi

# ---- 3. GPU + EGL sanity ----
"$OPENPI/.venv/bin/python" -c "import jax; print('jax devices:', jax.devices())"
MUJOCO_GL=egl "$SIMPY" -c "import mujoco; r=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light/><geom size=\"1\"/></worldbody></mujoco>'); import mujoco as m; d=m.MjData(r); ren=m.Renderer(r); ren.update_scene(d); print('EGL render OK', ren.render().shape)"
echo "ENV SETUP COMPLETE"
