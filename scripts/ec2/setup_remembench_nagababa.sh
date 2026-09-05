#!/bin/bash
# ReMemBench re-render/feature box setup for nagababa (g7e.24xlarge, 4x RTX PRO 6000).
#
# Builds everything needed to replay the ReMemBench Mem* demos at 256px on 4 GPUs:
#   /data/work/ReMemBench        - RoboCasa v0.2 fork (rsynced from the laptop working tree)
#   /data/work/robosuite         - ARISE robosuite @ 85abee22 (already present; verified, not re-cloned)
#   /data/work/remembench_env    - uv venv (py3.11) with both installed editable
#   /data/work/ReMemBench/robocasa/models/assets - v0.2 kitchen assets (~21GB, from UT Box)
#   /data/remembench_data        - Rutav/ReMemBench-Dataset (~11GB, from HF)
#
# Idempotent via per-stage marker files under /data/work/.remembench_markers/, so a
# re-run after an instance STOP (instance store is wiped) rebuilds only what is missing.
# The source tree itself is NOT fetched here -- push_remembench_src.sh rsyncs it, because
# the tri-integration branch is not pushed anywhere.
#
# Launch on the box:
#   tmux new -d -s rbsetup 'bash /data/setup_remembench_nagababa.sh 2>&1 | tee -a /data2/logs/rb_setup.log'
set -euxo pipefail

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR=/data/tmp
export UV_CACHE_DIR=/data/cache/uv
export HF_HOME=/data/cache/hf
unset MUJOCO_GL || true
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME" /data2/logs

WORK=/data/work
SRC="$WORK/ReMemBench"
ROBOSUITE="$WORK/robosuite"
VENV="$WORK/remembench_env"
DATA=/data/remembench_data
MARK="$WORK/.remembench_markers"
ROBOSUITE_SHA="85abee228d1c43ab1939bce33028099945d453b4"   # ARISE robosuite 1.5.2, same pin the laptop env uses
HF_DATASET_REPO="Rutav/ReMemBench-Dataset"
mkdir -p "$MARK"

done_mark() { [[ -f "$MARK/$1" ]]; }
set_mark()  { touch "$MARK/$1"; }

# ---- 0. system libs: EGL/OSMesa for headless MuJoCo, ffmpeg for LeRobot video encoding ----
if ! done_mark apt; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libegl1 libgl1 libosmesa6 libglu1-mesa ffmpeg
  set_mark apt
fi

# ---- 1. uv ----
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ---- 2. source trees ----
# ReMemBench must already have been rsynced (push_remembench_src.sh); it lives only on the laptop.
[[ -d "$SRC/robocasa" ]] || { echo "FATAL: $SRC missing -- run scripts/ec2/push_remembench_src.sh first"; exit 2; }
# robosuite: reuse the existing checkout if it is at the expected pin, else clone/checkout.
if [[ ! -d "$ROBOSUITE/.git" ]]; then
  git clone https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE"
fi
if [[ "$(git -C "$ROBOSUITE" rev-parse HEAD)" != "$ROBOSUITE_SHA" ]]; then
  git -C "$ROBOSUITE" fetch --all -q
  git -C "$ROBOSUITE" checkout -q "$ROBOSUITE_SHA"
fi

# ---- 3. venv (marker, not dir-exists: a half-built venv must be rebuilt) ----
if ! done_mark venv; then
  rm -rf "$VENV"
  uv venv --python 3.11 "$VENV"
  # Pins mirror the laptop env (/home/sarveshp/Research/envs/remembench_env): mujoco 3.3.1 +
  # numpy 2.2.5. PyOpenGL is an undeclared EGL dependency; imageio[ffmpeg]/opencv are used by
  # the obs-extraction + video-encoding path.
  uv pip install --python "$VENV/bin/python" \
    -e "$ROBOSUITE" -e "$SRC" \
    "mujoco==3.3.1" "numpy==2.2.5" h5py "imageio[ffmpeg]" opencv-python PyOpenGL \
    termcolor tqdm scipy numba pandas pyarrow huggingface_hub boto3
  # robocasa.utils.robomimic.robomimic_tensor_utils imports torch unconditionally; the render
  # path only needs its tensor helpers, so the CPU wheel is enough (and avoids a 3GB CUDA pull).
  uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
  set_mark venv
fi
PY="$VENV/bin/python"
printf 'DATASET_BASE_PATH = "%s"\n' "$DATA" > "$SRC/robocasa/macros_private.py"

# ---- 4. v0.2 kitchen assets (~21GB from UT Box) ----
if ! done_mark assets; then
  # download_kitchen_assets.py prompts before each multi-GB zip; feed it yes.
  ( cd "$SRC" && printf 'y\n%.0s' $(seq 50) | "$PY" robocasa/scripts/download_kitchen_assets.py )
  set_mark assets
fi
du -sh "$SRC/robocasa/models/assets"

# ---- 5. demos (~11GB from HF) ----
if ! done_mark demos; then
  mkdir -p "$DATA"
  "$VENV/bin/hf" download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$DATA"
  set_mark demos
fi
du -sh "$DATA"

# ---- 6. sanity: headless EGL env construction on GPU 0 ----
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PYEOF'
import json, h5py, pathlib
import robocasa
from robocasa.utils.mem_utils import *  # noqa: F401,F403  (registers Mem* envs if needed)
import robosuite
print("robosuite", robosuite.__version__, "robocasa", robocasa.__file__)
root = pathlib.Path("/data/remembench_data/MemHeatPot")
sess = sorted(p for p in root.iterdir() if p.is_dir())[0]
with h5py.File(sess / "demo_im128_notp.hdf5", "r") as f:
    env_args = json.loads(f["data"].attrs["env_args"])
    print("env", env_args["env_name"], "demos", len(f["data"]))
print("SETUP OK")
PYEOF

echo "REMEMBENCH BOX SETUP COMPLETE $(date -u +%FT%TZ)"
