#!/usr/bin/env bash
# uv-based environment setup for wsmv2.
#
# The two VLA backbones cannot share one environment — GR00T N1.7 pins torch +
# transformers<4.58 + huggingface-hub<1.0, while pi-0.5 (openpi jax-latest) pins
# jax/flax/orbax. So we create ONE uv venv per backbone, plus a shared workspace-model env.
#
#   ./install.sh groot   ->  .venv-groot   (GR00T N1.7 torch stack + WSM heads)
#   ./install.sh pi05    ->  .venv-pi05    (pi-0.5 openpi jax-latest stack)
#   ./install.sh wsm     ->  .venv-wsm     (workspace-model + 3D-flow tooling)
#   ./install.sh all     ->  all of the above
#
# Backbone source repos (gr00t / openpi) are expected as siblings or installed here;
# see internal_planning_and_todos for the pinned commits and the jax-latest fork.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found — install from https://docs.astral.sh/uv/"; exit 1; }
BACKEND="${1:-all}"

setup_groot() {
    echo "==== .venv-groot (GR00T N1.7, py3.10) ===="
    uv venv .venv-groot --python 3.10
    uv pip install --python .venv-groot -e ".[groot,wsm,dev]"
    # NOTE: gr00t itself + the pinned NVIDIA/Isaac-GR00T commit are installed separately
    # (see internal_planning_and_todos/05_infra.md); transformers pulls hub 1.x via its own
    # lock, so a final `pip install 'huggingface-hub<1.0'` with REAL pip may be required.
}
setup_pi05() {
    echo "==== .venv-pi05 (pi-0.5 / openpi jax-latest, py3.11) ===="
    uv venv .venv-pi05 --python 3.11
    uv pip install --python .venv-pi05 -e ".[pi05,dev]"
    # openpi (robocasa fork @ jax-latest) installed separately; robocasa/robosuite via --no-deps.
}
setup_wsm() {
    echo "==== .venv-wsm (workspace-model + 3D flow, py3.11) ===="
    uv venv .venv-wsm --python 3.11
    uv pip install --python .venv-wsm -e ".[wsm,dev]"
    # 3D-flow (DynaFLIP recipe) git installs go here: cotracker3 / vggt / tapip3d / spatialtracker-v2.
}

case "$BACKEND" in
    groot) setup_groot ;;
    pi05)  setup_pi05 ;;
    wsm)   setup_wsm ;;
    all)   setup_groot; setup_pi05; setup_wsm ;;
    *)     echo "usage: ./install.sh {groot|pi05|wsm|all}"; exit 1 ;;
esac
echo "==== done: $BACKEND ===="
