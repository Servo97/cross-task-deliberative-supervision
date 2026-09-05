#!/usr/bin/env bash
# Isolated FS-R1 8-H100 runtime canary. It cannot publish a score or launch the screen.
set -euo pipefail

CODE=/opt/ml/code
WORK=${FS_R1_WORK_ROOT:-/opt/ml/framesamp-r1-canary}
[[ "$WORK" == /opt/ml/* && "$WORK" != /opt/ml ]] || { echo "unsafe FS-R1 work root" >&2; exit 20; }
mkdir -p "$WORK" "$WORK/tmp" "$WORK/uv-cache" "$WORK/jax-cache" "$WORK/out"
export TMPDIR="$WORK/tmp" UV_CACHE_DIR="$WORK/uv-cache" JAX_COMPILATION_CACHE_DIR="$WORK/jax-cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_TRACEBACK_FILTERING=off
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true

required=(
  FS_R1_CANARY_ID FS_R1_MANIFEST_SHA256 FS_R1_NAMESPACE_S3 FS_R1_RECEIPT_S3
  FS_R1_EXPECTED_SOURCE_SHA256 FS_R1_PACKET_FILE_SHA256 FS_R1_PACKET_SHA256
  FS_R1_CHECKPOINT_S3 FS_R1_CHECKPOINT_ARCHIVE_SHA256 FS_R1_CHECKPOINT_SEMANTIC_SHA256
  FS_R1_OVERLAY_MANIFEST_SHA256 ROBOMME_EVAL_RUNTIME_S3 ROBOMME_EVAL_RUNTIME_SHA256
  ROBOMME_EVAL_VISION_S3 ROBOMME_EVAL_VISION_SHA256
  ROBOMME_EVAL_UPSTREAM_REPO ROBOMME_EVAL_UPSTREAM_COMMIT SM_USE_RESERVED_CAPACITY
)
for key in "${required[@]}"; do [[ -n "${!key:-}" ]] || { echo "missing $key" >&2; exit 20; }; done
[[ "$SM_USE_RESERVED_CAPACITY" == 1 ]] || { echo "p5 reserved-capacity routing flag drifted" >&2; exit 20; }

MANIFEST="$CODE/_robomme_framesamp_am_r1_canary_manifest.json"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || { echo "staged FS-R1 manifest absent" >&2; exit 21; }

# SageMaker changes only the entry mode from 0755 to 0777. Reproduce the launch-side source hash
# after normalizing exactly that path and reject every other byte/path/mode mutation.
python3 -B - "$CODE" "$FS_R1_EXPECTED_SOURCE_SHA256" <<'PY'
import hashlib, os, pathlib, stat, sys
root = pathlib.Path(sys.argv[1]); expected = sys.argv[2]; digest = hashlib.sha256()
def field(value):
    data = value if isinstance(value, bytes) else str(value).encode()
    digest.update(len(data).to_bytes(8, "big")); digest.update(data)
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix(); mode = stat.S_IMODE(path.lstat().st_mode)
    if relative == "_robomme_framesamp_am_r1_canary_manifest.json":
        if not path.is_file() or path.is_symlink(): raise SystemExit("staged manifest is not a regular file")
        continue
    if relative == "gpu_framesamp_am_r1_entry.sh":
        if mode != 0o777: raise SystemExit(f"runtime entry mode must be 0777, got {oct(mode)}")
        mode = 0o755
    field(relative); field(oct(mode))
    if path.is_symlink(): field("symlink"); field(os.readlink(path))
    elif path.is_dir(): field("directory")
    elif path.is_file():
        field("file")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): field(block)
    else: raise SystemExit(f"unsupported source entry {relative}")
actual = digest.hexdigest()
if actual != expected: raise SystemExit(f"source tree mismatch {actual} != {expected}")
print({"normalized_source_tree_sha256": actual, "runtime_entry_mode": "0o777"})
PY

# SageMaker flattens the submitted robomme_integration directory into /opt/ml/code. Reparent that
# immutable tree outside itself so ordinary ``import robomme_integration`` resolves without copying
# or mutating any submitted source bytes.
PKG_PARENT="$WORK/source-parent"
mkdir "$PKG_PARENT"
ln -s "$CODE" "$PKG_PARENT/robomme_integration"
[[ -L "$PKG_PARENT/robomme_integration" && "$(readlink -f "$PKG_PARENT/robomme_integration")" == "$CODE" ]] || exit 21

PYTHONPATH="$PKG_PARENT" python3 -B - "$MANIFEST" <<'PY'
import json, os, sys
from robomme_integration.eval.framesamp_am_r1_cloud import validate_environment, validate_manifest
manifest = json.load(open(sys.argv[1], encoding="utf-8")); validate_manifest(manifest)
keys = (
"FS_R1_CANARY_ID FS_R1_MANIFEST_SHA256 FS_R1_NAMESPACE_S3 FS_R1_RECEIPT_S3 "
"FS_R1_EXPECTED_SOURCE_SHA256 FS_R1_PACKET_FILE_SHA256 FS_R1_PACKET_SHA256 "
"FS_R1_CHECKPOINT_S3 FS_R1_CHECKPOINT_ARCHIVE_SHA256 FS_R1_CHECKPOINT_SEMANTIC_SHA256 "
"ROBOMME_EVAL_RUNTIME_S3 ROBOMME_EVAL_RUNTIME_SHA256 "
"ROBOMME_EVAL_VISION_S3 ROBOMME_EVAL_VISION_SHA256 ROBOMME_EVAL_UPSTREAM_REPO "
"ROBOMME_EVAL_UPSTREAM_COMMIT FS_R1_OVERLAY_MANIFEST_SHA256 SM_USE_RESERVED_CAPACITY"
).split()
validate_environment(manifest, {key: os.environ[key] for key in keys})
PY

location="${FS_R1_NAMESPACE_S3#s3://}"; bucket="${location%%/*}"; prefix="${location#*/}/"
receipt_location="${FS_R1_RECEIPT_S3#s3://}"; receipt_bucket="${receipt_location%%/*}"
receipt_key="${receipt_location#*/}"
[[ "$receipt_bucket" == "$bucket" && "$receipt_key" == "${prefix}canary.complete.json" ]] || exit 21
count="$(aws s3api list-objects-v2 --bucket "$bucket" --prefix "$prefix" --max-keys 1 --query KeyCount --output text)"
[[ "$count" == 0 ]] || { echo "FS-R1 canary namespace is not empty" >&2; exit 21; }

download_hashed() {
  local uri="$1" sha="$2" destination="$3"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || return 22
  aws s3 cp "$uri" "$destination.incomplete" --only-show-errors
  [[ "$(sha256sum "$destination.incomplete" | awk '{print $1}')" == "$sha" ]] || return 22
  mv "$destination.incomplete" "$destination"
}

download_hashed "$ROBOMME_EVAL_RUNTIME_S3" "$ROBOMME_EVAL_RUNTIME_SHA256" "$WORK/runtime.tgz"
download_hashed "$FS_R1_CHECKPOINT_S3" "$FS_R1_CHECKPOINT_ARCHIVE_SHA256" "$WORK/checkpoint.tgz"
download_hashed "$ROBOMME_EVAL_VISION_S3" "$ROBOMME_EVAL_VISION_SHA256" "$WORK/siglip_params.pkl"
mkdir "$WORK/runtime" "$WORK/checkpoint" "$WORK/upstream"
tar xzf "$WORK/runtime.tgz" -C "$WORK/runtime"
tar xzf "$WORK/checkpoint.tgz" -C "$WORK/checkpoint"

UPSTREAM="$WORK/upstream/robomme_policy_learning"
GIT_LFS_SKIP_SMUDGE=1 git init -q "$UPSTREAM"
git -C "$UPSTREAM" remote add origin "$ROBOMME_EVAL_UPSTREAM_REPO"
GIT_LFS_SKIP_SMUDGE=1 git -C "$UPSTREAM" fetch -q --depth=1 origin "$ROBOMME_EVAL_UPSTREAM_COMMIT"
git -C "$UPSTREAM" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$ROBOMME_EVAL_UPSTREAM_COMMIT" ]] || exit 23

# This released checkpoint is an upstream FrameSamp artifact. Its frozen JAX 0.5.3 / Orbax
# 0.11.13 runtime is part of the checkpoint contract; the project ed923 OpenPI environment uses
# JAX 0.10.1 / Orbax 0.12 and cannot restore this checkpoint representation.
cd "$UPSTREAM"; export UV_PROJECT_ENVIRONMENT="$UPSTREAM/.venv"; uv sync --frozen
PY="$UPSTREAM/.venv/bin/python"
[[ "$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == 3.11 ]] || exit 23
[[ "$($PY -c 'import jax, orbax.checkpoint as ocp; print(jax.__version__ + ":" + ocp.__version__)')" == "0.5.3:0.11.13" ]] || exit 23

OVERLAY="$WORK/policy-overlay"
PYTHONPATH="$PKG_PARENT:$UPSTREAM/src" "$PY" -B - "$UPSTREAM" "$OVERLAY" "$FS_R1_OVERLAY_MANIFEST_SHA256" <<'PY'
import sys
from robomme_integration.training.framesamp_am_policy_overlay import (
    stage_framesamp_am_policy_overlay, verify_framesamp_am_policy_overlay,
)
actual = stage_framesamp_am_policy_overlay(sys.argv[1], sys.argv[2])
if actual != sys.argv[3]: raise SystemExit(f"overlay SHA mismatch {actual} != {sys.argv[3]}")
verify_framesamp_am_policy_overlay(sys.argv[2], expected_manifest_sha256=sys.argv[3])
PY

one_dir() {
  local pattern="$1"; mapfile -t values < <(find "$WORK/runtime" -type d -path "$pattern" -print)
  [[ "${#values[@]}" == 1 ]] || { echo "expected one runtime path $pattern" >&2; exit 24; }
  echo "${values[0]}"
}
SIM_ENV="$(one_dir '*/env-v0.4.0')"
ROBOMME_SRC="$(one_dir '*/robomme-benchmark-f2b540e6/src')"
MANISKILL="$(one_dir '*/ManiSkill-07be6fbc')"
SIM_SITE="$(one_dir '*/env-v0.4.0/lib/python3.11/site-packages')"
RUNTIME_NORMALIZED="$WORK/runtime-normalized"; mkdir "$RUNTIME_NORMALIZED"
ln -s "$SIM_ENV" "$RUNTIME_NORMALIZED/env-v0.4.0"
ln -s "$MANISKILL" "$RUNTIME_NORMALIZED/ManiSkill-07be6fbc"
ln -s "${ROBOMME_SRC%/src}" "$RUNTIME_NORMALIZED/robomme-benchmark-f2b540e6"

export OPENPI_DATA_HOME="$WORK/vision"; mkdir -p "$OPENPI_DATA_HOME/pi05_vision_encoder"
mv "$WORK/siglip_params.pkl" "$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl"
CHECKPOINT="$WORK/checkpoint/perceptual-framesamp-modul/79999"
[[ -d "$CHECKPOINT/params" && -f "$CHECKPOINT/_CHECKPOINT_METADATA" ]] || exit 24
[[ -f "$WORK/checkpoint/perceptual-framesamp-modul/.EXTRACTED-$FS_R1_CHECKPOINT_SEMANTIC_SHA256" ]] || exit 24

export PYTHONPATH="$OVERLAY/src:$PKG_PARENT:$UPSTREAM/src:$UPSTREAM/.venv/lib/python3.11/site-packages:$SIM_SITE"
"$PY" -B -m robomme_integration.eval.framesamp_am_r1_canary \
  --output-dir "$WORK/out" \
  --policy-overlay "$OVERLAY" \
  --overlay-manifest-sha256 "$FS_R1_OVERLAY_MANIFEST_SHA256" \
  --official-checkout "$UPSTREAM" \
  --checkpoint "$CHECKPOINT" \
  --runtime-root "$RUNTIME_NORMALIZED"

PROOF="$WORK/out/canary.complete.json"
PYTHONPATH="$PKG_PARENT" python3 -B - "$PROOF" "$MANIFEST" <<'PY'
import json, sys
from robomme_integration.eval.framesamp_am_r1_cloud import validate_canary_receipt, validate_manifest
receipt = json.load(open(sys.argv[1], encoding="utf-8")); manifest = json.load(open(sys.argv[2], encoding="utf-8"))
validate_manifest(manifest); validate_canary_receipt(receipt, manifest)
PY

# The canary has exactly one isolated create-once write and publishes no outcome score.
count="$(aws s3api list-objects-v2 --bucket "$bucket" --prefix "$prefix" --max-keys 1 --query KeyCount --output text)"
[[ "$count" == 0 ]] || { echo "concurrent FS-R1 canary namespace collision" >&2; exit 25; }
aws s3api put-object --bucket "$receipt_bucket" --key "$receipt_key" --body "$PROOF" \
  --content-type application/json --if-none-match '*' >"$WORK/put.json"
aws s3 cp "$FS_R1_RECEIPT_S3" "$WORK/readback.json" --only-show-errors
cmp -s "$PROOF" "$WORK/readback.json" || { echo "FS-R1 receipt readback mismatch" >&2; exit 25; }
PYTHONPATH="$PKG_PARENT" python3 -B - "$WORK/readback.json" "$MANIFEST" <<'PY'
import json, sys
from robomme_integration.eval.framesamp_am_r1_cloud import validate_canary_receipt
validate_canary_receipt(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
PY
echo "FS-R1 P5 RUNTIME CANARY COMPLETE (NOT SCORED) $FS_R1_CANARY_ID"
