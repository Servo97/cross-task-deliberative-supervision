#!/usr/bin/env bash
# One-node MoveCube dense/multi-point VISReg-v2 producer. Scientific evidence is not emitted here.
set -euo pipefail

CODE_DIR=/opt/ml/code
WORK=/opt/ml/move-workspace-dense-v2
mkdir -p "$WORK" "$WORK/tmp" "$WORK/task"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export UV_CACHE_DIR="$WORK/uv-cache"
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true
CONTROL_PY="$(command -v python3)"
nvidia-smi -L
"$CONTROL_PY" -c 'import boto3, botocore; print("MOVE_DENSE_V2_CONTROL_OK")'

required=(
  ROBOMME_MOVE_DENSE_V2_RUN_ID ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256
  ROBOMME_MOVE_DENSE_V2_MANIFEST_SOURCE ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256
  ROBOMME_MOVE_DENSE_V2_MANIFEST_S3 ROBOMME_MOVE_DENSE_V2_PRODUCER_CLAIM_S3
  ROBOMME_MOVE_DENSE_V2_COMPLETION_CLAIM_S3 ROBOMME_MOVE_DENSE_V2_ARTIFACT_ROOT_S3
  OPENPI_FORK_S3 ROBOMME_DATA_S3 ROBOMME_DATA_PARENT_INVENTORY_S3
  ROBOMME_DATA_PARENT_INVENTORY_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 20; }
done

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "FATAL expected exactly 8 GPUs" >&2; exit 21; }
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == "NVIDIA H100 80GB HBM3" ]] || { echo "FATAL non-H100 GPU: $gpu_name" >&2; exit 21; }
done

# Reproduce launch_guardrails.source_tree_sha256 on the unpacked source.  The manifest is the one
# generated file added only after the source identity was computed, so it is the sole exclusion.
"$CONTROL_PY" - "$CODE_DIR" "$ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256" \
  "$ROBOMME_MOVE_DENSE_V2_MANIFEST_SOURCE" <<'PY'
import hashlib, os, pathlib, stat, sys

root, expected, excluded = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if "/" in excluded or not excluded:
    raise SystemExit("invalid generated manifest exclusion")
digest = hashlib.sha256()

def field(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)

paths = [path for path in root.rglob("*") if path.relative_to(root).as_posix() != excluded]
entry = "gpu_move_workspace_dense_v2_entry.sh"
entry_path = root / entry
if not entry_path.is_file() or stat.S_IMODE(entry_path.stat().st_mode) != 0o777:
    raise SystemExit("SageMaker runtime entry must be mode 0777")
for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    field(relative)
    # SageMaker changes the selected program from staged 0755 to runtime 0777. Require that exact
    # runtime mode, then normalize only this entry back to staged mode for source-tree hashing.
    mode = 0o755 if relative == entry else stat.S_IMODE(path.lstat().st_mode)
    field(oct(mode))
    if path.is_symlink():
        field("symlink"); field(os.readlink(path))
    elif path.is_dir():
        field("directory")
    elif path.is_file():
        field("file")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                field(block)
    else:
        raise SystemExit(f"unsupported source entry: {path}")
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"runtime source identity mismatch: {actual} != {expected}")
print(f"MOVE_DENSE_V2_SOURCE_OK sha256={actual}")
PY

publish_once() {
  local source="$1" destination="$2" existing="$WORK/existing-immutable"
  rm -f "$existing"
  if aws s3 cp "$destination" "$existing" --only-show-errors 2>/dev/null; then
    cmp -s "$source" "$existing" || { echo "FATAL immutable collision $destination" >&2; return 22; }
    return 0
  fi
  local location bucket key
  location="${destination#s3://}"; bucket="${location%%/*}"; key="${location#*/}"
  if aws s3api put-object --bucket "$bucket" --key "$key" --body "$source" \
      --if-none-match '*' >/dev/null; then
    return 0
  fi
  aws s3 cp "$destination" "$existing" --only-show-errors
  cmp -s "$source" "$existing" || { echo "FATAL immutable collision $destination" >&2; return 22; }
}

download_hashed() {
  local uri="$1" expected="$2" destination="$3"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "FATAL invalid SHA $expected" >&2; return 23; }
  aws s3 cp "$uri" "$destination" --only-show-errors
  [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected" ]] || {
    echo "FATAL checksum mismatch $uri" >&2; return 23;
  }
}

MANIFEST="$WORK/manifest.json"
cp "$CODE_DIR/$ROBOMME_MOVE_DENSE_V2_MANIFEST_SOURCE" "$MANIFEST"
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256" ]] || {
  echo "FATAL manifest SHA mismatch" >&2; exit 23;
}
[[ "$(sha256sum "$CODE_DIR/gpu_move_workspace_dense_v2_entry.sh" | awk '{print $1}')" == \
  "$("$CONTROL_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["entry_sha256"])' "$MANIFEST")" ]] || {
  echo "FATAL entry byte SHA mismatch" >&2; exit 23;
}
# Fail before any remote write unless the full manifest and every duplicated environment value
# agree. The on-node source rehash above supplies the trusted source identity.
PYTHONPATH="$CODE_DIR" "$CONTROL_PY" - "$MANIFEST" "$ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256" <<'PY'
import json, os, sys
from training.workspace_gpu_producer_dense_v2 import validate_runtime_environment

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
validate_runtime_environment(manifest, manifest_sha256=sys.argv[2], environment=os.environ)
print("MOVE_DENSE_V2_MANIFEST_ENV_OK")
PY
publish_once "$MANIFEST" "$ROBOMME_MOVE_DENSE_V2_MANIFEST_S3"

PRODUCER="$WORK/producer.json"
"$CONTROL_PY" - "$PRODUCER" <<'PY'
import json, os, sys
value = {
    "schema_version": 2,
    "kind": "robomme_move_workspace_dense_v2_producer",
    "run_id": os.environ["ROBOMME_MOVE_DENSE_V2_RUN_ID"],
    "source_tree_sha256": os.environ["ROBOMME_MOVE_DENSE_V2_SOURCE_SHA256"],
    "manifest_sha256": os.environ["ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256"],
    "training_job_name": os.environ.get("TRAINING_JOB_NAME") or os.environ.get("SM_TRAINING_JOB_NAME"),
}
if not value["training_job_name"]:
    raise SystemExit("training job name unavailable")
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY
publish_once "$PRODUCER" "$ROBOMME_MOVE_DENSE_V2_PRODUCER_CLAIM_S3"

PARENT="$WORK/robomme.parent.inventory.json"
download_hashed "$ROBOMME_DATA_PARENT_INVENTORY_S3" \
  "$ROBOMME_DATA_PARENT_INVENTORY_SHA256" "$PARENT"
OPENPI_SHA="${OPENPI_FORK_S3%.tgz}"; OPENPI_SHA="${OPENPI_SHA##*/}"
download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA" "$WORK/openpi.tgz"
OPENPI="$WORK/openpi"; mkdir "$OPENPI"; tar xzf "$WORK/openpi.tgz" -C "$OPENPI"
cd "$OPENPI"
export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
uv sync --frozen
PY="$OPENPI/.venv/bin/python"
"$PY" -c 'import jax, numpy, pyarrow, optax, orbax.checkpoint; assert len(jax.devices()) == 8; assert {d.platform for d in jax.devices()} == {"gpu"}; print("MOVE_DENSE_V2_GPU_OK", jax.devices())'
export ROBOMME_WORKSPACE_COMPUTE_PYTHON="$PY"

cd "$CODE_DIR"
PYTHONPATH="$CODE_DIR:$OPENPI/src" "$CONTROL_PY" -m training.workspace_gpu_producer_dense_v2 \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$ROBOMME_MOVE_DENSE_V2_MANIFEST_SHA256" \
  --run-id "$ROBOMME_MOVE_DENSE_V2_RUN_ID" \
  --claim-s3 "$ROBOMME_MOVE_DENSE_V2_COMPLETION_CLAIM_S3" \
  --parent-manifest "$PARENT" \
  --data-s3 "$ROBOMME_DATA_S3" \
  --artifact-root-s3 "$ROBOMME_MOVE_DENSE_V2_ARTIFACT_ROOT_S3" \
  --work-root "$WORK/task"

echo "ROBOMME MOVE WORKSPACE DENSE V2 COMPLETE run_id=$ROBOMME_MOVE_DENSE_V2_RUN_ID"
