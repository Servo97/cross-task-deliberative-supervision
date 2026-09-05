#!/usr/bin/env bash
# Isolated single-node entry for the H10 attention-matching E0 velocity-matching attempt (p5e/H200).
set -euo pipefail

echo "======== AMKV E0 | $(hostname) | $(date -u +%FT%TZ) ========"
CODE_DIR=/opt/ml/code
WORK=/opt/ml/work
mkdir -p "$WORK" "$WORK/tmp"
cd "$WORK"
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf"
export UV_CACHE_DIR="$WORK/uv-cache"
export PYTHONUNBUFFERED=1
# The clean-source receipt seals every extracted path byte-for-byte.  Importing
# the editable policy before the E0 runner rechecks that receipt would normally
# add ``__pycache__/*.pyc`` files and make an otherwise pristine source tree
# look tampered.  Keep the tree immutable instead of weakening the receipt
# verifier to ignore runtime-created files.
export PYTHONDONTWRITEBYTECODE=1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
unset PYTHONPATH PYTHONHOME || true
nvidia-smi -L

required=(
  AMKV_POLICY_SOURCE_S3 AMKV_POLICY_SOURCE_SHA256
  AMKV_POLICY_SOURCE_RECEIPT_S3 AMKV_POLICY_SOURCE_RECEIPT_SHA256 AMKV_CHECKPOINT_S3
  AMKV_CHECKPOINT_INVENTORY_S3 AMKV_CHECKPOINT_INVENTORY_SHA256
  AMKV_FIXTURES_S3 AMKV_FIXTURES_MANIFEST_SHA256 AMKV_OUTPUT_S3
  AMKV_RATIOS AMKV_RUN_ID RUN_MANIFEST_SOURCE RUN_MANIFEST_SHA256
  AMKV_CODE_SOURCE_TREE_SHA256 AMKV_POLICY_GIT_SHA AMKV_POLICY_TREE_SHA1
  SM_USE_RESERVED_CAPACITY
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 20; }
done
[[ "$AMKV_RATIOS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "FATAL unsafe AMKV_RATIOS" >&2; exit 20; }
[[ "$AMKV_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "FATAL unsafe AMKV_RUN_ID" >&2; exit 20; }
# p5e capacity is plan-backed: the launcher pins TrainingPlanArn and must disable the implicit
# reserved-capacity request.  A node that sees anything else was launched off-contract.
[[ "$SM_USE_RESERVED_CAPACITY" == 0 ]] || {
  echo "FATAL plan-backed p5e job requires SM_USE_RESERVED_CAPACITY=0" >&2; exit 20;
}

download_hashed() {
  local uri="$1" expected="$2" destination="$3"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "FATAL invalid SHA $expected" >&2; return 23; }
  aws s3 cp "$uri" "$destination" --only-show-errors
  [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected" ]] || {
    echo "FATAL checksum mismatch $uri" >&2; return 23;
  }
}

publish_once() {
  local source="$1" destination="$2" existing="$WORK/existing-immutable"
  rm -f "$existing"
  local location bucket key
  location="${destination#s3://}"
  bucket="${location%%/*}"
  key="${location#*/}"
  if aws s3api put-object --bucket "$bucket" --key "$key" --body "$source" \
      --if-none-match '*' >/dev/null 2>&1; then
    echo "PUBLISHED $destination"
    return 0
  fi
  aws s3 cp "$destination" "$existing" --only-show-errors
  cmp -s "$source" "$existing" || { echo "FATAL immutable collision $destination" >&2; return 22; }
  echo "PUBLISHED $destination (already present, identical)"
}

RUN_MANIFEST="$WORK/run_manifest.json"
cp "$CODE_DIR/$RUN_MANIFEST_SOURCE" "$RUN_MANIFEST"
python3 - "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256" <<'PY'
import hashlib, json, sys
path, expected = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
claimed = value.pop("manifest_sha256", None)
actual = hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
).hexdigest()
if claimed != expected or actual != expected:
    raise SystemExit(f"run manifest seal mismatch claimed={claimed} actual={actual} expected={expected}")
PY
echo "OK run manifest sealed sha256=$RUN_MANIFEST_SHA256 run_id=$AMKV_RUN_ID"

SRC="$WORK/robomme_policy_learning"
SOURCE_RECEIPT="$WORK/policy_source.receipt.json"
download_hashed "$AMKV_POLICY_SOURCE_RECEIPT_S3" "$AMKV_POLICY_SOURCE_RECEIPT_SHA256" "$SOURCE_RECEIPT"
download_hashed "$AMKV_POLICY_SOURCE_S3" "$AMKV_POLICY_SOURCE_SHA256" "$WORK/policy_source.tgz"
mkdir -p "$SRC"
python3 - "$SOURCE_RECEIPT" "$WORK/policy_source.tgz" "$SRC" <<'PY'
import hashlib, json, os, pathlib, shutil, stat, sys, tarfile

receipt_path, archive_path, destination = map(pathlib.Path, sys.argv[1:])
expected = {
    "schema_version": 1,
    "kind": "amkv_policy_source_stage_receipt",
    "component": "robomme_policy_learning",
}
receipt = json.load(open(receipt_path, encoding="utf-8"))
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"source receipt {key} mismatch: {receipt.get(key)!r} != {value!r}")
if set(receipt) != {"schema_version", "kind", "component", "git", "archive", "extracted_tree"}:
    raise SystemExit("source receipt top-level schema mismatch")
git = receipt["git"]
if git != {
    "git_sha": os.environ["AMKV_POLICY_GIT_SHA"],
    "git_tree_sha1": os.environ["AMKV_POLICY_TREE_SHA1"],
    "worktree_status": "clean_including_untracked_and_submodules",
}:
    raise SystemExit("source receipt Git identity mismatch")
archive = receipt["archive"]
if archive.get("sha256") != os.environ["AMKV_POLICY_SOURCE_SHA256"]:
    raise SystemExit("source receipt archive SHA mismatch")
if archive.get("uri") != os.environ["AMKV_POLICY_SOURCE_S3"]:
    raise SystemExit("source receipt archive URI mismatch")
if archive.get("bytes") != archive_path.stat().st_size:
    raise SystemExit("source receipt archive size mismatch")

def safe_key(value):
    path = pathlib.Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise SystemExit(f"unsafe source archive member {value!r}")
    return value

with tarfile.open(archive_path, mode="r:gz") as bundle:
    members = bundle.getmembers()
    names = [safe_key(member.name.rstrip("/")) for member in members]
    if len(names) != len(set(names)):
        raise SystemExit("source archive contains duplicate members")
    for member in members:
        target = destination / member.name.rstrip("/")
        for parent in target.parents:
            if parent == destination:
                break
            if parent.is_symlink():
                raise SystemExit(f"source archive traverses a symlink ancestor: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=False)
            target.chmod(member.mode)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(member)
            if stream is None:
                raise SystemExit(f"cannot read source archive member {member.name}")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            target.chmod(member.mode)
        elif member.issym():
            link = pathlib.Path(member.linkname)
            if link.is_absolute() or ".." in link.parts:
                raise SystemExit(f"unsafe source symlink target: {member.name} -> {member.linkname}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(member.linkname)
        else:
            raise SystemExit(f"unsupported source archive member type: {member.name}")

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

objects = []
for path in sorted(destination.rglob("*"), key=lambda item: item.relative_to(destination).as_posix()):
    status = path.lstat()
    record = {
        "key": safe_key(path.relative_to(destination).as_posix()),
        "mode": stat.S_IMODE(status.st_mode),
    }
    if stat.S_ISLNK(status.st_mode):
        record.update(type="symlink", target=os.readlink(path))
    elif stat.S_ISDIR(status.st_mode):
        record.update(type="directory")
    elif stat.S_ISREG(status.st_mode):
        record.update(type="file", size_bytes=status.st_size, sha256=file_sha(path))
    else:
        raise SystemExit(f"unsupported extracted source entry {path}")
    objects.append(record)
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
tree = receipt["extracted_tree"]
actual_sha = hashlib.sha256(canonical({"objects": objects}).encode()).hexdigest()
if tree.get("algorithm") != "amkv_extracted_source_tree_sha256_v1":
    raise SystemExit("source receipt extracted-tree algorithm mismatch")
if tree.get("objects") != objects or tree.get("tree_sha256") != actual_sha:
    raise SystemExit("extracted source tree does not match its create-once receipt")
totals = {
    "objects": len(objects),
    "files": sum(record["type"] == "file" for record in objects),
    "bytes": sum(record.get("size_bytes", 0) for record in objects),
}
if tree.get("totals") != totals:
    raise SystemExit("extracted source tree totals do not match its receipt")
print(f"OK source receipt verified tree={actual_sha} objects={len(objects)}")
PY
[[ -f "$SRC/pyproject.toml" && -d "$SRC/src" ]] || {
  echo "FATAL official policy source tree incomplete at $SRC" >&2; exit 24;
}

CKPT_DIR="$WORK/checkpoint"
INVENTORY="$WORK/checkpoint.inventory.json"
download_hashed "$AMKV_CHECKPOINT_INVENTORY_S3" "$AMKV_CHECKPOINT_INVENTORY_SHA256" "$INVENTORY"
mkdir -p "$CKPT_DIR"
aws s3 sync "$AMKV_CHECKPOINT_S3" "$CKPT_DIR" --only-show-errors
python3 - "$INVENTORY" "$CKPT_DIR" <<'PY'
import hashlib, json, pathlib, sys
inventory, root = sys.argv[1:]
root = pathlib.Path(root)
objects = json.load(open(inventory, encoding="utf-8"))["objects"]
for record in objects:
    path = root / record["key"]
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        raise SystemExit(f"checkpoint object missing or wrong size: {record['key']}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise SystemExit(f"checkpoint object checksum mismatch: {record['key']}")
for name in ("79999/params", "79999/assets", "79999/_CHECKPOINT_METADATA", "history_config.txt"):
    if not (root / name).exists():
        raise SystemExit(f"orbax checkpoint missing {name}")
print(f"OK checkpoint verified objects={len(objects)}")
PY

FIX_DIR="$WORK/fixtures"
mkdir -p "$FIX_DIR"
aws s3 sync "$AMKV_FIXTURES_S3" "$FIX_DIR" --only-show-errors
[[ "$(sha256sum "$FIX_DIR/manifest.json" | awk '{print $1}')" == "$AMKV_FIXTURES_MANIFEST_SHA256" ]] || {
  echo "FATAL fixtures manifest mismatch" >&2; exit 23;
}
python3 - "$FIX_DIR" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.load(open(root / "manifest.json", encoding="utf-8"))
digest = hashlib.sha256()
with (root / "fixtures.npz").open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != manifest["payload_sha256"]:
    raise SystemExit("fixtures payload_sha256 mismatch")
print("OK fixtures verified")
PY

cd "$SRC"
export UV_PROJECT_ENVIRONMENT="$WORK/policy-venv"
# Pin CPython 3.11 explicitly.  The official pyproject allows ">=3.11,<=3.12",
# but uv.lock ships cp311-only wheels for mujoco 2.3.7 (a transitive dep via
# openpi -> gym-aloha).  On 3.12 uv falls through to the sdist, whose build
# needs MUJOCO_PATH and fails.  3.11 is also the interpreter of the local
# environment in which the AM patch was validated bitwise, so pinning it keeps
# the node env identical to the audited one rather than merely working.
export UV_PYTHON=3.11
uv sync --frozen --python 3.11
PY_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
[[ -x "$PY_BIN" ]] || { echo "FATAL uv environment missing $PY_BIN" >&2; exit 24; }
PY_VERSION="$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$PY_VERSION" == "3.11" ]] || {
  echo "FATAL policy venv is python $PY_VERSION, expected 3.11 (mujoco wheel coverage)" >&2; exit 24;
}
echo "OK policy venv python=$PY_VERSION"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_ENABLE_X64=false
export TF_CUDNN_DETERMINISTIC=1
export PYTHONHASHSEED=0

"$PY_BIN" -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'
cd "$CODE_DIR"
PYTHONPATH="$CODE_DIR:$SRC/src" "$PY_BIN" -B -m robomme_integration.amkv.e0_run \
  --fixtures "$FIX_DIR" \
  --checkpoint "$CKPT_DIR/79999" \
  --policy-source "$SRC" \
  --ratios "$AMKV_RATIOS" \
  --runtime-dtype bfloat16 \
  --num-steps 10 \
  --model-seed 7 \
  --noise-seed 0 \
  --minimum-episodes 32 \
  --timing-repeats 3 \
  --run-manifest "$RUN_MANIFEST" \
  --run-id "$AMKV_RUN_ID" \
  --run-manifest-sha256 "$RUN_MANIFEST_SHA256" \
  --code-source-tree-sha256 "$AMKV_CODE_SOURCE_TREE_SHA256" \
  --policy-source-receipt "$SOURCE_RECEIPT" \
  --policy-source-receipt-sha256 "$AMKV_POLICY_SOURCE_RECEIPT_SHA256" \
  --policy-source-archive-sha256 "$AMKV_POLICY_SOURCE_SHA256" \
  --policy-git-sha "$AMKV_POLICY_GIT_SHA" \
  --policy-tree-sha1 "$AMKV_POLICY_TREE_SHA1" \
  --checkpoint-inventory-sha256 "$AMKV_CHECKPOINT_INVENTORY_SHA256" \
  --fixtures-manifest-sha256 "$AMKV_FIXTURES_MANIFEST_SHA256" \
  --out "$WORK/e0_results.json"
[[ -s "$WORK/e0_results.json" ]] || { echo "FATAL e0_run produced no results" >&2; exit 26; }

COMPLETE="$WORK/$AMKV_RUN_ID.complete.json"
python3 - "$COMPLETE" "$WORK/e0_results.json" <<'PY'
import hashlib, json, os, sys
path, results = sys.argv[1:]
value = {
    "schema_version": 1,
    "kind": "amkv_e0_velocity_matching_complete",
    "run_id": os.environ["AMKV_RUN_ID"],
    "ratios": os.environ["AMKV_RATIOS"],
    "run_manifest_sha256": os.environ["RUN_MANIFEST_SHA256"],
    "results_sha256": hashlib.sha256(open(results, "rb").read()).hexdigest(),
    "training_job_name": os.environ.get("TRAINING_JOB_NAME") or os.environ.get("SM_TRAINING_JOB_NAME"),
}
if not value["training_job_name"]:
    raise SystemExit("SageMaker training job name is missing from the completion proof")
open(path, "w", encoding="utf-8").write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY
publish_once "$WORK/e0_results.json" "${AMKV_OUTPUT_S3%/}/e0_results.json"
publish_once "$RUN_MANIFEST" "${AMKV_OUTPUT_S3%/}/run_manifest.json"
publish_once "$COMPLETE" "${AMKV_OUTPUT_S3%/}/$AMKV_RUN_ID.complete.json"
echo "AMKV E0 COMPLETE run_id=$AMKV_RUN_ID ratios=$AMKV_RATIOS output=$AMKV_OUTPUT_S3"
