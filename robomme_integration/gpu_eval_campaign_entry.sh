#!/usr/bin/env bash
# Stage one exact p5 runtime and run a sealed sequence of RoboMME fixed-50 evaluations.
set -euo pipefail

echo "======== RoboMME p5 eval campaign | $(hostname) | $(date -u +%FT%TZ) ========"
required=(
  ROBOMME_EVAL_QUEUE_SOURCE ROBOMME_EVAL_QUEUE_FILE_SHA256
  ROBOMME_EVAL_PREFLIGHT_SOURCE ROBOMME_EVAL_PREFLIGHT_SHA256
  ROBOMME_EVAL_PREFLIGHT_CLAIM_S3
  ROBOMME_EVAL_RECEIPT_SOURCE ROBOMME_EVAL_RECEIPT_FILE_SHA256
  ROBOMME_EVAL_LAUNCH_SOURCE ROBOMME_EVAL_LAUNCH_FILE_SHA256
  ROBOMME_EVAL_SOURCE_TREE_SHA256 ROBOMME_EVAL_GENERATED_FILES ROBOMME_EVAL_MAX_RUN_SECONDS
  ROBOMME_EVAL_RUNTIME_S3 ROBOMME_EVAL_RUNTIME_SHA256 OPENPI_FORK_S3 OPENPI_SHA256
  ROBOMME_EVAL_OPENPI_PROFILE
  ROBOMME_EVAL_UPSTREAM_REPO ROBOMME_EVAL_UPSTREAM_COMMIT
  ROBOMME_EVAL_VISION_S3 ROBOMME_EVAL_VISION_SHA256 ROBOMME_EVAL_VISION_BYTES
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 40; }
done
[[ "$ROBOMME_EVAL_MAX_RUN_SECONDS" =~ ^[0-9]+$ ]] || { echo "FATAL invalid runtime cap" >&2; exit 40; }
(( ROBOMME_EVAL_MAX_RUN_SECONDS <= 86400 )) || { echo "FATAL eval job exceeds 24 hours" >&2; exit 40; }

CODE=/opt/ml/code
WORK=/opt/ml/work/robomme-eval-campaign
SOURCE_ROOT="$WORK/source"
RUNTIME="$WORK/runtime"
OPENPI="$WORK/openpi"
UPSTREAM="$WORK/upstream/robomme_policy_learning"
VISION="$WORK/vision"
LINKS="$WORK/links"
mkdir -p "$WORK/tmp" "$SOURCE_ROOT" "$RUNTIME" "$WORK/upstream" "$VISION/pi05_vision_encoder" "$LINKS"
export TMPDIR="$WORK/tmp" UV_CACHE_DIR="$WORK/uv-cache" UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
unset PYTHONPATH PYTHONHOME || true

verify_file() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "FATAL missing staged file $path" >&2; return 41; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "FATAL staged file digest mismatch $path" >&2; return 41;
  }
}
verify_file "$CODE/$ROBOMME_EVAL_QUEUE_SOURCE" "$ROBOMME_EVAL_QUEUE_FILE_SHA256"
verify_file "$CODE/$ROBOMME_EVAL_PREFLIGHT_SOURCE" "$ROBOMME_EVAL_PREFLIGHT_SHA256"
verify_file "$CODE/$ROBOMME_EVAL_RECEIPT_SOURCE" "$ROBOMME_EVAL_RECEIPT_FILE_SHA256"
verify_file "$CODE/$ROBOMME_EVAL_LAUNCH_SOURCE" "$ROBOMME_EVAL_LAUNCH_FILE_SHA256"

python3 - \
  "$CODE/$ROBOMME_EVAL_LAUNCH_SOURCE" "$ROBOMME_EVAL_SOURCE_TREE_SHA256" \
  "$ROBOMME_EVAL_GENERATED_FILES" "$ROBOMME_EVAL_QUEUE_FILE_SHA256" \
  "$ROBOMME_EVAL_PREFLIGHT_SHA256" "$ROBOMME_EVAL_RECEIPT_FILE_SHA256" <<'PY'
import hashlib, json, pathlib, sys

path = pathlib.Path(sys.argv[1])
source_sha, generated, queue_sha, preflight_sha, receipt_sha = sys.argv[2:]
value = json.loads(path.read_text(encoding="utf-8"))
clean = dict(value)
claimed = clean.pop("launch_manifest_sha256", None)
actual = hashlib.sha256(
    json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
).hexdigest()
if claimed != actual:
    raise SystemExit("eval launch manifest self-seal mismatch")
expected = {
    "source_tree_sha256": source_sha,
    "generated_source_files": generated.split(","),
    "queue_file_sha256": queue_sha,
    "preflight_claim_sha256": preflight_sha,
    "runtime_receipt_file_sha256": receipt_sha,
}
drift = {key: (value.get(key), wanted) for key, wanted in expected.items() if value.get(key) != wanted}
if drift:
    raise SystemExit(f"eval launch/environment identity drift: {drift}")
print(f"LAUNCH_MANIFEST_OK sha256={claimed}")
PY

# Reproduce launch_guardrails.source_tree_sha256 on the actual unpacked SageMaker tree.  The only
# excluded paths are the four generated files named and sealed by the launch manifest.  The SageMaker
# training toolkit chmods the selected program from its staged 0755 to 0777 before invoking it (the
# 2026-09-04 preflight drift): require that runtime mode and normalize only this entry back to the
# submitted mode the launcher hashed; every other byte/path/mode mutation still fails.
python3 -B - "$CODE" "$ROBOMME_EVAL_SOURCE_TREE_SHA256" "$ROBOMME_EVAL_GENERATED_FILES" \
  gpu_eval_campaign_entry.sh <<'PY'
import hashlib, os, pathlib, stat, sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
excluded = set(sys.argv[3].split(","))
entry = sys.argv[4]
if len(excluded) != 4 or any("/" in item or not item for item in excluded):
    raise SystemExit("generated-source exclusion set is not the exact four root files")
digest = hashlib.sha256()

def field(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)

entry_path = root / entry
if entry_path.is_symlink() or not entry_path.is_file():
    raise SystemExit("SageMaker runtime entry is not a regular file")
entry_mode = stat.S_IMODE(entry_path.lstat().st_mode)
if entry_mode != 0o777:
    raise SystemExit(f"SageMaker runtime entry must be mode 0777, got {oct(entry_mode)}")
paths = [path for path in root.rglob("*") if path.relative_to(root).as_posix() not in excluded]
for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    mode = 0o755 if relative == entry else stat.S_IMODE(path.lstat().st_mode)
    field(relative)
    field(oct(mode))
    if path.is_symlink():
        field("symlink")
        field(os.readlink(path))
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
    raise SystemExit(f"sanitized source identity differs from preflight: {actual} != {expected}")
print(f"SOURCE_IDENTITY_OK sha256={actual}")
PY

# Make the isolated source tree importable as robomme_integration without changing its bytes.
cp -a "$CODE" "$SOURCE_ROOT/robomme_integration"

download_hashed() {
  local uri="$1" sha="$2" destination="$3" bytes="${4:-}"
  aws s3 cp "$uri" "$destination.incomplete" --only-show-errors
  [[ "$(sha256sum "$destination.incomplete" | awk '{print $1}')" == "$sha" ]] || {
    echo "FATAL checksum mismatch $uri" >&2; return 42;
  }
  if [[ -n "$bytes" ]]; then
    [[ "$(stat -c %s "$destination.incomplete")" == "$bytes" ]] || {
      echo "FATAL byte-count mismatch $uri" >&2; return 42;
    }
  fi
  mv "$destination.incomplete" "$destination"
}

# The scored node trusts the immutable published canary, not merely a caller-supplied local JSON.
# Re-fetch it and require byte identity before using it as a gate.  Parallel action canaries also
# bind a content-addressed, score-redacted evidence archive which is authenticated here.
aws s3 cp "$ROBOMME_EVAL_PREFLIGHT_CLAIM_S3" "$WORK/preflight.published.json" \
  --only-show-errors
cmp -s "$CODE/$ROBOMME_EVAL_PREFLIGHT_SOURCE" "$WORK/preflight.published.json" || {
  echo "FATAL staged preflight differs from immutable published claim" >&2; exit 42;
}
mapfile -t PREFLIGHT_EVIDENCE < <(python3 - "$WORK/preflight.published.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") == "native_parallel_action_passed":
    evidence = value.get("evidence", {})
    print(evidence.get("uri", ""))
    print(evidence.get("sha256", ""))
    print(evidence.get("bytes", ""))
PY
)
if [[ "${#PREFLIGHT_EVIDENCE[@]}" == 3 ]]; then
  download_hashed \
    "${PREFLIGHT_EVIDENCE[0]}" "${PREFLIGHT_EVIDENCE[1]}" \
    "$WORK/preflight-evidence.tgz" "${PREFLIGHT_EVIDENCE[2]}"
  python3 - "$WORK/preflight-evidence.tgz" "$WORK/preflight.published.json" <<'PY'
import hashlib, json, sys, tarfile

archive, claim_path = sys.argv[1:]
claim = json.load(open(claim_path, encoding="utf-8"))
with tarfile.open(archive, "r:gz") as stream:
    members = stream.getmembers()
    if [member.name for member in members] != ["action-canary.yaml", "attestation.json"]:
        raise SystemExit("parallel action-preflight evidence contains unapproved members")
    if any(not member.isfile() for member in members):
        raise SystemExit("parallel action-preflight evidence contains a non-file")
    config = stream.extractfile(members[0]).read()
    attestation = json.loads(stream.extractfile(members[1]).read())
if hashlib.sha256(config).hexdigest() != claim.get("probe", {}).get("benchmark_config_sha256"):
    raise SystemExit("parallel action-preflight evidence config mismatch")
if (
    attestation.get("kind") != "robomme_p5_parallel_action_canary_redacted_evidence"
    or attestation.get("preflight_id") != claim.get("preflight_id")
    or attestation.get("observed") != claim.get("observed")
    or attestation.get("redaction") != {
        "episode_result_records_published": False,
        "raw_artifact_digests_retained_in_observed_attestation": True,
        "raw_policy_logs_published": False,
        "raw_simulator_logs_published": False,
        "success_values_published": False,
    }
):
    raise SystemExit("parallel action-preflight redacted evidence contract mismatch")
PY
fi

download_hashed "$ROBOMME_EVAL_RUNTIME_S3" "$ROBOMME_EVAL_RUNTIME_SHA256" "$WORK/runtime.tgz"
download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA256" "$WORK/openpi.tgz"
download_hashed \
  "$ROBOMME_EVAL_VISION_S3" "$ROBOMME_EVAL_VISION_SHA256" \
  "$VISION/pi05_vision_encoder/siglip_params.pkl" "$ROBOMME_EVAL_VISION_BYTES"
tar xzf "$WORK/runtime.tgz" -C "$RUNTIME"
mkdir -p "$OPENPI"
tar xzf "$WORK/openpi.tgz" -C "$OPENPI"

cd "$OPENPI"
uv sync --frozen
PY="$OPENPI/.venv/bin/python"
[[ "$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == 3.11 ]] || {
  echo "FATAL policy runtime is not Python 3.11" >&2; exit 43;
}
if [[ "$ROBOMME_EVAL_OPENPI_PROFILE" == advanced ]]; then
  grep -Eq '^_WSM_PTRM[[:space:]]*=[[:space:]]*True$' \
    "$OPENPI/src/openpi/models/wsm_current_cond.py" || {
      echo "FATAL advanced eval OpenPI lacks PTRM restore support" >&2; exit 43;
    }
  grep -q 'self.wsm_jepa and train' "$OPENPI/src/openpi/models/pi0.py" || {
    echo "FATAL advanced eval OpenPI lacks JEPA checkpoint audit support" >&2; exit 43;
  }
elif [[ "$ROBOMME_EVAL_OPENPI_PROFILE" != standard ]]; then
  echo "FATAL unrecognized eval OpenPI profile $ROBOMME_EVAL_OPENPI_PROFILE" >&2
  exit 43
fi

GIT_LFS_SKIP_SMUDGE=1 git init -q "$UPSTREAM"
git -C "$UPSTREAM" remote add origin "$ROBOMME_EVAL_UPSTREAM_REPO"
GIT_LFS_SKIP_SMUDGE=1 git -C "$UPSTREAM" fetch -q --depth=1 origin "$ROBOMME_EVAL_UPSTREAM_COMMIT"
git -C "$UPSTREAM" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$ROBOMME_EVAL_UPSTREAM_COMMIT" ]] || {
  echo "FATAL upstream source commit drift" >&2; exit 44;
}

one_path() {
  local kind="$1" pattern="$2"
  mapfile -t values < <(find "$RUNTIME" -type "$kind" -path "$pattern" -print)
  [[ "${#values[@]}" == 1 ]] || {
    echo "FATAL expected one runtime path matching $pattern, got ${#values[@]}" >&2; exit 45;
  }
  echo "${values[0]}"
}
HARNESS_SRC="$(one_path d '*/robomme-v0.4.0/src')"
ROBOMME_SRC="$(one_path d '*/robomme-benchmark-f2b540e6/src')"
MANISKILL_SRC="$(one_path d '*/ManiSkill-07be6fbc')"
SIM_SITE="$(one_path d '*/env-v0.4.0/lib/python3.11/site-packages')"
# Same CUDA-torch overlay as gpu_eval_preflight_entry.sh (2026-09-05): the v0.4.0 runtime's torch is
# CPU-only and the native-EGL lanes read camera frames back through torch on the GPU.
if ! PYTHONPATH="$SIM_SITE" "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[entry] simulator site torch has no CUDA — overlaying torch==${ROBOMME_SIM_TORCH:-2.9.1+cu128} into $SIM_SITE"
  uv pip install --python "$PY" --target "$SIM_SITE" --reinstall-package torch --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${ROBOMME_SIM_TORCH:-2.9.1+cu128}" || { echo "FATAL simulator torch overlay failed" >&2; exit 46; }
  PYTHONPATH="$SIM_SITE" "$PY" -c "import torch, sys; print('[entry] simulator torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "FATAL simulator torch still has no CUDA" >&2; exit 46; }
fi
cat >"$LINKS/vla-eval" <<EOF
#!/usr/bin/env bash
exec "$PY" -m vla_eval.cli.main "\$@"
EOF
chmod 0555 "$LINKS/vla-eval"
ln -s "$HARNESS_SRC" "$LINKS/harness-src"
ln -s "$ROBOMME_SRC" "$LINKS/robomme-src"
ln -s "$MANISKILL_SRC" "$LINKS/maniskill-src"
ln -s "$SIM_SITE" "$LINKS/simulator-site"

# Reconstruct the receipt from actual staged state and require byte-for-byte agreement with the
# launcher-predicted receipt pinned by the queue.  This is the bridge from a capability preflight
# to the exact runtime used for scored evaluation.
PYTHONPATH="$SOURCE_ROOT:$UPSTREAM/src:$OPENPI/src:$ROBOMME_SRC:$OPENPI/.venv/lib/python3.11/site-packages:$SIM_SITE" \
  "$PY" - \
  "$CODE/$ROBOMME_EVAL_RECEIPT_SOURCE" "$WORK/runtime-receipt.actual.json" \
  "$CODE/$ROBOMME_EVAL_PREFLIGHT_SOURCE" "$ROBOMME_EVAL_SOURCE_TREE_SHA256" \
  "$ROBOMME_EVAL_RECEIPT_FILE_SHA256" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys

expected_path, actual_path, preflight_path, source_sha, expected_file_sha = map(pathlib.Path, sys.argv[1:])
source_sha = str(source_sha)
expected_file_sha = str(expected_file_sha)
expected = json.loads(expected_path.read_text(encoding="utf-8"))
preflight_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
if expected.get("preflight_claim_sha256") != preflight_sha:
    raise SystemExit("runtime receipt/preflight binding mismatch")
if expected.get("source_tree_sha256") != source_sha:
    raise SystemExit("runtime receipt/source binding mismatch")
if expected.get("runtime") != {
    "uri": os.environ["ROBOMME_EVAL_RUNTIME_S3"],
    "sha256": os.environ["ROBOMME_EVAL_RUNTIME_SHA256"],
}:
    raise SystemExit("runtime receipt/download environment mismatch")
if expected.get("openpi") != {
    "uri": os.environ["OPENPI_FORK_S3"],
    "sha256": os.environ["OPENPI_SHA256"],
}:
    raise SystemExit("OpenPI receipt/download environment mismatch")
if expected.get("vision", {}).get("uri") != os.environ["ROBOMME_EVAL_VISION_S3"]:
    raise SystemExit("vision receipt/download environment mismatch")
if expected.get("upstream", {}).get("repo") != os.environ["ROBOMME_EVAL_UPSTREAM_REPO"]:
    raise SystemExit("upstream receipt/clone environment mismatch")
if expected.get("upstream", {}).get("commit") != os.environ["ROBOMME_EVAL_UPSTREAM_COMMIT"]:
    raise SystemExit("upstream receipt/commit environment mismatch")
clean = dict(expected)
claimed = clean.pop("receipt_sha256", None)
actual_seal = hashlib.sha256(
    json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
).hexdigest()
if claimed != actual_seal:
    raise SystemExit("expected runtime receipt self-seal mismatch")
for name in ("policy_python", "vla_eval"):
    path = pathlib.Path(expected["paths"][name])
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"runtime executable missing: {name}={path}")
for name in (
    "harness_src",
    "robomme_src",
    "maniskill_src",
    "openpi_src",
    "policy_site",
    "simulator_site",
    "upstream_root",
    "vision_encoder_home",
):
    path = pathlib.Path(expected["paths"][name])
    if not path.is_dir():
        raise SystemExit(f"runtime directory missing: {name}={path}")
upstream = pathlib.Path(expected["paths"]["upstream_root"])
commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
if commit != expected["upstream"]["commit"]:
    raise SystemExit("runtime upstream commit mismatch")
for relative, digest in expected["upstream"]["critical_sha256"].items():
    actual = hashlib.sha256((upstream / relative).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"runtime upstream critical source drift: {relative}")
vision = pathlib.Path(expected["paths"]["vision_encoder_home"]) / "pi05_vision_encoder/siglip_params.pkl"
if vision.stat().st_size != expected["vision"]["bytes"]:
    raise SystemExit("runtime vision asset byte-count mismatch")
digest = hashlib.sha256()
with vision.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != expected["vision"]["sha256"]:
    raise SystemExit("runtime vision asset digest mismatch")
wrapper = pathlib.Path(expected["paths"]["vla_eval"])
if hashlib.sha256(wrapper.read_bytes()).hexdigest() != expected["vla_eval_wrapper"]["sha256"]:
    raise SystemExit("relocatable vla-eval wrapper digest mismatch")
payload = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
if hashlib.sha256(payload).hexdigest() != expected_file_sha:
    raise SystemExit("reconstructed runtime receipt file digest mismatch")
actual_path.write_bytes(payload)
if actual_path.read_bytes() != expected_path.read_bytes():
    raise SystemExit("reconstructed runtime receipt differs from queue-pinned receipt")
print(f"RUNTIME_RECEIPT_OK sha256={expected_file_sha}")
PY

export PYTHONPATH="$HARNESS_SRC:$ROBOMME_SRC:$MANISKILL_SRC:$OPENPI/src:$OPENPI/.venv/lib/python3.11/site-packages:$SIM_SITE:$UPSTREAM/src:$SOURCE_ROOT"
"$LINKS/vla-eval" run --help >/dev/null
"$PY" - <<'PY'
import jax, numpy, openpi, robomme_integration
if jax.device_count() != 8:
    raise RuntimeError(f"expected 8 H100 devices, got {jax.devices()}")
print({"jax_devices": len(jax.devices()), "numpy": numpy.__version__, "openpi": openpi.__file__})
PY

"$PY" -m robomme_integration.eval.campaign \
  --queue "$CODE/$ROBOMME_EVAL_QUEUE_SOURCE" \
  --source-root "$SOURCE_ROOT" \
  --native-preflight-claim "$CODE/$ROBOMME_EVAL_PREFLIGHT_SOURCE" \
  --runtime-receipt "$WORK/runtime-receipt.actual.json" \
  --work-root "$WORK/run" \
  --confirm-run

echo "ROBOMME P5 EVAL CAMPAIGN COMPLETE | $(date -u +%FT%TZ)"
