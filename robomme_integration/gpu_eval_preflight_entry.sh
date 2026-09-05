#!/usr/bin/env bash
# Certify native (no nested Docker) RoboMME imports and one rendered reset on a p5 SageMaker node.
set -euo pipefail

echo "======== RoboMME p5 native-eval preflight | $(hostname) | $(date -u +%FT%TZ) ========"
required=(
  ROBOMME_EVAL_RUNTIME_S3 ROBOMME_EVAL_RUNTIME_SHA256 OPENPI_FORK_S3 OPENPI_SHA256
  OPENPI_PROFILE
  PREFLIGHT_ID PREFLIGHT_CLAIM_S3 PREFLIGHT_MANIFEST_SOURCE PREFLIGHT_MANIFEST_SHA256
  ROBOMME_PREFLIGHT_MODE ROBOMME_PREFLIGHT_SOURCE_TREE_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 30; }
done

CODE=/opt/ml/code
WORK=/opt/ml/work/robomme-eval-preflight
SOURCE_PARENT="$WORK/source"
SOURCE_ROOT="$SOURCE_PARENT/robomme_integration"
mkdir -p "$WORK" "$WORK/tmp" "$SOURCE_PARENT"
export TMPDIR="$WORK/tmp" UV_CACHE_DIR="$WORK/uv-cache" UV_PROJECT_ENVIRONMENT="$WORK/openpi/.venv"
# SageMaker's Python 3.10 image exports its stdlib on PYTHONPATH.  The pinned OpenPI environment is
# Python 3.11; inheriting those 3.10 paths into uv's build subprocess creates an SRE ABI mismatch.
unset PYTHONPATH PYTHONHOME || true

python3 - "$CODE/$PREFLIGHT_MANIFEST_SOURCE" "$PREFLIGHT_MANIFEST_SHA256" <<'PY'
import hashlib, json, pathlib, sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
claimed = value.pop("manifest_sha256", None)
actual = hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
).hexdigest()
if claimed != expected or actual != claimed:
    raise SystemExit("preflight input manifest seal mismatch")
PY

# Reproduce the launcher's sanitized source identity while excluding only its generated manifest.
# The SageMaker training toolkit chmods the selected program from its staged 0755 to 0777 before
# invoking it (job sarvesh-rmme-p5-action-4dda9bf2f82aa472cd0a, 2026-09-04: 62f89437… != 06b7b05b…
# was exactly that one mode bit change).  Require that runtime mode and normalize only this entry
# back to the submitted mode the launcher hashed; every other byte/path/mode mutation still fails.
python3 -B - \
  "$CODE" "$ROBOMME_PREFLIGHT_SOURCE_TREE_SHA256" "$PREFLIGHT_MANIFEST_SOURCE" \
  gpu_eval_preflight_entry.sh <<'PY'
import hashlib, os, pathlib, stat, sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
excluded = {sys.argv[3]}
entry = sys.argv[4]
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
    raise SystemExit(f"preflight source identity drift: {actual} != {expected}")
print(f"SOURCE_IDENTITY_OK sha256={actual}")
PY
cp -a "$CODE" "$SOURCE_ROOT"

download_hashed() {
  local uri="$1" sha="$2" destination="$3"
  aws s3 cp "$uri" "$destination.incomplete" --only-show-errors
  [[ "$(sha256sum "$destination.incomplete" | awk '{print $1}')" == "$sha" ]] || {
    echo "FATAL checksum mismatch $uri" >&2; return 31;
  }
  mv "$destination.incomplete" "$destination"
}

download_hashed "$ROBOMME_EVAL_RUNTIME_S3" "$ROBOMME_EVAL_RUNTIME_SHA256" "$WORK/runtime.tgz"
download_hashed "$OPENPI_FORK_S3" "$OPENPI_SHA256" "$WORK/openpi.tgz"
mkdir "$WORK/runtime" "$WORK/openpi"
tar xzf "$WORK/runtime.tgz" -C "$WORK/runtime"
tar xzf "$WORK/openpi.tgz" -C "$WORK/openpi"

cd "$WORK/openpi"
uv sync --frozen
PY="$WORK/openpi/.venv/bin/python"
[[ "$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == 3.11 ]] || {
  echo "FATAL native evaluator requires Python 3.11" >&2; exit 32;
}
if [[ "$OPENPI_PROFILE" == advanced ]]; then
  grep -Eq '^_WSM_PTRM[[:space:]]*=[[:space:]]*True$' \
    "$WORK/openpi/src/openpi/models/wsm_current_cond.py" || {
      echo "FATAL preflight OpenPI lacks PTRM restore support" >&2; exit 32;
    }
  grep -q 'self.wsm_jepa and train' "$WORK/openpi/src/openpi/models/pi0.py" || {
    echo "FATAL preflight OpenPI lacks JEPA checkpoint audit support" >&2; exit 32;
  }
elif [[ "$OPENPI_PROFILE" != standard ]]; then
  echo "FATAL unrecognized OpenPI preflight profile $OPENPI_PROFILE" >&2
  exit 32
fi

one_dir() {
  local pattern="$1"
  mapfile -t values < <(find "$WORK/runtime" -type d -path "$pattern" -print)
  [[ "${#values[@]}" == 1 ]] || { echo "FATAL expected one directory matching $pattern" >&2; exit 33; }
  echo "${values[0]}"
}
SITE="$(one_dir '*/env-v0.4.0/lib/python3.11/site-packages')"
HARNESS="$(one_dir '*/robomme-v0.4.0/src')"
ROBOMME="$(one_dir '*/robomme-benchmark-f2b540e6/src')"
MANISKILL="$(one_dir '*/ManiSkill-07be6fbc')"
OPENPI_SITE="$WORK/openpi/.venv/lib/python3.11/site-packages"
[[ -d "$OPENPI_SITE" ]] || { echo "FATAL OpenPI site-packages absent" >&2; exit 33; }
export OPENPI_SITE ROBOMME_RUNTIME_SITE="$SITE"
# 2026-09-05 (preflight 0b3ea564…): the registered v0.4.0 runtime carries a CPU-only torch — it was
# built for the lavapipe / SAPIEN_RENDER_DEVICE=cpu docker path. The native-EGL p5 lanes render on
# the GPU and ManiSkill reads frames back through torch (`get_picture_cuda(...).torch()`), which
# raised "Torch not compiled with CUDA enabled" in every shard. Overlay the exact torch build the
# local paper-protocol runtime already uses (2.9.1+cu128) into the simulator site. The runtime
# tarball identity recorded in the claim is unchanged; the overlay is logged here.
if ! PYTHONPATH="$SITE" "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[entry] simulator site torch has no CUDA — overlaying torch==${ROBOMME_SIM_TORCH:-2.9.1+cu128} into $SITE"
  uv pip install --python "$PY" --target "$SITE" --reinstall-package torch --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${ROBOMME_SIM_TORCH:-2.9.1+cu128}" || { echo "FATAL simulator torch overlay failed" >&2; exit 35; }
  PYTHONPATH="$SITE" "$PY" -c "import torch, sys; print('[entry] simulator torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "FATAL simulator torch still has no CUDA" >&2; exit 35; }
fi
# Source trees stay first, but overlapping binary/Python dependencies must resolve from the
# uv-locked OpenPI environment.  The bundled evaluator site is a fallback for simulator-only
# packages; putting it ahead of OpenPI previously shadowed NumPy and made JAX unimportable.
export PYTHONPATH="$HARNESS:$ROBOMME:$MANISKILL:$OPENPI_SITE:$SITE:$CODE"
export ROBOMME_USE_LAVAPIPE=auto
export LP_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" - <<'PY'
import os
from pathlib import Path

import numpy

numpy_path = Path(numpy.__file__).resolve()
openpi_site = Path(os.environ["OPENPI_SITE"]).resolve()
if not numpy_path.is_relative_to(openpi_site):
    raise RuntimeError(
        f"NumPy must resolve from the OpenPI venv, got {numpy_path}; runtime={os.environ['ROBOMME_RUNTIME_SITE']}"
    )
import jax, mani_skill, robomme, sapien, torch, vla_eval
assert jax.device_count() == 8, jax.devices()
print({
    "jax_devices": [str(device) for device in jax.devices()],
    "numpy": numpy.__version__,
    "numpy_path": str(numpy_path),
    "torch": torch.__version__,
    "sapien": getattr(sapien, "__version__", "unknown"),
})
PY
"$PY" -m vla_eval.cli.main run --help >/dev/null
mkdir -p "$WORK/links"
cat >"$WORK/links/vla-eval" <<EOF
#!/usr/bin/env bash
exec "$PY" -m vla_eval.cli.main "\$@"
EOF
chmod 0555 "$WORK/links/vla-eval"
"$WORK/links/vla-eval" run --help >/dev/null

if [[ "$ROBOMME_PREFLIGHT_MODE" == "p5_parallel_workspace_action_v1" ]]; then
  action_required=(
    ROBOMME_EVAL_VISION_S3 ROBOMME_EVAL_VISION_SHA256 ROBOMME_EVAL_VISION_BYTES
    ROBOMME_EVAL_UPSTREAM_REPO ROBOMME_EVAL_UPSTREAM_COMMIT
  )
  for name in "${action_required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 33; }
  done
  VISION="$WORK/vision"
  UPSTREAM="$WORK/upstream/robomme_policy_learning"
  mkdir -p "$VISION/pi05_vision_encoder" "$WORK/upstream"
  download_hashed \
    "$ROBOMME_EVAL_VISION_S3" "$ROBOMME_EVAL_VISION_SHA256" \
    "$VISION/pi05_vision_encoder/siglip_params.pkl"
  [[ "$(stat -c %s "$VISION/pi05_vision_encoder/siglip_params.pkl")" == \
    "$ROBOMME_EVAL_VISION_BYTES" ]] || {
      echo "FATAL vision byte-count mismatch" >&2; exit 33;
    }
  GIT_LFS_SKIP_SMUDGE=1 git init -q "$UPSTREAM"
  git -C "$UPSTREAM" remote add origin "$ROBOMME_EVAL_UPSTREAM_REPO"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$UPSTREAM" fetch -q \
    --depth=1 origin "$ROBOMME_EVAL_UPSTREAM_COMMIT"
  git -C "$UPSTREAM" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$ROBOMME_EVAL_UPSTREAM_COMMIT" ]] || {
    echo "FATAL upstream source commit drift" >&2; exit 33;
  }
  export PYTHONPATH="$SOURCE_PARENT:$HARNESS:$WORK/openpi/src:$OPENPI_SITE:$UPSTREAM/src:$SITE"
  "$PY" -m robomme_integration.eval.p5_parallel_action_preflight \
    --manifest "$CODE/$PREFLIGHT_MANIFEST_SOURCE" \
    --source-root "$SOURCE_PARENT" \
    --work-root "$WORK/action-canary" \
    --policy-python "$PY" \
    --vla-eval "$WORK/links/vla-eval" \
    --harness-src "$HARNESS" \
    --robomme-src "$ROBOMME" \
    --maniskill-src "$MANISKILL" \
    --openpi-src "$WORK/openpi/src" \
    --policy-site "$OPENPI_SITE" \
    --simulator-site "$SITE" \
    --upstream-root "$UPSTREAM" \
    --vision-encoder-home "$VISION" \
    --timeout-seconds 3600 \
    --confirm-run || canary_rc=$?
  if [[ "${canary_rc:-0}" -ne 0 ]]; then
    # 2026-09-05 (preflight 6e3b28e4…, "lane p5-h100-gpu3 failed with 1"): per-lane launcher.log and
    # server logs live only on the node, so a lane failure was undiagnosable. Ship them to a
    # failure/ sibling of the evidence root (never the evidence namespace itself, which must stay
    # empty until a claim is published). Small text only; the contract evidence path is unchanged.
    failure_s3="${PREFLIGHT_CLAIM_S3%/manifests/claims/preflight/*}/artifacts/robomme/eval_preflight/$PREFLIGHT_ID/failure/$(date -u +%Y%m%dT%H%M%SZ)"
    echo "ACTION CANARY FAILED rc=$canary_rc — shipping lane diagnostics to $failure_s3" >&2
    # build_lane_commands is given work_root=<work-root>/evidence, so lanes live at evidence/lanes/<lane>/.
    aws s3 sync "$WORK/action-canary/evidence" "$failure_s3/evidence" --only-show-errors \
      --exclude "*" --include "*.log" --include "*.json" --include "*.txt" --include "*.yaml" || true
    aws s3 ls --recursive "$failure_s3/" | tail -20 || true
    exit "$canary_rc"
  fi
  echo "ROBOMME P5 PARALLEL ACTION PREFLIGHT COMPLETE id=$PREFLIGHT_ID"
  exit 0
elif [[ "$ROBOMME_PREFLIGHT_MODE" != "native_render_reset_v1" ]]; then
  echo "FATAL unrecognized preflight mode $ROBOMME_PREFLIGHT_MODE" >&2
  exit 33
fi

# This is the useful capability gate: exact RoboMME test metadata, demonstration generation, SAPIEN
# scene construction, and the first camera render.  No policy checkpoint is read and no score is made.
timeout 900 "$PY" - <<'PY'
import asyncio

from eval.benchmark import RoboMMEOfficialHistoryBenchmark

async def main():
    benchmark = RoboMMEOfficialHistoryBenchmark(
        tasks=["MoveCube"], dataset="test", max_steps=1,
        send_wrist_image=True, send_state=True, send_video_history=True,
    )
    task = {"name": "MoveCube", "env_id": "MoveCube", "episode_idx": 0}
    await benchmark.start_episode(task)
    observation = await benchmark.get_observation()
    history = observation.get("video_history", [])
    state_history = observation.get("video_state_history", [])
    if (
        not history
        or not state_history
        or len(history) != len(state_history)
        or not observation.get("episode_restart")
    ):
        raise RuntimeError(
            "RoboMME official-history reset returned missing/unpaired conditioning video/state "
            f"history: frames={len(history)} states={len(state_history)}"
        )
    env = getattr(benchmark, "_env", None)
    if env is not None and hasattr(env, "close"):
        env.close()
    print(
        f"ROBOMME_RENDER_RESET_OK demo_frames={len(history)} "
        f"demo_states={len(state_history)}"
    )

asyncio.run(main())
PY

cp "$CODE/$PREFLIGHT_MANIFEST_SOURCE" "$WORK/claim.json"
"$PY" - "$WORK/claim.json" <<'PY'
import hashlib, json, os, sys
path = sys.argv[1]
manifest = json.load(open(path, encoding="utf-8"))
claimed = manifest.pop("manifest_sha256", None)
actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if claimed != os.environ["PREFLIGHT_MANIFEST_SHA256"] or actual != claimed:
    raise SystemExit("preflight manifest seal mismatch")
manifest["manifest_sha256"] = claimed
manifest["status"] = "native_render_reset_passed"
open(path, "w", encoding="utf-8").write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

location="${PREFLIGHT_CLAIM_S3#s3://}"; bucket="${location%%/*}"; key="${location#*/}"
if ! aws s3api put-object --bucket "$bucket" --key "$key" --body "$WORK/claim.json" \
    --if-none-match '*' >/dev/null; then
  aws s3 cp "$PREFLIGHT_CLAIM_S3" "$WORK/existing.json" --only-show-errors
  cmp -s "$WORK/claim.json" "$WORK/existing.json" || {
    echo "FATAL preflight claim collision" >&2; exit 34;
  }
fi
echo "ROBOMME P5 NATIVE EVAL PREFLIGHT COMPLETE id=$PREFLIGHT_ID"
