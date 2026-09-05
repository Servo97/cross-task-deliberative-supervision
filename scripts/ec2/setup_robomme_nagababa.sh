#!/usr/bin/env bash
# Content-addressed RoboMME eval setup for nagababa. Invoke only under the shared lease.
set -euo pipefail

[[ "$(hostname)" == ip-10-242-9-112* ]] || { echo "FATAL nagababa only" >&2; exit 70; }
[[ -f /data/work/leases/robomme-eval/owner.json ]] || {
  echo "FATAL setup requires with_robomme_lease_nagababa.sh" >&2; exit 71;
}
: "${ROBOMME_SOURCE_S3:?}"
: "${ROBOMME_SOURCE_SHA256:?}"
: "${OPENPI_FORK_S3:?}"
: "${OPENPI_SHA256:?}"
: "${ROBOMME_EVAL_RUNTIME_S3:?}"
: "${ROBOMME_EVAL_RUNTIME_SHA256:?}"
: "${PALIGEMMA_TOKENIZER_S3:?}"
: "${PALIGEMMA_TOKENIZER_SHA256:?}"

ROOT=/data/work/robomme_eval
DOWNLOADS="$ROOT/downloads"
mkdir -p "$DOWNLOADS" /data/tmp
export TMPDIR=/data/tmp
export UV_CACHE_DIR="$ROOT/uv-cache"

materialize_archive() {
  local uri="$1" sha="$2" destination="$3" archive="$DOWNLOADS/$sha.tgz"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo "FATAL invalid archive SHA" >&2; return 72; }
  if [[ ! -f "$archive" ]]; then
    aws s3 cp "$uri" "$archive.incomplete" --only-show-errors
    [[ "$(sha256sum "$archive.incomplete" | awk '{print $1}')" == "$sha" ]] || {
      echo "FATAL archive checksum mismatch $uri" >&2; return 72;
    }
    mv "$archive.incomplete" "$archive"
  fi
  [[ "$(sha256sum "$archive" | awk '{print $1}')" == "$sha" ]] || return 72
  if [[ ! -f "$destination/.source-complete" ]]; then
    [[ ! -e "$destination" ]] || {
      echo "FATAL partial/colliding destination $destination" >&2; return 72;
    }
    local staging="${destination}.extracting.$$"
    mkdir -p "$staging"
    tar xzf "$archive" -C "$staging"
    touch "$staging/.source-complete"
    mv "$staging" "$destination"
  fi
}

SOURCE="$ROOT/source-$ROBOMME_SOURCE_SHA256"
OPENPI="$ROOT/openpi-$OPENPI_SHA256"
RUNTIME="$ROOT/runtime-$ROBOMME_EVAL_RUNTIME_SHA256"
materialize_archive "$ROBOMME_SOURCE_S3" "$ROBOMME_SOURCE_SHA256" "$SOURCE"
materialize_archive "$OPENPI_FORK_S3" "$OPENPI_SHA256" "$OPENPI"
materialize_archive "$ROBOMME_EVAL_RUNTIME_S3" "$ROBOMME_EVAL_RUNTIME_SHA256" "$RUNTIME"

[[ -f "$SOURCE/robomme_integration/eval/launch_gpu_fleet.py" ]] || {
  echo "FATAL source archive layout must contain robomme_integration/" >&2; exit 73;
}
[[ -f "$OPENPI/src/openpi/models/pi0_config.py" ]] || {
  echo "FATAL OpenPI archive layout is invalid" >&2; exit 73;
}

if ! docker image inspect wsm/robomme-cpu-runtime:ubuntu22 >/dev/null 2>&1; then
  mapfile -t IMAGE_TARS < <(find "$RUNTIME" -type f -name 'robomme-cpu-runtime*.tar' -print)
  [[ "${#IMAGE_TARS[@]}" == 1 ]] || {
    echo "FATAL runtime archive has no unique Docker image tar and image is not loaded" >&2; exit 74;
  }
  docker load -i "${IMAGE_TARS[0]}"
fi
mapfile -t VLA_EVALS < <(find "$RUNTIME" -type f -path '*/bin/vla-eval' -print)
[[ "${#VLA_EVALS[@]}" == 1 ]] || {
  echo "FATAL runtime archive must contain exactly one vla-eval executable" >&2; exit 74;
}
VLA_EVAL="${VLA_EVALS[0]}"

cd "$OPENPI"
export UV_PROJECT_ENVIRONMENT="$OPENPI/.venv"
if [[ ! -x "$OPENPI/.venv/bin/python" ]]; then
  uv sync --frozen
fi
PY="$OPENPI/.venv/bin/python"

OPENPI_DATA_HOME="$ROOT/openpi-data"
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
mkdir -p "$(dirname "$TOKENIZER")"
if [[ ! -f "$TOKENIZER" ]]; then
  aws s3 cp "$PALIGEMMA_TOKENIZER_S3" "$TOKENIZER.incomplete" --only-show-errors
  [[ "$(sha256sum "$TOKENIZER.incomplete" | awk '{print $1}')" == "$PALIGEMMA_TOKENIZER_SHA256" ]] || {
    echo "FATAL tokenizer checksum mismatch" >&2; exit 75;
  }
  mv "$TOKENIZER.incomplete" "$TOKENIZER"
fi
[[ "$(sha256sum "$TOKENIZER" | awk '{print $1}')" == "$PALIGEMMA_TOKENIZER_SHA256" ]] || exit 75

"$PY" -c 'import jax; assert len(jax.devices()) == 4; print(jax.devices())'
"$VLA_EVAL" run --help >/dev/null
docker image inspect wsm/robomme-cpu-runtime:ubuntu22 >/dev/null

python3 - "$ROOT/ENV.json" <<PY
import json, sys
value = {
    "schema_version": 1,
    "source_root": "$SOURCE",
    "source_sha256": "$ROBOMME_SOURCE_SHA256",
    "openpi_root": "$OPENPI",
    "openpi_sha256": "$OPENPI_SHA256",
    "policy_python": "$PY",
    "openpi_data_home": "$OPENPI_DATA_HOME",
    "runtime_root": "$RUNTIME",
    "runtime_sha256": "$ROBOMME_EVAL_RUNTIME_SHA256",
    "vla_eval": "$VLA_EVAL",
    "container_image": "wsm/robomme-cpu-runtime:ubuntu22",
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
echo "ROBOMME EC2 SETUP COMPLETE $ROOT/ENV.json"
