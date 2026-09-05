#!/usr/bin/env bash
# Stage the two released RoboMME positive controls with resumable, hash-checked downloads.
set -euo pipefail

ROOT="${ROBOMME_OFFICIAL_ROOT:-$HOME/Research/TRI/robomme_eval/official_reference}"
REPO="$ROOT/robomme_policy_learning"
REPO_URL="https://github.com/RoboMME/robomme_policy_learning.git"
REPO_COMMIT="ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
BENCH_REPO="$ROOT/robomme_benchmark"
BENCH_REPO_URL="https://github.com/RoboMME/robomme_benchmark.git"
# This is the submodule revision recorded by REPO_COMMIT.  The released-checkpoint control must
# not silently inherit a newer challenge runtime because task termination and success predicates
# are part of the measured model contract.
BENCH_REPO_COMMIT="856bc3a189d4172f3f47dbee4424d585f8d78db3"

usage() {
  echo "usage: $0 source | env | checkpoint {pi05_baseline|framesamp_modul} | status" >&2
  exit 2
}

stage_source() {
  mkdir -p "$ROOT"
  if [[ ! -d "$REPO/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$REPO_URL" "$REPO"
  fi
  git -C "$REPO" fetch --depth=1 origin "$REPO_COMMIT"
  git -C "$REPO" checkout --detach "$REPO_COMMIT"
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$REPO_COMMIT" ]]
  [[ "$(git -C "$REPO" ls-tree HEAD third_party/robomme_benchmark | awk '{print $3}')" == "$BENCH_REPO_COMMIT" ]] || {
    echo "policy source does not pin the expected RoboMME benchmark commit" >&2
    exit 8
  }
  if [[ ! -d "$BENCH_REPO/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$BENCH_REPO_URL" "$BENCH_REPO"
  fi
  git -C "$BENCH_REPO" fetch --depth=1 origin "$BENCH_REPO_COMMIT"
  git -C "$BENCH_REPO" checkout --detach "$BENCH_REPO_COMMIT"
  [[ "$(git -C "$BENCH_REPO" rev-parse HEAD)" == "$BENCH_REPO_COMMIT" ]]
  echo "OFFICIAL_SOURCE_READY policy_commit=$REPO_COMMIT benchmark_commit=$BENCH_REPO_COMMIT root=$REPO"
}

stage_env() {
  [[ -f "$REPO/uv.lock" ]] || { echo "official source is not staged" >&2; exit 3; }
  (
    cd "$REPO"
    # The released serving import path currently imports pytest from gemma_pytorch.py, so the
    # repository's locked dev dependency set is required even for inference.
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
  )
  "$REPO/.venv/bin/python" - <<'PY'
import jax
import mme_vla_suite
import pytest
from mme_vla_suite.policies import policy
print({"policy_python": "ready", "jax_devices": [str(x) for x in jax.devices()]})
PY
  echo "OFFICIAL_POLICY_ENV_READY root=$REPO/.venv"
}

checkpoint_spec() {
  case "$1" in
    pi05_baseline)
      METHOD="pi05_baseline"
      HF_REPO_ID="Yinpei/pi05_baseline"
      HF_FILENAME="pi05_baseline/79999.zip"
      HF_REVISION="80caa59a4933804b6521e72981b107ae0051d32a"
      URL="https://huggingface.co/Yinpei/pi05_baseline/resolve/80caa59a4933804b6521e72981b107ae0051d32a/pi05_baseline/79999.zip"
      SHA256="b8a9e9d78e4336e04582a01767a59ee132ea1e17295c7cf98b4952500de178e5"
      BYTES="11551742488"
      ;;
    framesamp_modul)
      METHOD="perceptual-framesamp-modul"
      HF_REPO_ID="Yinpei/perceptual-framesamp-modul"
      HF_FILENAME="79999.zip"
      HF_REVISION="c0f565dd40082f32863b3ee9db99de5ed5d3d4e0"
      URL="https://huggingface.co/Yinpei/perceptual-framesamp-modul/resolve/c0f565dd40082f32863b3ee9db99de5ed5d3d4e0/79999.zip"
      HISTORY_CONFIG_URL="https://huggingface.co/Yinpei/perceptual-framesamp-modul/resolve/c0f565dd40082f32863b3ee9db99de5ed5d3d4e0/history_config.txt"
      SHA256="2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
      BYTES="11878950895"
      ;;
    *) usage ;;
  esac
}

stage_checkpoint() {
  checkpoint_spec "$1"
  local destination="$ROOT/checkpoints/$METHOD"
  local archive="$destination/79999.zip"
  local partial="$archive.partial"
  local extracted_marker="$destination/.EXTRACTED-$SHA256"
  local backend="${ROBOMME_OFFICIAL_DOWNLOAD_BACKEND:-auto}"
  mkdir -p "$destination"
  if [[ ! -f "$extracted_marker" && ! -f "$archive" ]]; then
    local can_xet=0
    if [[ -x "$REPO/.venv/bin/python" ]] && \
      "$REPO/.venv/bin/python" -c 'import hf_xet, huggingface_hub' >/dev/null 2>&1; then
      can_xet=1
    fi
    if [[ "$backend" == "xet" && "$can_xet" != 1 ]]; then
      echo "Xet was requested but hf_xet/huggingface_hub is unavailable in the official environment" >&2
      exit 9
    fi
    if [[ "$backend" == "xet" || ( "$backend" == "auto" && "$can_xet" == 1 ) ]]; then
      local transfer_root="$destination/.hf-transfer"
      mkdir -p "$transfer_root/tmp" "$transfer_root/hf-home"
      HF_HOME="$transfer_root/hf-home" \
      HF_HUB_CACHE="$transfer_root/hf-home/hub" \
      HF_XET_CACHE="$transfer_root/hf-home/xet" \
      HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
      HF_XET_HIGH_PERFORMANCE=1 \
      HF_HUB_DISABLE_XET=0 \
      HF_HUB_DISABLE_PROGRESS_BARS=1 \
      TMPDIR="$transfer_root/tmp" \
        "$REPO/.venv/bin/python" - "$HF_REPO_ID" "$HF_FILENAME" "$HF_REVISION" "$transfer_root" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, revision, local_dir = sys.argv[1:]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    revision=revision,
    local_dir=local_dir,
)
print(f"XET_DOWNLOAD_COMPLETE path={path}", flush=True)
PY
      local downloaded="$transfer_root/$HF_FILENAME"
      [[ "$(stat -c %s "$downloaded")" == "$BYTES" ]] || {
        echo "Xet checkpoint byte count mismatch for $METHOD" >&2
        exit 10
      }
      [[ "$(sha256sum "$downloaded" | awk '{print $1}')" == "$SHA256" ]] || {
        echo "Xet checkpoint SHA-256 mismatch for $METHOD" >&2
        exit 11
      }
      mv "$downloaded" "$archive"
      rm -rf -- "$transfer_root"
    elif [[ "$backend" == "curl" || "$backend" == "auto" ]]; then
      curl --fail --location --retry 20 --retry-all-errors --continue-at - \
        --output "$partial" "$URL"
      mv "$partial" "$archive"
    else
      echo "unknown download backend: $backend (expected auto, xet, or curl)" >&2
      exit 12
    fi
    [[ "$(stat -c %s "$archive")" == "$BYTES" ]] || {
      echo "checkpoint byte count mismatch for $METHOD" >&2
      exit 4
    }
  fi
  if [[ ! -f "$extracted_marker" ]]; then
    [[ "$(stat -c %s "$archive")" == "$BYTES" ]]
    [[ "$(sha256sum "$archive" | awk '{print $1}')" == "$SHA256" ]]
    # -o repairs any files left by an interrupted extraction.  Readiness is published only after
    # the complete checkpoint tree has been validated below.
    unzip -q -o "$archive" -d "$destination"
  fi
  mapfile -t params < <(find "$destination" -type d -name params -print)
  [[ "${#params[@]}" == 1 ]] || {
    echo "expected exactly one extracted params directory for $METHOD; found ${#params[@]}" >&2
    exit 6
  }
  local checkpoint="${params[0]%/params}"
  [[ -d "$checkpoint/assets" ]] || { echo "checkpoint assets missing: $checkpoint" >&2; exit 7; }
  if [[ "$1" == "framesamp_modul" ]]; then
    local history_config="$(dirname "$checkpoint")/history_config.txt"
    if [[ ! -f "$history_config" ]]; then
      curl --fail --location --retry 20 --retry-all-errors \
        --output "$history_config.partial" "$HISTORY_CONFIG_URL"
      [[ "$(tr -d '\r\n' < "$history_config.partial")" == "perceptual-framesamp-modul.yaml" ]] || {
        echo "downloaded FrameSamp history_config.txt has unexpected content" >&2
        exit 13
      }
      mv "$history_config.partial" "$history_config"
    fi
    [[ "$(tr -d '\r\n' < "$history_config")" == "perceptual-framesamp-modul.yaml" ]] || {
      echo "FrameSamp checkpoint has an unexpected or missing history_config.txt" >&2
      exit 13
    }
  fi
  if [[ ! -f "$extracted_marker" ]]; then
    printf '%s\n' "$SHA256" > "$extracted_marker"
  fi
  # The extracted final checkpoint is the meaningful artifact; the verified archive and any
  # superseded sequential HTTP prefix are redundant local copies.
  if [[ "${ROBOMME_OFFICIAL_KEEP_ARCHIVES:-0}" != 1 ]]; then
    rm -f -- "$archive" "$partial"
  fi
  echo "OFFICIAL_CHECKPOINT_READY method=$METHOD checkpoint=$checkpoint sha256=$SHA256"
}

status() {
  echo "root=$ROOT"
  if [[ -d "$REPO/.git" ]]; then
    echo "source=$(git -C "$REPO" rev-parse HEAD)"
  else
    echo "source=missing"
  fi
  if [[ -d "$BENCH_REPO/.git" ]]; then
    echo "benchmark_source=$(git -C "$BENCH_REPO" rev-parse HEAD)"
  else
    echo "benchmark_source=missing"
  fi
  if [[ -x "$REPO/.venv/bin/python" ]] && \
    JAX_PLATFORMS=cpu "$REPO/.venv/bin/python" -c 'import jax, mme_vla_suite, pytest; from mme_vla_suite.policies import policy' >/dev/null 2>&1; then
    echo "policy_env=ready"
  else
    echo "policy_env=missing_or_incomplete"
  fi
  for name in pi05_baseline framesamp_modul; do
    checkpoint_spec "$name"
    local destination="$ROOT/checkpoints/$METHOD"
    if [[ -f "$destination/.EXTRACTED-$SHA256" ]]; then
      echo "$name=extracted_ready sha256=$SHA256"
    elif [[ -f "$destination/79999.zip" ]]; then
      echo "$name=archive_ready bytes=$(stat -c %s "$destination/79999.zip")"
    elif find "$destination/.hf-transfer" -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
      local xet_partial
      xet_partial="$(find "$destination/.hf-transfer" -type f -name '*.incomplete' -print -quit)"
      echo "$name=xet_downloading bytes=$(stat -c %s "$xet_partial")/$BYTES curl_fallback_bytes=$(stat -c %s "$destination/79999.zip.partial" 2>/dev/null || echo 0)"
    elif [[ -f "$destination/.hf-transfer/$HF_FILENAME" ]]; then
      echo "$name=xet_downloaded_verifying bytes=$(stat -c %s "$destination/.hf-transfer/$HF_FILENAME")/$BYTES"
    elif [[ -f "$destination/79999.zip.partial" ]]; then
      echo "$name=downloading bytes=$(stat -c %s "$destination/79999.zip.partial")/$BYTES"
    else
      echo "$name=missing"
    fi
    find "$destination" -type d -name params -printf "$name.checkpoint=%h\n" 2>/dev/null || true
  done
}

case "${1:-}" in
  source) [[ "$#" == 1 ]] || usage; stage_source ;;
  env) [[ "$#" == 1 ]] || usage; stage_env ;;
  checkpoint) [[ "$#" == 2 ]] || usage; stage_checkpoint "$2" ;;
  status) [[ "$#" == 1 ]] || usage; status ;;
  *) usage ;;
esac
