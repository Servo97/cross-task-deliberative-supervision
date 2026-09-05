#!/usr/bin/env bash
# H14 deliberation node entry — SKELETON (not yet exercised on a node).
#
# One vLLM replica per GPU, one client shard per replica, embarrassingly parallel: no TP, no NCCL,
# no cross-node collectives. Submitted by scripts/deliberation/launch_deliberation.py, one stage
# per submission (A7).
#
# The five vLLM-startup failures found on the local 5090 (see aug_22/h14_p0_status.md §1.1) are
# handled inside scripts/deliberation/serve_vllm.sh, not here. This file only owns: layout,
# per-GPU fan-out, resume-from-S3, and per-file S3 sync.
#
# Contract with the launcher (all env, all non-secret):
#   WSM_DELIB_STAGE            pass1 | embed | pass2
#   WSM_DELIB_RUN_ID           content-addressed id; also the S3 leaf
#   WSM_DELIB_S3_OUT           s3://.../<stage>/<run_id>
#   WSM_DELIB_PASS1_S3_IN      pass-1 store to resume from / consume
#   WSM_DELIB_EMBED_S3_IN      embedding store to consume (pass2)
#   WSM_DELIB_NUM_SHARDS       global shard count = n_nodes * gpus_per_node
#   WSM_DELIB_GPUS_PER_NODE    replicas to start on this node
#   WSM_DELIB_MODEL / _REASONING_EFFORT / _MAX_TOKENS / _CONCURRENCY / _EMBED_MODEL
#   WSM_DELIB_PROMPT_SHA / _SCHEMA_SHA   asserted against the code before any token is spent
set -euo pipefail
# Any errexit/unbound death now names its line instead of exiting silently. Attempt 8 died one
# second after the self-test with zero output; this line would have identified it immediately.
trap 'rc=$?; echo "[entry] DIED at line $LINENO (rc=$rc)" >&2' ERR

WORK="${WORK:-/opt/ml/input/data/work}"
[ -d "$WORK" ] || WORK=/tmp/delib
mkdir -p "$WORK"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${WSM_DELIB_PYTHON:-python3}"
STAGE="${WSM_DELIB_STAGE:?}"
RUN_ID="${WSM_DELIB_RUN_ID:?}"
S3_OUT="${WSM_DELIB_S3_OUT:?}"
GPUS="${WSM_DELIB_GPUS_PER_NODE:-8}"
NUM_SHARDS="${WSM_DELIB_NUM_SHARDS:-$GPUS}"
# A 2-job pass-2 split: each job owns a disjoint [offset, offset+count) slice of the GLOBAL shard
# space and writes into the SAME content-addressed edge store, so the halves merge and each half's
# structural resume also covers work the other half already finished.
SHARD_OFFSET="${WSM_DELIB_SHARD_OFFSET:-0}"
SHARD_COUNT="${WSM_DELIB_SHARD_COUNT:-$NUM_SHARDS}"

# ---- node rank (SageMaker gives hosts + current host in resourceconfig.json) -------------------
NODE_RANK=0
RC=/opt/ml/input/config/resourceconfig.json
if [ -f "$RC" ]; then
  NODE_RANK=$("$PY" -c "
import json,sys
c=json.load(open('$RC'))
print(sorted(c['hosts']).index(c['current_host']))
")
fi
echo "[entry] stage=$STAGE run_id=$RUN_ID node_rank=$NODE_RANK gpus=$GPUS num_shards=$NUM_SHARDS shards=[$SHARD_OFFSET,$((SHARD_OFFSET + SHARD_COUNT)))"

# ---- fail closed on prompt/schema drift BEFORE spending a single token ------------------------
"$PY" - <<PYCHK
import sys
sys.path.insert(0, "$REPO")
from workspace_models.labels import caption_segments as CS
from scripts.deliberation import pass2_prompt as P2
want_p, want_s = "${WSM_DELIB_PROMPT_SHA:-}", "${WSM_DELIB_SCHEMA_SHA:-}"
have_p, have_s = ((CS.prompt_sha("descriptor"), CS.schema_sha("descriptor"))
                  if "$STAGE" == "pass1" else (P2.prompt_sha(), P2.schema_sha()))
if want_p and want_p != have_p:
    raise SystemExit(f"prompt sha drift: launcher {want_p[:16]} != code {have_p[:16]}")
if want_s and want_s != have_s:
    raise SystemExit(f"schema sha drift: launcher {want_s[:16]} != code {have_s[:16]}")
print("[entry] prompt/schema shas match the launcher")
PYCHK

OUT="$WORK/out/$STAGE"
mkdir -p "$OUT"

# ---- node environment: vLLM venv, built here, mirroring robocasa_policyfeat_entry.sh -----------
# The SageMaker DLC has no vLLM. Fail FAST on a broken CUDA wheel, before any 20 GB download.
unset PYTHONPATH PYTHONHOME || true
nvidia-smi -L || true
command -v uv >/dev/null 2>&1 || {
  echo "[entry] uv not on PATH; installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}
command -v uv >/dev/null 2>&1 || { echo "[entry] FATAL: no uv"; exit 4; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uvcache}"
export TMPDIR="${TMPDIR:-$WORK/tmp}"; mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
VENV="$WORK/vllmenv"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.12
  # vLLM is pinned but its TRANSITIVE deps were not, so the node resolved transformers 5.16.1 while
  # every local run of this corpus used 5.15.1. transformers owns the multimodal PROCESSOR for
  # qwen3_5 -- the exact layer a 9-image request goes through -- so an unpinned minor bump there is
  # the same class of unforced deviation as the enforce_eager one (§28.3). vllm 0.27.1 only requires
  # transformers>=5.5.3, so pinning the proven version is legal. Override to re-test a newer one.
  # vllm + transformers are the SERVER. The shard CLIENTS additionally need:
  #   pandas + pyarrow  -- robocerebra_source.build_index reads 994 parquets via pd.read_parquet
  #   av                -- decode_views decodes the LeRobot mp4s (ALL domains, not just this one)
  # None of these is a vLLM dependency, so the node venv never had them and every client died at
  # import in ~1 s. `av` is pinned to 14.2.0 because that manylinux wheel BUNDLES FFmpeg -- the node
  # has no system FFmpeg and no apt egress, so an unpinned av can install and still fail to dlopen.
  uv pip install --python "$VENV/bin/python" "vllm==${WSM_VLLM_VERSION:-0.27.1}" \
    "transformers==${WSM_TRANSFORMERS_VERSION:-5.15.1}" \
    pandas pyarrow "av==${WSM_AV_VERSION:-14.2.0}"
fi
PYV="$VENV/bin/python"
"$PYV" -c "
import sys, torch, vllm
print('torch', torch.__version__, torch.version.cuda, 'cuda?', torch.cuda.is_available())
print('vllm', vllm.__version__)
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
assert 'Qwen3_5ForConditionalGeneration' in _MULTIMODAL_MODELS, 'Qwen3.8 not registered multimodal'
sys.exit(0 if torch.cuda.is_available() else 9)
"
# Client-side import preflight. The server can be perfectly healthy while every client dies at
# import; that cost a full node cycle (attempt 8). Check before the servers are even started.
"$PYV" - <<'PYCLIENT' || { echo "[entry] FATAL: shard-client deps missing in the venv"; exit 5; }
import importlib.util, sys
missing = [m for m in ("pandas", "pyarrow", "av", "PIL", "numpy")
           if not importlib.util.find_spec(m)]
if missing:
    print("[entry] client deps MISSING:", missing); sys.exit(1)
import av  # noqa: F401 -- prove the wheel can actually dlopen its bundled FFmpeg
print("[entry] client deps OK (pandas, pyarrow, av, PIL, numpy)")
PYCLIENT

export WSM_VLLM_PYTHON="$PYV"
cd "$REPO"

# ---- model weights: download ONCE into a shared cache -------------------------------------------
# Eight replicas each pulling ~31 GB from HF would be ~248 GB of egress and eight racing writers.
# Fetch once, then every replica reads the same snapshot.
export HF_HOME="${HF_HOME:-$WORK/hf}"
mkdir -p "$HF_HOME"
"$PYV" - <<PYDL
from huggingface_hub import snapshot_download
p = snapshot_download("${WSM_DELIB_MODEL}",
                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
                      ignore_patterns=["model_mtp.safetensors"], max_workers=16)
print("[entry] model at", p)
PYDL

# ---- inputs ------------------------------------------------------------------------------------
# RoboCasa mp4s (pass 1 decodes frames straight from them), the FROZEN keyframe label store, and the
# H13 caption store used only as a hint. All three are content-addressed upstream; a sync that lands
# ZERO files is FATAL (the _pi05_common.py contract -- a silent empty sync mislabels the whole run).
stage_or_die () {  # $1=s3 uri  $2=local dir  $3=label
  aws s3 sync "$1" "$2" --only-show-errors
  n=$(find "$2" -type f | wc -l)
  echo "[entry] staged $3: $n files"
  [ "$n" -gt 0 ] || { echo "[entry] FATAL: zero files staged for $3 from $1"; exit 3; }
}

DOMAIN="${WSM_DELIB_DOMAIN:-robocasa}"

# Per-ATTEMPT log prefix. A fixed name (`_logs/vllm_gpu0.log`) is shadowable by a resume sync and by
# the next attempt; the job name makes every attempt's replica logs independently addressable.
JOB_TAG="${TRAINING_JOB_NAME:-${SAGEMAKER_JOB_NAME:-attempt-$(date -u +%Y%m%dT%H%M%SZ)}}"
LOG_PREFIX="$S3_OUT/_logs/$JOB_TAG"
echo "[entry] replica logs -> $LOG_PREFIX/"

if [ "$STAGE" = "pass1" ] && [ "$DOMAIN" = "robocerebra" ]; then
  # RoboCerebra needs NO keyframe label store and NO caption hints: its segmentation is the
  # official per-frame `subtask_index` column and its hint is the per-segment subtask string, both
  # inside the dataset itself (workspace_models/labels/robocerebra_source.py). So the whole input
  # is ONE content-addressed tarball of the sealed `robocerebra_train_v1` LeRobot tree -- the same
  # object the sealed H12 arms trained from, not a re-export.
  DATA="$WORK/robocerebra"; mkdir -p "$DATA"
  TAR="$WORK/robocerebra_train.tar"
  aws s3 cp "${WSM_DELIB_RCB_DATA_S3:?}" "$TAR" --only-show-errors
  got=$("$PY" -c "
import hashlib,sys
h=hashlib.sha256()
with open('$TAR','rb') as f:
    for b in iter(lambda: f.read(1<<20), b''): h.update(b)
print(h.hexdigest())")
  want="${WSM_DELIB_RCB_DATA_SHA256:?}"
  [ "$got" = "$want" ] || { echo "[entry] FATAL: robocerebra tar sha $got != $want"; exit 3; }
  tar -xf "$TAR" -C "$DATA" && rm -f "$TAR"
  RCB_ROOT="$(dirname "$(find "$DATA" -type f -name info.json -path '*/meta/*' | head -1)")/.."
  RCB_ROOT="$(cd "$RCB_ROOT" && pwd)"
  n_par=$(find "$RCB_ROOT/data" -name '*.parquet' | wc -l)
  n_mp4=$(find "$RCB_ROOT/videos" -name '*.mp4' | wc -l)
  echo "[entry] robocerebra root=$RCB_ROOT parquet=$n_par mp4=$n_mp4 ($(du -sh "$DATA" | cut -f1))"
  [ "$n_par" -eq 994 ] && [ "$n_mp4" -eq 1988 ] || {
    echo "[entry] FATAL: expected 994 parquet / 1988 mp4, got $n_par / $n_mp4"; exit 3; }
elif [ "$STAGE" = "pass1" ]; then
  DATA="$WORK/robocasa"; LABELS="$WORK/labels"; CAPS="$WORK/captions"
  mkdir -p "$DATA" "$LABELS" "$CAPS"
  IFS=',' read -ra TASKS <<< "${WSM_TASKS:?}"
  for t in "${TASKS[@]}"; do
    stage_or_die "${WSM_DELIB_DATASET_S3:?}/composite/$t" "$DATA/composite/$t" "dataset/$t"
    stage_or_die "${WSM_DELIB_LABELS_S3:?}/$t"            "$LABELS/$t"          "labels/$t"
    aws s3 sync "${WSM_DELIB_CAPTIONS_S3:-}/$t" "$CAPS/$t" --only-show-errors || true
  done
  echo "[entry] dataset $(du -sh "$DATA" | cut -f1), labels $(find "$LABELS" -name '*.npz' | wc -l) npz"
fi

# ---- structural resume: pull what already exists, then let the per-stage validator re-parse it --
# Existence is never trusted; caption_segments.validate_existing_descriptors and
# pass2_deliberate.validate_bucket_file re-parse and shape-check every file they find.
if [ "${WSM_DELIB_VERIFY_RESUME:-1}" = "1" ]; then
  # _logs is EXCLUDED from the resume pull. Attempt 6 restored attempt 5's _logs into $OUT, and the
  # exit-trap `sync "$OUT" "$S3_OUT"` then re-pushed those stale files OVER the fresh ones this run
  # had just uploaded -- all 8 objects landed with one timestamp and attempt 5's content, and
  # attempt 6's replica logs were lost. Only OUTPUTS resume; logs are write-only per attempt.
  aws s3 sync "$S3_OUT" "$OUT" --exclude "_logs/*" --only-show-errors || true
  echo "[entry] resumed $(find "$OUT" -type f | wc -l) files from $S3_OUT"
fi

# ---- per-completed-file S3 sync (A7): NOT sync-at-exit ----------------------------------------
# Outputs every 60 s, AND the per-shard client logs -- those live in $WORK, are never synced, and
# are the only visibility into client progress between "replicas ready" and the first episode JSON
# (which, given episode-level write granularity, is ~30 min of silence). Uploaded to the
# per-attempt prefix so they cannot shadow another attempt's.
( while true; do
    aws s3 sync "$OUT" "$S3_OUT" --exclude "_logs/*" --only-show-errors || true
    for f in "$WORK"/pass1_shard*.log "$WORK"/pass2_shard*.log; do
      [ -f "$f" ] || continue
      aws s3 cp "$f" "$LOG_PREFIX/$(basename "$f")" --only-show-errors || true
    done
    sleep 60
  done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; aws s3 sync "$OUT" "$S3_OUT" --exclude "_logs/*" --only-show-errors || true' EXIT

# The parent APIServer traceback is the LAST thing in a vLLM log and is NEVER the root cause -- it
# only says "Engine core initialization failed. See root cause above. Failed core proc(s): {}".
# That message fires when a core-process SENTINEL becomes readable (v1/engine/utils.py:1247), i.e.
# the EngineCore CHILD exited during init and wrote its real traceback EARLIER in the same file.
# `tail -40` lands squarely on the useless parent frames, which is how the first node failure cost a
# log dive. Print context first and the distilled root cause LAST, so the final lines of the job log
# are the answer.
dump_vllm_failure () {  # $1=log  $2=gpu
  local log="$1" g="$2"
  echo "================= vLLM gpu$g FAILED — diagnostic dump ================="
  echo "---- kernel selection (which backend actually got picked) ----"
  grep -hE "Selected .* for Fp8LinearMethod|GDN prefill kernel|Using .*attention backend" \
    "$log" 2>/dev/null | head -6 || true
  echo "---- last 30 lines (parent APIServer frames; context only) ----"
  tail -30 "$log" 2>/dev/null || echo "(no log file at $log)"
  echo "---- EngineCore / Worker lines ----"
  grep -nE "EngineCore|Worker|WorkerProc" "$log" 2>/dev/null | head -30 || true
  echo "================= ROOT CAUSE (first error in the file) ================="
  # First real traceback, plus the first hard error line of any shape. One of these is the cause.
  # First traceback, ended at its terminating "SomeError: ..." line rather than a fixed window --
  # a fixed window pads the answer with whatever INFO noise followed it.
  awk '/Traceback \(most recent call last\)/{f=1}
       f{print; n++}
       f && /[A-Za-z_.]+(Error|Exception): / && n>1{exit}
       n>60{exit}' "$log" 2>/dev/null || true
  grep -nE "[A-Za-z_.]+(Error|Exception): |CUDA error|out of memory|No module named|not supported|Unsupported|Failed to import" \
    "$log" 2>/dev/null | head -12 || true
  echo "======================================================================="
}

# Every exit path that can fail AFTER the servers are up must dump. The self-test path did not, and
# that is why attempt 3's server-side traceback was never recovered.
dump_all_vllm_logs () {  # $1 = reason
  echo "[entry] dumping all vLLM replica logs (reason: $1)"
  for g in $(seq 0 $((GPUS - 1))); do
    [ -f "$WORK/vllm_gpu$g.log" ] || continue
    aws s3 cp "$WORK/vllm_gpu$g.log" "$LOG_PREFIX/vllm_gpu$g.log" --only-show-errors || true
  done
  for f in "$WORK"/pass1_shard*.log "$WORK"/pass2_shard*.log; do
    [ -f "$f" ] && aws s3 cp "$f" "$LOG_PREFIX/$(basename "$f")" --only-show-errors || true
  done
  # Only gpu0 is dumped to stdout -- it served the self-test and 8 full dumps would bury the answer.
  dump_vllm_failure "$WORK/vllm_gpu0.log" 0
  [ -s "$WORK/selftest_triage.txt" ] && aws s3 cp "$WORK/selftest_triage.txt" \
    "$LOG_PREFIX/selftest_triage.txt" --only-show-errors || true
  if [ -s "$WORK/selftest_triage.txt" ]; then
    echo "================= SELF-TEST TRIAGE (the distilled answer) ================="
    cat "$WORK/selftest_triage.txt"
    echo "=========================================================================="
  fi
}

# A bare `wait` waits on EVERY job of this shell -- which includes the 8 vLLM servers AND the
# never-exiting 60 s sync loop. It therefore can NEVER return: the job runs to max_run and is killed
# even when every shard finished, and a shard client that dies is invisible until then. Wait only on
# the CLIENT pids, and make a non-zero client exit fail the job loudly.
wait_clients () {  # $1 = stage label
  local rc=0 pid status
  for pid in "${CLIENT_PIDS[@]}"; do
    # Capture BEFORE testing: `if ! wait "$pid"; then status=$?` yields the status of the negation
    # (always 0), which would report every failure as "exited NON-ZERO (0)".
    # `wait "$pid"` as a SIMPLE command aborts the script under `set -e` when the client exited
    # non-zero -- before status is read, before any tail or upload. `|| status=$?` puts it in a
    # condition context, which suppresses errexit and preserves the real status.
    status=0; wait "$pid" || status=$?
    if [ "$status" != 0 ]; then
      echo "[entry] $1 shard client pid $pid exited NON-ZERO ($status)"
      rc=1
    fi
  done
  for f in "$WORK"/${1}_shard*.log; do
    [ -f "$f" ] && aws s3 cp "$f" "$LOG_PREFIX/$(basename "$f")" --only-show-errors || true
  done
  if [ "$rc" != 0 ]; then
    echo "[entry] ---- tail of each $1 shard log ----"
    for f in "$WORK"/${1}_shard*.log; do
      [ -f "$f" ] || continue
      echo "---- $(basename "$f") ----"; tail -25 "$f"
    done
    echo "[entry] FATAL: at least one $1 shard client failed"
    return 1
  fi
  echo "[entry] all $1 shard clients exited 0"
}

start_servers() {
  # ---- FlashInfer cubin consumers, both keyed on SM90 (Hopper) -------------------------------
  # DIAGNOSED 2026-09-02 from the replica log of attempt 6. The crash was NOT the GDN prefill
  # kernel: the traceback runs
  #   qwen_gdn_linear_attn.forward_cuda -> linear.py:598 -> quantization/fp8.py:479 apply
  #   -> BlockScaledMMLinearKernel:132 -> scaled_mm/flashinfer.py:194 apply_block_scaled_mm
  #   -> flashinfer/gemm/gemm_base.py fp8_blockscale_gemm_sm90 -> tvm_ffi -> !cubin.empty()
  # i.e. the FP8 BLOCK-SCALE LINEAR GEMM, selected because
  #   is_flashinfer_fp8_blockscale_gemm_supported() = VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
  #                                                   and has_flashinfer_fp8_blockscale_gemm()
  # and the kernel is sm90-only -- so the local 5090 (SM120) could never select it and the whole
  # corpus ran on a fallback kernel, while every Hopper node selects it and then cannot load the
  # cubin (no egress for prebuilt cubins, no nvcc to JIT). Setting the switch to 0 short-circuits
  # that `and`, and the chooser falls through to the Cutlass/Triton block-scaled kernels.
  #
  # Exported HERE rather than only in serve_vllm.sh on purpose: `exec env ...` in that script does
  # not use -i, so an exported var reaches vLLM even if a stale serve_vllm.sh is on the node.
  export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}"
  # Insurance, not a second diagnosis: with the FlashInfer base disqualified, the next block-scaled
  # candidate is DeepGEMM, which also compiles at runtime. The node's 196-package install list has
  # NO deep_gemm, so has_deep_gemm() is already False and this is a no-op there -- but stating it
  # explicitly means the selector cannot drift onto a second JIT path on a differently-built node.
  # Selection then lands on the CUTLASS block-scaled kernel, which is compiled into the vLLM wheel
  # and needs no toolchain, and which accepts the same act group_shape=(1,128) this checkpoint uses.
  export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
  echo "[entry] VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM"
  echo "[entry] VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=$VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"

  # STALE-BUNDLE ASSERTION. Attempt 6 shipped a current deliberation_entry.sh but an OLD
  # serve_vllm.sh (its startup echo had no `gdn=` field and the engine logged `requested=auto`),
  # so a fix that lived only in that file silently did nothing -- after a ~3 h queue wait. Fail in
  # seconds instead.
  if ! grep -q "gdn-prefill-backend" "$REPO/scripts/deliberation/serve_vllm.sh"; then
    echo "[entry] FATAL: serve_vllm.sh on this node predates the GDN-backend fix -- the source"
    echo "        bundle is STALE. Re-submit from a tree where scripts/deliberation/serve_vllm.sh"
    echo "        contains --gdn-prefill-backend."
    exit 2
  fi

  SERVER_PIDS=()
  for g in $(seq 0 $((GPUS - 1))); do
    WSM_GPU="$g" WSM_PORT=$((8100 + g)) WSM_MODEL="${WSM_DELIB_MODEL}" \
      WSM_ENFORCE_EAGER="${WSM_ENFORCE_EAGER:-1}" WSM_MAX_MODEL_LEN="${WSM_MAX_MODEL_LEN:-16384}" \
      WSM_VLLM_PYTHON="$PYV" WSM_ATTENTION_BACKEND="${WSM_ATTENTION_BACKEND:-}" \
      "$REPO/scripts/deliberation/serve_vllm.sh" >> "$WORK/vllm_gpu$g.log" 2>&1 &
    SERVER_PIDS+=($!)
  done
  for g in $(seq 0 $((GPUS - 1))); do
    pid="${SERVER_PIDS[$g]}"
    up=0
    for _ in $(seq 1 180); do
      if curl -sf -m 3 "http://127.0.0.1:$((8100 + g))/v1/models" >/dev/null; then up=1; break; fi
      # FAIL FAST: if the replica has already exited there is nothing to wait for. The first node
      # failure burned 30 minutes of a p5 polling a process that was long dead.
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[entry] vLLM on gpu $g EXITED during startup (pid $pid gone) — not waiting out the poll"
        break
      fi
      sleep 10
    done
    if [ "$up" != "1" ] && ! curl -sf -m 3 "http://127.0.0.1:$((8100 + g))/v1/models" >/dev/null; then
      echo "[entry] vLLM on gpu $g never came up"
      dump_vllm_failure "$WORK/vllm_gpu$g.log" "$g"
      aws s3 cp "$WORK/vllm_gpu$g.log" "$LOG_PREFIX/vllm_gpu$g.log" --only-show-errors || true
      exit 1
    fi
  done
  echo "[entry] $GPUS vLLM replicas ready"
  # The pins are only real if the engine acted on them. Echo the three lines that prove it, so a
  # SUCCESSFUL run also carries the evidence rather than only a failing one.
  echo "[entry] ---- kernel pins as the engine resolved them (gpu0) ----"
  grep -hE "Selected .* for Fp8LinearMethod|GDN prefill kernel" "$WORK/vllm_gpu0.log" 2>/dev/null \
    | head -4 || echo "[entry] (no kernel-selection lines found in gpu0 log)"
}

# ---- D8 vision self-test, ON THE NODE, before any shard spends a token -------------------------
# The local gate verified Qwen3.8-27B *NVFP4* vision on an RTX 5090. p5 runs *FP8* on H100, which
# cannot be reproduced on a 32 GB card (FP8 weights are ~30 GB resident -- it OOMs). Rather than
# submit an unverified serving config, the job proves the multimodal + structured-output path here,
# on the real server, with the real schema, in ~20 s. A broken config costs minutes, not the job.
vision_selftest() {
  "$PYV" - <<'PYVT'
import base64, io, json, os, sys, urllib.error, urllib.request
sys.path.insert(0, os.environ["REPO"])
import numpy as np
from PIL import Image
from workspace_models.labels.caption_segments import DESCRIPTOR_SCHEMA, DESCRIPTOR_SYSTEM

URL = os.environ["SELFTEST_URL"]
MODEL = os.environ["WSM_DELIB_MODEL"]

def tile(rgb):
    a = np.zeros((256, 256, 3), np.uint8); a[:, :] = rgb
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

TILES = [tile(c) for c in [(200, 30, 30), (30, 200, 30), (30, 30, 200)] * 3]

def post(payload, timeout=600):
    """-> (status, body_text). A 500 body is where vLLM puts the ACTUAL error; the previous version
    let urllib raise and threw that body away, which is why attempt 3 told us nothing."""
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()
        except Exception:
            body = "(could not read error body)"
        return e.code, body
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"

def build(n_images, schema, effort="low"):
    content = [{"type": "text", "text": "Task family: SelfTest\nEpisode goal: pick up the red block.\n"
                                        "Frames below are start / middle / end, three views each."}]
    for u in TILES[:n_images]:
        content.append({"type": "image_url", "image_url": {"url": u}})
    p = {"model": MODEL,
         "messages": [{"role": "system", "content": DESCRIPTOR_SYSTEM},
                      {"role": "user", "content": content}],
         "max_tokens": 2048, "temperature": 0.0,
         "chat_template_kwargs": {"reasoning_effort": effort}}
    if schema:
        p["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "segment_descriptor", "schema": DESCRIPTOR_SCHEMA, "strict": True}}
    return p

status, body = post(build(9, True))

if status != 200:
    # TRIAGE. One node run should name the culprit instead of costing another round trip. Each probe
    # removes ONE suspect; the first that succeeds isolates the failing factor.
    print(f"[selftest] FAIL: HTTP {status}")
    print("[selftest] ---- server error body (this is the real error) ----")
    print(body[:4000])
    print("[selftest] ---- triage ----")
    probes = [
        ("9 images + schema  (the real request)", build(9, True)),
        ("9 images, NO schema (isolates grammar)", build(9, False)),
        ("1 image  + schema  (isolates image COUNT)", build(1, True)),
        ("1 image,  NO schema (isolates multimodal at all)", build(1, False)),
        ("0 images + schema  (isolates vision entirely)", build(0, True)),
        ("0 images, NO schema (plain text baseline)", build(0, False)),
    ]
    rows = []
    for label, payload in probes:
        st, bd = post(payload, timeout=180)
        if st == 200:
            row = f"PASS       {label}"
        else:
            first = (bd or "").strip().replace("\n", " ")[:200]
            row = f"FAIL({st}) {label} :: {first}"
        print(f"[selftest]   {row}")
        rows.append(row)
    # Persist the ladder so the shell can re-print it AFTER the log dump. Otherwise the 30 lines of
    # vLLM context push the six rows -- the actual diagnostic -- far above the end of the job log.
    summary = os.environ.get("SELFTEST_TRIAGE_OUT")
    if summary:
        with open(summary, "w") as fh:
            fh.write(f"first request: HTTP {status}\n")
            fh.write((body or "")[:600].strip() + "\n")
            fh.write("\n".join(rows) + "\n")
    raise SystemExit("[selftest] FAIL: see the server error body and triage table above")

d = json.loads(body)
ch = d["choices"][0]
u = d.get("usage", {})
txt = ch["message"].get("content") or ""
try:
    obj = json.loads(txt)
except Exception as e:
    raise SystemExit(f"[selftest] FAIL: response was not JSON ({e}); first 300 chars: {txt[:300]!r}")
missing = [k for k in DESCRIPTOR_SCHEMA["required"] if k not in obj]
print(f"[selftest] finish={ch.get('finish_reason')} prompt_tok={u.get('prompt_tokens')} "
      f"completion_tok={u.get('completion_tokens')} missing={missing}")
if ch.get("finish_reason") == "length":
    raise SystemExit("[selftest] FAIL: truncated at 2048 -- the cap is wrong for this config")
if missing:
    raise SystemExit(f"[selftest] FAIL: schema fields missing {missing}")
# 9 images at 256px must cost ~64 vision tokens each; a wildly different count means the image
# preprocessing differs from the one the whole cost model was measured on.
if not (900 < int(u.get("prompt_tokens", 0)) < 2200):
    raise SystemExit(f"[selftest] FAIL: prompt_tokens {u.get('prompt_tokens')} outside the "
                     "expected 9-image band -- vision preprocessing differs from the pilot")
print("[selftest] PASS: multimodal + structured output + token accounting all match the pilot")
PYVT
}

case "$STAGE" in
  pass1)
    start_servers
    # The self-test runs against a LIVE server, so its failures are server-side and its logs are the
    # only place the cause is written. Attempt 3 died here and uploaded nothing.
    if ! REPO="$REPO" SELFTEST_URL="http://127.0.0.1:8100/v1/chat/completions" \
         WSM_DELIB_MODEL="$WSM_DELIB_MODEL" \
         SELFTEST_TRIAGE_OUT="$WORK/selftest_triage.txt" vision_selftest; then
      dump_all_vllm_logs "vision self-test failed"
      exit 1
    fi
    CLIENT_PIDS=()
    for g in $(seq 0 $((GPUS - 1))); do
      SHARD=$((NODE_RANK * GPUS + g))
      if [ "$DOMAIN" = "robocerebra" ]; then
        # Domain-nested store, matching what the embed stage globs. No --tasks (RoboCerebra
        # enumerates its own 947 BDDL stems), no --labels-root, no --caption-hints-root.
        "$PYV" -m workspace_models.labels.caption_segments \
          --spec descriptor --backend vllm \
          --model "${WSM_DELIB_MODEL}" \
          --vllm-base-url "http://127.0.0.1:$((8100 + g))/v1" \
          --out "$OUT/robocerebra" \
          --domain robocerebra \
          --dataset-root "$RCB_ROOT" \
          --shard "$SHARD" --num-shards "$NUM_SHARDS" \
          --concurrency "${WSM_DELIB_CONCURRENCY:-64}" \
          --reasoning-effort "${WSM_DELIB_REASONING_EFFORT:-low}" \
          --max-new-tokens "${WSM_DELIB_MAX_TOKENS:-2048}" \
          ${WSM_DELIB_LIMIT_SEGMENTS:+--limit-segments "$WSM_DELIB_LIMIT_SEGMENTS"} \
          >> "$WORK/pass1_shard$SHARD.log" 2>&1 &
      else
        "$PYV" -m workspace_models.labels.caption_segments \
          --spec descriptor --backend vllm \
          --model "${WSM_DELIB_MODEL}" \
          --vllm-base-url "http://127.0.0.1:$((8100 + g))/v1" \
          --out "$OUT" \
          --tasks "${WSM_TASKS:?A10 task list must be shipped by the launcher}" \
          --dataset-root "$WORK/robocasa" \
          --labels-root "$WORK/labels" \
          --caption-hints-root "$WORK/captions" \
          --shard "$SHARD" --num-shards "$NUM_SHARDS" \
          --concurrency "${WSM_DELIB_CONCURRENCY:-64}" \
          --reasoning-effort "${WSM_DELIB_REASONING_EFFORT:-low}" \
          --max-new-tokens "${WSM_DELIB_MAX_TOKENS:-2048}" \
          ${WSM_DELIB_LIMIT_SEGMENTS:+--limit-segments "$WSM_DELIB_LIMIT_SEGMENTS"} \
          >> "$WORK/pass1_shard$SHARD.log" 2>&1 &
      fi
      CLIENT_PIDS+=($!)
    done
    wait_clients pass1
    # POST-STAGE ASSERTION. wait_clients proves the clients did not CRASH; it cannot prove they did
    # the WORK. The `--tasks all` bug (§37.3) would have produced eight clean exits and an empty
    # store, so success is asserted against the corpus, not inferred from exit codes.
    # NB all 994 RoboCerebra episodes produce a descriptor file -- the 8 truncated demos are short
    # by SEGMENTS (8,869 vs the 8,887 the case definitions declare), not by episodes.
    if [ -n "${WSM_DELIB_EXPECT_EPISODES:-}" ]; then
      "$PYV" - "$OUT/${DOMAIN}" "${WSM_DELIB_EXPECT_EPISODES}" "${WSM_DELIB_EXPECT_SEGMENTS:-0}" <<'PYCHECK' || exit 6
import json, pathlib, sys
root, want_ep, want_seg = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
files = sorted(root.glob("*/ep_*.descriptors.json"))
seg = 0
bad = []
for f in files:
    try:
        seg += len(json.loads(f.read_text()).get("descriptors", []))
    except Exception as e:
        bad.append(f"{f.name}: {e}")
print(f"[post] {root.name}: {len(files)} episode files, {seg} segments, {len(bad)} unparsable")
fail = []
if len(files) != want_ep:
    fail.append(f"episodes {len(files)} != {want_ep}")
if want_seg and seg != want_seg:
    fail.append(f"segments {seg} != {want_seg}")
if bad:
    fail.append(f"{len(bad)} unparsable, e.g. {bad[:3]}")
if fail:
    print("[post] FATAL: " + "; ".join(fail)); sys.exit(1)
print("[post] corpus assertion PASSED")
PYCHECK
    fi
    ;;

  embed)
    aws s3 sync "${WSM_DELIB_PASS1_S3_IN}" "$WORK/pass1" --only-show-errors
    # RoboCerebra's pass 1 lands under its OWN content-addressed run_id prefix, not inside the
    # frozen 3-domain store, so the 4-domain index needs both merged into one domain-nested tree.
    # Comma-separated so a later domain needs no further entry change.
    if [ -n "${WSM_DELIB_PASS1_EXTRA_S3_IN:-}" ]; then
      IFS=',' read -ra EXTRA <<< "$WSM_DELIB_PASS1_EXTRA_S3_IN"
      for u in "${EXTRA[@]}"; do
        [ -n "$u" ] || continue
        aws s3 sync "$u" "$WORK/pass1" --only-show-errors
        echo "[entry] merged extra pass1 store $u"
      done
    fi
    for d in robocasa remembench robomme robocerebra; do
      n=$(find "$WORK/pass1/$d" -name 'ep_*.descriptors.json' 2>/dev/null | wc -l)
      echo "[entry] pass1/$d: $n episode files"
    done
    # The pass-1 store is DOMAIN-NESTED (<store>/<domain>/<Task>/ep_*.descriptors.json). Passing the
    # store root as --robocasa-descriptors indexes ZERO segments, because stage_index globs
    # `*/ep_*.descriptors.json` one level down. Each domain root must be handed in separately, and
    # the domain label comes from the flag, not from the file (the shared RoboCasa/ReMemBench code
    # path stamped `domain: robocasa` on ReMemBench files).
    "$PYV" scripts/deliberation/pass2_deliberate.py --stage index \
      --store "$OUT" \
      --robocasa-descriptors "$WORK/pass1/robocasa" \
      --remembench-descriptors "$WORK/pass1/remembench" \
      --robomme-descriptors "$WORK/pass1/robomme" \
      --robocerebra-descriptors "$WORK/pass1/robocerebra"
    "$PYV" scripts/deliberation/pass2_deliberate.py --stage embed \
      --store "$OUT" --embed-model "${WSM_DELIB_EMBED_MODEL}" --embed-device cuda
    MINE_ARGS=()
    [ -n "${WSM_DELIB_ANCHOR_DOMAINS:-}" ] && MINE_ARGS+=(--anchor-domains "$WSM_DELIB_ANCHOR_DOMAINS")
    "$PYV" scripts/deliberation/pass2_deliberate.py --stage mine --store "$OUT" "${MINE_ARGS[@]+"${MINE_ARGS[@]}"}"
    ;;

  pass2)
    aws s3 sync "${WSM_DELIB_EMBED_S3_IN}" "$OUT" --only-show-errors
    start_servers
    CLIENT_PIDS=()
    for g in $(seq 0 $((GPUS - 1))); do
      SHARD=$((SHARD_OFFSET + NODE_RANK * GPUS + g))
      [ "$SHARD" -lt $((SHARD_OFFSET + SHARD_COUNT)) ] || continue
      "$PYV" scripts/deliberation/pass2_deliberate.py --stage judge \
        --store "$OUT" --model "${WSM_DELIB_MODEL}" \
        --vllm-base-url "http://127.0.0.1:$((8100 + g))/v1" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS" \
        --concurrency "${WSM_DELIB_CONCURRENCY:-64}" \
        --reasoning-effort "${WSM_DELIB_REASONING_EFFORT:-medium}" \
        --max-tokens "${WSM_DELIB_MAX_TOKENS:-4096}" \
        >> "$WORK/pass2_shard$SHARD.log" 2>&1 &
      CLIENT_PIDS+=($!)
    done
    wait_clients pass2
    # POST-STAGE: a pass-2 job may not report success unless it judged the anchors it claimed to,
    # and unless every bucket agrees on effort/model. Follow-on (a) is a FULL redo; without this a
    # mis-pointed --embed-s3-in would re-judge one domain and look complete (§46.4).
    if [ -n "${WSM_DELIB_EXPECT_ANCHORS:-}" ]; then
      "$PYV" - "$OUT" "${WSM_DELIB_EXPECT_ANCHORS}" "${WSM_DELIB_EXPECT_DOMAIN_ANCHORS:-}" <<'PYP2' || exit 7
import collections, json, pathlib, sys
out, want_n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
want_dom = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else {}
roots = [p for p in (out / "edges").iterdir() if p.is_dir()] if (out / "edges").is_dir() else []
if len(roots) != 1:
    print(f"[post] FATAL: expected exactly one edge store, found {[r.name for r in roots]}"); sys.exit(1)
root = roots[0]
per, combos, n = collections.Counter(), collections.Counter(), 0
for f in root.rglob("*.bucket.json"):
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        print(f"[post] FATAL: unparsable bucket {f.name}: {e}"); sys.exit(1)
    n += 1
    per[d["anchor"].split("/")[0]] += 1
    combos[(d.get("model"), d.get("reasoning_effort"))] += 1
print(f"[post] edge_store={root.name} anchors={n} per_domain={dict(sorted(per.items()))}")
print(f"[post] judge combos (model, effort) = {dict(combos)}")
fail = []
if n != want_n:
    fail.append(f"judged {n} anchors, expected {want_n}")
if want_dom and dict(per) != want_dom:
    fail.append(f"per-domain {dict(sorted(per.items()))} != {want_dom}")
if len(combos) > 1:
    fail.append(f"HETEROGENEOUS judge config across buckets: {dict(combos)} (A17 forbids mixing)")
if fail:
    print("[post] FATAL: " + "; ".join(fail)); sys.exit(1)
(model, effort), = combos.keys()
(root / "_homogeneity.json").write_text(json.dumps(
    {"model": model, "reasoning_effort": effort, "anchors": n,
     "per_domain": dict(sorted(per.items())),
     "note": "one field settles homogeneity: every bucket in this store was judged by this "
             "(model, reasoning_effort) pair"}, indent=1))
print("[post] corpus + homogeneity assertions PASSED")
PYP2
    fi
    # QA only from the job that owns shard 0 -- otherwise two halves race on qa.json
    [ "$SHARD_OFFSET" = "0" ] && "$PYV" scripts/deliberation/pass2_deliberate.py \
      --stage qa --store "$OUT" || true
    ;;

  *) echo "[entry] unknown stage $STAGE"; exit 2 ;;
esac

echo "[entry] stage $STAGE complete; final sync"
