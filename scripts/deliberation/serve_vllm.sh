#!/usr/bin/env bash
# H14 deliberation: one vLLM OpenAI-compatible server per GPU.
#
# Local (RTX 5090, 32 GB): NVFP4 TP1 + --enforce-eager (s3 §1: required for 5090+NVFP4).
# p5   (H100 80 GB):       FP8 TP1, one replica per GPU, no --enforce-eager.
#
# GPU DISCIPLINE: this script NEVER selects a GPU for you. Pass WSM_GPU explicitly.
# On the CMU box GPU1 belongs to the RoboCerebra v3 ladder -- do not point this at it.
#
#   WSM_GPU=0 WSM_MODEL=unsloth/Qwen3.8-27B-NVFP4 scripts/deliberation/serve_vllm.sh
set -euo pipefail

: "${WSM_GPU:?set WSM_GPU (e.g. 0). This script refuses to guess a GPU.}"
MODEL="${WSM_MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
PORT="${WSM_PORT:-$((8100 + WSM_GPU))}"
MAX_LEN="${WSM_MAX_MODEL_LEN:-16384}"
UTIL="${WSM_GPU_UTIL:-0.90}"
MAX_IMAGES="${WSM_MAX_IMAGES:-12}"
PY="${WSM_VLLM_PYTHON:-/home/sarveshp/Research/envs/vllm_delib/bin/python}"
EAGER="${WSM_ENFORCE_EAGER:-1}"
LOG="${WSM_LOG:-/home/sarveshp/Research/TRI/wsm_data/deliberation/logs/vllm_gpu${WSM_GPU}.log}"

mkdir -p "$(dirname "$LOG")"

# NB: `set -e` + `[ test ] && cmd` kills the script when the test is false. Use if-blocks.
extra=()
if [ "$EAGER" = "1" ]; then extra+=(--enforce-eager); fi
if [ -n "${WSM_KV_CACHE_DTYPE:-}" ]; then extra+=(--kv-cache-dtype "$WSM_KV_CACHE_DTYPE"); fi

# This box has no CUDA toolkit (no nvcc), so FlashInfer's JIT-compiled top-k/top-p sampler cannot
# build and the engine dies in profile_run. The PyTorch sampler is exact and we decode at
# temperature 0 anyway. On a node WITH nvcc, unset WSM_FLASHINFER_SAMPLER to use it.
FI_SAMPLER="${WSM_FLASHINFER_SAMPLER:-0}"
# ...and the FLASHINFER *attention* backend JITs too, but only on the first REAL decode -- the dummy
# profile_run does not touch it, so the engine starts healthy and then dies with EngineDeadError on
# request 1 (observed 2026-08-22 03:56). TRITON_ATTN needs no toolchain. Override on a node with nvcc.
# NOTE: VLLM_ATTENTION_BACKEND no longer exists in 0.27.x -- the backend is chosen through
# --attention-config. Setting the old env var silently does nothing.
ATTN="${WSM_ATTENTION_BACKEND:-TRITON_ATTN}"
if [ -n "$ATTN" ]; then extra+=(--attention-config "{\"backend\": \"${ATTN}\"}"); fi

# Multimodal processor cache: OFF by default here. Observed 2026-08-22: after a burst of rejected
# requests, every subsequent multimodal request died with
#   AssertionError: Expected a cached item for mm_hash=...
# i.e. the cache can be left referencing entries the engine no longer holds, and the server stays up
# while failing 100% of image requests. Each of our prompts has unique frames, so the cache buys
# almost nothing (the pilot measured a 60% "MM cache hit rate" only because the same 9 images are
# retried); a poisoned cache midway through a 21k-segment cloud run would be far more expensive.
MM_CACHE_GB="${WSM_MM_PROCESSOR_CACHE_GB:-0}"
extra+=(--mm-processor-cache-gb "$MM_CACHE_GB")

# GDN PREFILL BACKEND — the 2026-09-01 node kill, diagnosed rather than guessed.
# Qwen3.8-27B is HYBRID: 48 `linear_attention` (gated-deltanet) layers + 16 full-attention. The GDN
# layers do NOT go through --attention-config; they resolve their own prefill kernel in
# `_resolve_gdn_prefill_backend` (mamba/gdn/qwen_gdn_linear_attn.py), where the default "auto" plus
# `is_device_capability(90)` — Hopper, i.e. BOTH H100 and H200 — selects **flashinfer with no
# further constraints**. FlashInfer then loads TVM-FFI cubins on the FIRST REAL PREFILL, and on a
# node with no egress for prebuilt cubins and no nvcc to JIT it dies with
#   tvm.error.InternalError: Assertion failed: !cubin.empty() || isPathValid(path_)
# -> EngineDeadError, after a perfectly healthy startup. The local 5090 is SM120, which matches
# neither the SM90 nor the SM10x branch, so it silently resolved to "triton" — which is why this
# never reproduced locally. Pinning triton makes both venues take the SAME kernel path.
extra+=(--gdn-prefill-backend "${WSM_GDN_PREFILL_BACKEND:-triton}")

# CUDA JIT is OPT-IN and OFF by default on this box. Pointing CUDA_HOME at the pip CUDA-13 toolkit
# (nvidia-cuda-nvcc-cu13) gets further than having no nvcc at all, and then fails harder: FlashInfer
# ships CUDA-12-era cccl headers, so `fp4_gemm_cutlass_sm120` dies with
#   "CUDA compiler and CUDA toolkit headers are incompatible"
# (observed 2026-08-22 04:05). With no CUDA_HOME, vLLM uses its prebuilt CutlassNvFp4LinearKernel
# and the Triton attention/GDN kernels, which need no toolchain at all. On a node with a MATCHED
# system CUDA toolkit, set WSM_USE_JIT_CUDA=1 (and expect a long first-run compile).
if [ "${WSM_USE_JIT_CUDA:-0}" = "1" ]; then
  CU="${WSM_CUDA_HOME:-$(dirname "$PY")/../lib/python3.12/site-packages/nvidia/cu13}"
  if [ -x "$CU/bin/nvcc" ]; then
    export CUDA_HOME="$CU"
    export PATH="$CU/bin:$PATH"
  fi
  # `ninja` lives in the venv bin and FlashInfer's JIT shells out to it by bare name.
  export PATH="$(dirname "$PY"):$PATH"
fi

echo "[serve_vllm] gpu=$WSM_GPU port=$PORT model=$MODEL max_len=$MAX_LEN attn=$ATTN "\
"gdn=${WSM_GDN_PREFILL_BACKEND:-triton} fi_bs_gemm=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0} "\
"cuda_home=${CUDA_HOME:-none} log=$LOG"
exec env CUDA_VISIBLE_DEVICES="$WSM_GPU" VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  VLLM_USE_FLASHINFER_SAMPLER="$FI_SAMPLER" \
  VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}" \
  VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}" \
  CUDA_HOME="${CUDA_HOME:-}" PATH="$PATH" \
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$UTIL" \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}, \"video\": 0}" \
    --reasoning-parser "${WSM_REASONING_PARSER:-qwen3}" \
    --no-enable-log-requests \
    "${extra[@]}"
# NOTE: no `| tee` -- a pipeline makes the server a child of a shell, and when the launching shell
# is reaped the whole tree goes with it (observed 2026-08-22: clean shutdown 78 s after startup).
# Redirect in the caller instead:  scripts/deliberation/serve_vllm.sh >> "$LOG" 2>&1
