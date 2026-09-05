#!/usr/bin/env bash
# H13 §7 — detached 2-GPU fan-out for stage-A2 segment captions (all 7500 post-train episodes).
#
# One shard per 5090, both setsid+nohup'd so they survive the launching shell/agent. Resume is
# content-validating (see caption_segments.validate_existing), so re-running this script after any
# interruption simply picks up the episodes whose JSON is missing OR structurally wrong.
#
#   scripts/labels/run_caption_segments.sh              # launch both shards (guards the GPUs)
#   scripts/labels/run_caption_segments.sh --force      # launch even if the GPUs look busy
#   scripts/labels/run_caption_segments.sh --finalize   # write per-task manifests when done
#
# Logs: ~/Research/TRI/wsm_data/wsm_labels_captions/_logs/shard{0,1}.log (+ .pid)
set -euo pipefail

REPO="${REPO:-$HOME/Research/TRI/wsmv2}"
PY="${PY:-$HOME/Research/envs/vlm_labeler/bin/python}"
OUT="${OUT:-$HOME/Research/TRI/wsm_data/wsm_labels_captions}"
LOGS="$OUT/_logs"
BATCH="${BATCH:-4}"
MODE="${1:-run}"

mkdir -p "$LOGS"
cd "$REPO"

if [[ "$MODE" == "--finalize" ]]; then
  "$PY" -m workspace_models.labels.caption_segments --out "$OUT" --finalize-only
  exit 0
fi

# --- Contention guard. This pipeline must never contend with an in-flight eval/train campaign:
# the robomme campaign lanes admit only on a nearly-EMPTY GPU (required_free_gpu_bytes ~= 33.4 GB),
# so squatting 17 GB of Qwen weights STALLS the campaign rather than merely slowing it.
#
# A point-in-time nvidia-smi sample is NOT sufficient on its own — a multi-cell campaign cycles
# between "staging" (GPUs idle for a minute or two) and "evaluating" (GPUs full), so an instantaneous
# reading catches an idle window and green-lights a launch that collides seconds later. Hence the
# campaign-state and process checks below, which see the whole queue rather than this instant.
if [[ "$MODE" != "--force" ]]; then
  busy=0

  while IFS=, read -r idx util mem; do
    util="${util// /}"; mem="${mem// /}"
    if (( util > 20 || mem > 2000 )); then
      echo "GPU $idx busy: util=${util}% mem=${mem}MiB" >&2
      busy=1
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)

  # live local eval/serve processes
  if pgrep -f "execution_model_server" > /dev/null 2>&1 || pgrep -f "bin/vla-eval" > /dev/null 2>&1; then
    echo "a local vla-eval / model-server process is running" >&2
    busy=1
  fi

  # any campaign cell not in a terminal state => more GPU work is queued behind the idle window
  # Only cells touched recently count: a long-dead campaign leaves non-terminal cells behind
  # forever, and those must not block this pipeline for the rest of time.
  pending=$(STALE_H="${STALE_H:-3}" python3 - <<'PY' 2>/dev/null || true
import glob, json, os, time
TERM = {"complete", "completed", "failed", "skipped", "cancelled", "aborted", "error",
        "terminal_failure", "cancelled_without_claim", "blocked_resource_admission"}
cutoff = time.time() - float(os.environ.get("STALE_H", 3)) * 3600
hits = []
for f in glob.glob(os.path.expanduser("~/Research/TRI/robomme_eval/campaigns/*/state/cells/*.json")):
    try:
        if os.path.getmtime(f) < cutoff:
            continue
        d = json.load(open(f))
    except Exception:
        continue
    if str(d.get("status", "")).lower() not in TERM:
        camp = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
        hits.append(f"{camp}/{d.get('cell_id')}={d.get('status')}")
print("; ".join(hits[:6]))
PY
)
  if [[ -n "${pending// /}" ]]; then
    echo "campaign cells still in flight: $pending" >&2
    busy=1
  fi

  if (( busy )); then
    echo "REFUSING to launch — other GPU work is in flight. Re-run with --force to override." >&2
    exit 3
  fi
fi

for shard in 0 1; do
  log="$LOGS/shard${shard}.log"
  if [[ -f "$LOGS/shard${shard}.pid" ]] && kill -0 "$(cat "$LOGS/shard${shard}.pid")" 2>/dev/null; then
    echo "shard $shard already running (pid $(cat "$LOGS/shard${shard}.pid")) — skipping"
    continue
  fi
  setsid nohup "$PY" -m workspace_models.labels.caption_segments \
      --out "$OUT" --device "cuda:${shard}" --shard "$shard" --num-shards 2 \
      --batch-size "$BATCH" >> "$log" 2>&1 < /dev/null &
  echo $! > "$LOGS/shard${shard}.pid"
  echo "shard $shard -> pid $(cat "$LOGS/shard${shard}.pid")  log $log"
done

echo
echo "progress:  find $OUT -name 'ep_*.captions.json' | wc -l    # target 7500"
echo "tail:      tail -f $LOGS/shard0.log"
echo "finalize:  $0 --finalize"
