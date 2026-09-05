#!/usr/bin/env bash
# H13 §7 — detached watcher: launch the full caption run the moment it is POSITIVELY safe.
#
# The captions run needs ~17 GB of GPU per shard. The robomme campaign's lanes admit only on a
# nearly-empty GPU, and it cycles staging (GPUs idle 1-2 min) <-> evaluating (GPUs full). So
# "the GPUs look free right now" is NOT evidence of safety — that race already caused two
# accidental launches. This watcher therefore waits for a POSITIVE completion signal read from the
# campaign's own cell state, and only then re-checks the launcher's guard.
#
# LAUNCH CONDITION (both must hold):
#   (a) every cell in the campaign's queue manifest has a state file whose status is terminal
#       -- a missing state file counts as NOT terminal (that cell has not run yet);
#       fallback, used only if the manifest/state cannot be read: no eval/model-server process
#       alive AND both GPUs < 1 GiB used, sustained across 3 consecutive polls (45 min);
#   (b) scripts/labels/run_caption_segments.sh's own guard passes at the launch instant.
#
# Read-only w.r.t. the campaign: it inspects state JSON and never signals, writes or touches it.
#
#   setsid nohup scripts/labels/caption_watcher.sh > /dev/null 2>&1 < /dev/null &
#
# NOT `set -e`: a transient nvidia-smi/JSON hiccup must never kill a multi-hour watcher.
set -uo pipefail

REPO="${REPO:-$HOME/Research/TRI/wsmv2}"
PY="${PY:-$HOME/Research/envs/vlm_labeler/bin/python}"
OUT="${OUT:-$HOME/Research/TRI/wsm_data/wsm_labels_captions}"
LOGS="$OUT/_logs"
LOG="$LOGS/watcher.log"
STATUS_DOC="$REPO/internal_planning_and_todos/aug_12/h13_captions_status.md"
CAMPAIGN="${CAMPAIGN:-pick-button-representation-fixed50-local5090-v3r}"
ROBOMME="$HOME/Research/TRI/robomme_eval"
POLL_S="${POLL_S:-900}"          # 15 min
NEED_STREAK="${NEED_STREAK:-3}"  # fallback path: 3 consecutive clean polls = 45 min
MAX_HOURS="${MAX_HOURS:-72}"
SPOTCHECK_DELAY_S="${SPOTCHECK_DELAY_S:-900}"

mkdir -p "$LOGS"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

# --- singleton, self-excluding: never count this shell (or its children) as "another watcher" ---
if [[ -f "$LOGS/watcher.pid" ]]; then
  old=$(cat "$LOGS/watcher.pid" 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" != "$$" ]] && kill -0 "$old" 2>/dev/null; then
    log "another watcher already running (pid $old) — this instance ($$) exits"
    exit 0
  fi
fi
echo "$$" > "$LOGS/watcher.pid"
trap 'rm -f "$LOGS/watcher.pid"' EXIT

# pgrep that can never match this watcher, its subshells, or its python helpers
proc_alive() {
  local pat="$1" pids
  pids=$(pgrep -f -- "$pat" 2>/dev/null | grep -vx -e "$$" -e "${PPID:-0}" || true)
  [[ -n "${pids// /}" ]]
}

# --- (a) positive completion: every queued cell has a terminal state file -----------------------
# exit 0 = all terminal, 1 = still pending, 2 = cannot read (caller falls back)
campaign_all_terminal() {
  CAMPAIGN="$CAMPAIGN" ROBOMME="$ROBOMME" python3 - <<'PY'
import json, os, sys
camp, robomme = os.environ["CAMPAIGN"], os.environ["ROBOMME"]
queue = os.path.join(robomme, "campaign-runtime", "queues", f"{camp}.queue.json")
state = os.path.join(robomme, "campaigns", camp, "state", "cells")
TERMINAL = {"complete", "completed", "failed", "terminal_failure", "skipped", "cancelled",
            "cancelled_without_claim", "aborted", "error"}
try:
    cells = json.load(open(queue))["cells"]
    ids = [c["cell_id"] for c in cells]
    if not ids:
        raise ValueError("empty cell list")
except Exception as e:
    print(f"UNREADABLE queue: {e}")
    sys.exit(2)
pending = []
for cid in ids:
    p = os.path.join(state, f"{cid}.json")
    if not os.path.exists(p):
        pending.append(f"{cid}=NOSTATE")
        continue
    try:
        st = str(json.load(open(p)).get("status", "")).lower()
    except Exception as e:
        print(f"UNREADABLE state {cid}: {e}")
        sys.exit(2)
    if st not in TERMINAL:
        pending.append(f"{cid}={st}")
if pending:
    print(f"PENDING {len(pending)}/{len(ids)}: {', '.join(pending[:4])}")
    sys.exit(1)
print(f"ALL_TERMINAL {len(ids)}/{len(ids)} cells")
sys.exit(0)
PY
}

gpus_idle() {  # both GPUs under 1 GiB
  local mem busy=0
  while read -r mem; do
    mem="${mem// /}"
    [[ -z "$mem" ]] && continue
    (( mem >= 1024 )) && busy=1
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  (( busy == 0 ))
}

log "watcher START pid=$$ campaign=$CAMPAIGN poll=${POLL_S}s max=${MAX_HOURS}h"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
streak=0
poll=0

while :; do
  poll=$((poll + 1))
  if (( $(date +%s) > deadline )); then
    log "poll $poll: DEADLINE ${MAX_HOURS}h reached without a safe window — exiting, nothing launched"
    exit 0
  fi

  msg=$(campaign_all_terminal); rc=$?
  safe=0
  if (( rc == 0 )); then
    safe=1; streak=0
    log "poll $poll: campaign $msg -> condition (a) SATISFIED"
  elif (( rc == 1 )); then
    streak=0
    log "poll $poll: campaign $msg -> waiting"
  else
    # fallback: process + memory quiet, sustained
    if proc_alive "execution_model_server" || proc_alive "bin/vla-eval"; then
      streak=0
      log "poll $poll: $msg; fallback: eval/model-server process alive -> waiting (streak reset)"
    elif ! gpus_idle; then
      streak=0
      log "poll $poll: $msg; fallback: GPU memory >= 1 GiB -> waiting (streak reset)"
    else
      streak=$((streak + 1))
      log "poll $poll: $msg; fallback: quiet streak $streak/$NEED_STREAK"
      (( streak >= NEED_STREAK )) && safe=1
    fi
  fi

  if (( safe )); then
    log "poll $poll: condition (a) met — invoking launcher (its guard is condition (b))"
    out=$("$REPO/scripts/labels/run_caption_segments.sh" 2>&1); lrc=$?
    while IFS= read -r line; do [[ -n "$line" ]] && log "  launcher| $line"; done <<< "$out"
    if (( lrc == 0 )); then
      p0=$(cat "$LOGS/shard0.pid" 2>/dev/null || echo '?')
      p1=$(cat "$LOGS/shard1.pid" 2>/dev/null || echo '?')
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      log "LAUNCHED at $ts — shard0 pid=$p0 shard1 pid=$p1 (logs $LOGS/shard{0,1}.log)"
      printf '\n- Full run launched by the watcher at %s — shard0 pid %s, shard1 pid %s; logs `%s/shard{0,1}.log`.\n' \
        "$ts" "$p0" "$p1" "$LOGS" >> "$STATUS_DOC" 2>/dev/null \
        && log "appended launch line to $STATUS_DOC" || log "WARN could not append to $STATUS_DOC"

      # honor the 1-task canary caveat: cross-task structural spot-check once the run is moving
      log "sleeping ${SPOTCHECK_DELAY_S}s before the cross-task spot-check"
      sleep "$SPOTCHECK_DELAY_S"
      sc=$(cd "$REPO" && "$PY" -m workspace_models.labels.qa_captions --spotcheck --min-tasks 2 2>&1)
      while IFS= read -r line; do [[ -n "$line" ]] && log "  $line"; done <<< "$sc"
      log "watcher DONE — the run's own logs are the record from here"
      exit 0
    fi
    log "poll $poll: launcher REFUSED (exit $lrc) — condition (b) failed; retrying next poll"
    streak=0
  fi
  sleep "$POLL_S"
done
