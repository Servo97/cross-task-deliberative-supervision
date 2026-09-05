#!/usr/bin/env bash
# Run exactly one RoboMME command under an atomic shared-box lease.
set -euo pipefail

[[ "$(hostname)" == ip-10-242-9-112* ]] || {
  echo "FATAL this lease is only valid on nagababa" >&2; exit 70;
}
[[ "${1:-}" == --run-id && -n "${2:-}" && "${3:-}" == -- ]] || {
  echo "usage: $0 --run-id <safe-id> -- <command> [args...]" >&2; exit 2;
}
RUN_ID="$2"
shift 3
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$#" -gt 0 ]] || {
  echo "FATAL unsafe run ID or empty command" >&2; exit 2;
}

LEASE_ROOT=/data/work/leases
LEASE="$LEASE_ROOT/robomme-eval"
TOKEN="${RUN_ID}-$$-$(date -u +%s)"
mkdir -p "$LEASE_ROOT"
if ! mkdir "$LEASE" 2>/dev/null; then
  echo "FATAL RoboMME EC2 lease already held:" >&2
  sed -n '1,120p' "$LEASE/owner.json" 2>/dev/null >&2 || true
  exit 71
fi
python3 - "$LEASE/owner.json" "$RUN_ID" "$TOKEN" "$$" <<'PY'
import datetime, json, os, socket, sys
path, run_id, token, pid = sys.argv[1:]
value = {
    "schema_version": 1,
    "owner": os.environ.get("USER", "unknown"),
    "run_id": run_id,
    "token": token,
    "host": socket.gethostname(),
    "pid": int(pid),
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

release() {
  if [[ -f "$LEASE/owner.json" ]] && grep -Fq "\"token\": \"$TOKEN\"" "$LEASE/owner.json"; then
    rm -f "$LEASE/owner.json"
    rmdir "$LEASE" 2>/dev/null || true
  fi
}
CHILD_PID=""
RECEIVED_SIGNAL=""
forward() {
  local signal_name="$1"
  RECEIVED_SIGNAL="$signal_name"
  if [[ -n "$CHILD_PID" ]]; then
    kill -s "$signal_name" -- "-$CHILD_PID" 2>/dev/null || true
  fi
}
trap release EXIT
trap 'forward INT' INT
trap 'forward TERM' TERM

export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
set +e
setsid "$@" &
CHILD_PID=$!
wait "$CHILD_PID"
STATUS=$?
if [[ -n "$RECEIVED_SIGNAL" ]]; then
  while kill -0 "$CHILD_PID" 2>/dev/null; do
    wait "$CHILD_PID" 2>/dev/null || true
  done
  STATUS=$((128 + $(kill -l "$RECEIVED_SIGNAL")))
fi
set -e
release
trap - EXIT INT TERM
exit "$STATUS"
