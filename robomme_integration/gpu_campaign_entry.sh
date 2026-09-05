#!/usr/bin/env bash
# Hold one p5/H100 or p5e/H200 node while serially executing sealed single-task cells.
set -euo pipefail

echo "======== RoboMME GPU campaign | $(hostname) | $(date -u +%FT%TZ) ========"
CODE_DIR=/opt/ml/code
WORK_ROOT=${ROBOMME_CAMPAIGN_WORK_ROOT:-/opt/ml/robomme-campaign}
required=(
  ROBOMME_CAMPAIGN_MANIFEST_SOURCE ROBOMME_CAMPAIGN_MANIFEST_SHA256
  ROBOMME_CAMPAIGN_ID ROBOMME_CAMPAIGN_ATTEMPT_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 40; }
done
[[ "$WORK_ROOT" == /opt/ml/* && "$WORK_ROOT" != /opt/ml ]] || {
  echo "FATAL unsafe campaign work root $WORK_ROOT" >&2; exit 40;
}
MANIFEST="$CODE_DIR/$ROBOMME_CAMPAIGN_MANIFEST_SOURCE"
[[ -f "$MANIFEST" ]] || { echo "FATAL staged campaign manifest absent" >&2; exit 41; }

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE_DIR" python3 -B - "$MANIFEST" <<'PY'
import json, os, pathlib, sys
from campaign import validate_manifest

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
validate_manifest(value, code_dir=path.parent)
if value["manifest_sha256"] != os.environ["ROBOMME_CAMPAIGN_MANIFEST_SHA256"]:
    raise SystemExit("campaign manifest environment digest mismatch")
if value["campaign_id"] != os.environ["ROBOMME_CAMPAIGN_ID"]:
    raise SystemExit("campaign ID environment mismatch")
if value["attempt_id"] != os.environ["ROBOMME_CAMPAIGN_ATTEMPT_ID"]:
    raise SystemExit("campaign attempt ID environment mismatch")
PY

exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE_DIR" \
  python3 -B -m campaign --manifest "$MANIFEST" --code-dir "$CODE_DIR" --work-root "$WORK_ROOT"
