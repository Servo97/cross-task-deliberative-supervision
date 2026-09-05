#!/usr/bin/env bash
# Run one official fixed-50 RoboMME evaluation on nagababa. Invoke only under the shared lease.
set -euo pipefail

[[ "$(hostname)" == ip-10-242-9-112* ]] || { echo "FATAL nagababa only" >&2; exit 70; }
[[ -f /data/work/leases/robomme-eval/owner.json ]] || {
  echo "FATAL evaluation requires with_robomme_lease_nagababa.sh" >&2; exit 71;
}
required=(
  ROBOMME_ARM ROBOMME_TASK ROBOMME_RUN_ID ROBOMME_EVAL_ID CHECKPOINT_S3
  CHECKPOINT_TREE_MANIFEST_S3 CHECKPOINT_TREE_MANIFEST_SHA256 COMPLETION_CLAIM_S3
  EVAL_OUTPUT_S3 EVAL_CLAIM_S3
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "FATAL missing $name" >&2; exit 72; }
done
[[ "$ROBOMME_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$ROBOMME_EVAL_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 72
[[ "$CHECKPOINT_TREE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 72

ROOT=/data/work/robomme_eval
ENV_JSON="$ROOT/ENV.json"
[[ -f "$ENV_JSON" ]] || { echo "FATAL run setup_robomme_nagababa.sh first" >&2; exit 73; }
eval "$(python3 - "$ENV_JSON" <<'PY'
import json, shlex, sys
v = json.load(open(sys.argv[1]))
for env, key in (
    ("SOURCE_ROOT", "source_root"),
    ("OPENPI_ROOT", "openpi_root"),
    ("POLICY_PY", "policy_python"),
    ("OPENPI_DATA_HOME", "openpi_data_home"),
    ("VLA_EVAL", "vla_eval"),
    ("CONTAINER_IMAGE", "container_image"),
):
    print(f"export {env}={shlex.quote(v[key])}")
PY
)"
export PYTHONPATH="$SOURCE_ROOT:$OPENPI_ROOT/src"
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"

# The lease coordinates cooperating agents; this catches an unleased process before we allocate.
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
  echo "FATAL one or more GPUs already have a compute process" >&2; exit 74
fi

case "$ROBOMME_TASK" in
  PickXtimes) BENCHMARK="$SOURCE_ROOT/robomme_integration/eval/configs/pickxtimes.yaml" ;;
  ButtonUnmaskSwap) BENCHMARK="$SOURCE_ROOT/robomme_integration/eval/configs/buttonunmaskswap.yaml" ;;
  *) echo "FATAL no locked single-task config for $ROBOMME_TASK" >&2; exit 75 ;;
esac
[[ -f "$BENCHMARK" ]] || exit 75

CKPT="$ROOT/checkpoints/$ROBOMME_RUN_ID/19999"
TREE="$ROOT/checkpoints/$ROBOMME_RUN_ID/tree-manifest.json"
CLAIM="$ROOT/checkpoints/$ROBOMME_RUN_ID/completion.json"
mkdir -p "$(dirname "$CKPT")"
if [[ ! -f "$CKPT/.verified" ]]; then
  [[ ! -e "$CKPT" ]] || { echo "FATAL partial local checkpoint $CKPT" >&2; exit 76; }
  mkdir -p "$CKPT"
  aws s3 sync "$CHECKPOINT_S3" "$CKPT" --only-show-errors \
    --exclude '*' --include 'params/*' --include 'assets/*'
  aws s3 cp "$CHECKPOINT_TREE_MANIFEST_S3" "$TREE.incomplete" --only-show-errors
  [[ "$(sha256sum "$TREE.incomplete" | awk '{print $1}')" == "$CHECKPOINT_TREE_MANIFEST_SHA256" ]] || {
    echo "FATAL checkpoint tree-manifest checksum mismatch" >&2; exit 76;
  }
  mv "$TREE.incomplete" "$TREE"
  PYTHONPATH="$SOURCE_ROOT" python3 -m robomme_integration.fleet.checkpoint \
    --root "$CKPT" --uri "$CHECKPOINT_S3" --output "$TREE" --verify \
    --expected-uri "$CHECKPOINT_S3" >/dev/null
  touch "$CKPT/.verified"
fi
aws s3 cp "$COMPLETION_CLAIM_S3" "$CLAIM.incomplete" --only-show-errors
python3 - "$CLAIM.incomplete" <<'PY'
import json, os, sys
v = json.load(open(sys.argv[1]))
expected = {
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "step": 19999,
    "checkpoint_uri": os.environ["CHECKPOINT_S3"],
    "tree_manifest_uri": os.environ["CHECKPOINT_TREE_MANIFEST_S3"],
    "tree_manifest_sha256": os.environ["CHECKPOINT_TREE_MANIFEST_SHA256"],
}
for key, value in expected.items():
    if v.get(key) != value:
        raise SystemExit(f"completion claim mismatch {key}: {v.get(key)!r} != {value!r}")
PY
mv "$CLAIM.incomplete" "$CLAIM"

OUT=/data2/robomme_evals/$ROBOMME_RUN_ID/$ROBOMME_EVAL_ID
[[ ! -e "$OUT" ]] || { echo "FATAL evaluation output already exists: $OUT" >&2; exit 77; }
mkdir -p "$(dirname "$OUT")"
EXTRA=()
if [[ "$ROBOMME_ARM" == q3 || "$ROBOMME_ARM" == wsm_cfg || \
      "$ROBOMME_ARM" == wsm_tanh || "$ROBOMME_ARM" == wsm_d8 ]]; then
  : "${ROBOMME_WORKSPACE_CHECKPOINT:?}"
  : "${ROBOMME_UPSTREAM_ROOT:?}"
  : "${ROBOMME_VISION_ENCODER_HOME:?}"
  EXTRA+=(
    --workspace-checkpoint "$ROBOMME_WORKSPACE_CHECKPOINT"
    --upstream-root "$ROBOMME_UPSTREAM_ROOT"
    --vision-encoder-home "$ROBOMME_VISION_ENCODER_HOME"
  )
fi

OPENPI_DATA_HOME="$OPENPI_DATA_HOME" "$POLICY_PY" -m robomme_integration.eval.launch_gpu_fleet \
  --source-root "$SOURCE_ROOT" \
  --checkpoint "$CKPT" \
  --arm "$ROBOMME_ARM" \
  --task "$ROBOMME_TASK" \
  --benchmark-config "$BENCHMARK" \
  --vla-eval "$VLA_EVAL" \
  --output-root "$OUT" \
  --eval-id "$ROBOMME_EVAL_ID" \
  --gpus 0,1,2,3 \
  --shards 16 \
  --cpu-range 0-191 \
  --container-image "$CONTAINER_IMAGE" \
  "${EXTRA[@]}"

[[ -f "$OUT/COMPLETED" ]] || { echo "FATAL evaluator did not complete" >&2; exit 78; }
SUMMARY="$OUT/result-claim.json"
python3 - "$OUT" "$SUMMARY" <<'PY'
import datetime, hashlib, json, os, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
launch = json.load(open(root / "eval/launch_manifest.json"))
if any(launch.get("returncodes", [])) or launch.get("episode_audit", {}).get("harness_failures"):
    raise SystemExit("evaluation launch manifest contains a process/harness failure")
candidates = []
for item in launch.get("materialized_results", []):
    path = pathlib.Path(item["path"])
    if path.suffix != ".json":
        continue
    value = json.load(open(path))
    episodes = []
    for task in value.get("tasks", []):
        episodes.extend(task.get("episodes", []))
    if episodes:
        candidates.append((path, episodes))
exact = [item for item in candidates if len(item[1]) == 50]
if len(exact) != 1 or launch.get("episode_audit", {}).get("episodes") != 50:
    raise SystemExit(f"fixed-50 evaluation has ambiguous episode outputs: {[(str(p), len(e)) for p,e in candidates]}")
episodes = exact[0][1]
success = sum(bool(ep.get("success", ep.get("metrics", {}).get("success", False))) for ep in episodes)
value = {
    "schema_version": 2,
    "kind": "robomme_fixed50_complete",
    "training_scope": "single_task",
    "run_id": os.environ["ROBOMME_RUN_ID"],
    "eval_id": os.environ["ROBOMME_EVAL_ID"],
    "task": os.environ["ROBOMME_TASK"],
    "arm": os.environ["ROBOMME_ARM"],
    "episodes": len(episodes),
    "successes": success,
    "checkpoint_uri": os.environ["CHECKPOINT_S3"],
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

EVIDENCE_DIR="$ROOT/evidence-files-$ROBOMME_EVAL_ID"
[[ ! -e "$EVIDENCE_DIR" ]] || { echo "FATAL stale evidence staging $EVIDENCE_DIR" >&2; exit 79; }
python3 - "$OUT" "$EVIDENCE_DIR" <<'PY'
import json, pathlib, shutil, sys
root, staging = map(pathlib.Path, sys.argv[1:])
root = root.resolve()
staging.mkdir(parents=True)
required = [root / "result-claim.json", root / "supervisor.json", root / "eval/launch_manifest.json"]
launch = json.load(open(required[-1]))
selected = [*required]
for record in launch.get("materialized_results", []):
    selected.append(pathlib.Path(record["path"]))
selected.extend(sorted(root.glob("server-*.log")))
selected.extend(sorted((root / "eval/logs").glob("*.log")))
for source in selected:
    source = source.resolve()
    if not source.is_file() or not source.is_relative_to(root):
        raise SystemExit(f"unsafe or missing evidence file: {source}")
    relative = source.relative_to(root)
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
PY
ARCHIVE="$ROOT/evidence-$ROBOMME_EVAL_ID.tgz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "$ARCHIVE" -C "$(dirname "$EVIDENCE_DIR")" "$(basename "$EVIDENCE_DIR")"
ARCHIVE_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
ARCHIVE_S3="${EVAL_OUTPUT_S3%/}/$ARCHIVE_SHA.tgz"
# Parse the S3 URI explicitly; use If-None-Match and compare an existing object on collision.
LOCATION="${ARCHIVE_S3#s3://}"; BUCKET="${LOCATION%%/*}"; KEY="${LOCATION#*/}"
if ! aws s3api put-object --bucket "$BUCKET" --key "$KEY" --body "$ARCHIVE" \
    --if-none-match '*' >/dev/null; then
  aws s3 cp "$ARCHIVE_S3" "$ARCHIVE.existing" --only-show-errors
  cmp -s "$ARCHIVE" "$ARCHIVE.existing" || { echo "FATAL evidence collision" >&2; exit 79; }
fi
python3 - "$SUMMARY" "$ARCHIVE_SHA" "$ARCHIVE_S3" <<'PY'
import json, sys
path, sha, uri = sys.argv[1:]
value = json.load(open(path))
value["evidence_archive_sha256"] = sha
value["evidence_archive_uri"] = uri
open(path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
LOCATION="${EVAL_CLAIM_S3#s3://}"; BUCKET="${LOCATION%%/*}"; KEY="${LOCATION#*/}"
if ! aws s3api put-object --bucket "$BUCKET" --key "$KEY" --body "$SUMMARY" \
    --if-none-match '*' >/dev/null; then
  aws s3 cp "$EVAL_CLAIM_S3" "$SUMMARY.existing" --only-show-errors
  cmp -s "$SUMMARY" "$SUMMARY.existing" || { echo "FATAL eval-claim collision" >&2; exit 79; }
fi
echo "ROBOMME EC2 FIXED50 COMPLETE task=$ROBOMME_TASK arm=$ROBOMME_ARM eval_id=$ROBOMME_EVAL_ID"
