#!/usr/bin/env python3
"""Approval-gated p5 job that certifies native RoboMME rendering before scored evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "launch"))
from launch_guardrails import (  # noqa: E402
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    QUEUE,
    ROLE_ARN,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
)

from robomme_integration.eval import (  # noqa: E402
    p5_parallel_action_preflight as action_canary,
)
from robomme_integration.eval import (  # noqa: E402
    parallel_campaign,
)
from robomme_integration.eval.launch_p5_campaign import (  # noqa: E402
    RUNTIME_S3,
    RUNTIME_SHA,
    UPSTREAM_COMMIT,
    UPSTREAM_CRITICAL_SHA256,
    UPSTREAM_REPO,
    VISION_BYTES,
    VISION_REVISION,
    VISION_S3,
    VISION_SHA256,
)
from robomme_integration.launch import (  # noqa: E402
    IMAGE,
    IMAGE_SHA,
    OPENPI,
    OPENPI_SHA,
    PTRM_OPENPI,
    PTRM_OPENPI_SHA,
    STUDY,
    STUDY_ROOT,
)

ENTRY = "gpu_eval_preflight_entry.sh"
STAGED_MANIFEST = "_robomme_p5_eval_preflight_manifest.json"
# The sanitized bundle ships the entry at 0755; the SageMaker training toolkit chmods the selected
# program to 0777 on the node, and the entry normalizes exactly that path back before re-hashing.
# Pin the submitted mode here so that normalization target is the mode the launcher hashed.
SUBMITTED_ENTRY_MODE = 0o755
SAGEMAKER_RUNTIME_ENTRY_MODE = 0o777
PRIORITY = 100
#: 100 = sweep class (default); 400 = standard class, allowed since 2026-09-05 on the lead's instruction.
ALLOWED_PRIORITIES = (100, 400)
MAX_RUN_SECONDS = 2 * 3600
VOLUME_GB = 100
ACTION_MAX_RUN_SECONDS = 4 * 3600
ACTION_VOLUME_GB = 200
ACTION_TEMPLATE = REPO_ROOT / "robomme_integration/eval/p5_q1_parallel_action_canary_v1.json"
# Account limit proven by the archived SageMaker ResourceLimitExceeded contract exercised in
# ``test_p5_quota_retry`` ("current quota ... is 10 Instances").  The live gate additionally
# counts current p5 training jobs and refuses any Batch waiter; a changed/unknown limit therefore
# fails conservatively rather than manufacturing capacity.
P5_CONCURRENCY_LIMIT = 10
ACTIVE_BATCH_STATUSES = (
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "SCHEDULED",
    "STARTING",
    "RUNNING",
)


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_action_template(path: Path) -> tuple[dict, str]:
    config_path = REPO_ROOT / action_canary.CANARY_CONFIG_PATH
    if not config_path.is_file() or config_path.read_bytes() != action_canary.CONFIG:
        raise SystemExit("physical p5 action-canary YAML differs from its embedded sealed bytes")
    payload = path.read_bytes()
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != action_canary.CANARY_TEMPLATE_KIND
        or value.get("template_id") != "p5-q1-8xh100-32action-v1"
        or value.get("benchmark_config_sha256") != action_canary.CONFIG_SHA256
        or not isinstance(value.get("cell"), dict)
        or "result_claim_s3" in value["cell"]
    ):
        raise SystemExit("p5 parallel action-canary template contract drift")
    cell = value["cell"]
    if (
        cell.get("task") != action_canary.CANARY_TASK
        or cell.get("arm") != action_canary.CANARY_ARM
        or cell.get("benchmark_config") != action_canary.CANARY_CONFIG_PATH
        or cell.get("benchmark_config_sha256") != action_canary.CONFIG_SHA256
        or cell.get("training_openpi") != {"uri": OPENPI, "sha256": OPENPI_SHA}
        or not isinstance(cell.get("workspace"), dict)
    ):
        raise SystemExit("p5 parallel action-canary template is not standard-ed923 Q1")
    return value, hashlib.sha256(payload).hexdigest()


def _s3_parts(uri: str) -> tuple[str, str]:
    location = uri.removeprefix("s3://")
    bucket, separator, key = location.partition("/")
    if not separator or not bucket or not key or uri != f"s3://{bucket}/{key}":
        raise SystemExit(f"invalid exact S3 URI: {uri}")
    return bucket, key


def _aws_json(*arguments: str, check: bool = True) -> dict:
    command = [
        "aws",
        *arguments,
        "--region",
        "us-west-2",
        "--output",
        "json",
        "--no-cli-pager",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        if check:
            raise SystemExit(f"AWS live gate failed ({' '.join(command[:3])}): {result.stderr[:500]}")
        return {"_returncode": result.returncode, "_stderr": result.stderr}
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise SystemExit(f"AWS live gate returned non-object JSON: {' '.join(command[:3])}")
    return value


def validate_action_submission_snapshot(snapshot: dict, *, job_name: str) -> dict:
    if snapshot.get("account") != EXECUTION_ACCOUNT:
        raise SystemExit("p5 action-canary live gate is in the wrong AWS account")
    if snapshot.get("claim_objects") or snapshot.get("evidence_objects"):
        raise SystemExit("p5 action-canary immutable namespace is not empty")
    batch = snapshot.get("batch_jobs")
    training = snapshot.get("training_jobs")
    if not isinstance(batch, list) or not isinstance(training, list):
        raise SystemExit("p5 action-canary live capacity snapshot is incomplete")
    duplicate_batch = [job for job in batch if job.get("jobName") == job_name]
    duplicate_training = [job for job in training if job.get("TrainingJobName") == job_name]
    if snapshot.get("training_job_name_exists") or duplicate_batch or duplicate_training:
        raise SystemExit("p5 action-canary has an active or historical duplicate job identity")
    waiting = [
        job for job in batch if job.get("status") in {"SUBMITTED", "PENDING", "RUNNABLE", "SCHEDULED", "STARTING"}
    ]
    if waiting:
        raise SystemExit("p5 queue already has committed waiting work; refusing backlog submission")
    p5_in_progress = sum(
        int(job.get("InstanceCount", 0)) for job in training if job.get("InstanceType") == "ml.p5.48xlarge"
    )
    if p5_in_progress >= P5_CONCURRENCY_LIMIT:
        raise SystemExit(f"p5 has no genuinely available node ({p5_in_progress}/{P5_CONCURRENCY_LIMIT})")
    return {
        "p5_in_progress": p5_in_progress,
        "p5_available": P5_CONCURRENCY_LIMIT - p5_in_progress,
        "waiting_jobs": 0,
        "namespace_empty": True,
        "duplicate_free": True,
    }


def collect_action_submission_snapshot(plan: dict, *, job_name: str) -> dict:
    manifest = plan["manifest"]
    claim_bucket, claim_key = _s3_parts(manifest["claim_s3"])
    evidence_bucket, evidence_key = _s3_parts(manifest["evidence_root_s3"])
    claim_objects = _aws_json(
        "s3api",
        "list-objects-v2",
        "--bucket",
        claim_bucket,
        "--prefix",
        claim_key,
        "--max-keys",
        "1",
    ).get("Contents", [])
    evidence_objects = _aws_json(
        "s3api",
        "list-objects-v2",
        "--bucket",
        evidence_bucket,
        "--prefix",
        evidence_key.rstrip("/") + "/",
        "--max-keys",
        "1",
    ).get("Contents", [])
    batch_jobs = []
    for status in ACTIVE_BATCH_STATUSES:
        response = _aws_json(
            "batch",
            "list-service-jobs",
            "--job-queue",
            QUEUE,
            "--job-status",
            status,
        )
        for job in response.get("jobSummaryList", []):
            batch_jobs.append({**job, "status": status})
    training_jobs = []
    summaries = []
    next_token = None
    while True:
        arguments = [
            "sagemaker",
            "list-training-jobs",
            "--status-equals",
            "InProgress",
            "--max-results",
            "100",
        ]
        if next_token is not None:
            arguments.extend(["--next-token", next_token])
        page = _aws_json(*arguments)
        summaries.extend(page.get("TrainingJobSummaries", []))
        next_token = page.get("NextToken")
        if not next_token:
            break
    for summary in summaries:
        name = summary.get("TrainingJobName")
        if not name:
            continue
        detail = _aws_json("sagemaker", "describe-training-job", "--training-job-name", name)
        resource = detail.get("ResourceConfig", {})
        training_jobs.append(
            {
                "TrainingJobName": name,
                "InstanceType": resource.get("InstanceType"),
                "InstanceCount": resource.get("InstanceCount", 0),
            }
        )
    exact = _aws_json(
        "sagemaker",
        "describe-training-job",
        "--training-job-name",
        job_name,
        check=False,
    )
    if exact.get("_returncode"):
        detail = str(exact.get("_stderr", "")).casefold()
        if not any(marker in detail for marker in ("could not find", "not found", "does not exist")):
            raise SystemExit(f"failed to audit exact p5 action-canary job name: {detail[:500]}")
        training_job_name_exists = False
    else:
        training_job_name_exists = True
    snapshot = {
        "account": str(_aws_json("sts", "get-caller-identity").get("Account")),
        "claim_objects": claim_objects,
        "evidence_objects": evidence_objects,
        "batch_jobs": batch_jobs,
        "training_jobs": training_jobs,
        "training_job_name_exists": training_job_name_exists,
    }
    snapshot["admission"] = validate_action_submission_snapshot(snapshot, job_name=job_name)
    return snapshot


def build_plan(args: argparse.Namespace, source_dir: Path) -> dict:
    if not args.dry_run and not args.confirm_submit:
        raise SystemExit("submission blocked: obtain explicit user approval, then pass --confirm-submit")
    if args.queue != QUEUE or args.role != ROLE_ARN:
        raise SystemExit("RoboMME p5 preflight must use the ordinary cam-robotics p5 queue/role")
    if args.priority not in ALLOWED_PRIORITIES:
        raise SystemExit(f"RoboMME p5 evaluation preflight must use priority in {sorted(ALLOWED_PRIORITIES)}")
    action_mode = bool(args.parallel_action_canary)
    max_run_seconds = (
        ACTION_MAX_RUN_SECONDS if action_mode and args.max_run_seconds == MAX_RUN_SECONDS else args.max_run_seconds
    )
    volume_size_gb = ACTION_VOLUME_GB if action_mode and args.volume_size_gb == VOLUME_GB else args.volume_size_gb
    maximum = ACTION_MAX_RUN_SECONDS if action_mode else MAX_RUN_SECONDS
    expected_volume = ACTION_VOLUME_GB if action_mode else VOLUME_GB
    if not 1 <= max_run_seconds <= maximum:
        label = "four hours" if action_mode else "two hours"
        raise SystemExit(f"RoboMME p5 preflight is capped at {label}")
    if volume_size_gb != expected_volume:
        raise SystemExit(f"RoboMME p5 preflight uses exactly {expected_volume} GiB in this mode")
    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}, None) as (staged, _, _):
        submitted_mode = stat.S_IMODE((staged / ENTRY).lstat().st_mode)
        if submitted_mode != SUBMITTED_ENTRY_MODE:
            raise SystemExit(
                f"submitted preflight entry mode drifted: {oct(submitted_mode)} != {oct(SUBMITTED_ENTRY_MODE)}"
            )
        source_sha = source_tree_sha256(staged)
    if action_mode and args.openpi_profile != "standard":
        raise SystemExit("the Q1 p5 parallel action canary is pinned to standard-ed923 OpenPI")
    openpi = (
        {"uri": OPENPI, "sha256": OPENPI_SHA}
        if args.openpi_profile == "standard"
        else {"uri": PTRM_OPENPI, "sha256": PTRM_OPENPI_SHA}
    )
    scientific = {
        "schema_version": 1,
        "kind": "robomme_p5_native_eval_preflight",
        "runtime": {"uri": RUNTIME_S3, "sha256": RUNTIME_SHA},
        "openpi": openpi,
        "vla_eval_entrypoint": {
            "kind": "python_module_wrapper",
            "module": "vla_eval.cli.main",
        },
        "image": {"uri": IMAGE, "sha256": IMAGE_SHA},
        "source_tree_sha256": source_sha,
    }
    if action_mode:
        template, template_sha = _load_action_template(args.action_template.resolve())
        scientific.update(
            preflight_mode=action_canary.CANARY_MODE,
            canary_template_sha256=template_sha,
            topology=parallel_campaign.p5_8xh100_topology().as_queue_topology(),
            cell=template["cell"],
            vision={
                "uri": VISION_S3,
                "revision": VISION_REVISION,
                "sha256": VISION_SHA256,
                "bytes": VISION_BYTES,
            },
            upstream={
                "repo": UPSTREAM_REPO,
                "commit": UPSTREAM_COMMIT,
                "critical_sha256": UPSTREAM_CRITICAL_SHA256,
            },
            probe={
                "actions_expected": action_canary.EXPECTED_ACTIONS,
                "arm": action_canary.CANARY_ARM,
                "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
                "benchmark_config_sha256": action_canary.CONFIG_SHA256,
                "dataset": "test",
                "episodes_per_lane": action_canary.EPISODES_PER_LANE,
                "max_steps": 1,
                "native_egl": True,
                "score_publication": False,
                "task": action_canary.CANARY_TASK,
            },
        )
    else:
        scientific["probe"] = {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
        }
    scientific_sha = hashlib.sha256(_canonical(scientific).encode()).hexdigest()
    preflight_id = f"p5-native-eval-v1-{scientific_sha[:20]}"
    claim_s3 = f"{STUDY_ROOT}/manifests/claims/preflight/{preflight_id}.json"
    manifest = {
        **scientific,
        "preflight_id": preflight_id,
        "claim_s3": claim_s3,
        "infrastructure": {
            "provider": "aws_sagemaker",
            "execution_account": EXECUTION_ACCOUNT,
            "queue": QUEUE,
            "role": ROLE_ARN,
            "instance_type": "ml.p5.48xlarge",
            "accelerator": "8xH100",
            "priority": args.priority,
            "max_run_seconds": max_run_seconds,
            "volume_size_gb": volume_size_gb,
        },
    }
    if action_mode:
        manifest["evidence_root_s3"] = f"{STUDY_ROOT}/artifacts/robomme/eval_preflight/{preflight_id}/evidence"
    manifest_sha = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    manifest["manifest_sha256"] = manifest_sha
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "ROBOMME_EVAL_RUNTIME_S3": RUNTIME_S3,
        "ROBOMME_EVAL_RUNTIME_SHA256": RUNTIME_SHA,
        "OPENPI_FORK_S3": openpi["uri"],
        "OPENPI_SHA256": openpi["sha256"],
        "OPENPI_PROFILE": args.openpi_profile,
        "PREFLIGHT_ID": preflight_id,
        "PREFLIGHT_CLAIM_S3": claim_s3,
        "PREFLIGHT_MANIFEST_SOURCE": STAGED_MANIFEST,
        "PREFLIGHT_MANIFEST_SHA256": manifest_sha,
        "ROBOMME_PREFLIGHT_MODE": (action_canary.CANARY_MODE if action_mode else "native_render_reset_v1"),
        "ROBOMME_PREFLIGHT_SOURCE_TREE_SHA256": source_sha,
    }
    if action_mode:
        environment.update(
            ROBOMME_EVAL_VISION_S3=VISION_S3,
            ROBOMME_EVAL_VISION_SHA256=VISION_SHA256,
            ROBOMME_EVAL_VISION_BYTES=str(VISION_BYTES),
            ROBOMME_EVAL_UPSTREAM_REPO=UPSTREAM_REPO,
            ROBOMME_EVAL_UPSTREAM_COMMIT=UPSTREAM_COMMIT,
        )
    return {
        "preflight_id": preflight_id,
        "claim_s3": claim_s3,
        "manifest": manifest,
        "manifest_json": _canonical(manifest) + "\n",
        "environment": environment,
        "source_sha": source_sha,
        "max_run_seconds": max_run_seconds,
        "volume_size_gb": volume_size_gb,
        "action_mode": action_mode,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-dir", type=Path, default=REPO_ROOT / "robomme_integration")
    value.add_argument("--queue", default=QUEUE)
    value.add_argument("--role", default=ROLE_ARN)
    value.add_argument("--priority", type=int, default=PRIORITY)
    value.add_argument("--max-run-seconds", type=int, default=MAX_RUN_SECONDS)
    value.add_argument("--volume-size-gb", type=int, default=VOLUME_GB)
    value.add_argument("--openpi-profile", choices=("standard", "advanced"), default="standard")
    value.add_argument("--parallel-action-canary", action="store_true")
    value.add_argument("--action-template", type=Path, default=ACTION_TEMPLATE)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--confirm-submit", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    plan = build_plan(args, source_dir)
    print(json.dumps(plan["manifest"], indent=2, sort_keys=True))
    if args.dry_run:
        print("DRY RUN ONLY — no AWS SDK loaded and no cloud write performed")
        return
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    job_name = (
        # One-shot identity is intentional: any historical attempt requires a reviewed successor
        # canary/source identity, never an in-place retry with ambiguous evidence ownership.
        f"sarvesh-rmme-p5-action-{plan['preflight_id'].rsplit('-', 1)[-1]}"
        if plan["action_mode"]
        else f"sarvesh-rmme-p5-eval-preflight-{stamp}"
    )
    if plan["action_mode"]:
        snapshot = collect_action_submission_snapshot(plan, job_name=job_name)
        print(json.dumps(snapshot["admission"], indent=2, sort_keys=True))
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=IMAGE,
        instance_type="ml.p5.48xlarge",
        volume_size=plan["volume_size_gb"],
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.benchmark", "Value": "RoboMME"},
            {
                "Key": "wsm.kind",
                "Value": "eval-action-preflight" if plan["action_mode"] else "eval-preflight",
            },
            {"Key": "wsm.preflight_id", "Value": plan["preflight_id"]},
        ],
        retry_config={"attempts": 1},
        job_name=job_name,
        queue=QUEUE,
        role=ROLE_ARN,
        priority=args.priority,
        max_run_seconds=plan["max_run_seconds"],
        secrets_manager_arn=None,
        confirmed=args.confirm_submit,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_sha"],
        staged_source_files={STAGED_MANIFEST: plan["manifest_json"]},
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
