#!/usr/bin/env python3
"""Submit an isolated two-step RoboMME policy-training canary.

This launcher deliberately does not share the production training namespace or entry point.  A
successful job may publish exactly one object: ``training_canary.complete.json`` under the canary
manifest namespace.  The receipt is an infrastructure/runtime proof, never scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
LAUNCH_UTILS = REPO_ROOT / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_UTILS))

from launch_guardrails import (  # noqa: E402
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    REGION,
    ROLE_ARN,
    TRAINING_PLAN_QUEUE,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
    training_plan_arn,
)

from robomme_integration import launch as production_launch  # noqa: E402
from robomme_integration.training import policy_canary as canary_contract  # noqa: E402

ENTRY = "gpu_policy_canary_entry.sh"
STAGED_MANIFEST = "_robomme_policy_canary_manifest.json"
KIND = "robomme_policy_training_canary_attempt"
RECEIPT_KIND = "robomme_policy_training_canary_complete"
CLAIM = "not_scientific_training_evidence"
CONTRACT_VERSION = "robomme-policy-training-canary-v1"
PRODUCTION_SOURCE_SHA256 = "528d2c580dfd43a300fce4046fec7bf8bd39a73f700170bcc81dd4316260de3d"
PRODUCTION_ENTRY_SHA256 = "d8a157dcbb9f8092dcf85bc700dbd4b6eae93c4f5c1f809caf1ddbc2bb34b7ca"
CANARY_ENTRY_SHA256 = "292edebfd1fee4e6a23138be9ad1115e5a50de84f98afd1daf82a002eda41dfc"
REFERENCE_RUN_ID = "st-v1-pickxtimes-gdn8_jepa_l01_k1-seed0-64dee36adb843a2f"
REFERENCE_SCIENTIFIC_SPEC_SHA256 = "64dee36adb843a2ffc548543e39c3c908609044d8c513363025e43d884db109c"
REFERENCE_MANIFEST_SHA256 = "294ea48d79ac664c2efc14089308d0f64bc436deee86fccb3bbab810376f0e8d"
REFERENCE_MANIFEST_BYTES_SHA256 = "217af64fc1732370a962786669e36981b24cca95de29867229273c78f9490811"
REFERENCE_MANIFEST_S3 = (
    f"{production_launch.STUDY_ROOT}/manifests/runs/train/{REFERENCE_RUN_ID}/{REFERENCE_RUN_ID}-attempt1.json"
)
STEPS = 2
FINAL_STEP = 1
PRIORITY = 400
MAX_RUN_SECONDS = 3 * 3600
VOLUME_SIZE_GB = 250
SUBMITTED_ENTRY_MODE = 0o755
SAGEMAKER_RUNTIME_ENTRY_MODE = 0o777
RETRY = {"attempts": 1}
ARM = "gdn8_jepa_l01_k1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ACTIVE_SERVICE_JOB_STATUSES = (
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "SCHEDULED",
    "STARTING",
    "RUNNING",
)
SERVICE_JOB_PAGE_SIZE = 100
MAX_SERVICE_JOB_PAGES_PER_STATUS = 100
SAGEMAKER_LIST_PAGE_SIZE = 100
MAX_SAGEMAKER_LIST_PAGES = 100

# Every delta from the frozen 528d production source must be reviewed here. This intentionally
# minimal bundle is independent of the p5 eval-canary source; production policy-training files are
# never allowed in this set.
ALLOWED_SOURCE_DELTAS = frozenset(
    {
        "gpu_policy_canary_entry.sh",
        "policy_canary_launch.py",
        "training/policy_canary.py",
        "tests/test_policy_canary.py",
    }
)

FORBIDDEN_ENVIRONMENT_KEYS = frozenset(
    {
        "OUTPUT_S3",
        "RUN_MANIFEST_SOURCE",
        "RUN_MANIFEST_SHA256",
        "RUN_MANIFEST_S3",
        "PRODUCER_CLAIM_S3",
        "COMPLETION_CLAIM_S3",
        "CHECKPOINT_TREE_MANIFEST_ROOT",
        "ROBOMME_SCIENTIFIC_SPEC_SHA256",
        "ROBOMME_FINAL_STEP",
        "ROBOMME_RUN_ID",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _seal(value: dict[str, Any]) -> tuple[dict[str, Any], str]:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    clean["manifest_sha256"] = _sha256_json(clean)
    return clean, _canonical_json(clean)


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise SystemExit(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _source_dir(value: str | None) -> Path:
    root = Path(value or Path(__file__).resolve().parent).resolve()
    required = (ENTRY, "gpu_train_entry.sh", "launch.py", "training/train.py", "training/policy_canary.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"invalid isolated RoboMME canary source directory {root}: missing {missing}")
    return root


def _source_sha(source_dir: Path) -> str:
    with prepared_source_bundle(
        source_dir,
        ENTRY,
        {"SAGEMAKER_PROGRAM": ENTRY},
    ) as (staged, _entry, _environment):
        return source_tree_sha256(staged)


def _runtime_source_sha(source_dir: Path) -> str:
    """Seal SageMaker Training Toolkit's one deterministic source-tree mutation.

    The toolkit classifies the shell entry as a COMMAND and executes ``chmod(path, 511)`` before
    invoking it. Decimal 511 is octal 0777. No other mode/path/byte mutation is accepted.
    """

    with prepared_source_bundle(
        source_dir,
        ENTRY,
        {"SAGEMAKER_PROGRAM": ENTRY},
    ) as (staged, _entry, _environment):
        entry = staged / ENTRY
        submitted_mode = stat.S_IMODE(entry.lstat().st_mode)
        if submitted_mode != SUBMITTED_ENTRY_MODE:
            raise SystemExit(
                f"submitted canary entry mode drifted: {oct(submitted_mode)} != {oct(SUBMITTED_ENTRY_MODE)}"
            )
        entry.chmod(SAGEMAKER_RUNTIME_ENTRY_MODE)
        return source_tree_sha256(staged)


def _tree_inventory(root: Path) -> dict[str, tuple[Any, ...]]:
    """Inventory one already-sanitized tree for an exact reviewed-delta comparison."""

    result: dict[str, tuple[Any, ...]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            result[relative] = ("symlink", mode, os.readlink(path))
        elif path.is_dir():
            result[relative] = ("directory", mode)
        elif path.is_file():
            result[relative] = (
                "file",
                mode,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
            )
        else:
            raise SystemExit(f"unsupported source entry type: {path}")
    return result


def validate_reviewed_source_delta(
    source_dir: Path,
    baseline_dir: Path,
    *,
    allowed: frozenset[str] = ALLOWED_SOURCE_DELTAS,
) -> dict[str, Any]:
    """Prove that the frozen production tree changed only at reviewed canary/eval paths."""

    baseline_dir = baseline_dir.resolve()
    if not (baseline_dir / "gpu_train_entry.sh").is_file():
        raise SystemExit(f"invalid production baseline directory: {baseline_dir}")
    with prepared_source_bundle(
        baseline_dir,
        "gpu_train_entry.sh",
        {"SAGEMAKER_PROGRAM": "gpu_train_entry.sh"},
    ) as (baseline, _entry, _environment):
        baseline_sha = source_tree_sha256(baseline)
        baseline_inventory = _tree_inventory(baseline)
    if baseline_sha != PRODUCTION_SOURCE_SHA256:
        raise SystemExit(f"production baseline source drifted: {baseline_sha} != {PRODUCTION_SOURCE_SHA256}")
    production_entry_sha = hashlib.sha256((baseline_dir / "gpu_train_entry.sh").read_bytes()).hexdigest()
    if production_entry_sha != PRODUCTION_ENTRY_SHA256:
        raise SystemExit("production baseline gpu_train_entry.sh bytes drifted")

    with prepared_source_bundle(
        source_dir,
        ENTRY,
        {"SAGEMAKER_PROGRAM": ENTRY},
    ) as (current, _entry, _environment):
        current_inventory = _tree_inventory(current)
    changed = {
        path
        for path in set(baseline_inventory) | set(current_inventory)
        if baseline_inventory.get(path) != current_inventory.get(path)
    }
    unreviewed = changed - allowed
    stale_allowlist = allowed - changed
    if unreviewed:
        raise SystemExit(f"unreviewed source delta from production 528d: {sorted(unreviewed)}")
    if stale_allowlist:
        raise SystemExit(f"reviewed source-delta allowlist contains unchanged paths: {sorted(stale_allowlist)}")
    return {
        "baseline_source_tree_sha256": baseline_sha,
        "production_entry_sha256": production_entry_sha,
        "reviewed_delta_paths": sorted(changed),
    }


def load_reference_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read production reference manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("production reference manifest must be a JSON object")
    claimed = _require_sha(value.get("manifest_sha256"), "reference manifest_sha256")
    clean = dict(value)
    clean.pop("manifest_sha256")
    actual = _sha256_json(clean)
    if actual != claimed:
        raise SystemExit(f"production reference manifest seal mismatch: {actual} != {claimed}")
    return value


def validate_reference_manifest(value: dict[str, Any], *, task: str, arm: str) -> dict[str, Any]:
    pinned_outer = {
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "manifest_s3": REFERENCE_MANIFEST_S3,
        "run_id": REFERENCE_RUN_ID,
        "scientific_spec_sha256": REFERENCE_SCIENTIFIC_SPEC_SHA256,
    }
    outer_drift = {
        name: (value.get(name), expected) for name, expected in pinned_outer.items() if value.get(name) != expected
    }
    if outer_drift:
        raise SystemExit(f"production reference is not the exact frozen Wave2 cell: {outer_drift}")
    if value.get("kind") != "robomme_gpu_training_attempt":
        raise SystemExit("canary reference must be a production GPU training attempt")
    scientific = value.get("scientific")
    if not isinstance(scientific, dict):
        raise SystemExit("production reference lacks its scientific specification")
    if value.get("scientific_spec_sha256") != _sha256_json(scientific):
        raise SystemExit("production reference scientific specification seal mismatch")
    if scientific.get("arm") != arm or scientific.get("task", {}).get("name") != task:
        raise SystemExit("production reference arm/task differs from the requested canary")
    if scientific.get("scope") != "single_task_v1":
        raise SystemExit("policy canary currently requires a single-task production reference")
    source = scientific.get("sources", {}).get("robomme_integration", {})
    if source.get("sanitized_source_tree_sha256") != PRODUCTION_SOURCE_SHA256:
        raise SystemExit("production reference is not bound to frozen source 528d")
    if source.get("entry") != "gpu_train_entry.sh" or source.get("entry_sha256") != PRODUCTION_ENTRY_SHA256:
        raise SystemExit("production reference entry identity drifted")
    training = scientific.get("training", {})
    required_training = {
        "steps": 20_000,
        "seed": 0,
        "batch_size": 64,
        "batch_unit": "steps",
        "effective_per_step_batch": 64,
        "action_horizon": 20,
        "full_finetune": True,
        "jax_devices": 8,
        "jax_processes": 1,
        "fsdp_devices": 1,
    }
    drift = {
        name: (training.get(name), expected)
        for name, expected in required_training.items()
        if training.get(name) != expected
    }
    if drift:
        raise SystemExit(f"production reference training/model topology drifted: {drift}")
    if scientific.get("mechanism") != production_launch._arm_spec(arm):
        raise SystemExit("production reference mechanism differs from the current reviewed arm")
    if scientific.get("sources", {}).get("openpi", {}).get("sha256") != production_launch.OPENPI_SHA:
        raise SystemExit("production reference OpenPI identity drifted")
    if scientific.get("sources", {}).get("tokenizer", {}).get("sha256") != production_launch.TOKENIZER_SHA:
        raise SystemExit("production reference tokenizer identity drifted")
    if scientific.get("sources", {}).get("image", {}).get("sha256") != production_launch.IMAGE_SHA:
        raise SystemExit("production reference image identity drifted")
    workspace = scientific.get("workspace_representation")
    if not isinstance(workspace, dict) or not workspace.get("task_bound"):
        raise SystemExit("GDN+JEPA canary requires the exact task-bound workspace representation")
    if workspace.get("omega_symbol") != "omega_t" or workspace.get("supervision") is not None:
        raise SystemExit("production reference workspace interface drifted")
    model_identity = {
        name: training[name]
        for name in (
            "seed",
            "batch_size",
            "batch_unit",
            "effective_per_step_batch",
            "action_horizon",
            "full_finetune",
            "optimizer",
            "peak_lr",
            "ema_decay",
            "fsdp_devices",
            "jax_devices",
            "jax_processes",
        )
    }
    exact_components = {
        "task": scientific.get("task"),
        "mechanism": scientific.get("mechanism"),
        "data": scientific.get("data"),
        "initialization": scientific.get("initialization"),
        "workspace_representation": workspace,
        "sources": scientific.get("sources"),
        "model_training_identity": model_identity,
    }
    component_drift = {
        name: _sha256_json(value)
        for name, value in exact_components.items()
        if _sha256_json(value) != canary_contract.IDENTITY_COMPONENT_SHA256[name]
    }
    if component_drift:
        raise SystemExit(
            f"production reference task/model/data/init/workspace/source identity drifted: {component_drift}"
        )
    return scientific


def authenticate_reference_manifest_bytes(path: Path) -> dict[str, str]:
    """Match the local exact reference bytes to the canonical create-once S3 manifest object."""

    try:
        local = path.resolve().read_bytes()
    except OSError as error:
        raise SystemExit(f"cannot reread local reference manifest bytes: {error}") from error
    local_sha = hashlib.sha256(local).hexdigest()
    if local_sha != REFERENCE_MANIFEST_BYTES_SHA256:
        raise SystemExit(
            f"local reference manifest byte identity drifted: {local_sha} != {REFERENCE_MANIFEST_BYTES_SHA256}"
        )
    completed = subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            REFERENCE_MANIFEST_S3,
            "-",
            "--region",
            REGION,
            "--only-show-errors",
        ],
        check=True,
        capture_output=True,
    )
    remote = completed.stdout
    remote_sha = hashlib.sha256(remote).hexdigest()
    if remote != local or remote_sha != REFERENCE_MANIFEST_BYTES_SHA256:
        raise SystemExit(
            "canonical S3 reference manifest bytes differ from the exact submitted Wave2 cell: "
            f"local={local_sha} remote={remote_sha}"
        )
    try:
        remote_value = json.loads(remote)
    except json.JSONDecodeError as error:
        raise SystemExit("canonical S3 reference manifest is malformed JSON") from error
    validate_reference_manifest(remote_value, task="PickXtimes", arm=ARM)
    return {
        "manifest_s3": REFERENCE_MANIFEST_S3,
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "bytes_sha256": remote_sha,
    }


def _environment_has_production_leak(environment: dict[str, str]) -> list[str]:
    leaked = sorted(FORBIDDEN_ENVIRONMENT_KEYS & environment.keys())
    for name, value in environment.items():
        if "/checkpoints/robomme/pi05/" in value or "/manifests/claims/train/" in value:
            leaked.append(name)
    return sorted(set(leaked))


def validate_cloud_snapshot(
    *,
    account: str,
    training_plan: dict[str, Any],
    namespace_objects: list[dict[str, Any]],
    active_batch_jobs: list[dict[str, Any]],
    active_training_jobs: list[dict[str, Any]],
    job_name: str,
) -> dict[str, int]:
    """Validate the final, read-only cloud snapshot immediately preceding submission."""

    if account != EXECUTION_ACCOUNT:
        raise SystemExit(f"submission requires AWS account {EXECUTION_ACCOUNT}; caller is {account}")
    if training_plan.get("TrainingPlanArn") != training_plan_arn(TRAINING_PLAN_QUEUE):
        raise SystemExit("p5e training-plan ARN drifted")
    expected_plan = {"Status": "Active", "TotalInstanceCount": 2, "AvailableInstanceCount": 1, "InUseInstanceCount": 1}
    drift = {
        name: (training_plan.get(name), expected)
        for name, expected in expected_plan.items()
        if training_plan.get(name) != expected
    }
    if drift:
        raise SystemExit(f"p5e plan is not exactly Total2/InUse1/Available1 Active: {drift}")
    if int(training_plan.get("UnhealthyInstanceCount", 0)) != 0:
        raise SystemExit("p5e training plan reports an unhealthy instance")
    if namespace_objects:
        raise SystemExit("canary S3 namespace is not empty")
    duplicate_batch = sorted(job.get("jobName") for job in active_batch_jobs if job.get("jobName") == job_name)
    duplicate_training = sorted(
        job.get("TrainingJobName")
        for job in active_training_jobs
        if job.get("CanaryIdMatch") or job.get("TrainingJobName") == job_name
    )
    if duplicate_batch or duplicate_training:
        raise SystemExit(f"active duplicate canary found: batch={duplicate_batch} training={duplicate_training}")
    return {"total": 2, "in_use": 1, "available": 1}


def _aws_json(*arguments: str) -> dict[str, Any]:
    command = ["aws", *arguments, "--region", REGION, "--output", "json", "--no-cli-pager"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise SystemExit(f"AWS command returned non-object JSON: {' '.join(command[:3])}")
    return value


def _list_active_service_jobs() -> list[dict[str, Any]]:
    """List every active SageMaker Training service job, with an explicit pagination bound."""

    jobs: list[dict[str, Any]] = []
    for status in ACTIVE_SERVICE_JOB_STATUSES:
        next_token: str | None = None
        for _page in range(MAX_SERVICE_JOB_PAGES_PER_STATUS):
            arguments = [
                "batch",
                "list-service-jobs",
                "--job-queue",
                TRAINING_PLAN_QUEUE,
                "--job-status",
                status,
                "--max-results",
                str(SERVICE_JOB_PAGE_SIZE),
            ]
            if next_token is not None:
                arguments.extend(["--next-token", next_token])
            response = _aws_json(*arguments)
            page = response.get("jobSummaryList")
            if not isinstance(page, list):
                raise SystemExit(f"Batch service-job listing for {status} returned no jobSummaryList")
            jobs.extend(page)
            token = response.get("nextToken")
            if token is None:
                break
            if not isinstance(token, str) or not token:
                raise SystemExit(f"Batch service-job listing for {status} returned an invalid nextToken")
            next_token = token
        else:
            raise SystemExit(
                "Batch service-job listing exceeded the audited pagination bound for "
                f"{status} ({MAX_SERVICE_JOB_PAGES_PER_STATUS} pages)"
            )
    return jobs


def _list_active_training_plans() -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    next_token: str | None = None
    for _page in range(MAX_SAGEMAKER_LIST_PAGES):
        arguments = [
            "sagemaker",
            "list-training-plans",
            "--filters",
            "Name=Status,Value=Active",
            "--max-results",
            str(SAGEMAKER_LIST_PAGE_SIZE),
        ]
        if next_token is not None:
            arguments.extend(["--next-token", next_token])
        response = _aws_json(*arguments)
        page = response.get("TrainingPlanSummaries")
        if not isinstance(page, list):
            raise SystemExit("SageMaker active training-plan listing returned no summaries")
        plans.extend(page)
        token = response.get("NextToken")
        if token is None:
            break
        if not isinstance(token, str) or not token:
            raise SystemExit("SageMaker active training-plan listing returned an invalid NextToken")
        next_token = token
    else:
        raise SystemExit(
            "SageMaker active training-plan listing exceeded the audited pagination bound "
            f"({MAX_SAGEMAKER_LIST_PAGES} pages)"
        )
    return plans


def _list_in_progress_training_jobs(*, canary_id: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    next_token: str | None = None
    for _page in range(MAX_SAGEMAKER_LIST_PAGES):
        arguments = [
            "sagemaker",
            "list-training-jobs",
            "--status-equals",
            "InProgress",
            "--max-results",
            str(SAGEMAKER_LIST_PAGE_SIZE),
        ]
        if next_token is not None:
            arguments.extend(["--next-token", next_token])
        response = _aws_json(*arguments)
        page = response.get("TrainingJobSummaries")
        if not isinstance(page, list):
            raise SystemExit("SageMaker in-progress training-job listing returned no summaries")
        summaries.extend(page)
        token = response.get("NextToken")
        if token is None:
            break
        if not isinstance(token, str) or not token:
            raise SystemExit("SageMaker in-progress training-job listing returned an invalid NextToken")
        next_token = token
    else:
        raise SystemExit(
            "SageMaker in-progress training-job listing exceeded the audited pagination bound "
            f"({MAX_SAGEMAKER_LIST_PAGES} pages)"
        )

    jobs: list[dict[str, Any]] = []
    for summary in summaries:
        name = summary.get("TrainingJobName")
        if not name:
            raise SystemExit(f"SageMaker in-progress summary omitted TrainingJobName: {summary}")
        detail = _aws_json("sagemaker", "describe-training-job", "--training-job-name", name)
        environment = detail.get("Environment")
        if not isinstance(environment, dict):
            raise SystemExit(f"SageMaker training job {name} omitted its Environment")
        jobs.append(
            {
                "TrainingJobName": name,
                "CanaryIdMatch": environment.get("ROBOMME_CANARY_ID") == canary_id,
            }
        )
    return jobs


def collect_cloud_snapshot(*, receipt_s3: str, namespace_s3: str, job_name: str, canary_id: str) -> dict[str, Any]:
    if not receipt_s3.startswith(namespace_s3.rstrip("/") + "/"):
        raise SystemExit("receipt escaped its canary namespace")
    location = namespace_s3.removeprefix("s3://")
    bucket, separator, prefix = location.partition("/")
    if not separator or not bucket or not prefix:
        raise SystemExit(f"invalid canary namespace URI: {namespace_s3}")
    account = _aws_json("sts", "get-caller-identity").get("Account")
    plans = _list_active_training_plans()
    matching_plans = [value for value in plans if value.get("TrainingPlanName") == "cam-robotics-tp"]
    if len(matching_plans) != 1:
        raise SystemExit(
            f"authoritative list-training-plans did not return exactly one active cam-robotics-tp: {matching_plans}"
        )
    plan = matching_plans[0]
    objects = _aws_json(
        "s3api",
        "list-objects-v2",
        "--bucket",
        bucket,
        "--prefix",
        prefix.rstrip("/") + "/",
        "--max-keys",
        "1",
    ).get("Contents", [])
    batch_jobs = _list_active_service_jobs()
    training_jobs = _list_in_progress_training_jobs(canary_id=canary_id)
    validate_cloud_snapshot(
        account=str(account),
        training_plan=plan,
        namespace_objects=list(objects),
        active_batch_jobs=batch_jobs,
        active_training_jobs=training_jobs,
        job_name=job_name,
    )
    return {"account": account, "training_plan": plan, "namespace_objects": objects}


def build_plan(args: argparse.Namespace, source_dir: Path) -> dict[str, Any]:
    if not args.dry_run and not args.confirm_submit:
        raise SystemExit("submission blocked: explicit approval exists, but --confirm-submit is still required")
    if args.queue != TRAINING_PLAN_QUEUE or args.role != ROLE_ARN or args.priority != PRIORITY:
        raise SystemExit("policy canary is pinned to p5e training-plan queue, execution role, and priority 400")
    if args.max_run_seconds != MAX_RUN_SECONDS or args.volume_size_gb != VOLUME_SIZE_GB:
        raise SystemExit(f"policy canary is pinned to {MAX_RUN_SECONDS}s and {VOLUME_SIZE_GB} GiB")
    if args.arm != ARM:
        raise SystemExit(f"this first generic policy-canary contract is audited only for {ARM}")
    if args.task != "PickXtimes":
        raise SystemExit("policy canary v1 is pinned to the exact PickXtimes Wave2 reference")
    baseline = Path(args.production_baseline_dir).resolve()
    source_audit = validate_reviewed_source_delta(source_dir, baseline)
    source_sha = _source_sha(source_dir)
    runtime_source_sha = _runtime_source_sha(source_dir)
    entry_sha = hashlib.sha256((source_dir / ENTRY).read_bytes()).hexdigest()
    if entry_sha != CANARY_ENTRY_SHA256:
        raise SystemExit(f"canary entry bytes drifted: {entry_sha} != {CANARY_ENTRY_SHA256}")
    reference = load_reference_manifest(Path(args.reference_manifest).resolve())
    scientific = validate_reference_manifest(reference, task=args.task, arm=args.arm)

    workspace = scientific["workspace_representation"]
    omega = workspace["omega"]
    identity = {
        "canary_source_tree_sha256": source_sha,
        "canary_runtime_source_tree_sha256": runtime_source_sha,
        "reference_run_id": reference["run_id"],
        "reference_scientific_spec_sha256": reference["scientific_spec_sha256"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "production_source_tree_sha256": PRODUCTION_SOURCE_SHA256,
        "task": scientific["task"],
        "arm": args.arm,
        "mechanism": scientific["mechanism"],
        "data": scientific["data"],
        "initialization": scientific["initialization"],
        "workspace_representation": workspace,
        "sources": scientific["sources"],
        "model_training_identity": {
            name: scientific["training"][name]
            for name in (
                "seed",
                "batch_size",
                "batch_unit",
                "effective_per_step_batch",
                "action_horizon",
                "full_finetune",
                "optimizer",
                "peak_lr",
                "ema_decay",
                "fsdp_devices",
                "jax_devices",
                "jax_processes",
            )
        },
    }
    identity_sha = _sha256_json(identity)
    canary_id = f"policy-canary-v1-{args.task.lower()}-{args.arm}-{identity_sha[:16]}"
    namespace = f"{production_launch.STUDY_ROOT}/manifests/canaries/policy_training/{canary_id}"
    receipt_s3 = f"{namespace}/training_canary.complete.json"
    job_name = f"sarvesh-rmme-policy-canary-{identity_sha[:20]}"
    if not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", job_name):
        raise SystemExit(f"invalid deterministic canary job name: {job_name}")

    infrastructure = {
        "provider": "aws_sagemaker",
        "execution_account": EXECUTION_ACCOUNT,
        "queue": args.queue,
        "training_plan_arn": training_plan_arn(args.queue),
        "role": args.role,
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "priority": args.priority,
        "max_run_seconds": args.max_run_seconds,
        "volume_size_gb": args.volume_size_gb,
        "attempts_in_job": 1,
        "deterministic_job_name": job_name,
    }
    manifest, manifest_json = _seal(
        {
            "schema_version": 1,
            "kind": KIND,
            "claim": CLAIM,
            "contract_version": CONTRACT_VERSION,
            "canary_id": canary_id,
            "identity_sha256": identity_sha,
            "identity": identity,
            "canary_execution": {
                "optimizer_steps": STEPS,
                "final_local_checkpoint_step": FINAL_STEP,
                "seed": 0,
                "batch_size": 64,
                "effective_per_step_batch": 64,
                "warmup_steps": 1,
                "decay_steps": 2,
                "peak_lr": scientific["training"]["peak_lr"],
                "decay_lr": scientific["training"]["decay_lr"],
                "diagnostics": [
                    "total_loss",
                    "action_loss",
                    "jepa_loss",
                    "sigreg_loss",
                    "gradient_norm",
                    "parameter_update_norm",
                    "parameters_finite",
                ],
                "checkpoint_scope": "node_local_ephemeral_only",
                "restore_smoke": "mandatory_local_save_and_restore",
            },
            "source": {
                "canary_source_tree_sha256": source_sha,
                "canary_runtime_source_tree_sha256": runtime_source_sha,
                "entry": ENTRY,
                "entry_sha256": entry_sha,
                "submitted_entry_mode": SUBMITTED_ENTRY_MODE,
                "sagemaker_runtime_entry_mode": SAGEMAKER_RUNTIME_ENTRY_MODE,
                **source_audit,
            },
            "infrastructure": infrastructure,
            "publication": {
                "namespace_s3": namespace,
                "receipt_s3": receipt_s3,
                "create_once": True,
                "only_allowed_object": "training_canary.complete.json",
                "production_checkpoint_or_deploy_publication": False,
            },
        }
    )
    if ALLOWED_SOURCE_DELTAS != canary_contract.REVIEWED_SOURCE_DELTAS:
        raise SystemExit("launcher and on-node reviewed source-delta contracts differ")
    canary_contract.validate_manifest_contract(manifest)
    environment = {
        "SM_USE_RESERVED_CAPACITY": "0",
        "ROBOMME_CANARY_KIND": KIND,
        "ROBOMME_CANARY_CLAIM": CLAIM,
        "ROBOMME_CANARY_ID": canary_id,
        "ROBOMME_CANARY_STEPS": str(STEPS),
        "ROBOMME_CANARY_MANIFEST_SOURCE": STAGED_MANIFEST,
        "ROBOMME_CANARY_MANIFEST_SHA256": manifest["manifest_sha256"],
        "ROBOMME_CANARY_NAMESPACE_S3": namespace,
        "ROBOMME_CANARY_RECEIPT_S3": receipt_s3,
        "ROBOMME_REFERENCE_RUN_ID": reference["run_id"],
        "ROBOMME_REFERENCE_SCIENTIFIC_SPEC_SHA256": reference["scientific_spec_sha256"],
        "ROBOMME_REFERENCE_SOURCE_SHA256": PRODUCTION_SOURCE_SHA256,
        "ROBOMME_ARM": args.arm,
        "ROBOMME_SCOPE": "single_task_canary",
        "ROBOMME_TASK": args.task,
        "ROBOMME_DATA_S3": scientific["data"]["dataset_s3"],
        "ROBOMME_DATA_PARENT_INVENTORY_S3": scientific["data"]["parent_inventory_uri"],
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": scientific["data"]["parent_inventory_sha256"],
        "ROBOMME_DATA_DERIVED_INVENTORY_SHA256": scientific["data"]["derived_task_inventory_sha256"],
        "INIT_S3": scientific["initialization"]["checkpoint_s3"],
        "INIT_INVENTORY_S3": scientific["initialization"]["inventory_uri"],
        "INIT_INVENTORY_SHA256": scientific["initialization"]["inventory_sha256"],
        "OPENPI_FORK_S3": scientific["sources"]["openpi"]["uri"],
        "OPENPI_REQUIRED_SENTINEL": "_WSM_GDN_JEPA",
        "PALIGEMMA_TOKENIZER_S3": scientific["sources"]["tokenizer"]["uri"],
        "PALIGEMMA_TOKENIZER_SHA256": scientific["sources"]["tokenizer"]["sha256"],
        "ROBOMME_WORKSPACE_S3": omega["uri"],
        "ROBOMME_WORKSPACE_MANIFEST_SHA256": omega["manifest_sha256"],
        "WSM_MAX_STEPS": "2",
        "WSM_SAVE_INTERVAL": "2",
        "WSM_WARMUP_STEPS": "1",
        "WSM_PEAK_LR": str(scientific["training"]["peak_lr"]),
        "WSM_DECAY_STEPS": "2",
        "WSM_DECAY_LR": str(scientific["training"]["decay_lr"]),
        "WSM_SEED": "0",
    }
    leaked = _environment_has_production_leak(environment)
    if leaked:
        raise SystemExit(f"production training output flags leaked into canary: {leaked}")
    oversized = {name: len(value.encode()) for name, value in environment.items() if len(value.encode()) > 512}
    if oversized:
        raise SystemExit(f"SageMaker canary environment values exceed 512 bytes: {oversized}")
    return {
        "canary_id": canary_id,
        "job_name": job_name,
        "namespace_s3": namespace,
        "receipt_s3": receipt_s3,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "environment": environment,
        "source_sha": source_sha,
        "runtime_source_sha": runtime_source_sha,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--task", required=True, choices=("PickXtimes",))
    value.add_argument("--arm", default=ARM, choices=(ARM,))
    value.add_argument("--reference-manifest", required=True)
    value.add_argument("--production-baseline-dir", required=True)
    value.add_argument("--source-dir")
    value.add_argument("--queue", default=TRAINING_PLAN_QUEUE)
    value.add_argument("--role", default=ROLE_ARN)
    value.add_argument("--priority", type=int, default=PRIORITY)
    value.add_argument("--max-run-seconds", type=int, default=MAX_RUN_SECONDS)
    value.add_argument("--volume-size-gb", type=int, default=VOLUME_SIZE_GB)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--confirm-submit", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    source_dir = _source_dir(args.source_dir)
    plan = build_plan(args, source_dir)
    print(
        f"kind={KIND} claim={CLAIM}\n"
        f"canary_id={plan['canary_id']} job_name={plan['job_name']}\n"
        f"reference={plan['manifest']['identity']['reference_run_id']}\n"
        f"source_sha256={plan['source_sha']}\n"
        f"runtime_source_sha256={plan['runtime_source_sha']}\n"
        f"receipt={plan['receipt_s3']}\n"
        f"queue={args.queue} priority={args.priority} max_run={args.max_run_seconds}s "
        f"volume={args.volume_size_gb}GiB dry={args.dry_run}"
    )
    if args.dry_run:
        print(json.dumps(plan["manifest"], sort_keys=True, indent=2))
        print(
            "DRY RUN ONLY — no AWS SDK loaded and no cloud write performed; "
            "canonical S3 reference-byte authentication is pending the final live gate"
        )
        return

    reference_auth = authenticate_reference_manifest_bytes(Path(args.reference_manifest))
    print(
        "REFERENCE AUTH "
        f"manifest_sha256={reference_auth['manifest_sha256']} "
        f"bytes_sha256={reference_auth['bytes_sha256']} "
        f"uri={reference_auth['manifest_s3']}"
    )
    snapshot = collect_cloud_snapshot(
        receipt_s3=plan["receipt_s3"],
        namespace_s3=plan["namespace_s3"],
        job_name=plan["job_name"],
        canary_id=plan["canary_id"],
    )
    print(
        "FINAL GATE "
        f"account={snapshot['account']} plan_status={snapshot['training_plan']['Status']} "
        f"total={snapshot['training_plan']['TotalInstanceCount']} "
        f"in_use={snapshot['training_plan']['InUseInstanceCount']} "
        f"available={snapshot['training_plan']['AvailableInstanceCount']} namespace_empty=true"
    )
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=production_launch.IMAGE,
        instance_type="ml.p5e.48xlarge",
        volume_size=args.volume_size_gb,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.study", "Value": production_launch.STUDY},
            {"Key": "wsm.kind", "Value": "policy-training-canary"},
            {"Key": "wsm.claim", "Value": CLAIM},
            {"Key": "wsm.task", "Value": args.task},
            {"Key": "wsm.arm", "Value": args.arm},
            {"Key": "wsm.canary_id", "Value": plan["canary_id"]},
        ],
        retry_config=RETRY,
        job_name=plan["job_name"],
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=None,
        confirmed=args.confirm_submit,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_sha"],
        staged_source_files={STAGED_MANIFEST: plan["manifest_json"] + "\n"},
    )
    print(f"QUEUED canary_id={plan['canary_id']} result={result!r}")


if __name__ == "__main__":
    main()
