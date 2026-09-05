"""Shared fail-closed guardrails for TRI SageMaker Batch launchers.

This module intentionally imports AWS SDKs lazily so validation and dry-run tests stay offline.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import shutil
import stat
import sys
import tempfile

try:
    import wsm_settings
except ModuleNotFoundError:  # run as scripts/launch/<launcher>.py: the repository root is not on sys.path yet
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import wsm_settings

# Identity and infrastructure defaults live in wsm_settings (env-overridable; README "Configuration").
# They are re-exported here so the launchers keep this module as their single import point.
REGION = wsm_settings.REGION
EXECUTION_ACCOUNT = wsm_settings.EXECUTION_ACCOUNT
# Study storage moved to account 141 (2026-07-22): the BatchOperator role has full read/write +
# object-copy there, whereas the old 124 bucket denied s3:ListBucketVersions/GetBucketVersioning
# cross-account. Storage now == execution == 141.
STORAGE_ACCOUNT = wsm_settings.STORAGE_ACCOUNT
LEGACY_ACCOUNT = wsm_settings.LEGACY_ACCOUNT
#: The S3 storage prefix owner. Every content address is minted under it, so it never moves -- it is
#: NOT the submitting identity (that is OWNER_EMAIL, the `tri.owner.email` SCP tag).
STUDY_OWNER = wsm_settings.STUDY_OWNER
OWNER_EMAIL = wsm_settings.OWNER_EMAIL
PROJECT_TAG = wsm_settings.PROJECT_TAG
IMAGE_OWNER = wsm_settings.IMAGE_OWNER
DEXJOCO_IMAGE_REPO = wsm_settings.DEXJOCO_IMAGE_REPO
WSM_ROBOCASA_S3 = wsm_settings.WSM_ROBOCASA_S3
LONG_CONTEXT_STUDY_S3 = wsm_settings.LONG_CONTEXT_STUDY_S3
ROLE_ARN = f"arn:aws:iam::{EXECUTION_ACCOUNT}:role/CAM-Robotics-Sagemaker-role-us-west-2"
QUEUE = "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
# Opt-in scavenger tier on the p5e training plan. NOT the default: reaching it requires an explicit
# --queue. Its priority classes are its own — jobs here run at the bottom of the plan and may be
# preempted, so the MULTI_DAY_PRIORITY rule below (a cam-robotics shared-queue policy) does not
# apply and priority 1 is the intended class. Approved for the 2026-08-03 norm-split retrains.
TRAINING_PLAN_QUEUE = "fss-tri-cam-robotics-p5e-48xlarge-us-west-2-training-plan"
ALLOWED_QUEUES = (QUEUE, TRAINING_PLAN_QUEUE)
#: The shared-compute roster spells the queues with a ``shared-compute__<region>__`` display
#: prefix; AWS Batch (and therefore TrainingQueue) wants the bare job-queue name. Accept the
#: roster spelling on the CLI and normalize, so a copy-paste from the roster cannot mint a
#: queue name that silently fails at submission time.
QUEUE_ALIASES = {f"shared-compute__{REGION}__{name}": name for name in ALLOWED_QUEUES}


#: Plan-backed queues do NOT attach to their reserved capacity implicitly. The job must pin the
#: flexible training plan itself: the SageMaker Estimator's ``training_plan`` kwarg becomes
#: ``ResourceConfig.TrainingPlanArn``, which the Batch queue forwards in its serviceRequestPayload.
#: Without it a submit is ACCEPTED and then sits in SCHEDULED forever against idle capacity — the
#: 2026-08-04 combo-arm stall. Derived from the queue rather than exposed as a flag so no launcher
#: can forget it. Reference: TRI-ML/vla_foundry_internal PR 822 (14c0d54) sagemaker/launch_training.py.
TRAINING_PLAN_ARNS = {
    TRAINING_PLAN_QUEUE: f"arn:aws:sagemaker:{REGION}:{EXECUTION_ACCOUNT}:training-plan/cam-robotics-tp",
}


def normalize_queue(value: str) -> str:
    return QUEUE_ALIASES.get(value, value)


def training_plan_arn(queue: str) -> str | None:
    """The training-plan ARN a queue's jobs must pin, or None for an ordinary on-demand queue."""
    return TRAINING_PLAN_ARNS.get(normalize_queue(queue))


DEFAULT_RESULTS_BUCKET = wsm_settings.RESULTS_BUCKET
MULTI_DAY_PRIORITY = 600
MULTI_DAY_THRESHOLD_SECONDS = 24 * 3600
#: User directive 2026-09-02: standard-class (400) experiments may run up to two days without
#: escalating to the sprint class — "run 48 hr runs on 400 priority; if they are stopped by some
#: ghost process, we can reevaluate." Longer than that still requires priority 600.
STANDARD_PRIORITY = 400
STANDARD_TWO_DAY_MAX_SECONDS = 48 * 3600
MAX_RUN_SECONDS = 5 * 24 * 3600
SECURE_ENTRY = "_wsm_secure_entry.sh"
_SENSITIVE_SOURCE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "gsheets_sa.json",
        "secrets.env",
    }
)
_SOURCE_EXCLUDED_NAMES = _SENSITIVE_SOURCE_NAMES | frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "wandb",
        # non-source artifacts that must not perturb the scientific source identity
        ".claude",
        "export.zip",
        "overleaf",
        "checkpoints",  # downloaded model weights (e.g. the openpi fork's 15GB checkpoints/) are not source
        "wsm_data",
        # Local dataset staging for the GCP/TPU workflow: 133 GB of lerobot parquet under the repo root.
        # Copying it into the staged bundle filled /tmp and aborted an archive build (2026-08-05); had it
        # fit, it would have silently entered the content-addressed source identity of every later run.
        ".gcp_tpu_staging_uscentral2",
    }
)

_SENSITIVE_ENV_KEYS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
    }
)


def add_guardrail_arguments(parser, *, default_max_run_seconds=MAX_RUN_SECONDS):
    """Add the common queue, timeout, secret-reference, and confirmation flags."""
    default_priority = MULTI_DAY_PRIORITY if default_max_run_seconds > MULTI_DAY_THRESHOLD_SECONDS else 1
    parser.add_argument(
        "--queue",
        default=QUEUE,
        type=normalize_queue,
        help=(
            f"cam-robotics queue; default {QUEUE}. Opt into {TRAINING_PLAN_QUEUE} explicitly "
            "(the shared-compute__<region>__ roster spelling is accepted and normalized)"
        ),
    )
    parser.add_argument("--role", default=ROLE_ARN, help="fixed cam-robotics execution role")
    parser.add_argument(
        "--priority",
        type=int,
        default=default_priority,
        help="must be 600 when --max-run-seconds is greater than one day",
    )
    parser.add_argument(
        "--max-run-seconds",
        type=int,
        default=default_max_run_seconds,
        help=f"SageMaker and Batch timeout, capped at {MAX_RUN_SECONDS} seconds (5 days)",
    )
    parser.add_argument(
        "--secrets-manager-arn",
        default=None,
        help=(
            "optional Secrets Manager JSON ARN with HF_TOKEN/WANDB_API_KEY/WANDB_ENTITY; "
            "the role fetches it on-node and only the ARN enters job metadata"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-submit",
        action="store_true",
        help="required for submission; pass only after explicit user approval",
    )


def require_submission_confirmation(*, dry_run, confirmed):
    if not dry_run and not confirmed:
        raise SystemExit("submission blocked: obtain explicit user approval, then rerun with --confirm-submit")


def validate_launch_contract(*, queue, role, priority, max_run_seconds, secrets_manager_arn=None):
    """Fail closed on infrastructure drift, excess runtime, and unsafe secret references."""
    if queue not in ALLOWED_QUEUES:
        raise SystemExit(
            f"all runs must use one of the approved cam-robotics queues {list(ALLOWED_QUEUES)}; got {queue}"
        )
    if role != ROLE_ARN:
        raise SystemExit(f"cam-robotics runs must use execution role {ROLE_ARN}; got {role}")
    if not 1 <= max_run_seconds <= MAX_RUN_SECONDS:
        raise SystemExit(f"--max-run-seconds must be in [1, {MAX_RUN_SECONDS}]")
    if (
        queue != TRAINING_PLAN_QUEUE
        and max_run_seconds > MULTI_DAY_THRESHOLD_SECONDS
        and priority != MULTI_DAY_PRIORITY
        and not (priority == STANDARD_PRIORITY and max_run_seconds <= STANDARD_TWO_DAY_MAX_SECONDS)
    ):
        raise SystemExit(
            f"runs longer than one day must use --priority {MULTI_DAY_PRIORITY}, or "
            f"--priority {STANDARD_PRIORITY} with --max-run-seconds <= "
            f"{STANDARD_TWO_DAY_MAX_SECONDS} (two-day standard class, 2026-09-02); got {priority}"
        )
    if secrets_manager_arn and not secrets_manager_arn.startswith("arn:aws:secretsmanager:"):
        raise SystemExit("--secrets-manager-arn must be an AWS Secrets Manager ARN")


def validate_and_confirm(args):
    """Validate before any caller-identity or S3 lookup can occur."""
    require_submission_confirmation(dry_run=args.dry_run, confirmed=args.confirm_submit)
    validate_launch_contract(
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=args.secrets_manager_arn,
    )


def resolve_user(arg):
    if arg:
        return arg
    if os.environ.get("WSM_USER"):
        return os.environ["WSM_USER"]
    import boto3

    arn = boto3.client("sts", region_name=REGION).get_caller_identity()["Arn"]
    return arn.rstrip("/").split("/")[-1].split("@")[0]


def load_aws_sdk():
    """Load submission-only dependencies after confirmation and dry-run gates."""
    import boto3
    import sagemaker
    from sagemaker.estimator import Estimator

    try:
        from sagemaker.train.aws_batch.training_queue import TrainingQueue
    except Exception:
        from sagemaker.aws_batch.training_queue import TrainingQueue
    return boto3, sagemaker, Estimator, TrainingQueue


def validate_caller_account(boto3_module):
    account = boto3_module.client("sts", region_name=REGION).get_caller_identity()["Account"]
    if account != EXECUTION_ACCOUNT:
        raise SystemExit(f"submission requires AWS account {EXECUTION_ACCOUNT}; caller is {account}")


def _ignore_sensitive_source_files(_directory, names):
    ignored = []
    for name in names:
        lower = name.lower()
        if (
            lower in _SOURCE_EXCLUDED_NAMES
            or lower.startswith(".env.")
            or (
                lower.endswith(".json")
                and any(
                    marker in lower
                    for marker in (
                        "credential",
                        "service-account",
                        "service_account",
                        "secret",
                    )
                )
            )
            or lower.endswith((".log", ".pem", ".key", ".pyc"))
        ):
            ignored.append(name)
    return ignored


def source_tree_sha256(root):
    """Hash the exact sanitized source tree without timestamps or filesystem ordering.

    Paths, entry types, permission bits, symlink targets, and regular-file bytes are covered. The
    digest is stable across staging directories but changes for any submitted source mutation that
    can affect execution.
    """
    root = pathlib.Path(root)
    digest = hashlib.sha256()

    def field(value):
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        field(relative)
        field(oct(stat.S_IMODE(mode)))
        if path.is_symlink():
            field("symlink")
            field(os.readlink(path))
        elif path.is_dir():
            field("directory")
        elif path.is_file():
            field("file")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    field(block)
        else:
            raise SystemExit(f"unsupported source entry type: {path}")
    return digest.hexdigest()


def write_staged_source_files(root, files):
    """Add generated, non-secret files to the already sanitized SageMaker source tree."""
    root = pathlib.Path(root).resolve()
    for relative, value in (files or {}).items():
        relative_path = pathlib.PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
            raise SystemExit(f"unsafe staged source path: {relative!r}")
        destination = root.joinpath(*relative_path.parts)
        resolved_parent = destination.parent.resolve()
        if not resolved_parent.is_relative_to(root):
            raise SystemExit(f"staged source path escapes through a symlink: {relative!r}")
        if destination.exists() or destination.is_symlink():
            raise SystemExit(f"staged source file would overwrite existing path: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
        temporary = destination.with_name("." + destination.name + ".incomplete")
        temporary.write_bytes(data)
        temporary.replace(destination)
        destination.chmod(0o600)


_SECURE_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
secret_json="$(aws secretsmanager get-secret-value --secret-id "$WSM_SECRETS_MANAGER_ARN" --region us-west-2 --query SecretString --output text)"
read_secret() {
  python3 -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$1" <<<"$secret_json"
}
export HF_TOKEN="$(read_secret HF_TOKEN)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$(read_secret WANDB_API_KEY)"
wandb_entity="$(read_secret WANDB_ENTITY)"
if [[ -n "$wandb_entity" ]]; then
  export WANDB_ENTITY="$wandb_entity"
fi
unset secret_json wandb_entity
exec "./$WSM_ORIGINAL_ENTRY" "$@"
"""


@contextlib.contextmanager
def prepared_source_bundle(source_dir, entry, environment, secrets_manager_arn=None, vendor_files=None):
    """Yield a secret-free source tree, safe entrypoint, and copied environment.

    A secret ARN enables an on-node wrapper that fetches its JSON through the execution role.
    Without an ARN, W&B is disabled and jobs rely only on role/S3 or image-cached assets.

    ``vendor_files`` is an optional sequence of ``(source_path, relative_destination)`` pairs copied
    into the staged tree before it is hashed and shipped, for node-side code that imports modules
    living outside ``source_dir`` (e.g. ``launch_guardrails`` / ``wsm_settings`` for a bundle that
    ships only ``robomme_integration/``). Opt-in per launcher so other bundles keep their identities.
    """
    source_dir = pathlib.Path(source_dir)
    sensitive_keys = sorted(key for key, value in environment.items() if key.upper() in _SENSITIVE_ENV_KEYS and value)
    if sensitive_keys:
        raise SystemExit(
            "plaintext sensitive environment keys are forbidden; use --secrets-manager-arn: "
            + ", ".join(sensitive_keys)
        )
    with tempfile.TemporaryDirectory(prefix="wsm-launch-source-") as tmp:
        staged = pathlib.Path(tmp) / "source"
        shutil.copytree(
            source_dir,
            staged,
            symlinks=True,
            ignore=_ignore_sensitive_source_files,
        )
        if not (staged / entry).exists():
            raise SystemExit(f"sanitized source-dir missing required {entry}")
        for vendor_src, vendor_rel in vendor_files or ():
            vendor_src = pathlib.Path(vendor_src)
            if not vendor_src.is_file():
                raise SystemExit(f"vendor file missing: {vendor_src}")
            vendor_dest = staged / vendor_rel
            vendor_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vendor_src, vendor_dest)

        safe_environment = dict(environment)
        launch_entry = entry
        if secrets_manager_arn:
            wrapper = staged / SECURE_ENTRY
            wrapper.write_text(_SECURE_WRAPPER, encoding="utf-8")
            wrapper.chmod(0o700)
            launch_entry = SECURE_ENTRY
            safe_environment.update(
                {
                    "SAGEMAKER_PROGRAM": SECURE_ENTRY,
                    "WSM_ORIGINAL_ENTRY": entry,
                    "WSM_SECRETS_MANAGER_ARN": secrets_manager_arn,
                }
            )
        else:
            safe_environment["SAGEMAKER_PROGRAM"] = entry
            safe_environment["WANDB_MODE"] = "disabled"

        yield staged, launch_entry, safe_environment


def submit_training_job(
    *,
    entry,
    source_dir,
    environment,
    image_uri,
    instance_type,
    volume_size,
    tags,
    retry_config,
    job_name,
    queue,
    role,
    priority,
    max_run_seconds,
    secrets_manager_arn,
    confirmed,
    keep_alive_period_in_seconds=300,
    disable_profiler=None,
    expected_source_tree_sha256=None,
    staged_source_files=None,
    vendor_files=None,
):
    """Submit one guarded SageMaker TrainingQueue job and return the queue result."""
    require_submission_confirmation(dry_run=False, confirmed=confirmed)
    validate_launch_contract(
        queue=queue,
        role=role,
        priority=priority,
        max_run_seconds=max_run_seconds,
        secrets_manager_arn=secrets_manager_arn,
    )
    boto3, sagemaker, Estimator, TrainingQueue = load_aws_sdk()
    validate_caller_account(boto3)
    with prepared_source_bundle(source_dir, entry, environment, secrets_manager_arn, vendor_files=vendor_files) as (
        safe_source_dir,
        safe_entry,
        safe_environment,
    ):
        if expected_source_tree_sha256:
            actual_source_sha256 = source_tree_sha256(safe_source_dir)
            if actual_source_sha256 != expected_source_tree_sha256:
                raise SystemExit(
                    "sanitized source bundle changed after manifest construction: "
                    f"expected={expected_source_tree_sha256} actual={actual_source_sha256}"
                )
        write_staged_source_files(safe_source_dir, staged_source_files)
        estimator_kwargs = {
            "entry_point": safe_entry,
            "source_dir": str(safe_source_dir),
            "sagemaker_session": sagemaker.Session(boto_session=boto3.session.Session(region_name=REGION)),
            "role": role,
            "image_uri": image_uri,
            "instance_count": 1,
            "instance_type": instance_type,
            "input_mode": "FastFile",
            "max_run": max_run_seconds,
            "environment": safe_environment,
            "volume_size": volume_size,
            "keep_alive_period_in_seconds": keep_alive_period_in_seconds,
            "tags": tags,
        }
        if disable_profiler is not None:
            estimator_kwargs["disable_profiler"] = disable_profiler
        # SageMaker Debugger's HOOK is a SEPARATE thing from the profiler, and the SDK attaches a
        # default DebugHookConfig unless it is explicitly switched off. As of 2026-08-14 SageMaker
        # rejects CreateTrainingJob outright for accounts not already onboarded to Debugger:
        #   "SageMaker Debugger is in maintenance mode and is not available to new customers"
        #   (Status Code: 400)
        # That 400 killed both H13 sealed eval submissions at the API layer — no Batch attempt, no
        # training job, no log stream, so it presents as an instant FAILED with an empty attempts
        # list. Nothing in this study ever reads a Debugger artifact, so the hook is disabled
        # unconditionally for every launcher rather than per-caller.
        estimator_kwargs["debugger_hook_config"] = False
        # Plan-backed queue: pin the plan (-> ResourceConfig.TrainingPlanArn) or the job never
        # leaves SCHEDULED. An explicitly pinned plan REPLACES the implicit reserved-capacity
        # request, so the caller must also have set SM_USE_RESERVED_CAPACITY=0 (the reference
        # implementation flips exactly that pair). Checked rather than silently rewritten: the
        # environment the caller sealed into its plan must be the environment the job receives.
        plan_arn = training_plan_arn(queue)
        if plan_arn:
            estimator_kwargs["training_plan"] = plan_arn
            if safe_environment.get("SM_USE_RESERVED_CAPACITY") != "0":
                raise SystemExit(
                    f"queue {queue} is plan-backed: pin {plan_arn} AND set "
                    "SM_USE_RESERVED_CAPACITY=0; got "
                    f"{safe_environment.get('SM_USE_RESERVED_CAPACITY')!r}"
                )
        elif safe_environment.get("SM_USE_RESERVED_CAPACITY") == "0":
            raise SystemExit(f"queue {queue} is not plan-backed but the job disables reserved capacity")
        estimator = Estimator(**estimator_kwargs)
        training_queue = TrainingQueue(queue_name=queue)
        return training_queue.map(
            estimator,
            inputs=[None],
            job_names=[job_name],
            priority=priority,
            share_identifier="default",
            retry_config=retry_config,
            timeout={"attemptDurationSeconds": max_run_seconds},
        )
