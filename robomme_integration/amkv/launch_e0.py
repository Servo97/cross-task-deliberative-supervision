#!/usr/bin/env python3
"""Approval-gated p5e launcher for the single H10 attention-matching E0 job.

One job, one node, one sealed manifest.  Hardware is fixed to the plan-backed
p5e/H200 queue; the scientific identity is exactly the staged checkpoint,
policy source, fixtures bundle, compression ratios, and code tree, so re-running
the same inputs re-derives the same ``run_id`` and the same output prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent
LAUNCH_UTILS = REPO_ROOT / "scripts" / "launch"
for _entry in (str(REPO_ROOT), str(LAUNCH_UTILS)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from launch_guardrails import (  # noqa: E402
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    ROLE_ARN,
    TRAINING_PLAN_QUEUE,
    _ignore_sensitive_source_files,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
    training_plan_arn,
)

from robomme_integration.amkv import stage_e0  # noqa: E402
from robomme_integration.launch import IMAGE, IMAGE_SHA  # noqa: E402

ENTRY = "e0_entry.sh"
STAGED_MANIFEST = "_amkv_e0_run_manifest.json"
PACKAGE_NAME = "robomme_integration"
RUNTIME_LAUNCH_HELPERS = ("build_deterministic_archive.py", "launch_guardrails.py")
#: launch_guardrails reads its identity defaults from the repo-root settings module, so that module
#: rides along at the staged root: the node imports it from PYTHONPATH=/opt/ml/code like the package.
RUNTIME_ROOT_HELPERS = ("wsm_settings.py",)
KIND = "amkv_e0_velocity_matching_attempt"
STUDY = stage_e0.STUDY
AMKV_ROOT = stage_e0.AMKV_ROOT
RESULTS_ROOT = f"{AMKV_ROOT}/results/e0"

QUEUE = TRAINING_PLAN_QUEUE
INSTANCE_TYPE = "ml.p5e.48xlarge"
ACCELERATOR = "8xH200"
RESERVED_CAPACITY = "0"
PRIORITY = 400
MAX_RUN_SECONDS = 6 * 3600
MAX_RUN_SECONDS_CAP = 24 * 3600
VOLUME_SIZE_GB = 400
RETRY = {"attempts": 1}
DEFAULT_RATIOS = "4,8"
RUNTIME_DTYPE = "bfloat16"
NUM_FLOW_STEPS = 10
MODEL_SEED = 7
NOISE_SEED = 0
MINIMUM_EPISODES = 32
TIMING_REPEATS = 3
#: Only legal in a dry run: it makes the shape of the plumbing inspectable before ``stage_e0``
#: has published the three real inputs, and can never reach a submission.
UNSTAGED_SHA256 = "0" * 64
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seal(value: dict) -> tuple[dict, str]:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    digest = hashlib.sha256(_canonical_json(clean).encode()).hexdigest()
    clean["manifest_sha256"] = digest
    return clean, _canonical_json(clean)


def _require_sha(value: str | None, flag: str) -> str:
    if not value or not HEX64.fullmatch(value):
        raise SystemExit(f"{flag} must be 64 lowercase hexadecimal characters")
    return value


def _ratios(value: str) -> tuple[int, ...]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts or any(not item.isdigit() for item in parts):
        raise SystemExit(f"--ratios must be comma-separated positive integers; got {value!r}")
    ratios = tuple(int(item) for item in parts)
    if any(ratio < 2 for ratio in ratios) or len(set(ratios)) != len(ratios):
        raise SystemExit(f"--ratios must be unique integers >= 2; got {value!r}")
    if list(ratios) != sorted(ratios):
        raise SystemExit(f"--ratios must be ascending; got {value!r}")
    return ratios


@contextlib.contextmanager
def staged_source(package_dir: Path | None = None) -> Iterator[Path]:
    """Yield a package-rooted SageMaker source tree: ``<root>/e0_entry.sh`` + ``<root>/robomme_integration/``.

    The node imports ``robomme_integration.amkv.e0_run`` with ``PYTHONPATH=/opt/ml/code``, so the
    staged root must be the package *parent*, unlike the flat RoboMME training bundle.
    """
    package_dir = Path(package_dir or PACKAGE_DIR).resolve()
    entry = package_dir / "amkv" / ENTRY
    if not entry.is_file() or not (package_dir / "__init__.py").is_file():
        raise SystemExit(f"invalid AMKV source package: {package_dir}")
    with tempfile.TemporaryDirectory(prefix="amkv-e0-source-") as temporary:
        root = Path(temporary) / "source"
        root.mkdir()
        shutil.copytree(
            package_dir,
            root / PACKAGE_NAME,
            symlinks=True,
            ignore=_ignore_sensitive_source_files,
        )
        shutil.copy2(entry, root / ENTRY)
        (root / ENTRY).chmod(0o755)
        # e0_run re-validates the source receipt through stage_e0 on the node.
        # stage_e0 intentionally imports these two shared, stdlib-only helpers
        # from <repo>/scripts/launch, so they are part of the sealed runtime
        # source tree rather than an undeclared host dependency.
        helpers = root / "scripts" / "launch"
        helpers.mkdir(parents=True)
        for name in RUNTIME_LAUNCH_HELPERS:
            shutil.copy2(LAUNCH_UTILS / name, helpers / name)
        for name in RUNTIME_ROOT_HELPERS:
            shutil.copy2(REPO_ROOT / name, root / name)
        yield root


CODE_PREFLIGHT_MARKER = "amkv_code_preflight.json"
CODE_PREFLIGHT_TESTS = "test_amkv_*.py"


def code_preflight_marker_path(archive_dir: Path | None = None) -> Path:
    return Path(archive_dir or stage_e0.DEFAULT_ARCHIVE_DIR) / CODE_PREFLIGHT_MARKER


def shipped_source_sha256(
    package_dir: Path | None = None,
    *,
    secrets_manager_arn: str | None = None,
) -> str:
    """The exact twice-sanitized tree SHA ``submit_training_job`` would ship."""

    with staged_source(package_dir) as source:
        with prepared_source_bundle(
            source,
            ENTRY,
            {"SAGEMAKER_PROGRAM": ENTRY},
            secrets_manager_arn,
        ) as (staged, _entry, _environment):
            return source_tree_sha256(staged)


def code_preflight(
    *,
    package_dir: Path | None = None,
    test_python: str,
    archive_dir: Path | None = None,
    secrets_manager_arn: str | None = None,
) -> dict:
    """Run this lane's tests against the tree that is about to ship, and seal it.

    Standing practice after E0 attempt 2: the lane's modules are edited by more
    than one writer, so "the suite passed earlier" is not evidence about the
    tree being packaged now.  Binding the test result to the sanitized tree SHA
    makes a drifted tree unshippable rather than merely suspicious.
    """

    repository = Path(__file__).resolve().parents[2]
    tests = sorted(str(path) for path in (repository / "robomme_integration" / "tests").glob(CODE_PREFLIGHT_TESTS))
    if not tests:
        raise SystemExit("no amkv tests found; refusing to seal an empty code preflight")
    completed = subprocess.run(
        [test_python, "-m", "pytest", *tests, "-q"],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository), "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    if completed.returncode != 0:
        raise SystemExit(f"amkv tests failed; not sealing a code preflight:\n{tail}")
    sha = shipped_source_sha256(package_dir, secrets_manager_arn=secrets_manager_arn)
    marker = {
        "schema_version": 1,
        "kind": "amkv_code_preflight",
        "sanitized_source_tree_sha256": sha,
        "test_selector": CODE_PREFLIGHT_TESTS,
        "test_files": len(tests),
        "pytest_summary": tail,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = code_preflight_marker_path(archive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"stage": "code-preflight", "marker": str(path), **marker}


def require_code_preflight(expected_tree_sha256: str, *, archive_dir: Path | None = None) -> dict:
    path = code_preflight_marker_path(archive_dir)
    if not path.is_file():
        raise SystemExit(
            "no code preflight on record; run `python -m robomme_integration.amkv.launch_e0 "
            "--code-preflight --test-python <interpreter>` before submitting"
        )
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("sanitized_source_tree_sha256") != expected_tree_sha256:
        raise SystemExit(
            "code preflight covers a different tree than the one about to ship: "
            f"sealed {marker.get('sanitized_source_tree_sha256')}, shipping {expected_tree_sha256}"
        )
    return marker


def _validate(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.confirm_submit:
        raise SystemExit("submission blocked: obtain explicit user approval, then pass --confirm-submit")
    if args.queue != QUEUE:
        raise SystemExit(f"AMKV E0 is pinned to the p5e training-plan queue {QUEUE}; got {args.queue}")
    if training_plan_arn(args.queue) is None:
        raise SystemExit(f"queue {args.queue} lost its required training-plan ARN")
    if args.role != ROLE_ARN:
        raise SystemExit(f"AMKV E0 must use execution role {ROLE_ARN}")
    if args.priority > PRIORITY:
        raise SystemExit(
            f"AMKV E0 must not exceed priority {PRIORITY}; priority {args.priority} needs a new user approval"
        )
    if args.priority < 1:
        raise SystemExit("--priority must be positive")
    if not 1 <= args.max_run_seconds <= MAX_RUN_SECONDS_CAP:
        raise SystemExit(f"AMKV E0 must fit within {MAX_RUN_SECONDS_CAP} seconds")
    if args.volume_size_gb != VOLUME_SIZE_GB:
        raise SystemExit(f"AMKV E0 requires the measured-safe {VOLUME_SIZE_GB} GiB ephemeral volume")
    # Plan-backed queue: the pinned TrainingPlanArn REPLACES the implicit reserved-capacity request.
    if args.reserved_capacity != RESERVED_CAPACITY:
        raise SystemExit(
            f"plan-backed queue {args.queue} requires SM_USE_RESERVED_CAPACITY="
            f"{RESERVED_CAPACITY}; got {args.reserved_capacity!r}"
        )
    if args.secrets_manager_arn and not args.secrets_manager_arn.startswith("arn:aws:secretsmanager:"):
        raise SystemExit("--secrets-manager-arn must be an AWS Secrets Manager ARN")


def build_plan(args: argparse.Namespace, source_dir: Path) -> dict:
    _validate(args)
    ratios = _ratios(args.ratios)
    checkpoint_inventory_sha = _require_sha(args.checkpoint_inventory_sha256, "--checkpoint-inventory-sha256")
    fixtures_manifest_sha = _require_sha(args.fixtures_manifest_sha256, "--fixtures-manifest-sha256")
    receipt_sha = _require_sha(args.policy_source_receipt_sha256, "--policy-source-receipt-sha256")
    receipt, actual_receipt_sha = stage_e0.load_source_receipt(
        Path(args.policy_source_receipt), expected_sha256=receipt_sha
    )
    if actual_receipt_sha != receipt_sha:
        raise SystemExit("source receipt changed after validation")
    policy_source_sha = receipt["archive"]["sha256"]
    unstaged = sorted(
        name
        for name, value in {
            "checkpoint_inventory": checkpoint_inventory_sha,
            "fixtures_manifest": fixtures_manifest_sha,
        }.items()
        if value == UNSTAGED_SHA256
    )
    if unstaged and not args.dry_run:
        raise SystemExit(
            "refusing to submit with unstaged placeholder inputs: "
            + ", ".join(unstaged)
            + "; run robomme_integration.amkv.stage_e0 first"
        )

    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}, args.secrets_manager_arn) as (
        staged,
        _entry,
        _environment,
    ):
        source_sha = source_tree_sha256(staged)
    entry_sha = hashlib.sha256((Path(source_dir) / ENTRY).read_bytes()).hexdigest()

    checkpoint_s3 = stage_e0.checkpoint_uri()
    checkpoint_inventory_s3 = stage_e0.checkpoint_inventory_uri(checkpoint_inventory_sha)
    policy_source_s3 = stage_e0.policy_source_uri(policy_source_sha)
    policy_source_receipt_s3 = stage_e0.source_receipt_uri(receipt_sha)
    fixtures_s3 = stage_e0.fixtures_uri(fixtures_manifest_sha)
    scientific = {
        "schema_version": 1,
        "study": STUDY,
        "benchmark": "RoboMME",
        "hypothesis": "H10",
        "experiment": "attention_matching_E0",
        "objective": "velocity_matching",
        "checkpoint": {
            "artifact": stage_e0.CHECKPOINT_ARTIFACT,
            "uri": checkpoint_s3,
            "inventory_uri": checkpoint_inventory_s3,
            "inventory_sha256": checkpoint_inventory_sha,
        },
        "policy_source": {
            "component": stage_e0.POLICY_SOURCE_COMPONENT,
            "uri": policy_source_s3,
            "sha256": policy_source_sha,
            "git_sha": stage_e0.PINNED_POLICY_GIT_SHA,
            "git_tree_sha1": stage_e0.PINNED_POLICY_TREE_SHA1,
            "receipt_uri": policy_source_receipt_s3,
            "receipt_sha256": receipt_sha,
            "extracted_tree_sha256": receipt["extracted_tree"]["tree_sha256"],
            "extracted_tree_objects": receipt["extracted_tree"]["totals"]["objects"],
            "extracted_tree_bytes": receipt["extracted_tree"]["totals"]["bytes"],
        },
        "fixtures": {"uri": fixtures_s3, "manifest_sha256": fixtures_manifest_sha},
        "ratios": list(ratios),
        "evaluation": {
            "runtime_dtype": RUNTIME_DTYPE,
            "num_flow_steps": NUM_FLOW_STEPS,
            "model_seed": MODEL_SEED,
            "noise_seed": NOISE_SEED,
            "minimum_episodes": MINIMUM_EPISODES,
            "timing_repeats": TIMING_REPEATS,
            "velocity_estimand": "same_full_teacher_x_t_at_each_flow_time",
            "timing_estimand": "python_unrolled_denoiser_microbenchmark_only",
            "speedup_claim_permitted": False,
        },
        "code": {
            "sanitized_source_tree_sha256": source_sha,
            "entry": ENTRY,
            "entry_sha256": entry_sha,
            "runner": f"{PACKAGE_NAME}.amkv.e0_run",
        },
        "image": {"uri": args.image_uri, "sha256": IMAGE_SHA},
    }
    scientific_sha = hashlib.sha256(_canonical_json(scientific).encode()).hexdigest()
    run_id = f"amkv-e0-{scientific_sha[:16]}"
    output = f"{RESULTS_ROOT}/{run_id}"
    infrastructure = {
        "provider": "aws_sagemaker",
        "execution_account": EXECUTION_ACCOUNT,
        "queue": args.queue,
        "training_plan_arn": training_plan_arn(args.queue),
        "role": args.role,
        "instance_type": INSTANCE_TYPE,
        "accelerator": ACCELERATOR,
        "priority": args.priority,
        "max_run_seconds": args.max_run_seconds,
        "volume_size_gb": args.volume_size_gb,
        "reserved_capacity": args.reserved_capacity,
        "attempts_in_job": RETRY["attempts"],
    }
    manifest, manifest_json = _seal(
        {
            "schema_version": 1,
            "kind": KIND,
            "run_id": run_id,
            "scientific_spec_sha256": scientific_sha,
            "scientific": scientific,
            "infrastructure": infrastructure,
            "output_s3": output,
        }
    )
    environment = {
        "SM_USE_RESERVED_CAPACITY": args.reserved_capacity,
        "AMKV_POLICY_SOURCE_S3": policy_source_s3,
        "AMKV_POLICY_SOURCE_SHA256": policy_source_sha,
        "AMKV_POLICY_SOURCE_RECEIPT_S3": policy_source_receipt_s3,
        "AMKV_POLICY_SOURCE_RECEIPT_SHA256": receipt_sha,
        "AMKV_CHECKPOINT_S3": checkpoint_s3,
        "AMKV_CHECKPOINT_INVENTORY_S3": checkpoint_inventory_s3,
        "AMKV_CHECKPOINT_INVENTORY_SHA256": checkpoint_inventory_sha,
        "AMKV_FIXTURES_S3": fixtures_s3,
        "AMKV_FIXTURES_MANIFEST_SHA256": fixtures_manifest_sha,
        "AMKV_OUTPUT_S3": output,
        "AMKV_RATIOS": ",".join(str(ratio) for ratio in ratios),
        "AMKV_RUN_ID": run_id,
        "AMKV_CODE_SOURCE_TREE_SHA256": source_sha,
        "AMKV_POLICY_GIT_SHA": stage_e0.PINNED_POLICY_GIT_SHA,
        "AMKV_POLICY_TREE_SHA1": stage_e0.PINNED_POLICY_TREE_SHA1,
        "RUN_MANIFEST_SOURCE": STAGED_MANIFEST,
        "RUN_MANIFEST_SHA256": manifest["manifest_sha256"],
    }
    oversized = {key: len(value.encode()) for key, value in environment.items() if len(value.encode()) > 512}
    if oversized:
        raise SystemExit(f"SageMaker environment values exceed 512 bytes: {oversized}")
    return {
        "run_id": run_id,
        "output": output,
        "manifest": manifest,
        "manifest_json": manifest_json + "\n",
        "environment": environment,
        "source_sha": source_sha,
        "unstaged_inputs": unstaged,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--policy-source-receipt",
        help="local create-once receipt emitted by stage_e0 source",
    )
    value.add_argument("--policy-source-receipt-sha256")
    value.add_argument("--checkpoint-inventory-sha256", default=UNSTAGED_SHA256)
    value.add_argument("--fixtures-manifest-sha256", default=UNSTAGED_SHA256)
    value.add_argument("--ratios", default=DEFAULT_RATIOS)
    value.add_argument("--source-package")
    value.add_argument("--queue", default=QUEUE)
    value.add_argument("--role", default=ROLE_ARN)
    value.add_argument("--image-uri", default=IMAGE)
    value.add_argument("--priority", type=int, default=PRIORITY)
    value.add_argument("--max-run-seconds", type=int, default=MAX_RUN_SECONDS)
    value.add_argument("--volume-size-gb", type=int, default=VOLUME_SIZE_GB)
    value.add_argument("--reserved-capacity", default=RESERVED_CAPACITY)
    value.add_argument("--secrets-manager-arn")
    value.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default; --no-dry-run alone is still refused without --confirm-submit",
    )
    value.add_argument("--confirm-submit", action="store_true")
    # Two preflights. Attempt 1 died because an unpinned
    # `uv sync` resolved an interpreter with no wheel for a transitive
    # dependency; attempt 2 died because the lane's modules changed between the
    # last green test run and the submit. The environment preflight has no
    # submission escape hatch; neither failure is detectable from a manifest.
    value.add_argument("--code-preflight", action="store_true", help="run the lane tests and seal the tree SHA")
    value.add_argument(
        "--test-python",
        default=str(stage_e0.OFFICIAL_REFERENCE / "robomme_policy_learning" / ".venv" / "bin" / "python"),
        help="interpreter used for --code-preflight (needs jax + the official policy package)",
    )
    value.add_argument(
        "--skip-code-preflight",
        action="store_true",
        help="opt-in escape hatch; re-opens the drifted-tree failure mode that killed attempt 2",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.code_preflight:
        print(
            json.dumps(
                code_preflight(
                    test_python=args.test_python,
                    secrets_manager_arn=args.secrets_manager_arn,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.confirm_submit:
        args.dry_run = False
    if args.confirm_submit:
        # Standing practice for this lane after E0 attempt 1: any submission
        # whose node builds an environment fresh must first be preceded by a
        # clean-scratch `uv sync --frozen`.  An already-built local venv cannot
        # observe the resolution step, which is where attempt 1 died.
        marker = stage_e0.require_env_preflight(source_dir=Path(stage_e0.POLICY_SOURCE_DIR))
        print(
            f"env-preflight OK python={marker['python']} uv_lock={marker['uv_lock_sha256'][:16]} "
            f"checked_at={marker['checked_at']}"
        )
    with staged_source(args.source_package) as source_dir:
        plan = build_plan(args, source_dir)
        manifest = plan["manifest"]
        print(
            f"experiment=amkv_e0 run_id={plan['run_id']} ratios={plan['environment']['AMKV_RATIOS']}\n"
            f"  checkpoint={plan['environment']['AMKV_CHECKPOINT_S3']}\n"
            f"  checkpoint_inventory={plan['environment']['AMKV_CHECKPOINT_INVENTORY_S3']}\n"
            f"  policy_source={plan['environment']['AMKV_POLICY_SOURCE_S3']}\n"
            f"  policy_source_receipt={plan['environment']['AMKV_POLICY_SOURCE_RECEIPT_S3']}\n"
            f"  fixtures={plan['environment']['AMKV_FIXTURES_S3']}\n"
            f"  output={plan['output']}\n"
            f"  manifest_sha256={manifest['manifest_sha256']} source_tree={plan['source_sha']}\n"
            f"  queue={args.queue} plan={manifest['infrastructure']['training_plan_arn']}\n"
            f"  instance={INSTANCE_TYPE} priority={args.priority} "
            f"max_run={args.max_run_seconds}s volume={args.volume_size_gb}GiB dry={args.dry_run}"
        )
        if args.confirm_submit and not args.skip_code_preflight:
            marker = require_code_preflight(plan["source_sha"])
            print(
                f"code-preflight OK tree={marker['sanitized_source_tree_sha256'][:16]} "
                f'tests="{marker["pytest_summary"]}" checked_at={marker["checked_at"]}'
            )
        if plan["unstaged_inputs"]:
            print(
                "WARNING unstaged placeholder inputs: "
                + ", ".join(plan["unstaged_inputs"])
                + " — this run_id is NOT the scientific identity; run stage_e0 first"
            )
        if args.dry_run:
            print(json.dumps(manifest, sort_keys=True, indent=2))
            print(json.dumps(plan["environment"], sort_keys=True, indent=2))
            print("DRY RUN ONLY — no AWS SDK loaded and no cloud write performed")
            return 0

        stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
        job_name = f"sarvesh-amkv-e0-{plan['run_id'][-16:]}-{stamp}"[:63]
        if not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", job_name):
            raise SystemExit(f"invalid SageMaker TrainingJobName after normalization: {job_name}")
        result = submit_training_job(
            entry=ENTRY,
            source_dir=source_dir,
            environment=plan["environment"],
            image_uri=args.image_uri,
            instance_type=INSTANCE_TYPE,
            volume_size=args.volume_size_gb,
            tags=[
                {"Key": "tri.project", "Value": PROJECT_TAG},
                {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
                {"Key": "wsm.study", "Value": STUDY},
                {"Key": "wsm.benchmark", "Value": "RoboMME"},
                {"Key": "wsm.experiment", "Value": "amkv_e0"},
                {"Key": "wsm.run_id", "Value": plan["run_id"]},
            ],
            retry_config=RETRY,
            job_name=job_name,
            queue=args.queue,
            role=args.role,
            priority=args.priority,
            max_run_seconds=args.max_run_seconds,
            secrets_manager_arn=args.secrets_manager_arn,
            confirmed=args.confirm_submit,
            disable_profiler=True,
            expected_source_tree_sha256=plan["source_sha"],
            staged_source_files={STAGED_MANIFEST: plan["manifest_json"]},
        )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
