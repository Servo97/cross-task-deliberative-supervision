#!/usr/bin/env python3
"""Submit only the exact sealed V4-C5 r4 canary under the user-authorized backlog policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robomme_integration import launch  # noqa: E402
from robomme_integration.cloud_admission_p5 import (  # noqa: E402
    collect_canary_backlog_admission,
)
from robomme_integration.v4_policy_canary_launch import (  # noqa: E402
    ENTRY,
    MAX_RUN_SECONDS,
    VOLUME_SIZE_GB,
    build,
)
from scripts.launch.launch_guardrails import OWNER_EMAIL, PROJECT_TAG  # noqa: E402

PLAN_SHA256 = "583dc1922fa93753e22d2dddfbe8cd7e77de09c23fe374f31dd72e88f5346022"
SOURCE_SHA256 = "12f6a2e9e7f3e36dfa11ba30e754caeb2b49326efff1045ed482ca90d3ee7d96"
CANARY_ID = "v4-policy-canary-48f0fa514f729a77242d"
MANIFEST_SHA256 = "5a089dc58deb6f8d329dd7dfca44435f221a928c101929bfb6111581853af457"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_exact_plan(*, source_dir: Path, plan_path: Path) -> dict:
    if _sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("V4-C5 r4 plan file SHA drifted")
    expected = json.loads(plan_path.read_text(encoding="utf-8"))
    actual = build(source_dir)
    if actual != expected:
        raise ValueError("V4-C5 r4 rebuilt plan differs from the sealed plan")
    if (
        actual["canary_id"] != CANARY_ID
        or actual["source_tree_sha256"] != SOURCE_SHA256
        or actual["manifest"]["manifest_sha256"] != MANIFEST_SHA256
    ):
        raise ValueError("V4-C5 r4 exact identity drifted")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--confirm-submit", action="store_true")
    args = parser.parse_args(argv)
    plan = validate_exact_plan(
        source_dir=args.source_dir.resolve(strict=True),
        plan_path=args.plan.resolve(strict=True),
    )
    print(
        json.dumps(
            {
                "canary_id": plan["canary_id"],
                "manifest_sha256": plan["manifest"]["manifest_sha256"],
                "prepared_source_tree_sha256": plan["source_tree_sha256"],
                "scheduling_policy": "user_authorized_backlog",
                "cloud_action": False,
            },
            sort_keys=True,
            indent=2,
        )
    )
    if not args.confirm_submit:
        print("DRY RUN ONLY — no cloud write")
        return 0
    admission = collect_canary_backlog_admission(
        canary_id=plan["canary_id"],
        job_name=plan["job_name"],
        namespace_s3=plan["manifest"]["publication"]["namespace_s3"],
    )
    print(json.dumps(admission, sort_keys=True, indent=2))
    result = launch.submit_training_job(
        entry=ENTRY,
        source_dir=args.source_dir.resolve(),
        environment=plan["environment"],
        image_uri=launch.IMAGE,
        instance_type="ml.p5.48xlarge",
        volume_size=VOLUME_SIZE_GB,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.kind", "Value": "v4-policy-training-canary"},
            {"Key": "wsm.canary_id", "Value": plan["canary_id"]},
        ],
        retry_config={"attempts": 1},
        job_name=plan["job_name"],
        queue=launch.QUEUE,
        role=launch.ROLE_ARN,
        priority=launch.PRIORITY,
        max_run_seconds=MAX_RUN_SECONDS,
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_tree_sha256"],
        staged_source_files=plan["staged_source_files"],
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
