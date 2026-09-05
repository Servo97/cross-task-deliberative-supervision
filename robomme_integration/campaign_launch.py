#!/usr/bin/env python3
"""Approval-gated SageMaker launcher for one sealed RoboMME training campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from robomme_integration import launch
from robomme_integration.campaign_plan import _load_spec, build_campaign_plan
from scripts.launch.launch_guardrails import OWNER_EMAIL, PROJECT_TAG


def collect_and_validate(manifest: dict, *, job_name: str) -> dict:
    """Dispatch to the hardware-specific live gate; kept injectable for no-submit tests."""
    hardware_name = manifest["infrastructure"]["hardware"]
    if hardware_name == "p5":
        from robomme_integration.cloud_admission_p5 import collect_and_validate as collect
    elif hardware_name == "p5e":
        from robomme_integration.cloud_admission import collect_and_validate as collect
    else:
        raise SystemExit(f"unsupported campaign hardware {hardware_name!r}")
    return collect(manifest, job_name=job_name)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--spec", type=Path, required=True)
    value.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    value.add_argument("--confirm-submit", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    spec = _load_spec(args.spec)
    plan = build_campaign_plan(spec, args.source_dir)
    manifest = plan["manifest"]
    infrastructure = manifest["infrastructure"]
    summary = {
        "campaign_id": plan["campaign_id"],
        "attempt_id": plan["attempt_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cells": [
            {
                "task": cell["task"],
                "arm": cell["arm"],
                "run_id": cell["run_id"],
                "completion_claim_s3": cell["completion_claim_s3"],
            }
            for cell in manifest["cells"]
        ],
        "infrastructure": infrastructure,
        "evaluation": manifest["evaluation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.confirm_submit:
        print("DRY RUN ONLY — pass --confirm-submit only after explicit user approval")
        return

    hardware_name = infrastructure["hardware"]
    hardware = launch.HARDWARE[hardware_name]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%m%d-%H%M%S")
    job_name = f"sarvesh-rmme-series-{hardware_name}-{plan['campaign_id'][-12:]}-{stamp}"[:63]
    if not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", job_name):
        raise SystemExit(f"invalid campaign SageMaker job name {job_name}")
    # This is deliberately the final operation before the guarded SDK submit. It performs no
    # writes and rejects plan drift, capacity ambiguity, active duplicates, and any pre-existing
    # create-once namespace across the campaign and every scientific cell.
    admission = collect_and_validate(manifest, job_name=job_name)
    print(json.dumps(admission, indent=2, sort_keys=True))
    result = launch.submit_training_job(
        entry=plan["entry"],
        source_dir=args.source_dir.resolve(),
        environment=plan["environment"],
        image_uri=launch.IMAGE,
        instance_type=hardware["instance_type"],
        volume_size=infrastructure["volume_size_gb"],
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.study", "Value": launch.STUDY},
            {"Key": "wsm.benchmark", "Value": "RoboMME"},
            {"Key": "wsm.scope", "Value": "single-task-series"},
            {"Key": "wsm.campaign_id", "Value": plan["campaign_id"]},
        ],
        retry_config=launch.RETRY,
        job_name=job_name,
        queue=hardware["queue"],
        role=launch.ROLE_ARN,
        priority=infrastructure["priority"],
        max_run_seconds=infrastructure["max_run_seconds"],
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_sha"],
        staged_source_files=plan["staged_source_files"],
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
