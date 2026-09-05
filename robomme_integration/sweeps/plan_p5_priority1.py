#!/usr/bin/env python3
"""Expand, validate, dry-run, or submit the approved RoboMME p5 backfill sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from robomme_integration import launch  # noqa: E402
from robomme_integration.training.arms import WORKSPACE_ARMS  # noqa: E402

DEFAULT_SPEC = Path(__file__).with_name("p5_priority1_single_task_v1.json")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise SystemExit("unsupported sweep schema")
    if (value.get("hardware"), value.get("priority")) != ("p5", 1):
        raise SystemExit("this sweep is pinned to p5 priority 1")
    return value


def expand(spec: dict, phase: str = "all") -> list[dict]:
    completed = {(record["task"], record["arm"]) for record in spec["core"]["completed_elsewhere"]}
    core_jobs = [
        {"task": task, "arm": arm}
        for task in spec["core"]["tasks"]
        for arm in spec["core"]["arms"]
        if (task, arm) not in completed
    ]
    workspace_jobs = []
    for task, artifact in spec["workspace"]["tasks"].items():
        encoder = artifact["encoder_id"]
        root = f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/{task}/{encoder}"
        for arm in spec["workspace"]["arms"]:
            if arm not in WORKSPACE_ARMS:
                raise SystemExit(f"non-workspace arm {arm} in workspace sweep")
            record = {
                "task": task,
                "arm": arm,
                "workspace_encoder_id": encoder,
                "workspace_s3": f"{root}/omega",
                "workspace_manifest_sha256": artifact["omega_manifest_file_sha256"],
            }
            if arm == "salient":
                record.update(
                    {
                        "supervision_s3": f"{root}/supervision",
                        "supervision_manifest_sha256": artifact["supervision_manifest_file_sha256"],
                    }
                )
            workspace_jobs.append(record)
    jobs = {
        "all": [*core_jobs, *workspace_jobs],
        "core": core_jobs,
        "workspace": workspace_jobs,
    }[phase]
    identities = [(record["task"], record["arm"]) for record in jobs]
    if len(identities) != len(set(identities)):
        raise SystemExit("sweep contains duplicate task/arm cells")
    expected = {
        "all": spec["expected_new_jobs"],
        "core": spec["expected_new_jobs"] - len(workspace_jobs),
        "workspace": len(workspace_jobs),
    }[phase]
    if len(jobs) != expected:
        raise SystemExit(f"expected {expected} {phase} jobs, expanded {len(jobs)}")
    return jobs


def _arguments(spec: dict, job: dict, *, dry_run: bool) -> list[str]:
    values = [
        "--task",
        job["task"],
        "--arm",
        job["arm"],
        "--hardware",
        spec["hardware"],
        "--priority",
        str(spec["priority"]),
        "--max-run-seconds",
        str(spec["max_run_seconds"]),
        "--volume-size-gb",
        str(spec["volume_size_gb"]),
    ]
    for key, flag in (
        ("workspace_encoder_id", "--workspace-encoder-id"),
        ("workspace_s3", "--workspace-s3"),
        ("workspace_manifest_sha256", "--workspace-manifest-sha256"),
        ("supervision_s3", "--supervision-s3"),
        ("supervision_manifest_sha256", "--supervision-manifest-sha256"),
    ):
        if key in job:
            values.extend([flag, job[key]])
    values.append("--dry-run" if dry_run else "--confirm-submit")
    return values


def _plan(spec: dict, job: dict) -> dict:
    parsed = launch.parser().parse_args(_arguments(spec, job, dry_run=True))
    parsed.queue = launch.HARDWARE[parsed.hardware]["queue"]
    plan = launch.build_plan(parsed, ROOT)
    return {
        "task": job["task"],
        "arm": job["arm"],
        "run_id": plan["run_id"],
        "attempt_id": plan["attempt_id"],
        "manifest_sha256": plan["manifest"]["manifest_sha256"],
        "output": plan["output"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("all", "core", "workspace"), default="all")
    parser.add_argument("--confirm-submit", action="store_true")
    args = parser.parse_args()
    spec = _load(args.spec)
    jobs = expand(spec, args.phase)
    plans = [_plan(spec, job) for job in jobs]
    payload = {
        "schema_version": 1,
        "sweep": spec["name"],
        "phase": args.phase,
        "job_count": len(plans),
        "jobs": plans,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not args.confirm_submit:
        print("DRY MANIFEST ONLY — pass --confirm-submit only after explicit user approval")
        return
    for index, job in enumerate(jobs, 1):
        print(
            f"SUBMIT {index}/{len(jobs)} task={job['task']} arm={job['arm']}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "launch.py"), *_arguments(spec, job, dry_run=False)],
            check=True,
        )


if __name__ == "__main__":
    main()
