#!/usr/bin/env python3
"""Build (but never submit) one sealed RoboMME single-task GPU campaign bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from robomme_integration import launch  # noqa: E402
from robomme_integration.campaign import validate_manifest  # noqa: E402

ENTRY = "gpu_campaign_entry.sh"
STAGED_MANIFEST = "_robomme_gpu_campaign_manifest.json"
CELL_MANIFEST_ROOT = "_robomme_gpu_campaign_cells"
DEFAULT_MINIMUM_FREE_BYTES = 64 * 1024**3
DEFAULT_RESERVE_SECONDS = 1_800
DEFAULT_CELL_SECONDS = 4 * 3_600
CELL_OPTION_FLAGS = {
    "workspace_encoder_id": "--workspace-encoder-id",
    "workspace_s3": "--workspace-s3",
    "workspace_manifest_sha256": "--workspace-manifest-sha256",
    "supervision_s3": "--supervision-s3",
    "supervision_manifest_sha256": "--supervision-manifest-sha256",
}


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seal(value: dict) -> tuple[dict, str]:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    digest = hashlib.sha256(_canonical(clean).encode()).hexdigest()
    clean["manifest_sha256"] = digest
    return clean, _canonical(clean) + "\n"


def _load_spec(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported campaign planning spec")
    return value


def _cell_arguments(spec: dict, cell: dict) -> list[str]:
    values = [
        "--scope",
        "single_task",
        "--task",
        cell["task"],
        "--arm",
        cell["arm"],
        "--hardware",
        spec["hardware"],
        "--priority",
        str(spec["priority"]),
        "--max-run-seconds",
        str(spec["max_run_seconds"]),
        "--volume-size-gb",
        str(spec["volume_size_gb"]),
        "--attempt-index",
        str(spec.get("attempt_index", 1)),
        "--dry-run",
    ]
    for key, flag in CELL_OPTION_FLAGS.items():
        if key in cell:
            values.extend([flag, str(cell[key])])
    return values


def build_campaign_plan(spec: dict, source_dir: Path) -> dict:
    """Expand exact cell plans and return files/environment for one future submission."""
    source_dir = source_dir.resolve()
    if not (source_dir / ENTRY).is_file() or not (source_dir / launch.ENTRY).is_file():
        raise ValueError("source directory lacks campaign/training entries")
    required = {"name", "hardware", "priority", "max_run_seconds", "volume_size_gb", "cells"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"campaign spec lacks {sorted(missing)}")
    if spec["hardware"] not in launch.HARDWARE:
        raise ValueError("campaign hardware must be p5 or p5e")
    if spec["priority"] != launch.PRIORITY:
        raise ValueError("two-day RoboMME campaigns are pinned to priority 400")
    if not 1 <= spec["max_run_seconds"] <= launch.MAX_RUN_SECONDS:
        raise ValueError("campaign max_run_seconds must lie in [1, 86400]")
    if not launch.VOLUME_MIN_GB <= spec["volume_size_gb"] <= launch.VOLUME_MAX_GB:
        raise ValueError("campaign volume is outside the bounded RoboMME range")
    attempt_index = spec.get("attempt_index", 1)
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index < 1:
        raise ValueError("campaign attempt_index must be positive")
    failure_policy = spec.get("failure_policy", "stop")
    if failure_policy not in {"stop", "continue"}:
        raise ValueError("campaign failure_policy must be stop or continue")
    if spec.get("evaluation_mode", "deferred") != "deferred":
        raise ValueError(
            "inline evaluation is blocked until the p5 native-render claim and runtime are wired; p5e is training-only"
        )
    cells = spec["cells"]
    if not isinstance(cells, list) or not cells:
        raise ValueError("campaign spec has no cells")
    identities = [(cell.get("task"), cell.get("arm")) for cell in cells if isinstance(cell, dict)]
    if len(identities) != len(cells) or len(identities) != len(set(identities)):
        raise ValueError("campaign cells must be unique task/arm objects")

    staged: dict[str, str] = {}
    records = []
    source_shas = set()
    for ordinal, cell in enumerate(cells):
        parsed = launch.parser().parse_args(_cell_arguments(spec, cell))
        parsed.queue = launch.HARDWARE[parsed.hardware]["queue"]
        plan = launch.build_plan(parsed, source_dir)
        source_shas.add(plan["source_sha"])
        staged_name = f"{CELL_MANIFEST_ROOT}/{ordinal:03d}-{cell['task'].lower()}-{cell['arm']}.json"
        environment = dict(plan["environment"])
        environment["RUN_MANIFEST_SOURCE"] = staged_name
        estimated = cell.get("estimated_train_seconds", spec.get("estimated_train_seconds", DEFAULT_CELL_SECONDS))
        if not isinstance(estimated, int) or isinstance(estimated, bool) or estimated < 1:
            raise ValueError(f"invalid estimated_train_seconds for {cell['task']}/{cell['arm']}")
        records.append(
            {
                "ordinal": ordinal,
                "cell_id": f"{ordinal:03d}-{cell['task'].lower()}-{cell['arm']}",
                "task": cell["task"],
                "arm": cell["arm"],
                "run_id": plan["run_id"],
                "attempt_id": plan["attempt_id"],
                "scientific_spec_sha256": plan["manifest"]["scientific_spec_sha256"],
                "run_manifest_source": staged_name,
                "run_manifest_sha256": plan["manifest"]["manifest_sha256"],
                "final_step": 19_999,
                "output_s3": plan["output"],
                "completion_claim_s3": plan["manifest"]["claims"]["completion"],
                "estimated_train_seconds": estimated,
                "environment": environment,
            }
        )
        staged[staged_name] = plan["manifest_json"] + "\n"
    if len(source_shas) != 1:
        raise AssertionError(f"campaign cells resolved different source trees: {source_shas}")
    run_protocols = {"v4" if record["run_id"].startswith("st-v4-") else "v1" for record in records}
    if len(run_protocols) != 1:
        raise ValueError("one serial campaign may not mix legacy-v1 and v4 scientific protocols")
    protocol = run_protocols.pop()

    reserve = spec.get("runtime_reserve_seconds", DEFAULT_RESERVE_SECONDS)
    minimum_free = spec.get("minimum_free_bytes", DEFAULT_MINIMUM_FREE_BYTES)
    evaluation = {
        "mode": "deferred",
        "required_gate": ("p5_training_only" if spec["hardware"] == "p5e" else "p5_native_render_reset_claim"),
        "reason": (
            "p5e is contractually training-only"
            if spec["hardware"] == "p5e"
            else "native simulator/runtime has not been sealed into this campaign job"
        ),
    }
    stable = {
        "schema_version": 1,
        "name": spec["name"],
        "ordered_cells": [
            {
                "run_id": record["run_id"],
                "scientific_spec_sha256": record["scientific_spec_sha256"],
            }
            for record in records
        ],
        "failure_policy": failure_policy,
        "evaluation": evaluation,
    }
    campaign_scientific_sha = hashlib.sha256(_canonical(stable).encode()).hexdigest()
    campaign_id = f"rmme-st-series-{protocol}-{campaign_scientific_sha[:20]}"
    attempt_id = f"{campaign_id}-attempt{attempt_index}"
    root = f"{launch.STUDY_ROOT}/campaigns/robomme/single_task_{protocol}/{campaign_id}"
    hardware = launch.HARDWARE[spec["hardware"]]
    manifest, manifest_json = _seal(
        {
            "schema_version": 1,
            "kind": "robomme_single_task_train_series",
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "campaign_scientific_sha256": campaign_scientific_sha,
            "infrastructure": {
                "provider": "aws_sagemaker",
                "hardware": spec["hardware"],
                "queue": hardware["queue"],
                "training_plan_arn": launch.training_plan_arn(hardware["queue"]),
                "instance_type": hardware["instance_type"],
                "accelerator": hardware["accelerator"],
                "priority": spec["priority"],
                "max_run_seconds": spec["max_run_seconds"],
                "volume_size_gb": spec["volume_size_gb"],
            },
            "runtime_reserve_seconds": reserve,
            "failure_policy": failure_policy,
            "evaluation": evaluation,
            "cleanup": {
                "remove_cell_work_after_attempt": True,
                "minimum_free_bytes": minimum_free,
                "durable_outputs": "deploy-only checkpoint plus immutable claims",
            },
            "cells": records,
            "claims": {
                "manifest": f"{root}/manifests/{attempt_id}.json",
                "attempt_result": f"{root}/claims/attempts/{attempt_id}.result.json",
                "completion": f"{root}/claims/complete.json",
            },
        }
    )
    validate_manifest(manifest)
    staged[STAGED_MANIFEST] = manifest_json
    environment = {
        "ROBOMME_CAMPAIGN_MANIFEST_SOURCE": STAGED_MANIFEST,
        "ROBOMME_CAMPAIGN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "ROBOMME_CAMPAIGN_ID": campaign_id,
        "ROBOMME_CAMPAIGN_ATTEMPT_ID": attempt_id,
    }
    if hardware["reserved_capacity"] is not None:
        environment["SM_USE_RESERVED_CAPACITY"] = hardware["reserved_capacity"]
    return {
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "environment": environment,
        "staged_source_files": staged,
        "source_sha": source_shas.pop(),
        "entry": ENTRY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    plan = build_campaign_plan(_load_spec(args.spec), args.source_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty campaign output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for relative, payload in plan["staged_source_files"].items():
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "campaign_id": plan["campaign_id"],
        "attempt_id": plan["attempt_id"],
        "manifest_sha256": plan["manifest"]["manifest_sha256"],
        "cells": [
            {key: cell[key] for key in ("cell_id", "task", "arm", "run_id", "completion_claim_s3")}
            for cell in plan["manifest"]["cells"]
        ],
        "evaluation": plan["manifest"]["evaluation"],
        "cloud_action": False,
    }
    (args.output_dir / "PLAN.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PLAN ONLY — no AWS SDK loaded and no cloud write performed")


if __name__ == "__main__":
    main()
