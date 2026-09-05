#!/usr/bin/env python3
"""Build one content-addressable all-16 workspace index from immutable producer claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .single_task import TASK_ORDER, task_manifest_sha256
from .workspace_index import load_workspace_index


def _read(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be a dictionary: {path}")
    return value


def build_index(pair_manifests: list[dict], task_claims: list[dict]) -> dict:
    """Cross-bind eight submitted pair manifests to exactly sixteen completed task claims."""
    if len(pair_manifests) != 8 or len(task_claims) != len(TASK_ORDER):
        raise ValueError("workspace index requires exactly eight pair manifests and sixteen task claims")

    expected: dict[str, dict] = {}
    pair_ids: set[str] = set()
    for manifest in pair_manifests:
        if manifest.get("kind") != "robomme_all16_workspace_pair_attempt":
            raise ValueError("unsupported workspace pair manifest")
        pair_id = manifest.get("pair_id")
        source_sha = manifest.get("source_tree_sha256")
        if not isinstance(pair_id, str) or pair_id in pair_ids:
            raise ValueError(f"invalid or duplicate pair identity: {pair_id!r}")
        pair_ids.add(pair_id)
        infrastructure = manifest.get("infrastructure", {})
        instance_type = infrastructure.get("instance_type")
        if instance_type not in {"ml.p5.48xlarge", "ml.p5e.48xlarge"}:
            raise ValueError(f"unsupported workspace producer instance: {instance_type!r}")
        records = manifest.get("tasks")
        if not isinstance(records, list) or len(records) != 2:
            raise ValueError(f"pair {pair_id} must bind exactly two tasks")
        for record in records:
            task = record.get("task")
            scientific = record.get("scientific", {})
            producer = scientific.get("producer", {})
            if task not in TASK_ORDER or task in expected:
                raise ValueError(f"invalid or multiply routed workspace task: {task!r}")
            if record.get("task_inventory_sha256") != scientific.get("task_inventory_sha256"):
                raise ValueError(f"task inventory drift in pair manifest for {task}")
            if producer.get("devices") != 4 or producer.get("hardware") not in {"p5", "p5e"}:
                raise ValueError(f"invalid four-device producer topology for {task}")
            expected[task] = {
                "pair_id": pair_id,
                "run_id": record.get("run_id"),
                "claim_s3": record.get("claim_s3"),
                "source_tree_sha256": source_sha,
                "instance_type": instance_type,
                "hardware": producer["hardware"],
                "accelerator": producer.get("accelerator"),
            }
    if tuple(task for task in TASK_ORDER if task in expected) != TASK_ORDER:
        raise ValueError("pair manifests do not cover the canonical all-16 task order")

    claims: dict[str, dict] = {}
    for claim in task_claims:
        task = claim.get("task")
        if task not in expected or task in claims:
            raise ValueError(f"invalid or duplicate task completion claim: {task!r}")
        identity = expected[task]
        for field in ("pair_id", "run_id", "source_tree_sha256"):
            if claim.get(field) != identity[field]:
                raise ValueError(f"task claim {task}.{field} disagrees with its submitted manifest")
        if (
            claim.get("kind") != "robomme_all16_workspace_task_complete"
            or claim.get("campaign") != "uniform_gpu_v1"
            or claim.get("task_manifest_sha256") != task_manifest_sha256(task)
        ):
            raise ValueError(f"invalid workspace completion contract for {task}")
        claims[task] = claim
    if tuple(task for task in TASK_ORDER if task in claims) != TASK_ORDER:
        raise ValueError("completion claims do not cover the canonical all-16 task order")

    tasks = {}
    for task in TASK_ORDER:
        claim = claims[task]
        identity = expected[task]
        tasks[task] = {
            "task_manifest_sha256": task_manifest_sha256(task),
            "encoder_id": claim["encoder_id"],
            "omega": claim["omega"],
            "supervision": claim["supervision"],
            "representation": claim["representation"],
            "producer": {
                "campaign": "uniform_gpu_v1",
                "pair_id": identity["pair_id"],
                "run_id": identity["run_id"],
                "source_tree_sha256": identity["source_tree_sha256"],
                "hardware": identity["hardware"],
                "instance_type": identity["instance_type"],
                "accelerator": identity["accelerator"],
            },
        }
    accelerators = sorted({record["producer"]["accelerator"] for record in tasks.values()})
    return {
        "schema_version": 1,
        "benchmark": "RoboMME",
        "scope": "all16",
        "campaign": "uniform_gpu_v1",
        "task_order": list(TASK_ORDER),
        "causal_contract": {
            "source_frames": "at_or_before_step_idx",
            "uses_future_execution_frames": False,
            "video_prefix": "benchmark_provided_only",
        },
        "producer_topology": {
            "node_accelerators": accelerators,
            "mixed_accelerators": len(accelerators) > 1,
            "devices_per_task": 4,
        },
        "tasks": tasks,
    }


def serialize(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=False) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", action="append", required=True)
    parser.add_argument("--task-claim", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_index(
        [_read(path) for path in args.pair_manifest],
        [_read(path) for path in args.task_claim],
    )
    payload = serialize(value)
    digest = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    load_workspace_index(args.output, expected_sha256=digest, require_supervision=True)
    print(digest)


if __name__ == "__main__":
    main()
