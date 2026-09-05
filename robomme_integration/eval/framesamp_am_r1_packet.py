"""Build the immutable, no-submit FS-R1 paired oracle-screen packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_oracle_screen_packet"
TASKS = ("VideoPlaceButton", "RouteStick")
BUDGETS = (256, 128, 64)
EPISODES = tuple(range(8))
SOURCE_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
POLICY_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
OVERLAY_MANIFEST_SHA256 = "5344b2a6898fe36cfec2565c2c2bcbab894244cfb30744a357c8585b33b8a85f"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _demo_identity(path: Path, task: str) -> dict[str, object]:
    audit = _load(path)
    if (
        audit.get("kind") != "robomme_test_demo_prefix_audit"
        or audit.get("task") != task
        or audit.get("episode_map")[:8] != list(EPISODES)
        or audit.get("summary", {}).get("all_prefixes_nonempty") is not True
    ):
        raise ValueError(f"invalid FS-D0 audit for {task}")
    records = audit.get("records")
    if not isinstance(records, list) or [row.get("episode") for row in records[:8]] != list(EPISODES):
        raise ValueError(f"invalid first-eight record map for {task}")
    return {
        "audit_file_sha256": _sha256_file(path),
        "audit_content_sha256": audit["content_sha256"],
        "test_metadata_sha256": audit["test_metadata_sha256"],
        "episodes": [
            {
                "episode": row["episode"],
                "seed": row["seed"],
                "difficulty": row["difficulty"],
                "demo_frames": row["front_frames"],
            }
            for row in records[:8]
        ],
    }


def _anchor_identity(
    progress_path: Path,
    run_manifest_path: Path,
    scorecard_path: Path,
) -> dict[str, object]:
    progress = _load(progress_path)
    run_manifest = _load(run_manifest_path)
    scorecard = _load(scorecard_path)
    if (
        run_manifest.get("method") != "perceptual-framesamp-modul"
        or run_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or run_manifest.get("policy_source_commit") != POLICY_COMMIT
        or run_manifest.get("action_horizon") != 20
        or run_manifest.get("execution_horizon") != 16
        or run_manifest.get("max_steps") != 1300
        or scorecard.get("successes") != 368
        or scorecard.get("episodes") != 800
    ):
        raise ValueError("released FrameSamp anchor identity mismatch")
    outcomes: dict[str, dict[str, bool]] = {}
    for task in TASKS:
        task_progress = progress.get(task)
        if not isinstance(task_progress, dict):
            raise ValueError(f"released anchor lacks task {task}")
        selected = {str(episode): task_progress[str(episode)] for episode in EPISODES}
        if not all(isinstance(value, bool) for value in selected.values()):
            raise ValueError(f"released anchor outcomes are not Boolean for {task}")
        outcomes[task] = selected
    return {
        "reuse_reason": "same released checkpoint/source/h20/e16/max1300 and exact episode indices",
        "progress_sha256": _sha256_file(progress_path),
        "run_manifest_sha256": _sha256_file(run_manifest_path),
        "scorecard_sha256": _sha256_file(scorecard_path),
        "fixed800_successes": 368,
        "fixed800_episodes": 800,
        "first8_outcomes": outcomes,
        "first8_successes": {task: sum(outcomes[task].values()) for task in TASKS},
    }


def _cell(task: str, fit_mass: bool, budget: int, sources: dict[str, str]) -> dict[str, object]:
    method = "output_mass" if fit_mass else "output_only"
    identity = {
        "task": task,
        "episodes": list(EPISODES),
        "method": method,
        "fit_mass": fit_mass,
        "budget": budget,
        "storage_dtype": "bfloat16",
        "bfloat16_quantization_parity_max_metric_increase": 0.05,
        "mass_ridge": 0.0,
        "value_ridge": 0.0,
        "action_horizon": 20,
        "execution_horizon": 16,
        "max_steps": 1300,
        "model_seed": 7,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "policy_commit": POLICY_COMMIT,
        "overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
        "source_files": sources,
    }
    identity_sha = hashlib.sha256(_canonical(identity)).hexdigest()
    return {
        "cell_id": f"fs-r1-{task.lower()}-{method}-m{budget}-{identity_sha[:16]}",
        "identity": identity,
        "identity_sha256": identity_sha,
        "state": "BLOCKED_PENDING_EXECUTOR_CANARY_AND_P5_ADMISSION",
    }


def build(
    *,
    demo_audits: dict[str, Path],
    anchor_progress: Path,
    anchor_run_manifest: Path,
    anchor_scorecard: Path,
    r0_receipt: Path,
) -> dict[str, object]:
    r0 = _load(r0_receipt)
    if (
        r0.get("status") != "HARD_GREEN"
        or r0.get("kind") != "robomme_framesamp_am_r0_real_later_cut_transition"
        or r0.get("stale_history_rejected") is not True
        or r0.get("ephemeral_artifacts_deleted_before_receipt") is not True
    ):
        raise ValueError("FS-R0 receipt does not unlock FS-R1")
    source_paths = {
        "r0_transition": SOURCE_ROOT / "training/framesamp_am_r0_transition.py",
        "sim_worker": SOURCE_ROOT / "eval/framesamp_r0_sim_worker.py",
        "multi_replan_sim_worker": SOURCE_ROOT / "eval/framesamp_r1_sim_worker.py",
        "per_cut_rollout": SOURCE_ROOT / "eval/framesamp_am_r1_rollout.py",
        "h100_canary": SOURCE_ROOT / "eval/framesamp_am_r1_canary.py",
        "result_publisher": SOURCE_ROOT / "eval/framesamp_am_r1_publish.py",
        "cloud_contract": SOURCE_ROOT / "eval/framesamp_am_r1_cloud.py",
        "gpu_entry": SOURCE_ROOT / "gpu_framesamp_am_r1_entry.sh",
        "cloud_launcher": SOURCE_ROOT / "eval/framesamp_am_r1_launch.py",
        "oracle_server": SOURCE_ROOT / "eval/framesamp_am_oracle_server.py",
        "teacher_producer": SOURCE_ROOT / "training/framesamp_am_teacher_producer.py",
        "query_bank": SOURCE_ROOT / "training/framesamp_am_query_bank.py",
        "artifact": SOURCE_ROOT / "training/framesamp_am_artifact.py",
        "oracle_route": SOURCE_ROOT / "training/framesamp_am_oracle_route.py",
    }
    sources = {name: _sha256_file(path) for name, path in source_paths.items()}
    anchor = _anchor_identity(anchor_progress, anchor_run_manifest, anchor_scorecard)
    cells = [
        _cell(task, fit_mass, budget, sources) for task in TASKS for fit_mass in (False, True) for budget in BUDGETS
    ]
    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scope": "paired_8_episode_screen_then_task_local_fixed50_promotion",
        "cloud_action": False,
        "this_packet_authorizes_submission": False,
        "tasks": list(TASKS),
        "episode_indices": list(EPISODES),
        "demo_audits": {task: _demo_identity(demo_audits[task], task) for task in TASKS},
        "released_anchor": anchor,
        "r0_gate": {
            "receipt_file_sha256": _sha256_file(r0_receipt),
            "receipt_sha256": r0["receipt_sha256"],
            "causal_cut_step": r0["causal_cut_step"],
            "requested_budget": r0["requested_budget"],
        },
        "oracle_cells": cells,
        "promotion": {
            "unit": "task_local_arm",
            "required_valid_episodes": 8,
            "allowed_harness_failures": 0,
            "required_fresh_attested_stack_fraction": 1.0,
            "persistent_oracle_payloads_allowed": False,
            "minimum_successes": {task: max(0, int(anchor["first8_successes"][task]) - 2) for task in TASKS},
            "interpretation": (
                "infrastructure/large-regression screen only; n=8 is not statistical evidence; "
                "every passing task-local row advances to the paired fixed50"
            ),
        },
        "p5_execution_plan": {
            "queue": "fss-tri-cam-robotics-p5-48xlarge-us-west-2",
            "instance_type": "ml.p5.48xlarge",
            "topology": "one node; eight H100 lanes map episodes 0..7; 12 oracle cells clumped sequentially",
            "service_jobs": 1,
            "scientific_cells": len(cells),
            "admission": (
                "no duplicate, empty namespace, executor canary HARD_GREEN; backlog allowed only "
                "under the explicit user scheduling override"
            ),
        },
        "executor": {
            "state": "IMPLEMENTED_PENDING_H100_CANARY",
            "missing": [],
            "runtime_gate": "exact 8xH100 canary receipt HARD_GREEN",
            "policy_runtime": {
                "source": "upstream_uv_lock",
                "jax": "0.5.3",
                "orbax_checkpoint": "0.11.13",
            },
            "canary_lanes": [
                {"lane": 0, "task": "VideoPlaceButton", "episode": 0, "budget": 256, "fit_mass": False},
                {"lane": 1, "task": "VideoPlaceButton", "episode": 1, "budget": 128, "fit_mass": False},
                {"lane": 2, "task": "VideoPlaceButton", "episode": 2, "budget": 64, "fit_mass": False},
                {"lane": 3, "task": "VideoPlaceButton", "episode": 3, "budget": 256, "fit_mass": True},
                {"lane": 4, "task": "VideoPlaceButton", "episode": 4, "budget": 128, "fit_mass": True},
                {"lane": 5, "task": "VideoPlaceButton", "episode": 5, "budget": 64, "fit_mass": True},
                {"lane": 6, "task": "RouteStick", "episode": 0, "budget": 64, "fit_mass": False},
                {"lane": 7, "task": "RouteStick", "episode": 1, "budget": 64, "fit_mass": True},
            ],
            "publication": "validated local claim plus byte-identical S3 If-None-Match create-once",
        },
    }
    packet["packet_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    return packet


def validate(packet: dict[str, object]) -> None:
    seal = packet.get("packet_sha256")
    unsigned = dict(packet)
    unsigned.pop("packet_sha256", None)
    if seal != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("FS-R1 packet seal mismatch")
    cells = packet.get("oracle_cells")
    if not isinstance(cells, list) or len(cells) != 12:
        raise ValueError("FS-R1 packet must contain 12 oracle cells")
    tuples = {(cell["identity"]["task"], cell["identity"]["fit_mass"], cell["identity"]["budget"]) for cell in cells}
    expected = {(task, fit_mass, budget) for task in TASKS for fit_mass in (False, True) for budget in BUDGETS}
    if tuples != expected or len({cell["cell_id"] for cell in cells}) != 12:
        raise ValueError("FS-R1 cell matrix is incomplete or duplicated")
    if packet.get("this_packet_authorizes_submission") is not False:
        raise ValueError("pre-executor FS-R1 packet must not authorize submission")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-demo-audit", type=Path, required=True)
    parser.add_argument("--route-demo-audit", type=Path, required=True)
    parser.add_argument("--anchor-progress", type=Path, required=True)
    parser.add_argument("--anchor-run-manifest", type=Path, required=True)
    parser.add_argument("--anchor-scorecard", type=Path, required=True)
    parser.add_argument("--r0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = build(
        demo_audits={"VideoPlaceButton": args.video_demo_audit, "RouteStick": args.route_demo_audit},
        anchor_progress=args.anchor_progress,
        anchor_run_manifest=args.anchor_run_manifest,
        anchor_scorecard=args.anchor_scorecard,
        r0_receipt=args.r0_receipt,
    )
    validate(packet)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    with args.output.open("xb") as stream:
        stream.write(json.dumps(packet, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"packet_sha256": packet["packet_sha256"], "cells": 12, "cloud_action": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
