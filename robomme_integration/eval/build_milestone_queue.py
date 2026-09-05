#!/usr/bin/env python3
"""Build the A19 checkpoint-maturity eval queues: one v4_70k checkpoint x retained milestones x 16 tasks.

Two commands, both CPU-only and cloud-read-only:

``template``
    Writes an UNRESOLVED queue for one sweep arm.  Every field that depends on the training run
    (run_id, digests, URIs, the deployed milestone set) is a ``<PLACEHOLDER>`` string, so
    ``campaign.validate_queue`` fails closed until ``fill`` has run.  Cells are ordered milestone-
    descending (the final step first), then in canonical task order; each cell is one task's 50
    fixed test indices against ``deploy/<milestone>`` of the run.

``fill``
    Resolves a template from the run's sealed training manifest and its milestone completion claim
    (local files, or S3 objects with ``--confirm-read-s3``), verifying the receipt chain first: the
    manifest self-seal and scientific digest, arm/scope/steps/recipe, the claim's kind, run/manifest
    binding, and that its ``steps`` equal the recipe's deploy set and address ``deploy/<step>``.

The result is a queue TEMPLATE for ``launch_p5_campaign --parallel-fixed50``, which attaches the
8-lane p5 topology and the preflight/runtime gates and seals the queue identity.  The node-side
stager re-verifies every receipt against S3 before a single episode runs.  Nothing here submits,
writes to S3, or touches a GPU.

Protocol universe (recorded in every queue as ``comparability.eval_protocol``): these lanes run the
execute-10 fixed-50 evaluator (predict 20, execute 10, model_seed 7, max 1,300 steps).  They are not
the paper protocol (h20/e16, ``robomme-paper856-h20-e16-fixed50-project-v1``), which exists only in
the local ``project_exact`` runner; the two ledgers are never pooled (CAMPAIGNS.md W4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from robomme_integration.eval import campaign
from robomme_integration.eval.build_existing_pick_button_queue import AwsReadStore, _nuisance
from robomme_integration.launch import (
    CANONICAL_PARENT_SHA256,
    INIT_INVENTORY_SHA,
    INIT_ROOT,
    STUDY_ROOT,
    V4_70K_RECIPE,
    V4_70K_TRAIN_STEPS,
    v4_70k_milestones,
)
from robomme_integration.training.single_task import TASK_ORDER

RUN_MANIFEST_ROOT = f"{STUDY_ROOT}/manifests/runs/train"
TRAIN_CLAIM_ROOT = f"{STUDY_ROOT}/manifests/claims/train"
CHECKPOINT_ROOT = f"{STUDY_ROOT}/checkpoints/robomme/pi05/multitask_v4/all16"
PUBLISH_ROOT = f"{STUDY_ROOT}/evaluations/fixed50_campaigns"
DEFAULT_SWEEP = Path(__file__).resolve().parent / "milestone_queues" / "a19_sweep.json"

RUN_ID = "<RUN_ID>"
ATTEMPT_ID = "<ATTEMPT_ID>"
SCIENTIFIC_SHA = "<SCIENTIFIC_SPEC_SHA256>"
MANIFEST_SHA = "<RUN_MANIFEST_SHA256>"
OPENPI_URI = "<OPENPI_URI>"
OPENPI_SHA = "<OPENPI_SHA256>"
WORKSPACE_UNRESOLVED = "<WORKSPACE_SERVING_UNRESOLVED>"
PLACEHOLDER = re.compile(r"<[A-Z0-9_:]+>")

FINAL_STEP = V4_70K_TRAIN_STEPS - 1
DEPLOY_MILESTONES = [*v4_70k_milestones(V4_70K_TRAIN_STEPS), FINAL_STEP]
TRAINING_NUISANCE = {
    "data_parent_inventory_sha256": CANONICAL_PARENT_SHA256,
    "data_task_inventory_sha256": None,
    "initialization_inventory_sha256": INIT_INVENTORY_SHA,
    "initialization_checkpoint_s3": INIT_ROOT,
    "seed": 0,
    "steps": V4_70K_TRAIN_STEPS,
    "action_horizon": 20,
    "window_len": None,
    "chunk_stride": None,
}
# The sealed p5 fixed-50 limits (p5_standard_wave1_fixed50_parallel_v1): a 21 h queue inside the
# 24 h job, 30 min reserve, the pre-registered 2 h admission budget per 50-episode cell (the true p5
# rate is unmeasured; wave 1 measures it), 128 GiB free-disk floor (= the p5 topology floor).
LIMITS = {
    "max_run_seconds": 75_600,
    "runtime_reserve_seconds": 1_800,
    "estimated_cell_seconds": 7_200,
    "minimum_free_bytes": 128 * 1024**3,
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sweep(path: Path) -> dict:
    sweep = json.loads(path.read_text(encoding="utf-8"))
    if sweep.get("recipe") != V4_70K_RECIPE or sweep.get("steps") != V4_70K_TRAIN_STEPS:
        raise SystemExit(f"sweep {path} is not the {V4_70K_RECIPE}/{V4_70K_TRAIN_STEPS} recipe")
    if sweep.get("deploy_milestones") != DEPLOY_MILESTONES:
        raise SystemExit(f"sweep deploy milestones must be {DEPLOY_MILESTONES}")
    for label, record in sweep["arms"].items():
        milestones = record["milestones"]
        if milestones == "all":
            record["milestones"] = list(DEPLOY_MILESTONES)
        elif not set(milestones) <= set(DEPLOY_MILESTONES) or milestones != sorted(milestones):
            raise SystemExit(f"{label}: milestones {milestones} are not a sorted subset of the deploy set")
        if not record["queue_id"].endswith("-parallel-v1"):
            raise SystemExit(f"{label}: parallel p5 queues must carry a -parallel-v1 queue_id")
    return sweep


def _config(source_root: Path, task: str) -> tuple[str, str]:
    relative = f"robomme_integration/eval/configs/{task.lower()}.yaml"
    path = source_root / relative
    if not path.is_file():
        raise SystemExit(f"missing per-task fixed-50 benchmark config {relative}")
    return relative, _sha256_file(path)


def build_template(sweep: dict, label: str, *, source_root: Path) -> dict:
    record = sweep["arms"][label]
    arm = record["arm"]
    queue_id = record["queue_id"]
    milestones = list(record["milestones"])
    publish = f"{PUBLISH_ROOT}/{queue_id}"
    configs = {task: dict(zip(("path", "sha256"), _config(source_root, task))) for task in TASK_ORDER}
    common = {key: value for key, value in TRAINING_NUISANCE.items() if key not in {"window_len", "chunk_stride"}}
    workspace_arm = arm in campaign.WORKSPACE_EVAL_ARMS or record.get("workspace") == "unresolved"
    cells = []
    for step in sorted(milestones, reverse=True):
        for task in TASK_ORDER:
            ordinal = len(cells)
            cell_id = f"{ordinal:03d}-{task.lower()}-{arm}-step{step}"
            cells.append(
                {
                    "ordinal": ordinal,
                    "cell_id": cell_id,
                    "task": task,
                    "arm": arm,
                    "run_id": RUN_ID,
                    "final_step": FINAL_STEP,
                    "checkpoint_step": step,
                    "deployed_milestones": list(DEPLOY_MILESTONES),
                    "scientific_spec_sha256": SCIENTIFIC_SHA,
                    "run_manifest_sha256": MANIFEST_SHA,
                    "training_openpi": {"uri": OPENPI_URI, "sha256": OPENPI_SHA},
                    "training_run_manifest_s3": f"{RUN_MANIFEST_ROOT}/{RUN_ID}/{ATTEMPT_ID}.json",
                    "training_output_s3": f"{CHECKPOINT_ROOT}/{arm}/seed0/{RUN_ID}",
                    "training_completion_claim_s3": (f"{TRAIN_CLAIM_ROOT}/{RUN_ID}/step-{FINAL_STEP}.complete.json"),
                    "training_completion_binding": campaign.TRAINING_COMPLETION_CURRENT,
                    "benchmark_config": configs[task]["path"],
                    "benchmark_config_sha256": configs[task]["sha256"],
                    "training_nuisance": dict(TRAINING_NUISANCE),
                    "eval_id": f"{RUN_ID}-s{step}-{task.lower()}-{queue_id}",
                    "result_claim_s3": f"{publish}/cells/{cell_id}/result.complete.json",
                    "workspace": WORKSPACE_UNRESOLVED if workspace_arm else None,
                    "ptrm": None,
                    "cfg_guidance_scale": 1.0,
                }
            )
    return {
        "schema_version": 1,
        "kind": campaign.MILESTONE_QUEUE_KIND,
        "queue_id": queue_id,
        "sweep_label": label,
        "publish_root_s3": publish,
        "claims": {"manifest": f"{publish}/manifest.json", "completion": f"{publish}/complete.json"},
        # Replaced by launch_p5_campaign --parallel-fixed50 with the sealed 8-lane p5 topology.
        "topology": {
            "policy_gpus": [0, 1, 2, 3, 4, 5, 6, 7],
            "simulator_gpus": [0, 1, 2, 3, 4, 5, 6, 7],
            "simulator_shards": 32,
            "cpu_range": "0-191",
            "base_port": 18100,
            "xla_memory_fraction": 0.65,
        },
        "retry": {"classifier_version": campaign.CLASSIFIER_VERSION, "max_attempts": 2},
        "limits": dict(LIMITS),
        "comparability": {
            "serving_openpi": {"uri": OPENPI_URI, "sha256": OPENPI_SHA},
            "task_benchmark_configs": configs,
            "task_common_training_nuisance": {task: dict(common) for task in TASK_ORDER},
            "sequence_geometry_policy": "manifest_verified_per_cell_not_assumed_common",
            "eval_protocol": dict(campaign.MILESTONE_EVAL_PROTOCOL),
        },
        "cells": cells,
    }


def unresolved_placeholders(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            found |= unresolved_placeholders(item)
    elif isinstance(value, list):
        for item in value:
            found |= unresolved_placeholders(item)
    elif isinstance(value, str):
        found |= set(PLACEHOLDER.findall(value))
    return found


def _read_document(reference: str, *, store: AwsReadStore | None, label: str) -> tuple[dict, str]:
    if reference.startswith("s3://"):
        if store is None:
            raise SystemExit(f"{label} is an S3 object; pass --confirm-read-s3 for read-only discovery")
        payload = store.read(reference)
        if payload is None:
            raise SystemExit(f"{label} is absent: {reference}")
        uri = reference
    else:
        path = Path(reference).expanduser()
        payload = path.read_bytes()
        uri = ""
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise SystemExit(f"{label} is not one JSON object")
    return value, uri


def verify_run(manifest: dict, claim: dict, *, arm: str, manifest_uri: str) -> dict:
    """Authenticate the manifest/claim pair for one v4_70k run and return the resolved bindings."""
    if manifest.get("kind") != "robomme_gpu_training_attempt" or manifest.get("schema_version") != 2:
        raise SystemExit("training manifest is not a sealed robomme_gpu_training_attempt v2")
    seal = manifest.get("manifest_sha256")
    if not isinstance(seal, str) or campaign._seal_digest(manifest, "manifest_sha256") != seal:
        raise SystemExit("training manifest self-seal mismatch")
    scientific = manifest.get("scientific")
    if not isinstance(scientific, dict):
        raise SystemExit("training manifest lacks its scientific payload")
    scientific_sha = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if scientific_sha != manifest.get("scientific_spec_sha256"):
        raise SystemExit("training manifest scientific digest mismatch")
    run_id = manifest.get("run_id")
    attempt_id = manifest.get("attempt_id")
    if (
        not isinstance(run_id, str)
        or not run_id.startswith(f"{campaign.MILESTONE_RUN_ID_PREFIX}{arm}-seed0-")
        or not isinstance(attempt_id, str)
        or not re.fullmatch(rf"{re.escape(run_id)}-attempt[1-9][0-9]*", attempt_id)
    ):
        raise SystemExit(f"manifest run/attempt identity is not a {V4_70K_RECIPE} run of {arm}")
    if manifest_uri and manifest.get("manifest_s3") != manifest_uri:
        raise SystemExit("training manifest does not name the object it was read from")
    expected_manifest_s3 = f"{RUN_MANIFEST_ROOT}/{run_id}/{attempt_id}.json"
    if manifest.get("manifest_s3") != expected_manifest_s3:
        raise SystemExit("training manifest URI is not canonical")
    training = scientific.get("training", {})
    if (
        scientific.get("arm") != arm
        or scientific.get("scope") != "multitask_v4"
        or scientific.get("task", {}).get("name") != "all16"
        or training.get("steps") != V4_70K_TRAIN_STEPS
        or training.get("recipe") != V4_70K_RECIPE
        or training.get("checkpoint_policy", {}).get("deploy_milestones") != DEPLOY_MILESTONES
    ):
        raise SystemExit(f"training manifest is not the {V4_70K_RECIPE} multitask recipe for {arm}")
    output = manifest.get("output_s3")
    if output != f"{CHECKPOINT_ROOT}/{arm}/seed0/{run_id}":
        raise SystemExit("training output is not the canonical multitask_v4 location")
    completion_uri = f"{TRAIN_CLAIM_ROOT}/{run_id}/step-{FINAL_STEP}.complete.json"
    if manifest.get("claims", {}).get("completion") != completion_uri:
        raise SystemExit("training manifest completion claim URI is not canonical")
    tree_root = manifest.get("checkpoint_tree_manifest_root")
    if tree_root != f"{STUDY_ROOT}/manifests/artifacts/checkpoints/{run_id}/milestones":
        raise SystemExit("training manifest checkpoint-tree root is not the milestones root")
    nuisance = _nuisance(scientific)
    if nuisance != TRAINING_NUISANCE:
        raise SystemExit(f"training nuisance drift: {nuisance} != {TRAINING_NUISANCE}")
    openpi = scientific.get("sources", {}).get("openpi")
    if not isinstance(openpi, dict) or set(openpi) != {"uri", "sha256"}:
        raise SystemExit("training manifest has no exact OpenPI source")

    expected_claim = {
        "schema_version": 1,
        "kind": campaign.MILESTONE_COMPLETION_KIND,
        "recipe": V4_70K_RECIPE,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "final_step": FINAL_STEP,
        "steps": DEPLOY_MILESTONES,
        "run_manifest_sha256": seal,
    }
    binding = campaign.verify_training_receipt_identity(
        claim,
        expected=expected_claim,
        scientific_spec_sha256=scientific_sha,
        expected_binding=campaign.TRAINING_COMPLETION_CURRENT,
        label=f"milestone completion for {run_id}",
    )
    records = claim.get("checkpoints")
    if not isinstance(records, list) or [record.get("step") for record in records] != DEPLOY_MILESTONES:
        raise SystemExit("completion claim does not enumerate the deploy milestone set")
    for record in records:
        step = record["step"]
        if record.get("checkpoint_uri") != f"{output}/deploy/{step}":
            raise SystemExit(f"completion record for step {step} does not address deploy/{step}")
        tree_sha = campaign._require_sha(record.get("tree_manifest_sha256"), f"step {step} tree digest")
        if record.get("tree_manifest_uri") != f"{tree_root}/step-{step}/{tree_sha}.json":
            raise SystemExit(f"completion record for step {step} has a non-canonical tree manifest URI")
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "scientific_spec_sha256": scientific_sha,
        "run_manifest_sha256": seal,
        "training_openpi": dict(openpi),
        "training_run_manifest_s3": expected_manifest_s3,
        "training_output_s3": output,
        "training_completion_claim_s3": completion_uri,
        "training_completion_binding": binding,
        "deployed_milestones": list(DEPLOY_MILESTONES),
    }


def fill_template(template: dict, resolved: dict, *, workspace: dict | None = None) -> dict:
    queue = json.loads(json.dumps(template))
    arm = queue["cells"][0]["arm"]
    if resolved["training_output_s3"] != f"{CHECKPOINT_ROOT}/{arm}/seed0/{resolved['run_id']}":
        raise SystemExit("resolved run does not belong to the template's arm")
    queue["comparability"]["serving_openpi"] = dict(resolved["training_openpi"])
    for cell in queue["cells"]:
        step = cell["checkpoint_step"]
        if step not in resolved["deployed_milestones"]:
            raise SystemExit(f"{cell['cell_id']} evaluates step {step}, which the run did not deploy")
        cell["run_id"] = resolved["run_id"]
        cell["deployed_milestones"] = list(resolved["deployed_milestones"])
        cell["scientific_spec_sha256"] = resolved["scientific_spec_sha256"]
        cell["run_manifest_sha256"] = resolved["run_manifest_sha256"]
        cell["training_openpi"] = dict(resolved["training_openpi"])
        cell["training_run_manifest_s3"] = resolved["training_run_manifest_s3"]
        cell["training_output_s3"] = resolved["training_output_s3"]
        cell["training_completion_claim_s3"] = resolved["training_completion_claim_s3"]
        cell["training_completion_binding"] = resolved["training_completion_binding"]
        cell["eval_id"] = f"{resolved['run_id']}-s{step}-{cell['task'].lower()}-{queue['queue_id']}"
        if cell["workspace"] == WORKSPACE_UNRESOLVED:
            if workspace is None or cell["task"] not in workspace:
                raise SystemExit(
                    f"{arm} is a workspace arm: its omega serving inputs are not resolvable from the "
                    "training receipts.  The p5 campaign's stage_workspace supports only task-bound "
                    "single-task producer claims; the M-arm Stage-E omega serving path "
                    "(workspace_models/overlays/rmme_serve_omega.py) is not wired into "
                    "launch_gpu_fleet.  Pass --workspace-json <task -> workspace block> once it is."
                )
            cell["workspace"] = json.loads(json.dumps(workspace[cell["task"]]))
    remaining = unresolved_placeholders(queue)
    if remaining:
        raise SystemExit(f"queue still carries unresolved placeholders: {sorted(remaining)}")
    return queue


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    value.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = value.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="write an unresolved milestone queue for one sweep arm")
    template.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    template.add_argument("--label", required=True, help="sweep arm label, e.g. M0-70k")
    template.add_argument("--output", type=Path, required=True)
    fill = commands.add_parser("fill", help="resolve a template from a run's manifest and completion claim")
    fill.add_argument("--template", type=Path, required=True)
    fill.add_argument("--run-manifest", required=True, help="local path or s3:// URI of the sealed manifest")
    fill.add_argument("--completion-claim", required=True, help="local path or s3:// URI of the claim")
    fill.add_argument("--workspace-json", type=Path, help="task -> workspace serving block (workspace arms)")
    fill.add_argument("--output", type=Path, required=True)
    fill.add_argument(
        "--confirm-read-s3",
        action="store_true",
        help="allow read-only S3 reads of the manifest/claim; never submits or writes cloud state",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    source_root = args.source_root.expanduser().resolve()
    if args.command == "template":
        sweep = load_sweep(args.sweep.expanduser().resolve())
        queue = build_template(sweep, args.label, source_root=source_root)
        _write(args.output.expanduser().resolve(), queue)
        print(
            f"TEMPLATE label={args.label} arm={queue['cells'][0]['arm']} cells={len(queue['cells'])} "
            f"milestones={sorted({cell['checkpoint_step'] for cell in queue['cells']})} "
            f"placeholders={sorted(unresolved_placeholders(queue))} output={args.output}"
        )
        return
    template = json.loads(args.template.expanduser().read_text(encoding="utf-8"))
    store = AwsReadStore() if args.confirm_read_s3 else None
    manifest, manifest_uri = _read_document(args.run_manifest, store=store, label="training manifest")
    claim, _ = _read_document(args.completion_claim, store=store, label="completion claim")
    resolved = verify_run(manifest, claim, arm=template["cells"][0]["arm"], manifest_uri=manifest_uri)
    workspace = (
        json.loads(args.workspace_json.expanduser().read_text(encoding="utf-8")) if args.workspace_json else None
    )
    queue = fill_template(template, resolved, workspace=workspace)
    campaign.validate_queue(
        campaign.seal_document(
            {
                **queue,
                "gates": {
                    "native_preflight": {
                        "preflight_id": "unsealed-template",
                        "claim_sha256": "0" * 64,
                        "source_tree_sha256": "0" * 64,
                    },
                    "runtime_receipt": {
                        "receipt_sha256": "0" * 64,
                        "runtime_artifact_sha256": "0" * 64,
                        "openpi_sha256": resolved["training_openpi"]["sha256"],
                    },
                },
            },
            field="queue_manifest_sha256",
        ),
        source_root=source_root,
    )
    _write(args.output.expanduser().resolve(), queue)
    print(
        f"FILLED run_id={resolved['run_id']} cells={len(queue['cells'])} output={args.output}\n"
        "This is an unsealed template: launch_p5_campaign --parallel-fixed50 attaches the topology and "
        "gates and seals it; the node stager re-verifies every receipt against S3."
    )


if __name__ == "__main__":
    main()
