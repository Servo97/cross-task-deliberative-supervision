#!/usr/bin/env python3
"""Resolve the 16 existing Pick/Button representation checkpoints into an eval queue draft.

The command is read-only with respect to AWS and writes only the requested local JSON file.  It
does not invent a preflight or runtime receipt; ``launch_p5_campaign`` adds those gates only after
an exact successful p5 preflight exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from robomme_integration.eval import campaign
from robomme_integration.launch import STUDY_ROOT

TASKS = ("PickXtimes", "ButtonUnmaskSwap")
ARMS = (
    "q3",
    "wsm_cfg",
    "wsm_tanh",
    "wsm_d8",
    "jepa_l01_k1",
    "jepa_l1_k32",
    "jepa_l01_k16",
    "salient",
)
CONFIGS = {
    "PickXtimes": "robomme_integration/eval/configs/pickxtimes.yaml",
    "ButtonUnmaskSwap": "robomme_integration/eval/configs/buttonunmaskswap.yaml",
}
WORKSPACE_CLAIM_ROOT = f"{STUDY_ROOT}/manifests/claims/workspace/uniform_gpu_v1"
RUN_MANIFEST_ROOT = f"{STUDY_ROOT}/manifests/runs/train"
TRAIN_CLAIM_ROOT = f"{STUDY_ROOT}/manifests/claims/train"
SEQUENCE_ARMS = frozenset({"q0", "q0_noforce", "q1", "a6", "q2", "q2_noforce", "q3"})


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class AwsReadStore:
    """Minimal, read-only S3 surface; no put/delete API is exposed."""

    def list(self, prefix: str) -> list[str]:
        bucket, key = _s3_parts(prefix)
        result = subprocess.run(
            [
                "aws",
                "s3api",
                "list-objects-v2",
                "--bucket",
                bucket,
                "--prefix",
                key,
                "--region",
                "us-west-2",
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
        if value.get("IsTruncated"):
            raise RuntimeError(f"refusing truncated S3 discovery under {prefix}")
        return [f"s3://{bucket}/{record['Key']}" for record in value.get("Contents", [])]

    def read(self, uri: str) -> bytes | None:
        result = subprocess.run(
            ["aws", "s3", "cp", uri, "-", "--only-show-errors", "--region", "us-west-2"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return result.stdout
        detail = result.stderr.decode(errors="replace").casefold()
        if any(marker in detail for marker in ("404", "not found", "nosuchkey")):
            return None
        raise RuntimeError(f"S3 read failed for {uri}: {detail[:500]}")


def _json(payload: bytes | None, label: str) -> dict | None:
    if payload is None:
        return None
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not one JSON object")
    return value


def _manifest_identity(manifest: dict, *, uri: str) -> tuple[str, str, str] | None:
    if manifest.get("kind") != "robomme_gpu_training_attempt":
        return None
    if manifest.get("schema_version") != 2:
        raise ValueError("training manifest schema is not the sealed single-task schema")
    seal = manifest.get("manifest_sha256")
    if not isinstance(seal, str) or campaign._seal_digest(manifest, "manifest_sha256") != seal:
        raise ValueError("training manifest self-seal mismatch")
    scientific = manifest.get("scientific")
    if not isinstance(scientific, dict):
        raise ValueError("training manifest lacks scientific payload")
    scientific_sha = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if scientific_sha != manifest.get("scientific_spec_sha256"):
        raise ValueError("training manifest scientific digest mismatch")
    task = scientific.get("task", {}).get("name")
    arm = scientific.get("arm")
    run_id = manifest.get("run_id")
    if not all(isinstance(value, str) for value in (task, arm, run_id)):
        raise ValueError("training manifest task/arm/run identity is malformed")
    attempt_id = manifest.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or not re.fullmatch(rf"{re.escape(run_id)}-attempt[1-9][0-9]*", attempt_id)
        or manifest.get("manifest_s3") != uri
        or not uri.endswith(f"/{run_id}/{attempt_id}.json")
    ):
        raise ValueError("training manifest attempt/object identity is malformed")
    completion = manifest.get("claims", {}).get("completion")
    if completion != f"{TRAIN_CLAIM_ROOT}/{run_id}/step-19999.complete.json":
        raise ValueError("training manifest completion URI is not canonical")
    tree_root = manifest.get("checkpoint_tree_manifest_root")
    if not isinstance(tree_root, str) or not tree_root.endswith(f"/{run_id}/step-19999"):
        raise ValueError("training manifest checkpoint-tree root is not run/step bound")
    return task, arm, run_id


def _verify_receipt_chain(store: AwsReadStore, *, manifest: dict, completion: dict) -> str:
    """Authenticate the deploy tree and return the sealed completion-binding mode."""
    run_id = manifest["run_id"]
    scientific_sha = manifest["scientific_spec_sha256"]
    checkpoint_uri = f"{manifest['output_s3']}/deploy/19999"
    expected = {
        "schema_version": 1,
        "kind": "robomme_gpu_checkpoint_complete",
        "run_id": run_id,
        "attempt_id": manifest["attempt_id"],
        "step": 19_999,
        "checkpoint_uri": checkpoint_uri,
        "run_manifest_sha256": manifest["manifest_sha256"],
    }
    binding = campaign.verify_training_receipt_identity(
        completion,
        expected=expected,
        scientific_spec_sha256=scientific_sha,
        expected_binding=None,
        label=f"training completion for {run_id}",
    )
    tree_sha = campaign._require_sha(completion.get("tree_manifest_sha256"), "checkpoint tree digest")
    tree_uri = campaign._safe_s3(completion.get("tree_manifest_uri"))
    expected_tree_uri = f"{campaign._safe_s3(manifest['checkpoint_tree_manifest_root'])}/{tree_sha}.json"
    if tree_uri != expected_tree_uri:
        raise ValueError("checkpoint tree URI is not the manifest-bound content address")
    tree_bytes = store.read(tree_uri)
    campaign.verify_checkpoint_tree_identity(
        tree_bytes,
        expected_sha256=tree_sha,
        checkpoint_uri=checkpoint_uri,
        label=f"checkpoint tree for {run_id}",
    )
    deploy_uri = f"{checkpoint_uri}/_DEPLOY_COMPLETE.json"
    deploy = _json(store.read(deploy_uri), deploy_uri)
    if deploy is None:
        raise ValueError(f"checkpoint deploy receipt is absent for {run_id}")
    campaign.verify_training_receipt_identity(
        deploy,
        expected={
            **expected,
            "kind": "robomme_gpu_deploy_checkpoint_complete",
            "tree_manifest_sha256": tree_sha,
        },
        scientific_spec_sha256=scientific_sha,
        expected_binding=binding,
        label=f"checkpoint deploy receipt for {run_id}",
    )
    return binding


def resolve_training(store: AwsReadStore, task: str, arm: str) -> tuple[str, dict, dict, str]:
    prefix = f"{RUN_MANIFEST_ROOT}/st-v1-{task.lower()}-{arm}-seed0-"
    candidates: list[tuple[str, dict, dict]] = []
    for uri in store.list(prefix):
        manifest = _json(store.read(uri), uri)
        if manifest is None or _manifest_identity(manifest, uri=uri) != (
            task,
            arm,
            manifest.get("run_id"),
        ):
            continue
        run_id = manifest["run_id"]
        completion_uri = f"{TRAIN_CLAIM_ROOT}/{run_id}/step-19999.complete.json"
        completion = _json(store.read(completion_uri), completion_uri)
        if completion is None:
            continue
        # Multiple immutable attempts may exist under one run.  The completion claim selects the
        # one authoritative sealed attempt; non-selected attempts are not identity drift.
        if completion.get("run_manifest_sha256") != manifest["manifest_sha256"]:
            continue
        if manifest.get("claims", {}).get("completion") != completion_uri:
            raise ValueError("sealed manifest changed its canonical completion claim")
        binding = _verify_receipt_chain(store, manifest=manifest, completion=completion)
        candidates.append((uri, manifest, completion, binding))
    unique = {(item[1]["run_id"], item[1]["manifest_sha256"]): item for item in candidates}
    if len(unique) != 1:
        identities = sorted(unique)
        raise ValueError(
            f"expected one completed {task}/{arm} training identity, found {identities}; "
            "do not select a retrain by timestamp"
        )
    return next(iter(unique.values()))


def resolve_workspace(store: AwsReadStore, task: str, scientific: dict) -> dict:
    expected = scientific.get("workspace_representation")
    if not isinstance(expected, dict) or not isinstance(expected.get("omega"), dict):
        raise ValueError(f"{task} workspace-serving checkpoint has no workspace scientific identity")
    candidates = []
    for uri in store.list(f"{WORKSPACE_CLAIM_ROOT}/{task}/"):
        payload = store.read(uri)
        claim = _json(payload, uri)
        if payload is None or claim is None:
            continue
        if (
            claim.get("kind") == "robomme_all16_workspace_task_complete"
            and claim.get("task") == task
            and claim.get("encoder_id") == expected.get("encoder_id")
            and claim.get("omega") == expected.get("omega")
        ):
            candidates.append((uri, payload, claim))
    if len(candidates) > 1:
        raise ValueError(f"multiple exact workspace producer claims exist for {task}")
    if candidates:
        uri, payload, claim = candidates[0]
        representation = claim.get("representation")
        if not isinstance(representation, dict):
            raise ValueError(f"workspace claim for {task} has no representation")
        return {
            "provenance_mode": campaign.WORKSPACE_PROVENANCE_CLAIM,
            "claim_s3": uri,
            "claim_sha256": hashlib.sha256(payload).hexdigest(),
            "encoder_id": claim["encoder_id"],
            "representation_s3": representation["uri"],
            "completion_sha256": representation["completion_sha256"],
            "step": representation["step"],
        }

    # The first Pick/Button producer predates task completion claims.  Its training manifest pins
    # the raw omega-manifest digest; that manifest, in turn, content-addresses the encoder identity
    # and exact checkpoint tree.  Accept only that complete cryptographic chain at its canonical
    # task/encoder path—never a newer producer claim with a different encoder.
    encoder_id = expected["encoder_id"]
    omega = expected["omega"]
    omega_uri = omega.get("uri")
    omega_sha = omega.get("manifest_sha256")
    canonical_root = f"{STUDY_ROOT}/artifacts/robomme/workspace/{task}/{encoder_id}"
    if omega_uri != f"{canonical_root}/omega":
        raise ValueError(f"{task} legacy omega URI is not the training-pinned canonical path")
    campaign._require_sha(omega_sha, f"{task} legacy omega manifest digest")
    omega_manifest_uri = f"{omega_uri}/MANIFEST.json"
    omega_payload = store.read(omega_manifest_uri)
    if omega_payload is None or hashlib.sha256(omega_payload).hexdigest() != omega_sha:
        raise ValueError(f"{task} training-pinned omega manifest is absent or corrupt")
    omega_manifest = _json(omega_payload, omega_manifest_uri)
    assert omega_manifest is not None
    identity = omega_manifest.get("encoder_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{task} legacy omega manifest lacks encoder identity")
    step = identity.get("checkpoint_step")
    if step != 10_000:
        raise ValueError(f"{task} legacy workspace checkpoint is not the intended step 10000")
    representation_uri = f"{canonical_root}/representation/step-{step}"
    completion_uri = f"{representation_uri}/WSM_GENERATION_COMPLETE.json"
    run_config_uri = f"{representation_uri}/WSM_RUN_CONFIG.json"
    best_uri = f"{representation_uri}/WSM_BEST.json"
    completion_payload = store.read(completion_uri)
    run_config_payload = store.read(run_config_uri)
    best_payload = store.read(best_uri)
    if completion_payload is None or run_config_payload is None or best_payload is None:
        raise ValueError(f"{task} legacy representation metadata chain is incomplete")
    completion = _json(completion_payload, completion_uri)
    assert completion is not None
    embedded = completion.get("embedded_sha256")
    if not isinstance(embedded, dict):
        raise ValueError(f"{task} legacy representation completion has no embedded seals")
    workspace = {
        "provenance_mode": campaign.WORKSPACE_PROVENANCE_LEGACY,
        "encoder_id": encoder_id,
        "omega_manifest_s3": omega_manifest_uri,
        "omega_manifest_sha256": omega_sha,
        "task_manifest_sha256": scientific.get("task", {}).get("task_manifest_sha256"),
        "representation_s3": representation_uri,
        "completion_sha256": hashlib.sha256(completion_payload).hexdigest(),
        "step": step,
        "checkpoint_tree_sha256": identity.get("checkpoint_tree_sha256"),
        "run_config_sha256": identity.get("run_config_sha256"),
        "best_sha256": embedded.get("WSM_BEST.json"),
        "materializer_sha256": identity.get("materializer_sha256"),
    }
    for name in (
        "encoder_id",
        "omega_manifest_sha256",
        "task_manifest_sha256",
        "completion_sha256",
        "checkpoint_tree_sha256",
        "run_config_sha256",
        "best_sha256",
        "materializer_sha256",
    ):
        campaign._require_sha(workspace[name], f"{task} legacy workspace {name}")
    campaign.verify_legacy_workspace_metadata(
        workspace,
        task=task,
        omega_payload=omega_payload,
        completion_payload=completion_payload,
        run_config_payload=run_config_payload,
        best_payload=best_payload,
    )
    return workspace


def _nuisance(scientific: dict) -> dict:
    data = scientific["data"]
    initialization = scientific["initialization"]
    training = scientific["training"]
    return {
        "data_parent_inventory_sha256": data["parent_inventory_sha256"],
        "data_task_inventory_sha256": data["derived_task_inventory_sha256"],
        "initialization_inventory_sha256": initialization["inventory_sha256"],
        "initialization_checkpoint_s3": initialization["checkpoint_s3"],
        "seed": training["seed"],
        "steps": training["steps"],
        "action_horizon": training["action_horizon"],
        "window_len": training.get("window_len"),
        "chunk_stride": training.get("chunk_stride"),
    }


def build_template(store: AwsReadStore, *, queue_id: str, source_root: Path) -> dict:
    publish = f"{STUDY_ROOT}/evaluations/fixed50_campaigns/{queue_id}"
    cells = []
    common: dict[str, dict] = {}
    configs: dict[str, dict] = {}
    serving_openpi: dict | None = None
    for task in TASKS:
        config = CONFIGS[task]
        config_sha = hashlib.sha256((source_root / config).read_bytes()).hexdigest()
        configs[task] = {"path": config, "sha256": config_sha}
        for arm in ARMS:
            manifest_uri, manifest, _completion, completion_binding = resolve_training(store, task, arm)
            scientific = manifest["scientific"]
            training_openpi = scientific.get("sources", {}).get("openpi")
            if not isinstance(training_openpi, dict):
                raise ValueError(f"completed {task}/{arm} has no exact OpenPI training source")
            if serving_openpi is not None and serving_openpi != training_openpi:
                raise ValueError(
                    "selected cells use different OpenPI training archives; split them into source-matched eval queues"
                )
            serving_openpi = training_openpi
            nuisance = _nuisance(scientific)
            invariant = {key: value for key, value in nuisance.items() if key not in {"window_len", "chunk_stride"}}
            if task in common and common[task] != invariant:
                raise ValueError(f"completed {task}/{arm} changed a common nuisance variable")
            common[task] = invariant
            workspace = resolve_workspace(store, task, scientific) if arm in campaign.WORKSPACE_EVAL_ARMS else None
            ordinal = len(cells)
            cell_id = f"{ordinal:03d}-{task.lower()}-{arm}"
            run_id = manifest["run_id"]
            cells.append(
                {
                    "ordinal": ordinal,
                    "cell_id": cell_id,
                    "task": task,
                    "arm": arm,
                    "run_id": run_id,
                    "final_step": 19_999,
                    "scientific_spec_sha256": manifest["scientific_spec_sha256"],
                    "run_manifest_sha256": manifest["manifest_sha256"],
                    "training_openpi": training_openpi,
                    "training_run_manifest_s3": manifest_uri,
                    "training_output_s3": manifest["output_s3"],
                    "training_completion_claim_s3": manifest["claims"]["completion"],
                    "training_completion_binding": completion_binding,
                    "benchmark_config": config,
                    "benchmark_config_sha256": config_sha,
                    "training_nuisance": nuisance,
                    "eval_id": f"{run_id}-fixed50-{queue_id}",
                    "result_claim_s3": f"{publish}/cells/{cell_id}/result.complete.json",
                    "workspace": workspace,
                    "ptrm": None,
                    "cfg_guidance_scale": 1.0,
                }
            )
    return {
        "schema_version": 1,
        "kind": campaign.QUEUE_KIND,
        "queue_id": queue_id,
        "publish_root_s3": publish,
        "claims": {"manifest": f"{publish}/manifest.json", "completion": f"{publish}/complete.json"},
        "topology": {
            "policy_gpus": [0, 1, 2, 3],
            "simulator_gpus": [4, 5, 6, 7],
            "simulator_shards": 16,
            "cpu_range": "0-191",
            "base_port": 18100,
            "xla_memory_fraction": 0.65,
        },
        "retry": {"classifier_version": campaign.CLASSIFIER_VERSION, "max_attempts": 2},
        "limits": {
            "max_run_seconds": 75_600,
            "runtime_reserve_seconds": 1_800,
            "estimated_cell_seconds": 3_600,
            "minimum_free_bytes": 64 * 1024**3,
        },
        "comparability": {
            "serving_openpi": serving_openpi,
            "task_benchmark_configs": configs,
            "task_common_training_nuisance": common,
            "sequence_geometry_policy": "manifest_verified_per_cell_not_assumed_common",
        },
        "cells": cells,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--queue-id", default="pick-button-representation-fixed50-v1")
    value.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    value.add_argument("--output", type=Path, required=True)
    value.add_argument(
        "--confirm-read-s3",
        action="store_true",
        help="perform read-only S3 discovery; never submits a job or writes cloud state",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.confirm_read_s3:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": TASKS,
                    "arms": ARMS,
                    "manifest_prefix": RUN_MANIFEST_ROOT,
                    "workspace_claim_prefix": WORKSPACE_CLAIM_ROOT,
                    "output": str(args.output),
                    "note": "no AWS command or local write performed; pass --confirm-read-s3",
                },
                indent=2,
            )
        )
        return
    output = args.output.expanduser().resolve()
    template = build_template(
        AwsReadStore(),
        queue_id=args.queue_id,
        source_root=args.source_root.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    temporary.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"SEALED_INPUTS_RESOLVED cells={len(template['cells'])} output={output}")
    print("This is a gate-free local draft; run launch_p5_campaign only with a real preflight claim.")


if __name__ == "__main__":
    main()
