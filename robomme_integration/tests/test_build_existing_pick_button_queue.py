from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from robomme_integration.eval import build_existing_pick_button_queue as builder
from robomme_integration.eval import campaign
from robomme_integration.launch import OPENPI, OPENPI_SHA


class MemoryReadStore:
    def __init__(self):
        self.values: dict[str, bytes] = {}

    def list(self, prefix: str) -> list[str]:
        return sorted(uri for uri in self.values if uri.startswith(prefix))

    def read(self, uri: str) -> bytes | None:
        return self.values.get(uri)


def _put_json(store: MemoryReadStore, uri: str, value: dict) -> bytes:
    payload = campaign._canonical(value)
    store.values[uri] = payload
    return payload


def _seed_store() -> MemoryReadStore:
    store = MemoryReadStore()
    workspace_by_task = {}
    for task in builder.TASKS:
        encoder = ("1" if task == "PickXtimes" else "2") * 64
        omega = {
            "uri": f"s3://bucket/workspace/{task}/{encoder}/omega",
            "manifest_sha256": "3" * 64,
        }
        claim = {
            "kind": "robomme_all16_workspace_task_complete",
            "task": task,
            "encoder_id": encoder,
            "omega": omega,
            "representation": {
                "uri": f"s3://bucket/workspace/{task}/{encoder}/representation/step-10000",
                "step": 10_000,
                "completion_sha256": "4" * 64,
            },
        }
        claim_uri = f"{builder.WORKSPACE_CLAIM_ROOT}/{task}/producer.complete.json"
        _put_json(store, claim_uri, claim)
        workspace_by_task[task] = {"encoder_id": encoder, "omega": omega}
        for arm in builder.ARMS:
            run_id = f"st-v1-{task.lower()}-{arm}-seed0-aaaaaaaaaaaaaaaa"
            scientific = {
                "task": {"name": task},
                "arm": arm,
                "sources": {"openpi": {"uri": OPENPI, "sha256": OPENPI_SHA}},
                "initialization": {
                    "checkpoint_s3": "s3://bucket/init/149999",
                    "inventory_sha256": "5" * 64,
                },
                "data": {
                    "parent_inventory_sha256": "6" * 64,
                    "derived_task_inventory_sha256": ("7" if task == "PickXtimes" else "8") * 64,
                },
                "training": {
                    "seed": 0,
                    "steps": 20_000,
                    "action_horizon": 20,
                    "window_len": 8 if arm in builder.SEQUENCE_ARMS else None,
                    "chunk_stride": 10 if arm in builder.SEQUENCE_ARMS else None,
                },
                "workspace_representation": workspace_by_task[task],
            }
            scientific_sha = hashlib.sha256(
                json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            completion_uri = f"{builder.TRAIN_CLAIM_ROOT}/{run_id}/step-19999.complete.json"
            output = f"s3://bucket/checkpoints/{task}/{arm}/seed0/{run_id}"
            attempt_id = f"{run_id}-attempt1"
            manifest_uri = f"{builder.RUN_MANIFEST_ROOT}/{run_id}/{attempt_id}.json"
            tree_root = f"s3://bucket/manifests/artifacts/checkpoints/{run_id}/step-19999"
            manifest = campaign.seal_document(
                {
                    "schema_version": 2,
                    "kind": "robomme_gpu_training_attempt",
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "manifest_s3": manifest_uri,
                    "scientific_spec_sha256": scientific_sha,
                    "scientific": scientific,
                    "output_s3": output,
                    "checkpoint_tree_manifest_root": tree_root,
                    "claims": {"completion": completion_uri},
                },
                field="manifest_sha256",
            )
            _put_json(store, manifest_uri, manifest)
            checkpoint_uri = f"{output}/deploy/19999"
            tree = {
                "schema_version": 1,
                "checkpoint_uri": checkpoint_uri,
                "objects": [{"key": "params/mock", "sha256": "9" * 64, "size_bytes": 1}],
            }
            tree_payload = campaign._canonical(tree)
            tree_sha = hashlib.sha256(tree_payload).hexdigest()
            tree_uri = f"{tree_root}/{tree_sha}.json"
            store.values[tree_uri] = tree_payload
            _put_json(
                store,
                completion_uri,
                {
                    "schema_version": 1,
                    "kind": "robomme_gpu_checkpoint_complete",
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "step": 19_999,
                    "checkpoint_uri": checkpoint_uri,
                    "scientific_spec_sha256": scientific_sha,
                    "run_manifest_sha256": manifest["manifest_sha256"],
                    "tree_manifest_sha256": tree_sha,
                    "tree_manifest_uri": tree_uri,
                },
            )
            _put_json(
                store,
                f"{checkpoint_uri}/_DEPLOY_COMPLETE.json",
                {
                    "schema_version": 1,
                    "kind": "robomme_gpu_deploy_checkpoint_complete",
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "step": 19_999,
                    "checkpoint_uri": checkpoint_uri,
                    "scientific_spec_sha256": scientific_sha,
                    "run_manifest_sha256": manifest["manifest_sha256"],
                    "tree_manifest_sha256": tree_sha,
                },
            )
    return store


def test_builder_resolves_exact_16_cells_without_inventing_runtime_gates():
    source = builder.Path(__file__).resolve().parents[2]
    template = builder.build_template(
        _seed_store(),
        queue_id="pick-button-representation-fixed50-v1",
        source_root=source,
    )
    assert "gates" not in template and "queue_manifest_sha256" not in template
    assert len(template["cells"]) == 16
    assert {(cell["task"], cell["arm"]) for cell in template["cells"]} == {
        (task, arm) for task in builder.TASKS for arm in builder.ARMS
    }
    assert {cell["training_completion_binding"] for cell in template["cells"]} == {
        campaign.TRAINING_COMPLETION_CURRENT
    }
    assert all(
        cell["workspace"] is not None for cell in template["cells"] if cell["arm"] in campaign.WORKSPACE_EVAL_ARMS
    )
    assert all(
        cell["workspace"] is None for cell in template["cells"] if cell["arm"] not in campaign.WORKSPACE_EVAL_ARMS
    )

    queue = dict(template)
    queue["gates"] = {
        "native_preflight": {
            "preflight_id": "p5-native-eval-v1-test",
            "claim_sha256": "9" * 64,
            "source_tree_sha256": "a" * 64,
        },
        "runtime_receipt": {
            "receipt_sha256": "b" * 64,
            "runtime_artifact_sha256": "c" * 64,
            "openpi_sha256": "d" * 64,
        },
    }
    queue = campaign.seal_document(queue, field="queue_manifest_sha256")
    campaign.validate_queue(queue, source_root=source)

    drifted = deepcopy(queue)
    drifted["cells"][0]["training_openpi"] = {
        "uri": f"s3://bucket/openpi/{'e' * 64}.tgz",
        "sha256": "e" * 64,
    }
    drifted = campaign.seal_document(drifted, field="queue_manifest_sha256")
    with pytest.raises(ValueError, match="training/serving OpenPI"):
        campaign.validate_queue(drifted, source_root=source)


def test_legacy_receipts_derive_scientific_identity_only_from_sealed_manifests():
    store = _seed_store()
    for uri, payload in list(store.values.items()):
        value = json.loads(payload)
        if value.get("kind") in {
            "robomme_gpu_checkpoint_complete",
            "robomme_gpu_deploy_checkpoint_complete",
        }:
            value.pop("scientific_spec_sha256")
            _put_json(store, uri, value)
    source = builder.Path(__file__).resolve().parents[2]
    template = builder.build_template(
        store,
        queue_id="pick-button-legacy-fixed50-v1",
        source_root=source,
    )
    assert len(template["cells"]) == 16
    assert {cell["training_completion_binding"] for cell in template["cells"]} == {campaign.TRAINING_COMPLETION_LEGACY}
    for cell in template["cells"]:
        manifest = json.loads(store.values[cell["training_run_manifest_s3"]])
        scientific_sha = hashlib.sha256(
            json.dumps(
                manifest["scientific"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        assert cell["scientific_spec_sha256"] == scientific_sha


def test_receipt_scientific_mismatch_and_mixed_legacy_mode_fail_closed():
    store = _seed_store()
    run_id = "st-v1-pickxtimes-q3-seed0-aaaaaaaaaaaaaaaa"
    completion_uri = f"{builder.TRAIN_CLAIM_ROOT}/{run_id}/step-19999.complete.json"
    completion = json.loads(store.values[completion_uri])
    completion["scientific_spec_sha256"] = "0" * 64
    _put_json(store, completion_uri, completion)
    with pytest.raises(ValueError, match="differs from the sealed training manifest"):
        builder.resolve_training(store, "PickXtimes", "q3")

    store = _seed_store()
    completion = json.loads(store.values[completion_uri])
    checkpoint_uri = completion["checkpoint_uri"]
    deploy_uri = f"{checkpoint_uri}/_DEPLOY_COMPLETE.json"
    deploy = json.loads(store.values[deploy_uri])
    deploy.pop("scientific_spec_sha256")
    _put_json(store, deploy_uri, deploy)
    with pytest.raises(ValueError, match="completion-binding mode drift"):
        builder.resolve_training(store, "PickXtimes", "q3")


def test_two_fully_authenticated_completed_retrains_are_an_explicit_collision():
    store = _seed_store()
    task, arm = "PickXtimes", "q3"
    original_uri = store.list(f"{builder.RUN_MANIFEST_ROOT}/st-v1-pickxtimes-q3-seed0-")[0]
    original = json.loads(store.values[original_uri])
    old_run_id = original["run_id"]
    new_run_id = f"st-v1-pickxtimes-q3-seed0-{'b' * 16}"
    attempt_id = f"{new_run_id}-attempt1"
    manifest_uri = f"{builder.RUN_MANIFEST_ROOT}/{new_run_id}/{attempt_id}.json"
    completion_uri = f"{builder.TRAIN_CLAIM_ROOT}/{new_run_id}/step-19999.complete.json"
    output = f"s3://bucket/checkpoints/{task}/{arm}/seed0/{new_run_id}"
    tree_root = f"s3://bucket/manifests/artifacts/checkpoints/{new_run_id}/step-19999"
    manifest = deepcopy(original)
    manifest.pop("manifest_sha256")
    manifest.update(
        {
            "run_id": new_run_id,
            "attempt_id": attempt_id,
            "manifest_s3": manifest_uri,
            "output_s3": output,
            "checkpoint_tree_manifest_root": tree_root,
            "claims": {"completion": completion_uri},
        }
    )
    manifest = campaign.seal_document(manifest, field="manifest_sha256")
    _put_json(store, manifest_uri, manifest)
    checkpoint_uri = f"{output}/deploy/19999"
    tree = {
        "schema_version": 1,
        "checkpoint_uri": checkpoint_uri,
        "objects": [{"key": "params/mock", "sha256": "9" * 64, "size_bytes": 1}],
    }
    tree_payload = campaign._canonical(tree)
    tree_sha = hashlib.sha256(tree_payload).hexdigest()
    tree_uri = f"{tree_root}/{tree_sha}.json"
    store.values[tree_uri] = tree_payload
    shared = {
        "schema_version": 1,
        "run_id": new_run_id,
        "attempt_id": attempt_id,
        "step": 19_999,
        "checkpoint_uri": checkpoint_uri,
        "scientific_spec_sha256": manifest["scientific_spec_sha256"],
        "run_manifest_sha256": manifest["manifest_sha256"],
        "tree_manifest_sha256": tree_sha,
    }
    _put_json(
        store,
        completion_uri,
        {
            **shared,
            "kind": "robomme_gpu_checkpoint_complete",
            "tree_manifest_uri": tree_uri,
        },
    )
    _put_json(
        store,
        f"{checkpoint_uri}/_DEPLOY_COMPLETE.json",
        {**shared, "kind": "robomme_gpu_deploy_checkpoint_complete"},
    )
    with pytest.raises(ValueError, match="expected one completed") as error:
        builder.resolve_training(store, task, arm)
    assert old_run_id in str(error.value) and new_run_id in str(error.value)
