from robomme_integration.training.single_task import TASK_ORDER, task_manifest_sha256
from robomme_integration.training.workspace_index_publish import build_index


def test_index_publisher_cross_binds_all_tasks_and_exposes_mixed_producer_topology():
    manifests = []
    claims = []
    for pair_index in range(8):
        hardware = "p5e" if pair_index == 0 else "p5"
        accelerator = "4xH200" if pair_index == 0 else "4xH100"
        pair_id = f"pair-{pair_index}"
        records = []
        for task in TASK_ORDER[pair_index * 2 : pair_index * 2 + 2]:
            run_id = f"run-{task}"
            records.append(
                {
                    "task": task,
                    "run_id": run_id,
                    "task_inventory_sha256": "a" * 64,
                    "claim_s3": f"s3://bucket/{task}.json",
                    "scientific": {
                        "task_inventory_sha256": "a" * 64,
                        "producer": {
                            "hardware": hardware,
                            "accelerator": accelerator,
                            "devices": 4,
                        },
                    },
                }
            )
            claims.append(
                {
                    "kind": "robomme_all16_workspace_task_complete",
                    "campaign": "uniform_gpu_v1",
                    "task": task,
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "source_tree_sha256": "b" * 64,
                    "task_manifest_sha256": task_manifest_sha256(task),
                    "encoder_id": "c" * 64,
                    "omega": {"uri": f"s3://bucket/{task}/omega", "manifest_sha256": "d" * 64},
                    "supervision": {
                        "uri": f"s3://bucket/{task}/supervision",
                        "manifest_sha256": "e" * 64,
                    },
                    "representation": {
                        "uri": f"s3://bucket/{task}/representation",
                        "step": 10000,
                        "completion_sha256": "f" * 64,
                    },
                }
            )
        manifests.append(
            {
                "kind": "robomme_all16_workspace_pair_attempt",
                "pair_id": pair_id,
                "source_tree_sha256": "b" * 64,
                "tasks": records,
                "infrastructure": {"instance_type": "ml.p5e.48xlarge" if pair_index == 0 else "ml.p5.48xlarge"},
            }
        )

    index = build_index(manifests, claims)
    assert tuple(index["tasks"]) == TASK_ORDER
    assert index["producer_topology"] == {
        "node_accelerators": ["4xH100", "4xH200"],
        "mixed_accelerators": True,
        "devices_per_task": 4,
    }
    assert index["tasks"][TASK_ORDER[0]]["producer"]["hardware"] == "p5e"
    assert index["tasks"][TASK_ORDER[-1]]["producer"]["hardware"] == "p5"
