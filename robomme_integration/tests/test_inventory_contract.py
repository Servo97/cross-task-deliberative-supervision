from __future__ import annotations

import json

import pytest

from robomme_integration.fleet.inventory import (
    BENCHMARK_TASKS,
    DATA_ARTIFACT,
    HF_REPO,
    HF_REVISION,
    NORM_STATS_SHA256,
    PI05_BASE_INIT_ARTIFACT,
    TASK_INSTRUCTION_VARIANTS,
    validate_inventory,
)
from robomme_integration.fleet.stage_dataset import _snapshot_inventory


def test_snapshot_distinguishes_benchmark_tasks_from_instruction_variants(tmp_path):
    meta = tmp_path / "meta"
    data = tmp_path / "data" / "chunk-000"
    meta.mkdir()
    data.mkdir(parents=True)

    tasks = [
        {"task_index": index, "task": f"instruction variant {index}"} for index in range(TASK_INSTRUCTION_VARIANTS)
    ]
    episodes = [
        {
            "episode_index": index,
            "tasks": [tasks[index % TASK_INSTRUCTION_VARIANTS]["task"]],
            "length": 1,
        }
        for index in range(1600)
    ]
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1600,
                "total_tasks": TASK_INSTRUCTION_VARIANTS,
                "total_chunks": 2,
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episodes),
        encoding="utf-8",
    )
    (meta / "episodes_stats.jsonl").write_text("{}\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in tasks),
        encoding="utf-8",
    )
    for index in range(1600):
        (data / f"episode_{index:06d}.parquet").touch()

    episode_count, task_variant_count, parquet_count = _snapshot_inventory(tmp_path)
    assert episode_count == 1600
    assert BENCHMARK_TASKS == 16
    assert task_variant_count == 116
    assert parquet_count == 1600


def test_v2_inventory_requires_strong_parquet_checksums(tmp_path):
    crc64 = "AAAAAAAAAAA="
    objects = [
        {
            "key": f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet",
            "size_bytes": 1,
            "etag": f'"etag-{index}"',
            "checksum_crc64nvme": crc64,
            "source_sha256": f"{index:064x}",
        }
        for index in range(1600)
    ]
    objects.extend(
        {
            "key": key,
            "size_bytes": 1,
            "etag": f'"{key}"',
            "checksum_crc64nvme": crc64,
        }
        for key in (
            "_robomme_source.json",
            "meta/info.json",
            "meta/episodes.jsonl",
            "meta/episodes_stats.jsonl",
            "meta/tasks.jsonl",
            "assets/robomme/norm_stats.json",
        )
    )
    manifest = {
        "schema_version": 2,
        "artifact": DATA_ARTIFACT,
        "root_s3": "s3://test-bucket/robomme",
        "content_addressing": "etag",
        "selection": {
            "name": "all16_v1",
            "hf_repo_id": HF_REPO,
            "hf_revision": HF_REVISION,
            "episodes": 1600,
            "benchmark_tasks": BENCHMARK_TASKS,
            "task_instruction_variants": TASK_INSTRUCTION_VARIANTS,
            "training_scope": "execution_frames_only",
            "norm_stats_sha256": NORM_STATS_SHA256,
        },
        "objects": objects,
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_inventory(
        path,
        artifact=DATA_ARTIFACT,
        root_s3="s3://test-bucket/robomme",
    ) == {"objects": 1606, "bytes": 1606}

    del objects[0]["source_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="requires source SHA-256"):
        validate_inventory(
            path,
            artifact=DATA_ARTIFACT,
            root_s3="s3://test-bucket/robomme",
        )


def test_pi05_base_initialization_has_a_distinct_inventory_identity(tmp_path):
    manifest = {
        "schema_version": 1,
        "artifact": PI05_BASE_INIT_ARTIFACT,
        "root_s3": "s3://test-bucket/pi05-base",
        "content_addressing": "etag",
        "selection": None,
        "objects": [
            {
                "key": "params/weights",
                "size_bytes": 1,
                "etag": '"params"',
            },
            {
                "key": "assets/example/norm_stats.json",
                "size_bytes": 1,
                "etag": '"assets"',
            },
        ],
    }
    path = tmp_path / "pi05-base.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_inventory(
        path,
        artifact=PI05_BASE_INIT_ARTIFACT,
        root_s3="s3://test-bucket/pi05-base",
    ) == {"objects": 2, "bytes": 2}
