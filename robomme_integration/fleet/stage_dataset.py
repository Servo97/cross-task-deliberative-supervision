#!/usr/bin/env python3
"""Stage the pinned public RoboMME LeRobot dataset once into the consolidated S3 prefix.

This performs large network/S3 writes and therefore refuses unless --confirm-stage is present
after explicit user approval. The HF snapshot remains in the standard shared cache; no second local
dataset copy is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from wsm_settings import RESULTS_BUCKET, STUDY_OWNER

from .build_inventory import build
from .inventory import (
    BENCHMARK_TASKS,
    DATA_ARTIFACT,
    HF_REPO,
    HF_REVISION,
    NORM_STATS_SHA256,
    TASK_INSTRUCTION_VARIANTS,
)

BUCKET = RESULTS_BUCKET
OWNER = STUDY_OWNER
DATA_ROOT = f"s3://{BUCKET}/{OWNER}/wsm_robocasa/datasets/robomme/v1/lerobot_all16"
STUDY_ROOT = f"s3://{BUCKET}/{OWNER}/wsm_robocasa/studies/long_context_v1"


def _snapshot_inventory(root: Path) -> tuple[int, int, int]:
    required = [
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "episodes_stats.jsonl",
        root / "meta" / "tasks.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"HF snapshot lacks required metadata: {missing}")
    with required[1].open(encoding="utf-8") as stream:
        episode_records = [json.loads(line) for line in stream if line.strip()]
    episodes = [record["episode_index"] for record in episode_records]
    with required[3].open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    info = json.loads(required[0].read_text(encoding="utf-8"))
    parquet = list((root / "data").rglob("*.parquet"))
    if len(episodes) != 1600 or len(set(episodes)) != 1600:
        raise ValueError(f"expected 1600 unique episodes, got {len(episodes)}")
    task_indices = [task.get("task_index") for task in tasks]
    task_text = [task.get("task") for task in tasks]
    if (
        len(tasks) != TASK_INSTRUCTION_VARIANTS
        or sorted(task_indices) != list(range(TASK_INSTRUCTION_VARIANTS))
        or len(set(task_text)) != TASK_INSTRUCTION_VARIANTS
    ):
        raise ValueError(
            f"expected {TASK_INSTRUCTION_VARIANTS} unique contiguous instruction variants, got {len(tasks)} records"
        )
    if (
        info.get("total_episodes") != 1600
        or info.get("total_tasks") != TASK_INSTRUCTION_VARIANTS
        or info.get("total_chunks") != 2
    ):
        raise ValueError("info.json counts differ from the pinned RoboMME contract")
    known_tasks = set(task_text)
    if any(
        not isinstance(record.get("tasks"), list)
        or not record["tasks"]
        or any(task not in known_tasks for task in record["tasks"])
        for record in episode_records
    ):
        raise ValueError("episode metadata references an unknown or empty instruction variant")
    if len(parquet) != 1600:
        raise ValueError(f"expected 1600 parquet files, got {len(parquet)}")
    return len(episodes), len(tasks), len(parquet)


def _publish_once(client, local_path: Path, uri: str) -> None:
    prefix = f"s3://{BUCKET}/"
    if not uri.startswith(prefix):
        raise ValueError(f"noncanonical destination {uri}")
    key = uri[len(prefix) :]
    body = local_path.read_bytes()
    try:
        client.put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*")
    except Exception as error:
        response = getattr(error, "response", {})
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
            raise
        existing = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        if existing != body:
            raise ValueError(f"immutable collision at {uri}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--inventory-output-dir", default="/tmp/robomme-inventory")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--inventory-workers", type=int, default=64)
    parser.add_argument("--confirm-stage", action="store_true")
    args = parser.parse_args()
    if not args.confirm_stage:
        raise SystemExit("dataset staging blocked: obtain explicit user approval and pass --confirm-stage")

    norm_stats = Path(args.norm_stats).resolve()
    if hashlib.sha256(norm_stats.read_bytes()).hexdigest() != NORM_STATS_SHA256:
        raise SystemExit("official norm_stats checksum mismatch")

    import boto3
    from huggingface_hub import snapshot_download

    client = boto3.client("s3")
    response = client.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"{OWNER}/wsm_robocasa/datasets/robomme/v1/lerobot_all16/",
        MaxKeys=2,
    )
    existing = response.get("Contents", [])
    marker_key = f"{OWNER}/wsm_robocasa/datasets/robomme/v1/lerobot_all16/_robomme_source.json"
    if any(item["Key"] == marker_key for item in existing):
        raise SystemExit("RoboMME dataset prefix is already sealed; refusing to restage")
    print(
        f"[stage] downloading {HF_REPO}@{HF_REVISION} into the shared HF cache",
        flush=True,
    )
    snapshot = Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            cache_dir=args.hf_cache_dir,
            max_workers=args.download_workers,
        )
    )
    episodes, task_variants, parquet = _snapshot_inventory(snapshot)
    print(
        f"[stage] snapshot verified path={snapshot} episodes={episodes} "
        f"benchmark_tasks={BENCHMARK_TASKS} task_instruction_variants={task_variants} "
        f"parquet={parquet}",
        flush=True,
    )

    # Upload the immutable payload first. The source marker is published last and seals the prefix.
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(snapshot),
            DATA_ROOT,
            "--only-show-errors",
            "--exclude",
            ".cache/*",
        ],
        check=True,
    )
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(norm_stats),
            f"{DATA_ROOT}/assets/robomme/norm_stats.json",
            "--only-show-errors",
        ],
        check=True,
    )
    marker_path = Path(args.inventory_output_dir).resolve() / "_robomme_source.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hf_repo_id": HF_REPO,
                "hf_revision": HF_REVISION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    _publish_once(client, marker_path, f"{DATA_ROOT}/_robomme_source.json")

    path, digest, uri, objects, size = build(
        DATA_ARTIFACT,
        DATA_ROOT,
        STUDY_ROOT,
        args.inventory_output_dir,
        workers=args.inventory_workers,
        allow_unversioned=True,
    )
    _publish_once(client, path, uri)
    print(
        f"[stage] COMPLETE data={DATA_ROOT} inventory={uri} sha256={digest} objects={objects} bytes={size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
