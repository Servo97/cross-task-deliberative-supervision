#!/usr/bin/env python3
"""Build the sealed Stage-S per-task language table from the canonical source-feature cache.

For each task, the vector = the mean (float64 accumulation, stored float16) of every
``lang_per_frame`` row of every selected T30 demo declared in the source-feature manifest. The
table is task-level conditioning shared identically by WSM training, omega generation, and online
eval, so it is NOT restricted by the WSM-internal train/val split.

Offline: verifies the source-feature manifest + task-prompt manifest by SHA before reading, writes
``task_lang_table.npz`` + a content-addressed manifest, then validates its own output. Never
uploads.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from . import stage_s_provenance as prov
    from . import validate_stage_s_source_features as sfv
    from . import validate_stage_s_task_lang_table as tlt
    from .validate_stage_s_task_prompts import load_task_prompts
except ImportError:  # direct-script / pytest execution
    import stage_s_provenance as prov  # type: ignore
    import validate_stage_s_source_features as sfv  # type: ignore
    import validate_stage_s_task_lang_table as tlt  # type: ignore
    from validate_stage_s_task_prompts import load_task_prompts  # type: ignore

_PRODUCING_CODE = [
    Path(__file__).resolve(),
    prov.REPO_ROOT / "scripts" / "launch" / "validate_stage_s_task_lang_table.py",
]


def producing_code_sha256() -> str:
    return prov.stage_s_code_sha256(_PRODUCING_CODE)


def build_task_lang_table(
    *,
    source_features_root: str | Path,
    source_features_manifest: str | Path,
    source_features_manifest_sha256: str,
    target_root: str | Path,
    labels_root: str | Path,
    task_prompt_manifest: str | Path,
    task_prompt_manifest_sha256: str,
    task_prompt_manifest_uri: str,
    feature_source_inventory_id: str,
    output_dir: str | Path,
    study_root: str,
    expected_tasks: int = 50,
    demos_per_task=150,
    seed: int = 0,
    dataset_name: str = tlt.DATASET_NAME,
    task_dir_globs=sfv.DEFAULT_TASK_DIR_GLOBS,
    task_prompt_artifact: str | None = None,
) -> tuple[Path, str, str, dict]:
    source_features_root = Path(source_features_root).resolve()
    source_features_manifest = Path(source_features_manifest).resolve()
    target_root = Path(target_root).resolve()
    labels_root = Path(labels_root).resolve()
    task_prompt_manifest = Path(task_prompt_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    if not study_root.startswith("s3://") or study_root.endswith("/"):
        raise ValueError("study_root must be an s3:// URI without a trailing slash")

    if prov.sha256_file(source_features_manifest) != source_features_manifest_sha256:
        raise ValueError("source-feature manifest sha256 does not match the file bytes")
    if prov.sha256_file(task_prompt_manifest) != task_prompt_manifest_sha256:
        raise ValueError("task-prompt manifest sha256 does not match the file bytes")

    # Validate the source features fully, then bind the loaded manifest for iteration.
    sfv.validate_source_manifest(
        features_root=source_features_root,
        target_root=target_root,
        labels_root=labels_root,
        manifest_path=source_features_manifest,
        expected_task_prompt_manifest_id=task_prompt_manifest_sha256,
        expected_feature_source_inventory_id=feature_source_inventory_id,
        expected_tasks=expected_tasks,
        demos_per_task=demos_per_task,
        seed=seed,
        dataset_name=dataset_name,
        task_dir_globs=task_dir_globs,
    )
    import json

    manifest = json.loads(source_features_manifest.read_text(encoding="utf-8"))
    source_manifest_id = sfv.source_manifest_id(manifest)
    prompt_map = load_task_prompts(
        task_prompt_manifest,
        target_root=target_root,
        expected_tasks=expected_tasks,
        task_dir_globs=task_dir_globs,
        **({"expected_artifact": task_prompt_artifact} if task_prompt_artifact else {}),
    )

    names = sorted(prompt_map)
    lang = np.zeros((len(names), tlt.LANGUAGE_DIM), dtype=np.float16)
    for row in manifest["tasks"]:
        task = row["task"]
        acc = np.zeros(tlt.LANGUAGE_DIM, dtype=np.float64)
        count = 0
        for episode in row["episodes"]:
            feats_path = source_features_root / episode["feats"]["path"]
            with np.load(feats_path, allow_pickle=False) as archive:
                per_frame = np.asarray(archive["lang_per_frame"], dtype=np.float64)
            acc += per_frame.sum(axis=0)
            count += per_frame.shape[0]
        if count == 0:
            raise ValueError(f"task {task!r} has zero frames; cannot form a language vector")
        lang[names.index(task)] = (acc / count).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "task_lang_table.npz"
    tmp_npz = output_dir / ".task_lang_table.npz.incomplete"
    with tmp_npz.open("wb") as stream:
        np.savez(stream, tasks=np.array(names), lang=lang)
    os.replace(tmp_npz, npz_path)

    manifest_out = {
        "schema_version": 1,
        "artifact": tlt.ARTIFACT,
        "global_language_mode": tlt.GLOBAL_LANGUAGE_MODE,
        "aggregation_rule": tlt.AGGREGATION_RULE,
        "dataset": {
            "name": dataset_name,
            "episode_subsample_seed": seed,
            "demos_per_task": sfv.canonical_demos_per_task(demos_per_task),
        },
        "task_prompt_manifest": {"id": task_prompt_manifest_sha256, "uri": task_prompt_manifest_uri},
        "source_features": {"manifest_id": source_manifest_id},
        "feature_source": {"checkpoint_inventory_id": feature_source_inventory_id},
        "producing_code": {"sha256": producing_code_sha256()},
        "table": {
            "path": "task_lang_table.npz",
            "size_bytes": npz_path.stat().st_size,
            "sha256": prov.sha256_file(npz_path),
            "dtype": "float16",
            "shape": [len(names), tlt.LANGUAGE_DIM],
            "tasks": names,
        },
    }
    table_id = tlt.table_id(manifest_out)
    manifest_path = output_dir / f"{table_id}.json"
    tmp_manifest = output_dir / f".{table_id}.json.incomplete"
    tmp_manifest.write_text(prov.canonical_json(manifest_out) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    tlt.validate_task_lang_table(
        manifest_path,
        expected_task_names=set(names),
        expected_tasks=expected_tasks,
        dataset_name=dataset_name,
        seed=seed,
    )
    uri = f"{study_root}/manifests/artifacts/workspace/task_lang_table/{table_id}.json"
    return manifest_path, table_id, uri, manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features-root", required=True)
    parser.add_argument("--source-features-manifest", required=True)
    parser.add_argument("--source-features-manifest-sha256", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--task-prompt-manifest", required=True)
    parser.add_argument("--task-prompt-manifest-sha256", required=True)
    parser.add_argument("--task-prompt-manifest-uri", required=True)
    parser.add_argument("--feature-source-inventory-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--task-prompt-artifact", default=None)
    sfv.add_dataset_shape_args(parser)
    args = parser.parse_args()
    path, digest, uri, manifest = build_task_lang_table(
        source_features_root=args.source_features_root,
        source_features_manifest=args.source_features_manifest,
        source_features_manifest_sha256=args.source_features_manifest_sha256,
        target_root=args.target_root,
        labels_root=args.labels_root,
        task_prompt_manifest=args.task_prompt_manifest,
        task_prompt_manifest_sha256=args.task_prompt_manifest_sha256,
        task_prompt_manifest_uri=args.task_prompt_manifest_uri,
        feature_source_inventory_id=args.feature_source_inventory_id,
        output_dir=args.output_dir,
        study_root=args.study_root,
        task_prompt_artifact=args.task_prompt_artifact,
        **sfv.dataset_shape_kwargs(args),
    )
    print(f"path={path}")
    print(f"table_id={digest}")
    print(f"canonical_uri={uri}")
    print(f"tasks={len(manifest['table']['tasks'])}")
    print("upload=false")


if __name__ == "__main__":
    main()
