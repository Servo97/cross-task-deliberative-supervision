#!/usr/bin/env python3
"""Validate the sealed Stage-S per-task language table shared by WSM training, omega gen, and eval.

One vector per task = the mean (over every cached frame of every selected T30 demo) of the frozen
pi tap's canonical-terse ``lang_per_frame``. Because it is task-level conditioning consumed
identically at train, omega-generation, and online-eval time, it is a single content-addressed
artifact whose identity changes if ANY input changes (task text, source features, feature-source
checkpoint, aggregation rule, or the stored bytes).

Manifest schema (strict)::

    {
      "schema_version": 1,
      "artifact": "pi05_stage_s_task_lang_table",
      "global_language_mode": "canonical_terse_task_instruction",
      "aggregation_rule": "<fixed string>",
      "dataset": {"name": "robocasa_target50_t30", "episode_subsample_seed": 0, "demos_per_task": 150},
      "task_prompt_manifest": {"id": "<64hex>", "uri": "s3://.../<id>.json"},
      "source_features": {"manifest_id": "<64hex>"},
      "feature_source": {"checkpoint_inventory_id": "<64hex>"},
      "producing_code": {"sha256": "<64hex>"},
      "table": {"path": "task_lang_table.npz", "size_bytes": 1, "sha256": "<64hex>",
                "dtype": "float16", "shape": [50, 2048], "tasks": ["...50 sorted names..."]}
    }

``table_id`` = SHA-256 of canonical JSON of the whole manifest. The npz contains EXACTLY
``{tasks, lang}`` — an ``expanded`` key (the historical per-task Qwen string) is explicitly
forbidden.

``dataset_name`` and ``expected_tasks`` are parameters (defaults = the RoboCasa target50 contract)
so a second dataset gets its own sealed table; ``dataset.demos_per_task`` may be the uniform int or
the ``{task: count}`` mapping used by the source/omega manifests.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from .stage_s_provenance import canonical_json_sha256, sha256_file
except ImportError:  # direct-script / pytest execution
    from stage_s_provenance import canonical_json_sha256, sha256_file

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT = "pi05_stage_s_task_lang_table"
DATASET_NAME = "robocasa_target50_t30"
GLOBAL_LANGUAGE_MODE = "canonical_terse_task_instruction"
AGGREGATION_RULE = "mean_over_all_frames_of_all_selected_t30_demos_float64acc_stored_float16"
LANGUAGE_DIM = 2048
NPZ_KEYS = frozenset({"tasks", "lang"})
TOP_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "global_language_mode",
        "aggregation_rule",
        "dataset",
        "task_prompt_manifest",
        "source_features",
        "feature_source",
        "producing_code",
        "table",
    }
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _exact_keys(value: object, expected: frozenset[str] | set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _fail(f"{label} must be 64 lowercase hex characters")
    return value  # type: ignore[return-value]


def table_id(manifest: dict) -> str:
    return canonical_json_sha256(manifest)


def load_task_lang_table(
    manifest_path: str | Path,
    *,
    table_path: str | Path | None = None,
    expected_task_names: set[str] | None = None,
    expected_tasks: int = 50,
    verify_sha: bool = True,
    dataset_name: str = DATASET_NAME,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Load the validated task->vector mapping (float16 [2048] per task)."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _exact_keys(manifest, TOP_KEYS, "task-lang-table manifest")
    if manifest["schema_version"] != 1 or manifest["artifact"] != ARTIFACT:
        _fail("task-lang-table manifest schema/artifact mismatch")
    if manifest["global_language_mode"] != GLOBAL_LANGUAGE_MODE:
        _fail(f"task-lang-table must use {GLOBAL_LANGUAGE_MODE!r}")
    if manifest["aggregation_rule"] != AGGREGATION_RULE:
        _fail(f"task-lang-table aggregation_rule must be {AGGREGATION_RULE!r}")
    dataset = _exact_keys(manifest["dataset"], {"name", "episode_subsample_seed", "demos_per_task"}, "dataset")
    if dataset["name"] != dataset_name or dataset["episode_subsample_seed"] != seed:
        _fail("task-lang-table dataset contract mismatch")
    _exact_keys(manifest["task_prompt_manifest"], {"id", "uri"}, "task_prompt_manifest")
    _hex64(manifest["task_prompt_manifest"]["id"], "task_prompt_manifest.id")
    _hex64(
        _exact_keys(manifest["source_features"], {"manifest_id"}, "source_features")["manifest_id"],
        "source_features.manifest_id",
    )
    _hex64(
        _exact_keys(manifest["feature_source"], {"checkpoint_inventory_id"}, "feature_source")[
            "checkpoint_inventory_id"
        ],
        "feature_source.checkpoint_inventory_id",
    )
    _hex64(_exact_keys(manifest["producing_code"], {"sha256"}, "producing_code")["sha256"], "producing_code.sha256")

    table = _exact_keys(manifest["table"], {"path", "size_bytes", "sha256", "dtype", "shape", "tasks"}, "table")
    if table["path"] != "task_lang_table.npz":
        _fail("table.path must be 'task_lang_table.npz'")
    if table["dtype"] != "float16":
        _fail("table.dtype must be float16")
    names = table["tasks"]
    if not isinstance(names, list) or len(names) != expected_tasks:
        _fail(f"table.tasks must enumerate exactly {expected_tasks} tasks")
    if any(not isinstance(t, str) or not t for t in names):
        _fail("table.tasks entries must be non-empty strings")
    if names != sorted(names):
        _fail("table.tasks must be sorted ascending")
    if len(set(names)) != len(names):
        _fail("table.tasks has duplicates")
    if table["shape"] != [expected_tasks, LANGUAGE_DIM]:
        _fail(f"table.shape must be [{expected_tasks},{LANGUAGE_DIM}]")
    _hex64(table["sha256"], "table.sha256")

    npz_path = Path(table_path) if table_path is not None else manifest_path.parent / table["path"]
    if not npz_path.is_file():
        _fail(f"task-lang-table npz is missing: {npz_path}")
    if npz_path.stat().st_size != table["size_bytes"]:
        _fail(f"task-lang-table size mismatch for {npz_path}")
    if verify_sha and sha256_file(npz_path) != table["sha256"]:
        _fail(f"task-lang-table sha256 mismatch for {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        keys = frozenset(archive.files)
        if "expanded" in keys:
            _fail("task-lang-table npz must not contain an 'expanded' key (historical Qwen string)")
        if keys != NPZ_KEYS:
            _fail(f"task-lang-table npz keys must be exactly {sorted(NPZ_KEYS)}; got {sorted(keys)}")
        npz_tasks = [str(t) for t in archive["tasks"]]
        lang = archive["lang"]
    if npz_tasks != names:
        _fail("task-lang-table npz tasks differ from the manifest task list")
    if lang.dtype != np.dtype(np.float16) or lang.shape != (expected_tasks, LANGUAGE_DIM):
        _fail(f"task-lang-table npz lang must be float16 [{expected_tasks},{LANGUAGE_DIM}]")
    if not np.isfinite(np.asarray(lang, dtype=np.float32)).all():
        _fail("task-lang-table npz lang contains NaN or infinity")

    if expected_task_names is not None and set(names) != set(expected_task_names):
        _fail("task-lang-table task set differs from the reference task set")
    return {t: np.asarray(lang[i]) for i, t in enumerate(names)}


def validate_task_lang_table(
    manifest_path: str | Path,
    *,
    table_path: str | Path | None = None,
    expected_task_names: set[str] | None = None,
    expected_tasks: int = 50,
    dataset_name: str = DATASET_NAME,
    seed: int = 0,
) -> dict[str, int]:
    mapping = load_task_lang_table(
        manifest_path,
        table_path=table_path,
        expected_task_names=expected_task_names,
        expected_tasks=expected_tasks,
        dataset_name=dataset_name,
        seed=seed,
    )
    return {"tasks": len(mapping)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--table", default=None)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = validate_task_lang_table(
        args.manifest,
        table_path=args.table,
        expected_tasks=args.expected_tasks,
        dataset_name=args.dataset_name,
        seed=args.seed,
    )
    print(
        f"[stage-s-task-lang-table] verified dataset={args.dataset_name} tasks={summary['tasks']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
