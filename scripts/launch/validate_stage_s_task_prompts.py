#!/usr/bin/env python3
"""Validate the canonical demo-independent terse prompt manifest shared by train and eval.

``artifact`` / ``expected_tasks`` / ``task_dir_globs`` are parameters so a second dataset can ship
its own prompt manifest (ReMemBench: ``remembench_train13_task_prompts``, 13 tasks, flat
``*/*/lerobot`` layout). Every default is the RoboCasa target50 contract.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

ARTIFACT = "robocasa_target50_task_prompts"
GLOBAL_LANGUAGE_MODE = "canonical_terse_task_instruction"
TOP_KEYS = {"schema_version", "artifact", "global_language_mode", "demo_derived", "tasks"}
#: Kept local (this module must stay importable standalone on a node); identical to the tuple in
#: validate_stage_s_policy_features.DEFAULT_TASK_DIR_GLOBS.
DEFAULT_TASK_DIR_GLOBS = ("atomic/*/*/lerobot", "composite/*/*/lerobot")


def _task_dirs(target_root: Path, *, task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS) -> set[str]:
    paths: list[Path] = []
    for pattern in task_dir_globs:
        paths += sorted(target_root.glob(pattern))
    return {path.parents[1].name for path in paths}


def load_task_prompts(
    manifest_path: str | Path,
    *,
    target_root: str | Path | None = None,
    expected_task_names: set[str] | list[str] | tuple[str, ...] | None = None,
    expected_tasks: int = 50,
    expected_artifact: str = ARTIFACT,
    task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS,
) -> dict[str, str]:
    """Load the exact validated task→prompt mapping used by the frozen Stage-S tap."""
    if (target_root is None) == (expected_task_names is None):
        raise ValueError("provide exactly one of target_root or expected_task_names")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != TOP_KEYS:
        raise ValueError(f"task-prompt manifest keys must be exactly {sorted(TOP_KEYS)}")
    if manifest["schema_version"] != 1 or manifest["artifact"] != expected_artifact:
        raise ValueError("task-prompt manifest schema/artifact mismatch")
    if manifest["global_language_mode"] != GLOBAL_LANGUAGE_MODE:
        raise ValueError(f"task prompts must use {GLOBAL_LANGUAGE_MODE!r}, got {manifest['global_language_mode']!r}")
    if manifest["demo_derived"] is not False:
        raise ValueError("task prompts must declare demo_derived=false")
    records = manifest["tasks"]
    if not isinstance(records, list) or len(records) != expected_tasks:
        raise ValueError(f"task-prompt manifest must enumerate exactly {expected_tasks} tasks")
    prompts: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"task", "prompt"}:
            raise ValueError(f"task-prompt record {index} must contain task,prompt exactly")
        task = record["task"]
        prompt = record["prompt"]
        if not isinstance(task, str) or not task or "/" in task or task in prompts:
            raise ValueError(f"invalid or duplicate task-prompt task {task!r}")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or prompt != prompt.strip()
            or "\n" in prompt
            or len(prompt) > 512
        ):
            raise ValueError(f"task {task!r} has invalid canonical terse prompt")
        prompts[task] = prompt
    reference_tasks = (
        _task_dirs(Path(target_root), task_dir_globs=task_dir_globs)
        if target_root is not None
        else set(expected_task_names or ())
    )
    if len(reference_tasks) != expected_tasks or set(prompts) != reference_tasks:
        raise ValueError(
            "task-prompt task set differs from the exact reference task set; "
            f"missing={sorted(reference_tasks - set(prompts))} "
            f"extra={sorted(set(prompts) - reference_tasks)}"
        )
    return prompts


def validate_task_prompts(
    manifest_path: str | Path,
    *,
    target_root: str | Path | None = None,
    expected_task_names: set[str] | list[str] | tuple[str, ...] | None = None,
    expected_tasks: int = 50,
    expected_artifact: str = ARTIFACT,
    task_dir_globs: Sequence[str] = DEFAULT_TASK_DIR_GLOBS,
) -> dict[str, int]:
    prompts = load_task_prompts(
        manifest_path,
        target_root=target_root,
        expected_task_names=expected_task_names,
        expected_tasks=expected_tasks,
        expected_artifact=expected_artifact,
        task_dir_globs=task_dir_globs,
    )
    return {"tasks": len(prompts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--artifact", default=ARTIFACT)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument(
        "--task-dir-glob",
        action="append",
        default=None,
        dest="task_dir_glob",
        help=f"repeatable task-directory glob (default: {' and '.join(DEFAULT_TASK_DIR_GLOBS)})",
    )
    args = parser.parse_args()
    summary = validate_task_prompts(
        args.manifest,
        target_root=args.target_root,
        expected_tasks=args.expected_tasks,
        expected_artifact=args.artifact,
        task_dir_globs=tuple(args.task_dir_glob or DEFAULT_TASK_DIR_GLOBS),
    )
    print(
        f"[stage-s-task-prompts] verified mode={GLOBAL_LANGUAGE_MODE} artifact={args.artifact} "
        f"tasks={summary['tasks']} demo_derived=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
