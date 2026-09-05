#!/usr/bin/env python3
"""Build the demo-independent canonical terse RoboCasa task-prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from .validate_stage_s_task_prompts import (
        ARTIFACT,
        GLOBAL_LANGUAGE_MODE,
        validate_task_prompts,
    )
except ImportError:
    from validate_stage_s_task_prompts import (
        ARTIFACT,
        GLOBAL_LANGUAGE_MODE,
        validate_task_prompts,
    )


def _task_dirs(target_root: Path) -> dict[str, Path]:
    paths = sorted(target_root.glob("atomic/*/*/lerobot"))
    paths += sorted(target_root.glob("composite/*/*/lerobot"))
    tasks: dict[str, Path] = {}
    for path in paths:
        task = path.parents[1].name
        if task in tasks:
            raise ValueError(f"duplicate target task directory {task!r}")
        tasks[task] = path
    return tasks


def _episode_language(dataset_dir: Path) -> Counter[str]:
    metadata = dataset_dir / "meta" / "episodes.jsonl"
    if not metadata.is_file():
        raise ValueError(f"task episode metadata is missing: {metadata}")
    counts: Counter[str] = Counter()
    with metadata.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            tasks = json.loads(line).get("tasks")
            if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str) or not tasks[0].strip():
                raise ValueError(f"{metadata}:{line_number} must contain one task string")
            prompt = tasks[0].strip()
            if prompt != tasks[0] or "\n" in prompt or len(prompt) > 512:
                raise ValueError(f"{metadata}:{line_number} has an invalid task prompt")
            counts[prompt] += 1
    if not counts:
        raise ValueError(f"task episode metadata has no prompts: {metadata}")
    return counts


def derive_prompts(
    target_root: str | Path,
    *,
    expected_tasks: int = 50,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    target_root = Path(target_root).resolve()
    tasks = _task_dirs(target_root)
    if len(tasks) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} target tasks; found {len(tasks)}")
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(tasks)
    if unknown:
        raise ValueError(f"prompt overrides contain unknown tasks: {sorted(unknown)}")
    prompts: dict[str, str] = {}
    for task, dataset_dir in sorted(tasks.items()):
        counts = _episode_language(dataset_dir)
        if task in overrides:
            prompt = overrides[task]
            if not isinstance(prompt, str):
                raise ValueError(f"prompt override for {task!r} must be a string")
            prompt = prompt.strip()
        elif len(counts) == 1:
            prompt = next(iter(counts))
        else:
            raise ValueError(
                f"task {task!r} has multiple episode prompts {dict(counts)}; supply an explicit reviewed override"
            )
        if not prompt or "\n" in prompt or len(prompt) > 512:
            raise ValueError(f"canonical prompt for {task!r} is invalid")
        prompts[task] = prompt
    return prompts


def build_task_prompt_manifest(
    target_root: str | Path,
    *,
    output_dir: str | Path,
    study_root: str,
    expected_tasks: int = 50,
    overrides: dict[str, str] | None = None,
) -> tuple[Path, str, str, dict]:
    if not study_root.startswith("s3://") or study_root.endswith("/"):
        raise ValueError("study_root must be an s3:// URI without a trailing slash")
    prompts = derive_prompts(target_root, expected_tasks=expected_tasks, overrides=overrides)
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "global_language_mode": GLOBAL_LANGUAGE_MODE,
        "demo_derived": False,
        "tasks": [{"task": task, "prompt": prompt} for task, prompt in sorted(prompts.items())],
    }
    data = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{digest}.json"
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(f"content-addressed task-prompt collision: {destination}")
    temporary = output / f".{digest}.json.incomplete"
    temporary.write_bytes(data)
    temporary.replace(destination)
    validate_task_prompts(destination, target_root=target_root, expected_tasks=expected_tasks)
    uri = f"{study_root}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{digest}.json"
    return destination, digest, uri, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument(
        "--overrides",
        default=None,
        help="optional reviewed JSON object mapping task directory names to canonical prompts",
    )
    args = parser.parse_args()
    overrides = None
    if args.overrides:
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise SystemExit("--overrides must contain a JSON object")
    path, digest, uri, manifest = build_task_prompt_manifest(
        args.target_root,
        output_dir=args.output_dir,
        study_root=args.study_root,
        overrides=overrides,
    )
    print(f"path={path}")
    print(f"sha256={digest}")
    print(f"canonical_uri={uri}")
    print(f"tasks={len(manifest['tasks'])}")
    print("upload=false")


if __name__ == "__main__":
    main()
