"""Download only exact reset artifacts selected by a schema-v2 episode manifest."""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vla_training.eval.eval_manifest import load_episode_manifest

REQUIRED_ARTIFACTS = ("ep_meta.json", "model.xml.gz", "states.npz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection(manifest_path: str | Path, requested_tasks: list[str]) -> dict:
    """Validate download-relevant fields and group exact episode records per task."""
    manifest = load_episode_manifest(manifest_path)
    requested = set(requested_tasks)
    if len(requested) != len(requested_tasks) or not requested:
        raise ValueError("requested tasks must be a non-empty duplicate-free list")
    unsafe = sorted(
        task
        for task in requested
        if not isinstance(task, str)
        or not task
        or task in {".", ".."}
        or "/" in task
        or "\\" in task
        or "\x00" in task
    )
    if unsafe:
        raise ValueError(f"requested tasks contain unsafe path components: {unsafe!r}")
    grouped = {task: [] for task in requested_tasks}
    for position, record in enumerate(manifest.get("episodes", [])):
        task = record.get("task")
        if task not in requested:
            continue
        episode_index = record["episode_index"]
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError(f"manifest episode {position} has invalid episode_index")
        reset = record.get("reset")
        if not isinstance(reset, dict) or reset.get("kind") != "heldout_demo":
            raise ValueError(f"manifest episode {position} is not heldout_demo")
        expected_relpath = f"{task}/extras/episode_{episode_index:06d}"
        if reset.get("extras_relpath") != expected_relpath:
            raise ValueError(
                f"manifest episode {position} extras_relpath={reset.get('extras_relpath')!r}; "
                f"expected {expected_relpath!r}"
            )
        source = reset.get("source")
        if not isinstance(source, str) or not source.startswith("s3://"):
            raise ValueError(f"manifest episode {position} has invalid reset source")
        artifacts = reset.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_ARTIFACTS):
            raise ValueError(f"manifest episode {position} has incomplete artifacts")
        for filename, descriptor in artifacts.items():
            if not isinstance(descriptor, dict):
                raise ValueError(f"manifest episode {position} {filename} descriptor is invalid")
            digest = descriptor.get("sha256")
            size = descriptor.get("size")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(size) is not int
                or size < 0
            ):
                raise ValueError(f"manifest episode {position} {filename} identity is invalid")
        grouped[task].append({"episode_index": episode_index, "source": source.rstrip("/"), "artifacts": artifacts})

    for task, records in grouped.items():
        if not records:
            raise ValueError(f"manifest has no selected episodes for requested task {task}")
        indices = [record["episode_index"] for record in records]
        if len(indices) != len(set(indices)):
            raise ValueError(f"manifest has duplicate selected episodes for {task}")
        sources = {record["source"] for record in records}
        if len(sources) != 1:
            raise ValueError(f"manifest has multiple source prefixes for {task}: {sorted(sources)}")
        records.sort(key=lambda record: record["episode_index"])
    return grouped


def _verify_episode(directory: Path, artifacts: dict) -> None:
    for filename in REQUIRED_ARTIFACTS:
        path = directory / filename
        descriptor = artifacts[filename]
        if not path.is_file():
            raise ValueError(f"selected reset artifact is missing: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != descriptor["size"] or digest != descriptor["sha256"]:
            raise ValueError(
                f"selected reset artifact mismatch {path}: size={size}/"
                f"{descriptor['size']} sha256={digest}/{descriptor['sha256']}"
            )


def _prepare_task(root: Path, task: str, records: list[dict]) -> tuple[str, int]:
    extras = root / task / "extras"
    extras.mkdir(parents=True, exist_ok=True)
    command = [
        "aws",
        "s3",
        "sync",
        f"{records[0]['source']}/extras",
        str(extras),
        "--only-show-errors",
        "--exclude",
        "*",
    ]
    for record in records:
        command.extend(("--include", f"episode_{record['episode_index']:06d}/*"))
    # One list/sync operation per task instead of 300 individual object requests.
    subprocess.run(command, check=True)
    for record in records:
        _verify_episode(extras / f"episode_{record['episode_index']:06d}", record["artifacts"])
    return task, len(records)


def prepare_selected_heldout(
    root: str | Path,
    manifest_path: str | Path,
    tasks: list[str],
    *,
    workers: int = 16,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root)
    selection = load_selection(manifest_path, tasks)
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = [pool.submit(_prepare_task, root, task, selection[task]) for task in tasks]
        for future in futures:
            task, count = future.result()
            print(f"[prep-heldout] {task}: exact selected={count} verified", flush=True)
