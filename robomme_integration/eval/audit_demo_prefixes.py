"""Measure the real reset-time demonstration prefix on a fixed RoboMME episode map.

This probe intentionally exercises ``EnvRunner.get_init_obs``.  Static task metadata saying that
history is enabled is not enough for FrameSamp experiments: each selected episode must actually
return aligned front images, wrist images, and states before policy execution begins.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
from statistics import fmean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wsm_settings import ROBOMME_EVAL_ROOT  # noqa: E402

SCHEMA_VERSION = 1
SUITES = {
    "VideoPlaceButton": "reference_object",
    "RouteStick": "imitation_procedural",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_episode(task: str, episode: int) -> dict[str, object]:
    # Imported only inside spawned workers.  The official runner deliberately uses a top-level
    # ``utils`` module, so the caller must launch this probe from its released examples directory.
    from env_runner import EnvRunner

    runner = EnvRunner(task, "", max_steps=1_300)
    try:
        seed, difficulty = runner.env_builder.resolve_episode(episode)
        runner.make_env(episode)
        observation = runner.get_init_obs()
        lengths = {
            "front_frames": len(observation["images"]),
            "wrist_frames": len(observation["wrist_images"]),
            "state_frames": len(observation["states"]),
        }
        if len(set(lengths.values())) != 1 or lengths["front_frames"] < 1:
            raise RuntimeError(f"unaligned or empty demonstration prefix: {lengths}")
        return {
            "difficulty": difficulty,
            "episode": episode,
            **lengths,
            "seed": seed,
            "task_goal": observation["task_goal"],
        }
    finally:
        runner.close_env()


def _parse_episodes(value: str) -> tuple[int, ...]:
    if value == "fixed50":
        return tuple(range(50))
    result = tuple(int(part) for part in value.split(","))
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("episodes must be unique nonnegative integers")
    return result


def build_audit(task: str, episodes: tuple[int, ...], workers: int) -> dict[str, object]:
    if task not in SUITES:
        raise ValueError(f"unsupported FS-D0 task {task!r}; expected one of {sorted(SUITES)}")
    if workers < 1:
        raise ValueError("workers must be positive")

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as pool:
        futures = {pool.submit(_probe_episode, task, episode): episode for episode in episodes}
        records = []
        failures = []
        for future in concurrent.futures.as_completed(futures):
            episode = futures[future]
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001 - the sealed audit records any simulator error.
                failures.append({"episode": episode, "error": f"{type(error).__name__}: {error}"})
    if failures:
        raise RuntimeError(f"demo-prefix probe failures: {sorted(failures, key=lambda x: x['episode'])}")

    records.sort(key=lambda record: int(record["episode"]))
    if [record["episode"] for record in records] != list(episodes):
        raise RuntimeError("probe results do not exactly cover the requested episode map")
    lengths = [int(record["front_frames"]) for record in records]
    metadata_path = (
        ROBOMME_EVAL_ROOT
        / "runtime-v0.4.0/robomme-benchmark-f2b540e6/src/robomme/env_metadata/test"
        / f"record_dataset_{task}_metadata.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"pinned test metadata is missing: {metadata_path}")

    result: dict[str, object] = {
        "episode_map": list(episodes),
        "episode_count": len(records),
        "generated_at_utc": os.environ.get("ROBOMME_AUDIT_TIMESTAMP", "UNSET"),
        "kind": "robomme_test_demo_prefix_audit",
        "records": records,
        "runtime": "robomme_eval/runtime-v0.4.0",
        "schema_version": SCHEMA_VERSION,
        "suite": SUITES[task],
        "summary": {
            "all_prefixes_nonempty": all(length > 0 for length in lengths),
            "maximum_frames": max(lengths),
            "mean_frames": fmean(lengths),
            "median_frames": median(lengths),
            "minimum_frames": min(lengths),
        },
        "task": task,
        "test_metadata_sha256": _sha256_file(metadata_path),
    }
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(SUITES))
    parser.add_argument("--episodes", type=_parse_episodes, default=tuple(range(50)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.task, args.episodes, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: audit[key] for key in ("task", "summary", "content_sha256")}))


if __name__ == "__main__":
    main()
