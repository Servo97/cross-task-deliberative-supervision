"""Pinned single-task episode manifests for the official RoboMME LeRobot snapshot.

RoboMME's LeRobot metadata contains 116 language-instruction variants for 16 environment tasks.
The environment task name is not stored as a separate per-frame field, so selecting by prompt text
would be ambiguous and brittle.  The pinned dataset is organized as 100 consecutive episodes per
environment task.  We lock that mapping to the exact upstream metadata bytes and fail closed if the
staged snapshot differs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EPISODES_SHA256 = "4b090c292a86386c82f57558e23430066c2e6429424c594a97100b730d7fab27"
TASKS_SHA256 = "8aa71a980c3dd2ce6a4fbede2ca602eb01a0203c5ef2c1bada9d363efeb9a43e"
EPISODES_PER_TASK = 100

# Order in Yinpei/robomme_data_lerobot@1510653c.  It is deliberately not inferred from language:
# multiple environment tasks share similar instructions, especially the four permanence tasks.
TASK_ORDER = (
    "PatternLock",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoPlaceButton",
    "VideoUnmaskSwap",
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "PickHighlight",
    "MoveCube",
    "InsertPeg",
    "RouteStick",
    "BinFill",
    "VideoPlaceOrder",
    "VideoRepick",
    "VideoUnmask",
)
TASK_EPISODES = {
    task: tuple(range(index * EPISODES_PER_TASK, (index + 1) * EPISODES_PER_TASK))
    for index, task in enumerate(TASK_ORDER)
}


def _read_pinned(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"required RoboMME metadata file is missing: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"RoboMME metadata identity mismatch for {path}: {digest} != {expected_sha256}")
    return payload


def select_task_episodes(dataset_root: str | Path, task_name: str) -> tuple[int, ...]:
    """Return the immutable 100-episode training manifest for ``task_name``.

    Both metadata files are content-verified even though episode selection only needs
    ``episodes.jsonl``.  This ties the range mapping to the same instruction table used by
    LeRobot when it materializes each sample's prompt.
    """
    if task_name not in TASK_EPISODES:
        raise ValueError(f"unknown RoboMME task {task_name!r}; expected one of {TASK_ORDER}")
    root = Path(dataset_root)
    episode_bytes = _read_pinned(root / "meta" / "episodes.jsonl", EPISODES_SHA256)
    task_bytes = _read_pinned(root / "meta" / "tasks.jsonl", TASKS_SHA256)

    episodes = [json.loads(line) for line in episode_bytes.splitlines() if line.strip()]
    tasks = [json.loads(line) for line in task_bytes.splitlines() if line.strip()]
    if len(episodes) != len(TASK_ORDER) * EPISODES_PER_TASK:
        raise ValueError(f"expected 1600 RoboMME episode records, found {len(episodes)}")
    episode_indices = [record.get("episode_index") for record in episodes]
    if episode_indices != list(range(len(episodes))):
        raise ValueError("RoboMME episodes must be unique, ordered, and contiguous from zero")
    if len(tasks) != 116 or [record.get("task_index") for record in tasks] != list(range(116)):
        raise ValueError("RoboMME task-instruction table is not the pinned contiguous 116-row table")
    known_prompts = {record.get("task") for record in tasks}
    if None in known_prompts or len(known_prompts) != 116:
        raise ValueError("RoboMME task-instruction strings must be present and unique")
    if any(
        not isinstance(record.get("tasks"), list)
        or len(record["tasks"]) != 1
        or record["tasks"][0] not in known_prompts
        for record in episodes
    ):
        raise ValueError("RoboMME episode metadata contains an invalid instruction reference")

    selected = TASK_EPISODES[task_name]
    if len(selected) != EPISODES_PER_TASK or len(set(selected)) != EPISODES_PER_TASK:
        raise AssertionError(f"internal task manifest for {task_name} is not 100 unique episodes")
    return selected


def task_manifest_sha256(task_name: str) -> str:
    """Stable identity to print into run manifests and logs."""
    if task_name not in TASK_EPISODES:
        raise ValueError(f"unknown RoboMME task {task_name!r}")
    payload = json.dumps(
        {
            "episodes_sha256": EPISODES_SHA256,
            "tasks_sha256": TASKS_SHA256,
            "task_name": task_name,
            "episodes": TASK_EPISODES[task_name],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
