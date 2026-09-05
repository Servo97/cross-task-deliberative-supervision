#!/usr/bin/env python3
"""Derive the 13-task ReMemBench canonical prompt manifest (Stage-S ``task_prompts`` analogue).

Mirrors ``scripts/launch/build_stage_s_task_prompts.py``: one canonical terse instruction per
task, derived from the per-episode instructions, canonical JSON + trailing newline, filename =
sha256 of those bytes.

Two deliberate departures, both forced by ReMemBench:

* ``artifact`` is ``remembench_train13_task_prompts``. The Stage-S validator pins
  ``robocasa_target50_task_prompts`` and a 50-task count as module constants.
* Nine tasks have a single instruction across all train episodes and derive cleanly. The other
  four vary per episode -- the fruit identity (``MemFruitInSink*``) or the ingredient plus the
  cook duration (``MemHeatPot*``) -- so the upstream builder's "exactly one distinct prompt"
  rule fails and an override is required. The overrides below genericise the varying slot in the
  same way RoboCasa's own ``PickPlaceCounterToCabinet`` prompt says "the object". THEY ARE A
  PROPOSAL AND NEED REVIEW: for ``MemHeatPot*`` the duration is part of the task, so collapsing
  it to "the specified time" removes information the task-level conditioning vector would
  otherwise carry.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vla_training.eval.remembench_tasks import REMEMBENCH_TASKS  # noqa: E402

ARTIFACT = "remembench_train13_task_prompts"
GLOBAL_LANGUAGE_MODE = "canonical_terse_task_instruction"

#: PROPOSED overrides for the four tasks whose episode language is not single-valued.
PROPOSED_OVERRIDES = {
    "MemFruitInSinkLeftFar": "Pick up the fruit and place it in the sink.",
    "MemFruitInSinkRightFar": "Pick up the fruit and place it in the sink.",
    "MemHeatPot": ("Turn on the stove, cook the food for the specified time, and turn off the stove."),
    "MemHeatPotMultiple": (
        "Turn on the stove with the first ingredient, add the second ingredient after the "
        "specified delay, and turn off the stove after the specified time."
    ),
}


def derive(worklist: dict, overrides: dict) -> dict:
    tasks = []
    report = []
    for task in worklist["tasks"]:
        counts = collections.Counter(episode["lang"] for episode in task["episodes"])
        if len(counts) == 1:
            prompt = next(iter(counts))
            source = "derived"
        elif task["task"] in overrides:
            prompt = overrides[task["task"]]
            source = "override"
        else:
            raise SystemExit(f"{task['task']} has {len(counts)} distinct instructions and no override")
        if not prompt or prompt != prompt.strip() or "\n" in prompt or len(prompt) > 512:
            raise SystemExit(f"{task['task']}: invalid prompt {prompt!r}")
        tasks.append({"task": task["task"], "prompt": prompt})
        report.append((task["task"], source, len(counts), prompt))
    names = {record["task"] for record in tasks}
    if names != set(REMEMBENCH_TASKS):
        raise SystemExit("prompt task set does not equal the 13 ReMemBench tasks")
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "global_language_mode": GLOBAL_LANGUAGE_MODE,
        "demo_derived": False,
        "tasks": sorted(tasks, key=lambda record: record["task"]),
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    worklist = json.loads(Path(args.worklist).read_text())
    manifest, report = derive(worklist, PROPOSED_OVERRIDES)
    data = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{digest}.json"
    if destination.exists() and destination.read_bytes() != data:
        raise SystemExit(f"content-addressed collision: {destination}")
    destination.write_bytes(data)
    for task, source, distinct, prompt in report:
        print(f"{task:32s} {source:8s} distinct={distinct:2d}  {prompt}")
    print(f"path={destination}")
    print(f"sha256={digest}")
    print("upload=false  # needs review of the 4 overrides + a ReMemBench prompt validator")


if __name__ == "__main__":
    main()
