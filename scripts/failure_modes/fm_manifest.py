#!/usr/bin/env python3
"""Build the failure-mode study's 20-reset-per-task manifests, and materialise the
demo-pinned reset artifacts + expert trajectories every downstream stage needs.

WHY DEMO-PINNED RESETS (read this before comparing to sealed numbers)
--------------------------------------------------------------------
The study compares each policy rollout against the *expert demonstration for that reset*.
That only means something if the reset actually is the demo's initial state.

* RoboCasa already works this way: ``reset.kind == "heldout_demo"`` replays
  ``ep_meta.json`` + ``model.xml.gz`` + ``states.npz[0]`` (``vla_training/eval/heldout_reset.py``),
  so the sealed 50x100 protocol and this study agree exactly.
* ReMemBench does NOT. Its sealed reset is ``ep_meta`` + ``seed``; ``ep_meta`` fixes the
  layout, style, object categories and instances, but the *pose* of every object is
  re-drawn from the seeded ``env.rng``. Bit-reproducible across replays, but NOT the
  demo's placement. Under that reset there is no ground-truth expert trajectory.

So this study puts ReMemBench on RoboCasa's footing: every reset here is demo-pinned via
the same ``heldout_reset`` code path, for both benchmarks. Consequence, stated loudly:
**the ReMemBench success rates produced by this study are NOT comparable to the sealed
cb24fe49 numbers** — different reset distribution, n=20 instead of 88x3. They are
comparable *to each other*, which is what a paired failure-mode comparison needs.

RESET SELECTION
---------------
* ReMemBench: demos ordered by ``(session, demo_index)`` exactly as
  ``build_remembench_episode_manifest.scan_demos`` orders them. The sealed held-out tail
  (last ceil(20%), floor 3) is taken first and keeps its sealed ``episode_index`` and seed.
  The remainder of the 20 is topped up from the TRAINING demos immediately preceding that
  tail, so the 20 resets are a contiguous suffix of the ordered demo list. Top-ups get
  ``episode_index = 1000 + k`` and ``seed = seed_for(20260803, task, episode_index)``.
* RoboCasa: the first 20 episodes per task (by ``episode_index``) of the sealed
  100-episode held-out manifest. All held-out; no top-up needed.

Outputs (on the box):
  <work>/manifests/fm_<bench>_manifest.json
  <work>/expert/<bench>/<task>/<reset_id>/{ep_meta.json,model.xml.gz,states.npz,actions.npy}
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_common import (  # noqa: E402
    RC_BASE_SEED,
    RC_TASKS,
    RESETS_PER_TASK,
    RMB_BASE_SEED,
    RMB_TASKS,
    TRAIN_TOPUP_INDEX_BASE,
    reset_id,
    seed_for,
    sha256_file,
    write_json_atomic,
)

DEMO_FILENAME = "demo_im128_notp.hdf5"
HELDOUT_FRACTION = 0.2
HELDOUT_MINIMUM = 3

#: LeRobot -> hdf5/policy action layout. Inverse of ``render_lerobot_shard.ACTION_SEGMENTS``.
#: lerobot = [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
#: policy  = [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
LEROBOT_TO_POLICY = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]

#: ReMemBench evaluation horizons (mirrors vla_training/eval/remembench_tasks.py).
RMB_HORIZONS = {
    "MemFruitInSinkRightFar": 1400,
    "MemHeatPot": 2600,
    "MemWashAndReturnLeft": 1000,
}
RMB_CATEGORIES = {
    "MemFruitInSinkRightFar": "spatial",
    "MemHeatPot": "prospective",
    "MemWashAndReturnLeft": "object_associative",
}


def _demo_index(demo_key: str) -> int:
    return int(demo_key.rsplit("_", 1)[1])


def heldout_count(n: int) -> int:
    return max(HELDOUT_MINIMUM, math.ceil(HELDOUT_FRACTION * n))


# --------------------------------------------------------------------------------------
# ReMemBench
# --------------------------------------------------------------------------------------
def scan_rmb_demos(data_root: str, task: str) -> list:
    """Ordered demo records for one task; ordering identical to the sealed builder."""
    import h5py

    task_dir = os.path.join(data_root, task)
    demos = []
    for session in sorted(os.listdir(task_dir)):
        path = os.path.join(task_dir, session, DEMO_FILENAME)
        if not os.path.isfile(path):
            continue
        with h5py.File(path, "r") as handle:
            for demo_key in handle["data"]:
                demos.append(
                    {
                        "session": session,
                        "path": path,
                        "demo_key": str(demo_key),
                        "demo_index": _demo_index(str(demo_key)),
                        "length": int(handle["data"][demo_key].attrs["num_samples"]),
                    }
                )
    demos.sort(key=lambda record: (record["session"], record["demo_index"]))
    return demos


def materialise_rmb_expert(demo: dict, out_dir: str) -> dict:
    """Write ep_meta.json / model.xml.gz / states.npz / actions.npy for one demo."""
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(demo["path"], "r") as handle:
        group = handle["data"][demo["demo_key"]]
        ep_meta = json.loads(group.attrs["ep_meta"])
        model_xml = group.attrs["model_file"]
        states = np.asarray(group["states"])
        actions = np.asarray(group["actions"])

    with open(os.path.join(out_dir, "ep_meta.json"), "w") as handle:
        json.dump(ep_meta, handle)
    with gzip.open(os.path.join(out_dir, "model.xml.gz"), "wt", encoding="utf-8") as handle:
        handle.write(model_xml)
    np.savez_compressed(os.path.join(out_dir, "states.npz"), states=states)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    return {
        "expert_length": int(len(states)),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "lang": ep_meta.get("lang"),
    }


def build_rmb(data_root: str, work: str, tasks) -> list:
    entries = []
    for task in tasks:
        demos = scan_rmb_demos(data_root, task)
        n_held = heldout_count(len(demos))
        if len(demos) < RESETS_PER_TASK:
            raise SystemExit(f"{task}: only {len(demos)} demos, need {RESETS_PER_TASK}")
        # Contiguous suffix of the ordered list: the sealed held-out tail, plus the
        # training demos immediately before it.
        n_topup = RESETS_PER_TASK - n_held
        heldout = demos[len(demos) - n_held :]
        topup = demos[len(demos) - n_held - n_topup : len(demos) - n_held]

        for episode_index, demo in enumerate(heldout):
            entries.append(_rmb_entry(task, demo, episode_index, "heldout", work))
        for offset, demo in enumerate(topup):
            episode_index = TRAIN_TOPUP_INDEX_BASE + offset
            entries.append(_rmb_entry(task, demo, episode_index, "train", work))
        print(
            f"[rmb] {task}: {len(demos)} demos -> {n_held} heldout + {n_topup} train = {RESETS_PER_TASK}",
            flush=True,
        )
    return entries


def _rmb_entry(task: str, demo: dict, episode_index: int, split: str, work: str) -> dict:
    rid = reset_id(split, episode_index)
    expert_dir = os.path.join(work, "expert", "remembench", task, rid)
    info = materialise_rmb_expert(demo, expert_dir)
    return {
        "bench": "remembench",
        "task": task,
        "category": RMB_CATEGORIES[task],
        "horizon": RMB_HORIZONS[task],
        "episode_index": episode_index,
        "reset_id": rid,
        "split": split,
        "seed": seed_for(RMB_BASE_SEED, task, episode_index),
        "reset": {"kind": "demo_pinned_v1", "extras_dir": expert_dir},
        "expert": {
            "states": os.path.join(expert_dir, "states.npz"),
            "actions": os.path.join(expert_dir, "actions.npy"),
            **info,
        },
        "source": {
            "session": demo["session"],
            "demo_key": demo["demo_key"],
            "demo_index": demo["demo_index"],
            "demo_length": demo["length"],
            "hdf5": demo["path"],
            "hdf5_sha256_prefix": None,
        },
    }


# --------------------------------------------------------------------------------------
# RoboCasa
# --------------------------------------------------------------------------------------
def build_rc(manifest_path: str, heldout_root: str, lerobot_root: str, work: str, tasks) -> list:
    with open(manifest_path, "rb") as handle:
        manifest = json.load(handle)
    by_task = {}
    for record in manifest["episodes"]:
        by_task.setdefault(record["task"], []).append(record)

    entries = []
    for task in tasks:
        records = sorted(by_task[task], key=lambda r: int(r["episode_index"]))
        if len(records) < RESETS_PER_TASK:
            raise SystemExit(f"{task}: only {len(records)} heldout episodes")
        for record in records[:RESETS_PER_TASK]:
            episode_index = int(record["episode_index"])
            rid = reset_id("heldout", episode_index)
            src_extras = os.path.join(heldout_root, task, "extras", f"episode_{episode_index:06d}")
            dst = os.path.join(work, "expert", "robocasa", task, rid)
            os.makedirs(dst, exist_ok=True)
            for name in ("ep_meta.json", "model.xml.gz", "states.npz"):
                if not os.path.exists(os.path.join(dst, name)):
                    shutil.copy2(os.path.join(src_extras, name), os.path.join(dst, name))
            states = np.load(os.path.join(dst, "states.npz"))["states"]
            actions = _rc_expert_actions(lerobot_root, task, episode_index)
            np.save(os.path.join(dst, "actions.npy"), actions)
            entries.append(
                {
                    "bench": "robocasa",
                    "task": task,
                    "category": record.get("split_set"),
                    "horizon": int(record["horizon"]),
                    "episode_index": episode_index,
                    "reset_id": rid,
                    "split": "heldout",
                    "seed": int(record["seed"]),
                    "reset": {"kind": "demo_pinned_v1", "extras_dir": dst},
                    "expert": {
                        "states": os.path.join(dst, "states.npz"),
                        "actions": os.path.join(dst, "actions.npy"),
                        "expert_length": int(len(states)),
                        "state_dim": int(states.shape[1]),
                        "action_dim": int(actions.shape[1]),
                        "lang": json.load(open(os.path.join(dst, "ep_meta.json"))).get("lang"),
                    },
                    "source": {"sealed_reset": record["reset"]},
                }
            )
        print(f"[robocasa] {task}: {RESETS_PER_TASK} heldout resets", flush=True)
    return entries


def _rc_expert_actions(lerobot_root: str, task: str, episode_index: int) -> np.ndarray:
    """Expert actions for one RoboCasa heldout episode, permuted to the policy layout."""
    import pandas as pd

    path = os.path.join(lerobot_root, task, "lerobot", "data", "chunk-000", f"episode_{episode_index:06d}.parquet")
    frame = pd.read_parquet(path, columns=["action"])
    actions = np.stack([np.asarray(row, dtype=np.float64) for row in frame["action"]])
    if actions.shape[1] != 12:
        raise SystemExit(f"{path}: action dim {actions.shape[1]} != 12")
    return actions[:, LEROBOT_TO_POLICY]


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=["remembench", "robocasa"], required=True)
    parser.add_argument("--work", default="/data2/failure_modes")
    parser.add_argument("--rmb-data-root", default="/data/remembench_data")
    parser.add_argument("--rc-manifest", default="/data/work/canonical_episode_manifest.json")
    parser.add_argument("--rc-heldout-root", default="/data/work/heldout_full")
    parser.add_argument("--rc-lerobot-root", default="/data2/failure_modes/rc_lerobot")
    parser.add_argument("--tasks", default=None, help="comma list; default = the study's 3")
    args = parser.parse_args()

    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(RMB_TASKS if args.bench == "remembench" else RC_TASKS)
    )

    if args.bench == "remembench":
        entries = build_rmb(args.rmb_data_root, args.work, tasks)
    else:
        entries = build_rc(args.rc_manifest, args.rc_heldout_root, args.rc_lerobot_root, args.work, tasks)

    out = os.path.join(args.work, "manifests", f"fm_{args.bench}_manifest.json")
    payload = {
        "schema_version": 1,
        "kind": "failure_mode_reset_manifest",
        "bench": args.bench,
        "reset_kind": "demo_pinned_v1",
        "resets_per_task": RESETS_PER_TASK,
        "tasks": tasks,
        "base_seed": RMB_BASE_SEED if args.bench == "remembench" else RC_BASE_SEED,
        "episodes": entries,
    }
    write_json_atomic(out, payload)
    print(f"wrote {out}  episodes={len(entries)}  sha256={sha256_file(out)[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
