#!/usr/bin/env python3
"""Build the immutable eval episode manifest for the ReMemBench memory benchmark.

ReMemBench episodes cannot be pinned by seed. Two of the thirteen variants
(``MemPutKBreadInMicrowave``, ``MemPutKBowlInCabinet``) sample their object counts from
the *global* ``np.random`` inside ``_setup_kitchen_references``, which no per-env seed
reaches — replaying seed 0 four times yields counts 2, 3, 2, 3. The reproducible pin is
the ``ep_meta`` recorded on every demo group in the ReMemBench hdf5s, replayed through
``env.set_ep_meta()`` before ``reset()``.

``ep_meta`` alone is not sufficient either: it fixes the layout/style, the object
categories and instances, and the sampled counts, but leaves the exact placement to
``env.rng``, which drifts ~0.15 m between replays. Seed *and* ``ep_meta`` together are
bit-identical across replays (verified on MemPutKBreadInMicrowave). Both are therefore
written into every episode record, and ``eval_pi_05.py``'s ``remembench_ep_meta`` reset
kind applies both.

The ``ep_meta`` blobs (~15 KB each) are embedded inline rather than published as side
artifacts, so the manifest is self-contained: its sha256 pins the eval set completely and
no ``--heldout-root`` fetch is needed at eval time.

Split: held-out = the last ``ceil(20%)`` demos per task by demo index, floor 3.

Usage:
    python scripts/launch/build_remembench_episode_manifest.py \
        --data-root ~/Research/TRI/remembench_data --output /tmp/remembench_heldout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vla_training.eval.eval_manifest import seal_episode_manifest  # noqa: E402
from vla_training.eval.remembench_tasks import (  # noqa: E402
    REMEMBENCH_TASKS,
    get_remembench_category,
    get_remembench_horizon,
    heldout_split,
)

#: Every session in Rutav/ReMemBench-Dataset ships this filename (128px, no third-person).
DEMO_FILENAME = "demo_im128_notp.hdf5"

#: Distinct from the RoboCasa heldout50 base seed so the two eval sets never collide.
BASE_SEED = 20260803

#: Held-out tail fraction per task, and its floor.
HELDOUT_FRACTION = 0.2
HELDOUT_MINIMUM = 3

_DEMO_INDEX = re.compile(r"_demo_(\d+)$")


def _demo_index(demo_key: str) -> int:
    """Numeric suffix of e.g. ``MemHeatPot_PandaOmron_demo_12``; the collection order."""
    match = _DEMO_INDEX.search(demo_key)
    if match is None:
        raise ValueError(f"unparseable demo key: {demo_key!r}")
    return int(match.group(1))


def _seed_for(base_seed: int, task: str, episode_index: int) -> int:
    """Stable per-episode seed, independent of hash randomization and shard count.

    Mirrors ``eval_manifest._seed_for`` so ReMemBench seeds are derived the same way as
    RoboCasa's; duplicated rather than imported because that helper is private.
    """
    raw = f"{int(base_seed)}\0{task}\0{int(episode_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def scan_demos(data_root: str | Path) -> dict:
    """``{task: [{session, demo_key, demo_index, length, ep_meta}, ...]}``, ordered.

    Walks ``<data_root>/<task>/<session>/demo_im128_notp.hdf5``. Demos are ordered by
    (session, demo_index) so the held-out tail is a deterministic function of the
    downloaded dataset alone.
    """
    import h5py

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"data root is missing: {root}")

    per_task: dict[str, list] = {}
    for task in REMEMBENCH_TASKS:
        task_dir = root / task
        if not task_dir.is_dir():
            raise ValueError(f"no demos downloaded for {task}: {task_dir} is missing")
        demos = []
        for session_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            path = session_dir / DEMO_FILENAME
            if not path.is_file():
                continue
            with h5py.File(path, "r") as handle:
                group = handle["data"]
                env_args = json.loads(group.attrs["env_args"])
                env_name = str(env_args["env_name"])
                if env_name != task:
                    raise ValueError(f"{path} env_args names {env_name!r}, expected {task!r}")
                for demo_key in group:
                    demo = group[demo_key]
                    demos.append(
                        {
                            "session": session_dir.name,
                            "demo_key": str(demo_key),
                            "demo_index": _demo_index(str(demo_key)),
                            "length": int(demo.attrs["num_samples"]),
                            "ep_meta": json.loads(demo.attrs["ep_meta"]),
                        }
                    )
        if not demos:
            raise ValueError(f"no {DEMO_FILENAME} found under {task_dir}")
        demos.sort(key=lambda record: (record["session"], record["demo_index"]))
        per_task[task] = demos
    return per_task


def build_manifest(
    per_task: dict,
    *,
    base_seed: int = BASE_SEED,
    task_set: str = "remembench",
    split: str = "target",
    fraction: float = HELDOUT_FRACTION,
    minimum: int = HELDOUT_MINIMUM,
) -> tuple[dict, bytes, str]:
    """Seal one manifest over the held-out tail of every task.

    Returns ``(manifest, canonical_bytes, sha256_hex)``.
    """
    entries = []
    selection = {}
    for task, demos in per_task.items():
        train, heldout = heldout_split(range(len(demos)), fraction=fraction, minimum=minimum)
        selection[task] = {
            "n_demos": len(demos),
            "n_train": len(train),
            "n_heldout": len(heldout),
            "category": get_remembench_category(task),
            "heldout_demo_keys": [demos[i]["demo_key"] for i in heldout],
        }
        for episode_index, demo_position in enumerate(heldout):
            demo = demos[demo_position]
            entries.append(
                {
                    "task": task,
                    "split_set": task_set,
                    "category": get_remembench_category(task),
                    "horizon": get_remembench_horizon(task),
                    "episode_index": episode_index,
                    "reset": {
                        "kind": "remembench_ep_meta",
                        "ep_meta": demo["ep_meta"],
                        "source": {
                            "task": task,
                            "session": demo["session"],
                            "demo_key": demo["demo_key"],
                            "demo_index": demo["demo_index"],
                            "demo_length": demo["length"],
                        },
                    },
                    "seed": _seed_for(base_seed, task, episode_index),
                }
            )

    payload = {
        "schema_version": 2,
        "kind": "robocasa_episode_manifest",
        "split": str(split),
        "task_sets": [task_set],
        "base_seed": int(base_seed),
        "policy_noise": {
            "kind": "pi_diffusion_sha256_v1",
            "key_fields": ["episode.seed", "env_step"],
        },
        # Deliberately absent: episodes_per_task. ReMemBench task sizes differ (12 to 56
        # demos), so the held-out tail is 3 to 12 episodes depending on the task; a
        # uniform count would either discard held-out data or oversample the small tasks.
        "selection": {
            "kind": "remembench_tail_fraction",
            "fraction": float(fraction),
            "minimum": int(minimum),
            "per_task": selection,
        },
        "episodes": entries,
    }
    manifest = seal_episode_manifest(payload)
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    return manifest, canonical, hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.path.expanduser("~/Research/TRI/remembench_data"),
        help="root holding <task>/<session>/%s" % DEMO_FILENAME,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--task-set", default="remembench")
    parser.add_argument("--fraction", type=float, default=HELDOUT_FRACTION)
    parser.add_argument("--minimum", type=int, default=HELDOUT_MINIMUM)
    parser.add_argument("--print-split", action="store_true", help="print the per-task split table")
    args = parser.parse_args()

    per_task = scan_demos(args.data_root)
    manifest, payload, digest = build_manifest(
        per_task,
        base_seed=args.base_seed,
        task_set=args.task_set,
        fraction=args.fraction,
        minimum=args.minimum,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_bytes(payload)
    temporary.replace(output)

    if args.print_split:
        selection = manifest["selection"]["per_task"]
        print(f"{'task':32s} {'category':19s} {'demos':>5s} {'train':>5s} {'heldout':>7s} {'horizon':>7s}")
        for task in REMEMBENCH_TASKS:
            record = selection[task]
            print(
                f"{task:32s} {record['category']:19s} {record['n_demos']:5d} "
                f"{record['n_train']:5d} {record['n_heldout']:7d} "
                f"{get_remembench_horizon(task):7d}"
            )
        print(
            f"{'TOTAL':32s} {'':19s} "
            f"{sum(r['n_demos'] for r in selection.values()):5d} "
            f"{sum(r['n_train'] for r in selection.values()):5d} "
            f"{sum(r['n_heldout'] for r in selection.values()):7d}"
        )
        print(f"episodes={len(manifest['episodes'])} bytes={len(payload)}")

    # Two distinct digests, both real: `manifest_sha256` is sealed INSIDE the manifest over
    # its payload (what validate_episode_manifest re-derives and what eval_pi_05 pins
    # against), while the printed digest is over the serialized FILE bytes and is the
    # artifact filename, matching the heldout50 convention.
    print(f"manifest_sha256 (sealed)  {manifest['manifest_sha256']}")
    print(f"file_sha256     (artifact) {digest}")


if __name__ == "__main__":
    main()
