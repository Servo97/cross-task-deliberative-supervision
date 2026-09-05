#!/usr/bin/env python3
"""Merge the per-shard result JSONs of one sharded RoboCerebra eval cell into one results file.

The merge unit is the ``per_trial`` row -- one dict per ``(mode, case, trial)``. Rows are UNIONED on
that key and every rate is RECOMPUTED from the union; per-shard ``success_rate``/``subtask_rate`` are
never averaged, because averaging rates over unequal shard sizes is wrong and averaging them over
equal ones is a coincidence waiting to break when ``trials`` stops dividing by K.

Refusals, all of them things that silently produce a plausible wrong number otherwise:

* two shards claiming the same (mode, case, trial) with DIFFERENT outcomes -- means the shards were
  not disjoint, or seeding was off and the runs diverged;
* shard files whose provenance disagrees on anything that defines the cell (arm, checkpoint,
  encoder, replan, switch_steps, seed, wsm, deterministic_seeding, num_shards);
* a shard set with a hole (missing shard index) or an incomplete shard, unless --allow-partial;
* any shard that ran without ``deterministic_seeding`` while claiming ``num_shards > 1``.

    python merge_eval_shards.py --shards out.shard*.json --out out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Provenance fields that DEFINE the eval cell. Shards must agree on every one of them.
CELL_KEYS = (
    "arm",
    "ckpt_sha",
    "encoder_sha",
    "budget_steps",
    "modes",
    "cases",
    "trials",
    "trial_start",
    "replan",
    "switch_steps",
    "seed",
    "random_actions",
    "wsm",
    "num_shards",
    "deterministic_seeding",
)

#: Per-trial fields that describe HOW LONG it took, not WHAT happened. Two runs of the same trial
#: may legitimately disagree on these and still be the same result, so the conflict check ignores
#: them (and only them).
NON_OUTCOME_KEYS = frozenset(
    {"shard", "wall_s", "env_steps_per_s", "mean_gather_batch", "min_gather_batch", "requests"}
)


def merge(paths: list[Path], *, allow_partial: bool = False) -> dict:
    blobs = []
    for path in paths:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if "per_trial" not in blob:
            raise SystemExit(
                f"{path}: no per_trial rows. This file predates sharded eval; merge it by hand or "
                "re-run the cell with the current harness."
            )
        blobs.append((path, blob))
    if not blobs:
        raise SystemExit("no shard files given")

    reference = blobs[0][1]["provenance"]
    for path, blob in blobs[1:]:
        prov = blob["provenance"]
        differing = [key for key in CELL_KEYS if prov.get(key) != reference.get(key)]
        if differing:
            raise SystemExit(
                f"{path} is not the same eval cell as {blobs[0][0]}: {differing} differ "
                + ", ".join(f"{k}={prov.get(k)!r} vs {reference.get(k)!r}" for k in differing)
            )

    num_shards = int(reference.get("num_shards", 1))
    if num_shards > 1 and not reference.get("deterministic_seeding"):
        raise SystemExit(
            "these shards ran WITHOUT deterministic_seeding: their action noise came from each "
            "server's mutable rng, so they are not the sharded form of any single run. Refusing to "
            "merge numbers that cannot be reproduced."
        )
    seen_shards = sorted({int(blob["provenance"]["shard"]) for _p, blob in blobs})
    if seen_shards != list(range(num_shards)) and not allow_partial:
        raise SystemExit(
            f"expected shards {list(range(num_shards))}, got {seen_shards} (pass --allow-partial to merge anyway)"
        )
    incomplete = [str(p) for p, blob in blobs if not blob.get("complete")]
    if incomplete and not allow_partial:
        raise SystemExit(f"incomplete shard(s): {incomplete} (pass --allow-partial to merge anyway)")

    trials: dict[tuple[str, str, int], dict] = {}
    for path, blob in blobs:
        for row in blob["per_trial"]:
            key = (row["mode"], row["case"], int(row["trial"]))
            previous = trials.get(key)
            if previous is not None:
                comparable = {k: v for k, v in row.items() if k not in NON_OUTCOME_KEYS}
                if {k: v for k, v in previous.items() if k not in NON_OUTCOME_KEYS} != comparable:
                    raise SystemExit(
                        f"{path}: (mode={key[0]} case={key[1]} trial={key[2]}) already merged from "
                        f"shard {previous.get('shard')} with a DIFFERENT outcome "
                        f"{previous} vs {row}. The shards are not disjoint, or their episodes "
                        "diverged -- either way the merged rate would be fiction."
                    )
                continue
            trials[key] = row

    per_case: dict[tuple[str, str], dict] = {}
    for (mode, case, _trial), row in sorted(trials.items()):
        bucket = per_case.setdefault(
            (mode, case),
            {
                "mode": mode,
                "case": case,
                "trials": 0,
                "successes": 0,
                "agent_subtasks": 0,
                "possible_subtasks": 0,
                "num_subtasks": row["num_subtasks"],
                "bddl": row["bddl"],
            },
        )
        bucket["trials"] += 1
        bucket["successes"] += int(row["success"])
        bucket["agent_subtasks"] += int(row["agent_subtasks"])
        bucket["possible_subtasks"] += int(row["possible_subtasks"])
    case_rows = []
    for bucket in per_case.values():
        bucket["success_rate"] = bucket["successes"] / bucket["trials"] if bucket["trials"] else 0.0
        bucket["subtask_rate"] = (
            bucket["agent_subtasks"] / bucket["possible_subtasks"] if bucket["possible_subtasks"] else 0.0
        )
        case_rows.append(bucket)

    by_mode: dict[str, dict] = {}
    for row in case_rows:
        bucket = by_mode.setdefault(
            row["mode"], {"successes": 0, "trials": 0, "agent_subtasks": 0, "possible_subtasks": 0}
        )
        for key in bucket:
            bucket[key] += row[key]

    provenance = dict(reference)
    provenance.pop("shard", None)
    provenance["merged_from"] = [str(p) for p, _b in blobs]
    provenance["merged_shards"] = seen_shards
    provenance["merged_trials"] = len(trials)
    return {
        "provenance": provenance,
        "per_case": case_rows,
        "per_trial": [trials[k] for k in sorted(trials)],
        "by_mode": by_mode,
        "complete": not incomplete and seen_shards == list(range(num_shards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", nargs="+", required=True, help="per-shard results json files")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="merge a shard set with holes / unfinished shards (labels the output)",
    )
    args = parser.parse_args()

    merged = merge([Path(p) for p in args.shards], allow_partial=args.allow_partial)
    Path(args.out).write_text(json.dumps(merged, indent=2))
    for mode, bucket in merged["by_mode"].items():
        print(
            f"{mode}: success {bucket['successes']}/{bucket['trials']} "
            f"subtask {bucket['agent_subtasks']}/{bucket['possible_subtasks']}"
        )
    print(
        f"merged {merged['provenance']['merged_trials']} trials from "
        f"{len(merged['provenance']['merged_from'])} shards -> {args.out} "
        f"(complete={merged['complete']})"
    )


if __name__ == "__main__":
    main()
