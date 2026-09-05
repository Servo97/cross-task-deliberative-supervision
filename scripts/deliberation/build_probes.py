"""H14 — planted known-CONTRAST probes for edge QA gate G-B (amendment A9, edge_schema.md §7).

A9: order-flip stability measures SELF-CONSISTENCY, which is necessary and not sufficient. A model
that answers CONTRAST to everything is perfectly stable. G-B needs pairs whose ground truth is fixed
by the TASK DEFINITION rather than by the model, so recovery rate is an accuracy measurement.

Ground truth comes from `_check_success` in the RoboCasa task sources (s2 §3), never from
embeddings and never from the model:

  burner_binding       KettleBoiling "place cookware on burner" (correct burner = the one the
                       distractor is NOT on) vs SearingMeat "place pan on burner" (correct burner =
                       the one the INSTRUCTION names). Same verb, same object class, different
                       completion condition -> CONTRAST.
  tool_binding         CuttingToolSelection grasp-knife vs grasp-peeler, from episodes whose
                       hidden `self.food` differs -> CONTRAST.
  accumulator_vs_place ScrubCuttingBoard wipe (contact timer AND swept extent) vs WashLettuce wash
                       (elapsed time only) -> CONTRAST.
  set_completion       PortionHotDogs place-on-plate-1 vs place-on-plate-2 (a duplicate FAILS)
                       -> CONTRAST.
  sanity_positive      two RinseSinkBasin rinse segments, different episodes -> EQUIVALENT.
                       Without positives, a model that always answers CONTRAST would score 1.0.

Probe ids are written to a file the judge never sees; they are injected into ordinary buckets and
are indistinguishable from mined candidates.

  python scripts/deliberation/build_probes.py --store ~/Research/TRI/wsm_data/deliberation/pass2_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# Families are defined over the subskill/object vocabulary the descriptor pass ACTUALLY produces
# (measured on the pilot: place 70, grasp 47, turn 22, reach/wait/navigate 7, open/wipe 4, ...).
# The first draft guessed verbs ("wash", "rinse", "scrub", "spray") that the model never emits, and
# three of five families matched nothing. Guessing a controlled vocabulary is how a QA gate silently
# becomes a no-op, so these are now grounded in the pilot's own histograms.
#
# object   "same"  -> the two segments must bind the SAME object class (that is what makes the pair
#                     deceptive); "diff" -> they must differ; "any" -> unconstrained.
# episode  same-episode pairs are excluded by default; set_completion NEEDS them, because "already
#          placed one on this plate" is a within-episode fact.
FAMILIES = [
    # same verb, same object (a stove knob), different completion condition: KettleBoiling's correct
    # burner is the one the distractor is NOT on; SearingMeat's is the one the INSTRUCTION names.
    dict(
        family="burner_binding",
        truth="CONTRAST",
        task_a="KettleBoiling",
        verbs_a=("turn",),
        task_b="SearingMeat",
        verbs_b=("turn",),
        object="same",
        same_episode=False,
    ),
    # same verb, same object (a faucet handle), different completion: RinseSinkBasin latches three
    # spout orientations; WashLettuce accumulates elapsed wash time.
    dict(
        family="faucet_binding",
        truth="CONTRAST",
        task_a="RinseSinkBasin",
        verbs_a=("turn",),
        task_b="WashLettuce",
        verbs_b=("turn",),
        object="same",
        same_episode=False,
    ),
    # the hidden variable: which tool the food identity selects. Different object by construction.
    dict(
        family="tool_binding",
        truth="CONTRAST",
        task_a="CuttingToolSelection",
        verbs_a=("grasp", "lift", "pick"),
        task_b="CuttingToolSelection",
        verbs_b=("grasp", "lift", "pick"),
        object="diff",
        same_episode=False,
    ),
    # same task, same object (the sponge), same scene: `place` completes on contact, `wipe` completes
    # on a contact TIMER plus a swept-extent threshold. The sharpest Tier-A look-alike we have.
    dict(
        family="accumulator_vs_place",
        truth="CONTRAST",
        task_a="ScrubCuttingBoard",
        verbs_a=("place",),
        # object="any": the pair binds sponge-vs-board across the two legs, so requiring an
        # identical object class filters out exactly the pairs the family is about.
        task_b="ScrubCuttingBoard",
        verbs_b=("wipe",),
        object="any",
        same_episode=True,
    ),
    # placing the second item on a plate that already holds one is a FAILURE; the frames look alike.
    dict(
        family="set_completion",
        truth="CONTRAST",
        task_a="PortionHotDogs",
        verbs_a=("place",),
        task_b="PortionHotDogs",
        verbs_b=("place",),
        object="same",
        same_episode=True,
    ),
    # Without positives, a model that always answers CONTRAST would score 1.0 on every gate above.
    dict(
        family="sanity_positive",
        truth="EQUIVALENT",
        task_a="RinseSinkBasin",
        verbs_a=("turn",),
        task_b="RinseSinkBasin",
        verbs_b=("turn",),
        object="same",
        same_episode=False,
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/Research/TRI/wsm_data/deliberation")
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    store = Path(args.store).expanduser()
    idx = store / "index" / "segments.jsonl"
    if not idx.is_file():
        raise SystemExit(f"missing {idx}; run pass2_deliberate.py --stage index first")
    rows = [json.loads(x) for x in idx.read_text().splitlines() if x.strip()]
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    rng = random.Random(args.seed)
    probes, shortfall = [], []
    for spec in FAMILIES:
        fam, truth = spec["family"], spec["truth"]
        ta, va, tb, vb = spec["task_a"], spec["verbs_a"], spec["task_b"], spec["verbs_b"]
        A = [r for r in by_task.get(ta, []) if str(r["descriptor"].get("subskill", "")).lower() in va]
        B = [r for r in by_task.get(tb, []) if str(r["descriptor"].get("subskill", "")).lower() in vb]
        made, tries, seen = 0, 0, set()
        # deterministic sweep, not rejection sampling: small pools made random draws unreliable
        pairs = [(a, b) for a in A for b in B]
        rng.shuffle(pairs)
        for a, b in pairs:
            if made >= args.per_family:
                break
            tries += 1
            if a["seg_id"] == b["seg_id"]:
                continue
            same_ep = a["episode"] == b["episode"] and a["task"] == b["task"]
            if same_ep and not spec["same_episode"]:
                continue
            if not same_ep and spec["same_episode"] and ta == tb:
                continue  # this family's mechanism is a WITHIN-episode fact
            oa = str(a["descriptor"]["target_object"].get("class", "")).lower()
            ob = str(b["descriptor"]["target_object"].get("class", "")).lower()
            if spec["object"] == "same" and oa != ob:
                continue  # a cross-task CONTRAST is only deceptive if the object class MATCHES
            if spec["object"] == "diff" and oa == ob:
                continue
            key = frozenset((a["seg_id"], b["seg_id"]))
            if key in seen:
                continue
            seen.add(key)
            probes.append(
                {
                    "probe_id": hashlib.blake2b(
                        f"{fam}|{a['seg_id']}|{b['seg_id']}".encode(), digest_size=8
                    ).hexdigest(),
                    "family": fam,
                    "ground_truth": truth,
                    "anchor": a["seg_id"],
                    "candidate": b["seg_id"],
                    "anchor_object": oa,
                    "candidate_object": ob,
                    "anchor_subskill": a["descriptor"]["subskill"],
                    "candidate_subskill": b["descriptor"]["subskill"],
                    "basis": "RoboCasa _check_success (s2 §3); not embedding-derived",
                }
            )
            made += 1
        if made < args.per_family:
            shortfall.append(
                {
                    "family": fam,
                    "made": made,
                    "wanted": args.per_family,
                    "pool_a": len(A),
                    "pool_b": len(B),
                    "candidate_pairs": len(pairs),
                }
            )

    out = store / "probes.json"
    out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "per_family": args.per_family,
                "n_probes": len(probes),
                "by_family": {s["family"]: sum(1 for p in probes if p["family"] == s["family"]) for s in FAMILIES},
                "by_ground_truth": {
                    t: sum(1 for p in probes if p["ground_truth"] == t) for t in ("CONTRAST", "EQUIVALENT")
                },
                "shortfall": shortfall,
                "probes": probes,
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "n_probes": len(probes),
                "by_family": {s["family"]: sum(1 for p in probes if p["family"] == s["family"]) for s in FAMILIES},
                "by_ground_truth": {
                    t: sum(1 for p in probes if p["ground_truth"] == t) for t in ("CONTRAST", "EQUIVALENT")
                },
                "shortfall": shortfall,
                "out": str(out),
            },
            indent=1,
        )
    )
    if shortfall:
        print(
            "\nNOTE: families short of quota need more pass-1 coverage of those tasks/verbs. "
            "A probe set with no sanity_positive members cannot detect an always-CONTRAST model."
        )


if __name__ == "__main__":
    main()
