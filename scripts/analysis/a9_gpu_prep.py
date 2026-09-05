"""A9 GPU prep: (a) planted known-CONTRAST probes over the LIVE corpus, packed into ordinary-shaped
buckets; (b) a seeded 150-bucket paired subset of the live mining for a `medium` re-judge.

Both write their own store roots; nothing under the live edge store is touched. The medium subset
reuses the live buckets' EXACT candidate lists and order_seed, so every judged pair is paired 1:1
with a live verdict; `reasoning_effort` alone changes the edge_store_id.

Probe ground truth is by CONSTRUCTION from task definitions (edge_schema.md §7 for RoboCasa; the
ReMemBench / RoboMME families below are documented in `basis`), never from embeddings.

  python scripts/analysis/a9_gpu_prep.py --stage probes
  python scripts/analysis/a9_gpu_prep.py --stage medium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.deliberation.build_probes import FAMILIES as ROBOCASA_FAMILIES  # noqa: E402

# ReMemBench / RoboMME families. Ground truth is the TASK-VARIANT contract: these suites ship
# mirrored variants whose only difference IS the completion condition, so a swap provably fails.
EXTRA_FAMILIES = [
    dict(
        family="rmb_return_side",
        truth="CONTRAST",
        domain="remembench",
        task_a="MemWashAndReturnLeft",
        verbs_a=("place", "put"),
        task_b="MemWashAndReturnRight",
        verbs_b=("place", "put"),
        object="same",
        same_episode=False,
        basis="return destination is the memorised ORIGINAL side; L/R variants swap it",
    ),
    dict(
        family="rmb_sink_side",
        truth="CONTRAST",
        domain="remembench",
        task_a="MemFruitInSinkLeftFar",
        verbs_a=("place", "put"),
        task_b="MemFruitInSinkRightFar",
        verbs_b=("place", "put"),
        object="same",
        same_episode=False,
        basis="target sink location differs (left-far vs right-far); same verb, same object",
    ),
    dict(
        family="rmb_oils_source",
        truth="CONTRAST",
        domain="remembench",
        task_a="MemRetrieveOilsFromCounterLL",
        verbs_a=("grasp", "pick", "lift"),
        task_b="MemRetrieveOilsFromCounterRR",
        verbs_b=("grasp", "pick", "lift"),
        object="same",
        same_episode=False,
        basis="which oil instance is correct is set by the memorised counter position (LL vs RR)",
    ),
    # RoboCasa sanity positives: edge_schema.md §7's `sanity_positive` used RinseSinkBasin, which
    # A10 DROPPED from the corpus (ceilinged), so its pool is empty here. Same construction on two
    # tasks that survived: same task, same verb, same object, different episodes.
    dict(
        family="rc_sanity_positive_knob",
        truth="EQUIVALENT",
        domain="robocasa",
        task_a="KettleBoiling",
        verbs_a=("turn",),
        task_b="KettleBoiling",
        verbs_b=("turn",),
        object="same",
        same_episode=False,
        basis="same task/verb/object across episodes; KB's burner rule is constant within task",
    ),
    dict(
        family="rc_sanity_positive_stir",
        truth="EQUIVALENT",
        domain="robocasa",
        task_a="StirVegetables",
        verbs_a=("stir",),
        task_b="StirVegetables",
        verbs_b=("stir",),
        object="same",
        same_episode=False,
        basis="same task/verb/object across episodes",
    ),
    dict(
        family="rmb_sanity_positive",
        truth="EQUIVALENT",
        domain="remembench",
        task_a="MemPutKBowlInCabinet",
        verbs_a=("place", "put"),
        task_b="MemPutKBowlInCabinet",
        verbs_b=("place", "put"),
        object="same",
        same_episode=False,
        basis="same task, same verb, same object, different episodes",
    ),
    dict(
        family="mme_unmask_swap",
        truth="CONTRAST",
        domain="robomme",
        task_a="ButtonUnmask",
        verbs_a=None,
        task_b="ButtonUnmaskSwap",
        verbs_b=None,
        object="same",
        same_episode=False,
        basis="the Swap variant inverts the observed button->target mapping; identical frames, "
        "opposite correct action",
    ),
    dict(
        family="mme_video_unmask_swap",
        truth="CONTRAST",
        domain="robomme",
        task_a="VideoUnmask",
        verbs_a=None,
        task_b="VideoUnmaskSwap",
        verbs_b=None,
        object="same",
        same_episode=False,
        basis="same as mme_unmask_swap for the video-cued variant",
    ),
    dict(
        family="mme_move_vs_stop",
        truth="CONTRAST",
        domain="robomme",
        task_a="MoveCube",
        verbs_a=None,
        task_b="StopCube",
        verbs_b=None,
        object="any",
        same_episode=False,
        basis="MoveCube completes on reaching a target pose; StopCube completes on HALTING at a "
        "cue -- continuing the same motion fails",
    ),
    dict(
        family="mme_sanity_positive",
        truth="EQUIVALENT",
        domain="robomme",
        task_a="MoveCube",
        verbs_a=None,
        task_b="MoveCube",
        verbs_b=None,
        object="same",
        same_episode=False,
        basis="same task, same verb, same object, different episodes",
    ),
]


def load_index(store: Path) -> list[dict]:
    return [json.loads(x) for x in (store / "index" / "segments.jsonl").read_text().splitlines() if x.strip()]


def build_probes(rows: list[dict], per_family: int, seed: int) -> tuple[list[dict], list[dict]]:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    rng = random.Random(seed)
    probes, shortfall = [], []
    for spec in list(ROBOCASA_FAMILIES) + EXTRA_FAMILIES:
        va, vb = spec.get("verbs_a"), spec.get("verbs_b")

        def pool(task, verbs):
            out = by_task.get(task, [])
            if verbs:
                out = [r for r in out if str(r["descriptor"].get("subskill", "")).lower() in verbs]
            return out

        A, B = pool(spec["task_a"], va), pool(spec["task_b"], vb)
        pairs = [(a, b) for a in A for b in B]
        rng.shuffle(pairs)
        made, seen = 0, set()
        for a, b in pairs:
            if made >= per_family:
                break
            if a["seg_id"] == b["seg_id"]:
                continue
            same_ep = a["episode"] == b["episode"] and a["task"] == b["task"]
            if same_ep and not spec["same_episode"]:
                continue
            if not same_ep and spec["same_episode"] and spec["task_a"] == spec["task_b"]:
                continue
            oa = str(a["descriptor"]["target_object"].get("class", "")).lower()
            ob = str(b["descriptor"]["target_object"].get("class", "")).lower()
            if spec["object"] == "same" and oa != ob:
                continue
            if spec["object"] == "diff" and oa == ob:
                continue
            sa = str(a["descriptor"].get("subskill", "")).lower()
            sb = str(b["descriptor"].get("subskill", "")).lower()
            if not va and sa != sb:
                continue  # verb-free families still need a look-alike: same subskill
            key = frozenset((a["seg_id"], b["seg_id"]))
            if key in seen:
                continue
            seen.add(key)
            probes.append(
                {
                    "probe_id": hashlib.blake2b(
                        f"{spec['family']}|{a['seg_id']}|{b['seg_id']}".encode(), digest_size=8
                    ).hexdigest(),
                    "family": spec["family"],
                    "ground_truth": spec["truth"],
                    "anchor": a["seg_id"],
                    "candidate": b["seg_id"],
                    "anchor_subskill": sa,
                    "candidate_subskill": sb,
                    "anchor_object": oa,
                    "candidate_object": ob,
                    "basis": spec.get("basis", "RoboCasa _check_success (s2 §3); not embedding-derived"),
                }
            )
            made += 1
        if made < per_family:
            shortfall.append(
                {"family": spec["family"], "made": made, "wanted": per_family, "pool_a": len(A), "pool_b": len(B)}
            )
    return probes, shortfall


def stage_probes(args) -> None:
    live = Path(args.live_store).expanduser()
    rows = load_index(live)
    by_id = {r["seg_id"]: r for r in rows}
    probes, shortfall = build_probes(rows, args.per_family, args.seed)

    # pack: group probes by anchor, fill each bucket to K=12 with that anchor's live mined
    # candidates so a probe is indistinguishable from an ordinary candidate.
    live_b = {}
    for line in (live / "mine" / "buckets.jsonl").read_text().splitlines():
        if line.strip():
            b = json.loads(line)
            live_b[b["anchor"]] = b
    groups = defaultdict(list)
    for p in probes:
        groups[p["anchor"]].append(p)

    rng = random.Random(args.seed)
    fillers_pool = [r["seg_id"] for r in rows]

    # family-balanced packing: round-robin over families so no family is starved by the bucket cap,
    # and a probe whose anchor already has a bucket rides along for free.
    by_fam = defaultdict(list)
    for p in probes:
        by_fam[p["family"]].append(p)
    chosen_anchors, placed = [], set()
    fams = sorted(by_fam)
    i = 0
    while len(chosen_anchors) < args.max_buckets:
        progress = False
        for fam in fams:
            if i >= len(by_fam[fam]):
                continue
            p = by_fam[fam][i]
            progress = True
            if p["anchor"] not in placed:
                if len(chosen_anchors) >= args.max_buckets:
                    break
                chosen_anchors.append(p["anchor"])
                placed.add(p["anchor"])
        if not progress:
            break
        i += 1
    order = chosen_anchors

    buckets, kept = [], []
    fam_seen = defaultdict(int)
    for anchor in order:
        ps = groups[anchor]
        cands = [p["candidate"] for p in ps]
        strata = [f"probe_{p['family']}" for p in ps]
        lb = live_b.get(anchor)
        fill = [c for c in (lb["candidates"] if lb else []) if c not in cands]
        fstrat = [s for c, s in zip(lb["candidates"], lb["strata"]) if c not in cands] if lb else []
        while len(cands) + len(fill) < 12:
            g = rng.choice(fillers_pool)
            if g != anchor and g not in cands and g not in fill:
                fill.append(g)
                fstrat.append("filler_random")
        cands = cands + fill[: 12 - len(cands)]
        strata = strata + fstrat[: 12 - len(strata)]
        idx = list(range(len(cands)))
        h = hashlib.blake2b(f"{anchor}|{args.order_seed}".encode(), digest_size=8).digest()
        random.Random(int.from_bytes(h, "big")).shuffle(idx)
        cands = [cands[i] for i in idx]
        strata = [strata[i] for i in idx]
        buckets.append(
            {
                "anchor": anchor,
                "candidates": cands,
                "strata": strata,
                "cosines": [0.0] * len(cands),
                "order_seed": args.order_seed,
            }
        )
        for p in ps:
            if p["candidate"] in cands:
                kept.append(p)
                fam_seen[p["family"]] += 1

    out = Path(args.probe_store).expanduser()
    (out / "index").mkdir(parents=True, exist_ok=True)
    (out / "mine").mkdir(parents=True, exist_ok=True)
    link = out / "index" / "segments.jsonl"
    if not link.exists():
        link.symlink_to(live / "index" / "segments.jsonl")
    with (out / "mine" / "buckets.jsonl").open("w") as f:
        for b in buckets:
            f.write(json.dumps(b) + "\n")
    (out / "probes.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "per_family": args.per_family,
                "order_seed": args.order_seed,
                "n_probes_built": len(probes),
                "n_probes_packed": len(kept),
                "by_family_packed": dict(fam_seen),
                "shortfall": shortfall,
                "n_buckets": len(buckets),
                "probes": kept,
                "all_probes": probes,
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "built": len(probes),
                "packed": len(kept),
                "buckets": len(buckets),
                "by_family_packed": dict(fam_seen),
                "shortfall": shortfall,
                "n_contrast_packed": sum(1 for p in kept if p["ground_truth"] == "CONTRAST"),
                "unused_anchor_segments": len(by_id) and None,
            },
            indent=1,
        )
    )


def stage_medium(args) -> None:
    live = Path(args.live_store).expanduser()
    live_b = [json.loads(x) for x in (live / "mine" / "buckets.jsonl").read_text().splitlines() if x.strip()]
    by_dom = defaultdict(list)
    for b in live_b:
        by_dom[b["anchor"].split("/")[0]].append(b)
    rng = random.Random(args.seed)
    per = args.n_buckets // len(by_dom)
    picked = []
    for dom in sorted(by_dom):
        pool = sorted(by_dom[dom], key=lambda b: b["anchor"])
        picked += rng.sample(pool, min(per, len(pool)))
    i = 0
    allb = sorted(live_b, key=lambda b: b["anchor"])
    have = {b["anchor"] for b in picked}
    while len(picked) < args.n_buckets and i < len(allb):
        if allb[i]["anchor"] not in have:
            picked.append(allb[i])
            have.add(allb[i]["anchor"])
        i += 1

    out = Path(args.medium_store).expanduser()
    (out / "index").mkdir(parents=True, exist_ok=True)
    (out / "mine").mkdir(parents=True, exist_ok=True)
    link = out / "index" / "segments.jsonl"
    if not link.exists():
        link.symlink_to(live / "index" / "segments.jsonl")
    with (out / "mine" / "buckets.jsonl").open("w") as f:
        for b in picked:
            f.write(json.dumps(b) + "\n")
    hist = defaultdict(int)
    for b in picked:
        hist[b["anchor"].split("/")[0]] += 1
    print(
        json.dumps(
            {
                "n_buckets": len(picked),
                "pairs": sum(len(b["candidates"]) for b in picked),
                "by_domain": dict(hist),
                "seed": args.seed,
                "out": str(out),
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("probes", "medium"))
    ap.add_argument("--live-store", default="~/Research/TRI/wsm_data/deliberation/pass2_store")
    ap.add_argument("--probe-store", default="~/Research/TRI/wsm_data/deliberation/a9_probe_store")
    ap.add_argument("--medium-store", default="~/Research/TRI/wsm_data/deliberation/a9_medium_store")
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--max-buckets", type=int, default=44)
    ap.add_argument("--n-buckets", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--order-seed", type=int, default=20260822)
    args = ap.parse_args()
    {"probes": stage_probes, "medium": stage_medium}[args.stage](args)


if __name__ == "__main__":
    main()
