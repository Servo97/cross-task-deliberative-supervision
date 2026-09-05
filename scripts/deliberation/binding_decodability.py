#!/usr/bin/env python3
"""H14 Stage-E — BINDING DECODABILITY: the Markovianization signature, measured on ω.

Pre-registered 2026-08-28, BEFORE any v2 cell ran. This is a REPORT metric and a floor, never a
selection metric — selection stays on the retrieval gate plus decode grounding. It exists because
the campaign's headline claim is Markovianization ("current observation + GDN state over ω is a
sufficient statistic"), and that claim has a direct, cheap signature on ω itself.

The test. Four RoboCasa tasks carry a per-episode bound variable that the success predicate reads
and that a single frame cannot reveal until it has been shown:

    SearingMeat, StirVegetables   knob          8 values
    CuttingToolSelection          cut_food      7 values
    RecycleBottlesByType          mystery_type  2 values, recycle_ends 2 values

For each task and slot, fit a NEAREST-CENTROID classifier on frame ω predicting the bound value,
k-fold BY EPISODE (never by frame — frames of one episode share a label and would leak), separately
for frames BEFORE and AFTER the reveal frame.

    reveal = the first frame of the first segment whose pass-1 descriptor NAMES the bound value;
             if no segment names it, the episode's first segment (i.e. "revealed from the start").
    Which of the two applied is recorded per episode as `reveal_source`, because
    "no segment names it" makes the before-window empty and the row uninformative — that has to be
    visible, not averaged away.

The Markovianization signature is `after >> chance` while `before ≈ chance`: ω has absorbed a fact
that the current frame alone does not carry, and did not hallucinate it before it was observable.
A model that scores high BEFORE the reveal is reading episode identity (a shortcut the funnel's own
ctrl-0 is very good at — bevf 0.998), not memory.

Chance is NOT 1/n_values — the slots are unbalanced. Two baselines are reported: the majority-class
rate, and the label-prior expected accuracy, which is what `lift` divides by (the majority rate can
degenerate to 0 on a short before-window and mint an infinite lift).

    python scripts/deliberation/binding_decodability.py \
        --omega ~/Research/TRI/wsm_data/deliberation/stage_e_runs/omega/E1 \
        --bindings ~/Research/TRI/wsm_data/deliberation/binding_annotations/597f3ff5e7cbd6ce \
        --labels ~/Research/TRI/wsm_data/deliberation/stage_e_labels/bd13c1a48f2dc5be
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

#: Frozen: task -> the slots whose value the success predicate reads AND which vary across episodes.
#: PanTransfer is deliberately absent — its only slot (`pan_container_cat`) takes ONE value in all
#: 510 episodes, so "decoding" it is decoding a constant.
GATE_SLOTS = {
    "SearingMeat": ["knob"],
    "StirVegetables": ["knob"],
    "CuttingToolSelection": ["cut_food"],
    "RecycleBottlesByType": ["mystery_type", "recycle_ends"],
}

# ==============================================================================================
# ReMemBench (added 2026-08-28 with the rmb tap; additive — the robocasa path above is untouched)
# ==============================================================================================
#: rmb's memory slots are TASK-LEVEL constants (the side/route IS the task variant), so unlike
#: RoboCasa they cannot be decoded within one task — inside `MemWashAndReturnLeft` every episode
#: has `return_side=left` and the classifier has one class. They vary across the sibling variants
#: of a FAMILY, so the unit here is the family: pool the variants, decode the slot.
#:
#: This makes the task-identity confound explicit rather than hiding it: `return_side` is a
#: bijection with the task name, so a high AFTER score is only evidence of memory to the extent
#: that the BEFORE score is at chance — which is exactly what the before/after split measures, and
#: exactly why this stays a REPORT metric.
#:
#: `MemFruitInSink/target_object` and `MemHeatPot/cook_food` are POSITIVE CONTROLS, not memory
#: slots: both are stated verbatim in the episode instruction, so ω sees them from frame 0 and
#: should decode BEFORE the reveal as well as after. A family whose "hidden" slot behaves like
#: these controls is not testing memory.
RMB_FAMILIES = {
    "MemWashAndReturn/return_side": {
        "slot": "return_side",
        "tasks": ["MemWashAndReturnLeft", "MemWashAndReturnRight", "MemWashAndReturnSameLocation"],
        "kind": "memory",
    },
    "MemFruitInSink/sink_source": {
        "slot": "sink_source",
        "tasks": ["MemFruitInSinkLeftFar", "MemFruitInSinkRightFar"],
        "kind": "memory",
    },
    "MemRetrieveOils/oils_route": {
        "slot": "oils_route",
        "tasks": [
            "MemRetrieveOilsFromCounterLL",
            "MemRetrieveOilsFromCounterLR",
            "MemRetrieveOilsFromCounterRL",
            "MemRetrieveOilsFromCounterRR",
        ],
        "kind": "memory",
    },
    "MemPutK/set_target": {
        "slot": "set_target",
        "tasks": ["MemPutKBowlInCabinet", "MemPutKBreadInMicrowave"],
        "kind": "memory",
    },
    "MemFruitInSink/target_object": {
        "slot": "target_object",
        "tasks": ["MemFruitInSinkLeftFar", "MemFruitInSinkRightFar"],
        "kind": "control_prompt_visible",
    },
    "MemHeatPot/cook_food": {"slot": "cook_food", "tasks": ["MemHeatPot"], "kind": "control_prompt_visible"},
}

#: The needle a rmb value is searched for in the descriptor text. The RoboCasa path searches for the
#: VALUE string itself; rmb's values are enum codes (`left_far`, `origin`) or tuples, which no
#: descriptor ever writes verbatim, so each slot states its surface form once, here, rather than
#: letting `reveal_frame` silently never match and mark every episode `no_segment_names_it`.
RMB_NEEDLE = {
    "return_side": {"left": "left", "right": "right", "origin": "same location"},
    "sink_source": {"left_far": "left", "right_far": "right"},
}


def rmb_needle(slot: str, value) -> str:
    """The descriptor surface form of a rmb binding value (see RMB_NEEDLE)."""
    if slot in RMB_NEEDLE:
        return RMB_NEEDLE[slot].get(str(value), str(value))
    if slot == "oils_route":
        # a route is (first leg, second leg); the FIRST leg is what the opening segments can name
        return str(value[0]) if isinstance(value, (list, tuple)) and value else str(value)
    if slot == "set_target":
        return str(value).split("->")[0].rstrip("s")  # 'bowls->cabinet' -> 'bowl'
    return str(value)


def load_bindings_domain(root: Path, domain: str) -> dict:
    """(task, episode) -> binding, for one domain."""
    out = {}
    with (root / "bindings.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["domain"] != domain or not record.get("binding"):
                continue
            out[(record["task"], int(record["episode"]))] = record["binding"]
    return out


def evaluate_remembench(omega_root: Path, bindings_root: Path, pass1_root: Path, folds: int = 5) -> dict:
    """Family-pooled before/after binding decodability on the ReMemBench half of an ω store."""
    bindings = load_bindings_domain(bindings_root, "remembench")
    report, reveal_sources = {}, Counter()
    for family, spec in RMB_FAMILIES.items():
        slot = spec["slot"]
        before_x, before_y, after_x, after_y = [], [], [], []
        sub_before_x, sub_before_y, sub_after_x, sub_after_y = [], [], [], []
        cuts, empty_before = [], 0
        for task in spec["tasks"]:
            task_dir = omega_root / "remembench" / task
            if not task_dir.exists():
                continue
            for demo in sorted(task_dir.glob("demo_*")):
                episode = int(demo.name.split("_")[1])
                binding = bindings.get((task, episode))
                if not binding or slot not in binding:
                    continue
                value = str(binding[slot])
                blob = np.load(demo / "w.npz")
                omega = blob["w"].astype(np.float32)
                frame_index = blob["frame_indices"].astype(np.int64)
                cut, source = reveal_frame(
                    pass1_root / task / f"ep_{episode:06d}.descriptors.json", rmb_needle(slot, binding[slot])
                )
                reveal_sources[source] += 1
                cuts.append(int(cut))
                before, after = omega[frame_index < cut], omega[frame_index >= cut]
                before_x.append(before)
                before_y.append(value)
                after_x.append(after)
                after_y.append(value)
                if len(before):
                    sub_before_x.append(before)
                    sub_before_y.append(value)
                    sub_after_x.append(after)
                    sub_after_y.append(value)
                else:
                    empty_before += 1
        n = len(before_y)
        row = {
            "kind": spec["kind"],
            "n_episodes": n,
            "n_empty_before_window": empty_before,
            "frac_reveal_at_frame_0": round(empty_before / n, 3) if n else None,
            "median_reveal_frame": int(np.median(cuts)) if cuts else None,
            "before_reveal": nearest_centroid_kfold(before_x, before_y, folds),
            "after_reveal": nearest_centroid_kfold(after_x, after_y, folds),
            # post-hoc SUBPOPULATION: only the episodes whose reveal is actually mid-episode, i.e.
            # the ones the before/after contrast is defined on at all. Labelled, never selected on.
            "nonempty_before_subset": {
                "n_episodes": len(sub_before_y),
                "before_reveal": nearest_centroid_kfold(sub_before_x, sub_before_y, folds),
                "after_reveal": nearest_centroid_kfold(sub_after_x, sub_after_y, folds),
            },
        }
        report[family] = row
    signature = []
    for key, row in report.items():
        after = (row.get("after_reveal") or {}).get("lift")
        before = (row.get("before_reveal") or {}).get("lift")
        if after is None:
            continue
        signature.append(
            {
                "cell_slot": key,
                "kind": row["kind"],
                "after_lift": after,
                "before_lift": before,
                "markovianization": bool(
                    row["kind"] == "memory" and after > 1.1 and (before is None or before < after)
                ),
            }
        )
    return {
        "omega": str(omega_root),
        "domain": "remembench",
        "per_family": report,
        "reveal_source_counts": dict(reveal_sources),
        "signature_rows": signature,
        "note": "REPORT metric and floor, never a selection metric; rmb slots are task-level "
        "constants so the unit is the task FAMILY and task identity is the standing "
        "confound — read AFTER only against BEFORE",
    }


def load_bindings(root: Path) -> dict:
    out = {}
    with (root / "bindings.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["domain"] != "robocasa" or not record.get("binding"):
                continue
            out[(record["task"], int(record["episode"]))] = record["binding"]
    return out


def reveal_frame(descriptor_path: Path, value: str) -> tuple[int, str]:
    """First frame of the first segment whose descriptor text names the bound value."""
    if not descriptor_path.exists():
        return 0, "missing_descriptor"
    blob = json.loads(descriptor_path.read_text())
    needles = [w for w in str(value).replace("_", " ").replace("'", " ").split() if len(w) > 2]
    for entry in blob.get("descriptors", []):
        text = json.dumps(entry.get("descriptor", {})).lower()
        if needles and all(n.lower() in text for n in needles):
            return int(entry["t0"]), "named_in_descriptor"
    segments = blob.get("descriptors") or []
    return (int(segments[0]["t0"]) if segments else 0), "no_segment_names_it"


def nearest_centroid_kfold(features: list, labels: list, folds: int = 5, seed: int = 0) -> dict:
    """k-fold BY EPISODE. features[i]/labels[i] are one episode's frames and its single label."""
    if len(features) < folds * 2:
        return {"n_episodes": len(features), "note": "too few episodes"}
    counts = Counter(labels)
    if len(counts) < 2:
        return {"n_episodes": len(features), "note": f"single class {list(counts)}"}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    hits = total = 0
    chance_hits = 0
    prior_hits = 0.0
    for fold in range(folds):
        test = set(order[fold::folds].tolist())
        train_index = [i for i in range(len(features)) if i not in test]
        by_label = defaultdict(list)
        for i in train_index:
            if len(features[i]):
                by_label[labels[i]].append(features[i].mean(0))
        if len(by_label) < 2:
            continue
        names = sorted(by_label)
        centroids = np.stack([np.mean(by_label[n], 0) for n in names])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        train_counts = Counter(labels[i] for i in train_index)
        majority = train_counts.most_common(1)[0][0]
        prior = {k: v / len(train_index) for k, v in train_counts.items()}
        for i in test:
            frames = features[i]
            if not len(frames):
                continue
            unit = frames / np.maximum(np.linalg.norm(frames, axis=1, keepdims=True), 1e-9)
            predicted = np.asarray(names)[(unit @ centroids.T).argmax(1)]
            hits += int((predicted == labels[i]).sum())
            chance_hits += int(len(frames) * (majority == labels[i]))
            prior_hits += len(frames) * prior.get(labels[i], 0.0)
            total += len(frames)
    if total == 0:
        return {"n_episodes": len(features), "note": "no scorable frames"}
    accuracy = hits / total
    # Two baselines. The majority-class rate is the intuitive one but degenerates to 0 on a small
    # window where the train-majority label happens to miss every test episode, which would mint an
    # infinite lift out of 127 frames. The PRIOR baseline — expected accuracy of guessing from the
    # training label distribution — is smooth, never 0 in practice, and is what `lift` divides by.
    chance_prior = prior_hits / total
    return {
        "n_episodes": len(features),
        "n_frames": int(total),
        "accuracy": round(accuracy, 4),
        "chance_majority": round(chance_hits / total, 4),
        "chance_prior": round(chance_prior, 4),
        "lift": (round(accuracy / chance_prior, 3) if chance_prior > 0.01 else None),
        "n_classes": len(counts),
    }


def evaluate(omega_root: Path, bindings_root: Path, pass1_root: Path, folds: int = 5) -> dict:
    bindings = load_bindings(bindings_root)
    report, reveal_sources = {}, Counter()
    for task, slots in GATE_SLOTS.items():
        task_dir = omega_root / "robocasa" / task
        if not task_dir.exists():
            report[task] = {"skipped": "no omega for this task"}
            continue
        for slot in slots:
            before_x, before_y, after_x, after_y = [], [], [], []
            for demo in sorted(task_dir.glob("demo_*")):
                episode = int(demo.name.split("_")[1])
                binding = bindings.get((task, episode))
                if not binding or slot not in binding:
                    continue
                value = str(binding[slot])
                blob = np.load(demo / "w.npz")
                omega = blob["w"].astype(np.float32)
                frame_index = blob["frame_indices"].astype(np.int64)
                cut, source = reveal_frame(pass1_root / task / f"ep_{episode:06d}.descriptors.json", value)
                reveal_sources[source] += 1
                before = omega[frame_index < cut]
                after = omega[frame_index >= cut]
                before_x.append(before)
                before_y.append(value)
                after_x.append(after)
                after_y.append(value)
            report[f"{task}/{slot}"] = {
                "before_reveal": nearest_centroid_kfold(before_x, before_y, folds),
                "after_reveal": nearest_centroid_kfold(after_x, after_y, folds),
            }
    signature = []
    for key, row in report.items():
        after = (row.get("after_reveal") or {}).get("lift")
        before = (row.get("before_reveal") or {}).get("lift")
        if after is None:
            continue
        signature.append(
            {
                "cell_slot": key,
                "after_lift": after,
                "before_lift": before,
                "markovianization": bool(after is not None and after > 1.1 and (before is None or before < after)),
            }
        )
    return {
        "omega": str(omega_root),
        "per_task": report,
        "reveal_source_counts": dict(reveal_sources),
        "signature_rows": signature,
        "note": "REPORT metric and floor, never a selection metric",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omega", required=True, help="an exported omega store root")
    ap.add_argument("--bindings", required=True)
    ap.add_argument(
        "--pass1",
        default="",
        help="pass-1 descriptor root for --domain (default: the domain's subdir under "
        "~/Research/TRI/wsm_data/deliberation/pass1_store)",
    )
    ap.add_argument("--domain", default="robocasa", choices=("robocasa", "remembench"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    pass1 = (
        Path(args.pass1).expanduser()
        if args.pass1
        else Path(f"~/Research/TRI/wsm_data/deliberation/pass1_store/{args.domain}").expanduser()
    )
    runner = evaluate if args.domain == "robocasa" else evaluate_remembench
    result = runner(Path(args.omega).expanduser(), Path(args.bindings).expanduser(), pass1, args.folds)
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
