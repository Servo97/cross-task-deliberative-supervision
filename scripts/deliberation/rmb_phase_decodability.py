#!/usr/bin/env python3
"""H14 Stage-E — ReMemBench PHASE decodability: the Markovianization signature, re-specified.

Pre-registered 2026-08-28 by coordinator ruling, BEFORE the rmb tap produced a single ω, and it
REPLACES the before/after reveal gate for ReMemBench. Report metrics only; nothing is selected on
them.

Why the reveal gate was withdrawn. `binding_decodability.py` defines reveal = "first segment whose
pass-1 DESCRIPTOR text names the bound value". Measured on the frozen rmb descriptor store that cut
lands at frame 0 for 70-100% of episodes (MemPutK 84/84, MemFruitInSink 35/40, MemWashAndReturn
56/80), because a descriptor of the opening segment already says "blue plate LEFT of sink". The
before-window is then empty and the contrast is undefined — and where it survives, the subset is 24
episodes of a single class. The definition also reads TEXT a VLM wrote, so it moves whenever the
descriptor store is regenerated.

What replaces it.

STEP 1 — slot classification (evidence = the instruction text in meta/episodes.jsonl and the task
definition in ReMemBench robocasa/environments/kitchen/memory/memory_env.py):

  LANGUAGE_GIVEN                   the value is stated in the instruction, so it is visible in
                                   language on EVERY frame and nothing about it is hidden.
  NOT_ACTION_RELEVANT              the value is an observable layout constant that the success
                                   predicate never reads — there is no frame at which the policy
                                   needs it, so there is no USE phase and no memory demand.
  OBSERVATION_GIVEN_THEN_OCCLUDED  the value is observable in the opening frames, and the action
                                   that depends on it happens later, when the current frame no
                                   longer disambiguates it. ONLY these are gated.

STEP 2 — two phases per gated episode, from the FROZEN segmentation's `subskill` field plus task
semantics. No descriptor text is read, so the phases cannot drift with a VLM re-run:

  CUE  = frames [0, t0 of the first segment whose subskill moves the object away from its origin)
  USE  = frames of the segment(s) whose action the slot's value determines

STEP 3 — three nearest-centroid measurements, k-fold BY EPISODE, chance = the training label prior,
each computed for every feature source (the raw pooled tap included, which is the control that
decides whether a slot is memory or merely perception):

  (a) ENCODING   frame-level features on CUE frames        — is the variable encoded while visible?
  (b) VISIBILITY frame-level features on USE frames        — MUST be ~chance for every source; if the
                                                             RAW TAP decodes it here the slot is
                                                             perception, not memory: report and drop
  (c) CARRYING   causal history pools (mean/max over all   — does the stream make the variable
                 frames <= t) on USE frames                  linearly accessible when it is needed?
                                                             This is the sufficient-statistic test
                                                             the GDN read rests on.

  Markovianization = (a) high, (b) ~chance, (c) high, with the trained encoder above the untrained
  encoder AND above the raw tap on (a) and (c).

STEP 4 — chance, n episodes, n frames and a Wilson 95% interval on every accuracy. The Wilson
interval is computed on the FRAME count and is therefore anticonservative (frames of one episode
share a label and are not independent); the episode count is reported next to it so the reader can
see the real unit, and the k-fold split is by episode so the estimate itself is not leaked.

  PYTHONPATH=. python scripts/deliberation/rmb_phase_decodability.py \
      --labels ~/Research/TRI/wsm_data/deliberation/stage_e_labels/ab38d9efc0c649a3 \
      --source raw_tap=~/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k \
      --source E1b_smoke=~/.../omega/E1b_smoke_rmb --source untrained=~/.../omega/untrained_rmb \
      --out ~/.../rmb_phase_decodability.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------- step 1: slots
#: Subskills that move the object away from where it was first observed. The first such segment
#: closes the CUE phase. Taken from the frozen rmb subskill vocabulary (place/reach/grasp/turn/
#: close/wipe/lift/wait/navigate/wash/insert/retract/open).
#:
#: `reach` and `grasp` are deliberately NOT here, and that is a measured choice rather than taste.
#: Half the MemWashAndReturn episodes open with `reach` and half open directly with `grasp` at t0=0
#: (the approach is folded inside the grasp segment), so counting `grasp` as move-away emptied the
#: CUE window for exactly those 20 episodes. It is also wrong on the physics: closing the gripper
#: does not move the fruit off its container — the fruit leaves its origin when it is carried to
#: the sink and washed. Excluding both makes the boundary uniform across the two segmentation
#: styles AND correct, and it gives every gated episode a non-empty CUE window.
MOVE_AWAY = ("lift", "navigate", "wipe", "wash")

GATED_SLOTS = {
    # PRIMARY. Fruit starts in `fruit_container` on the left (Left) or right (Right) of the sink; an
    # identical decoy `fruit_container2` sits on the opposite side. `_update_success` requires the
    # fruit to end in `fruit_container` — the ORIGINAL one. Both variants' instruction is byte
    # identical ("Wash the fruit and return it to the container."), so the side is not in language;
    # it is visible at the start and must be recalled at the return, when two identical containers
    # are in view. SameLocation is EXCLUDED from this contrast: its instruction says "to the same
    # location as before", a different string, so its label would be partly readable from language.
    "MemWashAndReturn/return_side": {
        "tasks": {"MemWashAndReturnLeft": "left", "MemWashAndReturnRight": "right"},
        "use_subskills": ("place",),
        "use_which": "last",
        "kind": "OBSERVATION_GIVEN_THEN_OCCLUDED",
        "evidence": "identical prompt across both variants; success predicate reads "
        "destination_container_name='fruit_container' (the origin container) with an "
        "identical decoy on the opposite side (memory_env.py MemWashAndReturn"
        "{Left,Right})",
    },
    # SECONDARY. `olive_oil` is placed at `oil_container_counter_loc2`, the decoy `canola_oil` at
    # `oil_container_counter_loc`; success is lifting the olive oil. All four variants share one
    # instruction ("Pick up the olive oil bottle."), so the side is not in language. Whether it is
    # MEMORY or merely PERCEPTION is exactly what measurement (b) decides — the robot base starts at
    # the stove and both counters may be in view, in which case this is a search task and the slot
    # must be dropped. The label is the SECOND element of oils_route (the olive side).
    "MemRetrieveOils/olive_side": {
        "tasks": {
            "MemRetrieveOilsFromCounterLL": "left",
            "MemRetrieveOilsFromCounterRL": "left",
            "MemRetrieveOilsFromCounterLR": "right",
            "MemRetrieveOilsFromCounterRR": "right",
        },
        "use_subskills": ("grasp", "lift"),
        "use_which": "last",
        "kind": "OBSERVATION_GIVEN_THEN_OCCLUDED",
        "evidence": "identical prompt across all four variants; success = olive_oil lifted, and "
        "olive_oil sits at oil_container_counter_loc2 = route[1] (memory_env.py "
        "MemRetrieveOilsFromCounter{LL,LR,RL,RR})",
    },
}

EXCLUDED_SLOTS = {
    "MemFruitInSink/target_object": {
        "kind": "LANGUAGE_GIVEN",
        "evidence": "instruction names it verbatim: 'Pick up the {orange|peach|kiwi|...} and place it in the sink.'",
    },
    "MemHeatPot/cook_food+wait_min": {
        "kind": "LANGUAGE_GIVEN",
        "evidence": "'Turn on the stove, cook the {food}, wait for {n} minutes, and turn off the "
        "stove.' — both slots verbatim in the instruction",
    },
    "MemHeatPotMultiple/cook_food+add_food+add_after_min+wait_min": {
        "kind": "LANGUAGE_GIVEN",
        "evidence": "'Turn on the stove with the {food}, add the {food2} after {n} minutes, and "
        "wait for {m} minutes...' — all four slots verbatim",
    },
    "MemPutK/set_target": {
        "kind": "LANGUAGE_GIVEN",
        "evidence": "'Put all the bowls in the cabinet...' / 'Put all the breads in the "
        "microwave...' — the target is the instruction",
    },
    "MemFruitInSink/sink_source": {
        "kind": "NOT_ACTION_RELEVANT",
        "evidence": "the fruit's start side (left_far/right_far) is an observable layout constant; "
        "the success predicate is 'fruit in the sink' for both variants, so no action "
        "the policy takes depends on the value and there is no USE phase to gate",
    },
    "MemWashAndReturnSameLocation/return_side=origin": {
        "kind": "LANGUAGE_GIVEN",
        "evidence": "its instruction is 'return it to the same location as before', a different "
        "string from the Left/Right variants, so 'origin' is partly readable from "
        "language; excluded to keep the gated contrast prompt-identical",
    },
    "MemHeatPot/burner_or_pot_identity": {
        "kind": "NO_SUCH_SLOT",
        "evidence": "the coordinator's candidate; the frozen binding table "
        "(binding_annotations/597f3ff5e7cbd6ce) carries NO pot/burner slot for "
        "MemHeatPot or MemHeatPotMultiple, and none is derivable from the instruction "
        "or episodes.jsonl, so it cannot be gated without fabricating a label",
    },
}


# ----------------------------------------------------------------------------- step 2: the phases
def episode_segments(segments: dict, domain: int, task_index: int, episode: int) -> list:
    rows = np.flatnonzero(
        (segments["domain"] == domain) & (segments["task"] == task_index) & (segments["episode"] == episode)
    )
    rows = rows[np.argsort(segments["segment"][rows])]
    return [(str(segments["subskill"][r]), int(segments["t0"][r]), int(segments["t1"][r])) for r in rows]


def phases(segs: list, use_subskills: tuple, use_which: str) -> tuple:
    """(cue_end, use_t0, use_t1) in EPISODE frame numbers, from the frozen segmentation only.

    When no segment carries a move-away subskill the object is at its origin right up to the action
    that consumes it, so the USE segment's start closes the CUE window. `cue_end` is then clamped to
    `use_t0` unconditionally, which makes CUE and USE disjoint BY CONSTRUCTION — measurement (b) is
    only a visibility control if it cannot see a frame that (a) also scored.
    """
    hits = [(t0, t1) for skill, t0, t1 in segs if skill in use_subskills]
    use_t0, use_t1 = (hits[-1] if use_which == "last" else hits[0]) if hits else (None, None)
    cue_end = next((t0 for skill, t0, _ in segs if skill in MOVE_AWAY), None)
    if cue_end is None:
        cue_end = use_t0 if use_t0 is not None else (segs[-1][2] if segs else 0)
    if use_t0 is not None:
        cue_end = min(cue_end, use_t0)
    return cue_end, use_t0, use_t1


# ------------------------------------------------------------------- step 3/4: the classifier
def wilson(hits: int, total: int, z: float = 1.96) -> list:
    if total == 0:
        return [None, None]
    p = hits / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def nearest_centroid_kfold(features: list, labels: list, folds: int = 5, seed: int = 0) -> dict:
    """k-fold BY EPISODE. features[i] is one episode's [n_i, D] rows, labels[i] its single label."""
    usable = [i for i in range(len(features)) if len(features[i])]
    if len(usable) < folds * 2 or len(set(labels[i] for i in usable)) < 2:
        return {"n_episodes": len(usable), "note": "too few episodes or a single class"}
    rng = np.random.default_rng(seed)
    order = rng.permutation(usable)
    hits = total = 0
    prior_hits = 0.0
    for fold in range(folds):
        test = set(order[fold::folds].tolist())
        train = [i for i in usable if i not in test]
        by_label = defaultdict(list)
        for i in train:
            by_label[labels[i]].append(features[i].mean(0))
        if len(by_label) < 2:
            continue
        names = sorted(by_label)
        centroids = np.stack([np.mean(by_label[n], 0) for n in names])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        counts = Counter(labels[i] for i in train)
        prior = {k: v / len(train) for k, v in counts.items()}
        for i in test:
            rows = features[i]
            unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
            predicted = np.asarray(names)[(unit @ centroids.T).argmax(1)]
            hits += int((predicted == labels[i]).sum())
            prior_hits += len(rows) * prior.get(labels[i], 0.0)
            total += len(rows)
    if total == 0:
        return {"n_episodes": len(usable), "note": "no scorable frames"}
    accuracy = hits / total
    chance = prior_hits / total
    return {
        "n_episodes": len(usable),
        "n_frames": int(total),
        "accuracy": round(accuracy, 4),
        "wilson95": wilson(hits, total),
        "chance_prior": round(chance, 4),
        "lift": round(accuracy / chance, 3) if chance > 0.01 else None,
        "beats_chance": bool(wilson(hits, total)[0] > chance),
    }


def causal_pools(features: np.ndarray) -> dict:
    """mean and max of every prefix: pools[k][t] = pool(features[:t+1]). Causal by construction."""
    cumulative = np.cumsum(features, 0)
    counts = np.arange(1, len(features) + 1, dtype=np.float32)[:, None]
    return {"mean": cumulative / counts, "max": np.maximum.accumulate(features, 0)}


# ------------------------------------------------------------------------------------------ main
def load_episode(source_root: Path, kind: str, task: str, episode: int):
    """-> (features [F,D] float32, frame_indices [F]) for one episode, or None."""
    if kind == "tap":
        path = source_root / task / f"demo_{episode:06d}" / "p.npz"
        key = "p"
    else:
        path = source_root / "remembench" / task / f"demo_{episode:06d}" / "w.npz"
        key = "w"
    if not path.exists():
        return None
    blob = np.load(path)
    return np.asarray(blob[key], np.float32), np.asarray(blob["frame_indices"], np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="stage_e_labels/<id> (frozen segmentation)")
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="name=path — a pooled tap root (name contains 'tap') or an ω store root",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    labels_dir = Path(args.labels).expanduser()
    blob = np.load(labels_dir / "segments.npz", allow_pickle=True)
    segments = {k: blob[k] for k in blob.files}
    vocab = json.loads((labels_dir / "vocab.json").read_text())
    rmb = vocab["domains"].index("remembench")

    sources = {}
    for entry in args.source:
        name, _, path = entry.partition("=")
        sources[name] = (Path(path).expanduser(), "tap" if "tap" in name else "omega")

    report = {
        "labels": str(labels_dir),
        "pre_registered": "2026-08-28, before any rmb ω existed; replaces the reveal gate for rmb",
        "slot_classification": {
            "gated": {
                k: {"kind": v["kind"], "evidence": v["evidence"], "tasks": sorted(v["tasks"])}
                for k, v in GATED_SLOTS.items()
            },
            "excluded": EXCLUDED_SLOTS,
        },
        "phase_definition": {
            "cue": f"frames [0, t0 of the first segment whose subskill is in {MOVE_AWAY}) — frozen segmentation only",
            "use": {
                k: f"{v['use_which']} segment with subskill in {v['use_subskills']}" for k, v in GATED_SLOTS.items()
            },
        },
        "slots": {},
    }

    for slot, spec in GATED_SLOTS.items():
        # ---- phases, computed ONCE and shared by every source, so all sources see one split ----
        episodes, boundaries = [], []
        for task, value in sorted(spec["tasks"].items()):
            task_index = vocab["tasks"].index(task)
            eps = sorted(
                {
                    int(e)
                    for d, t, e in zip(segments["domain"], segments["task"], segments["episode"])
                    if d == rmb and t == task_index
                }
            )
            for ep in eps:
                segs = episode_segments(segments, rmb, task_index, ep)
                if not segs:
                    continue
                cue_end, use_t0, use_t1 = phases(segs, spec["use_subskills"], spec["use_which"])
                episodes.append((task, ep, value))
                boundaries.append((cue_end, use_t0, use_t1, len(segs)))

        rows = {
            "n_episodes_total": len(episodes),
            "label_counts": dict(Counter(v for _, _, v in episodes)),
            "phase_boundaries": {
                "median_cue_end": int(np.median([b[0] for b in boundaries])),
                "n_missing_use_segment": int(sum(b[1] is None for b in boundaries)),
                "median_use_t0": int(np.median([b[1] for b in boundaries if b[1] is not None]))
                if any(b[1] is not None for b in boundaries)
                else None,
                "per_episode": [
                    {
                        "task": t,
                        "episode": e,
                        "label": v,
                        "cue_end": b[0],
                        "use_t0": b[1],
                        "use_t1": b[2],
                        "n_segments": b[3],
                    }
                    for (t, e, v), b in zip(episodes, boundaries)
                ],
            },
            "measurements": {},
        }

        for name, (root, kind) in sorted(sources.items()):
            cue_x, cue_y, use_x, use_y = [], [], [], []
            pooled_x = {"mean": [], "max": [], "mean+max": []}
            missing = 0
            for (task, ep, value), (cue_end, use_t0, use_t1, _n) in zip(episodes, boundaries):
                loaded = load_episode(root, kind, task, ep)
                if loaded is None:
                    missing += 1
                    continue
                features, frame_index = loaded
                pools = causal_pools(features)
                cue = frame_index < cue_end
                cue_x.append(features[cue])
                cue_y.append(value)
                if use_t0 is None:
                    use = np.zeros(len(frame_index), bool)
                else:
                    use = (frame_index >= use_t0) & (frame_index < use_t1)
                use_x.append(features[use])
                use_y.append(value)
                pooled_x["mean"].append(pools["mean"][use])
                pooled_x["max"].append(pools["max"][use])
                pooled_x["mean+max"].append(np.concatenate([pools["mean"][use], pools["max"][use]], 1))
            rows["measurements"][name] = {
                "missing_episodes": missing,
                "a_encoding_cue_frames": nearest_centroid_kfold(cue_x, cue_y, args.folds, args.seed),
                "b_visibility_use_frames": nearest_centroid_kfold(use_x, use_y, args.folds, args.seed),
                "c_carrying_causal_pool_at_use": {
                    pool: nearest_centroid_kfold(pooled_x[pool], use_y, args.folds, args.seed)
                    for pool in ("mean", "max", "mean+max")
                },
            }
        report["slots"][slot] = rows

    # ---- the signature, read off the numbers rather than asserted -------------------------------
    signature = []
    for slot, rows in report["slots"].items():
        for name, m in rows["measurements"].items():
            a, b = m["a_encoding_cue_frames"], m["b_visibility_use_frames"]
            c = m["c_carrying_causal_pool_at_use"]["mean+max"]
            signature.append(
                {
                    "slot": slot,
                    "source": name,
                    "a_encoding_acc": a.get("accuracy"),
                    "a_chance": a.get("chance_prior"),
                    "b_visibility_acc": b.get("accuracy"),
                    "b_chance": b.get("chance_prior"),
                    "c_carrying_acc": c.get("accuracy"),
                    "c_chance": c.get("chance_prior"),
                    "a_beats_chance": a.get("beats_chance"),
                    "b_beats_chance": b.get("beats_chance"),
                    "c_beats_chance": c.get("beats_chance"),
                }
            )
    report["signature_rows"] = signature
    report["reading"] = (
        "Markovianization = (a) beats chance, (b) does NOT (for EVERY source, raw tap included), "
        "(c) beats chance, and the trained encoder exceeds both the untrained encoder and the raw "
        "tap on (a) and (c). If the RAW TAP beats chance on (b), the slot is perception rather "
        "than memory and is dropped. Report metric only; nothing is selected on it."
    )
    text = json.dumps(report, indent=1)
    print(text if not args.out else text[:2000])
    if args.out:
        Path(args.out).expanduser().write_text(text)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
