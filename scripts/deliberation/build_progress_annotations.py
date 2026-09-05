#!/usr/bin/env python3
"""H14 A13(e) — per-frame PROGRESS-STATE annotations, derived from the FROZEN segmentation only.

Why this exists. H14's Markovianization gate has now been tried on two kinds of slot and both were
PERCEPTION, not memory: RoboCasa's predicate-bound variables (knob / food / layout) are visible from
frame 0, and ReMemBench's "hidden" sides are scene-layout constants that the raw frozen pi0.5 tap
decodes at USE time (§14.8, §17 of h14_p0_status.md). What a per-frame tap genuinely cannot carry is
PROGRESS STATE — what has already been done in THIS episode. Amendment A13(e) named this build.

What a progress label is. For each task family, the repeated unit of the task is identified, and

    progress(t) = the number of instances of that unit completed strictly before frame t

with two degenerate specialisations: a BOOLEAN unit (has the fruit been washed yet) and an
ACCUMULATOR unit (how much cook / scrub time has accrued), binned at fixed ABSOLUTE frame
thresholds so the label is not a rescaling of elapsed time.

The evidence is the frozen segmentation ONLY — `subskill`, `t0`, `t1` from
`stage_e_labels/<id>/segments.npz`. No descriptor free text is read, so the labels cannot drift when
the pass-1 VLM store is regenerated. `memory_dependency.kinds` from the pass-1 store is aggregated
into the manifest as PROVENANCE for the family assignment (it is what tags MemPutK / Pack… as
`set_completion` and MemHeatPot / Scrub… as `accumulator`); it never enters a label.

Per-family rules, written down before any feature was touched
-------------------------------------------------------------

  count        unit = one completed PLACE-LIKE segment (subskill in {place, insert}).
               progress(t) = #{segments s : subskill(s) in PLACE_LIKE and t1(s) <= t}, capped at
               CAP=4 (label "4+" absorbs everything above).  Primary window = frames at or after
               t1 of the FIRST place-like segment, i.e. exactly the frames on which progress >= 1,
               so the trivially time-predictable all-zero prefix is excluded.
               Families: rmb MemPutKBowlInCabinet, MemPutKBreadInMicrowave; robocasa
               PackIdenticalLunches, RecycleBottlesByType, PortionHotDogs, GatherTableware.

  boolean      unit = the wash.  progress(t) = 1 iff t >= t1 of the FIRST segment with subskill in
               {wash, wipe}, else 0.  `wipe` is included because the frozen segmentation writes the
               washing motion as `wipe` in most episodes (`wash` appears in only 11 segments of the
               whole corpus); an episode with neither is DROPPED, not guessed.  Primary window =
               ALL frames of a qualifying episode — a boolean needs its 0 class, and dropping it
               would leave one class and an undefined contrast.
               Family: rmb MemWashAndReturn{Left, Right, SameLocation}.

  accumulator_turn   cook-elapsed.  stove_on = t1 of the FIRST `turn` segment; stove_off = t0 of the
               LAST `turn` segment; an episode with fewer than two `turn` segments, or with
               stove_off <= stove_on, is DROPPED (the segmentation did not resolve a cook window).
               elapsed(t) = clip(t - stove_on, 0, stove_off - stove_on), binned at ABSOLUTE frame
               thresholds (fps 20): [0,80) -> 0 (<4 s), [80,240) -> 1 (4-12 s), [240, inf) -> 2
               (>=12 s).  Absolute rather than fractional bins on purpose: a fractional bin IS
               normalized time and the gate would be vacuous.  Primary window = frames >= stove_on.
               Family: rmb MemHeatPot, MemHeatPotMultiple.

  accumulator_dwell  scrub-time.  dwell(t) = total frames spent INSIDE segments with subskill in
               {wipe, scrub} up to t (an overlap sum, so it freezes between bouts and is therefore
               NOT a monotone function of normalized time).  Bins: [0,40) -> 0 (<2 s), [40,120) ->
               1 (2-6 s), [120, inf) -> 2 (>=6 s).  Primary window = frames >= t0 of the first
               wipe/scrub segment.
               Family: robocasa ScrubCuttingBoard.

The frame grid is the frozen pooled tap's own grid (`frame_indices` in `p.npz`, stride 8 with the
final frame appended) — verified identical to every omega export's grid, so one table serves every
feature source without resampling.

Rule-quality diagnostic (reported, never used to change a label): for the RoboCasa count families
the final progress count is compared against K_meta = the number of movable target objects in
`ep_meta.json`.  A place-like segment is taken as one completed unit, so an over-segmented placement
inflates the count and a placement the segmenter merged deflates it; the diagnostic says by how much
per task rather than asserting the proxy is exact.

    python scripts/deliberation/build_progress_annotations.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DELIB = Path("~/Research/TRI/wsm_data/deliberation").expanduser()
POOLED = Path("~/Research/TRI/wsm_data/wsm_pooled").expanduser()
ROBOCASA_ROOT = Path("~/Research/robocasa/datasets/v1.0/target/composite").expanduser()
DEFAULT_LABELS = DELIB / "stage_e_labels/adc1c7575dd70fa3"
DEFAULT_OUT = DELIB / "progress_annotations"

FPS = 20
PLACE_LIKE = ("place", "insert")
WASH_LIKE = ("wash", "wipe")
SCRUB_LIKE = ("wipe", "scrub")
COUNT_CAP = 4

TAP_ROOT = {"robocasa": POOLED / "pi_100k", "remembench": POOLED / "rmb_pi_100k"}

#: family -> rule.  `tasks` are vocab names; `domain` selects the tap root and the ep_meta reader.
FAMILIES = {
    "rmb_MemPutK": {
        "domain": "remembench",
        "rule": "count",
        "tasks": ("MemPutKBowlInCabinet", "MemPutKBreadInMicrowave"),
        "unit": "one item placed into the destination (a place-like segment completed)",
        "memory_dependency": "set_completion",
    },
    "rmb_MemWashAndReturn": {
        "domain": "remembench",
        "rule": "boolean",
        "tasks": ("MemWashAndReturnLeft", "MemWashAndReturnRight", "MemWashAndReturnSameLocation"),
        "unit": "the wash (has this fruit been washed yet)",
        "memory_dependency": "phase_boolean",
    },
    "rmb_MemHeatPot": {
        "domain": "remembench",
        "rule": "accumulator_turn",
        "tasks": ("MemHeatPot", "MemHeatPotMultiple"),
        "unit": "cook time accrued since the stove was turned on",
        "memory_dependency": "accumulator",
    },
    "rc_PackIdenticalLunches": {
        "domain": "robocasa",
        "rule": "count",
        "tasks": ("PackIdenticalLunches",),
        "unit": "one item packed",
        "memory_dependency": "set_completion",
    },
    "rc_RecycleBottlesByType": {
        "domain": "robocasa",
        "rule": "count",
        "tasks": ("RecycleBottlesByType",),
        "unit": "one bottle sorted",
        "memory_dependency": "set_completion",
    },
    "rc_PortionHotDogs": {
        "domain": "robocasa",
        "rule": "count",
        "tasks": ("PortionHotDogs",),
        "unit": "one bun/sausage portioned",
        "memory_dependency": "set_completion",
    },
    "rc_GatherTableware": {
        "domain": "robocasa",
        "rule": "count",
        "tasks": ("GatherTableware",),
        "unit": "one item gathered",
        "memory_dependency": "set_completion",
    },
    "rc_ScrubCuttingBoard": {
        "domain": "robocasa",
        "rule": "accumulator_dwell",
        "tasks": ("ScrubCuttingBoard",),
        "unit": "scrub time accrued",
        "memory_dependency": "accumulator",
    },
}

#: Named and NOT built, with the reason.  Kept in the manifest so the omission is on the record.
NOT_DERIVABLE = {
    "rmb_MemFruitInSink": "single pick-and-place; the task has no repeated unit, no accumulator and "
    "no phase boolean, so 'progress' is a two-state ('picked yet') label that "
    "is exactly the segmentation's own grasp boundary — no memory content "
    "beyond the current frame's gripper state",
    "rmb_MemRetrieveOils": "single lift; same reason",
    "rc_predicate_constant_others": "the remaining RoboCasa tasks (KettleBoiling, HeatKebabSandwich, "
    "CategorizeCondiments, SeparateFreezerRack, SearingMeat, "
    "StirVegetables, CuttingToolSelection, PanTransfer) either have "
    "one unit or a runtime accumulator with no segmentation-visible "
    "start event; not in the A13(e) list and not built",
}


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------- segmentation
def load_segments(labels_dir: Path):
    blob = np.load(labels_dir / "segments.npz", allow_pickle=True)
    seg = {k: blob[k] for k in blob.files}
    vocab = json.loads((labels_dir / "vocab.json").read_text())
    per = defaultdict(list)  # (domain_name, task_name, episode) -> [(segment, t0, t1, subskill)]
    for d, t, e, s, t0, t1, sk in zip(
        seg["domain"], seg["task"], seg["episode"], seg["segment"], seg["t0"], seg["t1"], seg["subskill"]
    ):
        per[(vocab["domains"][int(d)], vocab["tasks"][int(t)], int(e))].append((int(s), int(t0), int(t1), str(sk)))
    for key in per:
        per[key].sort()
    return per, vocab


# ------------------------------------------------------------------------------------- the rules
def label_count(segs):
    ends = [t1 for _s, _t0, t1, sk in segs if sk in PLACE_LIKE]
    if not ends:
        return None, "no place-like segment"
    ends.sort()

    def fn(t):
        return min(int(np.searchsorted(ends, t, side="right")), COUNT_CAP)

    return (fn, ends[0], f"{len(ends)} place-like segments"), None


def label_boolean(segs):
    ends = [t1 for _s, _t0, t1, sk in segs if sk in WASH_LIKE]
    if not ends:
        return None, "no wash/wipe segment"
    first = min(ends)

    def fn(t):
        return int(t >= first)

    return (fn, 0, f"first wash-like ends at {first}"), None


def label_accumulator_turn(segs):
    turns = [(t0, t1) for _s, t0, t1, sk in segs if sk == "turn"]
    if len(turns) < 2:
        return None, "fewer than two turn segments"
    on, off = turns[0][1], turns[-1][0]
    if off <= on:
        return None, "stove_off <= stove_on"
    edges = (80, 240)

    def fn(t):
        e = min(max(t - on, 0), off - on)
        return int(np.searchsorted(edges, e, side="right"))

    return (fn, on, f"cook window [{on},{off}) = {off - on} frames"), None


def label_accumulator_dwell(segs):
    bouts = [(t0, t1) for _s, t0, t1, sk in segs if sk in SCRUB_LIKE]
    if not bouts:
        return None, "no wipe/scrub segment"
    start = min(t0 for t0, _ in bouts)
    edges = (40, 120)

    def fn(t):
        dwell = sum(max(0, min(t1, t) - t0) for t0, t1 in bouts)
        return int(np.searchsorted(edges, dwell, side="right"))

    return (fn, start, f"{len(bouts)} bouts, total dwell {sum(t1 - t0 for t0, t1 in bouts)} frames"), None


RULES = {
    "count": label_count,
    "boolean": label_boolean,
    "accumulator_turn": label_accumulator_turn,
    "accumulator_dwell": label_accumulator_dwell,
}

LABEL_NAMES = {
    "count": {i: (f"{COUNT_CAP}+" if i >= COUNT_CAP else str(i)) for i in range(COUNT_CAP + 1)},
    "boolean": {0: "unwashed", 1: "washed"},
    "accumulator_turn": {0: "cook<4s", 1: "cook4-12s", 2: "cook>=12s"},
    "accumulator_dwell": {0: "scrub<2s", 1: "scrub2-6s", 2: "scrub>=6s"},
}


# ------------------------------------------------------------------------------------ ep_meta K
#: Which `object_cfgs` names the task's instruction actually asks the robot to MOVE, per task, read
#: off ep_meta's `lang` field.  Diagnostic only — it never touches a label.
ROBOCASA_MOVABLE = {
    "PackIdenticalLunches": ("vegetable", "meat"),  # "one eggplant and one steak in each"
    "PortionHotDogs": ("hotdog_bun", "sausage"),  # "one bun and one sausage on each plate"
    "GatherTableware": ("glass", "bowl"),  # "gather all objects into one cabinet"
    "RecycleBottlesByType": ("_middle",),  # only the middle bottles are sorted
}


def k_meta_observed(task: str, episode: int):
    """Number of movable target objects in the episode's ep_meta — the count diagnostic only."""
    prefixes = ROBOCASA_MOVABLE.get(task)
    if prefixes is None:
        return None
    hits = glob.glob(
        str(ROBOCASA_ROOT / task / "*" / "lerobot" / "extras" / f"episode_{episode:06d}" / "ep_meta.json")
    )
    if not hits:
        return None
    try:
        meta = json.loads(Path(hits[0]).read_text())
    except Exception:
        return None
    return sum(
        1 for c in (meta.get("object_cfgs") or []) if any(p in str(c.get("name", "")).lower() for p in prefixes)
    )


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--pass1",
        default=str(DELIB / "pass1_store"),
        help="pass-1 store, read ONLY for the memory_dependency provenance counts",
    )
    args = ap.parse_args()

    labels_dir = Path(args.labels).expanduser()
    per_episode, vocab = load_segments(labels_dir)
    seg_sha = file_sha(labels_dir / "segments.npz")

    rules_blob = json.dumps(
        {
            "families": FAMILIES,
            "place_like": PLACE_LIKE,
            "wash_like": WASH_LIKE,
            "scrub_like": SCRUB_LIKE,
            "count_cap": COUNT_CAP,
            "fps": FPS,
        },
        sort_keys=True,
    )
    ann_id = hashlib.sha256((code_sha() + rules_blob + seg_sha).encode()).hexdigest()[:16]
    out_dir = Path(args.out).expanduser() / ann_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        k: []
        for k in (
            "domain",
            "task",
            "episode",
            "frame",
            "segment_index",
            "progress",
            "normalized_time",
            "primary",
            "family",
        )
    }
    per_family = {}

    for family, spec in FAMILIES.items():
        domain, rule = spec["domain"], spec["rule"]
        tap_root = TAP_ROOT[domain]
        fam = {
            "rule": rule,
            "unit": spec["unit"],
            "tasks": list(spec["tasks"]),
            "memory_dependency_tag": spec["memory_dependency"],
            "label_names": LABEL_NAMES[rule],
            "dropped": Counter(),
            "per_task": {},
        }
        for task in spec["tasks"]:
            episodes = sorted(e for (d, t, e) in per_episode if d == domain and t == task)
            kept, dropped, final_counts, kmeta_rows = 0, Counter(), Counter(), []
            label_hist, primary_hist = Counter(), Counter()
            for ep in episodes:
                segs = per_episode[(domain, task, ep)]
                built, why = RULES[rule](segs)
                if built is None:
                    dropped[why] += 1
                    continue
                fn, window_start, _note = built
                tap = tap_root / task / f"demo_{ep:06d}" / "p.npz"
                if not tap.exists():
                    dropped["no pooled tap"] += 1
                    continue
                frames = np.asarray(np.load(tap)["frame_indices"], np.int64)
                t_end = max(s[2] for s in segs)
                seg_t0 = np.asarray([s[1] for s in segs], np.int64)
                for f in frames.tolist():
                    lab = fn(f)
                    rows["domain"].append(domain)
                    rows["task"].append(task)
                    rows["episode"].append(ep)
                    rows["frame"].append(f)
                    rows["segment_index"].append(int(np.searchsorted(seg_t0, f, side="right")) - 1)
                    rows["progress"].append(lab)
                    rows["normalized_time"].append(f / max(t_end - 1, 1))
                    rows["primary"].append(f >= window_start)
                    rows["family"].append(family)
                    label_hist[lab] += 1
                    if f >= window_start:
                        primary_hist[lab] += 1
                kept += 1
                final_counts[fn(t_end)] += 1
                if rule == "count" and domain == "robocasa":
                    k = k_meta_observed(task, ep)
                    if k is not None:
                        raw = sum(1 for _s, _t0, t1, sk in segs if sk in PLACE_LIKE)
                        kmeta_rows.append((raw, k))
            entry = {
                "n_episodes_kept": kept,
                "n_episodes_dropped": sum(dropped.values()),
                "dropped_reasons": dict(dropped),
                "label_hist_all_frames": {LABEL_NAMES[rule][k]: v for k, v in sorted(label_hist.items())},
                "label_hist_primary": {LABEL_NAMES[rule][k]: v for k, v in sorted(primary_hist.items())},
            }
            if kmeta_rows:
                raw = np.array([r for r, _ in kmeta_rows])
                km = np.array([k for _, k in kmeta_rows])
                entry["count_rule_diagnostic"] = {
                    "n": len(kmeta_rows),
                    "k_meta_mode": int(np.bincount(km).argmax()),
                    "raw_place_count_median": int(np.median(raw)),
                    "exact_match_rate": round(float((raw == km).mean()), 3),
                    "within_one_rate": round(float((np.abs(raw - km) <= 1).mean()), 3),
                }
            fam["per_task"][task] = entry
            fam["dropped"].update(dropped)
        fam["dropped"] = dict(fam["dropped"])
        per_family[family] = fam

    # --- memory_dependency provenance, structured field only, never a label ---------------------
    provenance = {}
    for family, spec in FAMILIES.items():
        kinds = Counter()
        for task in spec["tasks"]:
            sub = "remembench" if spec["domain"] == "remembench" else "robocasa"
            for path in glob.glob(os.path.join(args.pass1, sub, task, "*.descriptors.json")):
                try:
                    blob = json.loads(Path(path).read_text())
                except Exception:
                    continue
                for d in blob.get("descriptors", []):
                    md = (d.get("descriptor") or {}).get("memory_dependency") or {}
                    for k in md.get("kinds") or []:
                        kinds[str(k)] += 1
        provenance[family] = dict(kinds.most_common())

    arrays = {
        "domain": np.array(rows["domain"], dtype="U10"),
        "task": np.array(rows["task"], dtype="U40"),
        "episode": np.array(rows["episode"], np.int32),
        "frame": np.array(rows["frame"], np.int32),
        "segment_index": np.array(rows["segment_index"], np.int32),
        "progress_label": np.array(rows["progress"], np.int16),
        "normalized_time": np.array(rows["normalized_time"], np.float32),
        "primary_window": np.array(rows["primary"], bool),
        "family": np.array(rows["family"], dtype="U40"),
    }
    np.savez_compressed(out_dir / "progress.npz", **arrays)

    manifest = {
        "annotation_id": ann_id,
        "code_sha": code_sha(),
        "labels": str(labels_dir),
        "segments_sha": seg_sha,
        "amendment": "A13(e) — progress-state decodability",
        "evidence": "frozen segmentation subskill/t0/t1 ONLY; no descriptor free text",
        "frame_grid": "the frozen pooled tap's own frame_indices (stride 8 + final frame), "
        "verified identical to every omega export's grid",
        "rules": json.loads(rules_blob),
        "label_names": LABEL_NAMES,
        "families": per_family,
        "not_derivable": NOT_DERIVABLE,
        "memory_dependency_provenance": provenance,
        "n_rows": int(len(arrays["frame"])),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                "annotation_id": ann_id,
                "out": str(out_dir),
                "n_rows": manifest["n_rows"],
                "families": {
                    k: {t: v["per_task"][t]["n_episodes_kept"] for t in v["per_task"]} for k, v in per_family.items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
