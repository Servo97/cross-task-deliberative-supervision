"""WSM label stage A2: per-segment subtask CAPTIONS over the FROZEN keyframe segmentation.

H14 ADDITION (2026-08-22, `aug_22/deliberative_workspace_plan.md` §3 pass 1). Two ADDITIVE flags,
neither of which changes any existing default:

  --spec {caption,descriptor}   caption  = the H13 30-token imperative (unchanged, default)
                                descriptor = the H14 structured segment descriptor (plan §3):
                                subskill verb-frame / target object + state / spatial relation /
                                preconditions / postconditions / memory_dependency (the segment-level
                                LCR annotation) / failure_lookalikes.
  --backend {hf,vllm}           hf   = the transformers path this file has always used (default)
                                vllm = an OpenAI-compatible chat client against a local vLLM server
                                (`scripts/deliberation/serve_vllm.sh`), which is what makes 21k
                                segments affordable (s3 §4 change 1: HF batch-4 is ~90 tok/s/GPU).

Request granularity differs by spec and this is deliberate:
  caption    -> ONE request per EPISODE carrying every segment (amortizes the image prefill; the
                model must return exactly n_seg captions).
  descriptor -> ONE request per SEGMENT (9 images, ~1,050 input tok, hard max_tokens 2048). This is
                exactly the geometry s3 §3.2 costs the loop at, and it is what makes the hard
                per-request token cap meaningful. Frames are still decoded ONCE per episode.

Descriptors ENRICH, they do not re-segment: `keyframes` still comes from the frozen label npz, so
`segments_from_keyframes` yields byte-identical (t0, t1) to the caption store, and the existing
30-token caption is passed in as a hint when present.
Output: <out>/<Task>/ep_%06d.descriptors.json (a different suffix -> the two stores never collide).

H13 §7 (see internal_planning_and_todos/aug_12/h13_joint_wsm_tree.md). The v1 pipeline produced
subgoal TEXT but only its embeddings ever shipped; the strings are gone. This stage regenerates
them WITHOUT re-segmenting: the `keyframes` array frozen into every label npz defines the
segments, and Qwen only writes one imperative caption per fixed segment. That keeps R1's keypatch
head and R3's caption head on the SAME temporal structure (removes a confound from the R3-R1 /
R4-R2 marginals) and guarantees the captions join onto the existing label store by (Task, episode).

SEGMENTATION IS NOT A CHOICE HERE — it mirrors `train_wsm_base/data.py::per_frame_subgoal_idx`
byte-for-byte, which is what every downstream consumer already uses:

    idx = np.searchsorted(keyframes, frame_indices, side="left"); np.clip(idx, 0, K-1)

i.e. frame t belongs to the FIRST subgoal whose keyframe is >= t, with everything past the last
keyframe clamped into the last subgoal. In half-open [t0, t1) form that is exactly K segments:

    seg 0      = [0,            k_0 + 1)
    seg i      = [k_{i-1} + 1,  k_i + 1)      for 0 < i < K-1
    seg K-1    = [k_{K-2} + 1,  T)            <- absorbs the post-completion tail

K segments for K keyframes (so they index 1:1 onto `salient_global`), monotonic, covering [0, T).

Frames are decoded STRAIGHT from the RoboCasa v1.0 mp4s with PyAV (same approach and rationale as
extract_frames.py) — no intermediate ep*_frames.npz. Only ~3 frames/segment x 3 views are needed,
so materializing the stage-A0 dumps for all 7500 episodes would be pure waste.

Views + captions follow qwen_subgoals.py's conventions (agentview_left / agentview_right /
eye_in_hand, captioned "left" / "right" / "eye-in-hand").

Run in the vlm_labeler venv (transformers 4.57.x):
  ~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.caption_segments \
      --out ~/Research/TRI/wsm_data/wsm_labels_captions --device cuda:0 --shard 0 --num-shards 2

Output per episode: <out>/<Task>/ep_%06d.captions.json   (%06d = the label store's episode index)
  {"episode_id": int, "task": str, "n_frames": int, "keyframes": [int],
   "segments": [{"t0": int, "t1": int, "text": str}], "model": str, "prompt_sha": str,
   "views_used": [str]}
Plus <out>/<Task>/manifest.json (--finalize) and <out>/_provenance/<run>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from workspace_models.labels.extract_frames import episode_video_path, load_episode_meta
from workspace_models.labels.geometry import VIEW_LEROBOT_KEY, VIEWS

# Same view captions qwen_subgoals.py shows the model.
CAPTION = {"agentview_left": "left", "agentview_right": "right", "eye_in_hand": "eye-in-hand"}

DEFAULT_DATASET_ROOT = "~/Research/robocasa/datasets/v1.0/target"
DEFAULT_LABELS_ROOT = "~/Research/TRI/wsm_data/wsm_labels_pi_mirror"
DEFAULT_OUT = "~/Research/TRI/wsm_data/wsm_labels_captions"

MAX_CAPTION_TOKENS = 30  # hard spec cap (h13 tree §7.3)

SYSTEM = (
    "You are labeling a robot-manipulation episode (a Franka Panda arm in a kitchen) to produce "
    "short subtask descriptions. The episode has ALREADY been divided into a fixed, numbered list "
    "of consecutive time segments; you must NOT change, merge, split, or re-order them. "
    "For each segment you will see frames from its start, middle and end, and each timestamp shows "
    "THREE views captioned with the frame index: LEFT and RIGHT third-person agentviews, and an "
    "EYE-IN-HAND close-up from a camera on the gripper (objects there appear very large, "
    "truncated, or out of frame). "
    "Write ONE caption per segment describing what the robot does DURING that segment, as an "
    "imperative command addressed to the robot (e.g. 'open the left cabinet door', 'pick up the "
    "red mug from the counter', 'place the bowl in the sink'). "
    "Rules for every caption: start with a verb; at most 10 words; name the SPECIFIC object and, "
    "where it matters, which one (left/right, the color, the location); describe only what happens "
    "in THAT segment, not the whole task; do not mention frame indices, segment numbers, cameras, "
    "or views; do not write commentary, hedging, or explanations. If a segment shows the arm "
    "merely moving toward something, say so ('reach toward the drawer handle'); if it shows the "
    "task already finished and the arm settling, say so ('hold still after finishing the task'). "
    "Never leave a caption empty and never refuse. "
    'Respond with ONLY a JSON object: {"captions": [{"segment": int, "text": str}, ...]} '
    "containing EXACTLY one entry per segment, in ascending segment order."
)


# ------------------------------------------------------------------- H14 descriptor spec (plan §3)
# BEHAVIORAL, not visual. Every field exists because something downstream consumes it:
#   subskill/verb_frame + object + spatial_relation -> the embedding text that buckets pass 2
#   preconditions/postconditions                    -> the EQUIVALENT vs CONTRAST discriminator
#                                                      ("different completion condition" is the
#                                                      literal definition of CONTRAST in pass 2)
#   memory_dependency                               -> the segment-level LCR annotation (D3);
#                                                      this is the Markovianization label (§0.0)
#   failure_lookalikes                              -> seeds the mined hard negatives
# HARD per-request cap (s3 risk 2: the xhigh default is a 4-8x multiplier).
# 2048 -> 3072 on 2026-08-22: the v2.1 head is longer and completion_max reached 1,995/2,048 with a
# 3.4% malformed-JSON rate (vs 0.2% at v1's shorter head) -- cap squeeze, not a schema problem.
# Raising the ceiling changes neither prompt_sha nor schema_sha and does not move the measured
# average (2,859 tok/segment); it only stops the tail from being clipped.
DESCRIPTOR_MAX_TOKENS = 3072

DESCRIPTOR_SYSTEM = (
    "You are annotating ONE time segment of a robot-manipulation demonstration (a Franka Panda arm "
    "in a kitchen). You see frames from the START, MIDDLE and END of this segment; each timestamp "
    "shows THREE views: LEFT and RIGHT third-person agentviews and an EYE-IN-HAND close-up from a "
    "camera on the gripper (objects there appear very large, truncated, or out of frame).\n"
    "Describe this segment FUNCTIONALLY: what knowledge would a policy need to complete it. Do not "
    "describe pixels, lighting, camera angles, frame indices or segment numbers.\n"
    "Field rules:\n"
    "- subskill: one lowercase verb naming the manipulation primitive (reach, grasp, lift, place, "
    "open, close, push, pull, pour, turn, press, wipe, stir, insert, navigate, wait, retract).\n"
    "- verb_frame: the primitive as verb(role=value, ...) with the roles it actually binds, e.g. "
    "place(object=kettle, destination=front-left burner) or turn(object=stove knob, setting=on).\n"
    "- target_object: {class, attributes, state_before, state_after}. class is a bare noun "
    "('kettle', 'cabinet door'); attributes are discriminating properties only (color, side, size); "
    "state_before/state_after describe the object, not the arm.\n"
    "- spatial_relation: where the target sits relative to fixed scene landmarks (counter, stove, "
    "sink, cabinet, microwave) and to any distractor of the same class.\n"
    "- preconditions: what must already be true for this segment to be attemptable.\n"
    "- postconditions: what is true afterwards that was not true before (the effect).\n"
    "- memory_dependency: whether choosing the CORRECT action here needs information the current "
    "frames cannot show. Set depends_on_history true ONLY if a competent agent seeing just these "
    "frames could pick the wrong target, wrong destination, wrong count or wrong moment.\n"
    "  kinds is a LIST. List EVERY kind that applies, not just the most obvious one:\n"
    "    none               - the current frames fully determine the correct action.\n"
    "    hidden_binding     - a rule or identity fixed earlier and now occluded.\n"
    "    accumulator        - correctness depends on HOW MUCH has happened so far: elapsed time in "
    "contact, number of repetitions, or the extent of surface already covered. The giveaway is a "
    "segment that repeats a motion while the object's state barely changes, or a segment that must "
    "CONTINUE until a threshold no single frame can show.\n"
    "    set_completion     - which items are already done, placed, or in flight.\n"
    "    prospective        - an event was started earlier and the correct action is to wait for it "
    "and then act.\n"
    "    instruction_binding- the episode goal names the target, destination, count or burner, and "
    "the frames alone do not.\n"
    "  These are NOT mutually exclusive and you must not collapse them to one. A stirring segment "
    "whose goal says 'stir the vegetables' is BOTH instruction_binding AND accumulator: the goal "
    "names what to stir, and completion needs enough elapsed stirring. A segment that scrubs a board "
    'until clean is accumulator even when the goal names the board. Use ["none"] alone, never '
    "combined with anything else.\n"
    "  evidence names the earlier event or the accumulated quantity in <=15 words; use null only "
    'when kinds is exactly ["none"].\n'
    "- failure_lookalikes: 1-3 plausible WRONG completions that would look almost identical in "
    "these frames (wrong instance of the same class, wrong destination, stopping early). Each "
    "<=12 words. Empty list only if genuinely none exist.\n"
    "Be specific and terse. Never refuse, never hedge, never leave a field empty. "
    "Respond with ONLY the JSON object described by the schema."
)

# RoboMME ships exactly two views (front + wrist, both 256px) instead of RoboCasa's three, so the
# camera sentence in DESCRIPTOR_SYSTEM would be factually wrong there. Rather than mutate the frozen
# 3-view prompt (which would change its sha and orphan the running RoboCasa store), the 2-view
# domains get their own system prompt and therefore their own prompt_sha. Two shas in a
# multi-domain store is CORRECT: different camera geometry is a different prompt.
DESCRIPTOR_SYSTEM_2VIEW = DESCRIPTOR_SYSTEM.replace(
    "You see frames from the START, MIDDLE and END of this segment; each timestamp "
    "shows THREE views: LEFT and RIGHT third-person agentviews and an EYE-IN-HAND close-up from a "
    "camera on the gripper (objects there appear very large, truncated, or out of frame).",
    "You see frames from the START, MIDDLE and END of this segment; each timestamp "
    "shows TWO views: a FRONT third-person view of the workspace, and a WRIST close-up from a "
    "camera on the gripper (objects there appear very large, truncated, or out of frame).",
)
assert DESCRIPTOR_SYSTEM_2VIEW != DESCRIPTOR_SYSTEM, "2-view prompt rewrite did not apply"

DESCRIPTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "subskill": {"type": "string"},
        "verb_frame": {"type": "string"},
        "target_object": {
            "type": "object",
            "properties": {
                "class": {"type": "string"},
                "attributes": {"type": "array", "items": {"type": "string"}},
                "state_before": {"type": "string"},
                "state_after": {"type": "string"},
            },
            "required": ["class", "attributes", "state_before", "state_after"],
            "additionalProperties": False,
        },
        "spatial_relation": {"type": "string"},
        "preconditions": {"type": "array", "items": {"type": "string"}},
        "postconditions": {"type": "array", "items": {"type": "string"}},
        "memory_dependency": {
            "type": "object",
            "properties": {
                "depends_on_history": {"type": "boolean"},
                # v2.1: MULTI-LABEL. The v2 pilot showed a single enum forces the model to collapse
                # co-occurring mechanisms and it reliably picks the one the instruction states
                # outright: `accumulator` fired on 3 of 162 segments while Tier-A elapsed-time tasks
                # were 64 of them. A store that cannot represent accumulator AND instruction_binding
                # jointly under-represents exactly the demand the mem13 suite was built around.
                # NOTE: no `uniqueItems`. vLLM's grammar compiler returns HTTP 500 on it (isolated
                # 2026-08-22: a bare enum array compiles, +minItems compiles, +uniqueItems does not).
                # Duplicates are rejected by `validate_descriptor_record` instead -- the grammar
                # constrains what it can, the validator catches the rest.
                "kinds": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": [
                            "none",
                            "hidden_binding",
                            "accumulator",
                            "set_completion",
                            "prospective",
                            "instruction_binding",
                        ],
                    },
                },
                "evidence": {"type": ["string", "null"]},
            },
            "required": ["depends_on_history", "kinds", "evidence"],
            "additionalProperties": False,
        },
        "failure_lookalikes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "subskill",
        "verb_frame",
        "target_object",
        "spatial_relation",
        "preconditions",
        "postconditions",
        "memory_dependency",
        "failure_lookalikes",
    ],
    "additionalProperties": False,
}

MEMORY_KINDS = tuple(DESCRIPTOR_SCHEMA["properties"]["memory_dependency"]["properties"]["kinds"]["items"]["enum"])

# The user-message template for --spec descriptor. It is a NAMED CONSTANT and it is folded into
# `prompt_sha("descriptor")` because it is as load-bearing as the system prompt: the 2026-08-22 pilot
# showed the descriptors' `memory_dependency` field is determined by what this head does or does not
# carry, and a head change that left the sha alone would silently produce an incompatible store.
#
# `instruction` is the per-episode language goal from meta/episodes.jsonl. It was ALREADY being
# loaded by extract_frames.load_episode_meta (as `prompt`) and simply never reached the model.
# Without it, SearingMeat scored 0/10 on memory_dependency while correctly naming the burner in
# `verb_frame` -- the model could see WHICH burner was used but had no way to know the choice was
# CONSTRAINED, because "place it on the front left burner" lives only in the instruction (33 distinct
# instructions for that task, so it is a per-EPISODE hidden variable, not a per-task constant).
#
# `duration_s` / episode position are the other two things three frames cannot show. Tier-A
# accumulator tasks (elapsed wash time, stir duration, contact timer) collapsed to 8-33% mem-dep
# without them.
DESCRIPTOR_HEAD = (
    "Task family: {task}\n"
    "Episode goal (the language instruction given to the robot): {instruction}\n"
    "This is segment {si1} of {nseg} in the episode (frames {t0}..{t1} of {T}; "
    "{dur:.1f} s at {fps:g} fps; the episode is {frac0:.0%}-{frac1:.0%} elapsed here).{prior}\n"
    "Frames below are start / middle / end of THIS segment, each in {views} views.\n"
    "Note: a segment that repeats a motion without changing the object's state is prima facie an "
    "`accumulator`; a segment whose target or destination is fixed by the episode goal rather than "
    "by what is visible is `instruction_binding`."
)

SPECS = {
    "caption": {"system": SYSTEM, "suffix": ".captions.json", "key": "segments", "per_segment_request": False},
    "descriptor": {
        "system": DESCRIPTOR_SYSTEM,
        "suffix": ".descriptors.json",
        "key": "descriptors",
        "per_segment_request": True,
    },
}


def system_for(spec: str) -> str:
    return SPECS[spec]["system"]


def prompt_sha(spec: str = "caption", n_views: int = 3) -> str:
    """Content address of the FULL prompt for `spec`.

    `caption` stays sha256(SYSTEM) verbatim so the sealed H13 store keeps validating against it
    (verified: matches `wsm_labels_captions/**/ep_*.captions.json`). `descriptor` covers the system
    prompt AND the user-head template, because for that spec the head carries load-bearing content.
    """
    if spec == "descriptor":
        sysmsg = DESCRIPTOR_SYSTEM_2VIEW if n_views == 2 else DESCRIPTOR_SYSTEM
        return hashlib.sha256((sysmsg + "\n---HEAD---\n" + DESCRIPTOR_HEAD).encode()).hexdigest()
    return hashlib.sha256(system_for(spec).encode()).hexdigest()


def schema_sha(spec: str = "descriptor") -> str:
    if spec != "descriptor":
        return ""
    return hashlib.sha256(json.dumps(DESCRIPTOR_SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------- segments
def segments_from_keyframes(keyframes, n_frames: int) -> list[tuple[int, int]]:
    """Frozen `keyframes` -> [(t0, t1)) segments, matching data.py::per_frame_subgoal_idx.

    NOTE the LAST keyframe opens no boundary: everything after it is clamped back into the final
    subgoal, so K keyframes yield exactly K segments and the post-completion tail rides along in
    the last one. Only keyframes[:-1] become interior cut points.

    Defensive against label-store noise: keyframes are sorted, de-duplicated and clipped into
    [0, T-1], so every emitted segment is non-empty. Falls back to one whole-episode segment if
    nothing usable survives.
    """
    T = int(n_frames)
    ks = sorted({min(max(int(k), 0), T - 1) for k in np.asarray(keyframes).ravel().tolist()})
    if not ks:
        return [(0, T)]
    interior = sorted({k + 1 for k in ks[:-1] if 0 < k + 1 < T})
    bounds = [0] + interior + [T]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def segment_frames(t0: int, t1: int, per_seg: int) -> list[int]:
    """start / mid / end frame indices inside [t0, t1), de-duplicated, ascending."""
    if per_seg <= 1:
        return [(t0 + t1 - 1) // 2]
    if per_seg == 2:
        cand = [t0, t1 - 1]
    else:
        cand = [t0, (t0 + t1 - 1) // 2, t1 - 1]
    return sorted(dict.fromkeys(int(c) for c in cand))


def plan_frames(segs: list[tuple[int, int]], max_images: int) -> tuple[list[list[int]], int]:
    """Pick frames/segment so total images (frames x 3 views) stays under the budget."""
    for per_seg in (3, 2, 1):
        plan = [segment_frames(a, b, per_seg) for a, b in segs]
        if sum(len(p) for p in plan) * len(VIEWS) <= max_images:
            return plan, per_seg
    return [segment_frames(a, b, 1) for a, b in segs], 1


# ------------------------------------------------------------------------------------------ jobs
@dataclass
class Job:
    task: str
    ep: int
    n_frames: int
    keyframes: list
    segs: list
    plan: list
    root: Path
    out_path: Path
    frames: dict = field(default_factory=dict)  # view -> {frame_idx: np.ndarray}
    error: str = ""
    hints: list = field(default_factory=list)  # prior 30-token captions, one per segment (or [])
    instruction: str = ""  # per-episode language goal (meta/episodes.jsonl)
    fps: float = 20.0
    domain: str = "robocasa"  # robocasa | remembench | robomme | robocerebra
    views: tuple = VIEWS  # RoboMME and RoboCerebra have 2 views, not 3
    view_captions: dict = field(default_factory=lambda: dict(CAPTION))


def hints_for(hints_root: Path | None, task: str, ep: int, n_seg: int) -> list:
    """The existing H13 caption per segment, when the caption store has this episode.

    A hint is a prior, never a constraint: a missing/short/stale caption file simply yields [].
    The segmentation itself always comes from the frozen npz, so hints can never move a boundary.
    """
    if hints_root is None:
        return []
    p = hints_root / task / f"ep_{ep:06d}.captions.json"
    if not p.exists():
        return []
    try:
        segs = json.loads(p.read_text())["segments"]
    except Exception:
        return []
    if len(segs) != n_seg:
        return []
    return [str(s.get("text", "")) for s in segs]


def resolve_lerobot_dir(dataset_root: Path, task: str) -> Path | None:
    hits = sorted(dataset_root.glob(f"*/{task}/*/lerobot"))
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def read_label(path: Path) -> tuple[list, int]:
    d = np.load(path, allow_pickle=True)
    return [int(k) for k in np.asarray(d["keyframes"]).ravel().tolist()], int(d["n_frames"])


def validate_existing(path: Path, n_seg: int, T: int) -> bool:
    """Resume gate: the file must PARSE and structurally match this episode, not merely exist."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    segs = d.get("segments")
    if not isinstance(segs, list) or len(segs) != n_seg:
        return False
    prev = 0
    for s in segs:
        try:
            t0, t1, text = int(s["t0"]), int(s["t1"]), str(s["text"])
        except Exception:
            return False
        if t0 != prev or t1 <= t0 or not text.strip():
            return False
        prev = t1
    return prev == T


def validate_descriptor_record(d: dict) -> str:
    """'' if the per-segment descriptor satisfies the frozen schema, else the first violation."""
    if not isinstance(d, dict):
        return "not an object"
    for k in DESCRIPTOR_SCHEMA["required"]:
        if k not in d:
            return f"missing {k}"
    if not str(d["subskill"]).strip():
        return "empty subskill"
    if not str(d["verb_frame"]).strip():
        return "empty verb_frame"
    if not str(d["spatial_relation"]).strip():
        return "empty spatial_relation"
    tgt = d["target_object"]
    if not isinstance(tgt, dict):
        return "target_object not an object"
    for k in ("class", "attributes", "state_before", "state_after"):
        if k not in tgt:
            return f"target_object missing {k}"
    if not str(tgt["class"]).strip():
        return "empty target_object.class"
    if not isinstance(tgt["attributes"], list):
        return "target_object.attributes not a list"
    for k in ("preconditions", "postconditions", "failure_lookalikes"):
        if not isinstance(d[k], list):
            return f"{k} not a list"
    if not d["preconditions"] or not d["postconditions"]:
        return "empty pre/postconditions"
    md = d["memory_dependency"]
    if not isinstance(md, dict):
        return "memory_dependency not an object"
    for k in ("depends_on_history", "kinds", "evidence"):
        if k not in md:
            return f"memory_dependency missing {k}"
    if not isinstance(md["depends_on_history"], bool):
        return "depends_on_history not a bool"
    kinds = md["kinds"]
    if not isinstance(kinds, list) or not kinds:
        return "memory_dependency.kinds not a non-empty list"
    bad = [k for k in kinds if k not in MEMORY_KINDS]
    if bad:
        return f"memory_dependency.kinds {bad!r} not in {MEMORY_KINDS}"
    if len(set(kinds)) != len(kinds):
        return f"memory_dependency.kinds has duplicates: {kinds!r}"
    # "none" is a claim that NOTHING applies, so it cannot ride alongside a real mechanism.
    if "none" in kinds and len(kinds) > 1:
        return f"memory_dependency.kinds mixes 'none' with {kinds!r}"
    # the one cross-field consistency rule the whole LCR annotation rests on
    if bool(md["depends_on_history"]) != (kinds != ["none"]):
        return f"depends_on_history {md['depends_on_history']} disagrees with kinds {kinds!r}"
    if kinds != ["none"] and not str(md.get("evidence") or "").strip():
        return "memory_dependency.kinds != ['none'] but evidence empty"
    return ""


def memory_kinds_of(desc: dict) -> list:
    """The v2.1 `kinds` list, tolerating a v1/v2 store's scalar `kind`.

    Every consumer goes through this so a mixed-vintage store cannot silently read as all-`none`.
    """
    md = (desc or {}).get("memory_dependency") or {}
    ks = md.get("kinds")
    if isinstance(ks, list) and ks:
        return [str(k) for k in ks]
    k = md.get("kind")
    return [str(k)] if k else ["none"]


def has_memory_dependency(desc: dict) -> bool:
    return memory_kinds_of(desc) != ["none"]


def validate_existing_descriptors(path: Path, n_seg: int, T: int) -> bool:
    """Resume gate for --spec descriptor: structural, mirrors validate_existing (s3 §4 / A7)."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    recs = d.get("descriptors")
    if not isinstance(recs, list) or len(recs) != n_seg:
        return False
    prev = 0
    for r in recs:
        if not isinstance(r, dict):
            return False
        try:
            t0, t1 = int(r["t0"]), int(r["t1"])
        except Exception:
            return False
        if t0 != prev or t1 <= t0:
            return False
        if validate_descriptor_record(r.get("descriptor")):
            return False
        prev = t1
    return prev == T


def validate_for_spec(spec: str, path: Path, n_seg: int, T: int) -> bool:
    return (validate_existing_descriptors if spec == "descriptor" else validate_existing)(path, n_seg, T)


def episode_allowlist(args) -> set | None:
    """`--episodes` = a TOP-UP scope, read from a list or from a JSON receipt.

    Built for the 2026-08-28 RoboMME top-up: 33 ButtonUnmaskSwap episodes that pass 1 skipped as
    unreadable and that a cache repair made readable. Pointing this at the receipt itself
    (`_robomme_unreadable.RESOLVED.json`, key `topup_episodes`) keeps the scope auditable — no
    hand-retyped episode list can drift from the note that justifies the run.

    It is REFUSED with --num-shards > 1: sharding partitions the corpus, and an allowlist changes
    the pool being partitioned, so shard k would no longer own its contracted episode set. A top-up
    runs single-shard; the resume gate still protects it from redoing valid work.
    """
    raw = getattr(args, "episodes", "") or ""
    if not raw:
        return None
    if getattr(args, "num_shards", 1) != 1:
        raise SystemExit(
            "--episodes requires --num-shards 1 (an allowlist changes the pool that "
            "sharding partitions; shard k would stop owning its contracted set)"
        )
    if str(raw).endswith(".json"):
        blob = json.loads(Path(str(raw)).expanduser().read_text())
        ids = blob.get("topup_episodes") if isinstance(blob, dict) else blob
    else:
        ids = [x for x in str(raw).replace(" ", "").split(",") if x]
    return {int(x) for x in ids}


def build_robomme_jobs(args, tasks: list[str]) -> list[Job]:
    """RoboMME: no MP4s and no keyframe npz -- segments come from RLE of the official subgoal
    column, frames from image bytes embedded in the parquet."""
    from workspace_models.labels import robomme_source as RM

    root = Path(args.dataset_root).expanduser()
    out_root = Path(args.out).expanduser()
    idx = RM.build_index(root, tasks, out_root / "_robomme_index.json")
    allow = episode_allowlist(args)
    jobs, skipped = [], 0
    for ep_s, rec in sorted(idx.items(), key=lambda kv: int(kv[0])):
        ep = int(ep_s)
        if rec["task"] not in tasks:
            continue
        if allow is not None and ep not in allow:
            continue
        segs = [(int(a), int(b)) for a, b in rec["segments"]]
        if not segs:
            continue
        T = int(rec["n_frames"])
        out_path = out_root / rec["task"] / f"ep_{ep:06d}{SPECS[args.spec]['suffix']}"
        # resume gate deferred until after sharding -- see the note in build_jobs
        plan = [segment_frames(a, b, args.frames_per_segment) for a, b in segs]
        jobs.append(
            Job(
                rec["task"],
                ep,
                T,
                [b for _, b in segs],
                segs,
                plan,
                RM.episode_path(root, ep),
                out_path,
                hints=list(rec.get("hints") or []),
                instruction=str(rec.get("instruction", "")),
                fps=RM.FPS,
                domain="robomme",
                views=RM.VIEWS,
                view_captions=dict(RM.CAPTION),
            )
        )
    jobs.sort(key=lambda j: (j.task, j.ep))
    if getattr(args, "stratify_tasks", False):
        buckets: dict[str, list[Job]] = {}
        for j in jobs:
            buckets.setdefault(j.task, []).append(j)
        order = sorted(buckets)
        jobs = []
        for i in range(max(len(v) for v in buckets.values())):
            for t in order:
                if i < len(buckets[t]):
                    jobs.append(buckets[t][i])
    jobs = jobs[args.shard :: args.num_shards]  # STABLE partition of the WHOLE corpus
    kept = []
    for j in jobs:
        if (
            not args.force
            and j.out_path.exists()
            and validate_for_spec(args.spec, j.out_path, len(j.segs), j.n_frames)
        ):
            skipped += 1
            continue
        kept.append(j)
    jobs = kept
    if args.limit:
        jobs = jobs[: args.limit]
    if getattr(args, "limit_segments", 0):
        keep, tot = [], 0
        for j in jobs:
            if tot >= args.limit_segments:
                break
            keep.append(j)
            tot += len(j.segs)
        jobs = keep
    print(f"[shard {args.shard}/{args.num_shards}] robomme: {len(jobs)} to do, {skipped} already valid", flush=True)
    return jobs


def build_robocerebra_jobs(args, tasks: list[str]) -> list[Job]:
    """RoboCerebra: an ordinary MP4 LeRobot tree, but 2-view, with no keyframe npz — segments come
    from runs of the official per-frame `subtask_index` column (see `robocerebra_source`)."""
    from workspace_models.labels import robocerebra_source as RC

    root = Path(args.dataset_root).expanduser()
    out_root = Path(args.out).expanduser()
    idx = RC.build_index(root, out_root / "_robocerebra_index.json")
    allow = episode_allowlist(args)
    want = set(tasks) if tasks else None
    jobs, skipped = [], 0
    for ep_s, rec in sorted(idx.items(), key=lambda kv: int(kv[0])):
        ep = int(ep_s)
        if want is not None and rec["task"] not in want:
            continue
        if allow is not None and ep not in allow:
            continue
        segs = [(int(a), int(b)) for a, b in rec["segments"]]
        if not segs:
            continue
        out_path = out_root / rec["task"] / f"ep_{ep:06d}{SPECS[args.spec]['suffix']}"
        # resume gate deferred until after sharding -- see the note in build_jobs
        plan = [segment_frames(a, b, args.frames_per_segment) for a, b in segs]
        jobs.append(
            Job(
                rec["task"],
                ep,
                int(rec["n_frames"]),
                [b for _, b in segs],
                segs,
                plan,
                root,
                out_path,
                hints=list(rec.get("hints") or []),
                instruction=str(rec.get("instruction", "")),
                fps=RC.FPS,
                domain="robocerebra",
                views=RC.VIEWS,
                view_captions=dict(RC.CAPTION),
            )
        )
    jobs.sort(key=lambda j: (j.task, j.ep))
    if getattr(args, "stratify_tasks", False):
        buckets: dict[str, list[Job]] = {}
        for j in jobs:
            buckets.setdefault(j.task, []).append(j)
        order = sorted(buckets)
        jobs = []
        for i in range(max(len(v) for v in buckets.values())):
            for t in order:
                if i < len(buckets[t]):
                    jobs.append(buckets[t][i])
    jobs = jobs[args.shard :: args.num_shards]  # STABLE partition of the WHOLE corpus
    kept = []
    for j in jobs:
        if (
            not args.force
            and j.out_path.exists()
            and validate_for_spec(args.spec, j.out_path, len(j.segs), j.n_frames)
        ):
            skipped += 1
            continue
        kept.append(j)
    jobs = kept
    if args.limit:
        jobs = jobs[: args.limit]
    if getattr(args, "limit_segments", 0):
        keep, tot = [], 0
        for j in jobs:
            if tot >= args.limit_segments:
                break
            keep.append(j)
            tot += len(j.segs)
        jobs = keep
    print(
        f"[shard {args.shard}/{args.num_shards}] robocerebra: {len(jobs)} to do, {skipped} already valid", flush=True
    )
    return jobs


def build_jobs(args, tasks: list[str]) -> list[Job]:
    if getattr(args, "domain", "robocasa") == "robomme":
        return build_robomme_jobs(args, tasks)
    if getattr(args, "domain", "robocasa") == "robocerebra":
        return build_robocerebra_jobs(args, tasks)
    labels_root = Path(args.labels_root).expanduser()
    dataset_root = Path(args.dataset_root).expanduser()
    out_root = Path(args.out).expanduser()
    spec = args.spec
    hints_root = (
        Path(args.caption_hints_root).expanduser() if spec == "descriptor" and args.caption_hints_root else None
    )
    allow = episode_allowlist(args)
    jobs, skipped, missing = [], 0, []
    for task in tasks:
        root = resolve_lerobot_dir(dataset_root, task)
        if root is None:
            missing.append(task)
            continue
        try:
            ep_meta, ep_info = load_episode_meta(root)
        except Exception as e:
            missing.append(f"{task} (meta: {e})")
            continue
        fps = float(ep_info.get("fps") or 20)
        for lab in sorted((labels_root / task).glob("vlm_episode_pi_*.npz")):
            ep = int(re.search(r"(\d+)\.npz$", lab.name).group(1))
            if allow is not None and ep not in allow:
                continue
            meta = ep_meta.get(ep)
            if meta is None:
                missing.append(f"{task}/ep{ep}")
                continue
            kf, n_lab = read_label(lab)
            T = int(meta["length"])
            if n_lab != T:
                print(f"[warn] {task}/ep{ep}: label n_frames={n_lab} != meta length={T}; using meta", flush=True)
            segs = segments_from_keyframes(kf, T)
            out_path = out_root / task / f"ep_{ep:06d}{SPECS[spec]['suffix']}"
            # NOTE: the resume gate is deliberately NOT applied here. It runs AFTER sharding
            # (see below), because filtering first makes each shard partition a SHRINKING pool:
            # shard k would own different episodes on every invocation, one sweep of 8 sequential
            # shards would cover only 1-(7/8)^8 = 65.6% of the corpus, and the cross-venue handoff
            # contract ("shard k owns a fixed episode set") would be silently false.
            # Measured on 2026-08-22: a full 24-shard sweep produced 2,548/3,840 = 66.4%.
            if SPECS[spec]["per_segment_request"]:
                # one request per segment -> the image budget is per REQUEST, always 3 frames.
                plan = [segment_frames(a, b, args.frames_per_segment) for a, b in segs]
            else:
                plan, _ = plan_frames(segs, args.max_images)
            jobs.append(
                Job(
                    task,
                    ep,
                    T,
                    kf,
                    segs,
                    plan,
                    root,
                    out_path,
                    hints=hints_for(hints_root, task, ep, len(segs)),
                    instruction=str(meta.get("prompt", "")),
                    fps=fps,
                    domain=getattr(args, "domain", "robocasa"),
                )
            )
    jobs.sort(key=lambda j: (j.task, j.ep))
    if getattr(args, "stratify_tasks", False):
        # round-robin across tasks BEFORE any cap, so a --limit-segments canary is representative
        # of the suite rather than 100 episodes of whichever task sorts first.
        buckets: dict[str, list[Job]] = {}
        for j in jobs:
            buckets.setdefault(j.task, []).append(j)
        order = sorted(buckets)
        jobs = []
        for i in range(max(len(v) for v in buckets.values())):
            for t in order:
                if i < len(buckets[t]):
                    jobs.append(buckets[t][i])
    jobs = jobs[args.shard :: args.num_shards]  # STABLE partition of the WHOLE corpus
    # resume gate, applied to this shard's fixed episode set
    kept = []
    for j in jobs:
        if not args.force and j.out_path.exists() and validate_for_spec(spec, j.out_path, len(j.segs), j.n_frames):
            skipped += 1
            continue
        kept.append(j)
    jobs = kept
    if args.limit:
        jobs = jobs[: args.limit]
    if getattr(args, "limit_segments", 0):
        # canary sizing is in SEGMENTS (the unit the cost model and the pass-2 bucket use)
        keep, tot = [], 0
        for j in jobs:
            if tot >= args.limit_segments:
                break
            keep.append(j)
            tot += len(j.segs)
        jobs = keep
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(jobs)} to do, {skipped} already valid, "
        f"{len(missing)} unresolved",
        flush=True,
    )
    if missing:
        print(f"[warn] unresolved: {missing[:5]}{' ...' if len(missing) > 5 else ''}", flush=True)
    return jobs


# ---------------------------------------------------------------------------------------- decode
_INFO_CACHE: dict[str, dict] = {}


def _info(root: Path) -> dict:
    key = str(root)
    if key not in _INFO_CACHE:
        _INFO_CACHE[key] = json.loads((root / "meta" / "info.json").read_text())
    return _INFO_CACHE[key]


def decode_views(job: Job) -> Job:
    """Fill job.frames with the planned frames. RoboCasa/ReMemBench decode MP4 with PyAV;
    RoboMME has no MP4s and reads image bytes out of the parquet instead."""
    if getattr(job, "domain", "robocasa") == "robomme":
        from workspace_models.labels import robomme_source as RM

        return RM.decode_views(job)
    if getattr(job, "domain", "robocasa") == "robocerebra":
        from workspace_models.labels import robocerebra_source as RC

        return RC.decode_views(job)
    import av

    wanted = sorted({int(f) for p in job.plan for f in p})
    try:
        info = _info(job.root)
        stop = max(wanted)
        for view in VIEWS:
            path = episode_video_path(job.root, info, VIEW_LEROBOT_KEY[view], job.ep)
            got, need = {}, set(wanted)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                pos = 0
                for frame in container.decode(stream):
                    if pos in need:
                        got[pos] = frame.to_ndarray(format="rgb24")
                        need.discard(pos)
                        if not need:
                            break
                    if pos > stop:
                        break
                    pos += 1
            if need:
                raise RuntimeError(f"{path.name}: frames {sorted(need)[:5]} not decoded")
            job.frames[view] = got
    except Exception as e:
        job.error = f"decode: {e}"
    return job


# ---------------------------------------------------------------------------------------- prompt
def build_messages(job: Job):
    from PIL import Image

    caps = [CAPTION[v] for v in VIEWS]
    head = (
        f"Task: {job.task}\n"
        f"The episode has {len(job.segs)} segments, shown in order below "
        f"({', '.join(caps)} views per frame).\n"
        f"Write exactly {len(job.segs)} captions."
    )
    content = [{"type": "text", "text": head}]
    for si, ((t0, t1), fidx) in enumerate(zip(job.segs, job.plan)):
        content.append({"type": "text", "text": f"\n=== segment {si} (frames {t0}..{t1 - 1}) ==="})
        for f in fidx:
            for j, view in enumerate(VIEWS):
                tag = f"\nframe {f} {caps[j]}:" if j == 0 else f" {caps[j]}:"
                content.append({"type": "text", "text": tag})
                content.append({"type": "image", "image": Image.fromarray(job.frames[view][f])})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]


# ------------------------------------------------------------------------------- H14 vLLM client
def _png_data_url(arr) -> str:
    """RGB uint8 -> a lossless base64 PNG data URL (lossless keeps the store content-addressable)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False, compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_descriptor_messages(job: Job, si: int) -> list:
    """ONE segment -> one chat request: 3 frames x 3 views + task/instruction context."""
    t0, t1 = job.segs[si]
    views = job.views or VIEWS
    caps = [job.view_captions.get(v, v) for v in views]
    hint = job.hints[si] if job.hints and si < len(job.hints) else ""
    prior = f"\nA prior short label for this segment (a hint, may be wrong): {hint}" if hint else ""
    fps = job.fps or 20.0
    T = max(int(job.n_frames), 1)
    head = DESCRIPTOR_HEAD.format(
        task=job.task,
        instruction=job.instruction or "(not recorded)",
        si1=si + 1,
        nseg=len(job.segs),
        t0=t0,
        t1=t1 - 1,
        T=job.n_frames,
        dur=(t1 - t0) / fps,
        fps=fps,
        frac0=t0 / T,
        frac1=t1 / T,
        prior=prior,
        views=", ".join(caps),
    )
    content = [{"type": "text", "text": head}]
    for f in job.plan[si]:
        for j, view in enumerate(views):
            tag = f"\nframe {f} {caps[j]}:" if j == 0 else f" {caps[j]}:"
            content.append({"type": "text", "text": tag})
            content.append({"type": "image_url", "image_url": {"url": _png_data_url(job.frames[view][f])}})
    sysmsg = DESCRIPTOR_SYSTEM_2VIEW if len(views) == 2 else DESCRIPTOR_SYSTEM
    return [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": content},
    ]


def build_caption_messages_openai(job: Job) -> list:
    """The caption spec's episode-level prompt, re-expressed in OpenAI content parts."""
    caps = [CAPTION[v] for v in VIEWS]
    head = (
        f"Task: {job.task}\n"
        f"The episode has {len(job.segs)} segments, shown in order below "
        f"({', '.join(caps)} views per frame).\n"
        f"Write exactly {len(job.segs)} captions."
    )
    content = [{"type": "text", "text": head}]
    for si, ((t0, t1), fidx) in enumerate(zip(job.segs, job.plan)):
        content.append({"type": "text", "text": f"\n=== segment {si} (frames {t0}..{t1 - 1}) ==="})
        for f in fidx:
            for j, view in enumerate(VIEWS):
                tag = f"\nframe {f} {caps[j]}:" if j == 0 else f" {caps[j]}:"
                content.append({"type": "text", "text": tag})
                content.append({"type": "image_url", "image_url": {"url": _png_data_url(job.frames[view][f])}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


class VLLMChat:
    """Minimal OpenAI-compatible chat client (stdlib only, so it runs in ANY of our venvs).

    Returns (text, usage, finish_reason). `usage` is the server's own accounting -- every token
    number this campaign reports is MEASURED here, never estimated (A6).
    """

    def __init__(self, base_url: str, model: str, timeout: float = 600.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _post(self, path: str, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode()
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:400]
                last = RuntimeError(f"HTTP {e.code}: {body}")
                if e.code < 500:  # a 4xx is our bug -- retrying just burns time
                    raise last
            except Exception as e:  # noqa: BLE001 - transport flake; retry
                last = e
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise last  # type: ignore[misc]

    def chat(
        self,
        messages: list,
        *,
        max_tokens: int,
        reasoning_effort: str | None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: dict | None = None,
        schema_name: str = "out",
        seed: int | None = None,
    ) -> tuple[str, dict, str]:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            payload["seed"] = seed
        if reasoning_effort:
            # Qwen3.8's chat template reads `reasoning_effort` (default xhigh -> 4-8x cost, s3 risk 2)
            payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            }
        d = self._post("/chat/completions", payload)
        ch = d["choices"][0]
        msg = ch.get("message", {})
        text = msg.get("content") or ""
        usage = dict(d.get("usage", {}) or {})
        # vLLM 0.27's qwen3 reasoning parser puts the thinking block in `reasoning` (older builds:
        # `reasoning_content`). Empty content with non-empty reasoning means the token budget was
        # spent inside the thinking block -- surface it instead of letting it look like a parse bug.
        thinking = msg.get("reasoning") or msg.get("reasoning_content") or ""
        usage["reasoning_chars"] = len(thinking)
        usage["content_empty_but_reasoned"] = bool(thinking and not text)
        return text, usage, str(ch.get("finish_reason", ""))


def extract_descriptor(raw: str) -> dict:
    """Model output -> one schema-valid descriptor dict, or raise (caller resamples)."""
    d = parse_json(raw)
    if "descriptor" in d and isinstance(d["descriptor"], dict):
        d = d["descriptor"]
    why = validate_descriptor_record(d)
    if why:
        raise ValueError(why)
    return {k: d[k] for k in DESCRIPTOR_SCHEMA["required"]}


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))


_REFUSAL = re.compile(r"^(i (cannot|can't|am unable)|sorry|as an ai|unable to)", re.I)


def clean_caption(text: str) -> str:
    t = " ".join(str(text).split()).strip().strip('"').strip()
    t = re.sub(r"^(segment\s*\d+\s*[:.\-]\s*)", "", t, flags=re.I)
    return t.rstrip(".")


def extract_captions(raw: str, n_seg: int, tokenizer) -> list[str]:
    """Model output -> exactly n_seg clean captions, or raise (caller resamples)."""
    parsed = parse_json(raw)
    items = parsed.get("captions")
    if not isinstance(items, list):
        raise ValueError("no `captions` list")
    by_idx: dict[int, str] = {}
    for i, it in enumerate(items):
        if isinstance(it, dict):
            si = int(it.get("segment", i))
            txt = clean_caption(it.get("text", ""))
        else:
            si, txt = i, clean_caption(it)
        if 0 <= si < n_seg and si not in by_idx:
            by_idx[si] = txt
    out = []
    for si in range(n_seg):
        t = by_idx.get(si, "")
        if not t or _REFUSAL.match(t):
            raise ValueError(f"segment {si}: empty/refusal caption {t!r}")
        n_tok = len(tokenizer(t, add_special_tokens=False)["input_ids"])
        if n_tok > MAX_CAPTION_TOKENS:
            words = t.split()
            while (
                words and len(tokenizer(" ".join(words), add_special_tokens=False)["input_ids"]) > MAX_CAPTION_TOKENS
            ):
                words.pop()
            t = " ".join(words)
            if not t:
                raise ValueError(f"segment {si}: caption not truncatable under {MAX_CAPTION_TOKENS} tokens")
        out.append(t)
    return out


def write_output(job: Job, captions: list[str], model_id: str) -> None:
    rec = {
        "episode_id": job.ep,
        "task": job.task,
        "n_frames": job.n_frames,
        "keyframes": [int(k) for k in job.keyframes],
        "segments": [{"t0": int(a), "t1": int(b), "text": c} for (a, b), c in zip(job.segs, captions)],
        "model": model_id,
        "prompt_sha": prompt_sha(),
        "views_used": list(VIEWS),
    }
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = job.out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, job.out_path)  # atomic: a killed run never leaves a half-written json


def write_descriptor_output(job: Job, descs: list[dict], model_id: str, stats: list[dict]) -> None:
    rec = {
        "episode_id": job.ep,
        "task": job.task,
        "n_frames": job.n_frames,
        "keyframes": [int(k) for k in job.keyframes],
        "descriptors": [
            {
                "t0": int(a),
                "t1": int(b),
                "segment": i,
                "frames": [int(f) for f in job.plan[i]],
                "descriptor": d,
                "usage": u,
            }
            for i, ((a, b), d, u) in enumerate(zip(job.segs, descs, stats))
        ],
        "model": model_id,
        "spec": "descriptor",
        "domain": getattr(job, "domain", "robocasa"),
        "prompt_sha": prompt_sha("descriptor", len(job.views or VIEWS)),
        "schema_sha": schema_sha("descriptor"),
        "views_used": list(job.views or VIEWS),
    }
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = job.out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, job.out_path)


# ------------------------------------------------------------------------------------------ main
def run(args, tasks: list[str]) -> None:
    jobs = build_jobs(args, tasks)
    if not jobs:
        print("nothing to do", flush=True)
        return
    if args.backend == "vllm":
        run_vllm(args, jobs)
    else:
        run_hf(args, jobs)


def _episode_descriptors(client: VLLMChat, job: Job, args) -> tuple[list[dict] | None, list[dict]]:
    """All segments of one episode, issued as independent per-segment requests."""
    descs: list[dict] = []
    stats: list[dict] = []
    for si in range(len(job.segs)):
        msgs = build_descriptor_messages(job, si)
        got = None
        for attempt in range(args.retries + 1):
            try:
                text, usage, finish = client.chat(
                    msgs,
                    max_tokens=args.max_new_tokens,
                    reasoning_effort=args.reasoning_effort,
                    temperature=0.0 if attempt == 0 else 0.5,
                    json_schema=DESCRIPTOR_SCHEMA if args.structured_output else None,
                    schema_name="segment_descriptor",
                    seed=args.seed,
                )
                usage = dict(usage)
                usage["finish_reason"] = finish
                usage["attempt"] = attempt
                got = extract_descriptor(text)
                stats.append(usage)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == args.retries:
                    print(f"[{job.task}/ep{job.ep}/seg{si}] FAIL {type(e).__name__}: {str(e)[:180]}", flush=True)
                    return None, stats
        if got is None:
            return None, stats
        descs.append(got)
    return descs, stats


def run_vllm(args, jobs: list[Job]) -> None:
    """Concurrent OpenAI-compatible client. Decode stays threaded; the GPU is fed by CONCURRENCY,
    which is where vLLM's continuous batching lives (s3 §4: HF batch-4 was ~90 tok/s/GPU)."""
    from queue import Queue

    client = VLLMChat(args.vllm_base_url, args.model, timeout=args.request_timeout, retries=args.http_retries)
    decode_pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    todo: Queue = Queue()
    for j in jobs:
        todo.put(j)

    done = failed = 0
    n_seg_done = 0
    tok_in = tok_out = 0
    usage_log: list[dict] = []
    lock = __import__("threading").Lock()
    t_start = time.time()

    def worker() -> None:
        nonlocal done, failed, n_seg_done, tok_in, tok_out
        while True:
            try:
                job = todo.get_nowait()
            except Exception:  # noqa: BLE001 - Empty
                return
            try:
                decode_pool.submit(decode_views, job).result()
                if job.error:
                    with lock:
                        failed += 1
                    print(f"[{job.task}/ep{job.ep}] SKIP {job.error}", flush=True)
                    continue
                if args.spec == "descriptor":
                    descs, stats = _episode_descriptors(client, job, args)
                    ok = descs is not None
                    if ok:
                        write_descriptor_output(job, descs, args.model, stats)
                else:
                    msgs = build_caption_messages_openai(job)
                    text, usage, finish = client.chat(
                        msgs, max_tokens=args.max_new_tokens, reasoning_effort=args.reasoning_effort, seed=args.seed
                    )
                    usage = dict(usage)
                    usage["finish_reason"] = finish
                    stats = [usage]
                    caps = _captions_from_text(text, len(job.segs))
                    ok = caps is not None
                    if ok:
                        write_output(job, caps, args.model)
                with lock:
                    for u in stats:
                        tok_in += int(u.get("prompt_tokens", 0) or 0)
                        tok_out += int(u.get("completion_tokens", 0) or 0)
                        usage_log.append(u)
                    if ok:
                        done += 1
                        n_seg_done += len(job.segs)
                    else:
                        failed += 1
            except Exception as e:  # noqa: BLE001
                with lock:
                    failed += 1
                print(f"[{job.task}/ep{job.ep}] ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            finally:
                job.frames.clear()
                if (done + failed) % args.log_every == 0:
                    el = max(time.time() - t_start, 1e-6)
                    print(
                        f"[{args.shard}] {done}/{len(jobs)} ep ok ({n_seg_done} seg), "
                        f"{failed} failed, {n_seg_done / el * 60:.1f} seg/min, "
                        f"{tok_out / el:.0f} out-tok/s",
                        flush=True,
                    )

    threads = [__import__("threading").Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    decode_pool.shutdown(wait=False)

    el = max(time.time() - t_start, 1e-6)
    summary = {
        "backend": "vllm",
        "spec": args.spec,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "structured_output": bool(args.structured_output),
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "episodes_ok": done,
        "episodes_failed": failed,
        "segments_ok": n_seg_done,
        "wall_seconds": round(el, 2),
        "prompt_tokens": tok_in,
        "completion_tokens": tok_out,
        "segments_per_min": round(n_seg_done / el * 60, 2),
        "out_tok_per_s": round(tok_out / el, 1),
        "prompt_tokens_per_segment": round(tok_in / max(n_seg_done, 1), 1),
        "completion_tokens_per_segment": round(tok_out / max(n_seg_done, 1), 1),
        "truncated_requests": sum(1 for u in usage_log if u.get("finish_reason") == "length"),
        "n_requests": len(usage_log),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    out_root = Path(args.out).expanduser()
    (out_root / "_provenance").mkdir(parents=True, exist_ok=True)
    (out_root / "_provenance" / f"usage_shard{args.shard}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps({"summary": summary, "requests": usage_log}, indent=1)
    )


def _captions_from_text(text: str, n_seg: int) -> list[str] | None:
    """Tokenizer-free caption extraction for the vLLM path (word cap instead of a token cap)."""
    try:
        parsed = parse_json(text)
        items = parsed.get("captions")
        if not isinstance(items, list):
            return None
        by_idx: dict[int, str] = {}
        for i, it in enumerate(items):
            if isinstance(it, dict):
                si, txt = int(it.get("segment", i)), clean_caption(it.get("text", ""))
            else:
                si, txt = i, clean_caption(it)
            if 0 <= si < n_seg and si not in by_idx:
                by_idx[si] = txt
        out = []
        for si in range(n_seg):
            t = by_idx.get(si, "")
            if not t or _REFUSAL.match(t):
                return None
            out.append(" ".join(t.split()[:MAX_CAPTION_TOKENS]))
        return out
    except Exception:  # noqa: BLE001
        return None


def run_hf(args, jobs: list[Job]) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device, local_files_only=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"  # required for correct batched generation
    tokenizer = processor.tokenizer

    done = failed = 0
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    pending = [pool.submit(decode_views, j) for j in jobs[: args.prefetch]]
    nxt = args.prefetch
    cursor = 0

    while cursor < len(pending):
        batch: list[Job] = []
        while len(batch) < args.batch_size and cursor < len(pending):
            job = pending[cursor].result()
            cursor += 1
            if nxt < len(jobs):  # keep the decode pipeline full
                pending.append(pool.submit(decode_views, jobs[nxt]))
                nxt += 1
            if job.error:
                print(f"[{job.task}/ep{job.ep}] SKIP {job.error}", flush=True)
                failed += 1
                continue
            batch.append(job)
        if not batch:
            continue

        results = generate_batch(model, processor, tokenizer, batch, args)
        for job, caps in zip(batch, results):
            if caps is None:
                failed += 1
                continue
            write_output(job, caps, args.model)
            job.frames.clear()
            done += 1
        el = time.time() - t_start
        rate = done / el if el > 0 else 0
        print(
            f"[{args.shard}] {done}/{len(jobs)} ok, {failed} failed, "
            f"{rate * 60:.1f} ep/min, eta {(len(jobs) - done) / rate / 60:.0f} min"
            if rate > 0
            else f"[{args.shard}] {done}/{len(jobs)}",
            flush=True,
        )

    pool.shutdown(wait=False)
    el = time.time() - t_start
    print(
        f"[shard {args.shard}] DONE {done} ok / {failed} failed in {el / 60:.1f} min ({done / el * 60:.1f} ep/min)"
        if el > 0
        else "done",
        flush=True,
    )


def generate_batch(model, processor, tokenizer, batch: list[Job], args) -> list[list[str] | None]:
    """Batched greedy pass + per-episode resample on malformed output. Falls back to one-at-a-time
    if the batched path raises (variable image counts across a batch)."""
    import torch

    out: list[list[str] | None] = [None] * len(batch)
    convs = [build_messages(j) for j in batch]

    def _gen(conv_list, do_sample: bool):
        inputs = processor.apply_chat_template(
            conv_list,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=0.4 if do_sample else None,
            )
        n_in = inputs["input_ids"].shape[1]
        return [processor.decode(g[n_in:], skip_special_tokens=True) for g in gen]

    try:
        texts = _gen(convs, False)
    except Exception as e:
        if len(batch) > 1:
            print(f"[warn] batched generate failed ({e}); falling back to size 1", flush=True)
            texts = []
            for c in convs:
                try:
                    texts.append(_gen([c], False)[0])
                except Exception as e2:
                    print(f"[warn] single generate failed: {e2}", flush=True)
                    texts.append("")
        else:
            print(f"[warn] generate failed: {e}", flush=True)
            texts = [""]

    for i, (job, txt) in enumerate(zip(batch, texts)):
        n_seg = len(job.segs)
        for attempt in range(args.retries + 1):
            try:
                out[i] = extract_captions(txt, n_seg, tokenizer)
                break
            except Exception as e:
                if attempt == args.retries:
                    print(f"[{job.task}/ep{job.ep}] PARSE FAIL: {e} | raw={txt[:160]!r}", flush=True)
                    break
                txt = _gen([convs[i]], True)[0]
    return out


def finalize(args, tasks: list[str]) -> None:
    """Per-task manifest.json over whatever validated successfully."""
    out_root = Path(args.out).expanduser()
    labels_root = Path(args.labels_root).expanduser()
    spec = getattr(args, "spec", "caption")
    suffix, key = SPECS[spec]["suffix"], SPECS[spec]["key"]
    grand = {"tasks": {}, "total_expected": 0, "total_written": 0}
    for task in tasks:
        expected = sorted(
            int(re.search(r"(\d+)\.npz$", p.name).group(1)) for p in (labels_root / task).glob("vlm_episode_pi_*.npz")
        )
        eps, n_seg_hist = [], {}
        for ep in expected:
            p = out_root / task / f"ep_{ep:06d}{suffix}"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            eps.append(ep)
            k = len(d[key])
            n_seg_hist[k] = n_seg_hist.get(k, 0) + 1
        man = {
            "task": task,
            "model": args.model,
            "spec": spec,
            "prompt_sha": prompt_sha(spec),
            "views_used": list(VIEWS),
            "n_expected": len(expected),
            "n_written": len(eps),
            "missing": [e for e in expected if e not in set(eps)],
            "segments_per_episode_hist": {str(k): v for k, v in sorted(n_seg_hist.items())},
            "episodes": eps,
        }
        if eps:
            (out_root / task).mkdir(parents=True, exist_ok=True)
            (out_root / task / "manifest.json").write_text(json.dumps(man, indent=1))
        grand["tasks"][task] = {"expected": len(expected), "written": len(eps)}
        grand["total_expected"] += len(expected)
        grand["total_written"] += len(eps)
    (out_root / "_provenance").mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(grand, indent=1))
    print(f"finalize: {grand['total_written']}/{grand['total_expected']} episodes captioned", flush=True)


def write_provenance(args, tasks: list[str]) -> None:
    out_root = Path(args.out).expanduser()
    prov = out_root / "_provenance"
    prov.mkdir(parents=True, exist_ok=True)
    rec = {
        "model": args.model,
        "spec": args.spec,
        "prompt_sha": prompt_sha(args.spec),
        "system_prompt": system_for(args.spec),
        "schema_sha": schema_sha(args.spec),
        "code_sha256_16": code_sha(),
        "code_path": str(Path(__file__).resolve()),
        "views_used": list(VIEWS),
        "max_caption_tokens": MAX_CAPTION_TOKENS,
        "segmentation": "frozen keyframes; mirrors train_wsm_base/data.py::per_frame_subgoal_idx",
        "labels_root": str(Path(args.labels_root).expanduser()),
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "n_tasks": len(tasks),
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import subprocess

        rec["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:
        rec["git_head"] = "unknown"
    (prov / f"run_shard{args.shard}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rec, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="'all' or a comma-separated task list")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--labels-root",
        default=DEFAULT_LABELS_ROOT,
        help="mirror of the S3 label store (source of the FROZEN keyframes)",
    )
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--max-images", type=int, default=72, help="image budget per request (frames x 3 views); qwen_subgoals used 60"
    )
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="cap episodes (canary)")
    ap.add_argument(
        "--episodes",
        default="",
        help="TOP-UP scope: a comma-separated episode list, or a path to a JSON "
        "receipt (list, or dict with `topup_episodes`). Requires --num-shards 1.",
    )
    ap.add_argument("--force", action="store_true", help="recaption even if a valid file exists")
    ap.add_argument("--finalize-only", action="store_true")
    # ---------------------------------------------------------------- H14 additive flags (plan §3)
    ap.add_argument(
        "--spec",
        choices=sorted(SPECS),
        default="caption",
        help="caption = the H13 30-token imperative (default, unchanged); "
        "descriptor = the H14 structured segment descriptor",
    )
    ap.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="hf = transformers (default, unchanged); vllm = OpenAI-compatible client",
    )
    ap.add_argument("--vllm-base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument(
        "--concurrency", type=int, default=32, help="in-flight requests; this is what fills vLLM's continuous batch"
    )
    ap.add_argument(
        "--reasoning-effort",
        default="low",
        choices=("low", "medium", "xhigh", "off"),
        help="Qwen3.8 chat-template knob; the model default is xhigh = 4-8x cost",
    )
    ap.add_argument(
        "--structured-output",
        dest="structured_output",
        action="store_true",
        default=True,
        help="constrain decoding to the frozen descriptor JSON schema",
    )
    ap.add_argument("--no-structured-output", dest="structured_output", action="store_false")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--http-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument(
        "--frames-per-segment",
        type=int,
        default=3,
        help="descriptor spec: frames per request (x3 views = images per request)",
    )
    ap.add_argument(
        "--caption-hints-root",
        default=DEFAULT_OUT,
        help="descriptor spec: prior 30-token captions passed as a hint ('' disables)",
    )
    ap.add_argument("--limit-segments", type=int, default=0, help="canary cap in SEGMENTS (the cost model's unit)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--domain",
        choices=("robocasa", "remembench", "robomme", "robocerebra"),
        default="robocasa",
        help="robocasa/remembench share the MP4+keyframe-npz path; robomme reads "
        "parquet-embedded frames and RLE subgoal segmentation; robocerebra is an "
        "MP4 tree but 2-view and segmented by its subtask_index column",
    )
    ap.add_argument(
        "--stratify-tasks",
        action="store_true",
        help="round-robin episodes across tasks before --limit/--limit-segments",
    )
    args = ap.parse_args()

    if args.reasoning_effort == "off":
        args.reasoning_effort = None
    if args.spec == "descriptor":
        if args.backend != "vllm":
            raise SystemExit(
                "--spec descriptor requires --backend vllm (the HF path is the "
                "frozen H13 caption path and is deliberately left untouched)"
            )
        if args.max_new_tokens == 768:  # argparse default -> use the spec's hard cap
            args.max_new_tokens = DESCRIPTOR_MAX_TOKENS
        if args.max_new_tokens > DESCRIPTOR_MAX_TOKENS:
            raise SystemExit(
                f"--max-new-tokens {args.max_new_tokens} exceeds the frozen hard cap "
                f"{DESCRIPTOR_MAX_TOKENS} (plan §3 / s3 risk 2)"
            )

    labels_root = Path(args.labels_root).expanduser()
    if args.tasks == "all" and getattr(args, "domain", "robocasa") == "robocerebra":
        # RoboCerebra enumerates its own 947 BDDL stems from the dataset, and has NO keyframe label
        # store -- so `--tasks all` must NOT be resolved by listing labels_root. Doing so is wrong
        # twice over: on a node that directory does not exist (FileNotFoundError, instant client
        # death) and locally it yields ROBOCASA task names, which match nothing in the RoboCerebra
        # index and silently produce "0 to do, nothing to do" -- a job that succeeds having done
        # nothing. An empty list means "no filter" to build_robocerebra_jobs.
        tasks = []
    elif args.tasks == "all":
        tasks = sorted(p.name for p in labels_root.iterdir() if p.is_dir())
    else:
        tasks = [t for t in args.tasks.split(",") if t]
    if not tasks and getattr(args, "domain", "robocasa") != "robocerebra":
        raise SystemExit(f"no tasks under {labels_root}")

    if args.finalize_only:
        finalize(args, tasks)
        return
    write_provenance(args, tasks)
    run(args, tasks)
    if args.num_shards == 1:
        finalize(args, tasks)


if __name__ == "__main__":
    sys.exit(main())
