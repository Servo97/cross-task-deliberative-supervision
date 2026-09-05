#!/usr/bin/env python3
"""Per-task CORRECT vs DISTRACTOR target resolution — the memory-vs-control discriminator.

For every task the study touches, this file records (a) which object the policy is supposed
to move, (b) where it is supposed to go, (c) the plausible wrong place(s) it goes when it
ignored history, and (d) exactly how each was derived from the task's own source. Derivations
are quoted in ``docs`` so the README can be generated from here and nothing is folklore.

The candidate positions are resolved ONCE per episode, at reset, from the pinned scene —
except targets that genuinely move (SearingMeat's pan), which are re-read every step.

Classification (identical for every task, see ``classify``):
  at each step the AGENT (end-effector, or the manipulated object) has a nearest candidate;
  a candidate is COMMITTED TO at the first step where it is the nearest one AND within
  ``radius`` for ``consecutive`` steps running. Whichever of correct/distractor is committed
  to first names the rollout.
"""

from __future__ import annotations

import numpy as np
from fm_common import APPROACH_CONSECUTIVE_K, APPROACH_RADIUS_M

# --------------------------------------------------------------------------------------
# task specs
# --------------------------------------------------------------------------------------
# agent:      "ee" | "object:<name>"      what we track through space
# correct:    list of candidate specs     any one of these is the right place
# distractor: list of candidate specs     going here first means the wrong branch
#
# candidate spec kinds:
#   {"kind": "object", "name": ...}                body_xpos of a placed object
#   {"kind": "burner", "loc": ...}                 stove burner site ("@knob" = the episode's
#                                                  correct knob; "@others" expands to the rest)
#   {"kind": "burner_free"} / {"kind": "burner_occupied"}
#   {"kind": "fixture", "attr": ...}               fixture .pos
#   {"kind": "mirror", "of": ..., "about": ...}    point reflection in XY, for the
#                                                  left/right sibling-task location
TASK_SPECS = {
    # ---------------- ReMemBench ----------------
    "MemFruitInSinkRightFar": {
        "bench": "remembench",
        "agent": "ee",
        "manipulated": "fruit",
        "goal": [{"kind": "fixture", "attr": "sink"}],
        "correct": [{"kind": "object", "name": "fruit_container"}],
        "distractor": [
            {
                "kind": "mirror",
                "of": {"kind": "object", "name": "fruit_container"},
                "about": {"kind": "fixture", "attr": "sink"},
            }
        ],
        "docs": (
            "memory_env.py MemFruitInSinkRightFar: one fruit inside one fruit_container, "
            "placed on the sink counter with sample_region loc='right' and offset (+0.5, 0). "
            "The sibling task MemFruitInSinkLeftFar is byte-identical except offset (-0.5, 0). "
            "The two are visually indistinguishable once the robot has driven to the sink, so "
            "the memory demand is 'which side was the fruit on'. The distractor is therefore "
            "the mirror-image location: the container position reflected through the sink "
            "centre in the world XY plane, i.e. exactly where LeftFar would have put it. "
            "Approaching the mirror location first = the policy forgot the side."
        ),
    },
    "MemHeatPot": {
        "bench": "remembench",
        "agent": "ee",
        "manipulated": "meat",
        "goal": [{"kind": "burner", "loc": "@knob"}],
        "correct": [{"kind": "burner", "loc": "@knob"}],
        "distractor": [{"kind": "burner", "loc": "@others"}],
        "correct_knob": "front_left",
        "commitment_kind": "knob_state",
        "docs": (
            "memory_env.py MemHeatPot: the pan (meat_container) is placed at "
            "pan_location_on_stove = 'front_left', so the ONLY correct knob is front_left; "
            "the other three are physically adjacent and visually identical. "
            "Commitment here is measured on KNOB STATE, not on geometry: a knob is 'turned' "
            "when |angle| is in [0.35, 2pi-0.35], which is the env's own _check_stove_on "
            "test. Turning front_left first = approached_correct; turning any other knob "
            "first = approached_wrong; never turning one = no_commitment. "
            "GEOMETRY WAS TRIED FIRST AND REJECTED: EE distance to the front_left BURNER "
            "site labelled 0/20 expert demonstrations correct (18 wrong, 2 none), because "
            "the knobs live on the stove's front panel and no burner site coincides with "
            "one -- the expert's closest approach to the 'correct' burner is a median 30 cm. "
            "Knob state passes the same check. The burner-distance curves are still stored "
            "in the JSON as diagnostics, but they do not drive the label. "
            "The timing half of the task is reported separately as the prospective fields "
            "(turn_on/turn_off/wait-timer vs threshold, and failed_task)."
        ),
    },
    "MemWashAndReturnLeft": {
        "bench": "remembench",
        "agent": "object:fruit",
        "manipulated": "fruit",
        "goal": [{"kind": "fixture", "attr": "sink"}],
        "correct": [{"kind": "object", "name": "fruit_container"}],
        "distractor": [{"kind": "object", "name": "fruit_container2"}],
        "phase_gate": "place_success",
        "docs": (
            "memory_env.py MemWashAndReturn(Left): two identical container_set_train plates "
            "are placed, fruit_container on the LEFT of the sink (holding the fruit) and "
            "fruit_container2 on the RIGHT (empty). destination_container_name = "
            "'fruit_container', and the instruction is only 'Wash the fruit and return it to "
            "the container' — from the current observation alone the two plates are "
            "interchangeable, so returning to the right one requires remembering where the "
            "fruit came from. This is the cleanest memory-vs-control discriminator in the "
            "study. Tracked on the FRUIT (not the gripper) and gated on place_success, i.e. "
            "only the return leg after the fruit has touched the sink counts."
        ),
    },
    # ---------------- RoboCasa ----------------
    "KettleBoiling": {
        "bench": "robocasa",
        "agent": "object:obj",
        "manipulated": "obj",
        "goal": [{"kind": "burner_free"}],
        "correct": [{"kind": "burner_free"}],
        "distractor": [
            {"kind": "object", "name": "stove_distr"},
            {"kind": "burner_occupied"},
        ],
        "docs": (
            "kettle_boiling.py: places 'obj' (a non-electric kettle) on the counter and "
            "'stove_distr' (a pan or pot) ON a burner. _check_success wants the kettle in "
            "contact with the stove, within 0.15 m xy of SOME burner site, that burner's knob "
            "on, and the gripper away. Correct = the nearest burner NOT occupied by "
            "stove_distr; distractors = the explicitly named stove_distr cookware itself and "
            "the burner it already occupies. Tracked on the kettle."
        ),
    },
    "ScrubCuttingBoard": {
        "bench": "robocasa",
        "agent": "object:sponge",
        "manipulated": "sponge",
        "goal": [{"kind": "object", "name": "cutting_board"}],
        "correct": [{"kind": "object", "name": "cutting_board"}],
        "distractor": [{"kind": "fixture", "attr": "sink"}],
        "docs": (
            "scrub_cutting_board.py places exactly two objects, 'sponge' and 'cutting_board', "
            "and NO distractor — RoboCasa's use_distractors flag is stored but never read in "
            "this fork. The task registers self.sink purely as a spatial reference for "
            "choosing the counter, and 'clean the cutting board' plausibly invites carrying "
            "the sponge to the sink to rinse it, so the sink is the substitute distractor. "
            "Declared honestly: this is a derived distractor, not one the task author placed. "
            "Tracked on the sponge."
        ),
    },
    "SearingMeat": {
        "bench": "robocasa",
        "agent": "object:pan",
        "manipulated": "pan",
        "goal": [{"kind": "burner", "loc": "@knob"}],
        "correct": [{"kind": "burner", "loc": "@knob"}],
        "distractor": [{"kind": "burner", "loc": "@others"}],
        "secondary": {
            "agent": "object:meat",
            "correct": [{"kind": "object", "name": "pan", "dynamic": True}],
            "distractor": [{"kind": "object", "name": "meat_container"}],
        },
        "docs": (
            "searing_meat.py names ONE correct burner per episode: self.knob, drawn at reset "
            "and published as ep_meta['refs']['knob'], and the instruction spells it out "
            "('place it on the {knob} burner'). _check_success requires "
            "check_obj_location_on_stove(pan) == self.knob exactly. Primary commitment is "
            "therefore the pan against the named burner vs every other burner — the strongest "
            "distractor set in the study, since the wrong burners are adjacent and identical. "
            "A secondary commitment tracks the meat against the pan (correct, and it MOVES, "
            "so it is re-read every step) vs meat_container, the auto-generated plate the "
            "meat started on."
        ),
    },
}


# --------------------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------------------
def resolve_candidates(core, task: str, ep_meta: dict) -> dict:
    """{'correct': [(label, pos_or_None)], 'distractor': [...], 'goal': [...]} at reset.

    ``pos`` is None for dynamic candidates; those are re-read per step by
    :func:`dynamic_positions`.
    """
    import fm_env

    spec = TASK_SPECS[task]
    stove = fm_env.stove_probe(core)
    knob = _correct_knob(core, task, spec, ep_meta, stove)
    out = {}
    for role in ("goal", "correct", "distractor"):
        resolved = []
        for candidate in spec.get(role, []):
            resolved.extend(_resolve_one(core, candidate, stove, knob))
        out[role] = resolved
    out["correct_knob"] = knob
    out["stove_locations"] = list(stove[0]) if stove else []
    return out


def _correct_knob(core, task, spec, ep_meta, stove):
    if "correct_knob" in spec:
        return spec["correct_knob"]
    knob = getattr(core, "knob", None)
    if isinstance(knob, str):
        return knob
    refs = (ep_meta or {}).get("refs") or {}
    if isinstance(refs.get("knob"), str):
        return refs["knob"]
    pan_loc = getattr(core, "pan_location_on_stove", None)
    return pan_loc if isinstance(pan_loc, str) else None


def _resolve_one(core, candidate, stove, knob):
    import fm_env

    kind = candidate["kind"]
    if kind == "object":
        name = candidate["name"]
        if name not in (getattr(core, "obj_body_id", {}) or {}):
            return []
        if candidate.get("dynamic"):
            return [(f"obj:{name}", None, {"kind": "object", "name": name})]
        pos = fm_env.object_positions(core, [name])[0]
        return [(f"obj:{name}", pos, None)]
    if kind == "fixture":
        pos = fm_env.fixture_position(core, candidate["attr"])
        return [] if pos is None else [(f"fixture:{candidate['attr']}", pos, None)]
    if kind == "mirror":
        base = _resolve_one(core, candidate["of"], stove, knob)
        about = _resolve_one(core, candidate["about"], stove, knob)
        if not base or not about or base[0][1] is None or about[0][1] is None:
            return []
        origin = about[0][1]
        point = base[0][1]
        mirrored = point.copy()
        mirrored[:2] = 2.0 * origin[:2] - point[:2]
        return [(f"mirror:{base[0][0]}", mirrored, None)]
    if kind in ("burner", "burner_free", "burner_occupied"):
        if stove is None:
            return []
        locations, _angles, burners = stove
        if kind == "burner":
            loc = candidate["loc"]
            if loc == "@knob":
                return [(f"burner:{knob}", burners[locations.index(knob)], None)] if knob in locations else []
            if loc == "@others":
                return [
                    (f"burner:{name}", burners[index], None)
                    for index, name in enumerate(locations)
                    if name != knob and np.all(np.isfinite(burners[index]))
                ]
            return [(f"burner:{loc}", burners[locations.index(loc)], None)] if loc in locations else []
        occupier = _occupied_burner_index(core, locations, burners)
        picked = [
            (f"burner:{name}", burners[index], None)
            for index, name in enumerate(locations)
            if np.all(np.isfinite(burners[index]))
            and ((index == occupier) if kind == "burner_occupied" else (index != occupier))
        ]
        return picked
    raise ValueError(f"unknown candidate kind {kind!r}")


def _occupied_burner_index(core, locations, burners):
    """Index of the burner the KettleBoiling distractor cookware sits on, or -1."""
    import fm_env

    if "stove_distr" not in (getattr(core, "obj_body_id", {}) or {}):
        return -1
    distr = fm_env.object_positions(core, ["stove_distr"])[0]
    best, best_distance = -1, np.inf
    for index in range(len(locations)):
        if not np.all(np.isfinite(burners[index])):
            continue
        distance = float(np.linalg.norm(burners[index][:2] - distr[:2]))
        if distance < best_distance:
            best, best_distance = index, distance
    return best if best_distance < 0.20 else -1


def dynamic_positions(core, candidates) -> np.ndarray:
    """Positions for one resolved candidate list at the current sim state."""
    import fm_env

    out = []
    for _label, pos, dynamic in candidates:
        if dynamic is None:
            out.append(pos)
        else:
            out.append(fm_env.object_positions(core, [dynamic["name"]])[0])
    return np.stack(out) if out else np.zeros((0, 3))


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------
def classify(
    correct_distance: np.ndarray,
    distractor_distance: np.ndarray,
    *,
    radius: float = APPROACH_RADIUS_M,
    consecutive: int = APPROACH_CONSECUTIVE_K,
    gate: np.ndarray | None = None,
):
    """Nearest-candidate sustained-proximity commitment.

    ``correct_distance`` / ``distractor_distance`` are (T,) curves already reduced over
    each role's candidates with a min. ``gate`` optionally masks steps out (e.g. the
    MemWashAndReturn return leg only).

    Returns ``(label, t_correct, t_distractor)`` with -1 for "never committed".
    """
    correct = np.asarray(correct_distance, dtype=np.float64)
    wrong = np.asarray(distractor_distance, dtype=np.float64)
    n = len(correct)
    if n == 0:
        return "no_commitment", -1, -1
    if wrong.size == 0:
        wrong = np.full(n, np.inf)
    live = np.ones(n, dtype=bool) if gate is None else np.asarray(gate, dtype=bool)

    correct_ok = live & (correct < radius) & (correct <= wrong)
    wrong_ok = live & (wrong < radius) & (wrong < correct)
    t_correct = _sustained(correct_ok, consecutive)
    t_wrong = _sustained(wrong_ok, consecutive)

    if t_correct < 0 and t_wrong < 0:
        return "no_commitment", -1, -1
    if t_wrong >= 0 and (t_correct < 0 or t_wrong < t_correct):
        return "approached_wrong", t_correct, t_wrong
    return "approached_correct", t_correct, t_wrong


def _sustained(mask: np.ndarray, consecutive: int) -> int:
    if len(mask) < consecutive:
        return -1
    window = np.convolve(mask.astype(np.int32), np.ones(consecutive, dtype=np.int32), "valid")
    hits = np.flatnonzero(window == consecutive)
    return int(hits[0]) if len(hits) else -1


def knob_on(angles_over_time: np.ndarray) -> np.ndarray:
    """(T, K) boolean 'this knob is turned on', using the env's own test.

    ``MultiTaskBase._check_stove_on``: ``0.35 <= |angle| <= 2*pi - 0.35``.
    """
    angles = np.abs(np.asarray(angles_over_time, dtype=np.float64))
    return (angles >= 0.35) & (angles <= 2 * np.pi - 0.35)


def classify_knobs(
    locations, angles_over_time: np.ndarray, correct_knob: str, consecutive: int = APPROACH_CONSECUTIVE_K
):
    """Target commitment for knob tasks. Returns ``(label, t_correct, t_wrong)``.

    Same shape of answer as :func:`classify`, but read off the knob joints rather than
    geometry — see the MemHeatPot entry in ``TASK_SPECS`` for why geometry fails here.
    """
    if not locations or np.asarray(angles_over_time).size == 0:
        return "no_commitment", -1, -1
    on = knob_on(angles_over_time)
    if on.shape[1] != len(locations):
        return "no_commitment", -1, -1
    correct_index = locations.index(correct_knob) if correct_knob in locations else -1
    correct_mask = on[:, correct_index] if correct_index >= 0 else np.zeros(len(on), bool)
    wrong_mask = np.zeros(len(on), dtype=bool)
    for index in range(len(locations)):
        if index != correct_index:
            wrong_mask |= on[:, index]

    t_correct = _sustained(correct_mask, consecutive)
    t_wrong = _sustained(wrong_mask, consecutive)
    if t_correct < 0 and t_wrong < 0:
        return "no_commitment", -1, -1
    if t_wrong >= 0 and (t_correct < 0 or t_wrong < t_correct):
        return "approached_wrong", t_correct, t_wrong
    return "approached_correct", t_correct, t_wrong
