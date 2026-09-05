"""RoboCerebra/LIBERO 2-view geometry for the pi0.5 backbone: MolmoPoint pixels -> pi0.5 patch ids.

The pixel->patch math is IDENTICAL to pi_geometry.py — the pi0.5 tap is embodiment-agnostic, so
copying it here (rather than re-deriving it) is the point. Only the VIEW SET changes.

Why the math is what it is, for LIBERO frames specifically:
  * RoboCerebra LeRobot frames are native 256x256 squares. openpi's ``resize_with_pad(224, 224)``
    (robocasa_openpi/src/openpi/shared/image_tools.py) computes ratio = max(256/224, 256/224) = 8/7,
    resizes to 224x224 and pads ZERO rows/cols — i.e. a plain 256->224 resize, NO crop. (Contrast
    geometry.py, where GR00T's FractionalCenterCrop(0.95) is load-bearing.)
  * SigLIP So400m/14 tokenizes the 224 view into a 14x14 = 196 grid (16 px per patch).
  * The omega tap BIN-AVERAGES that 14x14 grid down to 8x8 = 64 tokens/view, with
    ``edges = linspace(0, 14, 9).astype(int) = [0,1,3,5,7,8,10,12,14]`` — see
    ``scripts/robocerebra/omega_tap.py:pool_grid`` (verbatim from Isaac-GR00T/wsm/pi05_tap_cache.py).
    The bins are NON-UNIFORM: output rows/cols cover 14-grid ranges [0],[1,2],[3,4],[5,6],[7],
    [8,9],[10,11],[12,13]. A label mapped through a uniform 8x8 split would be silently misaligned
    on 6 of 8 rows.

Why 2 views / 128 patches: RoboCerebra has exactly two cameras — ``image`` (third-person agentview)
and ``wrist_image`` (eye-in-hand). pi0.5 always lays out three image slots; RoboCerebra fills two
and zero-masks the third, so the tap emits base(64) then wrist(64) = 128 REAL tokens and every
downstream chance baseline is 1/128, not 1/192 (``scripts/robocerebra/omega_tap.py`` lines 4-12 and
98-102). ``VIEW_OFFSETS = {agentview: 0, eye_in_hand: 64}`` is therefore not a convention we are
free to choose — it is the tap's concatenation order.

Exported names mirror geometry.py / pi_geometry.py so extract_frames_robocerebra, qwen_subgoals,
molmo_points and build_salient_sets all swap geometry modules by name (``--geom pi_libero``). This
module additionally owns the stage-A Qwen view naming + system prompts, because the prompt has to
NAME the views: the view set and the text that describes it must not drift apart.
"""

from __future__ import annotations

import numpy as np

# --- pi0.5 model-view geometry (resize_with_pad 256-square -> 224; SigLIP 14x14; bin -> 8x8) ---
RENDER_WH = 256  # RoboCerebra LeRobot frames are native 256 squares
TARGET = 224  # pi0.5 / SigLIP input resolution
GRID_IN = 14  # SigLIP So400m/14 grid on 224 (224/16)
N_GRID = 8  # tap bin-averages 14x14 -> 8x8 = 64 tokens/view
_PATCH = TARGET // GRID_IN  # 16 px per SigLIP patch
_EDGES = np.linspace(0, GRID_IN, N_GRID + 1).astype(int)  # [0,1,3,5,7,8,10,12,14]
# No center-crop for pi (resize_with_pad on a square is a plain resize); kept for build/QC parity.
CROP_FRACTION = 1.0
_CROP = TARGET  # 224 (no crop)
_OFF = 0

# --- 2 RoboCerebra/LIBERO views (single source of truth; the tap's base->wrist order) ---
VIEWS = ("agentview", "eye_in_hand")
# LeRobot feature keys are BARE here (`image` / `wrist_image`), not `observation.images.*` as in
# RoboCasa365 — the RoboCerebra conversion wrote them flat (meta/info.json features).
VIEW_LEROBOT_KEY = {"agentview": "image", "eye_in_hand": "wrist_image"}
CLOSEUP_VIEWS = ()  # no dilation (same pilot finding as the RoboCasa geometries)
# Views a flat (non-per-view) Qwen object list falls back to: the third-person view only. Pointing
# at an object in the wrist close-up where it may be out of frame is how a label silently goes wrong.
AGENTVIEWS = ("agentview",)

PATCHES_PER_VIEW = N_GRID * N_GRID  # 64
VIEW_OFFSETS = {v: i * PATCHES_PER_VIEW for i, v in enumerate(VIEWS)}  # agentview 0, eye_in_hand 64
NUM_VIEWS = len(VIEWS)  # 2
NUM_PATCHES = PATCHES_PER_VIEW * NUM_VIEWS  # 128


def pixels_to_model_view(uv: np.ndarray) -> np.ndarray:
    """Pixels in the fed image (RENDER_WH space) -> pi0.5 224 model-view pixels (plain resize)."""
    return np.asarray(uv, dtype=np.float64) * (TARGET / RENDER_WH)


def patch_ids(uv_m: np.ndarray) -> np.ndarray:
    """224 model-view pixels [N,2] -> per-view 8x8 bin id [N] in [0,64); -1 if outside the 14x14 grid.
    Pixel -> 14x14 SigLIP cell -> non-uniform bin -> 8x8 cell (matches the tap's pool_grid)."""
    pc = np.floor(uv_m[:, 0] / _PATCH).astype(np.int64)  # SigLIP col 0..13
    pr = np.floor(uv_m[:, 1] / _PATCH).astype(np.int64)  # SigLIP row 0..13
    inside = (pc >= 0) & (pc < GRID_IN) & (pr >= 0) & (pr < GRID_IN)
    bc = np.clip(np.searchsorted(_EDGES, pc, side="right") - 1, 0, N_GRID - 1)  # bin col 0..7
    br = np.clip(np.searchsorted(_EDGES, pr, side="right") - 1, 0, N_GRID - 1)  # bin row 0..7
    ids = br * N_GRID + bc
    return np.where(inside, ids, -1)


def to_patches(points, dilate: bool = False) -> np.ndarray:
    """Point list [[x,y],...] (RENDER_WH pixels) -> unique per-view 8x8 patch ids."""
    if points is None or len(points) == 0:
        return np.empty(0, dtype=np.int64)
    ids = patch_ids(pixels_to_model_view(np.asarray(points, dtype=np.float64)))
    ids = ids[ids >= 0]
    if dilate:  # unused (CLOSEUP_VIEWS empty); kept for parity with the RoboCasa geometries
        rows, cols = ids // N_GRID, ids % N_GRID
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = rows + dr, cols + dc
                ok = (r >= 0) & (r < N_GRID) & (c >= 0) & (c < N_GRID)
                out.append(r[ok] * N_GRID + c[ok])
        ids = np.concatenate(out) if out else ids
    return np.unique(ids)


def to_global(local_ids: np.ndarray, view: str) -> np.ndarray:
    """Per-view 8x8 patch ids -> global ids over the concatenated 2-view grid (0..127)."""
    return np.asarray(local_ids, dtype=np.int64) + VIEW_OFFSETS[view]


# ---------------------------------------------------------------------------
# Stage-A (Qwen) view naming + system prompts.
#
# These live beside the view set on purpose: the system prompt NAMES the cameras and the JSON schema
# keys them, so a 2-view geometry with a 3-view prompt would produce objects under views that do not
# exist. qwen_subgoals.py picks these up via getattr and falls back to its RoboCasa text.
#
# The RoboCerebra pilot runs Qwen per GROUND-TRUTH SUBTASK SEGMENT (the dataset ships
# `subtask_index` + a per-frame subtask instruction), so the prompt asks for fine-grained subgoals
# WITHIN one already-short subtask, not a decomposition of a whole 900-frame episode.
# ---------------------------------------------------------------------------

QWEN_PROMPT_NAME = {"agentview": "agentview", "eye_in_hand": "eye_in_hand"}  # JSON keys
QWEN_CAPTION = {"agentview": "agentview", "eye_in_hand": "eye-in-hand"}  # image captions

_EMBODIMENT = (
    "You are labeling a robot-manipulation clip (a Franka Panda arm at a table in a LIBERO scene — "
    "a coffee table, kitchen table or study table with household objects, shelves, a fridge, a "
    "stove or a basket) for training a workspace model that must learn WHERE the action-relevant "
    "change happens. The clip is ONE SUBTASK of a longer multi-step episode and the Task line is "
    "that subtask's instruction. Each timestamp shows TWO views captioned with the frame index: an "
    "AGENTVIEW third-person camera and an EYE-IN-HAND close-up mounted on the gripper (objects "
    "there appear very large, truncated, or out of frame). "
)

QWEN_SYSTEM_SALIENT = _EMBODIMENT + (
    "Decompose this subtask into 3-5 FINE-GRAINED subgoals in temporal order (e.g. approach, "
    "align, grasp, lift/transport, place/release). For each subgoal pick the frame index where it "
    "is COMPLETED (the visible moment of completion). Then, SEPARATELY PER VIEW, list ONLY the "
    "objects DIRECTLY INVOLVED in the action at that exact moment: the specific object(s) being "
    "manipulated or about to be contacted; the robot gripper ONLY when it is touching or right "
    "next to the target; and any task EFFECT (e.g. 'wine pouring out of the bottle', 'the fridge "
    "door that is opening', 'the stove burner that just turned on'). "
    "Name AT MOST 1-3 objects per view. Do NOT list static background or scene furniture (the "
    "table surface, walls, floor, shelves/appliances that are NOT the current target, or any "
    "object not part of this subgoal's action). List an object under a view only if it is actually "
    "visible there; but EVERY subgoal MUST name at least the robot gripper/end-effector AND the "
    "primary manipulated/target object under SOME view — NEVER leave both views empty for a "
    "subgoal (the gripper is almost always visible, especially in the eye-in-hand view). Use "
    "short, specific noun phrases a pointing model can resolve (e.g. 'white bowl', 'wine bottle "
    "neck', 'fridge door handle', 'black gripper fingers'). "
    "Also produce an `expanded_prompt`: a 2-4 sentence step-by-step instruction that EXPANDS the "
    "terse subtask string into an ordered, lower-level description of how to accomplish it — "
    "grounded in what this specific clip shows. "
    'Respond with ONLY a JSON object: {"expanded_prompt": str, "subgoals": [{"name": str, '
    '"completion_frame": int, "salient_objects": {"agentview": [str, ...], '
    '"eye_in_hand": [str, ...]}}]}. completion_frame must be one of the captioned frame indices.'
)

QWEN_SYSTEM_CAUSAL_V1 = _EMBODIMENT.replace(
    "must learn WHERE the action-relevant change happens",
    "must mark the CAUSALLY RELEVANT region of each step — the places the action must physically "
    "touch or change in order to succeed, NOT merely the places that stand out",
) + (
    "Decompose this subtask into 3-5 FINE-GRAINED subgoals in temporal order. For each subgoal pick "
    "the frame index where it is COMPLETED. "
    "For each subgoal name EXACTLY TWO causal entities: "
    "(1) `manipulated` — the ONE object the gripper is acting on, or is just about to act on (e.g. "
    "'wine bottle', 'akita black bowl', 'fridge door handle'). If the subgoal acts on part of a "
    "fixture, name the PART that moves or is touched ('fridge door handle', not 'fridge'). "
    "(2) `goal` — the ONE place that object must end up, or the thing whose state must change as a "
    "result (e.g. 'the white bowl', 'the wooden two-layer shelf top', 'the basket interior'). If "
    "the outcome is a state change of the manipulated thing itself (opening a door, tilting a "
    "bottle), repeat that part as the goal. "
    "Then, SEPARATELY PER VIEW, report which of those two entities is actually VISIBLE in that "
    "view. Do NOT name the robot gripper, fingers, or arm. Do NOT name task EFFECTS. Do NOT name "
    "background or scene furniture. ONLY the two causal entities — this label is deliberately "
    "sparse. Omit an entity from a view where it is not visible, but EVERY subgoal MUST list its "
    "`manipulated` entity under at least ONE view. "
    'Respond with ONLY a JSON object: {"expanded_prompt": str, "subgoals": [{"name": str, '
    '"completion_frame": int, "causal_entities": {"manipulated": str, "goal": str}, '
    '"visible": {"agentview": [str, ...], "eye_in_hand": [str, ...]}}]}. '
    "Every string inside `visible` must be EXACTLY one of that subgoal's two causal-entity phrases, "
    "copied verbatim. completion_frame must be one of the captioned frame indices."
)

QWEN_SPECS = {"salient": QWEN_SYSTEM_SALIENT, "causal_v1": QWEN_SYSTEM_CAUSAL_V1}
