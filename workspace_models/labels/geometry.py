"""RoboCasa365 3-view eval-transform geometry: MolmoPoint pixels -> GR00T patch ids.

LOAD-BEARING. These constants MUST match the image preprocessing the FROZEN GR00T N1.7
backbone applies, or every salient-patch label is silently misaligned (the supervision would
point at the wrong tokens). GR00T applies a ``FractionalCenterCrop(crop_fraction=0.95)`` then
resizes to the 256 model view; Qwen3-VL tokenizes that (patch16 + 2x2 merge) into an 8x8 = 64
token grid per view.
  * crop:   Isaac-GR00T/gr00t/model/gr00t_n1d7/image_augmentations.py  (FractionalCenterCrop)
  * 0.95:   Isaac-GR00T/scripts/validate_hf_config_alignment.py        (crop_fraction A28 = 0.95)
  * eval:   Isaac-GR00T/wsm/backbone_tap.py forces processor.eval()    (CenterCrop, deterministic)

Difference vs the DexJoCo reference (Isaac-GR00T/wsm/vlm_label/build_salient_sets.py): RoboCasa
LeRobot frames are NATIVE 256x256, so there is no 640->256 resize (``RENDER_WH=256``). The 0.95
center-crop (-> 243, offset 6) and the 256/243 rescale REMAIN — this is NOT an identity map.

3 RoboCasa views replace DexJoCo front+wrist. ``eye_in_hand`` is the close-up; unlike the DexJoCo
wrist its points are NOT dilated (pilot v1: dilation made it far too dense). Global patch ids
concatenate the per-view 64-grids -> 192 total.
"""

from __future__ import annotations

import numpy as np

# --- model-view geometry (mirror GR00T FractionalCenterCrop + resize-to-256) ---
RENDER_WH = 256  # RoboCasa LeRobot frames are native 256 (DexJoCo: 640)
TARGET = 256  # GR00T model-view resolution
N_GRID = 8  # 8x8 merged-patch grid -> 64 tokens/view
CROP_FRACTION = 0.95
_CROP = int(TARGET * CROP_FRACTION)  # 243
_OFF = (TARGET - _CROP) // 2  # 6

# --- 3 RoboCasa views (single source of truth; LeRobot keys are robot0_<view>) ---
VIEWS = ("agentview_left", "agentview_right", "eye_in_hand")
VIEW_LEROBOT_KEY = {v: f"observation.images.robot0_{v}" for v in VIEWS}
# Pilot v1 finding: the eye_in_hand close-up already spreads points across the frame, so the
# DexJoCo-style 3x3 dilation made it ~30/64 patches/keyframe (signal washed out). No dilation.
CLOSEUP_VIEWS = ()

PATCHES_PER_VIEW = N_GRID * N_GRID  # 64
VIEW_OFFSETS = {v: i * PATCHES_PER_VIEW for i, v in enumerate(VIEWS)}  # 0, 64, 128
NUM_VIEWS = len(VIEWS)  # 3
NUM_PATCHES = PATCHES_PER_VIEW * NUM_VIEWS  # 192


def pixels_to_model_view(uv: np.ndarray) -> np.ndarray:
    """Pixels in the fed image (RENDER_WH space) -> GR00T 256 model-view pixels."""
    return (uv * (TARGET / RENDER_WH) - _OFF) * (TARGET / _CROP)


def patch_ids(uv_m: np.ndarray) -> np.ndarray:
    """Model-view pixels [N,2] -> per-view patch id [N] in [0,64); -1 if outside the crop."""
    cell = TARGET // N_GRID
    col = np.floor(uv_m[:, 0] / cell).astype(np.int64)
    row = np.floor(uv_m[:, 1] / cell).astype(np.int64)
    ids = row * N_GRID + col
    inside = (col >= 0) & (col < N_GRID) & (row >= 0) & (row < N_GRID)
    return np.where(inside, ids, -1)


def to_patches(points, dilate: bool = False) -> np.ndarray:
    """Point list [[x,y],...] -> unique per-view patch ids (3x3 dilation for close-up views)."""
    if points is None or len(points) == 0:
        return np.empty(0, dtype=np.int64)
    ids = patch_ids(pixels_to_model_view(np.asarray(points, dtype=np.float64)))
    ids = ids[ids >= 0]
    if dilate:  # 3x3 neighborhood: close-up single points are low-precision
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
    """Per-view patch ids -> global ids over the concatenated 3-view grid (0..191)."""
    return np.asarray(local_ids, dtype=np.int64) + VIEW_OFFSETS[view]
