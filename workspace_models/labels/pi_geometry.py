"""RoboCasa365 3-view geometry for the pi0.5 backbone: MolmoPoint pixels -> pi0.5 patch ids.

pi0.5 differs from GR00T: it resizes each view to 224x224 with resize_with_pad (RoboCasa frames are
native 256x256 squares, so this is a plain 256->224 resize, NO crop), SigLIP So400m/14 tokenizes to a
14x14=196 grid, and the WSM tap BIN-AVERAGES that 14x14 grid down to 8x8=64 tokens/view (matching the
GR00T 192-token cache schema). So a salient label must map a pixel through the SAME non-uniform 14->8
binning the tap uses, NOT a uniform 8x8 split.

Binning (mirrors Isaac-GR00T/wsm/pi05_tap_cache.py bin8x8): edges = linspace(0,14,9).astype(int) =
[0,1,3,5,7,8,10,12,14] -> the 8 output rows/cols cover 14-grid index ranges [0],[1,2],[3,4],[5,6],
[7],[8,9],[10,11],[12,13]. Exported names mirror geometry.py so build_salient_sets can swap modules.
"""

from __future__ import annotations

import numpy as np

# --- pi0.5 model-view geometry (resize_with_pad 256-square -> 224; SigLIP 14x14; bin -> 8x8) ---
RENDER_WH = 256  # RoboCasa LeRobot frames are native 256 squares
TARGET = 224  # pi0.5 / SigLIP input resolution
GRID_IN = 14  # SigLIP So400m/14 grid on 224 (224/16)
N_GRID = 8  # tap bin-averages 14x14 -> 8x8 = 64 tokens/view
_PATCH = TARGET // GRID_IN  # 16 px per SigLIP patch
_EDGES = np.linspace(0, GRID_IN, N_GRID + 1).astype(int)  # [0,1,3,5,7,8,10,12,14]
# No center-crop for pi (resize_with_pad on a square is a plain resize); kept for build/QC parity.
CROP_FRACTION = 1.0
_CROP = TARGET  # 224 (no crop)
_OFF = 0

# --- 3 RoboCasa views (single source of truth; identical to geometry.py) ---
VIEWS = ("agentview_left", "agentview_right", "eye_in_hand")
VIEW_LEROBOT_KEY = {v: f"observation.images.robot0_{v}" for v in VIEWS}
CLOSEUP_VIEWS = ()  # no dilation (pilot finding, same as GR00T geometry)

PATCHES_PER_VIEW = N_GRID * N_GRID  # 64
VIEW_OFFSETS = {v: i * PATCHES_PER_VIEW for i, v in enumerate(VIEWS)}  # 0, 64, 128
NUM_VIEWS = len(VIEWS)  # 3
NUM_PATCHES = PATCHES_PER_VIEW * NUM_VIEWS  # 192


def pixels_to_model_view(uv: np.ndarray) -> np.ndarray:
    """Pixels in the fed image (RENDER_WH space) -> pi0.5 224 model-view pixels (plain resize)."""
    return np.asarray(uv, dtype=np.float64) * (TARGET / RENDER_WH)


def patch_ids(uv_m: np.ndarray) -> np.ndarray:
    """224 model-view pixels [N,2] -> per-view 8x8 bin id [N] in [0,64); -1 if outside the 14x14 grid.
    Pixel -> 14x14 SigLIP cell -> non-uniform bin -> 8x8 cell (matches the tap's bin8x8)."""
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
    if dilate:  # unused for RoboCasa (CLOSEUP_VIEWS empty); kept for parity
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
    """Per-view 8x8 patch ids -> global ids over the concatenated 3-view grid (0..191)."""
    return np.asarray(local_ids, dtype=np.int64) + VIEW_OFFSETS[view]
