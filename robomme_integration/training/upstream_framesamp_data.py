"""Pure data assembly for the official RoboMME FrameSamp memory contract.

The official ``perceptual-framesamp-modul`` recipe exposes 512 front-view memory
tokens to the action expert: at most 32 frames are sampled uniformly from the
causal episode prefix, and every frame contributes a 4x4 grid of frozen SigLIP
features.  Short prefixes are padded on the right.

This module deliberately contains no JAX, Torch, LeRobot, or simulator imports.
It operates on the compact official feature tensors already used by the
RoboMME recurrent-memory adapter:

* image features: ``[steps, views, 64, image_dim]`` (8x8 spatial grid), and
* position features are generated directly at 4x4 from episode-global step
  indices with the upstream ``PosEmb3D`` formula.  Strict JAX/XLA parity should
  pass the official precomputed 4x4 table because NumPy trigonometric kernels
  are not bitwise identical to XLA's.

The 8x8 position cache is intentionally not an input.  Averaging adjacent
sinusoidal 8x8 embeddings is not equal to evaluating the embedding at each 4x4
cell center, which is what upstream does.

``step_idx`` is the dataset's episode-global step.  Consequently the inclusive
prefix ``[0, step_idx]`` naturally contains the complete demonstration followed
by causal execution history; no special demo/execution splice is needed.
"""

from __future__ import annotations

import dataclasses

import numpy as np

SOURCE_GRID = 8
TARGET_GRID = 4
SOURCE_TOKENS_PER_FRAME = SOURCE_GRID * SOURCE_GRID
TOKENS_PER_FRAME = TARGET_GRID * TARGET_GRID
MAX_FRAMES = 32
TOKEN_BUDGET = MAX_FRAMES * TOKENS_PER_FRAME
POSITION_DIM = 768
TEMPORAL_BASE = 10_000.0
SPATIAL_BASE = 1_000.0


@dataclasses.dataclass(frozen=True)
class FrameSampHistory:
    """One fixed-shape FrameSamp memory input.

    ``frame_indices`` is right-padded with ``-1`` and is retained for causal
    audits. ``frame_mask`` and ``token_mask`` mark the same valid prefix at frame
    and flattened-token granularity, respectively.
    """

    image: np.ndarray
    position: np.ndarray
    token_mask: np.ndarray
    frame_indices: np.ndarray
    frame_mask: np.ndarray

    def validate(self) -> None:
        if self.image.ndim != 2 or self.image.shape[0] != TOKEN_BUDGET:
            raise ValueError(f"FrameSamp image must be [{TOKEN_BUDGET}, D], got {self.image.shape}")
        if self.position.ndim != 2 or self.position.shape[0] != TOKEN_BUDGET:
            raise ValueError(f"FrameSamp position must be [{TOKEN_BUDGET}, D], got {self.position.shape}")
        if self.token_mask.shape != (TOKEN_BUDGET,) or self.token_mask.dtype != np.bool_:
            raise ValueError("FrameSamp token_mask has the wrong shape or dtype")
        if self.frame_indices.shape != (MAX_FRAMES,) or not np.issubdtype(self.frame_indices.dtype, np.signedinteger):
            raise ValueError("FrameSamp frame_indices has the wrong shape or dtype")
        if self.frame_mask.shape != (MAX_FRAMES,) or self.frame_mask.dtype != np.bool_:
            raise ValueError("FrameSamp frame_mask has the wrong shape or dtype")
        valid_frames = int(self.frame_mask.sum())
        if not 1 <= valid_frames <= MAX_FRAMES:
            raise ValueError("FrameSamp must contain between one and 32 valid frames")
        if not self.frame_mask[:valid_frames].all() or self.frame_mask[valid_frames:].any():
            raise ValueError("FrameSamp frame padding must be a right-padded prefix")
        if not np.array_equal(
            self.token_mask,
            np.repeat(self.frame_mask, TOKENS_PER_FRAME),
        ):
            raise ValueError("FrameSamp token and frame masks disagree")
        if np.any(self.frame_indices[:valid_frames] < 0) or np.any(self.frame_indices[valid_frames:] != -1):
            raise ValueError("FrameSamp frame-index padding is invalid")
        if np.any(np.diff(self.frame_indices[:valid_frames]) < 0):
            raise ValueError("FrameSamp frame indices must be chronological")
        if not np.isfinite(self.image).all() or not np.isfinite(self.position).all():
            raise ValueError("FrameSamp memory contains non-finite values")
        if np.any(self.image[~self.token_mask] != 0) or np.any(self.position[~self.token_mask] != 0):
            raise ValueError("FrameSamp padded tokens must be exact zeros")


def even_sampling_indices(step_idx: int) -> tuple[int, ...]:
    """Match the official inclusive, uniform FrameSamp index rule exactly.

    This is the pure NumPy equivalent of upstream ``even_sampling_indices``:

    * prefixes of at most 32 frames keep every frame;
    * longer prefixes use ``np.linspace(0, step_idx, 32, dtype=np.int32)``.
    """

    if isinstance(step_idx, bool) or not isinstance(step_idx, (int, np.integer)):
        raise TypeError(f"step_idx must be an integer, got {type(step_idx).__name__}")
    step_idx = int(step_idx)
    if step_idx < 0:
        raise ValueError("step_idx must be nonnegative")
    if step_idx < MAX_FRAMES:
        return tuple(range(step_idx + 1))
    return tuple(int(value) for value in np.linspace(0, step_idx, MAX_FRAMES, dtype=np.int32))


def _as_float32_features(value: np.ndarray, *, label: str) -> np.ndarray:
    """Decode compact bfloat16 bit payloads or cast ordinary numeric features."""

    value = np.asarray(value)
    if value.dtype == np.uint16:
        # The compact official cache stores bfloat16 payloads as uint16 because
        # NumPy's .npy header does not preserve the ml_dtypes dtype metadata.
        try:
            import ml_dtypes
        except ModuleNotFoundError as error:  # pragma: no cover - JAX environments provide it.
            raise RuntimeError(f"{label} contains bfloat16 bit payloads, but ml_dtypes is unavailable") from error
        # ``np.issubdtype(ml_dtypes.bfloat16, np.number)`` is false, so cast
        # before the generic NumPy numeric-dtype guard below.
        value = value.view(ml_dtypes.bfloat16).astype(np.float32)
    elif value.dtype.name == "bfloat16":
        # Also accept the in-memory upstream dtype before compact serialization.
        value = value.astype(np.float32)
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{label} must be numeric, got {value.dtype}")
    result = value.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _as_pooling_features(value: np.ndarray, *, label: str) -> tuple[np.ndarray, bool]:
    """Decode an image cache while preserving upstream bfloat16 pooling math.

    Official SigLIP features enter ``nnx.avg_pool`` as bfloat16.  Casting them
    to float32 before the reduction changes rounding, so compact uint16 bit
    payloads and in-memory bfloat16 arrays stay bfloat16 until after pooling.
    Ordinary numeric inputs use the exact float32 path.
    """

    value = np.asarray(value)
    is_bfloat16 = value.dtype == np.uint16 or value.dtype.name == "bfloat16"
    if value.dtype == np.uint16:
        try:
            import ml_dtypes
        except ModuleNotFoundError as error:  # pragma: no cover - JAX environments provide it.
            raise RuntimeError(f"{label} contains bfloat16 bit payloads, but ml_dtypes is unavailable") from error
        value = value.view(ml_dtypes.bfloat16)
    elif not is_bfloat16:
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"{label} must be numeric, got {value.dtype}")
        value = value.astype(np.float32, copy=False)
    if not np.isfinite(value.astype(np.float32, copy=False)).all():
        raise ValueError(f"{label} contains non-finite values")
    return value, is_bfloat16


def position_embedding_3d(
    step_indices: np.ndarray | tuple[int, ...] | list[int],
    *,
    dim: int = POSITION_DIM,
    temporal_base: float = TEMPORAL_BASE,
    spatial_base: float = SPATIAL_BASE,
) -> np.ndarray:
    """Reproduce upstream ``PosEmb3D`` formula/order for 4x4 using NumPy.

    One third of the output encodes episode-global time and two thirds encode
    the 4x4 spatial cell centers.  Upstream centers are ``4*i + 2`` in the
    original 16x16 image-patch coordinate system.
    """

    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be an integer")
    dim = int(dim)
    if dim % 6 or dim < 12:
        raise ValueError("PosEmb3D dim must be divisible by 6 and at least 12")
    steps = np.asarray(step_indices)
    if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer):
        raise ValueError("step_indices must be a rank-1 integer sequence")
    if not steps.size or np.any(steps < 0):
        raise ValueError("step_indices must be nonempty and nonnegative")
    temporal_base = float(temporal_base)
    spatial_base = float(spatial_base)
    if not np.isfinite(temporal_base) or temporal_base <= 0:
        raise ValueError("temporal_base must be finite and positive")
    if not np.isfinite(spatial_base) or spatial_base <= 0:
        raise ValueError("spatial_base must be finite and positive")

    width = dim // 6
    omega = np.arange(width, dtype=np.float32) / np.float32(width - 1)
    temporal_omega = np.float32(1.0) / np.power(np.float32(temporal_base), omega)
    spatial_omega = np.float32(1.0) / np.power(np.float32(spatial_base), omega)

    temporal_angles = np.einsum(
        "m,d->md",
        steps.astype(np.float32, copy=False),
        temporal_omega,
    )
    temporal = np.concatenate([np.sin(temporal_angles), np.cos(temporal_angles)], axis=-1)
    temporal = np.repeat(temporal[:, None, :], TOKENS_PER_FRAME, axis=1)

    y, x = np.mgrid[:TARGET_GRID, :TARGET_GRID]
    y = (TARGET_GRID * y + TARGET_GRID // 2).reshape(-1).astype(np.float32)
    x = (TARGET_GRID * x + TARGET_GRID // 2).reshape(-1).astype(np.float32)
    y_angles = np.einsum("m,d->md", y, spatial_omega)
    x_angles = np.einsum("m,d->md", x, spatial_omega)
    spatial = np.concatenate(
        [np.sin(y_angles), np.cos(y_angles), np.sin(x_angles), np.cos(x_angles)],
        axis=-1,
    )
    spatial = np.broadcast_to(spatial[None, :, :], (steps.size, TOKENS_PER_FRAME, 4 * width))
    result = np.concatenate([temporal, spatial], axis=-1).astype(np.float32, copy=False)
    if result.shape != (steps.size, TOKENS_PER_FRAME, dim):
        raise RuntimeError(f"unexpected PosEmb3D output shape {result.shape}")
    return result


def pool_front_8x8_to_4x4(tokens: np.ndarray, *, label: str = "tokens") -> np.ndarray:
    """Mean-pool the front view from ``[..., views, 64, D]`` to ``[..., 16, D]``.

    The view axis is intentionally consumed here: FrameSamp history is front-view
    only even though the live policy observation also contains a wrist camera.
    """

    tokens, is_bfloat16 = _as_pooling_features(tokens, label=label)
    if tokens.ndim < 3:
        raise ValueError(f"{label} must have shape [..., views, 64, D], got {tokens.shape}")
    if tokens.shape[-3] < 1 or tokens.shape[-2] != SOURCE_TOKENS_PER_FRAME:
        raise ValueError(f"{label} must have a nonempty view axis and 64 patches, got {tokens.shape}")
    front = tokens[..., 0, :, :]
    lead = front.shape[:-2]
    feature_dim = front.shape[-1]
    grid = front.reshape(*lead, TARGET_GRID, 2, TARGET_GRID, 2, feature_dim)
    # The official pool uses a non-overlapping 2x2 average with stride 2.
    if is_bfloat16:
        pooled = grid.mean(axis=(-4, -2))
    else:
        pooled = grid.mean(axis=(-4, -2), dtype=np.float32)
    return pooled.reshape(*lead, TOKENS_PER_FRAME, feature_dim).astype(np.float32, copy=False)


def assemble_framesamp_history(
    image_features: np.ndarray,
    step_idx: int,
    *,
    exact_position_features: np.ndarray | None = None,
) -> FrameSampHistory:
    """Build one exact 512-token causal FrameSamp input.

    ``image_features`` may contain more steps than ``step_idx``.  Only the
    inclusive episode prefix ending at ``step_idx`` is ever indexed, which is the
    causality boundary used by both training and online evaluation.  By default,
    formula-equivalent 4x4 positions are derived from selected global step
    indices.  For strict released-teacher parity, supply the official
    precomputed 4x4 table with shape ``[steps, 16, D]``; 8x8 positions are
    rejected rather than pooled.
    """

    image_features = np.asarray(image_features)
    if image_features.ndim != 4:
        raise ValueError(f"FrameSamp image source must be [steps, views, 64, D], got {image_features.shape}")
    if not len(image_features):
        raise ValueError("FrameSamp source episode is empty")
    if isinstance(step_idx, bool) or not isinstance(step_idx, (int, np.integer)):
        raise TypeError(f"step_idx must be an integer, got {type(step_idx).__name__}")
    step_idx = int(step_idx)
    if not 0 <= step_idx < len(image_features):
        raise IndexError(f"step_idx {step_idx} is outside an episode with {len(image_features)} steps")

    selected = even_sampling_indices(step_idx)
    if not selected or selected[-1] > step_idx:
        raise RuntimeError("FrameSamp selection leaked beyond the causal prefix")
    selected_array = np.asarray(selected, dtype=np.int64)
    image = pool_front_8x8_to_4x4(image_features[selected_array], label="image_features")
    if exact_position_features is None:
        position = position_embedding_3d(selected_array)
    else:
        exact_position_features = _as_float32_features(
            exact_position_features,
            label="exact_position_features",
        )
        if exact_position_features.ndim != 3 or exact_position_features.shape[:2] != (
            len(image_features),
            TOKENS_PER_FRAME,
        ):
            raise ValueError(
                "exact_position_features must be [steps, 16, D] and match the image episode, "
                f"got {exact_position_features.shape}"
            )
        position = exact_position_features[selected_array]

    valid_frames = len(selected)
    padded_image = np.zeros((MAX_FRAMES, TOKENS_PER_FRAME, image.shape[-1]), dtype=np.float32)
    padded_position = np.zeros(
        (MAX_FRAMES, TOKENS_PER_FRAME, position.shape[-1]),
        dtype=np.float32,
    )
    padded_image[:valid_frames] = image
    padded_position[:valid_frames] = position
    frame_mask = np.zeros((MAX_FRAMES,), dtype=np.bool_)
    frame_mask[:valid_frames] = True
    frame_indices = np.full((MAX_FRAMES,), -1, dtype=np.int32)
    frame_indices[:valid_frames] = selected_array.astype(np.int32)
    result = FrameSampHistory(
        image=padded_image.reshape(TOKEN_BUDGET, image.shape[-1]),
        position=padded_position.reshape(TOKEN_BUDGET, position.shape[-1]),
        token_mask=np.repeat(frame_mask, TOKENS_PER_FRAME),
        frame_indices=frame_indices,
        frame_mask=frame_mask,
    )
    result.validate()
    return result
