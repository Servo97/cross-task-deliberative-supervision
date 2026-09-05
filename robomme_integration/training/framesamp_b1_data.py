"""Pure data contract for the FrameSamp B1 representation-policy control.

B1 deliberately differs from released FrameSamp.  Released FrameSamp uniformly
resamples at most 32 frames from the complete growing causal prefix.  B1 keeps
the same 32-frame / 512-token compute budget but assigns stable physical
partitions:

* slots 0..15: a frozen, uniformly sampled initial demonstration;
* slots 16..31: the 16 most recent raw execution frames in chronological order.

Each partition is independently right padded.  Episode-global time features
remain attached to the selected source frames, while the two fixed partitions
determine the downstream MemoryAttention physical RoPE slots.  This module is
framework-free so the selection identity can be tested and sealed separately
from the released policy runtime.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKENS_PER_FRAME,
    pool_front_8x8_to_4x4,
    position_embedding_3d,
)

B1_SCHEMA_VERSION = 1
B1_PARTITION_KIND = "fixed_demo16_raw_recent16_right_padded_v1"
B1_DEMO_FRAMES = 16
B1_LIVE_FRAMES = 16

if B1_DEMO_FRAMES + B1_LIVE_FRAMES != MAX_FRAMES:  # pragma: no cover
    raise RuntimeError("B1 partitions must preserve the released 32-frame budget")


def _integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _uniform_inclusive(last: int, capacity: int) -> tuple[int, ...]:
    if last < capacity:
        return tuple(range(last + 1))
    return tuple(int(value) for value in np.linspace(0, last, capacity, dtype=np.int32))


@dataclasses.dataclass(frozen=True)
class FrameSampB1Selection:
    """Exact fixed-slot selection at one causal replan cut."""

    step_idx: int
    exec_start_idx: int
    frame_indices: np.ndarray
    frame_mask: np.ndarray
    demo_mask: np.ndarray
    live_mask: np.ndarray

    @property
    def demo_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.frame_indices[self.demo_mask])

    @property
    def live_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.frame_indices[self.live_mask])

    def validate(self) -> None:
        if self.frame_indices.shape != (MAX_FRAMES,) or not np.issubdtype(self.frame_indices.dtype, np.signedinteger):
            raise ValueError("B1 frame_indices must be signed integer [32]")
        for label, mask in (
            ("frame", self.frame_mask),
            ("demo", self.demo_mask),
            ("live", self.live_mask),
        ):
            if mask.shape != (MAX_FRAMES,) or mask.dtype != np.bool_:
                raise ValueError(f"B1 {label}_mask must be Boolean [32]")
        if np.any(self.demo_mask[B1_DEMO_FRAMES:]) or np.any(self.live_mask[:B1_DEMO_FRAMES]):
            raise ValueError("B1 demo/live masks crossed their physical partitions")
        if np.any(self.demo_mask & self.live_mask):
            raise ValueError("B1 demo/live partitions overlap")
        if not np.array_equal(self.frame_mask, self.demo_mask | self.live_mask):
            raise ValueError("B1 frame mask disagrees with its partitions")
        for offset, capacity, mask in (
            (0, B1_DEMO_FRAMES, self.demo_mask),
            (B1_DEMO_FRAMES, B1_LIVE_FRAMES, self.live_mask),
        ):
            valid = int(mask[offset : offset + capacity].sum())
            expected = np.zeros((capacity,), dtype=np.bool_)
            expected[:valid] = True
            if not np.array_equal(mask[offset : offset + capacity], expected):
                raise ValueError("B1 partitions must be independently right padded")
        if np.any(self.frame_indices[self.frame_mask] < 0) or np.any(self.frame_indices[~self.frame_mask] != -1):
            raise ValueError("B1 selected/padded frame indices are invalid")
        demo = self.frame_indices[self.demo_mask]
        live = self.frame_indices[self.live_mask]
        if not len(demo):
            raise ValueError("B1 requires a nonempty initial demonstration")
        if np.any(np.diff(demo) <= 0) or np.any(demo > self.exec_start_idx):
            raise ValueError("B1 demo indices are not strictly causal and chronological")
        if len(live) and (np.any(np.diff(live) != 1) or live[0] <= self.exec_start_idx or live[-1] != self.step_idx):
            raise ValueError("B1 live indices must be the chronological recent suffix")


@dataclasses.dataclass(frozen=True)
class FrameSampB1History:
    """Fixed-shape B1 tensors plus their explicit partition selection."""

    image: np.ndarray
    position: np.ndarray
    token_mask: np.ndarray
    selection: FrameSampB1Selection

    def validate(self) -> None:
        self.selection.validate()
        token_budget = MAX_FRAMES * TOKENS_PER_FRAME
        if self.image.ndim != 2 or self.image.shape[0] != token_budget:
            raise ValueError("B1 image must have 512 rows")
        if self.position.ndim != 2 or self.position.shape[0] != token_budget:
            raise ValueError("B1 position must have 512 rows")
        if self.token_mask.shape != (token_budget,) or self.token_mask.dtype != np.bool_:
            raise ValueError("B1 token_mask must be Boolean [512]")
        if not np.array_equal(
            self.token_mask,
            np.repeat(self.selection.frame_mask, TOKENS_PER_FRAME),
        ):
            raise ValueError("B1 token mask disagrees with its frame partitions")
        if not np.isfinite(self.image).all() or not np.isfinite(self.position).all():
            raise ValueError("B1 tensors contain non-finite values")
        if np.any(self.image[~self.token_mask] != 0) or np.any(self.position[~self.token_mask] != 0):
            raise ValueError("B1 padded tokens must be exact zeros")


def select_framesamp_b1(step_idx: int, exec_start_idx: int) -> FrameSampB1Selection:
    """Select the frozen-demo and recent-live frames for one inclusive cut.

    ``exec_start_idx`` is the final frame of the initial history, matching the
    released evaluator's initial ``add_buffer`` call.  Execution frames begin
    at ``exec_start_idx + 1``.
    """

    step_idx = _integer(step_idx, label="step_idx")
    exec_start_idx = _integer(exec_start_idx, label="exec_start_idx")
    if exec_start_idx < 0 or step_idx < exec_start_idx:
        raise ValueError("B1 requires 0 <= exec_start_idx <= step_idx")

    demo = _uniform_inclusive(exec_start_idx, B1_DEMO_FRAMES)
    live_start = exec_start_idx + 1
    if step_idx < live_start:
        live: tuple[int, ...] = ()
    else:
        live = tuple(range(max(live_start, step_idx - B1_LIVE_FRAMES + 1), step_idx + 1))

    indices = np.full((MAX_FRAMES,), -1, dtype=np.int32)
    demo_mask = np.zeros((MAX_FRAMES,), dtype=np.bool_)
    live_mask = np.zeros((MAX_FRAMES,), dtype=np.bool_)
    indices[: len(demo)] = demo
    demo_mask[: len(demo)] = True
    live_offset = B1_DEMO_FRAMES
    indices[live_offset : live_offset + len(live)] = live
    live_mask[live_offset : live_offset + len(live)] = True
    result = FrameSampB1Selection(
        step_idx=step_idx,
        exec_start_idx=exec_start_idx,
        frame_indices=indices,
        frame_mask=demo_mask | live_mask,
        demo_mask=demo_mask,
        live_mask=live_mask,
    )
    result.validate()
    return result


def assemble_framesamp_b1_history(
    image_features: np.ndarray,
    step_idx: int,
    exec_start_idx: int,
    *,
    exact_position_features: np.ndarray | None = None,
) -> FrameSampB1History:
    """Build the 512-token B1 input without moving frames across partitions."""

    image_features = np.asarray(image_features)
    if image_features.ndim != 4 or not len(image_features):
        raise ValueError("B1 image source must be nonempty [steps, views, 64, D]")
    selection = select_framesamp_b1(step_idx, exec_start_idx)
    if step_idx >= len(image_features):
        raise IndexError("B1 step_idx is outside the source episode")
    selected = selection.frame_indices[selection.frame_mask].astype(np.int64)
    image = pool_front_8x8_to_4x4(image_features[selected], label="image_features")

    if exact_position_features is None:
        position = position_embedding_3d(selected)
    else:
        position_source = np.asarray(exact_position_features)
        if (
            position_source.ndim != 3
            or position_source.shape[:2] != (len(image_features), TOKENS_PER_FRAME)
            or not np.issubdtype(position_source.dtype, np.number)
            or not np.isfinite(position_source).all()
        ):
            raise ValueError("exact_position_features must be finite numeric [steps,16,D]")
        position = position_source[selected].astype(np.float32, copy=False)

    image_slots = np.zeros((MAX_FRAMES, TOKENS_PER_FRAME, image.shape[-1]), dtype=np.float32)
    position_slots = np.zeros((MAX_FRAMES, TOKENS_PER_FRAME, position.shape[-1]), dtype=np.float32)
    image_slots[selection.frame_mask] = image
    position_slots[selection.frame_mask] = position
    history = FrameSampB1History(
        image=image_slots.reshape(MAX_FRAMES * TOKENS_PER_FRAME, image.shape[-1]),
        position=position_slots.reshape(MAX_FRAMES * TOKENS_PER_FRAME, position.shape[-1]),
        token_mask=np.repeat(selection.frame_mask, TOKENS_PER_FRAME),
        selection=selection,
    )
    history.validate()
    return history
