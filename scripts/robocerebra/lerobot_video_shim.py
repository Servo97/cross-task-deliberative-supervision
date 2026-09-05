"""PyAV decoder shim for LeRobot video features in the openpi venv.

Both of LeRobot 0.1.0's video backends are broken in that venv:

* ``torchcodec`` -- prebuilt ``libtorchcodec_decoder*.so`` do not load against torch 2.11+cu128.
* ``pyav`` -- LeRobot routes it through ``torchvision.io.VideoReader``, which torchvision 0.28
  removed.

The RoboCasa / ReMemBench path never notices, because it decodes through GR00T's own loader.
The LIBERO path -- which is what RoboCerebra post-training rides on, since we initialise from
the released ``pi05_libero`` checkpoint -- decodes through LeRobot and therefore does notice.

``install()`` replaces ``lerobot.common.datasets.video_utils.decode_video_frames`` with a direct
PyAV implementation. It must be called before the dataset is constructed, and it is inherited by
fork-started DataLoader workers.
"""

from __future__ import annotations

from pathlib import Path


def _decode_with_pyav(video_path, timestamps, tolerance_s, backend=None):
    import av
    import numpy as np
    import torch

    wanted = np.asarray(timestamps, dtype=np.float64)
    lo, hi = float(wanted.min()), float(wanted.max())

    with av.open(str(Path(video_path))) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek to the keyframe at or before the first wanted timestamp, then decode forward.
        if lo > 0:
            container.seek(int(max(lo - 1.0, 0) / float(stream.time_base)), stream=stream)
        times, frames = [], []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp > hi + max(tolerance_s, 1.0):
                break
            times.append(timestamp)
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise ValueError(f"decoded no frames from {video_path} for timestamps {timestamps}")

    available = np.asarray(times)
    picked = np.abs(available[None, :] - wanted[:, None]).argmin(axis=1)
    gaps = np.abs(available[picked] - wanted)
    if (gaps > tolerance_s).any():
        raise AssertionError(f"{video_path}: nearest decoded frame off by {gaps.max():.6f}s > tolerance {tolerance_s}")

    stack = np.stack([frames[i] for i in picked])  # [N, H, W, 3] uint8
    return torch.from_numpy(stack).permute(0, 3, 1, 2).to(torch.float32) / 255.0


def install() -> None:
    from lerobot.common.datasets import lerobot_dataset, video_utils

    video_utils.get_safe_default_codec = lambda: "pyav"
    video_utils.decode_video_frames = _decode_with_pyav
    lerobot_dataset.decode_video_frames = _decode_with_pyav
