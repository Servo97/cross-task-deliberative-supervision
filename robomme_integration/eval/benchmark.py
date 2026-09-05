"""RoboMME benchmark adapter that preserves paired demonstration proprioception."""

from __future__ import annotations

from typing import Any

import numpy as np
from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark
from vla_eval.types import Observation, Task


class RoboMMEOfficialHistoryBenchmark(RoboMMEBenchmark):
    """Reference-compatible history payload for demo-capable policies.

    The upstream harness already sends front/wrist video frames, but its initial payload omits the
    paired proprioception required by the paper and official evaluator.  This subclass adds only
    ``video_state_history`` and otherwise delegates reset, stepping, recording, and success logic
    to the pinned harness implementation.
    """

    def reset(self, task: Task) -> Any:
        raw = super().reset(task)
        joints = list(raw.get("joint_state_list", [])[:-1])
        grippers = list(raw.get("gripper_state_list", [])[:-1])
        if len(joints) != len(grippers):
            raise RuntimeError(
                f"RoboMME demonstration joint/gripper history lengths differ: {len(joints)} vs {len(grippers)}"
            )
        self._video_state_history = [
            np.concatenate(
                [
                    np.asarray(joint, dtype=np.float32),
                    np.asarray(gripper, dtype=np.float32)[:1],
                ]
            ).astype(np.float32)
            for joint, gripper in zip(joints, grippers, strict=True)
        ]
        if len(self._video_state_history) != len(self._video_frames):
            raise RuntimeError(
                "RoboMME demonstration frame/state history lengths differ: "
                f"{len(self._video_frames)} vs {len(self._video_state_history)}"
            )
        # Some official tasks (including PickXtimes and ButtonUnmaskSwap) have no subgoal marked
        # for demonstration, so the legitimate conditioning prefix is empty.  Preserve an
        # explicit first-observation envelope for that case; absence of the envelope remains a
        # transport error rather than being conflated with a genuine zero-length history.
        self._official_history_pending = True
        return raw

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        observation = super().make_obs(raw_obs, task)
        if not getattr(self, "_official_history_pending", False):
            return observation
        images = observation.get("images")
        if not isinstance(images, dict) or not images:
            # The upstream error-observation path is not the episode's usable first observation.
            # Leave the one-shot history envelope pending so it cannot be consumed by an empty
            # transport placeholder.
            return observation
        history = observation.get("video_history")
        if history is None:
            if self._video_state_history:
                raise RuntimeError("RoboMME suppressed nonempty demonstration frames before history transport")
            history = []
            observation["video_history"] = history
            observation["episode_restart"] = True
        elif not observation.get("episode_restart"):
            raise RuntimeError("RoboMME demonstration history lacks its first-observation marker")
        if len(history) != len(self._video_state_history):
            raise RuntimeError(
                "RoboMME demonstration frame/state history lengths differ at transport: "
                f"{len(history)} vs {len(self._video_state_history)}"
            )
        observation["video_state_history"] = list(self._video_state_history)
        self._video_state_history = []
        self._official_history_pending = False
        return observation
