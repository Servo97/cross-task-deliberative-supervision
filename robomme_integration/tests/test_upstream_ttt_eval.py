from __future__ import annotations

import numpy as np
import pytest

from robomme_integration.eval.upstream_ttt_runner import (
    UpstreamTTTEvalRunner,
    UpstreamTTTSession,
    _noise_seed,
    capture_dense_observation,
)


def _observation(marker: int, *, video: bool = False) -> dict:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    result = {
        "images": {"agentview": image, "wrist": image + 1},
        "states": np.asarray([marker, 1, 2, 3, 4, 5, 6, 7, 99], dtype=np.float32),
        "task_description": "Pick X Times",
    }
    if video:
        result["video_history"] = [image - 2, image - 1]
        result["video_state_history"] = [
            np.arange(8, dtype=np.float32),
            np.arange(8, dtype=np.float32) + 1,
        ]
    return result


def test_dense_capture_prepends_paired_demo_then_keeps_every_execution_observation():
    session = UpstreamTTTSession("PickXtimes", 3)
    raw = capture_dense_observation(session, _observation(4, video=True), state_dim=8)

    assert len(session.pending_images) == 3
    assert len(session.pending_states) == 3
    assert session.pending_exec_start_idx == 2
    assert session.saw_video_history
    assert session.dense_observations == 1
    np.testing.assert_array_equal(raw["observation/state"], [4, 1, 2, 3, 4, 5, 6, 7])
    assert raw["prompt"] == "pick x times"

    capture_dense_observation(session, _observation(5), state_dim=8)
    assert len(session.pending_images) == 4
    assert len(session.pending_states) == 4
    assert session.dense_observations == 2


def test_dense_capture_rejects_duplicate_or_unpaired_video_history():
    session = UpstreamTTTSession("PickXtimes", 0)
    capture_dense_observation(session, _observation(4, video=True), state_dim=8)
    with pytest.raises(RuntimeError, match="more than once"):
        capture_dense_observation(session, _observation(4, video=True), state_dim=8)

    unpaired = _observation(4)
    unpaired["video_history"] = [unpaired["images"]["agentview"]]
    with pytest.raises(ValueError, match="paired video_state_history"):
        capture_dense_observation(UpstreamTTTSession("PickXtimes", 1), unpaired, state_dim=8)


def test_noise_identity_is_session_and_plan_deterministic():
    left = UpstreamTTTSession("PickXtimes", 7)
    right = UpstreamTTTSession("PickXtimes", 7)
    assert _noise_seed(11, left) == _noise_seed(11, right)
    left.plans += 1
    assert _noise_seed(11, left) != _noise_seed(11, right)
    assert _noise_seed(12, right) != _noise_seed(11, right)


def test_diagnostics_exposes_dropped_partial_chunk_without_mutation():
    session = UpstreamTTTSession("ButtonUnmaskSwap", 2)
    capture_dense_observation(session, _observation(6), state_dim=8)
    session.step_idx = 19
    session.exec_start_idx = 4
    session.plans = 2
    assert UpstreamTTTEvalRunner.diagnostics(session) == {
        "plans": 2,
        "dense_observations": 1,
        "memory_steps": 20,
        "exec_start_idx": 4,
        "saw_video_history": False,
        "pending_observations_dropped": 1,
    }
