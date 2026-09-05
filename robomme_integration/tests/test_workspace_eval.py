from __future__ import annotations

import json
import os
import pathlib
import sys
import types

import numpy as np
import pytest

from wsm_settings import TRI_ROOT

HARNESS = pathlib.Path(
    os.environ.get(
        "ROBOMME_VLA_EVAL_SRC",
        str(TRI_ROOT / "vla-evaluation-harness" / "src"),
    )
)
if HARNESS.exists():
    sys.path.insert(0, str(HARNESS))

from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark  # noqa: E402

from gcp_tpu.run_execution_eval import main as execution_eval_main  # noqa: E402
from robomme_integration.eval.benchmark import RoboMMEOfficialHistoryBenchmark  # noqa: E402
from robomme_integration.eval.workspace_runner import (  # noqa: E402
    FEATURE_DIM,
    GPU_ENABLED_WORKSPACE_TRAINER_SHA256,
    LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256,
    OMEGA_DIM,
    OnlineWorkspaceRunner,
    TaskRoutedOnlineWorkspaceRunner,
    WorkspaceSession,
    capture_workspace_observation,
    causal_history_indices,
    requested_omega_steps,
    serving_model_arm,
    validate_workspace_trainer_implementation,
    workspace_window_for_arm,
)


def _observation(marker: int, *, video_markers: list[int] | None = None) -> dict:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    observation = {
        "images": {"agentview": image, "wrist": image},
        "states": np.asarray([marker] + [0] * 7, dtype=np.float32),
        "task_description": "Pick X times",
    }
    if video_markers is not None:
        observation["video_history"] = [np.full((8, 8, 3), value, dtype=np.uint8) for value in video_markers]
        observation["video_state_history"] = [
            np.asarray([value] + [0] * 7, dtype=np.float32) for value in video_markers
        ]
        observation["episode_restart"] = True
    return observation


class _FakeFrameEncoder:
    def __init__(self):
        self.calls: list[list[int]] = []

    def __call__(self, images: np.ndarray) -> np.ndarray:
        markers = np.asarray(images[:, 0, 0, 0], dtype=np.float32)
        self.calls.append(markers.astype(int).tolist())
        return np.repeat(markers[:, None], FEATURE_DIM, axis=1).astype(np.float16)


class _FakeOmegaEncoder:
    def __init__(self):
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, history: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append((history.copy(), mask.copy()))
        last = np.asarray([row[np.flatnonzero(valid)[-1], 0] for row, valid in zip(history, mask, strict=True)])
        return np.repeat(last[:, None], OMEGA_DIM, axis=1)


def _runner():
    frames = _FakeFrameEncoder()
    omega = _FakeOmegaEncoder()
    runner = OnlineWorkspaceRunner(
        frames,
        omega,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
        history_stride=10,
        max_history=128,
    )
    return runner, frames, omega


def _workspace_trainer_source() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "training/workspace_deliberative.py"


def _history_benchmark(state_history: list[np.ndarray]):
    benchmark = object.__new__(RoboMMEOfficialHistoryBenchmark)
    benchmark._official_history_pending = True
    benchmark._video_state_history = state_history
    return benchmark


def test_official_zero_demo_reset_emits_one_explicit_paired_empty_envelope(monkeypatch):
    benchmark = object.__new__(RoboMMEOfficialHistoryBenchmark)
    benchmark._video_frames = []
    raw_reset = {
        "joint_state_list": [np.zeros(7, dtype=np.float32)],
        "gripper_state_list": [np.zeros(1, dtype=np.float32)],
    }
    monkeypatch.setattr(RoboMMEBenchmark, "reset", lambda *_args: raw_reset)
    assert benchmark.reset({"name": "PickXtimes", "env_id": "PickXtimes"}) is raw_reset
    monkeypatch.setattr(
        RoboMMEBenchmark,
        "make_obs",
        lambda *_args: {"images": {}, "task_description": "Pick X times"},
    )
    invalid = benchmark.make_obs({}, {"name": "PickXtimes", "env_id": "PickXtimes"})
    assert invalid["images"] == {}
    assert "episode_restart" not in invalid
    assert benchmark._official_history_pending is True

    base = _observation(7)
    monkeypatch.setattr(RoboMMEBenchmark, "make_obs", lambda *_args: dict(base))

    first = benchmark.make_obs({}, {"name": "PickXtimes", "env_id": "PickXtimes"})
    assert first["episode_restart"] is True
    assert first["video_history"] == []
    assert first["video_state_history"] == []

    second = benchmark.make_obs({}, {"name": "PickXtimes", "env_id": "PickXtimes"})
    assert "episode_restart" not in second
    assert "video_history" not in second
    assert "video_state_history" not in second


def test_official_nonempty_demo_history_transport_is_unchanged(monkeypatch):
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (2, 3)]
    states = [np.asarray([value] + [0] * 7, dtype=np.float32) for value in (2, 3)]
    base = _observation(7)
    base.update(video_history=frames, episode_restart=True)
    monkeypatch.setattr(RoboMMEBenchmark, "make_obs", lambda *_args: dict(base))
    observation = _history_benchmark(states).make_obs({}, {"name": "MoveCube", "env_id": "MoveCube"})
    for actual, expected in zip(observation["video_history"], frames, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(observation["video_state_history"], states, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert observation["episode_restart"] is True

    mismatched = dict(base)
    mismatched["video_history"] = [*frames, frames[-1]]
    monkeypatch.setattr(RoboMMEBenchmark, "make_obs", lambda *_args: dict(mismatched))
    with pytest.raises(RuntimeError, match="lengths differ at transport"):
        _history_benchmark(states).make_obs({}, {"name": "MoveCube", "env_id": "MoveCube"})


def test_workspace_trainer_accepts_only_exact_or_reviewed_legacy_current_pair(tmp_path):
    trainer = _workspace_trainer_source()
    assert (
        validate_workspace_trainer_implementation(
            GPU_ENABLED_WORKSPACE_TRAINER_SHA256,
            trainer,
        )
        == "exact_full_source_sha256"
    )
    assert (
        validate_workspace_trainer_implementation(
            LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256,
            trainer,
        )
        == "reviewed_tpu_to_gpu_device_admission_only_v1"
    )

    with pytest.raises(ValueError, match="trainer implementation differs"):
        validate_workspace_trainer_implementation("0" * 64, trainer)

    one_byte_drift = tmp_path / "one-byte-drift.py"
    one_byte_drift.write_bytes(trainer.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="trainer implementation differs"):
        validate_workspace_trainer_implementation(
            LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256,
            one_byte_drift,
        )

    inference_drift = tmp_path / "inference-drift.py"
    source = trainer.read_bytes()
    needle = b"def encode(params, history, history_mask):"
    assert source.count(needle) == 1
    inference_drift.write_bytes(source.replace(needle, b"def encode_drift(params, history, history_mask):", 1))
    with pytest.raises(ValueError, match="trainer implementation differs"):
        validate_workspace_trainer_implementation(
            LEGACY_TPU_ONLY_WORKSPACE_TRAINER_SHA256,
            inference_drift,
        )


def test_all16_workspace_router_preserves_row_order_and_task_encoder_identity():
    from robomme_integration.training.single_task import TASK_ORDER

    frames = _FakeFrameEncoder()

    class OffsetOmega(_FakeOmegaEncoder):
        def __init__(self, offset: float):
            super().__init__()
            self.offset = offset

        def __call__(self, history: np.ndarray, mask: np.ndarray) -> np.ndarray:
            return super().__call__(history, mask) + self.offset

    def runner(offset: float) -> OnlineWorkspaceRunner:
        return OnlineWorkspaceRunner(
            frames,
            OffsetOmega(offset),
            state_mean=np.zeros(8, dtype=np.float32),
            state_std=np.ones(8, dtype=np.float32),
            history_stride=10,
            max_history=128,
        )

    routed = TaskRoutedOnlineWorkspaceRunner(
        frames,
        {task: pathlib.Path("/unused") / task for task in TASK_ORDER},
    )
    routed._runners = {
        "PickXtimes": runner(100.0),
        "ButtonUnmaskSwap": runner(200.0),
    }
    button = WorkspaceSession("ButtonUnmaskSwap", 0)
    pick = WorkspaceSession("PickXtimes", 0)
    capture_workspace_observation(button, _observation(3, video_markers=[1]))
    capture_workspace_observation(pick, _observation(7, video_markers=[2]))
    values = routed.windows([button, pick], window=1)
    np.testing.assert_array_equal(values[0], np.full((1, OMEGA_DIM), 203.0))
    np.testing.assert_array_equal(values[1], np.full((1, OMEGA_DIM), 107.0))


def test_workspace_history_and_steering_indices_match_training_geometry():
    assert causal_history_indices(0, stride=10, max_history=128) == (0,)
    assert causal_history_indices(16, stride=10, max_history=128) == (6, 16)
    assert causal_history_indices(1_276, stride=10, max_history=128)[0] == 6
    assert len(causal_history_indices(1_276, stride=10, max_history=128)) == 128
    assert requested_omega_steps(16, window=8, stride=10) == (0, 0, 0, 0, 0, 0, 6, 16)
    assert requested_omega_steps(86, window=8, stride=10) == (16, 26, 36, 46, 56, 66, 76, 86)


def test_online_workspace_uses_demo_dense_history_and_reuses_exact_d8_overlap():
    runner, frames, omega = _runner()
    session = WorkspaceSession("PickXtimes", 3)
    capture_workspace_observation(session, _observation(16, video_markers=list(range(16))))
    first = runner.windows([session], window=8)[0]
    assert frames.calls == [list(range(17))]
    assert first.shape == (8, OMEGA_DIM)
    assert first[:, 0].tolist() == [0, 0, 0, 0, 0, 0, 6, 16]
    assert len(omega.calls) == 1 and omega.calls[0][0].shape[0] == 3
    histories, masks = omega.calls[0]
    assert [int(mask.sum()) for mask in masks] == [1, 1, 2]
    assert histories[2, :2, 0].tolist() == [6, 16]

    for marker in range(17, 27):
        capture_workspace_observation(session, _observation(marker))
    second = runner.windows([session], window=8)[0]
    assert second[:, 0].tolist() == [0, 0, 0, 0, 0, 6, 16, 26]
    assert len(omega.calls) == 2 and omega.calls[1][0].shape[0] == 1
    assert set(session.omega_by_step) == {0, 6, 16, 26}


def test_online_workspace_batches_without_sharing_episode_state():
    runner, _frames, _omega = _runner()
    first = WorkspaceSession("PickXtimes", 1)
    second = WorkspaceSession("PickXtimes", 2)
    capture_workspace_observation(first, _observation(2, video_markers=[1]))
    capture_workspace_observation(second, _observation(102, video_markers=[101]))
    values = runner.windows([first, second], window=1)
    assert values[0][0, 0] == 2
    assert values[1][0, 0] == 102
    assert first.omega_by_step[1][0] != second.omega_by_step[1][0]
    with pytest.raises(ValueError, match="distinct"):
        runner.windows([first, first], window=1)


def test_workspace_capture_fails_closed_on_missing_or_repeated_demo_state():
    session = WorkspaceSession("PickXtimes", 0)
    missing = _observation(1, video_markers=[0])
    missing.pop("video_state_history")
    with pytest.raises(ValueError, match="paired"):
        capture_workspace_observation(session, missing)

    valid = WorkspaceSession("PickXtimes", 0)
    capture_workspace_observation(valid, _observation(1, video_markers=[0]))
    with pytest.raises(RuntimeError, match="after episode history began"):
        capture_workspace_observation(valid, _observation(2, video_markers=[0]))


def test_zero_demo_envelope_uses_current_frame_and_all_invalid_envelopes_fail_closed():
    runner, frames, omega = _runner()
    session = WorkspaceSession("PickXtimes", 0)
    capture_workspace_observation(session, _observation(9, video_markers=[]))
    value = runner.windows([session], window=1)[0]
    assert session.saw_video_history is True
    assert session.dense_observations == 1
    assert frames.calls == [[9]]
    assert len(omega.calls) == 1
    np.testing.assert_array_equal(value, np.full((1, OMEGA_DIM), 9.0))

    absent = WorkspaceSession("PickXtimes", 1)
    with pytest.raises(ValueError, match="explicit official episode_restart"):
        capture_workspace_observation(absent, _observation(4))

    mismatched = _observation(4, video_markers=[])
    mismatched["video_state_history"] = [np.zeros(8, dtype=np.float32)]
    with pytest.raises(ValueError, match="lengths differ"):
        capture_workspace_observation(WorkspaceSession("PickXtimes", 2), mismatched)

    late = _observation(10, video_markers=[])
    with pytest.raises(RuntimeError, match="after episode history began"):
        capture_workspace_observation(session, late)

    late_without_restart = _observation(11)
    late_without_restart.update(video_history=[], video_state_history=[])
    with pytest.raises(RuntimeError, match="after episode history began"):
        capture_workspace_observation(session, late_without_restart)

    repeated = WorkspaceSession("PickXtimes", 3)
    capture_workspace_observation(repeated, _observation(5, video_markers=[]))
    with pytest.raises(RuntimeError, match="after episode history began"):
        capture_workspace_observation(repeated, _observation(6, video_markers=[]))


def test_auxiliary_heads_are_train_only_and_workspace_windows_are_checkpoint_defined():
    assert serving_model_arm("jepa_l01_k1") == "s0"
    assert serving_model_arm("salient") == "s0"
    assert serving_model_arm("wsm_tanh") == "wsm_tanh"
    assert serving_model_arm("q0_noforce") == "q0"
    assert serving_model_arm("q2_noforce") == "q2"
    assert workspace_window_for_arm("wsm_cfg") == 1
    assert workspace_window_for_arm("q1") == 1
    assert workspace_window_for_arm("q3") == 1
    assert workspace_window_for_arm("wsm_d8") == 8
    assert workspace_window_for_arm("wsm_d8_drop05") == 8
    assert workspace_window_for_arm("wsm_d16") == 16
    assert workspace_window_for_arm("wsm_d16_drop05") == 16
    assert workspace_window_for_arm("gdn8_jepa_l01_k1") == 8
    assert serving_model_arm("gdn8_jepa_l01_k1") == "wsm_d8"
    assert workspace_window_for_arm("ptrm") == 8
    assert serving_model_arm("causal_v1") == "s0"
    with pytest.raises(ValueError, match="does not consume"):
        workspace_window_for_arm("salient")


def test_workspace_eval_dry_run_routes_all_provenance_paths(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    server = source / "robomme_integration/eval/execution_model_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# sealed workspace test server\n")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    workspace = tmp_path / "workspace" / "10000"
    workspace.mkdir(parents=True)
    for name in ("WSM_RUN_CONFIG.json", "WSM_BEST.json", "WSM_GENERATION_COMPLETE.json"):
        (workspace / name).write_text("{}\n")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    vision = tmp_path / "vision" / "pi05_vision_encoder"
    vision.mkdir(parents=True)
    (vision / "siglip_params.pkl").write_bytes(b"sealed")
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text("benchmark: test\n")
    executable = tmp_path / "vla-eval"
    executable.write_text("#!/bin/sh\n")

    names = (
        "checkpoint",
        "arm",
        "workspace_checkpoint",
        "upstream_root",
        "vision_encoder_home",
        "cfg_guidance_scale",
        "task_name",
        "model_seed",
        "chunk_size",
        "max_batch_size",
    )
    help_text = " ".join(f"--args.{name}" for name in names)
    monkeypatch.setattr(
        "gcp_tpu.run_execution_eval.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout=help_text),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_execution_eval.py",
            "--source-root",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--arm",
            "wsm_d8",
            "--workspace-checkpoint",
            str(workspace),
            "--upstream-root",
            str(upstream),
            "--vision-encoder-home",
            str(vision.parent),
            "--task",
            "PickXtimes",
            "--benchmark-config",
            str(benchmark),
            "--output-root",
            str(tmp_path / "output"),
            "--eval-id",
            "d8-dry-run",
            "--vla-eval",
            str(executable),
            "--dry-run",
        ],
    )
    assert execution_eval_main() == 0
    command = json.loads(capsys.readouterr().out)["server"]
    assert command[2] == "robomme_integration.eval.execution_model_server"
    for flag, value in (
        ("--args.arm", "wsm_d8"),
        ("--args.workspace_checkpoint", str(workspace)),
        ("--args.upstream_root", str(upstream)),
        ("--args.vision_encoder_home", str(vision.parent)),
    ):
        index = command.index(flag)
        assert command[index + 1] == value
