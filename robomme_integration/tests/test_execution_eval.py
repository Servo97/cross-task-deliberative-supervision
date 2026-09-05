from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import pathlib
import sys
import threading
import types
from unittest import mock

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

from vla_eval.model_servers.base import SessionContext  # noqa: E402

from gcp_tpu.run_execution_eval import (  # noqa: E402
    _server_argument_flag,
    _server_module,
)
from gcp_tpu.run_execution_eval import (  # noqa: E402
    main as execution_eval_main,
)
from robomme_integration.eval.execution_model_server import (  # noqa: E402
    RoboMMEExecutionModelServer,
    _LoadedModelState,
)
from robomme_integration.eval.execution_runner import PendingExecutionCommit  # noqa: E402
from robomme_integration.eval.launch_sharded import (  # noqa: E402
    _audit_episode_results,
    _build_shard_command,
    _container_command,
    _cpu_partitions,
)
from robomme_integration.eval.workspace_runner import (  # noqa: E402
    FEATURE_DIM,
    OMEGA_DIM,
    OnlineWorkspaceRunner,
    capture_workspace_observation,
)


class _Policy:
    def __init__(self, expose_norm: bool):
        self.expose_norm = expose_norm
        self.requests: list[dict] = []

    def infer_batch(self, requests):
        self.requests.extend(requests)
        results = []
        for request in requests:
            marker = float(np.asarray(request["observation/state"])[0])
            result = {
                "actions": np.full((20, 8), marker, dtype=np.float32),
                "policy_timing": {"fake": True},
            }
            if self.expose_norm:
                result.update(
                    norm_state=np.full((32,), marker, dtype=np.float32),
                    norm_actions=np.full((20, 32), marker, dtype=np.float32),
                )
            results.append(result)
        return results


class _Runner:
    def init_state(self):
        return {"h": np.zeros((1, 1), dtype=np.float32)}

    def condition_many(self, states, raw_observations):
        assert len(states) == len(raw_observations)
        return np.stack([np.full((4,), state["h"][0, 0], dtype=np.float32) for state in states])

    def commit_many(self, states, pending):
        assert all(isinstance(item, PendingExecutionCommit) for item in pending)
        return [
            {"h": state["h"] + np.asarray(item.actions, dtype=np.float32).mean()}
            for state, item in zip(states, pending, strict=True)
        ]

    @staticmethod
    def delta_norm(state, initial_state):
        return float(np.linalg.norm(state["h"] - initial_state["h"]))


def _obs(marker: int, *, video: bool = False) -> dict:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    result = {
        "images": {"agentview": image, "wrist": image + 1},
        "states": np.asarray([marker] + [0] * 8, dtype=np.float32),
        "task_description": "Pick X times",
    }
    if video:
        result["video_history"] = [image]
    return result


def _ctx(session: str) -> SessionContext:
    return SessionContext(session_id=session, episode_id=f"episode-{session}")


def _start(server, ctx, episode_idx=3):
    asyncio.run(
        server.on_episode_start(
            {"task": {"name": "PickXtimes", "env_id": "PickXtimes", "episode_idx": episode_idx}},
            ctx,
        )
    )


def _server(arm: str):
    server = _unloaded_server(arm=arm, task_name="PickXtimes")
    policy = _Policy(expose_norm=arm in {"q2", "q2_noforce"})
    server._policy = policy
    if arm in {"q2", "q2_noforce"}:
        server._runner = _Runner()
    return server, policy


def _unloaded_server(**kwargs):
    """Construct a test server while proving production construction requests eager loading."""
    with mock.patch.object(RoboMMEExecutionModelServer, "_load_model", autospec=True) as load:
        server = RoboMMEExecutionModelServer("/unused", **kwargs)
    load.assert_called_once_with(server)
    return server


def test_model_initialization_is_atomic_under_concurrent_first_use():
    server = _unloaded_server(
        arm="q3",
        task_name="PickXtimes",
        workspace_checkpoint="/workspace/10000",
    )
    policy = _Policy(expose_norm=True)
    runner = _Runner()
    workspace_runner = object()
    calls = 0
    start = threading.Barrier(9)
    initializing = threading.Event()
    release = threading.Event()

    def initialize():
        nonlocal calls
        calls += 1
        initializing.set()
        assert release.wait(timeout=5)
        return _LoadedModelState(policy, runner, workspace_runner, {"complete": True})

    server._initialize_model = initialize

    def load_after_barrier(_index):
        start.wait(timeout=5)
        server._load_model()
        # No caller may return from loading with only the policy half published.
        assert server._policy is policy
        assert server._runner is runner
        assert server._workspace_runner is workspace_runner

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load_after_barrier, index) for index in range(8)]
        start.wait(timeout=5)
        assert initializing.wait(timeout=5)
        release.set()
        for future in futures:
            future.result(timeout=5)

    assert calls == 1


def test_failed_model_initialization_clears_every_member_and_can_retry():
    server = _unloaded_server(
        arm="q3",
        task_name="PickXtimes",
        workspace_checkpoint="/workspace/10000",
    )
    server._runner = object()
    server._workspace_runner = object()
    policy = _Policy(expose_norm=True)
    runner = _Runner()
    workspace_runner = object()
    attempts = 0

    def initialize():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("broken workspace checkpoint")
        return _LoadedModelState(policy, runner, workspace_runner, {"complete": True})

    server._initialize_model = initialize
    with pytest.raises(ValueError, match="broken workspace checkpoint"):
        server._load_model()
    assert server._policy is None
    assert server._runner is None
    assert server._workspace_runner is None

    server._load_model()
    assert server._policy is policy
    assert server._runner is runner
    assert server._workspace_runner is workspace_runner
    assert attempts == 2


def test_server_never_advertises_readiness_after_eager_restore_failure():
    with mock.patch.object(
        RoboMMEExecutionModelServer,
        "_initialize_model",
        autospec=True,
        side_effect=RuntimeError("restore failed"),
    ):
        with pytest.raises(RuntimeError, match="restore failed"):
            RoboMMEExecutionModelServer("/unused", arm="s0", task_name="PickXtimes")


@pytest.mark.parametrize("arm", ("q2", "q2_noforce"))
def test_q2_commits_only_at_the_next_replan_and_keeps_sessions_isolated(arm):
    server, policy = _server(arm)
    ctx_a, ctx_b = _ctx("a"), _ctx("b")
    _start(server, ctx_a)
    _start(server, ctx_b)

    first = server.predict_batch([_obs(1, video=True), _obs(2)], [ctx_a, ctx_b])
    assert len(first) == 2
    assert "norm_state" not in first[0] and "norm_actions" not in first[0]
    assert server._episodes["a"].commits == server._episodes["b"].commits == 0
    assert np.array_equal(policy.requests[0]["robottt_cond"], np.zeros(4, dtype=np.float32))
    assert np.array_equal(policy.requests[1]["robottt_cond"], np.zeros(4, dtype=np.float32))
    assert server._episodes["a"].saw_video_history

    for _ in range(10):
        ctx_a._increment_step()
        ctx_b._increment_step()
    server.predict_batch([_obs(3), _obs(4)], [ctx_a, ctx_b])
    assert server._episodes["a"].commits == server._episodes["b"].commits == 1
    assert np.array_equal(policy.requests[2]["robottt_cond"], np.ones(4, dtype=np.float32))
    assert np.array_equal(policy.requests[3]["robottt_cond"], np.full(4, 2, dtype=np.float32))


def test_q3_zero_demo_reset_builds_current_frame_omega_and_returns_an_action():
    server = _unloaded_server(
        arm="q3",
        task_name="PickXtimes",
        workspace_checkpoint="/workspace/10000",
    )
    policy = _Policy(expose_norm=True)

    def frame_encoder(images):
        markers = np.asarray(images[:, 0, 0, 0], dtype=np.float32)
        return np.repeat(markers[:, None], FEATURE_DIM, axis=1).astype(np.float16)

    def omega_encoder(history, mask):
        last = np.asarray([row[np.flatnonzero(valid)[-1], 0] for row, valid in zip(history, mask, strict=True)])
        return np.repeat(last[:, None], OMEGA_DIM, axis=1)

    server._policy = policy
    server._runner = _Runner()
    server._workspace_runner = OnlineWorkspaceRunner(
        frame_encoder,
        omega_encoder,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
        history_stride=10,
        max_history=128,
    )
    ctx = _ctx("q3-empty-history")
    _start(server, ctx, episode_idx=0)
    observation = _obs(9)
    observation.update(
        episode_restart=True,
        video_history=[],
        video_state_history=[],
    )
    workspace = server._episodes[ctx.session_id].workspace
    assert workspace is not None
    capture_workspace_observation(workspace, observation)

    result = server.predict(observation, ctx)
    assert result["actions"].shape == (20, 8)
    assert np.all(result["actions"] == 9)
    assert workspace.saw_video_history is True
    assert workspace.dense_observations == 1
    assert policy.requests[0]["wsm_w_window"].shape == (1, OMEGA_DIM)
    assert np.all(policy.requests[0]["wsm_w_window"] == 9)


def test_s0_q0_and_q2_use_the_same_manifest_derived_diffusion_seed():
    base, base_policy = _server("s0")
    q0, q0_policy = _server("q0")
    q2, q2_policy = _server("q2")
    base_ctx, q0_ctx, q2_ctx = _ctx("base"), _ctx("q0"), _ctx("q2")
    _start(base, base_ctx, episode_idx=17)
    _start(q0, q0_ctx, episode_idx=17)
    _start(q2, q2_ctx, episode_idx=17)
    base.predict(_obs(1), base_ctx)
    q0.predict(_obs(1), q0_ctx)
    q2.predict(_obs(1), q2_ctx)
    assert (
        base_policy.requests[0]["policy_noise_seed"]
        == q0_policy.requests[0]["policy_noise_seed"]
        == q2_policy.requests[0]["policy_noise_seed"]
    )


def test_specs_match_robomme_components_and_single_task_binding_fails_closed():
    server, _ = _server("s0")
    assert set(server.get_observation_spec()) == {"agentview", "wrist", "state", "language"}
    assert set(server.get_action_spec()) == {"action"}
    ctx = _ctx("wrong-task")
    try:
        asyncio.run(
            server.on_episode_start(
                {"task": {"name": "ButtonUnmaskSwap", "episode_idx": 0}},
                ctx,
            )
        )
    except ValueError as error:
        assert "bound to PickXtimes" in str(error)
    else:
        raise AssertionError("single-task server accepted the wrong RoboMME task")


def test_multitask_server_accepts_multiple_known_tasks_and_requires_workspace_index():
    server = _unloaded_server(arm="s0", task_name="all16")
    server._policy = _Policy(expose_norm=False)
    pick, button = _ctx("pick"), _ctx("button")
    _start(server, pick)
    asyncio.run(server.on_episode_start({"task": {"name": "ButtonUnmaskSwap", "episode_idx": 1}}, button))
    assert server._episodes["pick"].task_name == "PickXtimes"
    assert server._episodes["button"].task_name == "ButtonUnmaskSwap"
    try:
        RoboMMEExecutionModelServer("/unused", arm="wsm_cfg", task_name="all16")
    except ValueError as error:
        assert "requires a sealed workspace index" in str(error)
    else:
        raise AssertionError("multitask workspace serving bypassed its sealed index gate")
    routed = _unloaded_server(
        arm="wsm_cfg",
        task_name="all16",
        workspace_checkpoint="/workspace",
        workspace_index="/workspace-index.json",
    )
    assert routed.workspace_index == "/workspace-index.json"


def test_sharded_launcher_keeps_manifest_and_cpu_shard_arguments_explicit(tmp_path):
    executable = tmp_path / "vla-eval"
    config = tmp_path / "pick.yaml"
    output = tmp_path / "results"
    command = _build_shard_command(
        executable,
        config,
        output,
        shard=3,
        shards=8,
        eval_id="pick-base-canary",
    )
    assert command == [
        str(executable),
        "run",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--shard-id",
        "3",
        "--num-shards",
        "8",
        "--yes",
        "--eval-id",
        "pick-base-canary",
    ]

    assert _cpu_partitions("0-239", 8) == [
        "0-29",
        "30-59",
        "60-89",
        "90-119",
        "120-149",
        "150-179",
        "180-209",
        "210-239",
    ]
    wrapped = _container_command(
        [*command, "--no-docker"],
        image="wsm/robomme-cpu-runtime:ubuntu22",
        name="pick-base-canary-shard-03",
        cpus="90-119",
        user="2001:2001",
        mounts=[tmp_path],
    )
    assert wrapped[:3] == ["sudo", "docker", "run"]
    assert "seccomp=unconfined" in wrapped
    assert "90-119" in wrapped
    assert wrapped[-len(command) - 2 :] == ["wsm/robomme-cpu-runtime:ubuntu22", *command, "--no-docker"]


def test_official_ttt_routes_to_the_session_isolated_server():
    assert _server_module("t1") == "robomme_integration.eval.upstream_ttt_model_server"
    assert _server_module("s0") == "robomme_integration.eval.execution_model_server"


def test_t1_dry_run_builds_one_complete_nested_cli(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    server = source / "robomme_integration/eval/upstream_ttt_model_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# sealed test server\n")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text("benchmark: test\n")
    executable = tmp_path / "vla-eval"
    executable.write_text("#!/bin/sh\n")

    help_text = " ".join(
        f"--args.{name}"
        for name in ("checkpoint", "upstream_root", "task_name", "model_seed", "chunk_size", "max_batch_size")
    )
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
            "t1",
            "--upstream-root",
            str(upstream),
            "--policy-pythonpath",
            str(overlay),
            "--task",
            "PickXtimes",
            "--benchmark-config",
            str(benchmark),
            "--output-root",
            str(tmp_path / "output"),
            "--eval-id",
            "t1-dry-run",
            "--vla-eval",
            str(executable),
            "--dry-run",
        ],
    )
    assert execution_eval_main() == 0
    payload = json.loads(capsys.readouterr().out)
    command = payload["server"]
    assert command[2] == "robomme_integration.eval.upstream_ttt_model_server"
    assert command[3:7] == [
        "--args.checkpoint",
        str(checkpoint),
        "--args.upstream_root",
        str(upstream),
    ]
    assert "--args.arm" not in command


def test_result_audit_rejects_harness_errors_but_not_task_failure(tmp_path):
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "PickXtimes",
                        "episodes": [
                            {"episode_id": 0, "metrics": {"success": False}, "steps": 50},
                            {
                                "episode_id": 1,
                                "metrics": {"success": False},
                                "steps": 0,
                                "failure_reason": "timeout",
                                "failure_detail": "timeout=30s",
                            },
                        ],
                    }
                ]
            }
        )
    )
    audit = _audit_episode_results([aggregate])
    assert audit["episodes"] == 2
    assert audit["harness_failures"] == [
        {
            "path": str(aggregate),
            "task": "PickXtimes",
            "episode_id": 1,
            "reason": "timeout",
            "detail": "timeout=30s",
        }
    ]


def test_eval_supervisor_adapts_to_flat_and_nested_server_clis():
    assert _server_argument_flag("usage: server --checkpoint CHECKPOINT", "checkpoint") == (
        "--checkpoint",
        "flat",
    )
    assert _server_argument_flag("usage: server --args.checkpoint CHECKPOINT", "checkpoint") == (
        "--args.checkpoint",
        "nested",
    )
