#!/usr/bin/env python3
"""Serve project RoboMME controls over the pinned paper-evaluation wire protocol.

This is deliberately separate from both ``execution_model_server.py`` (the vla-eval protocol)
and ``official_reference_eval.py`` (the currently running released-checkpoint campaign).  It
adapts only the execution-only, stateless project controls S0/Q0/A6.  All three predict 20 actions
and are replanned after the evaluator executes 16.

Current Q2 checkpoints are rejected.  They were trained with 10-step decision spacing and a
10-action fast-weight commit.  Under a 16-step execution contract, committing 10 omits six
executed actions while committing 16 changes the learned update's token distribution.  Neither is
an exact, train/serve-matched Q2 estimand; Q2 needs a stride-16 retrain (or a separately labelled
sensitivity study) before it may enter this protocol-matched table.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http
import logging
import re
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROTOCOL_ID = "robomme-paper856-h20-e16-fixed50-project-v1"
ACTION_HORIZON = 20
EXECUTION_HORIZON = 16
EPISODES_PER_TASK = 50
SUPPORTED_ARMS = frozenset({"s0", "q0", "a6", "v4_s0"})
Q2_BLOCKER = (
    "current Q2 is trained with stride/commit=10; execute=16 cannot preserve both complete "
    "causal commits and train/serve update parity. Retrain Q2 with stride/commit=16 or report a "
    "separately labelled c10/e16 or c16/e16 sensitivity study"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_sha256(name: str, value: str) -> str:
    value = str(value).lower()
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA256, got {value!r}")
    return value


def validate_project_arm(arm: str) -> str:
    arm = str(arm).lower()
    if arm == "q2":
        raise ValueError(Q2_BLOCKER)
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"project exact bridge supports only {sorted(SUPPORTED_ARMS)}, got {arm!r}")
    return arm


def validate_server_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail closed when the evaluator connected to a different server or checkpoint."""
    if not isinstance(metadata, dict):
        raise RuntimeError("project exact server metadata is not a dictionary")
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"project exact server identity mismatch: {mismatches}")


def _default_context_factory(session_id: str, episode_id: str):
    from vla_eval.model_servers.base import SessionContext

    return SessionContext(session_id=session_id, episode_id=episode_id, mode="sync")


def _default_tasks() -> frozenset[str]:
    from robomme_integration.training.single_task import TASK_EPISODES

    return frozenset(TASK_EPISODES)


def _create_execution_core(checkpoint: str, arm: str, model_seed: int):
    from robomme_integration.eval.execution_model_server import RoboMMEExecutionModelServer

    # ``chunk_size=10`` is the immutable project-training stride, but this bridge calls predict()
    # directly and therefore bypasses PredictModelServer's action buffer/trimming.  For these
    # stateless arms it has no model semantics: the returned plan remains the full (20, 8).
    return RoboMMEExecutionModelServer(
        checkpoint,
        arm=arm,
        task_name="all16",
        model_seed=model_seed,
        chunk_size=10,
        max_batch_size=1,
    )


class ProjectExactBridge:
    """Shared model core plus one lifecycle object per WebSocket connection."""

    def __init__(
        self,
        checkpoint: str,
        *,
        arm: str,
        checkpoint_sha256: str,
        project_source_sha256: str,
        openpi_source_sha256: str,
        model_seed: int = 7,
        core: Any | None = None,
        context_factory: Callable[[str, str], Any] = _default_context_factory,
        known_tasks: frozenset[str] | None = None,
    ) -> None:
        self.arm = validate_project_arm(arm)
        self.checkpoint_sha256 = _require_sha256("checkpoint_sha256", checkpoint_sha256)
        self.project_source_sha256 = _require_sha256("project_source_sha256", project_source_sha256)
        self.openpi_source_sha256 = _require_sha256("openpi_source_sha256", openpi_source_sha256)
        self.model_seed = int(model_seed)
        if self.model_seed != 7:
            raise ValueError("paper-matched project evaluation fixes model_seed=7")
        self.core = core if core is not None else _create_execution_core(checkpoint, self.arm, self.model_seed)
        self.context_factory = context_factory
        self.known_tasks = known_tasks if known_tasks is not None else _default_tasks()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "arm": self.arm,
            "checkpoint_sha256": self.checkpoint_sha256,
            "project_source_sha256": self.project_source_sha256,
            "openpi_source_sha256": self.openpi_source_sha256,
            "server_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_seed": self.model_seed,
            "action_horizon": ACTION_HORIZON,
            "execution_horizon": EXECUTION_HORIZON,
            "history_mode": "forbidden_execution_only",
            "q2_supported": False,
        }

    def connection(self) -> "ProjectExactConnection":
        return ProjectExactConnection(self)


class ProjectExactConnection:
    """Exact episode identity and step clock for one old-protocol connection."""

    def __init__(self, bridge: ProjectExactBridge) -> None:
        self.bridge = bridge
        self.ctx: Any | None = None
        self.task_name: str | None = None
        self.episode_idx: int | None = None
        self.decisions = 0
        self.closed = False

    async def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.closed or self.ctx is not None:
            raise RuntimeError("each project-exact connection permits exactly one episode reset")
        required = self.bridge.metadata
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in required.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"project episode reset contract mismatch: {mismatches}")
        task_name = str(payload.get("task_name", ""))
        if task_name not in self.bridge.known_tasks:
            raise ValueError(f"unknown RoboMME task in reset: {task_name!r}")
        episode_idx = int(payload.get("episode_idx", -1))
        if not 0 <= episode_idx < EPISODES_PER_TASK:
            raise ValueError(f"episode_idx must lie in [0,49], got {episode_idx}")
        session_id = str(uuid.uuid4())
        self.ctx = self.bridge.context_factory(session_id, f"{task_name}:{episode_idx}")
        self.task_name = task_name
        self.episode_idx = episode_idx
        await self.bridge.core.on_episode_start(
            {"task": {"name": task_name, "env_id": task_name, "episode_idx": episode_idx}},
            self.ctx,
        )
        return {"reset_finished": True, "protocol_id": PROTOCOL_ID}

    @staticmethod
    def _vla_observation(payload: dict[str, Any]) -> dict[str, Any]:
        required = (
            "observation/image",
            "observation/wrist_image",
            "observation/state",
            "prompt",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"paper-protocol inference request is missing {missing}")
        front = np.asarray(payload["observation/image"])
        wrist = np.asarray(payload["observation/wrist_image"])
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        if front.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("RoboMME paper-protocol images must be uint8")
        if front.ndim != 3 or front.shape[-1] != 3 or wrist.shape != front.shape:
            raise ValueError(f"front/wrist image geometry differs or is not HWC RGB: {front.shape}, {wrist.shape}")
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"RoboMME joint-angle state must be finite shape (8,), got {state.shape}")
        prompt = payload["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("RoboMME task prompt must be a nonempty string")
        return {
            "images": {"agentview": front, "wrist": wrist},
            "states": state,
            "task_description": prompt,
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.closed or self.ctx is None:
            raise RuntimeError("project-exact inference arrived before episode reset")
        expected_step = self.decisions * EXECUTION_HORIZON
        if int(self.ctx.step) != expected_step:
            raise RuntimeError(f"project exact step clock drifted: {self.ctx.step} != {expected_step}")
        result = self.bridge.core.predict(self._vla_observation(payload), self.ctx)
        if not isinstance(result, dict):
            raise RuntimeError("project policy result is not a dictionary")
        actions = np.asarray(result.get("actions"))
        if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
            raise RuntimeError(f"project action contract requires finite (20,8), got {actions.shape}")
        if "norm_state" in result or "norm_actions" in result:
            raise RuntimeError("model-space Q2 fields leaked from an execution-only control")
        for _ in range(EXECUTION_HORIZON):
            self.ctx._increment_step()
        self.decisions += 1
        return {**result, "actions": actions.astype(np.float32, copy=False)}

    def add_buffer(self, _payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("S0/Q0/A6 are execution-only; video-history add_buffer is forbidden")

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.ctx is not None:
            await self.bridge.core.on_episode_end(
                {
                    "protocol_id": PROTOCOL_ID,
                    "task_name": self.task_name,
                    "episode_idx": self.episode_idx,
                    "decision_count": self.decisions,
                    "connection_closed": True,
                },
                self.ctx,
            )


async def serve(bridge: ProjectExactBridge, host: str, port: int) -> None:
    import websockets
    import websockets.asyncio.server as websocket_server
    import websockets.frames
    from openpi_client import msgpack_numpy

    def health_check(
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> websocket_server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def handler(websocket: websocket_server.ServerConnection) -> None:
        connection = bridge.connection()
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(bridge.metadata))
        try:
            while True:
                payload = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(payload, dict):
                    raise ValueError("paper-protocol request is not a dictionary")
                if payload.get("reset", False):
                    response = await connection.reset(payload)
                elif payload.get("add_buffer", False):
                    response = connection.add_buffer(payload)
                else:
                    response = connection.infer(payload)
                await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Project exact bridge error; traceback sent in previous frame.",
            )
            raise
        finally:
            await connection.close()

    logging.info("project exact server arm=%s address=ws://%s:%d", bridge.arm, host, port)
    logging.info("project exact identity=%s", bridge.metadata)
    async with websocket_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm", required=True, choices=("s0", "q0", "a6", "q2", "v4_s0"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--project-source-sha256", required=True)
    parser.add_argument("--openpi-source-sha256", required=True)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    bridge = ProjectExactBridge(
        args.checkpoint,
        arm=args.arm,
        checkpoint_sha256=args.checkpoint_sha256,
        project_source_sha256=args.project_source_sha256,
        openpi_source_sha256=args.openpi_source_sha256,
        model_seed=args.model_seed,
    )
    asyncio.run(serve(bridge, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
