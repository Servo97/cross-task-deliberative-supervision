#!/usr/bin/env python3
"""Authenticated local policy/evaluator boundary for the offline FrameSamp-AM oracle.

The prompt is observation content, never route identity.  Reset authenticates a
task and episode.  Every replan must then carry the exact causal cut, trusted
artifact-index path+SHA, stack-receipt path+SHA, and ordered teacher-tap-stack
SHA.  The server reopens those receipts, verifies the active staged policy
overlay/checkpoint/code/dtype/device, resolves the seven dynamic JAX arrays,
and passes them explicitly to the already-compiled ``sample_actions`` method.

An offline artifact is valid only when its teacher taps came from the actual
on-policy history at that cut.  Therefore inference also requires an
independent online teacher-tap attestor.  With no attestor the adapter fails
closed; it does not pretend that matching task/episode/cut labels prove matching
history.  No official source or training ``compute_loss`` path is modified.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from robomme_integration.training.framesamp_am_artifact import ExpectedFrameSampAMIdentity
from robomme_integration.training.framesamp_am_index import load_framesamp_am_trusted_index
from robomme_integration.training.framesamp_am_oracle_route import (
    OfflineFrameSampAMOracleInputs,
    OfflineFrameSampAMStackManifest,
    load_offline_framesamp_am_stack_manifest,
    resolve_offline_framesamp_am_oracle_inputs,
)
from robomme_integration.training.framesamp_am_policy_overlay import verify_framesamp_am_policy_overlay

PROTOCOL_ID = "robomme-framesamp-am-e1-authenticated-compact-all-v1"
ACTION_HORIZON = 20
ACTION_DIM = 8
EXECUTION_HORIZON = 16
_HEX = frozenset("0123456789abcdef")
_DYNAMIC_NAMES = frozenset(
    {
        "framesamp_am_compact_k",
        "framesamp_am_compact_v",
        "framesamp_am_compact_beta",
        "framesamp_am_compact_mask",
        "framesamp_am_recent_positions",
        "framesamp_am_recent_mem_seq",
        "framesamp_am_recent_mem_mask",
    }
)


def _require_sha(value: object, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(character not in _HEX for character in value):
        raise ValueError(f"{label} must be a lowercase {'/'.join(map(str, lengths))}-hex SHA")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return int(value)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_receipt_file(root: Path, relative_path: object, *, label: str) -> Path:
    text = _require_nonempty(relative_path, label=label)
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or relative.as_posix() != text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink: {text}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:  # pragma: no cover - normalized path and symlink checks are stronger.
        raise ValueError(f"{label} escapes the configured artifact root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {text}")
    return resolved


def _require_source_below_overlay(value: object, overlay_root: Path, *, label: str) -> None:
    source = Path(inspect.getfile(value)).resolve(strict=True)
    try:
        source.relative_to(overlay_root)
    except ValueError as error:
        raise ValueError(f"{label} was not imported from the verified FrameSamp-AM policy overlay") from error


def derive_teacher_tap_stack_sha256(
    trusted_index_path: str | Path,
    *,
    expected_trusted_index_sha256: str,
    stack: OfflineFrameSampAMStackManifest,
) -> str:
    """Derive the ordered teacher-tap identity sealed by one stack receipt."""

    trusted = load_framesamp_am_trusted_index(
        trusted_index_path,
        expected_sha256=expected_trusted_index_sha256,
        verify_artifacts=False,
    )
    layers = []
    for pin in stack.layer_pins:
        loaded = trusted.resolve(
            ExpectedFrameSampAMIdentity(
                teacher_checkpoint_sha256=stack.teacher_checkpoint_sha256,
                teacher_code_sha=stack.teacher_code_sha,
                task_id=stack.task_id,
                episode_id=stack.episode_id,
                causal_cut_step=stack.causal_cut_step,
                layer_index=pin.layer_index,
                kv_head_index=0,
                requested_budget=stack.requested_budget,
                manifest_sha256=pin.manifest_sha256,
            )
        )
        layers.append(
            {
                "layer_index": pin.layer_index,
                "manifest_sha256": pin.manifest_sha256,
                "teacher_tap_sha256": loaded.manifest.teacher_tap_sha256,
            }
        )
    identity = {
        "kind": "robomme_framesamp_am_ordered_teacher_tap_stack_v1",
        "teacher_checkpoint_sha256": stack.teacher_checkpoint_sha256,
        "teacher_code_sha": stack.teacher_code_sha,
        "task_id": stack.task_id,
        "episode_id": stack.episode_id,
        "causal_cut_step": stack.causal_cut_step,
        "layers": layers,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


class OraclePolicyInvoker(Protocol):
    model_dtype: str

    def reset(self) -> None: ...

    def infer(self, observation: Mapping[str, Any], dynamic_inputs: Mapping[str, jax.Array]) -> Mapping[str, Any]: ...


class MMEVLAOraclePolicyInvoker:
    """Narrow runtime adapter around the verified staged ``MME_VLA_Policy``.

    The public released ``infer`` method cannot accept dynamic AM arrays.  This
    adapter reproduces its small transform/sample/output shell while bypassing
    ``_prepare_history``: schema-v2 compact-all memory is already represented by
    the authenticated layer stack and its raw-recent block is R=0.
    """

    def __init__(self, inner_policy: Any, *, verified_policy_overlay_root: str | Path) -> None:
        overlay = Path(verified_policy_overlay_root).resolve(strict=True)
        _require_source_below_overlay(type(inner_policy), overlay, label="MME_VLA_Policy")
        _require_source_below_overlay(type(inner_policy._model), overlay, label="HistoryPi0")
        from mme_vla_suite.models.integration.history_observation import HistAugObservation

        _require_source_below_overlay(HistAugObservation, overlay, label="HistAugObservation")
        required = (
            "_input_transform",
            "_output_transform",
            "_sample_actions",
            "_sample_kwargs",
            "_rng",
            "_model",
        )
        missing = [name for name in required if not hasattr(inner_policy, name)]
        if missing:
            raise ValueError(f"staged MME_VLA_Policy is missing runtime fields {missing}")
        self._inner = inner_policy
        self._observation_cls = HistAugObservation
        self.model_dtype = jnp.dtype(inner_policy._model.config.dtype).name

    def reset(self) -> None:
        self._inner.reset()

    def infer(self, observation: Mapping[str, Any], dynamic_inputs: Mapping[str, jax.Array]) -> Mapping[str, Any]:
        if set(dynamic_inputs) != _DYNAMIC_NAMES:
            raise ValueError("oracle invoker requires exactly the seven dynamic FrameSamp-AM inputs")
        overlap = set(self._inner._sample_kwargs) & set(dynamic_inputs)
        if overlap:
            raise ValueError(f"static policy sample_kwargs shadow dynamic AM inputs: {sorted(overlap)}")
        inputs = jax.tree.map(lambda value: value, dict(observation))
        inputs = self._inner._input_transform(inputs)
        model_observation = self._observation_cls.from_dict(
            jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
        )
        self._inner._rng, sample_rng = jax.random.split(self._inner._rng)
        start = time.monotonic()
        outputs = {
            "state": model_observation.state,
            "actions": self._inner._sample_actions(
                sample_rng,
                model_observation,
                **self._inner._sample_kwargs,
                **dynamic_inputs,
            ),
        }
        model_time = time.monotonic() - start
        outputs = jax.tree.map(lambda value: np.asarray(value[0, ...]), outputs)
        outputs = self._inner._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000
        return outputs


@dataclasses.dataclass(frozen=True)
class AuthenticatedOracleReplanReceipt:
    causal_cut_step: int
    trusted_index_relative_path: str
    trusted_index_sha256: str
    stack_receipt_relative_path: str
    stack_receipt_sha256: str
    teacher_tap_stack_sha256: str

    def validate(self) -> None:
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_nonempty(self.trusted_index_relative_path, label="trusted_index_relative_path")
        _require_sha(self.trusted_index_sha256, label="trusted_index_sha256")
        _require_nonempty(self.stack_receipt_relative_path, label="stack_receipt_relative_path")
        _require_sha(self.stack_receipt_sha256, label="stack_receipt_sha256")
        _require_sha(self.teacher_tap_stack_sha256, label="teacher_tap_stack_sha256")


@dataclasses.dataclass(frozen=True)
class AuthenticatedOracleEvaluatorRoute:
    """Evaluator-side explicit identity; no field is derived from the prompt."""

    task_id: str
    episode_id: str
    replans: tuple[AuthenticatedOracleReplanReceipt, ...]

    def validate(self) -> None:
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        if not isinstance(self.replans, tuple) or not self.replans:
            raise ValueError("authenticated evaluator route requires at least one replan receipt")
        for receipt in self.replans:
            receipt.validate()
        cuts = [receipt.causal_cut_step for receipt in self.replans]
        if cuts != sorted(set(cuts)):
            raise ValueError("authenticated evaluator causal cuts must be unique and strictly increasing")
        index_bindings = {
            (receipt.trusted_index_relative_path, receipt.trusted_index_sha256) for receipt in self.replans
        }
        if len(index_bindings) != 1:
            raise ValueError("one episode must remain bound to one trusted FrameSamp-AM index")

    def reset_payload(self, server_metadata: Mapping[str, Any]) -> dict[str, Any]:
        self.validate()
        return {**dict(server_metadata), "reset": True, "task_id": self.task_id, "episode_id": self.episode_id}

    def inference_payload(
        self,
        causal_cut_step: int,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.validate()
        matches = [receipt for receipt in self.replans if receipt.causal_cut_step == causal_cut_step]
        if len(matches) != 1:
            raise KeyError(f"no unique authenticated oracle receipt for causal cut {causal_cut_step}")
        receipt = matches[0]
        return {
            **dict(observation),
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            **dataclasses.asdict(receipt),
        }


HistoryAttestor = Callable[[str, str, int, Mapping[str, Any]], str]
OracleResolver = Callable[..., OfflineFrameSampAMOracleInputs]
TapDigestResolver = Callable[..., str]
StackLoader = Callable[..., OfflineFrameSampAMStackManifest]


class AuthenticatedFrameSampAMOracleBridge:
    """Verified policy/runtime identity shared by local server connections."""

    def __init__(
        self,
        invoker: OraclePolicyInvoker,
        *,
        artifact_root: str | Path,
        policy_overlay_root: str | Path,
        expected_policy_overlay_manifest_sha256: str,
        active_policy_checkpoint_sha256: str,
        expected_teacher_code_sha: str,
        active_model_dtype: str,
        expected_device_platform: str,
        device_or_sharding: Any,
        known_tasks: Sequence[str],
        history_attestor: HistoryAttestor | None,
        oracle_resolver: OracleResolver = resolve_offline_framesamp_am_oracle_inputs,
        tap_digest_resolver: TapDigestResolver = derive_teacher_tap_stack_sha256,
        stack_loader: StackLoader = load_offline_framesamp_am_stack_manifest,
        overlay_verifier: Callable[..., Mapping[str, Any]] = verify_framesamp_am_policy_overlay,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve(strict=True)
        if not self.artifact_root.is_dir():
            raise ValueError("FrameSamp-AM artifact root must be a directory")
        self.policy_overlay_root = Path(policy_overlay_root).resolve(strict=True)
        self.policy_overlay_manifest_sha256 = _require_sha(
            expected_policy_overlay_manifest_sha256,
            label="policy_overlay_manifest_sha256",
        )
        overlay = dict(
            overlay_verifier(
                self.policy_overlay_root,
                expected_manifest_sha256=self.policy_overlay_manifest_sha256,
            )
        )
        self.active_policy_checkpoint_sha256 = _require_sha(
            active_policy_checkpoint_sha256,
            label="active_policy_checkpoint_sha256",
        )
        self.expected_teacher_code_sha = _require_sha(
            expected_teacher_code_sha,
            label="expected_teacher_code_sha",
            lengths=(40, 64),
        )
        allowed_teacher_code = {
            overlay.get("official_policy_git_sha"),
            overlay.get("source_tree_sha256"),
            self.policy_overlay_manifest_sha256,
        }
        if self.expected_teacher_code_sha not in allowed_teacher_code:
            raise ValueError("expected teacher-code SHA is not bound to the verified policy overlay contract")
        self.active_model_dtype = jnp.dtype(active_model_dtype).name
        if jnp.dtype(invoker.model_dtype).name != self.active_model_dtype:
            raise ValueError("policy invoker dtype does not match the declared active model dtype")
        self.expected_device_platform = _require_nonempty(
            expected_device_platform,
            label="expected_device_platform",
        )
        self.device_or_sharding = device_or_sharding
        self.known_tasks = frozenset(_require_nonempty(task, label="known task") for task in known_tasks)
        if not self.known_tasks:
            raise ValueError("authenticated oracle server requires a nonempty task allowlist")
        self.history_attestor = history_attestor
        self.invoker = invoker
        self.oracle_resolver = oracle_resolver
        self.tap_digest_resolver = tap_digest_resolver
        self.stack_loader = stack_loader
        self.overlay_manifest = overlay

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "active_policy_checkpoint_sha256": self.active_policy_checkpoint_sha256,
            "expected_teacher_code_sha": self.expected_teacher_code_sha,
            "policy_overlay_manifest_sha256": self.policy_overlay_manifest_sha256,
            "policy_overlay_source_tree_sha256": self.overlay_manifest["source_tree_sha256"],
            "server_source_sha256": _sha256_file(Path(__file__)),
            "active_model_dtype": self.active_model_dtype,
            "device_platform": self.expected_device_platform,
            "action_horizon": ACTION_HORIZON,
            "execution_horizon": EXECUTION_HORIZON,
            "memory_partition": "compact_all_valid_framesamp_tokens_no_recent_v1",
            "history_attestation": "online_ordered_teacher_tap_stack_required",
        }

    def connection(self) -> "AuthenticatedFrameSampAMOracleConnection":
        return AuthenticatedFrameSampAMOracleConnection(self)


class AuthenticatedFrameSampAMOracleConnection:
    """One reset-bound episode with strictly advancing authenticated cuts."""

    def __init__(self, bridge: AuthenticatedFrameSampAMOracleBridge) -> None:
        self.bridge = bridge
        self.task_id: str | None = None
        self.episode_id: str | None = None
        self.last_causal_cut: int | None = None
        self.index_binding: tuple[str, str] | None = None
        self.closed = False

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed or self.task_id is not None:
            raise RuntimeError("one authenticated oracle connection permits exactly one reset")
        mismatches = {
            key: {"expected": expected, "actual": payload.get(key)}
            for key, expected in self.bridge.metadata.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"authenticated oracle reset contract mismatch: {mismatches}")
        task_id = _require_nonempty(payload.get("task_id"), label="task_id")
        episode_id = _require_nonempty(payload.get("episode_id"), label="episode_id")
        if task_id not in self.bridge.known_tasks:
            raise ValueError(f"unknown authenticated RoboMME task {task_id!r}")
        self.bridge.invoker.reset()
        self.task_id = task_id
        self.episode_id = episode_id
        return {"reset_finished": True, "protocol_id": PROTOCOL_ID}

    @staticmethod
    def _observation(payload: Mapping[str, Any]) -> dict[str, Any]:
        required = ("observation/image", "observation/wrist_image", "observation/state", "prompt")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"authenticated oracle inference is missing observation fields {missing}")
        front = np.asarray(payload["observation/image"])
        wrist = np.asarray(payload["observation/wrist_image"])
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        prompt = payload["prompt"]
        if front.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("authenticated oracle RGB observations must be uint8")
        if front.ndim != 3 or front.shape[-1] != 3 or wrist.shape != front.shape:
            raise ValueError("authenticated oracle front/wrist observations must share HWC RGB geometry")
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("authenticated oracle state must be finite shape (8,)")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("authenticated oracle prompt must be a nonempty string")
        return {
            "observation/image": front,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": prompt,
        }

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed or self.task_id is None or self.episode_id is None:
            raise RuntimeError("authenticated oracle inference arrived before reset")
        required_route = (
            "task_id",
            "episode_id",
            "causal_cut_step",
            "trusted_index_relative_path",
            "trusted_index_sha256",
            "stack_receipt_relative_path",
            "stack_receipt_sha256",
            "teacher_tap_stack_sha256",
        )
        missing = [name for name in required_route if name not in payload]
        if missing:
            raise ValueError(f"authenticated oracle inference is missing route fields {missing}")
        task_id = _require_nonempty(payload["task_id"], label="task_id")
        episode_id = _require_nonempty(payload["episode_id"], label="episode_id")
        if (task_id, episode_id) != (self.task_id, self.episode_id):
            raise ValueError("authenticated oracle inference task/episode differs from reset identity")
        causal_cut = _require_int(payload["causal_cut_step"], label="causal_cut_step")
        if self.last_causal_cut is not None and causal_cut <= self.last_causal_cut:
            raise ValueError(
                f"stale or repeated offline oracle cut {causal_cut}; last successful cut was {self.last_causal_cut}"
            )
        index_relative = _require_nonempty(
            payload["trusted_index_relative_path"],
            label="trusted_index_relative_path",
        )
        index_sha = _require_sha(payload["trusted_index_sha256"], label="trusted_index_sha256")
        index_binding = (index_relative, index_sha)
        if self.index_binding is not None and index_binding != self.index_binding:
            raise ValueError("trusted FrameSamp-AM index changed within one episode")
        index_path = _safe_receipt_file(
            self.bridge.artifact_root,
            index_relative,
            label="trusted_index_relative_path",
        )
        if _sha256_file(index_path) != index_sha:
            raise ValueError("trusted FrameSamp-AM index bytes do not match the request SHA")
        stack_path = _safe_receipt_file(
            self.bridge.artifact_root,
            payload["stack_receipt_relative_path"],
            label="stack_receipt_relative_path",
        )
        stack_sha = _require_sha(payload["stack_receipt_sha256"], label="stack_receipt_sha256")
        stack = self.bridge.stack_loader(stack_path, expected_sha256=stack_sha)
        stack_route = (
            stack.task_id,
            stack.episode_id,
            stack.causal_cut_step,
            stack.trusted_index_sha256,
            stack.teacher_checkpoint_sha256,
            stack.teacher_code_sha,
            stack.storage_dtype,
        )
        expected_route = (
            task_id,
            episode_id,
            causal_cut,
            index_sha,
            self.bridge.active_policy_checkpoint_sha256,
            self.bridge.expected_teacher_code_sha,
            self.bridge.active_model_dtype,
        )
        if stack_route != expected_route:
            raise ValueError(
                f"offline oracle stack receipt route mismatch: expected {expected_route!r}, got {stack_route!r}"
            )

        expected_tap_stack = self.bridge.tap_digest_resolver(
            index_path,
            expected_trusted_index_sha256=index_sha,
            stack=stack,
        )
        claimed_tap_stack = _require_sha(
            payload["teacher_tap_stack_sha256"],
            label="teacher_tap_stack_sha256",
        )
        if claimed_tap_stack != expected_tap_stack:
            raise ValueError("request teacher-tap stack SHA does not match the sealed per-layer artifacts")
        if self.bridge.history_attestor is None:
            raise RuntimeError(
                "offline oracle policy route is blocked without an online teacher-tap history attestor; "
                "task/episode/cut labels cannot prove on-policy history equality"
            )
        attested_tap_stack = _require_sha(
            self.bridge.history_attestor(task_id, episode_id, causal_cut, payload),
            label="attested_teacher_tap_stack_sha256",
        )
        if attested_tap_stack != expected_tap_stack:
            raise ValueError("actual on-policy history teacher taps do not match the offline oracle receipt")

        oracle = self.bridge.oracle_resolver(
            stack_path,
            expected_stack_manifest_sha256=stack_sha,
            trusted_index_path=index_path,
            expected_trusted_index_sha256=index_sha,
            active_policy_checkpoint_sha256=self.bridge.active_policy_checkpoint_sha256,
            active_model_dtype=self.bridge.active_model_dtype,
            expected_device_platform=self.bridge.expected_device_platform,
            device_or_sharding=self.bridge.device_or_sharding,
        )
        oracle.assert_request_identity(task_id=task_id, episode_id=episode_id, causal_cut_step=causal_cut)
        result = dict(self.bridge.invoker.infer(self._observation(payload), oracle.sample_actions_dynamic_inputs()))
        actions = np.asarray(result.get("actions"))
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"authenticated oracle action contract requires finite (20,8), got {actions.shape}")
        self.last_causal_cut = causal_cut
        self.index_binding = index_binding
        return {
            **result,
            "actions": actions.astype(np.float32, copy=False),
            "stack_receipt_sha256": stack_sha,
            "teacher_tap_stack_sha256": expected_tap_stack,
            "causal_cut_step": causal_cut,
        }

    def add_buffer(self, _payload: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("schema-v2 offline oracle is compact-all/R=0; server-side add_buffer is forbidden")

    def close(self) -> None:
        self.closed = True


async def serve_authenticated_oracle(
    bridge: AuthenticatedFrameSampAMOracleBridge,
    host: str,
    port: int,
) -> None:
    """Serve the authenticated bridge over the existing msgpack WebSocket shape."""

    import websockets
    import websockets.asyncio.server as websocket_server
    import websockets.frames
    from openpi_client import msgpack_numpy

    def health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def handler(websocket):
        connection = bridge.connection()
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(bridge.metadata))
        try:
            while True:
                payload = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(payload, dict):
                    raise ValueError("authenticated oracle request must be a dictionary")
                if payload.get("reset", False):
                    response = connection.reset(payload)
                elif payload.get("add_buffer", False):
                    response = connection.add_buffer(payload)
                else:
                    response = connection.infer(payload)
                await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            pass
        finally:
            connection.close()

    async with websocket_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        await server.serve_forever()
