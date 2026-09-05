from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robomme_integration.eval.framesamp_am_oracle_server import (
    AuthenticatedFrameSampAMOracleBridge,
    AuthenticatedOracleEvaluatorRoute,
    AuthenticatedOracleReplanReceipt,
)
from robomme_integration.training.attention_matching import (
    ARTIFACT_METHOD,
    MASS_SOLVER_DISABLED,
    VALUE_SOLVER,
)
from robomme_integration.training.framesamp_am_artifact import (
    KEY_TAP_STAGE,
    MEMORY_PARTITION_KIND,
    QUERY_TAP_STAGE,
    VALUE_TAP_STAGE,
)
from robomme_integration.training.framesamp_am_oracle_route import (
    OfflineFrameSampAMLayerPin,
    OfflineFrameSampAMOracleInputs,
    OfflineFrameSampAMStackManifest,
    load_offline_framesamp_am_stack_manifest,
)

CHECKPOINT_SHA = "a" * 64
TEACHER_CODE_SHA = "b" * 40
OVERLAY_SHA = "c" * 64
OVERLAY_TREE_SHA = "d" * 64
TASK_ID = "PickXtimes"
EPISODE_ID = "seed_3_episode_4"
M = 2


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stack(root, *, cut: int, index_sha: str):
    pins = tuple(
        OfflineFrameSampAMLayerPin(layer_index=layer, manifest_sha256=f"{layer + 1:064x}") for layer in range(18)
    )
    manifest = OfflineFrameSampAMStackManifest(
        trusted_index_sha256=index_sha,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        teacher_code_sha=TEACHER_CODE_SHA,
        task_id=TASK_ID,
        episode_id=EPISODE_ID,
        causal_cut_step=cut,
        requested_budget=M,
        storage_dtype="float32",
        resolved_attention_scale=256**-0.5,
        memory_partition_kind=MEMORY_PARTITION_KIND,
        artifact_method=ARTIFACT_METHOD,
        fit_mass=False,
        mass_solver=MASS_SOLVER_DISABLED,
        value_solver=VALUE_SOLVER,
        mass_ridge=0.0,
        value_ridge=1e-6,
        fit_queries_per_head=4,
        heldout_queries_per_head=2,
        payload_encoding="native_float32",
        query_tap_stage=QUERY_TAP_STAGE,
        key_tap_stage=KEY_TAP_STAGE,
        value_tap_stage=VALUE_TAP_STAGE,
        token_mask_sha256="e" * 64,
        frame_map_sha256="f" * 64,
        valid_source_tokens=64,
        layer_pins=pins,
    )
    path = root / f"stack_cut_{cut:04d}.json"
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path, _sha(path), manifest


class _CompiledInvoker:
    model_dtype = "float32"

    def __init__(self):
        self.compiled = jax.jit(lambda compact_v: jnp.ones((20, 8), dtype=jnp.float32) * jnp.mean(compact_v))
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, observation, dynamic_inputs):
        shapes = {name: value.shape for name, value in dynamic_inputs.items()}
        self.calls.append((observation["prompt"], shapes))
        return {"actions": np.asarray(self.compiled(dynamic_inputs["framesamp_am_compact_v"]))}


def _oracle(cut, *, stack_sha, index_sha, cpu):
    marker = jnp.asarray(float(cut), dtype=jnp.float32)
    compact = jax.device_put(jnp.ones((18, 1, M, 1, 256), dtype=jnp.float32) * marker, cpu)
    return OfflineFrameSampAMOracleInputs(
        stack_manifest_sha256=stack_sha,
        trusted_index_sha256=index_sha,
        teacher_checkpoint_sha256=CHECKPOINT_SHA,
        task_id=TASK_ID,
        episode_id=EPISODE_ID,
        causal_cut_step=cut,
        requested_budget=M,
        model_dtype="float32",
        device_platform="cpu",
        framesamp_am_compact_k=compact,
        framesamp_am_compact_v=compact,
        framesamp_am_compact_beta=jax.device_put(jnp.zeros((18, 1, M), dtype=jnp.float32), cpu),
        framesamp_am_compact_mask=jax.device_put(jnp.ones((18, 1, M), dtype=jnp.bool_), cpu),
        framesamp_am_recent_positions=jax.device_put(jnp.empty((1, 0), dtype=jnp.int32), cpu),
        framesamp_am_recent_mem_seq=jax.device_put(jnp.empty((1, 0, 1024), dtype=jnp.float32), cpu),
        framesamp_am_recent_mem_mask=jax.device_put(jnp.empty((1, 0), dtype=jnp.bool_), cpu),
    )


@pytest.fixture
def route_case(tmp_path):
    artifact_root = tmp_path / "artifacts"
    overlay_root = tmp_path / "overlay"
    artifact_root.mkdir()
    overlay_root.mkdir()
    index_path = artifact_root / "trusted_index.json"
    index_path.write_text("sealed test index\n", encoding="utf-8")
    index_sha = _sha(index_path)
    cuts = (5, 21)
    stacks = {cut: _write_stack(artifact_root, cut=cut, index_sha=index_sha) for cut in cuts}
    taps = {cut: hashlib.sha256(f"teacher-taps-at-cut-{cut}".encode()).hexdigest() for cut in cuts}
    cpu = jax.devices("cpu")[0]
    invoker = _CompiledInvoker()

    def overlay_verifier(_root, *, expected_manifest_sha256):
        assert expected_manifest_sha256 == OVERLAY_SHA
        return {
            "official_policy_git_sha": TEACHER_CODE_SHA,
            "source_tree_sha256": OVERLAY_TREE_SHA,
        }

    def stack_loader(path, *, expected_sha256):
        return load_offline_framesamp_am_stack_manifest(path, expected_sha256=expected_sha256)

    def tap_resolver(_index_path, *, expected_trusted_index_sha256, stack):
        assert expected_trusted_index_sha256 == index_sha
        return taps[stack.causal_cut_step]

    def oracle_resolver(stack_path, **kwargs):
        stack = stack_loader(stack_path, expected_sha256=kwargs["expected_stack_manifest_sha256"])
        assert kwargs["expected_trusted_index_sha256"] == index_sha
        assert kwargs["active_policy_checkpoint_sha256"] == CHECKPOINT_SHA
        return _oracle(
            stack.causal_cut_step,
            stack_sha=kwargs["expected_stack_manifest_sha256"],
            index_sha=index_sha,
            cpu=cpu,
        )

    def bridge(*, attestor=lambda _task, _episode, cut, _payload: taps[cut], checkpoint=CHECKPOINT_SHA):
        return AuthenticatedFrameSampAMOracleBridge(
            invoker,
            artifact_root=artifact_root,
            policy_overlay_root=overlay_root,
            expected_policy_overlay_manifest_sha256=OVERLAY_SHA,
            active_policy_checkpoint_sha256=checkpoint,
            expected_teacher_code_sha=TEACHER_CODE_SHA,
            active_model_dtype="float32",
            expected_device_platform="cpu",
            device_or_sharding=cpu,
            known_tasks=(TASK_ID,),
            history_attestor=attestor,
            oracle_resolver=oracle_resolver,
            tap_digest_resolver=tap_resolver,
            stack_loader=stack_loader,
            overlay_verifier=overlay_verifier,
        )

    receipts = tuple(
        AuthenticatedOracleReplanReceipt(
            causal_cut_step=cut,
            trusted_index_relative_path=index_path.relative_to(artifact_root).as_posix(),
            trusted_index_sha256=index_sha,
            stack_receipt_relative_path=stacks[cut][0].relative_to(artifact_root).as_posix(),
            stack_receipt_sha256=stacks[cut][1],
            teacher_tap_stack_sha256=taps[cut],
        )
        for cut in cuts
    )
    evaluator = AuthenticatedOracleEvaluatorRoute(task_id=TASK_ID, episode_id=EPISODE_ID, replans=receipts)
    return artifact_root, index_path, invoker, bridge, evaluator, taps


def _observation(prompt="this prompt is deliberately not a route identity"):
    return {
        "observation/image": np.zeros((64, 64, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((64, 64, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": prompt,
    }


def test_authenticated_route_reuses_one_compiled_policy_with_different_same_shape_cut_arrays(route_case):
    _artifact_root, _index_path, invoker, bridge_factory, evaluator, _taps = route_case
    bridge = bridge_factory()
    connection = bridge.connection()
    assert connection.reset(evaluator.reset_payload(bridge.metadata))["reset_finished"] is True
    compiled_identity = id(invoker.compiled)

    first = connection.infer(evaluator.inference_payload(5, _observation("mentions a completely different task")))
    second = connection.infer(evaluator.inference_payload(21, _observation("same prompt cannot select a receipt")))
    assert first["actions"].shape == second["actions"].shape == (20, 8)
    assert not np.array_equal(first["actions"], second["actions"])
    assert id(invoker.compiled) == compiled_identity
    assert len(invoker.calls) == 2 and invoker.resets == 1
    assert invoker.calls[0][1] == invoker.calls[1][1]
    assert first["causal_cut_step"] == 5 and second["causal_cut_step"] == 21


def test_missing_stale_mismatched_or_unattested_routes_fail_before_policy(route_case):
    artifact_root, index_path, invoker, bridge_factory, evaluator, taps = route_case

    connection = bridge_factory().connection()
    connection.reset(evaluator.reset_payload(connection.bridge.metadata))
    missing = evaluator.inference_payload(5, _observation())
    missing.pop("task_id")
    with pytest.raises(ValueError, match="missing route fields"):
        connection.infer(missing)
    connection.infer(evaluator.inference_payload(5, _observation()))
    calls_after_success = len(invoker.calls)
    with pytest.raises(ValueError, match="stale or repeated"):
        connection.infer(evaluator.inference_payload(5, _observation()))
    assert len(invoker.calls) == calls_after_success

    mismatched_cut = evaluator.inference_payload(5, _observation())
    mismatched_cut["causal_cut_step"] = 6
    fresh = bridge_factory().connection()
    fresh.reset(evaluator.reset_payload(fresh.bridge.metadata))
    with pytest.raises(ValueError, match="stack receipt route mismatch"):
        fresh.infer(mismatched_cut)

    unattested = bridge_factory(attestor=None).connection()
    unattested.reset(evaluator.reset_payload(unattested.bridge.metadata))
    with pytest.raises(RuntimeError, match="without an online teacher-tap history attestor"):
        unattested.infer(evaluator.inference_payload(5, _observation()))

    diverged = bridge_factory(attestor=lambda *_args: "9" * 64).connection()
    diverged.reset(evaluator.reset_payload(diverged.bridge.metadata))
    with pytest.raises(ValueError, match="actual on-policy history teacher taps"):
        diverged.infer(evaluator.inference_payload(5, _observation()))

    original = index_path.read_bytes()
    index_path.write_bytes(original + b"drift")
    drifted = bridge_factory().connection()
    drifted.reset(evaluator.reset_payload(drifted.bridge.metadata))
    with pytest.raises(ValueError, match="index bytes"):
        drifted.infer(evaluator.inference_payload(5, _observation()))
    index_path.write_bytes(original)
    assert index_path.parent == artifact_root
    assert taps[5] != taps[21]


def test_overlay_teacher_code_and_active_checkpoint_are_independent_hard_gates(route_case):
    _artifact_root, _index_path, _invoker, bridge_factory, evaluator, _taps = route_case
    wrong_checkpoint = bridge_factory(checkpoint="6" * 64)
    connection = wrong_checkpoint.connection()
    connection.reset(evaluator.reset_payload(wrong_checkpoint.metadata))
    with pytest.raises(ValueError, match="stack receipt route mismatch"):
        connection.infer(evaluator.inference_payload(5, _observation()))

    route = AuthenticatedOracleEvaluatorRoute(
        task_id=TASK_ID,
        episode_id=EPISODE_ID,
        replans=tuple(reversed(evaluator.replans)),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        route.validate()
