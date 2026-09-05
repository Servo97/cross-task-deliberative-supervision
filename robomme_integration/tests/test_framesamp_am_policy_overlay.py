from __future__ import annotations

import ast
import importlib.util
import os
import types
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from openpi.shared import nnx_utils

from robomme_integration.training.framesamp_am_flax_overlay import (
    HISTORY_PI0_RELATIVE_PATH,
    PATCHED_HISTORY_GEMMA_SHA256,
)
from robomme_integration.training.framesamp_am_oracle_route import (
    OfflineFrameSampAMOracleInputs,
)
from robomme_integration.training.framesamp_am_policy_overlay import (
    MEMORY_PARTITION_KIND,
    PATCHED_HISTORY_PI0_SHA256,
    stage_framesamp_am_policy_overlay,
    verify_framesamp_am_policy_overlay,
)
from wsm_settings import ROBOMME_EVAL_ROOT

OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "ROBOMME_OFFICIAL_POLICY_ROOT",
        str(ROBOMME_EVAL_ROOT / "official_reference" / "robomme_policy_learning"),
    )
)


def _function_ast(source: str, class_name: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.dump(child, include_attributes=False)
    raise AssertionError(f"missing {class_name}.{function_name}")


class _FakeLLM(nnx.Module):
    def __call__(self, embedded, **kwargs):
        suffix = embedded[-1]
        if suffix is None:
            return embedded, jnp.array(0, dtype=jnp.int32)
        compact_v = kwargs.get("framesamp_am_compact_v")
        if compact_v is None:
            marker = jnp.asarray(0, dtype=suffix.dtype)
        else:
            mask = kwargs["framesamp_am_compact_mask"][..., None, None]
            weighted = jnp.where(mask, compact_v, jnp.zeros_like(compact_v))
            marker = jnp.mean(weighted.astype(jnp.float32)).astype(suffix.dtype)
        return [None, suffix + marker], kwargs.get("kv_cache")


class _IdentityProjection(nnx.Module):
    def __call__(self, value):
        return value


def _minimal_model(sample_method):
    class MinimalHistoryPi0(nnx.Module):
        def __init__(self):
            self.integration_type = "modulation"
            self.action_horizon = 20
            self.action_dim = 2
            self.config = types.SimpleNamespace(
                action_expert_variant="gemma_300m",
                dtype="bfloat16",
            )
            self.PaliGemma = nnx.Dict(llm=_FakeLLM())
            self.action_out_proj = _IdentityProjection()

        def embed_prefix(self, observation):
            batch = observation.state.shape[0]
            return (
                jnp.zeros((batch, 2, 2), dtype=jnp.bfloat16),
                jnp.ones((batch, 2), dtype=jnp.bool_),
                jnp.array([True, False]),
                jnp.array([False, False]),
                None,
            )

        def embed_suffix(self, observation, noisy_actions, timestep):
            del observation, timestep
            batch = noisy_actions.shape[0]
            return (
                noisy_actions,
                jnp.ones((batch, 20), dtype=jnp.bool_),
                jnp.array([True] + [False] * 19),
                jnp.array([False] * 20),
                None,
            )

        def embed_memory(self, observation):
            batch = observation.state.shape[0]
            return (
                jnp.zeros((batch, 4, 1024), dtype=jnp.bfloat16),
                jnp.ones((batch, 4), dtype=jnp.bool_),
                None,
                None,
                None,
            )

    MinimalHistoryPi0.sample_actions = sample_method
    return MinimalHistoryPi0()


def _dynamic_inputs(value: float) -> dict[str, jax.Array]:
    layers, batch, compact_tokens = 18, 1, 2
    return {
        "framesamp_am_compact_k": jnp.ones((layers, batch, compact_tokens, 1, 256), dtype=jnp.bfloat16),
        "framesamp_am_compact_v": jnp.full((layers, batch, compact_tokens, 1, 256), value, dtype=jnp.bfloat16),
        "framesamp_am_compact_beta": jnp.zeros((layers, batch, compact_tokens), dtype=jnp.float32),
        "framesamp_am_compact_mask": jnp.ones((layers, batch, compact_tokens), dtype=jnp.bool_),
        "framesamp_am_recent_positions": jnp.empty((batch, 0), dtype=jnp.int32),
        "framesamp_am_recent_mem_seq": jnp.empty((batch, 0, 1024), dtype=jnp.bfloat16),
        "framesamp_am_recent_mem_mask": jnp.empty((batch, 0), dtype=jnp.bool_),
    }


def test_evaluation_policy_overlay_is_dynamic_after_module_jit_and_training_stays_closed(
    tmp_path,
    monkeypatch,
):
    if not (OFFICIAL_CHECKOUT / ".git").is_dir():
        pytest.skip("pinned RoboMME policy checkout is not available")
    destination = tmp_path / "policy_overlay"
    manifest_sha = stage_framesamp_am_policy_overlay(OFFICIAL_CHECKOUT, destination)
    manifest = verify_framesamp_am_policy_overlay(destination, expected_manifest_sha256=manifest_sha)
    assert manifest["patched_history_gemma_sha256"] == PATCHED_HISTORY_GEMMA_SHA256
    assert manifest["patched_history_pi0_sha256"] == PATCHED_HISTORY_PI0_SHA256
    assert manifest["memory_partition_kind"] == MEMORY_PARTITION_KIND
    assert manifest["training_compute_loss_status"] == "unchanged_not_implemented"
    assert manifest["policy_server_status"] == ("blocked_missing_authoritative_task_episode_cut_binding")
    assert "MME_VLA_Policy.infer" in manifest["policy_server_blocker"]

    official_path = OFFICIAL_CHECKOUT.joinpath(*HISTORY_PI0_RELATIVE_PATH.parts)
    staged_path = destination.joinpath(*HISTORY_PI0_RELATIVE_PATH.parts)
    official_source = official_path.read_text(encoding="utf-8")
    staged_source = staged_path.read_text(encoding="utf-8")
    assert _function_ast(official_source, "HistoryPi0", "compute_loss") == _function_ast(
        staged_source, "HistoryPi0", "compute_loss"
    )

    from mme_vla_suite.models.integration import history_pi0 as official

    spec = importlib.util.spec_from_file_location("staged_framesamp_am_history_pi0", staged_path)
    assert spec is not None and spec.loader is not None
    staged = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged)
    monkeypatch.setattr(official, "preprocess_observation", lambda rng, obs, train=False: obs)
    monkeypatch.setattr(staged, "preprocess_observation", lambda rng, obs, train=False: obs)

    observation = HistAugObservation(
        images={},
        image_masks={},
        state=jnp.zeros((1, 2), dtype=jnp.float32),
    )
    noise = jnp.ones((1, 20, 2), dtype=jnp.float32)
    official_jit = nnx_utils.module_jit(_minimal_model(official.HistoryPi0.sample_actions).sample_actions)
    staged_jit = nnx_utils.module_jit(_minimal_model(staged.HistoryPi0.sample_actions).sample_actions)

    routed_arrays = _dynamic_inputs(1.0)
    routed = OfflineFrameSampAMOracleInputs(
        stack_manifest_sha256="a" * 64,
        trusted_index_sha256="b" * 64,
        teacher_checkpoint_sha256="c" * 64,
        task_id="task_00",
        episode_id="episode_00",
        causal_cut_step=0,
        requested_budget=2,
        model_dtype="bfloat16",
        device_platform="cpu",
        **routed_arrays,
    )
    assert set(routed.sample_actions_dynamic_inputs()) == set(routed_arrays)

    official_baseline = official_jit(jax.random.key(1), observation, num_steps=1, noise=noise)
    staged_baseline = staged_jit(jax.random.key(1), observation, num_steps=1, noise=noise)
    assert jnp.array_equal(official_baseline, staged_baseline)

    first = staged_jit(
        jax.random.key(1),
        observation,
        num_steps=1,
        noise=noise,
        **routed.sample_actions_dynamic_inputs(),
    )
    second = staged_jit(
        jax.random.key(1),
        observation,
        num_steps=1,
        noise=noise,
        **_dynamic_inputs(2.0),
    )
    assert jnp.array_equal(first, -jnp.ones_like(first))
    assert jnp.array_equal(second, -2 * jnp.ones_like(second))
    assert not jnp.array_equal(first, second)

    partial = _dynamic_inputs(1.0)
    partial.pop("framesamp_am_compact_beta")
    with pytest.raises(ValueError, match="all seven dynamic"):
        staged_jit(
            jax.random.key(1),
            observation,
            num_steps=1,
            noise=noise,
            **partial,
        )

    wrong_integration = _minimal_model(staged.HistoryPi0.sample_actions)
    wrong_integration.integration_type = None
    wrong_integration_jit = nnx_utils.module_jit(wrong_integration.sample_actions)
    with pytest.raises(ValueError, match="requires modulation integration"):
        wrong_integration_jit(
            jax.random.key(1),
            observation,
            num_steps=1,
            noise=noise,
            **routed.sample_actions_dynamic_inputs(),
        )
