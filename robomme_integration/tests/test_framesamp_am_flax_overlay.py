from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from robomme_integration.training.framesamp_am_flax_overlay import (
    HISTORY_GEMMA_RELATIVE_PATH,
    OFFICIAL_HISTORY_GEMMA_SHA256,
    OFFICIAL_HISTORY_PI0_SHA256,
    PATCHED_HISTORY_GEMMA_SHA256,
    stage_framesamp_am_flax_overlay,
    verify_framesamp_am_flax_overlay,
)
from robomme_integration.training.framesamp_am_jax import (
    memory_attention_am_core,
    require_reviewed_model_patch,
)
from wsm_settings import ROBOMME_EVAL_ROOT

OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "ROBOMME_OFFICIAL_POLICY_ROOT",
        str(ROBOMME_EVAL_ROOT / "official_reference" / "robomme_policy_learning"),
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_and_import(tmp_path: Path):
    if not (OFFICIAL_CHECKOUT / ".git").is_dir():
        pytest.skip("pinned RoboMME policy checkout is not available")
    official_gemma = OFFICIAL_CHECKOUT.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts)
    before = _sha256(official_gemma)
    destination = tmp_path / "framesamp_am_overlay"
    manifest_sha = stage_framesamp_am_flax_overlay(OFFICIAL_CHECKOUT, destination)
    manifest = verify_framesamp_am_flax_overlay(destination, expected_manifest_sha256=manifest_sha)
    assert _sha256(official_gemma) == before == OFFICIAL_HISTORY_GEMMA_SHA256
    assert manifest["patched_history_gemma_sha256"] == PATCHED_HISTORY_GEMMA_SHA256
    assert manifest["history_pi0_sha256"] == OFFICIAL_HISTORY_PI0_SHA256
    assert manifest["history_pi0_status"] == "unchanged_policy_artifact_route_required"

    module_path = destination.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts)
    require_reviewed_model_patch(module_path)
    spec = importlib.util.spec_from_file_location("staged_framesamp_am_history_gemma", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, destination, manifest_sha


def _tap(collection, name: str):
    values = collection["framesamp_am_taps"][name]
    assert isinstance(values, tuple) and len(values) == 1
    return values[0]


def test_real_memory_attention_same_vars_preserves_teacher_and_matches_am_core(tmp_path):
    """One real-module test covers source isolation, taps, masks, and AM math."""

    from mme_vla_suite.models.integration import history_gemma as official

    staged, staged_root, manifest_sha = _stage_and_import(tmp_path)
    batch, action_tokens, source_tokens, width = 1, 20, 512, 1024
    x = jax.random.normal(jax.random.key(2), (batch, action_tokens, width)).astype(jnp.bfloat16)
    memory = jax.random.normal(jax.random.key(3), (batch, source_tokens, width)).astype(jnp.bfloat16)
    # Exercise official right-padding: masked physical slots must not enter the
    # AM denominator even though K/V retain fixed shape 512.
    source_mask = jnp.arange(source_tokens)[None, :] < 160
    teacher = official.MemoryAttention()
    consumer = staged.AMMemoryAttention()
    variables = teacher.init(jax.random.key(4), x, memory, source_mask)

    teacher_output = teacher.apply(variables, x, memory, source_mask)
    patched_vanilla_output, taps = consumer.apply(
        variables,
        x,
        memory,
        source_mask,
        capture_framesamp_am_taps=True,
        mutable=["framesamp_am_taps"],
    )
    assert jnp.array_equal(teacher_output, patched_vanilla_output)
    assert set(variables) == {"params"}

    q = _tap(taps, "q_post_rope_pre_scale")
    full_k = _tap(taps, "recent_k_post_rope")
    full_v = _tap(taps, "recent_v_post_projection")
    empty_memory = jnp.empty((batch, 0, width), dtype=x.dtype)
    empty_mask = jnp.empty((batch, 0), dtype=jnp.bool_)
    empty_positions = jnp.empty((batch, 0), dtype=jnp.int32)
    zero_beta = jnp.zeros((batch, source_tokens), dtype=jnp.float32)

    # M=512, beta=0 and the original mask is a strict identity transform,
    # including the padded tail.  Compact K is not projected or RoPE'd again.
    identity_output = consumer.apply(
        variables,
        x,
        empty_memory,
        empty_mask,
        full_k,
        full_v,
        zero_beta,
        source_mask,
        empty_positions,
    )
    assert jnp.array_equal(teacher_output, identity_output)

    # A nontrivial compact-all payload agrees with the existing reviewed JAX
    # reference before the unchanged out projection.
    compact_k = full_k[:, :17]
    compact_v = full_v[:, :17]
    beta = jax.random.normal(jax.random.key(5), (batch, 17), dtype=jnp.float32) * 0.1
    compact_mask = jnp.ones((batch, 17), dtype=jnp.bool_)
    _, compact_taps = consumer.apply(
        variables,
        x,
        empty_memory,
        empty_mask,
        compact_k,
        compact_v,
        beta,
        compact_mask,
        empty_positions,
        True,
        mutable=["framesamp_am_taps"],
    )
    expected_encoded, _ = memory_attention_am_core(
        q,
        compact_k,
        compact_v,
        beta,
        jnp.empty((batch, 0, 1, 256), dtype=x.dtype),
        jnp.empty((batch, 0, 1, 256), dtype=x.dtype),
        jnp.empty((batch, 0), dtype=jnp.bool_),
        scale=256**-0.5,
        query_position_offset=512,
    )
    assert jnp.array_equal(_tap(compact_taps, "encoded_pre_out"), expected_encoded)

    unrelated_source = staged_root / "src/mme_vla_suite/__init__.py"
    unrelated_source.write_bytes(unrelated_source.read_bytes() + b"# drift\n")
    with pytest.raises(ValueError, match="source tree SHA mismatch"):
        verify_framesamp_am_flax_overlay(staged_root, expected_manifest_sha256=manifest_sha)


def test_patched_memory_attention_preserves_released_q_scale_k_rope_order(tmp_path):
    _, staged_root, _ = _stage_and_import(tmp_path)
    source = staged_root.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts).read_text(encoding="utf-8")
    start = source.index("class AMMemoryAttention")
    block = source[start : source.index("\n\n@at.typecheck", start)]
    q_rope = block.index("q = _apply_rope(q, positions=q_positions)")
    q_scale = block.index("q *= head_dim**-0.5")
    k_rope = block.index("k = _apply_rope(k, positions=k_positions)")
    assert q_rope < q_scale < k_rope


def test_scanned_module_keeps_baseline_exact_and_routes_layer_axis(tmp_path, monkeypatch):
    """Exercise the actual Linen scan arity and a full-memory AM identity route."""

    from mme_vla_suite.models.integration import history_gemma as official
    from openpi.models.gemma import Config

    staged, _, _ = _stage_and_import(tmp_path)
    # Avoid constructing the unused 257k-token embedder in this focused module
    # test; no checkpoint parameter exercised below depends on its vocabulary.
    monkeypatch.setattr(official, "PALIGEMMA_VOCAB_SIZE", 8)
    monkeypatch.setattr(staged, "PALIGEMMA_VOCAB_SIZE", 8)
    configs = (
        Config(
            width=1024,
            depth=2,
            mlp_dim=32,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
        ),
    ) * 2

    class Harness(nn.Module):
        module_class: object
        capture: bool = False

        @nn.compact
        def __call__(self, embedded, positions, mask, mem_seq, mem_mask, **am):
            if self.capture:
                am["capture_framesamp_am_taps"] = True
            return self.module_class(
                configs=configs,
                embed_dtype="bfloat16",
                integration_type="modulation",
                name="llm",
            )(
                embedded,
                positions,
                mask,
                mem_seq=mem_seq,
                mem_mask=mem_mask,
                **am,
            )

    batch, source_tokens = 1, 512
    embedded = [
        jax.random.normal(jax.random.key(10), (batch, 2, 1024)).astype(jnp.bfloat16),
        jax.random.normal(jax.random.key(11), (batch, 20, 1024)).astype(jnp.bfloat16),
    ]
    positions = jnp.arange(22, dtype=jnp.int32)[None]
    mask = jnp.ones((batch, 22, 22), dtype=jnp.bool_)
    memory = jax.random.normal(jax.random.key(12), (batch, source_tokens, 1024)).astype(jnp.bfloat16)
    memory_mask = jnp.arange(source_tokens)[None, :] < 160
    mem_seq = [None, memory]
    mem_mask = [None, memory_mask]

    teacher = Harness(official.Module)
    patched = Harness(staged.Module)
    variables = teacher.init(jax.random.key(13), embedded, positions, mask, mem_seq, mem_mask)
    teacher_output = teacher.apply(variables, embedded, positions, mask, mem_seq, mem_mask)
    patched_output = patched.apply(variables, embedded, positions, mask, mem_seq, mem_mask)
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, teacher_output, patched_output))

    capture = Harness(staged.Module, capture=True)
    capture_output, scan_taps = capture.apply(
        variables,
        embedded,
        positions,
        mask,
        mem_seq,
        mem_mask,
        mutable=["framesamp_am_taps"],
    )
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, teacher_output, capture_output))
    taps = scan_taps["framesamp_am_taps"]["llm"]["layers"]["mem_attn"]
    compact_k = taps["recent_k_post_rope"][0]
    compact_v = taps["recent_v_post_projection"][0]
    # The lifted collection and runtime compact tensors both carry the layer
    # axis first, even for this one-layer realistic scan.
    assert compact_k.shape == (2, batch, source_tokens, 1, 256)
    assert compact_v.shape == compact_k.shape
    assert not jnp.array_equal(compact_k[0], compact_k[1])
    am = {
        "framesamp_am_compact_k": compact_k,
        "framesamp_am_compact_v": compact_v,
        "framesamp_am_compact_beta": jnp.zeros((2, batch, source_tokens), jnp.float32),
        "framesamp_am_compact_mask": jnp.broadcast_to(memory_mask, (2, batch, source_tokens)),
        "framesamp_am_recent_positions": jnp.empty((batch, 0), jnp.int32),
    }
    compact_output = patched.apply(
        variables,
        embedded,
        positions,
        mask,
        [None, jnp.empty((batch, 0, 1024), jnp.bfloat16)],
        [None, jnp.empty((batch, 0), jnp.bool_)],
        **am,
    )
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, teacher_output, compact_output))
