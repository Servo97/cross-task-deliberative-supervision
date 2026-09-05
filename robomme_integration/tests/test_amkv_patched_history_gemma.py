"""The AM patch must be the official forward until an artifact is supplied."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mme_vla_suite.models.integration import history_gemma as official
from openpi.models.gemma import Config

from robomme_integration.amkv.patched_history_gemma import (
    MEMORY_HEAD_DIM,
    MemoryAttentionAMPack,
    ModuleAM,
    patched_module_like,
)
from robomme_integration.training.upstream_framesamp_data import TOKEN_BUDGET

BATCH = 1
SUFFIX_TOKENS = 6
DEPTH = 2
PREFIX = Config(width=256, depth=DEPTH, mlp_dim=512, num_heads=4, num_kv_heads=1, head_dim=64)
EXPERT = Config(width=1024, depth=DEPTH, mlp_dim=512, num_heads=4, num_kv_heads=1, head_dim=64)
CONFIGS = [PREFIX, EXPERT]


def _module(cls, **kwargs):
    return cls(configs=CONFIGS, embed_dtype="float32", adarms=True, integration_type="modulation", **kwargs)


def _live_params(variables):
    """Make every branch load-bearing.

    pi0.5 adaRMS initialises its residual gates to zero, so a freshly
    initialised block is the identity and is *completely insensitive to the
    memory*: parity assertions against it would be vacuous.  Perturbing every
    leaf reproduces a trained model's live gates.
    """

    leaves, treedef = jax.tree.flatten(variables)
    keys = jax.random.split(jax.random.key(7), len(leaves))
    perturbed = [
        leaf + 0.1 * jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
        for leaf, key in zip(leaves, keys, strict=True)
    ]
    return jax.tree.unflatten(treedef, perturbed)


@pytest.fixture(scope="module")
def fixture():
    keys = jax.random.split(jax.random.key(0), 3)
    suffix = jax.random.normal(keys[0], (BATCH, SUFFIX_TOKENS, EXPERT.width))
    mem_seq = jax.random.normal(keys[1], (BATCH, TOKEN_BUDGET, EXPERT.width))
    adarms = jax.random.normal(keys[2], (BATCH, EXPERT.width))
    args = (
        [None, suffix],
        jnp.broadcast_to(jnp.arange(SUFFIX_TOKENS), (BATCH, SUFFIX_TOKENS)),
        jnp.ones((BATCH, SUFFIX_TOKENS, SUFFIX_TOKENS), dtype=bool),
        [None, adarms],
    )
    kwargs = {
        "kv_cache": None,
        "mem_seq": [None, mem_seq],
        "mem_mask": [None, jnp.ones((BATCH, TOKEN_BUDGET), dtype=bool)],
    }
    module = _module(official.Module)
    variables = _live_params(nn.Module.init(module, jax.random.key(1), *args, **kwargs))
    reference, _ = module.apply(variables, *args, **kwargs)
    perturbed, _ = module.apply(
        variables,
        args[0],
        args[1],
        args[2],
        args[3],
        kv_cache=None,
        mem_seq=[None, mem_seq * 3.0],
        mem_mask=kwargs["mem_mask"],
    )
    # Guard the guard: if the block were memory-insensitive every parity
    # assertion below would pass for the wrong reason.
    assert not np.array_equal(np.asarray(perturbed[1]), np.asarray(reference[1]))
    return args, kwargs, variables, np.asarray(reference[1])


def _capture_taps(fixture):
    args, kwargs, variables, _ = fixture
    _, _, taps = _module(ModuleAM, capture=True).apply(variables, *args, **kwargs)
    return taps


def _pack(taps, *, compact_stop: int):
    """Full teacher K/V split into an uncompressed compact block plus exact recent."""

    keys = taps.keys_post_rope[:, :, :compact_stop, :, :]
    values = taps.values_post_projection[:, :, :compact_stop, :, :]
    recent_keys = taps.keys_post_rope[:, :, compact_stop:, :, :]
    recent_values = taps.values_post_projection[:, :, compact_stop:, :, :]
    return MemoryAttentionAMPack(
        compact_keys=keys,
        compact_values=values,
        compact_beta_am=jnp.zeros(keys.shape[:3], dtype=jnp.float32)[:, :, :],
        recent_keys=recent_keys,
        recent_values=recent_values,
        recent_token_mask=jnp.ones(recent_keys.shape[:3], dtype=bool),
    )


def test_patched_module_reproduces_the_official_forward_bitwise(fixture):
    args, kwargs, variables, reference = fixture
    patched, _ = _module(ModuleAM).apply(variables, *args, **kwargs)
    assert np.array_equal(np.asarray(patched[1]), reference)


def test_tap_capture_does_not_perturb_the_forward(fixture):
    args, kwargs, variables, reference = fixture
    outputs, _, taps = _module(ModuleAM, capture=True).apply(variables, *args, **kwargs)
    assert np.array_equal(np.asarray(outputs[1]), reference)
    assert taps.queries_post_rope_pre_scale.shape == (DEPTH, BATCH, SUFFIX_TOKENS, 4, MEMORY_HEAD_DIM)
    assert taps.keys_post_rope.shape == (DEPTH, BATCH, TOKEN_BUDGET, 1, MEMORY_HEAD_DIM)
    assert taps.values_post_projection.shape == taps.keys_post_rope.shape
    assert np.isfinite(np.asarray(taps.queries_post_rope_pre_scale)).all()


def test_uncompressed_am_pack_reproduces_the_full_cache_bitwise(fixture):
    args, kwargs, variables, reference = fixture
    taps = _capture_taps(fixture)
    outputs, _ = _module(ModuleAM).apply(variables, *args, **kwargs, am_pack=_pack(taps, compact_stop=TOKEN_BUDGET))
    assert np.array_equal(np.asarray(outputs[1]), reference)


def test_same_denominator_contract_holds_on_real_shapes(fixture):
    """Compact-old + exact-recent under one softmax equals the full cache."""

    args, kwargs, variables, reference = fixture
    taps = _capture_taps(fixture)
    for compact_stop in (TOKEN_BUDGET - 16, TOKEN_BUDGET - 128, 256):
        outputs, _ = _module(ModuleAM).apply(
            variables, *args, **kwargs, am_pack=_pack(taps, compact_stop=compact_stop)
        )
        assert np.array_equal(np.asarray(outputs[1]), reference), compact_stop


def test_compaction_actually_changes_the_forward(fixture):
    args, kwargs, variables, reference = fixture
    taps = _capture_taps(fixture)
    keep = 64
    pack = MemoryAttentionAMPack(
        compact_keys=taps.keys_post_rope[:, :, :keep, :, :],
        compact_values=taps.values_post_projection[:, :, :keep, :, :],
        compact_beta_am=jnp.zeros((DEPTH, BATCH, keep), dtype=jnp.float32),
        recent_keys=jnp.zeros((DEPTH, BATCH, 0, 1, MEMORY_HEAD_DIM), dtype=taps.keys_post_rope.dtype),
        recent_values=jnp.zeros((DEPTH, BATCH, 0, 1, MEMORY_HEAD_DIM), dtype=taps.keys_post_rope.dtype),
        recent_token_mask=jnp.zeros((DEPTH, BATCH, 0), dtype=bool),
    )
    outputs, _ = _module(ModuleAM).apply(variables, *args, **kwargs, am_pack=pack)
    assert not np.array_equal(np.asarray(outputs[1]), reference)


def test_capture_and_am_pack_cannot_be_combined(fixture):
    args, kwargs, variables, _ = fixture
    taps = _capture_taps(fixture)
    with pytest.raises(ValueError, match="tap capture"):
        _module(ModuleAM, capture=True).apply(
            variables, *args, **kwargs, am_pack=_pack(taps, compact_stop=TOKEN_BUDGET)
        )


def test_patched_module_like_copies_the_official_settings():
    official_module = _module(official.Module)
    patched = patched_module_like(official_module, capture=True)
    assert isinstance(patched, ModuleAM)
    assert patched.capture is True
    assert patched.integration_type == official_module.integration_type
    assert patched.embed_dtype == official_module.embed_dtype
    assert list(patched.configs) == list(official_module.configs)


def test_patched_module_like_rejects_a_foreign_module():
    with pytest.raises(ValueError, match="patch contract drifted"):
        patched_module_like(object())
