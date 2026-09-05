"""CPU-JAX correctness tests for the frozen pi0.5 prefix tap fast path."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
OPENPI_SRC = Path(__file__).resolve().parents[3] / "robocasa_openpi" / "src"
sys.path.insert(0, str(OPENPI_SRC))

from openpi.models.pi0 import Pi0

from workspace_models.features.pi_backbone_tap import (
    N_IMG_TOK,
    N_VIEWS,
    SLOT_TO_LABEL,
    _bin8x8,
    _make_jax_postprocessor,
)


class _FakeImage:
    def __call__(self, value, *, train):
        assert train is False
        batch = value.shape[0]
        return jnp.ones((batch, 2, 4), dtype=jnp.float32), None


class _FakeLLM:
    def __call__(self, value, *, method=None, **_kwargs):
        if method == "embed":
            return jnp.full((*value.shape, 4), 2.0, dtype=jnp.float32)
        prefix, suffix = value
        assert suffix is None
        return (prefix + 1.0, None), None


def test_prefix_mask_is_concatenated_once_and_hidden_tap_is_read_only():
    batch = 2
    fake = SimpleNamespace(
        PaliGemma=SimpleNamespace(img=_FakeImage(), llm=_FakeLLM()),
        wsm=False,
    )
    observation = SimpleNamespace(
        images={
            "base": jnp.zeros((batch, 1), dtype=jnp.float32),
            "left": jnp.zeros((batch, 1), dtype=jnp.float32),
            "right": jnp.zeros((batch, 1), dtype=jnp.float32),
        },
        image_masks={
            "base": jnp.array([True, True]),
            "left": jnp.array([True, False]),
            "right": jnp.array([True, True]),
        },
        tokenized_prompt=jnp.zeros((batch, 3), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.array([[True, True, False], [True, False, False]]),
        wsm_w_window=None,
        wsm_lang=None,
    )
    embed_prefix = inspect.unwrap(Pi0.embed_prefix)
    tokens, mask, autoregressive = embed_prefix(fake, observation)
    assert tokens.shape == (batch, 9, 4)
    assert mask.shape == (batch, 9)
    assert autoregressive.shape == (9,)

    fake.embed_prefix = lambda obs: embed_prefix(fake, obs)
    hidden, hidden_mask = Pi0.tap_prefix_hidden(fake, observation)
    np.testing.assert_array_equal(np.asarray(hidden_mask), np.asarray(mask))
    np.testing.assert_array_equal(np.asarray(hidden), np.asarray(tokens + 1.0))


def test_on_device_pooling_matches_legacy_numpy_geometry():
    rng = np.random.default_rng(4)
    batch, width, language_tokens = 2, 16, 5
    phid = rng.normal(size=(batch, N_VIEWS * N_IMG_TOK + language_tokens, width)).astype(np.float32)
    pmask = np.ones(phid.shape[:2], dtype=bool)
    pmask[1, -2:] = False

    expected_slots = []
    for slot in range(N_VIEWS):
        start = slot * N_IMG_TOK
        expected_slots.append(np.stack([_bin8x8(phid[row, start : start + N_IMG_TOK]) for row in range(batch)]))
    expected_patch = np.concatenate([expected_slots[slot] for slot in SLOT_TO_LABEL], axis=1).astype(np.float16)
    language_mask = pmask.copy()
    language_mask[:, : N_VIEWS * N_IMG_TOK] = False
    weights = language_mask[..., None]
    expected_language = ((phid * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1)).astype(np.float16)

    postprocess = _make_jax_postprocessor()
    patch, language = postprocess(jnp.asarray(phid), jnp.asarray(pmask))
    patch, language = np.asarray(patch), np.asarray(language)
    assert patch.dtype == language.dtype == np.float16
    assert patch.shape == (batch, 192, width)
    np.testing.assert_allclose(patch, expected_patch, rtol=0, atol=2e-3)
    np.testing.assert_allclose(language, expected_language, rtol=0, atol=2e-3)
