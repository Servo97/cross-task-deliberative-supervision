from __future__ import annotations

import types

from robomme_integration.training.upstream_ttt_train import (
    _install_flax_param_leaf_compat,
    _jax010_preprocess_observation,
    _make_jax010_auto_mesh,
    _orbax_012_legacy_metadata_api,
)


def test_orbax_012_metadata_compat_is_narrow_and_restored():
    class FakeCheckpointer:
        def metadata(self, path):
            return types.SimpleNamespace(item_metadata={"params": path})

    original = FakeCheckpointer.metadata
    instance = FakeCheckpointer()
    with _orbax_012_legacy_metadata_api(FakeCheckpointer):
        assert instance.metadata("checkpoint") == {"params": "checkpoint"}
    assert FakeCheckpointer.metadata is original
    assert instance.metadata("checkpoint").item_metadata == {"params": "checkpoint"}


def test_orbax_metadata_compat_preserves_legacy_mapping():
    class LegacyCheckpointer:
        def metadata(self, path):
            return {"params": path}

    with _orbax_012_legacy_metadata_api(LegacyCheckpointer):
        assert LegacyCheckpointer().metadata("checkpoint") == {"params": "checkpoint"}


def test_jax010_preprocess_compat_contains_only_the_required_rng_sharding_seam():
    names = _jax010_preprocess_observation.__code__.co_names
    assert "activation_sharding_constraint" in names
    assert "vmap" in names
    assert "RandomCrop" in names
    assert "ColorJitter" in names


def test_jax010_mesh_compat_explicitly_selects_auto_axes():
    names = _make_jax010_auto_mesh.__code__.co_names
    assert "AxisType" in names
    assert "Auto" in names
    assert "make_mesh" in names


def test_flax_param_compat_preserves_the_official_array_valued_reset_contract():
    names = _install_flax_param_leaf_compat.__code__.co_names
    assert "TTTBase" in names
    reset_code = next(
        value
        for value in _install_flax_param_leaf_compat.__code__.co_consts
        if isinstance(value, types.CodeType) and value.co_name == "reset"
    )
    assert "map" in reset_code.co_names
    assert "get_memory_state" in reset_code.co_names
