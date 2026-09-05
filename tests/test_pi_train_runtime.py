"""Dependency-light tests for the p5 pi0.5 runtime topology guard."""

from __future__ import annotations

import pytest

from vla_training.train.train_base._pi05_common import (
    deferred_image_resize_enabled,
    runtime_parallelism_summary,
)


def _summary(**overrides):
    values = {
        "batch_size": 64,
        "num_workers": 32,
        "fsdp_devices": 1,
        "device_count": 8,
        "local_device_count": 8,
        "process_count": 1,
        "expected_devices": 8,
        "expected_processes": 1,
        "expected_batch_size": 64,
        "expected_num_workers": 32,
        "expected_fsdp_devices": 1,
    }
    values.update(overrides)
    return runtime_parallelism_summary(**values)


def test_p5_stage_s_topology_uses_all_eight_devices():
    result = _summary()
    assert result["data_parallel_replicas"] == 8
    assert result["per_device_batch_size"] == 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("device_count", 1, "device_count"),
        ("process_count", 2, "process_count"),
        ("batch_size", 32, "batch_size"),
        ("num_workers", 16, "num_workers"),
        ("fsdp_devices", 2, "fsdp_devices"),
    ),
)
def test_stage_s_effective_recipe_drift_fails_before_training(field, value, message):
    with pytest.raises(ValueError, match=message):
        _summary(**{field: value})


def test_invalid_divisibility_fails_without_an_expected_contract():
    with pytest.raises(ValueError, match="batch_size"):
        _summary(batch_size=63, expected_batch_size=None)
    with pytest.raises(ValueError, match="fsdp_devices"):
        _summary(fsdp_devices=3, expected_fsdp_devices=None)


def test_deferred_image_resize_is_default_off_and_requires_explicit_opt_in():
    assert not deferred_image_resize_enabled({}, environ={})
    assert deferred_image_resize_enabled({"defer_image_resize_to_model_preprocess": True}, environ={})
    assert deferred_image_resize_enabled(
        {"defer_image_resize_to_model_preprocess": False},
        environ={"OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS": "1"},
    )
    assert not deferred_image_resize_enabled(
        {"defer_image_resize_to_model_preprocess": True},
        environ={"OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS": "0"},
    )


@pytest.mark.parametrize("configured", [1, "true", None])
def test_deferred_image_resize_rejects_non_boolean_config(configured):
    with pytest.raises(ValueError, match="must be a boolean"):
        deferred_image_resize_enabled({"defer_image_resize_to_model_preprocess": configured}, environ={})


def test_deferred_image_resize_rejects_ambiguous_environment_value():
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        deferred_image_resize_enabled({}, environ={"OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS": "true"})
