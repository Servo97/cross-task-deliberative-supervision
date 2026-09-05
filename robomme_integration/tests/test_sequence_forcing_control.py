from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from robomme_integration.launch import PRODUCTION_ARM_IDS, _arm_spec
from robomme_integration.training import sequence_forcing as forcing
from robomme_integration.training.arms import (
    ARM_IDS,
    NOFORCE_ARM_IDS,
    ROBOTT_ARMS,
    SEQUENCE_ARMS,
    TRAINING_ARM_IDS,
)
from wsm_settings import ROBOMME_EVAL_ROOT

OPENPI_REPO = Path(
    os.environ.get(
        "ROBOMME_OPENPI_REPO",
        str(ROBOMME_EVAL_ROOT / "openpi" / "ed923b2c"),
    )
)


def test_noforce_arms_are_production_registered_and_form_the_fast_weight_factorial():
    assert NOFORCE_ARM_IDS == ("q0_noforce", "q2_noforce")
    assert set(NOFORCE_ARM_IDS).issubset(TRAINING_ARM_IDS)
    assert set(NOFORCE_ARM_IDS).issubset(ARM_IDS)
    assert set(NOFORCE_ARM_IDS).issubset(PRODUCTION_ARM_IDS)
    assert set(NOFORCE_ARM_IDS).issubset(SEQUENCE_ARMS)
    assert "q0_noforce" not in ROBOTT_ARMS
    assert "q2_noforce" in ROBOTT_ARMS
    assert _arm_spec("q0_noforce")["sequence_forcing"] == "shared_flow_time_tau_within_L8"
    assert _arm_spec("q0_noforce")["openpi_overlay"] == {
        "version": forcing.OVERLAY_VERSION,
        "base_archive_sha256": forcing.BASE_ARCHIVE_SHA256,
        "source_pi0_sha256": forcing.BASE_PI0_SHA256,
        "patched_pi0_sha256": forcing.PATCHED_PI0_SHA256,
        "scientific_delta": "share flow tau within L=8; epsilon stays per chunk",
    }
    assert _arm_spec("q2_noforce")["robottt_fast_weights_W_t"] is True


@pytest.fixture
def config_paths(monkeypatch, tmp_path):
    for name in ("data", "assets", "init"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("ROBOMME_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROBOMME_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("WSM_INIT_FROM", str(tmp_path / "init"))


@pytest.mark.parametrize(
    ("default_arm", "control_arm", "fast_weights"),
    [("q0", "q0_noforce", False), ("q2", "q2_noforce", True)],
)
def test_noforce_config_changes_no_recipe_knob(config_paths, default_arm, control_arm, fast_weights):
    from robomme_integration.training.config import build_train_config, validate_train_config

    default = build_train_config(default_arm)
    control = build_train_config(control_arm)
    validate_train_config(default, default_arm)
    validate_train_config(control, control_arm)

    # Names are intentionally collision-free; every scientific and optimization field is matched.
    assert default.name != control.name and default.exp_name != control.exp_name
    for field in (
        "model",
        "data",
        "lr_schedule",
        "optimizer",
        "ema_decay",
        "seed",
        "num_train_steps",
        "save_interval",
        "keep_period",
        "batch_size",
        "num_workers",
        "fsdp_devices",
    ):
        default_value = getattr(default, field)
        control_value = getattr(control, field)
        if field == "data":
            # The factory is a dataclass whose selected arm does not appear in its schema.
            assert default_value == control_value
        else:
            assert default_value == control_value
    assert control.model.robottt is fast_weights
    assert control.data.stage_q_window_len == 8
    assert control.data.stage_q_chunk_stride == 10
    assert control.data.stage_q_iid_steps is False
    assert control.batch_size * control.data.stage_q_window_len == 64


@pytest.mark.skipif(not (OPENPI_REPO / forcing.PI0_RELATIVE_PATH).is_file(), reason="OpenPI source unavailable")
def test_overlay_is_an_exact_three_site_source_delta_and_default_path_is_preserved():
    original = (OPENPI_REPO / forcing.PI0_RELATIVE_PATH).read_text()
    patched = forcing.patch_pi0_source(original)

    # Reversing the marker, optional keyword, and guarded override recovers every byte of the
    # original.  In particular the stock Q0/Q2 noise/time stream and all loss branches are intact.
    marker = f'{forcing.OVERLAY_MARKER_NAME} = "{forcing.OVERLAY_VERSION}"\n\n'
    recovered = patched.replace(marker, "", 1)
    recovered = recovered.replace(forcing._NEW_SIGNATURE, forcing._OLD_SIGNATURE, 1)
    recovered = recovered.replace(forcing._NEW_TIME_BLOCK, forcing._OLD_TIME_BLOCK, 1)
    assert recovered == original
    assert "flow_noise" not in patched
    assert "noise = jax.random.normal(noise_rng, actions.shape)" in patched


@pytest.mark.skipif(not (OPENPI_REPO / forcing.PI0_RELATIVE_PATH).is_file(), reason="OpenPI source unavailable")
def test_stager_copies_only_python_runtime_and_records_content(tmp_path):
    destination = tmp_path / "overlay"
    manifest = forcing.stage_openpi_overlay(
        OPENPI_REPO,
        destination,
        source_archive_sha256=forcing.BASE_ARCHIVE_SHA256,
    )
    assert manifest["overlay_version"] == forcing.OVERLAY_VERSION
    assert manifest["copied_roots"] == ["src/openpi", "scripts"]
    assert manifest["source_pi0_sha256"] != manifest["patched_pi0_sha256"]
    assert (destination / forcing.PI0_RELATIVE_PATH).is_file()
    assert (destination / "scripts/train.py").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / ".venv").exists()
    assert not (destination / "checkpoints").exists()
    assert not list(destination.rglob("__pycache__"))
    assert not list(destination.rglob("*.pyc"))
    assert (
        forcing.stage_openpi_overlay(
            OPENPI_REPO,
            destination,
            source_archive_sha256=forcing.BASE_ARCHIVE_SHA256,
        )
        == manifest
    )

    (destination / forcing.PI0_RELATIVE_PATH).write_text("corrupt\n")
    with pytest.raises(ValueError, match="failed content verification"):
        forcing.stage_openpi_overlay(
            OPENPI_REPO,
            destination,
            source_archive_sha256=forcing.BASE_ARCHIVE_SHA256,
        )

    with pytest.raises(ValueError, match="requires canonical ed923 archive"):
        forcing.stage_openpi_overlay(
            OPENPI_REPO,
            tmp_path / "wrong-archive",
            source_archive_sha256="0" * 64,
        )


def test_shared_tau_uses_stock_time_key_while_epsilon_stays_independent():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    batch_size, length, horizon, action_dim = 5, 8, 20, 8
    rng = jax.random.key(913)
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    del preprocess_rng
    stock_tau = jax.random.beta(time_rng, 1.5, 1, (batch_size * length,)) * 0.999 + 0.001
    shared_tau = forcing.shared_flow_time_from_stock_rng(rng, batch_size, length)
    shared_by_window = np.asarray(shared_tau).reshape(batch_size, length)

    np.testing.assert_array_equal(
        shared_by_window,
        np.broadcast_to(np.asarray(stock_tau).reshape(batch_size, length)[:, :1], (batch_size, length)),
    )
    assert np.all(shared_by_window == shared_by_window[:, :1])
    assert np.any(np.asarray(stock_tau).reshape(batch_size, length) != shared_by_window)

    # This is the exact unchanged stock noise draw performed by patched compute_loss.  Chunks in
    # one window receive different epsilon even though their tau values above are identical.
    epsilon = jax.random.normal(
        noise_rng,
        (batch_size * length, horizon, action_dim),
    ).reshape(batch_size, length, horizon, action_dim)
    assert not bool(jnp.array_equal(epsilon[:, :-1], epsilon[:, 1:]))
    assert all(not np.array_equal(np.asarray(epsilon[0, 0]), np.asarray(epsilon[0, t])) for t in range(1, length))


def test_unpatched_openpi_fails_closed_instead_of_silently_running(monkeypatch):
    pytest.importorskip("openpi.models.pi0")
    import openpi.models.pi0 as pi0_model

    if getattr(pi0_model, forcing.OVERLAY_MARKER_NAME, None) == forcing.OVERLAY_VERSION:
        pytest.skip("test interpreter already activated the overlay")
    with pytest.raises(RuntimeError, match="guarded node-local OpenPI shared-tau overlay"):
        forcing.validate_loaded_overlay()
