from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import pytest

from robomme_integration import launch
from robomme_integration.eval.workspace_runner import (
    TRAIN_ONLY_HEADS,
    WORKSPACE_WINDOWS,
    serving_model_arm,
)
from robomme_integration.training.arms import V4_ARM_IDS, V4_NEW_PARAMETER_SUBTREES
from robomme_integration.training.config import build_train_config, validate_train_config
from robomme_integration.training.differential_lr import install_parameter_group_optimizer
from robomme_integration.v4_experiment_matrix import (
    CFG_ARMS,
    SEALED_CFG_SCALES,
    expand,
    load_and_validate,
)


@pytest.fixture
def paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in ("data", "assets", "init", "workspace"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("ROBOMME_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROBOMME_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("WSM_INIT_FROM", str(tmp_path / "init"))
    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("WSM_MAX_STEPS", "20000")
    monkeypatch.setenv("WSM_DECAY_STEPS", "20000")
    monkeypatch.setenv("WSM_WARMUP_STEPS", "1000")
    monkeypatch.setenv("WSM_PEAK_LR", "5e-5")
    monkeypatch.setenv("WSM_DECAY_LR", "5e-6")
    return tmp_path


@pytest.mark.parametrize(
    ("arm", "window", "dropout"),
    [
        ("v4_wsm_gdn8_drop00", 8, 0.0),
        ("v4_wsm_gdn8_drop02", 8, 0.2),
        ("v4_wsm_gdn16_drop00", 16, 0.0),
        ("v4_wsm_gdn16_drop02", 16, 0.2),
    ],
)
def test_v4_gdn_factorial_identity(paths: Path, arm: str, window: int, dropout: float):
    config = build_train_config(arm)
    validate_train_config(config, arm)
    assert config.model.wsm_tanh
    assert config.model.wsm_cond_type == "gated_deltanet"
    assert config.model.wsm_cond_window == config.data.workspace_window == window
    assert config.model.wsm_cond_history_dropout == pytest.approx(dropout)
    assert (config.ema_decay, config.optimizer.weight_decay, config.optimizer.clip_gradient_norm) == (
        0.999,
        1e-6,
        10.0,
    )
    assert (config.lr_schedule.warmup_steps, config.lr_schedule.decay_steps) == (1000, 20000)


@pytest.mark.parametrize(
    "arm",
    [
        "v4_jepa_visreg_l01_k1",
        "v4_gdn8_jepa_visreg_l01_k1",
        "v4_cfg_jepa_visreg_l01_k1",
    ],
)
def test_v4_jepa_is_visreg_never_sigreg(paths: Path, arm: str):
    if arm == "v4_cfg_jepa_visreg_l01_k1":
        # Production loads the content-recorded CFG+JEPA overlay before constructing this config.
        import openpi.models.pi0_config as pi0_config

        prior = set(pi0_config._WORKSPACE_COMBO)
        pi0_config._WORKSPACE_COMBO.clear()
        pi0_config._WORKSPACE_COMBO.update({"current_cfg", "jepa_aux_target"})
        try:
            config = build_train_config(arm)
        finally:
            pi0_config._WORKSPACE_COMBO.clear()
            pi0_config._WORKSPACE_COMBO.update(prior)
    else:
        config = build_train_config(arm)
    validate_train_config(config, arm)
    model = config.model
    assert model.wsm_jepa and (model.wsm_jepa_weight, model.wsm_jepa_num_futures) == (0.1, 1)
    assert (
        model.wsm_jepa_regularizer,
        model.wsm_jepa_sigreg_weight,
        model.wsm_jepa_visreg_weight,
        model.wsm_jepa_visreg_slices,
        model.wsm_jepa_visreg_scale_weight,
        model.wsm_jepa_visreg_shape_weight,
        model.wsm_jepa_visreg_center_weight,
    ) == ("visreg", 0.0, 0.05, 128, 1.0, 1.0, 1.0)


def test_v4_schedule_drift_fails_closed(paths: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WSM_WARMUP_STEPS", "999")
    with pytest.raises(ValueError, match="5% warmup"):
        validate_train_config(build_train_config("v4_s0"), "v4_s0")


class _OptimizerModule:
    class AdamW:
        pass


class _Schedule:
    warmup_steps = 1
    peak_lr = 5e-5
    decay_steps = 20
    decay_lr = 5e-6

    @staticmethod
    def create():
        return optax.warmup_cosine_decay_schedule(
            init_value=2.5e-5,
            peak_value=5e-5,
            warmup_steps=1,
            decay_steps=20,
            end_value=5e-6,
        )


class _AdamW:
    b1 = 0.9
    b2 = 0.95
    eps = 1e-8
    weight_decay = 1e-6
    clip_gradient_norm = 10.0


def test_differential_lr_labels_numeric_ratio_and_one_global_clip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WSM_MAX_STEPS", "20")
    clip_calls: list[float] = []
    real_clip = optax.clip_by_global_norm

    def recorded_clip(value: float):
        clip_calls.append(value)
        return real_clip(value)

    monkeypatch.setattr(optax, "clip_by_global_norm", recorded_clip)
    module = _OptimizerModule()
    receipt = install_parameter_group_optimizer("v4_wsm_tanh", module)
    assert receipt["new_parameter_subtrees"] == ["wsm_tanh_cond"]
    tx = module.create_optimizer(_AdamW(), _Schedule())
    params = {
        "llm": {"kernel": jnp.ones((1,))},
        "wsm_tanh_cond": {"kernel": jnp.ones((1,))},
    }
    grads = jax.tree.map(lambda value: jnp.full_like(value, 1e-3), params)
    updates, _ = tx.update(grads, tx.init(params), params)
    ratio = float(updates["wsm_tanh_cond"]["kernel"][0] / updates["llm"]["kernel"][0])
    assert ratio == pytest.approx(6.0, rel=2e-2)
    assert clip_calls == [10.0]


def test_differential_lr_missing_named_subtree_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WSM_MAX_STEPS", "20")
    module = _OptimizerModule()
    install_parameter_group_optimizer("v4_q2", module)
    tx = module.create_optimizer(_AdamW(), _Schedule())
    with pytest.raises(ValueError, match="could not find new parameter subtrees"):
        tx.init({"llm": {"kernel": jnp.ones((1,))}})


def test_differential_lr_labels_actual_nnx_state_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WSM_MAX_STEPS", "20")

    class Model(nnx.Module):
        def __init__(self):
            rngs = nnx.Rngs(0)
            self.llm = nnx.Linear(1, 1, rngs=rngs)
            self.wsm_tanh_cond = nnx.Linear(1, 1, rngs=rngs)

    module = _OptimizerModule()
    install_parameter_group_optimizer("v4_wsm_tanh", module)
    tx = module.create_optimizer(_AdamW(), _Schedule())
    params = nnx.state(Model(), nnx.Param)
    state = tx.init(params)
    assert state is not None


def test_v4_matrix_phase_coverage_promotion_gates_and_cfg_cells():
    spec = load_and_validate()
    cells = expand(spec)
    assert set(spec["implemented_arms"]) == V4_ARM_IDS
    assert {record["method"] for record in spec["blocked_methods"]} == {
        "t1_naive_ttt",
        "r1_paper_robottt",
        "framesamp",
        "framesamp_plus_wsm",
    }
    training = [cell for cell in cells if cell["kind"] == "training"]
    assert len(training) == 40  # Phase A 26 + Phase B controls 6 + all16 core 8
    assert all(
        cell["task"] not in {"PickXtimes", "MoveCube"} for cell in training if cell["phase"].startswith("phase_b")
    )
    assert len([cell for cell in cells if cell["kind"] == "promotion_slot"]) == 18
    for arm in CFG_ARMS:
        grouped: dict[str, set[float]] = {}
        for cell in cells:
            if cell["kind"] == "evaluation" and cell["arm"] == arm:
                grouped.setdefault(cell["training_cell_id"], set()).add(cell["cfg_guidance_scale"])
        assert grouped and all(scales == set(SEALED_CFG_SCALES) for scales in grouped.values())


def test_v4_eval_and_serving_identities_cover_every_arm():
    server = Path("robomme_integration/eval/execution_model_server.py").read_text(encoding="utf-8")
    train_only = set(TRAIN_ONLY_HEADS)
    assert all(f'"{arm}"' in server for arm in V4_ARM_IDS - train_only)
    assert "V4_CFG_EVAL_SCALES = frozenset({0.5, 1.0, 1.5, 2.0})" in server
    assert WORKSPACE_WINDOWS["v4_wsm_gdn16_drop02"] == 16
    assert TRAIN_ONLY_HEADS["v4_jepa_visreg_l01_k1"] == ("wsm_jepa_head",)
    assert serving_model_arm("v4_jepa_visreg_l01_k1") == "v4_s0"
    assert serving_model_arm("v4_gdn8_jepa_visreg_l01_k1") == "v4_wsm_gdn8_drop00"
    assert serving_model_arm("v4_cfg_jepa_visreg_l01_k1") == "v4_wsm_cfg"


def _workspace_args(arm: str) -> list[str]:
    encoder = "a" * 64
    manifest = "b" * 64
    root = f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/PickXtimes/{encoder}"
    return [
        "--task",
        "PickXtimes",
        "--arm",
        arm,
        "--workspace-encoder-id",
        encoder,
        "--workspace-s3",
        f"{root}/omega",
        "--workspace-manifest-sha256",
        manifest,
        "--dry-run",
    ]


def test_v4_launch_manifest_binds_optimizer_sources_visreg_and_cfg_scale_sweep():
    source = Path(launch.__file__).resolve().parent
    plans = {}
    for arm in (
        "v4_wsm_gdn8_drop00",
        "v4_wsm_gdn8_drop02",
        "v4_wsm_gdn16_drop00",
        "v4_wsm_gdn16_drop02",
        "v4_gdn8_jepa_visreg_l01_k1",
        "v4_ptrm",
    ):
        plans[arm] = launch.build_plan(launch.parser().parse_args(_workspace_args(arm)), source)
    assert {plan["manifest"]["scientific"]["sources"]["openpi"]["sha256"] for plan in plans.values()} == {
        launch.PTRM_OPENPI_SHA
    }
    assert {plan["environment"]["OPENPI_REQUIRED_SENTINEL"] for plan in plans.values()} == {"_WSM_V4_ADVANCED"}
    parent = plans["v4_wsm_gdn8_drop00"]["manifest"]["scientific"]["training"]
    assert parent["ema_decay"] == 0.999
    assert parent["optimizer"]["weight_decay"] == 1e-6
    assert parent["optimizer"]["clip_gradient_norm"] == 10.0
    assert parent["parameter_groups"]["new_modules"] == {
        "subtrees": ["wsm_tanh_cond"],
        "active": True,
        "peak_lr": 3e-4,
        "decay_lr": 3e-5,
    }
    expected_dropout = {
        "v4_wsm_gdn8_drop00": 0.0,
        "v4_wsm_gdn8_drop02": 0.2,
        "v4_wsm_gdn16_drop00": 0.0,
        "v4_wsm_gdn16_drop02": 0.2,
        "v4_gdn8_jepa_visreg_l01_k1": 0.0,
        "v4_ptrm": 0.0,
    }
    assert {
        arm: plan["manifest"]["scientific"]["mechanism"]["train_history_dropout"] for arm, plan in plans.items()
    } == expected_dropout
    cfg = launch.build_plan(launch.parser().parse_args(_workspace_args("v4_cfg_jepa_visreg_l01_k1")), source)
    science = cfg["manifest"]["scientific"]
    assert science["evaluation_protocol"]["cfg_guidance_scales"] == list(SEALED_CFG_SCALES)
    assert science["mechanism"]["jepa"]["regularizer"] == "visreg"
    assert cfg["environment"]["OPENPI_REQUIRED_SENTINEL"] == "_WSM_CFG_JEPA_V4"


def test_legacy_arm_registry_is_immutable_prefix():
    # V4 identities append; old drop05/SIGReg identities remain distinct and are never relabelled.
    assert "wsm_d8_drop05" not in V4_ARM_IDS
    assert "gdn8_jepa_l01_k1" not in V4_ARM_IDS
    assert V4_NEW_PARAMETER_SUBTREES["v4_s0"] == ()
    v4_q0 = launch._arm_spec("v4_q0")
    assert v4_q0["sequence_no_fast_weight_control"] is True
    assert v4_q0["naive_ttt_control"] is False
    assert launch._arm_spec("q0")["naive_ttt_control"] is False
    assert launch._arm_spec("q0")["actual_naive_ttt_t1"] is False
    assert launch._arm_spec("v4_q2")["project_fast_weight_proxy"] is True
    spec = load_and_validate()
    assert (
        next(record for record in spec["blocked_methods"] if record["method"] == "t1_naive_ttt")["readiness"]
        == "blocked"
    )
