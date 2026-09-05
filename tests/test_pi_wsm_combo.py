"""S5 COMBO arm (gated-DeltaNet workspace READ + train-only JEPA aux TARGET in one run).

Until 2026-08-04 the two axes lived on mutually exclusive interfaces: `wsm_tanh` conditions the
action expert at inference, `wsm_jepa` is a train-only aux with a base serve path. The exclusion was
bookkeeping, not mechanism — the read is an additive vector on `adarms_cond` and the aux is a
separate loss term off the action-expert penultimate. These tests pin the relaxation to EXACTLY that
one pair and pin the properties the arm's comparability rests on.

1  Only {tanh, jepa} is allowed; every other pair still fails loudly, on both the model config and
   the trainer-side interface guard.
2  The combo param tree == the deltanet tree + the `wsm_jepa_head` subtree, exactly (so the sealed
   deltanet serve contract is untouched and `remove_extra_params` drops the head at load).
3  compute_loss(train=True) == the deltanet loss + ONE scalar over [B, H]: the read is unchanged by
   the aux and the aux is unchanged by the read.
4  sample_actions is byte-identical to the deltanet arm — the aux is train-time only.
5  The dataloader ships wsm_w_window AND wsm_w_target under the two gates, and the historical
   either/or is bit-unchanged when the new gate is off.
6  The two shipped recipes carry both axes and leave norm_split_nav off (sealed-table comparability).

Run: PYTHONPATH=. ~/Research/robocasa_openpi/.venv/bin/python -m pytest -q \
     tests/test_pi_wsm_combo.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wsm_settings import ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx", reason="flax nnx required")

from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.shared import array_typing as at  # noqa: E402

B, H, DW, K = 3, 10, 512, 8
ROBOCASA_CONFIG = REPO / "scripts/configs/train/pi05_stage_s1_gdnjepa_finetune.yaml"
REMEMBENCH_CONFIG = REPO / "scripts/configs/train/pi05_rmb_gdnjepa_finetune.yaml"


@pytest.fixture(autouse=True)
def _release_jax_memory():
    yield
    jax.clear_caches()


def _cfg(*, jepa: bool) -> Pi0Config:
    """The shipped combo recipe on a dummy-width backbone; jepa=False is the deltanet-only arm."""
    return Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=H,
        max_token_len=48,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        wsm_tanh=True,
        wsm_cond_type="gated_deltanet",
        wsm_cond_window=K,
        wsm_k_window=K,
        wsm_jepa=jepa,
        wsm_jepa_w_dim=DW,
        wsm_jepa_weight=0.1,
        wsm_jepa_sigreg_weight=0.05,
    )


def _obs_act(cfg, *, window: bool, targets: bool, seed: int = 5):
    obs_spec, act_spec = cfg.inputs_spec(batch_size=B)
    keys = jax.random.split(jax.random.key(seed), 64)
    idx = [0]

    def rnd(spec):
        idx[0] += 1
        key = keys[idx[0] % 64]
        if spec.dtype == jnp.float32:
            return jax.random.normal(key, spec.shape, spec.dtype)
        if spec.dtype == bool:
            return jnp.ones(spec.shape, bool)
        return jnp.zeros(spec.shape, spec.dtype)

    obs = jax.tree.map(rnd, obs_spec)
    obs = dataclasses.replace(
        obs,
        tokenized_prompt=jnp.ones((B, cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, cfg.max_token_len), bool),
    )
    if window:
        obs = dataclasses.replace(obs, wsm_w_window=jax.random.normal(jax.random.key(seed + 3), (B, K, DW)))
    if targets:
        obs = dataclasses.replace(
            obs,
            wsm_w_target=jax.random.normal(jax.random.key(seed + 1), (B, DW)),
            wsm_w_target_valid=jnp.asarray([True, True, False]),
        )
    act = jax.random.normal(jax.random.key(seed + 2), act_spec.shape, act_spec.dtype)
    return obs, act


# ------------------------- 1: exactly one pair is sanctioned -------------------------


def test_01_only_the_tanh_plus_jepa_pair_is_allowed():
    Pi0Config(pi05=True, wsm_tanh=True, wsm_jepa=True)  # the combo: must construct
    for kwargs in (
        {"wsm_cfg2": True, "wsm_tanh": True},
        {"wsm_cfg2": True, "wsm_jepa": True},
        {"wsm_cfg": True, "wsm_jepa": True},
        {"wsm": True, "wsm_tanh": True},
        {"wsm": True, "wsm_cfg2": True, "wsm_jepa": True},
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Pi0Config(pi05=True, **kwargs)


def test_01b_trainer_side_guard_quotes_the_same_pair():
    """The trainer resolves interfaces from env BEFORE openpi is imported, so it carries its own
    copy of the rule; drift between the two would fail 60k steps late or not at all."""
    from openpi.models import pi0_config as model_config

    from vla_training.train.train_base import _pi05_common

    assert _pi05_common._WORKSPACE_COMBO == {"tanh", "jepa"}
    assert model_config._WORKSPACE_COMBO == {"tanh", "jepa_aux_target"}
    # H13 (2026-08-12) added a SECOND sanctioned pair (the same read + the live joint WSM aux), so
    # both sides now quote a tuple of combos rather than one set. The invariant under test is
    # unchanged: the two copies of the rule must not drift. Naming differs by design — the trainer
    # resolves ENV flag names, the model config resolves interface names.
    assert _pi05_common._SANCTIONED_COMBOS == (_pi05_common._WORKSPACE_COMBO, {"tanh", "h13"})
    assert model_config._SANCTIONED_COMBOS == (
        model_config._WORKSPACE_COMBO,
        {"tanh", "h13_live_aux"},
    )
    assert len(_pi05_common._SANCTIONED_COMBOS) == len(model_config._SANCTIONED_COMBOS)
    source = (REPO / "vla_training/train/train_base/_pi05_common.py").read_text(encoding="utf-8")
    assert "if len(enabled) > 1 and set(enabled) not in _SANCTIONED_COMBOS:" in source
    # ...and the aux head must be excused from the checkpoint loader alongside the conditioner. The
    # ternary chain became an additive builder when H13 added three more subtrees; what matters is
    # that BOTH the conditioner and the aux head are appended when their flags are on (the resolved
    # regex string for this pair is unchanged — pinned end-to-end by test_10 in the salient suite).
    assert 'missing_subtrees.append("wsm_tanh_cond")' in source
    assert 'missing_subtrees.append("wsm_jepa_head")' in source


# ------------------------- 2: the tree is deltanet + the aux head, exactly -------------------------


def test_02_combo_tree_is_the_deltanet_tree_plus_the_jepa_head():
    with at.disable_typechecking():
        m_read = _cfg(jepa=False).create(jax.random.key(0))
        m_combo = _cfg(jepa=True).create(jax.random.key(0))
    read = {".".join(str(x) for x in k) for k, _ in nnx.to_flat_state(nnx.state(m_read))}
    combo = {".".join(str(x) for x in k) for k, _ in nnx.to_flat_state(nnx.state(m_combo))}
    assert read <= combo, f"the combo must contain the whole deltanet tree; missing {sorted(read - combo)[:5]}"
    extra = sorted(combo - read)
    assert extra and all("wsm_jepa_head" in path for path in extra), extra[:8]
    # The serve auto-detect keys off pos_decay_bias in the conditioner subtree, which the aux never
    # touches: a combo checkpoint is still detected as gated_deltanet at window K.
    decay = [p for p in combo if "pos_decay_bias" in p]
    assert len(decay) == 1 and "wsm_tanh_cond" in decay[0], decay


# ------------------------- 3 + 4: the two terms are additive and the aux is train-only ------------


def test_03_combo_loss_is_the_deltanet_loss_plus_one_scalar():
    with at.disable_typechecking():
        m_read = _cfg(jepa=False).create(jax.random.key(0))
        m_combo = _cfg(jepa=True).create(jax.random.key(0))
        nnx.update(m_combo, nnx.state(m_read))  # share every deltanet param; head stays as initialized
        cfg = _cfg(jepa=True)
        obs_both, act = _obs_act(cfg, window=True, targets=True)
        obs_read, _ = _obs_act(cfg, window=True, targets=False)

        rng = jax.random.key(11)
        l_read = np.asarray(m_read.compute_loss(rng, obs_read, act, train=True))
        l_combo = np.asarray(m_combo.compute_loss(rng, obs_both, act, train=True))
        diff = l_combo - l_read
        assert diff.shape == l_read.shape
        assert float(np.ptp(diff)) < 1e-5, f"the aux must be one scalar over [B,H]; spread={np.ptp(diff)}"
        assert float(diff.mean()) > 0.0

        # train=False: the aux is off, so the two arms agree exactly even with targets present.
        assert np.array_equal(
            np.asarray(m_read.compute_loss(rng, obs_read, act, train=False)),
            np.asarray(m_combo.compute_loss(rng, obs_both, act, train=False)),
        )
        # Fail-loud both ways: the read needs its window, the aux needs its targets.
        with pytest.raises(ValueError, match="wsm_w_target"):
            m_combo.compute_loss(rng, obs_read, act, train=True)
        obs_target_only, _ = _obs_act(cfg, window=False, targets=True)
        with pytest.raises(ValueError, match="wsm_w_window"):
            m_combo.compute_loss(rng, obs_target_only, act, train=True)


def test_04_sample_actions_is_byte_identical_to_the_deltanet_arm():
    with at.disable_typechecking():
        m_read = _cfg(jepa=False).create(jax.random.key(0))
        m_combo = _cfg(jepa=True).create(jax.random.key(0))
        nnx.update(m_combo, nnx.state(m_read))
        cfg = _cfg(jepa=True)
        obs, _ = _obs_act(cfg, window=True, targets=False)  # serve never ships targets
        a_read = np.asarray(m_read.sample_actions(jax.random.key(7), obs, num_steps=2))
        a_combo = np.asarray(m_combo.sample_actions(jax.random.key(7), obs, num_steps=2))
    assert np.array_equal(a_read, a_combo), "the aux must not touch the inference path"


# ------------------------- 5: the loader ships both keys, and only under the gate -----------------


def test_05_loader_gate_ships_window_and_target_together():
    # Source-level, not import-level: the loader module pulls in gr00t, which is absent from most of
    # the study's environments (the dedicated loader suite importorskips for the same reason).
    loader = ROBOCASA_OPENPI_SRC / "openpi" / "groot_utils" / "groot_openpi_dataset.py"
    source = loader.read_text(encoding="utf-8")
    assert source.count("if not _WSM_JEPA_TARGETS or _WSM_JEPA_WITH_WINDOW:") == 2, (
        "both the single-dataset and the mixture __getitem__ must honour the combo gate"
    )
    # Default OFF: every shipped S1/S2/S3 run keeps the historical either/or.
    assert '_WSM_JEPA_WITH_WINDOW = os.environ.get("WSM_JEPA_WITH_WINDOW", "0") == "1"' in source

    entry = (REPO / "vla_training/train/train_base/finetune_pi_05_with_workspace.py").read_text(encoding="utf-8")
    # The three gates must be exported together, and only for the tanh interface.
    for name in ("WSM_JEPA", "WSM_JEPA_TARGETS", "WSM_JEPA_WITH_WINDOW"):
        assert f'os.environ["{name}"] = "1"' in entry
    assert 'if args.interface != "tanh":' in entry


# ------------------------- 6: the shipped recipes -------------------------


@pytest.mark.parametrize("path", [ROBOCASA_CONFIG, REMEMBENCH_CONFIG])
def test_06_shipped_combo_recipes_carry_both_axes_and_no_norm_split(path):
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    model, data = recipe["model"], recipe["data"]
    assert model["jepa_aux"] is True
    assert model["cond_type"] == "gated_deltanet" and model["cond_window"] == 8
    assert model["jepa_weight"] == 0.1
    # sigreg 0.05 and k=1 are the defaults and must stay UNSET, so the arm reuses the exact jw01 aux.
    assert "sigreg_weight" not in model and "jepa_num_futures" not in model
    assert "jepa_regularizer" not in model
    # Sealed-table / ReMemBench comparability: the nav norm-stat split was never used by these sets.
    assert "norm_split_nav" not in data
    assert recipe["train"]["batch_size"] == 64


def test_06b_recipes_differ_from_their_deltanet_parents_only_by_the_aux():
    for combo_path, parent_path in (
        (ROBOCASA_CONFIG, REPO / "scripts/configs/train/pi05_stage_s1_deltanet_finetune.yaml"),
        (REMEMBENCH_CONFIG, REPO / "scripts/configs/train/pi05_rmb_deltanet_finetune.yaml"),
    ):
        combo = yaml.safe_load(combo_path.read_text(encoding="utf-8"))
        parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
        # Identity leaves must differ (checkpoints must not collide); everything else must not.
        for section, keys in (("model", ("config_name",)), ("train", ("exp_name", "output_dir"))):
            for key in keys:
                assert combo[section][key] != parent[section][key], (section, key)
                combo[section].pop(key), parent[section].pop(key)
        assert combo["data"] == parent["data"]
        assert combo["optim"] == parent["optim"]
        assert combo["train"] == parent["train"]
        assert set(combo["model"]) - set(parent["model"]) == {"jepa_aux", "jepa_weight"}
        assert {k: v for k, v in combo["model"].items() if k in parent["model"]} == parent["model"]
