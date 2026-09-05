"""Stage-Q training-recipe variants A6 (i.i.d. window steps) and A7 (staged recipe).

Both variants are opt-in; the load-bearing claims are (i) they do what the steelman asks and
(ii) their defaults are provably inert, so every existing Q0-Q3 result stays comparable.

  A6  data.iid_window_steps: true
      * identical SUPPORT: the samplable multiset of (episode, step) pairs, the number of items,
        and the per-epoch sample counts are byte-identical to the window loader;
      * decorrelated: an item's L steps are no longer 8 consecutive stride-8 steps of one episode;
      * deterministic under the run seed;
      * default off routes to the untouched StageQWindowDataset;
      * i.i.d. + fast weights ON is rejected at config-parse time (it would break the W chain).

  A7  staged: {phase1_steps, phase1_freeze: backbone, phase1_lr_scale_robottt}
      * phase 1: every non-robottt param is bitwise unchanged after a real train step, while the
        robottt subtree moves by exactly phase1_lr_scale_robottt x its unscaled step;
      * after the boundary: both move, and the step is BITWISE identical to staged=None;
      * absent block: the mask helpers are identity (no ops added to the graph).
      * a canary (num_train_steps=1) CLAMPS phase 1 to the run instead of refusing to launch.

  Divergence tripwire (all Stage-Q runs, added after the A7 attempt-1 NaN at step 8200)
      * 3 consecutive non-finite logged intervals => one-line diagnosis naming the broken param
        group (subtree vs backbone) + exit 1; a single bad interval only warns.

Run: PYTHONPATH=. ~/Research/envs/openpi-jax-latest/bin/python -m pytest -q \
     tests/test_stage_q_variants.py
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wsm_settings import ROBOCASA_OPENPI_ROOT, ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx", reason="flax nnx required")

from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.shared import array_typing as at  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402
from openpi.training import stage_q_windows as _sqw  # noqa: E402
from openpi.training import utils as training_utils  # noqa: E402

CONFIG_DIR = REPO / "scripts" / "configs" / "train"
Q0_IID_YAML = CONFIG_DIR / "pi05_stage_q0_iid_finetune.yaml"
Q2_STAGED_YAML = CONFIG_DIR / "pi05_stage_q2_staged_finetune.yaml"
SHARED_YAML = CONFIG_DIR / "pi05_stage_q_finetune.yaml"


def _load_train_module():
    """Import the fork's scripts/train.py (not a package) as a module."""
    path = ROBOCASA_OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_fork_train_variants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train = _load_train_module()


@pytest.fixture(autouse=True)
def _release_jax_memory():
    yield
    jax.clear_caches()


# ============================== A6: i.i.d. window steps ==============================


class _FakeStepDS:
    """all_steps trajectory-major, like the real GrootOpenpiSingleDataset."""

    def __init__(self, lengths, tag="d"):
        self.tag = tag
        self.trajectory_ids = np.asarray([f"{tag}_traj_{i}" for i in range(len(lengths))])
        self.trajectory_lengths = np.asarray(lengths)
        self.all_steps = [(f"{tag}_traj_{i}", b) for i, n in enumerate(lengths) for b in range(n)]

    def __len__(self):
        return len(self.all_steps)

    def __getitem__(self, i):
        traj, base = self.all_steps[i]
        return {"traj": traj, "base": int(base)}


LENGTHS = [80, 80, 64, 64, 40, 33, 17, 9]  # incl. two episodes too short for one window
WIN, STRIDE = 4, 2


def _both_datasets(seed=0):
    win = _sqw.StageQWindowDataset([], 5, window_len=WIN, chunk_stride=STRIDE, _datasets=[_FakeStepDS(LENGTHS)])
    iid = _sqw.StageQIidStepDataset(
        [], 5, window_len=WIN, chunk_stride=STRIDE, _datasets=[_FakeStepDS(LENGTHS)], seed=seed
    )
    return win, iid


def _support(dataset):
    return collections.Counter((step["traj"], step["base"]) for i in range(len(dataset)) for step in dataset[i])


def test_iid_dataset_has_the_exact_same_support_as_the_window_loader():
    win, iid = _both_datasets()
    assert len(iid) == len(win), "same number of items => same batches/epoch and same drop_last"
    assert _support(iid) == _support(win), (
        "A6 must change ONLY the correlation structure: the samplable (episode, step) multiset, "
        "the stride alignment, the trailing-step drop and the short-episode drop are all identical"
    )
    # And that support is exactly once per samplable step (no re-weighting sneaks in).
    assert set(_support(iid).values()) == {1}


def test_iid_dataset_decorrelates_items_that_the_window_loader_correlates():
    win, iid = _both_datasets()
    win_single = [len({s["traj"] for s in win[i]}) == 1 for i in range(len(win))]
    iid_single = [len({s["traj"] for s in iid[i]}) == 1 for i in range(len(iid))]
    assert all(win_single), "control: every window item is 8 steps of ONE episode (the G5 defect)"
    # 6 episodes survive the drop rule; P(all 4 steps from one episode) is a few percent.
    assert np.mean(iid_single) < 0.15, f"i.i.d. items must mix episodes (got {np.mean(iid_single):.2f})"

    # The window loader's other correlation: consecutive stride-aligned bases.
    def consecutive(item):
        bases = [s["base"] for s in item]
        return all(b2 - b1 == STRIDE for b1, b2 in zip(bases, bases[1:]))

    assert all(consecutive(win[i]) for i in range(len(win)))
    assert np.mean([consecutive(iid[i]) for i in range(len(iid))]) < 0.15


def test_iid_dataset_is_deterministic_under_the_seed():
    a = _sqw.StageQIidStepDataset([], 5, window_len=WIN, chunk_stride=STRIDE, _datasets=[_FakeStepDS(LENGTHS)], seed=7)
    b = _sqw.StageQIidStepDataset([], 5, window_len=WIN, chunk_stride=STRIDE, _datasets=[_FakeStepDS(LENGTHS)], seed=7)
    c = _sqw.StageQIidStepDataset([], 5, window_len=WIN, chunk_stride=STRIDE, _datasets=[_FakeStepDS(LENGTHS)], seed=8)

    def items(d):
        return [d[i] for i in range(len(d))]

    assert items(a) == items(b), "same seed => same grouping"
    assert items(a) != items(c), "seed must matter"
    # Spawned workers get the dataset by pickle; the grouping must survive the round trip.
    import pickle

    assert items(pickle.loads(pickle.dumps(a))) == items(a)


def test_iid_dataset_spans_multiple_source_datasets():
    """The pool is global, so an item may mix soups — the point of 'i.i.d. across the dataset'."""
    ds = _sqw.StageQIidStepDataset(
        [],
        5,
        window_len=WIN,
        chunk_stride=STRIDE,
        _datasets=[_FakeStepDS([80, 80], "a"), _FakeStepDS([80, 80], "b")],
        seed=3,
    )
    tags = [{s["traj"].split("_")[0] for s in ds[i]} for i in range(len(ds))]
    assert any(len(t) > 1 for t in tags), "items must be able to mix source datasets"


def test_window_loader_is_untouched_when_the_flag_is_off(monkeypatch):
    """Default off must route to the ORIGINAL class, and on must thread the run seed."""
    seen = {}

    def _rec(name):
        def _make(*args, **kwargs):
            seen["cls"], seen["kwargs"] = name, kwargs
            return object()

        return _make

    monkeypatch.setattr(_data_loader._stage_q_windows, "StageQWindowDataset", _rec("window"))
    monkeypatch.setattr(_data_loader._stage_q_windows, "StageQIidStepDataset", _rec("iid"))
    base = dict(repo_id="groot", data_dirs=[{"x": 1}], stage_q_window_len=8, stage_q_chunk_stride=8)

    _data_loader.create_torch_dataset(types.SimpleNamespace(**base, stage_q_iid_steps=False), 50, None, seed=11)
    assert seen["cls"] == "window" and "seed" not in seen["kwargs"]

    _data_loader.create_torch_dataset(types.SimpleNamespace(**base, stage_q_iid_steps=True), 50, None, seed=11)
    assert seen["cls"] == "iid" and seen["kwargs"]["seed"] == 11
    assert (seen["kwargs"]["window_len"], seen["kwargs"]["chunk_stride"]) == (8, 8)


def test_data_config_default_keeps_iid_off():
    dc = _config.StageQRobocasaSequenceDataConfig(stage_q_window_len=8)
    assert dc.stage_q_iid_steps is False
    assert _config.DataConfig().stage_q_iid_steps is False


# ---------------------- A6 guardrail: i.i.d. + fast weights = config error ----------------------


def _train_config(*, robottt: bool, iid: bool = False, staged=None, num_train_steps: int = 60000):
    return _config.TrainConfig(
        name="stage_q_variant_test",
        exp_name="stage_q_variant_test",
        model=Pi0Config(pi05=True, robottt=robottt),
        data=_config.StageQRobocasaSequenceDataConfig(stage_q_window_len=8, stage_q_iid_steps=iid),
        num_train_steps=num_train_steps,
        staged=staged,
    )


def test_iid_with_fast_weights_on_is_a_config_error():
    with pytest.raises(ValueError, match="incompatible with model.robottt"):
        _train_config(robottt=True, iid=True)
    # ... and the two legal combinations still build.
    assert _train_config(robottt=False, iid=True).data.stage_q_iid_steps is True
    assert _train_config(robottt=True, iid=False).data.stage_q_iid_steps is False


def test_iid_without_the_window_loader_is_a_config_error():
    """i.i.d. sampling is defined by the window enumeration; it is meaningless per-step."""
    per_step = dataclasses.replace(_config.DataConfig(), stage_q_iid_steps=True)  # window_len == 0
    with pytest.raises(ValueError, match="requires the Stage-Q window loader"):
        _config.TrainConfig(name="x", exp_name="x", model=Pi0Config(pi05=True), data=per_step)


# ============================== A7: staged recipe ==============================

B, L = 1, 2  # windows x chunk-steps; the freeze proof is shape independent


def _staged_config(staged, *, num_train_steps=100):
    model = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        robottt=True,
        robottt_token_dim=16,
        robottt_fast_hidden=8,
        robottt_num_registers=3,
        robottt_window_len=L,
        robottt_tbptt_segment=L,
    )
    return _config.TrainConfig(
        name="staged_test",
        exp_name="staged_test",
        model=model,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-3, decay_steps=50, decay_lr=1e-4),
        batch_size=B,
        num_train_steps=num_train_steps,
        staged=staged,
    )


def _window_batch(model_cfg, seed=4):
    obs_spec, act_spec = model_cfg.inputs_spec(batch_size=B * L)
    kx = jax.random.split(jax.random.key(seed), 64)
    idx = [0]

    def rnd(spec):
        idx[0] += 1
        k = kx[idx[0] % 64]
        if spec.dtype == jnp.float32:
            return jax.random.normal(k, spec.shape, spec.dtype)
        if spec.dtype == bool:
            return jnp.ones(spec.shape, bool)
        return jnp.zeros(spec.shape, spec.dtype)

    obs_flat = jax.tree.map(rnd, obs_spec)
    obs_flat = dataclasses.replace(
        obs_flat,
        tokenized_prompt=jnp.ones((B * L, model_cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B * L, model_cfg.max_token_len), bool),
    )
    act_flat = jax.random.normal(jax.random.key(seed + 1), act_spec.shape, act_spec.dtype)

    def to_window(x):
        return x.reshape(B, L, *x.shape[1:])

    return jax.tree.map(to_window, obs_flat), to_window(act_flat)


def _trained_regime(model):
    """De-zero the identity seams so the robottt subtree has a gradient at all.

    At EXACT init the fast-weight path is an exact identity by design (zero-init readout, zero-init
    W0 second layer, zero-init adaRMS Dense), which makes dL/d(robottt) *exactly* zero — a freeze
    test there would pass vacuously. A real A7 run does not sit at that point: it initializes from
    the S0 checkpoint, so the adaRMS seam it injects into is trained (only the robottt subtree is
    back-filled fresh). This reproduces that regime deterministically, exactly like the Q2-emergence
    test in test_stage_q_dispatch.py.
    """
    model.robottt_fast.readout.kernel[...] = 0.1 * jax.random.normal(
        jax.random.key(11), model.robottt_fast.readout.kernel[...].shape
    )
    model.robottt_fast.w0_w2[...] = 0.1 * jax.random.normal(jax.random.key(12), model.robottt_fast.w0_w2[...].shape)
    flat = dict(nnx.to_flat_state(nnx.state(model)))
    for i, (path, val) in enumerate(list(flat.items())):
        name = ".".join(str(x) for x in path)
        if name.endswith("_1.Dense_0.kernel") and ("pre_attention_norm" in name or "pre_ffw_norm" in name):
            flat[path] = type(val)(0.02 * jax.random.normal(jax.random.key(100 + i), val[...].shape))
    nnx.update(model, nnx.from_flat_state(flat))
    return model


def _init_state(config, rng, step=0):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    model = _trained_regime(config.model.create(rng))
    params = nnx.state(model)
    return training_utils.TrainState(
        step=step,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


def _split_params(params):
    """(robottt subtree leaves, everything else) as host numpy, keyed by path."""
    sub, other = {}, {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        key = "/".join(str(getattr(p, "key", p)) for p in path)
        (sub if key.startswith("robottt_fast") else other)[key] = np.asarray(leaf)
    assert sub and other, "the tiny model must have both a robottt subtree and a backbone"
    return sub, other


def _run_step(staged, step, seed=0):
    config = _staged_config(staged)
    with at.disable_typechecking():
        state = _init_state(config, jax.random.key(seed), step=step)
        before = _split_params(state.params)
        new_state, info = _train.stage_q_train_step(config, jax.random.key(7), state, _window_batch(config.model))
        after = _split_params(new_state.params)
        info = {k: float(v) for k, v in info.items()}
    return before, after, info


def _deltas(before, after):
    return {k: after[k] - before[k] for k in before}


def test_staged_phase1_freezes_the_backbone_and_scales_the_subtree_lr():
    staged = _config.StagedRecipe(phase1_steps=10, phase1_freeze="backbone", phase1_lr_scale_robottt=10.0)
    (sub0, oth0), (sub1, oth1), info = _run_step(staged, step=0)

    # 1) Frozen: every non-robottt parameter is BITWISE unchanged after a real optimizer step.
    for k in oth0:
        assert np.array_equal(oth0[k], oth1[k]), f"phase 1 must not move {k}"
    # 2) Trained: the robottt subtree moved.
    moved = [k for k in sub0 if not np.array_equal(sub0[k], sub1[k])]
    assert moved, "phase 1 must train the robottt subtree"
    assert info["staged_phase1"] == 1.0

    # 3) The move is exactly phase1_lr_scale_robottt x the unscaled step (post-optimizer LR scale).
    unscaled = _config.StagedRecipe(phase1_steps=10, phase1_lr_scale_robottt=1.0)
    (sub0b, _), (sub1b, _), _ = _run_step(unscaled, step=0)
    d10, d1 = _deltas(sub0, sub1), _deltas(sub0b, sub1b)
    for k in moved:
        # atol covers f32 rounding in Adam's eps/subtraction (observed max |diff| ~6e-8 on ~4e-3 steps).
        np.testing.assert_allclose(d10[k], 10.0 * d1[k], rtol=5e-3, atol=1e-6, err_msg=k)


def test_staged_phase2_resumes_the_full_finetune_bitwise():
    staged = _config.StagedRecipe(phase1_steps=10, phase1_lr_scale_robottt=10.0)
    (sub0, oth0), (sub1, oth1), info = _run_step(staged, step=10)  # first step AFTER the boundary
    assert info["staged_phase1"] == 0.0
    assert [k for k in oth0 if not np.array_equal(oth0[k], oth1[k])], "phase 2 must train the backbone"
    assert [k for k in sub0 if not np.array_equal(sub0[k], sub1[k])], "phase 2 must train the subtree"

    # And that step is BITWISE the step the unstaged recipe would have taken.
    (nsub0, noth0), (nsub1, noth1), ninfo = _run_step(None, step=10)
    for k in oth0:
        assert np.array_equal(oth1[k], noth1[k]), f"post-boundary step must equal staged=None on {k}"
    for k in sub0:
        assert np.array_equal(sub1[k], nsub1[k]), f"post-boundary step must equal staged=None on {k}"
    assert ninfo["loss"] == info["loss"] and "staged_phase1" not in ninfo


def test_staged_absent_is_an_identity_on_the_train_step():
    """Default = no ops added: the mask helpers return the SAME objects, so the graph is unchanged."""
    tree = {"robottt_fast": {"w": jnp.ones((2,))}, "backbone": {"w": jnp.ones((2,))}}
    assert _train.staged_mask_grads(None, 0, tree) is tree
    assert _train.staged_mask_updates(None, 0, tree) is tree
    assert _config.TrainConfig(name="x", exp_name="x").staged is None


def test_staged_mask_helpers_select_exactly_the_subtree():
    staged = _config.StagedRecipe(phase1_steps=5, phase1_lr_scale_robottt=3.0)
    tree = {
        "robottt_fast": {"proj_q": {"kernel": jnp.ones((2, 2))}},
        "PaliGemma": {"llm": {"w": jnp.ones((3,), jnp.bfloat16)}},
        "state_proj": {"kernel": jnp.ones((2,))},
    }
    g1 = _train.staged_mask_grads(staged, 0, tree)  # phase 1
    assert float(jnp.sum(g1["robottt_fast"]["proj_q"]["kernel"])) == 4.0
    assert float(jnp.sum(g1["PaliGemma"]["llm"]["w"])) == 0.0
    assert float(jnp.sum(g1["state_proj"]["kernel"])) == 0.0
    assert g1["PaliGemma"]["llm"]["w"].dtype == jnp.bfloat16, "masking must not promote dtypes"

    u1 = _train.staged_mask_updates(staged, 0, tree)
    assert float(jnp.sum(u1["robottt_fast"]["proj_q"]["kernel"])) == 12.0  # 4 leaves x 3.0
    assert float(jnp.sum(u1["state_proj"]["kernel"])) == 0.0

    g2 = _train.staged_mask_grads(staged, 5, tree)  # phase 2: everything passes through unscaled
    u2 = _train.staged_mask_updates(staged, 5, tree)
    for masked in (g2, u2):
        assert float(jnp.sum(masked["PaliGemma"]["llm"]["w"])) == 3.0
        assert float(jnp.sum(masked["robottt_fast"]["proj_q"]["kernel"])) == 4.0


def test_staged_boundary_needs_no_recompile():
    """The phase test is a traced compare on state.step -> one compiled step function for the run."""
    staged = _config.StagedRecipe(phase1_steps=3, phase1_lr_scale_robottt=10.0)
    f = jax.jit(lambda step, x: _train.staged_mask_updates(staged, step, x))
    x = {"robottt_fast": jnp.ones((2,)), "backbone": jnp.ones((2,))}
    lowered = [f(jnp.asarray(s, jnp.int32), x) for s in (0, 2, 3, 9)]
    assert [float(jnp.sum(o["robottt_fast"])) for o in lowered] == [20.0, 20.0, 2.0, 2.0]
    assert [float(jnp.sum(o["backbone"])) for o in lowered] == [0.0, 0.0, 2.0, 2.0]
    assert f._cache_size() == 1, "the boundary must not trigger a re-jit"


# ---------------------- A7 guardrails ----------------------


def test_staged_config_validation():
    with pytest.raises(ValueError, match="phase1_freeze"):
        _config.StagedRecipe(phase1_steps=10, phase1_freeze="head")
    with pytest.raises(ValueError, match="phase1_lr_scale_robottt"):
        _config.StagedRecipe(phase1_steps=10, phase1_lr_scale_robottt=0.0)
    with pytest.raises(ValueError, match="staged requires model.robottt"):
        _train_config(robottt=False, staged=_config.StagedRecipe(phase1_steps=10))


def test_staged_phase1_steps_zero_is_still_a_hard_error():
    """The nonsensical case stays fatal — clamping only applies to phase1_steps >= 1."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="phase1_steps"):
            _config.StagedRecipe(phase1_steps=bad)


def test_canary_clamps_phase1_to_the_whole_run_instead_of_raising(caplog):
    """Attempt-1 fix: `WSM_MAX_STEPS=1` + phase1_steps=15000 used to raise, so the staged recipe
    could never be canaried — the one cheap chance to catch the phase-1 divergence. Now it clamps."""
    with caplog.at_level("WARNING"):
        cfg = _train_config(robottt=True, staged=_config.StagedRecipe(phase1_steps=15000), num_train_steps=1)
    assert cfg.staged.phase1_steps == 1, "phase 1 must cover the whole 1-step canary run"
    assert "canary: staged phase1 clamped to 1 steps" in caplog.text
    # ... and the clamped recipe really is in phase 1 on the step the canary runs.
    assert float(_train._staged_phase1_indicator(cfg.staged, 0)) == 1.0
    # A run long enough for both phases is untouched (no clamp, no warning).
    caplog.clear()
    ok = _train_config(robottt=True, staged=_config.StagedRecipe(phase1_steps=100), num_train_steps=60000)
    assert ok.staged.phase1_steps == 100 and "clamped" not in caplog.text


# ---------------------- divergence tripwire (all Stage-Q runs) ----------------------


def _reduced(loss=1.0, grad_norm=1.0, param_norm=1.0):
    """A logging-interval `reduced_info` as the train loop builds it (already host-side)."""
    return {"loss": np.float32(loss), "grad_norm": np.float32(grad_norm), "param_norm": np.float32(param_norm)}


def _nan_params():
    """Tiny params tree in the nnx.State shape the mask helpers select on: NaN in the subtree only."""
    return {
        "robottt_fast": {"proj_q": {"kernel": jnp.array([1.0, jnp.nan])}},
        "PaliGemma": {"llm": {"kernel": jnp.array([1.0, 2.0])}},
    }


def test_nonfinite_tripwire_aborts_after_exactly_three_logged_intervals(capsys):
    streak = 0
    # Two strikes warn and keep going (a single bad interval must not kill a long run).
    for expected in (1, 2):
        streak = _train.check_nonfinite_metrics(
            8000 + 100 * expected, _reduced(loss=float("nan")), streak, params=_nan_params()
        )
        assert streak == expected
    # The third consecutive one is fatal, nonzero, and names the offending group.
    with pytest.raises(SystemExit) as exc:
        _train.check_nonfinite_metrics(8200, _reduced(loss=float("nan")), streak, params=_nan_params())
    assert exc.value.code == 1
    msg = capsys.readouterr().err
    assert "step=8200" in msg and "loss" in msg
    assert "robottt_fast" in msg and "backbone" not in msg.split("param groups=")[1].split("]")[0]


def test_nonfinite_tripwire_names_the_backbone_when_the_backbone_is_the_broken_one():
    params = {
        "robottt_fast": {"proj_q": {"kernel": jnp.array([1.0, 2.0])}},
        "PaliGemma": {"llm": {"kernel": jnp.array([jnp.inf, 2.0])}},
    }
    assert _train._nonfinite_param_groups(params, "robottt_fast") == ["backbone"]
    assert _train._nonfinite_param_groups(_nan_params(), "robottt_fast") == ["robottt_fast"]
    both = {"robottt_fast": jnp.array([jnp.nan]), "PaliGemma": jnp.array([jnp.nan])}
    assert _train._nonfinite_param_groups(both, "robottt_fast") == ["backbone", "robottt_fast"]
    # A healthy tree names nothing.
    healthy = {"robottt_fast": jnp.array([1.0]), "PaliGemma": jnp.array([2.0])}
    assert _train._nonfinite_param_groups(healthy, "robottt_fast") == []


def test_nonfinite_tripwire_is_silent_and_resets_on_healthy_intervals():
    assert _train.check_nonfinite_metrics(100, _reduced(), 0, params=_nan_params()) == 0
    # A recovered interval clears the streak, so 2 bad + 1 good + 2 bad never aborts.
    streak = 0
    for info in (_reduced(grad_norm=float("inf")), _reduced(grad_norm=float("inf"))):
        streak = _train.check_nonfinite_metrics(1, info, streak, params=_nan_params())
    assert streak == 2
    streak = _train.check_nonfinite_metrics(2, _reduced(), streak, params=_nan_params())
    assert streak == 0
    for info in (_reduced(loss=float("nan")), _reduced(loss=float("nan"))):
        streak = _train.check_nonfinite_metrics(3, info, streak, params=_nan_params())
    assert streak == 2  # still alive


def test_nonfinite_tripwire_watches_every_logged_metric_and_no_others():
    assert _train._nonfinite_metrics(_reduced()) == []
    assert _train._nonfinite_metrics(_reduced(param_norm=float("nan"))) == ["param_norm"]
    assert _train._nonfinite_metrics({"loss": np.nan, "grad_norm": np.inf}) == ["loss", "grad_norm"]
    # staged_phase1 / throughput keys are not diagnostics of divergence.
    assert _train._nonfinite_metrics({"staged_phase1": np.float32(1.0), "loss": np.float32(0.5)}) == []
    assert _train.NONFINITE_ABORT_STRIKES == 3


# ============================== shipped recipes ==============================


def _view(path):
    from utils import load_train_config

    return load_train_config(path)


def _seq_common():
    from vla_training.train.train_base import _pi05_seq_common as m

    return m


def test_shipped_a6_recipe_declares_iid_and_q0_only():
    m, cfg = _seq_common(), _view(Q0_IID_YAML)
    assert m.iid_window_steps(cfg) is True
    assert m.staged_block(cfg) is None
    assert cfg.model["stage_q_arm"] == "q0"
    m.check_declared_arm(cfg, m.derive_arm(False, False))  # Q0 launch is allowed
    for bad in (m.derive_arm(True, False), m.derive_arm(False, True), m.derive_arm(True, True)):
        with pytest.raises(ValueError, match="stage_q_arm"):
            m.check_declared_arm(cfg, bad)


def test_shipped_a7_recipe_declares_the_staged_block_and_q2_only():
    m, cfg = _seq_common(), _view(Q2_STAGED_YAML)
    assert m.iid_window_steps(cfg) is False
    # attempt 2: the LR scale came down from 10.0 (which diverged at step 8200) to 2.0.
    assert m.staged_block(cfg) == {
        "phase1_steps": 15000,
        "phase1_freeze": "backbone",
        "phase1_lr_scale_robottt": 2.0,
    }
    assert cfg.model["stage_q_arm"] == "q2"
    assert m.staged_block(cfg)["phase1_steps"] < int(cfg.train["max_steps"])
    m.check_declared_arm(cfg, m.derive_arm(True, False))  # Q2 launch is allowed
    with pytest.raises(ValueError, match="stage_q_arm"):
        m.check_declared_arm(cfg, m.derive_arm(False, False))
    # The block round-trips into the validated fork dataclass.
    assert _config.StagedRecipe(**m.staged_block(cfg)).subtree == "robottt_fast"


def test_shared_stage_q_recipe_keeps_both_variants_off():
    m, cfg = _seq_common(), _view(SHARED_YAML)
    assert m.iid_window_steps(cfg) is False and m.staged_block(cfg) is None
    m.check_declared_arm(cfg, m.derive_arm(True, True))  # undeclared arm => no constraint
    # The two variant recipes differ from the shared one ONLY in the documented keys.
    import yaml

    shared, a6, a7 = (yaml.safe_load(p.read_text()) for p in (SHARED_YAML, Q0_IID_YAML, Q2_STAGED_YAML))
    naming = {"model.config_name", "model.stage_q_arm", "train.exp_name", "train.output_dir"}

    def diff(a, b, prefix=""):
        out = set()
        for k in set(a) | set(b):
            if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
                out |= diff(a[k], b[k], f"{prefix}{k}.")
            elif a.get(k) != b.get(k):
                out.add(f"{prefix}{k}")
        return out

    assert diff(shared, a6) - naming == {"data.iid_window_steps"}
    assert diff(shared, a7) - naming == {"staged"}


def test_staged_block_rejects_unknown_keys():
    m = _seq_common()
    cfg = types.SimpleNamespace(raw={"staged": {"phase1_steps": 10, "phase_1_lr": 3}})
    with pytest.raises(ValueError, match="unknown staged keys"):
        m.staged_block(cfg)
    with pytest.raises(ValueError, match="requires phase1_steps"):
        m.staged_block(types.SimpleNamespace(raw={"staged": {"phase1_freeze": "backbone"}}))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} PASSED")
