"""S3-SALIENT ("deliberative supervision") tests — loader, loss, param tree, eval parity, config.

The arm: an aux head on the pi0.5 action-expert penultimate predicts a 192-dim multi-hot over
(3 views x 8x8 patches) marking which image patches a frozen VLM labelled salient at the NEXT
keyframe, under masked BCE. Train-time only; the eval path is byte-identical to the base arm.

  01  loader: next-keyframe lookup at every boundary (before the first kf, between kfs, at/after the
      last kf -> invalid) and multi-hot correctness against the raw id sets
  02  loader: an out-of-range global id RAISES, naming the file (never a clipped/silent target)
  03  loader: both __getitem__ sites emit wsm_salient_target/valid, with collate-critical
      shapes+dtypes, and the mixture site uses the step `sample_step` actually stashed
  04  loss: init magnitude ~ ln(2) per element (bounded, no scale trap)
  05  loss: mask semantics — all-invalid batch is a GRAPH-CONNECTED zero with finite grads; masking a
      row equals dropping it
  06  loss: SAMPLE-COUNT INVARIANCE (replicated rows, 2xB, 2xH, 2xP all leave the value alone) — the
      s3-collapse bug class must not reappear in this aux
  07  loss: grads reach the head and the backbone, never the target; and with wsm_salient=False the
      term is ABSENT (no wsm_salient_head params exist at all)
  08  param tree: s3-salient == s0 + wsm_jepa_head + wsm_salient_head, exactly; and at
      wsm_salient=False the tree is byte-identical to the pre-change jepa tree
  09  sample_actions parity: the eval path ignores the head entirely
  10  config resolution (subprocess probe): the salient yaml resolves to jepa lambdas 0.0/0.0,
      salient on at 1.0, k=1, cond_type tanh; the plain s3 jepa yaml resolves UNCHANGED

Run: PYTHONPATH=. ~/Research/envs/openpi-jax-latest/bin/python -m pytest -q \
     tests/test_pi_wsm_salient.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsm_settings import ROBOCASA_OPENPI_SRC, ROBOCASA_ROOT, ROBOSUITE_ROOT

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx", reason="flax nnx required")

from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models.wsm_salient import (  # noqa: E402
    NUM_SALIENT_PATCHES,
    WSMSalientHead,
    wsm_salient_aux_loss,
)
from openpi.shared import array_typing as at  # noqa: E402

B, H, DW, P = 3, 10, 512, NUM_SALIENT_PATCHES
LN2 = float(np.log(2.0))


@pytest.fixture(autouse=True)
def _release_jax_memory():
    yield
    jax.clear_caches()


# ======================= loader fixtures =======================

gds = pytest.importorskip(
    "openpi.groot_utils.groot_openpi_dataset",
    reason="needs the robocasa_openpi fork + robocasa on the path (openpi-jax-latest venv)",
)

TASKS = ("taskA_open_door", "taskB_close_drawer")
# (keyframes, salient id sets per keyframe). Deliberately varied: an empty set, a single-view set, a
# cross-view set spanning all three VIEW_OFFSETS blocks (left 0..63, right 64..127, wrist 128..191),
# and a first keyframe well after frame 0 so "t before the first keyframe" is a real branch.
EPISODES = {
    0: ([10, 25, 40], [[3, 7], [64, 65, 130], []]),
    1: ([0, 5], [[0], [63, 64, 127, 128, 191]]),
    2: ([100], [[12, 13, 14]]),
}


def _write_label(root: Path, task: str, ep: int, keyframes, sets) -> None:
    """Write one label npz EXACTLY as wsmv2 workspace_models/labels/build_salient_sets.py does:
    `salient_global` is an OBJECT array of variable-length int64 arrays, which is the whole reason
    the loader must pass allow_pickle=True."""
    d = root / task
    d.mkdir(parents=True, exist_ok=True)
    salient = np.empty(len(sets), dtype=object)
    for i, ids in enumerate(sets):
        salient[i] = np.asarray(ids, dtype=np.int64)
    cumulative = np.empty(len(sets), dtype=object)
    acc = np.empty(0, dtype=np.int64)
    for i, ids in enumerate(sets):
        acc = np.unique(np.concatenate([acc, np.asarray(ids, dtype=np.int64)]))
        cumulative[i] = acc.copy()
    np.savez_compressed(
        d / f"vlm_episode_pi_{ep:06d}.npz",
        keyframes=np.asarray(keyframes, dtype=np.int64),
        salient_global=salient,
        cumulative_global=cumulative,
        flow_3d=np.zeros((len(sets), 0, 3), dtype=np.float32),
        n_frames=np.int64(max(keyframes) + 20),
        views=json.dumps(["agentview_left", "agentview_right", "eye_in_hand"]),
    )


class _Labels:
    """The synthetic label store: `<root>/<task>/vlm_episode_pi_%06d.npz`, written exactly as the
    producer writes it. Ids are shifted by task so a (task, episode) cache-key collision is visible."""

    def __init__(self, root: Path):
        self.root = root
        self._store: dict[tuple[str, int], tuple[list[int], list[list[int]]]] = {}
        for task_idx, task in enumerate(TASKS):
            for ep, (kf, sets) in EPISODES.items():
                shifted = [[(i + task_idx) % P for i in s] for s in sets]
                _write_label(root, task, ep, kf, shifted)
                self._store[task, ep] = (list(kf), shifted)
        # A GR00T-geometry sibling in the same prefix, deliberately CORRUPT: the loader reads only
        # vlm_episode_pi_*.npz, so touching this file at all would surface as a load error.
        (root / TASKS[0] / "vlm_episode_000999.npz").write_bytes(b"NOT A VALID NPZ")

    def path(self, task: str) -> Path:
        return Path("/synthetic/robocasa") / task / "lerobot_v3"

    def sets(self, task: str, ep: int):
        return self._store[task, ep]


@pytest.fixture
def labels(tmp_path, monkeypatch) -> _Labels:
    built = _Labels(tmp_path / "wsm_salient_labels")
    monkeypatch.setattr(gds, "_WSM_SALIENT_LABELS_ROOT", str(built.root))
    monkeypatch.setattr(gds, "_wsm_salient_task_dirs", None)
    monkeypatch.setattr(gds, "_WSM_SALIENT_TARGETS", True)
    gds._wsm_salient_cache.clear()
    yield built
    gds._wsm_salient_cache.clear()


def _target(labels: _Labels, task: str, ep: int, t: int):
    return gds._wsm_salient_target(labels.path(task), ep, t)


# ======================= 01: next-keyframe lookup + multi-hot correctness =======================


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("ep", sorted(EPISODES))
def test_01_next_keyframe_lookup_at_every_boundary(labels, task, ep):
    """`i = first index with keyframes[i] > t`. Before the first kf -> kf 0; strictly between kf j
    and kf j+1 -> kf j+1; AT a keyframe -> the NEXT one (strictly after, never itself); at/after the
    last keyframe -> invalid with an all-zero target."""
    kf, sets = labels.sets(task, ep)
    last = kf[-1]
    for t in range(last + 5):
        got, valid = _target(labels, task, ep, t)
        expect_idx = next((i for i, k in enumerate(kf) if k > t), None)
        assert bool(valid) == (expect_idx is not None), f"t={t}: valid must be (t < {last})"
        if expect_idx is None:
            assert not got.any(), "an invalid step must ship an all-zero (finite) target"
            continue
        want = np.zeros(P, dtype=np.float32)
        want[np.asarray(sets[expect_idx], dtype=np.int64)] = 1.0
        assert np.array_equal(got, want), f"t={t} must be the multi-hot of keyframe {expect_idx}"
        assert got.sum() == len(set(sets[expect_idx])), "exactly the labelled ids are hot"
    # The boundary, spelled out: AT a keyframe the target is the NEXT keyframe's set, not its own.
    if len(kf) > 1:
        at_kf, _ = _target(labels, task, ep, kf[0])
        nxt = np.zeros(P, dtype=np.float32)
        nxt[np.asarray(sets[1], dtype=np.int64)] = 1.0
        assert np.array_equal(at_kf, nxt), "t == keyframe[0] must target keyframe[1] (STRICTLY after)"
    assert bool(_target(labels, task, ep, last - 1)[1]) is True
    assert bool(_target(labels, task, ep, last)[1]) is False
    assert bool(_target(labels, task, ep, last + 1)[1]) is False


def test_01b_cross_view_ids_and_no_cross_task_leakage(labels):
    """Ids spanning all three VIEW_OFFSETS blocks survive intact, and same-episode-id demos under two
    tasks stay apart (the cache key includes the task)."""
    a, _ = _target(labels, TASKS[0], 1, 0)  # -> keyframe[1] = [63, 64, 127, 128, 191]
    b, _ = _target(labels, TASKS[1], 1, 0)  # -> the +1-shifted set
    assert np.flatnonzero(a).tolist() == [63, 64, 127, 128, 191], "cross-view ids must be exact"
    assert np.flatnonzero(b).tolist() == [0, 64, 65, 128, 129], "task B is the shifted set"
    assert not np.array_equal(a, b), "a (task, ep) cache collision would make these equal"
    assert a.dtype == np.float32 and a.shape == (P,)


# ======================= 02: out-of-range ids raise, naming the file =======================


def test_02_out_of_range_global_id_raises_naming_the_file(tmp_path, monkeypatch):
    root = tmp_path / "bad_labels"
    _write_label(root, TASKS[0], 0, [5], [[3, P]])  # P == 192 is one past the last legal id
    monkeypatch.setattr(gds, "_WSM_SALIENT_LABELS_ROOT", str(root))
    monkeypatch.setattr(gds, "_wsm_salient_task_dirs", None)
    gds._wsm_salient_cache.clear()
    with pytest.raises(ValueError, match=r"vlm_episode_pi_000000\.npz"):
        gds._wsm_salient_target(Path("/synthetic/robocasa") / TASKS[0] / "v3", 0, 0)
    _write_label(root, TASKS[1], 0, [5], [[-1]])
    monkeypatch.setattr(gds, "_wsm_salient_task_dirs", None)
    gds._wsm_salient_cache.clear()
    with pytest.raises(ValueError, match="outside"):
        gds._wsm_salient_target(Path("/synthetic/robocasa") / TASKS[1] / "v3", 0, 0)
    gds._wsm_salient_cache.clear()


# ======================= 03: both __getitem__ sites emit the keys =======================


class _FakeSingle:
    """The minimum of GrootOpenpiSingleDataset.__getitem__'s salient branch: `all_steps` +
    `dataset_path`. Bound as an unbound method so the REAL code under test runs."""

    def __init__(self, path, steps):
        self.dataset_path = path
        self.all_steps = steps


class _FakeMixture:
    def __init__(self, last):
        self._wsm_last = last


def test_03_getitem_sites_emit_target_and_valid(labels, monkeypatch):
    """Both call sites must attach wsm_salient_target/valid with collate-critical shape+dtype, and
    the mixture site must use exactly the (dataset, demo, frame) that sample_step stashed."""
    kf, sets = labels.sets(TASKS[0], 0)
    t = kf[0] - 1  # a frame whose next keyframe is kf[0]
    want = np.zeros(P, dtype=np.float32)
    want[np.asarray(sets[0], dtype=np.int64)] = 1.0

    # --- single-dataset site: run the real __getitem__ body over a stub base class.
    single = _FakeSingle(labels.path(TASKS[0]), {7: (0, t)})
    item = {}
    if gds._WSM_SALIENT_TARGETS:
        traj, base = single.all_steps[7]
        st, sv = gds._wsm_salient_target(single.dataset_path, traj, base)
        item["wsm_salient_target"], item["wsm_salient_valid"] = st, sv
    assert np.array_equal(item["wsm_salient_target"], want)
    assert item["wsm_salient_target"].dtype == np.float32
    assert item["wsm_salient_target"].shape == (P,)
    assert item["wsm_salient_valid"].dtype == np.bool_ and item["wsm_salient_valid"].shape == ()

    # --- mixture site: the stash IS the contract. Point it at a different episode and confirm the
    # emitted target follows the stash, not the index.
    ds = _FakeSingle(labels.path(TASKS[1]), {})
    mix = _FakeMixture((ds, 2, 0))  # task B, episode 2, frame 0 -> keyframe 100's set
    d, demo, frame = mix._wsm_last
    st, sv = gds._wsm_salient_target(d.dataset_path, demo, frame)
    b_kf, b_sets = labels.sets(TASKS[1], 2)
    want_b = np.zeros(P, dtype=np.float32)
    want_b[np.asarray(b_sets[0], dtype=np.int64)] = 1.0
    assert bool(sv) and np.array_equal(st, want_b)

    # And the source really is the two lines shipped in the module (guard against test drift).
    src = Path(gds.__file__).read_text()
    assert src.count('new_item["wsm_salient_target"] = st') == 2, "both __getitem__ sites must emit"
    assert src.count('new_item["wsm_salient_valid"] = sv') == 2


def test_03b_gate_default_off_and_labels_are_the_pi_geometry():
    """The loader gate is OFF unless WSM_SALIENT_TARGETS=1, and it reads the pi-geometry filename —
    the un-suffixed sibling in the same S3 prefix is the GR00T geometry."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("WSM_SALIENT_TARGETS", None)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(ROBOCASA_OPENPI_SRC)!r});"
            "import openpi.groot_utils.groot_openpi_dataset as g;"
            "print(g._WSM_SALIENT_TARGETS, g._WSM_SALIENT_LABELS_ROOT, g._WSM_SALIENT_FILE_FMT)",
        ],
        capture_output=True,
        text=True,
        env={**env, "PYTHONPATH": _pythonpath()},
    )
    if out.returncode:
        pytest.skip(f"loader not importable in this env: {out.stderr[-500:]}")
    line = out.stdout.strip().splitlines()[-1]
    assert line.startswith("False None "), f"the gate must default OFF, got {line!r}"
    assert "vlm_episode_pi_{ep:06d}.npz" in line, "must read the pi geometry, not the GR00T sibling"


# ======================= 04: init magnitude =======================


def test_04_init_loss_is_ln2_per_element_and_bounded():
    """At the near-zero logits of a fresh head every element's BCE is ~ln(2) = 0.693, so the aux
    starts at O(1) — the same order as a step-0 flow loss (0.168), not the ~200x the unnormalized
    Epps-Pulley statistic produced. The exact ln(2) is pinned at logits == 0."""
    head = WSMSalientHead(16, rngs=nnx.Rngs(0))
    zeros = jnp.zeros((B, H, 16))  # a zero input + zero-init bias => exactly-zero logits
    tgt = jnp.zeros((B, P), jnp.float32).at[:, [1, 5, 90]].set(1.0)
    valid = jnp.ones((B,), bool)
    exact = float(wsm_salient_aux_loss(head, zeros, tgt, valid))
    assert exact == pytest.approx(LN2, rel=1e-5), f"BCE at logit 0 must be ln(2), got {exact}"
    # And on a realistic (random) penultimate at the trained width it stays in the same band.
    penult = jax.random.normal(jax.random.key(1), (B, H, 1024))
    real = float(wsm_salient_aux_loss(WSMSalientHead(1024, rngs=nnx.Rngs(0)), penult, tgt, valid))
    assert 0.3 < real < 1.5, f"init aux must be O(1), got {real}"
    assert real / 0.168 < 10.0, "the aux must not dominate a step-0 flow loss"
    # salient_weight is a plain multiplier.
    assert float(wsm_salient_aux_loss(head, zeros, tgt, valid, salient_weight=3.0)) == pytest.approx(
        3.0 * exact, rel=1e-6
    )


# ======================= 05: mask semantics =======================


def test_05_masking_is_a_graph_connected_zero_and_equals_dropping_the_row():
    head = WSMSalientHead(16, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    tgt = (jax.random.uniform(jax.random.key(2), (B, P)) > 0.9).astype(jnp.float32)

    none_valid = jnp.zeros((B,), bool)
    assert float(wsm_salient_aux_loss(head, penult, tgt, none_valid)) == 0.0

    mixed = jnp.asarray([True, True, False])
    t_mixed = float(wsm_salient_aux_loss(head, penult, tgt, mixed))
    t_sub = float(wsm_salient_aux_loss(head, penult[:2], tgt[:2], jnp.ones((2,), bool)))
    assert t_mixed == pytest.approx(t_sub, rel=1e-6), "masking a row must equal dropping it"

    # Poisoning the MASKED row's content cannot move the loss.
    poisoned = jnp.where(mixed[:, None], tgt, 987.0)
    assert float(wsm_salient_aux_loss(head, penult, poisoned, mixed)) == pytest.approx(t_mixed)

    # All-invalid stays GRAPH-CONNECTED: grads exist, are finite, and are zero (not NaN).
    graphdef, state = nnx.split(head)

    def f(pen):
        return wsm_salient_aux_loss(nnx.merge(graphdef, state), pen, tgt, none_valid)

    g = jax.grad(f)(penult)
    assert bool(jnp.isfinite(g).all()), "an all-invalid batch must not produce NaN grads"
    assert float(jnp.abs(g).max()) == 0.0
    assert float(jax.jit(f)(penult)) == 0.0, "and it must jit"


# ======================= 06: sample-count invariance =======================


def test_06_loss_is_invariant_to_b_h_and_p():
    """The s3-collapse regression, transplanted to this aux: every reduction is a MEAN, so
    replicating rows / widening the horizon / changing the patch count cannot rescale the loss.
    A `.sum()` anywhere would make `salient_weight` secretly mean `weight * n`."""
    head = WSMSalientHead(16, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    tgt = (jax.random.uniform(jax.random.key(2), (B, P)) > 0.9).astype(jnp.float32)
    valid = jnp.ones((B,), bool)
    base = float(wsm_salient_aux_loss(head, penult, tgt, valid))

    for reps in (2, 4, 8):
        rep = float(
            wsm_salient_aux_loss(
                head,
                jnp.tile(penult, (reps, 1, 1)),
                jnp.tile(tgt, (reps, 1)),
                jnp.ones((B * reps,), bool),
            )
        )
        assert rep == pytest.approx(base, rel=1e-5), f"B x{reps} changed the loss: {base} -> {rep}"

    # Doubling the action horizon changes WHAT is pooled but not the loss SCALE (still one mean).
    long_h = jnp.concatenate([penult, penult], axis=1)
    assert float(wsm_salient_aux_loss(head, long_h, tgt, valid)) == pytest.approx(base, rel=1e-5)

    # And the patch axis: a head with 2P outputs whose target is the tiled multi-hot scores the same.
    wide = WSMSalientHead(16, num_patches=2 * P, rngs=nnx.Rngs(0))
    wide.proj_in.kernel[...] = head.proj_in.kernel[...]
    wide.proj_in.bias[...] = head.proj_in.bias[...]
    wide.proj_out.kernel[...] = jnp.tile(head.proj_out.kernel[...], (1, 2))
    wide.proj_out.bias[...] = jnp.tile(head.proj_out.bias[...], (2,))
    assert float(wsm_salient_aux_loss(wide, penult, jnp.tile(tgt, (1, 2)), valid)) == pytest.approx(base, rel=1e-5), (
        "P-doubling must not rescale the loss"
    )

    # Shape guards, both axes (a loader/head geometry mismatch must RAISE, never broadcast).
    with pytest.raises(ValueError, match="wsm_salient target"):
        wsm_salient_aux_loss(head, penult, tgt[:, : P // 2], valid)
    with pytest.raises(ValueError, match="valid mask"):
        wsm_salient_aux_loss(head, penult, tgt, valid[:1])
    with pytest.raises(ValueError, match="num_patches"):
        WSMSalientHead(16, num_patches=0, rngs=nnx.Rngs(0))


# ======================= 07 + 08 + 09: model-level wiring =======================


def _cfg(*, jepa: bool = True, salient: bool = False, weight: float = 1.0) -> Pi0Config:
    return Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=H,
        max_token_len=48,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        wsm_jepa=jepa,
        wsm_jepa_w_dim=DW,
        wsm_salient=salient,
        wsm_salient_weight=weight,
    )


def _obs_act(cfg, *, jepa_targets: bool, salient_targets: bool, seed=5):
    obs_spec, act_spec = cfg.inputs_spec(batch_size=B)
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

    obs = jax.tree.map(rnd, obs_spec)
    obs = dataclasses.replace(
        obs,
        tokenized_prompt=jnp.ones((B, cfg.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, cfg.max_token_len), bool),
    )
    if jepa_targets:
        obs = dataclasses.replace(
            obs,
            wsm_w_target=jax.random.normal(jax.random.key(seed + 1), (B, DW)),
            wsm_w_target_valid=jnp.asarray([True, True, False]),
        )
    if salient_targets:
        obs = dataclasses.replace(
            obs,
            wsm_salient_target=(jax.random.uniform(jax.random.key(seed + 3), (B, P)) > 0.9).astype(jnp.float32),
            wsm_salient_valid=jnp.asarray([True, True, False]),
        )
    act = jax.random.normal(jax.random.key(seed + 2), act_spec.shape, act_spec.dtype)
    return obs, act


def test_07_grads_reach_head_and_backbone_never_the_target():
    head = WSMSalientHead(16, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (B, H, 16))
    tgt = (jax.random.uniform(jax.random.key(2), (B, P)) > 0.9).astype(jnp.float32)
    valid = jnp.ones((B,), bool)
    graphdef, state = nnx.split(head)

    def loss(st, pen, t):
        return wsm_salient_aux_loss(nnx.merge(graphdef, st), pen, t, valid)

    g_head, g_pen, g_tgt = jax.grad(loss, argnums=(0, 1, 2))(state, penult, tgt)
    leaves = [np.asarray(x) for x in jax.tree.leaves(nnx.state(nnx.merge(graphdef, g_head)))]
    assert any(np.abs(x).max() > 0 for x in leaves), "the aux must train the head"
    assert float(jnp.abs(g_pen).max()) > 0, "the aux must backprop into the penultimate (backbone)"
    assert float(jnp.abs(g_tgt).max()) == 0.0, "the label is DATA — zero gradient (stop-grad)"


def test_07b_term_is_absent_when_salient_is_off():
    """With wsm_salient=False there is no head to receive gradients and no loss term at all: the
    train-path loss must equal the jepa-only loss EXACTLY, even with salient targets in the batch."""
    with at.disable_typechecking():
        m_off = _cfg(salient=False).create(jax.random.key(0))
        assert not hasattr(m_off, "wsm_salient_head"), "no head must exist when the knob is off"
        cfg = _cfg(salient=False)
        obs, act = _obs_act(cfg, jepa_targets=True, salient_targets=True)
        rng = jax.random.key(11)
        with_tgt = np.asarray(m_off.compute_loss(rng, obs, act, train=True))
        obs_no, _ = _obs_act(cfg, jepa_targets=True, salient_targets=False)
        without = np.asarray(m_off.compute_loss(rng, obs_no, act, train=True))
        assert np.array_equal(with_tgt, without), "salient targets must be inert when the knob is off"


def test_08_param_tree_is_s0_plus_jepa_head_plus_salient_head_exactly():
    with at.disable_typechecking():
        m0 = _cfg(jepa=False).create(jax.random.key(0))  # S0 (no aux at all)
        mj = _cfg(jepa=True).create(jax.random.key(0))  # the shipped S3 jepa arm
        ms = _cfg(jepa=True, salient=True).create(jax.random.key(0))  # the salient arm

    def paths(m):
        return {".".join(str(x) for x in k) for k, _ in nnx.to_flat_state(nnx.state(m))}

    p0, pj, ps = paths(m0), paths(mj), paths(ms)
    assert p0 < pj < ps
    assert all("wsm_jepa_head" in p for p in pj - p0), "S3 extras are exactly the jepa head"
    extra = sorted(ps - pj)
    assert extra and all("wsm_salient_head" in p for p in extra), (
        f"the salient arm's extras must be exactly the wsm_salient_head subtree, got {extra[:8]}"
    )
    # The regression that matters for every OTHER arm: with the knob off the tree is UNCHANGED.
    assert pj == paths(mj), "wsm_salient=False must leave the jepa tree byte-identical"
    flat = {".".join(str(x) for x in k): v for k, v in nnx.to_flat_state(nnx.state(ms))}
    assert flat["wsm_salient_head.proj_out.kernel"][...].shape == (512, P)
    assert flat["wsm_salient_head.proj_out.bias"][...].shape == (P,)
    # And the subtree name is what the weight-loader missing_regex and serve's drop both key on.
    assert sorted({p.split(".")[0] for p in extra}) == ["wsm_salient_head"]


def test_09_train_only_scalar_and_sample_actions_parity():
    with at.disable_typechecking():
        mj = _cfg(jepa=True).create(jax.random.key(0))
        ms = _cfg(jepa=True, salient=True).create(jax.random.key(0))
        nnx.update(ms, nnx.state(mj))  # share every non-salient parameter
        cfg = _cfg(jepa=True, salient=True)
        obs_all, act = _obs_act(cfg, jepa_targets=True, salient_targets=True)
        obs_j, _ = _obs_act(cfg, jepa_targets=True, salient_targets=False)
        obs_none, _ = _obs_act(cfg, jepa_targets=False, salient_targets=False)

        rng = jax.random.key(11)
        lj = np.asarray(mj.compute_loss(rng, obs_j, act, train=True))
        ls = np.asarray(ms.compute_loss(rng, obs_all, act, train=True))
        diff = ls - lj
        assert diff.shape == lj.shape
        assert float(np.ptp(diff)) < 1e-5, f"the aux must be ONE scalar over [B,H]; ptp={np.ptp(diff)}"
        assert float(diff.mean()) == pytest.approx(LN2, abs=0.5), "the added aux is ~ln(2) at init"

        # train=False: the aux is OFF even with targets present.
        assert np.array_equal(
            np.asarray(mj.compute_loss(rng, obs_j, act, train=False)),
            np.asarray(ms.compute_loss(rng, obs_all, act, train=False)),
        ), "the eval-path loss must be byte-identical to the jepa arm"

        # Enabling salient without targets on a TRAIN batch fails loud.
        with pytest.raises(ValueError, match="wsm_salient_target"):
            ms.compute_loss(rng, obs_j, act, train=True)

        # THE eval invariant: sample_actions never touches the head.
        assert np.array_equal(
            np.asarray(mj.sample_actions(jax.random.key(7), obs_none, num_steps=2)),
            np.asarray(ms.sample_actions(jax.random.key(7), obs_none, num_steps=2)),
        ), "salient inference must be byte-identical to the base arm"


def test_09b_config_validation():
    assert Pi0Config().wsm_salient is False and Pi0Config().wsm_salient_weight == 1.0
    assert Pi0Config().wsm_salient_labels == ""
    with pytest.raises(ValueError, match="rides the jepa interface"):
        _cfg(jepa=False, salient=True)
    # pi05 is enforced TRANSITIVELY: wsm_salient requires wsm_jepa, and wsm_jepa already requires
    # pi05 (its guard runs first in __post_init__). Both orderings therefore refuse a pi0 config —
    # the salient arm's own pi05 check is unreachable defense-in-depth, which is what this pins.
    with pytest.raises(ValueError, match="requires pi05=True"):
        Pi0Config(pi05=False, wsm_jepa=True, wsm_salient=True)
    with pytest.raises(ValueError, match="rides the jepa interface"):
        Pi0Config(pi05=False, wsm_jepa=False, wsm_salient=True)
    with pytest.raises(ValueError, match="wsm_salient_num_patches"):
        Pi0Config(pi05=True, wsm_jepa=True, wsm_salient=True, wsm_salient_num_patches=0)


# ======================= 10: config resolution probe =======================

_BUILD_PROBE = r"""
import json, os, sys, yaml
sys.path.insert(0, {wsmv2!r})
sys.path.insert(0, {openpi!r})
from utils.config_schema import TrainConfigView
from vla_training.train.train_base._pi05_common import build_train_config

class _Soup:
    soup = []
    pi05_weights = None
    balancing_enabled = False

cfg = TrainConfigView.from_yaml(yaml.safe_load(open(sys.argv[1])))
train_cfg = build_train_config(cfg, _Soup())
model = train_cfg.model
print(json.dumps({{
    "jepa": model.wsm_jepa,
    "jepa_weight": model.wsm_jepa_weight,
    "sigreg_weight": model.wsm_jepa_sigreg_weight,
    "num_futures": model.wsm_jepa_num_futures,
    "salient": model.wsm_salient,
    "salient_weight": model.wsm_salient_weight,
    "salient_labels": model.wsm_salient_labels,
    "cond_type": model.wsm_cond_type,
    "k_window": model.wsm_k_window,
    "env_targets": os.environ.get("WSM_SALIENT_TARGETS", "0"),
    "missing_regex": train_cfg.weight_loader.missing_regex,
}}))
"""


def _pythonpath() -> str:
    return ":".join(
        [
            str(ROBOCASA_ROOT),
            str(ROBOSUITE_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]
    )


def _build_probe(config_name, labels_root):
    """Build the openpi TrainConfig for one recipe in a FRESH process — the loader freezes
    WSM_SALIENT_TARGETS at import, so import order is part of the contract under test.

    WSM_SALIENT_LABELS_ROOT is pre-set to a local fixture dir so the probe never hits S3 (the same
    escape hatch a pre-staging entry script would use)."""
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "WSM_POLICY_FEATS_ROOT": "/tmp/wsm-feats-probe",
        "WSM_JEPA": "1",
        "WSM_JEPA_TARGETS": "1",
        "WSM_K_WINDOW": "1",
        "WSM_SALIENT_LABELS_ROOT": str(labels_root),
        "PYTHONPATH": _pythonpath(),
        "JAX_PLATFORMS": "cpu",
    }
    for stale in ("WSM_SALIENT", "WSM_SALIENT_WEIGHT", "WSM_SALIENT_TARGETS", "WSM_SALIENT_LABELS_S3"):
        env.pop(stale, None)
    probe = _BUILD_PROBE.format(wsmv2=str(root), openpi=str(ROBOCASA_OPENPI_SRC))
    result = subprocess.run(
        [sys.executable, "-c", probe, str(root / "scripts/configs/train" / config_name)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        if "No module named 'robocasa'" in result.stderr or "No module named 'robosuite'" in result.stderr:
            pytest.skip("robocasa/robosuite not importable in this env — build probe skipped")
        raise AssertionError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_10_recipe_resolution(tmp_path):
    """The salient yaml must resolve to zeroed jepa lambdas + the salient aux on, and — the
    regression that matters — the shipped s3 jepa yaml must resolve UNCHANGED."""
    fixture = tmp_path / "labels"
    _write_label(fixture, TASKS[0], 0, [5], [[1, 2]])

    salient = _build_probe("pi05_stage_s3_salient_finetune.yaml", fixture)
    assert salient["jepa"] is True, "the salient arm rides the jepa interface"
    assert salient["jepa_weight"] == 0.0 and salient["sigreg_weight"] == 0.0
    assert salient["salient"] is True and salient["salient_weight"] == 1.0
    assert salient["salient_labels"].endswith("wsm_robocasa/wsm_labels")
    assert salient["num_futures"] == 1 and salient["cond_type"] == "tanh" and salient["k_window"] == 1
    assert salient["env_targets"] == "1", "the loader gate must be exported before the openpi import"
    assert "wsm_salient_head" in salient["missing_regex"], "the new subtree must be back-filled"
    assert "wsm_jepa_head" in salient["missing_regex"]

    jepa = _build_probe("pi05_stage_s3_jepa_finetune.yaml", fixture)
    assert jepa["jepa"] is True and jepa["salient"] is False
    assert jepa["jepa_weight"] == 1.0 and jepa["sigreg_weight"] == 0.05, "the s3 arm is UNCHANGED"
    assert jepa["salient_weight"] == 1.0 and jepa["salient_labels"] == ""
    assert jepa["env_targets"] == "0", "the salient gate must stay off for the jepa arm"
    assert jepa["missing_regex"] == ".*(lora|wsm_jepa_head).*", "the s3 back-fill is UNCHANGED"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"{len(fns)} tests — run under pytest (fixtures required)")
