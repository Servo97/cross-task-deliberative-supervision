"""S3 (JEPA) dataset TARGET-loader tests — the one load-bearing untested link (jul_29 readiness §2).

Contract (doc 12 D4 + `wsm_align.next_at_with_valid`): for a sampled native step t the loader must
ship `wsm_w_target` = the omega row of the FIRST grid frame STRICTLY AFTER t (never omega_t itself),
with `wsm_w_target_valid` False exactly when no strictly-future grid row exists (the terminal tail),
and it must never ship `wsm_w_window` in that mode. Omega rows exist only on the stride-8 cache grid,
so "the omega row of step t+1" means "the next omega-BEARING step"; on a stride-1 grid that is
literally `omega[t+1]`, and on the production stride-8 grid it is the row after the S1/S2 causal
window's newest row (test 05 pins that equivalence, which is the stride/subsample semantics).

Target under test: `openpi.groot_utils.groot_openpi_dataset._wsm_jepa_target` (:173-190) and both
`WSM_JEPA_TARGETS` call sites (:323-326 single, :476-479 mixture).

  01  next row for every non-terminal step (stride-1 grid: exact `omega[t+1]`)
  02  loader == `workspace_models.features.wsm_align.next_at_with_valid` on every native step of every
      grid layout (the loader's reimplementation is otherwise never compared to the reference)
  03  terminal valid boundary; the clamped row is a real finite row (the mask multiplies, not skips)
  04  no cross-episode / cross-task leakage, including under cache eviction and interleaving
  05  stride/subsample: target == the causal window's NEXT grid row, never the current one; and the
      target path ignores WSM_K_WINDOW / demo-CFG z-root
  06  `__getitem__` emits the target keys and NEVER `wsm_w_window` (+ collate-critical shape/dtype)
  07  mixture `__getitem__` target matches the step `sample_step` actually stashed
  08  loader-produced valid flags zero the JEPA contribution of terminal rows in the fork loss
  09  determinism: the target path consumes no RNG; seeded mixture draws reproduce exactly

Run: PYTHONPATH=. ~/Research/envs/openpi-jax-latest/bin/python -m pytest -q \
     tests/test_wsm_jepa_target_loader.py tests/test_pi_wsm_jepa.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsm_settings import ROBOCASA_OPENPI_SRC

sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))

from workspace_models.features import wsm_align  # noqa: E402  (the reference semantics)

gds = pytest.importorskip(
    "openpi.groot_utils.groot_openpi_dataset",
    reason="needs the robocasa_openpi fork + robocasa on the path (openpi-jax-latest venv)",
)

DW = 8  # tiny latent: every assertion here is about WHICH row, not the row's width

# Grid layouts. `fi` = native frame index of each omega row; `T` = native trajectory length.
# stride1 makes "the omega row of step t+1" literal; stride8 is production; irregular/late_start
# probe the off-grid and before-the-first-grid-frame branches.
LAYOUTS = {
    "stride1": (np.arange(12, dtype=np.int64), 12),
    "stride8": (np.arange(0, 48, 8, dtype=np.int64), 44),
    "irregular": (np.array([0, 3, 4, 9, 17, 18], dtype=np.int64), 22),
    "late_start": (np.array([5, 13, 21], dtype=np.int64), 25),
}
# demo_id per layout inside every task dir (fixed so the same ids exist under BOTH tasks -> a
# (task, demo) cache-key collision would be caught by test 04).
DEMOS = {"stride1": 0, "stride8": 1, "irregular": 2, "late_start": 3}
TASKS = ("taskA_open_door", "taskB_close_drawer")


def _rows(task_idx: int, demo_id: int, n_rows: int) -> np.ndarray:
    """[n_rows, DW] fp16 omega rows, globally unique per (task, demo, grid row).

    Values stay under 512 in steps of 0.25 so fp16 storage is EXACT -> every comparison below is
    exact array equality, never a tolerance.
    """
    base = 100.0 * (task_idx + 1) + 20.0 * demo_id
    ramp = 0.25 * np.arange(DW, dtype=np.float32)
    return (base + np.arange(n_rows, dtype=np.float32)[:, None] + ramp[None, :]).astype(np.float16)


class _Feats:
    """The synthetic omega cache: `<root>/<task>/demo_%06d/w.npz` exactly as the producer writes it."""

    def __init__(self, root: Path):
        self.root = root
        self.w: dict[tuple[str, int], np.ndarray] = {}
        self.fi: dict[tuple[str, int], np.ndarray] = {}
        self.length: dict[tuple[str, int], int] = {}
        for task_idx, task in enumerate(TASKS):
            for name, (fi, traj_len) in LAYOUTS.items():
                demo_id = DEMOS[name]
                w = _rows(task_idx, demo_id, len(fi))
                d = root / task / f"demo_{demo_id:06d}"
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "w.npz", w=w, frame_indices=fi.astype(np.int64))
                self.w[task, demo_id] = w
                self.fi[task, demo_id] = fi.astype(np.int64)
                self.length[task, demo_id] = traj_len

    def path(self, task: str) -> Path:
        """A plausible LeRobot dataset path: the loader finds the task by matching a path PART."""
        return Path("/synthetic/robocasa") / task / "lerobot_v3"

    def demo(self, layout: str, task: str = TASKS[0]):
        demo_id = DEMOS[layout]
        return task, demo_id, self.w[task, demo_id], self.fi[task, demo_id], self.length[task, demo_id]


@pytest.fixture
def feats(tmp_path, monkeypatch) -> _Feats:
    built = _Feats(tmp_path / "omega")
    monkeypatch.setattr(gds, "_WSM_FEATS_ROOT", str(built.root))
    monkeypatch.setattr(gds, "_wsm_task_dirs", None)  # recomputed from the patched root
    monkeypatch.setattr(gds, "_WSM_JEPA_TARGETS", True)
    monkeypatch.setattr(gds, "_WSM_Z_ROOT", None)
    monkeypatch.setattr(gds, "_WSM_CURRENT_ONLY", True)
    gds._wsm_demo_cache.clear()
    gds._wsm_z_cache.clear()
    yield built
    gds._wsm_demo_cache.clear()
    gds._wsm_z_cache.clear()


def _target(feats: _Feats, task: str, demo_id: int, t: int):
    return gds._wsm_jepa_target(feats.path(task), demo_id, t)


# ------------------------- 01: the next omega row, for every non-terminal step -------------------------


def test_01_target_is_the_next_omega_row_for_every_nonterminal_step(feats):
    """Stride-1 grid: `wsm_w_target` at step t is EXACTLY omega[t+1] — the contract, unmediated."""
    task, demo_id, w, fi, _ = feats.demo("stride1")
    assert np.array_equal(fi, np.arange(len(w)))  # the layout really is stride-1
    for t in range(len(w) - 1):  # every non-terminal omega-bearing step
        wt, valid = _target(feats, task, demo_id, t)
        assert bool(valid), f"t={t} has a next row and must be valid"
        assert np.array_equal(wt, w[t + 1].astype(np.float32)), f"t={t} must get omega[t+1]"
        assert not np.array_equal(wt, w[t].astype(np.float32)), f"t={t} must NOT get omega[t]"


# ------------------------- 02: parity with the wsm_align reference, all layouts -------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("task", TASKS)
def test_02_loader_matches_wsm_align_reference_on_every_native_step(feats, layout, task):
    """The loader reimplements `next_at_with_valid` (it cannot import wsm_align). Compare them
    step-by-step over the whole trajectory, including the tail past the last grid frame."""
    _, demo_id, w, fi, traj_len = feats.demo(layout, task)
    for t in range(traj_len + 4):  # +4: over-run the trajectory, the clamp must still agree
        wt, valid = _target(feats, task, demo_id, t)
        ref_w, ref_valid = wsm_align.next_at_with_valid(w, fi, t)
        assert np.array_equal(wt, np.asarray(ref_w, dtype=np.float32)), f"{layout} t={t} row mismatch"
        assert bool(valid) == bool(ref_valid), f"{layout} t={t} valid mismatch"
        assert wt.dtype == np.float32 and wt.shape == (DW,)


# ------------------------- 03: terminal-step semantics -------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_03_valid_is_false_exactly_on_the_terminal_tail(feats, layout):
    """valid is True iff a strictly-future grid row exists: True up to fi[-1]-1, False from fi[-1] on.
    The returned row is still the (clamped) LAST real row and finite — the loss MULTIPLIES by the
    mask rather than skipping the entry, so a non-finite target would poison the whole batch."""
    task, demo_id, w, fi, traj_len = feats.demo(layout)
    last = int(fi[-1])
    for t in range(traj_len + 4):
        wt, valid = _target(feats, task, demo_id, t)
        assert bool(valid) == (t < last), f"t={t}: valid must be (t < {last})"
        assert np.isfinite(wt).all(), "masked rows must still be finite (mask multiplies, not skips)"
        if not valid:
            assert np.array_equal(wt, w[-1].astype(np.float32)), "invalid steps clamp to the last row"
    # the boundary itself, spelled out
    assert bool(_target(feats, task, demo_id, last - 1)[1]) is True
    assert bool(_target(feats, task, demo_id, last)[1]) is False
    assert bool(_target(feats, task, demo_id, last + 1)[1]) is False


# ------------------------- 04: no cross-episode / cross-task leakage -------------------------


def test_04_no_cross_episode_or_cross_task_leakage(feats):
    """A demo's terminal step must clamp to ITS OWN last row — never the next episode's first row —
    and the (task, demo) cache key must keep same-id demos in different tasks apart."""
    task_a, task_b = TASKS
    a_task, a_demo, a_w, a_fi, _ = feats.demo("stride8", task_a)
    b_task, b_demo, b_w, b_fi, _ = feats.demo("stride8", task_b)
    assert a_demo == b_demo, "same demo id under two tasks: the cache key must include the task"

    wt_a, valid_a = _target(feats, a_task, a_demo, int(a_fi[-1]) + 3)  # past the end of episode A
    assert not bool(valid_a)
    assert np.array_equal(wt_a, a_w[-1].astype(np.float32))
    assert not np.array_equal(wt_a, b_w[0].astype(np.float32)), "leaked the next episode's first row"

    # Every step of A must return a row that belongs to A (rows are globally unique per task/demo).
    a_rows = {tuple(r) for r in a_w.astype(np.float32)}
    b_rows = {tuple(r) for r in b_w.astype(np.float32)}
    assert not (a_rows & b_rows)
    for t in range(int(a_fi[-1]) + 3):
        assert tuple(_target(feats, a_task, a_demo, t)[0]) in a_rows

    # Interleaved access across tasks AND across demos in one task (the real per-worker pattern).
    for t in range(0, 20, 3):
        for task in TASKS:
            for layout in sorted(LAYOUTS):
                _, demo_id, w, fi, _ = feats.demo(layout, task)
                got, valid = _target(feats, task, demo_id, t)
                ref, ref_valid = wsm_align.next_at_with_valid(w, fi, t)
                assert np.array_equal(got, np.asarray(ref, np.float32)), (task, layout, t)
                assert bool(valid) == bool(ref_valid)


def test_04b_cache_eviction_does_not_corrupt_targets(feats, monkeypatch):
    """A 1-entry LRU forces a reload on every alternation: the reloaded rows must still be the
    demo's own (an eviction/key bug would surface as another demo's row)."""
    monkeypatch.setattr(gds, "_WSM_DEMO_CACHE_SIZE", 1)
    gds._wsm_demo_cache.clear()
    pairs = [(task, layout) for task in TASKS for layout in sorted(LAYOUTS)]
    for t in (0, 7, 9, 40):
        for task, layout in pairs:
            _, demo_id, w, fi, _ = feats.demo(layout, task)
            got, valid = _target(feats, task, demo_id, t)
            ref, ref_valid = wsm_align.next_at_with_valid(w, fi, t)
            assert np.array_equal(got, np.asarray(ref, np.float32)), (task, layout, t)
            assert bool(valid) == bool(ref_valid)
    assert len(list(iter(gds._wsm_demo_cache))) <= 1


# ------------------------- 05: stride / subsample semantics -------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_05_target_is_the_causal_windows_next_grid_row_never_the_current_one(feats, layout):
    """Stride semantics (doc 12 D4): omega lives on the stride-8 grid, so the target for a native
    step t is the row AFTER the newest grid row <= t — i.e. the next SAMPLED step, in lock-step with
    the S1/S2 input window (`_wsm_causal_window(..., k=1)[-1]`). Never omega_t itself."""
    task, demo_id, w, fi, traj_len = feats.demo(layout)
    for t in range(traj_len + 2):
        wt, valid = _target(feats, task, demo_id, t)
        base = int(gds._wsm_causal_window(fi, t, 1)[-1])  # what S1/S2 would inject at t
        if t < int(fi[0]):
            # before the first grid frame both the window and the target clamp to row 0; documented
            # divergence from doc 15's next_k_at (which would say row 1). Production grids start at 0.
            assert base == 0 and np.array_equal(wt, w[0].astype(np.float32))
            continue
        if valid:
            assert np.array_equal(wt, w[base + 1].astype(np.float32)), f"t={t}: want row base+1"
            assert not np.array_equal(wt, w[base].astype(np.float32)), f"t={t}: leaked omega_t"
        else:
            assert base == len(w) - 1, "invalid only on the terminal tail"


def test_05b_target_path_ignores_k_window_and_the_demo_cfg_z_root(feats, monkeypatch):
    """S3 pins WSM_K_WINDOW=1, but the TARGET must not depend on it (or on the demo-CFG z root,
    which is what makes the S1/S2 window path stochastic)."""
    task, demo_id, _, _, _ = feats.demo("stride8")
    ref = [_target(feats, task, demo_id, t) for t in range(0, 40, 3)]
    for k in (1, 2, 4, 8):
        monkeypatch.setattr(gds, "_WSM_K_WINDOW", k)
        monkeypatch.setattr(gds, "_WSM_Z_ROOT", "/nonexistent/z" if k % 2 == 0 else None)
        gds._wsm_demo_cache.clear()
        got = [_target(feats, task, demo_id, t) for t in range(0, 40, 3)]
        assert all(np.array_equal(a[0], b[0]) and bool(a[1]) == bool(b[1]) for a, b in zip(ref, got, strict=True))


# ------------------------- 06/07: the two __getitem__ call sites -------------------------


def _canned_item(horizon: int = 4) -> dict:
    """The base LeRobot item `GrootOpenpi*Dataset.__getitem__` post-processes (shapes only matter to
    the concatenations it performs)."""
    return {
        "state.end_effector_position_relative": np.zeros((1, 3), np.float32),
        "state.end_effector_rotation_relative": np.zeros((1, 4), np.float32),
        "state.base_position": np.zeros((1, 3), np.float32),
        "state.base_rotation": np.zeros((1, 4), np.float32),
        "state.gripper_qpos": np.zeros((1, 2), np.float32),
        "action.end_effector_position": np.zeros((horizon, 3), np.float32),
        "action.end_effector_rotation": np.zeros((horizon, 3), np.float32),
        "action.gripper_close": np.zeros((horizon, 1), np.float32),
        "action.base_motion": np.zeros((horizon, 4), np.float32),
        "action.control_mode": np.zeros((horizon, 1), np.float32),
        "video.robot0_agentview_left": np.zeros((1, 2, 2, 3), np.uint8),
        "video.robot0_eye_in_hand": np.zeros((1, 2, 2, 3), np.uint8),
        "annotation.human.task_description": ["put the thing in the other thing"],
    }


class _FakeSingle(gds.GrootOpenpiSingleDataset):
    """Real `GrootOpenpiSingleDataset.__getitem__` (the code under test) over a stubbed base."""

    def __init__(self, dataset_path, all_steps, trajectory_ids=None, trajectory_lengths=None):
        self._dataset_path = Path(dataset_path)
        self._all_steps = list(all_steps)
        self._trajectory_ids = np.asarray(trajectory_ids if trajectory_ids is not None else [])
        self._trajectory_lengths = np.asarray(
            trajectory_lengths if trajectory_lengths is not None else [], dtype=np.int64
        )


def test_06_single_getitem_emits_the_target_keys_and_never_the_window(feats, monkeypatch):
    task, demo_id, w, fi, traj_len = feats.demo("stride8")
    monkeypatch.setattr(gds.LeRobotSingleDataset, "__getitem__", lambda self, i: _canned_item())
    steps = [(demo_id, t) for t in range(traj_len)]
    ds = _FakeSingle(feats.path(task), steps)

    for index, (_, t) in enumerate(steps):
        item = ds[index]
        assert "wsm_w_target" in item and "wsm_w_target_valid" in item
        assert "wsm_w_window" not in item, "S3 must NOT also ship the injected window"
        assert "wsm_lang" not in item
        ref_w, ref_valid = wsm_align.next_at_with_valid(w, fi, t)
        assert np.array_equal(item["wsm_w_target"], np.asarray(ref_w, np.float32))
        assert bool(item["wsm_w_target_valid"]) == bool(ref_valid)
        # collate-critical: [Dw] fp32 and a 0-d bool -> [B, Dw] / [B] after default_collate, which is
        # what Observation.wsm_w_target_valid: Bool[Array, "*b"] requires (a [B,1] would broadcast the
        # cosine mask into [B,B] silently).
        assert item["wsm_w_target"].shape == (DW,) and item["wsm_w_target"].dtype == np.float32
        assert np.asarray(item["wsm_w_target_valid"]).shape == ()
        assert np.asarray(item["wsm_w_target_valid"]).dtype == np.bool_


def test_06b_window_mode_is_unchanged_by_the_target_branch(feats, monkeypatch):
    """The flip side of the branch: with WSM_JEPA_TARGETS off the same dataset emits the S1/S2
    window and NO target keys (guards against the two modes bleeding into each other)."""
    monkeypatch.setattr(gds, "_WSM_JEPA_TARGETS", False)
    monkeypatch.setattr(gds, "_WSM_K_WINDOW", 1)
    monkeypatch.setattr(gds.LeRobotSingleDataset, "__getitem__", lambda self, i: _canned_item())
    task, demo_id, w, fi, _ = feats.demo("stride8")
    ds = _FakeSingle(feats.path(task), [(demo_id, 9)])
    item = ds[0]
    assert "wsm_w_target" not in item and "wsm_w_target_valid" not in item
    assert np.array_equal(item["wsm_w_window"], w[gds._wsm_causal_window(fi, 9, 1)].astype(np.float32))


class _FakeMulti(gds.GrootOpenpiMultiDataset):
    """Real `GrootOpenpiMultiDataset.sample_step` + `__getitem__` over stubbed single datasets."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        n = len(self.datasets)
        self._dataset_sampling_weights = np.full(n, 1.0 / n)
        self._trajectory_sampling_weights = [
            np.full(len(d.trajectory_ids), 1.0 / len(d.trajectory_ids)) for d in self.datasets
        ]


def _mixture_env(feats, monkeypatch):
    """Two single datasets (one per task), each holding all four demos."""
    used: list[tuple[Path, int, int]] = []

    def base_getitem(self, index):
        dataset, trajectory_id, base_index = self.sample_step(index)  # exactly as the real base does
        used.append((Path(dataset.dataset_path), int(trajectory_id), int(base_index)))
        return _canned_item()

    monkeypatch.setattr(gds.LeRobotMixtureDataset, "__getitem__", base_getitem)
    singles = []
    for task in TASKS:
        ids = [DEMOS[name] for name in sorted(LAYOUTS)]
        lens = [feats.length[task, i] for i in ids]
        singles.append(_FakeSingle(feats.path(task), [], trajectory_ids=ids, trajectory_lengths=lens))
    return _FakeMulti(singles), used


def test_07_mixture_getitem_target_matches_the_step_sample_step_stashed(feats, monkeypatch):
    """The mixture path attaches omega from `self._wsm_last`. Assert the stash IS the step the base
    built the item from, and that the emitted target is that step's next row — in the same task."""
    mix, used = _mixture_env(feats, monkeypatch)
    np.random.seed(0)
    for _ in range(120):
        item = mix[0]
        ds, demo_id, base_index = mix._wsm_last
        assert used[-1] == (Path(ds.dataset_path), int(demo_id), int(base_index)), "stash desync"
        task = [t for t in TASKS if t in Path(ds.dataset_path).parts][0]
        w, fi = feats.w[task, demo_id], feats.fi[task, demo_id]
        ref_w, ref_valid = wsm_align.next_at_with_valid(w, fi, base_index)
        assert "wsm_w_window" not in item
        assert np.array_equal(item["wsm_w_target"], np.asarray(ref_w, np.float32))
        assert bool(item["wsm_w_target_valid"]) == bool(ref_valid)
        # and the row belongs to THIS task's demo, not the other dataset in the mixture
        assert tuple(item["wsm_w_target"]) in {tuple(r) for r in w.astype(np.float32)}
    assert len({u[0] for u in used}) == 2, "both mixture datasets should have been sampled"


# ------------------------- 08: the valid flag zeroes the loss contribution -------------------------


def test_08_loader_valid_flags_zero_the_terminal_rows_jepa_contribution(feats):
    """End-to-end loader -> fork loss: a batch built from REAL loader outputs (with terminal rows in
    it) must score exactly the same as one built from the valid rows only, and must be invariant to
    the content of the masked rows' targets. (test_pi_wsm_jepa t03 pins the mask math on synthetic
    bools; this pins that the loader's dtypes/flags actually drive it.)"""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")
    from openpi.models.wsm_jepa import WSMJepaHead, wsm_jepa_aux_loss

    task, demo_id, _, fi, _ = feats.demo("stride8")
    steps = [0, 7, 8, int(fi[-1]) - 1, int(fi[-1]), int(fi[-1]) + 5]  # 4 valid, 2 terminal
    rows = [_target(feats, task, demo_id, t) for t in steps]
    w_target = jnp.asarray(np.stack([r[0] for r in rows]))
    valid = jnp.asarray(np.stack([np.asarray(r[1]) for r in rows]))
    assert w_target.shape == (len(steps), DW) and valid.shape == (len(steps),)
    assert valid.dtype == jnp.bool_
    keep = np.asarray(valid)
    assert keep.tolist() == [True, True, True, True, False, False]

    head = WSMJepaHead(6, DW, rngs=nnx.Rngs(0))
    penult = jax.random.normal(jax.random.key(1), (len(steps), 3, 6))
    rng = jax.random.key(2)

    full = float(wsm_jepa_aux_loss(head, penult, w_target, valid, rng, sigreg_weight=0.0))
    sub = float(wsm_jepa_aux_loss(head, penult[keep], w_target[keep], valid[keep], rng, sigreg_weight=0.0))
    # rel=1e-4, not exact: the two calls have different batch shapes, so XLA rounds the predictor
    # matmul/reduction differently (fp32; observed ~5e-5 drift, and it moves with which other test
    # module compiled first). The EXACT statement of "contributes zero" is the invariance below.
    assert full == pytest.approx(sub, rel=1e-4), "terminal rows must contribute ~zero"

    # Same shapes, so this one is exact: the loss cannot depend on the masked rows' CONTENT (in
    # production they carry the clamped last row).
    poisoned = w_target.at[~keep].set(jnp.full((int((~keep).sum()), DW), 987.0))
    full_poisoned = float(wsm_jepa_aux_loss(head, penult, poisoned, valid, rng, sigreg_weight=0.0))
    assert full_poisoned == full, "masked rows must contribute EXACTLY zero"

    # ...and the loader's flags are load-bearing: mismarking the terminal rows as valid changes the
    # loss (i.e. the terminal tail really would be trained to predict the repeated final latent).
    mismarked = float(wsm_jepa_aux_loss(head, penult, w_target, jnp.ones_like(valid), rng, sigreg_weight=0.0))
    assert mismarked != full

    # ...but the mask MULTIPLIES, so a non-finite masked row would poison everything — which is why
    # test 03 requires the loader to always ship a finite (clamped) row.
    nan_batch = w_target.at[~keep].set(jnp.nan)
    assert not np.isfinite(float(wsm_jepa_aux_loss(head, penult, nan_batch, valid, rng, sigreg_weight=0.0)))
    jax.clear_caches()


# ------------------------- 09: determinism -------------------------


def test_09_target_path_consumes_no_rng_and_is_repeatable(feats):
    """`_wsm_jepa_target` is a pure function of (demo, frame): identical under any global seed, on
    cache-miss and cache-hit alike, and it must not advance numpy's global RNG (the demo-CFG window
    path does draw — this one must not, or it would desync worker RNG streams)."""
    task, demo_id, _, _, _ = feats.demo("irregular")
    probes = list(range(0, 22, 2))

    np.random.seed(0)
    gds._wsm_demo_cache.clear()
    first = [_target(feats, task, demo_id, t) for t in probes]  # cache-miss then hits
    state_before = np.random.get_state()
    again = [_target(feats, task, demo_id, t) for t in probes]  # all cache hits
    assert np.array_equal(np.random.get_state()[1], state_before[1]), "target path must draw no RNG"

    np.random.seed(12345)
    gds._wsm_demo_cache.clear()
    reseeded = [_target(feats, task, demo_id, t) for t in reversed(probes)][::-1]
    for a, b, c in zip(first, again, reseeded, strict=True):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
        assert bool(a[1]) == bool(b[1]) == bool(c[1])


def test_09b_seeded_mixture_draws_reproduce_exactly(feats, monkeypatch):
    """Whole-item determinism: the same seed replays the same (demo, frame) draws AND the same
    targets (the mixture's sample_step is the only RNG consumer on this path)."""
    mix, _ = _mixture_env(feats, monkeypatch)

    def run():
        np.random.seed(7)
        out = []
        for _ in range(40):
            item = mix[0]
            ds, demo_id, base_index = mix._wsm_last
            out.append(
                (
                    str(ds.dataset_path),
                    demo_id,
                    base_index,
                    tuple(item["wsm_w_target"]),
                    bool(item["wsm_w_target_valid"]),
                )
            )
        return out

    assert run() == run()


if __name__ == "__main__":
    raise SystemExit("run under pytest (fixtures required)")
