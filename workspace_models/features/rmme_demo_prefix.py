#!/usr/bin/env python3
"""rmme_demo_prefix — the `[demo omega prefix ; live omega window]` read, as pure numpy.

This is the ONE piece of new mechanism the RoboMME FrameSamp+Modul parity arm (M3) needs, and it
is deliberately dependency-free so it can be unit-tested on CPU and imported verbatim by BOTH the
training loader and the eval server. Train/serve divergence in exactly this function is the failure
mode that has already invalidated one eval in this study.

WHY THERE IS NO SECOND OMEGA STORE
----------------------------------
RoboMME stores the demonstration video as the episode's OWN leading rows: `is_demo=True` for
`step_idx < exec_start_idx`, execution after, in one episode-global frame index
(verified on all 1,600 episodes; `upstream_framesamp_data.py` states the same thing about the
official sampler: "the inclusive prefix [0, step_idx] naturally contains the complete demonstration
followed by causal execution history; no special demo/execution splice is needed").

So a per-episode omega store over the whole episode ALREADY contains the demo omegas:

    demo omega  = omega[frame_indices <  exec_start_idx]
    live omega  = omega[frame_indices >= exec_start_idx]

and Stage-E's causal trunk is causal within the demo too, so those rows are exactly `omega_t` for
`t < exec_start_idx`. The "K demo tokens" are an INDEX RULE over the existing store, not a second
artifact.

THE WINDOW
----------
    slots  [0, k_demo)                    demo prefix, uniform-inclusive over the demo grid
    slots  [k_demo, k_demo + k_live)      the existing causal live window, oldest -> newest

`segment_flag` is returned alongside (0 = demo, 1 = live), but the DEFAULT arm does not need a new
parameter to consume it: the GDN conditioner's `pos_decay_bias` is already `[window_len, num_heads]`
— a per-SLOT learned bias — and the prefix always occupies the same fixed leading slots. The
segment identity is therefore already learnable from position, the window length stays structurally
readable from the checkpoint (`pos_decay_bias.shape[0] == k_demo + k_live`), and M3 is a one-line
`cond_window` diff on the sealed recipe. The explicit flag is returned for (a) preflight assertions,
(b) the diagnostic that measures whether the prefix slots contribute at all, and (c) the
pre-registered fallback arm that adds a learned `[2, w_dim]` segment embedding if they do not.

TASKS WITH NO DEMONSTRATION (7 of 16)
-------------------------------------
Measured over the sealed corpus: PickXtimes, StopCube, SwingXtimes, BinFill, PickHighlight,
ButtonUnmask and ButtonUnmaskSwap have `exec_start_idx == 0` on all 100 episodes — no demo video at
all. The other 9 have one on all 100. A multitask arm must still ship a static shape, so the demo
slots are filled by CLAMPING to the earliest available omega (the oldest live frame), which is the
same clamp-to-zero convention `robomme_integration/eval/workspace_runner.py::requested_omega_steps`
already uses for a short live window. Zero-filling was rejected: it would feed the conditioner a
vector the encoder never produces, i.e. a second, uncontrolled distribution shift confined to
exactly the 7 tasks that carry the C3 counting stratum.

`prefix_valid` marks which demo slots are real, so the arm can be scored with the no-demo tasks
separated out and the pre-registered per-suite reading is not silently averaged over a cell where
the mechanism cannot apply.

THE STORE-LEVEL CALL (what the trainer and the server actually import)
--------------------------------------------------------------------
`window_for_step(arm, episode, step, ...)` is the ONE definition both sides use. `episode` is a
loaded Stage-E omega store record; `step` is the EPISODE-GLOBAL frame index of the current decision
(the parquet's `step_idx` at train time, `exec_start_idx + env_steps_since_reset` at serve time).

Two index spaces exist and confusing them is the whole hazard, so they are named:

    STEP space  the episode's own frame index 0..T-1. Both sides can compute it.
    ROW  space  an index into the omega store, which lives on the tap's stride-8 grid.

`row_for_step` is the causal bridge: the newest grid row at or before `step`. The live window
strides in STEP space (default 10 = `robomme_integration/training/config.py::CHUNK_STRIDE`), so an
arm keeps the sealed recipe's temporal extent (16 slots x 10 steps = 150 steps of history) even
though the store is 8x sparser than the legacy dense `omega_f16.npy`. When the store is dense
(`frame_indices == arange(T)`) the step-space rule collapses to `requested_omega_steps` exactly —
asserted in `--self-test`, which is what makes the bit-identity claim mean something.

Self-test (real data, CPU, no model):

    PYTHONPATH=<repo> python -m workspace_models.features.rmme_demo_prefix --self-test \\
        --dataset-root ~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/snapshots/1510653c...
"""

from __future__ import annotations

import numpy as np

#: The paper's FrameSamp budget is 32 frames over the WHOLE inclusive prefix. Our tokens are
#: 512-wide pooled omegas rather than 16 spatial tokens per frame, so the budget that matches the
#: sealed GDN recipe is k_live = cond_window = 16 and k_demo = 8. Both are config values; these are
#: the pre-registered defaults, not constants.
DEFAULT_K_DEMO = 8
DEFAULT_K_LIVE = 16
DEFAULT_STRIDE = 10  # robomme_integration/training/config.py::CHUNK_STRIDE


def uniform_inclusive(count: int, n: int) -> np.ndarray:
    """`count` indices into `range(n)`, chronological, both endpoints included.

    Identical rule to `robomme_integration/sequence.py::uniformly_sample_prefix` (integer linspace)
    and to the official `even_sampling_indices` in `upstream_framesamp_data.py`: keep everything
    when the source is short, otherwise `linspace(0, n-1, count)`.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)


def demo_prefix_slots(frame_indices: np.ndarray, exec_start_idx: int, k_demo: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows into the omega store for the demo prefix, prefix_valid mask), both length `k_demo`.

    `frame_indices` is the tap/omega store's episode-global grid (stride 8 + final frame).
    Rows with `frame_indices < exec_start_idx` are the demonstration.
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1 or frame_indices.size < 1:
        raise ValueError(f"frame_indices must be a nonempty 1-D grid, got {frame_indices.shape}")
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    if k_demo < 1:
        raise ValueError(f"k_demo must be positive, got {k_demo}")
    demo_rows = np.flatnonzero(frame_indices < int(exec_start_idx))
    if demo_rows.size == 0:
        # No demonstration for this task: clamp every prefix slot to the earliest omega there is.
        return np.zeros((k_demo,), dtype=np.int64), np.zeros((k_demo,), dtype=bool)
    picked = demo_rows[uniform_inclusive(k_demo, demo_rows.size)]
    if picked.size < k_demo:
        # Fewer demo grid points than slots: LEFT-pad by repeating the oldest, so the newest demo
        # frame always lands in the slot adjacent to the live window (chronology is preserved and
        # the recurrence still ends on the most recent demo evidence).
        picked = np.concatenate([np.full(k_demo - picked.size, picked[0], dtype=np.int64), picked])
    return picked.astype(np.int64), np.ones((k_demo,), dtype=bool)


def live_window_slots(decision_row: int, k_live: int, stride: int) -> np.ndarray:
    """The EXISTING causal live window, reproduced bit-for-bit.

    Verbatim `robomme_integration/eval/workspace_runner.py::requested_omega_steps`: oldest-to-newest
    with clamp-to-zero. Reproduced (not imported) because this module must stay dependency-free for
    the serve path; `--self-test` asserts the two agree.
    """
    if decision_row < 0 or k_live < 1 or stride < 1:
        raise ValueError("invalid live window geometry")
    return np.asarray(
        [max(decision_row - stride * offset, 0) for offset in range(k_live - 1, -1, -1)],
        dtype=np.int64,
    )


def build_window(
    omega: np.ndarray,
    frame_indices: np.ndarray,
    exec_start_idx: int,
    decision_row: int,
    *,
    k_demo: int = DEFAULT_K_DEMO,
    k_live: int = DEFAULT_K_LIVE,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """([k_demo + k_live, D] float32, segment_flag [K] int8, prefix_valid [k_demo] bool).

    `decision_row` indexes `omega` / `frame_indices` (the omega-store row of the current decision),
    NOT the episode-global frame index. Slot order is oldest -> newest within each segment, demo
    first, so the GDN reads out at the newest LIVE omega exactly as it does today.
    """
    omega = np.asarray(omega)
    if omega.ndim != 2 or omega.shape[0] != np.asarray(frame_indices).shape[0]:
        raise ValueError(f"omega {omega.shape} does not match the frame grid {np.asarray(frame_indices).shape}")
    prefix, prefix_valid = demo_prefix_slots(frame_indices, exec_start_idx, k_demo)
    live = live_window_slots(int(decision_row), k_live, stride)
    if int(live.max()) >= omega.shape[0]:
        raise IndexError(f"decision row {decision_row} exceeds omega history {omega.shape[0]}")
    rows = np.concatenate([prefix, live])
    window = omega[rows].astype(np.float32, copy=True)
    if not np.isfinite(window).all():
        raise ValueError("RoboMME omega window contains non-finite values")
    segment = np.concatenate([np.zeros(k_demo, np.int8), np.ones(k_live, np.int8)])
    return window, segment, prefix_valid


# ----------------------------------------------------------------------- the serve-aligned grid
#: The execution horizon the sealed fixed-800 protocol commits per decision. The server therefore
#: observes exactly one frame every EXECUTION_HORIZON env steps after the demonstration ends.
EXECUTION_HORIZON = 16
DEMO_GRID_STRIDE = 8  # = pi_pooled_tap.STRIDE; every demo frame is available at reset


def serve_aligned_grid(
    n_frames: int, exec_start_idx: int, *, demo_stride: int = DEMO_GRID_STRIDE, live_stride: int = EXECUTION_HORIZON
) -> np.ndarray:
    """The frame grid a RoboMME policy arm's omega store MUST use.

    MEASURED DEFECT this exists to prevent (2026-09-02, before any policy submit). The plain
    `wsm_pooled` grid is `arange(0, n, 8)`, anchored at frame 0. At serve the current frame's
    episode-global index is `exec_start_idx + 16 * decision`, and 16 == 0 (mod 8), so every serve
    frame is congruent to `exec_start_idx` mod 8. Over the 900 episodes that carry a demonstration,
    `exec_start_idx % 8 == 0` for only **167 (18.6 %)** — for the other 81.4 % **not one live frame
    ever lands on the store's grid**, so an online producer would emit zero live omegas and the
    "live" half of the window would silently be filled from demo rows.

    Anchoring the live half at `exec_start_idx` with stride == the execution horizon makes the
    served sequence and the stored sequence IDENTICAL, which is what D7 can then certify end to
    end rather than on the prefix alone. The demo half keeps stride 8: every demo frame is handed
    over at reset, so it is exactly reproducible, and a denser prefix gives `demo_prefix_slots`
    more to sample from.

        grid = arange(0, D, 8)  U  arange(D, n, 16)

    NO trailing final frame. `pi_pooled_tap.frame_selection` appends `n-1` so the last frame is
    represented in a REPRESENTATION corpus; in a POLICY store it is actively harmful, because the
    server never observes it (the episode ends) and the stored sequence would then be one row
    longer than the served one — the sequence identity D7 certifies would be false at exactly the
    last decision.
    """
    n_frames, exec_start_idx = int(n_frames), int(exec_start_idx)
    if n_frames < 1 or exec_start_idx < 0 or exec_start_idx > n_frames:
        raise ValueError(f"invalid episode geometry n={n_frames} exec_start={exec_start_idx}")
    if demo_stride < 1 or live_stride < 1:
        raise ValueError("strides must be positive")
    demo = np.arange(0, exec_start_idx, demo_stride, dtype=np.int64)
    live = np.arange(exec_start_idx, n_frames, live_stride, dtype=np.int64)
    grid = np.concatenate([demo, live])
    if grid.size == 0:
        raise ValueError(f"serve-aligned grid is empty for n={n_frames} D={exec_start_idx}")
    return grid


def serve_frame_indices(exec_start_idx: int, decisions: int, live_stride: int = EXECUTION_HORIZON) -> np.ndarray:
    """The episode-global frame index of each decision, as the server sees them."""
    return int(exec_start_idx) + int(live_stride) * np.arange(int(decisions), dtype=np.int64)


# ------------------------------------------------------------------- STEP space <-> ROW space
def row_for_step(frame_indices: np.ndarray, step: int) -> int:
    """The newest omega-store row at or before episode-global frame `step`. CAUSAL by construction.

    The store lives on the tap's stride-8 grid, so most steps have no row of their own; taking the
    newest row at or before `step` is the only choice that never reads a future frame. `step`
    before the first grid point clamps to row 0 (the grid always starts at 0, so this only fires
    for a negative step, which is rejected).
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if int(step) < 0:
        raise ValueError(f"step must be nonnegative, got {step}")
    row = int(np.searchsorted(frame_indices, int(step), side="right")) - 1
    return max(row, 0)


def live_window_rows(
    frame_indices: np.ndarray, step: int, k_live: int, stride_steps: int = DEFAULT_STRIDE
) -> np.ndarray:
    """The causal live window, oldest -> newest, striding in STEP space and clamped at step 0.

    Preserves the sealed recipe's temporal extent (k_live x stride_steps steps of history) on a
    sparse store. On a DENSE store this is exactly `requested_omega_steps` / `live_window_slots`.
    """
    if k_live < 1 or stride_steps < 1:
        raise ValueError("invalid live window geometry")
    steps = [max(int(step) - stride_steps * offset, 0) for offset in range(k_live - 1, -1, -1)]
    return np.asarray([row_for_step(frame_indices, s) for s in steps], dtype=np.int64)


# ------------------------------------------------------------------------------- store loading
def load_episode(omega_root, task: str, episode: int, *, pooled_root=None, domain: str = "robomme") -> dict:
    """Load one episode's Stage-E omega record + its demo/live split.

    `omega_root` is either `.../omega/<cell>` (the domain subdir is appended) or a root that
    already points inside the domain. `exec_start_idx` comes from the TAP store (`p.npz`), which is
    where the split is recorded; if `pooled_root` is omitted the caller must supply
    `exec_start_idx` itself via `episode_record`.
    """
    from pathlib import Path

    omega_root = Path(omega_root).expanduser()
    base = omega_root / domain if (omega_root / domain).is_dir() else omega_root
    w_path = base / task / f"demo_{episode:06d}" / "w.npz"
    if not w_path.is_file():
        raise FileNotFoundError(f"no omega record at {w_path}")
    with np.load(w_path) as blob:
        w = np.asarray(blob["w"])
        frame_indices = np.asarray(blob["frame_indices"], dtype=np.int64)
        lang_global = np.asarray(blob["lang_global"])
        encoder_id = str(blob["encoder_id"]) if "encoder_id" in blob else ""
    exec_start_idx = None
    if pooled_root is not None:
        p_path = Path(pooled_root).expanduser() / task / f"demo_{episode:06d}" / "p.npz"
        with np.load(p_path) as blob:
            exec_start_idx = int(np.asarray(blob["exec_start_idx"]))
            if not np.array_equal(np.asarray(blob["frame_indices"], dtype=np.int64), frame_indices):
                raise ValueError(f"tap and omega grids disagree for {task}/ep{episode}")
    return episode_record(
        w, frame_indices, exec_start_idx, task=task, episode=episode, lang_global=lang_global, encoder_id=encoder_id
    )


def episode_record(
    w, frame_indices, exec_start_idx, *, task: str = "", episode: int = -1, lang_global=None, encoder_id: str = ""
) -> dict:
    """Validate and package one episode's omega record. Pure; no I/O."""
    w = np.asarray(w)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if w.ndim != 2 or w.shape[0] != frame_indices.shape[0]:
        raise ValueError(f"omega {w.shape} does not match grid {frame_indices.shape}")
    if exec_start_idx is None:
        raise ValueError("exec_start_idx is required; it is the demo/live split")
    if int(exec_start_idx) < 0:
        raise ValueError(f"exec_start_idx must be nonnegative, got {exec_start_idx}")
    return {
        "w": w,
        "frame_indices": frame_indices,
        "exec_start_idx": int(exec_start_idx),
        "task": task,
        "episode": int(episode),
        "lang_global": lang_global,
        "encoder_id": encoder_id,
        "has_demo": int(exec_start_idx) > 0,
    }


# ------------------------------------------------------- THE shared call (trainer AND server)
def window_for_step(
    arm: str,
    episode: dict,
    step: int,
    *,
    k_demo: int = DEFAULT_K_DEMO,
    k_live: int = DEFAULT_K_LIVE,
    stride_steps: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """([K, D] float32 oldest->newest, segment_flag [K] int8, prefix_valid [k_demo] bool).

    `step` is the EPISODE-GLOBAL frame index of the current decision. This is the one definition
    the training loader and the eval server both import; a divergence here is the failure mode that
    has already invalidated one eval in this study.

    arms: m1_live_only (K = k_live) | m2_demo_only (K = k_demo) | m3_demo_live (K = k_demo+k_live).
    m3_ctrl is m3_demo_live fed a store exported by the structure-free encoder — it is a STORE
    swap, never a code path, which is what keeps the ablation one-factor.
    """
    w, grid = episode["w"], episode["frame_indices"]
    start = episode["exec_start_idx"]
    if arm == "m3_ctrl":
        arm = "m3_demo_live"
    if arm not in ("m1_live_only", "m2_demo_only", "m3_demo_live"):
        raise ValueError(f"unknown RoboMME workspace arm {arm!r}")

    prefix, prefix_valid = demo_prefix_slots(grid, start, k_demo)
    live = live_window_rows(grid, step, k_live, stride_steps)
    if arm == "m1_live_only":
        rows, segment, prefix_valid = live, np.ones(k_live, np.int8), prefix_valid[:0]
    elif arm == "m2_demo_only":
        rows, segment = prefix, np.zeros(k_demo, np.int8)
    else:
        rows = np.concatenate([prefix, live])
        segment = np.concatenate([np.zeros(k_demo, np.int8), np.ones(k_live, np.int8)])
    window = np.asarray(w[rows], dtype=np.float32)
    if not np.isfinite(window).all():
        raise ValueError(f"non-finite omega window for {episode['task']}/ep{episode['episode']} step {step}")
    return window, segment, prefix_valid


def arm_window(arm: str, omega, frame_indices, exec_start_idx, decision_row, **kw):
    """Dispatch the four pre-registered reads. Returns the same triple as `build_window`.

    m1_live_only   the standard read: k_demo effectively 0
    m2_demo_only   the prefix alone, read out at the newest DEMO omega
    m3_demo_live   the parity arm: [demo prefix ; live window]
    """
    k_demo = int(kw.pop("k_demo", DEFAULT_K_DEMO))
    k_live = int(kw.pop("k_live", DEFAULT_K_LIVE))
    if arm == "m1_live_only":
        window, segment, valid = build_window(
            omega, frame_indices, exec_start_idx, decision_row, k_demo=1, k_live=k_live, **kw
        )
        return window[1:], segment[1:], valid[:0]
    if arm == "m2_demo_only":
        prefix, valid = demo_prefix_slots(frame_indices, exec_start_idx, k_demo)
        window = np.asarray(omega)[prefix].astype(np.float32, copy=True)
        return window, np.zeros(k_demo, np.int8), valid
    if arm == "m3_demo_live":
        return build_window(omega, frame_indices, exec_start_idx, decision_row, k_demo=k_demo, k_live=k_live, **kw)
    raise ValueError(f"unknown RoboMME workspace arm {arm!r}")


# ------------------------------------------------------------------------------------- self-test
def _self_test(dataset_root: str, episodes: int) -> None:
    import sys
    from pathlib import Path

    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robomme_integration.eval.workspace_runner import requested_omega_steps
    from workspace_models.features.pi_pooled_tap import frame_selection
    from workspace_models.labels.robomme_source import TASK_ORDER, episode_path

    # 1. the live rule must be BIT-IDENTICAL to the shipped serve rule
    for decision in (0, 1, 7, 40, 137):
        for k in (1, 8, 16, 24):
            for stride in (1, 10, 16):
                assert tuple(live_window_slots(decision, k, stride)) == requested_omega_steps(
                    decision, window=k, stride=stride
                ), (decision, k, stride)
    print("[self-test] live window == workspace_runner.requested_omega_steps on 60 geometries")

    # 1b. on a DENSE store (row == step) the STEP-space rule collapses to the same sequence, which
    #     is what lets a sparse-store arm claim the sealed recipe's temporal geometry.
    dense = np.arange(4_096, dtype=np.int64)
    for step in (0, 1, 7, 40, 137, 999):
        for k in (1, 8, 16, 24):
            for stride in (1, 10, 16):
                assert tuple(live_window_rows(dense, step, k, stride)) == requested_omega_steps(
                    step, window=k, stride=stride
                ), (step, k, stride)
    print("[self-test] step-space live window == requested_omega_steps on a dense store (72 geometries)")

    # 2. the prefix rule, on the REAL grids of every episode requested
    root = Path(dataset_root).expanduser()
    n_with = n_without = 0
    per_task: dict[str, list[int]] = {}
    step = max(1, 1600 // max(episodes, 1))
    for ep in range(0, 1600, step):
        table = pq.read_table(episode_path(root, ep), columns=["is_demo", "exec_start_idx"])
        n = table.num_rows
        grid = frame_selection(n)
        raw = table.column("exec_start_idx").to_pylist()
        start = int(raw[0][0] if isinstance(raw[0], (list, tuple)) else raw[0])
        omega = np.arange(len(grid) * 4, dtype=np.float32).reshape(len(grid), 4)
        decision = len(grid) - 1
        window, segment, valid = build_window(omega, grid, start, decision)
        assert window.shape == (DEFAULT_K_DEMO + DEFAULT_K_LIVE, 4), window.shape
        assert segment.tolist() == [0] * DEFAULT_K_DEMO + [1] * DEFAULT_K_LIVE
        # the newest slot is always the current decision's omega
        assert np.array_equal(window[-1], omega[decision])
        task = TASK_ORDER[ep // 100]
        if start > 0:
            n_with += 1
            assert valid.all(), (task, ep)
            prefix_rows, _ = demo_prefix_slots(grid, start, DEFAULT_K_DEMO)
            assert (grid[prefix_rows] < start).all(), (task, ep, grid[prefix_rows], start)
            assert (np.diff(prefix_rows) >= 0).all(), "prefix must be chronological"
            # the newest demo slot is the LAST demo grid point
            assert prefix_rows[-1] == np.flatnonzero(grid < start)[-1], (task, ep)
            per_task.setdefault(task, []).append(int(np.flatnonzero(grid < start).size))
        else:
            n_without += 1
            assert not valid.any(), (task, ep)
            assert np.array_equal(window[:DEFAULT_K_DEMO], np.repeat(omega[:1], DEFAULT_K_DEMO, 0))
        # --- the STORE-LEVEL call: causality, monotonicity, and the demo/live invariants ---
        record = episode_record(omega, grid, start, task=task, episode=ep)
        for probe in (start, start + 1, n // 2, n - 1):
            probe = int(min(max(probe, 0), n - 1))
            row = row_for_step(grid, probe)
            assert grid[row] <= probe, (task, ep, probe, grid[row])  # never reads the future
            assert row == len(grid) - 1 or grid[row + 1] > probe, (task, ep, probe)
            w3s, seg3, valid3 = window_for_step("m3_demo_live", record, probe)
            w1s, seg1, _ = window_for_step("m1_live_only", record, probe)
            w2s, seg2, _ = window_for_step("m2_demo_only", record, probe)
            assert w3s.shape == (DEFAULT_K_DEMO + DEFAULT_K_LIVE, 4)
            assert w1s.shape == (DEFAULT_K_LIVE, 4) and w2s.shape == (DEFAULT_K_DEMO, 4)
            assert seg3.tolist() == [0] * DEFAULT_K_DEMO + [1] * DEFAULT_K_LIVE
            assert seg1.tolist() == [1] * DEFAULT_K_LIVE and seg2.tolist() == [0] * DEFAULT_K_DEMO
            # M3 is exactly [M2 ; M1] -> the ablation really is one factor
            assert np.array_equal(w3s[:DEFAULT_K_DEMO], w2s)
            assert np.array_equal(w3s[DEFAULT_K_DEMO:], w1s)
            # the newest LIVE slot is the causal row for this step
            assert np.array_equal(w3s[-1], omega[row_for_step(grid, probe)])
            # every LIVE slot is at or before the decision; every DEMO slot is strictly before exec
            live_rows = live_window_rows(grid, probe, DEFAULT_K_LIVE)
            assert (grid[live_rows] <= probe).all(), (task, ep, probe)
            assert (np.diff(live_rows) >= 0).all()
            if start > 0:
                demo_rows, _ = demo_prefix_slots(grid, start, DEFAULT_K_DEMO)
                assert (grid[demo_rows] < start).all(), (task, ep)
                assert valid3.all()
            else:
                assert not valid3.any()
        # m1/m2/m3 must be genuinely distinct objects
        w1, _, _ = arm_window("m1_live_only", omega, grid, start, decision)
        w2, _, _ = arm_window("m2_demo_only", omega, grid, start, decision)
        w3, _, _ = arm_window("m3_demo_live", omega, grid, start, decision)
        assert w1.shape == (DEFAULT_K_LIVE, 4) and w2.shape == (DEFAULT_K_DEMO, 4)
        assert w3.shape == (DEFAULT_K_DEMO + DEFAULT_K_LIVE, 4)
        assert np.array_equal(w3[DEFAULT_K_DEMO:], w1) and np.array_equal(w3[:DEFAULT_K_DEMO], w2)
    print(
        f"[self-test] prefix rule OK on {n_with + n_without} real episodes ({n_with} with a demo, {n_without} without)"
    )
    for task in sorted(per_task):
        lengths = per_task[task]
        print(
            f"  {task:<18} demo grid points: min {min(lengths)} med "
            f"{int(np.median(lengths))} max {max(lengths)}  (k_demo={DEFAULT_K_DEMO})"
        )
    print("[self-test] PASS")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--dataset-root",
        default="~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
        "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20",
    )
    ap.add_argument("--episodes", type=int, default=160)
    args = ap.parse_args()
    if not args.self_test:
        raise SystemExit("this module is a library; pass --self-test to exercise it")
    _self_test(args.dataset_root, args.episodes)


if __name__ == "__main__":
    main()
