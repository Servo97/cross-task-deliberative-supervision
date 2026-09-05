#!/usr/bin/env python3
"""Serve-side omega for the RoboMME demo-prefix arms: prefix at reset, live window per decision.

This is the counterpart of `rmme_workspace_dataset.py`. Both call the SAME
`rmme_demo_prefix.window_for_step`, so the training loader and the policy server cannot drift in
the slot allocation. What CAN drift is which frames each side feeds the encoder, and this module is
explicit about that (see LIVE-GRID ALIGNMENT below) rather than assuming it away.

THE STATE MACHINE, matched to the RoboMME harness

  episode_restart   `obs["video_history"]` + `obs["video_state_history"]` arrive as one aligned
                    block (`eval/workspace_runner.py::capture_workspace_observation` already
                    validates and requires this envelope). `exec_start_idx = len(video_history)`.
                    The demonstration's grid rows are EXACTLY `arange(0, exec_start_idx, 8)` —
                    reproducible at reset without knowing the episode length, because
                    `pi_pooled_tap.frame_selection` is `arange(0, n, 8)` plus a final frame that
                    is always >= exec_start_idx. So the prefix half of the store is causally
                    reconstructible at serve; this is what D7 gates on.
  each decision     one current frame. Its episode-global index is
                    `exec_start_idx + execution_horizon * decision`. If that index is on the
                    stride-8 grid it is pushed through the producer; the window is then built at
                    `step = that index`, and `row_for_step` selects the newest row at or before it.

  window            `window_for_step(arm.read, record, step)` -> [K, 512], oldest -> newest, the
                    same array the loader produced at training time.

LIVE-GRID ALIGNMENT — a defect this self-test FOUND, and the grid that fixes it
  The plain `wsm_pooled` grid is `arange(0, n, 8)`. Serve frames are `exec_start_idx + 16k`, and
  16 == 0 (mod 8), so every serve frame is congruent to `exec_start_idx` mod 8. Measured over the
  900 episodes with a demonstration: `exec_start_idx % 8 == 0` on only **167 (18.6 %)**. On the
  other 81.4 % **not one live frame lands on the store grid** — an online producer would emit zero
  live omegas and the live half of the window would be silently filled from demo rows.

  The policy-facing omega store therefore uses `rmme_demo_prefix.serve_aligned_grid`:
  `arange(0, D, 8) U arange(D, n, 16)`. The served sequence is then IDENTICAL to the stored one,
  so D7 certifies the whole window instead of only its prefix. `realized_grid_coverage()` is kept
  and asserted at 1.0 — a coverage below 1.0 is now a hard error, not a reported gap.

  NOTE this is a store-grid change only. The Stage-E ENCODER corpus keeps the plain stride-8 tap
  (so the A3 audit and the 4-tap retrain are untouched); the policy store is exported from a
  second, serve-aligned tap pass.

WHAT THIS MODULE DOES NOT DO
  It does not modify `eval/execution_model_server.py` or `eval/workspace_runner.py`. Those are the
  sealed v1 path whose arms are scored. `attach()` installs this producer for the four registered
  demo-prefix arm ids only; every other arm keeps the legacy `OnlineWorkspaceRunner` untouched.

Self-test (CPU, real grids, a stub producer so no model or checkpoint is needed):

    PYTHONPATH=<repo> python -m workspace_models.overlays.rmme_serve_omega --self-test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from workspace_models.features.rmme_demo_prefix import (
    DEFAULT_STRIDE,
    DEMO_GRID_STRIDE,
    EXECUTION_HORIZON,
    episode_record,
    row_for_step,
    serve_aligned_grid,
    serve_frame_indices,
    window_for_step,
)
from workspace_models.overlays.rmme_arms import BY_ID

DOMAIN = "robomme"


def demo_grid(exec_start_idx: int) -> np.ndarray:
    """The store's demo grid rows, reconstructed at reset. Exact — see the docstring."""
    if int(exec_start_idx) < 0:
        raise ValueError("exec_start_idx must be nonnegative")
    return np.arange(0, int(exec_start_idx), DEMO_GRID_STRIDE, dtype=np.int64)


class RmmeServeOmegaSession:
    """All omega state for one RoboMME episode. One session per harness connection."""

    def __init__(self, producer, *, arm_id: str, execution_horizon: int = 16, stride_steps: int = DEFAULT_STRIDE):
        arm = BY_ID.get(arm_id)
        if arm is None:
            raise ValueError(f"{arm_id!r} is not a registered RoboMME demo-prefix arm")
        self.arm = arm
        self.producer = producer
        self.execution_horizon = int(execution_horizon)
        self.stride_steps = int(stride_steps)
        self.exec_start_idx: int | None = None
        self.frames: list[int] = []  # episode-global index of every produced row
        self.omega: list[np.ndarray] = []
        self.decisions = 0
        self.skipped_offgrid = 0

    # ------------------------------------------------------------------ reset
    def on_episode_restart(self, demo_pooled) -> int:
        """`demo_pooled[i]` = the pooled tap token for demo frame i. Returns rows produced."""
        if self.exec_start_idx is not None:
            raise RuntimeError("RoboMME omega session received a second episode_restart")
        demo_pooled = np.asarray(demo_pooled)
        if demo_pooled.ndim != 2:
            raise ValueError(f"demo pooled tokens must be [D_frames, dim], got {demo_pooled.shape}")
        self.exec_start_idx = int(demo_pooled.shape[0])
        self.producer.reset()
        for index in demo_grid(self.exec_start_idx):
            self.omega.append(np.asarray(self.producer.step(demo_pooled[int(index)]), dtype=np.float32).reshape(-1))
            self.frames.append(int(index))
        return len(self.frames)

    # ------------------------------------------------------------ per decision
    def step_index(self, decision: int) -> int:
        if self.exec_start_idx is None:
            raise RuntimeError("RoboMME omega session used before episode_restart")
        return self.exec_start_idx + self.execution_horizon * int(decision)

    def on_decision(self, pooled_t) -> int:
        """Push the current frame. On the serve-aligned grid EVERY decision frame is a grid row —
        that is the whole point of the grid — so an off-grid step is a fatal contract violation,
        never a silently skipped frame."""
        step = self.step_index(self.decisions)
        self.decisions += 1
        if self.execution_horizon != EXECUTION_HORIZON:
            raise RuntimeError(
                f"serve-aligned grid assumes execution horizon {EXECUTION_HORIZON}, got {self.execution_horizon}"
            )
        if (step - self.exec_start_idx) % self.execution_horizon != 0:
            self.skipped_offgrid += 1
            raise RuntimeError(
                f"decision frame {step} is off the serve-aligned grid "
                f"(exec_start={self.exec_start_idx}) — the store and the server "
                f"would diverge"
            )
        self.omega.append(np.asarray(self.producer.step(pooled_t), dtype=np.float32).reshape(-1))
        self.frames.append(step)
        if not self.omega:
            raise RuntimeError("RoboMME omega session has no rows; episode_restart never ran")
        return step

    def record(self) -> dict:
        return episode_record(
            np.stack(self.omega), np.asarray(self.frames, dtype=np.int64), self.exec_start_idx, task="", episode=-1
        )

    def window(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return window_for_step(
            self.arm.read,
            self.record(),
            int(step),
            k_demo=max(self.arm.k_demo, 1),
            k_live=max(self.arm.k_live, 1),
            stride_steps=self.stride_steps,
        )

    # ------------------------------------------------------------ diagnostics
    def realized_grid_coverage(self, episode_length: int | None = None) -> dict:
        """How much of the store's grid this episode actually reproduced."""
        frames = np.asarray(self.frames, dtype=np.int64)
        start = int(self.exec_start_idx or 0)
        live = frames[frames >= start]
        expected_live = None
        if episode_length is not None:
            grid = serve_aligned_grid(int(episode_length), start)
            expected_live = int((grid >= start).sum())
        return {
            "exec_start_idx": start,
            "demo_rows": int((frames < start).sum()),
            "demo_rows_expected": int(demo_grid(start).size),
            "demo_exact": bool((frames < start).sum() == demo_grid(start).size),
            "live_rows": int(live.size),
            "live_rows_expected": expected_live,
            "live_coverage": None if not expected_live else round(live.size / expected_live, 4),
            "decisions": self.decisions,
            "skipped_offgrid": self.skipped_offgrid,
            "execution_horizon": self.execution_horizon,
        }


# ------------------------------------------------------------------------------- the D7 gate
def d7_preflight(
    omega_root,
    pooled_root,
    encoder_ckpt,
    *,
    demos: int = 20,
    lang_mode: str = "stored",
    cos_bar: float = 0.999,
    out: str | Path | None = None,
) -> dict:
    """Run the shipped D7 identity gate for the robomme domain, with the encoder_step check.

    Delegates to `scripts/deliberation/stage_e_omega_parity.py` rather than reimplementing it —
    that script is the gate the other three domains were certified with, and it already refuses a
    checkpoint whose step differs from the store's `_meta.json` `encoder_step` (h14 §41.2).
    `lang_mode` MUST be `stored`: `taskmean` recomputes a mean over the demos parity happens to
    sample and fails a CORRECT encoder (§39.3).
    """
    import subprocess
    import sys

    encoder_ckpt = str(encoder_ckpt)
    if encoder_ckpt.endswith("encoder_best.pt"):
        raise ValueError("D7 must use the encoder.pt that EXPORTED the store, not encoder_best.pt")
    if lang_mode == "taskmean":
        raise ValueError("taskmean is a §25.3 diagnostic, never a gate; use --lang-mode stored")
    meta = Path(omega_root).expanduser() / "_meta.json"
    if meta.is_file():
        recorded = json.loads(meta.read_text()).get("encoder_step")
        if recorded is None:
            raise ValueError(
                f"{meta} has no encoder_step; the store predates the §41.2 fix and cannot be gated — re-export it"
            )
    repo = Path(__file__).resolve().parents[2]
    out = Path(out).expanduser() if out else repo / "parity_robomme.json"
    cmd = [
        sys.executable,
        str(repo / "scripts/deliberation/stage_e_omega_parity.py"),
        "--domain",
        DOMAIN,
        "--encoder",
        encoder_ckpt,
        "--pooled-root",
        str(Path(pooled_root).expanduser()),
        "--omega-root",
        str(Path(omega_root).expanduser()),
        "--lang-mode",
        lang_mode,
        "--demos",
        str(demos),
        "--cos-bar",
        str(cos_bar),
        "--out",
        str(out),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    result = {
        "returncode": completed.returncode,
        "cmd": cmd,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"D7 identity gate FAILED for robomme under lang-mode={lang_mode}; "
            f"do NOT submit M1/M2/M3.\n{completed.stdout[-2000:]}\n"
            f"{completed.stderr[-2000:]}"
        )
    if out.is_file():
        result["report"] = json.loads(out.read_text())
    return result


def attach(server, *, producer_factory) -> set:
    """Install this omega path on an execution server for the demo-prefix arms ONLY.

    Returns the set of arm ids it took ownership of. Every other arm keeps the sealed
    `OnlineWorkspaceRunner`; this function never removes or rebinds an existing arm.
    """
    owned = set(BY_ID)
    arm = getattr(server, "arm", None)
    if arm not in owned:
        return set()
    server._rmme_producer_factory = producer_factory
    server._rmme_sessions = {}
    return {arm}


# ------------------------------------------------------------------------------------ self-test
class _StubProducer:
    """Returns a vector whose every entry is the CALL INDEX, so slot allocation is exactly testable."""

    def __init__(self, dim: int = 512):
        self.dim, self.calls = dim, 0

    def reset(self) -> None:
        self.calls = 0

    def step(self, _p):
        value = np.full((self.dim,), float(self.calls), dtype=np.float32)
        self.calls += 1
        return value


def _self_test(dataset_root: str, per_task: int, tasks: tuple[str, ...]) -> None:
    import pyarrow.parquet as pq

    from workspace_models.features.pi_pooled_tap import frame_selection
    from workspace_models.labels.robomme_source import episode_path, episodes_of

    root = Path(dataset_root).expanduser()
    checked = 0
    coverage = []
    broken_before: list[tuple[bool, int, int]] = []
    for task in tasks:
        for ep in list(episodes_of(task))[:per_task]:
            table = pq.read_table(episode_path(root, ep), columns=["exec_start_idx"])
            n = table.num_rows
            raw = table.column("exec_start_idx").to_pylist()
            start = int(raw[0][0] if isinstance(raw[0], (list, tuple)) else raw[0])

            # (1) the reset-time demo grid must equal the SERVE-ALIGNED store's demo rows EXACTLY
            store_grid = serve_aligned_grid(n, start)
            assert np.array_equal(demo_grid(start), store_grid[store_grid < start]), (task, ep)
            # and every serve frame must be a store row -> coverage 1.0 by construction
            served = serve_frame_indices(start, int(np.ceil((n - start) / EXECUTION_HORIZON)))
            served = served[served < n]
            assert np.isin(served, store_grid).all(), (task, ep, "serve frame off the store grid")
            # the plain wsm_pooled grid is what would have broken: prove it, per episode
            plain = frame_selection(n)
            plain_hits = int(np.isin(served, plain).sum())
            broken_before.append((start % 8 == 0, plain_hits, served.size))

            for arm_id in ("v4_wsm_gdn_demo8_live16_drop02", "v4_wsm_gdn_live16_drop02", "v4_wsm_gdn_demo8_drop02"):
                session = RmmeServeOmegaSession(_StubProducer(4), arm_id=arm_id, execution_horizon=16)
                produced = session.on_episode_restart(np.zeros((max(start, 0), 4), np.float32))
                assert produced == demo_grid(start).size, (task, ep, produced)
                # (2) drive the episode decision by decision
                decision = 0
                while session.step_index(decision) < n:
                    step = session.on_decision(np.zeros((4,), np.float32))
                    assert step == start + 16 * decision
                    record = session.record()
                    assert record["exec_start_idx"] == start
                    # (3) causality: the newest row never sits after the current step
                    assert record["frame_indices"][row_for_step(record["frame_indices"], step)] <= step
                    window, segment, valid = session.window(step)
                    spec = BY_ID[arm_id]
                    k = (spec.k_demo if spec.read != "m1_live_only" else 0) + (
                        spec.k_live if spec.read != "m2_demo_only" else 0
                    )
                    assert window.shape == (k, 4), (arm_id, window.shape, k)
                    if spec.read != "m1_live_only":
                        assert segment[: spec.k_demo].tolist() == [0] * spec.k_demo
                        assert bool(valid.all()) == (start > 0)
                        # demo slots come from rows produced BEFORE any live frame
                        assert (window[: spec.k_demo, 0] < max(produced, 1)).all() or start == 0
                    if spec.read != "m2_demo_only":
                        assert segment[-spec.k_live :].tolist() == [1] * spec.k_live
                    decision += 1
                    checked += 1
                    if decision > 400:
                        break
                if arm_id == "v4_wsm_gdn_demo8_live16_drop02":
                    cov = session.realized_grid_coverage(episode_length=n)
                    assert cov["demo_exact"], (task, ep, cov)
                    assert cov["skipped_offgrid"] == 0, (task, ep, cov)
                    assert cov["live_coverage"] == 1.0, (task, ep, cov)
                    coverage.append((task, ep, cov))
    print(
        f"[serve] {checked} decision-windows over {len(coverage)} real episodes: "
        f"reset grid exact, causality holds, segments correct"
    )
    print("[serve] LIVE-GRID ALIGNMENT on the serve-aligned grid:")
    print(f"  {'task':<18}{'exec_start':>11}{'live_prod':>10}{'live_store':>11}{'coverage':>10}{'offgrid':>9}")
    for task, ep, cov in coverage[:8]:
        print(
            f"  {task:<18}{cov['exec_start_idx']:>11}{cov['live_rows']:>10}"
            f"{cov['live_rows_expected']:>11}{str(cov['live_coverage']):>10}"
            f"{cov['skipped_offgrid']:>9}"
        )
    values = [c["live_coverage"] for _t, _e, c in coverage if c["live_coverage"] is not None]
    print(
        f"  live coverage over {len(values)} episodes: min {min(values):.3f} "
        f"median {float(np.median(values)):.3f} max {max(values):.3f}  (bar: exactly 1.0)"
    )
    aligned = sum(1 for ok, _h, _s in broken_before if ok)
    hits = sum(h for _ok, h, _s in broken_before)
    served_total = sum(s for _ok, _h, s in broken_before)
    print(
        f"[serve] COUNTERFACTUAL on the plain wsm_pooled grid (arange(0,n,8)): "
        f"{hits}/{served_total} serve frames would have been grid rows, and only "
        f"{aligned}/{len(broken_before)} episodes have exec_start_idx % 8 == 0"
    )
    print("[serve] PASS")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--dataset-root",
        default="~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
        "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20",
    )
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--tasks", default="VideoPlaceOrder,MoveCube,VideoUnmask,PickXtimes")
    args = ap.parse_args()
    if not args.self_test:
        raise SystemExit("this module is an overlay; pass --self-test to exercise it")
    _self_test(args.dataset_root, args.per_task, tuple(t.strip() for t in args.tasks.split(",") if t.strip()))


if __name__ == "__main__":
    main()
