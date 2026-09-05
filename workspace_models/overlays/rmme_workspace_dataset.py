#!/usr/bin/env python3
"""Training-side plumbing: the [demo ; live] omega window per RoboMME training sample.

The sealed `RoboMMEWorkspaceExecutionDataset` reads the legacy DENSE `omega_f16.npy` (one row per
episode step) and slices `step - stride*arange(...)` clamped at 0. The M-arms read the Stage-E
`w.npz` store instead, which lives on the tap's stride-8 grid and covers demo AND execution frames.
This wrapper is the bridge; it does not modify the sealed class.

Everything it produces goes through `workspace_models.features.rmme_demo_prefix.window_for_step`,
which is the SAME call the eval server makes. One definition, both sides — a divergence here is
what invalidated an earlier eval in this study.

Row contract, unchanged from the sealed class:

  * only EXECUTION rows carry a workspace token (`is_demo` rows raise);
  * `step` is the episode-global `step_idx` straight off the parquet;
  * the emitted key is `wsm_w_window` [K, 512] float32, oldest -> newest, which
    `RoboMMEInputs` already accepts at any K >= 1 (`training/data.py:151-154`).

Additive keys, used by preflight and by the per-suite reading, ignored by the model:

  * `wsm_w_segment`      [K] int8   0 = demo slot, 1 = live slot
  * `wsm_w_prefix_valid` [k_demo]   False on the 7 tasks that have no demonstration video

Dry run (CPU, real grids and real step indices, synthetic omega so every slot is checkable):

    PYTHONPATH=<repo> python -m workspace_models.overlays.rmme_workspace_dataset --dry-run
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from workspace_models.features.rmme_demo_prefix import (
    DEFAULT_K_DEMO,
    DEFAULT_K_LIVE,
    DEFAULT_STRIDE,
    demo_prefix_slots,
    episode_record,
    live_window_rows,
    load_episode,
    window_for_step,
)
from workspace_models.overlays.rmme_arms import BY_ID


@dataclasses.dataclass(frozen=True)
class WindowSpec:
    read: str
    k_demo: int
    k_live: int
    stride_steps: int = DEFAULT_STRIDE

    @classmethod
    def for_arm(cls, arm_id: str, stride_steps: int = DEFAULT_STRIDE) -> "WindowSpec":
        arm = BY_ID.get(arm_id)
        if arm is None:
            raise ValueError(f"{arm_id!r} is not a registered RoboMME demo-prefix arm (known: {sorted(BY_ID)})")
        return cls(arm.read, arm.k_demo, arm.k_live, stride_steps)

    @property
    def window(self) -> int:
        return (self.k_demo if self.read != "m1_live_only" else 0) + (
            self.k_live if self.read != "m2_demo_only" else 0
        )


class RmmeStageEWorkspaceDataset:
    """Wrap any indexable RoboMME row source and attach the arm's omega window.

    `rows` yields `(episode_index, step_idx, is_demo, task)` for row `i`; on the node that is the
    sealed dataset's own scalar columns, and in `--dry-run` it is built straight from the parquet.
    `base[i]` (optional) supplies the transformed sample the window is attached to.
    """

    def __init__(
        self,
        rows,
        *,
        spec: WindowSpec,
        omega_root,
        pooled_root=None,
        base=None,
        domain: str = "robomme",
        record_loader=None,
    ):
        self.rows = rows
        self.spec = spec
        self.omega_root = Path(omega_root).expanduser() if omega_root is not None else None
        self.pooled_root = Path(pooled_root).expanduser() if pooled_root is not None else None
        self.base = base
        self.domain = domain
        self._record_loader = record_loader
        self._cache: dict[tuple[str, int], dict] = {}

    def _record(self, task: str, episode: int) -> dict:
        key = (task, int(episode))
        record = self._cache.get(key)
        if record is None:
            if self._record_loader is not None:
                record = self._record_loader(task, int(episode))
            else:
                record = load_episode(
                    self.omega_root, task, int(episode), pooled_root=self.pooled_root, domain=self.domain
                )
            self._cache[key] = record
        return record

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        episode, step, is_demo, task = self.rows[int(index)]
        if is_demo:
            raise IndexError("workspace tokens are defined only on RoboMME execution rows")
        record = self._record(task, int(episode))
        window, segment, prefix_valid = window_for_step(
            self.spec.read,
            record,
            int(step),
            k_demo=max(self.spec.k_demo, 1),
            k_live=max(self.spec.k_live, 1),
            stride_steps=self.spec.stride_steps,
        )
        if window.shape != (self.spec.window, record["w"].shape[1]):
            raise RuntimeError(f"window {window.shape} != expected ({self.spec.window}, {record['w'].shape[1]})")
        data = dict(self.base[int(index)]) if self.base is not None else {}
        data["wsm_w_window"] = window.astype(np.float32, copy=False)
        data["wsm_w_segment"] = segment
        data["wsm_w_prefix_valid"] = prefix_valid
        return data


# ------------------------------------------------------------------------------------- dry run
def _dry_run(dataset_root: str, episodes_per_task: int, tasks: tuple[str, ...]) -> None:
    import pyarrow.parquet as pq

    from workspace_models.features.pi_pooled_tap import frame_selection
    from workspace_models.labels.robomme_source import episode_path, episodes_of

    root = Path(dataset_root).expanduser()
    checked = {"rows": 0, "episodes": 0, "with_demo": 0, "without_demo": 0}
    records: dict[tuple[str, int], dict] = {}
    rows: list[tuple[int, int, bool, str]] = []

    for task in tasks:
        for ep in list(episodes_of(task))[:episodes_per_task]:
            table = pq.read_table(episode_path(root, ep), columns=["is_demo", "exec_start_idx", "step_idx"])
            n = table.num_rows
            grid = frame_selection(n)
            raw = table.column("exec_start_idx").to_pylist()
            start = int(raw[0][0] if isinstance(raw[0], (list, tuple)) else raw[0])
            demo_col = np.asarray(
                [v[0] if isinstance(v, (list, tuple)) else v for v in table.column("is_demo").to_pylist()], dtype=bool
            )
            step_col = np.asarray(
                [v[0] if isinstance(v, (list, tuple)) else v for v in table.column("step_idx").to_pylist()],
                dtype=np.int64,
            )
            # SYNTHETIC omega whose value IS its row index, so "which slot got which row" is an
            # exact equality test rather than a plausibility check.
            w = np.repeat(np.arange(len(grid), dtype=np.float32)[:, None], 512, axis=1)
            records[(task, ep)] = episode_record(w, grid, start, task=task, episode=ep)
            checked["episodes"] += 1
            checked["with_demo" if start else "without_demo"] += 1
            # sample execution rows across the episode: first, quartiles, last
            exec_rows = np.flatnonzero(~demo_col)
            picks = exec_rows[np.unique(np.linspace(0, exec_rows.size - 1, 6).astype(int))]
            rows.extend((ep, int(step_col[r]), bool(demo_col[r]), task) for r in picks)

    print(
        f"[dry-run] {checked['episodes']} real episodes "
        f"({checked['with_demo']} with a demo, {checked['without_demo']} without), "
        f"{len(rows)} execution rows"
    )

    for arm_id in (
        "v4_wsm_gdn_live16_drop02",
        "v4_wsm_gdn_demo8_drop02",
        "v4_wsm_gdn_demo8_live16_drop02",
        "v4_wsm_gdn_demo8_live16_drop02_ctrl0b",
    ):
        spec = WindowSpec.for_arm(arm_id)
        ds = RmmeStageEWorkspaceDataset(
            rows, spec=spec, omega_root=None, record_loader=lambda task, ep: records[(task, ep)]
        )
        assert len(ds) == len(rows)
        for i in range(len(ds)):
            episode, step, _is_demo, task = rows[i]
            item = ds[i]
            window = item["wsm_w_window"]
            segment = item["wsm_w_segment"]
            record = records[(task, episode)]
            grid, start = record["frame_indices"], record["exec_start_idx"]
            assert window.shape == (spec.window, 512), (arm_id, window.shape)
            assert window.dtype == np.float32
            # every slot's VALUE is its omega row -> read the allocation straight back out
            got = window[:, 0].astype(np.int64)
            if spec.read != "m1_live_only":
                want_demo, valid = demo_prefix_slots(grid, start, spec.k_demo)
                assert np.array_equal(got[: spec.k_demo], want_demo), (arm_id, task, episode, step)
                assert np.array_equal(segment[: spec.k_demo], np.zeros(spec.k_demo, np.int8))
                assert np.array_equal(item["wsm_w_prefix_valid"], valid)
                if start:
                    assert (grid[got[: spec.k_demo]] < start).all(), "demo slot after exec start"
                else:
                    assert (got[: spec.k_demo] == 0).all(), "no-demo clamp broke"
            if spec.read != "m2_demo_only":
                want_live = live_window_rows(grid, step, spec.k_live, spec.stride_steps)
                assert np.array_equal(got[-spec.k_live :], want_live), (arm_id, task, episode, step)
                assert np.array_equal(segment[-spec.k_live :], np.ones(spec.k_live, np.int8))
                assert grid[got[-1]] <= step, "newest live slot reads the future"
            checked["rows"] += 1
        print(
            f"[dry-run] {arm_id:<40} K={spec.window:>2}  "
            f"{len(rows)} rows: demo slots = prefix omega, live slots = causal window  OK"
        )

    # the four arms must actually differ on the same row
    windows = {}
    for arm_id in ("v4_wsm_gdn_live16_drop02", "v4_wsm_gdn_demo8_drop02", "v4_wsm_gdn_demo8_live16_drop02"):
        spec = WindowSpec.for_arm(arm_id)
        ds = RmmeStageEWorkspaceDataset(
            rows, spec=spec, omega_root=None, record_loader=lambda task, ep: records[(task, ep)]
        )
        windows[arm_id] = ds[0]["wsm_w_window"]
    m1, m2, m3 = (
        windows["v4_wsm_gdn_live16_drop02"],
        windows["v4_wsm_gdn_demo8_drop02"],
        windows["v4_wsm_gdn_demo8_live16_drop02"],
    )
    assert m1.shape != m2.shape and m3.shape[0] == m1.shape[0] + m2.shape[0]
    assert np.array_equal(m3[:DEFAULT_K_DEMO], m2) and np.array_equal(m3[DEFAULT_K_DEMO:], m1)
    print(
        f"[dry-run] arms distinct and compositional: M3 == [M2 ; M1] "
        f"({DEFAULT_K_DEMO} + {DEFAULT_K_LIVE} = {DEFAULT_K_DEMO + DEFAULT_K_LIVE})"
    )
    print(f"[dry-run] PASS — {checked['rows']} arm x row windows verified")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--dataset-root",
        default="~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
        "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20",
    )
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument(
        "--tasks",
        default="VideoPlaceOrder,MoveCube,VideoUnmask,PickXtimes",
        help="default deliberately includes PickXtimes, which has NO demo video",
    )
    args = ap.parse_args()
    if not args.dry_run:
        raise SystemExit("this module is a library; pass --dry-run to exercise it")
    _dry_run(args.dataset_root, args.per_task, tuple(t.strip() for t in args.tasks.split(",") if t.strip()))


if __name__ == "__main__":
    main()
