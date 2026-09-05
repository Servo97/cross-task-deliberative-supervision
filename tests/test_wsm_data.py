"""Correctness tests for the async WSM-training data pipeline (data.py): WSMSampleDataset + wsm_collate.

Builds tiny SYNTHETIC cache files (small dims) matching the on-disk schema, so the real code path
(load_demo -> cond_lang -> _gather_targets -> collate) is exercised with no dependency on the
multi-hundred-GB feature cache. CPU-only, no model, no GPU.

Run:  PYTHONPATH=. <torch-python> -m pytest tests/test_wsm_data.py -q
  or: PYTHONPATH=. <torch-python> tests/test_wsm_data.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from workspace_models.train.train_wsm_base.data import WSMSampleDataset, worker_init_fn, wsm_collate

P, DB, DP, DL = 4, 8, 6, 8  # tiny patch-grid / backbone / proprio / lang dims


def _make_demo(root: Path, task: str, demo_id: int, F: int, seed: int) -> dict:
    """Write one synthetic cached demo (patch_tokens.npy + feats.npz + labels npz); return a manifest row."""
    rng = np.random.default_rng(seed)
    fd = root / task / f"demo_{demo_id:06d}"
    fd.mkdir(parents=True, exist_ok=True)
    np.save(fd / "patch_tokens.npy", rng.standard_normal((F, P, DB)).astype(np.float16))
    frame_indices = (np.arange(F) * 8).astype(np.int64)
    n_sub = 3
    np.savez(
        fd / "feats.npz",
        state_emb=rng.standard_normal((F, DP)).astype(np.float16),
        lang_global=rng.standard_normal(DL).astype(np.float16),
        subgoal_embs=rng.standard_normal((n_sub, DL)).astype(np.float16),
        frame_indices=frame_indices,
    )
    keyframes = np.array([8, 8 * (F // 2), 8 * (F - 1)], dtype=np.int64)  # 3 keyframes in-range
    salient = np.array([np.array([0, 1]), np.array([2]), np.array([1, 3])], dtype=object)
    lab = root / task / f"vlm_episode_{demo_id:06d}.npz"
    np.savez(lab, keyframes=keyframes, salient_global=salient, cumulative_global=salient)
    return {
        "task": task,
        "demo_id": demo_id,
        "feature_dir": str(fd),
        "labels_path": str(lab),
        "partner_demo_ids": json.dumps([(demo_id + 1) % 4]),
    }


def _rows(root: Path) -> list[dict]:
    return [_make_demo(root, "TaskA", i, F=5 + i, seed=i) for i in range(4)]


def test_sample_shapes_and_dtype():
    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        ds = WSMSampleDataset(rows, dropout=0.4, mode="next", align=False, seed=0)
        it = ds[2]
        assert it["patch"].dtype == torch.float16 and it["lang"].dtype == torch.float16
        assert it["F"] == 7 and it["patch"].shape == (7, P, DB)  # F == 5 + idx(2)
        assert it["state"].shape == (7, DP) and it["lang"].shape == (7, DL)
        assert it["partner"] is None
        assert len(it["sup"]) == 3  # 3 keyframes, each has salient patches
        for pos, feats in it["sup"]:
            assert 0 <= pos < it["F"] and feats.dtype == torch.float16 and feats.shape[1] == DB


def test_collate_padding_is_zero_beyond_valid():
    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        ds = WSMSampleDataset(rows, dropout=0.0, align=False, seed=0)
        batch = wsm_collate([ds[i] for i in range(4)])
        B, T = batch["patch"].shape[0], batch["patch"].shape[1]
        assert B == 4 and T == max(int(x) for x in batch["t_valid"])
        for i in range(B):
            Fi = int(batch["t_valid"][i])
            assert torch.count_nonzero(batch["patch"][i, Fi:]) == 0
            assert torch.count_nonzero(batch["state"][i, Fi:]) == 0
            assert torch.count_nonzero(batch["lang"][i, Fi:]) == 0
        assert len(batch["sup"]) == B and len(batch["kf_pos"]) == B


def test_seed_determinism():
    """Same seed -> identical per-frame language dropout (reproducible runs)."""
    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        a = WSMSampleDataset(rows, dropout=0.5, align=False, seed=0)[1]["lang"]
        b = WSMSampleDataset(rows, dropout=0.5, align=False, seed=0)[1]["lang"]
        assert torch.equal(a, b)


def test_dropout_zero_is_all_global():
    """dropout=0 => every frame uses a SUBGOAL emb (never the global); dropout=1 => all global."""
    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        ds0 = WSMSampleDataset(rows, dropout=0.0, align=False, seed=0)[0]["lang"]
        ds1 = WSMSampleDataset(rows, dropout=1.0, align=False, seed=0)[0]["lang"]
        # under dropout=1 all frames share the single global vector -> all rows identical
        assert torch.allclose(ds1, ds1[0:1].expand_as(ds1))
        assert not torch.allclose(ds0, ds0[0:1].expand_as(ds0))


def test_align_path_builds_partner_tensors():
    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        ds = WSMSampleDataset(rows, dropout=0.4, align=True, seed=0, lut_rows=rows)
        batch = wsm_collate([ds[i] for i in range(4)])
        for k in ("patch_p", "state_p", "lang_p", "t_valid_p", "kf_pos_p"):
            assert k in batch
        assert batch["patch_p"].shape[0] == 4 and batch["patch_p"].shape[-1] == DB
        assert len(batch["kf_pos_p"]) == 4


def test_worker_equivalence():
    """workers>0 must yield the same batch as workers=0 for a fixed sampler order."""
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        ds = WSMSampleDataset(rows, dropout=0.0, align=False, seed=0)  # dropout 0 => worker-rng-independent

        def first(nw):
            ld = DataLoader(
                ds,
                batch_size=2,
                shuffle=False,
                drop_last=True,
                collate_fn=wsm_collate,
                num_workers=nw,
                prefetch_factor=2 if nw else None,
                worker_init_fn=worker_init_fn if nw else None,
            )
            return next(iter(ld))

        b0, b2 = first(0), first(2)
        assert torch.equal(b0["patch"], b2["patch"]) and torch.equal(b0["lang"], b2["lang"])


def test_worker_init_unwraps_subset():
    # attempt-6 producer crash: rank-strided val sharding wraps the dataset in Subset, which has no
    # .seed — worker_init_fn must resolve through wrapper .dataset chains to the WSMSampleDataset.
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as td:
        rows = _rows(Path(td))
        ds = WSMSampleDataset(rows, dropout=0.0, align=False, seed=0)
        sub = torch.utils.data.Subset(ds, list(range(0, len(ds), 2)))
        ld = DataLoader(
            sub,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            collate_fn=wsm_collate,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )
        batches = list(ld)
        assert len(batches) == 1 and batches[0]["patch"].shape[0] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("all data-pipeline tests passed")
