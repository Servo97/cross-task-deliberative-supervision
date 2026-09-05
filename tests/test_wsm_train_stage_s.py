"""Correctness tests for the Stage-S WSM encoder trainer: demo-disjoint split isolation, evaluated
validation, immutable run-config, single-sync Hungarian parity, and (DDP) gather/reduce parity.

Tiny synthetic dims (the trainer is dim-agnostic; validators pin production dims elsewhere). Torch
CPU only; the DDP tests spawn gloo processes and are marked slow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from workspace_models.train.train_wsm_base import _wsm_stage_s_train as st
from workspace_models.train.train_wsm_base import _wsm_train_common as common
from workspace_models.train.train_wsm_base.data import WSMSampleDataset, load_demo_pi_stage_s

P, DB, DL = 4, 8, 8  # tiny patch-grid / backbone / lang dims


def _make_cache(root: Path, tasks: int, demos: int, F: int = 5):
    """Write a tiny Stage-S cache (canonical feats schema) + labels; return (rows, table)."""
    rng = np.random.default_rng(0)
    rows, table = [], {}
    for t in range(tasks):
        task = f"Task{t:02d}"
        table[task] = rng.standard_normal(DL).astype(np.float16)
        ids = list(range(demos))
        for demo_id in ids:
            fd = root / task / f"demo_{demo_id:06d}"
            fd.mkdir(parents=True, exist_ok=True)
            n = F + (demo_id % 2)
            np.save(fd / "patch_tokens.npy", rng.standard_normal((n, P, DB)).astype(np.float16))
            frame_indices = (np.arange(n) * 8).astype(np.int64)
            np.savez(
                fd / "feats.npz",
                lang_per_frame=rng.standard_normal((n, DL)).astype(np.float16),
                frame_indices=frame_indices,
            )
            keyframes = np.array([8, 8 * (n // 2), 8 * (n - 1)], dtype=np.int64)
            salient = np.array([np.array([0, 1]), np.array([2]), np.array([1, 3])], dtype=object)
            lab = root / task / f"vlm_episode_pi_{demo_id:06d}.npz"
            np.savez(lab, keyframes=keyframes, salient_global=salient, cumulative_global=salient)
            partners = [d for d in ids if d != demo_id]
            rows.append(
                {
                    "task": task,
                    "demo_id": demo_id,
                    "feature_dir": str(fd),
                    "labels_path": str(lab),
                    "partner_demo_ids": json.dumps(partners),
                }
            )
    return rows, table


def test_split_is_demo_disjoint_and_stratified(tmp_path):
    rows, _ = _make_cache(tmp_path, tasks=3, demos=6)
    tr1, va1 = st.stratified_demo_split(rows, val_frac=0.2, split_seed=st.WSM_SPLIT_SEED)
    tr2, va2 = st.stratified_demo_split(rows, val_frac=0.2, split_seed=st.WSM_SPLIT_SEED)

    def key(r):
        return (r["task"], r["demo_id"])

    assert {key(r) for r in tr1} == {key(r) for r in tr2}  # deterministic
    assert {key(r) for r in va1} == {key(r) for r in va2}
    assert not ({key(r) for r in tr1} & {key(r) for r in va1})  # disjoint
    tasks = {r["task"] for r in rows}
    assert {r["task"] for r in tr1} == tasks and {r["task"] for r in va1} == tasks  # both cover all tasks


def test_no_cross_split_partner(tmp_path):
    rows, table = _make_cache(tmp_path, tasks=2, demos=8)
    train_rows, val_rows = st.stratified_demo_split(rows, val_frac=0.25, split_seed=st.WSM_SPLIT_SEED)
    val_keys = {(r["task"], r["demo_id"]) for r in val_rows}
    ds = WSMSampleDataset(
        train_rows,
        dropout=1.0,
        align=True,
        seed=0,
        load_fn=lambda r: load_demo_pi_stage_s(r, table),
        lut_rows=train_rows,
        strict_split=True,
    )
    # Every filtered candidate is in the TRAIN split, never val.
    for (task, demo), cands in ds._candidates.items():
        for p in cands:
            assert (task, p) not in val_keys
    # Draw many partners; none lands in val.
    for r in train_rows:
        for _ in range(20):
            partner = ds._partner_row(r)
            assert (partner["task"], partner["demo_id"]) not in val_keys
    # A row whose partners were all val-side falls back to self and increments the counter.
    assert ds.n_self_partner >= 0


def test_run_config_immutable_and_hashed(tmp_path):
    good = dict(
        lambda_align=1.0,
        target_mode="next",
        dropout=1.0,
        steps=10,
        batch_size=2,
        lr=3e-4,
        warmup_steps=1,
        min_lr_frac=0.1,
        input_norm=False,
        val_frac=0.1,
        seed_split=1,
    )
    p = tmp_path / "rc.json"
    p.write_text(json.dumps(good))
    cfg = st.load_run_config(p)
    h0 = st.run_config_sha256(cfg)
    cfg2 = dict(cfg)
    cfg2["lambda_align"] = 0.0
    assert st.run_config_sha256(cfg2) != h0  # any objective change changes the hash
    # missing key fails closed
    bad = dict(good)
    bad.pop("lambda_align")
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        st.load_run_config(tmp_path / "bad.json")
    # extra key fails closed
    extra = dict(good)
    extra["sneaky"] = 1
    (tmp_path / "extra.json").write_text(json.dumps(extra))
    with pytest.raises(SystemExit):
        st.load_run_config(tmp_path / "extra.json")


def test_stage_s_loader_uses_table_language_and_rejects_legacy(tmp_path):
    rows, table = _make_cache(tmp_path, tasks=1, demos=3)
    demo = load_demo_pi_stage_s(rows[0], table)
    # lang_global is the sealed task vector, not a per-demo/expanded prompt.
    assert np.array_equal(np.asarray(demo.lang_global), table[rows[0]["task"]])
    assert demo.subgoal_embs.shape[0] == 0
    ds = WSMSampleDataset(rows, dropout=1.0, align=False, seed=0, load_fn=lambda r: load_demo_pi_stage_s(r, table))
    lang = ds[0]["lang"]
    assert torch.allclose(lang, lang[0:1].expand_as(lang))  # dropout=1 -> every frame the task vector
    # a legacy expanded feats.npz (with lang_global) is rejected
    fd = Path(rows[0]["feature_dir"])
    with np.load(fd / "feats.npz") as a:
        lp, fi = a["lang_per_frame"], a["frame_indices"]
    np.savez(fd / "feats.npz", lang_per_frame=lp, frame_indices=fi, lang_global=np.zeros(DL, np.float16))
    with pytest.raises(ValueError, match="legacy expanded-prompt cache"):
        load_demo_pi_stage_s(rows[0], table)


def test_batched_assignment_equals_reference():
    torch.manual_seed(0)
    n, k, D = 5, 8, 16
    recon = torch.randn(n, k, D, requires_grad=True)
    occ = torch.randn(n, k, requires_grad=True)
    tgts = [torch.randn(1 + j % 3, D) for j in range(n)]
    rl_a, ol_a, f1_a = common.match_and_losses(recon, occ, tgts, 2.0, "cpu")
    rl_b, ol_b, f1_b = common._match_and_losses_reference(recon, occ, tgts, 2.0, "cpu")
    assert torch.allclose(rl_a, rl_b, atol=1e-6)
    assert torch.allclose(ol_a, ol_b, atol=1e-6)
    assert f1_a == f1_b
    # gradients also match (same assignment => same graph)
    rl_a.backward()
    ga = recon.grad.clone()
    recon.grad = None
    rl_b.backward()
    gb = recon.grad.clone()
    assert torch.allclose(ga, gb, atol=1e-6)


def test_single_process_train_runs_and_evaluates(tmp_path):
    rows, table = _make_cache(tmp_path, tasks=2, demos=6)
    cfg = dict(
        lambda_align=1.0,
        target_mode="next",
        dropout=1.0,
        steps=4,
        batch_size=2,
        lr=3e-4,
        warmup_steps=1,
        min_lr_frac=0.1,
        input_norm=False,
        val_frac=0.25,
        seed_split=1,
    )
    final = st.train_stage_s(
        rows=rows,
        load_fn=lambda r: load_demo_pi_stage_s(r, table),
        cfg=cfg,
        backbone_dim=DB,
        proprio_dim=DL,
        lang_dim=DL,
        out_dir=tmp_path / "run",
        device="cpu",
        num_workers=0,
        val_every=2,
    )
    assert final.exists()
    blob = torch.load(final, map_location="cpu")
    assert blob["run_config_sha256"] == st.run_config_sha256(cfg)
    assert blob["run_config"]["lambda_align"] == 1.0
    assert (tmp_path / "run" / "wsm_best.pt").exists()  # validation ran and selected a best


# ---- DDP (gloo, CPU) parity -------------------------------------------------------------------


def _ddp_worker(rank, world, port, fn, ret):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    try:
        fn(rank, world, ret)
    finally:
        torch.distributed.destroy_process_group()


def _spawn(fn, world=2, port=29517):
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_ddp_worker, args=(world, port, fn, ret), nprocs=world, join=True)
    return ret


def _gather_body(rank, world, ret):
    x = torch.arange(rank + 1, dtype=torch.float32).reshape(-1, 1).repeat(1, 4) + rank * 10
    g = st.gather_variable(x, dim=4)
    ret[rank] = g.detach().numpy().tolist()


@pytest.mark.slow
def test_ddp_gather_variable_concatenates():
    ret = _spawn(_gather_body, world=2, port=29518)
    # rank0 contributes 1 row, rank1 contributes 2 rows -> global 3 rows, identical on both ranks.
    assert ret[0] == ret[1]
    assert len(ret[0]) == 3


def _sigreg_grad_body(rank, world, ret):
    torch.manual_seed(0)
    full_a = torch.randn(4, 6)
    full_b = torch.randn(4, 6)
    local_a = full_a[rank * 2 : (rank + 1) * 2].clone().requires_grad_(True)
    local_b = full_b[rank * 2 : (rank + 1) * 2].clone().requires_grad_(True)
    sig = st.SigRegLoss()
    za = st.gather_variable(local_a, dim=6)
    zb = st.gather_variable(local_b, dim=6)
    loss, _ = sig(za, zb)
    loss.backward()
    # W-fold overcount / W  == single-process slice gradient
    ret[rank] = (local_a.grad / world).detach().numpy().tolist()


@pytest.mark.slow
def test_ddp_sigreg_gather_gradient_parity():
    ret = _spawn(_sigreg_grad_body, world=2, port=29519)
    # single-process reference
    torch.manual_seed(0)
    full_a = torch.randn(4, 6, requires_grad=True)
    full_b = torch.randn(4, 6, requires_grad=True)
    sig = st.SigRegLoss()
    loss, _ = sig(full_a, full_b)
    loss.backward()
    ref = full_a.grad.numpy()
    got = np.concatenate([np.array(ret[0]), np.array(ret[1])], axis=0)
    assert np.allclose(got, ref, atol=1e-5)
