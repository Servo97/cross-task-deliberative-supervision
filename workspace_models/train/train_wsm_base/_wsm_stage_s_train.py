"""Stage-S (D0) WSM encoder trainer: demonstration-disjoint split, evaluated validation, immutable
objective config, single-sync Hungarian, and optional 8-GPU DDP.

This is a SEPARATE core from the legacy `_wsm_train_common.train()` so the groot/pi historical
entries stay byte-identical. It reuses the pure loss helpers (`match_and_losses`,
`collect_align_pairs`, checkpoint/lr helpers) from that module; the new behavior is:

  * split-by-demonstration FIRST, then build partner lookup separately inside each split
    (`WSMSampleDataset(strict_split=True)`), so training can never draw a validation partner;
  * a real, deterministic, evaluated validation pass (`--val-every`) that also selects `wsm_best.pt`;
  * an immutable `--run-config` JSON whose SHA is stored in the checkpoint — you cannot silently
    launch a `lambda_align=0` recipe while claiming the arm includes SigReg;
  * DDP (one process/GPU) activated by `WORLD_SIZE>1` (torchrun); single-process stays the default
    and is unchanged.

Governed by internal_planning_and_todos/_archive/handover_to_opus/{03,04}_packet_*.md.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

from workspace_models.networks.sigreg_loss import SigRegLoss
from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig
from workspace_models.train.train_wsm_base._wsm_train_common import (
    _atomic_torch_save,
    _to_gpu_fp32,
    collect_align_pairs,
    match_and_losses,
)
from workspace_models.train.train_wsm_base.data import (
    WSMSampleDataset,
    _gather_targets,
    worker_init_fn,
    wsm_collate,
)

WSM_SPLIT_SEED = 20260722
# The objective knobs that are frozen into the run-config identity. Changing ANY of these changes
# run_config_sha256, which is stored in the checkpoint and (via packet 05) the encoder_id.
RUN_CONFIG_KEYS = (
    "lambda_align",
    "target_mode",
    "dropout",
    "steps",
    "batch_size",
    "lr",
    "warmup_steps",
    "min_lr_frac",
    "input_norm",
    "val_frac",
    "seed_split",
)


# --------------------------------------------------------------------------------------------------
# Distributed helpers (no-ops in single-process).
# --------------------------------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def gather_variable(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Autograd-aware global concatenation of a per-rank [m,dim] tensor (m may differ per rank).

    In single-process this is the identity. Under DDP it all-gathers padded rows (differentiably, so
    grad routes back to each rank's local rows) and slices by the gathered per-rank sizes, so every
    rank ends up with the SAME global set for the anti-collapse statistics."""
    if not is_dist():
        return x
    world = get_world_size()
    m = torch.tensor([x.shape[0]], device=x.device)
    sizes = [torch.zeros_like(m) for _ in range(world)]
    dist.all_gather(sizes, m)
    counts = [int(s.item()) for s in sizes]
    max_m = max(counts) if counts else 0
    # cat (not zeros+copy) so x's autograd graph reaches `padded` even when x has 0 rows; and when
    # it STILL doesn't require grad (0-row placeholder), force it while grad mode is on. Otherwise a
    # rank with no local rows records no autograd node here, its backward never issues the gather's
    # reduce_scatter, and every other rank blocks in that collective -> NCCL hang (audit 2026-07-23).
    padded = torch.cat([x, x.new_zeros(max_m - x.shape[0], dim)], dim=0)
    if torch.is_grad_enabled() and not padded.requires_grad:
        padded = padded.detach().requires_grad_()
    from torch.distributed.nn.functional import all_gather as diff_all_gather

    gathered = diff_all_gather(padded)  # list[world] of [max_m, dim], autograd-aware
    parts = [g[: counts[r]] for r, g in enumerate(gathered) if counts[r] > 0]
    return torch.cat(parts, dim=0) if parts else x.new_zeros(0, dim)


# --------------------------------------------------------------------------------------------------
# Split, config, and step.
# --------------------------------------------------------------------------------------------------
def stratified_demo_split(rows: list[dict], val_frac: float, split_seed: int) -> tuple[list[dict], list[dict]]:
    """Per task, hold out ceil(val_frac * n) demonstrations for validation (deterministic).

    Returns (train_rows, val_rows), disjoint by (task, demo_id). Every task appears in BOTH splits
    (assuming >=2 demos/task) so the cross-demo alignment always has same-task partners on each side."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(str(r["task"]), []).append(r)
    train_rows, val_rows = [], []
    for task in sorted(by_task):
        demos = sorted(by_task[task], key=lambda r: int(r["demo_id"]))
        n = len(demos)
        n_val = max(1, math.ceil(val_frac * n)) if n >= 2 else 0
        order = np.random.default_rng(split_seed).permutation(n)
        val_pos = set(order[:n_val].tolist())
        for i, r in enumerate(demos):
            (val_rows if i in val_pos else train_rows).append(r)
    return train_rows, val_rows


def run_config_sha256(cfg: dict) -> str:
    import hashlib

    payload = {k: cfg[k] for k in RUN_CONFIG_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_run_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in RUN_CONFIG_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"--run-config is missing required keys: {missing}")
    extra = set(cfg) - set(RUN_CONFIG_KEYS)
    if extra:
        raise SystemExit(f"--run-config has unexpected keys: {sorted(extra)}")
    return cfg


def _average_gradients(model) -> None:
    """Manual DDP: all-reduce-SUM then divide by world_size so every rank holds identical, correct
    global-mean gradients. We do NOT use nn.DistributedDataParallel because WorkspaceModel exposes
    separate encode()/decode() methods (DDP only hooks forward, so it would miss decode's params).

    With differentiable all-gather feeding the global SigReg statistic, the alignment gradient is
    over-counted W-fold by the gather's reduce_scatter backward; averaging by W cancels it exactly,
    while the local recon/occ gradients average to the global mean. See packet 04 D5."""
    if not is_dist():
        return
    world = get_world_size()
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)  # keep every rank's reduce set identical
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world


def run_step_stage_s(model, sigreg, batch, cfg, feat_scale, device, train, opt=None):
    """One optimizer step. recon/occ are local per rank; the SigReg alignment is computed on the
    GLOBAL (gathered) positive set so it is a single statistic across the DDP batch, not eight
    independent local regularizers. Returns a metrics dict (or None if no supervised points)."""
    align_on = cfg["lambda_align"] != 0
    dim = model.cfg.dim

    patch = _to_gpu_fp32(batch, "patch", device)
    state = _to_gpu_fp32(batch, "state", device)
    lang = _to_gpu_fp32(batch, "lang", device)
    w = model.encode(patch, state, lang)
    if align_on:
        w_p = model.encode(
            _to_gpu_fp32(batch, "patch_p", device),
            _to_gpu_fp32(batch, "state_p", device),
            _to_gpu_fp32(batch, "lang_p", device),
        )

    w_sup, lang_sup, tgts = [], [], []
    for b, sup_b in enumerate(batch["sup"]):
        for pos, feats in sup_b:
            w_sup.append(w[b, pos])
            lang_sup.append(lang[b, pos])
            tgts.append(feats)
    if not tgts and not train:
        # Eval: no backward pass exists, so balancing the two forward gathers suffices.
        if align_on:
            za, zb = collect_align_pairs(w, w_p, batch["kf_pos"], batch["kf_pos_p"], device)
            _ = gather_variable(za if za is not None else w.new_zeros(0, dim), dim)
            _ = gather_variable(zb if zb is not None else w.new_zeros(0, dim), dim)
        return None
    if not tgts:
        # TRAIN with an empty local supervised set: an early return here would skip the finite
        # all_reduce, backward (whose gather-backward issues collectives), and _average_gradients
        # while the other ranks issue all of them -> NCCL hang (audit 2026-07-23). Take a zero-loss
        # path instead: stay on the graph, join every collective, contribute zero gradients.
        recon_loss = occ_loss = w.sum() * 0.0
        f1, assign_ms = 0.0, 0.0
    else:
        recon, occ = model.decode(torch.stack(w_sup).float(), torch.stack(lang_sup).float())
        recon_loss, occ_loss, f1, assign_ms = match_and_losses(
            recon, occ, tgts, feat_scale, device, return_assign_ms=True
        )
    if align_on:
        za, zb = collect_align_pairs(w, w_p, batch["kf_pos"], batch["kf_pos_p"], device)
        za_g = gather_variable(za if za is not None else w.new_zeros(0, dim), dim)
        zb_g = gather_variable(zb if zb is not None else w.new_zeros(0, dim), dim)
        if za_g.shape[0] >= 1:
            al, alc = sigreg(za_g, zb_g)
        else:
            al, alc = w.sum() * 0.0, {"inv": 0.0, "var": 0.0, "cov": 0.0}
        loss = recon_loss + occ_loss + cfg["lambda_align"] * al
    else:
        al, alc = 0.0, {"inv": 0.0, "var": 0.0, "cov": 0.0}
        loss = recon_loss + occ_loss

    if train:
        finite = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
        all_reduce_sum(finite)  # abort/skip must be consistent across ranks
        opt.zero_grad(set_to_none=True)
        if int(finite.item()) < get_world_size():
            return dict(
                recon=float(recon_loss),
                occ=float(occ_loss),
                align=float(al),
                occ_f1=f1,
                assign_ms=assign_ms,
                skipped=True,
                **alc,
            )
        loss.backward()
        _average_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return dict(recon=float(recon_loss), occ=float(occ_loss), align=float(al), occ_f1=f1, assign_ms=assign_ms, **alc)


@torch.no_grad()
def validate(model, sigreg, val_loader, cfg, feat_scale, device) -> dict:
    """Deterministic evaluated validation. Sums (loss*count, count) locally, reduces across ranks,
    reports the exact global means. Model is left in its prior train/eval mode by the caller."""
    totals = {"recon": 0.0, "occ": 0.0, "align": 0.0, "occ_f1": 0.0}
    n = 0
    for batch in val_loader:
        m = run_step_stage_s(model, sigreg, batch, cfg, feat_scale, device, train=False)
        if m is None:
            continue
        for k in totals:
            totals[k] += m[k]
        n += 1
    packed = torch.tensor([totals["recon"], totals["occ"], totals["align"], totals["occ_f1"], float(n)], device=device)
    all_reduce_sum(packed)
    count = max(float(packed[4].item()), 1.0)
    return {
        "val/recon": float(packed[0].item()) / count,
        "val/occ": float(packed[1].item()) / count,
        "val/align": float(packed[2].item()) / count,
        "val/occ_f1": float(packed[3].item()) / count,
        "val/batches": int(packed[4].item()),
    }


def _lr_at(step: int, cfg: dict) -> float:
    if step <= cfg["warmup_steps"]:
        return cfg["lr"] * step / max(1, cfg["warmup_steps"])
    prog = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["steps"] - cfg["warmup_steps"]))
    return cfg["lr"] * (cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))


def _feat_scale(train_rows, load_fn, target_mode) -> float:
    sample = []
    for r in train_rows[:8]:
        sample += [feats for _, feats in _gather_targets(load_fn(r), target_mode)]
    return float(torch.cat(sample).float().pow(2).mean().sqrt()) if sample else 1.0


def train_stage_s(
    *,
    rows: list[dict],
    load_fn,
    cfg: dict,
    backbone_dim: int = 2048,
    proprio_dim: int = 2048,
    lang_dim: int = 2048,
    out_dir: str | Path,
    device: str | None = None,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    val_every: int = 1000,
    val_batches: int = 0,
    log_every: int = 25,
    save_every: int = 0,
    extra_provenance: dict | None = None,
    model_overrides: dict | None = None,
) -> Path:
    """Train the Stage-S WSM encoder. Returns the final checkpoint path."""
    world = get_world_size()
    rank = get_rank()
    if device is None:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        else:
            device = "cpu"

    torch.manual_seed(0)
    train_rows, val_rows = stratified_demo_split(rows, cfg["val_frac"], cfg["seed_split"])
    align_on = cfg["lambda_align"] != 0

    # feat_scale on rank 0 (train rows only), broadcast so all ranks agree.
    feat_scale = _feat_scale(train_rows, load_fn, cfg["target_mode"]) if rank == 0 else 0.0
    if is_dist():
        fs = torch.tensor([feat_scale], device=device)
        dist.broadcast(fs, src=0)
        feat_scale = float(fs.item())

    model_kwargs = dict(
        backbone_dim=backbone_dim, proprio_dim=proprio_dim, lang_dim=lang_dim, input_norm=cfg["input_norm"]
    )
    model_kwargs.update(model_overrides or {})  # tests shrink n_layers/k_slots; prod keeps defaults
    model = WorkspaceModel(WSMConfig(**model_kwargs)).to(device)
    if is_dist():
        for p in model.parameters():  # start every rank from rank-0's identical init
            dist.broadcast(p.data, src=0)
    sigreg = SigRegLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    train_ds = WSMSampleDataset(
        train_rows,
        dropout=cfg["dropout"],
        mode=cfg["target_mode"],
        align=align_on,
        seed=0,
        load_fn=load_fn,
        lut_rows=train_rows,
        strict_split=True,
    )
    val_ds = WSMSampleDataset(
        val_rows,
        dropout=cfg["dropout"],
        mode=cfg["target_mode"],
        align=align_on,
        seed=0,
        load_fn=load_fn,
        lut_rows=val_rows,
        strict_split=True,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True, seed=0) if is_dist() else None
    gen = torch.Generator()
    gen.manual_seed(0)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=wsm_collate,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        generator=gen,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc_sha = run_config_sha256(cfg)
    if rank == 0:
        print(
            f"[wsm:pi-stage-s] world={world} train={len(train_rows)} val={len(val_rows)} "
            f"feat_scale={feat_scale:.2f} run_config_sha256={rc_sha[:16]} "
            f"lambda_align={cfg['lambda_align']} steps={cfg['steps']} batch={cfg['batch_size']}",
            flush=True,
        )

    def _val_loader():
        # Shard validation across ranks (rank-strided, disjoint). validate() all-reduces the
        # (sum, count) pair, so the reported means are exact global means — previously every rank
        # walked the FULL val set, paying world_size x the compute for identical numbers (audit
        # 2026-07-23). A rank with zero val batches still enters the all_reduce (n=0), so no hang.
        # num_workers=0: worker_init_fn resolves `.seed` through wrappers, but val is a few dozen
        # rank-sharded batches — main-process loading is cheap and removes the worker path entirely
        # (attempt-6 died here: Subset has no `.seed`).
        ds = Subset(val_ds, list(range(rank, len(val_ds), world))) if is_dist() else val_ds
        return DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=wsm_collate,
            num_workers=0,
        )

    import dataclasses

    full_cfg = dataclasses.asdict(model.cfg)  # every WSMConfig field, so the omega producer rebuilds exactly

    def _save(name: str, step: int, best_val: float | None):
        if rank != 0:
            return
        blob = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": full_cfg,
            "run_config": {k: cfg[k] for k in RUN_CONFIG_KEYS},
            "run_config_sha256": rc_sha,
            "feat_scale": feat_scale,
            "step": int(step),
            "best_val": best_val,
            "provenance": extra_provenance or {},
        }
        _atomic_torch_save(blob, out / name)

    best_val = None
    t0, step, done, epoch = time.time(), 0, False, 0
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, cfg)
            m = run_step_stage_s(model, sigreg, batch, cfg, feat_scale, device, True, opt)
            if m and not all(math.isfinite(m.get(k, 0.0)) for k in ("recon", "occ", "align")):
                # Save from WHATEVER rank detected the NaN (rank-tagged name): _save() is a rank-0
                # no-op, so a divergence on rank>0 previously aborted with no debug checkpoint.
                _atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "cfg": full_cfg,
                        "step": int(step),
                        "metrics": {k: float(v) for k, v in m.items()},
                        "rank": int(rank),
                    },
                    out / f"wsm_diverged_step{step}_rank{rank}.pt",
                )
                raise RuntimeError(f"[wsm] NON-FINITE loss at step {step} (rank {rank}): {m} — aborting.")
            if rank == 0 and m and step % log_every == 0:
                print(
                    f"[train] step {step} recon {m['recon']:.4f} occ {m['occ']:.4f} "
                    f"occ_f1 {m['occ_f1']:.3f} align {m['align']:.3f} "
                    f"(inv {m['inv']:.4f} var {m['var']:.3f} cov {m['cov']:.3f}) "
                    f"assign {m['assign_ms']:.1f}ms ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if val_every and step % val_every == 0:
                vloader = _val_loader()
                vm = validate(model, sigreg, vloader, cfg, feat_scale, device)
                if rank == 0:
                    print(f"[val] step {step} " + " ".join(f"{k} {v}" for k, v in vm.items()), flush=True)
                    score = vm["val/recon"] + vm["val/occ"]
                    if best_val is None or score < best_val:
                        best_val = score
                        _save("wsm_best.pt", step, best_val)
            if save_every and step % save_every == 0 and step < cfg["steps"]:
                _save(f"wsm_step{step}.pt", step, best_val)
            if step >= cfg["steps"]:
                done = True
                break
        epoch += 1
    _save(f"wsm_step{cfg['steps']}.pt", cfg["steps"], best_val)
    if rank == 0:
        print(f"[done:pi-stage-s] {cfg['steps']} steps -> {out}/wsm_step{cfg['steps']}.pt", flush=True)
    return out / f"wsm_step{cfg['steps']}.pt"
