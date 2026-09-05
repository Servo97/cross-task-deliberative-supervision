"""Shared WSM-base trainer core (backbone-agnostic): Hungarian salient-patch recon + occupancy +
cross-demo workspace-latent alignment. Trains ONLY the WorkspaceModel on cached FROZEN features.

The groot (N1.7) and pi0.5 trainers are thin wrappers that call `train()` with their own per-demo
`load_fn` (data.load_demo / data.load_demo_pi) and `proprio_dim` (groot state_emb 1536 / pi
lang_per_frame 2048). All the loop/loss/IO logic — async DataLoader, fp16->fp32-on-GPU transfer,
Hungarian match, NaN guards, --resume, --profile — lives here so both backbones stay in lock-step.

See internal_planning_and_todos/07_wsm_preprocessing_and_revised_plan.md + 09 (dataloader plan).
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from workspace_models.networks.sigreg_loss import SigRegLoss
from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig
from workspace_models.train.train_wsm_base.data import (
    WSMSampleDataset,
    _gather_targets,
    worker_init_fn,
    wsm_collate,
)
from wsm_settings import WSM_DATA_ROOT


def _checkpoint_payload(model, opt, args, feat_scale: float, step: int, loader_generator=None) -> dict:
    """One-file, backward-compatible checkpoint with optimizer and RNG state."""
    blob = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "cfg": vars(args),
        "feat_scale": feat_scale,
        "step": int(step),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        blob["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        blob["loader_generator_state"] = loader_generator.get_state()
    return blob


def _atomic_torch_save(blob: dict, path: Path) -> None:
    """Publish the canonical filename only after torch.save completes."""
    tmp = path.with_name(f".{path.name}.incomplete")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def _match_and_losses_reference(recon, occ, target_feats_list, feat_scale, device):
    """Reference implementation with a per-supervised-point GPU->CPU sync (kept for parity tests).

    Superseded on the hot path by `match_and_losses`, which produces IDENTICAL assignments/losses
    with a single batched transfer. Do not use in training; the batched version is the contract."""
    n, k, _ = recon.shape
    occ_target = torch.zeros(n, k, device=device)
    recon_losses, correct, fp, fn = [], 0, 0, 0
    for j in range(n):
        tgt = target_feats_list[j].to(device, dtype=torch.float32) / feat_scale
        pred = recon[j].float() / feat_scale
        with torch.no_grad():
            cost = torch.cdist(pred, tgt).pow(2) / pred.shape[-1] - F.logsigmoid(occ[j].float()).unsqueeze(1) * 0.1
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
        rows_t, cols_t = torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)
        occ_target[j, rows_t] = 1.0
        recon_losses.append(F.mse_loss(pred[rows_t], tgt[cols_t]))
        with torch.no_grad():
            on = occ[j].float() > 0
            correct += int(on[rows_t].sum())
            fp += int(on.sum()) - int(on[rows_t].sum())
            fn += len(rows) - int(on[rows_t].sum())
    recon_loss = torch.stack(recon_losses).mean() if recon_losses else recon.sum() * 0.0
    occ_loss = F.binary_cross_entropy_with_logits(occ.float(), occ_target)
    f1 = 2 * correct / max(2 * correct + fp + fn, 1)
    return recon_loss, occ_loss, f1


def match_and_losses(recon, occ, target_feats_list, feat_scale, device, *, return_assign_ms=False):
    """recon [N,k,2048], occ [N,k]; target_feats_list[j] = [m_j,2048] frozen targets for sup point j.

    Semantics are byte-identical to `_match_and_losses_reference` (scipy Hungarian is deterministic),
    but every cost matrix is built on-GPU and moved to the host in ONE transfer instead of N — the
    per-point ``cost.cpu().numpy()`` was forcing a GPU/CPU sync per supervised point. Assignments run
    in a small thread pool over the host-side views. A NaN/Inf cost (a bf16 spike) is sanitized so
    scipy never raises; the finite-loss guard in run_step then skips that batch's gradient."""
    n, k, _ = recon.shape
    device = recon.device
    preds, tgts, costs = [], [], []
    for j in range(n):
        tgt = target_feats_list[j].to(device, dtype=torch.float32) / feat_scale  # [m,2048]
        pred = recon[j].float() / feat_scale  # [k,2048] (carries grad)
        with torch.no_grad():
            cost = torch.cdist(pred, tgt).pow(2) / pred.shape[-1] - F.logsigmoid(occ[j].float()).unsqueeze(1) * 0.1
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)  # [k,m]
        preds.append(pred)
        tgts.append(tgt)
        costs.append(cost)

    t0 = time.perf_counter()
    # ONE host transfer for all cost matrices (variable [k,m_j]); slice back on the host side.
    with torch.no_grad():
        flat = torch.cat([c.reshape(-1) for c in costs]).cpu().numpy() if costs else None
    offsets, off = [], 0
    for c in costs:
        offsets.append((off, c.shape[0], c.shape[1]))
        off += c.numel()

    def _assign(spec):
        start, kk, mm = spec
        mat = flat[start : start + kk * mm].reshape(kk, mm)
        return linear_sum_assignment(mat)

    if costs:
        with ThreadPoolExecutor(max_workers=min(8, len(costs))) as pool:
            assignments = list(pool.map(_assign, offsets))
    else:
        assignments = []
    assign_ms = (time.perf_counter() - t0) * 1e3

    occ_target = torch.zeros(n, k, device=device)
    recon_losses, correct, fp, fn = [], 0, 0, 0
    for j, (rows, cols) in enumerate(assignments):
        rows_t = torch.as_tensor(rows, device=device)
        cols_t = torch.as_tensor(cols, device=device)
        occ_target[j, rows_t] = 1.0
        recon_losses.append(F.mse_loss(preds[j][rows_t], tgts[j][cols_t]))
        with torch.no_grad():
            on = occ[j].float() > 0
            correct += int(on[rows_t].sum())
            fp += int(on.sum()) - int(on[rows_t].sum())
            fn += len(rows) - int(on[rows_t].sum())
    recon_loss = torch.stack(recon_losses).mean() if recon_losses else recon.sum() * 0.0
    occ_loss = F.binary_cross_entropy_with_logits(occ.float(), occ_target)
    f1 = 2 * correct / max(2 * correct + fp + fn, 1)
    if return_assign_ms:
        return recon_loss, occ_loss, f1, assign_ms
    return recon_loss, occ_loss, f1


def collect_align_pairs(w, w_p, kf_pos_list, kf_pos_p_list, device):
    """Matched cross-demo keyframe latents (same task, different demo), subgoal-index aligned up to
    the shorter decomposition -> (z_a [M,D], z_b [M,D]) for the SigReg/VICReg alignment."""
    a_list, c_list = [], []
    for b, (kd, kp) in enumerate(zip(kf_pos_list, kf_pos_p_list)):
        m = min(len(kd), len(kp))
        if m == 0:
            continue
        a_list.append(w[b, kd[:m].to(device)])
        c_list.append(w_p[b, kp[:m].to(device)])
    if not a_list:
        return None, None
    return torch.cat(a_list).float(), torch.cat(c_list).float()


def _to_gpu_fp32(batch, key, device):
    """fp16 H2D transfer (overlapped via pinned memory) then cast to fp32 on the GPU."""
    return batch[key].to(device, non_blocking=True).float()


def run_step(model, sigreg, batch, args, feat_scale, device, train, opt=None):
    align_on = args.lambda_align != 0  # recon/occ-only (Step 2a) -> skip partner encode (~halves step)
    patch = _to_gpu_fp32(batch, "patch", device)
    state = _to_gpu_fp32(batch, "state", device)
    lang = _to_gpu_fp32(batch, "lang", device)
    # FP32 forward (NO bf16 autocast): the WSM is tiny (38.7M) and the run is data-bound, so fp32 costs
    # ~nothing in wall-clock while eliminating the bf16-precision NaN class. Loader ships fp16 -> fp32 here.
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
    if not tgts:
        return None
    recon, occ = model.decode(torch.stack(w_sup).float(), torch.stack(lang_sup).float())
    recon_loss, occ_loss, f1 = match_and_losses(recon, occ, tgts, feat_scale, device)
    if align_on:
        za, zb = collect_align_pairs(w, w_p, batch["kf_pos"], batch["kf_pos_p"], device)
        al, alc = sigreg(za, zb) if za is not None else (w.sum() * 0.0, {"inv": 0.0, "var": 0.0, "cov": 0.0})
        loss = recon_loss + occ_loss + args.lambda_align * al
    else:
        al, alc = 0.0, {"inv": 0.0, "var": 0.0, "cov": 0.0}
        loss = recon_loss + occ_loss
    if train:
        if not torch.isfinite(loss):  # a NaN/Inf slipped through -> skip this batch's gradient (don't corrupt weights)
            opt.zero_grad(set_to_none=True)
            print(
                f"[warn] non-finite loss (recon {float(recon_loss):.3f} occ {float(occ_loss):.3f}) — step skipped",
                flush=True,
            )
            return dict(recon=float(recon_loss), occ=float(occ_loss), align=float(al), occ_f1=f1, skipped=True, **alc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return dict(recon=float(recon_loss), occ=float(occ_loss), align=float(al), occ_f1=f1, **alc)


def _profile(loader, model, sigreg, args, feat_scale, device, opt):
    """Good-hygiene profiler: time data-wait (next batch) vs GPU compute over N steps, then exit.
    data-wait ~0% => the loader keeps the GPU fed (compute-bound, the goal)."""
    it = iter(loader)

    def _next():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    for _ in range(3):  # warmup: spin up workers + cudnn autotune
        run_step(model, sigreg, _next(), args, feat_scale, device, True, opt)
    torch.cuda.synchronize()
    data_t = comp_t = 0.0
    t = time.perf_counter()
    for _ in range(args.profile_steps):
        batch = _next()
        t1 = time.perf_counter()
        data_t += t1 - t
        run_step(model, sigreg, batch, args, feat_scale, device, True, opt)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        comp_t += t2 - t1
        t = t2
    n, tot = args.profile_steps, data_t + comp_t
    print(
        f"[profile] {n} steps batch={args.batch_size} workers={args.num_workers} "
        f"prefetch={args.prefetch_factor}: {tot / n * 1e3:.1f} ms/step | "
        f"data-wait {data_t / n * 1e3:.1f} ms ({100 * data_t / tot:.0f}%) | "
        f"compute {comp_t / n * 1e3:.1f} ms ({100 * comp_t / tot:.0f}%) | "
        f"{n * args.batch_size / tot:.1f} demos/s",
        flush=True,
    )


def build_parser(default_out: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000, help="LR linear warmup steps")
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="cosine-decay floor as a fraction of peak LR")
    ap.add_argument(
        "--input-norm",
        action="store_true",
        help="LayerNorm raw encoder inputs (stability; set for GR00T's large-RMS patch features)",
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="per-frame subgoal->global lang dropout "
        "(1.0 = global only; set 1.0 to match the global-language precompute+eval deploy)",
    )
    ap.add_argument("--target-mode", default="next", choices=["next", "cumul"])
    ap.add_argument("--lambda-align", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (0 = end only)")
    ap.add_argument(
        "--resume-from", default="", help="checkpoint .pt to resume (or 'auto' = latest wsm_step*.pt in --out)"
    )
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (per-demo load+collate)")
    ap.add_argument("--prefetch-factor", type=int, default=4, help="batches prefetched per worker")
    ap.add_argument(
        "--profile-steps", type=int, default=0, help=">0: time data-wait vs compute over N steps, then exit"
    )
    ap.add_argument("--device", default="cuda:0")
    return ap


def train(
    load_fn,
    proprio_dim: int,
    backbone: str,
    default_out: str = str(WSM_DATA_ROOT / "wsm_runs" / "wsm_base_v1"),
) -> None:
    """Backbone-agnostic WSM-base training entry. `load_fn` (per-demo loader) + `proprio_dim` are the
    ONLY backbone-specific knobs; everything else is shared."""
    args = build_parser(default_out).parse_args()
    import pandas as pd
    from torch.utils.data import DataLoader

    device = args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    df = pd.read_parquet(args.manifest)
    rows = df.to_dict("records")
    n_val = max(1, int(len(rows) * args.val_frac))
    perm = rng.permutation(len(rows))
    train_rows = [rows[i] for i in perm[n_val:]]

    sample = []
    for r in train_rows[:8]:
        sample += [feats for _, feats in _gather_targets(load_fn(r), args.target_mode)]
    feat_scale = float(torch.cat(sample).float().pow(2).mean().sqrt()) if sample else 1.0

    model = WorkspaceModel(WSMConfig(proprio_dim=proprio_dim, input_norm=args.input_norm)).to(device)
    sigreg = SigRegLoss().to(device)  # VICReg-style anti-collapse alignment (the core contribution)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Resume model + optimizer moments + RNG. Legacy model-only checkpoints remain loadable, but an
    # Adam reset is reported explicitly rather than silently described as harmless.
    start_step = 0
    resume_blob = None
    if args.resume_from:
        rp = args.resume_from
        if rp == "auto":
            cks = sorted(out.glob("wsm_step*.pt"), key=lambda p: int(p.stem.replace("wsm_step", "")))
            rp = str(cks[-1]) if cks else ""
        if rp:
            resume_blob = torch.load(rp, map_location=device)
            model.load_state_dict(resume_blob["model"])
            start_step = int(resume_blob.get("step", 0))
            if "feat_scale" in resume_blob:
                feat_scale = float(resume_blob["feat_scale"])
            if "optimizer" in resume_blob:
                opt.load_state_dict(resume_blob["optimizer"])
            else:
                print("[wsm] WARNING: legacy checkpoint has no optimizer state; Adam moments restart", flush=True)
            if "torch_rng_state" in resume_blob:
                torch.set_rng_state(resume_blob["torch_rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state_all" in resume_blob:
                torch.cuda.set_rng_state_all(resume_blob["cuda_rng_state_all"])
            restored = "restored" if "optimizer" in resume_blob else "RESET"
            print(
                f"[wsm] RESUMED from {rp} @ step {start_step} (feat_scale {feat_scale:.2f}, optimizer={restored})",
                flush=True,
            )
        else:
            print("[wsm] --resume-from auto: no checkpoint found in --out, starting fresh", flush=True)

    # Async pipeline: per-demo load+collate in `num_workers` processes, prefetched + pinned, so the
    # GPU never waits on disk/CPU. shuffle+drop_last gives clean random epochs over the train demos.
    align_on = args.lambda_align != 0
    ds = WSMSampleDataset(
        train_rows, dropout=args.dropout, mode=args.target_mode, align=align_on, seed=0, load_fn=load_fn, lut_rows=rows
    )
    gen = torch.Generator()
    gen.manual_seed(0)
    if resume_blob is not None and "loader_generator_state" in resume_blob:
        gen.set_state(resume_blob["loader_generator_state"].cpu())
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=wsm_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=gen,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    print(
        f"[wsm:{backbone}] {len(train_rows)} train demos | feat_scale {feat_scale:.2f} | "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | proprio_dim={proprio_dim} "
        f"target={args.target_mode} lambda_align={args.lambda_align} | batch={args.batch_size} "
        f"workers={args.num_workers} prefetch={args.prefetch_factor} | start_step={start_step} -> {args.steps}",
        flush=True,
    )

    if args.profile_steps:
        _profile(loader, model, sigreg, args, feat_scale, device, opt)
        return

    import math

    def lr_at(s: int) -> float:
        """Linear warmup -> cosine decay to min_lr_frac*lr by --steps. Constant-3e-4 was a driver of the
        late-training divergence; the decay tail prevents drift, warmup the early spikes."""
        if s <= args.warmup_steps:
            return args.lr * s / max(1, args.warmup_steps)
        prog = min(1.0, (s - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

    t0, step, done = time.time(), start_step, False
    while not done:
        for batch in loader:  # one pass = one epoch; re-shuffles each pass
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            m = run_step(model, sigreg, batch, args, feat_scale, device, True, opt)
            # ABORT-ON-DIVERGENCE: catch a NaN/Inf loss at the SOURCE (this is where the groot encoder
            # silently went NaN before; the finetune then logged 0.0 and masked it). Save a debug ckpt + fail.
            if m and not all(math.isfinite(m.get(k, 0.0)) for k in ("recon", "occ", "align")):
                _atomic_torch_save(
                    _checkpoint_payload(model, opt, args, feat_scale, step, gen), out / f"wsm_diverged_step{step}.pt"
                )
                raise RuntimeError(
                    f"[wsm] NON-FINITE loss at step {step}: {m} — encoder diverged; "
                    f"saved debug ckpt, aborting (check input-norm / LR / inputs)."
                )
            if m and step % args.log_every == 0:
                print(
                    f"[train] step {step} recon {m['recon']:.4f} occ {m['occ']:.4f} occ_f1 {m['occ_f1']:.3f} "
                    f"| align {m['align']:.3f} (inv {m['inv']:.4f} var {m['var']:.3f} cov {m['cov']:.3f}) "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if args.save_every and step % args.save_every == 0 and step < args.steps:
                _atomic_torch_save(
                    _checkpoint_payload(model, opt, args, feat_scale, step, gen), out / f"wsm_step{step}.pt"
                )
                print(f"[ckpt] step {step} -> {out}/wsm_step{step}.pt", flush=True)
            if step >= args.steps:
                done = True
                break
    _atomic_torch_save(
        _checkpoint_payload(model, opt, args, feat_scale, args.steps, gen), out / f"wsm_step{args.steps}.pt"
    )
    print(f"[done:{backbone}] {args.steps} steps -> {out}/wsm_step{args.steps}.pt", flush=True)
