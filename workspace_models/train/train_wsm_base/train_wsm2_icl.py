#!/usr/bin/env python3
"""train_wsm2_icl — WSMv2 encoder-phase trainer (doc 15 §2 objective + §3 gate, all instrumented).

Small-tensor, single-GPU, fully GPU-RESIDENT (perf mandate): the entire dataset (frozen w streams +
pooled demo tokens + task-mean lang) packs into ~2 GB of GPU tensors — zero dataloader workers, zero
per-step H2D. bf16 autocast + fused AdamW. A full P0a pilot runs in minutes on one 5090.

Objective (D10–D14):  L = Σ_k mask·(1 − cos(pred_k(z), w1[g+k])) + λ_sig·EP(z)/B + λ_sigd·EP(d_win)/N
                          + λ_phase·MSE(phase(hist), g/(F1−1))
  demo2 = context (bidirectional DemoEncoder over the FULL partner demo), demo1's OWN future frozen-w =
  target; branch dropout (p_demo2=.3 → null-window, p_hist=.1 → null-history) + lang dropout (p=.5).

Gate metrics (§3, logged every --gate-every steps on a FIXED held-out-pair eval set):
  G1 Δ_content per k = L(wrong-task window, same τ + pad) − L(real)   [the go/no-go signal]
  G2 Δ_demo    per k = L(mem-nulled) − L(real)                        [weak GO; NO-go if flat]
  both split lang-on/lang-off;  G3 z variance decomposition;  G4 persistence floor per k;
  G5 mean |ca_g| per fusion block (gate wake-up).

Pairing (D15–D16, label-free): per task, registry = LAST 5 episode indices (excluded from BOTH pair
roles, kept in the BC soup elsewhere); partners = 4 per demo, seed-0, same-task, non-registry, ≠self;
5% of (demo1,partner) pairs held out for the gate. The 6 POC composite-unseen tasks are EXCLUDED from
encoder training entirely (D16c). registry.json + sha written next to the ckpt.

  PYTHONPATH=. python workspace_models/train/train_wsm_base/train_wsm2_icl.py \
      --w-root ~/Research/TRI/wsm_data/wsm_policy_feats/groot_v2_50k --p-root ~/Research/TRI/wsm_data/wsm_pooled/v2_50k \
      --out ~/Research/TRI/wsm_data/wsm2_runs/v2_50k [--tasks A,B,C] [--steps 20000] [--device cuda:1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace_models.networks.demo_encoder import DemoEncoder  # noqa: E402
from workspace_models.networks.demo_fusion import HistoryDemoFusion  # noqa: E402
from workspace_models.networks.sigreg_loss import sigreg_epps_pulley  # noqa: E402

POC_CUNSEEN = (
    "ArrangeTea",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "PanTransfer",
    "WashFruitColander",
    "WeighIngredients",
)
KS = (1, 2, 4, 8)


# ---------------------------------------------------------------- data packing (GPU-resident)
class Pack:
    """All demos of all tasks packed into flat GPU tensors + index metadata.

    partner_mode (critique F2 lever): 'uniform' = any same-task demo; 'matched' = sample partners from
    the 8 nearest same-task demos by INITIAL-SCENE latent similarity (cos of the mean of the first 3 w
    frames) — a label-free proxy for same-scene-config pairing, which makes demo2's CONTENT (which
    cabinet, where the mug is) genuinely predictive of demo1's future instead of lang-redundant."""

    def __init__(
        self,
        w_root: Path,
        p_root: Path,
        tasks: list[str],
        device: str,
        registry_n: int = 5,
        partner_mode: str = "uniform",
    ):
        w_list, p_list, meta, init_fp = [], [], [], []  # meta rows: (task_id, ep, F, start)
        self.task_names = tasks
        task_lang = {}
        start = 0
        for ti, task in enumerate(tasks):
            demos = sorted((w_root / task).glob("demo_*"))
            eps = []
            for d in demos:
                pf = p_root / task / d.name / "p.npz"
                if not (d / "w.npz").exists() or not pf.exists():
                    continue
                wd = np.load(d / "w.npz")
                pd_ = np.load(pf)
                w = wd["w"]
                p = pd_["p"]
                if len(w) != len(p) or len(w) < 12:  # need room for k=8 targets + a window
                    continue
                w_list.append(torch.from_numpy(w.astype(np.float16)))
                init_fp.append(w[:3].astype(np.float32).mean(0))
                p_list.append(torch.from_numpy(p.astype(np.float16)))
                meta.append((ti, int(d.name.split("_")[1]), len(w), start))
                task_lang.setdefault(ti, []).append(pd_["lang_global"].astype(np.float32))
                start += len(w)
            if not eps:
                pass
        self.w = torch.cat(w_list).to(device)  # [Ntot,512] fp16
        self.p = torch.cat(p_list).to(device)  # [Ntot,512] fp16
        m = np.array([r[:3] for r in meta], dtype=np.int64)
        self.task_id = torch.from_numpy(m[:, 0]).to(device)
        self.ep = m[:, 1]
        self.F = torch.from_numpy(m[:, 2]).to(device)
        self.start = torch.from_numpy(np.array([r[3] for r in meta], dtype=np.int64)).to(device)
        # task-MEAN lang (S8 serve parity: the serve feeds the task-mean table)
        lang = np.stack([np.mean(task_lang[ti], 0) for ti in range(len(tasks))])
        self.lang = torch.from_numpy(lang).to(device)  # [T,2048] fp32
        self.n_demos, self.device = len(meta), device
        # registry: last `registry_n` episode indices per task -> excluded from pair roles
        self.registry: dict[str, list[int]] = {}
        excluded = np.zeros(self.n_demos, dtype=bool)
        tid_np = m[:, 0]
        for ti, task in enumerate(tasks):
            rows = np.nonzero(tid_np == ti)[0]
            reg_rows = rows[np.argsort(self.ep[rows])][-registry_n:]
            excluded[reg_rows] = True
            self.registry[task] = sorted(int(self.ep[r]) for r in reg_rows)
        self.trainable = np.nonzero(~excluded)[0]
        # partners: 4 per trainable demo, same task, trainable, != self (seed-0 deterministic)
        rng = np.random.default_rng(0)
        partners = np.full((self.n_demos, 4), -1, dtype=np.int64)
        fp = np.stack(init_fp)
        fp = fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-8)
        for ti in range(len(tasks)):
            rows = [r for r in np.nonzero(tid_np == ti)[0] if not excluded[r]]
            if partner_mode in ("matched", "mixed") and len(rows) > 9:
                sim = fp[rows] @ fp[rows].T
                np.fill_diagonal(sim, -2.0)
                top8 = np.argsort(-sim, axis=1)[:, :8]  # per row: 8 nearest same-task demos
                for i, r in enumerate(rows):
                    pick = rng.choice(8, size=4, replace=False)
                    partners[r] = np.asarray(rows)[top8[i][pick]]
                    if partner_mode == "mixed":  # slots 2,3 -> uniform (long-horizon diversity)
                        cands = [x for x in rows if x != r]
                        partners[r, 2:] = rng.choice(cands, size=2, replace=len(cands) < 2)
            else:
                for r in rows:
                    cands = [x for x in rows if x != r]
                    partners[r] = rng.choice(cands, size=4, replace=len(cands) < 4)
        self.partners = partners
        heldout = rng.random((self.n_demos, 4)) < 0.05  # 5% held-out PAIRS
        self.pair_train = (~heldout) & (partners >= 0)
        self.pair_held = heldout & (partners >= 0)
        # wrong-task partner per demo (for Δ_content): a random trainable demo of a DIFFERENT task
        wrong = np.zeros(self.n_demos, dtype=np.int64)
        for r in self.trainable:
            while True:
                c = int(rng.choice(self.trainable))
                if tid_np[c] != tid_np[r]:
                    wrong[r] = c
                    break
        self.wrong = wrong

    def gather_frames(self, demo_idx: torch.Tensor, frame_idx: torch.Tensor, which: str) -> torch.Tensor:
        """[B, L] grid indices (already clamped per-demo) -> [B, L, 512] from the packed tensor."""
        flat = self.start[demo_idx][:, None] + frame_idx
        src = self.w if which == "w" else self.p
        return src[flat]


# ---------------------------------------------------------------- batch construction (all-GPU)
def make_batch(
    pack: Pack,
    rows: np.ndarray,
    part_col: np.ndarray,
    dev: str,
    K: int,
    W: int,
    jitter: int,
    rng: np.random.Generator,
    wrong_task: bool = False,
    fixed_g: np.ndarray | None = None,
):
    d1 = torch.from_numpy(rows).to(dev)
    F1 = pack.F[d1]
    g = (
        torch.from_numpy(fixed_g).to(dev)
        if fixed_g is not None
        else (torch.rand(len(rows), device=dev) * (F1 - 1).float()).long()
    )
    ar = torch.arange(K, device=dev)
    hist_idx = (g[:, None] - (K - 1) + ar[None]).clamp_min(0)  # [B,K]
    hist = pack.gather_frames(d1, hist_idx, "w").float()
    ks = torch.tensor(KS, device=dev)
    tgt_idx = g[:, None] + ks[None]
    valid = tgt_idx < F1[:, None]
    tgt = pack.gather_frames(d1, tgt_idx.clamp(max=(F1 - 1)[:, None]), "w").float()  # [B,4,512]
    # partner (or wrong-task) demo2 + proportional tau + jitter
    p_rows = pack.wrong[rows] if wrong_task else pack.partners[rows, part_col]
    d2 = torch.from_numpy(p_rows).to(dev)
    M2 = pack.F[d2]
    tau = torch.round(g.float() / (F1 - 1).clamp_min(1).float() * (M2 - 1).float()).long()
    if jitter:
        j = torch.from_numpy(rng.integers(-jitter, jitter + 1, len(rows))).to(dev)
        tau = (tau + j).clamp(torch.zeros_like(M2), M2 - 1)
    off = torch.arange(-W, W + 1, device=dev)[None].expand(len(rows), -1)  # [B,41]
    raw = tau[:, None] + off
    wmask = (raw >= 0) & (raw < M2[:, None])
    widx = raw.clamp(torch.zeros_like(raw), (M2 - 1)[:, None])
    lang = pack.lang[pack.task_id[d1]]
    return dict(
        d1=d1, d2=d2, g=g, F1=F1, M2=M2, hist=hist, tgt=tgt, valid=valid, widx=widx, woff=off, wmask=wmask, lang=lang
    )


def encode_windows(
    enc: DemoEncoder, pack: Pack, b: dict, lang_keep=None, tok2_budget: int = 24_000_000
) -> torch.Tensor:
    """Encode each UNIQUE demo2 in the batch ONCE (full-length, bidirectional), then gather windows.

    Memory-bounded: encoding all uniques padded to the BATCH-max length OOMs (composites reach M~580 while
    the mean is ~74, and the float rel-bias mask materializes [U*heads, M, M] softmax buffers). So sort
    uniques by length and encode in chunks whose Σ heads·M_max² stays under tok2_budget — per-chunk padding
    is tight, total memory ≈ the unpadded sum. Same math, bounded peak."""
    d2 = b["d2"]
    uniq, inv = torch.unique(d2, return_inverse=True)
    Ms = pack.F[uniq]
    order = torch.argsort(Ms, descending=True)
    lk_u = None
    if lang_keep is not None:  # broadcast the per-sample draw to the unique-demo level (first hit wins)
        lk_u = torch.ones(len(uniq), dtype=torch.bool, device=d2.device)
        lk_u.scatter_(0, inv, lang_keep)
    d_out = torch.empty(len(uniq), int(Ms.max()), enc.dim, device=d2.device, dtype=torch.float32)
    i = 0
    order_list = order.tolist()
    ms_list = Ms.tolist()
    H = enc.n_heads
    while i < len(order_list):
        m_max = ms_list[order_list[i]]
        n = max(1, min(len(order_list) - i, tok2_budget // max(H * m_max * m_max, 1)))
        sel = order[i : i + n]
        ms_sel = Ms[sel]
        mx = int(ms_sel.max())
        ar = torch.arange(mx, device=d2.device)
        fidx = ar[None].expand(len(sel), -1).clamp(max=(ms_sel - 1)[:, None])
        toks = pack.gather_frames(uniq[sel], fidx, "p").float()
        pad = ar[None] >= ms_sel[:, None]
        d_tok = enc(
            toks, pack.lang[pack.task_id[uniq[sel]]], pad_mask=pad, lang_keep=None if lk_u is None else lk_u[sel]
        )
        d_out[sel, :mx] = d_tok.float()
        i += n
    return d_out[inv[:, None], b["widx"]]  # [B,41,512]


def jepa_loss(fus: HistoryDemoFusion, z: torch.Tensor, b: dict) -> tuple[torch.Tensor, dict]:
    preds = fus.predict_future(z)
    per_k = {}
    tot = 0.0
    for i, k in enumerate(KS):
        cos = Fn.cosine_similarity(preds[str(k)], b["tgt"][:, i], dim=-1)
        m = b["valid"][:, i]
        lk = ((1 - cos) * m).sum() / m.sum().clamp_min(1)
        per_k[k] = lk
        tot = tot + lk
    return tot / len(KS), per_k


# ---------------------------------------------------------------- gate evaluation (§3)
@torch.no_grad()
def gate_eval(enc, fus, pack, eval_rows, eval_part, eval_g, dev, K, W, step) -> dict:
    enc.eval()
    fus.eval()
    rng = np.random.default_rng(123)
    out = {}
    for lang_on in (True, False):
        lk = torch.full((len(eval_rows),), lang_on, dtype=torch.bool, device=dev)
        res = {}
        for mode, wrong in (("real", False), ("wrong", True)):
            b = make_batch(pack, eval_rows, eval_part, dev, K, W, 0, rng, wrong_task=wrong, fixed_g=eval_g)
            dwin = encode_windows(enc, pack, b, lang_keep=lk)
            z = fus(b["hist"], dwin, b["woff"], b["wmask"], b["lang"], lang_keep=lk)["z"]
            _, per_k = jepa_loss(fus, z, b)
            res[mode] = {k: float(v) for k, v in per_k.items()}
        b = make_batch(pack, eval_rows, eval_part, dev, K, W, 0, rng, fixed_g=eval_g)
        dwin = encode_windows(enc, pack, b, lang_keep=lk)
        z0 = fus(
            b["hist"],
            dwin,
            b["woff"],
            b["wmask"],
            b["lang"],
            lang_keep=lk,
            drop_demo=torch.ones(len(eval_rows), dtype=torch.bool, device=dev),
        )["z"]
        _, per_k0 = jepa_loss(fus, z0, b)
        tag = "lang" if lang_on else "nolang"
        for k in KS:
            out[f"dC_k{k}_{tag}"] = res["wrong"][k] - res["real"][k]  # G1 Δ_content
            out[f"dD_k{k}_{tag}"] = float(per_k0[k]) - res["real"][k]  # G2 Δ_demo
    # G3 variance decomposition of z on the real/lang batch
    b = make_batch(pack, eval_rows, eval_part, dev, K, W, 0, np.random.default_rng(7), fixed_g=eval_g)
    dwin = encode_windows(enc, pack, b)
    z = fus(b["hist"], dwin, b["woff"], b["wmask"], b["lang"])["z"].float()
    tid = pack.task_id[b["d1"]]
    total = z.var(0).mean()
    task_means = torch.stack([z[tid == t].mean(0) for t in tid.unique()])
    out["var_between_task_frac"] = float(task_means.var(0).mean() / total.clamp_min(1e-8))
    out["z_eff_rank"] = float(torch.linalg.matrix_rank(z - z.mean(0), tol=1e-3))
    enc.train()
    fus.train()
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w-root", required=True)
    ap.add_argument("--p-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="", help="subset (pilot); default = all shared tasks minus POC-cunseen")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k-hist", type=int, default=16)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--jitter", type=int, default=5)
    ap.add_argument("--p-demo2", type=float, default=0.3)
    ap.add_argument("--p-hist", type=float, default=0.1)
    ap.add_argument("--p-lang", type=float, default=0.5)
    ap.add_argument("--lam-sig", type=float, default=0.05)
    ap.add_argument("--lam-sigd", type=float, default=0.01)
    ap.add_argument("--lam-phase", type=float, default=0.1)
    ap.add_argument("--gate-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--warmup-demo-on", type=int, default=0, help="force p_demo2=0 for the first N steps (F7)")
    ap.add_argument(
        "--partner-mode",
        default="uniform",
        choices=["uniform", "matched", "mixed"],
        help="matched = init-scene-similar partners (F2 lever; label-free config proxy)",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    w_root, p_root, out = Path(args.w_root).expanduser(), Path(args.p_root).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    shared = sorted(
        set(d.name for d in w_root.iterdir() if d.is_dir()) & set(d.name for d in p_root.iterdir() if d.is_dir())
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or [t for t in shared if t not in POC_CUNSEEN]
    for t in tasks:
        assert t in shared, f"task {t} missing from w/p roots"
    print(f"[wsm2] {len(tasks)} tasks (excluded POC-cunseen: {sorted(set(shared) - set(tasks))[:8]}...)", flush=True)

    pack = Pack(w_root, p_root, tasks, dev, partner_mode=args.partner_mode)
    reg_json = json.dumps(pack.registry, sort_keys=True)
    reg_hash = hashlib.sha256(reg_json.encode()).hexdigest()[:16]
    (out / "registry.json").write_text(reg_json)
    print(
        f"[wsm2] {pack.n_demos} demos packed ({pack.w.shape[0]} grid frames, "
        f"{(pack.w.numel() + pack.p.numel()) * 2 / 1e9:.2f} GB on {dev}); registry sha={reg_hash}",
        flush=True,
    )

    # fixed held-out-pair eval set for the gate
    hr, hc = np.nonzero(pack.pair_held)
    keep = np.isin(hr, pack.trainable)
    hr, hc = hr[keep], hc[keep]
    ne = min(2048, len(hr))
    sel = np.random.default_rng(1).choice(len(hr), ne, replace=False)
    eval_rows, eval_part = hr[sel], hc[sel]
    eval_g = np.random.default_rng(2).integers(0, np.maximum(pack.F.cpu().numpy()[eval_rows] - 9, 1))
    print(f"[wsm2] gate eval set: {ne} held-out pairs", flush=True)
    # G4 persistence floor on the eval set (the bar the JEPA must beat)
    with torch.no_grad():
        b = make_batch(
            pack, eval_rows, eval_part, dev, args.k_hist, args.window, 0, np.random.default_rng(3), fixed_g=eval_g
        )
        w_now = b["hist"][:, -1]
        for i, k in enumerate(KS):
            cos = Fn.cosine_similarity(w_now, b["tgt"][:, i], dim=-1)
            m = b["valid"][:, i]
            print(f"[wsm2] G4 persistence floor k={k}: {(1 - cos)[m].mean():.4f}", flush=True)

    enc = DemoEncoder().to(dev)
    fus = HistoryDemoFusion(k_hist=args.k_hist, window=args.window).to(dev)
    params = list(enc.parameters()) + list(fus.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))
    )
    # ca_g magnitude probe (G5): hook each fusion block's ada output, grab the ca gate chunk
    ca_mag: dict[int, float] = {}
    for bi, blk in enumerate(fus.blocks):

        def _hook(mod, inp, out_, bi=bi):
            ca_mag[bi] = float(out_.chunk(9, dim=-1)[5].abs().mean())

        blk.ada.register_forward_hook(_hook)

    rng = np.random.default_rng(42)
    tr = pack.trainable
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rows = rng.choice(tr, args.batch)
        cols = rng.integers(0, 4, args.batch)
        ok = pack.pair_train[rows, cols]
        cols = np.where(ok, cols, (cols + 1) % 4)  # dodge held-out pairs
        b = make_batch(pack, rows, cols, dev, args.k_hist, args.window, args.jitter, rng)
        lang_keep = torch.from_numpy(rng.random(args.batch) >= args.p_lang).to(dev)
        p_demo2 = 0.0 if step <= args.warmup_demo_on else args.p_demo2
        drop_demo = torch.from_numpy(rng.random(args.batch) < p_demo2).to(dev)
        drop_hist = torch.from_numpy(rng.random(args.batch) < args.p_hist).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dwin = encode_windows(enc, pack, b, lang_keep=lang_keep)
            o = fus(
                b["hist"],
                dwin,
                b["woff"],
                b["wmask"],
                b["lang"],
                drop_demo=drop_demo,
                drop_hist=drop_hist,
                lang_keep=lang_keep,
            )
            z = o["z"].float()
            l_jepa, per_k = jepa_loss(fus, z, b)
            l_sig = sigreg_epps_pulley(z, step) / args.batch  # /B: batch-invariant (F6)
            vt = dwin.float()[b["wmask"]]
            l_sigd = sigreg_epps_pulley(vt[:: max(1, len(vt) // 2048)], step) / min(len(vt), 2048)
            l_phase = Fn.mse_loss(fus.predict_phase(b["hist"]), (b["g"].float() / (b["F1"] - 1).clamp_min(1).float()))
            loss = l_jepa + args.lam_sig * l_sig + args.lam_sigd * l_sigd + args.lam_phase * l_phase
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            ips = step * args.batch / (time.time() - t0)
            pk = " ".join(f"k{k}={float(v):.3f}" for k, v in per_k.items())
            cg = " ".join(f"b{i}={v:.4f}" for i, v in sorted(ca_mag.items()))
            print(
                f"[wsm2] {step} loss={float(loss):.4f} jepa[{pk}] sig={float(l_sig):.4f} "
                f"phase={float(l_phase):.4f} |ca_g|[{cg}] {ips:.0f} samp/s",
                flush=True,
            )
        if step % args.gate_every == 0 or step == args.steps:
            g = gate_eval(enc, fus, pack, eval_rows, eval_part, eval_g, dev, args.k_hist, args.window, step)
            print(f"[wsm2] GATE @{step}: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(g.items())), flush=True)
            torch.save(
                {
                    "demo_encoder": enc.state_dict(),
                    "fusion": fus.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "registry_sha": reg_hash,
                    "w_root": str(w_root),
                    "p_root": str(p_root),
                    "gate": g,
                },
                out / f"wsm2_step{step}.pt",
            )
    print(f"[wsm2] DONE {args.steps} steps in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
