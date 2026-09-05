#!/usr/bin/env python3
"""H14 A13(e) §19.5 — SEQUENCE read-out of progress state over the ω history.

Pre-registered by the coordinator 2026-08-29, after §19's pooled-linear gate failed 8/8 and BEFORE
any sequence number existed. It is the discriminator §19.5 named: §19 can only conclude that a
NEAREST-CENTROID or a RIDGE read-out over ω (frame-level or causally pooled) does not beat a clock.
It cannot separate

    H_absent      ω does not contain progress state at all
    H_nonlinear   ω contains it, but a pooled linear probe cannot extract it

A causal sequence model can. Same labels (`progress_annotations/2aca11911650aebf`), same primary
window, same 5-fold BY EPISODE split, same time-confound controls — only the read-out changes.

PROBE (simplest thing that can carry state; stated because the brief allowed a GDN-style block):
a **1-layer GRU, hidden 64**, on the per-frame input, then a linear head to the class logits. The
repo's GDN blocks live in `robomme_integration/` and are policy-side modules wired to an action
head; none is usable off the shelf as a bare sequence encoder, so the GRU is used — it is strictly
the simpler choice and a GRU that cannot beat a clock is evidence about the representation, not
about the block. A 512 -> 64 input projection precedes the recurrence for every source alike, so the
recurrent capacity is identical across sources regardless of input width.

The probe is CAUSAL: it is fed the FULL episode from frame 0 and scored only on the primary-window
frames, so it sees exactly the history a deployed read-out would have.

Capacity-matched sources (identical probe, identical budget, identical folds):

    E1b        full RoboCasa Stage-E cell / rmb 1,000-step smoke, as in §19
    ctrl-0b    lambda_del = 0
    untrained  the same architecture at init
    raw_tap    wsm_pooled/{pi_100k, rmb_pi_100k} — if THIS passes, a GDN read over raw tokens
               suffices and the encoder adds nothing
    time_only  the SAME probe fed only [normalized_time, frame_index/1000] — the confound, given
               a sequence model of its own so the comparison is probe-vs-probe
    shuffled   E1b features with the label sequences PERMUTED ACROSS EPISODES (resampled to each
               episode's own length, so the label marginal and the monotone within-episode shape
               survive and only the feature<->label link is destroyed) — the chance floor

Training budget, fixed in advance and identical for every (family, source, fold): Adam lr 1e-3,
full-batch over the training episodes, at most 200 epochs, EARLY STOPPING on the accuracy of a
validation split drawn from the TRAINING episodes only (20%, seeded), patience 20, best-epoch
weights restored. The test fold is never used for any decision.

Reported per family: accuracy with Wilson 95% on the frame count (anticonservative — frames of an
episode are correlated; the episode count is beside it), delta vs the time-only probe, and the same
within global normalized-time quintiles.

SIGNATURE: E1b+probe beats time_only+probe AND beats raw_tap+probe, by Wilson lower bound.

    python scripts/deliberation/progress_sequence_readout.py \
        --annotations ~/Research/TRI/wsm_data/deliberation/progress_annotations/<id> \
        --out ~/Research/TRI/wsm_data/deliberation/progress_sequence_readout.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DELIB = Path("~/Research/TRI/wsm_data/deliberation").expanduser()
POOLED = Path("~/Research/TRI/wsm_data/wsm_pooled").expanduser()

SOURCES = {
    "robocasa": {
        "E1b": (DELIB / "stage_e_runs/omega/E1b", "omega"),
        "ctrl-0b": (DELIB / "stage_e_runs/omega/ctrl-0b", "omega"),
        "untrained": (DELIB / "stage_e_runs/omega/untrained", "omega"),
        "raw_tap": (POOLED / "pi_100k", "tap"),
    },
    "remembench": {
        "E1b": (DELIB / "stage_e_runs_rmb_smoke/omega/E1b_smoke_rmb", "omega"),
        "ctrl-0b": (DELIB / "stage_e_runs_rmb_smoke/omega/ctrl0b_smoke_rmb", "omega"),
        "untrained": (DELIB / "stage_e_runs_rmb_smoke/omega/untrained_rmb", "omega"),
        "raw_tap": (POOLED / "rmb_pi_100k", "tap"),
    },
}
N_TIME_BINS = 5
HIDDEN = 64
PROJ = 64
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
VAL_FRAC = 0.2


def wilson(hits: int, total: int, z: float = 1.96) -> list:
    if total == 0:
        return [None, None]
    p = hits / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


class SeqProbe(nn.Module):
    """Linear 512->64, then a 1-layer GRU(64), then a linear head. Causal by construction."""

    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_in, PROJ)
        self.gru = nn.GRU(PROJ, HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, n_classes)

    def forward(self, x):
        h, _ = self.gru(torch.tanh(self.proj(x)))
        return self.head(h)


def load_features(root: Path, kind: str, domain: str, task: str, episode: int):
    if kind == "tap":
        path, key = root / task / f"demo_{episode:06d}" / "p.npz", "p"
    else:
        path, key = root / domain / task / f"demo_{episode:06d}" / "w.npz", "w"
    if not path.exists():
        return None
    blob = np.load(path)
    return np.asarray(blob[key], np.float32), np.asarray(blob["frame_indices"], np.int64)


def pad_stack(seqs: list, dim: int):
    lens = [len(s) for s in seqs]
    out = np.zeros((len(seqs), max(lens), dim), np.float32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, np.array(lens)


def run_fold(
    train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m, n_classes: int, seed: int, device: str
):
    """One fold. Returns per-frame predictions for the TEST episodes (list of arrays)."""
    torch.manual_seed(seed)
    d_in = train_x.shape[-1]
    model = SeqProbe(d_in, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.CrossEntropyLoss(reduction="none")

    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).long().to(device)
    tm = torch.from_numpy(train_m).float().to(device)
    vx = torch.from_numpy(val_x).to(device)
    vy = torch.from_numpy(val_y).long().to(device)
    vm = torch.from_numpy(val_m).bool().to(device)

    best_acc, best_state, since = -1.0, None, 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(tx)
        loss = (lossfn(logits.reshape(-1, n_classes), ty.reshape(-1)) * tm.reshape(-1)).sum() / tm.sum().clamp(min=1)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx).argmax(-1)
            acc = float(((pred == vy) & vm).sum()) / max(float(vm.sum()), 1.0)
        if acc > best_acc + 1e-5:
            best_acc, since = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(test_x).to(device)).argmax(-1).cpu().numpy()
    return pred, best_acc


def evaluate_source(episodes: list, n_classes: int, folds: int, seed: int, device: str) -> dict:
    """episodes = [(X [T,D], y [T], primary_mask [T], tbin [T]), ...]."""
    n = len(episodes)
    if n < folds * 2:
        return {"note": "too few episodes"}
    d = episodes[0][0].shape[-1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    hits = total = 0
    bin_hits, bin_total = Counter(), Counter()
    val_accs = []
    for fold in range(folds):
        test_ids = order[fold::folds]
        train_ids = np.array([i for i in order if i not in set(test_ids.tolist())])
        n_val = max(2, int(round(VAL_FRAC * len(train_ids))))
        val_ids, fit_ids = train_ids[:n_val], train_ids[n_val:]

        fit_frames = np.concatenate([episodes[i][0] for i in fit_ids], 0)
        mu = fit_frames.mean(0, keepdims=True)
        sd = fit_frames.std(0, keepdims=True) + 1e-6

        def pack(ids):
            xs = [(episodes[i][0] - mu) / sd for i in ids]
            x, _ = pad_stack(xs, d)
            lens = [len(episodes[i][1]) for i in ids]
            T = x.shape[1]
            y = np.zeros((len(ids), T), np.int64)
            m = np.zeros((len(ids), T), bool)
            b = np.zeros((len(ids), T), np.int64)
            for j, i in enumerate(ids):
                L = lens[j]
                y[j, :L] = episodes[i][1]
                m[j, :L] = episodes[i][2]
                b[j, :L] = episodes[i][3]
            return x, y, m, b

        fx, fy, fm, _ = pack(fit_ids)
        vx, vy, vm, _ = pack(val_ids)
        sx, sy, sm, sb = pack(test_ids)
        pred, vacc = run_fold(fx, fy, fm, vx, vy, vm, sx, sy, sm, n_classes, seed + fold, device)
        val_accs.append(round(vacc, 4))
        correct = (pred == sy) & sm
        hits += int(correct.sum())
        total += int(sm.sum())
        for bb in range(N_TIME_BINS):
            sel = sm & (sb == bb)
            bin_hits[bb] += int((correct & sel).sum())
            bin_total[bb] += int(sel.sum())

    per_bin = [
        {
            "time_bin": b,
            "n_frames": bin_total[b],
            "accuracy": round(bin_hits[b] / bin_total[b], 4),
            "wilson95": wilson(bin_hits[b], bin_total[b]),
        }
        for b in range(N_TIME_BINS)
        if bin_total[b]
    ]
    return {
        "n_episodes": n,
        "n_frames": int(total),
        "accuracy": round(hits / total, 4),
        "wilson95": wilson(hits, total),
        "hits": int(hits),
        "val_accuracy_per_fold": val_accs,
        "per_time_bin": per_bin,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ann_dir = Path(args.annotations).expanduser()
    manifest = json.loads((ann_dir / "manifest.json").read_text())
    tbl = np.load(ann_dir / "progress.npz", allow_pickle=True)
    wanted = [f for f in manifest["families"] if not args.families or f in args.families.split(",")]

    report = {
        "annotation_id": manifest["annotation_id"],
        "pre_registered": "2026-08-29 by coordinator, before any sequence number existed",
        "probe": {
            "arch": f"Linear(d_in->{PROJ}) + tanh + 1-layer GRU(hidden {HIDDEN}) + Linear head",
            "why_gru": "the repo's GDN blocks (robomme_integration/) are policy-side modules "
            "bound to an action head, not usable off the shelf as a bare sequence "
            "encoder; the GRU is the simpler choice the brief preferred",
            "causal": "fed the FULL episode from frame 0, scored only on primary-window frames",
            "budget": f"Adam lr {LR}, full batch, <= {MAX_EPOCHS} epochs, early stop on a "
            f"{int(VAL_FRAC * 100)}% validation split of the TRAIN episodes, "
            f"patience {PATIENCE}, best weights restored",
            "capacity_match": "identical probe and budget for every source; the 512->64 "
            "projection makes the recurrent width identical regardless of "
            "input dimensionality",
            "device": args.device,
        },
        "signature": "E1b beats time_only AND beats raw_tap, by Wilson lower bound",
        "families": {},
    }

    for family in wanted:
        spec = manifest["families"][family]
        domain = "remembench" if family.startswith("rmb_") else "robocasa"
        sel = tbl["family"] == family
        if not sel.any():
            continue
        # time bins are the SAME quantiles §19 used: computed on the primary window only
        prim = sel & tbl["primary_window"]
        edges = np.quantile(tbl["normalized_time"][prim], np.linspace(0, 1, N_TIME_BINS + 1)[1:-1])

        keys = defaultdict(list)
        for i in np.flatnonzero(sel):
            keys[f"{tbl['task'][i]}|{tbl['episode'][i]}"].append(i)
        ep_keys = sorted(keys)

        raw_labels = sorted(set(tbl["progress_label"][prim].tolist()))
        remap = {v: i for i, v in enumerate(raw_labels)}
        n_classes = len(raw_labels)

        meta = []  # per episode: (labels, primary mask, tbin, ntime, frames, task, ep)
        for key in ep_keys:
            idx = np.array(keys[key])
            idx = idx[np.argsort(tbl["frame"][idx])]
            task, ep = key.split("|")
            y = np.array([remap.get(int(v), 0) for v in tbl["progress_label"][idx]], np.int64)
            m = tbl["primary_window"][idx].copy()
            m &= np.array([int(v) in remap for v in tbl["progress_label"][idx]])
            nt = tbl["normalized_time"][idx]
            meta.append(
                (y, m, np.searchsorted(edges, nt, side="right").astype(np.int64), nt, tbl["frame"][idx], task, int(ep))
            )

        fam = {
            "rule": spec["rule"],
            "n_episodes": len(meta),
            "n_classes": n_classes,
            "class_names": [spec["label_names"][str(v)] for v in raw_labels],
            "n_primary_frames": int(sum(int(m.sum()) for _y, m, *_r in meta)),
            "time_bin_edges": [round(float(e), 4) for e in edges],
            "sources": {},
        }

        # ---- the four feature sources ----------------------------------------------------------
        feature_cache = {}
        for name, (root, kind) in SOURCES[domain].items():
            episodes, missing = [], 0
            for y, m, b, _nt, frames, task, ep in meta:
                loaded = load_features(root, kind, domain, task, ep)
                if loaded is None:
                    missing += 1
                    continue
                feats, findex = loaded
                pos = {int(f): j for j, f in enumerate(findex)}
                take = [pos[int(f)] for f in frames if int(f) in pos]
                if len(take) != len(frames):
                    missing += 1
                    continue
                episodes.append((feats[np.array(take)], y, m, b))
            feature_cache[name] = episodes
            t0 = time.time()
            fam["sources"][name] = {
                "missing_episodes": missing,
                **evaluate_source(episodes, n_classes, args.folds, args.seed, args.device),
            }
            fam["sources"][name]["seconds"] = round(time.time() - t0, 1)
            print(
                f"  {family:26} {name:10} acc "
                f"{fam['sources'][name].get('accuracy')} "
                f"({fam['sources'][name]['seconds']}s)",
                flush=True,
            )

        # ---- control (i): the SAME probe fed only the clock -------------------------------------
        clock = [
            (np.stack([nt, frames / 1000.0], 1).astype(np.float32), y, m, b) for (y, m, b, nt, frames, _t, _e) in meta
        ]
        t0 = time.time()
        fam["sources"]["time_only"] = {
            "input": "[normalized_time, frame_index/1000]",
            **evaluate_source(clock, n_classes, args.folds, args.seed, args.device),
        }
        fam["sources"]["time_only"]["seconds"] = round(time.time() - t0, 1)
        print(
            f"  {family:26} {'time_only':10} acc {fam['sources']['time_only']['accuracy']} "
            f"({fam['sources']['time_only']['seconds']}s)",
            flush=True,
        )

        # ---- control (iii): label sequences permuted ACROSS episodes = the chance floor ---------
        base = feature_cache["E1b"]
        rng = np.random.default_rng(args.seed + 999)
        perm = rng.permutation(len(base))
        shuffled = []
        for i, (x, _y, m, b) in enumerate(base):
            src_y = base[perm[i]][1]
            warp = np.minimum((np.arange(len(x)) * len(src_y) // max(len(x), 1)), len(src_y) - 1)
            shuffled.append((x, src_y[warp], m, b))
        t0 = time.time()
        fam["sources"]["label_shuffled"] = {
            "note": "E1b features, label sequences permuted across episodes and resampled to each "
            "episode's own length — marginal and monotone shape preserved, link destroyed",
            **evaluate_source(shuffled, n_classes, args.folds, args.seed, args.device),
        }
        fam["sources"]["label_shuffled"]["seconds"] = round(time.time() - t0, 1)
        print(
            f"  {family:26} {'shuffled':10} acc "
            f"{fam['sources']['label_shuffled']['accuracy']} "
            f"({fam['sources']['label_shuffled']['seconds']}s)",
            flush=True,
        )

        # ---- the registered signature, read off the numbers -------------------------------------
        def acc(n):
            return fam["sources"][n].get("accuracy")

        def lo(n):
            return (fam["sources"][n].get("wilson95") or [None])[0]

        fam["signature"] = {
            "e1b_minus_time_only": round(acc("E1b") - acc("time_only"), 4),
            "e1b_minus_raw_tap": round(acc("E1b") - acc("raw_tap"), 4),
            "e1b_beats_time_only_wilson": bool(lo("E1b") > acc("time_only")),
            "e1b_beats_raw_tap_wilson": bool(lo("E1b") > acc("raw_tap")),
            "passes": bool(lo("E1b") > acc("time_only") and lo("E1b") > acc("raw_tap")),
            "best_source": max(SOURCES[domain], key=lambda n: acc(n)),
        }
        report["families"][family] = fam
        print(f"  -> {family} signature {fam['signature']}", flush=True)

    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).expanduser().write_text(text)
        print(f"-> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
