#!/usr/bin/env python3
"""H14 A13(e) — PROGRESS-STATE decodability with a TIME-CONFOUND control.

Pre-registered 2026-08-29, together with `build_progress_annotations.py` and BEFORE any measurement
was taken. Report metrics only; nothing is selected on them.

Why this gate and not the previous two. H14's Markovianization gate has been run on two kinds of
slot and both turned out to be PERCEPTION rather than memory: RoboCasa's predicate-bound variables
are visible from frame 0 (§14.8), and ReMemBench's "hidden" sides are scene-layout constants that
the RAW frozen pi0.5 tap decodes at USE time (§17). The one thing a per-frame tap cannot carry is
what has ALREADY BEEN DONE in this episode. That is what is measured here.

The confound this gate must survive. Progress is monotone in time within an episode, so ANY feature
that encodes elapsed time will "decode progress". The gate therefore carries its own null:

  (iv) TIME-ONLY BASELINE — the same classifier, the same folds, on the single scalar
       `normalized_time`. This is the confound made explicit.
  (v)  TIME-MATCHED — normalized_time is cut into GLOBAL QUINTILES over the family's scorable
       frames, and accuracy is reported WITHIN each bin against the within-bin training prior. If
       the label is already ~determined by the time bin, the within-bin prior is high and the
       within-bin lift collapses; a feature that beats its within-bin prior is carrying something
       elapsed time does not give.

Measurements, per family x source:

  (i)   FRAME     frame-level features w_t
  (ii)  POOLED    causal history pools over the FULL episode stream (mean and max over frames <= t),
                  evaluated on the scorable frames
  (iii) the same two for the RAW POOLED TAP — the control that decides whether the ω stream adds
        anything over the frozen backbone
  (iv)  time-only, (v) time-matched, as above

  SIGNATURE (pre-registered): (ii) for E1b beats (iv), AND beats the raw tap's (ii), both WITHIN
  time bins, while (i) for the raw tap is ~ the time-only baseline. Read as: the ω stream carries
  progress beyond what elapsed time and the current frame give.

Protocol. Nearest-centroid, 5-fold BY EPISODE, chance = the TRAINING-frame label prior scored per
test frame (so an imbalanced family cannot look good by predicting the majority). Cosine metric for
the high-dimensional feature sources, matching `rmb_phase_decodability.py`; EUCLIDEAN for the
one-dimensional time-only baseline, where cosine would erase the feature (every scalar row
normalises to +-1). Labels are per FRAME here, not per episode as in the binding/phase probes, so
centroids are built from training FRAMES grouped by label.

Wilson 95% intervals are on the FRAME count and are therefore anticonservative (frames of one
episode are correlated and many share a label); the episode count is printed beside every interval
and the split is by episode, so the estimate itself is not leaked across the fold boundary.

    PYTHONPATH=. python scripts/deliberation/progress_decodability.py \
        --annotations ~/Research/TRI/wsm_data/deliberation/progress_annotations/<id> \
        --out ~/Research/TRI/wsm_data/deliberation/progress_decodability.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DELIB = Path("~/Research/TRI/wsm_data/deliberation").expanduser()
POOLED = Path("~/Research/TRI/wsm_data/wsm_pooled").expanduser()

#: domain -> {source name: (root, kind)}.  `omega` roots are <root>/<domain>/<task>/demo_%06d/w.npz;
#: `tap` roots are <root>/<task>/demo_%06d/p.npz.  RoboCasa uses the FULL Stage-E cells; ReMemBench
#: has only the smokes, which is stated in every rmb row rather than glossed.
DEFAULT_SOURCES = {
    "robocasa": {
        "E1b": (DELIB / "stage_e_runs/omega/E1b", "omega"),
        "ctrl-0b": (DELIB / "stage_e_runs/omega/ctrl-0b", "omega"),
        "untrained": (DELIB / "stage_e_runs/omega/untrained", "omega"),
        "raw_tap": (POOLED / "pi_100k", "tap"),
    },
    "remembench": {
        "E1b_smoke": (DELIB / "stage_e_runs_rmb_smoke/omega/E1b_smoke_rmb", "omega"),
        "ctrl-0b_smoke": (DELIB / "stage_e_runs_rmb_smoke/omega/ctrl0b_smoke_rmb", "omega"),
        "untrained": (DELIB / "stage_e_runs_rmb_smoke/omega/untrained_rmb", "omega"),
        "raw_tap": (POOLED / "rmb_pi_100k", "tap"),
    },
}

N_TIME_BINS = 5


def wilson(hits: int, total: int, z: float = 1.96) -> list:
    if total == 0:
        return [None, None]
    p = hits / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def causal_pools(features: np.ndarray) -> dict:
    """mean and max of every prefix: pool[t] = f(features[:t+1]).  Causal by construction."""
    counts = np.arange(1, len(features) + 1, dtype=np.float32)[:, None]
    return {"mean": np.cumsum(features, 0) / counts, "max": np.maximum.accumulate(features, 0)}


def _ridge_fit(x: np.ndarray, y: np.ndarray, names: np.ndarray, ridge: float = 1e-2):
    """Multinomial least squares (one-vs-all ridge) — the CAPACITY CONTROL for the gate.

    Nearest centroid is the pre-registered probe, but a negative result from a weak probe is not a
    finding, so every measurement is repeated with a full linear read-out. Closed form, so it costs
    one 513x513 solve per fold rather than an optimiser.
    """
    xb = np.concatenate([x, np.ones((len(x), 1), np.float32)], 1)
    onehot = np.zeros((len(y), len(names)), np.float32)
    onehot[np.arange(len(y)), np.searchsorted(names, y)] = 1.0
    gram = xb.T @ xb
    gram.flat[:: gram.shape[0] + 1] += ridge * (np.trace(gram) / gram.shape[0] + 1e-6)
    return np.linalg.solve(gram, xb.T @ onehot)


def nearest_centroid_kfold(per_ep: list, metric: str = "cosine", folds: int = 5, seed: int = 0) -> dict:
    """per_ep = [(X [n,D], y [n] int, bin [n] int), ...] — one entry per EPISODE.

    k-fold BY EPISODE; centroids from training FRAMES grouped by label; chance = the training-frame
    label prior scored on each test frame.  Also returns the TIME-MATCHED breakdown: within each
    global normalized-time bin, accuracy against the within-bin training prior; the MAJORITY-class
    rate (a stricter reference than the prior); and the RIDGE capacity control.
    """
    usable = [i for i, (x, y, _b) in enumerate(per_ep) if len(x)]
    all_labels = set()
    for i in usable:
        all_labels.update(per_ep[i][1].tolist())
    if len(usable) < folds * 2 or len(all_labels) < 2:
        return {"n_episodes": len(usable), "note": "too few episodes or a single class"}

    rng = np.random.default_rng(seed)
    order = rng.permutation(usable)
    hits = total = 0
    prior_hits = 0.0
    majority_hits = 0
    ridge_hits = 0
    ridge_bin_hits = Counter()
    bin_hits = Counter()
    bin_total = Counter()
    bin_prior = defaultdict(float)

    for fold in range(folds):
        test = set(order[fold::folds].tolist())
        train = [i for i in usable if i not in test]
        tx = np.concatenate([per_ep[i][0] for i in train], 0)
        ty = np.concatenate([per_ep[i][1] for i in train], 0)
        tb = np.concatenate([per_ep[i][2] for i in train], 0)
        names = np.array(sorted(set(ty.tolist())))
        if len(names) < 2:
            continue
        centroids = np.stack([tx[ty == n].mean(0) for n in names]).astype(np.float32)
        if metric == "cosine":
            centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        prior = {int(k): v / len(ty) for k, v in Counter(ty.tolist()).items()}
        majority = Counter(ty.tolist()).most_common(1)[0][0]
        weights = _ridge_fit(tx, ty, names)
        per_bin_prior = {}
        for b in range(N_TIME_BINS):
            m = tb == b
            n = int(m.sum())
            per_bin_prior[b] = {int(k): v / n for k, v in Counter(ty[m].tolist()).items()} if n else {}

        for i in test:
            x, y, b = per_ep[i]
            if metric == "cosine":
                unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
                pred = names[(unit @ centroids.T).argmax(1)]
            else:
                d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
                pred = names[d2.argmin(1)]
            correct = pred == y
            xb = np.concatenate([x, np.ones((len(x), 1), np.float32)], 1)
            ridge_correct = names[(xb @ weights).argmax(1)] == y
            hits += int(correct.sum())
            ridge_hits += int(ridge_correct.sum())
            majority_hits += int((y == majority).sum())
            total += len(y)
            prior_hits += float(sum(prior.get(int(v), 0.0) for v in y))
            for bb in range(N_TIME_BINS):
                m = b == bb
                if not m.any():
                    continue
                bin_hits[bb] += int(correct[m].sum())
                ridge_bin_hits[bb] += int(ridge_correct[m].sum())
                bin_total[bb] += int(m.sum())
                bin_prior[bb] += float(sum(per_bin_prior[bb].get(int(v), 0.0) for v in y[m]))

    if total == 0:
        return {"n_episodes": len(usable), "note": "no scorable frames"}

    accuracy = hits / total
    chance = prior_hits / total
    per_bin = []
    tm_hits = tm_total = tm_ridge = 0
    tm_prior = 0.0
    for b in range(N_TIME_BINS):
        if not bin_total[b]:
            continue
        per_bin.append(
            {
                "time_bin": b,
                "n_frames": bin_total[b],
                "accuracy": round(bin_hits[b] / bin_total[b], 4),
                "ridge_accuracy": round(ridge_bin_hits[b] / bin_total[b], 4),
                "within_bin_chance": round(bin_prior[b] / bin_total[b], 4),
                "wilson95": wilson(bin_hits[b], bin_total[b]),
            }
        )
        tm_hits += bin_hits[b]
        tm_ridge += ridge_bin_hits[b]
        tm_total += bin_total[b]
        tm_prior += bin_prior[b]
    tm_acc = tm_hits / tm_total if tm_total else None
    tm_chance = tm_prior / tm_total if tm_total else None
    tm_ridge_acc = tm_ridge / tm_total if tm_total else None
    return {
        "n_episodes": len(usable),
        "n_frames": int(total),
        "accuracy": round(accuracy, 4),
        "wilson95": wilson(hits, total),
        "chance_prior": round(chance, 4),
        "majority_rate": round(majority_hits / total, 4),
        "lift": round(accuracy / chance, 3) if chance > 0.01 else None,
        "beats_chance": bool(wilson(hits, total)[0] > chance),
        "ridge": {
            "accuracy": round(ridge_hits / total, 4),
            "wilson95": wilson(ridge_hits, total),
            "beats_chance": bool(wilson(ridge_hits, total)[0] > chance),
            "beats_majority": bool(wilson(ridge_hits, total)[0] > majority_hits / total),
        },
        "time_matched": {
            "accuracy": round(tm_acc, 4) if tm_acc is not None else None,
            "ridge_accuracy": round(tm_ridge_acc, 4) if tm_ridge_acc is not None else None,
            "within_bin_chance": round(tm_chance, 4) if tm_chance is not None else None,
            "lift": round(tm_acc / tm_chance, 3) if tm_chance and tm_chance > 0.01 else None,
            "ridge_lift": round(tm_ridge_acc / tm_chance, 3) if tm_chance and tm_chance > 0.01 else None,
            "beats_within_bin_chance": bool(wilson(tm_hits, tm_total)[0] > tm_chance) if tm_total else None,
            "ridge_beats_within_bin_chance": bool(wilson(tm_ridge, tm_total)[0] > tm_chance) if tm_total else None,
            "per_bin": per_bin,
        },
    }


def load_features(root: Path, kind: str, domain: str, task: str, episode: int):
    if kind == "tap":
        path, key = root / task / f"demo_{episode:06d}" / "p.npz", "p"
    else:
        path, key = root / domain / task / f"demo_{episode:06d}" / "w.npz", "w"
    if not path.exists():
        return None
    blob = np.load(path)
    return np.asarray(blob[key], np.float32), np.asarray(blob["frame_indices"], np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="progress_annotations/<id>")
    ap.add_argument(
        "--window",
        default="primary",
        choices=("primary", "all"),
        help="primary = the family's own progress>=1 window (pre-registered); "
        "all = every frame, reported as a secondary row",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default="", help="comma list; default = all")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ann_dir = Path(args.annotations).expanduser()
    manifest = json.loads((ann_dir / "manifest.json").read_text())
    tbl = np.load(ann_dir / "progress.npz", allow_pickle=True)
    keep = tbl["primary_window"] if args.window == "primary" else np.ones(len(tbl["frame"]), bool)

    wanted = [f for f in manifest["families"] if not args.families or f in args.families.split(",")]

    report = {
        "annotation_id": manifest["annotation_id"],
        "window": args.window,
        "pre_registered": "2026-08-29, with the annotation builder, before any measurement",
        "protocol": {
            "classifier": "nearest centroid on training FRAMES grouped by label",
            "metric": "cosine for feature sources, euclidean for the 1-D time-only baseline",
            "folds": f"{args.folds}-fold BY EPISODE, seed {args.seed}",
            "chance": "training-frame label prior scored per test frame",
            "time_bins": f"{N_TIME_BINS} global quantile bins of normalized_time, computed once "
            "per family over its scorable frames",
            "wilson_caveat": "intervals are on the frame count and are anticonservative; the "
            "episode count is reported beside every interval",
        },
        "signature": "(ii) pooled ω for E1b beats the time-only baseline AND the raw tap's (ii), "
        "both WITHIN time bins, while (i) frame-level raw tap is ~ time-only.",
        "families": {},
    }

    for family in wanted:
        spec = manifest["families"][family]
        domain = "remembench" if family.startswith("rmb_") else "robocasa"
        m = keep & (tbl["family"] == family)
        if not m.any():
            continue
        ep_key = np.array([f"{t}|{e}" for t, e in zip(tbl["task"][m], tbl["episode"][m])])
        ntime = tbl["normalized_time"][m]
        edges = np.quantile(ntime, np.linspace(0, 1, N_TIME_BINS + 1)[1:-1])
        tbin = np.searchsorted(edges, ntime, side="right").astype(np.int32)

        idx_by_ep = defaultdict(list)
        for i, k in enumerate(ep_key):
            idx_by_ep[k].append(i)
        frames_all = tbl["frame"][m]
        labels_all = tbl["progress_label"][m].astype(np.int32)

        fam = {
            "rule": spec["rule"],
            "unit": spec["unit"],
            "tasks": spec["tasks"],
            "n_episodes": len(idx_by_ep),
            "n_frames": int(m.sum()),
            "label_counts": {
                spec["label_names"].get(str(k), str(k)): v for k, v in sorted(Counter(labels_all.tolist()).items())
            },
            "time_bin_edges": [round(float(e), 4) for e in edges],
            "measurements": {},
        }

        # -------- (iv) the time-only baseline, same folds and same bins -------------------------
        # Two forms, because a fair null has to have the same capacity as the probe it nulls:
        #   scalar   normalized_time itself (euclidean nearest centroid = threshold at midpoints)
        #   onehot20 an indicator over 20 equal-width time slices, so the SAME cosine-centroid and
        #            the SAME ridge read-out can express an ARBITRARY function of elapsed time.
        time_scalar, time_onehot = [], []
        onehot_bin = np.clip((ntime * 20).astype(int), 0, 19)
        for key in sorted(idx_by_ep):
            i = np.array(idx_by_ep[key])
            time_scalar.append((ntime[i][:, None].astype(np.float32), labels_all[i], tbin[i]))
            oh = np.zeros((len(i), 20), np.float32)
            oh[np.arange(len(i)), onehot_bin[i]] = 1.0
            time_onehot.append((oh, labels_all[i], tbin[i]))
        fam["measurements"]["time_only_baseline"] = {
            "note": "the confound made explicit — nothing but elapsed time is given to the probe",
            "frame": nearest_centroid_kfold(time_scalar, "euclid", args.folds, args.seed),
            "onehot20": nearest_centroid_kfold(time_onehot, "cosine", args.folds, args.seed),
        }

        # -------- the feature sources ------------------------------------------------------------
        for name, (root, kind) in DEFAULT_SOURCES[domain].items():
            variants = {"frame": [], "pool_mean": [], "pool_max": []}
            missing = 0
            for key in sorted(idx_by_ep):
                task, ep = key.split("|")
                i = np.array(idx_by_ep[key])
                loaded = load_features(root, kind, domain, task, int(ep))
                if loaded is None:
                    missing += 1
                    continue
                feats, findex = loaded
                pools = causal_pools(feats)  # over the FULL episode stream
                pos = {int(f): j for j, f in enumerate(findex)}
                sel = np.array([pos[int(f)] for f in frames_all[i] if int(f) in pos])
                if len(sel) != len(i):  # grid mismatch — refuse to guess
                    missing += 1
                    continue
                variants["frame"].append((feats[sel], labels_all[i], tbin[i]))
                variants["pool_mean"].append((pools["mean"][sel], labels_all[i], tbin[i]))
                variants["pool_max"].append((pools["max"][sel], labels_all[i], tbin[i]))
            fam["measurements"][name] = {
                "root": str(root),
                "kind": kind,
                "missing_episodes": missing,
                **{
                    v: nearest_centroid_kfold(variants[v], "cosine", args.folds, args.seed)
                    for v in ("frame", "pool_mean", "pool_max")
                },
            }
        report["families"][family] = fam

    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).expanduser().write_text(text)
        print(f"-> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
