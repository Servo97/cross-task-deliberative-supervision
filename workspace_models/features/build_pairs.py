"""Cross-demo pairing -> manifest.parquet (the training index + the alignment pairs).

The core-contribution objective aligns a demo's workspace latents with those of a DIFFERENT demo of
the SAME task. This stage builds, for every demo, a deterministic set of candidate same-task
partners (the data loader samples one per step) and records the subgoal-keyframe correspondence
(aligned by subgoal index up to the shorter decomposition). One manifest row per demo: it is the
training index — the loader reads it, mmaps `patch_tokens.npy` for the demo + a sampled partner,
loads the small label/feat files, and builds a batch.

Reads only the LABEL artifacts (ep*_subgoals.json + ep*_frames.npz frame_indices), so it runs before
or after the GPU feature stage. Governed by internal_planning_and_todos/07_wsm_preprocessing_and_revised_plan.md.

  python -m workspace_models.features.build_pairs --frames-dir ~/Research/TRI/wsm_data/wsm_vlm_rc \
      --cache-root ~/Research/TRI/wsm_data/wsm_cache --tasks OpenDrawer,PrepareCoffee,ArrangeBreadBasket \
      --n-partners 4 --out ~/Research/TRI/wsm_data/wsm_cache/manifest.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_VERSION = "wsm-v1"


def _demos_for_task(task_dir: Path, label_suffix: str = "") -> list[dict]:
    recs = []
    for sj in sorted(task_dir.glob("ep*_subgoals.json")):
        ep = int(sj.name[2:5])
        fp = sj.with_name(sj.name.replace("_subgoals.json", "_frames.npz"))
        labels = task_dir / f"vlm_episode{label_suffix}_{ep:06d}.npz"  # "" groot, "_pi" pi-geometry
        if not fp.exists() or not labels.exists():
            continue
        sub = json.loads(sj.read_text())
        fidx = np.load(fp, allow_pickle=True)["frame_indices"]  # npz member-lazy: frames_* not read
        kfs = [int(s.get("completion_frame", 0)) for s in sub.get("subgoals", [])]
        recs.append(
            {
                "ep": ep,
                "F": int(len(fidx)),
                "keyframes": kfs,
                "n_subgoals": len(kfs),
                "frames_path": str(fp),
                "labels_path": str(labels),
            }
        )
    return recs


def build_stage_s_pairs(
    *,
    source_features_manifest: str | Path,
    features_root: str | Path,
    labels_root: str | Path,
    n_partners: int = 4,
    seed: int = 0,
) -> "pd.DataFrame":
    """Stage-S pairs: rows come from the source-feature manifest (never a frames scan). Only demos
    with an available teacher label are trainable; unlabeled demos are excluded and counted. Partner
    candidates are same-task LABELED demos (so a partner always has salient targets)."""
    manifest = json.loads(Path(source_features_manifest).read_text(encoding="utf-8"))
    features_root = Path(features_root)
    labels_root = Path(labels_root)
    rng = np.random.default_rng(seed)
    rows, excluded = [], 0
    for task_rec in manifest["tasks"]:
        task = task_rec["task"]
        labeled = [e for e in task_rec["episodes"] if e["teacher_labels"]["available"]]
        excluded += len(task_rec["episodes"]) - len(labeled)
        ids = [int(e["episode_index"]) for e in labeled]
        by_ep = {int(e["episode_index"]): e for e in labeled}
        if len(labeled) < 2:
            print(f"[pairs:stage-s] {task}: <2 labeled demos, cannot pair — skip", flush=True)
            continue
        for e in labeled:
            ep = int(e["episode_index"])
            others = [x for x in ids if x != ep]
            k = min(n_partners, len(others))
            partners = sorted(int(x) for x in rng.choice(others, size=k, replace=False))
            keyframes = list(
                np.load(labels_root / by_ep[ep]["teacher_labels"]["path"], allow_pickle=True)["keyframes"]
            )
            rows.append(
                {
                    "task": task,
                    "demo_id": ep,
                    "n_frames": int(e["frame_count"]),
                    "n_subgoals": len(keyframes),
                    "keyframes": json.dumps([int(k) for k in keyframes]),
                    "partner_demo_ids": json.dumps(partners),
                    "labels_path": str(labels_root / by_ep[ep]["teacher_labels"]["path"]),
                    "feature_dir": str(features_root / task / f"demo_{ep:06d}"),
                    "cache_version": "stage-s-v1",
                }
            )
        print(f"[pairs:stage-s] {task}: {len(labeled)} labeled demos, {n_partners} partners each", flush=True)
    print(f"[pairs:stage-s] excluded {excluded} unlabeled demos (features exist; no teacher label)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage-s",
        action="store_true",
        help="build pairs from a source-feature manifest (Stage-S); excludes unlabeled demos",
    )
    ap.add_argument("--source-features-manifest", default=None, help="Stage-S: source-feature manifest")
    ap.add_argument("--features-root", default=None, help="Stage-S: canonical feature cache root")
    ap.add_argument("--labels-root", default=None, help="Stage-S: teacher-label root")
    ap.add_argument("--frames-dir", default=None, help="legacy label root (<root>/<task>/ep*)")
    ap.add_argument("--cache-root", default=None, help="legacy feature cache root (<root>/<task>/demo_<ep>/)")
    ap.add_argument("--tasks", default=None, help="legacy comma-separated task names")
    ap.add_argument("--n-partners", type=int, default=4, help="candidate same-task partners per demo")
    ap.add_argument("--label-suffix", default="", help="'' for groot labels, '_pi' for pi-geometry labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.stage_s:
        if not (args.source_features_manifest and args.features_root and args.labels_root):
            raise SystemExit("--stage-s requires --source-features-manifest, --features-root, --labels-root")
        df = build_stage_s_pairs(
            source_features_manifest=args.source_features_manifest,
            features_root=args.features_root,
            labels_root=args.labels_root,
            n_partners=args.n_partners,
            seed=args.seed,
        )
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(
            f"\n[pairs:stage-s] manifest -> {out}  ({len(df)} rows across "
            f"{df['task'].nunique() if len(df) else 0} tasks)",
            flush=True,
        )
        return

    if not (args.frames_dir and args.cache_root and args.tasks):
        raise SystemExit("legacy mode requires --frames-dir, --cache-root, --tasks")

    frames_root = Path(args.frames_dir).expanduser()
    cache_root = Path(args.cache_root).expanduser()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rng = np.random.default_rng(args.seed)
    rows = []
    for task in tasks:
        recs = _demos_for_task(frames_root / task, args.label_suffix)
        if len(recs) < 2:
            print(f"[pairs] {task}: <2 demos, cannot pair — skip", flush=True)
            continue
        ids = [r["ep"] for r in recs]
        by_ep = {r["ep"]: r for r in recs}
        for r in recs:
            others = [e for e in ids if e != r["ep"]]
            k = min(args.n_partners, len(others))
            partners = sorted(int(x) for x in rng.choice(others, size=k, replace=False))
            rows.append(
                {
                    "task": task,
                    "demo_id": r["ep"],
                    "n_frames": r["F"],
                    "n_subgoals": r["n_subgoals"],
                    "keyframes": json.dumps(r["keyframes"]),
                    "partner_demo_ids": json.dumps(partners),
                    # subgoal correspondence = index-aligned up to min(n_sub); store partner keyframes too
                    "partner_keyframes": json.dumps({str(p): by_ep[p]["keyframes"] for p in partners}),
                    "frames_path": r["frames_path"],
                    "labels_path": r["labels_path"],
                    "feature_dir": str(cache_root / task / f"demo_{r['ep']:06d}"),
                    "cache_version": CACHE_VERSION,
                }
            )
        print(f"[pairs] {task}: {len(recs)} demos, {args.n_partners} partners each", flush=True)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"\n[pairs] manifest -> {out}  ({len(df)} demo rows across {df['task'].nunique()} tasks)", flush=True)


if __name__ == "__main__":
    main()
