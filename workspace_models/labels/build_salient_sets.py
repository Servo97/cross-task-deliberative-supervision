"""WSM label stage C: MolmoPoint pixels -> 8x8 patch ids -> per-episode SalientSet npz.

RoboCasa365 3-view port of Isaac-GR00T/wsm/vlm_label/build_salient_sets.py. Pure numpy (+ PIL
only for --qc-dir). Produces a per-episode npz the WSM-base trainer consumes:
  keyframes      [K] int64                 (episode frame indices, Qwen completion frames)
  salient_<view> [K] object                (per-keyframe per-view LOCAL patch-id arrays, [0,64))
  flow_3d        [K,0,3] float32           (placeholder for the v2.0 3D-flow stage)
  n_frames       int64
  views          str (json list)

The DexJoCo reference scored VLM labels against a MuJoCo sim-oracle (IoU). RoboCasa unseen tasks
have NO oracle, so geometry correctness is verified VISUALLY via --qc-dir: it applies the GR00T
center-crop+resize to the frame and overlays the selected patch cells + the mapped points, so a
human can confirm the salient patches land on the right objects. (See geometry.py — the crop is
load-bearing; an off geometry silently mislabels every episode.)

  python -m workspace_models.labels.build_salient_sets --task <Task> --in ~/Research/TRI/wsm_data/wsm_vlm_rc_v0 \
      --qc-dir ~/Research/TRI/wsm_data/wsm_vlm_rc_v0/_qc
  ... --geom pi_libero          # 2-view RoboCerebra/LIBERO (128 global ids)
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

GEOM_MODULES = {"groot": "geometry", "pi": "pi_geometry", "pi_libero": "pi_geometry_libero"}
# npz filename suffix. Both pi geometries keep "_pi": the trainer's filename is fixed, so label
# specs/embodiments are separated by ROOT (see --out), never by filename.
GEOM_SUFFIX = {"groot": "", "pi": "_pi", "pi_libero": "_pi"}
_QC_COLORS = [
    (255, 64, 64),
    (64, 255, 64),
    (64, 160, 255),
    (255, 220, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 140, 0),
    (180, 120, 255),
]


def _qc_overlay(
    frame: np.ndarray, obj_points: dict, patches: np.ndarray, out_png: Path, G, title: str = "", scale: int = 2
) -> None:
    """Apply the backbone crop+resize to `frame`, shade selected patch cells, plot mapped points.

    The shaded cells are the TRUE token cells: for the pi geometries the 14->8 bins are NON-UNIFORM
    (`G._EDGES`), so drawing a uniform 8x8 split would put 6 of 8 boundaries in the wrong place and
    the overlay — which is the only acceptance test we have, there being no oracle — would "verify"
    a geometry it never rendered. Points are colored PER OBJECT with a legend, because the question
    the overlay must answer is whether a point lands on the object it NAMES, not merely somewhere
    plausible. Rendered at `scale`x with NEAREST so cells and points stay legible.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).resize((G.TARGET, G.TARGET))  # native 256, defensive
    img = img.crop((G._OFF, G._OFF, G._OFF + G._CROP, G._OFF + G._CROP)).resize((G.TARGET, G.TARGET)).convert("RGB")
    S = max(int(scale), 1)
    W = G.TARGET * S
    img = img.resize((W, W), Image.NEAREST)
    dr = ImageDraw.Draw(img, "RGBA")

    edges = getattr(G, "_EDGES", None)  # pi: non-uniform 14->8 bins; groot: uniform
    if edges is not None:
        px = [int(e) * G._PATCH * S for e in edges]
    else:
        px = [i * (G.TARGET // G.N_GRID) * S for i in range(G.N_GRID + 1)]
    for k in range(1, G.N_GRID):  # faint grid on the true cell boundaries
        dr.line([(px[k], 0), (px[k], W)], fill=(0, 0, 0, 70))
        dr.line([(0, px[k]), (W, px[k])], fill=(0, 0, 0, 70))
    for pid in np.atleast_1d(patches):
        r, c = int(pid) // G.N_GRID, int(pid) % G.N_GRID
        dr.rectangle([px[c], px[r], px[c + 1], px[r + 1]], fill=(255, 0, 0, 70))

    legend = []
    for i, (name, pts) in enumerate(sorted((obj_points or {}).items())):
        col = _QC_COLORS[i % len(_QC_COLORS)]
        if not pts:
            legend.append((col, f"{name}: NO POINTS"))
            continue
        for x, y in G.pixels_to_model_view(np.asarray(pts, dtype=np.float64)):
            x, y = float(x) * S, float(y) * S
            dr.ellipse([x - 2 * S, y - 2 * S, x + 2 * S, y + 2 * S], outline=(0, 0, 0, 255), width=1)
            dr.ellipse([x - 2 * S + 1, y - 2 * S + 1, x + 2 * S - 1, y + 2 * S - 1], fill=col + (255,))
        legend.append((col, f"{name} ({len(pts)})"))
    lines = ([(None, title)] if title else []) + legend
    if lines:
        dr.rectangle([0, 0, W, 4 + 11 * len(lines)], fill=(0, 0, 0, 150))
        for j, (col, txt) in enumerate(lines):
            if col is not None:
                dr.rectangle([3, 4 + 11 * j + 2, 10, 4 + 11 * j + 9], fill=col + (255,))
            dr.text((13 if col is not None else 3, 4 + 11 * j), txt[:70], fill=(255, 255, 255, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument(
        "--out",
        dest="out_dir",
        default="",
        help="label root for the npz output (default: --in). Point a new spec at its OWN "
        "root: the trainer's filename is fixed at vlm_episode_pi_%%06d.npz, so specs "
        "are separated by ROOT, never by filename.",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="label-spec suffix on the INPUT points json (ep*_points<tag>.json), e.g. "
        "'_causal_v1'. Not applied to the npz name — see --out.",
    )
    ap.add_argument(
        "--geom",
        choices=sorted(GEOM_MODULES),
        default="groot",
        help="patch geometry: groot (256 crop-0.95, 8x8), pi (224 resize, 14x14->bin "
        "8x8, 3 RoboCasa views) or pi_libero (same math, 2 RoboCerebra views)",
    )
    ap.add_argument("--qc-dir", default="", help="if set, write geometry-QC overlay PNGs here")
    args = ap.parse_args()

    G = importlib.import_module("workspace_models.labels." + GEOM_MODULES[args.geom])
    VIEWS = G.VIEWS
    suffix = GEOM_SUFFIX[args.geom]

    task_dir = Path(args.in_dir).expanduser() / args.task
    out_task_dir = (Path(args.out_dir).expanduser() / args.task) if args.out_dir else task_dir
    out_task_dir.mkdir(parents=True, exist_ok=True)
    pts_name = f"_points{args.tag}.json"
    rows = []
    for pj in sorted(task_dir.glob(f"ep*{pts_name}")):
        ep = int(pj.name[2:5])
        pts = json.loads(pj.read_text())
        frames_npz = np.load(pj.with_name(pj.name.replace(pts_name, "_frames.npz")), allow_pickle=True)
        fidx = frames_npz["frame_indices"]
        kfs, sal = [], {v: [] for v in VIEWS}
        for kf in pts["keyframes"]:
            kfs.append(int(kf["frame"]))
            row = int(np.argmin(np.abs(fidx - int(kf["frame"]))))
            for view in VIEWS:
                by_obj = kf["views"].get(view, {})
                pooled = [p for obj_pts in by_obj.values() for p in obj_pts]
                ids = G.to_patches(pooled, dilate=(view in G.CLOSEUP_VIEWS))
                sal[view].append(ids)
                if args.qc_dir:
                    _qc_overlay(
                        frames_npz[f"frames_{view}"][row],
                        by_obj,
                        ids,
                        Path(args.qc_dir).expanduser()
                        / args.task
                        / f"ep{ep:03d}_kf{int(kf['frame']):03d}_{view}{suffix}{args.tag}.png",
                        G,
                        title=f"{args.task} kf{int(kf['frame'])} {view} | {kf.get('subgoal', '')}",
                    )
        # Combine the 3 views into GLOBAL patch ids (0..191) per keyframe, + the cumulative union.
        # next-keyframe target = salient_global[k]; cumulative target = cumulative_global[k].
        salient_global, cumulative_global, acc = [], [], np.empty(0, dtype=np.int64)
        for ki in range(len(kfs)):
            parts = [G.to_global(sal[v][ki], v) for v in VIEWS if len(sal[v][ki])]
            g = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
            salient_global.append(g)
            acc = np.unique(np.concatenate([acc, g]))
            cumulative_global.append(acc.copy())
        out = out_task_dir / f"vlm_episode{suffix}_{ep:06d}.npz"
        np.savez_compressed(
            out,
            keyframes=np.asarray(kfs, dtype=np.int64),
            salient_global=np.asarray(salient_global, dtype=object),  # next-keyframe target set
            cumulative_global=np.asarray(cumulative_global, dtype=object),  # cumulative target set
            flow_3d=np.zeros((len(kfs), 0, 3), dtype=np.float32),  # v2.0 placeholder
            n_frames=np.int64(frames_npz["n_frames"]),
            views=json.dumps(list(VIEWS)),
            **{f"salient_{v}": np.asarray(sal[v], dtype=object) for v in VIEWS},
        )
        counts = {v: int(sum(len(s) for s in sal[v])) for v in VIEWS}
        rows.append({"ep": ep, "n_kf": len(kfs), "patches": counts})
        print(rows[-1], flush=True)

    report = out_task_dir / f"rc_report{suffix}{args.tag}.json"
    report.write_text(json.dumps(rows, indent=2))
    if rows:
        tot = {v: float(np.mean([r["patches"][v] for r in rows])) for v in VIEWS}
        print(
            f"\n=== {args.task} (n={len(rows)} episodes) ===\n"
            f"mean keyframes/ep: {np.mean([r['n_kf'] for r in rows]):.1f}\n"
            f"mean salient patches/ep per view: "
            f"{', '.join(f'{v}={tot[v]:.1f}' for v in VIEWS)}\n-> {report}",
            flush=True,
        )


if __name__ == "__main__":
    main()
