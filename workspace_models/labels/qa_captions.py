"""QA sheet for stage-A2 segment captions: self-contained HTML, frames + caption per segment.

Renders the SAME frames the captioner showed Qwen (start/mid/end x 3 views per segment) next to
the caption it produced, plus the structural asserts, so the labels can be eyeballed before any
training consumes them (h13 tree §7.4). Images are inlined as base64 JPEG — no external requests.

  ~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.qa_captions \
      --tasks CloseBlenderLid,ArrangeTea --per-task 5 \
      --out internal_planning_and_todos/aug_12/h13_captions_qa.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from workspace_models.labels.caption_segments import (
    CAPTION,
    DEFAULT_DATASET_ROOT,
    DEFAULT_LABELS_ROOT,
    DEFAULT_OUT,
    MAX_CAPTION_TOKENS,
    VIEWS,
    Job,
    decode_views,
    plan_frames,
    read_label,
    resolve_lerobot_dir,
    segments_from_keyframes,
)


def thumb(arr, px: int = 132) -> str:
    from PIL import Image

    im = Image.fromarray(arr).resize((px, px), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def check(rec: dict, segs: list, T: int) -> list[str]:
    """Structural asserts -> list of failure strings (empty = pass)."""
    bad = []
    s = rec["segments"]
    if len(s) != len(segs):
        bad.append(f"segment count {len(s)} != {len(segs)} from keyframes")
    prev = 0
    for i, seg in enumerate(s):
        if seg["t0"] != prev:
            bad.append(f"seg{i}: t0={seg['t0']} != prev end {prev} (gap/overlap)")
        if seg["t1"] <= seg["t0"]:
            bad.append(f"seg{i}: empty [{seg['t0']},{seg['t1']})")
        if not str(seg["text"]).strip():
            bad.append(f"seg{i}: empty text")
        prev = seg["t1"]
    if prev != T:
        bad.append(f"coverage ends at {prev} != T={T}")
    if not (2 <= len(s) <= 8):
        bad.append(f"NOTE segment count {len(s)} outside the typical 2-8")
    return bad


CSS = """
body{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;padding:24px;
background:#0f1115;color:#e6e8eb}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:26px 0 8px;color:#9ecbff}
.meta{color:#8b949e;font-size:12px;margin-bottom:18px}
.ep{border:1px solid #262b33;border-radius:8px;margin:14px 0;padding:12px;background:#151920}
.seg{display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-top:1px solid #222831}
.seg:first-of-type{border-top:none}
.cap{flex:1;min-width:190px}
.cap .t{font-size:14px;font-weight:600;color:#d2f4d2}
.cap .x{font-size:11px;color:#8b949e;margin-top:3px}
.frames{display:flex;gap:5px;flex-wrap:wrap}
figure{margin:0;text-align:center}figure img{display:block;border-radius:3px}
figcaption{font-size:9px;color:#6e7681;margin-top:2px}
.ok{color:#3fb950}.bad{color:#f85149;font-weight:600}
.note{color:#d29922}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #262b33;padding:3px 8px}
"""


def spotcheck(croot: Path, lroot: Path, per_task: int, min_tasks: int) -> int:
    """Non-fatal early QA over whatever has been written so far, ACROSS tasks.

    The canary covered a single task; a systematic failure on other tasks (wrong segment count,
    empty text, a task whose videos fail to decode) would otherwise only surface at the end. Prints
    WARN lines and always returns 0 — this must never take a healthy run down.
    """
    tasks = sorted(
        p.name
        for p in croot.iterdir()
        if p.is_dir() and not p.name.startswith("_") and any(p.glob("ep_*.captions.json"))
    )
    print(f"[spotcheck] {len(tasks)} task(s) with captions: {', '.join(tasks[:8])}{' ...' if len(tasks) > 8 else ''}")
    if len(tasks) < min_tasks:
        print(
            f"[spotcheck] WARN only {len(tasks)} task(s) written, wanted >= {min_tasks} "
            f"— too early, or the run is stuck on one task"
        )

    total = bad_eps = 0
    for task in tasks:
        for fp in sorted(croot.glob(f"{task}/ep_*.captions.json"))[:per_task]:
            total += 1
            try:
                rec = json.loads(fp.read_text())
                ep, T = int(rec["episode_id"]), int(rec["n_frames"])
                kf, _ = read_label(lroot / task / f"vlm_episode_pi_{ep:06d}.npz")
                segs = segments_from_keyframes(kf, T)
                fails = [f for f in check(rec, segs, T) if not f.startswith("NOTE")]
                if [(s["t0"], s["t1"]) for s in rec["segments"]] != segs:
                    fails.append("segment extents != frozen keyframes")
                for s in rec["segments"]:
                    if len(str(s["text"]).split()) > 25:
                        fails.append(f"caption suspiciously long: {s['text'][:60]!r}")
            except Exception as e:
                fails = [f"unreadable: {e}"]
            if fails:
                bad_eps += 1
                print(f"[spotcheck] WARN {task}/ep_{fp.stem}: {'; '.join(fails[:3])}")
    verdict = "OK" if bad_eps == 0 and len(tasks) >= min_tasks else "WARN"
    print(
        f"[spotcheck] {verdict}: {total} episodes checked across {len(tasks)} task(s), "
        f"{bad_eps} with structural problems"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spotcheck", action="store_true", help="non-fatal cross-task structural check over what exists so far"
    )
    ap.add_argument("--min-tasks", type=int, default=2)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--captions-root", default=DEFAULT_OUT)
    ap.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--max-images", type=int, default=72)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    croot = Path(args.captions_root).expanduser()
    droot = Path(args.dataset_root).expanduser()
    lroot = Path(args.labels_root).expanduser()

    if args.spotcheck:
        raise SystemExit(spotcheck(croot, lroot, args.per_task, args.min_tasks))
    if not args.tasks or not args.out:
        raise SystemExit("--tasks and --out are required unless --spotcheck is given")
    parts, n_ep, n_bad, n_seg_tot = [], 0, 0, 0

    for task in [t for t in args.tasks.split(",") if t]:
        files = sorted(croot.glob(f"{task}/ep_*.captions.json"))[: args.per_task]
        if not files:
            parts.append(f"<h2>{html.escape(task)}</h2><p class='bad'>no caption files</p>")
            continue
        parts.append(f"<h2>{html.escape(task)} &mdash; {len(files)} episodes</h2>")
        root = resolve_lerobot_dir(droot, task)
        for fp in files:
            rec = json.loads(fp.read_text())
            ep = int(rec["episode_id"])
            kf, _ = read_label(lroot / task / f"vlm_episode_pi_{ep:06d}.npz")
            T = int(rec["n_frames"])
            segs = segments_from_keyframes(kf, T)
            fails = check(rec, segs, T)
            n_ep += 1
            n_seg_tot += len(rec["segments"])
            hard = [f for f in fails if not f.startswith("NOTE")]
            if hard:
                n_bad += 1
            plan, _ = plan_frames(segs, args.max_images)
            job = Job(task, ep, T, kf, segs, plan, root, fp)
            decode_views(job)

            status = (
                "<span class='ok'>PASS</span>"
                if not fails
                else " ".join(
                    f"<span class='{'note' if f.startswith('NOTE') else 'bad'}'>{html.escape(f)}</span>" for f in fails
                )
            )
            rows = [
                f"<div class='ep'><div class='x'><b>ep {ep:06d}</b> &nbsp; T={T} &nbsp; "
                f"keyframes={kf} &nbsp; {len(rec['segments'])} segments &nbsp; {status}</div>"
            ]
            for i, seg in enumerate(rec["segments"]):
                imgs = ""
                if not job.error:
                    for f in plan[i]:
                        for v in VIEWS:
                            imgs += (
                                f"<figure><img src='data:image/jpeg;base64,{thumb(job.frames[v][f])}'>"
                                f"<figcaption>{f} {CAPTION[v]}</figcaption></figure>"
                            )
                else:
                    imgs = f"<span class='bad'>{html.escape(job.error)}</span>"
                rows.append(
                    f"<div class='seg'><div class='cap'><div class='t'>{i}. "
                    f"{html.escape(seg['text'])}</div><div class='x'>frames [{seg['t0']}, "
                    f"{seg['t1']}) &nbsp; len {seg['t1'] - seg['t0']}</div></div>"
                    f"<div class='frames'>{imgs}</div></div>"
                )
            rows.append("</div>")
            parts.append("".join(rows))
            job.frames.clear()

    head = (
        f"<h1>H13 segment-caption QA</h1><div class='meta'>"
        f"{n_ep} episodes, {n_seg_tot} segments, <b>{n_bad}</b> with structural failures &middot; "
        f"caption cap {MAX_CAPTION_TOKENS} tokens &middot; segments are the FROZEN keyframes "
        f"(mirrors train_wsm_base/data.py::per_frame_subgoal_idx)</div>"
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"<!doctype html><meta charset='utf-8'><title>H13 caption QA</title><style>{CSS}</style>{head}{''.join(parts)}"
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) — {n_ep} episodes, {n_bad} failing")


if __name__ == "__main__":
    main()
