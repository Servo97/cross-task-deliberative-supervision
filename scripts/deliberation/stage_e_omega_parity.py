#!/usr/bin/env python3
"""D7 EXPERT-REPLAY ORACLE for the Stage-E online ω producer.

Pin D7: "every new eval harness passes an EXPERT-REPLAY ORACLE before policy evals". The producer
in `workspace_models/features/stage_e_omega_producer.py` is a new eval-path component, so before a
single rollout is scored it must be shown to reproduce the ω the policy was actually TRAINED on.

The reference is not a reimplementation — it is the shipped store
`stage_e_runs_md/omega/<cell>/remembench/<Task>/demo_%06d/w.npz`, written by
`train_stage_e.export_omega_store` and consumed verbatim by the Stage-P post-trains.

THREE STAGES, because they falsify different things.

  batch        `omega_episode` (whole episode in one pass) vs the store. Falsifies: loader,
               adapter routing (§16.1), fp16 round trip, lang handling. If this drifts, nothing
               downstream is worth measuring.
  online       `reset()/step()` one grid frame at a time vs the store. Falsifies the causal-prefix
               and absolute-time-embedding semantics the SERVE path depends on. `encode_fused`'s
               mask is a pure causal band, so this should match batch to float noise, and that
               claim is measured rather than asserted.
  lang-substitution   (diagnostic, not a pass/fail gate) The store's ω is conditioned on each
               demo's OWN `lang_global` — the mean over that demo's frames of the tap's masked-mean
               language embedding. A live rollout has no demo and cannot know an episode mean in
               advance without reading the future, so serving must condition on a fixed per-task
               vector instead. This stage quantifies that train/serve gap BEFORE it can be
               mistaken for a result: it re-runs the producer with the task-mean lang and reports
               how far ω moves, plus the within-task spread of `lang_global` itself.

GATE (pre-registered, and NOT to be tuned against): per frame, cos(mine, shipped) >= 0.999 and
max|Δ| within the fp16 storage floor of the stored value. The store is fp16 while the producer
returns fp32, so the floor is fp16 quantisation (~1e-3 relative), never zero. Observed maxima are
always PRINTED so a tighten/loosen decision is made on numbers.

  PYTHONPATH=. python scripts/deliberation/stage_e_omega_parity.py \
      --encoder ~/Research/TRI/wsm_data/deliberation/stage_e_runs_md/E1b_aebbc9a04fa66a94/encoder.pt \
      --omega-root ~/Research/TRI/wsm_data/deliberation/stage_e_runs_md/omega/E1b_s20260828/remembench \
      --pooled-root ~/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k \
      --demos 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.features.stage_e_omega_producer import (  # noqa: E402
    StageEOmegaProducer,
    load_stage_e,
)

COS_BAR = 0.999


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


def fp16_floor(reference: np.ndarray) -> float:
    """Largest |Δ| attributable purely to storing `reference` as fp16: half an ulp per element."""
    ref = np.abs(reference.astype(np.float32))
    ulp = np.where(ref > 0, np.spacing(ref.astype(np.float16)).astype(np.float32), np.float32(6e-8))
    return float(ulp.max())


def pick_demos(omega_root: Path, pooled_root: Path, want: int) -> list[tuple[str, Path, Path]]:
    """Deterministic, spread across tasks: round-robin the tasks so one task cannot dominate."""
    per_task: dict[str, list] = {}
    for task_dir in sorted(omega_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for demo_dir in sorted(task_dir.iterdir()):
            w, p = demo_dir / "w.npz", pooled_root / task_dir.name / demo_dir.name / "p.npz"
            if w.exists() and p.exists():
                per_task.setdefault(task_dir.name, []).append((task_dir.name, w, p))
    picked, round_index = [], 0
    while len(picked) < want:
        added = False
        for task in sorted(per_task):
            if round_index < len(per_task[task]):
                picked.append(per_task[task][round_index])
                added = True
                if len(picked) == want:
                    break
        if not added:
            break
        round_index += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--omega-root", required=True, help=".../omega/<cell>/remembench")
    ap.add_argument("--pooled-root", required=True, help=".../wsm_pooled/rmb_pi_100k")
    ap.add_argument("--domain", default="remembench")
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument(
        "--online-frames", type=int, default=0, help="cap grid frames scored in the online stage per demo (0 = all)"
    )
    ap.add_argument(
        "--lang-mode",
        default="demo",
        choices=("demo", "taskmean", "running", "per_frame", "task_line", "stored"),
        help="conditioning fed to the producer. 'demo' = each demo's own lang_global "
        "(what Stage-E TRAINED on; the parity reference). 'taskmean' = the "
        "per-task vector (serve convention a). 'running' = causal running mean of "
        "the per-frame tap language (serve convention b; needs --lang-root)",
    )
    ap.add_argument(
        "--lang-root", default="", help="rmb_lang_pf store (lang.npz with lang_per_frame), required for 'running'"
    )
    ap.add_argument("--cos-bar", type=float, default=COS_BAR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    omega_root, pooled_root = Path(args.omega_root).expanduser(), Path(args.pooled_root).expanduser()
    encoder, blob = load_stage_e(args.encoder, args.device)
    ckpt_step = blob.get("step") if isinstance(blob, dict) else None
    print(
        f"[enc] {Path(args.encoder).name} encoder_id={blob.get('encoder_id')} "
        f"step={blob.get('step')} domains={blob['domains']} index={blob.get('domain_index')}",
        flush=True,
    )

    demos = pick_demos(omega_root, pooled_root, args.demos)
    if len(demos) < args.demos:
        print(f"FATAL: only {len(demos)} paired demos found, wanted {args.demos}")
        return 2
    print(f"[gate] {len(demos)} demos across {len({d[0] for d in demos})} tasks\n", flush=True)

    # Task-mean lang, for the substitution diagnostic. Built from the SAME demos scored, so the
    # number reported is the one the gate's own episodes would have seen.
    lang_by_task: dict[str, list[np.ndarray]] = {}
    for task, _w, p in demos:
        lang_by_task.setdefault(task, []).append(np.asarray(np.load(p)["lang_global"], dtype=np.float32))
    task_mean = {t: np.mean(v, axis=0) for t, v in lang_by_task.items()}

    # CHECKPOINT/STORE IDENTITY. Parity is only meaningful against the checkpoint that produced the
    # store. `encoder_best.pt` (best eval step) is a different model from the final one that exported
    # ω whenever best != final, and comparing them fails a correct encoder.
    meta_p = Path(args.omega_root).expanduser().parent / "_meta.json"
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:  # noqa: BLE001
            meta = {}
        want_step = meta.get("encoder_step")
        got_step = ckpt_step
        if want_step is not None and got_step is not None and int(want_step) != int(got_step):
            print(
                f"FATAL: ω store was exported by step {want_step} but --encoder is step "
                f"{got_step} ({Path(args.encoder).name}). Point --encoder at the checkpoint that "
                f"produced the store (normally encoder.pt, not encoder_best.pt)."
            )
            return 2

    rows, worst = [], {"batch_cos": 1.0, "online_cos": 1.0, "batch_abs": 0.0, "online_abs": 0.0}
    for task, w_path, p_path in demos:
        blob_w = np.load(w_path)
        w_ref = np.asarray(blob_w["w"], dtype=np.float32)  # [F,512] fp16 on disk
        pz = np.load(p_path)
        p, lang = np.asarray(pz["p"]), np.asarray(pz["lang_global"], dtype=np.float32)
        if len(p) != len(w_ref):
            print(f"FATAL {task}/{p_path.parent.name}: p has {len(p)} frames, w has {len(w_ref)}")
            return 2
        if not np.array_equal(
            np.asarray(pz["frame_indices"], np.int64), np.asarray(blob_w["frame_indices"], np.int64)
        ):
            print(f"FATAL {task}/{p_path.parent.name}: frame grids differ between p.npz and w.npz")
            return 2

        # The conditioning under test. 'demo' reproduces training exactly; the other two are the
        # candidate CAUSAL serve conventions, scored against the same shipped ω.
        if args.lang_mode == "demo":
            lang_used = lang
        elif args.lang_mode == "taskmean":
            lang_used = task_mean[task]
        elif args.lang_mode == "stored":
            # The conditioning vector TRAINING ACTUALLY USED. `export_omega_store` writes
            # `lang_global=corpus.lang[episode_index]`, i.e. the post-`--lang-mode` vector, so this
            # makes D7 a true identity check for any lang mode whose conditioning is one vector per
            # episode (episode_mean, task_mean). The `taskmean` mode CANNOT do this: it recomputes a
            # mean over the demos parity happens to sample, while training averaged over the whole
            # loaded corpus -- different vectors, so a `--lang-mode serve` run fails the gate for a
            # reason that has nothing to do with the producer. Measured 2026-09-02: taskmean gives
            # cos 0.99984 / max|Δ| 1.4e-01 against an fp16 floor of 3.9e-03.
            if "lang_global" not in blob_w.files:
                print(f"FATAL {task}/{p_path.parent.name}: w.npz has no lang_global")
                return 2
            lang_used = np.asarray(blob_w["lang_global"], dtype=np.float32)
        elif args.lang_mode == "per_frame":
            # RoboCerebra: the frame's OWN subtask instruction, shipped inside p.npz by
            # rcb_pooled_tap. This is the only mode where train and serve condition on the same
            # object -- it is what the harness re-pins and what the policy is served (§24.10), and
            # Stage-E trains on it via `train_stage_e --lang-mode per_frame`. Parity here is
            # therefore an IDENTITY check, not a serve-convention approximation.
            if "lang_per_frame" not in pz.files:
                print(f"FATAL {task}/{p_path.parent.name}: p.npz has no lang_per_frame")
                return 2
            lpf = np.asarray(pz["lang_per_frame"], dtype=np.float32)
            if len(lpf) != len(p):
                print(f"FATAL {task}/{p_path.parent.name}: lang stream {len(lpf)} != {len(p)}")
                return 2
            lang_used = lpf
        elif args.lang_mode == "task_line":
            # The episode goal, constant and known at reset. Causal, and Stage-E-shaped (one vector
            # per episode), so it is the fallback if per_frame is rejected.
            if "lang_task_line" not in pz.files:
                print(f"FATAL {task}/{p_path.parent.name}: p.npz has no lang_task_line")
                return 2
            lang_used = np.asarray(pz["lang_task_line"], dtype=np.float32)
        else:
            lz = np.load(Path(args.lang_root).expanduser() / task / p_path.parent.name / "lang.npz")
            lpf = np.asarray(lz["lang_per_frame"], dtype=np.float32)
            if len(lpf) != len(p):
                print(f"FATAL {task}/{p_path.parent.name}: lang stream {len(lpf)} != {len(p)} frames")
                return 2
            # Running mean in float32 then rounded to fp16 — the same order `pi_pooled_tap` used to
            # form lang_global, so convention (b) converges to the training statistic exactly.
            lang_used = (
                (np.cumsum(lpf, axis=0) / np.arange(1, len(lpf) + 1)[:, None]).astype(np.float16).astype(np.float32)
            )

        prod = StageEOmegaProducer(encoder, args.domain, lang_used, args.device)
        w_batch = prod.omega_episode(p).float().cpu().numpy()
        c_b = cosine(w_batch, w_ref)
        a_b = float(np.abs(w_batch - w_ref).max())

        n_online = len(p) if args.online_frames <= 0 else min(len(p), args.online_frames)
        prod.reset()
        w_online = np.stack([prod.step(p[i]).float().cpu().numpy() for i in range(n_online)])
        c_o = cosine(w_online, w_ref[:n_online])
        a_o = float(np.abs(w_online - w_ref[:n_online]).max())

        sub = StageEOmegaProducer(encoder, args.domain, task_mean[task], args.device)
        w_sub = sub.omega_episode(p).float().cpu().numpy()
        c_s = cosine(w_sub, w_ref)

        floor = fp16_floor(w_ref)
        worst["batch_cos"] = min(worst["batch_cos"], float(c_b.min()))
        worst["online_cos"] = min(worst["online_cos"], float(c_o.min()))
        worst["batch_abs"] = max(worst["batch_abs"], a_b)
        worst["online_abs"] = max(worst["online_abs"], a_o)
        rows.append(
            {
                "task": task,
                "demo": p_path.parent.name,
                "frames": int(len(p)),
                "lang_mode": args.lang_mode,
                "batch_cos_min": float(c_b.min()),
                "batch_cos_mean": float(c_b.mean()),
                "batch_absmax": a_b,
                "online_cos_min": float(c_o.min()),
                "online_absmax": a_o,
                "online_frames": int(n_online),
                "langsub_cos_min": float(c_s.min()),
                "langsub_cos_mean": float(c_s.mean()),
                "fp16_floor": floor,
            }
        )
        print(
            f"  {task:26s} {p_path.parent.name} F={len(p):4d}  "
            f"batch cos>={c_b.min():.6f} |Δ|<={a_b:.2e}  "
            f"online cos>={c_o.min():.6f} |Δ|<={a_o:.2e}  "
            f"(floor {floor:.2e})  langsub cos>={c_s.min():.4f}",
            flush=True,
        )

    floor_all = max(r["fp16_floor"] for r in rows)
    batch_ok = worst["batch_cos"] >= args.cos_bar and worst["batch_abs"] <= floor_all
    online_ok = worst["online_cos"] >= args.cos_bar and worst["online_abs"] <= floor_all
    verdict = "PASS" if (batch_ok and online_ok) else "FAIL"

    langsub_min = min(r["langsub_cos_min"] for r in rows)
    langsub_mean = float(np.mean([r["langsub_cos_mean"] for r in rows]))
    within = {}
    for t, v in lang_by_task.items():
        if len(v) > 1:
            m = np.stack(v)
            within[t] = float(cosine(m, np.broadcast_to(task_mean[t], m.shape)).min())

    print(f"\n-- gate (bar: cos >= {args.cos_bar}, max|Δ| <= fp16 floor {floor_all:.2e}) --")
    print(
        f"  batch   worst cos {worst['batch_cos']:.6f}  max|Δ| {worst['batch_abs']:.2e}  "
        f"-> {'PASS' if batch_ok else 'FAIL'}"
    )
    print(
        f"  online  worst cos {worst['online_cos']:.6f}  max|Δ| {worst['online_abs']:.2e}  "
        f"-> {'PASS' if online_ok else 'FAIL'}"
    )
    print("\n-- lang-substitution diagnostic (NOT a gate) --")
    print(f"  ω(task-mean lang) vs shipped ω: worst-frame cos {langsub_min:.4f}, per-demo mean {langsub_mean:.4f}")
    if within:
        print(
            f"  within-task cos(demo lang_global, task mean): min {min(within.values()):.4f} over {len(within)} tasks"
        )
    print(f"\nVERDICT: {verdict}")

    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(
            json.dumps(
                {
                    "encoder": str(args.encoder),
                    "encoder_id": blob.get("encoder_id"),
                    "omega_root": str(omega_root),
                    "cos_bar": args.cos_bar,
                    "fp16_floor": floor_all,
                    "worst": worst,
                    "verdict": verdict,
                    "langsub": {
                        "worst_frame_cos": langsub_min,
                        "mean_cos": langsub_mean,
                        "within_task_cos_min": (min(within.values()) if within else None),
                    },
                    "rows": rows,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
