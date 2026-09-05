#!/usr/bin/env python3
"""What is ω actually a function of — the visual stream, or the prompt?

Motivated by the Stage-E serve blocker (§25.3): swapping each demo's own `lang_global` for a
per-task vector barely moved ω for the E1b cells (per-demo mean cos 0.994) but decorrelated the
λ_del = 0 control (0.917 mean, 20 % of demos below 0.90). That asymmetry is a property of the
ENCODERS, so it is measurable directly and it is a finding about what the deliberative contrastive
term does, independent of any serve convention.

DESIGN — a two-way variance decomposition, not a pile of ablations. Take K episodes. Encode the
full K x K grid ω(p_i, lang_j): every episode's visual stream crossed with every episode's language
vector. At each frame position t,

    M     = mean_ij ω(i,j)                       grand mean
    R_i   = mean_j ω(i,j) - M                    VISION main effect (row)
    C_j   = mean_i ω(i,j) - M                    LANGUAGE main effect (column)
    SS_vision = K * sum_i |R_i|^2 ,  SS_lang = K * sum_j |C_j|^2
    SS_total  = sum_ij |ω(i,j) - M|^2 ,  SS_inter = SS_total - SS_vision - SS_lang

and the reported fractions are these sums averaged over frame positions. A trunk that grounds ω in
what the robot SEES puts its variance in SS_vision; a trunk whose ω is largely a restatement of the
prompt puts it in SS_lang. The interaction term is reported rather than hidden: it is the part of ω
that is genuinely about this stream under this instruction.

Episodes are truncated to the shortest in the set so the grid is rectangular; the frame grid is the
cached stride-8 one, so position t means the same thing across episodes.

Cross-checks alongside the decomposition (p held fixed at each episode's own stream):
  cos to the episode's own ω under (a) the task-mean lang, (b) another task's lang, (c) a zero
  lang vector — the last being the "how much is left without language at all" bound.

  PYTHONPATH=. python scripts/deliberation/stage_e_lang_vision_sensitivity.py \
      --encoder .../E1b_aebbc9a04fa66a94/encoder.pt --pooled-root .../wsm_pooled/rmb_pi_100k
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


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(den, 1e-12)


def pick(pooled_root: Path, per_task: int, tasks: list[str] | None) -> list[tuple[str, Path]]:
    out = []
    for task_dir in sorted(pooled_root.iterdir()):
        if not task_dir.is_dir() or (tasks and task_dir.name not in tasks):
            continue
        demos = [d for d in sorted(task_dir.iterdir()) if (d / "p.npz").exists()]
        out.extend((task_dir.name, d / "p.npz") for d in demos[:per_task])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--pooled-root", required=True)
    ap.add_argument("--domain", default="remembench")
    ap.add_argument("--per-task", type=int, default=1, help="episodes per task in the grid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pooled_root = Path(args.pooled_root).expanduser()
    encoder, blob = load_stage_e(args.encoder, args.device)
    label = args.label or Path(args.encoder).parent.name
    chosen = pick(pooled_root, args.per_task, None)
    if len(chosen) < 3:
        print(f"FATAL: need >=3 episodes, found {len(chosen)}")
        return 2

    streams, langs, names = [], [], []
    for task, p_path in chosen:
        z = np.load(p_path)
        streams.append(np.asarray(z["p"]))
        langs.append(np.asarray(z["lang_global"], dtype=np.float32))
        names.append(f"{task}/{p_path.parent.name}")
    T = min(len(s) for s in streams)
    streams = [s[:T] for s in streams]
    K = len(streams)
    print(f"[{label}] encoder_id={blob.get('encoder_id')}  grid {K}x{K} episodes, T={T} frames", flush=True)

    # Full K x K grid: omega[i][j] = encode(vision_i, language_j)
    grid = np.zeros((K, K, T, 512), dtype=np.float32)
    for j, lang in enumerate(langs):
        prod = StageEOmegaProducer(encoder, args.domain, lang, args.device)
        for i, p in enumerate(streams):
            grid[i, j] = prod.omega_episode(p).float().cpu().numpy()

    M = grid.mean(axis=(0, 1))  # [T,512]
    R = grid.mean(axis=1) - M  # [K,T,512] vision main effect
    C = grid.mean(axis=0) - M  # [K,T,512] language main effect
    ss_vision = K * (R**2).sum(axis=(0, 2))  # [T]
    ss_lang = K * (C**2).sum(axis=(0, 2))  # [T]
    ss_total = ((grid - M) ** 2).sum(axis=(0, 1, 3))  # [T]
    ss_inter = ss_total - ss_vision - ss_lang
    frac = lambda x: float(np.mean(x / np.maximum(ss_total, 1e-12)))  # noqa: E731
    f_vision, f_lang, f_inter = frac(ss_vision), frac(ss_lang), frac(ss_inter)

    # Interventions with each episode's OWN visual stream held fixed.
    task_of = [n.split("/")[0] for n in names]
    by_task: dict[str, list[int]] = {}
    for i, t in enumerate(task_of):
        by_task.setdefault(t, []).append(i)
    task_mean = {t: np.mean([langs[i] for i in idx], axis=0) for t, idx in by_task.items()}
    zero = np.zeros_like(langs[0])

    cos_task, cos_other, cos_zero = [], [], []
    for i, p in enumerate(streams):
        own = grid[i, i]
        other_j = (i + K // 2) % K
        cos_task.append(
            cosine(
                StageEOmegaProducer(encoder, args.domain, task_mean[task_of[i]], args.device)
                .omega_episode(p)
                .float()
                .cpu()
                .numpy(),
                own,
            )
        )
        cos_other.append(cosine(grid[i, other_j], own))
        cos_zero.append(
            cosine(
                StageEOmegaProducer(encoder, args.domain, zero, args.device).omega_episode(p).float().cpu().numpy(),
                own,
            )
        )
    agg = lambda v: (float(np.mean([x.mean() for x in v])), float(np.min([x.min() for x in v])))  # noqa: E731
    m_task, w_task = agg(cos_task)
    m_other, w_other = agg(cos_other)
    m_zero, w_zero = agg(cos_zero)

    print(
        f"  variance in ω:  VISION {100 * f_vision:5.1f}%   LANGUAGE {100 * f_lang:5.1f}%   "
        f"interaction {100 * f_inter:5.1f}%"
    )
    print(
        f"  cos to own ω:   task-mean lang {m_task:.4f} (worst {w_task:.4f})   "
        f"other-task lang {m_other:.4f} (worst {w_other:.4f})   "
        f"zero lang {m_zero:.4f} (worst {w_zero:.4f})"
    )

    result = {
        "label": label,
        "encoder": str(args.encoder),
        "encoder_id": blob.get("encoder_id"),
        "K": K,
        "T": T,
        "episodes": names,
        "frac_vision": f_vision,
        "frac_lang": f_lang,
        "frac_interaction": f_inter,
        "cos_taskmean_mean": m_task,
        "cos_taskmean_worst": w_task,
        "cos_othertask_mean": m_other,
        "cos_othertask_worst": w_other,
        "cos_zerolang_mean": m_zero,
        "cos_zerolang_worst": w_zero,
    }
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(json.dumps(result, indent=1))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
