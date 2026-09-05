"""Build the per-task language table for WSM-conditioned EVAL.

The WSM encoder was trained with the Qwen expanded-prompt embedding as its per-frame `cond_lang`
(subgoal-dropout=1.0 -> global). Eval scenes are novel and have NO online Qwen expansion, so to keep
the online encoder IN-DISTRIBUTION we feed it a per-task REPRESENTATIVE expanded-prompt embedding =
the mean of that task's cached `lang_global` vectors. The eval wrapper uses table[task] as both the
encoder's cond_lang and the modulator's lang. Per-demo (train) -> per-task-mean (eval) is a benign
within-distribution shift; no retrain.

Reads `lang_global` from each demo's record (w.npz from generate_policy_features, or feats.npz from the
feature cache) under <root>/<task>/<demo>/, averages per task, writes one npz:

  task_lang_table.npz   { tasks: [T] str, lang: [T, lang_dim] fp16,  expanded: [T] str (if --cache-root) }

`expanded` (optional): a per-task REPRESENTATIVE Qwen expanded-prompt STRING (the first cache demo's
`expanded_prompt`). The WSM eval tap fed the backbone the EXPANDED prompt at cache time; serving the terse
env instruction online shifts the encoder's patch/lang INPUTS off-distribution (a WSM-only handicap — the
baseline policy is unaffected). With `expanded` shipped, serve_pi_05_wsm.py --tap-prompt expanded feeds the
tap the per-task expanded string, matching training. Needs --cache-root (the feats cache carries the string;
w.npz does not).

  python -m workspace_models.features.make_task_lang_table \
      --root ~/Research/TRI/wsm_data/wsm_policy_feats/pi_step100000 --cache-root ~/Research/TRI/wsm_data/wsm_cache_pi \
      --out ~/Research/TRI/wsm_data/wsm_policy_feats/pi_step100000/task_lang_table.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_lang(demo_dir: Path) -> np.ndarray | None:
    for fname in ("w.npz", "feats.npz"):
        fp = demo_dir / fname
        if fp.exists():
            d = np.load(fp)
            if "lang_global" in d.files:
                return np.asarray(d["lang_global"], dtype=np.float32)
    return None


def _read_expanded(task_cache_dir: Path) -> str | None:
    """The first cache demo's Qwen expanded_prompt string (per-task canonical for the eval tap)."""
    for demo in sorted(task_cache_dir.glob("demo_*")):
        fp = demo / "feats.npz"
        if fp.exists():
            d = np.load(fp, allow_pickle=True)
            if "expanded_prompt" in d.files:
                return str(d["expanded_prompt"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="<root>/<task>/<demo>/ with w.npz (or feats.npz)")
    ap.add_argument(
        "--cache-root",
        default=None,
        help="feature cache (<root>/<task>/demo_*/feats.npz) to pull per-task expanded_prompt strings",
    )
    ap.add_argument("--out", required=True, help="output task_lang_table.npz")
    ap.add_argument(
        "--allow-legacy-expanded-prompts",
        action="store_true",
        help="opt into the historical Qwen-expanded per-task table (Stage-S forbids it)",
    )
    args = ap.parse_args()
    if not args.allow_legacy_expanded_prompts:
        raise SystemExit(
            "legacy expanded-prompt task table is Stage-S-incompatible: use "
            "scripts/launch/build_stage_s_task_lang_table.py for the canonical terse table; pass "
            "--allow-legacy-expanded-prompts only for legacy (non-Stage-S) experiments"
        )

    root = Path(args.root).expanduser()
    cache_root = Path(args.cache_root).expanduser() if args.cache_root else None
    tasks, langs, expanded = [], [], []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        vecs = [v for demo in sorted(task_dir.glob("demo_*")) if (v := _read_lang(demo)) is not None]
        if not vecs:
            print(f"[lang-table] {task_dir.name}: no lang_global found — skip", flush=True)
            continue
        exp = _read_expanded(cache_root / task_dir.name) if cache_root else None
        if cache_root and not exp:
            print(f"[lang-table] {task_dir.name}: WARNING no expanded_prompt in cache — using ''", flush=True)
        tasks.append(task_dir.name)
        langs.append(np.mean(vecs, axis=0))
        expanded.append(exp or "")
        print(
            f"[lang-table] {task_dir.name}: mean over {len(vecs)} demos"
            f"{' | expanded=' + repr(exp[:48]) + '...' if exp else ''}",
            flush=True,
        )

    if not tasks:
        raise SystemExit(f"no tasks with lang_global under {root}")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    save = dict(tasks=np.array(tasks), lang=np.stack(langs).astype(np.float16))
    if cache_root:
        save["expanded"] = np.array(expanded)
    np.savez(out, **save)
    print(
        f"[lang-table] wrote {len(tasks)} tasks x {langs[0].shape[0]} dim"
        f"{' (+expanded strings)' if cache_root else ''} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
