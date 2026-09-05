#!/usr/bin/env python3
"""build_eval_registry — the pre-registered demo2 registry for ALL tasks (doc 15 D15/D16).

Rule (identical to the WSMv2 trainer): per task, the LAST 5 episode indices present in the feats root.
The trainer's registry (encoder-phase tasks only) must be a strict subset — asserted here. Output is the
single registry consumed by BOTH the post-train dataset install (partner-role exclusion) and the serve
(demo2 selection + membership assert). sha16 printed for the provenance stamp.

  PYTHONPATH=. python workspace_models/features/build_eval_registry.py \
      --feats-root ~/Research/TRI/wsm_data/wsm_policy_feats/groot_65k \
      --trainer-registry ~/Research/TRI/wsm_data/wsm2_runs/orig_65k_matched/registry.json \
      --out ~/Research/TRI/wsm_data/wsm_demo_tokens/orig_65k_matched/registry_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats-root", required=True)
    ap.add_argument("--trainer-registry", default=None, help="wsm2 run registry.json (subset assert)")
    ap.add_argument("--registry-n", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np

    root = Path(args.feats_root).expanduser()
    reg = {}
    for tdir in sorted(d for d in root.iterdir() if d.is_dir()):
        eps = []
        for d in sorted(tdir.glob("demo_*")):
            if not (d / "w.npz").exists():
                continue
            # IDENTICAL filter to the trainer's Pack (len >= 12: room for k=8 targets + a window) — the
            # registry rule must operate on the same demo universe or the subset assert fires.
            if len(np.load(d / "w.npz")["frame_indices"]) < 12:
                continue
            eps.append(int(d.name.split("_")[1]))
        if eps:
            reg[tdir.name] = sorted(eps)[-args.registry_n :]
    if args.trainer_registry:
        tr = json.loads(Path(args.trainer_registry).expanduser().read_text())
        for task, eps in tr.items():
            assert reg.get(task) == eps, f"registry mismatch for {task}: trainer {eps} vs eval {reg.get(task)}"
        print(f"[registry] trainer registry ({len(tr)} tasks) is a strict subset — consistent")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(reg, sort_keys=True)
    out.write_text(txt)
    sha = hashlib.sha256(txt.encode()).hexdigest()[:16]
    print(f"[registry] {len(reg)} tasks -> {out}  sha16={sha}")


if __name__ == "__main__":
    main()
