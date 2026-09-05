#!/usr/bin/env python3
"""H14 Stage-E — label artifact **v2**: binding-aware CONTRAST, built from the v1 artifact + sidecar.

Why v2 exists. A9 measured the frozen schema's CONTRAST verdicts at letter-rule precision 0.172:
72% of them adjudicate to positives, because the schema's "may differ in colour or instance" clause
licenses exactly the substitutions that flip a task's success predicate. The programmatic binding
table (`build_binding_annotations.py`, binding_id 597f3ff5e7cbd6ce) reads the deciding variable —
which burner, which food, which side, which colour, which count — straight out of episode metadata,
with no model in the loop, at 0.94 precision against the intent rule; its union with Qwen CONTRAST
recovers 39/45 planted probes (0.867), clearing F3.

Relabel rules (coordinator, 2026-08-28), applied to the FROZEN v1 edge set:

  (i)   any edge flagged CONTRAST-binding becomes a hard negative at FULL strength, whatever Qwen
        said — this is the rule that moves 37,809 edges out of the positive set;
  (ii)  a Qwen CONTRAST edge that is NOT binding-flagged stays a hard negative at HALF strength
        (intent-rule precision 0.672 — better than a coin flip, not good enough to trust fully);
  (iii) positives = Qwen EQUIVALENT ∪ ANALOGOUS minus (i).

**Hard-negative strength is stored as `hardneg` ∈ [0, 1], which is NOT the trainer's denominator
multiplier.** 0 means "an ordinary negative, like any other frame pair"; 1 means "as repulsive as
the funnel's `--contrast-weight` made every CONTRAST edge". The trainer maps
`m = 1 + hardneg·(contrast_weight − 1)`, so hardneg=0 → m=1.0 and hardneg=1 → m=contrast_weight.
The distinction matters: a literal multiplier of 0 would DELETE the pair from the SupCon denominator
— a second, opposite special role rather than none — and none is what "not a hard negative" means.

`segments.npz`, `vocab.json` and **`gate_pairs.npz` are copied verbatim** from v1. The retrieval
gate's ground truth must not move between v1 and v2 cells, or the funnel's headline number stops
being a comparison.

    python scripts/deliberation/build_edge_labels_v2.py \
        --labels   ~/Research/TRI/wsm_data/deliberation/stage_e_labels/bd13c1a48f2dc5be \
        --sidecar  ~/Research/TRI/wsm_data/deliberation/binding_annotations/597f3ff5e7cbd6ce/relabel_bd13c1a48f2dc5be_strict \
        --out      ~/Research/TRI/wsm_data/deliberation/stage_e_labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

EDGE_KINDS = ("EQUIVALENT", "ANALOGOUS", "CONTRAST")
CONFIDENCES = ("high", "med", "low")

#: (i) full strength for a binding-flagged edge; (ii) half for a Qwen CONTRAST that the binding
#: table cannot corroborate. Both are pre-registered here, not tuned.
HARDNEG_BINDING = 1.0
HARDNEG_QWEN_ONLY = 0.5


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="the FROZEN v1 artifact directory")
    ap.add_argument("--sidecar", required=True, help="relabel_<label_id>_<slotset> directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    v1 = Path(args.labels).expanduser()
    sidecar = Path(args.sidecar).expanduser()
    edges = dict(np.load(v1 / "edges_E1.npz"))
    segments = dict(np.load(v1 / "segments.npz", allow_pickle=True))
    vocab = json.loads((v1 / "vocab.json").read_text())
    flagged = np.load(sidecar / "edges_binding.npz", allow_pickle=True)
    sidecar_manifest = json.loads((sidecar / "manifest.json").read_text())

    knobs = {
        "base_label_id": v1.name,
        "binding_id": sidecar_manifest["binding_id"],
        "sidecar_code_sha": sidecar_manifest["code_sha"],
        "slot_set": sidecar_manifest.get("slot_set"),
        "hardneg_binding": HARDNEG_BINDING,
        "hardneg_qwen_only": HARDNEG_QWEN_ONLY,
        "builder_code_sha": code_sha(),
        "rules": "i: binding-flagged -> hard negative 1.0; ii: Qwen CONTRAST not flagged -> 0.5; "
        "iii: positives = Qwen EQUIVALENT+ANALOGOUS minus (i)",
    }
    label_id = hashlib.sha256(json.dumps(knobs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    out_dir = Path(args.out).expanduser() / label_id
    if out_dir.exists() and not args.force:
        print(f"[skip] {out_dir} exists; --force to rebuild")
        print(json.dumps(json.loads((out_dir / "manifest.json").read_text())["counts"], indent=1))
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flagged pairs are unordered; index both directions so lookup is orientation-free.
    flagged_pairs = set()
    for a, b in zip(flagged["src"].tolist(), flagged["dst"].tolist()):
        flagged_pairs.add((a, b))
        flagged_pairs.add((b, a))
    is_flagged = np.fromiter(
        ((int(a), int(b)) in flagged_pairs for a, b in zip(edges["src"], edges["dst"])), bool, len(edges["src"])
    )

    kind = edges["kind"].copy()
    was_positive = kind <= EDGE_KINDS.index("ANALOGOUS")
    original_kind = kind.copy()

    hardneg = np.zeros(len(kind), np.float32)
    # (i) — flip every flagged edge to CONTRAST at full strength, whatever Qwen said.
    kind[is_flagged] = EDGE_KINDS.index("CONTRAST")
    hardneg[is_flagged] = HARDNEG_BINDING
    # (ii) — a Qwen CONTRAST the binding table cannot corroborate stays a half-strength negative.
    qwen_only = (original_kind == EDGE_KINDS.index("CONTRAST")) & ~is_flagged
    hardneg[qwen_only] = HARDNEG_QWEN_ONLY

    removed = was_positive & is_flagged
    task = segments["task"]
    tasks = vocab["tasks"]
    per_task_removed = Counter()
    per_task_kept = Counter()
    for endpoint in ("src", "dst"):
        for key in edges[endpoint][removed]:
            per_task_removed[tasks[int(task[key])]] += 1
        for key in edges[endpoint][was_positive & ~is_flagged]:
            per_task_kept[tasks[int(task[key])]] += 1

    out = {
        "src": edges["src"],
        "dst": edges["dst"],
        "kind": kind,
        "conf": edges["conf"],
        "stratum": edges["stratum"],
        "cosine": edges["cosine"],
        "disagree": edges["disagree"],
        "weight": edges["weight"],
        "hardneg": hardneg,
        "binding": is_flagged,
        "orig_kind": original_kind,
    }
    np.savez_compressed(out_dir / "edges_E1b.npz", **out)
    for name in ("segments.npz", "vocab.json", "gate_pairs.npz"):
        shutil.copy2(v1 / name, out_dir / name)

    counts = {
        "n_edges": int(len(kind)),
        "n_binding_flagged_in_artifact": int(is_flagged.sum()),
        "n_positives_v1": int(was_positive.sum()),
        "n_positives_v2": int((kind <= EDGE_KINDS.index("ANALOGOUS")).sum()),
        "n_positives_removed": int(removed.sum()),
        "removed_from": {EDGE_KINDS[k]: int((original_kind[removed] == k).sum()) for k in (0, 1)},
        "n_hardneg_full": int((hardneg == HARDNEG_BINDING).sum()),
        "n_hardneg_half": int((hardneg == HARDNEG_QWEN_ONLY).sum()),
        "n_qwen_contrast_v1": int((original_kind == 2).sum()),
        "n_qwen_contrast_corroborated": int(((original_kind == 2) & is_flagged).sum()),
    }
    manifest = {
        "label_id": label_id,
        "version": 2,
        "knobs": knobs,
        "counts": counts,
        "per_task_positives_removed": dict(per_task_removed.most_common()),
        "per_task_positives_kept": dict(per_task_kept.most_common()),
        "copied_verbatim_from_v1": ["segments.npz", "vocab.json", "gate_pairs.npz"],
        "gate_note": (
            "gate_pairs.npz is UNCHANGED from v1 on purpose: the retrieval gate's ground "
            "truth must not move between v1 and v2 cells, or the comparison is void"
        ),
        "sidecar_manifest": sidecar_manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                "label_id": label_id,
                "counts": counts,
                "per_task_positives_removed": manifest["per_task_positives_removed"],
            },
            indent=1,
        )
    )
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
