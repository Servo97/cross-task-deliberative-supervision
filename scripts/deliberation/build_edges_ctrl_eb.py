#!/usr/bin/env python3
"""H14 Stage-E — label artifact **v2b**: adds the `ctrl-Eb` edge set beside v2's `E1b`.

Why v2b exists (A14, seed-replication of the primary contrast). The funnel left E1 vs ctrl-E
INDETERMINATE at n=2 seeds: ctrl-E's 11.96 lift sits inside E1's own same-config seed spread
(8.0 / 12.9). The pre-registered resolution is 3 seeds x {E1b, ctrl-Eb, E1b-analog05}, paired by
seed. `ctrl-Eb` is the control that makes the comparison read as "Qwen positives vs embedding
positives" and NOTHING else:

    E1b     = Qwen EQUIVALENT+ANALOGOUS positives, minus binding-flagged, + v2 hard negatives
    ctrl-Eb = top-k descriptor-embedding positives, minus binding-flagged, + THE SAME v2 hard
              negatives (byte-identical rows, strengths included)

The old `ctrl-E` differed from E1 on TWO axes at once — its positives came from embedding-nearest
mining AND it carried no hard negatives at all (edges_ctrl-E.npz is 137,452 rows, every one
EQUIVALENT). Half of the E1b-vs-ctrl-E gap could therefore have been the missing denominator term.
v2b closes that: the hard-negative half of the objective is identical between the two arms.

Rule for the overlap, applied because it is forced, not chosen. 50,041 of ctrl-E's 137,452
embedding positives are v2 hard negatives — expected, since "binding differs" is exactly what
descriptor cosine cannot see. A pair cannot be a positive and a hard negative in the same SupCon
kernel (it would be attracted by the numerator and extra-repelled in the denominator), so the
binding rule wins, EXACTLY as v2 rule (i) already made it win over Qwen's own verdict:

    ctrl-Eb positives = ctrl-E positives MINUS every pair that is a v2 hard negative
    ctrl-Eb negatives = v2's CONTRAST rows verbatim (`hardneg`, `binding`, `conf`, `weight`)

`edges_E1b.npz`, `segments.npz`, `vocab.json` and `gate_pairs.npz` are copied BYTE-IDENTICALLY from
v2 (sha256 of each recorded in the manifest), so every cell in the replication reads one artifact
directory and the retrieval gate's ground truth is the same object the funnel scored.

    python scripts/deliberation/build_edges_ctrl_eb.py \
        --v2      ~/Research/TRI/wsm_data/deliberation/stage_e_labels/ab38d9efc0c649a3 \
        --v1      ~/Research/TRI/wsm_data/deliberation/stage_e_labels/bd13c1a48f2dc5be \
        --out     ~/Research/TRI/wsm_data/deliberation/stage_e_labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

EDGE_KINDS = ("EQUIVALENT", "ANALOGOUS", "CONTRAST")
COLUMNS = ("src", "dst", "kind", "conf", "stratum", "cosine", "disagree", "weight", "hardneg", "binding", "orig_kind")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", required=True, help="the FROZEN v2 artifact directory (E1b)")
    ap.add_argument("--v1", required=True, help="the FROZEN v1 artifact directory (ctrl-E source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    v2, v1 = Path(args.v2).expanduser(), Path(args.v1).expanduser()
    e1b = dict(np.load(v2 / "edges_E1b.npz"))
    ctrl_e = dict(np.load(v1 / "edges_ctrl-E.npz"))

    knobs = {
        "base_label_id": v2.name,
        "ctrl_e_label_id": v1.name,
        "e1b_sha": sha256_file(v2 / "edges_E1b.npz"),
        "ctrl_e_sha": sha256_file(v1 / "edges_ctrl-E.npz"),
        "builder_code_sha": code_sha(),
        "rule": (
            "ctrl-Eb = ctrl-E positives minus every pair that is a v2 hard negative, "
            "plus v2's CONTRAST rows verbatim (hardneg/binding/conf/weight preserved)"
        ),
    }
    label_id = hashlib.sha256(json.dumps(knobs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    out_dir = Path(args.out).expanduser() / label_id
    if out_dir.exists() and not args.force:
        print(f"[skip] {out_dir} exists; --force to rebuild")
        print(json.dumps(json.loads((out_dir / "manifest.json").read_text())["counts"], indent=1))
        return

    negative = e1b["kind"] == EDGE_KINDS.index("CONTRAST")
    ns, nd = e1b["src"][negative].astype(np.int64), e1b["dst"][negative].astype(np.int64)
    neg_pairs = set(zip(ns.tolist(), nd.tolist())) | set(zip(nd.tolist(), ns.tolist()))

    cs, cd = ctrl_e["src"].astype(np.int64), ctrl_e["dst"].astype(np.int64)
    collides = np.fromiter(((int(a), int(b)) in neg_pairs for a, b in zip(cs, cd)), bool, len(cs))
    keep = ~collides

    def column(name, mask_pos, mask_neg):
        if name in ctrl_e:
            left = ctrl_e[name][mask_pos]
        else:  # v2-only columns: an embedding positive is not a hard negative and never binding.
            fill = {
                "hardneg": np.float32(0.0),
                "binding": False,
                "orig_kind": np.int8(EDGE_KINDS.index("EQUIVALENT")),
            }[name]
            left = np.full(int(mask_pos.sum()), fill, dtype=np.asarray(fill).dtype)
        return np.concatenate([left, e1b[name][mask_neg]])

    out = {name: column(name, keep, negative) for name in COLUMNS}
    for name in COLUMNS:  # dtype parity with the v2 artifact, so the trainer's casts are no-ops
        out[name] = out[name].astype(e1b[name].dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "edges_ctrl-Eb.npz", **out)
    for name in ("edges_E1b.npz", "segments.npz", "vocab.json", "gate_pairs.npz"):
        shutil.copy2(v2 / name, out_dir / name)

    counts = {
        "ctrl_e_positives_v1": int(len(cs)),
        "ctrl_e_positives_dropped_as_v2_hardneg": int(collides.sum()),
        "ctrl_eb_positives": int(keep.sum()),
        "ctrl_eb_hard_negatives": int(negative.sum()),
        "ctrl_eb_hardneg_full": int((e1b["hardneg"][negative] == 1.0).sum()),
        "ctrl_eb_hardneg_half": int((e1b["hardneg"][negative] == 0.5).sum()),
        "e1b_positives": int((e1b["kind"] <= 1).sum()),
        "e1b_hard_negatives": int(negative.sum()),
    }
    manifest = {
        "label_id": label_id,
        "version": "2b",
        "knobs": knobs,
        "counts": counts,
        "copied_verbatim_from_v2": {
            name: sha256_file(v2 / name) for name in ("edges_E1b.npz", "segments.npz", "vocab.json", "gate_pairs.npz")
        },
        "gate_note": (
            "gate_pairs.npz is byte-identical to v2 and therefore to v1: the retrieval "
            "gate's ground truth must not move across the seed replication"
        ),
        "v2_manifest": json.loads((v2 / "manifest.json").read_text()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"label_id": label_id, "counts": counts}, indent=1))
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
