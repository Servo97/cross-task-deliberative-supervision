#!/usr/bin/env python3
"""H14 Stage-E build 1 — pass-2 edge store -> FRAME-LEVEL training labels (content-addressed).

Authority: `internal_planning_and_todos/aug_22/deliberative_workspace_plan.md` §4 + amendments
A1 (circularity / ctrl-E / disagreement subset), A2 (frame-level SupCon), A4 (the encoder-cell
funnel). Contract: `scripts/deliberation/edge_schema.md` §8 — edges are DATA for a contrastive
objective and nothing else.

What this produces, once, for every cell of the funnel:

    <out>/<label_id>/
      segments.npz            per-segment: domain, task, episode, segment index, [t0, t1)
      vocab.json              domain / task id vocabularies (so int ids are readable)
      edges_E1.npz            the Qwen edges          (EQUIVALENT / ANALOGOUS / CONTRAST)
      edges_ctrl-E.npz        top-k descriptor-embedding neighbours, NO Qwen        (A1b)
      edges_ctrl-S.npz        type-preserving SHUFFLED rewire of edges_E1           (§6 ctrl-S)
      edges_ctrl-T.npz        same-task positives, NO Qwen                          (§6 ctrl-T)
      gate_pairs.npz          the held-out retrieval-gate pair set, incl. the A1d
                              DISAGREEMENT flag (Qwen verdict vs descriptor-cosine ranking)
      manifest.json           counts, A1c quota-floor measurement, every sha

`label_id` = sha256 over {edge_store_id, index sha, embedding manifest sha, every knob below,
builder code sha}. Change a knob, get a new artifact; nothing downstream can silently drift.

Frame mapping (A2): a segment's [t0, t1) are EPISODE frame indices, while the frozen-tap store
caches a strided subset. The trainer resolves segment -> frames through the tap's own
`frame_indices`, so this artifact stores the episode-frame range and never a cached-frame index —
the mapping belongs to whichever tap the cell consumes.

    python scripts/deliberation/build_edge_labels.py \
        --edge-store ~/Research/TRI/wsm_data/deliberation/pass2_store/edges/<edge_store_id> \
        --index      ~/Research/TRI/wsm_data/deliberation/pass2_store/index/segments.jsonl \
        --embed      ~/Research/TRI/wsm_data/deliberation/pass2_store/embed \
        --out        ~/Research/TRI/wsm_data/deliberation/stage_e_labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---- frozen consumer-side vocabularies -------------------------------------------------------
# MUST stay identical in ORDER to train_stage_e.DOMAINS: the `domain` column written here is an
# index into this tuple, and Corpus reads it back as `DOMAINS.index(<tap name>)`. Appending (never
# inserting) keeps every frozen artifact's indices valid. Without robocerebra here, line ~90's
# `DOMAINS.index(r["domain"])` raises ValueError on the first robocerebra row and the 4-domain v1
# build dies outright.
DOMAINS = ("robocasa", "remembench", "robomme", "robocerebra")
EDGE_KINDS = ("EQUIVALENT", "ANALOGOUS", "CONTRAST")  # UNRELATED is dropped, never stored
CONFIDENCES = ("high", "med", "low")
STRATA = ("within_task", "cross_task", "cross_domain", "mined_hard_neg")

#: Default consumer weights (§8: `confidence` -> per-edge weight; `low` excluded by default).
#: ANALOGOUS defaults to a FULL-weight positive; the `E1-analog05` cell down-weights it to 0.5 as
#: the sensitivity arm (coordinator ruling 2026-08-28, replacing the E1-lowconf cell: with
#: 211.6k high / 21.9k med / 20 low, the low-confidence exclusion flag is a measured no-op, so
#: an ANALOGOUS-weight sweep is the informative sensitivity axis instead).
#: These are the artifact's DEFAULTS; the trainer recomputes per-edge weight from `kind`/`conf`
#: plus its own knobs (including lambda_xdom for cross-domain edges), so a cell never needs a
#: fresh label artifact.
KIND_WEIGHT = {"EQUIVALENT": 1.0, "ANALOGOUS": 1.0, "CONTRAST": 1.0}
CONFIDENCE_WEIGHT = {"high": 1.0, "med": 0.5, "low": 0.25}


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------------
def load_index(path: Path) -> tuple[dict, dict]:
    """seg_id -> key, plus the per-segment arrays in key order."""
    keys, rows = {}, []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            keys[record["seg_id"]] = len(rows)
            rows.append(record)
    tasks = sorted({r["task"] for r in rows})
    task_id = {t: i for i, t in enumerate(tasks)}
    segments = {
        "domain": np.array([DOMAINS.index(r["domain"]) for r in rows], np.int8),
        "task": np.array([task_id[r["task"]] for r in rows], np.int32),
        "episode": np.array([int(r["episode"]) for r in rows], np.int32),
        "segment": np.array([int(r["segment"]) for r in rows], np.int16),
        "t0": np.array([int(r["t0"]) for r in rows], np.int32),
        "t1": np.array([int(r["t1"]) for r in rows], np.int32),
        "subskill": np.array([str((r.get("descriptor") or {}).get("subskill", "")) for r in rows]),
    }
    return keys, {"segments": segments, "tasks": tasks, "n": len(rows)}


def read_buckets(edge_store: Path, keys: dict) -> tuple[dict, list]:
    """Parse every bucket; return the raw edge rows and the per-bucket cosine ranking.

    A bucket whose anchor or candidate is missing from the index is dropped with a count, never
    silently mapped onto a neighbouring id.
    """
    src, dst, kind, conf, stratum, cosine, disagree = [], [], [], [], [], [], []
    dropped = Counter()
    bucket_files = sorted(edge_store.glob("buckets/*/*/*.bucket.json"))
    for path in bucket_files:
        try:
            bucket = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            dropped["unparseable_bucket"] += 1
            continue
        verdicts = bucket.get("verdicts") or []
        # A1d: the informative subset is where the Qwen verdict DISAGREES with cosine ranking.
        # Rank the bucket's candidates by descending descriptor cosine, then a positive in the
        # bottom half — or a CONTRAST in the top half — is a pair cosine ranking would have got
        # wrong. Buckets missing cosines cannot contribute to the gate (flag stays False).
        cosines = [v.get("cosine") for v in verdicts]
        order = None
        if all(isinstance(c, (int, float)) for c in cosines) and len(cosines) > 1:
            rank = np.empty(len(cosines), np.int32)
            rank[np.argsort(-np.asarray(cosines, np.float64), kind="stable")] = np.arange(len(cosines))
            order = rank
        half = len(verdicts) / 2.0
        for position, verdict in enumerate(verdicts):
            edge_type = verdict.get("type")
            if edge_type not in EDGE_KINDS:
                if edge_type != "UNRELATED":
                    dropped["bad_type"] += 1
                continue
            a, b = verdict.get("anchor_id"), verdict.get("candidate_id")
            if a not in keys or b not in keys or a == b:
                dropped["unmapped_or_self"] += 1
                continue
            confidence = verdict.get("confidence")
            if confidence not in CONFIDENCES:
                dropped["bad_confidence"] += 1
                continue
            stratum_name = verdict.get("stratum")
            src.append(keys[a])
            dst.append(keys[b])
            kind.append(EDGE_KINDS.index(edge_type))
            conf.append(CONFIDENCES.index(confidence))
            stratum.append(STRATA.index(stratum_name) if stratum_name in STRATA else -1)
            value = cosines[position]
            cosine.append(float(value) if isinstance(value, (int, float)) else np.nan)
            if order is None:
                disagree.append(False)
            elif edge_type == "CONTRAST":
                disagree.append(bool(order[position] < half))
            else:
                disagree.append(bool(order[position] >= half))
    edges = {
        "src": np.asarray(src, np.int32),
        "dst": np.asarray(dst, np.int32),
        "kind": np.asarray(kind, np.int8),
        "conf": np.asarray(conf, np.int8),
        "stratum": np.asarray(stratum, np.int8),
        "cosine": np.asarray(cosine, np.float32),
        "disagree": np.asarray(disagree, bool),
    }
    return edges, [dict(dropped), len(bucket_files)]


# --------------------------------------------------------------------------------------------
# Derived control edge sets (A4 funnel)
# --------------------------------------------------------------------------------------------
def weights_for(edges: dict) -> np.ndarray:
    kind_w = np.asarray([KIND_WEIGHT[k] for k in EDGE_KINDS], np.float32)[edges["kind"]]
    conf_w = np.asarray([CONFIDENCE_WEIGHT[c] for c in CONFIDENCES], np.float32)[edges["conf"]]
    return kind_w * conf_w


def shuffled_rewire(edges: dict, seed: int) -> dict:
    """ctrl-S: type-preserving rewire. Destinations are permuted WITHIN each edge type, so the
    type histogram, every source's degree, and the per-type confidence mix survive; only *which*
    segment a source is tied to is destroyed. Self-loops are re-drawn."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in edges.items()}
    for k in range(len(EDGE_KINDS)):
        idx = np.flatnonzero(edges["kind"] == k)
        if len(idx) < 2:
            continue
        permuted = rng.permutation(idx)
        for _ in range(8):
            collide = out["src"][idx] == edges["dst"][permuted]
            if not collide.any():
                break
            permuted[collide] = rng.permutation(permuted[collide])
        out["dst"][idx] = edges["dst"][permuted]
        keep = out["src"][idx] != out["dst"][idx]
        if not keep.all():
            out["dst"][idx[~keep]] = out["src"][idx[~keep]]  # marked, dropped below
    valid = out["src"] != out["dst"]
    return {k: v[valid] for k, v in out.items()}


def task_id_positives(segments: dict, n_target: int, seed: int) -> dict:
    """ctrl-T: positives = 'same task, different episode', NO Qwen. Sampled to the SAME edge count
    as E1's positive set so the two cells pay the same contrastive budget."""
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for key, (task, episode) in enumerate(zip(segments["task"], segments["episode"])):
        by_task[(int(segments["domain"][key]), int(task))].append((key, int(episode)))
    src, dst = [], []
    task_keys = list(by_task)
    if not task_keys:
        return {
            k: np.zeros(0, dtype=v)
            for k, v in (
                ("src", np.int32),
                ("dst", np.int32),
                ("kind", np.int8),
                ("conf", np.int8),
                ("stratum", np.int8),
                ("cosine", np.float32),
                ("disagree", bool),
            )
        }
    sizes = np.asarray([len(by_task[t]) for t in task_keys], np.float64)
    probability = sizes / sizes.sum()
    while len(src) < n_target:
        picks = rng.choice(len(task_keys), size=min(4096, n_target - len(src)), p=probability)
        for p in picks:
            members = by_task[task_keys[p]]
            if len(members) < 2:
                continue
            i, j = rng.integers(0, len(members), 2)
            if members[i][1] == members[j][1]:
                continue
            src.append(members[i][0])
            dst.append(members[j][0])
    src, dst = np.asarray(src[:n_target], np.int32), np.asarray(dst[:n_target], np.int32)
    return {
        "src": src,
        "dst": dst,
        "kind": np.zeros(len(src), np.int8),
        "conf": np.zeros(len(src), np.int8),
        "stratum": np.full(len(src), -1, np.int8),
        "cosine": np.full(len(src), np.nan, np.float32),
        "disagree": np.zeros(len(src), bool),
    }


def embedding_positives(embed_dir: Path, keys: dict, k: int, batch: int = 2048) -> dict:
    """ctrl-E (A1b): positives = top-k descriptor-embedding neighbours, NO Qwen. This — not
    ctrl-T — is the 'is the deliberation worth it' cell, because EQUIVALENT is a subset of
    embedding-nearest BY CONSTRUCTION of the mining step."""
    ids = json.loads((embed_dir / "ids.json").read_text())
    embeddings = np.load(embed_dir / "emb.npy")
    row_of = np.full(len(ids), -1, np.int64)
    for row, seg_id in enumerate(ids):
        row_of[row] = keys.get(seg_id, -1)
    usable = np.flatnonzero(row_of >= 0)
    x = embeddings[usable].astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    mapped = row_of[usable]
    src, dst, cosine = [], [], []
    for start in range(0, len(x), batch):
        block = x[start : start + batch]
        similarity = block @ x.T
        similarity[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf
        top = np.argpartition(-similarity, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(start, start + len(block)), k)
        src.append(mapped[rows])
        dst.append(mapped[top.reshape(-1)])
        cosine.append(similarity[np.repeat(np.arange(len(block)), k), top.reshape(-1)])
    src = np.concatenate(src).astype(np.int32)
    dst = np.concatenate(dst).astype(np.int32)
    return {
        "src": src,
        "dst": dst,
        "kind": np.zeros(len(src), np.int8),
        "conf": np.zeros(len(src), np.int8),
        "stratum": np.full(len(src), -1, np.int8),
        "cosine": np.concatenate(cosine).astype(np.float32),
        "disagree": np.zeros(len(src), bool),
    }


# --------------------------------------------------------------------------------------------
# A1c quota-floor measurement (a miss is a HOLD, not a footnote)
# --------------------------------------------------------------------------------------------
def quota_report(edges: dict, segments: dict) -> dict:
    positive = edges["kind"] <= 1
    accepted = positive & (edges["conf"] < CONFIDENCES.index("low"))
    stratum = edges["stratum"][accepted]
    cross_task = np.isin(stratum, [STRATA.index("cross_task"), STRATA.index("cross_domain")])
    cross_domain = stratum == STRATA.index("cross_domain")
    # Structural cross-task/cross-domain, measured from the segment table rather than the driver's
    # stratum label — a stratum is provenance, and the floor is about what was ACCEPTED.
    src_task = segments["task"][edges["src"][accepted]]
    dst_task = segments["task"][edges["dst"][accepted]]
    src_domain = segments["domain"][edges["src"][accepted]]
    dst_domain = segments["domain"][edges["dst"][accepted]]
    n = max(int(accepted.sum()), 1)
    measured_cross_task = (
        float(((src_task != dst_task) | (src_domain != dst_domain)).mean()) if accepted.any() else 0.0
    )
    measured_cross_domain = float((src_domain != dst_domain).mean()) if accepted.any() else 0.0
    # Every task must contribute >= 1 cross-task EQUIVALENT edge, or its isolation is flagged.
    equivalent_cross = (edges["kind"] == 0) & accepted
    e_src, e_dst = edges["src"][equivalent_cross], edges["dst"][equivalent_cross]
    cross = segments["task"][e_src] != segments["task"][e_dst]
    contributing = set(segments["task"][e_src[cross]].tolist()) | set(segments["task"][e_dst[cross]].tolist())
    isolated = sorted(set(range(int(segments["task"].max()) + 1)) - contributing)
    return {
        "n_accepted_positives": int(accepted.sum()),
        "stratum_cross_task_or_domain_frac": round(float(cross_task.mean()) if accepted.any() else 0.0, 4),
        "stratum_cross_domain_frac": round(float(cross_domain.mean()) if accepted.any() else 0.0, 4),
        "measured_cross_task_frac": round(measured_cross_task, 4),
        "measured_cross_domain_frac": round(measured_cross_domain, 4),
        "floor_cross_task_0.40": bool(measured_cross_task >= 0.40),
        "floor_cross_domain_0.15": bool(measured_cross_domain >= 0.15),
        "isolated_task_ids": isolated,
        "n_positive_edges_total": int(positive.sum()),
        "denominator": n,
    }


# --------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge-store", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--ctrl-e-k", type=int, default=0, help="top-k for ctrl-E; 0 = match E1's mean accepted-positive out-degree"
    )
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--allow-empty-domain", default="", help="comma list of domains permitted to contribute zero edges/gate pairs"
    )
    args = ap.parse_args()

    edge_store = Path(args.edge_store).expanduser()
    index_path = Path(args.index).expanduser()
    embed_dir = Path(args.embed).expanduser()

    keys, index = load_index(index_path)
    segments = index["segments"]
    print(f"[index] {index['n']} segments, {len(index['tasks'])} tasks", flush=True)

    edges, (dropped, n_buckets) = read_buckets(edge_store, keys)
    print(f"[edges] {n_buckets} buckets -> {len(edges['src'])} typed edges; dropped={dropped}", flush=True)

    accepted = (edges["kind"] <= 1) & (edges["conf"] < CONFIDENCES.index("low"))
    degree = np.bincount(edges["src"][accepted], minlength=index["n"])
    ctrl_e_k = args.ctrl_e_k or max(1, int(round(float(degree.mean()))))

    knobs = {
        "kind_weight": KIND_WEIGHT,
        "confidence_weight": CONFIDENCE_WEIGHT,
        "ctrl_e_k": ctrl_e_k,
        "seed": args.seed,
        "edge_store_id": edge_store.name,
        "index_sha": file_sha(index_path),
        "embed_manifest": json.loads((embed_dir / "manifest.json").read_text()),
        "builder_code_sha": code_sha(),
        "domains": DOMAINS,
        "edge_kinds": EDGE_KINDS,
        "confidences": CONFIDENCES,
        "strata": STRATA,
    }
    label_id = hashlib.sha256(json.dumps(knobs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    out_dir = Path(args.out).expanduser() / label_id
    if out_dir.exists() and not args.force:
        print(f"[skip] {out_dir} exists (content-addressed); pass --force to rebuild")
        print(json.dumps(json.loads((out_dir / "manifest.json").read_text())["counts"], indent=1))
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "segments.npz", **segments)
    (out_dir / "vocab.json").write_text(json.dumps({"domains": list(DOMAINS), "tasks": index["tasks"]}, indent=1))

    variants = {
        "E1": edges,
        "ctrl-S": shuffled_rewire(edges, args.seed),
        "ctrl-T": task_id_positives(segments, int(accepted.sum()), args.seed),
        "ctrl-E": embedding_positives(embed_dir, keys, ctrl_e_k),
    }
    counts = {}
    for name, variant in variants.items():
        variant = dict(variant)
        variant["weight"] = weights_for(variant)
        np.savez_compressed(out_dir / f"edges_{name}.npz", **variant)
        counts[name] = {
            "n_edges": int(len(variant["src"])),
            "by_kind": {EDGE_KINDS[k]: int((variant["kind"] == k).sum()) for k in range(3)},
            "by_confidence": {CONFIDENCES[c]: int((variant["conf"] == c).sum()) for c in range(3)},
            "n_disagree": int(variant["disagree"].sum()),
        }

    # Retrieval-gate pair set (A1d + A2): the DISAGREEMENT subset only.
    gate = edges["disagree"] & (edges["conf"] < CONFIDENCES.index("low"))
    cross_task = segments["task"][edges["src"]] != segments["task"][edges["dst"]]
    gate = gate & cross_task
    np.savez_compressed(
        out_dir / "gate_pairs.npz",
        src=edges["src"][gate],
        dst=edges["dst"][gate],
        kind=edges["kind"][gate],
        cosine=edges["cosine"][gate],
    )

    manifest = {
        "label_id": label_id,
        "knobs": knobs,
        "counts": counts,
        "n_buckets": n_buckets,
        "dropped": dropped,
        "n_segments": index["n"],
        "segments_per_domain": {d: int((segments["domain"] == i).sum()) for i, d in enumerate(DOMAINS)},
        "quota_A1c": quota_report(edges, segments),
        "gate_pairs": {
            "n": int(gate.sum()),
            "by_kind": {EDGE_KINDS[k]: int((edges["kind"][gate] == k).sum()) for k in range(3)},
            "definition": (
                "cross-task, non-low-confidence pairs where the Qwen verdict DISAGREES "
                "with descriptor-cosine ranking inside the anchor's bucket: a positive "
                "ranked in the bottom half, or a CONTRAST ranked in the top half"
            ),
        },
        "mean_accepted_positive_outdegree": round(float(degree.mean()), 3),
    }

    # ---- PER-DOMAIN NON-EMPTINESS (4-domain corpus, §42) ---------------------------------------
    # A domain can be present in the segment table yet contribute no edges and no gate pairs -- the
    # artifact then looks complete, Stage E loads it, and that domain silently shapes nothing. Both
    # the objective (edges) and the go/no-go (gate pairs) are asserted per domain.
    sd, dd = segments["domain"][edges["src"]], segments["domain"][edges["dst"]]
    positive_mask = edges["kind"] <= EDGE_KINDS.index("ANALOGOUS")
    per_dom = {}
    for i, d in enumerate(DOMAINS):
        touches = (sd == i) | (dd == i)
        per_dom[d] = {
            "segments": int((segments["domain"] == i).sum()),
            "edges": int(touches.sum()),
            "positives": int((touches & positive_mask).sum()),
            "gate_pairs": int(
                (gate & ((segments["domain"][edges["src"]] == i) | (segments["domain"][edges["dst"]] == i))).sum()
            ),
        }
    manifest["per_domain"] = per_dom
    allow_empty = {d.strip() for d in (args.allow_empty_domain or "").split(",") if d.strip()}
    starved = {
        d: v
        for d, v in per_dom.items()
        if v["segments"] > 0
        and d not in allow_empty
        and (v["edges"] == 0 or v["positives"] == 0 or v["gate_pairs"] == 0)
    }
    if starved:
        raise SystemExit(
            "[labels] FATAL: domains present in the segment table contribute no edges/positives/"
            f"gate pairs: {starved}. A domain that shapes neither the objective nor the go/no-go "
            "must not ship inside an artifact that claims to cover it "
            "(--allow-empty-domain to override deliberately)."
        )
    print("[labels] per-domain " + json.dumps(per_dom))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {k: manifest[k] for k in ("label_id", "counts", "quota_A1c", "gate_pairs", "segments_per_domain")},
            indent=1,
        )
    )
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
