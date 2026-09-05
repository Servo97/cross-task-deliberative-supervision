"""H14 pass 2 — bucketed cross-task deliberation driver (plan §3; amendments A1c, A6, A7, A9).

Stages, each separately runnable and separately resumable (A7: three resumable stages, not one
24 h chained job):

  index  descriptor stores (3 domains)      -> index/segments.jsonl
  embed  descriptor text                    -> embed/emb.npy + embed/ids.json
  mine   embeddings + STRATIFIED QUOTAS     -> mine/buckets.jsonl
  judge  one bucketed request per anchor    -> edges/<edge_store_id>/buckets/**.bucket.json
  qa     computable gates                   -> edges/<edge_store_id>/qa.json

Design points that are NOT free choices:

* Mining is stratified with pre-registered quotas (A1c). Pure top-k embedding mining makes
  `EQUIVALENT ⊂ embedding-nearest` BY CONSTRUCTION, so an encoder that merely reproduces descriptor
  cosine geometry would pass the retrieval gate -- H13's degeneracy in a new guise. Forced
  cross-task and cross-domain candidates are the only way the judge can disagree with cosine.
* The judge never sees task or domain names (`pass2_prompt.build_bucket_messages`), so the "same
  task ⇒ EQUIVALENT" shortcut that E1-ctrl-T exists to measure is not available to it.
* `stratum` is written driver-side, after the verdict, so quotas cannot leak into judgments.
* A bucket with `finish_reason == "length"` is INVALID even if it parses: a truncated bucket loses
  verdicts silently, and a lost verdict is indistinguishable downstream from UNRELATED (A6).

  python scripts/deliberation/pass2_deliberate.py --stage index --limit-anchors 200
  python scripts/deliberation/pass2_deliberate.py --stage embed --embed-device cpu
  python scripts/deliberation/pass2_deliberate.py --stage mine
  python scripts/deliberation/pass2_deliberate.py --stage judge --shard 0 --num-shards 1
  python scripts/deliberation/pass2_deliberate.py --stage qa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.deliberation import pass2_prompt as P2  # noqa: E402
from workspace_models.labels.caption_segments import (  # noqa: E402
    VLLMChat,
    memory_kinds_of,
    parse_json,
)

DEFAULT_STORE = "~/Research/TRI/wsm_data/deliberation"

# ------------------------------------------------------------------- frozen mining quotas (A1c §5)
QUOTAS = {"within_task": 3, "cross_task": 4, "cross_domain": 2, "mined_hard_neg": 3}
K_PER_BUCKET = sum(QUOTAS.values())  # 12
QUOTA_FLOORS = {"cross_task_or_domain_frac": 0.40, "cross_domain_frac": 0.15}
MINING_SEED = 20260822

# Tasks whose SOURCE segmentation strings carry no object binding. The RoboMME audit found
# PatternLock's per-step `simple_subgoal` degenerates to bare directions ("move right",
# "move backward-left") with no object at all -- the RLE gives usable BOUNDARIES but the language is
# not a usable hint, so a descriptor there rests on pixels alone. Marked, NOT dropped (coordinator,
# 2026-08-22): the mining quotas already protect against junk positives, and dropping the task would
# also drop the only imitation-suite anchors of its kind. Consumers that weight or filter edges can
# read the flag; nothing here silently excludes them.
LOW_CONFIDENCE_LANGUAGE_TASKS = frozenset({"PatternLock"})


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------ index
def embed_text_for(desc: dict) -> str:
    """The text the bucketing embedding sees.

    `failure_lookalikes` is EXCLUDED on purpose: it is used to MINE hard negatives, so folding it
    into the embedding would make the hard-negative stratum a function of the same vector it is
    supposed to be an independent probe of.
    """
    t = desc.get("target_object", {}) or {}
    kinds = "+".join(memory_kinds_of(desc))
    return " | ".join(
        [
            str(desc.get("subskill", "")),
            str(desc.get("verb_frame", "")),
            f"{t.get('class', '')} ({', '.join(t.get('attributes') or [])})",
            f"{t.get('state_before', '')} -> {t.get('state_after', '')}",
            str(desc.get("spatial_relation", "")),
            "pre: " + "; ".join(desc.get("preconditions") or []),
            "post: " + "; ".join(desc.get("postconditions") or []),
            "memory: " + kinds,
        ]
    )


def stage_index(args) -> None:
    store = Path(args.store).expanduser()
    out = store / "index" / "segments.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, n_files, n_bad = [], 0, 0
    requested, absent = [], []
    for domain, root in [
        ("robocasa", args.robocasa_descriptors),
        ("remembench", args.remembench_descriptors),
        ("robomme", args.robomme_descriptors),
        ("robocerebra", args.robocerebra_descriptors),
    ]:
        if not root:
            continue
        requested.append(domain)
        r = Path(root).expanduser()
        if not r.is_dir():
            # NOT a skip. A caller that passed --<domain>-descriptors is asserting that domain is
            # part of this corpus; silently continuing produced a 3-domain index, an exit 0, and a
            # delta mined against the wrong candidate pool. Recorded and raised below.
            print(f"[index] {domain}: {r} ABSENT", flush=True)
            absent.append(domain)
            continue
        for p in sorted(r.glob("*/ep_*.descriptors.json")):
            n_files += 1
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                n_bad += 1
                continue
            task = d.get("task") or p.parent.name
            ep = int(d.get("episode_id", -1))
            for rec in d.get("descriptors", []):
                desc = rec.get("descriptor") or {}
                if not desc:
                    continue
                si = int(rec.get("segment", 0))
                rows.append(
                    {
                        "seg_id": f"{domain}/{task}/{ep:06d}/{si}",
                        "domain": domain,
                        "task": task,
                        "episode": ep,
                        "segment": si,
                        "t0": int(rec.get("t0", 0)),
                        "t1": int(rec.get("t1", 0)),
                        "descriptor": desc,
                        "embed_text": embed_text_for(desc),
                        "low_confidence_language": task in LOW_CONFIDENCE_LANGUAGE_TASKS,
                    }
                )
    rows.sort(key=lambda r: r["seg_id"])
    if args.limit_anchors:
        rows = rows[: args.limit_anchors * 4]  # keep a candidate pool around the anchors
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tasks = sorted({r["task"] for r in rows})
    per_domain = {d: 0 for d in requested}
    for r in rows:
        per_domain[r["domain"]] = per_domain.get(r["domain"], 0) + 1
    print(
        json.dumps(
            {
                "segments": len(rows),
                "files": n_files,
                "unparsable": n_bad,
                "domains": sorted({r["domain"] for r in rows}),
                "n_tasks": len(tasks),
                "per_domain_segments": per_domain,
                "out": str(out),
            },
            indent=1,
        )
    )

    # ---- fail-closed on the silent-success class -------------------------------------------
    empty = [d for d in requested if per_domain.get(d, 0) == 0]
    if absent or empty:
        raise SystemExit(
            "[index] FATAL: every domain passed via --<domain>-descriptors must contribute "
            f"segments. absent roots={absent or 'none'}; zero-segment domains={empty or 'none'}. "
            "This is the failure mode that silently produced a 3-domain index and an exit 0."
        )
    if getattr(args, "expect_domain_segments", ""):
        want = {}
        for part in args.expect_domain_segments.split(","):
            if not part.strip():
                continue
            d, _, n = part.partition("=")
            want[d.strip()] = int(n)
        bad = {d: (per_domain.get(d, 0), n) for d, n in want.items() if per_domain.get(d, 0) != n}
        if bad:
            raise SystemExit(f"[index] FATAL: segment counts differ from --expect-domain-segments (got, want): {bad}")
        print(f"[index] per-domain counts match --expect-domain-segments: {want}", flush=True)


def load_index(store: Path) -> list[dict]:
    p = store / "index" / "segments.jsonl"
    if not p.is_file():
        raise SystemExit(f"missing {p}; run --stage index first")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------ embed
def stage_embed(args) -> None:
    """Real text embeddings (s3 §4 change 2: PaliGemma@max_len 48 cannot carry 1.5k-token texts)."""
    store = Path(args.store).expanduser()
    rows = load_index(store)
    texts = [r["embed_text"] for r in rows]
    outdir = store / "embed"
    outdir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.embed_model)
    mdl = AutoModel.from_pretrained(args.embed_model, dtype=torch.float32).to(args.embed_device)
    mdl.eval()
    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), args.embed_batch):
        batch = texts[i : i + args.embed_batch]
        enc = tok(batch, padding=True, truncation=True, max_length=args.embed_max_len, return_tensors="pt").to(
            args.embed_device
        )
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        # last-token pooling (the Qwen3-Embedding convention); mask-safe
        idx = enc["attention_mask"].sum(dim=1) - 1
        v = h[torch.arange(h.size(0), device=h.device), idx]
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        vecs.append(v.cpu().numpy())
        if (i // args.embed_batch) % 20 == 0:
            print(f"[embed] {i + len(batch)}/{len(texts)}", flush=True)
    emb = np.concatenate(vecs, axis=0).astype(np.float32)
    np.save(outdir / "emb.npy", emb)
    (outdir / "ids.json").write_text(json.dumps([r["seg_id"] for r in rows]))
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "embed_model": args.embed_model,
                "dim": int(emb.shape[1]),
                "n": int(emb.shape[0]),
                "max_len": args.embed_max_len,
                "pooling": "last_token_l2",
                "wall_s": round(time.time() - t0, 1),
                "text_sha256": canonical_sha(texts[:1000]),
            },
            indent=1,
        )
    )
    print(
        json.dumps({"n": int(emb.shape[0]), "dim": int(emb.shape[1]), "wall_s": round(time.time() - t0, 1)}, indent=1)
    )


# ------------------------------------------------------------------------------------------- mine
def _postconditions_differ(a: dict, b: dict) -> bool:
    """Jaccard over postcondition token sets -- the CONTRAST signal the hard-neg stratum mines."""

    def toks(d):
        return {w.lower() for s in (d.get("postconditions") or []) for w in s.split()}

    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return True
    return len(ta & tb) / len(ta | tb) < 0.34


def stage_mine(args) -> None:
    store = Path(args.store).expanduser()
    rows = load_index(store)
    emb = np.load(store / "embed" / "emb.npy")
    ids = json.loads((store / "embed" / "ids.json").read_text())
    if ids != [r["seg_id"] for r in rows]:
        raise SystemExit("embed/ids.json does not match index/segments.jsonl -- re-run embed")
    n = len(rows)
    task = np.array([r["task"] for r in rows])
    domain = np.array([r["domain"] for r in rows])
    episode = np.array([r["episode"] for r in rows])

    anchors = list(range(n))
    if getattr(args, "anchor_allowlist", ""):
        # DELTA MINING (top-up closure): mine buckets for an explicit anchor set only, while the
        # CANDIDATE pool stays the whole corpus. The frozen store's anchors keep their frozen
        # buckets -- nothing here re-mines or re-judges them. A new anchor can never duplicate a
        # frozen pair, because a frozen bucket cannot contain a segment that did not yet exist.
        wanted = json.loads(Path(args.anchor_allowlist).expanduser().read_text())
        wanted = set(wanted["anchors"] if isinstance(wanted, dict) else wanted)
        pos = {sid: i for i, sid in enumerate(ids)}
        missing = sorted(wanted - set(pos))
        if missing:
            raise SystemExit(f"{len(missing)} allowlisted anchors absent from the index, e.g. {missing[:3]}")
        anchors = sorted(pos[s] for s in wanted)
        print(f"[mine] anchor allowlist: {len(anchors)} anchors, candidate pool {n}", flush=True)
    elif getattr(args, "anchor_domains", ""):
        # DELTA MINING BY DOMAIN -- the same contract as --anchor-allowlist (candidates stay the
        # WHOLE corpus; frozen anchors keep their frozen buckets), but expressed declaratively so
        # a cluster job needs no seg_id file. The seg_ids only exist after the on-node index runs,
        # so shipping a list would mean generating an artifact from an artifact that does not yet
        # exist; a domain name is knowable at submit time.
        want_dom = {d.strip() for d in args.anchor_domains.split(",") if d.strip()}
        anchors = [i for i in anchors if domain[i] in want_dom]
        if not anchors:
            raise SystemExit(
                f"--anchor-domains {sorted(want_dom)} selected 0 anchors from the "
                f"index; is that domain's pass-1 store staged?"
            )
        print(f"[mine] anchor domains {sorted(want_dom)}: {len(anchors)} anchors, candidate pool {n}", flush=True)
    elif args.limit_anchors:
        rng = random.Random(MINING_SEED)
        anchors = sorted(rng.sample(anchors, min(args.limit_anchors, n)))

    seen_pairs: set = set()
    buckets = []
    t0 = time.time()
    for ai in anchors:
        sims = emb @ emb[ai]
        sims[ai] = -2.0
        same_task = task == task[ai]
        same_dom = domain == domain[ai]
        same_ep = (episode == episode[ai]) & same_task
        sims[same_ep] = -2.0  # a neighbouring segment of the SAME episode is not a cross-task edge

        picked: list[tuple[int, str]] = []
        used: set = set()

        def take(mask, k, stratum):
            cand = np.where(mask)[0]
            if cand.size == 0:
                return
            order = cand[np.argsort(-sims[cand])]
            got = 0
            for j in order:
                j = int(j)
                if j in used or sims[j] <= -1.0:
                    continue
                key = frozenset((ids[ai], ids[j]))
                if key in seen_pairs:
                    continue
                used.add(j)
                seen_pairs.add(key)
                picked.append((j, stratum))
                got += 1
                if got >= k:
                    return

        take(same_task, QUOTAS["within_task"], "within_task")
        take((~same_task) & same_dom, QUOTAS["cross_task"], "cross_task")
        take(~same_dom, QUOTAS["cross_domain"], "cross_domain")
        # hard negatives: high cosine but a DIFFERENT completion condition
        hn_mask = np.zeros(n, dtype=bool)
        top = np.argsort(-sims)[: args.hard_neg_pool]
        for j in top:
            j = int(j)
            if j in used or sims[j] <= -1.0:
                continue
            if _postconditions_differ(rows[ai]["descriptor"], rows[j]["descriptor"]):
                hn_mask[j] = True
        take(hn_mask, QUOTAS["mined_hard_neg"], "mined_hard_neg")

        if not picked:
            continue
        # order shuffled by blake2b(anchor, seed) so stratum never correlates with position
        h = hashlib.blake2b(f"{ids[ai]}|{args.order_seed}".encode(), digest_size=8).digest()
        rng = random.Random(int.from_bytes(h, "big"))
        rng.shuffle(picked)
        buckets.append(
            {
                "anchor": ids[ai],
                "candidates": [ids[j] for j, _ in picked],
                "strata": [s for _, s in picked],
                "cosines": [round(float(sims[j]), 4) for j, _ in picked],
                "order_seed": args.order_seed,
            }
        )

    out = store / "mine" / "buckets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for b in buckets:
            f.write(json.dumps(b) + "\n")
    hist: dict = {}
    for b in buckets:
        for s in b["strata"]:
            hist[s] = hist.get(s, 0) + 1
    print(
        json.dumps(
            {
                "anchors": len(buckets),
                "pairs": sum(len(b["candidates"]) for b in buckets),
                "candidates_per_bucket_mean": round(
                    sum(len(b["candidates"]) for b in buckets) / max(len(buckets), 1), 2
                ),
                "stratum_histogram": hist,
                "quotas": QUOTAS,
                "wall_s": round(time.time() - t0, 1),
                "out": str(out),
            },
            indent=1,
        )
    )


# ------------------------------------------------------------------------------------------ judge
def edge_store_id(args, corpus_sha: str) -> str:
    return canonical_sha(
        {
            "prompt_sha": P2.prompt_sha(),
            "schema_sha": P2.schema_sha(),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "k_per_bucket": K_PER_BUCKET,
            "quotas": QUOTAS,
            "mining_seed": MINING_SEED,
            "order_seed": args.order_seed,
            "corpus_manifest_sha": corpus_sha,
        }
    )


def bucket_path(edges_root: Path, anchor: str) -> Path:
    dom, task, ep, si = anchor.split("/")
    return edges_root / "buckets" / dom / task / f"{ep}_{si}.bucket.json"


def validate_bucket_file(path: Path, candidates: list[str]) -> bool:
    """Structural resume gate (A7). Mirrors caption_segments.validate_existing in spirit:
    re-parse and shape-check; NEVER trust existence."""
    try:
        d = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    if d.get("candidates") != candidates:
        return False
    if (d.get("usage") or {}).get("finish_reason") == "length":
        return False  # a truncated bucket lost verdicts -- A6
    vs = d.get("verdicts")
    if not isinstance(vs, list) or len(vs) != len(candidates):
        return False
    for i, v in enumerate(vs):
        if P2.validate_verdict(v, i):
            return False
        if v.get("candidate_id") != candidates[i]:
            return False
    return True


def stage_judge(args) -> None:
    store = Path(args.store).expanduser()
    rows = {r["seg_id"]: r for r in load_index(store)}
    bl = [json.loads(x) for x in (store / "mine" / "buckets.jsonl").read_text().splitlines() if x]
    corpus_sha = canonical_sha(sorted(rows))
    esid = edge_store_id(args, corpus_sha)
    edges_root = store / "edges" / esid
    (edges_root / "buckets").mkdir(parents=True, exist_ok=True)

    # Zero buckets GLOBALLY means mining produced nothing: every shard would print "0 to do" and
    # exit 0, and the job would report success with an empty edge store. Fail closed. A single
    # shard legitimately gets 0 only when there are fewer buckets than shards.
    if not bl:
        raise SystemExit(
            f"[judge] FATAL: {store / 'mine' / 'buckets.jsonl'} contains ZERO buckets — nothing to "
            "judge. Mining produced no work; a silent exit 0 here would ship an empty edge store."
        )
    n_global = len(bl)
    bl = bl[args.shard :: args.num_shards]
    if not bl and n_global >= args.num_shards:
        raise SystemExit(
            f"[judge] FATAL: shard {args.shard}/{args.num_shards} got 0 of {n_global} buckets; "
            "the stride partition is broken."
        )
    todo = []
    resumed = 0
    for b in bl:
        p = bucket_path(edges_root, b["anchor"])
        if not args.force and p.exists() and validate_bucket_file(p, b["candidates"]):
            resumed += 1
            continue
        todo.append(b)
    print(
        f"[judge] shard {args.shard}/{args.num_shards}: {len(todo)} to do, "
        f"{resumed} already valid, edge_store_id={esid[:16]}",
        flush=True,
    )
    if args.dry_run:
        if todo:
            b = todo[0]
            msgs = P2.build_bucket_messages(
                rows[b["anchor"]]["descriptor"], [rows[c]["descriptor"] for c in b["candidates"]]
            )
            print("=== DRY RUN: bucket 0 request ===")
            print(msgs[1]["content"][:4000])
            print(f"... system {len(P2.SYSTEM)} chars, user {len(msgs[1]['content'])} chars")
        return
    if not todo:
        return

    client = VLLMChat(args.vllm_base_url, args.model, timeout=args.request_timeout, retries=3)
    stats = {"ok": 0, "fail": 0, "tok_in": 0, "tok_out": 0, "truncated": 0}
    lock = __import__("threading").Lock()
    t0 = time.time()

    def do(b):
        anchor, cands = b["anchor"], b["candidates"]
        msgs = P2.build_bucket_messages(rows[anchor]["descriptor"], [rows[c]["descriptor"] for c in cands])
        for attempt in range(args.retries + 1):
            try:
                text, usage, finish = client.chat(
                    msgs,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    temperature=0.0 if attempt == 0 else 0.4,
                    json_schema=P2.VERDICT_SCHEMA if args.structured_output else None,
                    schema_name="bucket_verdicts",
                    seed=args.order_seed,
                )
                usage = dict(usage)
                usage["finish_reason"] = finish
                if finish == "length":
                    raise ValueError("truncated bucket (finish_reason=length)")
                vs = parse_json(text).get("verdicts")
                if not isinstance(vs, list) or len(vs) != len(cands):
                    raise ValueError(
                        f"expected {len(cands)} verdicts, got {len(vs) if isinstance(vs, list) else type(vs).__name__}"
                    )
                out = []
                for i, v in enumerate(vs):
                    why = P2.validate_verdict(v, i)
                    if why:
                        raise ValueError(f"verdict {i}: {why}")
                    out.append(
                        {
                            **v,
                            "candidate_id": cands[i],
                            "anchor_id": anchor,
                            "stratum": b["strata"][i],
                            "cosine": b["cosines"][i],
                            "low_confidence_language": bool(
                                rows[anchor].get("low_confidence_language")
                                or rows[cands[i]].get("low_confidence_language")
                            ),
                        }
                    )
                rec = {
                    "anchor": anchor,
                    "candidates": cands,
                    "verdicts": out,
                    "usage": usage,
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "prompt_sha": P2.prompt_sha(),
                    "schema_sha": P2.schema_sha(),
                    "candidate_order_seed": b["order_seed"],
                }
                p = bucket_path(edges_root, anchor)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec, indent=1))
                tmp.replace(p)
                with lock:
                    stats["ok"] += 1
                    stats["tok_in"] += int(usage.get("prompt_tokens", 0) or 0)
                    stats["tok_out"] += int(usage.get("completion_tokens", 0) or 0)
                    stats["truncated"] += int(finish == "length")
                return
            except Exception as e:  # noqa: BLE001
                if attempt == args.retries:
                    print(f"[judge] {anchor} FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
                    with lock:
                        stats["fail"] += 1

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(do, todo))
    el = max(time.time() - t0, 1e-6)
    summary = {
        "edge_store_id": esid,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "buckets_ok": stats["ok"],
        "buckets_failed": stats["fail"],
        "truncated": stats["truncated"],
        "wall_s": round(el, 1),
        "prompt_tokens": stats["tok_in"],
        "completion_tokens": stats["tok_out"],
        "tokens_in_per_anchor": round(stats["tok_in"] / max(stats["ok"], 1), 1),
        "tokens_out_per_anchor": round(stats["tok_out"] / max(stats["ok"], 1), 1),
        "anchors_per_min": round(stats["ok"] / el * 60, 2),
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    (edges_root / "_provenance").mkdir(parents=True, exist_ok=True)
    (edges_root / "_provenance" / f"judge_shard{args.shard}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "prompt_sha": P2.prompt_sha(),
                "schema_sha": P2.schema_sha(),
                "quotas": QUOTAS,
                "args": vars(args),
            },
            indent=1,
        )
    )


# --------------------------------------------------------------------------------------------- qa
def _auc(scores: list[float], labels: list[int]) -> float:
    """Rank AUC of `scores` at separating label 1 from label 0 (ties get mid-ranks)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rp = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def stage_qa(args) -> None:
    store = Path(args.store).expanduser()
    rows = {r["seg_id"]: r for r in load_index(store)}
    corpus_sha = canonical_sha(sorted(rows))
    esid = args.edge_store_id or edge_store_id(args, corpus_sha)
    edges_root = store / "edges" / esid
    files = sorted((edges_root / "buckets").rglob("*.bucket.json"))
    if not files:
        raise SystemExit(f"no buckets under {edges_root}")

    verdicts = []
    for p in files:
        d = json.loads(p.read_text())
        verdicts.extend(d["verdicts"])

    by_type: dict = {}
    by_stratum: dict = {}
    for v in verdicts:
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
        by_stratum.setdefault(v["stratum"], {})
        by_stratum[v["stratum"]][v["type"]] = by_stratum[v["stratum"]].get(v["type"], 0) + 1

    pos = [v for v in verdicts if v["type"] in ("EQUIVALENT", "ANALOGOUS")]
    n_pos = len(pos)
    cross_td = sum(1 for v in pos if v["stratum"] in ("cross_task", "cross_domain"))
    cross_d = sum(1 for v in pos if v["stratum"] == "cross_domain")

    # ---- G-D (A1a): can raw descriptor cosine already predict EQUIVALENT vs CONTRAST?
    ec = [v for v in verdicts if v["type"] in ("EQUIVALENT", "CONTRAST")]
    auc = _auc([v["cosine"] for v in ec], [1 if v["type"] == "EQUIVALENT" else 0 for v in ec])

    # every task must contribute >= 1 cross-task EQUIVALENT edge (plan §3 QA c)
    tasks = sorted({r["task"] for r in rows.values()})
    contributing = set()
    for v in verdicts:
        if v["type"] == "EQUIVALENT" and v["stratum"] in ("cross_task", "cross_domain"):
            contributing.add(rows[v["anchor_id"]]["task"])
            contributing.add(rows[v["candidate_id"]]["task"])

    qa = {
        "edge_store_id": esid,
        "n_buckets": len(files),
        "n_verdicts": len(verdicts),
        "type_histogram": by_type,
        "type_by_stratum": by_stratum,
        "confidence_histogram": {c: sum(1 for v in verdicts if v["confidence"] == c) for c in P2.CONFIDENCES},
        "G_E_quota_floors": {
            "positives": n_pos,
            "cross_task_or_domain_frac": round(cross_td / max(n_pos, 1), 4),
            "floor": QUOTA_FLOORS["cross_task_or_domain_frac"],
            "cross_domain_frac": round(cross_d / max(n_pos, 1), 4),
            "cross_domain_floor": QUOTA_FLOORS["cross_domain_frac"],
            "PASS": (cross_td / max(n_pos, 1) >= QUOTA_FLOORS["cross_task_or_domain_frac"]),
        },
        "G_D_A1a_cosine_auc": {
            "n_equivalent_vs_contrast": len(ec),
            "auc": None if auc != auc else round(auc, 4),  # NaN-safe
            "hold_threshold": 0.90,
            "VERDICT": (
                "INSUFFICIENT DATA"
                if auc != auc
                else (
                    "HOLD -- deliberation adds nothing over embedding"
                    if auc >= 0.90
                    else "PROCEED -- Qwen disagrees with cosine enough to be informative"
                )
            ),
        },
        "coverage": {
            "n_tasks": len(tasks),
            "tasks_with_cross_task_EQUIVALENT": len(contributing & set(tasks)),
            "isolated_tasks": sorted(set(tasks) - contributing),
        },
        "low_confidence_language": {
            "tasks": sorted(LOW_CONFIDENCE_LANGUAGE_TASKS),
            "verdicts_touching_a_flagged_segment": sum(1 for v in verdicts if v.get("low_confidence_language")),
            "note": "flagged, never silently excluded; a consumer may down-weight these edges",
        },
        "wilson_95_EQUIVALENT_rate": [round(x, 4) for x in _wilson(by_type.get("EQUIVALENT", 0), len(verdicts))],
    }
    (edges_root / "qa.json").write_text(json.dumps(qa, indent=1))
    print(json.dumps(qa, indent=1))


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("index", "embed", "mine", "judge", "qa", "shas"))
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--robocasa-descriptors", default="~/Research/TRI/wsm_data/deliberation/descriptors/robocasa")
    ap.add_argument("--remembench-descriptors", default="")
    ap.add_argument("--robomme-descriptors", default="")
    ap.add_argument("--robocerebra-descriptors", default="")
    ap.add_argument(
        "--expect-domain-segments",
        default="",
        help="index: 'dom=N,dom=N' exact per-domain segment counts to assert after "
        "building the index; a mismatch fails the stage",
    )
    ap.add_argument("--limit-anchors", type=int, default=0)
    # embed
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--embed-max-len", type=int, default=512)
    # mine
    ap.add_argument("--hard-neg-pool", type=int, default=64)
    ap.add_argument(
        "--anchor-allowlist",
        default="",
        help="mine: JSON list (or {'anchors': [...]}) of seg_ids to mine buckets FOR. "
        "Candidates still come from the whole corpus. Used to close a pass-1 "
        "top-up without re-mining the frozen anchors.",
    )
    ap.add_argument(
        "--anchor-domains",
        default="",
        help="mine: restrict ANCHORS to these domains (comma-separated); candidates "
        "still come from the whole corpus. The declarative form of "
        "--anchor-allowlist, for delta mining a newly added domain.",
    )
    ap.add_argument("--order-seed", type=int, default=20260822)
    # judge
    ap.add_argument("--vllm-base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    ap.add_argument("--reasoning-effort", default="medium", choices=("low", "medium", "xhigh", "off"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--structured-output", dest="structured_output", action="store_true", default=True)
    ap.add_argument("--no-structured-output", dest="structured_output", action="store_false")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="judge: build + print request 0, issue nothing")
    ap.add_argument("--edge-store-id", default="")
    args = ap.parse_args()
    if args.reasoning_effort == "off":
        args.reasoning_effort = None

    {
        "index": stage_index,
        "embed": stage_embed,
        "mine": stage_mine,
        "judge": stage_judge,
        "qa": stage_qa,
        "shas": lambda a: print(
            json.dumps(
                {
                    "pass2_prompt_sha": P2.prompt_sha(),
                    "pass2_schema_sha": P2.schema_sha(),
                    "quotas": QUOTAS,
                    "k_per_bucket": K_PER_BUCKET,
                    "quota_floors": QUOTA_FLOORS,
                    "mining_seed": MINING_SEED,
                },
                indent=1,
            )
        ),
    }[args.stage](args)


if __name__ == "__main__":
    main()
