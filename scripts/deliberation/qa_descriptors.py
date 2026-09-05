"""H14 P0 — pass-1 descriptor QA: schema validation over the whole store + an eyeball sheet.

Two jobs, deliberately separate:

  --validate   EVERY descriptor in the store is re-parsed and re-checked against the frozen
               validator (not the schema the server enforced -- the point is to catch a store
               written by a DIFFERENT prompt/schema sha, which structured decoding cannot catch).
  --sheet      a self-contained HTML page: N segments, real frames side by side with the descriptor,
               stratified across tasks so the eyeball is not 20 views of one task.

  python scripts/deliberation/qa_descriptors.py \
      --store ~/Research/TRI/wsm_data/deliberation/descriptors/robocasa --sheet 20
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.labels.caption_segments import (  # noqa: E402
    Job,
    decode_views,
    memory_kinds_of,
    prompt_sha,
    resolve_lerobot_dir,
    schema_sha,
    segment_frames,
    validate_descriptor_record,
)
from workspace_models.labels.geometry import VIEWS  # noqa: E402

# Expected episode counts per domain, so "complete" is checked against a number rather than a
# feeling.
#
# RoboMME's expectation was 1,567 = 1,600 minus 33 "corrupt in the published snapshot" (status
# §12.3). That diagnosis was WRONG and the receipt says so: the corruption was local HuggingFace
# cache damage, the S3 mirror was clean, the cache was repaired and re-verified 1600/1600, and the
# 33 episodes were labelled on 2026-08-29 (§18.8). Carrying the subtraction forward would let a
# store that is 19 episodes short keep reporting `complete: true`, which is exactly what it did.
# The expectation is therefore the full published count.
EXPECTED_EPISODES = {"robocasa": 1950, "robomme": 1600, "remembench": 323, "robocerebra": 994}


def _coverage(store: Path, domains) -> dict:
    out = {}
    for dom, expected in EXPECTED_EPISODES.items():
        d = store / dom
        have = len(list(d.glob("*/ep_*.descriptors.json"))) if d.is_dir() else 0
        row = {
            "episodes_present": have,
            "episodes_expected": expected,
            "missing": max(expected - have, 0),
            "complete": have >= expected,
        }
        gap = d / "_robomme_unreadable.json"
        if gap.is_file():
            try:
                bad = json.loads(gap.read_text())
                row["corrupt_upstream_episodes"] = len(bad)
                row["corrupt_by_task"] = {}
                for b in bad:
                    row["corrupt_by_task"][b["task"]] = row["corrupt_by_task"].get(b["task"], 0) + 1
                row["note"] = (
                    "HISTORICAL RECORD ONLY — `_robomme_unreadable.json` lists what the "
                    "2026-08-23 sweep skipped and is deliberately left byte-unchanged. Those "
                    "episodes were NOT corrupt upstream (local HF cache damage; the S3 mirror was "
                    "clean), the cache was repaired, and all of them are labelled as of "
                    "2026-08-30. They are NOT excluded from `episodes_expected`."
                )
            except Exception:
                pass
        out[dom] = row
    tot_have = sum(v["episodes_present"] for v in out.values())
    tot_exp = sum(v["episodes_expected"] for v in out.values())
    out["TOTAL"] = {
        "episodes_present": tot_have,
        "episodes_expected": tot_exp,
        "missing": max(tot_exp - tot_have, 0),
        "complete": tot_have >= tot_exp,
    }
    return out


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def img_tag(arr, w: int = 176) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=82)
    b = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b}" width="{w}">'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/Research/TRI/wsm_data/deliberation/descriptors/robocasa")
    ap.add_argument("--dataset-root", default="~/Research/robocasa/datasets/v1.0/target")
    ap.add_argument("--sheet", type=int, default=20)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-html", default="")
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    store = Path(args.store).expanduser()
    # A single-domain store is <store>/<Task>/ep_*.json; the full-corpus store adds a domain level,
    # <store>/<domain>/<Task>/ep_*.json. Accept both so one QA pass can cover the whole corpus.
    files = sorted(store.glob("*/ep_*.descriptors.json"))
    if not files:
        files = sorted(store.glob("*/*/ep_*.descriptors.json"))
    if not files:
        raise SystemExit(f"no descriptor files under {store}")

    n_seg = n_bad = 0
    violations: Counter = Counter()
    subskills: Counter = Counter()
    mem_kinds: Counter = Counter()
    by_task: dict = {}
    tok_in: list[int] = []
    tok_out: list[int] = []
    truncated = 0
    multi_label = 0
    prompt_shas: Counter = Counter()
    models: Counter = Counter()
    domains: Counter = Counter()
    dom_views: dict = {}
    dom_sha: dict = {}
    schema_shas: Counter = Counter()
    lookalikes: list[int] = []
    pre_n: list[int] = []
    post_n: list[int] = []
    pool: list[tuple] = []

    for p in files:
        d = json.loads(p.read_text())
        prompt_shas[d.get("prompt_sha", "")] += 1
        schema_shas[d.get("schema_sha", "")] += 1
        models[d.get("model", "")] += 1
        # Derive from the path first: it is authoritative, and the shared RoboCasa/ReMemBench code
        # path stamped `domain: robocasa` on ReMemBench files until 2026-08-23. The field is only a
        # fallback for single-domain stores.
        rel = p.relative_to(store).parts
        dom = rel[0] if len(rel) >= 3 else (d.get("domain") or "robocasa")
        domains[dom] += 1
        dom_views[dom] = ",".join(d.get("views_used") or [])
        dom_sha[dom] = (d.get("prompt_sha") or "")[:12]
        task = d.get("task", p.parent.name)
        by_task.setdefault(task, {"segments": 0, "invalid": 0, "mem_dep": 0})
        for rec in d.get("descriptors", []):
            n_seg += 1
            by_task[task]["segments"] += 1
            desc = rec.get("descriptor") or {}
            why = validate_descriptor_record(desc)
            if why:
                n_bad += 1
                by_task[task]["invalid"] += 1
                violations[why.split(" ")[0] + " " + " ".join(why.split(" ")[1:2])] += 1
                continue
            subskills[str(desc["subskill"]).lower()] += 1
            ks = memory_kinds_of(desc)
            for k in ks:
                mem_kinds[k] += 1
            if ks != ["none"]:
                by_task[task]["mem_dep"] += 1
                if len(ks) > 1:
                    multi_label += 1
            lookalikes.append(len(desc["failure_lookalikes"]))
            pre_n.append(len(desc["preconditions"]))
            post_n.append(len(desc["postconditions"]))
            u = rec.get("usage") or {}
            if u.get("prompt_tokens"):
                tok_in.append(int(u["prompt_tokens"]))
            if u.get("completion_tokens"):
                tok_out.append(int(u["completion_tokens"]))
            if u.get("finish_reason") == "length":
                truncated += 1
            pool.append((task, int(d["episode_id"]), rec))

    report = {
        "store": str(store),
        "files": len(files),
        "segments": n_seg,
        "schema_valid": n_seg - n_bad,
        "invalid": n_bad,
        "schema_valid_rate": round((n_seg - n_bad) / max(n_seg, 1), 4),
        "violations": dict(violations.most_common(10)),
        "prompt_sha_matches_code": prompt_shas.most_common(1)[0][0] == prompt_sha("descriptor")
        if prompt_shas
        else False,
        "schema_sha_matches_code": schema_shas.most_common(1)[0][0] == schema_sha("descriptor")
        if schema_shas
        else False,
        "distinct_prompt_shas": len(prompt_shas),
        "distinct_schema_shas": len(schema_shas),
        # A store may legitimately be written by two venues at different quantizations (local NVFP4
        # vs p5 FP8 -- an H100 cannot run NVFP4 kernels). That is a provenance axis, so it is
        # REPORTED rather than allowed to pass silently.
        "model_histogram": dict(models.most_common()),
        "single_model_store": len(models) <= 1,
        "domain_histogram": dict(domains.most_common()),
        "coverage": _coverage(store, domains),
        "domain_geometry": {d: {"views": dom_views.get(d), "prompt_sha": dom_sha.get(d)} for d in sorted(domains)},
        "truncated": truncated,
        "tokens": {
            "prompt_mean": round(statistics.mean(tok_in), 1) if tok_in else 0,
            "completion_mean": round(statistics.mean(tok_out), 1) if tok_out else 0,
            "completion_p95": sorted(tok_out)[int(0.95 * (len(tok_out) - 1))] if tok_out else 0,
            "completion_max": max(tok_out) if tok_out else 0,
            "total_per_segment": round(
                (statistics.mean(tok_in) if tok_in else 0) + (statistics.mean(tok_out) if tok_out else 0), 1
            ),
        },
        "memory_dependency_kinds": dict(mem_kinds.most_common()),
        "multi_label_segments": multi_label,
        "multi_label_rate": round(multi_label / max(n_seg - n_bad, 1), 4),
        "memory_dependency_rate": round((n_seg - n_bad - mem_kinds.get("none", 0)) / max(n_seg - n_bad, 1), 4),
        "distinct_subskills": len(subskills),
        "top_subskills": dict(subskills.most_common(15)),
        "failure_lookalikes_mean": round(statistics.mean(lookalikes), 2) if lookalikes else 0,
        "preconditions_mean": round(statistics.mean(pre_n), 2) if pre_n else 0,
        "postconditions_mean": round(statistics.mean(post_n), 2) if post_n else 0,
        "per_task": by_task,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=1))
    print("\nper task: " + json.dumps(by_task))
    out_json = Path(args.out_json).expanduser() if args.out_json else store.parent / "qa_pass1.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_json}")

    if not args.sheet:
        return

    # -------------------------------------------------------------- stratified eyeball sheet
    rng = random.Random(args.seed)
    by_t: dict = {}
    for t, ep, rec in pool:
        by_t.setdefault(t, []).append((ep, rec))
    tasks = sorted(by_t)
    picks: list[tuple] = []
    i = 0
    while len(picks) < args.sheet and tasks:
        t = tasks[i % len(tasks)]
        if by_t[t]:
            ep, rec = by_t[t].pop(rng.randrange(len(by_t[t])))
            picks.append((t, ep, rec))
        i += 1
        if i > 10000:
            break

    dataset_root = Path(args.dataset_root).expanduser()
    rows = []
    for t, ep, rec in picks:
        root = resolve_lerobot_dir(dataset_root, t)
        frames_html = "<i>frames unavailable</i>"
        if root is not None:
            segs = [(int(rec["t0"]), int(rec["t1"]))]
            plan = [segment_frames(segs[0][0], segs[0][1], 3)]
            job = Job(t, ep, int(rec["t1"]), [], segs, plan, root, Path("/dev/null"))
            job = decode_views(job)
            if not job.error:
                cells = []
                for f in plan[0]:
                    for v in VIEWS:
                        cells.append(f"<figure>{img_tag(job.frames[v][f])}<figcaption>f{f} {v}</figcaption></figure>")
                frames_html = "".join(cells)
            else:
                frames_html = f"<i>{esc(job.error)}</i>"
        rows.append(f"""
<section>
  <h2>{esc(t)} — ep {ep}, segment {rec.get("segment")} (frames {rec["t0"]}..{rec["t1"] - 1})</h2>
  <div class="frames">{frames_html}</div>
  <pre>{esc(json.dumps(rec["descriptor"], indent=1))}</pre>
</section>""")

    html = f"""<title>H14 pass-1 descriptor eyeball sheet</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:1100px}}
 section{{border-top:1px solid #ccc;padding-top:1rem;margin-top:1.5rem}}
 h2{{font-size:15px;margin:0 0 .5rem}}
 .frames{{display:flex;flex-wrap:wrap;gap:.4rem}}
 figure{{margin:0}} figcaption{{font-size:11px;color:#666;text-align:center}}
 pre{{background:#f6f6f6;padding:.75rem;overflow-x:auto;font-size:12px}}
 table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:.25rem .5rem}}
</style>
<h1>H14 pass-1 descriptor eyeball sheet</h1>
<p>{len(picks)} segments, stratified round-robin across {len(by_t)} tasks (seed {args.seed}).
Store <code>{esc(store)}</code>. Schema-valid {report["schema_valid"]}/{report["segments"]}
({report["schema_valid_rate"]:.1%}); truncated {report["truncated"]};
memory-dependency rate {report["memory_dependency_rate"]:.1%}.</p>
{"".join(rows)}
"""
    out_html = Path(args.out_html).expanduser() if args.out_html else store.parent / "pass1_eyeball_sheet.html"
    out_html.write_text(html)
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
