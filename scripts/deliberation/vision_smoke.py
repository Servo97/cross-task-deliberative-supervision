"""H14 P0 gate D8 — vLLM VISION smoke for Qwen3.8-27B, plus the A6 reasoning_effort A/B.

s3 risk 1: the official vLLM recipe verifies TEXT serving only; the vision tower is present but
unvalidated. This runs REAL pass-1 geometry (3 RoboCasa frames x 3 views = 9 images per request,
through caption_segments' own decode + prompt builders) against a live server and answers, with
measurements rather than belief:

  V1  does a 9-image multimodal request return 200 at all?
  V2  does the model SEE? -- an ablation, not a vibe check: the same request is re-issued with the
      images replaced by pure-noise tiles. If descriptors are unchanged, the vision path is inert.
  V3  vision token accounting: measured prompt_tokens with/without images -> tokens per image.
      (s3 §3.2 costs the loop at 64 vision tokens per 256px view; this checks it.)
  A6  reasoning_effort low vs medium: 20 paired requests, measured completion_tokens. If the two
      are identical the knob is NOT plumbing through chat_template_kwargs and the whole cost model
      is unpinned.

  python scripts/deliberation/vision_smoke.py --n 10 --base-url http://127.0.0.1:8100/v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.labels.caption_segments import (  # noqa: E402
    DESCRIPTOR_SCHEMA,
    Job,
    VLLMChat,
    build_descriptor_messages,
    decode_views,
    extract_descriptor,
    hints_for,
    prompt_sha,
    read_label,
    resolve_lerobot_dir,
    schema_sha,
    segment_frames,
    segments_from_keyframes,
)
from workspace_models.labels.extract_frames import load_episode_meta  # noqa: E402
from workspace_models.labels.geometry import VIEWS  # noqa: E402

MEM16 = (
    "ScrubCuttingBoard",
    "KettleBoiling",
    "SearingMeat",
    "RinseSinkBasin",
    "GetToastedBread",
    "WashLettuce",
    "StirVegetables",
    "PackIdenticalLunches",
    "PanTransfer",
    "HeatKebabSandwich",
    "PortionHotDogs",
    "RecycleBottlesByType",
    "CuttingToolSelection",
    "SeparateFreezerRack",
    "GatherTableware",
    "CategorizeCondiments",
)


def build_smoke_jobs(n: int, labels_root: Path, dataset_root: Path, hints_root: Path) -> list[Job]:
    """One episode from each of the first n mem16 tasks -- real store, real geometry."""
    jobs: list[Job] = []
    for task in MEM16[:n]:
        root = resolve_lerobot_dir(dataset_root, task)
        if root is None:
            continue
        ep_meta, _ = load_episode_meta(root)
        labs = sorted((labels_root / task).glob("vlm_episode_pi_*.npz"))
        if not labs:
            continue
        lab = labs[0]
        ep = int(lab.stem.split("_")[-1])
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        kf, _ = read_label(lab)
        T = int(meta["length"])
        segs = segments_from_keyframes(kf, T)
        plan = [segment_frames(a, b, 3) for a, b in segs]
        jobs.append(
            Job(task, ep, T, kf, segs, plan, root, Path("/dev/null"), hints=hints_for(hints_root, task, ep, len(segs)))
        )
    return jobs


def noise_messages(msgs: list) -> list:
    """Same prompt, images -> deterministic noise tiles (the V2 ablation)."""
    from workspace_models.labels.caption_segments import _png_data_url

    rng = np.random.default_rng(0)
    out = json.loads(json.dumps(msgs))  # deep copy
    for part in out[1]["content"]:
        if part.get("type") == "image_url":
            part["image_url"]["url"] = _png_data_url(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
    return out


def strip_images(msgs: list) -> list:
    out = json.loads(json.dumps(msgs))
    out[1]["content"] = [p for p in out[1]["content"] if p.get("type") != "image_url"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    ap.add_argument("--n", type=int, default=10, help="images-bearing requests for the smoke")
    ap.add_argument("--ab-n", type=int, default=20, help="A6 low-vs-medium paired requests")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--labels-root", default="~/Research/TRI/wsm_data/wsm_labels_pi_mirror")
    ap.add_argument("--dataset-root", default="~/Research/robocasa/datasets/v1.0/target")
    ap.add_argument("--hints-root", default="~/Research/TRI/wsm_data/wsm_labels_captions")
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/vision_smoke.json")
    args = ap.parse_args()

    client = VLLMChat(args.base_url, args.model, timeout=900.0, retries=2)
    jobs = build_smoke_jobs(
        args.n,
        Path(args.labels_root).expanduser(),
        Path(args.dataset_root).expanduser(),
        Path(args.hints_root).expanduser(),
    )
    print(f"[smoke] {len(jobs)} episodes across {len({j.task for j in jobs})} tasks", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(decode_views, jobs))
    jobs = [j for j in jobs if not j.error]
    print(f"[smoke] {len(jobs)} decoded ok", flush=True)
    if not jobs:
        raise SystemExit("no episodes decoded -- cannot smoke")

    # one segment per episode (the middle one), i.e. exactly the pass-1 request geometry
    requests = [(j, len(j.segs) // 2) for j in jobs][: args.n]
    msgs = [build_descriptor_messages(j, si) for j, si in requests]
    n_img = sum(1 for p in msgs[0][1]["content"] if p.get("type") == "image_url")
    print(f"[smoke] images per request = {n_img}", flush=True)

    report: dict = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_sha_descriptor": prompt_sha("descriptor"),
        "schema_sha_descriptor": schema_sha("descriptor"),
        "images_per_request": n_img,
        "views": list(VIEWS),
        "max_tokens": args.max_tokens,
    }

    def one(m, effort="low", schema=True):
        t = time.time()
        text, usage, finish = client.chat(
            m,
            max_tokens=args.max_tokens,
            reasoning_effort=effort,
            json_schema=DESCRIPTOR_SCHEMA if schema else None,
            schema_name="segment_descriptor",
            seed=20260822,
        )
        return {"text": text, "usage": usage, "finish": finish, "sec": round(time.time() - t, 2)}

    # ---------------------------------------------------------------- V1 real images
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        real = list(pool.map(one, msgs))
    wall = time.time() - t0
    ok, bad = [], []
    for (job, si), r in zip(requests, real):
        try:
            ok.append(
                {
                    "task": job.task,
                    "episode": job.ep,
                    "segment": si,
                    "hint": job.hints[si] if job.hints else "",
                    "descriptor": extract_descriptor(r["text"]),
                    "usage": r["usage"],
                    "finish": r["finish"],
                }
            )
        except Exception as e:  # noqa: BLE001
            bad.append(
                {"task": job.task, "err": f"{type(e).__name__}: {e}", "raw": r["text"][:400], "finish": r["finish"]}
            )
    report["V1_real_images"] = {
        "n": len(real),
        "schema_valid": len(ok),
        "invalid": len(bad),
        "wall_s": round(wall, 2),
        "prompt_tokens_mean": round(statistics.mean([r["usage"].get("prompt_tokens", 0) for r in real]), 1),
        "completion_tokens_mean": round(statistics.mean([r["usage"].get("completion_tokens", 0) for r in real]), 1),
        "completion_tokens_max": max(r["usage"].get("completion_tokens", 0) for r in real),
        "truncated": sum(1 for r in real if r["finish"] == "length"),
        "latency_s_median": round(statistics.median(r["sec"] for r in real), 2),
        "throughput_seg_per_min": round(len(real) / wall * 60, 2),
        "out_tok_per_s": round(sum(r["usage"].get("completion_tokens", 0) for r in real) / wall, 1),
        "failures": bad[:3],
    }
    report["descriptors"] = ok

    # ---------------------------------------------------------------- V2 noise ablation
    nmsgs = [noise_messages(m) for m in msgs[: min(4, len(msgs))]]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        noisy = list(pool.map(one, nmsgs))
    same = 0
    noise_recs = []
    for i, r in enumerate(noisy):
        try:
            d = extract_descriptor(r["text"])
        except Exception:  # noqa: BLE001
            d = None
        noise_recs.append({"task": requests[i][0].task, "descriptor": d})
        if d and i < len(ok) and ok[i]["task"] == requests[i][0].task:
            if (
                d.get("target_object", {}).get("class", "").lower()
                == ok[i]["descriptor"]["target_object"]["class"].lower()
                and d.get("subskill") == ok[i]["descriptor"]["subskill"]
            ):
                same += 1
    report["V2_noise_ablation"] = {
        "n": len(noisy),
        "identical_subskill_and_object": same,
        "verdict": ("VISION INERT (identical on noise)" if same == len(noisy) and noisy else "vision is load-bearing"),
        "noise_descriptors": noise_recs,
    }

    # ---------------------------------------------------------------- V3 vision token accounting
    tmsgs = [strip_images(m) for m in msgs[:3]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        textonly = list(pool.map(lambda m: one(m, "low", True), tmsgs))
    p_img = statistics.mean(r["usage"].get("prompt_tokens", 0) for r in real[:3])
    p_txt = statistics.mean(r["usage"].get("prompt_tokens", 0) for r in textonly)
    report["V3_vision_tokens"] = {
        "prompt_tokens_with_images": round(p_img, 1),
        "prompt_tokens_text_only": round(p_txt, 1),
        "vision_tokens_total": round(p_img - p_txt, 1),
        "vision_tokens_per_image": round((p_img - p_txt) / n_img, 1),
        "s3_assumption_per_view": 64,
    }

    # ---------------------------------------------------------------- A6 low vs medium
    ab_msgs = (msgs * ((args.ab_n // max(len(msgs), 1)) + 1))[: args.ab_n]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        lo = list(pool.map(lambda m: one(m, "low"), ab_msgs))
        t_med = time.time()
        me = list(pool.map(lambda m: one(m, "medium"), ab_msgs))
        med_wall = time.time() - t_med
    lo_c = [r["usage"].get("completion_tokens", 0) for r in lo]
    me_c = [r["usage"].get("completion_tokens", 0) for r in me]
    try:
        xh = [one(msgs[0], "xhigh")]
        xh_c = xh[0]["usage"].get("completion_tokens", 0)
        xh_fin = xh[0]["finish"]
    except Exception as e:  # noqa: BLE001
        xh_c, xh_fin = -1, f"error: {e}"
    report["A6_reasoning_effort"] = {
        "n_pairs": args.ab_n,
        "low_completion_tokens": {
            "mean": round(statistics.mean(lo_c), 1),
            "median": statistics.median(lo_c),
            "max": max(lo_c),
        },
        "medium_completion_tokens": {
            "mean": round(statistics.mean(me_c), 1),
            "median": statistics.median(me_c),
            "max": max(me_c),
        },
        "xhigh_single_request_completion_tokens": xh_c,
        "xhigh_finish": xh_fin,
        "medium_over_low_ratio": round(statistics.mean(me_c) / max(statistics.mean(lo_c), 1e-9), 3),
        "identical": lo_c == me_c,
        "knob_works": (lo_c != me_c),
        "medium_truncated": sum(1 for r in me if r["finish"] == "length"),
        "low_truncated": sum(1 for r in lo if r["finish"] == "length"),
        "medium_wall_s": round(med_wall, 2),
    }

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    printable = {k: v for k, v in report.items() if k not in ("descriptors",)}
    printable["V2_noise_ablation"] = {k: v for k, v in report["V2_noise_ablation"].items() if k != "noise_descriptors"}
    print(json.dumps(printable, indent=1))
    print(f"\nwrote {out}")
    if ok:
        print("\n=== sample descriptor ===")
        print(json.dumps(ok[0], indent=1)[:2500])


if __name__ == "__main__":
    main()
