"""H14 P0 — is pass 1 actually READING THE PIXELS, or re-writing its own text prior?

The first noise ablation (`vision_smoke.py` V2) was confounded: the production prompt carries the
TASK NAME and the prior 30-token caption as a hint, and 3 of 4 requests reproduced the same
subskill+object class from PURE NOISE images. That does not show the vision tower is inert -- it
shows the text prior alone is sufficient for those two coarse fields.

This matters beyond the smoke. If the task name plus the existing caption determine the descriptor,
pass 1 is largely paraphrasing the caption store, and the "cross-task deliberative structure" would
be built on text we already had -- an A1-class circularity one stage earlier than the panel found it.

Four conditions on the SAME segments, 2x2 (images x text prior):

  A  real images  + task name + caption hint   <- the production prompt
  B  real images  + NO task   + NO hint        <- can vision alone carry it?
  C  noise images + task name + caption hint   <- can the text prior alone carry it?
  D  noise images + NO task   + NO hint        <- the floor: nothing to go on

Reported per field: agreement with A. D is the discriminating control -- if D also agrees, the
metric cannot tell conditions apart and nothing here is evidence.

  python scripts/deliberation/prior_ablation.py --n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.deliberation.vision_smoke import build_smoke_jobs  # noqa: E402
from workspace_models.labels.caption_segments import (  # noqa: E402
    CAPTION,
    DESCRIPTOR_SCHEMA,
    DESCRIPTOR_SYSTEM,
    VLLMChat,
    _png_data_url,
    decode_views,
    extract_descriptor,
    memory_kinds_of,
)
from workspace_models.labels.geometry import VIEWS  # noqa: E402


def messages(job, si: int, *, real_images: bool, text_prior: bool, rng) -> list:
    t0, t1 = job.segs[si]
    caps = [CAPTION[v] for v in VIEWS]
    if text_prior:
        hint = job.hints[si] if job.hints and si < len(job.hints) else ""
        prior = f"\nA prior short label for this segment (a hint, may be wrong): {hint}" if hint else ""
        head = (
            f"Task: {job.task}\n"
            f"This is segment {si + 1} of {len(job.segs)} in the episode "
            f"(frames {t0}..{t1 - 1} of {job.n_frames}).{prior}\n"
            f"Frames below are start / middle / end of THIS segment, each in "
            f"{', '.join(caps)} views."
        )
    else:
        head = (
            f"This is segment {si + 1} of {len(job.segs)} of a robot-manipulation episode "
            f"(frames {t0}..{t1 - 1} of {job.n_frames}).\n"
            f"Frames below are start / middle / end of THIS segment, each in "
            f"{', '.join(caps)} views."
        )
    content = [{"type": "text", "text": head}]
    for f in job.plan[si]:
        for j, view in enumerate(VIEWS):
            tag = f"\nframe {f} {caps[j]}:" if j == 0 else f" {caps[j]}:"
            content.append({"type": "text", "text": tag})
            arr = job.frames[view][f] if real_images else rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
            content.append({"type": "image_url", "image_url": {"url": _png_data_url(arr)}})
    return [{"role": "system", "content": DESCRIPTOR_SYSTEM}, {"role": "user", "content": content}]


def norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="~/Research/TRI/wsm_data/deliberation/prior_ablation.json")
    args = ap.parse_args()

    jobs = build_smoke_jobs(
        args.n,
        Path("~/Research/TRI/wsm_data/wsm_labels_pi_mirror").expanduser(),
        Path("~/Research/robocasa/datasets/v1.0/target").expanduser(),
        Path("~/Research/TRI/wsm_data/wsm_labels_captions").expanduser(),
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [j for j in pool.map(decode_views, jobs) if not j.error]
    reqs = [(j, len(j.segs) // 2) for j in jobs][: args.n]
    print(f"[ablation] {len(reqs)} segments x 4 conditions", flush=True)

    client = VLLMChat(args.base_url, args.model, timeout=900.0, retries=2)
    conds = {
        "A_img_prior": (True, True),
        "B_img_only": (True, False),
        "C_prior_only": (False, True),
        "D_neither": (False, False),
    }

    results: dict = {}
    for name, (img, prior) in conds.items():
        rng = np.random.default_rng(0)
        msgs = [messages(j, si, real_images=img, text_prior=prior, rng=rng) for j, si in reqs]

        def one(m):
            try:
                text, usage, finish = client.chat(
                    m,
                    max_tokens=args.max_tokens,
                    reasoning_effort="low",
                    json_schema=DESCRIPTOR_SCHEMA,
                    schema_name="segment_descriptor",
                    seed=20260822,
                )
                return extract_descriptor(text)
            except Exception as e:  # noqa: BLE001
                return {"_error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results[name] = list(pool.map(one, msgs))
        print(f"[ablation] {name} done", flush=True)

    base = results["A_img_prior"]
    report: dict = {
        "n": len(reqs),
        "segments": [f"{j.task}/ep{j.ep}/seg{si}" for j, si in reqs],
        "agreement_with_A": {},
    }
    for name in conds:
        if name == "A_img_prior":
            continue
        agree = {"subskill": 0, "object_class": 0, "both": 0, "memory_kind": 0, "n_ok": 0}
        for a, b in zip(base, results[name]):
            if "_error" in a or "_error" in b:
                continue
            agree["n_ok"] += 1
            s = norm(a["subskill"]) == norm(b["subskill"])
            o = norm(a["target_object"]["class"]) == norm(b["target_object"]["class"])
            agree["subskill"] += int(s)
            agree["object_class"] += int(o)
            agree["both"] += int(s and o)
            agree["memory_kind"] += int(set(memory_kinds_of(a)) == set(memory_kinds_of(b)))
        n = max(agree["n_ok"], 1)
        report["agreement_with_A"][name] = {
            **agree,
            "subskill_frac": round(agree["subskill"] / n, 3),
            "object_class_frac": round(agree["object_class"] / n, 3),
            "both_frac": round(agree["both"] / n, 3),
            "memory_kind_frac": round(agree["memory_kind"] / n, 3),
        }
    report["per_segment"] = [
        {
            "segment": report["segments"][i],
            **{
                name: (
                    {"err": results[name][i]["_error"]}
                    if "_error" in results[name][i]
                    else {
                        "subskill": results[name][i]["subskill"],
                        "object": results[name][i]["target_object"]["class"],
                        "memory_kind": "+".join(memory_kinds_of(results[name][i])),
                    }
                )
                for name in conds
            },
        }
        for i in range(len(reqs))
    ]

    b = report["agreement_with_A"]["B_img_only"]["both_frac"]
    c = report["agreement_with_A"]["C_prior_only"]["both_frac"]
    d = report["agreement_with_A"]["D_neither"]["both_frac"]
    report["reading"] = {
        "control_ok": d < min(b, c),
        "vision_sufficient": b,
        "text_prior_sufficient": c,
        "floor": d,
        "verdict": (
            "METRIC CANNOT DISCRIMINATE (floor agrees as much as the informative conditions)"
            if d >= min(b, c)
            else "text prior alone reproduces the coarse fields -- descriptors add detail, not identity"
            if c >= 0.75 and c >= b
            else "vision carries the descriptor"
            if b >= 0.75
            else "both signals contribute; neither alone reproduces A"
        ),
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_segment"}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
