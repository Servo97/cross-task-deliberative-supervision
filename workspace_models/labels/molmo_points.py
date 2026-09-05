"""WSM label stage B: MolmoPoint-8B points for each (keyframe x view x salient object).

RoboCasa365 3-view port of Isaac-GR00T/wsm/vlm_label/molmo_points.py (DexJoCo front+wrist).
Reads ep*_subgoals.json (stage A) + ep*_frames.npz, queries `Point to the <object>` per object
on each of the 3 views of each completion keyframe, and decodes the pointing tokens to pixel
coordinates in the fed (256x256) frame (RENDER_WH in geometry.py).

NOTE: RoboCasa frames are native 256 (smaller than DexJoCo's 640); if pointing accuracy is poor
on the agentviews, revisit with an upsample knob (feed a larger image, set RENDER_WH to match).

The VIEW SET is injectable via `--geom` (default `groot` = the 3 RoboCasa views); everything else in
this stage is view-agnostic. `pi_libero` = the 2 RoboCerebra/LIBERO views.

Run in the vlm_labeler venv (transformers==4.57.x, trust_remote_code):
  ~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.molmo_points \
      --task <Task> --in ~/Research/TRI/wsm_data/wsm_vlm_rc_v0 --device cuda:1

Output per episode: <in>/<task>/ep{idx:03d}_points.json
  {"keyframes": [{"frame": int, "subgoal": str,
                  "views": {view: {object: [[x,y], ...]} for view in VIEWS}}], "model": ...}
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ID = "allenai/MolmoPoint-8B"
GEOM_MODULES = {"groot": "geometry", "pi": "pi_geometry", "pi_libero": "pi_geometry_libero"}


def _patch_molmo_batching() -> int:
    """Fix a batching bug in MolmoPoint's vendored remote code: it uses Python ``and`` on two
    multi-element tensors (``should_embed = (input_patch_ids >= 0) and (input_patch_ids < ...)``),
    which works at batch=1 (1-element tensor is truthy-evaluable) but raises "Boolean value of
    Tensor with more than one value is ambiguous" at batch>1. The result is used as a tensor mask
    one line down, so the correct operator is elementwise ``&``.

    Idempotent; patches every cached copy (hub snapshot + dynamic-module tree) so it survives HF
    re-copying the trust_remote_code file. Local fix for an upstream bug — call before loading.
    """
    import glob
    import pathlib

    bad = "(input_patch_ids >= 0) and (input_patch_ids < (bounds.patch_end-1))"
    good = "(input_patch_ids >= 0) & (input_patch_ids < (bounds.patch_end-1))"
    n = 0
    root = pathlib.Path.home() / ".cache" / "huggingface"
    for f in glob.glob(str(root / "**" / "modeling_molmo_point.py"), recursive=True):
        p = pathlib.Path(f)
        t = p.read_text()
        if bad in t:
            p.write_text(t.replace(bad, good))
            n += 1
    if n:
        print(f"[molmo] patched batched-pointing bug (and->&) in {n} cached file(s)", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument(
        "--tag",
        default="",
        help="label-spec suffix: reads ep*_subgoals<tag>.json, writes ep*_points<tag>.json "
        "(e.g. '_causal_v1'). Empty = the original salient artifacts.",
    )
    ap.add_argument(
        "--geom",
        choices=sorted(GEOM_MODULES),
        default="groot",
        help="view set: groot/pi = 3 RoboCasa views, pi_libero = 2 RoboCerebra views",
    )
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="pointing queries per forward. KEEP AT 1: MolmoPoint's vendored pointing "
        "path is not batch-safe (batch>1 mis-shapes the point-embedding mask and "
        "would yield WRONG coordinates); >1 auto-falls-back to sequential anyway. "
        "Parallelize across GPUs/processes instead (see run_pipeline.py).",
    )
    ap.add_argument("--shard", type=int, default=0, help="process only episodes with idx %% num_shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if args.batch_size > 1:
        _patch_molmo_batching()  # partial fix only; pointing path still not fully batch-safe

    VIEWS = tuple(importlib.import_module("workspace_models.labels." + GEOM_MODULES[args.geom]).VIEWS)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, padding_side="left")

    def _sample_metadata(messages: list) -> dict:
        """Per-sample pointing metadata via a single-sample processor call (CPU, ~ms).

        The batched processor CONCATENATES token_pooling across the batch while the generated
        pointing tokens index per-sample structures — slicing the batch metadata is wrong
        (caused an all-empty sweep). Preprocessing is deterministic, so a per-sample re-run
        yields exactly the structures the model saw for that sample.
        """
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_pointing_metadata=True,
        )
        if "metadata" in inputs:
            return inputs["metadata"]
        return {k: inputs[k] for k in ("token_pooling", "subpatch_mapping", "image_sizes") if k in inputs}

    def _generate(messages_list: list) -> list[list[list[float]]]:
        """Run a batch of single-image pointing chats; returns per-query point lists."""
        inputs = processor.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        for k in ("metadata", "token_pooling", "subpatch_mapping", "image_sizes"):
            inputs.pop(k, None)  # drop batch-level metadata (unused; see _sample_metadata)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        # the custom code keeps the ViT in fp32 while the rest is bf16 and relies on autocast
        # to bridge the connector boundary (F.linear fp32-input x bf16-weight).
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        results = []
        for i in range(len(messages_list)):
            text = processor.decode(out[i][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
            try:
                meta = _sample_metadata(messages_list[i])
                pts = model.extract_image_points(
                    text, meta["token_pooling"], meta["subpatch_mapping"], meta["image_sizes"]
                )
                results.append([[float(p[-2]), float(p[-1])] for p in pts])  # rows: [obj_id, img_num, x, y]
            except Exception as e:
                print(f"    extract fail: {e}", flush=True)
                results.append([])
        return results

    def _msg(img: Image.Image, query: str) -> list:
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": f"Point to the {query}"}],
            }
        ]

    # END-TO-END probe: the batched path must actually EXTRACT points (a weaker no-exception
    # probe once silently returned all-empty results).
    batch_size = max(args.batch_size, 1)
    if batch_size > 1:
        probe = Image.new("RGB", (224, 224), (255, 255, 255))
        for x in range(80, 144):
            for y in range(80, 144):
                probe.putpixel((x, y), (220, 20, 20))
        try:
            res = _generate([_msg(probe, "red square"), _msg(probe, "red square")])
            if not all(len(r) > 0 for r in res):
                raise RuntimeError(f"batched probe extracted {[len(r) for r in res]} points")
            print(f"[molmo] batched probe OK: {[len(r) for r in res]} points", flush=True)
        except Exception as e:
            print(f"[molmo] batched pointing failed probe ({e}); falling back to sequential", flush=True)
            batch_size = 1

    def point_many(pairs: list[tuple[Image.Image, str]]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            try:
                out.extend(_generate([_msg(img, q) for img, q in chunk]))
            except Exception as e:
                print(f"    batch fail ({e}); retrying sequentially", flush=True)
                for img, q in chunk:
                    try:
                        out.extend(_generate([_msg(img, q)]))
                    except Exception as e2:
                        print(f"    point fail ({q}): {e2}", flush=True)
                        out.append([])
        return out

    task_dir = Path(args.in_dir).expanduser() / args.task
    sub_name = f"_subgoals{args.tag}.json"
    for sj in sorted(task_dir.glob(f"ep*{sub_name}"))[args.shard :: args.num_shards]:
        out_path = sj.with_name(sj.name.replace(sub_name, f"_points{args.tag}.json"))
        if out_path.exists():
            continue
        sub = json.loads(sj.read_text())
        d = np.load(sj.with_name(sj.name.replace(sub_name, "_frames.npz")), allow_pickle=True)
        framesets = {v: d[f"frames_{v}"] for v in VIEWS}
        fidx = d["frame_indices"]
        # collect every (keyframe x view x object) query for this episode, then batch
        queries: list[tuple[Image.Image, str]] = []
        slots: list[tuple[int, str, str]] = []  # (keyframe list idx, view, object)
        kf_entries = []
        for sg in sub["subgoals"]:
            kf = int(sg["completion_frame"])
            row = int(np.argmin(np.abs(fidx - kf)))
            kf_entries.append({"frame": kf, "subgoal": sg["name"], "views": {v: {} for v in VIEWS}})
            objs_by_view = sg.get("salient_objects", {})
            for view in VIEWS:
                img = Image.fromarray(framesets[view][row])
                objs = objs_by_view.get(view, []) if isinstance(objs_by_view, dict) else objs_by_view
                for obj in objs or []:
                    queries.append((img, obj))
                    slots.append((len(kf_entries) - 1, view, obj))
        for (ki, view, obj), pts in zip(slots, point_many(queries)):
            kf_entries[ki]["views"][view][obj] = pts
        for e in kf_entries:
            found = {v: sum(len(p) for p in e["views"][v].values()) for v in VIEWS}
            print(f"[{sj.name}] kf={e['frame']} points={found}", flush=True)
        out_path.write_text(
            json.dumps({"keyframes": kf_entries, "model": args.model, "spec": sub.get("spec", "salient")}, indent=2)
        )


if __name__ == "__main__":
    main()
