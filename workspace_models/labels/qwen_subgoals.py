"""WSM label stage A: Qwen3-VL subgoal/keyframe decomposition + per-view target objects.

RoboCasa365 3-view port of Isaac-GR00T/wsm/vlm_label/qwen_subgoals.py (DexJoCo front+wrist).
Feeds the subsampled episode frames (from extract_frames.py) to Qwen as an interleaved
multi-image conversation (left, right, eye-in-hand per timestamp) and asks for ONE JSON object:
fine-grained subgoals, each with a completion keyframe and the objects to point at, listed
SEPARATELY PER VIEW. Those names feed MolmoPoint in stage B.

TWO LABEL SPECS share this stage (`--spec`); only the system prompt and the object schema differ,
the downstream stages are identical:

  salient   (default, the original) — embodiment-INCLUSIVE saliency: the gripper + task effects +
            manipulated objects, 1-3 per view. "What stands out / is involved."
  causal_v1 — causal-confusion framing (de Haan et al. 2019): supervise what the action must
            CAUSALLY touch, not what is salient. EXACTLY TWO entities per subgoal — the
            manipulated object and its goal slot (receptacle / appliance / counter region) — and
            NOTHING else (no gripper, no arm, no water/flame/steam effects, no background). Strictly
            sparser than `salient` by construction.

The VIEW SET is injectable via `--geom` (default `groot` = the 3 RoboCasa views, byte-identical
prompt to before). A geometry module may override the view naming and the system prompts
(`QWEN_PROMPT_NAME` / `QWEN_CAPTION` / `QWEN_SPECS`) because the prompt NAMES the cameras — see
pi_geometry_libero.py, which carries the 2-view RoboCerebra/LIBERO text.

Run in the vlm_labeler venv (transformers 4.57.x):
  ~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.qwen_subgoals \
      --task <Task> --in ~/Research/TRI/wsm_data/wsm_vlm_rc_v0 --model Qwen/Qwen3-VL-8B-Instruct --device cuda:0
  ... --spec causal_v1 --tag _causal_v1     # namespaced; never overwrites the salient artifacts
  ... --geom pi_libero                      # 2-view RoboCerebra/LIBERO

Output per episode: <in>/<task>/ep{idx:03d}_subgoals<tag>.json
  {"subgoals": [{"name", "completion_frame", "salient_objects": {view: [str,...] for view in VIEWS},
                 "causal_roles": {"manipulated": str, "goal": str}   # causal_v1 only
                }],
   "model", "prompt", "spec"}
`salient_objects` is the union the pointing stage consumes and carries the same meaning in both
specs ("point at these in this view"), so stages B and C need no spec awareness.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

GEOM_MODULES = {"groot": "geometry", "pi": "pi_geometry", "pi_libero": "pi_geometry_libero"}

# Short JSON keys the model uses per view, mapped to the canonical VIEWS (RoboCasa defaults; a
# geometry module overrides them with QWEN_PROMPT_NAME / QWEN_CAPTION).
_PROMPT_NAME = {"agentview_left": "left", "agentview_right": "right", "eye_in_hand": "eye_in_hand"}
_CAPTION = {"agentview_left": "left", "agentview_right": "right", "eye_in_hand": "eye-in-hand"}
# Views a FLAT (non-per-view) object list falls back to: the third-person cameras.
_AGENTVIEWS = ("agentview_left", "agentview_right")


def view_naming(G) -> tuple[tuple[str, ...], dict, dict, dict, tuple[str, ...]]:
    """(VIEWS, prompt_name, from_prompt, caption, agentviews) for a geometry module."""
    prompt_name = getattr(G, "QWEN_PROMPT_NAME", _PROMPT_NAME)
    caption = getattr(G, "QWEN_CAPTION", _CAPTION)
    return (
        tuple(G.VIEWS),
        prompt_name,
        {v: k for k, v in prompt_name.items()},
        caption,
        tuple(getattr(G, "AGENTVIEWS", _AGENTVIEWS)),
    )


SYSTEM = (
    "You are labeling a robot-manipulation episode (a Franka Panda arm in a kitchen) for training "
    "a workspace model that must learn WHERE the action-relevant change happens. You will see "
    "frames sampled from one episode; each timestamp shows THREE views captioned with the frame "
    "index: LEFT and RIGHT third-person agentviews, and an EYE-IN-HAND close-up from a camera on "
    "the gripper (objects there appear very large, truncated, or out of frame). "
    "Decompose the task into 4-6 FINE-GRAINED subgoals in temporal order (e.g. approach, grasp, "
    "lift/transport, place/act, finish). For each subgoal pick the frame index where it is "
    "COMPLETED (the visible moment of completion). Then, SEPARATELY PER VIEW, list ONLY the "
    "objects DIRECTLY INVOLVED in the action at that exact moment: the specific object(s) being "
    "manipulated or about to be contacted; the robot gripper ONLY when it is touching or right "
    "next to the target; and any task EFFECT (e.g. 'running faucet water', 'stove flame', "
    "'the drawer that is opening', 'dispensed coffee'). "
    "Name AT MOST 1-3 objects per view. Do NOT list static background or scene furniture "
    "(countertops, walls, floors, cabinets/drawers/appliances that are NOT the current target, "
    "knife blocks, decorations, or any object not part of this subgoal's action). List an object "
    "under a view only if it is actually visible there; but EVERY subgoal MUST name at least the "
    "robot gripper/end-effector AND the primary manipulated/target object under SOME view — NEVER "
    "leave all three views empty for a subgoal (the gripper is almost always visible, especially in "
    "the eye-in-hand view; the target object, e.g. the drawer handle being pulled, is too). Use "
    "short, specific noun phrases a pointing model can resolve (e.g. 'drawer handle', 'red mug', "
    "'faucet handle', 'black gripper fingers'). "
    "Also produce an `expanded_prompt`: a 2-4 sentence step-by-step instruction that EXPANDS the "
    "terse task string into an ordered, lower-level description of how to accomplish it (the kind of "
    "richer instruction you'd give to guide the robot through the substeps) — grounded in what this "
    "specific demo shows. "
    'Respond with ONLY a JSON object: {"expanded_prompt": str, "subgoals": [{"name": str, '
    '"completion_frame": int, "salient_objects": {"left": [str, ...], "right": [str, ...], '
    '"eye_in_hand": [str, ...]}}]}. completion_frame must be one of the captioned frame indices.'
)


SYSTEM_CAUSAL_V1 = (
    "You are labeling a robot-manipulation episode (a Franka Panda arm in a kitchen) to mark the "
    "CAUSALLY RELEVANT region of each step — the places the action must physically touch or change "
    "in order to succeed, NOT merely the places that stand out or look interesting. You will see "
    "frames sampled from one episode; each timestamp shows THREE views captioned with the frame "
    "index: LEFT and RIGHT third-person agentviews, and an EYE-IN-HAND close-up from a camera on "
    "the gripper (objects there appear very large, truncated, or out of frame). "
    "Decompose the task into 4-6 FINE-GRAINED subgoals in temporal order (e.g. approach, grasp, "
    "lift/transport, place/act, finish). For each subgoal pick the frame index where it is "
    "COMPLETED (the visible moment of completion). "
    "For each subgoal name EXACTLY TWO causal entities: "
    "(1) `manipulated` — the ONE object the gripper is acting on, or is just about to act on, "
    "during that subgoal (e.g. 'red apple', 'kettle handle', 'stove knob'). If the subgoal acts on "
    "part of a fixture, name the PART that moves or is touched ('microwave door handle', not "
    "'microwave'). "
    "(2) `goal` — the ONE place that object must end up, or the thing whose state must change as a "
    "result (e.g. 'the sink basin', 'the open microwave interior', 'the cabinet shelf', 'the "
    "front-left stove burner'). If the subgoal's outcome is a state change of the manipulated thing "
    "itself (turning a knob, opening a door), repeat that part as the goal. "
    "Then, SEPARATELY PER VIEW, report which of those two entities is actually VISIBLE in that "
    "view. "
    "Do NOT name the robot gripper, fingers, or arm. Do NOT name task EFFECTS (running water, "
    "stove flame, steam, dispensed liquid). Do NOT name background or scene furniture "
    "(countertops, walls, floors, cabinets/drawers/appliances that are NOT this subgoal's target). "
    "ONLY the two causal entities — this label is deliberately sparse. "
    "Omit an entity from a view where it is not visible, but EVERY subgoal MUST list its "
    "`manipulated` entity under at least ONE view. "
    "Use short, specific noun phrases a pointing model can resolve (e.g. 'drawer handle', 'red "
    "mug', 'faucet handle'). "
    "Also produce an `expanded_prompt`: a 2-4 sentence step-by-step instruction that EXPANDS the "
    "terse task string into an ordered, lower-level description of how to accomplish it (the kind of "
    "richer instruction you'd give to guide the robot through the substeps) — grounded in what this "
    "specific demo shows. "
    'Respond with ONLY a JSON object: {"expanded_prompt": str, "subgoals": [{"name": str, '
    '"completion_frame": int, "causal_entities": {"manipulated": str, "goal": str}, '
    '"visible": {"left": [str, ...], "right": [str, ...], "eye_in_hand": [str, ...]}}]}. '
    "Every string inside `visible` must be EXACTLY one of that subgoal's two causal-entity "
    "phrases, copied verbatim. completion_frame must be one of the captioned frame indices."
)

SPECS = {"salient": SYSTEM, "causal_v1": SYSTEM_CAUSAL_V1}


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    raw = re.sub(r",\s*([}\]])", r"\1", m.group(0))  # tolerate trailing commas
    return json.loads(raw)


def normalize_objects(sg: dict, G) -> dict:
    """Coerce a subgoal's salient_objects into {view: [str]} over canonical VIEWS, tolerating
    the per-view dict form (left/right/eye_in_hand) and a flat list fallback."""
    views, _, from_prompt, _, agentviews = view_naming(G)
    so = sg.get("salient_objects", {})
    out = {v: [] for v in views}
    if isinstance(so, dict):
        for pk, objs in so.items():
            view = from_prompt.get(pk, pk if pk in out else None)
            if view in out and isinstance(objs, list):
                out[view] = [str(o) for o in objs]
    elif isinstance(so, list):  # flat list -> apply to the third-person agentviews
        shared = [str(o) for o in so]
        for v in agentviews:
            out[v] = list(shared)
    return out


def normalize_causal(sg: dict, G) -> tuple[dict, dict]:
    """causal_v1 subgoal -> ({view: [phrase]}, {"manipulated": str, "goal": str}).

    The two causal phrases are the ONLY things that may be pointed at; `visible` is filtered
    against them (case/whitespace-insensitive) so a hallucinated extra entity — a gripper, a
    background object — cannot leak into the pointing stage and re-introduce the saliency labels
    this spec exists to replace. Tolerates the older per-view `salient_objects` shape (same
    filter) and, if the model omits `visible` entirely, falls back to the two agentviews only
    (never eye_in_hand: the close-up frequently does NOT contain the goal slot, and pointing at an
    absent object is exactly how a sparse label silently becomes a wrong one).
    """
    views, _, from_prompt, _, agentviews = view_naming(G)
    ce = sg.get("causal_entities", {}) or {}
    roles = {r: str(ce.get(r, "") or "").strip() for r in ("manipulated", "goal")}
    allowed = {v.lower(): v for v in roles.values() if v}
    out = {v: [] for v in views}
    if not allowed:
        return out, roles

    vis = sg.get("visible")
    if not isinstance(vis, dict):
        vis = sg.get("salient_objects")
    if isinstance(vis, dict):
        for pk, objs in vis.items():
            view = from_prompt.get(pk, pk if pk in out else None)
            if view in out and isinstance(objs, list):
                seen: list[str] = []
                for o in objs:
                    canon = allowed.get(str(o).strip().lower())
                    if canon and canon not in seen:
                        seen.append(canon)
                out[view] = seen
    if not any(out.values()):  # no usable per-view report -> both entities on the agentviews
        both = list(dict.fromkeys(v for v in roles.values() if v))
        for v in agentviews:
            out[v] = list(both)
    return out, roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument(
        "--spec",
        choices=sorted(SPECS),
        default="salient",
        help="label spec: 'salient' (embodiment-inclusive saliency, the original) or "
        "'causal_v1' (manipulated object + goal slot ONLY)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for the output filename, e.g. '_causal_v1' -> ep*_subgoals_causal_v1.json. "
        "Namespaces a new spec's artifacts so they never overwrite the salient ones.",
    )
    ap.add_argument(
        "--geom",
        choices=sorted(GEOM_MODULES),
        default="groot",
        help="view set (and, if the module provides them, the view names + system "
        "prompts): groot/pi = 3 RoboCasa views, pi_libero = 2 RoboCerebra views",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=20, help="frames per request (3 views each -> token budget)")
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="output budget; 768 truncated long-composite subgoal JSON (~10 subgoals) -> parse "
        "fail. Subgoals are bounded by max_frames (~20), so 2048 fits the full output.",
    )
    ap.add_argument("--shard", type=int, default=0, help="process only episodes with idx %% num_shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    G = importlib.import_module("workspace_models.labels." + GEOM_MODULES[args.geom])
    VIEWS, _, _, CAPTION, _ = view_naming(G)
    specs = getattr(G, "QWEN_SPECS", SPECS)
    if set(specs) != set(SPECS):
        raise SystemExit(f"{args.geom}: QWEN_SPECS keys {sorted(specs)} != {sorted(SPECS)}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    processor = AutoProcessor.from_pretrained(args.model)

    task_dir = Path(args.in_dir).expanduser() / args.task
    for fp in sorted(task_dir.glob("ep*_frames.npz"))[args.shard :: args.num_shards]:
        out_path = fp.with_name(fp.name.replace("_frames.npz", f"_subgoals{args.tag}.json"))
        if out_path.exists():
            continue
        d = np.load(fp, allow_pickle=True)
        framesets = {v: d[f"frames_{v}"] for v in VIEWS}
        fidx, prompt = d["frame_indices"], str(d["prompt"])
        keep = np.linspace(0, len(fidx) - 1, min(args.max_frames, len(fidx))).round().astype(int)
        keep = np.unique(keep)

        caps = [CAPTION[v] for v in VIEWS]
        content = [{"type": "text", "text": f"Task: {prompt}\nEpisode frames ({', '.join(caps)} views):"}]
        for i in keep:
            for j, view in enumerate(VIEWS):
                head = f"\nframe {int(fidx[i])} {caps[j]}:" if j == 0 else f" {caps[j]}:"
                content.append({"type": "text", "text": head})
                content.append({"type": "image", "image": Image.fromarray(framesets[view][i])})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": specs[args.spec]}]},
            {"role": "user", "content": content},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        parsed, text = None, ""
        for attempt in range(3):  # greedy first; resample on malformed JSON
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=attempt > 0,
                    temperature=0.4 if attempt > 0 else None,
                )
            text = processor.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            try:
                parsed = parse_json(text)
                valid = set(int(x) for x in fidx)
                for sg in parsed["subgoals"]:
                    if int(sg["completion_frame"]) not in valid:  # snap to nearest sampled frame
                        sg["completion_frame"] = int(fidx[np.argmin(np.abs(fidx - int(sg["completion_frame"])))])
                    if args.spec == "causal_v1":
                        sg["salient_objects"], sg["causal_roles"] = normalize_causal(sg, G)
                    else:
                        sg["salient_objects"] = normalize_objects(sg, G)
                if args.spec == "causal_v1" and not any(
                    sg["salient_objects"][v] for sg in parsed["subgoals"] for v in VIEWS
                ):  # every subgoal empty => the causal schema was not followed at all; resample
                    raise ValueError("causal_v1: no subgoal named a visible causal entity")
                parsed.setdefault("expanded_prompt", str(prompt))  # fall back to the terse task
                break
            except Exception as e:
                print(f"[{fp.name}] parse attempt {attempt + 1} failed: {e}", flush=True)
                parsed = None
        if parsed is None:
            print(f"[{fp.name}] PARSE FAIL after retries\nraw: {text[:300]}", flush=True)
            continue
        parsed.update({"model": args.model, "prompt": prompt, "spec": args.spec, "geom": args.geom})
        out_path.write_text(json.dumps(parsed, indent=2))
        kf = [sg["completion_frame"] for sg in parsed["subgoals"]]
        print(f"[{fp.name}] {len(parsed['subgoals'])} subgoals, keyframes={kf}", flush=True)


if __name__ == "__main__":
    main()
