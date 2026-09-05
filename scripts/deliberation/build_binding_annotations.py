#!/usr/bin/env python3
"""H14 — LLM-FREE per-episode BINDING annotations + a same-subskill CONTRAST-binding relabel.

Why this exists (A9 §15.2): planted-probe recovery is 0.889 where the deciding difference is in the
pass-1 descriptor text and 0.296 where it lives only in the task's success predicate. The variables
that decide those probes — which burner, which tool, which side, which colour, which count — are
NOT latent: every domain writes them into episode metadata. This module reads that metadata and
emits, per episode, a `binding` dict of globally-namespaced slots. Nothing here calls a model.

    binding slot        example values                     source
    knob                rear_left / front_center           robocasa ep_meta["refs"]["knob"]
    cut_food            avocado / potato                   robocasa ep_meta["refs"]["food"]
    recycle_ends        (stool_1_room, stool_3_room)       robocasa fixture_refs["end_ref_plastic"]
                                                             / ["end_ref_glass"]
    mystery_type        bottled_water / alcohol            object_cfgs[mystery_middle].obj_groups
    pan_container_cat   pan                                object_cfgs[vegetable_container].info.cat
    return_side         left / right / origin              remembench task name (task-level constant)
    oils_route          (left, right)                      remembench task name
    cook_food/wait_min  scallops / 3.0                     remembench episodes.jsonl prompt
    cube_targets        ("red",) / ("green","red")        robomme instruction
    unmask_swap         True / False                       robomme task name (Unmask family only)
    repeat_count        2 / 3      ordinal 3 / 5           robomme instruction
    button_phase        before / after                     robomme instruction

Only slots the SUCCESS PREDICATE actually reads are emitted as `binding`. Eight of the thirteen
RoboCasa tasks run an identical predicate in every episode (see ROBOCASA_PREDICATE_CONSTANT) and
therefore get an EMPTY binding — their look-alike pairs differ in within-episode progress, not in
any per-episode value, and this table must not pretend otherwise.

Two stages:

  build   ->  <out>/<binding_id>/bindings.jsonl  (+ manifest.json)
              one record per (domain, task, episode) with its `binding` dict

  relabel ->  <out>/<binding_id>/relabel_<label_id>/
                edges_binding.npz    src, dst, orig_kind, slot_id, agree_qwen
                manifest.json        counts, per-task counts, Qwen agreement
              A SIDECAR of the frozen label artifact (`--labels …/stage_e_labels/<label_id>`).
              It READS segments.npz / edges_E1.npz / vocab.json and writes NOTHING into them.

Relabel rule (frozen, deliberately conservative):
    an edge (u, v) becomes CONTRAST-binding iff
      (a) both segments' episodes have a binding record, AND
      (b) segments[u].subskill == segments[v].subskill, AND
      (c) some slot key is present in BOTH bindings and its values differ.
    Slot keys are globally namespaced, so a cross-task pair is compared only on the slots the two
    tasks actually share (e.g. `knob` for SearingMeat vs StirVegetables); tasks with no shared slot
    are never relabelled by geometry alone.

Usage (no GPU, no network):
    python scripts/deliberation/build_binding_annotations.py build
    python scripts/deliberation/build_binding_annotations.py relabel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DELIB = Path("~/Research/TRI/wsm_data/deliberation").expanduser()
ROBOCASA_ROOT = Path("~/Research/robocasa/datasets/v1.0/target/composite").expanduser()
RMB_ROOT = Path("~/Research/TRI/wsm_data/remembench_v02/train").expanduser()
ROBOMME_INDEX = DELIB / "pass1_store/robomme/_robomme_index.json"
DEFAULT_LABELS = DELIB / "stage_e_labels/bd13c1a48f2dc5be"
DEFAULT_OUT = DELIB / "binding_annotations"

EDGE_KINDS = ("EQUIVALENT", "ANALOGOUS", "CONTRAST")  # index order of edges_*.npz `kind`

#: STRICT slot set: slots whose value change flips the COMPLETION CONDITION (a swap fails).
#: The excluded slots (`cook_food`, `add_food`, `target_object`, `set_target`, `pan_container_cat`)
#: name which object is present, not which of several the policy must pick, so a difference there
#: is an object substitution (ANALOGOUS), not a broken binding.
STRICT_SLOTS = frozenset(
    {
        "knob",
        "cut_food",
        "recycle_ends",
        "mystery_type",  # robocasa, predicate-verified
        "return_side",
        "sink_source",
        "oils_route",
        "wait_min",
        "add_after_min",  # remembench
        "cube_targets",
        "repeat_count",
        "ordinal",
        "placement_ordinal",  # robomme
        "button_phase",
        "unmask_swap",
        "n_buttons",
    }
)


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# ==============================================================================================
# RoboCasa — ep_meta.json under <task>/<date>/lerobot/extras/episode_%06d/
# ==============================================================================================
def _objs(meta: dict) -> dict:
    return {c["name"]: c for c in (meta.get("object_cfgs") or [])}


def _cat(objs: dict, name: str):
    c = objs.get(name)
    return ((c or {}).get("info") or {}).get("cat")


def _fixture(objs: dict, name: str):
    return ((objs.get(name) or {}).get("placement") or {}).get("fixture")


def _region_ref(objs: dict, name: str):
    pl = (objs.get(name) or {}).get("placement") or {}
    return (pl.get("sample_region_kwargs") or {}).get("ref")


def _pos_x(objs: dict, name: str):
    pl = (objs.get(name) or {}).get("placement") or {}
    p = pl.get("pos")
    if isinstance(p, list) and p and isinstance(p[0], (int, float)):
        return round(float(p[0]), 3)
    return None


#: Which RoboCasa tasks have a variable the SUCCESS PREDICATE actually reads per episode.
#: Established by reading all 13 `_check_success` implementations, not guessed from ep_meta:
#: the other eight run an identical predicate in every episode (relative side ordering, fixed
#: object names, fixed counts, fixed rack indices, or a pure runtime accumulator), so a
#: per-episode binding table cannot separate their segments and must not pretend to.
ROBOCASA_PREDICATE_BOUND = {
    "SearingMeat": ["knob"],
    "StirVegetables": ["knob"],
    "CuttingToolSelection": ["cut_food"],
    "RecycleBottlesByType": ["recycle_ends", "mystery_type"],
    "PanTransfer": ["pan_container_cat"],  # partial: + a runtime touch latch
}
ROBOCASA_PREDICATE_CONSTANT = (
    "ScrubCuttingBoard",
    "KettleBoiling",
    "GatherTableware",
    "HeatKebabSandwich",
    "CategorizeCondiments",
    "PackIdenticalLunches",
    "PortionHotDogs",
    "SeparateFreezerRack",
)


def robocasa_binding(task: str, meta: dict) -> tuple[dict, dict]:
    """ep_meta -> (predicate-bound slots, other per-episode scene facts).

    Only the first dict feeds the relabel rule. The second is recorded for provenance and for a
    later within-episode progress annotation, and is never compared.
    """
    objs = _objs(meta)
    refs = meta.get("refs") or {}
    fx = meta.get("fixture_refs") or {}
    b: dict = {}

    if task in ("SearingMeat", "StirVegetables") and refs.get("knob"):
        b["knob"] = refs["knob"]  # _check_success reads self.knob
    if task == "CuttingToolSelection" and refs.get("food"):
        b["cut_food"] = refs["food"]  # _CUTTING_MAP[self.food] flips the predicate
    if task == "RecycleBottlesByType":
        b["recycle_ends"] = (fx.get("end_ref_plastic"), fx.get("end_ref_glass"))
        b["mystery_type"] = (objs.get("mystery_middle") or {}).get("obj_groups")
    if task == "PanTransfer":
        b["pan_container_cat"] = _cat(objs, "vegetable_container")

    scene = {
        "objects": {n: _cat(objs, n) for n in sorted(objs)},
        "fixture_refs": fx,
        "predicate_constant_across_episodes": task in ROBOCASA_PREDICATE_CONSTANT,
    }
    return ({k: v for k, v in b.items() if v not in (None, (), (None, None))}, scene)


def robocasa_episodes() -> dict:
    out = {}
    for task_dir in sorted(ROBOCASA_ROOT.glob("*")):
        task = task_dir.name
        for p in sorted(task_dir.glob("*/lerobot/extras/episode_*/ep_meta.json")):
            ep = int(p.parent.name.split("_")[-1])
            try:
                meta = json.loads(p.read_text())
            except Exception:
                continue
            binding, scene = robocasa_binding(task, meta)
            out[(task, ep)] = {
                "binding": binding,
                "scene": scene,
                "lang": meta.get("lang", ""),
                "source": "ep_meta.json (refs / fixture_refs / object_cfgs)",
            }
    return out


# ==============================================================================================
# ReMemBench — task-level constants (the side/route binding IS the task variant) + prompt parse
# ==============================================================================================
RMB_TASK_SLOTS = {
    "MemWashAndReturnLeft": {"return_side": "left"},
    "MemWashAndReturnRight": {"return_side": "right"},
    "MemWashAndReturnSameLocation": {"return_side": "origin"},
    "MemFruitInSinkLeftFar": {"sink_source": "left_far"},
    "MemFruitInSinkRightFar": {"sink_source": "right_far"},
    "MemRetrieveOilsFromCounterLL": {"oils_route": ("left", "left")},
    "MemRetrieveOilsFromCounterLR": {"oils_route": ("left", "right")},
    "MemRetrieveOilsFromCounterRL": {"oils_route": ("right", "left")},
    "MemRetrieveOilsFromCounterRR": {"oils_route": ("right", "right")},
    "MemPutKBowlInCabinet": {"set_target": "bowls->cabinet"},
    "MemPutKBreadInMicrowave": {"set_target": "breads->microwave"},
    "MemHeatPot": {},
    "MemHeatPotMultiple": {},
}


def remembench_binding(task: str, prompt: str) -> dict:
    b = dict(RMB_TASK_SLOTS.get(task, {}))
    m = re.search(r"Pick up the ([a-z ]+?) and place it in the sink", prompt)
    if m:
        b["target_object"] = m.group(1).strip()
    m = re.search(r"cook the ([a-z ]+?), wait for ([0-9.]+) minutes", prompt)
    if m:
        b["cook_food"], b["wait_min"] = m.group(1).strip(), float(m.group(2))
    m = re.search(
        r"stove with the ([a-z ]+?), add the ([a-z ]+?) after ([0-9.]+) minutes, "
        r"and wait for ([0-9.]+) minutes",
        prompt,
    )
    if m:
        b["cook_food"], b["add_food"] = m.group(1).strip(), m.group(2).strip()
        b["add_after_min"], b["wait_min"] = float(m.group(3)), float(m.group(4))
    return b


def remembench_episodes() -> dict:
    out = {}
    for f in sorted(RMB_ROOT.glob("*/*/lerobot/meta/episodes.jsonl")):
        task = f.parents[3].name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = (r.get("tasks") or [""])[0]
            out[(task, int(r["episode_index"]))] = {
                "binding": remembench_binding(task, prompt),
                "lang": prompt,
                "source": "task name + episodes.jsonl prompt",
            }
    return out


# ==============================================================================================
# RoboMME — the instruction carries colour / count / ordinal / button phase verbatim
# ==============================================================================================
ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
COLOURS = ("red", "green", "blue", "purple", "white", "black")


def robomme_binding(task: str, instr: str) -> dict:
    # `swap` is only a completion-condition flip INSIDE the unmask family (the container that hides
    # a cube is shuffled after the reveal); asserting it across unrelated task families would flag
    # every Unmask-vs-anything pair, so it is scoped to that family.
    b: dict = {}
    if "Unmask" in task:
        b["unmask_swap"] = task.endswith("Swap")
    seq = [c for c in re.findall(r"\b(" + "|".join(COLOURS) + r")\b(?=\s+cubes?)", instr)]
    if seq:
        b["cube_targets"] = tuple(seq)
    m = re.search(r"reaches the target for the (\w+) time", instr)
    if m and m.group(1) in ORDINAL:
        b["ordinal"] = ORDINAL[m.group(1)]
    m = re.search(r"(?:repeating this(?: back and forth)? (?:action|motion) )?(\w+) times", instr)
    if m and m.group(1) in NUMWORD:
        b["repeat_count"] = NUMWORD[m.group(1)]
    m = re.search(r"put (\w+) (?:red|green|blue|purple|white|black) cubes? into the bin", instr)
    if m and m.group(1) in NUMWORD:
        b["repeat_count"] = NUMWORD[m.group(1)]
    m = re.search(r"right (before|after) the button was pressed", instr)
    if m:
        b["button_phase"] = m.group(1)
    m = re.search(r"on the (\w+) target it was previously placed on", instr)
    if m and m.group(1) in ORDINAL:
        b["placement_ordinal"] = ORDINAL[m.group(1)]
    if "press both buttons" in instr:
        b["n_buttons"] = 2
    elif "press the button" in instr:
        b["n_buttons"] = 1
    return b


def robomme_episodes() -> dict:
    out = {}
    if not ROBOMME_INDEX.exists():
        return out
    idx = json.loads(ROBOMME_INDEX.read_text())
    for k, v in idx.items():
        instr = v.get("instruction", "") or ""
        out[(v["task"], int(k))] = {
            "binding": robomme_binding(v["task"], instr),
            "lang": instr,
            "source": "_robomme_index.json instruction",
        }
    return out


# ==============================================================================================
# build
# ==============================================================================================
def canon(v):
    if isinstance(v, (list, tuple)):
        return json.dumps([canon(x) for x in v], sort_keys=True)
    return json.dumps(v, sort_keys=True)


RCB_CORPUS = DELIB / "robocerebra_corpus.json"


def robocerebra_episodes() -> dict:
    """RoboCerebra: EMPTY-BUT-PRESENT, deliberately (§24.4).

    Its only candidate slot is `distractor`, and applying §17.1's classification that slot is an
    observable layout constant that no action depends on and that the success predicate never reads
    -- the NOT_ACTION_RELEVANT class that already excluded rmb's `sink_source`. Mining it would
    label near-EQUIVALENT pairs as hard negatives, which is the v1 labelling bug in reverse.

    So every episode is emitted with `binding: {}` rather than the domain being omitted. Absence and
    "present with no action-relevant slots" are different facts: the strict relabel sidecar reads a
    missing domain as "not annotated yet", and would then have no way to distinguish a domain that
    was assessed and found to have none from one that was never looked at.
    """
    if not RCB_CORPUS.is_file():
        return {}
    corpus = json.loads(RCB_CORPUS.read_text())
    out = {}
    for ep in corpus["episodes"]:
        bddl = ep["bddl_file"]
        task = bddl[:-5] if bddl.endswith(".bddl") else bddl
        out[(task, int(ep["episode_index"]))] = {
            "binding": {},
            "lang": ep.get("task_line", ""),
            "source": "robocerebra_corpus.json (no action-relevant slot; distractor is NOT_ACTION_RELEVANT per §17.1)",
            "scene": {"predicate_constant_across_episodes": False},
        }
    return out


def stage_build(args) -> None:
    per_domain = {
        "robocasa": robocasa_episodes(),
        "remembench": remembench_episodes(),
        "robomme": robomme_episodes(),
        "robocerebra": robocerebra_episodes(),
    }
    records = []
    for dom, table in per_domain.items():
        for (task, ep), rec in sorted(table.items()):
            records.append(
                {
                    "domain": dom,
                    "task": task,
                    "episode": ep,
                    "binding": {k: (list(v) if isinstance(v, tuple) else v) for k, v in rec["binding"].items()},
                    "n_slots": len(rec["binding"]),
                    "predicate_constant": bool((rec.get("scene") or {}).get("predicate_constant_across_episodes")),
                    "source": rec["source"],
                    "lang": rec["lang"],
                }
            )

    payload = "\n".join(json.dumps(r, sort_keys=True) for r in records)
    binding_id = hashlib.sha256((code_sha() + payload).encode()).hexdigest()[:16]
    out = Path(args.out).expanduser() / binding_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "bindings.jsonl").write_text(payload + "\n")

    # feasibility table: per task, how many episodes carry >=1 slot, and which slots
    per_task = defaultdict(
        lambda: {"n_episodes": 0, "n_with_binding": 0, "slots": Counter(), "distinct_bindings": set(), "domain": ""}
    )
    for r in records:
        t = per_task[r["task"]]
        t["domain"] = r["domain"]
        t["n_episodes"] += 1
        if r["n_slots"]:
            t["n_with_binding"] += 1
        for k in r["binding"]:
            t["slots"][k] += 1
        t["distinct_bindings"].add(canon(sorted(r["binding"].items())))
    feas = {}
    for t, v in sorted(per_task.items()):
        n_distinct = len(v["distinct_bindings"])
        cover = v["n_with_binding"] / max(1, v["n_episodes"])
        feas[t] = {
            "domain": v["domain"],
            "n_episodes": v["n_episodes"],
            "binding_coverage": round(cover, 4),
            "n_distinct_bindings": n_distinct,
            "slots": sorted(v["slots"]),
            # yes    = separates within-task pairs (>=2 distinct bindings) and cross-task
            # partial = one binding for the whole task (a task-level constant): separates
            #           cross-task pairs only
            # no      = the success predicate is identical in every episode of this task
            "verdict": (
                "yes" if cover >= 0.99 and n_distinct > 1 else "partial" if cover >= 0.99 and v["slots"] else "no"
            ),
        }
    manifest = {
        "binding_id": binding_id,
        "code_sha": code_sha(),
        "n_records": len(records),
        "per_domain": {d: len(t) for d, t in per_domain.items()},
        "sources": {
            "robocasa": str(ROBOCASA_ROOT),
            "remembench": str(RMB_ROOT),
            "robomme": str(ROBOMME_INDEX),
            "robocerebra": str(RCB_CORPUS),
        },
        # Explicit, so "assessed and empty" is never read as "not annotated".
        "domains_with_no_action_relevant_slots": ["robocerebra"],
        "feasibility_per_task": feas,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                "binding_id": binding_id,
                "n_records": len(records),
                "per_domain": manifest["per_domain"],
                "out": str(out),
            },
            indent=1,
        )
    )


# ==============================================================================================
# relabel
# ==============================================================================================
def stage_relabel(args) -> None:
    import numpy as np

    root = Path(args.out).expanduser() / args.binding_id
    bindings = {}
    for line in (root / "bindings.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            bindings[(r["domain"], r["task"], r["episode"])] = r["binding"]

    labels = Path(args.labels).expanduser()
    seg = np.load(labels / "segments.npz", allow_pickle=True)
    edges = np.load(labels / "edges_E1.npz", allow_pickle=True)
    vocab = json.loads((labels / "vocab.json").read_text())
    domains, tasks = vocab["domains"], vocab["tasks"]

    seg_key = [(domains[int(d)], tasks[int(t)], int(e)) for d, t, e in zip(seg["domain"], seg["task"], seg["episode"])]
    seg_bind = [bindings.get(k) for k in seg_key]
    subskill = seg["subskill"]

    src, dst, kind = edges["src"], edges["dst"], edges["kind"]
    hit_src, hit_dst, hit_kind, hit_slot = [], [], [], []
    slot_counter, task_counter, agree = Counter(), Counter(), Counter()
    per_task_counter = Counter()
    n_comparable = 0
    for i in range(len(src)):
        u, v = int(src[i]), int(dst[i])
        bu, bv = seg_bind[u], seg_bind[v]
        if not bu or not bv:
            continue
        if subskill[u] != subskill[v]:
            continue
        shared = set(bu) & set(bv)
        if args.slots == "strict":
            shared &= STRICT_SLOTS
        if not shared:
            continue
        n_comparable += 1
        diff = sorted(k for k in shared if canon(bu[k]) != canon(bv[k]))
        if not diff:
            continue
        hit_src.append(u)
        hit_dst.append(v)
        hit_kind.append(int(kind[i]))
        hit_slot.append("+".join(diff))
        for k in diff:
            slot_counter[k] += 1
        task_counter["|".join(sorted((seg_key[u][1], seg_key[v][1])))] += 1
        per_task_counter[seg_key[u][1]] += 1
        per_task_counter[seg_key[v][1]] += 1
        agree[EDGE_KINDS[int(kind[i])]] += 1

    out = root / f"relabel_{labels.name}_{args.slots}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "edges_binding.npz",
        src=np.asarray(hit_src, dtype=np.int32),
        dst=np.asarray(hit_dst, dtype=np.int32),
        orig_kind=np.asarray(hit_kind, dtype=np.int8),
        slot=np.asarray(hit_slot, dtype="<U64"),
        agree_qwen=np.asarray([k == EDGE_KINDS.index("CONTRAST") for k in hit_kind], dtype=bool),
    )

    n_hits = len(hit_src)
    n_qwen_contrast_comparable = 0
    n_qwen_contrast_flagged = agree["CONTRAST"]
    for i in range(len(src)):
        if int(kind[i]) != EDGE_KINDS.index("CONTRAST"):
            continue
        u, v = int(src[i]), int(dst[i])
        bu, bv = seg_bind[u], seg_bind[v]
        sh = set(bu or ()) & set(bv or ())
        if args.slots == "strict":
            sh &= STRICT_SLOTS
        if bu and bv and subskill[u] == subskill[v] and sh:
            n_qwen_contrast_comparable += 1

    man = {
        "binding_id": args.binding_id,
        "label_id": labels.name,
        "code_sha": code_sha(),
        "rule": (
            "same subskill AND >=1 shared binding slot whose values differ "
            "=> CONTRAST-binding (a new edge subtype, stored as a sidecar)"
        ),
        "n_edges_in_E1": int(len(src)),
        "n_edges_comparable": n_comparable,
        "n_relabelled": n_hits,
        "relabelled_from": {k: int(v) for k, v in agree.items()},
        "n_positives_relabelled": int(agree["EQUIVALENT"] + agree["ANALOGOUS"]),
        "qwen_agreement": {
            "of_binding_flagged_edges_qwen_said_CONTRAST": round(n_qwen_contrast_flagged / max(1, n_hits), 4),
            "of_qwen_CONTRAST_edges_that_are_comparable_binding_also_flagged": round(
                n_qwen_contrast_flagged / max(1, n_qwen_contrast_comparable), 4
            ),
            "n_qwen_CONTRAST_comparable": n_qwen_contrast_comparable,
        },
        "slot_set": args.slots,
        "by_slot": dict(slot_counter.most_common()),
        "by_task_endpoint": dict(per_task_counter.most_common()),
        "by_task_pair": dict(task_counter.most_common(40)),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "by_task_pair"}, indent=1))
    print(f"[write] {out}")


# ==============================================================================================
# check — does the binding rule separate the A9 planted probes and the sheet CONTRAST edges?
# ==============================================================================================
def stage_check(args) -> None:
    root = Path(args.out).expanduser() / args.binding_id
    bindings, subskill = {}, {}
    for line in (root / "bindings.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            bindings[(r["domain"], r["task"], r["episode"])] = r["binding"]
    for line in (DELIB / "pass2_store/index/segments.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            subskill[r["seg_id"]] = str((r["descriptor"] or {}).get("subskill", ""))

    def key(seg_id):
        d, t, e, _ = seg_id.split("/")
        return (d, t, int(e))

    def flags(a, b):
        ka, kb = key(a), key(b)
        if ka not in bindings or kb not in bindings:
            return None, "episode_missing_from_table"
        ba, bb = bindings[ka], bindings[kb]
        if not ba or not bb:
            return False, "no_predicate_bound_slot_for_task"
        if subskill.get(a) != subskill.get(b):
            return False, "different_subskill"
        shared = set(ba) & set(bb)
        if args.slots == "strict":
            shared &= STRICT_SLOTS
        if not shared:
            return False, "no_shared_slot"
        diff = sorted(k for k in shared if canon(ba[k]) != canon(bb[k]))
        return (bool(diff), "+".join(diff) if diff else "all_slots_equal")

    out = {"slot_set": args.slots, "binding_id": args.binding_id}
    pr = json.loads((DELIB / "qa_pass2_probe_recovery.json").read_text())
    rows = []
    for p_ in pr["probes"]:
        if p_["ground_truth"] != "CONTRAST":
            continue
        f, why = flags(p_["anchor"], p_["candidate"])
        rows.append(
            {
                "probe_id": p_["probe_id"],
                "family": p_["family"],
                "qwen": p_["verdict"],
                "binding_flags_contrast": f,
                "reason": why,
            }
        )
    fam = defaultdict(lambda: {"n": 0, "flagged": 0, "qwen_recovered": 0})
    for r, p_ in zip(rows, [x for x in pr["probes"] if x["ground_truth"] == "CONTRAST"]):
        v = fam[r["family"]]
        v["n"] += 1
        v["flagged"] += bool(r["binding_flags_contrast"])
        v["qwen_recovered"] += p_["verdict"] in ("CONTRAST", "UNRELATED")
    out["planted_probes"] = {
        "n": len(rows),
        "binding_flags_contrast": sum(bool(r["binding_flags_contrast"]) for r in rows),
        "by_family": {k: dict(v) for k, v in sorted(fam.items())},
        "rows": rows,
    }

    sheet = json.loads((DELIB / "qa_pass2_accuracy_sheet.json").read_text())
    srows = []
    for e in sheet["edges"]:
        if e["qwen"] != "CONTRAST":
            continue
        f, why = flags(e["anchor"], e["candidate"])
        srows.append(
            {
                "edge_index": e["edge_index"],
                "stratum": e["stratum"],
                "letter_rule": e["adjudicator"],
                "binding_flags_contrast": f,
                "reason": why,
            }
        )
    out["sheet_contrast_edges"] = {
        "n": len(srows),
        "binding_flags_contrast": sum(bool(r["binding_flags_contrast"]) for r in srows),
        "rows": srows,
    }

    ra = DELIB / "qa_pass2_contrast_readjudication.json"
    if ra.exists():
        d = json.loads(ra.read_text())
        by_uid = {f"S{r['edge_index']:03d}": r for r in srows}
        agree = Counter()
        for e in d["sheet_contrast"]["edges"]:
            r = by_uid.get(e["uid"])
            if r is None or r["binding_flags_contrast"] is None:
                continue
            agree[f"intent={e['intent_rule'] == 'CONTRAST'}|binding={bool(r['binding_flags_contrast'])}"] += 1
        out["binding_vs_intent_rule_on_sheet"] = dict(agree)

    dest = root / f"check_{args.slots}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("planted_probes", "sheet_contrast_edges")}
            | {
                "probes": {k: v for k, v in out["planted_probes"].items() if k != "rows"},
                "sheet": {k: v for k, v in out["sheet_contrast_edges"].items() if k != "rows"},
            },
            indent=1,
        )
    )
    print(f"[write] {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "relabel", "check"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--binding-id", default=None)
    ap.add_argument(
        "--slots",
        choices=["strict", "all"],
        default="strict",
        help="strict = only slots whose change flips the completion condition",
    )
    a = ap.parse_args()
    if a.stage in ("relabel", "check") and not a.binding_id:
        cands = sorted(Path(a.out).expanduser().glob("*/bindings.jsonl"))
        if len(cands) != 1:
            raise SystemExit("pass --binding-id (found %d candidates)" % len(cands))
        a.binding_id = cands[0].parent.name
    {"build": stage_build, "relabel": stage_relabel, "check": stage_check}[a.stage](a)


if __name__ == "__main__":
    main()
