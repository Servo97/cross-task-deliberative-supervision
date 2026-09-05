"""H14 pass-2 deliberation prompt + verdict schema. FROZEN LITERALS -- content-addressed.

The definitions here are the SAME WORDS as `scripts/deliberation/edge_schema.md` §2. If the two ever
drift, the md is documentation and this file is the contract; but they must not drift, so any edit
touches both and re-pins the shas.

    python scripts/deliberation/pass2_prompt.py --shas
"""

from __future__ import annotations

import hashlib
import json

EDGE_TYPES = ("EQUIVALENT", "ANALOGOUS", "CONTRAST", "UNRELATED")
CONFIDENCES = ("high", "med", "low")
MEMORY_RELATIONS = ("same_kind", "different_kind", "one_sided", "none")
MAX_RATIONALE_WORDS = 25

SYSTEM = (
    "You are deciding, for one ANCHOR segment of a robot-manipulation demonstration and a numbered "
    "list of CANDIDATE segments, what relation each candidate has to the anchor. You are given "
    "structured functional descriptions, not images. Judge POLICY KNOWLEDGE and COMPLETION "
    "CONDITIONS. Visual or scene similarity is not evidence; two segments in the same kitchen doing "
    "the same-looking motion can still be a CONTRAST, and two segments in different scenes with "
    "different objects can still be EQUIVALENT.\n"
    "\n"
    "Relation types:\n"
    "- EQUIVALENT: the SAME policy knowledge completes both segments. An agent that can do one, "
    "with no new information, can do the other. Bound objects may differ in colour or instance, but "
    "the verb frame, the roles it binds, and the completion condition are the same.\n"
    "- ANALOGOUS: the knowledge TRANSFERS under a substitution of object or scene, but something "
    "must be re-grounded -- a different object class filling the same role, a different receptacle, "
    "a different appliance of the same kind. Same skill, different binding.\n"
    "- CONTRAST: superficially similar -- same verb, or the same object class, or a near-identical "
    "scene -- but a DIFFERENT COMPLETION CONDITION. Succeeding at one while treating it as the "
    "other produces a wrong outcome. This is the deceptive look-alike.\n"
    "- UNRELATED: neither the skill nor the completion condition is shared.\n"
    "\n"
    "Adjudication order, applied in this order for every candidate:\n"
    "1. Do the completion conditions differ in a way that would make a swap FAIL? If yes -> "
    "CONTRAST, even when the verb and the object class match.\n"
    "2. Otherwise, would the same knowledge complete both with nothing re-grounded? -> EQUIVALENT.\n"
    "3. Otherwise, does it transfer under an object or scene substitution? -> ANALOGOUS.\n"
    "4. Otherwise -> UNRELATED.\n"
    "\n"
    "confidence is high when the descriptions state the completion condition explicitly, med when "
    "you infer it, low when the descriptions are too thin to decide. Do not use high to mean "
    "'obvious'.\n"
    "memory_relation compares the two segments' memory dependence: same_kind if both depend on the "
    "same kind of earlier information, different_kind if both depend on history but of different "
    "kinds, one_sided if exactly one depends on history, none if neither does.\n"
    f"rationale: at most {MAX_RATIONALE_WORDS} words, states the DECIDING difference or the shared "
    "completion condition. Never mention frame indices, cameras, views, or candidate numbers.\n"
    "\n"
    "Return exactly one verdict per candidate, in the order the candidates were given, as ONLY the "
    "JSON object described by the schema."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "integer"},
                    "type": {"type": "string", "enum": list(EDGE_TYPES)},
                    "confidence": {"type": "string", "enum": list(CONFIDENCES)},
                    "memory_relation": {"type": "string", "enum": list(MEMORY_RELATIONS)},
                    "rationale": {"type": "string"},
                },
                "required": ["candidate", "type", "confidence", "memory_relation", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def render_segment(desc: dict, *, header: str) -> str:
    """One segment's pass-1 descriptor -> the compact text pass 2 reads.

    Deliberately drops `failure_lookalikes` from the ANCHOR/CANDIDATE body: those strings seed the
    mining stratum, and echoing them into the judge's context would let the mined hard negatives
    announce themselves (the same leak `stratum` is kept driver-side to avoid).
    """
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from workspace_models.labels.caption_segments import memory_kinds_of

    t = desc.get("target_object", {}) or {}
    attrs = ", ".join(t.get("attributes") or []) or "-"
    md = desc.get("memory_dependency", {}) or {}
    kinds = memory_kinds_of(desc)
    kind_str = "+".join(kinds)
    return (
        f"{header}\n"
        f"  subskill: {desc.get('subskill', '')}\n"
        f"  verb_frame: {desc.get('verb_frame', '')}\n"
        f"  object: {t.get('class', '')} [{attrs}]\n"
        f"  object_state: {t.get('state_before', '')} -> {t.get('state_after', '')}\n"
        f"  spatial: {desc.get('spatial_relation', '')}\n"
        f"  preconditions: {'; '.join(desc.get('preconditions') or [])}\n"
        f"  postconditions: {'; '.join(desc.get('postconditions') or [])}\n"
        f"  memory_dependency: {kind_str}"
        f"{' -- ' + str(md.get('evidence')) if kinds != ['none'] else ''}\n"
    )


def build_bucket_messages(
    anchor_desc: dict,
    candidate_descs: list[dict],
    *,
    anchor_domain: str = "",
    candidate_domains: list[str] | None = None,
) -> list[dict]:
    """One anchor + K candidates -> one chat request (plan §3: bucket, do not iterate pairs).

    Task and domain names are NOT shown. A model told two segments come from the same task would
    have a free shortcut to EQUIVALENT, which is exactly the trivial pairing that E1-ctrl-T exists
    to measure. The judge sees only functional content.
    """
    parts = [render_segment(anchor_desc, header="ANCHOR:")]
    for i, d in enumerate(candidate_descs):
        parts.append(render_segment(d, header=f"CANDIDATE {i}:"))
    body = (
        f"There are {len(candidate_descs)} candidates, numbered 0..{len(candidate_descs) - 1}.\n\n"
        + "\n".join(parts)
        + f"\nReturn exactly {len(candidate_descs)} verdicts, one per candidate, in order."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def prompt_sha() -> str:
    return hashlib.sha256(SYSTEM.encode()).hexdigest()


def schema_sha() -> str:
    return hashlib.sha256(json.dumps(VERDICT_SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_verdict(v: dict, index: int) -> str:
    """'' if the verdict record satisfies the frozen schema, else the first violation."""
    if not isinstance(v, dict):
        return "not an object"
    for k in ("candidate", "type", "confidence", "memory_relation", "rationale"):
        if k not in v:
            return f"missing {k}"
    if int(v["candidate"]) != index:
        return f"candidate index {v['candidate']} != position {index}"
    if v["type"] not in EDGE_TYPES:
        return f"type {v['type']!r} not in {EDGE_TYPES}"
    if v["confidence"] not in CONFIDENCES:
        return f"confidence {v['confidence']!r} not in {CONFIDENCES}"
    if v["memory_relation"] not in MEMORY_RELATIONS:
        return f"memory_relation {v['memory_relation']!r} not in {MEMORY_RELATIONS}"
    r = str(v["rationale"]).strip()
    if not r:
        return "empty rationale"
    if len(r.split()) > MAX_RATIONALE_WORDS:
        return f"rationale {len(r.split())} words > {MAX_RATIONALE_WORDS}"
    return ""


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "pass2_prompt_sha256": prompt_sha(),
                "pass2_schema_sha256": schema_sha(),
                "edge_types": EDGE_TYPES,
                "confidences": CONFIDENCES,
                "memory_relations": MEMORY_RELATIONS,
                "max_rationale_words": MAX_RATIONALE_WORDS,
                "system_prompt_chars": len(SYSTEM),
            },
            indent=1,
        )
    )
