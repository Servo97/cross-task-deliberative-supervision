# H14 edge schema + pass-2 deliberation contract (FROZEN)

Authority: `internal_planning_and_todos/aug_22/deliberative_workspace_plan.md` §3, §6, and the
adversarial-panel amendments **A1 (circularity)**, **A2 (frame-level SupCon)**, **A6 (measured cost)**,
**A9 (accuracy, not self-consistency)**.

This file is the frozen definition of what pass 2 emits and what the encoder is allowed to consume.
It is content-addressed: `prompt_sha` / `schema_sha` are sha256 over the literals in
`scripts/deliberation/pass2_prompt.py`, printed by `python -m scripts.deliberation.pass2_prompt --shas`.
Changing any literal changes the sha, which changes `edge_store_id`, which invalidates every
downstream encoder. That is intentional — an edge store and the prompt that produced it are one object.

---

## 1. Unit of judgment

A **pair** is (anchor_segment, candidate_segment). A segment is identified by a **global segment id**:

```
seg_id = "<domain>/<task>/<episode_id>/<segment_index>"
    domain ∈ {robocasa, remembench, robomme}
```

`segment_index` indexes the frozen segmentation for that domain and NEVER moves:

| domain | segmentation source | contract |
|---|---|---|
| robocasa | `keyframes` in `wsm_labels_pi_mirror/<Task>/vlm_episode_pi_%06d.npz` via `caption_segments.segments_from_keyframes` | mirrors `train_wsm_base/data.py::per_frame_subgoal_idx` byte-for-byte |
| remembench | causal_v1 keyframe store, same npz schema and same function | same |
| robomme | RLE of the official per-step `simple_subgoal` column | verified 80/80 episodes contiguous+covering, 0 empty (`robomme_subgoal_audit.json`) |

Pairs are **unordered**: the store holds one record per {anchor, candidate} set, and the driver
dedups symmetric duplicates before spending tokens.

## 2. Edge types — BEHAVIORAL definitions (frozen wording)

The definitions below are the ones that appear verbatim in the pass-2 system prompt. They are
written about **policy knowledge and completion conditions**, never about pixels, because a
visual-similarity edge is exactly the degenerate signal A1 warns about.

| type | frozen definition | what it buys |
|---|---|---|
| `EQUIVALENT` | The SAME policy knowledge completes both segments. A policy that can do one, with no new information, can do the other. The bound objects may differ in colour or instance, but the verb frame, the roles it binds, and the completion condition are the same. | SupCon positive |
| `ANALOGOUS` | The knowledge TRANSFERS under a substitution of object or scene, but something must be re-grounded — a different object class filling the same role, a different receptacle, a different appliance of the same kind. Same skill, different binding. | SupCon positive (weight < EQUIVALENT) |
| `CONTRAST` | Superficially similar — same verb, or the same object class, or a near-identical scene — but a **different completion condition**. Succeeding at one while treating it as the other produces a wrong outcome. This is the deceptive look-alike. | hard negative |
| `UNRELATED` | Neither the skill nor the completion condition is shared. Not a hard negative — just far apart. | dropped (not a training signal) |

**Adjudication order** (frozen, so ties are not left to the model's mood):
`CONTRAST` is checked FIRST. If the completion conditions differ in a way that would make a swap
fail, the pair is `CONTRAST` even when the verb and the object class match. Only then
`EQUIVALENT` → `ANALOGOUS` → `UNRELATED`.

## 3. Per-pair verdict record

```jsonc
{
  "anchor":      "robocasa/KettleBoiling/000001/3",
  "candidate":   "robocasa/SearingMeat/000042/2",
  "type":        "EQUIVALENT",          // one of the four above
  "confidence":  "high",                 // {high, med, low}
  "rationale":   "both place cookware on the burner named by the instruction",  // <= 25 words
  "memory_relation": "same_kind",        // {same_kind, different_kind, one_sided, none}
  "stratum":     "cross_task"            // provenance of the CANDIDATE (see §5), not a judgment
}
```

Field rules, all enforced by the driver's structural validator (A7 resume gate operates on these):

- `type` ∈ the four literals. Anything else = record rejected, anchor re-queued.
- `confidence` ∈ {high, med, low}. **Low-confidence edges are EXCLUDED from the default objective**
  (plan §4); a sensitivity arm includes them. They are still stored — exclusion is a consumer
  decision, never a producer decision.
- `rationale` ≤ 25 words, non-empty, must not name frame indices or camera views.
- `memory_relation` compares the two segments' pass-1 `memory_dependency.kind` as JUDGED BY THE
  MODEL, and is checked against the pass-1 fields by the QA harness. Disagreement rate is a
  reported QA statistic, not a rejection rule.
- `stratum` is written by the DRIVER, not the model. The model never learns which candidates were
  forced in, so the quota mechanism cannot leak into the verdict.

## 4. Bucket record (what one request produces)

One request per anchor carrying K candidates (plan §3: k=12 after symmetry dedup, +25% mined hard
negatives). The stored unit is the bucket, so a resumed shard can tell a complete bucket from a
truncated one:

```jsonc
{
  "anchor": "robocasa/KettleBoiling/000001/3",
  "candidates": ["...", "..."],          // exactly the K ids sent, in the order sent
  "verdicts": [ {...}, {...} ],           // exactly K records, one per candidate, same order
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "stop"},
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "reasoning_effort": "medium",
  "prompt_sha": "...", "schema_sha": "...",
  "candidate_order_seed": 20260822        // A9: the order this bucket was presented in
}
```

**Structural resume gate** (mirrors `caption_segments.validate_existing`): a bucket file is valid
only if it parses, `len(verdicts) == len(candidates)`, every verdict's `candidate` equals the
candidate at the same index, and every record passes §3. `finish_reason == "length"` ⇒ INVALID
regardless of content — a truncated bucket silently loses verdicts (A6), and a lost verdict is
indistinguishable from an UNRELATED one downstream.

## 5. Candidate mining — stratified quotas (A1c, FROZEN)

Mining from embedding neighbourhoods alone makes `EQUIVALENT ⊂ embedding-nearest` by construction.
Every bucket therefore carries candidates from four strata with **pre-registered quotas**:

| stratum | how the candidate is drawn | quota per bucket (K=12) |
|---|---|---|
| `within_task` | top-k′ descriptor-cosine neighbours from the SAME task | 3 |
| `cross_task` | top-k″ cosine neighbours restricted to a DIFFERENT task, same domain | 4 |
| `cross_domain` | top cosine neighbours restricted to a DIFFERENT domain | 2 |
| `mined_hard_neg` | high cosine, **different postconditions** or a `failure_lookalikes` string-match | 3 |

Pre-registered floors, measured on the pilot and re-measured on the full store
(**a miss is a HOLD, not a footnote**):

- ≥ **40%** of accepted positives (EQUIVALENT ∪ ANALOGOUS) have `stratum = cross_task` or
  `cross_domain`.
- ≥ **15%** of accepted positives have `stratum = cross_domain`.
- every task contributes ≥ 1 cross-task EQUIVALENT edge, or its isolation is flagged (plan §3 QA c).

Candidate order inside a bucket is shuffled by `blake2b(anchor_id, seed)` so stratum never
correlates with position.

## 6. QA gates before any training consumes edges

Plan §3 (a)-(d) stand. A9 replaces "self-consistency is enough" with:

| gate | procedure | bar |
|---|---|---|
| **G-A: accuracy sheet** | ≥ **200** edges, stratified over the 4 types × 4 strata, rendered as frames + both descriptors + verdict + rationale; adjudicated blind to the model's verdict | agreement with adjudication, reported with **Wilson 95%** bounds; pre-registered floor **lower bound ≥ 0.70** for EQUIVALENT∪CONTRAST |
| **G-B: planted CONTRAST probes** | pairs constructed from task DEFINITIONS, not from embeddings — known deceptive look-alikes whose ground truth is CONTRAST by construction (§7) | **recovery rate ≥ 0.70**, Wilson lower bound reported |
| **G-C: order-flip stability** | 500 pairs judged twice at different `candidate_order_seed` | flip rate **< 10%** or HOLD (necessary, NOT sufficient — A9) |
| **G-D: A1a AUC gate** | AUC of raw descriptor-cosine at predicting EQUIVALENT-vs-CONTRAST on the probe set | **AUC ≥ ~0.90 ⇒ HOLD pass 2**: deliberation adds nothing over embedding |
| **G-E: quota floors** | §5 | as stated |
| **G-F: provenance** | model id, `prompt_sha`, `schema_sha`, `reasoning_effort`, seeds, code sha, git head — content-addressed like every label store | present or the store is not consumable |

## 7. Planted known-CONTRAST probes (A9, construction rule)

Probes are built from the RoboCasa composite task sources (s2 §3), where the completion condition is
readable from `_check_success`. Each probe pairs two segments that look alike but whose completion
condition provably differs. Implementation: `scripts/deliberation/build_probes.py`.

**The families are defined over the subskill/object vocabulary the descriptor pass ACTUALLY emits**,
measured on the pilot (`place` 70, `grasp` 47, `turn` 22, `reach`/`wait`/`navigate` 7, `open`/`wipe`
4, …). The first draft guessed verbs — `wash`, `rinse`, `scrub`, `spray` — that the model never
produces, and three of five families matched **zero** pairs. Guessing a controlled vocabulary is how
a QA gate silently becomes a no-op, so any new family must be checked against the histogram.

| family | pair construction | ground truth |
|---|---|---|
| `burner_binding` | `KettleBoiling` `turn`(stove knob) vs `SearingMeat` `turn`(stove knob) — same verb, same object; KB's correct burner is the one the distractor is NOT on, SM's is the one the instruction NAMES | CONTRAST |
| `faucet_binding` | `RinseSinkBasin` `turn`(faucet handle) vs `WashLettuce` `turn`(faucet handle) — same verb, same object; RSB latches three spout orientations, WL accumulates elapsed wash time | CONTRAST |
| `tool_binding` | `CuttingToolSelection` `grasp`(knife) vs `grasp`(peeler), different episodes ⇒ different `self.food` | CONTRAST |
| `accumulator_vs_place` | `ScrubCuttingBoard`, SAME episode: `place`(sponge) vs `wipe`(sponge on board) — `place` completes on release, `wipe` completes on a contact TIMER plus a swept-extent threshold | CONTRAST |
| `set_completion` | `PortionHotDogs`, SAME episode, two `place` segments on the same object class — placing a second item on a plate that already holds one FAILS | CONTRAST |
| `sanity_positive` | two `RinseSinkBasin` `turn`(faucet handle) segments from DIFFERENT episodes | EQUIVALENT — without positives, a model that always answers CONTRAST scores 1.0 on every row above |

Note two families deliberately allow **same-episode** pairs: "one is already on this plate" and "the
sponge has been in contact for N seconds" are within-episode facts, and excluding same-episode pairs
(the default, correct for every other family) removes exactly the pairs those families are about.

Measured on the 307-segment pilot slice: **57 probes, all six families populated, 47 CONTRAST /
10 EQUIVALENT.** Probes are injected into ordinary buckets, indistinguishable from mined candidates;
their ids live in `probes.json`, which the model never sees.

## 8. What the encoder may consume (D3 boundary)

Edges and descriptors are **DATA for a contrastive objective**. They are never text targets for the
VLA and never text targets through ω (H13 R3b: language-as-target through w cost −8 pp even when the
head discriminated). Concretely, the only permitted consumption is:

- `EQUIVALENT ∪ ANALOGOUS` → positive pairs for `SupCon_deliberative`
- `CONTRAST` → weighted hard negatives in the same term
- `UNRELATED` → nothing
- `confidence` → per-edge weight; `low` excluded by default
- **A2**: the term operates on **FRAMES** carrying their segment's edge labels, not on a mean-pooled
  `z_seg` — a w16 GDN window spans ≈1.1 segments, so a segment-pooled objective is invariant to the
  content the read actually consumes.

## 9. On-disk layout

```
<store>/edges/<edge_store_id>/
  buckets/<domain>/<task>/<anchor_ep>_<anchor_seg>.bucket.json   # §4, one per anchor
  probes.json                                                    # §7, ids only
  manifest.json                                                  # counts, quota measurements, shas
  _provenance/run_shard<N>_<ts>.json                             # model/prompt/schema/seed/git head
edge_store_id = sha256(canonical_json({
    prompt_sha, schema_sha, model, reasoning_effort, max_tokens,
    k_per_bucket, quotas, mining_seed, embedding_model_id, corpus_manifest_sha
}))
```
