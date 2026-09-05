# Project narrative — long-context robot policies via workspace tokens (June → September 2026)

Source document for the project documentation: motivations, the logical chain from one result to
the next, and the system flows. Numbers carry their protocol; every section names the file that
holds the sealed table. Written 2026-09-04. Companion indexes: `README.md` (framing),
`hypothesis_ledger.md` (claims), `rwm/README.md` (H14 tree), `PAPER_STATE.md` (paper wording).

---

## 1. The problem and the framing

**Motivation.** Vision-language-action (VLA) policies such as π0.5 act from the current image and
the instruction. Many household tasks are not solvable that way: the right action depends on an
event the robot saw earlier (which drawer it already checked, where it put the bowl, how many times
it pressed the button). Appending raw history frames is costly and, worse, teaches the policy to
latch onto incidental historical detail (causal confusion; de Haan 2019). We want a **compressed,
task-relevant state** that is cheap to read at deployment.

**Three axes we refuse to conflate** (`README.md`): long-HORIZON tasks (many steps), long-CONTEXT
policies (the policy receives history), and MEMORY-DEMANDING tasks (the current observation is an
insufficient state, the POMDP sense). Benchmarks are placed by demand — RoboCasa365 ≈ none,
ReMemBench and RoboMME = yes, RoboCerebra = long-horizon with memory modes — and mechanisms are
placed by what they do with history.

**Operational test — long-context reasoning (LCR).** Holding the current observation and
instruction fixed, does changing a task-relevant earlier event change the policy's output? The
paper's central line became a **double contingency**: mechanism value tracks the task's memory
demand AND the backbone, not a mechanism ranking.

---

## 2. The object: workspace tokens ω

A **workspace model (WSM)** is a small encoder that maps the policy backbone's frozen features at
a frame to a 512-d latent ω (v1: one token per sampled frame; v2 target: a multi-token workspace).
Design history, in order:

| step | what we built | what we learned | where |
|---|---|---|---|
| Keyframe-patch head on GR00T N1.7 (June) | encoder–decoder WSM predicting salient key-patches; decoder localizes salient patches at 6–11× chance | the encoder learns something real, but **probe-selected encoders decode worse** — select on the decode metric, not on probes | PAPER_STATE 06-28; memory `wsm-decoder-sanity` |
| Injecting ω into the policy (June) | first WSM evals negative on both backbones; a GR00T 0 % result was a NaN-weight encoder served by mistake | injection at inference is fragile: an eval-time distribution shift the policy never saw | PAPER_STATE 06-29 |
| Model-free pivot (June 28) | ω_{t+1} as a JEPA + SIGReg **auxiliary target** at the action head's penultimate layer; ω never used at inference, so the OOD failure vanishes | became the "training-time aux" axis | memory `wsm-jepa-penultimate-model-free-design` |
| Steering interfaces (July) | tanh single-token modulation; classifier-free guidance (CFG) on ω; **gated DeltaNet (GDN) recurrent read over a window of ω** feeding AdaLN of the action expert | GDN is the only interface that ingests *history* rather than one token | README "mechanism axes" |
| Frozen taps | ω is produced from the frozen π0.5 backbone (`pretrain150k/pi05/mg60_bal33/run/149999`) pooled to 512-d; one tap per benchmark, same network | lets one encoder serve every benchmark and keeps train↔serve identical | `workspace_models/features/*_pooled_tap.py` |

---

## 3. How the study found GDN to be the right way to ingest workspace tokens

The 3-axis study (July–August; `eval_results_final.md`, `remembench_results_final.md`, ledger
H1–H9) crossed steering × aux × fast weights, one factor at a time, on RoboCasa365 (5,000 rollouts
per arm, held-out reset) and ReMemBench (264 rollouts per arm, 13 memory tasks).

1. **H1 — the memory interface pays only where memory is demanded.** On ReMemBench the recurrent
   GDN read over ω beat tanh steering and the base: Spatial +14.9, Object-Set +12.5, overall +5.5
   (deltanet > tanh > base). On RoboCasa, after correcting a normalization bug (H6), **no mechanism
   beat plain fine-tuning (58.2)**; the earlier +4.1 "edge" was norm-stat headroom. The interface
   claim rests entirely on the memory benchmark — the first half of the double contingency.
2. **H2 — window curve.** ω-window {2, 8, 16, 32}: aggregate success declines monotonically with
   width (37.5 → 36.8 → 35.8 → 32.7 ≈ base 31.3); category optima differ (Spatial/Object-Set peak
   at w8, Prospective at w2). Wider windows did not bridge minutes-scale prospective memory.
3. **H7 — the decline is causal confusion, and it is repairable.** Pre-registered sign pattern: train
   with **history dropout** (p = 0.5, newest frame never masked). It hurt w8 (36.8 → 34.1) and
   rescued w16 (35.8 → **38.2**, the best ReMemBench cell) and w32 (32.7 → 34.8). Causal-content
   supervision was the second remedy (35.1 aux-only). The RoboCasa twin was a wash (58.1), so the
   rescue is **demand-gated**. Recipe adopted: **GDN over a w16 window + history dropout**.
4. **H3 — aux objectives do not compose with the read** (four instances; single-task heatpot shows
   active conflict). **H4 — fast weights (RoboTTT) never paid.** **H9 — PTRM test-time width scaling
   is inert** (K = 32 vs K = 1 within noise on both benchmarks; its +6.9 over base is training-side
   shaping; the read contributes ≈0 on ReMemBench at n = 880).
5. **H8 — backbone generality fails.** On GR00T N1.7 every mechanism hurt the aggregate; only the
   fruitRF within-task spatial-memory gain survived. Claims are scoped to π0.5 — the second half of
   the double contingency.
6. **H12 — RoboCerebra (third benchmark, protocol v3).** Initialised from released `pi05_libero`
   and fine-tuned 15k steps, **no workspace mechanism produced a memory-specific gain**; two were
   detectably worse on the memory stratum (bound ≈0.04× the base rate). A generality limit was
   found first: the canonical RoboCasa ω encoder **collapses** on LIBERO frames (coherence gap
   0.785 → 0.030), so ω had to be re-trained per domain.
7. **H13 — live/joint WSM supervision at post-train (RoboCasa) is negative**: inert alone, and
   composed with the GDN read it *subtracts* 5–7 pp; a caption-alignment head learned the centroid.
   Lesson banked as gate G4: an auxiliary whose discriminative term never beats chance is a HOLD.

**Net position after the study:** a recurrent GDN read over frozen ω helps exactly where memory is
demanded and the backbone is π0.5; the ω *content* itself (per-domain JEPA/SIGReg encoder) was the
weak link — it collapsed cross-domain and carried no decodable memory state. That is what H14 set
out to change.

---

## 4. H14 — cross-task deliberative workspace supervision (DWS), the current campaign

**Idea (user framing, plan §0.0).** Train ω to be the *missing sufficient statistic*: supervise the
encoder with **cross-task structure** — which moments in different tasks are functionally the same
(EQUIVALENT), similar (ANALOGOUS), or deliberately opposed (CONTRAST) — decided by a reasoning VLM
deliberating over pairs of segment descriptions. Then the GDN read over ω-history is the recurrent
carrier of that statistic (Markovianization), and the LCR counterfactual is its test.

### 4.1 System flow — the deliberation loop (Stage Q)

```
demonstrations (RoboCasa 1,950 eps · ReMemBench 323 · RoboMME 1,600 · RoboCerebra 994)
   │  segmentation from official subgoal columns / task structure
   ▼
pass-1  Qwen3.8-27B (vision, reasoning effort low) → per-segment DESCRIPTORS
        (what changed, memory-dependency kind, completion condition)        28,722 segments
   ▼
embed + mine  Qwen3-Embedding-0.6B → candidate cross-task pairs by cosine, floors on
        cross-task/cross-domain share (fullmine: 315,670 pairs)
   ▼
pass-2  Qwen judges bucketed pairs → TYPED EDGES {EQUIVALENT, ANALOGOUS, CONTRAST, UNRELATED}
        + memory-dependency descriptors (28,505 anchors; stores are content-addressed by
        model × effort × code sha; never mixed)
   ▼
labels  edges → SupCon positives; HARD NEGATIVES from a deterministic per-episode BINDING TABLE
        (the judge's CONTRAST precision was 0.17, the table's 0.94); label v2b + ctrl-Eb variants
```

### 4.2 System flow — encoder training (Stage E)

```
frozen π0.5 taps (one per domain, same network) → per-frame 512-d pooled features
   ▼
StageEEncoder  (domain adapters + shared trunk, serve-consistent language conditioning:
                task-mean for robocasa/rmb/robomme, per-frame subtask instruction for robocerebra)
   losses:  frame-level SupCon over typed edges (λ_del)  +  JEPA  +  SIGReg
   controls: ctrl-0b (λ_del = 0) · ctrl-S (edges rewired) · ctrl-T (same-task positives) ·
             ctrl-Eb (embedding-mined positives, same hard negatives) · ctrl-1Db (one domain)
   gates:   retrieval lift vs chance (the go/no-go) · per-domain eff-rank floors (validity only)
   ▼
ω stores per domain (train-time grid + a serve-aligned grid for RoboMME) · D7 parity: the
online producer must reproduce the stored ω at cos ≥ 0.999 (`--lang-mode stored`)
```

**Sealed result (encoder stage, 2026-08-31; `rwm/evidence/E-encoder-stage.md`).** ω retrieves
functionally matching moments across tasks on held-out episodes at **16.25× chance** (3 seeds);
removing the deliberative term leaves retrieval **at chance**, rewiring the edges falls **below
chance**, same-task positives fall below chance, and deliberation-mined positives beat
embedding-mined positives with the same hard negatives (+0.062, 3/3 seeds). Label quality was
measured, not assumed (EQUIVALENT precision 0.933 blind). Decodability probes were **negative**:
layout-type labels are perception, and no probe found progress state in ω (`E-decodability.md`).
The paper may claim a *structure* result; the *memory-content* and *policy* claims wait on §4.3.

### 4.3 System flow — policy stage (Stage P)

```
π0.5 (RoboCasa H300+MG pretrain step 149,999)  ──post-train──▶  policy with a GDN read
      inputs at each step: images + instruction + ω-window from the frozen Stage-E encoder
      read: gated DeltaNet over the last K ω (K = 16; RoboMME parity arm K = 24 = [8 demo ; 16 live])
      → conditioning vector → AdaLN of the action expert (flow-matching head)
      train-only history dropout 0.2–0.5; pos_decay_bias per slot (init −4.0 so old slots are read)
      serve: the SAME window rule (one shared function) and the SAME ω producer, D7-gated
```

Arms per benchmark (`rwm/todos/T-policy-arms.md`): ReMemBench P1′/P2′/P3′ (E1b-ω vs ctrl-0b-ω,
two seeds), RoboCerebra R1/R2, RoboMME M1/M2/M3/M3-ctrl — each paired against a **base curve**
trained under the same recipe with milestones kept (H14.9), evaluated under the sealed protocols
(`E-anchors.md`). Primary contrasts: E1b-ω − ctrl-0b-ω (is it the *structure*?), arm − base
(does memory-stratified success move?), interference rule (>5 pp below base is a finding).

### 4.4 What went wrong on the way, and the rules it left

| event | rule now (E-lessons) |
|---|---|
| first ReMemBench policy arms trained on an encoder conditioned on a non-causal episode-mean language vector; no serve convention reproduced ω | **train-time conditioning must be a statistic the server computes causally at rollout**; D7 parity in `stored` mode before any submit |
| zero-init decaying GDN read moved the conditioning by 7e-6 for the oldest slots | init −4.0; effective horizon ≪ window at init — bears on earlier w16/dnw8 arms |
| ω-store grid misaligned with serve frames on 81 % of RoboMME demo episodes | one shared `window_for_step` both sides import; serve-aligned store |
| five silent-success bugs (empty stores exiting 0) | count assertions and fail-closed loaders everywhere |
| two GPUs, one node type, one shared queue | canary any node entry on a real node before building a chain; cheapest evidence first |

---

## 5. Where things stand (2026-09-04) and how to follow along

- Encoder stage sealed; the serve-consistent 4-domain retrain (10 Stage-E cells) fires when the
  local pass-2 judge finishes (≈01:15 UTC 09-04) and the label chain runs.
- Base curves landed for ReMemBench (60k, 4 milestones) and RoboMME (70k, 7 milestones);
  RoboCerebra's is running (≈25 h). Checkpoint selection follows the base curve (H14.9).
- RoboMME taps (encoder + serve-aligned policy stores) landed; the FrameSamp+Modul parity arms
  (demo-prefix ω) are staged.
- Open decisions and the live board: `rwm/board.md`; hypothesis status: `rwm/README.md`.

## 6. Glossary

ω (workspace token) · GDN (gated DeltaNet recurrent read) · AdaLN (adaptive layer norm conditioning
of the action expert) · tap (frozen backbone feature extractor) · Stage Q/E/P (deliberation /
encoder / policy) · E1b / ctrl-0b / ctrl-Eb (deliberative encoder / structure-free control /
embedding-positive control) · D7 (train↔serve ω parity gate) · LCR (long-context reasoning
counterfactual) · fixed-800 (RoboMME protocol: 16 tasks × 50 fixed test episodes, predict 20 /
execute 16) · protocol v3 (RoboCerebra subtask-completion scorer) · 264-rollout lane (ReMemBench:
88 held-out episodes × 3).
