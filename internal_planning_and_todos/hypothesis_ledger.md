# Hypothesis ledger (weakest-hypothesis loop — skill: weakest-hypothesis, arXiv:2301.12987)

Rule: each row's CLAIM is the weakest hypothesis entailing all cited evidence. Stronger variants
listed with their discriminating experiment + status. Update on every new sealed/published row.
Evidence files: eval_results_final.md (robocasa sealed), remembench_results_final.md (rmb box-tier).

## H1 — Memory interface value

- **CLAIM (weakest):** An inference-time recurrent read over workspace history improves success on
  memory-demanding tasks within the window's time span (ReMemBench Spatial +14.9, Obj-Set +12.5,
  overall +5.5 over baseline; ordering deltanet > tanh > base holds). On the near-fully-observed
  RoboCasa benchmark under corrected action normalization, no tested mechanism edge survives so
  far (tanh −0.2, dw16 −0.3, jw01 −0.6, salient −2.9, A6 −3.4, Q2 −12.5 vs n-baseline 58.2).
- **Refuted stronger variant:** "recurrent read wins on RoboCasa (+4.1)" — margin was largely
  norm-stat headroom; killed by n-wave rows n1–n8 (pending n3 dw8, the last discriminator).
- **REFUTED 2026-08-05:** "dw8 retains an edge under corrected norms" — n3 sealed eval: dw8 56.8
  (−1.4 vs n-baseline 58.2); the inverted-U also fails to replicate (dw16 57.9 > dw8 56.8).
  n-wave complete, 8/8: NO mechanism beats corrected plain finetune on RoboCasa. H1's RoboCasa
  clause is now maximally weak and closed; the interface claim rests entirely on ReMemBench.
  - "value scales with memory demand generally (beyond these 2 benchmarks)" — discriminator:
    RoboMME (other agent's lane) — PARKED for us.
  - "holds across backbones" — discriminator: GR00T port — PARKED.

## H2 — Time-scale boundary

- **CLAIM (weakest):** The w=8 recurrent window (~seconds of context) does not improve
  minutes-scale prospective memory (Prospective: no arm beats baseline 8.3; deltanet 3.3).
- **DISCRIMINATORS RESOLVED 2026-08-06 (full curve {2,8,16,32}):**
  - "No feasible window bridges prospective demand" — ADOPTED: Prospective at w32 = 5.0 (vs base
    8.3); window scaling does not bridge minutes-scale demand. In fact Prospective peaks at w2
    (13.3) — recency, not depth, is what helps timing.
  - "Inverted-U peak w8 like robocasa" — REFUTED at the aggregate: overall is monotone
    decreasing (w2 37.5 > w8 36.8 > w16 35.8 > w32 32.7 ≈ base 31.3); w32 erases most of the
    workspace benefit. Nuance kept: category-optimal windows DIFFER (Spatial and Obj-Set peak at
    w8; Prospective at w2) — no single window serves all memory types; the aggregate ordering
    w2>w8 is 0.7pt and inside noise (the defensible claim is the decline at w16/w32, not w2>w8).
  - Single-task corroboration: fruitRF needs w>2 (dnw2 = base) — window demand is task-specific.

## H3 — Aux-objective composition

- **CLAIM (weakest):** Adding the predictive aux to the conditioning read does not change
  aggregate success on either benchmark (robocasa combo 57.4 vs 58.2/59.9 old-stats; rmb combo
  36.9 vs 36.8). Aggregate substitutes, both benchmarks.
- **Suggestive (NOT adopted — evidence at CI boundary):** the aux redistributes the category
  profile (Prospective 3.3→8.3, Obj-Assoc 6.7→11.9, Spatial 55.6→48.1) and widens the
  conditioning gate 2.2x. Discriminator: would need larger-n per-category eval — PARKED.
- **Evidence 2026-08-05:** robocasa combo sealed 58.8 (77.1, 50.5, 46.4) — below old-stats
  deltanet-alone 59.9. THIRD independent substitutes instance (robocasa jw01k16, rmb combo,
  robocasa combo). Claim upgraded from 2 to 3 benchmarks/frames.
- **Mechanism lead (from H5b evidence):** jw01k16 heatpot single-task 33.3 vs soup 0.0 suggests
  the "competition" is multi-task interference between objectives, sharpening WHY aggregate
  composition fails. Suggestive; fruit cells + any future capacity-scaled combo would test.
- **INTERFERENCE-AS-EXPLANATION REFUTED 2026-08-06 (heatpot combo-k16 cell):** single-task
  gdn+jepa on heatpot = 10.0 — BELOW both components (jepa 33.3, dnw8 16.7) and base (13.3).
  Non-composition persists with soup interference removed ⇒ the conflict is intrinsic to
  combining the objectives, not a mixture artifact. Weakest claim now: gdn+jepa do not compose
  in ANY tested regime (4 instances: robocasa jw01k16, robocasa combo sealed, rmb combo-k1,
  rmb combo-k16 36.1 ≈ deltanet 36.8) and actively conflict single-task on heatpot.
- Multi-task combo-k16: 36.1 (50.9, 1.7, 6.7, 85.1) — aggregate ≈ deltanet again; Obj-Set 85.1
  is best-of-study (suggestive only, CI); its Prospective 1.7 = worst-of-study, matching the
  single-task heatpot conflict signature across two independent regimes.
- **JEPA's effect is task-dependent IN SIGN** (single-task cells): best arm on heatpot (+20.0 vs
  base) and worst anywhere on fruitLF (38.9 vs base 66.7, −27.8). "JEPA = small uniform gain" is
  refuted; the aux helps or hurts depending on whether prediction targets align with the task's
  causal structure (candidate framing; not adopted — n=18-30 per cell).
- Serve-side gate-width signal (combo gate 2.2x wider) formally DISCOUNTED — did not predict
  outcome.

## H4 — Fast weights / supervision content

- **CLAIM (weakest):** Per-episode fast weights do not improve success under any tested recipe or
  normalization (Q2 ≤ its controls in all 4 tested conditions); richer supervision content
  (salient patches) is neutral-to-negative on both stats variants.
- Scope correction 2026-08-06: the salient evidence is RoboCasa-only — no ReMemBench labels of
  any spec ever existed (discovered during causal_v1 build). The causal_v1 arm will be the FIRST
  supervision-content cell on the memory benchmark; pilot generates salient+causal on the same
  demos so sparsity/content comparisons have a control.
- Sub-instruction supervision cell — PARKED (needs label regen).

## H5 — Locus of memory benefit (single vs multi-task)

- **Evidence 2026-08-05 (original 4 tasks, 24 cells, box):** the designed Δ-survival read is
  UNANSWERABLE on these tasks — confirmed empirically (3 no-soup-gap; oilsRL saturated: ALL arms
  100%, base rose 55.6→100 single-task, gap closed from below). fruitRF/fruitLF extension carries
  the read (1/12 trained).
- **CLAIM (weakest, provisional):** single-task training changes which mechanisms pay per task;
  two sign-consistent but individually-noisy signals NOT yet adopted as findings:
  (a) dnw8 > base on all 3 non-saturated tasks (+3.3/+6.1/+13.3; 3/3 sign test p≈0.125) despite
  zero per-task soup gap — would invert the disambiguation story if it replicates on fruit cells;
  (b) jw01k16 heatpot 33.3% (best cell in sweep) vs 0.0% soup Prospective — JEPA's prospective
  suppression looks like multi-task interference, not the objective (feeds H3).
- **DISCRIMINATOR RESOLVED 2026-08-05 (fruitRF): Δ SURVIVES AND AMPLIFIES.** Soup Δ(dnw8−base)
  +22.2 → single-task +50.0 (16.7 vs 66.7, n=18, ±~15). H5b (task-disambiguation) REFUTED as the
  sole account; H5a adopted in weakest form: *on the unsaturated spatial task with a soup gap,
  the memory advantage is within-task and is larger without soup interference.* Read "clearly
  survives," not "exactly doubles" (n=18).
- Supporting: dnw8>base sign-consistent 4/4 non-saturated tasks (p=0.125; design ceiling 5/5 =
  0.0625 — stays suggestive by construction). ALL mechanisms lift fruitRF single-task (tanh 50.0,
  jepa 61.1 vs base 16.7) while the soup showed near-nothing for them — further interference
  evidence feeding H3.
- Caveat: fruitRF window ordering non-monotone (dnw2 16.7 = base; dnw8 66.7; dnw16 50.0) — its
  memory demand needs w>2; connects to H2's task-dependent-window picture.
- **FINAL 2026-08-06: sign test 5/5** (fruitLF dnw8 77.8 > base 66.7; Δ+11.1 single-task vs soup
  Δ+11.1 — survives exactly), p=0.0625 = the design ceiling. fruitLF dnw16 88.9 (best cell);
  jepa 38.9 (below base — jepa's fruit benefit is RF-specific). H5a stands as adopted.
- **Competing hypotheses (discriminator QUEUED — now 36-cell sweep, p5 prio 1):**
  - H5a memory-within-task: deltanet-vs-base ordering survives single-task training.
  - H5b task-disambiguation: ordering collapses single-task.
  - Read is per-task ordering replicated across tasks; magnitudes not interpretable (n small,
    epoch counts differ across tasks).
- **Design correction 2026-08-05 (loop-caught):** the original 4 picks were made on per-CATEGORY
  deltas; at per-TASK granularity 3 of 4 have soup Δ(dnw8−base)=0 or negative and oilsRL is
  saturated (mechanism arms 9/9) — Δ-survival untestable on them. Extension: +2 spatial columns
  with a soup-Δ gradient — fruitRF (Δ+22, unsaturated: the discriminating cell) and fruitLF
  (Δ+11). Original 4 columns retained (within-task 6-arm orderings + window curve still
  informative; heatpot's negative soup Δ has its own read: does single-task training rescue
  deltanet's prospective deficit?).

## H6 — Normalization

- **CLAIM (weakest):** Pooling nav-dim stats over fixed-base-dominated data suppressed base-action
  scale 1.44x; correcting it raises plain-finetune success +2.4 (58.2 vs 55.8), concentrated in
  composite_seen (+4.4). Mechanism arms gained less than the baseline from the same fix.
- **Stronger variant outstanding:** "norm fix also changes rmb conclusions" — weak prior against
  (rmb tasks are uniformly navigation-heavy; pooling dilution minimal). Discriminator: rmb
  norm-split arm — PARKED (only if a reviewer forces it).

## H7 — Causal confusion in the memory channel (de Haan lens) — FULLY RESOLVED 2026-08-08

- **CLAIM (weakest):** none adopted yet. The window-curve decline (w2>w8>w16>w32→base) and the
  multi-task interference results are CONSISTENT WITH history-borne spurious correlates
  (causal confusion), but capacity/optimization explanations are not excluded.
- **Discriminators (QUEUED, causal-confusion wave on p5e):**
  - ω-history dropout (p=0.5, newest-never-masked) on w8/w16/w32: confusion ⇒ curve flattens /
    w16-w32 rescued; capacity ⇒ no change. Either outcome is a finding.
  - CFG on ReMemBench + serve-time guidance sweep: null-intervention training; guidance>1
    amplifies the causal effect of ω. Never previously run on the memory benchmark.
  - Causal keypatch labels (causal_v1: manipulated object + goal slot, sparser than salient):
    does supervising CAUSAL relevance succeed where salience was neutral-to-negative?
- Related evidence already in hand: Prospective peaks at w2 (history maximally spurious for
  timing); all mechanisms lift fruitRF single-task while soup shows nothing (interference).

### H7 resolution (2026-08-08, box, 264 rollouts/arm)
- (a) **Confusion CONFIRMED at long windows**: hist-dropout p=0.5 hurts w8 (36.8→34.1), rescues
  w16 (35.8→38.2) and w32 (32.7→34.8). The pre-registered sign pattern fired in the confusion
  direction; capacity account refuted for the decline's dominant component. NEW BEST rmb cell:
  dropout-w16 38.2 (and ObjAssoc 17.4, first arm to lift the floored category). Weakest claim:
  "on ReMemBench, random history deletion during training improves wider-window deltanet while
  hurting w8" — recipe implication (dropout as default at w>8) is a stronger variant, one cell
  from adoption — RESOLVED 2026-08-09: dropout-w16 ROBOCASA twin (s1-8d906977) = 58.1
  (75.8/50.3/46.1) vs n-baseline 58.2, dnw16 57.9 — a wash. The rescue is DEMAND-GATED: dropout
  repairs history-borne confusion where history is load-bearing (rmb +2.4) and is a free no-op
  where it isn't (rc −0.1, no cost unlike w8-rmb's −2.7). Recipe claim adopted in its weakest
  form: "wider window + history dropout" on memory-demanding benchmarks; harmless elsewhere.
- (b) CFG at conditional serve = +3.4 (34.7; Prospective 11.7 best-equal): the CFG interface
  transfers to memory demand at tanh-comparable magnitude. Dose-response not produced (s=1.0
  only; parked — robocasa sweep was flat).
- (c) causal_v1 = 35.1 (best aux-only; > jepa 33.9): causal-content supervision recovers
  deltanet-level Spatial (55.6 = the interface's best) with NO serve-time read; Prospective 1.7
  worst-anywhere (aux-family act-at-wrong-time). Salient-content rmb comparison never run
  (parked — would separate content from difficulty). H7 arc complete: history-borne confusion
  confirmed (a), and BOTH remedies work — intervene on history (dropout, 38.2) or supervise
  causal content (35.1 aux-only, spatial-parity).

## H8 — Backbone generality (GR00T N1.7) — RESOLVED 2026-08-09

- **ADOPTED (weakest sufficient):** (i) pi0.5 mechanism orderings do NOT replicate on GR00T at
  the aggregate — every mechanism hurts groot multi-task (dnw8 −8.1 w/ Spatial inverted;
  jw01k16 −3.9 w/ heatpot sign-flip; ttt −8.9); the paper's aggregate mechanism-value claims
  are SCOPED to pi0.5. (ii) The fruitRF within-task spatial-memory gain is the one
  cross-backbone-stable signal (all mechanisms help on both backbones). (iii) groot base 35.3 >
  pi base 31.3 entirely via ObjAssoc (24.4 vs 6.7) — the stronger backbone already encodes what
  mechanisms add on pi. (iv) RoboTTT does not pay on its native backbone (multi −8.9).
- Alternative account carried honestly: all groot mechanism hypers are pi-tuned (the H6
  staleness lesson, cross-backbone); discriminating sweep NOT commissioned (parked). Serve
  fidelity excluded as the account: groot dnw8 served cache-exact ω (bit-identical, §27) and
  still lost.
- Full 12-cell table + protocol: aug_07/groot_sync_validation.md §28-29. Evidence tier: box,
  same manifest/protocol as the pi rmb rows; single-task n=18-30 (orderings only).

## H6 addendum (2026-08-07) — AUDIT VERDICT: NO BUG; collapse explained, contest resolved
- Artifact evidence: all 8 n-wave ckpt norm-stats blobs byte-identical (sha 1b012d64...), all 10
  old twins byte-identical (3ebb5506...); diff = exactly dims 7/8/9 std x1.52/1.47/1.66 (+ means
  x2.4-2.8); env propagation verified per arm family; recipes differ ONLY in the knob.
- Per-task decomposition DECISIVE: mechanism regression is perfectly UNIFORM across mobile vs
  fixed-base tasks (pooled diff 0.00) — and uniformity is what a correct nav-split predicts
  (means moved for fixed-base tasks too + a global loss rebalance).
- Explanation (compound, evidence-ranked): (1) REGRESSION TO THE MEAN — old edge vs edge-lost
  r=−0.805; same-config replicate pair differs 29.9pp task-level; only salient's n-wave delta is
  >3σ; (2) HYPERPARAM STALENESS — hard evidence: w8/w16 optimum inverted under correction;
  (3) AUX:FLOW REBALANCE — flow loss −14.5% ⇒ fixed aux weights effectively +17%; predicts
  salient (w=1.0) worst and jw01 (λ=0.1) least-hit, as observed.
- Premise corrections: salient NEVER had an old edge (−0.88); q-arms did NOT collapse (q0
  improved). "Every mechanism lost its edge" overstated — the honest sentence: the old edges
  were partly noise, and the correction removed the shared headroom that inflated them.
- **Discriminators SUBMITTED (p5e @400, user-approved):** n-salient @ salient_weight 0.855
  (derived rebalance restoration; recovery ⇒ staleness, flat ⇒ no real salient effect) and
  n-dw32 (locate the corrected window optimum).
- 2026-08-08 RESOLVED: salient@0.855 = 56.5 (75.2/49.7/42.1) — recovers +1.2 of the 2.9
  deficit, still −1.7 under baseline, edge-of-noise; dw32 = 56.2 (75.2/48.4/42.5) — corrected
  window curve {w8 56.8, w16 57.9, w32 56.2} is an inverted U peaking at w16, ENTIRELY below
  baseline 58.2. VERDICT: staleness is a minor component; retuning rescues nothing. H1's
  RoboCasa refutation now stands WITHOUT the hyperparameter qualifier (tested grid: windows
  {8,16,32}, salient weights {1.0,0.855}). Discharged per loop step 7c — evidence, not fiat.
  Evidence rows: eval_results_final.md audit-discriminator table. H1's RoboCasa clause softens accordingly:
  "no mechanism beats corrected baseline AT OLD-STATS-TUNED HYPERPARAMETERS" — RESOLVED 2026-08-08:
  the re-tuned cells (56.5, 56.2) stay below baseline; the qualifier is DISCHARGED (see above).

## H9 — Test-time width scaling in the conditioning head (PTRM x GDN) — RESOLVED 2026-08-08

- **CLAIM (weakest sufficient):** none yet — pre-registered: "on RoboCasa multi-task at this
  scale/protocol, K noisy recursive-read trajectories + learned Q selection CHANGE success vs the
  same checkpoint's K=1 deterministic read." Ordering claim only; no magnitude, no benchmark
  transfer, no 'reasoning' language.
- Source: PTRM (arXiv:2605.19943) — inference-only width scaling on TRM via per-step latent
  Gaussian noise + ACT-Q-head selection. Port necessity: GDN is neither recursive nor Q-headed,
  and flow control has no exact-match bit → ONE train arm TRM-ifies the read (weight-tied
  refinement core T=4 + linear Q on z predicting per-sample flow loss, per-sample depth
  supervision), then PTRM is a pure eval knob. Design + correctness params:
  aug_07/ptrm_design.md.
- **Discriminators (one ckpt, ladder):** D0 offline Q-vs-loss corr + σ-dispersion (verifier real?
  σ* pick; Q flat ⇒ Maze-Hard-mode negative, documented); E0 K1σ0 vs dnw8 56.8 (recursion tax);
  E1 K32σ* best-Q (the claim); E2 K32σ* random-select (verifier vs noise-ensembling — E1≈E2
  kills the Q story even if E1>E0).
- Stronger variants parked (each = untested commitments): "scales like the paper" (needs K
  curve), "transfers to ReMemBench/GR00T" (needs those cells), "Q generalizes across tasks"
  (needs per-task D0 split).
- Known kill-modes pre-registered: Q uninformative / trajectory collapse under RMSNorm
  contraction / recursion tax E0<dnw8 beyond CI. Any is a publishable negative for deliberate
  test-time compute in the memory head; cross-benchmark claims unaffected.
- **RESOLUTION (sealed triple, 2026-08-08):** E0 58.1 / E1 best-Q 57.0 / E2 random 56.7.
  Adopted (weakest sufficient): (i) TRM-ifying the read is FREE on corrected RoboCasa
  (58.1 ≈ baseline 58.2; +1.3 over dnw8, edge-of-noise, ordering only); (ii) PTRM width scaling
  does NOT improve success at K=32 σ=0.3 (E1−E0 = −1.1); (iii) the Q head is NOT a functioning
  verifier (E1−E2 = +0.3 ≈ 0; converges with D0a's tail-picking null + negative Q-drift) — the
  transfer fails at the paper's own Maze-Hard joint. Mechanism: depth-inert recursion + no
  exact-correctness bit in flow control ⇒ E[loss|z] target never forced rank structure.
  Refuted stronger variants (kept): "width scaling transfers to control" (E1); "a loss-predicting
  Q is a usable verifier" (E1−E2 + D0a). D0b (box, resolved 2026-08-09): within-chunk Spearman
  Q↔realized loss −0.005 ± 0.256 AND oracle−random ≈ 0.3% relative ⇒ no selection headroom even
  for a perfect verifier; Q does track ACROSS-chunk difficulty (ρ=0.32) — a difficulty meter,
  not a verifier. Undischarged (parked): σ/K grid beyond {0.3, 32}; Q retrained with
  variance-reduced targets.
  Evidence rows: eval_results_final.md PTRM section; aug_08/ptrm_d0a_results.md.
- **ReMemBench twin (box, 264 rollouts/cell, 2026-08-09):** E0 38.2 / E1 38.2 / E2 37.6
  (cat-weighted). E1=E0 to 0.01pp with IDENTICAL pooled successes (105/264 both); E1−E2 = exactly
  1 rollout (SE ~3.0pp). Replicates the robocasa verdict on the memory-demanding benchmark:
  noise-ensembling buys nothing, Q-selection ≈ random. The inference-time machinery is inert on
  both benchmarks. Where the +6.9 over base LIVES was left to E3z (below) — and is NOT settled;
  the earlier "is training-side" reading is withdrawn as stronger than the evidence forces.
- **E3z zero-cond ablation (box, 264 rollouts, 2026-08-09) — the pre-registered dichotomy was
  FALSE.** Same ckpt, same protocol, K=1/σ=0, conditioning vector forced to 0 at the adaRMS seam
  AFTER the head runs: 36.74 pooled / **34.80 cat-weighted** vs E0's 39.77 / 38.20. That is
  neither pole: it did not collapse to ~31.3 and it is not ≈38.2. Paired on identical episodes,
  seeds and action-noise draws (fold_in, not split): 18 E0-wins vs 10 E3z-wins of 28 discordant,
  **McNemar exact p=0.185**, 95% cluster-bootstrap CI on the pooled difference
  **[−0.76, +7.20]pp** — the read's contribution is NOT resolved from zero at this sample size.
  Adopted (weakest sufficient): (i) the inference-time procedure is inert (E1−E0 = exactly 0
  net, 14/14 symmetric discordance — a strictly stronger null than E3z's 18/10); (ii) zeroing the
  read costs 3.0pp pooled / 3.4pp cat-weighted as a POINT estimate, bounded above by ~7pp and not
  distinguishable from 0 at n=264. Both "the read is load-bearing" and "the head is decorative"
  remain live; this cell as sized cannot separate them, which is itself the result.
  Discriminators (cheapest first): (a) DISCHARGED with no run — base IS measured under this
  manifest: arm s0-9e47bc75, cb24fe49, 264 rollouts, box, published (remembench_results_final.md).
  With base anchored at 31.3: E0−E3z = +3.4 and E3z−base = +3.5 cat-weighted — the point-estimate
  split is ~half serve-time read, ~half training-side shaping, neither leg individually resolved
  from zero at n=264. Suggestive (NOT adopted, n=60/cat): the zeroed read's cost concentrates in
  Prospective (11.7→1.7, 7→1 successes of 60) — E0's Prospective is best-of-any-read-arm and the
  read appears to carry it; (b) enlarge E0+E3z to ~10 rollouts/episode (~900 rollouts,
  ~3.4x, ≈2x4h) — from the 10.6% discordance rate, significance at the observed split needs ~47
  discordant pairs and 80% power needs ~96, vs the 28 observed; (c) parked/exploratory: the served
  read is tiny (trained head max|v|=0.0156, gate max|tanh α|=0.0066), so "load-bearing but
  under-scaled" predicts an α-scaling serve sweep would move the number (no retraining).
  Provenance: attempt 1 was killed by the label gate (self-test read `wsm_cond_window` off Pi0,
  which never stores it; a too-generous stub had hidden it; no rollouts ran); attempt 2 logged the
  zero-vector proof on 4/4 servers (trained head max|v|=0.0156 → served 0.0).
  Evidence rows: /data/work/remembench_evals/ptrm_rmb_{E0,E1,E2,E3zero}/results.json (box).
- **ENLARGEMENT RESOLVED (n=880/cell, 10 rollouts/ep, 2026-08-10): the read is NOT load-bearing.**
  E0_n880 37.8/36.3 vs E3z_n880 39.1/37.4 (pooled/cat): paired E0−E3z = **−1.25pp**, discordant
  36/47, McNemar p=0.27, 95% CI **[−3.07, +0.57]pp** — the n=264 (+3.03) and n=880 (−1.25)
  samples straddle zero and the larger CI brackets it. ADOPTED (weakest, forced by both samples):
  the serve-time read contributes ≈0 on rmb; PTRM's +6.9 over base is training-side shaping —
  the claim withdrawn on 08-09 returns, now evidence-forced rather than presumed. The α-scaling
  sweep (parked (c)) is moot: a read this close to zero has nothing to scale.
- **Curve RESOLVED (40k 34.9, 60k 33.8 cat): monotone DECLINE 38.2→36.4→34.9→33.8** (monotone
  chance 1/24). "PTRM needs more training" refuted on BOTH benchmarks (robocasa 120k 57.5 <
  dnw8-60k 59.9; rmb −4.4 over 4×). The study's 15k budget was on the right side of the peak.
- **Protocol lesson (replay check):** n880 idx0-2 does NOT bitwise-replay the sealed 264s despite
  0 noise-base mismatches — 23 (E0) / 21 (E3z) BIDIRECTIONAL episode flips. Mechanism: serve
  batch composition changes kernel numerics (the known B-dependent tap kernels), closed-loop
  chaos decorrelates episode outcomes. Implication: cross-run cell numbers carry ~±2pp
  serve-condition sensitivity ON TOP of sampling noise; only within-run paired contrasts and
  same-condition comparisons are clean. The 4-point curve and the arm table are same-condition;
  the n880 pair is internally paired — all stand.
- **120k default-stats cell (exploratory, 2026-08-09): "PTRM needs more flops" REFUTED on
  RoboCasa.** PTRM×deltanet at 2× budget (120k steps; two deltas — stats + budget; LR cosine
  horizon 100k, floor last 20k): 57.5 (74.4/51.4/44.6), complete 50/50×100. Sits +1.7 over
  baseline 55.8 but −2.4 under deltanet-w8 59.9 at HALF the budget. More training flops do not
  close the leaderboard gap. (Caveat: no 60k default-stats PTRM cell exists, so budget and stats
  are confounded within this row; the ordering vs dnw8 is the claim, not the magnitude.)

## H10 — Training-free KV compaction (Attention Matching) — TRANSFERRED to RoboMME lane 2026-08-10

- **TRANSFER (user lane rule: our scope = ReMemBench + RoboCasa):** no measurement was produced
  from this side (two p5e attempts, ~7 min total, both failed pre-measurement: env resolution;
  then an am_pack keyword bug caused by concurrent cross-lane edits to amkv/). All staged
  artifacts remain valid; the AM program including E0 continues as the RoboMME evidence ladder's
  step 4. Hand-off record: aug_10/am_kv_design.md. The host analysis stands as OUR contribution:
  our study arms have no history KV (AM moot on them) — the compressed-state design already
  avoids the cache AM exists to shrink.
- Original hypothesis text below, now the RoboMME lane's to test:

- **HYPOTHESIS (weakest form to test):** AM at 4–8× on valid causal-history KV of the
  FrameSamp+Modul teacher preserves the action velocity field (per flow time) and rollout success
  within noise; i.e., the raw history cache is redundant enough that training-free compression
  matches it.
- **FULL-MEMORY CALIBRATION SEALED:** released step-79999/seed-7 FrameSamp+Modul reaches
  368/800 = 46.0%, Wilson 95% [42.5737%, 49.4646%], with counting/permanence/reference/imitation
  69.5/27.0/35.5/52.0 versus the paper's 65.22/25.11/36.33/51.39 and 44.51 overall. All 16 tasks
  have exactly 50 Boolean outcomes. Relative to the released no-memory control at 19.125%, the
  protocol-matched separation is +26.875 points (2.405x, +140.5% relative). This validates the
  full-memory teacher endpoint; it is not AM evidence.
- Host decision + why our study arms are NOT a host (no history KV), why v1 workspace tokens are
  NOT a host (GDN read, no attention), and the v2-coordination boundary: aug_10/am_kv_design.md.
- Durable E1 plumbing includes an atomic ordered 18-layer stack receipt and authenticated server
  route; 41 foundation tests pass. Overlay v2 pins policy manifest
  `88fec4b85eea0407dd474b248af9405e6eba26f3d2d47be78d465f892c1bc664`, module
  `12e8112bf530121e7732da4caa241915eb041e0234db7d5746a45212e0f550d1`, patched-Gemma SHA prefix
  `3e3d7fd8…`, and source-tree SHA prefix `113c786e…`; 11 CPU tests and Ruff pass. This is mechanism
  plumbing, not outcome evidence.
- Current durable schema v2 is compact-all/R0: every on-policy replan requires a fresh artifact
  attested to the actual history at that cut. A demo-only receipt cannot be reused at later cuts;
  compact-old plus raw-recent would be a separate future partition.
- Remaining E1 P0 gates are a real online teacher-tap history producer/attestor and released-
  step-79999 full-action parity. The superseded overlay reordered the released
  `Q-RoPE → Q-scale → K-RoPE` sequence; `max_abs=0.0029296875`/453 nonzero action elements is its
  preserved pre-fix diagnostic, not a v2 result. V2 restores exact order and adds source regression,
  flushed phase logs, and parity-only mode. Its completed checkpoint-restore/JIT comparison is a
  definite FAIL-CLOSED: bitwise false, `max_abs=0.0048828125`,
  `mean_abs=0.0005074137588962913`, 476/640. The remaining suspect is graph-scan arity/collections;
  never relax exact parity. E1 was not submitted, and no AM rollout claim exists until both gates
  pass.
- Ladder: E0 offline velocity matching (fit chunk k, eval chunk k+1, both ratios, wall-clock
  table) → gate → E1 paired rollouts → gate → E2 compaction-aware finetune
  (L_flow + λ‖v_compact − v_full‖², AM frozen, no grads through selection/solves).
- E0 is a separate teacher-forced diagnostic under `robomme_integration/amkv/`, not a durable E1
  artifact route. Its staged inputs deep-verify as 25 S3 objects / 12,307,384,809 bytes, with only
  a 35,291-byte source receipt newly uploaded. The first user-approved ratios-4/8 p5e/H200 job
  `amkv-e0-569e7fbeb67ffc88` (manifest `9015a07d…`, service `59d7330f…`) failed its runtime identity
  gate when importing `e0_run` added 17 pycache entries. All eight upstream gates passed, but no
  model load/metrics/results occurred: it is infra-smoke, not evidence. The bytecode-disabled fix
  preserves sealed import `8920cb…`/227 and passes 129 tests. Corrected run
  `amkv-e0-a026fc5cd7275f32` (manifest
  `248a180152960c861476d15dd4b64b87f8b5f0582b142e48f252de0d0c69236c`, service
  `19b803b5-0960-4fa5-ab8e-0299ce3527b4`) completed and result `04e47626…` is
  RELEASED/VERIFIED with all 12 evidence checks true. Primary fresh 8x AM is 1.4193% pooled
  rel-Δv / 7.27% action drift; random-drop 8x is 3.2600% / 20.18%; destroyed memory is
  4.2517% / 26.52%; stale 8x reuse is 7.6488% / 60.01%. Verdict remains INDETERMINATE—not
  FAIL—because destroyed/AM=2.995545 is just below the pre-registered 3.0 control-scale cutoff.
  Interpret the stale collapse as evidence that refresh/routing is the next bottleneck; do not
  infer deployable E1 or rollout evidence from this teacher-forced oracle cell.
- Pre-registered readings in the design doc; the E3z serve-numerics lesson applies — all
  contrasts within-run paired, velocity deltas relative, gates on rel-Δv not absolute.

## H11 (PARKED, proposed 2026-08-10) — history-append arm + AM serve compaction, our lane

- The missing long-context-policy axis cell: pi0.5 rmb finetune with K raw history frames
  appended to the prefix (arm A), then the same ckpt served with AM 4x/8x on the appended
  history KV only (arm B). Yields the measured 3-way the cost argument needs: raw history vs
  AM-compacted history vs learned compressed ω, identical 264-rollout protocol.
- Cost: one p5e finetune (~15k) + two box cells + AM serve integration in OUR openpi fork
  (clean implementation, no robomme_integration imports, per the lane/reconciliation rule).
- Parked because: not user-greenlit yet; consumes no compute until then. Design choice to make
  first: frame count/stride vs context length.

## H12 (SUBMITTED / CAPACITY-WAITING 2026-08-10) — RoboMME mechanism replication

- **HYPOTHESIS (weakest):** at least one compressed-representation intervention yields a repeatable
  positive single-task effect on RoboMME after separating representation read, train-only auxiliary
  shaping, persistent fast weights `W_t`, and sequence-sampling nuisance. This is not yet a claim:
  the existing PickXtimes/ButtonUnmaskSwap representation checkpoints are unevaluated.
- **Factorial:** Q0/Q1/Q2/Q3 cross persistent `W_t` with tanh steering from workspace tokens
  `omega_t`. Fast-weight contrasts are Q2-Q0 and Q3-Q1; workspace-tanh contrasts are Q1-Q0 and
  Q3-Q2. Both halves must agree in sign before mechanism attribution.
- **Sequence discriminator:** existing Q0/Q2 independently sample `tau` and Gaussian action noise
  per flattened chunk. q0_noforce/q2_noforce broadcast the first stock `tau` draw across each
  contiguous L=8 window while leaving action noise independent. Their paired contrasts isolate
  shared-`tau`; they are not memory arms.
- **Representation discriminators:** GDN must beat both S0 and tanh; JEPA must beat S0; PTRM E0
  must beat its exact GDN-K8 parent; GDN+JEPA must beat `max(GDN-K8, JEPA-k1)`. PTRM remains
  deterministic K1/sigma0 because inference-time width/Q selection is already refuted elsewhere.
- **Promotion gate:** headline = both +30% relative and +5 percentage points absolute over the
  matched control. Fixed-50 screen = at least five additional paired successes, reported with
  Wilson intervals and paired McNemar counts/tests where maps match. This is an effect-size screen,
  not automatic statistical significance.
- **Progression:** PickXtimes dense anchor → MoveCube replication after its legitimate two-point
  workspace target is supported → PickHighlight negative control → all-16 only if at least two
  memory-demanding tasks are positive, pooled gain is at least +5 points, and the negative control
  is not harmed.
- **Current implementation/evidence state:** Wave 1
  (`p5e_pick_anchor_core_v3.json`) serializes fresh S0/Q0 anchors, Q1, PTRM E0, q0_noforce, and
  q2_noforce on one p5e/H200 node: 79,800 seconds estimated training plus 3,600 reserve,
  priority 400, <=24h, 300 GiB. The preemption-safe train path passes 79 focused tests. It was
  submitted exactly once as Batch service `09dbd7e0-79e6-46e7-8385-fee60fc0356a`; the latest
  readback is SCHEDULED with a Pending SageMaker child awaiting training-plan capacity. Submitted source
  `d2216a8d…`, campaign `rmme-st-series-v1-86321f09f2b755aaea85`, manifest `e095616c…`, and
  archived SageMaker source SHA-256 `0f0848b5…` are the retry provenance. GDN-K8+JEPA is production-wired through a canonical-ed923 config-only
  overlay with no model/loss-math change, but is deferred to Wave 2 until the individual screens
  justify composition. Bulk p5 eval is implemented but **NOT SUBMITTED** and requires a fresh
  source-bound standard-`ed923` native-EGL custom-benchmark preflight plus runtime receipt. The
  local two-RTX-5090 path is now source/runtime sealed: the exact SigLIP GCS generation was copied
  once and verified locally/AWS; native EGL passed 246/246 paired history; the resolver rejects the
  newer mismatched encoder and verifies the original Pick/Button omega/tree identities. The exact
  16-cell fixed-50 queue binds source `68709e15…`, preflight `948f460d…`, receipt `afd83569…`, and
  manifest `b3d52bb6…`; dry-finalization passed. It is not scored and requires explicit approval.
  Existing specialist inventory is 48/48 trained, 14/48 valid fixed-50.
- **Done check:** a common Pick episode map contains every promoted/control outcome; the winner
  repeats on MoveCube and does not harm PickHighlight; only then is an all-16 run eligible. Exact
  operational checklist: `aug_10/robomme_two_day_campaign.md`.

## H12 — RoboCerebra campaign (3rd benchmark, crossed axes) — OPEN 2026-08-10, at the G1 fork

- Integration complete (994/996 eps → LeRobot, sealed stats, harness ported; tree:
  aug_10/robocerebra_ablation_tree.md). G0 PASS: pi05_libero zero-shot floor = 1/100 episode,
  1/760 subtasks (sane actions; the benchmark simply demolishes reactive policies). Protocol
  finding: episode success partly ARTIFACTUAL under resume re-pins (1/10 episode with 0/70
  earned subtasks observed) — subtask completion is the primary metric.
- **G1 FAIL — the frozen robocasa ω encoder COLLAPSES cross-domain** (canonical 0883c9bd,
  2-view/128-token convention, matched-sampling in-domain control, bootstrap CIs):
  temporal-coherence gap 0.785 → 0.030, effective rank 10.9 → 6.2 (non-overlapping CIs),
  between-episode variance 0.441 → 0.043. Confounds ruled out: token RMS in-regime (0.916 vs
  ~0.97 trained); agentview-only indistinguishable (not the view convention). Label pilot NOT
  spent (localization from a collapsed code = noise).
- **FINDING (paper-relevant regardless of next step):** the workspace latent does not survive
  RoboCasa→LIBERO transfer even with matched token statistics — a real generality limit of the
  frozen-encoder design; mechanism claims are encoder-domain-scoped.
- FORK (user decision, pre-registered in the tree): (a) retrain-lite the encoder on RoboCerebra
  frames — cheapest variant is label-free SIGReg/JEPA-style on replay frames, validated against
  the now-quantified G1 metrics (in-domain reference: gap 0.785, rank ~11); full recipe adds the
  VLM label pass; (b) drop ω arms → campaign reduces to base + the negative-transfer finding.
  All ω-consuming arms (A1/A2/A4, A3's target) blocked until resolved.

## H13 — live/joint WSM supervision at RoboCasa post-train — **CLOSED 2026-08-19 (all readings negative or null)**

- **Claim:** live/joint WSM supervision during RoboCasa post-training helps. 2x2x2: ±LeJEPA align
  x ±language decode x ±gdn8, keypatch decode always on; aux-only (serve-identical, stripped
  ckpts). Plan/authority: `aug_12/h13_joint_wsm_tree.md`.
- **Pre-registered readings:** (a) R1 − base > +2pp => live decode aux shapes representations
  (train-side mechanism, PTRM-consistent); (b) R2 − R1 = latent-alignment marginal — if
  R2 ~ s3_jw01 (frozen targets), "live" adds nothing; (c) R3 − R1 and R4 − R2 = language marginal;
  (d) R5–R8 vs dnw8 59.9 = composition, with the pre-registered risk of redundancy (both mechanisms
  feed temporal context) so gains may not stack; (e) any arm > 5pp BELOW base with clean canary
  telemetry = an interference FINDING, not a bug hunt.
- **Kill criteria:** canary collapse (w eff-rank -> 1) or flow-loss starvation after one lambda
  adjustment.
- **Anchors (never re-run, pinned at P0.2/P0.5):** base 55.8, dnw8 59.9, s3_jw01 58.2; the nearest
  negative prior is the existing direct salient aux at 54.9 (−0.9), which R1 must separate from.
  Pins: `aug_12/h13_pins.md`.
- **RESULT (a), 2026-08-15 — NULL.** R1 `h13_dec` = **55.48** task-weighted (sealed
  `eval-h13a-step59999-2db75948794388cb`, complete; atomic 74.2 / comp-seen 46.2 / comp-unseen 43.7).
  R1 − base = **−0.34pp** against a pre-registered threshold of > +2pp; R1 − salient = +0.58pp.
  Live keypatch decode through a 512-d w bottleneck, with SIGReg and unblocked gradients into the
  backbone, does NOT move RoboCasa success. NOISE CAVEAT: single cross-run cell, no replicate, study
  spread ~±2pp — the defensible claim is `R1 ≈ base ≈ salient`, not an ordering; atomic ties base to
  the decimal. Reading (e) does not fire (this is inertness, not interference). Foreshadowed by the
  canary: R1's flow curve matched the s0 control at 9/10 points, and the build record stated at the
  time that parity establishes *no harm*, not a large effect. **(b)/(c)/(d) now carry the campaign**;
  being within-family contrasts they are less exposed to the cross-run noise limiting (a). H13
  remains OPEN on those.
- **RESULTS (b)(c)(d), 2026-08-16.** R2 54.74, R3 50.82, R4 48.78 (all sealed/complete, 5000
  rollouts). **(b) NULL/negative:** R2 − R1 = −0.74 and R2 − s3_jw01 = **−3.42**, i.e. a LIVE
  alignment target is worse than the frozen cached one at matched lambda/k. **(c)/(d) NEGATIVE:**
  language decode costs −4.66 (R3−R1) and −5.96 (R4−R2), consistent across all three splits and far
  outside the ~±2pp noise. **Reading (e) FIRES for R4 (−7.04 vs base)**, R3 on the boundary (−5.00):
  the pre-registered interference finding. Mechanism visible in canary telemetry: `lang_cos`
  saturated to 0.905 while normalised `lang_infonce` never beat chance (~1.0) — the head learned the
  caption CENTROID (distinct captions have mean pairwise cosine 0.841), so the aux spent capacity on
  a degenerate solution. **Net H13 verdict so far: live/joint WSM supervision at post-train does not
  help RoboCasa; the caption-alignment head AS IMPLEMENTED actively harms.** WEAKEST-CLAIM
  DISCIPLINE: (c)/(d) indict THIS head — InfoNCE-in-batch against a narrow-cone frozen-LLM embedding
  that let it satisfy the objective by predicting the centroid — NOT "language supervision hurts",
  which is untested; discriminator **R3b** (hard-negative / segment-classification target, or lower
  lambda_lang), not built. (b) indicts target LIVENESS at matched lambda/k, not latent alignment as
  such (the frozen arm is +2.34 over base); discriminator **R2b** (live, k=16), parked. Open: the
  gdn8 composition (R5/R6 submitted on C5 pass). **R7/R8 DROPPED as SUPERSEDED** (user, 2026-08-17):
  they carry the degenerate head. **(c)/(d) re-registered** on a fixed head — R3b/R4b (+ gdn8 twins
  R7b/R8b) swap InfoNCE-vs-embedding for **cross-entropy over the 8661-caption vocabulary**, which a
  centroid cannot satisfy; CE/ln V keeps the O(1) scale rule; frozen embedding demoted to telemetry.
  Gate: G4 is load-bearing — top-1 caption accuracy must beat BOTH chance (1/V) and the in-batch
  majority-class rate by end-of-canary, else HOLD (no lambda tuning to force it).
- **RESULT (d) for R5, 2026-08-18 — NEGATIVE, and it SUBTRACTS.** R5 `h13_dec_gdn8` = **53.04**
  (sealed, complete, 5000 rollouts, per_env_isolated_v1). vs dnw8 59.88 = **−6.84**; vs R1 55.48 =
  −2.44; vs base 55.82 = −2.78, with the damage concentrated in composite-unseen (38.44 vs 43.69).
  (d)'s pre-registered risk was that the two temporal-context mechanisms would not STACK; the
  observed effect is worse — the live decode aux DESTROYS 6.84pp of dnw8's existing gain and lands
  the arm below plain base. An aux that is ~inert alone (R1 ≈ base) is actively harmful composed with
  the history module. **R6 `h13_dec_jepa_gdn8` = 54.62** (sealed): vs dnw8 −5.26, vs R2 −0.12, vs base
  −1.20. **(d) NEGATIVE on BOTH gdn8 cells** — the aux destroys 5.3–6.8pp of dnw8's gain and both
  land below plain base, reproduced across two aux configurations, so it is a regularity not one
  cell. Noise caveat: single cell each, ~±2pp spread — magnitudes soft, signs safe.

## H12 RoboCerebra — RE-CLOSED 2026-08-22 under protocol v3 (corrected scoring). Verdict: bounded null, ~8x tighter than v2, with two arms detectably WORSE on the memory stratum.

Supersedes the 2026-08-17 closure, which was withdrawn 2026-08-18 for a scoring fault. The fault was
in the SCORER only — action and observation digests are identical pre/post fix, so v3 re-counts the
same rollouts. Acceptance evidence (digest invariance, expert oracle, verbatim authors'-scorer
shadow agreeing bit-for-bit, unit tests) in `aug_18/robocerebra_forensics.md`.

**Evidence.** 5 arms x 6 modes, 800 trials/arm, budget 15k, protocol v3; subtask completion primary;
paired per-case contrasts vs plain fine-tune under **common random numbers** (verified: 800/800
coordinates share a byte-identical initial observation across all five arms, and 0/800 share an
action trajectory — so the pairing is variance-reduced AND no arm is inert). Tables in
`eval_results_final.md`.

**Amendment 2026-08-23 — A5 (Stage-Q fast weights, test-time / RoboTTT proxy) folded, verdict
unchanged.** A5 ran at **reduced N: 80 trials, not 800**. Its launcher gave all 8 shard runners the
same `--wsm-env-id env0`; the server's duplicate-env guard rejected 7 of 8 on first inference and
the merger ran `--allow-partial` (`complete=False`, shards `[5]` and `[7]`). Sharding is by trial
index, so the survivors still cover all 6 modes x 10 cases at 1 trial per coordinate. The 80 that
ran are clean (the guard rejects every `env0` request in a duplicated window, so no contaminated
inference executed; each episode reset `W` to the meta-learned init). Contrast is against base
**restricted to the same 80 coordinates** — 80/80 identical initial observation, 0/80 identical
action trajectory.

| A5 vs matched base | rate | Δ [95% CI] |
|---|---|---|
| no-memory | 24.12% (55/228) vs 27.19% | -3.42 [-8.86, +2.03] |
| memory | 30.62% (169/552) vs 29.53% | +0.88 [-2.78, +4.55] |

Both intervals cover zero: **A5 ≈ base on both strata and refutes none of readings #1-#4; it adds
one more arm to #1's refutation** (no arm improves the memory stratum). It does **not** tighten the
bound — the reduced cell resolves only Δ > ±3.7 pp on the memory stratum, ~3x looser than the
full-cell ±1.15 pp, so the headline bound stays at ≈0.04x base from the five 800-trial arms.

*Mechanistic note, load-bearing for interpretation.* The A5 checkpoint **self-suppressed its
fast-weight mechanism during training** (|tanh α| = 2.4e-4, 4.4x below the 1e-3 init; read scale
~5e-5), yet the mechanism is **not inert at serve time** (0/8 identical action digests vs A0 in the
serve gates; 0/80 here). Any ≈base result is therefore a property of **training, not serving**.
This is the third instance of pi0.5 post-training muting an optional read pathway: PTRM read ≈0,
H13 gates quiet, A5 gate below init. That recurrence — not A5's null itself — is the transferable
finding.

### Verdict against the pre-registered readings

| pre-registered reading | outcome under v3 |
|---|---|
| **#1** Δ concentrated in Memory modes, ~0 on Ideal ⇒ double contingency generalizes | **REFUTED, and more sharply than under v2.** No arm improves the memory stratum. Δ_mem: A1 −1.69 [−3.01, −0.37], A2 +0.49 [−0.65, +1.62], A3 −0.70 [−1.56, +0.16], A4 −1.59 [−2.87, −0.32]. Two intervals exclude zero, both negative. |
| **#2** A2 > A1-family on Memory modes ⇒ causal-confusion remedy scales with horizon | **ORDERING NOW RESOLVED, MECHANISM READING NOT SUPPORTED.** v2 saw no ordering; v3 does: A2 − A1 on the memory stratum = **+2.17 pp (SE 0.90), 95% CI [+0.42, +3.93]**, separated. But A2 sits at base (+0.49, interval spans zero) while A1 sits below it (−1.69, interval excludes zero). History dropout therefore **removes the 8-frame window's deficit rather than adding memory benefit** — a repair, not a remedy that scales. |
| **#3** all arms ≈ base ⇒ protocol absorbs demand, or demand is instruction-mediated | **PARTIALLY REFUTED.** "≈ base" holds for A2 and A3 but fails for A1 and A4, both detectably below base on the memory stratum. The surviving claim is weaker than #3 as written. |
| **#4** Ideal-mode Δ>0 would contradict the RoboCasa no-demand null ⇒ treat as confound | **not triggered.** Ideal Δ: −2.75 (1.74), +0.61 (2.02), +0.22 (1.55), +0.71 (1.46) — nothing separated from zero. |

### Weakest claim the evidence entails

> On RoboCerebra, initialised from released pi05_libero and fine-tuned to 15k steps, **no workspace
> mechanism — recurrent ω read (w=8 or w=16+dropout), train-only JEPA auxiliary, or PTRM recursive
> head — produces a detectable memory-demand-specific gain, and two (ω read w=8, PTRM) are
> detectably worse on the memory stratum.** Bound: the paired design under common random numbers
> resolves Δ ≈ **1.15 pp at 95%** on a base memory rate of 30.72%, i.e. **≈0.04x the base rate**.

This is still a bounded null, but a far stronger one than v2's. v2 could only resolve Δ ≈ 0.30x its
base rate; v3 resolves ≈0.04x — roughly **8x tighter in relative terms** — because the corrected
scorer recovers ~20x more signal from the identical rollouts. The v2 closure's headline verdict
(no memory-specific gain) survives; its precision, and its claim that all arms were merely ≈ base,
do not.

### Level caveat — required whenever these numbers are quoted

The re-pin hands the policy the ground-truth state at every subtask boundary; for a `Place` subtask
the object is already grasped. First-segment completion — the only cell where no re-pin credit is
possible — is **7.0%**, against 26-52% for later segments. The metric measures "finish a subtask you
were handed mid-way". That is the authors' protocol and is comparable to their 7.84% Ideal, but the
4x gap to that published number is **not yet a claim** and needs an independent check before it is
made one.

### Stronger variants considered and refused

* *"Workspace mechanisms do not help long-horizon manipulation."* Refused — over-generalises from
  one benchmark and one init. The same mechanisms win on RoboCasa (+4.1) and ReMemBench (+6.9).
* *"The mechanisms actively hurt."* Still refused, but less comfortably than under v2: 2 of 10
  stratum-level intervals now exclude zero, both negative (A1, A4 on memory), with no multiplicity
  correction across 8 arm x stratum comparisons. The honest read remains null-to-slightly-negative.
* *"ω transfers but the benchmark is saturated."* Refused — but the v2 grounds for this ("base is at
  1.7-3.8%, nowhere near a ceiling") were a scorer artifact. Restated on v3 grounds: base is at
  26-31% against an open-loop demo-paced reference of 16-26%, so the benchmark is not saturated,
  though the re-pin makes the headline easier than "do the task".

### Metric divergence — WITHDRAWN as stated, replaced

The v2 closure recorded **`Memory_Execution` at 0/200 episode successes for every one of the five
arms** at its highest-in-benchmark subtask rate, and read that as episode success tracking episode
LENGTH rather than skill. **Under v3 the pattern vanishes**: MemExec episode success is 23-34/200
and MemExpl 61-68/200 across the five arms. The zero was produced by the missing
`simulate_resume_completion` — the goal pointer never finished advancing, so `all_done` could not
fire — not by episode length.

Replacement, weaker and protocol-level: **under a re-pin protocol, episode success is substantially
manufactured by the re-pins** (v3 advances the goal pointer at every boundary by construction), so
it is not a skill measure under either scorer. Subtask completion stays the primary metric — which
is what the pre-registration said, for a reason that turned out to be different from the one
recorded.

### Discriminators still open (cheapest first)

1. **Is it the encoder or the mechanism?** ω here is a domain-adapted retrain (G1b attempt 2), not
   the study's canonical encoder, which COLLAPSED on RoboCerebra (G1 FAIL: coherence gap
   0.785 → 0.030). Cheapest test: score the arms' ω-read layer against a shuffled-ω control.
2. **Budget — RESOLVED 2026-08-23, REFUTED as an explanation.** A0-long (`a0_base-5a2b7e82`),
   three SAME-RUN checkpoints, Ideal only, v3, CRN-paired (100/100 identical env inits across
   budgets, 100/100 distinct trajectories):

   | checkpoint | steps | epochs | Ideal subtask completion |
   |---|---|---|---|
   | a0_long@15k | 15,000 | 4.23 | 32.63% (248/760) |
   | a0_long@30k | 30,000 | 8.46 | 31.05% (236/760) |
   | a0_long@45k | 44,999 | 12.69 | 31.84% (242/760) |

   Paired: 30k−15k **−0.90** [−3.87, +2.08]; 45k−15k **−1.41** [−3.42, +0.60]; 45k−30k **−0.51**
   [−4.05, +3.02]. **No contrast separates from zero**; the trend is flat-to-slightly-negative.
   3x the compute (4.23 → 12.70 epochs) buys nothing, with a minimum detectable difference of
   ±2.01 pp. Under-training was ranked the #3 factor in the forensics; it is now **excluded** as
   the explanation for the level, at this resolution. Residual: the authors ran ~40 epochs, ~3.2x
   beyond our 45k, so a threshold above 12.7 epochs is not excluded — but a monotone budget effect
   is. Free reproducibility check: an independently trained 15k checkpoint (`a0_probe`) scored
   31.58% vs a0_long@15k's 32.63%, i.e. ~1 pp across separate training runs.
3. **Protocol absorption.** The 150-step timer plus GT re-pins may absorb the memory demand; a
   `resume=False` ablation separates this, at the cost of comparability with the paper.
4. **A5 / Stage-Q** is not yet scoreable: it needs its checkpoint pulled AND the
   `serve_pi05_libero_stageq.py` integration, without which the arm is inert by construction. The
   CRN check above is exactly the test for that failure mode — an inert A5 would share A0's action
   digest on shared coordinates.

* **Generality limit carried forward from G1:** the study's canonical ω encoder does NOT survive
  RoboCasa → LIBERO transfer (effective rank 10.9 → 6.2, between-episode variance 0.44 → 0.043).
  Mechanism claims in this study are encoder-domain-scoped.

## H14 — Deliberative Workspace Supervision (DWS), ENCODER STAGE — **COMPLETE 2026-08-31; policy stage open**

Authority: `aug_22/deliberative_workspace_plan.md` (+ amendments A11–A16); evidence
`aug_22/h14_p0_status.md` §§14–20. All rows are OFFLINE encoder metrics — no policy row exists yet,
so nothing here is a success-rate claim.

- **POLICY STAGE, 2026-09-02:** the RoboMME **E0 anchor is IN** — `v4_s0` fixed-800 = 143/800 = **17.875 %** [15.38, 20.68], Δ vs released base −1.25 pp [−5.05, +2.55] (bounded null, MDE 6.1), −28.1 pp vs teacher, all four suites inside MDE (§25.15 / CAMPAIGNS §W4 RESULT). It LICENSES only "the training+eval pipeline reproduces the published base rate"; it licenses **nothing** about workspace memory, DWS, or any mechanism — it is a base policy with no memory interface. The three rmb Stage-P arms that would have arbitrated the policy-level claim are SUPERSEDED and pre-registered NON-EVALUABLE (§25.8, unserveable encoder conditioning contract), so **H14 still has no policy-level evidence for or against DWS.**

- **CLAIM (weakest):** Training the workspace encoder on cross-task pairings mined by VLM
  deliberation over segment descriptors makes ω retrieve functionally matching moments **across
  tasks, on held-out episodes** far above chance — **16.25x mean lift (3/3 seeds, RoboCasa+rmb)**,
  11.83x (3/3 seeds, RoboCasa only) — and the **typed pairing STRUCTURE is what produces it**, not
  the presence of a contrastive term. Scope conditions: pi0.5 frozen pooled tap, 12k steps x batch
  64, the A1d disagreement subset of cross-task pairs, two domains. **No memory-content and no
  policy claim is entailed.**
- **Control battery (the reason the claim is structural).** Same objective, same budget, same seeds:

  | arm | what changes | mean retrieval lift | seeds |
  |---|---|---:|---|
  | E1b | deliberative labels, both domains | **16.25** (16.08 / 14.45 / 18.23) | 3 |
  | ctrl-1Db | same labels, one domain | 10.66 | 3 |
  | ctrl-Eb | positives from text-embedding top-k, SAME hard negatives | 8.07 | 3 |
  | ctrl-0b | λ_del = 0 (no deliberative term) | **0.97 — AT CHANCE** (one run scored 0.00) | 3 |
  | ctrl-S | same edges, type-preserving rewire | **1.13 — BELOW chance**, coherence collapses 0.86→0.478 | 1 |
  | ctrl-T | positives restricted to same task | **0.20 — BELOW chance** | 1 |

  ctrl-S/ctrl-T are single-domain, single-seed (§14.9) and are never quoted as like-for-like against
  the multi-domain arms. Chance = 1x; ctrl-1Db/ctrl-S/ctrl-T score on 188 anchors at chance 0.0084,
  the multi-domain arms on 319 at 0.0062.
- **Pre-registered contrasts and outcomes** (paired by seed on the Wilson-95 LB of top-1;
  all-same-sign criterion fixed before results, §20.1):

  | contrast | mean Δ | all same sign | verdict |
  |---|---:|---|---|
  | E1b − ctrl-0b (the deliberative term) | **+0.0892** | YES (3/3 +) | the term is the whole effect; ~17x separation, MDE n=1 |
  | E1b − ctrl-Eb (Qwen vs embedding positives), multi-domain | +0.0488 | YES (3/3 +) | corroborating, NOT independent — §20.6 confound |
  | E1b − ctrl-Eb, single-domain (§18.4) | **+0.0618** | YES (3/3 +) | **the clean version**: 11.83x vs 4.03x, MDE n=2, ran 3 |
  | E1b − ctrl-1Db (domain mixing), as pre-registered | +0.0129 | YES (3/3 +) | **artifact — withdrawn**, see defect note |
  | E1b(rc→rc) − ctrl-1Db (like-for-like) | **−0.0116** | NO (+,−,−) | adding rmb does NOT improve RoboCasa retrieval |
  | E1b-analog05 − E1b (ANALOGOUS at 0.5) | +0.0055 | NO (−,+,−) | **NULL; the v1 "+8.5" headline is WITHDRAWN** (MDE 360 seeds/arm) |

- **Caveats that must travel with every number above.**
  (a) **Seed spread is the error statement**: same-config runs differ by ~5 lift units (E1 7.996 vs
  E1-seed2 12.93); per-arm lift SD 1.89 (E1b) / 1.00 (ctrl-Eb) / 0.25 (ctrl-1Db). Any within-family
  Δ smaller than that is noise, which is how analog05 died.
  (b) **ctrl-1Db baseline defect, recorded not papered over**: the all-same-sign criterion was
  pre-registered on the WHOLE-gate statistic, but ctrl-1Db is the one arm that changes the gate
  population (188 anchors @ 0.0084 vs 319 @ 0.0062), so the whole-gate Δ is not a comparison of
  encoders — it measures that E1b can retrieve rmb↔rmb pairs at all. The correct pre-registration
  would have named the rc→rc stratum. The domain-mixing question is **NOT resolved** and at n=39
  seeds for the observed effect is not worth resolving by adding seeds.
  (c) **§20's Qwen-vs-embedding gap is composition-confounded**: ctrl-Eb's embedding-mined positives
  cross the domain boundary 4.5x less often at artifact level (0.0324 vs 0.1464) and 7.5x less
  in-batch (0.0239 vs 0.1784). §18 (single-domain, both arms structurally at zero cross-domain) is
  the clean read; §20 corroborates it.
  (d) **G1b/bevf/eff-rank are validity floors, never selection metrics** — they anti-correlate with
  retrieval across 21 cells, and ctrl-0b PASSES G1b on both domains on 3/3 seeds while retrieving at
  or below chance. Selecting on them would have picked the inert control every time.
- **Label-quality evidence the claim rests on** (A9/A12/A13, §15, §14.8): EQUIVALENT precision
  **0.933** [0.841, 0.974] (F1 PASS) under blind adjudication of 240 edges; low-vs-medium reasoning
  effort κ **0.838** (F4 PASS — no re-judge bought); CONTRAST precision 0.172 (F2 FAIL) and planted
  probe recovery 0.533 (F3 FAIL) ⇒ the pre-registered HOLD, which bites on POLICY spend not the
  local funnel. Fixed programmatically, not by re-prompting: a deterministic per-episode **binding
  table** flags hard negatives at **0.94 precision** against the memory-intent rule, and
  Qwen ∪ binding recovers 39/45 planted probes (0.867) ⇒ **F3 cleared by the union**. Pass-1 corpus
  **complete: 3,873 / 3,873 episodes, 19,853 segments, 0 schema-invalid, 0 truncated** across three
  domains. Owed and NOT run: a pass-2 delta for the 52 topped-up RoboMME episodes / 217 segments —
  it must mint a NEW edge_store_id, and no cell trained so far is affected.
- **DECODABILITY LINE — CLOSED NEGATIVE** (A15/A16; §14.8, §17.4.1, §19.4, §19.5). Three label
  classes, three misses, and the negatives are load-bearing:

  | probe | result |
  |---|---|
  | RoboCasa bound slots, before vs after reveal | `before` tracks `after` on every slot ⇒ **perception** (a knob, a food item, a recycling layout are visible from frame 0). 357/750 episodes have an empty before-window, so the populations differ too |
  | rmb "hidden" sides (`return_side`, `olive_side`) | the **raw frozen per-frame tap** decodes them at USE time (0.664 / 0.588, Wilson LB above chance) ⇒ **perception, both slots DROPPED** by the rule registered before the numbers existed |
  | PROGRESS STATE, 8 families, 107k frames, pooled-linear (ridge + nearest-centroid) | **0/8 pass**; no source clears the time-only baseline; `untrained` ≥ E1b in 6/8; `ctrl-0b` worst in 7/8 |
  | same labels, causal GRU-64 sequence read-out (the form the GDN read takes) | **0/8 pass**; E1b below the time-only probe in 8/8, ≤ its own label-shuffled POSITIONAL floor in 4/8 |

  The progress class is the **first clean perception control** — `raw_tap · frame` ≤ time-only in
  8/8, so the label is genuinely memory-shaped and the failure is about ω, not about the label.
  **H_absent survives** (ω as trained carries no progress state a history read can use);
  **H_nonlinear is unsupported** but bounded only **in this sample regime** — 48–120 train
  episodes/fold against a 512-d input, and the GRU scores BELOW the ridge for E1b in 5/8, i.e. the
  recurrent probe overfits. Mechanism reading: progress lives in EVENTS, and no term in
  {JEPA, SIGReg, edge SupCon} asks ω to mark them.
- **Stronger variants considered and REFUSED** (each with its discriminator and status):

  | refused claim | why refused | discriminator | status |
  |---|---|---|---|
  | "DWS Markovianizes the task / ω is a sufficient statistic" | no decodability probe supports it (4 rows above) | policy-level memory-stratified eval on E1b-ω → GDN read, cluster-blocked design, SCP now lifted | **SUBMITTED / in flight** |
  | "the deliberative labels carry the bound variable" | binding decodability splits **4–1**: ctrl-Eb beats E1b on `CuttingToolSelection/cut_food` after-reveal on 3/3 seeds | per-slot statement only; no cheap discriminator | parked (report per slot, never pooled) |
  | "ω carries progress state, the probe was too weak" | GRU-64 loses to a 2-d clock in 8/8 (§19.5) | **E3 = event-marked ω** — per-frame segment-boundary/subskill-completion target from the frozen segmentation (free labels, no LLM), then re-run §19/§19.5 | queued, NOT run |
  | "mixing domains improves the encoder" | like-for-like Δ is mixed-sign, mean −0.0116 | n=39 seeds/arm for the observed effect | **refused as not worth resolving** |
  | "down-weighting ANALOGOUS positives helps" | n=3 paired null, mixed signs | 360 seeds/arm | **WITHDRAWN** |
  | "Qwen beats embedding positives independently in both corpora" | §20's arms differ in cross-domain positive rate as well as positive source | a `ctrl-Eb` variant mined to MATCH E1b's cross-domain rate | parked; §18 stands as the clean result |
  | "CONTRAST removal helps" (v1 canary reading) | inside seed spread | — | **WITHDRAWN** |
- **STATUS.** Encoder stage **COMPLETE** — 12-cell single-domain funnel + 9-cell seed replication +
  12-cell multi-domain funnel, all local on 5090s; the p5 package (`32f56f0402f59856`) is validated
  and now unblocked but buys nothing the local runs have not produced. The **Markovianization claim
  is CONDITIONED on policy-level memory-stratified evaluation** (cluster-blocked design, SCP deny
  lifted 2026-08-31; policy arms submitted). Until those land, the paper may claim the
  **structure result (C2-style) and NOT a memory-content result**.
