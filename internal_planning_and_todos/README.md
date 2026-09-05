# Long-Context Robot Policies — project front door

## The framing (paper scope)

Three orthogonal axes, deliberately not conflated:

| axis | definition | can exist without the others because |
|---|---|---|
| Long-HORIZON task | many control steps, semantically distinct subtask stages | a long task can stay ~Markovian (current obs suffices) |
| Long-CONTEXT policy | the policy RECEIVES history (tokens/window/state) | a policy can receive 1000s of history tokens and ignore them |
| MEMORY-dependent task | current obs is INSUFFICIENT: correct action depends on earlier events (Kaelbling '98 POMDP sense) | the demand is a property of the task, not the policy |

**Long-Context Reasoning (LCR), operational definition:** holding the current observation and
instruction fixed, does changing a task-relevant event in the preceding trajectory change the
policy's output? Benchmarks are placed by DEMAND (RoboCasa ≈ none, ReMemBench/RoboMME = yes);
mechanisms are placed by what they do with history.

Why not just append history frames: cost, redundancy, and causal confusion — policies latch onto
incidental historical detail (de Haan '19), errors compound in closed loop. The design goal is a
COMPRESSED task-relevant state, cheap to query at deployment. The completed v1 experiments use a
single 512-d workspace latent ω per sampled frame; the RoboMME audit shows that this mean-pooled
representation is much smaller than the successful 512-token spatial-temporal teacher. The v2
target is therefore a multi-token workspace whose per-layer action-side K/V and attention-mass
bias preserve the teacher's output and denominator. Steering and persistent fast weights remain
separate axes layered on only after that representation passes its oracle compression ladder.

## The mechanism axes (what we ablate)

1. **Steering** (inference-time read of ω): tanh single-token · CFG · gated-DeltaNet recurrent
   read over a w-window (+ train-only history-dropout, the causal-confusion intervention)
   (+ PTRM recursive-refinement head with a loss-predicting Q).
2. **Training-time aux** (no inference read): JEPA future-ω prediction (λ, k) · salient-keypatch
   · causal-keypatch supervision.
3. **Fast weights**: RoboTTT test-time adaptation (2×2 with its recipe control).

One factor at a time; sealed content-addressed study store; completion claims are the only
completion authority; heldout-reset evals with pinned episode manifests; final checkpoint only.

## Headline results (details: the two evidence files)

- RoboCasa (no memory demand): default recipe deltanet-w8 59.9 vs base 55.8 (leaderboard rows);
  corrected normalization: NO mechanism beats plain finetune (58.2) — retuning doesn't rescue.
- ReMemBench (memory demand): every read mechanism pays; best cells deltanet-w16+dropout 38.2
  and PTRM-head 38.2 vs base 31.3; causal-confusion arc closed (dropout sign pattern + two
  remedies); PTRM test-time compute fails at the verifier, its training-time head is the gain.
- Cross-backbone (GR00T): aggregate transfer FAILS (every mechanism hurts); only fruitRF
  within-task memory is backbone-stable. Claims scoped to pi0.5.
- The paper's central line: mechanism value tracks the task's memory DEMAND and the BACKBONE —
  a double contingency, not a mechanism ranking.

## File map (read in this order)

| file | what |
|---|---|
| eval_results_final.md | ALL final numbers, tables only |
| remembench_results_final.md | ReMemBench evidence + protocol + caveats |
| hypothesis_ledger.md | H1–H12: weakest-form claims, discriminators, refutations (Bennett loop) |
| PAPER_STATE.md | chronological snapshots for the paper-writing chat |
| aug_10/robomme_two_day_campaign.md | current RoboMME mechanism matrix, one-node train/eval campaign, scale gates, handoff |
| aug_07/groot_sync_validation.md | GR00T H8: design §1–21, results §28–29, infra lessons §22–27 |
| aug_07/ptrm_design.md | PTRM H9: honest port design, correctness params, ladder |
| HOW_TO_sagemaker_and_ec2.md | shared infra pool (queues, plan-ARN, box, priorities) |
| 01/02/03_*.md, 1x_*.md | early protocol/findings/plan docs (stable) |
| _archive/ | dated execution logs (audit) + the RoboMME agent's governing/status docs (jul_21) — contents unchanged |
| ROBOMME_ALIGNMENT.md | standing instructions aligning the RoboMME agent's ablations/framing with this study |

## Documentation closure rule

Every active research/status write-up must leave a **Future-self unblock** section. It must name
every unresolved cell or operational dependency, the exact next local/cloud action, required
artifact or approval, and an objective done check. Do the cheap local plumbing and validation now
instead of leaving vague labels such as `port_required`; distinguish `implemented but awaiting
artifact/compute` from `research implementation required`. Explicitly list parked work and the
reason it should not consume compute. When a result seals, update both its evidence table and this
unblock list in the same change so stale TODOs cannot survive completed work.

## Live state (2026-08-22)

H14 DWS ACTIVE — descriptors → typed edges → offline cross-task contrastive encoder → frozen-ω +
GDN read unchanged; method card + plan: `aug_22/deliberative_workspace_plan.md`. H13 CLOSED
(all arms ≤ base; composition subtracts). RoboCerebra rescored under protocol v3 (scorer fixed).

Study H1–H9 resolved and the PTRM extension is complete, including all six single-task cells.
PTRM verdict: inference-time compute inert; serve-time read ≈0 (n=880 ablation); gain is
training-side; more training HURTS on both benchmarks (curve declines 38.2→33.8).

H12 RoboCerebra: 5-arm post-train table RUNNING on p5e (A0 v1 + A1–A4 v2 after the missing_regex
fix; eval plumbing PASS ⇒ G2 closed). Authority: `aug_10/robocerebra_ablation_tree.md`.
H13 OPENED 2026-08-12: 8-run 2×2×2 live-WSM-supervision study on RoboCasa (live enc-dec aux ×
LeJEPA alignment × language decode × gdn8), user-greenlit for p5 @400. Execution authority:
`aug_12/h13_joint_wsm_tree.md`. Failure-mode video/metric study: box pipeline autonomous; viewer
via `wsm_data/failure_modes/serve_viewer.sh`.

RoboMME calibration: the paper-pinned released pi0.5 step-79999/seed-7 control is sealed at
153/800 = 19.125% versus the paper's 17.93% nine-cell report. Released FrameSamp+Modul is now
also sealed under the same h20/e16 harness at **368/800 = 46.0%** (Wilson 95% CI
[42.5737%, 49.4646%]): counting 69.5%, permanence 27.0%, reference 35.5%, imitation 52.0%, versus
the paper's 65.22/25.11/36.33/51.39 and 44.51% overall. All 16 tasks have exactly 50 Boolean
outcomes. Against the protocol-matched released no-memory control, this is +26.875 points
(2.405x, +140.5% relative), closely tracking the paper's 44.51% versus 17.93% separation.

The durable FrameSamp + Attention-Matching route now pins overlay-v2 policy manifest
`88fec4b85eea0407dd474b248af9405e6eba26f3d2d47be78d465f892c1bc664`, module
`12e8112bf530121e7732da4caa241915eb041e0234db7d5746a45212e0f550d1`, patched-Gemma SHA prefix
`3e3d7fd8…`, and source-tree SHA prefix `113c786e…`, plus an atomic ordered 18-layer stack receipt
and authenticated routing. The existing foundation passes 41 focused tests; the v2 ordering repair
passes 11 CPU tests and Ruff. Its sealed schema-v2 oracle is compact-all with a genuine zero-length
recent block: every on-policy
replan needs a fresh artifact attested to that exact causal history, and one demonstration artifact
must never be reused at later cuts. The completed feasibility audit shows that offline receipts
cannot cover unseen later on-policy cuts: compact-all/R0 needs a synchronous snapshot, two full
tapped teacher denoise trajectories, 18 fits, fixed-M masked stack, and compact serve at every
replan in one episode/process—roughly 3x neural work plus fits. It is an ephemeral causal oracle,
not a deployable speed recipe; do not persist every stack.

The deployable recommendation is separately versioned B1: compact a fixed demo once, append a
fixed-shape masked raw-live region at `beta_AM=0`, and use one softmax denominator. B1 first needs an
uncompressed control because freezing demo selection and physical RoPE slots departs from official
FrameSamp's uniform resampling of the whole growing prefix. The current producer is all-layer
teacher-query AM, not the paper's sequential layer-on-policy variant. The superseded overlay had
reordered the released `Q-RoPE → Q-scale → K-RoPE` sequence; its preserved pre-fix diagnostic was
`max_abs=0.0029296875` with 453 nonzero action elements. Overlay v2 restores exact order and adds a
source regression, flushed phase logs, and parity-only mode. The completed checkpoint-restore/JIT
comparison is a definite **FAIL-CLOSED**: `bitwise_equal=false`, `max_abs=0.0048828125`,
`mean_abs=0.0005074137588962913`, and 476/640 action elements differ. E1 was therefore not
submitted. The remaining suspect is graph-scan arity/collections; diagnose and repair it, but never
relax exact parity. The sealed 46.0% unmodified FrameSamp evaluation is unaffected and does not
close this gate.

H10 E0 is a separate offline, teacher-forced diagnostic. Its S3 inputs were deep-verified as 25
objects / 12,307,384,809 bytes; the only new upload was a 35,291-byte source receipt. At 18:21 UTC,
the user-approved priority-400 wave submitted three unique p5/H100 PickXtimes WSM cells and one
p5e/H200 AMKV 4x/8x cell (an exact 75/25 hardware split). The old AMKV service failed at its runtime
scientific-identity gate before model load: importing `e0_run` added 17 `__pycache__` entries. All
eight H200/artifact/environment/JAX gates had passed, but it produced no metrics or result. Preserve
that attempt as infra-smoke. The fix sets `PYTHONDONTWRITEBYTECODE=1` and invokes `python -B`; sealed
archive import remains `8920cb…`/227 and the full suite passes 129 tests.

Corrected priority-400 p5e/H200 run `amkv-e0-a026fc5cd7275f32` / service `19b803b5…` **SUCCEEDED**.
Manifest `248a180152960c861476d15dd4b64b87f8b5f0582b142e48f252de0d0c69236c` and result SHA prefix
`04e476…` are **RELEASED / VERIFIED** with all 12 evidence checks passing. At the preregistered 8×
primary arm, pooled rel-Δv is 1.4193%, worst-flow mean rel-Δv is 1.9352%, and free-running relative
action error is 7.27%; random-drop8 is 3.26% / 20.18%, destroyed memory 4.25% / 26.52%, stale AM8
7.65% / 60.01%, and AM8-f1 1.61% / 6.65% (rel-Δv / action). The threshold verdict is deliberately
**INDETERMINATE**: destroyed/AM is 2.995545, just below the preregistered 3.0 scale floor. Interpret
this as an insufficient control scale, not compression failure. KV storage is exactly 8× smaller;
the Python-unrolled timings are slower and explicitly nonclaimable. E0 is an oracle diagnostic of
compressibility, not deployable E1, and it does not clear E1's separate overlay-parity gate. Exact
old/new receipts live in `ROBOMME_ALIGNMENT.md` and `aug_10/am_kv_design.md`.

RoboMME H12 Wave 1 is now **submitted once and waiting for p5e capacity**. The commissioned specialist
inventory is 48/48 trained and only 14/48 fixed-50 evaluated; 16 trained representation cells on
PickXtimes/ButtonUnmaskSwap are still unevaluated. Wave 1
(`p5e_pick_anchor_core_v3.json`) serializes fresh S0/Q0 anchors, Q1, PTRM E0, and the two
shared-`tau` Q0/Q2 controls on one p5e/H200: 79,800 seconds estimated training plus a 3,600-second
reserve, priority 400, at most 24 hours, 300 GiB. The train lifecycle fix is audited and **79
focused tests pass**. The submitted source tree is `d2216a8d…`; campaign
`rmme-st-series-v1-86321f09f2b755aaea85` / manifest `e095616c…`, and the exact campaign plus six
run prefixes were empty in S3 before submission. Batch service
`09dbd7e0-79e6-46e7-8385-fee60fc0356a` is SCHEDULED; its SageMaker child is Pending and explicitly
awaiting training-plan capacity, with no container start. The exact unopened submitted tarball is
preserved locally at SHA-256 `0f0848b5…`. GDN+JEPA's
canonical-ed923 overlay is complete but deferred to Wave 2. Existing approved p5 GDN services are
separate: d16 has a sealed step-19,999 final and awaits eval; d8-drop is RUNNABLE and d16-drop is
SCHEDULED/Pending for capacity.

Bulk p5 evaluation is implemented but stays fail-closed until the source-frozen standard-`ed923`
native-EGL rendered-reset preflight proves the custom benchmark and paired nonempty/equal-length
`video_history`/`video_state_history`, then seals its exact runtime receipt. The local two-RTX-5090
adapter is now operationally sealed for the first 16 trained Pick/Button cells, 50 episodes each
(800 total). The exact 1,659,216,368-byte SigLIP object was copied once from pinned GCS generation
`1785853172203742`, verified as SHA `f16e9312…`, and conditionally repaired in AWS. Native-EGL
preflight passed 246/246 paired video/state frames. Queue construction authenticates the original
representations through the training-pinned omega manifests and exact real checkpoint trees while
rejecting the newer mismatched claim. The sealed queue has internal manifest SHA `b3d52bb6…`, binds
source `68709e15…`, preflight `948f460d…`, and runtime receipt `afd83569…`; dry-finalization passed.
No scored eval is running; it still requires explicit launch approval.
Scale Pick → MoveCube → PickHighlight negative control → all16; headline
promotion requires both +30% relative and +5pp absolute, and all16 additionally requires
replication on two memory-demanding tasks with no negative-control harm. Exact definitions,
status, and checklist: `aug_10/robomme_two_day_campaign.md`.

### Future-self unblock

- Resolve the three exact existing GDN service states through verified completion/failure claims;
  do not duplicate live suffixes.
- Preserve the exact submitted Wave-1 tarball/source identities; do not rebuild or resubmit while
  service `09dbd7e0…` is live. Any retry must use the preserved tarball or an explicitly new identity.
- Preserve the verified SigLIP generation/SHA, legacy chain (`dd5a17e4…` Pick, `b9ce…` Button),
  source/preflight/runtime receipts, and sealed 16-cell queue. Never regenerate or substitute these
  identities casually. Ask explicitly before starting its 800 scored episodes;
  retain the source-matched p5 preflight/campaign as the separately approved fallback/next lane.
- Score existing checkpoints before broad new training. Keep paper R1 distinct from project Q2,
  and fix MoveCube's two-point/dense representation route before workspace replication.
- When scores seal, update the result table, H12, PAPER_STATE, alignment doc, and the Aug 10
  checklist together.
