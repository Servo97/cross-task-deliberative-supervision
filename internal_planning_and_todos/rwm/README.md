# Research world model — H14 cross-task deliberative workspace supervision (DWS)

One file per hypothesis node (`hypotheses/`), sealed evidence summaries (`evidence/`), the
experiment that would move each open node (`todos/`), and a live board (`board.md`). Protocol:
`PROTOCOL.md` (project-localized from the global `research-world-model` skill). Authorities this
directory points at, never duplicates: `aug_22/deliberative_workspace_plan.md` (design + amendments
A1–A19), `aug_22/h14_p0_status.md` (§§14–51, all measurements), `hypothesis_ledger.md` (H14 row),
`PAPER_STATE.md`, `sep_02/robomme_wsm_parity.md`, `robomme_integration/CAMPAIGNS.md`.

## The tree (2026-09-03)

| id | node (weakest form of the claim) | status | evidence | moves it |
|---|---|---|---|---|
| [H14](hypotheses/H14.md) | Cross-task deliberative supervision trains ω to be the missing sufficient statistic; the GDN read over ω-history Markovianizes memory-demanding tasks for π0.5 | **OPEN at policy level**; encoder stage sealed as a *structure* result | E-encoder-stage, E-anchors | every policy arm below |
| [H14.1](hypotheses/H14.1.md) | The typed pairing STRUCTURE produces cross-task retrieval, not the presence of a contrastive term (C2, offline) | **CONFIRMED** (offline; 3 seeds) | E-encoder-stage | policy-level version = H14.5 with ctrl-0b arms |
| [H14.2](hypotheses/H14.2.md) | Deliberation-mined positives beat text-embedding positives with the same hard negatives ("is Qwen worth it") | **CONFIRMED** (offline; +0.062, 3/3 seeds) | E-encoder-stage | judge-effort question = H14.8 |
| [H14.3](hypotheses/H14.3.md) | Mixing domains improves the encoder | **REFUSED** — like-for-like Δ −0.012 mixed sign; n≈39 seeds to resolve | E-encoder-stage | not worth resolving |
| [H14.4](hypotheses/H14.4.md) | ω carries memory content a history read can use (bound slots, hidden sides, progress state) | **CLOSED-NEGATIVE** → H_absent; layout = perception | E-decodability | E3 event-marked ω (T-e3) |
| [H14.5](hypotheses/H14.5.md) | C1: the deliberative cross-domain encoder ≥ per-domain / structure-free encoder under the identical GDN read, on ≥2 of 3 benchmarks' memory suites; C3: gains concentrate where memory is demanded | **OPEN** — no policy-level evidence for or against | E-anchors | T-policy-arms (rmb P′, rcb R, RoboMME M) |
| [H14.6](hypotheses/H14.6.md) | User: low-data benchmarks (RoboCerebra, ReMemBench) gain most from cross-task deliberative ω via GDN conditioning | **OPEN** | E-anchors (rcb H12 null is the comparator) | T-policy-arms (R1−R2 headline) |
| [H14.7](hypotheses/H14.7.md) | User: the demo shown at the start can be learned by workspace tokens; [demo ω ; live ω] → GDN → AdaLN gives FrameSamp+Modul-style gains on RoboMME | **OPEN**; two pre-GPU defects fixed; one paper finding (zero-init GDN decay) | E-lessons, E-anchors | T-policy-arms (M-arms), tap refire (T-tap) |
| [H14.8](hypotheses/H14.8.md) | User: judge at max reasoning effort yields better labels → better model | **OPEN, unmeasured** — both pilots failed on harness defects (09-03); recommendation: park at low | E-labels (κ 0.838 low/medium) | T-judge-effort |
| [H14.9](hypotheses/H14.9.md) | User: checkpoints must be trained to saturation (50–70k); select by evaluated curve, never by a wall-clock rule | **OPEN** — rmb + RoboMME base curves LANDED (milestones in S3, 09-03); rcb resubmitted after a quota rejection; evals pending GPU1 / venue decision | E-anchors (rcb 15k≈30k prior) | T-maturity |
| H14.10 | C4 (stretch): K-token workspace > 1-token at fixed signal | **NOT STARTED** (E2, gated on H14.5) | — | — |
| [H14.11](hypotheses/H14.11.md) | Constraint: train-time conditioning must be a statistic the server computes causally at rollout | **CONFIRMED as a failure mode** (P1/P2/P3 retired); now a design rule | E-lessons | — |
| H14.12 | Label quality is measured, not assumed (EQUIV 0.933; CONTRAST via binding table 0.94) | **CONFIRMED / handled** | E-labels | H14.8 |

## How to read a status
CONFIRMED carries its scope in the node file (offline metric ≠ success rate). CLOSED-NEGATIVE
carries an MDE. OPEN nodes have a `todos/T-*.md` with the exact READY lines and kill criteria.

## Live state
`board.md` (jobs, watchers, blocked-on, next ≤5 actions). Rewritten at every status report.
