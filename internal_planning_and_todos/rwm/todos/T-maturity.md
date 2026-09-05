# T — Checkpoint-maturity curves (moves H14.9; de-confounds H14.5–H14.7)

| benchmark | base job (queued p5 @400) | milestones | eval per milestone | venue / cost | selection |
|---|---|---|---|---|---|
| RoboMME | `mt-v4-70k-…-28d80fb948f834df` (92beb42e…), 86,150 s | 10k…60k, 69999 | fixed-800 paper protocol h20/e16 | local `project_exact` runner ≈1.5 h/eval (p5 fixed-50 lanes run execute-10 → not the paper universe; port DEFERRED) | base curve → s*; paper protocol also reports mean of last 3 |
| ReMemBench | `s0-6ab9621b9d58b326` (b33d75ea…), 86,400 s | 15k, 30k, 45k, 59999 | 264 rollouts | local 2×5090 ≈2.7 h/eval → 4 evals ≈11 h base; 16 with the ω arms ≈43 h | base curve → s* |
| RoboCerebra | `a0_base-169c383cda9d32a9` (e2e28599…), 172,800 s | 15k, 30k, 45k, 59999 | curve cell (Ideal 10×10, K=8, CRN) then full v3 at s* | local: 0.66 h base / 3.0 h ω per curve cell; full v3 6.7 h base / 25.4 h ω → ≈42 h on two lanes for base+R1+R2 | base curve → s* (prior s* = 15k) |

Watchers: `scratchpad/watch_job.sh {RMME_M0_70K, RMB_BASE60K, RCB_BASE60K_48H} <arn>` print status
transitions and `<LABEL>_STEP_MS` from CloudWatch once RUNNING.

**Decision 2026-09-04 00:45Z (user: "p5 now"):** RoboMME milestone curve is scored on p5 under the
execute-10 lane for SELECTION only (internally consistent; not comparable to the 17.875 anchor); the
chosen step(s) get paper-protocol (h20/e16) scoring locally once GPU1 is back. Chain: preflight
`p5-native-eval-v1-4dda9bf2f82aa472cd0a` (queued @100, frozen snapshot of `robomme_integration/`) →
claim → campaign of 112 fixed-50 cells (`scratchpad/fire_rmme_campaign.sh`, priority 100, ≤28 node-h
at the admission budget) → per-milestone success → base curve → s*.

Deferred builds (fallbacks exist): (a) paper-protocol h20/e16 lane on the p5 fleet (needs a runtime
bundle with benchmark commit 856bc3…; the p5 bundle ships f2b540e6) — fallback local runner; (b)
serve-side Stage-E ω consumer for rmb evals — required before any P′ eval, no fallback; (c)
`run_cell_local.sh` parameterization (~10 lines).

Order of local-lane work after pass-2 lands: rmb base curve (needs nothing new) → rcb base curve
cells → RoboMME base curve (when M0-70k lands) → ω arms as they land.

Rule: the reported step is chosen on the base curve and applied to every arm; per-arm best is a
flagged secondary. A base curve still rising at the last milestone = raise the budget first.
