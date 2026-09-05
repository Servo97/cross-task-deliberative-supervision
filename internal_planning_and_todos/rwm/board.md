# Board — 2026-09-05 19:55Z — HAND-OFF (TRI SageMaker via the lead only; project home is github.com/Servo97/cross-task-deliberative-supervision)

## Hand-off state
| item | state | where it lives now |
|---|---|---|
| **TRI AWS** | account access ending 09-04 (user). SageMaker/S3 unusable from now on. Salvage COMPLETE 02:17Z 09-05 (241 GB total) (`wsm_data/s3_salvage/pull_salvage.sh`, log `pull_salvage.log`): rcb tap store, deliberation artifacts, manifests, evaluations, RoboMME M0-70k milestones (87 GB) | local disk |
| rmb base-60k `s0-6ab9621b9d58b326` | TRAINED; **all four milestones local** (47 GB). Evals NOT run: local CUDA died (§55.5) | `wsmv2_scratch/rmb60k/ckpt/` |
| RoboMME M0-70k `28d80fb9…` | TRAINED; **all 7 milestones salvaged locally 21:31Z** (deploy/10000…60000, 69999; 84 GB); execute-10 milestone evals never fired (launcher backlog guard) | `wsm_data/s3_salvage/…/mt-v4-70k-…-28d80fb948f834df/deploy/` |
| RoboMME preflight | **FIRED 19:49Z 09-05 at priority 400** (user instruction): `p5-native-eval-v1-fa05c92950e9717361a5`, arn `…/5c56263d-35e1-4141-a0be-0bc1841770aa`, 14,400 s, RUNNABLE; launchers now `ALLOWED_PRIORITIES=(100,400)` (repo commit `d15d1b0`), snapshot re-synced (source tree `08a0f88d…`). On its claim → 112-cell milestone campaign (READY to rebuild from the persisted queue JSON) | — |
| rcb base-60k `a7cf2047…` | **SUCCEEDED 00:08Z 09-05** (≈25 h at 400); milestones 15000/30000/45000/59999 **salvaged locally 02:17Z** | `s3_salvage/…/robocerebra/checkpoints/pi05/a0_base/a0_base-a7cf20474a789a40/` |
| Stage-E #3 `772597789979f88a` | **LANDED 01:05Z 09-05** — all 10 cells trained (job flagged FAILED by a false path assertion, fixed). E1b 30–39× vs ctrl-0b 0.6–6.7× (3/3 seeds); ctrl-Eb 29–30×. Only E1b s28 `109e99680ca5c198` passes G1b on all taps; E1b-4tap FAILs robocasa eff-rank → RoboMME M-arms blocked by the predicate (user call). §56 | encoders + ω stores pulled to `s3_salvage/…/stage_e/772597789979f88a/` |
| `<V2C>` labels `91461e8df6f2d143`, 4 tap stores | sealed, local | `wsm_data/deliberation/stage_e_labels/`, `wsm_data/wsm_pooled/`, `s3_salvage/` |
| local box | GPU1 hardware-flaky (dies under load → CUDA dead box-wide until reboot). GPU0 only, `JAX_PLATFORMS=cuda` | — |
| RoboTok (arXiv 2609.03199) | sniffed 09-04: kinematic hand-trajectory retrieval trained to imitate DTW; eval GT = same DTW top-20 (circular); no semantics, no VLM baseline, no code. Proposed CPU-only DTW baseline vs our labels on 28,722 segments — user has not said go | — |

## Blocked-on chain
Stage-E #3 SUCCEEDED → retrieval gates → D7 `--lang-mode stored` parity → ω arms (rmb P′ needs the serve-side ω consumer build; rcb R1/R2 need the `TPY` fix; RoboMME M-arms need the preflight claim) → milestone evals (H14.9) → readings.

## Next ≤5 actions
1. Stage-E #3 lands (≈1 h after start) → gates + D7 parity chain.
2. p5 queue drains → submit the RoboMME preflight → claim → fire the 112-cell campaign (script to recreate under `ready/`; queue JSON persists).
3. rmb base curve cells 1→4 land → A19 selection on the base curve.
4. rcb base lands ≈00:30Z → curve cells locally (after the rmb lane frees, or interleave — user call on lane order).
5. Judge effort parked at LOW unless told otherwise.

(previous board below)

## In flight

| job | what | status | since | watcher |
|---|---|---|---|---|
| ~~local GPU0 · pass-2~~ | **DONE 01:07Z** — 8,869/8,869 buckets, 0 failed, 0 truncated (shard 0 45,837 s; shard 1 resume 28,230 s); S3 mirror 8,869 objects | complete | — | **executor DONE 01:45Z**: store validated (§53.1: 8,869 valid, one (model, effort), S3 mirror name-identical), `<V2C>` `91461e8df6f2d143` uploaded, robomme eff-rank 7.82 → bar 6.26/7.70, Stage-E dry-run `5fe2556ba063477a` |
| p5 @400 `…rcerebra-a0-base-a7cf2047…` (48 h) | RoboCerebra base-60k curve (#3) | **RUNNING** since ≈23:20Z; 1,500 ms/step (steps 0–800) → 60k ≈ 25 h → lands ≈00:30Z 09-05 | 09-03 17:12Z | `RCB_BASE60K_48H_2` |
| ~~executor~~ phase-2 settings module | **DONE 00:55Z** — `wsm_settings.py` + guardrails re-exports, 84 files edited, dry-run manifests identical except source digests, test set unchanged (986 passed), `tests/test_wsm_settings.py` 10/10; one node-bundle bug caught by an existing test and fixed (E0 bundle now ships the module) — `sep_03/code_hygiene_audit.md` §5 | — | — |
| p5 @100 preflight `p5-native-eval-v1-4dda9bf2f82aa472cd0a` (09c50a26…) | RoboMME eval-lane preflight on a frozen snapshot of `robomme_integration/` (`wsm_data/wsmv2_scratch/rmme_eval_snapshot_0904`) — USER DECISION 00:45Z: score the M0-70k milestones on p5 under execute-10 for selection; paper-protocol scoring of the chosen steps locally later | QUEUED | 00:47Z | `RMME_PREFLIGHT` → then `scratchpad/fire_rmme_campaign.sh` (112 cells, queue filled from the sealed M0-70k manifest + claim) |

## Landed since the 17:20Z board

| run | result | artefacts / next |
|---|---|---|
| RoboMME tap (`rmme-stage-tap-0903-170832`) | **SUCCEEDED** | encoder ω store `robomme/stage/wsm_pooled/rmme_pi_100k` → stratified eff-rank → `<RMME TAP PREFIX>` for the Stage-E READY |
| RoboMME tapserve (`rmme-stage-tapserve-0903-170901`) | **SUCCEEDED** | serve-aligned policy ω store (1,600 episodes) → ω export once E1b-4tap exists |
| ReMemBench base-60k | SUCCEEDED (09-03 08:53Z) | 4 milestones in S3; evals wait on GPU1 |
| RoboMME M0-70k | SUCCEEDED (09-03 11:29Z) | 7 milestones in S3; eval venue decision open |
| code hygiene phase 1/1b | DONE | ruff clean (0 findings on repo files), 461 files formatted, 976 tests pass; 28 fail + 14 errors = env-missing modules and sealed-identity pins (`sep_03/code_hygiene_audit.md` §3b) |
| docs | `ONBOARDING.md` approved 09-04 → mentee share link https://claude.ai/claude-code/onboard/ERF-5GZUZKMB (re-upload with the ShareOnboardingGuide tool after edits); `PROJECT_NARRATIVE.md` (documentation source); `sep_03/code_hygiene_audit.md` |

## Failed (disposition)
PILOT-1/PILOT-2 judge pilots — harness defects, no measurement (T-judge-effort; recommendation: park at low).
rcb base-60k #2 — quota rejection at admission (10 × p5 account cap) → #3 running.

## Local machine
GPU1 dead since 09-02 20:03Z ("Unknown Error"); needs `sudo nvidia-smi -r -i 1` or reboot. Until then no
2-GPU evals (RoboMME fixed-800 runner, rmb 2-server lane). The judge resume on GPU0 checkpoints per bucket.

## Blocked-on chain
~~pass-2 lands → `<V2C>` chain + rmme eff-rank~~ DONE (§53) → **Stage-E 10 cells `5fe2556ba063477a` (p5 @400, READY)** → D7
parity → ω arms (rmb P′ · rcb R · RoboMME M) → milestone evals (H14.9) → readings.

## Next ≤5 actions
1. **Stage-E FIRED 09-04 ≈01:50Z** — run_id `5fe2556ba063477a`, 10 cells (8 three-tap + E1b-4tap / ctrl-0b-4tap), `<V2C>` = `91461e8df6f2d143`, robomme tap + bar 6.26/7.70, p5 @400, 21,600 s (`scratchpad/fire_stage_e.sh`; status §53.4–53.5). On SUCCEEDED: retrieval gates per cell → D7 `--lang-mode stored` parity on every exported ω store → ω arms.
2. USER: reboot for GPU1 (safe now: nothing local is running; resume with `claude --continue`). RoboMME eval venue DECIDED (p5 execute-10 for selection; preflight queued). `ONBOARDING.md` published: https://claude.ai/claude-code/onboard/ERF-5GZUZKMB.
3. Phase 2 (settings module) DONE; phase 3 (archive `scripts/ec2/`, shell/yaml literals) is a user call.
4. rcb base curve lands ≈00:30Z 09-05 → curve cells (local, after GPU1).
5. Judge effort: parked at low unless the user says otherwise.
