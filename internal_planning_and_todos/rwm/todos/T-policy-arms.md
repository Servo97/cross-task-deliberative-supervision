# T — Policy arms (moves H14.5, H14.6, H14.7; the Markovianization arbiter)

Chain (each step gated on the previous; coordinator fires, executors never submit):

| # | step | state | READY / runbook | cost |
|---|---|---|---|---|
| 1 | local rcb pass-2 delta (NVFP4 @12,288) | **DONE 09-04 01:07Z** — 8,869/8,869, 0 failed/truncated; homogeneity one (model, effort); A1a AUC 0.823 | store `62fdafc322025fee` | local (GPU0 alone after GPU1 died) |
| 2 | `<V2C>` label chain: 3-way union → v2 → v2b → ctrl-Eb labels; validator + A1a/A1c floors | **DONE** — union 28,722 segments (9,708/1,333/8,812/8,869); v2 `f610b2226f91169c` → v2b `ce68cd05fd55c32b` → **`<V2C>` = `91461e8df6f2d143`**; cross-task-or-domain 0.484, cross-domain 0.15 | status §53.2 | CPU |
| 3 | Stage E, 10 cells (8 three-tap + E1b-4tap / ctrl-0b-4tap second wave), serve-consistent `--lang-mode serve` | **FIRED 09-04** (dry-run run_id `5fe2556ba063477a`; `scratchpad/fire_stage_e.sh`) | status §53.4; p5 @400, 21,600 s | 1 node × ~1 h |
| 3b | RoboMME tap + tapserve (4th domain + policy ω store) | **DONE 09-03** — eff-rank 7.82 [7.57, 8.06] → bar 6.26; `<RMME TAP PREFIX>` = `…/robomme/stage/wsm_pooled/rmme_pi_100k` | status §53.3 | — |
| 4 | D7 parity `--lang-mode stored` on every exported ω store (cos ≥ 0.999, `encoder_step` match) | after 3 | `stage_e_omega_parity.py` | CPU/GPU minutes |
| 5a | rmb P1′ (E1b s28) / P2′ (ctrl-0b s28) / P3′ (E1b s29), 60k milestones | gated on 4 + serve-side ω consumer (NOT BUILT, deferred) | status §22.9 + §50.7 | 3 × ≤9 h p5 |
| 5b | rcb R1 (E1b-ω) / R2 (ctrl-0b-ω), 60k, save 15k, 48 h @400 | gated on 4; ω artefacts content-addressed `omega/features/<sha>.tar`, `omega/encoder/<sha>.pt` | status §51 (variant 48 h) | 2 × ~24 h p5 |
| 5c | RoboMME M1/M2/M3/M3-ctrl, recipe v4_70k, −4.0 decay init | gated on 3b + 4 + prefix-ablation canary | parity A6–A11, status §49.6(b) | 4 × ~14 h p5 |
| 6 | evals: rmb 264-rollout local (2.7 h/arm/ckpt); rcb v3 curve cells + full 800 (0.66–3.0 h / 6.7–25.4 h); RoboMME fixed-800 paper protocol (local 1.5 h/eval; p5 port deferred) | after 5 + H14.9 milestone selection | T-maturity.md | local lane ≈85 h + RoboMME |
| 7 | readings: P1′−P2′ paired memory-stratified (MDE 7.4); R1−R2 & R1−base CRN (≈4 pp); M3−M1, M3−M3-ctrl (5 pp); interference rule >5 pp | | ledger + PAPER_STATE update via the weakest-hypothesis loop | |

Kill criteria: retrieval gate below chance at the Stage-E canary (kills the encoder); validity
predicate FAIL on any domain; D7 FAIL blocks that arm; any arm >5 pp below its anchor →
interference finding (still reported).
