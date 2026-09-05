# E — Benchmark anchors and sealed comparators (what every policy arm is measured against)

| benchmark | protocol | anchor / comparator | value | note |
|---|---|---|---|---|
| RoboMME | fixed-800 = 16 tasks × 50 fixed test indices, predict 20 / execute 16, max 1,300 steps, model_seed 7; code id `robomme-paper856-h20-e16-fixed50-project-v1` | our base `v4_s0` (60k, E0 anchor, 2026-09-02) | **143/800 = 17.875 %** [15.38, 20.68] | Δ vs released base −1.25 pp [−5.05, +2.55] bounded null (MDE 6.1); all 4 suites inside MDE; 33 renderer recycles |
| RoboMME | same | released π0.5 (sealed) | 153/800 = 19.125 % | unpaired vs ours (different seeding) — never McNemar |
| RoboMME | same | released FrameSamp+Modul teacher | 368/800 = 46.00 % | the headroom; Permanence is its weakest suite (+9 pp) |
| RoboMME suites (n=200) | | v4_s0: counting 24.0 / permanence 23.5 / reference 12.5 / imitation 11.5 % | | vs sealed base 27.0 / 18.0 / 19.5 / 12.0 — all bounded nulls (MDE 9.5–12.8) |
| ReMemBench | sealed 264-rollout lane: 88 held-out episodes × 3, 13 Mem tasks, per-task horizon 1,000–3,200 steps | base 31.3 / dnw8 36.8 / dropout-w16 38.2 | | per-domain ω already helps here; paired MDE 7.4 pp; evaluated locally only (2.7 h/arm on 2×5090) |
| RoboCerebra | protocol v3 (authors' scorer port), 800 trials/arm = 4 modes × 10 cases × 10 + 2 memory modes × 10 × 20; subtask completion PRIMARY | H12 table re-scored 2026-08-22 | no arm beats base on the memory stratum; gdn8/ptrm worse; bounded null 0.04× base | level caveat: re-pin = "finish a handed-over subtask" (first-segment 7.0 %) |
| RoboCerebra budget | 15k vs 30k (G3 probe, 2026-08-14) | subtask completion 1.58 % vs 1.32 % | Δ −0.26 pp | 15k chosen for the H12 pairing |
| RoboCerebra budget curve | A0-long, v3 CRN, Ideal 10×10 | 32.63 / 31.05 / 31.84 % at 15k/30k/45k | paired Δ −0.90, −1.41 pp (se 1.4–1.5); MDE₈₀ ≈ 4 pp | s* = 15k under the A19 rule so far |
| RoboCasa (reference) | H13 closed 2026-08-19 | live/joint WSM supervision helps nothing; composition with history read −5 to −7 pp | | interference rule origin (>5 pp below anchor) |

Pipeline-calibration statement (ledger): the RoboMME anchor licenses only "training + eval
reproduce the published base rate"; it licenses nothing about memory or DWS.
