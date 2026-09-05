# E — Encoder stage (SEALED 2026-08-31)

Authority: `hypothesis_ledger.md` H14 row; `aug_22/h14_p0_status.md` §§14, 18, 20; figure
`figures/presentation/14_h14_encoder_stage.png` (recomputed from sealed run dirs). Protocol:
offline top-1 retrieval on the A1d disagreement subset of cross-task pairs, held-out episodes;
π0.5 frozen pooled tap (`wsm_pooled/pi_100k`, rmb `rmb_pi_100k`); 12k steps × batch 64; local
5090s. Lift = hit rate / chance; chance 0.0062 (319 anchors, multi-domain) or 0.0084 (188, single).

| arm | manipulation | lift (mean; per seed) | seeds |
|---|---|---:|---|
| E1b | deliberative labels v2b, both domains | 16.25 (16.08 / 14.45 / 18.23) | 3 |
| E1b (RoboCasa only) | same, single domain | 11.83 | 3 |
| ctrl-1Db | same labels, one domain | 10.66 | 3 |
| ctrl-Eb | embedding-mined positives, same hard negatives | 8.07 (single-domain 4.03) | 3 |
| ctrl-0b | λ_del = 0 | 0.97 (one run 0.00) | 3 |
| ctrl-S | type-preserving rewire of the same edges | 1.13; temporal coherence 0.86 → 0.478 | 1 |
| ctrl-T | same-task positives only | 0.20 | 1 |
| E1b-analog05 | ANALOGOUS weight 0.5 | Δ vs E1b +0.0055, signs (−,+,−) | 3 |

Pre-registered paired contrasts (Wilson-95 LB of top-1; all-same-sign criterion fixed before results):

| contrast | mean Δ | 3/3 same sign | verdict |
|---|---:|---|---|
| E1b − ctrl-0b | +0.0892 | yes | the deliberative term is the whole effect |
| E1b − ctrl-Eb (multi-domain) | +0.0488 | yes | corroborating; composition-confounded (§20.6) |
| E1b − ctrl-Eb (single-domain, §18.4) | +0.0618 | yes | clean result, MDE n=2 |
| E1b − ctrl-1Db (whole gate) | +0.0129 | yes | artifact — withdrawn (gate population differs) |
| E1b(rc→rc) − ctrl-1Db | −0.0116 | no | domain mixing does not help RoboCasa |
| E1b-analog05 − E1b | +0.0055 | no | NULL (MDE 360 seeds/arm) |

Caveats that travel with every number: seed spread ≈5 lift units within a config (per-arm lift SD
1.89 E1b / 1.00 ctrl-Eb / 0.25 ctrl-1Db); ctrl-S/ctrl-T single-domain single-seed; G1b / eff-rank /
bevf are validity floors, never selection metrics (anti-correlate with retrieval across 21 cells;
ctrl-0b passes G1b on both domains 3/3 seeds at chance retrieval). Serve-consistent 4-domain
retrain (2026-09 cells E1b-4tap / ctrl-0b-4tap etc.) will produce a NEW sealed table; this one
stays the reference for the structure claim.

Untrained-encoder note: passes 2/3 legacy G1b bars (near-identity AdaLN init) → G1b = collapse
floor only; retrieval gate = the go/no-go. Per-domain G1b bars for the retrain = 0.80× stratified
raw-tap eff-rank: robocasa 10.121 → 8.10, rmb 7.47 → 5.98, robocerebra 4.50 → 3.60; the sealed rmb
5.90 reproduces under neither protocol (provenance unknown — never cite as measured).

## 2026-09-05 addendum — serve-consistent retrain on `<V2C>` (§56)
10 cells landed (job FAILED only on a path assertion; fixed). Retrieval lift: E1b 35.3/39.3/30.5, ctrl-0b 0.6/1.0/6.7, ctrl-Eb 29.8/29.3, E1b-4tap 35.4, ctrl-0b-4tap 7.0. G1b: only E1b s20260828 `109e99680ca5c198` has no FAIL (rmb/robocasa INDET, rcb PASS); E1b-4tap FAILs robocasa eff-rank 6.9 < 8.1 → RoboMME M-arms blocked under the pre-registered predicate. Encoders + ω stores local at `wsm_data/s3_salvage/…/stage_e/772597789979f88a/`.
