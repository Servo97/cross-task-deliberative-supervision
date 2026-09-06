# E — RoboMME base maturity curve (A19), sealed 2026-09-06

Source: status §57. Campaign `a19-m0-70k-milestones-fixed50-p5-parallel-v1`, run
`mt-v4-70k-all16-v4_s0-seed0-28d80fb948f834df`, execute-10 lane (`p5_fixed50_vla_eval_predict20_execute10_v1`,
model seed 7), 50 episodes × 16 tasks per milestone. NOT comparable to the paper-protocol anchor (17.875 %, h20/e16).

| step | succ/800 | % | Wilson 95 % |
|---|---|---|---|
| 10000 | 96 | 12.00 | 9.93–14.44 |
| 20000 | 103 | 12.88 | 10.73–15.37 |
| 30000 | 113 | 14.12 | 11.88–16.71 |
| 40000 | 121 | 15.12 | 12.81–17.77 |
| 50000 | 134 | 16.75 | 14.32–19.50 |
| 60000 | 132 | 16.50 | 14.09–19.23 |
| 69999 | 115 | 14.38 | 12.11–16.98 |

Readings: +4.75 pp 10k→50k (≈2.6 SE of a two-point difference), plateau 50–60k, −2.4 pp at 69999 (n.s.).
A19 rule at MDE 5 pp → s* = 10000 (degenerate); 2 pp → 40000; 2 SE (3.7 pp) → 30000; argmax 50000.
Per-task grid: `E-rmme-base-curve-a19.json`. Decision on the common arm step: pending (user), see §57.
