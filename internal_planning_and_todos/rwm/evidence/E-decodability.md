# E — Decodability of memory content in ω (SEALED 2026-08-29, CLOSED NEGATIVE)

Authority: `aug_22/h14_p0_status.md` §14.8, §17.4.1, §19.4, §19.5; plan A15/A16. Rule registered
before the numbers: a label decodable from the RAW frozen per-frame tap at use time is perception,
not memory, and is dropped.

| probe | population | read-out | result | classification |
|---|---|---|---|---|
| RoboCasa bound slots (knob, food item, recycling layout), before vs after reveal | 750 episodes; 357 have an empty before-window | linear | `before` tracks `after` on every slot | perception |
| rmb hidden sides `return_side` / `olive_side` | rmb demos | linear on raw tap | 0.664 / 0.588, Wilson LB above chance at use time | perception → slots DROPPED |
| binding (A13) per slot | 4 slots | linear, before/after | splits 4–1 between E1b and ctrl-Eb (ctrl-Eb wins `CuttingToolSelection/cut_food` after-reveal 3/3 seeds) | report per slot only, never pooled |
| progress state, 8 families, 107k frames | pooled ω, all cells | ridge + nearest-centroid vs normalized-time baseline | 0/8 beat time-only; `untrained` ≥ E1b 6/8; `ctrl-0b` worst 7/8 | absent |
| progress state, same labels | causal GRU-64 (the GDN's form) | sequence | 0/8; E1b below time-only 8/8; ≤ label-shuffled positional floor 4/8 | absent; probe overfits (48–120 train eps/fold vs 512-d) |
| perception control for progress | raw_tap · frame | linear | ≤ time-only 8/8 | label is genuinely memory-shaped |

Readings: H_absent survives (ω as trained carries no progress state a history read can use);
H_nonlinear unsupported, bounded only in this sample regime. Mechanism: progress lives in EVENTS;
no term in {JEPA, SIGReg, edge SupCon} marks them. Pre-registered next cell E3 = event-marked ω
(todos/T-e3.md). Paper discipline: claim C2 structure; condition Markovianization on policy-level
memory-stratified deltas.
