# T — E3 event-marked ω (moves H14.4; pre-registered in plan A16, NOT run)

**Cell:** add a per-frame event target to the Stage-E objective — segment boundary / subskill
completion derived from the frozen segmentation (free labels, no LLM) — keep everything else as
E1b; re-run the progress probes (§19 pooled-linear, §19.5 GRU-64) and the retrieval gate.

**Pre-registered reading:** E3 beats the time-only baseline on ≥k/8 progress families (fix k before
running; suggest k ≥ 3 with the label-shuffled positional floor as the null) AND keeps retrieval
within seed spread of E1b → "progress lives in events and can be written into ω without losing
structure". Fails → H_absent stands and the paper says progress is not representable by these terms.

**Cost:** one local 5090 cell ≈1.5 h + probes ≈1 h. **Venue:** local, after the pass-2 judge
frees the GPUs (≈07:30Z 09-03) and behind the eval lane's queue. **Status:** queued, unscheduled —
lower priority than any policy-arm eval; schedule only in a GPU gap.
