# T — Judge effort pilots → A17 decision (moves H14.8)

| step | state | detail |
|---|---|---|
| PILOT-2 pass-2 xhigh | RUNNABLE p5 @400 since 09-02 15:10Z (`h14-delib-pass2-afb60016d29b8fc1-…`) | §15.3 150 paired buckets + 200 rcb anchors; measures tok/anchor, truncation at 12,288, κ vs low/medium |
| PILOT-1 pass-1 xhigh | RUNNABLE (`h14-delib-pass1-2dc442271412b867-…`) | 100 RoboCasa segments; robocasa client path never node-run → a 2nd attempt is likely |
| on landing | one executor, A17 harness (status §45–46; validated 12/17 reproduction) | R1 blind re-grade of 40 disagreements (max right ≥60 % AND +10 pairs, one-sided sign test); R2 planted-CONTRAST strict ≥0.60 / loose ≥0.75; R3 ≥50 % descriptors newly state completion conditions |
| verdict → redo | ALL 28,505 anchors (+28,722 segments if pass-1) in ONE homogeneous store; J-job wave from PILOT-2's rate (2×→5, 4×→9, 5×→11, 8×→17 jobs; `--num-shards 8J --shard-offset 8j`); embed prefix `fullmine_4ee34e407ff4b71c`; wall ≈ T / concurrent nodes (5× band: 4.0 d @1 node, 1.1 d @4) | 77–153 node-h pass-2, +13–26 pass-1 |
| verdict → stay low | current low-effort stores remain the label authority; write the A17 outcome in the plan doc | 0 |

Constraints: never mix effort/model/quantization inside a label artifact; the local low-effort rcb
delta keeps running as the baseline either way; `caption_segments.py` edit freeze until pass-2 lands.

## Outcome 2026-09-03 — both pilots FAILED on harness defects; the effort question is unmeasured

| pilot | ran | failure | fix needed |
|---|---|---|---|
| PILOT-2 (pass-2 xhigh, 409 anchors / 4,308 pairs) | 23:52→04:22Z, hit max_run 15,870 s, ≈0 buckets in S3 | judge clients: `KeyError: 'strata'` on many anchors (pilot store built without the field the judge reads) + `TimeoutError: timed out` per request (HTTP client timeout sized for low-effort completions; xhigh chains exceed it) | build the pilot store with the full schema; scale the client timeout and max_run with effort; re-pilot ≈4–8 h node |
| PILOT-1 (pass-1 xhigh, 100 segments) | 00:02→00:11Z | shard clients refused `--max-new-tokens 8192 exceeds the frozen hard cap 3072` (`caption_segments.py`) | lift the cap after pass-2 lands (§38.5 makes code_sha provenance-only), or accept 3072 (truncates xhigh) |

**Recommendation (coordinator):** park H14.8 at LOW for this campaign. Evidence for the recipe already
rests on low-effort labels (structure result, κ(low, medium) 0.838), the xhigh requests are long
enough to blow the low-effort timeout (so a redo would be several× the 77–153 node-h under a 10-node
account quota), and nothing downstream is blocked on it. Re-pilot only if label quality becomes the
binding constraint on a policy result. User decides.
