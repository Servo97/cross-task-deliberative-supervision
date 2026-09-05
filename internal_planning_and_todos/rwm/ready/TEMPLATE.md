# R-<slug> — <one-line title>

state: PROPOSED | READY | FIRED | LANDED | FAILED | WITHDRAWN
owner: <mentee github handle>
moves: <rwm node, e.g. H14.5>
opened: <YYYY-MM-DD>

## What it produces
<artefact and where it must land under the data root, e.g. `s3_salvage/…/checkpoints/pi05/remembench/<run_id>/`>

## Pre-registered reading
- statistic: <e.g. P1′ − P2′ success on the memory stratum, paired by episode>
- MDE / CI: <e.g. 7.4 pp at 264 rollouts>
- kill criterion: <e.g. retrieval gate below chance at the canary; D7 FAIL blocks the arm>

## Command (dry-run form, verbatim)
```
cd <repo> && python scripts/launch/<launcher>.py --dry-run <args…>
```
- dry-run run id: `<16 hex>`
- manifest sha256: `<64 hex>`
- source tree sha256: `<64 hex>`
- tests: `python -m pytest <files>` → <n passed>
- commit: `<git sha>`

## Cost
priority: 400 | 100 · instance: ml.p5.48xlarge × 1 · max_run_seconds: <≤172800> · expected: <h>

## Log (append-only; lead writes FIRED/LANDED lines)
- <YYYY-MM-DD HH:MMZ> PROPOSED by <who>: <note>
