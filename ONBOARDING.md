# Cross-task deliberative supervision — onboarding (one page)

Repo: `github.com/Servo97/cross-task-deliberative-supervision` (private). This replaces the TRI-internal
`wsmv2` checkout. **TRI AWS (SageMaker, S3) is no longer available to this project.** Everything below
assumes the local box and whatever compute you bring.

## What we are testing
Robot policies fail on tasks whose current image is not enough to act (memory demand). We give the
policy a compressed **workspace token stream ω** and a recurrent read over it (gated DeltaNet), and ask
whether ω trained with **cross-task deliberative supervision** — a reasoning VLM judging which moments
across tasks are functionally the same — makes memory-demanding tasks Markovian.
Story so far: `internal_planning_and_todos/PROJECT_NARRATIVE.md`. Hypothesis tree and status:
`internal_planning_and_todos/rwm/README.md`, `rwm/board.md`.

## Read these first (30 minutes)
1. `internal_planning_and_todos/rwm/README.md` — every hypothesis, status, and what would move it.
2. `internal_planning_and_todos/rwm/board.md` — what was running when TRI access ended, and the hand-off state.
3. `internal_planning_and_todos/aug_22/deliberative_workspace_plan.md` — design + amendments A1–A19 (amendments override the body).
4. `internal_planning_and_todos/aug_22/h14_p0_status.md` — every measurement by section number. Cite sections.
5. `internal_planning_and_todos/hypothesis_ledger.md` — how claims are worded (weakest form the evidence forces).

## What you have without TRI
| asset | where | state |
|---|---|---|
| code: taps, deliberation pipeline, Stage-E trainer, launchers, evals, tests | this repo | complete; launchers are SageMaker-specific (see "compute") |
| sealed deliberation labels `<V2C>` = `91461e8df6f2d143` (28,722 segments, 4 domains, 261k edges) | `~/Research/TRI/wsm_data/deliberation/stage_e_labels/` | local |
| frozen π0.5 tap stores (RoboCasa `pi_100k`, ReMemBench `rmb_pi_100k`, RoboMME `rmme_pi_100k`, RoboCerebra `rcb_pi_libero`) | `~/Research/TRI/wsm_data/wsm_pooled/`, `wsm_data/s3_salvage/` | local |
| ReMemBench base-60k milestones 15k/30k/45k/59999 (`s0-6ab9621b9d58b326`) | `wsm_data/wsmv2_scratch/rmb60k/ckpt/` | local, all four |
| RoboMME base-70k milestones (`mt-v4-70k-…-28d80fb9`) | `wsm_data/s3_salvage/…/checkpoints/robomme/…` | pulled 09-04 if the salvage finished (check `s3_salvage/pull_salvage.log`) |
| RoboCerebra base-60k, Stage-E retrain (10 cells) | were in flight on TRI p5 on 09-04 | **presumed lost** unless salvaged before access ended |
| π0.5 tap checkpoint, Stage-P policy checkpoints, RoboCerebra raw data | `wsm_data/local_ckpts/`, `wsm_data/robocerebra/` | local |

## Compute reality
- Local box: two RTX 5090 (32 GB). **GPU1 is hardware-flaky** (drops off the bus under load; then CUDA
  dies box-wide until a reboot). Use `CUDA_VISIBLE_DEVICES=0` only. Export `JAX_PLATFORMS=cuda` so a
  CPU fallback fails loudly instead of running 30× slower.
- Fits on one 5090: the VLM judge (Qwen3.8-27B NVFP4), Stage-E encoder cells (~1 h each), D7 parity,
  ReMemBench evals with one server + one worker (~5.4 h per 264 rollouts), RoboCasa/RoboCerebra evals.
- Does **not** fit: π0.5 post-training (60k steps at batch 32 needs an 8×H100 node, ~8–25 h). The
  launchers in `scripts/launch/` and `robomme_integration/eval/` target SageMaker Batch; porting the
  node entries (`*_entry.sh`) to `torchrun`/slurm on a university cluster is the first infra TODO.

## How work happens here
- **Hypothesis → TODO → evidence → seal.** An idea becomes a node file in `rwm/hypotheses/` the same day;
  its experiment gets `rwm/todos/T-*.md` with command, cost, kill criterion; results go to the status
  doc first, then a pointer in the node; closed questions compress into `rwm/evidence/E-*.md` losslessly.
- **Weakest-hypothesis rule.** A claim earns only the strength the evidence forces.
- **Cheapest evidence first.** Canary → local probe → cluster arm. Pre-register the reading and the MDE.
- **Numbers travel with protocol, n, seeds, CI, run id.** No exceptions.
- Keep `rwm/board.md` current and republish the dashboard (`rwm/PUBLISH.md`).

## Where the code is
| piece | path |
|---|---|
| encoder objectives, Stage-E trainer | `workspace_models/networks/`, `workspace_models/train/train_wsm_base/train_stage_e.py` |
| frozen π0.5 taps | `workspace_models/features/*_pooled_tap.py` |
| deliberation pipeline (descriptors → mine → typed edges → labels) | `scripts/deliberation/` |
| policy post-training + eval per benchmark | `robomme_integration/`, `scripts/launch/submit_pi_stage_s.py`, `scripts/launch/submit_robocerebra.py` |
| local ReMemBench eval lane | `scripts/remembench/run_cell_local.sh` (`CKPT/STEP/CKPT_URI/SDE/NW` env) |
| node entries | `*_entry.sh` at the root and in `robomme_integration/` |
| settings (account ids, paths, owner — env-overridable) | `wsm_settings.py` |
| tests | `tests/`, `robomme_integration/tests/` |

## Environments, tests, lint
- `./install.sh groot | pi05 | wsm` builds one uv env per backbone (torch vs jax pins conflict).
- Tests: `python -m pytest tests/ robomme_integration/tests/`; modules needing jax/flax skip elsewhere.
- Lint: `uvx ruff check .` and `uvx ruff format .` before pushing. No absolute home paths or account
  ids in new code (read `wsm_settings.py`); never commit data, checkpoints, credentials.

## Your first week
1. Reboot-safe local check: `CUDA_VISIBLE_DEVICES=0` + `JAX_PLATFORMS=cuda`, run one ReMemBench base
   milestone cell (`rwm/todos/T-maturity.md`) — this finishes the base maturity curve nobody has yet.
2. Retrain one Stage-E cell locally (E1b, seed 20260828) on the four tap stores + `<V2C>` labels; check
   the retrieval gate (`rwm/todos/T-policy-arms.md` step 3).
3. Pick an OPEN node in `rwm/README.md`; propose the cheapest discriminating step in one paragraph.
