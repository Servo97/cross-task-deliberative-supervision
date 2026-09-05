# Cross-task deliberative supervision — onboarding (one page)

Repo: `github.com/Servo97/cross-task-deliberative-supervision` (private). You work on **Babel** (CMU
slurm). Large runs go to TRI SageMaker **through the project lead only** — see "Two-tier compute".

## What we are testing
Robot policies fail on tasks whose current image is not enough to act (memory demand). We give the
policy a compressed **workspace token stream ω** and a recurrent read over it (gated DeltaNet), and ask
whether ω trained with **cross-task deliberative supervision** — a reasoning VLM judging which moments
across tasks are functionally the same — makes memory-demanding tasks Markovian.
Story: `internal_planning_and_todos/PROJECT_NARRATIVE.md`. Hypotheses and status: `rwm/README.md`, `rwm/board.md`.

## Read these first (30 minutes)
1. `internal_planning_and_todos/rwm/README.md` — every hypothesis, status, what would move it.
2. `internal_planning_and_todos/rwm/board.md` — hand-off state, what is blocked, next actions.
3. `internal_planning_and_todos/aug_22/deliberative_workspace_plan.md` — design + amendments A1–A19 (amendments override the body).
4. `internal_planning_and_todos/aug_22/h14_p0_status.md` — every measurement by section; **§56** is the latest encoder result. Cite sections.
5. `internal_planning_and_todos/hypothesis_ledger.md` — how claims are worded (weakest form the evidence forces).

## Babel setup (day one)
1. Install the lab's cluster skill so your Claude knows Babel's rules:
   `git clone https://github.com/simchowitzlabpublic/maxlab-infra-skills ~/maxlab-infra-skills && cd ~/maxlab-infra-skills && ./bin/install.sh`
2. Partitions: `preempt` for restartable GPU work (L40S / A100 / RTX PRO 6000, requeue on); `maxlab`
   (3 × 8 RTX PRO 6000 96 GB, 3-day limit, **needs Max's approval**); `maxlab-cpu` for shells with
   `/data` access; `general` non-preemptible. `/data` and `/scratch` exist **only on compute nodes**.
3. Data root (team artifacts): `/data/group_data/maxlab/common_datasets/sarveshp/cross_task_deliberation`.
   The code defaults to `~/Research/TRI/wsm_data`; on a compute node make it resolve there:
   `mkdir -p ~/Research/TRI && ln -s /data/group_data/maxlab/common_datasets/sarveshp/cross_task_deliberation ~/Research/TRI/wsm_data`
   and `export WSM_DATA_ROOT=$HOME/Research/TRI/wsm_data` (read by `wsm_settings.py`).
4. Envs: `./install.sh pi05 | groot | wsm` builds one uv env per backbone (torch vs jax pins conflict).
   Tests: `python -m pytest tests/ robomme_integration/tests/`. Lint: `uvx ruff check . && uvx ruff format .`.

| asset | path under the data root | state |
|---|---|---|
| sealed labels `<V2C>` `91461e8df6f2d143` (28,722 segments, 4 domains, 261k edges) | `deliberation/stage_e_labels/91461e8df6f2d143/` | 25 MB |
| frozen π0.5 tap stores (RoboCasa, ReMemBench, RoboMME, RoboCerebra) | `wsm_pooled/`, `s3_salvage/…/wsm_pooled/rcb_pi_libero/` | 2.7 GB |
| Stage-E encoders + ω stores, 10 cells (status §56) | `s3_salvage/studies/long_context_v1/artifacts/deliberation/stage_e/772597789979f88a/cells/` | selected encoder `E1b_109e99680ca5c198` |
| base maturity milestones: ReMemBench 15k/30k/45k/59999 · RoboMME 10k…60k, 69999 · RoboCerebra 15k/30k/45k/59999 | `wsmv2_scratch/rmb60k/ckpt/`, `s3_salvage/…/checkpoints/robomme/…/deploy/`, `s3_salvage/…/robocerebra/checkpoints/pi05/a0_base/` | trained, **none evaluated** |
| π0.5 tap checkpoint (frozen backbone for taps and serving) | `local_ckpts/pi05_on_149999/` | 12 GB |

## Two-tier compute (how a run happens)
- **Tier 1 — Babel, you drive.** Judge passes (Qwen3.8-27B), Stage-E encoder cells (~1 h/cell/GPU),
  D7 parity, local evals (ReMemBench 264 rollouts ≈ 2.7 h on 2 GPUs; RoboCasa/RoboCerebra sims),
  RoboTok-style baselines. Preempt jobs must checkpoint and resume; never disable requeue.
- **Tier 2 — TRI SageMaker, lead fires.** π0.5 post-training arms (8×H100, 8–25 h) and the
  RoboMME/ReMemBench eval fleets. You never touch AWS. You produce a **READY file** in
  `internal_planning_and_todos/rwm/ready/` (protocol in `ready/README.md`): the exact launcher line,
  its dry-run manifest hash and run id, cost, gates, and what it produces. Push. The lead pulls,
  re-dry-runs on their machine, fires with `--confirm-submit`, appends the job id, and syncs results
  back into the data root. Your Claude and the lead's Claude speak through that file and `board.md`.

## How work happens here
- **Hypothesis → TODO → evidence → seal.** An idea becomes `rwm/hypotheses/H*.md` the same day; its
  experiment gets `rwm/todos/T-*.md` (command, cost, kill criterion); results go to the status doc,
  then a pointer in the node; closed questions compress into `rwm/evidence/E-*.md` losslessly.
- **Weakest-hypothesis rule.** A claim earns only the strength the evidence forces.
- **Cheapest evidence first.** Canary → Babel probe → cluster arm. Pre-register the reading and MDE.
- **Numbers travel with protocol, n, seeds, CI, run id.** Rewrite `rwm/board.md` before every status.
- Never serve or evaluate unverified weights (finite-leaf check first). Train-time conditioning must
  be a statistic the policy server can compute causally at rollout.

## Where the code is
| piece | path |
|---|---|
| encoder objectives, Stage-E trainer | `workspace_models/networks/`, `workspace_models/train/train_wsm_base/train_stage_e.py` |
| frozen π0.5 taps | `workspace_models/features/*_pooled_tap.py` |
| deliberation pipeline (descriptors → mine → typed edges → labels) | `scripts/deliberation/` |
| post-training + eval launchers (SageMaker; tier 2) | `scripts/launch/`, `robomme_integration/eval/` |
| local ReMemBench eval lane | `scripts/remembench/run_cell_local.sh` (`CKPT/STEP/CKPT_URI/SDE/NW` env) |
| node entries | `*_entry.sh` at the root and in `robomme_integration/` |
| settings (paths, account, owner — env-overridable) | `wsm_settings.py` |

## Your first week
1. Evaluate the four ReMemBench base milestones on Babel (`rwm/todos/T-maturity.md`, 2 GPUs per cell) → the first evaluated maturity curve.
2. D7 parity for encoder `109e99680ca5c198` (`--lang-mode stored`) on its exported ω stores; write the result as a status section.
3. Pick an OPEN node in `rwm/README.md`; propose the cheapest discriminating step in one paragraph; if it needs tier 2, write the READY file.
