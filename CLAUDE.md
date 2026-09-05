# Project instructions — cross-task deliberative supervision

Start every session by reading `ONBOARDING.md`, then `internal_planning_and_todos/rwm/board.md`.

## Research world model (standing)
- Hypothesis tree in `internal_planning_and_todos/rwm/` (one file per node, sealed evidence in
  `evidence/`, TODOs in `todos/`, live board `board.md`, READY queue `ready/`). Protocol: `rwm/PROTOCOL.md`.
- Capture user-stated hypotheses the same day as a node file. Seal evidence losslessly; re-derive the
  tree (weakest-hypothesis loop) after each seal. Rewrite `board.md` before any status report and
  before compaction.
- Dashboard `rwm/dashboard.html` mirrors the board and tree; after a board rewrite update it and
  republish per `rwm/PUBLISH.md`.
- Numbers carry protocol, n, seeds, CI, run id. Cite `aug_22/h14_p0_status.md` sections, not memory.

## Two-tier compute
- **Babel (mentees):** follow the `babel` skill from maxlab-infra-skills. Data root
  `/data/group_data/maxlab/common_datasets/sarveshp/cross_task_deliberation` (compute nodes only);
  symlink it to `~/Research/TRI/wsm_data` and export `WSM_DATA_ROOT`. `maxlab` partition needs Max's
  approval; `preempt` jobs must checkpoint/resume. Propose runs and installs before doing them.
- **TRI SageMaker (lead only):** mentee sessions never submit. A cluster run is requested by writing
  `rwm/ready/R-<slug>.md` per `rwm/ready/README.md` with a validated `--dry-run`. The lead's session
  re-dry-runs, fires with `--confirm-submit`, records the job id in the same file, and syncs results
  into the data root. Submits carry `tri.project` and `tri.owner.email`; priority 400 default, 100 for
  sweeps, never ≥600 without the lead's explicit say-so; 48 h max.

## Safety
- Never serve or evaluate unverified weights: finite-leaf check every checkpoint first.
- Train-time conditioning must be a statistic the policy server computes causally at rollout.
- Sealed trees are content-addressed: any edit under `robomme_integration/` or the launchers changes
  run ids — the READY file's dry-run must be redone after any such edit.
- Read-only inspection needs no approval; runs, submits, installs, and deletions do.

## Code hygiene
- `uvx ruff check .` and `uvx ruff format .` clean before every commit.
- Paths, account ids, owner email come from `wsm_settings.py` (env-overridable); none in new code.
- Never commit data, checkpoints, credentials, or scratch dumps.
