# Project instructions — cross-task deliberative supervision

Start every session by reading `ONBOARDING.md`, then `internal_planning_and_todos/rwm/board.md`.

## Research world model (standing)
- The hypothesis tree lives in `internal_planning_and_todos/rwm/` (one markdown file per node, sealed
  evidence in `evidence/`, TODOs in `todos/`, live board in `board.md`). Protocol: `rwm/PROTOCOL.md`.
- Capture user-stated hypotheses the same day as a node file. Seal evidence losslessly; re-derive
  the tree (weakest-hypothesis loop) after each seal. Rewrite `board.md` before any status report
  and before compaction.
- Dashboard: `internal_planning_and_todos/rwm/dashboard.html` mirrors the board and tree. After
  updating the board, update the HTML and republish per `rwm/PUBLISH.md` (same artifact URL).
- Numbers carry protocol, n, seeds, CI, run id. Cite `aug_22/h14_p0_status.md` sections, not memory.

## Compute and safety
- No TRI AWS. Do not use `scripts/launch/*` submit paths; they target SageMaker Batch and will fail
  without those credentials. Read-only dry runs are fine for reading a plan.
- Local box: use `CUDA_VISIBLE_DEVICES=0` only (GPU1 is hardware-flaky) and export
  `JAX_PLATFORMS=cuda`. Check `nvidia-smi` before starting anything.
- Never serve or evaluate unverified weights: every checkpoint gets a finite-leaf check first.
- Train-time conditioning must be a statistic the policy server computes causally at rollout.
- Propose code, runs, and installs before doing them; read-only inspection needs no approval.

## Code hygiene
- `uvx ruff check .` and `uvx ruff format .` clean before every commit.
- Account ids, owner email, and home paths come from `wsm_settings.py` (env-overridable); none in new code.
- Never commit data, checkpoints, credentials, or `internal_planning_and_todos/_archive`-style dumps.
