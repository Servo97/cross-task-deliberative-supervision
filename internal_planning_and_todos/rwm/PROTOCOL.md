# RWM protocol — project localization (wsmv2 / H14)

Global rules: `~/.claude/skills/research-world-model/SKILL.md`. Local pins:

| item | here |
|---|---|
| authority for numbers | `aug_22/h14_p0_status.md` §N (executors write there first); ledger row `hypothesis_ledger.md` "H14"; paper-facing wording `PAPER_STATE.md` |
| authority for design | `aug_22/deliberative_workspace_plan.md` §§0–11 + amendments A1–A19 (amendments override) |
| what "sealed" means | run ids + S3 prefixes + protocol id + n/seeds/CI recorded in the status doc; figure scripts recompute from sealed run dirs |
| evidence file naming | `evidence/E-<topic>.md`; one table per question; every number carries protocol + n + pointer |
| node ids | `H14.k` fixed forever; user-stated intuitions are quoted verbatim in **Origin** |
| TODO files | `todos/T-<topic>.md`: gates → READY line (dry-run form) → cost (measured rate) → kill criteria → who fires (coordinator only; executors never submit) |
| board | `board.md`, rewritten (not appended) at every status report; archival detail goes to nodes/evidence |
| compaction | README + board + touched nodes current; memory pointer `[[h14-deliberative-workspace-campaign]]` → `rwm/README.md` |
| claim discipline | weakest-hypothesis loop on every seal; stronger variants live in the node's "refused" table with their discriminator |
| cost discipline | lean-ops: canary → local → cluster; one executor at a time unless two are both blocking; token budget is a study constraint |
