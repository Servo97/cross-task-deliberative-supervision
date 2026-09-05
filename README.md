# Cross-task deliberative supervision for long-context robot policies

Can a compressed workspace token stream ω, trained with cross-task deliberative supervision (a
reasoning VLM judging which moments across tasks are functionally the same) and read by a gated
DeltaNet, make memory-demanding manipulation tasks Markovian?

- New here: read [`ONBOARDING.md`](ONBOARDING.md).
- Story and system flows: [`internal_planning_and_todos/PROJECT_NARRATIVE.md`](internal_planning_and_todos/PROJECT_NARRATIVE.md).
- Hypotheses, evidence, live board: [`internal_planning_and_todos/rwm/`](internal_planning_and_todos/rwm/).
- Design authority: [`aug_22/deliberative_workspace_plan.md`](internal_planning_and_todos/aug_22/deliberative_workspace_plan.md); measurements: [`aug_22/h14_p0_status.md`](internal_planning_and_todos/aug_22/h14_p0_status.md).

Benchmarks: RoboCasa365, ReMemBench, RoboCerebra, RoboMME. Backbones: π0.5 (primary), GR00T N1.7.

## Layout
| path | contents |
|---|---|
| `workspace_models/` | encoder networks, objectives, Stage-E trainer, frozen-tap feature extractors |
| `scripts/deliberation/` | descriptor pass, embedding/mining, typed-edge judging, label chain |
| `scripts/launch/`, `robomme_integration/` | post-training and eval launchers (SageMaker Batch; need porting) |
| `scripts/remembench/`, `scripts/robocerebra/` | local eval lanes |
| `vla_training/` | π0.5 / GR00T training and serving code |
| `tests/` | unit and contract tests |
| `wsm_settings.py` | env-overridable settings (account ids, paths, owner) |
| `internal_planning_and_todos/` | narrative, plan, status, research world model |

## Setup
```bash
./install.sh pi05      # or groot | wsm — one uv env per backbone
python -m pytest tests/ robomme_integration/tests/
uvx ruff check . && uvx ruff format --check .
```
