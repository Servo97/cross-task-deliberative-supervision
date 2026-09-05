# T — RoboMME tap refire (unblocks H14.7 and the 4-tap Stage-E cells)

**Failure (2026-09-03 00:30Z):** both jobs FAILED at the first shard — `ModuleNotFoundError: No
module named 'robocasa'` from `wsm_robocasa_configs.py:20` (imports `robocasa.utils.dataset_registry`
at import time); the CPU `--plan-only` preflight passed because it never imports the config module.
`robomme_stage_entry.sh` never installed robocasa/robosuite; the proven RoboCerebra entry does.

**Fix (applied):** the RoboCerebra install block mirrored into the openpi venv setup (clone
robosuite `85abee22…` + robocasa `be22d659…`, `scripts/install_robocasa_deps.sh`, PyOpenGL), plus a
fail-fast import of `robocasa` and `wsm_robocasa_configs` with the node's `WSM_CONFIGS_DIR` before
any data materialization. `bash -n` clean.

| job | READY (run from wsmv2/; dry-run = drop `--confirm-submit`) | max_run |
|---|---|---|
| tap (encoder store `wsm_pooled/rmme_pi_100k`) | `submit_robomme_stage.py --phases tap --priority 400 --max-run-seconds 12600 --openpi-source-s3 …/openpi/fd252276….tgz --image-uri …groot-dexjoco@sha256:79859289…` | 12,600 s |
| tapserve (policy store, serve-aligned grid) | same with `--phases tapserve` | 12,600 s |

After landing: stratified eff-rank on the rmme tap → bar 0.80× as a number → `<RMME TAP PREFIX>`
for the Stage-E READY; tapserve store → policy ω export once E1b-4tap exists. Expected early
markers: `staged wsm_configs …, sha verified`; `robocasa + wsm_robocasa_configs import OK`;
`root=… parquet=1600`; per-GPU `200 episodes to do`.
