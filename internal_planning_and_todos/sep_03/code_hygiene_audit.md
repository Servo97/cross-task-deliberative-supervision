# Code-hygiene audit before the first GitHub push (read-only, 2026-09-03)

Scope = everything `git ls-files -co --exclude-standard` would push after the `.gitignore` update:
51 tracked files today + the untracked trees (`robomme_integration/`, `scripts/`, `tests/`,
`workspace_models/`, `vla_training/`, entries, `figures/`, `draw.io/`, `overleaf/`, `gcp_tpu/`),
16 MB total, no file over 5 MB, no notebooks or pickles. Nothing in the codebase was modified except
`.gitignore` (three lines) and two new docs (`ONBOARDING.md`, this file).

## 1. Secrets and identity

| check | result | action |
|---|---|---|
| hard-coded credentials (AKIA…, hf_…, sk-…, private keys, `password=`) | **none** — the only hit writes creds from env vars (`scripts/ec2/push_box_creds.sh:22`) | none; that script serves the retired EC2 box → archive candidate (phase 3) |
| AWS account ids `141701954645` / `124224456861` | 132 files in `scripts/`, 30 in `robomme_integration/`, 5 `tests/`, 2 `gcp_tpu/`, 1 `vla_training/` | internal-infra disclosure, not a credential → **push to a PRIVATE repo**; centralize into one settings module later (phase 2) |
| personal absolute paths `/home/sarveshp/…`, `sarvesh.patil` | 162 `scripts/`, 59 `robomme_integration/`, 16 `tests/`, 8 `gcp_tpu/`, 5 `workspace_models/` | same as above; defaults should come from env with documented fallbacks (phase 2) |
| e-mail addresses `@tri.global` | 17 `scripts/`, 11 `robomme_integration/` (submit tags) | fine for a private org repo |
| stray root files | `export.zip` (July-14 handoff tarball), `temp_launch_training.py` | **now gitignored** (not deleted) |
| `.claude/` | only `skills/` (28 KB) | shareable; `settings.local.json` gitignored pre-emptively |

## 2. Static findings (`uvx ruff@0.12.10 check`, default rules E4/E7/E9/F, 568 files, `_archive` excluded)

| rule | count | nature | fix class |
|---|---:|---|---|
| I001 unsorted imports | 128 | cosmetic | auto (`--fix`), phase 1 |
| E702 statements chained with `;` | 115 | style; concentrated in `tests/test_wsm_demo_cfg.py` (17), `figures/presentation/make_plots.py` (17), `workspace_models/train/train_wsm_base/_wsm_train_common.py` (15), `train_wsm2_icl.py` (8), `scripts/analysis/a17_effort_verdict.py` (7) | manual, phase 1b |
| F401 unused imports | 42 (22 `scripts/`, 7 `vla_training/`, 4 `robomme_integration/`) | dead code | auto, phase 1 |
| E402 import not at top | 35 | mostly deliberate (`sys.path` shims, jax/torch ordering) | review, keep with `# noqa` where deliberate |
| E741 ambiguous name (`l`, `O`) | 12 | readability | manual, phase 1b |
| F841 unused variable | 9 | dead code / hidden bug candidates | manual review |
| E731 lambda assignment | 5 | style | manual |
| E401 / E701 / F541 | 4 / 4 / 3 | style | auto / manual |
| F722 forward-annotation syntax | 2 | **false positive**: jaxtyping shape strings `at.Float[at.Array, "l b _t _k _h"]` in `robomme_integration/amkv/patched_history_gemma.py:366` (live path: FrameSamp teacher producer + 4 tests) | ruff config `lint.ignore = ["F722"]` or per-line `# noqa: F722` |
| E9 syntax errors | 0 | every file parses | — |

Per package: `tests/` 91, `scripts/` 50, `workspace_models/` 45, `vla_training/` 32, `figures/` 25,
`robomme_integration/` 12, `gcp_tpu/` 5. No finding is a correctness bug on a sealed path; the
F841s deserve a look before any refactor.

## 3. Why nothing was auto-fixed today

Every Python tree is content-addressed by a launcher (`robomme_integration/` source tree sha, the
wsmv2/openpi archives, `caption_segments.py` code sha). Sealed runs stay reproducible from their S3
archives regardless, but edits re-mint every READY run id and invalidate the staged M-arm tree and
the Stage-E READY line while jobs are in flight. The user also asked for conservative pruning.

## 3b. Phase 1 + 1b DONE (2026-09-04 00:40Z, user-approved "fix the semicolon stuff and formatting")

| step | result |
|---|---|
| `ruff check --fix` I001/F401/E401/F541 on 541 files (excluded: `caption_segments.py` freeze, out-of-repo dirs) | 157 fixed |
| `ruff format` on the same set | 461 files reformatted (semicolon chains and E701 gone by construction) |
| manual: E741 `l` → `line`/`lab`/`legacy` (10), F841 dead assignments (9), E731 lambdas → `def` (5) | done by hand, one site each; `_items`/`_per` kept where the RHS has a side effect or exercises a path |
| `pyproject.toml` | `extend-exclude` for out-of-repo trees; `lint.ignore = ["F722"]` (jaxtyping shape strings); `per-file-ignores` E402 for `tests/**` and one sys.path-shimmed script |
| ruff after | **0 findings** on repo files (only the gitignored `temp_launch_training.py` remains) |
| compile | 541/541 |
| tests (`sm_launch` env, `--continue-on-collection-errors`) | 41 → **28 failed / 976 passed / 26 skipped / 14 errors**; the 14 fixed were one real regression from the autofix — `submit_pi_stage_s_eval.py` lost a `FINAL_STEP` re-export that 14 tests read — restored with `# noqa: F401`. The remaining 28 + 14 are missing modules in this env (jax/flax/augmax/robocasa/openpi_client/safetensors), one `/tmp` fixture, four `test_policy_canary` pinned digests and one `workspace_runner` trainer-source pin (both encode a sealed identity; re-seal deliberately, never auto-update), and one eval-launcher regex test |
| identity | every content-addressed run id in the tree has moved (expected); the sealed S3 archives are untouched; every READY line is re-dry-run at fire time anyway |

## 4. Proposed plan (each phase a single reviewed commit, after the in-flight READY lines are fired)

| phase | change | risk |
|---|---|---|
| 0 (done) | `.gitignore`: `export.zip`, `temp_launch_training.py`, `.claude/settings.local.json`; `ONBOARDING.md` | none |
| 1 | `ruff check --fix` for I001/F401/E401/F541 (177 fixes, behaviour-preserving); add `[tool.ruff]` to `pyproject.toml` with `line-length`, `lint.ignore = ["F722"]`, per-file E402 allowances | re-mints run ids → dry-run every READY line again |
| 1b | de-chain the 115 `;` lines in the five hot files; rename the 12 ambiguous names; review the 9 unused variables | manual, per file, with tests |
| 2 | one `wsm_settings.py` (or env-backed dataclass) for account ids, buckets, study root, user/owner; launchers import it; absolute `/home/<user>` defaults replaced by `Path.home()`/env | touches 190+ files; do it as one commit with the launcher tests green |
| 3 | archive `scripts/ec2/` (retired box) and `gcp_tpu/` under `_archive/` or a `legacy/` branch; decide whether `overleaf/` and `figures/` live in this repo | user decision |

Decisions for the user: private vs public repo (public requires phase 2 first); include `overleaf/`,
`gcp_tpu/`, `figures/`, `draw.io/`, `.claude/skills/`?

## 5. Phase 2 result (settings module) — DONE 2026-09-04, no commit

One env-backed module, `wsm_settings.py` (repo root), now owns every account id, bucket/study root, submitter
identity, and `$HOME`-anchored path. Defaults reproduce today's literals; `WSM_<NAME>` overrides; `python
wsm_settings.py` prints the table. `scripts/launch/launch_guardrails.py` imports it (with a `sys.path` append
shim when run as a script) and re-exports `EXECUTION_ACCOUNT / STORAGE_ACCOUNT / LEGACY_ACCOUNT / REGION /
DEFAULT_RESULTS_BUCKET / STUDY_OWNER / OWNER_EMAIL / PROJECT_TAG / IMAGE_OWNER / DEXJOCO_IMAGE_REPO /
WSM_ROBOCASA_S3 / LONG_CONTEXT_STUDY_S3` plus the module itself, so launchers keep one import point and no
launcher API changed. `STUDY_OWNER` (S3 prefix; content addresses) and `OWNER_EMAIL` (SCP tag) stay independent.
README gained a "Configuration" section; `tests/test_wsm_settings.py` pins the defaults (10 tests).

### 5.1 Files touched (84 edited + 2 new + README section)

| tree | files | names |
|---|---:|---|
| `robomme_integration/` | 19 | `launch_e0.py`, `stage_e0.py`, `campaign_launch.py`, `audit_demo_prefixes.py`, `framesamp_am_r1_launch.py`, `framesamp_am_r1_screen_launch.py`, `framesamp_b1_launch.py`, `launch_p5_campaign.py`, `launch_p5_preflight.py`, `local_rtx5090_campaign.py`, `stage_dataset.py`, `launch.py`, `move_workspace_dense_v2_launch.py`, `policy_canary_launch.py`, `bridge_workspace_artifacts.py`, `resume_videorepick_a6.py`, `framesamp_am_teacher_fixture_export.py`, `v4_policy_canary_launch.py`, `workspace_launch.py` |
| `robomme_integration/tests/` | 12 | `test_amkv_patch_contract.py`, `test_demo_robottt.py`, `test_execution_eval.py`, `test_framesamp_am_flax_overlay.py`, `test_framesamp_am_policy_overlay.py`, `test_framesamp_am_teacher_producer.py`, `test_framesamp_b1_policy_overlay.py`, `test_gdn8_jepa_combo.py`, `test_gdn_jepa_overlay.py`, `test_sequence_forcing_control.py`, `test_training_adapter.py`, `test_workspace_eval.py` |
| `root/` | 1 | `pyproject.toml` |
| `scripts/analysis/` | 1 | `ptrm_d0a_diagnostic.py` |
| `scripts/data/` | 3 | `build_heldout_eval_manifest.py`, `port_mimicgen_to_s3.py`, `port_target_extras_to_s3.py` |
| `scripts/deliberation/` | 3 | `launch_deliberation.py`, `launch_stage_e.py`, `local_pass1.py` |
| `scripts/launch/` | 18 | `finalize_producer_omega.py`, `launch_guardrails.py`, `strip_h13_checkpoint.py`, `submit_evals.py`, `submit_finetunes.py`, `submit_groot_rmb.py`, `submit_pi_stage_s.py`, `submit_pi_stage_s_eval.py`, `submit_policyfeats.py`, `submit_pretrains.py`, `submit_robocerebra.py`, `submit_robocerebra_stage.py`, `submit_robomme_stage.py`, `submit_stage_s_producer.py`, `submit_v4_r4_backlog.py`, `submit_wsm.py`, `submit_wsm_canary.py`, `submit_wsm_cfg.py` |
| `scripts/robocerebra/` | 3 | `omega_window.py`, `serve_pi05_libero_wsm.py`, `test_gather_eval.py` |
| `tests/` | 16 | `extract_deltanet_jax_parity.py`, `extract_groot_seam_dims.py`, `extract_jepa_target_pi_semantics.py`, `extract_seq_windows_pi.py`, `test_groot_seq_collator.py`, `test_pi_wsm_cfg.py`, `test_pi_wsm_combo.py`, `test_pi_wsm_deltanet.py`, `test_pi_wsm_jepa.py`, `test_pi_wsm_salient.py`, `test_robottt_fast_weights.py`, `test_serve_groot_batched.py`, `test_stage_q_dispatch.py`, `test_stage_q_variants.py`, `test_submit_pi_stage_s.py`, `test_wsm_jepa_target_loader.py` |
| `vla_training/` | 3 | `stage_e_serve.py`, `_pi05_common.py`, `_pi05_seq_common.py` |
| `workspace_models/` | 5 | `phase_battery.py`, `pi_backbone_tap.py`, `_wsm_train_common.py`, `train_wsm_from_groot_17.py`, `train_wsm_from_pi_05.py` |
| root (new) | 2 | `wsm_settings.py`, `tests/test_wsm_settings.py` |

`pyproject.toml`: `[tool.pytest.ini_options] pythonpath = ["."]` (bare `pytest` imports the root module) and a
hatch `force-include` so non-editable wheels ship `wsm_settings.py`. `robomme_integration/amkv/launch_e0.py`:
the E0 node bundle now copies `wsm_settings.py` to its staged root next to the two runtime helpers it already
ships (`launch_guardrails.py` needs it there; caught by `test_staged_source_root_is_package_rooted_for_the_node_pythonpath`).
Incidental: one pre-existing dead assignment (F841 `ok`) removed in `scripts/data/port_mimicgen_to_s3.py`.
Excluded as instructed: `workspace_models/labels/caption_segments.py` (edit freeze; it has no literals anyway),
`gcp_tpu/`, `figures/`, `overleaf/`, `internal_planning_and_todos/`, `temp_launch_training.py`.

### 5.2 Literal counts, Python files (`141701954645|124224456861|/home/sarveshp|sarvesh\.patil`; files with hits / matches)

| tree | before | after | remaining = |
|---|---|---|---|
| `scripts/` | 28 / 111 | 5 / 10 | comments + docstrings only |
| `workspace_models/` | 5 / 6 | 0 / 0 | — |
| `vla_training/` | 3 / 4 | 0 / 0 | — |
| `robomme_integration/` | 45 / 96 | 14 / 51 | 8 node-side modules + 6 test-pin files |
| `tests/` | 20 / 61 | 6 / 47 (5 / 26 excluding the new default pins in `test_wsm_settings.py`) | test pins |
| `wsm_settings.py` | — | 1 / 12 | the single home of the defaults |

Every remaining Python occurrence and why it stays:

| where | lines | reason |
|---|---|---|
| `scripts/launch/submit_robocerebra.py` 50-51, `submit_pi_stage_s.py` 54-55 + 1407-1408, `submit_stage_s_producer.py` 230 + 232 | 8 | comments explaining why `--user` (storage prefix) is frozen and independent of the SCP tag |
| `scripts/launch/submit_pretrains.py` 7, `submit_groot_rmb.py` 73 | 2 | docstring/comment history of the account-124 era |
| `robomme_integration/training/workspace_gpu_producer_dense_v2.py` 22-34, `training/policy_canary.py` 38 + 631-634, `training/v4_policy_canary.py` 34 + 202, `eval/framesamp_b1_cloud.py` 31-42, `eval/framesamp_am_r1_screen_cloud.py` 37-128, `eval/run_local_fixed50_queue.py` 21-22 (imported by node-side `eval/campaign.py`), `training/framesamp_am_teacher_producer.py` 1237, `amkv/episodes.py` 70 | 23 lines | node-side: run from the isolated `robomme_integration/` bundle (`PYTHONPATH=$CODE_DIR` or the package parent), which never contains the repo root; also sealed study addresses / manifest values (`test_policy_canary` pins them) |
| `robomme_integration/tests/test_gpu_pipeline.py`, `test_amkv_launch_e0.py`, `test_campaign.py`, `test_cloud_admission_p5.py`, `test_p5_quota_retry.py`, `test_move_workspace_dense_v2.py` | 13 lines | expected-plan values, fake ARNs, one deliberately wrong role |
| `tests/test_submit_evals_guardrails.py` (11), `test_watch_p5_action_canary.py` (8), `test_submit_pi_stage_s.py` (2), `test_submit_pi_stage_s_eval.py` (1), `test_stage_s_eval_inputs.py` (2) | 24 | pins of the resolved constants' values, offline image/S3 inputs, a fake guardrails module |
| `tests/test_wsm_settings.py` | 17 | the new default pins (by design) |

Non-Python, untouched by design: `scripts/**/*.yaml` 422 matches (422 in `scripts/configs/`: sealed run
configs carrying S3 paths), `scripts/**/*.sh` 75 matches in 24 local driver scripts
(`scripts/robocerebra/run_*.sh`, `scripts/deliberation/run_stage_e_*.sh`, `scripts/ec2/*`, `scripts/launch/*.sh`
— 5090-box helpers, phase-3 archive candidates), `robomme_integration/**/*.json` 5036 matches (sealed
inventories/manifests; content addresses), a handful of `.md`. The node entries (`*_entry.sh` at the root and
under `robomme_integration/`) contain **zero** literals: they take everything from the launcher-sealed environment.

### 5.3 Identity check (`--dry-run` before vs after, same CLI, `sm_launch` python)

| launcher | raw diff | after normalizing sha256 / 16-hex ids / stamps |
|---|---|---|
| `scripts/launch/submit_pi_stage_s.py` (s0, test fixtures) | byte-identical | 0 |
| `scripts/launch/submit_pi_stage_s_eval.py` (s0, test fixtures) | byte-identical | 0 |
| `scripts/launch/submit_robocerebra_stage.py` (tap phase) | byte-identical | 0 |
| `scripts/launch/submit_robomme_stage.py` (tap phase, parity-doc line) | byte-identical | 0 |
| `scripts/launch/submit_robocerebra.py` (a0_base, 30k) | wsmv2 archive sha + derived `spec_sha256`/`run_id`/URIs/job-name stamp | 0 |
| `robomme_integration/launch.py` (ButtonUnmaskSwap s0) | `sanitized_source_tree_sha256` + derived run/attempt/manifest ids and their URIs | 0 |
| `scripts/deliberation/launch_stage_e.py` (8-cell multidomain line) | `created_utc` only | 0 |
| `scripts/deliberation/launch_deliberation.py` (pass2, explicit max-run) | `created_utc` only | 0 |

Every S3 URI prefix, bucket, owner, queue, role, training-plan ARN, image URI and SCP tag is unchanged; only
source-tree digests and the ids derived from them moved (expected: the wsmv2 and `robomme_integration/` trees changed).
Node-bundle simulation (isolated interpreter, package-only `sys.path`): every node-side module that imports in
the `sm_launch` env still imports without the repo root; the only failures are the env-missing `jax`/`openpi`/`einops`.

### 5.4 Tests and lint

| check | before | after |
|---|---|---|
| `pytest tests robomme_integration/tests` (`sm_launch`) | 28 failed / 976 passed / 26 skipped / 14 errors | 28 failed, 986 passed, 26 skipped, 3 warnings, 14 errors, 2 subtests passed |
| failure + error set (42 entries) | — | **identical** (missing `jax`/`flax`/`augmax`/`robocasa`/`openpi_client`/`safetensors`; the `/tmp/robomme-fs-b1-overlay-v1` fixture; four `test_policy_canary` pinned digests; the `test_workspace_eval` trainer-source pin; `test_submit_evals_guardrails::test_every_submit_launcher_uses_shared_fail_closed_policy` (launcher count 15 ≠ 11 tripwire) and `test_submit_pi_stage_s_eval::test_arm_step_and_workspace_mismatches_fail_closed` (regex)) |
| new tests | — | `tests/test_wsm_settings.py` 10 passed (also under bare `pytest`) |
| `ruff check` / `ruff format --check` on the 85 touched Python files | — | clean |
| `ruff check .` whole repo | 0 (phase 1) | 1 pre-existing I001 in `scripts/launch/build_deterministic_archive.py` (not touched here; autofixable) |

### 5.5 Reviewer notes

- Env names were checked against every existing `WSM_*` variable in this repo and in `internal_training`: no collision
  (`WSM_DATA` exists and is unrelated to `WSM_DATA_ROOT`; `WSM_USER` keeps its guardrails meaning).
- `robomme_integration/launch.py` lost its unused `REGION` import; tests read `launch.EXECUTION_ACCOUNT`, which stays.
- `scripts/launch/submit_v4_r4_backlog.py` imports the tags as `from scripts.launch.launch_guardrails import`
  (repo-root style like its sibling imports); it never had the literal `from launch_guardrails import` form.
- Nothing is blocked. Not done on purpose: shell/yaml/json literals (phase 3), the frozen `caption_segments.py`.
