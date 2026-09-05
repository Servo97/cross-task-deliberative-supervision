# RoboMME one-node training campaigns

`campaign_launch.py` holds one p5/H100 or p5e/H200 SageMaker node while running a bounded ordered
list of single-task cells. Every cell retains the same immutable run manifest, run ID, checkpoint
prefix, tree manifest, producer claim, and completion claim as a standalone launch. The campaign
adds orchestration receipts; it does not merge scientific identities.

The current PickXtimes anchor is specified in `sweeps/p5e_pick_anchor_core_v3.json`. It first
refreshes S0/Q0 on the same H200 recipe, then fills Q1, deterministic PTRM E0, and both shared-tau
controls. GDN+JEPA remains implemented but is deferred until the individual GDN and JEPA screens
justify a composition run. Inspect the campaign without loading the AWS SDK or writing cloud state:

```bash
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python \
  -m robomme_integration.campaign_launch \
  --spec robomme_integration/sweeps/p5e_pick_anchor_core_v3.json
```

Submission requires the same command with `--confirm-submit` after explicit user approval. The
launcher pins priority 400, the selected p5/p5e queue, the p5e TrainingPlanArn when applicable,
one attempt, at most 86,400 seconds, and 250--400 GiB. Resubmitting after preemption or a deliberate
runtime-budget defer requires incrementing `attempt_index` in the spec. Exact completed cells are
verified through their scientific identity, deploy receipt, and content-addressed checkpoint-tree
manifest, then skipped.

Within a node, task data, initialization, tokenizer, OpenPI archives/environments, workspace omega
trees, supervision trees, and JAX/XLA compilation cache use shared staging. Artifact reuse
recomputes content receipts; compilation cache keys remain framework/code/shape specific. Mutable
optimizer/checkpoint state is never shared. Each cell receives its own work/checkpoint tree, which
is deleted after its durable claim is verified (or after a failed attempt). A 64-GiB free-space
floor prevents unbounded cache growth.

Before starting a cell, the supervisor requires `estimated_train_seconds + runtime_reserve_seconds`
to remain under the job deadline. Otherwise it publishes an attempt result with
`deferred_runtime_budget`, exits successfully and resumably, and deliberately leaves the stable
campaign-completion claim absent.

Evaluation is intentionally not performed by this entry. p5e is training-only. p5 same-node eval
remains fail-closed until a successful native-render-reset claim and the exact simulator/runtime
bundle are wired into a separate eval-campaign entry. Training completion therefore records
evaluation as deferred and must never be interpreted as scored RoboMME evidence.

---

# "Correct RoboMME runs" wave (2026-08-28) — plan + pre-registered eval

Authority: `internal_planning_and_todos/aug_22/deliberative_workspace_plan.md` §5 (Stage P, RoboMME
row) and §11 amendment A5 role (ii). Governance still binding: `ROBOMME_ALIGNMENT.md` unblock ledger
(rows SUPERSEDED / PARK on the legacy all-16 v1 workspace representation).

## W1. State reconciled at 2026-08-28 (supersedes `robomme_execution_state.json` @ 2026-08-19T23:35:52Z)

| item | 08-19 state | verified 08-28 | source |
|---|---|---|---|
| V4-C5 r4 policy canary | RUNNABLE | **Completed** 2026-08-20T03:00:17Z, sealed `training_canary.complete.json` | SageMaker `AWSBatchsarvesh-rmme-v4-canary-dfd16058b170335e85ae94dbd1fa0169` |
| FS-R1 r3 | RUNNABLE | **Failed** 2026-08-20T02:49:46Z (r1 08-19T12:19, r2 08-19T17:30 also Failed) | `AWSBatchsarvesh-rmme-fs-r1-732bbce7…` |
| FS-B1 | RUNNABLE | **Failed** 2026-08-20T02:48:22Z, no receipt published | `AWSBatchsarvesh-rmme-fs-b1-898e5270…` |
| Batch service-job records | 3 queued | purged (>7 d retention); `describe-service-job` returns `does not exist` | Batch API |
| p5 queue depth | 13 RUNNABLE | **0 queued, 2 RUNNING (both other users)** | `list-service-jobs` |
| V4 training lane | BLOCKED on V4-C5 | **UNBLOCKED** | canary claim above |

## W2. Arm admissibility for the wave

| # | arm | scope | ω required | trained already | admissible now | blocker |
|---|---|---|---|---|---|---|
| a | `s0` | all16 | no | **yes** — `mt-v1-all16-s0-seed0-e10998982e6d8ea8` @59999 | n/a (train done) | needs only the fixed-800 eval below |
| a' | `v4_s0` | all16 | no | no | **YES** | none — this is the wave's one train job |
| b | `wsm_d16_drop05` | all16 | **all-16 index** | no | **NO** | index unsealable: 13/16 task claims |
| b' | `v4_wsm_gdn16_drop02` | all16 | **all-16 index** | no | **NO** | same |
| c | `official_recipe_lerobot` | all16 | no | no | **NO** | `artifacts/initialization/pi05_base/` is empty + no two-view evaluator |
| d | `v4_*` workspace arms | single_task Pick | per-task ω (exists) | no | yes, but out of A5 scope (single-task legacy) | — |

### Why (b) is hard-blocked

`launch.py::_workspace_spec` requires `--workspace-index-s3/--workspace-index-sha256` for any
multitask workspace arm; `training/workspace_index.py::load_workspace_index` fails closed unless the
index carries **all 16** tasks in canonical order. Verified in S3:

| | count | detail |
|---|---|---|
| `manifests/claims/workspace/uniform_gpu_v1/<Task>/…complete.json` | **13/16** | missing PatternLock, MoveCube, RouteStick |
| `manifests/claims/workspace/uniform_gpu_v1/pairs/*.complete.json` | **5/8** | pairs 0, 4, 5 never completed |
| `artifacts/robomme/workspace/all16/` | **empty** | no index was ever published |
| `artifacts/robomme/workspace_dense_v2/` | absent | v2 producer never run |

Root cause is a *supervision*-stage hard fail, not an ω-capacity issue:
`training/workspace_supervision_cache.py:43` rejects any grounded subgoal without exactly one point
and `:164` raises `episode … has no point-grounded salient event`; `workspace_gpu_producer.py`
builds supervision **before** the representation, so the whole task lane dies even though
`wsm_d16_drop05 ∉ SUPERVISION_ARMS` and needs no supervision at all.

Unblocking is therefore a **contract change, not a bug fix**, and is refused here: it would alter the
supervision semantics under which the 13 sealed tasks were produced, and `ROBOMME_ALIGNMENT.md`
already ruled "do not launch another point-only producer … do not spend four multitask runs on
one-token/mean-pooled v1". Two admissible routes, both requiring a decision:

| route | work | cost |
|---|---|---|
| R1 | make supervision optional in the producer + claim schema (index already allows `supervision: null`), re-run pairs 0/4/5 for the 3 missing tasks, publish index | producer change + 1 p5 node (~3 h) |
| R2 | build the dense-v2 representation (the pre-registered v2 contract) and publish a v2 index | M-size build + compute |

## W3. Wave submission plan — READY, NOT SUBMITTED

One p5 node. `campaign_launch.py` is single-task-only (`campaign_plan.py::_cell_arguments` pins
`--scope single_task`), so the multitask arm goes through `launch.py` directly; there is nothing to
pack.

> **Run identity is source-tree-pinned and therefore volatile.**
> `launch.py:656-658` folds `sanitized_source_tree_sha256` into the `scientific` block, whose SHA-256
> becomes `scientific_spec_sha256` → `run_id` → every S3 path and claim URI. `prepared_source_bundle`
> copies the *whole* `robomme_integration/` tree, so **editing this very file changes the run_id**
> (appending §W1–W5 moved it from `…e6fc997399df569f` to `…4e000757dd9767d3`). The submit *command*
> below is stable; the derived identity is not. **Always re-run the dry run immediately before
> submitting and record the identity it prints** — do not reuse a stale run_id from this table.

| field | value (as of the tree state that includes §W1–W5) |
|---|---|
| arm / scope | `v4_s0` / `multitask` (all16, 1,600 episodes) |
| run_id | `mt-v4-all16-v4_s0-seed0-4e000757dd9767d3` |
| scientific_spec_sha256 | `4e000757dd9767d3756d34cb611c5fb5dfa20fa6500d5efbeb74ae3c4f97e30c` |
| manifest_sha256 | `b7b6187361facf66715a492b3747ee56c4b78435137e517370fc4c4623e995a4` |
| recipe | robomme_v4: 60,000 steps, batch 64, h20, full finetune, AdamW b2 .95 wd 1e-6 clip 10, warmup 3,000, peak 5e-5 → 5e-6, EMA .999, 8 devices |
| queue / priority | `fss-tri-cam-robotics-p5-48xlarge-us-west-2` / **400** |
| instance / volume | `ml.p5.48xlarge` (8×H100-80GB) / 400 GiB |
| max_run_seconds | **79,200** (derivation below) |
| mid-run ckpt sync | `save_interval` 5,000 + `remote_resume: true` → `…/steps/<step>` is uploaded during the run; `success_retention: [59999]` prunes only after a sealed final |
| output_s3 | `…/checkpoints/robomme/pi05/multitask_v4/all16/v4_s0/seed0/<run_id>` |
| completion claim | `…/manifests/claims/train/<run_id>/step-59999.complete.json` |
| dry-run artifact | `~/Research/TRI/wsm_data/robomme/aug28_correct_runs/v4_s0_all16_p5_dryrun.json` |

`max_run` derivation: the v1 all16 60k-step run `mt-v1-all16-s0` ran producer-claim
2026-08-04T22:56:54Z → completion 2026-08-05T06:17:10Z = **26,416 s** wall incl. staging (`q0`
independently 26,343 s). Train-only ≈ 24,000 s on H200; ×1.25 for H100 = 30,000 s; ×2.5 = 75,000 s;
+3,600 s startup/staging = **78,600 → 79,200 s** (cap 86,400). This identity cannot terminate Batch
jobs, and a timeout is recoverable from the step-55,000 remote checkpoint.

Verbatim submit command (dry run = same line without `--confirm-submit`):

```bash
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python \
  -m robomme_integration.launch \
  --scope multitask --arm v4_s0 \
  --hardware p5 --priority 400 \
  --max-run-seconds 79200 --volume-size-gb 400 \
  --confirm-submit
```

Run from `/home/sarveshp/Research/TRI/wsmv2`.

## W4. Pre-registered evaluation (fixed-800, within-protocol)

Fixed before any arm is scored. No arm is scored on any other protocol and no row is pooled across
protocol universes (legacy `f2b540e6`/execute-10, released h20/e16, v4 are three separate ledgers).

| parameter | value |
|---|---|
| protocol id | `released_h20_e16_fixed800` (paper label) — **the code constant is `robomme-paper856-h20-e16-fixed50-project-v1`**, see the naming note below |
| episodes | 16 tasks × 50 fixed indices = **800** Boolean outcomes, `dataset: test` split |
| horizon / commit | predict **20**, execute **16**, diffusion context steps 0,16,32,… |
| max steps | 1,300 · `action_space: joint_angle` |
| model_seed | 7 |
| checkpoint | final step only (v4 multitask: 59,999) |
| runner | `robomme_integration/eval/project_exact_runner.py` (implemented, audited, never launched) |
| harness rule | any harness error invalidates the cell; serve-time knobs must self-label in server logs |
| sealed controls | released pi0.5 **153/800 = 19.125 %**; released FrameSamp+Modul **368/800 = 46.00 %** |

### Naming drift: the pre-registration's protocol id does not exist in code (recorded 2026-09-01)

`released_h20_e16_fixed800` appears **exactly once on the whole filesystem — in this file**. The
constant the runner actually stamps into every scorecard is

    PROTOCOL_ID = "robomme-paper856-h20-e16-fixed50-project-v1"   # project_exact_server.py:31

The two describe the same protocol (predict 20 / execute 16 / 50 fixed test indices × 16 tasks =
800), and the `project_exact_*` trio is the only implementation of this universe. But nothing joins
them except this paragraph: a change to either name, or a second implementation claiming the same
paper label, would not be caught by any test. **Cite both ids whenever a fixed-800 cell is reported,
and treat the code constant as the one that defines the universe** — it is what lands in the sealed
artefact.

Arm registration (2026-09-01): `v4_s0` was added to the four arm gates that guard this runner —
`project_exact_server.SUPPORTED_ARMS` (:35) and its `--arm` choices (:312),
`project_exact_runner.METHODS` (:48), and `project_exact_eval.py` `--arm` choices (:176) — so the
sealed `method` string reads `project-exact-v4_s0` and matches its own `checkpoint_uri`. This
registers a new arm; it changes no sealed universe. Serving behaviour is provably identical to
`s0`: `v4_s0` is in `EXECUTION_ARMS` and in none of `WORKSPACE_ARMS` (training or eval),
`WORKSPACE_STEERING_ARMS`, `FAST_WEIGHT_ARMS`, or `cfg_arms`.

### CRN seeding — already present, do NOT port

`eval/execution_model_server.py:551-553` sets, for every request,
`policy_noise_seed = blake2s("{model_seed}:{task}:{episode_idx}:{step}", digest_size=4)`.
The payload carries **no arm**, so two of our arms see identical noise at every protocol coordinate:
this is the RoboMME equivalent of the robocerebra `blake2b(mode|case|trial|step)` rule
(`scripts/robocerebra/eval_seeding.py:44-56`) and needs no port. Difference: blake2s + little-endian
vs blake2b + big-endian; keep RoboMME's, do not unify (it would break the sealed cells).

**Pairing boundary (load-bearing).** The sealed 19.125 / 46.00 controls were produced by
`eval/official_reference_eval.py`, which hands `model_seed=7` to the *upstream* policy server and
never uses our blake2s rule. Therefore:

| contrast | pairing | test |
|---|---|---|
| our arm vs our arm (e.g. GDN − `v4_s0`) | **paired** (identical CRN) | exact McNemar + paired bootstrap CI |
| our arm vs sealed base 19.125 / teacher 46.00 | **unpaired** | two-proportion z + Wilson CIs, never McNemar |

### Detectable Δ (80 % power, α = .05 two-sided)

Unpaired uses Δ = 2.80·√(2·p̄(1−p̄)/n); paired McNemar uses Δ = 2.80·√(π_d/n) at an assumed
discordance π_d = 0.25 (the sealed base-vs-teacher pairing showed π_d = 0.416, so this is
conservative).

| stratum | n | base rate (sealed) | MDE unpaired (vs sealed control) | MDE paired (vs our own base) |
|---|---:|---:|---:|---:|
| overall | 800 | 19.1 % | **6.1 pp** | **5.0 pp** |
| counting_temporal — **C3 target** | 200 | 27.0 % | **12.8 pp** | **9.9 pp** |
| permanence_spatial | 200 | 18.0 % | 11.2 pp | 9.9 pp |
| reference_object | 200 | 19.5 % | 11.4 pp | 9.9 pp |
| imitation_procedural | 200 | 12.0 % | 9.5 pp | 9.9 pp |
| single task | 50 | — | 24.2 pp | 19.8 pp |

### Pre-registered readings

| # | claim | rule |
|---|---|---|
| C3 | gains concentrate where memory is demanded | **counting_temporal** is the named target (base competent at 27.0; teacher +42.5 pp there with a literally empty demo prefix ⇒ pure execution-history conditioning). Permanence is **gated on E2** (K-token) per A5 — at n=200 its MDE is ~11 pp and no ω-single-token arm is powered for it |
| G-interference | H13 rule | any arm >5 pp below its E0 anchor is reported as an interference finding, not dropped |
| G-null | bounded null | an arm inside ±MDE is reported as a bounded null with the MDE stated, never as "no difference" |

### W4 RESULT — `v4_s0` fixed-800, E0 anchor (2026-09-02)

| cell | episodes | successes | rate | Wilson 95 % |
|---|---:|---:|---:|---|
| **project-exact-v4_s0** | 800 (16 x 50) | 143 | **17.875 %** | [15.38, 20.68] |
| released pi0.5 (sealed) | 800 | 153 | 19.125 % | [16.55, 22.00] |
| released FrameSamp+Modul (sealed) | 800 | 368 | 46.00 % | [42.57, 49.46] |

| contrast (unpaired two-proportion z + Wilson) | Δ | 95 % CI on Δ | z | p | MDE | reading |
|---|---:|---|---:|---:|---:|---|
| v4_s0 − released base | **−1.25 pp** | [−5.05, +2.55] | −0.64 | 0.520 | 6.1 | **G-null: bounded null** |
| v4_s0 − released teacher | **−28.12 pp** | [−32.48, −23.77] | −12.07 | <1e−4 | 6.1 | teacher ahead, unambiguous |

| suite (n = 200) | v4_s0 | Wilson 95 % | sealed base | Δ | MDE | reading |
|---|---:|---|---:|---:|---:|---|
| counting_temporal (C3 target) | 24.0 % (48) | [18.6, 30.4] | 27.0 % | −3.0 pp | 12.8 | bounded null |
| permanence_spatial | 23.5 % (47) | [18.2, 29.8] | 18.0 % | +5.5 pp | 11.2 | bounded null |
| reference_object | 12.5 % (25) | [8.6, 17.8] | 19.5 % | −7.0 pp | 11.4 | bounded null |
| imitation_procedural | 11.5 % (23) | [7.8, 16.7] | 12.0 % | −0.5 pp | 9.5 | bounded null |

| cell identity | value |
|---|---|
| protocol_id | `robomme-paper856-h20-e16-fixed50-project-v1` |
| method / arm | `project-exact-v4_s0` / `v4_s0` |
| model_seed / horizons | 7 / predict 20, execute 16, max 1,300 |
| checkpoint sha256 | `b00846018c36b2a7d7c45d88eb6bb971e7e967bdc72e2ea8c63e348a0ac46071` |
| evaluator sha256 | `10cd93f61e9763e6b4255462f8081ec38dcb8bd961501dd49972b57313d48b04` |
| reference evaluator sha256 | `e82019b40e474036a1892a265a8ddf736165b331deac565e6b8f6ee323a2175d` |
| project / openpi / server sha256 | `f65362111377efdf…` / `ed923b2c27d2f608…` / `5091623525f1186c…` |
| policy / benchmark / maniskill commits | `ecf086c3` / `856bc3a1` / `07be6fbc` (all tracked trees clean) |
| renderer recycles | **33** (pre-registered ~31 = 800/25; ceiling 40; budget 64) |
| G-interference | does NOT fire (rule >5 pp below anchor; observed 1.25 pp) |

Renderer recycles are NORMAL operation for this lane (~25-episode simulator lifetime) and do **not**
invalidate a cell: they occur at environment construction between episodes, the ledger is atomic per
episode, and CRN is keyed on (task, episode, step) independently of attempt. What invalidates a cell
is an error inside a scored episode, or an exhausted restart budget leaving the 800 incomplete.

### Effective training data — the corruption is LOCAL ONLY; S3 (and therefore cloud training) is clean

Audited 2026-08-28. The reported "33/100 ButtonUnmaskSwap episodes corrupt, byte-identical to HF" is
wrong on all three counts, and it does **not** affect cloud training.

| copy | state |
|---|---|
| local HF cache `~/.cache/huggingface/…/snapshots/1510653c…` | **54 corrupt parquets**: 52 unreadable (`OSError` — thrift ×35, snappy ×16, EOS ×1) + 2 (eps 124, 132) that decode but fail their HF SHA-256. Span **two** tasks: ButtonUnmaskSwap 100–199 (50/100 readable, 48/100 byte-faithful) and PatternLock eps 70, 85 (98/100). Other 14 tasks 100/100 |
| **S3 mirror** `datasets/robomme/v1/lerobot_all16` (what a cloud job downloads) | **CLEAN.** 9/9 of the locally-corrupt episodes (70, 85, 106, 115, 124, 132, 140, 150, 161) `pq.read_table()` successfully from S3 and differ byte-wise from the local copies; control ep 500 is byte-identical S3↔local |

Consequences:

1. **The W3 submit is safe.** `download_policy: complete_verified_all16_inventory` pulls from S3, so
   all16 training gets the true 100 episodes/task. The invariant holds in the cloud.
2. **The local cache is now REPAIRED** (2026-08-28) — see §W6.
3. ~~Inventory gap to close.~~ **RETRACTED — the claim was wrong.** All 1,600 parquet records in the
   sealed inventory *do* carry `source_sha256`; the earlier note sampled `objects[0]`, which is
   `.gitattributes`, a non-parquet object that legitimately has none. The sealed-inventory path is
   already verified end to end in both directions — see §W6.

## W6. Local HF-cache repair (2026-08-28) — DONE, sha-verified, 1,600/1,600 readable

Receipt: `~/Research/TRI/wsm_data/robomme/aug28_correct_runs/cache_repair.json`.

The local HuggingFace cache stores each LFS object as a symlink into `blobs/<oid>`, where the blob
filename **is** the expected sha256. That gives a free ground truth: hash the blob, compare to its
own name.

| stage | result |
|---|---|
| audit before | 1,600 audited · 1,546 match · **54 mismatch** |
| repair | 54 attempted · **54 repaired** · 0 failed · 3.63 GB downloaded |
| audit after | 1,600 audited · **1,600 match** · 0 mismatch |
| loader open-check after | **1,600/1,600 readable** (`pq.read_table` on every episode — what `load_dataset("parquet", …)` does before step 0) |
| frame counts after | all 16 tasks **100/100 episodes**, frame-exact vs published meta, total **768,897** frames |

Write protocol: download the S3 object to a temp file in the blob's own directory, sha256-verify it
against the expected oid, and only then `os.replace()` atomically over the blob. A failed
verification leaves the original untouched and writes nothing.

**Three-way agreement** holds for all 1,600 parquets:
`inventory.source_sha256` == HF blob filename == repaired file content.

Corrupt files spanned two tasks — ButtonUnmaskSwap 52, PatternLock 2 — and the corruption was
**local only**: the bad files failed their own HF LFS sha256, and the S3 mirror was never affected,
so no cloud training run was ever compromised.

Direction note: the in-repo `fleet/repair_dataset.py` repairs the **S3 prefix from HuggingFace**.
This repair was the opposite direction — **local cache from the verified-clean S3 mirror** — for
which there is no in-repo tool.

### The `source_sha256` inventory check is already implemented — nothing was added

| side | mechanism |
|---|---|
| manifest | `fleet/inventory.py::validate_inventory` rejects a schema-v2 RoboMME inventory unless **every** `data/*.parquet` record carries both `source_sha256` and `checksum_crc64nvme` |
| download | `fleet/inventory.py::materialize().fetch()` hashes each object as it streams, and on `source SHA-256 mismatch` unlinks the temp file and fails **before** it is moved into place |
| sealed inventory | `…/inventories/data/e77968b4….json` validates clean: 1,608 objects, 129,485,391,999 bytes, all 1,600 parquets carry `source_sha256` |

**Residual gap (the real one):** the `huggingface_hub` path that populated the local cache performs
no post-download sha256 verification — which is exactly why this corruption went unnoticed for a
month. The S3 materialize path would have caught it. Any future local snapshot should be run through
`fleet/audit_dataset.py` once after download.

### Pass-1 top-up owed (NOT run — GPUs allocated)

The 33 episodes pass 1 skipped as unreadable are all now readable. Note recorded at
`~/Research/TRI/wsm_data/deliberation/pass1_store/robomme/_robomme_unreadable.RESOLVED.json`
(the original `_robomme_unreadable.json` was left byte-unchanged as the historical record).

All 33 are **ButtonUnmaskSwap** (global indices):

```
112 114 120 130 131 134 135 136 137 138 139 140 141 142 143 144 145 146
147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
```

Every RoboMME run manifest must still record effective per-task counts, and no per-task RoboMME rate
may be reported without them.

## W5. Local RTX 5090 feasibility for the two arms — **NO** on both counts

Asked: can the multi-task base or the multi-task GDN arm finish in ~36 h on ONE RTX 5090 (32 GB)?
Answer is no twice over; memory alone is disqualifying.

### Memory (decisive, independent of time)

pi0.5 deploy params at step 59,999 measure **12,434,925,754 B** in fp32 = **3.109 B parameters**
(11.58 GiB per copy). The pinned recipe is `full_finetune: true`, AdamW, `ema_decay: 0.999`,
`jax_devices: 8`, **`fsdp_devices: 1`** — i.e. pure data parallelism, so every device holds a
*complete* copy of params + grads + m + v + EMA.

| configuration | resident state per GPU | vs 32 GB 5090 |
|---|---:|---|
| all-fp32 (params, grads, m, v, EMA) | **57.9 GiB** | 1.9× over, before any activation |
| bf16 params+grads, fp32 m/v/EMA | **46.3 GiB** | 1.5× over, before any activation |
| 2×5090 with `fsdp_devices: 2` | ~23–29 GiB | fits state only, no activation headroom — and changing `fsdp_devices` changes the pinned scientific identity |

No batch size fixes this: the deficit is optimizer state, not activations.

### Time (moot, reported for completeness)

Measured anchor: `mt-v1-all16-s0` 60,000 steps in 26,416 s wall (`q0` 26,343 s) on 8×H200 ⇒ ~24,000 s
train ⇒ 0.400 s/step aggregate = **3.20 H200-s/step**. RTX 5090 bf16 dense ≈ 209.5 TFLOPS vs H200 SXM
≈ 989.5 TFLOPS ⇒ 0.212× ⇒ **15.1 s/step on one 5090** (compute-bound bound; ignores the 1.79 vs
4.8 TB/s bandwidth gap, so it is optimistic).

| arm | steps | scale | 1×5090 wall | ≤36 h? |
|---|---:|---:|---:|---|
| multi-task base (`v4_s0`, all16) | 60,000 | 1.0 | **252 h (10.5 d)** | no — 7.0× over |
| multi-task GDN w16+dropout | 60,000 | 1.5¹ | **378 h (15.8 d)** | no — 10.5× over |
| single-task base (reference) | 20,000 | 1.0 | 84 h (3.5 d) | no |
| single-task GDN (reference) | 20,000 | 1.5¹ | 126 h (5.3 d) | no |

¹ GDN scale factor from the v4 Phase-A estimates: 16,200 s (`v4_wsm_gdn8_drop02`) ÷ 10,800 s
(`v4_s0`) at equal steps.

**Verdict:** both arms are cloud-only. There is no local fallback for RoboMME policy training, and
none should be attempted; the 5090 lane remains an *evaluation/encoder* lane only.
