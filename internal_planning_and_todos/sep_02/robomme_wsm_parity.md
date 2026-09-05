# RoboMME × workspace-model feature parity — FrameSamp+Modul ↔ [demo ω ; live ω] + GDN

Written 2026-09-02. Paper: **arXiv:2603.04639**, *RoboMME: Benchmarking and Understanding Memory for
Robotic Generalist Policies*. Authority for the eval lane: `robomme_integration/CAMPAIGNS.md` §W4.
Authority for the ω mechanism: `internal_planning_and_todos/aug_22/deliberative_workspace_plan.md`
(A3/A5/D7) and `h14_p0_status.md` §§32–42.

Nothing here has been submitted. Section 6 is the READY set.

---

## 0. Pre-registration (written before any RoboMME ω arm exists)

| # | statement | fixed value |
|---|---|---|
| P1 | Protocol is the proven fixed-800 lane, unchanged | `robomme-paper856-h20-e16-fixed50-project-v1`, 16×50, h20/e16, max 1,300, seed 7 |
| P2 | Pairing | arm-vs-arm **paired** (identical blake2s CRN, `execution_model_server.py:551-553`); arm-vs-sealed-control **unpaired** (two-proportion z + Wilson) |
| P3 | Primary readings | **M3 − M1** (does the demo prefix add) and **M3 − M3-ctrl** (does deliberative structure add). Both paired |
| P4 | Ceiling | released FrameSamp+Modul 46.00 % is the ceiling, reported unpaired, never as a paired contrast |
| P5 | MDE (80 % power, α=.05) | overall 6.1 pp unpaired / **5.0 pp paired**; per suite 9.5–12.8 pp |
| P6 | Stratum prediction | gains concentrate in **imitation_procedural** and **reference_object** (the two suites where 4/4 tasks carry a demo video). **counting_temporal has no demo video in any of its 4 tasks** — M3 is predicted to be a bounded null there, and any counting gain is attributable to the live read, not the prefix |
| P7 | Bounded null | an arm inside ±MDE is reported as a bounded null with the MDE stated |
| P8 | Interference | any arm >5 pp below its own E0 anchor (17.875) is reported as an interference finding, not dropped |
| P9 | No-demo stratum | the 7 no-demo tasks are reported **separately** from the 9 demo tasks in every M2/M3 row. Pooling them would average over a cell where the mechanism cannot apply |
| P10 | Gate order | D7 ω-parity PASS → preflight (§5.6) → policy submit. A parity failure blocks M1/M2/M3 |

---

## 1. METHOD CARD

### 1.1 The paper's FrameSamp + Modul

| element | paper | source |
|---|---|---|
| memory source | the growing causal prefix `[0, step_idx]` of front-view frames | §Perceptual memory; `even_sampling_indices` |
| sampling | **uniform, inclusive**: keep all frames if ≤32, else `linspace(0, step_idx, 32)` | reimplemented exactly in `robomme_integration/training/upstream_framesamp_data.py:even_sampling_indices` |
| per frame | π₀.₅'s own vision encoder, 8×8 patch grid bin-pooled to **4×4 = 16 tokens**, width 1024 (+`PosEmb3D` at 4×4) | `upstream_framesamp_data` module docstring |
| budget | **32 × 16 = 512 tokens**, "matching the number of image tokens in the current observation"; short prefixes right-padded | paper §4; module constants |
| views | **front only** | module docstring |
| demo handling | **none special.** The demonstration is the episode's own leading frames, so the inclusive prefix already contains it | module docstring, verbatim |
| Modul | "before entering the feed-forward block in each layer, the action features cross-attend to memory tokens via multi-head attention… projected into scale and shift parameters that modulate the normalized action features through **AdaLN**" | paper §Integration |
| where | **every action-expert layer** | ibid. |
| alternatives compared | Memory-as-Context (concat into the VLM expert), Memory-as-Expert (extra blockwise-causal expert) | ibid. |
| training | one model over all 16 tasks; max 1,300 env steps/episode; report = mean of last 3 checkpoints × 3 seeds | paper §5 |
| result | **44.51 %** overall vs π₀.₅ no-memory **17.93 %**; human 90.5 % | Table 3 |

Sealed local controls on our protocol (same 800 keys, step-79999, seed 7):

| model | counting | permanence | reference | imitation | overall |
|---|---:|---:|---:|---:|---:|
| paper π₀.₅ | 28.78 | 17.00 | 17.17 | 8.78 | 17.93 |
| **released π₀.₅ (sealed)** | 27.0 | 18.0 | 19.5 | 12.0 | **19.125** |
| paper FrameSamp+Modul | 65.22 | 25.11 | 36.33 | 51.39 | 44.51 |
| **released FrameSamp+Modul (sealed)** | 69.5 | 27.0 | 35.5 | 52.0 | **46.00** |
| **our `v4_s0` base (M0, DONE)** | 24.0 | 23.5 | 12.5 | 11.5 | **17.875** [15.38, 20.68] |

### 1.2 Parity mapping, cell by cell

| paper cell | our cell | parity |
|---|---|---|
| demo frames → π₀.₅ encoder → memory tokens | demo frames → **pi0.5 pooled tap → Stage-E encoder → K_demo ω tokens** | **approximate — this is the contribution.** Theirs is 512 raw spatial tokens; ours is 8 deliberatively-supervised 512-d latents. The whole point of the arm is that the bottleneck is the manipulated variable |
| live frames → memory tokens | live frames → same tap+encoder → `ω_t` | approximate, same reason |
| one undifferentiated uniform sample over `[0, step_idx]` | **two fixed segments**: `[demo prefix ; live window]` | **deliberate deviation.** Their single uniform sample lets the demo swamp the live history (VideoPlaceOrder: median 924 demo frames of 1,127 → ~26 of 32 slots are demo). Fixed budgets make demo-vs-live a controlled factor instead of a length artifact |
| segment identity | **fixed slot allocation** — prefix always occupies slots `[0,k_demo)` | parity via position. The GDN's `pos_decay_bias` is already `[window_len, num_heads]`, a per-slot learned bias, so segment identity is learnable with **zero new parameters** and `window_len` stays structurally readable from the checkpoint |
| uniform-inclusive sampling rule | `uniform_inclusive` = integer `linspace` incl. both endpoints | **exact** — same rule as their `even_sampling_indices` and our `sequence.py::uniformly_sample_prefix` |
| memory → cross-attn → AdaLN scale/shift on action features | ω window → **GDN gated delta-rule read** → one additive vector into the action expert's `adarms_cond` | **approximate.** Theirs is cross-attention over all memory tokens at every layer; ours is a recurrence read out at the newest slot, added once. Recorded risk: a decaying recurrence can wash out strictly-older prefix slots — §5.7 is the diagnostic and the fallback |
| every action-expert layer | the sealed single-seam `wsm_tanh_cond` conditioner | approximate; changing the seam is a second factor and is out of scope for this arm |
| front view only | front + wrist + zeroed right slot (the policy's own tree) | deviation, inherited from the existing tap contract |
| 512 tokens | 8 + 16 = **24** ω tokens | approximate by construction (21× fewer tokens) |
| training recipe | the sealed multitask v4 recipe, one-line `cond_window` diff | exact w.r.t. our own base |

**Three approximate-parity cells recorded in code, not folklore:**
1. `rmme_pooled_tap.py` §3 — the tap sets `image_mask[right_wrist]=True` where the policy sets `False`.
2. `rmme_pooled_tap.py` §2 — RoboMME's 8-d joint state is normalized by the RoboCasa tap config's stats (`pi05_rc_mg60_bal33`). Consistent train↔serve; not the policy's own normalization.
3. `rmme_demo_prefix.py` — the 7 no-demo tasks clamp the prefix slots to the earliest ω rather than zero-filling.

---

## 2. ABLATION TREE (pre-registered)

Every arm: multitask all16, the sealed v4 recipe, **one factor = the ω window fed to the GDN**.

| arm | ω read | window | new params vs M1 | status |
|---|---|---|---|---|
| **M0** | none (base) | — | — | **DONE**: 17.875 [15.38, 20.68], 143/800 |
| **M1** | live-ω only (our standard read) | `[ω_{t-15·s} … ω_t]`, K=16 | 0 | to train |
| **M2** | demo-ω only, read out at the newest demo slot | `[ω^demo_1 … ω^demo_8]`, K=8 | 0 | to train |
| **M3** | **demo + live (the parity arm)** | `[ω^demo_1..8 ; ω_{t-15s}..ω_t]`, K=24 | 0 (window length only) | to train |
| **M3-ctrl** | M3's window, ω from the **structure-free encoder (ctrl-0b)** | K=24 | 0 | to train |
| M4 *(optional)* | FrameSamp raw frames, no ω | 512 raw tokens | — | **not recommended** — see §2.2 |

Registered readings:

| reading | contrast | pairing | MDE | what it decides |
|---|---|---|---:|---|
| **R1 (primary)** | M3 − M1 | paired | 5.0 pp | does the demonstration prefix add anything over the live read |
| **R2 (primary)** | M3 − M3-ctrl | paired | 5.0 pp | does *deliberative structure* in ω add anything, holding the window fixed |
| R3 | M2 − M0 | unpaired vs own anchor | 6.1 pp | is the demo prefix alone sufficient (isolates "is the live read carrying M3") |
| R4 | M1 − M0 | paired | 5.0 pp | replication of the sealed H13/H12 null on a 4th benchmark |
| R5 | M3 vs 46.00 | **unpaired**, Wilson | 6.1 pp | distance to the teacher ceiling. Never McNemar |

Per-suite predictions (n=200 each, MDE 9.5–12.8 pp), registered:

| suite | tasks with a demo video | prediction for M3 − M1 |
|---|---:|---|
| imitation_procedural | **4/4** (MoveCube, InsertPeg, PatternLock, RouteStick) | largest movement; teacher gains +40.0 pp here |
| reference_object | **3/4** (VideoRepick, VideoPlaceButton, VideoPlaceOrder; **PickHighlight has none**) | second; teacher +16.0 pp |
| permanence_spatial | **2/4** (VideoUnmask, VideoUnmaskSwap) | bounded null expected at n=200 |
| counting_temporal (C3 target) | **0/4** | **bounded null predicted.** The teacher's +42.5 pp here comes from execution-history conditioning with an empty prefix, i.e. M1's axis, not M3's |

### 2.1 The stratification that makes this readable

900 of 1,600 training episodes carry a demo prefix; 700 do not. **9 of the 16 eval tasks** have one.
Per P9, M2/M3 are reported as three rows: all-16, demo-9, no-demo-7. The demo-9 row is where R1/R2
are actually powered (n=450 at eval; MDE ≈ 6.6 pp paired).

### 2.2 On M4 (raw-frame FrameSamp arm)

**Recommend not reviving it.** The FS-R1/FS-B1 canaries both failed on 2026-08-20 and the causes
are structural, not incidental:

| canary | failure | cause |
|---|---|---|
| FS-R1 r1 | flat-package import | packaging |
| FS-R1 r2 | checkpoint restore | project JAX 0.10.1 / Orbax 0.12 vs the checkpoint's upstream JAX 0.5.3 / Orbax 0.11.13 |
| FS-R1 r3 | Failed 02:49:46Z, **no receipt** | not diagnosed |
| FS-B1 | Failed 02:48:22Z, **no receipt** | not diagnosed |

Reviving M4 means resolving a two-JAX-runtime split inside one node and re-diagnosing two
receipt-less failures — days, not hours. **The teacher's sealed 46.00 already occupies the cell M4
would fill**: "is the workspace bottleneck helping or hurting vs raw frames" is answered by
M3 vs 46.00 (R5), unpaired but on identical 800 keys. If a paired version is later required, that is
a separate packet.

---

## 3. WHERE THE DEMO PREFIX LIVES (measured, not assumed)

`Yinpei/robomme_data_lerobot@1510653c`, `meta/info.json`: **`total_videos: 0`** — there are no MP4s.
Frames are encoded image bytes inside the parquet (`{"bytes":…, "path":…}`). Every episode is ONE
parquet whose leading `exec_start_idx` rows are the demonstration:

```
columns: image, wrist_image, state[8], actions[8], exec_start_idx, is_demo, step_idx, epis_idx,
         simple_subgoal, grounded_subgoal, simple_subgoal_online, grounded_subgoal_online, …
```

Verified on **all 1,600 episodes**: `is_demo` is a contiguous leading prefix and
`is_demo.sum() == exec_start_idx[0]` on every one; `step_idx` is episode-global and starts at 0.

| task | suite | eps w/ demo | demo frames min/med/max | ep len med | tapped frames | tapped demo |
|---|---|---:|---|---:|---:|---:|
| PatternLock | imitation | 100 | 24 / 96 / 284 | 192 | 2,738 | 1,345 |
| MoveCube | imitation | 100 | 148 / 238 / 299 | 440 | 5,056 | 2,801 |
| InsertPeg | imitation | 100 | 203 / 239 / 293 | 477 | 6,100 | 3,022 |
| RouteStick | imitation | 100 | 100 / 175 / 350 | 350 | 4,735 | 2,341 |
| VideoPlaceButton | reference | 100 | 703 / 757 / 818 | 959 | 12,195 | 9,535 |
| VideoPlaceOrder | reference | 100 | 702 / 924 / 1,145 | 1,127 | 14,179 | 11,508 |
| VideoRepick | reference | 100 | 147 / 312 / 385 | 682 | 8,705 | 3,505 |
| **PickHighlight** | reference | **0** | — | 331 | 4,447 | 0 |
| VideoUnmask | permanence | 100 | 66 / 66 / 66 | 177 | 2,836 | 900 |
| VideoUnmaskSwap | permanence | 100 | 114 / 168 / 216 | 367 | 4,488 | 1,950 |
| ButtonUnmask | permanence | **0** | — | 229 | 3,458 | 0 |
| ButtonUnmaskSwap | permanence | **0** | — | 444 | 5,124 | 0 |
| PickXtimes | counting | **0** | — | 543 | 6,844 | 0 |
| StopCube | counting | **0** | — | 311 | 4,083 | 0 |
| SwingXtimes | counting | **0** | — | 442 | 5,561 | 0 |
| BinFill | counting | **0** | — | 619 | 7,666 | 0 |
| **TOTAL** | | **900 / 1,600** | | | **98,215** | **36,907** |

768,897 frames total; 292,040 (38.0 %) are demo frames; the stride-8 grid keeps 98,215
(36,907 demo + 61,308 live).

**Two corrections to the paper's own text.** It lists PickHighlight under "tasks that use
video-based observations"; in the released data it has **no** demo prefix on any of its 100
episodes. And the Imitation suite is described as uniformly video-based, which the data confirms
(4/4) — but ButtonUnmask/ButtonUnmaskSwap in Permanence have none, so the suite is 2/4.

**Serve side.** The simulator supplies the prefix at reset, not the dataset: `EnvRunner.get_init_obs()`
returns aligned `images` / `wrist_images` / `states` lists
(`robomme_integration/eval/audit_demo_prefixes.py`), and the harness delivers them as
`obs["video_history"]` + `obs["video_state_history"]` inside an `episode_restart` envelope
(`eval/workspace_runner.py::capture_workspace_observation`).

**Consequence — there is no second ω store.** Because the demo is the episode's own leading frames,
a per-episode ω store over the whole episode already contains the demo ω. "K demo tokens" is an
index rule over the existing store:

```
demo ω = ω[frame_indices <  exec_start_idx]        live ω = ω[frame_indices >= exec_start_idx]
```

---

## 4. WHAT WAS BUILT

| file | what | validation |
|---|---|---|
| `workspace_models/features/rmme_pooled_tap.py` | the RoboMME pi0.5 pooled tap; `wsm_pooled` `p.npz` + `is_demo`/`exec_start_idx`/`n_frames_episode` | `--plan-only` on the real tree, node argv, shards 0/8 and 3/8: **200 eps each, OK**; zero-work and unknown-task paths **exit 1**; ruff clean |
| `workspace_models/features/rmme_demo_prefix.py` | the `[demo ; live]` window rule + `m1/m2/m3` dispatch, dependency-free | `--self-test` on **all 1,600 real episodes: PASS**; live rule proven **bit-identical** to the sealed `workspace_runner.requested_omega_steps` on 60 geometries |
| `robomme_stage_entry.sh` | the p5 node entry (tap / omega / parity) | `bash -n` clean; carries every §36–42 trap (see below) |
| `scripts/launch/submit_robomme_stage.py` | approval-gated launcher | `--dry-run` clean; `--print-configs-upload` reproduces sha `026255fa…` |

Node-entry traps already paid for elsewhere, carried here rather than rediscovered:

| trap (h14 §) | in the entry |
|---|---|
| §37.4 ERR trap naming `$LINENO` | line 39 |
| §37.1 `status=0; wait "$pid" \|\| status=$?` | `wait_clients` |
| §36.3 never a bare `wait` (the sync loop never exits) | explicit `TAP_PIDS` only |
| §34.6 per-attempt log prefix, shipped every cycle | `ship_log` → `logs/$ATTEMPT/` |
| §37.2 client-dep preflight that **imports** for real | `import jax, pyarrow, PIL, numpy, torch` before any shard |
| §37.3 / §38.2 silent success with zero work | `--plan-only` preflight; shard fatal on zero episodes |
| §38.3 post-stage corpus assertion | `n_done == n_npz == 1600` or exit 6 |
| §41.2 parity must use the checkpoint that EXPORTED the store | `encoder_best.pt` rejected in **both** entry and launcher |
| §39.3 `taskmean` fails a correct encoder | gate mode is `stored`; `taskmean` demoted to a measured alternative |
| §24.3 multi-domain cell needs a raw-tap eff-rank | A3 audit is a required tap-job **output** (`--stratify-files --max-files 48 --seed 20260822`) |
| CAMPAIGNS §W6 | dataset comes through the sealed inventory (per-object `source_sha256`), never `s3 sync` |

Deliberately **not** touched: `scripts/deliberation/caption_segments.py` (frozen, §38.5), every
sealed config, and `robomme_integration/` — the new modules live under `workspace_models/` so the
RoboMME `run_id` (which folds `sanitized_source_tree_sha256` over the whole
`robomme_integration/` tree) does not move.

---

## 5. BUILD PLAN

### 5.1 (a) The tap — READY

| field | value |
|---|---|
| entry / launcher | `robomme_stage_entry.sh` / `scripts/launch/submit_robomme_stage.py` |
| backbone | `pretrain150k/pi05/mg60_bal33/run/149999` — **the same frozen network robocasa and remembench are tapped from**, and the checkpoint every RoboMME arm initialises from |
| pool | frozen WSMv1 `18c26a7d54d48058…` (verified present in S3) |
| dataset | sealed inventory `e77968b4c72c7589…` (verified present, 381,780 B) |
| output | `…/studies/long_context_v1/robomme/stage/wsm_pooled/rmme_pi_100k` |
| store size (est.) | ≈ 510 MB (98,215 × 512 fp16 `p` + 2048 fp16 per-frame lang) |
| max_run | **12,600 s** — 98,215 frames / 5.97 fr/s/GPU (measured, RoboCerebra) / 8 GPUs = 2,056 s, ×1.5 for 3-view vs 2-view = 3,084 s, ×2.5 = 7,710 s, + 3,600 s startup |
| queue / priority / instance | `fss-tri-cam-robotics-p5-48xlarge-us-west-2` / **400** / `ml.p5.48xlarge`, `SM_USE_RESERVED_CAPACITY=1` |

**One prerequisite upload** (coordinator action — the launcher never writes to S3). The tap builds
its policy from `pi05_rc_mg60_bal33`, which lives in `internal_training/robocasa/
wsm_robocasa_configs.py` and is **not** in the openpi fork (the fork ships an 81-line variant that
defines no mg60 config; shas `4609f0fa…` vs `42dff5df…`, verified 2026-09-02):

```bash
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2020-01-01' \
    -czf /tmp/wsm_configs.tgz -C /home/sarveshp/Research/TRI/internal_training robocasa/wsm_robocasa_configs.py
aws s3 cp /tmp/wsm_configs.tgz s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1/artifacts/configs/026255fa3593dff3acd9559edf9013cb73f722f457e5803cd6e168829ff893a3.tgz
```

(sha256 `026255fa3593dff3acd9559edf9013cb73f722f457e5803cd6e168829ff893a3`, 2,587 B, deterministic —
reproduce with `submit_robomme_stage.py --print-configs-upload`.)

### 5.2 (b) Stage E — RoboMME as a 4th tap

RoboMME **already contributes edges** to the label artifact — 86,711 edges / 63,706 positives /
23,260 gate pairs (h14 §42.3), the second-largest block after RoboCasa — and they are **silently
dropped today** because its segments have no episode (`episode_of == -1`, no tap). The tap turns
them on.

**Recommendation: ADD TWO CELLS TO THE SAME Stage-E NODE RUN, do not replace the 3-tap cells.**

| option | verdict |
|---|---|
| join the running 3-tap design (replace its cells with 4-tap) | **no.** The cell ids are dry-run derived, the §20.x domain-mixing readings are calibrated on the 3-domain corpus, and a 4th domain with 86,711 edges materially re-weights the objective. That is a design change, not a data top-up |
| **same node, +2 cells `E1b-4tap` / `ctrl-0b-4tap`** | **yes.** 8 cells run on 8 GPUs in ~30 min wall (§41.3) against a 21,600 s ceiling, so 10 cells cost no extra node and give the paired 3-tap-vs-4-tap comparison for free |
| follow-on E1b-4tap retrain | fallback if the tap misses the Stage-E submit |

Timing: the tap is **independent of the whole pass-1 → embed → pass-2 chain** (it needs only the
dataset and two checkpoints) and takes ≈3.5 h. `<V2C>` is ≥16.5 h out (pass-2 delta `max_run`
59,540 s). **The tap can be ready first if fired now.**

Hard blocker if it is not: `train_stage_e.py:196` fails closed on a multi-domain cell whose tap has
no raw-tap effective rank. The tap job produces it (`tap_stats_robomme.json`); it must then be
merged into `raw_tap_erank_stratified.json` and the robomme bar pre-registered as
**fail = 0.80 × raw, pass = fail × (8.0/6.5)**, exactly as robocerebra's 4.50 → 3.60 / 4.43 was.
Expected: the same frozen backbone and the same 192-token geometry as robocasa (10.121), narrowed by
RoboMME's flat 2-view scenes — a value between remembench's 7.47 and robocasa's 10.12 is the
prior, but **the bar is set from the measurement, never from this prior**.

### 5.3 (c) ω stores

**One store, not two.** `train_stage_e.py --export-omega` writes
`omega/robomme/<Task>/demo_%06d/w.npz` = `{w [F,512] fp16, frame_indices, lang_global}` over the
whole episode. The demo prefix is `w[frame_indices < exec_start_idx]`; `exec_start_idx` travels in
the tap store's `p.npz` (additive field) so the split is recoverable without re-reading parquet.

| quantity | value |
|---|---|
| episodes | 1,600 |
| ω rows | 98,215 |
| store size | ≈ 100 MB fp16 |
| K_demo default | **8** (uniform-inclusive over the demo grid) |
| K_live default | **16** (`cond_window` of the sealed GDN w16+dropout arm) |
| window | **24** |

Why K_demo = 8: the paper's budget is 32 frames over the *whole* prefix, and its own demo/live split
is length-determined (median 26/32 slots demo on VideoPlaceOrder). Our 8/16 split keeps the live read
at its sealed capacity so **M3 − M1 is a pure addition**, not a reallocation. Demo grid points range
3–144 (measured), so 8 is a genuine sub-sample everywhere except the shortest PatternLock episodes,
where the left-pad branch fires.

### 5.4 (d) Training side

`build_window()` is imported by the loader; the batch gains one `[24,512]` `wsm_w_window` instead of
`[16,512]`. `RoboMMEInputs` already accepts an arbitrary-K `wsm_w_window`
(`data.py:127` validates `[K≥1, 512]` finite), so **no transform change is needed**. Config diff vs
the sealed multitask recipe:

```yaml
# arms.py:  + "v4_wsm_gdn_demo8_live16_drop02"
# config.py: deltanet_window 16 -> 24 for that arm id only
model:
  cond_type: gated_deltanet
  cond_window: 24            # <- the one-line diff (8 demo + 16 live)
  wsm_cond_history_dropout: 0.2
data:
  workspace_demo_frames: 8   # <- selects the prefix slots
```

`pos_decay_bias` becomes `[24, num_heads]`, so serve auto-detects the window from the checkpoint
exactly as it does today, and a K mismatch is still a hard error
(`wsm_current_cond.py`: "omega window K != trained window_len").

### 5.5 (e) Serve side

`eval/execution_model_server.py` + `eval/workspace_runner.py` already ingest the demo video:
`capture_workspace_observation` requires the `episode_restart` envelope and appends
`video_history` frames before any execution frame, and `OnlineWorkspaceRunner` is constructed with
`require_video_history=True`. The extension is:

1. record `exec_start_idx = len(video_history)` on the session at `episode_restart`;
2. produce ω online through the **pi0.5 tap + Stage-E encoder** (D7-gated, `--lang-mode stored`,
   with the `encoder_step` check) instead of the legacy frozen-SigLIP `CheckpointWorkspaceEncoder`;
3. replace `requested_omega_steps` with `rmme_demo_prefix.arm_window(arm, …)`, which reproduces the
   old rule bit-for-bit for `m1_live_only` (proven in the self-test).

**Do not modify `workspace_runner.py` in place** — it is the sealed v1 legacy path and its arms are
scored. The extension goes in a sibling module selected by arm id.

### 5.6 (f) Preflight gates before any policy submit

| gate | assertion |
|---|---|
| G-parity | D7 identity on held-out training demos: worst cos ≥ bar, max\|Δ\| ≤ fp16 floor, under `--lang-mode stored`, encoder = the `encoder.pt` that exported the store |
| G-flow | `w_t` provably non-zero at the modulator on a real batch, through the **sealed openpi fork**, for each of M1/M2/M3 (§22.4 pattern). `from_pretrained` bypasses the modulator patch — the eval path must exist |
| G-distinct | M1/M2/M3 windows differ on the same episode/step: asserted in `rmme_demo_prefix._self_test` and re-asserted server-side |
| G-shape | `pos_decay_bias.shape[0] == k_demo + k_live` in each restored checkpoint |
| G-prefix | on the 9 demo tasks `prefix_valid.all()`; on the 7 no-demo tasks `not prefix_valid.any()` — and the arm still produces finite actions |
| G4 (standing) | the aux/contrastive term must beat chance by end-of-canary, else HOLD |

### 5.7 The one diagnostic this design owes

The GDN is a *decaying* recurrence read out at the newest slot; the demo prefix is strictly the
oldest part of the window. If the decay washes it out, M3 ≡ M1 for mechanical reasons and R1 would
read as "the demo does not help" when it actually means "the read cannot see it". Registered
diagnostic, run on the M3 canary before the full 800: **ablate the prefix slots at inference
(replace with the clamp fill) and measure the change in the conditioning vector's norm.** A change
below the fp16 floor is a mechanism failure, not a result, and promotes the fallback arm — a learned
`[2, w_dim]` segment embedding added to ω before projection (+1,024 params, same subtree).

---

## 6. READY SET

Prerequisite first (coordinator, one upload — see §5.1). Then, verbatim:

**READY — RoboMME pooled tap, p5 @400:**

```
/home/sarveshp/Research/envs/sm_launch/bin/python scripts/launch/submit_robomme_stage.py \
  --phases tap --priority 400 --max-run-seconds 12600 \
  --openpi-source-s3 s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1/code/openpi/fd2522761b1d912be1687891657b9f9af504b74b61d14bfa6d1b75d4de105e1e.tgz \
  --image-uri 141701954645.dkr.ecr.us-west-2.amazonaws.com/sarvesh.patil-groot-dexjoco@sha256:798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2 \
  --confirm-submit
```

Run from `/home/sarveshp/Research/TRI/wsmv2`. Dry run = the same line without `--confirm-submit`
(validated 2026-09-02, offline, clean).

Expected early markers, in order: `[entry] staged wsm_configs …, sha verified`;
`[tap] root=… parquet=1600`; `[rmme-tap:plan] OK 2 episodes …`; then eight
`[rmme-tap] 200 episodes to do (shard i/8, 16 tasks)`; finally
`[tap] complete: 1600 done markers / 1600 p.npz` and `tap_stats_robomme.json`. Any shard death
prints 25 lines of every shard log and fails the job.

| gated on the tap | fires |
|---|---|
| Stage-E `E1b-4tap` / `ctrl-0b-4tap` (+2 cells, same node) | tap SUCCEEDED **and** robomme's eff-rank merged into `raw_tap_erank_stratified.json` **and** `<V2C>` |
| `--phases omega,parity` | a Stage-E `encoder.pt` exists |
| M1 / M2 / M3 / M3-ctrl policy arms | D7 parity PASS + §5.6 preflight |

---

## 7. OPEN DESIGN QUESTIONS THE PAPER DOES NOT SETTLE

| # | question | chosen default | why |
|---|---|---|---|
| Q1 | Does FrameSamp treat the demo specially? | **No — and neither do we at the ω level; we do at the window level** | The paper is silent; the reimplementation's docstring is explicit that the inclusive prefix already contains the demo. But their split is length-determined, which confounds demo capacity with episode length. Fixed 8/16 slots make it a controlled factor |
| Q2 | K_demo? | **8** | Keeps K_live at the sealed 16 so M3 − M1 is a pure addition. 8 is a real sub-sample of the measured 3–144 demo grid points |
| Q3 | Segment flag representation? | **fixed slot allocation (positional)** | Zero new parameters, one-line config diff, window length stays structurally readable from `pos_decay_bias`. Learned `[2, w_dim]` embedding is the pre-registered fallback, promoted only by the §5.7 diagnostic |
| Q4 | The 7 tasks with no demo video? | **clamp prefix slots to the earliest ω**; report separately (P9) | Zero-fill would feed the conditioner a vector the encoder never produces — a second distribution shift confined to exactly the counting stratum. Clamping matches the existing `requested_omega_steps` clamp-to-zero convention |
| Q5 | Which frozen network to tap? | **`pi05_on/149999`** (the RoboCasa pretrain the arms init from) | Same frozen net as robocasa/remembench ⇒ RoboMME adds **no** new network to Stage E, unlike RoboCerebra. Alternative (RoboMME's own config + norm stats) is the fallback if the A3 audit says irreconcilable |
| Q6 | Views? | **the policy's 3-slot tree with the right wrist zeroed** | Matches `RoboMMEInputs` and keeps the 192-token geometry so the A3 audit is apples-to-apples. The paper uses front-only for FrameSamp |
| Q7 | Where does Modul enter? | **the single sealed `wsm_tanh_cond` seam**, not every layer | Changing the seam is a second factor. Recorded as an approximate-parity cell |
| Q8 | Modul is cross-attention; ours is a recurrence | **keep the GDN** | Swapping the read is a second factor and a new parameter subtree. §5.7 is the diagnostic that would force it |
| Q9 | Training steps / LR / EMA? | **the sealed v4 recipe** (60k, batch 64, warmup 3,000, 5e-5→5e-6, EMA .999) | The paper's Appendix B.4 is not reproduced in the HTML; our own base is the correct comparator and is already sealed at 17.875 |
| Q10 | Are the 3 no-demo permanence/counting suites in scope? | **yes, but only as the P9 no-demo row** | They are the C3 target and must not be silently averaged into the R1/R2 readings |


---

# ADDENDUM — 2026-09-02, after the tap fired (c) (d) (e) (f) + READY skeletons

Two findings below change the plan. Both were caught by CPU validation on real data, before any
policy submit and before any GPU time was spent on an arm.

## A1. DEFECT FOUND AND FIXED — the policy ω store needs a SERVE-ALIGNED grid

The `wsm_pooled` grid is `arange(0, n, 8)`, anchored at frame 0. At serve the current frame's
episode-global index is `exec_start_idx + 16·decision` (execution horizon 16), and 16 ≡ 0 (mod 8),
so **every serve frame is congruent to `exec_start_idx` mod 8**.

Measured over the 900 episodes that carry a demonstration:

| `exec_start_idx % 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| episodes | **167** | 89 | 220 | 71 | 103 | 74 | 108 | 68 |

**Only 167 / 900 (18.6 %) are aligned.** On the other 81.4 % *not one live frame ever lands on the
store's grid* — an online producer would emit **zero** live ω and the "live" half of the window
would silently fill from demo rows. The serve self-test measured exactly this before the fix: live
coverage 0.000 on 9 of 12 episodes, 0.507 on the aligned ones.

**Fix — a second grid for the policy store only** (`rmme_demo_prefix.serve_aligned_grid`):

```
grid(n, D) = arange(0, D, 8)  ∪  arange(D, n, 16)          # no trailing final frame
```

Demo half keeps stride 8 (every demo frame arrives at reset, so it is exactly reproducible and
gives `demo_prefix_slots` a denser pool). Live half strides at exactly the execution horizon, so
the served sequence and the stored sequence are **identical** — D7 then certifies the whole window
instead of only its prefix. The trailing `n-1` append is dropped deliberately: the server never
observes the final frame, so keeping it would make the stored sequence one row longer than the
served one and the identity would be false at the last decision.

| grid | tapped frames | demo | live | used by |
|---|---:|---:|---:|---|
| `wsm_pooled` (cross-domain) | 98,215 | 36,907 | 61,308 | the Stage-E **encoder corpus**; A3 audit; the 4-tap retrain |
| `serve_aligned` (policy) | **67,491** | 36,907 | 30,584 | the **policy ω store** M1/M2/M3 read |

**This does not disturb the running tap or the Stage-E design.** The encoder corpus is unchanged,
so A3, the eff-rank bar and the 10-cell run are untouched. What is needed is one more tap pass on
the serve-aligned grid — a new `tapserve` phase in the same entry, 67,491 frames.

Post-fix serve self-test: **live coverage exactly 1.000 on 12/12 real episodes, 0 off-grid frames**,
and off-grid is now a hard error rather than a counted skip.

## A2. DEFECT FOUND — the sealed `pos_decay_bias` zero-init makes M3 mechanically M1

The §5.7 diagnostic, run at initialisation on 96 windows off real episode grids:

| `pos_decay_bias` init | clamp rel-median | above fp16 floor | verdict |
|---|---:|---:|---|
| **0.0 — the sealed default** | **6.85e-06** | **0.00** | FAIL |
| −8 on the 8 demo slots only | 1.79e-05 | 0.00 | FAIL |
| −8 on all 24 slots | 4.73e-01 | 1.00 | PASS |
| −4 on the 16 live slots only | 1.79e-01 | 1.00 | PASS |
| **−4 on all 24 slots — pre-registered** | **3.82e-01** | **1.00** | PASS |

`gamma_i = exp(-softplus(W_decay·ω_i + pos_decay_bias_i))`. At the zero-init the 8-slot prefix moves
the conditioning vector by ~7e-6 relative — *below the fp16 floor on 100 % of windows*. Row 2 shows
the decay that erases the prefix is the one applied at the **live** slots, not within the prefix.

Caveat stated plainly: this is measured at random init with random ω, so it bounds what the
architecture can **propagate at initialisation**, not what a trained model does. It matters because
the gradient to the prefix slots carries the same product of gammas — a model that starts at 1e-5
leakage has almost no signal with which to learn otherwise.

**Pre-registered fix: `pos_decay_bias` init = −4.0, applied IDENTICALLY to M1/M2/M3/M3-ctrl**, so
the ablation stays one-factor (the factor is still the window). It makes the M-family differ from
the sealed parent by a second thing, which is exactly why **M1, not the sealed arm, is M3's paired
baseline**. Locked as a regression in the diagnostic's self-test: zero-init must FAIL and −4 must
PASS, or the finding has been silently undone.

**Requires a 2-line openpi-fork diff** (the fork is content-addressed and separately versioned —
this is *not* a `robomme_integration/` edit, and the default preserves every sealed checkpoint):

```
models/wsm_current_cond.py  WSMGatedDeltaNetConditioner.__init__
  + pos_decay_bias_init: float = 0.0
  - jnp.zeros((self.window_len, self.num_heads), dtype=jnp.float32)
  + jnp.full((self.window_len, self.num_heads), pos_decay_bias_init, dtype=jnp.float32)
models/pi0_config.py        plumb `wsm_cond_pos_decay_bias_init` to the conditioner
```

It moves the openpi tarball's content address, so re-derive every run_id from the dry run at fire
time.

**D2 ledger — fix applied, re-addressed, verified (2026-09-02, executor 2):**

| item | value |
|---|---|
| diff applied to | the **policy** archive `24bd889d…` (what every `V4_ADVANCED_GDN_ARMS` cell pins). 3 files, not 2: `models/wsm_current_cond.py` (kwarg + `jnp.full` init + sentinel `_WSM_POS_DECAY_BIAS_INIT = True` + finiteness guard), `models/pi0_config.py` (`wsm_cond_pos_decay_bias_init: float = 0.0` + finiteness guard + refuses the kwarg on `wsm_cond_type="tanh"`), `models/pi0.py` (passes the kwarg to the constructor — the plumbing the pre-registration attributed to `pi0_config.py` alone) |
| new address | `s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1/code/openpi/445d9902a5502d6ce4661c8c42dfa8a2b3ecc3439cb67b41941d1df2b61574dd.tgz` — 524,639 B, uploaded 2026-09-02 15:24:02Z, sha256 re-verified on download; `build_deterministic_archive.py` reproduces it byte-for-byte from the patched tree AND from the extracted S3 object, and reproduces `24bd889d…` from the unpatched tree; `diff -rq` = exactly the 3 files, entry lists identical (205) |
| tap archive `fd252276…` | **does not need the diff**: `Pi05BackboneTap` → `get_config("pi05_rc_mg60_bal33")` → `Pi0Config(pi05=True)` → `wsm_tanh=False` → `pi0.py` never constructs any WSM conditioner; `omega`/`parity` phases use the torch Stage-E encoder. It also carries no `_WSM_PTRM` sentinel, so the launcher could never pair it with an M-arm |
| −4.0 identical on all four arms | staged `build_train_config`: `wsm_cond_pos_decay_bias_init = −4.0` for M1/M2/M3/M3-ctrl; `pos_decay_bias` = `[16,2]` / `[8,2]` / `[24,2]` / `[24,2]`, every value −4.0, fp32; γ at zero logit = 0.982 |
| verification numbers | JAX fork on CPU (random weights): M3 clamp-prefix relative change **0.586** at init (zero-init regime ≈ 5e-6); ablation `--self-test` on 48 real-grid windows: zero-init 4.94e-06 → FAIL, −4.0 3.75e-01 above-floor 1.000 → PASS, prefix-blind control 0 → FAIL |

## A3. What was built and validated (all CPU, real data, node argv)

| deliverable | file | validation |
|---|---|---|
| (c) shared index rule | `workspace_models/features/rmme_demo_prefix.py` | self-test **PASS on all 1,600 real episodes**; live rule **bit-identical** to sealed `requested_omega_steps` on 60 row-space + 72 step-space geometries; `serve_aligned_grid`; `row_for_step`; `load_episode`; `window_for_step` = the one call both sides import |
| (d) arm configs | `workspace_models/overlays/rmme_arms.py` | `--check` PASS with and without openpi; idempotent; patches `ARM_IDS`/`V4_ARM_IDS`/`WORKSPACE_ARMS`/`V4_NEW_PARAMETER_SUBTREES`/`V4_DELTANET_RECIPES`/`WORKSPACE_WINDOWS`; asserts the sealed parent is untouched |
| (d) dataset plumbing | `workspace_models/overlays/rmme_workspace_dataset.py` | `--dry-run` on **20 real episodes (15 demo, 5 no-demo PickXtimes), 120 execution rows, 480 arm×row windows**: demo slots = prefix ω rows, live slots = causal window, M3 == [M2 ; M1] exactly |
| (e) serve side | `workspace_models/overlays/rmme_serve_omega.py` | `--self-test` **678 decision-windows over 12 real episodes**: reset grid exact, live coverage **1.000**, off-grid fatal, causality holds; `d7_preflight` wraps the shipped gate with the `encoder_step` + `encoder_best.pt` + `taskmean` refusals |
| (f) prefix ablation | `scripts/analysis/rmme_prefix_ablation.py` | `--self-test` PASS: zero-init FAILs, −4 PASSes, prefix-blind negative control scores exactly 0.0 |
| tap, both grids | `workspace_models/features/rmme_pooled_tap.py` | `--plan-only` shards on both grids: 11,366 frames (`wsm_pooled`) / 7,926 (`serve_aligned`) per shard 0/8 |
| entry + launcher | `robomme_stage_entry.sh`, `scripts/launch/submit_robomme_stage.py` | `bash -n` clean, dry-run clean, `tapserve` wired, max_run re-derived |

`robomme_integration/` is untouched — verified by `git status`: the only new paths are under
`workspace_models/`, `scripts/`, and the repo root entry.

## A4. Corrections to the body of this document

| § | was | now |
|---|---|---|
| 5.1 max_run | 12,600 s (98,215 frames) | **16,800 s** for `tap,tapserve` (165,706 frames: 3,468 s at the measured rate ×1.5 for 3-view = 5,202 s, ×2.5 = 13,005 s, + 3,600 s startup). Single-grid stays 12,600 s |
| 5.3 ω store | one store | **two tap stores, one ω store.** The encoder corpus is `rmme_pi_100k`; the policy ω store is exported from `rmme_pi_100k_serve` |
| 5.5 serve side | "replace `requested_omega_steps`" | built as `rmme_serve_omega.py`; `OMEGA_TAP_OVERRIDE` aside, the `omega`/`parity` phases now default to the **serve-aligned** store and refuse to run against the cross-domain one |
| 5.7 diagnostic | "owed" | **built and already fired at init — it FAILED the sealed default**, which is why A2 exists |
| Q3 | segment flag = fixed slot allocation, zero new params | unchanged, but it only works with A2's decay-bias init; without it the slots are unreadable |

## A5. READY SKELETONS — M1 / M2 / M3 / M3-ctrl (p5 @400, NOT submitted)

`max_run` per the coordinator's formula: sealed `v4_s0` wall **26,416 s** × 1.25 (H200→H100) × 2.5
+ 3,600 = **86,150 s** (cap 86,400). Honest note: a GDN arm measured 1.5× the base in v4 Phase-A
(16,200 s vs 10,800 s), so the *expected* wall is 33,020 × 1.5 ≈ **49,530 s** and 86,150 s is
1.74× that, not 2.5×. Carrying a true 2.5× would need 123,825 s, over the cap — so these arms
**must** rely on mid-run checkpoint sync (`save_interval` 5,000 + `remote_resume: true`), and a
timeout is a resumable, expected outcome rather than a lost run.

Placeholders, all resolved by the Stage-E run: `<OMEGA_INDEX_S3>` / `<OMEGA_INDEX_SHA>` are the
published all-16 workspace index over the ω store (`load_workspace_index` fails closed unless all
16 tasks are present in canonical order — the store covers all 1,600 episodes, so it is
publishable). E1b-4tap for M1/M2/M3, ctrl-0b-4tap for M3-ctrl.

Run every line from `/home/sarveshp/Research/TRI/wsmv2`. **Re-run the dry run immediately before
submitting and record the identity it prints** — `run_id` folds the source-tree sha and the
openpi tarball, and both move with the A2 fork diff.

```
# M1 — live-ω only, K=16                       (paired baseline for M3)
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --scope multitask --arm v4_wsm_gdn_live16_drop02 \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_E1B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_E1B_4TAP> \
  --confirm-submit

# M2 — demo-ω only, K=8
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --scope multitask --arm v4_wsm_gdn_demo8_drop02 \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_E1B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_E1B_4TAP> \
  --confirm-submit

# M3 — [8 demo ; 16 live], K=24                 THE PARITY ARM
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --scope multitask --arm v4_wsm_gdn_demo8_live16_drop02 \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_E1B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_E1B_4TAP> \
  --confirm-submit

# M3-ctrl — identical recipe, ctrl-0b ω store   (isolates deliberative structure)
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --scope multitask --arm v4_wsm_gdn_demo8_live16_drop02_ctrl0b \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_CTRL0B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_CTRL0B_4TAP> \
  --confirm-submit
```

**READY — the serve-aligned tap** (fire alongside or after the running tap; independent of Stage E):

```
/home/sarveshp/Research/envs/sm_launch/bin/python scripts/launch/submit_robomme_stage.py \
  --phases tapserve --priority 400 --max-run-seconds 12600 \
  --openpi-source-s3 s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1/code/openpi/fd2522761b1d912be1687891657b9f9af504b74b61d14bfa6d1b75d4de105e1e.tgz \
  --image-uri 141701954645.dkr.ecr.us-west-2.amazonaws.com/sarvesh.patil-groot-dexjoco@sha256:798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2 \
  --confirm-submit
```

Gate order before any of the four policy submits, unchanged from §5.6 plus one new row:

| # | gate |
|---|---|
| 1 | `tapserve` SUCCEEDED, 1,600/1,600 `p.npz` on the serve-aligned grid |
| 2 | ω exported from the serve-aligned tap; D7 `--lang-mode stored` PASS with the `encoder_step` check |
| 3 | **A2 fork diff applied and the openpi tarball re-addressed**; `pos_decay_bias` init −4 confirmed in the built config |
| 4 | `rmme_prefix_ablation.py` on the M3 canary checkpoint: verdict **PASS** (clamp rel-median ≥ 0.05, above fp16 floor on ≥ 95 %) |
| 5 | G-flow / G-distinct / G-shape / G-prefix per §5.6 |

---

# ADDENDUM 2 — 2026-09-02, executor 2: A2 applied to the POLICY archive, verified on CPU, READY lines re-derived

Handover state: the previous executor had built and uploaded `445d9902…` but had neither verified nor
recorded it, and its READY skeletons could not run (A8). Everything below is CPU-only; no job was
submitted, terminated or modified; both local GPUs were left to the pass-2 judge.

## A6. Archive re-address

| | old | new |
|---|---|---|
| policy archive (M1/M2/M3/M3-ctrl) | `24bd889d3c0b95a7b01cd6ad30a91fdc266fa115fb2ef5ec89fe45c9c5260900` (523,956 B) | **`445d9902a5502d6ce4661c8c42dfa8a2b3ecc3439cb67b41941d1df2b61574dd`** (524,639 B) |
| tap archive (tap / tapserve / omega / parity) | `fd2522761b1d912be1687891657b9f9af504b74b61d14bfa6d1b75d4de105e1e` | unchanged — needs no diff (D2 ledger, A2) |
| S3 prefix | `…/studies/long_context_v1/code/openpi/<sha>.tgz` | same prefix |
| launcher pairing | `PTRM_OPENPI` for `V4_ADVANCED_GDN_ARMS` | the staged `launch.py` selects `445d9902…` **only** for the four M-arm ids; every sealed cell keeps its own pin; `--openpi-source-s3` with anything else is refused |

Content of the diff, verified by extraction: `24bd889d` + {`models/wsm_current_cond.py`,
`models/pi0_config.py`, `models/pi0.py`} and nothing else. Node-side `_WSM_V4_ADVANCED` sentinel
greps (`gpu_train_entry.sh:376`) all pass on `445d9902`; the new `_WSM_POS_DECAY_BIAS_INIT` sentinel
is checked launcher-side (`rmme_arms.assert_openpi_archive`, PASS on `445d`, FAIL on `24bd`). A
mis-paired archive cannot fail silently: the staged config passes `wsm_cond_pos_decay_bias_init` to
`Pi0Config`, which is a `TypeError` on any archive lacking the field.

Scratch identities, so nobody chases them: `7c198140…` (arch_ref) is the previous executor's tarball
of its local staging tree (`openpi_stage`, 13 files off `fd252276`); not on S3, referenced by nothing.

## A7. CPU verification of the built config (gate 3 of the A5 table, and G-flow / G-shape at init)

Path exercised: staged `build_train_config(arm)` + `validate_train_config` → `nnx.eval_shape(model.create)`
through `pi0.py.__init__` — the same constructor `BaseModel.load` runs at serve
(`policy_config.create_trained_policy` → `model.load(restore_params)` → `nnx.eval_shape(self.create)`
→ `nnx.merge`), so on the pi side there is no `from_pretrained`-style bypass of the conditioner (that
gotcha is groot-side). Env `openpi-jax-latest`, `JAX_PLATFORMS=cpu`, archive `445d9902` unpacked.

| arm | K | `pos_decay_bias` | all −4.0 | ‖c(w)‖ | ‖c(w)−c(0)‖ | dropout fires (train) | eval refuses rng | K≠window hard error | clamp-prefix rel @init |
|---|---:|---|---|---:|---:|---|---|---|---:|
| M1 `v4_wsm_gdn_live16_drop02` | 16 | [16, 2] | yes | 5.22e-3 | 5.22e-3 | yes | yes | yes | n/a |
| M2 `v4_wsm_gdn_demo8_drop02` | 8 | [8, 2] | yes | 4.24e-3 | 4.24e-3 | yes | yes | yes | n/a |
| M3 `v4_wsm_gdn_demo8_live16_drop02` | 24 | [24, 2] | yes | 4.73e-3 | 4.73e-3 | yes | yes | yes | **0.586** |
| M3-ctrl `…_ctrl0b` | 24 | [24, 2] | yes | 4.73e-3 | 4.73e-3 | yes | yes | yes | **0.586** |

Controls: default `pos_decay_bias_init=0.0` still yields an all-zero `[24,2]` (sealed checkpoints
reproduce); `Pi0Config(wsm_cond_type="tanh", wsm_cond_pos_decay_bias_init=-4.0)` is refused.
`rmme_prefix_ablation.py --self-test`: PASS (zero-init FAIL 4.94e-06 / −4.0 PASS 3.75e-01 / blind
control FAIL); its `POS_DECAY_BIAS_INIT` equals the staged config's kwarg on all four arms and
`load_conditioner(decay_bias_init=<config kwarg>)` yields `[K,2]` all −4.0 for K = 16/8/24/24.
Scripts: `scratchpad/a4_verify/verify_cond.py` (+ `.json`).

## A8. Overlay manifest defect — found by the first staged dry run, fixed

The sealed `launch.py` rejects the M-arm ids outright (`--arm` choices), so the four arms can only
launch from the overlay's **staged copy** (`rmme_arms.py --stage-tree`). The A5 skeletons omitted
that step. Dry-running the staged tree then showed the scientific manifest mis-describing the arms:

| manifest field | before | after (fix in `rmme_arms._build_patches` item 5) |
|---|---|---|
| `mechanism.steering` | `null` (arm absent from `_arm_spec.deltanet_windows`) | `gated_deltanet_k16` / `k8` / `k24` / `k24` |
| `mechanism.train_history_dropout` | **0.5** (sealed 0.2-set names only the two sealed GDN arms) while `config.py` trained 0.2 | 0.2 |
| `mechanism.pos_decay_bias_init` | absent | −4.0 |
| `mechanism.omega_window` | absent | `{k_demo, k_live, window, read, omega_cell}` per arm |
| `mechanism.advanced_openpi_capability` | absent | `source_sha256 445d9902…`, `base 24bd889d…`, requires `pos_decay_bias_init`, parent `v4_wsm_gdn16_drop02` |

`stage_tree` now asserts the windows and the init reached `_arm_spec`. `--check` never exercised
`_arm_spec`, which is how this got through A3. Sealed `robomme_integration/` untouched (diff of the
staged copy vs sealed = exactly `training/arms.py`, `training/config.py`, `launch.py`,
`eval/workspace_runner.py`).

Verified from the re-staged tree, every arm: `sources.openpi.sha256 = 445d9902…`,
`OPENPI_REQUIRED_SENTINEL=_WSM_V4_ADVANCED`, `OPENPI_FORK_S3 = …/445d9902….tgz`,
`checkpoint_policy = {save_interval 5000, remote_resume true}`, `WSM_SAVE_INTERVAL=5000` (the entry
asserts `== 5000`), `SM_USE_RESERVED_CAPACITY=1`, queue `fss-tri-cam-robotics-p5-48xlarge-us-west-2`,
`ml.p5.48xlarge`, priority 400, max_run 86,150, volume 400; tags at submit
(`launch.py:962-963`): `tri.project=LONG-CONTEXT-VLA`, `tri.owner.email=sarvesh.patil.pi@tri.global`.

## A9. Sealed-identity drift (finding, not caused here) and the M0 wall-time cross-check

| | M0 as submitted (08-31 15:08) | sealed dry run today |
|---|---|---|
| `sanitized_source_tree_sha256` | `14851174bba76788…` | `ad536b4a48690ce8…` |
| `v4_s0` run_id | `mt-v4-all16-v4_s0-seed0-de6c37b2b8f53b36` (SUCCEEDED) | `…-bf7204e131565f04` |
| cause | — | `eval/project_exact_{eval,runner,server}.py` (09-01 18:06) + `CAMPAIGNS.md` (09-02 04:04) entered `robomme_integration/` after M0's submit |

Consequence: none for M0 (identity is minted at submit; the SUCCEEDED job and its manifest are the
anchor) and none for the M-arms (minted at fire time from the staged tree). The body's "the RoboMME
run_id does not move" is true of this packet — the tree hash was unchanged by it (re-verified after
staging) — but false as a standing statement; every `.py`/`.md` under `robomme_integration/` folds in.

M0 measured on p5 (`describe-service-job`, read-only): queue wait 6,854 s, **job wall 28,473 s**
(incl. startup), `attemptDurationSeconds` 79,200, both SCP tags present. So the A5 `max_run` note
reads: 86,150 s = 2.02× a GDN arm's expected wall on p5 (28,473 × 1.5), not 2.5×, and the ×1.25
H200→H100 factor in A5 is spare margin (M0 already ran on H100). Mid-run sync + `remote_resume` is
what makes a timeout resumable — unchanged.

## A10. READY — M1 / M2 / M3 / M3-ctrl, re-derived (p5 @400, NOT submitted)

Placeholders `<OMEGA_INDEX_S3_*>` / `<OMEGA_INDEX_SHA_*>` are unchanged from A5 and still unresolvable
(A11). The launcher enforces `<OMEGA_INDEX_S3_*>` =
`s3://…/studies/long_context_v1/artifacts/robomme/workspace/all16/<OMEGA_INDEX_SHA_*>.json`.

Step 0 — stage (re-run at fire time; the staged tree sha folds into every run_id; the sealed
checkout is never mutated):

```
PYTHONPATH=. /home/sarveshp/Research/TRI/internal_training/.venv/bin/python \
  -m workspace_models.overlays.rmme_arms --stage-tree /home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_m_arms/robomme_integration --overwrite \
  && ln -sfn /home/sarveshp/Research/TRI/wsmv2/scripts /home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_m_arms/scripts
```

Run the four lines from the staged parent (NOT from `wsmv2/` — with `-m`, the cwd precedes
`PYTHONPATH`, and `wsmv2/robomme_integration` would shadow the staged package; the sealed launcher
then rejects the arm id, loudly). Dry run = the same line with `--dry-run` in place of
`--confirm-submit`; record the identity it prints.

```
# M1 — live-ω only, K=16                       (paired baseline for M3)
cd /home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_m_arms && PYTHONPATH=$PWD \
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --source-dir $PWD/robomme_integration \
  --scope multitask --arm v4_wsm_gdn_live16_drop02 \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_E1B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_E1B_4TAP> \
  --confirm-submit

# M2 — demo-ω only, K=8
cd /home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_m_arms && PYTHONPATH=$PWD \
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --source-dir $PWD/robomme_integration \
  --scope multitask --arm v4_wsm_gdn_demo8_drop02 \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_E1B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_E1B_4TAP> \
  --confirm-submit

# M3 — [8 demo ; 16 live], K=24                 THE PARITY ARM
cd /home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_m_arms && PYTHONPATH=$PWD \
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --source-dir $PWD/robomme_integration \
  --scope multitask --arm v4_wsm_gdn_demo8_live16_drop02 \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_E1B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_E1B_4TAP> \
  --confirm-submit

# M3-ctrl — identical recipe, ctrl-0b ω store   (isolates deliberative structure)
cd /home/sarveshp/Research/TRI/wsm_data/wsmv2_scratch/rmme_m_arms && PYTHONPATH=$PWD \
/home/sarveshp/Research/TRI/internal_training/.venv/bin/python -m robomme_integration.launch \
  --source-dir $PWD/robomme_integration \
  --scope multitask --arm v4_wsm_gdn_demo8_live16_drop02_ctrl0b \
  --hardware p5 --priority 400 \
  --max-run-seconds 86150 --volume-size-gb 400 \
  --workspace-index-s3 <OMEGA_INDEX_S3_CTRL0B_4TAP> \
  --workspace-index-sha256 <OMEGA_INDEX_SHA_CTRL0B_4TAP> \
  --confirm-submit
```

Dry-run identities today with a `00…00` placeholder index (they move with the real sha and with any
re-stage; staged tree `a6cff5015c58b5a0…`, entry `1ac8681b4db8746c…`, openpi `445d9902…`):

| arm | run_id (placeholder index) |
|---|---|
| M1 | `mt-v4-all16-v4_wsm_gdn_live16_drop02-seed0-79e4cb9ff587036c` |
| M2 | `mt-v4-all16-v4_wsm_gdn_demo8_drop02-seed0-468c78eb392ea328` |
| M3 | `mt-v4-all16-v4_wsm_gdn_demo8_live16_drop02-seed0-c5fffe2adc133c36` |
| M3-ctrl | `mt-v4-all16-v4_wsm_gdn_demo8_live16_drop02_ctrl0b-seed0-8c3e28c02064aa47` |

## A11. Gate status (the A5 gate-order table, rows 1–5)

| # | gate | status | blocked on |
|---|---|---|---|
| 1 | `tapserve` SUCCEEDED, 1,600/1,600 `p.npz` on the serve-aligned grid | **BLOCKED** | `rmme-stage-tapserve-0902-151326` (`3f03b4a0-3f58-45bc-9e64-2a101160a58a`) RUNNABLE on p5 |
| 2 | ω exported from the serve-aligned tap; D7 `--lang-mode stored` PASS with `encoder_step` | **BLOCKED** | row 1, and a Stage-E `encoder.pt` (E1b-4tap / ctrl-0b-4tap), itself gated on `rmme-stage-tap-0902-145300` (`f9f914db-1ba6-48e8-86f3-136608e99bc2`, RUNNABLE) + eff-rank merge + `<V2C>` |
| 3 | A2 fork diff applied, tarball re-addressed, −4 confirmed in the built config | **PASS** | — (A6, A7) |
| 4 | `rmme_prefix_ablation.py` on the M3 canary checkpoint, verdict PASS | **BLOCKED** (script PASS at init) | an M3 canary checkpoint, i.e. rows 1–2 then a submit |
| 5 | G-flow / G-distinct / G-shape / G-prefix | G-flow **PASS at init through the D2 fork** (real-batch pass on the node's ω store still owed); G-shape **PASS by construction** (checkpoint re-check at restore); G-distinct **PASS** (A3, unchanged); G-prefix **BLOCKED** | rows 1–2 (no ω store exists) |

Corrections to the body of this document, executor 2:

| § | was | now |
|---|---|---|
| A2 | "2-line openpi-fork diff", plumbed through `pi0_config.py` | 3 files; `pi0.py` carries the constructor plumbing; sentinel + two guards added (D2 ledger) |
| A5 skeletons | run from `wsmv2/`, sealed `robomme_integration.launch` | must run from the **staged** tree (Step 0, A10); the sealed launcher rejects the M-arm ids |
| A3 overlay row | "`--check` PASS" | `--check` never reached `_arm_spec`; the manifest defect (A8) is fixed and asserted at stage time |
| §4 / A3 | "the RoboMME run_id … does not move" | unchanged by this packet; already moved on 09-01 by the W4 eval-lane files (A9) |
| A5 max_run | 26,416 s × 1.25 (H200→H100) × 2.5 + 3,600 | M0 measured 28,473 s job wall on p5; 86,150 s = 2.02× the expected GDN wall (A9) |
| handover | "A2 not yet applied" | applied, built and uploaded by the previous executor; unverified and unrecorded until now |

## A12. 70k recipe (A19) — pointer

Authority: `aug_22/h14_p0_status.md` §49. `launch.py --multitask-train-steps 70000` selects recipe
`v4_70k` (70k steps, warmup 3,500, cosine → 5e-6 at 70k; milestones 10k…60k + 69999 retained and each
exported as `deploy/<step>/{params,assets}`); the 60k path is byte-identical modulo the source-tree fold.

The overlay applies to the changed tree: `--stage-tree` now patches **5** files (A8's four plus
`eval/launch_p5_campaign.py`, where the D2 archive `445d9902…` is registered so an M-arm milestone eval
queue is not refused as unregistered). Re-staged dry runs at 70k (placeholder index `00…0`, staged tree
`48278e8d1100d50e…`, entry `3718fcd18896f77b…`): M1 `…647c0081e54405c1`, M2 `…528257ed0184bb11`,
M3 `…44c9c0b4b4e9ce53`, M3-ctrl `…7366b702de257f25`; all report `steps 70000`, `recipe v4_70k`,
`deploy_milestones` 10k…69999, `pos_decay_bias_init −4.0`, openpi `445d9902…`.

READY (b): the A10 lines with `--multitask-train-steps 70000` inserted after `--arm <id>` (Step 0 first;
placeholders and gates unchanged; A11 rows 1–2 still blocked). Eval of the M-arms is additionally blocked
on an ω serving path in the p5 campaign (§49.7) — a build item needing an explicit go.
