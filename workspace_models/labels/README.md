# workspace_models/labels/ — WSM salient-patch label pipeline (RoboCasa365, 3 views)

Produces the **WSM-base supervision targets**: per-episode salient image-patch sets at Qwen
keyframes, across the 3 RoboCasa views. A faithful 3-view RoboCasa port of the DexJoCo reference
(`Isaac-GR00T/wsm/vlm_label/`). The frozen-VLM-feature reconstruction + presence targets the
keyframe-patch head trains on are derived from these patch ids (the head reconstructs the frozen
GR00T patch feature at each salient patch).

## Pipeline (4 stages)

| Stage | Script | Env | GPU | Output |
|---|---|---|---|---|
| A0 | `extract_frames` | sim venv (robocasa + lerobot) | no | `ep*_frames.npz` (3 views, 256x256) |
| A | `qwen_subgoals` | vlm_labeler (transformers 4.57) | yes (Qwen3-VL-8B) | `ep*_subgoals.json` (keyframes + per-view objects) |
| B | `molmo_points` | vlm_labeler (trust_remote_code) | yes (MolmoPoint-8B) | `ep*_points.json` (per-view points) |
| C | `build_salient_sets` | any (numpy; PIL for QC) | no | `vlm_episode_*.npz` (per-view patch ids) + `rc_report.json` |

```
python -m workspace_models.labels.extract_frames     --task <Task> --episodes 0,1,2,3,4 --stride 4 --out ~/Research/TRI/wsm_data/wsm_vlm_rc_v0
~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.qwen_subgoals  --task <Task> --in ~/Research/TRI/wsm_data/wsm_vlm_rc_v0 --device cuda:0
~/Research/envs/vlm_labeler/bin/python -m workspace_models.labels.molmo_points   --task <Task> --in ~/Research/TRI/wsm_data/wsm_vlm_rc_v0 --device cuda:1
python -m workspace_models.labels.build_salient_sets --task <Task> --in ~/Research/TRI/wsm_data/wsm_vlm_rc_v0 --qc-dir ~/Research/TRI/wsm_data/wsm_vlm_rc_v0/_qc
```

## The load-bearing bit: geometry (`geometry.py`)

The pixel→patch map MUST match what the frozen GR00T backbone tokenizes, or every label is
silently misaligned. GR00T applies `FractionalCenterCrop(0.95)` → resize 256 → Qwen3-VL 8×8 grid
(64 tokens/view). RoboCasa frames are **native 256** (DexJoCo was 640), so the 640→256 resize
drops out (`RENDER_WH=256`) but the **0.95 crop + 256/243 rescale remain** — it is *not* an
identity map. `eye_in_hand` is the close-up → single points are 3×3-dilated. Global ids over the
3 concatenated 8×8 grids span 0..191 (`VIEW_OFFSETS = {left:0, right:64, eye_in_hand:128}`).

Because RoboCasa unseen tasks have **no sim oracle**, geometry is verified **visually** via
`--qc-dir`: it crops/resizes the frame exactly as GR00T does and overlays the selected patch cells
+ mapped points. Eyeball a few overlays before trusting any labels.

## RoboCerebra / LIBERO (2 views, `--geom pi_libero`)

Same four stages, three flag-level differences — the pipeline is NOT forked.

| | RoboCasa365 | RoboCerebra |
|---|---|---|
| geometry | `geometry.py` (3 views, 192) / `pi_geometry.py` | `pi_geometry_libero.py` (2 views, **128**: `agentview`=`image` @0, `eye_in_hand`=`wrist_image` @64 — the `scripts/robocerebra/omega_tap.py` concat order) |
| stage A0 | `extract_frames` (soup + filter_key, 1 npz/episode) | `extract_frames_robocerebra` (explicit episodes, **1 npz per ground-truth subtask segment**; `subtask_index` + `task_index`→`meta/tasks.jsonl`) |
| Qwen prompt | in `qwen_subgoals.py` | `QWEN_SPECS`/`QWEN_CAPTION` in the geometry module (the prompt names the cameras) |

Stages A/B/C are unchanged code: `--geom` selects the view set, and because each SEGMENT is emitted
as its own short single-task "episode", Qwen decomposes a real subtask instead of a 900-frame
composite. Keyframes stay EPISODE-GLOBAL. A0 needs two envs (pyarrow for the segment table, `av`
for the decode) — see that module's docstring. Driver: `scripts/labels/robocerebra_label_pilot.sh`.

## Differences vs the DexJoCo reference
- 2 views (front+wrist) → 3 (agentview_left/right + eye_in_hand); single source of truth in `geometry.py`.
- `RENDER_WH` 640 → 256 (native RoboCasa); crop 0.95 kept.
- Per-view salient-object dict keyed by the 3 views (was `salient_objects` + `wrist_salient_objects`).
- Qwen system prompt rewritten for the Panda-in-kitchen embodiment + 3 views.
- Oracle IoU scoring → visual `--qc-dir` overlay (no RoboCasa oracle for unseen tasks).

## Status / TODO
- [ ] Pick the 2-3 first target tasks (1 atomic_seen + 1 composite_seen + 1 composite_unseen) and run A0→C; eyeball QC overlays.
- [ ] Decide label-gen compute (which GPU box / SageMaker) for Qwen + Molmo at 50-task scale.
- [ ] Next: `backbone_tap` (3-view) cache from our GR00T pretrain → `train_head` frozen-probe (see [[wsm-reference-impl-exists]]).
