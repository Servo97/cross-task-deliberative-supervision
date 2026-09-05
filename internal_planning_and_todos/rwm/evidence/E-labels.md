# E — Label quality (SEALED 2026-08-28/31; judge-effort question OPEN → H14.8)

Authority: plan A9, A11–A13, A17; `aug_22/h14_p0_status.md` §14.8, §15, §21, §42, §45–47.

| measurement | result | protocol |
|---|---|---|
| EQUIVALENT precision | 0.933 [0.841, 0.974] — F1 PASS | blind adjudication of 240 edges |
| low vs medium effort agreement | κ 0.838 — F4 PASS (no re-judge bought) | 200-anchor medium pilot vs low |
| CONTRAST precision | 0.172 letter / 0.672 intent — F2 FAIL | same adjudication; intent re-adjudicated (`a9_readjudicate_intent.py`) |
| planted-probe recovery (Qwen alone) | 0.533 — F3 FAIL | 45 planted probes |
| binding table hard negatives | 0.94 precision vs the memory-intent rule | deterministic per-episode binding (`build_binding_annotations.py`) |
| Qwen ∪ binding probe recovery | 39/45 = 0.867 — F3 CLEARED by the union | |
| pass-1 corpus | 3,873/3,873 episodes, 19,853 segments (3 domains) + RoboCerebra 994 episodes / 8,869 segments; 0 schema-invalid, 0 truncated | pass-1 at effort low, NVFP4 (local) / FP8 (node) — quantization affects the JUDGE only |
| pass-2 stores | frozen `fb22b06b…` + §21 delta `28f639a8…` + rcb delta `62fdafc322025fee` (running, NVFP4 @12,288 max tokens) → 3-way union → label v2b chain → `<V2C>` (= `build_edges_ctrl_eb.py` output id) | never mix model/effort/quantization inside one store |
| label v1 defect | 37,809 binding-contrasts were POSITIVES in `bd13c1a48f2dc5be` → v2 relabel | recorded, all cells re-run on v2/v2b |
| edge-store addressing | `edge_store_id` folds model + `caption_segments.py` code_sha → mid-corpus edits orphan a store (freeze rule) | §38.5 makes code_sha provenance-only after pass-2 lands |

Open: does xhigh effort change labels materially (A17 R1/R2/R3)? Pilots queued (H14.8).
