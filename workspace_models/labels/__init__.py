"""WSM salient-patch label pipeline (RoboCasa365, 3 views).

A faithful 3-view RoboCasa port of the DexJoCo reference under Isaac-GR00T/wsm/vlm_label/:
  extract_frames -> qwen_subgoals -> molmo_points -> build_salient_sets
producing per-episode salient-keyframe-patch labels (the WSM-base supervision targets).
The load-bearing pixel->patch geometry lives in ``geometry.py`` (single source of truth).
Governed by internal_planning_and_todos/04_wsm_roadmap.md ("R1 — Labels").
"""
