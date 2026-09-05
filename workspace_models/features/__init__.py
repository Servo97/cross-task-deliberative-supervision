"""Frozen-backbone feature extraction + caching for WSM training (the preprocessing 'features' stage).

Runs the FROZEN GR00T N1.7 backbone once per demo frame and caches the workspace-encoder inputs
(patch_tokens, state_emb, lang_emb) so training never touches the VLM. 3-view RoboCasa; the language
fed to the tap is the EXPANDED prompt (Qwen subtask decomposition). See
internal_planning_and_todos/07_wsm_preprocessing_and_revised_plan.md and [[wsm-reference-impl-exists]].
"""
