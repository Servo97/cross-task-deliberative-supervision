"""VLA (action-policy) training + eval for wsmv2 — base VLA and WSM-augmented finetuning.

This package trains the *action policy* (GR00T N1.7 / pi0.5) on RoboCasa365, with optional
WSM auxiliary heads. The WSM module/networks themselves live in the sibling ``workspace_models``
package; here we only consume them as an aux objective. Governed by README.md +
internal_planning_and_todos/04_wsm_roadmap.md.
"""
