"""Verbatim port of the RoboCerebra authors' completion bookkeeping.

Transcribed from ``code/evaluation/episode.py`` (``update_completion_tracking``,
``handle_segment_transition``) and ``code/evaluation/resume.py``
(``simulate_resume_completion``), keeping their control flow and ordering rather than our inlined
form. Used ONLY as a shadow oracle under ``--shadow-authors-scorer``: it runs alongside the live
v3 counters on the same rollout, and the two must agree exactly on every trial.

The point is to catch an ordering or off-by-one error in the inlined v3 scorer that a
re-implementation sharing v3's structure would reproduce rather than expose. This file is
structured like theirs; eval_robocerebra_openpi.py is structured like ours.
"""

from __future__ import annotations


class AuthorsScorer:
    """Their ``episode_stats`` dict and the three functions that mutate it."""

    def __init__(self, env, goal, resume_handler, *, dynamic_shift_description: bool) -> None:
        self._env, self._goal, self._resume_handler = env, goal, resume_handler
        self.total_agent_subtasks = 0
        self.total_resume_skipped = 0
        self.total_resume_completed = 0
        self.skip_increment = False
        self.total_completed_prev = 0
        if dynamic_shift_description:
            # episode.py:129-137 -- opens already pinned one subtask ahead.
            self.skip_increment = True

    def record_resume_completion(self, gained: int) -> None:
        """Observe the count that the live scorer's ``simulate_resume_completion`` returned.

        ``simulate_resume_completion`` MUTATES ``env._state_progress``. Two scorers sharing one env
        must not both call it -- the second would always find the pointer already advanced and
        report 0. The shadow therefore records the live count and keeps its own tally, so the two
        totals stay independently derived without double-mutating the env.
        """
        self.total_resume_completed += int(gained)

    # resume.py:56 -- kept for the standalone unit test, which owns its own env stub.
    def simulate_resume_completion(self, current_step: int) -> int:
        rh = self._resume_handler
        if not rh or current_step not in rh["step_to_prior_subtasks"]:
            return 0
        completed_by_resume = []
        env = self._env
        if hasattr(env, "_state_progress"):
            for subtask in rh["step_to_prior_subtasks"][current_step]:
                obj_id = subtask["object"]
                action_index = subtask["action_index"]
                if obj_id in env._state_progress:
                    if env._state_progress[obj_id] <= action_index:
                        env._state_progress[obj_id] = action_index + 1
                        completed_by_resume.append(f"{obj_id}_{action_index}")
        self.total_resume_completed += len(completed_by_resume)
        return len(completed_by_resume)

    def baseline(self) -> None:
        """episode.py:196-198 -- the completion baseline taken after the warm-up no-ops."""
        _, self.total_completed_prev, _ = self._env._check_success(self._goal)

    def on_transition(self) -> None:
        """episode.py:365-379 -- resume re-pin sets skip_increment and re-baselines."""
        self.skip_increment = True
        _, self.total_completed_prev, _ = self._env._check_success(self._goal)

    # episode.py:408-436 + the unconditional clear at episode.py:310-312
    def update_completion_tracking(self) -> int:
        _, total_completed_now, _ = self._env._check_success(self._goal)
        diff = total_completed_now - self.total_completed_prev
        seg_diff = 0
        if diff > 0:
            if self.skip_increment:
                self.total_resume_skipped += diff
            else:
                self.total_agent_subtasks += diff
                seg_diff = diff
        self.total_completed_prev = total_completed_now
        # The caller clears the flag AFTER accounting, once per loop iteration.
        if self.skip_increment:
            self.skip_increment = False
        return seg_diff
