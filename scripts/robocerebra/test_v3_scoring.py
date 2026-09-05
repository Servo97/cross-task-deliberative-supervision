"""Synthetic unit tests for the protocol-v3 scorer. Pure Python, no sim, no GPU.

Covers the three behaviours the v2 scorer got wrong or could not express:
  1. the FIRST completion inside a segment is credited to the agent (v2 discarded it);
  2. credit created by the re-pin is charged to RESUME, never to the agent;
  3. the dynamic-shift offset case (episode opens pinned one subtask ahead).

Run: python3 scripts/robocerebra/test_v3_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class FakeEnv:
    """Minimal stand-in exposing the two surfaces the scorer touches."""

    def __init__(self, progress: dict[str, int]) -> None:
        self._state_progress = dict(progress)
        self._total = 0

    def set_total(self, n: int) -> None:
        self._total = n

    def _check_success(self, goal):
        return {}, self._total, False


def v3_scorer(events, *, shift=False, prior_at_repin=None):
    """Re-run of the v3 bookkeeping in eval_robocerebra_openpi.py:run_episode.

    `events` is a list of ("step", total) or ("repin", seg, resume_gained).
    Returns (agent, resume_credited, resume_skipped, per_segment_agent).
    """
    agent = resume_credited = resume_skipped = 0
    skip_increment = bool(shift)
    if shift and prior_at_repin:
        resume_credited += prior_at_repin
    total_prev = 0
    seg_agent, segs = 0, []
    for ev in events:
        if ev[0] == "repin":
            segs.append(seg_agent)
            seg_agent = 0
            resume_credited += ev[2]
            skip_increment = True
            total_prev = ev[3]  # re-baseline after the pin
            continue
        total_now = ev[1]
        gained = total_now - total_prev
        if gained > 0:
            if skip_increment:
                resume_skipped += gained
            else:
                agent += gained
                seg_agent += gained
        total_prev = total_now
        if skip_increment:
            skip_increment = False
    segs.append(seg_agent)
    return agent, resume_credited, resume_skipped, segs


def v2_scorer(events, *, shift=False):
    """The sealed v2 rule: skip_increment is STICKY -- cleared only by a positive gain."""
    agent = 0
    skip = True  # step_states is not None
    total_prev = 0
    for ev in events:
        if ev[0] == "repin":
            skip = True
            total_prev = ev[3]
            continue
        gained = ev[1] - total_prev
        if gained > 0:
            if not skip:
                agent += gained
            skip = False
            total_prev = ev[1]
    return agent


FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")
    if not ok:
        FAILS.append(name)


print("1. first completion inside a segment is credited to the agent")
# Segment 1: nothing pinned, policy completes goal 1 at step 3.
ev = [("step", 0), ("step", 0), ("step", 1), ("step", 1)]
check("v3 credits it", v3_scorer(ev)[0], 1)
check("v2 DISCARDS it (the bug)", v2_scorer(ev), 0)

print("\n2. re-pin credit is charged to resume, not the agent")
# Re-pin into segment 2: the pinned state already satisfies goal 1 (resume_gained=1, baseline=1).
# Then the policy completes goal 2 at step 3 -- that one IS the agent's.
ev = [("step", 0), ("repin", 2, 1, 1), ("step", 1), ("step", 2), ("step", 2)]
a, rc, rs, segs = v3_scorer(ev)
check("agent gets only its own completion", a, 1)
check("resume credited the pinned goal", rc, 1)
check("nothing double-counted as skipped", rs, 0)
check("per-segment agent counts", segs, [0, 1])
check("v2 discards the agent's completion", v2_scorer(ev), 0)

print("\n2b. a goal that flips one step AFTER the pin is resume, not agent")
# The monitor's pointer lags the teleport by a step: without the one-step skip this lands on agent.
ev = [("repin", 2, 0, 0), ("step", 1), ("step", 1)]
a, rc, rs, _ = v3_scorer(ev)
check("charged to resume_skipped", rs, 1)
check("agent unaffected", a, 0)

print("\n3. dynamic-shift offset: episode opens pinned one subtask ahead")
# shift=True -> skip_increment starts True, and the prior subtask is resume-credited at init.
ev = [("step", 1), ("step", 2), ("step", 2)]
a, rc, rs, _ = v3_scorer(ev, shift=True, prior_at_repin=1)
check("init pin credited to resume", rc, 1)
check("first post-pin flip skipped", rs, 1)
check("the next genuine completion counts", a, 1)

print("\n4. regression: v3 >= v2 on every event stream")
import random

random.seed(0)
worse = 0
for _ in range(2000):
    tot, ev = 0, []
    for _ in range(random.randint(3, 25)):
        if random.random() < 0.15:
            ev.append(("repin", 0, random.randint(0, 1), tot))
        else:
            if random.random() < 0.25:
                tot += 1
            ev.append(("step", tot))
    if v3_scorer(ev)[0] < v2_scorer(ev):
        worse += 1
check("v3 never scores below v2 (2000 random streams)", worse, 0)

print("\n" + ("ALL v3 SCORER UNIT TESTS PASS" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
