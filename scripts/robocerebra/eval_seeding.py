#!/usr/bin/env python3
"""Deterministic per-request seeds for the RoboCerebra eval — the thing that makes K>1 legal.

``openpi.policies.policy.Policy.infer`` advances a MUTABLE rng
(``self._rng, sample_rng = jax.random.split(self._rng)``). With one env per server that stream is a
function of the request index, which is itself a function of the protocol, so a rerun reproduces it.
Put K envs behind one server and the same stream becomes a function of *arrival order*: env 3's
action noise depends on how fast env 5's mujoco step happened to run. Results stop being
reproducible AND stop matching K=1.

openpi already ships the fix: an observation field ``policy_noise_seed`` (scalar uint32) routed
through ``_pop_policy_noise_seed`` / ``_noise_from_seed``, whose docstring is literally *"One
explicit JAX noise row, independent of mutable policy request order."* This module is the one place
that decides what goes into it.

    seed = blake2b("<mode>|<case>|<trial>|<step>")[:4]   as a big-endian uint32

The four fields are the complete PROTOCOL coordinate of a decision point and nothing else:

* NOT the arm — so every arm draws the same noise at the same protocol coordinate. That is common
  random numbers, and it strictly tightens the paired arm-vs-base contrast the campaign reports.
* NOT the env slot / shard / worker — so which runner owns a trial cannot change its outcome.
* NOT the arrival order — so gather composition, K, and scheduling cannot change its outcome.

``step`` is the post-wait env step (``wsm_t``), i.e. the same integer the ω server keys its grid on.
Request steps are protocol-determined (a request happens iff ``action_plan`` is empty, which happens
every ``--replan`` steps and at every subtask boundary), so the (mode, case, trial) -> {step} set is
identical across arms and across K.

``episode_rng_seed`` does the same job for the HARNESS-side rng that ``Random_Disturbance``/``Mix``
consume to pick a distractor: sharding trials across runners re-cuts a single global
``random.Random(seed)`` stream, so that stream has to become per-episode before trials may be split.

stdlib only: imported by the sim venv (harness), the openpi venv (servers) and the tests.
"""

from __future__ import annotations

import hashlib

UINT32_MAX = 0xFFFFFFFF


def _blake2b_uint32(payload: str) -> int:
    """Stable 32-bit digest of ``payload``. blake2b with digest_size=4, big-endian.

    Stable across processes and interpreter restarts (unlike ``hash()``, which is salted) and
    stable across machines (unlike anything derived from object identity).
    """
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def policy_noise_seed(mode: str, case: str, trial: int, step: int) -> int:
    """uint32 for ``obs['policy_noise_seed']`` at protocol coordinate (mode, case, trial, step)."""
    return _blake2b_uint32(f"{mode}|{case}|{int(trial)}|{int(step)}")


def episode_rng_seed(mode: str, case: str, trial: int, base_seed: int) -> int:
    """uint32 seed for the harness-side ``random.Random`` of ONE episode.

    Domain-separated from ``policy_noise_seed`` by the ``rng|`` prefix and by carrying the run's
    ``--seed``, so the two never collide and a run can still be re-rolled wholesale.
    """
    return _blake2b_uint32(f"rng|{int(base_seed)}|{mode}|{case}|{int(trial)}")


def check_uint32(value: int, what: str = "seed") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{what} must be a python int, got {value!r}")
    if not 0 <= value <= UINT32_MAX:
        raise ValueError(f"{what} outside uint32: {value}")
    return value


def _self_test() -> None:
    """No GPU, no venv, no imports beyond stdlib."""
    seeds = {}
    for mode in ("Ideal", "Mix"):
        for case in ("case1", "case10"):
            for trial in range(3):
                for step in (0, 5, 150, 895):
                    seed = policy_noise_seed(mode, case, trial, step)
                    check_uint32(seed, "policy_noise_seed")
                    seeds[(mode, case, trial, step)] = seed
    # determinism
    for key, want in seeds.items():
        assert policy_noise_seed(*key) == want, key
    # Spread over the whole 6-mode x 10-case x 10-trial x 270-step campaign coordinate space.
    # uint32 is openpi's contract, so collisions are a birthday fact, not a bug: n^2/2^33 pairs are
    # expected to collide, and a collision only means two UNRELATED (episode, step) pairs happen to
    # draw the same noise row — noise rows are i.i.d. draws anyway, so nothing downstream cares. The
    # bar is therefore "no worse than uniform", checked at ~5 sigma, not "zero".
    campaign = {
        policy_noise_seed(m, c, t, s)
        for m in (
            "Ideal",
            "Observation_Mismatching",
            "Random_Disturbance",
            "Mix",
            "Memory_Execution",
            "Memory_Exploration",
        )
        for c in [f"case{i}" for i in range(1, 11)]
        for t in range(10)
        for s in range(0, 1350, 5)
    }
    total = 6 * 10 * 10 * len(range(0, 1350, 5))
    collisions = total - len(campaign)
    expected = total * total / 2**33
    assert collisions <= expected + 5 * (expected**0.5) + 5, (
        f"{collisions} collisions in {total} draws, uniform expects ~{expected:.1f}"
    )
    # independence from anything not in the key
    assert policy_noise_seed("Ideal", "case1", 0, 5) != policy_noise_seed("Ideal", "case1", 1, 5)
    assert policy_noise_seed("Ideal", "case1", 0, 5) != policy_noise_seed("Ideal", "case2", 0, 5)
    assert episode_rng_seed("Ideal", "case1", 0, 7) != policy_noise_seed("Ideal", "case1", 0, 0)
    print(
        f"[eval_seeding] OK: {total} campaign coordinates -> {len(campaign)} distinct uint32 "
        f"seeds ({collisions} birthday collisions, uniform expects {expected:.1f}); "
        f"blake2b(mode|case|trial|step) is stable and carries no arm, no env slot, no arrival order"
    )


if __name__ == "__main__":
    _self_test()
