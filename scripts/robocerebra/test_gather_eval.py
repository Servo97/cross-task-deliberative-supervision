#!/usr/bin/env python3
"""Outcome-neutrality tests for gather-batched, K-envs-per-GPU RoboCerebra eval.

Four questions, in the order they can bite:

  A  per-env ω-window isolation      -- does WHO ELSE is in my batch change MY actions?
  B  cross-env episode-boundary      -- does env i's reset / re-pin perturb env j?
  C  determinism                     -- does (mode, case, trial) fix the outcome, at K=1 and K=8?
  D  throughput                      -- what did all of that buy?  (separate: needs the sim, see
                                        run_eval_sharded.sh and the D procedure in the report)

``--cpu`` runs A/B/C at the WRAPPER level with a deterministic ω source: no GPU, no checkpoint, no
jax, seconds. It proves the state plumbing -- which env's ω went into which request -- is invariant
to batch composition. It CANNOT prove XLA numerics, because there is no XLA.

``--gpu`` runs A/B/C again against the REAL A1 checkpoint, the REAL frozen tap and the REAL pinned ω
encoder, in-process (no sim, no websocket), and compares ACTION CHUNKS bitwise. That is the test
that can fail for a numerical reason, and it is the one that licenses the claim.

    # CPU, seconds
    python scripts/robocerebra/test_gather_eval.py --cpu

    # real weights: arm + frozen tap + ω encoder, ~25 GB measured on one 5090
    cd ~/Research/robocasa_openpi && CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=~/Research/TRI/wsmv2:~/Research/robocasa:~/Research/robosuite \\
      .venv/bin/python .../test_gather_eval.py --gpu \\
        --checkpoint /home/.../ckpts/a1_gdn_w8/15000 --config pi05_robocerebra_gdn_w8

    # what the policy call's answer actually depends on; and the bit-identity mode
    ... --kernel        ... --serial
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from eval_seeding import episode_rng_seed, policy_noise_seed  # noqa: E402

from wsm_settings import WSM_DATA_ROOT  # noqa: E402

OMEGA_DIM = 512
SWITCH = 150
REPLAN = 5
PROMPTS = [
    "pick up the plate",
    "open the microwave",
    "put the bowl in the sink",
    "close the drawer",
    "move the mug to the tray",
    "turn on the faucet",
    "place the book on the shelf",
    "pick up the ketchup",
]


# =================================================================================================
# fixtures
# =================================================================================================
def fixture_obs(
    env_id: str,
    t: int,
    episode_len: int,
    *,
    episode_id: str,
    seed: int,
    repin: bool = False,
    noise_seed: int | None = None,
) -> dict:
    """A request whose IMAGE CONTENT is a pure function of (episode_id, t).

    Deliberate: the ω a correct server computes for env X at step t must depend on X's own pixels and
    X's own causal prefix and on nothing else, so making the pixels a function of the episode
    coordinate is what lets the same coordinate be replayed in a different batch and compared.
    """
    rng = np.random.default_rng(abs(hash((episode_id, t, seed))) % (2**32))
    obs = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.normal(0, 0.2, 8).astype(np.float32),
        "prompt": PROMPTS[(t // SWITCH) % len(PROMPTS)],
        "wsm_env_id": env_id,
        "wsm_episode_id": episode_id,
        "wsm_t": int(t),
        "wsm_episode_len": int(episode_len),
        "wsm_repin": bool(repin),
    }
    if noise_seed is not None:
        obs["policy_noise_seed"] = np.uint32(noise_seed)
    return obs


def episode_requests(
    episode_id: str,
    env_id: str,
    *,
    segments: int,
    steps: int,
    seed: int,
    noise_key: tuple[str, str, int] | None = None,
) -> list[dict]:
    """The harness's request stream for one episode: every REPLAN steps, re-pin at each boundary."""
    episode_len = SWITCH * segments
    out = []
    for i in range(steps):
        t = i * REPLAN
        repin = t > 0 and t % SWITCH == 0
        noise = policy_noise_seed(*noise_key, t) if noise_key else None
        out.append(
            fixture_obs(env_id, t, episode_len, episode_id=episode_id, seed=seed, repin=repin, noise_seed=noise)
        )
    return out


class PureSource:
    """Deterministic ω source with per-env causal prefixes and NO shared mutable state.

    ω is a hash of (env_id, this env's prefix length, the request's own pixels). A wrapper that mixes
    envs up -- wrong window key, `due` evaluated after someone else's push, a re-pin clearing the
    wrong slot -- moves the prefix length or the pixels and the hash changes, loudly.
    """

    def __init__(self):
        self.prefix: dict[str, int] = {}
        self.batches: list[list[str]] = []  # one entry per encode_batch call: the envs in it
        self.calls: list[tuple[str, int]] = []
        self.resets: list[str] = []

    def reset(self, env_id: str) -> None:
        self.resets.append(env_id)
        self.prefix.pop(env_id, None)

    def encode(self, env_id: str, obs: dict) -> np.ndarray:
        return self.encode_batch([env_id], [obs])[0]

    def encode_batch(self, env_ids, obs_list) -> list[np.ndarray]:
        env_ids, obs_list = list(env_ids), list(obs_list)
        self.batches.append(list(env_ids))
        out = []
        for env_id, obs in zip(env_ids, obs_list):
            self.calls.append((env_id, int(np.asarray(obs["wsm_t"]).item())))
            depth = self.prefix.get(env_id, 0)
            self.prefix[env_id] = depth + 1
            pixel = int(np.asarray(obs["observation/image"]).astype(np.int64).sum())
            rng = np.random.default_rng((abs(hash((env_id, depth, pixel))) % (2**32)))
            out.append(rng.normal(0, 1, OMEGA_DIM).astype(np.float32))
        return out


class RecordingPolicy:
    """Fake policy: records the exact request it was handed and returns a function of it.

    The "action" is a hash of the ω window + the noise seed, i.e. of everything the real policy is
    allowed to depend on. Bitwise equality of these stand-in actions is exactly the claim the real
    GPU test then re-checks against XLA.
    """

    def __init__(self):
        self.seen: list[dict] = []
        self.batch_sizes: list[int] = []

    def _act(self, obs: dict) -> dict:
        self.seen.append({k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()})
        window = np.asarray(obs["wsm_w_window"], dtype=np.float64)
        seed = int(np.asarray(obs.get("policy_noise_seed", 0)).item())
        digest = np.random.default_rng((int(abs(window.sum() * 1e6)) ^ seed) % (2**32))
        return {"actions": digest.normal(0, 1, (10, 7)).astype(np.float32)}

    def infer(self, obs: dict, **_kw) -> dict:
        self.batch_sizes.append(1)
        return self._act(obs)

    def infer_batch(self, obs_list, **_kw) -> list[dict]:
        obs_list = list(obs_list)
        self.batch_sizes.append(len(obs_list))
        return [self._act(obs) for obs in obs_list]

    @property
    def metadata(self) -> dict:
        return {}


def build_wrapper(policy, source, *, window=8, max_envs=8):
    from serve_pi05_libero_wsm import OmegaPiInferWrapper

    return OmegaPiInferWrapper(policy, source, window=window, max_envs=max_envs, log_every=0)


# =================================================================================================
# A / B / C on CPU
# =================================================================================================
def _run_solo(requests: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray], PureSource]:
    policy, source = RecordingPolicy(), PureSource()
    wrapper = build_wrapper(policy, source)
    actions = [wrapper.infer_batch([obs])[0]["actions"] for obs in requests]
    windows = [np.array(seen["wsm_w_window"], copy=True) for seen in policy.seen]
    return actions, windows, source


def _run_cobatched(
    requests: list[dict],
    others: list[list[dict]],
) -> tuple[list[np.ndarray], list[np.ndarray], PureSource, list[int]]:
    """Same stream, but every request rides in a batch with one request from each other env."""
    policy, source = RecordingPolicy(), PureSource()
    wrapper = build_wrapper(policy, source)
    actions, windows = [], []
    for i, obs in enumerate(requests):
        batch = [obs] + [stream[i] for stream in others if i < len(stream)]
        results = wrapper.infer_batch(batch)
        actions.append(results[0]["actions"])
        windows.append(np.array(policy.seen[-len(batch)]["wsm_w_window"], copy=True))
    return actions, windows, source, policy.batch_sizes


def test_a_cpu() -> None:
    print("\n--- A (cpu): per-env ω-window isolation -------------------------------------------")
    target = episode_requests(
        "Ideal/case1/trial0", "env0", segments=6, steps=40, seed=1, noise_key=("Ideal", "case1", 0)
    )
    # Seven UNRELATED envs: different episode lengths, different phases, different pixels, and one
    # that re-pins on a different schedule -- i.e. every axis on which they could bleed into env0.
    others = [
        episode_requests(
            f"Mix/case{j}/trial{j}",
            f"env{j}",
            segments=4 + j % 5,
            steps=40,
            seed=100 + j,
            noise_key=("Mix", f"case{j}", j),
        )
        for j in range(1, 8)
    ]
    solo_actions, solo_windows, solo_source = _run_solo(target)
    co_actions, co_windows, co_source, batch_sizes = _run_cobatched(target, others)

    bad_window = [i for i, (a, b) in enumerate(zip(solo_windows, co_windows)) if not np.array_equal(a, b)]
    bad_action = [i for i, (a, b) in enumerate(zip(solo_actions, co_actions)) if not np.array_equal(a, b)]
    assert not bad_window, f"ω window differs at requests {bad_window[:5]}"
    assert not bad_action, f"action chunk differs at requests {bad_action[:5]}"
    assert solo_source.prefix["env0"] == co_source.prefix["env0"], (
        f"encoder prefix depth for env0: solo {solo_source.prefix['env0']} vs co-batched {co_source.prefix['env0']}"
    )
    tap_calls_solo = len(solo_source.batches)
    tap_calls_co = len(co_source.batches)
    rows_co = sum(len(b) for b in co_source.batches)
    print(
        f"[A] 40 requests, env0 alone vs env0 + 7 unrelated envs (K=8): "
        f"{len(solo_windows)}/{len(solo_windows)} ω windows bit-identical, "
        f"{len(solo_actions)}/{len(solo_actions)} action chunks bit-identical, "
        f"env0 encoder prefix depth {solo_source.prefix['env0']} both ways"
    )
    print(
        f"[A] gather: batch sizes {sorted(set(batch_sizes))} (mean "
        f"{np.mean(batch_sizes):.2f}); ω tap calls {tap_calls_solo} solo vs {tap_calls_co} for "
        f"{rows_co} due-env frames co-batched -> {rows_co / max(tap_calls_co, 1):.2f} frames/call"
    )


def test_b_cpu() -> None:
    print("\n--- B (cpu): cross-env episode-boundary / re-pin isolation ------------------------")
    victim = episode_requests(
        "Ideal/case2/trial3", "envA", segments=6, steps=40, seed=5, noise_key=("Ideal", "case2", 3)
    )
    solo_actions, solo_windows, solo_source = _run_solo(victim)

    # envB restarts a NEW episode every 7 requests (t=0 reset: new ServeWindow, encoder prefix
    # cleared, brand new episode identity) and re-pins in between. Every one of those events lands
    # in the same batch as one of envA's requests.
    disruptor: list[dict] = []
    for i in range(len(victim)):
        cycle = i % 7
        t = cycle * REPLAN
        disruptor.append(
            fixture_obs(
                "envB",
                t,
                SWITCH * 3,
                episode_id=f"Mix/case9/trial{i // 7}",
                seed=900 + i // 7,
                repin=(cycle == 3),
                noise_seed=policy_noise_seed("Mix", "case9", i // 7, t),
            )
        )

    policy, source = RecordingPolicy(), PureSource()
    wrapper = build_wrapper(policy, source)
    co_actions, co_windows = [], []
    for i, obs in enumerate(victim):
        results = wrapper.infer_batch([obs, disruptor[i]])
        co_actions.append(results[0]["actions"])
        co_windows.append(np.array(policy.seen[-2]["wsm_w_window"], copy=True))

    resets_b = source.resets.count("envB")
    assert source.resets.count("envA") == 1, (
        f"envA's encoder prefix was cleared {source.resets.count('envA')} times; expected exactly "
        "the one t=0 reset of its own episode"
    )
    bad = [i for i, (a, b) in enumerate(zip(solo_actions, co_actions)) if not np.array_equal(a, b)]
    assert not bad, f"envA perturbed by envB's boundaries at requests {bad[:5]}"
    assert all(np.array_equal(a, b) for a, b in zip(solo_windows, co_windows))
    assert solo_source.prefix["envA"] == source.prefix["envA"]
    print(
        f"[B] envB ran {resets_b} episode restarts + {len(victim) // 7} re-pins inside envA's "
        f"batches; envA: {len(co_actions)}/{len(co_actions)} action chunks and ω windows "
        f"bit-identical to solo, encoder prefix reset exactly 1x (its own t=0), depth "
        f"{source.prefix['envA']}"
    )


def test_c_cpu() -> None:
    print("\n--- C (cpu): determinism of (mode, case, trial) at K=1 and K=8 --------------------")
    key = ("Random_Disturbance", "case4", 2)
    target = episode_requests("Random_Disturbance/case4/trial2", "env0", segments=6, steps=30, seed=11, noise_key=key)
    others = [
        episode_requests(
            f"Ideal/case{j}/trial0", f"env{j}", segments=6, steps=30, seed=200 + j, noise_key=("Ideal", f"case{j}", 0)
        )
        for j in range(1, 8)
    ]

    runs = {
        "K=1 run1": _run_solo(target)[0],
        "K=1 run2": _run_solo(target)[0],
        "K=8 run1": _run_cobatched(target, others)[0],
        "K=8 run2": _run_cobatched(target, others)[0],
        # scheduling jitter: the same 8 envs, but env0 lands in a DIFFERENT batch position/size
        "K=8 shuffled": _run_cobatched(target, others[3:] + others[:3])[0],
        "K=3 partial": _run_cobatched(target, others[:2])[0],
    }
    reference = runs["K=1 run1"]
    for label, actions in runs.items():
        mismatched = [i for i, (a, b) in enumerate(zip(reference, actions)) if not np.array_equal(a, b)]
        assert not mismatched, f"{label} differs from K=1 run1 at {mismatched[:5]}"
    # and the harness-side rng that Random_Disturbance consumes
    seeds = {episode_rng_seed(*key[:2], key[2], 7) for _ in range(3)}
    assert len(seeds) == 1
    print(
        f"[C] {len(reference)} decision steps x {len(runs)} configurations "
        f"({', '.join(runs)}): all bit-identical to K=1 run1"
    )
    print(
        f"[C] episode rng seed for {key} is {seeds.pop()} on every call "
        "(function of the coordinate, not of iteration order)"
    )


def test_failure_isolation() -> None:
    """One malformed request must not kill the K-1 healthy episodes sharing its gather window."""
    print("\n--- failure isolation: one bad request in a window of 8 -------------------------")
    from serve_pi05_libero_wsm import BatchValidationError

    policy, source = RecordingPolicy(), PureSource()
    wrapper = build_wrapper(policy, source, max_envs=8)
    good = [
        episode_requests(
            f"Ideal/case{j}/trial0", f"env{j}", segments=6, steps=12, seed=j, noise_key=("Ideal", f"case{j}", 0)
        )
        for j in range(8)
    ]
    for i in range(2):  # everyone healthy and mid-episode
        wrapper.infer_batch([stream[i] for stream in good])

    def fault_cases(step: int) -> dict:
        """The four ways a real runner goes wrong, all at request index ``step``."""
        return {
            "unknown env id (restarted shard)": fixture_obs(
                "env99", step * REPLAN, 900, episode_id="Ideal/case9/trial0", seed=9, noise_seed=1
            ),
            "step went backwards": fixture_obs("env3", 5, 900, episode_id="Ideal/case3/trial0", seed=3, noise_seed=1),
            "episode identity changed": fixture_obs(
                "env3", step * REPLAN, 900, episode_id="SOMETHING/ELSE", seed=3, noise_seed=1
            ),
            "missing identity field": {k: v for k, v in good[3][step].items() if k != "wsm_t"},
        }

    for step, (label, bad) in enumerate(fault_cases(0).items(), start=2):
        bad = fault_cases(step)[label]
        bad_env = str(bad.get("wsm_env_id", ""))
        healthy = [stream[step] for stream in good if stream[step]["wsm_env_id"] != bad_env]
        results = wrapper.infer_batch([*healthy, bad])
        assert len(results) == len(healthy) + 1
        assert isinstance(results[-1], BaseException), f"{label}: the bad request was SERVED"
        served = sum(1 for r in results[:-1] if isinstance(r, dict))
        assert served == len(healthy), f"{label}: only {served}/{len(healthy)} healthy envs served"
        print(
            f"    {label:<34} -> {served}/{len(healthy)} healthy envs served, bad request raised "
            f"{type(results[-1]).__name__}"
        )

    # two runners sharing one env id must BOTH fail: that is a launch bug, and serving them
    # independently would interleave one ω window between two episodes.
    step = 6
    dup = wrapper.infer_batch(
        [
            good[0][step],
            good[1][step],
            fixture_obs("env0", 999, 900, episode_id="Ideal/case0/trial0", seed=0, noise_seed=1),
        ]
    )
    assert isinstance(dup[0], BatchValidationError), type(dup[0])
    assert isinstance(dup[2], BatchValidationError), type(dup[2])
    assert isinstance(dup[1], dict), "the unrelated third env should still have been served"
    print(
        f"    {'duplicate env id (two runners, 1 slot)':<34} -> both copies rejected, the unrelated env still served"
    )
    # and the survivors are still live: they keep stepping afterwards
    after = wrapper.infer_batch([stream[7] for stream in good if stream[7]["wsm_env_id"] != "env0"])
    assert all(isinstance(r, dict) for r in after), "healthy envs did not survive the bad windows"
    print(f"    survivors keep running: {len(after)}/{len(after)} served on the next window")


def test_fixed_pad() -> None:
    print("\n--- kernel pinning: FixedPadBatchPolicy -------------------------------------------")

    class Counter:
        def __init__(self):
            self.rows: list[int] = []

        def infer(self, obs, **_kw):
            return {"actions": np.zeros((10, 7), np.float32), "tag": obs["tag"]}

        def infer_batch(self, obs_list, **_kw):
            obs_list = list(obs_list)
            self.rows.append(len(obs_list))
            return [{"actions": np.zeros((10, 7), np.float32), "tag": o["tag"]} for o in obs_list]

    from serve_batching import FixedPadBatchPolicy, openpi_buckets

    inner = Counter()
    padded = FixedPadBatchPolicy(inner, pad_batch=8)
    for n in (1, 2, 3, 5, 8):
        out = padded.infer_batch([{"tag": i} for i in range(n)])
        assert [o["tag"] for o in out] == list(range(n)), f"routing broke at n={n}"
    assert inner.rows == [8] * 5, f"padded row counts {inner.rows}, expected five 8-row calls"
    # oversized input must be CHUNKED at the pad, never sent as a third row count
    inner.rows.clear()
    out = padded.infer_batch([{"tag": i} for i in range(13)])
    assert [o["tag"] for o in out] == list(range(13))
    assert inner.rows == [8, 8], f"13 rows -> {inner.rows}, expected [8, 8]"
    # a pad that is not an openpi bucket must be refused, not silently re-padded by openpi
    try:
        FixedPadBatchPolicy(Counter(), pad_batch=6)
    except ValueError as error:
        assert "_BATCH_BUCKETS" in str(error)
    else:
        raise AssertionError("pad_batch=6 must be refused")
    print(
        f"[pad] openpi buckets {openpi_buckets()}; n in (1,2,3,5,8,13) -> inner row counts "
        f"all 8, routing preserved, non-bucket pad refused"
    )


def test_merge() -> None:
    print("\n--- shard merge ------------------------------------------------------------------")
    import tempfile

    from merge_eval_shards import merge

    base_prov = {
        "arm": "A1_gdn_w8",
        "ckpt_sha": "abc",
        "encoder_sha": "09a1",
        "budget_steps": 15000,
        "note": "",
        "modes": ["Ideal"],
        "cases": ["case1"],
        "trials": 8,
        "trial_start": 0,
        "replan": 5,
        "switch_steps": 150,
        "seed": 7,
        "random_actions": False,
        "wsm": True,
        "num_shards": 4,
        "deterministic_seeding": True,
    }
    truth = {t: (t % 3 == 0, t + 1) for t in range(8)}
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for shard in range(4):
            rows = [
                {
                    "mode": "Ideal",
                    "case": "case1",
                    "trial": t,
                    "success": truth[t][0],
                    "agent_subtasks": truth[t][1],
                    "possible_subtasks": 9,
                    "num_subtasks": 6,
                    "bddl": "x.bddl",
                    "shard": shard,
                }
                for t in range(shard, 8, 4)
            ]
            path = pathlib.Path(tmp) / f"r.shard{shard}.json"
            path.write_text(
                json.dumps(
                    {"provenance": {**base_prov, "shard": shard}, "per_case": [], "per_trial": rows, "complete": True}
                )
            )
            paths.append(path)
        merged = merge(paths)
        row = merged["per_case"][0]
        assert row["trials"] == 8 and row["successes"] == sum(v[0] for v in truth.values())
        assert row["agent_subtasks"] == sum(v[1] for v in truth.values())
        assert abs(row["success_rate"] - row["successes"] / 8) < 1e-12
        assert merged["complete"]

        # a shard that disagrees about an already-merged trial must be refused
        bad = pathlib.Path(tmp) / "r.shard9.json"
        rows = [
            {
                "mode": "Ideal",
                "case": "case1",
                "trial": 0,
                "success": not truth[0][0],
                "agent_subtasks": 99,
                "possible_subtasks": 9,
                "num_subtasks": 6,
                "bddl": "x.bddl",
                "shard": 9,
            }
        ]
        bad.write_text(
            json.dumps({"provenance": {**base_prov, "shard": 1}, "per_case": [], "per_trial": rows, "complete": True})
        )
        try:
            merge([paths[0], bad], allow_partial=True)
        except SystemExit as error:
            assert "DIFFERENT outcome" in str(error), error
        else:
            raise AssertionError("conflicting shards must be refused")
    print(
        f"[merge] 4 shards x 2 trials -> 8 unioned trials, rates recomputed "
        f"(success {row['successes']}/8 = {row['success_rate']:.3f}, subtask "
        f"{row['agent_subtasks']}/{row['possible_subtasks']}); conflicting shard refused"
    )


# =================================================================================================
# A / B / C with REAL weights
# =================================================================================================
def build_real(args):
    """The exact serve-time stack, in process: arm policy + frozen tap + pinned ω encoder."""
    import serve_pi05_libero_wsm as srv

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    t0 = time.monotonic()
    policy, train_config = srv.build_arm_policy(checkpoint, args.config, None)
    window = srv.resolve_window(train_config, policy, None)
    tap = srv.Pi05Tap.from_checkpoint(args.tap_checkpoint, "pi05_libero", None, pad_batch=args.tap_pad_batch)
    encoder, cfg, sha = srv.load_omega_encoder(srv.PINNED_ENCODER, "cuda", srv.PINNED_ENCODER_SHA256)
    source = srv.Pi05OmegaSource(tap, encoder, cfg, "cuda")
    print(
        f"[gpu] built arm={args.config} K={window} tap=pi05_libero(pad {args.tap_pad_batch}) "
        f"encoder={sha[:12]} in {time.monotonic() - t0:.0f}s"
    )
    return policy, source, window, tap


def _gpu_stream(wrapper, requests, batch_builder=None):
    chunks = []
    for i, obs in enumerate(requests):
        batch = [obs] if batch_builder is None else batch_builder(i, obs)
        chunks.append(np.array(wrapper.infer_batch(batch)[0]["actions"], copy=True))
    return chunks


def _report_diff(label: str, a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    deltas = [float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) for x, y in zip(a, b)]
    identical = sum(1 for x, y in zip(a, b) if np.array_equal(x, y))
    print(f"    {label}: {identical}/{len(a)} chunks BIT-IDENTICAL, max|Δaction| = {max(deltas):.3e}")
    return identical == len(a)


def test_gpu(args) -> None:
    import serve_pi05_libero_wsm as srv
    from serve_batching import FixedPadBatchPolicy

    policy, source, window, tap = build_real(args)
    padded = FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch)

    steps = args.gpu_steps
    target = episode_requests(
        "Ideal/case1/trial0", "env0", segments=6, steps=steps, seed=1, noise_key=("Ideal", "case1", 0)
    )
    others = [
        episode_requests(
            f"Mix/case{j}/trial{j}",
            f"env{j}",
            segments=4 + j % 5,
            steps=steps,
            seed=100 + j,
            noise_key=("Mix", f"case{j}", j),
        )
        for j in range(1, 8)
    ]

    def fresh():
        source._prefix.clear()
        tap.calls = 0
        tap.real_rows = 0
        return srv.OmegaPiInferWrapper(padded, source, window=window, max_envs=8, log_every=0)

    # ---- A: solo vs co-batched with 7 unrelated envs -------------------------------------------
    print("\n--- A (gpu, real weights): per-env ω-window isolation -----------------------------")
    t0 = time.monotonic()
    solo = _gpu_stream(fresh(), target)
    solo_s, solo_tap = time.monotonic() - t0, tap.calls
    t0 = time.monotonic()
    co = _gpu_stream(fresh(), target, lambda i, obs: [obs] + [s[i] for s in others])
    co_s, co_tap, co_rows = time.monotonic() - t0, tap.calls, tap.real_rows
    ok_a = _report_diff("env0 alone vs env0 + 7 unrelated envs", solo, co)
    print(
        f"    solo {steps} requests in {solo_s:.1f}s ({solo_tap} tap calls); "
        f"co-batched 8x{steps} requests in {co_s:.1f}s ({co_tap} tap calls carrying "
        f"{co_rows} frames = {co_rows / max(co_tap, 1):.2f} frames/call)"
    )
    assert ok_a, "A FAILED: batch composition changed env0's actions"

    # ---- B: env boundary in a neighbour ---------------------------------------------------------
    print("\n--- B (gpu, real weights): cross-env episode-boundary isolation -------------------")
    disruptor = []
    for i in range(steps):
        cycle = i % 5
        t = cycle * REPLAN
        disruptor.append(
            fixture_obs(
                "envB",
                t,
                SWITCH * 3,
                episode_id=f"Mix/case9/trial{i // 5}",
                seed=900 + i // 5,
                repin=(cycle == 2),
                noise_seed=policy_noise_seed("Mix", "case9", i // 5, t),
            )
        )
    t0 = time.monotonic()
    co_b = _gpu_stream(fresh(), target, lambda i, obs: [obs, disruptor[i]])
    ok_b = _report_diff("env0 alone vs env0 + an env restarting/re-pinning beside it", solo, co_b)
    print(
        f"    neighbour performed {steps // 5} episode restarts and {steps // 5} re-pins "
        f"inside env0's batches, in {time.monotonic() - t0:.1f}s"
    )
    assert ok_b, "B FAILED: a neighbour's episode boundary perturbed env0"

    # ---- C: determinism -------------------------------------------------------------------------
    print("\n--- C (gpu, real weights): determinism at K=1 and K=8 -----------------------------")
    solo2 = _gpu_stream(fresh(), target)
    ok_c1 = _report_diff("K=1 run1 vs K=1 run2", solo, solo2)
    co2 = _gpu_stream(fresh(), target, lambda i, obs: [obs] + [s[i] for s in others])
    ok_c8 = _report_diff("K=8 run1 vs K=8 run2", co, co2)
    shuffled = _gpu_stream(fresh(), target, lambda i, obs: [obs] + [s[i] for s in others[3:] + others[:3]])
    ok_cs = _report_diff("K=8 vs K=8 with the other envs reordered", co, shuffled)
    partial = _gpu_stream(fresh(), target, lambda i, obs: [obs] + [s[i] for s in others[:2]])
    ok_cp = _report_diff("K=1 vs K=3 (partial gather window)", solo, partial)
    assert ok_c1 and ok_c8 and ok_cs and ok_cp, "C FAILED: outcome is not a function of the coordinate"

    # ---- the number the legacy path differs by, measured rather than assumed --------------------
    print("\n--- legacy reference: unbatched Policy.infer (batch 1, no bucket pad) -------------")
    legacy_wrapper = fresh()
    legacy = [np.array(legacy_wrapper.infer(dict(obs))["actions"], copy=True) for obs in target]
    _report_diff("legacy Policy.infer vs pinned-8 batched path", legacy, solo)
    print("    The legacy path runs a batch-1 executable and the batched path a batch-8 one, so they")
    print("    are NOT bit-equal and cannot be: see --kernel for the isolated measurement. Legacy is")
    print("    preserved unchanged behind WSM_ENVS_PER_GPU=1. The claim A/B/C establish is the one")
    print("    that licenses K>1: within the batched path the outcome is invariant to K, to batch")
    print("    composition and to ordering, bitwise. Cells scored on the two paths are not")
    print("    interchangeable -- score every arm on one path.")


def test_kernel(args) -> None:
    """Isolate WHAT the policy call's answer depends on: row count, or also composition/order?

    No ω wrapper and no gather -- one fixed observation, one fixed ω window, one fixed noise seed,
    pushed through Policy.infer and Policy.infer_batch at several row counts. This is the
    measurement that says whether ``--policy-pad-batch`` is load-bearing or decorative.
    """
    import serve_pi05_libero_wsm as srv

    policy, train_config = srv.build_arm_policy(
        pathlib.Path(args.checkpoint).expanduser().resolve(), args.config, None
    )
    window = srv.resolve_window(train_config, policy, None)
    rng = np.random.default_rng(0)

    def request(seed: int, prompt: str) -> dict:
        r = np.random.default_rng(seed)
        return {
            "observation/image": r.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": r.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": r.normal(0, 0.2, 8).astype(np.float32),
            "prompt": prompt,
            "wsm_w_window": r.normal(0, 1, (window, OMEGA_DIM)).astype(np.float32),
            "policy_noise_seed": np.uint32(seed),
        }

    base = request(123456789, "pick up the plate")
    others = [request(900 + j, "open the microwave") for j in range(7)]
    del rng

    runs = {
        "infer (batch 1) #1": np.asarray(policy.infer(dict(base))["actions"]),
        "infer (batch 1) #2": np.asarray(policy.infer(dict(base))["actions"]),
        "infer_batch n=1 -> bucket 4": np.asarray(policy.infer_batch([dict(base)])[0]["actions"]),
        "infer_batch n=4 -> bucket 4": np.asarray(policy.infer_batch([dict(base)] * 4)[0]["actions"]),
        "infer_batch n=5 -> bucket 8": np.asarray(policy.infer_batch([dict(base)] * 5)[0]["actions"]),
        "infer_batch n=8 -> bucket 8": np.asarray(policy.infer_batch([dict(base)] * 8)[0]["actions"]),
        "infer_batch n=8, 7 OTHER envs": np.asarray(
            policy.infer_batch([dict(base)] + [dict(o) for o in others])[0]["actions"]
        ),
        "infer_batch n=8, those reordered": np.asarray(
            policy.infer_batch([dict(base)] + [dict(o) for o in others[3:] + others[:3]])[0]["actions"]
        ),
    }
    reference = runs["infer (batch 1) #1"]
    scale = float(np.abs(reference).max())
    print(f"\n--- kernel sensitivity: action chunk {reference.shape}, |a|max = {scale:.4f} --------")
    print(f"{'variant':<34} {'bit-eq vs infer':<16} {'max|Δ|':>10} {'rel':>10}")
    for label, actions in runs.items():
        delta = np.abs(actions.astype(np.float64) - reference.astype(np.float64))
        print(
            f"{label:<34} {str(np.array_equal(actions, reference)):<16} {delta.max():>10.3e} "
            f"{delta.max() / max(scale, 1e-9):>10.3e}"
        )
    pairs = [
        ("infer_batch n=1 -> bucket 4", "infer_batch n=4 -> bucket 4"),
        ("infer_batch n=5 -> bucket 8", "infer_batch n=8 -> bucket 8"),
        ("infer_batch n=8 -> bucket 8", "infer_batch n=8, 7 OTHER envs"),
        ("infer_batch n=8, 7 OTHER envs", "infer_batch n=8, those reordered"),
        ("infer_batch n=4 -> bucket 4", "infer_batch n=8 -> bucket 8"),
    ]
    print("\n  pairwise:")
    for left, right in pairs:
        print(f"    {left:<32} vs {right:<32} bit-equal = {np.array_equal(runs[left], runs[right])}")
    print("\n  Reading: everything at the SAME row count is bit-equal regardless of who else was in")
    print("  the batch or in what order -- composition and ordering are provably irrelevant. Row")
    print("  COUNT is not: bucket 4 and bucket 8 differ, which is exactly why --policy-pad-batch")
    print("  pins one row count for the whole run.")


def test_serial_gpu(args) -> None:
    """--policy-batch serial: gather the TAP, keep the policy at batch 1, get bit-identity.

    The claim: a gathered K=8 window served in serial mode returns exactly what eight separate
    unbatched K=1 requests would. That is the mode for anyone who needs the new path's numbers to be
    comparable with numbers already scored on the unbatched path.
    """
    import serve_pi05_libero_wsm as srv
    from serve_batching import FixedPadBatchPolicy

    policy, source, window, tap = build_real(args)
    steps = max(4, args.gpu_steps // 2)
    target = episode_requests(
        "Ideal/case1/trial0", "env0", segments=6, steps=steps, seed=1, noise_key=("Ideal", "case1", 0)
    )
    others = [
        episode_requests(
            f"Mix/case{j}/trial{j}",
            f"env{j}",
            segments=4 + j % 5,
            steps=steps,
            seed=100 + j,
            noise_key=("Mix", f"case{j}", j),
        )
        for j in range(1, 8)
    ]

    def run(mode, batch_builder=None):
        source._prefix.clear()
        tap.calls = 0
        wrapped = FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch, mode=mode)
        wrapper = srv.OmegaPiInferWrapper(wrapped, source, window=window, max_envs=8, log_every=0)
        return _gpu_stream(wrapper, target, batch_builder), tap.calls

    print("\n--- serial mode: K=8 gather, batch-1 policy ---------------------------------------")
    source._prefix.clear()
    tap.calls = 0
    legacy_wrapper = srv.OmegaPiInferWrapper(
        FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch, mode="serial"),
        source,
        window=window,
        max_envs=8,
        log_every=0,
    )
    legacy = [np.array(legacy_wrapper.infer(dict(obs))["actions"], copy=True) for obs in target]
    legacy_tap = tap.calls

    serial_k8, serial_tap = run("serial", lambda i, obs: [obs] + [s[i] for s in others])
    batched_k8, batched_tap = run("batched", lambda i, obs: [obs] + [s[i] for s in others])
    ok = _report_diff("K=1 unbatched Policy.infer vs K=8 gather in SERIAL mode", legacy, serial_k8)
    _report_diff("K=1 unbatched Policy.infer vs K=8 gather in BATCHED mode", legacy, batched_k8)
    print(
        f"    tap calls: {legacy_tap} at K=1, {serial_tap} for 8x the frames in serial mode "
        f"(the gather still collapses the tap), {batched_tap} in batched mode"
    )
    assert ok, "serial mode is supposed to be bit-identical to the unbatched path"


def test_noise_seed_gpu(args) -> None:
    """policy_noise_seed really is what drives the action, and the unseeded path really is not."""
    import serve_pi05_libero_wsm as srv
    from serve_batching import FixedPadBatchPolicy

    policy, source, window, tap = build_real(args)
    padded = FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch)
    print("\n--- policy_noise_seed: seeded vs unseeded -----------------------------------------")

    def stream(noise_key, tag):
        source._prefix.clear()
        wrapper = srv.OmegaPiInferWrapper(padded, source, window=window, max_envs=8, log_every=0)
        reqs = episode_requests("Ideal/case1/trial0", tag, segments=6, steps=4, seed=1, noise_key=noise_key)
        return [np.array(wrapper.infer_batch([o])[0]["actions"], copy=True) for o in reqs]

    a = stream(("Ideal", "case1", 0), "env0")
    b = stream(("Ideal", "case1", 0), "env0")
    c = stream(("Ideal", "case1", 1), "env0")  # different TRIAL -> different noise
    d = stream(None, "env0")
    e = stream(None, "env0")
    same_ab = all(np.array_equal(x, y) for x, y in zip(a, b))
    same_ac = all(np.array_equal(x, y) for x, y in zip(a, c))
    same_de = all(np.array_equal(x, y) for x, y in zip(d, e))
    print(f"    seeded, same (mode,case,trial) twice : identical = {same_ab}")
    print(f"    seeded, trial 0 vs trial 1           : identical = {same_ac} (must be False)")
    print(f"    UNSEEDED, same stream twice          : identical = {same_de} (mutable server rng)")
    assert same_ab and not same_ac and not same_de


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="A/B/C + merge + pad, no GPU")
    parser.add_argument("--gpu", action="store_true", help="A/B/C against real weights")
    parser.add_argument("--noise-seed", action="store_true", help="seeded-vs-unseeded GPU check")
    parser.add_argument(
        "--serial", action="store_true", help="verify --policy-batch serial is bit-identical to the unbatched path"
    )
    parser.add_argument(
        "--kernel", action="store_true", help="isolate what the policy call depends on: row count vs composition"
    )
    parser.add_argument("--checkpoint", default=str(WSM_DATA_ROOT / "robocerebra" / "ckpts" / "a1_gdn_w8" / "15000"))
    parser.add_argument("--config", default="pi05_robocerebra_gdn_w8")
    parser.add_argument(
        "--tap-checkpoint",
        default=str(WSM_DATA_ROOT / "robocerebra/openpi_assets/openpi-assets/checkpoints/pi05_libero"),
    )
    parser.add_argument("--tap-pad-batch", type=int, default=16)
    parser.add_argument("--policy-pad-batch", type=int, default=8)
    parser.add_argument(
        "--gpu-steps", type=int, default=12, help="decision steps per fixture episode in the GPU tests"
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING, force=True)

    if args.cpu or not (args.gpu or args.noise_seed or args.kernel or args.serial):
        print("=" * 96)
        print("gather-batched RoboCerebra eval: CPU fixture tests")
        print("=" * 96)
        test_a_cpu()
        test_b_cpu()
        test_c_cpu()
        test_failure_isolation()
        test_fixed_pad()
        test_merge()
        print("\nCPU: ALL PASS")
    if args.gpu:
        print("=" * 96)
        print("gather-batched RoboCerebra eval: REAL-WEIGHTS tests")
        print("=" * 96)
        test_gpu(args)
        print("\nGPU: ALL PASS")
    if args.noise_seed:
        test_noise_seed_gpu(args)
        print("\nNOISE-SEED: PASS")
    if args.serial:
        test_serial_gpu(args)
        print("\nSERIAL: PASS")
    if args.kernel:
        test_kernel(args)


if __name__ == "__main__":
    main()
