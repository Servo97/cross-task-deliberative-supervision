"""Unit tests for the pi batched-inference gather (vla_training/eval/pi_batch_gather.py).

Pure stdlib + numpy — NO jax/GPU: Policy.infer_batch is replaced by FakeInferBatch, which reproduces the
policy's bucket/pad math (via plan_buckets, the reference twin of policy.py's _BATCH_BUCKETS chunking) on
numpy arrays and records every padded batch it saw. Proves: (1) correct result routing under 8 concurrent
threads, (2) bucketing + zero-padding math, (3) the <=20ms-or-full-bucket latency window.

Run:  PYTHONPATH=. python3 tests/test_pi_batch_gather.py     (also pytest-compatible)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_training.eval.pi_batch_gather import (  # noqa: E402
    BUCKETS,
    BatchedPolicy,
    BatchGather,
    bucket_size,
    plan_buckets,
)

STATE_DIM = 3


class FakeInferBatch:
    """Stand-in for Policy.infer_batch: chunk+pad per plan_buckets (K in {4,8}, zero rows), 'run the
    model' (actions = 2*state), unpad, per-sample result dicts. Records (n_real, k_pad) per chunk and the
    padded state batches so tests can assert the padding math."""

    def __init__(self, delay_s: float = 0.0, fail: Exception | None = None):
        self.delay_s = delay_s
        self.fail = fail
        self.calls: list[tuple[int, int]] = []  # (n_real, k_pad) per chunk
        self.padded_states: list[np.ndarray] = []  # the (k_pad, STATE_DIM) stacked+padded batches
        self._lock = threading.Lock()

    def __call__(self, obs_list: list[dict]) -> list[dict]:
        if self.fail is not None:
            raise self.fail
        results = []
        for start, stop, k_pad in plan_buckets(len(obs_list)):
            chunk = obs_list[start:stop]
            rows = [np.asarray(o["state"], dtype=np.float32) for o in chunk]
            states = np.stack(rows + [np.zeros_like(rows[0])] * (k_pad - len(rows)))  # zero-pad rows
            with self._lock:
                self.calls.append((len(chunk), k_pad))
                self.padded_states.append(states)
            if self.delay_s:
                time.sleep(self.delay_s)
            actions = states * 2.0
            for i, obs in enumerate(chunk):  # unpad: real rows only
                results.append({"id": obs["id"], "actions": actions[i], "policy_timing": {"infer_ms": 0.0}})
        return results


def _obs(i: int) -> dict:
    return {"id": i, "state": np.full(STATE_DIM, float(i + 1), dtype=np.float32)}


def test_bucket_math():
    assert BUCKETS == (4, 8)
    assert [bucket_size(n) for n in (1, 2, 3, 4)] == [4, 4, 4, 4]
    assert [bucket_size(n) for n in (5, 6, 7, 8)] == [8, 8, 8, 8]
    for bad in (0, -1, 9):
        try:
            bucket_size(bad)
            raise AssertionError(f"bucket_size({bad}) should raise")
        except ValueError:
            pass
    assert plan_buckets(0) == []
    assert plan_buckets(1) == [(0, 1, 4)]
    assert plan_buckets(4) == [(0, 4, 4)]
    assert plan_buckets(5) == [(0, 5, 8)]
    assert plan_buckets(8) == [(0, 8, 8)]
    assert plan_buckets(9) == [(0, 8, 8), (8, 9, 4)]
    assert plan_buckets(20) == [(0, 8, 8), (8, 16, 8), (16, 20, 4)]
    # chunk sizes cover [0, n) exactly and every pad size is a bucket >= the chunk
    for n in range(1, 40):
        plan = plan_buckets(n)
        assert plan[0][0] == 0 and plan[-1][1] == n
        for (s0, s1, k), nxt in zip(plan, plan[1:] + [None]):
            assert k in BUCKETS and k >= s1 - s0
            if nxt is not None:
                assert nxt[0] == s1


def test_padding_math():
    for n in (1, 2, 4, 5, 8, 9, 20):
        fake = FakeInferBatch()
        results = fake([_obs(i) for i in range(n)])
        # routing/order preserved through chunk+pad+unpad
        assert [r["id"] for r in results] == list(range(n))
        for i, r in enumerate(results):
            np.testing.assert_array_equal(r["actions"], np.full(STATE_DIM, 2.0 * (i + 1), dtype=np.float32))
        # padded batches: bucket shapes match the plan; pad rows are zero; real rows intact
        assert fake.calls == [(s1 - s0, k) for s0, s1, k in plan_buckets(n)]
        i = 0
        for (n_real, k_pad), states in zip(fake.calls, fake.padded_states):
            assert states.shape == (k_pad, STATE_DIM) and k_pad in BUCKETS
            for row in range(n_real):
                np.testing.assert_array_equal(states[row], np.full(STATE_DIM, float(i + 1), dtype=np.float32))
                i += 1
            assert not states[n_real:].any(), "pad rows must be zero"


def test_routing_under_8_concurrent_threads():
    fake = FakeInferBatch(delay_s=0.03)
    gather = BatchGather(fake, max_batch=8, max_wait_ms=200.0)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results: dict[int, dict] = {}
    errors: list[BaseException] = []

    def worker(i: int):
        try:
            barrier.wait()
            results[i] = gather.infer(_obs(i))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    gather.close()

    assert not errors, errors
    assert len(results) == n_threads
    for i, r in enumerate(results[i] for i in range(n_threads)):
        assert r["id"] == i, f"thread {i} got result for id {r['id']} — MISROUTED"
        np.testing.assert_array_equal(r["actions"], np.full(STATE_DIM, 2.0 * (i + 1), dtype=np.float32))
        assert r["policy_timing"]["gather_batch_n"] >= 1
        assert "gather_ms" in r["policy_timing"]
    # every request served exactly once, no chunk over the bucket cap, and real coalescing happened
    assert sum(n for n, _ in fake.calls) == n_threads
    assert all(n <= 8 and k in BUCKETS for n, k in fake.calls)
    assert max(n for n, _ in fake.calls) >= 2, f"no batching occurred: {fake.calls}"


def test_full_bucket_dispatches_before_window():
    # window is 10s: only the bucket-full trigger can return this fast
    fake = FakeInferBatch()
    gather = BatchGather(fake, max_batch=8, max_wait_ms=10_000.0)
    barrier = threading.Barrier(8)
    done: list[dict] = []

    def worker(i: int):
        barrier.wait()
        done.append(gather.infer(_obs(i)))

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=8.0)
    elapsed = time.monotonic() - t0
    gather.close()
    assert len(done) == 8
    assert elapsed < 5.0, f"full bucket did not dispatch early (took {elapsed:.2f}s of a 10s window)"
    assert fake.calls == [(8, 8)], fake.calls


def test_lone_request_respects_latency_window():
    fake = FakeInferBatch()
    gather = BatchGather(fake, max_batch=8, max_wait_ms=60.0)
    t0 = time.monotonic()
    result = gather.infer(_obs(0))
    elapsed = time.monotonic() - t0
    gather.close()
    assert result["id"] == 0
    # gathered for the window (>= ~60ms) but no longer: never held hostage waiting for a full bucket
    assert 0.04 <= elapsed <= 1.0, f"lone request took {elapsed * 1000:.1f}ms (window 60ms)"
    assert fake.calls == [(1, 4)], fake.calls


def test_error_propagates_to_every_caller():
    fake = FakeInferBatch(fail=ValueError("boom"))
    gather = BatchGather(fake, max_batch=8, max_wait_ms=100.0)
    barrier = threading.Barrier(3)
    caught: list[BaseException] = []

    def worker(i: int):
        barrier.wait()
        try:
            gather.infer(_obs(i))
        except ValueError as e:
            caught.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    gather.close()
    assert len(caught) == 3 and all(str(e) == "boom" for e in caught)


def test_kwargs_and_close_fail_loud():
    gather = BatchGather(FakeInferBatch(), max_wait_ms=10.0)
    try:
        gather.infer(_obs(0), noise=np.zeros(3))
        raise AssertionError("per-request kwargs must raise")
    except NotImplementedError:
        pass
    assert gather.infer(_obs(1), noise=None)["id"] == 1  # None kwargs are fine (drop-in .infer signature)
    gather.close()
    try:
        gather.infer(_obs(2))
        raise AssertionError("infer after close must raise")
    except RuntimeError:
        pass


def test_batched_policy_facade():
    class _FakePolicy:
        metadata = {"who": "fake"}

        def __init__(self):
            self.infer_batch = FakeInferBatch()

    policy = _FakePolicy()
    batched = BatchedPolicy(policy, max_wait_ms=10.0)
    out = batched.infer(_obs(4))
    assert out["id"] == 4
    assert batched.metadata == {"who": "fake"}
    batched.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
