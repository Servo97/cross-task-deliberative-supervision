"""Unit tests for the BATCHED ROUTER policy server (vla_training/eval/serve_groot_batched.py). NO GPU:
a fake batch-capable policy (identity-ish echo on the batch dim) + fake per-identity conditioners, real
zmq REQ clients over tcp://127.0.0.1:<ephemeral>. Covers: replies routed to the right client, batch
sizes >1 under load, ping answered immediately (never held for the gather window), max-batch early
flush, per-request malformed-input isolation, and wsm_cfg per-identity reset isolation.

Run:  PYTHONPATH=. ~/Research/envs/robocasa_env/bin/python tests/test_serve_groot_batched.py
  or: .../python -m pytest tests/test_serve_groot_batched.py -q
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_training.eval.serve_groot_batched import (  # noqa: E402
    BatchedPolicyServer,
    WSMCfgIdentityStates,
    _from_bytes,
    _to_bytes,
)


# ---- fakes ---------------------------------------------------------------------------------------
class FakePolicy:
    """Batch-capable stand-in for Gr00tSimPolicyWrapper.get_action: echoes state.id per row (so routing
    is checkable), asserts the language list stacked to len B, records batch sizes, sleeps to force
    queuing. reset just logs (baseline reset path)."""

    def __init__(self, sleep_s: float = 0.0):
        self.sleep_s = sleep_s
        self.batch_sizes: list[int] = []
        self.reset_calls: list = []

    def get_action(self, observation, options=None):
        b = observation["state.id"].shape[0]
        assert observation["video.cam"].shape == (b, 1, 4, 4, 3), observation["video.cam"].shape
        assert len(observation["lang"]) == b, (len(observation["lang"]), b)
        self.batch_sizes.append(b)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        action = {"action.echo": observation["state.id"].astype(np.float64)}  # (B,1,D)
        return action, {"bsz": np.full((b,), b, dtype=np.int64)}

    def reset(self, options=None):
        self.reset_calls.append(options)
        return {"status": "ok"}


class FakeConditioner:
    """WSMEvalConditioner API subset: reset(lang) clears the buffer; step returns
    w_window [K=1, 2] = [[len(buffer_after_step), patch[0]]] so per-row pairing is checkable."""

    def __init__(self):
        self._lang = None
        self.buf: list[float] = []

    def reset(self, lang_vec):
        self._lang = np.asarray(lang_vec, dtype=np.float32)
        self.buf = []

    def step(self, patch, proprio):
        v = float(np.asarray(patch).ravel()[0])
        self.buf.append(v)
        return np.asarray([[float(len(self.buf)), v]], dtype=np.float32), self._lang


class FakeCfgPolicy(FakePolicy):
    """Mirrors install_wsm_cfg_batched's seam: per batched forward, step each pending identity's
    conditioner with its row (state.id stands in for the tapped patch) and return the w_t rows."""

    def __init__(self, states: WSMCfgIdentityStates, **kw):
        super().__init__(**kw)
        self.states = states

    def get_action(self, observation, options=None):
        b = observation["state.id"].shape[0]
        rows = self.states.conditioner_rows(
            [observation["state.id"][i].ravel() for i in range(b)],
            [observation["state.id"][i].ravel() for i in range(b)],
        )
        action, info = super().get_action(observation, options)
        action["action.wt"] = np.stack([np.asarray(r, dtype=np.float64) for r in rows])[:, None, :]  # (B,1,2)
        return action, info


# ---- zmq harness ---------------------------------------------------------------------------------
class MiniClient:
    """eval_runner_groot.ZmqPolicyClient subset with an explicit zmq identity (per-identity asserts)."""

    def __init__(self, ctx, port, identity: bytes, timeout_ms: int = 10_000):
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.IDENTITY, identity)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://127.0.0.1:{port}")

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(_to_bytes(req))
        return _from_bytes(self.sock.recv())

    def close(self):
        self.sock.close(linger=0)


@contextmanager
def _server(policy, **kw):
    srv = BatchedPolicyServer(policy, host="127.0.0.1", port=0, **kw)  # port=0 -> ephemeral
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    ctx = zmq.Context()
    try:
        yield srv, ctx
    finally:
        killer = MiniClient(ctx, srv.port, b"_killer", timeout_ms=3_000)
        try:
            killer.call("kill")
        except Exception:
            pass
        killer.close()
        th.join(timeout=5)
        assert not th.is_alive(), "server thread did not exit after kill"
        srv.close()
        ctx.term()


def _obs(val: float, d: int = 3):
    return {
        "video.cam": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
        "state.id": np.full((1, 1, d), val, dtype=np.float32),
        "lang": ["do-it"],
    }


def _run_threads(workers):
    errs: list[str] = []

    def _wrap(fn):
        try:
            fn()
        except Exception:
            errs.append(traceback.format_exc())

    ts = [threading.Thread(target=_wrap, args=(fn,)) for fn in workers]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not errs, "client thread failures:\n" + "\n".join(errs)


# ---- tests ---------------------------------------------------------------------------------------
def test_interleaved_routing_and_batching():
    """4 REQ clients interleave ping/reset/get_action; every reply routes to its sender, and gathered
    batches >1 hit the ONE batched forward."""
    policy = FakePolicy(sleep_s=0.02)
    with _server(policy, gather_ms=60, max_batch=8) as (srv, ctx):
        barrier = threading.Barrier(4)

        def worker(cid):
            def run():
                cli = MiniClient(ctx, srv.port, f"cli{cid}".encode())
                try:
                    assert cli.call("ping")["status"] == "ok"
                    assert cli.call("reset", {"options": {"task": f"t{cid}"}})["status"] == "ok"
                    barrier.wait()
                    for j in range(5):
                        val = cid * 100 + j
                        resp = cli.call("get_action", {"observation": _obs(val), "options": None})
                        assert isinstance(resp, (list, tuple)) and len(resp) == 2, resp  # (action, info) wire
                        action, info = resp
                        echo = np.asarray(action["action.echo"])
                        assert echo.shape == (1, 1, 3), echo.shape  # row keeps batch dim
                        assert np.allclose(echo, val), f"cid={cid} j={j}: routed {echo.ravel()[0]} != {val}"
                        assert np.asarray(info["bsz"]).shape == (1,)
                finally:
                    cli.close()

            return run

        _run_threads([worker(c) for c in range(4)])
        assert sum(srv.batch_sizes) == 20, srv.batch_sizes  # every request served exactly once
        assert max(srv.batch_sizes) >= 2, f"no batching under load: {srv.batch_sizes}"
        assert len(policy.reset_calls) == 4  # baseline reset -> policy.reset
        assert {c["task"] for c in policy.reset_calls} == {f"t{c}" for c in range(4)}
    print(f"  PASS test_interleaved_routing_and_batching (batch sizes {srv.batch_sizes})")


def test_ping_immediate_while_gathering():
    """ping is answered inside the gather window, NOT queued behind the batch flush."""
    policy = FakePolicy()
    with _server(policy, gather_ms=400, max_batch=8) as (srv, ctx):
        barrier = threading.Barrier(3)
        elapsed: dict[int, float] = {}

        def worker(cid):
            def run():
                cli = MiniClient(ctx, srv.port, f"g{cid}".encode())
                try:
                    barrier.wait()
                    t0 = time.monotonic()
                    resp = cli.call("get_action", {"observation": _obs(cid), "options": None})
                    elapsed[cid] = time.monotonic() - t0
                    assert np.allclose(np.asarray(resp[0]["action.echo"]), cid)
                finally:
                    cli.close()

            return run

        ts = [threading.Thread(target=worker(c)) for c in (1, 2)]
        for t in ts:
            t.start()
        barrier.wait()
        time.sleep(0.1)  # both get_actions are now gathering
        pinger = MiniClient(ctx, srv.port, b"pinger")
        t0 = time.monotonic()
        assert pinger.call("ping")["status"] == "ok"
        rtt = time.monotonic() - t0
        pinger.close()
        for t in ts:
            t.join(timeout=10)
        assert rtt < 0.2, f"ping waited for the batch window: rtt={rtt:.3f}s"
        assert all(e >= 0.3 for e in elapsed.values()), f"batch flushed early? {elapsed}"  # window honored
        assert srv.batch_sizes == [2], srv.batch_sizes
    print(f"  PASS test_ping_immediate_while_gathering (ping rtt {rtt * 1e3:.1f}ms, batch {srv.batch_sizes})")


def test_max_batch_flushes_early():
    """the batch flushes the moment it reaches --max-batch, not at gather_ms."""
    policy = FakePolicy()
    with _server(policy, gather_ms=5000, max_batch=2) as (srv, ctx):
        barrier = threading.Barrier(2)
        elapsed: dict[int, float] = {}

        def worker(cid):
            def run():
                cli = MiniClient(ctx, srv.port, f"m{cid}".encode())
                try:
                    barrier.wait()
                    t0 = time.monotonic()
                    resp = cli.call("get_action", {"observation": _obs(cid), "options": None})
                    elapsed[cid] = time.monotonic() - t0
                    assert np.allclose(np.asarray(resp[0]["action.echo"]), cid)
                finally:
                    cli.close()

            return run

        _run_threads([worker(c) for c in (1, 2)])
        assert srv.batch_sizes == [2], srv.batch_sizes
        assert all(e < 2.0 for e in elapsed.values()), f"waited for gather_ms=5000: {elapsed}"
    print(f"  PASS test_max_batch_flushes_early (elapsed {max(elapsed.values()):.3f}s)")


def test_per_request_malformed_isolated():
    """a malformed / batch-incompatible request errors THAT identity only; the rest of the batch runs."""
    policy = FakePolicy()
    with _server(policy, gather_ms=150, max_batch=8) as (srv, ctx):
        results: dict[str, object] = {}

        def sender(name, obs, barrier):
            def run():
                cli = MiniClient(ctx, srv.port, name.encode())
                try:
                    barrier.wait()  # same gathered batch
                    results[name] = cli.call("get_action", {"observation": obs, "options": None})
                finally:
                    cli.close()

            return run

        # wave 1: good + bad leading batch dim (2 != 1)
        bad = _obs(5.0)
        bad["state.id"] = np.zeros((2, 1, 3), dtype=np.float32)
        b1 = threading.Barrier(2)
        _run_threads([sender("ok1", _obs(7.0), b1), sender("bad1", bad, b1)])
        assert np.allclose(np.asarray(results["ok1"][0]["action.echo"]), 7.0)
        assert isinstance(results["bad1"], dict) and "leading batch dim" in results["bad1"]["error"]

        # wave 2: two individually-valid obs whose per-key shapes mismatch -> exactly one errors
        b2 = threading.Barrier(2)
        _run_threads([sender("okA", _obs(1.0, d=3), b2), sender("okB", _obs(2.0, d=5), b2)])
        replies = [results["okA"], results["okB"]]
        errors = [r for r in replies if isinstance(r, dict) and "error" in r]
        goods = [r for r in replies if isinstance(r, (list, tuple))]
        assert len(errors) == 1 and "mismatch" in errors[0]["error"], replies
        assert len(goods) == 1
        assert sum(srv.batch_sizes) == 2, srv.batch_sizes  # only the good rows reach the forward
    print(f"  PASS test_per_request_malformed_isolated (batch sizes {srv.batch_sizes})")


def test_wsm_cfg_identity_isolation():
    """wsm_cfg hooks: each zmq identity owns its conditioner; reset clears ONLY that identity; get_action
    before reset errors that request; batched rows pair with the right identity's conditioner."""
    table = {"t1": [1.0], "t2": [2.0]}
    states = WSMCfgIdentityStates(FakeConditioner, table)
    policy = FakeCfgPolicy(states)
    with _server(
        policy,
        gather_ms=100,
        max_batch=8,
        reset_hook=states.reset,
        validate_hook=states.validate,
        batch_hook=states.on_batch,
    ) as (srv, ctx):
        A = MiniClient(ctx, srv.port, b"A")
        B = MiniClient(ctx, srv.port, b"B")
        C = MiniClient(ctx, srv.port, b"C")
        try:
            assert A.call("reset", {"options": {"task": "t1"}})["status"] == "ok"
            assert B.call("reset", {"options": {"task": "t2"}})["status"] == "ok"

            def wt(cli, val):
                resp = cli.call("get_action", {"observation": _obs(val), "options": None})
                assert isinstance(resp, (list, tuple)), resp
                row = np.asarray(resp[0]["action.wt"])
                assert row.shape == (1, 1, 2), row.shape
                assert row[0, 0, 1] == val, f"row paired with wrong identity: {row} vs val {val}"
                return row[0, 0, 0]  # per-identity step count

            # simultaneous wave: rows must pair per identity inside one gathered batch
            counts: dict[str, float] = {}
            barrier = threading.Barrier(2)

            def wave(name, cli, val):
                def run():
                    barrier.wait()
                    counts[name] = wt(cli, val)

                return run

            _run_threads([wave("A", A, 7.0), wave("B", B, 9.0)])
            assert counts == {"A": 1.0, "B": 1.0}, counts

            assert wt(A, 8.0) == 2.0  # A's causal buffer grows
            assert A.call("reset", {"options": {"task": "t1"}})["status"] == "ok"
            assert wt(A, 5.0) == 1.0, "reset did not clear A's buffer"
            assert wt(B, 4.0) == 2.0, "A's reset leaked into B's buffer"  # the isolation claim

            r = C.call("get_action", {"observation": _obs(0.0), "options": None})
            assert isinstance(r, dict) and "before reset" in r["error"], r
            r = A.call("reset", {"options": {"task": "nope"}})
            assert isinstance(r, dict) and "known task" in r["error"], r

            assert set(states._conds) == {b"A", b"B"}, set(states._conds)  # keyed by zmq identity
            assert max(srv.batch_sizes) >= 1 and sum(srv.batch_sizes) == 5, srv.batch_sizes
        finally:
            A.close()
            B.close()
            C.close()
    print(f"  PASS test_wsm_cfg_identity_isolation (batch sizes {srv.batch_sizes})")


if __name__ == "__main__":
    test_interleaved_routing_and_batching()
    test_ping_immediate_while_gathering()
    test_max_batch_flushes_early()
    test_per_request_malformed_isolated()
    test_wsm_cfg_identity_isolation()
    print("ALL BATCHED-SERVER TESTS PASS")
