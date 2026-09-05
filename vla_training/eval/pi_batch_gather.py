"""Thread-safe request gather for BATCHED pi0.5/openpi inference (Policy.infer_batch).

Collects concurrent `infer` requests for at most `max_wait_ms` (default 20ms) OR until a full bucket
(`max_batch`, default 8) is gathered — whichever comes first — then issues ONE `infer_batch(obs_list)`
call and routes each per-sample result dict back to its caller. Padding to a bucket K in {4, 8} happens
INSIDE Policy.infer_batch (openpi/policies/policy.py); `plan_buckets`/`bucket_size` here are the pure
reference twin of that math (stdlib-only, unit-testable without jax — tests/test_pi_batch_gather.py).

Usage in a serve script (drop-in around the built openpi policy):

    policy = build_wsm_cfg_policy(...)                    # openpi Policy (has .infer_batch)
    batched = BatchedPolicy(policy, max_wait_ms=20.0)     # BasePolicy facade; .infer is thread-safe
    WebsocketPolicyServer(policy=batched, ...).serve_forever()

CAVEAT (why this alone does not speed up today's server): websocket_policy_server._handler calls
`policy.infer(obs)` synchronously on the asyncio loop thread, so requests from different connections
never overlap — the gather would only ever see batches of 1 (+ up to max_wait_ms latency). To actually
batch across N eval clients the handler must offload the blocking call, e.g.
`await asyncio.get_running_loop().run_in_executor(None, self._policy.infer, obs)`. This module is the
gather side only; the server change is intentionally NOT made here (running evals depend on it).

Per-request kwargs (e.g. `noise=`) are not batchable and raise loudly. This module is stdlib-only.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

# Must mirror _BATCH_BUCKETS in robocasa_openpi/src/openpi/policies/policy.py (ascending).
BUCKETS: tuple[int, ...] = (4, 8)


def bucket_size(n: int, buckets: Sequence[int] = BUCKETS) -> int:
    """Smallest bucket >= n (the padded batch size K a chunk of n samples is stacked to)."""
    if n < 1:
        raise ValueError(f"bucket_size needs n >= 1, got {n}")
    for k in buckets:
        if k >= n:
            return int(k)
    raise ValueError(f"n={n} exceeds the largest bucket {max(buckets)}; chunk first (see plan_buckets)")


def plan_buckets(n: int, buckets: Sequence[int] = BUCKETS) -> list[tuple[int, int, int]]:
    """Chunk n requests exactly as Policy.infer_batch does: greedy chunks of the largest bucket, the tail
    padded to the smallest bucket that fits. Returns [(start, stop, k_pad), ...]."""
    max_bucket = int(max(buckets))
    plan = []
    for start in range(0, n, max_bucket):
        stop = min(start + max_bucket, n)
        plan.append((start, stop, bucket_size(stop - start, buckets)))
    return plan


class BatchGather:
    """Thread-safe gather queue: blocks each calling thread in `infer` until its result is ready, while a
    single worker thread coalesces concurrent requests into one `infer_batch` call per window/bucket."""

    _SENTINEL = object()

    def __init__(
        self,
        infer_batch: Callable[[list[dict]], list[dict]],
        *,
        max_batch: int = BUCKETS[-1],
        max_wait_ms: float = 20.0,
    ):
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        self._infer_batch = infer_batch
        self._max_batch = int(max_batch)
        self._max_wait_s = float(max_wait_ms) / 1000.0
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._run, name="pi-batch-gather", daemon=True)
        self._worker.start()

    def infer(self, obs: dict, **kwargs) -> dict:
        """Enqueue one request and block until its per-sample result is routed back. Thread-safe; call
        from any number of threads concurrently. Raises whatever `infer_batch` raised for its batch."""
        extra = {k: v for k, v in kwargs.items() if v is not None}
        if extra:
            raise NotImplementedError(
                f"[batch-gather] per-request kwargs are not batchable: {sorted(extra)} (drop them or "
                f"call the unbatched policy directly)"
            )
        if self._closed.is_set() or not self._worker.is_alive():
            raise RuntimeError("[batch-gather] gather is closed")
        fut: Future = Future()
        t_enqueue = time.monotonic()
        self._queue.put((obs, fut))
        result = fut.result()
        # Annotate observability alongside openpi's policy_timing (per-sample dicts, never shared).
        if isinstance(result, dict) and isinstance(result.get("policy_timing"), dict):
            result["policy_timing"]["gather_ms"] = (time.monotonic() - t_enqueue) * 1000
        return result

    def close(self, timeout_s: float = 5.0) -> None:
        """Stop the worker; pending/unrouted requests get a RuntimeError."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(self._SENTINEL)
        self._worker.join(timeout_s)
        self._drain_closed()  # catch any request that raced past the closed check

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._drain_closed()
                return
            batch = [item]
            deadline = time.monotonic() + self._max_wait_s
            stop = False
            while len(batch) < self._max_batch:  # gather for <= max_wait_ms OR until the bucket is full
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is self._SENTINEL:
                    stop = True
                    break
                batch.append(nxt)
            self._dispatch(batch)
            if stop:
                self._drain_closed()
                return

    def _dispatch(self, batch: list[tuple[dict, Future]]) -> None:
        obs_list = [obs for obs, _ in batch]
        futures = [fut for _, fut in batch]
        try:
            results = self._infer_batch(obs_list)
            if len(results) != len(obs_list):
                raise RuntimeError(
                    f"[batch-gather] infer_batch returned {len(results)} results for {len(obs_list)} "
                    f"requests — routing would misalign"
                )
        except BaseException as exc:  # noqa: BLE001 — deliver to the callers; keep the worker alive
            for fut in futures:
                fut.set_exception(exc)
            return
        for fut, result in zip(futures, results):
            if isinstance(result, dict) and isinstance(result.get("policy_timing"), dict):
                result["policy_timing"]["gather_batch_n"] = len(batch)
            fut.set_result(result)

    def _drain_closed(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not self._SENTINEL:
                item[1].set_exception(RuntimeError("[batch-gather] gather closed before dispatch"))


class BatchedPolicy:
    """openpi BasePolicy facade over BatchGather -> policy.infer_batch. `.infer` is thread-safe and
    batched; metadata proxies the wrapped policy (the websocket server sends it on connect)."""

    def __init__(self, policy: Any, *, max_batch: int = BUCKETS[-1], max_wait_ms: float = 20.0):
        self._policy = policy
        self._gather = BatchGather(policy.infer_batch, max_batch=max_batch, max_wait_ms=max_wait_ms)

    def infer(self, obs: dict, **kwargs) -> dict:
        return self._gather.infer(obs, **kwargs)

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def close(self) -> None:
        self._gather.close()
