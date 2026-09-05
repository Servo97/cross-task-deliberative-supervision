#!/usr/bin/env python3
"""Kernel-pinning replicate-pad around an openpi ``Policy``'s batched call.

THE PROBLEM THIS SOLVES
-----------------------
``openpi.policies.policy.Policy.infer_batch`` zero-pads each chunk to the smallest of
``_BATCH_BUCKETS = (4, 8)`` that fits. A gather window that happened to collect 3 requests therefore
runs a DIFFERENT XLA executable than one that collected 6 — mathematically the same row-independent
forward, but not necessarily the same bf16 rounding. If batch composition can move the last bits, it
can move an action, and a 900-step closed loop turns a last-bit action difference into a different
episode. Then "gather-batching is free" stops being provable.

The ω tap already solved exactly this problem for itself: ``--tap-pad-batch 16`` replicates row 0 up
to the store's build batch so the SAME kernel runs whether 1 or 16 frames are real (measured on this
campaign's tap: ``max|Δtok| = 0.000e+00`` between 1-real-padded-to-16 and 16-real). This module is
that trick applied to the policy call: every batched call is replicate-padded to ONE fixed row count,
so K, the gather window and scheduling cannot select a kernel.

Replication (not zero-padding) is deliberate: a zero row is a legal-but-weird observation whose
prompt tokenises to nothing, and openpi already appends zero rows underneath us when n < bucket.
Copying row 0 keeps every padded call inside the real input distribution and, because a transformer
batch is row-independent, cannot touch the real rows' outputs.

WHAT IS NOT CHANGED
-------------------
``infer`` is a straight delegation. The K=1-no-gather legacy path calls ``infer`` and is therefore
byte-identical to what it was before this file existed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def openpi_buckets() -> tuple[int, ...]:
    """``Policy._BATCH_BUCKETS``, read from openpi rather than duplicated here."""
    from openpi.policies import policy as _policy

    return tuple(_policy._BATCH_BUCKETS)  # noqa: SLF001 - single source of truth, on purpose


class FixedPadBatchPolicy:
    """Delegating facade that pins ``infer_batch`` to a fixed padded row count.

    ``pad_batch`` must be one of openpi's buckets, otherwise openpi would pad *again* underneath
    (a 6-row call still becomes an 8-row executable) and the "one fixed kernel" claim would be a
    claim about a number that is not the one that runs.

    ``pad_batch=0`` disables the pad entirely and restores stock openpi bucketing.

    ``mode="serial"`` is the BIT-IDENTITY escape hatch: a gathered batch is executed as N separate
    batch-1 ``infer`` calls, i.e. the same executable the unbatched legacy path runs. Combined with
    ``policy_noise_seed`` -- which removes the only order-dependence ``Policy.infer`` has -- each
    request becomes a pure function of its own observation, so K envs reproduce a K=1 run exactly.
    It gives up the batched-policy speedup, not the gather itself: whatever the caller collapsed
    UPSTREAM of the policy (for the ω arms, the tap, which is the dominant cost and is bit-exact
    when packed) is still collapsed. Serial mode REQUIRES seeds and says so rather than silently
    serving order-dependent noise.
    """

    def __init__(self, policy: Any, *, pad_batch: int = 8, mode: str = "batched"):
        self._policy = policy  # FIRST: __getattr__ delegates through it, so it must always exist
        pad = int(pad_batch)
        if mode not in ("batched", "serial"):
            raise ValueError(f"mode must be 'batched' or 'serial', got {mode!r}")
        if pad < 0:
            raise ValueError(f"pad_batch must be >= 0, got {pad_batch}")
        if pad and pad not in openpi_buckets():
            raise ValueError(
                f"pad_batch={pad} is not one of openpi's _BATCH_BUCKETS {openpi_buckets()}; openpi "
                f"would zero-pad it up to the next bucket and the pinned kernel would not be the "
                f"one this number names"
            )
        if not callable(getattr(policy, "infer_batch", None)):
            raise TypeError(f"{type(policy).__name__} has no infer_batch; nothing to pin")
        self._pad = pad
        self._mode = mode
        self.calls: list[int] = []  # realised REAL-row counts, for the throughput/parity tests

    # ---------------------------------------------------------------- unbatched: untouched
    def infer(self, obs: dict, **kwargs) -> dict:
        return self._policy.infer(obs, **kwargs)

    # ---------------------------------------------------------------- batched: pinned
    def infer_batch(self, obs_list: Sequence[dict], **kwargs) -> list[dict]:
        requests = list(obs_list)
        if not requests:
            return []
        if self._mode == "serial":
            unseeded = sum(1 for obs in requests if "policy_noise_seed" not in obs)
            if unseeded:
                raise RuntimeError(
                    f"[fixed-pad] mode='serial' needs policy_noise_seed on every request; {unseeded} "
                    f"of {len(requests)} had none. Without it Policy.infer draws action noise from a "
                    "mutable, arrival-ordered rng, so serving K envs is not reproducible at any batch "
                    "size. Run the harness with --deterministic-seeding."
                )
            self.calls.extend([1] * len(requests))
            return [self._policy.infer(obs, **kwargs) for obs in requests]
        if not self._pad:
            self.calls.append(len(requests))
            return self._policy.infer_batch(requests, **kwargs)

        results: list[dict] = []
        for start in range(0, len(requests), self._pad):
            chunk = requests[start : start + self._pad]
            n = len(chunk)
            self.calls.append(n)
            padded = chunk + [dict(chunk[0])] * (self._pad - n)
            out = self._policy.infer_batch(padded, **kwargs)
            if len(out) != len(padded):
                raise RuntimeError(f"[fixed-pad] infer_batch returned {len(out)} results for {len(padded)} rows")
            results.extend(out[:n])
        return results

    @property
    def pad_batch(self) -> int:
        return self._pad

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def metadata(self) -> dict:
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta["policy_pad_batch"] = self._pad
        meta["policy_batch_mode"] = self._mode
        return meta

    def __getattr__(self, name: str):
        # _model / _input_transform / ... — the ω tap reaches into the policy for those.
        return getattr(self._policy, name)


def gather_settings() -> tuple[int, int, float]:
    """``(k_envs, max_batch, max_wait_ms)`` for the websocket server's gather, from the environment.

    ``WSM_ENVS_PER_GPU`` is the switch the whole campaign already uses (openpi's
    ``WebsocketPolicyServer`` reads the same variable), so gather turns on for exactly the runs that
    launch K env runners and for nothing else.
    """
    import os

    k_envs = int(os.environ.get("WSM_ENVS_PER_GPU", "1"))
    if k_envs < 1:
        raise ValueError(f"WSM_ENVS_PER_GPU must be >= 1, got {k_envs}")
    max_batch = int(os.environ.get("WSM_GATHER_MAX_BATCH", "8"))
    if max_batch < 1:
        raise ValueError(f"WSM_GATHER_MAX_BATCH must be >= 1, got {max_batch}")
    wait_ms = float(os.environ.get("WSM_GATHER_WAIT_MS", "20"))
    if not wait_ms >= 0:
        raise ValueError(f"WSM_GATHER_WAIT_MS must be >= 0, got {wait_ms}")
    return k_envs, max_batch, wait_ms
