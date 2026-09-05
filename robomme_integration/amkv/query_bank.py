"""Disjoint action-query banks for training-free AM fits.

The anti-leak rule for H10/E0: an AM artifact is *fitted* with queries from one
policy chunk and *evaluated* with queries from a different chunk.  Fitting and
scoring on the same queries measures interpolation, not compression.

Disjointness is proven here on the actual arrays -- every query row is hashed
and the two banks must share none -- rather than asserted from metadata.  The
banks additionally carry the full route (fixture, chunk role, causal cut, flow
times, noise draw, layer set) so a reported number can never be unlabelled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

QUERY_BANK_SCHEMA_VERSION = 1
FIT_CHUNK = "fit_chunk"
EVAL_CHUNK = "eval_chunk"
CHUNK_ROLES = (FIT_CHUNK, EVAL_CHUNK)
QUERY_TAP_STAGE = "post_rope_pre_scale"
QUERY_HEAD_COUNT = 4


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _row_digests(queries: np.ndarray) -> set[bytes]:
    """One digest per (layer, head, sample) query vector."""

    flat = np.ascontiguousarray(queries).reshape(-1, queries.shape[-1])
    dtype = str(flat.dtype).encode()
    return {hashlib.sha256(dtype + row.tobytes()).digest() for row in flat}


@dataclasses.dataclass(frozen=True)
class ActionQueryBank:
    """Every action-expert memory query produced by one policy chunk.

    ``queries`` is ``[layers, 4 query heads, samples, head_dim]`` at serve
    precision, tapped post-RoPE and before the single ``head_dim**-0.5`` scale.
    ``samples`` enumerates ``flow_times x action_tokens`` in that order.
    """

    fixture_id: str
    chunk_role: str
    step_idx: int
    flow_times: tuple[float, ...]
    action_token_count: int
    noise_sha256: str
    queries: np.ndarray
    schema_version: int = QUERY_BANK_SCHEMA_VERSION
    query_tap_stage: str = QUERY_TAP_STAGE

    def __post_init__(self) -> None:
        self.validate()

    @property
    def layer_count(self) -> int:
        return int(self.queries.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.queries.shape[2])

    def validate(self) -> None:
        if self.schema_version != QUERY_BANK_SCHEMA_VERSION:
            raise ValueError(f"unsupported query-bank schema {self.schema_version}")
        if self.query_tap_stage != QUERY_TAP_STAGE:
            raise ValueError("queries must be tapped post-RoPE and before the single attention scale")
        if self.chunk_role not in CHUNK_ROLES:
            raise ValueError(f"chunk_role must be one of {CHUNK_ROLES}, got {self.chunk_role!r}")
        if not self.fixture_id:
            raise ValueError("fixture_id must be a nonempty identifier")
        if int(self.step_idx) < 0:
            raise ValueError("step_idx must be nonnegative")
        queries = np.asarray(self.queries)
        if queries.ndim != 4 or queries.shape[1] != QUERY_HEAD_COUNT:
            raise ValueError(f"queries must be [layers, 4, samples, head_dim], got {queries.shape}")
        if not queries.shape[0] or not queries.shape[2] or not queries.shape[3]:
            raise ValueError("query bank has an empty layer, sample, or feature axis")
        if not np.isfinite(queries).all():
            raise ValueError("query bank contains non-finite values")
        if not self.flow_times or len(set(self.flow_times)) != len(self.flow_times):
            raise ValueError("flow_times must be nonempty and unique")
        if any(not np.isfinite(value) for value in self.flow_times):
            raise ValueError("flow_times must be finite")
        if int(self.action_token_count) <= 0:
            raise ValueError("action_token_count must be positive")
        if queries.shape[2] != len(self.flow_times) * int(self.action_token_count):
            raise ValueError(
                "query sample count must equal flow_times x action_token_count: "
                f"{queries.shape[2]} != {len(self.flow_times)} x {self.action_token_count}"
            )
        if len(self.noise_sha256) != 64:
            raise ValueError("noise_sha256 must be a sha256 hex digest")

    def layer(self, layer_index: int) -> np.ndarray:
        """Return ``[4, samples, head_dim]`` for one layer."""

        if not 0 <= layer_index < self.layer_count:
            raise IndexError(f"layer {layer_index} outside [0, {self.layer_count})")
        return np.asarray(self.queries[layer_index])

    def bank_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.identity()).encode()).hexdigest()

    def identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fixture_id": self.fixture_id,
            "chunk_role": self.chunk_role,
            "step_idx": int(self.step_idx),
            "flow_times": [float(value) for value in self.flow_times],
            "action_token_count": int(self.action_token_count),
            "noise_sha256": self.noise_sha256,
            "query_tap_stage": self.query_tap_stage,
            "layer_count": self.layer_count,
            "sample_count": self.sample_count,
            "head_dim": int(self.queries.shape[-1]),
            "dtype": str(np.asarray(self.queries).dtype),
            "queries_sha256": array_sha256(self.queries),
        }

    def label(self) -> dict[str, object]:
        """Self-labelling record; an unlabelled AM number is discarded."""

        return {"bank_id": self.bank_id(), **self.identity()}


@dataclasses.dataclass(frozen=True)
class DisjointQueryBankPair:
    """A fit/eval bank pair with a proven-empty row intersection."""

    fit: ActionQueryBank
    heldout: ActionQueryBank
    shared_row_count: int
    fit_row_count: int
    heldout_row_count: int

    def validate(self) -> None:
        if self.shared_row_count:
            raise ValueError(f"query banks leak: {self.shared_row_count} identical query rows are in both splits")
        if self.fit.chunk_role != FIT_CHUNK or self.heldout.chunk_role != EVAL_CHUNK:
            raise ValueError("a disjoint pair must be (fit_chunk, eval_chunk)")
        if self.fit.step_idx == self.heldout.step_idx:
            raise ValueError("fit and held-out banks must come from different policy chunks")
        if self.fit.layer_count != self.heldout.layer_count:
            raise ValueError("fit and held-out banks describe different layer counts")
        if self.fit.queries.shape[-1] != self.heldout.queries.shape[-1]:
            raise ValueError("fit and held-out banks describe different head dimensions")

    def label(self) -> dict[str, object]:
        return {
            "fit": self.fit.label(),
            "heldout": self.heldout.label(),
            "shared_query_rows": self.shared_row_count,
            "fit_query_rows": self.fit_row_count,
            "heldout_query_rows": self.heldout_row_count,
            "disjointness_proof": "sha256_per_query_row_intersection_empty_v1",
        }


def pair_disjoint_banks(fit: ActionQueryBank, heldout: ActionQueryBank) -> DisjointQueryBankPair:
    """Prove, on the arrays, that no query row is shared between the splits."""

    fit.validate()
    heldout.validate()
    fit_rows = _row_digests(fit.queries)
    heldout_rows = _row_digests(heldout.queries)
    pair = DisjointQueryBankPair(
        fit=fit,
        heldout=heldout,
        shared_row_count=len(fit_rows & heldout_rows),
        fit_row_count=len(fit_rows),
        heldout_row_count=len(heldout_rows),
    )
    pair.validate()
    return pair


def bank_from_traced_queries(
    queries_by_step: np.ndarray,
    *,
    fixture_id: str,
    chunk_role: str,
    step_idx: int,
    flow_times: tuple[float, ...],
    noise_sha256: str,
) -> ActionQueryBank:
    """Fold a ``[flow_steps, layers, 4, tokens, head_dim]`` trace into one bank.

    Flow steps are concatenated along the sample axis in schedule order, so a
    bank spans the whole denoising trajectory of its chunk rather than a single
    flow time.  AM must survive every query the chunk actually issues.
    """

    array = np.asarray(queries_by_step)
    if array.ndim != 5:
        raise ValueError(f"traced queries must be [flow_steps, layers, 4, tokens, head_dim], got {array.shape}")
    steps, layers, heads, tokens, head_dim = array.shape
    if steps != len(flow_times):
        raise ValueError(f"trace has {steps} flow steps but {len(flow_times)} flow times")
    if heads != QUERY_HEAD_COUNT:
        raise ValueError("official MemoryAttention has exactly four action-query heads")
    # [steps, layers, 4, tokens, H] -> [layers, 4, steps*tokens, H]
    folded = np.transpose(array, (1, 2, 0, 3, 4)).reshape(layers, heads, steps * tokens, head_dim)
    return ActionQueryBank(
        fixture_id=fixture_id,
        chunk_role=chunk_role,
        step_idx=int(step_idx),
        flow_times=tuple(float(value) for value in flow_times),
        action_token_count=int(tokens),
        noise_sha256=noise_sha256,
        queries=np.ascontiguousarray(folded),
    )
