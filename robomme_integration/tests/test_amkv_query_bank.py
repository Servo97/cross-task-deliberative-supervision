"""Query banks must be labelled and provably disjoint."""

from __future__ import annotations

import numpy as np
import pytest

from robomme_integration.amkv import query_bank

FLOW_TIMES = (1.0, 0.9, 0.8)
TOKENS = 4
LAYERS = 3
HEAD_DIM = 8


def _trace(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(len(FLOW_TIMES), LAYERS, query_bank.QUERY_HEAD_COUNT, TOKENS, HEAD_DIM)).astype(
        np.float32
    )


def _bank(seed: int, *, role: str, step: int) -> query_bank.ActionQueryBank:
    return query_bank.bank_from_traced_queries(
        _trace(seed),
        fixture_id=f"ep000001-t{step:05d}",
        chunk_role=role,
        step_idx=step,
        flow_times=FLOW_TIMES,
        noise_sha256="a" * 64,
    )


def test_folding_preserves_every_query_and_its_order():
    trace = _trace(0)
    bank = _bank(0, role=query_bank.FIT_CHUNK, step=47)
    assert bank.queries.shape == (LAYERS, query_bank.QUERY_HEAD_COUNT, len(FLOW_TIMES) * TOKENS, HEAD_DIM)
    assert bank.sample_count == len(FLOW_TIMES) * TOKENS
    # flow step s, token k lands at sample s*TOKENS + k
    for step in range(len(FLOW_TIMES)):
        for token in range(TOKENS):
            assert np.array_equal(bank.queries[:, :, step * TOKENS + token, :], trace[step, :, :, token, :])


def test_bank_identity_is_deterministic_and_content_addressed():
    first = _bank(0, role=query_bank.FIT_CHUNK, step=47)
    same = _bank(0, role=query_bank.FIT_CHUNK, step=47)
    other = _bank(1, role=query_bank.FIT_CHUNK, step=47)
    assert first.bank_id() == same.bank_id()
    assert first.bank_id() != other.bank_id()
    assert first.label()["queries_sha256"] == same.identity()["queries_sha256"]


def test_disjoint_pair_accepts_banks_from_different_chunks():
    pair = query_bank.pair_disjoint_banks(
        _bank(0, role=query_bank.FIT_CHUNK, step=47), _bank(1, role=query_bank.EVAL_CHUNK, step=63)
    )
    assert pair.shared_row_count == 0
    label = pair.label()
    assert label["fit"]["chunk_role"] == query_bank.FIT_CHUNK
    assert label["heldout"]["step_idx"] == 63
    assert label["disjointness_proof"]


def test_identical_banks_are_rejected_as_a_leak():
    with pytest.raises(ValueError, match="leak"):
        query_bank.pair_disjoint_banks(
            _bank(0, role=query_bank.FIT_CHUNK, step=47), _bank(0, role=query_bank.EVAL_CHUNK, step=63)
        )


def test_a_single_shared_query_row_is_caught():
    fit = _bank(0, role=query_bank.FIT_CHUNK, step=47)
    leaked = _trace(1)
    leaked[0, 0, 0, 0, :] = fit.queries[0, 0, 0, :]
    heldout = query_bank.bank_from_traced_queries(
        leaked,
        fixture_id="ep000001-t00063",
        chunk_role=query_bank.EVAL_CHUNK,
        step_idx=63,
        flow_times=FLOW_TIMES,
        noise_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="leak"):
        query_bank.pair_disjoint_banks(fit, heldout)


def test_same_chunk_pairs_are_rejected_even_when_rows_differ():
    with pytest.raises(ValueError, match="different policy chunks"):
        query_bank.pair_disjoint_banks(
            _bank(0, role=query_bank.FIT_CHUNK, step=47), _bank(1, role=query_bank.EVAL_CHUNK, step=47)
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"chunk_role": "other"}, "chunk_role"),
        ({"flow_times": (1.0, 1.0, 0.8)}, "unique"),
        ({"noise_sha256": "abc"}, "sha256"),
        ({"fixture_id": ""}, "fixture_id"),
    ],
)
def test_bank_validation_rejects_unlabelled_or_inconsistent_banks(mutation, match):
    fields = {
        "fixture_id": "ep000001-t00047",
        "chunk_role": query_bank.FIT_CHUNK,
        "step_idx": 47,
        "flow_times": FLOW_TIMES,
        "action_token_count": TOKENS,
        "noise_sha256": "a" * 64,
        "queries": np.zeros((LAYERS, query_bank.QUERY_HEAD_COUNT, len(FLOW_TIMES) * TOKENS, HEAD_DIM), np.float32),
    }
    fields.update(mutation)
    with pytest.raises(ValueError, match=match):
        query_bank.ActionQueryBank(**fields)


def test_non_finite_queries_are_rejected():
    trace = _trace(0)
    trace[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        query_bank.bank_from_traced_queries(
            trace,
            fixture_id="ep000001-t00047",
            chunk_role=query_bank.FIT_CHUNK,
            step_idx=47,
            flow_times=FLOW_TIMES,
            noise_sha256="a" * 64,
        )


def test_head_count_is_pinned_to_the_official_geometry():
    trace = _trace(0)[:, :, :2]
    with pytest.raises(ValueError, match="four action-query heads"):
        query_bank.bank_from_traced_queries(
            trace,
            fixture_id="ep000001-t00047",
            chunk_role=query_bank.FIT_CHUNK,
            step_idx=47,
            flow_times=FLOW_TIMES,
            noise_sha256="a" * 64,
        )
