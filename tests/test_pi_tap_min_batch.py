"""The pi serve's kernel-matching tap pad: default OFF, and a no-op on the real rows when ON.

The pi0.5 tap is jitted per input shape and XLA picks a different kernel below B=8. The omega cache
every workspace arm trained against was built at B=32, while this serve taps B=1 -- worth up to
max|d omega| ~ 1.43 on |omega| ~ 2.8. WSM_TAP_MIN_BATCH=8 pads the call to the cache's kernel.

Two properties matter and both are tested here without JAX or a GPU:
  * UNSET is byte-for-byte today's behaviour, so no sealed arm's serve can change by accident;
  * padding rows are copies of row 0 and only the real rows are returned, so the wrapper cannot
    leak a padding row into omega or reorder the real ones.
The numerical half of the claim -- that the real rows come back bit-identical from the REAL tap --
needs the GPU and is proved by scripts/proof on the box; see the report for the measured 0.000000.

Run: PYTHONPATH=. python3 -m pytest tests/test_pi_tap_min_batch.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_training.eval.serve_pi_05_wsm import (  # noqa: E402
    TAP_MIN_BATCH_ENV,
    pad_tap_batch,
    tap_min_batch,
)

VIEWS = ("agentview_left", "eye_in_hand", "agentview_right")


def _frames(rows):
    return {
        v: np.stack([np.full((2, 2, 3), 10 * (i + 1) + k, np.uint8) for i in range(rows)]) for k, v in enumerate(VIEWS)
    }


def test_unset_means_todays_behaviour():
    assert tap_min_batch({}) == 0
    assert tap_min_batch({TAP_MIN_BATCH_ENV: ""}) == 0
    assert tap_min_batch({TAP_MIN_BATCH_ENV: "8"}) == 8


@pytest.mark.parametrize("bad", ["eight", "-1"])
def test_garbage_fails_loud_rather_than_falling_back(bad):
    with pytest.raises(RuntimeError, match=TAP_MIN_BATCH_ENV):
        tap_min_batch({TAP_MIN_BATCH_ENV: bad})


@pytest.mark.parametrize("rows,min_batch", [(1, 0), (2, 0), (4, 4), (8, 2)])
def test_no_padding_when_not_requested_or_already_large_enough(rows, min_batch):
    frames, state, prompts = (
        _frames(rows),
        np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        [f"p{i}" for i in range(rows)],
    )
    out_f, out_s, out_p, real = pad_tap_batch(frames, state, prompts, min_batch)
    assert real == rows and out_p == prompts
    assert all(np.array_equal(out_f[v], frames[v]) for v in VIEWS)
    assert np.array_equal(out_s, state)


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_padding_preserves_the_real_rows_and_copies_row_zero(rows):
    frames = _frames(rows)
    state = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    prompts = [f"p{i}" for i in range(rows)]
    out_f, out_s, out_p, real = pad_tap_batch(frames, state, prompts, 8)

    assert real == rows, "the wrapper must know how many rows are real"
    assert out_s.shape[0] == 8 and out_p[:rows] == prompts
    for v in VIEWS:
        assert out_f[v].shape[0] == 8
        # the real rows are untouched and still FIRST, so result[:real] is the real answer
        assert np.array_equal(out_f[v][:rows], frames[v])
        # every padding row is a copy of row 0 -- no foreign content enters the call
        for j in range(rows, 8):
            assert np.array_equal(out_f[v][j], frames[v][0])
    assert np.array_equal(out_s[:rows], state)
    assert all(np.array_equal(out_s[j], state[0]) for j in range(rows, 8))
    assert out_p[rows:] == [prompts[0]] * (8 - rows)


def test_wrapper_returns_only_the_real_rows_when_padding_is_on(monkeypatch):
    """End-to-end through _tap_batch with a fake tap: omega must never see a padding row."""
    from types import SimpleNamespace

    from vla_training.eval.serve_pi_05_wsm import WSMPiInferWrapper

    monkeypatch.setenv(TAP_MIN_BATCH_ENV, "8")
    seen = {}

    class FakeTap:
        def tap(self, frames, state, prompts):
            seen["rows"] = len(prompts)
            n = len(prompts)
            # row-distinct output: keeping the wrong row, or reordering, changes the assertion
            return SimpleNamespace(
                patch_tokens=np.arange(n, dtype=np.float32)[:, None, None],
                lang_emb=np.arange(n, dtype=np.float32)[:, None] * 10.0,
            )

    wrapper = WSMPiInferWrapper.__new__(WSMPiInferWrapper)
    wrapper._tap = FakeTap()
    obs = [
        {
            "observation/image": np.zeros((2, 2, 3), np.uint8),
            "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
            "observation/right_image": np.zeros((2, 2, 3), np.uint8),
            "observation/state": np.zeros(3, np.float32),
        }
        for _ in range(2)
    ]
    patch, proprio = wrapper._tap_batch(obs, ["a", "b"])

    assert seen["rows"] == 8, "the tap must be CALLED at the padded size"
    assert patch.shape[0] == 2 and proprio.shape[0] == 2, "only real rows may be returned"
    assert np.array_equal(patch.reshape(-1), [0.0, 1.0])  # rows 0,1 -- not a padding row
    assert np.array_equal(proprio.reshape(-1), [0.0, 10.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
