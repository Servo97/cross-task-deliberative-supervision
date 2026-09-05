"""GR00T contiguous-window enumeration must match the pi side's exactly.

RoboTTT's recipe includes *which* frames form a window. If the two backbones enumerate windows
differently, the sync validation is comparing two different recipes and any groot-pi gap is
uninterpretable — so this is pinned against a fixture produced by calling the shipped pi function
`openpi.groot_utils.groot_openpi_dataset.contiguous_episode_windows`
(see `tests/fixtures/extract_seq_windows_pi.py`), not against a restatement of its rules.

The module under test is import-light on purpose: `contiguous_episode_windows` is a pure function
with no gr00t dependency, so this runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_training.train.train_base._groot_seq_common import contiguous_episode_windows

FIXTURE = Path(__file__).parent / "fixtures" / "seq_windows_pi.json"


def test_matches_pi_enumeration():
    data = json.loads(FIXTURE.read_text())
    lengths = data["lengths"]
    checked = 0
    for case in data["cases"]:
        wl, cs = case["window_len"], case["chunk_stride"]
        for traj_i, length in enumerate(lengths):
            expected = case["by_traj"].get(str(traj_i), [])
            got = contiguous_episode_windows(length, wl, cs)
            assert got == expected, (
                f"length={length} window_len={wl} chunk_stride={cs}: groot enumerated {got}, pi enumerated {expected}"
            )
            checked += 1
    assert checked == len(lengths) * len(data["cases"])


def test_windows_are_contiguous_ordered_and_non_overlapping():
    """The three structural properties the fast-weight chain depends on."""
    wins = contiguous_episode_windows(200, window_len=8, chunk_stride=8)
    assert wins, "expected windows for a 200-frame episode"
    seen = set()
    for w in wins:
        assert len(w) == 8
        assert w == sorted(w), "window steps must be in temporal order"
        assert all(b - a == 8 for a, b in zip(w, w[1:])), "steps must be evenly strided"
        assert not (seen & set(w)), "windows must not overlap"
        seen |= set(w)


def test_short_episodes_are_dropped_not_padded():
    """Fail-closed: a padded window would repeat frames into the recurrence or cross an episode."""
    assert contiguous_episode_windows(0, 8, 8) == []
    assert contiguous_episode_windows(7, 8, 8) == []  # only 1 strided step available
    assert contiguous_episode_windows(56, 8, 8) == []  # steps 0..48 == 7 steps, one short
    # 57 frames at stride 8 gives steps 0,8,...,56 == 8 steps, so exactly one window fits.
    assert contiguous_episode_windows(57, 8, 8) == [[0, 8, 16, 24, 32, 40, 48, 56]]
    assert contiguous_episode_windows(64, 8, 8) == [[0, 8, 16, 24, 32, 40, 48, 56]]


def test_trailing_partial_window_is_dropped():
    """120 frames at stride 8 = 15 steps; only the first 8 form a window, the tail is discarded."""
    wins = contiguous_episode_windows(120, window_len=8, chunk_stride=8)
    assert wins == [[0, 8, 16, 24, 32, 40, 48, 56]]


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="window_len"):
        contiguous_episode_windows(100, 0, 8)
    with pytest.raises(ValueError, match="chunk_stride"):
        contiguous_episode_windows(100, 8, 0)
