"""`build_sequence_collator` against the REAL GR00T N1.7 data collator + Cosmos processor.

WHY THIS FILE EXISTS. `vlm_content` is the one key the stock collator cannot stack: it is a
`{"text", "images"}` dict that the collator flattens across the batch into a SINGLE processor call,
and it explicitly refuses pre-tokenized keys (`processing_gr00t_n1d7.py`: `input_ids`,
`attention_mask`, `pixel_values`, `image_grid_thw` -> `raise Exception("Not implemented")`). Every
other RoboTTT test uses pure-python fixtures; this one is the only place the sequence collator meets
the actual `Qwen3VLProcessor`, so it is the only place a shape or ordering mistake would surface.

THE LOAD-BEARING ASSERTION is `test_window_major_order_matches_stock_collator`: the sequence
collator's VLM tensors must be BITWISE the tensors the stock collator produces for the same B*L
single-step items presented in WINDOW-MAJOR order (`for item: for step:`, flat index `b*L + t`).
That is the order a `[B, L, ...]` tensor takes under a row-major `reshape(B*L, ...)`, which is how
the action head will pair observations with fast-weight state. Get it wrong — step-major, say — and
every observation is paired with another timestep's conditioning: the batch still has the right
shape, the loss still goes down, and the policy is meaningless. The test therefore also asserts the
STEP-major arrangement does NOT match, so it cannot pass vacuously.

RUNNING IT. Needs the gr00t venv and a local HF cache for the gated `nvidia/Cosmos-Reason2-2B`
(weights not required here — processor only, but the cache asset ships the full repo):

    env -u HF_TOKEN ~/Research/Isaac-GR00T/.venv/bin/python \
        -m pytest tests/test_groot_seq_collator.py -q          # if pytest is in that venv
    env -u HF_TOKEN ~/Research/Isaac-GR00T/.venv/bin/python \
        tests/test_groot_seq_collator.py                       # self-contained runner, no pytest

It SKIPS cleanly everywhere else, so the openpi-venv suite is unaffected.
"""

from __future__ import annotations

import contextlib
import re as _re
import sys
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # the gr00t venv has no pytest; see __main__ below

    class _PytestShim:
        """Just enough pytest to let this file import and self-run inside the gr00t venv."""

        @staticmethod
        def mark_skipif(condition, reason=""):
            def decorate(fn):
                fn.__skip__ = (bool(condition), reason)
                return fn

            return decorate

        class mark:  # noqa: N801 - mirrors the pytest namespace
            skipif = None

        @staticmethod
        @contextlib.contextmanager
        def raises(expected, match=None):
            try:
                yield
            except expected as exc:
                if match and not _re.search(match, str(exc)):
                    raise AssertionError(f"{exc!r} does not match {match!r}") from None
            else:
                raise AssertionError(f"did not raise {expected}")

    _PytestShim.mark.skipif = staticmethod(_PytestShim.mark_skipif)
    pytest = _PytestShim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
B, L, VIEWS = 2, 3, 2  # 2 windows x 3 chunk-steps x 2 camera views
STATE_DIM, ACTION_HORIZON, ACTION_DIM = 29, 16, 29
IMG = 32  # tiny frames: this test is about batching, not pixels


def _build_collator():
    """The REAL Gr00tN1d7DataCollator, resolved offline. Returns None if unavailable."""
    try:
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    except Exception:
        return None
    try:
        return Gr00tN1d7DataCollator(
            model_name=MODEL_NAME,
            transformers_loading_kwargs={"local_files_only": True},
        )
    except Exception:
        return None


_COLLATOR = _build_collator()
requires_gr00t = pytest.mark.skipif(
    _COLLATOR is None,
    reason="needs the gr00t venv + a local HF cache for the gated nvidia/Cosmos-Reason2-2B",
)


def _vlm_content(collator, rng, text: str) -> dict:
    """One step's `vlm_content`, built the way `_apply_vlm_processing` builds it.

    Same structure the production transform emits: the chat template rendered to a string with one
    image placeholder per view, plus the matching list of PIL images. Using the collator's own
    processor for the template is deliberate — a hand-written string could drift from the real one.
    """
    import numpy as np
    from PIL import Image

    images = [Image.fromarray(rng.integers(0, 255, (IMG, IMG, 3), dtype=np.uint8)) for _ in range(VIEWS)]
    conversation = [
        {
            "role": "user",
            "content": [*[{"type": "image", "image": im} for im in images], {"type": "text", "text": text}],
        }
    ]
    rendered = collator.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    return {"text": rendered, "images": images, "conversation": conversation}


def _single_step(collator, rng, text: str) -> dict:
    """One stock (non-sequence) feature dict: exactly the keys the transform emits."""
    import numpy as np

    return {
        "state": rng.standard_normal((STATE_DIM,)).astype(np.float32),
        "action": rng.standard_normal((ACTION_HORIZON, ACTION_DIM)).astype(np.float32),
        "action_mask": np.ones((ACTION_HORIZON, ACTION_DIM), dtype=bool),
        "embodiment_id": 10,
        "vlm_content": _vlm_content(collator, rng, text),
    }


def _windows(collator, seed: int = 0):
    """(window features, window-major flat single-step features) built from ONE rng stream.

    Both views are constructed from the same draws in the same order, so any difference the
    equivalence test reports is the collator's doing and not the fixture's.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    steps = [[_single_step(collator, rng, f"window {b} step {t}") for t in range(L)] for b in range(B)]

    features = []
    for per_step in steps:
        window = {
            "state": np.stack([d["state"] for d in per_step]),
            "action": np.stack([d["action"] for d in per_step]),
            "action_mask": np.stack([d["action_mask"] for d in per_step]),
            "embodiment_id": np.asarray([d["embodiment_id"] for d in per_step]),
            "vlm_content": [d["vlm_content"] for d in per_step],  # the LIST the collator flattens
            "seq_len": np.asarray(L, dtype=np.int64),
            "loss_mask": np.ones(L, dtype=np.float32),
            "reset": np.asarray([t == 0 for t in range(L)], dtype=np.bool_),
        }
        features.append(window)

    window_major = [steps[b][t] for b in range(B) for t in range(L)]  # flat index b*L + t
    return features, window_major


VLM_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


@requires_gr00t
def test_real_processor_collator_flattens_windows():
    from vla_training.train.train_base._groot_seq_common import build_sequence_collator

    features, _ = _windows(_COLLATOR)
    batch = build_sequence_collator(_COLLATOR)(features)["inputs"]

    # VLM keys arrive at batch B*L — one pseudo-sample per (window, step).
    assert batch["input_ids"].shape[0] == B * L, batch["input_ids"].shape
    assert batch["attention_mask"].shape[0] == B * L
    # Qwen3VL packs patches flat and describes them with one grid row per IMAGE, so the image count
    # is B*L*VIEWS — the check that the per-step image lists were concatenated, not dropped.
    assert batch["image_grid_thw"].shape[0] == B * L * VIEWS, batch["image_grid_thw"].shape

    # Everything else keeps its window axis: [B, L, ...].
    assert tuple(batch["state"].shape) == (B, L, STATE_DIM)
    assert tuple(batch["action"].shape) == (B, L, ACTION_HORIZON, ACTION_DIM)
    assert tuple(batch["embodiment_id"].shape) == (B, L)
    assert tuple(batch["loss_mask"].shape) == (B, L)
    assert tuple(batch["reset"].shape) == (B, L)
    assert int(batch["seq_window_len"]) == L


@requires_gr00t
def test_window_major_order_matches_stock_collator():
    """The invariant: sequence-collated VLM tensors == stock collation of the SAME items, b*L + t."""
    import torch

    from vla_training.train.train_base._groot_seq_common import build_sequence_collator

    features, window_major = _windows(_COLLATOR)
    seq = build_sequence_collator(_COLLATOR)(features)["inputs"]
    stock = _COLLATOR(window_major)["inputs"]

    for key in VLM_KEYS:
        assert key in seq and key in stock, key
        assert seq[key].shape == stock[key].shape, (key, seq[key].shape, stock[key].shape)
        assert torch.equal(seq[key], stock[key]), f"{key} differs from window-major stock collation"

    # Non-vacuity: the STEP-major arrangement (t*B + b) must NOT match. If it did, this test would
    # be blind to exactly the transposition it exists to catch.
    step_major = [window_major[b * L + t] for t in range(L) for b in range(B)]
    other = _COLLATOR(step_major)["inputs"]
    assert not torch.equal(seq["input_ids"], other["input_ids"]), (
        "step-major collation is indistinguishable from window-major — the fixture cannot "
        "discriminate the two orders, so the equivalence assertion above proves nothing"
    )


@requires_gr00t
def test_pre_tokenized_keys_are_still_refused():
    """The vendored guard must still fire — the sequence path must not smuggle tokens past it."""
    features, _ = _windows(_COLLATOR)
    features[0]["input_ids"] = [[1, 2, 3]]
    with pytest.raises(Exception, match="Not implemented"):
        build = __import__("vla_training.train.train_base._groot_seq_common", fromlist=["build_sequence_collator"])
        build.build_sequence_collator(_COLLATOR)(features)


@requires_gr00t
def test_ragged_windows_raise():
    from vla_training.train.train_base._groot_seq_common import build_sequence_collator

    features, _ = _windows(_COLLATOR)
    features[1]["vlm_content"] = features[1]["vlm_content"][:-1]  # L-1 instead of L
    with pytest.raises(ValueError, match="ragged window lengths"):
        build_sequence_collator(_COLLATOR)(features)


@requires_gr00t
def test_non_sequence_batch_is_byte_identical_to_stock():
    """A plain single-step batch must pass straight through — the wrapper is inert off the seq path."""
    import numpy as np
    import torch

    from vla_training.train.train_base._groot_seq_common import build_sequence_collator

    rng = np.random.default_rng(7)
    plain = [_single_step(_COLLATOR, rng, f"step {i}") for i in range(B)]
    wrapped = build_sequence_collator(_COLLATOR)(plain)["inputs"]
    stock = _COLLATOR(plain)["inputs"]
    assert set(wrapped) == set(stock)
    assert "seq_window_len" not in wrapped
    for key in stock:
        assert torch.equal(wrapped[key], stock[key]), key


if __name__ == "__main__":  # self-contained runner: the gr00t venv has no pytest
    if _COLLATOR is None:
        print("SKIP: gr00t venv / Cosmos processor cache unavailable")
        raise SystemExit(0)
    failures = 0
    for name, fn in sorted(dict(globals()).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        skip, reason = getattr(fn, "__skip__", (False, ""))
        if skip:
            print(f"SKIP {name}: {reason}")
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - a standalone runner reports, it does not re-raise
            failures += 1
            import traceback

            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILED'}")
    raise SystemExit(1 if failures else 0)
