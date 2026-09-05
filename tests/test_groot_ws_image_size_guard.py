"""The GR00T websocket adapter must REFUSE the wrong observation resolution.

THE HAZARD. `run_remembench_eval.py --obs-image-size` defaults to 224 because that is pi0.5's
training resolution and every sealed pi arm has to stay byte-identical. GR00T N1.7 trains at 256
(`image_target_size: [256, 256]`). A groot serve fed 224 px frames resamples every frame a SECOND
time in front of the resample training used. It does not crash, it does not warn, and it degrades
every number the server produces — the worst possible failure shape, because the run completes and
looks fine.

`run_remembench_box.sh` sets 256 for `SERVE_KIND=groot`, but that is one launcher. Anything run by
hand, by a future harness, or from a copy-pasted command line inherits the 224 default. So the guard
lives on the RECEIVING end, where every client has to pass through it, and this test is what keeps
it there.

Pure numpy — no gr00t, no torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_training.eval.serve_groot_ws import (  # noqa: E402
    CAMERA_MAP,
    GROOT_NATIVE_IMAGE_SIZE,
    STATE_DIM,
    pack_observation,
)


def _request(size: int, *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    request = {wire: rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for wire in CAMERA_MAP}
    request["observation/state"] = np.zeros(STATE_DIM, dtype=np.float32)
    request["prompt"] = "put the pot on the stove"
    return request


def test_the_native_size_is_256_not_224():
    """A constant, but a load-bearing one: it is the value the pi default silently disagrees with."""
    assert GROOT_NATIVE_IMAGE_SIZE == 256


def test_native_frames_are_accepted():
    obs = pack_observation(_request(256), expected_image_size=GROOT_NATIVE_IMAGE_SIZE)
    for groot_key in CAMERA_MAP.values():
        assert obs[groot_key].shape == (1, 1, 256, 256, 3)
        assert obs[groot_key].dtype == np.uint8


def test_the_pi_default_224_is_refused_loudly():
    """The exact hazard: a runner launched without --obs-image-size 256."""
    with pytest.raises(ValueError) as excinfo:
        pack_observation(_request(224), expected_image_size=GROOT_NATIVE_IMAGE_SIZE)
    message = str(excinfo.value)
    assert "224x224" in message and "256x256" in message
    # The error has to name the fix, or whoever hits it at 3am re-derives it from scratch.
    assert "--obs-image-size" in message


@pytest.mark.parametrize("size", [112, 224, 320, 512])
def test_every_wrong_size_is_refused(size):
    with pytest.raises(ValueError, match="expected 256x256"):
        pack_observation(_request(size), expected_image_size=GROOT_NATIVE_IMAGE_SIZE)


def test_a_single_bad_camera_is_enough():
    """A mixed-resolution request must fail, not pass because two of three cameras are right."""
    request = _request(256)
    request["observation/wrist_image"] = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="observation/wrist_image"):
        pack_observation(request, expected_image_size=GROOT_NATIVE_IMAGE_SIZE)


def test_non_square_frames_are_refused():
    request = _request(256)
    request["observation/image"] = np.zeros((256, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected 256x256"):
        pack_observation(request, expected_image_size=GROOT_NATIVE_IMAGE_SIZE)


def test_the_check_can_be_disabled_for_a_deliberate_ablation():
    """`--expected-image-size 0` -> None. The escape hatch must exist and must be explicit."""
    obs = pack_observation(_request(224), expected_image_size=None)
    assert obs["video.robot0_agentview_left"].shape == (1, 1, 224, 224, 3)


def test_the_guard_is_on_by_default():
    """Called WITHOUT the keyword the old signature is preserved (no check) — so the DEFAULT that
    matters is the server's, and it must be the native size, not None."""
    import inspect

    from vla_training.eval.serve_groot_ws import GrootWebsocketPolicy

    default = inspect.signature(GrootWebsocketPolicy.__init__).parameters["expected_image_size"]
    assert default.default == GROOT_NATIVE_IMAGE_SIZE, (
        "the adapter must guard by default; an opt-in guard would not have caught the 224 launch"
    )


# --------------------------------------------------------------------------------------------
# The policy-reply container. Found on the box 2026-08-07: Gr00tSimPolicyWrapper.get_action
# returns `(action_dict, extras)`, NOT a bare dict. A static contract check cannot see this — the
# keys, the layout and the chunk length are all exactly right, only the container differs — and the
# symptom is an unrelated `TypeError: '<' not supported between instances of 'dict' and 'dict'`
# from `sorted(action_dict)`. These pin the shape that actually comes back.
# --------------------------------------------------------------------------------------------

from vla_training.eval.serve_groot_ws import (  # noqa: E402
    ACTION_DIM,
    ACTION_LAYOUT,
    normalize_policy_reply,
    unpack_actions,
)

HORIZON = 16


def _action_dict(batched: bool = True) -> dict:
    out = {}
    for key, span in ACTION_LAYOUT:
        width = span.stop - span.start
        value = np.full((HORIZON, width), float(span.start), dtype=np.float32)
        out[key] = value[None, ...] if batched else value
    return out


def test_the_wrapper_tuple_is_unwrapped():
    """(action_dict, extras) — the shape the real Gr00tSimPolicyWrapper returns."""
    actions = _action_dict()
    unwrapped = normalize_policy_reply((actions, {"extras": 1}))
    assert unwrapped is actions
    assert set(unwrapped) == {k for k, _ in ACTION_LAYOUT}


def test_a_bare_dict_still_works():
    actions = _action_dict()
    assert normalize_policy_reply(actions) is actions


def test_a_reply_that_is_not_action_first_is_refused():
    with pytest.raises(ValueError, match="expected the action dict first"):
        normalize_policy_reply((["not-a-dict"], _action_dict()))


def test_an_unrecognized_reply_type_is_refused():
    with pytest.raises(ValueError, match="unrecognized policy reply type"):
        normalize_policy_reply(np.zeros((16, 12), dtype=np.float32))


def test_unpack_actions_accepts_the_real_tuple_reply():
    chunk = unpack_actions((_action_dict(), {"extras": 1}))
    assert chunk.shape == (HORIZON, ACTION_DIM)
    assert chunk.dtype == np.float32
    # Every layout slot filled from its own key: slot value == that key's span start.
    for _key, span in ACTION_LAYOUT:
        assert np.all(chunk[:, span] == float(span.start))


def test_unpack_actions_rejects_non_finite():
    reply = _action_dict()
    reply["action.gripper_close"][0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        unpack_actions((reply, {}))
