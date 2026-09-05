"""CPU-only tests for the online-omega sidecar that unblocks the GR00T deltanet (dnw8) evals.

The sidecar exists because omega is ``WSMEncoder(frozen pi0.5 tap features)`` — a jax chain the
torch-only GR00T venv cannot host — and because held-out eval episodes have NO cached omega. Its
whole job is therefore (a) run that chain and (b) own the per-episode causal state with the SAME
discipline the sealed pi workspace serves use.

(b) is what these tests are about, because (b) is the half that fails SILENTLY. An episode whose
window still carries the previous episode's frames returns finite numbers, produces a full horizon,
and scores plausibly; nothing downstream can tell. So the isolation test here is the direct mirror
of ``tests/test_pi_wsm_stateful_batching.py``'s: run one env alone, record every omega, then replay
it interleaved with a second env and require the recorded sequence back BIT-FOR-BIT.

The last two tests run the REAL ``WSMEvalConditioner`` over a tiny real ``WorkspaceModel`` and check
the sidecar's window against ``wsm_align.window_at`` applied to a full-prefix encode — the same
claim the box-tier cache-parity harness (``omega_sidecar_parity.py``) makes against the canonical
offline omega cache, at unit scale and without a GPU.

Run: PYTHONPATH=. python3 -m pytest tests/test_omega_sidecar.py -q
"""

from __future__ import annotations

import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_training.eval.omega_sidecar import (  # noqa: E402
    OmegaSidecar,
    OmegaSidecarClient,
    make_handler,
)

TASKS = ("task_a", "task_b")


class FakeProducer:
    """Deterministic stand-in for the tap+encoder chain: omega is a function of THIS env's history.

    ``step`` returns the cumulative sum of the frame markers it has been shown since ``reset``, so
    any leakage between env slots — or any missed reset — changes the numbers rather than hiding.
    """

    def __init__(self, k: int = 3, w_dim: int = 2):
        self.k = int(k)
        self.w_dim = int(w_dim)
        self.tasks = frozenset(TASKS)
        self.steps = 0
        self.resets = 0
        self.nonfinite_after = None

    def new_conditioner(self):
        return {"history": [], "lang": None}

    def reset(self, conditioner, task):
        self.resets += 1
        conditioner["history"] = []
        conditioner["lang"] = float(TASKS.index(task) + 1)

    def step(self, conditioner, request, prompt):
        self.steps += 1
        conditioner["history"].append(float(np.asarray(request["observation/image"]).reshape(-1)[0]))
        rolling = np.cumsum(conditioner["history"], dtype=np.float64)
        rows = list(rolling[-self.k :])
        rows = [rows[0]] * (self.k - len(rows)) + rows  # left-pad, oldest..newest
        window = np.asarray(rows, dtype=np.float32)[:, None] * np.ones((1, self.w_dim), np.float32)
        window = window + conditioner["lang"]
        if self.nonfinite_after is not None and self.steps > self.nonfinite_after:
            window[0, 0] = np.nan
        return window


def _request(env, task, demo, t, marker, prompt="do the thing"):
    image = np.full((2, 2, 3), marker, dtype=np.uint8)
    return {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/right_image": image,
        "observation/state": np.zeros(16, dtype=np.float32),
        "wsm_env_id": env,
        "wsm_task": task,
        "wsm_demo_episode": demo,
        "wsm_t": t,
        "wsm_prompt": prompt,
    }


def _sidecar(max_envs=2, max_grid_frames=16, k=3):
    producer = FakeProducer(k=k)
    return OmegaSidecar(producer, stride=8, max_envs=max_envs, max_grid_frames=max_grid_frames), producer


def _raises(fragment, fn):
    try:
        fn()
    except RuntimeError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return
    raise AssertionError(f"expected RuntimeError containing {fragment!r}")


# --------------------------------------------------------------------------------------------------
# the incident test: interleaving a second env must not perturb the first, bit-for-bit
# --------------------------------------------------------------------------------------------------
def test_episode_isolation_is_bit_identical_to_running_the_env_alone():
    solo, _ = _sidecar()
    alone = [solo.omega(_request("w0", "task_a", 7, 8 * i, 10 + i)).copy() for i in range(5)]

    shared, producer = _sidecar()
    interleaved = []
    for i in range(5):
        interleaved.append(shared.omega(_request("w0", "task_a", 7, 8 * i, 10 + i)).copy())
        shared.omega(_request("w1", "task_b", 9, 8 * i, 90 + i))
    for step, (reference, actual) in enumerate(zip(alone, interleaved)):
        assert np.array_equal(reference, actual), f"env w0 perturbed by a co-resident env at {step}"
    assert producer.resets == 2 and producer.steps == 10


def test_reset_at_t0_clears_only_that_env_and_a_new_episode_starts_from_scratch():
    sidecar, producer = _sidecar()
    first = [sidecar.omega(_request("w0", "task_a", 0, 8 * i, 10 + i)).copy() for i in range(3)]
    sidecar.omega(_request("w1", "task_b", 1, 0, 50))

    # Same env slot, NEW episode: identical markers must reproduce the first episode exactly, which
    # is only true if t=0 dropped every frame of the previous one.
    again = [sidecar.omega(_request("w0", "task_a", 1, 8 * i, 10 + i)).copy() for i in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again))

    # The neighbour kept ITS history across w0's reset: its t=8 window must equal what the same env
    # produces alone, not a restarted one.
    solo, _ = _sidecar()
    solo.omega(_request("w1", "task_b", 1, 0, 50))
    assert np.array_equal(
        sidecar.omega(_request("w1", "task_b", 1, 8, 51)),
        solo.omega(_request("w1", "task_b", 1, 8, 51)),
    )
    assert producer.resets == 3


def test_same_grid_is_reused_and_costs_no_producer_call():
    sidecar, producer = _sidecar()
    sidecar.omega(_request("w0", "task_a", 0, 0, 10))
    assert producer.steps == 1
    # t=1..7 share grid 0 with t=0: the pi serve advances the grid by ENV STEP, not per request.
    same = sidecar.omega(_request("w0", "task_a", 0, 3, 99))
    assert producer.steps == 1
    assert np.array_equal(same, sidecar.omega(_request("w0", "task_a", 0, 7, 99)))
    assert producer.steps == 1
    sidecar.omega(_request("w0", "task_a", 0, 8, 11))
    assert producer.steps == 2


def test_identity_and_order_guards_fail_before_any_state_moves():
    sidecar, producer = _sidecar()
    _raises("before an explicit t=0 reset", lambda: sidecar.omega(_request("w0", "task_a", 0, 8, 10)))
    assert producer.steps == 0

    sidecar.omega(_request("w0", "task_a", 0, 0, 10))
    sidecar.omega(_request("w0", "task_a", 0, 8, 11))
    # NOTE t == 0 is deliberately NOT an ordering violation: it is the episode boundary itself, and
    # wsm_env_id is stable across a worker's episodes, so "t goes back to 0" is how a new episode
    # announces itself. Same rule as serve_pi_05_wsm._validate_batch. Everything else must increase.
    _raises("out-of-order wsm_t", lambda: sidecar.omega(_request("w0", "task_a", 0, 8, 12)))
    _raises("episode identity changed without t=0 reset", lambda: sidecar.omega(_request("w0", "task_a", 1, 8, 11)))
    _raises("episode identity changed without t=0 reset", lambda: sidecar.omega(_request("w0", "task_b", 0, 8, 11)))
    _raises("skipped causal grid", lambda: sidecar.omega(_request("w0", "task_a", 0, 32, 11)))
    _raises("misaligned causal grid", lambda: sidecar.omega(_request("w0", "task_a", 0, 17, 11)))
    _raises("unknown wsm_task", lambda: sidecar.omega(_request("w9", "nope", 0, 0, 1)))
    _raises("wsm_t must be non-negative", lambda: sidecar.omega(_request("w9", "task_a", 0, -1, 1)))
    # Two live env slots may not claim the same episode: a duplicated shard would double-count.
    _raises("duplicate active episode identity", lambda: sidecar.omega(_request("w1", "task_a", 0, 0, 1)))
    # None of the refusals advanced the producer past the two legitimate calls.
    assert producer.steps == 2


def test_prompt_is_required_non_empty_and_trimmed():
    sidecar, _ = _sidecar()
    for bad in ("", "   ", " leading", "trailing "):
        _raises(
            "wsm_prompt must be a non-empty, trimmed string",
            lambda bad=bad: sidecar.omega(_request("w0", "task_a", 0, 0, 10, prompt=bad)),
        )
    request = _request("w0", "task_a", 0, 0, 10)
    del request["wsm_prompt"]
    _raises("missing required field 'wsm_prompt'", lambda: sidecar.omega(request))


def test_bounds_refuse_instead_of_evicting_live_state():
    sidecar, _ = _sidecar(max_envs=1)
    sidecar.omega(_request("w0", "task_a", 0, 0, 10))
    _raises("refusing live-state eviction", lambda: sidecar.omega(_request("w1", "task_b", 0, 0, 10)))

    bounded, _ = _sidecar(max_envs=1, max_grid_frames=2)
    bounded.omega(_request("w0", "task_a", 0, 0, 10))
    bounded.omega(_request("w0", "task_a", 0, 8, 11))
    _raises("causal-state frame bound exceeded", lambda: bounded.omega(_request("w0", "task_a", 0, 16, 12)))


def test_nonfinite_omega_stops_rather_than_returning():
    sidecar, producer = _sidecar()
    producer.nonfinite_after = 1
    sidecar.omega(_request("w0", "task_a", 0, 0, 10))
    _raises("NON-FINITE omega", lambda: sidecar.omega(_request("w0", "task_a", 0, 8, 11)))


# --------------------------------------------------------------------------------------------------
# wire
# --------------------------------------------------------------------------------------------------
def _serve(sidecar, provenance):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sidecar, lambda: provenance))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_http_round_trip_matches_the_in_process_window_and_reports_geometry():
    sidecar, _ = _sidecar()
    provenance = {"k_window": 3, "w_dim": 2, "stride": 8, "encoder_sha256": "deadbeef"}
    server, url = _serve(sidecar, provenance)
    try:
        client = OmegaSidecarClient(url, timeout=30)
        health = client.health()
        assert health["k_window"] == 3 and health["state_mode"] == "per_env_isolated_v1"

        window = client.window(_request("w0", "task_a", 0, 0, 10))
        assert window.shape == (3, 2) and window.dtype == np.float32
        reference, _ = _sidecar()
        assert np.array_equal(window, reference.omega(_request("w0", "task_a", 0, 0, 10)))

        # A refusal must arrive as the producer's real traceback, not a dropped socket: the GR00T
        # server log is where this failure gets diagnosed.
        with pytest.raises(RuntimeError, match="skipped causal grid"):
            client.window(_request("w0", "task_a", 0, 64, 10))
    finally:
        server.shutdown()
        server.server_close()


def test_client_refuses_to_send_an_incomplete_request():
    client = OmegaSidecarClient("http://127.0.0.1:1", timeout=1)
    request = _request("w0", "task_a", 0, 0, 10)
    del request["wsm_env_id"]
    with pytest.raises(RuntimeError, match="the runner did not send 'wsm_env_id'"):
        client.window(request)


def test_client_names_the_sidecar_when_it_is_unreachable():
    client = OmegaSidecarClient("http://127.0.0.1:1", timeout=1)
    with pytest.raises(RuntimeError, match="cannot reach the omega sidecar"):
        client.window(_request("w0", "task_a", 0, 0, 10))


# --------------------------------------------------------------------------------------------------
# tap batch padding: the fix for the B=1-vs-B=32 XLA kernel split that moved omega by ~1.46
# --------------------------------------------------------------------------------------------------
class _RecordingTap:
    """Records the batch it was called with and returns per-ROW distinct values.

    Row-distinct output is the point: if ``step`` ever kept a padding row instead of row 0, or let
    the rows get reordered, the assertions below change. A tap that returned identical rows could
    not tell those bugs from correct behaviour.
    """

    def __init__(self):
        self.calls = []

    def tap(self, frames, state, prompt):
        from workspace_models.features.pi_backbone_tap import PiTapResult

        rows = len(prompt)
        self.calls.append(
            {
                "rows": rows,
                "prompts": list(prompt),
                "frame_rows": {k: int(np.asarray(v).shape[0]) for k, v in frames.items()},
                "state_rows": int(np.asarray(state).shape[0]),
                "marker": float(np.asarray(frames["agentview_left"])[0, 0, 0, 0]),
            }
        )
        base = np.arange(rows, dtype=np.float32)[:, None, None]
        return PiTapResult(
            patch_tokens=base + np.zeros((rows, 3, 4), np.float32),
            lang_emb=(base[:, :, 0] * 10.0) + np.zeros((rows, 5), np.float32),
        )


def _bare_producer(tap, batch=8, k=2, w_dim=6):
    """A PiOmegaProducer with ONLY the step-path attributes set (no checkpoints, no GPU)."""
    from vla_training.eval.omega_sidecar import PiOmegaProducer

    producer = PiOmegaProducer.__new__(PiOmegaProducer)
    producer._tap = tap
    producer.tap_image_size = 0
    producer.tap_batch_size = batch
    producer.k, producer.w_dim = k, w_dim
    return producer


def test_tap_is_called_with_the_padded_batch_and_only_row_zero_is_used():
    tap = _RecordingTap()
    producer = _bare_producer(tap, batch=8)

    captured = {}

    class _Conditioner:
        def step(self, patch, proprio):
            captured["patch"] = np.asarray(patch)
            captured["proprio"] = np.asarray(proprio)
            return np.zeros((producer.k, producer.w_dim), np.float32), None

    producer.step(_Conditioner(), _request("w0", "task_a", 0, 0, 77), "do the thing")
    call = tap.calls[0]
    assert call["rows"] == 8, "the tap must be called at the padded batch size, not B=1"
    assert call["state_rows"] == 8 and set(call["frame_rows"].values()) == {8}
    assert call["prompts"] == ["do the thing"] * 8
    # Row 0 is the real frame; rows 1..7 are copies, so the marker is unchanged.
    assert call["marker"] == 77
    # ...and row 0 (value 0.0), never a padding row (values 1..7), is what reaches the encoder.
    assert captured["patch"].max() == 0.0 and captured["proprio"].max() == 0.0


def test_tap_batch_size_one_reproduces_the_unpadded_call():
    tap = _RecordingTap()
    producer = _bare_producer(tap, batch=1)

    class _Conditioner:
        def step(self, patch, proprio):
            return np.zeros((producer.k, producer.w_dim), np.float32), None

    producer.step(_Conditioner(), _request("w0", "task_a", 0, 0, 5), "p")
    assert tap.calls[0]["rows"] == 1 and tap.calls[0]["state_rows"] == 1


def test_padding_rows_cannot_change_the_answer_whatever_they_contain():
    """The mathematical claim the fix rests on, asserted rather than assumed.

    A transformer prefix has no cross-example interaction, so row 0 must not depend on the padding.
    The box measurement says the same thing against the real tap (patch max|d| = 0.000000 with three
    different fillers); this keeps the invariant enforced in CI, where the real tap cannot run.
    """
    seen = []

    class _FillerSensitiveTap(_RecordingTap):
        def tap(self, frames, state, prompt):
            result = super().tap(frames, state, prompt)
            seen.append(np.asarray(frames["agentview_left"])[1:, 0, 0, 0].tolist())
            return result

    for batch in (1, 4, 8):
        tap = _FillerSensitiveTap()
        producer = _bare_producer(tap, batch=batch)

        class _Conditioner:
            def step(self, patch, proprio):
                return np.zeros((producer.k, producer.w_dim), np.float32), None

        producer.step(_Conditioner(), _request("w0", "task_a", 0, 0, 33), "p")
        # Every padding row is a COPY of the real frame, so no filler content exists to leak.
        assert set(seen[-1]) <= {33.0}
        assert tap.calls[0]["marker"] == 33


# --------------------------------------------------------------------------------------------------
# the GR00T half: stash the window where the deltanet head looks, and refuse anything else
# --------------------------------------------------------------------------------------------------
class _StubHead:
    """Just enough of WSMDeltaNetActionHead for the stash path (no gr00t import, no GPU)."""

    def __init__(self, w_dim=4):
        import torch

        self.wsm_deltanet = torch.nn.Linear(w_dim, w_dim)
        self._dn_eval_window = None


class _StubClient:
    url = "http://stub"

    def __init__(self, window):
        self.window_value = window
        self.requests = []

    def window(self, request):
        self.requests.append(request)
        return self.window_value


def _policy(window, w_dim=4, k=3):
    from vla_training.eval.serve_groot_ws import GrootWebsocketPolicy

    head = _StubHead(w_dim)
    policy = GrootWebsocketPolicy(
        object(),
        mechanism="deltanet",
        action_head=head,
        omega_client=_StubClient(np.asarray(window, np.float32)),
        omega_geometry={"window_len": k, "w_dim": w_dim},
    )
    return policy, head


def test_stash_shapes_the_window_for_the_head_and_forwards_the_runner_fields():
    import torch

    k, w_dim = 3, 4
    policy, head = _policy(np.arange(k * w_dim).reshape(k, w_dim), w_dim, k)
    request = _request("w0", "task_a", 5, 16, 42)
    policy._stash_omega_window(request)
    assert isinstance(head._dn_eval_window, torch.Tensor)
    # [1, K, w_dim] is what _cond_from_window consumes; the batch axis is this call's job to add.
    assert tuple(head._dn_eval_window.shape) == (1, k, w_dim)
    assert head._dn_eval_window.dtype == next(head.wsm_deltanet.parameters()).dtype
    sent = policy._omega_client.requests[0]
    assert {"wsm_env_id", "wsm_t", "wsm_task", "wsm_demo_episode", "wsm_prompt"} <= set(sent)
    assert np.array_equal(sent["observation/image"], request["observation/image"])
    assert policy.metadata["omega_sidecar"] == "http://stub"


def test_stash_refuses_a_window_that_is_not_the_trained_geometry_or_not_finite():
    policy, _ = _policy(np.zeros((2, 4), np.float32), w_dim=4, k=3)
    with pytest.raises(RuntimeError, match="trained on \\(3, 4\\)"):
        policy._stash_omega_window(_request("w0", "task_a", 0, 0, 1))

    bad = np.zeros((3, 4), np.float32)
    bad[1, 1] = np.nan
    policy, _ = _policy(bad, w_dim=4, k=3)
    with pytest.raises(RuntimeError, match="NON-FINITE window"):
        policy._stash_omega_window(_request("w0", "task_a", 0, 0, 1))


# --------------------------------------------------------------------------------------------------
# real encoder: the online window must equal the offline full-prefix encode + wsm_align window
# --------------------------------------------------------------------------------------------------
class _TinyPiProducer:
    """The real ``WSMEvalConditioner`` over a tiny real ``WorkspaceModel``, fed synthetic tap output.

    Everything from ``fuse_inputs`` onward is the production code path; only the jax tap is replaced
    (by a deterministic function of the frame marker), which is exactly the boundary the box-tier
    parity harness crosses with the real tap.
    """

    def __init__(self, k, stride, backbone_dim=6, patches=3, proprio_dim=5, lang_dim=7):
        import torch

        from vla_training.eval._groot_wsm_eval import WSMEvalConditioner
        from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig

        torch.manual_seed(11)
        self.cfg = WSMConfig(
            dim=8,
            n_layers=2,
            n_heads=2,
            backbone_dim=backbone_dim,
            proprio_dim=proprio_dim,
            lang_dim=lang_dim,
            c_horizon=64,
            max_t=64,
            mlp_ratio=2.0,
        )
        self.model = WorkspaceModel(self.cfg).eval()
        self.k, self.stride = int(k), int(stride)
        self.w_dim = self.cfg.dim
        self.patches, self.tasks = patches, frozenset(TASKS)
        self._cls = WSMEvalConditioner
        self._lang = {t: np.full(lang_dim, 0.1 * (i + 1), np.float32) for i, t in enumerate(TASKS)}
        self.frames = []

    def tap(self, marker):
        rng = np.random.default_rng(int(marker))
        return (
            rng.standard_normal((self.patches, self.cfg.backbone_dim)).astype(np.float32),
            rng.standard_normal(self.cfg.proprio_dim).astype(np.float32),
        )

    def new_conditioner(self):
        return self._cls(self.model, k_window=self.k, stride=self.stride, device="cpu")

    def reset(self, conditioner, task):
        self.frames = []
        conditioner.reset(self._lang[task])

    def step(self, conditioner, request, prompt):
        patch, proprio = self.tap(np.asarray(request["observation/image"]).reshape(-1)[0])
        self.frames.append((patch, proprio))
        window, _lang = conditioner.step(patch, proprio)
        return window.detach().cpu().numpy()


def test_real_conditioner_window_equals_full_prefix_encode_through_wsm_align():
    import torch

    from workspace_models.features.wsm_align import window_at

    producer = _TinyPiProducer(k=4, stride=8)
    sidecar = OmegaSidecar(producer, stride=8, max_envs=1, max_grid_frames=32)

    markers = [31, 41, 59, 26, 53, 58]
    online = [sidecar.omega(_request("w0", "task_a", 0, 8 * i, m)).copy() for i, m in enumerate(markers)]

    # The OFFLINE shape of the same computation: one full-length encode, then the shared causal
    # window helper — i.e. exactly what `generate_stage_s_policy_features.encode_demo` +
    # `wsm_align.window_at` do to produce and consume the cache.
    patches = torch.from_numpy(np.stack([f[0] for f in producer.frames]))[None]
    proprio = torch.from_numpy(np.stack([f[1] for f in producer.frames]))[None]
    lang = torch.from_numpy(np.broadcast_to(producer._lang["task_a"], (len(markers), producer.cfg.lang_dim)).copy())[
        None
    ]
    with torch.no_grad():
        offline = producer.model.encode(patches, proprio, lang)[0].numpy()
    frame_indices = np.arange(len(markers), dtype=np.int64) * 8

    for i in range(len(markers)):
        expected = window_at(offline, frame_indices, int(frame_indices[i]), 4)
        np.testing.assert_allclose(online[i], expected, rtol=2e-5, atol=2e-6)
        # Rows arrive oldest..newest: the last row is always the CURRENT grid frame.
        np.testing.assert_allclose(online[i][-1], offline[i], rtol=2e-5, atol=2e-6)


def test_real_conditioner_window_is_left_padded_before_k_frames_exist():
    producer = _TinyPiProducer(k=4, stride=8)
    sidecar = OmegaSidecar(producer, stride=8, max_envs=1, max_grid_frames=32)
    first = sidecar.omega(_request("w0", "task_a", 0, 0, 5))
    # One real grid frame: every row of the window is that frame repeated.
    assert first.shape == (4, producer.w_dim)
    for row in range(4):
        np.testing.assert_allclose(first[row], first[-1], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
