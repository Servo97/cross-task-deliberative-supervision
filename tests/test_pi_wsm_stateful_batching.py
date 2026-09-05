"""CPU-only tests for identity-aware batched pi online-workspace serving.

Fake tap/conditioner/policy components prove routing and state isolation without JAX, a GPU, RoboCasa,
or a checkpoint. The final test exercises WSMEvalConditioner.step_many with a tiny fake torch encoder.
Run: PYTHONPATH=. python3 tests/test_pi_wsm_stateful_batching.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vla_training.eval.serve_pi_05_wsm as serve_pi_wsm  # noqa: E402
from vla_training.eval.serve_pi_05_wsm import WSMPiInferWrapper  # noqa: E402


class FakeTap:
    def __init__(self):
        self.calls = []

    def tap(self, frames, state, prompts):
        batch = len(prompts)
        self.calls.append(
            {
                "batch": batch,
                "prompts": list(prompts),
                "markers": np.asarray(frames["agentview_left"])[:, 0, 0, 0].tolist(),
            }
        )
        marker = np.asarray(frames["agentview_left"], dtype=np.float32)[:, 0, 0, 0]
        return SimpleNamespace(
            patch_tokens=marker[:, None, None],
            lang_emb=(marker + 0.5)[:, None],
        )


class FakeConditioner:
    step_many_groups = []

    def __init__(self, k=1):
        self.k = int(k)
        self.history = []
        self.lang = None
        self.single_steps = 0

    def reset(self, lang):
        self.history = []
        self.lang = np.asarray(lang, dtype=np.float32)

    def _result(self):
        # One newest omega_t still summarizes the full causal history.
        encoded = list(np.cumsum(self.history, dtype=np.float32))
        values = encoded[-self.k :]
        values = [values[0]] * (self.k - len(values)) + values
        return np.asarray(values, dtype=np.float32)[:, None], self.lang.copy()

    def step(self, patch, proprio):
        self.single_steps += 1
        self.history.append(float(np.asarray(patch).reshape(-1)[0]))
        return self._result()

    @classmethod
    def step_many(cls, conditioners, patches, proprio):
        for conditioner, patch in zip(conditioners, patches):
            conditioner.history.append(float(np.asarray(patch).reshape(-1)[0]))
        groups = {}
        for conditioner in conditioners:
            groups.setdefault(len(conditioner.history), 0)
            groups[len(conditioner.history)] += 1
        cls.step_many_groups.append(tuple(sorted(groups.items())))
        return [conditioner._result() for conditioner in conditioners]


class FakePolicy:
    metadata = {"fake": True}

    def __init__(self):
        self.infer_calls = []
        self.batch_calls = []

    @staticmethod
    def _result(obs):
        for key in ("wsm_env_id", "wsm_task", "wsm_demo_episode", "wsm_t", "wsm_prompt"):
            assert key not in obs
        seed = int(obs["policy_noise_seed"])
        return {
            "request_id": obs["request_id"],
            "omega": np.asarray(obs["wsm_w_window"], dtype=np.float32).copy(),
            "lang": np.asarray(obs["wsm_lang"], dtype=np.float32).copy(),
            "noise_draw": float(np.random.default_rng(seed).standard_normal()),
        }

    def infer(self, obs, **kwargs):
        self.infer_calls.append((obs, dict(kwargs)))
        return self._result(obs)

    def infer_batch(self, obs_list, **kwargs):
        self.batch_calls.append((list(obs_list), dict(kwargs)))
        return [self._result(obs) for obs in obs_list]


TABLE = {
    "task_a": np.asarray([100.0], dtype=np.float32),
    "task_b": np.asarray([200.0], dtype=np.float32),
}


def _obs(env, task, demo, t, marker, seed=None):
    image = np.full((2, 2, 3), marker, dtype=np.uint8)
    return {
        "request_id": f"{env}:{task}:{demo}:{t}",
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/right_image": image,
        "observation/state": np.asarray([marker], dtype=np.float32),
        "prompt": f"terse {task}",
        "wsm_env_id": env,
        "wsm_task": task,
        "wsm_demo_episode": demo,
        "wsm_t": t,
        "policy_noise_seed": int(marker if seed is None else seed),
    }


def _wrapper(max_envs=2, max_grid_frames=16, require_wsm_prompt=False):
    policy = FakePolicy()
    tap = FakeTap()
    template = FakeConditioner()
    wrapper = WSMPiInferWrapper(
        policy,
        tap,
        template,
        TABLE,
        stride=8,
        max_envs=max_envs,
        max_grid_frames=max_grid_frames,
        conditioner_factory=FakeConditioner,
        require_wsm_prompt=require_wsm_prompt,
    )
    return wrapper, policy, tap, template


def _raises(fragment, fn):
    try:
        fn()
    except RuntimeError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return
    raise AssertionError(f"expected RuntimeError containing {fragment!r}")


def test_batch_isolation_routing_and_one_call_per_stage():
    FakeConditioner.step_many_groups.clear()
    wrapper, policy, tap, _ = _wrapper()
    out = wrapper.infer_batch([_obs("env_a", "task_a", 1, 0, 10), _obs("env_b", "task_b", 2, 0, 20)])
    assert [item["request_id"] for item in out] == [
        "env_a:task_a:1:0",
        "env_b:task_b:2:0",
    ]
    np.testing.assert_array_equal(out[0]["omega"][:, 0], [10])
    np.testing.assert_array_equal(out[1]["omega"][:, 0], [20])
    np.testing.assert_array_equal(out[0]["lang"], [100])
    np.testing.assert_array_equal(out[1]["lang"], [200])
    assert [call["batch"] for call in tap.calls] == [2]
    assert len(policy.batch_calls) == 1 and not policy.infer_calls
    assert FakeConditioner.step_many_groups == [((1, 2),)]

    out = wrapper.infer_batch([_obs("env_b", "task_b", 2, 8, 21), _obs("env_a", "task_a", 1, 8, 11)])
    assert [item["request_id"] for item in out] == [
        "env_b:task_b:2:8",
        "env_a:task_a:1:8",
    ]
    np.testing.assert_array_equal(out[0]["omega"][:, 0], [41])
    np.testing.assert_array_equal(out[1]["omega"][:, 0], [21])
    assert [call["batch"] for call in tap.calls] == [2, 2]
    assert len(policy.batch_calls) == 2
    assert FakeConditioner.step_many_groups[-1] == ((2, 2),)


def test_same_grid_reuse_and_reset_only_target_env():
    FakeConditioner.step_many_groups.clear()
    wrapper, policy, tap, _ = _wrapper()
    wrapper.infer_batch([_obs("env_a", "task_a", 1, 0, 10), _obs("env_b", "task_b", 2, 0, 20)])
    reused = wrapper.infer_batch([_obs("env_a", "task_a", 1, 1, 99), _obs("env_b", "task_b", 2, 1, 88)])
    assert len(tap.calls) == 1
    np.testing.assert_array_equal(reused[0]["omega"][:, 0], [10])
    np.testing.assert_array_equal(reused[1]["omega"][:, 0], [20])
    for item in reused:
        timing = item["policy_timing"]
        assert timing["wsm_request_batch_n"] == 2
        assert timing["wsm_new_grid_batch_n"] == 0
        assert timing["wsm_tap_amortized_ms"] == 0.0
        assert timing["wsm_encoder_amortized_ms"] == 0.0

    out = wrapper.infer_batch([_obs("env_a", "task_a", 3, 0, 30), _obs("env_b", "task_b", 2, 8, 21)])
    np.testing.assert_array_equal(out[0]["omega"][:, 0], [30])
    np.testing.assert_array_equal(out[1]["omega"][:, 0], [41])
    assert tap.calls[-1]["batch"] == 2
    # A reset history has F=1; B retained its private history and has F=2.
    assert FakeConditioner.step_many_groups[-1] == ((1, 1), (2, 1))
    assert len(policy.batch_calls) == 3


def test_identity_and_order_guards_are_fail_loud_before_routing():
    wrapper, policy, tap, _ = _wrapper()
    missing = _obs("env_a", "task_a", 1, 0, 10)
    missing.pop("wsm_env_id")
    _raises("missing required identity", lambda: wrapper.infer(missing))
    assert not tap.calls and not policy.infer_calls

    duplicate = _obs("env_a", "task_a", 1, 0, 10)
    _raises(
        "duplicate wsm_env_id",
        lambda: wrapper.infer_batch([duplicate, dict(duplicate)]),
    )
    _raises(
        "duplicate active episode identity",
        lambda: wrapper.infer_batch([_obs("env_a", "task_a", 1, 0, 10), _obs("env_b", "task_a", 1, 0, 20)]),
    )
    _raises(
        "before an explicit t=0 reset",
        lambda: wrapper.infer(_obs("env_a", "task_a", 1, 1, 10)),
    )

    wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    _raises(
        "episode identity changed without t=0",
        lambda: wrapper.infer(_obs("env_a", "task_b", 1, 1, 11)),
    )
    _raises(
        "episode identity changed without t=0",
        lambda: wrapper.infer(_obs("env_a", "task_a", 9, 1, 11)),
    )
    wrapper.infer(_obs("env_a", "task_a", 1, 1, 11))
    _raises(
        "out-of-order",
        lambda: wrapper.infer(_obs("env_a", "task_a", 1, 1, 11)),
    )
    _raises(
        "misaligned causal grid",
        lambda: wrapper.infer(_obs("env_a", "task_a", 1, 10, 12)),
    )
    _raises(
        "skipped causal grid",
        lambda: wrapper.infer(_obs("env_a", "task_a", 1, 16, 12)),
    )


def test_state_bounds_fail_instead_of_evicting():
    wrapper, _, _, _ = _wrapper(max_envs=1)
    wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    _raises(
        "active env-state bound exceeded",
        lambda: wrapper.infer(_obs("env_b", "task_b", 2, 0, 20)),
    )
    assert set(wrapper._states) == {"env_a"}

    wrapper, _, _, _ = _wrapper(max_envs=1, max_grid_frames=1)
    wrapper.infer(_obs("env_a", "task_a", 1, 0, 10))
    _raises(
        "causal-state frame bound exceeded",
        lambda: wrapper.infer(_obs("env_a", "task_a", 1, 8, 11)),
    )


def test_k1_keeps_inner_infer_and_single_conditioner_semantics():
    wrapper, policy, tap, template = _wrapper(max_envs=1)
    first = wrapper.infer(_obs("env_a", "task_a", 1, 0, 10, seed=123))
    np.testing.assert_array_equal(first["omega"][:, 0], [10])
    assert len(policy.infer_calls) == 1 and not policy.batch_calls
    assert [call["batch"] for call in tap.calls] == [1]
    assert template.single_steps == 1
    assert policy.infer_calls[0][0]["policy_noise_seed"] == 123

    same_grid = wrapper.infer(_obs("env_a", "task_a", 1, 1, 99, seed=124))
    np.testing.assert_array_equal(same_grid["omega"][:, 0], [10])
    assert len(tap.calls) == 1 and template.single_steps == 1

    metadata = wrapper.metadata
    assert metadata["wsm_state_mode"] == "per_env_isolated_v1"
    assert metadata["infer_batch"] is True
    assert metadata["wsm_stride"] == 8
    assert metadata["wsm_max_envs"] == 1
    assert metadata["wsm_required_identity_fields"] == ["wsm_env_id", "wsm_task", "wsm_demo_episode", "wsm_t"]
    assert metadata["wsm_required_signal_fields"] == []
    reset = wrapper.infer(_obs("env_a", "task_b", 2, 0, 30, seed=125))
    np.testing.assert_array_equal(reset["omega"][:, 0], [30])
    np.testing.assert_array_equal(reset["lang"], [200])
    assert set(wrapper._states) == {"env_a"}


def test_required_canonical_prompt_fails_closed_and_stays_private():
    wrapper, policy, tap, _ = _wrapper(max_envs=1, require_wsm_prompt=True)
    missing = _obs("env_a", "task_a", 1, 0, 10)
    _raises("missing required signal field 'wsm_prompt'", lambda: wrapper.infer(missing))
    assert not wrapper._states and not tap.calls and not policy.infer_calls

    supplied = _obs("env_a", "task_a", 1, 0, 10)
    supplied["wsm_prompt"] = "canonical task a"
    wrapper.infer(supplied)
    assert tap.calls[0]["prompts"] == ["canonical task a"]
    assert "wsm_prompt" not in policy.infer_calls[0][0]
    assert wrapper.metadata["wsm_required_signal_fields"] == ["wsm_prompt"]

    cached = _obs("env_a", "task_a", 1, 1, 11)
    cached["wsm_prompt"] = "  not-trimmed"
    _raises("must be a non-empty, trimmed string", lambda: wrapper.infer(cached))
    assert wrapper._states["env_a"].last_t == 0


def test_seeded_policy_result_is_batch_grouping_and_order_independent():
    batched, _, _, _ = _wrapper(max_envs=2)
    rows = [
        _obs("env_a", "task_a", 1, 0, 10, seed=8675309),
        _obs("env_b", "task_b", 2, 0, 20, seed=42),
    ]
    batch_results = {item["request_id"]: item for item in batched.infer_batch(rows[::-1])}

    singles = {}
    for row in rows:
        wrapper, _, _, _ = _wrapper(max_envs=1)
        item = wrapper.infer(row)
        singles[item["request_id"]] = item

    assert batch_results.keys() == singles.keys()
    for request_id in singles:
        assert batch_results[request_id]["noise_draw"] == singles[request_id]["noise_draw"]
        np.testing.assert_array_equal(batch_results[request_id]["omega"], singles[request_id]["omega"])


def test_timing_reports_completed_per_request_amortized_values_not_batch_totals():
    wrapper, _, _, _ = _wrapper(max_envs=2)
    # Calls, in order: wrapper start, prepare start, tap start/end, encoder start/end,
    # prepare end, policy start/end, wrapper end.
    ticks = iter([0.0, 0.0, 1.0, 5.0, 6.0, 10.0, 12.0, 20.0, 28.0, 40.0])
    original_clock = serve_pi_wsm.time.perf_counter
    serve_pi_wsm.time.perf_counter = lambda: next(ticks)
    try:
        out = wrapper.infer_batch([_obs("env_a", "task_a", 1, 0, 10), _obs("env_b", "task_b", 2, 0, 20)])
    finally:
        serve_pi_wsm.time.perf_counter = original_clock

    assert out[0]["policy_timing"] is not out[1]["policy_timing"]
    for item in out:
        timing = item["policy_timing"]
        assert timing == {
            "wsm_request_batch_n": 2,
            "wsm_new_grid_batch_n": 2,
            "wsm_tap_amortized_ms": 2000.0,
            "wsm_encoder_amortized_ms": 2000.0,
            "wsm_prepare_amortized_ms": 6000.0,
            "policy_call_amortized_ms": 4000.0,
            "wsm_end_to_end_amortized_ms": 20000.0,
        }


def test_real_conditioner_batches_fusion_and_equal_length_temporal_work():
    import torch

    from vla_training.eval._groot_wsm_eval import WSMEvalConditioner

    class FakeEncoder:
        def __init__(self):
            self.fusion_calls = []
            self.temporal_calls = []

        def fuse_inputs(self, patch, proprio, cond):
            self.fusion_calls.append((int(patch.shape[0]), int(patch.shape[1])))
            fused = patch.mean(dim=(-1, -2))[..., None] + proprio.mean(dim=-1)[..., None]
            projected_cond = cond.mean(dim=-1)[..., None]
            return fused, projected_cond

        def encode_fused(self, fused, cond):
            self.temporal_calls.append((int(fused.shape[0]), int(fused.shape[1])))
            return torch.cumsum(fused + cond, dim=1)

    def conditioner(encoder, lang):
        item = WSMEvalConditioner(encoder, k_window=1, stride=8, device="cpu")
        item.reset(np.asarray([lang], dtype=np.float32))
        return item

    batch_encoder = FakeEncoder()
    left = conditioner(batch_encoder, 100)
    right = conditioner(batch_encoder, 200)
    batched = WSMEvalConditioner.step_many(
        [left, right],
        [np.asarray([[10.0]]), np.asarray([[20.0]])],
        [np.asarray([0.5]), np.asarray([1.5])],
    )
    assert batch_encoder.fusion_calls == [(2, 1)]
    assert batch_encoder.temporal_calls == [(2, 1)]

    seq_encoder = FakeEncoder()
    expected = [
        conditioner(seq_encoder, 100).step(np.asarray([[10.0]]), np.asarray([0.5])),
        conditioner(seq_encoder, 200).step(np.asarray([[20.0]]), np.asarray([1.5])),
    ]
    for actual, reference in zip(batched, expected):
        torch.testing.assert_close(actual[0], reference[0])
        torch.testing.assert_close(actual[1], reference[1])

    # Make histories asynchronous. New-frame fusion is still one B=2 call; the causal temporal stack
    # groups F=2 and F=3 separately.
    left.step(np.asarray([[11.0]]), np.asarray([0.5]))
    batch_encoder.fusion_calls.clear()
    batch_encoder.temporal_calls.clear()
    WSMEvalConditioner.step_many(
        [left, right],
        [np.asarray([[12.0]]), np.asarray([[21.0]])],
        [np.asarray([0.5]), np.asarray([1.5])],
    )
    assert batch_encoder.fusion_calls == [(2, 1)]
    assert sorted(batch_encoder.temporal_calls) == [(1, 2), (1, 3)]
    assert not hasattr(left, "_patches") and not hasattr(left, "_proprio")
    assert all(tuple(token.shape) == (1,) for token in left._fused + left._conds)


def test_real_conditioner_rejects_nonfinite_fused_inputs():
    import torch

    from vla_training.eval._groot_wsm_eval import WSMEvalConditioner

    class NonfiniteEncoder:
        def fuse_inputs(self, patch, proprio, cond):
            batch = patch.shape[0]
            fused = torch.full((batch, 1, 1), float("nan"), device=patch.device)
            return fused, torch.zeros_like(fused)

        def encode_fused(self, fused, cond):
            # Deliberately hide the corrupt input in the output.  The combined graph check must
            # still inspect the cached fused/condition tensors, not only omega.
            return torch.zeros_like(fused)

    conditioner = WSMEvalConditioner(NonfiniteEncoder(), k_window=1, stride=8, device="cpu")
    conditioner.reset(np.asarray([1.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="NON-FINITE online workspace graph"):
        conditioner.step(np.asarray([[1.0]]), np.asarray([2.0]))


def test_workspace_encoder_forward_composes_exactly_and_cached_frames_match():
    import torch

    from workspace_models.networks.workspace_latent import WorkspaceEncoder
    from workspace_models.networks.wsm_model import WSMConfig

    for input_norm in (False, True):
        torch.manual_seed(7)
        cfg = WSMConfig(
            dim=8,
            n_layers=2,
            n_heads=2,
            backbone_dim=6,
            proprio_dim=5,
            lang_dim=7,
            c_horizon=10,
            max_t=10,
            mlp_ratio=2.0,
            input_norm=input_norm,
        )
        encoder = WorkspaceEncoder(cfg).eval()
        patches = torch.randn(2, 4, 3, cfg.backbone_dim)
        proprio = torch.randn(2, 4, cfg.proprio_dim)
        lang = torch.randn(2, 4, cfg.lang_dim)
        with torch.no_grad():
            direct = encoder(patches, proprio, lang)
            fused, cond = encoder.fuse_inputs(patches, proprio, lang)
            composed = encoder.encode_fused(fused, cond)
        torch.testing.assert_close(direct, composed, rtol=0, atol=0)

        fused_frames, cond_frames = [], []
        for frame in range(patches.shape[1]):
            with torch.no_grad():
                fused_frame, cond_frame = encoder.fuse_inputs(
                    patches[:, frame : frame + 1],
                    proprio[:, frame : frame + 1],
                    lang[:, frame : frame + 1],
                )
            fused_frames.append(fused_frame[:, 0])
            cond_frames.append(cond_frame[:, 0])
            with torch.no_grad():
                incremental = encoder.encode_fused(
                    torch.stack(fused_frames, dim=1),
                    torch.stack(cond_frames, dim=1),
                )
            torch.testing.assert_close(incremental[:, -1], direct[:, frame], rtol=2e-5, atol=2e-6)


def test_websocket_server_gather_is_strictly_opt_in_and_reports_capability():
    import os

    openpi_src = Path(__file__).resolve().parents[3] / "robocasa_openpi" / "src"
    sys.path.insert(0, str(openpi_src))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class Batchable:
        metadata = {}

        def infer(self, obs):
            return obs

        def infer_batch(self, obs):
            return obs

    class Singleton:
        metadata = {}
        infer_batch = None

        def infer(self, obs):
            return obs

    prior = os.environ.get("WSM_ENVS_PER_GPU")
    try:
        os.environ["WSM_ENVS_PER_GPU"] = "1"
        server = WebsocketPolicyServer(Batchable(), metadata={})
        assert server._concurrent is False
        assert server._metadata["infer_batch"] is True
        assert server._metadata["server_concurrent"] is False
        assert server._metadata["server_state_mode"] == "stateless_v1"

        os.environ["WSM_ENVS_PER_GPU"] = "2"
        server = WebsocketPolicyServer(Batchable(), metadata={})
        try:
            assert server._concurrent is True
            assert server._metadata["server_concurrent"] is True
            assert server._metadata["server_batch_envs"] == 2
        finally:
            server._policy.close()

        server = WebsocketPolicyServer(Singleton(), metadata={})
        assert server._concurrent is False
        assert server._metadata["infer_batch"] is False
    finally:
        if prior is None:
            os.environ.pop("WSM_ENVS_PER_GPU", None)
        else:
            os.environ["WSM_ENVS_PER_GPU"] = prior


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
