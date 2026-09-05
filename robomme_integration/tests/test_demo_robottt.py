from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

from wsm_settings import ROBOCASA_OPENPI_SRC

OPENPI = pathlib.Path(
    os.environ.get(
        "ROBOMME_OPENPI_SRC",
        str(ROBOCASA_OPENPI_SRC),
    )
)
if OPENPI.exists():
    sys.path.insert(0, str(OPENPI))

from robomme_integration.eval.runner import (  # noqa: E402
    DemoRoboTTTRunner,
    audit_demo_robottt_checkpoint,
)
from robomme_integration.training.demo_robottt import (  # noqa: E402
    install_demo_robottt_patch,
)


def _tree_distance(a, b) -> float:
    import jax
    import jax.numpy as jnp

    return float(
        jnp.sqrt(sum(jnp.sum(jnp.square(x - y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)))
    )


def _tiny_module():
    import openpi.models.robottt_fast_weights as robottt
    from flax import nnx

    cls = install_demo_robottt_patch()
    config = robottt.RoboTTTConfig(
        fast_weights=True,
        token_dim=16,
        fast_hidden=8,
        num_registers=2,
        cond_dim=8,
        state_dim=4,
        action_dim=3,
        action_horizon=2,
        base_inner_lr=0.05,
        gate_init=0.1,
        window_len=2,
        tbptt_segment=2,
    )
    return cls(config, rngs=nnx.Rngs(0))


def test_observation_only_context_commit_changes_fast_weights():
    import jax.numpy as jnp

    module = _tiny_module()
    w0 = module.init_state(1)
    state = jnp.zeros((1, 4), dtype=jnp.float32)
    tokens = jnp.arange(1 * 3 * 2048, dtype=jnp.float32).reshape(1, 3, 2048) / 2048
    mask = jnp.ones((1, 3), dtype=bool)

    committed = module.commit_context(w0, state, tokens, mask)
    assert _tree_distance(committed, w0) > 0


def test_checkpoint_audit_requires_a_trained_gate_and_visual_subtree():
    import jax.numpy as jnp

    module = _tiny_module()

    class _Model:
        robottt = True
        robottt_fast = module

    class _Policy:
        _model = _Model()

    module.alpha[...] = module.alpha[...] + jnp.asarray(0.01)
    summary = audit_demo_robottt_checkpoint(_Policy())
    assert summary["param_tensors"] > 0
    assert summary["inner_lr"] > 0
    assert summary["gate_max_abs_tanh"] > 0


def test_padding_slots_do_not_update_and_demo_changes_execution_state():
    import jax.numpy as jnp

    module = _tiny_module()
    state = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    actions = jnp.ones((1, 2, 2, 3), dtype=jnp.float32)
    tokens = jnp.stack(
        [
            jnp.zeros((1, 3, 2048), dtype=jnp.float32),
            jnp.ones((1, 3, 2048), dtype=jnp.float32),
        ],
        axis=1,
    )
    token_mask = jnp.ones((1, 2, 3), dtype=bool)

    no_updates = jnp.zeros((1, 2), dtype=bool)
    _, w_padding = module.run_demo_sequence(
        state,
        actions,
        tokens,
        token_mask,
        no_updates,
        no_updates,
    )
    assert _tree_distance(w_padding, module.init_state(1)) == 0

    demo_then_execution = jnp.asarray([[True, False]])
    execution = jnp.asarray([[False, True]])
    _, w_with_demo = module.run_demo_sequence(
        state,
        actions,
        tokens,
        token_mask,
        demo_then_execution,
        execution,
    )
    _, w_without_demo = module.run_demo_sequence(
        state,
        actions,
        tokens,
        token_mask,
        no_updates,
        execution,
    )
    assert _tree_distance(w_with_demo, w_without_demo) > 0


def test_execution_commit_excludes_the_unexecuted_action_suffix():
    import jax.numpy as jnp

    module = _tiny_module()
    state = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    tokens = jnp.ones((1, 1, 3, 2048), dtype=jnp.float32)
    token_mask = jnp.ones((1, 1, 3), dtype=bool)
    demo = jnp.zeros((1, 1), dtype=bool)
    execution = jnp.ones((1, 1), dtype=bool)
    actions_a = jnp.asarray([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    actions_b = actions_a.at[:, :, 1].set(jnp.asarray([99.0, 98.0, 97.0]))

    _, w_a = module.run_demo_sequence(
        state,
        actions_a,
        tokens,
        token_mask,
        demo,
        execution,
        commit_action_steps=1,
    )
    _, w_b = module.run_demo_sequence(
        state,
        actions_b,
        tokens,
        token_mask,
        demo,
        execution,
        commit_action_steps=1,
    )
    assert _tree_distance(w_a, w_b) == 0


def test_stateful_inference_uses_fixed_buckets_and_keeps_rows_isolated():
    import jax
    import jax.numpy as jnp

    seen: dict[str, int] = {}

    class _Policy:
        def infer_batch(self, requests):
            seen["requests"] = len(requests)
            return [
                {
                    "actions": np.zeros((20, 8), dtype=np.float32),
                    "norm_state": np.full((32,), index, dtype=np.float32),
                    "norm_actions": np.full((20, 32), index, dtype=np.float32),
                }
                for index in range(len(requests))
            ]

    runner = DemoRoboTTTRunner.__new__(DemoRoboTTTRunner)
    runner._jax = jax
    runner._jnp = jnp
    runner._policy = _Policy()
    runner.execution_commit_steps = 10

    def observation_batch(rows):
        seen["context_bucket"] = len(rows)
        transformed = {"state": np.zeros((len(rows), 32), dtype=np.float32)}
        return object(), transformed

    runner._observation_batch = observation_batch
    runner._encode = lambda observation: (
        jnp.zeros((4, 17, 2048), dtype=jnp.float32),
        jnp.ones((4, 17), dtype=bool),
    )

    def condition(w, state, tokens, token_mask):
        seen["fast_batch"] = int(state.shape[0])
        return jnp.zeros((state.shape[0], 1024), dtype=jnp.float32)

    runner._condition = condition
    fast_states = [{"w": jnp.full((1, 2, 2), index, dtype=jnp.float32)} for index in range(3)]
    results, pending = runner.infer_batch(
        fast_states,
        [{"row": index} for index in range(3)],
    )
    assert seen == {"context_bucket": 4, "fast_batch": 4, "requests": 3}
    assert len(results) == len(pending) == 3
    assert all(item.actions.shape == (1, 10, 32) for item in pending)
    assert [float(item.actions[0, 0, 0]) for item in pending] == [0, 1, 2]


def test_context_order_is_not_collapsed_to_an_unordered_average():
    import jax.numpy as jnp

    module = _tiny_module()
    w0 = module.init_state(1)
    state = jnp.zeros((1, 4), dtype=jnp.float32)
    mask = jnp.ones((1, 3), dtype=bool)
    first = jnp.zeros((1, 3, 2048), dtype=jnp.float32)
    second = jnp.ones((1, 3, 2048), dtype=jnp.float32)

    forward = module.commit_context(w0, state, first, mask)
    forward = module.commit_context(forward, state, second, mask)
    reverse = module.commit_context(w0, state, second, mask)
    reverse = module.commit_context(reverse, state, first, mask)
    assert _tree_distance(forward, reverse) > 0


def test_uniform_prefix_contract_matches_numpy_reference():
    from robomme_integration.sequence import uniformly_sample_prefix

    selected, valid = uniformly_sample_prefix(np.arange(100), 16, pad_index=100)
    assert len(selected) == 16
    assert selected[0] == 0 and selected[-1] == 99
    assert all(a < b for a, b in zip(selected, selected[1:]))
    assert valid.all()
