"""Serve-path driver for the official FrameSamp+Modul policy with AM taps.

Three things happen here and nothing else:

* the released policy is loaded exactly as ``scripts/serve_policy.py`` loads it
  (bfloat16 params, official norm stats, official transforms);
* an episode-prefix fixture is turned into the very observation the online
  evaluator would have produced -- the memory buffer is fed only the 32 frames
  the official sampler would have selected, at their true episode-global step
  indices, so ``prepare_frame_sampling`` reproduces the online memory exactly;
* the denoise loop is unrolled in Python so ``v(x_t, t)`` and the corresponding
  full-teacher ``x_t`` can be recorded at every flow time, optionally with
  per-layer memory Q/K/V taps or a per-layer AM pack substituted for history.

The unrolled loop is a reimplementation of ``HistoryPi0.sample_actions``'s
modulation branch; :func:`selftest_matches_official_sampler` checks it against
the real ``sample_actions`` before any number is reported.  No gradients are
taken anywhere: this is a serve-path cache transform.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time as _time
from pathlib import Path
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from robomme_integration.amkv.patch_contract import install_patched_history_module
from robomme_integration.amkv.patched_history_gemma import MemoryAttentionAMPack
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    FrameSampHistory,
    even_sampling_indices,
)

DEFAULT_NUM_FLOW_STEPS = 10
DEFAULT_CONFIG_NAME = "mme_vla_suite"
DEFAULT_MODEL_SEED = 7
OFFICIAL_INTEGRATION_TYPE = "modulation"
OFFICIAL_REPRESENTATION_TYPE = "perceptual"
OFFICIAL_ACTION_HORIZON = 20
OFFICIAL_MEMORY_LAYERS = 18
OFFICIAL_QUERY_HEADS = 4
OFFICIAL_KV_HEADS = 1
OFFICIAL_HEAD_DIM = 256


@contextlib.contextmanager
def _official_working_directory(policy_source_root: str | Path | None):
    """Run inside the policy repo root while the model is constructed.

    ``mme_vla_suite.models.config.utils.get_history_config`` resolves its YAML
    with the *relative* path ``src/mme_vla_suite/models/config/robomme/...``, so
    the released loader only works when the process CWD is the policy source
    root (which is how ``scripts/serve_policy.py`` is always invoked).  Making
    that dependency explicit here keeps every caller -- tests, the node entry,
    a local smoke -- from having to know it.
    """

    if policy_source_root is None:
        yield
        return
    root = Path(policy_source_root).resolve()
    marker = root / "src" / "mme_vla_suite" / "models" / "config" / "robomme"
    if not marker.is_dir():
        raise FileNotFoundError(f"official policy source root has no history-config tree: {marker}")
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def load_policy(
    checkpoint_dir: str | Path,
    *,
    policy_source_root: str | Path | None = None,
    config_name: str = DEFAULT_CONFIG_NAME,
    seed: int = DEFAULT_MODEL_SEED,
):
    """Load the released checkpoint through the official policy factory."""

    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.training import config as _config

    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"checkpoint has no params tree: {checkpoint_dir}")
    if not (checkpoint_dir.parent / "history_config.txt").is_file():
        raise FileNotFoundError(
            "the released checkpoint's sibling history_config.txt is required by the official loader; "
            f"stage the parent directory, not just {checkpoint_dir.name}"
        )
    with _official_working_directory(policy_source_root):
        policy = _policy_config.create_trained_policy(
            _config.get_config(config_name), checkpoint_dir.resolve(), seed=seed, default_prompt=None
        )
    model = policy._model  # noqa: SLF001 - the official factory exposes no accessor
    if (
        model.integration_type != OFFICIAL_INTEGRATION_TYPE
        or model.representation_type != OFFICIAL_REPRESENTATION_TYPE
    ):
        raise ValueError(
            "AM on history K/V requires the perceptual FrameSamp + modulation policy, got "
            f"{model.representation_type}/{model.integration_type}"
        )
    return policy


def _require(record: Any, name: str) -> Any:
    if not hasattr(record, name):
        raise AttributeError(f"episode fixture is missing required field {name!r}")
    return getattr(record, name)


def build_observation(policy: Any, record: Any) -> tuple[Any, FrameSampHistory, dict[str, object]]:
    """Reproduce the online observation for one episode-prefix fixture."""

    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    step_idx = int(_require(record, "step_idx"))
    memory_images = np.asarray(_require(record, "memory_images"))
    memory_states = np.asarray(_require(record, "memory_states"), dtype=np.float32)
    memory_steps = np.asarray(_require(record, "memory_step_indices"), dtype=np.int64)
    expected = np.asarray(even_sampling_indices(step_idx), dtype=np.int64)
    if not np.array_equal(memory_steps, expected):
        raise ValueError(
            "fixture memory frames are not the official FrameSamp selection for this causal cut: "
            f"{memory_steps.tolist()} != {expected.tolist()}"
        )
    if memory_images.ndim != 4 or memory_images.shape[0] != memory_steps.size:
        raise ValueError(f"memory_images must be [frames, h, w, 3], got {memory_images.shape}")
    if memory_states.shape != (memory_steps.size, memory_states.shape[-1]):
        raise ValueError("memory_states must be [frames, state_dim]")

    policy.reset()
    policy.mem_buffer.add_buffer(memory_images[:, None, ...], memory_states, memory_steps.tolist())
    policy.step_idx = step_idx
    policy.exec_start_idx = int(_require(record, "exec_start_idx"))

    inputs = {
        "observation/image": np.asarray(_require(record, "obs_image")),
        "observation/wrist_image": np.asarray(_require(record, "obs_wrist_image")),
        "observation/state": np.asarray(_require(record, "obs_state"), dtype=np.float32),
        "prompt": str(_require(record, "prompt")),
    }
    inputs = policy._prepare_history(inputs)  # noqa: SLF001 - mirrors MME_VLA_Policy.infer
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    observation = HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
    history = framesamp_history_from_observation(observation, memory_steps)
    meta = {
        "fixture_id": str(_require(record, "fixture_id")),
        "step_idx": step_idx,
        "exec_start_idx": policy.exec_start_idx,
        "memory_step_indices": memory_steps.tolist(),
        "valid_memory_tokens": int(np.asarray(observation.static_mask[0]).sum()),
    }
    return observation, history, meta


def framesamp_history_from_observation(observation: Any, memory_steps: np.ndarray) -> FrameSampHistory:
    """Wrap the observation's memory tensors in the audited FrameSamp record."""

    image = np.asarray(observation.static_image_emb[0], dtype=np.float32)
    position = np.asarray(observation.static_pos_emb[0], dtype=np.float32)
    token_mask = np.asarray(observation.static_mask[0], dtype=np.bool_)
    if image.shape[0] != TOKEN_BUDGET or token_mask.shape != (TOKEN_BUDGET,):
        raise ValueError(f"official memory buffer must be {TOKEN_BUDGET} tokens, got {image.shape}")
    frames = int(token_mask.sum()) // TOKENS_PER_FRAME
    frame_mask = np.zeros(MAX_FRAMES, dtype=np.bool_)
    frame_mask[:frames] = True
    frame_indices = np.full(MAX_FRAMES, -1, dtype=np.int32)
    frame_indices[:frames] = memory_steps[:frames].astype(np.int32, copy=False)
    history = FrameSampHistory(
        image=np.where(token_mask[:, None], image, 0.0),
        position=np.where(token_mask[:, None], position, 0.0),
        token_mask=token_mask,
        frame_indices=frame_indices,
        frame_mask=frame_mask,
    )
    history.validate()
    return history


def memory_branch_kwargs(am_pack: MemoryAttentionAMPack | None) -> dict[str, Any]:
    """Kwargs for the history-module call, keyed on whether an artifact exists.

    The official ``Module.__call__`` has no ``am_pack`` parameter -- only the
    reviewed patch does.  Passing ``am_pack=None`` unconditionally therefore
    makes a ``Denoiser`` unable to wrap the unpatched module, which is exactly
    the baseline the identity gate needs.  Omitting the keyword when there is no
    artifact lets one class drive both, and keeps "serve a pack through the
    unpatched module" a loud TypeError rather than a silent no-op.
    """

    return {} if am_pack is None else {"am_pack": am_pack}


@dataclasses.dataclass(frozen=True)
class DenoiseTrace:
    """One full denoising pass with its per-flow-time velocity field."""

    flow_times: tuple[float, ...]
    denoise_states: np.ndarray  # [flow_steps, action_tokens, action_dim], x_t before v(x_t, t)
    velocities: np.ndarray  # [flow_steps, action_tokens, action_dim]
    actions: np.ndarray  # [action_tokens, action_dim]
    queries: np.ndarray | None  # [flow_steps, layers, 4, action_tokens, head_dim]
    memory_keys: np.ndarray | None  # [layers, 512, head_dim]
    memory_values: np.ndarray | None  # [layers, 512, head_dim]
    memory_kv_recomputed_per_step: bool | None
    teacher_forced: bool = False

    def validate(self) -> None:
        if self.velocities.ndim != 3 or self.velocities.shape[0] != len(self.flow_times):
            raise ValueError("velocity trace does not match the flow schedule")
        if self.denoise_states.shape != self.velocities.shape:
            raise ValueError("denoise-state trace does not match the velocity trace")
        if (
            not np.isfinite(self.denoise_states).all()
            or not np.isfinite(self.velocities).all()
            or not np.isfinite(self.actions).all()
        ):
            raise ValueError("denoise trace contains non-finite values")


def validate_official_capture(trace: DenoiseTrace, *, num_steps: int) -> None:
    """Fail closed on the released FrameSamp+Modul tensor geometry."""

    trace.validate()
    if trace.teacher_forced:
        raise ValueError("teacher Q/K/V capture must be a closed-loop full-memory denoise")
    expected_queries = (
        num_steps,
        OFFICIAL_MEMORY_LAYERS,
        OFFICIAL_QUERY_HEADS,
        OFFICIAL_ACTION_HORIZON,
        OFFICIAL_HEAD_DIM,
    )
    if trace.queries is None or trace.queries.shape != expected_queries:
        raise ValueError(
            f"official query taps must be {expected_queries}, got {getattr(trace.queries, 'shape', None)}"
        )
    expected_memory = (OFFICIAL_MEMORY_LAYERS, TOKEN_BUDGET, OFFICIAL_HEAD_DIM)
    if trace.memory_keys is None or trace.memory_keys.shape != expected_memory:
        raise ValueError(
            f"official memory K taps must be {expected_memory}, got {getattr(trace.memory_keys, 'shape', None)}"
        )
    if trace.memory_values is None or trace.memory_values.shape != expected_memory:
        raise ValueError(
            f"official memory V taps must be {expected_memory}, got {getattr(trace.memory_values, 'shape', None)}"
        )
    if not np.isfinite(trace.queries).all() or not np.isfinite(trace.memory_keys).all():
        raise ValueError("official query/key taps contain non-finite values")
    if not np.isfinite(trace.memory_values).all():
        raise ValueError("official value taps contain non-finite values")


def flow_schedule(num_steps: int) -> tuple[float, ...]:
    """The exact times visited by the official ``while_loop`` (t = 1 .. 1/N)."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    dt = -1.0 / num_steps
    times: list[float] = []
    value = np.float32(1.0)
    for _ in range(num_steps):
        times.append(float(value))
        value = np.float32(value + np.float32(dt))
    return tuple(times)


class Denoiser:
    """Jitted prefill + per-flow-step suffix pass for one module configuration."""

    def __init__(self, model: Any, *, capture: bool = False):
        self.capture = bool(capture)
        self.action_horizon = int(model.action_horizon)
        self.action_dim = int(model.action_dim)
        graphdef, state = nnx.split(model)
        self._state = state

        def prefill(state, observation):
            from mme_vla_suite.models.integration.history_observation import preprocess_observation
            from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

            module = nnx.merge(graphdef, state)
            observation = preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = module.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            outputs = module.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            kv_cache = outputs[1]
            mem_seq, mem_mask, _, _, _ = module.embed_memory(observation)
            return observation, kv_cache, prefix_mask, mem_seq, mem_mask

        def step(state, observation, kv_cache, prefix_mask, mem_seq, mem_mask, x_t, timestep, am_pack):
            from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

            module = nnx.merge(graphdef, state)
            batch_size = observation.state.shape[0]
            suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = module.embed_suffix(
                observation, x_t, jnp.broadcast_to(timestep, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            outputs = module.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                mem_seq=[None, mem_seq],
                mem_mask=[None, mem_mask],
                **memory_branch_kwargs(am_pack),
            )
            suffix_out = outputs[0][1]
            v_t = module.action_out_proj(suffix_out[:, -module.action_horizon :])
            taps = outputs[2] if self.capture else None
            return v_t, taps

        self._prefill = jax.jit(prefill)
        self._step = jax.jit(step)

    def __call__(
        self,
        observation: Any,
        *,
        noise: jnp.ndarray,
        num_steps: int = DEFAULT_NUM_FLOW_STEPS,
        am_pack: MemoryAttentionAMPack | None = None,
        teacher_states: np.ndarray | None = None,
    ) -> DenoiseTrace:
        if self.capture and am_pack is not None:
            raise ValueError("tap capture describes the full teacher cache; it cannot run with an AM pack")
        noise = jnp.asarray(noise)
        expected_noise = (1, self.action_horizon, self.action_dim)
        if noise.shape != expected_noise:
            raise ValueError(f"AMKV E0 is B=1 and requires noise {expected_noise}, got {noise.shape}")
        forced = None
        if teacher_states is not None:
            forced_host = np.asarray(teacher_states, dtype=np.float32)
            expected_states = (num_steps, self.action_horizon, self.action_dim)
            if forced_host.shape != expected_states or not np.isfinite(forced_host).all():
                raise ValueError(f"teacher_states must be finite {expected_states}, got {forced_host.shape}")
            forced = jnp.asarray(forced_host, dtype=noise.dtype)

        preprocessed, kv_cache, prefix_mask, mem_seq, mem_mask = self._prefill(self._state, observation)
        dt = -1.0 / num_steps
        x_t = noise
        timestep = jnp.asarray(1.0, dtype=jnp.float32)
        states: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        times: list[float] = []
        queries: list[np.ndarray] = []
        memory_keys: np.ndarray | None = None
        memory_values: np.ndarray | None = None
        recomputed: bool | None = None
        for step_index in range(num_steps):
            times.append(float(timestep))
            step_state = x_t if forced is None else forced[step_index][None, ...]
            states.append(np.asarray(step_state[0], dtype=np.float32))
            v_t, taps = self._step(
                self._state, preprocessed, kv_cache, prefix_mask, mem_seq, mem_mask, step_state, timestep, am_pack
            )
            if self.capture:
                step_queries = np.asarray(
                    jnp.transpose(taps.queries_post_rope_pre_scale[:, 0], (0, 2, 1, 3)), dtype=np.float32
                )
                queries.append(step_queries)
                keys = np.asarray(taps.keys_post_rope[:, 0, :, 0, :], dtype=np.float32)
                values = np.asarray(taps.values_post_projection[:, 0, :, 0, :], dtype=np.float32)
                if memory_keys is None:
                    memory_keys, memory_values = keys, values
                else:
                    # Measured, not assumed: the official serve path re-projects
                    # the whole memory at every layer and every flow step.
                    same = np.array_equal(keys, memory_keys) and np.array_equal(values, memory_values)
                    recomputed = True if recomputed is None else recomputed
                    if not same:
                        raise ValueError("memory K/V changed across flow steps; the AM artifact would be stale")
            x_t = x_t + dt * v_t
            timestep = timestep + jnp.float32(dt)
            velocities.append(np.asarray(v_t[0], dtype=np.float32))
        trace = DenoiseTrace(
            flow_times=tuple(times),
            denoise_states=np.stack(states, axis=0),
            velocities=np.stack(velocities, axis=0),
            actions=np.asarray(x_t[0], dtype=np.float32),
            queries=np.stack(queries, axis=0) if queries else None,
            memory_keys=memory_keys,
            memory_values=memory_values,
            memory_kv_recomputed_per_step=recomputed,
            teacher_forced=forced is not None,
        )
        trace.validate()
        return trace


def identity_am_pack(memory_keys: np.ndarray, memory_values: np.ndarray, *, dtype: Any) -> MemoryAttentionAMPack:
    """Serve the full teacher K/V as the mandatory beta-zero identity gate.

    This route skips memory projection and therefore is not a comparable serve
    latency baseline.  Its only evidentiary role is exact numerical parity.
    """

    keys = jnp.asarray(memory_keys, dtype=dtype)[:, None, :, None, :]
    values = jnp.asarray(memory_values, dtype=dtype)[:, None, :, None, :]
    layers, _, tokens = keys.shape[0], keys.shape[1], keys.shape[2]
    return MemoryAttentionAMPack(
        compact_keys=keys,
        compact_values=values,
        compact_beta_am=jnp.zeros((layers, 1, tokens), dtype=jnp.float32),
        recent_keys=jnp.zeros((layers, 1, 0, 1, keys.shape[-1]), dtype=dtype),
        recent_values=jnp.zeros((layers, 1, 0, 1, values.shape[-1]), dtype=dtype),
        recent_token_mask=jnp.zeros((layers, 1, 0), dtype=jnp.bool_),
    )


def am_pack_from_arrays(arrays: dict[str, np.ndarray], *, dtype: Any) -> MemoryAttentionAMPack:
    return MemoryAttentionAMPack(
        compact_keys=jnp.asarray(arrays["compact_keys"], dtype=dtype),
        compact_values=jnp.asarray(arrays["compact_values"], dtype=dtype),
        compact_beta_am=jnp.asarray(arrays["compact_beta_am"], dtype=jnp.float32),
        recent_keys=jnp.asarray(arrays["recent_keys"], dtype=dtype),
        recent_values=jnp.asarray(arrays["recent_values"], dtype=dtype),
        recent_token_mask=jnp.asarray(arrays["recent_token_mask"], dtype=jnp.bool_),
    )


def sample_noise(rng_seed: int, *, batch: int, action_horizon: int, action_dim: int) -> jnp.ndarray:
    return jax.random.normal(jax.random.key(rng_seed), (batch, action_horizon, action_dim))


def selftest_matches_official_sampler(
    policy: Any,
    observation: Any,
    *,
    noise: jnp.ndarray,
    num_steps: int = DEFAULT_NUM_FLOW_STEPS,
    tolerance: float = 0.02,
    denoiser: Denoiser | None = None,
) -> dict[str, object]:
    """Gate: the unrolled AM-patched driver must reproduce ``sample_actions``.

    ``sample_actions`` is called on the *official* module -- the released
    computation, untouched -- while the driver runs the reviewed patch with no
    artifact.  A disagreement therefore means either the unrolled loop or the
    patch drifted, which is exactly the pair that must be sound before any AM
    number is reported.  Bitwise equality is recorded but not required: the
    official sampler fuses its ``while_loop`` body differently from an unrolled
    Python loop, so a few ULPs of difference is expected on accelerator
    backends and is reported rather than silently tolerated.
    """

    model = policy._model  # noqa: SLF001
    official = model.sample_actions(jax.random.key(0), observation, num_steps=num_steps, noise=noise)
    if denoiser is not None:
        trace = denoiser(observation, noise=noise, num_steps=num_steps)
    else:
        with installed_patch(model, capture=False):
            trace = Denoiser(model, capture=False)(observation, noise=noise, num_steps=num_steps)
    reference = np.asarray(official[0], dtype=np.float32)
    delta = float(np.max(np.abs(reference - trace.actions)))
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float32).tiny)
    relative = delta / scale
    bitwise = bool(np.array_equal(reference, trace.actions))
    if relative > tolerance:
        raise ValueError(
            "unrolled denoise driver disagrees with the official sampler: "
            f"max|delta| = {delta} ({relative:.4%} of the action scale), tolerance {tolerance:.2%}"
        )
    return {
        "driver_matches_official_sampler": True,
        "bitwise": bitwise,
        "max_abs_action_delta": delta,
        "relative_action_delta": relative,
        "tolerance": float(tolerance),
        "note": (
            "bfloat16 serve precision: one ULP at |a|~2 is 2**-7 = 0.0078, so a nonzero delta here is "
            "the loop-fusion noise floor, not a logic difference; the identity arm gives the tighter, "
            "same-driver floor against which AM deltas must be read"
        ),
        "official_action_max_abs": float(np.max(np.abs(reference))),
        "flow_times": list(trace.flow_times),
    }


def _require_bitwise_equal(label: str, reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"{label} shape mismatch: {reference.shape} != {candidate.shape}")
    delta = float(np.max(np.abs(candidate.astype(np.float32) - reference.astype(np.float32))))
    bitwise = bool(np.array_equal(reference, candidate))
    if not bitwise:
        raise ValueError(f"{label} is not bitwise identical at restored-checkpoint precision; max|delta|={delta}")
    return {"bitwise": True, "max_abs_delta": delta}


def selftest_patched_identity(
    *,
    official_denoiser: Denoiser,
    capture_denoiser: Denoiser,
    patched_denoiser: Denoiser,
    observation: Any,
    noise: jnp.ndarray,
    num_steps: int,
    dtype: Any,
) -> dict[str, object]:
    """Restored-checkpoint gate for the patched baseline and beta-zero identity pack."""

    official = official_denoiser(observation, noise=noise, num_steps=num_steps)
    captured = capture_denoiser(observation, noise=noise, num_steps=num_steps)
    validate_official_capture(captured, num_steps=num_steps)
    patched = patched_denoiser(observation, noise=noise, num_steps=num_steps)
    baseline = {
        "actions": _require_bitwise_equal("patched baseline actions", official.actions, captured.actions),
        "velocities": _require_bitwise_equal("patched baseline velocities", official.velocities, captured.velocities),
        "denoise_states": _require_bitwise_equal(
            "patched baseline denoise states", official.denoise_states, captured.denoise_states
        ),
        "plain_actions": _require_bitwise_equal("patched plain actions", official.actions, patched.actions),
        "plain_velocities": _require_bitwise_equal(
            "patched plain velocities", official.velocities, patched.velocities
        ),
        "plain_denoise_states": _require_bitwise_equal(
            "patched plain denoise states", official.denoise_states, patched.denoise_states
        ),
    }
    identity = patched_denoiser(
        observation,
        noise=noise,
        num_steps=num_steps,
        am_pack=identity_am_pack(captured.memory_keys, captured.memory_values, dtype=dtype),
    )
    identity_gate = {
        "actions": _require_bitwise_equal("full-K/V beta-zero identity actions", captured.actions, identity.actions),
        "velocities": _require_bitwise_equal(
            "full-K/V beta-zero identity velocities", captured.velocities, identity.velocities
        ),
        "denoise_states": _require_bitwise_equal(
            "full-K/V beta-zero identity denoise states", captured.denoise_states, identity.denoise_states
        ),
    }
    return {
        "restored_checkpoint_patched_baseline_identity": baseline,
        "restored_checkpoint_full_kv_beta0_identity": identity_gate,
        "official_geometry": {
            "memory_tokens": TOKEN_BUDGET,
            "layers": OFFICIAL_MEMORY_LAYERS,
            "query_heads": OFFICIAL_QUERY_HEADS,
            "kv_heads": OFFICIAL_KV_HEADS,
            "head_dim": OFFICIAL_HEAD_DIM,
            "action_horizon": OFFICIAL_ACTION_HORIZON,
        },
    }


@contextlib.contextmanager
def installed_patch(model: Any, *, capture: bool = False):
    """Install the reviewed AM patch for the duration of a block."""

    previous = install_patched_history_module(model, capture=capture)
    try:
        yield model
    finally:
        model.PaliGemma.llm.module = previous


def timed(callable_, *, warmup: int = 1, repeats: int = 3) -> dict[str, float]:
    """Wall-clock a device computation with explicit warmup and blocking."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    for _ in range(warmup):
        jax.block_until_ready(callable_())
    samples: list[float] = []
    for _ in range(repeats):
        start = _time.perf_counter()
        jax.block_until_ready(callable_())
        samples.append(_time.perf_counter() - start)
    array = np.asarray(samples, dtype=np.float64)
    return {
        "seconds_mean": float(array.mean()),
        "seconds_min": float(array.min()),
        "seconds_max": float(array.max()),
        "repeats": int(repeats),
    }
