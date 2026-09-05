"""Demo-capable RoboTTT extensions for the isolated RoboMME process.

The completed execution-only Q2 artifact remains reproducible.  This module is installed only for
the new ``q2v`` arm and adds the missing video-ICL semantics:

* front-view SigLIP tokens are spatially pooled and read by learned register cross-attention;
* demonstration observations update ``W`` without requiring action labels;
* execution observations condition the action expert and commit the generated/teacher action chunk;
* the demonstration prefix is loss-masked, while gradients flow through its fast-weight updates.

Nothing here changes VLM/prefix tokens.  The contextual registers are a read-only side tap and the
result still reaches Pi through the existing near-zero tanh-gated ``adarms_cond`` seam.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

_DEMO_CLASS: type | None = None


def encode_context_tokens(model, observation):
    """Encode one observation batch into a compact, read-only visual-language context.

    We reuse the pretrained front-camera SigLIP tokens, pool the 16x16 patch grid to 4x4, and append
    one masked-mean language token.  ``stop_gradient`` avoids retaining a second SigLIP backward
    graph for every demonstration frame; the ordinary execution loss still full-finetunes Pi.
    """
    import einops
    import jax
    import jax.numpy as jnp

    if "base_0_rgb" not in observation.images:
        raise ValueError("demo-capable RoboTTT requires base_0_rgb")
    vision, _ = model.PaliGemma.img(
        observation.images["base_0_rgb"],
        train=False,
    )
    token_count = int(vision.shape[1])
    side = math.isqrt(token_count)
    if side * side != token_count or side % 4:
        raise ValueError(f"expected a square SigLIP patch grid divisible by 4, got {token_count} tokens")
    block = side // 4
    vision = vision.reshape(vision.shape[0], 4, block, 4, block, vision.shape[-1])
    vision = vision.mean(axis=(2, 4)).reshape(vision.shape[0], 16, vision.shape[-1])
    vision_mask = einops.repeat(
        observation.image_masks["base_0_rgb"],
        "b -> b n",
        n=16,
    )

    tokens = [vision]
    masks = [vision_mask]
    if observation.tokenized_prompt is not None:
        if observation.tokenized_prompt_mask is None:
            raise ValueError("tokenized prompt and mask must be supplied together")
        language = model.PaliGemma.llm(
            observation.tokenized_prompt,
            method="embed",
        )
        language_mask = observation.tokenized_prompt_mask
        weights = language_mask.astype(language.dtype)[..., None]
        pooled = jnp.sum(language * weights, axis=1, keepdims=True)
        pooled = pooled / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1)
        tokens.append(pooled)
        masks.append(jnp.any(language_mask, axis=1, keepdims=True))

    return (
        jax.lax.stop_gradient(jnp.concatenate(tokens, axis=1)),
        jnp.concatenate(masks, axis=1),
    )


def install_demo_robottt_patch() -> type:
    """Replace the OpenPI fast-weight class in this process only and return the subclass."""
    global _DEMO_CLASS
    if _DEMO_CLASS is not None:
        return _DEMO_CLASS

    import jax
    import jax.numpy as jnp
    import openpi.models.robottt_fast_weights as _base
    from flax import nnx

    class DemoRoboTTTFastWeights(_base.RoboTTTFastWeights):
        """RoboTTT fast weights whose registers read current visual-language tokens."""

        context_dim = 2048  # Pi0.5 PaliGemma/SigLIP embedding width.

        def __init__(self, cfg, *, rngs):
            super().__init__(cfg, rngs=rngs)
            d = cfg.token_dim
            self.context_in = nnx.Linear(self.context_dim, d, rngs=rngs)
            self.context_q = nnx.Linear(d, d, rngs=rngs)
            self.context_k = nnx.Linear(d, d, rngs=rngs)
            self.context_v = nnx.Linear(d, d, rngs=rngs)
            self.context_out = nnx.Linear(d, d, rngs=rngs)

        def contextual_registers(self, context_tokens, context_token_mask):
            """Cross-attend learned registers to a read-only VLM side tap."""
            if context_tokens.shape[-1] != self.context_dim:
                raise ValueError(f"context width {context_tokens.shape[-1]} != expected {self.context_dim}")
            if context_token_mask.shape != context_tokens.shape[:2]:
                raise ValueError(f"context mask {context_token_mask.shape} != {context_tokens.shape[:2]}")
            batch = context_tokens.shape[0]
            registers = jnp.broadcast_to(
                self.registers[...],
                (batch, self.cfg.num_registers, self.cfg.token_dim),
            )
            context = self.context_in(context_tokens)
            q = self.context_q(registers)
            k = self.context_k(context)
            v = self.context_v(context)
            logits = jnp.einsum("bnd,bmd->bnm", q, k) / math.sqrt(self.cfg.token_dim)
            logits = jnp.where(
                context_token_mask[:, None, :],
                logits,
                jnp.finfo(logits.dtype).min,
            )
            attended = jnp.einsum(
                "bnm,bmd->bnd",
                jax.nn.softmax(logits, axis=-1),
                v,
            )
            return registers + self.context_out(attended)

        def _visual_tokens(self, state, actions, context_tokens, context_token_mask):
            registers = self.contextual_registers(context_tokens, context_token_mask)
            tokens = [registers, self.state_in(state)[:, None, :]]
            if actions is not None:
                tokens.append(self.action_in(actions))
            return jnp.concatenate(tokens, axis=1)

        def condition_visual(self, w, state, context_tokens, context_token_mask):
            tokens = self._visual_tokens(
                state,
                None,
                context_tokens,
                context_token_mask,
            )
            q = self.proj_q(tokens)
            pooled = jax.vmap(self._apply_single)(w, q)
            projected = self.readout(pooled)
            gate = jnp.tanh(self.alpha[...]) * self.cfg.alpha_scale
            return gate.astype(projected.dtype) * projected

        def commit_context(self, w, state, context_tokens, context_token_mask):
            """Observation-only update used by the demonstration prefix."""
            tokens = self._visual_tokens(
                state,
                None,
                context_tokens,
                context_token_mask,
            )
            k = self.proj_k(tokens)
            v = self.proj_v(tokens)
            eta = self.inner_lr()
            return jax.vmap(lambda wi, ki, vi: self._commit_single(wi, ki, vi, eta))(w, k, v)

        def commit_visual(
            self,
            w,
            state,
            actions,
            context_tokens,
            context_token_mask,
        ):
            """Execution update from current visual context plus the finalized action chunk."""
            tokens = self._visual_tokens(
                state,
                actions,
                context_tokens,
                context_token_mask,
            )
            k = self.proj_k(tokens)
            v = self.proj_v(tokens)
            eta = self.inner_lr()
            return jax.vmap(lambda wi, ki, vi: self._commit_single(wi, ki, vi, eta))(w, k, v)

        @staticmethod
        def _select_rows(mask, selected, fallback):
            def select_leaf(new, old):
                shape = (mask.shape[0],) + (1,) * (new.ndim - 1)
                return jnp.where(mask.reshape(shape), new, old)

            return jax.tree.map(select_leaf, selected, fallback)

        def run_demo_sequence(
            self,
            state_seq,
            action_seq,
            context_tokens,
            context_token_mask,
            demo_step_mask,
            execution_step_mask,
            *,
            commit_action_steps=None,
            tbptt_segment=None,
        ):
            """Thread ``W`` across loss-masked demo context and execution decisions.

            All arrays are batch-major.  Returned conditions are time-major ``[T,B,C]`` and are
            nonzero only at execution slots.  Padding slots update nothing.
            """
            if state_seq.shape[:2] != action_seq.shape[:2]:
                raise ValueError("state/action sequence geometry differs")
            if context_tokens.shape[:2] != state_seq.shape[:2]:
                raise ValueError("context/state sequence geometry differs")
            if context_token_mask.shape[:2] != state_seq.shape[:2]:
                raise ValueError("context-token mask sequence geometry differs")
            if demo_step_mask.shape != state_seq.shape[:2]:
                raise ValueError("demo step mask sequence geometry differs")
            if execution_step_mask.shape != state_seq.shape[:2]:
                raise ValueError("execution step mask sequence geometry differs")
            commit_steps = int(action_seq.shape[-2] if commit_action_steps is None else commit_action_steps)
            if not 1 <= commit_steps <= action_seq.shape[-2]:
                raise ValueError(
                    "commit_action_steps must lie in [1, action horizon], got "
                    f"{commit_steps} for horizon {action_seq.shape[-2]}"
                )

            segment = int(tbptt_segment if tbptt_segment is not None else self.cfg.tbptt_segment)
            if segment < 1:
                raise ValueError("tbptt_segment must be positive")
            w0 = self.init_state(state_seq.shape[0])

            def body(carry, values):
                w, time_index = carry
                state_t, action_t, tokens_t, token_mask_t, demo_t, execution_t = values
                w = jax.lax.cond(
                    jnp.logical_and(
                        time_index > 0,
                        (time_index % segment) == 0,
                    ),
                    lambda tree: jax.tree.map(jax.lax.stop_gradient, tree),
                    lambda tree: tree,
                    w,
                )
                condition = self.condition_visual(
                    w,
                    state_t,
                    tokens_t,
                    token_mask_t,
                )
                condition = jnp.where(execution_t[:, None], condition, 0)

                from_context = self.commit_context(
                    w,
                    state_t,
                    tokens_t,
                    token_mask_t,
                )
                from_execution = self.commit_visual(
                    w,
                    state_t,
                    # Only the block completed before the next replan may enter memory.  With
                    # horizon=20 and stride=10, committing all 20 teacher actions would expose the
                    # next decision to ten overlapping target actions.
                    action_t[:, :commit_steps],
                    tokens_t,
                    token_mask_t,
                )
                w_next = self._select_rows(demo_t, from_context, w)
                w_next = self._select_rows(execution_t, from_execution, w_next)
                return (w_next, time_index + 1), condition

            sequence = (
                jnp.swapaxes(state_seq, 0, 1),
                jnp.swapaxes(action_seq, 0, 1),
                jnp.swapaxes(context_tokens, 0, 1),
                jnp.swapaxes(context_token_mask, 0, 1),
                jnp.swapaxes(demo_step_mask, 0, 1),
                jnp.swapaxes(execution_step_mask, 0, 1),
            )
            (w_final, _), conditions = jax.lax.scan(
                body,
                (w0, jnp.asarray(0)),
                sequence,
            )
            return conditions, w_final

    _base.RoboTTTFastWeights = DemoRoboTTTFastWeights
    # Bound-method form lets serving freeze/JIT the full Pi model with OpenPI's low-overhead
    # ``module_jit`` helper.  It is a read-only side tap and adds no Pi parameters.
    import openpi.models.pi0 as _pi0

    if not hasattr(_pi0.Pi0, "robottt_context_tokens"):

        def robottt_context_tokens(self, observation):
            return encode_context_tokens(self, observation)

        _pi0.Pi0.robottt_context_tokens = robottt_context_tokens
    _DEMO_CLASS = DemoRoboTTTFastWeights
    return DemoRoboTTTFastWeights


def demo_stage_q_train_step(config, rng, state, batch):
    """JIT-able optimizer step for fixed ``[demo prefix, execution]`` sequences."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    model = nnx.merge(state.model_def, state.params)
    model.train()

    observation, actions, loss_mask, context_mask = batch
    demo_slots = int(config.data.stage_q_demo_frames)
    execution_slots = int(config.data.stage_q_window_len)
    expected_length = demo_slots + execution_slots
    if actions.shape[1] != expected_length:
        raise ValueError(f"demo sequence length {actions.shape[1]} != {expected_length}")
    if loss_mask.shape != actions.shape[:2] or context_mask.shape != actions.shape[:2]:
        raise ValueError("demo sequence-role masks do not match actions")

    def loss_fn(model, step_rng, obs, target_actions, action_loss_mask, demo_mask):
        batch_size, length = target_actions.shape[:2]
        execution_mask = action_loss_mask > 0

        if getattr(model, "robottt", False):
            obs_flat = jax.tree.map(
                lambda value: value.reshape(
                    batch_size * length,
                    *value.shape[2:],
                ),
                obs,
            )
            context_tokens, context_token_mask = encode_context_tokens(
                model,
                obs_flat,
            )
            context_tokens = context_tokens.reshape(
                batch_size,
                length,
                *context_tokens.shape[1:],
            )
            context_token_mask = context_token_mask.reshape(
                batch_size,
                length,
                *context_token_mask.shape[1:],
            )
            state_seq = obs_flat.state.reshape(batch_size, length, -1)
            conditions, _ = model.robottt_fast.run_demo_sequence(
                state_seq,
                target_actions,
                context_tokens,
                context_token_mask,
                demo_mask,
                execution_mask,
                commit_action_steps=config.data.stage_q_chunk_stride,
            )
            conditions = jnp.swapaxes(conditions, 0, 1)
        else:
            conditions = None

        # Execution positions are a fixed suffix, so no dynamic boolean indexing enters the JIT.
        obs_exec = jax.tree.map(
            lambda value: value[:, demo_slots : demo_slots + execution_slots],
            obs,
        )
        actions_exec = target_actions[
            :,
            demo_slots : demo_slots + execution_slots,
        ]
        obs_exec = jax.tree.map(
            lambda value: value.reshape(
                batch_size * execution_slots,
                *value.shape[2:],
            ),
            obs_exec,
        )
        actions_exec = actions_exec.reshape(
            batch_size * execution_slots,
            *actions_exec.shape[2:],
        )
        if conditions is not None:
            conditions_exec = conditions[
                :,
                demo_slots : demo_slots + execution_slots,
            ].reshape(batch_size * execution_slots, -1)
            obs_exec = dataclasses.replace(
                obs_exec,
                robottt_cond=conditions_exec,
            )
        losses = model.compute_loss(
            step_rng,
            obs_exec,
            actions_exec,
            train=True,
        )
        return jnp.mean(losses)

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model,
        train_rng,
        observation,
        actions,
        loss_mask,
        context_mask,
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }


def demo_patch_metadata() -> dict[str, Any]:
    """Stable scientific identity fields for manifests and logs."""
    return {
        "context_source": "front_siglip_16x16_meanpool_to_4x4_plus_language_mean",
        "context_backbone_gradient": "stop_gradient",
        "register_contextualizer": "single_head_cross_attention",
        "demo_update": "observation_only_commit",
        "execution_update": "visual_state_completed_action_block_commit",
        "execution_commit_steps": 10,
        "policy_integration": "tanh_gated_single_adarms_cond",
        "vlm_prefix_mutation": False,
    }
