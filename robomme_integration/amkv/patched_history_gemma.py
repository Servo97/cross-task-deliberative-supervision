"""Reviewed AM patch of the official scanned ``MemoryAttention`` stack.

This is a *vendored derivative* of
``mme_vla_suite/models/integration/history_gemma.py`` at policy commit
``ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`` (base SHA256 pinned in
:mod:`robomme_integration.training.framesamp_am_jax`).  The official file is
never edited: a live reference evaluation runs against it.  Instead this copy
implements exactly the five patch points named by
``OfficialMemoryAttentionAMPatchContract``:

1. tap Q immediately after RoPE and *before* the single ``head_dim**-0.5``
   scale;
2. tap memory V immediately after ``kv_einsum_mem`` and memory K immediately
   after RoPE at the tokens' original ``0..511`` logical positions;
3. thread the compact K/V/beta arrays through ``nn.scan`` with ``in_axes=0`` so
   layer ``l`` can only consume layer ``l``'s artifact;
4. consume compact K without projection, RMSNorm, or a second RoPE, giving
   ``beta=0`` to any exact recent tokens and sharing one softmax denominator
   (delegated to
   :func:`robomme_integration.training.framesamp_am_jax.memory_attention_am_core`);
5. generate action-query RoPE positions from the fixed logical offset
   ``mem_len`` (512), never from the compact length ``M`` or recent length
   ``R``.

Everything else -- parameter names, initializers, dtypes, mask constant,
einsum order, ``out_einsum_mem`` and the modulation path -- is byte-for-byte
the official computation, so the released checkpoint loads unchanged and the
``am_pack=None`` path is the unpatched forward.

Two deliberate deviations from the official file, both documented rather than
hidden:

* ``@at.typecheck`` is dropped from the patched classes.  jaxtyping's global
  axis names collide across the tap/return shapes and the decorator is a debug
  aid, not semantics; explicit shape assertions replace it.
* ``capture`` is a *static* module field, not a runtime argument, so no new
  entry is needed in ``nn.scan``'s ``static_argnums``.  With ``capture=False``
  the return signature is the official ``(outputs, kv_cache)`` pair and
  ``HistoryPi0.sample_actions`` still runs; with ``capture=True`` the call
  returns a third element carrying per-layer taps stacked by the scan.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from mme_vla_suite.models.integration.utils import Attention_with_MemoryExpert, _name
from mme_vla_suite.models.representation.utils import kernel_init_out_proj
from openpi.models.gemma import (
    PALIGEMMA_VOCAB_SIZE,
    Attention,
    Config,
    Embedder,
    RMSNorm,
    _apply_rope,
    _gated_residual,
)

from robomme_integration.training.framesamp_am_jax import memory_attention_am_core

# Official hardcoded MemoryAttention geometry.  Mirrored here so the patch can
# assert it instead of silently adapting to a different head layout.
MEMORY_NUM_HEADS = 4
MEMORY_NUM_KV_HEADS = 1
MEMORY_HEAD_DIM = 256
MEMORY_WIDTH = 1024
OFFICIAL_MASK_FILL = -2.3819763e38


@dataclasses.dataclass(frozen=True)
class MemoryAttentionAMPack:
    """One layer's compact-old plus exact-recent memory substitution.

    ``compact_*`` arrays are already projected and RoPE-applied at their
    original logical positions; they must not pass through ``mem_rms_norm``,
    ``kv_einsum_mem``, or ``_apply_rope`` again.  ``recent_*`` arrays are raw
    teacher K/V for the tokens kept exact and receive ``beta_am = 0``.
    """

    compact_keys: jax.Array  # [B, M, 1, H]
    compact_values: jax.Array  # [B, M, 1, H]
    compact_beta_am: jax.Array  # [B, M] float32
    recent_keys: jax.Array  # [B, R, 1, H]
    recent_values: jax.Array  # [B, R, 1, H]
    recent_token_mask: jax.Array  # [B, R] bool

    def tree_flatten(self):
        return (
            (
                self.compact_keys,
                self.compact_values,
                self.compact_beta_am,
                self.recent_keys,
                self.recent_values,
                self.recent_token_mask,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


jax.tree_util.register_pytree_node(
    MemoryAttentionAMPack,
    MemoryAttentionAMPack.tree_flatten,
    MemoryAttentionAMPack.tree_unflatten,
)


@dataclasses.dataclass(frozen=True)
class MemoryAttentionTaps:
    """Serve-precision Q/K/V taps for one layer at one forward call."""

    queries_post_rope_pre_scale: jax.Array  # [B, T, 4, H]
    keys_post_rope: jax.Array  # [B, S, 1, H]
    values_post_projection: jax.Array  # [B, S, 1, H]

    def tree_flatten(self):
        return (
            (
                self.queries_post_rope_pre_scale,
                self.keys_post_rope,
                self.values_post_projection,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


jax.tree_util.register_pytree_node(
    MemoryAttentionTaps,
    MemoryAttentionTaps.tree_flatten,
    MemoryAttentionTaps.tree_unflatten,
)


class MemoryRMSNorm(nn.Module):
    """Unmodified official memory RMSNorm (copied verbatim)."""

    @nn.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype)

        modulation = nn.Dense(x.shape[-1] * 2, kernel_init=kernel_init_out_proj, dtype=dtype)(cond)
        scale, shift = jnp.split(modulation, 2, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype)


class MemoryAttentionAM(nn.Module):
    """Official memory cross-attention with an AM tap/substitution seam.

    ``am_pack is None`` reproduces the official computation exactly.  With a
    pack, the projection/RoPE of memory K/V is *skipped entirely* (that is the
    measured serve saving: the official path re-projects all 512 memory tokens
    at every layer and every flow step) and compact+recent share one softmax.
    """

    capture: bool = False

    @nn.compact
    def __call__(self, x, mem_seq, mem_mask, am_pack: MemoryAttentionAMPack | None = None):
        # x: [B, T, D], mem_seq: [B, S, D], mem_mask: [B, S]
        B, mem_len, mem_width = mem_seq.shape
        B, x_len, x_width = x.shape
        num_heads, num_kv_heads, head_dim, width = (
            MEMORY_NUM_HEADS,
            MEMORY_NUM_KV_HEADS,
            MEMORY_HEAD_DIM,
            MEMORY_WIDTH,
        )
        assert mem_width == x_width == width
        if self.capture and am_pack is not None:
            raise ValueError("tap capture describes the full teacher cache; it cannot run with an AM pack")
        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim),
            name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim),
            name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
        )
        rms_norm = MemoryRMSNorm(name="mem_rms_norm")
        x = rms_norm(x)
        q = q_einsum("BTD,NDH->BTNH", x)

        if am_pack is None:
            mem_seq = rms_norm(mem_seq)
            k, v = kv_einsum("BSD,2KDH->2BSKH", mem_seq)
            q_positions = einops.repeat(jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B)
            k_positions = einops.repeat(jnp.arange(mem_len), "t -> b t", b=B)
            # Preserve the released operation order exactly. Even though Q/K
            # RoPE and Q scaling are algebraically independent, reordering them
            # changes the compiled BF16 graph and can move the restored baseline
            # by one ULP.
            q = _apply_rope(q, positions=q_positions)
            # Patch point 1: query tap, post-RoPE and pre-scale.
            queries_post_rope_pre_scale = q
            q = q * head_dim**-0.5
            # Patch point 2: K is tapped after RoPE at logical positions
            # 0..mem_len-1; V is tapped straight after projection.
            k = _apply_rope(k, positions=k_positions)

            q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
            logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
            attn_mask = mem_mask[:, None, None, None, :]  # (B, 1, 1, 1, S)
            masked_logits = jnp.where(attn_mask, logits, OFFICIAL_MASK_FILL)
            probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
            encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
            taps = (
                MemoryAttentionTaps(
                    queries_post_rope_pre_scale=queries_post_rope_pre_scale,
                    keys_post_rope=k,
                    values_post_projection=v,
                )
                if self.capture
                else None
            )
        else:
            q_positions = einops.repeat(jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B)
            # Patch point 5: the action-query offset is the teacher's logical
            # memory span, never compact M or recent R.
            q = _apply_rope(q, positions=q_positions)
            queries_post_rope_pre_scale = q
            # Patch points 3 and 4: layer-local compact arrays consumed without
            # projection or RoPE, exact recent tokens with beta_am = 0, one
            # shared softmax denominator.
            encoded, _log_mass = memory_attention_am_core(
                queries_post_rope_pre_scale,
                am_pack.compact_keys,
                am_pack.compact_values,
                am_pack.compact_beta_am,
                am_pack.recent_keys,
                am_pack.recent_values,
                am_pack.recent_token_mask,
                scale=head_dim**-0.5,
                query_position_offset=mem_len,
            )
            taps = None

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width),
            name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded), taps


class HistoryBlockAM(nn.Module):
    """Official ``HistoryBlock`` with the AM pack and tap outputs threaded."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    integration_type: str | None = None
    capture: bool = False

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        mem_seq,
        mem_mask,
        deterministic=True,  # noqa: FBT002
        am_pack=None,
    ):
        if self.integration_type == "modulation":
            mem_attn = MemoryAttentionAM(name="mem_attn", capture=self.capture)

        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        if self.integration_type == "expert":
            attn = Attention_with_MemoryExpert(configs=self.configs, name="attn")
        else:
            attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                name = (
                    _name("pre_attention_norm", i)
                    if self.integration_type != "expert"
                    else _name("pre_attention_norm", i - 1)
                )
                x, gate = RMSNorm(name=name)(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        taps = None
        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # Add Memory Modulation before FFN
                if i == len(xs) - 1 and self.integration_type == "modulation":
                    mem_mod_vec, taps = mem_attn(x, mem_seq[-1], mem_mask[-1], am_pack)
                    x = MemoryRMSNorm(name="mem_rms_norm_ffn")(x, mem_mod_vec)

                name = _name("pre_ffw_norm", i) if self.integration_type != "expert" else _name("pre_ffw_norm", i - 1)
                x, gate = RMSNorm(name=name)(x, adarms_cond[i])  # noqa: PLW2901

                name = _name("mlp", i) if self.integration_type != "expert" else _name("mlp", i - 1)
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=name,
                    lora_config=config.lora_configs.get("ffn"),
                )(x)

            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        if self.capture:
            return xs, (kv_cache, taps)
        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


class ModuleAM(nn.Module):
    """Official history transformer with per-layer AM packs and taps.

    ``capture=False`` and ``am_pack=None`` is the unpatched forward, including
    its exact ``(outputs, kv_cache)`` return signature.
    """

    configs: Sequence[Config]
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    integration_type: str | None = None
    capture: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        embed_dim = self.configs[0].width if self.integration_type != "expert" else self.configs[1].width
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=embed_dim,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            HistoryBlockAM,
            prevent_cse=False,
            static_argnums=(7,),  # 0=xs, 7=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                0,
            ),  # 0=kv_cache, 1..6 broadcast, 7=am_pack (one artifact per layer)
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
            capture=self.capture,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i) if self.integration_type != "expert" else _name("final_norm", i - 1))
            for i in range(len(self.configs))
        ]

    def embed(self, tokens):
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def __call__(
        self,
        embedded,
        positions,
        mask,
        adarms_cond=None,
        *,
        kv_cache: KVCache | None = None,
        mem_seq=None,
        mem_mask=None,
        deterministic: bool = True,
        am_pack: Any = None,
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, scanned = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            mem_seq,
            mem_mask,
            deterministic,
            am_pack,
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        outputs = [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if self.capture:
            kv_cache, taps = scanned
            return outputs, kv_cache, taps
        return outputs, scanned

    def init(self, use_adarms: Sequence[bool], mem_mods: Sequence[bool]):
        """Convenience method for initializing all parameters (official signature)."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            mem_seq=[jnp.zeros((1, 4, c.width)) if m else None for c, m in zip(self.configs, mem_mods, strict=True)],
            mem_mask=[jnp.ones((1, 4), dtype=bool) if m else None for m in mem_mods],
        )


PATCHED_MODULE_FIELDS = (
    "configs",
    "embed_dtype",
    "dropout",
    "dropout_bdims",
    "adarms",
    "integration_type",
)


def patched_module_like(official_module, *, capture: bool = False) -> ModuleAM:
    """Build a :class:`ModuleAM` with an official ``Module``'s exact settings.

    The parameter tree is identical (no new parameters, identical ``name=``
    strings), so a released checkpoint restored for the official module can be
    applied through this one without any remapping.
    """

    values = {}
    for field in PATCHED_MODULE_FIELDS:
        if not hasattr(official_module, field):
            raise ValueError(f"official history module is missing field {field!r}; patch contract drifted")
        values[field] = getattr(official_module, field)
    return ModuleAM(capture=capture, **values)
