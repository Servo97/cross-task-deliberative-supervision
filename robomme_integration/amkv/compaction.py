"""Fit per-layer AM artifacts for the official FrameSamp memory and pack them.

Compression ratios are defined over the *whole* valid memory budget: at ratio
``r`` the policy is served ``round(valid / r)`` memory tokens in total.  With
``exact_recent_frames = f`` the newest ``f`` frames (16 tokens each) are kept
bit-exact and only the older tokens are compacted, so

    served = 16 * f  (exact)  +  M  (compact),   16 * f + M = round(valid / r)

and compact+recent share one softmax denominator at serve time.  ``f = 0`` is
the plain "compact all history" arm.

Nothing here trains: selection is RMS-highest-attention, the mass fit is NNLS
and the value fit is OLS, all on frozen teacher taps, all in float64, with no
gradient path anywhere.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from robomme_integration.training.attention_matching import (
    AttentionMatchingArtifact,
    AttentionMatchingMetrics,
    attend_compact_old_and_recent,
    scaled_dot_product_attention,
)
from robomme_integration.training.framesamp_attention_matching import (
    FrameSampAttentionMatchingResult,
    fit_framesamp_attention_matching,
)
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    TOKENS_PER_FRAME,
    FrameSampHistory,
)

COMPACTION_METHOD = "framesamp_history_am_v1"


@dataclasses.dataclass(frozen=True)
class CompactionPlan:
    """Token budget for one ratio, resolved against a concrete prefix."""

    ratio: float
    valid_tokens: int
    valid_frames: int
    exact_recent_frames: int
    compact_size: int
    recent_size: int

    @property
    def served_tokens(self) -> int:
        return self.compact_size + self.recent_size

    @property
    def effective_ratio(self) -> float:
        return self.valid_tokens / self.served_tokens

    def validate(self) -> None:
        if not np.isfinite(self.ratio) or self.ratio <= 1.0:
            raise ValueError(f"compression ratio must exceed 1, got {self.ratio}")
        if not 1 <= self.valid_frames <= MAX_FRAMES:
            raise ValueError("valid frame count outside the official FrameSamp range")
        if self.valid_tokens != self.valid_frames * TOKENS_PER_FRAME:
            raise ValueError("valid token count disagrees with the valid frame count")
        if not 0 <= self.exact_recent_frames < self.valid_frames:
            raise ValueError("exact recent frames must leave at least one frame to compact")
        if self.recent_size != self.exact_recent_frames * TOKENS_PER_FRAME:
            raise ValueError("recent token count disagrees with the exact recent frame count")
        if self.compact_size < 1:
            raise ValueError("compression ratio leaves no room for compact tokens after the exact recent block")
        if self.served_tokens >= self.valid_tokens:
            raise ValueError("compaction plan does not reduce the served memory")

    def label(self) -> dict[str, object]:
        self.validate()
        return {
            "method": COMPACTION_METHOD,
            "requested_ratio": float(self.ratio),
            "effective_ratio": float(self.effective_ratio),
            "valid_tokens": int(self.valid_tokens),
            "valid_frames": int(self.valid_frames),
            "exact_recent_frames": int(self.exact_recent_frames),
            "compact_tokens": int(self.compact_size),
            "recent_exact_tokens": int(self.recent_size),
            "served_tokens": int(self.served_tokens),
        }


def plan_compaction(history: FrameSampHistory, ratio: float, *, exact_recent_frames: int = 0) -> CompactionPlan:
    history.validate()
    valid_frames = int(history.frame_mask.sum())
    valid_tokens = valid_frames * TOKENS_PER_FRAME
    served = int(round(valid_tokens / float(ratio)))
    recent = int(exact_recent_frames) * TOKENS_PER_FRAME
    plan = CompactionPlan(
        ratio=float(ratio),
        valid_tokens=valid_tokens,
        valid_frames=valid_frames,
        exact_recent_frames=int(exact_recent_frames),
        compact_size=served - recent,
        recent_size=recent,
    )
    plan.validate()
    return plan


def old_history_view(history: FrameSampHistory, plan: CompactionPlan) -> FrameSampHistory:
    """Mask off the exact-recent frames, leaving the compactible prefix.

    This is a *fit-time view* of the memory, not a serve input: the recent
    frames are zeroed so the object still satisfies the official right-padded
    invariant, while the recent tokens themselves are served exactly from the
    untouched teacher taps.
    """

    history.validate()
    plan.validate()
    if plan.exact_recent_frames == 0:
        return history
    keep = plan.valid_frames - plan.exact_recent_frames
    frame_mask = np.zeros(MAX_FRAMES, dtype=np.bool_)
    frame_mask[:keep] = True
    token_mask = np.repeat(frame_mask, TOKENS_PER_FRAME)
    frame_indices = np.where(frame_mask, history.frame_indices, -1).astype(history.frame_indices.dtype, copy=False)
    image = np.where(token_mask[:, None], history.image, 0.0).astype(history.image.dtype, copy=False)
    position = np.where(token_mask[:, None], history.position, 0.0).astype(history.position.dtype, copy=False)
    view = FrameSampHistory(
        image=image,
        position=position,
        token_mask=token_mask,
        frame_indices=frame_indices,
        frame_mask=frame_mask,
    )
    view.validate()
    return view


def recent_token_slice(plan: CompactionPlan) -> slice:
    """Physical rows of the exact-recent block inside the 512-token buffer."""

    plan.validate()
    stop = plan.valid_tokens
    return slice(stop - plan.recent_size, stop)


@dataclasses.dataclass(frozen=True)
class LayerCompaction:
    """One layer's fitted artifact plus fit-time and held-out diagnostics."""

    layer_index: int
    result: FrameSampAttentionMatchingResult
    fit_metrics: AttentionMatchingMetrics
    heldout_output_relative_l2: float
    heldout_log_mass_rmse: float

    @property
    def artifact(self) -> AttentionMatchingArtifact:
        return self.result.artifact

    def label(self) -> dict[str, object]:
        return {
            "layer_index": int(self.layer_index),
            "compact_tokens": int(self.artifact.target_size),
            "source_tokens": int(self.artifact.source_size),
            "artifact_sha256": self.artifact.sha256(),
            "mass_solver": self.result.mass_solver,
            "value_solver": self.result.value_solver,
            "fit_output_relative_l2": float(self.fit_metrics.output_relative_l2),
            "fit_log_mass_rmse": float(self.fit_metrics.log_mass_rmse),
            "heldout_output_relative_l2": float(self.heldout_output_relative_l2),
            "heldout_log_mass_rmse": float(self.heldout_log_mass_rmse),
            "selected_step_indices": self.result.selected_step_indices.tolist(),
        }


def _combined_attention(
    queries: np.ndarray,
    artifact: AttentionMatchingArtifact,
    recent_keys: np.ndarray,
    recent_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if recent_keys.shape[0] == 0:
        return scaled_dot_product_attention(
            queries,
            artifact.keys,
            artifact.values,
            beta_am=artifact.beta_am,
            scale=artifact.scale,
        )
    return attend_compact_old_and_recent(queries, artifact, recent_keys, recent_values)


def fit_layer_compaction(
    history: FrameSampHistory,
    plan: CompactionPlan,
    *,
    layer_index: int,
    fit_queries: np.ndarray,
    heldout_queries: np.ndarray,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    mass_ridge: float = 0.0,
    value_ridge: float = 0.0,
) -> LayerCompaction:
    """Fit one layer's artifact on the fit bank and score it on the held-out bank.

    ``fit_queries`` / ``heldout_queries`` are ``[4, samples, head_dim]``.  The
    keys/values are the physical 512-row teacher taps for this layer.
    """

    plan.validate()
    view = old_history_view(history, plan)
    result = fit_framesamp_attention_matching(
        view,
        fit_queries,
        keys_post_rope,
        values_post_projection,
        plan.compact_size,
        layer_index=layer_index,
        mass_ridge=mass_ridge,
        value_ridge=value_ridge,
    )
    if result.effective_target_size != plan.compact_size:
        raise ValueError(
            f"layer {layer_index} produced {result.effective_target_size} compact tokens, expected {plan.compact_size}"
        )

    valid = np.flatnonzero(history.token_mask)
    recent = recent_token_slice(plan)
    recent_keys = np.asarray(keys_post_rope, dtype=np.float64)[recent]
    recent_values = np.asarray(values_post_projection, dtype=np.float64)[recent]
    full_keys = np.asarray(keys_post_rope, dtype=np.float64)[valid]
    full_values = np.asarray(values_post_projection, dtype=np.float64)[valid]

    heldout = np.asarray(heldout_queries, dtype=np.float64)
    flat = heldout.reshape(-1, heldout.shape[-1])
    teacher_output, teacher_log_mass = scaled_dot_product_attention(
        flat, full_keys, full_values, scale=result.artifact.scale
    )
    compact_output, compact_log_mass = _combined_attention(flat, result.artifact, recent_keys, recent_values)
    difference = compact_output - teacher_output
    denominator = max(float(np.linalg.norm(teacher_output)), np.finfo(np.float64).tiny)
    return LayerCompaction(
        layer_index=int(layer_index),
        result=result,
        fit_metrics=result.metrics,
        heldout_output_relative_l2=float(np.linalg.norm(difference) / denominator),
        heldout_log_mass_rmse=float(np.sqrt(np.mean(np.square(compact_log_mass - teacher_log_mass)))),
    )


def fit_all_layers(
    history: FrameSampHistory,
    plan: CompactionPlan,
    *,
    fit_queries: np.ndarray,
    heldout_queries: np.ndarray,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    mass_ridge: float = 0.0,
    value_ridge: float = 0.0,
) -> tuple[LayerCompaction, ...]:
    """Fit every layer.  Arrays are ``[layers, ...]`` in scan order."""

    keys = np.asarray(keys_post_rope)
    values = np.asarray(values_post_projection)
    if keys.ndim != 3 or keys.shape[1] != TOKEN_BUDGET:
        raise ValueError(f"keys must be [layers, {TOKEN_BUDGET}, head_dim], got {keys.shape}")
    if values.shape[:2] != keys.shape[:2]:
        raise ValueError("key and value taps describe different layers or token counts")
    layers = keys.shape[0]
    if np.asarray(fit_queries).shape[0] != layers or np.asarray(heldout_queries).shape[0] != layers:
        raise ValueError("query banks and teacher taps describe different layer counts")
    return tuple(
        fit_layer_compaction(
            history,
            plan,
            layer_index=index,
            fit_queries=np.asarray(fit_queries)[index],
            heldout_queries=np.asarray(heldout_queries)[index],
            keys_post_rope=keys[index],
            values_post_projection=values[index],
            mass_ridge=mass_ridge,
            value_ridge=value_ridge,
        )
        for index in range(layers)
    )


def stack_am_pack_arrays(
    fits: tuple[LayerCompaction, ...],
    plan: CompactionPlan,
    *,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    runtime_dtype: np.dtype | str,
) -> dict[str, np.ndarray]:
    """Build the ``[layers, 1, ...]`` arrays consumed by the scanned AM patch.

    The compact payload is cast once, here, to the runtime dtype -- that cast is
    the only quantization in the serve path and is recorded with the run.
    """

    plan.validate()
    if not fits:
        raise ValueError("no layer fits to stack")
    dtype = np.dtype(runtime_dtype)
    keys = np.asarray(keys_post_rope)
    values = np.asarray(values_post_projection)
    recent = recent_token_slice(plan)
    compact_keys = np.stack([fit.artifact.keys for fit in fits], axis=0).astype(dtype)
    compact_values = np.stack([fit.artifact.values for fit in fits], axis=0).astype(dtype)
    compact_beta = np.stack([fit.artifact.beta_am for fit in fits], axis=0).astype(np.float32)
    recent_keys = keys[:, recent, :].astype(dtype)
    recent_values = values[:, recent, :].astype(dtype)
    if compact_keys.shape[1] != plan.compact_size or recent_keys.shape[1] != plan.recent_size:
        raise ValueError("stacked AM pack disagrees with its compaction plan")
    layers = compact_keys.shape[0]
    return {
        # [L, B=1, M, KV heads=1, H]
        "compact_keys": compact_keys[:, None, :, None, :],
        "compact_values": compact_values[:, None, :, None, :],
        "compact_beta_am": compact_beta[:, None, :],
        "recent_keys": recent_keys[:, None, :, None, :],
        "recent_values": recent_values[:, None, :, None, :],
        "recent_token_mask": np.ones((layers, 1, plan.recent_size), dtype=np.bool_),
    }


def served_heldout_diagnostics(
    fits: tuple[LayerCompaction, ...],
    plan: CompactionPlan,
    *,
    heldout_queries: np.ndarray,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    pack_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], ...]:
    """Score the *quantized served payload*, not the float64 fit artifact.

    The solver diagnostics in :class:`LayerCompaction` describe its float64
    solution.  Production E0 casts compact and exact-recent K/V to bfloat16 once
    before the Linen scan consumes them; these metrics deliberately reconstruct
    attention from those post-cast arrays and are the held-out evidence rows.
    """

    plan.validate()
    if not fits:
        raise ValueError("served diagnostics require at least one fitted layer")
    queries = np.asarray(heldout_queries)
    keys = np.asarray(keys_post_rope)
    values = np.asarray(values_post_projection)
    layers = len(fits)
    if queries.ndim != 4 or queries.shape[0] != layers:
        raise ValueError("held-out query bank must be [layers, 4, samples, head_dim]")
    if keys.shape[:2] != (layers, TOKEN_BUDGET) or values.shape[:2] != keys.shape[:2]:
        raise ValueError("teacher K/V taps disagree with the fitted layer stack")

    compact_keys = np.asarray(pack_arrays["compact_keys"])
    compact_values = np.asarray(pack_arrays["compact_values"])
    compact_beta = np.asarray(pack_arrays["compact_beta_am"])
    recent_keys = np.asarray(pack_arrays["recent_keys"])
    recent_values = np.asarray(pack_arrays["recent_values"])
    expected_compact = (layers, 1, plan.compact_size, 1, keys.shape[-1])
    expected_recent = (layers, 1, plan.recent_size, 1, keys.shape[-1])
    if compact_keys.shape != expected_compact or compact_values.shape != expected_compact:
        raise ValueError(f"served compact K/V must be {expected_compact}")
    if recent_keys.shape != expected_recent or recent_values.shape != expected_recent:
        raise ValueError(f"served recent K/V must be {expected_recent}")
    if compact_beta.shape != (layers, 1, plan.compact_size):
        raise ValueError("served beta_AM shape disagrees with the compact payload")
    for name, array in (
        ("compact_keys", compact_keys),
        ("compact_values", compact_values),
        ("compact_beta_am", compact_beta),
        ("recent_keys", recent_keys),
        ("recent_values", recent_values),
    ):
        if not np.isfinite(array.astype(np.float32, copy=False)).all():
            raise ValueError(f"served {name} contains non-finite values after runtime quantization")

    # ``FrameSampHistory.validate`` and ``CompactionPlan`` jointly guarantee
    # that valid teacher rows are exactly this right-padded physical prefix.
    valid = np.arange(plan.valid_tokens, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for layer_index, fit in enumerate(fits):
        q = queries[layer_index].reshape(-1, queries.shape[-1]).astype(np.float64)
        teacher_output, teacher_log_mass = scaled_dot_product_attention(
            q,
            keys[layer_index, valid].astype(np.float64),
            values[layer_index, valid].astype(np.float64),
            scale=fit.artifact.scale,
        )
        served_keys = np.concatenate(
            [compact_keys[layer_index, 0, :, 0], recent_keys[layer_index, 0, :, 0]], axis=0
        ).astype(np.float64)
        served_values = np.concatenate(
            [compact_values[layer_index, 0, :, 0], recent_values[layer_index, 0, :, 0]], axis=0
        ).astype(np.float64)
        served_beta = np.concatenate(
            [compact_beta[layer_index, 0], np.zeros(plan.recent_size, dtype=np.float32)]
        ).astype(np.float64)
        served_output, served_log_mass = scaled_dot_product_attention(
            q,
            served_keys,
            served_values,
            beta_am=served_beta,
            scale=fit.artifact.scale,
        )
        denominator = max(float(np.linalg.norm(teacher_output)), np.finfo(np.float64).tiny)
        rows.append(
            {
                "layer_index": layer_index,
                "served_storage_dtype": str(compact_keys.dtype),
                "served_heldout_output_relative_l2": float(
                    np.linalg.norm(served_output - teacher_output) / denominator
                ),
                "served_heldout_log_mass_rmse": float(np.sqrt(np.mean(np.square(served_log_mass - teacher_log_mass)))),
                "served_payload_finite_after_quantization": True,
            }
        )
    return tuple(rows)


def random_subset_pack_arrays(
    plan: CompactionPlan,
    *,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    seed: int,
    runtime_dtype: np.dtype | str,
) -> dict[str, np.ndarray]:
    """Control arm: keep a uniformly random token subset, fit nothing.

    This is what an implementer gets for free without Attention Matching -- no
    RMS selection, no mass correction, no value solve.  If AM does not beat it
    at the same budget, the method is not what is buying the retention.
    """

    plan.validate()
    dtype = np.dtype(runtime_dtype)
    keys = np.asarray(keys_post_rope)
    values = np.asarray(values_post_projection)
    layers = keys.shape[0]
    generator = np.random.default_rng(seed)
    # One shared draw across layers: the token identity is a property of the
    # memory, not of a layer, exactly as in the RMS-selected artifact.
    chosen = np.sort(generator.choice(plan.valid_tokens - plan.recent_size, size=plan.compact_size, replace=False))
    recent = recent_token_slice(plan)
    return {
        "compact_keys": keys[:, chosen, :].astype(dtype)[:, None, :, None, :],
        "compact_values": values[:, chosen, :].astype(dtype)[:, None, :, None, :],
        "compact_beta_am": np.zeros((layers, 1, plan.compact_size), dtype=np.float32),
        "recent_keys": keys[:, recent, :].astype(dtype)[:, None, :, None, :],
        "recent_values": values[:, recent, :].astype(dtype)[:, None, :, None, :],
        "recent_token_mask": np.ones((layers, 1, plan.recent_size), dtype=np.bool_),
        "selected_indices": chosen,
    }


def destroyed_pack_arrays(
    *,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    runtime_dtype: np.dtype | str,
) -> dict[str, np.ndarray]:
    """Control arm: keep the full cache but zero every memory value.

    The memory modulation then carries no episode information at all, so this
    measures how far the velocity field can move *at most* when history is
    removed.  Any AM error must be read against this scale: a compression whose
    error is a small fraction of it is only as meaningful as the memory itself.
    """

    dtype = np.dtype(runtime_dtype)
    keys = np.asarray(keys_post_rope)
    values = np.asarray(values_post_projection)
    layers, tokens = keys.shape[0], keys.shape[1]
    return {
        "compact_keys": keys.astype(dtype)[:, None, :, None, :],
        "compact_values": np.zeros_like(values, dtype=dtype)[:, None, :, None, :],
        "compact_beta_am": np.zeros((layers, 1, tokens), dtype=np.float32),
        "recent_keys": keys[:, :0, :].astype(dtype)[:, None, :, None, :],
        "recent_values": values[:, :0, :].astype(dtype)[:, None, :, None, :],
        "recent_token_mask": np.ones((layers, 1, 0), dtype=np.bool_),
    }


def payload_quantization_parity(
    fits: tuple[LayerCompaction, ...],
    plan: CompactionPlan,
    *,
    heldout_queries: np.ndarray,
    keys_post_rope: np.ndarray,
    values_post_projection: np.ndarray,
    runtime_dtype: np.dtype | str,
) -> dict[str, object]:
    """Isolate the served payload cast from every other source of AM error.

    The fit runs in float64 but the policy is served a ``runtime_dtype`` copy of
    the compact keys/values.  That cast is the only quantization in the serve
    path, and it must be reported as its own number rather than folded into the
    end-to-end velocity delta: the OLS replacement values are *new* vectors, not
    teacher rows, so their rounding error is not bounded by the teacher's own
    representation error.  Beta stays float32 because it is added to float32
    logits, which is exact from any sealed payload dtype.

    Returns the per-layer held-out attention error before and after the cast.
    """

    plan.validate()
    if not fits:
        raise ValueError("no layer fits to check")
    dtype = np.dtype(runtime_dtype)
    keys = np.asarray(keys_post_rope)
    values = np.asarray(values_post_projection)
    heldout = np.asarray(heldout_queries)
    if heldout.shape[0] != len(fits):
        raise ValueError("held-out bank and layer fits describe different layer counts")
    recent = recent_token_slice(plan)
    rows: list[dict[str, float]] = []
    for index, fit in enumerate(fits):
        artifact = fit.artifact
        valid = np.arange(plan.valid_tokens)
        full_keys = keys[index][valid].astype(np.float64)
        full_values = values[index][valid].astype(np.float64)
        recent_keys = keys[index][recent].astype(np.float64)
        recent_values = values[index][recent].astype(np.float64)
        flat = heldout[index].reshape(-1, heldout.shape[-1]).astype(np.float64)
        teacher, _ = scaled_dot_product_attention(flat, full_keys, full_values, scale=artifact.scale)
        denominator = max(float(np.linalg.norm(teacher)), np.finfo(np.float64).tiny)
        exact, _ = _combined_attention(flat, artifact, recent_keys, recent_values)
        quantized_artifact = dataclasses.replace(
            artifact,
            keys=artifact.keys.astype(dtype).astype(np.float64),
            values=artifact.values.astype(dtype).astype(np.float64),
        )
        quantized, _ = _combined_attention(
            flat,
            quantized_artifact,
            recent_keys.astype(dtype).astype(np.float64),
            recent_values.astype(dtype).astype(np.float64),
        )
        rows.append(
            {
                "layer_index": int(fit.layer_index),
                "heldout_relative_l2_float64": float(np.linalg.norm(exact - teacher) / denominator),
                "heldout_relative_l2_quantized": float(np.linalg.norm(quantized - teacher) / denominator),
                "quantization_only_relative_l2": float(
                    np.linalg.norm(quantized - exact) / max(float(np.linalg.norm(exact)), np.finfo(np.float64).tiny)
                ),
            }
        )
    return {
        "payload_dtype": dtype.name,
        "beta_dtype": "float32_exact_from_any_sealed_payload",
        "layers": rows,
        "quantization_only_relative_l2_mean": float(np.mean([row["quantization_only_relative_l2"] for row in rows])),
        "quantization_only_relative_l2_max": float(np.max([row["quantization_only_relative_l2"] for row in rows])),
        "heldout_relative_l2_float64_mean": float(np.mean([row["heldout_relative_l2_float64"] for row in rows])),
        "heldout_relative_l2_quantized_mean": float(np.mean([row["heldout_relative_l2_quantized"] for row in rows])),
    }


def served_kv_bytes(plan: CompactionPlan, *, layers: int, head_dim: int, itemsize: int) -> dict[str, int]:
    """Memory footprint of the served memory K/V, full versus compacted."""

    plan.validate()
    per_token = 2 * head_dim * itemsize  # K and V
    full = layers * plan.valid_tokens * per_token
    compact = layers * plan.served_tokens * per_token
    return {
        "full_kv_bytes": int(full),
        "compact_kv_bytes": int(compact),
        "kv_bytes_ratio": float(full / compact),
    }
