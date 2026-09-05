"""Sealed storage and runtime boundary for RoboMME FrameSamp AM artifacts.

Attention Matching is fitted in float64, but a fitted NumPy object is not a
scientific artifact: it does not say which teacher, episode, layer, tap points,
query bank, mask, or positional convention produced it.  This module creates a
two-file bundle (``manifest.json`` + ``payload.npz``) that seals those facts and
refuses to serialize until a lower-precision payload passes a held-out parity
gate.

The v1 runtime entrypoint is intentionally narrow.  The artifact compresses
all valid FrameSamp tokens, compact keys are already projected and
RoPE-applied, and action queries retain the teacher's logical offset of 512.
Consequently v1 requires an empty raw-recent block: appending any unchanged
teacher token would duplicate evidence.  A future compact-old + raw-recent
variant needs a separately sealed disjoint partition before it may reuse the
shared-softmax kernel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from robomme_integration.training.attention_matching import (
    ARTIFACT_METHOD,
    MASS_SOLVER,
    MASS_SOLVER_DISABLED,
    VALUE_SOLVER,
    AttentionMatchingArtifact,
    AttentionMatchingMetrics,
    attend_compact_old_and_recent,
    evaluate_attention_matching,
)
from robomme_integration.training.framesamp_attention_matching import (
    FrameSampAttentionMatchingResult,
)
from robomme_integration.training.upstream_framesamp_data import (
    TOKEN_BUDGET,
    FrameSampHistory,
)

BUNDLE_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
PAYLOAD_FILENAME = "payload.npz"
QUERY_TAP_STAGE = "post_rope_pre_scale"
KEY_TAP_STAGE = "post_rope"
VALUE_TAP_STAGE = "post_projection"
COMPACT_KEY_OPERATION = "consume_without_projection_or_rope"
RECENT_MEMORY_KIND = "raw_uncompressed"
MEMORY_PARTITION_KIND = "compact_all_valid_framesamp_tokens_no_recent_v1"
_PAYLOAD_ARRAYS = frozenset(
    {
        "selected_indices",
        "keys",
        "values",
        "beta_am",
        "source_physical_indices",
        "selected_physical_indices",
        "selected_step_indices",
        "selected_patch_indices",
    }
)


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_float(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" and at least {minimum}" if minimum is not None else ""
        raise ValueError(f"{label} must be finite{qualifier}")
    return result


def _require_sha(value: object, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
    value = _require_nonempty(value, label=label)
    if len(value) not in lengths or any(char not in "0123456789abcdef" for char in value):
        allowed = "/".join(str(length) for length in lengths)
        raise ValueError(f"{label} must be a lowercase {allowed}-hex SHA")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    """Hash named arrays including name, dtype, shape, and exact bytes."""

    if not arrays:
        raise ValueError("at least one array is required for a bundle hash")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if not name:
            raise ValueError("array hash names must be nonempty")
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class QuantizationParityThresholds:
    """Maximum allowed metric increase caused solely by storage quantization."""

    output_rmse_increase: float = 1e-5
    output_relative_l2_increase: float = 1e-5
    log_mass_rmse_increase: float = 1e-5
    relative_mass_rmse_increase: float = 1e-5

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            _require_float(getattr(self, field.name), label=field.name, minimum=0.0)

    def enforce(
        self,
        reference: AttentionMatchingMetrics,
        stored: AttentionMatchingMetrics,
        *,
        split: str,
    ) -> None:
        self.validate()
        reference.validate()
        stored.validate()
        failures: list[str] = []
        for metric in dataclasses.fields(AttentionMatchingMetrics):
            name = metric.name
            limit = float(getattr(self, f"{name}_increase"))
            increase = float(getattr(stored, name) - getattr(reference, name))
            if increase > limit:
                failures.append(f"{name} increase {increase:.6g} > {limit:.6g}")
        if failures:
            raise ValueError(f"{split} storage-quantization parity gate failed: " + "; ".join(failures))


def _metrics_dict(metrics: AttentionMatchingMetrics) -> dict[str, float]:
    metrics.validate()
    return {field.name: float(getattr(metrics, field.name)) for field in dataclasses.fields(metrics)}


def _metrics_from_dict(value: object, *, label: str) -> AttentionMatchingMetrics:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    names = {field.name for field in dataclasses.fields(AttentionMatchingMetrics)}
    if set(value) != names:
        raise ValueError(f"{label} fields mismatch: expected {sorted(names)}, got {sorted(value)}")
    result = AttentionMatchingMetrics(
        **{name: _require_float(value[name], label=f"{label}.{name}", minimum=0.0) for name in names}
    )
    result.validate()
    return result


def _thresholds_dict(value: QuantizationParityThresholds) -> dict[str, float]:
    value.validate()
    return {field.name: float(getattr(value, field.name)) for field in dataclasses.fields(value)}


def _thresholds_from_dict(value: object) -> QuantizationParityThresholds:
    if not isinstance(value, dict):
        raise ValueError("quantization_parity_thresholds must be an object")
    names = {field.name for field in dataclasses.fields(QuantizationParityThresholds)}
    if set(value) != names:
        raise ValueError("quantization_parity_threshold fields mismatch")
    result = QuantizationParityThresholds(
        **{
            name: _require_float(value[name], label=f"quantization_parity_thresholds.{name}", minimum=0.0)
            for name in names
        }
    )
    result.validate()
    return result


@dataclasses.dataclass(frozen=True)
class FrameSampAMManifest:
    """Complete scientific and numerical identity for one layer/head artifact."""

    schema_version: int
    artifact_method: str
    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    layer_index: int
    kv_head_index: int
    query_head_count: int
    query_tap_stage: str
    key_tap_stage: str
    value_tap_stage: str
    resolved_attention_scale: float
    fit_query_bank_spec: str
    heldout_query_bank_spec: str
    fit_query_bank_sha256: str
    heldout_query_bank_sha256: str
    teacher_tap_sha256: str
    token_mask_sha256: str
    frame_map_sha256: str
    fit_queries_per_head: int
    heldout_queries_per_head: int
    valid_source_tokens: int
    physical_source_tokens: int
    key_dim: int
    value_dim: int
    logical_key_position_span: int
    query_position_offset: int
    memory_partition_kind: str
    requested_budget: int
    effective_budget: int
    fit_mass: bool
    mass_solver: str
    value_solver: str
    mass_ridge: float
    value_ridge: float
    float64_train_metrics: AttentionMatchingMetrics
    stored_train_metrics: AttentionMatchingMetrics
    float64_heldout_metrics: AttentionMatchingMetrics
    stored_heldout_metrics: AttentionMatchingMetrics
    quantization_parity_thresholds: QuantizationParityThresholds
    storage_dtype: str
    payload_encoding: str
    stored_numeric_artifact_sha256: str
    payload_sha256: str

    def validate(self) -> None:
        if _require_int(self.schema_version, label="schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported FrameSamp AM bundle schema {self.schema_version}")
        if self.artifact_method != ARTIFACT_METHOD:
            raise ValueError(f"unsupported artifact method {self.artifact_method!r}")
        _require_sha(self.teacher_checkpoint_sha256, label="teacher_checkpoint_sha256")
        _require_sha(self.teacher_code_sha, label="teacher_code_sha", lengths=(40, 64))
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_int(self.layer_index, label="layer_index")
        if _require_int(self.kv_head_index, label="kv_head_index") != 0:
            raise ValueError("official MemoryAttention has exactly one KV head at index 0")
        if _require_int(self.query_head_count, label="query_head_count", minimum=1) != 4:
            raise ValueError("official MemoryAttention has exactly four query heads")
        if (self.query_tap_stage, self.key_tap_stage, self.value_tap_stage) != (
            QUERY_TAP_STAGE,
            KEY_TAP_STAGE,
            VALUE_TAP_STAGE,
        ):
            raise ValueError("manifest Q/K/V tap stages are not the official AM tap convention")
        scale = _require_float(self.resolved_attention_scale, label="resolved_attention_scale", minimum=0.0)
        if scale == 0:
            raise ValueError("resolved_attention_scale must be finite and positive")
        _require_nonempty(self.fit_query_bank_spec, label="fit_query_bank_spec")
        _require_nonempty(self.heldout_query_bank_spec, label="heldout_query_bank_spec")
        for label in (
            "fit_query_bank_sha256",
            "heldout_query_bank_sha256",
            "teacher_tap_sha256",
            "token_mask_sha256",
            "frame_map_sha256",
            "stored_numeric_artifact_sha256",
            "payload_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        if self.fit_query_bank_sha256 == self.heldout_query_bank_sha256:
            raise ValueError("held-out query bank must be distinct from the fit query bank")
        _require_int(self.fit_queries_per_head, label="fit_queries_per_head", minimum=1)
        _require_int(self.heldout_queries_per_head, label="heldout_queries_per_head", minimum=1)
        valid = _require_int(self.valid_source_tokens, label="valid_source_tokens", minimum=1)
        if _require_int(self.physical_source_tokens, label="physical_source_tokens", minimum=1) != TOKEN_BUDGET:
            raise ValueError("physical FrameSamp source length must remain 512")
        _require_int(self.key_dim, label="key_dim", minimum=1)
        _require_int(self.value_dim, label="value_dim", minimum=1)
        if _require_int(self.logical_key_position_span, label="logical_key_position_span", minimum=1) != TOKEN_BUDGET:
            raise ValueError("logical FrameSamp key span must remain 512")
        if _require_int(self.query_position_offset, label="query_position_offset") != TOKEN_BUDGET:
            raise ValueError("action-query RoPE offset must remain 512")
        if self.memory_partition_kind != MEMORY_PARTITION_KIND:
            raise ValueError("unsupported FrameSamp AM memory partition")
        requested = _require_int(self.requested_budget, label="requested_budget", minimum=1)
        effective = _require_int(self.effective_budget, label="effective_budget", minimum=1)
        if effective != min(requested, valid):
            raise ValueError("effective budget must be min(requested budget, valid source tokens)")
        if not isinstance(self.fit_mass, bool):
            raise ValueError("fit_mass must be Boolean")
        expected_mass_solver = MASS_SOLVER if self.fit_mass else MASS_SOLVER_DISABLED
        if self.mass_solver != expected_mass_solver or self.value_solver != VALUE_SOLVER:
            raise ValueError("solver identity is incompatible with this artifact method")
        for label in ("mass_ridge", "value_ridge"):
            _require_float(getattr(self, label), label=label, minimum=0.0)
        for metrics in (
            self.float64_train_metrics,
            self.stored_train_metrics,
            self.float64_heldout_metrics,
            self.stored_heldout_metrics,
        ):
            metrics.validate()
        self.quantization_parity_thresholds.enforce(
            self.float64_train_metrics, self.stored_train_metrics, split="train"
        )
        self.quantization_parity_thresholds.enforce(
            self.float64_heldout_metrics, self.stored_heldout_metrics, split="heldout"
        )
        expected_encoding = {
            "float32": "native_float32",
            "float16": "native_float16",
            "bfloat16": "uint16_bfloat16_bits",
        }
        if self.storage_dtype not in expected_encoding:
            raise ValueError("storage_dtype must be float32, float16, or bfloat16 (never float64)")
        if self.payload_encoding != expected_encoding[self.storage_dtype]:
            raise ValueError("payload encoding disagrees with storage dtype")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = dataclasses.asdict(self)
        for name in (
            "float64_train_metrics",
            "stored_train_metrics",
            "float64_heldout_metrics",
            "stored_heldout_metrics",
        ):
            result[name] = _metrics_dict(getattr(self, name))
        result["quantization_parity_thresholds"] = _thresholds_dict(self.quantization_parity_thresholds)
        return result

    @classmethod
    def from_dict(cls, value: object) -> "FrameSampAMManifest":
        if not isinstance(value, dict):
            raise ValueError("FrameSamp AM manifest must be a JSON object")
        names = {field.name for field in dataclasses.fields(cls)}
        if set(value) != names:
            raise ValueError(
                f"manifest fields mismatch: missing={sorted(names - set(value))}, "
                f"unexpected={sorted(set(value) - names)}"
            )
        decoded = dict(value)
        for name in (
            "float64_train_metrics",
            "stored_train_metrics",
            "float64_heldout_metrics",
            "stored_heldout_metrics",
        ):
            decoded[name] = _metrics_from_dict(decoded[name], label=name)
        decoded["quantization_parity_thresholds"] = _thresholds_from_dict(decoded["quantization_parity_thresholds"])
        result = cls(**decoded)
        result.validate()
        return result

    def scientific_sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class ExpectedFrameSampAMIdentity:
    """Identity a runtime must state before loading a bundle.

    ``manifest_sha256`` must come from a trusted experiment index rather than
    from the bundle being loaded.  It seals every query/mask/tap/metric field in
    addition to the human-readable routing fields below.
    """

    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    layer_index: int
    kv_head_index: int
    requested_budget: int
    manifest_sha256: str

    def validate(self) -> None:
        _require_sha(self.teacher_checkpoint_sha256, label="teacher_checkpoint_sha256")
        _require_sha(self.teacher_code_sha, label="teacher_code_sha", lengths=(40, 64))
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_int(self.layer_index, label="layer_index")
        if _require_int(self.kv_head_index, label="kv_head_index") != 0:
            raise ValueError("expected KV head must be 0")
        _require_int(self.requested_budget, label="requested_budget", minimum=1)
        _require_sha(self.manifest_sha256, label="manifest_sha256")

    def check(self, manifest: FrameSampAMManifest) -> None:
        self.validate()
        manifest.validate()
        for field in dataclasses.fields(self):
            if field.name == "manifest_sha256":
                continue
            expected = getattr(self, field.name)
            actual = getattr(manifest, field.name)
            if expected != actual:
                raise ValueError(
                    f"FrameSamp AM identity mismatch for {field.name}: expected {expected!r}, got {actual!r}"
                )
        actual_manifest_sha = manifest.scientific_sha256()
        if self.manifest_sha256 != actual_manifest_sha:
            raise ValueError(
                "FrameSamp AM identity mismatch for manifest_sha256: "
                f"expected {self.manifest_sha256!r}, got {actual_manifest_sha!r}"
            )


@dataclasses.dataclass(frozen=True)
class LoadedFrameSampAMArtifact:
    manifest: FrameSampAMManifest
    artifact: AttentionMatchingArtifact
    source_physical_indices: np.ndarray
    selected_physical_indices: np.ndarray
    selected_step_indices: np.ndarray
    selected_patch_indices: np.ndarray

    def validate(self) -> None:
        self.manifest.validate()
        self.artifact.validate(
            expected_source_size=self.manifest.valid_source_tokens,
            expected_target_size=self.manifest.effective_budget,
            expected_key_dim=self.manifest.key_dim,
            expected_value_dim=self.manifest.value_dim,
        )
        expected_scale = 1.0 / math.sqrt(self.manifest.key_dim)
        if self.artifact.scale != expected_scale:
            raise ValueError("stored attention scale is not the official single pre-scale")
        source = np.asarray(self.source_physical_indices)
        selected_physical = np.asarray(self.selected_physical_indices)
        selected = self.artifact.selected_indices.astype(np.int64, copy=False)
        if source.shape != (self.manifest.valid_source_tokens,) or source.dtype != np.int32:
            raise ValueError("stored source physical map has the wrong shape or dtype")
        if np.any(source < 0) or np.any(source >= TOKEN_BUDGET) or np.any(np.diff(source) <= 0):
            raise ValueError("stored source physical map is not a unique ordered 512-buffer subset")
        if selected_physical.dtype != np.int32:
            raise ValueError("stored selected physical map must use int32")
        if not np.array_equal(selected_physical, source[selected]):
            raise ValueError("stored selected physical map disagrees with selected indices")
        steps = np.asarray(self.selected_step_indices)
        patches = np.asarray(self.selected_patch_indices)
        if steps.shape != (self.manifest.effective_budget,) or steps.dtype != np.int32:
            raise ValueError("stored selected_step_indices must be int32 with effective-budget length")
        if patches.shape != (self.manifest.effective_budget,) or patches.dtype != np.int16:
            raise ValueError("stored selected_patch_indices must be int16 with effective-budget length")
        if np.any(steps < 0) or np.any(patches < 0) or np.any(patches >= 16):
            raise ValueError("stored selected logical step/patch map is outside its valid range")


def _official_query_bank(value: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(value)
    if raw.ndim != 3 or raw.shape[0] != 4 or not raw.shape[1] or not raw.shape[2]:
        raise ValueError(f"{label} must be [4 query heads, samples, head_dim]")
    numeric = raw.astype(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite values")
    return raw, numeric.reshape(-1, numeric.shape[-1])


def _teacher_taps(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    key_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_keys = np.asarray(keys)
    raw_values = np.asarray(values)
    if raw_keys.ndim != 2 or raw_values.ndim != 2:
        raise ValueError("teacher K/V taps must be rank-2 matrices")
    if raw_keys.shape[0] != TOKEN_BUDGET or raw_values.shape[0] != TOKEN_BUDGET:
        raise ValueError("teacher K/V taps must retain the physical 512-token buffer")
    if raw_keys.shape[1] != key_dim or raw_values.shape[0] != raw_keys.shape[0]:
        raise ValueError("teacher K/V tap dimensions disagree with the query/artifact")
    keys64 = raw_keys.astype(np.float64)
    values64 = raw_values.astype(np.float64)
    if not np.isfinite(keys64).all() or not np.isfinite(values64).all():
        raise ValueError("teacher K/V taps contain non-finite values")
    return raw_keys, raw_values


def _metrics_close(left: AttentionMatchingMetrics, right: AttentionMatchingMetrics) -> bool:
    return all(
        math.isclose(float(getattr(left, field.name)), float(getattr(right, field.name)), rel_tol=1e-12, abs_tol=1e-14)
        for field in dataclasses.fields(AttentionMatchingMetrics)
    )


def _storage_dtype(name: str) -> np.dtype:
    if name == "bfloat16":
        try:
            import ml_dtypes
        except ModuleNotFoundError as error:  # pragma: no cover - policy environments provide it.
            raise RuntimeError("bfloat16 storage requires ml_dtypes") from error
        return np.dtype(ml_dtypes.bfloat16)
    if name not in ("float32", "float16"):
        raise ValueError("storage_dtype must be float32, float16, or bfloat16; float64 is forbidden")
    return np.dtype(name)


def _quantized_artifact(
    source: AttentionMatchingArtifact,
    storage_dtype: str,
) -> AttentionMatchingArtifact:
    dtype = _storage_dtype(storage_dtype)
    result = AttentionMatchingArtifact(
        source_size=source.source_size,
        selected_indices=np.asarray(source.selected_indices, dtype=np.int32),
        keys=np.asarray(source.keys, dtype=dtype),
        values=np.asarray(source.values, dtype=dtype),
        beta_am=np.asarray(source.beta_am, dtype=dtype),
        scale=source.scale,
        method=source.method,
        schema_version=source.schema_version,
    )
    result.validate()
    return result


def _encode_float_payload(array: np.ndarray, storage_dtype: str) -> np.ndarray:
    quantized = np.ascontiguousarray(np.asarray(array, dtype=_storage_dtype(storage_dtype)))
    if storage_dtype == "bfloat16":
        return quantized.view(np.uint16)
    return quantized


def _decode_float_payload(array: np.ndarray, storage_dtype: str, *, label: str) -> np.ndarray:
    array = np.asarray(array)
    if storage_dtype == "bfloat16":
        if array.dtype != np.uint16:
            raise ValueError(f"{label} must contain uint16 bfloat16 bit payloads")
        return array.view(_storage_dtype(storage_dtype)).copy()
    expected = _storage_dtype(storage_dtype)
    if array.dtype != expected:
        raise ValueError(f"{label} dtype mismatch: expected {expected}, got {array.dtype}")
    return array.copy()


def seal_framesamp_am_artifact(
    destination: str | Path,
    result: FrameSampAttentionMatchingResult,
    history: FrameSampHistory,
    fit_queries_post_rope_pre_scale: np.ndarray,
    heldout_queries_post_rope_pre_scale: np.ndarray,
    teacher_keys_post_rope: np.ndarray,
    teacher_values_post_projection: np.ndarray,
    *,
    teacher_checkpoint_sha256: str,
    teacher_code_sha: str,
    task_id: str,
    episode_id: str,
    causal_cut_step: int,
    fit_query_bank_spec: str,
    heldout_query_bank_spec: str,
    storage_dtype: str = "float32",
    parity_thresholds: QuantizationParityThresholds = QuantizationParityThresholds(),
) -> FrameSampAMManifest:
    """Quantize, parity-check, and atomically seal one scientific AM bundle."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing AM bundle: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"AM bundle parent does not exist: {destination.parent}")
    result.validate(history)
    fit_raw, fit_queries = _official_query_bank(fit_queries_post_rope_pre_scale, label="fit query bank")
    heldout_raw, heldout_queries = _official_query_bank(
        heldout_queries_post_rope_pre_scale, label="heldout query bank"
    )
    if fit_queries.shape[1] != result.artifact.keys.shape[1] or heldout_queries.shape[1] != fit_queries.shape[1]:
        raise ValueError("fit/heldout query dimensions disagree with compact keys")
    raw_keys, raw_values = _teacher_taps(
        teacher_keys_post_rope,
        teacher_values_post_projection,
        key_dim=fit_queries.shape[1],
    )
    valid = result.source_physical_indices.astype(np.int64, copy=False)
    full_keys = raw_keys[valid]
    full_values = raw_values[valid]
    expected_selected_keys = np.asarray(full_keys, dtype=np.float64)[result.artifact.selected_indices]
    if not np.array_equal(result.artifact.keys, expected_selected_keys):
        raise ValueError("artifact keys do not match selected post-RoPE teacher keys")
    expected_scale = 1.0 / math.sqrt(result.artifact.keys.shape[1])
    if result.artifact.scale != expected_scale:
        raise ValueError("artifact attention scale is not the official single pre-scale")
    float64_train = evaluate_attention_matching(fit_queries, full_keys, full_values, result.artifact)
    if not _metrics_close(float64_train, result.metrics):
        raise ValueError("recorded fit metrics do not reproduce from the supplied query/K/V taps")
    float64_heldout = evaluate_attention_matching(heldout_queries, full_keys, full_values, result.artifact)
    stored_artifact = _quantized_artifact(result.artifact, storage_dtype)
    stored_train = evaluate_attention_matching(fit_queries, full_keys, full_values, stored_artifact)
    stored_heldout = evaluate_attention_matching(heldout_queries, full_keys, full_values, stored_artifact)
    parity_thresholds.enforce(float64_train, stored_train, split="train")
    parity_thresholds.enforce(float64_heldout, stored_heldout, split="heldout")

    causal_cut_step = _require_int(causal_cut_step, label="causal_cut_step")
    valid_frame_indices = history.frame_indices[history.frame_mask]
    if not valid_frame_indices.size or int(valid_frame_indices[-1]) != causal_cut_step:
        raise ValueError("causal_cut_step must equal the final frame in the causal FrameSamp prefix")
    fit_hash = array_bundle_sha256(queries_post_rope_pre_scale=fit_raw)
    heldout_hash = array_bundle_sha256(queries_post_rope_pre_scale=heldout_raw)
    if fit_hash == heldout_hash:
        raise ValueError("held-out query bank must be distinct from the fit query bank")

    payload_encoding = {
        "float32": "native_float32",
        "float16": "native_float16",
        "bfloat16": "uint16_bfloat16_bits",
    }.get(storage_dtype)
    if payload_encoding is None:
        _storage_dtype(storage_dtype)  # Raise the canonical error.
        raise AssertionError("unreachable")
    payload = {
        "selected_indices": np.asarray(stored_artifact.selected_indices, dtype=np.int32),
        "keys": _encode_float_payload(stored_artifact.keys, storage_dtype),
        "values": _encode_float_payload(stored_artifact.values, storage_dtype),
        "beta_am": _encode_float_payload(stored_artifact.beta_am, storage_dtype),
        "source_physical_indices": np.asarray(result.source_physical_indices, dtype=np.int32),
        "selected_physical_indices": np.asarray(result.selected_physical_indices, dtype=np.int32),
        "selected_step_indices": np.asarray(result.selected_step_indices, dtype=np.int32),
        "selected_patch_indices": np.asarray(result.selected_patch_indices, dtype=np.int16),
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        payload_path = temporary / PAYLOAD_FILENAME
        with payload_path.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        manifest = FrameSampAMManifest(
            schema_version=BUNDLE_SCHEMA_VERSION,
            artifact_method=result.artifact.method,
            teacher_checkpoint_sha256=teacher_checkpoint_sha256,
            teacher_code_sha=teacher_code_sha,
            task_id=task_id,
            episode_id=episode_id,
            causal_cut_step=causal_cut_step,
            layer_index=result.layer_index,
            kv_head_index=result.kv_head_index,
            query_head_count=result.query_head_count,
            query_tap_stage=result.query_tap_stage,
            key_tap_stage=result.key_tap_stage,
            value_tap_stage=result.value_tap_stage,
            resolved_attention_scale=float(result.artifact.scale),
            fit_query_bank_spec=fit_query_bank_spec,
            heldout_query_bank_spec=heldout_query_bank_spec,
            fit_query_bank_sha256=fit_hash,
            heldout_query_bank_sha256=heldout_hash,
            teacher_tap_sha256=array_bundle_sha256(
                keys_post_rope=raw_keys,
                values_post_projection=raw_values,
            ),
            token_mask_sha256=array_bundle_sha256(token_mask=history.token_mask),
            frame_map_sha256=array_bundle_sha256(
                frame_indices=history.frame_indices,
                frame_mask=history.frame_mask,
            ),
            fit_queries_per_head=fit_raw.shape[1],
            heldout_queries_per_head=heldout_raw.shape[1],
            valid_source_tokens=result.artifact.source_size,
            physical_source_tokens=TOKEN_BUDGET,
            key_dim=result.artifact.keys.shape[1],
            value_dim=result.artifact.values.shape[1],
            logical_key_position_span=result.teacher_key_position_span,
            query_position_offset=result.query_position_offset,
            memory_partition_kind=MEMORY_PARTITION_KIND,
            requested_budget=result.requested_target_size,
            effective_budget=result.effective_target_size,
            fit_mass=result.fit_mass,
            mass_solver=result.mass_solver,
            value_solver=result.value_solver,
            mass_ridge=result.mass_ridge,
            value_ridge=result.value_ridge,
            float64_train_metrics=float64_train,
            stored_train_metrics=stored_train,
            float64_heldout_metrics=float64_heldout,
            stored_heldout_metrics=stored_heldout,
            quantization_parity_thresholds=parity_thresholds,
            storage_dtype=storage_dtype,
            payload_encoding=payload_encoding,
            stored_numeric_artifact_sha256=stored_artifact.sha256(),
            payload_sha256=_sha256_file(payload_path),
        )
        manifest.validate()
        manifest_path = temporary / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_framesamp_am_artifact(
    bundle: str | Path,
    *,
    expected: ExpectedFrameSampAMIdentity,
) -> LoadedFrameSampAMArtifact:
    """Load only after payload integrity and explicit scientific identity match."""

    bundle = Path(bundle)
    if not bundle.is_dir():
        raise FileNotFoundError(f"FrameSamp AM bundle is not a directory: {bundle}")
    files = {path.name for path in bundle.iterdir()}
    if files != {MANIFEST_FILENAME, PAYLOAD_FILENAME}:
        raise ValueError(f"FrameSamp AM bundle file set mismatch: {sorted(files)}")
    manifest_path = bundle / MANIFEST_FILENAME
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("FrameSamp AM manifest is not valid UTF-8 JSON") from error
    manifest = FrameSampAMManifest.from_dict(raw_manifest)
    expected.check(manifest)
    payload_path = bundle / PAYLOAD_FILENAME
    if _sha256_file(payload_path) != manifest.payload_sha256:
        raise ValueError("FrameSamp AM payload SHA256 mismatch")
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            if set(payload.files) != _PAYLOAD_ARRAYS:
                raise ValueError("FrameSamp AM payload array set mismatch")
            selected_indices = np.asarray(payload["selected_indices"]).copy()
            keys = _decode_float_payload(payload["keys"], manifest.storage_dtype, label="keys")
            values = _decode_float_payload(payload["values"], manifest.storage_dtype, label="values")
            beta_am = _decode_float_payload(payload["beta_am"], manifest.storage_dtype, label="beta_am")
            maps = {
                name: np.asarray(payload[name]).copy()
                for name in (
                    "source_physical_indices",
                    "selected_physical_indices",
                    "selected_step_indices",
                    "selected_patch_indices",
                )
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("FrameSamp"):
            raise
        raise ValueError("FrameSamp AM payload is unreadable or invalid") from error
    artifact = AttentionMatchingArtifact(
        source_size=manifest.valid_source_tokens,
        selected_indices=selected_indices,
        keys=keys,
        values=values,
        beta_am=beta_am,
        scale=manifest.resolved_attention_scale,
        method=manifest.artifact_method,
    )
    if artifact.sha256() != manifest.stored_numeric_artifact_sha256:
        raise ValueError("stored numeric artifact checksum mismatch")
    loaded = LoadedFrameSampAMArtifact(
        manifest=manifest,
        artifact=artifact,
        **maps,
    )
    loaded.validate()
    return loaded


@dataclasses.dataclass(frozen=True)
class FrameSampAMRuntimeInputs:
    """Explicit stage/position assertions required by the runtime consumer."""

    queries_post_rope_pre_scale: np.ndarray
    query_positions: np.ndarray
    recent_keys_post_rope: np.ndarray
    recent_values_post_projection: np.ndarray
    recent_physical_positions: np.ndarray
    recent_token_mask: np.ndarray
    query_tap_stage: str
    recent_key_tap_stage: str
    recent_value_tap_stage: str
    compact_key_operation: str
    recent_memory_kind: str

    def validate(self, loaded: LoadedFrameSampAMArtifact) -> None:
        loaded.validate()
        queries = np.asarray(self.queries_post_rope_pre_scale)
        positions = np.asarray(self.query_positions)
        recent_keys = np.asarray(self.recent_keys_post_rope)
        recent_values = np.asarray(self.recent_values_post_projection)
        recent_positions = np.asarray(self.recent_physical_positions)
        recent_mask = np.asarray(self.recent_token_mask)
        if queries.ndim != 3 or queries.shape[0] != loaded.manifest.query_head_count:
            raise ValueError("runtime queries must be [4 heads, action tokens, head_dim]")
        if positions.shape != (queries.shape[1],) or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError("runtime query_positions must be one integer per action token")
        expected_positions = np.arange(
            loaded.manifest.query_position_offset,
            loaded.manifest.query_position_offset + queries.shape[1],
            dtype=np.int64,
        )
        if not np.array_equal(positions.astype(np.int64, copy=False), expected_positions):
            raise ValueError("runtime action-query positions do not preserve teacher offset 512")
        if recent_keys.ndim != 2 or recent_values.ndim != 2 or not recent_keys.shape[1] or not recent_values.shape[1]:
            raise ValueError("runtime recent K/V must be rank-2 matrices with nonzero feature width")
        if recent_keys.shape[0] != recent_values.shape[0]:
            raise ValueError("runtime recent K/V token counts differ")
        if loaded.manifest.memory_partition_kind == MEMORY_PARTITION_KIND and recent_keys.shape[0] != 0:
            raise ValueError(
                "v1 artifact compacts all valid FrameSamp tokens and therefore requires R=0; "
                "compact-old + recent needs a disjoint-partition artifact schema"
            )
        if recent_positions.shape != (recent_keys.shape[0],) or not np.issubdtype(recent_positions.dtype, np.integer):
            raise ValueError("runtime recent physical positions must be one integer per token")
        if recent_mask.shape != (recent_keys.shape[0],) or recent_mask.dtype != np.bool_:
            raise ValueError("runtime recent token mask must be Boolean with one entry per token")
        if recent_mask.size and np.any(np.diff(recent_mask.astype(np.int8)) > 0):
            raise ValueError("runtime recent mask must be a valid prefix followed by right padding")
        physical_positions = recent_positions.astype(np.int64, copy=False)
        valid_positions = physical_positions[recent_mask]
        if valid_positions.size and (
            np.any(valid_positions < 0)
            or np.any(valid_positions >= loaded.manifest.logical_key_position_span)
            or np.any(np.diff(valid_positions) <= 0)
        ):
            raise ValueError("valid recent positions must be unique increasing teacher slots in 0..511")
        if np.any(physical_positions[~recent_mask] != 0):
            raise ValueError("right-padded recent positions must use canonical safe sentinel 0")
        if queries.shape[-1] != loaded.artifact.keys.shape[1] or recent_keys.shape[1] != queries.shape[-1]:
            raise ValueError("runtime query/recent/compact key dimensions differ")
        if recent_values.shape[1] != loaded.artifact.values.shape[1]:
            raise ValueError("runtime recent/compact value dimensions differ")
        if not all(
            np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in (queries, recent_keys, recent_values)
        ):
            raise ValueError("runtime attention inputs contain non-finite values")
        if (
            self.query_tap_stage != QUERY_TAP_STAGE
            or self.recent_key_tap_stage != KEY_TAP_STAGE
            or self.recent_value_tap_stage != VALUE_TAP_STAGE
        ):
            raise ValueError("runtime Q/K/V inputs are not at the sealed post-projection/RoPE stages")
        if self.compact_key_operation != COMPACT_KEY_OPERATION:
            raise ValueError("compact keys must not be projected or RoPE-applied again")
        if self.recent_memory_kind != RECENT_MEMORY_KIND:
            raise ValueError("recent memory must remain exact/uncompressed")


def attend_framesamp_am_runtime(
    loaded: LoadedFrameSampAMArtifact,
    inputs: FrameSampAMRuntimeInputs,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference runtime for v1 compact-all artifacts (raw-recent must be empty)."""

    inputs.validate(loaded)
    queries = np.asarray(inputs.queries_post_rope_pre_scale)
    flat_queries = queries.reshape(-1, queries.shape[-1])
    recent_mask = np.asarray(inputs.recent_token_mask)
    output, log_mass = attend_compact_old_and_recent(
        flat_queries,
        loaded.artifact,
        np.asarray(inputs.recent_keys_post_rope)[recent_mask],
        np.asarray(inputs.recent_values_post_projection)[recent_mask],
    )
    return (
        output.reshape(queries.shape[0], queries.shape[1], output.shape[-1]),
        log_mass.reshape(queries.shape[0], queries.shape[1]),
    )
