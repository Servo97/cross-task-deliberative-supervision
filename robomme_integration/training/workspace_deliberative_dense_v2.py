"""Train the immutable dense/multi-point v2 workspace representation with VISReg only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .single_task import TASK_EPISODES, task_manifest_sha256
from .workspace_deliberative import (
    FEATURE_DIM,
    INPUT_DIM,
    MAX_EVENTS,
    STATE_DIM,
    _run_config,
    encode,
    init_params,
    predict,
    sha256_file,
    train,
)
from .workspace_supervision_dense_v2 import (
    ARTIFACT,
    MAX_TARGET_ROLES,
    SCHEMA_VERSION,
    TARGET_SEMANTICS,
)

PROTOCOL = "robomme_move_workspace_dense_multipoint_visreg_v2"
VISREG_SLICES = 128


def _manifest_without_hash(manifest: dict) -> bytes:
    value = dict(manifest)
    value.pop("manifest_sha256", None)
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def verify_dense_supervision_manifest(
    root: str | Path,
    task_name: str,
    *,
    verify_hashes: bool,
) -> tuple[Path, dict]:
    task_root = Path(root) / task_name
    path = task_root / "MANIFEST.json"
    if not path.is_file():
        raise ValueError(f"dense v2 supervision manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "task_name": task_name,
        "task_manifest_sha256": task_manifest_sha256(task_name),
        "episodes": list(TASK_EPISODES[task_name]),
        "feature_dim": FEATURE_DIM,
        "patch_grid": 8,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"dense v2 supervision {key} mismatch")
    if hashlib.sha256(_manifest_without_hash(manifest)).hexdigest() != manifest.get("manifest_sha256"):
        raise ValueError("dense v2 supervision manifest SHA-256 mismatch")
    semantics = manifest.get("target_semantics", {})
    if (
        semantics.get("name") != TARGET_SEMANTICS
        or semantics.get("role_order") != "left_to_right_coordinate_occurrence_in_grounded_text"
        or semantics.get("max_target_roles") != MAX_TARGET_ROLES
    ):
        raise ValueError("dense v2 target semantics drifted")
    if manifest.get("causal_training_contract") != {
        "encoder_inputs": "frame_mean_and_state_at_or_before_decision_t_only",
        "targets": "ordered_grounded_roles_anchored_at_or_before_t_minus_min_lag",
        "grounded_text_or_coordinates_in_encoder": False,
        "future_frames_in_encoder": False,
        "uses_labels_at_inference": False,
    }:
        raise ValueError("dense v2 causal/no-leakage contract drifted")
    records = manifest.get("records", [])
    if [int(record["episode"]) for record in records] != list(TASK_EPISODES[task_name]):
        raise ValueError("dense v2 supervision record order/set mismatch")
    for record in records:
        roles = int(record.get("target_roles", -1))
        if not 1 <= roles <= MAX_TARGET_ROLES:
            raise ValueError("dense v2 target-role count outside supported range")
        relative = f"episode_{int(record['episode'])}/supervision.npz"
        file_path = task_root / relative
        if record.get("path") != relative or not file_path.is_file():
            raise ValueError(f"dense v2 supervision file is missing: {file_path}")
        if file_path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"dense v2 supervision size mismatch: {file_path}")
        if verify_hashes and sha256_file(file_path) != record.get("sha256"):
            raise ValueError(f"dense v2 supervision hash mismatch: {file_path}")
    return task_root, manifest


class DenseWorkspaceBatchSampler:
    """Causal sampler whose output target slots are ordered grounded roles, not segments."""

    def __init__(
        self,
        root: str | Path,
        task_name: str,
        *,
        seed: int,
        min_lag: int,
        future_delta: int,
        history_stride: int,
        max_history: int,
        verify_hashes: bool,
        cache_episodes: int = 16,
    ):
        if min_lag < 1 or future_delta < 1 or history_stride < 1 or max_history < 1:
            raise ValueError("invalid dense v2 history/target geometry")
        self.task_root, self.manifest = verify_dense_supervision_manifest(root, task_name, verify_hashes=verify_hashes)
        self.task_name = task_name
        self.min_lag = min_lag
        self.future_delta = future_delta
        self.history_stride = history_stride
        self.max_history = max_history
        self.cache_episodes = cache_episodes
        self.records = {int(record["episode"]): record for record in self.manifest["records"]}
        episodes = np.asarray(self.manifest["episodes"], dtype=np.int64)
        shuffled = np.random.default_rng(seed).permutation(episodes)
        val_count = max(1, len(episodes) // 10)
        self.val_episodes = tuple(sorted(int(value) for value in shuffled[:val_count]))
        self.train_episodes = tuple(sorted(int(value) for value in shuffled[val_count:]))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.state_mean, self.state_std = self._state_stats(self.train_episodes)
        self.train_groups = self._candidate_groups(self.train_episodes)
        self.val_groups = self._candidate_groups(self.val_episodes)
        if not self.train_groups or not self.val_groups:
            raise ValueError("dense v2 train/validation candidate groups are empty")

    def _load(self, episode: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(episode, None)
        if cached is None:
            path = self.task_root / self.records[episode]["path"]
            with np.load(path) as source:
                cached = {key: source[key] for key in source.files}
            roles = int(self.records[episode]["target_roles"])
            if cached["frame_mean_f16"].shape != (
                int(self.records[episode]["steps"]),
                FEATURE_DIM,
            ):
                raise ValueError(f"dense v2 frame shape mismatch for episode {episode}")
            if (
                cached["target_feature_f16"].shape != (roles, FEATURE_DIM)
                or cached["target_attention_f32"].shape != (roles, 64)
                or cached["target_anchor_i32"].shape != (roles,)
            ):
                raise ValueError(f"dense v2 target shape mismatch for episode {episode}")
            anchors = np.asarray(cached["target_anchor_i32"], dtype=np.int64)
            if np.any(np.diff(anchors) < 0):
                raise ValueError(f"dense v2 targets are not chronological for episode {episode}")
            # Within a segment, role order must be exact 0..P-1.  This catches producer/order drift.
            for event in np.unique(cached["target_event_i16"]):
                roles_for_event = cached["target_role_i16"][cached["target_event_i16"] == event]
                if not np.array_equal(roles_for_event, np.arange(len(roles_for_event))):
                    raise ValueError(f"dense v2 target role ordering drift for episode {episode}")
        self._cache[episode] = cached
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return cached

    def _state_stats(self, episodes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        count = 0
        total = np.zeros((STATE_DIM,), dtype=np.float64)
        square = np.zeros((STATE_DIM,), dtype=np.float64)
        for episode in episodes:
            state = np.asarray(self._load(episode)["state_f32"], dtype=np.float64)
            total += state.sum(axis=0)
            square += np.square(state).sum(axis=0)
            count += len(state)
        mean = total / count
        variance = np.maximum(square / count - np.square(mean), 1e-8)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    def _candidate_groups(self, episodes: tuple[int, ...]) -> dict[int, list[tuple[int, int]]]:
        groups: dict[int, list[tuple[int, int]]] = {}
        for episode in episodes:
            arrays = self._load(episode)
            anchors = np.asarray(arrays["target_anchor_i32"], dtype=np.int64)
            stop = int(self.records[episode]["steps"]) - self.future_delta
            for decision in range(self.min_lag, stop, self.history_stride):
                count = int(np.searchsorted(anchors, decision - self.min_lag, side="right"))
                if count:
                    groups.setdefault(count, []).append((episode, decision))
        return {
            count: rows
            for count, rows in groups.items()
            if count <= MAX_EVENTS and len({episode for episode, _ in rows}) >= 2
        }

    def history(self, episode: int, decision: int, *, mask_current: bool) -> tuple[np.ndarray, np.ndarray]:
        arrays = self._load(episode)
        if not 0 <= decision < len(arrays["frame_mean_f16"]):
            raise IndexError(f"decision {decision} outside episode {episode}")
        start = decision % self.history_stride
        indices = np.arange(start, decision + 1, self.history_stride, dtype=np.int64)[-self.max_history :]
        length = len(indices)
        history = np.zeros((self.max_history, INPUT_DIM), dtype=np.float32)
        mask = np.zeros((self.max_history,), dtype=np.bool_)
        history[:length, :FEATURE_DIM] = np.asarray(arrays["frame_mean_f16"][indices], dtype=np.float32)
        history[:length, FEATURE_DIM:] = (
            np.asarray(arrays["state_f32"][indices], dtype=np.float32) - self.state_mean
        ) / self.state_std
        mask[:length] = True
        if mask_current:
            history[length - 1] = 0.0
            mask[length - 1] = False
        return history, mask

    def _one(self, episode: int, decision: int, *, mask_current: bool) -> dict[str, np.ndarray]:
        arrays = self._load(episode)
        history, history_mask = self.history(episode, decision, mask_current=mask_current)
        anchors = np.asarray(arrays["target_anchor_i32"], dtype=np.int64)
        count = int(np.searchsorted(anchors, decision - self.min_lag, side="right"))
        target = np.zeros((MAX_EVENTS, FEATURE_DIM), dtype=np.float32)
        target_attention = np.zeros((MAX_EVENTS, 64), dtype=np.float32)
        presence = np.zeros((MAX_EVENTS,), dtype=np.float32)
        target[:count] = np.asarray(arrays["target_feature_f16"][:count], dtype=np.float32)
        target_attention[:count] = np.asarray(arrays["target_attention_f32"][:count], dtype=np.float32)
        presence[:count] = 1.0
        future = np.asarray(arrays["frame_mean_f16"][decision + self.future_delta], dtype=np.float32)
        return {
            "history": history,
            "history_mask": history_mask,
            "event_target": target,
            "event_attention": target_attention,
            "event_presence": presence,
            "future_target": future,
        }

    def sample(
        self,
        *,
        split: str,
        batch_size: int,
        rng: np.random.Generator,
        mask_probability: float,
    ) -> dict:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("dense v2 batch_size must be positive and even")
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0,1]")
        groups = self.train_groups if split == "train" else self.val_groups
        counts = tuple(sorted(groups))
        rows = []
        for _ in range(batch_size // 2):
            count = int(rng.choice(counts))
            candidates = groups[count]
            first = candidates[int(rng.integers(len(candidates)))]
            second = first
            for _ in range(64):
                proposal = candidates[int(rng.integers(len(candidates)))]
                if proposal[0] != first[0]:
                    second = proposal
                    break
            if second[0] == first[0]:
                raise RuntimeError(f"could not form cross-episode dense v2 pair for count {count}")
            for episode, decision in (first, second):
                rows.append(
                    self._one(
                        episode,
                        decision,
                        mask_current=bool(rng.random() < mask_probability),
                    )
                )
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def visreg_loss(
    z,
    rng,
    *,
    num_slices: int = VISREG_SLICES,
    scale_weight: float = 1.0,
    shape_weight: float = 1.0,
    center_weight: float = 1.0,
    eps: float = 1e-4,
):
    """Sample-count-invariant VISReg, matching the v4 policy implementation."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    n, width = z.shape
    mean = z.mean(axis=0)
    center = jnp.mean(mean**2)
    centered = z - mean
    std = jnp.sqrt(jnp.mean(centered**2, axis=0) + eps)
    scale = jnp.mean((1.0 - std) ** 2)
    normalized = centered / jax.lax.stop_gradient(std)
    directions = jax.random.normal(rng, (width, num_slices), dtype=z.dtype)
    directions /= jnp.linalg.norm(directions, axis=0, keepdims=True) + 1e-8
    projection = jnp.sort(normalized @ directions, axis=0)
    quantiles = ndtri(jnp.arange(1, n + 1, dtype=jnp.float32) / (n + 1.0)).astype(projection.dtype)[:, None]
    shape = jnp.mean((projection - quantiles) ** 2)
    return scale_weight * scale + shape_weight * shape + center_weight * center


def init_params_dense_v2(rng):
    """Append a dense 8x8 attention decoder without changing the legacy parameter family."""
    import jax
    import jax.numpy as jnp

    params = init_params(rng)
    key = jax.random.fold_in(rng, 0x44563241)
    params["dense_attention_w"] = jax.random.normal(key, (FEATURE_DIM, 64), dtype=jnp.float32) * math.sqrt(
        2.0 / (FEATURE_DIM + 64)
    )
    params["dense_attention_b"] = jnp.zeros((64,), dtype=jnp.float32)
    return params


def predict_dense_v2(params, omega):
    reconstruction, occurrence, future = predict(params, omega)
    attention_logits = reconstruction @ params["dense_attention_w"] + params["dense_attention_b"]
    return reconstruction, occurrence, future, attention_logits


def dense_v2_loss_and_metrics(params, batch, *, weights: dict[str, float]):
    import jax
    import jax.numpy as jnp
    import optax

    if weights.get("sigreg", None) != 0.0:
        raise ValueError("dense v2 forbids SIGReg; expected exact weight 0")
    omega = encode(params, batch["history"], batch["history_mask"])
    reconstruction, occurrence, future, attention_logits = predict_dense_v2(params, omega)
    presence = batch["event_presence"]

    def unit(value):
        return value * jax.lax.rsqrt(jnp.sum(jnp.square(value), axis=-1, keepdims=True) + 1e-6)

    cosine = jnp.sum(unit(reconstruction) * unit(batch["event_target"]), axis=-1)
    recon = jnp.sum((1.0 - cosine) * presence) / jnp.maximum(jnp.sum(presence), 1.0)
    occurrence_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(occurrence, presence))
    attention_each = -jnp.sum(batch["event_attention"] * jax.nn.log_softmax(attention_logits, axis=-1), axis=-1)
    attention_loss = jnp.sum(attention_each * presence) / jnp.maximum(jnp.sum(presence), 1.0)
    jepa = jnp.mean(1.0 - jnp.sum(unit(future) * unit(batch["future_target"]), axis=-1))
    visreg = visreg_loss(
        omega,
        jax.random.key(0),
        num_slices=int(weights["visreg_slices"]),
        scale_weight=float(weights["visreg_scale"]),
        shape_weight=float(weights["visreg_shape"]),
        center_weight=float(weights["visreg_center"]),
    )
    loss = (
        recon
        + weights["occ"] * occurrence_loss
        + weights["attention"] * attention_loss
        + weights["jepa"] * jepa
        + weights["visreg"] * visreg
    )
    return loss, {
        "loss": loss,
        "recon": recon,
        "occ": occurrence_loss,
        "attention": attention_loss,
        "jepa": jepa,
        "visreg": visreg,
        "sigreg": jnp.asarray(0.0, dtype=loss.dtype),
        "omega_std": jnp.mean(jnp.std(omega, axis=0)),
    }


def dense_v2_run_config(args, sampler: DenseWorkspaceBatchSampler) -> dict:
    result = _run_config(args, sampler)
    result.update(
        {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "implementation_sha256": sha256_file(Path(__file__)),
            "supervision_artifact": ARTIFACT,
            "target_semantics": TARGET_SEMANTICS,
            "ema_decay": args.ema_decay,
        }
    )
    result["loss_weights"] = {
        "occ": args.occ_weight,
        "attention": args.attention_weight,
        "jepa": args.jepa_weight,
        "sigreg": 0.0,
        "visreg": args.visreg_weight,
        "visreg_slices": args.visreg_slices,
        "visreg_scale": args.visreg_scale_weight,
        "visreg_shape": args.visreg_shape_weight,
        "visreg_center": args.visreg_center_weight,
    }
    result["optimizer_contract"] = {
        "optimizer": "AdamW",
        "peak_lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "global_gradient_clip": args.clip_gradient_norm,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASK_EPISODES))
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--clip-gradient-norm", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--min-lag", type=int, default=40)
    parser.add_argument("--future-delta", type=int, default=20)
    parser.add_argument("--history-stride", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=128)
    parser.add_argument("--mask-probability", type=float, default=0.2)
    parser.add_argument("--occ-weight", type=float, default=0.1)
    parser.add_argument("--attention-weight", type=float, default=0.1)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--visreg-weight", type=float, default=0.05)
    parser.add_argument("--visreg-slices", type=int, default=128)
    parser.add_argument("--visreg-scale-weight", type=float, default=1.0)
    parser.add_argument("--visreg-shape-weight", type=float, default=1.0)
    parser.add_argument("--visreg-center-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-checkpoints", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-devices", type=int, default=4)
    parser.add_argument("--skip-supervision-hashes", action="store_true")
    parser.add_argument("--one-step-canary", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.sigreg_weight != 0.0:
        raise SystemExit("dense v2 is VISReg-only and requires --sigreg-weight=0")
    if (
        args.visreg_weight != 0.05
        or args.visreg_slices != 128
        or (args.visreg_scale_weight, args.visreg_shape_weight, args.visreg_center_weight) != (1.0, 1.0, 1.0)
    ):
        raise SystemExit("dense v2 VISReg contract requires 0.05, 128 slices, split 1/1/1")
    if args.ema_decay != 0.999:
        raise SystemExit("dense v2 requires EMA decay 0.999")
    if args.one_step_canary:
        if os.environ.get("WSM_MOVE_DENSE_V2_CANARY") != "1":
            raise SystemExit("dense v2 one-step canary requires WSM_MOVE_DENSE_V2_CANARY=1")
    elif not args.cpu_smoke and os.environ.get("WSM_MOVE_DENSE_V2_REP_ALLOW_RUN") != "1":
        raise SystemExit("dense v2 production training requires its reviewed v2 run gate")
    train(
        args,
        sampler_class=DenseWorkspaceBatchSampler,
        loss_function=dense_v2_loss_and_metrics,
        run_config_builder=dense_v2_run_config,
        init_params_function=init_params_dense_v2,
    )


if __name__ == "__main__":
    main()
