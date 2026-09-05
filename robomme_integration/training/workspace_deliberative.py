"""Train and materialize a causal RoboMME workspace encoder with long-lag supervision.

The representation is deliberately small and TPU-native: frozen per-frame Pi/SigLIP features and
proprioception update a 512-wide recurrent workspace state ``omega_t``.  Training requires later
states to decode *historical* point-grounded patch values, predicts a future frozen feature (JEPA),
and applies paired-progress/variance/covariance regularization (SigReg).  Grounded subgoals are used
only to build training targets; neither labels nor future frames are encoder inputs or policy inputs.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .single_task import TASK_EPISODES, task_manifest_sha256
from .workspace_supervision_cache import FEATURE_DIM, sha256_file

OMEGA_DIM = 512
STATE_DIM = 8
INPUT_DIM = FEATURE_DIM + STATE_DIM
MAX_EVENTS = 16


def _manifest_without_hash(manifest: dict) -> bytes:
    value = dict(manifest)
    value.pop("manifest_sha256", None)
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def verify_supervision_manifest(root: str | Path, task_name: str, *, verify_hashes: bool) -> tuple[Path, dict]:
    task_root = Path(root) / task_name
    path = task_root / "MANIFEST.json"
    if not path.is_file():
        raise ValueError(f"WSM supervision manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "artifact": "robomme_wsm_long_lag_supervision",
        "task_name": task_name,
        "task_manifest_sha256": task_manifest_sha256(task_name),
        "episodes": list(TASK_EPISODES[task_name]),
        "feature_dim": FEATURE_DIM,
        "patch_grid": 8,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"WSM supervision {key} mismatch: {manifest.get(key)!r} != {value!r}")
    actual_manifest_sha = hashlib.sha256(_manifest_without_hash(manifest)).hexdigest()
    if manifest.get("manifest_sha256") != actual_manifest_sha:
        raise ValueError("WSM supervision manifest SHA-256 mismatch")
    causal = manifest.get("causal_training_contract", {})
    if causal != {
        "encoder_inputs": "frames_at_or_before_decision_t",
        "target": "event_anchor_at_or_before_t_minus_min_lag",
        "current_frame_masking_required": True,
        "uses_labels_at_inference": False,
    }:
        raise ValueError("WSM supervision causal contract drifted")
    records = manifest.get("records", [])
    if [int(record["episode"]) for record in records] != list(TASK_EPISODES[task_name]):
        raise ValueError("WSM supervision record order/set mismatch")
    for record in records:
        episode = int(record["episode"])
        relative = f"episode_{episode}/supervision.npz"
        if record.get("path") != relative:
            raise ValueError(f"WSM supervision path drift for episode {episode}")
        file_path = task_root / relative
        if not file_path.is_file() or file_path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"WSM supervision file missing/size mismatch: {file_path}")
        if verify_hashes and sha256_file(file_path) != record.get("sha256"):
            raise ValueError(f"WSM supervision hash mismatch for episode {episode}")
        if not 1 <= len(record.get("events", [])) <= MAX_EVENTS:
            raise ValueError(f"episode {episode} event count outside [1,{MAX_EVENTS}]")
    return task_root, manifest


class WorkspaceBatchSampler:
    """Deterministic episode-disjoint train/validation sampler with paired progress."""

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
            raise ValueError("invalid WSM history/target geometry")
        self.task_root, self.manifest = verify_supervision_manifest(
            root,
            task_name,
            verify_hashes=verify_hashes,
        )
        self.task_name = task_name
        self.min_lag = min_lag
        self.future_delta = future_delta
        self.history_stride = history_stride
        self.max_history = max_history
        self.cache_episodes = cache_episodes
        self.records = {int(record["episode"]): record for record in self.manifest["records"]}
        episodes = np.asarray(self.manifest["episodes"], dtype=np.int64)
        split_rng = np.random.default_rng(seed)
        shuffled = split_rng.permutation(episodes)
        val_count = max(1, len(episodes) // 10)
        self.val_episodes = tuple(sorted(int(value) for value in shuffled[:val_count]))
        self.train_episodes = tuple(sorted(int(value) for value in shuffled[val_count:]))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.state_mean, self.state_std = self._state_stats(self.train_episodes)
        self.train_groups = self._candidate_groups(self.train_episodes)
        self.val_groups = self._candidate_groups(self.val_episodes)
        if not self.train_groups or not self.val_groups:
            raise ValueError("WSM train/validation candidate groups are empty")

    def _load(self, episode: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(episode, None)
        if cached is None:
            path = self.task_root / self.records[episode]["path"]
            with np.load(path) as source:
                cached = {key: source[key] for key in source.files}
            if cached["frame_mean_f16"].shape != (int(self.records[episode]["steps"]), FEATURE_DIM):
                raise ValueError(f"WSM frame shape mismatch for episode {episode}")
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
            record = self.records[episode]
            anchors = np.asarray([event["anchor_step"] for event in record["events"]], dtype=np.int64)
            stop = int(record["steps"]) - self.future_delta
            for decision in range(self.min_lag, stop, self.history_stride):
                count = int(np.searchsorted(anchors, decision - self.min_lag, side="right"))
                if count:
                    groups.setdefault(count, []).append((episode, decision))
        # Paired progress must span demonstrations, not two times from one episode.
        return {
            count: values
            for count, values in groups.items()
            if len({episode for episode, _ in values}) >= 2 and count <= MAX_EVENTS
        }

    def _one(self, episode: int, decision: int, *, mask_current: bool) -> dict[str, np.ndarray]:
        arrays = self._load(episode)
        history, mask = self.history(episode, decision, mask_current=mask_current)
        anchors = np.asarray(arrays["event_anchor_i32"], dtype=np.int64)
        count = int(np.searchsorted(anchors, decision - self.min_lag, side="right"))
        targets = np.zeros((MAX_EVENTS, FEATURE_DIM), dtype=np.float32)
        presence = np.zeros((MAX_EVENTS,), dtype=np.float32)
        targets[:count] = np.asarray(arrays["event_feature_f16"][:count], dtype=np.float32)
        presence[:count] = 1.0
        future = np.asarray(arrays["frame_mean_f16"][decision + self.future_delta], dtype=np.float32)
        return {
            "history": history,
            "history_mask": mask,
            "event_target": targets,
            "event_presence": presence,
            "future_target": future,
        }

    def history(self, episode: int, decision: int, *, mask_current: bool) -> tuple[np.ndarray, np.ndarray]:
        """Build the exact causal encoder input used by training and omega materialization."""
        arrays = self._load(episode)
        if not 0 <= decision < len(arrays["frame_mean_f16"]):
            raise IndexError(f"decision {decision} outside episode {episode}")
        start = decision % self.history_stride
        indices = np.arange(start, decision + 1, self.history_stride, dtype=np.int64)[-self.max_history :]
        length = len(indices)
        history = np.zeros((self.max_history, INPUT_DIM), dtype=np.float32)
        mask = np.zeros((self.max_history,), dtype=np.bool_)
        frames = np.asarray(arrays["frame_mean_f16"][indices], dtype=np.float32)
        state = (np.asarray(arrays["state_f32"][indices], dtype=np.float32) - self.state_mean) / self.state_std
        history[:length, :FEATURE_DIM] = frames
        history[:length, FEATURE_DIM:] = state
        mask[:length] = True
        if mask_current:
            history[length - 1] = 0.0
            mask[length - 1] = False
        return history, mask

    def sample(self, *, split: str, batch_size: int, rng: np.random.Generator, mask_probability: float) -> dict:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("WSM batch_size must be positive and even for paired-progress SigReg")
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
            for _attempt in range(64):
                proposal = candidates[int(rng.integers(len(candidates)))]
                if proposal[0] != first[0]:
                    second = proposal
                    break
            if second[0] == first[0]:
                raise RuntimeError(f"could not form cross-episode WSM pair for progress count {count}")
            for episode, decision in (first, second):
                rows.append(
                    self._one(
                        episode,
                        decision,
                        mask_current=bool(rng.random() < mask_probability),
                    )
                )
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _normal(rng, shape, fan_in: int, fan_out: int):
    import jax

    return jax.random.normal(rng, shape, dtype=np.float32) * math.sqrt(2.0 / (fan_in + fan_out))


def init_params(rng):
    import jax
    import jax.numpy as jnp

    keys = iter(jax.random.split(rng, 14))
    params = {
        "input_w": _normal(next(keys), (INPUT_DIM, OMEGA_DIM), INPUT_DIM, OMEGA_DIM),
        "input_b": jnp.zeros((OMEGA_DIM,), dtype=jnp.float32),
        "wz": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "uz": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "bz": jnp.zeros((OMEGA_DIM,), dtype=jnp.float32),
        "wr": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "ur": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "br": jnp.zeros((OMEGA_DIM,), dtype=jnp.float32),
        "wn": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "un": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "bn": jnp.zeros((OMEGA_DIM,), dtype=jnp.float32),
        "slots": jax.random.normal(next(keys), (MAX_EVENTS, OMEGA_DIM), dtype=jnp.float32) * 0.02,
        "decoder_w": _normal(next(keys), (OMEGA_DIM, OMEGA_DIM), OMEGA_DIM, OMEGA_DIM),
        "decoder_b": jnp.zeros((OMEGA_DIM,), dtype=jnp.float32),
        "recon_w": _normal(next(keys), (OMEGA_DIM, FEATURE_DIM), OMEGA_DIM, FEATURE_DIM),
        "recon_b": jnp.zeros((FEATURE_DIM,), dtype=jnp.float32),
        "occ_w": _normal(next(keys), (OMEGA_DIM, 1), OMEGA_DIM, 1),
        "occ_b": jnp.zeros((1,), dtype=jnp.float32),
        "jepa_w": _normal(next(keys), (OMEGA_DIM, FEATURE_DIM), OMEGA_DIM, FEATURE_DIM),
        "jepa_b": jnp.zeros((FEATURE_DIM,), dtype=jnp.float32),
    }
    return params


def _layer_norm(value, epsilon: float = 1e-5):
    import jax.numpy as jnp

    return (value - jnp.mean(value, axis=-1, keepdims=True)) * jax_lax_rsqrt(
        jnp.var(value, axis=-1, keepdims=True) + epsilon
    )


def jax_lax_rsqrt(value):
    from jax import lax

    return lax.rsqrt(value)


def encode(params, history, history_mask):
    import jax
    import jax.numpy as jnp

    normalized = _layer_norm(history)
    embedded = jax.nn.silu(normalized @ params["input_w"] + params["input_b"])
    time_major = jnp.swapaxes(embedded, 0, 1)
    mask_major = jnp.swapaxes(history_mask, 0, 1)
    initial = jnp.zeros((history.shape[0], OMEGA_DIM), dtype=jnp.float32)

    def update(hidden, values):
        current, valid = values
        z = jax.nn.sigmoid(current @ params["wz"] + hidden @ params["uz"] + params["bz"])
        r = jax.nn.sigmoid(current @ params["wr"] + hidden @ params["ur"] + params["br"])
        candidate = jnp.tanh(current @ params["wn"] + (r * hidden) @ params["un"] + params["bn"])
        proposed = (1.0 - z) * candidate + z * hidden
        hidden = jnp.where(valid[:, None], proposed, hidden)
        return hidden, hidden

    final, _ = jax.lax.scan(update, initial, (time_major, mask_major))
    return _layer_norm(final)


def predict(params, omega):
    import jax

    slots = _layer_norm(omega[:, None, :] + params["slots"][None, :, :])
    hidden = jax.nn.silu(slots @ params["decoder_w"] + params["decoder_b"])
    reconstruction = hidden @ params["recon_w"] + params["recon_b"]
    occurrence = (hidden @ params["occ_w"] + params["occ_b"])[..., 0]
    future = omega @ params["jepa_w"] + params["jepa_b"]
    return reconstruction, occurrence, future


def _unit(value):
    import jax.numpy as jnp

    return value * jax_lax_rsqrt(jnp.sum(jnp.square(value), axis=-1, keepdims=True) + 1e-6)


def loss_and_metrics(params, batch, *, weights: dict[str, float]):
    import jax.numpy as jnp
    import optax

    omega = encode(params, batch["history"], batch["history_mask"])
    reconstruction, occurrence, future = predict(params, omega)
    presence = batch["event_presence"]
    cosine = jnp.sum(_unit(reconstruction) * _unit(batch["event_target"]), axis=-1)
    recon_loss = jnp.sum((1.0 - cosine) * presence) / jnp.maximum(jnp.sum(presence), 1.0)
    occurrence_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(occurrence, presence))
    jepa_loss = jnp.mean(1.0 - jnp.sum(_unit(future) * _unit(batch["future_target"]), axis=-1))

    paired = omega.reshape((-1, 2, OMEGA_DIM))
    invariance = jnp.mean(jnp.square(_unit(paired[:, 0]) - _unit(paired[:, 1])))
    standard_deviation = jnp.sqrt(jnp.var(omega, axis=0) + 1e-4)
    variance = jnp.mean(jnp.maximum(0.0, 1.0 - standard_deviation))
    centered = omega - jnp.mean(omega, axis=0, keepdims=True)
    covariance_matrix = centered.T @ centered / jnp.maximum(omega.shape[0] - 1, 1)
    covariance = jnp.sum(jnp.square(covariance_matrix)) - jnp.sum(jnp.square(jnp.diag(covariance_matrix)))
    covariance = covariance / (OMEGA_DIM * (OMEGA_DIM - 1))
    sigreg = invariance + 0.1 * variance + 0.01 * covariance
    loss = recon_loss + weights["occ"] * occurrence_loss + weights["jepa"] * jepa_loss + weights["sigreg"] * sigreg
    metrics = {
        "loss": loss,
        "recon": recon_loss,
        "occ": occurrence_loss,
        "jepa": jepa_loss,
        "sigreg": sigreg,
        "omega_std": jnp.mean(standard_deviation),
    }
    return loss, metrics


def _train_step(params, opt_state, batch, *, optimizer, weights, loss_function=loss_and_metrics):
    import jax
    import optax

    (loss, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(
        params,
        batch,
        weights=weights,
    )
    gradients = jax.lax.pmean(gradients, "devices")
    metrics = jax.lax.pmean(metrics, "devices")
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    metrics = {**metrics, "grad_norm": optax.global_norm(gradients), "loss_check": metrics["loss"]}
    return params, opt_state, metrics


def _train_step_ema(
    params,
    opt_state,
    ema_params,
    batch,
    *,
    optimizer,
    weights,
    loss_function,
    ema_decay,
):
    import jax

    params, opt_state, metrics = _train_step(
        params,
        opt_state,
        batch,
        optimizer=optimizer,
        weights=weights,
        loss_function=loss_function,
    )
    ema_params = jax.tree.map(
        lambda average, current: ema_decay * average + (1.0 - ema_decay) * current,
        ema_params,
        params,
    )
    return params, opt_state, ema_params, metrics


def _eval_step(params, batch, *, weights, loss_function=loss_and_metrics):
    import jax

    _, metrics = loss_function(params, batch, weights=weights)
    return jax.lax.pmean(metrics, "devices")


def _reshape_for_devices(batch: dict, device_count: int) -> dict:
    size = next(iter(batch.values())).shape[0]
    if size % device_count:
        raise ValueError(f"batch size {size} is not divisible by {device_count}")
    return {key: value.reshape((device_count, size // device_count, *value.shape[1:])) for key, value in batch.items()}


def _manager(directory: Path, *, max_to_keep: int):
    import orbax.checkpoint as ocp

    directory.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        directory,
        item_handlers={"state": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=False),
    )


def checkpoint_completion_payload(
    *,
    step: int,
    run_config_sha256: str,
    embedded_hashes: dict[str, str],
    parameter_source: str | None = None,
    ema_decay: float | None = None,
) -> dict:
    value = {
        "schema_version": 1,
        "step": step,
        "run_config_sha256": run_config_sha256,
        "embedded_sha256": embedded_hashes,
    }
    if parameter_source == "ema":
        if ema_decay is None:
            raise ValueError("EMA checkpoint completion requires ema_decay")
        value["parameter_source"] = parameter_source
        value["ema_decay"] = ema_decay
    elif parameter_source is not None:
        raise ValueError(f"unknown checkpoint parameter source {parameter_source!r}")
    return value


def _run_config(args, sampler: WorkspaceBatchSampler) -> dict:
    return {
        "schema_version": 1,
        "implementation_sha256": sha256_file(Path(__file__)),
        "task": args.task,
        "task_manifest_sha256": task_manifest_sha256(args.task),
        "supervision_manifest_sha256": sampler.manifest["manifest_sha256"],
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "expected_devices": args.expected_devices,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "clip_gradient_norm": args.clip_gradient_norm,
        "min_lag": args.min_lag,
        "future_delta": args.future_delta,
        "history_stride": args.history_stride,
        "max_history": args.max_history,
        "mask_probability": args.mask_probability,
        "omega_dim": OMEGA_DIM,
        "max_events": MAX_EVENTS,
        "max_checkpoints": args.max_checkpoints,
        "log_interval": args.log_interval,
        "val_interval": args.val_interval,
        "val_batches": args.val_batches,
        "save_interval": args.save_interval,
        "loss_weights": {"occ": args.occ_weight, "jepa": args.jepa_weight, "sigreg": args.sigreg_weight},
        "train_episodes": list(sampler.train_episodes),
        "validation_episodes": list(sampler.val_episodes),
        "state_mean": sampler.state_mean.tolist(),
        "state_std": sampler.state_std.tolist(),
    }


def train(
    args,
    *,
    sampler_class=WorkspaceBatchSampler,
    loss_function=loss_and_metrics,
    run_config_builder=_run_config,
    init_params_function=init_params,
) -> None:
    import jax
    import optax

    if args.task not in TASK_EPISODES:
        raise ValueError(f"unknown task {args.task!r}")
    devices = jax.devices()
    if len(devices) != args.expected_devices:
        raise SystemExit(f"expected {args.expected_devices} JAX devices, found {len(devices)}")
    platforms = {device.platform for device in devices}
    if not args.cpu_smoke and platforms not in ({"tpu"}, {"gpu"}):
        raise SystemExit("WSM deliberative production training requires homogeneous TPU or GPU devices")
    if args.batch_size % len(devices):
        raise ValueError("batch size must be divisible by JAX device count")
    if args.steps < 1 or args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("steps must be positive and warmup_steps must be in [0, steps)")
    if min(args.log_interval, args.val_interval, args.val_batches, args.save_interval, args.max_checkpoints) < 1:
        raise ValueError("logging, validation, save, and retention intervals/counts must be positive")
    if args.val_interval % args.save_interval:
        raise ValueError("save_interval must divide val_interval so every scored checkpoint exists")
    required_checkpoints = math.ceil(args.steps / args.save_interval)
    if args.max_checkpoints < required_checkpoints:
        raise ValueError(
            f"max_checkpoints={args.max_checkpoints} cannot retain all {required_checkpoints} scored candidates"
        )
    sampler = sampler_class(
        args.supervision_root,
        args.task,
        seed=args.seed,
        min_lag=args.min_lag,
        future_delta=args.future_delta,
        history_stride=args.history_stride,
        max_history=args.max_history,
        verify_hashes=not args.skip_supervision_hashes,
    )
    run_config = run_config_builder(args, sampler)
    config_bytes = json.dumps(run_config, indent=2, sort_keys=True).encode() + b"\n"
    run_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    output = Path(args.output_root) / args.task
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "RUN_CONFIG.json"
    if config_path.exists() and config_path.read_bytes() != config_bytes:
        raise ValueError(f"immutable WSM run config collision at {config_path}")
    if not args.one_step_canary:
        temporary = output / ".RUN_CONFIG.json.incomplete"
        temporary.write_bytes(config_bytes)
        os.replace(temporary, config_path)

    weights = run_config["loss_weights"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_gradient_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    initial_params = init_params_function(jax.random.key(args.seed))
    initial_opt = optimizer.init(initial_params)
    ema_decay = getattr(args, "ema_decay", None)
    if ema_decay is not None and not 0.0 < float(ema_decay) < 1.0:
        raise ValueError("EMA decay must be in (0,1)")
    initial_ema = initial_params if ema_decay is not None else None
    manager = None if args.one_step_canary else _manager(output / "checkpoints", max_to_keep=args.max_checkpoints)
    start_step = 0
    if manager is not None and manager.latest_step() is not None:
        template = {"params": initial_params, "opt_state": initial_opt, "step": np.asarray(0, dtype=np.int64)}
        if ema_decay is not None:
            template["ema_params"] = initial_ema
        restored = manager.restore(manager.latest_step(), items={"state": template})["state"]
        initial_params, initial_opt, start_step = restored["params"], restored["opt_state"], int(restored["step"])
        if ema_decay is not None:
            initial_ema = restored["ema_params"]

    params = jax.device_put(initial_params)
    opt_state = jax.device_put(initial_opt)
    ema_params = None if initial_ema is None else jax.device_put(initial_ema)
    if ema_decay is None:
        train_step = jax.pmap(
            functools.partial(
                _train_step,
                optimizer=optimizer,
                weights=weights,
                loss_function=loss_function,
            ),
            axis_name="devices",
            in_axes=(None, None, 0),
            out_axes=(None, None, 0),
        )
    else:
        train_step = jax.pmap(
            functools.partial(
                _train_step_ema,
                optimizer=optimizer,
                weights=weights,
                loss_function=loss_function,
                ema_decay=float(ema_decay),
            ),
            axis_name="devices",
            in_axes=(None, None, None, 0),
            out_axes=(None, None, None, 0),
        )
    eval_step = jax.pmap(
        functools.partial(_eval_step, weights=weights, loss_function=loss_function),
        axis_name="devices",
        in_axes=(None, 0),
    )
    fixed_validation = tuple(
        _reshape_for_devices(
            sampler.sample(
                split="val",
                batch_size=args.batch_size,
                rng=np.random.default_rng(np.random.SeedSequence([args.seed, 0x56414C, index])),
                mask_probability=1.0,
            ),
            len(devices),
        )
        for index in range(args.val_batches)
    )
    best_value = math.inf
    best_step = None
    best_path = output / "BEST.json"
    if start_step and best_path.is_file():
        prior_best = json.loads(best_path.read_text(encoding="utf-8"))
        if prior_best.get("run_config_sha256") != run_config_sha256:
            raise ValueError("BEST.json belongs to a different immutable WSM run config")
        if prior_best.get("best_step") is not None:
            best_step = int(prior_best["best_step"])
            best_value = float(prior_best["best_score"])
    interval_started = time.perf_counter()
    for step in range(start_step, args.steps):
        step_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x54524149, step]))
        batch = sampler.sample(
            split="train",
            batch_size=args.batch_size,
            rng=step_rng,
            mask_probability=args.mask_probability,
        )
        batch = _reshape_for_devices(batch, len(devices))
        if ema_decay is None:
            params, opt_state, metrics = train_step(params, opt_state, batch)
        else:
            params, opt_state, ema_params, metrics = train_step(params, opt_state, ema_params, batch)
        if step == start_step or (step + 1) % args.log_interval == 0:
            values = {key: float(np.asarray(value[0])) for key, value in metrics.items()}
            if not all(math.isfinite(value) for value in values.values()) or values["grad_norm"] <= 0:
                raise FloatingPointError(f"invalid WSM metrics at step {step + 1}: {values}")
            elapsed = time.perf_counter() - interval_started
            print(
                f"[wsm-train] step={step + 1} "
                + " ".join(f"{key}={value:.5f}" for key, value in values.items())
                + f" interval_seconds={elapsed:.2f}",
                flush=True,
            )
            interval_started = time.perf_counter()
        if args.one_step_canary:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "task": args.task,
                        "run_config_sha256": run_config_sha256,
                        "device_count": len(devices),
                        "step": step + 1,
                        "wrote_checkpoint": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        if (step + 1) % args.val_interval == 0 or step + 1 == args.steps:
            aggregate: dict[str, list[float]] = {}
            for fixed_val in fixed_validation:
                validation = eval_step(params if ema_params is None else ema_params, fixed_val)
                for key, value in validation.items():
                    aggregate.setdefault(key, []).append(float(np.asarray(value[0])))
            val_values = {key: float(np.mean(values)) for key, values in aggregate.items()}
            score = val_values["recon"] + args.jepa_weight * val_values["jepa"]
            print(
                f"[wsm-val] step={step + 1} score={score:.6f} "
                + " ".join(f"{key}={value:.5f}" for key, value in val_values.items()),
                flush=True,
            )
            if score < best_value:
                best_value, best_step = score, step + 1
        if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
            state = {
                "params": jax.tree.map(np.asarray, params),
                "opt_state": jax.tree.map(np.asarray, opt_state),
                "step": np.asarray(step + 1, dtype=np.int64),
            }
            if ema_params is not None:
                state["ema_params"] = jax.tree.map(np.asarray, ema_params)
            manager.save(step + 1, {"state": state})
            manager.wait_until_finished()
            best_payload = {
                "best_step": best_step,
                "best_score": None if best_step is None else best_value,
                "latest_step": step + 1,
                "run_config_sha256": run_config_sha256,
            }
            temporary = output / ".BEST.json.incomplete"
            temporary.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            best_path = output / "BEST.json"
            os.replace(temporary, best_path)
            generation = output / "checkpoints" / str(step + 1)
            embedded = {
                "WSM_RUN_CONFIG.json": config_path.read_bytes(),
                "WSM_BEST.json": best_path.read_bytes(),
            }
            embedded_hashes = {}
            for name, payload in embedded.items():
                destination = generation / name
                temporary = generation / f".{name}.incomplete"
                temporary.write_bytes(payload)
                os.replace(temporary, destination)
                embedded_hashes[name] = hashlib.sha256(payload).hexdigest()
            generation_payload = checkpoint_completion_payload(
                step=step + 1,
                run_config_sha256=run_config_sha256,
                embedded_hashes=embedded_hashes,
                parameter_source="ema" if ema_params is not None else None,
                ema_decay=None if ema_decay is None else float(ema_decay),
            )
            temporary = generation / ".WSM_GENERATION_COMPLETE.json.incomplete"
            temporary.write_text(json.dumps(generation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, generation / "WSM_GENERATION_COMPLETE.json")
    manager.wait_until_finished()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--clip-gradient-norm", type=float, default=1.0)
    parser.add_argument("--min-lag", type=int, default=40)
    parser.add_argument("--future-delta", type=int, default=20)
    parser.add_argument("--history-stride", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=128)
    parser.add_argument("--mask-probability", type=float, default=0.5)
    parser.add_argument("--occ-weight", type=float, default=0.1)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
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
    if args.one_step_canary and os.environ.get("WSM_TPU_CANARY") != "1":
        raise SystemExit("WSM deliberative one-step canary requires WSM_TPU_CANARY=1")
    if not args.one_step_canary and not args.cpu_smoke and os.environ.get("WSM_WSM_REP_ALLOW_RUN") != "1":
        raise SystemExit("WSM deliberative production requires WSM_WSM_REP_ALLOW_RUN=1 after canary")
    train(args)


if __name__ == "__main__":
    main()
