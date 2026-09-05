"""Materialize hash-locked causal omega_t caches from a trained deliberative WSM encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .single_task import TASK_EPISODES, task_manifest_sha256
from .workspace_deliberative import WorkspaceBatchSampler, _manager, encode, init_params, sha256_file


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty checkpoint tree: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _encoder_identity(
    *,
    run_config_sha256: str,
    checkpoint_step: int,
    checkpoint_tree_sha256: str,
    materializer_sha256: str,
    parameter_source: str | None = None,
) -> dict:
    """Build a v1-byte-compatible identity, extended only for the v2 EMA path."""
    value = {
        "schema_version": 1,
        "run_config_sha256": run_config_sha256,
        "checkpoint_step": checkpoint_step,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "materializer_sha256": materializer_sha256,
    }
    if parameter_source == "ema":
        value["parameter_source"] = parameter_source
    elif parameter_source is not None:
        raise ValueError(f"unknown encoder parameter source {parameter_source!r}")
    return value


def materialize(
    args,
    *,
    sampler_class=WorkspaceBatchSampler,
    trainer_path: Path | None = None,
    materializer_path: Path | None = None,
    required_protocol: str | None = None,
    init_params_function=init_params,
) -> None:
    import jax
    import optax

    train_root = Path(args.train_root) / args.task
    run_config_path = train_root / "RUN_CONFIG.json"
    best_path = train_root / "BEST.json"
    if not run_config_path.is_file() or not best_path.is_file():
        raise ValueError("WSM train root is missing RUN_CONFIG.json or BEST.json")
    run_config_bytes = run_config_path.read_bytes()
    run_config = json.loads(run_config_bytes)
    run_config_sha = hashlib.sha256(run_config_bytes).hexdigest()
    trainer_path = trainer_path or Path(__file__).with_name("workspace_deliberative.py")
    if run_config.get("implementation_sha256") != sha256_file(trainer_path):
        raise ValueError("WSM trainer implementation differs from the checkpoint run config")
    if required_protocol is not None and run_config.get("protocol") != required_protocol:
        raise ValueError("WSM checkpoint protocol identity mismatch")
    if run_config.get("task") != args.task or run_config.get("task_manifest_sha256") != task_manifest_sha256(
        args.task
    ):
        raise ValueError("WSM checkpoint task identity mismatch")
    if run_config.get("omega_dim") != 512:
        raise ValueError("WSM checkpoint omega dimension mismatch")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if best.get("run_config_sha256") != run_config_sha:
        raise ValueError("WSM BEST.json/config identity mismatch")
    step = int(args.step if args.step is not None else best["best_step"])
    if step < 1:
        raise ValueError("selected WSM checkpoint step must be positive")

    devices = jax.devices()
    if len(devices) != args.expected_devices:
        raise SystemExit(f"expected {args.expected_devices} devices, found {len(devices)}")
    platforms = {device.platform for device in devices}
    if not args.cpu_smoke and platforms not in ({"tpu"}, {"gpu"}):
        raise SystemExit("production omega materialization requires homogeneous TPU or GPU devices")
    batch_size = args.batch_size
    if batch_size % len(devices):
        raise ValueError("materialization batch must be divisible by device count")
    per_device = batch_size // len(devices)

    sampler = sampler_class(
        args.supervision_root,
        args.task,
        seed=int(run_config["seed"]),
        min_lag=int(run_config["min_lag"]),
        future_delta=int(run_config["future_delta"]),
        history_stride=int(run_config["history_stride"]),
        max_history=int(run_config["max_history"]),
        verify_hashes=not args.skip_supervision_hashes,
    )
    if sampler.manifest["manifest_sha256"] != run_config.get("supervision_manifest_sha256"):
        raise ValueError("WSM materializer supervision manifest drifted from training")
    if (
        sampler.state_mean.tolist() != run_config["state_mean"]
        or sampler.state_std.tolist() != run_config["state_std"]
    ):
        raise ValueError("WSM materializer state normalization drifted from training")
    if list(sampler.train_episodes) != run_config["train_episodes"]:
        raise ValueError("WSM materializer train split drifted")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(run_config["learning_rate"]),
        warmup_steps=int(run_config.get("warmup_steps", 500)),
        decay_steps=int(run_config["steps"]),
        end_value=float(run_config["learning_rate"]) * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(run_config.get("clip_gradient_norm", 1.0))),
        optax.adamw(schedule, weight_decay=float(run_config["weight_decay"])),
    )
    initial_params = init_params_function(jax.random.key(int(run_config["seed"])))
    initial_opt = optimizer.init(initial_params)
    manager = _manager(train_root / "checkpoints", max_to_keep=int(run_config.get("max_checkpoints", 10)))
    if step not in manager.all_steps():
        raise ValueError(f"selected WSM checkpoint {step} not present; available={manager.all_steps()}")
    template = {"params": initial_params, "opt_state": initial_opt, "step": np.asarray(0, dtype=np.int64)}
    use_ema = run_config.get("ema_decay") is not None
    if use_ema:
        template["ema_params"] = initial_params
    state = manager.restore(step, items={"state": template})["state"]
    if int(state["step"]) != step:
        raise ValueError("restored WSM train-state step mismatch")
    params = jax.device_put(state["ema_params"] if use_ema else state["params"])
    encoder = jax.pmap(encode, in_axes=(None, 0, 0))

    checkpoint_root = train_root / "checkpoints" / str(step)
    checkpoint_sha = _tree_sha256(checkpoint_root)
    encoder_identity = _encoder_identity(
        run_config_sha256=run_config_sha,
        checkpoint_step=step,
        checkpoint_tree_sha256=checkpoint_sha,
        materializer_sha256=sha256_file(materializer_path or Path(__file__)),
        parameter_source="ema" if use_ema else None,
    )
    encoder_id = hashlib.sha256(
        (json.dumps(encoder_identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    if args.one_batch_canary:
        episode = TASK_EPISODES[args.task][0]
        steps = int(sampler.records[episode]["steps"])
        valid = min(batch_size, steps)
        histories, masks = zip(
            *(sampler.history(episode, decision, mask_current=False) for decision in range(valid)),
            strict=True,
        )
        history = np.stack(histories)
        mask = np.stack(masks)
        if valid < batch_size:
            history = np.concatenate([history, np.zeros((batch_size - valid, *history.shape[1:]), np.float32)])
            mask = np.concatenate([mask, np.zeros((batch_size - valid, *mask.shape[1:]), np.bool_)])
        history = history.reshape((len(devices), per_device, *history.shape[1:]))
        mask = mask.reshape((len(devices), per_device, *mask.shape[1:]))
        omega = np.asarray(encoder(params, history, mask)).reshape((batch_size, -1))[:valid]
        if omega.shape != (valid, 512) or not np.isfinite(omega).all():
            raise RuntimeError(f"invalid omega materializer canary output: {omega.shape}")
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": args.task,
                    "checkpoint_step": step,
                    "checkpoint_tree_sha256": checkpoint_sha,
                    "encoder_id": encoder_id,
                    "device_count": len(devices),
                    "rows": valid,
                    "omega_shape": list(omega.shape),
                    "wrote_cache": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    output_root = Path(args.output_root) / args.task
    manifest_path = output_root / "MANIFEST.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("encoder_id") == encoder_id:
            print(f"[wsm-materialize] already complete encoder_id={encoder_id}", flush=True)
            return
        raise ValueError(f"omega cache collision at {manifest_path}")

    records = []
    for ordinal, episode in enumerate(TASK_EPISODES[args.task], 1):
        steps = int(sampler.records[episode]["steps"])
        chunks = []
        for begin in range(0, steps, batch_size):
            end = min(begin + batch_size, steps)
            histories, masks = zip(
                *(sampler.history(episode, decision, mask_current=False) for decision in range(begin, end)),
                strict=True,
            )
            history = np.stack(histories)
            mask = np.stack(masks)
            valid = len(history)
            if valid < batch_size:
                history = np.concatenate([history, np.zeros((batch_size - valid, *history.shape[1:]), np.float32)])
                mask = np.concatenate([mask, np.zeros((batch_size - valid, *mask.shape[1:]), np.bool_)])
            history = history.reshape((len(devices), per_device, *history.shape[1:]))
            mask = mask.reshape((len(devices), per_device, *mask.shape[1:]))
            value = np.asarray(encoder(params, history, mask)).reshape((batch_size, -1))[:valid]
            chunks.append(value)
        omega = np.concatenate(chunks).astype(np.float16)
        if omega.shape != (steps, 512) or not np.isfinite(omega).all():
            raise RuntimeError(f"invalid materialized omega for episode {episode}: {omega.shape}")
        destination = output_root / f"episode_{episode}" / "omega_f16.npy"
        if destination.exists():
            existing = np.load(destination, mmap_mode="r")
            if existing.shape != omega.shape or existing.dtype != omega.dtype or not np.array_equal(existing, omega):
                raise ValueError(f"partial omega cache collision at {destination}")
        else:
            _atomic_npy(destination, omega)
        records.append(
            {
                "episode": episode,
                "steps": steps,
                "path": f"episode_{episode}/omega_f16.npy",
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        print(f"[wsm-materialize] {ordinal}/100 episode={episode} steps={steps}", flush=True)

    manifest = {
        "schema_version": 1,
        "task_name": args.task,
        "task_manifest_sha256": task_manifest_sha256(args.task),
        "omega_dim": 512,
        "episodes": list(TASK_EPISODES[args.task]),
        "encoder_id": encoder_id,
        "encoder_identity": encoder_identity,
        "causal_contract": {
            "source_frames": "at_or_before_step_idx",
            "uses_future_execution_frames": False,
            "video_prefix": "benchmark_provided_only",
        },
        "representation": {
            "history_stride": run_config["history_stride"],
            "max_history": run_config["max_history"],
            "uses_labels_at_inference": False,
        },
        "records": records,
    }
    if required_protocol is not None:
        manifest["protocol"] = required_protocol
        manifest["supervision_artifact"] = run_config.get("supervision_artifact")
        manifest["target_semantics"] = run_config.get("target_semantics")
    unhashed = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest["manifest_sha256"] = hashlib.sha256(unhashed).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / ".MANIFEST.json.incomplete"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"[wsm-materialize] complete encoder_id={encoder_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASK_EPISODES))
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-devices", type=int, default=4)
    parser.add_argument("--skip-supervision-hashes", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--one-batch-canary", action="store_true")
    args = parser.parse_args()
    if args.one_batch_canary and not args.cpu_smoke and os.environ.get("WSM_TPU_CANARY") != "1":
        raise SystemExit("TPU omega-materializer canary requires WSM_TPU_CANARY=1")
    if not args.one_batch_canary and not args.cpu_smoke and os.environ.get("WSM_WSM_MATERIALIZE_ALLOW_RUN") != "1":
        raise SystemExit("omega materialization requires WSM_WSM_MATERIALIZE_ALLOW_RUN=1 after canary")
    materialize(args)


if __name__ == "__main__":
    main()
