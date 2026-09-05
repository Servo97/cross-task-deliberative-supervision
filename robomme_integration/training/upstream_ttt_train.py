"""Faithful official RoboMME recurrent-TTT + modulation training on the pinned LeRobot data.

The representation and policy integration classes are imported unchanged from the pinned official
RoboMME repository.  This file supplies only experiment controls that the upstream trainer lacks:
single-task LeRobot selection, compact feature I/O, explicit schedule/seed identity, full optimizer
checkpoint resume, bounded rolling checkpoints, TPU topology guards, and a checkpoint-free canary.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .single_task import TASK_EPISODES, task_manifest_sha256

UPSTREAM_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
UPSTREAM_ARCHIVE_SHA256 = "7696d94808f149f0774c7681eb855a0e11cada18e8def0e2961d0347dec8e3fa"
_CRITICAL_SHA256 = {
    "src/mme_vla_suite/models/representation/ttt.py": "21d10bd7c1b86044923f6d97e64d0bbb34ca518b4b0f6bea3dfadc4e78a32237",
    "src/mme_vla_suite/models/representation/recur_mem.py": "2d9da8706c96900a6c2a6777a3550ee7547c5375455ca4e5aba9969aabb9c42a",
    "src/mme_vla_suite/models/representation/mem_encoder.py": "a353328fdac81b2b47a9ad680463adf63b1f9b16c72e859929a4962f5513c648",
    "src/mme_vla_suite/models/integration/history_pi0.py": "a48dbfd412268a4ee689e10e6a6fd26c044eaa2b03cea8c075e2a34de49c4e57",
    "src/mme_vla_suite/models/integration/history_gemma.py": "a4882087a74b52b08a7a002a2a8bf7d64324af3ff05daf99c15a17f30bab60d1",
    "src/mme_vla_suite/models/config/robomme/recurrent-ttt-modul.yaml": "c42ff3b719bf8d9d075559ef93351fd4e70a412c8dbee54796a6e646cec3dc6d",
}


@contextlib.contextmanager
def _orbax_012_legacy_metadata_api(checkpointer_class):
    """Expose Orbax 0.12 item metadata to the pinned 0.11-era upstream loader only."""
    original = checkpointer_class.metadata

    def metadata_mapping(self, *args, **kwargs):
        metadata = original(self, *args, **kwargs)
        return getattr(metadata, "item_metadata", metadata)

    checkpointer_class.metadata = metadata_mapping
    try:
        yield
    finally:
        checkpointer_class.metadata = original


def _jax010_preprocess_observation(
    rng,
    observation,
    *,
    train: bool = False,
    image_keys=None,
    image_resolution=None,
):
    """Preserve upstream augmentation while aligning its mapped RNG batch sharding."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.training import sharding

    del image_keys  # Upstream intentionally processes only the images present to reduce memory.
    image_resolution = image_resolution or model_module.IMAGE_RESOLUTION
    batch_shape = observation.state.shape[:-1]
    out_images = {}
    for key, input_image in observation.images.items():
        image = input_image
        if image.shape[1:3] != image_resolution:
            model_module.logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = model_module.image_tools.resize_with_pad(image, *image_resolution)
        if train:
            image = image / 2.0 + 0.5
            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    model_module.augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    model_module.augmax.Resize(width, height),
                    model_module.augmax.Rotate((-5, 5)),
                ]
            transforms += [model_module.augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            sub_rngs = jax.random.split(rng, image.shape[0])
            sub_rngs = sharding.activation_sharding_constraint(sub_rngs)
            image = jax.vmap(model_module.augmax.Chain(*transforms))(sub_rngs, image)
            image = image * 2.0 - 1.0
        out_images[key] = image

    out_masks = {
        key: (
            jnp.ones(batch_shape, dtype=jnp.bool)
            if key not in observation.image_masks
            else jnp.asarray(observation.image_masks[key])
        )
        for key in out_images
    }
    return model_module.Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def _install_jax010_preprocess_compat() -> None:
    """Patch only the bound base-observation helper used by official history preprocessing."""
    from mme_vla_suite.models.integration import history_observation

    history_observation._preprocess_observation = _jax010_preprocess_observation


def _make_jax010_auto_mesh(num_fsdp_devices: int):
    """Match the known-working OpenPI mesh under JAX 0.10 sharding-in-types defaults."""
    import jax
    from openpi.training import sharding

    if jax.device_count() % num_fsdp_devices:
        raise ValueError("device count must be divisible by FSDP devices")
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(
        mesh_shape,
        (sharding.BATCH_AXIS, sharding.FSDP_AXIS),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )


def _install_flax_param_leaf_compat() -> None:
    """Keep the official TTT reset contract array-valued under modern Flax NNX."""
    import flax.nnx as nnx
    import jax
    from mme_vla_suite.models.representation.ttt import TTTBase

    def reset(self):
        return jax.tree.map(
            lambda value: value.value if isinstance(value, nnx.Param) else value,
            self.get_memory_state(),
            is_leaf=lambda value: isinstance(value, nnx.Param),
        )

    TTTBase.reset = reset


def _required_local_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    if value.startswith(("gs://", "s3://", "http://", "https://")):
        raise ValueError(f"{name} must be a node-local path")
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream_source(root: str | Path) -> Path:
    root = Path(root).resolve()
    for relative, expected in _CRITICAL_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing pinned upstream source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"upstream source drift for {relative}: {actual} != {expected}")
    import openpi

    openpi_path = Path(openpi.__file__).resolve()
    if root not in openpi_path.parents:
        raise ValueError(f"openpi resolved outside pinned upstream tree: {openpi_path}")
    return root


def _upstream_train_module(root: Path):
    path = root / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("_wsm_pinned_robomme_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official trainer helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(root: Path):
    import flax.nnx as nnx
    import openpi.training.optimizer as optimizer
    from mme_vla_suite.training import config as upstream_config

    task_name = os.environ.get("ROBOMME_TASK")
    if task_name not in TASK_EPISODES:
        raise ValueError(f"ROBOMME_TASK must be one of {tuple(TASK_EPISODES)}")
    data_root = _required_local_path("ROBOMME_DATA_ROOT")
    assets_root = _required_local_path("ROBOMME_ASSETS_ROOT")
    init_params = _required_local_path("WSM_INIT_FROM")
    cache_root = _required_local_path("ROBOMME_UPSTREAM_CACHE_ROOT")
    for path, label in ((data_root, "data"), (assets_root, "assets"), (init_params, "init"), (cache_root, "cache")):
        if not path.exists():
            raise ValueError(f"RoboMME upstream-TTT {label} path is missing: {path}")

    steps = int(os.environ.get("WSM_MAX_STEPS", "20000"))
    warmup = int(os.environ.get("WSM_WARMUP_STEPS", "1000"))
    peak_lr = float(os.environ.get("WSM_PEAK_LR", "5e-5"))
    decay_steps = int(os.environ.get("WSM_DECAY_STEPS", str(steps)))
    decay_lr = float(os.environ.get("WSM_DECAY_LR", "5e-6"))
    seed = int(os.environ.get("WSM_SEED", "0"))
    save_interval = int(os.environ.get("WSM_SAVE_INTERVAL", "5000"))
    if not (steps > 0 and 0 <= warmup < decay_steps and peak_lr > 0 and decay_lr >= 0 and seed >= 0):
        raise ValueError("invalid explicit upstream-TTT schedule or seed")
    if save_interval < 1:
        raise ValueError("WSM_SAVE_INTERVAL must be positive")

    history_yaml = root / "src/mme_vla_suite/models/config/robomme/recurrent-ttt-modul.yaml"
    base = upstream_config.get_config("mme_vla_suite")
    model = dataclasses.replace(
        base.model,
        pi05=True,
        action_horizon=20,
        max_token_len=200,
        use_history=True,
        history_config=str(history_yaml),
        discrete_state_input=False,
    )
    data = upstream_config.RoboMMEDataConfig(
        repo_id="robomme",
        assets=upstream_config.AssetsConfig(assets_dir=str(assets_root), asset_id="robomme"),
        base_config=upstream_config.DataConfig(prompt_from_task=True),
    )
    experiment = os.environ.get("WSM_EXP_NAME", f"st-{task_name}-upstream-ttt-seed{seed}")
    return dataclasses.replace(
        base,
        name="pi05_robomme_upstream_ttt",
        exp_name=experiment,
        project_name=os.environ.get("WANDB_PROJECT", "wsm-robomme"),
        model=model,
        data=data,
        weight_loader=upstream_config.MMEVLAWeightLoader(str(init_params)),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=warmup,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=decay_lr,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=0.99,
        seed=seed,
        batch_size=int(os.environ.get("WSM_BATCH_SIZE", "64")),
        num_workers=int(os.environ.get("WSM_NUM_WORKERS", "4")),
        num_train_steps=steps,
        log_interval=int(os.environ.get("WSM_LOG_INTERVAL", "100")),
        save_interval=save_interval,
        keep_period=None,
        overwrite=False,
        resume=os.environ.get("WSM_RESUME") == "1",
        wandb_enabled=os.environ.get("WANDB_MODE", "online") != "disabled",
        fsdp_devices=int(os.environ.get("WSM_FSDP_DEVICES", "4")),
        checkpoint_base_dir=os.environ.get("WSM_CKPT_BASE", "./checkpoints/robomme"),
        dataset_path=str(cache_root),
    )


def _initialize_checkpoint_dir(config):
    import openpi.training.checkpoints as checkpoints
    import orbax.checkpoint as ocp
    from etils import epath

    directory = epath.Path(config.checkpoint_dir).resolve()
    resuming = False
    if directory.exists():
        if config.resume:
            resuming = True
        else:
            raise FileExistsError(f"checkpoint directory exists without resume: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manager = ocp.CheckpointManager(
        directory,
        item_handlers={
            "assets": checkpoints.CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=None,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    if resuming and not tuple(manager.all_steps()):
        resuming = False
    return manager, resuming


def _save_state(manager, state, data_loader, step: int) -> None:
    import openpi.shared.normalize as normalize
    import openpi.training.checkpoints as checkpoints

    def save_assets(directory):
        config = data_loader.data_config()
        if config.norm_stats is not None and config.asset_id is not None:
            normalize.save(directory / config.asset_id, config.norm_stats)

    with checkpoints.at.disable_typechecking():
        train_state, params = checkpoints._split_params(state)
    manager.save(
        step,
        {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        },
    )


def _restore_state(manager, state):
    import openpi.training.checkpoints as checkpoints

    with checkpoints.at.disable_typechecking():
        train_state, params = checkpoints._split_params(state)
        restored = manager.restore(
            manager.latest_step(),
            items={"train_state": train_state, "params": {"params": params}},
        )
    return checkpoints._merge_params(restored["train_state"], restored["params"])


def _init_train_state(config, init_rng, mesh, train_helpers, *, resume: bool):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.optimizer as optimizer
    import openpi.training.sharding as sharding
    import openpi.training.utils as training_utils
    import orbax.checkpoint as ocp

    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    if resume:
        return shape, state_sharding
    # The pinned official RoboMME/OpenPI tree predates Orbax 0.12 and indexes the return value of
    # PyTreeCheckpointer.metadata() directly. Orbax 0.12 wraps the same mapping in
    # StepMetadata.item_metadata. Keep the official source hash-clean and expose the legacy view
    # only for this one initialization call; resume and all later checkpoint operations retain the
    # installed Orbax API.
    with _orbax_012_legacy_metadata_api(ocp.PyTreeCheckpointer):
        partial = train_helpers._load_weights_and_validate(config.weight_loader, shape.params.to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated,
        out_shardings=state_sharding,
    )(init_rng, partial)
    return state, state_sharding


def _train_step(config, rng, state, batch):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(module, step_rng, observation, actions):
        chunked_loss, stats = module.compute_loss(step_rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), stats

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, stats), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model,
        train_rng,
        observation,
        actions,
    )
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads)}, stats


def _validate_topology(config) -> None:
    import jax

    expected = int(os.environ.get("WSM_EXPECTED_JAX_DEVICES", "4"))
    topology = {
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
        "platforms": sorted({device.platform for device in jax.devices()}),
    }
    if topology != {
        "device_count": expected,
        "local_device_count": expected,
        "process_count": 1,
        "platforms": ["tpu"],
    }:
        raise SystemExit(f"unexpected TPU topology: {topology}")
    if config.batch_size % expected or config.fsdp_devices > expected or expected % config.fsdp_devices:
        raise SystemExit("invalid batch/FSDP topology")


def run(config, root: Path, *, one_step_canary: bool) -> None:
    import jax
    import jax.numpy as jnp
    import openpi.training.sharding as sharding
    import openpi.training.utils as training_utils
    import tqdm_loggable.auto as tqdm
    import wandb
    from flax.training import common_utils

    from gcp_tpu.train import install_checkpoint_memory_patch

    from .upstream_ttt_data import create_official_ttt_data_loader

    install_checkpoint_memory_patch()
    _install_jax010_preprocess_compat()
    _install_flax_param_leaf_compat()
    _validate_topology(config)
    train_helpers = _upstream_train_module(root)
    train_helpers.init_logging()
    logging.info(f"Running faithful upstream recurrent-TTT on: {platform.node()}")
    wandb.init(mode="disabled" if not config.wandb_enabled else "online", project=config.project_name)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = _make_jax010_auto_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_config = config.data.create(config.assets_dirs, config.model)
    loader = create_official_ttt_data_loader(
        data_root=os.environ["ROBOMME_DATA_ROOT"],
        cache_root=config.dataset_path,
        task_name=os.environ["ROBOMME_TASK"],
        repo_id="Yinpei/robomme_data_lerobot",
        data_config=data_config,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=data_sharding,
        num_workers=config.num_workers,
        seed=config.seed,
        verify_hashes=os.environ.get("WSM_VERIFY_FEATURE_HASHES", "1") == "1",
    )
    manager = None
    resuming = False
    if not one_step_canary:
        manager, resuming = _initialize_checkpoint_dir(config)
    state, state_sharding = _init_train_state(config, init_rng, mesh, train_helpers, resume=resuming)
    if resuming:
        state = _restore_state(manager, state)
    jax.block_until_ready(state)
    start_step = int(state.step)
    loader.set_start_step(start_step)
    data_iter = iter(loader)
    load_started = time.perf_counter()
    batch = next(data_iter)
    loader_seconds = time.perf_counter() - load_started
    logging.info(
        f"Initialized deterministic data cursor at optimizer step {start_step} in {loader_seconds:.2f}s:\n"
        f"{training_utils.array_tree_to_info(batch)}"
    )
    step_fn = jax.jit(
        functools.partial(_train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated, replicated),
        donate_argnums=(1,),
    )

    if one_step_canary:
        started = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info, stats = step_fn(train_rng, state, batch)
        jax.block_until_ready((state, info, stats))
        metrics = jax.device_get(info)
        if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
            raise RuntimeError(f"non-finite upstream-TTT canary metrics: {metrics}")
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "arm": "upstream_recurrent_ttt_modulation",
                    "task": os.environ["ROBOMME_TASK"],
                    "task_manifest_sha256": task_manifest_sha256(os.environ["ROBOMME_TASK"]),
                    "upstream_commit": UPSTREAM_COMMIT,
                    "seed": config.seed,
                    "device_count": jax.device_count(),
                    "loader_seconds": loader_seconds,
                    "step_seconds_including_compile": time.perf_counter() - started,
                    "train_state_step": int(state.step),
                    "metrics": {key: float(value) for key, value in metrics.items()},
                    "wrote_checkpoint": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    infos = []
    progress = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
    )
    interval_started = time.perf_counter()
    for step in progress:
        with sharding.set_mesh(mesh):
            state, info, stats = step_fn(train_rng, state, batch)
        infos.append(info)
        batch = next(data_iter)
        if step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            if not all(np.isfinite(np.asarray(value)).all() for value in reduced.values()):
                raise RuntimeError(f"non-finite upstream-TTT metrics at step {step}: {reduced}")
            elapsed = time.perf_counter() - interval_started
            reduced = {
                **reduced,
                "performance/step_ms": 1000.0 * elapsed / len(infos),
                "performance/samples_per_second": config.batch_size * len(infos) / elapsed,
            }
            progress.write(f"Step {step}: " + ", ".join(f"{key}={float(value):.4f}" for key, value in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
            interval_started = time.perf_counter()
        completed_steps = int(state.step)
        if completed_steps % config.save_interval == 0 or completed_steps == config.num_train_steps:
            _save_state(manager, state, loader, step)
    manager.wait_until_finished()
    print("[upstream-ttt] finalized training", flush=True)
    if os.environ.get("WSM_TPU_IMMEDIATE_EXIT", "1") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--one-step-canary", action="store_true")
    args = parser.parse_args()
    root = verify_upstream_source(_required_local_path("ROBOMME_UPSTREAM_ROOT"))
    config = build_config(root)
    print(
        f"[upstream-ttt] task={os.environ['ROBOMME_TASK']} "
        f"task_manifest_sha256={task_manifest_sha256(os.environ['ROBOMME_TASK'])} "
        f"upstream_commit={UPSTREAM_COMMIT} upstream_archive_sha256={UPSTREAM_ARCHIVE_SHA256} "
        f"seed={config.seed} steps={config.num_train_steps} batch={config.batch_size} "
        f"workers={config.num_workers} fsdp={config.fsdp_devices} save_interval={config.save_interval} "
        f"warmup={config.lr_schedule.warmup_steps} peak_lr={config.lr_schedule.peak_lr} "
        f"decay_steps={config.lr_schedule.decay_steps} decay_lr={config.lr_schedule.decay_lr}",
        flush=True,
    )
    if args.dry_run:
        return
    if args.one_step_canary:
        if os.environ.get("WSM_TPU_CANARY") != "1":
            raise SystemExit("one-step canary requires WSM_TPU_CANARY=1")
    elif os.environ.get("WSM_TPU_ALLOW_RUN") != "1" or os.environ.get("WSM_UPSTREAM_TTT_ALLOW_RUN") != "1":
        raise SystemExit("production upstream-TTT requires both TPU and post-canary allow gates")
    run(config, root, one_step_canary=args.one_step_canary)


if __name__ == "__main__":
    main()
