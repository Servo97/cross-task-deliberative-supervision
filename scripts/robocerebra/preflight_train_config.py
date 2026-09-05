#!/usr/bin/env python3
"""Local, CPU-only preflight for the RoboCerebra TrainConfigs.

Builds the *real* openpi data loader for a registered config and pulls one batch, so that the
data path, the transform stack, the norm stats and the model-facing shapes are all proven before
a queue job is spent. Deliberately does not touch the model weights -- that is the canary's job.

    HF_LEROBOT_HOME=... ROBOCEREBRA_ASSETS_DIR=... JAX_PLATFORMS=cpu \
    python preflight_train_config.py --config pi05_robocerebra_base
"""

from __future__ import annotations

import argparse
import dataclasses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_robocerebra_base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=2)
    # DEFAULT 2, NOT 0, deliberately. openpi spawns its DataLoader workers, so a preflight with
    # num_workers=0 exercises only the parent process and will happily pass while the real run dies
    # on its first batch -- which is exactly what happened on 2026-08-11 (the PyAV shim was
    # installed in the parent and absent in every spawned worker). Any preflight that claims the
    # data path works must cross a process boundary.
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--compute-loss",
        action="store_true",
        help="also run the model's compute_loss on a batch (CPU, slow, but it is "
        "the only check that exercises the mechanism arms' shape guards)",
    )
    args = parser.parse_args()

    import os
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from openpi.training import lerobot_video_shim

    os.environ[lerobot_video_shim.ENV_FLAG] = "1"
    lerobot_video_shim.install()

    import numpy as np
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size, num_workers=args.num_workers)
    print(
        f"config={cfg.name} batch={cfg.batch_size} num_workers={cfg.num_workers} "
        f"steps={cfg.num_train_steps} save={cfg.save_interval} keep={cfg.keep_period}"
    )
    if cfg.num_workers == 0:
        print(
            "WARNING: num_workers=0 does not exercise the spawned-worker path; "
            "a pass here does NOT mean the queue run will load data"
        )
    print(
        f"model pi05={cfg.model.pi05} action_horizon={cfg.model.action_horizon} "
        f"action_dim={cfg.model.action_dim} discrete_state_input={cfg.model.discrete_state_input}"
    )
    print(f"weight_loader={cfg.weight_loader}")

    loader = _data_loader.create_data_loader(cfg, sharding=None, num_batches=args.batches)
    data_config = loader.data_config()
    print(
        f"repo_id={data_config.repo_id} asset_id={data_config.asset_id} "
        f"prompt_from_task={data_config.prompt_from_task}"
    )
    assert data_config.norm_stats is not None, "norm stats did not resolve -- check ROBOCEREBRA_ASSETS_DIR"

    # Stage-Q (RoboTTT) arms sample WINDOWS, not steps: every leaf gains a leading L axis and the
    # trainer flattens [B, L, ...] -> [B*L, ...] before the pi forward. The batch shapes and the
    # loss call below therefore both have to be window-aware, or this script would "pass" on a
    # config whose real train step it never exercised.
    window_len = int(getattr(data_config, "stage_q_window_len", 0) or 0)
    if window_len:
        print(
            f"STAGE-Q window loader: L={window_len} "
            f"chunk_stride={getattr(data_config, 'stage_q_chunk_stride', None)} "
            f"iid_steps={getattr(data_config, 'stage_q_iid_steps', False)} "
            f"robottt={getattr(cfg.model, 'robottt', False)}; batch_size counts WINDOWS, so the "
            f"effective per-step pi batch is {cfg.batch_size * window_len}"
        )

    for i, (observation, actions) in enumerate(loader):
        print(f"--- batch {i} ---")
        acts = np.asarray(actions)
        print(f"actions {acts.shape} {acts.dtype} finite={np.isfinite(acts).all()} absmax={np.abs(acts).max():.3f}")
        state = np.asarray(observation.state)
        print(f"state {state.shape} finite={np.isfinite(state).all()} absmax={np.abs(state).max():.3f}")
        for name, img in observation.images.items():
            arr = np.asarray(img)
            print(
                f"image[{name}] {arr.shape} {arr.dtype} range=({arr.min():.3f},{arr.max():.3f}) "
                f"mask={np.asarray(observation.image_masks[name]).tolist()}"
            )
        if getattr(observation, "tokenized_prompt", None) is not None:
            tok = np.asarray(observation.tokenized_prompt)
            print(f"tokenized_prompt {tok.shape} nonpad={int(np.asarray(observation.tokenized_prompt_mask).sum())}")
        for key in ("wsm_w_window", "wsm_w_target", "wsm_w_target_valid"):
            extra = getattr(observation, key, None)
            if extra is None:
                continue
            arr = np.asarray(extra)
            if arr.dtype == bool:
                print(f"{key} {arr.shape} valid={arr.sum()}/{arr.size}")
            else:
                print(f"{key} {arr.shape} finite={np.isfinite(arr).all()} rms={float(np.sqrt((arr**2).mean())):.3f}")

    if args.compute_loss:
        # THE check that matters for mechanism arms. Inspecting the shapes the loader emits proves
        # nothing on its own -- arm A3 shipped with `wsm_w_target (B,1,512)` printed happily by this
        # very script, and died on the first training step because the k=1 loss contract wants the
        # squeezed (B,512). Only running the real loss exercises the mechanism's own shape guards.
        import jax
        import jax.numpy as jnp

        print("running compute_loss (CPU, one batch) — this exercises the mechanism code paths")
        # `create` takes a raw PRNG key (train.py: `config.model.create(model_rng)`), not nnx.Rngs.
        model = cfg.model.create(jax.random.key(0))
        if window_len:
            # Byte-for-byte the body of `scripts/train.py::stage_q_train_step.loss_fn`, minus the
            # gradient. Running the plain per-step `compute_loss` on a [B, L, ...] batch would
            # either blow up on rank or (worse) silently fold L into the batch without ever
            # threading the fast weights, so the check would not cover the arm's own code path.
            b, length = actions.shape[0], actions.shape[1]
            obs_flat = jax.tree.map(lambda x: x.reshape(b * length, *x.shape[2:]), observation)
            actions_flat = actions.reshape(b * length, *actions.shape[2:])
            if getattr(model, "robottt", False):
                state_seq = obs_flat.state.reshape(b, length, -1)
                o_seq, w_final = model.robottt_fast.run_sequence(state_seq, actions)
                o_seq = np.asarray(o_seq)
                print(
                    f"robottt run_sequence OK: O_seq {o_seq.shape} finite={np.isfinite(o_seq).all()} "
                    f"absmax={np.abs(o_seq).max():.3e} "
                    f"W_final leaves={[f'{k}{tuple(v.shape)}' for k, v in sorted(w_final.items())]}"
                )
                robottt_cond = jnp.swapaxes(jnp.asarray(o_seq), 0, 1).reshape(b * length, -1)
                observation_for_loss = dataclasses.replace(obs_flat, robottt_cond=robottt_cond)
            else:
                observation_for_loss = obs_flat
            loss = model.compute_loss(jax.random.key(1), observation_for_loss, actions_flat, train=True)
        else:
            loss = model.compute_loss(jax.random.key(1), observation, actions, train=True)
        loss = np.asarray(loss)
        assert np.isfinite(loss).all(), f"non-finite loss {loss}"
        print(f"compute_loss OK: shape={loss.shape} mean={float(loss.mean()):.4f}")

    # Shape contract against the released pi05_libero checkpoint.
    assert cfg.model.action_horizon == 10, "action_horizon must be 10 to load pi05_libero params"
    assert cfg.model.discrete_state_input is False, "pi05_libero was trained with discrete_state_input=False"
    expected_action_shape = (window_len, 10, cfg.model.action_dim) if window_len else (10, cfg.model.action_dim)
    assert acts.shape[1:] == expected_action_shape, (
        f"unexpected action chunk {acts.shape}, expected (B, *{expected_action_shape})"
    )
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
