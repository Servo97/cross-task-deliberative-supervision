#!/usr/bin/env python3
"""pi0.5 Stage-Q RoboTTT policy server (Q2: fast weights, no workspace).

Serves a Stage-Q fast-weights checkpoint with the per-env online TTT update running server-side:
per episode reset (wsm_t == 0) W restarts from the meta-learned init; per chunk the policy is
conditioned on O_t = f_W(state) computed from the HELD entering W; after each executed chunk W
advances by exactly one inner-GD commit on the model's OWN normalized action chunk. All identity,
ordering, isolation, and K-batching semantics come from WSMPiInferWrapper in workspace-free mode —
no frozen tap, no WSM encoder, no omega injection.

Example:

  python vla_training/eval/serve_pi_05_robottt.py \
      --finetune-ckpt <stage-q-q2/59999> --port 8000

Run in the openpi-jax-latest environment.
"""

from __future__ import annotations

import argparse


def build_robottt_policy(finetune_ckpt: str, config_name: str, max_token_len: int):
    """Rebuild the Q2 policy: pi05 + robottt=True with the shared Stage-Q geometry (packet 07 —
    every RoboTTT hyperparameter is fixed across the 2x2, so serve needs no per-run knobs), norm
    stats from the checkpoint's assets/. expose_norm_actions=True is REQUIRED: the wrapper's commit
    consumes the result's model-space norm_state/norm_actions rows."""
    import os

    import openpi.models.pi0_config as pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as _config

    model = pi0_config.Pi0Config(pi05=True, max_token_len=int(max_token_len), robottt=True)
    cfg = _config.TrainConfig(
        name=config_name,
        exp_name="pi05_rc365_stage_q_serve",
        model=model,
        data=_config.LeRobotRobocasaDataConfig(data_dirs=[]),
    )
    print(
        f"[serve-pi-robottt] config={config_name} pi05={model.pi05} robottt={model.robottt} "
        f"max_token_len={model.max_token_len}",
        flush=True,
    )
    return policy_config.create_trained_policy(cfg, os.path.expanduser(finetune_ckpt), expose_norm_actions=True)


def assert_robottt_loaded(policy) -> None:
    """Refuse to serve unless the TRAINED robottt_fast subtree was actually restored.

    Finiteness catches diverged checkpoints. The trained-vs-init check uses the tanh gate `alpha`:
    it is initialized to the constant gate_init everywhere, so an alpha that is still exactly that
    constant means the subtree was re-initialized instead of loaded (the from_pretrained-bypass
    failure mode) — in which case O_t is a meaningless near-zero read and the eval would silently
    score a broken policy.

    Returns the TRAINED scalar summary (inner-LR eta and the tanh-gate magnitudes) so the caller can
    stamp it into the A0 probe log; raising on failure is still the primary job."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    if not getattr(model, "robottt", False) or not hasattr(model, "robottt_fast"):
        raise RuntimeError("[serve-pi-robottt] model has no robottt_fast subtree; wrong config")
    fast = model.robottt_fast
    arrays = [jnp.asarray(v.value) for _path, v in nnx.state(fast, nnx.Param).flat_state()]
    n_bad = sum(int(not bool(jnp.isfinite(a).all())) for a in arrays)
    if n_bad:
        raise RuntimeError(
            f"[serve-pi-robottt] robottt_fast has NON-FINITE params ({n_bad}/{len(arrays)} bad) "
            "— diverged ckpt; refusing."
        )
    alpha = jnp.asarray(fast.alpha.value)
    gate_init = float(fast.cfg.gate_init)
    if bool(jnp.all(alpha == gate_init)):
        raise RuntimeError(
            "[serve-pi-robottt] robottt_fast.alpha is still exactly gate_init everywhere "
            "(untrained init) — the trained subtree was NOT restored. Refusing to serve."
        )
    eta = float(fast.inner_lr())
    gate = float(jnp.max(jnp.abs(jnp.tanh(alpha))))
    gate_mean = float(jnp.mean(jnp.abs(jnp.tanh(alpha))))
    base_lr = float(fast.cfg.base_inner_lr)
    print(
        f"[serve-pi-robottt] ✓ robottt_fast restored: {len(arrays)} param tensors, all finite, "
        f"max|tanh(alpha)|={gate:.4g}, mean|tanh(alpha)|={gate_mean:.4g}, inner_lr eta={eta:.4g} "
        f"(= base {base_lr:.4g} x softplus {eta / base_lr if base_lr else float('nan'):.4g})",
        flush=True,
    )
    # The two A0 startup numbers: did the model train its updates toward zero? (small eta and/or a
    # near-zero gate mean the online update barely moves the policy at all).
    return {
        "n_param_tensors": len(arrays),
        "inner_lr_eta": eta,
        "base_inner_lr": base_lr,
        "softplus_log_inner_lr": eta / base_lr if base_lr else float("nan"),
        "alpha_max_abs_tanh": gate,
        "alpha_mean_abs_tanh": gate_mean,
        "alpha_dim": int(alpha.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True, help="Stage-Q q2 finetune ckpt (orbax step dir)")
    ap.add_argument(
        "--config-name",
        default="pi05_robocasa_stage_q_q2",
        help="TrainConfig provenance label (matches the training leaf)",
    )
    ap.add_argument("--max-token-len", type=int, default=200)
    ap.add_argument("--stride", type=int, default=8, help="serve replan cadence == training chunk_stride")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from openpi.serving import websocket_policy_server

    from vla_training.eval._robottt_ablation import ablation_from_env, probe_from_env
    from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner
    from vla_training.eval.serve_pi_05_wsm import WSMPiInferWrapper

    # Parse the serve-only ablation ONCE, before any model work: garbage tokens or a missing
    # ROBOTTT_ABLATION_ACK=smoke must fail before a GPU is touched. Empty/unset == the decisive path.
    ablation = ablation_from_env()
    probe = probe_from_env()
    print(
        f"[serve-pi-robottt] robottt_ablation={ablation.describe()} probe_log={getattr(probe, 'path', None)}",
        flush=True,
    )

    policy = build_robottt_policy(args.finetune_ckpt, args.config_name, args.max_token_len)
    scalars = assert_robottt_loaded(policy)
    runner = RoboTTTServeRunner(policy)
    if probe is not None:
        probe.log(
            "startup",
            ckpt=args.finetune_ckpt,
            config_name=args.config_name,
            stride=int(args.stride),
            ablation=ablation.spec,
            ablation_detail=ablation.as_metadata(),
            eta_effective=float(scalars["inner_lr_eta"]) * float(ablation.eta_scale),
            **scalars,
        )

    wrapped = WSMPiInferWrapper(
        policy,
        None,  # no frozen tap
        None,  # no workspace conditioner
        None,  # no task language table
        stride=args.stride,
        robottt_runner=runner,
        robottt_ablation=ablation,
        robottt_probe=probe,
    )
    print(
        f"[serve-pi-robottt] ✓ pi0.5 Q2 server ready on {args.host}:{args.port} "
        f"(fast weights ONLINE, stride={args.stride}, ablation={ablation.describe()})",
        flush=True,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
