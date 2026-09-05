#!/usr/bin/env python3
"""Serve a pi05_libero-family checkpoint over websockets, without editing the openpi fork.

``openpi/scripts/serve_policy.py`` cannot serve this checkpoint in our fork: it calls
``train_config.data.create(train_config.assets_dirs, ...)`` unconditionally, and our
``config.py:238`` norm-stats fallback dispatches to
``groot_openpi_dataset._convert_stats_from_repo_meta``, which does not exist -- so a norm-stats
lookup that *should* return ``None`` raises ``AttributeError`` instead. Passing ``norm_stats=``
to ``create_trained_policy`` does not help, because that ``data.create`` call happens first.

The fix that needs no upstream edit: point ``assets_base_dir`` at a directory laid out as
``<base>/<config name>/<asset_id>/norm_stats.json``. For a released checkpoint that is exactly
its own ``assets/`` subtree, so we materialise ``<base>/pi05_libero -> <ckpt>/assets`` and let
the normal loader find it.

Serving a released checkpoint zero-shot therefore uses **the checkpoint's own LIBERO norm
stats** -- which is the correct choice for a zero-shot floor. Our fresh RoboCerebra stats belong
to the finetuned arms, not to this row.

    python serve_pi05_libero.py --checkpoint <dir> --port 8000

K ENVS PER GPU
--------------
``WSM_ENVS_PER_GPU=K`` (K>1) makes openpi's ``WebsocketPolicyServer`` wrap the policy in its
gather-batching facade: K connections' ``infer`` calls coalesce into one ``Policy.infer_batch``.
This server needs no per-client state to do that -- an A0/A3 policy is stateless across requests,
which is exactly why the ω server needed a whole identity layer and this one does not.

Two things still have to be right for K>1 to be outcome-neutral, and both are handled here:

* **rng.** ``Policy.infer`` splits a MUTABLE rng per call, so with K envs each env's action noise
  would depend on global arrival order. The harness's ``--deterministic-seeding`` sends
  ``policy_noise_seed`` (uint32, derived from mode|case|trial|step) with every request and openpi
  draws that row's noise from it instead. Nothing here has to opt in: ``Policy.infer`` and
  ``Policy.infer_batch`` both pop the field. This server only *reports* whether it saw one, so a
  results file can never claim determinism it did not have.
* **kernel selection.** ``Policy.infer_batch`` pads to the smallest of ``(4, 8)`` that fits, so a
  3-request window and a 6-request window run different executables. ``--policy-pad-batch``
  replicate-pads every batched call to one fixed row count. See ``serve_batching.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


class _NoiseSeedWitness:
    """Counts requests that carried ``policy_noise_seed``. Pure observability, zero semantics.

    Exists because "we ran K=8 with deterministic seeding" is a claim a results file makes about a
    server it cannot see. If the harness forgets ``--deterministic-seeding``, K=8 still runs, still
    produces numbers, and those numbers are order-dependent -- silently. This makes the server say so.
    """

    def __init__(self, policy, *, warn_after: int = 50):
        self._policy = policy
        self._warn_after = int(warn_after)
        self.seeded = 0
        self.unseeded = 0
        self._warned = False

    def _tally(self, obs_list) -> None:
        for obs in obs_list:
            if "policy_noise_seed" in obs:
                self.seeded += 1
            else:
                self.unseeded += 1
        if not self._warned and self.unseeded >= self._warn_after:
            self._warned = True
            logging.warning(
                "[serve-rc] %d requests arrived WITHOUT policy_noise_seed. Under K>1 that makes "
                "action noise a function of arrival order: not reproducible, and not equal to K=1. "
                "Re-run the harness with --deterministic-seeding.",
                self.unseeded,
            )

    def infer(self, obs: dict, **kwargs) -> dict:
        self._tally([obs])
        return self._policy.infer(obs, **kwargs)

    def infer_batch(self, obs_list, **kwargs) -> list:
        self._tally(obs_list)
        return self._policy.infer_batch(obs_list, **kwargs)

    @property
    def metadata(self) -> dict:
        return dict(getattr(self._policy, "metadata", {}) or {})

    def __getattr__(self, name: str):
        return getattr(self._policy, name)


def main() -> None:
    import openpi.policies.policy_config as policy_config
    import openpi.serving.websocket_policy_server as websocket_policy_server
    import openpi.training.config as _config
    from serve_batching import FixedPadBatchPolicy, gather_settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint dir holding params/ and assets/")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument(
        "--assets-link-root",
        default=None,
        help="where to materialise the <config>/<asset_id> assets view (default: <checkpoint>/../_serve_assets)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument(
        "--policy-batch",
        default="batched",
        choices=["batched", "serial"],
        help="batched (default): ONE padded policy call per gather window. serial: "
        "execute a gathered window as N batch-1 Policy.infer calls — the same "
        "executable the unbatched path runs, hence bit-identical to a K=1 "
        "seeded run, at no speedup for a policy-only server. Requires "
        "--deterministic-seeding on the harness.",
    )
    parser.add_argument(
        "--policy-pad-batch",
        type=int,
        default=8,
        help="replicate-pad every BATCHED policy call to this many rows so one XLA "
        "kernel serves every gather size (must be an openpi bucket: 4 or 8). "
        "0 = stock openpi bucketing. The K=1 legacy path never batches at all.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    link_root = pathlib.Path(args.assets_link_root or checkpoint.parent / "_serve_assets").resolve()
    link_root.mkdir(parents=True, exist_ok=True)
    link = link_root / args.config
    if not link.exists():
        link.symlink_to(checkpoint / "assets", target_is_directory=True)

    train_config = dataclasses.replace(_config.get_config(args.config), assets_base_dir=str(link_root))
    policy = policy_config.create_trained_policy(train_config, checkpoint, default_prompt=args.default_prompt)

    k_envs, gather_max_batch, gather_wait_ms = gather_settings()
    if args.policy_pad_batch and gather_max_batch % args.policy_pad_batch:
        raise SystemExit(
            f"gather max batch {gather_max_batch} is not a multiple of --policy-pad-batch "
            f"{args.policy_pad_batch}: the last chunk of a full window would run a DIFFERENT padded "
            "row count than the others."
        )
    served = _NoiseSeedWitness(FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch, mode=args.policy_batch))
    metadata = dict(getattr(policy, "metadata", {}) or {})
    metadata.update(
        {
            "policy_pad_batch": int(args.policy_pad_batch),
            "policy_batch": args.policy_batch,
            "gather_max_batch": gather_max_batch,
            "gather_wait_ms": gather_wait_ms,
            "envs_per_gpu": k_envs,
        }
    )

    logging.info("serving %s from %s on %s:%d", args.config, checkpoint, args.host, args.port)
    if k_envs > 1:
        logging.info(
            "gather-batching ON: %d env slots, <=%d per batch, %.0f ms window, policy=%s (pad %d rows)",
            k_envs,
            gather_max_batch,
            gather_wait_ms,
            args.policy_batch,
            args.policy_pad_batch,
        )
    else:
        logging.info("gather-batching OFF (WSM_ENVS_PER_GPU=1): LEGACY Policy.infer at batch 1")
    websocket_policy_server.WebsocketPolicyServer(
        policy=served,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
