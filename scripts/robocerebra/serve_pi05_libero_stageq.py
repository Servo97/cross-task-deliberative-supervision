#!/usr/bin/env python3
"""Serve the RoboCerebra Stage-Q "Q2" arm (A5): persistent per-episode RoboTTT fast weights.

WHY THIS FILE EXISTS
--------------------
The A5 arm trains with an online inner-GD loop over L=8 chunk-step windows. That loop lives
ENTIRELY outside the model: ``Pi0`` only reads ``observation.robottt_cond`` and adds it to
``adarms_cond``. Nothing in ``serve_pi05_libero.py`` computes that vector, and ``LiberoInputs``
drops the key even if something did. Served by the plain server, A5 is therefore INERT: identical
to A0 by construction, and it would report a null with no error anywhere in the logs. This server
is the missing half.

WHAT IT DOES, PER ENV, PER CHUNK
--------------------------------
1. ``wsm_t == 0`` -> that env's fast weights ``W`` are (re)initialised to the meta-learned ``W_0``.
2. APPLY: ``O_t = tanh(alpha) * readout(pool(f_W(Q_t)))`` from the HELD entering ``W`` and the
   current MODEL-SPACE (normalised) state. The same vector rides through every Euler step of the
   chunk; ``W`` is not mutated here.
3. The policy runs with ``robottt_cond = O_t`` injected.
4. COMMIT: exactly ONE inner-GD step from the just-finalised chunk's model-space
   ``(norm_state, norm_actions)``, at the END of the same call — i.e. strictly before the next
   request of that env is conditioned.

All four steps are the RoboCasa Q2 semantics (``vla_training/eval/serve_pi_05_robottt.py`` +
``WSMPiInferWrapper`` in workspace-free mode), reproduced here on the RoboCerebra identity contract
(``serve_pi05_libero_wsm.py``'s per-env state machine) instead of the RoboCasa one. The fast-weight
runner itself is IMPORTED from ``vla_training/eval/_robottt_serve_runner.py`` rather than copied, so
the normalised-space condition/commit contract has exactly one implementation.

CHUNK CADENCE — the one real train/serve deviation
--------------------------------------------------
RoboCasa serves at replan 8 and trains at ``stage_q_chunk_stride = 8``: one commit per executed
chunk IS one training chunk-step. RoboCerebra serves at replan 5 (and re-plans early at every
subtask boundary), while the Stage-Q data config keeps the RoboCasa constant
``stage_q_chunk_stride = 8``. There is no cadence that is simultaneously "one commit per executed
chunk" and "one commit per 8 native frames".

We commit on **every executed chunk** (replan 5, plus the boundary re-plans), because that is the
invariant the mechanism is defined by (D-1: the update consumes the chunk that was actually
executed) and it is what the RoboCasa serve does. The consequence is that W advances ~1.6x more
often per frame at serve than per training chunk-step. ``--commit-stride N`` sub-samples commits to
every Nth executed chunk if that deviation ever needs to be probed; N=1 is the decisive path.

CLI SHAPE
---------
Deliberately the same as ``serve_pi05_libero.py`` (``--checkpoint --config --port
--policy-pad-batch``) so the v3 ladder can drive it with only the script path changed. Two things
the launcher must still get right:

* the harness must run with ``--wsm`` (it is what makes the runner send ``wsm_env_id`` / ``wsm_t`` /
  ``wsm_episode_len`` / ``wsm_repin``). Without them this server FAILS CLOSED rather than sharing
  one W between K episodes — a shared fast-weight singleton is the correctness trap this whole
  identity layer exists to prevent.
* ``--max-envs K`` (as the omega arms do), so K concurrent runners get K isolated slots.

Self-test::

    python serve_pi05_libero_stageq.py --self-test      # CPU only, no checkpoint
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # serve_batching.py
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # vla_training.eval.*

# The identity contract `eval_robocerebra_openpi.py::wsm_fields` emits under `--wsm`. Identical to
# the omega server's, on purpose: one harness flag serves every stateful arm.
IDENTITY_KEYS = ("wsm_env_id", "wsm_t", "wsm_episode_len")
OPTIONAL_KEYS = ("wsm_episode_id", "wsm_repin")
SIGNAL_KEYS = IDENTITY_KEYS + OPTIONAL_KEYS

DEFAULT_CONFIG = "pi05_robocerebra_stageq_q2"


class BatchValidationError(RuntimeError):
    """A request was rejected BEFORE any fast-weight state was mutated.

    Load-bearing under gather-batching: ``_validate_batch`` runs to completion before
    ``_prepare_batch`` touches a single ``W``, so a malformed request can be re-offered on its own
    without double-committing anyone. Anything raised LATER (non-finite ``O_t`` or ``W``) is a plain
    RuntimeError and stays fatal for the window, because by then state has moved.
    """


@dataclasses.dataclass
class _EpisodeState:
    """All mutable RoboTTT state owned by exactly one rollout client."""

    episode_id: str
    episode_len: int
    w: Any  # the live fast weights (a JAX pytree, batch axis 1)
    w0: Any  # this episode's starting W (same object; for ||W - W_0||)
    last_t: int = -1
    chunks: int = 0  # executed chunks seen since this episode's reset
    commits: int = 0  # inner-GD steps actually taken (== chunks when commit_stride == 1)
    repins: int = 0
    o_norm: float | None = None  # L2 of the last injected O_t (observability only)


class StageQPiInferWrapper:
    """Per-env RoboTTT fast-weight lifecycle around one pi0.5 policy, for K concurrent clients.

    The isolation guarantee: ``W`` lives in ``_EpisodeState``, keyed by ``wsm_env_id``, and every
    read/write goes through that env's own slot. Nothing about a request's treatment depends on who
    it was batched with — ``condition`` takes the env's own ``W`` and its own observation, and
    ``commit`` takes the env's own result row. Batch composition can therefore only change the
    policy's XLA kernel (see ``--policy-batch serial`` / ``--policy-pad-batch``), never the state
    machine. Fixture A in ``--self-test`` proves that at the action level.

    Re-pins: RoboCerebra teleports the sim to the demo's ground-truth state at every subtask
    boundary. Unlike the omega window — which describes a workspace the robot is no longer in and is
    cleared — ``W`` is an accumulated update, not a snapshot of the scene, and training never saw a
    teleport at all. ``--repin-scope keep`` (default) therefore carries W across the boundary, i.e.
    fast weights are episode-persistent as the recipe says. ``--repin-scope reset`` re-inits W at
    each boundary (the "fast weights are subtask-local" reading) and is offered only as a probe.
    """

    STATE_MODE = "per_env_isolated_stageq_v1"

    def __init__(
        self,
        policy,
        runner,
        *,
        max_envs: int = 1,
        commit_stride: int = 1,
        repin_resets_w: bool = False,
        fast_weights: str = "live",
        log_every: int = 20,
        metadata_extra: dict | None = None,
    ):
        if fast_weights not in ("live", "zero"):
            raise ValueError(f"fast_weights must be 'live' or 'zero', got {fast_weights!r}")
        if max_envs < 1:
            raise ValueError(f"max_envs must be >= 1, got {max_envs}")
        if commit_stride < 1:
            raise ValueError(f"commit_stride must be >= 1, got {commit_stride}")
        self._policy = policy
        self._runner = runner
        self._max_envs = int(max_envs)
        self._commit_stride = int(commit_stride)
        self._repin_resets_w = bool(repin_resets_w)
        self._fast_weights = fast_weights
        self._log_every = int(log_every)
        self._metadata_extra = dict(metadata_extra or {})
        self._cond_dim = int(runner.cond_dim()) if hasattr(runner, "cond_dim") else None
        if self._cond_dim is None:
            self._cond_dim = int(runner._module.cfg.cond_dim)  # noqa: SLF001
        self._states: dict[str, _EpisodeState] = {}
        self._lock = threading.RLock()
        self._commits = 0
        #: realised gather sizes, newest last (the K>1 canary reads it).
        self.batch_sizes: list[int] = []
        if fast_weights == "zero":
            logging.warning("*" * 100)
            logging.warning(
                "[serve-rc-q2] --fast-weights zero: O_t is forced to EXACT ZEROS and W "
                "never commits. This is the A0-parity CONTROL, not the A5 arm."
            )
            logging.warning("*" * 100)

    # ------------------------------------------------------------------ field parsing
    @staticmethod
    def _scalar(obs: dict, key: str):
        if key not in obs:
            raise BatchValidationError(
                f"[serve-rc-q2] missing required identity field {key!r}. This server keeps a "
                "PER-ENV fast-weight state and cannot key it without the identity fields; run the "
                "harness with --wsm (and give every runner its own --wsm-env-id)."
            )
        arr = np.asarray(obs[key])
        if arr.ndim != 0:
            raise BatchValidationError(f"[serve-rc-q2] field {key!r} must be scalar; got shape {arr.shape}")
        return arr.item()

    @staticmethod
    def _as_int(value, key: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise BatchValidationError(f"[serve-rc-q2] {key} must be an integer; got {value!r}")
        return int(value)

    def _identity(self, obs: dict) -> tuple[str, str, int, int, bool]:
        env_id = str(self._scalar(obs, "wsm_env_id"))
        if not env_id:
            raise BatchValidationError("[serve-rc-q2] wsm_env_id must be a non-empty string")
        t = self._as_int(self._scalar(obs, "wsm_t"), "wsm_t")
        if t < 0:
            raise BatchValidationError(f"[serve-rc-q2] wsm_t must be non-negative; got {t}")
        episode_len = self._as_int(self._scalar(obs, "wsm_episode_len"), "wsm_episode_len")
        if episode_len < 1:
            raise BatchValidationError(f"[serve-rc-q2] wsm_episode_len must be >= 1; got {episode_len}")
        episode_id = str(self._scalar(obs, "wsm_episode_id")) if "wsm_episode_id" in obs else ""
        repin = bool(np.asarray(obs["wsm_repin"]).item()) if "wsm_repin" in obs else False
        return env_id, episode_id, t, episode_len, repin

    def _validate_batch(self, obs_list: Sequence[dict]) -> list[tuple[str, str, int, int, bool]]:
        identities = [self._identity(obs) for obs in obs_list]

        seen: set[str] = set()
        duplicates: set[str] = set()
        for env_id, *_rest in identities:
            (duplicates if env_id in seen else seen).add(env_id)
        if duplicates:
            raise BatchValidationError(f"[serve-rc-q2] duplicate wsm_env_id values in one batch: {sorted(duplicates)}")

        new_envs = {env_id for env_id, _eid, t, _len, _rp in identities if t == 0 and env_id not in self._states}
        if len(self._states) + len(new_envs) > self._max_envs:
            raise BatchValidationError(
                f"[serve-rc-q2] active env-state bound exceeded: have {len(self._states)} "
                f"({sorted(self._states)}), new {sorted(new_envs)}, max {self._max_envs}; refusing "
                "live-state eviction (evicting a slot would silently restart that episode's fast "
                "weights). Give the K runners env0..env{K-1} and REUSE those ids across cells, or "
                "raise --max-envs."
            )

        reset_envs = {env_id for env_id, _eid, t, _len, _rp in identities if t == 0}
        active = {
            state.episode_id: env_id
            for env_id, state in self._states.items()
            if env_id not in reset_envs and state.episode_id
        }
        for env_id, episode_id, t, episode_len, _repin in identities:
            state = self._states.get(env_id)
            if t == 0:
                owner = active.get(episode_id)
                if episode_id and owner is not None and owner != env_id:
                    raise BatchValidationError(
                        f"[serve-rc-q2] duplicate active episode identity {episode_id!r} on envs "
                        f"{owner!r} and {env_id!r}"
                    )
                if episode_id:
                    active[episode_id] = env_id
                continue
            if state is None:
                raise BatchValidationError(f"[serve-rc-q2] env {env_id!r} sent t={t} before an explicit t=0 reset")
            if (episode_id, episode_len) != (state.episode_id, state.episode_len):
                raise BatchValidationError(
                    f"[serve-rc-q2] episode identity changed without a t=0 reset for env {env_id!r}: "
                    f"active={(state.episode_id, state.episode_len)!r}, "
                    f"got={(episode_id, episode_len)!r}"
                )
            if t <= state.last_t:
                raise BatchValidationError(
                    f"[serve-rc-q2] out-of-order wsm_t for env {env_id!r}: last={state.last_t}, got={t}"
                )
        return identities

    # ------------------------------------------------------------------ fast-weight lifecycle
    def _fresh_state(self, episode_id: str, episode_len: int) -> _EpisodeState:
        w = self._runner.init_state()
        # Same object, not a copy: w0 is the episode's starting point, kept alive for ||W - W_0||.
        return _EpisodeState(episode_id=episode_id, episode_len=episode_len, w=w, w0=w)

    def _prepare_batch(self, obs_list: Sequence[dict]) -> tuple[list[dict], list]:
        identities = self._validate_batch(obs_list)

        # 1. episode boundaries: a fresh W from the meta-learned init, for that env slot only.
        for env_id, episode_id, t, episode_len, _repin in identities:
            if t != 0:
                continue
            self._states[env_id] = self._fresh_state(episode_id, episode_len)
            logging.info(
                "[serve-rc-q2] env=%s episode=%r start: len=%d, W <- meta-learned init",
                env_id,
                episode_id,
                episode_len,
            )

        # 2. subtask-boundary re-pins.
        for env_id, _episode_id, t, _episode_len, repin in identities:
            if t == 0 or not repin:
                continue
            state = self._states[env_id]
            state.repins += 1
            if self._repin_resets_w:
                state.w = state.w0
            logging.info(
                "[serve-rc-q2] env=%s t=%d RE-PIN #%d: W %s",
                env_id,
                t,
                state.repins,
                "reset to the episode init" if self._repin_resets_w else "carried across",
            )

        # 3. APPLY. Each env conditions on ITS OWN held-entering W; nothing here reads another row.
        injected: list[dict] = []
        for obs, (env_id, _episode_id, t, _episode_len, _repin) in zip(obs_list, identities):
            state = self._states[env_id]
            item = {key: value for key, value in obs.items() if key not in SIGNAL_KEYS}
            if self._fast_weights == "zero":
                cond = np.zeros((self._cond_dim,), dtype=np.float32)
            else:
                cond = np.asarray(self._runner.condition(state.w, item), dtype=np.float32)
                if cond.shape != (self._cond_dim,):
                    raise RuntimeError(
                        f"[serve-rc-q2] O_t has shape {cond.shape}, expected ({self._cond_dim},) for env {env_id!r}"
                    )
                if not np.isfinite(cond).all():
                    raise RuntimeError(
                        f"[serve-rc-q2] NON-FINITE robottt_cond at env={env_id!r} t={t} (diverged fast weights?)"
                    )
            # Injected AFTER the signal-key strip: robottt_cond is a real model input, and
            # `LiberoStageQInputs` is the transform that forwards it (`LiberoInputs` would drop it).
            item["robottt_cond"] = cond
            state.o_norm = float(np.linalg.norm(cond))
            state.last_t = t
            injected.append(item)
        return injected, identities

    @staticmethod
    def _commit_inputs(result: dict) -> tuple[np.ndarray, np.ndarray]:
        """The MODEL-SPACE (normalised, padded) state/action rows of one result (fail closed).

        The commit must consume exactly the representation training consumed. Re-deriving it by
        inverting Unnormalize would silently drift (quantile stats, zero-padded tail dims), so the
        policy is built with ``expose_norm_actions=True`` and anything else refuses loudly.
        """
        if not (isinstance(result, dict) and "norm_state" in result and "norm_actions" in result):
            raise RuntimeError(
                "[serve-rc-q2] the RoboTTT commit needs model-space norm_state/norm_actions on the "
                "policy result; build the policy with expose_norm_actions=True"
            )
        return (
            np.asarray(result["norm_state"], dtype=np.float32),
            np.asarray(result["norm_actions"], dtype=np.float32),
        )

    def _commit(self, identities, results) -> None:
        """Exactly ONE conditional commit per executed chunk, per env (serving invariant D-1).

        Runs at the END of the same infer/infer_batch call that produced the chunk, i.e. strictly
        before that env's next condition — so "commit before the next apply" is already the live
        order and needs no hoisting.
        """
        for (env_id, _episode_id, t, _episode_len, _repin), result in zip(identities, results):
            state = self._states[env_id]
            # The model-space contract is checked in EVERY mode, including the zero control: the
            # mode may change what happens to W, never what the loop is allowed to commit on.
            norm_state, norm_actions = self._commit_inputs(result)
            state.chunks += 1
            due = self._fast_weights == "live" and (state.chunks % self._commit_stride == 0)
            if due:
                w_next = self._runner.commit(state.w, norm_state, norm_actions)
                if not self._runner.is_finite(w_next):
                    raise RuntimeError(
                        f"[serve-rc-q2] NON-FINITE fast weights after commit at env {env_id!r} "
                        f"t={t} (commit #{state.commits + 1})"
                    )
                state.w = w_next
                state.commits += 1
                self._commits += 1
                if self._log_every and self._commits % self._log_every == 0:
                    logging.info(
                        "[serve-rc-q2] env=%s t=%d commit #%d: ||W-W_0||=%.4g ||O_t||=%.4g",
                        env_id,
                        t,
                        state.commits,
                        self._w_delta(state),
                        state.o_norm or 0.0,
                    )
            # The model-space rows exist only to feed the commit; keep them out of client responses
            # (and therefore out of the persisted rollout shards).
            result.pop("norm_state", None)
            result.pop("norm_actions", None)

    def _w_delta(self, state: _EpisodeState) -> float:
        import jax
        import jax.numpy as jnp

        leaves = jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), state.w, state.w0))
        return float(np.sqrt(sum(float(x) for x in leaves)))

    # ------------------------------------------------------------------ policy calls
    def infer(self, obs: dict, **kwargs) -> dict:
        """LEGACY single-request path (K=1, gather off): one env's O_t, ``Policy.infer`` at batch 1."""
        with self._lock:
            self.batch_sizes.append(1)
            injected, identities = self._prepare_batch([obs])
            result = self._policy.infer(injected[0], **kwargs)
            self._commit(identities, [result])
            return result

    def _serve(self, obs_list: Sequence[dict], **kwargs) -> list[dict]:
        injected, identities = self._prepare_batch(list(obs_list))
        results = self._policy.infer_batch(injected, **kwargs)
        if len(results) != len(injected):
            raise RuntimeError(
                f"[serve-rc-q2] policy.infer_batch returned {len(results)} results for {len(injected)} requests"
            )
        self._commit(identities, results)
        return results

    @staticmethod
    def _duplicated_env_ids(obs_list: Sequence[dict]) -> set[str]:
        seen: dict[str, int] = {}
        for obs in obs_list:
            try:
                env_id = str(np.asarray(obs["wsm_env_id"]).item())
            except Exception:  # noqa: BLE001 - unparseable requests are rejected on their own merits
                continue
            seen[env_id] = seen.get(env_id, 0) + 1
        return {env_id for env_id, count in seen.items() if count > 1}

    def infer_batch(self, obs_list: Sequence[dict], **kwargs) -> list[dict]:
        """Serve one gather window. One entry per request, IN ORDER; an entry may be an Exception,
        which the gather routes to that caller alone. Without the per-request fallback one malformed
        runner would kill the K-1 healthy multi-hour episodes batched beside it."""
        with self._lock:
            requests = list(obs_list)
            if not requests:
                return []
            self.batch_sizes.append(len(requests))
            try:
                return self._serve(requests, **kwargs)
            except BatchValidationError as error:
                if len(requests) == 1:
                    raise
                logging.warning(
                    "[serve-rc-q2] window of %d rejected at validation (%s); re-offering "
                    "each request on its own so healthy envs survive",
                    len(requests),
                    error,
                )

            duplicated = self._duplicated_env_ids(requests)
            results: list[Any] = []
            for obs in requests:
                try:
                    env_id = str(np.asarray(obs.get("wsm_env_id", "")).item())
                except Exception:  # noqa: BLE001
                    env_id = ""
                if env_id and env_id in duplicated:
                    results.append(
                        BatchValidationError(
                            f"[serve-rc-q2] duplicate wsm_env_id {env_id!r} in one batch: two runners "
                            "share an env slot, which would interleave ONE fast-weight state between "
                            "two episodes. Give every runner its own --wsm-env-id."
                        )
                    )
                    continue
                try:
                    results.append(self._serve([obs], **kwargs)[0])
                except Exception as error:  # noqa: BLE001 - routed to this caller only
                    results.append(error)
            return results

    @property
    def metadata(self) -> dict:
        metadata = dict(getattr(self._policy, "metadata", {}) or {})
        metadata.update(
            {
                "stageq_state_mode": self.STATE_MODE,
                "stageq_fast_weights": self._fast_weights,
                "stageq_commit_stride": self._commit_stride,
                "stageq_commit_cadence": "one inner-GD step per EXECUTED chunk (harness replan cadence)",
                "stageq_repin_resets_w": self._repin_resets_w,
                "stageq_max_envs": self._max_envs,
                "stageq_cond_dim": self._cond_dim,
                "stageq_required_identity_fields": list(IDENTITY_KEYS),
                "stageq_optional_fields": list(OPTIONAL_KEYS),
                **self._metadata_extra,
            }
        )
        return metadata

    def gather_stats(self) -> dict:
        sizes = list(self.batch_sizes)
        return {
            "policy_calls": len(sizes),
            "requests": sum(sizes),
            "mean_batch": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max_batch": max(sizes) if sizes else 0,
            "hist": {size: sizes.count(size) for size in sorted(set(sizes))},
            "commits": self._commits,
        }


# =============================================================================================
# build helpers
# =============================================================================================
def link_assets(checkpoint: pathlib.Path, config_name: str, link_root: str | None) -> pathlib.Path:
    """``<root>/<config> -> <ckpt>/assets`` (the ``serve_pi05_libero.py`` trick, verbatim)."""
    root = pathlib.Path(link_root or checkpoint.parent / "_serve_assets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    link = root / config_name
    if not link.exists():
        link.symlink_to(checkpoint / "assets", target_is_directory=True)
    return root


def build_arm_policy(checkpoint: pathlib.Path, config_name: str, link_root: str | None):
    """Build the Q2 policy with ``expose_norm_actions=True``. Returns ``(policy, train_config)``."""
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config
    from openpi.training.config import AssetsConfig

    root = link_assets(checkpoint, config_name, link_root)
    train_config = _config.get_config(config_name)
    if not getattr(train_config.model, "robottt", False):
        raise SystemExit(
            f"[serve-rc-q2] config {config_name!r} has robottt=False: it consumes no robottt_cond "
            "and must be served by scripts/robocerebra/serve_pi05_libero.py. Refusing to attach a "
            "fast-weight loop to a policy that cannot read it."
        )
    data = train_config.data
    if getattr(data, "assets", None) is not None and getattr(data.assets, "assets_dir", None):
        logging.warning(
            "[serve-rc-q2] ignoring assets_dir=%s from the config; serving the checkpoint's own assets via %s",
            data.assets.assets_dir,
            root,
        )
        data = dataclasses.replace(data, assets=AssetsConfig(assets_dir=None, asset_id=data.assets.asset_id))
    train_config = dataclasses.replace(train_config, assets_base_dir=str(root), data=data)
    # expose_norm_actions is REQUIRED, not optional: the commit consumes the result's model-space
    # rows, and there is no other way to get them out of Policy.
    policy = policy_config.create_trained_policy(train_config, checkpoint, expose_norm_actions=True)
    return policy, train_config


def assert_robottt_loaded(policy, *, allow_untrained: bool = False) -> dict:
    """Refuse to serve unless the TRAINED ``robottt_fast`` subtree was actually restored.

    Finiteness catches diverged checkpoints. The trained-vs-init check uses the tanh gate ``alpha``:
    it is initialised to the constant ``gate_init`` everywhere, so an alpha still exactly equal to
    that constant means the subtree was re-initialised rather than loaded — in which case ``O_t`` is
    a meaningless near-zero read and the eval would silently score a broken policy. (This is the
    RoboCasa ``serve_pi_05_robottt.assert_robottt_loaded`` check, unchanged.)

    Note the sharper version of the same trap: ``readout`` is ZERO-initialised, so an untrained
    subtree gives ``O_t == 0`` for every W — the mechanism is a mathematical no-op at init, which is
    exactly why "it ran without error" proves nothing here.
    """
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model  # noqa: SLF001
    if not getattr(model, "robottt", False) or not hasattr(model, "robottt_fast"):
        raise SystemExit("[serve-rc-q2] served model has no robottt_fast subtree; wrong config")
    fast = model.robottt_fast
    arrays = [jnp.asarray(v.value) for _path, v in nnx.state(fast, nnx.Param).flat_state()]
    n_bad = sum(int(not bool(jnp.isfinite(a).all())) for a in arrays)
    if n_bad:
        raise SystemExit(
            f"[serve-rc-q2] robottt_fast has NON-FINITE params ({n_bad}/{len(arrays)} "
            "bad) — diverged checkpoint; refusing to serve."
        )
    alpha = jnp.asarray(fast.alpha[...])
    readout_kernel = jnp.asarray(fast.readout.kernel[...])
    untrained_alpha = bool(jnp.all(alpha == float(fast.cfg.gate_init)))
    untrained_readout = bool(jnp.all(readout_kernel == 0.0))
    if (untrained_alpha or untrained_readout) and not allow_untrained:
        raise SystemExit(
            "[serve-rc-q2] robottt_fast looks UNTRAINED "
            f"(alpha==gate_init: {untrained_alpha}, readout kernel all-zero: {untrained_readout}) — "
            "the trained subtree was NOT restored, so O_t would be identically zero and this arm "
            "would score exactly like A0. Refusing. Pass --allow-untrained-fast-weights only for a "
            "deliberate plumbing smoke test."
        )
    eta = float(fast.inner_lr())
    gate = float(jnp.max(jnp.abs(jnp.tanh(alpha))))
    gate_mean = float(jnp.mean(jnp.abs(jnp.tanh(alpha))))
    base_lr = float(fast.cfg.base_inner_lr)
    scalars = {
        "n_param_tensors": len(arrays),
        "inner_lr_eta": eta,
        "base_inner_lr": base_lr,
        "alpha_max_abs_tanh": gate,
        "alpha_mean_abs_tanh": gate_mean,
        "alpha_dim": int(alpha.size),
        "token_dim": int(fast.cfg.token_dim),
        "cond_dim": int(fast.cfg.cond_dim),
        "num_registers": int(fast.cfg.num_registers),
        "window_len_trained": int(fast.cfg.window_len),
    }
    logging.info(
        "[serve-rc-q2] robottt_fast restored: %d param tensors, all finite, "
        "max|tanh(alpha)|=%.4g mean=%.4g, eta=%.4g (base %.4g), d=%d C=%d N=%d",
        len(arrays),
        gate,
        gate_mean,
        eta,
        base_lr,
        scalars["token_dim"],
        scalars["cond_dim"],
        scalars["num_registers"],
    )
    return scalars


def build_runner(policy):
    """The RoboCasa serve runner, imported (not copied) so the normalised-space contract is single."""
    from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner

    return RoboTTTServeRunner(policy)


# =============================================================================================
# CPU-only self test: the two fixtures
# =============================================================================================
_EPISODE_ID = "Ideal/case1/trial0"


def _obs(
    env_id: str,
    t: int,
    episode_len: int,
    *,
    repin: bool = False,
    episode_id: str = _EPISODE_ID,
    seed: int | None = None,
) -> dict:
    """One RoboCerebra request, exactly as ``eval_robocerebra_openpi.py`` builds it under --wsm."""
    rng = np.random.default_rng(abs(hash((env_id, t))) % (2**32))
    item = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.normal(0, 1, 8).astype(np.float32),
        "prompt": "pick up the plate",
        "wsm_env_id": env_id,
        "wsm_episode_id": episode_id,
        "wsm_t": t,
        "wsm_episode_len": episode_len,
        "wsm_repin": repin,
    }
    if seed is not None:
        item["policy_noise_seed"] = np.uint32(seed)
    return item


def _fast_weight_module(*, token_dim: int = 64, fast_hidden: int = 32, cond_dim: int = 16):
    """A REAL ``RoboTTTFastWeights`` at toy width — the shipped module, not a re-implementation."""
    from flax import nnx
    from openpi.models.robottt_fast_weights import RoboTTTConfig, RoboTTTFastWeights

    cfg = RoboTTTConfig(
        fast_weights=True,
        workspace=False,
        token_dim=token_dim,
        fast_hidden=fast_hidden,
        cond_dim=cond_dim,
        num_registers=4,
        state_dim=32,
        action_dim=32,
        action_horizon=10,
    )
    module = RoboTTTFastWeights(cfg, rngs=nnx.Rngs(0))
    # An untrained subtree has a ZERO readout, so O_t would be identically zero and the fixture
    # would pass trivially. Give it a trained-looking head (this is what a real checkpoint has).
    rng = np.random.default_rng(7)
    import jax.numpy as jnp

    module.readout.kernel[...] = jnp.asarray(rng.normal(0, 0.5, module.readout.kernel[...].shape), jnp.float32)
    module.readout.bias[...] = jnp.asarray(rng.normal(0, 0.1, module.readout.bias[...].shape), jnp.float32)
    module.alpha[...] = jnp.asarray(rng.normal(0.5, 0.2, (cond_dim,)), jnp.float32)
    return module


class _FakeStateTransform:
    """Stands in for ``policy._input_transform``: raw request -> the model-space ``state`` row."""

    def __init__(self, state_dim: int = 32):
        self._state_dim = state_dim

    def __call__(self, data: dict) -> dict:
        raw = np.asarray(data["observation/state"], dtype=np.float32)
        state = np.zeros((self._state_dim,), dtype=np.float32)
        state[: raw.shape[0]] = raw
        return {"state": state}


class _FastWeightFakePolicy:
    """Row-independent stand-in for the pi policy that still exposes the REAL fast-weight module.

    Actions are a deterministic, injective-enough function of (model-space state, O_t), so if two
    envs' fast weights were ever crossed the action array WOULD move — which is what makes fixture
    A a real isolation test rather than a tautology. ``infer_batch`` is a plain per-row loop, i.e.
    exactly row-independent, so any K=1 vs K=8 difference must come from the state machine.

    The actions are squashed into [-1.2, 1.2]. That is not cosmetic: the commit's K/V tokens are
    built from the executed action chunk, so an unbounded fake action makes the inner GD step
    diverge within ~13 commits (observed) and the fixture would die on the finiteness guard rather
    than testing isolation. A real policy's normalised actions are O(1) for the same reason.
    """

    def __init__(self, module, *, horizon: int = 10, action_dim: int = 32):
        import types

        self._model = types.SimpleNamespace(robottt=True, robottt_fast=module)
        self._input_transform = _FakeStateTransform()
        self._horizon = horizon
        self._action_dim = action_dim
        self.seen: list[dict] = []

    def infer(self, obs: dict, **_kwargs) -> dict:
        self.seen.append(obs)
        state = self._input_transform(obs)["state"]
        cond = np.asarray(obs["robottt_cond"], dtype=np.float32)
        ramp = (np.arange(self._horizon, dtype=np.float32) + 1.0)[:, None]
        actions = (
            np.tanh(0.1 * ramp * state[None, :]) * (1.0 + 0.1 * float(np.tanh(cond.sum())))
            + 0.05 * float(np.tanh(cond[0]))
            + 0.05 * float(np.tanh(np.dot(cond, cond)))
        )
        return {
            "actions": np.ascontiguousarray(actions[:, :7]),
            "norm_state": state,
            "norm_actions": np.ascontiguousarray(actions),
        }

    def infer_batch(self, obs_list, **kwargs) -> list[dict]:
        return [self.infer(obs, **kwargs) for obs in obs_list]

    @property
    def metadata(self) -> dict:
        return {}


def _make_wrapper(max_envs: int, **kwargs) -> tuple[StageQPiInferWrapper, _FastWeightFakePolicy]:
    module = _fast_weight_module()
    policy = _FastWeightFakePolicy(module)
    runner = build_runner(policy)
    return StageQPiInferWrapper(policy, runner, max_envs=max_envs, log_every=0, **kwargs), policy


def _expect_raises(fn, needle: str) -> None:
    try:
        fn()
    except RuntimeError as error:
        assert needle in str(error), f"wrong error: {error}"
        return
    raise AssertionError(f"expected a RuntimeError containing {needle!r}")


def _fixture_a() -> None:
    """K=1 vs K=8 bit-identity: per-env fast-weight isolation under gather-batching."""
    print("-" * 92)
    print("FIXTURE A — K=1 vs K=8 bit-identity (per-env fast-weight isolation)")
    print("-" * 92)

    episode_len, replan, n_chunks = 900, 5, 40
    steps = [i * replan for i in range(n_chunks)]
    repin_at = {10, 22, 31}  # chunk indices where the subject env crosses a subtask boundary
    subject = "env3"

    def subject_obs(i: int) -> dict:
        return _obs(subject, steps[i], episode_len, repin=(i in repin_at), seed=1000 + i)

    # ---- run 1: the subject alone, K=1, the legacy single-request path.
    solo, _ = _make_wrapper(max_envs=1)
    solo_actions = [np.asarray(solo.infer(subject_obs(i))["actions"]) for i in range(n_chunks)]
    solo_state = solo._states[subject]  # noqa: SLF001
    solo_trace = (solo_state.chunks, solo_state.commits, solo._w_delta(solo_state))  # noqa: SLF001

    # ---- run 2: the same subject inside K=8 windows whose OTHER seven envs are churning:
    #      staggered starts, mid-batch t=0 resets, mid-batch re-pins, and a batch size that swings
    #      between 1 and 8 so the subject is not even always in the same row.
    shared, _ = _make_wrapper(max_envs=8)
    neighbours = [f"env{j}" for j in range(8) if f"env{j}" != subject]
    clocks = {name: -1 for name in neighbours}
    episodes = {name: "" for name in neighbours}
    order_rng = np.random.default_rng(0)
    batch_actions: list[np.ndarray] = []
    sizes: list[int] = []
    neighbour_resets = neighbour_repins = 0
    for i in range(n_chunks):
        batch = [subject_obs(i)]
        for j, name in enumerate(neighbours):
            if i < j:  # staggered arrival: env j joins at window j
                continue
            if (i + 2 * j) % 9 == 8:  # sparse participation -> window size swings up to K=8
                continue
            local = clocks[name] + 1
            if local == 0 or (local > 0 and local % 13 == 0):
                clocks[name], local = 0, 0  # a mid-batch NEW episode for that neighbour
                episodes[name] = f"ep/{name}/{i}"
                neighbour_resets += 1
                batch.append(_obs(name, 0, episode_len, episode_id=episodes[name], seed=i))
                continue
            clocks[name] = local
            repin = local % 7 == 3  # a mid-batch re-pin for that neighbour
            neighbour_repins += int(repin)
            batch.append(
                _obs(name, local * replan, episode_len, repin=repin, episode_id=episodes[name], seed=i * 31 + j)
            )
        order = order_rng.permutation(len(batch))
        shuffled = [batch[k] for k in order]
        sizes.append(len(shuffled))
        results = shared.infer_batch(shuffled)
        bad = [r for r in results if isinstance(r, Exception)]
        assert not bad, f"window {i} produced exceptions: {bad}"
        row = next(k for k, obs in enumerate(shuffled) if str(obs["wsm_env_id"]) == subject)
        batch_actions.append(np.asarray(results[row]["actions"]))

    shared_state = shared._states[subject]  # noqa: SLF001
    shared_trace = (shared_state.chunks, shared_state.commits, shared._w_delta(shared_state))  # noqa: SLF001

    deltas = [float(np.max(np.abs(a - b))) for a, b in zip(solo_actions, batch_actions)]
    max_delta = max(deltas)
    assert all(a.shape == b.shape for a, b in zip(solo_actions, batch_actions))
    assert max_delta == 0.0, f"K=1 vs K=8 diverged: max|Δaction| = {max_delta:.3e}"
    assert solo_trace[:2] == shared_trace[:2], f"commit counts differ: {solo_trace} vs {shared_trace}"
    assert solo_trace[2] == shared_trace[2], f"||W-W_0|| differs: {solo_trace[2]} vs {shared_trace[2]}"
    # The mechanism must actually be doing something, or bit-identity is trivially satisfied.
    assert solo_trace[2] > 0.0, "W never moved: the commit loop is inert, fixture A proves nothing"
    spread = max(deltas[1:]) if len(deltas) > 1 else 0.0

    print(f"  subject env        : {subject}, {n_chunks} chunks @ replan {replan}, {len(repin_at)} own re-pins")
    print(
        f"  neighbour churn    : {len(neighbours)} envs, {neighbour_resets} mid-batch t=0 resets, "
        f"{neighbour_repins} mid-batch re-pins"
    )
    print(
        f"  realised K         : min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.2f} "
        f"(hist { {s: sizes.count(s) for s in sorted(set(sizes))} })"
    )
    print(
        f"  W trajectory       : chunks={solo_trace[0]} commits={solo_trace[1]} "
        f"||W-W_0||={solo_trace[2]:.6f} (K=1) vs {shared_trace[2]:.6f} (K=8)"
    )
    print(f"  max|Δaction| over {n_chunks} chunks = {max_delta:.1f}   (post-first-chunk max {spread:.1f})")
    print("  PASS: K=8 with churning neighbours is BIT-IDENTICAL to K=1")


def _model_transforms_for(cfg):
    """``ModelTransformFactory()(cfg)`` when it imports, else the identical PI05 list inline.

    ``openpi.training.config`` imports ``robocasa`` at module scope in this fork, which the serve
    path always has on ``PYTHONPATH`` but a bare ``--self-test`` invocation may not. The fallback is
    the PI05 branch of that factory verbatim; it is only ever taken in the fixture.
    """
    try:
        from openpi.training import config as _config
    except ImportError as error:  # robocasa/robosuite not on PYTHONPATH
        import openpi.models.tokenizer as _tokenizer
        import openpi.transforms as transforms

        print(
            f"  [note] ModelTransformFactory unavailable ({error.name} missing); using the inline PI05 transform list"
        )
        return transforms.Group(
            inputs=[
                transforms.InjectDefaultPrompt(None),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(cfg.max_token_len), discrete_state_input=cfg.discrete_state_input
                ),
                transforms.PadStatesAndActions(cfg.action_dim),
            ]
        )
    return _config.ModelTransformFactory()(cfg)


def _wake_adarms(model, *, seed: int = 11) -> int:
    """Give the adaptive-RMSNorm modulation projections trained-looking weights. Returns how many.

    ``gemma.RMSNorm`` builds its modulation as ``nn.Dense(3*d, kernel_init=zeros)`` — adaLN-ZERO.
    At random init that Dense is identically zero, so ``adarms_cond`` (and therefore ``robottt_cond``,
    which is added to it) has PROVABLY no effect on the output. A fresh model can only ever show
    "live == plain", which would make the second half of fixture B unfalsifiable. Waking these
    kernels is the model-side twin of giving ``robottt_fast.readout`` non-zero values: both are
    zero-init seams that a trained checkpoint has moved off zero, and both must be off zero for the
    fixture to be able to fail.
    """
    import jax.numpy as jnp
    from flax import nnx

    rng = np.random.default_rng(seed)
    state = nnx.state(model, nnx.Param)
    touched = 0
    for path, var in state.flat_state():
        names = [str(part) for part in path]
        if "llm" in names and names[-1] == "kernel" and any(n.endswith("_norm_1") for n in names):
            var.value = jnp.asarray(rng.normal(0, 0.05, var.value.shape), var.value.dtype)
            touched += 1
    nnx.update(model, state)
    return touched


def _tiny_pi0_policies():
    """A tiny REAL pi0.5 (dummy gemma variants) served two ways: plain LIBERO, and Stage-Q.

    Both policies wrap the SAME model instance, so any action difference is the ``robottt_cond``
    seam and nothing else. No checkpoint is needed — the point of the fixture is the wiring, and a
    random-init model exercises the identical code path a trained one does. The one thing that
    cannot be left at init is the ``robottt_fast`` readout (zero-init => O_t identically zero), so
    it is given trained-looking values, exactly as ``assert_robottt_loaded`` demands of a real ckpt.
    """
    import jax.numpy as jnp
    import numpy as _np
    import openpi.transforms as transforms
    from flax import nnx
    from openpi.models import pi0 as _pi0
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import libero_policy, libero_stageq_policy
    from openpi.policies import policy as _policy
    from openpi.shared.normalize import NormStats

    cfg = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        robottt=True,
    )
    model = _pi0.Pi0(cfg, rngs=nnx.Rngs(0))

    rng = _np.random.default_rng(3)
    fast = model.robottt_fast
    fast.readout.kernel[...] = jnp.asarray(rng.normal(0, 0.05, fast.readout.kernel[...].shape), jnp.float32)
    fast.readout.bias[...] = jnp.asarray(rng.normal(0, 0.02, fast.readout.bias[...].shape), jnp.float32)
    fast.alpha[...] = jnp.asarray(rng.normal(0.5, 0.1, fast.alpha[...].shape), jnp.float32)
    n_adarms = _wake_adarms(model)

    norm_stats = {
        "state": NormStats(mean=_np.zeros(32, _np.float32), std=_np.ones(32, _np.float32)),
        "actions": NormStats(mean=_np.zeros(32, _np.float32), std=_np.ones(32, _np.float32)),
    }
    model_transforms = _model_transforms_for(cfg)

    def build(inputs_transform, *, expose_norm_actions: bool):
        return _policy.Policy(
            model,
            transforms=[inputs_transform, transforms.Normalize(norm_stats), *model_transforms.inputs],
            output_transforms=[
                *model_transforms.outputs,
                transforms.Unnormalize(norm_stats),
                libero_policy.LiberoOutputs(),
            ],
            expose_norm_actions=expose_norm_actions,
        )

    plain = build(libero_policy.LiberoInputs(model_type=cfg.model_type), expose_norm_actions=False)
    stageq = build(libero_stageq_policy.LiberoStageQInputs(model_type=cfg.model_type), expose_norm_actions=True)
    return model, plain, stageq, n_adarms


def _fixture_b() -> None:
    """Zeroed fast weights == plain serve; live fast weights != plain serve."""
    print("-" * 92)
    print("FIXTURE B — zeroed O_t reproduces plain serve; live W changes the actions")
    print("-" * 92)

    model, plain, stageq, n_adarms = _tiny_pi0_policies()
    episode_len, replan, n_chunks = 900, 5, 4
    requests = [_obs("env0", i * replan, episode_len, seed=500 + i) for i in range(n_chunks)]

    # A K=1 client of `serve_pi05_libero.py`: the raw request, straight into Policy.infer. The wsm_*
    # signal keys are stripped exactly as the wrapper strips them, so the two paths see one input.
    plain_actions = []
    for request in requests:
        item = {k: v for k, v in request.items() if k not in SIGNAL_KEYS}
        plain_actions.append(np.asarray(plain.infer(item)["actions"]))

    def run(mode: str):
        runner = build_runner(stageq)
        wrapper = StageQPiInferWrapper(stageq, runner, max_envs=1, fast_weights=mode, log_every=0)
        actions, o_norms, w_deltas = [], [], []
        for request in requests:
            actions.append(np.asarray(wrapper.infer(dict(request))["actions"]))
            state = wrapper._states["env0"]  # noqa: SLF001
            o_norms.append(state.o_norm)
            w_deltas.append(wrapper._w_delta(state))  # noqa: SLF001
        return actions, o_norms, w_deltas

    zero_actions, zero_o, zero_w = run("zero")
    live_actions, live_o, live_w = run("live")

    zero_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(plain_actions, zero_actions))
    live_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(plain_actions, live_actions))
    live_per_chunk = [float(np.max(np.abs(a - b))) for a, b in zip(plain_actions, live_actions)]

    assert zero_delta == 0.0, f"zeroed O_t did NOT reproduce plain serve: max|Δ| = {zero_delta:.3e}"
    assert live_delta > 0.0, (
        "live fast weights produced IDENTICAL actions to plain serve — the "
        "robottt_cond seam is not wired (check LiberoStageQInputs is in the "
        "config's data_transforms)"
    )
    assert all(d > 0.0 for d in live_per_chunk), f"some chunks unchanged under live W: {live_per_chunk}"
    assert zero_w[-1] == 0.0, "the zero control committed anyway"
    assert live_w[-1] > 0.0, "live W never moved"
    assert len(set(round(x, 9) for x in live_o)) > 1, "O_t never changed: the commit does not feed back"

    print(
        f"  model              : tiny real Pi0 (dummy gemma), pi05=True robottt=True, "
        f"action_horizon={model.action_horizon}, cond_dim={model.robottt_fast.cfg.cond_dim}, "
        f"{n_adarms} adaLN-zero modulation kernels woken"
    )
    print(f"  chunks             : {n_chunks} @ replan {replan}, CPU, no checkpoint")
    print(f"  zero-W vs plain    : max|Δaction| = {zero_delta:.1f}   <-- must be 0.0")
    print(f"  live-W vs plain    : max|Δaction| = {live_delta:.6f}  per chunk {[f'{d:.4f}' for d in live_per_chunk]}")
    print(f"  ||O_t||  live      : {[f'{x:.4f}' for x in live_o]}   zero: {[f'{x:.1f}' for x in zero_o]}")
    print(f"  ||W-W_0|| live     : {[f'{x:.4f}' for x in live_w]}   zero: {[f'{x:.1f}' for x in zero_w]}")
    print("  PASS: the zero control is byte-identical to plain serve AND the live loop moves the actions")


def _fixture_c() -> None:
    """The identity/ordering guards — a rejected request must not move any env's W."""
    print("-" * 92)
    print("FIXTURE C — identity, ordering and isolation guards")
    print("-" * 92)
    episode_len = 900
    guarded, policy = _make_wrapper(max_envs=1)
    guarded.infer(_obs("env0", 0, episode_len))
    guarded.infer(_obs("env0", 5, episode_len))
    before = (guarded._states["env0"].commits, guarded._w_delta(guarded._states["env0"]))  # noqa: SLF001
    checks = [
        (lambda: guarded.infer(_obs("env0", 5, episode_len)), "out-of-order"),
        (lambda: guarded.infer(_obs("env0", 3, episode_len)), "out-of-order"),
        (lambda: guarded.infer(_obs("env0", 10, episode_len, episode_id="other")), "identity changed"),
        (lambda: guarded.infer(_obs("env0", 10, 1350)), "identity changed"),
        (lambda: guarded.infer(_obs("env9", 10, episode_len)), "before an explicit t=0 reset"),
        (lambda: guarded.infer(_obs("env9", 0, episode_len)), "env-state bound exceeded"),
        (
            lambda: guarded.infer({k: v for k, v in _obs("env0", 10, episode_len).items() if k != "wsm_t"}),
            "missing required identity field",
        ),
        (
            lambda: guarded.infer({k: v for k, v in _obs("env0", 10, episode_len).items() if k != "wsm_env_id"}),
            "run the harness with --wsm",
        ),
    ]
    for fn, needle in checks:
        _expect_raises(fn, needle)
    after = (guarded._states["env0"].commits, guarded._w_delta(guarded._states["env0"]))  # noqa: SLF001
    assert before == after, f"a rejected request moved live fast weights: {before} -> {after}"

    duplicated = guarded.infer_batch([_obs("env0", 10, episode_len), _obs("env0", 11, episode_len)])
    assert len(duplicated) == 2 and all(isinstance(r, BatchValidationError) for r in duplicated), (
        f"both copies of a duplicated env id must be rejected: {duplicated}"
    )

    # A result without the model-space rows must refuse rather than commit on robot-space actions.
    class _NoNormPolicy(_FastWeightFakePolicy):
        def infer(self, obs, **kwargs):
            out = super().infer(obs, **kwargs)
            out.pop("norm_actions")
            return out

    module = _fast_weight_module()
    bad_policy = _NoNormPolicy(module)
    bad = StageQPiInferWrapper(bad_policy, build_runner(bad_policy), max_envs=1, log_every=0)
    _expect_raises(lambda: bad.infer(_obs("env0", 0, episode_len)), "expose_norm_actions=True")

    # The signal keys must never reach the policy, and robottt_cond always must.
    last = policy.seen[-1]
    assert set(SIGNAL_KEYS).isdisjoint(last), f"signal keys leaked into the policy: {sorted(last)}"
    assert "robottt_cond" in last and last["robottt_cond"].dtype == np.float32
    print(
        f"  validator rejects all {len(checks)} malformed cases and leaves live W untouched "
        f"(commits {before[0]}, ||W-W_0||={before[1]:.6f} before and after)"
    )
    print("  a duplicated env id rejects BOTH copies without raising for the window")
    print("  a result missing norm_state/norm_actions refuses instead of committing robot-space rows")
    print(f"  wsm_* signal keys stripped, robottt_cond[{last['robottt_cond'].shape[0]}] injected")
    print("  PASS")


def _self_test() -> None:
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    print("=" * 92)
    print("serve_pi05_libero_stageq self-test (CPU only, no checkpoint)")
    print("=" * 92)
    _fixture_a()
    _fixture_b()
    _fixture_c()
    print("=" * 92)
    print("ALL PASS")


# =============================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--self-test", action="store_true", help="CPU-only fixtures A/B/C; no GPU, no checkpoint, then exit"
    )
    parser.add_argument("--checkpoint", help="trained Q2 checkpoint dir (params/ + assets/)")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="the arm's registered TrainConfig name (must have robottt=True)"
    )
    parser.add_argument(
        "--assets-link-root",
        default=None,
        help="where to materialise <config>/<asset_id> (default <ckpt>/../_serve_assets)",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=None,
        help="concurrent env slots (default: WSM_ENVS_PER_GPU, which is also what "
        "flips the server's gather-batching on)",
    )
    parser.add_argument(
        "--commit-stride",
        type=int,
        default=1,
        help="commit W every Nth EXECUTED chunk. 1 (default) = the invariant: one "
        "inner-GD step per executed chunk, i.e. the harness replan cadence "
        "(5, plus the subtask-boundary re-plans). N>1 only exists to probe the "
        "train(stride 8)/serve(replan 5) cadence deviation.",
    )
    parser.add_argument(
        "--repin-scope",
        default="keep",
        choices=["keep", "reset"],
        help="what a wsm_repin does to W. keep (default): fast weights persist "
        "across the subtask-boundary teleport. reset: re-init W at every "
        "boundary (a probe, not the arm).",
    )
    parser.add_argument(
        "--fast-weights",
        default="live",
        choices=["live", "zero"],
        help="live (default) = the A5 arm. zero = O_t forced to zeros and no "
        "commits: the A0-parity CONTROL, provably identical to plain serve.",
    )
    parser.add_argument(
        "--allow-untrained-fast-weights",
        action="store_true",
        help="serve even if robottt_fast looks re-initialised (plumbing smoke only)",
    )
    parser.add_argument(
        "--policy-batch",
        default="batched",
        choices=["batched", "serial"],
        help="batched (default): ONE padded policy call per gather window. serial: "
        "one batch-1 Policy.infer per request — the same executable the K=1 "
        "path runs, hence bit-identical to a K=1 seeded run. Requires "
        "--deterministic-seeding on the harness.",
    )
    parser.add_argument(
        "--policy-pad-batch",
        type=int,
        default=8,
        help="replicate-pad every BATCHED policy call to this many rows so one XLA "
        "kernel serves every gather size (must be an openpi bucket: 4 or 8). "
        "0 = stock openpi bucketing.",
    )
    parser.add_argument("--log-every", type=int, default=20, help="log W stats every N commits; 0 = off")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")

    if args.self_test:
        _self_test()
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required (or pass --self-test)")

    from openpi.serving import websocket_policy_server
    from serve_batching import FixedPadBatchPolicy, gather_settings

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    logging.info("[serve-rc-q2] building %s from %s", args.config, checkpoint)
    policy, train_config = build_arm_policy(checkpoint, args.config, args.assets_link_root)
    scalars = assert_robottt_loaded(policy, allow_untrained=args.allow_untrained_fast_weights)
    runner = build_runner(policy)

    k_envs, gather_max_batch, gather_wait_ms = gather_settings()
    if args.policy_pad_batch and gather_max_batch % args.policy_pad_batch:
        raise SystemExit(
            f"[serve-rc-q2] gather max batch {gather_max_batch} is not a multiple of "
            f"--policy-pad-batch {args.policy_pad_batch}: the last chunk of a full window would run "
            "a DIFFERENT padded row count than the others."
        )
    # Wrapping the INNER policy (not the wrapper) is deliberate: the wrapper must see the real
    # request list to key per-env state, so pad rows may only appear after O_t is injected.
    batched = FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch, mode=args.policy_batch)

    wrapped = StageQPiInferWrapper(
        batched,
        runner,
        max_envs=int(k_envs if args.max_envs is None else args.max_envs),
        commit_stride=args.commit_stride,
        repin_resets_w=(args.repin_scope == "reset"),
        fast_weights=args.fast_weights,
        log_every=args.log_every,
        metadata_extra={
            "stageq_policy_pad_batch": int(args.policy_pad_batch),
            "stageq_policy_batch": args.policy_batch,
            "stageq_gather_max_batch": gather_max_batch,
            "stageq_gather_wait_ms": gather_wait_ms,
            "stageq_envs_per_gpu": k_envs,
            "stageq_config": args.config,
            "stageq_checkpoint": str(checkpoint),
            "stageq_train_chunk_stride": int(getattr(train_config.data, "stage_q_chunk_stride", -1)),
            "stageq_train_window_len": int(getattr(train_config.data, "stage_q_window_len", -1)),
            **{f"stageq_{k}": v for k, v in scalars.items()},
        },
    )

    logging.info(
        "[serve-rc-q2] READY on %s:%d — fast_weights=%s commit_stride=%d repin=%s eta=%.4g max|tanh(alpha)|=%.4g",
        args.host,
        args.port,
        args.fast_weights,
        args.commit_stride,
        args.repin_scope,
        scalars["inner_lr_eta"],
        scalars["alpha_max_abs_tanh"],
    )
    logging.info(
        "[serve-rc-q2] the harness MUST run with --wsm (this server keys W by wsm_env_id) "
        "and every runner needs its own --wsm-env-id"
    )
    if k_envs > 1:
        logging.info(
            "[serve-rc-q2] gather-batching ON: %d env slots, <=%d per batch, %.0f ms window, policy=%s (pad %d rows)",
            k_envs,
            gather_max_batch,
            gather_wait_ms,
            args.policy_batch,
            args.policy_pad_batch,
        )
    else:
        logging.info(
            "[serve-rc-q2] gather-batching OFF (WSM_ENVS_PER_GPU=1): LEGACY single-request "
            "path, Policy.infer at batch 1"
        )
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
