#!/usr/bin/env python3
"""WSM-CONDITIONED pi0.5 (openpi/JAX) policy server — the eval-side twin of
finetune_pi_05_with_wsm.py, and the pi counterpart of serve_groot_wsm.py.

The pi WSM finetune trained ONLY a zero-init TokenModulator inside Pi0.embed_prefix (the whole pi0.5
backbone frozen). At TRAIN time the GrootOpenpi dataset injected the precomputed causal omega_t window +
language into Observation.wsm_w_window/wsm_lang, so the modulator fired. At EVAL there is NO such path:
obs.wsm_w_window is None -> embed_prefix SKIPS the modulator -> the policy deploys as the pi Eval1
BASELINE and the trained modulator never runs. This server produces ONLINE omega_t per decision step so the
modulator actually conditions:

  1. BUILD the wsm policy from the finetune ckpt with a wsm=True config. UNLIKE groot, pi's modulator
     is PART of the JAX model (Pi0 instantiates self.wsm_modulator iff Pi0Config.wsm=True) and its
     trained weights live IN the ckpt (it was the only trainable param). create_trained_policy builds
     the model from Pi0Config(wsm=True, wsm_k_window=K) and BaseModelConfig.load intersect-trees the
     restored params with the fresh model state, so the wsm_modulator subtree (present in BOTH) is
     loaded. NO attach/restore step (unlike groot, whose HF from_pretrained drops the modulator).
  2. TAP the FROZEN backbone (Pi05BackboneTap) with the SAME finetune ckpt (its backbone is unchanged
     from pretrain, so features match the precompute) -> patch_tokens [1,192,2048] + lang_emb [1,2048].
  3. ENCODE via the backbone-agnostic WSMEvalConditioner: for pi the encoder's proprio slot = lang_emb
     (pi bakes state into the prompt; proprio_dim=2048) and cond_lang = the per-task global vector
     (task_lang_table). -> (w_window [K,512], lang [2048]).
  4. INJECT w_window/lang into the obs dict (keys wsm_w_window/wsm_lang) so RobocasaInputs passes them
     -> Observation.wsm_w_window/wsm_lang -> embed_prefix modulates the SigLIP image tokens.

Episode/step signaling is IN-BAND (openpi has NO reset channel — the websocket server only calls
policy.infer(obs)). The eval client stamps each obs with `wsm_t` (env step) + `wsm_task` (gym task
name). The wrapped infer reads them: wsm_t==0 -> new episode (conditioner.reset(table[task])); the
stride-8 grid is advanced by ENV STEP (floor(wsm_t/stride)), NOT per infer (pi replans every ~5 steps
!= stride 8), so we tap+step exactly once per new grid frame and reuse the latest window otherwise.

  python vla_training/eval/serve_pi_05_wsm.py \
      --finetune-ckpt <target_ft_wsm/pi_step100000_k2/.../59999> \
      --encoder-ckpt <wsm_runs/pi_wsm_v1/wsm_step100000.pt> \
      --task-lang-table <wsm_policy_feats/pi_step100000/task_lang_table.npz> \
      --k-window 2 --stride 8 --config-name pi05_robocasa_target_ft --port 8000

Run in the openpi-jax-latest env (jax + openpi + robocasa). Holds TWO openpi model instances (tap +
action) + a PyTorch encoder on one GPU — heavy but fine on a p5 H100 node.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from vla_training.eval._robottt_ablation import (
    RoboTTTAblation,
    apply_post_commit,
    delta_norm,
)

# pi feeds the WSM encoder's proprio slot with the backbone lang embedding (pi0.5 packs robot state into
# the prompt, so there is no separate state vector); proprio_dim=2048. groot would use state_emb=1536.
_PI_PROPRIO_DIM = 2048


#: Env knob for the frozen tap's CALL batch size. UNSET = today's behaviour, byte-for-byte, so this
#: patch cannot change any sealed arm's serve unless someone opts in.
TAP_MIN_BATCH_ENV = "WSM_TAP_MIN_BATCH"


def tap_min_batch(environ: Mapping[str, str] | None = None) -> int:
    """Rows the frozen tap must be called with; 0 means "call it with whatever we have".

    WHY THIS EXISTS (measured 2026-08-08, GR00T omega-sidecar work). The pi0.5 tap is jitted per
    input shape, and XLA selects a different kernel for small batches. Tapping the SAME frame at
    B=1/2/4 versus B=8/32 changes the patch tokens by up to 13-22 absolute, which the frozen
    WorkspaceModel amplifies into **max|d omega| ~ 1.43 on |omega| ~ 2.8**. The omega CACHE every
    workspace arm was trained against was built at B=32
    (`stage_s_cache_features.cache_task`, batch_size=32), while this serve taps the new-grid rows of
    one request batch -- with WSM_ENVS_PER_GPU=1 that is B=1. So the sealed pi workspace arms served
    omega from a different kernel than the one their conditioner was trained on.

    Setting WSM_TAP_MIN_BATCH=8 pads the call up to the cache's kernel. It is DEFAULT OFF because
    turning it on changes served omega, and the published pi numbers were produced without it; the
    A/B is the experiment, not a bug fix to apply retroactively.
    """
    env = os.environ if environ is None else environ
    raw = env.get(TAP_MIN_BATCH_ENV)
    if raw is None or raw == "":
        return 0
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"[serve-pi-wsm] {TAP_MIN_BATCH_ENV} must be an integer, got {raw!r}") from error
    if value < 0:
        raise RuntimeError(f"[serve-pi-wsm] {TAP_MIN_BATCH_ENV} must be >= 0, got {value}")
    return value


def pad_tap_batch(frames: dict, state: np.ndarray, prompts: Sequence[str], min_batch: int):
    """Replicate rows up to ``min_batch``; returns (frames, state, prompts, real_rows).

    Padding rows are copies of row 0. A transformer PREFIX has no cross-example interaction, so they
    cannot change the real rows' outputs -- only which XLA kernel runs. That claim is asserted
    against the real tap by `tests/test_pi_tap_min_batch.py` and by the box proof script, not merely
    argued here.
    """
    real_rows = len(prompts)
    if min_batch <= real_rows:
        return frames, state, list(prompts), real_rows
    extra = min_batch - real_rows
    frames = {
        key: np.concatenate([value, np.repeat(np.asarray(value)[:1], extra, axis=0)]) for key, value in frames.items()
    }
    state = np.concatenate([state, np.repeat(np.asarray(state)[:1], extra, axis=0)])
    return frames, state, list(prompts) + [prompts[0]] * extra, real_rows


def build_wsm_policy(finetune_ckpt: str, config_name: str, k_window: int, max_token_len: int):
    """Build the pi0.5 WSM policy from the finetune ckpt with a wsm=True config.

    The finetune's config_name (pi05_robocasa_target_ft) is NOT in the static openpi registry — it was
    built dynamically by _pi05_common.build_train_config at train time. We reconstruct the matching
    TrainConfig DIRECTLY (same model arch + RoboCasa transforms), with the only deltas vs the finetune
    being: wsm=True + wsm_k_window=K (so Pi0 instantiates the TokenModulator and create_trained_policy
    intersect-tree-loads its TRAINED weights from the ckpt — the modulator subtree is present in BOTH
    the fresh model and the restored params) and EMPTY data_dirs (norm stats come from the ckpt's
    assets/, never from the absent eval-node datasets — same trick as wsm_serve_rc.WSM_SERVE_NO_DATA).
    No GroupedSoup / dataset registry needed at deploy.
    """
    import os

    import openpi.models.pi0_config as pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as _config

    model = pi0_config.Pi0Config(pi05=True, max_token_len=int(max_token_len), wsm=True, wsm_k_window=int(k_window))
    cfg = _config.TrainConfig(
        name=config_name,
        exp_name="pi05_rc365_target_ft",
        model=model,
        data=_config.LeRobotRobocasaDataConfig(data_dirs=[]),
    )
    print(
        f"[serve-pi-wsm] config={config_name} pi05={model.pi05} wsm={model.wsm} "
        f"K={model.wsm_k_window} max_token_len={model.max_token_len}",
        flush=True,
    )
    policy = policy_config.create_trained_policy(cfg, os.path.expanduser(finetune_ckpt))
    return policy


def assert_modulator_loaded(policy) -> None:
    """Fail loudly unless the served JAX model has wsm enabled AND the modulator's zero-init last layer
    (gen) departed from all-zeros — i.e. the TRAINED modulator was actually restored. A zero gen ==
    identity == the Eval1 baseline (the exact 'serves as baseline' failure we must never repeat)."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    if not getattr(model, "wsm", False) or not hasattr(model, "wsm_modulator"):
        raise RuntimeError(
            "[serve-pi-wsm] served model has NO wsm_modulator — refusing to serve a "
            "baseline policy (was the config built with wsm=True?)"
        )
    mod = model.wsm_modulator
    # collect modulator param arrays (flat_state yields (path, VariableState); .value is the array);
    # assert all finite and the zero-init gen kernel is no longer all-zero.
    arrays = [jnp.asarray(v.value) for _path, v in nnx.state(mod, nnx.Param).flat_state()]
    n_bad = sum(int(not bool(jnp.isfinite(a).all())) for a in arrays)
    if n_bad:
        raise RuntimeError(
            f"[serve-pi-wsm] wsm_modulator has NON-FINITE params "
            f"({n_bad}/{len(arrays)} bad) — diverged ckpt; refusing."
        )
    gen_l2 = float(jnp.linalg.norm(jnp.asarray(mod.gen.kernel.value)))
    if gen_l2 == 0.0:
        raise RuntimeError(
            "[serve-pi-wsm] wsm_modulator.gen kernel is ALL ZERO (untrained zero-init) "
            "-> the modulator is an exact identity == the Eval1 BASELINE. Refusing to "
            "serve. (Did create_trained_policy load the finetune ckpt's modulator?)"
        )
    print(
        f"[serve-pi-wsm] ✓ modulator present: {len(arrays)} param tensors, all finite, "
        f"gen-kernel L2={gen_l2:.4g} (>0 => trained, fires at eval)",
        flush=True,
    )


@dataclass
class _EpisodeState:
    """All mutable online-workspace state owned by exactly one rollout client."""

    task: str
    demo_episode: Any
    conditioner: Any
    last_t: int = -1
    last_grid: int = -1
    frames_seen: int = 0
    current: tuple[np.ndarray, np.ndarray] | None = None
    # RoboTTT (Stage Q): per-env, GPU-resident fast-weight state W. Reset to the learned init at every
    # episode boundary (t=0), held fixed through all Euler/CFG passes of one chunk, then committed exactly
    # once after the executed chunk. None => no RoboTTT runner attached (Q0/Q1). Distinct from omega.
    robottt_w: Any = None
    robottt_commits: int = 0  # bookkeeping/spy: executed chunks seen since this episode's reset
    # The per-episode initial W (the meta-learned init this episode started from). Held so the A0
    # probe can report ||W - W_init|| and so the reset/decay ablations have an exact target. Same
    # pytree object as the initial robottt_w — no copy, no extra device traffic.
    robottt_w0: Any = None
    # L2 of the last conditioned O_t (probe only; the wrapper already holds O_t as a numpy row, so
    # this costs no extra forward pass). None until the first condition of the episode.
    robottt_o_norm: float | None = None


@dataclass(frozen=True)
class _WSMBatchTiming:
    """Completed wall times for one wrapper batch.

    Batch totals stay private. Responses expose only per-request amortized values, plus the
    corresponding batch cardinalities, so downstream summaries cannot accidentally count the same
    batch duration once per row.
    """

    request_batch_n: int
    new_grid_rows: frozenset[int]
    tap_batch_ms: float
    encoder_batch_ms: float
    prepare_batch_ms: float


class WSMPiInferWrapper:
    """Identity-aware online-``omega_t`` wrapper for one or many concurrent pi clients.

    Every request must carry scalar ``wsm_env_id``, ``wsm_task``, ``wsm_demo_episode``, and
    ``wsm_t``. State is keyed by env ID and includes immutable task/demo identity, a private causal
    conditioner buffer, the last grid, and the current ``omega_t`` window. ``wsm_t == 0`` resets
    only that env slot. Later requests must be strictly time-ordered and preserve episode identity.

    New Stage-S interfaces additionally set ``require_wsm_prompt``. In that mode every request must
    carry a non-empty canonical ``wsm_prompt`` used only by the frozen representation tap; it is
    removed before the action-policy transform sees the observation.

    ``infer`` deliberately retains the K=1 path: one frozen-tap call followed by ``policy.infer``.
    ``infer_batch`` batches new-grid tap work, groups independent conditioner buffers by equal history
    length for batched WorkspaceModel encoding, then calls ``policy.infer_batch`` once in request order.
    Env slots and causal frames per slot are bounded; overflow fails instead of evicting live state.
    """

    STATE_MODE = "per_env_isolated_v1"
    _IDENTITY_KEYS = ("wsm_env_id", "wsm_task", "wsm_demo_episode", "wsm_t")
    _SIGNAL_KEYS = _IDENTITY_KEYS + ("wsm_prompt",)

    def __init__(
        self,
        policy,
        tap,
        conditioner,
        task_lang_table,
        *,
        stride: int,
        expanded_table=None,
        max_envs: int | None = None,
        max_grid_frames: int = 1024,
        conditioner_factory: Callable[[], Any] | None = None,
        require_wsm_prompt: bool = False,
        robottt_runner: Any | None = None,
        robottt_ablation: Any | None = None,
        robottt_probe: Any | None = None,
    ):
        self._policy = policy
        self._tap = tap
        self._table = task_lang_table
        self._expanded = expanded_table
        self._require_wsm_prompt = bool(require_wsm_prompt)
        # Stage-Q RoboTTT fast-weight lifecycle. When present, each env owns a private W: init at reset,
        # condition (held) per chunk, commit exactly once per executed chunk. None keeps Q0/Q1 unchanged.
        self._robottt_runner = robottt_runner
        # Serve-only ablation knobs + A0 probe (both inert by default). An INERT ablation must leave
        # the commit path bit-identical to the decisive eval — see _commit_robottt.
        self._ablation = robottt_ablation if robottt_ablation is not None else RoboTTTAblation()
        self._probe = robottt_probe
        if self._ablation.active and robottt_runner is None:
            raise ValueError(
                f"[serve-pi-wsm] ROBOTTT_ABLATION={self._ablation.spec!r} but this server has no "
                "RoboTTT runner attached; the ablation would be a silent no-op"
            )
        if self._ablation.active:
            print(
                f"[serve-pi-wsm] *** ABLATED SERVE PATH: robottt_ablation={self._ablation.spec} "
                "*** (smoke tier; NOT comparable to sealed numbers)",
                flush=True,
            )
        if self._ablation.commit_first:
            # G7 does not apply at serve: _commit_robottt already runs at the END of the same
            # infer/infer_batch call that produced the chunk, i.e. strictly BEFORE the next request
            # is conditioned in _prepare_batch. There is never a pending uncommitted chunk to hoist.
            print(
                "[serve-pi-wsm] *** commitfirst REQUESTED BUT IS A NO-OP *** the serve loop already "
                "commits the executed chunk before the next condition (infer -> _commit_robottt -> "
                "next _prepare_batch); W trajectory is unchanged by this flag",
                flush=True,
            )
        # eta printed into every probe line: the trained step times the ablation's scale (if any).
        self._probe_eta = None
        if self._probe is not None and robottt_runner is not None:
            inner_lr = getattr(robottt_runner, "inner_lr", None)
            if callable(inner_lr):
                self._probe_eta = float(inner_lr()) * float(self._ablation.eta_scale)
        # Workspace-free mode (Q2: RoboTTT only): no tap, no encoder, no omega injection. All identity/
        # ordering/isolation validation and the RoboTTT lifecycle are unchanged. tap and conditioner must
        # be BOTH present (workspace on) or BOTH absent, and workspace-free requires a RoboTTT runner —
        # otherwise this wrapper adds nothing over the plain policy server.
        self._workspace = tap is not None
        if (tap is None) != (conditioner is None):
            raise ValueError("tap and conditioner must both be set (workspace) or both be None")
        if not self._workspace:
            if robottt_runner is None:
                raise ValueError("workspace-free wrapper requires a RoboTTT runner (Q2 serve)")
            if require_wsm_prompt:
                raise ValueError("workspace-free mode has no tap; wsm_prompt must not be required")
        # Stated at startup because it CHANGES SERVED OMEGA: an arm run with the pad is not
        # comparable to a sealed arm run without it, and after the fact the server log is the only
        # place that distinction survives.
        _min_batch = tap_min_batch()
        _batch_note = (
            f"kernel-matched pad to {_min_batch} rows (NOT the sealed configuration)"
            if _min_batch
            else "sealed default: the tap is called with the request batch"
        )
        print(
            f"[serve-pi-wsm] tap call batch: {TAP_MIN_BATCH_ENV}="
            f"{_min_batch if _min_batch else 'unset'} ({_batch_note})",
            flush=True,
        )
        self._stride = int(stride)
        if self._stride < 1:
            raise ValueError(f"stride must be >= 1, got {self._stride}")
        self._max_envs = int(os.environ.get("WSM_ENVS_PER_GPU", "1") if max_envs is None else max_envs)
        self._max_grid_frames = int(max_grid_frames)
        if self._max_envs < 1 or self._max_grid_frames < 1:
            raise ValueError(
                f"max_envs and max_grid_frames must be >= 1; got {self._max_envs}, {self._max_grid_frames}"
            )
        self._conditioner_template = conditioner
        self._conditioner_factory = conditioner_factory or self._default_conditioner_factory
        self._template_claimed = False
        self._states: dict[str, _EpisodeState] = {}
        self._lock = threading.RLock()

        # Source compatibility for the explicitly out-of-scope demo wrapper. The base class never
        # reads these singleton aliases.
        self._cond = conditioner
        self._last_grid = -1
        self._cur = None

    def _default_conditioner_factory(self):
        template = self._conditioner_template
        try:
            return type(template)(
                template.encoder,
                k_window=template.k,
                stride=template.stride,
                device=template.device,
            )
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "[serve-pi-wsm] cannot clone the online conditioner for another env; "
                "pass conditioner_factory=... (each env needs a private causal buffer)"
            ) from exc

    def _new_conditioner(self):
        if not self._template_claimed:
            self._template_claimed = True
            return self._conditioner_template
        conditioner = self._conditioner_factory()
        if conditioner is self._conditioner_template:
            raise RuntimeError(
                "[serve-pi-wsm] conditioner_factory returned the shared template; "
                "cross-client causal-state contamination would result"
            )
        return conditioner

    @staticmethod
    def _scalar(obs: dict, key: str):
        if key not in obs:
            raise RuntimeError(f"[serve-pi-wsm] missing required identity field {key!r}")
        arr = np.asarray(obs[key])
        if arr.ndim != 0:
            raise RuntimeError(f"[serve-pi-wsm] identity field {key!r} must be scalar; got shape {arr.shape}")
        return arr.item()

    def _identity(self, obs: dict) -> tuple[str, str, int | str, int]:
        raw_env_id = self._scalar(obs, "wsm_env_id")
        raw_task = self._scalar(obs, "wsm_task")
        raw_demo = self._scalar(obs, "wsm_demo_episode")
        raw_t = self._scalar(obs, "wsm_t")
        if raw_env_id is None or raw_task is None:
            raise RuntimeError("[serve-pi-wsm] wsm_env_id and wsm_task cannot be null")
        env_id, task = str(raw_env_id), str(raw_task)
        if not env_id or not task:
            raise RuntimeError("[serve-pi-wsm] wsm_env_id and wsm_task must be non-empty")
        if isinstance(raw_demo, (bool, np.bool_)) or not isinstance(raw_demo, (int, np.integer, str)):
            raise RuntimeError(f"[serve-pi-wsm] wsm_demo_episode must be an integer or string; got {raw_demo!r}")
        demo_episode = int(raw_demo) if isinstance(raw_demo, (int, np.integer)) else raw_demo
        if isinstance(raw_t, (bool, np.bool_)) or not isinstance(raw_t, (int, np.integer)):
            raise RuntimeError(f"[serve-pi-wsm] wsm_t must be a non-negative integer; got {raw_t!r}")
        t = int(raw_t)
        if t < 0:
            raise RuntimeError(f"[serve-pi-wsm] wsm_t must be non-negative; got {t}")
        if self._workspace and task not in self._table:
            raise RuntimeError(f"[serve-pi-wsm] unknown wsm_task={task!r}; table has {len(self._table)} tasks")
        return env_id, task, demo_episode, t

    def _prompt_for_tap(self, obs: dict, task: str) -> str:
        if self._require_wsm_prompt:
            if "wsm_prompt" not in obs:
                raise RuntimeError("[serve-pi-wsm] missing required signal field 'wsm_prompt'")
            raw = np.asarray(obs["wsm_prompt"])
            if raw.ndim != 0:
                raise RuntimeError(f"[serve-pi-wsm] signal field 'wsm_prompt' must be scalar; got shape {raw.shape}")
            prompt = raw.item()
            if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
                raise RuntimeError("[serve-pi-wsm] signal field 'wsm_prompt' must be a non-empty, trimmed string")
            return prompt

        expanded = self._expanded.get(task) if self._expanded else None
        return str(expanded or obs.get("wsm_prompt") or obs.get("prompt") or "")

    def _tap_batch(self, observations: Sequence[dict], prompts: Sequence[str]):
        """One frozen-backbone call for B new-grid observations, preserving row order."""
        if len(observations) != len(prompts) or not observations:
            raise ValueError("_tap_batch requires equally sized, non-empty observations/prompts")
        frames = {
            "agentview_left": np.stack([np.asarray(obs["observation/image"], dtype=np.uint8) for obs in observations]),
            "eye_in_hand": np.stack(
                [np.asarray(obs["observation/wrist_image"], dtype=np.uint8) for obs in observations]
            ),
            "agentview_right": np.stack(
                [np.asarray(obs["observation/right_image"], dtype=np.uint8) for obs in observations]
            ),
        }
        state = np.stack([np.asarray(obs["observation/state"], dtype=np.float32) for obs in observations])
        # Kernel-matching pad (default OFF; see tap_min_batch). Only the real rows are kept, so the
        # returned arrays are exactly what an unpadded call of this shape would have produced --
        # except that the tap ran on the same kernel the omega cache was built with.
        frames, state, call_prompts, real_rows = pad_tap_batch(
            frames, state, [str(prompt) for prompt in prompts], tap_min_batch()
        )
        result = self._tap.tap(frames, state, call_prompts)
        patch = np.asarray(result.patch_tokens, dtype=np.float32)[:real_rows]
        proprio = np.asarray(result.lang_emb, dtype=np.float32)[:real_rows]
        if patch.shape[0] != len(observations) or proprio.shape[0] != len(observations):
            raise RuntimeError(
                f"[serve-pi-wsm] tap batch mismatch: requested {len(observations)}, "
                f"got patch={patch.shape}, lang={proprio.shape}"
            )
        return patch, proprio

    def _tap_frame(self, obs: dict, prompt: str):
        """K=1 compatibility helper used by the historical demo-only subclass."""
        patch, proprio = self._tap_batch([obs], [prompt])
        return patch[0], proprio[0]

    @staticmethod
    def _numpy_float32(value) -> np.ndarray:
        value = value.detach() if hasattr(value, "detach") else value
        value = value.float() if hasattr(value, "float") else value
        value = value.cpu() if hasattr(value, "cpu") else value
        value = value.numpy() if hasattr(value, "numpy") else value
        return np.asarray(value, dtype=np.float32)

    def _validate_batch(self, obs_list: Sequence[dict]) -> list[tuple[str, str, int | str, int]]:
        identities = [self._identity(obs) for obs in obs_list]
        if self._require_wsm_prompt:
            # Validate the private signal before resetting or advancing any per-env causal state.
            for obs, (_env_id, task, _demo, _t) in zip(obs_list, identities):
                self._prompt_for_tap(obs, task)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for env_id, _task, _demo, _t in identities:
            (duplicates if env_id in seen else seen).add(env_id)
        if duplicates:
            raise RuntimeError(f"[serve-pi-wsm] duplicate wsm_env_id values in one batch: {sorted(duplicates)}")

        new_envs = {env_id for env_id, _task, _demo, t in identities if t == 0 and env_id not in self._states}
        if len(self._states) + len(new_envs) > self._max_envs:
            raise RuntimeError(
                f"[serve-pi-wsm] active env-state bound exceeded: have {len(self._states)}, "
                f"new {len(new_envs)}, max {self._max_envs}; refusing live-state eviction"
            )

        reset_envs = {env_id for env_id, _task, _demo, t in identities if t == 0}
        active_episodes = {
            (state.task, state.demo_episode): env_id
            for env_id, state in self._states.items()
            if env_id not in reset_envs
        }
        for env_id, task, demo_episode, t in identities:
            state = self._states.get(env_id)
            if t == 0:
                owner = active_episodes.get((task, demo_episode))
                if owner is not None and owner != env_id:
                    raise RuntimeError(
                        f"[serve-pi-wsm] duplicate active episode identity "
                        f"({task!r}, {demo_episode!r}) on envs {owner!r} and {env_id!r}"
                    )
                active_episodes[(task, demo_episode)] = env_id
                continue
            if state is None:
                raise RuntimeError(f"[serve-pi-wsm] env {env_id!r} sent t={t} before an explicit t=0 reset")
            if (task, demo_episode) != (state.task, state.demo_episode):
                raise RuntimeError(
                    f"[serve-pi-wsm] episode identity changed without t=0 reset for env {env_id!r}: "
                    f"active={(state.task, state.demo_episode)!r}, got={(task, demo_episode)!r}"
                )
            if t <= state.last_t:
                raise RuntimeError(
                    f"[serve-pi-wsm] out-of-order wsm_t for env {env_id!r}: last={state.last_t}, got={t}"
                )
            grid = math.floor(t / self._stride)
            if grid > state.last_grid + 1:
                raise RuntimeError(
                    f"[serve-pi-wsm] skipped causal grid for env {env_id!r}: "
                    f"last={state.last_grid}, got={grid} (t={t}, stride={self._stride})"
                )
            if grid != state.last_grid and t != grid * self._stride:
                raise RuntimeError(
                    f"[serve-pi-wsm] misaligned causal grid for env {env_id!r}: "
                    f"new grid {grid} must be observed at t={grid * self._stride}, got t={t}. "
                    "Use replan_steps == stride."
                )
        return identities

    @staticmethod
    def _step_conditioners(conditioners, patches, proprio):
        if len(conditioners) == 1:
            return [conditioners[0].step(patches[0], proprio[0])]
        step_many = getattr(type(conditioners[0]), "step_many", None)
        if not callable(step_many) or any(type(item) is not type(conditioners[0]) for item in conditioners):
            raise RuntimeError(
                "[serve-pi-wsm] batched requests require a homogeneous conditioner type with "
                "step_many; serial encoder fallback is intentionally disabled"
            )
        results = step_many(conditioners, patches, proprio)
        if len(results) != len(conditioners):
            raise RuntimeError(
                f"[serve-pi-wsm] conditioner step_many returned {len(results)} results for {len(conditioners)} envs"
            )
        return results

    def _prepare_batch(self, obs_list: Sequence[dict]) -> tuple[list[dict], list, _WSMBatchTiming]:
        prepare_started = time.perf_counter()
        if not obs_list:
            return [], [], _WSMBatchTiming(0, frozenset(), 0.0, 0.0, 0.0)
        identities = self._validate_batch(obs_list)

        for env_id, task, demo_episode, t in identities:
            if t != 0:
                continue
            if self._workspace:
                state = self._states.get(env_id)
                conditioner = state.conditioner if state is not None else self._new_conditioner()
                conditioner.reset(self._table[task])
            else:
                conditioner = None
            new_state = _EpisodeState(task=task, demo_episode=demo_episode, conditioner=conditioner)
            # Episode boundary: (re)initialize this env's private RoboTTT fast weights to the learned W_0.
            # A prior episode's online W is discarded with the old state; a fresh W starts from the init.
            if self._robottt_runner is not None:
                new_state.robottt_w = self._robottt_runner.init_state()
                # Same object, not a copy: the ablation's reset/decay targets and the probe's
                # ||W - W_init|| both need the episode's starting point kept alive.
                new_state.robottt_w0 = new_state.robottt_w
                if self._probe is not None:
                    self._probe.log(
                        "reset",
                        env_id=env_id,
                        task=task,
                        t=int(t),
                        commit_idx=0,
                        w_delta_norm=0.0,
                        o_norm=None,
                        eta_effective=self._probe_eta,
                        eta_scale=float(self._ablation.eta_scale),
                        ablation=self._ablation.spec,
                        ops=[],
                        wall_ms=0.0,
                    )
            self._states[env_id] = new_state

        tap_rows: list[int] = []
        prompts: list[str] = []
        for row, (obs, (env_id, task, _demo_episode, t)) in enumerate(zip(obs_list, identities)):
            state = self._states[env_id]
            grid = math.floor(t / self._stride)
            if grid != state.last_grid:
                if state.frames_seen >= self._max_grid_frames:
                    raise RuntimeError(
                        f"[serve-pi-wsm] causal-state frame bound exceeded for env {env_id!r}: "
                        f"max {self._max_grid_frames}"
                    )
                if self._workspace:
                    tap_rows.append(row)
                    prompts.append(self._prompt_for_tap(obs, task))
                else:
                    # No tap/encoder work in workspace-free mode; keep the identical grid bookkeeping
                    # so ordering/alignment validation stays bitwise the same across modes.
                    state.last_grid = grid
                    state.frames_seen += 1

        tap_batch_ms = 0.0
        encoder_batch_ms = 0.0
        if tap_rows:
            tap_started = time.perf_counter()
            patch, proprio = self._tap_batch([obs_list[row] for row in tap_rows], prompts)
            tap_batch_ms = (time.perf_counter() - tap_started) * 1000.0
            conditioners = [self._states[identities[row][0]].conditioner for row in tap_rows]
            encoder_started = time.perf_counter()
            conditioned = self._step_conditioners(conditioners, patch, proprio)
            for row, (omega_window, lang) in zip(tap_rows, conditioned):
                env_id, _task, _demo_episode, t = identities[row]
                state = self._states[env_id]
                grid = math.floor(t / self._stride)
                omega_np = self._numpy_float32(omega_window)
                lang_np = self._numpy_float32(lang)
                if not (np.isfinite(omega_np).all() and np.isfinite(lang_np).all()):
                    raise RuntimeError(
                        f"[serve-pi-wsm] NON-FINITE omega_t at env={env_id!r} t={t} grid={grid} "
                        "(mismatched/diverged encoder checkpoint?)"
                    )
                state.current = (omega_np, lang_np)
                state.last_grid = grid
                state.frames_seen += 1
            # The numpy conversions above synchronize CUDA-backed encoder outputs, so this is a
            # completed encoder latency rather than asynchronous dispatch time.
            encoder_batch_ms = (time.perf_counter() - encoder_started) * 1000.0

        injected: list[dict] = []
        for obs, (env_id, _task, _demo_episode, t) in zip(obs_list, identities):
            state = self._states[env_id]
            if self._workspace and state.current is None:
                raise RuntimeError(f"[serve-pi-wsm] no omega_t for env {env_id!r}; explicit t=0 reset was skipped")
            state.last_t = t
            item = dict(obs)
            if self._workspace:
                omega_window, lang = state.current
                item["wsm_w_window"] = np.asarray(omega_window, dtype=np.float32)
                item["wsm_lang"] = np.asarray(lang, dtype=np.float32)
            for key in self._SIGNAL_KEYS:
                item.pop(key, None)
            # RoboTTT APPLY: condition on the HELD-FIXED fast weights (W_{t-1}). This same vector rides
            # through every Euler/CFG evaluation of this chunk; W is not mutated here (the single commit
            # happens after the executed chunk, in _commit_robottt). robottt_cond is a real model input,
            # so it is injected AFTER the signal-key strip and never removed.
            if self._robottt_runner is not None:
                cond = np.asarray(self._robottt_runner.condition(state.robottt_w, obs), dtype=np.float32)
                if not np.isfinite(cond).all():
                    raise RuntimeError(
                        f"[serve-pi-wsm] NON-FINITE robottt_cond at env={env_id!r} t={t} (diverged fast weights?)"
                    )
                item["robottt_cond"] = cond
                if self._probe is not None:
                    # O_t is already on the host here, so the A0 magnitude costs no extra forward.
                    state.robottt_o_norm = float(np.linalg.norm(cond))
            injected.append(item)
        return (
            injected,
            identities,
            _WSMBatchTiming(
                request_batch_n=len(obs_list),
                new_grid_rows=frozenset(tap_rows),
                tap_batch_ms=tap_batch_ms,
                encoder_batch_ms=encoder_batch_ms,
                prepare_batch_ms=(time.perf_counter() - prepare_started) * 1000.0,
            ),
        )

    @staticmethod
    def _annotate_timing(
        results: Sequence[dict],
        batch_timing: _WSMBatchTiming,
        *,
        policy_batch_ms: float,
        end_to_end_batch_ms: float,
    ) -> list[dict]:
        """Attach unambiguous per-request telemetry without duplicating batch totals.

        Tap/encoder values are the new-grid operation's total divided by the number of rows that
        caused that work; cached-grid rows receive zero contribution. Preparation, policy-call, and
        end-to-end values divide the completed batch wall time by every request in the wrapper batch.
        """
        request_n = batch_timing.request_batch_n
        if request_n != len(results) or request_n < 1:
            raise RuntimeError(
                f"[serve-pi-wsm] timing/result batch mismatch: timing={request_n}, results={len(results)}"
            )
        new_grid_n = len(batch_timing.new_grid_rows)
        tap_per_active = batch_timing.tap_batch_ms / new_grid_n if new_grid_n else 0.0
        encoder_per_active = batch_timing.encoder_batch_ms / new_grid_n if new_grid_n else 0.0
        annotated: list[dict] = []
        for row, raw_result in enumerate(results):
            if not isinstance(raw_result, dict):
                raise RuntimeError(
                    f"[serve-pi-wsm] policy result row {row} is not a dict: {type(raw_result).__name__}"
                )
            result = dict(raw_result)
            timing = dict(result.get("policy_timing", {}) or {})
            is_new_grid = row in batch_timing.new_grid_rows
            timing.update(
                {
                    "wsm_request_batch_n": request_n,
                    "wsm_new_grid_batch_n": new_grid_n,
                    "wsm_tap_amortized_ms": tap_per_active if is_new_grid else 0.0,
                    "wsm_encoder_amortized_ms": encoder_per_active if is_new_grid else 0.0,
                    "wsm_prepare_amortized_ms": batch_timing.prepare_batch_ms / request_n,
                    "policy_call_amortized_ms": policy_batch_ms / request_n,
                    "wsm_end_to_end_amortized_ms": end_to_end_batch_ms / request_n,
                }
            )
            result["policy_timing"] = timing
            annotated.append(result)
        return annotated

    @staticmethod
    def _robottt_commit_inputs(result: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract the MODEL-SPACE (normalized, padded) state/action pair from one result (fail closed).

        RoboTTT commits on the finalized chunk the policy produced — never a ground-truth label — and
        it must consume exactly the representation training consumed: the normalized model rows, not
        the unnormalized robot-space actions the client receives. The policy must be built with
        expose_norm_actions=True; anything else is a serve/train mismatch and refuses loudly.
        """
        if not (isinstance(result, dict) and "norm_state" in result and "norm_actions" in result):
            raise RuntimeError(
                "[serve-pi-wsm] RoboTTT commit needs model-space norm_state/norm_actions on the "
                "policy result; build the policy with expose_norm_actions=True"
            )
        return (
            np.asarray(result["norm_state"], dtype=np.float32),
            np.asarray(result["norm_actions"], dtype=np.float32),
        )

    def _commit_robottt(self, identities, injected, results) -> None:
        """Exactly ONE conditional commit per executed chunk, per env (serving invariant, D-1).

        The held-W condition already happened in _prepare_batch; here each env's private W advances
        once, from the just-finalized chunk, and is finite-checked. No-op when no runner is attached.

        This is also the ONLY place the serve-time ablation knobs touch W (freeze/reset/decay/eta).
        With the default (inert) ablation `apply_post_commit` returns the committed pytree unchanged,
        so the decisive-eval path runs exactly the code it ran before.

        Ordering note (A5/G7): this runs at the END of the same infer/infer_batch call that produced
        the chunk — strictly before the next request is conditioned — so "commit before the next
        condition" is already the live order.
        """
        if self._robottt_runner is None:
            return
        ablation = self._ablation
        for (env_id, task, _demo, t), _item, result in zip(identities, injected, results):
            state = self._states[env_id]
            # The model-space contract is enforced in EVERY mode (including freeze): an ablation may
            # change what happens to W, never what the serve loop is allowed to commit on.
            norm_state, norm_actions = self._robottt_commit_inputs(result)
            commit_started = time.perf_counter()
            state.robottt_commits += 1
            ops: tuple[str, ...] = ()
            if ablation.freeze:
                # A1: no commit at all. W stays at this episode's init for the whole episode, so the
                # policy sees a constant O_t computed from W_0 — the causality control.
                ops = ("freeze",)
            else:
                w_next = self._robottt_runner.commit(state.robottt_w, norm_state, norm_actions)
                w_next, ops = apply_post_commit(
                    ablation, state.robottt_w, w_next, state.robottt_w0, state.robottt_commits
                )
                if not self._robottt_runner.is_finite(w_next):
                    raise RuntimeError(f"[serve-pi-wsm] NON-FINITE fast weights after commit at env {env_id!r}")
                state.robottt_w = w_next
            if self._probe is not None:
                self._probe.log(
                    "commit",
                    env_id=env_id,
                    task=task,
                    t=int(t),
                    commit_idx=state.robottt_commits,
                    w_delta_norm=delta_norm(state.robottt_w, state.robottt_w0),
                    o_norm=state.robottt_o_norm,
                    eta_effective=self._probe_eta,
                    eta_scale=float(ablation.eta_scale),
                    ablation=ablation.spec,
                    ops=list(ops),
                    wall_ms=(time.perf_counter() - commit_started) * 1000.0,
                )
            # The model-space rows exist only to feed the commit; keep them out of client responses
            # (and therefore out of persisted rollout shards).
            result.pop("norm_state", None)
            result.pop("norm_actions", None)

    def infer(self, obs: dict, **kwargs) -> dict:
        """Single-client path: preserve the original inner ``policy.infer`` semantics."""
        with self._lock:
            batch_started = time.perf_counter()
            injected, identities, batch_timing = self._prepare_batch([obs])
            policy_started = time.perf_counter()
            result = self._policy.infer(injected[0], **kwargs)
            policy_batch_ms = (time.perf_counter() - policy_started) * 1000.0
            self._commit_robottt(identities, injected, [result])
            end_to_end_batch_ms = (time.perf_counter() - batch_started) * 1000.0
            return self._annotate_timing(
                [result],
                batch_timing,
                policy_batch_ms=policy_batch_ms,
                end_to_end_batch_ms=end_to_end_batch_ms,
            )[0]

    def infer_batch(self, obs_list: Sequence[dict], **kwargs) -> list[dict]:
        """Prepare isolated online workspaces, then issue one batched policy call."""
        with self._lock:
            if not obs_list:
                return []
            batch_started = time.perf_counter()
            injected, identities, batch_timing = self._prepare_batch(list(obs_list))
            if not injected:
                return []
            policy_started = time.perf_counter()
            results = self._policy.infer_batch(injected, **kwargs)
            policy_batch_ms = (time.perf_counter() - policy_started) * 1000.0
            if len(results) != len(injected):
                raise RuntimeError(
                    f"[serve-pi-wsm] policy.infer_batch returned {len(results)} results for {len(injected)} requests"
                )
            self._commit_robottt(identities, injected, results)
            end_to_end_batch_ms = (time.perf_counter() - batch_started) * 1000.0
            return self._annotate_timing(
                results,
                batch_timing,
                policy_batch_ms=policy_batch_ms,
                end_to_end_batch_ms=end_to_end_batch_ms,
            )

    @property
    def metadata(self) -> dict:
        metadata = dict(getattr(self._policy, "metadata", {}) or {})
        metadata.update(
            {
                "wsm_state_mode": self.STATE_MODE,
                "infer_batch": True,
                "wsm_stride": self._stride,
                "wsm_max_envs": self._max_envs,
                "wsm_max_grid_frames": self._max_grid_frames,
                "wsm_required_identity_fields": list(self._IDENTITY_KEYS),
                "wsm_required_signal_fields": (["wsm_prompt"] if self._require_wsm_prompt else []),
                "wsm_robottt": self._robottt_runner is not None,
                "wsm_workspace": self._workspace,
                # Self-describing results: "" == the unmodified path, anything else is a smoke-tier
                # ablated arm and must never be reported as a sealed number.
                "robottt_ablation": self._ablation.spec,
                "robottt_ablation_detail": self._ablation.as_metadata(),
                "robottt_probe_log": getattr(self._probe, "path", None),
            }
        )
        return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True, help="pi WSM finetune ckpt dir (orbax step dir, e.g. .../59999)")
    ap.add_argument("--encoder-ckpt", required=True, help="FROZEN WorkspaceModel ckpt (wsm_step*.pt)")
    ap.add_argument("--task-lang-table", required=True, help="task_lang_table.npz (make_task_lang_table)")
    ap.add_argument(
        "--config-name",
        default="pi05_robocasa_target_ft",
        help="the pi WSM finetune's config_name (provenance label on the rebuilt TrainConfig)",
    )
    ap.add_argument(
        "--configs-dir",
        default=None,
        help="dir holding a standalone wsm_robocasa_configs.py for the backbone TAP "
        "(default: the tap's WSM_CONFIGS_DIR)",
    )
    ap.add_argument("--max-token-len", type=int, default=200, help="pi05 packs discretized state into the prompt")
    ap.add_argument("--k-window", type=int, default=2)
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--lang-dim", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=8, help="cache grid stride (advance grid by env step)")
    ap.add_argument(
        "--tap-prompt",
        default="expanded",
        choices=["expanded", "terse"],
        help="prompt fed to the backbone TAP: 'expanded' = per-task Qwen prompt (matches the "
        "cache, closes the WSM-only encoder-input shift); 'terse' = the env task string "
        "(the A/B 'without-fix' arm).",
    )
    ap.add_argument("--device", default="cuda", help="device for the PyTorch WSM encoder")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # This server has no RoboTTT runner (omega-only WSM interface), so an ablation set here would be
    # a silent no-op mislabeled as an arm. Fail at startup instead.
    from vla_training.eval._robottt_ablation import ablation_from_env

    _ablation = ablation_from_env()
    if _ablation.active:
        raise SystemExit(
            f"[serve-pi-wsm] ROBOTTT_ABLATION={_ablation.raw!r} set, but this server serves the "
            "omega-only WSM interface (no fast weights). Use serve_pi_05_robottt.py (q2) or "
            "serve_pi_05_wsm_cfg.py --interface tanh_robottt (q3)."
        )

    from openpi.serving import websocket_policy_server

    from vla_training.eval._groot_wsm_eval import (
        WSMEvalConditioner,
        load_task_expanded_table,
        load_task_lang_table,
    )
    from workspace_models.features.generate_policy_features import load_wsm

    # 1. action policy: the finetune ckpt built with wsm=True so the modulator is present + loaded.
    print(f"[serve-pi-wsm] building wsm policy from {args.finetune_ckpt}", flush=True)
    policy = build_wsm_policy(args.finetune_ckpt, args.config_name, args.k_window, args.max_token_len)
    assert_modulator_loaded(policy)  # never silently serve the baseline

    # 2. frozen backbone tap (SAME finetune ckpt; its backbone == pretrain, matches the precompute).
    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

    tap_config = "pi05_rc_mg60_bal33"  # the tap's own config (registers the RoboCasa pi05 transforms)
    print(f"[serve-pi-wsm] loading backbone tap (config={tap_config})", flush=True)
    tap = Pi05BackboneTap(
        args.finetune_ckpt, config_name=tap_config, **({"configs_dir": args.configs_dir} if args.configs_dir else {})
    )

    # 3. frozen WSM encoder + online-omega_t conditioner + per-task language table.
    encoder, _meta = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PI_PROPRIO_DIM)
    conditioner = WSMEvalConditioner(encoder, k_window=args.k_window, stride=args.stride, device=args.device)
    table = load_task_lang_table(args.task_lang_table)

    # --tap-prompt expanded: feed the backbone tap the per-task Qwen prompt (matches the cache features
    # the encoder was trained on). 'terse' = the env task string (the A/B without-fix arm).
    expanded_table = None
    if args.tap_prompt == "expanded":
        expanded_table = load_task_expanded_table(args.task_lang_table)
        if not expanded_table or not any(expanded_table.values()):
            raise RuntimeError(
                "[serve-pi-wsm] --tap-prompt expanded but task_lang_table has no non-empty "
                "'expanded' strings — regenerate it with `make_task_lang_table --cache-root`."
            )
        print(
            f"[serve-pi-wsm] tap-prompt=EXPANDED ({sum(bool(v) for v in expanded_table.values())}/"
            f"{len(expanded_table)} per-task strings present)",
            flush=True,
        )
    else:
        print("[serve-pi-wsm] tap-prompt=TERSE (env task string) — WSM-only encoder-input shift (A/B arm)", flush=True)

    wrapped = WSMPiInferWrapper(policy, tap, conditioner, table, stride=args.stride, expanded_table=expanded_table)
    print(
        f"[serve-pi-wsm] ✓ WSM-conditioned pi0.5 server ready on {args.host}:{args.port} "
        f"(K={args.k_window}, stride={args.stride}, {len(table)} tasks, encoder={args.encoder_ckpt})",
        flush=True,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
