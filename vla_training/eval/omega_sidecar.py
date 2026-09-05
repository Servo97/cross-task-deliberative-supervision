#!/usr/bin/env python3
"""ONLINE omega for a torch-only GR00T serve — the JAX producer chain, out of process.

WHY THIS EXISTS. The GR00T gated-DeltaNet arms (`dnw8`) condition on a causal WINDOW of omega, and
omega is not a function of the observation alone: the study's shared cache defines it as
``WSMEncoder(frozen pi0.5 backbone tap features)`` (omega manifest ``artifact:
pi05_workspace_omega`` -> ``frozen_pi_feature_source`` + ``workspace_model.config.backbone_dim
2048``). The disk cache covers the 323 TRAIN demos only; ReMemBench eval episodes are HELD OUT and
have no cached omega, and feeding a train demo's omega would be an oracle leak. So omega must be
produced ONLINE, exactly as the pi workspace arms do it in ``serve_pi_05_wsm_cfg.py``
(``Pi05BackboneTap`` -> ``WorkspaceModel`` -> ``causal_window_indices``).

That producer is a jax/openpi process. The GR00T serve is a torch-only venv that deliberately does
not carry the openpi SERVER package (`serve_groot_ws.serve`'s docstring says so). Rather than
install jax next to GR00T — which would fork the pinned openpi tree the pi arms are sealed against,
and put two XLA/torch CUDA runtimes in one address space — the producer runs as its own process,
co-located with its GR00T serve on the same GPU, and answers over localhost HTTP.

WHAT LIVES WHERE (the split is deliberate).
  * THIS PROCESS owns ALL episode state: one causal conditioner buffer per ``wsm_env_id``, reset at
    ``wsm_t == 0``, with the pi serve's exact identity/ordering discipline (see ``OmegaSidecar``).
    The GR00T serve stays stateless with respect to omega, so there is exactly ONE implementation of
    the episode-boundary rules in the study and it is the one the sealed pi arms already ran.
  * THE GR00T SERVE sends the SAME raw camera frames the runner sent it plus the ``wsm_*`` identity
    fields, and stashes the returned window on ``action_head._dn_eval_window``. See
    ``OmegaSidecarClient``.

FIDELITY (each claim is pinned to the code that defines it; a drift here is silent and fatal).
  * IMAGE SIZE. The pi arms' frames arrive at 224 px: ``run_remembench_eval.RESIZE = 224`` (:64) is
    the client default and ``run_remembench_box.sh:47`` keeps it for every non-groot serve, while a
    groot serve gets the env's NATIVE 256. ``run_remembench_eval.run_episode.prep`` (:225) skips the
    resample entirely when the requested size equals the native one — so the 256 px frame a groot
    serve receives IS the untouched env render, and applying ``resize_with_pad(., 224, 224)`` here
    reproduces the pi client's own ``prep`` BIT-FOR-BIT (``convert_to_uint8`` is the identity on the
    uint8 frames ``serve_groot_ws.pack_observation`` already hard-asserts). ``--tap-image-size 0``
    passes frames through untouched; that is what the offline-cache parity harness uses, because the
    cache was tapped from 256 px LeRobot frames (``stage_s_cache_features`` manifest
    ``frame_grid.image_hw [256, 256]``).

    AND THE SIZE DIFFERENCE TURNS OUT NOT TO MATTER — measured, 2026-08-08, not argued. The tap's
    OWN input transform ends in ``transforms.ResizeImages`` (openpi ``transforms.py:185-191``),
    which applies the same ``resize_with_pad``, and that helper short-circuits when the frame is
    already the target size. So a 256 px and a 224 px request converge on the identical model input:
    taps at ``--tap-image-size 224`` and ``0`` (native 256) agree to **patch max|d| = 0.000000** and
    give the same omega. An earlier revision of this file claimed a residual train/serve shift here;
    it does not exist, and the flag is kept only as an explicit statement of the wire contract.
  * TAP BATCH SIZE — THE ONE THAT DOES MATTER. The frozen tap is jitted per input shape, and XLA
    picks a different kernel for small batches: tapping the SAME frame at B=1/2/4 vs B=8/32 changes
    the patch tokens by up to 13-22 absolute (corr 0.9988), which the WorkspaceModel amplifies into
    **max|d omega| = 1.46 on |omega| ~ 2.8** — half the signal. The cache was built at B=32
    (``stage_s_cache_features.cache_task`` ``batch_size=32``), so an online B=1 tap CANNOT reproduce
    it. A transformer prefix has no cross-example interaction, so padding the batch with copies of
    the frame changes only which kernel runs, never the mathematics — verified: row 0 comes back
    **bit-identical to the cache (patch max|d| = 0.000000, lang 0.000000) whatever the filler rows
    contain**, and omega then matches the cache at the fp16 floor (9.5e-04). ``--tap-batch-size``
    therefore defaults to 8, which is what makes online omega equal to the omega the conditioner was
    actually trained on. Setting it to 1 reproduces the sealed pi arms' serve-time behaviour, which
    carries the 1.46 offset; that is a labelled ablation, not the default.
  * TAP. ``Pi05BackboneTap(config pi05_rc_mg60_bal33, --tap-ckpt)``, identical construction to
    ``serve_pi_05_wsm_cfg.main`` (:685). On the box the checkpoint is ``/data/work/ckpts/tap_149999``
    — the H300+MG feature source the omega manifest's ``frozen_pi_feature_source`` records.
  * ENCODER. ``load_wsm(--encoder-ckpt, device, proprio_dim=2048)``, identical to
    ``serve_pi_05_wsm_cfg.main`` (:689). ``--expect-encoder-sha256`` compares the file's digest
    against the omega manifest's ``encoder_checkpoint.sha256`` so a mismatched encoder — the GR00T
    Eval2 NaN-encoder failure class — cannot be served by accident.
  * WINDOW. ``WSMEvalConditioner(encoder, k_window=K, stride=8)``, whose ``step_many`` takes the
    window through the SHARED ``wsm_align.causal_window_indices`` (:161-165) — the same helper the
    GR00T train loader uses via ``wsm_align.window_at``, so train and serve pad identically
    (left-pad by repeating the oldest real grid row) and the rows arrive oldest..newest.
  * PROMPT. The frozen tap is fed the canonical private ``wsm_prompt`` and nothing else, matching
    ``--tap-prompt terse`` + ``require_wsm_prompt=True`` on the pi workspace serve
    (``serve_pi_05_wsm.WSMPiInferWrapper._prompt_for_tap``). The cache used the same canonical terse
    manifest (omega manifest ``conditioning.global_language_mode:
    canonical_terse_task_instruction``). The runner supplies it via ``--task-prompt-manifest``.

MEMORY. jax must NOT preallocate: this process shares a 97 GB GPU with the GR00T serve that is the
actual policy. ``main`` sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` before jax is imported unless
the caller pinned ``XLA_PYTHON_CLIENT_MEM_FRACTION`` itself, and reports both processes' usage at
startup and on ``/health``.

  python vla_training/eval/omega_sidecar.py \
      --tap-ckpt /data/work/ckpts/tap_149999 \
      --encoder-ckpt /data/work/wsm_artifacts/rmb/encoder.pt \
      --task-lang-table /data/work/wsm_artifacts/rmb/task_lang_table.npz \
      --configs-dir /data/work/rmb/rc_configs --k-window 8 --stride 8 --port 6000

Run in the openpi-jax venv (jax + openpi + torch), one process per GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import pathlib
import threading
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

#: The resolution the pi workspace serves actually received. NOT a guess: ``run_remembench_eval.py``
#: defines ``RESIZE = 224  # pi0.5 RoboCasa input resolution`` (:64) and ``run_remembench_box.sh``
#: (:47) hands that to every non-groot serve. A groot serve gets 256 (the env's native render), so
#: the sidecar must down-convert to reproduce the pi client's bytes. 0 == pass through.
PI_SERVE_IMAGE_SIZE = 224
#: Batch size the frozen tap is CALLED with. Not a throughput knob: the tap is jitted per input
#: shape and XLA selects a different kernel below B=8, which moves omega by ~1.46 on |omega|~2.8.
#: The omega cache was built at B=32 (stage_s_cache_features.cache_task batch_size=32), so the
#: online tap must be padded up to a batch that lands on the same kernel. Verified bit-exact at 8.
TAP_BATCH_SIZE = 8
#: pi0.5 packs the robot state into the prompt, so the WSM encoder's proprio slot is fed the
#: backbone LANGUAGE embedding (2048), not a state vector. Mirrors serve_pi_05_wsm._PI_PROPRIO_DIM.
PI_PROPRIO_DIM = 2048
#: The tap's own openpi config, verbatim from serve_pi_05_wsm_cfg.main (:680).
TAP_CONFIG_NAME = "pi05_rc_mg60_bal33"
#: Camera wire keys, in the order ``serve_pi_05_wsm._tap_batch`` (:378-388) maps them onto the tap's
#: LABEL view names. The tap itself re-orders model slots; this mapping must not.
WIRE_TO_TAP_VIEW = {
    "observation/image": "agentview_left",
    "observation/wrist_image": "eye_in_hand",
    "observation/right_image": "agentview_right",
}
#: Everything a request must carry. Same field names the sealed runner already sends.
REQUIRED_FIELDS = (
    "observation/image",
    "observation/wrist_image",
    "observation/right_image",
    "observation/state",
    "wsm_env_id",
    "wsm_task",
    "wsm_demo_episode",
    "wsm_t",
    "wsm_prompt",
)
#: The startup self-test replays the cache's own inputs, so the bar is essentially exact: the
#: measured agreement is 0.000000 and the smallest real regression (the small-batch kernel) is
#: 13-22 absolute on |cache|max ~45. 1e-3 relative is far below the latter and far above float noise.
_SELFTEST_MAX_REL = 1e-3
#: Wire path for one omega window, and for the capability/provenance frame.
OMEGA_PATH = "/omega"
HEALTH_PATH = "/health"
#: State-machine identifier published on /health. A client that does not recognize it must refuse to
#: run rather than assume the episode-boundary rules it was written against.
STATE_MODE = "per_env_isolated_v1"


def pin_jax_memory_env() -> dict:
    """Pin the XLA memory policy BEFORE jax is imported. Call from every entry point.

    THIS IS NOT ONLY ABOUT MEMORY. Measured 2026-08-08: the frozen tap's patch tokens depend on the
    XLA memory configuration, because the autotuner's algorithm choice does. The same frame, at the
    same tap batch size, on the same GPU, reproduced the omega cache EXACTLY (patch max|d| =
    0.000000) under ``PREALLOCATE=false``, ``MEM_FRACTION=0.15`` and ``MEM_FRACTION=0.45`` -- and
    did NOT under jax's default (preallocate 75%) on an idle GPU, where omega came out up to 0.99
    away from the cache. The parity gate inherited that default because its shell set no XLA vars,
    which is the entire reason it failed while the identical command with a pinned policy passed.

    So the policy is a CORRECTNESS setting, not a courtesy to co-tenants, and it lives in one
    function that both the sidecar and the parity harness call -- a call site cannot pick a
    different one, and a shell that forgets to export anything still gets the verified policy.
    Returns what it set, for logging.
    """
    if not os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    }


def assert_tap_reproduces_cache(
    producer,
    *,
    source_features_root,
    frames_dir,
    lerobot_root,
    prompt_manifest,
    task: str,
    demo: int,
    grid_frame: int = 0,
) -> float:
    """Refuse to serve unless the tap reproduces the CACHED patch tokens for a known frame.

    The 2026-08-08 gate failures were environments that silently moved the tap onto a different XLA
    kernel. Nothing downstream could see it: omega stayed finite, the window kept its shape, the
    rollouts would have completed and scored. The only thing that separates a good sidecar from that
    one is a comparison against bytes already on disk, so it is made at STARTUP against the very
    cache the conditioner was trained on.

    IT MUST REPLAY THE CACHE'S OWN INPUTS. An earlier version fed a zero state and a placeholder
    prompt on the theory that a kernel regime shift would dominate; it does not. pi0.5 packs the
    robot state into the prompt, so state/prompt differences move the patch tokens FURTHER than a
    kernel change does (measured: zero state alone = 61.4 absolute, vs 13-22 for the small-batch
    kernel). The reference is therefore the real state, the canonical terse prompt, and the demo's
    own native frames -- exactly what `stage_s_cache_features` tapped -- and the bar is essentially
    exact, because with the tap batch padded to the cache's kernel the measured difference is
    0.000000.
    """
    # Deferred, and the direction is deliberate: the parity harness already owns the loaders that
    # define "the cache's own inputs", and having two copies of that definition is how a self-test
    # ends up certifying something the parity gate does not.
    from vla_training.eval.omega_sidecar_parity import (
        demo_frames,
        demo_states,
        load_canonical_prompt,
    )

    prompt = load_canonical_prompt(pathlib.Path(prompt_manifest), task)
    frames = demo_frames(pathlib.Path(frames_dir), task, int(demo))
    states = demo_states(pathlib.Path(lerobot_root), task, int(demo), frames["frame_indices"])
    cached = np.asarray(
        np.load(
            pathlib.Path(source_features_root).expanduser() / task / f"demo_{int(demo):06d}" / "patch_tokens.npy",
            mmap_mode="r",
        )[grid_frame],
        dtype=np.float32,
    )
    request = {wire: np.ascontiguousarray(frames[wire][grid_frame]) for wire in WIRE_TO_TAP_VIEW}
    request["observation/state"] = np.asarray(states[grid_frame], dtype=np.float32)
    # The cache tapped NATIVE frames; the serve wire may be a different size, and resize_with_pad is
    # idempotent, so the check pins the cache's geometry rather than the wire's.
    saved, producer.tap_image_size = producer.tap_image_size, 0
    try:
        observed = np.asarray(producer.tap_row(request, prompt), dtype=np.float32)
    finally:
        producer.tap_image_size = saved
    delta = float(np.abs(observed - cached).max())
    scale = float(np.abs(cached).max()) or 1.0
    if delta > _SELFTEST_MAX_REL * scale:
        raise SystemExit(
            f"[omega-sidecar] STARTUP SELF-TEST FAILED: the frozen tap does not reproduce the "
            f"cached patch tokens for {task}/demo_{demo:06d} frame {grid_frame} "
            f"(max|d| {delta:.5f} vs |cache|max {scale:.4f}, tap_batch={producer.tap_batch_size}). "
            f"This is the 2026-08-08 failure mode: an XLA memory/autotune regime, or a tap batch "
            f"size, that moves the tap onto a kernel other than the one the B=32 cache build used, "
            f"producing omega that stays finite and plausible while differing from the omega the "
            f"conditioner was trained on. Refusing to serve."
        )
    print(
        f"[omega-sidecar] OK startup self-test: the tap reproduces the cached patch tokens for "
        f"{task}/demo_{demo:06d} frame {grid_frame} (max|d| {delta:.6f}, |cache|max {scale:.4f}, "
        f"tap_batch={producer.tap_batch_size})",
        flush=True,
    )
    return delta


# --------------------------------------------------------------------------------------------------
# episode state machine (backbone-agnostic; the producer is injected)
# --------------------------------------------------------------------------------------------------
@dataclass
class _OmegaEpisodeState:
    """All mutable causal state owned by exactly one rollout env slot."""

    task: str
    demo_episode: int | str
    conditioner: Any
    last_t: int = -1
    last_grid: int = -1
    frames_seen: int = 0
    current: np.ndarray | None = None


class OmegaSidecar:
    """Per-``wsm_env_id`` causal omega state with the pi serve's exact discipline.

    Every rule below is a port of ``serve_pi_05_wsm.WSMPiInferWrapper._validate_batch`` /
    ``_prepare_batch``, not a re-derivation, because the failure it guards against is silent: an
    episode whose window carries a previous episode's frames scores plausibly and wrongly, and
    nothing downstream can tell. The pi incident test (``tests/test_pi_wsm_stateful_batching.py``)
    is mirrored one-for-one in ``tests/test_omega_sidecar.py`` against this class.

      * ``wsm_t == 0`` resets ONLY that env slot (a fresh conditioner buffer + this task's language).
      * a non-zero ``wsm_t`` from an env that never sent ``t == 0`` is refused.
      * ``wsm_t`` must strictly increase within an env slot.
      * ``(wsm_task, wsm_demo_episode)`` may not change without a ``t == 0`` reset, and two live env
        slots may not claim the same episode identity.
      * the causal grid may not skip (``floor(t/stride)`` advances by at most one) and a new grid
        must be observed exactly at ``t == grid * stride`` (i.e. ``replan_steps == stride``).
      * env slots and per-episode grid frames are BOUNDED; overflow fails instead of evicting live
        state, because eviction would silently restart a running episode's history.
      * a non-finite omega stops the process. The GR00T Eval2 incident was a NaN encoder that still
        produced scoreable rollouts; a conditioning path that is not finite must never return.
    """

    def __init__(
        self,
        producer,
        *,
        stride: int,
        max_envs: int = 1,
        max_grid_frames: int = 1024,
    ):
        self._producer = producer
        self._stride = int(stride)
        if self._stride < 1:
            raise ValueError(f"stride must be >= 1, got {self._stride}")
        self._max_envs = int(max_envs)
        self._max_grid_frames = int(max_grid_frames)
        if self._max_envs < 1 or self._max_grid_frames < 1:
            raise ValueError(
                f"max_envs and max_grid_frames must be >= 1; got {self._max_envs}, {self._max_grid_frames}"
            )
        self._states: dict[str, _OmegaEpisodeState] = {}
        self._lock = threading.RLock()
        self._episodes = 0
        self._calls = 0

    # -- request parsing ---------------------------------------------------------------------
    @staticmethod
    def _scalar(request: dict, key: str):
        if key not in request:
            raise RuntimeError(f"[omega-sidecar] missing required field {key!r}")
        value = np.asarray(request[key])
        if value.ndim != 0:
            raise RuntimeError(f"[omega-sidecar] field {key!r} must be scalar; got shape {value.shape}")
        return value.item()

    def _identity(self, request: dict) -> tuple[str, str, int | str, int, str]:
        raw_env_id = self._scalar(request, "wsm_env_id")
        raw_task = self._scalar(request, "wsm_task")
        raw_demo = self._scalar(request, "wsm_demo_episode")
        raw_t = self._scalar(request, "wsm_t")
        raw_prompt = self._scalar(request, "wsm_prompt")
        env_id, task = str(raw_env_id), str(raw_task)
        if not env_id or not task:
            raise RuntimeError("[omega-sidecar] wsm_env_id and wsm_task must be non-empty")
        if isinstance(raw_demo, (bool, np.bool_)) or not isinstance(raw_demo, (int, np.integer, str)):
            raise RuntimeError(f"[omega-sidecar] wsm_demo_episode must be an integer or string; got {raw_demo!r}")
        demo_episode = int(raw_demo) if isinstance(raw_demo, (int, np.integer)) else str(raw_demo)
        if isinstance(raw_t, (bool, np.bool_)) or not isinstance(raw_t, (int, np.integer)):
            raise RuntimeError(f"[omega-sidecar] wsm_t must be a non-negative integer; got {raw_t!r}")
        t = int(raw_t)
        if t < 0:
            raise RuntimeError(f"[omega-sidecar] wsm_t must be non-negative; got {t}")
        # The canonical private tap prompt. Required, and validated BEFORE any state is touched:
        # a blank prompt would silently shift the frozen tap's language conditioning away from the
        # one the encoder was trained under, and every omega after it would be wrong but finite.
        if not isinstance(raw_prompt, str) or not raw_prompt.strip() or raw_prompt != raw_prompt.strip():
            raise RuntimeError(
                "[omega-sidecar] wsm_prompt must be a non-empty, trimmed string (the canonical "
                "terse task instruction the omega cache was conditioned on). Launch the runner "
                "with --task-prompt-manifest."
            )
        if task not in self._producer.tasks:
            raise RuntimeError(
                f"[omega-sidecar] unknown wsm_task={task!r}; the task-language table has "
                f"{len(self._producer.tasks)} tasks"
            )
        return env_id, task, demo_episode, t, raw_prompt

    def _validate(self, request: dict) -> tuple[str, str, int | str, int, str]:
        env_id, task, demo_episode, t, prompt = self._identity(request)
        state = self._states.get(env_id)
        if t == 0:
            if state is None and len(self._states) + 1 > self._max_envs:
                raise RuntimeError(
                    f"[omega-sidecar] active env-state bound exceeded: have {len(self._states)}, "
                    f"max {self._max_envs}; refusing live-state eviction"
                )
            owner = next(
                (
                    other
                    for other, live in self._states.items()
                    if other != env_id and (live.task, live.demo_episode) == (task, demo_episode)
                ),
                None,
            )
            if owner is not None:
                raise RuntimeError(
                    f"[omega-sidecar] duplicate active episode identity "
                    f"({task!r}, {demo_episode!r}) on envs {owner!r} and {env_id!r}"
                )
            return env_id, task, demo_episode, t, prompt
        if state is None:
            raise RuntimeError(f"[omega-sidecar] env {env_id!r} sent t={t} before an explicit t=0 reset")
        if (task, demo_episode) != (state.task, state.demo_episode):
            raise RuntimeError(
                f"[omega-sidecar] episode identity changed without t=0 reset for env {env_id!r}: "
                f"active={(state.task, state.demo_episode)!r}, got={(task, demo_episode)!r}"
            )
        if t <= state.last_t:
            raise RuntimeError(f"[omega-sidecar] out-of-order wsm_t for env {env_id!r}: last={state.last_t}, got={t}")
        grid = math.floor(t / self._stride)
        if grid > state.last_grid + 1:
            raise RuntimeError(
                f"[omega-sidecar] skipped causal grid for env {env_id!r}: "
                f"last={state.last_grid}, got={grid} (t={t}, stride={self._stride})"
            )
        if grid != state.last_grid and t != grid * self._stride:
            raise RuntimeError(
                f"[omega-sidecar] misaligned causal grid for env {env_id!r}: new grid {grid} must "
                f"be observed at t={grid * self._stride}, got t={t}. Use replan_steps == stride."
            )
        return env_id, task, demo_episode, t, prompt

    # -- the one public entry point ----------------------------------------------------------
    def omega(self, request: dict) -> np.ndarray:
        """One causal omega window ``[K, w_dim]`` float32, oldest..newest, for this request."""
        with self._lock:
            env_id, task, demo_episode, t, prompt = self._validate(request)
            if t == 0:
                previous = self._states.get(env_id)
                conditioner = previous.conditioner if previous is not None else self._producer.new_conditioner()
                self._producer.reset(conditioner, task)
                self._states[env_id] = _OmegaEpisodeState(
                    task=task, demo_episode=demo_episode, conditioner=conditioner
                )
                self._episodes += 1
                # Logged at INFO, not debug: after the fact this line is the ONLY on-box proof that
                # omega was reset at each episode boundary rather than carried across one.
                logging.info(
                    "episode %d reset: env=%s task=%s demo=%s (%d windows served so far)",
                    self._episodes,
                    env_id,
                    task,
                    demo_episode,
                    self._calls,
                )
            state = self._states[env_id]
            grid = math.floor(t / self._stride)
            if grid != state.last_grid:
                if state.frames_seen >= self._max_grid_frames:
                    raise RuntimeError(
                        f"[omega-sidecar] causal-state frame bound exceeded for env {env_id!r}: "
                        f"max {self._max_grid_frames}"
                    )
                window = np.asarray(self._producer.step(state.conditioner, request, prompt), dtype=np.float32)
                if not np.isfinite(window).all():
                    raise RuntimeError(
                        f"[omega-sidecar] NON-FINITE omega at env={env_id!r} t={t} grid={grid} "
                        "(mismatched/diverged encoder checkpoint?) — refusing to serve"
                    )
                state.current = window
                state.last_grid = grid
                state.frames_seen += 1
            if state.current is None:
                raise RuntimeError(f"[omega-sidecar] no omega for env {env_id!r}; explicit t=0 reset was skipped")
            state.last_t = t
            self._calls += 1
            if self._calls % 50 == 0:
                logging.info(
                    "omega window %d served (episodes=%d, live envs=%d, grid frames=%d)",
                    self._calls,
                    self._episodes,
                    len(self._states),
                    state.frames_seen,
                )
            return state.current

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "state_mode": STATE_MODE,
                "stride": self._stride,
                "max_envs": self._max_envs,
                "max_grid_frames": self._max_grid_frames,
                "live_envs": sorted(self._states),
                "episodes": self._episodes,
                "calls": self._calls,
                "grid_frames": {k: v.frames_seen for k, v in self._states.items()},
            }


# --------------------------------------------------------------------------------------------------
# the real producer: frozen pi0.5 tap -> frozen WorkspaceModel -> causal window
# --------------------------------------------------------------------------------------------------
class PiOmegaProducer:
    """``Pi05BackboneTap`` -> ``WSMEvalConditioner``, i.e. the pi workspace serve's omega chain.

    Constructed exactly as ``serve_pi_05_wsm_cfg.main`` constructs it (:685 tap, :689-692 encoder +
    conditioner, :693 table), so the only thing this class adds is the image down-convert that turns
    a groot-native 256 px request back into the 224 px bytes a pi serve would have received.
    """

    def __init__(
        self,
        *,
        tap_ckpt: str,
        encoder_ckpt: str,
        task_lang_table: str,
        k_window: int,
        stride: int,
        device: str = "cuda",
        configs_dir: str | None = None,
        tap_image_size: int = PI_SERVE_IMAGE_SIZE,
        tap_batch_size: int = TAP_BATCH_SIZE,
        expect_encoder_sha256: str | None = None,
        tap=None,
    ):
        """``tap`` is injectable ONLY so the cache-parity harness can replay the offline pipeline's
        already-tapped features through this exact class (``omega_sidecar_parity.py --stage
        encoder``). Serving always leaves it None and builds the real frozen tap; anything else
        would be a producer the sealed pi arms never ran."""
        from vla_training.eval._groot_wsm_eval import WSMEvalConditioner, load_task_lang_table
        from workspace_models.features.generate_policy_features import load_wsm

        self.k = int(k_window)
        self.stride = int(stride)
        self.device = device
        self.tap_image_size = int(tap_image_size)
        self.tap_batch_size = int(tap_batch_size)
        if self.tap_batch_size < 1:
            raise ValueError(f"tap_batch_size must be >= 1, got {self.tap_batch_size}")
        self.encoder_sha256 = _sha256_file(encoder_ckpt)
        if expect_encoder_sha256 and self.encoder_sha256 != expect_encoder_sha256:
            # The omega cache manifest records the encoder its w was produced with. Serving a
            # different encoder reproduces the GR00T Eval2 failure exactly: structurally healthy,
            # numerically meaningless, and impossible to detect from the results.
            raise SystemExit(
                f"[omega-sidecar] --encoder-ckpt {encoder_ckpt} hashes to {self.encoder_sha256} "
                f"but the expected (cache-manifest) digest is {expect_encoder_sha256}. Refusing "
                f"to produce omega from an encoder the trained conditioner never saw."
            )
        if tap is None:
            from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

            logging.info("loading frozen feature-source tap (config=%s, ckpt=%s)", TAP_CONFIG_NAME, tap_ckpt)
            tap = Pi05BackboneTap(
                tap_ckpt,
                config_name=TAP_CONFIG_NAME,
                **({"configs_dir": configs_dir} if configs_dir else {}),
            )
        self._tap = tap
        logging.info("loading frozen WorkspaceModel encoder (sha256=%s)", self.encoder_sha256)
        self._encoder, self._encoder_meta = load_wsm(encoder_ckpt, device, proprio_dim=PI_PROPRIO_DIM)
        self._conditioner_cls = WSMEvalConditioner
        self._table = load_task_lang_table(task_lang_table)
        self.tasks = frozenset(self._table)
        self.w_dim = int(getattr(getattr(self._encoder, "cfg", None), "dim", 0) or self._encoder.encoder.cfg.dim)
        logging.info(
            "producer ready: K=%d stride=%d w_dim=%d tasks=%d tap_image_size=%s tap_batch=%d encoder_step=%s",
            self.k,
            self.stride,
            self.w_dim,
            len(self._table),
            self.tap_image_size or "native",
            self.tap_batch_size,
            self._encoder_meta.get("step", "?"),
        )

    def new_conditioner(self):
        return self._conditioner_cls(self._encoder, k_window=self.k, stride=self.stride, device=self.device)

    def reset(self, conditioner, task: str) -> None:
        conditioner.reset(self._table[task])

    def _frames(self, request: dict) -> dict:
        """The three camera frames, as the pi client would have sent them.

        ``resize_with_pad`` is openpi's own helper — the SAME function
        ``run_remembench_eval.run_episode.prep`` calls — and it short-circuits when the frame is
        already the requested size, so a parity run that feeds 256 px cache frames with
        ``--tap-image-size 0`` (or 256) touches no pixels at all.
        """
        from openpi_client import image_tools

        frames = {}
        for wire_key, view in WIRE_TO_TAP_VIEW.items():
            frame = np.asarray(request[wire_key])
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise RuntimeError(f"[omega-sidecar] {wire_key}: expected (H,W,3), got {frame.shape}")
            if frame.dtype != np.uint8:
                # serve_groot_ws.pack_observation already asserts uint8 on the same bytes; if it is
                # not uint8 here the frames did not come from the sealed runner and the
                # convert_to_uint8-vs-resize order that makes this bit-identical no longer holds.
                raise RuntimeError(
                    f"[omega-sidecar] {wire_key}: expected uint8 (the runner's wire dtype), got {frame.dtype}"
                )
            if self.tap_image_size:
                frame = image_tools.resize_with_pad(frame, self.tap_image_size, self.tap_image_size)
            frames[view] = np.ascontiguousarray(frame)[None]
        return frames

    def step(self, conditioner, request: dict, prompt: str) -> np.ndarray:
        """One new grid frame: tap -> encoder -> causal window. Mirrors ``_tap_batch``, padded.

        The single real frame is REPLICATED to ``tap_batch_size`` rows and only row 0 is kept. That
        is not an optimization and not an approximation: a transformer prefix has no cross-example
        interaction, so the padding rows cannot change row 0's value -- they only put the call on
        the same XLA kernel the B=32 cache build used. Skipping this makes omega differ from the
        trained cache by ~1.46 (see the module docstring's TAP BATCH SIZE note).
        """
        patch, proprio = self.tap_row(request, prompt, with_lang=True)
        window, _lang = conditioner.step(patch, proprio)
        window = window.detach().float().cpu().numpy() if hasattr(window, "detach") else window
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (self.k, self.w_dim):
            raise RuntimeError(f"[omega-sidecar] window shape {window.shape} != expected ({self.k}, {self.w_dim})")
        return window

    def tap_row(self, request: dict, prompt: str, *, with_lang: bool = False):
        """Tap ONE observation, padded to ``tap_batch_size`` rows, and return row 0.

        Shared by the serve path and the startup self-test on purpose: the self-test's whole value
        is that it exercises the same call the rollouts will make, padding included.
        """
        state = np.asarray(request["observation/state"], dtype=np.float32).reshape(1, -1)
        frames = self._frames(request)
        pad = self.tap_batch_size
        if pad > 1:
            frames = {view: np.repeat(value, pad, axis=0) for view, value in frames.items()}
            state = np.repeat(state, pad, axis=0)
        result = self._tap.tap(frames, state, [str(prompt)] * pad)
        patch = np.asarray(result.patch_tokens, dtype=np.float32)
        proprio = np.asarray(result.lang_emb, dtype=np.float32)
        if patch.shape[0] != pad or proprio.shape[0] != pad:
            raise RuntimeError(
                f"[omega-sidecar] tap batch mismatch: requested {pad}, got patch={patch.shape}, lang={proprio.shape}"
            )
        return (patch[0], proprio[0]) if with_lang else patch[0]

    @property
    def provenance(self) -> dict:
        return {
            "tap_config": TAP_CONFIG_NAME,
            "encoder_sha256": self.encoder_sha256,
            "encoder_step": self._encoder_meta.get("step"),
            "proprio_dim": PI_PROPRIO_DIM,
            "k_window": self.k,
            "stride": self.stride,
            "w_dim": self.w_dim,
            "tap_image_size": self.tap_image_size,
            "tap_batch_size": self.tap_batch_size,
            "tasks": sorted(self.tasks),
        }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(os.path.expanduser(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------------
# wire: localhost HTTP with npz bodies
# --------------------------------------------------------------------------------------------------
# WHY npz OVER HTTP AND NOT zmq/msgpack. Both venvs ship numpy but at DIFFERENT major versions
# (openpi 2.2.5, groot 1.26.4), so the wire has to be a format whose reader does not depend on the
# writer's numpy — npz is exactly that. stdlib http.server + urllib means zero new dependencies in
# either venv, a `curl`-inspectable health endpoint, and an error path that returns the producer's
# real traceback as the response body instead of an opaque socket drop.
def _pack(arrays: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _unpack(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_handler(sidecar: OmegaSidecar, provenance_fn: Callable[[], dict]):
    """One handler class bound to this process's sidecar (no globals, so tests can run many)."""

    class OmegaHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A003 - BaseHTTPRequestHandler API
            logging.debug("sidecar http: " + fmt, *args)

        def _respond(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != HEALTH_PATH:
                self._respond(404, b"unknown path", "text/plain")
                return
            payload = json.dumps(
                {**provenance_fn(), **sidecar.stats, "gpu_memory": gpu_memory_report()},
                sort_keys=True,
            ).encode()
            self._respond(200, payload, "application/json")

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != OMEGA_PATH:
                self._respond(404, b"unknown path", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = _unpack(self.rfile.read(length))
                missing = [key for key in REQUIRED_FIELDS if key not in request]
                if missing:
                    raise RuntimeError(f"[omega-sidecar] request is missing {missing}")
                window = sidecar.omega(request)
            except Exception:
                # The producer's real traceback, verbatim, as the body. The client re-raises it, so
                # a sidecar failure shows up in the GR00T server log with a stack that names the
                # actual cause instead of "connection reset".
                detail = traceback.format_exc()
                logging.error("omega failed:\n%s", detail)
                self._respond(500, detail.encode(), "text/plain")
                return
            self._respond(200, _pack({"w_window": window}), "application/octet-stream")

    return OmegaHandler


class OmegaSidecarClient:
    """Minimal client for the GR00T serve. stdlib only — nothing new enters the torch venv."""

    def __init__(self, url: str, *, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = float(timeout)

    def _post(self, path: str, payload: bytes) -> bytes:
        request = urllib.request.Request(
            self.url + path, data=payload, headers={"Content-Type": "application/octet-stream"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"[omega-sidecar-client] {self.url}{path} failed:\n{detail}") from None
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"[omega-sidecar-client] cannot reach the omega sidecar at {self.url}: {error}. "
                f"A deltanet serve CANNOT run without it (there is no cached omega for held-out "
                f"episodes and serving unconditioned would silently be the baseline)."
            ) from None

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.url + HEALTH_PATH, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"[omega-sidecar-client] cannot reach the omega sidecar at {self.url}: {error}"
            ) from None

    def window(self, request: dict) -> np.ndarray:
        """Send the SAME frames the runner sent, plus identity; get back ``[K, w_dim]`` float32."""
        payload = {}
        for key in REQUIRED_FIELDS:
            if key not in request:
                raise RuntimeError(
                    f"[omega-sidecar-client] the runner did not send {key!r}. A deltanet cell needs "
                    f"the wsm_* identity fields and --task-prompt-manifest; without them the "
                    f"episode boundary and the frozen tap's language are both undefined."
                )
            value = request[key]
            payload[key] = value if isinstance(value, np.ndarray) else np.asarray(value)
        window = _unpack(self._post(OMEGA_PATH, _pack(payload)))["w_window"]
        return np.asarray(window, dtype=np.float32)


# --------------------------------------------------------------------------------------------------
# memory reporting (the sidecar SHARES a GPU with the policy it feeds)
# --------------------------------------------------------------------------------------------------
def gpu_memory_report() -> dict:
    """Per-process GPU memory on the visible device, from nvidia-smi (no torch/jax import).

    Reported at startup and on /health because the whole placement question for this design is
    whether a jax producer and a torch GR00T serve co-exist on one 97 GB card. A number in the log
    is the only way that claim survives being repeated.
    """
    import subprocess

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
        totals = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except Exception as error:  # nvidia-smi absent (CPU test host) is not a serve failure
        return {"error": str(error), "cuda_visible_devices": visible}
    processes = {}
    for line in apps.splitlines():
        pid, used = (part.strip() for part in line.split(","))
        processes[int(pid)] = int(used)
    return {
        "cuda_visible_devices": visible,
        "self_pid": os.getpid(),
        "self_mib": processes.get(os.getpid()),
        "per_pid_mib": processes,
        "per_gpu_used_total_mib": [[int(part.strip()) for part in line.split(",")] for line in totals.splitlines()],
    }


# --------------------------------------------------------------------------------------------------
def serve(sidecar: OmegaSidecar, provenance_fn, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(sidecar, provenance_fn))
    logging.info("omega sidecar listening on http://%s:%d%s", host, port, OMEGA_PATH)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tap-ckpt",
        required=True,
        help="frozen pi0.5 feature-source ckpt; must be the omega cache manifest's "
        "frozen_pi_feature_source (box: /data/work/ckpts/tap_149999)",
    )
    parser.add_argument("--encoder-ckpt", required=True, help="FROZEN WorkspaceModel ckpt")
    parser.add_argument("--task-lang-table", required=True, help="task_lang_table.npz")
    parser.add_argument("--configs-dir", default=None, help="dir holding wsm_robocasa_configs.py for the tap")
    parser.add_argument(
        "--k-window",
        type=int,
        required=True,
        help="causal window length; MUST equal the served conditioner's trained "
        "pos_decay_bias window (the GR00T serve cross-checks it on /health)",
    )
    parser.add_argument("--stride", type=int, default=8, help="cache grid stride == the runner's --replan-steps")
    parser.add_argument(
        "--tap-image-size",
        type=int,
        default=PI_SERVE_IMAGE_SIZE,
        help=f"edge length fed to the frozen tap (default {PI_SERVE_IMAGE_SIZE} = "
        "the resolution every sealed pi workspace serve received). 0 = pass "
        "the request's frames through untouched (offline-cache parity only).",
    )
    parser.add_argument(
        "--tap-batch-size",
        type=int,
        default=TAP_BATCH_SIZE,
        help=f"rows the frozen tap is called with (default {TAP_BATCH_SIZE}). The "
        "single real frame is replicated and only row 0 is used; this exists "
        "because the tap is jitted per shape and B<8 selects a different XLA "
        "kernel than the B=32 the omega cache was built with, moving omega by "
        "~1.46 on |omega|~2.8. 1 = the sealed pi arms' serve behaviour "
        "(carries that offset); a labelled ablation, not the default.",
    )
    parser.add_argument(
        "--expect-encoder-sha256",
        default=None,
        help="refuse to start unless --encoder-ckpt hashes to this (the omega cache "
        "manifest's encoder_checkpoint.sha256)",
    )
    parser.add_argument(
        "--selftest-source-features",
        default=None,
        metavar="ROOT",
        help="cached source-features root. When given, the sidecar taps ONE cached "
        "frame at startup and refuses to serve unless it reproduces that "
        "demo's stored patch tokens. This is the guard against the 2026-08-08 "
        "failure: an XLA regime that silently moves the tap to a different "
        "kernel, yielding finite, plausible, WRONG omega.",
    )
    parser.add_argument(
        "--selftest-frames-dir", default=None, help="extract_frames output; required with --selftest-source-features"
    )
    parser.add_argument(
        "--selftest-lerobot-root",
        default=None,
        help="<root>/<task>/*/lerobot; required with --selftest-source-features",
    )
    parser.add_argument(
        "--selftest-prompt-manifest",
        default=None,
        help="canonical task-prompt manifest; required with --selftest-source-features",
    )
    parser.add_argument("--selftest-task", default="MemHeatPot")
    parser.add_argument("--selftest-demo", type=int, default=0)
    parser.add_argument(
        "--max-envs", type=int, default=1, help="live env slots; 1 matches WSM_ENVS_PER_GPU=1 on the pi serves"
    )
    parser.add_argument("--max-grid-frames", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--host", default="127.0.0.1", help="localhost by default: this is a co-located sidecar, not a service"
    )
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # BEFORE any jax import, and it is a CORRECTNESS setting as well as a co-residency one -- see
    # pin_jax_memory_env for the measurement that makes that true.
    logging.info("jax memory policy: %s", json.dumps(pin_jax_memory_env(), sort_keys=True))

    producer = PiOmegaProducer(
        tap_ckpt=args.tap_ckpt,
        encoder_ckpt=args.encoder_ckpt,
        task_lang_table=args.task_lang_table,
        k_window=args.k_window,
        stride=args.stride,
        device=args.device,
        configs_dir=args.configs_dir,
        tap_image_size=args.tap_image_size,
        tap_batch_size=args.tap_batch_size,
        expect_encoder_sha256=args.expect_encoder_sha256,
    )
    if args.selftest_source_features:
        missing = [
            name
            for name, value in (
                ("--selftest-frames-dir", args.selftest_frames_dir),
                ("--selftest-lerobot-root", args.selftest_lerobot_root),
                ("--selftest-prompt-manifest", args.selftest_prompt_manifest),
            )
            if not value
        ]
        if missing:
            raise SystemExit(
                f"--selftest-source-features needs {missing}: the check must replay the CACHE's own "
                f"inputs (real state + canonical prompt), not stand-ins."
            )
        assert_tap_reproduces_cache(
            producer,
            source_features_root=args.selftest_source_features,
            frames_dir=args.selftest_frames_dir,
            lerobot_root=args.selftest_lerobot_root,
            prompt_manifest=args.selftest_prompt_manifest,
            task=args.selftest_task,
            demo=args.selftest_demo,
        )
    else:
        logging.warning(
            "no --selftest-source-features: serving WITHOUT the tap/cache startup "
            "check that catches an XLA regime shift"
        )
    sidecar = OmegaSidecar(
        producer,
        stride=args.stride,
        max_envs=args.max_envs,
        max_grid_frames=args.max_grid_frames,
    )
    logging.info("GPU memory after load: %s", json.dumps(gpu_memory_report(), sort_keys=True))
    logging.info("omega sidecar OK: %s", json.dumps(producer.provenance, sort_keys=True))
    serve(sidecar, lambda: producer.provenance, args.host, args.port)


if __name__ == "__main__":
    main()
