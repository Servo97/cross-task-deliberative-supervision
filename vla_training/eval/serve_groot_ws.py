#!/usr/bin/env python3
"""openpi-WEBSOCKET policy server for GR00T N1.7 — the ReMemBench backbone-generality eval path.

WHY THIS EXISTS. The sealed ReMemBench harness (``scripts/remembench/run_remembench_eval.py``)
speaks ONE protocol: openpi's websocket + msgpack framing, a flat ``observation/*`` request, and a
flat 12-dim action chunk in reply. Every GR00T server in this repo speaks zmq with per-key nested
action dicts (``serve_groot_batched.py``, ``serve_groot_wsm*.py``). Rather than fork the runner —
which would put the two backbones on different episode/reset/seed code and quietly destroy the
comparison the arm exists to make — this is a thin ADAPTER: the same runner, the same manifest, the
same reset and seeding, with only the obs/action wire mapping translated.

THE MAPPING IS THE WHOLE FILE. It is the exact inverse of what the runner does, and each half is
pinned to the code that defines it:

  request -> GR00T observation
    observation/image        -> video.robot0_agentview_left      uint8   (1,1,H,W,3)
    observation/right_image  -> video.robot0_agentview_right      uint8   (1,1,H,W,3)
    observation/wrist_image  -> video.robot0_eye_in_hand          uint8   (1,1,H,W,3)
    observation/state (16,)  -> five state.* keys                 float32 (1,1,D)
        [0:3]  state.end_effector_position_relative
        [3:7]  state.end_effector_rotation_relative
        [7:10] state.base_position
        [10:14] state.base_rotation
        [14:16] state.gripper_qpos
      (the split is read off the runner's own np.concatenate in run_episode, which is the ground
       truth for what arrives on the wire; the same order the training export uses)
    prompt (str)             -> annotation.human.task_description [str]   (list, never a bare str)

  GR00T action dict -> reply
    actions[:, 0:3]   = action.end_effector_position
    actions[:, 3:6]   = action.end_effector_rotation
    actions[:, 6:7]   = action.gripper_close
    actions[:, 7:11]  = action.base_motion
    actions[:, 11:12] = action.control_mode
      (exact inverse of ``run_remembench_eval.convert_action``, which feeds the ReMemBench gym
       wrapper's PandaOmronKeyConverter.unmap_action)

TWO THINGS THAT ARE NOT MECHANICAL:

* ``policy_noise_seed``. The runner's reproducibility contract is that the 3 rollouts of one pinned
  reset differ ONLY in the diffusion noise, and that rollout 0 reproduces a 1-rollout run. On pi
  that is a JAX rng key. GR00T draws its initial flow-matching sample with a bare ``torch.randn``
  (gr00t_n1d7.py, get_action path) off the GLOBAL torch RNG, so the seed is honoured by reseeding
  torch immediately before each forward. Without this the arm's rollouts would be
  ambient-RNG-dependent and the paired per-episode analysis would be meaningless.
* ``wsm_*`` keys. The runner always sends ``wsm_t / wsm_task / wsm_env_id / wsm_demo_episode`` (and
  ``wsm_prompt`` when a prompt manifest is passed) so a workspace arm is drop-in comparable. A
  BASELINE groot serve ignores them — but it must ignore them EXPLICITLY, because silently passing
  an unknown key into ``Gr00tSimPolicyWrapper.check_observation`` fails its shape asserts. Any
  future conditioned groot arm keys episode state off ``wsm_env_id`` + ``wsm_t == 0``; the
  websocket wire has NO reset endpoint (``WebsocketClientPolicy.reset`` is a no-op), so ``wsm_t==0``
  is the ONLY episode boundary signal available. That seam is marked below.

The eval-frame pixel-consistency (site-hiding) fix is env-side in the ReMemBench checkout
(robocasa/wrappers/gym_wrapper.py hide_debug_visuals), so this adapter inherits it for free and
must not do any frame post-processing of its own.

  python vla_training/eval/serve_groot_ws.py --model-path <groot HF ckpt dir> --port 5900
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

import numpy as np

#: request key -> GR00T video key. The runner's three camera slots, in the RoboCasa order the
#: training export and robocasa_panda_modality.py both use.
CAMERA_MAP = {
    "observation/image": "video.robot0_agentview_left",
    "observation/right_image": "video.robot0_agentview_right",
    "observation/wrist_image": "video.robot0_eye_in_hand",
}
#: (GR00T state key, slice) over the runner's flat 16-dim observation/state.
STATE_SPLIT = (
    ("state.end_effector_position_relative", slice(0, 3)),
    ("state.end_effector_rotation_relative", slice(3, 7)),
    ("state.base_position", slice(7, 10)),
    ("state.base_rotation", slice(10, 14)),
    ("state.gripper_qpos", slice(14, 16)),
)
STATE_DIM = 16
#: action key -> destination slice in the flat 12-dim reply row.
ACTION_LAYOUT = (
    ("action.end_effector_position", slice(0, 3)),
    ("action.end_effector_rotation", slice(3, 6)),
    ("action.gripper_close", slice(6, 7)),
    ("action.base_motion", slice(7, 11)),
    ("action.control_mode", slice(11, 12)),
)
ACTION_DIM = 12
#: keys the runner sends that a baseline groot serve consumes or drops, never forwards.
RUNNER_CONTROL_KEYS = frozenset(
    {"prompt", "wsm_t", "wsm_task", "wsm_env_id", "wsm_demo_episode", "wsm_prompt", "policy_noise_seed"}
)


#: GR00T N1.7's own training resolution. The RUNNER's `--obs-image-size` defaults to 224 because
#: that is pi0.5's training resolution and every sealed pi arm must stay byte-identical — but a
#: groot launch that inherits that default sends 224 px frames into a processor whose
#: `image_target_size` is [256, 256]. The result is a SECOND resample in front of the one training
#: used. It does not crash, it does not warn, and it silently degrades every groot number.
#:
#: `run_remembench_box.sh` already sets 256 for `SERVE_KIND=groot`, but that is one script; anything
#: launched by hand, by a future harness, or by a copy-pasted command line inherits 224. So the
#: SERVER refuses the wrong size — a permanent guard on the receiving end, not a convention on the
#: sending end. It is cross-checked against the checkpoint's own `image_target_size` at startup, so
#: even this constant cannot disagree with the model being served.
GROOT_NATIVE_IMAGE_SIZE = 256


def force_transformers_local_files_only() -> None:
    """Make every transformers hub resolution in this process cache-only.

    The TRAIN path gets this through `config.training.transformers_local_files_only = True`
    (`_groot_common.build_and_run_groot`, driven by `WSM_TRANSFORMERS_LOCAL_FILES_ONLY`). The SERVE
    path has no equivalent: `Gr00tPolicy` calls `AutoModel.from_pretrained(model_dir)` with no
    loading kwargs, and GR00T then constructs its Qwen3VL backbone with its OWN
    `from_pretrained("nvidia/Cosmos-Reason2-2B")` (qwen3_backbone.py:80) — a GATED repo id, not a
    path. Even with a fully warm cache transformers still resolves that id through the hub for the
    etag, and a gated 401 is not treated as "offline", so it raises instead of falling back.

    `HF_HUB_OFFLINE=1` is NOT the fix (it turns a would-be-cached lookup into OfflineModeIsEnabled —
    recorded in the PHASE-1 gated-backbone writeup). `local_files_only=True` is. `cached_files` is
    the single chokepoint every config / weight / tokenizer / processor resolution passes through,
    so forcing the flag there covers all of them without guessing at call sites.

    Idempotent, and a no-op once applied.
    """
    from transformers.utils import hub as hub_module

    if getattr(hub_module, "_wsm_local_files_only", False):
        return

    def _wrap(name):
        original = getattr(hub_module, name, None)
        if original is None:
            return None

        def patched(*args, **kwargs):
            kwargs["local_files_only"] = True
            return original(*args, **kwargs)

        patched.__name__ = getattr(original, "__name__", name)
        return patched

    for name in ("cached_files", "cached_file"):
        patched = _wrap(name)
        if patched is not None:
            setattr(hub_module, name, patched)
            # transformers re-exports these; rebind the copies the callers actually hold.
            for module_name in (
                "transformers.configuration_utils",
                "transformers.modeling_utils",
                "transformers.tokenization_utils_base",
                "transformers.processing_utils",
                "transformers.image_processing_base",
                "transformers.feature_extraction_utils",
            ):
                module = sys.modules.get(module_name)
                if module is not None and hasattr(module, name):
                    setattr(module, name, patched)
    hub_module._wsm_local_files_only = True
    logging.info("transformers hub resolution forced to local_files_only=True (gated backbone repo, no token)")


def _omega_url_from_device(base_port: int | None) -> str | None:
    """http://127.0.0.1:(base_port + this process's physical GPU index), or None if unset.

    `CUDA_VISIBLE_DEVICES` is the only thing that distinguishes the N workers of a batched cell from
    each other, and the launcher sets it to exactly one physical index per worker. Refusing anything
    else is deliberate: a multi-device or empty value means the caller's one-sidecar-per-GPU
    assumption does not hold, and silently defaulting to port+0 would send every worker's frames to
    one producer that would then refuse them as a second live episode.
    """
    if base_port is None:
        return None
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not visible.isdigit():
        raise SystemExit(
            f"--omega-sidecar-base-port needs CUDA_VISIBLE_DEVICES to name exactly one physical "
            f"GPU (got {visible!r}); pass --omega-sidecar with an explicit URL instead."
        )
    return f"http://127.0.0.1:{int(base_port) + int(visible)}"


def pack_observation(request: dict, expected_image_size: int | None = None) -> dict:
    """Flat openpi request -> the flat batch-1 GR00T observation Gr00tSimPolicyWrapper wants.

    `expected_image_size` (edge length) is asserted on every camera when set. Pass None ONLY for a
    deliberate resolution ablation.
    """
    obs: dict = {}
    for wire_key, groot_key in CAMERA_MAP.items():
        if wire_key not in request:
            raise ValueError(f"request missing camera {wire_key!r}")
        frame = np.asarray(request[wire_key])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{wire_key}: expected (H,W,3), got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"{wire_key}: expected uint8, got {frame.dtype}")
        if expected_image_size is not None and (
            frame.shape[0] != expected_image_size or frame.shape[1] != expected_image_size
        ):
            raise ValueError(
                f"{wire_key}: got {frame.shape[0]}x{frame.shape[1]}, expected "
                f"{expected_image_size}x{expected_image_size}. The client is sending the WRONG "
                f"resolution — almost certainly the runner's `--obs-image-size` default of 224 "
                f"(pi0.5's training resolution) on a GR00T serve. Every frame would be resampled a "
                f"second time in front of the resample training used, silently degrading every "
                f"number this server produces. Launch the runner with `--obs-image-size "
                f"{expected_image_size}` (run_remembench_box.sh does this for SERVE_KIND=groot), or "
                f"pass `--expected-image-size 0` here for a deliberate resolution ablation."
            )
        # (H,W,3) -> (B=1, T=1, H, W, 3); check_observation hard-asserts 5 dims and T == 1.
        obs[groot_key] = np.ascontiguousarray(frame)[None, None, ...]

    if "observation/state" not in request:
        raise ValueError("request missing observation/state")
    state = np.asarray(request["observation/state"], dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"observation/state: expected {STATE_DIM} dims, got {state.shape[0]}")
    for groot_key, span in STATE_SPLIT:
        obs[groot_key] = np.ascontiguousarray(state[span], dtype=np.float32)[None, None, :]

    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("request missing a non-empty string `prompt`")
    # A LIST of length B, not a bare str — check_observation asserts list/tuple.
    obs["annotation.human.task_description"] = [prompt]

    unknown = set(request) - set(CAMERA_MAP) - {"observation/state"} - RUNNER_CONTROL_KEYS
    if unknown:
        raise ValueError(f"unrecognized request keys (refusing to forward blindly): {sorted(unknown)}")
    return obs


def normalize_policy_reply(reply):
    """`Gr00tSimPolicyWrapper.get_action` returns `(action_dict, extras)`, not a bare dict.

    Verified against the real wrapper + the target_ft_bal33 checkpoint on the box (2026-08-07):
    a 2-tuple whose first element is the per-key action dict, each value `(B=1, T=16, D)`. The
    STATIC contract check missed this because the keys, the layout and the chunk length were all
    exactly right — only the container differed, and `sorted(action_dict)` over a tuple of dicts
    fails with an unrelated `TypeError: '<' not supported between instances of 'dict' and 'dict'`
    that names nothing useful. Normalising here keeps `unpack_actions` a pure layout function.
    """
    if isinstance(reply, dict):
        return reply
    if isinstance(reply, (tuple, list)):
        if not reply or not isinstance(reply[0], dict):
            raise ValueError(
                f"policy reply is a {type(reply).__name__} of "
                f"{[type(r).__name__ for r in reply]}; expected the action dict first"
            )
        return reply[0]
    raise ValueError(f"unrecognized policy reply type {type(reply).__name__}")


def unpack_actions(action_dict: dict) -> np.ndarray:
    """GR00T per-key action dict -> the flat (T, 12) chunk the runner slices."""
    action_dict = normalize_policy_reply(action_dict)
    rows = None
    out = None
    for key, span in ACTION_LAYOUT:
        if key not in action_dict:
            raise ValueError(f"policy reply missing {key!r}; got {sorted(action_dict)}")
        value = np.asarray(action_dict[key], dtype=np.float32)
        if value.ndim == 3:  # (B=1, T, D) — drop the batch dim
            if value.shape[0] != 1:
                raise ValueError(f"{key}: expected batch 1, got {value.shape}")
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"{key}: expected (T,D) after squeeze, got {value.shape}")
        width = span.stop - span.start
        if value.shape[1] != width:
            raise ValueError(f"{key}: expected width {width}, got {value.shape[1]}")
        if rows is None:
            rows = value.shape[0]
            out = np.zeros((rows, ACTION_DIM), dtype=np.float32)
        elif value.shape[0] != rows:
            raise ValueError(f"{key}: horizon {value.shape[0]} != {rows}")
        out[:, span] = value
    if out is None:
        raise ValueError("empty action layout")
    if not np.all(np.isfinite(out)):
        raise RuntimeError("policy produced non-finite actions")
    return out


class GrootWebsocketPolicy:
    """openpi ``.infer(obs) -> dict`` surface over a GR00T policy."""

    def __init__(
        self,
        policy,
        *,
        honor_noise_seed: bool = True,
        model_path: str = "",
        expected_image_size: int | None = GROOT_NATIVE_IMAGE_SIZE,
        mechanism: str = "none",
        action_head=None,
        omega_client=None,
        omega_geometry: dict | None = None,
    ):
        self._policy = policy
        self._honor_noise_seed = honor_noise_seed
        self._model_path = model_path
        self._expected_image_size = expected_image_size
        self._calls = 0
        # Recurrent-mechanism episode state. `mechanism == "none"` leaves every code path below
        # untouched, so the sealed PHASE-1 baseline adapter stays byte-for-byte what §21 certified.
        self._mechanism = mechanism
        self._action_head = action_head
        self._episodes = 0
        self._ttt_env_id = None
        # DELTANET only: the co-located omega producer. This process CANNOT compute omega itself
        # (it is WSMEncoder(frozen pi0.5 tap features), a jax chain), and it deliberately holds NO
        # omega episode state — the sidecar owns all of it, keyed on wsm_env_id and reset at
        # wsm_t == 0, which is the same single implementation the sealed pi arms ran.
        self._omega_client = omega_client
        self._omega_geometry = omega_geometry or {}
        self._omega_windows = 0

    @property
    def metadata(self) -> dict:
        """First frame on connect — the client blocks until it arrives."""
        return {
            "server": "serve_groot_ws",
            "backbone": "groot_n17",
            "model_path": self._model_path,
            "action_layout": [k for k, _ in ACTION_LAYOUT],
            "honors_policy_noise_seed": self._honor_noise_seed,
            # On the metadata frame so the CLIENT can see what the server will accept, before it
            # spends an episode discovering it.
            "expected_image_size": self._expected_image_size,
            # So a client can SEE which mechanism the server re-attached before it spends an
            # episode. A baseline-serving run under a mechanism arm's name is the failure mode the
            # whole re-attach machinery exists to prevent; publishing it here makes it auditable
            # from the runner log as well as the server log.
            "mechanism": self._mechanism,
            # A deltanet arm is only meaningful when a producer is attached, so publish it for the
            # same reason: "conditioned" must be auditable from the RUNNER log too.
            "omega_sidecar": getattr(self._omega_client, "url", None),
        }

    def infer(self, request: dict) -> dict:
        import torch

        # Log the size actually received ONCE, so a passing run still leaves proof in the log of
        # what resolution it ran at (a silent 224 run is otherwise indistinguishable from a 256 one
        # after the fact).
        if self._calls == 0:
            first = np.asarray(request.get("observation/image", np.zeros((0, 0, 3))))
            logging.info(
                "first observation frame: %sx%s (expected %s)",
                first.shape[0] if first.ndim == 3 else "?",
                first.shape[1] if first.ndim == 3 else "?",
                self._expected_image_size,
            )
        observation = pack_observation(request, expected_image_size=self._expected_image_size)
        seed = request.get("policy_noise_seed")
        if self._honor_noise_seed:
            if seed is None:
                raise ValueError(
                    "policy_noise_seed absent: the runner's pinned-rollout contract requires it. "
                    "Pass --ignore-noise-seed only for a deliberately unpinned smoke test."
                )
            # GR00T's initial flow-matching sample is a bare torch.randn off the GLOBAL RNG, so
            # this reseed IS the per-step noise pin. Both generators: the forward runs on CUDA but
            # transforms may draw on CPU.
            value = int(np.asarray(seed).reshape(()).item()) & 0xFFFFFFFF
            torch.manual_seed(value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(value)
        # Episode state is keyed on request["wsm_env_id"] and rebuilt when request["wsm_t"] == 0 —
        # the websocket wire has no reset endpoint, so wsm_t==0 is the only boundary signal, and
        # `wsm_env_id` is STABLE for a worker's whole life (PHASE-1 §10; do not "fix" that).
        if self._mechanism == "ttt":
            self._ttt_episode_boundary(request)
        if self._mechanism == "deltanet":
            self._stash_omega_window(request)
        try:
            with torch.inference_mode():
                action_dict = self._policy.get_action(observation)
        finally:
            if self._mechanism == "deltanet":
                # Never leave a window behind. A stale stash would let a later request whose sidecar
                # call failed still condition on an OLD step's omega and return plausible actions;
                # the head's own "no w_window" hard-fail is what must happen instead.
                self._action_head._dn_eval_window = None
        if self._mechanism == "ttt":
            # APPLY-THEN-COMMIT at chunk granularity (07a D-1): the chunk above was produced by the
            # ENTERING W; only now does W advance. Exactly once per inference == once per executed
            # chunk, because the runner replans on a fixed stride and each replan is one request.
            # OUTSIDE the inference_mode block on purpose — the commit is a gradient step.
            from vla_training.eval._groot_robottt_eval import robottt_serve_commit

            robottt_serve_commit(self._action_head)
        self._calls += 1
        return {"actions": unpack_actions(action_dict)}

    def _stash_omega_window(self, request: dict) -> None:
        """Fetch this step's causal omega window from the sidecar and stash it for the head.

        The GR00T venv cannot produce omega (see `omega_sidecar`'s docstring), so the ONLY thing
        this process does is forward the frames it was already given and place the answer where
        `WSMDeltaNetActionHead.get_action_with_features` looks for it. Deliberately NO caching, NO
        per-episode buffer and NO reset handling here: duplicating the pi serve's episode-boundary
        rules in a second place is precisely how the two would drift.

        Every failure is fatal by design. An unreachable sidecar, a wrong-shaped window or a
        non-finite value must stop the rollout, because the alternative — a deltanet checkpoint
        served with no conditioning — is the "runs fine as the baseline under the arm's name"
        failure the whole attach/restore machinery exists to prevent.
        """
        import torch

        window = self._omega_client.window(request)
        expected = (int(self._omega_geometry["window_len"]), int(self._omega_geometry["w_dim"]))
        if window.shape != expected:
            raise RuntimeError(
                f"omega sidecar returned {window.shape}, but this checkpoint's conditioner was "
                f"trained on {expected} (window_len from pos_decay_bias, w_dim from proj_q). "
                f"Restart the sidecar with --k-window {expected[0]}."
            )
        if not np.all(np.isfinite(window)):
            raise RuntimeError(
                "omega sidecar returned a NON-FINITE window; refusing to condition temb with NaN "
                "(see the GR00T Eval2 NaN-encoder incident)."
            )
        reference = next(self._action_head.wsm_deltanet.parameters())
        self._action_head._dn_eval_window = torch.from_numpy(np.ascontiguousarray(window))[None].to(
            device=reference.device, dtype=reference.dtype
        )
        self._omega_windows += 1
        if self._omega_windows == 1:
            logging.info(
                "first omega window from the sidecar: shape %s |w| mean %.5g",
                tuple(window.shape),
                float(np.abs(window).mean()),
            )

    def _ttt_episode_boundary(self, request: dict) -> None:
        """Reset the fast weights at wsm_t == 0. Fail loud if the boundary signal is absent."""
        from vla_training.eval._groot_robottt_eval import robottt_serve_reset

        if "wsm_t" not in request:
            raise ValueError(
                "a RoboTTT serve requires 'wsm_t': it is the ONLY episode-boundary signal on this "
                "wire (WebsocketClientPolicy.reset is a no-op), and without it the fast weights "
                "would carry state across episodes — a chain that crosses an episode boundary is "
                "the one failure this mechanism cannot tolerate."
            )
        step = int(np.asarray(request["wsm_t"]).reshape(()).item())
        env_id = request.get("wsm_env_id")
        env_id = None if env_id is None else str(np.asarray(env_id).reshape(()).item())
        if step < 0:
            raise ValueError(f"wsm_t must be non-negative, got {step}")
        if step == 0 or self._episodes == 0 or env_id != self._ttt_env_id:
            robottt_serve_reset(self._action_head, batch=1)
            self._episodes += 1
            self._ttt_env_id = env_id
            logging.info(
                "robottt fast weights reset to W_0 (episode %d, env_id=%s, wsm_t=%d)", self._episodes, env_id, step
            )


def serve(policy, host: str, port: int) -> None:
    """Minimal openpi-compatible websocket server.

    Deliberately self-contained: it imports only ``openpi_client`` (pure python, already required
    on the eval box) rather than the openpi SERVER package, which is a jax install that has no
    business inside the GR00T venv. The framing is openpi's own msgpack_numpy — imported from the
    client so server and client can never drift.
    """
    import websockets.sync.server
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()

    def handler(connection) -> None:
        peer = getattr(connection, "remote_address", "?")
        logging.info("client connected: %s", peer)
        connection.send(packer.pack(policy.metadata))
        while True:
            try:
                message = connection.recv()
            except Exception:
                logging.info("client disconnected: %s", peer)
                return
            try:
                request = msgpack_numpy.unpackb(message)
                connection.send(packer.pack(policy.infer(request)))
            except Exception:
                detail = traceback.format_exc()
                logging.error("inference failed:\n%s", detail)
                # A str reply makes the client raise RuntimeError with this text (see
                # WebsocketClientPolicy.infer), which surfaces the real traceback in the runner log
                # instead of an opaque connection drop.
                connection.send(detail)
                raise

    with websockets.sync.server.serve(handler, host, port, compression=None, max_size=None) as server:
        logging.info("serve_groot_ws listening on ws://%s:%d", host, port)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="GR00T HF checkpoint dir")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5900)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--no-sim-wrapper", action="store_true")
    parser.add_argument(
        "--ignore-noise-seed",
        action="store_true",
        help="do NOT reseed torch per request; unpinned rollouts (smoke tests only)",
    )
    parser.add_argument(
        "--expected-image-size",
        type=int,
        default=GROOT_NATIVE_IMAGE_SIZE,
        help=(
            "edge length the server REQUIRES on every incoming camera frame (default "
            f"{GROOT_NATIVE_IMAGE_SIZE} = GR00T N1.7's training resolution). The runner's own "
            "--obs-image-size defaults to 224 for pi0.5 compatibility; sending that here would "
            "double-resample every frame and silently degrade the numbers. 0 disables the check "
            "(deliberate resolution ablation only). Cross-checked against the checkpoint's "
            "image_target_size at startup."
        ),
    )
    parser.add_argument(
        "--hf-offline-shim",
        default=None,
        metavar="PATH",
        help=(
            "python module providing install(), run BEFORE the policy is constructed, to resolve "
            "the GATED nvidia/Cosmos-Reason2-2B backbone from a local HF cache. Without it (and "
            "without HF_TOKEN) Gr00tN1d7DataCollator dies building the processor. On the box: "
            "/data/work/groot_smoke/hf_offline_shim.py"
        ),
    )
    # ADDITIVE. Unset ("none") reproduces the sealed PHASE-1 baseline adapter byte-for-byte; every
    # existing pi and groot baseline serve is untouched.
    parser.add_argument(
        "--mechanism",
        choices=["none", "deltanet", "ttt"],
        default="none",
        help=(
            "re-attach a train-time mechanism after the checkpoint loads. REQUIRED for a "
            "mechanism arm: from_pretrained bypasses the train-path patch, so without this the "
            "conditioner is absent, its trained tensors are dropped, and the server silently "
            "serves the BASELINE policy under the arm's name. NOTE the JEPA arms are NOT here: "
            "their aux head is a TRAIN-time target that `sample_actions` never touches, so they "
            "deploy as `none` — exactly as the pi s3 arms do (run_remembench_box.sh's comment on "
            "SERVE_KIND=base). Use --expect-subtree to prove the checkpoint is the right one."
        ),
    )
    parser.add_argument(
        "--expect-subtree",
        default=None,
        metavar="PREFIX",
        help=(
            "assert the checkpoint contains at least one tensor under PREFIX, then continue. "
            "For an arm that serves as BASE but was trained with a mechanism — the JEPA cells, "
            "whose aux head is dropped at serve — this is the only thing standing between "
            "'served the jepa checkpoint' and 'served the base checkpoint under the jepa arm's "
            "name'; nothing else in the serve path would notice. e.g. "
            "--expect-subtree action_head.jepa_predictor."
        ),
    )
    parser.add_argument(
        "--omega-sidecar",
        default=None,
        metavar="URL",
        help=(
            "REQUIRED for --mechanism deltanet: the co-located online-omega producer "
            "(vla_training/eval/omega_sidecar.py), e.g. http://127.0.0.1:6000. The conditioner "
            "reads a causal window of omega = WSMEncoder(frozen pi0.5 tap features) — a jax "
            "chain that cannot run in this torch-only venv — and there is NO cached omega for "
            "held-out eval episodes. The sidecar owns all per-episode omega state; this process "
            "only forwards frames and stashes the answer."
        ),
    )
    parser.add_argument(
        "--omega-sidecar-base-port",
        type=int,
        default=None,
        metavar="PORT",
        help=(
            "alternative to --omega-sidecar for multi-GPU batches: the sidecar URL is derived as "
            "http://127.0.0.1:(PORT + CUDA_VISIBLE_DEVICES). A batch launches N workers with ONE "
            "shared serve-extra string but a different device each, so a fixed URL would point "
            "every worker at one GPU's producer — which the sidecar would then refuse as a "
            "second live env (it never evicts live episode state). Deriving from the device is "
            "what makes one-sidecar-per-GPU expressible in a single command line."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    # Registers the RoboCasa NEW_EMBODIMENT modality config; MUST happen before the policy is
    # constructed, exactly as on the training path (_groot_common.load_modality_config).
    from vla_training.train.train_base._groot_common import load_modality_config

    load_modality_config()

    # GATED BACKBONE. GR00T builds its processor with
    # Qwen3VLProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B") — a gated repo — so without
    # either HF_TOKEN or a locally-resolved cache the policy dies inside Gr00tN1d7DataCollator
    # before a single frame is served. The shim resolves it offline (local_files_only) exactly the
    # way the TRAIN entry does with its cache asset, so serve and train agree on the mechanism.
    if args.hf_offline_shim:
        import importlib.util

        spec = importlib.util.spec_from_file_location("hf_offline_shim", args.hf_offline_shim)
        if spec is None or spec.loader is None:
            raise SystemExit(f"--hf-offline-shim: cannot load {args.hf_offline_shim}")
        shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shim)
        logging.info("HF offline shim installed from %s: %s", args.hf_offline_shim, shim.install())
        # The shim only neutralises the tokenizer's unconditional `model_info` round-trip. The
        # CONFIG/WEIGHT resolution for the gated repo id is a separate hub call and needs this.
        force_transformers_local_files_only()
        cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        repo_dir = os.path.join(cache, "hub", "models--nvidia--Cosmos-Reason2-2B")
        if not os.path.isdir(repo_dir):
            # local_files_only turns a cold cache into an opaque "not found"; say so here instead.
            raise SystemExit(
                f"--hf-offline-shim needs a WARM HF cache and none is at {repo_dir}. Point HF_HOME "
                f"at the cache that holds nvidia/Cosmos-Reason2-2B (box: HF_HOME=/data/work/hf) or "
                f"export HF_TOKEN."
            )
        logging.info("gated backbone resolved offline from %s", repo_dir)
    elif not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        # Fail here with the fix, rather than 40 lines deep in transformers with a stack trace that
        # names neither the gated repo nor the shim.
        raise SystemExit(
            "the GR00T backbone repo nvidia/Cosmos-Reason2-2B is GATED and no credential or local "
            "resolution is configured. Either export HF_TOKEN, or pass --hf-offline-shim "
            "<path/to/hf_offline_shim.py> (box: /data/work/groot_smoke/hf_offline_shim.py). "
            "Without one of these the policy dies constructing Gr00tN1d7DataCollator."
        )

    logging.info("loading GR00T checkpoint: %s", args.model_path)
    inner = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
        strict=not args.no_strict,
    )
    if args.expect_subtree:
        # Cheap, and it runs BEFORE a single episode is spent. A checkpoint whose mechanism subtree
        # is missing is either the wrong checkpoint or a train that silently ran as the baseline.
        # The shard INDEX already names every tensor, so this costs a 100 KB json read rather than
        # a 12 GB materialisation — which matters when four servers start at once.
        index_path = os.path.join(args.model_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as handle:
                names = json.load(handle)["weight_map"].keys()
        else:
            from vla_training.eval._groot_wsm_deltanet_eval import load_checkpoint_state_dict

            names = load_checkpoint_state_dict(args.model_path).keys()
        keys = [k for k in names if k.startswith(args.expect_subtree)]
        if not keys:
            raise SystemExit(
                f"--expect-subtree {args.expect_subtree!r}: NO tensor with that prefix in "
                f"{args.model_path}. Either this is not the checkpoint you meant to serve, or the "
                f"train ran as the baseline under a mechanism arm's name."
            )
        logging.info("checkpoint carries %d tensors under %r (e.g. %s)", len(keys), args.expect_subtree, keys[0])

    action_head = None
    omega_client = None
    omega_geometry = None
    if args.mechanism == "deltanet":
        # ATTACH first, THEN restore — a strict load before the attach has nothing to load into.
        # The geometry (K, heads, head_dim, dims) is recovered from the checkpoint tensors
        # themselves, so no flag here can disagree with the trained recipe.
        from vla_training.eval._groot_wsm_deltanet_eval import attach_and_restore_deltanet

        geometry = attach_and_restore_deltanet(inner.model, args.model_path)
        action_head = inner.model.action_head
        logging.info("deltanet conditioner re-attached and restored: %s", geometry)
        # THE SEAM, NOW WIRED. The head raises unless `action_head._dn_eval_window` holds the causal
        # omega window [1, K, w_dim] for the current step, and this process cannot compute one: the
        # study's shared cache defines omega as `WSMEncoder(frozen pi0.5 backbone features)` (omega
        # manifest `artifact: pi05_workspace_omega`, `frozen_pi_feature_source`, `backbone_dim
        # 2048`), which is a jax chain, and the disk cache covers TRAIN demos only — ReMemBench
        # eval episodes are HELD OUT, so there is nothing to look up and a train demo's omega would
        # be an oracle leak. `--omega-sidecar` points at the co-located producer that runs the
        # SAME chain the pi workspace arms run online (`serve_pi_05_wsm_cfg.py`: Pi05BackboneTap ->
        # WorkspaceModel -> causal K-window). It owns every piece of per-episode omega state,
        # keyed on wsm_env_id and reset at wsm_t == 0; this process forwards frames and stashes.
        omega_url = args.omega_sidecar or _omega_url_from_device(args.omega_sidecar_base_port)
        if not omega_url:
            raise SystemExit(
                "--mechanism deltanet requires --omega-sidecar URL (or --omega-sidecar-base-port). "
                "Without a producer the head hard-fails on the first inference (by design) — a "
                "mechanism eval must not be runnable unconditioned by accident. Start "
                "vla_training/eval/omega_sidecar.py on this GPU first."
            )
        from vla_training.eval.omega_sidecar import OmegaSidecarClient

        omega_client = OmegaSidecarClient(omega_url)
        health = omega_client.health()
        omega_geometry = {"window_len": geometry["window_len"], "w_dim": geometry["w_dim"]}
        # CROSS-CHECK BEFORE AN EPISODE IS SPENT. The trained window length is structural
        # (pos_decay_bias) and the sidecar's K is a flag; a disagreement is a silently mis-shaped
        # conditioning input, so it must fail at startup rather than mid-rollout.
        if int(health.get("k_window", -1)) != geometry["window_len"]:
            raise SystemExit(
                f"omega sidecar serves K={health.get('k_window')} but this checkpoint's "
                f"pos_decay_bias says the conditioner was trained on K={geometry['window_len']}. "
                f"Restart the sidecar with --k-window {geometry['window_len']}."
            )
        if int(health.get("w_dim", -1)) != geometry["w_dim"]:
            raise SystemExit(
                f"omega sidecar serves w_dim={health.get('w_dim')}, checkpoint expects "
                f"{geometry['w_dim']}: the encoder does not match the trained conditioner."
            )
        if health.get("state_mode") != "per_env_isolated_v1":
            raise SystemExit(
                f"omega sidecar reports state_mode={health.get('state_mode')!r}; this serve is "
                f"written against 'per_env_isolated_v1' episode-boundary rules."
            )
        logging.info(
            "omega sidecar OK at %s: %s",
            omega_url,
            json.dumps(
                {
                    k: health.get(k)
                    for k in ("k_window", "stride", "w_dim", "tap_image_size", "encoder_sha256", "state_mode")
                },
                sort_keys=True,
            ),
        )

    elif args.mechanism == "ttt":
        # Same attach-then-restore contract as the deltanet arm. RoboTTT reads NO omega — its
        # recurrent state comes from the policy's own transitions (§24) — so unlike the deltanet
        # arm it is fully servable from this process.
        from vla_training.eval._groot_robottt_eval import attach_and_restore_robottt

        geometry = attach_and_restore_robottt(inner, args.model_path)
        action_head = inner.model.action_head
        logging.info("robottt fast weights re-attached and restored: %s", geometry)

    # RESOLUTION CROSS-CHECK. The checkpoint's own processor states the size it was trained at, so
    # the guard is derived from the model rather than trusted from a flag: a serve pointed at a
    # 224-trained checkpoint with the 256 default now fails at STARTUP instead of producing a full
    # set of quietly-degraded episodes.
    expected_image_size = args.expected_image_size or None
    target = getattr(getattr(inner, "processor", None), "image_target_size", None)
    if target:
        ckpt_size = int(target[0])
        if any(int(v) != ckpt_size for v in target):
            raise SystemExit(f"checkpoint image_target_size is non-square: {target}")
        if expected_image_size is None:
            logging.warning(
                "image-size check DISABLED (--expected-image-size 0); the checkpoint was trained at %s", ckpt_size
            )
        elif expected_image_size != ckpt_size:
            raise SystemExit(
                f"--expected-image-size {expected_image_size} disagrees with this checkpoint's "
                f"image_target_size {target}. Serving at the wrong resolution double-resamples "
                f"every frame. Pass --expected-image-size {ckpt_size} (and launch the runner with "
                f"--obs-image-size {ckpt_size})."
            )
        else:
            logging.info(
                "image size %s confirmed against the checkpoint's image_target_size %s", expected_image_size, target
            )
    else:
        logging.warning(
            "checkpoint exposes no image_target_size; the %s guard is unverified against the model",
            expected_image_size,
        )

    policy = inner if args.no_sim_wrapper else Gr00tSimPolicyWrapper(inner)
    serve(
        GrootWebsocketPolicy(
            policy,
            honor_noise_seed=not args.ignore_noise_seed,
            model_path=args.model_path,
            expected_image_size=expected_image_size,
            mechanism=args.mechanism,
            action_head=action_head,
            omega_client=omega_client,
            omega_geometry=omega_geometry,
        ),
        args.host,
        args.port,
    )


if __name__ == "__main__":
    main()
