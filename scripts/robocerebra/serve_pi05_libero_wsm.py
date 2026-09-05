#!/usr/bin/env python3
"""ω-CONDITIONED pi0.5 policy server for the RoboCerebra arms A1/A2/A4.

``serve_pi05_libero.py`` serves a pi05_libero-family checkpoint with no ω: ``Observation.wsm_w_window``
stays ``None``, and a ``wsm_tanh``/``gated_deltanet`` model raises (pi0.py:631) rather than silently
deploying as base. This file is the missing eval-side half of the RoboCerebra ω train path — the
2-view/128-token analog of ``vla_training/eval/serve_pi_05_wsm.py`` (RoboCasa, 3 views, 192 tokens).

Per decision step it:

  1. builds the trained arm's policy from ``--config`` + ``--checkpoint`` (assets-symlink trick,
     inherited verbatim from ``serve_pi05_libero.py``);
  2. taps a pi05 backbone on the request's 2 views + prompt, exactly as ``omega_tap.py`` did when the
     ω feature store was built (``embed_prefix`` -> ``PaliGemma.llm`` -> ``pool_grid`` over image
     slots 0 and 1 -> ``[128, 2048]``);
  3. feeds the PINNED frozen ω encoder ``(tokens, pooled_img, pooled_lang)`` over this episode's
     causal prefix and takes the newest ω_t;
  4. pushes ω_t into a per-env ``ServeWindow`` and injects the resulting ``[K, 512]`` window as
     ``wsm_w_window``, after stripping every ``wsm_*`` signal key the client used to say *who* and
     *when* it is.

--------------------------------------------------------------------------------------------------
THE SLOT CONVENTION (differs from RoboCasa — read before touching)
--------------------------------------------------------------------------------------------------
``WorkspaceEncoder.forward(patches, proprio, cond_lang)``. For RoboCerebra:

    patches   <- tokens       [128, 2048]   base grid then wrist grid, 8x8 bin-averaged
    proprio   <- pooled_img   [2048]        masked mean over the IMAGE token span
    cond_lang <- pooled_lang  [2048]        masked mean over the LANGUAGE token span

``pooled_img`` occupies the proprio slot (RoboCasa's pi server puts ``lang_emb`` there). That is not
a transcription slip: ``pi05_libero`` has ``discrete_state_input=False``, so the prompt tokens carry
no discretised robot state and the language span is a pure task embedding — using it twice would
leave the proprio slot content-free. ``omega_tap.py`` / ``precompute_omega.py`` built the whole
feature store this way; the encoder was trained on it and must be served it.

The image span is masked as ``3 * 196`` positions even though only 2 views are real: pi0.5 always
lays out 3 image slots and RoboCerebra zero-masks the third, so the language span still starts at
``3 * N_IMG_TOK``. Reproduced from ``omega_tap.embed_batch`` (``pool_grid`` is *imported* from it, so
the 14x14 -> 8x8 binning has exactly one definition).

--------------------------------------------------------------------------------------------------
THE STRIDE (the subtle part)
--------------------------------------------------------------------------------------------------
Training ω was tapped at 64 uniformly-spaced frames per episode
(``np.unique(np.linspace(0, length - 1, 64).astype(int))``), so the ω grid stride is
``episode_length / 64`` and VARIES per episode — measured over all 994 stored episodes: mean 14.5,
min 1.4, max 35.2 frames. A K=8 window therefore spans the last ~12.5% of the episode, NOT a fixed
number of env steps, and a K=16 window the last ~25%.

At eval the episode length is known at episode start: ``max_steps = switch_steps(150) * num_subtasks``.
So the serve-side stride is

    stride = max(1, round(episode_len / 64))

taken from the client's ``wsm_episode_len``. That, and only that, reproduces the training-time
"window = last 12.5% of the episode" semantics. A fixed stride (say 8, RoboCasa's) would make the
window span 6.4% of a 6-subtask episode and 4.3% of a 9-subtask one — two different mechanisms
sharing one checkpoint. See ``--frames-per-episode`` to change the 64 (don't, unless the store
changes).

Residual, stated because it is real and small: requests arrive every ``--replan`` (5) env steps, and
``ServeWindow.due`` fires on ``step - last_encoded >= stride``, so the realised cadence is
``ceil(stride / replan) * replan`` (15 for stride 14) rather than the exact 14.06. Over a 900-step
episode that is 60 encodes instead of 64, i.e. grid position i sits at env step ~15i instead of
~14.06i. ``ServeWindow`` is the canonical shared buffer and is deliberately NOT reimplemented here to
close a ~6% cadence gap.

--------------------------------------------------------------------------------------------------
WHICH BACKBONE THE TAP RUNS (--tap-source; default deviates from the build brief, on purpose)
--------------------------------------------------------------------------------------------------
The ω store was built by tapping the RELEASED ``pi05_libero`` checkpoint
(``run_g1b_pipeline.sh``: ``CKPT=$WSM/openpi_assets/openpi-assets/checkpoints/pi05_libero``), and the
pinned encoder was trained on those features. The RoboCerebra arms are FULL finetunes
(``freeze_filter`` defaults to ``nnx.Nothing`` — nothing is frozen), so an arm's backbone has drifted
30k steps away from the one that produced the store. Tapping the served arm would therefore feed the
frozen encoder out-of-distribution features — the same class of train/serve skew that invalidated a
previous GR00T eval, arriving through the tap instead of through the encoder file.

``--tap-source frozen`` (DEFAULT) taps ``--tap-checkpoint`` (the released pi05_libero) with
``--tap-config pi05_libero``: byte-identical to how the store was built. ``--tap-source policy`` taps
the served arm in-process (one model instead of two, ~6 GB less GPU) and prints a loud banner saying
the served ω is not the trained ω. Both are in-process; neither shells out.

``--tap-pad-batch`` (default 16) replicates row 0 up to 16 rows before calling the tap because the
store was built with ``--batch-size 16`` and XLA picks a different kernel for B=1 — measured on the
RoboCasa pi tap as up to max|Δω| ~ 1.43 on |ω| ~ 2.8. A transformer PREFIX has no cross-example
interaction, so the pad rows cannot change the real row's output, only which kernel runs. Set 0 to
disable.

--------------------------------------------------------------------------------------------------
K ENVS PER GPU (``WSM_ENVS_PER_GPU``) — what gets batched, and why that is not a semantics change
--------------------------------------------------------------------------------------------------
``WSM_ENVS_PER_GPU=K`` (K>1) makes openpi's ``WebsocketPolicyServer`` coalesce K clients' requests
into one ``infer_batch`` call (<= ``WSM_GATHER_MAX_BATCH``, default 8, or a ``WSM_GATHER_WAIT_MS``
window, default 20). Two calls then happen once per window instead of once per env:

* **the ω tap.** This is the whole point. The tap is priced per CALL, not per frame — measured on
  this box: 1 frame 2145 ms, 8 frames 2264 ms, 16 frames 2568 ms — because ``--tap-pad-batch 16``
  already pays for 16 rows whatever happens. Packing the due envs' real frames into that same
  already-paid-for call is measured BIT-EXACT against the 1-real-row-padded-to-16 call
  (``max|Δtok| = 0.000e+00``): a transformer PREFIX has no cross-example interaction, and the pad
  pins which kernel runs. So this is a pure scheduling win, not an approximation.
* **the policy.** ``--policy-pad-batch`` (default 8) replicate-pads it so ONE XLA executable serves
  every window size — without that, openpi's ``(4, 8)`` bucketing makes a 3-request window and a
  6-request window different kernels, and gather composition would move the last bits of an action.
  With it, the outcome is invariant to K, composition and ordering, bitwise (measured). It is NOT
  bit-equal to the unbatched batch-1 path — a different row count is a different kernel — so score
  every arm on one path. ``--policy-batch serial`` keeps the tap gathered but runs the policy at
  batch 1, which IS bit-identical to the unbatched path, at a smaller speedup.

Per-env state (``ServeWindow``, encoder prefix, step counters, re-pin flags) is keyed by the
client's ``wsm_env_id`` — the launcher gives each runner its own — and a batch containing another
env's ``t=0`` reset or ``wsm_repin`` provably does not perturb its neighbours
(``test_gather_eval.py`` A/B). ``/healthz`` is answered in the HTTP ``process_request`` hook before
the websocket upgrade, and openpi's wire protocol has no other control message, so nothing that
should bypass the gather ever enters it.

Requests must carry ``policy_noise_seed`` (harness ``--deterministic-seeding``); otherwise
``Policy.infer`` draws action noise from a mutable, arrival-ordered rng and K>1 is not reproducible.

``WSM_ENVS_PER_GPU=1`` (default) keeps the LEGACY path: no gather, ``Policy.infer`` at batch 1,
unchanged.

--------------------------------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------------------------------
    # openpi venv; wsmv2 (WorkspaceEncoder) + robocasa + robosuite on PYTHONPATH, because
    # openpi.training.config imports openpi.groot_utils.groot_openpi_dataset -> robocasa.
    PYTHONPATH=~/Research/TRI/wsmv2:~/Research/robocasa:~/Research/robosuite \
    ~/Research/robocasa_openpi/.venv/bin/python \
        scripts/robocerebra/serve_pi05_libero_wsm.py \
        --checkpoint <arm_ckpt_step_dir> --config pi05_robocerebra_gdn_w8 --port 8000

    # env side (LIBERO sim venv), --wsm makes the client stamp the identity fields
    ... eval_robocerebra_openpi.py --wsm --arm A1_gdn_w8 --ckpt-sha <sha> --encoder-sha 09a1107d...

    # CPU-only window/stride unit test — no GPU, no checkpoint, no torch, no jax, no robocasa
    .../python scripts/robocerebra/serve_pi05_libero_wsm.py --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import pathlib
import sys
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

# ``pool_grid``/``N_IMG_TOK`` come from the authoritative tap so the 14x14 -> 8x8 binning and the
# 196-token span have exactly one definition in the campaign. omega_tap.py keeps every heavy import
# inside main(), so this costs nothing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from omega_tap import N_IMG_TOK, pool_grid  # noqa: E402

# The ω encoder class lives in `workspace_models/`, i.e. the wsmv2 repo root two levels up. Put it
# on the path HERE rather than relying on the caller exporting PYTHONPATH: this server is launched
# from the openpi checkout (which needs robocasa/robosuite on the path for its own reasons), and a
# caller that gets that export right but omits the wsmv2 root fails only after both ~3B models are
# already resident on the GPU. Found exactly that way by the first plumbing run against a real arm.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from wsm_settings import WSM_DATA_ROOT  # noqa: E402

# ---------------------------------------------------------------------------------------------
# PINNED ω encoder. This exact file produced the training-time ω feature store (verified: it
# reproduces the stored features to 3.5e-5; the a3 encoder differs by 13.4). Serving any other
# encoder is the failure that invalidated a previous GR00T eval, so the sha is checked at startup
# and a mismatch refuses to start.
# ---------------------------------------------------------------------------------------------
PINNED_ENCODER = str(WSM_DATA_ROOT / "robocerebra" / "omega_retrain_a2" / "encoder_best.pt")
PINNED_ENCODER_SHA256 = "09a1107d486ae6bfe3112e4858c3a9101e8a934297b21b8fbb13cb3118acc483"
PINNED_ENCODER_STEP = 1000
PINNED_ENCODER_CFG = {
    "dim": 512,
    "n_layers": 4,
    "n_heads": 8,
    "backbone_dim": 2048,
    "proprio_dim": 2048,
    "lang_dim": 2048,
    "input_norm": False,
}

# The released pi05_libero checkpoint the ω store was tapped from (run_g1b_pipeline.sh).
DEFAULT_TAP_CHECKPOINT = str(WSM_DATA_ROOT / "robocerebra/openpi_assets/openpi-assets/checkpoints/pi05_libero")

OMEGA_DIM = 512
FRAMES_PER_EPISODE = 64  # the tap's --frames-per-ep; defines the ω grid
N_IMG_SLOTS = 3  # pi0.5 always lays out 3 image slots; RoboCerebra fills 2

#: Client -> server fields. Every one of these is stripped before the policy transform runs.
IDENTITY_KEYS = ("wsm_env_id", "wsm_t", "wsm_episode_len")
OPTIONAL_KEYS = ("wsm_episode_id", "wsm_repin")
SIGNAL_KEYS = IDENTITY_KEYS + OPTIONAL_KEYS


def sha256_file(path: str | pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def grid_stride_for_episode(episode_len: int, frames_per_episode: int = FRAMES_PER_EPISODE) -> int:
    """Serve-side ω grid stride for an episode of ``episode_len`` env steps.

    ``episode_len / 64`` rounded, floored at 1 — the eval-time reconstruction of the training tap's
    ``np.linspace(0, length - 1, 64)`` cadence. See the module docstring.
    """
    if episode_len < 1:
        raise ValueError(f"episode_len must be >= 1, got {episode_len}")
    if frames_per_episode < 1:
        raise ValueError(f"frames_per_episode must be >= 1, got {frames_per_episode}")
    return max(1, int(round(episode_len / frames_per_episode)))


# =============================================================================================
# ω source: pi05 tap + frozen encoder (the only GPU-touching part; imports stay local)
# =============================================================================================
class Pi05Tap:
    """The ``omega_tap.py`` prefix tap, in-process, over one pi05 policy's model + input transform.

    ``embed`` is ``omega_tap.embed_batch`` with two deliberate additions: an optional replicate-pad up
    to the store's build batch (kernel parity, see the module docstring) and the fp16 round-trip the
    store itself imposed (``tokens`` were saved as float16 and re-read as float32, so the encoder was
    trained on fp16-quantised tokens — reproducing that costs nothing and removes a needless delta).
    """

    def __init__(self, policy, *, pad_batch: int = 16, tokens_fp16: bool = True, label: str = "tap"):
        self._model = policy._model
        self._input_transform = getattr(policy, "_input_transform", None) or policy.input_transform
        self._pad_batch = int(pad_batch)
        self._tokens_fp16 = bool(tokens_fp16)
        self._label = label
        self.calls = 0  # tap forwards issued (the eval's dominant cost — count it, don't guess)
        self.real_rows = 0  # real frames those forwards carried; calls/real_rows == the gather win
        if self._pad_batch < 0:
            raise ValueError(f"pad_batch must be >= 0, got {pad_batch}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | pathlib.Path, config_name: str, assets_link_root: str | None = None, **kwargs
    ) -> "Pi05Tap":
        """Build a private tap policy (the released pi05_libero the ω store was built from)."""
        import openpi.policies.policy_config as policy_config
        import openpi.training.config as _config

        checkpoint = pathlib.Path(checkpoint).expanduser().resolve()
        link_root = link_assets(checkpoint, config_name, assets_link_root)
        train_config = dataclasses.replace(_config.get_config(config_name), assets_base_dir=str(link_root))
        policy = policy_config.create_trained_policy(train_config, checkpoint)
        return cls(policy, label=f"{config_name}@{checkpoint.name}", **kwargs)

    @staticmethod
    def example_from_obs(obs: dict) -> dict:
        """The 4 fields the tap consumes. Built explicitly so no ``wsm_*`` key can leak into it."""
        return {
            "observation/image": np.asarray(obs["observation/image"], dtype=np.uint8),
            "observation/wrist_image": np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
            "observation/state": np.asarray(obs["observation/state"], dtype=np.float32),
            "prompt": str(np.asarray(obs["prompt"]).item() if np.asarray(obs["prompt"]).ndim == 0 else obs["prompt"]),
        }

    def embed(self, examples: Sequence[dict]):
        """[B] examples -> (tokens [B,128,2048] f32, pooled_img [B,2048], pooled_lang [B,2048]).

        More than ``pad_batch`` examples are CHUNKED at ``pad_batch`` rather than sent as one wide
        call, because the whole point of the pad is that exactly one row count ever reaches XLA. A
        21-row call would silently be a third kernel. ``embed([x])`` is one chunk and is therefore
        bit-identical to what this method did before chunking existed.
        """
        rows = list(examples)
        if not rows:
            raise ValueError("Pi05Tap.embed requires at least one example")
        if self._pad_batch and len(rows) > self._pad_batch:
            parts = [self._embed_chunk(rows[i : i + self._pad_batch]) for i in range(0, len(rows), self._pad_batch)]
            return tuple(np.concatenate(pieces, axis=0) for pieces in zip(*parts))
        return self._embed_chunk(rows)

    def _embed_chunk(self, examples: Sequence[dict]):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        real_rows = len(examples)
        if real_rows == 0:
            raise ValueError("Pi05Tap.embed requires at least one example")
        call = list(examples)
        if self._pad_batch > real_rows:
            # Replicated rows are copies of row 0. A transformer PREFIX has no cross-example
            # interaction, so they cannot change the real rows' outputs — only which XLA kernel runs.
            call = call + [dict(examples[0])] * (self._pad_batch - real_rows)

        self.calls += 1
        self.real_rows += real_rows
        transformed = [self._input_transform(dict(example)) for example in call]
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *transformed)
        obs = _model.Observation.from_dict(batch)
        prefix_tokens, prefix_mask, ar_mask = self._model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (hidden, _), _ = self._model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
        hidden = np.asarray(hidden.astype(jnp.float32))[:real_rows]
        mask = np.asarray(prefix_mask)[:real_rows]

        n_img_total = N_IMG_SLOTS * N_IMG_TOK
        img_mask = mask.copy()
        img_mask[:, n_img_total:] = False
        lang_mask = mask.copy()
        lang_mask[:, :n_img_total] = False

        def masked_mean(x, m):
            m3 = m[..., None]
            return (x * m3).sum(1) / np.maximum(m3.sum(1), 1)

        tokens = np.stack(
            [
                np.concatenate([pool_grid(hidden[b, :N_IMG_TOK]), pool_grid(hidden[b, N_IMG_TOK : 2 * N_IMG_TOK])])
                for b in range(hidden.shape[0])
            ]
        )
        if self._tokens_fp16:
            # The store round-tripped tokens through float16; the encoder was trained on that.
            tokens = tokens.astype(np.float16)
        tokens = tokens.astype(np.float32)
        return tokens, masked_mean(hidden, img_mask), masked_mean(hidden, lang_mask)


def load_omega_encoder(path: str, device: str, expected_sha: str | None):
    """Load the PINNED frozen ω encoder, verifying its sha256 first. Refuses on mismatch.

    Load recipe is ``precompute_omega.py``'s, except ``strict=True``: that script used
    ``strict=False``, which would silently tolerate a renamed/missing tensor. All 54 ``encoder.*``
    keys match the module exactly, so strict loading is the same weights with a louder failure mode.
    """
    from types import SimpleNamespace

    import torch

    from workspace_models.networks.workspace_latent import WorkspaceEncoder

    path = str(pathlib.Path(path).expanduser())
    actual = sha256_file(path)
    if expected_sha and actual != expected_sha:
        raise SystemExit(
            f"[serve-rc-wsm] ω ENCODER SHA MISMATCH — refusing to start.\n"
            f"  file     {path}\n  expected {expected_sha}\n  actual   {actual}\n"
            "Serving an encoder other than the one that built the ω feature store makes every number "
            "produced by this server meaningless (cf. the GR00T eval that read 0% for exactly this "
            "reason). Fix the --encoder path, do not relax this check."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["cfg"])
    for key, want in PINNED_ENCODER_CFG.items():
        got = getattr(cfg, key, None)
        if got != want:
            raise SystemExit(f"[serve-rc-wsm] encoder cfg.{key}={got!r}, expected {want!r}")
    scale = float(blob.get("feat_scale", 1.0))
    if scale != 1.0:
        raise SystemExit(
            f"[serve-rc-wsm] encoder was saved with feat_scale={scale}; the ω store was "
            "built without any feature scaling — refusing to guess where to apply it."
        )
    encoder = WorkspaceEncoder(cfg)
    encoder.load_state_dict(
        {k[len("encoder.") :]: v for k, v in blob["model"].items() if k.startswith("encoder.")},
        strict=True,
    )
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        if not bool(torch.isfinite(parameter).all()):
            raise SystemExit("[serve-rc-wsm] ω encoder holds NON-FINITE weights — refusing to serve.")
    logging.info("[serve-rc-wsm] ω encoder %s step=%s sha256=%s cfg=%s", path, blob.get("step"), actual, blob["cfg"])
    return encoder, cfg, actual


class Pi05OmegaSource:
    """Per-env causal ω_t: tap one frame, fuse it, re-run the frozen temporal stack over the prefix.

    ``WorkspaceEncoder`` is causal with ``c_horizon=1000`` — far longer than the ~64 grid frames of a
    RoboCerebra episode — so ω_t depends on the WHOLE prefix, and a truncated history would not
    reproduce the ω the arms trained on. ``fuse_inputs`` is frame-local (PatchPool + two linear
    projections), so only its ``[dim]`` output is cached per frame and the quadratic re-projection of
    raw patch grids never happens. ``encode_fused`` over frames ``0..i`` then yields exactly the
    ``ω_i`` that ``precompute_omega.py`` produced for the same frames, because the causal mask makes
    the full-episode forward and the running-prefix forward agree row by row.

    ``time_emb`` is indexed by GRID POSITION, so the prefix must not be cleared mid-episode unless you
    intend position 0 (which always looked like "episode start" in training) to reappear — see
    ``--repin-scope``.
    """

    def __init__(self, tap: Pi05Tap, encoder, cfg, device: str):
        self._tap = tap
        self._encoder = encoder
        self._cfg = cfg
        self._device = device
        self._prefix: dict[str, tuple[list, list]] = {}

    def reset(self, env_id: str) -> None:
        self._prefix.pop(env_id, None)

    def encode(self, env_id: str, obs: dict) -> np.ndarray:
        """One env. Exactly ``encode_batch`` with a one-row batch, so there is one definition."""
        return self.encode_batch([env_id], [obs])[0]

    def encode_batch(self, env_ids: Sequence[str], obs_list: Sequence[dict]) -> list[np.ndarray]:
        """N envs whose ω grid position is due -> N ω_t, using **ONE** backbone tap call.

        This is the whole performance change. The tap is per-CALL priced, not per-frame (measured on
        this box: 1 frame 2145 ms, 8 frames 2264 ms, 16 frames 2568 ms) because ``--tap-pad-batch 16``
        already pays for 16 rows whatever happens, so N due envs served one-at-a-time pay N x 2.1 s
        for work that costs 2.3 s together. Packing N real frames into that already-paid-for 16-row
        call is measured BIT-EXACT against the 1-real-row-padded-to-16 call this replaced
        (``max|Δtok| = 0.000e+00`` per row), which is what makes the speedup outcome-neutral rather
        than merely fast: a transformer PREFIX has no cross-example interaction, so a row cannot see
        its neighbours, and the 16-row pad pins which kernel runs.

        The frozen ω encoder is still stepped PER ENV, because each env's causal prefix has its own
        length and its own grid position. That loop is a 4-layer/512-dim forward over <=64 frames —
        microseconds — and keeping it per-env is what keeps the state isolation obvious.
        """
        import torch

        env_ids = list(env_ids)
        obs_list = list(obs_list)
        if len(env_ids) != len(obs_list):
            raise ValueError(f"{len(env_ids)} env ids vs {len(obs_list)} observations")
        if not env_ids:
            return []
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"encode_batch needs distinct envs, got {env_ids}")

        tokens, pooled_img, pooled_lang = self._tap.embed([Pi05Tap.example_from_obs(obs) for obs in obs_list])
        for i, env_id in enumerate(env_ids):
            if not (
                np.isfinite(tokens[i]).all() and np.isfinite(pooled_img[i]).all() and np.isfinite(pooled_lang[i]).all()
            ):
                raise RuntimeError(f"[serve-rc-wsm] NON-FINITE backbone tap for env {env_id!r}")

        def frame(array) -> "torch.Tensor":
            """[...] -> [1, 1, ...]: one batch row, one timestep, exactly as the encoder wants."""
            return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self._device)[None, None]

        omegas: list[np.ndarray] = []
        with torch.no_grad():
            for i, env_id in enumerate(env_ids):
                fused_list, cond_list = self._prefix.setdefault(env_id, ([], []))
                if len(fused_list) >= int(self._cfg.max_t):
                    raise RuntimeError(
                        f"[serve-rc-wsm] env {env_id!r} exceeded the encoder's time-embedding "
                        f"capacity ({self._cfg.max_t} grid frames); was the episode ever reset?"
                    )
                fused, cond = self._encoder.fuse_inputs(frame(tokens[i]), frame(pooled_img[i]), frame(pooled_lang[i]))
                fused_list.append(fused[0, 0])
                cond_list.append(cond[0, 0])
                omega = self._encoder.encode_fused(torch.stack(fused_list)[None], torch.stack(cond_list)[None])
                if not bool(torch.isfinite(omega).all()):
                    raise RuntimeError(
                        f"[serve-rc-wsm] NON-FINITE ω at env {env_id!r}, grid {len(fused_list) - 1} "
                        "(mismatched/diverged encoder checkpoint?)"
                    )
                omegas.append(omega[0, -1].float().cpu().numpy().astype(np.float32))
        return omegas


# =============================================================================================
# identity-aware wrapper
# =============================================================================================
class BatchValidationError(RuntimeError):
    """A request was rejected BEFORE any ω state was mutated.

    The distinction is load-bearing under gather-batching. A validation failure is provably
    side-effect-free (``_validate_batch`` runs to completion before ``_prepare_batch`` touches a
    single ``ServeWindow``), so the window's other K-1 requests -- which belong to unrelated, healthy
    episodes -- can be safely re-offered one at a time instead of dying with it. Any failure raised
    LATER (a non-finite ω, a model error) is a plain RuntimeError and stays fatal for the window,
    because by then state has moved and a retry would double-push.
    """


@dataclasses.dataclass
class _EpisodeState:
    """All mutable ω state owned by exactly one rollout client."""

    episode_id: str
    episode_len: int
    stride: int
    window: Any  # ServeWindow (openpi.training.robocerebra_omega)
    last_t: int = -1
    encodes: int = 0
    repins: int = 0
    current: np.ndarray | None = None


class OmegaPiInferWrapper:
    """Online-ω wrapper around one pi0.5 policy, for one or many concurrent RoboCerebra clients.

    Every request carries ``wsm_env_id``, ``wsm_t`` (env step within the episode, post-wait) and
    ``wsm_episode_len``; optionally ``wsm_episode_id`` and ``wsm_repin``. ``wsm_t == 0`` resets that
    env slot and nothing else. Later requests must be strictly time-ordered and must not change the
    episode identity without a ``t == 0``. Validation runs over the whole batch BEFORE any state is
    mutated, so a malformed batch cannot half-reset a live episode.

    ``wsm_repin`` exists because RoboCerebra's ``resume=True`` protocol teleports the sim to the
    demo's ground-truth state at every subtask boundary. The ω already in the buffer describes a
    workspace the robot is no longer in, so the window is cleared and re-seeded from the post-re-pin
    frame on that same request (``ServeWindow.due`` is True on an empty buffer, so the re-seed is not
    a special case). Whether the ENCODER prefix is also cleared is ``repin_resets_encoder``: the
    default leaves it, because ω_i's ``time_emb`` position must keep counting grid frames from the
    episode start to match training, and clearing it would make every subtask look like a fresh
    episode to the encoder.
    """

    STATE_MODE = "per_env_isolated_rc_v1"

    def __init__(
        self,
        policy,
        source,
        *,
        window: int,
        frames_per_episode: int = FRAMES_PER_EPISODE,
        max_envs: int = 1,
        repin_resets_encoder: bool = False,
        log_every: int = 20,
        metadata_extra: dict | None = None,
    ):
        from openpi.training.robocerebra_omega import ServeWindow  # canonical buffer; never reimplemented

        self._policy = policy
        self._source = source
        self._ServeWindow = ServeWindow
        self._window = int(window)
        self._frames_per_episode = int(frames_per_episode)
        self._max_envs = int(max_envs)
        self._repin_resets_encoder = bool(repin_resets_encoder)
        self._log_every = int(log_every)
        self._metadata_extra = dict(metadata_extra or {})
        if self._window < 1 or self._max_envs < 1:
            raise ValueError(f"window and max_envs must be >= 1; got {window}, {max_envs}")
        self._states: dict[str, _EpisodeState] = {}
        self._lock = threading.RLock()
        self._encodes = 0
        #: realised gather sizes, newest last. Test D reads it; a K>1 canary that finds all 1s has a
        #: gather that never gathered (sparse arrivals) and is paying the wait for nothing.
        self.batch_sizes: list[int] = []

    # ------------------------------------------------------------------ field parsing
    @staticmethod
    def _scalar(obs: dict, key: str):
        if key not in obs:
            raise BatchValidationError(f"[serve-rc-wsm] missing required identity field {key!r}")
        arr = np.asarray(obs[key])
        if arr.ndim != 0:
            raise BatchValidationError(f"[serve-rc-wsm] field {key!r} must be scalar; got shape {arr.shape}")
        return arr.item()

    @staticmethod
    def _as_int(value, key: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise BatchValidationError(f"[serve-rc-wsm] {key} must be an integer; got {value!r}")
        return int(value)

    def _identity(self, obs: dict) -> tuple[str, str, int, int, bool]:
        env_id = str(self._scalar(obs, "wsm_env_id"))
        if not env_id:
            raise BatchValidationError("[serve-rc-wsm] wsm_env_id must be a non-empty string")
        t = self._as_int(self._scalar(obs, "wsm_t"), "wsm_t")
        if t < 0:
            raise BatchValidationError(f"[serve-rc-wsm] wsm_t must be non-negative; got {t}")
        episode_len = self._as_int(self._scalar(obs, "wsm_episode_len"), "wsm_episode_len")
        if episode_len < 1:
            raise BatchValidationError(f"[serve-rc-wsm] wsm_episode_len must be >= 1; got {episode_len}")
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
            raise BatchValidationError(
                f"[serve-rc-wsm] duplicate wsm_env_id values in one batch: {sorted(duplicates)}"
            )

        new_envs = {env_id for env_id, _eid, t, _len, _rp in identities if t == 0 and env_id not in self._states}
        if len(self._states) + len(new_envs) > self._max_envs:
            raise BatchValidationError(
                f"[serve-rc-wsm] active env-state bound exceeded: have {len(self._states)} "
                f"({sorted(self._states)}), new {sorted(new_envs)}, max {self._max_envs}; refusing "
                "live-state eviction. Env slots are never garbage-collected on disconnect, because "
                "a closed socket does not prove the episode ended. So a server's id set is fixed "
                "for its lifetime: give the K runners env0..env{K-1} and REUSE those ids across "
                "cells (a t=0 request resets an existing slot for free). To cycle through fresh "
                "ids instead, raise --max-envs or restart the server."
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
                        f"[serve-rc-wsm] duplicate active episode identity {episode_id!r} on envs "
                        f"{owner!r} and {env_id!r}"
                    )
                if episode_id:
                    active[episode_id] = env_id
                continue
            if state is None:
                raise BatchValidationError(f"[serve-rc-wsm] env {env_id!r} sent t={t} before an explicit t=0 reset")
            if (episode_id, episode_len) != (state.episode_id, state.episode_len):
                raise BatchValidationError(
                    f"[serve-rc-wsm] episode identity changed without a t=0 reset for env {env_id!r}: "
                    f"active={(state.episode_id, state.episode_len)!r}, "
                    f"got={(episode_id, episode_len)!r}"
                )
            if t <= state.last_t:
                raise BatchValidationError(
                    f"[serve-rc-wsm] out-of-order wsm_t for env {env_id!r}: last={state.last_t}, got={t}"
                )
        return identities

    # ------------------------------------------------------------------ ω lifecycle
    def _prepare_batch(self, obs_list: Sequence[dict]) -> list[dict]:
        identities = self._validate_batch(obs_list)

        for env_id, episode_id, t, episode_len, _repin in identities:
            if t != 0:
                continue
            stride = grid_stride_for_episode(episode_len, self._frames_per_episode)
            self._source.reset(env_id)
            self._states[env_id] = _EpisodeState(
                episode_id=episode_id,
                episode_len=episode_len,
                stride=stride,
                window=self._ServeWindow(w=self._window, stride=stride, dim=OMEGA_DIM),
            )
            logging.info(
                "[serve-rc-wsm] env=%s episode=%r start: len=%d stride=%d (=round(%d/%d)) "
                "K=%d -> window spans ~%.1f%% of the episode",
                env_id,
                episode_id,
                episode_len,
                stride,
                episode_len,
                self._frames_per_episode,
                self._window,
                100.0 * self._window / self._frames_per_episode,
            )

        for env_id, _episode_id, t, _episode_len, repin in identities:
            if t == 0 or not repin:
                continue
            state = self._states[env_id]
            state.window.reset()
            state.repins += 1
            if self._repin_resets_encoder:
                self._source.reset(env_id)
            logging.info(
                "[serve-rc-wsm] env=%s t=%d RE-PIN #%d: ω window cleared%s",
                env_id,
                t,
                state.repins,
                " + encoder prefix cleared" if self._repin_resets_encoder else "",
            )

        # ---------------------------------------------------------------- ONE tap call per window
        # `due` is read for every env BEFORE any push, so the set of envs that encode on this request
        # is a pure function of each env's own (grid position, t) — never of who it was batched with.
        due = [i for i, (env_id, _eid, t, _len, _rp) in enumerate(identities) if self._states[env_id].window.due(t)]
        omegas: dict[int, np.ndarray] = {}
        if due:
            encode_batch = getattr(self._source, "encode_batch", None)
            if callable(encode_batch):
                vectors = encode_batch([identities[i][0] for i in due], [obs_list[i] for i in due])
            else:  # source predating the batched protocol (the CPU fakes); one call per env
                vectors = [self._source.encode(identities[i][0], obs_list[i]) for i in due]
            if len(vectors) != len(due):
                raise RuntimeError(f"[serve-rc-wsm] ω source returned {len(vectors)} vectors for {len(due)} envs")
            omegas = dict(zip(due, vectors))

        injected: list[dict] = []
        for index, (obs, (env_id, _episode_id, t, _episode_len, _repin)) in enumerate(zip(obs_list, identities)):
            state = self._states[env_id]
            if index in omegas:
                omega = np.asarray(omegas[index], dtype=np.float32).reshape(-1)
                if omega.shape[0] != OMEGA_DIM:
                    raise RuntimeError(f"[serve-rc-wsm] ω dim {omega.shape[0]} != {OMEGA_DIM} for env {env_id!r}")
                if not np.isfinite(omega).all():
                    raise RuntimeError(
                        f"[serve-rc-wsm] NON-FINITE ω at env={env_id!r} t={t} — refusing to serve "
                        "(mismatched/diverged encoder checkpoint?)"
                    )
                state.window.push(omega, t)  # ServeWindow re-checks finiteness; belt and braces
                state.encodes += 1
                self._encodes += 1
                if self._log_every and self._encodes % self._log_every == 0:
                    logging.info(
                        "[serve-rc-wsm] env=%s t=%d grid=%d ω rms=%.4f absmax=%.3f",
                        env_id,
                        t,
                        state.encodes - 1,
                        float(np.sqrt((omega**2).mean())),
                        float(np.abs(omega).max()),
                    )
            window = state.window.get()
            if window.shape != (self._window, OMEGA_DIM) or not np.isfinite(window).all():
                raise RuntimeError(f"[serve-rc-wsm] bad ω window {window.shape} for env {env_id!r} at t={t}")
            state.last_t = t

            item = {key: value for key, value in obs.items() if key not in SIGNAL_KEYS}
            item["wsm_w_window"] = np.ascontiguousarray(window, dtype=np.float32)
            injected.append(item)
        return injected

    # ------------------------------------------------------------------ policy calls
    def infer(self, obs: dict, **kwargs) -> dict:
        """LEGACY single-request path (K=1, gather off). Byte-identical to before gather existed:
        one env's ω, then ``Policy.infer`` at batch 1 with no bucket padding."""
        with self._lock:
            self.batch_sizes.append(1)
            return self._policy.infer(self._prepare_batch([obs])[0], **kwargs)

    def _serve(self, obs_list: Sequence[dict], **kwargs) -> list[dict]:
        injected = self._prepare_batch(list(obs_list))
        # batched vs serial (the bit-identity mode) is decided by the FixedPadBatchPolicy this
        # wrapper was handed — one implementation, shared with the plain A0/A3 server.
        results = self._policy.infer_batch(injected, **kwargs)
        if len(results) != len(injected):
            raise RuntimeError(
                f"[serve-rc-wsm] policy.infer_batch returned {len(results)} results for {len(injected)} requests"
            )
        return results

    @staticmethod
    def _duplicated_env_ids(obs_list: Sequence[dict]) -> set[str]:
        """env ids appearing more than once in one window. Fatal for EVERY copy: two runners sharing
        an id is a launch bug, and serving them independently would silently interleave one ω
        window between two episodes — the exact corruption the identity layer exists to prevent."""
        seen: dict[str, int] = {}
        for obs in obs_list:
            try:
                env_id = str(np.asarray(obs["wsm_env_id"]).item())
            except Exception:  # noqa: BLE001 - unparseable requests are rejected on their own merits
                continue
            seen[env_id] = seen.get(env_id, 0) + 1
        return {env_id for env_id, count in seen.items() if count > 1}

    def infer_batch(self, obs_list: Sequence[dict], **kwargs) -> list[dict]:
        """Serve one gather window. Returns one entry per request, IN ORDER; an entry may be an
        Exception, which the gather routes to that caller alone (see ``batch_gather``).

        Without the per-request fallback below, K=8 means one malformed runner -- a stale env id, a
        restarted shard, a step counter that went backwards -- raises for the whole window and kills
        seven healthy multi-hour episodes with it. Observed exactly that way while testing.
        """
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
                    "[serve-rc-wsm] window of %d rejected at validation (%s); "
                    "re-offering each request on its own so healthy envs survive",
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
                            f"[serve-rc-wsm] duplicate wsm_env_id {env_id!r} in one batch: two runners "
                            "share an env slot. Give every runner its own --wsm-env-id."
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
                "wsm_state_mode": self.STATE_MODE,
                "wsm_window": self._window,
                "wsm_frames_per_episode": self._frames_per_episode,
                "wsm_stride_rule": "max(1, round(wsm_episode_len / frames_per_episode))",
                "wsm_required_identity_fields": list(IDENTITY_KEYS),
                "wsm_optional_fields": list(OPTIONAL_KEYS),
                "wsm_repin_resets_encoder": self._repin_resets_encoder,
                "wsm_max_envs": self._max_envs,
                **self._metadata_extra,
            }
        )
        return metadata

    def gather_stats(self) -> dict:
        """Observability for the K>1 canary: did the gather actually gather, and how much tap did it save."""
        sizes = list(self.batch_sizes)
        tap = getattr(getattr(self._source, "_tap", None), "calls", None)
        rows = getattr(getattr(self._source, "_tap", None), "real_rows", None)
        return {
            "policy_calls": len(sizes),
            "requests": sum(sizes),
            "mean_batch": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max_batch": max(sizes) if sizes else 0,
            "hist": {size: sizes.count(size) for size in sorted(set(sizes))},
            "tap_calls": tap,
            "tap_real_rows": rows,
            "tap_rows_per_call": (rows / tap) if tap else None,
        }


# =============================================================================================
# build helpers
# =============================================================================================
def link_assets(checkpoint: pathlib.Path, config_name: str, link_root: str | None) -> pathlib.Path:
    """The ``serve_pi05_libero.py`` assets trick: ``<root>/<config> -> <ckpt>/assets``.

    ``config.py``'s norm-stats fallback dispatches to a function that does not exist in this fork, so
    a lookup that *should* return ``None`` raises ``AttributeError`` instead. Pointing
    ``assets_base_dir`` at a directory laid out as ``<base>/<config>/<asset_id>/norm_stats.json``
    makes the normal loader find the stats and never reach the fallback. ``save_state`` replicates
    the sealed RoboCerebra stats into ``<ckpt>/assets/<asset_id>/`` at every save, which is what makes
    a trained checkpoint self-sufficient here.
    """
    root = pathlib.Path(link_root or checkpoint.parent / "_serve_assets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    link = root / config_name
    if not link.exists():
        link.symlink_to(checkpoint / "assets", target_is_directory=True)
    return root


def build_arm_policy(checkpoint: pathlib.Path, config_name: str, link_root: str | None):
    """Build the trained arm's policy and return ``(policy, train_config)``."""
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config
    from openpi.training.config import AssetsConfig

    root = link_assets(checkpoint, config_name, link_root)
    train_config = _config.get_config(config_name)
    data = train_config.data
    if getattr(data, "assets", None) is not None and getattr(data.assets, "assets_dir", None):
        # ROBOCEREBRA_ASSETS_DIR is a TRAIN-side knob baked in at config-registry import. At serve the
        # checkpoint's own assets are authoritative, so drop it and let the symlink root answer.
        logging.warning(
            "[serve-rc-wsm] ignoring assets_dir=%s from the config; serving the checkpoint's own assets via %s",
            data.assets.assets_dir,
            root,
        )
        data = dataclasses.replace(data, assets=AssetsConfig(assets_dir=None, asset_id=data.assets.asset_id))
    train_config = dataclasses.replace(train_config, assets_base_dir=str(root), data=data)
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    return policy, train_config


def resolve_window(train_config, policy, explicit: int | None) -> int:
    """K, resolved from the CONFIG and cross-checked STRUCTURALLY against the loaded weights.

    ``model.wsm_cond_window`` is the trained recipe's K. It is not merely declarative: openpi's
    ``BaseModelConfig.load`` runs ``check_pytree_equality(..., check_shapes=True)`` against the
    restored params, so a config/checkpoint K disagreement fails at load rather than here. We then
    re-read K off the built module's ``pos_decay_bias`` ``[K, H]`` leaf — the same structural recovery
    the RoboCasa serve path uses — so the number injected into ``wsm_w_window`` provably matches the
    tensor the conditioner will index. ``--window`` is honoured only as an assertion.
    """
    model_config = train_config.model
    if not getattr(model_config, "wsm_tanh", False):
        raise SystemExit(
            f"[serve-rc-wsm] config {train_config.name!r} has wsm_tanh=False: it reads no ω and must "
            "be served by scripts/robocerebra/serve_pi05_libero.py (arms A0/A3). Refusing to attach "
            "an ω path to a policy that cannot consume it."
        )
    window = int(model_config.wsm_cond_window)

    conditioner = getattr(policy._model, "wsm_tanh_cond", None)
    if conditioner is None:
        raise SystemExit(
            "[serve-rc-wsm] served model has no wsm_tanh_cond subtree — refusing to serve a policy that will ignore ω."
        )
    bias = getattr(conditioner, "pos_decay_bias", None)
    if bias is not None:
        structural = int(np.asarray(getattr(bias, "value", bias)).shape[0])
        if structural != window:
            raise SystemExit(
                f"[serve-rc-wsm] K disagreement: config says {window}, the loaded pos_decay_bias says {structural}"
            )
        logging.info(
            "[serve-rc-wsm] K=%d (config wsm_cond_window, confirmed structurally by pos_decay_bias[%d, ...])",
            window,
            structural,
        )
    else:
        logging.info(
            "[serve-rc-wsm] K=%d from config wsm_cond_window (cond_type=%s has no pos_decay_bias to cross-check)",
            window,
            model_config.wsm_cond_type,
        )
    # Third, independent witness: the DATA side's OmegaSpec.window is what the training loader sliced
    # out of the store. If it ever disagrees with the module's window_len the arm was trained on a
    # window of a different length than it read, and no serve-side choice can be right.
    spec = getattr(train_config.data, "omega", None)
    if spec is not None and int(spec.window) != window:
        raise SystemExit(
            f"[serve-rc-wsm] config {train_config.name!r} is internally inconsistent: "
            f"OmegaSpec.window={spec.window} but wsm_cond_window={window}"
        )
    if explicit is not None and int(explicit) != window:
        raise SystemExit(f"[serve-rc-wsm] --window {explicit} disagrees with the checkpoint's K={window}")
    return window


# =============================================================================================
# CPU-only self test
# =============================================================================================
class _FakeSource:
    """Deterministic ω without a GPU: ω_i = e_0 * i, one distinguishable vector per encode."""

    def __init__(self, nonfinite_at: int | None = None):
        self.calls: list[tuple[str, int]] = []
        self.resets: list[str] = []
        self._n = 0
        self._nonfinite_at = nonfinite_at

    def reset(self, env_id: str) -> None:
        self.resets.append(env_id)

    def encode(self, env_id: str, obs: dict) -> np.ndarray:
        self.calls.append((env_id, int(np.asarray(obs["wsm_t"]).item())))
        omega = np.zeros(OMEGA_DIM, dtype=np.float32)
        omega[0] = float(self._n)
        if self._nonfinite_at is not None and self._n == self._nonfinite_at:
            omega[1] = np.nan
        self._n += 1
        return omega


class _FakePolicy:
    def __init__(self):
        self.seen: list[dict] = []

    def infer(self, obs: dict) -> dict:
        self.seen.append(obs)
        return {"actions": np.zeros((10, 7), dtype=np.float32)}

    def infer_batch(self, obs_list):
        return [self.infer(obs) for obs in obs_list]


def _obs(
    env_id: str, t: int, episode_len: int, *, repin: bool = False, episode_id: str = "Ideal/case1/trial0"
) -> dict:
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the plate",
        "wsm_env_id": env_id,
        "wsm_episode_id": episode_id,
        "wsm_t": t,
        "wsm_episode_len": episode_len,
        "wsm_repin": repin,
    }


def _expect_raises(fn, needle: str) -> None:
    try:
        fn()
    except RuntimeError as error:
        assert needle in str(error), f"wrong error: {error}"
        return
    raise AssertionError(f"expected a RuntimeError containing {needle!r}")


def _self_test() -> None:
    """CPU-only unit test of the stride + window semantics. No GPU, no checkpoint, no torch, no jax."""
    import tempfile

    from openpi.training.robocerebra_omega import OmegaStore, ServeWindow

    print("=" * 92)
    print("serve_pi05_libero_wsm self-test (CPU only)")
    print("=" * 92)

    # -------------------------------------------------------------- 1. the stride rule
    cases = {150 * 6: 14, 150 * 8: 19, 150 * 9: 21, 150 * 4: 9, 64: 1, 30: 1, 1031: 16}
    for episode_len, want in cases.items():
        got = grid_stride_for_episode(episode_len)
        assert got == want, f"stride({episode_len}) = {got}, expected {want}"
    print(
        f"[1] stride rule max(1, round(len/64)) OK on {len(cases)} lengths: "
        + ", ".join(f"{k}->{v}" for k, v in sorted(cases.items()))
    )

    # -------------------------------------------------------------- 2. ServeWindow == OmegaStore.window
    # Built against a SYNTHETIC store read by the REAL OmegaStore, so the oracle is the shipped
    # training-side code path, not a re-implementation of it living in this test.
    rng = np.random.default_rng(0)
    episode_len = 150 * 6
    frames = np.unique(np.linspace(0, episode_len - 1, FRAMES_PER_EPISODE).astype(int))
    vecs = rng.normal(0, 1, (len(frames), OMEGA_DIM)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "store"
        (root / "episode_000000").mkdir(parents=True)
        np.savez_compressed(root / "episode_000000" / "w.npz", w=vecs, frame_indices=frames)
        store = OmegaStore(root)
        stride = grid_stride_for_episode(episode_len)
        for k in (8, 16):
            serve = ServeWindow(w=k, stride=stride, dim=OMEGA_DIM)
            for i, frame in enumerate(frames):
                serve.push(vecs[i], int(frame))
                train_window = store.window(0, int(frame), k)
                assert serve.get().shape == (k, OMEGA_DIM)
                assert np.array_equal(serve.get(), train_window), f"serve window != train window at grid {i} (K={k})"
        print(
            f"[2] ServeWindow == OmegaStore.window at all {len(frames)} grid frames, K in (8, 16), "
            f"synthetic 900-step episode (stride {stride})"
        )

    # -------------------------------------------------------------- 3. same, against the REAL store
    real_root = WSM_DATA_ROOT / "robocerebra" / "omega_features"
    if real_root.is_dir():
        store = OmegaStore(real_root)
        episode = store.episodes()[0]
        real_vecs, real_idx = store._load(episode)
        serve = ServeWindow(w=8, stride=int(round(store.grid_stride_for(episode))), dim=OMEGA_DIM)
        for i, frame in enumerate(real_idx):
            serve.push(real_vecs[i], int(frame))
            assert np.array_equal(serve.get(), store.window(episode, int(frame), 8))
        strides = [store.grid_stride_for(e) for e in store.episodes()[:100]]
        print(
            f"[3] REAL store episode {episode}: {len(real_idx)} ω, ServeWindow parity at every "
            f"frame; measured train stride over 100 eps min={min(strides):.1f} "
            f"mean={np.mean(strides):.1f} max={max(strides):.1f}"
        )
    else:
        print(f"[3] SKIP real-store parity ({real_root} not present)")

    # -------------------------------------------------------------- 4. the actual serve cadence
    policy, source = _FakePolicy(), _FakeSource()
    wrapper = OmegaPiInferWrapper(policy, source, window=8, max_envs=2, log_every=0)
    replan, k = 5, 8
    steps = list(range(0, episode_len, replan))
    for t in steps:
        wrapper.infer(_obs("env0", t, episode_len))
    encoded_at = [t for _env, t in source.calls]
    assert encoded_at[0] == 0 and encoded_at == sorted(set(encoded_at))
    gaps = set(np.diff(encoded_at).tolist())
    assert gaps == {15}, f"expected a uniform 15-step realised cadence (stride 14, replan 5), got {gaps}"
    last = policy.seen[-1]
    assert set(SIGNAL_KEYS).isdisjoint(last), f"signal keys leaked into the policy: {sorted(last)}"
    assert last["wsm_w_window"].shape == (k, OMEGA_DIM) and last["wsm_w_window"].dtype == np.float32
    assert set(last) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
        "wsm_w_window",
    }, sorted(last)
    newest = np.array([last["wsm_w_window"][j, 0] for j in range(k)])
    assert np.array_equal(newest, np.arange(len(encoded_at) - k, len(encoded_at), dtype=np.float32)), (
        f"window is not the last {k} ω, oldest-first: {newest}"
    )
    print(
        f"[4] 900-step episode @ replan 5: {len(steps)} requests -> {len(encoded_at)} encodes "
        f"(cadence 15, ideal 14.06), window = last {k} ω oldest-first, all wsm_* signal keys stripped"
    )

    # -------------------------------------------------------------- 5. re-pin
    before = len(source.calls)
    wrapper.infer(_obs("env0", episode_len, episode_len, repin=True))
    assert len(source.calls) == before + 1, "re-pin must force an immediate re-encode"
    window = policy.seen[-1]["wsm_w_window"]
    assert np.array_equal(window, np.repeat(window[-1:], k, axis=0)), (
        "after a re-pin the window must be the single fresh ω, repeat-padded"
    )
    assert source.resets == ["env0"], f"encoder prefix must survive a re-pin by default: {source.resets}"
    wrapper_e = OmegaPiInferWrapper(
        _FakePolicy(), (src := _FakeSource()), window=8, repin_resets_encoder=True, log_every=0
    )
    wrapper_e.infer(_obs("e", 0, episode_len))
    wrapper_e.infer(_obs("e", 5, episode_len, repin=True))
    assert src.resets == ["e", "e"], f"--repin-scope window+encoder must clear the prefix: {src.resets}"
    print(
        "[5] re-pin: window cleared + re-seeded on the same request, repeat-padded to K; encoder "
        "prefix kept by default, cleared under repin_resets_encoder"
    )

    # -------------------------------------------------------------- 6. identity / ordering guards
    guarded = OmegaPiInferWrapper(_FakePolicy(), _FakeSource(), window=8, max_envs=1, log_every=0)
    guarded.infer(_obs("env0", 0, episode_len))
    guarded.infer(_obs("env0", 5, episode_len))
    checks = [
        (lambda: guarded.infer(_obs("env0", 5, episode_len)), "out-of-order"),
        (lambda: guarded.infer(_obs("env0", 3, episode_len)), "out-of-order"),
        (lambda: guarded.infer(_obs("env0", 10, episode_len, episode_id="other")), "identity changed"),
        (lambda: guarded.infer(_obs("env0", 10, 1350)), "identity changed"),
        (lambda: guarded.infer(_obs("env9", 10, episode_len)), "before an explicit t=0 reset"),
        (lambda: guarded.infer(_obs("env9", 0, episode_len)), "env-state bound exceeded"),
        (
            lambda: guarded.infer({k2: v for k2, v in _obs("env0", 10, episode_len).items() if k2 != "wsm_t"}),
            "missing required identity field",
        ),
    ]
    for fn, needle in checks:
        _expect_raises(fn, needle)
    # Under gather-batching a rejected request is RETURNED as an exception in its own slot rather
    # than raised for the window, so one bad client cannot kill the K-1 healthy episodes beside it.
    # Duplicate env ids are the exception: both copies are rejected, because two runners sharing a
    # slot is a launch bug and serving them would interleave one ω window between two episodes.
    duplicated = guarded.infer_batch([_obs("env0", 10, episode_len), _obs("env0", 11, episode_len)])
    assert len(duplicated) == 2 and all(isinstance(r, BatchValidationError) for r in duplicated), (
        f"both copies of a duplicated env id must be rejected: {duplicated}"
    )
    assert all("duplicate wsm_env_id" in str(r) for r in duplicated)
    assert guarded._states["env0"].last_t == 5, "a rejected request must not advance live state"
    print(
        f"[6] validator rejects all {len(checks)} malformed cases and leaves live state untouched; "
        "a duplicated env id rejects both copies without raising for the window"
    )

    # -------------------------------------------------------------- 7. non-finite ω never serves
    bad = OmegaPiInferWrapper(_FakePolicy(), _FakeSource(nonfinite_at=0), window=8, log_every=0)
    _expect_raises(lambda: bad.infer(_obs("env0", 0, episode_len)), "NON-FINITE")
    print("[7] non-finite ω raises instead of falling back to zeros")

    print("=" * 92)
    print("ALL PASS")


# =============================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--self-test", action="store_true", help="CPU-only window/stride unit test; no GPU, no checkpoint, then exit"
    )
    parser.add_argument("--checkpoint", help="trained arm checkpoint dir (params/ + assets/)")
    parser.add_argument(
        "--config", default="pi05_robocerebra_gdn_w8", help="the arm's registered TrainConfig name (A1/A2/A4)"
    )
    parser.add_argument(
        "--assets-link-root",
        default=None,
        help="where to materialise <config>/<asset_id> (default <ckpt>/../_serve_assets)",
    )
    parser.add_argument("--encoder", default=PINNED_ENCODER, help="PINNED ω encoder .pt")
    parser.add_argument(
        "--encoder-sha256",
        default=PINNED_ENCODER_SHA256,
        help="required sha256 of --encoder; 'none' disables the check (do not)",
    )
    parser.add_argument(
        "--tap-source",
        default="frozen",
        choices=["frozen", "policy"],
        help="frozen: tap the released pi05_libero the ω store was built from "
        "(train/serve parity). policy: tap the served arm (cheaper, SKEWED)",
    )
    parser.add_argument("--tap-checkpoint", default=DEFAULT_TAP_CHECKPOINT)
    parser.add_argument("--tap-config", default="pi05_libero")
    parser.add_argument("--tap-assets-link-root", default=None)
    parser.add_argument(
        "--tap-pad-batch",
        type=int,
        default=16,
        help="replicate-pad the tap call to this many rows (the ω store's build "
        "batch); 0 disables. Kernel parity, not semantics.",
    )
    parser.add_argument(
        "--window", type=int, default=None, help="assert K; K itself always comes from the checkpoint/config"
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=FRAMES_PER_EPISODE,
        help="the tap's frames-per-episode; sets the stride rule's denominator",
    )
    parser.add_argument(
        "--repin-scope",
        default="window",
        choices=["window", "window+encoder"],
        help="what a wsm_repin clears (default: the ω window only)",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=None,
        help="concurrent env slots (default: WSM_ENVS_PER_GPU, which is also what "
        "flips the server's gather-batching on)",
    )
    parser.add_argument(
        "--policy-batch",
        default="batched",
        choices=["batched", "serial"],
        help="batched (default): ONE padded policy call per gather window — fastest, "
        "and provably invariant to K/composition/order, but a different XLA "
        "kernel from the unbatched path. serial: batch the ω TAP (the dominant "
        "cost, and bit-exact) but issue one batch-1 Policy.infer per request — "
        "bit-identical to a K=1 seeded run, at a smaller speedup. Requires "
        "--deterministic-seeding on the harness.",
    )
    parser.add_argument(
        "--policy-pad-batch",
        type=int,
        default=8,
        help="replicate-pad every BATCHED policy call to this many rows so one XLA "
        "kernel serves every gather size (must be an openpi bucket: 4 or 8). "
        "0 = stock openpi bucketing. The unbatched K=1 legacy path never pads.",
    )
    parser.add_argument("--log-every", type=int, default=20, help="log ω rms every N encodes; 0 = off")
    parser.add_argument("--device", default="cuda", help="device for the PyTorch ω encoder")
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

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()

    # 1. the trained arm.
    logging.info("[serve-rc-wsm] building %s from %s", args.config, checkpoint)
    policy, train_config = build_arm_policy(checkpoint, args.config, args.assets_link_root)
    window = resolve_window(train_config, policy, args.window)
    logging.info(
        "[serve-rc-wsm] arm: cond_type=%s K=%d history_dropout=%s ptrm_steps=%s action_horizon=%d",
        train_config.model.wsm_cond_type,
        window,
        train_config.model.wsm_cond_history_dropout,
        train_config.model.wsm_ptrm_steps,
        train_config.model.action_horizon,
    )

    # 2. the backbone tap.
    if args.tap_source == "policy":
        logging.warning("*" * 100)
        logging.warning(
            "[serve-rc-wsm] --tap-source policy: tapping the SERVED ARM's backbone. The ω "
            "store (and hence the frozen encoder) was built on the RELEASED pi05_libero "
            "backbone, and these arms are FULL finetunes, so the served ω is NOT the "
            "trained ω. This is a debug/ablation mode, not a comparable arm."
        )
        logging.warning("*" * 100)
        tap = Pi05Tap(policy, pad_batch=args.tap_pad_batch, label=f"policy:{args.config}")
    else:
        logging.info(
            "[serve-rc-wsm] tap: %s @ %s (the checkpoint the ω store was built from)",
            args.tap_config,
            args.tap_checkpoint,
        )
        tap = Pi05Tap.from_checkpoint(
            args.tap_checkpoint, args.tap_config, args.tap_assets_link_root, pad_batch=args.tap_pad_batch
        )
    logging.info("[serve-rc-wsm] tap call batch: pad to %s rows (store built at 16)", args.tap_pad_batch or "no pad")

    # 3. the pinned frozen ω encoder.
    expected_sha = None if args.encoder_sha256.lower() == "none" else args.encoder_sha256
    if expected_sha is None:
        logging.warning("[serve-rc-wsm] --encoder-sha256 none: the train/serve ω parity check is OFF")
    encoder, cfg, encoder_sha = load_omega_encoder(args.encoder, args.device, expected_sha)
    source = Pi05OmegaSource(tap, encoder, cfg, args.device)

    # 4. kernel pinning for the POLICY call, the twin of --tap-pad-batch for the tap call. Wrapping
    #    the inner openpi policy (not the ω wrapper) is deliberate: the ω wrapper must see the REAL
    #    request list to key per-env state, so the pad rows may only appear after ω is injected.
    from serve_batching import FixedPadBatchPolicy, gather_settings  # local dir, on sys.path above

    k_envs, gather_max_batch, gather_wait_ms = gather_settings()
    if args.policy_pad_batch and gather_max_batch % args.policy_pad_batch:
        raise SystemExit(
            f"[serve-rc-wsm] gather max batch {gather_max_batch} is not a multiple of "
            f"--policy-pad-batch {args.policy_pad_batch}: the last chunk of a full window would be "
            "short and would run a DIFFERENT padded row count than the others. A gather WIDER than "
            "the policy pad is fine and often optimal — the tap batches over the whole window (its "
            "cost is per-CALL) while the policy runs ceil(n/pad) calls all pinned at pad rows."
        )
    policy = FixedPadBatchPolicy(policy, pad_batch=args.policy_pad_batch, mode=args.policy_batch)

    wrapped = OmegaPiInferWrapper(
        policy,
        source,
        window=window,
        frames_per_episode=args.frames_per_episode,
        max_envs=int(k_envs if args.max_envs is None else args.max_envs),
        repin_resets_encoder=(args.repin_scope == "window+encoder"),
        log_every=args.log_every,
        metadata_extra={
            "wsm_policy_pad_batch": int(args.policy_pad_batch),
            "wsm_policy_batch": args.policy_batch,
            "wsm_gather_max_batch": gather_max_batch,
            "wsm_gather_wait_ms": gather_wait_ms,
            "wsm_envs_per_gpu": k_envs,
            "wsm_config": args.config,
            "wsm_checkpoint": str(checkpoint),
            "wsm_encoder": str(args.encoder),
            "wsm_encoder_sha256": encoder_sha,
            "wsm_encoder_step": PINNED_ENCODER_STEP,
            "wsm_tap_source": args.tap_source,
            "wsm_tap_checkpoint": (str(checkpoint) if args.tap_source == "policy" else str(args.tap_checkpoint)),
            "wsm_tap_pad_batch": int(args.tap_pad_batch),
            "wsm_cond_type": train_config.model.wsm_cond_type,
        },
    )
    logging.info(
        "[serve-rc-wsm] READY on %s:%d — K=%d, ω encoder sha %s, tap %s",
        args.host,
        args.port,
        window,
        encoder_sha[:12],
        args.tap_source,
    )
    if k_envs > 1:
        logging.info(
            "[serve-rc-wsm] gather-batching ON: %d env slots, <=%d per batch, %.0f ms "
            "window, tap pad %d rows, policy=%s (pad %d rows)",
            k_envs,
            gather_max_batch,
            gather_wait_ms,
            args.tap_pad_batch,
            args.policy_batch,
            args.policy_pad_batch,
        )
    else:
        logging.info(
            "[serve-rc-wsm] gather-batching OFF (WSM_ENVS_PER_GPU=1): LEGACY single-request "
            "path, Policy.infer at batch 1, unchanged"
        )
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
