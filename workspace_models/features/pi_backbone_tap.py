"""Tap points into the frozen pi0.5 (openpi/JAX) backbone for WSM features (3-view RoboCasa port).

Port of Isaac-GR00T/wsm/pi05_tap_cache.py, adapted to RoboCasa's 3 views + the wsmv2 cache schema.
Loads the frozen pi05 RoboCasa policy (config pi05_rc_mg60_bal33 + a local pretrain ckpt), runs
embed_prefix -> PaliGemma.llm over {3 images + state-in-prompt}, and for each frame returns:

* patch_tokens [B, 192, 2048] fp16 — each view's 14x14 SigLIP grid bin-averaged to 8x8=64, in the
  LABEL view order (agentview_left, agentview_right, eye_in_hand) so it aligns with the salient-patch
  global ids. NOTE the model's image slots are NOT in that order — RobocasaInputs maps
  base_0_rgb<-observation/image=agentview_left (slot0), left_wrist_0_rgb<-wrist_image=eye_in_hand
  (slot1), right_wrist_0_rgb<-right_image=agentview_right (slot2). So we REORDER slots [0,2,1].
* lang_emb [B, 2048] fp16 — masked mean over the language tokens (pi0.5 discretizes the robot STATE
  into the prompt, so this already carries proprio -> the pi WSM is patch+lang, no separate state_emb).

Run in the openpi-jax-latest env (jax 0.10.1 + openpi + robocasa). Frozen backbone (eval / no grad).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from wsm_settings import INTERNAL_TRAINING_ROOT

GRID_IN, GRID_OUT, N_IMG_TOK = 14, 8, 196  # SigLIP So400m/14 @224 -> 14x14=196; bin -> 8x8=64
_EDGES = np.linspace(0, GRID_IN, GRID_OUT + 1).astype(int)  # [0,1,3,5,7,8,10,12,14]
N_VIEWS = 3
PATCHES_PER_VIEW = GRID_OUT * GRID_OUT  # 64
# model image slots -> LABEL view order (agentview_left, agentview_right, eye_in_hand):
# slot0=agentview_left, slot1=eye_in_hand, slot2=agentview_right  ==>  take slots [0,2,1].
SLOT_TO_LABEL = (0, 2, 1)
# location of the wsm RoboCasa openpi configs (registers pi05_rc_mg60_bal33). Env-overridable so the
# SageMaker node points it at the uploaded internal_training/robocasa (the local default is dev-only).
WSM_CONFIGS_DIR = os.environ.get("WSM_CONFIGS_DIR", str(INTERNAL_TRAINING_ROOT / "robocasa"))


def _bin8x8(tok196: np.ndarray) -> np.ndarray:
    """[196, D] (14x14) -> [64, D] (8x8) by non-uniform bin-averaging (matches pi_geometry)."""
    g = tok196.reshape(GRID_IN, GRID_IN, -1)
    out = np.empty((GRID_OUT, GRID_OUT, g.shape[-1]), dtype=g.dtype)
    for i in range(GRID_OUT):
        for j in range(GRID_OUT):
            out[i, j] = g[_EDGES[i] : _EDGES[i + 1], _EDGES[j] : _EDGES[j + 1]].mean(axis=(0, 1))
    return out.reshape(GRID_OUT * GRID_OUT, -1)


def _make_jax_postprocessor():
    """Compile exact 14x14 -> 8x8 binning and language pooling on the accelerator.

    Only the final fp16 `[B,192,D]` patches and `[B,D]` language vectors cross to the host; the
    substantially larger full prefix hidden state stays device-resident.
    """
    import jax
    import jax.numpy as jnp

    edges = tuple(int(value) for value in _EDGES)

    @jax.jit
    def postprocess(phid, pmask):
        phid = phid.astype(jnp.float32)
        batch, _sequence, width = phid.shape
        image = phid[:, : N_VIEWS * N_IMG_TOK].reshape(batch, N_VIEWS, GRID_IN, GRID_IN, width)
        pooled_slots = []
        for slot in range(N_VIEWS):
            cells = [
                jnp.mean(
                    image[:, slot, edges[row] : edges[row + 1], edges[col] : edges[col + 1]],
                    axis=(1, 2),
                )
                for row in range(GRID_OUT)
                for col in range(GRID_OUT)
            ]
            pooled_slots.append(jnp.stack(cells, axis=1))
        patch = jnp.concatenate([pooled_slots[slot] for slot in SLOT_TO_LABEL], axis=1)

        language_mask = pmask.at[:, : N_VIEWS * N_IMG_TOK].set(False)
        weights = language_mask[..., None].astype(phid.dtype)
        language = jnp.sum(phid * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
        return patch.astype(jnp.float16), language.astype(jnp.float16)

    return postprocess


@dataclass
class PiTapResult:
    patch_tokens: np.ndarray  # [B, 192, 2048] fp16 — bin-averaged, LABEL view order
    lang_emb: np.ndarray  # [B, 2048] fp16 — masked-mean language (carries discretized state)


class Pi05BackboneTap:
    """Frozen pi0.5 RoboCasa backbone tap (embed_prefix -> LLM -> bin-avg -> reorder)."""

    def __init__(self, ckpt: str, config_name: str = "pi05_rc_mg60_bal33", configs_dir: str = WSM_CONFIGS_DIR):
        import dataclasses

        from openpi.policies import policy_config
        from openpi.training import config as _config

        if configs_dir and configs_dir not in sys.path:
            sys.path.insert(0, configs_dir)
        import wsm_robocasa_configs  # noqa: F401  (import registers the RoboCasa pi05 configs)

        wsm_robocasa_configs.install()
        cfg = _config.get_config(config_name)
        # Inference tap: the pretrain soup isn't on disk locally, so don't let data.create() recompute
        # norm stats from data_dirs. Clearing data_dirs skips that fallback -> create_trained_policy then
        # loads norm stats from the checkpoint's assets/norm_stats.json (what the model was trained with).
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dirs=None))
        self.policy = policy_config.create_trained_policy(cfg, os.path.expanduser(ckpt))
        self.input_transform = getattr(self.policy, "_input_transform", None) or self.policy.input_transform
        self.model = self.policy._model
        from openpi.shared import nnx_utils

        if not hasattr(self.model, "tap_prefix_hidden"):
            raise RuntimeError("pinned OpenPI source lacks Pi0.tap_prefix_hidden; refusing the slow eager tap")
        self._tap_prefix_hidden = nnx_utils.module_jit(self.model.tap_prefix_hidden)
        self._postprocess = _make_jax_postprocessor()

    def tap(self, frames: dict, state: np.ndarray, prompt) -> PiTapResult:
        """frames: label_view -> uint8 [B,H,W,C] (keys agentview_left/agentview_right/eye_in_hand);
        state: [B, D] float32; prompt: str or [B] canonical terse task instructions."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        b = next(iter(frames.values())).shape[0]
        prompts = [prompt] * b if isinstance(prompt, str) else list(prompt)
        examples = [
            {
                "observation/image": np.asarray(frames["agentview_left"][i], dtype=np.uint8),
                "observation/wrist_image": np.asarray(frames["eye_in_hand"][i], dtype=np.uint8),
                "observation/right_image": np.asarray(frames["agentview_right"][i], dtype=np.uint8),
                "observation/state": np.asarray(state[i], dtype=np.float32),
                "prompt": prompts[i],
            }
            for i in range(b)
        ]

        ts = [self.input_transform(e) for e in examples]
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *ts)
        obs = _model.Observation.from_dict(batch)
        phid, pmask = self._tap_prefix_hidden(obs)
        patch, lang_emb = self._postprocess(phid, pmask)
        return PiTapResult(
            patch_tokens=np.asarray(patch),
            lang_emb=np.asarray(lang_emb),
        )


def _smoke() -> None:
    """Synthetic shape smoke: load the policy, run a tiny random batch, check [B,192,2048]+[B,2048]."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/Research/TRI/wsm_data/wsm_ckpts/pi05_on/149999")
    ap.add_argument("--config", default="pi05_rc_mg60_bal33")
    ap.add_argument("--state-dim", type=int, default=8)
    args = ap.parse_args()
    tap = Pi05BackboneTap(args.ckpt, args.config)
    b = 2
    frames = {
        v: np.random.randint(0, 256, (b, 256, 256, 3), dtype=np.uint8)
        for v in ("agentview_left", "agentview_right", "eye_in_hand")
    }
    state = np.zeros((b, args.state_dim), dtype=np.float32)
    r = tap.tap(frames, state, "open the drawer")
    print(
        f"OK: patch_tokens {r.patch_tokens.shape} {r.patch_tokens.dtype} finite={np.isfinite(r.patch_tokens).all()} | "
        f"lang_emb {r.lang_emb.shape} finite={np.isfinite(r.lang_emb).all()}"
    )
    assert r.patch_tokens.shape == (b, 192, 2048) and r.lang_emb.shape == (b, 2048)


if __name__ == "__main__":
    _smoke()
