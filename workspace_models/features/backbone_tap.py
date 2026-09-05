"""Tap points into the frozen GR00T N1.7 backbone for WSM features (3-view RoboCasa port).

Faithful port of Isaac-GR00T/wsm/backbone_tap.py — it is already embodiment-agnostic (reads
`policy.modality_configs`), so the SAME code yields 3 RoboCasa views (192 patch tokens) when the
checkpoint's modality config has robot0_agentview_left/right + eye_in_hand. Extracts, for a batch
of observation frames, exactly the workspace-encoder inputs:

* ``patch_tokens`` [B, n_img, 2048] — RAW per-frame vision patch tokens from the frozen
  Cosmos-Reason2-2B (Qwen3-VL) backbone (``backbone_features`` at ``image_mask`` positions, tapped
  BEFORE process_backbone_output mutates the BatchFeature). 64 tokens/256x256 view; 3 views -> 192.
* ``lang_emb`` [B, 2048] — masked mean over non-image valid tokens. For WSM we pass the EXPANDED
  prompt (Qwen subtask decomposition) as the text, so these vision-language features are conditioned
  on the nuanced task (per doc 07).
* ``state_emb`` [B, 1, 1536] — GR00T's own CategorySpecificMLP state encoder.

Runs under torch.inference_mode with the processor in eval mode (CenterCrop) so cached features are
deterministic. Backbone is frozen (tune_llm=tune_visual=false).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


@dataclass
class TapResult:
    """Per-batch frozen-backbone features (all torch tensors on the model device)."""

    patch_tokens: torch.Tensor  # [B, n_img, 2048] bf16 — raw backbone features at image positions
    lang_emb: torch.Tensor  # [B, 2048] bf16 — masked mean over text positions
    state_emb: torch.Tensor  # [B, 1, 1536] bf16 — GR00T state-encoder embedding
    backbone_features: torch.Tensor  # [B, seq, 2048] — full sequence (raw, pre-vlln)
    image_mask: torch.Tensor  # [B, seq] bool
    text_mask: torch.Tensor  # [B, seq] bool — non-image, non-padding
    n_img_per_sample: torch.Tensor  # [B] — image-token count (== n_img for fixed V,T)

    def shapes(self) -> dict[str, tuple]:
        return {k: tuple(getattr(self, k).shape) for k in self.__dataclass_fields__}


class BackboneTap:
    """Run the Gr00tPolicy preprocessing + frozen backbone, stopping before the DiT."""

    def __init__(self, policy: Gr00tPolicy, model_path: str | None = None):
        self.policy = policy
        self.model_path = model_path  # checkpoint dir (for statistics.json state dims)
        self.model = policy.model  # Gr00tN1d7
        if hasattr(policy.processor, "eval"):  # eval transforms (CenterCrop): deterministic cache
            policy.processor.eval()
        self.model.eval()

    def obs_from_frames(
        self,
        images: dict[str, np.ndarray],
        state: dict[str, np.ndarray] | np.ndarray,
        text: str | list[str],
    ) -> dict[str, Any]:
        """Build a batched policy-style observation from raw frames.

        images: video_key -> uint8 [B,H,W,C]; state: keyed dict or flat [B,D] (split by ckpt state
        dims); text: the EXPANDED prompt (one string or per-sample list).
        """
        video_keys = self.policy.modality_configs["video"].modality_keys
        state_keys = self.policy.modality_configs["state"].modality_keys
        b = next(iter(images.values())).shape[0]

        video = {k: np.asarray(images[k], dtype=np.uint8)[:, None] for k in video_keys}  # [B,1,H,W,C]

        if isinstance(state, np.ndarray):
            dims = self._state_dims()
            split, off = {}, 0
            for k in state_keys:
                split[k] = state[:, off : off + dims[k]].astype(np.float32)
                off += dims[k]
            state = split
        state_b = {k: np.asarray(v, dtype=np.float32)[:, None] for k, v in state.items()}  # [B,1,D]

        texts = [text] * b if isinstance(text, str) else list(text)
        lang = {self.policy.language_key: np.asarray(texts, dtype=object)[:, None]}  # [B,1]
        return {"video": video, "state": state_b, "language": lang}

    def _state_dims(self) -> dict[str, int]:
        import json
        import os

        if self.model_path is None:
            raise ValueError("flat-state splitting needs model_path (statistics.json); pass a keyed state dict")
        stats_path = os.path.join(str(self.model_path), "statistics.json")
        if not os.path.exists(stats_path):  # intermediate checkpoint-N dirs keep it under processor/
            stats_path = os.path.join(str(self.model_path), "processor", "statistics.json")
        tag = self.policy.embodiment_tag.value
        with open(stats_path) as f:
            stats = json.load(f)[tag]["state"]
        return {k: len(v["mean"]) for k, v in stats.items()}

    @torch.inference_mode()
    def tap(self, observation: dict[str, Any]) -> TapResult:
        """Extract frozen-backbone features for a batched observation."""
        policy, model = self.policy, self.model

        processed = []
        for obs in policy._unbatch_observation(observation):
            vla = policy._to_vla_step_data(obs)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
            processed.append(policy.processor(messages))
        collated = policy.collate_fn(processed)
        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
        batch = collated["inputs"]

        backbone_inputs, action_inputs = model.prepare_input(batch)
        backbone_out = model.backbone(backbone_inputs)

        feats = backbone_out["backbone_features"]  # [B, seq, 2048] — RAW (pre-vlln)
        attn = backbone_out["backbone_attention_mask"].bool()
        img_mask = backbone_out["image_mask"].bool()
        text_mask = (~img_mask) & attn

        n_img = img_mask.sum(dim=1)
        if not (n_img == n_img[0]).all():
            raise ValueError(f"ragged image-token counts across batch: {n_img.tolist()}")
        b, d = feats.shape[0], feats.shape[-1]
        patch_tokens = feats[img_mask].reshape(b, int(n_img[0]), d)

        denom = text_mask.sum(dim=1, keepdim=True).clamp(min=1)
        lang_emb = (feats * text_mask.unsqueeze(-1)).sum(dim=1) / denom

        state = action_inputs["state"]
        state = state.view(state.shape[0], 1, -1)
        state_emb = model.action_head.state_encoder(state, action_inputs["embodiment_id"])

        return TapResult(
            patch_tokens=patch_tokens,
            lang_emb=lang_emb,
            state_emb=state_emb,
            backbone_features=feats,
            image_mask=img_mask,
            text_mask=text_mask,
            n_img_per_sample=n_img,
        )


def load_tap(model_path: str, embodiment_tag: str = "new_embodiment", device: str = "cuda:0") -> BackboneTap:
    """Load a (pretrained/finetuned) GR00T checkpoint and wrap it in a BackboneTap."""
    policy = Gr00tPolicy(model_path=model_path, embodiment_tag=embodiment_tag, device=device, strict=False)
    return BackboneTap(policy, model_path=model_path)
