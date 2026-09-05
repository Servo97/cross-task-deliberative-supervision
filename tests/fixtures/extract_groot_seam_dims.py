"""Generate `groot_n17_seam_dims.json`: the REAL widths the GR00T collator feeds RoboTTT.

WHY THIS EXISTS. `tests/test_groot_robottt_wiring.py` used to synthesise its tensors at
`RoboTTTConfig`'s defaults, so its "production dims" were whatever the config said rather than
whatever the data was. On 2026-08-08 that let the TTT canary reach the node with a 64-wide
`state_in` against a 132-wide state (`mat1 and mat2 shapes cannot be multiplied (2x132 and
64x256)`). The wiring test now takes its dims from this fixture and derives them with the
PRODUCTION function `robottt_seam_dims`, so a test that passes at the wrong width is no longer
possible without this fixture being wrong — and this script is what makes it right, by measuring.

WHAT IS MEASURED. The real `Gr00tN1d7Processor` (built exactly as `setup.py` builds it, from a real
`Gr00tN1d7Config` and wsmv2's own `robocasa_panda_modality`) is run on a synthetic-but-correctly-
shaped RoboCasa step; the resulting per-step feature dicts go through wsmv2's window builder and
`build_sequence_collator` wrapped around the REAL `Gr00tN1d7DataCollator`; the collated batch then
goes through `flatten_window_batch` and the action head's own `state.reshape(B, L, -1)`. Only the
PIXELS and the numeric values are synthetic — every shape on the path is produced by the shipped
GR00T code. The script asserts `robottt_seam_dims(processor, model_config)` reproduces the measured
shapes, so running it IS the live cross-check.

RUN (gr00t venv + the local Cosmos-Reason2-2B processor cache; no HF token, no network):

    env -u HF_TOKEN ~/Research/Isaac-GR00T/.venv/bin/python \
        tests/fixtures/extract_groot_seam_dims.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import VLAStepData
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

from vla_training.train.train_base._groot_robottt_common import (
    flatten_window_batch,
    robottt_seam_dims,
)
from vla_training.train.train_base._groot_seq_common import build_sequence_collator
from vla_training.train.train_base.robocasa_panda_modality import robocasa365_panda_config

OUT = Path(__file__).with_name("groot_n17_seam_dims.json")
TAG = EmbodimentTag.NEW_EMBODIMENT
B, L, IMG = 2, 3, 64

#: Raw per-key widths of the RoboCasa PandaOmron modality (robocasa_panda_modality.py header):
#: state 16 channels, action 12 channels. Both are PADDED by the processor, which is the whole point.
STATE_DIMS = {
    "base_position": 3,
    "base_rotation": 4,
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
}
ACTION_DIMS = {
    "base_motion": 4,
    "control_mode": 1,
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
}


def _stats(dims: dict[str, int]) -> dict:
    """Unit statistics. Normalization VALUES are irrelevant here; only shapes are measured."""
    return {
        k: {
            "min": [-1.0] * d,
            "max": [1.0] * d,
            "mean": [0.0] * d,
            "std": [1.0] * d,
            "q01": [-0.9] * d,
            "q99": [0.9] * d,
        }
        for k, d in dims.items()
    }


def build_processor(mc: Gr00tN1d7Config) -> Gr00tN1d7Processor:
    """Mirrors `Gr00tN1d7Pipeline._create_processor_and_datasets` (setup.py:182-208) argument for
    argument, including `max_action_horizon=mc.action_horizon` — the line that makes H 40 and not
    the modality config's 16."""
    return Gr00tN1d7Processor(
        modality_configs={TAG.value: robocasa365_panda_config},
        statistics={TAG.value: {"state": _stats(STATE_DIMS), "action": _stats(ACTION_DIMS)}},
        use_percentiles=mc.use_percentiles,
        image_crop_size=mc.image_crop_size,
        image_target_size=mc.image_target_size,
        random_rotation_angle=mc.random_rotation_angle,
        color_jitter_params={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08},
        model_name=mc.model_name,
        model_type=mc.backbone_model_type,
        formalize_language=mc.formalize_language,
        max_state_dim=mc.max_state_dim,
        max_action_dim=mc.max_action_dim,
        max_action_horizon=mc.action_horizon,
        apply_sincos_state_encoding=mc.apply_sincos_state_encoding,
        use_albumentations=mc.use_albumentations_transforms,
        extra_augmentation_config=mc.extra_augmentation_config,
        shortest_image_edge=mc.shortest_image_edge,
        crop_fraction=mc.crop_fraction,
        use_relative_action=True,  # _groot_common.py sets this on every groot arm
        exclude_state=mc.exclude_state,
        state_dropout_prob=0.2,  # the ttt yamls' model.state_dropout_prob
        use_mean_std=mc.use_mean_std,
        transformers_loading_kwargs={"local_files_only": True},
    )


def _window(per_step: list[dict]) -> dict:
    """Byte-for-byte the stacking `_groot_seq_common.WindowShardedSequenceDataset._build_window` does."""
    window = {}
    for key in set(per_step[0]):
        vals = [d[key] for d in per_step]
        if key == "vlm_content":
            window[key] = vals
        elif isinstance(vals[0], (int, float, bool)):
            window[key] = np.asarray(vals)
        else:
            window[key] = np.stack([np.asarray(v) for v in vals])
    window["seq_len"] = np.asarray(len(per_step), dtype=np.int64)
    window["loss_mask"] = np.ones(len(per_step), dtype=np.float32)
    window["reset"] = np.asarray([i == 0 for i in range(len(per_step))], dtype=np.bool_)
    return window


def main() -> int:
    mc = Gr00tN1d7Config()
    processor = build_processor(mc)
    views = robocasa365_panda_config["video"].modality_keys
    state_t = len(robocasa365_panda_config["state"].delta_indices)
    action_t = len(robocasa365_panda_config["action"].delta_indices)
    rng = np.random.default_rng(0)

    def step(text: str) -> dict:
        data = VLAStepData(
            images={v: [rng.integers(0, 255, (IMG, IMG, 3), dtype=np.uint8)] for v in views},
            states={k: rng.standard_normal((state_t, d)).astype(np.float32) for k, d in STATE_DIMS.items()},
            actions={k: rng.standard_normal((action_t, d)).astype(np.float32) for k, d in ACTION_DIMS.items()},
            text=text,
            embodiment=TAG,
        )
        return processor([{"type": "episode_step", "content": data}])

    features = [_window([step(f"window {b} step {t}") for t in range(L)]) for b in range(B)]
    one = features[0]
    batch = build_sequence_collator(processor.collator)(features)["inputs"]

    length = int(batch["seq_window_len"])
    flat, bsz, length = flatten_window_batch(batch, length)
    state_seq = batch["state"].reshape(bsz, length, -1).to(torch.float32)
    action_seq = batch["action"].to(torch.float32)

    # THE CROSS-CHECK: the production derivation must reproduce the measured tensors exactly.
    derived = robottt_seam_dims(processor, mc)
    assert derived["state_dim"] == int(state_seq.shape[-1]), (derived, state_seq.shape)
    assert derived["action_dim"] == int(action_seq.shape[-1]), (derived, action_seq.shape)
    assert derived["action_horizon"] == int(action_seq.shape[-2]), (derived, action_seq.shape)
    assert derived["state_history_length"] == int(batch["state"].shape[2]), batch["state"].shape

    fixture = {
        "note": "MEASURED from the real Gr00tN1d7Processor + Gr00tN1d7DataCollator, offline. "
        "Regenerate with tests/fixtures/extract_groot_seam_dims.py in the gr00t venv.",
        # The two objects `robottt_seam_dims` reads. The wiring test replays THESE through the
        # production derivation, so its dims are the production dims by construction.
        "processor": {
            "max_state_dim": int(processor.max_state_dim),
            "max_action_dim": int(processor.max_action_dim),
            "max_action_horizon": int(processor.max_action_horizon),
        },
        "model_config": {
            "max_state_dim": int(mc.max_state_dim),
            "max_action_dim": int(mc.max_action_dim),
            "action_horizon": int(mc.action_horizon),
            "state_history_length": int(mc.state_history_length),
            "input_embedding_dim": int(mc.input_embedding_dim),
        },
        "raw_modality": {
            "state_channels": sum(STATE_DIMS.values()),
            "state_timesteps": state_t,
            "action_channels": sum(ACTION_DIMS.values()),
            "action_timesteps": action_t,
        },
        "per_step": {
            "state": list(one["state"].shape[1:]),
            "action": list(one["action"].shape[1:]),
            "dtype": str(batch["state"].dtype),
        },
        "collated": {
            "state": list(batch["state"].shape),
            "action": list(batch["action"].shape),
            "B": int(bsz),
            "L": int(length),
        },
        "seam": {
            "state_seq": list(state_seq.shape),
            "action_seq": list(action_seq.shape),
            "state_dim": int(state_seq.shape[-1]),
            "action_dim": int(action_seq.shape[-1]),
            "action_horizon": int(action_seq.shape[-2]),
            "dtype": str(state_seq.dtype),
        },
        "flat": {"state": list(flat["state"].shape), "action": list(flat["action"].shape)},
        "cond_dim": int(mc.input_embedding_dim),
    }
    OUT.write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps(fixture, indent=2))
    print(f"\nwrote {OUT}")
    print(f"derivation OK: {derived['dims_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
