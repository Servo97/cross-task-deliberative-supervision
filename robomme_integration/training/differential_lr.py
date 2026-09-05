"""Fail-closed Optax parameter groups for the immutable RoboMME v4 protocol.

OpenPI's pinned ``AdamW`` config accepts one schedule.  The v4 experiments require one global
gradient clip followed by AdamW at 5e-5 for the pretrained policy and 3e-4 for newly initialized
mechanism subtrees.  Applying ``optax.multi_transform`` outside OpenPI's stock optimizer would clip
each group separately, which is not the approved recipe.  This patch therefore reconstructs the
reviewed AdamW chain exactly as::

    global_clip(10) -> multi_transform(backbone_adamw, new_module_adamw)

It refuses unknown optimizer/schedule values, weight-decay masks, missing named subtrees, and a
second installation.  Legacy arms never call this module.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .arms import V4_ARM_IDS, V4_NEW_PARAMETER_SUBTREES

BACKBONE_PEAK_LR = 5e-5
BACKBONE_DECAY_LR = 5e-6
NEW_MODULE_PEAK_LR = 3e-4
NEW_MODULE_DECAY_LR = 3e-5
WEIGHT_DECAY = 1e-6
GLOBAL_CLIP_NORM = 10.0


def _path_head(path: Sequence[Any]) -> str:
    if not path:
        raise ValueError("RoboMME v4 optimizer encountered an empty parameter path")
    value = path[0]
    return str(getattr(value, "key", getattr(value, "name", value)))


def install_parameter_group_optimizer(arm: str, optimizer_module: Any | None = None) -> dict:
    """Install the exact v4 two-rate optimizer for ``arm`` and return its sealed recipe receipt."""

    if arm not in V4_ARM_IDS:
        raise ValueError(f"differential LR is defined only for RoboMME v4 arms, got {arm!r}")
    new_subtrees = tuple(V4_NEW_PARAMETER_SUBTREES[arm])
    if not new_subtrees:
        return recipe_receipt(arm)

    import jax
    import optax

    if optimizer_module is None:
        import openpi.training.optimizer as optimizer_module

    if getattr(optimizer_module, "_ROBOMME_V4_DIFFERENTIAL_LR", False):
        raise RuntimeError("RoboMME v4 differential-LR optimizer was installed more than once")

    def create_optimizer(optimizer, lr_schedule, weight_decay_mask=None):
        if weight_decay_mask is not None:
            raise ValueError("RoboMME v4 differential LR has not certified a weight-decay mask")
        actual_optimizer = (
            optimizer.b1,
            optimizer.b2,
            optimizer.eps,
            optimizer.weight_decay,
            optimizer.clip_gradient_norm,
        )
        if actual_optimizer != (0.9, 0.95, 1e-8, WEIGHT_DECAY, GLOBAL_CLIP_NORM):
            raise ValueError(f"RoboMME v4 optimizer config drifted: {actual_optimizer}")
        actual_schedule = (
            lr_schedule.warmup_steps,
            lr_schedule.peak_lr,
            lr_schedule.decay_steps,
            lr_schedule.decay_lr,
        )
        if actual_schedule[1] != BACKBONE_PEAK_LR or actual_schedule[3] != BACKBONE_DECAY_LR:
            raise ValueError(f"RoboMME v4 backbone schedule drifted: {actual_schedule}")
        warmup_steps, _, decay_steps, _ = actual_schedule
        expected_steps = int(os.environ.get("WSM_MAX_STEPS", str(decay_steps)))
        if (warmup_steps, decay_steps) != (max(1, expected_steps // 20), expected_steps):
            raise ValueError(f"RoboMME v4 warmup/decay geometry drifted: {actual_schedule}")
        backbone_schedule = lr_schedule.create()
        new_module_schedule = optax.warmup_cosine_decay_schedule(
            init_value=NEW_MODULE_PEAK_LR / (warmup_steps + 1),
            peak_value=NEW_MODULE_PEAK_LR,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=NEW_MODULE_DECAY_LR,
        )
        adamw_kwargs = {
            "b1": optimizer.b1,
            "b2": optimizer.b2,
            "eps": optimizer.eps,
            "weight_decay": optimizer.weight_decay,
        }
        transforms = {
            "backbone": optax.adamw(backbone_schedule, **adamw_kwargs),
            "new_module": optax.adamw(new_module_schedule, **adamw_kwargs),
        }

        def labels(params):
            top_levels: set[str] = set()

            def label(path, _leaf):
                head = _path_head(path)
                top_levels.add(head)
                return "new_module" if head in new_subtrees else "backbone"

            result = jax.tree_util.tree_map_with_path(label, params)
            missing = sorted(set(new_subtrees) - top_levels)
            if missing:
                raise ValueError(
                    f"RoboMME v4 {arm} optimizer could not find new parameter subtrees {missing}; "
                    f"observed top levels={sorted(top_levels)}"
                )
            return result

        grouped = optax.multi_transform(transforms, labels)
        return optax.chain(optax.clip_by_global_norm(GLOBAL_CLIP_NORM), grouped)

    optimizer_module.create_optimizer = create_optimizer
    optimizer_module._ROBOMME_V4_DIFFERENTIAL_LR = True
    optimizer_module._ROBOMME_V4_DIFFERENTIAL_LR_RECEIPT = recipe_receipt(arm)
    return recipe_receipt(arm)


def recipe_receipt(arm: str) -> dict:
    if arm not in V4_ARM_IDS:
        raise ValueError(f"unknown RoboMME v4 arm {arm!r}")
    return {
        "schema_version": 1,
        "protocol": "robomme_v4",
        "arm": arm,
        "global_clip_before_parameter_groups": GLOBAL_CLIP_NORM,
        "weight_decay": WEIGHT_DECAY,
        "backbone_lr": {"peak": BACKBONE_PEAK_LR, "decay": BACKBONE_DECAY_LR},
        "new_module_lr": {"peak": NEW_MODULE_PEAK_LR, "decay": NEW_MODULE_DECAY_LR},
        "new_parameter_subtrees": list(V4_NEW_PARAMETER_SUBTREES[arm]),
        "new_parameter_group_active": bool(V4_NEW_PARAMETER_SUBTREES[arm]),
    }
