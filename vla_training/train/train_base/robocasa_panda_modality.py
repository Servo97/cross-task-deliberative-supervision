# RoboCasa365 PandaOmron modality config for GR00T N1.7 fine-tuning under
# EmbodimentTag.NEW_EMBODIMENT. Imported (which registers it) by the GR00T adapters BEFORE
# config.validate(). Ported verbatim from the proven reference
# (internal_planning_and_todos/ported_raw/reference_code/robocasa_panda_modality.py).
#
# Mirrors the proven GR00T N1.5 PandaOmron recipe (43% atomic_seen) and the WORKING N1.7
# DexJoCo sibling config:
#   - 16-step action horizon (action_indices=range(16))
#   - explicit ABSOLUTE / NON_EEF / DEFAULT action_configs per key. RoboCasa actions are already
#     per-step OSC deltas, so they are treated as-is (no relative conversion) and min/max-normalized.
#
#   state (16) = base_position[3] + base_rotation[4] + eef_position_relative[3]
#                + eef_rotation_relative[4] + gripper_qpos[2]
#   action (12) = base_motion[4] + control_mode[1] + eef_position[3] + eef_rotation[3] + gripper_close[1]
# Cameras: robot0_agentview_left / robot0_agentview_right / robot0_eye_in_hand.
# Language: annotation.human.task_description.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# All RoboCasa action channels are absolute targets to the env (OSC deltas / velocity / binary),
# normalized as-is — no EEF-relative conversion. One config per modality key.
_abs = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)

robocasa365_panda_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_position",
            "base_rotation",
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "base_motion",
            "control_mode",
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
        ],
        action_configs=[_abs, _abs, _abs, _abs, _abs],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(robocasa365_panda_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
