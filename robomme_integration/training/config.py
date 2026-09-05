"""Canonical isolated pi0.5 configs for RoboMME training arms."""

from __future__ import annotations

import ast
import inspect
import os
import textwrap

from .arms import (
    NOFORCE_ARM_IDS,
    OFFICIAL_RECIPE_LEROBOT_ARM,
    OFFICIAL_RECIPE_LEROBOT_LABEL,
    OFFICIAL_RECIPE_LEROBOT_STEPS,
    ROBOTT_ARMS,
    SEQUENCE_ARMS,
    TRAINING_ARM_IDS,
    V4_ARM_IDS,
    WORKSPACE_ARMS,
    Arm,
)
from .data import RoboMMEDataConfigFactory
from .single_task import TASK_EPISODES

ROBOMME_REPO_ID = "Yinpei/robomme_data_lerobot"
ROBOMME_DATASET_REVISION = "1510653cccb4d9e5165fb3141c06d88053decc20"
DEMO_FRAMES = 16
EXECUTION_WINDOW = 8
CHUNK_STRIDE = 10
DELTANET_RECIPES = {
    "wsm_d8": (8, 0.0),
    "wsm_d8_drop05": (8, 0.5),
    "wsm_d16": (16, 0.0),
    "wsm_d16_drop05": (16, 0.5),
}
GDN8_JEPA_ARM = "gdn8_jepa_l01_k1"
V4_GDN8_JEPA_ARM = "v4_gdn8_jepa_visreg_l01_k1"
V4_CFG_JEPA_ARM = "v4_cfg_jepa_visreg_l01_k1"
V4_JEPA_ARMS = frozenset({"v4_jepa_visreg_l01_k1", V4_GDN8_JEPA_ARM, V4_CFG_JEPA_ARM})
V4_DELTANET_RECIPES = {
    "v4_wsm_gdn8_drop00": (8, 0.0),
    "v4_wsm_gdn8_drop02": (8, 0.2),
    "v4_wsm_gdn16_drop00": (16, 0.0),
    "v4_wsm_gdn16_drop02": (16, 0.2),
}


def _required_local_path(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    if value.startswith(("s3://", "gs://", "http://", "https://")):
        raise ValueError(f"{name} must be an inventory-verified node-local path")
    return value


def _assert_current_pi05_does_not_embed_proprio() -> None:
    """Fail closed if the pinned modified Pi0 stops guarding state tokens behind ``not pi05``."""
    import openpi.models.pi0 as pi0_model

    tree = ast.parse(textwrap.dedent(inspect.getsource(pi0_model.Pi0.embed_suffix)))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def is_pi05_negative_guard(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Attribute)
            and isinstance(node.operand.value, ast.Name)
            and node.operand.value.id == "self"
            and node.operand.attr == "pi05"
        )

    state_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "obs"
        and node.attr == "state"
    ]
    if not state_reads:
        raise ValueError("unreviewed Pi0 suffix implementation: expected guarded obs.state reads")
    for read in state_reads:
        current = parents.get(read)
        while current is not None and not (isinstance(current, ast.If) and is_pi05_negative_guard(current.test)):
            current = parents.get(current)
        if current is None:
            raise ValueError("current Pi0 may embed proprioception for pi0.5; refusing official recipe approximation")


def build_train_config(arm: Arm):
    if arm not in TRAINING_ARM_IDS:
        raise ValueError(f"unknown RoboMME arm {arm!r}")
    import flax.nnx as nnx
    import openpi.models.pi0_config as pi0_config
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders

    official_recipe_lerobot = arm == OFFICIAL_RECIPE_LEROBOT_ARM
    v4 = arm in V4_ARM_IDS
    sequence = arm in SEQUENCE_ARMS
    demo_context = arm in {"q0v", "q2v"}
    robottt = arm in ROBOTT_ARMS
    workspace_cfg = arm in {"wsm_cfg", "v4_wsm_cfg", V4_CFG_JEPA_ARM}
    workspace_tanh = arm in {"q1", "q3", "wsm_tanh", "v4_wsm_tanh"}
    gdn8_jepa = arm in {GDN8_JEPA_ARM, V4_GDN8_JEPA_ARM}
    workspace_deltanet = arm in DELTANET_RECIPES or arm in V4_DELTANET_RECIPES or gdn8_jepa
    deltanet_window, history_dropout = (
        (8, 0.0) if gdn8_jepa else V4_DELTANET_RECIPES.get(arm, DELTANET_RECIPES.get(arm, (1, 0.0)))
    )
    workspace_ptrm = arm in {"ptrm", "v4_ptrm"}
    jepa_recipes = {
        "jepa_l01_k1": (0.1, 1),
        "jepa_l1_k32": (1.0, 32),
        "jepa_l01_k16": (0.1, 16),
        GDN8_JEPA_ARM: (0.1, 1),
        "v4_jepa_visreg_l01_k1": (0.1, 1),
        V4_GDN8_JEPA_ARM: (0.1, 1),
        V4_CFG_JEPA_ARM: (0.1, 1),
    }
    supervision_arm = arm in {"salient", "causal_v1"}
    workspace_jepa = arm in jepa_recipes or supervision_arm
    workspace_arm = arm in WORKSPACE_ARMS
    data_root = _required_local_path("ROBOMME_DATA_ROOT")
    assets_root = _required_local_path("ROBOMME_ASSETS_ROOT")
    init_params = _required_local_path("WSM_INIT_FROM")
    steps = int(os.environ.get("WSM_MAX_STEPS", "60000"))
    warmup_steps = int(os.environ.get("WSM_WARMUP_STEPS", str(max(1, steps // 20)) if v4 else "1000"))
    peak_lr = float(os.environ.get("WSM_PEAK_LR", "5e-5" if v4 else "2.5e-5"))
    decay_steps = int(os.environ.get("WSM_DECAY_STEPS", str(steps)))
    decay_lr = float(os.environ.get("WSM_DECAY_LR", "5e-6" if v4 else "2.5e-6"))
    seed = int(os.environ.get("WSM_SEED", "42"))
    if steps < 1 or warmup_steps < 0 or decay_steps < 1:
        raise ValueError("WSM_MAX_STEPS/WSM_DECAY_STEPS must be positive and WSM_WARMUP_STEPS nonnegative")
    if warmup_steps >= decay_steps:
        raise ValueError("WSM_WARMUP_STEPS must be smaller than WSM_DECAY_STEPS")
    if not (peak_lr > 0.0 and decay_lr >= 0.0):
        raise ValueError("WSM_PEAK_LR must be positive and WSM_DECAY_LR nonnegative")
    if seed < 0:
        raise ValueError("WSM_SEED must be nonnegative")
    if v4 and (peak_lr, decay_lr) != (5e-5, 5e-6):
        raise ValueError(
            "RoboMME v4 requires the sealed backbone schedule 5e-5 -> 5e-6; "
            "new-module 3e-4 -> 3e-5 is installed as an exact parameter group"
        )
    if official_recipe_lerobot:
        expected_environment = {
            "ROBOMME_RECIPE_LABEL": OFFICIAL_RECIPE_LEROBOT_LABEL,
            "WSM_MAX_STEPS": str(OFFICIAL_RECIPE_LEROBOT_STEPS),
            "WSM_WARMUP_STEPS": "10000",
            "WSM_PEAK_LR": "5e-5",
            "WSM_DECAY_STEPS": "100000",
            "WSM_DECAY_LR": "5e-5",
            "WSM_SEED": "42",
            "WSM_SAVE_INTERVAL": "10000",
        }
        drift = {
            name: (os.environ.get(name), expected)
            for name, expected in expected_environment.items()
            if os.environ.get(name) != expected
        }
        if drift:
            raise ValueError(f"official_recipe_lerobot environment is incomplete or drifted: {drift}")
        if os.environ.get("WSM_KEEP_PERIOD", "").strip():
            raise ValueError(
                "official_recipe_lerobot forbids periodic local retention; milestones are sealed remotely"
            )
        _assert_current_pi05_does_not_embed_proprio()
    task_name = os.environ.get("ROBOMME_TASK") or None
    if task_name is not None and task_name not in TASK_EPISODES:
        raise ValueError(f"unknown ROBOMME_TASK {task_name!r}; expected one of {tuple(TASK_EPISODES)}")
    default_exp_name = f"pi05_robomme_{task_name}_{arm}" if task_name else f"pi05_robomme_{arm}"
    workspace_root = os.environ.get("ROBOMME_WORKSPACE_ROOT") if workspace_arm else None
    workspace_index = os.environ.get("ROBOMME_WORKSPACE_INDEX") if workspace_arm else None
    if workspace_arm:
        if not workspace_root:
            raise ValueError(f"{arm} requires ROBOMME_WORKSPACE_ROOT")
        if workspace_root.startswith(("s3://", "gs://", "http://", "https://")):
            raise ValueError("ROBOMME_WORKSPACE_ROOT must be a node-local verified cache")
        if task_name is None:
            if not workspace_index:
                raise ValueError(f"multitask {arm} requires ROBOMME_WORKSPACE_INDEX")
            if workspace_index.startswith(("s3://", "gs://", "http://", "https://")):
                raise ValueError("ROBOMME_WORKSPACE_INDEX must be a node-local verified file")
        elif workspace_index:
            raise ValueError("single-task workspace training forbids an all-16 workspace index")
    salient_root = os.environ.get("ROBOMME_SALIENT_ROOT") if supervision_arm else None
    if supervision_arm and not salient_root:
        raise ValueError(f"{arm} requires ROBOMME_SALIENT_ROOT")
    if salient_root and salient_root.startswith(("s3://", "gs://", "http://", "https://")):
        raise ValueError("ROBOMME_SALIENT_ROOT must be a node-local verified cache")
    jepa_weight, jepa_futures = jepa_recipes.get(arm, (0.0, 1))
    final_only = os.environ.get("WSM_FINAL_ONLY_CHECKPOINTS", "0") == "1"
    save_interval = steps if final_only else int(os.environ.get("WSM_SAVE_INTERVAL", "5000"))
    keep_period_text = os.environ.get("WSM_KEEP_PERIOD", "").strip()
    keep_period = None if final_only or not keep_period_text else int(keep_period_text)
    if keep_period is not None and keep_period < 1:
        raise ValueError("WSM_KEEP_PERIOD must be positive when set")

    model_kwargs = dict(
        pi05=True,
        action_horizon=20,
        max_token_len=64 if official_recipe_lerobot else 200,
        discrete_state_input=False,
        wsm_cfg2=workspace_cfg,
        wsm_cfg_p_drop=float(os.environ.get("WSM_CFG_P_DROP", "0.2")),
        wsm_tanh=workspace_tanh or workspace_deltanet or workspace_ptrm,
        wsm_tanh_gate_init=float(os.environ.get("WSM_TANH_GATE_INIT", "0.001")),
        wsm_cond_type=(
            "gated_deltanet_ptrm" if workspace_ptrm else "gated_deltanet" if workspace_deltanet else "tanh"
        ),
        wsm_cond_window=8 if workspace_ptrm else deltanet_window,
        wsm_cond_num_heads=2,
        wsm_cond_head_dim=256,
        wsm_jepa=workspace_jepa,
        wsm_jepa_weight=jepa_weight,
        wsm_jepa_sigreg_weight=0.0 if supervision_arm else 0.05,
        wsm_jepa_num_futures=jepa_futures,
        wsm_salient=supervision_arm,
        wsm_salient_weight=1.0,
        wsm_salient_num_patches=64,
        wsm_w_dim=512,
        robottt=robottt,
        robottt_token_dim=256,
        robottt_fast_hidden=128,
        robottt_num_registers=16,
        robottt_base_inner_lr=0.1,
        robottt_gate_init=0.001,
        robottt_window_len=(DEMO_FRAMES + EXECUTION_WINDOW if demo_context else EXECUTION_WINDOW),
        # The new visual-context projections start from scratch.  Detaching W at the demo/execution
        # boundary would prevent the execution loss from teaching those projections, so q2v keeps
        # the complete 24-step chain differentiable.  The old q2 recipe remains byte-identical.
        robottt_tbptt_segment=(DEMO_FRAMES + EXECUTION_WINDOW if demo_context else EXECUTION_WINDOW),
    )
    # The campaign's ordinary OpenPI archive predates PTRM/history-dropout kwargs. Keep advanced
    # fields conditional so all existing arms remain source-compatible and scientifically stable.
    if workspace_ptrm:
        model_kwargs.update(wsm_ptrm_steps=4, wsm_ptrm_q_weight=0.1)
    if history_dropout:
        model_kwargs["wsm_cond_history_dropout"] = history_dropout
    if arm in V4_JEPA_ARMS:
        model_kwargs.update(
            wsm_jepa_regularizer="visreg",
            wsm_jepa_sigreg_weight=0.0,
            wsm_jepa_visreg_weight=0.05,
            wsm_jepa_visreg_slices=128,
            wsm_jepa_visreg_scale_weight=1.0,
            wsm_jepa_visreg_shape_weight=1.0,
            wsm_jepa_visreg_center_weight=1.0,
        )
    model = pi0_config.Pi0Config(**model_kwargs)
    return _config.TrainConfig(
        name=f"pi05_robomme_{arm}",
        exp_name=os.environ.get("WSM_EXP_NAME", default_exp_name),
        project_name=os.environ.get("WANDB_PROJECT", "wsm-robomme"),
        model=model,
        data=RoboMMEDataConfigFactory(
            repo_id=ROBOMME_REPO_ID,
            lerobot_root=data_root,
            task_name=task_name,
            execution_only=not demo_context,
            stage_q_window_len=EXECUTION_WINDOW if sequence else 0,
            stage_q_chunk_stride=CHUNK_STRIDE,
            stage_q_demo_frames=DEMO_FRAMES if demo_context else 0,
            stage_q_iid_steps=arm == "a6",
            workspace_root=workspace_root,
            workspace_index=workspace_index,
            workspace_window=8 if workspace_ptrm else deltanet_window,
            workspace_stride=CHUNK_STRIDE,
            workspace_jepa_futures=jepa_futures if workspace_jepa else 0,
            workspace_jepa_with_window=gdn8_jepa,
            salient_root=salient_root,
            official_recipe_lerobot=official_recipe_lerobot,
            assets=_config.AssetsConfig(assets_dir=assets_root, asset_id="robomme"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            init_params,
            missing_regex=(
                ".*(lora|wsm_tanh_cond|wsm_jepa_head).*"
                if gdn8_jepa
                else ".*(lora|wsm_cfg2_cond|wsm_jepa_head).*"
                if arm == V4_CFG_JEPA_ARM
                else ".*(lora|robottt_fast|wsm_tanh_cond).*"
                if arm == "q3"
                else ".*(lora|robottt_fast).*"
                if robottt
                else ".*(lora|wsm_cfg2_cond).*"
                if workspace_cfg
                else ".*(lora|wsm_tanh_cond).*"
                if workspace_tanh or workspace_deltanet or workspace_ptrm
                else ".*(lora|wsm_jepa_head|wsm_salient_head).*"
                if supervision_arm
                else ".*(lora|wsm_jepa_head).*"
                if workspace_jepa
                else ".*lora.*"
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=decay_lr,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-6 if v4 else 1e-10,
            clip_gradient_norm=10.0 if v4 else 1.0,
        ),
        freeze_filter=(nnx_utils.PathRegex(".*img.*") if official_recipe_lerobot else nnx.Nothing()),
        ema_decay=0.999 if official_recipe_lerobot or v4 else 0.99,
        seed=seed,
        resume=os.environ.get("WSM_RESUME") == "1",
        num_train_steps=steps,
        save_interval=save_interval,
        # Orbax already keeps the newest generation. Remote sync is the recovery authority, so
        # retaining every 5k generation locally only multiplies ephemeral disk use.
        keep_period=keep_period,
        batch_size=8 if sequence else 64,
        num_workers=int(os.environ.get("WSM_NUM_WORKERS", "32")),
        fsdp_devices=1,
        checkpoint_base_dir=os.environ.get("WSM_CKPT_BASE", "./checkpoints/robomme"),
        wandb_enabled=os.environ.get("WANDB_MODE", "online") != "disabled",
    )


def validate_train_config(config, arm: Arm) -> None:
    official_recipe_lerobot = arm == OFFICIAL_RECIPE_LEROBOT_ARM
    v4 = arm in V4_ARM_IDS
    sequence = arm in SEQUENCE_ARMS
    demo_context = arm in {"q0v", "q2v"}
    noforce = arm in NOFORCE_ARM_IDS
    if bool(config.model.robottt) != (arm in ROBOTT_ARMS):
        raise ValueError(f"{arm} lost its RoboTTT identity")
    if bool(config.model.wsm_cfg2) != (arm in {"wsm_cfg", "v4_wsm_cfg", V4_CFG_JEPA_ARM}):
        raise ValueError(f"{arm} lost its current-only WSM-CFG identity")
    steering_arms = {
        "q1",
        "q3",
        "wsm_tanh",
        "v4_wsm_tanh",
        *DELTANET_RECIPES,
        *V4_DELTANET_RECIPES,
        GDN8_JEPA_ARM,
        V4_GDN8_JEPA_ARM,
        "ptrm",
        "v4_ptrm",
    }
    if bool(config.model.wsm_tanh) != (arm in steering_arms):
        raise ValueError(f"{arm} lost its WSM tanh/DeltaNet identity")
    if bool(config.model.wsm_jepa) != (
        arm.startswith("jepa_") or arm in {GDN8_JEPA_ARM, "salient", "causal_v1", *V4_JEPA_ARMS}
    ):
        raise ValueError(f"{arm} lost its JEPA-target interface identity")
    if config.data.task_name is not None and config.data.task_name not in TASK_EPISODES:
        raise ValueError(f"unknown RoboMME single-task selection {config.data.task_name!r}")
    if config.model.action_horizon != 20 or config.model.discrete_state_input:
        raise ValueError("RoboMME requires horizon=20 and discrete_state_input=False")
    if v4:
        optimizer = config.optimizer
        if (
            optimizer.b1,
            optimizer.b2,
            optimizer.eps,
            optimizer.weight_decay,
            optimizer.clip_gradient_norm,
        ) != (0.9, 0.95, 1e-8, 1e-6, 10.0):
            raise ValueError("RoboMME v4 AdamW/weight-decay/global-clip recipe drifted")
        if config.ema_decay != 0.999:
            raise ValueError("RoboMME v4 requires EMA 0.999")
        if (config.lr_schedule.peak_lr, config.lr_schedule.decay_lr) != (5e-5, 5e-6):
            raise ValueError("RoboMME v4 requires backbone LR 5e-5 -> 5e-6")
        if (config.lr_schedule.warmup_steps, config.lr_schedule.decay_steps) != (
            max(1, config.num_train_steps // 20),
            config.num_train_steps,
        ):
            raise ValueError("RoboMME v4 requires 5% warmup and decay through the final step")
    if official_recipe_lerobot:
        if config.data.task_name is not None or not config.data.execution_only:
            raise ValueError("official_recipe_lerobot must use execution-only all16 data")
        if config.num_train_steps != 80_000 or config.seed != 42 or config.batch_size != 64:
            raise ValueError("official_recipe_lerobot requires 80k steps, seed 42, batch 64")
        if config.model.max_token_len != 64 or not config.model.pi05:
            raise ValueError("official_recipe_lerobot requires pi0.5 with max_token_len=64")
        schedule = config.lr_schedule
        if (
            schedule.warmup_steps,
            schedule.peak_lr,
            schedule.decay_steps,
            schedule.decay_lr,
        ) != (10_000, 5e-5, 100_000, 5e-5):
            raise ValueError("official_recipe_lerobot requires 10k warmup then constant 5e-5")
        optimizer = config.optimizer
        if (
            optimizer.b1,
            optimizer.b2,
            optimizer.eps,
            optimizer.weight_decay,
            optimizer.clip_gradient_norm,
        ) != (0.9, 0.95, 1e-8, 1e-10, 1.0):
            raise ValueError("official_recipe_lerobot AdamW recipe drifted")
        freeze_pattern = getattr(getattr(config.freeze_filter, "pattern", None), "pattern", None)
        if freeze_pattern != ".*img.*":
            raise ValueError("official_recipe_lerobot must freeze exactly the image/SigLIP subtree")
        if config.ema_decay != 0.999 or config.save_interval != 10_000 or config.keep_period is not None:
            raise ValueError("official_recipe_lerobot EMA/checkpoint recipe drifted")
        if not config.data.official_recipe_lerobot:
            raise ValueError("official_recipe_lerobot lost its exact two-view data adapter")
        if any(
            (
                config.model.robottt,
                config.model.wsm_cfg2,
                config.model.wsm_tanh,
                config.model.wsm_jepa,
                config.data.stage_q_window_len,
                config.data.workspace_root is not None,
            )
        ):
            raise ValueError("official_recipe_lerobot must remain an unconditioned baseline diagnostic")
    if sequence:
        if config.data.stage_q_window_len != EXECUTION_WINDOW or config.data.stage_q_chunk_stride != CHUNK_STRIDE:
            raise ValueError("RoboMME Q arms require L=8, stride=10")
        if config.batch_size * config.data.stage_q_window_len != 64:
            raise ValueError("RoboMME Q arms require effective batch 64")
        if bool(config.data.stage_q_iid_steps) != (arm == "a6"):
            raise ValueError(f"{arm} lost its exact Stage-Q iid-sampling identity")
        if config.data.stage_q_iid_steps and config.model.robottt:
            raise ValueError("A6 iid regrouping is incompatible with a fast-weight chain")
        if noforce and config.data.stage_q_iid_steps:
            raise ValueError("shared-tau controls require contiguous windows, never IID regrouping")
    elif config.batch_size != 64:
        raise ValueError("RoboMME S0 requires global batch 64")
    if demo_context:
        expected_chain = DEMO_FRAMES + EXECUTION_WINDOW
        if config.data.execution_only or config.data.stage_q_demo_frames != DEMO_FRAMES:
            raise ValueError("RoboMME video-context arms require a 16-frame demo prefix")
        if config.model.robottt_window_len != expected_chain or config.model.robottt_tbptt_segment != expected_chain:
            raise ValueError("RoboMME video-context arms require full-BPTT over 24 steps")
    elif config.data.stage_q_demo_frames:
        raise ValueError(f"{arm} unexpectedly enables demo-prefix training")
    if arm in WORKSPACE_ARMS:
        if not config.data.workspace_root:
            raise ValueError(f"{arm} requires a local workspace cache")
        if (config.data.task_name is None) != bool(config.data.workspace_index):
            raise ValueError(f"{arm} requires exactly one routing identity: ROBOMME_TASK or all-16 workspace index")
        if config.model.wsm_w_dim != 512:
            raise ValueError("RoboMME workspace cache contract requires omega_dim=512")
    if (
        arm in DELTANET_RECIPES
        or arm in V4_DELTANET_RECIPES
        or arm
        in {
            GDN8_JEPA_ARM,
            V4_GDN8_JEPA_ARM,
            "ptrm",
            "v4_ptrm",
        }
    ):
        expected_type = "gated_deltanet_ptrm" if arm in {"ptrm", "v4_ptrm"} else "gated_deltanet"
        expected_window, expected_dropout = (
            (8, 0.0)
            if arm in {GDN8_JEPA_ARM, V4_GDN8_JEPA_ARM, "ptrm", "v4_ptrm"}
            else V4_DELTANET_RECIPES.get(arm, DELTANET_RECIPES.get(arm))
        )
        if config.model.wsm_cond_type != expected_type or config.model.wsm_cond_window != expected_window:
            raise ValueError(f"{arm} requires its K={expected_window} gated-DeltaNet conditioner")
        expected_futures = 1 if arm in {GDN8_JEPA_ARM, V4_GDN8_JEPA_ARM} else 0
        if (
            config.data.workspace_window != expected_window
            or config.data.workspace_jepa_futures != expected_futures
            or bool(config.data.workspace_jepa_with_window) != (arm in {GDN8_JEPA_ARM, V4_GDN8_JEPA_ARM})
        ):
            raise ValueError(
                f"{arm} loader must expose its exact K={expected_window} causal omega window "
                f"and JEPA-future count {expected_futures}"
            )
        actual_dropout = float(getattr(config.model, "wsm_cond_history_dropout", 0.0))
        if actual_dropout != expected_dropout:
            raise ValueError(f"{arm} history-dropout drifted: {actual_dropout} != {expected_dropout}")
        if arm in {"ptrm", "v4_ptrm"}:
            if (config.model.wsm_ptrm_steps, config.model.wsm_ptrm_q_weight) != (4, 0.1):
                raise ValueError("PTRM requires T=4 and Q-loss weight 0.1")
            if config.model.wsm_jepa or config.model.wsm_salient:
                raise ValueError("the first RoboMME PTRM arm must remain unconfounded")
    elif (
        arm
        in {
            "q1",
            "q3",
            "wsm_cfg",
            "wsm_tanh",
            "v4_wsm_cfg",
            "v4_wsm_tanh",
            V4_CFG_JEPA_ARM,
        }
        and config.data.workspace_window != 1
    ):
        raise ValueError(f"{arm} requires current-only K=1 workspace input")
    if arm.startswith("jepa_") or arm in {
        GDN8_JEPA_ARM,
        "salient",
        "causal_v1",
        *V4_JEPA_ARMS,
    }:
        expected = {
            "jepa_l01_k1": (0.1, 1),
            "jepa_l1_k32": (1.0, 32),
            "jepa_l01_k16": (0.1, 16),
            GDN8_JEPA_ARM: (0.1, 1),
            "v4_jepa_visreg_l01_k1": (0.1, 1),
            V4_GDN8_JEPA_ARM: (0.1, 1),
            V4_CFG_JEPA_ARM: (0.1, 1),
            "salient": (0.0, 1),
            "causal_v1": (0.0, 1),
        }[arm]
        if (config.model.wsm_jepa_weight, config.model.wsm_jepa_num_futures) != expected:
            raise ValueError(f"{arm} JEPA recipe drifted")
        if config.data.workspace_jepa_futures != expected[1]:
            raise ValueError(f"{arm} loader/head future count mismatch")
    if arm in V4_JEPA_ARMS:
        visreg = (
            config.model.wsm_jepa_regularizer,
            config.model.wsm_jepa_sigreg_weight,
            config.model.wsm_jepa_visreg_weight,
            config.model.wsm_jepa_visreg_slices,
            config.model.wsm_jepa_visreg_scale_weight,
            config.model.wsm_jepa_visreg_shape_weight,
            config.model.wsm_jepa_visreg_center_weight,
        )
        if visreg != ("visreg", 0.0, 0.05, 128, 1.0, 1.0, 1.0):
            raise ValueError(f"{arm} VISReg recipe drifted: {visreg}")
    if arm in {"salient", "causal_v1"}:
        if not config.model.wsm_salient or config.model.wsm_salient_num_patches != 64:
            raise ValueError(f"RoboMME {arm} arm requires a 64-way base-view keypatch head")
        if not config.data.salient_root:
            raise ValueError(f"RoboMME {arm} arm requires task-bound supervision labels")
