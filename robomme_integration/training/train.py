#!/usr/bin/env python3
"""Run one isolated RoboMME pi0.5 arm without modifying shared RoboCasa code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _openpi_train_module():
    import openpi

    root = Path(openpi.__file__).resolve().parents[2]
    path = root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train_robomme", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["openpi_train_robomme"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from .arms import NOFORCE_ARM_IDS, OFFICIAL_RECIPE_LEROBOT_ARM, OFFICIAL_RECIPE_LEROBOT_LABEL
    from .config import TRAINING_ARM_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=TRAINING_ARM_IDS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from .arms import V4_ARM_IDS
    from .config import SEQUENCE_ARMS, WORKSPACE_ARMS, build_train_config, validate_train_config
    from .data import (
        install_data_loader_patch,
        install_official_recipe_two_view_patch,
        validate_official_recipe_two_view_patch,
    )

    install_data_loader_patch()
    if args.arm == OFFICIAL_RECIPE_LEROBOT_ARM:
        # Stock project OpenPI requires its three default camera keys during preprocessing.  The
        # official RoboMME adapter uses only front + left wrist, so isolate that exact contract to
        # this diagnostic process; every project arm keeps the historical masked third camera.
        install_official_recipe_two_view_patch()
    if args.arm == "q2v":
        # Must be installed before TrainConfig creates/restores the Pi model.
        from .demo_robottt import install_demo_robottt_patch

        install_demo_robottt_patch()
    config = build_train_config(args.arm)
    validate_train_config(config, args.arm)
    if args.arm in V4_ARM_IDS:
        from .differential_lr import install_parameter_group_optimizer

        optimizer_receipt = install_parameter_group_optimizer(args.arm)
        print(
            "[robomme] v4_optimizer=" + json.dumps(optimizer_receipt, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    if args.arm == OFFICIAL_RECIPE_LEROBOT_ARM:
        validate_official_recipe_two_view_patch()
        print(
            f"[robomme] diagnostic_label={OFFICIAL_RECIPE_LEROBOT_LABEL} "
            "claim=RECIPE_MATCHED_NOT_EXACT_SOURCE_OR_DATA_REPRODUCTION",
            flush=True,
        )
    manifest_identity = "all16"
    if config.data.task_name is not None:
        from .single_task import task_manifest_sha256

        manifest_identity = task_manifest_sha256(config.data.task_name)
    print(
        f"[robomme] arm={args.arm} task={config.data.task_name or 'all16'} "
        f"task_manifest_sha256={manifest_identity} "
        f"steps={config.num_train_steps} batch={config.batch_size} "
        f"workers={config.num_workers} robottt={config.model.robottt} "
        f"wsm_cfg2={config.model.wsm_cfg2} "
        f"wsm_tanh={config.model.wsm_tanh} wsm_jepa={config.model.wsm_jepa} "
        f"L={config.data.stage_q_window_len} stride={config.data.stage_q_chunk_stride} "
        f"demo_frames={config.data.stage_q_demo_frames} "
        f"flow_tau_topology={'shared_within_window' if args.arm in NOFORCE_ARM_IDS else 'stock_independent'} "
        "flow_epsilon_topology=stock_independent_per_chunk",
        flush=True,
    )
    if args.dry_run:
        return
    if args.arm in NOFORCE_ARM_IDS:
        # This imports Pi0 and therefore verifies the interpreter is using the staged, guarded
        # OpenPI overlay before we reserve/initialize accelerators or restore a checkpoint.
        from .sequence_forcing import validate_loaded_overlay

        validate_loaded_overlay()
    if args.arm in SEQUENCE_ARMS and os.environ.get("WSM_STAGE_Q_ALLOW_RUN") != "1":
        raise SystemExit("Q arms require WSM_STAGE_Q_ALLOW_RUN=1 after the one-step canary")
    if args.arm in WORKSPACE_ARMS and os.environ.get("WSM_WSM_POLICY_ALLOW_RUN") != "1":
        raise SystemExit(f"{args.arm} requires WSM_WSM_POLICY_ALLOW_RUN=1")

    import jax

    expected = int(os.environ.get("WSM_EXPECTED_JAX_DEVICES", "8"))
    if jax.device_count() != expected or jax.process_count() != 1:
        raise SystemExit(f"unexpected JAX topology devices={jax.device_count()} processes={jax.process_count()}")
    train_module = _openpi_train_module()
    if args.arm in NOFORCE_ARM_IDS:
        from .sequence_forcing import build_noforce_stage_q_train_step

        train_module.stage_q_train_step = build_noforce_stage_q_train_step(train_module)
    if args.arm in {"q0v", "q2v"}:
        from .demo_robottt import demo_stage_q_train_step

        train_module.stage_q_train_step = demo_stage_q_train_step
    train_module.main(config)


if __name__ == "__main__":
    main()
