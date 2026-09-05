#!/usr/bin/env python3
"""Stage-Q pi0.5 sequence finetune — the SINGLE entry for the RoboTTT 2x2 (Q0/Q1/Q2/Q3).

One entry, four arms. The arm is chosen by exactly two flags:

    --fast-weights {off,on}   RoboTTT online update (Pi0Config.robottt)
    --workspace   {off,on}    promoted Stage-S read M* (Pi0Config.wsm_tanh)

    Q0 = off/off (control)   Q1 = off/on (workspace)
    Q2 = on/off  (VANILLA paper-faithful RoboTTT)   Q3 = on/on (both)

Both flags map into ONE `StageQArms` dataclass; every other hyperparameter is shared by construction
(same loader, optimizer, steps, batch, full-finetune surface, RoboTTT geometry), so the 2x2 differs
ONLY in the two booleans. There is no standalone RoboTTT path — Q2 is a config point.

This mirrors finetune_pi_05_with_workspace.py: a --dry-run fully validates the arm derivation and the
TrainConfig without importing OpenPI. A real run additionally requires the canary-gated sequence-loss
dispatch (7a §E); until that is wired, the run path fails closed rather than silently training the
Stage-S per-step loss.
"""

from __future__ import annotations

import argparse
import os

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "pi05", "finetune"


def _on(value: str) -> bool:
    return value == "on"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast-weights", required=True, choices=("off", "on"))
    ap.add_argument("--workspace", required=True, choices=("off", "on"))
    ap.add_argument(
        "--config",
        default="scripts/configs/train/pi05_stage_q_finetune.yaml",
        help="shared recipe for all four Q arms",
    )
    ap.add_argument("--dry-run", action="store_true", help="validate arm derivation without OpenPI")
    args = ap.parse_args()

    from vla_training.train.train_base._pi05_seq_common import (
        build_stage_q_train_config,
        derive_arm,
        stage_q_env,
    )

    arms = derive_arm(_on(args.fast_weights), _on(args.workspace))
    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    stage_q_env(cfg, arms)
    print(
        f"[pi-stage-q] arm={arms.name} fast_weights={arms.fast_weights} workspace={arms.workspace} "
        f"pi0_flags={arms.pi0_flags()} missing_regex={arms.missing_regex()} "
        f"features={os.environ.get('WSM_POLICY_FEATS_ROOT')}",
        flush=True,
    )
    if args.dry_run:
        train_cfg = build_stage_q_train_config(cfg, gs, arms)
        print(
            f"[pi-stage-q] dry-run OK: config_name={train_cfg.name} steps={train_cfg.num_train_steps} "
            f"robottt={train_cfg.model.robottt} wsm_tanh={train_cfg.model.wsm_tanh} "
            f"iid_window_steps={train_cfg.data.stage_q_iid_steps} staged={train_cfg.staged}",
            flush=True,
        )
        return

    if os.environ.get("WSM_STAGE_Q_ALLOW_RUN") != "1":
        raise SystemExit(
            "[pi-stage-q] canary gate: set WSM_STAGE_Q_ALLOW_RUN=1 to run. The sequence dispatch "
            "(7a §E) is wired (StageQWindowDataset -> stage_q_train_step); the gate stays until the "
            "Q0==baseline / Q2!=Q0 parity tests pass in CI and one on-node canary completes."
        )
    from vla_training.train.train_base._pi05_common import run_train_config

    # Dispatch the STAGE-Q builder's config — never the Stage-S builder, which drops the robottt
    # flag (a Q2 run would silently train Q0; audit 2026-07-23). Assert the arm actually reached
    # the model config and the window loader is in force before spending compute.
    train_cfg = build_stage_q_train_config(cfg, gs, arms)
    flags = arms.pi0_flags()
    if bool(train_cfg.model.robottt) != bool(flags["robottt"]) or bool(train_cfg.model.wsm_tanh) != bool(
        flags["wsm_tanh"]
    ):
        raise SystemExit(
            f"[pi-stage-q] arm {arms.name} did not reach the model config: "
            f"model.robottt={train_cfg.model.robottt} model.wsm_tanh={train_cfg.model.wsm_tanh} "
            f"expected {flags}"
        )
    if int(getattr(train_cfg.data, "stage_q_window_len", 0) or 0) < 1:
        raise SystemExit("[pi-stage-q] window loader not in force (stage_q_window_len < 1); refusing")
    run_train_config(train_cfg, cfg, gs)


if __name__ == "__main__":
    main()
