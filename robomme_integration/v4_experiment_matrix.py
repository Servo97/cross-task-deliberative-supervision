#!/usr/bin/env python3
"""Validate and expand the no-submit RoboMME v4 evidence matrix.

CFG guidance values expand to distinct evaluation identities.  The output deliberately retains
blocked implementation gaps so a figure plan can never silently substitute Q2 for paper RoboTTT
or workspace history plumbing for FrameSamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robomme_integration.training.arms import V4_ARM_IDS, WORKSPACE_ARMS

DEFAULT_SPEC = Path(__file__).with_name("sweeps") / "robomme_v4_experiment_matrix.json"
CFG_ARMS = frozenset({"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"})
SEALED_CFG_SCALES = (0.5, 1.0, 1.5, 2.0)


def load_and_validate(path: Path = DEFAULT_SPEC) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("cloud_action") is not False:
        raise ValueError("RoboMME v4 matrix must be schema 1 and no-cloud")
    if value.get("implemented_arms") != [arm for arm in value["implemented_arms"]]:
        raise ValueError("implemented arms must be an ordered list")
    if set(value["implemented_arms"]) != V4_ARM_IDS:
        raise ValueError("RoboMME v4 matrix and arm registry differ")
    protocol = value.get("protocol", {})
    expected = {
        "warmup_fraction": 0.05,
        "ema_decay": 0.999,
        "cfg_train_dropout": 0.2,
        "cfg_eval_scales": list(SEALED_CFG_SCALES),
        "gdn_history_dropout": [0.0, 0.2],
    }
    drift = {
        key: (protocol.get(key), expected_value)
        for key, expected_value in expected.items()
        if protocol.get(key) != expected_value
    }
    if drift:
        raise ValueError(f"RoboMME v4 protocol matrix drifted: {drift}")
    optimizer = protocol.get("optimizer")
    if optimizer != {
        "name": "AdamW",
        "backbone_peak_lr": 5e-5,
        "new_module_peak_lr": 3e-4,
        "weight_decay": 1e-6,
        "global_clip_norm": 10.0,
    }:
        raise ValueError("RoboMME v4 optimizer matrix drifted")
    jepa = protocol.get("jepa")
    if jepa != {
        "weight": 0.1,
        "futures": 1,
        "regularizer": "visreg",
        "visreg_weight": 0.05,
        "visreg_slices": 128,
        "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
    }:
        raise ValueError("RoboMME v4 JEPA-VISReg matrix drifted")
    blocked = {record["method"]: record["readiness"] for record in value.get("blocked_methods", [])}
    if blocked != {
        "t1_naive_ttt": "blocked",
        "r1_paper_robottt": "blocked",
        "framesamp": "blocked",
        "framesamp_plus_wsm": "blocked",
    }:
        raise ValueError("RoboMME v4 missing-method gates drifted")
    return value


def expand(value: dict) -> list[dict]:
    cells: list[dict] = []
    for phase, phase_spec in value["phases"].items():
        scope = phase_spec["scope"]
        for task in phase_spec["tasks"]:
            for arm in value["arm_sets"][phase_spec["arm_set"]]:
                training_id = f"{phase}::{scope}::{task}::{arm}::seed0"
                workspace = arm in WORKSPACE_ARMS
                readiness = (
                    "ready"
                    if not workspace
                    else "all16_workspace_index_required"
                    if scope == "multitask"
                    else "task_workspace_artifact_required"
                )
                scales = SEALED_CFG_SCALES if arm in CFG_ARMS else (1.0,)
                cells.append(
                    {
                        "kind": "training",
                        "cell_id": training_id,
                        "phase": phase,
                        "scope": scope,
                        "task": task,
                        "arm": arm,
                        "seed": 0,
                        "steps": 60_000 if scope == "multitask" else 20_000,
                        "warmup_steps": 3_000 if scope == "multitask" else 1_000,
                        "readiness": readiness,
                    }
                )
                for scale in scales:
                    cells.append(
                        {
                            "kind": "evaluation",
                            "cell_id": f"{training_id}::fixed50::cfg{scale:g}",
                            "training_cell_id": training_id,
                            "phase": phase,
                            "scope": scope,
                            "task": task,
                            "arm": arm,
                            "cfg_guidance_scale": scale,
                            "episodes_per_task": 50,
                            "readiness": f"checkpoint_and_{readiness}",
                        }
                    )
            for slot in range(int(phase_spec.get("promoted_slots", 0))):
                cells.append(
                    {
                        "kind": "promotion_slot",
                        "cell_id": f"{phase}::{scope}::{task}::promoted{slot + 1}",
                        "phase": phase,
                        "scope": scope,
                        "task": task,
                        "arm": None,
                        "promotion_rank": slot + 1,
                        "promotion_source": phase_spec["promotion_source"],
                        "readiness": "blocked_until_phase_a_promotion",
                    }
                )
    identities = [cell["cell_id"] for cell in cells]
    if len(identities) != len(set(identities)):
        raise AssertionError("RoboMME v4 expanded duplicate cell identities")
    scientific_training = [
        (cell["scope"], cell["task"], cell["arm"], cell["seed"]) for cell in cells if cell["kind"] == "training"
    ]
    if len(scientific_training) != len(set(scientific_training)):
        raise AssertionError("RoboMME v4 repeats a scientific training identity across phases")
    for arm in CFG_ARMS:
        grouped: dict[str, set[float]] = {}
        for cell in cells:
            if cell["kind"] == "evaluation" and cell["arm"] == arm:
                grouped.setdefault(cell["training_cell_id"], set()).add(cell["cfg_guidance_scale"])
        if any(scales != set(SEALED_CFG_SCALES) for scales in grouped.values()):
            raise AssertionError(f"{arm} matrix has an incomplete CFG scale sweep")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = load_and_validate(args.spec)
    payload = {
        "schema_version": 1,
        "matrix": value["name"],
        "cloud_action": False,
        "blocked_methods": value["blocked_methods"],
        "cells": expand(value),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
