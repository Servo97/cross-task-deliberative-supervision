#!/usr/bin/env python3
"""Canonical RoboMME mechanism registry and single-/multitask experiment expansion.

This registry is intentionally broader than the production arm allowlist.  It makes missing ports
visible instead of silently dropping or mislabelling them; only records with readiness ``ready``
may be handed to a launcher.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from robomme_integration.eval.challenge_results import CHALLENGE_TASK_ORDER
from robomme_integration.training.arms import ARM_IDS

# ``robomme_arm`` names an existing implementation.  A null value is an explicit implementation
# gap.  R1 is never aliased to legacy Q2, and D8 never claims persistent fast weights.
METHODS = (
    {
        "id": "s0",
        "label": "pi0.5 post-training",
        "tier": "core",
        "axis": "control",
        "fast_weights": False,
        "workspace": False,
        "steering": None,
        "aux": None,
        "robomme_arm": "s0",
        "single_task": "ready",
    },
    {
        "id": "q0",
        "label": "sequence-matched no-fast-weight control",
        "tier": "core",
        "axis": "fast_weights",
        "fast_weights": False,
        "workspace": False,
        "steering": None,
        "aux": None,
        "robomme_arm": "q0",
        "single_task": "ready",
    },
    {
        "id": "q0_noforce",
        "label": "Q0 shared-tau L=8 sequence control",
        "tier": "core",
        "axis": "sequence_forcing",
        "fast_weights": False,
        "workspace": False,
        "steering": None,
        "aux": "shared_flow_time_tau_independent_epsilon",
        "robomme_arm": "q0_noforce",
        "single_task": "ready",
    },
    {
        "id": "q1",
        "label": "sequence-matched control + WSM tanh",
        "tier": "expanded",
        "axis": "cross_axis",
        "fast_weights": False,
        "workspace": True,
        "steering": "tanh",
        "aux": None,
        "robomme_arm": "q1",
        "single_task": "workspace_artifact_required",
        "paper_priority": "medium",
    },
    {
        "id": "a6",
        "label": "IID-regrouping no-fast-weight control",
        "tier": "core",
        "axis": "fast_weights",
        "fast_weights": False,
        "workspace": False,
        "steering": None,
        "aux": None,
        "robomme_arm": "a6",
        "single_task": "ready",
    },
    {
        "id": "t1",
        "label": "official RoboMME TTT+Modul",
        "tier": "core",
        "axis": "fast_weights",
        "fast_weights": True,
        "workspace": False,
        "steering": "layer_modulation",
        "aux": "linear_recurrent_ttt",
        "robomme_arm": "t1",
        "single_task": "separate_tpu_path",
    },
    {
        "id": "q2",
        "label": "legacy project Q2 (not paper RoboTTT)",
        "tier": "core",
        "axis": "fast_weights",
        "fast_weights": True,
        "workspace": False,
        "steering": "tanh",
        "aux": "short_context_meta_objective",
        "robomme_arm": "q2",
        "single_task": "ready",
    },
    {
        "id": "q2_noforce",
        "label": "Q2 fast weights + shared-tau L=8 sequence control",
        "tier": "core",
        "axis": "sequence_forcing",
        "fast_weights": True,
        "workspace": False,
        "steering": "tanh",
        "aux": "shared_flow_time_tau_independent_epsilon",
        "robomme_arm": "q2_noforce",
        "single_task": "ready",
    },
    {
        "id": "r1",
        "label": "paper-steelman vanilla RoboTTT",
        "tier": "core",
        "axis": "fast_weights",
        "fast_weights": True,
        "workspace": False,
        "steering": "per_layer_tanh",
        "aux": "maml_meta_learning",
        "robomme_arm": None,
        "single_task": "implementation_required",
    },
    {
        "id": "q3",
        "label": "project Q2 fast weights + WSM tanh",
        "tier": "core",
        "axis": "cross_axis",
        "fast_weights": True,
        "workspace": True,
        "steering": "tanh",
        "aux": "short_context_meta_objective",
        "robomme_arm": "q3",
        "single_task": "workspace_artifact_required",
    },
    {
        "id": "wsm_cfg",
        "label": "WSM + CFG",
        "tier": "core",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "cfg",
        "aux": None,
        "robomme_arm": "wsm_cfg",
        "single_task": "workspace_artifact_required",
    },
    {
        "id": "wsm_tanh",
        "label": "WSM + tanh modulation",
        "tier": "core",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "tanh",
        "aux": None,
        "robomme_arm": "wsm_tanh",
        "single_task": "workspace_artifact_required",
    },
    {
        "id": "wsm_d2",
        "label": "WSM + gated DeltaNet K=2",
        "tier": "expanded",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_k2",
        "aux": None,
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s1_deltanet_w2_finetune.yaml",
    },
    {
        "id": "wsm_d8",
        "label": "WSM + gated DeltaNet K=8",
        "tier": "core",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_k8",
        "aux": None,
        "robomme_arm": "wsm_d8",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s1_deltanet_finetune.yaml",
    },
    {
        "id": "wsm_d8_drop05",
        "label": "WSM + gated DeltaNet K=8 + history dropout p=.5",
        "tier": "core",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_k8",
        "aux": "train_history_dropout_.5",
        "robomme_arm": "wsm_d8_drop05",
        "single_task": "workspace_artifact_required",
        "paper_priority": "high",
        "remembench_config": "scripts/configs/train/pi05_rmb_deltanet_drop_finetune.yaml",
    },
    {
        "id": "ptrm",
        "label": "WSM GDN K=8 + probabilistic TRM",
        "tier": "core",
        "axis": "deliberative_reasoning",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_ptrm_k8",
        "aux": "recursive_T4_q_huber.1",
        "robomme_arm": "ptrm",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_norm_s1_ptrm_finetune.yaml",
        "eval_protocol": "E0_only_K1_sigma0",
        "paper_priority": "medium",
    },
    {
        "id": "wsm_d16",
        "label": "WSM + gated DeltaNet K=16",
        "tier": "expanded",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_k16",
        "aux": None,
        "robomme_arm": "wsm_d16",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s1_deltanet_w16_finetune.yaml",
        "paper_priority": "medium",
    },
    {
        "id": "wsm_d16_drop05",
        "label": "WSM + gated DeltaNet K=16 + history dropout p=.5",
        "tier": "core",
        "axis": "steering",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_k16",
        "aux": "train_history_dropout_.5",
        "robomme_arm": "wsm_d16_drop05",
        "single_task": "workspace_artifact_required",
        "paper_priority": "high",
        "robocasa_config": "scripts/configs/train/pi05_norm_s1_deltanet_w16_drop05_finetune.yaml",
        "remembench_config": "scripts/configs/train/pi05_rmb_deltanet_w16_drop_finetune.yaml",
    },
    {
        "id": "jepa_l1_k1",
        "label": "JEPA+SigReg lambda=1 K=1",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k1_sigreg.05",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_jepa_finetune.yaml",
    },
    {
        "id": "jepa_l1_k2",
        "label": "JEPA+SigReg lambda=1 K=2",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k2_sigreg.05",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_k2_finetune.yaml",
    },
    {
        "id": "jepa_l1_k4",
        "label": "JEPA+SigReg lambda=1 K=4",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k4_sigreg.05",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_k4_finetune.yaml",
    },
    {
        "id": "jepa_l1_k8",
        "label": "JEPA+SigReg lambda=1 K=8",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k8_sigreg.05",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_k8_finetune.yaml",
    },
    {
        "id": "jepa_l1_k16",
        "label": "JEPA+SigReg lambda=1 K=16",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k16_sigreg.05",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_k16_finetune.yaml",
    },
    {
        "id": "jepa_l1_k32",
        "label": "JEPA+SigReg lambda=1 K=32",
        "tier": "core",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k32_sigreg.05",
        "robomme_arm": "jepa_l1_k32",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_k32_finetune.yaml",
    },
    {
        "id": "jepa_l01_k1",
        "label": "JEPA+SigReg lambda=.1 K=1",
        "tier": "core",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda.1_k1_sigreg.05",
        "robomme_arm": "jepa_l01_k1",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_jw01_finetune.yaml",
    },
    {
        "id": "jepa_l01_k16",
        "label": "JEPA+SigReg lambda=.1 K=16",
        "tier": "core",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda.1_k16_sigreg.05",
        "robomme_arm": "jepa_l01_k16",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_jw01k16_finetune.yaml",
    },
    {
        "id": "jepa_sigreg005",
        "label": "JEPA + low SigReg .005",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k1_sigreg.005",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_lam0005_finetune.yaml",
    },
    {
        "id": "jepa_visreg",
        "label": "JEPA + VISReg",
        "tier": "expanded",
        "axis": "representation_objective",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "jepa_lambda1_k1_visreg.05",
        "robomme_arm": None,
        "single_task": "port_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s3_visreg_finetune.yaml",
    },
    {
        "id": "gdn8_jepa_l01_k1",
        "label": "WSM GDN K=8 + JEPA lambda=.1 K=1 (SigReg=.05)",
        "tier": "core",
        "axis": "cross_axis",
        "fast_weights": False,
        "workspace": True,
        "steering": "gated_deltanet_k8",
        "aux": "jepa_lambda.1_k1_sigreg.05",
        "robomme_arm": "gdn8_jepa_l01_k1",
        "single_task": "workspace_artifact_required",
        "robocasa_config": "scripts/configs/train/pi05_stage_s1_gdnjepa_finetune.yaml",
        "paper_priority": "deprioritized",
    },
    {
        "id": "salient",
        "label": "salient-patch deliberative supervision",
        "tier": "core",
        "axis": "deliberative_supervision",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "salient_patch_reconstruction_64",
        "robomme_arm": "salient",
        "single_task": "workspace_supervision_required",
    },
    {
        "id": "causal_v1",
        "label": "causal-content keypatch supervision",
        "tier": "expanded",
        "axis": "deliberative_supervision",
        "fast_weights": False,
        "workspace": True,
        "steering": None,
        "aux": "causal_v1_keypatch",
        "robomme_arm": "causal_v1",
        "single_task": "workspace_supervision_required",
        "paper_priority": "low",
        "remembench_config": "scripts/configs/train/pi05_rmb_causal_finetune.yaml",
    },
)
MULTITASK_READY = frozenset({"s0", "q0", "a6", "q2"})


def _multitask_readiness(method: dict) -> str:
    if method["id"] in MULTITASK_READY:
        return "ready"
    if method["id"] == "r1":
        return "implementation_required"
    if method["id"] == "t1":
        return "multitask_ttt_port_required"
    if method["workspace"] and method["robomme_arm"] in ARM_IDS:
        return "all16_workspace_artifacts_required"
    return "multitask_adapter_required"


def validate_registry(repo_root: Path | None = None) -> None:
    ids = [method["id"] for method in METHODS]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate RoboMME method IDs")
    known = set(ARM_IDS) | {"t1"}
    invalid = [m["robomme_arm"] for m in METHODS if m["robomme_arm"] and m["robomme_arm"] not in known]
    if invalid:
        raise ValueError(f"unknown implemented RoboMME arms: {invalid}")
    if next(method for method in METHODS if method["id"] == "r1")["robomme_arm"] is not None:
        raise ValueError("paper RoboTTT R1 must not be aliased to Q2")
    if next(method for method in METHODS if method["id"] == "wsm_d8")["fast_weights"]:
        raise ValueError("gated DeltaNet D8 has transient recurrence, not persistent fast weights")
    if repo_root:
        config_keys = ("robocasa_config", "remembench_config")
        missing = [m[key] for m in METHODS for key in config_keys if m.get(key) and not (repo_root / m[key]).is_file()]
        if missing:
            raise ValueError(f"missing referenced RoboCasa configurations: {missing}")


def expanded_cells(*, tier: str = "all") -> list[dict]:
    if tier not in {"core", "expanded", "all"}:
        raise ValueError("tier must be core, expanded, or all")
    methods = [m for m in METHODS if tier == "all" or m["tier"] == tier]
    cells = []
    for method in methods:
        for task in CHALLENGE_TASK_ORDER:
            cells.append(
                {
                    "cell_id": f"st::{task}::{method['id']}",
                    "method_id": method["id"],
                    "task": task,
                    "training_scope": "single_task",
                    "eval_episodes": 50,
                    "challenge_comparable": False,
                    "readiness": method["single_task"],
                }
            )
        cells.append(
            {
                "cell_id": f"mt::all16::{method['id']}",
                "method_id": method["id"],
                "task": "all16",
                "training_scope": "multitask",
                "eval_episodes": 800,
                "challenge_comparable": True,
                "readiness": _multitask_readiness(method),
            }
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("core", "expanded", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_registry(Path(__file__).resolve().parents[1])
    cells = expanded_cells(tier=args.tier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".json":
        args.output.write_text(json.dumps({"methods": METHODS, "cells": cells}, indent=2) + "\n")
        return
    fields = tuple(cells[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)


if __name__ == "__main__":
    main()
