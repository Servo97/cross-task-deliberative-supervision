#!/usr/bin/env python3
"""Approval-gated p5e/H200 or p5/H100 launcher for RoboMME training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
LAUNCH_UTILS = REPO_ROOT / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_UTILS))
from launch_guardrails import (  # noqa: E402
    DEFAULT_RESULTS_BUCKET,
    DEXJOCO_IMAGE_REPO,
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    QUEUE,
    ROLE_ARN,
    STUDY_OWNER,
    TRAINING_PLAN_QUEUE,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
    training_plan_arn,
)

from robomme_integration.fleet.task_inventory import (  # noqa: E402
    CANONICAL_PARENT_SHA256,
    CANONICAL_TASK_DERIVED_SHA256,
)
from robomme_integration.training import (  # noqa: E402
    cfg_jepa_overlay,
    gdn_jepa_overlay,
    sequence_forcing,
)
from robomme_integration.training.arms import (  # noqa: E402
    ARM_IDS,
    OFFICIAL_RECIPE_LEROBOT_ARM,
    OFFICIAL_RECIPE_LEROBOT_LABEL,
    OFFICIAL_RECIPE_LEROBOT_MILESTONES,
    OFFICIAL_RECIPE_LEROBOT_STEPS,
    ROBOTT_ARMS,
    SEQUENCE_ARMS,
    V4_ARM_IDS,
    V4_NEW_PARAMETER_SUBTREES,
    WORKSPACE_ARMS,
)
from robomme_integration.training.single_task import (  # noqa: E402
    TASK_EPISODES,
    TASK_ORDER,
    task_manifest_sha256,
)

ENTRY = "gpu_train_entry.sh"
STAGED_MANIFEST = "_robomme_gpu_run_manifest.json"
STUDY = "long_context_v1"
OWNER = STUDY_OWNER
PRIORITY = 400
P5_BACKFILL_PRIORITY = 1
MAX_RUN_SECONDS = 24 * 3600
TRAIN_STEPS = 20_000
MULTITASK_TRAIN_STEPS = 60_000
# A19 checkpoint-maturity recipe (2026-09-02): a multitask v4 arm may train for 70k steps with every
# 10k milestone retained and exported as a deploy-only params+assets tree.  The sealed 60k path is
# untouched: the flag defaults to 60k and a 60k plan is byte-identical to the pre-A19 launcher
# (modulo the source-tree digest, which any edit under robomme_integration/ necessarily moves).
V4_70K_RECIPE = "v4_70k"
V4_70K_TRAIN_STEPS = 70_000
V4_70K_MILESTONE_INTERVAL = 10_000
MULTITASK_TRAIN_STEP_CHOICES = (MULTITASK_TRAIN_STEPS, V4_70K_TRAIN_STEPS)
SAVE_INTERVAL = 5_000
VOLUME_MIN_GB = 250
VOLUME_MAX_GB = 400
RETRY = {"attempts": 1}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

STUDY_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/studies/{STUDY}"
DATA_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/datasets/robomme/v1/lerobot_all16"
DATA_INVENTORY = f"{STUDY_ROOT}/manifests/inventories/data/{CANONICAL_PARENT_SHA256}.json"
INIT_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/pretrain150k/pi05/mg60_bal33/run/149999"
INIT_INVENTORY_SHA = "34932efcfeee9b11181a5915ecce3be47aaeb01b5bf9e3f5057c022f4db01b04"
INIT_INVENTORY = f"{STUDY_ROOT}/manifests/inventories/init/{INIT_INVENTORY_SHA}.json"
PI05_BASE_INIT_ROOT = f"{STUDY_ROOT}/artifacts/initialization/pi05_base"
PI05_BASE_INIT_INVENTORY_ROOT = f"{STUDY_ROOT}/manifests/inventories/init/pi05_base"
TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TOKENIZER = f"{STUDY_ROOT}/artifacts/tokenizers/paligemma/{TOKENIZER_SHA}.model"
OPENPI_SHA = "ed923b2c27d2f608d62cc4b5ca89d5b80c14739dba1ab81d6f53d8013bcb66ad"
OPENPI = f"{STUDY_ROOT}/code/openpi/{OPENPI_SHA}.tgz"
PTRM_OPENPI_SHA = "24bd889d3c0b95a7b01cd6ad30a91fdc266fa115fb2ef5ec89fe45c9c5260900"
PTRM_OPENPI = f"{STUDY_ROOT}/code/openpi/{PTRM_OPENPI_SHA}.tgz"
HISTORY_DROPOUT_ARMS = frozenset({"wsm_d8_drop05", "wsm_d16_drop05", "v4_wsm_gdn8_drop02", "v4_wsm_gdn16_drop02"})
V4_ADVANCED_GDN_ARMS = frozenset(
    {
        "v4_wsm_gdn8_drop00",
        "v4_wsm_gdn8_drop02",
        "v4_wsm_gdn16_drop00",
        "v4_wsm_gdn16_drop02",
        "v4_gdn8_jepa_visreg_l01_k1",
        "v4_ptrm",
    }
)
SUPERVISION_ARMS = frozenset({"salient", "causal_v1"})
IMAGE_SHA = "798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"
IMAGE = f"{DEXJOCO_IMAGE_REPO}@sha256:{IMAGE_SHA}"
PRODUCTION_ARM_IDS = tuple(arm for arm in ARM_IDS if arm not in {"q0v", "q2v"})
HARDWARE = {
    "p5e": {
        "queue": TRAINING_PLAN_QUEUE,
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "reserved_capacity": "0",
    },
    "p5": {
        "queue": QUEUE,
        "instance_type": "ml.p5.48xlarge",
        "accelerator": "8xH100-80GB-HBM3",
        "reserved_capacity": "1",
    },
}


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seal(value: dict) -> tuple[dict, str]:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    digest = hashlib.sha256(_canonical_json(clean).encode()).hexdigest()
    clean["manifest_sha256"] = digest
    return clean, _canonical_json(clean)


def v4_70k_milestones(train_steps: int) -> tuple[int, ...]:
    """Intermediate milestones retained by the v4_70k recipe: every 10k step before the final one."""
    if train_steps != V4_70K_TRAIN_STEPS:
        raise ValueError(f"the {V4_70K_RECIPE} recipe is defined for {V4_70K_TRAIN_STEPS} steps, got {train_steps}")
    return tuple(range(V4_70K_MILESTONE_INTERVAL, train_steps, V4_70K_MILESTONE_INTERVAL))


def _source_dir(value: str | None) -> Path:
    root = Path(value or Path(__file__).resolve().parent).resolve()
    if not (root / ENTRY).is_file() or not (root / "training" / "train.py").is_file():
        raise SystemExit(f"invalid isolated RoboMME source directory: {root}")
    return root


def _require_sha(value: str | None, flag: str) -> str:
    if not value or not HEX64.fullmatch(value):
        raise SystemExit(f"{flag} must be 64 lowercase hexadecimal characters")
    return value


def _validate_contract(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.confirm_submit:
        raise SystemExit("submission blocked: obtain explicit user approval, then pass --confirm-submit")
    hardware = HARDWARE[args.hardware]
    if args.queue != hardware["queue"]:
        raise SystemExit(f"RoboMME {args.hardware} training must use queue {hardware['queue']}; got {args.queue}")
    if args.role != ROLE_ARN:
        raise SystemExit(f"RoboMME training must use execution role {ROLE_ARN}")
    allowed_priorities = {PRIORITY}
    if args.hardware == "p5":
        allowed_priorities.add(P5_BACKFILL_PRIORITY)
    if args.priority not in allowed_priorities:
        allowed = "/".join(str(value) for value in sorted(allowed_priorities))
        raise SystemExit(
            f"RoboMME {args.hardware} training must use priority {allowed}; priority 600 needs a new user approval"
        )
    if not 1 <= args.max_run_seconds <= MAX_RUN_SECONDS:
        raise SystemExit(f"RoboMME GPU jobs must fit within {MAX_RUN_SECONDS} seconds")
    if not VOLUME_MIN_GB <= args.volume_size_gb <= VOLUME_MAX_GB:
        raise SystemExit(f"RoboMME single-task volume must be in [{VOLUME_MIN_GB}, {VOLUME_MAX_GB}] GiB")
    if args.attempt_index < 1:
        raise SystemExit("--attempt-index must be positive")
    if args.secrets_manager_arn and not args.secrets_manager_arn.startswith("arn:aws:secretsmanager:"):
        raise SystemExit("--secrets-manager-arn must be an AWS Secrets Manager ARN")
    if args.arm not in PRODUCTION_ARM_IDS:
        raise SystemExit(f"{args.arm} is demo-guided and excluded from the current production scope")
    if args.scope == "single_task" and not args.task:
        raise SystemExit("single_task scope requires --task")
    if args.scope == "multitask" and args.task:
        raise SystemExit("multitask scope covers all16 and forbids --task")
    base_init_fields = (
        args.pi05_base_init_s3,
        args.pi05_base_init_inventory_s3,
        args.pi05_base_init_inventory_sha256,
    )
    if args.arm == OFFICIAL_RECIPE_LEROBOT_ARM:
        if args.scope != "multitask" or args.task:
            raise SystemExit("official_recipe_lerobot is an all16-only multitask diagnostic")
        if not all(base_init_fields):
            raise SystemExit("official_recipe_lerobot requires the three distinct --pi05-base-init-* arguments")
    elif any(base_init_fields):
        raise SystemExit("only official_recipe_lerobot accepts --pi05-base-init-* arguments")
    multitask_train_steps = int(getattr(args, "multitask_train_steps", MULTITASK_TRAIN_STEPS))
    if multitask_train_steps != MULTITASK_TRAIN_STEPS and (
        args.scope != "multitask" or args.arm not in V4_ARM_IDS or args.arm == OFFICIAL_RECIPE_LEROBOT_ARM
    ):
        raise SystemExit(
            f"--multitask-train-steps {multitask_train_steps} ({V4_70K_RECIPE}) is defined only for "
            "multitask v4 arms; single-task, legacy-v1 and official_recipe_lerobot keep their sealed "
            "step counts"
        )


def _workspace_spec(args: argparse.Namespace) -> dict | None:
    single_task_fields = (
        args.workspace_encoder_id,
        args.workspace_s3,
        args.workspace_manifest_sha256,
    )
    index_fields = (args.workspace_index_s3, args.workspace_index_sha256)
    if args.arm not in WORKSPACE_ARMS:
        if any(single_task_fields) or any(index_fields) or args.supervision_s3 or args.supervision_manifest_sha256:
            raise SystemExit(f"{args.arm} forbids workspace/supervision artifact arguments")
        return None
    if args.scope == "multitask":
        if any(single_task_fields) or args.supervision_s3 or args.supervision_manifest_sha256:
            raise SystemExit("multitask workspace arms accept only the sealed all-16 workspace index")
        if not all(index_fields):
            raise SystemExit(f"multitask {args.arm} requires --workspace-index-s3 and --workspace-index-sha256")
        index_sha = _require_sha(args.workspace_index_sha256, "--workspace-index-sha256")
        expected_index = f"{STUDY_ROOT}/artifacts/robomme/workspace/all16/{index_sha}.json"
        if args.workspace_index_s3 != expected_index:
            raise SystemExit(f"--workspace-index-s3 must be {expected_index}")
        return {
            "index": {"uri": expected_index, "sha256": index_sha},
            "task_bound": False,
            "router": "pinned_episode_manifest_v1",
            "tasks": list(TASK_ORDER),
            "omega_symbol": "omega_t",
            "requires_supervision": args.arm in SUPERVISION_ARMS,
        }
    if any(index_fields):
        raise SystemExit("single-task workspace arms forbid the all-16 workspace index")
    if not all(single_task_fields):
        raise SystemExit(
            f"{args.arm} requires --workspace-encoder-id, --workspace-s3, and --workspace-manifest-sha256"
        )
    encoder = _require_sha(args.workspace_encoder_id, "--workspace-encoder-id")
    workspace_sha = _require_sha(args.workspace_manifest_sha256, "--workspace-manifest-sha256")
    expected_root = f"{STUDY_ROOT}/artifacts/robomme/workspace/{args.task}/{encoder}"
    expected_workspace = f"{expected_root}/omega"
    if args.workspace_s3.rstrip("/") != expected_workspace:
        raise SystemExit(f"--workspace-s3 must be {expected_workspace}")
    supervision = None
    if args.arm in SUPERVISION_ARMS:
        supervision_sha = _require_sha(args.supervision_manifest_sha256, "--supervision-manifest-sha256")
        expected_supervision = f"{expected_root}/supervision"
        if not args.supervision_s3 or args.supervision_s3.rstrip("/") != expected_supervision:
            raise SystemExit(f"--supervision-s3 must be {expected_supervision}")
        supervision = {"uri": expected_supervision, "manifest_sha256": supervision_sha}
    elif args.supervision_s3 or args.supervision_manifest_sha256:
        raise SystemExit("only salient/causal_v1 arms accept supervision artifacts")
    return {
        "encoder_id": encoder,
        "omega": {"uri": expected_workspace, "manifest_sha256": workspace_sha},
        "supervision": supervision,
        "task_bound": True,
        "omega_symbol": "omega_t",
    }


def _arm_spec(arm: str) -> dict:
    deltanet_windows = {
        "wsm_d8": 8,
        "wsm_d8_drop05": 8,
        "wsm_d16": 16,
        "wsm_d16_drop05": 16,
        "gdn8_jepa_l01_k1": 8,
        "v4_wsm_gdn8_drop00": 8,
        "v4_wsm_gdn8_drop02": 8,
        "v4_wsm_gdn16_drop00": 16,
        "v4_wsm_gdn16_drop02": 16,
        "v4_gdn8_jepa_visreg_l01_k1": 8,
    }
    jepa = {
        "jepa_l01_k1": {"lambda": 0.1, "futures": 1},
        "jepa_l1_k32": {"lambda": 1.0, "futures": 32},
        "jepa_l01_k16": {"lambda": 0.1, "futures": 16},
        "gdn8_jepa_l01_k1": {"lambda": 0.1, "futures": 1, "sigreg": 0.05},
        "v4_jepa_visreg_l01_k1": {
            "lambda": 0.1,
            "futures": 1,
            "regularizer": "visreg",
            "sigreg_weight": 0.0,
            "visreg_weight": 0.05,
            "visreg_slices": 128,
            "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
        },
        "v4_gdn8_jepa_visreg_l01_k1": {
            "lambda": 0.1,
            "futures": 1,
            "regularizer": "visreg",
            "sigreg_weight": 0.0,
            "visreg_weight": 0.05,
            "visreg_slices": 128,
            "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
        },
        "v4_cfg_jepa_visreg_l01_k1": {
            "lambda": 0.1,
            "futures": 1,
            "regularizer": "visreg",
            "sigreg_weight": 0.0,
            "visreg_weight": 0.05,
            "visreg_slices": 128,
            "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
        },
        "salient": {"lambda": 0.0, "futures": 1, "salient_patches": 64},
        "causal_v1": {
            "lambda": 0.0,
            "futures": 1,
            "salient_patches": 64,
            "label_spec": "causal_v1",
        },
    }.get(arm)
    spec = {
        "sequence_windows": arm in SEQUENCE_ARMS,
        "sequence_forcing": (
            "shared_flow_time_tau_within_L8"
            if arm in {"q0_noforce", "q2_noforce"}
            else "independent_flow_time_tau_per_chunk"
            if arm in SEQUENCE_ARMS
            else None
        ),
        "robottt_fast_weights_W_t": arm in ROBOTT_ARMS,
        "workspace_tokens_omega_t": arm in WORKSPACE_ARMS,
        "steering": (
            "cfg"
            if arm in {"wsm_cfg", "v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}
            else "gated_deltanet_ptrm_k8"
            if arm in {"ptrm", "v4_ptrm"}
            else f"gated_deltanet_k{deltanet_windows[arm]}"
            if arm in deltanet_windows
            else "tanh"
            if arm in {"q1", "q3", "wsm_tanh", "v4_wsm_tanh"}
            else None
        ),
        "jepa": jepa,
        "ptrm": (
            {
                "steps": 4,
                "q_loss": "huber_log_flow_loss",
                "q_weight": 0.1,
                "train_noise": 0.0,
                "eval_width_is_inference_only": True,
            }
            if arm in {"ptrm", "v4_ptrm"}
            else None
        ),
        # Deprecated non-claiming field. Q0 is a sequence/no-fast-weight control, never the still-
        # blocked upstream naive-TTT T1. Q2 is likewise a project fast-weight proxy, not RoboTTT.
        "naive_ttt_control": False,
        "actual_naive_ttt_t1": False,
        "sequence_no_fast_weight_control": arm in {"q0", "q0_noforce", "v4_q0"},
        "project_fast_weight_proxy": arm == "v4_q2",
        "iid_regrouping_control": arm == "a6",
    }
    if arm == OFFICIAL_RECIPE_LEROBOT_ARM:
        spec["diagnostic"] = {
            "identity": OFFICIAL_RECIPE_LEROBOT_ARM,
            "label": OFFICIAL_RECIPE_LEROBOT_LABEL,
            "recipe_matched": True,
            "exact_official_source_reproduction": False,
            "exact_official_data_reproduction": False,
        }
    if arm in HISTORY_DROPOUT_ARMS:
        spec["train_history_dropout"] = 0.2 if arm in {"v4_wsm_gdn8_drop02", "v4_wsm_gdn16_drop02"} else 0.5
    elif arm in {
        "v4_wsm_gdn8_drop00",
        "v4_wsm_gdn16_drop00",
        "v4_gdn8_jepa_visreg_l01_k1",
        "v4_ptrm",
    }:
        # Zero is a scientific treatment, not an omitted/default field.  Keeping it explicit makes
        # the GDN 0/.2 factorial and the source-matched GDN+JEPA/PTRM controls self-describing.
        spec["train_history_dropout"] = 0.0
    if arm in {"q0_noforce", "q2_noforce"}:
        spec["openpi_overlay"] = {
            "version": sequence_forcing.OVERLAY_VERSION,
            "base_archive_sha256": sequence_forcing.BASE_ARCHIVE_SHA256,
            "source_pi0_sha256": sequence_forcing.BASE_PI0_SHA256,
            "patched_pi0_sha256": sequence_forcing.PATCHED_PI0_SHA256,
            "scientific_delta": "share flow tau within L=8; epsilon stays per chunk",
        }
    if arm == "gdn8_jepa_l01_k1":
        spec["openpi_overlay"] = {
            "kind": gdn_jepa_overlay.OVERLAY_KIND,
            "version": gdn_jepa_overlay.OVERLAY_VERSION,
            "manifest_sha256": gdn_jepa_overlay._expected_manifest()["manifest_sha256"],
            "runtime_tree_sha256": gdn_jepa_overlay.PATCHED_RUNTIME_TREE_SHA256,
            "base_archive_sha256": gdn_jepa_overlay.BASE_ARCHIVE_SHA256,
            "model_math_changed": False,
        }
    if arm == "v4_gdn8_jepa_visreg_l01_k1":
        spec["advanced_openpi_capability"] = {
            "source_sha256": PTRM_OPENPI_SHA,
            "requires": [
                "gated_deltanet",
                "history_dropout",
                "gated_deltanet_ptrm",
                "tanh_plus_jepa_pair",
                "visreg",
            ],
            "source_matched_parent": "v4_wsm_gdn8_drop00",
        }
    if arm == "v4_cfg_jepa_visreg_l01_k1":
        spec["openpi_overlay"] = {
            "kind": cfg_jepa_overlay.OVERLAY_KIND,
            "version": cfg_jepa_overlay.OVERLAY_VERSION,
            "manifest_sha256": cfg_jepa_overlay._expected_manifest()["manifest_sha256"],
            "runtime_tree_sha256": cfg_jepa_overlay.PATCHED_RUNTIME_TREE_SHA256,
            "base_archive_sha256": cfg_jepa_overlay.BASE_ARCHIVE_SHA256,
            "model_math_changed": False,
        }
    if arm in V4_ARM_IDS:
        spec["protocol"] = "robomme_v4"
        spec["new_parameter_subtrees"] = list(V4_NEW_PARAMETER_SUBTREES[arm])
    return spec


def build_plan(args: argparse.Namespace, source_dir: Path) -> dict:
    if args.queue is None:
        args.queue = HARDWARE[args.hardware]["queue"]
    _validate_contract(args)
    if args.data_s3 != DATA_ROOT or args.data_inventory_s3 != DATA_INVENTORY:
        raise SystemExit("RoboMME data must use the registered consolidated dataset and strong v2 inventory")
    if args.data_inventory_sha256 != CANONICAL_PARENT_SHA256:
        raise SystemExit("RoboMME parent inventory SHA differs from the strong v2 identity")
    official_recipe_lerobot = args.arm == OFFICIAL_RECIPE_LEROBOT_ARM
    if official_recipe_lerobot:
        if (
            args.init_s3 != INIT_ROOT
            or args.init_inventory_s3 != INIT_INVENTORY
            or args.init_inventory_sha256 != INIT_INVENTORY_SHA
        ):
            raise SystemExit("official_recipe_lerobot forbids repurposing the legacy --init-* H300+MG channel")
        base_sha = _require_sha(
            args.pi05_base_init_inventory_sha256,
            "--pi05-base-init-inventory-sha256",
        )
        if args.pi05_base_init_s3 != PI05_BASE_INIT_ROOT:
            raise SystemExit(f"--pi05-base-init-s3 must be {PI05_BASE_INIT_ROOT}")
        expected_base_inventory = f"{PI05_BASE_INIT_INVENTORY_ROOT}/{base_sha}.json"
        if args.pi05_base_init_inventory_s3 != expected_base_inventory:
            raise SystemExit(f"--pi05-base-init-inventory-s3 must be {expected_base_inventory}")
        if base_sha == INIT_INVENTORY_SHA:
            raise SystemExit("pi0.5-base inventory must not alias the H300+MG inventory")
        initialization = {
            "recipe": "published pi0.5 base",
            "artifact": "pi05_base_init",
            "checkpoint_s3": args.pi05_base_init_s3,
            "inventory_uri": args.pi05_base_init_inventory_s3,
            "inventory_sha256": base_sha,
        }
    else:
        if args.init_s3 != INIT_ROOT or args.init_inventory_s3 != INIT_INVENTORY:
            raise SystemExit("initialization must be the registered H300+MG checkpoint/inventory")
        if args.init_inventory_sha256 != INIT_INVENTORY_SHA:
            raise SystemExit("H300+MG inventory SHA drifted")
        initialization = {
            "recipe": "RoboCasa H300+MG balanced",
            "checkpoint_s3": args.init_s3,
            "inventory_uri": args.init_inventory_s3,
            "inventory_sha256": args.init_inventory_sha256,
        }
    advanced_openpi = args.arm == "ptrm" or args.arm in HISTORY_DROPOUT_ARMS or args.arm in V4_ADVANCED_GDN_ARMS
    expected_openpi = PTRM_OPENPI if advanced_openpi else OPENPI
    expected_openpi_sha = PTRM_OPENPI_SHA if advanced_openpi else OPENPI_SHA
    if args.openpi_source_s3 is None:
        args.openpi_source_s3 = expected_openpi
    if args.openpi_source_s3 != expected_openpi or args.tokenizer_s3 != TOKENIZER:
        raise SystemExit("OpenPI/tokenizer differs from the registered current campaign source")
    if args.tokenizer_sha256 != TOKENIZER_SHA or args.image_uri != IMAGE:
        raise SystemExit("tokenizer/image digest drifted")
    workspace = _workspace_spec(args)
    hardware = HARDWARE[args.hardware]
    plan_arn = training_plan_arn(args.queue)
    if args.hardware == "p5e" and not plan_arn:
        raise SystemExit("p5e queue lost its required training-plan ARN")
    if args.hardware == "p5" and plan_arn:
        raise SystemExit("ordinary p5 queue unexpectedly resolved to a training plan")

    with prepared_source_bundle(
        source_dir,
        ENTRY,
        {"SAGEMAKER_PROGRAM": ENTRY},
        args.secrets_manager_arn,
    ) as (staged, _entry, _environment):
        source_sha = source_tree_sha256(staged)
    entry_sha = hashlib.sha256((source_dir / ENTRY).read_bytes()).hexdigest()
    multitask_train_steps = int(getattr(args, "multitask_train_steps", MULTITASK_TRAIN_STEPS))
    train_steps = (
        OFFICIAL_RECIPE_LEROBOT_STEPS
        if official_recipe_lerobot
        else TRAIN_STEPS
        if args.scope == "single_task"
        else multitask_train_steps
    )
    final_step = train_steps - 1
    # The v4_70k checkpoint-maturity recipe is an explicit opt-in; nothing below this line changes
    # for the sealed 60k multitask, 20k single-task or 80k official-recipe paths.
    v4_70k = args.scope == "multitask" and not official_recipe_lerobot and train_steps != MULTITASK_TRAIN_STEPS
    if v4_70k and args.arm not in V4_ARM_IDS:
        raise SystemExit(f"{V4_70K_RECIPE} requires a multitask v4 arm")
    milestones = v4_70k_milestones(train_steps) if v4_70k else ()
    task_inventory_sha = CANONICAL_TASK_DERIVED_SHA256[args.task] if args.scope == "single_task" else None
    sequence = args.arm in SEQUENCE_ARMS
    task_spec = (
        {
            "name": args.task,
            "episodes": list(TASK_EPISODES[args.task]),
            "task_manifest_sha256": task_manifest_sha256(args.task),
        }
        if args.scope == "single_task"
        else {
            "name": "all16",
            "tasks": list(TASK_ORDER),
            "episodes": 1_600,
            "task_manifest_sha256": CANONICAL_PARENT_SHA256,
        }
    )
    data_spec = {
        "repo_id": "Yinpei/robomme_data_lerobot",
        "revision": "1510653cccb4d9e5165fb3141c06d88053decc20",
        "dataset_s3": args.data_s3,
        "parent_inventory_uri": args.data_inventory_s3,
        "parent_inventory_sha256": args.data_inventory_sha256,
        "derived_task_inventory_sha256": task_inventory_sha,
        "download_policy": (
            "shared_metadata_plus_exact_100_task_parquets"
            if args.scope == "single_task"
            else "complete_verified_all16_inventory"
        ),
        "views": ["base_0_rgb", "left_wrist_0_rgb"],
        "state_dim": 8,
        "action_dim": 8,
    }
    training_spec = {
        "steps": train_steps,
        "seed": 0,
        "batch_size": 8 if sequence else 64,
        "batch_unit": "windows" if sequence else "steps",
        "effective_per_step_batch": 64,
        "action_horizon": 20,
        "window_len": 8 if sequence else None,
        "chunk_stride": 10 if sequence else None,
        "full_finetune": True,
        "optimizer": "AdamW",
        "peak_lr": 5e-5,
        "warmup_steps": max(1, train_steps // 20) if args.arm in V4_ARM_IDS else 1_000,
        "decay_steps": train_steps,
        "decay_lr": 5e-6,
        "ema_decay": 0.99,
        "num_workers": 32,
        "jax_devices": 8,
        "jax_processes": 1,
        "fsdp_devices": 1,
        "checkpoint_policy": {
            "save_interval": SAVE_INTERVAL,
            "remote_resume": True,
            "upload_only_finalized_orbax": True,
            "training_retention": f"newest_plus_step{train_steps // 2}",
            "success_retention": [final_step],
        },
    }
    if args.arm in V4_ARM_IDS:
        training_spec.update(
            protocol="robomme_v4",
            optimizer={
                "name": "AdamW",
                "b1": 0.9,
                "b2": 0.95,
                "eps": 1e-8,
                "weight_decay": 1e-6,
                "clip_gradient_norm": 10.0,
                "clip_order": "one_global_clip_before_parameter_groups",
            },
            parameter_groups={
                "pretrained_backbone_and_action_model": {
                    "peak_lr": 5e-5,
                    "decay_lr": 5e-6,
                },
                "new_modules": {
                    "subtrees": list(V4_NEW_PARAMETER_SUBTREES[args.arm]),
                    "active": bool(V4_NEW_PARAMETER_SUBTREES[args.arm]),
                    "peak_lr": 3e-4,
                    "decay_lr": 3e-5,
                },
            },
            ema_decay=0.999,
        )
    if v4_70k:
        # Declarative labels.  The executable retention is gpu/checkpoint_transport.retention_set /
        # finalize_success (ROBOMME_CHECKPOINT_MILESTONES / ROBOMME_SUCCESS_CHECKPOINT_MILESTONES) and
        # the entry's per-milestone deploy loop; both are asserted on the node.
        training_spec["recipe"] = V4_70K_RECIPE
        training_spec["checkpoint_policy"] = {
            "save_interval": SAVE_INTERVAL,
            "remote_resume": True,
            "upload_only_finalized_orbax": True,
            "training_retention": (
                "local_newest;remote_newest_plus_steps_" + "_".join(str(step) for step in milestones)
            ),
            "success_retention": [*milestones, final_step],
            "deploy_milestones": [*milestones, final_step],
            "deploy_layout": "deploy/<step>/{params,assets}+_DEPLOY_COMPLETE.json+tree_manifest",
        }
    if official_recipe_lerobot:
        data_spec.update(
            adapter="project_consolidated_lerobot_all16",
            official_custom_loader_parity_proven=False,
            policy_image_inputs=["base_0_rgb", "left_wrist_0_rgb"],
            masked_padding_image_inputs=[],
            policy_proprio_input=False,
            state_role="delta_action_transform_and_required_observation_structure_only",
        )
        training_spec.update(
            seed=42,
            batch_size=64,
            batch_unit="steps",
            effective_per_step_batch=64,
            max_token_len=64,
            full_finetune=False,
            freeze_filter=".*img.*",
            optimizer={
                "name": "AdamW",
                "b1": 0.9,
                "b2": 0.95,
                "eps": 1e-8,
                "weight_decay": 1e-10,
                "clip_gradient_norm": 1.0,
            },
            peak_lr=5e-5,
            warmup_steps=10_000,
            decay_steps=100_000,
            decay_lr=5e-5,
            schedule_semantics="10k_linear_warmup_then_constant_5e-5",
            ema_decay=0.999,
            checkpoint_policy={
                "save_interval": 10_000,
                "remote_resume": True,
                "upload_only_finalized_orbax": True,
                "training_retention": "local_newest;remote_newest_plus_steps_60000_70000",
                "success_retention": list(OFFICIAL_RECIPE_LEROBOT_MILESTONES),
            },
        )
    scientific = {
        "schema_version": 2,
        "study": STUDY,
        "benchmark": "RoboMME",
        "scope": (
            "multitask_recipe_diagnostic_v1"
            if official_recipe_lerobot
            else f"{args.scope}_v4"
            if args.arm in V4_ARM_IDS
            else f"{args.scope}_v1"
        ),
        "task": task_spec,
        "arm": args.arm,
        "mechanism": _arm_spec(args.arm),
        "initialization": initialization,
        "data": data_spec,
        "training": training_spec,
        "workspace_representation": workspace,
        "sources": {
            "robomme_integration": {
                "sanitized_source_tree_sha256": source_sha,
                "entry": ENTRY,
                "entry_sha256": entry_sha,
            },
            "openpi": {"uri": args.openpi_source_s3, "sha256": expected_openpi_sha},
            "tokenizer": {"uri": args.tokenizer_s3, "sha256": args.tokenizer_sha256},
            "image": {"uri": args.image_uri, "sha256": IMAGE_SHA},
        },
    }
    if args.arm in {"v4_wsm_cfg", "v4_cfg_jepa_visreg_l01_k1"}:
        scientific["evaluation_protocol"] = {
            "cfg_guidance_scales": [0.5, 1.0, 1.5, 2.0],
            "identity_rule": "each guidance scale is a distinct immutable evaluation cell",
        }
    if official_recipe_lerobot:
        scientific["reporting_contract"] = {
            "label": OFFICIAL_RECIPE_LEROBOT_LABEL,
            "human_label": "recipe-matched LeRobot diagnostic",
            "forbidden_claim": "exact official source/data reproduction",
        }
    scientific_sha = hashlib.sha256(_canonical_json(scientific).encode()).hexdigest()
    if official_recipe_lerobot:
        run_id = f"mt-diagnostic-v1-all16-{args.arm}-seed42-{scientific_sha[:16]}"
        output = f"{STUDY_ROOT}/checkpoints/robomme/pi05/diagnostics/{args.arm}/all16/seed42/{run_id}"
    elif args.arm in V4_ARM_IDS:
        run_id = (
            f"st-v4-{args.task.lower()}-{args.arm}-seed0-{scientific_sha[:16]}"
            if args.scope == "single_task"
            else f"mt-v4-70k-all16-{args.arm}-seed0-{scientific_sha[:16]}"
            if v4_70k
            else f"mt-v4-all16-{args.arm}-seed0-{scientific_sha[:16]}"
        )
        output = (
            f"{STUDY_ROOT}/checkpoints/robomme/pi05/single_task_v4/{args.task}/{args.arm}/seed0/{run_id}"
            if args.scope == "single_task"
            else f"{STUDY_ROOT}/checkpoints/robomme/pi05/multitask_v4/all16/{args.arm}/seed0/{run_id}"
        )
    else:
        run_id = (
            f"st-v1-{args.task.lower()}-{args.arm}-seed0-{scientific_sha[:16]}"
            if args.scope == "single_task"
            else f"mt-v1-all16-{args.arm}-seed0-{scientific_sha[:16]}"
        )
        output = (
            f"{STUDY_ROOT}/checkpoints/robomme/pi05/single_task_v1/{args.task}/{args.arm}/seed0/{run_id}"
            if args.scope == "single_task"
            else f"{STUDY_ROOT}/checkpoints/robomme/pi05/multitask_v1/all16/{args.arm}/seed0/{run_id}"
        )
    attempt_id = f"{run_id}-attempt{args.attempt_index}"
    manifest_s3 = f"{STUDY_ROOT}/manifests/runs/train/{run_id}/{attempt_id}.json"
    producer = f"{STUDY_ROOT}/manifests/claims/train/{run_id}/producers/{attempt_id}.json"
    completion = f"{STUDY_ROOT}/manifests/claims/train/{run_id}/step-{final_step}.complete.json"
    tree_root = (
        f"{STUDY_ROOT}/manifests/artifacts/checkpoints/{run_id}/scientific-milestones"
        if official_recipe_lerobot
        else f"{STUDY_ROOT}/manifests/artifacts/checkpoints/{run_id}/milestones"
        if v4_70k
        else f"{STUDY_ROOT}/manifests/artifacts/checkpoints/{run_id}/step-{final_step}"
    )
    infrastructure = {
        "provider": "aws_sagemaker",
        "execution_account": EXECUTION_ACCOUNT,
        "queue": args.queue,
        "training_plan_arn": plan_arn,
        "role": args.role,
        "instance_type": hardware["instance_type"],
        "accelerator": hardware["accelerator"],
        "priority": args.priority,
        "max_run_seconds": args.max_run_seconds,
        "volume_size_gb": args.volume_size_gb,
        "attempt_index": args.attempt_index,
        "attempts_in_job": 1,
    }
    manifest, manifest_json = _seal(
        {
            "schema_version": 2,
            "kind": "robomme_gpu_training_attempt",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "scientific_spec_sha256": scientific_sha,
            "scientific": scientific,
            "infrastructure": infrastructure,
            "output_s3": output,
            "manifest_s3": manifest_s3,
            "claims": {"producer": producer, "completion": completion},
            "checkpoint_tree_manifest_root": tree_root,
        }
    )
    environment = {
        "ROBOMME_ARM": args.arm,
        "ROBOMME_SCOPE": args.scope,
        "ROBOMME_RUN_ID": run_id,
        "ROBOMME_ATTEMPT_ID": attempt_id,
        "ROBOMME_SCIENTIFIC_SPEC_SHA256": scientific_sha,
        "ROBOMME_FINAL_STEP": str(final_step),
        "WSM_MAX_STEPS": str(train_steps),
        "WSM_SAVE_INTERVAL": "10000" if official_recipe_lerobot else str(SAVE_INTERVAL),
        "WSM_WARMUP_STEPS": (
            "10000"
            if official_recipe_lerobot
            else str(max(1, train_steps // 20))
            if args.arm in V4_ARM_IDS
            else "1000"
        ),
        "WSM_PEAK_LR": "5e-5",
        "WSM_DECAY_STEPS": "100000" if official_recipe_lerobot else str(train_steps),
        "WSM_DECAY_LR": "5e-5" if official_recipe_lerobot else "5e-6",
        "WSM_SEED": "42" if official_recipe_lerobot else "0",
        "ROBOMME_CHECKPOINT_MILESTONES": (
            "60000,70000"
            if official_recipe_lerobot
            else ",".join(str(step) for step in milestones)
            if v4_70k
            else str(train_steps // 2)
        ),
        "OPENPI_FORK_S3": args.openpi_source_s3,
        "OPENPI_REQUIRED_SENTINEL": (
            "_WSM_V4_ADVANCED"
            if args.arm in V4_ADVANCED_GDN_ARMS
            else "_WSM_PTRM"
            if args.arm == "ptrm"
            else "_ROBOMME_SEQUENCE_FORCING"
            if args.arm in {"q0_noforce", "q2_noforce"}
            else "_WSM_GDN_JEPA"
            if args.arm == "gdn8_jepa_l01_k1"
            else "_WSM_CFG_JEPA_V4"
            if args.arm == "v4_cfg_jepa_visreg_l01_k1"
            else "_WSM_HISTORY_DROPOUT"
            if args.arm in HISTORY_DROPOUT_ARMS
            else "_OFFICIAL_RECIPE_LEROBOT"
            if official_recipe_lerobot
            else ""
        ),
        "ROBOMME_DATA_S3": args.data_s3,
        "ROBOMME_DATA_PARENT_INVENTORY_S3": args.data_inventory_s3,
        "ROBOMME_DATA_PARENT_INVENTORY_SHA256": args.data_inventory_sha256,
        "INIT_S3": args.init_s3,
        "INIT_INVENTORY_S3": args.init_inventory_s3,
        "INIT_INVENTORY_SHA256": args.init_inventory_sha256,
        "PALIGEMMA_TOKENIZER_S3": args.tokenizer_s3,
        "PALIGEMMA_TOKENIZER_SHA256": args.tokenizer_sha256,
        "OUTPUT_S3": output,
        "RUN_MANIFEST_SOURCE": STAGED_MANIFEST,
        "RUN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "RUN_MANIFEST_S3": manifest_s3,
        "PRODUCER_CLAIM_S3": producer,
        "COMPLETION_CLAIM_S3": completion,
        "CHECKPOINT_TREE_MANIFEST_ROOT": tree_root,
    }
    if hardware["reserved_capacity"] is not None:
        environment["SM_USE_RESERVED_CAPACITY"] = hardware["reserved_capacity"]
    if official_recipe_lerobot:
        for name in ("INIT_S3", "INIT_INVENTORY_S3", "INIT_INVENTORY_SHA256"):
            environment.pop(name)
        environment.update(
            {
                "ROBOMME_RECIPE_LABEL": OFFICIAL_RECIPE_LEROBOT_LABEL,
                "ROBOMME_PI05_BASE_INIT_S3": args.pi05_base_init_s3,
                "ROBOMME_PI05_BASE_INIT_INVENTORY_S3": args.pi05_base_init_inventory_s3,
                "ROBOMME_PI05_BASE_INIT_INVENTORY_SHA256": base_sha,
                "ROBOMME_SUCCESS_CHECKPOINT_MILESTONES": "60000,70000",
            }
        )
    if v4_70k:
        environment.update(
            {
                "ROBOMME_RECIPE": V4_70K_RECIPE,
                "ROBOMME_SUCCESS_CHECKPOINT_MILESTONES": ",".join(str(step) for step in milestones),
            }
        )
    if args.scope == "single_task":
        environment.update(
            {
                "ROBOMME_TASK": args.task,
                "ROBOMME_DATA_DERIVED_INVENTORY_SHA256": task_inventory_sha,
            }
        )
    if workspace:
        if workspace.get("index"):
            environment.update(
                {
                    "ROBOMME_WORKSPACE_INDEX_S3": workspace["index"]["uri"],
                    "ROBOMME_WORKSPACE_INDEX_SHA256": workspace["index"]["sha256"],
                    "ROBOMME_REQUIRE_SUPERVISION": "1" if workspace["requires_supervision"] else "0",
                }
            )
        else:
            environment.update(
                {
                    "ROBOMME_WORKSPACE_S3": workspace["omega"]["uri"],
                    "ROBOMME_WORKSPACE_MANIFEST_SHA256": workspace["omega"]["manifest_sha256"],
                }
            )
            if workspace["supervision"]:
                environment.update(
                    {
                        "ROBOMME_SUPERVISION_S3": workspace["supervision"]["uri"],
                        "ROBOMME_SUPERVISION_MANIFEST_SHA256": workspace["supervision"]["manifest_sha256"],
                    }
                )
    oversized = {key: len(value.encode()) for key, value in environment.items() if len(value.encode()) > 512}
    if oversized:
        raise SystemExit(f"SageMaker environment values exceed 512 bytes: {oversized}")
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "output": output,
        "manifest_s3": manifest_s3,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "environment": environment,
        "source_sha": source_sha,
        "recipe": V4_70K_RECIPE if v4_70k else None,
        "deploy_milestones": [*milestones, final_step] if v4_70k else [final_step],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scope", choices=("single_task", "multitask"), default="single_task")
    value.add_argument("--task", choices=tuple(TASK_EPISODES))
    value.add_argument("--arm", required=True, choices=PRODUCTION_ARM_IDS)
    value.add_argument("--source-dir")
    value.add_argument("--data-s3", default=DATA_ROOT)
    value.add_argument("--data-inventory-s3", default=DATA_INVENTORY)
    value.add_argument("--data-inventory-sha256", default=CANONICAL_PARENT_SHA256)
    value.add_argument("--init-s3", default=INIT_ROOT)
    value.add_argument("--init-inventory-s3", default=INIT_INVENTORY)
    value.add_argument("--init-inventory-sha256", default=INIT_INVENTORY_SHA)
    value.add_argument("--pi05-base-init-s3")
    value.add_argument("--pi05-base-init-inventory-s3")
    value.add_argument("--pi05-base-init-inventory-sha256")
    value.add_argument(
        "--openpi-source-s3",
        help="advanced override; defaults to the arm-specific content-addressed OpenPI fork",
    )
    value.add_argument("--tokenizer-s3", default=TOKENIZER)
    value.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA)
    value.add_argument("--image-uri", default=IMAGE)
    value.add_argument("--workspace-encoder-id")
    value.add_argument("--workspace-s3")
    value.add_argument("--workspace-manifest-sha256")
    value.add_argument("--workspace-index-s3")
    value.add_argument("--workspace-index-sha256")
    value.add_argument("--supervision-s3")
    value.add_argument("--supervision-manifest-sha256")
    value.add_argument(
        "--hardware",
        choices=tuple(HARDWARE),
        default="p5e",
        help="p5e is training-only H200 capacity; p5 is H100 capacity usable for training or eval",
    )
    value.add_argument(
        "--queue",
        help="advanced override; must exactly match the queue pinned by --hardware",
    )
    value.add_argument("--role", default=ROLE_ARN)
    value.add_argument("--priority", type=int, default=PRIORITY)
    value.add_argument("--max-run-seconds", type=int, default=MAX_RUN_SECONDS)
    value.add_argument("--volume-size-gb", type=int, default=300)
    value.add_argument("--attempt-index", type=int, default=1)
    value.add_argument(
        "--multitask-train-steps",
        type=int,
        choices=MULTITASK_TRAIN_STEP_CHOICES,
        default=MULTITASK_TRAIN_STEPS,
        help=(
            "multitask v4 arms only: 60000 is the sealed recipe; 70000 selects the v4_70k "
            "checkpoint-maturity recipe (every 10k milestone retained and deployed)"
        ),
    )
    value.add_argument("--secrets-manager-arn")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--confirm-submit", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.queue is None:
        args.queue = HARDWARE[args.hardware]["queue"]
    source_dir = _source_dir(args.source_dir)
    plan = build_plan(args, source_dir)
    init_uri = args.pi05_base_init_s3 if args.arm == OFFICIAL_RECIPE_LEROBOT_ARM else args.init_s3
    print(
        f"scope={args.scope} task={args.task or 'all16'} arm={args.arm} "
        f"run_id={plan['run_id']} attempt={plan['attempt_id']}\n"
        f"  data_parent={args.data_inventory_s3}\n"
        f"  data_task_sha={plan['manifest']['scientific']['data']['derived_task_inventory_sha256']}\n"
        f"  initialization={init_uri}\n"
        f"  openpi={args.openpi_source_s3}\n"
        f"  output={plan['output']}\n"
        f"  manifest={plan['manifest_s3']} sha256={plan['manifest']['manifest_sha256']}\n"
        f"  queue={args.queue} plan={plan['manifest']['infrastructure']['training_plan_arn']}\n"
        f"  priority={args.priority} max_run={args.max_run_seconds}s volume={args.volume_size_gb}GiB "
        f"dry={args.dry_run}"
    )
    if plan["recipe"] is not None:
        print(
            f"  recipe={plan['recipe']} steps={plan['manifest']['scientific']['training']['steps']} "
            f"deploy_milestones={plan['deploy_milestones']}"
        )
    if args.dry_run:
        print(json.dumps(plan["manifest"], sort_keys=True, indent=2))
        print("DRY RUN ONLY — no AWS SDK loaded and no cloud write performed")
        return

    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    safe_arm = "official-lr" if args.arm == OFFICIAL_RECIPE_LEROBOT_ARM else args.arm.replace("_", "-")
    job_name = (f"sarvesh-rmme-{(args.task or 'all16')[:10]}-{safe_arm}-{plan['run_id'][-16:]}-{stamp}")[:63]
    if not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", job_name):
        raise SystemExit(f"invalid SageMaker TrainingJobName after normalization: {job_name}")
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=args.image_uri,
        instance_type=HARDWARE[args.hardware]["instance_type"],
        volume_size=args.volume_size_gb,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.benchmark", "Value": "RoboMME"},
            {"Key": "wsm.task", "Value": args.task or "all16"},
            {"Key": "wsm.arm", "Value": args.arm},
            {"Key": "wsm.run_id", "Value": plan["run_id"]},
        ],
        retry_config=RETRY,
        job_name=job_name,
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=args.secrets_manager_arn,
        confirmed=args.confirm_submit,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_sha"],
        staged_source_files={STAGED_MANIFEST: plan["manifest_json"] + "\n"},
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
