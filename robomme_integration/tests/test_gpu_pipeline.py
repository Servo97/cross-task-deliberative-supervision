from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from robomme_integration import launch
from robomme_integration.eval.launch_gpu_fleet import EVALUABLE_ARMS
from robomme_integration.eval.launch_gpu_fleet import main as gpu_eval_main
from robomme_integration.eval.run_local_fixed50_queue import (
    DEFAULT_SPEC as LOCAL_FIXED50_SPEC,
)
from robomme_integration.eval.run_local_fixed50_queue import _paths as local_eval_paths
from robomme_integration.eval.run_local_fixed50_queue import _read_spec as read_local_eval_spec
from robomme_integration.fleet.checkpoint import build as build_checkpoint_manifest
from robomme_integration.fleet.checkpoint import deploy_receipts_equivalent
from robomme_integration.fleet.checkpoint import verify as verify_checkpoint_manifest
from robomme_integration.fleet.task_inventory import (
    CANONICAL_TASK_DERIVED_SHA256,
    derive,
)
from robomme_integration.gpu.checkpoint_transport import CheckpointWatcher, S3Transport, tree_summary
from robomme_integration.sweeps.plan_p5_priority1 import DEFAULT_SPEC as P5_SWEEP_SPEC
from robomme_integration.sweeps.plan_p5_priority1 import _load as load_p5_sweep
from robomme_integration.sweeps.plan_p5_priority1 import expand as expand_p5_sweep
from robomme_integration.training.single_task import TASK_EPISODES

CANONICAL_PARENT = Path(
    "/tmp/robomme-inventory-v2/e77968b4c72c7589d92c1e85b1c6f7bf81aa49dd74472fb88dcead4277b5dad2.json"
)


def test_canonical_task_views_exactly_partition_parent_dataset():
    if not CANONICAL_PARENT.is_file():
        pytest.skip("canonical operational inventory is not staged")
    parent = json.loads(CANONICAL_PARENT.read_text())
    parent_data = {record["key"] for record in parent["objects"] if record["key"].startswith("data/")}
    union: set[str] = set()
    for task in TASK_EPISODES:
        view, digest = derive(CANONICAL_PARENT, task, root_s3=launch.DATA_ROOT)
        assert digest == CANONICAL_TASK_DERIVED_SHA256[task]
        task_data = {record["key"] for record in view["objects"] if record["key"].startswith("data/")}
        assert len(task_data) == 100
        assert union.isdisjoint(task_data)
        union.update(task_data)
    assert union == parent_data
    assert len(union) == 1600


def test_p5e_plan_preserves_scientific_arms_and_fail_closed_routing():
    source = Path(launch.__file__).resolve().parent
    s0_args = launch.parser().parse_args(["--task", "ButtonUnmaskSwap", "--arm", "s0", "--dry-run"])
    s0 = launch.build_plan(s0_args, source)
    manifest = s0["manifest"]
    assert manifest["infrastructure"] == {
        "provider": "aws_sagemaker",
        "execution_account": "141701954645",
        "queue": launch.TRAINING_PLAN_QUEUE,
        "training_plan_arn": "arn:aws:sagemaker:us-west-2:141701954645:training-plan/cam-robotics-tp",
        "role": launch.ROLE_ARN,
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "priority": 400,
        "max_run_seconds": 86400,
        "volume_size_gb": 300,
        "attempt_index": 1,
        "attempts_in_job": 1,
    }
    train = manifest["scientific"]["training"]
    assert (train["steps"], train["seed"], train["effective_per_step_batch"]) == (20000, 0, 64)
    assert (train["peak_lr"], train["decay_lr"], train["fsdp_devices"]) == (5e-5, 5e-6, 1)
    assert manifest["scientific"]["mechanism"]["robottt_fast_weights_W_t"] is False
    assert manifest["scientific"]["workspace_representation"] is None
    assert s0["environment"]["SM_USE_RESERVED_CAPACITY"] == "0"
    assert s0["environment"]["WSM_SAVE_INTERVAL"] == "5000"

    encoder = "a" * 64
    omega_sha = "b" * 64
    q3_args = launch.parser().parse_args(
        [
            "--task",
            "PickXtimes",
            "--arm",
            "q3",
            "--workspace-encoder-id",
            encoder,
            "--workspace-s3",
            f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/PickXtimes/{encoder}/omega",
            "--workspace-manifest-sha256",
            omega_sha,
            "--dry-run",
        ]
    )
    q3 = launch.build_plan(q3_args, source)["manifest"]["scientific"]
    assert q3["mechanism"]["robottt_fast_weights_W_t"] is True
    assert q3["mechanism"]["workspace_tokens_omega_t"] is True
    assert q3["mechanism"]["steering"] == "tanh"
    assert q3["workspace_representation"]["omega_symbol"] == "omega_t"

    invalid = launch.parser().parse_args(
        ["--task", "ButtonUnmaskSwap", "--arm", "s0", "--priority", "600", "--dry-run"]
    )
    with pytest.raises(SystemExit, match="priority 400"):
        launch.build_plan(invalid, source)
    with pytest.raises(SystemExit):
        launch.parser().parse_args(["--task", "ButtonUnmaskSwap", "--arm", "q2v", "--dry-run"])


def test_p5_and_p5e_change_infrastructure_not_scientific_identity():
    source = Path(launch.__file__).resolve().parent
    common = ["--task", "PickXtimes", "--arm", "s0", "--dry-run"]
    p5e = launch.build_plan(launch.parser().parse_args(common), source)
    p5 = launch.build_plan(launch.parser().parse_args([*common, "--hardware", "p5"]), source)

    assert p5["run_id"] == p5e["run_id"]
    assert p5["manifest"]["scientific_spec_sha256"] == p5e["manifest"]["scientific_spec_sha256"]
    assert p5["manifest"]["scientific"] == p5e["manifest"]["scientific"]
    assert p5["manifest"]["infrastructure"] == {
        "provider": "aws_sagemaker",
        "execution_account": "141701954645",
        "queue": launch.QUEUE,
        "training_plan_arn": None,
        "role": launch.ROLE_ARN,
        "instance_type": "ml.p5.48xlarge",
        "accelerator": "8xH100-80GB-HBM3",
        "priority": 400,
        "max_run_seconds": 86400,
        "volume_size_gb": 300,
        "attempt_index": 1,
        "attempts_in_job": 1,
    }
    assert p5["environment"]["SM_USE_RESERVED_CAPACITY"] == "1"
    assert p5e["environment"]["SM_USE_RESERVED_CAPACITY"] == "0"

    p5_backfill = launch.build_plan(
        launch.parser().parse_args([*common, "--hardware", "p5", "--priority", "1"]),
        source,
    )
    assert p5_backfill["run_id"] == p5["run_id"]
    assert p5_backfill["manifest"]["scientific"] == p5["manifest"]["scientific"]
    assert p5_backfill["manifest"]["infrastructure"]["priority"] == 1

    p5e_backfill = launch.parser().parse_args([*common, "--hardware", "p5e", "--priority", "1"])
    with pytest.raises(SystemExit, match="p5e training must use priority 400"):
        launch.build_plan(p5e_backfill, source)

    wrong_queue = launch.parser().parse_args([*common, "--hardware", "p5", "--queue", launch.TRAINING_PLAN_QUEUE])
    with pytest.raises(SystemExit, match="p5 training must use queue"):
        launch.build_plan(wrong_queue, source)


def test_new_memory_ports_are_fail_closed_and_source_capability_pinned():
    source = Path(launch.__file__).resolve().parent
    encoder = "a" * 64
    omega_sha = "b" * 64
    supervision_sha = "c" * 64
    root = f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/PickXtimes/{encoder}"
    common = [
        "--task",
        "PickXtimes",
        "--workspace-encoder-id",
        encoder,
        "--workspace-s3",
        f"{root}/omega",
        "--workspace-manifest-sha256",
        omega_sha,
        "--dry-run",
    ]

    q1 = launch.build_plan(launch.parser().parse_args([*common, "--arm", "q1"]), source)
    q1_mechanism = q1["manifest"]["scientific"]["mechanism"]
    assert q1_mechanism["sequence_windows"] is True
    assert q1_mechanism["robottt_fast_weights_W_t"] is False
    assert q1_mechanism["steering"] == "tanh"
    assert q1["manifest"]["scientific"]["sources"]["openpi"]["sha256"] == launch.OPENPI_SHA

    d16 = launch.build_plan(launch.parser().parse_args([*common, "--arm", "wsm_d16"]), source)
    assert d16["manifest"]["scientific"]["mechanism"]["steering"] == "gated_deltanet_k16"
    assert d16["manifest"]["scientific"]["sources"]["openpi"]["sha256"] == launch.OPENPI_SHA

    dropped = launch.build_plan(launch.parser().parse_args([*common, "--arm", "wsm_d16_drop05"]), source)
    dropped_scientific = dropped["manifest"]["scientific"]
    assert dropped_scientific["mechanism"]["steering"] == "gated_deltanet_k16"
    assert dropped_scientific["mechanism"]["train_history_dropout"] == 0.5
    assert dropped_scientific["sources"]["openpi"]["sha256"] == launch.PTRM_OPENPI_SHA
    assert dropped["environment"]["OPENPI_REQUIRED_SENTINEL"] == "_WSM_HISTORY_DROPOUT"

    combo = launch.build_plan(launch.parser().parse_args([*common, "--arm", "gdn8_jepa_l01_k1"]), source)
    combo_scientific = combo["manifest"]["scientific"]
    assert combo_scientific["sources"]["openpi"]["sha256"] == launch.OPENPI_SHA
    assert combo_scientific["mechanism"]["steering"] == "gated_deltanet_k8"
    assert combo_scientific["mechanism"]["jepa"] == {
        "lambda": 0.1,
        "futures": 1,
        "sigreg": 0.05,
    }
    assert combo_scientific["mechanism"]["openpi_overlay"] == {
        "kind": launch.gdn_jepa_overlay.OVERLAY_KIND,
        "version": launch.gdn_jepa_overlay.OVERLAY_VERSION,
        "manifest_sha256": launch.gdn_jepa_overlay._expected_manifest()["manifest_sha256"],
        "runtime_tree_sha256": launch.gdn_jepa_overlay.PATCHED_RUNTIME_TREE_SHA256,
        "base_archive_sha256": launch.OPENPI_SHA,
        "model_math_changed": False,
    }
    assert combo["environment"]["OPENPI_REQUIRED_SENTINEL"] == "_WSM_GDN_JEPA"

    with pytest.raises(SystemExit, match="supervision-manifest-sha256"):
        launch.build_plan(launch.parser().parse_args([*common, "--arm", "causal_v1"]), source)
    causal = launch.build_plan(
        launch.parser().parse_args(
            [
                *common,
                "--arm",
                "causal_v1",
                "--supervision-s3",
                f"{root}/supervision",
                "--supervision-manifest-sha256",
                supervision_sha,
            ]
        ),
        source,
    )
    causal_scientific = causal["manifest"]["scientific"]
    assert causal_scientific["mechanism"]["jepa"]["label_spec"] == "causal_v1"
    assert causal_scientific["workspace_representation"]["supervision"]["manifest_sha256"] == supervision_sha


def test_priority1_sweep_is_balanced_nonduplicative_and_omits_sealed_tpu_cells():
    spec = load_p5_sweep(P5_SWEEP_SPEC)
    jobs = expand_p5_sweep(spec)
    core_jobs = expand_p5_sweep(spec, "core")
    workspace_jobs = expand_p5_sweep(spec, "workspace")
    identities = {(record["task"], record["arm"]) for record in jobs}
    assert len(jobs) == len(identities) == 44
    assert len(core_jobs) == 28
    assert len(workspace_jobs) == 16
    assert core_jobs + workspace_jobs == jobs
    assert set(spec["core"]["tasks"]) == {
        "PickXtimes",
        "StopCube",
        "ButtonUnmaskSwap",
        "VideoUnmaskSwap",
        "PickHighlight",
        "VideoRepick",
        "MoveCube",
        "PatternLock",
    }
    assert {
        ("PickXtimes", "s0"),
        ("PickXtimes", "q0"),
        ("PickXtimes", "q2"),
        ("ButtonUnmaskSwap", "s0"),
    }.isdisjoint(identities)
    assert sum(record["arm"] == "salient" for record in jobs) == 2
    assert all(record.get("supervision_s3") for record in jobs if record["arm"] == "salient")


def test_multitask_all16_is_60k_and_workspace_arms_require_sealed_router_index():
    source = Path(launch.__file__).resolve().parent
    args = launch.parser().parse_args(["--scope", "multitask", "--arm", "s0", "--hardware", "p5", "--dry-run"])
    plan = launch.build_plan(args, source)
    scientific = plan["manifest"]["scientific"]
    assert scientific["scope"] == "multitask_v1"
    assert scientific["task"]["name"] == "all16"
    assert scientific["task"]["episodes"] == 1600
    assert scientific["training"]["steps"] == 60000
    assert scientific["data"]["derived_task_inventory_sha256"] is None
    assert plan["run_id"].startswith("mt-v1-all16-s0-seed0-")
    assert plan["environment"]["ROBOMME_FINAL_STEP"] == "59999"
    assert plan["environment"]["ROBOMME_CHECKPOINT_MILESTONES"] == "30000"
    assert "ROBOMME_TASK" not in plan["environment"]
    assert "ROBOMME_DATA_DERIVED_INVENTORY_SHA256" not in plan["environment"]
    entry = (source / "gpu_train_entry.sh").read_text()
    assert "--artifact robomme_lerobot_all16" in entry
    assert 'ROBOMME_COMPAT="$CODE_DIR/compat"' in entry
    assert 'PYTHONPATH="$ROBOMME_COMPAT:$CODE_DIR:$OPENPI/src"' in entry
    assert (source / "compat/robocasa/utils/groot_utils/groot_dataset.py").is_file()

    blocked = launch.parser().parse_args(["--scope", "multitask", "--arm", "wsm_cfg", "--dry-run"])
    with pytest.raises(SystemExit, match="workspace-index-s3"):
        launch.build_plan(blocked, source)

    index_sha = "a" * 64
    index_uri = f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/all16/{index_sha}.json"
    routed = launch.parser().parse_args(
        [
            "--scope",
            "multitask",
            "--arm",
            "wsm_cfg",
            "--workspace-index-s3",
            index_uri,
            "--workspace-index-sha256",
            index_sha,
            "--dry-run",
        ]
    )
    routed_plan = launch.build_plan(routed, source)
    representation = routed_plan["manifest"]["scientific"]["workspace_representation"]
    assert representation == {
        "index": {"uri": index_uri, "sha256": index_sha},
        "task_bound": False,
        "router": "pinned_episode_manifest_v1",
        "tasks": list(launch.TASK_ORDER),
        "omega_symbol": "omega_t",
        "requires_supervision": False,
    }
    assert routed_plan["environment"]["ROBOMME_WORKSPACE_INDEX_S3"] == index_uri
    assert routed_plan["environment"]["ROBOMME_WORKSPACE_INDEX_SHA256"] == index_sha
    assert routed_plan["environment"]["ROBOMME_REQUIRE_SUPERVISION"] == "0"
    assert "ROBOMME_TASK" not in routed_plan["environment"]

    ptrm = launch.parser().parse_args(
        [
            "--scope",
            "multitask",
            "--arm",
            "ptrm",
            "--workspace-index-s3",
            index_uri,
            "--workspace-index-sha256",
            index_sha,
            "--hardware",
            "p5",
            "--dry-run",
        ]
    )
    ptrm_plan = launch.build_plan(ptrm, source)
    ptrm_scientific = ptrm_plan["manifest"]["scientific"]
    assert ptrm_scientific["mechanism"]["ptrm"]["steps"] == 4
    assert ptrm_scientific["sources"]["openpi"] == {
        "uri": launch.PTRM_OPENPI,
        "sha256": launch.PTRM_OPENPI_SHA,
    }
    assert ptrm_plan["environment"]["OPENPI_REQUIRED_SENTINEL"] == "_WSM_PTRM"

    ambiguous = launch.parser().parse_args(
        ["--scope", "multitask", "--task", "PickXtimes", "--arm", "s0", "--dry-run"]
    )
    with pytest.raises(SystemExit, match="forbids --task"):
        launch.build_plan(ambiguous, source)


def test_official_recipe_lerobot_is_non_aliasing_all16_and_exactly_self_labels():
    source = Path(launch.__file__).resolve().parent
    base_sha = "d" * 64
    base_args = [
        "--scope",
        "multitask",
        "--arm",
        "official_recipe_lerobot",
        "--hardware",
        "p5",
        "--pi05-base-init-s3",
        launch.PI05_BASE_INIT_ROOT,
        "--pi05-base-init-inventory-s3",
        f"{launch.PI05_BASE_INIT_INVENTORY_ROOT}/{base_sha}.json",
        "--pi05-base-init-inventory-sha256",
        base_sha,
        "--dry-run",
    ]
    plan = launch.build_plan(launch.parser().parse_args(base_args), source)
    scientific = plan["manifest"]["scientific"]
    assert scientific["scope"] == "multitask_recipe_diagnostic_v1"
    assert scientific["mechanism"]["diagnostic"] == {
        "identity": "official_recipe_lerobot",
        "label": launch.OFFICIAL_RECIPE_LEROBOT_LABEL,
        "recipe_matched": True,
        "exact_official_source_reproduction": False,
        "exact_official_data_reproduction": False,
    }
    assert scientific["reporting_contract"]["forbidden_claim"] == ("exact official source/data reproduction")
    assert scientific["initialization"] == {
        "recipe": "published pi0.5 base",
        "artifact": "pi05_base_init",
        "checkpoint_s3": launch.PI05_BASE_INIT_ROOT,
        "inventory_uri": f"{launch.PI05_BASE_INIT_INVENTORY_ROOT}/{base_sha}.json",
        "inventory_sha256": base_sha,
    }
    training = scientific["training"]
    assert (
        training["steps"],
        training["seed"],
        training["batch_size"],
        training["max_token_len"],
        training["action_horizon"],
        training["ema_decay"],
    ) == (80_000, 42, 64, 64, 20, 0.999)
    assert (training["warmup_steps"], training["peak_lr"], training["decay_lr"]) == (
        10_000,
        5e-5,
        5e-5,
    )
    assert training["freeze_filter"] == ".*img.*"
    assert training["optimizer"]["weight_decay"] == 1e-10
    assert training["checkpoint_policy"]["success_retention"] == [60_000, 70_000, 79_999]
    assert scientific["data"]["policy_image_inputs"] == ["base_0_rgb", "left_wrist_0_rgb"]
    assert scientific["data"]["masked_padding_image_inputs"] == []
    assert scientific["data"]["policy_proprio_input"] is False
    environment = plan["environment"]
    assert not {"INIT_S3", "INIT_INVENTORY_S3", "INIT_INVENTORY_SHA256"}.intersection(environment)
    assert environment["ROBOMME_PI05_BASE_INIT_S3"] == launch.PI05_BASE_INIT_ROOT
    assert environment["OPENPI_REQUIRED_SENTINEL"] == "_OFFICIAL_RECIPE_LEROBOT"
    assert environment["ROBOMME_CHECKPOINT_MILESTONES"] == "60000,70000"
    assert environment["ROBOMME_SUCCESS_CHECKPOINT_MILESTONES"] == "60000,70000"
    assert "WSM_KEEP_PERIOD" not in environment
    assert plan["run_id"].startswith("mt-diagnostic-v1-all16-official_recipe_lerobot-seed42-")
    entry = (source / "gpu_train_entry.sh").read_text()
    assert "for step in 60000 70000 79999" in entry
    assert 'source="$WORK/scientific-step-$step"' in entry
    assert "robomme_gpu_diagnostic_checkpoint_set_complete" in entry

    missing_base = launch.parser().parse_args(
        ["--scope", "multitask", "--arm", "official_recipe_lerobot", "--dry-run"]
    )
    with pytest.raises(SystemExit, match="three distinct"):
        launch.build_plan(missing_base, source)
    single_task = launch.parser().parse_args(
        [
            "--task",
            "PickXtimes",
            "--arm",
            "official_recipe_lerobot",
            "--pi05-base-init-s3",
            launch.PI05_BASE_INIT_ROOT,
            "--pi05-base-init-inventory-s3",
            f"{launch.PI05_BASE_INIT_INVENTORY_ROOT}/{base_sha}.json",
            "--pi05-base-init-inventory-sha256",
            base_sha,
            "--dry-run",
        ]
    )
    with pytest.raises(SystemExit, match="all16-only"):
        launch.build_plan(single_task, source)

    # Training identity must never fall through a generic project evaluator.  A dedicated
    # two-view/all-16 evaluator is required before this checkpoint family can be scored.
    assert "official_recipe_lerobot" not in EVALUABLE_ARMS


def test_official_recipe_lerobot_is_rejected_by_single_task_eval_queue(tmp_path):
    spec = tmp_path / "invalid-official-single-task-eval.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "jobs": [
                    {
                        "task": "PickXtimes",
                        "arm": "official_recipe_lerobot",
                    }
                ],
            }
        )
    )
    with pytest.raises(SystemExit, match="unsupported local fixed-50 arm"):
        read_local_eval_spec(spec)


def test_finalized_checkpoint_retention_keeps_recovery_then_final_only(tmp_path):
    class MemoryTransport:
        run_root = "s3://bucket/run"

        def __init__(self):
            self.steps: dict[int, dict] = {}
            self.latest = None

        def upload_step(self, local_root, step):
            marker = {"schema_version": 1, "step": step, "tree": tree_summary(local_root / str(step))}
            self.steps[step] = marker
            return marker

        def publish_latest(self, step):
            assert step in self.steps
            self.latest = step

        def list_steps(self):
            return set(self.steps)

        def delete_step(self, step):
            del self.steps[step]

        def read_json(self, uri):
            step = int(uri.split("/steps/", 1)[1].split("/", 1)[0])
            return self.steps[step]

    local = tmp_path / "checkpoints"
    transport = MemoryTransport()
    watcher = CheckpointWatcher(
        local,
        transport,  # type: ignore[arg-type]
        tmp_path / "state.json",
        milestones={10000},
        final_step=19999,
    )
    for step in (5000, 10000, 15000, 19999):
        root = local / str(step)
        root.mkdir(parents=True)
        (root / "_CHECKPOINT_METADATA").write_text(f'{{"step":{step}}}\n')
        (root / "train_state").mkdir()
        (root / "train_state" / "state").write_bytes(bytes([step % 251]) * 17)
        (root / "params").mkdir()
        (root / "params" / "weights").write_bytes(bytes([step % 253]) * 19)
        (root / "assets").mkdir()
        (root / "assets" / "norm.json").write_text("{}\n")
        watcher.sync_once()
    assert transport.latest == 19999
    assert set(transport.steps) == {10000, 19999}
    watcher.finalize_success()
    assert transport.latest == 19999
    assert set(transport.steps) == {19999}
    assert json.loads((tmp_path / "state.json").read_text())["uploaded_steps"] == [19999]
    deploy_manifest = tmp_path / "deploy-tree.json"
    deploy_uri = "s3://bucket/run/steps/19999"
    digest = build_checkpoint_manifest(local / "19999", deploy_uri, deploy_manifest, workers=4)
    deployed = json.loads(deploy_manifest.read_text())
    assert {record["key"].split("/", 1)[0] for record in deployed["objects"]} == {
        "params",
        "assets",
    }
    assert verify_checkpoint_manifest(local / "19999", deploy_manifest, expected_uri=deploy_uri) == digest

    entry = (Path(launch.__file__).resolve().parent / "gpu_train_entry.sh").read_text()
    assert 'CHECKPOINT_URI="${OUTPUT_S3%/}/deploy/$ROBOMME_FINAL_STEP"' in entry
    assert 'aws s3 sync "$FINAL_DIR/params" "${CHECKPOINT_URI%/}/params"' in entry
    assert 'aws s3 sync "$FINAL_DIR/assets" "${CHECKPOINT_URI%/}/assets"' in entry
    assert '"checkpoint_uri": os.environ["CHECKPOINT_URI"]' in entry
    assert 'aws s3 rm "${OUTPUT_S3%/}/steps" --recursive' in entry


def test_scientific_milestones_are_required_and_retained_after_success(tmp_path):
    class MemoryTransport:
        run_root = "s3://bucket/run"

        def __init__(self):
            self.steps: dict[int, dict] = {}
            self.latest = None

        def upload_step(self, local_root, step):
            marker = {"schema_version": 1, "step": step, "tree": tree_summary(local_root / str(step))}
            self.steps[step] = marker
            return marker

        def publish_latest(self, step):
            assert step in self.steps
            self.latest = step

        def list_steps(self):
            return set(self.steps)

        def delete_step(self, step):
            del self.steps[step]

        def read_json(self, uri):
            step = int(uri.split("/steps/", 1)[1].split("/", 1)[0])
            return self.steps[step]

    local = tmp_path / "checkpoints"
    transport = MemoryTransport()
    watcher = CheckpointWatcher(
        local,
        transport,  # type: ignore[arg-type]
        tmp_path / "state.json",
        milestones={60_000, 70_000},
        final_step=79_999,
    )
    for step in (50_000, 60_000, 70_000, 79_999):
        root = local / str(step)
        root.mkdir(parents=True)
        (root / "_CHECKPOINT_METADATA").write_text(f'{{"step":{step}}}\n')
        watcher.sync_once()
    watcher.finalize_success(success_milestones={60_000, 70_000})
    assert transport.latest == 79_999
    assert set(transport.steps) == {60_000, 70_000, 79_999}
    assert json.loads((tmp_path / "state.json").read_text())["uploaded_steps"] == [
        60_000,
        70_000,
        79_999,
    ]

    incomplete = MemoryTransport()
    incomplete.steps = {79_999: transport.steps[79_999]}
    missing = CheckpointWatcher(
        local,
        incomplete,  # type: ignore[arg-type]
        tmp_path / "missing-state.json",
        milestones={60_000, 70_000},
        final_step=79_999,
    )
    with pytest.raises(RuntimeError, match="required scientific checkpoint milestones"):
        missing.finalize_success(success_milestones={60_000, 70_000})


def test_retry_receipt_ignores_only_attempt_scoped_provenance():
    first = {
        "schema_version": 1,
        "kind": "robomme_gpu_deploy_checkpoint_complete",
        "run_id": "same-run",
        "attempt_id": "same-run-attempt1",
        "step": 60_000,
        "checkpoint_uri": "s3://bucket/run/deploy/60000",
        "tree_manifest_uri": "s3://bucket/manifests/tree.json",
        "tree_manifest_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
    }
    retry = {
        **first,
        "attempt_id": "same-run-attempt2",
        # The attempt manifest includes attempt_id/infrastructure and therefore has a new seal.
        "run_manifest_sha256": "c" * 64,
    }
    assert deploy_receipts_equivalent(first, retry)
    assert not deploy_receipts_equivalent(
        first,
        {**retry, "tree_manifest_sha256": "d" * 64},
    )
    assert not deploy_receipts_equivalent(
        first,
        {**retry, "run_id": "different-run", "attempt_id": "different-run-attempt2"},
    )
    with pytest.raises(ValueError, match="attempt_id"):
        deploy_receipts_equivalent(
            first,
            {key: value for key, value in retry.items() if key != "attempt_id"},
        )
    with pytest.raises(ValueError, match="attempt_id"):
        deploy_receipts_equivalent(first, {**retry, "attempt_id": "different-run-attempt2"})
    with pytest.raises(ValueError, match="run-manifest"):
        deploy_receipts_equivalent(
            first,
            {key: value for key, value in retry.items() if key != "run_manifest_sha256"},
        )

    entry = (Path(launch.__file__).resolve().parent / "gpu_train_entry.sh").read_text()
    # The official-recipe, A19 v4_70k milestone, and legacy final-step paths all use semantic
    # create-once publication for deploy and completion receipts.  Mismatched scientific identities
    # are never overwritten.
    assert entry.count('publish_attempt_receipt_once "$COMPLETE" "$COMPLETION_CLAIM_S3"') == 3
    assert 'publish_attempt_receipt_once "$deploy_complete" "$deploy_complete_s3"' in entry
    assert 'publish_attempt_receipt_once "$DEPLOY_COMPLETE" "$DEPLOY_COMPLETE_S3"' in entry


def test_retry_plan_changes_attempt_manifest_but_not_scientific_run_identity():
    source = Path(launch.__file__).resolve().parent
    common = ["--task", "PickXtimes", "--arm", "s0", "--dry-run"]
    first = launch.build_plan(launch.parser().parse_args(common), source)
    retry = launch.build_plan(
        launch.parser().parse_args([*common, "--attempt-index", "2"]),
        source,
    )

    assert first["run_id"] == retry["run_id"]
    assert first["manifest"]["scientific_spec_sha256"] == retry["manifest"]["scientific_spec_sha256"]
    assert first["attempt_id"] != retry["attempt_id"]
    assert first["manifest"]["manifest_sha256"] != retry["manifest"]["manifest_sha256"]


def test_fresh_s3_run_skips_retry_backoff_when_latest_pointer_is_absent(tmp_path):
    calls = []
    sleeps = []

    def missing_pointer(command, **_kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=255, stdout="", stderr="404 Not Found")

    transport = S3Transport(
        "s3://bucket/run",
        runner=missing_pointer,
        sleeper=sleeps.append,
    )
    assert transport.restore_latest(tmp_path) is None
    assert len(calls) == 1
    assert calls[0][1:3] == ["s3api", "head-object"]
    assert sleeps == []


def test_local_fixed50_queue_exactly_covers_the_nine_sealed_single_task_finals(tmp_path):
    jobs = read_local_eval_spec(LOCAL_FIXED50_SPEC)
    identities = {(job["task"], job["arm"]) for job in jobs}
    assert len(jobs) == len(identities) == 9
    assert identities == {
        ("PickXtimes", "a6"),
        ("StopCube", "s0"),
        ("StopCube", "q0"),
        ("StopCube", "a6"),
        ("StopCube", "q2"),
        ("ButtonUnmaskSwap", "q0"),
        ("ButtonUnmaskSwap", "a6"),
        ("ButtonUnmaskSwap", "q2"),
        ("VideoUnmaskSwap", "s0"),
    }
    source = Path(launch.__file__).resolve().parent.parent
    plans = [local_eval_paths(tmp_path, source, job) for job in jobs]
    assert len({str(plan["output"]) for plan in plans}) == 9
    for job, plan in zip(jobs, plans, strict=True):
        config = Path(plan["config"])
        text = config.read_text()
        assert f"tasks: [{job['task']}]" in text
        assert "episodes_per_task: 50" in text
        assert "send_video_history: true" in text
        assert str(plan["eval_id"]).endswith("-fixed50-local5090-v1")


def test_local_fixed50_ready11_queue_covers_only_newly_sealed_cells(tmp_path):
    spec = LOCAL_FIXED50_SPEC.with_name("local_ready11_fixed50_v2.json")
    jobs = read_local_eval_spec(spec)
    identities = {(job["task"], job["arm"]) for job in jobs}
    assert len(jobs) == len(identities) == 11
    assert identities == {
        ("VideoUnmaskSwap", "q0"),
        ("VideoUnmaskSwap", "a6"),
        ("VideoUnmaskSwap", "q2"),
        ("PickHighlight", "s0"),
        ("PickHighlight", "q0"),
        ("PickHighlight", "a6"),
        ("PickHighlight", "q2"),
        ("VideoRepick", "s0"),
        ("VideoRepick", "q0"),
        ("VideoRepick", "q2"),
        ("MoveCube", "s0"),
    }
    assert {
        ("PickXtimes", "a6"),
        ("StopCube", "s0"),
        ("ButtonUnmaskSwap", "q0"),
        ("VideoUnmaskSwap", "s0"),
    }.isdisjoint(identities)
    source = Path(launch.__file__).resolve().parent.parent
    plans = [local_eval_paths(tmp_path, source, job) for job in jobs]
    assert len({str(plan["output"]) for plan in plans}) == 11
    for job, plan in zip(jobs, plans, strict=True):
        text = Path(plan["config"]).read_text()
        assert f"tasks: [{job['task']}]" in text
        assert "episodes_per_task: 50" in text
        assert "max_steps: 1300" in text
        assert "send_video_history: true" in text


def test_fixed50_gpu_eval_distributes_shards_across_four_policy_servers(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    server = source / "robomme_integration/eval/execution_model_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# sealed server\n")
    (source / "robomme_integration/compat/robocasa").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    config = tmp_path / "button.yaml"
    config.write_text("server: {url: 'ws://127.0.0.1:8000'}\n")
    venv = tmp_path / "eval-env" / "bin"
    venv.mkdir(parents=True)
    executable = venv / "vla-eval"
    executable.write_text("#!/bin/sh\n")
    help_text = " ".join(
        f"--args.{name}"
        for name in (
            "checkpoint",
            "arm",
            "task_name",
            "model_seed",
            "chunk_size",
            "max_batch_size",
        )
    )
    monkeypatch.setattr(
        "robomme_integration.eval.launch_gpu_fleet.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout=help_text),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_gpu_fleet.py",
            "--source-root",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--arm",
            "s0",
            "--task",
            "ButtonUnmaskSwap",
            "--benchmark-config",
            str(config),
            "--vla-eval",
            str(executable),
            "--output-root",
            str(tmp_path / "results"),
            "--eval-id",
            "button-s0-final50",
            "--dry-run",
        ],
    )
    assert gpu_eval_main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpus"] == [0, 1, 2, 3]
    assert payload["ports"] == [8000, 8001, 8002, 8003]
    assert payload["shards"] == 16
    launcher = payload["launcher_command"]
    urls = [launcher[index + 1] for index, value in enumerate(launcher) if value == "--server-url"]
    assert urls == [f"ws://127.0.0.1:{port}" for port in payload["ports"]]
    assert launcher[launcher.index("--container-pythonpath") + 1] == str(source)


def test_all16_gpu_eval_uses_one_multitask_checkpoint_and_server_identity(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    server = source / "robomme_integration/eval/execution_model_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# sealed server\n")
    (source / "robomme_integration/compat/robocasa").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    config = tmp_path / "all16.yaml"
    config.write_text("server: {url: 'ws://127.0.0.1:8000'}\n")
    executable = tmp_path / "eval-env" / "bin" / "vla-eval"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    help_text = " ".join(
        f"--args.{name}" for name in ("checkpoint", "arm", "task_name", "model_seed", "chunk_size", "max_batch_size")
    )
    monkeypatch.setattr(
        "robomme_integration.eval.launch_gpu_fleet.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout=help_text),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_gpu_fleet.py",
            "--scope",
            "multitask",
            "--native-simulator",
            "--simulator-pythonpath",
            str(source),
            "--simulator-gpus",
            "0,1",
            "--pin-native-cpus",
            "--gpus",
            "0,1",
            "--shards",
            "8",
            "--cpu-range",
            "0-127",
            "--xla-memory-fraction",
            "0.65",
            "--source-root",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--arm",
            "s0",
            "--benchmark-config",
            str(config),
            "--vla-eval",
            str(executable),
            "--output-root",
            str(tmp_path / "results"),
            "--eval-id",
            "all16-s0-fixed800",
            "--dry-run",
        ],
    )
    assert gpu_eval_main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "all16"
    assert payload["gpus"] == [0, 1]
    assert payload["shards"] == 8
    assert payload["xla_memory_fraction"] == 0.65
    assert str(source / "robomme_integration/compat") in payload["policy_pythonpath"]
    assert all("all16" in command for command in payload["server_commands"])
    assert "--no-docker" in payload["launcher_command"]
    assert "--container-image" not in payload["launcher_command"]
    assert payload["launcher_command"][payload["launcher_command"].index("--native-gpus") + 1] == "0,1"
    assert "--pin-native-cpus" in payload["launcher_command"]
