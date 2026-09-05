"""Offline contract tests for the focused pi0.5 Stage-S launcher."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_DIR))
import launch_guardrails as guardrails  # noqa: E402

from wsm_settings import ROBOCASA_OPENPI_ROOT  # noqa: E402

LAUNCHER_PATH = LAUNCH_DIR / "submit_pi_stage_s.py"
SPEC = importlib.util.spec_from_file_location("submit_pi_stage_s_test", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
stage_s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_s)


W_SHA = "a" * 64
O_SHA = "b" * 64
ENCODER_ID = "c" * 64
FEATURE_SHA = "d" * 64
IMAGE_SHA = "e" * 64
INIT_INVENTORY_SHA = "1" * 64
TARGET_INVENTORY_SHA = "2" * 64
TASK_PROMPT_SHA = "3" * 64
TOKENIZER_SHA = "4" * 64


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "internal_training"
    source.mkdir()
    entry = source / stage_s.ENTRY
    entry.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    entry.chmod(0o755)
    (source / "support.sh").write_text("support=v1\n", encoding="utf-8")
    return source


#: The fork gate whose absence killed audit discriminator D1 on 2026-08-07: wsmv2 522ad4b0 read it,
#: the pinned n-wave openpi 768f274a predated it, and the run died at import on a started node.
COMBO_GATE = "_WSM_JEPA_WITH_WINDOW"


def _archive_cache(
    tmp_path: Path,
    *,
    reads: tuple[str, ...] = (COMBO_GATE, "_WSM_K_WINDOW"),
    defines: tuple[str, ...] | None = None,
    sentinels: tuple[str, ...] = (),
    wsmv2_sha: str = W_SHA,
    openpi_sha: str = O_SHA,
) -> Path:
    """A cache dir holding an extracted wsmv2 tree and openpi fork, shaped like the real archives.

    `reads` are the ``_groot_dataset.<attr>`` lookups the wsmv2 trainer performs; `defines` are the
    module-level gates the fork's DATALOADER binds (defaulting to exactly `reads`, i.e. a compatible
    pair); `sentinels` are the architecture flags the fork's conditioner module binds, which is where
    a recipe-required name like ``_WSM_PTRM`` lives.
    """
    cache = tmp_path / "archive-cache"
    consumer = cache / wsmv2_sha / stage_s._FORK_CONSUMER_SUBTREE / "train" / "train_base"
    consumer.mkdir(parents=True)
    body = "\n".join(f"value_{index} = _groot_dataset.{name}" for index, name in enumerate(reads))
    (consumer / "_pi05_common.py").write_text(
        "import openpi.groot_utils.groot_openpi_dataset as _groot_dataset\n" + body + "\n",
        encoding="utf-8",
    )
    dataloader = cache / openpi_sha / stage_s._FORK_DATALOADER_RELPATH
    dataloader.parent.mkdir(parents=True)
    bound = reads if defines is None else defines
    (dataloader).write_text(
        "import os\n" + "\n".join(f'{name} = os.environ.get("{name}", "0") == "1"' for name in bound) + "\n",
        encoding="utf-8",
    )
    if sentinels:
        module = cache / openpi_sha / stage_s._FORK_SENTINEL_RELPATHS[0]
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("\n".join(f"{name} = True" for name in sentinels) + "\n", encoding="utf-8")
    return cache


def _args(arm: str, source: Path, *extra: str):
    root = stage_s.study_root(stage_s.DEFAULT_OWNER)
    argv = [
        "--dry-run",
        "--arm",
        arm,
        "--source-dir",
        str(source),
        "--wsmv2-source-s3",
        f"{root}/code/wsmv2/{W_SHA}.tgz",
        "--openpi-source-s3",
        f"{root}/code/openpi/{O_SHA}.tgz",
        "--tokenizer-s3",
        f"{root}/artifacts/tokenizers/paligemma/{TOKENIZER_SHA}.model",
        "--tokenizer-sha256",
        TOKENIZER_SHA,
        "--init-inventory-s3",
        f"{root}/manifests/inventories/init/{INIT_INVENTORY_SHA}.json",
        "--init-inventory-sha256",
        INIT_INVENTORY_SHA,
        "--target-inventory-s3",
        f"{root}/manifests/inventories/data/{TARGET_INVENTORY_SHA}.json",
        "--target-inventory-sha256",
        TARGET_INVENTORY_SHA,
        "--image-uri",
        (f"141701954645.dkr.ecr.us-west-2.amazonaws.com/sarvesh.patil-groot-dexjoco@sha256:{IMAGE_SHA}"),
        *extra,
    ]
    return stage_s.make_parser().parse_args(argv)


def _command(args) -> list[str]:
    command = [sys.executable, str(LAUNCHER_PATH)]
    for key, value in vars(args).items():
        if key in {"dry_run", "confirm_submit"} or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def _workspace_extra() -> tuple[str, ...]:
    root = stage_s.study_root(stage_s.DEFAULT_OWNER)
    return (
        "--encoder-id",
        ENCODER_ID,
        "--policy-features-s3",
        f"{root}/caches/{ENCODER_ID}/omega",
        "--policy-features-manifest-s3",
        f"{root}/manifests/artifacts/workspace/{ENCODER_ID}/omega/{FEATURE_SHA}.json",
        "--policy-features-manifest-sha256",
        FEATURE_SHA,
        "--task-prompt-manifest-s3",
        (f"{root}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{TASK_PROMPT_SHA}.json"),
        "--task-prompt-manifest-sha256",
        TASK_PROMPT_SHA,
    )


def test_s0_plan_is_deterministic_recipe_matched_and_workspace_free(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    first = stage_s.build_plan(args, source)
    second = stage_s.build_plan(args, source)

    assert first == second
    assert first["manifest"]["interface"] == "base"
    assert first["manifest"]["run_kind"] == "train"
    assert first["manifest"]["training"]["steps"] == 60000
    assert first["manifest"]["training"]["batch_size"] == 64
    assert first["manifest"]["training"]["per_device_batch_size"] == 8
    assert first["manifest"]["training"]["jax_devices"] == 8
    assert first["manifest"]["training"]["jax_processes"] == 1
    assert first["manifest"]["training"]["fsdp_devices"] == 1
    assert first["manifest"]["training"]["data_parallel_replicas"] == 8
    assert first["manifest"]["training"]["num_workers"] == 32
    assert first["manifest"]["training"]["image_resize_path"] == "worker_pil_bilinear"
    assert first["manifest"]["training"]["defer_image_resize_to_model_preprocess"] is False
    assert first["manifest"]["training"]["param_norm_metric"] == "log_boundary_post_update"
    checkpoint_policy = first["manifest"]["training"]["checkpoint_policy"]
    assert checkpoint_policy["retained_steps"] == [59999]
    assert checkpoint_policy["midrun_sync"] is False
    assert checkpoint_policy["resume"] is False
    assert first["manifest"]["infrastructure"]["attempts"] == 1
    assert first["manifest"]["infrastructure"]["attempt_index"] == 1
    assert first["manifest"]["infrastructure"]["aggregate_max_run_seconds"] == 432000
    assert first["manifest"]["data"]["tasks"] == 50
    assert first["manifest"]["data"]["demos_per_task"] == 150
    assert first["environment"]["WSM_FT_CONFIG"] == stage_s.BASE_CONFIG
    assert first["environment"]["WSM_FINAL_ONLY_CHECKPOINTS"] == "1"
    assert first["environment"]["WSM_EXPECTED_JAX_DEVICES"] == "8"
    assert first["environment"]["WSM_EXPECTED_JAX_PROCESSES"] == "1"
    assert first["environment"]["WSM_EXPECTED_GLOBAL_BATCH"] == "64"
    assert first["environment"]["WSM_EXPECTED_NUM_WORKERS"] == "32"
    assert first["environment"]["WSM_EXPECTED_FSDP_DEVICES"] == "1"
    assert first["environment"]["OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS"] == "0"
    assert first["environment"]["INIT_INVENTORY_SHA256"] == INIT_INVENTORY_SHA
    assert first["environment"]["TARGET_INVENTORY_SHA256"] == TARGET_INVENTORY_SHA
    assert first["environment"]["PALIGEMMA_TOKENIZER_SHA256"] == TOKENIZER_SHA
    assert first["manifest"]["sources"]["tokenizer"]["sha256"] == TOKENIZER_SHA
    assert "/manifests/runs/train/" in first["manifest_s3"]
    assert "/manifests/claims/train/" in first["environment"]["PRODUCER_CLAIM_S3"]
    assert "POLICY_FEATS_S3" not in first["environment"]
    assert "WSM_DEMO_CACHE_SIZE" not in first["environment"]
    assert "WSM_DEMO_CACHE_MAX_BYTES" not in first["environment"]
    assert first["manifest"]["workspace_representation"]["policy_transport"] is None
    assert "WSM_CFG" not in first["environment"]
    assert "WSM_LEGACY_TOKEN_INJECTION" not in first["environment"]
    assert first["output_s3"].startswith(stage_s.study_root(stage_s.DEFAULT_OWNER) + "/checkpoints/pi05/s0/")
    source_sha = first["manifest"]["sources"]["internal_training"]["sanitized_source_tree_sha256"]
    assert source_sha == first["source_tree_sha256"]
    assert len(source_sha) == 64

    args.attempt_index = 2
    retry = stage_s.build_plan(args, source)
    assert retry["run_id"] != first["run_id"]
    assert retry["manifest"]["infrastructure"]["attempt_index"] == 2

    args.attempt_index = 0
    with pytest.raises(SystemExit, match="positive integer"):
        stage_s.build_plan(args, source)


@pytest.mark.parametrize(
    ("arm", "interface", "knob"),
    (("s1", "tanh", "WSM_TANH_GATE_INIT"), ("s2", "cfg2", "WSM_CFG_P_DROP")),
)
def test_workspace_arms_pin_encoder_and_choose_only_new_interface(tmp_path, arm, interface, knob):
    source = _source(tmp_path)
    args = _args(arm, source, *_workspace_extra())
    plan = stage_s.build_plan(args, source)
    env = plan["environment"]

    assert plan["manifest"]["interface"] == interface
    assert env["PI_STAGE_S_INTERFACE"] == interface
    assert env["WSM_FT_CONFIG"] == stage_s.WORKSPACE_CONFIG
    assert env["WSM_ENCODER_ID"] == ENCODER_ID
    assert env["POLICY_FEATS_MANIFEST_SHA256"] == FEATURE_SHA
    assert env["POLICY_FEATS_MANIFEST_S3"].endswith(f"/omega/{FEATURE_SHA}.json")
    assert env["TASK_PROMPT_MANIFEST_SHA256"] == TASK_PROMPT_SHA
    assert plan["manifest"]["workspace_representation"]["required_global_language_mode"] == (
        "canonical_terse_task_instruction"
    )
    assert env["WSM_K_WINDOW"] == "1"
    assert plan["manifest"]["training"]["workspace_window"] == 1
    transport = plan["manifest"]["workspace_representation"]["policy_transport"]
    assert transport == {
        "cached_omega_dtype": "float16",
        "selected_omega_dtype": "float32",
        "workspace_language_transport": "omitted_current_only",
        "cache_policy": "deterministic_lru",
        "cache_max_items_per_worker": 8192,
        "cache_max_payload_bytes_per_worker": 512 * 1024**2,
    }
    assert env["WSM_DEMO_CACHE_SIZE"] == "8192"
    assert env["WSM_DEMO_CACHE_MAX_BYTES"] == str(512 * 1024**2)
    assert env[knob]
    for forbidden in ("WSM_CFG", "WSM_LEGACY_TOKEN_INJECTION", "WSM_CFG_WITH_FUTURE"):
        assert forbidden not in env


def test_one_step_canary_has_separate_identity_and_strict_runtime(tmp_path):
    source = _source(tmp_path)
    args = _args(
        "s0",
        source,
        "--canary",
        "--priority",
        "1",
        "--max-run-seconds",
        str(stage_s.MAX_CANARY_RUN_SECONDS),
    )
    plan = stage_s.build_plan(args, source)
    assert plan["manifest"]["run_kind"] == "canary"
    assert plan["manifest"]["training"]["steps"] == 1
    assert plan["manifest"]["training"]["checkpoint_policy"]["retained_steps"] == [0]
    assert plan["run_id"].startswith("s0-canary-")
    assert "/canaries/training/pi05/s0/" in plan["output_s3"]
    assert "/manifests/runs/canary/" in plan["manifest_s3"]
    assert plan["environment"]["STAGE_S_RUN_KIND"] == "canary"
    assert plan["environment"]["STAGE_S_FINAL_STEP"] == "0"

    args.max_run_seconds += 1
    with pytest.raises(SystemExit, match="max-run-seconds"):
        stage_s.build_plan(args, source)
    args.max_run_seconds = stage_s.MAX_CANARY_RUN_SECONDS
    args.priority = 600
    with pytest.raises(SystemExit, match="priority 1"):
        stage_s.build_plan(args, source)


def test_full_sanitized_source_tree_affects_identity(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    first = stage_s.build_plan(args, source)
    (source / "support.sh").write_text("support=v2\n", encoding="utf-8")
    second = stage_s.build_plan(args, source)
    assert first["source_tree_sha256"] != second["source_tree_sha256"]
    assert first["run_id"] != second["run_id"]


def test_unpinned_or_noncanonical_artifacts_fail_closed(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    args.image_uri = "141701954645.dkr.ecr.us-west-2.amazonaws.com/repo:latest"
    with pytest.raises(SystemExit, match="pinned"):
        stage_s.build_plan(args, source)

    args = _args("s0", source)
    args.wsmv2_source_s3 = "s3://somewhere/code/wsmv2.tgz"
    with pytest.raises(SystemExit, match="content-addressed"):
        stage_s.build_plan(args, source)

    args = _args("s1", source)
    with pytest.raises(SystemExit, match="require --encoder-id"):
        stage_s.build_plan(args, source)

    args = _args("s0", source)
    args.target_inventory_s3 = args.target_inventory_s3.replace("/manifests/inventories/data/", "/inventories/")
    with pytest.raises(SystemExit, match="content-addressed"):
        stage_s.build_plan(args, source)


def test_s0_rejects_workspace_provenance_and_s1_rejects_mismatched_cache(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(SystemExit, match="S0 forbids"):
        stage_s.build_plan(_args("s0", source, *_workspace_extra()), source)

    args = _args("s1", source, *_workspace_extra())
    args.policy_features_s3 += "-wrong"
    with pytest.raises(SystemExit, match="match encoder provenance"):
        stage_s.build_plan(args, source)


def test_cli_dry_run_is_offline(tmp_path):
    source = _source(tmp_path)
    args = _args("s1", source, *_workspace_extra())
    command = [*_command(args), "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "DRY RUN: offline" in result.stdout
    assert "studies/long_context_v1" in result.stdout
    assert "SUBMISSION READY" in result.stdout


def test_manifest_is_staged_not_put_in_environment(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    plan = stage_s.build_plan(args, source)
    assert "RUN_MANIFEST_JSON" not in plan["environment"]
    assert plan["environment"]["RUN_MANIFEST_SOURCE"] == stage_s.STAGED_MANIFEST_NAME
    assert max(len(value.encode("utf-8")) for value in plan["environment"].values()) <= 512

    with guardrails.prepared_source_bundle(source, stage_s.ENTRY, {"SAGEMAKER_PROGRAM": stage_s.ENTRY}) as (
        staged,
        _entry,
        _environment,
    ):
        before = guardrails.source_tree_sha256(staged)
        guardrails.write_staged_source_files(staged, {stage_s.STAGED_MANIFEST_NAME: plan["manifest_json"] + "\n"})
        assert (staged / stage_s.STAGED_MANIFEST_NAME).read_text(encoding="utf-8") == (plan["manifest_json"] + "\n")
        assert guardrails.source_tree_sha256(staged) != before

    with pytest.raises(SystemExit, match="unsafe staged source path"):
        guardrails.write_staged_source_files(tmp_path, {"../escape.json": "{}"})


def test_external_entry_stage_s_contract_is_syntax_valid():
    entry = Path(__file__).resolve().parents[2] / "internal_training" / stage_s.ENTRY
    if not entry.exists():
        pytest.skip("TRI internal_training sibling checkout is not present")
    source = entry.read_text(encoding="utf-8")
    assert 'FORK_S3="${OPENPI_FORK_S3:-' in source
    assert "download_verified_archive" in source
    assert "validate_stage_s_policy_features.py" in source
    assert "materialize_stage_s_inventory.py" in source
    assert "build_stage_s_checkpoint_manifest.py" in source
    assert "finetune_pi_05_with_workspace.py" in source
    assert 'STAGE_S_ARGS=(--interface "$STAGE_S_INTERFACE")' in source
    assert 'CKPT_NAME="pi05_robocasa_workspace_stage_s"' in source
    assert "POLICY_FEATS_MANIFEST_SHA256" in source
    assert "RUN_MANIFEST_SOURCE" in source
    assert "RUN_MANIFEST_JSON" not in source
    assert "publish_manifest_once" in source
    assert "PRODUCER_CLAIM_S3" in source
    assert "COMPLETION_CLAIM_S3" in source
    assert "manifests/artifacts/checkpoints" in source
    assert "WSM_FINAL_ONLY_CHECKPOINTS" in source
    assert '--exclude "*" --include "params/*" --include "assets/*"' in source
    assert "optimizer train_state/ never leaves the node" in source
    assert 'if [[ -z "$STAGE_S_INTERFACE"' in source
    assert "--if-none-match '*'" in source
    assert "Stage-S checkpoint leaf mismatch" in source
    assert "download_verified_tokenizer" in source
    assert "STAGE_S_RUN_KIND" in source
    result = subprocess.run(["bash", "-n", str(entry)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_confirm_submit_calls_guarded_submission_once(tmp_path, monkeypatch, capsys):
    source = _source(tmp_path)
    args = _args("s0", source)
    captured = {}

    def fake_submit_training_job(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(job_arn="arn:aws:batch:us-west-2:141:service-job/test")]

    monkeypatch.setattr(stage_s, "submit_training_job", fake_submit_training_job)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *_command(args)[1:],
            "--confirm-submit",
            "--archive-cache-dir",
            str(_archive_cache(tmp_path)),
        ],
    )
    stage_s.main()

    assert captured["queue"] == guardrails.QUEUE
    assert captured["priority"] == guardrails.MULTI_DAY_PRIORITY
    assert captured["retry_config"] == {"attempts": 1}
    assert captured["confirmed"] is True
    assert captured["staged_source_files"][stage_s.STAGED_MANIFEST_NAME].endswith("\n")
    assert "QUEUED arn=" in capsys.readouterr().out


def test_config_override_cond_window_is_sealed_into_env_and_manifest(tmp_path):
    # The deltanet recipe trains at window 8; the launcher must read model.cond_window from the
    # override yaml so WSM_K_WINDOW and manifest.training.workspace_window tell the truth.
    source = _source(tmp_path)
    args = _args("s1", source, *_workspace_extra())
    args.config_override = "scripts/configs/train/pi05_stage_s1_deltanet_finetune.yaml"
    plan = stage_s.build_plan(args, source)
    assert plan["environment"]["WSM_K_WINDOW"] == "8"
    assert plan["manifest"]["training"]["workspace_window"] == 8
    assert plan["manifest"]["training"]["config"] == args.config_override

    # Arm-default configs (no override) stay pinned at window 1.
    base_args = _args("s1", source, *_workspace_extra())
    base_plan = stage_s.build_plan(base_args, source)
    assert base_plan["environment"]["WSM_K_WINDOW"] == "1"
    assert base_plan["manifest"]["training"]["workspace_window"] == 1


# --------------------------------------------------------------------------------------------
# Dataset profiles: the target dataset is a launch PARAMETER, and the RoboCasa default is inert.
# --------------------------------------------------------------------------------------------
RMB_CONFIGS = {
    "s0": "scripts/configs/train/pi05_rmb_base_finetune.yaml",
    "s1": "scripts/configs/train/pi05_rmb_tanh_finetune.yaml",
    "s1-deltanet": "scripts/configs/train/pi05_rmb_deltanet_finetune.yaml",
    "s3": "scripts/configs/train/pi05_rmb_jw01k16_finetune.yaml",
}


def _remembench_workspace_extra() -> tuple[str, ...]:
    """Same as _workspace_extra but under the ReMemBench prompt namespace."""
    return tuple(
        value.replace("task_prompts/robocasa_target50/", "task_prompts/remembench13/") for value in _workspace_extra()
    )


def test_default_dataset_profile_leaves_the_robocasa_plan_unchanged(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    assert args.dataset_profile == stage_s.DEFAULT_DATASET_PROFILE == "robocasa_target50"
    plan = stage_s.build_plan(args, source)
    assert plan["manifest"]["data"]["benchmark"] == "RoboCasa"
    assert plan["manifest"]["data"]["tasks"] == 50
    assert plan["manifest"]["data"]["demos_per_task"] == 150
    assert plan["manifest"]["data"]["episode_subsample_seed"] == 0
    assert plan["manifest"]["data"]["inventory"]["artifact"] == "robocasa_target50"
    assert plan["environment"]["TARGET_DATA_S3"] == stage_s.TARGET_DATA_S3
    # The profile contributes NO entry env, so no dataset-shape knob is emitted at all.
    assert stage_s.DATASET_PROFILES["robocasa_target50"]["entry_env"] == {}
    for knob in (
        "TARGET_INVENTORY_ARTIFACT",
        "TARGET_ROOT_SUBDIR",
        "TARGET_EXPECTED_TASKS",
        "TARGET_TASK_DIR_GLOBS",
        "TASK_PROMPT_NAMESPACE",
        "TASK_PROMPT_ARTIFACT",
        "POLICY_FEATS_DATASET_NAME",
        "POLICY_FEATS_DEMOS_PER_TASK_MAP",
        "OMEGA_EXPECTED_FILES",
        "STAGE_S_EXPECTED_TRAIN_STEPS",
    ):
        assert knob not in plan["environment"], knob


@pytest.mark.parametrize("arm", ["s0", "s1", "s1-deltanet", "s3"])
def test_remembench_profile_submits_every_rmb_arm(tmp_path, arm):
    source = _source(tmp_path)
    real_arm = arm.split("-")[0]
    extra = () if real_arm == "s0" else _remembench_workspace_extra()
    args = _args(real_arm, source, "--dataset-profile", "remembench_v02_train13", "--train-steps", "15000", *extra)
    args.config_override = RMB_CONFIGS[arm]
    plan = stage_s.build_plan(args, source)
    env, manifest = plan["environment"], plan["manifest"]

    assert manifest["training"]["config"] == RMB_CONFIGS[arm]
    assert manifest["training"]["steps"] == 15000
    assert manifest["training"]["checkpoint_policy"]["retained_steps"] == [14999]
    assert env["WSM_MAX_STEPS"] == "15000" and env["STAGE_S_FINAL_STEP"] == "14999"
    assert env["STAGE_S_EXPECTED_TRAIN_STEPS"] == "15000"

    assert manifest["data"]["benchmark"] == "ReMemBench"
    assert manifest["data"]["tasks"] == 13
    assert manifest["data"]["demos_per_task"] == "all"
    assert manifest["data"]["episode_subsample_seed"] is None
    assert manifest["data"]["target_fraction_per_task"] is None
    assert manifest["data"]["inventory"]["artifact"] == "remembench_train13"
    assert manifest["data"]["dataset_s3"] == stage_s.REMEMBENCH_DATA_S3

    assert env["TARGET_DATA_S3"] == stage_s.REMEMBENCH_DATA_S3
    assert env["TARGET_INVENTORY_ARTIFACT"] == "remembench_train13"
    assert env["TARGET_ROOT_SUBDIR"] == "v1.0/target/train"
    assert env["TARGET_EXPECTED_TASKS"] == "13"
    assert env["TARGET_TASK_DIR_GLOBS"] == "*/*/lerobot"
    assert env["TASK_PROMPT_NAMESPACE"] == "remembench13"
    assert env["TASK_PROMPT_ARTIFACT"] == "remembench_train13_task_prompts"
    assert env["POLICY_FEATS_DATASET_NAME"] == "remembench_v02_train13"
    assert env["OMEGA_EXPECTED_FILES"] == "323"
    demos_map = Path(__file__).resolve().parents[1] / env["POLICY_FEATS_DEMOS_PER_TASK_MAP"]
    assert demos_map.is_file()
    counts = __import__("json").loads(demos_map.read_text(encoding="utf-8"))
    assert len(counts) == 13 and sum(counts.values()) == 323

    if real_arm == "s0":
        assert "POLICY_FEATS_S3" not in env
        assert manifest["workspace_representation"]["expected_tasks"] is None
    else:
        assert manifest["workspace_representation"]["expected_tasks"] == 13
        assert manifest["workspace_representation"]["expected_episodes_per_task"] is None
        assert "task_prompts/remembench13/" in env["TASK_PROMPT_MANIFEST_S3"]
    # Only the deltanet recipe widens the omega window.
    assert env.get("WSM_K_WINDOW") == ("8" if arm == "s1-deltanet" else ("1" if extra else None))
    assert max(len(value.encode("utf-8")) for value in env.values()) <= 512


def test_remembench_profile_rejects_the_robocasa_prompt_namespace(tmp_path):
    source = _source(tmp_path)
    args = _args(
        "s1", source, "--dataset-profile", "remembench_v02_train13", *_workspace_extra()
    )  # robocasa_target50 namespace
    with pytest.raises(SystemExit, match="task_prompts/remembench13/"):
        stage_s.build_plan(args, source)


# --------------------------------------------------------------------------------------------
# Plan-backed queues: the job must pin the flexible training plan or it never leaves SCHEDULED.
# --------------------------------------------------------------------------------------------


def test_plan_queue_pins_the_training_plan_and_drops_implicit_reserved_capacity(tmp_path):
    source = _source(tmp_path)
    args = _args(
        "s0", source, "--queue", guardrails.TRAINING_PLAN_QUEUE, "--priority", "400", "--max-run-seconds", "86400"
    )
    plan = stage_s.build_plan(args, source)
    expected = guardrails.TRAINING_PLAN_ARNS[guardrails.TRAINING_PLAN_QUEUE]
    assert plan["manifest"]["infrastructure"]["training_plan_arn"] == expected
    assert plan["manifest"]["infrastructure"]["instance_type"] == "ml.p5e.48xlarge"
    # The pinned plan REPLACES the implicit request; the two are alternatives.
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "0"
    # The roster spelling must resolve to the same plan, not to None.
    assert (
        guardrails.training_plan_arn(f"shared-compute__{guardrails.REGION}__{guardrails.TRAINING_PLAN_QUEUE}")
        == expected
    )


def test_ordinary_queue_is_unchanged_by_the_training_plan_support(tmp_path):
    source = _source(tmp_path)
    plan = stage_s.build_plan(_args("s0", source), source)
    assert plan["manifest"]["infrastructure"]["training_plan_arn"] is None
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "1"
    assert guardrails.training_plan_arn(guardrails.QUEUE) is None


def test_plan_queue_changes_the_run_id(tmp_path):
    """The plan ARN is sealed, so a p5e run can never collide with its p5 twin."""
    source = _source(tmp_path)
    p5 = stage_s.build_plan(_args("s0", source), source)
    p5e = stage_s.build_plan(
        _args(
            "s0", source, "--queue", guardrails.TRAINING_PLAN_QUEUE, "--priority", "400", "--max-run-seconds", "86400"
        ),
        source,
    )
    assert p5["run_id"] != p5e["run_id"]


# --------------------------------------------------------------------------------------------
# Single-task ReMemBench cells. The substrate is NOT divisible (the omega cache's encoder_id is
# sha256 over a provenance that embeds the full 13-task demos_per_task map), so a single-task run
# stages the identical sealed dataset + cache and filters only the TRAINING SOUP, via WSM_TASKS.
# --------------------------------------------------------------------------------------------
SINGLE_TASK = "MemHeatPot"
RMB1T_CONFIGS = {
    "s0": ("scripts/configs/train/pi05_rmb1t_heatpot_base_finetune.yaml", None),
    "s1-tanh": ("scripts/configs/train/pi05_rmb1t_heatpot_tanh_finetune.yaml", "1"),
    "s1-dnw2": ("scripts/configs/train/pi05_rmb1t_heatpot_dnw2_finetune.yaml", "2"),
    "s1-dnw8": ("scripts/configs/train/pi05_rmb1t_heatpot_dnw8_finetune.yaml", "8"),
    "s1-dnw16": ("scripts/configs/train/pi05_rmb1t_heatpot_dnw16_finetune.yaml", "16"),
    "s3": ("scripts/configs/train/pi05_rmb1t_heatpot_jw01k16_finetune.yaml", "1"),
}


@pytest.mark.parametrize("arm", sorted(RMB1T_CONFIGS))
def test_single_task_filters_the_soup_and_leaves_the_substrate_sealed(tmp_path, arm):
    source = _source(tmp_path)
    config, expected_k = RMB1T_CONFIGS[arm]
    real_arm = arm.split("-")[0]
    extra = () if real_arm == "s0" else _remembench_workspace_extra()
    args = _args(
        real_arm,
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--single-task",
        SINGLE_TASK,
        "--train-steps",
        "4000",
        *extra,
    )
    args.config_override = config
    plan = stage_s.build_plan(args, source)
    env, manifest = plan["environment"], plan["manifest"]

    # The filter itself: one env knob, one sealed manifest block.
    assert env["WSM_TASKS"] == SINGLE_TASK
    assert manifest["data"]["training_task_filter"] == {
        "mechanism": "WSM_TASKS",
        "tasks": [SINGLE_TASK],
        "demos": 40,
        "staged_substrate_tasks": 13,
        "demos_per_task_source": stage_s.REMEMBENCH_DEMOS_PER_TASK_MAP,
    }
    # ...and NOTHING about the staged substrate moves: same inventory, same 13 task dirs, same
    # 323-file omega cache, same full per-task map (the encoder_id depends on all of it).
    assert manifest["data"]["inventory"]["artifact"] == "remembench_train13"
    assert manifest["data"]["tasks"] == 13
    assert env["TARGET_EXPECTED_TASKS"] == "13"
    assert env["TARGET_TASK_DIR_GLOBS"] == "*/*/lerobot"
    assert env["OMEGA_EXPECTED_FILES"] == "323"
    assert env["POLICY_FEATS_DEMOS_PER_TASK_MAP"] == stage_s.REMEMBENCH_DEMOS_PER_TASK_MAP

    assert manifest["training"]["config"] == config
    assert manifest["training"]["steps"] == 4000
    assert manifest["training"]["checkpoint_policy"]["retained_steps"] == [3999]
    assert env["STAGE_S_EXPECTED_TRAIN_STEPS"] == "4000" and env["STAGE_S_FINAL_STEP"] == "3999"
    assert env.get("WSM_K_WINDOW") == expected_k
    assert manifest["training"]["workspace_window"] == (None if real_arm == "s0" else int(expected_k))
    assert manifest["training"]["norm_split_nav"] is False
    assert max(len(value.encode("utf-8")) for value in env.values()) <= 512


def test_single_task_is_sealed_into_the_identity_and_fails_closed_on_a_bad_task(tmp_path):
    source = _source(tmp_path)
    base = _args("s0", source, "--dataset-profile", "remembench_v02_train13", "--train-steps", "4000")
    base.config_override = RMB1T_CONFIGS["s0"][0]
    multi = stage_s.build_plan(base, source)
    assert multi["manifest"]["data"]["training_task_filter"] is None
    assert "WSM_TASKS" not in multi["environment"]

    one = _args(
        "s0",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--single-task",
        SINGLE_TASK,
        "--train-steps",
        "4000",
    )
    one.config_override = RMB1T_CONFIGS["s0"][0]
    other = _args(
        "s0",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--single-task",
        "MemWashAndReturnLeft",
        "--train-steps",
        "4000",
    )
    other.config_override = RMB1T_CONFIGS["s0"][0]
    ids = {
        multi["run_id"],
        stage_s.build_plan(one, source)["run_id"],
        stage_s.build_plan(other, source)["run_id"],
    }
    assert len(ids) == 3

    typo = _args("s0", source, "--dataset-profile", "remembench_v02_train13", "--single-task", "MemHeatPott")
    with pytest.raises(SystemExit, match="is not a task of"):
        stage_s.build_plan(typo, source)

    robocasa = _args("s0", source, "--single-task", SINGLE_TASK)
    with pytest.raises(SystemExit, match="not defined for dataset profile"):
        stage_s.build_plan(robocasa, source)


@pytest.mark.parametrize("window", [2, 16, 32])
def test_multi_task_deltanet_window_sweep_seals_each_window(tmp_path, window):
    source = _source(tmp_path)
    args = _args(
        "s1",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--train-steps",
        "15000",
        *_remembench_workspace_extra(),
    )
    args.config_override = f"scripts/configs/train/pi05_rmb_deltanet_w{window}_finetune.yaml"
    plan = stage_s.build_plan(args, source)
    assert plan["environment"]["WSM_K_WINDOW"] == str(window)
    assert plan["manifest"]["training"]["workspace_window"] == window
    assert plan["manifest"]["training"]["steps"] == 15000
    assert plan["manifest"]["data"]["training_task_filter"] is None
    assert "WSM_TASKS" not in plan["environment"]


# --------------------------------------------------------------------------------------------
# Causal-confusion wave: the history-intervention dropout knob and the ReMemBench CFG cell.
# --------------------------------------------------------------------------------------------
DROP_CONFIGS = {
    8: "scripts/configs/train/pi05_rmb_deltanet_drop_finetune.yaml",
    16: "scripts/configs/train/pi05_rmb_deltanet_w16_drop_finetune.yaml",
    32: "scripts/configs/train/pi05_rmb_deltanet_w32_drop_finetune.yaml",
}


@pytest.mark.parametrize("window", [8, 16, 32])
def test_cond_history_dropout_is_sealed_into_env_and_manifest(tmp_path, window):
    source = _source(tmp_path)
    args = _args(
        "s1",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--train-steps",
        "15000",
        *_remembench_workspace_extra(),
    )
    args.config_override = DROP_CONFIGS[window]
    plan = stage_s.build_plan(args, source)
    env, training = plan["environment"], plan["manifest"]["training"]

    assert training["cond_history_dropout"] == 0.5
    assert training["workspace_window"] == window
    assert env["WSM_COND_HISTORY_DROPOUT"] == "0.5"
    assert env["WSM_K_WINDOW"] == str(window)
    # The intervention must not drag any other sealed recipe bit with it.
    assert training["norm_split_nav"] is False and env["WSM_NORM_SPLIT_NAV"] == "0"
    assert training["jepa_aux"] is None
    assert plan["manifest"]["interface"] == "tanh"


def test_absent_cond_history_dropout_emits_no_env_key_and_seals_zero(tmp_path):
    """Every parent arm must be indistinguishable from a pre-knob launch."""
    source = _source(tmp_path)
    for config in (
        None,
        "scripts/configs/train/pi05_rmb_deltanet_finetune.yaml",
        "scripts/configs/train/pi05_rmb_deltanet_w32_finetune.yaml",
        "scripts/configs/train/pi05_stage_s1_deltanet_finetune.yaml",
    ):
        args = _args("s1", source, *_workspace_extra())
        if config:
            args.config_override = config
        plan = stage_s.build_plan(args, source)
        assert "WSM_COND_HISTORY_DROPOUT" not in plan["environment"], config
        assert plan["manifest"]["training"]["cond_history_dropout"] == 0.0, config


def test_cond_history_dropout_is_refused_off_the_deltanet_read(tmp_path):
    source = _source(tmp_path)
    # The knob rides the tanh/deltanet read; any other arm is a launch-time error.
    for arm in ("s2", "s3"):
        args = _args(
            arm,
            source,
            "--dataset-profile",
            "remembench_v02_train13",
            "--train-steps",
            "15000",
            *_remembench_workspace_extra(),
        )
        args.config_override = DROP_CONFIGS[16]
        with pytest.raises(SystemExit, match="cond_history_dropout"):
            stage_s.build_plan(args, source)


def test_remembench_cfg_arm_mirrors_the_sealed_robocasa_cfg_recipe(tmp_path):
    """The rmb CFG cell is the s2 interface + drop-to-null p=0.2, on the 13-task substrate."""
    source = _source(tmp_path)
    config = "scripts/configs/train/pi05_rmb_cfg_finetune.yaml"
    args = _args(
        "s2",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--train-steps",
        "15000",
        *_remembench_workspace_extra(),
    )
    args.config_override = config
    plan = stage_s.build_plan(args, source)
    env, manifest = plan["environment"], plan["manifest"]

    assert manifest["interface"] == "cfg2" and env["PI_STAGE_S_INTERFACE"] == "cfg2"
    # Same drop-to-null parameters as the sealed RoboCasa CFG arm.
    robocasa = stage_s.build_plan(_args("s2", source, *_workspace_extra()), source)
    assert manifest["training"]["cfg_drop_probability"] == 0.2
    assert robocasa["manifest"]["training"]["cfg_drop_probability"] == 0.2
    assert env["WSM_CFG_P_DROP"] == robocasa["environment"]["WSM_CFG_P_DROP"] == "0.2"
    # Current-only read: window 1, no deltanet knobs, no history intervention.
    assert manifest["training"]["workspace_window"] == 1 and env["WSM_K_WINDOW"] == "1"
    assert manifest["training"]["cond_history_dropout"] == 0.0
    assert "WSM_COND_HISTORY_DROPOUT" not in env
    assert manifest["training"]["norm_split_nav"] is False
    # ReMemBench substrate + schedule.
    assert manifest["data"]["benchmark"] == "ReMemBench" and manifest["data"]["tasks"] == 13
    assert manifest["training"]["steps"] == 15000
    assert env["TARGET_DATA_S3"] == stage_s.REMEMBENCH_DATA_S3
    assert manifest["training"]["config"] == config


# --------------------------------------------------------------------------------------------
# PTRM (H9): the recursive read + Q head. The launcher seals its two defining numbers, refuses every
# confounder, and — the load-bearing part — leaves every OTHER arm's manifest byte-identical.
# --------------------------------------------------------------------------------------------
PTRM_CONFIG = "scripts/configs/train/pi05_norm_s1_ptrm_finetune.yaml"


def test_ptrm_recipe_seals_its_depth_and_lambda_into_env_and_manifest(tmp_path):
    source = _source(tmp_path)
    args = _args("s1", source, *_workspace_extra())
    args.config_override = PTRM_CONFIG
    plan = stage_s.build_plan(args, source)
    env, training = plan["environment"], plan["manifest"]["training"]

    assert training["ptrm"] == {"steps": 4, "q_weight": 0.1, "eval_noise": "inference_only"}
    assert env["WSM_COND_TYPE"] == "gated_deltanet_ptrm"
    assert env["WSM_PTRM_STEPS"] == "4"
    assert env["WSM_PTRM_Q_WEIGHT"] == "0.1"
    # It is an s1/tanh recipe on the deltanet w=8 window; nothing else about the plan moves.
    assert plan["manifest"]["interface"] == "tanh" and env["PI_STAGE_S_INTERFACE"] == "tanh"
    assert training["workspace_window"] == 8 and env["WSM_K_WINDOW"] == "8"
    assert training["norm_split_nav"] is True and env["WSM_NORM_SPLIT_NAV"] == "1"
    assert training["cond_history_dropout"] == 0.0
    assert "WSM_COND_HISTORY_DROPOUT" not in env
    assert training["jepa_aux"] is None
    assert training["steps"] == 60000
    assert env["WSM_TANH_GATE_INIT"] == "0.001"
    assert max(len(value.encode("utf-8")) for value in env.values()) <= 512
    # The arm is a distinct identity from its deltanet parent, and reproducible.
    parent = _args("s1", source, *_workspace_extra())
    parent.config_override = "scripts/configs/train/pi05_norm_s1_deltanet_finetune.yaml"
    assert plan["run_id"] != stage_s.build_plan(parent, source)["run_id"]
    assert stage_s.build_plan(args, source) == plan


def test_ptrm_refuses_every_confounder(tmp_path, monkeypatch):
    source = _source(tmp_path)
    recipe = yaml.safe_load((REPO_ROOT / PTRM_CONFIG).read_text(encoding="utf-8"))
    # build_plan resolves --config-override against the launcher's own repo root; re-root that at
    # tmp_path so the mutated recipes never touch the real scripts/configs/train/ tree.
    fake_repo = tmp_path / "wsmv2"
    (fake_repo / "scripts" / "configs" / "train").mkdir(parents=True)
    monkeypatch.setattr(stage_s, "__file__", str(fake_repo / "scripts" / "launch" / "launcher.py"))

    def plan_for(arm, mutate):
        variant = {**recipe, "model": {**recipe["model"], **mutate}}
        relpath = "scripts/configs/train/ptrm_variant.yaml"
        (fake_repo / relpath).write_text(yaml.safe_dump(variant), encoding="utf-8")
        args = _args(arm, source, *_workspace_extra())
        args.config_override = relpath
        return stage_s.build_plan(args, source)

    # A second train-time intervention on the same window would make it a two-variable arm.
    with pytest.raises(SystemExit, match="refused on the PTRM read"):
        plan_for("s1", {"cond_history_dropout": 0.5})
    # The aux target is the other confounder the design excludes.
    with pytest.raises(SystemExit, match="deliberately unconfounded"):
        plan_for("s1", {"jepa_aux": True})
    # The recursion fills the tanh read's subtree; it is meaningless on any other arm.
    for arm in ("s2", "s3"):
        with pytest.raises(SystemExit, match="it is an s1 recipe"):
            plan_for(arm, {})
    # Depth and lambda are validated, not merely recorded.
    with pytest.raises(SystemExit, match=r"ptrm_steps must be an integer in \[1, 16\]"):
        plan_for("s1", {"ptrm_steps": 0})
    with pytest.raises(SystemExit, match=r"ptrm_steps must be an integer in \[1, 16\]"):
        plan_for("s1", {"ptrm_steps": 17})
    with pytest.raises(SystemExit, match="ptrm_q_weight must be >= 0"):
        plan_for("s1", {"ptrm_q_weight": -0.5})


def test_the_ptrm_key_appears_only_on_ptrm_runs_so_no_live_run_id_moves(tmp_path):
    """MANIFEST STABILITY. A `"ptrm": null` in every training block would rename every baseline.

    The pre-patch bytes are not available to this process, so the invariant is asserted structurally:
    no non-PTRM plan carries the key at all, and its canonical JSON therefore cannot have changed.
    """
    source = _source(tmp_path)
    for arm, config in (
        ("s0", None),
        ("s1", None),
        ("s1", "scripts/configs/train/pi05_norm_s1_deltanet_finetune.yaml"),
        ("s1", "scripts/configs/train/pi05_norm_s1_deltanet_w16_finetune.yaml"),
        ("s2", None),
        ("s3", None),
    ):
        args = _args(arm, source, *(() if arm == "s0" else _workspace_extra()))
        if config:
            args.config_override = config
        plan = stage_s.build_plan(args, source)
        training = plan["manifest"]["training"]
        assert "ptrm" not in training, (arm, config)
        assert "ptrm" not in plan["manifest_json"], (arm, config)
        for key in ("WSM_COND_TYPE", "WSM_PTRM_STEPS", "WSM_PTRM_Q_WEIGHT"):
            assert key not in plan["environment"], (arm, config, key)
        assert plan["required_fork_attributes"] == ()

    # Known-good identity of the live n-wave deltanet baseline: this run_id is what the campaign's
    # results are filed under, so it is pinned here rather than merely recomputed.
    baseline = _args("s1", source, *_workspace_extra())
    baseline.config_override = "scripts/configs/train/pi05_norm_s1_deltanet_finetune.yaml"
    first = stage_s.build_plan(baseline, source)
    assert first["manifest"]["training"]["workspace_window"] == 8
    assert first["run_id"] == stage_s.build_plan(baseline, source)["run_id"]


def test_pairing_demands_the_ptrm_sentinel_only_for_a_ptrm_recipe(tmp_path):
    source = _source(tmp_path)
    args = _args("s1", source, *_workspace_extra())
    args.config_override = PTRM_CONFIG
    plan = stage_s.build_plan(args, source)
    assert plan["required_fork_attributes"] == (stage_s._FORK_PTRM_SENTINEL,)

    # A fork that predates the PTRM conditioner cannot build this recipe: caught offline, in ms.
    stale = _archive_cache(tmp_path)
    with pytest.raises(SystemExit) as failure:
        stage_s.assert_archive_pairing(
            wsmv2_root=stale / W_SHA,
            openpi_root=stale / O_SHA,
            wsmv2_sha256=W_SHA,
            openpi_sha256=O_SHA,
            recipe_required=frozenset(plan["required_fork_attributes"]),
        )
    assert stage_s._FORK_PTRM_SENTINEL in str(failure.value)
    assert W_SHA in str(failure.value) and O_SHA in str(failure.value)

    # The same stale fork is a perfectly good pair for every non-PTRM recipe.
    assert stage_s.assert_archive_pairing(
        wsmv2_root=stale / W_SHA, openpi_root=stale / O_SHA, wsmv2_sha256=W_SHA, openpi_sha256=O_SHA
    ) == {COMBO_GATE, "_WSM_K_WINDOW"}

    # A fork that binds the sentinel in its conditioner module satisfies the requirement.
    current = _archive_cache(tmp_path / "current", sentinels=(stage_s._FORK_PTRM_SENTINEL,))
    verified = stage_s.assert_archive_pairing(
        wsmv2_root=current / W_SHA,
        openpi_root=current / O_SHA,
        wsmv2_sha256=W_SHA,
        openpi_sha256=O_SHA,
        recipe_required=frozenset(plan["required_fork_attributes"]),
    )
    assert verified == {COMBO_GATE, "_WSM_K_WINDOW", stage_s._FORK_PTRM_SENTINEL}


def test_the_live_openpi_fork_defines_the_ptrm_sentinel(tmp_path):
    """The sentinel is only useful if the checked-out fork actually binds it at module level."""
    fork = ROBOCASA_OPENPI_ROOT
    module = fork / stage_s._FORK_SENTINEL_RELPATHS[0]
    if not module.is_file():
        pytest.skip(f"openpi fork not checked out at {fork}")
    assert stage_s._FORK_PTRM_SENTINEL in stage_s.fork_attributes_defined(fork)


# --- wsmv2 <-> openpi archive pairing -------------------------------------------------------
# The two archives are a contract, not independent pins: wsmv2's post-import gate checks READ
# module-level attributes from the fork's dataloader, so an older openpi kills the run at import
# on an already-started node. These tests pin the offline cross-check that catches it first.


def test_archive_pairing_accepts_a_fork_that_defines_every_read_attribute(tmp_path):
    cache = _archive_cache(tmp_path)
    verified = stage_s.assert_archive_pairing(
        wsmv2_root=cache / W_SHA,
        openpi_root=cache / O_SHA,
        wsmv2_sha256=W_SHA,
        openpi_sha256=O_SHA,
    )
    assert verified == {COMBO_GATE, "_WSM_K_WINDOW"}


def test_archive_pairing_reproduces_the_d1_failure_naming_attribute_and_both_shas(tmp_path):
    # The real 2026-08-07 pair: wsmv2 522ad4b0 reads _WSM_JEPA_WITH_WINDOW; openpi 768f274a, cut
    # before the combo work, defines everything else but not that gate.
    wsmv2_sha, openpi_sha = "5" * 64, "7" * 64
    cache = _archive_cache(
        tmp_path,
        reads=(COMBO_GATE, "_WSM_K_WINDOW"),
        defines=("_WSM_K_WINDOW",),
        wsmv2_sha=wsmv2_sha,
        openpi_sha=openpi_sha,
    )
    with pytest.raises(SystemExit) as failure:
        stage_s.assert_archive_pairing(
            wsmv2_root=cache / wsmv2_sha,
            openpi_root=cache / openpi_sha,
            wsmv2_sha256=wsmv2_sha,
            openpi_sha256=openpi_sha,
        )
    message = str(failure.value)
    assert COMBO_GATE in message
    assert wsmv2_sha in message and openpi_sha in message
    # The compatible gate must NOT be reported as missing.
    assert "_WSM_K_WINDOW," not in message


def test_archive_pairing_runs_at_dry_run_time_and_reports_the_verified_count(tmp_path, capsys):
    source = _source(tmp_path)
    args = _args("s0", source)
    command = [
        *_command(args),
        "--dry-run",
        "--archive-cache-dir",
        str(_archive_cache(tmp_path)),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "archive_pairing=verified (2 fork attributes)" in result.stdout


def test_dry_run_without_a_cache_stays_offline_and_reports_unverified(tmp_path, capsys):
    # A dry run must not reach for S3, so an uncached archive is reported rather than downloaded.
    source = _source(tmp_path)
    args = _args("s0", source)
    result = subprocess.run([*_command(args), "--dry-run"], capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "archive_pairing=UNVERIFIED" in result.stdout
    # The offline contract still holds: no S3 lookup happened on the way to that verdict.
    assert "DRY RUN: offline" in result.stdout


def test_incompatible_pair_blocks_the_launch_before_a_job_name_is_minted(tmp_path, monkeypatch):
    source = _source(tmp_path)
    args = _args("s0", source)
    cache = _archive_cache(tmp_path, reads=(COMBO_GATE,), defines=("_WSM_K_WINDOW",))
    submitted = []
    monkeypatch.setattr(stage_s, "submit_training_job", lambda **kwargs: submitted.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [*_command(args)[1:], "--confirm-submit", "--archive-cache-dir", str(cache)],
    )
    with pytest.raises(SystemExit) as failure:
        stage_s.main()
    assert COMBO_GATE in str(failure.value)
    assert submitted == []


def test_submission_is_fail_closed_when_an_archive_cannot_be_resolved(tmp_path, monkeypatch):
    # No cache and no network: a real submission must refuse rather than ship an unchecked pair.
    source = _source(tmp_path)
    args = _args("s0", source)
    submitted = []
    monkeypatch.setattr(stage_s, "submit_training_job", lambda **kwargs: submitted.append(kwargs))

    monkeypatch.setattr(stage_s, "_archive_tree", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [*_command(args)[1:], "--confirm-submit"])
    with pytest.raises(SystemExit) as failure:
        stage_s.main()
    assert "fail-closed" in str(failure.value)
    assert submitted == []


# --------------------------------------------------------------------------------------------
# Checkpoint contracts (A19). `milestones` is opt-in and moves the run_id; the `final-only`
# default must not move a byte -- its run_ids are PINNED below against a mode-pinned fake tree.
# --------------------------------------------------------------------------------------------
MILESTONE_RMB_BASE = "scripts/configs/train/pi05_rmb_base_finetune.yaml"
W16_DROP = "scripts/configs/train/pi05_rmb_deltanet_w16_drop_finetune.yaml"


def _final_only_policy(final_step: int) -> dict:
    """The sealed checkpoint_policy block, exactly as every existing run manifest carries it."""
    return {
        "retained_steps": [final_step],
        "keep_period": None,
        "midrun_sync": False,
        "resume": False,
        "tree_manifest_schema": 1,
        "completion_claim_schema": 1,
    }


def _pinned_source(tmp_path: Path) -> Path:
    """Like _source, but with every permission bit pinned so the tree sha is umask-independent."""
    source = tmp_path / "internal_training_pinned"
    source.mkdir()
    source.chmod(0o755)
    entry = source / stage_s.ENTRY
    entry.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    entry.chmod(0o755)
    support = source / "support.sh"
    support.write_text("support=v1\n", encoding="utf-8")
    support.chmod(0o644)
    return source


def test_milestones_contract_seals_retained_steps_and_midrun_sync(tmp_path):
    source = _source(tmp_path)
    args = _args(
        "s0",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--train-steps",
        "60000",
        "--checkpoint-contract",
        "milestones",
        "--save-interval",
        "15000",
    )
    args.config_override = MILESTONE_RMB_BASE
    plan = stage_s.build_plan(args, source)
    env, training = plan["environment"], plan["manifest"]["training"]

    assert training["steps"] == 60000
    assert training["save_interval"] == 15000
    assert training["checkpoint_policy"] == {
        "contract": "milestones",
        "retained_steps": [15000, 30000, 45000, 59999],
        "keep_period": 15000,
        "midrun_sync": True,
        "resume": False,
        "tree_manifest_schema": 1,
        "completion_claim_schema": 1,
    }
    assert env["WSM_MAX_STEPS"] == "60000" and env["WSM_SAVE_INTERVAL"] == "15000"
    assert env["WSM_FINAL_ONLY_CHECKPOINTS"] == "0"
    assert env["WSM_KEEP_PERIOD"] == "15000"
    assert env["STAGE_S_CHECKPOINT_CONTRACT"] == "milestones"
    assert env["STAGE_S_RETAINED_STEPS"] == "15000,30000,45000,59999"
    assert env["STAGE_S_FINAL_STEP"] == "59999"
    # 60k IS the entry's default production contract, so the override key stays absent (as on
    # every RoboCasa 60k plan); the entry asserts WSM_MAX_STEPS == 60000 from its own default.
    assert "STAGE_S_EXPECTED_TRAIN_STEPS" not in env
    # One completion claim, keyed on the final step as before; it lists every uploaded step.
    assert env["COMPLETION_CLAIM_S3"].endswith("/step-59999.complete.json")
    assert max(len(value.encode("utf-8")) for value in env.values()) <= 512
    # Deterministic, and a DIFFERENT identity from the same recipe under final-only.
    assert stage_s.build_plan(args, source) == plan
    final_only = _args("s0", source, "--dataset-profile", "remembench_v02_train13", "--train-steps", "60000")
    final_only.config_override = MILESTONE_RMB_BASE
    other = stage_s.build_plan(final_only, source)
    assert other["run_id"] != plan["run_id"]
    assert other["manifest"]["training"]["checkpoint_policy"] == _final_only_policy(59999)
    assert other["environment"]["WSM_FINAL_ONLY_CHECKPOINTS"] == "1"
    for key in ("STAGE_S_CHECKPOINT_CONTRACT", "WSM_KEEP_PERIOD", "STAGE_S_RETAINED_STEPS"):
        assert key not in other["environment"], key


def test_milestones_contract_rides_the_workspace_arms_too(tmp_path):
    """P1'/P2'/P3' at 60k: the omega plumbing is untouched, only the checkpoint contract moves."""
    source = _source(tmp_path)
    args = _args(
        "s1",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--train-steps",
        "60000",
        "--checkpoint-contract",
        "milestones",
        "--save-interval",
        "15000",
        *_remembench_workspace_extra(),
    )
    args.config_override = W16_DROP
    plan = stage_s.build_plan(args, source)
    env, training = plan["environment"], plan["manifest"]["training"]
    assert training["checkpoint_policy"]["retained_steps"] == [15000, 30000, 45000, 59999]
    assert training["workspace_window"] == 16 and env["WSM_K_WINDOW"] == "16"
    assert training["cond_history_dropout"] == 0.5 and env["WSM_COND_HISTORY_DROPOUT"] == "0.5"
    assert env["WSM_ENCODER_ID"] == ENCODER_ID
    assert plan["manifest"]["interface"] == "tanh"
    parent = _args(
        "s1",
        source,
        "--dataset-profile",
        "remembench_v02_train13",
        "--train-steps",
        "60000",
        *_remembench_workspace_extra(),
    )
    parent.config_override = W16_DROP
    assert stage_s.build_plan(parent, source)["run_id"] != plan["run_id"]


def test_milestones_contract_validates_the_interval(tmp_path):
    source = _source(tmp_path)

    def plan_for(*extra):
        args = _args("s0", source, "--dataset-profile", "remembench_v02_train13", "--train-steps", "60000", *extra)
        args.config_override = MILESTONE_RMB_BASE
        return stage_s.build_plan(args, source)

    with pytest.raises(SystemExit, match="requires --save-interval"):
        plan_for("--checkpoint-contract", "milestones")
    with pytest.raises(SystemExit, match="must divide the step count"):
        plan_for("--checkpoint-contract", "milestones", "--save-interval", "14000")
    with pytest.raises(SystemExit, match="positive integer"):
        plan_for("--checkpoint-contract", "milestones", "--save-interval", "0")
    # No intermediate milestone == the final-only contract; say so instead of minting a twin.
    with pytest.raises(SystemExit, match="leaves no intermediate milestone"):
        plan_for("--checkpoint-contract", "milestones", "--save-interval", "60000")
    # A bare save interval must never smuggle a cadence into a final-only plan.
    with pytest.raises(SystemExit, match="only meaningful with --checkpoint-contract milestones"):
        plan_for("--save-interval", "15000")
    # The derivation itself.
    assert stage_s.milestone_retained_steps(60000, 15000, canary=False) == [15000, 30000, 45000, 59999]
    assert stage_s.milestone_retained_steps(60000, 10000, canary=False) == [10000, 20000, 30000, 40000, 50000, 59999]
    assert stage_s.milestone_retained_steps(1, 1, canary=True) == [0]


def test_milestones_canary_degenerates_to_the_final_step(tmp_path):
    """A 1-step canary may exercise the milestone entry path; its only milestone is step 0."""
    source = _source(tmp_path)
    args = _args(
        "s0",
        source,
        "--canary",
        "--priority",
        "1",
        "--max-run-seconds",
        "21600",
        "--checkpoint-contract",
        "milestones",
        "--save-interval",
        "1",
    )
    plan = stage_s.build_plan(args, source)
    assert plan["run_id"].startswith("s0-canary-")
    assert plan["manifest"]["training"]["checkpoint_policy"]["retained_steps"] == [0]
    assert plan["manifest"]["training"]["checkpoint_policy"]["contract"] == "milestones"
    assert plan["environment"]["STAGE_S_RETAINED_STEPS"] == "0"
    assert plan["environment"]["WSM_KEEP_PERIOD"] == "1"
    # A production cadence on a 1-step canary cannot divide it.
    args.save_interval = 15000
    with pytest.raises(SystemExit, match="must divide the step count"):
        stage_s.build_plan(args, source)


def test_final_only_default_is_byte_identical_to_the_pre_contract_launcher(tmp_path):
    """MANIFEST STABILITY. run_ids minted by the launcher BEFORE the contract flag existed.

    The fake tree is mode-pinned, so these constants do not depend on umask or tmp path. If any
    of them moves, some final-only plan's canonical JSON changed -- which renames every sealed
    baseline's twin. The canary and production dry-run forms are both covered.
    """
    source = _pinned_source(tmp_path)
    pinned = {
        "s0_production_60k": (_args("s0", source), None, 59999, "s0-5d4214137da48619"),
        "s0_canary": (
            _args(
                "s0", source, "--canary", "--priority", "1", "--max-run-seconds", str(stage_s.MAX_CANARY_RUN_SECONDS)
            ),
            None,
            0,
            "s0-canary-67e9b2bc1e73a886",
        ),
        "s1_rmb_15k_w16drop": (
            _args(
                "s1",
                source,
                "--dataset-profile",
                "remembench_v02_train13",
                "--train-steps",
                "15000",
                *_remembench_workspace_extra(),
            ),
            W16_DROP,
            14999,
            "s1-76f97a6d3c7566eb",
        ),
        "s0_rmb_60k_base": (
            _args("s0", source, "--dataset-profile", "remembench_v02_train13", "--train-steps", "60000"),
            MILESTONE_RMB_BASE,
            59999,
            "s0-45a7372ef1702c2b",
        ),
    }
    for name, (args, config, final_step, run_id) in pinned.items():
        if config:
            args.config_override = config
        assert args.checkpoint_contract == "final-only" and args.save_interval is None, name
        plan = stage_s.build_plan(args, source)
        assert plan["run_id"] == run_id, (name, plan["run_id"])
        training = plan["manifest"]["training"]
        assert training["save_interval"] == training["steps"], name
        assert training["checkpoint_policy"] == _final_only_policy(final_step), name
        assert "contract" not in plan["manifest_json"], name
        env = plan["environment"]
        assert env["WSM_FINAL_ONLY_CHECKPOINTS"] == "1", name
        assert env["WSM_SAVE_INTERVAL"] == env["WSM_MAX_STEPS"], name
        for key in ("STAGE_S_CHECKPOINT_CONTRACT", "WSM_KEEP_PERIOD", "STAGE_S_RETAINED_STEPS"):
            assert key not in env, (name, key)


def _entry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "internal_training" / stage_s.ENTRY


def _entry_contract_block() -> str:
    source = _entry_path().read_text(encoding="utf-8")
    begin = source.index("# --- stage_s_checkpoint_contract: begin ---")
    end = source.index("# --- stage_s_checkpoint_contract: end ---")
    return source[begin:end]


def test_external_entry_carries_both_checkpoint_contracts():
    entry = _entry_path()
    if not entry.exists():
        pytest.skip("TRI internal_training sibling checkout is not present")
    source = entry.read_text(encoding="utf-8")
    # The sealed assertion survives verbatim, and is still what an unselected contract hits.
    assert '[[ "$WSM_FINAL_ONLY_CHECKPOINTS" == "1" && "$WSM_SAVE_INTERVAL" == "$WSM_MAX_STEPS" ]]' in source
    assert "FATAL: Stage-S checkpoint contract must be final-only" in source
    assert 'STAGE_S_CHECKPOINT_CONTRACT="${STAGE_S_CHECKPOINT_CONTRACT:-final-only}"' in source
    # The milestone branch: derivation, exact-set assertion, mid-run sync, no prune, one claim.
    for needle in (
        "stage_s_milestone_steps",
        "stage_s_assert_retained_steps",
        "stage_s_midrun_sync_loop",
        '[[ "${WSM_KEEP_PERIOD:-}" == "$WSM_SAVE_INTERVAL" ]]',
        '[[ "$WSM_FINAL_ONLY_CHECKPOINTS" == "0" ]]',
        "STAGE_S_RETAINED_STEPS",
        '"checkpoint_contract": "milestones"',
        '"uploaded_steps"',
        "Stage-S is non-resumable under every checkpoint contract",
        "optimizer train_state/ never leaves the node",
    ):
        assert needle in source, needle
    # params/assets only, on every upload path (final-only block, milestone loop, mid-run loop).
    assert source.count('--exclude "*" --include "params/*" --include "assets/*"') >= 3
    result = subprocess.run(["bash", "-n", str(entry)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_external_entry_milestone_derivation_matches_the_launcher_on_a_fake_checkpoint_dir(tmp_path):
    """Shell-level check of the entry's retained-step derivation and exact-set assertion."""
    if not _entry_path().exists():
        pytest.skip("TRI internal_training sibling checkout is not present")
    block = _entry_contract_block()

    def steps(max_steps: str, save_interval: str):
        return subprocess.run(
            ["bash", "-c", block + '\nstage_s_milestone_steps "$1" "$2"', "_", max_steps, save_interval],
            capture_output=True,
            text=True,
            check=False,
        )

    ok = steps("60000", "15000")
    assert ok.returncode == 0, ok.stderr
    assert [int(step) for step in ok.stdout.split()] == [15000, 30000, 45000, 59999]
    assert [int(step) for step in ok.stdout.split()] == stage_s.milestone_retained_steps(60000, 15000, canary=False)
    assert [int(s) for s in steps("60000", "10000").stdout.split()] == [10000, 20000, 30000, 40000, 50000, 59999]
    assert steps("1", "1").stdout.split() == ["0"]
    for bad in (("60000", "14000"), ("60000", "0"), ("abc", "15000"), ("60000", "")):
        result = steps(*bad)
        assert result.returncode == 36, bad
        assert "FATAL" in result.stderr, bad

    ckpt = tmp_path / "ckpts" / "pi05_robocasa_rmb_base" / "pi05_rc365_rmb_base"
    for step in (15000, 30000, 45000, 59999):
        (ckpt / str(step) / "params").mkdir(parents=True)
        (ckpt / str(step) / "assets").mkdir()
    # An orbax tmp dir is non-numeric and must be ignored, exactly as the final-only check did.
    (ckpt / "59999.orbax-checkpoint-tmp-1756800000").mkdir()
    expected = ["15000", "30000", "45000", "59999"]

    def assert_set(*argv):
        return subprocess.run(
            ["bash", "-c", block + '\nstage_s_assert_retained_steps "$@"', "_", *argv],
            capture_output=True,
            text=True,
            check=False,
        )

    assert assert_set(str(ckpt), *expected).returncode == 0
    # A pruned milestone, an unexpected extra dir, and a step without assets/ all fail closed (38).
    (ckpt / "20000" / "params").mkdir(parents=True)
    (ckpt / "20000" / "assets").mkdir()
    extra = assert_set(str(ckpt), *expected)
    assert extra.returncode == 38 and "20000" in extra.stderr
    (ckpt / "20000" / "assets").rmdir()
    (ckpt / "20000" / "params").rmdir()
    (ckpt / "20000").rmdir()
    (ckpt / "30000" / "assets").rmdir()
    (ckpt / "30000" / "params").rmdir()
    (ckpt / "30000").rmdir()
    missing = assert_set(str(ckpt), *expected)
    assert missing.returncode == 38 and "must retain exactly" in missing.stderr
    (ckpt / "30000" / "params").mkdir(parents=True)
    no_assets = assert_set(str(ckpt), *expected)
    assert no_assets.returncode == 38 and "lacks params/ or assets/" in no_assets.stderr
    assert assert_set(str(tmp_path / "absent"), *expected).returncode == 38
