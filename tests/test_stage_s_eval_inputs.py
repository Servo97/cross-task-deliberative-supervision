"""Offline producer-to-consumer provenance tests for focused Stage-S evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_DIR))
import submit_pi_stage_s as train_launcher
import submit_pi_stage_s_eval as eval_launcher

from scripts.launch import validate_stage_s_eval_inputs as validator

ENCODER_ID = "5" * 64
ENCODER_SHA = "6" * 64
LANG_SHA = "7" * 64
PROMPT_SHA = "8" * 64
FEATURE_MANIFEST_SHA = "9" * 64
TAP_TREE_SHA = "a1" * 32
WORKSPACE_SHA = "b2" * 32

# Arms whose SERVE conditions on omega and therefore carries the workspace artifact set. q1
# deliberately reuses the tanh serve contract; q3 is the combined tanh_robottt interface. s3 is
# deliberately absent: it consumes omega at TRAIN time only (JEPA aux target) and serves as base.
WORKSPACE_ARMS = ("s1", "s2", "q1", "q3")


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "internal_training"
    source.mkdir()
    for entry in (train_launcher.ENTRY, eval_launcher.ENTRY):
        path = source / entry
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        path.chmod(0o755)
    return source


def _train_workspace_extra() -> list[str]:
    root = train_launcher.study_root(train_launcher.DEFAULT_OWNER)
    return [
        "--encoder-id",
        ENCODER_ID,
        "--policy-features-s3",
        f"{root}/caches/{ENCODER_ID}/omega",
        "--policy-features-manifest-s3",
        (f"{root}/manifests/artifacts/workspace/{ENCODER_ID}/omega/{FEATURE_MANIFEST_SHA}.json"),
        "--policy-features-manifest-sha256",
        FEATURE_MANIFEST_SHA,
        "--task-prompt-manifest-s3",
        (f"{root}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{PROMPT_SHA}.json"),
        "--task-prompt-manifest-sha256",
        PROMPT_SHA,
    ]


def _eval_workspace_extra() -> list[str]:
    root = eval_launcher.study_root(eval_launcher.DEFAULT_OWNER)
    return [
        "--encoder-id",
        ENCODER_ID,
        "--encoder-checkpoint-s3",
        f"{root}/artifacts/workspace/{ENCODER_ID}/encoder.pt",
        "--encoder-checkpoint-sha256",
        ENCODER_SHA,
        "--task-lang-table-s3",
        f"{root}/artifacts/workspace/{ENCODER_ID}/task_lang_table.npz",
        "--task-lang-table-sha256",
        LANG_SHA,
        "--task-prompt-manifest-s3",
        (f"{root}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{PROMPT_SHA}.json"),
        "--task-prompt-manifest-sha256",
        PROMPT_SHA,
        "--tap-checkpoint-s3",
        eval_launcher.INIT_S3,
        "--tap-tree-manifest-s3",
        f"{root}/manifests/artifacts/tap/{TAP_TREE_SHA}.json",
        "--tap-tree-manifest-sha256",
        TAP_TREE_SHA,
        "--workspace-artifacts-manifest-s3",
        f"{root}/manifests/artifacts/workspace/{ENCODER_ID}/{WORKSPACE_SHA}.json",
        "--workspace-artifacts-manifest-sha256",
        WORKSPACE_SHA,
    ]


def _train_plan(source: Path, arm: str = "s0") -> dict:
    root = train_launcher.study_root(train_launcher.DEFAULT_OWNER)
    init_sha = "1" * 64
    data_sha = "2" * 64
    tok_sha = "d" * 64
    args = train_launcher.make_parser().parse_args(
        [
            "--dry-run",
            "--arm",
            arm,
            "--source-dir",
            str(source),
            "--wsmv2-source-s3",
            f"{root}/code/wsmv2/{'a' * 64}.tgz",
            "--openpi-source-s3",
            f"{root}/code/openpi/{'b' * 64}.tgz",
            "--tokenizer-s3",
            f"{root}/artifacts/tokenizers/paligemma/{tok_sha}.model",
            "--tokenizer-sha256",
            tok_sha,
            "--init-inventory-s3",
            f"{root}/manifests/inventories/init/{init_sha}.json",
            "--init-inventory-sha256",
            init_sha,
            "--target-inventory-s3",
            f"{root}/manifests/inventories/data/{data_sha}.json",
            "--target-inventory-sha256",
            data_sha,
            "--image-uri",
            (f"141701954645.dkr.ecr.us-west-2.amazonaws.com/stage-s@sha256:{'c' * 64}"),
            *(_train_workspace_extra() if arm in ("s1", "s2", "s3", "q1", "q3") else []),
        ]
    )
    return train_launcher.build_plan(args, source)


def _canonical_file(payload: dict) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _fixture(tmp_path: Path, *, arm: str = "s0", completion_tree_sha: str | None = None):
    source = _source(tmp_path)
    root = train_launcher.study_root(train_launcher.DEFAULT_OWNER)
    training = _train_plan(source, arm)
    tree_sha = "3" * 64
    tree_uri = (
        f"{root}/manifests/artifacts/checkpoints/{training['run_id']}/step-{train_launcher.FINAL_STEP}/{tree_sha}.json"
    )
    checkpoint_uri = f"{training['output_s3']}/{train_launcher.FINAL_STEP}"
    completion = {
        "schema_version": 1,
        "kind": "pi_stage_s_checkpoint_complete",
        "run_id": training["run_id"],
        "step": train_launcher.FINAL_STEP,
        "checkpoint_uri": checkpoint_uri,
        "tree_manifest_uri": tree_uri,
        "tree_manifest_sha256": completion_tree_sha or tree_sha,
        "run_manifest_sha256": training["manifest"]["manifest_sha256"],
        "producer_id": "approved-train-attempt",
    }
    completion_bytes = _canonical_file(completion)
    completion_sha = hashlib.sha256(completion_bytes).hexdigest()
    training_bytes = (training["manifest_json"] + "\n").encode("utf-8")
    training_file_sha = hashlib.sha256(training_bytes).hexdigest()
    episode_sha = "4" * 64
    eval_args = eval_launcher.make_parser().parse_args(
        [
            "--dry-run",
            "--arm",
            arm,
            "--source-dir",
            str(source),
            "--training-run-id",
            training["run_id"],
            "--training-manifest-sha256",
            training["manifest"]["manifest_sha256"],
            "--training-manifest-file-sha256",
            training_file_sha,
            "--train-completion-claim-s3",
            (f"{root}/manifests/claims/train/{training['run_id']}/step-{train_launcher.FINAL_STEP}.complete.json"),
            "--train-completion-claim-sha256",
            completion_sha,
            "--checkpoint-step",
            str(train_launcher.FINAL_STEP),
            "--checkpoint-tree-manifest-s3",
            tree_uri,
            "--checkpoint-tree-manifest-sha256",
            tree_sha,
            "--episode-manifest-s3",
            (f"{root}/manifests/artifacts/eval/heldout50/{episode_sha}.json"),
            "--episode-manifest-sha256",
            episode_sha,
            "--wsmv2-source-s3",
            f"{root}/code/wsmv2/{'a' * 64}.tgz",
            "--openpi-source-s3",
            f"{root}/code/openpi/{'b' * 64}.tgz",
            "--image-uri",
            (f"141701954645.dkr.ecr.us-west-2.amazonaws.com/stage-s@sha256:{'c' * 64}"),
            *(_eval_workspace_extra() if arm in WORKSPACE_ARMS else []),
        ]
    )
    evaluation = eval_launcher.build_plan(eval_args, source)
    paths = {
        "training": tmp_path / "training.json",
        "completion": tmp_path / "completion.json",
        "evaluation": tmp_path / "evaluation.json",
    }
    paths["training"].write_bytes(training_bytes)
    paths["completion"].write_bytes(completion_bytes)
    paths["evaluation"].write_text(evaluation["manifest_json"] + "\n", encoding="utf-8")
    return evaluation, paths, completion


def _install_environment(monkeypatch, evaluation: dict) -> None:
    for key, value in evaluation["environment"].items():
        monkeypatch.setenv(key, value)


def test_training_completion_and_eval_manifests_are_bound_end_to_end(tmp_path, monkeypatch):
    evaluation, paths, _completion = _fixture(tmp_path)
    _install_environment(monkeypatch, evaluation)
    validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])


def test_rehashed_but_cross_linked_completion_claim_is_rejected(tmp_path, monkeypatch):
    evaluation, paths, _completion = _fixture(tmp_path, completion_tree_sha="0" * 64)
    _install_environment(monkeypatch, evaluation)
    with pytest.raises(ValueError, match="completion tree SHA"):
        validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])


@pytest.mark.parametrize(
    ("arm", "serve_interface", "train_interface"),
    (
        ("q1", "tanh", "q1"),
        ("q3", "tanh_robottt", "q3"),
        ("s1", "tanh", "tanh"),
        ("q0", "base", "q0"),
        ("q2", "robottt_fast", "q2"),
        # s3 is the widest serve/train split: it TRAINS through `jepa` (omega cache staged, aux
        # target) and SERVES through plain `base` (nothing read at inference).
        ("s3", "base", "jepa"),
    ),
)
def test_every_servable_arm_binds_serve_and_train_interfaces_end_to_end(
    tmp_path, monkeypatch, arm, serve_interface, train_interface
):
    """Launcher plan -> staged manifests -> on-node validator, per arm, both interface maps."""
    evaluation, paths, _completion = _fixture(tmp_path, arm=arm)
    assert evaluation["environment"]["PI_STAGE_S_INTERFACE"] == serve_interface
    assert evaluation["manifest"]["interface"] == serve_interface
    assert evaluation["manifest"]["arm"] == arm
    training = json.loads(paths["training"].read_text(encoding="utf-8"))
    assert training["interface"] == train_interface
    _install_environment(monkeypatch, evaluation)
    validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])


def test_s3_serves_base_and_refuses_a_workspace_interface(tmp_path, monkeypatch):
    """S3's checkpoint carries a `wsm_jepa_head` subtree that inference never touches; serving it
    through a workspace interface would silently claim an omega read that S3 does not have."""
    evaluation, paths, _completion = _fixture(tmp_path, arm="s3")
    assert evaluation["manifest"]["workspace_representation"] is None
    assert evaluation["manifest"]["protocol"]["server_state_mode"] == "stateless_v1"
    _install_environment(monkeypatch, evaluation)
    monkeypatch.setenv("PI_STAGE_S_INTERFACE", "tanh")
    with pytest.raises(ValueError, match="must serve via 'base'"):
        validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])


def test_q3_refuses_a_mismatched_serve_interface(tmp_path, monkeypatch):
    """A q3 checkpoint served through the workspace-free Q2 interface must refuse (and vice
    versa the map pins tanh_robottt) — the arm comes from TRAIN_RUN_ID, never the interface."""
    evaluation, paths, _completion = _fixture(tmp_path, arm="q3")
    _install_environment(monkeypatch, evaluation)
    monkeypatch.setenv("PI_STAGE_S_INTERFACE", "robottt_fast")
    with pytest.raises(ValueError, match="must serve via 'tanh_robottt'"):
        validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])


def test_q1_refuses_the_base_interface(tmp_path, monkeypatch):
    evaluation, paths, _completion = _fixture(tmp_path, arm="q1")
    _install_environment(monkeypatch, evaluation)
    monkeypatch.setenv("PI_STAGE_S_INTERFACE", "base")
    with pytest.raises(ValueError, match="must serve via 'tanh'"):
        validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])


def test_q3_workspace_artifacts_are_required_on_node(tmp_path, monkeypatch):
    """The combined interface may never silently degrade to workspace-free serving: dropping any
    workspace artifact variable from a q3 job environment fails the on-node validator loudly."""
    evaluation, paths, _completion = _fixture(tmp_path, arm="q3")
    _install_environment(monkeypatch, evaluation)
    monkeypatch.delenv("ENCODER_CKPT_S3")
    with pytest.raises(ValueError, match="ENCODER_CKPT_S3"):
        validator.validate_run_manifests(paths["evaluation"], paths["training"], paths["completion"])
