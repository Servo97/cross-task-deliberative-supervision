"""Offline tests for the demo-independent canonical terse task-prompt manifest."""

from __future__ import annotations

import json

import pytest

from scripts.launch import build_stage_s_task_prompts as builder
from scripts.launch import validate_stage_s_task_prompts as validator
from vla_training.eval.eval_pi_05 import workspace_prompt_fields


def _fixture(tmp_path):
    target = tmp_path / "target"
    for index, task in enumerate(("TaskA", "TaskB")):
        kind = "atomic" if index == 0 else "composite"
        (target / kind / task / "date" / "lerobot").mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact": validator.ARTIFACT,
        "global_language_mode": validator.GLOBAL_LANGUAGE_MODE,
        "demo_derived": False,
        "tasks": [
            {"task": "TaskA", "prompt": "Perform task A"},
            {"task": "TaskB", "prompt": "Perform task B"},
        ],
    }
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return target, path, manifest


def test_exact_demo_independent_terse_contract_passes(tmp_path):
    target, path, _manifest = _fixture(tmp_path)
    assert validator.validate_task_prompts(path, target_root=target, expected_tasks=2) == {"tasks": 2}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda manifest: manifest.update(demo_derived=True), "demo_derived=false"),
        (
            lambda manifest: manifest.update(global_language_mode="per_demo_expanded"),
            "canonical_terse_task_instruction",
        ),
        (lambda manifest: manifest["tasks"].pop(), "exactly 2 tasks"),
    ),
)
def test_demo_derived_wrong_mode_and_incomplete_task_set_fail(tmp_path, mutation, message):
    target, path, manifest = _fixture(tmp_path)
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validator.validate_task_prompts(path, target_root=target, expected_tasks=2)


def test_eval_can_validate_same_schema_against_exact_task_table_names(tmp_path):
    _target, path, _manifest = _fixture(tmp_path)
    assert validator.validate_task_prompts(path, expected_task_names={"TaskA", "TaskB"}, expected_tasks=2) == {
        "tasks": 2
    }
    assert validator.load_task_prompts(path, expected_task_names={"TaskA", "TaskB"}, expected_tasks=2) == {
        "TaskA": "Perform task A",
        "TaskB": "Perform task B",
    }


def test_eval_routes_only_the_exact_canonical_prompt_to_private_wsm_field():
    prompts = {"TaskA": "Perform task A"}
    assert workspace_prompt_fields("TaskA", prompts) == {"wsm_prompt": "Perform task A"}
    assert workspace_prompt_fields("TaskA", None) == {}
    with pytest.raises(ValueError, match="missing task"):
        workspace_prompt_fields("TaskB", prompts)


def test_builder_derives_one_demo_independent_prompt_per_task(tmp_path):
    target = tmp_path / "target"
    prompts = {"TaskA": "Perform task A", "TaskB": "Perform task B"}
    for index, (task, prompt) in enumerate(prompts.items()):
        family = "atomic" if index == 0 else "composite"
        metadata = target / family / task / "capture" / "lerobot" / "meta"
        metadata.mkdir(parents=True)
        (metadata / "episodes.jsonl").write_text(
            "".join(json.dumps({"episode_index": episode, "tasks": [prompt]}) + "\n" for episode in range(3)),
            encoding="utf-8",
        )
    path, digest, uri, manifest = builder.build_task_prompt_manifest(
        target,
        output_dir=tmp_path / "out",
        study_root="s3://bucket/owner/studies/long_context_v1",
        expected_tasks=2,
    )
    assert path.name == f"{digest}.json"
    assert uri.endswith(f"/robocasa_target50/{digest}.json")
    assert manifest["demo_derived"] is False
    assert {record["task"]: record["prompt"] for record in manifest["tasks"]} == prompts


def test_builder_requires_reviewed_override_for_multiple_episode_prompts(tmp_path):
    target = tmp_path / "target"
    metadata = target / "atomic" / "TaskA" / "capture" / "lerobot" / "meta"
    metadata.mkdir(parents=True)
    (metadata / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["Prompt A"]})
        + "\n"
        + json.dumps({"episode_index": 1, "tasks": ["Prompt B"]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reviewed override"):
        builder.derive_prompts(target, expected_tasks=1)
    assert builder.derive_prompts(target, expected_tasks=1, overrides={"TaskA": "Reviewed prompt"}) == {
        "TaskA": "Reviewed prompt"
    }
