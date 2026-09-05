#!/usr/bin/env python3
"""Offline validators used on-node by the exact pi Stage-S RoboCasa eval entry.

The launcher is responsible for choosing canonical URIs.  This script independently checks that
the staged eval manifest, immutable training manifest, workspace-artifact manifest, downloaded
single-file artifacts, and frozen-tap tree all agree with the job environment before a policy
server can start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from scripts.launch.validate_stage_s_task_prompts import (
    GLOBAL_LANGUAGE_MODE,
    validate_task_prompts,
)

HEX = set("0123456789abcdef")


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is missing")
    return value


def require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in HEX for character in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def load_semantic_manifest(path: str | Path, expected: str, label: str) -> dict:
    file_hash_env = {
        "eval manifest": "EVAL_RUN_MANIFEST_FILE_SHA256",
        "training manifest": "TRAIN_RUN_MANIFEST_FILE_SHA256",
    }.get(label)
    if file_hash_env:
        _expect(sha256_file(path), require_env(file_hash_env), f"{label} exact file hash")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = payload.get("manifest_sha256")
    unsealed = dict(payload)
    unsealed.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical_json(unsealed)).hexdigest()
    if claimed != expected or actual != expected:
        raise ValueError(f"{label} seal mismatch: claimed={claimed!r} expected={expected!r} actual={actual!r}")
    return payload


def load_training_completion_claim(path: str | Path) -> dict:
    _expect(
        sha256_file(path),
        require_env("TRAIN_COMPLETION_CLAIM_SHA256"),
        "training completion claim exact file hash",
    )
    claim = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "kind",
        "run_id",
        "step",
        "checkpoint_uri",
        "tree_manifest_uri",
        "tree_manifest_sha256",
        "run_manifest_sha256",
        "producer_id",
    }
    missing = required - set(claim)
    if missing:
        raise ValueError(f"training completion claim missing {sorted(missing)}")
    _expect(claim.get("schema_version"), 1, "training completion schema_version")
    _expect(claim.get("kind"), "pi_stage_s_checkpoint_complete", "training completion kind")
    if not isinstance(claim.get("producer_id"), str) or not claim["producer_id"]:
        raise ValueError("training completion producer_id must be a non-empty string")
    return claim


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def validate_run_manifests(eval_path: str | Path, training_path: str | Path, completion_path: str | Path) -> None:
    eval_manifest = load_semantic_manifest(eval_path, require_env("EVAL_RUN_MANIFEST_SHA256"), "eval manifest")
    training_manifest = load_semantic_manifest(
        training_path,
        require_env("TRAIN_RUN_MANIFEST_SHA256"),
        "training manifest",
    )
    completion = load_training_completion_claim(completion_path)
    interface = require_env("PI_STAGE_S_INTERFACE")
    training_run_id = require_env("TRAIN_RUN_ID")
    # The serve interface no longer determines the arm one-to-one: base serves s0, the q0 control
    # (bitwise-identical serving) AND s3 (its JEPA head is train-time only, so sample_actions is
    # byte-identical to s0), tanh serves BOTH s1 and q1 (identical serve-time contract),
    # robottt_fast serves q2, and tanh_robottt serves q3 (workspace read + fast weights
    # combined). The arm therefore comes from the pinned training run id; the serve interface must
    # then agree with it. Stage-Q training runs record their TRAINING-side interface name (q0-q3)
    # and s3 records `jepa`, which differ from the serve name. Keep in lockstep with
    # submit_pi_stage_s_eval.py, eval_manifest.validate_evaluation_provenance, and
    # internal_training aggregate_eval.py.
    arm = training_run_id.split("-", 1)[0]
    serve_interface_by_arm = {
        "s0": "base",
        "s1": "tanh",
        "s2": "cfg2",
        "s3": "base",
        "q0": "base",
        "q1": "tanh",
        "q2": "robottt_fast",
        "q3": "tanh_robottt",
        # H13: aux-only, so serve is plain base (see submit_pi_stage_s_eval.py).
        "h13a": "base",
        "h13b": "base",
        "h13c": "base",
        "h13d": "base",
        "h13c2": "base",
        "h13d2": "base",
        "h13e": "tanh",
        "h13f": "tanh",
        "h13g": "tanh",
        "h13h": "tanh",
        "h13g2": "tanh",
        "h13h2": "tanh",
    }
    train_interface_by_arm = {
        "s0": "base",
        "s1": "tanh",
        "s2": "cfg2",
        "s3": "jepa",
        "q0": "q0",
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
        # All four H13 arms record `interface: "h13"` in their train manifests.
        "h13a": "h13",
        "h13b": "h13",
        "h13c": "h13",
        "h13d": "h13",
        "h13c2": "h13",
        "h13d2": "h13",
        # gdn8 arms record interface "tanh" in their TRAIN manifests (they ride the tanh dispatch).
        "h13e": "tanh",
        "h13f": "tanh",
        "h13g": "tanh",
        "h13h": "tanh",
        "h13g2": "tanh",
        "h13h2": "tanh",
    }
    if arm not in serve_interface_by_arm:
        raise ValueError(f"unsupported Stage-S arm {arm!r} (run id {training_run_id!r})")
    if serve_interface_by_arm[arm] != interface:
        raise ValueError(f"arm {arm!r} must serve via {serve_interface_by_arm[arm]!r}; got {interface!r}")
    checkpoint_step = int(require_env("STEP"))
    checkpoint_uri = require_env("CKPT_S3").rstrip("/")

    _expect(eval_manifest.get("schema_version"), 1, "eval schema_version")
    # STAGE_S_CANARY=1 jobs (E4 shakedown) carry the canary kind + a strict task subset; the env
    # flag and the manifest kind must agree in BOTH directions so a decisive run can never smuggle
    # a subset and a canary can never masquerade as decisive.
    canary = os.environ.get("STAGE_S_CANARY") == "1"
    _expect(
        eval_manifest.get("kind"),
        "pi_stage_s_robocasa_eval_canary" if canary else "pi_stage_s_robocasa_eval_run",
        "eval kind",
    )
    _expect(eval_manifest.get("arm"), arm, "eval arm")
    _expect(eval_manifest.get("interface"), interface, "eval interface")
    _expect(eval_manifest.get("eval_run_id"), require_env("EVAL_RUN_ID"), "eval_run_id")
    _expect(eval_manifest.get("results_s3"), require_env("RESULTS_S3"), "results_s3")
    _expect(
        eval_manifest.get("manifest_s3"),
        require_env("EVAL_RUN_MANIFEST_S3"),
        "eval manifest_s3",
    )

    run = eval_manifest.get("training_run") or {}
    _expect(run.get("run_id"), training_run_id, "eval training run_id")
    _expect(
        run.get("manifest_uri"),
        require_env("TRAIN_RUN_MANIFEST_S3"),
        "eval training manifest_uri",
    )
    _expect(
        run.get("manifest_sha256"),
        require_env("TRAIN_RUN_MANIFEST_SHA256"),
        "eval training manifest_sha256",
    )
    _expect(
        run.get("manifest_file_sha256"),
        require_env("TRAIN_RUN_MANIFEST_FILE_SHA256"),
        "eval training manifest_file_sha256",
    )
    _expect(run.get("checkpoint_uri"), checkpoint_uri, "eval checkpoint_uri")
    _expect(run.get("checkpoint_step"), checkpoint_step, "eval checkpoint_step")
    _expect(
        run.get("checkpoint_tree_manifest"),
        {
            "uri": require_env("CHECKPOINT_TREE_MANIFEST_S3"),
            "file_sha256": require_env("CHECKPOINT_TREE_MANIFEST_SHA256"),
            "schema_version": 1,
        },
        "eval checkpoint_tree_manifest",
    )

    _expect(
        run.get("completion_claim"),
        {
            "uri": require_env("TRAIN_COMPLETION_CLAIM_S3"),
            "file_sha256": require_env("TRAIN_COMPLETION_CLAIM_SHA256"),
            "schema_version": 1,
        },
        "eval training completion_claim",
    )
    _expect(completion.get("run_id"), training_run_id, "completion run_id")
    _expect(completion.get("step"), checkpoint_step, "completion step")
    _expect(completion.get("checkpoint_uri"), checkpoint_uri, "completion checkpoint_uri")
    _expect(
        completion.get("tree_manifest_uri"),
        require_env("CHECKPOINT_TREE_MANIFEST_S3"),
        "completion tree URI",
    )
    _expect(
        completion.get("tree_manifest_sha256"),
        require_env("CHECKPOINT_TREE_MANIFEST_SHA256"),
        "completion tree SHA",
    )
    _expect(
        completion.get("run_manifest_sha256"),
        require_env("TRAIN_RUN_MANIFEST_SHA256"),
        "completion run manifest SHA",
    )

    protocol = eval_manifest.get("protocol") or {}
    if canary:
        canary_tasks = (protocol.get("canary") or {}).get("tasks")
        wsm_tasks = require_env("WSM_TASKS").split(",")
        if not isinstance(canary_tasks, list) or not canary_tasks or len(canary_tasks) >= 50:
            raise ValueError("canary manifest must pin a non-empty strict task subset")
        _expect(canary_tasks, wsm_tasks, "canary tasks vs WSM_TASKS")
        expected_total = 100 * len(canary_tasks)
    else:
        _expect(protocol.get("canary"), None, "protocol.canary")
        expected_total = 5000
    expected_protocol = {
        "benchmark": "RoboCasa",
        "split": "target",
        "task_sets": ["atomic_seen", "composite_seen", "composite_unseen"],
        "num_tasks": 50,
        "episodes_per_task": 100,
        "rollouts_per_reset": 1,
        "total_rollouts": expected_total,
        "seed": int(require_env("SEED")),
        "policy_noise_kind": "pi_diffusion_sha256_v1",
        "replan_steps": 8,
        "exec_steps": 8,
        "workspace_stride": 8,
        "num_workers": 8,
        "envs_per_gpu": int(require_env("WSM_ENVS_PER_GPU")),
        "server_state_mode": require_env("PI_WSM_SERVER_STATE_MODE"),
        "video": require_env("VIDEO"),
    }
    for key, expected in expected_protocol.items():
        _expect(protocol.get(key), expected, f"protocol.{key}")
    episode = protocol.get("episode_manifest") or {}
    _expect(episode.get("uri"), require_env("EPISODE_MANIFEST_S3"), "episode uri")
    _expect(
        episode.get("file_sha256"),
        require_env("EPISODE_MANIFEST_SHA256"),
        "episode file_sha256",
    )
    _expect(episode.get("schema_version"), 2, "episode schema_version")

    sources = eval_manifest.get("sources") or {}
    for component, uri_name, sha_name in (
        ("wsmv2", "WSM_REPO_S3", "WSM_REPO_SHA256"),
        ("openpi", "OPENPI_FORK_S3", "OPENPI_FORK_SHA256"),
    ):
        source = sources.get(component) or {}
        _expect(source.get("uri"), require_env(uri_name), f"sources.{component}.uri")
        _expect(source.get("sha256"), require_env(sha_name), f"sources.{component}.sha256")

    _expect(training_manifest.get("schema_version"), 1, "training schema_version")
    _expect(training_manifest.get("run_id"), training_run_id, "training run_id")
    _expect(training_manifest.get("arm"), arm, "training arm")
    _expect(
        training_manifest.get("interface"),
        train_interface_by_arm[arm],
        "training interface",
    )
    _expect(
        training_manifest.get("manifest_s3"),
        require_env("TRAIN_RUN_MANIFEST_S3"),
        "training manifest_s3",
    )
    output = str(training_manifest.get("output_s3", "")).rstrip("/")
    _expect(f"{output}/{checkpoint_step}", checkpoint_uri, "training checkpoint identity")

    # Workspace block: REQUIRED for the omega-conditioned serves (s1/s2/q1/q3), FORBIDDEN for the
    # workspace-free serves (s0/s3/q0 base, q2 robottt_fast, and all four H13 arms) — both
    # directions. s3 consumed omega at TRAIN time only, so no workspace artifact may reach its eval
    # job; the H13 arms never consumed an omega cache at all (their encoder is live), so the same
    # prohibition is even stricter for them.
    workspace = eval_manifest.get("workspace_representation")
    if arm in ("s0", "s3", "q0", "q2", "h13a", "h13b", "h13c", "h13d", "h13c2", "h13d2"):
        _expect(workspace, None, f"{arm} workspace_representation")
    else:
        if not isinstance(workspace, dict):
            raise ValueError("s1/s2/q1/q3 workspace_representation must be a mapping")
        _expect(workspace.get("encoder_id"), require_env("WSM_ENCODER_ID"), "encoder_id")
        _expect(workspace.get("workspace_window"), 1, "workspace_window")
        expected = {
            "encoder_checkpoint": {
                "uri": require_env("ENCODER_CKPT_S3"),
                "sha256": require_env("ENCODER_CKPT_SHA256"),
            },
            "task_lang_table": {
                "uri": require_env("TASK_LANG_TABLE_S3"),
                "sha256": require_env("TASK_LANG_TABLE_SHA256"),
            },
            "task_prompt_manifest": {
                "uri": require_env("TASK_PROMPT_MANIFEST_S3"),
                "file_sha256": require_env("TASK_PROMPT_MANIFEST_SHA256"),
                "schema_version": 1,
            },
            "tap_prompt": {
                "mode": "terse",
                "global_language_mode": "canonical_terse_task_instruction",
                "canonical_task_prompt_manifest_id": require_env("TASK_PROMPT_MANIFEST_SHA256"),
                "demo_derived": False,
            },
            "frozen_tap_checkpoint": {
                "uri": require_env("TAP_CKPT_S3"),
                "tree_manifest_uri": require_env("TAP_TREE_MANIFEST_S3"),
                "tree_manifest_sha256": require_env("TAP_TREE_MANIFEST_SHA256"),
            },
            "workspace_artifacts_manifest": {
                "uri": require_env("WORKSPACE_ARTIFACT_MANIFEST_S3"),
                "sha256": require_env("WORKSPACE_ARTIFACT_MANIFEST_SHA256"),
                "schema_version": 1,
            },
        }
        for key, value in expected.items():
            _expect(workspace.get(key), value, f"workspace_representation.{key}")
        training_workspace = training_manifest.get("workspace_representation") or {}
        _expect(
            training_workspace.get("encoder_id"),
            require_env("WSM_ENCODER_ID"),
            "training encoder_id",
        )
        _expect(
            training_workspace.get("required_global_language_mode"),
            GLOBAL_LANGUAGE_MODE,
            "training global language mode",
        )
        _expect(
            training_workspace.get("task_prompt_manifest"),
            {
                "uri": require_env("TASK_PROMPT_MANIFEST_S3"),
                "sha256": require_env("TASK_PROMPT_MANIFEST_SHA256"),
            },
            "training task prompt manifest",
        )
    print(
        f"[stage-s-inputs] run manifests OK arm={arm} train={training_run_id} eval={require_env('EVAL_RUN_ID')}",
        flush=True,
    )


def _artifact_descriptor(uri_name: str, sha_name: str) -> dict:
    return {"uri": require_env(uri_name), "sha256": require_env(sha_name)}


def validate_workspace_artifacts(
    artifact_manifest_path: str | Path,
    encoder_checkpoint: str | Path,
    task_lang_table: str | Path,
    task_prompt_manifest_path: str | Path,
    tap_checkpoint: str | Path,
    tap_tree_manifest_path: str | Path,
    *,
    workers: int,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    artifact_manifest_path = Path(artifact_manifest_path)
    task_prompt_manifest_path = Path(task_prompt_manifest_path)
    tap_tree_manifest_path = Path(tap_tree_manifest_path)
    _expect(
        sha256_file(artifact_manifest_path),
        require_env("WORKSPACE_ARTIFACT_MANIFEST_SHA256"),
        "workspace artifact manifest file hash",
    )
    artifacts = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    expected_artifacts = {
        "schema_version": 1,
        "kind": "pi_stage_s_workspace_eval_artifacts",
        "encoder_id": require_env("WSM_ENCODER_ID"),
        "encoder_checkpoint": _artifact_descriptor("ENCODER_CKPT_S3", "ENCODER_CKPT_SHA256"),
        "task_lang_table": _artifact_descriptor("TASK_LANG_TABLE_S3", "TASK_LANG_TABLE_SHA256"),
        "task_prompt_manifest": _artifact_descriptor("TASK_PROMPT_MANIFEST_S3", "TASK_PROMPT_MANIFEST_SHA256"),
        "tap_prompt": {
            "mode": "terse",
            "global_language_mode": "canonical_terse_task_instruction",
            "canonical_task_prompt_manifest_id": require_env("TASK_PROMPT_MANIFEST_SHA256"),
            "demo_derived": False,
        },
        "frozen_tap_checkpoint": {
            "uri": require_env("TAP_CKPT_S3"),
            "tree_manifest_uri": require_env("TAP_TREE_MANIFEST_S3"),
            "tree_manifest_sha256": require_env("TAP_TREE_MANIFEST_SHA256"),
        },
    }
    _expect(artifacts, expected_artifacts, "workspace artifact manifest content")
    _expect(
        sha256_file(encoder_checkpoint),
        require_env("ENCODER_CKPT_SHA256"),
        "encoder checkpoint hash",
    )
    _expect(
        sha256_file(task_lang_table),
        require_env("TASK_LANG_TABLE_SHA256"),
        "task language table hash",
    )
    _expect(
        sha256_file(task_prompt_manifest_path),
        require_env("TASK_PROMPT_MANIFEST_SHA256"),
        "task prompt manifest file hash",
    )
    import numpy as np

    with np.load(task_lang_table, allow_pickle=False) as table:
        if "tasks" not in table.files:
            raise ValueError("task language table has no tasks array")
        lang_tasks = {str(value) for value in table["tasks"].tolist()}
        if len(lang_tasks) != 50:
            raise ValueError(f"task language table must contain 50 unique tasks; got {len(lang_tasks)}")
        if "expanded" in table.files:
            expanded = [str(value).strip() for value in table["expanded"].tolist()]
            if any(expanded):
                raise ValueError("legacy first-demo expanded task prompts are forbidden in focused Stage-S")
    validate_task_prompts(task_prompt_manifest_path, expected_task_names=lang_tasks, expected_tasks=50)
    _expect(
        sha256_file(tap_tree_manifest_path),
        require_env("TAP_TREE_MANIFEST_SHA256"),
        "tap tree manifest file hash",
    )
    tree = json.loads(tap_tree_manifest_path.read_text(encoding="utf-8"))
    _expect(tree.get("schema_version"), 1, "tap tree schema_version")
    _expect(tree.get("kind"), "wsm_artifact_tree_manifest", "tap tree kind")
    _expect(tree.get("artifact_uri"), require_env("TAP_CKPT_S3"), "tap artifact_uri")
    if set(tree) != {"schema_version", "kind", "artifact_uri", "files"}:
        raise ValueError(f"tap tree manifest has unexpected keys {sorted(set(tree))}")
    records = tree.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("tap tree manifest files must be a non-empty list")

    expected_files: dict[str, dict] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError(f"tap tree files[{index}] must contain path,size,sha256 exactly")
        relative = PurePosixPath(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe tap tree path {record['path']!r}")
        name = relative.as_posix()
        if name in expected_files:
            raise ValueError(f"duplicate tap tree path {name}")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"tap tree {name} has invalid size")
        require_sha(record["sha256"], f"tap tree {name} sha256")
        expected_files[name] = record
    if not any(name.startswith("params/") for name in expected_files):
        raise ValueError("tap tree manifest contains no params/ files")

    tap_root = Path(tap_checkpoint)
    for path in tap_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"tap checkpoint contains symlink {path}")
    actual_files = {path.relative_to(tap_root).as_posix(): path for path in tap_root.rglob("*") if path.is_file()}
    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    if missing or extra:
        raise ValueError(f"tap tree file-set mismatch: missing={missing[:10]} extra={extra[:10]}")

    def verify_one(item: tuple[str, Path]) -> None:
        name, path = item
        descriptor = expected_files[name]
        size = path.stat().st_size
        if size != descriptor["size"]:
            raise ValueError(f"tap file {name} size={size}, expected={descriptor['size']}")
        digest = sha256_file(path)
        if digest != descriptor["sha256"]:
            raise ValueError(f"tap file {name} sha256={digest}, expected={descriptor['sha256']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify_one, sorted(actual_files.items())))
    print(
        f"[stage-s-inputs] workspace artifacts OK encoder={require_env('WSM_ENCODER_ID')} "
        f"tap_files={len(actual_files)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--eval-manifest", required=True)
    run.add_argument("--training-manifest", required=True)
    run.add_argument("--training-completion-claim", required=True)
    workspace = subparsers.add_parser("workspace")
    workspace.add_argument("--artifact-manifest", required=True)
    workspace.add_argument("--encoder-checkpoint", required=True)
    workspace.add_argument("--task-lang-table", required=True)
    workspace.add_argument("--task-prompt-manifest", required=True)
    workspace.add_argument("--tap-checkpoint", required=True)
    workspace.add_argument("--tap-tree-manifest", required=True)
    workspace.add_argument("--workers", type=int, default=int(os.environ.get("WSM_ARTIFACT_VERIFY_WORKERS", "16")))
    args = parser.parse_args()
    if args.command == "run":
        validate_run_manifests(args.eval_manifest, args.training_manifest, args.training_completion_claim)
    else:
        validate_workspace_artifacts(
            args.artifact_manifest,
            args.encoder_checkpoint,
            args.task_lang_table,
            args.task_prompt_manifest,
            args.tap_checkpoint,
            args.tap_tree_manifest,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
