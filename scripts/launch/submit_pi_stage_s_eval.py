#!/usr/bin/env python3
"""Approval-gated exact RoboCasa evaluation for one pi0.5 Stage-S checkpoint.

This is intentionally separate from the historical ``submit_evals.py`` paths.  It accepts exactly
one servable arm (s0-s3, q0-q3), derives the checkpoint and results locations from immutable
study identities,
stages a sealed eval-run manifest, and targets only account-141 cam-robotics.  ``--dry-run`` is
fully offline.  Real submission remains intentionally disabled until the listed production blockers
are closed; once enabled it will still require prior user approval and ``--confirm-submit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
from datetime import datetime, timezone

from launch_guardrails import (
    EXECUTION_ACCOUNT,
    PROJECT_TAG,
    STORAGE_ACCOUNT,
    add_guardrail_arguments,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
    validate_and_confirm,
)
from submit_pi_stage_s import (
    DEFAULT_OWNER,
    FINAL_STEP,  # noqa: F401  re-exported: tests and tooling read the final step from this module
    INIT_S3,
    STUDY,
    content_addressed_archive,
    image_digest,
    study_root,
)

ENTRY = "robocasa_eval_entry.sh"
INSTANCE_TYPE = "ml.p5.48xlarge"
STAGED_MANIFEST_NAME = "_stage_s_eval_run_manifest.json"
TASK_SETS = "atomic_seen,composite_seen,composite_unseen"
NUM_TASKS = 50
EPISODES_PER_TASK = 100
NUM_WORKERS = 8
CADENCE = 8
POLICY_NOISE_KIND = "pi_diffusion_sha256_v1"
GLOBAL_LANGUAGE_MODE = "canonical_terse_task_instruction"
K_CHOICES = (1, 4, 8)
# Study-fixed base seed of the sealed heldout50 episode manifest (E1 BASE_SEED). The rollout runner
# refuses any config seed that differs from the manifest's embedded base_seed, so this — not the
# historical default 7 — is the only valid protocol seed (E4 canary attempt-3 failure, 2026-07-27).
EPISODE_BASE_SEED = 20260723
RETRY = {
    # One approval authorizes at most one five-day attempt. A retry needs fresh approval.
    "attempts": 1,
    "evaluateOnExit": [{"action": "EXIT", "onStatusReason": "*"}],
}
# Decisive submissions were fail-closed until the E4 shakedown passed. It did on 2026-07-27:
# evalcanary-s0-step59999-2d76eaf6e16044e6 (K=8, 2 tasks x 100 exact resets) — realized batching
# enforced client-side, stateless isolation verified, 399 rollouts/hour, peak 45/80 GB, and the
# cross-account run-manifest + producer-claim published create-once. No blockers remain; decisive
# runs still require per-run user approval + --confirm-submit.
SUBMISSION_BLOCKERS = ()

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# q0 serves bitwise-identically to base (control arm, no fast weights). q2 serves via the
# workspace-free robottt_fast interface (server-side online TTT). q1 reuses the tanh serve contract
# (identical serve-time inputs to s1: encoder + lang table + prompt manifest + frozen tap). q3
# serves via the combined tanh_robottt interface (tanh workspace read + online fast weights).
# s3 (JEPA+SigReg aux) consumes omega only as a TRAIN-time target — its `wsm_jepa_head` subtree is
# never touched by sample_actions — so it serves through the plain base interface exactly like
# s0/q0: no workspace artifacts, no robottt, stateless.
_TRAIN_RUN_ID = re.compile(r"^(s[0123]|q[0123]|h13[a-h]2?)-[0-9a-f]{16}$")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seal_manifest(value: dict) -> tuple[dict, str]:
    sealed = dict(value)
    sealed.pop("manifest_sha256", None)
    checksum = hashlib.sha256(_canonical_json(sealed).encode("utf-8")).hexdigest()
    sealed["manifest_sha256"] = checksum
    return sealed, _canonical_json(sealed)


def _require_sha(value: str | None, flag: str) -> str:
    if value is None or _HEX64.fullmatch(value) is None:
        raise SystemExit(f"{flag} must be 64 lowercase hexadecimal characters")
    return value


def _content_addressed_json(uri: str, expected: str, *, flag: str) -> None:
    if uri != expected:
        raise SystemExit(f"{flag} must be the canonical content-addressed URI {expected}; got {uri}")


def resolve_source_dir(value: str | None) -> pathlib.Path:
    path = pathlib.Path(value or pathlib.Path(__file__).resolve().parents[3] / "internal_training").resolve()
    if not (path / ENTRY).is_file():
        raise SystemExit(f"source-dir {path} is missing {ENTRY}")
    return path


def _workspace_plan(args: argparse.Namespace, root: str) -> dict | None:
    fields = {
        "encoder_id": args.encoder_id,
        "encoder_checkpoint_s3": args.encoder_checkpoint_s3,
        "encoder_checkpoint_sha256": args.encoder_checkpoint_sha256,
        "task_lang_table_s3": args.task_lang_table_s3,
        "task_lang_table_sha256": args.task_lang_table_sha256,
        "task_prompt_manifest_s3": args.task_prompt_manifest_s3,
        "task_prompt_manifest_sha256": args.task_prompt_manifest_sha256,
        "tap_checkpoint_s3": args.tap_checkpoint_s3,
        "tap_tree_manifest_s3": args.tap_tree_manifest_s3,
        "tap_tree_manifest_sha256": args.tap_tree_manifest_sha256,
        "workspace_artifacts_manifest_s3": args.workspace_artifacts_manifest_s3,
        "workspace_artifacts_manifest_sha256": args.workspace_artifacts_manifest_sha256,
    }
    # Workspace artifacts are REQUIRED for every arm whose serve conditions on omega (s1/s2/q1/q3)
    # and FORBIDDEN for every arm whose serve does not (s0/s3/q0 base, q2 robottt_fast, and all four
    # H13 arms) — both directions checked. s3 trains ON omega but reads nothing at inference, so it
    # belongs on the forbidden side; the H13 arms never touch an omega cache at all (their encoder is
    # LIVE inside the train graph), so the prohibition is stricter still.
    # Keep in lockstep with validate_stage_s_eval_inputs.py and the entry.
    if args.arm in ("s0", "s3", "q0", "q2", "h13a", "h13b", "h13c", "h13d", "h13c2", "h13d2"):
        supplied = sorted(name for name, value in fields.items() if value is not None)
        if supplied:
            raise SystemExit(f"{args.arm} forbids workspace artifact arguments: {supplied}")
        return None

    missing = sorted(name for name, value in fields.items() if value is None)
    if missing:
        raise SystemExit("s1/s2/q1/q3 require all workspace artifact arguments; missing " + ", ".join(missing))

    encoder_id = _require_sha(args.encoder_id, "--encoder-id")
    encoder_sha = _require_sha(args.encoder_checkpoint_sha256, "--encoder-checkpoint-sha256")
    lang_sha = _require_sha(args.task_lang_table_sha256, "--task-lang-table-sha256")
    prompt_manifest_sha = _require_sha(args.task_prompt_manifest_sha256, "--task-prompt-manifest-sha256")
    tap_manifest_sha = _require_sha(args.tap_tree_manifest_sha256, "--tap-tree-manifest-sha256")
    workspace_manifest_sha = _require_sha(
        args.workspace_artifacts_manifest_sha256,
        "--workspace-artifacts-manifest-sha256",
    )

    expected_encoder = f"{root}/artifacts/workspace/{encoder_id}/encoder.pt"
    expected_lang = f"{root}/artifacts/workspace/{encoder_id}/task_lang_table.npz"
    expected_workspace_manifest = f"{root}/manifests/artifacts/workspace/{encoder_id}/{workspace_manifest_sha}.json"
    expected_tap_manifest = f"{root}/manifests/artifacts/tap/{tap_manifest_sha}.json"
    # The prompt-manifest namespace is per-dataset (ReMemBench ships its own 13-task manifest); the
    # default keeps every existing RoboCasa invocation byte-identical.
    expected_prompt_manifest = (
        f"{root}/manifests/artifacts/workspace/task_prompts/"
        f"{getattr(args, 'task_prompt_namespace', 'robocasa_target50')}/"
        f"{prompt_manifest_sha}.json"
    )
    if args.encoder_checkpoint_s3 != expected_encoder:
        raise SystemExit(f"--encoder-checkpoint-s3 must be {expected_encoder}")
    if args.task_lang_table_s3 != expected_lang:
        raise SystemExit(f"--task-lang-table-s3 must be {expected_lang}")
    if args.tap_checkpoint_s3.rstrip("/") != INIT_S3:
        raise SystemExit(f"--tap-checkpoint-s3 must be the recipe-matched frozen H300+MG checkpoint {INIT_S3}")
    _content_addressed_json(
        args.workspace_artifacts_manifest_s3,
        expected_workspace_manifest,
        flag="--workspace-artifacts-manifest-s3",
    )
    _content_addressed_json(
        args.task_prompt_manifest_s3,
        expected_prompt_manifest,
        flag="--task-prompt-manifest-s3",
    )
    _content_addressed_json(
        args.tap_tree_manifest_s3,
        expected_tap_manifest,
        flag="--tap-tree-manifest-s3",
    )
    return {
        "encoder_id": encoder_id,
        "workspace_window": 1,
        "encoder_checkpoint": {
            "uri": expected_encoder,
            "sha256": encoder_sha,
        },
        "task_lang_table": {"uri": expected_lang, "sha256": lang_sha},
        "task_prompt_manifest": {
            "uri": expected_prompt_manifest,
            "file_sha256": prompt_manifest_sha,
            "schema_version": 1,
        },
        "tap_prompt": {
            "mode": "terse",
            "global_language_mode": GLOBAL_LANGUAGE_MODE,
            "canonical_task_prompt_manifest_id": prompt_manifest_sha,
            "demo_derived": False,
        },
        "frozen_tap_checkpoint": {
            "uri": INIT_S3,
            "tree_manifest_uri": expected_tap_manifest,
            "tree_manifest_sha256": tap_manifest_sha,
        },
        "workspace_artifacts_manifest": {
            "uri": expected_workspace_manifest,
            "sha256": workspace_manifest_sha,
            "schema_version": 1,
        },
    }


_PTRM_EVAL_SELECTS = ("q", "random", "mean")


def _ptrm_eval_plan(args: argparse.Namespace, workspace: dict | None) -> dict | None:
    """The PTRM (H9) inference triple, or None when this eval does not sweep it.

    K/sigma/selection are the WHOLE experiment on a fixed PTRM checkpoint (E0 K=1 sigma=0, E1
    best-Q, E2 random-select), so they are sealed like any other protocol constant and exported to
    the serve process, which reads them at policy-construction time.  Three rules, all fail-closed:

    * ALL THREE OR NONE.  A half-specified triple would silently take serve's defaults and file a
      K=1/sigma=0 number under an E1 label, which is the one failure this launcher exists to prevent.
    * The `ptrm_eval` key is written ONLY when the triple is given.  An always-present
      `"ptrm_eval": null` would change every existing eval spec's canonical JSON and therefore every
      live eval_run_id — the same provenance break the train launcher refuses for its `ptrm` block.
    * The knobs require an omega-conditioned serve.  On a base/robottt_fast arm the serve process
      never reads them, so accepting them would produce a PTRM-labelled run that is bit-identical to
      its own control.  (serve additionally refuses non-deterministic knobs against a checkpoint
      with no PTRM subtree, so a wrong-checkpoint sweep dies before the first rollout.)
    """
    supplied = {
        "--ptrm-eval-k": args.ptrm_eval_k,
        "--ptrm-eval-sigma": args.ptrm_eval_sigma,
        "--ptrm-eval-select": args.ptrm_eval_select,
    }
    if all(value is None for value in supplied.values()):
        return None
    missing = sorted(flag for flag, value in supplied.items() if value is None)
    if missing:
        raise SystemExit(
            "the PTRM eval knobs are one sealed triple (--ptrm-eval-k, --ptrm-eval-sigma, "
            "--ptrm-eval-select); missing " + ", ".join(missing)
        )
    if workspace is None:
        raise SystemExit(
            f"{args.arm} serves no omega conditioner, so the PTRM eval knobs would be ignored; "
            "they belong on the omega-conditioned arms (s1/s2/q1/q3)"
        )
    if type(args.ptrm_eval_k) is not int or args.ptrm_eval_k < 1:
        raise SystemExit(f"--ptrm-eval-k must be an integer >= 1, got {args.ptrm_eval_k!r}")
    sigma = float(args.ptrm_eval_sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise SystemExit(f"--ptrm-eval-sigma must be finite and >= 0, got {args.ptrm_eval_sigma!r}")
    if args.ptrm_eval_select not in _PTRM_EVAL_SELECTS:
        raise SystemExit(f"--ptrm-eval-select must be one of {_PTRM_EVAL_SELECTS}, got {args.ptrm_eval_select!r}")
    return {"k": int(args.ptrm_eval_k), "sigma": sigma, "select": args.ptrm_eval_select}


def build_plan(args: argparse.Namespace, source_dir: pathlib.Path) -> dict:
    if type(args.attempt_index) is not int or args.attempt_index < 1:
        raise SystemExit("--attempt-index must be a positive integer")
    root = study_root(args.user)
    wsmv2_sha = content_addressed_archive(args.wsmv2_source_s3, component="wsmv2", root=root)
    openpi_sha = content_addressed_archive(args.openpi_source_s3, component="openpi", root=root)
    container_sha = image_digest(args.image_uri)
    training_manifest_sha = _require_sha(args.training_manifest_sha256, "--training-manifest-sha256")
    training_manifest_file_sha = _require_sha(args.training_manifest_file_sha256, "--training-manifest-file-sha256")
    episode_manifest_sha = _require_sha(args.episode_manifest_sha256, "--episode-manifest-sha256")
    checkpoint_tree_sha = _require_sha(args.checkpoint_tree_manifest_sha256, "--checkpoint-tree-manifest-sha256")
    train_completion_sha = _require_sha(args.train_completion_claim_sha256, "--train-completion-claim-sha256")

    run_match = _TRAIN_RUN_ID.fullmatch(args.training_run_id)
    if run_match is None:
        raise SystemExit("--training-run-id must have form <arm>-<16 lowercase hex> for a servable arm")
    if run_match.group(1) != args.arm:
        raise SystemExit(f"training run {args.training_run_id} belongs to {run_match.group(1)}, not {args.arm}")
    if args.checkpoint_step < 1:
        raise SystemExit("--checkpoint-step must be a positive final-step index")
    # Only-final-checkpoints is enforced by the completion claim, not a schedule constant: the
    # step-<N>.complete.json path is DERIVED from --checkpoint-step and content-validated below,
    # and Stage-S runs write that claim only at their true final step (no mid-run checkpoint
    # sync, no resume). The former `!= FINAL_STEP` gate hard-coded the 60k-era schedule and
    # rejected legitimately longer sealed runs (first hit: the 120k PTRM arm, 2026-08-09).
    if args.envs_per_gpu not in K_CHOICES:
        raise SystemExit(f"--envs-per-gpu must be one of {K_CHOICES}")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        raise SystemExit("--guidance-scale must be finite and nonnegative")
    if args.arm != "s2" and args.guidance_scale != 1.0:
        raise SystemExit("only S2 sweeps guidance; every other arm requires --guidance-scale=1.0")

    # E1's sealed 50x100 held-out-reset manifest lives at the heldout50 artifact path (the filename
    # is sha256 of the file bytes; the embedded manifest_sha256 is the protocol-level identity).
    expected_episode_manifest = f"{root}/manifests/artifacts/eval/heldout50/{episode_manifest_sha}.json"
    _content_addressed_json(
        args.episode_manifest_s3,
        expected_episode_manifest,
        flag="--episode-manifest-s3",
    )
    training_manifest_s3 = f"{root}/manifests/runs/train/{args.training_run_id}.json"
    train_completion_claim_s3 = (
        f"{root}/manifests/claims/train/{args.training_run_id}/step-{args.checkpoint_step}.complete.json"
    )
    _content_addressed_json(
        args.train_completion_claim_s3,
        train_completion_claim_s3,
        flag="--train-completion-claim-s3",
    )
    checkpoint_s3 = f"{root}/checkpoints/pi05/{args.arm}/{args.training_run_id}/{args.checkpoint_step}"
    checkpoint_tree_manifest_s3 = (
        f"{root}/manifests/artifacts/checkpoints/{args.training_run_id}/"
        f"step-{args.checkpoint_step}/{checkpoint_tree_sha}.json"
    )
    _content_addressed_json(
        args.checkpoint_tree_manifest_s3,
        checkpoint_tree_manifest_s3,
        flag="--checkpoint-tree-manifest-s3",
    )
    canary_tasks = None
    if args.canary_tasks:
        canary_tasks = [task.strip() for task in args.canary_tasks.split(",") if task.strip()]
        if not canary_tasks or len(canary_tasks) != len(set(canary_tasks)):
            raise SystemExit("--canary-tasks must be a non-empty, duplicate-free comma list")
        if len(canary_tasks) >= NUM_TASKS:
            raise SystemExit("a canary must be a strict task subset; run the decisive eval instead")

    workspace = _workspace_plan(args, root)
    ptrm_eval = _ptrm_eval_plan(args, workspace)
    # Serve-side arm->interface map. Keep in lockstep with validate_stage_s_eval_inputs.py,
    # eval_manifest.validate_evaluation_provenance, internal_training/robocasa/aggregate_eval.py,
    # and the entry's server-launch branches.
    interface = {
        "s0": "base",
        "s1": "tanh",
        "s2": "cfg2",
        "s3": "base",
        "q0": "base",
        "q1": "tanh",
        "q2": "robottt_fast",
        "q3": "tanh_robottt",
        # H13 R1-R4 are aux-only: every H13 subtree is train-time and is dropped at load by
        # BaseModel.load(remove_extra_params=True), so they serve through the PLAIN BASE interface,
        # exactly like s3 (whose train interface is `jepa` but whose serve interface is `base`).
        "h13a": "base",
        "h13b": "base",
        "h13c": "base",
        "h13d": "base",
        "h13c2": "base",
        "h13d2": "base",
        # gdn8 arms KEEP the gated-DeltaNet conditioner at serve, so they serve through `tanh`
        # exactly like s1 — only the H13 aux subtrees are stripped.
        "h13e": "tanh",
        "h13f": "tanh",
        "h13g": "tanh",
        "h13h": "tanh",
        "h13g2": "tanh",
        "h13h2": "tanh",
    }[args.arm]
    # The entry derives the same split (base -> stateless_v1, everything else -> per_env) and
    # refuses a mismatch, so s3 must land in the stateless set with s0/q0.
    state_mode = (
        "stateless_v1"
        if args.arm in ("s0", "s3", "q0", "h13a", "h13b", "h13c", "h13d", "h13c2", "h13d2")
        else "per_env_isolated_v1"
    )

    with prepared_source_bundle(
        source_dir,
        ENTRY,
        {"SAGEMAKER_PROGRAM": ENTRY},
        args.secrets_manager_arn,
    ) as (staged_source, _safe_entry, _safe_environment):
        internal_training_sha = source_tree_sha256(staged_source)

    spec = {
        "schema_version": 1,
        "kind": ("pi_stage_s_robocasa_eval_canary" if canary_tasks else "pi_stage_s_robocasa_eval_run"),
        "study": STUDY,
        "arm": args.arm,
        "interface": interface,
        "training_run": {
            "run_id": args.training_run_id,
            "manifest_uri": training_manifest_s3,
            "manifest_sha256": training_manifest_sha,
            "manifest_file_sha256": training_manifest_file_sha,
            "completion_claim": {
                "uri": train_completion_claim_s3,
                "file_sha256": train_completion_sha,
                "schema_version": 1,
            },
            "checkpoint_uri": checkpoint_s3,
            "checkpoint_step": args.checkpoint_step,
            "checkpoint_tree_manifest": {
                "uri": checkpoint_tree_manifest_s3,
                "file_sha256": checkpoint_tree_sha,
                "schema_version": 1,
            },
        },
        "protocol": {
            "benchmark": "RoboCasa",
            "split": "target",
            "task_sets": TASK_SETS.split(","),
            "num_tasks": NUM_TASKS,
            "episodes_per_task": EPISODES_PER_TASK,
            "rollouts_per_reset": 1,
            "total_rollouts": (
                len(canary_tasks) * EPISODES_PER_TASK if canary_tasks else NUM_TASKS * EPISODES_PER_TASK
            ),
            # E4 shakedown of the exact serve+rollout stack (K batching, state isolation,
            # throughput, memory) on a strict task subset. Never decisive; results/manifests live
            # under canary-only prefixes.
            "canary": {"tasks": canary_tasks} if canary_tasks else None,
            "episode_manifest": {
                "uri": expected_episode_manifest,
                "file_sha256": episode_manifest_sha,
                "schema_version": 2,
            },
            "seed": EPISODE_BASE_SEED,
            "policy_noise_kind": POLICY_NOISE_KIND,
            "replan_steps": CADENCE,
            "exec_steps": CADENCE,
            "workspace_stride": CADENCE,
            "num_workers": NUM_WORKERS,
            "envs_per_gpu": args.envs_per_gpu,
            "server_state_mode": state_mode,
            "video": args.video,
        },
        "workspace_representation": workspace,
        "guidance": {
            "scale": args.guidance_scale if args.arm == "s2" else None,
            "cfg_drop_probability_at_train": 0.2 if args.arm == "s2" else None,
            "tanh_gate": args.arm in ("s1", "q1", "q3"),
            "legacy_direct_token": False,
            "future_or_demo_conditioning": False,
        },
        "sources": {
            "wsmv2": {"uri": args.wsmv2_source_s3, "sha256": wsmv2_sha},
            "openpi": {"uri": args.openpi_source_s3, "sha256": openpi_sha},
            "internal_training": {
                "sanitized_source_tree_sha256": internal_training_sha,
                "entry_path": ENTRY,
                "entry_sha256": _sha256_file(source_dir / ENTRY),
            },
            "image": {"uri": args.image_uri, "sha256": container_sha},
        },
        "infrastructure": {
            "execution_account": EXECUTION_ACCOUNT,
            "storage_account": STORAGE_ACCOUNT,
            "queue": args.queue,
            "role": args.role,
            "instance_type": INSTANCE_TYPE,
            "priority": args.priority,
            "max_run_seconds": args.max_run_seconds,
            "retry_attempts": RETRY["attempts"],
            "attempt_index": args.attempt_index,
        },
    }
    if ptrm_eval is not None:
        # Conditional key, deliberately: `_canonical_json` serializes every key, so a
        # `"ptrm_eval": null` sitting in a non-PTRM eval spec would move its spec_sha256 and rename
        # a live eval_run_id. Written here — after the spec literal — so the diff cannot leak into
        # the runs that already exist.
        spec["ptrm_eval"] = ptrm_eval
    spec_sha = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
    run_prefix = "evalcanary" if canary_tasks else "eval"
    eval_run_id = f"{run_prefix}-{args.arm}-step{args.checkpoint_step}-{spec_sha[:16]}"
    eval_root = f"{root}/evals/canary" if canary_tasks else f"{root}/evals"
    manifest_kind = "eval_canary" if canary_tasks else "eval"
    results_s3 = (
        f"{eval_root}/robocasa/pi05/{args.arm}/{args.training_run_id}/step-{args.checkpoint_step}/{eval_run_id}"
    )
    manifest_s3 = f"{root}/manifests/runs/{manifest_kind}/{eval_run_id}.json"
    producer_claim_s3 = f"{root}/manifests/claims/{manifest_kind}/{eval_run_id}.json"
    manifest, manifest_json = _seal_manifest(
        {
            **spec,
            "eval_run_id": eval_run_id,
            "spec_sha256": spec_sha,
            "results_s3": results_s3,
            "manifest_s3": manifest_s3,
            "producer_claim_s3": producer_claim_s3,
        }
    )
    manifest_file_sha256 = hashlib.sha256((manifest_json + "\n").encode("utf-8")).hexdigest()

    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "MODEL": "pi05",
        "BACKBONE": "pi",
        "PI_STAGE_S_INTERFACE": interface,
        "CKPT_S3": checkpoint_s3,
        "STEP": str(args.checkpoint_step),
        "RESULTS_S3": results_s3,
        "NUM_TRIALS": str(EPISODES_PER_TASK),
        "EVAL_SPLIT": "target",
        "TASK_SETS": TASK_SETS,
        "NUM_WORKERS": str(NUM_WORKERS),
        "VIDEO": args.video,
        "REPLAN_STEPS": str(CADENCE),
        "EXEC_STEPS": str(CADENCE),
        "SEED": str(EPISODE_BASE_SEED),
        "WSM_HELDOUT": "1",
        "WSM_ROLLOUTS_PER_DEMO": "1",
        "WSM_ENVS_PER_GPU": str(args.envs_per_gpu),
        "PI_WSM_SERVER_STATE_MODE": state_mode,
        **({"STAGE_S_CANARY": "1", "WSM_TASKS": ",".join(canary_tasks)} if canary_tasks else {}),
        "OPENPI_FORK_S3": args.openpi_source_s3,
        "OPENPI_FORK_SHA256": openpi_sha,
        "WSM_REPO_S3": args.wsmv2_source_s3,
        "WSM_REPO_SHA256": wsmv2_sha,
        "EPISODE_MANIFEST_S3": expected_episode_manifest,
        "EPISODE_MANIFEST_SHA256": episode_manifest_sha,
        "TRAIN_RUN_ID": args.training_run_id,
        "TRAIN_RUN_MANIFEST_S3": training_manifest_s3,
        "TRAIN_RUN_MANIFEST_SHA256": training_manifest_sha,
        "TRAIN_RUN_MANIFEST_FILE_SHA256": training_manifest_file_sha,
        "TRAIN_COMPLETION_CLAIM_S3": train_completion_claim_s3,
        "TRAIN_COMPLETION_CLAIM_SHA256": train_completion_sha,
        "CHECKPOINT_TREE_MANIFEST_S3": checkpoint_tree_manifest_s3,
        "CHECKPOINT_TREE_MANIFEST_SHA256": checkpoint_tree_sha,
        "EVAL_RUN_ID": eval_run_id,
        "EVAL_RUN_MANIFEST_SOURCE": STAGED_MANIFEST_NAME,
        "EVAL_RUN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "EVAL_RUN_MANIFEST_FILE_SHA256": manifest_file_sha256,
        "EVAL_RUN_MANIFEST_S3": manifest_s3,
        "EVAL_PRODUCER_CLAIM_S3": producer_claim_s3,
        "STAGE_S_REQUIRE_BATCHING": "1" if args.envs_per_gpu > 1 else "0",
    }
    if workspace is not None:
        environment.update(
            {
                "WSM_K_WINDOW": "1",
                "WSM_ENCODER_ID": workspace["encoder_id"],
                "ENCODER_CKPT_S3": workspace["encoder_checkpoint"]["uri"],
                "ENCODER_CKPT_SHA256": workspace["encoder_checkpoint"]["sha256"],
                "TASK_LANG_TABLE_S3": workspace["task_lang_table"]["uri"],
                "TASK_LANG_TABLE_SHA256": workspace["task_lang_table"]["sha256"],
                "WSM_TAP_PROMPT": workspace["tap_prompt"]["mode"],
                "WSM_GLOBAL_LANGUAGE_MODE": workspace["tap_prompt"]["global_language_mode"],
                "TASK_PROMPT_MANIFEST_S3": workspace["task_prompt_manifest"]["uri"],
                "TASK_PROMPT_MANIFEST_SHA256": workspace["task_prompt_manifest"]["file_sha256"],
                "TAP_CKPT_S3": workspace["frozen_tap_checkpoint"]["uri"],
                "TAP_TREE_MANIFEST_S3": workspace["frozen_tap_checkpoint"]["tree_manifest_uri"],
                "TAP_TREE_MANIFEST_SHA256": workspace["frozen_tap_checkpoint"]["tree_manifest_sha256"],
                "WORKSPACE_ARTIFACT_MANIFEST_S3": workspace["workspace_artifacts_manifest"]["uri"],
                "WORKSPACE_ARTIFACT_MANIFEST_SHA256": workspace["workspace_artifacts_manifest"]["sha256"],
                "GUIDANCE_SCALE": str(args.guidance_scale),
            }
        )
    if ptrm_eval is not None:
        # The serve process reads these at policy-construction time (they ride the same Pi0Config
        # path the deltanet geometry does), so they must be in the JOB environment, not a post-build
        # patch. Absent by default: unset means K=1/sigma=0/'q', serve's deterministic PTRM-off read.
        environment.update(
            {
                "WSM_PTRM_EVAL_K": str(ptrm_eval["k"]),
                "WSM_PTRM_EVAL_SIGMA": repr(ptrm_eval["sigma"]),
                "WSM_PTRM_EVAL_SELECT": ptrm_eval["select"],
            }
        )
    forbidden = {
        "WSM_CFG",
        "WSM_EVAL",
        "WSM_DEMO_CFG",
        "WSM_LEGACY_TOKEN_INJECTION",
        "WSM_CFG_WITH_FUTURE",
    }
    leaked = sorted(forbidden.intersection(environment))
    if leaked:
        raise AssertionError(f"Stage-S eval selected forbidden legacy variables: {leaked}")
    oversized = {
        key: len(value.encode("utf-8")) for key, value in environment.items() if len(value.encode("utf-8")) > 512
    }
    if oversized:
        raise AssertionError(f"SageMaker environment value exceeds 512 bytes: {oversized}")
    return {
        "canary": bool(canary_tasks),
        "eval_run_id": eval_run_id,
        "results_s3": results_s3,
        "manifest_s3": manifest_s3,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "manifest_file_sha256": manifest_file_sha256,
        "source_tree_sha256": internal_training_sha,
        "environment": environment,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "s0",
            "s1",
            "s2",
            "s3",
            "h13a",
            "h13b",
            "h13c",
            "h13d",
            "h13c2",
            "h13d2",
            "h13e",
            "h13f",
            "h13g",
            "h13h",
            "h13g2",
            "h13h2",
            "q0",
            "q1",
            "q2",
            "q3",
        ),
    )
    parser.add_argument("--user", default=DEFAULT_OWNER, help="account-124 storage owner")
    parser.add_argument("--source-dir", default=None, help="local internal_training source tree")
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--training-manifest-file-sha256", required=True)
    parser.add_argument("--train-completion-claim-s3", required=True)
    parser.add_argument("--train-completion-claim-sha256", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--checkpoint-tree-manifest-s3", required=True)
    parser.add_argument("--checkpoint-tree-manifest-sha256", required=True)
    parser.add_argument("--episode-manifest-s3", required=True)
    parser.add_argument("--episode-manifest-sha256", required=True)
    parser.add_argument("--wsmv2-source-s3", required=True)
    parser.add_argument("--openpi-source-s3", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--envs-per-gpu", type=int, choices=K_CHOICES, default=1)
    parser.add_argument(
        "--canary-tasks",
        default=None,
        help="comma list of tasks for an E4 shakedown canary (K batching/state isolation/"
        "throughput/memory). Canary runs use canary-only ids and prefixes, are never decisive, "
        "and are exempt from the decisive-eval submission blockers.",
    )
    parser.add_argument("--video", choices=("none", "first", "all"), default="first")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--encoder-id", default=None)
    parser.add_argument("--encoder-checkpoint-s3", default=None)
    parser.add_argument("--encoder-checkpoint-sha256", default=None)
    parser.add_argument("--task-lang-table-s3", default=None)
    parser.add_argument("--task-lang-table-sha256", default=None)
    parser.add_argument("--task-prompt-manifest-s3", default=None)
    parser.add_argument("--task-prompt-manifest-sha256", default=None)
    parser.add_argument(
        "--task-prompt-namespace",
        default="robocasa_target50",
        choices=("robocasa_target50", "remembench13"),
        help="dataset namespace of the canonical task-prompt manifest",
    )
    parser.add_argument("--tap-checkpoint-s3", default=None)
    parser.add_argument("--tap-tree-manifest-s3", default=None)
    parser.add_argument("--tap-tree-manifest-sha256", default=None)
    parser.add_argument("--workspace-artifacts-manifest-s3", default=None)
    parser.add_argument("--workspace-artifacts-manifest-sha256", default=None)
    # PTRM (H9) inference triple. All three or none; omitting them leaves the eval spec, its
    # spec_sha256 and therefore the eval_run_id byte-identical to every eval submitted before this
    # flag existed, so no live result is orphaned.
    parser.add_argument(
        "--ptrm-eval-k",
        type=int,
        default=None,
        help="PTRM parallel rollouts K (>=1); requires --ptrm-eval-sigma and --ptrm-eval-select",
    )
    parser.add_argument(
        "--ptrm-eval-sigma",
        type=float,
        default=None,
        help="PTRM per-micro-step noise scale (finite, >=0); 0.0 is the deterministic read",
    )
    parser.add_argument(
        "--ptrm-eval-select",
        choices=_PTRM_EVAL_SELECTS,
        default=None,
        help="which rollout is decoded: q (best-Q, the claim), random (verifier control), mean (ensembling control)",
    )
    parser.add_argument(
        "--attempt-index",
        type=int,
        default=1,
        help="1 for the first attempt; increment only after fresh user approval for a retry",
    )
    add_guardrail_arguments(parser)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    validate_and_confirm(args)
    source_dir = resolve_source_dir(args.source_dir)
    plan = build_plan(args, source_dir)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    owner = args.user.replace(".", "-")
    job_name = f"{owner}-pi-{args.arm}-eval-{plan['eval_run_id'][-16:]}-{stamp}"[:63].rstrip("-")
    print(
        f"arm={args.arm} interface={plan['manifest']['interface']} "
        f"eval_run_id={plan['eval_run_id']}\n"
        f"  checkpoint={plan['manifest']['training_run']['checkpoint_uri']}\n"
        f"  episode_manifest={args.episode_manifest_s3} "
        f"sha256={args.episode_manifest_sha256}\n"
        f"  results={plan['results_s3']}\n"
        f"  manifest={plan['manifest_s3']} "
        f"sha256={plan['manifest']['manifest_sha256']}\n"
        f"  K={args.envs_per_gpu} cadence={CADENCE} workers={NUM_WORKERS}\n"
        f"  queue={args.queue} priority={args.priority} max_run={args.max_run_seconds}s "
        f"retry_attempts={RETRY['attempts']} dry={args.dry_run}"
        # Conditional line: a PTRM-free eval prints exactly what it printed before the knobs existed.
        + (
            f"\n  ptrm_eval K={ptrm['k']} sigma={ptrm['sigma']:g} select={ptrm['select']}"
            if (ptrm := plan["manifest"].get("ptrm_eval")) is not None
            else ""
        )
    )
    # A real job gets a unique producer identity. The entry claims the deterministic results prefix
    # with create-once S3 semantics, so two identical submissions cannot write the same shards.
    plan["environment"]["EVAL_PRODUCER_ID"] = job_name

    if args.dry_run:
        print("  [DRY RUN: offline; no AWS SDK load, S3 lookup, source upload, or submission]")
        print(json.dumps(plan["manifest"], sort_keys=True, indent=2))
        if plan["canary"]:
            print("  CANARY: exempt from decisive-eval blockers; submits with --confirm-submit")
        elif SUBMISSION_BLOCKERS:
            print("  SUBMISSION BLOCKED pending:")
            for blocker in SUBMISSION_BLOCKERS:
                print(f"    - {blocker}")
        else:
            print("  E4 shakedown passed; submits with per-run approval + --confirm-submit")
        return

    if not plan["canary"] and SUBMISSION_BLOCKERS:
        raise SystemExit("Stage-S eval submission remains fail-closed: " + "; ".join(SUBMISSION_BLOCKERS))

    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=args.image_uri,
        instance_type=INSTANCE_TYPE,
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.arm", "Value": args.arm},
            {"Key": "wsm.eval_run_id", "Value": plan["eval_run_id"]},
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
        expected_source_tree_sha256=plan["source_tree_sha256"],
        staged_source_files={STAGED_MANIFEST_NAME: plan["manifest_json"] + "\n"},
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
