"""Offline contract tests for the focused pi0.5 Stage-S evaluator launcher."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "launch"
sys.path.insert(0, str(LAUNCH_DIR))
LAUNCHER_PATH = LAUNCH_DIR / "submit_pi_stage_s_eval.py"
SPEC = importlib.util.spec_from_file_location("submit_pi_stage_s_eval_test", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
stage_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_eval)


W_SHA = "a" * 64
O_SHA = "b" * 64
IMAGE_SHA = "c" * 64
TRAIN_MANIFEST_SHA = "d" * 64
TRAIN_MANIFEST_FILE_SHA = "e" * 64
COMPLETION_SHA = "f" * 64
TREE_SHA = "1" * 64
EPISODE_SHA = "2" * 64
ENCODER_ID = "3" * 64
ENCODER_SHA = "4" * 64
LANG_SHA = "5" * 64
PROMPT_SHA = "6" * 64
TAP_TREE_SHA = "7" * 64
WORKSPACE_SHA = "8" * 64


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "internal_training"
    source.mkdir()
    entry = source / stage_eval.ENTRY
    entry.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    entry.chmod(0o755)
    (source / "support.py").write_text("VALUE = 1\n", encoding="utf-8")
    return source


def _workspace_extra() -> tuple[str, ...]:
    root = stage_eval.study_root(stage_eval.DEFAULT_OWNER)
    return (
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
        stage_eval.INIT_S3,
        "--tap-tree-manifest-s3",
        f"{root}/manifests/artifacts/tap/{TAP_TREE_SHA}.json",
        "--tap-tree-manifest-sha256",
        TAP_TREE_SHA,
        "--workspace-artifacts-manifest-s3",
        f"{root}/manifests/artifacts/workspace/{ENCODER_ID}/{WORKSPACE_SHA}.json",
        "--workspace-artifacts-manifest-sha256",
        WORKSPACE_SHA,
    )


def _args(arm: str, source: Path, *extra: str):
    root = stage_eval.study_root(stage_eval.DEFAULT_OWNER)
    run_id = f"{arm}-{'9' * 16}"
    argv = [
        "--dry-run",
        "--arm",
        arm,
        "--source-dir",
        str(source),
        "--training-run-id",
        run_id,
        "--training-manifest-sha256",
        TRAIN_MANIFEST_SHA,
        "--training-manifest-file-sha256",
        TRAIN_MANIFEST_FILE_SHA,
        "--train-completion-claim-s3",
        (f"{root}/manifests/claims/train/{run_id}/step-{stage_eval.FINAL_STEP}.complete.json"),
        "--train-completion-claim-sha256",
        COMPLETION_SHA,
        "--checkpoint-step",
        str(stage_eval.FINAL_STEP),
        "--checkpoint-tree-manifest-s3",
        (f"{root}/manifests/artifacts/checkpoints/{run_id}/step-{stage_eval.FINAL_STEP}/{TREE_SHA}.json"),
        "--checkpoint-tree-manifest-sha256",
        TREE_SHA,
        "--episode-manifest-s3",
        (f"{root}/manifests/artifacts/eval/heldout50/{EPISODE_SHA}.json"),
        "--episode-manifest-sha256",
        EPISODE_SHA,
        "--wsmv2-source-s3",
        f"{root}/code/wsmv2/{W_SHA}.tgz",
        "--openpi-source-s3",
        f"{root}/code/openpi/{O_SHA}.tgz",
        "--image-uri",
        (f"141701954645.dkr.ecr.us-west-2.amazonaws.com/stage-s@sha256:{IMAGE_SHA}"),
        *extra,
    ]
    return stage_eval.make_parser().parse_args(argv)


def _command(args) -> list[str]:
    command = [sys.executable, str(LAUNCHER_PATH)]
    for key, value in vars(args).items():
        if key in {"dry_run", "confirm_submit"} or value is None:
            continue
        command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def test_s0_exact_plan_is_deterministic_and_workspace_free(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    first = stage_eval.build_plan(args, source)
    second = stage_eval.build_plan(args, source)
    assert first == second
    protocol = first["manifest"]["protocol"]
    assert protocol["num_tasks"] == 50
    assert protocol["episodes_per_task"] == 100
    assert protocol["total_rollouts"] == 5000
    assert protocol["task_sets"] == [
        "atomic_seen",
        "composite_seen",
        "composite_unseen",
    ]
    assert protocol["replan_steps"] == protocol["exec_steps"] == 8
    assert protocol["policy_noise_kind"] == "pi_diffusion_sha256_v1"
    assert first["manifest"]["workspace_representation"] is None
    assert first["manifest"]["infrastructure"]["retry_attempts"] == 1
    assert first["manifest"]["infrastructure"]["attempt_index"] == 1
    assert first["manifest"]["infrastructure"]["priority"] == 600
    assert first["manifest"]["infrastructure"]["max_run_seconds"] == 432000
    assert "/manifests/runs/eval/" in first["manifest_s3"]
    assert "/manifests/claims/eval/" in first["environment"]["EVAL_PRODUCER_CLAIM_S3"]
    assert first["environment"]["PI_STAGE_S_INTERFACE"] == "base"
    assert first["environment"]["STAGE_S_REQUIRE_BATCHING"] == "0"
    assert "ENCODER_CKPT_S3" not in first["environment"]

    args.attempt_index = 2
    retry = stage_eval.build_plan(args, source)
    assert retry["eval_run_id"] != first["eval_run_id"]
    assert retry["manifest"]["infrastructure"]["attempt_index"] == 2


def test_canary_plan_uses_canary_prefixes_and_is_submittable(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source, "--canary-tasks", "TurnOnMicrowave,PrepareCoffee", "--envs-per-gpu", "8")
    plan = stage_eval.build_plan(args, source)
    assert plan["canary"] is True
    manifest = plan["manifest"]
    assert manifest["kind"] == "pi_stage_s_robocasa_eval_canary"
    assert plan["eval_run_id"].startswith("evalcanary-s0-")
    assert "/evals/canary/" in plan["results_s3"]
    assert "/manifests/runs/eval_canary/" in plan["manifest_s3"]
    assert "/manifests/claims/eval_canary/" in plan["environment"]["EVAL_PRODUCER_CLAIM_S3"]
    assert manifest["protocol"]["canary"] == {"tasks": ["TurnOnMicrowave", "PrepareCoffee"]}
    assert manifest["protocol"]["total_rollouts"] == 200
    assert plan["environment"]["STAGE_S_CANARY"] == "1"
    assert plan["environment"]["WSM_TASKS"] == "TurnOnMicrowave,PrepareCoffee"
    assert plan["environment"]["STAGE_S_REQUIRE_BATCHING"] == "1"

    decisive = stage_eval.build_plan(_args("s0", source), source)
    assert decisive["canary"] is False
    assert decisive["manifest"]["protocol"]["canary"] is None

    with pytest.raises(SystemExit, match="duplicate-free"):
        stage_eval.build_plan(_args("s0", source, "--canary-tasks", "TaskA,TaskA"), source)
    with pytest.raises(SystemExit, match="strict task subset"):
        stage_eval.build_plan(_args("s0", source, "--canary-tasks", ",".join(f"T{i}" for i in range(50))), source)


@pytest.mark.parametrize(
    ("arm", "interface", "guidance"),
    (
        ("s1", "tanh", 1.0),
        ("s2", "cfg2", 2.0),
        # q1 reuses the tanh serve contract (identical serve-time inputs to s1); q3 serves the
        # combined tanh_robottt interface. Both are omega-conditioned, so both require the full
        # workspace artifact set and the tanh-style fixed guidance of 1.0.
        ("q1", "tanh", 1.0),
        ("q3", "tanh_robottt", 1.0),
    ),
)
def test_workspace_arms_pin_demo_independent_representation(tmp_path, arm, interface, guidance):
    source = _source(tmp_path)
    args = _args(
        arm,
        source,
        "--envs-per-gpu",
        "8",
        "--guidance-scale",
        str(guidance),
        *_workspace_extra(),
    )
    plan = stage_eval.build_plan(args, source)
    workspace = plan["manifest"]["workspace_representation"]
    assert plan["manifest"]["interface"] == interface
    assert plan["environment"]["PI_STAGE_S_INTERFACE"] == interface
    assert workspace["encoder_id"] == ENCODER_ID
    assert workspace["workspace_window"] == 1
    assert workspace["tap_prompt"] == {
        "mode": "terse",
        "global_language_mode": "canonical_terse_task_instruction",
        "canonical_task_prompt_manifest_id": PROMPT_SHA,
        "demo_derived": False,
    }
    assert plan["environment"]["WSM_TAP_PROMPT"] == "terse"
    assert plan["environment"]["STAGE_S_REQUIRE_BATCHING"] == "1"
    assert plan["manifest"]["protocol"]["server_state_mode"] == "per_env_isolated_v1"
    for forbidden in (
        "WSM_CFG",
        "WSM_EVAL",
        "WSM_DEMO_CFG",
        "WSM_LEGACY_TOKEN_INJECTION",
        "WSM_CFG_WITH_FUTURE",
    ):
        assert forbidden not in plan["environment"]


@pytest.mark.parametrize("arm", ("s0", "s3", "q0"))
def test_base_serving_arms_are_workspace_free_and_stateless(tmp_path, arm):
    """s0, q0 and s3 all serve the plain `base` interface.

    s3 (JEPA+SigReg) consumes omega only as a TRAIN-time aux target — its `wsm_jepa_head` subtree
    is never touched by sample_actions — so its eval job must be indistinguishable from s0's: no
    workspace artifacts in either direction, stateless server (the entry derives exactly this from
    interface==base and refuses a mismatch), and no guidance sweep.
    """
    source = _source(tmp_path)
    plan = stage_eval.build_plan(_args(arm, source), source)
    assert plan["manifest"]["interface"] == "base"
    assert plan["environment"]["PI_STAGE_S_INTERFACE"] == "base"
    assert plan["manifest"]["workspace_representation"] is None
    assert plan["manifest"]["protocol"]["server_state_mode"] == "stateless_v1"
    assert plan["environment"]["PI_WSM_SERVER_STATE_MODE"] == "stateless_v1"
    assert plan["manifest"]["guidance"]["scale"] is None
    assert plan["manifest"]["guidance"]["tanh_gate"] is False
    for key in (
        "WSM_ENCODER_ID",
        "ENCODER_CKPT_S3",
        "TASK_LANG_TABLE_S3",
        "TASK_PROMPT_MANIFEST_S3",
        "TAP_CKPT_S3",
        "WORKSPACE_ARTIFACT_MANIFEST_S3",
        "GUIDANCE_SCALE",
        "WSM_K_WINDOW",
    ):
        assert key not in plan["environment"]
    with pytest.raises(SystemExit, match=f"{arm} forbids workspace"):
        stage_eval.build_plan(_args(arm, source, *_workspace_extra()), source)
    with pytest.raises(SystemExit, match="guidance-scale=1.0"):
        stage_eval.build_plan(_args(arm, source, "--guidance-scale", "2.0"), source)


def test_arm_step_and_workspace_mismatches_fail_closed(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    args.checkpoint_step = 5000
    with pytest.raises(SystemExit, match="only final step"):
        stage_eval.build_plan(args, source)

    args = _args("s0", source)
    args.training_run_id = f"s1-{'9' * 16}"
    with pytest.raises(SystemExit, match="belongs to s1"):
        stage_eval.build_plan(args, source)

    with pytest.raises(SystemExit, match="require all workspace"):
        stage_eval.build_plan(_args("s1", source), source)

    with pytest.raises(SystemExit, match="s0 forbids workspace"):
        stage_eval.build_plan(_args("s0", source, *_workspace_extra()), source)

    # Both directions across the Q arms: the omega-conditioned serves (q1/q3) REQUIRE the workspace
    # artifact set; the workspace-free serves (q0 base, q2 robottt_fast) FORBID it.
    with pytest.raises(SystemExit, match="require all workspace"):
        stage_eval.build_plan(_args("q3", source), source)
    with pytest.raises(SystemExit, match="require all workspace"):
        stage_eval.build_plan(_args("q1", source), source)
    with pytest.raises(SystemExit, match="q0 forbids workspace"):
        stage_eval.build_plan(_args("q0", source, *_workspace_extra()), source)
    with pytest.raises(SystemExit, match="q2 forbids workspace"):
        stage_eval.build_plan(_args("q2", source, *_workspace_extra()), source)
    with pytest.raises(SystemExit, match="s3 forbids workspace"):
        stage_eval.build_plan(_args("s3", source, *_workspace_extra()), source)

    # The TRAIN_RUN_ID regex must admit s3 (it was pinned to s[012] while s3 eval was absent), and
    # the arm is still taken from the run id, never from --arm.
    args = _args("s0", source)
    args.training_run_id = f"s3-{'9' * 16}"
    with pytest.raises(SystemExit, match="belongs to s3"):
        stage_eval.build_plan(args, source)


def test_dry_run_is_offline_and_real_submission_is_hard_blocked(tmp_path):
    source = _source(tmp_path)
    args = _args("s0", source)
    dry = subprocess.run(
        [*_command(args), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert dry.returncode == 0, (dry.stdout, dry.stderr)
    assert "DRY RUN: offline" in dry.stdout
    # E4 passed 2026-07-27 (evalcanary-s0-step59999-2d76eaf6e16044e6): decisive submission is no
    # longer hard-blocked, but it still refuses without the explicit --confirm-submit approval flag.
    assert "E4 shakedown passed" in dry.stdout

    unconfirmed = subprocess.run(
        _command(args),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert unconfirmed.returncode != 0
    assert "--confirm-submit" in (unconfirmed.stderr + unconfirmed.stdout)


# --------------------------------------------------------------------------------------------
# PTRM (H9) inference knobs. K/sigma/selection ARE the experiment on a fixed checkpoint (E0/E1/E2),
# so the launcher seals them, exports them to the serve process, refuses a half-specified triple —
# and, the load-bearing part, leaves every eval submitted before the flags existed byte-identical.
# --------------------------------------------------------------------------------------------
PTRM_KNOBS = ("--ptrm-eval-k", "32", "--ptrm-eval-sigma", "0.3", "--ptrm-eval-select", "q")


def test_ptrm_eval_knobs_are_sealed_into_the_spec_and_exported_to_serve(tmp_path):
    source = _source(tmp_path)
    args = _args("s1", source, "--envs-per-gpu", "8", *_workspace_extra(), *PTRM_KNOBS)
    plan = stage_eval.build_plan(args, source)
    env = plan["environment"]

    assert plan["manifest"]["ptrm_eval"] == {"k": 32, "sigma": 0.3, "select": "q"}
    assert env["WSM_PTRM_EVAL_K"] == "32"
    assert env["WSM_PTRM_EVAL_SIGMA"] == "0.3"
    assert env["WSM_PTRM_EVAL_SELECT"] == "q"
    assert max(len(value.encode("utf-8")) for value in env.values()) <= 512
    # Nothing else about the eval moves: same interface, same workspace artifacts, same protocol.
    off = stage_eval.build_plan(_args("s1", source, "--envs-per-gpu", "8", *_workspace_extra()), source)
    assert plan["manifest"]["interface"] == off["manifest"]["interface"] == "tanh"
    assert plan["manifest"]["protocol"] == off["manifest"]["protocol"]
    assert plan["manifest"]["workspace_representation"] == off["manifest"]["workspace_representation"]
    # E0/E1/E2 are distinct runs with distinct result prefixes, and each is reproducible.
    assert plan["eval_run_id"] != off["eval_run_id"]
    assert plan["results_s3"] != off["results_s3"]
    assert stage_eval.build_plan(args, source) == plan
    for select in ("random", "mean"):
        control = stage_eval.build_plan(
            _args(
                "s1",
                source,
                "--envs-per-gpu",
                "8",
                *_workspace_extra(),
                "--ptrm-eval-k",
                "32",
                "--ptrm-eval-sigma",
                "0.3",
                "--ptrm-eval-select",
                select,
            ),
            source,
        )
        assert control["manifest"]["ptrm_eval"]["select"] == select
        assert control["environment"]["WSM_PTRM_EVAL_SELECT"] == select
        assert control["eval_run_id"] != plan["eval_run_id"]

    # And the flags survive the real CLI, offline.
    dry = subprocess.run([*_command(args), "--dry-run"], capture_output=True, text=True, check=False, timeout=20)
    assert dry.returncode == 0, (dry.stdout, dry.stderr)
    assert "ptrm_eval K=32 sigma=0.3 select=q" in dry.stdout
    assert '"ptrm_eval"' in dry.stdout


def test_partial_or_invalid_ptrm_eval_knobs_are_refused(tmp_path):
    source = _source(tmp_path)
    workspace = ("--envs-per-gpu", "8", *_workspace_extra())
    # ALL THREE OR NONE: a half-specified triple would silently take serve's K=1/sigma=0 defaults
    # and file a PTRM-off number under a PTRM-on label.
    for partial in (
        ("--ptrm-eval-k", "32"),
        ("--ptrm-eval-sigma", "0.3"),
        ("--ptrm-eval-select", "q"),
        ("--ptrm-eval-k", "32", "--ptrm-eval-sigma", "0.3"),
        ("--ptrm-eval-k", "32", "--ptrm-eval-select", "random"),
        ("--ptrm-eval-sigma", "0.3", "--ptrm-eval-select", "random"),
    ):
        with pytest.raises(SystemExit, match="sealed triple"):
            stage_eval.build_plan(_args("s1", source, *workspace, *partial), source)

    # The values are validated, not merely recorded.
    with pytest.raises(SystemExit, match="ptrm-eval-k must be an integer >= 1"):
        stage_eval.build_plan(
            _args(
                "s1", source, *workspace, "--ptrm-eval-k", "0", "--ptrm-eval-sigma", "0.3", "--ptrm-eval-select", "q"
            ),
            source,
        )
    for bad_sigma in ("-0.1", "nan", "inf"):
        with pytest.raises(SystemExit, match="ptrm-eval-sigma must be finite and >= 0"):
            stage_eval.build_plan(
                _args(
                    "s1",
                    source,
                    *workspace,
                    "--ptrm-eval-k",
                    "32",
                    "--ptrm-eval-sigma",
                    bad_sigma,
                    "--ptrm-eval-select",
                    "q",
                ),
                source,
            )
    bad_select = _args("s1", source, *workspace, *PTRM_KNOBS)
    bad_select.ptrm_eval_select = "argmax"
    with pytest.raises(SystemExit, match="ptrm-eval-select must be one of"):
        stage_eval.build_plan(bad_select, source)

    # A base / robottt_fast serve never reads the knobs, so accepting them there would produce a
    # PTRM-labelled run that is bit-identical to its own control.
    for arm in ("s0", "s3", "q0", "q2"):
        with pytest.raises(SystemExit, match="serves no omega conditioner"):
            stage_eval.build_plan(_args(arm, source, *PTRM_KNOBS), source)


def test_the_ptrm_eval_key_appears_only_when_swept_so_no_live_eval_run_id_moves(tmp_path):
    """MANIFEST STABILITY. A `"ptrm_eval": null` in every spec would rename every live eval.

    eval_run_id is derived from sha256 of the canonical spec JSON, which serializes every key, so an
    unconditional key would move every id the campaign's results are filed under. Two locks:

    * structural — no knob-free plan carries the key or the env vars, on any arm;
    * pinned — the fixture eval_run_ids below were recomputed from the pre-patch launcher and are
      unchanged. The same equality was checked against the live dw32 discriminator eval args
      (eval-s1-step59999-fcc4eb81d42e3201, spec_sha256 fcc4eb81d42e3201...): its whole `--dry-run`
      stdout hashed to 76656b06fc23def717fb4a416e8a7a5792970fb54370aa5f3fe4afc275844558 both
      before and after this patch.
    """
    source = _source(tmp_path)
    frozen = {
        "s0": ("eval-s0-step59999-2992474f01d0ea97", ()),
        "s1": ("eval-s1-step59999-3d6c8acae30fb2c7", _workspace_extra()),
        "s2": ("eval-s2-step59999-c44d1e3942998af4", ("--guidance-scale", "2.0", *_workspace_extra())),
        "s3": ("eval-s3-step59999-4dfa1e9e94033224", ()),
        "q0": ("eval-q0-step59999-8dc30c2d8952a293", ()),
        "q1": ("eval-q1-step59999-a8bb6bf70d2ce41e", _workspace_extra()),
        "q3": ("eval-q3-step59999-ace13aa4318c4ef0", _workspace_extra()),
    }
    for arm, (eval_run_id, extra) in frozen.items():
        plan = stage_eval.build_plan(_args(arm, source, *extra), source)
        assert plan["eval_run_id"] == eval_run_id, arm
        assert "ptrm_eval" not in plan["manifest"], arm
        assert "ptrm_eval" not in plan["manifest_json"], arm
        for key in ("WSM_PTRM_EVAL_K", "WSM_PTRM_EVAL_SIGMA", "WSM_PTRM_EVAL_SELECT"):
            assert key not in plan["environment"], (arm, key)


def test_external_entry_has_exact_deploy_and_batch_telemetry_contract():
    entry = Path(__file__).resolve().parents[2] / "internal_training" / stage_eval.ENTRY
    if not entry.exists():
        pytest.skip("TRI internal_training sibling checkout is not present")
    source = entry.read_text(encoding="utf-8")
    assert "validate_stage_s_eval_inputs.py" in source
    assert "validate_artifact_tree.py" in source
    # The interface enum and both Q serve branches must exist in the entry: the workspace-free
    # robottt_fast (Q2) server and the combined tanh_robottt (Q3) server, the latter launched with
    # the robottt_fast arg style (--finetune-ckpt "$CKPT" --stride "$EXEC_STEPS") PLUS the tanh
    # workspace artifacts and the Stage-Q provenance config-name.
    assert "base|tanh|cfg2|robottt_fast|tanh_robottt" in source
    assert "serve_pi_05_robottt.py" in source
    assert '"$STAGE_S_INTERFACE" == "tanh_robottt"' in source
    assert "pi05_robocasa_stage_q_q3" in source
    assert source.count('--stride "$EXEC_STEPS"') >= 2
    assert "TRAIN_COMPLETION_CLAIM_S3" in source
    assert "CHECKPOINT_TREE_MANIFEST_S3" in source
    assert "canonical_terse_task_instruction" in source
    assert "--tap-prompt terse" in source
    assert '--task-prompt-manifest "$TASK_PROMPT_MANIFEST_FILE"' in source
    assert source.count('--exclude "*" --include "params/*" --include "assets/*"') >= 2
    assert "--require-realized-batching" in source
    assert "prep_heldout_root.py" in source
    assert '--episode-manifest "$EPISODE_MANIFEST"' in source
    assert not list((Path(__file__).resolve().parents[1] / "vla_training" / "eval").glob("*.rej"))
    checked = subprocess.run(
        ["bash", "-n", str(entry)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
