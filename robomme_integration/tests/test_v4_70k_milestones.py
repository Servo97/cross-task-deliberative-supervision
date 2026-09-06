"""A19 checkpoint maturity: the v4_70k training recipe and the milestone eval queues (CPU-only)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from robomme_integration import launch
from robomme_integration.eval import build_milestone_queue as builder
from robomme_integration.eval import campaign, launch_p5_campaign, parallel_campaign
from robomme_integration.gpu.checkpoint_transport import CheckpointWatcher, tree_summary
from robomme_integration.tests.test_eval_campaign import MemoryStore, _write_json
from robomme_integration.tests.test_p5_eval_campaign_launch import _parallel_preflight
from robomme_integration.training.single_task import TASK_ORDER

SOURCE = Path(launch.__file__).resolve().parent
REPO_ROOT = SOURCE.parent
MILESTONES = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000]
DEPLOY = [*MILESTONES, 69_999]


def _plan(argv: list[str]) -> dict:
    args = launch.parser().parse_args(argv)
    return launch.build_plan(args, SOURCE)


# --------------------------------------------------------------------------- training recipe


def test_v4_70k_is_an_explicit_opt_in_and_the_60k_path_is_byte_identical():
    base = ["--scope", "multitask", "--arm", "v4_s0", "--hardware", "p5", "--dry-run"]
    plan60 = _plan(base)
    plan60_explicit = _plan([*base, "--multitask-train-steps", "60000"])
    assert plan60["manifest"] == plan60_explicit["manifest"]
    assert plan60["environment"] == plan60_explicit["environment"]
    assert plan60["run_id"].startswith("mt-v4-all16-v4_s0-seed0-")
    training60 = plan60["manifest"]["scientific"]["training"]
    assert "recipe" not in training60
    assert training60["steps"] == 60_000 and training60["warmup_steps"] == 3_000
    assert training60["checkpoint_policy"] == {
        "save_interval": 5_000,
        "remote_resume": True,
        "upload_only_finalized_orbax": True,
        "training_retention": "newest_plus_step30000",
        "success_retention": [59_999],
    }
    assert plan60["environment"]["ROBOMME_CHECKPOINT_MILESTONES"] == "30000"
    assert "ROBOMME_RECIPE" not in plan60["environment"]
    assert "ROBOMME_SUCCESS_CHECKPOINT_MILESTONES" not in plan60["environment"]
    assert plan60["recipe"] is None and plan60["deploy_milestones"] == [59_999]

    plan70 = _plan([*base, "--multitask-train-steps", "70000"])
    scientific70 = plan70["manifest"]["scientific"]
    training70 = scientific70["training"]
    assert plan70["run_id"].startswith("mt-v4-70k-all16-v4_s0-seed0-")
    assert plan70["run_id"] != plan60["run_id"]
    assert plan70["manifest"]["scientific_spec_sha256"] != plan60["manifest"]["scientific_spec_sha256"]
    assert training70["recipe"] == "v4_70k"
    assert (training70["steps"], training70["warmup_steps"], training70["decay_steps"]) == (70_000, 3_500, 70_000)
    assert (training70["peak_lr"], training70["decay_lr"]) == (5e-5, 5e-6)
    assert training70["checkpoint_policy"] == {
        "save_interval": 5_000,
        "remote_resume": True,
        "upload_only_finalized_orbax": True,
        "training_retention": "local_newest;remote_newest_plus_steps_10000_20000_30000_40000_50000_60000",
        "success_retention": DEPLOY,
        "deploy_milestones": DEPLOY,
        "deploy_layout": "deploy/<step>/{params,assets}+_DEPLOY_COMPLETE.json+tree_manifest",
    }
    # The 70k spec differs from the sealed 60k spec ONLY in the training block, and there only in
    # the step count, its derived schedule, and the checkpoint contract.
    scientific60 = plan60["manifest"]["scientific"]
    assert {key for key in scientific60 if scientific60[key] != scientific70.get(key)} == {"training"}
    assert set(scientific70) == set(scientific60)
    assert {key for key in set(training60) | set(training70) if training60.get(key) != training70.get(key)} == {
        "steps",
        "warmup_steps",
        "decay_steps",
        "checkpoint_policy",
        "recipe",
    }
    env = plan70["environment"]
    assert env["WSM_MAX_STEPS"] == "70000" and env["ROBOMME_FINAL_STEP"] == "69999"
    assert env["WSM_WARMUP_STEPS"] == "3500" and env["WSM_DECAY_STEPS"] == "70000"
    assert env["WSM_SAVE_INTERVAL"] == "5000" and env["WSM_DECAY_LR"] == "5e-6"
    assert env["ROBOMME_RECIPE"] == "v4_70k"
    assert env["ROBOMME_CHECKPOINT_MILESTONES"] == "10000,20000,30000,40000,50000,60000"
    assert env["ROBOMME_SUCCESS_CHECKPOINT_MILESTONES"] == "10000,20000,30000,40000,50000,60000"
    assert plan70["manifest"]["checkpoint_tree_manifest_root"].endswith(f"/{plan70['run_id']}/milestones")
    assert plan70["manifest"]["claims"]["completion"].endswith(f"/train/{plan70['run_id']}/step-69999.complete.json")
    assert plan70["output"].endswith(f"/multitask_v4/all16/v4_s0/seed0/{plan70['run_id']}")
    assert plan70["recipe"] == "v4_70k" and plan70["deploy_milestones"] == DEPLOY
    assert plan70["manifest"]["infrastructure"]["max_run_seconds"] == 24 * 3600
    assert launch.v4_70k_milestones(70_000) == tuple(MILESTONES)
    with pytest.raises(ValueError):
        launch.v4_70k_milestones(60_000)

    # A GDN workspace arm gets exactly the same 70k contract (the M-arms are one-line clones of it).
    index_sha = "a" * 64
    gdn = _plan(
        [
            "--scope",
            "multitask",
            "--arm",
            "v4_wsm_gdn16_drop02",
            "--hardware",
            "p5",
            "--workspace-index-s3",
            f"{launch.STUDY_ROOT}/artifacts/robomme/workspace/all16/{index_sha}.json",
            "--workspace-index-sha256",
            index_sha,
            "--multitask-train-steps",
            "70000",
            "--dry-run",
        ]
    )
    assert gdn["run_id"].startswith("mt-v4-70k-all16-v4_wsm_gdn16_drop02-seed0-")
    assert gdn["manifest"]["scientific"]["training"]["checkpoint_policy"]["deploy_milestones"] == DEPLOY
    assert gdn["environment"]["ROBOMME_SUCCESS_CHECKPOINT_MILESTONES"] == env["ROBOMME_SUCCESS_CHECKPOINT_MILESTONES"]


def test_v4_70k_is_refused_outside_multitask_v4_arms():
    with pytest.raises(SystemExit, match="defined only for multitask v4 arms"):
        _plan(
            [
                "--scope",
                "single_task",
                "--task",
                "PickXtimes",
                "--arm",
                "v4_s0",
                "--multitask-train-steps",
                "70000",
                "--dry-run",
            ]
        )
    with pytest.raises(SystemExit, match="defined only for multitask v4 arms"):
        _plan(
            [
                "--scope",
                "multitask",
                "--arm",
                "s0",
                "--hardware",
                "p5",
                "--multitask-train-steps",
                "70000",
                "--dry-run",
            ]
        )
    base_sha = "b" * 64
    with pytest.raises(SystemExit, match="defined only for multitask v4 arms"):
        _plan(
            [
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
                "--multitask-train-steps",
                "70000",
                "--dry-run",
            ]
        )
    with pytest.raises(SystemExit):
        launch.parser().parse_args(["--scope", "multitask", "--arm", "v4_s0", "--multitask-train-steps", "65000"])


def test_entry_asserts_the_70k_contract_and_keeps_the_sealed_paths():
    entry = (SOURCE / "gpu_train_entry.sh").read_text(encoding="utf-8")
    assert 'elif [[ "${ROBOMME_RECIPE:-}" == v4_70k ]]; then' in entry
    assert '[[ "$WSM_MAX_STEPS" == 70000 && "$ROBOMME_FINAL_STEP" == 69999 ]]' in entry
    assert '"${ROBOMME_SUCCESS_CHECKPOINT_MILESTONES:-}" == 10000,20000,30000,40000,50000,60000' in entry
    assert '[[ "$ROBOMME_RUN_ID" == mt-v4-70k-all16-* ]]' in entry
    # sealed recipes refuse maturity metadata; the 60k assertion is unchanged
    assert '[[ -z "${ROBOMME_RECIPE:-}" && -z "${ROBOMME_SUCCESS_CHECKPOINT_MILESTONES:-}" ]]' in entry
    assert '[[ "$WSM_MAX_STEPS" == 60000 && "$ROBOMME_FINAL_STEP" == 59999 ]]' in entry
    assert '[[ "$WSM_MAX_STEPS" == 20000 && "$ROBOMME_FINAL_STEP" == 19999 ]]' in entry
    assert '[[ "$WSM_MAX_STEPS" == 80000 && "$ROBOMME_FINAL_STEP" == 79999 ]]' in entry
    assert entry.count("deploy_recipe_step() {") == 1
    assert "for step in 60000 70000 79999; do" in entry
    assert 'for step in "${MILESTONE_STEPS[@]}"; do' in entry
    assert '"kind": "robomme_gpu_milestone_checkpoint_set_complete"' in entry
    # the prune of steps/ follows the completion claim in both milestone recipes and the normal path
    assert entry.count('aws s3 rm "${OUTPUT_S3%/}/steps" --recursive --only-show-errors') == 3
    branch = entry[entry.index('\nif [[ "${ROBOMME_RECIPE:-}" == v4_70k ]]; then') :]
    branch = branch[: branch.index("\nfi\n")]
    assert branch.index('publish_attempt_receipt_once "$COMPLETE" "$COMPLETION_CLAIM_S3"') < branch.index(
        'aws s3 rm "${OUTPUT_S3%/}/steps"'
    )
    assert subprocess.run(["bash", "-n", str(SOURCE / "gpu_train_entry.sh")], check=False).returncode == 0


def test_v4_70k_retention_keeps_every_10k_milestone_during_training_and_after_success(tmp_path):
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

    def generation(root: Path, step: int) -> None:
        (root / str(step)).mkdir(parents=True)
        (root / str(step) / "_CHECKPOINT_METADATA").write_text(f'{{"step":{step}}}\n')

    local = tmp_path / "checkpoints"
    transport = MemoryTransport()
    watcher = CheckpointWatcher(
        local,
        transport,
        tmp_path / "state.json",
        milestones=set(MILESTONES),
        final_step=69_999,  # type: ignore[arg-type]
    )
    schedule = [*range(5_000, 70_000, 5_000), 69_999]
    for step in schedule:
        generation(local, step)
        watcher.sync_once()
        expected = {step} | {item for item in MILESTONES if item <= step}
        assert set(transport.steps) == expected, step
        assert transport.latest == step
    assert set(transport.steps) == set(DEPLOY)  # 65000 was pruned once 69999 became the newest
    watcher.finalize_success(success_milestones=set(MILESTONES))
    assert set(transport.steps) == set(DEPLOY)
    assert transport.latest == 69_999
    assert json.loads((tmp_path / "state.json").read_text())["uploaded_steps"] == DEPLOY

    incomplete = MemoryTransport()
    incomplete.steps = {step: transport.steps[step] for step in DEPLOY if step != 40_000}
    missing = CheckpointWatcher(
        local,
        incomplete,
        tmp_path / "missing.json",
        milestones=set(MILESTONES),
        final_step=69_999,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match=r"milestones were not uploaded: \[40000\]"):
        missing.finalize_success(success_milestones=set(MILESTONES))
    assert set(incomplete.steps) == set(DEPLOY) - {40_000}  # nothing pruned on a failed finalize


_STUB_AWS = r"""#!/usr/bin/env bash
# Local stand-in for the AWS CLI surface the entry uses: s3 cp/sync/rm and s3api put-object,
# addressing s3://bucket/key as $FAKE_S3/bucket/key.
set -euo pipefail
: "${FAKE_S3:?}"
map() { local u="$1"; if [[ "$u" == s3://* ]]; then echo "$FAKE_S3/${u#s3://}"; else echo "$u"; fi; }
svc="$1"; shift
case "$svc" in
  s3)
    op="$1"; shift
    args=(); recursive=0; delete=0
    while (($#)); do
      case "$1" in
        --only-show-errors|--no-follow-symlinks) ;;
        --recursive) recursive=1 ;;
        --delete) delete=1 ;;
        --region|--exclude|--include) shift ;;
        *) args+=("$1") ;;
      esac
      shift
    done
    case "$op" in
      cp)
        src="$(map "${args[0]}")"; dst="$(map "${args[1]}")"
        [[ -f "$src" ]] || { echo "fatal error: An error occurred (404) when calling the HeadObject operation: Not Found" >&2; exit 1; }
        if [[ "$dst" == "-" ]]; then cat "$src"; else mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; fi ;;
      sync)
        src="$(map "${args[0]}")"; dst="$(map "${args[1]}")"
        [[ -d "$src" ]] || { echo "sync source missing: $src" >&2; exit 1; }
        if (( delete )); then rm -rf "$dst"; fi
        mkdir -p "$dst"; cp -a "$src/." "$dst/" ;;
      rm)
        target="$(map "${args[0]}")"
        if (( recursive )); then rm -rf "$target"; else rm -f "$target"; fi ;;
      *) echo "unsupported s3 op $op" >&2; exit 2 ;;
    esac ;;
  s3api)
    op="$1"; shift
    [[ "$op" == put-object ]] || { echo "unsupported s3api $op" >&2; exit 2; }
    bucket=""; key=""; body=""; once=0
    while (($#)); do
      case "$1" in
        --bucket) bucket="$2"; shift ;;
        --key) key="$2"; shift ;;
        --body) body="$2"; shift ;;
        --if-none-match) once=1; shift ;;
        --region) shift ;;
      esac
      shift
    done
    dst="$FAKE_S3/$bucket/$key"
    if (( once )) && [[ -e "$dst" ]]; then echo "An error occurred (PreconditionFailed)" >&2; exit 1; fi
    mkdir -p "$(dirname "$dst")"; cp "$body" "$dst"; echo "{}" ;;
  *) echo "unsupported service $svc" >&2; exit 2 ;;
esac
"""


def _entry_fragments() -> str:
    entry = (SOURCE / "gpu_train_entry.sh").read_text(encoding="utf-8")

    def between(start: str, end: str) -> str:
        first = entry.index(start)
        return entry[first : entry.index(end, first)]

    helpers = between("publish_once() {", "download_hashed() {")
    exporter = between(
        'RECEIPTS="$WORK/scientific-checkpoints.jsonl"',
        'if [[ "$ROBOMME_ARM" == official_recipe_lerobot ]]; then\n  for step in 60000',
    )
    branch_start = entry.index('\nif [[ "${ROBOMME_RECIPE:-}" == v4_70k ]]; then')
    branch = entry[branch_start : entry.index("\nfi\n", branch_start) + len("\nfi\n")]
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + helpers + "\n" + exporter + "\n" + branch


def _milestone_fixture(tmp_path: Path, *, drop_marker_for: int | None = None) -> tuple[dict, Path]:
    run_id = f"mt-v4-70k-all16-v4_s0-seed0-{'a' * 16}"
    output_s3 = "s3://bucket/study/checkpoints/robomme/pi05/multitask_v4/all16/v4_s0/seed0/" + run_id
    fake = tmp_path / "fake_s3"
    remote_run = fake / output_s3[len("s3://") :]
    work = tmp_path / "work"
    work.mkdir()
    local_experiment = work / "checkpoints" / "pi05_robomme_v4_s0" / run_id
    steps = (10_000, 20_000, 29_999)
    payloads = {}
    for step in steps:
        generation = tmp_path / "generations" / str(step)
        (generation / "params").mkdir(parents=True)
        (generation / "assets").mkdir()
        (generation / "train_state").mkdir()
        payloads[step] = bytes([step % 251]) * 33
        (generation / "params" / "weights").write_bytes(payloads[step])
        (generation / "assets" / "norm.json").write_text('{"step": %d}\n' % step)
        (generation / "train_state" / "state").write_bytes(b"optimizer" * (step % 7 or 1))
        (generation / "_CHECKPOINT_METADATA").write_text('{"step": %d}\n' % step)
        remote = remote_run / "steps" / str(step)
        shutil.copytree(generation, remote)
        if drop_marker_for != step:
            (remote / "_UPLOAD_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "step": step,
                        "source_marker": "_CHECKPOINT_METADATA",
                        "tree": tree_summary(generation),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    # Orbax keeps only the newest local generation.
    shutil.copytree(tmp_path / "generations" / "29999", local_experiment / "29999")
    (remote_run / "LATEST.json").write_text('{"schema_version":1,"step":29999}\n')
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "aws").write_text(_STUB_AWS, encoding="utf-8")
    (stub_dir / "aws").chmod(0o755)
    script = tmp_path / "entry-fragment.sh"
    script.write_text(_entry_fragments(), encoding="utf-8")
    env = {
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "FAKE_S3": str(fake),
        "WORK": str(work),
        "CODE_DIR": str(SOURCE),
        "OUTPUT_S3": output_s3,
        "ROBOMME_ARM": "v4_s0",
        "ROBOMME_RUN_ID": run_id,
        "ROBOMME_ATTEMPT_ID": f"{run_id}-attempt1",
        "ROBOMME_SCIENTIFIC_SPEC_SHA256": "c" * 64,
        "RUN_MANIFEST_SHA256": "d" * 64,
        "CHECKPOINT_TREE_MANIFEST_ROOT": f"s3://bucket/study/manifests/artifacts/checkpoints/{run_id}/milestones",
        "COMPLETION_CLAIM_S3": f"s3://bucket/study/manifests/claims/train/{run_id}/step-29999.complete.json",
        "ROBOMME_RECIPE": "v4_70k",
        "ROBOMME_SUCCESS_CHECKPOINT_MILESTONES": "10000,20000",
        "ROBOMME_FINAL_STEP": "29999",
        "WSM_EXP_NAME": run_id,
        "WSM_CKPT_BASE": str(work / "checkpoints"),
        "HOME": str(tmp_path),
    }
    return {
        "env": env,
        "fake": fake,
        "remote_run": remote_run,
        "run_id": run_id,
        "output_s3": output_s3,
        "payloads": payloads,
        "work": work,
    }, script


def test_entry_deploys_every_milestone_then_seals_then_prunes(tmp_path):
    fixture, script = _milestone_fixture(tmp_path)
    result = subprocess.run(["bash", str(script)], env=fixture["env"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"ROBOMME GPU TRAINING COMPLETE run_id={fixture['run_id']} recipe=v4_70k steps=10000,20000,29999"
        in result.stdout
    )
    remote_run = fixture["remote_run"]
    tree_root = fixture["fake"] / "bucket/study/manifests/artifacts/checkpoints" / fixture["run_id"] / "milestones"
    receipts = {}
    for step in (10_000, 20_000, 29_999):
        deploy = remote_run / "deploy" / str(step)
        assert (deploy / "params" / "weights").read_bytes() == fixture["payloads"][step]
        assert (deploy / "assets" / "norm.json").is_file()
        assert not (deploy / "train_state").exists()  # deploy-only: no optimizer state
        receipt = json.loads((deploy / "_DEPLOY_COMPLETE.json").read_text())
        assert receipt["kind"] == "robomme_gpu_deploy_checkpoint_complete"
        assert receipt["step"] == step and receipt["recipe"] == "v4_70k" and "diagnostic_label" not in receipt
        assert receipt["checkpoint_uri"] == f"{fixture['output_s3']}/deploy/{step}"
        assert receipt["run_id"] == fixture["run_id"] and receipt["scientific_spec_sha256"] == "c" * 64
        manifests = list((tree_root / f"step-{step}").glob("*.json"))
        assert len(manifests) == 1
        tree_sha = hashlib.sha256(manifests[0].read_bytes()).hexdigest()
        assert manifests[0].stem == tree_sha == receipt["tree_manifest_sha256"]
        assert (
            receipt["tree_manifest_uri"]
            == f"{fixture['env']['CHECKPOINT_TREE_MANIFEST_ROOT']}/step-{step}/{tree_sha}.json"
        )
        tree = json.loads(manifests[0].read_text())
        assert tree["checkpoint_uri"] == receipt["checkpoint_uri"]
        assert {record["key"] for record in tree["objects"]} == {"params/weights", "assets/norm.json"}
        receipts[step] = receipt
    completion = json.loads((fixture["fake"] / fixture["env"]["COMPLETION_CLAIM_S3"][len("s3://") :]).read_text())
    assert completion["kind"] == campaign.MILESTONE_COMPLETION_KIND
    assert completion["steps"] == [10_000, 20_000, 29_999] and completion["final_step"] == 29_999
    assert completion["recipe"] == "v4_70k" and completion["run_id"] == fixture["run_id"]
    assert [record["step"] for record in completion["checkpoints"]] == completion["steps"]
    for record in completion["checkpoints"]:
        assert record["tree_manifest_sha256"] == receipts[record["step"]]["tree_manifest_sha256"]
        assert record["checkpoint_uri"] == f"{fixture['output_s3']}/deploy/{record['step']}"
    # recovery generations pruned only after everything above was sealed
    assert not (remote_run / "steps").exists() and not (remote_run / "LATEST.json").exists()
    assert not list(fixture["work"].glob("scientific-step-*"))


def test_entry_never_prunes_when_a_milestone_deploy_fails(tmp_path):
    fixture, script = _milestone_fixture(tmp_path, drop_marker_for=20_000)
    result = subprocess.run(["bash", str(script)], env=fixture["env"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    remote_run = fixture["remote_run"]
    assert (remote_run / "deploy" / "10000" / "_DEPLOY_COMPLETE.json").is_file()  # first milestone exported
    assert not (remote_run / "deploy" / "20000").exists()
    assert not (remote_run / "deploy" / "29999").exists()
    assert (remote_run / "steps" / "20000" / "_CHECKPOINT_METADATA").is_file()  # nothing pruned
    assert (remote_run / "steps" / "29999").is_dir() and (remote_run / "LATEST.json").is_file()
    assert not (fixture["fake"] / fixture["env"]["COMPLETION_CLAIM_S3"][len("s3://") :]).exists()


# --------------------------------------------------------------------------- eval milestone cells


def _sweep() -> dict:
    return builder.load_sweep(builder.DEFAULT_SWEEP)


def _synthetic_run(label: str = "M0-70k") -> tuple[dict, dict, dict[int, bytes]]:
    """A REAL 70k launcher manifest plus the completion claim the entry would seal for it."""
    arm = _sweep()["arms"][label]["arm"]
    plan = _plan(
        ["--scope", "multitask", "--arm", arm, "--hardware", "p5", "--multitask-train-steps", "70000", "--dry-run"]
    )
    manifest = plan["manifest"]
    tree_root = manifest["checkpoint_tree_manifest_root"]
    trees: dict[int, bytes] = {}
    records = []
    for step in DEPLOY:
        checkpoint_uri = f"{manifest['output_s3']}/deploy/{step}"
        payload = bytes([step % 251]) * 3
        tree = {
            "schema_version": 1,
            "checkpoint_uri": checkpoint_uri,
            "objects": [
                {"key": "params/mock", "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            ],
        }
        tree_bytes = campaign._canonical(tree)
        trees[step] = tree_bytes
        tree_sha = hashlib.sha256(tree_bytes).hexdigest()
        records.append(
            {
                "step": step,
                "checkpoint_uri": checkpoint_uri,
                "tree_manifest_uri": f"{tree_root}/step-{step}/{tree_sha}.json",
                "tree_manifest_sha256": tree_sha,
            }
        )
    claim = {
        "schema_version": 1,
        "kind": campaign.MILESTONE_COMPLETION_KIND,
        "recipe": "v4_70k",
        "run_id": manifest["run_id"],
        "attempt_id": manifest["attempt_id"],
        "scientific_spec_sha256": manifest["scientific_spec_sha256"],
        "final_step": 69_999,
        "steps": list(DEPLOY),
        "checkpoints": records,
        "run_manifest_sha256": manifest["manifest_sha256"],
    }
    return manifest, claim, trees


def _seal_with_gates(queue: dict, gates: dict) -> dict:
    return campaign.seal_document({**queue, "gates": gates}, field="queue_manifest_sha256")


def _gates_for(tmp_path: Path, openpi: dict) -> tuple[Path, Path, dict]:
    """Preflight/receipt gate pair bound to the queue's actual serving archive."""
    runtime_artifact = {"uri": "s3://bucket/runtime.tgz", "sha256": "1" * 64}
    preflight = {
        "schema_version": 1,
        "kind": campaign.PREFLIGHT_KIND,
        "preflight_id": "p5-native-eval-v1-milestones",
        "runtime": runtime_artifact,
        "openpi": openpi,
        "vla_eval_entrypoint": {"kind": "python_module_wrapper", "module": "vla_eval.cli.main"},
        "image": {"uri": "image", "sha256": "3" * 64},
        "probe": {
            "benchmark_adapter": "robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark",
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
        },
        "source_tree_sha256": "4" * 64,
        "claim_s3": "s3://bucket/preflight.json",
        "infrastructure": {"instance_type": "ml.p5.48xlarge", "accelerator": "8xH100"},
    }
    preflight = campaign.seal_document(preflight, field="manifest_sha256")
    preflight["status"] = "native_render_reset_passed"
    preflight_path = tmp_path / "preflight.json"
    preflight_sha = _write_json(preflight_path, preflight)
    runtime_root = tmp_path / "runtime"
    paths = {
        "policy_python": runtime_root / "openpi/.venv/bin/python",
        "vla_eval": runtime_root / "eval/bin/vla-eval",
        "harness_src": runtime_root / "harness/src",
        "robomme_src": runtime_root / "robomme/src",
        "maniskill_src": runtime_root / "maniskill",
        "openpi_src": runtime_root / "openpi/src",
        "policy_site": runtime_root / "openpi/site-packages",
        "simulator_site": runtime_root / "eval/site-packages",
        "upstream_root": runtime_root / "upstream",
        "vision_encoder_home": runtime_root / "vision",
    }
    for name, path in paths.items():
        if name in {"policy_python", "vla_eval"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)
        else:
            path.mkdir(parents=True, exist_ok=True)
    vision = paths["vision_encoder_home"] / "pi05_vision_encoder/siglip_params.pkl"
    vision.parent.mkdir(parents=True)
    vision.write_bytes(b"pinned")
    receipt = campaign.seal_document(
        {
            "schema_version": 1,
            "kind": campaign.RUNTIME_KIND,
            "status": "staged_and_verified",
            "preflight_claim_sha256": preflight_sha,
            "runtime": runtime_artifact,
            "openpi": openpi,
            "vla_eval_wrapper": {
                "kind": "python_module_wrapper",
                "module": "vla_eval.cli.main",
                "sha256": hashlib.sha256(paths["vla_eval"].read_bytes()).hexdigest(),
            },
            "paths": {name: str(path) for name, path in paths.items()},
            "render_environment": {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "ROBOMME_USE_LAVAPIPE": "0"},
        },
        field="receipt_sha256",
    )
    receipt_path = tmp_path / "runtime-receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    gates = {
        "native_preflight": {
            "preflight_id": preflight["preflight_id"],
            "claim_sha256": preflight_sha,
            "source_tree_sha256": preflight["source_tree_sha256"],
        },
        "runtime_receipt": {
            "receipt_sha256": receipt_sha,
            "runtime_artifact_sha256": runtime_artifact["sha256"],
            "openpi_sha256": openpi["sha256"],
        },
    }
    return preflight_path, receipt_path, gates


def test_a19_sweep_templates_fail_closed_until_filled():
    sweep = _sweep()
    assert set(sweep["arms"]) == {"M0-70k", "M1", "M2", "M3", "M3-ctrl"}
    assert sweep["arms"]["M0-70k"]["milestones"] == DEPLOY and sweep["arms"]["M3"]["milestones"] == DEPLOY
    for label in ("M1", "M2", "M3-ctrl"):
        assert sweep["arms"][label]["milestones"] == [30_000, 40_000, 50_000, 60_000, 69_999]
    for label, record in sweep["arms"].items():
        template = builder.build_template(sweep, label, source_root=REPO_ROOT)
        expected_cells = 16 * len(record["milestones"])
        assert len(template["cells"]) == expected_cells
        assert [cell["ordinal"] for cell in template["cells"]] == list(range(expected_cells))
        # milestone-descending, then canonical task order: the final step lands first
        assert [cell["checkpoint_step"] for cell in template["cells"][:16]] == [69_999] * 16
        assert [cell["task"] for cell in template["cells"][:16]] == list(TASK_ORDER)
        assert template["comparability"]["eval_protocol"] == campaign.MILESTONE_EVAL_PROTOCOL
        assert template["comparability"]["eval_protocol"]["execute_steps"] == 10
        assert template["comparability"]["eval_protocol"]["paper_protocol_matched"] is False
        assert set(template["comparability"]["task_benchmark_configs"]) == set(TASK_ORDER)
        placeholders = builder.unresolved_placeholders(template)
        assert {
            "<RUN_ID>",
            "<ATTEMPT_ID>",
            "<SCIENTIFIC_SPEC_SHA256>",
            "<RUN_MANIFEST_SHA256>",
            "<OPENPI_URI>",
            "<OPENPI_SHA256>",
        } <= placeholders
        if label == "M0-70k":
            assert "<WORKSPACE_SERVING_UNRESOLVED>" not in placeholders
        else:
            assert "<WORKSPACE_SERVING_UNRESOLVED>" in placeholders
        dummy_gates = {
            "native_preflight": {"preflight_id": "x", "claim_sha256": "0" * 64, "source_tree_sha256": "0" * 64},
            "runtime_receipt": {
                "receipt_sha256": "0" * 64,
                "runtime_artifact_sha256": "0" * 64,
                "openpi_sha256": "0" * 64,
            },
        }
        with pytest.raises(ValueError):
            campaign.validate_queue(_seal_with_gates(template, dummy_gates))
    committed = REPO_ROOT / "robomme_integration/eval/milestone_queues"
    for label, record in sweep["arms"].items():
        on_disk = json.loads((committed / f"{record['queue_id']}.template.json").read_text(encoding="utf-8"))
        assert on_disk == builder.build_template(sweep, label, source_root=REPO_ROOT)


def test_filled_milestone_queue_validates_stages_and_dry_runs_on_the_p5_topology(tmp_path):
    manifest, claim, trees = _synthetic_run("M0-70k")
    resolved = builder.verify_run(manifest, claim, arm="v4_s0", manifest_uri=manifest["manifest_s3"])
    assert resolved["deployed_milestones"] == DEPLOY
    assert resolved["training_completion_binding"] == campaign.TRAINING_COMPLETION_CURRENT
    template = builder.build_template(_sweep(), "M0-70k", source_root=REPO_ROOT)
    queue = builder.fill_template(template, resolved)
    assert not builder.unresolved_placeholders(queue)
    assert len(queue["cells"]) == 112
    assert {cell["run_id"] for cell in queue["cells"]} == {manifest["run_id"]}
    assert queue["comparability"]["serving_openpi"] == {"uri": launch.OPENPI, "sha256": launch.OPENPI_SHA}
    preflight, receipt, gates = _gates_for(tmp_path, queue["comparability"]["serving_openpi"])
    sealed = _seal_with_gates(queue, gates)
    campaign.validate_queue(sealed, source_root=REPO_ROOT)

    # every milestone cell resolves deploy/<milestone>, never deploy/<final> for an earlier step
    store = MemoryStore()
    store.values[manifest["manifest_s3"]] = campaign._canonical(manifest)
    store.values[manifest["claims"]["completion"]] = campaign._canonical(claim)
    for record in claim["checkpoints"]:
        store.values[record["tree_manifest_uri"]] = trees[record["step"]]
        deploy = {
            "schema_version": 1,
            "kind": "robomme_gpu_deploy_checkpoint_complete",
            "run_id": manifest["run_id"],
            "attempt_id": manifest["attempt_id"],
            "scientific_spec_sha256": manifest["scientific_spec_sha256"],
            "step": record["step"],
            "checkpoint_uri": record["checkpoint_uri"],
            "tree_manifest_uri": record["tree_manifest_uri"],
            "tree_manifest_sha256": record["tree_manifest_sha256"],
            "run_manifest_sha256": manifest["manifest_sha256"],
            "recipe": "v4_70k",
        }
        store.values[f"{record['checkpoint_uri']}/_DEPLOY_COMPLETE.json"] = campaign._canonical(deploy)

    class LocalStager(campaign.AwsStager):
        synced: list[str] = []

        def _sync(self, uri: str, destination: Path, *, checkpoint: bool = False) -> None:
            assert checkpoint
            self.synced.append(uri)
            step = int(uri.rsplit("/", 1)[1])
            (destination / "params").mkdir(parents=True)
            (destination / "assets").mkdir()
            (destination / "params/mock").write_bytes(bytes([step % 251]) * 3)

    stager = LocalStager(store)
    for step in (69_999, 30_000, 10_000):
        cell = next(
            item for item in sealed["cells"] if item["checkpoint_step"] == step and item["task"] == "PickXtimes"
        )
        assert campaign.checkpoint_step(cell) == step
        destination = tmp_path / "stage" / str(step) / "checkpoint" / str(step)
        assert stager.stage_checkpoint(cell, destination) == f"{manifest['output_s3']}/deploy/{step}"
    assert stager.synced == [f"{manifest['output_s3']}/deploy/{step}" for step in (69_999, 30_000, 10_000)]

    # fail closed: a step the run never deployed, and a claim that stops enumerating a milestone
    rogue = dict(sealed["cells"][0], checkpoint_step=35_000)
    with pytest.raises(ValueError, match="never deployed"):
        stager.stage_checkpoint(rogue, tmp_path / "rogue")
    truncated = dict(claim, steps=[step for step in DEPLOY if step != 40_000])
    truncated["checkpoints"] = [record for record in claim["checkpoints"] if record["step"] != 40_000]
    store.values[manifest["claims"]["completion"]] = campaign._canonical(truncated)
    cell_40k = next(item for item in sealed["cells"] if item["checkpoint_step"] == 40_000)
    with pytest.raises(ValueError, match="identity drift"):
        stager.stage_checkpoint(cell_40k, tmp_path / "truncated")
    store.values[manifest["claims"]["completion"]] = campaign._canonical(claim)

    # the parallel dry run addresses each lane at the cell's own milestone directory
    runtime = campaign.verify_gates(
        sealed, preflight_claim=preflight, runtime_receipt=receipt, require_runtime_paths=False
    )
    parallel_queue = dict(sealed)
    parallel_queue.pop("queue_manifest_sha256")
    parallel_queue["topology"] = parallel_campaign.p5_8xh100_topology().as_queue_topology()
    parallel_queue = campaign.seal_document(parallel_queue, field="queue_manifest_sha256")
    campaign.validate_queue(parallel_queue, source_root=REPO_ROOT)
    payload = parallel_campaign.dry_run_payload(parallel_queue, REPO_ROOT, runtime, tmp_path / "work")
    assert len(payload["cells"]) == 112
    assert {cell["lane_id"] for cell in payload["cells"]} == {f"p5-h100-gpu{gpu}" for gpu in range(8)}
    first = payload["cells"][0]
    command = first["launch_command"]
    assert first["task"] == "PatternLock" and first["arm"] == "v4_s0"
    assert command[command.index("--checkpoint") + 1].endswith("/checkpoint/69999")
    assert command[command.index("--task") + 1] == "PatternLock"
    assert command[command.index("--benchmark-config") + 1].endswith("/eval/configs/patternlock.yaml")
    later = next(cell for cell in payload["cells"] if cell["cell_id"].endswith("-step10000"))
    assert later["launch_command"][later["launch_command"].index("--checkpoint") + 1].endswith("/checkpoint/10000")


def test_milestone_queue_validation_rejects_off_set_steps_and_protocol_drift(tmp_path):
    manifest, claim, _trees = _synthetic_run("M0-70k")
    resolved = builder.verify_run(manifest, claim, arm="v4_s0", manifest_uri=manifest["manifest_s3"])
    queue = builder.fill_template(builder.build_template(_sweep(), "M0-70k", source_root=REPO_ROOT), resolved)
    _preflight, _receipt, gates = _gates_for(tmp_path, queue["comparability"]["serving_openpi"])
    campaign.validate_queue(_seal_with_gates(queue, gates), source_root=REPO_ROOT)

    def mutated(**changes):
        value = json.loads(json.dumps(queue))
        for path, replacement in changes.items():
            node = value
            keys = path.split(".")
            for key in keys[:-1]:
                node = node[int(key)] if key.isdigit() else node[key]
            node[keys[-1]] = replacement
        return _seal_with_gates(value, gates)

    with pytest.raises(ValueError, match="not in the run's deployed milestone set"):
        campaign.validate_queue(mutated(**{"cells.0.checkpoint_step": 35_000}), source_root=REPO_ROOT)
    with pytest.raises(ValueError, match="deployed milestone set ending at its final step"):
        campaign.validate_queue(mutated(**{"cells.0.deployed_milestones": MILESTONES}), source_root=REPO_ROOT)
    with pytest.raises(ValueError, match="execute-10 fixed-50 evaluation protocol"):
        campaign.validate_queue(
            mutated(**{"comparability.eval_protocol": {**campaign.MILESTONE_EVAL_PROTOCOL, "execute_steps": 16}}),
            source_root=REPO_ROOT,
        )
    with pytest.raises(ValueError, match="eval_id is not queue/run exact"):
        campaign.validate_queue(mutated(**{"cells.5.eval_id": queue["cells"][4]["eval_id"]}), source_root=REPO_ROOT)
    with pytest.raises(ValueError, match="invalid multitask v4_70k run_id"):
        campaign.validate_queue(
            mutated(**{"cells.0.run_id": queue["cells"][0]["run_id"].replace("mt-v4-70k-", "mt-v4-")}),
            source_root=REPO_ROOT,
        )
    duplicate = json.loads(json.dumps(queue))
    duplicate["cells"][1] = dict(duplicate["cells"][0], ordinal=1, cell_id=duplicate["cells"][1]["cell_id"])
    with pytest.raises(ValueError, match="duplicate task/arm/eval/run/step identity"):
        campaign.validate_queue(_seal_with_gates(duplicate, gates), source_root=REPO_ROOT)
    # a single-task queue can never smuggle milestone fields
    from robomme_integration.tests.test_eval_campaign import _gates, _queue

    _p, _r, single_gates = _gates(tmp_path / "single")
    single = _queue(tmp_path / "single-source", single_gates, ("s0",))
    single.pop("queue_manifest_sha256")
    single["cells"][0]["checkpoint_step"] = 19_999
    with pytest.raises(ValueError, match="may not carry milestone fields"):
        campaign.validate_queue(campaign.seal_document(single, field="queue_manifest_sha256"))


def test_builder_fill_verifies_the_receipt_chain_and_refuses_unresolved_workspace(tmp_path):
    manifest, claim, _trees = _synthetic_run("M0-70k")
    with pytest.raises(SystemExit, match="not a v4_70k run of v4_wsm_gdn16_drop02"):
        builder.verify_run(manifest, claim, arm="v4_wsm_gdn16_drop02", manifest_uri="")
    with pytest.raises(SystemExit, match="does not enumerate the deploy milestone set"):
        builder.verify_run(manifest, {**claim, "checkpoints": claim["checkpoints"][:-1]}, arm="v4_s0", manifest_uri="")
    with pytest.raises(ValueError, match="identity drift"):
        builder.verify_run(manifest, {**claim, "steps": claim["steps"][:-1]}, arm="v4_s0", manifest_uri="")
    tampered = json.loads(json.dumps(manifest))
    tampered["scientific"]["training"]["steps"] = 60_000
    with pytest.raises(SystemExit, match="self-seal mismatch"):
        builder.verify_run(tampered, claim, arm="v4_s0", manifest_uri="")
    # the sealed 60k recipe has no milestone set and is refused as a source for milestone cells
    plan60 = _plan(["--scope", "multitask", "--arm", "v4_s0", "--hardware", "p5", "--dry-run"])
    with pytest.raises(SystemExit, match="not a v4_70k run"):
        builder.verify_run(plan60["manifest"], claim, arm="v4_s0", manifest_uri="")

    # end to end through the CLI with local receipt files
    manifest_path = tmp_path / "manifest.json"
    claim_path = tmp_path / "claim.json"
    manifest_path.write_bytes(campaign._canonical(manifest))
    claim_path.write_bytes(campaign._canonical(claim))
    template_path = tmp_path / "template.json"
    filled_path = tmp_path / "filled.json"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    module = ["python3", "-m", "robomme_integration.eval.build_milestone_queue", "--source-root", str(REPO_ROOT)]
    run = subprocess.run(
        [*module, "template", "--label", "M0-70k", "--output", str(template_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "cells=112" in run.stdout
    run = subprocess.run(
        [
            *module,
            "fill",
            "--template",
            str(template_path),
            "--run-manifest",
            str(manifest_path),
            "--completion-claim",
            str(claim_path),
            "--output",
            str(filled_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"FILLED run_id={manifest['run_id']} cells=112" in run.stdout
    filled = json.loads(filled_path.read_text(encoding="utf-8"))
    assert not builder.unresolved_placeholders(filled)
    # an S3 reference without --confirm-read-s3 is refused before any cloud read
    run = subprocess.run(
        [
            *module,
            "fill",
            "--template",
            str(template_path),
            "--run-manifest",
            "s3://bucket/x.json",
            "--completion-claim",
            str(claim_path),
            "--output",
            str(tmp_path / "never.json"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0 and "--confirm-read-s3" in run.stderr
    assert not (tmp_path / "never.json").exists()

    # a workspace arm cannot be filled without an omega serving descriptor
    sweep = _sweep()
    m3_arm = sweep["arms"]["M3"]["arm"]
    m3_template = builder.build_template(sweep, "M3", source_root=REPO_ROOT)
    resolved_v4s0 = builder.verify_run(manifest, claim, arm="v4_s0", manifest_uri="")
    with pytest.raises(SystemExit, match="does not belong to the template's arm"):
        builder.fill_template(m3_template, resolved_v4s0)
    m3_run_id = resolved_v4s0["run_id"].replace("v4_s0", m3_arm)
    m3_resolved = {
        **resolved_v4s0,
        "run_id": m3_run_id,
        "training_output_s3": f"{builder.CHECKPOINT_ROOT}/{m3_arm}/seed0/{m3_run_id}",
    }
    with pytest.raises(SystemExit, match="omega serving inputs are not resolvable"):
        builder.fill_template(m3_template, m3_resolved)


def test_p5_launcher_dry_runs_a_filled_milestone_queue_on_the_parallel_topology(tmp_path):
    manifest, claim, _trees = _synthetic_run("M0-70k")
    resolved = builder.verify_run(manifest, claim, arm="v4_s0", manifest_uri=manifest["manifest_s3"])
    queue = builder.fill_template(builder.build_template(_sweep(), "M0-70k", source_root=REPO_ROOT), resolved)
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preflight = _parallel_preflight(SOURCE, tmp_path / "parallel-preflight.json")
    args = launch_p5_campaign.parser().parse_args(
        ["--queue-template", str(queue_path), "--native-preflight-claim", str(preflight), "--parallel-fixed50"]
    )
    plan = launch_p5_campaign.build_plan(args, SOURCE)
    sealed = plan["queue"]
    assert sealed["kind"] == campaign.MILESTONE_QUEUE_KIND
    assert sealed["queue_id"] == "a19-m0-70k-milestones-fixed50-p5-parallel-v1"
    assert sealed["topology"]["execution_mode"] == parallel_campaign.PARALLEL_EXECUTION_MODE
    assert len(sealed["topology"]["lanes"]) == 8
    assert plan["environment"]["ROBOMME_EVAL_OPENPI_PROFILE"] == "standard"
    assert plan["launch"]["infrastructure"]["priority"] == 100
    assert plan["launch"]["infrastructure"]["max_run_seconds"] == 86_400
    assert sealed["limits"] == {
        "max_run_seconds": 75_600,
        "runtime_reserve_seconds": 1_800,
        "estimated_cell_seconds": 7_200,
        "minimum_free_bytes": 128 * 1024**3,
    }
    assert len(sealed["cells"]) == 112
