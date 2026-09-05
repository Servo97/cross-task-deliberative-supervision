from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from robomme_integration.amkv import launch_e0, stage_e0

REAL_PACKAGE = Path(launch_e0.PACKAGE_DIR)
REAL_ENTRY = REAL_PACKAGE / "amkv" / launch_e0.ENTRY
CHECKPOINT_SHA = "a" * 64
SOURCE_SHA = "b" * 64
FIXTURES_SHA = "c" * 64
REQUIRED_ENVIRONMENT = (
    "AMKV_POLICY_SOURCE_S3",
    "AMKV_POLICY_SOURCE_SHA256",
    "AMKV_POLICY_SOURCE_RECEIPT_S3",
    "AMKV_POLICY_SOURCE_RECEIPT_SHA256",
    "AMKV_CHECKPOINT_S3",
    "AMKV_CHECKPOINT_INVENTORY_S3",
    "AMKV_CHECKPOINT_INVENTORY_SHA256",
    "AMKV_FIXTURES_S3",
    "AMKV_FIXTURES_MANIFEST_SHA256",
    "AMKV_OUTPUT_S3",
    "AMKV_RATIOS",
    "AMKV_RUN_ID",
    "RUN_MANIFEST_SOURCE",
    "RUN_MANIFEST_SHA256",
    "SM_USE_RESERVED_CAPACITY",
    "AMKV_CODE_SOURCE_TREE_SHA256",
    "AMKV_POLICY_GIT_SHA",
    "AMKV_POLICY_TREE_SHA1",
)


def fake_package(root: Path) -> Path:
    """A minimal package-rooted source tree, so plan tests never copy the real 7 MB package."""
    package = root / "robomme_integration"
    (package / "amkv").mkdir(parents=True)
    (package / "__init__.py").write_text("# sealed\n", encoding="utf-8")
    (package / "amkv" / "__init__.py").write_text("# sealed\n", encoding="utf-8")
    shutil.copy2(REAL_PACKAGE / "amkv" / "stage_e0.py", package / "amkv" / "stage_e0.py")
    entry = package / "amkv" / launch_e0.ENTRY
    entry.write_text(REAL_ENTRY.read_text(encoding="utf-8"), encoding="utf-8")
    entry.chmod(0o755)
    return package


def fake_source_receipt(root: Path, *, archive_sha: str = SOURCE_SHA) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    objects = [
        {
            "key": "pyproject.toml",
            "mode": 0o644,
            "type": "file",
            "size_bytes": 7,
            "sha256": hashlib.sha256(b"project").hexdigest(),
        }
    ]
    tree = {
        "algorithm": stage_e0.SOURCE_TREE_ALGORITHM,
        "tree_sha256": hashlib.sha256(stage_e0.canonical_json({"objects": objects}).encode()).hexdigest(),
        "objects": objects,
        "totals": {"objects": 1, "files": 1, "bytes": 7},
    }
    document = {
        "schema_version": stage_e0.SOURCE_RECEIPT_SCHEMA_VERSION,
        "kind": stage_e0.SOURCE_RECEIPT_KIND,
        "component": stage_e0.POLICY_SOURCE_COMPONENT,
        "git": {
            "git_sha": stage_e0.PINNED_POLICY_GIT_SHA,
            "git_tree_sha1": stage_e0.PINNED_POLICY_TREE_SHA1,
            "worktree_status": "clean_including_untracked_and_submodules",
        },
        "archive": {
            "uri": stage_e0.policy_source_uri(archive_sha),
            "sha256": archive_sha,
            "bytes": 123,
        },
        "extracted_tree": tree,
    }
    payload = (stage_e0.canonical_json(document) + "\n").encode()
    path = root / "source-receipt.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def staged_args(package: Path, *extra: str):
    receipt, receipt_sha = fake_source_receipt(package.parent / "receipt")
    return launch_e0.parser().parse_args(
        [
            "--policy-source-receipt",
            str(receipt),
            "--policy-source-receipt-sha256",
            receipt_sha,
            "--checkpoint-inventory-sha256",
            CHECKPOINT_SHA,
            "--fixtures-manifest-sha256",
            FIXTURES_SHA,
            "--source-package",
            str(package),
            *extra,
        ]
    )


def plan_for(package: Path, *extra: str) -> dict:
    args = staged_args(package, *extra)
    with launch_e0.staged_source(package) as source_dir:
        return launch_e0.build_plan(args, source_dir)


def test_dry_run_manifest_is_deterministic_and_pins_the_plan_backed_p5e_contract(tmp_path):
    package = fake_package(tmp_path / "a")
    first = plan_for(package)
    second = plan_for(package)

    assert first["run_id"] == second["run_id"] == f"amkv-e0-{first['manifest']['scientific_spec_sha256'][:16]}"
    assert first["manifest"] == second["manifest"]
    assert first["environment"] == second["environment"]
    assert first["manifest"]["kind"] == "amkv_e0_velocity_matching_attempt"
    assert first["manifest"]["infrastructure"] == {
        "provider": "aws_sagemaker",
        "execution_account": "141701954645",
        "queue": launch_e0.QUEUE,
        "training_plan_arn": "arn:aws:sagemaker:us-west-2:141701954645:training-plan/cam-robotics-tp",
        "role": launch_e0.ROLE_ARN,
        "instance_type": "ml.p5e.48xlarge",
        "accelerator": "8xH200",
        "priority": 400,
        "max_run_seconds": 6 * 3600,
        "volume_size_gb": 400,
        "reserved_capacity": "0",
        "attempts_in_job": 1,
    }
    scientific = first["manifest"]["scientific"]
    assert scientific["ratios"] == [4, 8]
    assert scientific["checkpoint"]["inventory_sha256"] == CHECKPOINT_SHA
    assert scientific["policy_source"]["sha256"] == SOURCE_SHA
    assert scientific["policy_source"]["receipt_sha256"] == first["environment"]["AMKV_POLICY_SOURCE_RECEIPT_SHA256"]
    assert scientific["fixtures"]["manifest_sha256"] == FIXTURES_SHA
    assert scientific["evaluation"]["runtime_dtype"] == "bfloat16"
    assert scientific["evaluation"]["speedup_claim_permitted"] is False
    assert scientific["code"]["sanitized_source_tree_sha256"] == first["source_sha"]
    assert first["source_sha"] == launch_e0.shipped_source_sha256(package)
    assert scientific["image"]["sha256"] == launch_e0.IMAGE_SHA
    assert first["output"].endswith(f"/results/e0/{first['run_id']}")
    assert set(first["environment"]) == set(REQUIRED_ENVIRONMENT)
    assert first["environment"]["SM_USE_RESERVED_CAPACITY"] == "0"
    assert first["environment"]["AMKV_RATIOS"] == "4,8"
    assert first["environment"]["RUN_MANIFEST_SOURCE"] == launch_e0.STAGED_MANIFEST
    assert first["environment"]["RUN_MANIFEST_SHA256"] == first["manifest"]["manifest_sha256"]

    # Every scientific input is inside the identity: changing one moves the run id and the output.
    ratios = plan_for(package, "--ratios", "4,8,16")
    assert ratios["run_id"] != first["run_id"]
    assert ratios["output"] != first["output"]
    other = launch_e0.parser().parse_args(
        [
            "--policy-source-receipt",
            staged_args(package).policy_source_receipt,
            "--policy-source-receipt-sha256",
            staged_args(package).policy_source_receipt_sha256,
            "--checkpoint-inventory-sha256",
            "d" * 64,
            "--fixtures-manifest-sha256",
            FIXTURES_SHA,
            "--source-package",
            str(package),
        ]
    )
    with launch_e0.staged_source(package) as source_dir:
        assert launch_e0.build_plan(other, source_dir)["run_id"] != first["run_id"]


def test_manifest_seal_matches_the_on_node_verification_idiom(tmp_path):
    plan = plan_for(fake_package(tmp_path / "seal"))
    document = json.loads(plan["manifest_json"])
    claimed = document.pop("manifest_sha256")
    import hashlib

    actual = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    assert claimed == actual == plan["environment"]["RUN_MANIFEST_SHA256"]


def test_priority_above_400_is_rejected(tmp_path):
    package = fake_package(tmp_path / "priority")
    with pytest.raises(SystemExit, match="must not exceed priority 400"):
        plan_for(package, "--priority", "600")
    assert plan_for(package, "--priority", "100")["manifest"]["infrastructure"]["priority"] == 100


def test_non_p5e_queue_and_role_drift_are_rejected(tmp_path):
    package = fake_package(tmp_path / "queue")
    with pytest.raises(SystemExit, match="pinned to the p5e training-plan queue"):
        plan_for(package, "--queue", "fss-tri-cam-robotics-p5-48xlarge-us-west-2")
    with pytest.raises(SystemExit, match="execution role"):
        plan_for(package, "--role", "arn:aws:iam::141701954645:role/other")
    assert launch_e0.training_plan_arn(launch_e0.QUEUE) is not None


def test_plan_backed_queue_requires_reserved_capacity_zero(tmp_path):
    package = fake_package(tmp_path / "reserved")
    with pytest.raises(SystemExit, match="SM_USE_RESERVED_CAPACITY=0"):
        plan_for(package, "--reserved-capacity", "1")


def test_runtime_cap_volume_and_ratio_contracts(tmp_path):
    package = fake_package(tmp_path / "caps")
    with pytest.raises(SystemExit, match="86400 seconds"):
        plan_for(package, "--max-run-seconds", str(24 * 3600 + 1))
    with pytest.raises(SystemExit, match="400 GiB"):
        plan_for(package, "--volume-size-gb", "250")
    with pytest.raises(SystemExit, match="ascending"):
        plan_for(package, "--ratios", "8,4")
    with pytest.raises(SystemExit, match="comma-separated positive integers"):
        plan_for(package, "--ratios", "4,x")


def test_submission_is_refused_without_confirm_and_with_unstaged_inputs(tmp_path, monkeypatch, capsys):
    package = fake_package(tmp_path / "submit")
    with pytest.raises(SystemExit, match="--confirm-submit"):
        plan_for(package, "--no-dry-run")

    def forbidden(**_kwargs):
        raise AssertionError("dry run must never reach submit_training_job")

    monkeypatch.setattr(launch_e0, "submit_training_job", forbidden)
    receipt, receipt_sha = fake_source_receipt(tmp_path / "main-receipt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_e0.py",
            "--dry-run",
            "--source-package",
            str(package),
            "--policy-source-receipt",
            str(receipt),
            "--policy-source-receipt-sha256",
            receipt_sha,
        ],
    )
    assert launch_e0.main() == 0
    output = capsys.readouterr().out
    assert "DRY RUN ONLY" in output
    assert "unstaged placeholder inputs" in output

    unstaged = launch_e0.parser().parse_args(
        [
            "--source-package",
            str(package),
            "--policy-source-receipt",
            str(receipt),
            "--policy-source-receipt-sha256",
            receipt_sha,
            "--no-dry-run",
            "--confirm-submit",
        ]
    )
    with pytest.raises(SystemExit, match="unstaged placeholder"):
        with launch_e0.staged_source(package) as source_dir:
            launch_e0.build_plan(unstaged, source_dir)


def test_staged_source_root_is_package_rooted_for_the_node_pythonpath(tmp_path):
    package = fake_package(tmp_path / "root")
    with launch_e0.staged_source(package) as source_dir:
        assert (source_dir / launch_e0.ENTRY).is_file()
        assert (source_dir / "robomme_integration" / "__init__.py").is_file()
        assert (source_dir / "robomme_integration" / "amkv" / launch_e0.ENTRY).is_file()
        for helper in launch_e0.RUNTIME_LAUNCH_HELPERS:
            assert (source_dir / "scripts" / "launch" / helper).is_file()
        isolated = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                (
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from robomme_integration.amkv import stage_e0; "
                    "assert stage_e0.SOURCE_RECEIPT_KIND == 'amkv_policy_source_stage_receipt'"
                ),
                str(source_dir),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert isolated.returncode == 0, isolated.stderr
    with pytest.raises(SystemExit, match="invalid AMKV source package"):
        with launch_e0.staged_source(tmp_path / "missing"):
            pass


def test_entry_script_is_valid_bash_and_asserts_the_whole_environment_contract():
    assert subprocess.run(["bash", "-n", str(REAL_ENTRY)], capture_output=True).returncode == 0
    text = REAL_ENTRY.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    for name in REQUIRED_ENVIRONMENT:
        assert name in text, f"{name} is absent from {REAL_ENTRY.name}"
        assert "FATAL missing $name" in text
    assert "unset PYTHONPATH PYTHONHOME" in text
    assert "uv sync --frozen" in text
    assert 'UV_PROJECT_ENVIRONMENT="$WORK/policy-venv"' in text
    for pinned in (
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9",
        "JAX_ENABLE_X64=false",
        "TF_CUDNN_DETERMINISTIC=1",
        "PYTHONHASHSEED=0",
    ):
        assert pinned in text
    assert 'PYTHONPATH="$CODE_DIR:$SRC/src"' in text
    assert "-m robomme_integration.amkv.e0_run" in text
    assert "--if-none-match '*'" in text
    assert 'publish_once "$WORK/e0_results.json"' in text
    assert "$AMKV_RUN_ID.complete.json" in text


def test_entry_environment_contract_matches_the_launcher_environment(tmp_path):
    plan = plan_for(fake_package(tmp_path / "contract"))
    text = REAL_ENTRY.read_text(encoding="utf-8")
    for name in plan["environment"]:
        assert name in text


def test_staging_destinations_are_content_addressed_under_the_amkv_root():
    assert stage_e0.AMKV_ROOT == (
        "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1/amkv"
    )
    assert stage_e0.policy_source_uri(SOURCE_SHA) == (
        f"{stage_e0.AMKV_ROOT}/code/robomme_policy_learning/{SOURCE_SHA}.tgz"
    )
    assert stage_e0.source_receipt_uri(FIXTURES_SHA) == (
        f"{stage_e0.AMKV_ROOT}/manifests/source_receipts/robomme_policy_learning/{FIXTURES_SHA}.json"
    )
    assert stage_e0.checkpoint_uri() == (f"{stage_e0.AMKV_ROOT}/artifacts/checkpoints/framesamp_modul_79999")
    assert stage_e0.checkpoint_inventory_uri(CHECKPOINT_SHA) == (
        f"{stage_e0.AMKV_ROOT}/manifests/inventories/amkv/{CHECKPOINT_SHA}.json"
    )
    assert stage_e0.fixtures_uri(FIXTURES_SHA) == (f"{stage_e0.AMKV_ROOT}/artifacts/amkv_fixtures/{FIXTURES_SHA}")
    assert stage_e0.split_uri(f"{stage_e0.AMKV_ROOT}/x") == (
        "sagemaker-us-west-2-141701954645",
        "sarvesh.patil/wsm_robocasa/studies/long_context_v1/amkv/x",
    )


def test_checkpoint_inventory_is_deterministic_and_self_addressed(tmp_path):
    root = tmp_path / "79999"
    (root / "params").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "params" / "shard").write_bytes(b"weights")
    (root / "assets" / "norm.json").write_bytes(b"{}")
    (root / "_CHECKPOINT_METADATA").write_bytes(b"meta")
    document, digest = stage_e0.build_inventory(
        root, artifact="framesamp_modul_79999", root_s3=stage_e0.checkpoint_uri()
    )
    import hashlib

    assert hashlib.sha256(document.encode()).hexdigest() == digest
    assert stage_e0.build_inventory(root, artifact="framesamp_modul_79999", root_s3=stage_e0.checkpoint_uri()) == (
        document,
        digest,
    )
    parsed = json.loads(document)
    assert [record["key"] for record in parsed["objects"]] == [
        "_CHECKPOINT_METADATA",
        "assets/norm.json",
        "params/shard",
    ]
    assert parsed["totals"] == {"objects": 3, "bytes": 7 + 2 + 4}

    # The official loader reads the step directory's SIBLING history_config.txt,
    # so the staged root is the step parent, not the step directory.
    (root.parent / "history_config.txt").write_bytes(b"perceptual-framesamp-modul.yaml")
    parent_document, parent_digest = stage_e0.build_inventory(
        root.parent, artifact="framesamp_modul_79999", root_s3=stage_e0.checkpoint_uri()
    )
    assert "history_config.txt" in [record["key"] for record in json.loads(parent_document)["objects"]]
    plan = stage_e0.stage_checkpoint(checkpoint_dir=root.parent, dry_run=True)
    assert plan["action"] == "planned"
    assert plan["inventory_sha256"] == parent_digest
    (root.parent / "history_config.txt").unlink()
    with pytest.raises(SystemExit, match="missing"):
        stage_e0.stage_checkpoint(checkpoint_dir=root.parent, dry_run=True)
    (root.parent / "history_config.txt").write_bytes(b"perceptual-framesamp-modul.yaml")
    (root / "_CHECKPOINT_METADATA").unlink()
    with pytest.raises(SystemExit, match="missing"):
        stage_e0.stage_checkpoint(checkpoint_dir=root.parent, dry_run=True)


def test_fixtures_staging_binds_the_payload_to_the_manifest(tmp_path):
    bundle = tmp_path / "fixtures"
    bundle.mkdir()
    payload = bundle / "fixtures.npz"
    payload.write_bytes(b"npz-bytes")
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps({"payload_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="payload_sha256 mismatch"):
        stage_e0.stage_fixtures(bundle_dir=bundle, dry_run=True)

    manifest.write_text(json.dumps({"payload_sha256": stage_e0.sha256_file(payload)}), encoding="utf-8")
    plan = stage_e0.stage_fixtures(bundle_dir=bundle, dry_run=True)
    assert plan["action"] == "planned"
    assert plan["manifest_sha256"] == stage_e0.sha256_file(manifest)
    assert plan["uri"] == stage_e0.fixtures_uri(plan["manifest_sha256"])


def test_staging_is_create_once_and_never_overwrites(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_aws(*arguments, check=True):
        calls.append(list(arguments))
        return subprocess.CompletedProcess(arguments, 0, stdout="{}", stderr="")

    monkeypatch.setattr(stage_e0, "aws", fake_aws)
    monkeypatch.setattr(stage_e0, "list_prefix", lambda _prefix: {"a": 1, "b": 2})
    assert stage_e0.sync_once(tmp_path, f"{stage_e0.AMKV_ROOT}/x", {"a": 1, "b": 2}) == "exists"
    assert not calls
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        stage_e0.sync_once(tmp_path, f"{stage_e0.AMKV_ROOT}/x", {"a": 1})

    body = tmp_path / "object.json"
    body.write_text("{}", encoding="utf-8")
    assert stage_e0.put_object_once(body, f"{stage_e0.AMKV_ROOT}/x/object.json") == "created"
    assert calls[-1][:2] == ["s3api", "put-object"]
    assert "--if-none-match" in calls[-1] and "*" in calls[-1]


def test_s3_connectivity_failure_is_never_mislabelled_as_an_absent_object(monkeypatch):
    monkeypatch.setattr(
        stage_e0,
        "aws",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            255,
            stdout="",
            stderr="Could not connect to the endpoint URL",
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        stage_e0.object_exists(f"{stage_e0.AMKV_ROOT}/missing")

    monkeypatch.setattr(
        stage_e0,
        "aws",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            255,
            stdout="",
            stderr="An error occurred (404) when calling HeadObject: Not Found",
        ),
    )
    assert stage_e0.object_exists(f"{stage_e0.AMKV_ROOT}/missing") is False


def test_policy_source_staging_requires_the_exact_clean_whole_git_tree(tmp_path, monkeypatch):
    source = tmp_path / "policy"
    source.mkdir()
    (source / "used_call_chain.py").write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=AMKV Test",
            "-c",
            "user.email=amkv@example.invalid",
            "commit",
            "-m",
            "pinned",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr(stage_e0, "PINNED_POLICY_GIT_SHA", commit)
    monkeypatch.setattr(stage_e0, "PINNED_POLICY_TREE_SHA1", tree)
    assert stage_e0.require_pinned_clean_policy_source(source) == {
        "git_sha": commit,
        "git_tree_sha1": tree,
    }
    staged = stage_e0.stage_source(
        source_dir=source,
        archive_dir=tmp_path / "archive",
        dry_run=True,
    )
    receipt, receipt_sha = stage_e0.load_source_receipt(
        Path(staged["receipt"]), expected_sha256=staged["receipt_sha256"]
    )
    assert receipt_sha == staged["receipt_sha256"]
    assert receipt["archive"]["sha256"] == staged["sha256"]
    assert receipt["git"] == {
        "git_sha": commit,
        "git_tree_sha1": tree,
        "worktree_status": "clean_including_untracked_and_submodules",
    }
    assert [record["key"] for record in receipt["extracted_tree"]["objects"]] == ["used_call_chain.py"]
    assert receipt["extracted_tree"]["totals"]["files"] == 1
    assert staged["receipt_uri"] == stage_e0.source_receipt_uri(receipt_sha)

    (source / "used_call_chain.py").write_text("PINNED = False\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not clean"):
        stage_e0.require_pinned_clean_policy_source(source)


def test_launcher_derives_archive_identity_from_the_validated_source_receipt(tmp_path):
    package = fake_package(tmp_path / "package")
    receipt, receipt_sha = fake_source_receipt(tmp_path / "receipt")
    args = launch_e0.parser().parse_args(
        [
            "--policy-source-receipt",
            str(receipt),
            "--policy-source-receipt-sha256",
            receipt_sha,
            "--checkpoint-inventory-sha256",
            CHECKPOINT_SHA,
            "--fixtures-manifest-sha256",
            FIXTURES_SHA,
        ]
    )
    with launch_e0.staged_source(package) as source_dir:
        plan = launch_e0.build_plan(args, source_dir)
    assert plan["manifest"]["scientific"]["policy_source"]["sha256"] == SOURCE_SHA

    tampered = bytearray(receipt.read_bytes())
    tampered[-2] = ord(" ")
    receipt.write_bytes(tampered)
    with launch_e0.staged_source(package) as source_dir:
        with pytest.raises(SystemExit, match="source receipt digest mismatch"):
            launch_e0.build_plan(args, source_dir)


def test_entry_pins_the_interpreter_that_has_mujoco_wheels():
    """Regression: attempt 1 died building mujoco 2.3.7 from source on 3.12.

    uv.lock carries cp311-only wheels for mujoco (openpi -> gym-aloha -> mujoco)
    while the official pyproject allows up to 3.12, so an unpinned `uv sync`
    can resolve an interpreter with no wheel and fall through to an sdist whose
    build requires MUJOCO_PATH.  3.11 is also the audited local interpreter.
    """

    entry = (pathlib.Path(__file__).resolve().parents[1] / "amkv" / "e0_entry.sh").read_text()
    assert "uv sync --frozen --python 3.11" in entry
    assert "UV_PYTHON=3.11" in entry
    # and the resolved interpreter is asserted, not assumed
    assert "PY_VERSION" in entry and '"3.11"' in entry
    assert "MUJOCO_PATH" in entry  # the reason is recorded next to the pin


def test_entry_keeps_the_receipted_policy_tree_immutable_during_imports():
    """Runtime imports must not manufacture bytecode before the receipt recheck."""

    entry = (pathlib.Path(__file__).resolve().parents[1] / "amkv" / "e0_entry.sh").read_text()
    bytecode_gate = "export PYTHONDONTWRITEBYTECODE=1"
    assert bytecode_gate in entry
    assert entry.index(bytecode_gate) < entry.index("uv sync --frozen")
    assert '"$PY_BIN" -B -m robomme_integration.amkv.e0_run' in entry


def test_env_preflight_marker_binds_to_the_lock_and_interpreter(tmp_path, monkeypatch):
    """Standing practice: a submission that builds an env fresh needs a clean-scratch sync."""

    source = tmp_path / "policy"
    source.mkdir()
    (source / "uv.lock").write_text("lock-v1", encoding="utf-8")
    archive = tmp_path / "archive"

    with pytest.raises(SystemExit, match="no environment preflight on record"):
        stage_e0.require_env_preflight(source_dir=source, archive_dir=archive)

    archive.mkdir()
    marker = {
        "uv_lock_sha256": stage_e0.lock_sha256(source),
        "python": stage_e0.PINNED_PYTHON,
        "checked_at": "2026-08-10T00:00:00Z",
    }
    stage_e0.env_preflight_marker_path(archive).write_text(json.dumps(marker), encoding="utf-8")
    assert stage_e0.require_env_preflight(source_dir=source, archive_dir=archive)["python"] == "3.11"

    # a changed lock invalidates the preflight rather than silently passing
    (source / "uv.lock").write_text("lock-v2", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale"):
        stage_e0.require_env_preflight(source_dir=source, archive_dir=archive)


def test_env_preflight_rejects_an_unpinned_interpreter(tmp_path):
    source = tmp_path / "policy"
    source.mkdir()
    (source / "uv.lock").write_text("lock", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    stage_e0.env_preflight_marker_path(archive).write_text(
        json.dumps({"uv_lock_sha256": stage_e0.lock_sha256(source), "python": "3.12"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="expected 3.11"):
        stage_e0.require_env_preflight(source_dir=source, archive_dir=archive)


def test_submit_path_requires_the_environment_preflight_without_an_escape_hatch():
    source = (pathlib.Path(__file__).resolve().parents[1] / "amkv" / "launch_e0.py").read_text()
    assert "require_env_preflight" in source
    assert "--skip-env-preflight" not in source
    assert "if args.confirm_submit:" in source


def test_code_preflight_binds_test_results_to_the_tree_being_shipped(tmp_path):
    """Standing practice after E0 attempt 2: verify the tree you actually ship.

    The lane's modules have more than one writer, so a suite that passed before
    an edit is not evidence about the package being uploaded now.
    """

    archive = tmp_path / "archive"
    archive.mkdir()
    with pytest.raises(SystemExit, match="no code preflight on record"):
        launch_e0.require_code_preflight("a" * 64, archive_dir=archive)

    launch_e0.code_preflight_marker_path(archive).write_text(
        json.dumps(
            {
                "sanitized_source_tree_sha256": "a" * 64,
                "pytest_summary": "115 passed",
                "checked_at": "2026-08-10T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    assert launch_e0.require_code_preflight("a" * 64, archive_dir=archive)["pytest_summary"] == "115 passed"
    with pytest.raises(SystemExit, match="different tree than the one about to ship"):
        launch_e0.require_code_preflight("b" * 64, archive_dir=archive)


def test_shipped_tree_sha_is_the_sanitized_package_tree():
    first = launch_e0.shipped_source_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    # deterministic: the same tree hashes the same way twice
    assert launch_e0.shipped_source_sha256() == first


def test_submit_path_requires_the_code_preflight_unless_explicitly_skipped():
    source = (pathlib.Path(__file__).resolve().parents[1] / "amkv" / "launch_e0.py").read_text()
    assert 'require_code_preflight(plan["source_sha"])' in source
    assert "args.confirm_submit and not args.skip_code_preflight" in source
