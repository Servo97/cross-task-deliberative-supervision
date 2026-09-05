#!/usr/bin/env python3
"""Build or approval-submit one sealed, node-resident RoboMME p5 eval campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "launch"))

from launch_guardrails import (  # noqa: E402
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    QUEUE,
    ROLE_ARN,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
)

from robomme_integration.eval import (  # noqa: E402
    campaign,
    parallel_campaign,
)
from robomme_integration.eval import (  # noqa: E402
    p5_parallel_action_preflight as action_canary,
)
from robomme_integration.launch import (  # noqa: E402
    IMAGE,
    IMAGE_SHA,
    OPENPI,
    OPENPI_SHA,
    PTRM_OPENPI,
    PTRM_OPENPI_SHA,
    STUDY,
    STUDY_ROOT,
)

ENTRY = "gpu_eval_campaign_entry.sh"
STAGED_QUEUE = "_robomme_eval_campaign_queue.json"
STAGED_PREFLIGHT = "_robomme_eval_campaign_preflight.json"
STAGED_RECEIPT = "_robomme_eval_campaign_runtime_receipt.json"
STAGED_LAUNCH = "_robomme_eval_campaign_launch.json"
GENERATED_FILES = (STAGED_QUEUE, STAGED_PREFLIGHT, STAGED_RECEIPT, STAGED_LAUNCH)
# Same entry-mode contract as launch_p5_preflight: staged 0755, toolkit runtime 0777, normalized
# back by gpu_eval_campaign_entry.sh before it re-hashes the unpacked tree.
SUBMITTED_ENTRY_MODE = 0o755
SAGEMAKER_RUNTIME_ENTRY_MODE = 0o777

RUNTIME_SHA = "60da89c378241f75b3244be408c845989ac79f06831e63e81191851c3e3803f2"
RUNTIME_S3 = f"{STUDY_ROOT}/artifacts/robomme/eval_runtime/v0.4.0/{RUNTIME_SHA}.tgz"

PRIORITY = 100
#: 100 = sweep class (default); 400 = standard class, allowed since 2026-09-05 on the lead's instruction.
ALLOWED_PRIORITIES = (100, 400)
MAX_RUN_SECONDS = 24 * 3600
VOLUME_GB = 200
STAGING_RESERVE_SECONDS = 2 * 3600
WORK_ROOT = Path("/opt/ml/work/robomme-eval-campaign")
SOURCE_ROOT = WORK_ROOT / "source"
RUNTIME_ROOT = WORK_ROOT / "runtime"
OPENPI_ROOT = WORK_ROOT / "openpi"
UPSTREAM_ROOT = WORK_ROOT / "upstream/robomme_policy_learning"
VISION_HOME = WORK_ROOT / "vision"
LINK_ROOT = WORK_ROOT / "links"

UPSTREAM_REPO = "https://github.com/RoboMME/robomme_policy_learning.git"
UPSTREAM_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
UPSTREAM_CRITICAL_SHA256 = {
    "src/mme_vla_suite/shared/siglip_tokenizer.py": (
        "72fb842327467a4d7cb0f770a514278d67b20721c84d59c82e6cae25f4ce0858"
    ),
    "src/mme_vla_suite/shared/data_utils.py": ("dda1583743528403aa97a4bde8c0305deacfb5a618c9c61937703e59ae76d27a"),
}
VISION_REVISION = "59bd9ff4d58ea0638064bda851fd7d477ee9708c"
VISION_SHA256 = "f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293"
VISION_BYTES = 1_659_216_368
VISION_S3 = f"{STUDY_ROOT}/artifacts/vision_encoders/pi05/{VISION_REVISION}/siglip_params.pkl"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _pretty(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(value: object, *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"uri", "sha256"}:
        raise SystemExit(f"{label} must contain exactly uri and sha256")
    parsed = urlparse(value["uri"] if isinstance(value["uri"], str) else "")
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise SystemExit(f"{label} URI is not an exact S3 object")
    if not isinstance(value["sha256"], str) or not HEX64.fullmatch(value["sha256"]):
        raise SystemExit(f"{label} SHA-256 is invalid")
    if value["sha256"] not in value["uri"]:
        raise SystemExit(f"{label} URI is not content addressed")
    return dict(value)


def _validate_preflight(
    payload: bytes,
    *,
    source_sha256: str,
    expected_openpi: dict,
    parallel_topology: dict | None = None,
) -> tuple[dict, str]:
    try:
        claim = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid native preflight claim: {error}") from error
    if not isinstance(claim, dict):
        raise SystemExit("native preflight claim must be one JSON object")
    if claim.get("kind") != campaign.PREFLIGHT_KIND:
        raise SystemExit("native preflight claim has the wrong kind")
    preflight_id = claim.get("preflight_id")
    expected_claim_s3 = f"{STUDY_ROOT}/manifests/claims/preflight/{preflight_id}.json"
    if not isinstance(preflight_id, str) or claim.get("claim_s3") != expected_claim_s3:
        raise SystemExit("native preflight claim URI is not canonical for its identity")
    _artifact(claim.get("runtime"), label="preflight runtime")
    _artifact(claim.get("openpi"), label="preflight OpenPI")
    if claim.get("runtime") != {"uri": RUNTIME_S3, "sha256": RUNTIME_SHA}:
        raise SystemExit("preflight runtime differs from the registered native evaluator")
    if claim.get("vla_eval_entrypoint") != {
        "kind": "python_module_wrapper",
        "module": "vla_eval.cli.main",
    }:
        raise SystemExit("preflight did not exercise the relocatable vla-eval module wrapper")
    infrastructure = claim.get("infrastructure")
    expected_infra = {
        "queue": QUEUE,
        "role": ROLE_ARN,
        "instance_type": "ml.p5.48xlarge",
        "accelerator": "8xH100",
    }
    if not isinstance(infrastructure, dict) or any(
        infrastructure.get(key) != expected for key, expected in expected_infra.items()
    ):
        raise SystemExit("preflight was not run through the exact cam-robotics p5 path")
    if infrastructure.get("priority") not in ALLOWED_PRIORITIES:
        raise SystemExit(
            f"preflight ran at priority {infrastructure.get('priority')}, not in {sorted(ALLOWED_PRIORITIES)}"
        )
    if parallel_topology is not None:
        try:
            action_canary.validate_success_claim(
                claim,
                source_sha256=source_sha256,
                expected_openpi=expected_openpi,
                expected_topology=parallel_topology,
                expected_image={"uri": IMAGE, "sha256": IMAGE_SHA},
                expected_vision={
                    "uri": VISION_S3,
                    "revision": VISION_REVISION,
                    "sha256": VISION_SHA256,
                    "bytes": VISION_BYTES,
                },
                expected_upstream={
                    "repo": UPSTREAM_REPO,
                    "commit": UPSTREAM_COMMIT,
                    "critical_sha256": UPSTREAM_CRITICAL_SHA256,
                },
                expected_infrastructure=expected_infra,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        if claim.get("status") != "native_render_reset_passed":
            raise SystemExit("native rendered-reset preflight has not passed")
        if claim.get("source_tree_sha256") != source_sha256:
            raise SystemExit("current sanitized eval source differs from the successful preflight; rerun preflight")
        if claim.get("openpi") != expected_openpi:
            raise SystemExit("preflight OpenPI differs from the queue cells' exact training source")
        if claim.get("image") != {"uri": IMAGE, "sha256": IMAGE_SHA}:
            raise SystemExit("preflight image differs from the campaign image")
        if claim.get("probe") != {
            "benchmark_adapter": ("robomme_integration.eval.benchmark:RoboMMEOfficialHistoryBenchmark"),
            "task": "MoveCube",
            "dataset": "test",
            "episode_idx": 0,
            "rendered_reset": True,
            "require_demo_history": True,
            "require_demo_state_history": True,
        }:
            raise SystemExit("preflight did not certify the exact paired test/demo-history rendered reset")
        sealed = dict(claim)
        manifest_sha = sealed.pop("manifest_sha256", None)
        sealed.pop("status", None)
        if not isinstance(manifest_sha, str) or campaign._seal_digest(sealed, "manifest_sha256") != manifest_sha:
            raise SystemExit("preflight manifest self-seal mismatch")
    return claim, _sha(payload)


def _vla_eval_wrapper() -> bytes:
    policy_python = OPENPI_ROOT / ".venv/bin/python"
    return (f'#!/usr/bin/env bash\nexec "{policy_python}" -m vla_eval.cli.main "$@"\n').encode()


def build_runtime_receipt(claim: dict, preflight_file_sha256: str, source_sha256: str) -> dict:
    receipt = {
        "schema_version": 1,
        "kind": campaign.RUNTIME_KIND,
        "status": "staged_and_verified",
        "preflight_claim_sha256": preflight_file_sha256,
        "source_tree_sha256": source_sha256,
        "generated_source_files_excluded": list(GENERATED_FILES),
        "runtime": claim["runtime"],
        "openpi": claim["openpi"],
        "vla_eval_wrapper": {
            "kind": "python_module_wrapper",
            "module": "vla_eval.cli.main",
            "sha256": _sha(_vla_eval_wrapper()),
        },
        "upstream": {
            "repo": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "critical_sha256": UPSTREAM_CRITICAL_SHA256,
        },
        "vision": {
            "uri": VISION_S3,
            "revision": VISION_REVISION,
            "sha256": VISION_SHA256,
            "bytes": VISION_BYTES,
        },
        "paths": {
            "policy_python": str(OPENPI_ROOT / ".venv/bin/python"),
            "vla_eval": str(LINK_ROOT / "vla-eval"),
            "harness_src": str(LINK_ROOT / "harness-src"),
            "robomme_src": str(LINK_ROOT / "robomme-src"),
            "maniskill_src": str(LINK_ROOT / "maniskill-src"),
            "openpi_src": str(OPENPI_ROOT / "src"),
            "policy_site": str(OPENPI_ROOT / ".venv/lib/python3.11/site-packages"),
            "simulator_site": str(LINK_ROOT / "simulator-site"),
            "upstream_root": str(UPSTREAM_ROOT),
            "vision_encoder_home": str(VISION_HOME),
        },
        "render_environment": {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "ROBOMME_USE_LAVAPIPE": "auto",
        },
    }
    return campaign.seal_document(receipt, field="receipt_sha256")


def finalize_queue(
    template: dict,
    *,
    claim: dict,
    claim_sha256: str,
    receipt: dict,
    topology: dict | None = None,
) -> dict:
    if topology is not None:
        queue_id = template.get("queue_id")
        if (
            "queue_manifest_sha256" in template
            or not isinstance(queue_id, str)
            or not queue_id.endswith("-parallel-v1")
        ):
            raise ValueError("parallel p5 topology requires a fresh unsealed *-parallel-v1 queue identity")
    queue = dict(template)
    queue.pop("queue_manifest_sha256", None)
    if topology is not None:
        queue["topology"] = dict(topology)
    receipt_bytes = _pretty(receipt)
    queue["gates"] = {
        "native_preflight": {
            "preflight_id": claim["preflight_id"],
            "claim_sha256": claim_sha256,
            "source_tree_sha256": claim["source_tree_sha256"],
        },
        "runtime_receipt": {
            "receipt_sha256": _sha(receipt_bytes),
            "runtime_artifact_sha256": claim["runtime"]["sha256"],
            "openpi_sha256": claim["openpi"]["sha256"],
        },
    }
    return campaign.seal_document(queue, field="queue_manifest_sha256")


def _validate_launch(args: argparse.Namespace) -> None:
    if args.queue != QUEUE or args.role != ROLE_ARN:
        raise SystemExit("RoboMME evaluation must use the ordinary cam-robotics p5 queue/role")
    if args.priority not in ALLOWED_PRIORITIES:
        raise SystemExit(f"RoboMME p5 evaluation must use priority in {sorted(ALLOWED_PRIORITIES)}")
    if not 1 <= args.max_run_seconds <= MAX_RUN_SECONDS:
        raise SystemExit("RoboMME p5 evaluation is capped at 24 hours")
    if args.volume_size_gb != VOLUME_GB:
        raise SystemExit("RoboMME p5 evaluation uses exactly 200 GiB of bounded scratch disk")


def build_plan(args: argparse.Namespace, source_dir: Path) -> dict:
    _validate_launch(args)
    topology = (
        parallel_campaign.p5_8xh100_topology().as_queue_topology()
        if getattr(args, "parallel_fixed50", False)
        else None
    )
    with prepared_source_bundle(source_dir, ENTRY, {"SAGEMAKER_PROGRAM": ENTRY}, None) as (
        staged,
        _,
        _,
    ):
        submitted_mode = stat.S_IMODE((staged / ENTRY).lstat().st_mode)
        if submitted_mode != SUBMITTED_ENTRY_MODE:
            raise SystemExit(
                f"submitted campaign entry mode drifted: {oct(submitted_mode)} != {oct(SUBMITTED_ENTRY_MODE)}"
            )
        source_sha = source_tree_sha256(staged)
    template = json.loads(args.queue_template.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise SystemExit("eval queue template must contain one JSON object")
    expected_openpi = _artifact(
        template.get("comparability", {}).get("serving_openpi"),
        label="queue training-matched OpenPI",
    )
    registered_openpi = {
        OPENPI_SHA: {"uri": OPENPI, "sha256": OPENPI_SHA, "profile": "standard"},
        PTRM_OPENPI_SHA: {
            "uri": PTRM_OPENPI,
            "sha256": PTRM_OPENPI_SHA,
            "profile": "advanced",
        },
    }
    registered = registered_openpi.get(expected_openpi["sha256"])
    if registered is None or expected_openpi != {key: registered[key] for key in ("uri", "sha256")}:
        raise SystemExit("queue uses an unregistered OpenPI training archive")
    preflight_bytes = args.native_preflight_claim.read_bytes()
    claim, claim_file_sha = _validate_preflight(
        preflight_bytes,
        source_sha256=source_sha,
        expected_openpi=expected_openpi,
        parallel_topology=topology,
    )
    receipt = build_runtime_receipt(claim, claim_file_sha, source_sha)
    queue = finalize_queue(
        template,
        claim=claim,
        claim_sha256=claim_file_sha,
        receipt=receipt,
        topology=topology,
    )
    campaign.validate_queue(queue, source_root=REPO_ROOT)
    if queue["limits"]["max_run_seconds"] > args.max_run_seconds - STAGING_RESERVE_SECONDS:
        raise SystemExit("sealed eval queue must leave two hours inside the p5 job for exact runtime staging")
    queue_bytes = _pretty(queue)
    receipt_bytes = _pretty(receipt)
    launch = campaign.seal_document(
        {
            "schema_version": 1,
            "kind": "robomme_p5_eval_campaign_launch",
            "source_tree_sha256": source_sha,
            "generated_source_files": list(GENERATED_FILES),
            "queue_id": queue["queue_id"],
            "queue_manifest_sha256": queue["queue_manifest_sha256"],
            "parallel_topology_sha256": (topology["parallel_topology_sha256"] if topology is not None else None),
            "queue_file_sha256": _sha(queue_bytes),
            "preflight_id": claim["preflight_id"],
            "preflight_claim_sha256": claim_file_sha,
            "runtime_receipt_file_sha256": _sha(receipt_bytes),
            "runtime": claim["runtime"],
            "openpi": claim["openpi"],
            "vision": receipt["vision"],
            "upstream": receipt["upstream"],
            "infrastructure": {
                "provider": "aws_sagemaker",
                "execution_account": EXECUTION_ACCOUNT,
                "queue": QUEUE,
                "role": ROLE_ARN,
                "instance_type": "ml.p5.48xlarge",
                "accelerator": "8xH100",
                "priority": args.priority,
                "max_run_seconds": args.max_run_seconds,
                "staging_reserve_seconds": STAGING_RESERVE_SECONDS,
                "volume_size_gb": VOLUME_GB,
            },
        },
        field="launch_manifest_sha256",
    )
    launch_bytes = _pretty(launch)
    staged_files = {
        STAGED_QUEUE: queue_bytes,
        STAGED_PREFLIGHT: preflight_bytes,
        STAGED_RECEIPT: receipt_bytes,
        STAGED_LAUNCH: launch_bytes,
    }
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "ROBOMME_EVAL_QUEUE_SOURCE": STAGED_QUEUE,
        "ROBOMME_EVAL_QUEUE_FILE_SHA256": _sha(queue_bytes),
        "ROBOMME_EVAL_PREFLIGHT_SOURCE": STAGED_PREFLIGHT,
        "ROBOMME_EVAL_PREFLIGHT_SHA256": claim_file_sha,
        "ROBOMME_EVAL_PREFLIGHT_CLAIM_S3": claim["claim_s3"],
        "ROBOMME_EVAL_RECEIPT_SOURCE": STAGED_RECEIPT,
        "ROBOMME_EVAL_RECEIPT_FILE_SHA256": _sha(receipt_bytes),
        "ROBOMME_EVAL_LAUNCH_SOURCE": STAGED_LAUNCH,
        "ROBOMME_EVAL_LAUNCH_FILE_SHA256": _sha(launch_bytes),
        "ROBOMME_EVAL_SOURCE_TREE_SHA256": source_sha,
        "ROBOMME_EVAL_GENERATED_FILES": ",".join(GENERATED_FILES),
        "ROBOMME_EVAL_MAX_RUN_SECONDS": str(args.max_run_seconds),
        "ROBOMME_EVAL_OPENPI_PROFILE": registered["profile"],
        "ROBOMME_EVAL_RUNTIME_S3": claim["runtime"]["uri"],
        "ROBOMME_EVAL_RUNTIME_SHA256": claim["runtime"]["sha256"],
        "OPENPI_FORK_S3": claim["openpi"]["uri"],
        "OPENPI_SHA256": claim["openpi"]["sha256"],
        "ROBOMME_EVAL_UPSTREAM_REPO": UPSTREAM_REPO,
        "ROBOMME_EVAL_UPSTREAM_COMMIT": UPSTREAM_COMMIT,
        "ROBOMME_EVAL_VISION_S3": VISION_S3,
        "ROBOMME_EVAL_VISION_SHA256": VISION_SHA256,
        "ROBOMME_EVAL_VISION_BYTES": str(VISION_BYTES),
    }
    return {
        "source_sha256": source_sha,
        "queue": queue,
        "receipt": receipt,
        "launch": launch,
        "preflight_claim": claim,
        "environment": environment,
        "staged_files": staged_files,
    }


def validate_published_preflight(
    plan: dict,
    *,
    claim_payload: bytes,
    evidence_payload: bytes | None,
) -> None:
    staged = plan["staged_files"][STAGED_PREFLIGHT]
    if claim_payload != staged:
        raise SystemExit("local preflight claim differs from its immutable S3 object")
    claim = json.loads(claim_payload)
    evidence = claim.get("evidence")
    if claim.get("status") == action_canary.CANARY_STATUS:
        if (
            not isinstance(evidence, dict)
            or evidence_payload is None
            or len(evidence_payload) != evidence.get("bytes")
            or hashlib.sha256(evidence_payload).hexdigest() != evidence.get("sha256")
        ):
            raise SystemExit("immutable action-preflight evidence is absent or corrupt")
    elif evidence_payload is not None:
        raise SystemExit("render-reset preflight unexpectedly supplied action evidence")


def _read_s3_bytes(uri: str) -> bytes:
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--only-show-errors", "--region", "us-west-2"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise SystemExit(f"failed to authenticate immutable preflight object {uri}: {result.stderr[:500]!r}")
    return result.stdout


def verify_published_preflight(plan: dict) -> None:
    claim = plan["preflight_claim"]
    claim_payload = _read_s3_bytes(claim["claim_s3"])
    evidence = claim.get("evidence")
    evidence_payload = _read_s3_bytes(evidence["uri"]) if claim.get("status") == action_canary.CANARY_STATUS else None
    validate_published_preflight(plan, claim_payload=claim_payload, evidence_payload=evidence_payload)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-dir", type=Path, default=REPO_ROOT / "robomme_integration")
    value.add_argument("--queue-template", type=Path, required=True)
    value.add_argument("--native-preflight-claim", type=Path, required=True)
    value.add_argument("--queue", default=QUEUE)
    value.add_argument("--role", default=ROLE_ARN)
    value.add_argument("--priority", type=int, default=PRIORITY)
    value.add_argument("--max-run-seconds", type=int, default=MAX_RUN_SECONDS)
    value.add_argument("--volume-size-gb", type=int, default=VOLUME_GB)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "--parallel-fixed50",
        action="store_true",
        help="seal one disjoint H100/port/CPU lane per concurrent fixed50 cell",
    )
    value.add_argument(
        "--confirm-submit",
        action="store_true",
        help="submit only after explicit user approval; otherwise this command is a dry run",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    args.queue_template = args.queue_template.expanduser().resolve()
    args.native_preflight_claim = args.native_preflight_claim.expanduser().resolve()
    plan = build_plan(args, source_dir)
    print(json.dumps(plan["launch"], indent=2, sort_keys=True))
    if args.dry_run or not args.confirm_submit:
        print("DRY RUN ONLY — no AWS SDK loaded and no cloud read/write performed")
        return
    verify_published_preflight(plan)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=IMAGE,
        instance_type="ml.p5.48xlarge",
        volume_size=VOLUME_GB,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": OWNER_EMAIL},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.benchmark", "Value": "RoboMME"},
            {"Key": "wsm.kind", "Value": "eval-campaign"},
        ],
        retry_config={"attempts": 1},
        job_name=f"sarvesh-rmme-eval-{plan['queue']['queue_id'][:24]}-{stamp}",
        queue=QUEUE,
        role=ROLE_ARN,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=None,
        confirmed=True,
        disable_profiler=True,
        expected_source_tree_sha256=plan["source_sha256"],
        staged_source_files=plan["staged_files"],
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
