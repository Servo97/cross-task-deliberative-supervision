#!/usr/bin/env python3
"""Approval-gated launcher for the RoboCerebra (H12) pi0.5 ablation.

Deliberately thin. Everything about queues, roles, priority caps, training-plan pinning and the
sanitized source bundle is delegated to ``launch_guardrails``; this file only decides *what* the
node is told to do. It is the sibling of ``submit_pi_stage_s.py``, not a variant of it: that
launcher is bound to RoboCasa/ReMemBench object inventories, a 50-task materializer and a
RoboCasa Phase-1 init checkpoint, none of which exist on this campaign. RoboCerebra is a single
content-addressed LeRobot tarball finetuned from the *released* pi05_libero checkpoint.

``--dry-run`` is fully offline apart from building the two source archives locally: no AWS SDK is
loaded, nothing is uploaded, nothing is submitted. Submission additionally requires prior explicit
user approval and ``--confirm-submit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_deterministic_archive import build_archive
from launch_guardrails import (
    DEFAULT_RESULTS_BUCKET,
    DEXJOCO_IMAGE_REPO,
    OWNER_EMAIL,
    PROJECT_TAG,
    QUEUE,
    REGION,
    STUDY_OWNER,
    TRAINING_PLAN_QUEUE,
    add_guardrail_arguments,
    normalize_queue,
    submit_training_job,
    training_plan_arn,
    validate_and_confirm,
    wsm_settings,
)

STUDY = "long_context_v1"
DEFAULT_OWNER = STUDY_OWNER
#: The `tri.owner.email` SCP tag. Deliberately INDEPENDENT of --user: --user names the S3 storage
#: prefix, and every content address in this study is minted under `sarvesh.patil`, so that prefix
#: can never move -- while the submitting identity is now sarvesh.patil.pi@tri.global. Deriving the
#: tag from --user (the old f"{args.user}@tri.global") tags a DEACTIVATED address, and org SCP
#: p-ahpdy5vv denies batch:SubmitServiceJob unless tri.owner.email is present, non-empty and valid.
#: Same split as submit_pi_stage_s.py (§22.5 deviation 3).
DEFAULT_OWNER_EMAIL = OWNER_EMAIL
ENTRY = "robocerebra_pi05_entry.sh"
#: Must equal ``STAGED_MANIFEST_NAME`` in the entry script. A constant on both sides rather than an
#: env var, so the launcher's env dict and the entry's env reads stay a matched pair.
STAGED_MANIFEST_NAME = "_robocerebra_run_manifest.json"

#: The RoboCasa/GR00T runtime image in account 141, pinned BY DIGEST. A moving ``:latest`` would
#: silently change the scientific artifact between arms.
IMAGE_URI = f"{DEXJOCO_IMAGE_REPO}@sha256:798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"

#: Each queue's service environment is bound to one instance family, so the instance type is a
#: function of ``--queue``, never a free parameter.
QUEUE_INSTANCE_TYPES = {
    QUEUE: "ml.p5.48xlarge",
    TRAINING_PLAN_QUEUE: "ml.p5e.48xlarge",
}
#: Short spellings accepted on the CLI. ``normalize_queue`` still handles the roster's
#: ``shared-compute__<region>__`` prefix.
QUEUE_ALIASES = {"p5": QUEUE, "p5e-plan": TRAINING_PLAN_QUEUE}

#: Campaign-wide caps, tighter than the guardrail module's global ones. This campaign is an
#: ablation on a shared queue and is never allowed to outrank production work.
MAX_PRIORITY = 400
DEFAULT_PRIORITY = 400
#: Raised 24 h -> 48 h on 2026-09-02 (user: "run 48 hr runs on 400 priority") for the A19 60k
#: budget curve — the live rate is 1.36-1.39 s/step, so 60k needs ~24 h and a 24 h cap left a
#: 1-5 % margin on step 59999 (h14_p0_status.md §51). Priority stays capped at 400.
MAX_RUN_SECONDS = 48 * 3600
DEFAULT_TRAIN_STEPS = 30_000
#: One attempt. A retry is a new launch and therefore returns through the approval gate.
RETRY = {"attempts": 1}

#: arm -> (registered openpi config name, needs the omega cache AT TRAIN TIME).
#:
#: The flag is about TRAINING, which is all this launcher controls. Every mechanism arm needs the
#: omega feature store to train, including A3: JEPA consumes omega as an auxiliary *target*, so it
#: reads the same store even though the trained policy never sees omega again.
#:
#: Do not confuse this with what an arm needs at SERVE time, which is a different set:
#:   A1/A2/A4 read omega online (the server must run the encoder);
#:   A0/A3 do not (they serve exactly like a plain pi05_libero checkpoint).
#: A4 in particular DOES read omega at serve — PTRM is a recursive head layered on A1's read, so
#: its serve cost equals A1's; it is "training-side" only in the sense that its *gain* shows up
#: without extra inference-time search.
ARMS: dict[str, tuple[str, bool]] = {
    "a0_base": ("pi05_robocerebra_base", False),
    "a1_gdn_w8": ("pi05_robocerebra_gdn_w8", True),
    "a2_gdn_w16_hd05": ("pi05_robocerebra_gdn_w16_hd05", True),
    "a3_jepa": ("pi05_robocerebra_jepa", True),
    "a4_ptrm": ("pi05_robocerebra_ptrm", True),
    # A5 = the project's Q2 cell (pi0.5 fast weights + L=8 sequence windows). NOT paper RoboTTT/R1:
    # the official RoboTTT is GR00T-only. It reads no omega -- the fast weights are internal state,
    # so no omega artifacts are attached.
    "a5_stageq_q2": ("pi05_robocerebra_stageq_q2", False),
}

#: Arms whose policy reads omega at inference (documented here so the eval side has one source of
#: truth; not used by this launcher, which only trains).
SERVE_READS_OMEGA = frozenset({"a1_gdn_w8", "a2_gdn_w16_hd05", "a4_ptrm"})

#: All five configs are registered and CPU-preflighted (real dataloader, one batch each).
IMPLEMENTED_ARMS = frozenset(ARMS)

#: Environment keys the node entry does not read: SageMaker/Batch infrastructure only. Every other
#: key the launcher sets must appear in the entry (checked on every invocation, see
#: ``verify_entry_env_contract``).
INFRA_ONLY_ENV = frozenset({"SAGEMAKER_PROGRAM", "SM_USE_RESERVED_CAPACITY"})

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def study_root(owner: str) -> str:
    if not _OWNER.fullmatch(owner):
        raise SystemExit(f"invalid --user storage owner {owner!r}")
    return f"s3://{DEFAULT_RESULTS_BUCKET}/{owner}/wsm_robocasa/studies/{STUDY}"


def campaign_root(owner: str) -> str:
    return f"{study_root(owner)}/robocerebra"


def content_addressed(uri: str, *, flag: str, prefix: str, extension: str) -> str:
    """Validate ``<prefix>/<64hex><extension>`` and return the digest the key claims."""
    match = re.fullmatch(rf"{re.escape(prefix)}/([0-9a-f]{{64}}){re.escape(extension)}", uri)
    if match is None:
        raise SystemExit(f"{flag} must be content-addressed at {prefix}/<64hex>{extension}; got {uri}")
    return match.group(1)


def resolve_queue(value: str) -> str:
    queue = QUEUE_ALIASES.get(value, value)
    queue = normalize_queue(queue)
    if queue not in QUEUE_INSTANCE_TYPES:
        raise SystemExit(f"--queue must be one of {sorted(QUEUE_ALIASES)} (or their full names); got {value}")
    return queue


def enforce_campaign_caps(args: argparse.Namespace) -> None:
    """Campaign limits on top of ``validate_launch_contract``."""
    if args.priority > MAX_PRIORITY:
        raise SystemExit(
            f"the RoboCerebra ablation is never allowed above priority {MAX_PRIORITY}; got {args.priority}"
        )
    if not 1 <= args.priority:
        raise SystemExit(f"--priority must be a positive integer; got {args.priority}")
    if args.max_run_seconds > MAX_RUN_SECONDS:
        raise SystemExit(
            f"--max-run-seconds must be <= {MAX_RUN_SECONDS} on this campaign; got {args.max_run_seconds}"
        )
    if args.train_steps < 1:
        raise SystemExit("--train-steps must be a positive integer")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be a positive integer when given")


def resolve_source_dir(value: str | None) -> pathlib.Path:
    path = pathlib.Path(value or pathlib.Path(__file__).resolve().parents[3] / "internal_training").resolve()
    if not (path / ENTRY).is_file():
        raise SystemExit(f"source-dir {path} is missing {ENTRY}")
    return path


def verify_entry_env_contract(source_dir: pathlib.Path, environment: dict) -> list[str]:
    """Fail closed when the launcher sets a variable the entry never reads.

    Cheap, and it closes the one drift this pair is most exposed to: the launcher and the entry
    are two files that share an undeclared interface. Returns the checked (non-infra) key list.
    """
    entry_text = (source_dir / ENTRY).read_text(encoding="utf-8")
    checked = sorted(key for key in environment if key not in INFRA_ONLY_ENV)
    missing = [key for key in checked if key not in entry_text]
    if missing:
        raise SystemExit(
            f"{ENTRY} never reads these variables the launcher sets: {missing}. "
            "Either the entry lost a read or the launcher gained a dead key."
        )
    return checked


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_code_archives(args: argparse.Namespace, root: str) -> dict[str, dict]:
    """Build both deterministic source archives locally and return {component: {...}}.

    Built on ``--dry-run`` too: the archive digests ARE part of the sealed spec, so a dry run that
    skipped them would print a run_id that no real submission could reproduce.
    """
    wsmv2_dir = pathlib.Path(__file__).resolve().parents[2]
    openpi_dir = pathlib.Path(args.openpi_source_dir).resolve()
    if not (openpi_dir / "pyproject.toml").is_file():
        raise SystemExit(f"--openpi-source-dir {openpi_dir} is not an openpi checkout")
    cache = pathlib.Path(args.archive_cache_dir).resolve()
    archives = {}
    for component, source in (("wsmv2", wsmv2_dir), ("openpi", openpi_dir)):
        path, digest, uri = build_archive(source, output_dir=cache / component, component=component, study_root=root)
        archives[component] = {
            "sha256": digest,
            "uri": uri,
            "path": str(path),
            "bytes": path.stat().st_size,
            "source_dir": str(source),
        }
    return archives


def publish_archive_once(local_path: pathlib.Path, uri: str) -> None:
    """Create-once upload. The key IS the digest, so an existing key is the same bytes."""
    location = uri[len("s3://") :]
    bucket, key = location.split("/", 1)
    completed = subprocess.run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(local_path),
            "--if-none-match",
            "*",
            "--region",
            REGION,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if completed.returncode == 0:
        print(f"  published {uri}")
        return
    detail = completed.stderr.decode(errors="replace")
    if "PreconditionFailed" in detail or "412" in detail:
        print(f"  already published (content-addressed, identical by construction) {uri}")
        return
    raise SystemExit(f"failed to publish {uri}: {detail[:500]}")


def build_plan(args: argparse.Namespace, source_dir: pathlib.Path) -> dict:
    config_name, wants_omega = ARMS[args.arm]
    config_name = args.config_name or config_name
    if args.arm not in IMPLEMENTED_ARMS:
        raise NotImplementedError(
            f"arm {args.arm} maps to openpi config {config_name!r}, which is not registered yet. "
            "Register it in openpi/training/robocerebra_configs.py, then add the arm to "
            "IMPLEMENTED_ARMS."
        )
    root = study_root(args.user)
    namespace = campaign_root(args.user)
    plan_arn = training_plan_arn(args.queue)

    dataset_sha = content_addressed(
        args.dataset_tar_s3,
        flag="--dataset-tar-s3",
        prefix=f"{namespace}/data/lerobot",
        extension=".tar",
    )
    norm_stats_sha = content_addressed(
        args.norm_stats_s3,
        flag="--norm-stats-s3",
        prefix=f"{namespace}/assets/norm_stats",
        extension=".json",
    )
    init_sha = content_addressed(
        args.init_ckpt_tar_s3,
        flag="--init-ckpt-tar-s3",
        prefix=f"{namespace}/init",
        extension=".tar",
    )
    tokenizer_sha = content_addressed(
        args.tokenizer_s3,
        flag="--tokenizer-s3",
        prefix=f"{root}/artifacts/tokenizers/paligemma",
        extension=".model",
    )
    artifacts = {
        "lerobot_dataset_tar": {"uri": args.dataset_tar_s3, "sha256": dataset_sha},
        "norm_stats": {"uri": args.norm_stats_s3, "sha256": norm_stats_sha},
        "init_checkpoint_tar": {"uri": args.init_ckpt_tar_s3, "sha256": init_sha},
        "paligemma_tokenizer": {"uri": args.tokenizer_s3, "sha256": tokenizer_sha},
    }

    # Omega is a mechanism-arm input. It is passed ONLY when the arm consumes it, so a base-arm
    # spec cannot carry an omega key and therefore cannot have its run_id perturbed by one.
    omega_features_s3 = None
    omega_encoder_s3 = None
    if wants_omega:
        if not (args.omega_features_tar_s3 and args.omega_encoder_s3):
            raise SystemExit(f"arm {args.arm} requires both omega artifacts")
        omega_features_s3 = args.omega_features_tar_s3
        omega_encoder_s3 = args.omega_encoder_s3
        artifacts["omega_features_tar"] = {
            "uri": omega_features_s3,
            "sha256": content_addressed(
                omega_features_s3,
                flag="--omega-features-tar-s3",
                prefix=f"{namespace}/omega/features",
                extension=".tar",
            ),
        }
        artifacts["omega_encoder"] = {
            "uri": omega_encoder_s3,
            "sha256": content_addressed(
                omega_encoder_s3,
                flag="--omega-encoder-s3",
                prefix=f"{namespace}/omega/encoder",
                extension=".pt",
            ),
        }

    archives = build_code_archives(args, root)
    exp_name = args.exp_name or f"rcerebra_{args.arm}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", exp_name):
        raise SystemExit(f"--exp-name must be [A-Za-z0-9._-]+; got {exp_name!r}")

    # The SEALED spec: scientific identity only. Queue, priority, instance type, timestamps and job
    # names are launch logistics -- folding them in would give the same experiment a different
    # run_id on a different queue, and would make the create-once manifest collide with itself.
    training = {"train_steps": args.train_steps}
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    spec = {
        "schema_version": 1,
        "kind": "robocerebra_pi05_finetune",
        "campaign": "robocerebra_h12",
        "study": STUDY,
        "arm": args.arm,
        "config_name": config_name,
        "exp_name": exp_name,
        "training": training,
        "artifacts": {name: value["sha256"] for name, value in artifacts.items()},
        "sources": {
            "wsmv2": archives["wsmv2"]["sha256"],
            "openpi": archives["openpi"]["sha256"],
        },
    }
    spec_sha = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    run_id = f"{args.arm}-{spec_sha[:16]}"
    checkpoint_uri = f"{namespace}/checkpoints/pi05/{args.arm}/{run_id}"
    manifest_uri = f"{namespace}/manifests/runs/{run_id}.json"
    manifest = {
        **spec,
        "run_id": run_id,
        "spec_sha256": spec_sha,
        "artifact_uris": {name: value["uri"] for name, value in artifacts.items()},
        "source_uris": {component: value["uri"] for component, value in archives.items()},
        "checkpoint_uri": checkpoint_uri,
        "manifest_uri": manifest_uri,
        "completion_claim_uri": (f"{namespace}/checkpoints/pi05/{args.arm}/claims/{run_id}.complete.json"),
        "image_uri": args.image_uri,
    }

    environment = {
        # A pinned training plan REPLACES the implicit reserved-capacity request; the two are
        # alternatives, not a pair. submit_training_job re-checks this exact pairing.
        "SM_USE_RESERVED_CAPACITY": "0" if plan_arn else "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "OPENPI_FORK_S3": archives["openpi"]["uri"],
        "WSMV2_S3": archives["wsmv2"]["uri"],
        "DATASET_TAR_S3": args.dataset_tar_s3,
        "NORM_STATS_S3": args.norm_stats_s3,
        "INIT_CKPT_TAR_S3": args.init_ckpt_tar_s3,
        "TOKENIZER_S3": args.tokenizer_s3,
        "TOKENIZER_SHA256": tokenizer_sha,
        "CONFIG_NAME": config_name,
        "EXP_NAME": exp_name,
        "TRAIN_STEPS": str(args.train_steps),
        "CHECKPOINT_URI": checkpoint_uri,
        "RUN_ID": run_id,
        "MANIFEST_URI": manifest_uri,
    }
    if args.batch_size is not None:
        environment["BATCH_SIZE"] = str(args.batch_size)
    if args.save_interval is not None:
        environment["SAVE_INTERVAL"] = str(args.save_interval)
    if omega_features_s3:
        environment["OMEGA_FEATURES_TAR_S3"] = omega_features_s3
    if omega_encoder_s3:
        environment["OMEGA_ENCODER_S3"] = omega_encoder_s3

    checked_env_keys = verify_entry_env_contract(source_dir, environment)

    return {
        "run_id": run_id,
        "spec": spec,
        "spec_sha256": spec_sha,
        "manifest": manifest,
        "manifest_json": canonical_json(manifest),
        "environment": environment,
        "checked_env_keys": checked_env_keys,
        "archives": archives,
        "instance_type": QUEUE_INSTANCE_TYPES[args.queue],
        "training_plan_arn": plan_arn,
        "checkpoint_uri": checkpoint_uri,
        "manifest_uri": manifest_uri,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--config-name",
        default=None,
        help="registered openpi config; defaults to the arm's config",
    )
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the config's batch size; omitted keeps the recipe default (256)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="override TrainConfig.save_interval (canary: small value to force a MID-RUN save so "
        "the mid-run checkpoint sync is actually exercised)",
    )
    parser.add_argument("--exp-name", default=None, help="defaults to rcerebra_<arm>")
    parser.add_argument("--user", default=DEFAULT_OWNER, help="S3 storage owner prefix")
    parser.add_argument(
        "--owner-email",
        default=DEFAULT_OWNER_EMAIL,
        help=(
            "value of the required `tri.owner.email` SCP tag; independent of --user, which only "
            f"names the S3 storage prefix (default {DEFAULT_OWNER_EMAIL})"
        ),
    )
    parser.add_argument("--source-dir", default=None, help="internal_training checkout")
    parser.add_argument(
        "--openpi-source-dir",
        default=str(wsm_settings.ROBOCASA_OPENPI_ROOT),
        help="openpi fork checkout to archive and ship",
    )
    parser.add_argument(
        "--archive-cache-dir",
        default=str(pathlib.Path.home() / ".cache" / "wsm_launch_archives"),
        help="where the deterministic <sha256>.tgz archives are built",
    )
    parser.add_argument("--image-uri", default=IMAGE_URI)

    namespace = campaign_root(DEFAULT_OWNER)
    root = study_root(DEFAULT_OWNER)
    parser.add_argument(
        "--dataset-tar-s3",
        default=f"{namespace}/data/lerobot/8ce6785b6f57ef3e34d6ca55fd0e3f30be8e19255869886838727635ffc0aa29.tar",
    )
    parser.add_argument(
        "--norm-stats-s3",
        default=f"{namespace}/assets/norm_stats/3ba87639c650f13f1405a032bb025e6b2f2010de3f4404883cedd20fe6542b76.json",
    )
    parser.add_argument(
        "--init-ckpt-tar-s3",
        default=f"{namespace}/init/1cfbc327805272daf2d1512faaeaef733edc2ac4b2873f41f471d6896a5d4211.tar",
    )
    parser.add_argument(
        "--tokenizer-s3",
        default=f"{root}/artifacts/tokenizers/paligemma/"
        "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6.model",
    )
    parser.add_argument(
        "--omega-features-tar-s3",
        default=f"{namespace}/omega/features/a19c452facc49bc76e0edea88e63d1651ab895e34adb76551ed42e02ccb51180.tar",
    )
    parser.add_argument(
        "--omega-encoder-s3",
        default=f"{namespace}/omega/encoder/09a1107d486ae6bfe3112e4858c3a9101e8a934297b21b8fbb13cb3118acc483.pt",
    )
    add_guardrail_arguments(parser, default_max_run_seconds=MAX_RUN_SECONDS)
    # add_guardrail_arguments derives its default priority from the runtime; this campaign pins it.
    parser.set_defaults(priority=DEFAULT_PRIORITY)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.queue = resolve_queue(args.queue)
    validate_and_confirm(args)  # approval gate + shared infrastructure contract
    enforce_campaign_caps(args)  # campaign-specific caps on top
    source_dir = resolve_source_dir(args.source_dir)
    plan = build_plan(args, source_dir)

    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    # SageMaker job names accept only [A-Za-z0-9-]; arm ids carry underscores.
    arm_slug = args.arm.replace("_", "-")
    owner_slug = args.user.split(".")[0].replace("_", "-")
    job_name = (f"{owner_slug}-rcerebra-{arm_slug}-{plan['spec_sha256'][:8]}-{stamp}")[:63].rstrip("-")

    print(
        f"arm={args.arm} config={plan['spec']['config_name']} run_id={plan['run_id']}\n"
        f"  job_name={job_name}\n"
        f"  image={args.image_uri}\n"
        f"  wsmv2={plan['archives']['wsmv2']['uri']}\n"
        f"  openpi={plan['archives']['openpi']['uri']}\n"
        f"  dataset={args.dataset_tar_s3}\n"
        f"  checkpoint={plan['checkpoint_uri']}\n"
        f"  manifest={plan['manifest_uri']}\n"
        f"  queue={args.queue} instance={plan['instance_type']} "
        f"training_plan={plan['training_plan_arn']}\n"
        f"  priority={args.priority} max_run={args.max_run_seconds}s dry={args.dry_run}\n"
        f"  entry env cross-check OK ({len(plan['checked_env_keys'])} keys read by {ENTRY}): "
        f"{plan['checked_env_keys']}"
    )
    print("--- sealed spec ---")
    print(json.dumps(plan["spec"], sort_keys=True, indent=2))
    print(f"--- spec_sha256 = {plan['spec_sha256']} -> run_id = {plan['run_id']} ---")
    print("--- environment ---")
    print(json.dumps(plan["environment"], sort_keys=True, indent=2))

    if args.dry_run:
        print("  [DRY RUN: archives built locally; nothing uploaded, nothing submitted]")
        for component, archive in plan["archives"].items():
            print(
                f"    {component}: sha256={archive['sha256']} bytes={archive['bytes']} "
                f"-> {archive['uri']} (NOT uploaded)"
            )
        print("  SUBMISSION READY only after explicit approval and --confirm-submit")
        return

    for component, archive in plan["archives"].items():
        publish_archive_once(pathlib.Path(archive["path"]), archive["uri"])

    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=args.image_uri,
        instance_type=plan["instance_type"],
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": args.owner_email},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.campaign", "Value": "robocerebra_h12"},
            {"Key": "wsm.arm", "Value": args.arm},
            {"Key": "wsm.run_id", "Value": plan["run_id"]},
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
        staged_source_files={STAGED_MANIFEST_NAME: plan["manifest_json"] + "\n"},
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
