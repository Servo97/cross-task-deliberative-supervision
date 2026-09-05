#!/usr/bin/env python3
"""Approval-gated launcher for the RoboMME stage job (tap / omega / parity).

Thin by design, exactly like `submit_robocerebra_stage.py`: queues, priority caps, role, the
sanitized source bundle and the SCP tags are all delegated to `launch_guardrails`. This file only
decides what the node is told to do.

Phases (see `robomme_stage_entry.sh`):

  tap      1,600 episodes -> the `wsm_pooled` pooled-token store Stage E consumes as its FOURTH
           domain (cross-domain grid: stride 8 + final frame). The tap is an INPUT to Stage E, not
           an output, so it does NOT wait for labels.
  tapserve the SAME tap on the SERVE-ALIGNED grid -> the POLICY omega store. Not optional: a serve
           frame is `exec_start_idx + 16k` and 16 == 0 (mod 8), so on the 81.4 % of demo episodes
           whose `exec_start_idx % 8 != 0` NOT ONE live frame lands on the stride-8 grid and an
           online producer would emit zero live omegas (measured over all 900 demo episodes:
           only 167 are aligned). 67,491 frames vs the cross-domain grid's 98,215.
  omega   re-export omega from an existing Stage-E checkpoint (the normal path exports omega
          inside the Stage-E job itself via `train_stage_e --export-omega`).
  parity  the D7 expert-replay oracle. GATES the M1/M2/M3 policy arms: the serve-side incremental
          producer must reproduce the shipped omega frame-exactly.

`--dry-run` is fully offline apart from building the source archive: no AWS SDK, no upload, no
submit. Submission additionally requires prior explicit approval and `--confirm-submit`.

ONE PREREQUISITE THE ROBOCEREBRA LAUNCHER DID NOT HAVE. The tap builds its policy from the
RoboCasa config `pi05_rc_mg60_bal33`, which lives in `internal_training/robocasa/
wsm_robocasa_configs.py` and is NOT in the openpi fork (the fork carries a 81-line variant that
defines no mg60 config — verified 2026-09-02, shas 4609f0fa vs 42dff5df). So a small
content-addressed tarball of that one file must exist in S3 before the tap can run. Build and
upload it with `--print-configs-upload`; the sha is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from launch_guardrails import (  # noqa: E402
    OWNER_EMAIL,
    PROJECT_TAG,
    STUDY_OWNER,
    WSM_ROBOCASA_S3,
    add_guardrail_arguments,
    normalize_queue,
    submit_training_job,
    validate_and_confirm,
)

STUDY = "long_context_v1"
ENTRY = "robomme_stage_entry.sh"
DEFAULT_OWNER = STUDY_OWNER
#: Independent of --user on purpose: --user is the frozen S3 storage prefix every content address
#: in this study is minted under; the live submitting identity is different, and SCP p-ahpdy5vv
#: denies a submit tagged with the dead one.
DEFAULT_OWNER_EMAIL = OWNER_EMAIL

S3 = WSM_ROBOCASA_S3
STUDY_ROOT = f"{S3}/studies/{STUDY}"

#: The sealed all-16 RoboMME LeRobot corpus and its parent inventory. The inventory carries a
#: `source_sha256` for every one of the 1,600 parquets and `fleet/inventory.py::materialize`
#: verifies each object as it streams (CAMPAIGNS §W6) — this is why the tap does not `s3 sync`.
RMME_DATA_S3 = f"{S3}/datasets/robomme/v1/lerobot_all16"
RMME_DATA_INVENTORY_SHA = "e77968b4c72c7589d92c1e85b1c6f7bf81aa49dd74472fb88dcead4277b5dad2"
RMME_DATA_INVENTORY_S3 = f"{STUDY_ROOT}/manifests/inventories/data/{RMME_DATA_INVENTORY_SHA}.json"

#: The frozen tap backbone: the RoboCasa H300+MG pretrain every RoboMME arm initialises from
#: (`robomme_integration/launch.py::INIT_ROOT`). Using it means RoboMME adds NO new frozen network
#: to the Stage-E encoder — unlike RoboCerebra, which had to bring `pi05_libero`.
TAP_CKPT_S3 = f"{S3}/pretrain150k/pi05/mg60_bal33/run/149999"

POOL_SHA = "18c26a7d54d48058302d9dc0fc155a27da66cf35559e5104e954b93390532e30"
POOL_S3 = f"{STUDY_ROOT}/artifacts/workspace/pool/{POOL_SHA}.pt"

#: Deterministic tarball of internal_training/robocasa/wsm_robocasa_configs.py (tar --sort=name
#: --owner=0 --group=0 --numeric-owner --mtime='UTC 2020-01-01'), rebuildable by
#: --print-configs-upload. Content-addressed so the tap's config definition cannot drift.
WSM_CONFIGS_SHA = "026255fa3593dff3acd9559edf9013cb73f722f457e5803cd6e168829ff893a3"
WSM_CONFIGS_S3 = f"{STUDY_ROOT}/artifacts/configs/{WSM_CONFIGS_SHA}.tgz"
WSM_CONFIGS_SOURCE = pathlib.Path("~/Research/TRI/internal_training").expanduser()
WSM_CONFIGS_MEMBER = "robocasa/wsm_robocasa_configs.py"

QUEUE_INSTANCE_TYPES = {
    "fss-tri-cam-robotics-p5-48xlarge-us-west-2": "ml.p5.48xlarge",
    "fss-tri-cam-robotics-p5e-48xlarge-us-west-2-training-plan": "ml.p5e.48xlarge",
}
INSTANCE_TYPE = "ml.p5.48xlarge"
GPUS = 8
RETRY = {"attempts": 1}

#: max_run, sized from a MEASURED rate rather than assumed.
#: Frames (measured over all 1,600 episodes): cross-domain grid 98,215 (36,907 demo + 61,308 live);
#: serve-aligned grid 67,491 (36,907 demo + 30,584 live); both grids together 165,706.
#: At the RoboCerebra tap's measured 5.97 frames/s/GPU (pad_batch 16) over 8 GPUs that is 3,468 s,
#: derated 1.5x for 3-view/192-token vs 2-view/128-token -> 5,202 s; x2.5 headroom = 13,005 s;
#: + 3,600 s startup (uv sync, a 12 GB checkpoint, a ~129 GB verified dataset materialization).
TAP_ESTIMATE_S = 5202
TAP_ONLY_ESTIMATE_S = 3084
DEFAULT_MAX_RUN = 16_800


def _configs_tarball(destination: pathlib.Path) -> str:
    """Rebuild the deterministic configs tarball and return its sha256."""
    source = WSM_CONFIGS_SOURCE / WSM_CONFIGS_MEMBER
    if not source.is_file():
        raise SystemExit(f"missing {source}; the tap config cannot be content-addressed")
    subprocess.run(
        [
            "tar",
            "--sort=name",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--mtime=UTC 2020-01-01",
            "-czf",
            str(destination),
            "-C",
            str(WSM_CONFIGS_SOURCE),
            WSM_CONFIGS_MEMBER,
        ],
        check=True,
    )
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_environment(args) -> dict:
    env = {
        "RMME_PHASES": args.phases,
        "OPENPI_FORK_S3": args.openpi_source_s3,
        "RMME_DATA_S3": args.rmme_data_s3,
        "RMME_DATA_INVENTORY_S3": args.rmme_data_inventory_s3,
        "RMME_DATA_INVENTORY_SHA256": args.rmme_data_inventory_sha256,
        "TAP_CKPT_S3": args.tap_ckpt_s3,
        "POOL_CKPT_S3": args.pool_ckpt_s3,
        "POOL_CKPT_SHA256": args.pool_ckpt_sha256,
        "WSM_CONFIGS_S3": args.wsm_configs_s3,
        "WSM_CONFIGS_SHA256": args.wsm_configs_sha256,
        "OUTPUT_S3": args.output_s3,
        "NUM_GPUS": str(GPUS),
        # The p5 guardrail checks the VALUE, not the presence: plain p5 needs "1"; only the
        # training-plan queue takes "0".
        "SM_USE_RESERVED_CAPACITY": "0" if "training-plan" in args.queue else "1",
        "SAGEMAKER_PROGRAM": ENTRY,
    }
    if "omega" in args.phases or "parity" in args.phases:
        if not args.encoder_ckpt_uri:
            raise SystemExit("--encoder-ckpt-uri is required for the omega/parity phases")
        if args.encoder_ckpt_uri.endswith("encoder_best.pt"):
            raise SystemExit(
                "encoder_best.pt is the BEST-EVAL step, not the checkpoint that exported the omega "
                "store; parity refuses a step mismatch (h14 §41.2). Pass encoder.pt."
            )
        env["ENCODER_CKPT_URI"] = args.encoder_ckpt_uri
        env["OMEGA_LANG_MODE"] = args.lang_mode
        env["PARITY_DEMOS"] = str(args.parity_demos)
        if args.stage_e_labels_dir:
            env["STAGE_E_LABELS_DIR"] = args.stage_e_labels_dir
        if args.omega_root_override:
            env["OMEGA_ROOT_OVERRIDE"] = args.omega_root_override
    return env


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_guardrail_arguments(ap, default_max_run_seconds=DEFAULT_MAX_RUN)
    ap.add_argument("--user", default=DEFAULT_OWNER, help="S3 storage owner prefix")
    ap.add_argument(
        "--owner-email",
        default=DEFAULT_OWNER_EMAIL,
        help="value of the required tri.owner.email SCP tag; independent of --user",
    )
    ap.add_argument("--phases", default="tap", help="comma list of tap,omega,parity")
    ap.add_argument("--source-dir", default=None)
    # Not `required=True`: --print-configs-upload is an offline helper that needs neither.
    ap.add_argument("--openpi-source-s3", default="")
    ap.add_argument("--image-uri", default="")
    ap.add_argument("--rmme-data-s3", default=RMME_DATA_S3)
    ap.add_argument("--rmme-data-inventory-s3", default=RMME_DATA_INVENTORY_S3)
    ap.add_argument("--rmme-data-inventory-sha256", default=RMME_DATA_INVENTORY_SHA)
    ap.add_argument("--tap-ckpt-s3", default=TAP_CKPT_S3)
    ap.add_argument("--pool-ckpt-s3", default=POOL_S3)
    ap.add_argument("--pool-ckpt-sha256", default=POOL_SHA)
    ap.add_argument("--wsm-configs-s3", default=WSM_CONFIGS_S3)
    ap.add_argument("--wsm-configs-sha256", default=WSM_CONFIGS_SHA)
    ap.add_argument(
        "--print-configs-upload",
        action="store_true",
        help="rebuild the deterministic tap-config tarball, print its sha and the "
        "exact aws s3 cp line, and exit without touching AWS",
    )
    ap.add_argument("--output-s3", default=f"{STUDY_ROOT}/robomme/stage")
    ap.add_argument(
        "--encoder-ckpt-uri",
        default="",
        help="the FINAL Stage-E checkpoint (encoder.pt) that exported the omega store "
        "-- NOT encoder_best.pt, which is a different model whenever best != "
        "final (h14 §41.2). Parity refuses a mismatch.",
    )
    ap.add_argument("--stage-e-labels-dir", default="")
    ap.add_argument("--omega-root-override", default="")
    ap.add_argument(
        "--lang-mode",
        default="stored",
        choices=("stored", "demo", "taskmean", "per_frame", "task_line"),
        help="the conditioning contract the omega store and the D7 gate use. `stored` "
        "is the gate mode (h14 §39.3); `taskmean` fails a CORRECT encoder and is "
        "a diagnostic only",
    )
    ap.add_argument("--parity-demos", type=int, default=20)
    ap.add_argument("--instance-type", default="", help="defaults to the family bound to --queue")
    return ap


def main() -> None:
    args = parser().parse_args()

    if args.print_configs_upload:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "wsm_configs.tgz"
            sha = _configs_tarball(path)
            size = path.stat().st_size
        print(
            json.dumps(
                {
                    "member": WSM_CONFIGS_MEMBER,
                    "sha256": sha,
                    "bytes": size,
                    "matches_pinned": sha == WSM_CONFIGS_SHA,
                    "target": f"{STUDY_ROOT}/artifacts/configs/{sha}.tgz",
                },
                indent=1,
            )
        )
        print(
            "\n# rebuild + upload (coordinator action; this launcher never writes to S3):\n"
            f"tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2020-01-01' \\\n"
            f"    -czf /tmp/wsm_configs.tgz -C {WSM_CONFIGS_SOURCE} {WSM_CONFIGS_MEMBER}\n"
            f"aws s3 cp /tmp/wsm_configs.tgz {STUDY_ROOT}/artifacts/configs/{sha}.tgz"
        )
        return

    missing = [f"--{n.replace('_', '-')}" for n in ("openpi_source_s3", "image_uri") if not getattr(args, n)]
    if missing:
        raise SystemExit(f"{', '.join(missing)} are required to dry-run or submit")
    args.queue = normalize_queue(args.queue)
    inst = args.instance_type or QUEUE_INSTANCE_TYPES.get(args.queue, INSTANCE_TYPE)
    for phase in args.phases.split(","):
        if phase.strip() not in ("tap", "tapserve", "omega", "parity"):
            raise SystemExit(f"unknown phase {phase!r}")
    validate_and_confirm(args)

    env = build_environment(args)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    phase_tag = args.phases.replace(",", "-")
    job_name = f"rmme-stage-{phase_tag}-{stamp}"[:63].rstrip("-")

    print(
        f"phases={args.phases} lang_mode={args.lang_mode}\n"
        f"  wsmv2 = the sanitized source bundle (no separate tarball)\n"
        f"  data={args.rmme_data_s3}\n  inventory={args.rmme_data_inventory_s3}\n"
        f"  tap_ckpt={args.tap_ckpt_s3}\n  pool={args.pool_ckpt_s3}\n"
        f"  configs={args.wsm_configs_s3}\n  output={args.output_s3}\n"
        f"  queue={args.queue} instance={inst} priority={args.priority} "
        f"max_run={args.max_run_seconds}s dry={args.dry_run}\n"
        f"  tap estimate {TAP_ESTIMATE_S}s for both grids (98,215 + 67,491 frames) "
        f"x2.5 + startup; single-grid {TAP_ONLY_ESTIMATE_S}s"
    )
    if args.dry_run:
        print("  [DRY RUN: offline; nothing submitted]")
        print(json.dumps(env, indent=1, sort_keys=True))
        print("  SUBMISSION READY only after explicit approval and --confirm-submit")
        return

    source_dir = args.source_dir or str(pathlib.Path(__file__).resolve().parents[2])
    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=env,
        image_uri=args.image_uri,
        instance_type=inst,
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": args.owner_email},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.campaign", "Value": "robomme_h14"},
            {"Key": "wsm.phases", "Value": args.phases},
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
    )
    print(json.dumps({"job_name": job_name, "phases": args.phases, "result": str(result)}, indent=1))


if __name__ == "__main__":
    main()
