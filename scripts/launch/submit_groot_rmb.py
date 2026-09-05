#!/usr/bin/env python3
"""Approval-gated GR00T N1.7 ReMemBench post-training launcher (backbone-generality arm).

WHY A SEPARATE LAUNCHER. ``submit_pi_stage_s.py`` is the sealed pi0.5 Stage-S plan machinery
(openpi fork pin, PaliGemma tokenizer, omega feature manifests, producer/completion claims) — none
of which exists on the GR00T side. ``submit_finetunes.py`` is the historical pre-guardrails GR00T
launcher: no dataset profile, no plan-ARN pinning, a hardcoded account-124 dataset root, and it
predates the ``launch_guardrails.submit_training_job`` contract. This launcher is the narrow middle:
the GR00T finetune entry, the ReMemBench dataset profile, and the guardrailed submit path with the
p5e training-plan ARN pinned automatically.

WHAT IT SEALS PER RUN (all recorded in the deterministic run id):
  * the config yaml BYTES (sha256), so a silent recipe edit yields a different run id,
  * the init checkpoint URI  — THE SAME phase-1 pretrain every GR00T study arm started from,
  * the dataset root + profile, the task filter, and the step budget,
  * the wsmv2 source archive digest and the ECR image digest.

  # dry run (fully offline, prints the sealed plan + env):
  ~/Research/TRI/internal_training/.venv/bin/python scripts/launch/submit_groot_rmb.py \
      --cell rmb-base --dry-run
  # real submit (requires prior user approval):
  ... --cell rmb-base --confirm-submit
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

from launch_guardrails import (
    DEFAULT_RESULTS_BUCKET,
    DEXJOCO_IMAGE_REPO,
    STUDY_OWNER,
    TRAINING_PLAN_QUEUE,
    WSM_ROBOCASA_S3,
    add_guardrail_arguments,
    submit_training_job,
    training_plan_arn,
    validate_and_confirm,
)

ENTRY = "robocasa_groot_finetune_entry.sh"
#: The COMBINED MECHANISM CANARY entry. It is a WRAPPER around ENTRY, not a copy: it runs that
#: exact script once per mechanism segment with a short step budget, so what the canary validates
#: is literally the code path the real submits take (uv/numpy/mujoco resolution, gated-backbone
#: cache, ReMemBench sync + fail-fast, init sync, omega staging, torchrun dispatch, and the HF save
#: of a PATCHED action head). See its header for why the segments run independently.
CANARY_ENTRY = "robocasa_groot_rmb_canary_combined_entry.sh"
STUDY = "long_context_v1"
DEFAULT_OWNER = STUDY_OWNER
BACKBONE = "groot_n17"

#: Queue -> instance family. Each queue's service environment is bound to ONE family, so the
#: instance type is a function of --queue, not a free parameter.
QUEUE_INSTANCE_TYPES = {
    "fss-tri-cam-robotics-p5-48xlarge-us-west-2": "ml.p5.48xlarge",
    TRAINING_PLAN_QUEUE: "ml.p5e.48xlarge",
}

#: THE init checkpoint. Provenance: phase-1 RoboCasa365 pretrain, balanced arm, step 150000 — the
#: exact checkpoint `target_ft/groot_bal33` (the held-out-reset campaign's groot base-FT) started
#: from. Originally written to account 124224456861; mirrored into account 141 on 2026-08-07
#: (weights + processor/statistics/experiment_cfg; optimizer.pt / rng_state_* / scheduler.pt /
#: trainer_state.json excluded, exactly as the entry's own INIT sync excludes them) because study
#: storage == execution == 141 since 2026-07-22 and the p5/p5e node role has no cross-account read
#: on the 124 bucket.
INIT_S3 = f"{WSM_ROBOCASA_S3}/pretrain150k/groot/mg60_bal33/groot-mg60-bal33/checkpoint-150000"
REMEMBENCH_DATA_S3 = f"{WSM_ROBOCASA_S3}/datasets/remembench_v02"
#: FULL HF cache for the GATED nvidia/Cosmos-Reason2-2B backbone (4.6 GB unpacked). The node
#: unpacks it into $HF_HOME/hub and resolves with transformers ``local_files_only=True``, so
#: training needs no HF_TOKEN — which matters because launch_guardrails.prepared_source_bundle
#: REFUSES a plaintext HF_TOKEN in the environment and the submitting role has no
#: secretsmanager:GetSecretValue.
#:
#: A PROCESSOR-ONLY cache (11 MB) was tried first and FAILED at run
#: groot-rmb-base-8beebf5d25f22990: the weights are needed after all. GR00T constructs the Qwen3VL
#: backbone with its own ``Qwen3VLForConditionalGeneration.from_pretrained(model_name)``
#: (gr00t/model/modules/qwen3_backbone.py:80) BEFORE the study checkpoint is loaded over it, so a
#: cache without model.safetensors yields ``checkpoint_files[0] is None`` ->
#: ``AttributeError: 'NoneType' object has no attribute 'endswith'`` in transformers
#: modeling_utils.py:4924. Do not slim this asset back down.
BACKBONE_PROCESSOR_S3 = f"{WSM_ROBOCASA_S3}/assets/backbones/cosmos_reason2_2b_full_hfcache.tgz"
#: Per-task demo counts of the sealed ReMemBench train split — the SAME map the pi Stage-S profile
#: validates --single-task against, so a typo can never reach the node.
DEMOS_PER_TASK_MAP = "scripts/configs/data/remembench_v02_train13_demos_per_task.json"

#: The study's shared IMAGE-DERIVED omega cache, content-addressed by encoder-chain id. It is
#: MODEL-AGNOSTIC — the identical cache the pi ReMemBench mechanism arms consumed — which is what
#: lets a groot-vs-pi mechanism delta be read as a BACKBONE effect. Regenerating a groot-specific
#: cache would introduce a second difference and destroy that reading.
#: Verified 2026-08-07: 323 w.npz, layout <task>/demo_%06d/w.npz (MemHeatPot 40, fruitRF 20).
OMEGA_ENCODER_ID = "ba39e9088539c24ec078051c1adf311198853bc59cdc79b3f481d29b70f78efa"
OMEGA_EXPECTED_FILES = 323

#: The GR00T runtime image in account 141, pinned BY DIGEST (a moving `:latest` would silently
#: change the scientific artifact).
IMAGE_URI = f"{DEXJOCO_IMAGE_REPO}@sha256:798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"

#: The node-entry environment that reshapes the entry's dataset validators for ReMemBench. This is
#: the GR00T mirror of submit_pi_stage_s.DATASET_PROFILES["remembench_v02_train13"]["entry_env"].
#: Object keys under the dataset root start with "train/", so the task dirs land one level below
#: the materialize destination — hence TARGET_ROOT_SUBDIR = <materialize>/train.
REMEMBENCH_ENTRY_ENV = {
    "TARGET_DATA_S3": REMEMBENCH_DATA_S3,
    "TARGET_MATERIALIZE_SUBDIR": "v1.0/target",
    "TARGET_ROOT_SUBDIR": "v1.0/target/train",
    "TARGET_EXPECTED_TASKS": "13",
    "TARGET_TASK_DIR_GLOBS": "*/*/lerobot",
    "WSM_SOUP_KIND": "remembench13",
}

#: The three PHASE-1 cells. Every cell shares the init, the recipe surface and the dataset; they
#: differ only in soup filter and step budget (see the yaml headers for the scaling rationale).
CELLS: dict[str, dict] = {
    "rmb-base": {
        "config": "scripts/configs/train/groot17_rmb_base_finetune.yaml",
        "single_task": None,
        "steps": 15000,
        "save_interval": 5000,
    },
    "rmb1t-heatpot": {
        "config": "scripts/configs/train/groot17_rmb1t_heatpot_base_finetune.yaml",
        "single_task": "MemHeatPot",
        "steps": 4000,
        "save_interval": 2000,
    },
    "rmb1t-fruitRF": {
        "config": "scripts/configs/train/groot17_rmb1t_fruitRF_base_finetune.yaml",
        "single_task": "MemFruitInSinkRightFar",
        "steps": 4000,
        "save_interval": 2000,
    },
}

#: PHASE-2 mechanism arms. Each is a PHASE-1 cell with exactly one mechanism added: same init, same
#: dataset, same step budget, same recipe surface. `arm` and `mechanism_env` are the only additions
#: to the sealed spec, so a base-vs-arm difference can only be the mechanism.
#: `dnw8`  = gated DeltaNet workspace conditioner, K=8, injected additively on the DiT `temb`.
#: `jw01k16` = model-free JEPA aux, lambda 0.1, k=16, train-time only (never in the inference graph).
MECHANISMS: dict[str, dict] = {
    "dnw8": {
        "arm": "s1-deltanet",
        "train_script": "vla_training/train/train_base/finetune_groot_17_with_wsm_deltanet.py",
        "config_suffix": "dnw8",
        "env": {
            "WSM_W_DIM": "512",
            "WSM_DN_WINDOW": "8",
            "WSM_DN_HEADS": "2",
            "WSM_DN_HEAD_DIM": "256",
            "WSM_DN_GATE_INIT": "1e-3",
            "WSM_DN_HISTORY_DROPOUT": "0.0",
        },
    },
    "jw01k16": {
        "arm": "s3-jepa",
        "train_script": "vla_training/train/train_base/finetune_groot_17_with_wsm_jepa.py",
        "config_suffix": "jw01k16",
        "env": {
            "WSM_W_DIM": "512",
            "WSM_JEPA_WEIGHT": "0.1",
            "WSM_SIGREG_WEIGHT": "0.05",
            "WSM_JEPA_NUM_FUTURES": "16",
            "WSM_JEPA_DIRECT": "0",
        },
    },
    # `ttt` = RoboTTT fast weights (REDUCED FORM), the one STATEFUL arm. It is also the only
    # mechanism that reads NO omega: its recurrent state is learned from the policy's own
    # transitions, so `needs_omega` is False and the entry's POLICY_FEATS_S3 block stays unset —
    # shipping the cache anyway would imply a dependency that does not exist.
    "ttt": {
        "arm": "robottt-fast",
        "train_script": "vla_training/train/train_base/finetune_groot_17_with_robottt.py",
        "config_suffix": "ttt",
        "needs_omega": False,
        "env": {
            "WSM_SEQ_WINDOW_LEN": "8",
            "WSM_SEQ_CHUNK_STRIDE": "8",
            "WSM_TTT_TBPTT_SEGMENT": "8",
            "WSM_TTT_TOKEN_DIM": "256",
            "WSM_TTT_FAST_HIDDEN": "128",
            "WSM_TTT_NUM_REGISTERS": "16",
            "WSM_TTT_INNER_LR": "0.1",
            "WSM_TTT_LEARN_INNER_LR": "1",
            "WSM_TTT_GATE_INIT": "1e-3",
        },
    },
}

for _base_cell, _base in list(CELLS.items()):
    for _mech_name, _mech in MECHANISMS.items():
        CELLS[f"{_base_cell}-{_mech_name}"] = {
            **_base,
            "config": _base["config"].replace("_base_finetune.yaml", f"_{_mech['config_suffix']}_finetune.yaml"),
            "mechanism": _mech_name,
        }


def mechanism_config(base_config: str, mechanism: str) -> str:
    """The one mechanical derivation base cell -> mechanism arm config (used above and by the canary)."""
    return base_config.replace("_base_finetune.yaml", f"_{MECHANISMS[mechanism]['config_suffix']}_finetune.yaml")


#: THE GATE. One job on the smallest cell (MemHeatPot, 40 demos) runs BOTH mechanism train scripts
#: back to back for CANARY_STEPS steps each, against the real ReMemBench data and the real omega
#: cache. It exists because the local smokes cannot reach the node's environment: the robocasa pins
#: (numpy 2.2.5 / mujoco 3.3.1), numba, the sm_120 torch gap and the staged N1.7 weights are all
#: only exercised on a p5e. Nothing about it is a scientific artifact — its ONLY output is
#: PASS/FAIL, and the 6 real mechanism trains do not go out until it passes.
#:
#: It deliberately reuses the SEALED per-arm config bytes (not a canary-specific yaml) and overrides
#: only the step budget via env, so a canary pass is evidence about the configs that will actually
#: run. save_interval == steps so exactly one save happens: saving a patched action head through HF
#: is itself a failure mode a 15k-step run would only discover at the end.
CANARY_STEPS = 300
CANARY_CELL = "rmb1t-heatpot-canary"


def _canary_cell(mechanisms: tuple[str, ...]) -> dict:
    return {
        "base_config": CELLS["rmb1t-heatpot"]["config"],
        "config": None,
        "single_task": CELLS["rmb1t-heatpot"]["single_task"],
        "steps": CANARY_STEPS,
        "save_interval": CANARY_STEPS,
        "entry": CANARY_ENTRY,
        "canary_mechanisms": mechanisms,
    }


CELLS[CANARY_CELL] = _canary_cell(("dnw8", "jw01k16"))
#: RoboTTT gets its OWN canary rather than a third segment on the one above, for two reasons: its
#: source archive necessarily differs (the port landed after the dnw8/jw01k16 archive was pinned,
#: and one archive per submission set is the rule that stopped the drift in PHASE 1 §6), and it is
#: the only arm that changes the DATALOADER and the COLLATOR rather than just the head — so its
#: failure modes are disjoint and there is nothing to learn from bundling them.
CELLS["rmb1t-heatpot-ttt-canary"] = _canary_cell(("ttt",))

#: The wsmv2 import surface the GR00T entry needs. Deliberately NOT the whole repo: the entry only
#: untars and runs finetune_groot_17.py, and a whole-repo archive would fold papers/, results/,
#: overleaf/ and local checkpoint trees into the scientific source identity.
ARCHIVE_PATHS = ("utils", "vla_training", "workspace_models", "scripts", "pyproject.toml")
_ARCHIVE_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", ".claude"}
_ARCHIVE_EXCLUDE_SUFFIX = (".pyc", ".pyo")

_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RETRY = {"attempts": 1}  # a retry is a new launch and returns through the approval gate


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def study_root(owner: str) -> str:
    if not _OWNER.fullmatch(owner):
        raise SystemExit(f"invalid --user storage owner {owner!r}")
    return f"s3://{DEFAULT_RESULTS_BUCKET}/{owner}/wsm_robocasa/studies/{STUDY}"


def resolve_source_dir(arg: str | None, entry: str = ENTRY) -> pathlib.Path:
    """internal_training dir (ships the GR00T FT entry). Default: <repo-parent>/internal_training."""
    path = pathlib.Path(arg or os.environ.get("WSM_SOURCE_DIR") or repo_root().parent / "internal_training")
    # The base ENTRY is required even for the canary: the canary entry *invokes* it on the node.
    for required in {ENTRY, entry}:
        if not (path / required).exists():
            raise SystemExit(f"source-dir {path} missing {required} — pass --source-dir <internal_training>")
    return path


def build_repo_archive(destination_dir: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Deterministic, content-addressed tarball of the wsmv2 import surface.

    Byte-for-byte reproducible (sorted entries, zeroed mtime/uid/gid/uname, fixed modes), so the
    digest identifies the CODE and nothing about when or where it was packed.
    """
    root = repo_root()
    with tempfile.TemporaryDirectory(prefix="groot-rmb-archive-") as tmp:
        staged = pathlib.Path(tmp) / "source"
        staged.mkdir()
        for name in ARCHIVE_PATHS:
            source = root / name
            if not source.exists():
                raise SystemExit(f"archive path missing from repo: {source}")
            if source.is_dir():
                shutil.copytree(
                    source,
                    staged / name,
                    symlinks=True,
                    ignore=lambda _d, names: [
                        n for n in names if n in _ARCHIVE_EXCLUDE_DIRS or n.endswith(_ARCHIVE_EXCLUDE_SUFFIX)
                    ],
                )
            else:
                shutil.copy2(source, staged / name)

        incomplete = destination_dir / ".wsmv2.tgz.incomplete"
        incomplete.unlink(missing_ok=True)
        with incomplete.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for path in sorted(staged.rglob("*"), key=lambda item: item.relative_to(staged).as_posix()):
                        relative = path.relative_to(staged).as_posix()
                        info = tarfile.TarInfo(relative)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        if path.is_symlink():
                            info.type, info.mode = tarfile.SYMTYPE, 0o777
                            info.linkname = os.readlink(path)
                            archive.addfile(info)
                        elif path.is_dir():
                            info.type, info.mode = tarfile.DIRTYPE, 0o755
                            archive.addfile(info)
                        elif path.is_file():
                            info.type = tarfile.REGTYPE
                            info.mode = 0o755 if stat.S_IMODE(path.lstat().st_mode) & 0o111 else 0o644
                            info.size = path.stat().st_size
                            with path.open("rb") as stream:
                                archive.addfile(info, stream)
                        else:
                            raise SystemExit(f"unsupported source entry type: {path}")
        digest = hashlib.sha256(incomplete.read_bytes()).hexdigest()
        final = destination_dir / f"{digest}.tgz"
        if final.exists():
            incomplete.unlink()
        else:
            incomplete.replace(final)
    return final, digest


def s3_object_exists(uri: str) -> bool:
    bucket, _, key = uri[len("s3://") :].partition("/")
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def publish_archive(local: pathlib.Path, uri: str) -> None:
    """Create-once publish: never overwrite an existing content-addressed object."""
    if s3_object_exists(uri):
        print(f"    archive already published (create-once): {uri}")
        return
    subprocess.run(["aws", "s3", "cp", str(local), uri, "--only-show-errors"], check=True)
    print(f"    archive published: {uri}")


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_plan(args) -> dict:
    cell = CELLS[args.cell]
    root = study_root(args.user)
    canary_mechs = cell.get("canary_mechanisms")

    def sha_of(relative: str) -> str:
        path = repo_root() / relative
        if not path.exists():
            raise SystemExit(f"config missing: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if canary_mechs:
        # One segment per mechanism, each pinned to the SEALED arm config it is gating.
        canary_segments = [
            {
                "name": m,
                "mechanism": m,
                "train_script": MECHANISMS[m]["train_script"],
                "config": mechanism_config(cell["base_config"], m),
                "config_sha256": sha_of(mechanism_config(cell["base_config"], m)),
            }
            for m in canary_mechs
        ]
        config_sha = None
    else:
        canary_segments = None
        config_sha = sha_of(cell["config"])

    single_task = cell["single_task"]
    if single_task is not None:
        demo_map = json.loads((repo_root() / DEMOS_PER_TASK_MAP).read_text())
        if single_task not in demo_map:
            raise SystemExit(f"--cell {args.cell} names task {single_task!r}, absent from {DEMOS_PER_TASK_MAP}")
        train_demos = demo_map[single_task]
    else:
        train_demos = sum(json.loads((repo_root() / DEMOS_PER_TASK_MAP).read_text()).values())

    steps = args.train_steps or cell["steps"]
    plan_arn = training_plan_arn(args.queue)
    instance_type = QUEUE_INSTANCE_TYPES.get(args.queue)
    if instance_type is None:
        raise SystemExit(f"no instance family registered for queue {args.queue}")

    archive_dir = pathlib.Path(args.archive_dir or (repo_root().parent / "_wsmv2_archives"))
    archive_dir.mkdir(parents=True, exist_ok=True)
    if args.pin_archive:
        # PIN an already-built archive instead of re-packing the working tree. The tree is SHARED
        # with other in-flight workstreams, so two submits minutes apart can otherwise seal
        # different source digests for cells that are supposed to differ only in soup and steps.
        # (Observed 2026-08-07: a concurrent edit to _groot_wsm_jepa_common.py — inert on the
        # baseline import path, but it moved the digest.) Pin once, submit the whole cell set.
        archive_sha = args.pin_archive
        archive_path = archive_dir / f"{archive_sha}.tgz"
        if not archive_path.exists():
            raise SystemExit(f"--pin-archive {archive_sha}: no local archive at {archive_path}")
        actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual != archive_sha:
            raise SystemExit(f"pinned archive digest mismatch: file is {actual}")
    else:
        archive_path, archive_sha = build_repo_archive(archive_dir)
    archive_uri = f"{root}/code/wsmv2/{archive_sha}.tgz"

    entry = cell.get("entry", ENTRY)
    spec = {
        "study": STUDY,
        "backbone": BACKBONE,
        "benchmark": "ReMemBench",
        "cell": args.cell,
        "arm": ("canary" if canary_mechs else MECHANISMS[cell["mechanism"]]["arm"] if cell.get("mechanism") else "s0"),
        "config": (
            {"canary_segments": canary_segments} if canary_mechs else {"path": cell["config"], "sha256": config_sha}
        ),
        "init_s3": INIT_S3,
        "dataset": {
            "profile": "remembench_v02_train13",
            "root_s3": REMEMBENCH_DATA_S3,
            "tasks": 13,
            "demos_per_task_map": DEMOS_PER_TASK_MAP,
        },
        "training_task_filter": {"tasks": [single_task]} if single_task else None,
        "train_demos": train_demos,
        "train_steps": steps,
        "save_interval": cell["save_interval"],
        "source": {"wsmv2_uri": archive_uri, "wsmv2_sha256": archive_sha},
        "image": {"uri": IMAGE_URI},
        "backbone_processor_s3": BACKBONE_PROCESSOR_S3,
        "runtime": {
            "queue": args.queue,
            "instance_type": instance_type,
            "training_plan_arn": plan_arn,
            "priority": args.priority,
            "max_run_seconds": args.max_run_seconds,
        },
    }
    # ---- KEYS THAT EXIST ONLY WHEN THEY MEAN SOMETHING ----------------------------------------
    # A key added to `spec` with a NULL value still changes the canonical JSON and therefore the
    # run id. PHASE 2 added `"mechanism": None` unconditionally, which silently re-derived all
    # three PHASE-1 baseline run ids away from the ones actually submitted and recorded (rmb-base
    # 1380447392902905 / heatpot 31abd9dbb4a69db9 / fruitRF cae5ded0126b1032) — the launcher could
    # no longer reproduce the identity of its own live runs, and `output_s3` is derived from it.
    # Both this key and `entry` are therefore ADDED ONLY when non-default, which restores the
    # baselines exactly and leaves the mechanism cells (where the key is non-null either way)
    # unchanged. Regression: `--cell rmb-base --pin-archive 1cbc00a8… --dry-run` must print
    # groot-rmb-base-1380447392902905.
    if cell.get("mechanism"):
        # Mechanism provenance goes INTO the sealed spec, so a mechanism change yields a new run id
        # and can never be confused with a rerun of the baseline cell.
        mech_spec = MECHANISMS[cell["mechanism"]]
        spec["mechanism"] = {"name": cell["mechanism"], **mech_spec["env"]}
        if mech_spec.get("needs_omega", True):
            # Only arms that actually consume the shared cache record it. Sealing an omega id into
            # an arm that never reads one would claim a dependency the run does not have.
            spec["mechanism"]["omega_encoder_id"] = OMEGA_ENCODER_ID
            spec["mechanism"]["omega_files"] = OMEGA_EXPECTED_FILES
    if entry != ENTRY:
        spec["entry"] = entry
    spec_sha = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    run_id = f"groot-{args.cell}-{spec_sha[:16]}"
    output_s3 = f"{root}/checkpoints/{BACKBONE}/{args.cell}/{run_id}"

    entry = cell.get("entry", ENTRY)
    environment = {
        # An explicitly pinned plan REPLACES the implicit reserved-capacity request; the pair is
        # re-checked inside submit_training_job, so a mismatch fails before any AWS write.
        "SM_USE_RESERVED_CAPACITY": "0" if plan_arn else "1",
        "SAGEMAKER_PROGRAM": entry,
        "INIT_S3": INIT_S3,
        "WSM_REPO_S3": archive_uri,
        "BACKBONE_PROCESSOR_S3": BACKBONE_PROCESSOR_S3,
        "WSM_MAX_STEPS": str(steps),
        "WSM_SAVE_INTERVAL": str(cell["save_interval"]),
        "OUTPUT_S3": output_s3,
        "GROOT_RMB_RUN_ID": run_id,
        "WANDB_PROJECT": "wsm-robocasa",
        "WANDB_RUN_GROUP": f"remembench-groot-{args.cell}",
        **REMEMBENCH_ENTRY_ENV,
    }
    needs_omega = any(MECHANISMS[m].get("needs_omega", True) for m in (canary_mechs or ()))
    if canary_mechs:
        # The canary entry sets TRAIN_SCRIPT / WSM_FT_CONFIG / WSM_MAX_STEPS / OUTPUT_S3 PER SEGMENT,
        # so a global WSM_FT_CONFIG here would be dead weight and a misleading record. What IS global
        # is both mechanisms' knobs: WSM_DN_* and WSM_JEPA_*/WSM_SIGREG_* are disjoint namespaces and
        # each train script reads only its own, so there is no cross-talk. WSM_W_DIM is shared and
        # must agree — asserted, because a silent disagreement would make one segment gate the wrong
        # geometry.
        merged: dict[str, str] = {}
        for m in canary_mechs:
            for key, value in MECHANISMS[m]["env"].items():
                if merged.get(key, value) != value:
                    raise SystemExit(f"canary mechanism env conflict on {key}: {merged[key]!r} vs {value!r}")
                merged[key] = value
        environment.update(merged)
        environment["CANARY_SEGMENTS"] = " ".join(
            f"{s['name']}:{s['train_script']}:{s['config']}" for s in canary_segments
        )
        environment["CANARY_STEPS"] = str(steps)
        environment["CANARY_SAVE_INTERVAL"] = str(cell["save_interval"])
        if needs_omega:
            environment["POLICY_FEATS_S3"] = f"{root}/caches/{OMEGA_ENCODER_ID}/omega"
            environment["OMEGA_EXPECTED_FILES"] = str(OMEGA_EXPECTED_FILES)
    else:
        environment["WSM_FT_CONFIG"] = cell["config"]
    if cell.get("mechanism"):
        mech = MECHANISMS[cell["mechanism"]]
        # TRAIN_SCRIPT + POLICY_FEATS_S3 are the entry's two ADDITIVE mechanism knobs; unset, the
        # entry reproduces the sealed PHASE-1 baseline byte-for-byte.
        environment["TRAIN_SCRIPT"] = mech["train_script"]
        if mech.get("needs_omega", True):
            environment["POLICY_FEATS_S3"] = f"{root}/caches/{OMEGA_ENCODER_ID}/omega"
            # Fail the node fast on a short/empty omega sync rather than discovering it per-sample —
            # or, worse, training as the baseline under this arm's name.
            environment["OMEGA_EXPECTED_FILES"] = str(OMEGA_EXPECTED_FILES)
        environment.update(mech["env"])
        environment["WANDB_RUN_GROUP"] = f"remembench-groot-{args.cell}"

    if single_task:
        # utils.soup._filter_soup_tasks restricts the materialized soup and RAISES on a name that
        # is not present, so a typo cannot silently train on the wrong set. The node still
        # materializes all 13 task dirs (the entry's fail-fast counts against WSM_TASKS).
        environment["WSM_TASKS"] = single_task

    oversized = {k: len(v.encode()) for k, v in environment.items() if len(v.encode()) > 512}
    if oversized:
        raise AssertionError(f"SageMaker environment value exceeds 512 bytes: {oversized}")

    return {
        "spec": spec,
        "spec_sha256": spec_sha,
        "run_id": run_id,
        "output_s3": output_s3,
        "environment": environment,
        "instance_type": instance_type,
        "archive_path": archive_path,
        "archive_uri": archive_uri,
        "entry": entry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=sorted(CELLS))
    parser.add_argument("--user", default=DEFAULT_OWNER, help="checkpoint S3 prefix owner")
    parser.add_argument("--source-dir", default=None, help="internal_training dir (FT entry)")
    parser.add_argument("--archive-dir", default=None, help="local dir for the content-addressed tarball")
    parser.add_argument("--train-steps", type=int, default=0, help="override the cell's step budget")
    parser.add_argument(
        "--pin-archive",
        default=None,
        metavar="SHA256",
        help=(
            "reuse an already-built local archive instead of re-packing the working tree. Use this "
            "to give every cell of a submission set ONE source identity when the tree is shared "
            "with concurrent workstreams."
        ),
    )
    add_guardrail_arguments(parser, default_max_run_seconds=86400)
    parser.set_defaults(queue=TRAINING_PLAN_QUEUE, priority=400, max_run_seconds=86400)
    args = parser.parse_args()

    validate_and_confirm(args)
    plan = build_plan(args)
    source_dir = resolve_source_dir(args.source_dir, plan["entry"])
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    job_name = f"{args.user.replace('.', '-')}-groot-rmb-{args.cell}-{stamp}"[:63].rstrip("-")

    print(json.dumps(plan["spec"], indent=2, sort_keys=True))
    print(
        f"\n=== {args.cell} -> {job_name}\n"
        f"    run_id={plan['run_id']}  spec_sha256={plan['spec_sha256']}\n"
        f"    entry={plan['entry']}  source_dir={source_dir}\n"
        f"    archive={plan['archive_path']} -> {plan['archive_uri']}\n"
        f"    out={plan['output_s3']}\n"
        f"    queue={args.queue} instance={plan['instance_type']} "
        f"priority={args.priority} max_run={args.max_run_seconds}s"
    )
    if args.dry_run:
        print("\n[DRY RUN: no archive upload, no AWS submission] environment:")
        print(json.dumps(plan["environment"], indent=2, sort_keys=True))
        return

    publish_archive(plan["archive_path"], plan["archive_uri"])
    result = submit_training_job(
        entry=plan["entry"],
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=IMAGE_URI,
        instance_type=plan["instance_type"],
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": "GROOT-DEXJOCO"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
        retry_config=RETRY,
        job_name=job_name,
        queue=args.queue,
        role=args.role,
        priority=args.priority,
        max_run_seconds=args.max_run_seconds,
        secrets_manager_arn=args.secrets_manager_arn,
        confirmed=args.confirm_submit,
    )
    arn = getattr(result[0], "job_arn", "?") if result else "?"
    print(f"    QUEUED ✓  run_id={plan['run_id']}  arn={arn}")


if __name__ == "__main__":
    main()
