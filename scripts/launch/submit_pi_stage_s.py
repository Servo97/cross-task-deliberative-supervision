#!/usr/bin/env python3
"""Approval-gated pi0.5 Stage-S post-training launcher (S0/S1/S2).

This launcher is deliberately narrower than the historical finetune launchers.  It accepts one
explicit arm per invocation, pins both source archives and the ECR image by SHA-256 digest, writes a
deterministic run manifest, and targets only account-141 cam-robotics.  ``--dry-run`` is fully
offline. Real submission requires prior user approval and ``--confirm-submit``. Every artifact
reference is content-addressed; node startup verifies the referenced bytes before optimizer work.

Checkpoint contracts: the default ``final-only`` retains one step and is what every sealed run was
minted under; ``--checkpoint-contract milestones --save-interval N`` (A19) retains every multiple of N
plus the final step, with mid-run params/assets sync and no pruning, and moves the run_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from datetime import datetime, timezone

import yaml
from launch_guardrails import (
    DEFAULT_RESULTS_BUCKET,
    EXECUTION_ACCOUNT,
    OWNER_EMAIL,
    PROJECT_TAG,
    REGION,
    STORAGE_ACCOUNT,
    STUDY_OWNER,
    WSM_ROBOCASA_S3,
    add_guardrail_arguments,
    prepared_source_bundle,
    source_tree_sha256,
    submit_training_job,
    training_plan_arn,
    validate_and_confirm,
)

ENTRY = "robocasa_pi05_finetune_entry.sh"
INSTANCE_TYPE = "ml.p5.48xlarge"
# Each queue's service environment is bound to ONE instance family, so the instance type is a
# function of --queue, not a free parameter: the p5e training plan rejects a p5 request. Both
# entries are 8-GPU single-node, which is what the runtime topology contract asserts.
QUEUE_INSTANCE_TYPES = {
    "fss-tri-cam-robotics-p5-48xlarge-us-west-2": INSTANCE_TYPE,
    "fss-tri-cam-robotics-p5e-48xlarge-us-west-2-training-plan": "ml.p5e.48xlarge",
}
STUDY = "long_context_v1"
DEFAULT_OWNER = STUDY_OWNER
#: The live submitting identity. NOT derivable from DEFAULT_OWNER: the internship-era
#: sarvesh.patil@tri.global is deactivated, while every content address in this study is minted
#: under the `sarvesh.patil` S3 prefix and can never move.
DEFAULT_OWNER_EMAIL = OWNER_EMAIL
INIT_S3 = f"{WSM_ROBOCASA_S3}/pretrain150k/pi05/mg60_bal33/run/149999"
TARGET_DATA_S3 = f"{WSM_ROBOCASA_S3}/datasets/v1.0/target"
REMEMBENCH_DATA_S3 = f"{WSM_ROBOCASA_S3}/datasets/remembench_v02"
ROBOSUITE_SHA = "85abee228d1c43ab1939bce33028099945d453b4"
ROBOCASA_SHA = "be22d659b02db8f6d7f3a3c3edc742934fdcbaae"
BASE_CONFIG = "scripts/configs/train/pi05_target_finetune.yaml"
WORKSPACE_CONFIG = "scripts/configs/train/pi05_workspace_finetune.yaml"
STAGE_Q_CONFIG = "scripts/configs/train/pi05_stage_q_finetune.yaml"
STAGE_S3_CONFIG = "scripts/configs/train/pi05_stage_s3_jepa_finetune.yaml"
# H13 (live/joint WSM). Two arms, one interface: h13a = R1 (live encoder + keypatch decode through
# w), h13b = R2 (+ LeJEPA against a LIVE t+k target). Neither consumes the omega cache.
STAGE_H13_CONFIGS = {
    "h13a": "scripts/configs/train/pi05_stage_h13_dec_finetune.yaml",
    "h13b": "scripts/configs/train/pi05_stage_h13_dec_jepa_finetune.yaml",
    "h13c": "scripts/configs/train/pi05_stage_h13_dec_lang_finetune.yaml",
    "h13d": "scripts/configs/train/pi05_stage_h13_dec_jepa_lang_finetune.yaml",
    # R5-R8: the same four recipes composed with the gdn8 history module. These arms READ the omega
    # window at inference, so unlike h13a-d they ARE omega consumers (see H13_GDN_ARMS below).
    "h13e": "scripts/configs/train/pi05_stage_h13_dec_gdn8_finetune.yaml",
    "h13f": "scripts/configs/train/pi05_stage_h13_dec_jepa_gdn8_finetune.yaml",
    "h13g": "scripts/configs/train/pi05_stage_h13_dec_lang_gdn8_finetune.yaml",
    "h13h": "scripts/configs/train/pi05_stage_h13_dec_jepa_lang_gdn8_finetune.yaml",
    # R3b/R4b/R7b/R8b: the language factor re-run with a CE-over-vocabulary head. h13c/d/g/h are
    # SUPERSEDED (their InfoNCE head collapsed to the caption centroid and cost ~5pp).
    "h13c2": "scripts/configs/train/pi05_stage_h13_dec_lang2_finetune.yaml",
    "h13d2": "scripts/configs/train/pi05_stage_h13_dec_jepa_lang2_finetune.yaml",
    "h13g2": "scripts/configs/train/pi05_stage_h13_dec_lang2_gdn8_finetune.yaml",
    "h13h2": "scripts/configs/train/pi05_stage_h13_dec_jepa_lang2_gdn8_finetune.yaml",
}
#: The gdn8 subset of the H13 arms. They compose the sanctioned {tanh read + H13 live aux} pair, so
#: they dispatch through PI_STAGE_S_INTERFACE=tanh (the entry then stages + validates the omega
#: cache) and they serve with the conditioner KEPT and every H13 subtree stripped.
H13_GDN_ARMS = frozenset({"h13e", "h13f", "h13g", "h13h", "h13g2", "h13h2"})
#: (lejepa, language) factor set per H13 arm. ONE table, so the arm/recipe consistency check below
#: cannot forget a family: h13e-h13h are the gdn8 twins of h13a-h13d and carry the same factors.
H13_FACTORS = {
    "h13a": (False, False),
    "h13b": (True, False),
    "h13c": (False, True),
    "h13d": (True, True),
    "h13e": (False, False),
    "h13f": (True, False),
    "h13g": (False, True),
    "h13h": (True, True),
    # The lang2 family carries the language factor via `h13_lang_cls`, not `h13_lang`, so their
    # (lejepa, language) entries record lang=False here and the CE flag is checked separately below.
    "h13c2": (False, False),
    "h13d2": (True, False),
    "h13g2": (False, False),
    "h13h2": (True, False),
}
#: Arms whose language factor is the R3b CE head (model.h13_lang_cls) rather than the superseded
#: InfoNCE head (model.h13_lang).
H13_LANG_CLS_ARMS = frozenset({"h13c2", "h13d2", "h13g2", "h13h2"})
H13_ARMS = frozenset(STAGE_H13_CONFIGS)
STAGE_Q_WINDOW_LEN = 8  # chunk-steps per window; global batch = 8 windows x 8 = 64 per-step samples
STAGE_Q_GLOBAL_BATCH_WINDOWS = 8
FEATURE_MANIFEST_NAME = "manifest.json"
STAGED_MANIFEST_NAME = "_stage_s_run_manifest.json"
INIT_INVENTORY_ARTIFACT = "pi05_h300_mg_init"
TARGET_INVENTORY_ARTIFACT = "robocasa_target50"
INIT_INVENTORY_NAMESPACE = "init"
TARGET_INVENTORY_NAMESPACE = "data"

# --- Dataset profiles ------------------------------------------------------------------------
# The target dataset is a PARAMETER of the launch, not a constant of the study. Each profile owns
# (a) the inventory artifact identity, (b) the S3 root, (c) the provenance facts the run manifest
# records, and (d) the node-entry environment that reshapes the validators. The RoboCasa profile is
# the default and its ``entry_env`` is EMPTY: every entry knob already defaults to the target50
# contract, so a RoboCasa plan's environment dict is byte-identical to the pre-profile launcher.
#: Repo-relative per-task demo counts for the ReMemBench train split. This file is DERIVED from the
#: published omega manifest / inventory (it is the same map the node's feature validator consumes as
#: POLICY_FEATS_DEMOS_PER_TASK_MAP), so ``--single-task`` validates its argument and reports the
#: run's demo count against the artifact rather than against a literal typed here.
REMEMBENCH_DEMOS_PER_TASK_MAP = "scripts/configs/data/remembench_v02_train13_demos_per_task.json"
#: How a single-task run is realized. The staged SUBSTRATE stays the whole sealed dataset (13 task
#: dirs, 323 omega files) because the omega cache's identity is not divisible: ``encoder_id`` =
#: sha256(encoder_provenance) and that provenance embeds the full 13-task ``demos_per_task`` map, so
#: a filtered map/expected-task count would derive a DIFFERENT encoder_id and fail the node's
#: content-addressed feature validator. The training SOUP is filtered instead, at the one chokepoint
#: that already exists for it (``utils.soup._filter_soup_tasks``, env ``WSM_TASKS``), which fails
#: loud on a task name that is not in the materialized soup.
SINGLE_TASK_MECHANISM = "WSM_TASKS"
DEFAULT_DATASET_PROFILE = "robocasa_target50"
DATASET_PROFILES: dict[str, dict] = {
    "robocasa_target50": {
        "inventory_artifact": TARGET_INVENTORY_ARTIFACT,
        "dataset_s3": TARGET_DATA_S3,
        "task_prompt_namespace": "robocasa_target50",
        "benchmark": "RoboCasa",
        "tasks": 50,
        "target_fraction_per_task": 0.30,
        "demos_per_task": 150,
        "episode_subsample_seed": 0,
        "expected_episodes_per_task": 150,
        # Single-task runs are a ReMemBench-only design (see SINGLE_TASK_MECHANISM); RoboCasa has no
        # per-task demo map to validate the argument against, so --single-task fails closed here.
        "single_task_demos_map": None,
        "entry_env": {},
    },
    "remembench_v02_train13": {
        "inventory_artifact": "remembench_train13",
        "dataset_s3": REMEMBENCH_DATA_S3,
        # The URI namespace segment is "remembench13" (pinned by publish_stage_s_artifact.py and
        # submit_pi_stage_s_eval.py); the manifest's own artifact FIELD is remembench_train13_task_prompts.
        "task_prompt_namespace": "remembench13",
        "benchmark": "ReMemBench",
        "tasks": 13,
        # Not a fraction/seed selection: the train split is the complement of a held-out tail, so
        # every remaining demo is used and per-task counts differ (9..44, 323 total).
        "target_fraction_per_task": None,
        "demos_per_task": "all",
        "episode_subsample_seed": None,
        "expected_episodes_per_task": None,
        "total_demos": 323,
        "single_task_demos_map": REMEMBENCH_DEMOS_PER_TASK_MAP,
        "entry_env": {
            "TARGET_INVENTORY_ARTIFACT": "remembench_train13",
            # Object keys start with "train/", so the task dirs land one level below the
            # materialize destination.
            "TARGET_ROOT_SUBDIR": "v1.0/target/train",
            "TARGET_EXPECTED_TASKS": "13",
            "TARGET_TASK_DIR_GLOBS": "*/*/lerobot",
            "TASK_PROMPT_NAMESPACE": "remembench13",
            "TASK_PROMPT_ARTIFACT": "remembench_train13_task_prompts",
            "POLICY_FEATS_DATASET_NAME": "remembench_v02_train13",
            "POLICY_FEATS_DEMOS_PER_TASK_MAP": REMEMBENCH_DEMOS_PER_TASK_MAP,
            "OMEGA_EXPECTED_FILES": "323",
        },
    },
}
TRAIN_STEPS = 60_000
CANARY_TRAIN_STEPS = 1
# The production final-checkpoint step (final step = steps - 1). Canaries have their own final step
# (0) computed locally; only production checkpoints are evaluable, so the eval launcher pins this.
FINAL_STEP = TRAIN_STEPS - 1
MAX_CANARY_RUN_SECONDS = 6 * 3600
EXPECTED_JAX_DEVICES = 8
EXPECTED_JAX_PROCESSES = 1
GLOBAL_BATCH_SIZE = 64
NUM_WORKERS = 32
OMEGA_CACHE_MAX_ITEMS = 8192
OMEGA_CACHE_MAX_BYTES = 512 * 1024**2
FSDP_DEVICES = 1
# One five-day attempt is the aggregate five-day maximum. A retry is a new launch and therefore
# returns through the user's explicit-approval gate.
RETRY = {"attempts": 1}

# --- Checkpoint contracts ----------------------------------------------------------------------
# `final-only` is the sealed default: save_interval == steps, one retained step (the final one), no
# mid-run sync. Every existing run_id was minted under it, so its spec and environment must not move
# by a byte. `milestones` (A19 checkpoint-maturity protocol) is selected EXPLICITLY with
# --checkpoint-contract milestones --save-interval N: the trainer saves every N steps, orbax keeps every
# multiple of N (keep_period == N on top of max_to_keep=1, so nothing is ever pruned), the entry syncs
# each committed milestone's params/+assets/ while training runs and re-syncs all of them at the end,
# and ONE completion claim lists every uploaded step. The contract is sealed through
# training.save_interval + training.checkpoint_policy, so the two contracts can never share a run_id.
FINAL_ONLY_CONTRACT = "final-only"
MILESTONES_CONTRACT = "milestones"
CHECKPOINT_CONTRACTS = (FINAL_ONLY_CONTRACT, MILESTONES_CONTRACT)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMAGE = re.compile(
    rf"^{EXECUTION_ACCOUNT}\.dkr\.ecr\.{REGION}\.amazonaws\.com/"
    r"[a-z0-9][a-z0-9._/-]*@sha256:([0-9a-f]{64})$"
)


def study_root(owner: str) -> str:
    if not _OWNER.fullmatch(owner):
        raise SystemExit(f"invalid --user storage owner {owner!r}")
    return f"s3://{DEFAULT_RESULTS_BUCKET}/{owner}/wsm_robocasa/studies/{STUDY}"


def content_addressed_archive(uri: str, *, component: str, root: str) -> str:
    """Validate canonical ``code/<component>/<sha256>.tgz`` and return its digest."""
    match = re.fullmatch(
        rf"{re.escape(root)}/code/{re.escape(component)}/([0-9a-f]{{64}})\.tgz",
        uri,
    )
    if match is None:
        raise SystemExit(
            f"--{component}-source-s3 must be content-addressed at {root}/code/{component}/<64hex>.tgz; got {uri}"
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# wsmv2 <-> openpi archive pairing
#
# The two source archives are NOT independent axes. wsmv2's trainer modules reach into the fork's
# dataloader for its import-time gates (``_groot_dataset._WSM_*`` / ``_NAV_SPLIT_*``) to fail loud
# when a knob did not survive import order. Those reads are a CONTRACT ON THE FORK: pin a wsmv2
# archive whose checks postdate the openpi archive and the run dies at import with an
# ``AttributeError`` — after a node has already started.
#
# This happened on 2026-08-07: wsmv2 522ad4b0 (combo-era gate checks) paired with the n-wave's
# openpi 768f274a (predates ``_WSM_JEPA_WITH_WINDOW``) killed audit discriminator D1 at 17:24Z and
# would have killed D2. The pairing is checkable offline in milliseconds, so it is checked here
# rather than discovered on a GPU.
# ---------------------------------------------------------------------------

#: ``_groot_dataset.<attr>`` reads in the wsmv2 trainer modules. The alias is the fixed local name
#: every call site uses (``import ... as _groot_dataset``); a new alias would silently narrow this
#: scan, so keep the import spelling uniform.
_FORK_ATTRIBUTE_READ = re.compile(r"_groot_dataset\.(_[A-Za-z0-9_]+)")
#: Module-level bindings in the fork's dataloader. Import-time gates are all plain assignments at
#: column 0, which is exactly what the wsmv2 checks depend on being present.
_FORK_ATTRIBUTE_BINDING = r"^{name}\s*[:=]"
#: Where the trainer-side reads live. Scanning the whole archive would sweep in this launcher's own
#: docstrings and the tests that deliberately reference missing attributes.
_FORK_CONSUMER_SUBTREE = "vla_training"
#: The fork module every one of those reads targets.
_FORK_DATALOADER_RELPATH = pathlib.Path("src/openpi/groot_utils/groot_openpi_dataset.py")
#: Additional fork modules whose module-level bindings can satisfy a requirement. These carry
#: ARCHITECTURE sentinels rather than loader gates: a recipe that selects a conditioner the pinned
#: openpi archive does not implement is the same failure as a missing loader gate, and it is just as
#: checkable offline. Scanned only when present, so an older archive still reports the dataloader's
#: gates instead of dying on a file that predates the module.
_FORK_SENTINEL_RELPATHS = (pathlib.Path("src/openpi/models/wsm_current_cond.py"),)
#: Fork sentinel a PTRM recipe requires (openpi.models.wsm_current_cond._WSM_PTRM).
_FORK_PTRM_SENTINEL = "_WSM_PTRM"
#: Fork sentinel every H13 recipe requires. The trainer selects the live encoder by CONFIG, so the
#: attribute scan alone would not notice an openpi archive that predates H13 and would silently ship
#: no future frame (R2) / no live encoder at all.
_FORK_H13_SENTINEL = "_WSM_H13_FUTURE_FRAMES"
#: Fork sentinel a LANGUAGE recipe requires (openpi dataloader must be able to ship caption targets).
_FORK_H13_LANG_SENTINEL = "_WSM_H13_LANG_TARGETS"
#: Canonical location of the content-addressed caption-embedding asset.
_CAPTION_EMB_URI = re.compile(r"^s3://[^/]+/[^/]+/wsm_robocasa/studies/[^/]+/artifacts/captions/[0-9a-f]{64}/?$")


def fork_attributes_read(wsmv2_root: pathlib.Path) -> set[str]:
    """Every ``_groot_dataset._X`` attribute the wsmv2 tree expects the fork to define."""
    consumer_root = pathlib.Path(wsmv2_root) / _FORK_CONSUMER_SUBTREE
    if not consumer_root.is_dir():
        raise SystemExit(f"wsmv2 archive is missing its {_FORK_CONSUMER_SUBTREE}/ trainer tree: {consumer_root}")
    found: set[str] = set()
    for module in sorted(consumer_root.rglob("*.py")):
        found.update(_FORK_ATTRIBUTE_READ.findall(module.read_text(encoding="utf-8")))
    return found


def fork_attributes_defined(openpi_root: pathlib.Path) -> set[str]:
    """Every module-level gate/sentinel the fork binds, across the dataloader and the model modules."""
    dataloader = pathlib.Path(openpi_root) / _FORK_DATALOADER_RELPATH
    if not dataloader.is_file():
        raise SystemExit(f"openpi archive is missing {_FORK_DATALOADER_RELPATH}: {dataloader}")
    sources = [dataloader.read_text(encoding="utf-8")]
    for relpath in _FORK_SENTINEL_RELPATHS:
        module = pathlib.Path(openpi_root) / relpath
        if module.is_file():
            sources.append(module.read_text(encoding="utf-8"))
    return {
        match.group(1)
        for source in sources
        for match in re.finditer(r"^(_[A-Za-z0-9_]+)\s*[:=]", source, re.MULTILINE)
    }


def assert_archive_pairing(
    *,
    wsmv2_root: pathlib.Path,
    openpi_root: pathlib.Path,
    wsmv2_sha256: str,
    openpi_sha256: str,
    recipe_required: frozenset[str] = frozenset(),
) -> set[str]:
    """Fail closed when the paired openpi archive cannot satisfy the wsmv2 archive's fork reads.

    ``recipe_required`` adds the sentinels THIS run's recipe needs on top of the reads scanned out of
    the wsmv2 tree (e.g. the PTRM conditioner): the trainer selects those by config, not by an
    attribute lookup, so scanning alone would not notice an archive that cannot build them.

    Returns the set of verified attributes so the caller can report what was actually covered.
    """
    required = fork_attributes_read(wsmv2_root) | set(recipe_required)
    available = fork_attributes_defined(openpi_root)
    missing = sorted(required - available)
    if missing:
        raise SystemExit(
            "incompatible source-archive pairing: the openpi fork does not define "
            + ", ".join(missing)
            + f", which wsmv2 {wsmv2_sha256} requires (trainer reads + recipe sentinels). "
            f"openpi={openpi_sha256} wsmv2={wsmv2_sha256}. "
            "The paired archives are a contract, not independent pins: this run would die at "
            "import with AttributeError after a node had already started. Pair a newer openpi "
            "archive (or an older wsmv2)."
        )
    return required


def _archive_tree(
    sha256: str,
    *,
    uri: str,
    component: str,
    cache_dir: pathlib.Path | None,
    workspace: pathlib.Path,
    allow_download: bool,
) -> pathlib.Path | None:
    """Materialize a published source archive locally, or None when offline and uncached.

    Resolution order: an already-extracted ``<sha>/`` directory, a cached ``<sha>.tgz``, then S3 —
    the last only when ``allow_download`` (a dry run stays strictly offline).
    """
    import tarfile

    if cache_dir is not None:
        extracted = pathlib.Path(cache_dir) / sha256
        if extracted.is_dir():
            return extracted
        tarball = pathlib.Path(cache_dir) / f"{sha256}.tgz"
        if tarball.is_file():
            destination = workspace / component
            with tarfile.open(tarball) as archive:
                archive.extractall(destination)
            return destination
    if not allow_download:
        return None
    import boto3

    bucket, _, key = uri[len("s3://") :].partition("/")
    downloaded = workspace / f"{component}.tgz"
    boto3.client("s3", region_name=REGION).download_file(bucket, key, str(downloaded))
    destination = workspace / component
    with tarfile.open(downloaded) as archive:
        archive.extractall(destination)
    return destination


def verify_archive_pairing(
    args,
    *,
    wsmv2_sha256: str,
    openpi_sha256: str,
    recipe_required: frozenset[str] = frozenset(),
) -> str:
    """Cross-check the shipped wsmv2 archive's fork reads against the paired openpi archive.

    Fail-closed on a real submission: an unresolvable archive is treated as a failed check, because
    the whole point is that nothing reaches a node without the pairing being proven. A dry run
    stays offline by contract, so it reports UNVERIFIED instead of reaching for S3.
    """
    import tempfile

    cache_dir = getattr(args, "archive_cache_dir", None)
    allow_download = not args.dry_run
    with tempfile.TemporaryDirectory(prefix="wsm-archive-pairing-") as workspace_name:
        workspace = pathlib.Path(workspace_name)
        wsmv2_root = _archive_tree(
            wsmv2_sha256,
            uri=args.wsmv2_source_s3,
            component="wsmv2",
            cache_dir=cache_dir,
            workspace=workspace,
            allow_download=allow_download,
        )
        openpi_root = _archive_tree(
            openpi_sha256,
            uri=args.openpi_source_s3,
            component="openpi",
            cache_dir=cache_dir,
            workspace=workspace,
            allow_download=allow_download,
        )
        if wsmv2_root is None or openpi_root is None:
            unresolved = "wsmv2" if wsmv2_root is None else "openpi"
            if allow_download:
                raise SystemExit(
                    f"cannot verify the source-archive pairing: the {unresolved} archive did not "
                    "resolve. Submission is fail-closed on this check."
                )
            return (
                f"UNVERIFIED ({unresolved} archive not available offline; pass "
                "--archive-cache-dir or drop --dry-run to check it)"
            )
        verified = assert_archive_pairing(
            wsmv2_root=wsmv2_root,
            openpi_root=openpi_root,
            wsmv2_sha256=wsmv2_sha256,
            openpi_sha256=openpi_sha256,
            recipe_required=recipe_required,
        )
    return f"verified ({len(verified)} fork attributes)"


def content_addressed_inventory(uri: str, claimed_sha256: str, *, artifact: str, namespace: str, root: str) -> str:
    """Validate canonical manifests/inventories/<namespace>/<sha256>.json."""
    if not _HEX64.fullmatch(claimed_sha256):
        raise SystemExit(f"inventory SHA-256 for {artifact} must be 64 lowercase hex characters")
    expected = f"{root}/manifests/inventories/{namespace}/{claimed_sha256}.json"
    if uri != expected:
        raise SystemExit(f"inventory for {artifact} must be content-addressed at {expected}; got {uri}")
    return claimed_sha256


def content_addressed_tokenizer(uri: str, claimed_sha256: str, *, root: str) -> str:
    """Validate the immutable PaliGemma tokenizer location and return its digest."""
    if not _HEX64.fullmatch(claimed_sha256):
        raise SystemExit("--tokenizer-sha256 must be 64 lowercase hex characters")
    expected = f"{root}/artifacts/tokenizers/paligemma/{claimed_sha256}.model"
    if uri != expected:
        raise SystemExit(f"--tokenizer-s3 must be content-addressed at {expected}; got {uri}")
    return claimed_sha256


def image_digest(image_uri: str) -> str:
    match = _IMAGE.fullmatch(image_uri)
    if match is None:
        raise SystemExit(
            "--image-uri must be an account-141 us-west-2 ECR URI pinned as "
            "<repository>@sha256:<64hex>; tags such as :latest are forbidden"
        )
    return match.group(1)


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


def resolve_source_dir(value: str | None) -> pathlib.Path:
    path = pathlib.Path(value or pathlib.Path(__file__).resolve().parents[3] / "internal_training").resolve()
    if not (path / ENTRY).is_file():
        raise SystemExit(f"source-dir {path} is missing {ENTRY}")
    return path


def single_task_filter(args: argparse.Namespace, profile: dict) -> dict | None:
    """Resolve ``--single-task`` into the sealed training-soup filter (None when unset).

    The returned block is recorded under ``data.training_task_filter`` and is the ONLY difference
    between a single-task run and its 13-task parent: same inventory, same materialized substrate,
    same omega cache, same validators. ``demos`` is read out of the derived per-task map so the
    manifest states the run's real training mass instead of a number typed at the call site.
    """
    task = getattr(args, "single_task", None)
    if not task:
        return None
    demos_map_rel = profile.get("single_task_demos_map")
    if not demos_map_rel:
        raise SystemExit(f"--single-task is not defined for dataset profile {args.dataset_profile!r}")
    map_path = pathlib.Path(__file__).resolve().parents[2] / demos_map_rel
    if not map_path.is_file():
        raise SystemExit(f"per-task demo map missing from the local wsmv2 tree: {map_path}")
    with map_path.open(encoding="utf-8") as stream:
        demos_by_task = json.load(stream)
    if task not in demos_by_task:
        raise SystemExit(
            f"--single-task {task!r} is not a task of {args.dataset_profile}; choose one of {sorted(demos_by_task)}"
        )
    demos = demos_by_task[task]
    if type(demos) is not int or demos < 1:
        raise SystemExit(f"per-task demo map has an invalid count for {task!r}: {demos!r}")
    return {
        "mechanism": SINGLE_TASK_MECHANISM,
        "tasks": [task],
        "demos": demos,
        "staged_substrate_tasks": profile["tasks"],
        "demos_per_task_source": demos_map_rel,
    }


def milestone_retained_steps(train_steps: int, save_interval: int | None, *, canary: bool) -> list[int]:
    """Retained steps under the milestones contract.

    Every positive multiple of ``save_interval`` BELOW ``train_steps``, then the final step
    ``train_steps - 1`` (which the openpi trainer always writes). ``save_interval`` must divide
    ``train_steps`` so the last interval boundary coincides with the run's end. A production run must
    have at least one intermediate milestone -- otherwise the plan IS the final-only contract and must
    say so; a canary (1 step) may degenerate to ``[0]`` so the milestone entry path can be exercised
    end to end for one step. The node entry re-derives this set from WSM_MAX_STEPS/WSM_SAVE_INTERVAL and
    refuses a plan whose sealed list disagrees.
    """
    if save_interval is None:
        raise SystemExit("--checkpoint-contract milestones requires --save-interval")
    if type(save_interval) is not int or save_interval < 1:
        raise SystemExit(f"--save-interval must be a positive integer, got {save_interval!r}")
    if train_steps % save_interval != 0:
        raise SystemExit(
            f"--save-interval {save_interval} must divide the step count {train_steps} so the last "
            "milestone boundary coincides with the end of the run"
        )
    if save_interval >= train_steps and not canary:
        raise SystemExit(
            f"--save-interval {save_interval} leaves no intermediate milestone below {train_steps} "
            "steps; that is the final-only contract, so drop --checkpoint-contract"
        )
    return [*range(save_interval, train_steps, save_interval), train_steps - 1]


def build_plan(args: argparse.Namespace, source_dir: pathlib.Path) -> dict:
    if type(args.attempt_index) is not int or args.attempt_index < 1:
        raise SystemExit("--attempt-index must be a positive integer")
    if args.queue not in QUEUE_INSTANCE_TYPES:
        raise SystemExit(f"no instance type is registered for queue {args.queue}")
    instance_type = QUEUE_INSTANCE_TYPES[args.queue]
    # Plan-backed queues need the job to pin the flexible training plan itself; derived from the
    # queue (see launch_guardrails.training_plan_arn), sealed below, and None on ordinary queues so
    # their plan and their submission are byte-identical to every prior run.
    plan_arn = training_plan_arn(args.queue)
    root = study_root(args.user)
    profile = DATASET_PROFILES[getattr(args, "dataset_profile", DEFAULT_DATASET_PROFILE)]
    target_inventory_artifact = profile["inventory_artifact"]
    target_data_s3 = profile["dataset_s3"]
    training_task_filter = single_task_filter(args, profile)
    init_inventory_sha = content_addressed_inventory(
        args.init_inventory_s3,
        args.init_inventory_sha256,
        artifact=INIT_INVENTORY_ARTIFACT,
        namespace=INIT_INVENTORY_NAMESPACE,
        root=root,
    )
    target_inventory_sha = content_addressed_inventory(
        args.target_inventory_s3,
        args.target_inventory_sha256,
        artifact=target_inventory_artifact,
        namespace=TARGET_INVENTORY_NAMESPACE,
        root=root,
    )
    tokenizer_sha = content_addressed_tokenizer(args.tokenizer_s3, args.tokenizer_sha256, root=root)
    wsmv2_sha = content_addressed_archive(args.wsmv2_source_s3, component="wsmv2", root=root)
    openpi_sha = content_addressed_archive(args.openpi_source_s3, component="openpi", root=root)
    container_sha = image_digest(args.image_uri)
    # Hash the exact sanitized tree that SageMaker will receive. This covers every internal_training
    # support script, not only the entrypoint, and the submission helper rechecks it after staging to
    # close the build-plan/submission race.
    with prepared_source_bundle(
        source_dir,
        ENTRY,
        {"SAGEMAKER_PROGRAM": ENTRY},
        args.secrets_manager_arn,
    ) as (staged_source, _safe_entry, _safe_environment):
        internal_training_sha = source_tree_sha256(staged_source)

    # Arms that consume the omega cache (S1/S2, S3's train-time targets, AND the workspace-on Q
    # arms) share the same required feature args + env plumbing; q0/q2 must NOT receive them (the
    # entry forbids POLICY_FEATS_S3).
    # The gdn8 H13 arms consume the omega cache exactly as s1 does; the aux-only H13 arms (h13a-d)
    # must NOT, and are refused the feature arguments by the else-branch below.
    workspace_arm = args.arm in {"s1", "s2", "s3", "q1", "q3"} | H13_GDN_ARMS
    feature_fields = (
        args.encoder_id,
        args.policy_features_s3,
        args.policy_features_manifest_s3,
        args.policy_features_manifest_sha256,
        args.task_prompt_manifest_s3,
        args.task_prompt_manifest_sha256,
    )
    if workspace_arm:
        if not all(feature_fields):
            raise SystemExit(
                "S1/S2 require --encoder-id, --policy-features-s3, "
                "--policy-features-manifest-s3, --policy-features-manifest-sha256, "
                "--task-prompt-manifest-s3, and --task-prompt-manifest-sha256"
            )
        if not _HEX64.fullmatch(args.encoder_id):
            raise SystemExit("--encoder-id must be a 64-character lowercase SHA-256 identifier")
        if not _HEX64.fullmatch(args.policy_features_manifest_sha256):
            raise SystemExit("--policy-features-manifest-sha256 must be 64 lowercase hex characters")
        if not _HEX64.fullmatch(args.task_prompt_manifest_sha256):
            raise SystemExit("--task-prompt-manifest-sha256 must be 64 lowercase hex characters")
        expected_prompt_manifest = (
            f"{root}/manifests/artifacts/workspace/task_prompts/"
            f"{profile['task_prompt_namespace']}/{args.task_prompt_manifest_sha256}.json"
        )
        if args.task_prompt_manifest_s3 != expected_prompt_manifest:
            raise SystemExit(f"--task-prompt-manifest-s3 must be content-addressed at {expected_prompt_manifest}")
        expected_features = f"{root}/caches/{args.encoder_id}/omega"
        if args.policy_features_s3.rstrip("/") != expected_features:
            raise SystemExit(f"--policy-features-s3 must match encoder provenance exactly: {expected_features}")
        expected_feature_manifest = (
            f"{root}/manifests/artifacts/workspace/{args.encoder_id}/omega/{args.policy_features_manifest_sha256}.json"
        )
        if args.policy_features_manifest_s3 != expected_feature_manifest:
            raise SystemExit(f"--policy-features-manifest-s3 must be content-addressed at {expected_feature_manifest}")
        policy_features_s3 = expected_features
        feature_manifest_s3 = expected_feature_manifest
        task_prompt_manifest_s3 = expected_prompt_manifest
        task_prompt_manifest_sha = args.task_prompt_manifest_sha256
        encoder_id = args.encoder_id
        feature_manifest_sha = args.policy_features_manifest_sha256
    else:
        if any(feature_fields):
            raise SystemExit("S0 forbids encoder/policy-feature arguments")
        policy_features_s3 = None
        feature_manifest_s3 = None
        task_prompt_manifest_s3 = None
        task_prompt_manifest_sha = None
        encoder_id = None
        feature_manifest_sha = None

    run_kind = "canary" if args.canary else "train"
    # The production step count is a launch parameter (the ReMemBench arms run a shorter schedule on
    # a ~4%-sized dataset). It defaults to the RoboCasa campaign's 60k, and the node entry asserts
    # the two agree via STAGE_S_EXPECTED_TRAIN_STEPS.
    production_steps = int(getattr(args, "train_steps", None) or TRAIN_STEPS)
    if production_steps < 1:
        raise SystemExit("--train-steps must be a positive integer")
    train_steps = CANARY_TRAIN_STEPS if args.canary else production_steps
    final_step = train_steps - 1
    if args.canary:
        if args.max_run_seconds > MAX_CANARY_RUN_SECONDS:
            raise SystemExit(f"Stage-S canaries require --max-run-seconds <= {MAX_CANARY_RUN_SECONDS}")
        if args.priority != 1:
            raise SystemExit("Stage-S canaries require --priority 1")
    # Checkpoint contract. Resolved here, sealed below (training.save_interval + checkpoint_policy) and
    # exported to the entry (WSM_SAVE_INTERVAL / WSM_FINAL_ONLY_CHECKPOINTS, plus the milestone keys).
    contract = getattr(args, "checkpoint_contract", None) or FINAL_ONLY_CONTRACT
    if contract not in CHECKPOINT_CONTRACTS:
        raise SystemExit(f"--checkpoint-contract must be one of {CHECKPOINT_CONTRACTS}, got {contract!r}")
    save_interval_arg = getattr(args, "save_interval", None)
    if contract == MILESTONES_CONTRACT:
        retained_steps = milestone_retained_steps(train_steps, save_interval_arg, canary=args.canary)
        save_interval = int(save_interval_arg)
        keep_period = save_interval
        midrun_sync = True
    else:
        if save_interval_arg is not None:
            raise SystemExit(
                "--save-interval is only meaningful with --checkpoint-contract milestones; the "
                "final-only contract saves once, at the final step"
            )
        save_interval = train_steps
        retained_steps = [final_step]
        keep_period = None
        midrun_sync = False

    # Stage-Q (RoboTTT 2x2): q0/q2 are omega-independent; q1/q3 additionally stage + validate the
    # omega cache exactly like S1/S2 (the workspace flag IS the promoted Stage-S read). The entry
    # dispatches all four to finetune_pi_05_seq.py (window loader + sequence train step);
    # batch_size counts WINDOWS there (8 x L=8 = 64 per-step samples).
    interface = {
        "s0": "base",
        "s1": "tanh",
        "s2": "cfg2",
        "s3": "jepa",
        "h13a": "h13",
        "h13b": "h13",
        "h13c": "h13",
        "h13d": "h13",
        # R5-R8 ride the tanh interface (the S5 combo pattern): the entry's tanh branch stages and
        # validates the omega cache the gated-DeltaNet reads, and the recipe's model.h13 turns the
        # live aux on. A separate interface would fork the omega plumbing for no benefit.
        "h13e": "tanh",
        "h13f": "tanh",
        "h13g": "tanh",
        "h13h": "tanh",
        # lang2 family: aux-only twins dispatch as h13, gdn8 twins as tanh (same rule as above).
        "h13c2": "h13",
        "h13d2": "h13",
        "h13g2": "tanh",
        "h13h2": "tanh",
        "q0": "q0",
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
    }[args.arm]
    stage_q_arm = args.arm in {"q0", "q1", "q2", "q3"}
    h13_arm = args.arm in H13_ARMS
    if stage_q_arm:
        config = STAGE_Q_CONFIG
    elif args.arm == "s3":
        config = STAGE_S3_CONFIG
    elif h13_arm:
        config = STAGE_H13_CONFIGS[args.arm]
    else:
        config = BASE_CONFIG if args.arm == "s0" else WORKSPACE_CONFIG
    if args.config_override:
        # Recipe-variant runs (e.g. A6 iid-loader q0, A7 staged q2). The override path is sealed
        # into the run manifest and its CONTENT is pinned by the wsmv2 archive sha — same
        # provenance guarantees as the arm-default config. Must live under scripts/configs/train/.
        if not args.config_override.startswith("scripts/configs/train/"):
            raise SystemExit(
                f"--config-override must be repo-relative under scripts/configs/train/: {args.config_override}"
            )
        config = args.config_override
    # The omega-window length is part of the sealed recipe: override configs may widen it
    # (model.cond_window, e.g. the s1 deltanet arm trains at 8) and the manifest/env must say so —
    # the in-process recipe would win at train time anyway, but a manifest that claims window 1 for
    # a window-8 run is a provenance lie. Arm-default configs all run window 1.
    # The action norm-stat nav split (data.norm_split_nav) is sealed for the same reason: it changes
    # the norm_stats blob written into the checkpoint's assets, so a manifest that omits it cannot
    # distinguish an n-wave run from its parent. Arm-default configs never set it.
    # The S5 combo arm (model.jepa_aux) is sealed for the same reason: it adds a second loss term and
    # a `wsm_jepa_head` subtree to an otherwise ordinary s1 checkpoint, so a manifest that omitted it
    # could not distinguish the combo run from its deltanet parent. The interface, serve contract and
    # arm label are unchanged (the aux is train-only), so nothing else in the plan moves.
    # model.cond_history_dropout is sealed for the same reason as cond_window: it is a TRAIN-TIME
    # intervention on the deltanet recurrence (each historical window element is deleted with this
    # probability per sample; the newest element never is) that leaves the checkpoint SHAPE identical,
    # so nothing downstream could otherwise tell a dropout run apart from its parent w-arm.
    # model.cond_type: gated_deltanet_ptrm (H9) is sealed under its own `ptrm` block for the same
    # reason again: it changes the module behind wsm_tanh_cond and adds a Q loss term, and its
    # depth/lambda are the two numbers that define the arm. The block is written ONLY on a PTRM run —
    # an always-present `"ptrm": null` would change every other arm's canonical manifest JSON and
    # therefore its run_id, which is exactly the provenance break the GR00T bug taught us to refuse.
    cond_window = 1
    cond_history_dropout = 0.0
    norm_split_nav = False
    jepa_aux = None
    ptrm = None
    if args.config_override:
        override_path = pathlib.Path(__file__).resolve().parents[2] / args.config_override
        if not override_path.is_file():
            raise SystemExit(f"--config-override not found in the local wsmv2 tree: {override_path}")
        with override_path.open() as fh:
            override_recipe = yaml.safe_load(fh) or {}
        override_model = override_recipe.get("model") or {}
        cond_type = str(override_model.get("cond_type", "tanh"))
        cond_window = int(override_model.get("cond_window", 1))
        if cond_window < 1:
            raise SystemExit(f"model.cond_window must be >= 1, got {cond_window}")
        cond_history_dropout = float(override_model.get("cond_history_dropout", 0.0))
        if not 0.0 <= cond_history_dropout <= 0.9:
            raise SystemExit(f"model.cond_history_dropout must be in [0.0, 0.9], got {cond_history_dropout}")
        if cond_history_dropout > 0.0:
            if cond_type == "gated_deltanet_ptrm":
                raise SystemExit(
                    "model.cond_history_dropout is refused on the PTRM read: the v1 arm isolates the "
                    "recursion, and a second train-time intervention on the same window would make "
                    "it a two-variable experiment"
                )
            if cond_type != "gated_deltanet":
                raise SystemExit(
                    "model.cond_history_dropout intervenes on the gated-DeltaNet window; it "
                    f"requires model.cond_type: gated_deltanet, got {cond_type!r}"
                )
            if args.arm != "s1":
                raise SystemExit(
                    f"model.cond_history_dropout rides the tanh/deltanet read (arm s1), got --arm {args.arm}"
                )
        norm_split_nav = (override_recipe.get("data") or {}).get("norm_split_nav", False)
        if type(norm_split_nav) is not bool:
            raise SystemExit(f"data.norm_split_nav must be a boolean, got {norm_split_nav!r}")
        combo_aux = override_model.get("jepa_aux", False)
        if type(combo_aux) is not bool:
            raise SystemExit(f"model.jepa_aux must be a boolean, got {combo_aux!r}")
        if combo_aux:
            if args.arm != "s1":
                raise SystemExit(f"model.jepa_aux rides the tanh read; it is an s1 recipe, got --arm {args.arm}")
            jepa_aux = {
                "lambda_jepa": float(override_model.get("jepa_weight", 1.0)),
                "regularizer": str(override_model.get("jepa_regularizer", "sigreg")),
                "lambda_sigreg": float(override_model.get("sigreg_weight", 0.05)),
                "num_futures": int(override_model.get("jepa_num_futures", 1)),
                "train_time_only": True,
            }
        if cond_type == "gated_deltanet_ptrm":
            if args.arm != "s1":
                raise SystemExit(
                    "model.cond_type: gated_deltanet_ptrm fills the tanh read's wsm_tanh_cond "
                    f"subtree; it is an s1 recipe, got --arm {args.arm}"
                )
            if combo_aux:
                raise SystemExit(
                    "the PTRM arm is deliberately unconfounded and refuses model.jepa_aux: the "
                    "recursion must be the only delta versus its deltanet parent"
                )
            ptrm_steps = override_model.get("ptrm_steps", 4)
            if type(ptrm_steps) is not int or not 1 <= ptrm_steps <= 16:
                raise SystemExit(f"model.ptrm_steps must be an integer in [1, 16], got {ptrm_steps!r}")
            ptrm_q_weight = float(override_model.get("ptrm_q_weight", 0.1))
            if not ptrm_q_weight >= 0.0:
                raise SystemExit(f"model.ptrm_q_weight must be >= 0, got {ptrm_q_weight}")
            ptrm = {
                "steps": ptrm_steps,
                "q_weight": ptrm_q_weight,
                # Stated in the manifest because it is the paper-fidelity claim a reader would
                # otherwise have to take on trust: training is noiseless and the K-rollout noise
                # exists only at inference, which is what makes PTRM a knob on this checkpoint.
                "eval_noise": "inference_only",
            }
    # H13 (live/joint WSM). Sealed under its own `h13` block for exactly the reason `ptrm` is: the
    # arm adds loss terms and three checkpoint subtrees while leaving the SHAPE of everything a
    # reader would otherwise inspect untouched, so a manifest without these numbers could not tell R1
    # from R2 — nor either from a base run. The block is written ONLY on an H13 run; an always-present
    # `"h13": null` would change every other arm's canonical JSON and therefore its run_id.
    h13 = None
    if h13_arm:
        h13_path = pathlib.Path(__file__).resolve().parents[2] / config
        if not h13_path.is_file():
            raise SystemExit(f"H13 recipe not found in the local wsmv2 tree: {h13_path}")
        with h13_path.open() as fh:
            h13_model = (yaml.safe_load(fh) or {}).get("model") or {}
        if h13_model.get("h13") is not True:
            raise SystemExit(f"--arm {args.arm} requires model.h13: true in {config}")
        h13_jepa_on = h13_model.get("h13_jepa", False)
        if type(h13_jepa_on) is not bool:
            raise SystemExit(f"model.h13_jepa must be a boolean, got {h13_jepa_on!r}")
        want_jepa, want_lang = H13_FACTORS[args.arm]
        if h13_jepa_on != want_jepa:
            raise SystemExit(f"--arm {args.arm} expects model.h13_jepa={want_jepa}, got {h13_jepa_on} in {config}")
        h13_lang_on = h13_model.get("h13_lang", False)
        if type(h13_lang_on) is not bool:
            raise SystemExit(f"model.h13_lang must be a boolean, got {h13_lang_on!r}")
        if h13_lang_on != want_lang:
            raise SystemExit(f"--arm {args.arm} expects model.h13_lang={want_lang}, got {h13_lang_on} in {config}")
        want_cls = args.arm in H13_LANG_CLS_ARMS
        h13_lang_cls_on = h13_model.get("h13_lang_cls", False)
        if type(h13_lang_cls_on) is not bool:
            raise SystemExit(f"model.h13_lang_cls must be a boolean, got {h13_lang_cls_on!r}")
        if h13_lang_cls_on != want_cls:
            raise SystemExit(f"--arm {args.arm} expects model.h13_lang_cls={want_cls}, got {h13_lang_cls_on}")
        caption_emb_s3 = str(h13_model.get("caption_emb_s3", ""))
        if (h13_lang_on or h13_lang_cls_on) and not _CAPTION_EMB_URI.fullmatch(caption_emb_s3):
            raise SystemExit(
                "a language arm requires model.caption_emb_s3 content-addressed at "
                f"{{study}}/artifacts/captions/<sha256>/, got {caption_emb_s3!r}"
            )
        if h13_model.get("salient") or h13_model.get("jepa_aux"):
            raise SystemExit(
                "an H13 recipe must not also enable the frozen-target salient/jepa aux: two decoders "
                "on the same labels (or two targets on the same penultimate) make the arm unreadable"
            )
        # The gdn8 arms carry cond_type/cond_window in their ARM-DEFAULT yaml, but the block above
        # only reads those from an --config-override. Without this the manifest would seal
        # workspace_window=1 and the environment would export WSM_K_WINDOW=1 while the conditioner
        # trained at 8 — the loader would ship 1-row windows to an 8-window recurrence, and the
        # manifest would be a provenance lie about it. Read them from the arm's own recipe.
        if args.arm in H13_GDN_ARMS:
            cond_type = str(h13_model.get("cond_type", "tanh"))
            if cond_type != "gated_deltanet":
                raise SystemExit(
                    f"--arm {args.arm} is a gdn8 arm and requires model.cond_type: gated_deltanet, got {cond_type!r}"
                )
            cond_window = int(h13_model.get("cond_window", 1))
            if cond_window < 2:
                raise SystemExit(f"--arm {args.arm} expects the dnw8 window (8), got cond_window={cond_window}")
        elif h13_model.get("cond_type", "tanh") != "tanh":
            raise SystemExit(
                f"--arm {args.arm} is aux-only and must not set model.cond_type "
                f"({h13_model.get('cond_type')!r}); that is the h13e-h13h family"
            )
        h13 = {
            "live_encoder": True,
            # None for h13a-d; the dnw8 composition for h13e-h13h.
            "gdn8": (
                {
                    "cond_type": "gated_deltanet",
                    "cond_window": cond_window,
                    "tanh_gate_init": 0.001,
                    "merged_from": "pi05_stage_s1_deltanet_finetune.yaml (the sealed 59.9 anchor)",
                    "serve": "conditioner KEPT, H13 aux subtrees stripped",
                    "eval_server_state_mode": "per_env_isolated_v1",
                }
                if args.arm in H13_GDN_ARMS
                else None
            ),
            # The single most consequential fact about this arm, stated where a reader will find it.
            "stop_gradient": False,
            "encoder_subtree": "wsm_enc",
            "w_dim": int(h13_model.get("h13_w_dim", 512)),
            "enc_layers": int(h13_model.get("h13_enc_layers", 4)),
            "enc_heads": int(h13_model.get("h13_enc_heads", 8)),
            "decoder": "wsm_salient_head_192_multihot_from_w",
            "lambda_dec": float(h13_model.get("h13_dec_weight", 0.5)),
            "lambda_sigreg": float(h13_model.get("h13_sigreg_weight", 0.05)),
            # NOT content-addressed in the study store (same caveat the salient arm carries): what is
            # pinned is the prefix plus the [pi-h13] log line the trainer emits at build time.
            "keypatch_labels_s3": str(
                h13_model.get(
                    "labels_s3",
                    f"{WSM_ROBOCASA_S3}/wsm_labels",
                )
            ),
            "keypatch_label_geometry": "pi (vlm_episode_pi_*.npz), 3 views x 8x8 = 192 global ids",
            "lejepa": (
                {
                    "lambda_jepa": float(h13_model.get("h13_jepa_weight", 0.1)),
                    "num_futures": int(h13_model.get("h13_future_k", 1)),
                    "grid_stride": int(h13_model.get("h13_future_stride", 8)),
                    "target": "live_encoder_on_future_frame",
                    "target_detached": False,
                    "teacher": "none (no EMA, no stop-grad — SIGReg on both sides)",
                    "reference_row": "s3_jw01 (frozen target, same lambda and k) = 0.5816",
                }
                if h13_jepa_on
                else None
            ),
            "aux_subtrees": ["wsm_enc", "wsm_dec"]
            + (["wsm_h13_jepa_head"] if h13_jepa_on else [])
            + (["wsm_lang_head"] if h13_lang_on else [])
            + (["wsm_lang_cls"] if h13_lang_cls_on else []),
            # R3/R4 language decode. Sealed for the same reason as `lejepa`: it adds a loss term and
            # a checkpoint subtree while leaving everything a reader would otherwise inspect intact.
            "language_cls": (
                {
                    "objective": "cross-entropy over the deduplicated caption vocabulary",
                    "lambda_lang": float(h13_model.get("h13_lang_cls_weight", 0.5)),
                    "vocab_size": int(h13_model.get("h13_lang_vocab", 8661)),
                    "normalisation": "CE / ln V (1.0 at init, vocabulary-size invariant)",
                    "id_definition": "index into the sorted unique caption strings",
                    "supersedes": "the InfoNCE-vs-frozen-embedding head (h13c/d/g/h), which "
                    "collapsed to the caption centroid and cost ~5pp",
                    "caption_cls_s3": caption_emb_s3,
                }
                if h13_lang_cls_on
                else None
            ),
            "language": (
                {
                    "lambda_lang": float(h13_model.get("h13_lang_weight", 0.5)),
                    "temperature": float(h13_model.get("h13_lang_temperature", 0.07)),
                    "objective": "InfoNCE-in-batch (normalised by ln B) + cosine",
                    "target": "frozen pi0.5 text tower on the ACTIVE segment's caption",
                    "target_detached": True,
                    "segment_lookup": "stored [t0,t1) intervals (never searchsorted over keyframes)",
                    "caption_emb_s3": caption_emb_s3,
                    "caption_emb_sha256": caption_emb_s3.rstrip("/").rsplit("/", 1)[-1],
                    "not_used": "feats.npz subgoal_embs (image-conditioned)",
                }
                if h13_lang_on
                else None
            ),
            "serve": "aux_subtrees stripped; sample_actions byte-identical to s0",
            # The SEALED eval protocol this run will be scored under, pinned at TRAIN time so the
            # later eval phase cannot silently drift onto the checked-in scripts/configs/eval/
            # pi05_eval.yaml — which is NOT the protocol (it has num_trials 10, seed 7, replan 5).
            # R1-R4 are aux-only and serve the vanilla stack, hence stateless_v1.
            "eval_protocol": {
                "benchmark": "RoboCasa",
                "split": "target",
                "task_sets": ["atomic_seen", "composite_seen", "composite_unseen"],
                "tasks": 50,
                "episodes_per_task": 100,
                "seed": 20260723,
                "exec_steps": 8,
                "replan_steps": 8,
                "envs_per_gpu": 8,
                "server_state_mode": "stateless_v1",
                "metric": "avg_task_weighted",
                "episode_manifest_sha256": ("d57ab80be9ee2c14a70d0d28dd3722586200e9e5fe43207d1d11fc48d22d889a"),
            },
        }
    entry_sha = _sha256_file(source_dir / ENTRY)
    spec = {
        "schema_version": 1,
        "study": STUDY,
        "run_kind": run_kind,
        "arm": args.arm,
        "interface": interface,
        "backbone": "pi0.5",
        "initialization": {
            "recipe": "H300+MG balanced",
            "checkpoint_s3": INIT_S3,
            "inventory": {
                "uri": args.init_inventory_s3,
                "sha256": init_inventory_sha,
                "artifact": INIT_INVENTORY_ARTIFACT,
            },
        },
        "data": {
            "benchmark": profile["benchmark"],
            "tasks": profile["tasks"],
            "target_fraction_per_task": profile["target_fraction_per_task"],
            "demos_per_task": profile["demos_per_task"],
            "episode_subsample_seed": profile["episode_subsample_seed"],
            "dataset_s3": target_data_s3,
            "inventory": {
                "uri": args.target_inventory_s3,
                "sha256": target_inventory_sha,
                "artifact": target_inventory_artifact,
            },
            "robosuite_sha": ROBOSUITE_SHA,
            "robocasa_sha": ROBOCASA_SHA,
            # None on every multi-task run (the historical shape); a sealed block on the
            # single-task cells. The substrate above is IDENTICAL either way.
            "training_task_filter": training_task_filter,
        },
        "training": {
            "config": config,
            "steps": train_steps,
            "save_interval": save_interval,
            "checkpoint_policy": {
                "retained_steps": retained_steps,
                "keep_period": keep_period,
                "midrun_sync": midrun_sync,
                "resume": False,
                "tree_manifest_schema": 1,
                "completion_claim_schema": 1,
            },
            "batch_size": STAGE_Q_GLOBAL_BATCH_WINDOWS if stage_q_arm else GLOBAL_BATCH_SIZE,
            "batch_unit": "windows" if stage_q_arm else "steps",
            "per_device_batch_size": (STAGE_Q_GLOBAL_BATCH_WINDOWS if stage_q_arm else GLOBAL_BATCH_SIZE)
            // EXPECTED_JAX_DEVICES,
            "num_workers": NUM_WORKERS,
            "image_resize_path": "worker_pil_bilinear",
            "defer_image_resize_to_model_preprocess": False,
            "param_norm_metric": "log_boundary_post_update",
            "jax_devices": EXPECTED_JAX_DEVICES,
            "jax_processes": EXPECTED_JAX_PROCESSES,
            "fsdp_devices": FSDP_DEVICES,
            "data_parallel_replicas": EXPECTED_JAX_DEVICES // FSDP_DEVICES,
            "train_seed": 42,
            "norm_split_nav": norm_split_nav,
            "workspace_window": cond_window if workspace_arm else None,
            # Train-only historical-element deletion inside the deltanet recurrence; the newest
            # (current-timestep) element is never deleted and nothing is deleted at serve.
            "cond_history_dropout": cond_history_dropout if workspace_arm else None,
            "jepa_aux": jepa_aux,
            "tanh_gate_init": 0.001 if (args.arm == "s1" or args.arm in H13_GDN_ARMS) else None,
            "cfg_drop_probability": 0.2 if args.arm == "s2" else None,
            "legacy_direct_token": False,
            "legacy_cfg": False,
            "future_conditioning": False,
            # Stage-Q (RoboTTT 2x2): the two arm booleans + the shared window recipe (07a).
            "stage_q_fast_weights": (args.arm in {"q2", "q3"}) if stage_q_arm else None,
            "stage_q_workspace": (args.arm in {"q1", "q3"}) if stage_q_arm else None,
            "stage_q_window_len": STAGE_Q_WINDOW_LEN if stage_q_arm else None,
            "stage_q_chunk_stride": 8 if stage_q_arm else None,
            "stage_q_tbptt_segment": 8 if stage_q_arm else None,
            "stage_q_inner_loss": "mean_normalized_mse_d9" if stage_q_arm else None,
        },
        "sources": {
            "wsmv2": {"uri": args.wsmv2_source_s3, "sha256": wsmv2_sha},
            "openpi": {"uri": args.openpi_source_s3, "sha256": openpi_sha},
            "internal_training": {
                "sanitized_source_tree_sha256": internal_training_sha,
                "entry_path": ENTRY,
                "entry_sha256": entry_sha,
            },
            "image": {"uri": args.image_uri, "sha256": container_sha},
            "tokenizer": {"uri": args.tokenizer_s3, "sha256": tokenizer_sha},
        },
        "workspace_representation": {
            "encoder_id": encoder_id,
            "policy_features_s3": policy_features_s3,
            "feature_manifest_schema": 1 if workspace_arm else None,
            "expected_tasks": profile["tasks"] if workspace_arm else None,
            "expected_episodes_per_task": (profile["expected_episodes_per_task"] if workspace_arm else None),
            "feature_manifest_uri": feature_manifest_s3,
            "feature_manifest_sha256": feature_manifest_sha,
            "encoder_id_definition": ("sha256(canonical_json(encoder_provenance))" if workspace_arm else None),
            "required_subgoal_dropout": 1.0 if workspace_arm else None,
            "required_global_language_mode": ("canonical_terse_task_instruction" if workspace_arm else None),
            "task_prompt_manifest": (
                {"uri": task_prompt_manifest_s3, "sha256": task_prompt_manifest_sha} if workspace_arm else None
            ),
            "policy_transport": (
                {
                    "cached_omega_dtype": "float16",
                    "selected_omega_dtype": "float32",
                    "workspace_language_transport": "omitted_current_only",
                    "cache_policy": "deterministic_lru",
                    "cache_max_items_per_worker": OMEGA_CACHE_MAX_ITEMS,
                    "cache_max_payload_bytes_per_worker": OMEGA_CACHE_MAX_BYTES,
                }
                if workspace_arm
                else None
            ),
        },
        "infrastructure": {
            "execution_account": EXECUTION_ACCOUNT,
            "storage_account": STORAGE_ACCOUNT,
            "queue": args.queue,
            "training_plan_arn": plan_arn,
            "role": args.role,
            "instance_type": instance_type,
            "priority": args.priority,
            "max_run_seconds": args.max_run_seconds,
            "attempts": RETRY["attempts"],
            "attempt_index": args.attempt_index,
            "aggregate_max_run_seconds": args.max_run_seconds * RETRY["attempts"],
        },
    }
    if h13 is not None:
        # Same insert-only rule as `ptrm` below: a key that exists on nothing but the runs it
        # describes cannot perturb any other arm's spec_sha256.
        spec["training"]["h13"] = h13
    if ptrm is not None:
        # Inserted rather than declared above with a None default, ON PURPOSE: `_canonical_json`
        # serializes every key, so a `"ptrm": null` sitting in the training block of an s0/s1/s2/s3
        # plan would change that plan's spec_sha256 and therefore its run_id — silently renaming
        # every live baseline. A key that only exists on the runs it describes cannot do that.
        spec["training"]["ptrm"] = ptrm
    if contract == MILESTONES_CONTRACT:
        # Insert-only, like `ptrm`/`h13`: a key that exists on nothing but milestone runs cannot perturb
        # any final-only plan's spec_sha256 -- every existing run_id was minted without it.
        spec["training"]["checkpoint_policy"]["contract"] = MILESTONES_CONTRACT
    spec_sha = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
    run_id = f"{args.arm}{'-canary' if args.canary else ''}-{spec_sha[:16]}"
    output_namespace = "canaries/training" if args.canary else "checkpoints"
    output_s3 = f"{root}/{output_namespace}/pi05/{args.arm}/{run_id}"
    manifest_s3 = f"{root}/manifests/runs/{run_kind}/{run_id}.json"
    producer_claim_s3 = f"{root}/manifests/claims/{run_kind}/{run_id}/producer.json"
    completion_claim_s3 = f"{root}/manifests/claims/{run_kind}/{run_id}/step-{final_step}.complete.json"
    manifest, manifest_json = _seal_manifest(
        {
            **spec,
            "run_id": run_id,
            "spec_sha256": spec_sha,
            "output_s3": output_s3,
            "manifest_s3": manifest_s3,
            "claims": {
                "producer": producer_claim_s3,
                "completion": completion_claim_s3,
            },
        }
    )

    environment = {
        # An explicitly pinned training plan REPLACES the implicit reserved-capacity request; the
        # two are alternatives, not a pair (reference: vla_foundry_internal PR 822). Ordinary
        # queues keep "1", so their environment is unchanged. submit_training_job re-checks this.
        "SM_USE_RESERVED_CAPACITY": "0" if plan_arn else "1",
        "SAGEMAKER_PROGRAM": ENTRY,
        "PI_STAGE_S_INTERFACE": interface,
        "INIT_S3": INIT_S3,
        "INIT_INVENTORY_S3": args.init_inventory_s3,
        "INIT_INVENTORY_SHA256": init_inventory_sha,
        "TARGET_DATA_S3": target_data_s3,
        "TARGET_INVENTORY_S3": args.target_inventory_s3,
        "TARGET_INVENTORY_SHA256": target_inventory_sha,
        "WSM_REPO_S3": args.wsmv2_source_s3,
        "OPENPI_FORK_S3": args.openpi_source_s3,
        "PALIGEMMA_TOKENIZER_S3": args.tokenizer_s3,
        "PALIGEMMA_TOKENIZER_SHA256": tokenizer_sha,
        "WSM_FT_CONFIG": config,
        "WSM_MAX_STEPS": str(train_steps),
        "WSM_SAVE_INTERVAL": str(save_interval),
        # "1" on the sealed default. "0" only under the milestones contract: the trainer then honours
        # WSM_SAVE_INTERVAL and WSM_KEEP_PERIOD instead of forcing save_interval = steps.
        "WSM_FINAL_ONLY_CHECKPOINTS": "1" if contract == FINAL_ONLY_CONTRACT else "0",
        "WSM_EXPECTED_JAX_DEVICES": str(EXPECTED_JAX_DEVICES),
        "WSM_EXPECTED_JAX_PROCESSES": str(EXPECTED_JAX_PROCESSES),
        # Stage-Q batch_size counts WINDOWS (8 x L=8 = 64 per-step samples = the S0 recipe).
        "WSM_EXPECTED_GLOBAL_BATCH": str(STAGE_Q_GLOBAL_BATCH_WINDOWS if stage_q_arm else GLOBAL_BATCH_SIZE),
        "WSM_EXPECTED_NUM_WORKERS": str(NUM_WORKERS),
        "WSM_EXPECTED_FSDP_DEVICES": str(FSDP_DEVICES),
        "OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS": "0",
        "WSM_NORM_SPLIT_NAV": "1" if norm_split_nav else "0",
        "STAGE_S_RUN_ID": run_id,
        "STAGE_S_RUN_KIND": run_kind,
        "STAGE_S_FINAL_STEP": str(final_step),
        "OUTPUT_S3": output_s3,
        "RUN_MANIFEST_SOURCE": STAGED_MANIFEST_NAME,
        "RUN_MANIFEST_SHA256": manifest["manifest_sha256"],
        "RUN_MANIFEST_S3": manifest_s3,
        "PRODUCER_CLAIM_S3": producer_claim_s3,
        "COMPLETION_CLAIM_S3": completion_claim_s3,
        "WANDB_PROJECT": "wsm-robocasa",
        "WANDB_RUN_GROUP": f"long-context-stage-s-{run_kind}-{args.arm}",
    }
    # Dataset-shape knobs for the node entry. The RoboCasa profile contributes NOTHING here (its
    # entry_env is empty and every entry default IS the target50 contract), so the historical plan's
    # environment is unchanged; only a non-default profile/schedule adds keys.
    environment.update(profile["entry_env"])
    if training_task_filter:
        # utils.soup._filter_soup_tasks: restricts the materialized soup to these task names and
        # raises if any of them is absent, so a typo can never silently train on the wrong set.
        environment[SINGLE_TASK_MECHANISM] = ",".join(training_task_filter["tasks"])
    if production_steps != TRAIN_STEPS:
        environment["STAGE_S_EXPECTED_TRAIN_STEPS"] = str(production_steps)
    if contract == MILESTONES_CONTRACT:
        # The entry's milestone branch is selected by the PLAN, never by a bare save interval: without
        # STAGE_S_CHECKPOINT_CONTRACT it keeps asserting the sealed final-only contract (exit 36).
        # WSM_KEEP_PERIOD == WSM_SAVE_INTERVAL is what makes orbax keep every milestone (max_to_keep=1
        # keeps the newest; keep_period keeps every step % N == 0), and the entry re-derives the
        # retained set from WSM_MAX_STEPS/WSM_SAVE_INTERVAL and cross-checks it against this list.
        environment["STAGE_S_CHECKPOINT_CONTRACT"] = MILESTONES_CONTRACT
        environment["WSM_KEEP_PERIOD"] = str(save_interval)
        environment["STAGE_S_RETAINED_STEPS"] = ",".join(str(step) for step in retained_steps)
    if workspace_arm:
        environment.update(
            {
                "POLICY_FEATS_S3": policy_features_s3,
                "POLICY_FEATS_MANIFEST_S3": feature_manifest_s3,
                "POLICY_FEATS_MANIFEST_SHA256": feature_manifest_sha,
                "TASK_PROMPT_MANIFEST_S3": task_prompt_manifest_s3,
                "TASK_PROMPT_MANIFEST_SHA256": task_prompt_manifest_sha,
                "WSM_ENCODER_ID": encoder_id,
                "WSM_K_WINDOW": str(cond_window),
                "WSM_DEMO_CACHE_SIZE": str(OMEGA_CACHE_MAX_ITEMS),
                "WSM_DEMO_CACHE_MAX_BYTES": str(OMEGA_CACHE_MAX_BYTES),
            }
        )
    if args.arm == "s1":
        environment["WSM_TANH_GATE_INIT"] = "0.001"
        if cond_history_dropout > 0.0:
            # Env wins over yaml inside _pi05_common, so the sealed plan (not just the staged repo)
            # is what selects the intervention; a plan whose environment lacks the key can never
            # silently run one.
            environment["WSM_COND_HISTORY_DROPOUT"] = repr(cond_history_dropout)
        if ptrm is not None:
            # Same rule as the dropout knob: the PLAN selects the conditioner, not merely the staged
            # yaml, so a run whose environment lacks these keys can never quietly be its parent arm.
            environment["WSM_COND_TYPE"] = "gated_deltanet_ptrm"
            environment["WSM_PTRM_STEPS"] = str(ptrm["steps"])
            environment["WSM_PTRM_Q_WEIGHT"] = repr(ptrm["q_weight"])
    elif args.arm == "s2":
        environment["WSM_CFG_P_DROP"] = "0.2"
    elif h13_arm:
        # Same rule as the PTRM/dropout knobs: the PLAN selects the recipe, not merely the staged
        # yaml, so a run whose environment lacks these keys can never quietly be a different arm.
        # (Env wins over yaml inside _pi05_common.)
        environment["WSM_H13_DEC_WEIGHT"] = repr(h13["lambda_dec"])
        environment["WSM_H13_SIGREG_WEIGHT"] = repr(h13["lambda_sigreg"])
        environment["WSM_H13_W_DIM"] = str(h13["w_dim"])
        environment["WSM_H13_ENC_LAYERS"] = str(h13["enc_layers"])
        environment["WSM_H13_ENC_HEADS"] = str(h13["enc_heads"])
        environment["WSM_SALIENT_LABELS_S3"] = h13["keypatch_labels_s3"]
        if h13["lejepa"]:
            environment["WSM_H13_JEPA_WEIGHT"] = repr(h13["lejepa"]["lambda_jepa"])
            environment["WSM_H13_FUTURE_K"] = str(h13["lejepa"]["num_futures"])
            environment["WSM_H13_FUTURE_STRIDE"] = str(h13["lejepa"]["grid_stride"])
        if h13["language_cls"]:
            # Same rule as every other H13 knob: the PLAN selects the recipe, not merely the staged
            # yaml, so a run whose environment lacks these keys can never quietly be a different arm.
            environment["WSM_H13_LANG_CLS"] = "1"
            environment["WSM_H13_LANG_CLS_WEIGHT"] = repr(h13["language_cls"]["lambda_lang"])
            environment["WSM_H13_LANG_VOCAB"] = str(h13["language_cls"]["vocab_size"])
            environment["WSM_H13_CAPTION_EMB_S3"] = h13["language_cls"]["caption_cls_s3"]
        if h13["language"]:
            environment["WSM_H13_LANG_WEIGHT"] = repr(h13["language"]["lambda_lang"])
            environment["WSM_H13_LANG_TEMPERATURE"] = repr(h13["language"]["temperature"])
            environment["WSM_H13_CAPTION_EMB_S3"] = h13["language"]["caption_emb_s3"]

    forbidden = {
        "WSM_CFG",
        "WSM_LEGACY_TOKEN_INJECTION",
        "WSM_CFG_WITH_FUTURE",
        "WSM_Z_WINDOWS_ROOT",
        "Z_WINDOWS_S3",
    }
    leaked = sorted(forbidden.intersection(environment))
    if leaked:
        raise AssertionError(f"Stage-S plan selected forbidden legacy variables: {leaked}")
    # SageMaker TrainingEnvironmentValue is capped at 512 bytes. The manifest itself is staged as a
    # source file; only short references/checksums enter job metadata.
    oversized = {
        key: len(value.encode("utf-8")) for key, value in environment.items() if len(value.encode("utf-8")) > 512
    }
    if oversized:
        raise AssertionError(f"SageMaker environment value exceeds 512 bytes: {oversized}")
    return {
        "run_id": run_id,
        "output_s3": output_s3,
        "manifest_s3": manifest_s3,
        "manifest": manifest,
        "manifest_json": manifest_json,
        "source_tree_sha256": internal_training_sha,
        "environment": environment,
        # Fork sentinels THIS recipe needs the paired openpi archive to define. Outside the manifest
        # on purpose (it is a launch-time compatibility fact, not a property of the run), so adding
        # it moves no run_id.
        "required_fork_attributes": ((_FORK_PTRM_SENTINEL,) if ptrm is not None else ())
        + ((_FORK_H13_SENTINEL,) if h13 is not None else ())
        + ((_FORK_H13_LANG_SENTINEL,) if (h13 or {}).get("language") else ()),
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
            "h13e",
            "h13f",
            "h13g",
            "h13h",
            "h13c2",
            "h13d2",
            "h13g2",
            "h13h2",
            "q0",
            "q1",
            "q2",
            "q3",
        ),
        help=(
            "h13a = R1 (live encoder + keypatch decode through w); h13b = R2 (+ LeJEPA); "
            "h13c = R3 (R1 + language); h13d = R4 (R2 + language); "
            "h13e-h13h = R5-R8, the same four composed with the gdn8 history module"
        ),
    )
    parser.add_argument(
        "--config-override",
        default=None,
        help="repo-relative train config replacing the arm default (recipe-variant runs; sealed into the manifest)",
    )
    parser.add_argument(
        "--dataset-profile",
        default=DEFAULT_DATASET_PROFILE,
        choices=sorted(DATASET_PROFILES),
        help="target dataset identity: inventory artifact, S3 root, prompt namespace, and the "
        "node-entry validator shape (default reproduces the RoboCasa target50 launch exactly)",
    )
    parser.add_argument(
        "--single-task",
        default=None,
        help="train on ONE task of the profile's dataset (ReMemBench single-task cells). The staged "
        "substrate, inventory and omega cache are unchanged; only the training soup is filtered "
        f"(env {SINGLE_TASK_MECHANISM}), and the choice is sealed into the run manifest.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=TRAIN_STEPS,
        help=f"production optimizer steps (default {TRAIN_STEPS}; canaries always run 1)",
    )
    parser.add_argument(
        "--checkpoint-contract",
        default=FINAL_ONLY_CONTRACT,
        choices=CHECKPOINT_CONTRACTS,
        help=(
            "final-only (default; the sealed contract every existing run_id was minted under) or "
            "milestones (A19: save every --save-interval steps, retain every multiple plus the final "
            "step, mid-run params/assets sync, never prune, one completion claim listing all of them)"
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help=(
            "milestones contract only: checkpoint every N optimizer steps. N must divide --train-steps "
            "and leave at least one milestone below it (e.g. --train-steps 60000 --save-interval 15000 "
            "retains 15000, 30000, 45000, 59999). Refused under final-only."
        ),
    )
    parser.add_argument("--user", default=DEFAULT_OWNER, help="account-124 storage prefix owner")
    # The storage-prefix owner and the SCP's owner-email tag are DIFFERENT identities and must not
    # be derived from one another. Every content address in this study is minted under the
    # `sarvesh.patil` prefix, so --user is frozen there forever; the submitting IAM identity is now
    # sarvesh.patil.pi@tri.global, and org SCP p-ahpdy5vv denies batch:SubmitServiceJob unless
    # `tri.owner.email` and `tri.project` are present and non-empty. Deriving the tag from --user
    # (the old `f"{args.user}@tri.global"`) would either tag a DEACTIVATED address or move the
    # entire study root to reach the live one.
    parser.add_argument(
        "--owner-email",
        default=DEFAULT_OWNER_EMAIL,
        help=(
            "value of the required `tri.owner.email` SCP tag; independent of --user, which only "
            f"names the S3 storage prefix (default {DEFAULT_OWNER_EMAIL})"
        ),
    )
    parser.add_argument("--source-dir", default=None, help="local internal_training source directory")
    parser.add_argument("--wsmv2-source-s3", required=True)
    parser.add_argument("--openpi-source-s3", required=True)
    parser.add_argument(
        "--tokenizer-s3",
        required=True,
        help="content-addressed PaliGemma tokenizer artifact",
    )
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument(
        "--init-inventory-s3",
        required=True,
        help="content-addressed H300+MG initialization object inventory",
    )
    parser.add_argument("--init-inventory-sha256", required=True)
    parser.add_argument(
        "--target-inventory-s3",
        required=True,
        help="content-addressed target-dataset object inventory (artifact per --dataset-profile)",
    )
    parser.add_argument("--target-inventory-sha256", required=True)
    parser.add_argument(
        "--image-uri",
        required=True,
        help="account-141 ECR image pinned with @sha256:<digest>",
    )
    parser.add_argument("--encoder-id", default=None, help="S1/S2: 64hex encoder provenance ID")
    parser.add_argument("--policy-features-s3", default=None, help="S1/S2: canonical omega cache")
    parser.add_argument(
        "--policy-features-manifest-s3",
        default=None,
        help="S1/S2: canonical content-addressed omega manifest",
    )
    parser.add_argument(
        "--policy-features-manifest-sha256",
        default=None,
        help="S1/S2: raw SHA-256 of the canonical omega manifest",
    )
    parser.add_argument(
        "--task-prompt-manifest-s3",
        default=None,
        help="S1/S2: canonical demo-independent terse task-prompt manifest",
    )
    parser.add_argument("--task-prompt-manifest-sha256", default=None)
    parser.add_argument(
        "--canary",
        action="store_true",
        help=(
            "run exactly one optimizer step under canary namespaces; requires priority 1 and "
            f"max runtime <= {MAX_CANARY_RUN_SECONDS}s"
        ),
    )
    parser.add_argument(
        "--attempt-index",
        type=int,
        default=1,
        help="1 for the first attempt; increment only after fresh user approval for a retry",
    )
    parser.add_argument(
        "--archive-cache-dir",
        default=None,
        type=pathlib.Path,
        help=(
            "directory holding published source archives as <sha256>.tgz (or extracted <sha256>/). "
            "Lets --dry-run verify the wsmv2<->openpi pairing without touching S3; a real "
            "submission downloads them when they are not cached"
        ),
    )
    add_guardrail_arguments(parser)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    validate_and_confirm(args)
    source_dir = resolve_source_dir(args.source_dir)
    plan = build_plan(args, source_dir)
    # The wsmv2 archive's fork reads must be satisfiable by the openpi archive it is paired with.
    # Checked before the job name is even minted: an incompatible pair is a launch-time error here
    # instead of an import-time AttributeError on a running node (2026-08-07, D1).
    pairing = verify_archive_pairing(
        args,
        wsmv2_sha256=plan["manifest"]["sources"]["wsmv2"]["sha256"],
        openpi_sha256=plan["manifest"]["sources"]["openpi"]["sha256"],
        recipe_required=frozenset(plan["required_fork_attributes"]),
    )
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    owner = args.user.replace(".", "-")
    job_name = (f"{owner}-pi-stage-{plan['manifest']['run_kind']}-{args.arm}-{plan['run_id'][-16:]}-{stamp}")[
        :63
    ].rstrip("-")

    task_filter = plan["manifest"]["data"]["training_task_filter"]
    policy = plan["manifest"]["training"]["checkpoint_policy"]
    print(
        f"arm={args.arm} kind={plan['manifest']['run_kind']} "
        f"interface={plan['manifest']['interface']} run_id={plan['run_id']}\n"
        + (
            f"  single_task={task_filter['tasks'][0]} demos={task_filter['demos']} "
            f"(substrate {task_filter['staged_substrate_tasks']} tasks)\n"
            if task_filter
            else ""
        )
        + f"  image={args.image_uri}\n"
        f"  wsmv2={args.wsmv2_source_s3}\n"
        f"  openpi={args.openpi_source_s3}\n"
        f"  archive_pairing={pairing}\n"
        f"  tokenizer={args.tokenizer_s3}\n"
        f"  init_inventory={args.init_inventory_s3}\n"
        f"  target_inventory={args.target_inventory_s3}\n"
        f"  checkpoint_contract={policy.get('contract', FINAL_ONLY_CONTRACT)} "
        f"save_interval={plan['manifest']['training']['save_interval']} "
        f"retained_steps={policy['retained_steps']}\n"
        f"  output={plan['output_s3']}\n"
        f"  manifest={plan['manifest_s3']} sha256={plan['manifest']['manifest_sha256']}\n"
        f"  queue={args.queue} priority={args.priority} max_run={args.max_run_seconds}s "
        f"dry={args.dry_run}"
    )
    if args.dry_run:
        print("  [DRY RUN: offline; no AWS SDK load, source upload, or submission]")
        print(json.dumps(plan["manifest"], sort_keys=True, indent=2))
        print("  SUBMISSION READY only after explicit approval and --confirm-submit")
        return

    result = submit_training_job(
        entry=ENTRY,
        source_dir=source_dir,
        environment=plan["environment"],
        image_uri=args.image_uri,
        instance_type=QUEUE_INSTANCE_TYPES[args.queue],
        volume_size=1000,
        tags=[
            {"Key": "tri.project", "Value": PROJECT_TAG},
            {"Key": "tri.owner.email", "Value": args.owner_email},
            {"Key": "wsm.study", "Value": STUDY},
            {"Key": "wsm.arm", "Value": args.arm},
            {"Key": "wsm.run_kind", "Value": plan["manifest"]["run_kind"]},
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
        expected_source_tree_sha256=plan["source_tree_sha256"],
        staged_source_files={STAGED_MANIFEST_NAME: plan["manifest_json"] + "\n"},
    )
    print(f"QUEUED arn={getattr(result[0], 'job_arn', '?') if result else '?'}")


if __name__ == "__main__":
    main()
