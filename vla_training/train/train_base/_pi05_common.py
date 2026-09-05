"""Shared pi0.5 (openpi / JAX) wiring for the base-VLA adapters.

Ported/generalized from the proven reference (ported_raw/reference_code/{wsm_robocasa_configs,
wsm_train_rc}.py): build an openpi ``TrainConfig`` from the YAML recipe + the shared ``GroupedSoup``
(soup + per-dataset balancing weights), apply the vision-LR + wandb-media patches, and dispatch to
the fork's ``scripts/train.py:main``.

pi05-venv-only: all openpi/jax/optax/wandb imports live inside functions, so this module is
import-safe elsewhere. Both ``pretrain_pi_05`` and ``finetune_pi_05`` call ``build_and_run_pi05``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from wsm_settings import WSM_ROBOCASA_S3

_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
_DEFER_IMAGE_RESIZE_ENV = "OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS"
_NORM_SPLIT_NAV_ENV = "WSM_NORM_SPLIT_NAV"
# Captured at IMPORT, unlike every other knob here, because `build_train_config` WRITES this variable
# back for the dataloader (see there): reading it at call time would make our own export shadow the
# yaml on a second build in the same process. Every entry point imports this module from inside
# main(), after the environment is set, so an env override still wins.
_JEPA_NUM_FUTURES_ENV = os.environ.get("WSM_JEPA_NUM_FUTURES")
# The single sanctioned multi-interface combination (S5 combo arm): the tanh/gated-DeltaNet workspace
# READ plus the train-only JEPA aux TARGET. Mirrors openpi's Pi0Config._WORKSPACE_COMBO; kept as a
# literal here because this module must stay importable outside the pi05 venv.
_WORKSPACE_COMBO = {"tanh", "jepa"}
# H13 R5-R8: the same tanh/gated-DeltaNet READ composed with the LIVE joint WSM aux. Mirrors openpi's
# Pi0Config._H13_COMBO for the same reason as above (this module must import outside the pi05 venv).
_H13_COMBO = {"tanh", "h13"}
_SANCTIONED_COMBOS = (_WORKSPACE_COMBO, _H13_COMBO)
# Which module fills the `wsm_tanh_cond` subtree. Mirrors openpi's Pi0Config.wsm_cond_type validation,
# again as a literal so this module imports outside the pi05 venv.
_COND_TYPES = ("tanh", "gated_deltanet", "gated_deltanet_ptrm")
# Conditioners whose recurrence consumes the WHOLE omega window, so the loader's frozen WSM_K_WINDOW
# has to be widened to match the recipe (and checked after the openpi import).
_WINDOWED_COND_TYPES = ("gated_deltanet", "gated_deltanet_ptrm")


def deferred_image_resize_enabled(
    data_config: Mapping[str, object], *, environ: Mapping[str, str] | None = None
) -> bool:
    """Resolve the experimental worker-PIL -> model-JAX resize deferral.

    This is deliberately default-off: the two resize implementations are within one uint8 code
    on the covered camera shapes, but are not bit-identical. An explicit YAML boolean or strict
    ``OPENPI_DEFER_IMAGE_RESIZE_TO_MODEL_PREPROCESS={0,1}`` override is required.
    """
    configured = data_config.get("defer_image_resize_to_model_preprocess", False)
    if type(configured) is not bool:
        raise ValueError(f"data.defer_image_resize_to_model_preprocess must be a boolean, got {configured!r}")
    environment = os.environ if environ is None else environ
    override = environment.get(_DEFER_IMAGE_RESIZE_ENV)
    if override is None:
        return configured
    if override not in {"0", "1"}:
        raise ValueError(f"{_DEFER_IMAGE_RESIZE_ENV} must be exactly 0 or 1, got {override!r}")
    return override == "1"


def norm_split_nav_enabled(data_config: Mapping[str, object], *, environ: Mapping[str, str] | None = None) -> bool:
    """Resolve ``data.norm_split_nav`` (default False = the historical single-blob merge).

    The action norm-stat blob pools every per-dataset mean/std with uniform weights, including the
    fixed-base tasks whose base_motion block is identically zero — so the reordered nav dims 7,8,9
    come out ~1.5x squashed and training z-scores base actions against a diluted scale. With the
    knob on, the fork pools those dims over the MOBILE datasets only (per-dataset std > 0.05 on
    that dim), leaving every other action dim, the state stats, and the merge weights untouched.
    Env wins over yaml, as with every other knob here.
    """
    configured = data_config.get("norm_split_nav", False)
    if type(configured) is not bool:
        raise ValueError(f"data.norm_split_nav must be a boolean, got {configured!r}")
    environment = os.environ if environ is None else environ
    override = environment.get(_NORM_SPLIT_NAV_ENV)
    if override is None:
        return configured
    if override not in {"0", "1"}:
        raise ValueError(f"{_NORM_SPLIT_NAV_ENV} must be exactly 0 or 1, got {override!r}")
    return override == "1"


def export_norm_split_nav(cfg) -> bool:
    """Resolve the knob from the recipe and publish it as ``WSM_NORM_SPLIT_NAV`` for the fork.

    Exported BEFORE the openpi imports, like every other loader-facing knob here: the fork's
    ``compute_overall_statistics`` reads the variable while merging the per-dataset stats, and the
    merged blob is what gets written into the checkpoint's ``assets/norm_stats.json``. A run whose
    recipe says ``norm_split_nav: true`` but whose environment does not would ship the old blob.
    """
    enabled = norm_split_nav_enabled(cfg.data.raw)
    os.environ[_NORM_SPLIT_NAV_ENV] = "1" if enabled else "0"
    return enabled


def assert_norm_split_nav_loader(expected: bool) -> None:
    """Post-import check that the LOADER we actually imported honours the exported nav-split knob.

    Same fail-loud contract as the JEPA-horizon / salient-targets / deltanet-K checks below: after
    the openpi imports, confirm the module that will produce the stats agrees with the recipe.
    The failure mode here is not import order (the fork reads ``WSM_NORM_SPLIT_NAV`` at CALL time
    inside ``compute_overall_statistics``, not at import) but an openpi ARCHIVE that predates the
    knob: it would ignore the variable entirely and silently write the OLD single-blob norm stats
    into ``assets/norm_stats.json``, producing a run that claims to be nav-split and is not. That
    is exactly the class of silent mismatch the n-wave forensic audit had to rule out by hand.
    Nothing here changes behaviour; it converts a silent wrong-stats run into a launch-time error.
    """
    import openpi.groot_utils.groot_openpi_dataset as _groot_dataset

    for attribute in ("_NAV_SPLIT_ENV", "_NAV_SPLIT_ACTION_DIMS", "_NAV_SPLIT_STD_FLOOR"):
        if not hasattr(_groot_dataset, attribute):
            raise ValueError(
                f"the imported openpi dataloader has no {attribute}: this archive predates the "
                f"nav-split knob and would ignore {_NORM_SPLIT_NAV_ENV} while writing the old "
                "single-blob action norm stats"
            )
    loader_env = _groot_dataset._NAV_SPLIT_ENV  # noqa: SLF001  (the fork's knob name)
    if loader_env != _NORM_SPLIT_NAV_ENV:
        raise ValueError(
            f"the dataloader reads {loader_env!r} but this module exports "
            f"{_NORM_SPLIT_NAV_ENV!r}; the nav-split knob would never reach the stats merge"
        )
    exported = os.environ.get(_NORM_SPLIT_NAV_ENV)
    if exported != ("1" if expected else "0"):
        raise ValueError(
            f"{_NORM_SPLIT_NAV_ENV}={exported!r} at loader-import time but the recipe resolved "
            f"norm_split_nav={expected}; something clobbered the export between resolution and "
            "the openpi import"
        )


def runtime_parallelism_summary(
    *,
    batch_size: int,
    num_workers: int,
    fsdp_devices: int,
    device_count: int,
    local_device_count: int,
    process_count: int,
    expected_devices: int | None = None,
    expected_processes: int | None = None,
    expected_batch_size: int | None = None,
    expected_num_workers: int | None = None,
    expected_fsdp_devices: int | None = None,
) -> dict[str, int]:
    """Validate and summarize the effective JAX topology before an expensive run starts.

    Stage-S is a one-process p5.48xlarge job. OpenPI interprets ``batch_size`` as global and shards
    its leading axis across the product of the data and FSDP mesh axes. These checks keep the sealed
    run manifest from claiming an 8-GPU recipe while a broken CUDA/container setup silently exposes
    fewer devices, or while config drift changes a throughput-relevant training setting.
    """
    values = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "fsdp_devices": fsdp_devices,
        "device_count": device_count,
        "local_device_count": local_device_count,
        "process_count": process_count,
    }
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError(f"runtime topology values must be positive integers, got {values}")
    expected = {
        "device_count": expected_devices,
        "process_count": expected_processes,
        "batch_size": expected_batch_size,
        "num_workers": expected_num_workers,
        "fsdp_devices": expected_fsdp_devices,
    }
    for name, wanted in expected.items():
        if wanted is not None and values[name] != wanted:
            raise ValueError(f"Stage-S runtime contract mismatch for {name}: expected={wanted} actual={values[name]}")
    if local_device_count > device_count:
        raise ValueError(f"local_device_count={local_device_count} exceeds device_count={device_count}")
    if device_count % fsdp_devices:
        raise ValueError(f"device_count={device_count} must be divisible by fsdp_devices={fsdp_devices}")
    if batch_size % device_count:
        raise ValueError(f"global batch_size={batch_size} must be divisible by device_count={device_count}")
    return {
        **values,
        "data_parallel_replicas": device_count // fsdp_devices,
        "per_device_batch_size": batch_size // device_count,
    }


_SALIENT_LABELS_S3_DEFAULT = f"{WSM_ROBOCASA_S3}/wsm_labels"
_SALIENT_LABEL_GLOB = "vlm_episode_pi_*.npz"
# The aws-cli --include filter matches against the FULL key remainder after the prefix, so the
# nested <task>/vlm_episode_pi_N.npz keys need a leading wildcard; Path.rglob above does not
# (it matches basenames recursively). Divergence here cost a training job on 2026-08-01.
_SALIENT_LABEL_S3_FILTER = "*vlm_episode_pi_*.npz"


def _stage_salient_labels(labels_s3: str) -> Path:
    """Sync the pi-geometry salient-label npz files to a node-local dir and return it.

    WHERE. Beside the omega cache the entry already stages (``$WORK/wsm_policy_feats``), i.e.
    ``<feats_root>/../wsm_salient_labels``, so both staged assets share one scratch volume and one
    cleanup. With no omega root (local/dry runs) it falls back to ``$TMPDIR/wsm_salient_labels``.

    WHAT. Only ``vlm_episode_pi_*.npz`` — the un-suffixed ``vlm_episode_*.npz`` siblings in the same
    prefix are the GR00T patch geometry and would silently mislabel the pi arm. ~22 MB / 7500 files.

    IDEMPOTENT. A dir that already holds >0 matching files is left alone (retry/resume re-enters this
    function). A sync that lands ZERO files is FATAL: an empty label root would train the aux on
    nothing but the loader's KeyError, or worse, on an all-invalid mask that quietly zeroes the term.

    Single-process only (asserted by the caller via jax_processes), so there is no rank coordination
    here — one process owns the destination directory.
    """
    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    base = Path(feats_root).parent if feats_root else Path(os.environ.get("TMPDIR", "/tmp"))
    dest = base / "wsm_salient_labels"
    existing = len(list(dest.rglob(_SALIENT_LABEL_GLOB))) if dest.is_dir() else 0
    if existing:
        print(f"[pi-salient] labels already staged: {dest} ({existing} files)", flush=True)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            labels_s3.rstrip("/"),
            str(dest),
            "--exclude",
            "*",
            "--include",
            _SALIENT_LABEL_S3_FILTER,
            "--only-show-errors",
        ],
        check=True,
    )
    count = len(list(dest.rglob(_SALIENT_LABEL_GLOB)))
    if count == 0:
        raise RuntimeError(
            f"salient label sync from {labels_s3} landed 0 files matching {_SALIENT_LABEL_GLOB} in "
            f"{dest} — refusing to train the deliberative-supervision aux with no labels"
        )
    print(f"[pi-salient] staged {count} label files: {labels_s3} -> {dest}", flush=True)
    return dest


# Matches BOTH caption artifacts: the R3/R4 `capemb` cache and the R3b `capcls` cache
# (which adds seg_id). One glob, so a new artifact layout cannot silently count zero.
_CAPTION_EMB_GLOB = "ep_*.cap*.npz"
_CAPTION_EMB_S3_FILTER = "*ep_*.cap*.npz"


def _stage_caption_embeddings(emb_s3: str) -> Path:
    """Sync the R3/R4 caption-embedding cache node-local and return its root.

    Same contract as `_stage_salient_labels`: single-owner, idempotent, and a sync that lands ZERO
    files is FATAL rather than a silently all-invalid language target. ~160 MB / 7500 files.
    """
    if not emb_s3:
        raise ValueError("the language arm requires model.caption_emb_s3 (or WSM_H13_CAPTION_EMB_S3)")
    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    base = Path(feats_root).parent if feats_root else Path(os.environ.get("TMPDIR", "/tmp"))
    dest = base / "wsm_caption_emb"
    existing = len(list(dest.rglob(_CAPTION_EMB_GLOB))) if dest.is_dir() else 0
    if existing:
        print(f"[pi-lang] caption embeddings already staged: {dest} ({existing} files)", flush=True)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            emb_s3.rstrip("/"),
            str(dest),
            "--exclude",
            "*",
            "--include",
            _CAPTION_EMB_S3_FILTER,
            "--only-show-errors",
        ],
        check=True,
    )
    count = len(list(dest.rglob(_CAPTION_EMB_GLOB)))
    if count == 0:
        raise RuntimeError(
            f"caption-embedding sync from {emb_s3} landed 0 files matching {_CAPTION_EMB_GLOB} in "
            f"{dest} - refusing to train the language aux with no targets"
        )
    print(f"[pi-lang] staged {count} caption-embedding files: {emb_s3} -> {dest}", flush=True)
    return dest


def _optional_positive_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _prefix_llm_freeze_filter():
    """Freeze ONLY the prefix Gemma-2B (SigLIP + action expert train) — used by WSM stages; the
    official base-VLA recipe freezes nothing (freeze: none -> nnx.Nothing())."""
    import flax.nnx as nnx
    import openpi.shared.nnx_utils as nnx_utils

    return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))


def _wsm_modulator_only_freeze_filter():
    """WSM-conditioned finetune: train ONLY the zero-init TokenModulator; freeze the whole pi0.5
    backbone + action expert. trainable = All(Param, Not(freeze)) => only wsm_modulator.* params."""
    import flax.nnx as nnx
    import openpi.shared.nnx_utils as nnx_utils

    return nnx.Not(nnx_utils.PathRegex(".*wsm_modulator.*"))


def build_train_config(cfg, gs):
    """(TrainConfigView, GroupedSoup) -> openpi TrainConfig. Imports openpi/jax (pi05 venv only).

    Weight source priority: train.weight_loader (pretrain: gs:// base) -> train.init_from
    (finetune: phase-1 checkpoint) -> the published pi05 base.
    """
    # S3 JEPA horizon k. Exported BEFORE the openpi imports below: the dataloader module reads it at
    # IMPORT time (groot_openpi_dataset._WSM_JEPA_NUM_FUTURES) and openpi.training.config imports
    # that module, so a yaml-only k would leave the loader shipping k=1 targets. Env wins over yaml,
    # as with every other knob here; the consistency check after the imports catches import-order
    # breakage rather than letting the two ks silently disagree.
    # Action norm-stat nav split (see export_norm_split_nav). Exported before the openpi imports
    # for the same reason as the knobs below: it is read on the dataloader side of the fork.
    norm_split_nav = export_norm_split_nav(cfg)
    print(f"[pi05] norm_split_nav={norm_split_nav}", flush=True)

    jepa_num_futures = int(_JEPA_NUM_FUTURES_ENV or cfg.model.get("jepa_num_futures", 1))
    if jepa_num_futures < 1:
        raise ValueError(f"jepa_num_futures must be >= 1, got {jepa_num_futures}")
    if os.environ.get("WSM_JEPA", "0") == "1":
        os.environ["WSM_JEPA_NUM_FUTURES"] = str(jepa_num_futures)

    # STEERING conditioner selection for the tanh interface (s1). "tanh" is the shipped MLP read and
    # every existing arm resolves to it; "gated_deltanet" (and its PTRM variant, which wraps the same
    # read in a recursive core) swaps the module that fills the SAME wsm_tanh_cond subtree and
    # additionally needs a real window. Exported here for the same reason as
    # the JEPA horizon above: the dataloader freezes WSM_K_WINDOW at IMPORT, so a yaml-only
    # cond_window would leave the loader shipping K=1 rows to a K>1 conditioner. The launcher pins
    # WSM_K_WINDOW=1 for every workspace arm, so the recipe (not the env) is authoritative here; the
    # post-import check below catches import-order breakage rather than letting the two Ks disagree.
    cond_type = str(os.environ.get("WSM_COND_TYPE") or cfg.model.get("cond_type", "tanh"))
    if cond_type not in _COND_TYPES:
        raise ValueError(f"cond_type must be one of {_COND_TYPES}, got {cond_type!r}")
    cond_window = int(os.environ.get("WSM_COND_WINDOW") or cfg.model.get("cond_window", 1))
    if cond_window < 1:
        raise ValueError(f"cond_window must be >= 1, got {cond_window}")
    cond_num_heads = int(os.environ.get("WSM_COND_HEADS") or cfg.model.get("cond_heads", 2))
    cond_head_dim = int(os.environ.get("WSM_COND_HEAD_DIM") or cfg.model.get("cond_head_dim", 256))
    # History-intervention dropout (causal-confusion wave): a TRAIN-ONLY per-sample, per-element
    # deletion of HISTORICAL omega-window elements inside the deltanet recurrence (the newest element
    # is never deleted; nothing is ever deleted at serve). It adds no parameters and no dataloader
    # dependency, so unlike cond_window it does not have to beat the openpi import — it is resolved
    # here purely to keep every conditioner knob in one block. Env wins over yaml, as everywhere else.
    # 0.0 (the default, and the value every pre-existing recipe resolves to) is byte-identical.
    cond_history_dropout = float(
        os.environ.get("WSM_COND_HISTORY_DROPOUT") or cfg.model.get("cond_history_dropout", 0.0)
    )
    if not 0.0 <= cond_history_dropout <= 0.9:
        raise ValueError(f"cond_history_dropout must be in [0.0, 0.9], got {cond_history_dropout}")
    if cond_history_dropout > 0.0 and cond_type != "gated_deltanet":
        raise ValueError(
            "cond_history_dropout deletes elements of the gated-DeltaNet omega window; it is "
            f"meaningless with cond_type={cond_type!r}"
        )
    # PTRM (H9): recursion depth T and the Q-loss lambda. Resolved in this block with every other
    # conditioner knob, env-first like the rest; unlike cond_window they have no dataloader side, so
    # they do not have to beat the openpi import. The defaults reproduce the design's v1 recipe.
    ptrm_steps = int(os.environ.get("WSM_PTRM_STEPS") or cfg.model.get("ptrm_steps", 4))
    ptrm_q_weight = float(os.environ.get("WSM_PTRM_Q_WEIGHT") or cfg.model.get("ptrm_q_weight", 0.1))
    if ptrm_steps < 1:
        raise ValueError(f"ptrm_steps must be >= 1, got {ptrm_steps}")
    if not ptrm_q_weight >= 0.0:
        raise ValueError(f"ptrm_q_weight must be >= 0, got {ptrm_q_weight}")
    deltanet_on = cond_type in _WINDOWED_COND_TYPES and os.environ.get("WSM_TANH", "0") == "1"
    if deltanet_on:
        os.environ["WSM_K_WINDOW"] = str(cond_window)
        os.environ["WSM_COND_HISTORY_DROPOUT"] = repr(cond_history_dropout)

    # S3-SALIENT ("deliberative supervision"): the next-keyframe salient-patch BCE aux. Resolved and
    # exported HERE, before the openpi imports below, for the same reason as the JEPA horizon and the
    # deltanet window — the dataloader freezes WSM_SALIENT_TARGETS at IMPORT, and
    # openpi.training.config imports it, so a yaml-only knob would leave the loader shipping no
    # targets while the model demanded them. Env wins over yaml, as with every other knob here; the
    # post-import consistency check catches import-order breakage instead of letting the two disagree.
    salient_on = os.environ.get("WSM_SALIENT", "0") == "1" or bool(cfg.model.get("salient", False))
    salient_weight = float(os.environ.get("WSM_SALIENT_WEIGHT") or cfg.model.get("salient_weight", 1.0))
    salient_labels_s3 = str(
        os.environ.get("WSM_SALIENT_LABELS_S3") or cfg.model.get("labels_s3", _SALIENT_LABELS_S3_DEFAULT)
    )
    # H13 (live/joint WSM, aug_12/h13_joint_wsm_tree.md). Resolved and exported in this same
    # pre-import block as every other loader-facing knob: the R1 decode consumes the SAME sealed
    # keypatch labels the salient arm used (so the loader's WSM_SALIENT_TARGETS gate must be on), and
    # R2 additionally needs the loader to ship the t+k FRAME, which is frozen at import as
    # WSM_H13_FUTURE_FRAMES. Env wins over yaml, as everywhere here; the post-import consistency
    # checks below catch import-order breakage instead of letting the two sides disagree.
    h13_on = os.environ.get("WSM_H13", "0") == "1" or bool(cfg.model.get("h13", False))
    h13_jepa_on = os.environ.get("WSM_H13_JEPA", "0") == "1" or bool(cfg.model.get("h13_jepa", False))
    if h13_jepa_on and not h13_on:
        raise ValueError("model.h13_jepa is the R2 term on top of the R1 live encoder; set model.h13 too")
    h13_dec_weight = float(os.environ.get("WSM_H13_DEC_WEIGHT") or cfg.model.get("h13_dec_weight", 0.5))
    h13_sigreg_weight = float(os.environ.get("WSM_H13_SIGREG_WEIGHT") or cfg.model.get("h13_sigreg_weight", 0.05))
    h13_jepa_weight = float(os.environ.get("WSM_H13_JEPA_WEIGHT") or cfg.model.get("h13_jepa_weight", 0.1))
    h13_w_dim = int(os.environ.get("WSM_H13_W_DIM") or cfg.model.get("h13_w_dim", 512))
    h13_enc_layers = int(os.environ.get("WSM_H13_ENC_LAYERS") or cfg.model.get("h13_enc_layers", 4))
    h13_enc_heads = int(os.environ.get("WSM_H13_ENC_HEADS") or cfg.model.get("h13_enc_heads", 8))
    h13_future_k = int(os.environ.get("WSM_H13_FUTURE_K") or cfg.model.get("h13_future_k", 1))
    h13_future_stride = int(os.environ.get("WSM_H13_FUTURE_STRIDE") or cfg.model.get("h13_future_stride", 8))
    # R3/R4 language decode. Like every other loader-facing gate, exported BEFORE the openpi import
    # (the dataloader freezes WSM_H13_LANG_TARGETS at import) and re-checked after it.
    h13_lang_on = os.environ.get("WSM_H13_LANG", "0") == "1" or bool(cfg.model.get("h13_lang", False))
    if h13_lang_on and not h13_on:
        raise ValueError("model.h13_lang is the R3/R4 term on top of the R1 live encoder; set model.h13 too")
    h13_lang_weight = float(os.environ.get("WSM_H13_LANG_WEIGHT") or cfg.model.get("h13_lang_weight", 0.5))
    # R3b: CE over the caption vocabulary. Same loader gate + staged cache as the embedding head
    # (the capcls artifact carries seg_id alongside emb), only the objective differs.
    h13_lang_cls_on = os.environ.get("WSM_H13_LANG_CLS", "0") == "1" or bool(cfg.model.get("h13_lang_cls", False))
    h13_lang_vocab = int(os.environ.get("WSM_H13_LANG_VOCAB") or cfg.model.get("h13_lang_vocab", 8661))
    if h13_lang_cls_on and not h13_on:
        raise ValueError("model.h13_lang_cls is an R3b term on the live encoder; set model.h13 too")
    h13_lang_temperature = float(
        os.environ.get("WSM_H13_LANG_TEMPERATURE") or cfg.model.get("h13_lang_temperature", 0.07)
    )
    h13_caption_emb_s3 = str(os.environ.get("WSM_H13_CAPTION_EMB_S3") or cfg.model.get("caption_emb_s3", ""))
    h13_caption_emb_root = ""
    if h13_lang_on or h13_lang_cls_on:
        os.environ["WSM_H13_LANG_TARGETS"] = "1"
        h13_caption_emb_root = os.environ.get("WSM_H13_CAPTION_EMB_ROOT") or str(
            _stage_caption_embeddings(h13_caption_emb_s3)
        )
        os.environ["WSM_H13_CAPTION_EMB_ROOT"] = h13_caption_emb_root
    if h13_jepa_on:
        # The loader freezes these at IMPORT (groot_openpi_dataset._WSM_H13_FUTURE_*), and
        # openpi.training.config imports that module — a yaml-only knob would leave the batch with no
        # future frame while the model demanded one.
        os.environ["WSM_H13_FUTURE_FRAMES"] = "1"
        os.environ["WSM_H13_FUTURE_K"] = str(h13_future_k)
        os.environ["WSM_H13_FUTURE_STRIDE"] = str(h13_future_stride)
    # Both the salient arm and H13 decode the SAME sealed 192-way label store, so the staging and the
    # loader gate are shared; only WHICH head reads them differs.
    labels_on = salient_on or h13_on
    salient_labels_root = ""
    if labels_on:
        os.environ["WSM_SALIENT_TARGETS"] = "1"
        # A pre-set root means someone already staged the labels (an entry script, or a test/local
        # run pointing at a fixture): honor it and skip the sync entirely. Otherwise stage from S3.
        salient_labels_root = os.environ.get("WSM_SALIENT_LABELS_ROOT") or str(
            _stage_salient_labels(salient_labels_s3)
        )
        os.environ["WSM_SALIENT_LABELS_ROOT"] = salient_labels_root

    import flax.nnx as nnx
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders

    # Post-import nav-split check for the Stage-S arms (s0/s1/s2/s3), matching the Stage-Q path and
    # the loader-facing knob checks below. `norm_split_nav` was exported at the top of this function,
    # before these imports; this confirms the openpi archive we just imported actually implements the
    # knob, so an archive predating it fails here instead of silently writing OLD single-blob action
    # norm stats into assets/norm_stats.json under a recipe that claims the split. That silent
    # mismatch is the one the n-wave forensic audit had to rule out by byte-comparing all 8 blobs.
    assert_norm_split_nav_loader(norm_split_nav)

    m, o, t = cfg.model, cfg.optim, cfg.train
    defer_image_resize = deferred_image_resize_enabled(cfg.data.raw)
    # Workspace features and policy interfaces are separate, explicit gates. Supplying a feature root no
    # longer silently selects the failed direct-token path. WSM_CFG is retained only for historical
    # checkpoint reproduction; new Stage-S runs use WSM_CFG2 or WSM_TANH.
    feats_on = bool(os.environ.get("WSM_POLICY_FEATS_ROOT"))
    requested = {
        "legacy_token_injection": os.environ.get("WSM_LEGACY_TOKEN_INJECTION", "0") == "1",
        "legacy_cfg": os.environ.get("WSM_CFG", "0") == "1",
        "current_cfg": os.environ.get("WSM_CFG2", "0") == "1",
        "tanh": os.environ.get("WSM_TANH", "0") == "1",
        # S3 (packet 08): train-time-only JEPA+SigReg aux target — a workspace CONSUMER (needs the
        # omega cache) but not a READ (nothing at inference), so it joins the exclusion set here.
        "jepa": os.environ.get("WSM_JEPA", "0") == "1",
        # H13: a train-only aux whose workspace latent is produced LIVE off this model's own backbone
        # features. It is a workspace interface for provenance purposes (it defines the checkpoint's
        # extra subtrees and the loader contract) but NOT an omega-cache consumer — see the
        # feature-root split below.
        "h13": h13_on,
    }
    enabled = [name for name, value in requested.items() if value]
    # S5 combo arm: `tanh` (an inference READ on adarms_cond) + `jepa` (a train-only aux TARGET at
    # the action-expert penultimate) is the ONE sanctioned pair — they touch disjoint tensors and the
    # aux head is dropped at serve, so the checkpoint still serves through the plain tanh contract.
    # Selected only by the recipe (model.jepa_aux), never by a bare env flip; every other pair fails.
    if len(enabled) > 1 and set(enabled) not in _SANCTIONED_COMBOS:
        raise ValueError(
            f"workspace policy interfaces are mutually exclusive, got {enabled}; sanctioned "
            f"combinations are {[sorted(c) for c in _SANCTIONED_COMBOS]}"
        )
    # The feature-root requirement belongs to the OMEGA CONSUMERS only. H13 builds its own w inside
    # the training graph from the SigLIP tokens the policy already computes, so it needs no cache —
    # and must not be handed one, because staging an omega cache it never reads would put a false
    # encoder_id/feature-manifest provenance block on a run that does not depend on either.
    omega_consumers = [name for name in enabled if name != "h13"]
    if omega_consumers and not feats_on:
        raise ValueError(f"workspace interface {omega_consumers[0]} requires WSM_POLICY_FEATS_ROOT")
    if feats_on and not omega_consumers:
        raise ValueError(
            "WSM_POLICY_FEATS_ROOT is set but no omega-consuming interface was selected; use "
            "WSM_CFG2=1 or WSM_TANH=1 (legacy reproduction requires WSM_CFG=1 or "
            "WSM_LEGACY_TOKEN_INJECTION=1). The H13 live encoder reads no cache."
        )
    wsm_on = feats_on and requested["legacy_token_injection"]
    wsm_cfg_on = feats_on and requested["legacy_cfg"]
    wsm_cfg2_on = feats_on and requested["current_cfg"]
    wsm_tanh_on = feats_on and requested["tanh"]
    wsm_jepa_on = feats_on and requested["jepa"]
    if wsm_jepa_on and os.environ.get("WSM_JEPA_TARGETS") != "1":
        raise ValueError(
            "WSM_JEPA=1 requires WSM_JEPA_TARGETS=1 so the loader ships wsm_w_target (never "
            "wsm_w_window) — refusing a JEPA arm silently training without its targets"
        )
    # S3 regularizer selection: which isotropy regularizer runs on the action-expert penultimate.
    # "sigreg" = the shipped Epps-Pulley sketch (default; existing S3 configs are unchanged);
    # "visreg" = VISReg (arXiv:2606.02572), sliced-Wasserstein shape sketch + explicit variance term,
    # whose gradient does NOT vanish under collapse. YAML `model.jepa_regularizer` is the source of
    # truth; WSM_JEPA_REGULARIZER overrides it for one-off arms.
    jepa_regularizer = str(os.environ.get("WSM_JEPA_REGULARIZER") or m.get("jepa_regularizer", "sigreg"))
    if jepa_regularizer not in ("sigreg", "visreg"):
        raise ValueError(f"jepa_regularizer must be 'sigreg' or 'visreg', got {jepa_regularizer!r}")
    # The two aux lambdas. Both were previously pinned at the Pi0Config defaults; they are yaml knobs
    # now so a lambda sweep is a config change, not a code change (defaults reproduce every prior run).
    sigreg_weight = float(os.environ.get("WSM_SIGREG_WEIGHT") or m.get("sigreg_weight", 0.05))
    jepa_weight = float(os.environ.get("WSM_JEPA_WEIGHT") or m.get("jepa_weight", 1.0))
    visreg_weight = float(os.environ.get("WSM_VISREG_WEIGHT") or m.get("visreg_weight", 0.05))
    visreg_slices = int(os.environ.get("WSM_VISREG_SLICES") or m.get("visreg_slices", 128))
    visreg_split = (
        float(m.get("visreg_scale_weight", 1.0)),
        float(m.get("visreg_shape_weight", 1.0)),
        float(m.get("visreg_center_weight", 1.0)),
    )
    if wsm_jepa_on:
        # The s3 forensics' #2 open item: an aux arm must state its own weights in the log.
        import openpi.groot_utils.groot_openpi_dataset as _groot_dataset

        loader_k = int(_groot_dataset._WSM_JEPA_NUM_FUTURES)  # noqa: SLF001  (the loader's frozen k)
        if loader_k != jepa_num_futures:
            raise ValueError(
                f"dataloader k={loader_k} but the recipe asks for k={jepa_num_futures}; the loader "
                "froze WSM_JEPA_NUM_FUTURES at import, so the targets would not match the head"
            )
        # S5 combo: the conditioner needs wsm_w_window and the aux needs wsm_w_target from the SAME
        # batch. The loader froze that either/or at import too, so a stale/unset gate would either
        # starve the deltanet (no window -> fail-loud in compute_loss) or ship a window to a
        # target-only arm. Fail here, not 60k steps later.
        loader_both = bool(_groot_dataset._WSM_JEPA_WITH_WINDOW)  # noqa: SLF001
        if wsm_tanh_on and not loader_both:
            raise ValueError(
                "the combo arm (tanh read + jepa aux) needs WSM_JEPA_WITH_WINDOW=1 exported BEFORE "
                "the openpi import; the dataloader froze it at 0 and would ship no wsm_w_window"
            )
        if loader_both and not wsm_tanh_on:
            raise ValueError(
                "WSM_JEPA_WITH_WINDOW=1 without the tanh interface: a target-only S3 arm must never "
                "ship wsm_w_window (it would change nothing but the batch, silently)"
            )
        print(
            f"[pi-jepa] regularizer={jepa_regularizer} lambda_jepa={jepa_weight} "
            f"lambda_sigreg={sigreg_weight} num_futures={jepa_num_futures} "
            f"visreg_w={visreg_weight} visreg_slices={visreg_slices} "
            f"visreg_split(scale/shape/center)={visreg_split}",
            flush=True,
        )
    if h13_on:
        # The H13 arm states its own loss terms in the log — the s3 forensics' #2 open item, applied
        # to a joint arm: a run whose log omits the lambdas cannot be told apart after the fact.
        import openpi.groot_utils.groot_openpi_dataset as _groot_dataset

        for attribute in ("_WSM_H13_FUTURE_FRAMES", "_WSM_H13_FUTURE_K", "_WSM_H13_FUTURE_STRIDE"):
            if not hasattr(_groot_dataset, attribute):
                raise ValueError(
                    f"the imported openpi dataloader has no {attribute}: this archive predates H13 "
                    "and could never ship the live encoder's future frame"
                )
        if not _groot_dataset._WSM_SALIENT_TARGETS:  # noqa: SLF001  (the loader's frozen gate)
            raise ValueError(
                "the dataloader froze WSM_SALIENT_TARGETS=0 at import but the H13 recipe decodes the "
                "keypatch labels through w; openpi was imported before this module exported the gate"
            )
        loader_future = bool(_groot_dataset._WSM_H13_FUTURE_FRAMES)  # noqa: SLF001
        if h13_jepa_on and not loader_future:
            raise ValueError(
                "WSM_H13_JEPA=1 requires WSM_H13_FUTURE_FRAMES=1 exported BEFORE the openpi import; "
                "the dataloader froze it at 0 and would ship no t+k frame for the live target"
            )
        if loader_future and not h13_jepa_on:
            raise ValueError(
                "WSM_H13_FUTURE_FRAMES=1 without the R2 alignment term: the loader would pay for an "
                "extra video decode per sample that nothing in the loss reads"
            )
        if h13_jepa_on:
            loader_k = int(_groot_dataset._WSM_H13_FUTURE_K)  # noqa: SLF001
            loader_stride = int(_groot_dataset._WSM_H13_FUTURE_STRIDE)  # noqa: SLF001
            if (loader_k, loader_stride) != (h13_future_k, h13_future_stride):
                raise ValueError(
                    f"dataloader (k, stride) = ({loader_k}, {loader_stride}) but the recipe asks for "
                    f"({h13_future_k}, {h13_future_stride}); the loader froze both at import, so the "
                    "shipped future frame would not be the one the recipe claims"
                )
        if _groot_dataset._WSM_SALIENT_LABELS_ROOT != salient_labels_root:  # noqa: SLF001
            raise ValueError(
                f"dataloader label root {_groot_dataset._WSM_SALIENT_LABELS_ROOT!r} != staged "  # noqa: SLF001
                f"{salient_labels_root!r}; the loader froze the root at import"
            )
        # Same single-owner staging contract as the salient arm (one process syncs and owns the dir).
        expected_processes = _optional_positive_env("WSM_EXPECTED_JAX_PROCESSES")
        if expected_processes not in (None, 1):
            raise ValueError(
                "H13 stages its keypatch labels from a single process, but the run declares "
                f"WSM_EXPECTED_JAX_PROCESSES={expected_processes}"
            )
        h13_label_files = len(list(Path(salient_labels_root).rglob(_SALIENT_LABEL_GLOB)))
        if h13_label_files == 0:
            raise RuntimeError(
                f"H13 found 0 label files under {salient_labels_root}; the decode term would train "
                "on nothing but an all-invalid mask"
            )
        if h13_lang_on or h13_lang_cls_on:
            if not hasattr(_groot_dataset, "_WSM_H13_LANG_TARGETS"):
                raise ValueError(
                    "the imported openpi dataloader has no _WSM_H13_LANG_TARGETS: this archive "
                    "predates R3/R4 and could never ship the caption target"
                )
            if not _groot_dataset._WSM_H13_LANG_TARGETS:  # noqa: SLF001
                raise ValueError(
                    "the dataloader froze WSM_H13_LANG_TARGETS=0 at import but the recipe selects "
                    "the language arm; openpi was imported before this module exported the gate"
                )
            if _groot_dataset._WSM_H13_CAPTION_EMB_ROOT != h13_caption_emb_root:  # noqa: SLF001
                raise ValueError(
                    f"dataloader caption root {_groot_dataset._WSM_H13_CAPTION_EMB_ROOT!r} != "  # noqa: SLF001
                    f"staged {h13_caption_emb_root!r}; the loader froze the root at import"
                )
            caption_files = len(list(Path(h13_caption_emb_root).rglob(_CAPTION_EMB_GLOB)))
            if caption_files == 0:
                raise RuntimeError(f"language arm found 0 caption-embedding files under {h13_caption_emb_root}")
            print(
                f"[pi-lang] caption_emb files={caption_files} lambda_lang={h13_lang_weight} "
                f"temperature={h13_lang_temperature} target=frozen_pi05_text_tower "
                f"lookup=active_segment_interval emb_s3={h13_caption_emb_s3}",
                flush=True,
            )
        print(
            f"[pi-h13] labels files={h13_label_files} "
            f"live_encoder=on w_dim={h13_w_dim} layers={h13_enc_layers} "
            f"heads={h13_enc_heads} lambda_dec={h13_dec_weight} lambda_sigreg={h13_sigreg_weight} "
            f"jepa={'on' if h13_jepa_on else 'off'} lambda_jepa={h13_jepa_weight} "
            f"future(k={h13_future_k}, stride={h13_future_stride}) "
            f"labels_s3={salient_labels_s3} labels_root={salient_labels_root} "
            f"stop_grad=NONE (gradients flow into the backbone)",
            flush=True,
        )
    if salient_on:
        if not wsm_jepa_on:
            raise ValueError(
                "the salient arm rides the jepa interface; select it with WSM_JEPA=1 "
                "(--interface jepa) and zero the jepa lambdas in the recipe"
            )
        # The label staging above is single-owner: one process syncs and owns the destination dir.
        # Stage-S is a one-process job, and the launcher pins that contract in the env — check it
        # here rather than calling jax.process_count(), which would initialize the backend early.
        expected_processes = _optional_positive_env("WSM_EXPECTED_JAX_PROCESSES")
        if expected_processes not in (None, 1):
            raise ValueError(
                f"the salient arm stages its labels from a single process, but the run declares "
                f"WSM_EXPECTED_JAX_PROCESSES={expected_processes}"
            )
        # Same post-import contract as the JEPA horizon: the loader froze WSM_SALIENT_TARGETS at
        # import, so if openpi was imported before our export the gate is OFF and the arm would ship
        # no targets while the model demands them. Fail here, not 60k steps later.
        import openpi.groot_utils.groot_openpi_dataset as _groot_dataset

        if not _groot_dataset._WSM_SALIENT_TARGETS:  # noqa: SLF001  (the loader's frozen gate)
            raise ValueError(
                "the dataloader froze WSM_SALIENT_TARGETS=0 at import but the recipe selects the "
                "salient arm; openpi was imported before this module exported the gate"
            )
        if _groot_dataset._WSM_SALIENT_LABELS_ROOT != salient_labels_root:  # noqa: SLF001
            raise ValueError(
                f"dataloader label root {_groot_dataset._WSM_SALIENT_LABELS_ROOT!r} != staged "  # noqa: SLF001
                f"{salient_labels_root!r}; the loader froze the root at import"
            )
        label_files = len(list(Path(salient_labels_root).rglob(_SALIENT_LABEL_GLOB)))
        # The aux arm states its own weight in the log (the s3 forensics' #2 open item), plus the
        # exact label provenance: these labels are NOT content-addressed in the study store.
        print(
            f"[pi-salient] weight={salient_weight} labels_s3={salient_labels_s3} "
            f"labels_root={salient_labels_root} files={label_files} "
            f"(riding jepa: lambda_jepa={jepa_weight} lambda_sigreg={sigreg_weight})",
            flush=True,
        )
    # New Stage-S readers select newest omega_t only; K=2 remains a legacy-checkpoint contract.
    wsm_k = int(os.environ.get("WSM_K_WINDOW", "1" if (wsm_cfg2_on or wsm_tanh_on) else "2"))
    if cond_type != "tanh" and not wsm_tanh_on:
        raise ValueError(
            f"cond_type={cond_type!r} selects the module behind the wsm_tanh_cond subtree and "
            "requires the tanh interface (WSM_TANH=1)"
        )
    if wsm_tanh_on:
        if deltanet_on:
            import openpi.groot_utils.groot_openpi_dataset as _groot_dataset

            loader_k = int(_groot_dataset._WSM_K_WINDOW)  # noqa: SLF001  (the loader's frozen K)
            if loader_k != cond_window or wsm_k != cond_window:
                raise ValueError(
                    f"dataloader K={loader_k} / config K={wsm_k} but the deltanet recipe asks for "
                    f"cond_window={cond_window}; the loader froze WSM_K_WINDOW at import, so the "
                    "shipped window would not match the trained conditioner"
                )
        print(
            f"[pi-tanh] cond_type={cond_type} cond_window={cond_window} "
            + (f"heads={cond_num_heads} head_dim={cond_head_dim} " if deltanet_on else "")
            + f"loader_K={wsm_k} gate_init={os.environ.get('WSM_TANH_GATE_INIT', '0.001')} "
            + (f"history_dropout={cond_history_dropout} " if deltanet_on else "")
            # The PTRM arm states its own two numbers, for the same reason every aux arm does: a log
            # that omits them cannot be told apart from its deltanet parent after the fact.
            + (
                f"ptrm_steps={ptrm_steps} ptrm_q_weight={ptrm_q_weight} ptrm_eval_noise=inference_only "
                if cond_type == "gated_deltanet_ptrm"
                else ""
            )
            + f"jepa_aux={'on' if wsm_jepa_on else 'off'}",
            flush=True,
        )
    if wsm_tanh_on and wsm_jepa_on:
        # The combo arm states BOTH active loss terms in one line so the log proves the run is not a
        # silently-degraded single-axis arm.
        print(
            f"[pi-combo] loss = flow + tanh_read(cond_type={cond_type}, window={cond_window}) "
            f"+ jepa_aux(lambda_jepa={jepa_weight}, lambda_{jepa_regularizer}="
            f"{sigreg_weight if jepa_regularizer == 'sigreg' else visreg_weight}, "
            f"k={jepa_num_futures})",
            flush=True,
        )
    if wsm_on:
        freeze = _wsm_modulator_only_freeze_filter()
    elif wsm_cfg_on or wsm_cfg2_on or wsm_tanh_on:
        # Every action-expert interface is a FULL finetune (same trainable surface as S0) plus its small
        # conditioner. Only the historical direct-token reproduction remains modulator-only.
        freeze = nnx.Nothing()
    else:
        freeze = nnx.Nothing() if str(m.get("freeze", "none")) == "none" else _prefix_llm_freeze_filter()
    peak_lr = float(o["lr"])
    # WSM_INIT_FROM (set by the SageMaker FT entry to the node-local synced ckpt) wins, since orbax
    # cannot read the s3:// init_from recorded in the YAML.
    ckpt = os.environ.get("WSM_INIT_FROM") or t.get("weight_loader") or t.get("init_from") or _PI05_BASE
    num_train_steps = int(os.environ.get("WSM_MAX_STEPS") or t.get("max_steps", 80000))
    final_only = os.environ.get("WSM_FINAL_ONLY_CHECKPOINTS", "0") == "1"
    if final_only and os.environ.get("WSM_RESUME") == "1":
        raise ValueError("final-only Stage-S jobs are single-attempt and forbid WSM_RESUME")
    save_interval = (
        num_train_steps if final_only else int(os.environ.get("WSM_SAVE_INTERVAL") or t.get("save_interval", 5000))
    )
    keep_period = None if final_only else int(os.environ.get("WSM_KEEP_PERIOD", "5000"))
    # Back-fill EVERY newly initialized interface subtree when loading the shared pretrain checkpoint.
    # `_load_weights_and_validate` compares key SETS, so one un-excused subtree kills the run at
    # init_train_state — after the node has already paid for the multi-GB init fetch (the A1 lesson,
    # 2026-08-12). Built additively from the enabled flags rather than as a ternary chain so a new arm
    # cannot be added without its subtree; every pre-existing arm resolves to its historical string
    # (the base arm keeps the literal ".*lora.*", not an equivalent-but-different regex).
    missing_subtrees = ["lora"]
    if wsm_cfg_on:
        missing_subtrees.append("wsm_cfg_cond")
    if wsm_cfg2_on:
        missing_subtrees.append("wsm_cfg2_cond")
    if wsm_tanh_on:
        missing_subtrees.append("wsm_tanh_cond")
    if wsm_jepa_on:
        missing_subtrees.append("wsm_jepa_head")
    if salient_on:
        missing_subtrees.append("wsm_salient_head")
    if h13_on:
        missing_subtrees += ["wsm_enc", "wsm_dec"]
    if h13_jepa_on:
        missing_subtrees.append("wsm_h13_jepa_head")
    if h13_lang_on:
        missing_subtrees.append("wsm_lang_head")
    if h13_lang_cls_on:
        missing_subtrees.append("wsm_lang_cls")
    if wsm_on:
        missing_subtrees.append("wsm_modulator")
    missing_regex = ".*lora.*" if missing_subtrees == ["lora"] else ".*(" + "|".join(missing_subtrees) + ").*"
    print(f"[pi05] weight-loader missing_regex={missing_regex!r}", flush=True)
    return _config.TrainConfig(
        name=str(m.get("config_name", "pi05_robocasa")),
        exp_name=str(t.get("exp_name", "pi05_rc365")),
        project_name=os.environ.get("WANDB_PROJECT", "wsm-robocasa"),
        model=pi0_config.Pi0Config(
            pi05=bool(m.get("pi05", True)),
            max_token_len=int(m.get("max_token_len", 200)),
            wsm=wsm_on,
            wsm_k_window=wsm_k,
            wsm_cfg=wsm_cfg_on,
            wsm_cfg2=wsm_cfg2_on,
            wsm_tanh=wsm_tanh_on,
            wsm_cond_type=cond_type,
            wsm_cond_window=cond_window,
            wsm_cond_num_heads=cond_num_heads,
            wsm_cond_head_dim=cond_head_dim,
            wsm_cond_history_dropout=cond_history_dropout,
            wsm_ptrm_steps=ptrm_steps,
            wsm_ptrm_q_weight=ptrm_q_weight,
            wsm_jepa=wsm_jepa_on,
            wsm_jepa_weight=jepa_weight,
            wsm_jepa_sigreg_weight=sigreg_weight,
            wsm_jepa_num_futures=jepa_num_futures,
            wsm_jepa_regularizer=jepa_regularizer,
            wsm_jepa_visreg_weight=visreg_weight,
            wsm_jepa_visreg_slices=visreg_slices,
            wsm_jepa_visreg_scale_weight=visreg_split[0],
            wsm_jepa_visreg_shape_weight=visreg_split[1],
            wsm_jepa_visreg_center_weight=visreg_split[2],
            wsm_salient=salient_on,
            wsm_salient_weight=salient_weight,
            wsm_salient_labels=salient_labels_s3 if salient_on else "",
            wsm_h13=h13_on,
            wsm_h13_jepa=h13_jepa_on,
            wsm_h13_w_dim=h13_w_dim,
            wsm_h13_enc_layers=h13_enc_layers,
            wsm_h13_enc_heads=h13_enc_heads,
            wsm_h13_dec_weight=h13_dec_weight,
            wsm_h13_sigreg_weight=h13_sigreg_weight,
            wsm_h13_jepa_weight=h13_jepa_weight,
            wsm_h13_jepa_num_futures=h13_future_k,
            wsm_h13_labels=salient_labels_s3 if h13_on else "",
            wsm_h13_lang=h13_lang_on,
            wsm_h13_lang_weight=h13_lang_weight,
            wsm_h13_lang_temperature=h13_lang_temperature,
            wsm_h13_lang_cls=h13_lang_cls_on,
            wsm_h13_lang_cls_weight=float(
                os.environ.get("WSM_H13_LANG_CLS_WEIGHT") or cfg.model.get("h13_lang_cls_weight", 0.5)
            ),
            wsm_h13_lang_vocab=h13_lang_vocab,
            wsm_tanh_gate_init=float(os.environ.get("WSM_TANH_GATE_INIT", "0.001")),
            wsm_cfg_p_drop=float(os.environ.get("WSM_CFG_P_DROP", "0.2")),
            wsm_cfg_with_future=(wsm_cfg_on and os.environ.get("WSM_CFG_WITH_FUTURE", "0") == "1"),
        ),
        # Our soup + per-dataset balancing weights (None => openpi native size power-law = OFF/finetune).
        data=_config.LeRobotRobocasaDataConfig(
            data_dirs=gs.soup,
            dataset_weights=gs.pi05_weights,
            defer_image_resize_to_model_preprocess=defer_image_resize,
        ),
        # Back-fill the one newly initialized interface subtree when loading the shared S0 checkpoint.
        # Distinct names keep old wsm_cfg_cond checkpoints from being reinterpreted as current-only CFG2.
        weight_loader=weight_loaders.CheckpointWeightLoader(ckpt, missing_regex=missing_regex),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(o.get("warmup_steps", 1000)),
            peak_lr=peak_lr,
            decay_steps=int(o.get("decay_steps", t.get("max_steps", 80000))),
            decay_lr=float(o.get("decay_lr", peak_lr / 10)),
        ),
        freeze_filter=freeze,
        ema_decay=m.get("ema_decay"),
        # Resume-on-retry: WSM_RESUME=1 (set by the entry when it synced a prior attempt's checkpoints into
        # the local ckpt dir) -> openpi restores train_state from the latest COMPLETE orbax step instead of
        # restarting from step 0. overwrite stays False (mutually exclusive with resume); on a fresh start
        # WSM_RESUME is unset so the ckpt dir is created clean.
        resume=os.environ.get("WSM_RESUME") == "1",
        # POC override: WSM_MAX_STEPS / WSM_SAVE_INTERVAL (set by the co-located launcher) win over the YAML,
        # mirroring the GR00T path (_groot_common) so a small-task curve uses modest steps without a new YAML.
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        keep_period=keep_period,
        batch_size=int(t.get("batch_size", 256)),
        num_workers=int(t.get("num_workers", 32)),
        checkpoint_base_dir=os.environ.get("WSM_CKPT_BASE", str(t.get("output_dir", "./checkpoints/wsm_robocasa"))),
    )


def assert_weight_loader_covers(train_cfg) -> None:
    """Fail fast if the arm adds a param subtree its weight loader will not back-fill.

    Ported from the robocerebra bundle (`openpi/training/robocerebra_configs.py:138`) and
    generalized: instead of a hardcoded BASE_CKPT_SUBTREES list, the baseline is DERIVED by building
    the same Pi0Config with every workspace/robottt flag off. That makes the check self-maintaining —
    a future arm cannot add a subtree the baseline list forgot to mention.

    Model-only and jax.eval_shape-ONLY: no parameters are materialized, no checkpoint bytes are read,
    it costs a couple of seconds, and it runs BEFORE `init_train_state` touches the multi-GB init.
    That ordering is the whole point — arm A1 reached a node, paid for the entire setup, and died at
    `init_train_state` with `symmetric difference of key sets: {'wsm_tanh_cond'}` about 7 minutes in
    (2026-08-12, one node-hour).

    NEGATIVE CONTROL, run every time rather than once by hand: when the arm does add subtrees, the
    base-arm regex ``.*lora.*`` must FAIL to cover them. A check that passes for every possible regex
    is not a check, and this is the cheapest way to keep proving it does discriminate.
    """
    import dataclasses
    import re

    import jax
    from flax import nnx

    def subtrees(model_config) -> set[str]:
        return set(jax.eval_shape(lambda r: nnx.state(model_config.create(r)), jax.random.key(0)).to_pure_dict())

    model_config = train_cfg.model
    off = {
        field.name: False
        for field in dataclasses.fields(model_config)
        if field.type is bool or isinstance(getattr(model_config, field.name), bool)
    }
    # Only the interface switches are forced off; `pi05` and the like must survive or the baseline
    # would be a different architecture entirely.
    off = {name: False for name in off if name.startswith(("wsm", "robottt")) and name not in {"wsm_cfg_with_future"}}
    # Non-boolean interface SELECTORS must be reset too, not just the flags. `wsm_cond_type` is a
    # string: zeroing only the booleans leaves cond_type='gated_deltanet' with wsm_tanh=False, which
    # Pi0Config.__post_init__ rightly rejects ("only meaningful with wsm_tanh=True") — so the gate
    # itself would raise on every gdn8 arm before it could check anything. Caught by the R5/R8
    # recipe probe, 2026-08-15.
    base_config = dataclasses.replace(model_config, **off, wsm_cfg_with_future=False, wsm_cond_type="tanh")
    extra = sorted(subtrees(model_config) - subtrees(base_config))
    regex = getattr(train_cfg.weight_loader, "missing_regex", None)
    if extra and not regex:
        raise SystemExit(
            f"arm adds subtrees {extra} but its weight loader has no missing_regex; training would "
            "fail at init_train_state with a pytree structure mismatch"
        )
    if extra:
        pattern = re.compile(regex)
        uncovered = [s for s in extra if not pattern.fullmatch(f"{s}/dummy/kernel")]
        if uncovered:
            raise SystemExit(
                f"subtree(s) {uncovered} are absent from the pretrain checkpoint and NOT matched by "
                f"missing_regex={regex!r}. Training would fail at init_train_state with a pytree "
                "structure mismatch — after the init fetch has already been paid for."
            )
        control = re.compile(".*lora.*")
        if all(control.fullmatch(f"{s}/dummy/kernel") for s in extra):
            raise SystemExit(
                f"weight-loader check is inert: the base-arm regex '.*lora.*' already covers {extra}, "
                "so this gate could not distinguish a correct loader from a forgotten one"
            )
    # THE MERGE ITSELF, reproduced locally. Subtree-name coverage is necessary but NOT sufficient:
    # H13 canary C2 (`h13b-e3fabdd1cef58d4a`, 2026-08-12) passed the coverage check above and then
    # died at `init_train_state` — after fetching 12.5 GB — because `wsm_enc.blocks` was an
    # `nnx.List`, whose INTEGER keys make `flax.traverse_util.flatten_dict(params, sep="/")` raise
    # `TypeError: sequence item 2: expected str instance, int found` inside `_merge_params`. Running
    # the real `_merge_params` here against a synthetic "checkpoint" (the baseline arm's own tree)
    # exercises that flatten, the regex back-fill, and the dtype/shape reconciliation in one call, so
    # the whole loader contract is proven on the workstation instead of on a node.
    import flax.traverse_util  # noqa: PLC0415
    from openpi.training import weight_loaders as _weight_loaders  # noqa: PLC0415

    arm_shapes = jax.eval_shape(lambda r: nnx.state(model_config.create(r)), jax.random.key(0)).to_pure_dict()
    base_shapes = jax.eval_shape(lambda r: nnx.state(base_config.create(r)), jax.random.key(0)).to_pure_dict()
    try:
        flat_arm = flax.traverse_util.flatten_dict(arm_shapes, sep="/")
    except TypeError as error:
        raise SystemExit(
            f"the arm's param tree cannot be flattened with sep='/' ({error}). A path component is "
            "not a string — almost always an `nnx.List` (integer-indexed) somewhere in a new module. "
            "openpi's _merge_params does exactly this flatten, so training would die at "
            "init_train_state AFTER the multi-GB init fetch. Use nnx.Dict with string keys."
        ) from error
    if regex:
        merged = _weight_loaders._merge_params(  # noqa: SLF001  (the exact function the loader calls)
            base_shapes, arm_shapes, missing_regex=regex
        )
        flat_merged = flax.traverse_util.flatten_dict(merged, sep="/")
        if set(flat_merged) != set(flat_arm):
            missing = sorted(set(flat_arm) - set(flat_merged))[:5]
            raise SystemExit(
                f"_merge_params did not reproduce the arm's full param tree; {len(set(flat_arm) - set(flat_merged))} "
                f"leaf/leaves would be absent at init_train_state, e.g. {missing}"
            )
    print(
        f"[weight-loader check] extra_subtrees={extra or 'none'} covered by {regex!r} "
        f"(negative control '.*lora.*' rejected; _merge_params reproduced {len(flat_arm)} leaves)",
        flush=True,
    )


def _apply_vision_lr_patch(scale: float) -> None:
    """Scale the SigLIP vision tower's effective LR by `scale` via optax.multi_transform.
    1.0 (the official recipe) is a no-op. Ported from ported_raw/.../wsm_train_rc.py."""
    if scale == 1.0:
        return
    import jax
    import openpi.training.optimizer as _optimizer
    import optax

    _orig_create = _optimizer.create_optimizer

    def _path_label(path, _leaf) -> str:
        joined = "/" + "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path) + "/"
        return "vis" if "/img/" in joined else "rest"

    def create_optimizer(optimizer, lr_schedule, weight_decay_mask=None):
        tx_rest = _orig_create(optimizer, lr_schedule, weight_decay_mask)
        tx_vis = optax.chain(_orig_create(optimizer, lr_schedule, weight_decay_mask), optax.scale(scale))

        def labels(params):
            return jax.tree_util.tree_map_with_path(_path_label, params)

        return optax.multi_transform({"vis": tx_vis, "rest": tx_rest}, labels)

    _optimizer.create_optimizer = create_optimizer
    print(f"[pi05] optimizer patched: vision tower LR x{scale}", flush=True)


def _maybe_strip_wandb_media() -> None:
    """Drop image/video/audio from wandb.log unless WSM_LOG_MEDIA=1 (keeps runs lean)."""
    if os.environ.get("WSM_LOG_MEDIA", "0") == "1":
        return
    import wandb
    from wandb.sdk.wandb_run import Run as _WandbRun

    def _is_media(v) -> bool:
        items = v if isinstance(v, (list, tuple)) else [v]
        return any(isinstance(x, (wandb.Image, wandb.Video, wandb.Audio, wandb.Html)) for x in items)

    _orig = _WandbRun.log

    def _scalar_only(self, data, *a, **k):
        if isinstance(data, dict):
            data = {key: v for key, v in data.items() if not _is_media(v)}
            if not data:
                return None
        return _orig(self, data, *a, **k)

    _WandbRun.log = _scalar_only


def _wandb_resume_allow_patch() -> None:
    """Coerce wandb.init(resume='must') -> 'allow'. openpi's train.py uses resume='must' whenever orbax is
    resuming, which HARD-CRASHES if the recorded run id does not exist on the wandb server (observed on the
    pi-CFG 80k relaunch: ckpt resume worked, wandb refused, job died at setup). 'allow' resumes when the run
    exists and creates a fresh run otherwise — strictly more robust for retry/relaunch flows."""
    import wandb

    _orig_init = wandb.init

    def _init(*a, **k):
        if k.get("resume") == "must":
            k["resume"] = "allow"
        return _orig_init(*a, **k)

    wandb.init = _init


def _openpi_train_main():
    """Load the fork's scripts/train.py and return its main(config: TrainConfig)."""
    import openpi  # noqa: F401  (ensure fork importable)

    fork_root = Path(openpi.__file__).resolve().parents[2]  # <fork>/src/openpi/__init__.py -> <fork>
    train_py = fork_root / "scripts" / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"openpi train.py not found at {train_py}")
    spec = importlib.util.spec_from_file_location("openpi_train", train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["openpi_train"] = mod  # spawn workers re-import as __mp_main__
    spec.loader.exec_module(mod)
    return mod.main


def build_and_run_pi05(cfg, gs) -> None:
    """Apply patches, build the openpi TrainConfig, and dispatch to the fork's train.py."""
    run_train_config(build_train_config(cfg, gs), cfg, gs)


def run_train_config(train_cfg, cfg, gs) -> None:
    """Apply run patches and dispatch an ALREADY-BUILT openpi TrainConfig to the fork's train.py.

    Split out of build_and_run_pi05 so Stage-Q can dispatch its own builder's config (the Stage-S
    builder must never run under a Q arm — it drops the robottt flag; audit 2026-07-23).
    """
    # Entry gate FIRST: eval_shape-only, seconds, and it runs before openpi's train.py touches the
    # init checkpoint (the A1 ordering lesson). Cheap enough to run on every arm, so the base arms
    # also get a standing proof that their loader is still correct.
    assert_weight_loader_covers(train_cfg)
    _apply_vision_lr_patch(float(cfg.model.get("vision_lr_scale", 1.0)))
    _maybe_strip_wandb_media()
    _wandb_resume_allow_patch()
    import jax

    topology = runtime_parallelism_summary(
        batch_size=int(train_cfg.batch_size),
        num_workers=int(train_cfg.num_workers),
        fsdp_devices=int(train_cfg.fsdp_devices),
        device_count=int(jax.device_count()),
        local_device_count=int(jax.local_device_count()),
        process_count=int(jax.process_count()),
        expected_devices=_optional_positive_env("WSM_EXPECTED_JAX_DEVICES"),
        expected_processes=_optional_positive_env("WSM_EXPECTED_JAX_PROCESSES"),
        expected_batch_size=_optional_positive_env("WSM_EXPECTED_GLOBAL_BATCH"),
        expected_num_workers=_optional_positive_env("WSM_EXPECTED_NUM_WORKERS"),
        expected_fsdp_devices=_optional_positive_env("WSM_EXPECTED_FSDP_DEVICES"),
    )
    print(
        f"[pi05] dispatch openpi train: name={train_cfg.name} steps={train_cfg.num_train_steps} "
        f"global_bs={topology['batch_size']} per_device_bs={topology['per_device_batch_size']} "
        f"devices={topology['device_count']} local_devices={topology['local_device_count']} "
        f"processes={topology['process_count']} mesh="
        f"{topology['data_parallel_replicas']}x{topology['fsdp_devices']} "
        f"workers={topology['num_workers']} "
        f"defer_image_resize={train_cfg.data.defer_image_resize_to_model_preprocess} "
        f"balancing={'ON' if gs.balancing_enabled else 'OFF'}",
        flush=True,
    )
    window_len = int(getattr(train_cfg.data, "stage_q_window_len", 0) or 0)
    if window_len:
        print(
            f"[pi05] Stage-Q windows: batch_size counts WINDOWS (L={window_len}, "
            f"stride={train_cfg.data.stage_q_chunk_stride}, robottt={train_cfg.model.robottt}, "
            f"wsm_tanh={train_cfg.model.wsm_tanh}); effective per-step pi batch="
            f"{int(train_cfg.batch_size) * window_len}",
            flush=True,
        )
    _openpi_train_main()(train_cfg)
