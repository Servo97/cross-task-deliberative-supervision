#!/usr/bin/env python3
"""Stage-S pi0.5 workspace-read target finetune.

This is the only new entry point for S1/S2. It selects one explicit action-expert interface:

* tanh: adarms_cond += tanh(alpha) * P(omega_t), one policy suffix pass, no null/dropout.
* cfg2: current-only P(omega_t)/P(null_t), CFG drop-to-null during training.

Both modes full-finetune the same pi0.5 surface as S0, use the identical adarms_cond seam, and leave
all VLM/prefix tokens unchanged. Historical WSM_CFG and direct-token paths are not selected here.

S5 COMBO: a tanh-interface recipe may additionally set ``model.jepa_aux: true`` to carry the S3
train-only JEPA+SigReg aux in the SAME run. It is a recipe knob, not a fourth --interface, because
the arm's identity, manifest and serve contract are unchanged (the aux head is train-only and is
dropped at load time); the loader then ships wsm_w_window AND wsm_w_target from one omega cache.
"""

from __future__ import annotations

import argparse
import os

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "pi05", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interface", required=True, choices=("tanh", "cfg2", "jepa", "h13"))
    ap.add_argument(
        "--config",
        default="scripts/configs/train/pi05_workspace_finetune.yaml",
        help="shared recipe for S1/S2 (S3 passes its own jepa yaml)",
    )
    ap.add_argument("--dry-run", action="store_true", help="validate recipe selection without OpenPI")
    args = ap.parse_args()

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    env_name = {"tanh": "WSM_TANH", "cfg2": "WSM_CFG2", "jepa": "WSM_JEPA", "h13": "WSM_H13"}[args.interface]
    os.environ[env_name] = "1"
    if args.interface == "jepa":
        # S3: the loader ships wsm_w_target (never wsm_w_window); nothing is read at inference.
        os.environ["WSM_JEPA_TARGETS"] = "1"
    # H13 (R1/R2): the workspace latent is produced LIVE inside the training graph off the policy's
    # own SigLIP tokens. There is no omega cache to stage and no wsm_w_* key in the batch — only the
    # sealed keypatch labels (via the shared WSM_SALIENT_TARGETS gate) and, for R2, the t+k FRAME.
    # Those exports are recipe-dependent, so they land in the block after `load_recipe` below — which
    # still precedes the `_pi05_common` (and therefore openpi) import that freezes them.
    # The K guard must know the recipe: the gated-deltanet conditioner and its PTRM variant
    # (cond_type in the yaml) legitimately consume a cond_window-long omega window, everything else
    # reads newest-only.
    # 2026-08-01: the blanket K==1 form of this guard killed the first s1-deltanet run at startup.
    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    # Imported HERE, not at module scope: `_pi05_common` captures WSM_JEPA_NUM_FUTURES at its own
    # import, and the combo block below depends on this module's env exports landing first.
    from vla_training.train.train_base._pi05_common import _WINDOWED_COND_TYPES

    cond_type = str(os.environ.get("WSM_COND_TYPE") or cfg.model.get("cond_type", "tanh"))
    cond_window = int(os.environ.get("WSM_COND_WINDOW") or cfg.model.get("cond_window", 1))
    k_window = int(os.environ.get("WSM_K_WINDOW", "1"))

    def _export_h13_recipe():
        """Export the H13 loader gates from the recipe. Shared by --interface h13 (R1-R4) and by the
        tanh+h13 combo (R5-R8), so the two paths can never drift."""
        h13_jepa = cfg.model.get("h13_jepa", False)
        if type(h13_jepa) is not bool:
            raise ValueError(f"model.h13_jepa must be a boolean, got {h13_jepa!r}")
        if h13_jepa:
            os.environ["WSM_H13_JEPA"] = "1"
            os.environ["WSM_H13_FUTURE_FRAMES"] = "1"
            os.environ["WSM_H13_FUTURE_K"] = str(cfg.model.get("h13_future_k", 1))
            os.environ["WSM_H13_FUTURE_STRIDE"] = str(cfg.model.get("h13_future_stride", 8))
        h13_lang = cfg.model.get("h13_lang", False)
        if type(h13_lang) is not bool:
            raise ValueError(f"model.h13_lang must be a boolean, got {h13_lang!r}")
        if h13_lang:
            os.environ["WSM_H13_LANG"] = "1"
        h13_lang_cls = cfg.model.get("h13_lang_cls", False)
        if type(h13_lang_cls) is not bool:
            raise ValueError(f"model.h13_lang_cls must be a boolean, got {h13_lang_cls!r}")
        if h13_lang_cls:
            os.environ["WSM_H13_LANG_CLS"] = "1"
        print(
            f"[pi-workspace-ft] h13: live encoder ON (lambda_dec="
            f"{cfg.model.get('h13_dec_weight', 0.5)} sigreg={cfg.model.get('h13_sigreg_weight', 0.05)}) "
            f"lejepa={'on' if h13_jepa else 'off'} lang={'on' if h13_lang else 'off'} lang_cls={'on' if h13_lang_cls else 'off'} "
            f"(lambda_jepa={cfg.model.get('h13_jepa_weight', 0.1)} "
            f"k={cfg.model.get('h13_future_k', 1)} stride={cfg.model.get('h13_future_stride', 8)})",
            flush=True,
        )

    if args.interface == "h13":
        if feats_root:
            raise ValueError(
                "the H13 live encoder reads no omega cache; WSM_POLICY_FEATS_ROOT must be unset so "
                "the run cannot claim a feature-manifest provenance it never consults"
            )
        _export_h13_recipe()
    if args.interface == "tanh" and cond_type in _WINDOWED_COND_TYPES:
        if k_window not in (1, cond_window):
            raise ValueError(
                f"{cond_type} trains at cond_window={cond_window}; WSM_K_WINDOW={k_window} "
                f"matches neither that nor the overridable launcher default 1"
            )
        os.environ["WSM_K_WINDOW"] = str(cond_window)
    else:
        if k_window != 1:
            raise ValueError(
                f"new Stage-S {args.interface} reads newest omega_t only; require WSM_K_WINDOW=1, "
                f"got {k_window} (K=2 is legacy-checkpoint reproduction only)"
            )
        os.environ["WSM_K_WINDOW"] = "1"
    conflicts = [
        name
        for name in ("WSM_LEGACY_TOKEN_INJECTION", "WSM_CFG", "WSM_CFG2", "WSM_TANH", "WSM_JEPA", "WSM_H13")
        if os.environ.get(name, "0") == "1" and name != env_name
    ]
    if conflicts:
        raise ValueError(f"{env_name}=1 conflicts with already enabled interfaces {conflicts}")
    # S5 COMBO ARM (tanh read + train-only JEPA aux in one run). Selected by the RECIPE, not by a new
    # CLI flag or a bare env flip, so the SageMaker entry keeps dispatching `--interface tanh` and the
    # sealed manifest/interface/serve contract are those of the plain deltanet arm. The three env
    # exports must land BEFORE `_pi05_common` (and therefore openpi) is imported below: the dataloader
    # freezes all of WSM_JEPA_TARGETS / WSM_JEPA_WITH_WINDOW / WSM_JEPA_NUM_FUTURES at import.
    # H13 gdn8 COMBO (R5-R8): the gated-DeltaNet workspace READ plus the H13 live aux, in one run.
    # Selected by the RECIPE (model.h13 on a tanh-interface arm), exactly like model.jepa_aux below,
    # so the SageMaker entry keeps dispatching `--interface tanh` and the omega staging / manifest /
    # serve contract stay those of the plain deltanet arm. Set AFTER the conflicts check: WSM_H13
    # rides alongside WSM_TANH here and is a sanctioned pair (_SANCTIONED_COMBOS), not a conflict.
    h13_recipe = cfg.model.get("h13", False)
    if type(h13_recipe) is not bool:
        raise ValueError(f"model.h13 must be a boolean, got {h13_recipe!r}")
    if h13_recipe and args.interface == "tanh":
        if not feats_root:
            raise ValueError(
                "the H13+gdn8 combo READS the omega window at inference (gated_deltanet), so it "
                "requires WSM_POLICY_FEATS_ROOT - unlike the aux-only R1-R4 arms"
            )
        os.environ["WSM_H13"] = "1"
        _export_h13_recipe()
        print(
            f"[pi-workspace-ft] combo: gdn8 read (cond_type={cond_type} window={cond_window}) "
            f"+ H13 live aux; serve keeps the conditioner and strips every H13 subtree",
            flush=True,
        )
    elif h13_recipe and args.interface != "h13":
        raise ValueError(
            f"model.h13 rides either --interface h13 (aux-only) or --interface tanh (the gdn8 "
            f"combo); got --interface {args.interface}"
        )
    jepa_aux = cfg.model.get("jepa_aux", False)
    if type(jepa_aux) is not bool:
        raise ValueError(f"model.jepa_aux must be a boolean, got {jepa_aux!r}")
    if jepa_aux:
        if args.interface == "h13":
            raise ValueError(
                "model.jepa_aux (a FROZEN-cache omega target) and the H13 live encoder would put two "
                "different targets on the same penultimate and make reading (b) unattributable"
            )
        if args.interface != "tanh":
            raise ValueError(
                f"model.jepa_aux rides the tanh interface (the aux is additive to the workspace "
                f"read); got --interface {args.interface}. A target-only arm is --interface jepa."
            )
        os.environ["WSM_JEPA"] = "1"
        os.environ["WSM_JEPA_TARGETS"] = "1"
        os.environ["WSM_JEPA_WITH_WINDOW"] = "1"
        print(
            f"[pi-workspace-ft] combo: jepa aux ON alongside the tanh read "
            f"(lambda_jepa={cfg.model.get('jepa_weight', 1.0)} "
            f"sigreg={cfg.model.get('sigreg_weight', 0.05)} "
            f"k={cfg.model.get('jepa_num_futures', 1)})",
            flush=True,
        )
    if not feats_root and not args.dry_run and args.interface != "h13":
        raise ValueError("set WSM_POLICY_FEATS_ROOT to the precomputed pi omega-token directory")

    print(
        f"[pi-workspace-ft] interface={args.interface} {env_name}=1 "
        f"features={feats_root} p_drop={os.environ.get('WSM_CFG_P_DROP', '0.2')} "
        f"gate_init={os.environ.get('WSM_TANH_GATE_INIT', '0.001')}",
        flush=True,
    )
    if args.dry_run:
        print("[pi-workspace-ft] dry-run OK; no OpenPI build or training")
        return

    from vla_training.train.train_base._pi05_common import build_and_run_pi05

    build_and_run_pi05(cfg, gs)


if __name__ == "__main__":
    main()
