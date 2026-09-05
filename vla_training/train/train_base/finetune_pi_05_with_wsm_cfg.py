#!/usr/bin/env python3
"""WSM-CFG pi0.5 target finetune (the pi twin of finetune_groot_17_with_wsm_cfg).

Identical operating point to finetune_pi_05.py (combined-30% target soup, init from the Phase-1 pretrain
ckpt, FULL finetune) EXCEPT a zero-init WSMCfgConditioner is added. `_pi05_common.build_train_config` keys
the CFG path off TWO envs:
  * WSM_POLICY_FEATS_ROOT -> the precomputed pi w_t dir (gates BOTH the dataset's wsm_w_window injection
    AND the conditioner)
  * WSM_CFG=1             -> select CFG over the legacy image-token injection (Pi0Config.wsm_cfg=True)
The conditioner turns w_t into an ADDITIVE vector on the action expert's adarms_cond, trained with
per-component CFG dropout to a learned null (WSM_CFG_P_DROP). The VLM/perception backbone is NEVER touched.
freeze=none => a FULL finetune (== baseline) + the net-new conditioner, so CFG-FT vs baseline-FT is
apples-to-apples; zero-init => step 0 is the exact baseline. Eval guidance is set per-serve-process
(serve_pi_05_wsm_cfg.py --guidance-scale), never here.

  WSM_POLICY_FEATS_ROOT=~/Research/TRI/wsm_data/wsm_policy_feats/pi_step100000 WSM_CFG=1 WSM_CFG_P_DROP=0.2 \
    python vla_training/train/train_base/finetune_pi_05_with_wsm_cfg.py \
      --config scripts/configs/train/pi05_wsm_cfg_finetune.yaml
  python ... --dry-run    # build/log soup + CFG config; no jax/openpi
"""

from __future__ import annotations

import argparse
import os

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "pi05", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="scripts/configs/train/pi05_wsm_cfg_finetune.yaml",
        help="train YAML (default: pi05_wsm_cfg_finetune.yaml)",
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log GroupedSoup + CFG config; no jax/openpi")
    args = ap.parse_args()

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    p_drop = float(os.environ.get("WSM_CFG_P_DROP", "0.2"))
    with_future = os.environ.get("WSM_CFG_WITH_FUTURE", "0") == "1"
    # Force the CFG gate on for this launcher (it is the CFG entry point). If left unset the user would
    # silently get the legacy image-token injection instead — fail-loud is via the env we set here.
    os.environ.setdefault("WSM_CFG", "1")
    print(
        f"[finetune_pi_05_with_wsm_cfg] WSM_POLICY_FEATS_ROOT={feats_root} WSM_CFG={os.environ['WSM_CFG']} "
        f"p_drop={p_drop} with_future={with_future}"
    )
    if with_future:
        # Demo-CFG (doc 15 pi port): slot -2 of the shipped window = precomputed frozen-fusion z (z4.npz);
        # requires the fork dataset's WSM_Z_WINDOWS_ROOT to be set, else it would silently feed a PAST frame.
        if not os.environ.get("WSM_Z_WINDOWS_ROOT") and not args.dry_run:
            raise SystemExit(
                "[wsm-demo-cfg] WSM_CFG_WITH_FUTURE=1 requires WSM_Z_WINDOWS_ROOT (z4.npz root) "
                "— without it slot -2 would be a causal PAST frame, not the demo latent."
            )
    if not feats_root and not args.dry_run:
        raise ValueError(
            "set WSM_POLICY_FEATS_ROOT to the precomputed pi w_t dir (generate_policy_features "
            "--backbone pi); it gates BOTH the dataset injection and Pi0Config.wsm_cfg"
        )

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    if args.dry_run:
        print("[finetune_pi_05_with_wsm_cfg] dry-run OK (soup + CFG config; skipping openpi build + dispatch).")
        return

    from vla_training.train.train_base._pi05_common import build_and_run_pi05

    build_and_run_pi05(cfg, gs)


if __name__ == "__main__":
    main()
