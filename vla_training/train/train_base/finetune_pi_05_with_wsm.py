#!/usr/bin/env python3
"""WSM-CONDITIONED pi0.5 target finetune (the pi twin of finetune_groot_17_with_wsm).

Identical to finetune_pi_05.py (combined-30% target soup, init from the Phase-1 pretrain ckpt) EXCEPT
WSM-conditioning is switched on, which `_pi05_common.build_train_config` keys off `WSM_POLICY_FEATS_ROOT`:
  * Pi0Config(wsm=True, wsm_k_window=K) -> a zero-init TokenModulator in Pi0.embed_prefix
  * freeze_filter = train ONLY the modulator (whole pi0.5 backbone frozen)
  * the GrootOpenpi dataset injects wsm_w_window/wsm_lang per sample from the precomputed w.npz (the
    SAME WSM_POLICY_FEATS_ROOT) -> Observation.wsm_* -> the modulator.
Zero-init => step 0 is the exact baseline, so any Eval2 gain is attributable to w_t (apples-to-apples
vs the pi0.5 Eval1). Requires the pi w_t on the node + WSM_POLICY_FEATS_ROOT / WSM_K_WINDOW set.

  WSM_POLICY_FEATS_ROOT=~/Research/TRI/wsm_data/wsm_policy_feats/pi_step100000 WSM_K_WINDOW=2 \
    python vla_training/train/train_base/finetune_pi_05_with_wsm.py \
      --config scripts/configs/train/pi05_target_finetune.yaml
  python ... --dry-run    # build/log soup + WSM config; no jax/openpi
"""

from __future__ import annotations

import argparse
import os

from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "pi05", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=None, help="train YAML (default: scripts/configs/train/pi05_target_finetune.yaml)"
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log GroupedSoup + WSM config; no jax/openpi")
    args = ap.parse_args()

    feats_root = os.environ.get("WSM_POLICY_FEATS_ROOT")
    # Historical reproduction only. _pi05_common intentionally refuses to infer this failed direct-token
    # interface merely from the presence of a feature root.
    os.environ.setdefault("WSM_LEGACY_TOKEN_INJECTION", "1")
    k_window = int(os.environ.get("WSM_K_WINDOW", "2"))
    print(f"[finetune_pi_05_with_wsm] LEGACY direct-token mode WSM_POLICY_FEATS_ROOT={feats_root} k_window={k_window}")
    if not feats_root and not args.dry_run:
        raise ValueError(
            "set WSM_POLICY_FEATS_ROOT to the precomputed pi w_t dir (generate_policy_features "
            "--backbone pi); it gates BOTH the dataset injection and Pi0Config.wsm"
        )

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    if args.dry_run:
        print("[finetune_pi_05_with_wsm] dry-run OK (soup + WSM config; skipping openpi build + dispatch).")
        return

    from vla_training.train.train_base._pi05_common import build_and_run_pi05

    build_and_run_pi05(cfg, gs)


if __name__ == "__main__":
    main()
