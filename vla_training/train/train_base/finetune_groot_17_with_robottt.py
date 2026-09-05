#!/usr/bin/env python3
"""RoboTTT fast-weights (reduced form) GR00T N1.7 target finetune — the torch twin of the pi0.5
`robottt_fast` arm, for the ReMemBench sync validation.

Identical to finetune_groot_17.py (same soup, same init-from, same freeze/optim) EXCEPT:

  * the dataset emits CONTIGUOUS WINDOWS of L chunk-steps from one episode instead of single steps
    (GR00T's stock sharded dataset actively shuffles step indices, so contiguity had to be rebuilt),
  * the collator flattens `vlm_content` window-major so the VLM tensors arrive at batch B*L,
  * the DiT's global AdaLN bus `temb` gets an additive gated read of a per-sample FAST WEIGHT `W`
    that is updated by gradient descent along the window (apply-then-commit at chunk granularity).

Unlike the deltanet arm this is STATEFUL across policy calls — that is the whole axis it tests.

  WSM_SEQ_WINDOW_LEN=8 WSM_SEQ_CHUNK_STRIDE=8 \
    python vla_training/train/train_base/finetune_groot_17_with_robottt.py \
      --config scripts/configs/train/groot17_rmb_ttt_finetune.yaml
  python ... --dry-run     # build/log soup + window recipe + TTT config, no torch/gr00t
"""

from __future__ import annotations

import argparse
import os

from utils.subsample import uniform_num_demos
from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "groot_17", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="scripts/configs/train/groot17_rmb_ttt_finetune.yaml",
        help="RoboTTT fast-weights train YAML",
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log soup + window recipe + TTT config; no gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    # SOUP MASS — mirrors finetune_groot_17.py exactly (selected by the soup, never by a flag):
    #   * target50: filter_key set => deterministic seed-0 keep-first-N episode subsample.
    #   * remembench13: filter_key None on every meta => NATIVE FULL MASS. uniform_num_demos raises
    #     by contract on such a soup, so it must not be called.
    full_mass = all(m.get("filter_key") is None for m in gs.soup)
    num_demos = None if full_mass else uniform_num_demos(gs.soup)

    window_len = int(os.environ.get("WSM_SEQ_WINDOW_LEN", "8"))
    chunk_stride = int(os.environ.get("WSM_SEQ_CHUNK_STRIDE", "8"))
    print(
        f"[finetune_groot_17_with_robottt] subsample first {num_demos}/dir (seed 0) | "
        f"window_len={window_len} chunk_stride={chunk_stride} "
        f"tbptt={os.environ.get('WSM_TTT_TBPTT_SEGMENT', window_len)} "
        f"d={os.environ.get('WSM_TTT_TOKEN_DIM', 256)} h={os.environ.get('WSM_TTT_FAST_HIDDEN', 128)} "
        f"eta={os.environ.get('WSM_TTT_INNER_LR', 0.1)}"
    )

    # EFFECTIVE BATCH. One item is now L chunk-steps, so a config batch_size of N puts N*L samples
    # through the backbone. The yaml therefore carries batch_size = baseline/L and this prints the
    # product, because a silent L-fold blow-up is an OOM at step 0 on a full node.
    print(
        f"[finetune_groot_17_with_robottt] batch {cfg.train.get('batch_size')} windows "
        f"= {int(cfg.train.get('batch_size', 0)) * window_len} chunk-steps/step"
    )

    if args.dry_run:
        print("[finetune_groot_17_with_robottt] dry-run OK (soup + window recipe + TTT config; skipping gr00t).")
        return

    from vla_training.train.train_base._groot_common import build_and_run_groot
    from vla_training.train.train_base._groot_robottt_common import (
        install_robottt_action_head,
        install_robottt_collator,
        install_robottt_dataset,
    )

    # ORDER MATTERS: the dataset and the collator must BOTH be patched before the pipeline builds,
    # and the action head's fail-loud check exists precisely because a half-applied set of patches
    # would otherwise train as the baseline under this arm's name.
    install_robottt_dataset(window_len=window_len, chunk_stride=chunk_stride)
    install_robottt_collator()
    install_robottt_action_head()

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("robottt finetune requires train.init_from = the Phase-1 pretrain checkpoint")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,
    )


if __name__ == "__main__":
    main()
