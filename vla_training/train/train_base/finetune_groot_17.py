#!/usr/bin/env python3
"""Phase-2 base-VLA TARGET-FINETUNE of GR00T N1.7 on the 50 RoboCasa365 target tasks (Isaac-GR00T).

ONE combined finetune over atomic_seen + composite_seen + composite_unseen at the configured demo
fraction (default 30% via data.subsample.target_fraction; composite_unseen first appears here).
Single human source => balancing OFF => one GR00T dataset spec (mix_ratio 1.0) over all 50 dirs.

GR00T ignores robocasa's filter_key, so the 30% is applied by a deterministic loader subsample
(utils.subsample via _groot_common.install_episode_subsample) that keeps the SAME first-150 episodes
(seed-0 shuffle by episode_index value) as pi0.5 — identical episodes across backbones. Inits from
the Phase-1 pretrain checkpoint (train.init_from).

ALSO drives the ReMemBench finetunes (``soup: remembench13``, groot17_rmb*_finetune.yaml): the same
entry, the same init, the same recipe surface — only the soup and the step schedule change. That
soup is NATIVE FULL MASS (filter_key None on every meta), so the episode-subsample machinery is
skipped entirely rather than being called with a count it cannot derive; see the branch in main().

Runs in the groot venv with Isaac-GR00T + robocasa importable.

  python vla_training/train/train_base/finetune_groot_17.py --config scripts/configs/train/groot17_target_finetune.yaml
  python ... --config scripts/configs/train/groot17_rmb_base_finetune.yaml    # ReMemBench 13-task
  python ... --dry-run     # build + log the GroupedSoup + mass resolution, no torch/gr00t
"""

from __future__ import annotations

import argparse

from utils.subsample import uniform_num_demos
from vla_training.train.train_base._adapter_common import load_recipe

BACKBONE, PHASE = "groot_17", "finetune"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=None, help="train YAML (default: scripts/configs/train/groot17_target_finetune.yaml)"
    )
    ap.add_argument("--dry-run", action="store_true", help="build/log GroupedSoup + subsample only; no torch/gr00t")
    args = ap.parse_args()

    cfg, gs = load_recipe(args.config, backbone=BACKBONE, phase=PHASE)
    # SOUP MASS. Two mutually exclusive regimes, selected by the soup itself, never by a flag:
    #   * target50 (the RoboCasa recipe): every ds_meta carries the same filter_key (e.g.
    #     '150_demos'); GR00T has no native filter_key, so we mirror pi0.5's selection with a
    #     deterministic per-dir episode subsample.
    #   * remembench13: filter_key is None on every meta — NATIVE FULL MASS (all 323 demos across
    #     13 tasks; the train split is already the complement of a held-out tail, so there is
    #     nothing to select). utils.subsample.uniform_num_demos raises by contract on such a soup,
    #     so it must not be called, and episode_subsample_num_demos stays None => the
    #     LeRobotEpisodeLoader patch is never installed and every episode is used.
    full_mass = all(m.get("filter_key") is None for m in gs.soup)
    if full_mass:
        num_demos = None
        print(
            f"[finetune_groot_17] native full mass: no episode subsample "
            f"({len(gs.soup)} dataset dirs, filter_key=None on all)"
        )
    else:
        num_demos = uniform_num_demos(gs.soup)
        print(f"[finetune_groot_17] deterministic episode subsample: keep first {num_demos}/dir (seed 0)")

    if args.dry_run:
        print("[finetune_groot_17] dry-run OK (GroupedSoup + mass resolution; skipping gr00t dispatch).")
        return

    # WSM_INIT_FROM (set by the SageMaker FT entry to the node-local synced ckpt) wins over the
    # s3:// init_from recorded in the YAML (gr00t loads from a local HF dir).
    import os

    from vla_training.train.train_base._groot_common import build_and_run_groot

    init_from = os.environ.get("WSM_INIT_FROM") or cfg.train.get("init_from")
    if not init_from:
        raise ValueError("finetune requires train.init_from = the Phase-1 pretrain checkpoint")
    build_and_run_groot(
        cfg,
        gs,
        start_from_checkpoint=str(init_from),
        visual_lr_scale=float(cfg.model.get("visual_lr_scale", 1.0)),
        episode_subsample_num_demos=num_demos,
    )


if __name__ == "__main__":
    main()
