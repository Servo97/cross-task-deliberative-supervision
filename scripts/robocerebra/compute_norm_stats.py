#!/usr/bin/env python3
"""Compute openpi norm stats for a RoboCerebra LeRobot dataset, written outside the openpi fork.

This is ``openpi/scripts/compute_norm_stats.py`` with two deliberate differences:

1. ``--assets-dir`` defaults into ``wsm_data`` instead of ``<openpi>/assets``, so nothing is
   written inside the openpi checkout.
2. It also dumps a side-by-side diff against the **released pi05_libero norm stats**. That
   comparison is the decision record for the one normalization choice this dataset forces:
   post-training starts from the released ``pi05_libero`` checkpoint, so either we keep that
   checkpoint's LIBERO stats (consistency with init) or we re-base onto RoboCerebra's own
   distribution (distribution match). Whichever is chosen has to be chosen once, explicitly,
   and sealed -- silent re-basing is the failure mode we already paid for.

Stats are accumulated over the *same* transform stack the trainer uses (repack + LiberoInputs),
so the numbers describe exactly what the model sees, not the raw parquet columns.

    HF_LEROBOT_HOME=<...> uv run --project <openpi> python compute_norm_stats.py \
        --repo-id wsmv2/robocerebra_train --assets-dir <wsm_data>/robocerebra/assets
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import tqdm

LIBERO_REFERENCE_URL = (
    "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/"
    "assets/physical-intelligence/libero/norm_stats.json"
)


class RemoveStrings:
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main() -> None:
    import lerobot_video_shim

    lerobot_video_shim.install()
    import openpi.shared.normalize as normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id resolved under HF_LEROBOT_HOME")
    parser.add_argument(
        "--assets-dir", required=True, help="where <asset_id>/norm_stats.json is written (keep out of the openpi tree)"
    )
    parser.add_argument(
        "--config", default="pi05_libero", help="openpi TrainConfig whose data pipeline defines the transforms"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--reference", default=None, help="path to the released pi05_libero norm_stats.json for the diff report"
    )
    args = parser.parse_args()

    base = _config.get_config(args.config)
    # Bootstrap seed. openpi's `_load_norm_stats` fallback in our fork calls
    # `groot_openpi_dataset._convert_stats_from_repo_meta`, which does not exist there, so a
    # *missing* norm_stats.json raises AttributeError instead of returning None -- i.e. the
    # first-ever stats computation for a new repo_id cannot bootstrap itself. Seeding the
    # released pi05_libero stats at the expected path sidesteps it; the value is irrelevant
    # because RunningStats below never reads it, and it gets overwritten at the end.
    seed_target = pathlib.Path(args.assets_dir) / args.config / args.repo_id / "norm_stats.json"
    if not seed_target.exists():
        if not args.reference:
            raise SystemExit("first run for this repo_id needs --reference to seed norm stats")
        seed_target.parent.mkdir(parents=True, exist_ok=True)
        seed_target.write_text(pathlib.Path(args.reference).read_text())
        print(f"seeded bootstrap norm stats at {seed_target}")

    config = dataclasses.replace(
        base,
        data=dataclasses.replace(base.data, repo_id=args.repo_id),
        assets_base_dir=args.assets_dir,
    )
    # config.assets_dirs == <assets-dir>/<config name>; keep write and read on the same path.
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
    )
    if args.max_frames is not None and args.max_frames < len(dataset):
        num_batches, shuffle = args.max_frames // args.batch_size, True
    else:
        num_batches, shuffle = len(dataset) // args.batch_size, False
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="norm stats"):
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))
    norm_stats = {key: running.get_statistics() for key, running in stats.items()}

    output = config.assets_dirs / args.repo_id
    normalize.save(output, norm_stats)
    print(f"wrote {output / 'norm_stats.json'}")

    if args.reference:
        reference = json.loads(pathlib.Path(args.reference).read_text())["norm_stats"]
        print(f"\ndiff vs released pi05_libero stats ({LIBERO_REFERENCE_URL}):")
        for key in ("state", "actions"):
            ours = norm_stats[key]
            for field in ("mean", "std", "q01", "q99"):
                mine = np.asarray(getattr(ours, field))[: len(reference[key][field])]
                theirs = np.asarray(reference[key][field])
                print(f"  {key}.{field}")
                print(f"    robocerebra {np.round(mine, 4).tolist()}")
                print(f"    pi05_libero {np.round(theirs, 4).tolist()}")
                print(f"    abs delta   {np.round(np.abs(mine - theirs), 4).tolist()}")


if __name__ == "__main__":
    main()
