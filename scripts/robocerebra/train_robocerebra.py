#!/usr/bin/env python3
"""Train entrypoint for the RoboCerebra arms = openpi's own ``scripts/train.py`` + the video shim.

Why a wrapper rather than calling openpi's train.py directly: the RoboCerebra dataset is a plain
LeRobot ``repo_id`` dataset with two *video* features, so every sample decodes mp4 through
LeRobot. Both of LeRobot 0.1.0's decoder backends are broken in the openpi venv (torchcodec's
prebuilt .so will not load against torch 2.11+cu128; the pyav route goes through a
``torchvision.io.VideoReader`` that torchvision 0.28 removed). ``lerobot_video_shim.install()``
swaps in a direct PyAV decoder and must run **before** the dataset is constructed -- fork-started
DataLoader workers inherit it, spawn-started ones would not, which is why it is installed at
import time in the parent rather than inside the loader.

Everything else is stock: the config comes from openpi's tyro CLI over the registered configs, so
``--data.repo_id``/``--batch-size``/``--num-train-steps``-style overrides all work unchanged.

    python train_robocerebra.py pi05_robocerebra_base --exp-name=<name> --overwrite
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def _load_openpi_train_main():
    """Import the fork's ``scripts/train.py`` by path (it is a script, not a package module)."""
    import openpi  # noqa: F401  -- ensures the fork is importable before we go looking for it

    openpi_root = pathlib.Path(openpi.__file__).resolve().parents[2]
    train_py = openpi_root / "scripts" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"openpi train.py not found at {train_py}")
    spec = importlib.util.spec_from_file_location("_openpi_train", train_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_openpi_train"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    # The shim lives in the openpi fork, not here: DataLoader workers are SPAWNED, so each one has
    # to install it itself (openpi's `_worker_init_fn` does, gated on this env var) and a worker
    # cannot be relied on to have wsmv2 on its path. Setting the var here rather than only in the
    # node entry keeps local runs and queue runs on the same code path.
    import os

    from openpi.training import lerobot_video_shim

    os.environ[lerobot_video_shim.ENV_FLAG] = "1"
    lerobot_video_shim.install()
    print("[robocerebra] PyAV video shim installed (parent) + enabled for spawned workers", flush=True)

    train = _load_openpi_train_main()
    import openpi.training.config as _config

    config = _config.cli()
    print(
        f"[robocerebra] config={config.name} batch={config.batch_size} "
        f"steps={config.num_train_steps} save_interval={config.save_interval} "
        f"keep_period={config.keep_period}",
        flush=True,
    )
    print(f"[robocerebra] weight_loader={config.weight_loader}", flush=True)
    train.main(config)


if __name__ == "__main__":
    main()
