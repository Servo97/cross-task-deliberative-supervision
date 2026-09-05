"""Materialize the dense/multi-point VISReg-v2 workspace encoder under its own identity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .single_task import TASK_EPISODES
from .workspace_deliberative_dense_v2 import (
    PROTOCOL,
    DenseWorkspaceBatchSampler,
    init_params_dense_v2,
)
from .workspace_materialize import materialize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASK_EPISODES))
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-devices", type=int, default=4)
    parser.add_argument("--skip-supervision-hashes", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--one-batch-canary", action="store_true")
    args = parser.parse_args()
    if args.one_batch_canary:
        if not args.cpu_smoke and os.environ.get("WSM_MOVE_DENSE_V2_CANARY") != "1":
            raise SystemExit("dense v2 materializer canary requires WSM_MOVE_DENSE_V2_CANARY=1")
    elif not args.cpu_smoke and os.environ.get("WSM_MOVE_DENSE_V2_MATERIALIZE_ALLOW_RUN") != "1":
        raise SystemExit("dense v2 omega materialization requires its reviewed v2 run gate")
    materialize(
        args,
        sampler_class=DenseWorkspaceBatchSampler,
        trainer_path=Path(__file__).with_name("workspace_deliberative_dense_v2.py"),
        materializer_path=Path(__file__),
        required_protocol=PROTOCOL,
        init_params_function=init_params_dense_v2,
    )


if __name__ == "__main__":
    main()
