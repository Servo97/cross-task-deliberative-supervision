"""Stage-S (D0) pi0.5 WSM encoder training entry.

Consumes the sealed Stage-S artifacts (canonical task-language table + source-feature manifest +
pairs manifest, all SHA-verified) and the immutable objective run-config, then trains the frozen-
probe WorkspaceModel with demonstration-disjoint train/val, evaluated validation, and optional
8-GPU DDP (torchrun). It refuses to run if any pairs row references a demo not declared in the
source-feature manifest.

  torchrun --standalone --nproc_per_node 8 -m \
    workspace_models.train.train_wsm_base.train_wsm_pi_stage_s \
    --manifest pairs.parquet --run-config run_config.json \
    --task-lang-table task_lang_table.npz --task-lang-table-manifest <id>.json \
    --source-features-manifest <id>.json --out <run-dir>

Run in the torch env; see internal_planning_and_todos/_archive/handover_to_opus/{03,04}_packet_*.md.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from workspace_models.train.train_wsm_base._wsm_stage_s_train import (
    get_rank,
    load_run_config,
    train_stage_s,
)
from workspace_models.train.train_wsm_base.data import load_demo_pi_stage_s


def _maybe_init_dist() -> None:
    import os

    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        backend = "nccl" if __import__("torch").cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def _load_table(path: str, manifest: str) -> dict:
    from scripts.launch.validate_stage_s_task_lang_table import load_task_lang_table

    # The manifest pins the npz sha; load_task_lang_table verifies it.
    return load_task_lang_table(manifest, table_path=path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Stage-S pairs.parquet (build_pairs --stage-s)")
    ap.add_argument("--run-config", required=True, help="immutable objective config JSON")
    ap.add_argument("--task-lang-table", required=True, help="task_lang_table.npz")
    ap.add_argument("--task-lang-table-manifest", required=True, help="content-addressed table manifest")
    ap.add_argument("--source-features-manifest", required=True, help="source-feature manifest (join check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--backbone-dim", type=int, default=2048)
    ap.add_argument("--proprio-dim", type=int, default=2048)
    ap.add_argument("--lang-dim", type=int, default=2048)
    args = ap.parse_args()

    _maybe_init_dist()
    cfg = load_run_config(args.run_config)
    table = _load_table(args.task_lang_table, args.task_lang_table_manifest)

    import pandas as pd

    rows = pd.read_parquet(args.manifest).to_dict("records")
    # Hash-join guard: every training demo must be declared in the source-feature manifest.
    src = json.loads(Path(args.source_features_manifest).read_text(encoding="utf-8"))
    declared = {(r["task"], int(e["episode_index"])) for r in src["tasks"] for e in r["episodes"]}
    missing = [(r["task"], int(r["demo_id"])) for r in rows if (str(r["task"]), int(r["demo_id"])) not in declared]
    if missing:
        raise SystemExit(f"{len(missing)} pairs rows are not in the source-feature manifest (e.g. {missing[:5]})")
    for r in rows:
        if str(r["task"]) not in table:
            raise SystemExit(f"pairs row task {r['task']!r} is absent from the task-language table")

    load_fn = partial(load_demo_pi_stage_s, table={k: np.asarray(v) for k, v in table.items()})
    final = train_stage_s(
        rows=rows,
        load_fn=load_fn,
        cfg=cfg,
        backbone_dim=args.backbone_dim,
        proprio_dim=args.proprio_dim,
        lang_dim=args.lang_dim,
        out_dir=args.out,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        val_every=args.val_every,
        log_every=args.log_every,
        save_every=args.save_every,
        extra_provenance={
            "task_lang_table_manifest": Path(args.task_lang_table_manifest).name,
            "source_features_manifest": Path(args.source_features_manifest).name,
        },
    )
    if get_rank() == 0:
        print(f"[stage-s-train] final checkpoint: {final}", flush=True)


if __name__ == "__main__":
    main()
