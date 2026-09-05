#!/usr/bin/env python
"""Rewrite a LOCAL WSM manifest.parquet -> manifest_node.parquet with on-node (/opt/ml/work) paths.

The trainer entry (robocasa_wsm_train_entry.sh) syncs the cache to /opt/ml/work/wsm_cache and the labels
to /opt/ml/work/wsm_labels, so the manifest's feature_dir/labels_path/frames_path (which build_pairs
writes as absolute LOCAL paths) must be remapped before upload. This makes that step reproducible
instead of ad hoc. The node cache path is ALWAYS /opt/ml/work/wsm_cache (the entry syncs whichever
CACHE_S3 — groot wsm_cache or pi wsm_cache_pi — into that same dir), so one mapping serves both backbones.

  python scripts/launch/make_node_manifest.py \
      --in ~/Research/TRI/wsm_data/wsm_cache/manifest.parquet --out ~/Research/TRI/wsm_data/wsm_cache/manifest_node.parquet
  # pi: --in ~/Research/TRI/wsm_data/wsm_cache_pi/manifest.parquet --out ~/Research/TRI/wsm_data/wsm_cache_pi/manifest_node.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# local-root -> on-node-root (entry syncs CACHE_S3 -> wsm_cache, LABELS_S3 -> wsm_labels)
DEFAULT_MAP = {
    str(Path.home() / "Research/TRI/wsm_data/wsm_cache_pi"): "/opt/ml/work/wsm_cache",
    str(Path.home() / "Research/TRI/wsm_data/wsm_cache"): "/opt/ml/work/wsm_cache",
    str(Path.home() / "Research/TRI/wsm_data/wsm_vlm_rc"): "/opt/ml/work/wsm_labels",
}
PATH_COLS = ("feature_dir", "labels_path", "frames_path")


def remap(value: str, mapping: dict[str, str]) -> str:
    for local, node in mapping.items():  # longest local root first (wsm_cache_pi before wsm_cache)
        if value.startswith(local):
            return node + value[len(local) :]
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="local manifest.parquet (build_pairs output)")
    ap.add_argument("--out", required=True, help="manifest_node.parquet to write")
    args = ap.parse_args()

    mapping = dict(sorted(DEFAULT_MAP.items(), key=lambda kv: -len(kv[0])))
    df = pd.read_parquet(Path(args.inp).expanduser())
    for col in PATH_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: remap(str(v), mapping))
    out = Path(args.out).expanduser()
    df.to_parquet(out, index=False)
    print(f"[node-manifest] {len(df)} rows -> {out}")
    print(f"  feature_dir[0]: {df['feature_dir'].iloc[0]}")
    print(f"  labels_path[0]: {df['labels_path'].iloc[0]}")


if __name__ == "__main__":
    main()
