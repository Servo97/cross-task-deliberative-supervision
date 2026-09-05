#!/usr/bin/env python3
"""Export an ω store from an UNTRAINED Stage-E encoder — the negative control for any ω metric.

`train_stage_e.py` already builds a frozen untrained encoder for the G1b control but never writes
its ω store, so a report metric computed on the trained store (binding decodability, retrieval)
has no matched baseline on disk. This writes one, through the SAME `Corpus` and the SAME
`export_omega_store`, so the only difference from a trained cell is the weights.

It matters most for binding decodability: an AdaLN-Zero trunk initialises near-identity, so an
untrained encoder inherits the tap's own structure and can already score above chance. A trained
number is only readable next to this one.

  PYTHONPATH=. python scripts/deliberation/export_untrained_omega.py \
      --labels ~/Research/TRI/wsm_data/deliberation/stage_e_labels/ab38d9efc0c649a3 \
      --tap robocasa=~/Research/TRI/wsm_data/wsm_pooled/pi_100k \
      --tap remembench=~/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k \
      --out ~/Research/TRI/wsm_data/deliberation/stage_e_runs/omega/untrained_rmb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.networks.stage_e_encoder import StageEEncoder  # noqa: E402
from workspace_models.networks.wsm_model import WSMConfig  # noqa: E402
from workspace_models.train.train_wsm_base.train_stage_e import DOMAINS, Corpus, export_omega_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tap", action="append", required=True, help="domain=/path, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    taps = {}
    for entry in args.tap:
        name, _, root = entry.partition("=")
        if name not in DOMAINS:
            raise SystemExit(f"unknown domain {name!r}; expected one of {DOMAINS}")
        taps[name] = str(Path(root).expanduser())

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    corpus = Corpus(Path(args.labels).expanduser(), taps, device)
    cfg = WSMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_dec_layers=2,
        n_heads=args.n_heads,
        k_slots=32,
        backbone_dim=corpus.feat_dim,
        proprio_dim=corpus.lang_dim,
        lang_dim=corpus.lang_dim,
        c_horizon=1000,
        max_t=max(1200, corpus.max_len + 8),
        mlp_ratio=4.0,
        input_norm=False,
    )
    specs = {
        name: {"feat_dim": corpus.feat_dim, "lang_dim": corpus.lang_dim, "index": DOMAINS.index(name)} for name in taps
    }
    encoder = StageEEncoder(cfg, specs).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(
        f"[untrained] {corpus.n_episodes} episodes, domains={encoder.domains} index={encoder.domain_index}", flush=True
    )
    export_omega_store(encoder, corpus, Path(args.out).expanduser(), f"untrained/seed{args.seed}", device)


if __name__ == "__main__":
    main()
