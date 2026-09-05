#!/usr/bin/env python3
"""CFG-CONDITIONED GR00T N1.7 policy server (eval counterpart of finetune_groot_17_with_wsm_cfg).

Loads the CFG finetune ckpt, re-attaches the WSMCfgConditioner (the HF load drops it), restores the
trained conditioner weights, sets the eval guidance scale, and installs the online-w_t conditioner so the
action head's two-pass CFG get_action fires:  v = v_uncond + s*(v_cond - v_uncond).
  s = 0  -> exact baseline (single uncond pass)         s = 1 -> pure conditional        s > 1 -> amplified

  python vla_training/eval/serve_groot_wsm_cfg.py \
      --finetune-ckpt <target_ft_wsm/.../checkpoint-XXXX> --encoder-ckpt <wsm_step50000.pt> \
      --task-lang-table <wsm_policy_feats/groot_step50000/task_lang_table.npz> \
      --guidance-scale 1.0 --port 5600
"""

from __future__ import annotations

import argparse

_PROPRIO_DIM = {"groot": 1536}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True, help="CFG finetune model dir (HF checkpoint-XXXX)")
    ap.add_argument("--encoder-ckpt", required=True, help="FROZEN WorkspaceModel ckpt (wsm_step*.pt)")
    ap.add_argument("--task-lang-table", required=True, help="task_lang_table.npz (next to the precompute _meta.json)")
    ap.add_argument("--guidance-scale", type=float, default=0.0, help="CFG scale s (0=baseline, 1=cond, >1=amplified)")
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--p-drop", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=8, help="cache grid stride (== eval exec_steps)")
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--no-sim-wrapper", action="store_true")
    args = ap.parse_args()

    import json
    from pathlib import Path

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    from vla_training.eval._groot_wsm_cfg_eval import (
        install_wsm_cfg_eval,
        load_task_lang_table,
        restore_cfg_weights,
    )
    from vla_training.eval._groot_wsm_eval import WSMEvalConditioner
    from vla_training.train.train_base._groot_wsm_cfg_common import attach_wsm_cfg
    from workspace_models.features.generate_policy_features import load_wsm

    print(f"[serve-cfg] loading finetune ckpt {args.finetune_ckpt}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag), model_path=args.finetune_ckpt, device=args.device
    )
    # re-attach the conditioner + temb wrapper (from_pretrained bypasses the train-path patch), set s, restore
    attach_wsm_cfg(
        policy.model,
        w_dim=args.w_dim,
        p_drop=args.p_drop,
        with_future=False,
        diag_every=1,
        guidance_scale=args.guidance_scale,
    )
    n = restore_cfg_weights(policy.model, args.finetune_ckpt)
    assert n > 0, "no conditioner weights restored — refusing to serve an untrained (≈baseline) conditioner"

    encoder, _meta = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PROPRIO_DIM["groot"])
    # encoder-provenance guard (same as serve_groot_wsm): served encoder must match the precompute the
    # conditioner trained on (encoder step == task-lang-table _meta wsm_step). Finite-w is guarded in step().
    meta_path = Path(args.task_lang_table).expanduser().parent / "_meta.json"
    if meta_path.exists():
        pstep = json.loads(meta_path.read_text()).get("wsm_step")
        estep = _meta.get("step")
        if estep is not None and pstep is not None and int(estep) != int(pstep):
            raise RuntimeError(
                f"[serve-cfg] ENCODER/PRECOMPUTE MISMATCH: encoder step {estep} != precompute "
                f"wsm_step {pstep} ({meta_path}). Serving an encoder the conditioner wasn't trained on."
            )
        print(f"[serve-cfg] encoder provenance OK: step {estep} == precompute {pstep}", flush=True)
    else:
        print(f"[serve-cfg] WARNING: no _meta.json at {meta_path}; cannot verify encoder provenance.", flush=True)

    conditioner = WSMEvalConditioner(encoder, k_window=1, stride=args.stride, device=args.device)
    table = load_task_lang_table(args.task_lang_table)

    server_policy = policy if args.no_sim_wrapper else Gr00tSimPolicyWrapper(policy)
    install_wsm_cfg_eval(server_policy, conditioner, table)  # POC: no w_next table (self-conditioning)

    print(
        f"[serve-cfg] ✓ CFG server ready on {args.host}:{args.port} (s={args.guidance_scale}, "
        f"{len(table)} tasks, encoder={args.encoder_ckpt})",
        flush=True,
    )
    PolicyServer(policy=server_policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
