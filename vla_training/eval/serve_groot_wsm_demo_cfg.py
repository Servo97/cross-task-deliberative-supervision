#!/usr/bin/env python3
"""DEMO-CFG GR00T N1.7 policy server (eval counterpart of finetune_groot_17_with_wsm_demo_cfg).

Loads the demo-CFG finetune ckpt, re-attaches the frozen fusion + conditioner (HF load drops attachments),
restores the TRAINED conditioner weights, installs the online demo-conditioning (registry demo -> phase
ladder -> ±W window -> frozen fusion -> z), and serves the inherited two-pass CFG:
    v = v_uncond + s*(v_cond - v_uncond)
Controls are serve flags on this ONE checkpoint: --control {none,null,fusion_null,wrong_task,shuffle,
frozen_phase} (doc 15 D20). One serve process per (s, control).

  python vla_training/eval/serve_groot_wsm_demo_cfg.py \
      --finetune-ckpt <.../checkpoint-XXXX> --fusion-ckpt <wsm2_runs/.../wsm2_step20000.pt> \
      --encoder-ckpt <wsm_runs/groot_wsm_base/wsm_step65000.pt> \
      --demo-tokens-root <wsm_demo_tokens/orig_65k_matched> --registry <.../registry_eval.json> \
      --task-lang-table <wsm_policy_feats/groot_65k/task_lang_table.npz> \
      --guidance-scale 1.5 --control none --port 5600
"""

from __future__ import annotations

import argparse

_PROPRIO_DIM = {"groot": 1536}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True, help="demo-CFG finetune model dir (HF checkpoint-XXXX)")
    ap.add_argument("--fusion-ckpt", required=True, help="WSMv2 ckpt (fusion weights + registry sha)")
    ap.add_argument("--encoder-ckpt", required=True, help="FROZEN WSMv1 ckpt for online w_t (65k arm)")
    ap.add_argument("--demo-tokens-root", required=True, help="d.npz root (frozen DemoEncoder outputs)")
    ap.add_argument("--registry", required=True, help="registry_eval.json (pre-registered demo2 set)")
    ap.add_argument("--task-lang-table", required=True)
    ap.add_argument("--guidance-scale", type=float, default=0.0)
    ap.add_argument(
        "--control", default="none", choices=["none", "null", "fusion_null", "wrong_task", "shuffle", "frozen_phase"]
    )
    ap.add_argument("--demo-index", type=int, default=0, help="which registry demo (variance check)")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--p-drop", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=8, help="cache grid stride (== eval exec_steps)")
    ap.add_argument("--log-dir", default=None, help="per-episode tau/meta npz logs")
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--no-sim-wrapper", action="store_true")
    args = ap.parse_args()

    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    from vla_training.eval._groot_wsm_cfg_eval import load_task_lang_table, restore_cfg_weights
    from vla_training.eval._groot_wsm_demo_eval import (
        DemoManager,
        WSMEvalConditioner,
        install_wsm_demo_cfg_eval,
    )
    from vla_training.train.train_base._groot_wsm_demo_cfg_common import attach_wsm_demo_cfg
    from workspace_models.features.generate_policy_features import load_wsm

    print(f"[serve-demo-cfg] loading finetune ckpt {args.finetune_ckpt}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag), model_path=args.finetune_ckpt, device=args.device
    )
    # re-attach (frozen fusion from the wsm2 ckpt + conditioner + temb wrapper), set s, restore trained weights
    attach_wsm_demo_cfg(
        policy.model,
        args.fusion_ckpt,
        w_dim=args.w_dim,
        p_drop=args.p_drop,
        diag_every=1,
        guidance_scale=args.guidance_scale,
    )
    n = restore_cfg_weights(policy.model, args.finetune_ckpt)
    assert n > 0, "no conditioner weights restored — refusing to serve"
    ah = policy.model.action_head
    # S5 guard: an A1-lineage ckpt passes restore counts but has an untouched (zero) slot2 projection —
    # that would silently serve as w_t-only CFG and fake a 'demo refuted' result.
    out_l2 = float(torch.linalg.norm(ah.wsm_cfg.proj_next[-1].weight.detach().float()))
    assert out_l2 > 0, "[serve-demo-cfg] proj_next output is ALL ZERO — not a demo-CFG ckpt (slot2 untrained)"
    print(f"[serve-demo-cfg] slot2 trained: proj_next L2={out_l2:.4g}", flush=True)

    encoder, _meta = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PROPRIO_DIM["groot"])
    conditioner = WSMEvalConditioner(encoder, k_window=ah.wsm_fusion.k_hist, stride=args.stride, device=args.device)
    table = load_task_lang_table(args.task_lang_table)
    mgr = DemoManager(
        args.demo_tokens_root,
        args.registry,
        fusion_registry_sha=ah._wsm2_meta.get("registry_sha", "?"),
        device=args.device,
        control=args.control,
        demo_index=args.demo_index,
    )

    server_policy = policy if args.no_sim_wrapper else Gr00tSimPolicyWrapper(policy)
    install_wsm_demo_cfg_eval(
        server_policy,
        conditioner,
        table,
        mgr,
        ah.wsm_fusion,
        window=args.window,
        control=args.control,
        log_dir=args.log_dir,
    )

    print(
        f"[serve-demo-cfg] ✓ ready on {args.host}:{args.port} (s={args.guidance_scale} "
        f"control={args.control} W={args.window} demo_idx={args.demo_index} "
        f"meta={ah._wsm2_meta})",
        flush=True,
    )
    PolicyServer(policy=server_policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
