#!/usr/bin/env python3
"""WSM-CONDITIONED GR00T N1.7 policy server (the eval-side counterpart of finetune_groot_17_with_wsm).

Mirrors gr00t/eval/run_gr00t_server.py (Gr00tPolicy + Gr00tSimPolicyWrapper + PolicyServer, embodiment
NEW_EMBODIMENT) but layers in ONLINE w_t so the trained zero-init modulator actually fires at eval
(without this, the modulator falls back to identity and the policy deploys as the Eval1 baseline):

  1. install_wsm_action_head -> the loaded action head is WSMConditioned + carries a modulator.
  2. restore_modulator_weights -> copy the TRAINED modulator from the finetune ckpt (the HF load drops
     it, since install attaches a fresh zero-init modulator AFTER from_pretrained).
  3. WSMEvalConditioner (frozen encoder) + install_wsm_eval -> per get_action, tap the obs, push to the
     per-episode causal buffer, encode the prefix, and set action_head._wsm_cond; reset(task) clears it.

The rollout client (eval_groot_17.py) calls reset(task) per env.reset, supplying the task whose
task_lang_table row is the encoder cond_lang + modulator lang (train/eval-consistent global language).

  python vla_training/eval/serve_groot_wsm.py \
      --finetune-ckpt <target_ft_wsm/.../checkpoint-XXXX> --encoder-ckpt <wsm_step100000.pt> \
      --task-lang-table <wsm_policy_feats/groot_step100000/task_lang_table.npz> \
      --k-window 2 --port 5600
"""

from __future__ import annotations

import argparse

# groot proprio dim fed to the WSM encoder (state_emb 1536); pi would be 2048 (lang_per_frame) + a JAX serve.
_PROPRIO_DIM = {"groot": 1536, "pi": 2048}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True, help="WSM finetune model dir (HF checkpoint-XXXX)")
    ap.add_argument("--encoder-ckpt", required=True, help="FROZEN WorkspaceModel ckpt (wsm_step*.pt)")
    ap.add_argument("--task-lang-table", required=True, help="task_lang_table.npz (make_task_lang_table)")
    ap.add_argument("--backbone", default="groot", choices=["groot"], help="pi is served via JAX, separately")
    ap.add_argument("--k-window", type=int, default=2)
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--lang-dim", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=8, help="cache grid stride (== eval exec_steps)")
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--no-sim-wrapper", action="store_true", help="skip Gr00tSimPolicyWrapper (debug)")
    args = ap.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    from vla_training.eval._groot_wsm_eval import (
        WSMEvalConditioner,
        install_wsm_eval,
        load_task_lang_table,
        restore_modulator_weights,
    )
    from vla_training.train.train_base._groot_wsm_common import attach_wsm_modulator
    from workspace_models.features.generate_policy_features import load_wsm

    # 1. build the policy from the WSM finetune ckpt. NOTE: Gr00tPolicy loads via from_pretrained, which
    #    BYPASSES Gr00tN1d7Pipeline._create_model (the train-path hook), so the modulator is NOT attached
    #    by load (the ckpt's action_head.modulator.* show up as "unused weights"). We attach it directly.
    print(f"[serve-wsm] loading finetune ckpt {args.finetune_ckpt}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag), model_path=args.finetune_ckpt, device=args.device
    )
    # 2. reclass the action head + attach a fresh zero-init modulator, THEN restore the TRAINED weights.
    attach_wsm_modulator(policy.model, w_dim=args.w_dim, k_window=args.k_window, lang_dim=args.lang_dim)
    n = restore_modulator_weights(policy.model, args.finetune_ckpt)
    assert n > 0, "no modulator weights restored — refusing to serve an identity (baseline) policy"

    # 3. frozen encoder + online-w_t conditioner + task language table
    encoder, _meta = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PROPRIO_DIM[args.backbone])

    # ENCODER PROVENANCE GUARD: the served encoder MUST be the one the modulator trained on. The
    # task-lang-table lives in the precompute dir next to _meta.json (records wsm_step + wsm_ckpt); if the
    # served encoder's trained step != the precompute's, we are serving a DIFFERENT encoder -> garbage/NaN w
    # -> silent 0% (the 2026-06-26 groot bug). Fail loud. See [[groot-eval2-nan-encoder-bug]].
    import json as _json
    from pathlib import Path as _Path

    _meta_path = _Path(args.task_lang_table).expanduser().parent / "_meta.json"
    _enc_step = _meta.get("step")
    if _meta_path.exists():
        _pmeta = _json.loads(_meta_path.read_text())
        _pstep = _pmeta.get("wsm_step")
        if _enc_step is not None and _pstep is not None and int(_enc_step) != int(_pstep):
            raise RuntimeError(
                f"[serve-wsm] ENCODER/PRECOMPUTE MISMATCH: --encoder-ckpt is step {_enc_step} but the "
                f"task-lang-table's precompute ({_meta_path}) used step {_pstep} ({_pmeta.get('wsm_ckpt')}). "
                f"Serving an encoder the modulator was NOT trained on -> garbage/NaN w. Pass the encoder "
                f"matching the finetune's WSM_POLICY_FEATS_ROOT."
            )
        print(
            f"[serve-wsm] encoder provenance OK: encoder step {_enc_step} == precompute wsm_step {_pstep}", flush=True
        )
    else:
        print(
            f"[serve-wsm] WARNING: no _meta.json at {_meta_path}; cannot verify the served encoder matches "
            f"the modulator's precompute — ensure --encoder-ckpt is the one the finetune trained on.",
            flush=True,
        )

    conditioner = WSMEvalConditioner(encoder, k_window=args.k_window, stride=args.stride, device=args.device)
    table = load_task_lang_table(args.task_lang_table)

    server_policy = policy if args.no_sim_wrapper else Gr00tSimPolicyWrapper(policy)
    install_wsm_eval(server_policy, conditioner, table)  # patch the SERVER-FACING object; tap via inner

    print(
        f"[serve-wsm] ✓ WSM-conditioned server ready on {args.host}:{args.port} "
        f"(K={args.k_window}, {len(table)} tasks, encoder={args.encoder_ckpt})",
        flush=True,
    )
    PolicyServer(policy=server_policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
