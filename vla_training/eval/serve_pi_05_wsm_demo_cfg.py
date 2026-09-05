#!/usr/bin/env python3
"""DEMO-CFG pi0.5 policy server — eval counterpart of the pi demo-CFG finetune (doc 15 pi port).

The pi demo finetune trained on windows [z, w_t] (slot -2 = frozen-fusion demo latent via z4.npz; slot -1
= causal w_t) with wsm_cfg_with_future=True. This serve reproduces that ONLINE: per episode pick a
pre-registered registry demo (d.npz), per grid step run the frozen TORCH fusion (torch coexists with JAX
in this process — same as the WSM encoder) on [K=16 causal history x ±W demo window] -> z, and inject
wsm_w_window = [z, w_t] into the obs. The jitted two-pass CFG (shared VLM-prefix KV cache) guides with s.

Controls (serve-time, one ckpt): none | fusion_null (C1b: the fusion's learned null-window — the primary
contrast) | wrong_task (C3) | shuffle (C4) | frozen_phase (C5). NOTE pi has no slot-level null (C1): the
with_future model always reads slot -2; s=0 (both-null unconditional) + C1b cover the pre-registered
contrasts. Phase = demo-paced prop-clamp (G6 winner); tau/clamp telemetry logged per episode.

  python vla_training/eval/serve_pi_05_wsm_demo_cfg.py \
      --finetune-ckpt <pi_demo_.../59999> --fusion-ckpt <wsm2_runs/pi_100k_matched/wsm2_step20000.pt> \
      --encoder-ckpt <wsm_runs/pi_wsm_v1/wsm_step100000.pt> \
      --demo-tokens-root <wsm_demo_tokens/pi_100k_matched> --registry <.../registry_eval.json> \
      --task-lang-table <wsm_policy_feats/pi_step100000/task_lang_table.npz> \
      --guidance-scale 1.5 --control none --port 8000
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from vla_training.eval.serve_pi_05_wsm import _PI_PROPRIO_DIM, WSMPiInferWrapper


def build_policy(finetune_ckpt: str, config_name: str, max_token_len: int, guidance_scale: float, p_drop: float):
    import os

    import openpi.models.pi0_config as pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as _config

    model = pi0_config.Pi0Config(
        pi05=True,
        max_token_len=int(max_token_len),
        wsm_cfg=True,
        wsm_cfg_p_drop=float(p_drop),
        wsm_cfg_with_future=True,  # demo slot LIVE
        wsm_cfg_guidance_scale=float(guidance_scale),
    )
    cfg = _config.TrainConfig(
        name=config_name,
        exp_name="pi05_rc365_wsm_cfg_ft",
        model=model,
        data=_config.LeRobotRobocasaDataConfig(data_dirs=[]),
    )
    print(f"[serve-pi-demo] config={config_name} with_future=True s={guidance_scale}", flush=True)
    return policy_config.create_trained_policy(cfg, os.path.expanduser(finetune_ckpt))


class DemoPiInferWrapper(WSMPiInferWrapper):
    """Extends the proven pi WSM wrapper: history ring (K=16) + registry demo + phase clamp + torch
    fusion -> injects wsm_w_window = [z, omega_t] (the exact training layout)."""

    # Demo-guided state is intentionally outside the current study and remains singleton-only. Prevent
    # the base wrapper's identity-aware batch method from being selected for this different state machine.
    infer_batch = None

    def __init__(
        self, policy, tap, conditioner, table, *, stride, expanded_table, demo_mgr, fusion, window: int, control: str
    ):
        import os

        if int(os.environ.get("WSM_ENVS_PER_GPU", "1")) != 1:
            raise RuntimeError("demo-guided pi wrapper is singleton-only; require WSM_ENVS_PER_GPU=1")
        super().__init__(policy, tap, conditioner, table, stride=stride, expanded_table=expanded_table)
        self._mgr, self._fusion, self._W, self._control = demo_mgr, fusion, window, control
        self._demo = None
        self._meta = None
        self._taus: list[int] = []
        self._ep = 0

    def infer(self, obs: dict, **kwargs) -> dict:
        import torch

        from workspace_models.features.wsm_align import demo_window_at

        t = int(obs.get("wsm_t", 0))
        task = obs.get("wsm_task")
        if task is not None and not isinstance(task, str):
            task = str(np.asarray(task).item()) if np.ndim(task) else str(task)

        if t == 0:
            if task is None or task not in self._table:
                raise RuntimeError(f"[serve-pi-demo] wsm_t==0 needs a known wsm_task; got {task!r}")
            if self._taus:
                M = self._meta["M"]
                clamp = float((np.asarray(self._taus) >= M - 1).mean())
                print(
                    f"[serve-pi-demo] ep{self._ep} {self._meta} steps={len(self._taus)} clamp_frac={clamp:.2f}",
                    flush=True,
                )
            self._cond.reset(self._table[task])
            self._last_grid = -1
            self._demo, self._meta = self._mgr.pick(task)
            self._taus, self._ep = [], self._ep + 1

        grid = math.floor(t / self._stride)
        if grid != self._last_grid:
            prompt = (
                (self._expanded.get(task) if self._expanded else None)
                or obs.get("wsm_prompt")
                or obs.get("prompt")
                or ""
            )
            patch, proprio = self._tap_frame(obs, prompt)
            w_window, lang = self._cond.step(patch, proprio)  # [K=16, 512] causal ring
            ww = w_window.float()
            if not torch.isfinite(ww).all():
                raise RuntimeError(f"[serve-pi-demo] NON-FINITE w at t={t}")
            d = self._demo
            M = d.shape[0]
            tau = M // 2 if self._control == "frozen_phase" else min(grid, M - 1)
            self._taus.append(int(tau))
            idx, off, mask = demo_window_at(M, int(tau), self._W)
            fp = next(self._fusion.parameters())
            with torch.no_grad():
                o = self._fusion(
                    ww[None].to(fp.device, fp.dtype),
                    d[torch.from_numpy(idx).to(fp.device)][None].to(fp.dtype),
                    torch.from_numpy(off).to(fp.device)[None],
                    torch.from_numpy(mask).to(fp.device)[None],
                    torch.as_tensor(self._table[task], dtype=fp.dtype, device=fp.device)[None],
                    drop_demo=torch.tensor([self._control == "fusion_null"], device=fp.device),
                )
            z = o["z"][0].float().cpu().numpy()
            if not np.isfinite(z).all():
                raise RuntimeError("[serve-pi-demo] NON-FINITE z")
            w_t = ww[-1].cpu().numpy()
            self._cur = (
                np.stack([z, w_t]).astype(np.float32),  # [2,512]: [-2]=z, [-1]=w_t
                lang.float().cpu().numpy(),
            )
            self._last_grid = grid

        if self._cur is None:
            raise RuntimeError("[serve-pi-demo] no window computed before infer")
        w_window, lang = self._cur
        inj = dict(obs)
        inj["wsm_w_window"] = np.asarray(w_window, dtype=np.float32)
        inj["wsm_lang"] = np.asarray(lang, dtype=np.float32)
        for k in ("wsm_t", "wsm_task", "wsm_prompt"):
            inj.pop(k, None)
        return self._policy.infer(inj, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True)
    ap.add_argument("--fusion-ckpt", required=True)
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--demo-tokens-root", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--task-lang-table", required=True)
    ap.add_argument("--guidance-scale", type=float, required=True)
    ap.add_argument(
        "--control", default="none", choices=["none", "fusion_null", "wrong_task", "shuffle", "frozen_phase"]
    )
    ap.add_argument("--demo-index", type=int, default=0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--config-name", default="pi05_robocasa_wsm_cfg_ft")
    ap.add_argument("--configs-dir", default=None)
    ap.add_argument("--max-token-len", type=int, default=200)
    ap.add_argument("--p-drop", type=float, default=0.2)
    ap.add_argument("--k-window", type=int, default=16, help="history ring for the fusion (== fusion.k_hist)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--tap-prompt", default="expanded", choices=["expanded", "terse"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import torch
    from openpi.serving import websocket_policy_server

    from vla_training.eval._groot_wsm_demo_eval import DemoManager
    from vla_training.eval._groot_wsm_eval import (
        WSMEvalConditioner,
        load_task_expanded_table,
        load_task_lang_table,
    )
    from vla_training.eval.serve_pi_05_wsm_cfg import assert_cfg_conditioner_loaded
    from workspace_models.features.generate_policy_features import load_wsm
    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap
    from workspace_models.networks.demo_fusion import HistoryDemoFusion

    policy = build_policy(args.finetune_ckpt, args.config_name, args.max_token_len, args.guidance_scale, args.p_drop)
    assert_cfg_conditioner_loaded(policy, args.guidance_scale)
    # S5 twin for pi: slot2 must be TRAINED (a w_t-only CFG ckpt would silently serve as no-demo)
    import jax.numpy as jnp

    pn = float(jnp.linalg.norm(jnp.asarray(policy._model.wsm_cfg_cond.proj_next_out.kernel.value)))
    assert pn > 0, "[serve-pi-demo] proj_next_out is ALL ZERO — not a demo-CFG ckpt"
    print(f"[serve-pi-demo] slot2 trained: proj_next_out L2={pn:.4g}", flush=True)

    ck = torch.load(args.fusion_ckpt, map_location="cpu", weights_only=False)
    fusion = HistoryDemoFusion()
    fusion.load_state_dict(ck["fusion"])
    fusion = fusion.to(args.device).eval()
    fusion.requires_grad_(False)
    mgr = DemoManager(
        args.demo_tokens_root,
        args.registry,
        fusion_registry_sha=ck.get("registry_sha", "?"),
        device=args.device,
        control=args.control,
        demo_index=args.demo_index,
    )

    tap = Pi05BackboneTap(
        args.finetune_ckpt,
        config_name="pi05_rc_mg60_bal33",
        **({"configs_dir": args.configs_dir} if args.configs_dir else {}),
    )
    encoder, _ = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PI_PROPRIO_DIM)
    conditioner = WSMEvalConditioner(encoder, k_window=args.k_window, stride=args.stride, device=args.device)
    table = load_task_lang_table(args.task_lang_table)
    expanded = load_task_expanded_table(args.task_lang_table) if args.tap_prompt == "expanded" else None

    wrapped = DemoPiInferWrapper(
        policy,
        tap,
        conditioner,
        table,
        stride=args.stride,
        expanded_table=expanded,
        demo_mgr=mgr,
        fusion=fusion,
        window=args.window,
        control=args.control,
    )
    print(
        f"[serve-pi-demo] ✓ ready on {args.host}:{args.port} (s={args.guidance_scale} "
        f"control={args.control} W={args.window})",
        flush=True,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
