"""ONLINE w_t for CFG-conditioned GR00T eval (server-side). The CFG-trained policy conditions the action
denoiser on the workspace latent; at eval we tap the rollout, encode the causal prefix, and feed the
NEWEST w_t into action_input so the action head's two-pass CFG get_action fires (guidance scale set on the
head). For demo-ICL later, the same seam carries w_t (+ w_next) from a human demo instead of the rollout.

Reuses the injection-era WSMEvalConditioner (frozen encoder + per-episode causal buffer + finite-w guard);
we just take the newest latent (k_window=1) as the single w_t the CFG conditioner expects. See doc 12.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vla_training.eval._groot_wsm_eval import WSMEvalConditioner


def restore_cfg_weights(model, finetune_ckpt: str | Path) -> int:
    """Copy the trained WSMCfgConditioner weights (action_head.wsm_cfg.*) from the finetune ckpt — the HF
    load drops them (we attach a fresh conditioner after from_pretrained). Returns #tensors restored."""
    ck = Path(finetune_ckpt).expanduser()
    sd = None
    if ck.is_dir():
        from safetensors.torch import load_file

        shards = sorted(ck.glob("*.safetensors"))
        if shards:
            for p in shards:
                part = load_file(str(p))
                sd = part if sd is None else {**sd, **part}
        else:
            for p in sorted(ck.glob("pytorch_model*.bin")):
                part = torch.load(p, map_location="cpu")
                sd = part if sd is None else {**sd, **part}
    else:
        blob = torch.load(ck, map_location="cpu")
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    if sd is None:
        raise FileNotFoundError(f"[wsm-cfg-eval] no model weights under {ck}")
    pref = "action_head.wsm_cfg."
    cfg_sd = {k.split(pref, 1)[1]: v for k, v in sd.items() if pref in k}
    if not cfg_sd:
        raise RuntimeError(f"[wsm-cfg-eval] no '{pref}*' weights in {ck} — was this a CFG finetune ckpt?")
    missing, unexpected = model.action_head.wsm_cfg.load_state_dict(cfg_sd, strict=False)
    print(
        f"[wsm-cfg-eval] restored {len(cfg_sd)} conditioner tensors from {ck} "
        f"(missing={list(missing)} unexpected={list(unexpected)})",
        flush=True,
    )
    return len(cfg_sd)


def install_wsm_cfg_eval(
    policy,
    conditioner: WSMEvalConditioner,
    task_lang_table: dict[str, np.ndarray],
    w_next_table: dict[str, np.ndarray] | None = None,
) -> None:
    """Patch the server-facing policy: per get_action, tap the obs -> conditioner -> set action_input.w_t
    (newest causal latent). reset(task) clears the buffer + sets the task language. The head's two-pass
    CFG get_action reads action_input.w_t and self.guidance_scale. (w_next_table is for the future ICL
    path: a per-task demo future latent; None in the POC.)"""
    from workspace_models.features.backbone_tap import BackboneTap

    inner = getattr(policy, "policy", policy)
    tap = BackboneTap(inner)
    ah = inner.model.action_head  # WSMCfgActionHead (reads ah._cfg_eval)
    cdt = next(ah.wsm_cfg.parameters()).dtype
    _orig_get_action = inner.get_action
    _orig_reset = policy.reset
    state = {"task": None}

    def get_action(observation, *args, **kwargs):
        r = tap.tap(observation)
        # The conditioner builds a single [1,w_dim] w_t; a B>1 call would broadcast the SAME w_t to all
        # samples (silently wrong per-sample conditioning). The rollout serve is B=1 — assert it. (review fix)
        assert r.patch_tokens.shape[0] == 1, (
            f"[wsm-cfg-eval] expected batch 1 at serve, got {r.patch_tokens.shape[0]} (w_t would mis-broadcast)"
        )
        patch = r.patch_tokens[0].float().cpu().numpy()
        proprio = r.state_emb[0, 0].float().cpu().numpy()
        w_window, _lang = conditioner.step(patch, proprio)  # [K=1, w_dim] (finite-w guarded inside)
        w_t = w_window[-1].unsqueeze(0).to(dtype=cdt)  # [1, w_dim] newest
        w_next = None
        if w_next_table is not None and state["task"] in w_next_table:
            w_next = torch.as_tensor(w_next_table[state["task"]], dtype=cdt, device=w_t.device).unsqueeze(0)
        ah._cfg_eval = (w_t, w_next)  # head stash -> two-pass CFG get_action
        return _orig_get_action(observation, *args, **kwargs)

    def reset(options=None):
        task = (options or {}).get("task") if isinstance(options, dict) else None
        if task is None or task not in task_lang_table:
            raise RuntimeError(
                f"[wsm-cfg-eval] reset needs a known task; got {task!r} (table has {len(task_lang_table)} tasks)"
            )
        state["task"] = task
        conditioner.reset(task_lang_table[task])
        return _orig_reset(options) if _orig_reset is not None else {"status": "ok"}

    inner.get_action = get_action
    policy.reset = reset
    print(
        f"[wsm-cfg-eval] installed: stride={conditioner.stride} tasks={len(task_lang_table)} "
        f"(get_action sets action_input.w_t; guidance on the head)",
        flush=True,
    )
