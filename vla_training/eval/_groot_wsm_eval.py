"""ONLINE omega_t for WSM-conditioned GR00T eval (server-side). The recon found there is NO eval-time omega_t
path: training injects precomputed omega_t, but at eval the modulator falls back to identity, so a
WSM-conditioned policy would DEPLOY AS THE Eval1 BASELINE. This builds the missing piece.

At eval the policy decides every exec_steps==stride==8 env steps, so each get_action == ONE new stride-8
grid frame — exactly the grid the encoder was trained on. We keep a per-EPISODE causal buffer of the
backbone's patch tokens (+ proprio), run the FROZEN WorkspaceEncoder over the running prefix 1..t each
step (the encoder is full-history, c_horizon=1000 >> episode length, so a truncated window would NOT
reproduce trained omega_t — keep the whole prefix; it's cheap: <~150 grid tokens, ~24M-param encoder), and
take the causal K-window via the SHARED wsm_align helper so train/eval omega_t match. The window + the task
language vector are set on the action head's `_wsm_cond`, which the trained zero-init TokenModulator then
applies (identical seam to training, _groot_wsm_common.install_wsm_action_head).

Episode boundaries come from the PolicyServer's `reset` endpoint (client calls it per env.reset); reset
clears the buffer and sets the current task's language (task_lang_table, the per-task mean expanded-prompt
embedding — the same lang the precompute used, train/eval consistent). Backbone-agnostic core
(WSMEvalConditioner); GR00T glue installs it onto a Gr00tPolicy. See internal_planning_and_todos (09).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from workspace_models.features.wsm_align import causal_window_indices


class WSMEvalConditioner:
    """Online-omega_t core with a compact, per-episode fused-input cache.

    Raw patch grids are projected exactly once when a frame arrives. Each env stores only fused frame
    tokens and projected language conditions [F, dim]; the frozen causal temporal stack is rerun over
    that compact prefix. step_many batches frame fusion across all advancing envs, then batches
    temporal encoding by equal history length.
    """

    def __init__(self, encoder, k_window: int, stride: int = 8, device: str = "cuda:0"):
        self.encoder = encoder
        self.k = int(k_window)
        self.stride = int(stride)
        self.device = device
        self.reset(None)

    def reset(self, lang_vec) -> None:
        """Drop only this episode's compact cache and install its task-language vector."""
        self._fused: list[torch.Tensor] = []
        self._conds: list[torch.Tensor] = []
        self._lang = (
            None
            if lang_vec is None
            else torch.as_tensor(np.asarray(lang_vec), dtype=torch.float32, device=self.device)
        )

    @classmethod
    @torch.no_grad()
    def step_many(cls, conditioners, patches, proprio):
        """Advance independent caches with batched fusion and temporal encoder calls.

        Frame-local PatchPool/proprio/language projection is batched by shared encoder and input shape.
        Temporal calls are then grouped by equal post-append history length, preserving exact full-prefix
        causal semantics for asynchronous clients.
        """
        conditioners = list(conditioners)
        patches = list(patches)
        proprio = list(proprio)
        if not (len(conditioners) == len(patches) == len(proprio)):
            raise ValueError("WSMEvalConditioner.step_many needs equal conditioner/patch/proprio lengths")
        if not conditioners:
            return []

        patch_tensors = []
        proprio_tensors = []
        for conditioner, patch, prop in zip(conditioners, patches, proprio):
            if conditioner._lang is None:
                raise RuntimeError("WSMEvalConditioner.step_many before reset(lang_vec) — no episode language set")
            if not (
                callable(getattr(conditioner.encoder, "fuse_inputs", None))
                and callable(getattr(conditioner.encoder, "encode_fused", None))
            ):
                raise RuntimeError(
                    "served WorkspaceModel lacks fuse_inputs/encode_fused; refusing the quadratic "
                    "raw-history projection fallback"
                )
            patch_tensors.append(torch.as_tensor(np.asarray(patch), dtype=torch.float32, device=conditioner.device))
            proprio_tensors.append(torch.as_tensor(np.asarray(prop), dtype=torch.float32, device=conditioner.device))

        fusion_groups = {}
        for index, conditioner in enumerate(conditioners):
            key = (
                id(conditioner.encoder),
                str(conditioner.device),
                tuple(patch_tensors[index].shape),
                tuple(proprio_tensors[index].shape),
                tuple(conditioner._lang.shape),
            )
            fusion_groups.setdefault(key, []).append(index)

        for indices in fusion_groups.values():
            first = conditioners[indices[0]]
            patch_batch = torch.stack([patch_tensors[index] for index in indices])[:, None]
            proprio_batch = torch.stack([proprio_tensors[index] for index in indices])[:, None]
            lang_batch = torch.stack([conditioners[index]._lang for index in indices])[:, None]
            fused, cond = first.encoder.fuse_inputs(patch_batch, proprio_batch, lang_batch)
            if fused.shape[:2] != (len(indices), 1) or cond.shape != fused.shape:
                raise RuntimeError(
                    f"[wsm-eval] invalid fused frame batch: fused={tuple(fused.shape)} cond={tuple(cond.shape)}"
                )
            for batch_index, source_index in enumerate(indices):
                conditioners[source_index]._fused.append(fused[batch_index, 0])
                conditioners[source_index]._conds.append(cond[batch_index, 0])

        temporal_groups = {}
        for index, conditioner in enumerate(conditioners):
            key = (
                id(conditioner.encoder),
                str(conditioner.device),
                len(conditioner._fused),
                conditioner.k,
                conditioner.stride,
                tuple(conditioner._fused[0].shape),
                tuple(conditioner._conds[0].shape),
            )
            temporal_groups.setdefault(key, []).append(index)

        results = [None] * len(conditioners)
        for indices in temporal_groups.values():
            first = conditioners[indices[0]]
            frames = len(first._fused)
            fused_batch = torch.stack([torch.stack(conditioners[index]._fused) for index in indices])
            cond_batch = torch.stack([torch.stack(conditioners[index]._conds) for index in indices])
            omega = first.encoder.encode_fused(fused_batch, cond_batch)
            # Check the complete cached input and output graph at one synchronization boundary.  A
            # Python truth test immediately after ``fuse_inputs`` forced an extra CUDA round trip on
            # every new grid frame; the policy needs the completed omega below anyway, so combining
            # these checks preserves fail-loud behavior without serializing fusion and temporal work.
            finite = torch.isfinite(fused_batch).all() & torch.isfinite(cond_batch).all() & torch.isfinite(omega).all()
            if not finite:
                raise RuntimeError(
                    f"[wsm-eval] NON-FINITE online workspace graph at F={frames} (omega finite frac "
                    f"{torch.isfinite(omega).float().mean().item():.3f}) — refusing to serve"
                )
            frame_indices = np.arange(frames, dtype=np.int64) * first.stride
            window_indices = causal_window_indices(frame_indices, int(frame_indices[-1]), first.k)
            window_indices = torch.as_tensor(window_indices, device=first.device)
            for batch_index, source_index in enumerate(indices):
                results[source_index] = (
                    omega[batch_index, window_indices],
                    conditioners[source_index]._lang,
                )
        return results

    @torch.no_grad()
    def step(self, patch, proprio) -> tuple[torch.Tensor, torch.Tensor]:
        """K=1 request path through the same compact-cache implementation."""
        return self.step_many([self], [patch], [proprio])[0]


# --------------------------------------------------------------------------------------------------
# GR00T glue (heavy imports deferred): tap the policy's obs -> conditioner -> set action_head._wsm_cond.
# --------------------------------------------------------------------------------------------------
def load_task_lang_table(path: str | Path) -> dict[str, np.ndarray]:
    """task_lang_table.npz {tasks:[T] str, lang:[T,lang_dim] fp16} -> {task_name: lang_vec fp32}."""
    d = np.load(Path(path).expanduser())
    return {str(t): np.asarray(v, dtype=np.float32) for t, v in zip(d["tasks"], d["lang"])}


def load_task_expanded_table(path: str | Path) -> dict[str, str]:
    """Per-task Qwen EXPANDED-prompt STRING from task_lang_table.npz (key 'expanded', written by
    make_task_lang_table --cache-root). {} if absent. Used by serve --tap-prompt expanded so the eval
    backbone tap matches the cache's expanded-prompt features (closes the WSM-only encoder-input shift)."""
    d = np.load(Path(path).expanduser(), allow_pickle=True)
    if "expanded" not in d.files:
        return {}
    return {str(t): str(e) for t, e in zip(d["tasks"], d["expanded"])}


def restore_modulator_weights(model, finetune_ckpt: str | Path) -> int:
    """install_wsm_action_head attaches a FRESH zero-init modulator AFTER the HF load, so the trained
    modulator weights in the finetune ckpt are NOT loaded by from_pretrained. Copy them in explicitly.
    Returns the number of modulator tensors restored (asserts > 0)."""
    ck = Path(finetune_ckpt).expanduser()
    sd = None
    if ck.is_dir():
        st = ck / "model.safetensors"
        if st.exists():
            from safetensors.torch import load_file

            sd = load_file(str(st))
        else:  # sharded or .bin fallback
            for p in sorted(ck.glob("*.safetensors")) or sorted(ck.glob("pytorch_model*.bin")):
                from safetensors.torch import load_file

                part = load_file(str(p)) if p.suffix == ".safetensors" else torch.load(p, map_location="cpu")
                sd = part if sd is None else {**sd, **part}
    else:
        blob = torch.load(ck, map_location="cpu")
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    if sd is None:
        raise FileNotFoundError(f"[wsm-eval] no model weights found under {ck}")
    pref = "action_head.modulator."
    # match any key containing the modulator prefix (tolerate a leading 'model.' etc.), strip up to it
    mod_sd = {k.split(pref, 1)[1]: v for k, v in sd.items() if pref in k}
    if not mod_sd:
        raise RuntimeError(f"[wsm-eval] no '{pref}*' weights in {ck} — was this a WSM finetune ckpt?")
    missing, unexpected = model.action_head.modulator.load_state_dict(mod_sd, strict=False)
    print(
        f"[wsm-eval] restored {len(mod_sd)} modulator tensors from {ck} "
        f"(missing={list(missing)} unexpected={list(unexpected)})",
        flush=True,
    )
    return len(mod_sd)


def install_wsm_eval(
    policy,
    conditioner: WSMEvalConditioner,
    task_lang_table: dict[str, np.ndarray],
    embodiment_tag: str = "new_embodiment",
) -> None:
    """Patch the SERVER-FACING policy in place: per get_action, tap the obs (same BackboneTap that built
    the cache, a 2nd frozen backbone forward), feed the conditioner, and set action_head._wsm_cond before
    the policy's own forward. reset(task) clears the buffer + sets the task language. `policy` may be a
    Gr00tPolicy or a Gr00tSimPolicyWrapper (which holds the inner Gr00tPolicy as .policy) — the server
    calls the outermost object, so we patch THAT but tap/modulate via the inner Gr00tPolicy."""
    from workspace_models.features.backbone_tap import BackboneTap

    inner = getattr(policy, "policy", policy)  # unwrap Gr00tSimPolicyWrapper if present
    tap = BackboneTap(inner)  # reuse the policy's model for the tap
    ah = inner.model.action_head
    # PATCH get_action on the INNER policy: Gr00tSimPolicyWrapper.get_action converts the flat sim obs to
    # the nested form and calls self.policy.get_action(new_obs) — that nested obs is EXACTLY what
    # BackboneTap.tap (-> policy._unbatch_observation) needs. (Patching the outer wrapper would feed tap
    # the un-converted flat obs.) reset stays on the OUTERMOST object the server's reset endpoint calls.
    _orig_get_action = inner.get_action
    _orig_reset = policy.reset

    def _tap_obs(observation):
        r = tap.tap(observation)  # TapResult (batch 1)
        patch = r.patch_tokens[0].float().cpu().numpy()  # [P, backbone_dim]
        proprio = r.state_emb[0, 0].float().cpu().numpy()  # [Dp]  (state_emb is [B,1,Dp])
        return patch, proprio

    def get_action(observation, *args, **kwargs):
        patch, proprio = _tap_obs(observation)
        w_window, lang = conditioner.step(patch, proprio)
        dt = ah.modulator.gen[-1].weight.dtype
        ah._wsm_cond = (w_window.unsqueeze(0).to(dt), lang.unsqueeze(0).to(dt))  # [1,K,w_dim], [1,lang_dim]
        return _orig_get_action(observation, *args, **kwargs)

    def reset(options=None):
        task = (options or {}).get("task") if isinstance(options, dict) else None
        if task is None or task not in task_lang_table:
            raise RuntimeError(
                f"[wsm-eval] reset needs a known task; got {task!r} (table has {len(task_lang_table)} tasks)"
            )
        conditioner.reset(task_lang_table[task])
        return _orig_reset(options) if _orig_reset is not None else {"status": "ok"}

    inner.get_action = get_action  # inner = the obs tap needs; sim wrapper calls self.policy.get_action
    policy.reset = reset  # outer = what the server's reset endpoint invokes
    print(
        f"[wsm-eval] installed: K={conditioner.k} stride={conditioner.stride} "
        f"tasks={len(task_lang_table)} (get_action on inner, reset on outer)",
        flush=True,
    )
