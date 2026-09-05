"""Gated-DeltaNet workspace conditioner (w=8) wired into GR00T N1.7 — the torch twin of the pi0.5
`s1/deltanet` arm (`openpi/models/wsm_current_cond.py::WSMGatedDeltaNetConditioner`).

SEAM. The conditioner emits ONE additive vector that is added to the DiT's `temb` — the single global
AdaLN conditioning bus feeding all 32 transformer blocks plus the output head. That is the exact
GR00T analogue of the pi0.5 `adarms_cond` seam every Stage-S conditioner uses, which is what keeps
the two backbones' arms comparable. Implemented by patching `timestep_encoder.forward` IN PLACE
(never wrapping it in a submodule: a wrapper renames its params to `timestep_encoder.orig.*` in the
saved checkpoint and breaks `from_pretrained` at serve, since the DiT trains under
`tune_diffusion_model`). Idiom copied verbatim from `_groot_wsm_cfg_common.py`.

STATELESS. The delta-rule state S is built inside ONE call from the omega window that was already
shipped and starts at ZERO every call. Nothing carries across policy calls — this is the STEERING
axis, not the RoboTTT fast-weight axis. Do not add a cross-call carry here; that is a different arm.

OMEGA. `w_window [B, K, 512]` rides in on the dataset's per-sample dict, which the GR00T collator
auto-stacks (any unknown numpy key falls through to its `else` branch). The window is built by
`wsm_align.window_at`, whose `causal_window_indices` is kept in lock-step with the pi loader's
`_wsm_causal_window` — same left-padding-repeats-the-oldest-real-row policy, so the two backbones
consume byte-identical windows from the same model-agnostic omega cache.

FINITE GUARD (non-negotiable). A previous eval served an encoder whose weights were NaN and the arm
scored 0% while looking structurally healthy; the failure was invisible because a NaN conditioner
silently poisons `temb`. Every path here hard-fails on a non-finite window or conditioning vector
rather than propagating it. See [[groot-eval2-nan-encoder-bug]].

GR00T-venv-only: every gr00t/torch import is inside a function.
"""

from __future__ import annotations

from vla_training.train.train_base._groot_wsm_common import assert_wt_coverage  # re-export (same contract)

__all__ = [
    "install_wsm_deltanet_dataset",
    "attach_wsm_deltanet",
    "install_wsm_deltanet_action_head",
    "window_len_from_state_dict",
    "assert_wt_coverage",
]


def window_len_from_state_dict(state_dict, prefix: str = "") -> int:
    """Recover the TRAINED window length from `pos_decay_bias`'s leading axis.

    The parameter is [window_len, num_heads], which is deliberately how the trained window is made
    structurally readable from a checkpoint — serve auto-detects it instead of needing a new eval
    flag that could silently disagree with the recipe.
    """
    keys = [k for k in state_dict if k.endswith("pos_decay_bias")]
    if prefix:
        keys = [k for k in keys if k.startswith(prefix)]
    if len(keys) != 1:
        raise KeyError(f"expected exactly one *pos_decay_bias in the checkpoint, found {len(keys)}: {keys[:5]}")
    return int(state_dict[keys[0]].shape[0])


def install_wsm_deltanet_dataset(feats_root: str, window_len: int = 8, demo_fraction: float = 0.30, soup=None) -> None:
    """Attach the causal omega window `w_window [K, 512]` per sample (the conditioner's input).

    `soup` is the resolved dataset soup (list of `{"path", "task", ...}`) used to map a lerobot dir
    to its task name. Pass `gs.soup` from the train entry. It defaults to the RoboCasa target soup
    ONLY for backwards compatibility — on remembench13 the default would resolve zero tasks and
    every sample would raise, because `combined_target_soup` globs a different layout entirely.
    """
    from pathlib import Path

    import gr00t.data.dataset.factory as factory_module
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    # load_w (not load_w_and_lang): the study's shared omega cache is image-derived and carries no
    # lang_global, and this arm never reads language anyway.
    from workspace_models.features.wsm_align import load_w, window_at

    if soup is None:
        from utils.soup import combined_target_soup

        soup = combined_target_soup(demo_fraction)
    feats_root = Path(feats_root).expanduser()
    task_by_dir = {str(Path(m["path"]).resolve()): m["task"] for m in soup}
    task_names = set(task_by_dir.values())
    if not task_names:
        raise ValueError("[wsm-dn] empty soup: cannot map any lerobot dir to a task")

    class WSMDeltaNetShardedSingleStepDataset(ShardedSingleStepDataset):
        def _wsm_task(self) -> str:
            key = str(Path(self.dataset_path).resolve())
            if key in task_by_dir:
                return task_by_dir[key]
            parts = set(Path(self.dataset_path).parts)
            for t in task_names:
                if t in parts:
                    return t
            raise KeyError(f"[wsm-dn] cannot resolve task for dataset_path={self.dataset_path}")

        def _wsm_load(self, ep_idx: int):
            if getattr(self, "_wsm_key", None) == ep_idx:  # per-episode cache (1 load / shard episode)
                return self._wsm_val
            ep_index = int(self.episode_loader.episodes_metadata[ep_idx]["episode_index"])
            fdir = feats_root / self._wsm_task() / f"demo_{ep_index:06d}"
            if not (fdir / "w.npz").exists():
                raise FileNotFoundError(f"[wsm-dn] policy features missing: {fdir}/w.npz")
            w, fi = load_w(fdir)
            self._wsm_key, self._wsm_val = ep_idx, (w, fi)
            return self._wsm_val

        def get_shard(self, idx: int) -> list:
            datapoints = []
            for ep_idx, step_indices in self.sharded_episodes[idx]:
                episode_data = self.episode_loader[ep_idx]
                self._wsm_cur_ep_idx = ep_idx
                for step_index in step_indices:
                    datapoints.append(self.get_datapoint(episode_data, step_index))
            return datapoints

        def get_datapoint(self, episode_data, step_index: int) -> dict:
            out = super().get_datapoint(episode_data, step_index)
            w, fi = self._wsm_load(self._wsm_cur_ep_idx)
            window = np.asarray(window_at(w, fi, int(step_index), window_len), dtype=np.float32)
            if not np.isfinite(window).all():
                raise ValueError(
                    f"[wsm-dn] non-finite omega window from {self._wsm_task()} "
                    f"demo/step={self._wsm_cur_ep_idx}/{step_index} — refusing to train on a NaN "
                    "conditioner input (see the NaN-encoder incident)"
                )
            out["w_window"] = window.astype(np.float16)  # [K, 512]
            return out

    factory_module.ShardedSingleStepDataset = WSMDeltaNetShardedSingleStepDataset
    print(
        f"[wsm-dn] dataset patched: feats_root={feats_root} window_len={window_len} "
        f"tasks={len(task_names)} (input=w_window)",
        flush=True,
    )


def attach_wsm_deltanet(
    model,
    w_dim: int = 512,
    cond_dim: int | None = None,
    window_len: int = 8,
    num_heads: int = 2,
    head_dim: int = 256,
    gate_init: float = 1e-3,
    history_dropout: float = 0.0,
    log_every: int = 50,
):
    """Reclass model.action_head, patch the DiT timestep encoder, attach the conditioner.

    Returns the conditioner (the only net-new trainable module; projector + DiT train as in baseline
    finetune).
    """
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    from workspace_models.networks.wsm_gated_deltanet import WSMGatedDeltaNetConditioner

    dit = model.action_head.model
    inner_dim = int(dit.proj_out_2.in_features)  # 1536 on the released N1.7 checkpoint == temb width
    cond_dim = int(cond_dim or inner_dim)

    te = dit.timestep_encoder
    if not getattr(te, "_wsm_patched", False):
        _orig_te_forward = te.forward

        def _patched_te_forward(timestep, _orig=_orig_te_forward, _mod=te):
            temb = _orig(timestep)  # [B, inner_dim]
            cond = getattr(_mod, "_wsm_cond", None)
            return temb if cond is None else temb + cond.to(device=temb.device, dtype=temb.dtype)

        te.forward = _patched_te_forward
        te._wsm_cond = None
        te._wsm_patched = True

    def _cond_from_window(head, w_window):
        """[B, K, 512] -> [B, cond_dim], with the finite guard on both ends."""
        if w_window is None:
            return None
        x = w_window.to(dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "[wsm-dn] non-finite omega window reached the conditioner — hard-fail rather than "
                "poison temb with NaN (see [[groot-eval2-nan-encoder-bug]])"
            )
        cond = head.wsm_deltanet(x)
        if not torch.isfinite(cond).all():
            raise RuntimeError(
                "[wsm-dn] conditioner produced a non-finite vector from a finite window — the "
                "conditioner weights are corrupt; refusing to condition temb"
            )
        return cond

    class WSMDeltaNetActionHead(Gr00tN1d7ActionHead):
        def forward(self, backbone_output, action_input):
            assert getattr(self.config, "expand_batch", None) in (None, 1), (
                "[wsm-dn] expand_batch unsupported (the conditioner batch would mismatch the expanded temb)"
            )
            w_window = action_input["w_window"] if "w_window" in action_input else None
            # FAIL-LOUD on step 0: if the collator dropped 'w_window' the conditioner is a silent
            # no-op and this trains as plain baseline — the "deploys as baseline" failure class.
            if self.training and not self._dn_checked:
                assert w_window is not None, (
                    "[wsm-dn] NO 'w_window' in action_input on the first train batch — the GR00T "
                    "collator did not pass the dataset's 'w_window' key through. The conditioner "
                    "would be a SILENT no-op (= baseline). Verify install_wsm_deltanet_dataset ran."
                )
                self._dn_checked = True
            cond = _cond_from_window(self, w_window)
            self.model.timestep_encoder._wsm_cond = cond
            try:
                out = super().forward(backbone_output, action_input)
            finally:
                self.model.timestep_encoder._wsm_cond = None
            if self.training and log_every and (self._dn_step % log_every == 0) and cond is not None:
                with torch.no_grad():
                    gate = torch.tanh(self.wsm_deltanet.alpha)
                    print(
                        f"[wsm-dn] step {self._dn_step} flow {float(out['loss'].detach()):.4f} "
                        f"| cond |mu| {float(cond.abs().mean()):.5f} sd {float(cond.std()):.5f} "
                        f"| tanh(alpha) |mu| {float(gate.abs().mean()):.5f} "
                        f"| K={self.wsm_deltanet.window_len} ACTIVE",
                        flush=True,
                    )
            self._dn_step += 1
            return out

        @torch.no_grad()
        def get_action_with_features(
            self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None
        ):
            """SERVE. Single-pass conditioning: hold `temb += cond` across every Euler step.

            Unlike the CFG arm this does NOT reimplement the sampler — the conditioner is one
            additive vector with no per-step branching, so setting the stash around the vendored
            call is both sufficient and far less fragile.
            """
            w_window = action_input["w_window"] if "w_window" in action_input else None
            if w_window is None:
                w_window = getattr(self, "_dn_eval_window", None)
            if w_window is None:
                raise RuntimeError(
                    "[wsm-dn] serve path has no 'w_window' (neither in action_input nor the "
                    "_dn_eval_window stash). Serving without the conditioner would silently be the "
                    "baseline policy under this checkpoint's name."
                )
            self.model.timestep_encoder._wsm_cond = _cond_from_window(self, w_window)
            try:
                return super().get_action_with_features(
                    backbone_features, state_features, embodiment_id, backbone_output, action_input, options
                )
            finally:
                self.model.timestep_encoder._wsm_cond = None

    ah = model.action_head
    ah.__class__ = WSMDeltaNetActionHead
    ah._dn_step = 0
    ah._dn_checked = False
    ah._dn_eval_window = None

    ref = next((p for p in ah.parameters() if p.requires_grad), None)
    dev = ref.device if ref is not None else next(ah.parameters()).device
    dt = ref.dtype if ref is not None else next(ah.parameters()).dtype
    ah.wsm_deltanet = WSMGatedDeltaNetConditioner(
        w_dim=w_dim,
        cond_dim=cond_dim,
        window_len=window_len,
        num_heads=num_heads,
        head_dim=head_dim,
        gate_init=gate_init,
        history_dropout=history_dropout,
    ).to(device=dev, dtype=dt)
    ah.wsm_deltanet.requires_grad_(True)
    print(
        f"[wsm-dn] attached: cond_dim={cond_dim} w_dim={w_dim} window_len={window_len} "
        f"heads={num_heads}x{head_dim} gate_init={gate_init} history_dropout={history_dropout} "
        f"dtype={dt} params={sum(p.numel() for p in ah.wsm_deltanet.parameters()) / 1e6:.2f}M",
        flush=True,
    )
    return ah.wsm_deltanet


def install_wsm_deltanet_action_head(
    w_dim: int = 512,
    cond_dim: int | None = None,
    window_len: int = 8,
    num_heads: int = 2,
    head_dim: int = 256,
    gate_init: float = 1e-3,
    history_dropout: float = 0.0,
) -> None:
    """TRAIN path: monkeypatch Gr00tN1d7Pipeline._create_model to attach the conditioner post-load."""
    import gr00t.model.gr00t_n1d7.setup as setup_module

    _orig_create = setup_module.Gr00tN1d7Pipeline._create_model

    def _patched_create(self):
        model = _orig_create(self)  # HF strict-load happens here
        attach_wsm_deltanet(
            model,
            w_dim=w_dim,
            cond_dim=cond_dim,
            window_len=window_len,
            num_heads=num_heads,
            head_dim=head_dim,
            gate_init=gate_init,
            history_dropout=history_dropout,
        )
        return model

    setup_module.Gr00tN1d7Pipeline._create_model = _patched_create
    print(
        f"[wsm-dn] action head patched (train path): gated delta-rule read at temb (window_len={window_len})",
        flush=True,
    )
