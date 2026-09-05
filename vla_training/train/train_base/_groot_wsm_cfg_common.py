"""CFG (classifier-free-guidance) workspace conditioning for GR00T N1.7 — the model-free POC (doc 12).

Conditions the action DENOISER on the workspace latent via AdaLN: a learned vector derived from
[w_t, w_{t+1}] is ADDED to the DiT's timestep embedding `temb` (the single source of all the DiT's
AdaLN scale/shifts). The VLM backbone + its perception tokens (`vl_embs`) are NEVER touched — unlike the
failed injection, which modulated the backbone image tokens. Trained with per-component CFG dropout so the
model learns both a conditional and an unconditional policy; at eval the serve extrapolates the velocity.

Three least-invasive hooks (subclass + monkeypatch; NO edits to the vendored Isaac-GR00T repo):

  install_wsm_cfg_dataset(feats_root):
     attach `w_t [w_dim] fp16` (the newest causal workspace latent at the sampled step) per sample. The
     `w_next` slot is left unattached for the POC (-> the conditioner uses its learned null); demo-ICL
     later attaches it too. GR00T's collator auto-stacks the numpy key into action_input.w_t.

  install_wsm_cfg_action_head(...):
     _create_model -> reclass action_head to WSMCfgActionHead, attach a WSMCfgConditioner, and patch the
     DiT's timestep_encoder.forward IN PLACE so temb += cfg_cond (param names preserved -> ckpt round-trips).
     forward() applies per-component dropout (training) + a periodic conditional-vs-unconditional flow-loss
     diagnostic (is w informative?). get_action_with_features() runs the CFG two-pass denoising
     (v = v_uncond + s*(v_cond - v_uncond)).

Grounded against the DEPLOYED pinned NVIDIA Isaac-GR00T (NOT robocasa_groot): the action head is
gr00t.model.gr00t_n1d7.Gr00tN1d7ActionHead (get_action -> get_action_with_features; NO future_tokens;
sa_embs = cat(state, action); action_decoder = CategorySpecificMLP, so CFG must combine at the velocity);
the DiT is AlternateVLDiT (temb from timestep_encoder drives all AdaLN + proj_out; proj_out_2 -> inner_dim).
GR00T-venv-only. Reuses assert_wt_coverage from the injection module (same coverage contract).
"""

from __future__ import annotations

from vla_training.train.train_base._groot_wsm_common import assert_wt_coverage  # re-export (same contract)

__all__ = ["install_wsm_cfg_dataset", "install_wsm_cfg_action_head", "attach_wsm_cfg", "assert_wt_coverage"]


def cfg_velocities(vel_fn, te, sa, ts, cond, uncond, s: float, mode: str = "batched"):
    """One guided velocity: v = v_u + s*(v_c - v_u). PERF (perf-first mandate): 'batched' runs cond+uncond
    as ONE forward with the two halves stacked on the batch dim (the DiT is batch-equivariant: attention/
    LN/temb all per-sample), ~2x over 'seq' on the denoiser. vel_fn(sa, ts) must accept any batch size
    (tile its captured encoder state to sa.shape[0]). te._wsm_cond is consumed by the patched
    timestep_encoder ([B,dim] rows align with the stacked batch). Exactness vs 'seq' is unit-tested AND
    self-verified in production via WSM_VERIFY_2PASS=1 (first Euler step, logged)."""
    import torch

    b = sa.shape[0]
    if mode == "seq":
        te._wsm_cond = cond
        pv_c = vel_fn(sa, ts)
        te._wsm_cond = uncond
        pv_u = vel_fn(sa, ts)
    else:
        te._wsm_cond = torch.cat([cond.expand(b, -1), uncond.expand(b, -1)], dim=0)
        pv = vel_fn(torch.cat([sa, sa], dim=0), torch.cat([ts, ts], dim=0))
        pv_c, pv_u = pv[:b], pv[b:]
    te._wsm_cond = None
    return pv_u + s * (pv_c - pv_u)


def install_wsm_cfg_dataset(feats_root: str, demo_fraction: float = 0.30, with_future: bool = False) -> None:
    """Attach the newest causal workspace latent w_t (and optionally w_next for ICL) per sample."""
    from pathlib import Path

    import gr00t.data.dataset.factory as factory_module
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    from utils.soup import combined_target_soup
    from workspace_models.features.wsm_align import causal_window_indices, load_w, next_at

    feats_root = Path(feats_root).expanduser()
    task_by_dir = {str(Path(m["path"]).resolve()): m["task"] for m in combined_target_soup(demo_fraction)}
    task_names = set(task_by_dir.values())

    class WSMCfgShardedSingleStepDataset(ShardedSingleStepDataset):
        def _wsm_task(self) -> str:
            key = str(Path(self.dataset_path).resolve())
            if key in task_by_dir:
                return task_by_dir[key]
            for t in task_names:
                if t in set(Path(self.dataset_path).parts):
                    return t
            raise KeyError(f"[wsm-cfg] cannot resolve task for dataset_path={self.dataset_path}")

        def _wsm_load(self, ep_idx: int):
            if getattr(self, "_wsm_key", None) == ep_idx:
                return self._wsm_val
            ep_index = int(self.episode_loader.episodes_metadata[ep_idx]["episode_index"])
            fdir = feats_root / self._wsm_task() / f"demo_{ep_index:06d}"
            if not (fdir / "w.npz").exists():
                raise FileNotFoundError(f"[wsm-cfg] policy features missing: {fdir}/w.npz")
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
            newest = int(causal_window_indices(fi, int(step_index), 1)[-1])  # newest grid frame <= t
            out["w_t"] = np.asarray(w[newest], dtype=np.float16)  # [w_dim]
            if with_future:
                out["w_next"] = np.asarray(next_at(w, fi, int(step_index)), dtype=np.float16)  # [w_dim]
            return out

    factory_module.ShardedSingleStepDataset = WSMCfgShardedSingleStepDataset
    print(
        f"[wsm-cfg] dataset patched: feats_root={feats_root} tasks={len(task_names)} "
        f"with_future={with_future} (target=w_t)",
        flush=True,
    )


def attach_wsm_cfg(
    model,
    w_dim: int = 512,
    cond_dim: int | None = None,
    p_drop: float = 0.2,
    with_future: bool = False,
    diag_every: int = 100,
    guidance_scale: float = 0.0,
):
    """Reclass model.action_head -> WSMCfgActionHead, wrap the DiT timestep_encoder, attach the conditioner.
    Returns the conditioner (the only net-new trainable module; projector+DiT train as in baseline FT)."""
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    from workspace_models.networks.wsm_cfg_cond import WSMCfgConditioner

    dit = model.action_head.model
    inner_dim = int(dit.proj_out_2.in_features)  # == temb dim (cond_dim default)
    cond_dim = cond_dim or inner_dim

    # Patch timestep_encoder.forward IN PLACE (don't wrap in a submodule — that would rename its params to
    # timestep_encoder.orig.* in the saved ckpt and break from_pretrained at serve, since the DiT/timestep
    # encoder is trained under tune_diffusion_model). temb += _wsm_cond (a plain attr, not saved).
    te = dit.timestep_encoder
    if not getattr(te, "_wsm_patched", False):
        _orig_te_forward = te.forward  # bound method of the original encoder

        def _patched_te_forward(timestep, _orig=_orig_te_forward, _mod=te):
            temb = _orig(timestep)  # [B, inner_dim]
            cond = getattr(_mod, "_wsm_cond", None)
            return temb if cond is None else temb + cond.to(device=temb.device, dtype=temb.dtype)

        te.forward = _patched_te_forward
        te._wsm_cond = None
        te._wsm_patched = True

    class WSMCfgActionHead(Gr00tN1d7ActionHead):
        def _cfg_vec(self, w_t, w_next, *, force_uncond):
            if w_t is None and w_next is None:
                return None
            return self.wsm_cfg(w_t, w_next, training=self.training, force_uncond=force_uncond)

        def forward(self, backbone_output, action_input):
            assert getattr(self.config, "expand_batch", None) in (None, 1), (
                "[wsm-cfg] expand_batch unsupported (cfg_cond batch would mismatch the expanded temb)"
            )
            # TRAIN: w_t rides in action_input (the collator auto-stacks the dataset's numpy key).
            w_t = action_input["w_t"] if "w_t" in action_input else None
            w_next = action_input["w_next"] if "w_next" in action_input else None
            # FAIL-LOUD: if the collator dropped 'w_t', the conditioner is a silent no-op and the policy
            # trains as plain baseline (the "deploys as baseline" failure class). Catch it on step 0, not 60k.
            if self.training and not self._wt_checked:
                assert w_t is not None or w_next is not None, (
                    "[wsm-cfg] NO 'w_t' in action_input on the first train batch — the GR00T collator did not "
                    "pass the dataset's 'w_t' key through. The conditioner would be a SILENT no-op (= baseline). "
                    "Verify install_wsm_cfg_dataset ran and the collator stacks extra numpy keys."
                )
                self._wt_checked = True
            self.model.timestep_encoder._wsm_cond = self._cfg_vec(
                w_t, w_next, force_uncond=False
            )  # per-component dropout inside
            out = super().forward(backbone_output, action_input)
            self.model.timestep_encoder._wsm_cond = None
            if (
                self.training
                and self._diag_every
                and (self._step % self._diag_every == 0)
                and (w_t is not None or w_next is not None)
            ):
                self.model.timestep_encoder._wsm_cond = self._cfg_vec(w_t, w_next, force_uncond=True)
                with torch.no_grad():
                    uout = super().forward(
                        backbone_output, action_input
                    )  # NOTE: independent noise -> use running mean
                self.model.timestep_encoder._wsm_cond = None
                lc, lu = float(out["loss"].detach()), float(uout["loss"].detach())
                self._diag_c = 0.98 * self._diag_c + 0.02 * lc if self._diag_n else lc
                self._diag_u = 0.98 * self._diag_u + 0.02 * lu if self._diag_n else lu
                self._diag_n += 1
                gap = self._diag_u - self._diag_c
                print(
                    f"[wsm-cfg] step {self._step} flow {lc:.4f} | diag(EMA) L_cond {self._diag_c:.4f} "
                    f"L_uncond {self._diag_u:.4f} gap {gap:+.4f} "
                    f"({'w informative' if gap > 0 else 'w not yet informative'}) "
                    f"[EMA-smoothed; noisy — independent denoise noise per pass, trust the trend not a step]",
                    flush=True,
                )
            self._step += 1
            return out

        @torch.no_grad()
        def get_action_with_features(
            self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None
        ):
            """Faithful mirror of Isaac-GR00T Gr00tN1d7ActionHead.get_action_with_features (pinned SHA) with a
            CFG two-pass per Euler step, combined at the VELOCITY (after the nonlinear action_decoder):
            v = v_uncond + s*(v_cond - v_uncond). s==0 -> single (learned-uncond) pass. NO future_tokens
            (Isaac N1.7: sa_embs = cat(state, action)). Inherited get_action() calls this. RTC path preserved."""
            from transformers import BatchFeature

            s = float(getattr(self, "guidance_scale", 0.0))
            # EVAL: w_t/w_next come from a head stash set by the serve (the obs->action_input pipeline does NOT
            # carry extra obs keys through at inference — same reason the injection path stashed on the head).
            w_t, w_next = getattr(self, "_cfg_eval", (None, None))
            cond = self._cfg_vec(w_t, w_next, force_uncond=False)
            uncond = self._cfg_vec(w_t, w_next, force_uncond=True)

            vl_embeds = backbone_features
            batch_size, device = vl_embeds.shape[0], vl_embeds.device
            actions = torch.randn(
                (batch_size, self.config.action_horizon, self.action_dim), dtype=vl_embeds.dtype, device=device
            )
            dt = 1.0 / self.num_inference_timesteps
            vel_strength = torch.ones_like(actions)
            if "action" in action_input:  # RTC inpainting (preserve parent behavior)
                assert options is not None and {
                    "action_horizon",
                    "rtc_overlap_steps",
                    "rtc_frozen_steps",
                    "rtc_ramp_rate",
                } <= set(options), "RTC options missing"
                ahp = options["action_horizon"]
                actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                    :, ahp - options["rtc_overlap_steps"] : ahp, :
                ]
                vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
                inter = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
                tt = torch.linspace(0.0, 1.0, inter + 2, device=device)
                ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * tt)
                ramp = (ramp / ramp[-1].clamp_min(1e-8))[1:-1]
                vel_strength[:, options["rtc_frozen_steps"] : options["rtc_overlap_steps"], :] = ramp[
                    None, :, None
                ].to(device)

            def _tile(x, n):  # tile leading batch dim to n*B
                return None if x is None else x.repeat(n, *([1] * (x.dim() - 1)))

            def _vel(sa, ts):  # DiT -> decode -> action velocity
                n = sa.shape[0] // batch_size  # 1 (seq) or 2 (batched two-pass)
                vl = _tile(vl_embeds, n)
                eid = _tile(embodiment_id, n)
                if self.config.use_alternate_vl_dit:
                    mo = self.model(
                        hidden_states=sa,
                        encoder_hidden_states=vl,
                        timestep=ts,
                        image_mask=_tile(backbone_output.image_mask, n),
                        backbone_attention_mask=_tile(backbone_output.backbone_attention_mask, n),
                    )
                else:
                    mo = self.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts)
                return self.action_decoder(mo, eid)[:, -self.action_horizon :]

            # `uncond` is the LEARNED null projection (proper CFG unconditional = the in-distribution "w dropped"
            # case). s=0 -> learned unconditional (≈ baseline, NOT byte-identical). NOT _wsm_cond=None (OOD).
            for t in range(self.num_inference_timesteps):
                td = int((t / float(self.num_inference_timesteps)) * self.num_timestep_buckets)
                ts = torch.full((batch_size,), td, device=device)
                af = self.action_encoder(actions, ts, embodiment_id)
                if self.config.add_pos_embed:
                    af = af + self.position_embedding(torch.arange(af.shape[1], device=device)).unsqueeze(0)
                sa = torch.cat((state_features, af), dim=1)  # Isaac N1.7: NO future_tokens
                import os as _os

                te = self.model.timestep_encoder
                if s == 0.0 or cond is None:
                    te._wsm_cond = uncond
                    pv = _vel(sa, ts)
                    te._wsm_cond = None
                else:
                    mode = _os.environ.get("WSM_TWOPASS_MODE", "batched")
                    pv = cfg_velocities(_vel, te, sa, ts, cond, uncond, s, mode=mode)
                    if (
                        t == 0
                        and _os.environ.get("WSM_VERIFY_2PASS", "0") == "1"
                        and not getattr(self, "_wsm_2pass_ok", False)
                    ):
                        pv_seq = cfg_velocities(_vel, te, sa, ts, cond, uncond, s, mode="seq")
                        d = float((pv - pv_seq).abs().max())
                        # bf16 rounding puts max|diff| on the quantization grid of the LARGEST velocity
                        # element (~2^-8 relative), so a fixed abs threshold false-fails whenever max|v|
                        # is large — it killed every s>=1.5 sweep eval at d=0.0625. Judge relative error.
                        rel = d / max(float(pv_seq.abs().max()), 1e-6)
                        # guidance amplifies per-pass rounding by (|s| + |1-s|): v = (1-s)v_u + s*v_c.
                        # The flat rel<1e-2 bound still false-failed at s=4 (7x amplification) — scale it.
                        tol = 1e-2 * max(1.0, abs(s) + abs(1.0 - s))
                        print(
                            f"[wsm-cfg] 2PASS VERIFY max|batched-seq|={d:.3e} rel={rel:.3e} "
                            f"tol={tol:.3e} (mode={mode} s={s})",
                            flush=True,
                        )
                        assert d < 5e-2 or rel < tol, (
                            f"batched two-pass diverges from sequential (abs {d}, rel {rel}, tol {tol})"
                        )
                        self._wsm_2pass_ok = True  # verify once, not per action call
                actions = actions + dt * pv * vel_strength
            return BatchFeature(
                data={"action_pred": actions, "backbone_features": vl_embeds, "state_features": state_features}
            )

    ah = model.action_head
    ah.__class__ = WSMCfgActionHead
    # NB: do NOT store `te` as an action-head attribute (e.g. ah._te = te) — that re-registers the
    # timestep_encoder params under a 2nd name and safetensors refuses to save the "shared tensors".
    # The conditioning rides on dit.timestep_encoder._wsm_cond, reached via self.model.timestep_encoder.
    ah._step = 0
    ah._diag_every = int(diag_every)
    ah._diag_c = ah._diag_u = 0.0
    ah._diag_n = 0
    ah._wt_checked = False  # one-time fail-loud "did w_t reach action_input"
    ah._cfg_eval = (None, None)  # eval w_t/w_next stash (set by the serve)
    ah.guidance_scale = float(guidance_scale)
    ref = next((p for p in ah.parameters() if p.requires_grad), None)
    dev = ref.device if ref is not None else next(ah.parameters()).device
    dtp = ref.dtype if ref is not None else next(ah.parameters()).dtype
    ah.wsm_cfg = WSMCfgConditioner(w_dim=w_dim, cond_dim=cond_dim, p_drop=p_drop).to(device=dev, dtype=dtp)
    ah.wsm_cfg.requires_grad_(True)
    print(
        f"[wsm-cfg] attached: w_dim={w_dim} cond_dim={cond_dim} p_drop={p_drop} with_future={with_future} "
        f"diag_every={diag_every} dtype={dtp} params={sum(p.numel() for p in ah.wsm_cfg.parameters()) / 1e3:.1f}K "
        f"(VLM untouched; conditioning rides on the DiT temb)",
        flush=True,
    )
    return ah.wsm_cfg


def install_wsm_cfg_action_head(
    w_dim: int = 512, p_drop: float = 0.2, with_future: bool = False, diag_every: int = 100
) -> None:
    """TRAIN path: monkeypatch Gr00tN1d7Pipeline._create_model to attach CFG conditioning post-load."""
    import gr00t.model.gr00t_n1d7.setup as setup_module

    _orig_create = setup_module.Gr00tN1d7Pipeline._create_model

    def _patched_create(self):
        model = _orig_create(self)
        attach_wsm_cfg(model, w_dim=w_dim, p_drop=p_drop, with_future=with_future, diag_every=diag_every)
        return model

    setup_module.Gr00tN1d7Pipeline._create_model = _patched_create
    print(
        f"[wsm-cfg] action head patched (train path): AdaLN-CFG on the DiT temb "
        f"(p_drop={p_drop}, with_future={with_future})",
        flush=True,
    )
