"""WSM-conditioned GR00T N1.7 wiring (the canary): inject the precomputed workspace latent w_t into
the GR00T action head via a zero-init TokenModulator — WITHOUT editing the vendored Isaac-GR00T repo.

Three least-invasive hooks (subclass + monkeypatch, the same idiom as _groot_common.install_*):

  install_wsm_dataset(feats_root, k_window):
     factory.ShardedSingleStepDataset -> WSMShardedSingleStepDataset, whose get_datapoint appends
     `w_window [K,512] fp16` (causal last-K grid latents at the sampled native step) + `lang_global
     [2048] fp16` to the processor dict. The GR00T collator auto-stacks any extra numpy key, so both
     flow untouched into action_input.{w_window,lang_global} (no processor/collator edit).

  install_wsm_action_head(w_dim, k_window, lang_dim):
     Gr00tN1d7Pipeline._create_model -> reclass model.action_head to WSMConditionedActionHead and
     attach a trainable TokenModulator AFTER the HF strict-load (so no missing/unexpected keys). The
     head modulates the post-vlln backbone tokens (vl_embeds) over the image-token positions, in BOTH
     train (forward) and inference (get_action) paths, via the shared process_backbone_output seam.
     Zero-init => identical to the Eval1 baseline at step 0; the modulator is the only net-new module
     (swept into Stage0Trainer's base-LR group automatically since requires_grad=True).

GR00T-venv-only: every gr00t/torch import is inside a function, so this module stays import-safe
elsewhere. See internal_planning_and_todos (WSM-conditioned canary integration spec).
"""

from __future__ import annotations


def install_wsm_dataset(feats_root: str, k_window: int = 2, demo_fraction: float = 0.30) -> None:
    """Monkeypatch the single-step dataset to attach the causal w-window + lang per sample."""
    from pathlib import Path

    import gr00t.data.dataset.factory as factory_module
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    from utils.soup import combined_target_soup
    from workspace_models.features.wsm_align import load_w_and_lang, window_at

    feats_root = Path(feats_root).expanduser()
    # authoritative lerobot-dir -> task map (same soup the dataloader is built from)
    task_by_dir = {str(Path(m["path"]).resolve()): m["task"] for m in combined_target_soup(demo_fraction)}
    task_names = set(task_by_dir.values())

    class WSMShardedSingleStepDataset(ShardedSingleStepDataset):
        def _wsm_task(self) -> str:
            key = str(Path(self.dataset_path).resolve())
            if key in task_by_dir:
                return task_by_dir[key]
            parts = set(Path(self.dataset_path).parts)  # fallback: task as a path component
            for t in task_names:
                if t in parts:
                    return t
            raise KeyError(f"[wsm] cannot resolve task for dataset_path={self.dataset_path}")

        def _wsm_load(self, ep_idx: int):
            if getattr(self, "_wsm_key", None) == ep_idx:  # per-episode cache (1 load / shard episode)
                return self._wsm_val
            ep_index = int(self.episode_loader.episodes_metadata[ep_idx]["episode_index"])
            fdir = feats_root / self._wsm_task() / f"demo_{ep_index:06d}"
            if not (fdir / "w.npz").exists():
                raise FileNotFoundError(f"[wsm] policy features missing for sampled demo: {fdir}/w.npz")
            w, fi, lang = load_w_and_lang(fdir)
            assert lang is not None, f"[wsm] w.npz has no lang_global: {fdir} (re-run generate_policy_features)"
            self._wsm_key, self._wsm_val = ep_idx, (w, fi, lang)
            return self._wsm_val

        def get_shard(self, idx: int) -> list:  # mirror base + stash ep_idx for get_datapoint
            datapoints = []
            for ep_idx, step_indices in self.sharded_episodes[idx]:
                episode_data = self.episode_loader[ep_idx]
                self._wsm_cur_ep_idx = ep_idx
                for step_index in step_indices:
                    datapoints.append(self.get_datapoint(episode_data, step_index))
            return datapoints

        def get_datapoint(self, episode_data, step_index: int) -> dict:
            out = super().get_datapoint(episode_data, step_index)  # the processor's flat dict
            w, fi, lang = self._wsm_load(self._wsm_cur_ep_idx)
            out["w_window"] = np.asarray(window_at(w, fi, int(step_index), k_window), dtype=np.float16)  # [K,512]
            out["lang_global"] = np.asarray(lang, dtype=np.float16)  # [2048]
            return out

    factory_module.ShardedSingleStepDataset = WSMShardedSingleStepDataset
    print(f"[wsm] dataset patched: feats_root={feats_root} k_window={k_window} tasks={len(task_names)}", flush=True)


def assert_wt_coverage(feats_root: str, demo_fraction: float = 0.30, num_demos: int = 150, seed: int = 0) -> None:
    """FAIL-FAST coverage contract: every demo the policy will sample (the seed-`seed` first-`num_demos`
    keep-set of EACH task) MUST have a precomputed w.npz. With the cache complete (7500/7500) this holds,
    so we train on the FULL keep-set via the normal episode subsample — NOT a w_t-filtered subset (the old
    install_wsm_episode_filter biased exactly the composite splits whose demos had failed preprocessing).
    Raises with a per-task gap summary at STARTUP rather than dying mid-dataloading or skewing the mix."""
    from pathlib import Path

    from utils.soup import combined_target_soup
    from utils.subsample import episode_index_keep_set

    feats_root = Path(feats_root).expanduser()
    soup = combined_target_soup(demo_fraction)
    gaps, checked = {}, 0
    for m in soup:
        task, path = m["task"], m["path"]
        keep = episode_index_keep_set(path, num_demos, seed)
        if keep is None:
            continue
        avail = {int(p.parent.name.split("_")[1]) for p in (feats_root / task).glob("demo_*/w.npz")}
        missing = sorted(set(keep) - avail)
        checked += len(keep)
        if missing:
            gaps[task] = (len(missing), len(keep), missing[:5])
    if gaps:
        lines = [f"  {t}: {n}/{k} MISSING w.npz (e.g. demos {ex})" for t, (n, k, ex) in sorted(gaps.items())]
        raise RuntimeError(
            f"[wsm] w_t coverage INCOMPLETE under {feats_root} for the seed-{seed} first-{num_demos} keep-set "
            f"({len(gaps)} tasks short):\n"
            + "\n".join(lines)
            + "\n  -> re-run generate_policy_features for these, or the policy would sample demos without w_t."
        )
    print(
        f"[wsm] w_t coverage OK: all {checked} demos in the seed-{seed} first-{num_demos} keep-set "
        f"({len(soup)} task-dirs) have w.npz under {feats_root}",
        flush=True,
    )


def attach_wsm_modulator(model, w_dim: int = 512, k_window: int = 2, lang_dim: int = 2048):
    """Reclass model.action_head -> WSMConditionedActionHead and attach a zero-init TokenModulator.
    Used in BOTH paths: at TRAIN via install_wsm_action_head's _create_model patch, and at SERVE applied
    DIRECTLY to the loaded policy.model — because Gr00tPolicy loads via from_pretrained, bypassing
    Gr00tN1d7Pipeline._create_model, so the patch never fires there. Returns the modulator."""
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    from workspace_models.networks.token_modulator import TokenModulator

    class WSMConditionedActionHead(Gr00tN1d7ActionHead):
        def _wsm_stash(self, action_input) -> None:
            self._wsm_cond = (
                action_input["w_window"] if "w_window" in action_input else None,
                action_input["lang_global"] if "lang_global" in action_input else None,
            )

        def forward(self, backbone_output, action_input):
            self._wsm_stash(action_input)
            return super().forward(backbone_output, action_input)

        def get_action(self, backbone_output, action_input, options=None):
            self._wsm_stash(action_input)
            return super().get_action(backbone_output, action_input, options)

        def process_backbone_output(self, backbone_output):
            out = super().process_backbone_output(backbone_output)  # vlln + vl_self_attention
            cond = getattr(self, "_wsm_cond", None)
            if cond is not None and cond[0] is not None:
                w_window, lang = cond
                feats = out["backbone_features"]  # [B, seq, 2048] (post-vlln)
                mod = self.modulator(feats, w_window, lang)  # zero-init => mod == feats at step 0
                img_mask = getattr(backbone_output, "image_mask", None)
                if img_mask is not None:  # rewrite only the 192 vision patch tokens
                    out["backbone_features"] = torch.where(img_mask.unsqueeze(-1).to(torch.bool), mod, feats)
                else:  # fallback: modulate all tokens (identity at init either way)
                    out["backbone_features"] = mod
            return out

    ah = model.action_head
    ah.__class__ = WSMConditionedActionHead  # reclass: adds the 4 overrides
    ah._wsm_cond = None
    token_dim = int(model.config.backbone_embedding_dim)
    ref = next((p for p in ah.parameters() if p.requires_grad), None)
    dev = ref.device if ref is not None else next(ah.parameters()).device
    dt = ref.dtype if ref is not None else next(ah.parameters()).dtype
    ah.modulator = TokenModulator(w_dim=w_dim, lang_dim=lang_dim, token_dim=token_dim, k_window=k_window).to(
        device=dev, dtype=dt
    )
    ah.modulator.requires_grad_(True)  # only net-new trainable module
    print(
        f"[wsm] modulator attached: w_dim={w_dim} k={k_window} token_dim={token_dim} "
        f"dtype={dt} params={sum(p.numel() for p in ah.modulator.parameters()) / 1e6:.2f}M",
        flush=True,
    )
    return ah.modulator


def install_wsm_action_head(w_dim: int = 512, k_window: int = 2, lang_dim: int = 2048) -> None:
    """TRAIN path: monkeypatch Gr00tN1d7Pipeline._create_model to attach the modulator post-load."""
    import gr00t.model.gr00t_n1d7.setup as setup_module

    _orig_create = setup_module.Gr00tN1d7Pipeline._create_model

    def _patched_create(self):
        model = _orig_create(self)  # HF strict-load happens here
        attach_wsm_modulator(model, w_dim=w_dim, k_window=k_window, lang_dim=lang_dim)
        return model

    setup_module.Gr00tN1d7Pipeline._create_model = _patched_create
    print(f"[wsm] action head patched (train path): zero-init modulator (k_window={k_window})", flush=True)
