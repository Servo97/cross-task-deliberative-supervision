"""GR00T demo-CFG wiring (WSMv2, doc 15) — composes ON TOP of the proven _groot_wsm_cfg_common machinery.

Train-time pipeline per sample:
  dataset patch  -> w_t [512] (slot1, unchanged from the CFG POC) + the fusion inputs:
                    hist_win [K,512] (causal frozen-w window), demo_win [Wn,512] (d.npz tokens of a
                    MATCHED same-task partner demo, proportional tau + jitter), demo_off [Wn] int8,
                    demo_mask [Wn] bool
  head subclass  -> z = FROZEN HistoryDemoFusion(hist_win, demo_win, ...) computed ONLINE in the action
                    head (fp32, no_grad), then action_input["w_next"] = z and the PROVEN WSMCfgActionHead
                    forward runs unchanged: conditioner(w_t, z) -> DiT temb, per-component CFG dropout,
                    zero-init => step 0 == baseline. Eval two-pass/guidance/RTC all inherited.

Partner selection mirrors the WSMv2 encoder phase EXACTLY (matched mode: 4 nearest same-task demos by
initial-scene latent cosine, seed-0, registry demos excluded from the partner role) — train/post-train
pairing consistency is load-bearing (the mixed-partner run showed inconsistency kills the demo channel).

Provenance (D19): attach stamps {wsm2 ckpt id, registry sha, d.npz meta} onto the head; the serve asserts
all three against its own artifacts. gr00t/torch imports live inside functions (venv-only idiom).
"""

from __future__ import annotations

import json
from pathlib import Path

DEMO_KEYS = ("hist_win", "demo_win", "demo_off", "demo_mask", "wsm_task_lang")


def build_partner_manifest(
    feats_root: Path,
    demo_tokens_root: Path,
    tasks: set[str],
    registry: dict[str, list[int]],
    n_partners: int = 4,
    top: int = 8,
) -> dict:
    """{task: {ep: [partner_eps]}} — matched partners by init-scene w-cosine (seed-0), registry excluded."""
    import numpy as np

    manifest: dict[str, dict[int, list[int]]] = {}
    rng = np.random.default_rng(0)
    for task in sorted(tasks):
        eps, fps = [], []
        for d in sorted((feats_root / task).glob("demo_*")):
            ep = int(d.name.split("_")[1])
            if not (demo_tokens_root / task / d.name / "d.npz").exists():
                continue
            w = np.load(d / "w.npz")["w"]
            eps.append(ep)
            fps.append(w[:3].astype(np.float32).mean(0))
        reg = set(registry.get(task, []))
        cand_rows = [i for i, e in enumerate(eps) if e not in reg]  # partners never come from the registry
        fp = np.stack(fps)
        fp /= np.linalg.norm(fp, axis=1, keepdims=True) + 1e-8
        m: dict[int, list[int]] = {}
        for i, ep in enumerate(eps):  # every demo1 (incl registry) gets partners
            sims = fp[cand_rows] @ fp[i]
            order = np.argsort(-sims)
            near = [cand_rows[j] for j in order if cand_rows[j] != i][:top]
            pick = rng.choice(len(near), size=min(n_partners, len(near)), replace=False)
            m[ep] = [eps[near[j]] for j in pick]
        manifest[task] = m
    return manifest


def install_wsm_demo_cfg_dataset(
    feats_root: str,
    demo_tokens_root: str,
    registry_path: str,
    demo_fraction: float = 0.30,
    k_hist: int = 16,
    window: int = 20,
    jitter: int = 5,
) -> None:
    """Patch the GR00T dataset to emit w_t + the fusion inputs per sample (mirrors install_wsm_cfg_dataset)."""
    import gr00t.data.dataset.factory as factory_module
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    from utils.soup import combined_target_soup
    from workspace_models.features.wsm_align import (
        causal_window_indices,
        demo_window_at,
        load_w,
        proportional_tau,
    )

    feats_root = Path(feats_root).expanduser()
    dtok_root = Path(demo_tokens_root).expanduser()
    registry = json.loads(Path(registry_path).expanduser().read_text())
    task_by_dir = {str(Path(m["path"]).resolve()): m["task"] for m in combined_target_soup(demo_fraction)}
    task_names = set(task_by_dir.values())
    manifest = build_partner_manifest(feats_root, dtok_root, task_names, registry)
    # task-MEAN lang table (S8 serve parity — the serve feeds the same table; per-demo lang is never used)
    task_lang = {}
    for task in task_names:
        langs = []
        for d in sorted((feats_root / task).glob("demo_*"))[:40]:
            z = np.load(d / "w.npz")
            if "lang_global" in z.files:
                langs.append(z["lang_global"].astype(np.float32))
        task_lang[task] = np.mean(langs, axis=0) if langs else np.zeros(2048, dtype=np.float32)
    n_pairs = sum(len(v) for v in manifest.values())
    print(
        f"[wsm-demo-cfg] partner manifest: {len(manifest)} tasks, {n_pairs} demo1 entries (matched, "
        f"registry-excluded partners)",
        flush=True,
    )

    class WSMDemoCfgShardedSingleStepDataset(ShardedSingleStepDataset):
        def _wsm_task(self) -> str:
            key = str(Path(self.dataset_path).resolve())
            if key in task_by_dir:
                return task_by_dir[key]
            for t in task_names:
                if t in set(Path(self.dataset_path).parts):
                    return t
            raise KeyError(f"[wsm-demo-cfg] cannot resolve task for {self.dataset_path}")

        def _wsm_load(self, ep_idx: int):
            if getattr(self, "_wsm_key", None) == ep_idx:
                return self._wsm_val
            ep = int(self.episode_loader.episodes_metadata[ep_idx]["episode_index"])
            task = self._wsm_task()
            w, fi = load_w(feats_root / task / f"demo_{ep:06d}")
            partners = manifest[task].get(ep)
            if not partners:
                raise KeyError(f"[wsm-demo-cfg] no partners for {task}/demo_{ep:06d}")
            self._wsm_key, self._wsm_val = ep_idx, (task, ep, w, fi, partners)
            return self._wsm_val

        def _demo_tokens(self, task: str, ep: int):
            key = (task, ep)
            cache = getattr(self, "_dtok_cache", None)
            if cache is None:
                cache = self._dtok_cache = {}
            if key not in cache:
                if len(cache) > 8:
                    cache.pop(next(iter(cache)))
                z = np.load(dtok_root / task / f"demo_{ep:06d}" / "d.npz")
                cache[key] = z["d"]
            return cache[key]

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
            task, ep, w, fi, partners = self._wsm_load(self._wsm_cur_ep_idx)
            t = int(step_index)
            hidx = causal_window_indices(fi, t, k_hist)
            out["w_t"] = np.asarray(w[hidx[-1]], dtype=np.float16)  # slot1 (unchanged POC path)
            out["hist_win"] = np.asarray(w[hidx], dtype=np.float16)  # [K,512]
            rng = getattr(self, "_wsm_rng", None) or np.random.default_rng()
            self._wsm_rng = rng
            p_ep = partners[int(rng.integers(0, len(partners)))]
            d = self._demo_tokens(task, p_ep)  # [M,512] fp16
            g = int(np.nonzero(np.asarray(fi) <= t)[0][-1]) if (np.asarray(fi) <= t).any() else 0
            tau = proportional_tau(g, len(fi), len(d), jitter=jitter, rng=rng)
            idx, off, mask = demo_window_at(len(d), tau, window)
            out["demo_win"] = np.asarray(d[idx], dtype=np.float16)  # [2W+1,512]
            out["demo_off"] = off.astype(np.int8)
            out["demo_mask"] = mask
            out["wsm_task_lang"] = np.asarray(task_lang[task], dtype=np.float16)  # [2048] task-mean (S8)
            return out

    factory_module.ShardedSingleStepDataset = WSMDemoCfgShardedSingleStepDataset
    print(
        f"[wsm-demo-cfg] dataset patched: K={k_hist} W={window} jitter={jitter} feats={feats_root} dtok={dtok_root}",
        flush=True,
    )


def attach_wsm_demo_cfg(
    model, fusion_ckpt: str, w_dim: int = 512, p_drop: float = 0.2, diag_every: int = 100, guidance_scale: float = 0.0
):
    """attach_wsm_cfg(with_future=True) + a head subclass that computes w_next = frozen-fusion z online."""
    import torch

    from vla_training.train.train_base._groot_wsm_cfg_common import attach_wsm_cfg
    from workspace_models.networks.demo_fusion import HistoryDemoFusion

    conditioner = attach_wsm_cfg(
        model, w_dim=w_dim, p_drop=p_drop, with_future=True, diag_every=diag_every, guidance_scale=guidance_scale
    )
    ah = model.action_head
    ck = torch.load(Path(fusion_ckpt).expanduser(), map_location="cpu", weights_only=False)
    fusion = HistoryDemoFusion()
    fusion.load_state_dict(ck["fusion"])
    fusion = fusion.to(device=next(ah.parameters()).device, dtype=torch.float32).eval()  # FROZEN (D17);
    # the Trainer re-moves the whole head (incl. this submodule) later — _fusion_z resolves device at call time
    fusion.requires_grad_(False)

    base_cls = ah.__class__  # WSMCfgActionHead (dynamic)

    class WSMDemoCfgActionHead(base_cls):
        def _fusion_z(self, action_input):
            missing = [k for k in DEMO_KEYS if k not in action_input]
            if missing:
                raise AssertionError(
                    f"[wsm-demo-cfg] fusion keys missing from action_input: {missing} — "
                    f"did install_wsm_demo_cfg_dataset run / collator stack numpy keys?"
                )
            # Resolve device/dtype from the fusion's OWN params AT CALL TIME — attach runs inside
            # _create_model while the model is still on CPU; the Trainer moves everything to CUDA after.
            # A device captured at attach is stale (the exact cuda-vs-cpu crash of the first demo launch).
            fp = next(self.wsm_fusion.parameters())
            dv, dt = fp.device, fp.dtype
            with torch.no_grad():
                o = self.wsm_fusion(
                    action_input["hist_win"].to(dv, dt),
                    action_input["demo_win"].to(dv, dt),
                    action_input["demo_off"].to(dv).long(),
                    action_input["demo_mask"].to(dv).bool(),
                    action_input["wsm_task_lang"].to(dv, dt),
                )
            return o["z"]

        def forward(self, backbone_output, action_input):
            # Mutate IN PLACE: action_input is a transformers BatchFeature (dict subclass with ATTRIBUTE
            # access — the parent reads action_input.embodiment_id). Wrapping it in dict() strips attribute
            # access and crashes the parent (launch-2 failure). Item assignment preserves the type.
            action_input["w_next"] = self._fusion_z(action_input).to(next(self.wsm_cfg.parameters()).dtype)
            return super().forward(backbone_output, action_input)

    ah.__class__ = WSMDemoCfgActionHead
    ah.wsm_fusion = fusion  # net-new module: safe to register
    ah._wsm2_meta = {
        "fusion_ckpt": str(fusion_ckpt),
        "registry_sha": ck.get("registry_sha", "?"),
        "wsm2_step": int(ck.get("step", -1)),
    }
    n = sum(p.numel() for p in fusion.parameters())
    print(f"[wsm-demo-cfg] attached: frozen fusion {n / 1e6:.1f}M (z -> w_next slot) meta={ah._wsm2_meta}", flush=True)
    return conditioner


def install_wsm_demo_cfg_action_head(
    fusion_ckpt: str, w_dim: int = 512, p_drop: float = 0.2, diag_every: int = 100
) -> None:
    """TRAIN path: monkeypatch Gr00tN1d7Pipeline._create_model to attach demo-CFG post-load."""
    import gr00t.model.gr00t_n1d7.setup as setup_module

    _orig_create = setup_module.Gr00tN1d7Pipeline._create_model

    def _patched_create(self):
        model = _orig_create(self)
        attach_wsm_demo_cfg(model, fusion_ckpt, w_dim=w_dim, p_drop=p_drop, diag_every=diag_every)
        return model

    setup_module.Gr00tN1d7Pipeline._create_model = _patched_create
    print("[wsm-demo-cfg] action head patched (train path): frozen fusion -> w_next -> AdaLN-CFG on temb", flush=True)
