"""ONLINE demo-conditioning for WSMv2 demo-CFG GR00T eval (doc 15 D19–D20).

Per rollout: reset(task) picks a PRE-REGISTERED registry demo, loads its frozen demo tokens (d.npz) once;
per get_action: tap -> online w_t -> K=16 causal history ring -> phase ladder -> ±W window slice -> FROZEN
fusion -> z -> head stash (w_t, z) -> the inherited two-pass CFG get_action fires.

Controls (all serve-time switches on ONE checkpoint — doc 15 D20):
  none         real demo, real phase                        (A2)
  null         slot2 -> learned null (z never computed)     (C1)
  fusion_null  fusion runs with the null-window (drop_demo) (C1b — separates demo effect from history passthrough)
  wrong_task   registry demo of a DIFFERENT task            (C3)
  shuffle      real demo, token order permuted              (C4 — the headline control)
  frozen_phase real demo, window pinned at the demo middle  (C5)

Phase ladder v1 (D9): 'prop' = demo-paced clamp tau = min(g, M-1) (g = grid steps elapsed); 'frozen' via
the control. Per-episode tau trace + clamp fraction logged at reset boundaries (the saturation telemetry
the plan requires from eval #1).

Provenance guards (D19): d.npz _meta.registry_sha == fusion ckpt registry_sha; served demo must be in the
eval registry (unless --allow-any-demo); task-match unless control=wrong_task; conditioner proj_next != 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vla_training.eval._groot_wsm_eval import WSMEvalConditioner  # noqa: F401  (re-export for the serve)

CONTROLS = ("none", "null", "fusion_null", "wrong_task", "shuffle", "frozen_phase")


class DemoManager:
    """Loads + transforms the eval demo2 per episode; enforces registry + provenance."""

    def __init__(
        self,
        dtok_root: str,
        registry_path: str,
        fusion_registry_sha: str,
        device: str,
        control: str = "none",
        demo_index: int = 0,
        allow_any_demo: bool = False,
    ):
        assert control in CONTROLS, f"control {control!r} not in {CONTROLS}"
        self.root = Path(dtok_root).expanduser()
        self.registry = json.loads(Path(registry_path).expanduser().read_text())
        self.sha = fusion_registry_sha
        self.device, self.control, self.demo_index = device, control, int(demo_index)
        self.allow_any = allow_any_demo
        self._rng = np.random.default_rng(7)

    def _load(self, task: str, ep: int) -> torch.Tensor:
        z = np.load(self.root / task / f"demo_{ep:06d}" / "d.npz", allow_pickle=True)
        meta = json.loads(str(z["_meta"]))
        if self.sha not in ("?", "") and meta.get("registry_sha") not in ("?", self.sha):
            raise RuntimeError(
                f"[demo-mgr] d.npz registry sha {meta.get('registry_sha')} != fusion ckpt "
                f"sha {self.sha} — demo tokens from a different WSMv2 run. Refusing."
            )
        d = torch.from_numpy(z["d"].astype(np.float32)).to(self.device)
        if not torch.isfinite(d).all():
            raise RuntimeError(f"[demo-mgr] NON-FINITE demo tokens: {task}/demo_{ep:06d}")
        return d

    def pick(self, task: str) -> tuple[torch.Tensor, dict]:
        """Demo tokens [M,512] for this episode + a provenance record."""
        if task not in self.registry:
            raise RuntimeError(f"[demo-mgr] task {task!r} not in the eval registry ({len(self.registry)} tasks)")
        src_task = task
        if self.control == "wrong_task":
            others = sorted(t for t in self.registry if t != task)
            src_task = others[hash(task) % len(others)]  # deterministic per task
        eps = self.registry[src_task]
        ep = eps[min(self.demo_index, len(eps) - 1)]
        d = self._load(src_task, ep)
        if self.control == "shuffle":
            perm = torch.from_numpy(self._rng.permutation(len(d))).to(self.device)
            d = d[perm]
        return d, {"demo_task": src_task, "demo_ep": ep, "control": self.control, "M": int(len(d))}


def install_wsm_demo_cfg_eval(
    policy,
    conditioner,
    task_lang_table: dict[str, np.ndarray],
    demo_mgr: DemoManager,
    fusion,
    *,
    window: int = 20,
    control: str = "none",
    log_dir: str | None = None,
) -> None:
    """Patch the server-facing policy for demo-CFG: history ring (K from the fusion) + phase ladder +
    window slice + frozen fusion -> z -> head stash. Mirrors install_wsm_cfg_eval's proven seams."""
    from workspace_models.features.backbone_tap import BackboneTap
    from workspace_models.features.wsm_align import demo_window_at

    inner = getattr(policy, "policy", policy)
    tap = BackboneTap(inner)
    ah = inner.model.action_head
    cdt = next(ah.wsm_cfg.parameters()).dtype
    dev = next(ah.parameters()).device
    K = fusion.k_hist
    _orig_get_action = inner.get_action
    _orig_reset = policy.reset
    st: dict = {"task": None, "demo": None, "meta": None, "g": 0, "taus": [], "ep": 0}

    def _flush_log():
        if not st["taus"]:
            return
        M = st["meta"]["M"] if st["meta"] else 1
        taus = np.asarray(st["taus"])
        clamp_frac = float((taus >= M - 1).mean())
        print(
            f"[demo-eval] ep{st['ep']} {st['task']} demo={st['meta']} steps={len(taus)} "
            f"tau_end={int(taus[-1])}/{M - 1} clamp_frac={clamp_frac:.2f}",
            flush=True,
        )
        if log_dir:
            out = Path(log_dir).expanduser()
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"ep{st['ep']:04d}_{st['task']}.npz", taus=taus, meta=json.dumps(st["meta"]))

    def get_action(observation, *args, **kwargs):
        r = tap.tap(observation)
        assert r.patch_tokens.shape[0] == 1, "[demo-eval] serve is B=1 only"
        patch = r.patch_tokens[0].float().cpu().numpy()
        proprio = r.state_emb[0, 0].float().cpu().numpy()
        w_window, _lang = conditioner.step(patch, proprio)  # [K,512] causal ring (finite-guarded)
        hist = w_window.to(dev, torch.float32).unsqueeze(0)  # [1,K,512]
        w_t = w_window[-1].unsqueeze(0).to(dtype=cdt)  # slot1

        z = None
        if control != "null":
            d = st["demo"]
            M = d.shape[0]
            tau = M // 2 if control == "frozen_phase" else min(st["g"], M - 1)  # phase ladder v1 (prop-clamp)
            st["taus"].append(tau)
            idx, off, mask = demo_window_at(M, tau, window)
            dw = d[torch.from_numpy(idx).to(dev)].unsqueeze(0)  # [1,41,512]
            offt = torch.from_numpy(off).to(dev).unsqueeze(0)
            maskt = torch.from_numpy(mask).to(dev).unsqueeze(0)
            lang = torch.from_numpy(np.asarray(task_lang_table[st["task"]], dtype=np.float32)).to(dev).unsqueeze(0)
            with torch.no_grad():
                o = fusion(hist, dw, offt, maskt, lang, drop_demo=torch.tensor([control == "fusion_null"], device=dev))
            z = o["z"].to(dtype=cdt)
            if not torch.isfinite(z).all():
                raise RuntimeError("[demo-eval] NON-FINITE z — refusing to condition")
        st["g"] += 1
        ah._cfg_eval = (w_t, z)
        return _orig_get_action(observation, *args, **kwargs)

    def reset(options=None):
        task = (options or {}).get("task") if isinstance(options, dict) else None
        if task is None or task not in task_lang_table:
            raise RuntimeError(f"[demo-eval] reset needs a known task; got {task!r}")
        _flush_log()
        st.update(task=task, g=0, taus=[], ep=st["ep"] + 1)
        st["demo"], st["meta"] = demo_mgr.pick(task)
        conditioner.reset(task_lang_table[task])
        return _orig_reset(options) if _orig_reset is not None else {"status": "ok"}

    inner.get_action = get_action
    policy.reset = reset
    print(
        f"[demo-eval] installed: control={control} K={K} W={window} stride={conditioner.stride} "
        f"registry_sha={demo_mgr.sha}",
        flush=True,
    )
