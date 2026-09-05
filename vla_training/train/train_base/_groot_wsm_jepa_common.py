"""Model-free WSM (JEPA aux-target) GR00T N1.7 wiring — the alternative to the injection canary
(_groot_wsm_common.py). Instead of INJECTING a precomputed w_t into the backbone via a TokenModulator,
this uses w as a TRAINING TARGET only: it JEPA-aligns the action head's PENULTIMATE features to the next
workspace latent w_{t+1} and SIGReg-regularizes them. w is never in the inference graph -> the injection
recipe's eval-OOD failure mode is structurally impossible. See internal_planning_and_todos/12.

Two least-invasive hooks (subclass + monkeypatch; NO edits to the vendored Isaac-GR00T repo):

  install_wsm_jepa_dataset(feats_root):
     ShardedSingleStepDataset -> WSMJepaShardedSingleStepDataset, whose get_datapoint appends
     `w_next [512] fp16` (the next grid-frame latent after the sampled native step) to the processor
     dict; the GR00T collator auto-stacks it into action_input.w_next [B,512].

  install_wsm_jepa_action_head(...):
     Gr00tN1d7Pipeline._create_model -> reclass model.action_head to WSMJepaActionHead, register a
     forward_pre_hook on the DiT's proj_out_2 (captures the penultimate hidden [B,seq,512] WITH grad,
     no vendored edit), and attach a trainable JEPAPredictor. forward() adds
     lambda_jepa*JEPA + lambda_sigreg*SIGReg to the flow loss. get_action() is untouched -> inference
     is byte-identical to the baseline (the hook only reads).

GR00T-venv-only: every gr00t/torch import is inside a function. Reuses assert_wt_coverage from the
injection module (coverage contract is identical).
"""

from __future__ import annotations

from vla_training.train.train_base._groot_wsm_common import assert_wt_coverage  # re-export (same contract)

__all__ = ["install_wsm_jepa_dataset", "install_wsm_jepa_action_head", "assert_wt_coverage"]


def install_wsm_jepa_dataset(feats_root: str, demo_fraction: float = 0.30, num_futures: int = 1, soup=None) -> None:
    """Attach the next-grid workspace latent(s) w_next per sample (the JEPA target).

    k == 1: `w_next [512]`, `w_next_valid` scalar bool — the shipped contract, byte-identical.
    k > 1: `w_next [k, 512]`, `w_next_valid [k]` — grid offsets +1..+k, rows past the end of the
    grid clamped to the last available row and MASKED OUT.

    The k > 1 branch delegates to `wsm_align.next_k_at(ks=(1..k))`, which is exactly equivalent to
    the pi loader's `_wsm_jepa_target` k > 1 branch (`groot_openpi_dataset.py`): pi takes the first
    k grid rows STRICTLY AFTER the frame and pads by repeating the last available row; next_k_at
    takes base+1..base+k clamped to the last row, where base is the newest grid row <= t. Since
    future[j] == base+1+j on a sorted grid, the selected indices, the padding, and the valid mask all
    agree. `tests/test_groot_wsm_jepa_k.py::test_target_matches_pi_semantics` pins that equivalence
    against a fixture extracted from the pi loader — it is the invariant a bug here would break, and
    a target-pipeline bug of exactly this class once cost a 12.4-avg collapse.
    """
    from pathlib import Path

    import gr00t.data.dataset.factory as factory_module
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    from workspace_models.features.wsm_align import load_w, next_at_with_valid, next_k_at

    if num_futures < 1:
        raise ValueError(f"num_futures must be >= 1, got {num_futures}")
    future_ks = tuple(range(1, int(num_futures) + 1))
    # `soup` maps a lerobot dir to its task. Pass gs.soup from the train entry; the
    # combined_target_soup default is RoboCasa-only and resolves zero tasks on remembench13.
    if soup is None:
        from utils.soup import combined_target_soup

        soup = combined_target_soup(demo_fraction)

    feats_root = Path(feats_root).expanduser()
    task_by_dir = {str(Path(m["path"]).resolve()): m["task"] for m in soup}
    task_names = set(task_by_dir.values())
    if not task_names:
        raise ValueError("[wsm-jepa] empty soup: cannot map any lerobot dir to a task")

    class WSMJepaShardedSingleStepDataset(ShardedSingleStepDataset):
        def _wsm_task(self) -> str:
            key = str(Path(self.dataset_path).resolve())
            if key in task_by_dir:
                return task_by_dir[key]
            parts = set(Path(self.dataset_path).parts)
            for t in task_names:
                if t in parts:
                    return t
            raise KeyError(f"[wsm-jepa] cannot resolve task for dataset_path={self.dataset_path}")

        def _wsm_load(self, ep_idx: int):
            if getattr(self, "_wsm_key", None) == ep_idx:
                return self._wsm_val
            ep_index = int(self.episode_loader.episodes_metadata[ep_idx]["episode_index"])
            fdir = feats_root / self._wsm_task() / f"demo_{ep_index:06d}"
            if not (fdir / "w.npz").exists():
                raise FileNotFoundError(f"[wsm-jepa] policy features missing: {fdir}/w.npz")
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
            if num_futures == 1:
                w_next, valid = next_at_with_valid(w, fi, int(step_index))  # [512], scalar bool
            else:
                w_next, valid = next_k_at(w, fi, int(step_index), ks=future_ks)  # [k,512], [k]
            out["w_next"] = np.asarray(w_next, dtype=np.float16)
            out["w_next_valid"] = np.asarray(valid, dtype=np.bool_)  # terminal-tail mask
            return out

    factory_module.ShardedSingleStepDataset = WSMJepaShardedSingleStepDataset
    print(f"[wsm-jepa] dataset patched: feats_root={feats_root} tasks={len(task_names)} (target=w_next)", flush=True)


def attach_wsm_jepa(
    model,
    w_dim: int = 512,
    jepa_weight: float = 1.0,
    sigreg_weight: float = 0.05,
    direct: bool = False,
    log_every: int = 50,
    num_futures: int = 1,
):
    """Reclass model.action_head -> WSMJepaActionHead, hook the DiT penultimate, attach a JEPAPredictor.
    Returns the predictor (the only net-new trainable module; the DiT/projector train as in baseline FT)."""

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    from workspace_models.networks.jepa_align_head import JEPAPredictor, wsm_jepa_sigreg_loss

    class WSMJepaActionHead(Gr00tN1d7ActionHead):
        def forward(self, backbone_output, action_input):
            out = super().forward(backbone_output, action_input)  # runs DiT -> proj_out_2 hook fires
            if not self.training:
                return out
            w_next = action_input["w_next"] if "w_next" in action_input else None
            w_next_valid = action_input["w_next_valid"] if "w_next_valid" in action_input else None
            penult = getattr(self, "_jepa_penult", None)  # [B, seq, Dp] captured by the hook
            if w_next is None or w_next_valid is None or penult is None or self.jepa_predictor is None:
                raise RuntimeError(
                    "[wsm-jepa] missing w_next/w_next_valid, hook output, or predictor on a "
                    "training batch; refusing to silently run as the baseline"
                )
            H = int(action_input["action"].shape[1])
            penult_act = penult[:, -H:, :]  # action tokens = last H (pred[:,-H:])
            aux, m = wsm_jepa_sigreg_loss(
                penult_act,
                w_next,
                self.jepa_predictor,
                jepa_weight=jepa_weight,
                sigreg_weight=sigreg_weight,
                global_step=self._jepa_step,
                target_valid=w_next_valid,
                num_futures=num_futures,
            )
            out["loss"] = out["loss"] + aux
            self._jepa_last = m
            if self._jepa_step % log_every == 0:
                print(
                    f"[wsm-jepa] step {self._jepa_step} flow {float(out['loss'].detach()) - float(aux.detach()):.4f} "
                    f"| jepa {m['jepa']:.4f} (cos {m['cos']:.3f}) sigreg {m['sigreg']:.3f} "
                    f"valid {m['jepa_valid']}/{m['jepa_total']} k={m['num_futures']} "
                    f"[w={jepa_weight}/{sigreg_weight}] ACTIVE",
                    flush=True,
                )
            self._jepa_step += 1
            return out

    ah = model.action_head
    ah.__class__ = WSMJepaActionHead
    ah._jepa_step = 0
    ah._jepa_last = {}
    ah._jepa_penult = None

    # capture the DiT penultimate (input to proj_out_2 == the line-302 hidden, [B,seq,inner_dim]) with grad
    dit = ah.model
    in_dim = int(dit.proj_out_2.in_features)  # inner_dim (== 512 == w_dim)

    def _pre_hook(_module, inp):
        ah._jepa_penult = inp[0]

    dit.proj_out_2.register_forward_pre_hook(_pre_hook)

    ref = next((p for p in ah.parameters() if p.requires_grad), None)
    dev = ref.device if ref is not None else next(ah.parameters()).device
    dt = ref.dtype if ref is not None else next(ah.parameters()).dtype
    ah.jepa_predictor = JEPAPredictor(in_dim=in_dim, w_dim=w_dim, direct=direct, num_futures=num_futures).to(
        device=dev, dtype=dt
    )
    ah.jepa_predictor.requires_grad_(True)
    print(
        f"[wsm-jepa] attached: penult_dim={in_dim} w_dim={w_dim} direct={direct} k={num_futures} "
        f"lambda_jepa={jepa_weight} lambda_sigreg={sigreg_weight} dtype={dt} "
        f"predictor_params={sum(p.numel() for p in ah.jepa_predictor.parameters()) / 1e3:.1f}K",
        flush=True,
    )
    return ah.jepa_predictor


def install_wsm_jepa_action_head(
    w_dim: int = 512, jepa_weight: float = 1.0, sigreg_weight: float = 0.05, direct: bool = False, num_futures: int = 1
) -> None:
    """TRAIN path: monkeypatch Gr00tN1d7Pipeline._create_model to attach the JEPA aux post-load."""
    import gr00t.model.gr00t_n1d7.setup as setup_module

    _orig_create = setup_module.Gr00tN1d7Pipeline._create_model

    def _patched_create(self):
        model = _orig_create(self)  # HF strict-load happens here
        attach_wsm_jepa(
            model,
            w_dim=w_dim,
            jepa_weight=jepa_weight,
            sigreg_weight=sigreg_weight,
            direct=direct,
            num_futures=num_futures,
        )
        return model

    setup_module.Gr00tN1d7Pipeline._create_model = _patched_create
    print(
        f"[wsm-jepa] action head patched (train path): JEPA+SIGReg aux at the DiT penultimate "
        f"(lambda {jepa_weight}/{sigreg_weight}, direct={direct}, k={num_futures})",
        flush=True,
    )
