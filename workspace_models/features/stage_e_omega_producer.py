#!/usr/bin/env python3
"""Stage-E online ω producer — the serve-time counterpart of `train_stage_e.export_omega_store`.

WHY THIS FILE EXISTS. The sealed ReMemBench serve path (`serve_pi_05_wsm_cfg.py --encoder-ckpt`)
taps the frozen pi backbone for `[192, 2048]` patch tokens and runs a `WorkspaceModel` that pools
them INTERNALLY. A Stage-E encoder is a different family: `cfg.backbone_dim == 512` (it consumes
ALREADY-POOLED tokens) and its state dict has no `decoder.*` at all, so both ω loaders in the tree
(`load_wsm`, `load_wsm_stage_s`) reject it. Nothing here edits those paths; this is a new component.

THE CHAIN, and why each link is the one it is:

    frozen pi tap  ->  [F,192,2048] patch tokens + [F,2048] per-frame language
      |                (§16: the rmb tap is the SAME frozen `mg60_bal33/run/149999` backbone
      |                 RoboCasa was tapped from — provenance, not coincidence)
      v
    frozen WSMv1 PatchPool  ->  p_t [512]
      |                (`pi_pooled_tap.load_pool` over `wsm_runs/pi_wsm_v1/wsm_step100000.pt`,
      |                 the pooler §16 verified at cos 0.999997 mean / 0.999992 min against the
      |                 archived `wsm_pooled/pi_100k` cache on four re-pooled demos)
      v
    Stage-E domain adapter[`remembench`] + shared causal trunk  ->  ω_t [512]
                       (§16.1: the adapter is selected by the GLOBAL domain id
                        `DOMAINS.index("remembench") == 1`, never by position — positional
                        matching routes every ReMemBench frame through RoboCasa's adapter)

EXACTNESS OF THE ONLINE PATH. `WorkspaceEncoder.encode_fused` adds an ABSOLUTE time embedding
`time_emb[:T]` and masks with `(src > tgt) | (src < tgt - (c_horizon - 1))` — a pure banded causal
mask. Row t therefore sees exactly `[t - c_horizon + 1, t]` whether it is the last row of a
(t+1)-length prefix or an interior row of the full episode. The incremental producer is not an
approximation of the batch export; it is the same computation, and `omega_sidecar_parity`-style
gating measures only float nondeterminism plus the fp16 storage floor.

THE fp16 ROUND TRIP IS PART OF THE CONTRACT, not an accident. `train_stage_e.Corpus` stores
`feat` as fp16 and `lang` as fp16 and the adapter casts back with `.float()`. Serving fp32 straight
from the pool would feed the trunk a different tensor than training did, so `_as_trained` reproduces
the round trip explicitly.

Everything here is inference-only and allocates no gradients.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from workspace_models.networks.stage_e_encoder import StageEEncoder

# The GLOBAL domain ordering `train_stage_e.DOMAINS` tags every episode with. Duplicated as a
# literal rather than imported so that serving does not drag the trainer (and its torch/np/edge
# machinery) into the policy server process; asserted against the checkpoint in `load_stage_e`.
DOMAINS = ("robocasa", "remembench", "robomme", "robocerebra")

_REQUIRED_KEYS = ("model", "adapters", "cfg", "domains")


def _as_trained(x: np.ndarray | torch.Tensor, device) -> torch.Tensor:
    """fp16 round trip + float32 cast — exactly what Corpus storage + DomainAdapter.float() do."""
    t = torch.as_tensor(np.asarray(x)) if not isinstance(x, torch.Tensor) else x
    return t.to(device=device, dtype=torch.float16).float()


def load_stage_e(ckpt_path: str | Path, device: str = "cuda"):
    """Load a Stage-E encoder checkpoint. Refuses non-finite weights (the NaN-encoder lesson).

    Returns (encoder, blob). The encoder is eval-mode with grads disabled.
    """
    path = Path(ckpt_path).expanduser()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in _REQUIRED_KEYS if k not in blob]
    if missing:
        raise ValueError(
            f"{path} is not a Stage-E encoder checkpoint (missing {missing}). "
            f"A sealed Stage-S WorkspaceModel goes through load_wsm/load_wsm_stage_s instead."
        )
    cfg = SimpleNamespace(**blob["cfg"])
    domains = list(blob["domains"])
    index = list(blob.get("domain_index", range(len(domains))))
    if len(index) != len(domains):
        raise ValueError(f"{path}: domains {domains} and domain_index {index} disagree in length")
    for name, idx in zip(domains, index):
        if name in DOMAINS and DOMAINS.index(name) != int(idx):
            raise ValueError(
                f"{path}: domain '{name}' carries index {idx} but the global ordering says "
                f"{DOMAINS.index(name)} — refusing to serve a checkpoint whose adapter routing "
                f"disagrees with train_stage_e.DOMAINS (see §16.1)."
            )
    specs = {
        name: {"feat_dim": int(cfg.backbone_dim), "lang_dim": int(cfg.lang_dim), "index": int(idx)}
        for name, idx in zip(domains, index)
    }

    encoder = StageEEncoder(cfg, specs)
    trunk_sd = {k[len("encoder.") :]: v for k, v in blob["model"].items() if k.startswith("encoder.")}
    if len(trunk_sd) != len(blob["model"]):
        raise ValueError(f"{path}: 'model' holds keys without the 'encoder.' prefix")
    encoder.trunk.load_state_dict(trunk_sd, strict=True)
    encoder.adapters.load_state_dict(blob["adapters"], strict=True)

    bad = [n for n, p in encoder.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise ValueError(
            f"Stage-E encoder {path} has NON-FINITE weights in {len(bad)} tensors; "
            f"refusing to serve (a diverged encoder emits NaN ω -> 0% eval)"
        )
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, blob


class StageEServeEncoder(torch.nn.Module):
    """Presents a Stage-E encoder through the `WorkspaceModel` surface the SEALED serve path uses.

    `WSMEvalConditioner` (`vla_training/eval/_groot_wsm_eval.py`) drives ω from exactly two methods,
    `fuse_inputs(patches, proprio, cond_lang)` and `encode_fused(fused, cond)`, and refuses an
    encoder that lacks them. Implementing that pair — rather than writing a second conditioner —
    keeps the causal window, stride and K-row selection on the sealed code path, so the only new
    logic in the serve chain is the front end this class supplies:

        patches [B,T,192,2048]  ->  frozen WSMv1 PatchPool  ->  p [B,T,512]  ->  Stage-E adapter

    `proprio` is accepted and IGNORED, which is the Stage-E input contract rather than an oversight:
    pi0.5 discretises robot state into the prompt, so the pooled store is vision+language only and
    `proprio_proj` was never trained (see `stage_e_encoder` module docstring).
    """

    def __init__(self, encoder: StageEEncoder, domain: str, pool, patch_norm=None, device: str = "cuda") -> None:
        super().__init__()
        if domain not in encoder.adapters:
            raise ValueError(f"domain '{domain}' not in checkpoint adapters {list(encoder.adapters)}")
        self.encoder, self.domain, self.device = encoder, domain, device
        self.cfg = encoder.cfg
        self.adapter = encoder.adapters[domain]
        self.pool, self.patch_norm = pool, patch_norm

    def fuse_inputs(
        self, patches: torch.Tensor, proprio: torch.Tensor, cond_lang: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del proprio  # Stage-E's pooled contract is proprio-free by construction.
        x = patches.to(self.device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            if self.patch_norm is not None:
                x = self.patch_norm(x)
            p = self.pool(x)  # [B,T,192,2048] -> [B,T,512]
        # The fp16 round trip the pooled store and Corpus both applied, reproduced on the serve path.
        p = p.float().half().float()
        f, c = self.adapter(p, cond_lang.to(self.device).float())
        return f, self.encoder.trunk.lang_proj(c)

    def encode_fused(self, fused: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.encoder.trunk.encode_fused(fused, cond)


class StageEOmegaProducer:
    """ω from pooled tap tokens, for one episode of one domain.

    Two entry points, deliberately both present:

      * `omega_episode(p)`  — whole episode in one pass. This is byte-for-byte the call
        `train_stage_e.export_omega_store` makes, so it is the reference the parity gate scores
        the online path against.
      * `reset()` / `step(p_t)` — the SERVE path: grows a causal prefix one grid frame at a time
        and returns only the newest ω. Frame-local `fused`/`cond` rows are cached across steps
        (`WorkspaceEncoder.fuse_inputs` documents them as safe to cache during an online episode),
        so a step costs one trunk pass over the prefix, not a re-projection of it.
    """

    def __init__(
        self, encoder: StageEEncoder, domain: str, lang_global, device: str = "cuda", max_prefix: int | None = None
    ) -> None:
        if domain not in encoder.adapters:
            raise ValueError(f"domain '{domain}' not in checkpoint adapters {list(encoder.adapters)}")
        self.encoder, self.domain, self.device = encoder, domain, device
        self.cfg = encoder.cfg
        self.adapter = encoder.adapters[domain]
        lang = _as_trained(np.asarray(lang_global, dtype=np.float32), device)
        if lang.shape[-1] != int(self.cfg.lang_dim):
            raise ValueError(f"lang_global dim {lang.shape[-1]} != cfg.lang_dim {self.cfg.lang_dim}")
        # [D] = one vector held for the whole episode (how Stage-E was TRAINED: Corpus broadcasts
        # each demo's lang_global across every frame). [T,D] = a time-varying stream, which is what
        # a causal serve convention must use because the episode mean is not knowable in advance.
        if lang.ndim == 1:
            self._lang = lang.reshape(1, 1, -1)
        elif lang.ndim == 2:
            self._lang = lang.unsqueeze(0)
        else:
            raise ValueError(f"lang must be [D] or [T,D]; got {tuple(lang.shape)}")
        # The prefix is NEVER slid. `encode_fused` indexes an ABSOLUTE time embedding (`time_emb[:T]`),
        # so dropping leading frames would renumber every retained frame and silently produce a
        # different ω than the batch export. Truncation would only ever be a speed optimisation, and
        # it is not needed: at the stride-8 grid the longest ReMemBench horizon (1400 env steps) is
        # ~175 grid frames against a c_horizon of 1000 and max_t >= 1200, so nothing is ever
        # unreachable. Exceeding capacity is an error, not something to paper over.
        self.max_prefix = int(max_prefix) if max_prefix else int(self.cfg.max_t)
        if self.max_prefix > int(self.cfg.max_t):
            raise ValueError(f"max_prefix {self.max_prefix} exceeds time-embedding capacity {self.cfg.max_t}")
        self._fused: list[torch.Tensor] = []
        self._cond: torch.Tensor | None = None
        self.n_steps = 0

    # -- shared front end -----------------------------------------------------------------
    @torch.no_grad()
    def _project(self, p: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """p [B,T,feat_dim] (already fp16-round-tripped float32) -> (fused, cond) [B,T,dim].

        `offset` is the absolute grid position of p's first frame; it selects the matching slice of
        a time-varying language stream and is ignored for a held constant.
        """
        b, t = p.shape[0], p.shape[1]
        if self._lang.shape[1] == 1:
            lang = self._lang.expand(b, t, -1)
        else:
            if offset + t > self._lang.shape[1]:
                raise ValueError(f"language stream has {self._lang.shape[1]} frames; needed {offset + t}")
            lang = self._lang[:, offset : offset + t].expand(b, t, -1)
        f, c = self.adapter(p, lang)
        return f, self.encoder.trunk.lang_proj(c)

    # -- batch path (the export reference) ------------------------------------------------
    @torch.no_grad()
    def omega_episode(self, p) -> torch.Tensor:
        """p [F, feat_dim] -> ω [F, dim] float32."""
        x = _as_trained(p, self.device)
        if x.ndim != 2:
            raise ValueError(f"omega_episode expects [F,feat_dim]; got {tuple(x.shape)}")
        if x.shape[0] > int(self.cfg.max_t):
            raise ValueError(f"episode length {x.shape[0]} exceeds time-embedding capacity {self.cfg.max_t}")
        fused, cond = self._project(x[None])
        return self.encoder.trunk.encode_fused(fused, cond)[0]

    # -- online path (the serve path) ------------------------------------------------------
    def reset(self) -> None:
        """Episode boundary. The GDN read rebuilds its conditioner at wsm_t == 0; ω must too."""
        self._fused.clear()
        self._cond = None
        self.n_steps = 0

    @torch.no_grad()
    def step(self, p_t, lang_t=None) -> torch.Tensor:
        """One new grid frame's pooled tokens [feat_dim] -> that frame's ω [dim] float32.

        `lang_t` overrides the conditioning for THIS frame only — the serve path for a causal
        running-mean convention, where the vector is not knowable until the frame arrives.
        """
        x = _as_trained(np.asarray(p_t).reshape(1, 1, -1), self.device)
        if lang_t is not None:
            saved = self._lang
            self._lang = _as_trained(np.asarray(lang_t, dtype=np.float32), self.device).reshape(1, 1, -1)
            try:
                fused, cond = self._project(x)
            finally:
                self._lang = saved
        else:
            fused, cond = self._project(x, offset=len(self._fused))
        self._fused.append(fused[:, 0])
        self._cond = cond[:, 0] if self._cond is None else torch.cat([self._cond, cond[:, 0]], 0)
        if len(self._fused) > self.max_prefix:
            raise ValueError(
                f"episode reached {len(self._fused)} grid frames, past the trunk's capacity "
                f"{self.max_prefix}; ω would have to be renumbered to continue — refusing"
            )
        seq = torch.stack(self._fused, dim=1)  # [1,T,dim]
        cond_seq = self._cond.unsqueeze(0)  # [1,T,dim]
        omega = self.encoder.trunk.encode_fused(seq, cond_seq)
        self.n_steps += 1
        return omega[0, -1]
