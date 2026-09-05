"""H14 Stage-E encoder: per-domain input adapters -> ONE shared causal trunk -> ω_t (amendment A3).

Why adapters exist. The three domains are tapped from DIFFERENT frozen networks (pi0.5 backbone for
RoboCasa/ReMemBench, a frozen SigLIP for RoboMME). A3 makes the bridge a MEASURED decision:
"design default: per-domain input adapters (LayerNorm + affine) into the shared trunk. If stats are
irreconcilable, RoboMME drops out of the joint encoder (it stays in the deliberation corpus)."
LayerNorm removes the per-tap RMS/scale mismatch that is a NaN hazard first and a
silent-domination hazard second; the affine that follows is the only per-domain parameter, so the
representation the deliberative objective shapes lives in the SHARED trunk by construction.

The trunk is `WorkspaceEncoder` unchanged (E1 keeps the 1x512 ω — pin D4: never change signal and
capacity in one arm), so:
  * the checkpoint carries a real `encoder.*` state_dict and every existing ω consumer, gate script
    and `encoder_id` manifest reads it without a code change;
  * `encode_fused` is called verbatim, which means the C-horizon causal window, the AdaLN-Zero
    stack and `out_norm` are the same objects the sealed Stage-S/a2 numbers were measured on.

Input contract. Stage E consumes POOLED tap tokens (`p.npz`, [F, 512]), not raw patch tokens:
`workspace_models/features/pool_patch_tokens.py` is the sanctioned WSMv2 encoder-phase contract
("the ENTIRE WSMv2 encoder phase becomes a small-tensor job — p.npz + w.npz + lang only"), and the
pool it applies is frozen. The proprio slot is therefore empty: pi0.5 discretises robot state into
the prompt, and the per-frame prompt embedding is not part of the pooled store for any domain. The
pooled tokens are vision+language-only — exactly the proprio-free contract that file already
declares — and `proprio_proj` / `pool` are left untrained rather than fed a fabricated tensor.
"""

from __future__ import annotations

import torch
from torch import nn

from workspace_models.networks.workspace_latent import WorkspaceEncoder


class DomainAdapter(nn.Module):
    """LayerNorm + affine into the shared trunk width (A3's design default, one per domain)."""

    def __init__(self, feat_dim: int, lang_dim: int, dim: int) -> None:
        super().__init__()
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_proj = nn.Linear(feat_dim, dim)
        self.lang_norm = nn.LayerNorm(lang_dim)

    def forward(self, feat: torch.Tensor, lang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feat_proj(self.feat_norm(feat)), self.lang_norm(lang)


class StageEEncoder(nn.Module):
    """(pooled tap tokens, language, domain) -> ω [B, T, dim] through one shared causal trunk."""

    def __init__(self, cfg, domain_specs: dict) -> None:
        """domain_specs: {domain_name: {"feat_dim": int, "lang_dim": int, "index": int}}.

        `index` is the value `forward`'s `domain_index` tensor actually carries — the caller's GLOBAL
        domain id (`train_stage_e.DOMAINS.index(name)`), not this encoder's local ordering. It must
        be given whenever more than one domain is loaded: the local order is `sorted(domain_specs)`,
        which for {robocasa, remembench} is (remembench, robocasa) — the REVERSE of the global
        (robocasa, remembench, robomme) — so a positional match would silently route each domain's
        frames through the other domain's adapter. Single-domain callers may omit it (local 0 ==
        global 0 for robocasa, and every run before the rmb tap existed was single-domain, so those
        checkpoints are unaffected).
        """
        super().__init__()
        self.cfg = cfg
        self.trunk = WorkspaceEncoder(cfg)
        self.domains = tuple(sorted(domain_specs))
        self.domain_index = tuple(int(domain_specs[name].get("index", i)) for i, name in enumerate(self.domains))
        if len(set(self.domain_index)) != len(self.domain_index):
            raise ValueError(f"duplicate domain indices {self.domain_index} for {self.domains}")
        self.adapters = nn.ModuleDict(
            {
                name: DomainAdapter(int(spec["feat_dim"]), int(spec["lang_dim"]), cfg.dim)
                for name, spec in domain_specs.items()
            }
        )
        if any(int(spec["lang_dim"]) != cfg.lang_dim for spec in domain_specs.values()):
            raise ValueError(
                "every domain's lang_dim must equal cfg.lang_dim: the trunk's "
                "lang_proj is SHARED, and a per-domain language projection would put "
                "the AdaLN condition in a different space per domain"
            )

    def trunk_parameters(self):
        """Trunk params that actually receive gradient. `pool`/`proprio_proj` are unused on the
        pooled-token input path and are excluded so AdamW's weight decay cannot quietly shrink
        weights no loss ever touches (they still ship in the checkpoint, unmodified)."""
        skip = {"pool.", "proprio_proj."}
        return [p for n, p in self.trunk.named_parameters() if not any(n.startswith(s) for s in skip)]

    def trainable_parameters(self):
        return list(self.adapters.parameters()) + self.trunk_parameters()

    def forward(self, feat: torch.Tensor, lang: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor:
        """feat [B,T,feat_dim], lang [B,T,lang_dim], domain_index [B] int64 into self.domains.

        Padding must be on the RIGHT: the trunk is causal, so a valid frame never attends to a
        padded one and right-padding is exactly invisible to every frame the loss reads.
        """
        fused = feat.new_zeros(feat.shape[0], feat.shape[1], self.cfg.dim, dtype=torch.float32)
        cond = fused.clone()
        for name, index in zip(self.domains, self.domain_index):
            rows = domain_index == index
            if not bool(rows.any()):
                continue
            f, c = self.adapters[name](feat[rows].float(), lang[rows].float())
            fused[rows] = f
            cond[rows] = self.trunk.lang_proj(c)
        return self.trunk.encode_fused(fused, cond)

    def state_payload(self) -> dict:
        """Checkpoint body. `model` keeps the `encoder.<k>` prefix every existing loader expects."""
        return {
            "model": {f"encoder.{k}": v for k, v in self.trunk.state_dict().items()},
            "adapters": self.adapters.state_dict(),
            "domains": list(self.domains),
            "domain_index": list(self.domain_index),
        }
