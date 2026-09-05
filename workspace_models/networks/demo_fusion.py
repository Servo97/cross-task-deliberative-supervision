"""HistoryDemoFusion — causal policy-history queries cross-attend a windowed slice of demo2 tokens.

The WSMv2 fusion (doc 15, D6): queries = K frozen-WSMv1 history tokens (strictly CAUSAL self-attention —
the policy side never sees its own future); keys/values = the ±W demo-window tokens from the bidirectional
DemoEncoder (the demo IS fully observed, so past AND future demo tokens are legitimately visible), with a
learned relative-offset embedding (−W..+W) + a 3-way past/now/future embedding on the keys; language as
the AdaLN condition; ALL gates zero-init.

Exactness at init (unit-tested): the time/branch/offset embeddings are zero-init and every AdaLN gate is
zero-init, so at step 0  z == out_norm(hist[-1])  bitwise — demo information flows in only as gates open,
and the post-train starts as the exact w_t-conditioned baseline.

Branch dropout (D13) = the CFG-null pretraining: drop_demo -> the demo window is REPLACED by a small
learned null-token set (never sees demo values — unit-tested); drop_hist -> history tokens replaced by a
learned null-history set. The same nulled modes are what the conditioner's slot-dropout and s=0 serve at
post-train, and drop_demo doubles as the C1b fusion-null control at eval.

Train-only heads: per-k JEPA predictors (D10) and a phase head (D9). The phase head reads the DETACHED raw
history only — a hard no-grad boundary (unit-tested): wired on z it would manufacture the exact phase
channel the D4 PE-strip removed.

Perf: batched end-to-end, no python loops over batch; 16×41 attention is trivial — the module adds <5 ms
at serve; bf16/compile friendly.
"""

from __future__ import annotations

import torch
from torch import nn

from workspace_models.networks.adaln_zero import AdaLNZeroCrossBlock


class HistoryDemoFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        lang_dim: int = 2048,
        n_heads: int = 8,
        depth: int = 4,
        k_hist: int = 16,
        window: int = 20,
        n_null_window: int = 8,
        mlp_ratio: float = 4.0,
        jepa_ks: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.dim, self.k_hist, self.window = dim, k_hist, window
        self.lang_proj = nn.Linear(lang_dim, dim)
        self.null_lang = nn.Parameter(torch.zeros(dim))
        # query-side embeddings — ZERO-INIT so z == out_norm(hist[-1]) exactly at step 0.
        self.hist_time_emb = nn.Parameter(torch.zeros(k_hist, dim))
        self.branch_emb = nn.Parameter(torch.zeros(dim))
        # key-side embeddings — zero-init (keys unchanged at init; gates are the real gate anyway).
        self.rel_emb = nn.Parameter(torch.zeros(2 * window + 1, dim))  # offsets −W..+W
        self.pnf_emb = nn.Parameter(torch.zeros(3, dim))  # past / now / future
        # learned null sets (branch dropout / CFG-null / C1b control).
        self.null_window = nn.Parameter(torch.zeros(n_null_window, dim))
        self.null_hist = nn.Parameter(torch.zeros(k_hist, dim))
        self.blocks = nn.ModuleList(AdaLNZeroCrossBlock(dim, n_heads, mlp_ratio) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim)
        # train-only heads
        self.predictors = nn.ModuleDict(
            {str(k): nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)) for k in jepa_ks}
        )
        self.phase_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        causal = torch.triu(torch.ones(k_hist, k_hist, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)  # True = masked (torch MHA)

    def predict_phase(self, hist: torch.Tensor) -> torch.Tensor:
        """[B] phase estimate in [0,1] from the RAW history only. detach() = the hard no-grad boundary:
        the phase loss must never sculpt the fusion/encoder representations (shortcut critique F1)."""
        h = hist.detach()
        return torch.sigmoid(self.phase_head(torch.cat([h.mean(1), h[:, -1]], dim=-1))).squeeze(-1)

    def forward(
        self,
        hist: torch.Tensor,
        demo_win: torch.Tensor,
        demo_off: torch.Tensor,
        demo_mask: torch.Tensor,
        lang: torch.Tensor,
        *,
        drop_demo: torch.Tensor | None = None,
        drop_hist: torch.Tensor | None = None,
        lang_keep: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        """hist [B,K,dim] frozen-w history (oldest..newest); demo_win [B,Wn,dim] (Wn = 2*window+1);
        demo_off [B,Wn] int offsets in [−W, W]; demo_mask [B,Wn] bool True=VALID; lang [B,lang_dim];
        drop_demo/drop_hist [B] bool (True -> use the learned null set); lang_keep [B] bool.
        Returns dict: z [B,dim], h_all [B,K,dim], attn (list of [B,K,Wn'] | None)."""
        b = hist.shape[0]
        # --- queries: history + zero-init time/branch embs; optional null-history replacement ---
        q = hist
        if drop_hist is not None:
            q = torch.where(drop_hist[:, None, None], self.null_hist.expand(b, -1, -1).to(q.dtype), q)
        q = q + self.hist_time_emb + self.branch_emb

        # --- keys: window + rel-offset + past/now/future embs; null-window replacement on drop ---
        off_idx = (demo_off + self.window).long().clamp_(0, 2 * self.window)  # [B,Wn]
        pnf = (torch.sign(demo_off).long() + 1).clamp_(0, 2)  # past0 now1 future2
        k = demo_win + self.rel_emb[off_idx] + self.pnf_emb[pnf]
        key_pad = ~demo_mask  # True = PAD
        if drop_demo is not None:
            nw = self.null_window[None].expand(b, -1, -1).to(k.dtype)
            pad_to = k.shape[1] - nw.shape[1]
            nw_full = torch.cat([nw, nw[:, -1:].expand(b, pad_to, -1)], dim=1) if pad_to > 0 else nw[:, : k.shape[1]]
            null_pad = torch.zeros_like(key_pad)
            null_pad[:, self.null_window.shape[0] :] = True  # only the null set is valid
            k = torch.where(drop_demo[:, None, None], nw_full, k)
            key_pad = torch.where(drop_demo[:, None], null_pad, key_pad)

        cond = self.lang_proj(lang)
        if lang_keep is not None:
            cond = torch.where(lang_keep[:, None], cond, self.null_lang.expand(b, -1).to(cond.dtype))
        cond = cond[:, None, :]  # [B,1,dim]

        attns = []
        x = q
        for blk in self.blocks:
            out = blk(
                x, k, cond, self_attn_mask=self.causal_mask, memory_key_padding_mask=key_pad, need_weights=need_weights
            )
            x, w = out if need_weights else (out, None)
            attns.append(w)
        h_all = x
        z = self.out_norm(h_all[:, -1])
        return {"z": z, "h_all": h_all, "attn": attns}

    def predict_future(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """JEPA predictions per horizon k: {k: [B,dim]} (train only; targets = frozen w1[t+k])."""
        return {k: p(z) for k, p in self.predictors.items()}
