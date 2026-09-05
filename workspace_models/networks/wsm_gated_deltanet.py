"""Torch port of the pi0.5/JAX `WSMGatedDeltaNetConditioner` (openpi `models/wsm_current_cond.py`).

The STEERING variant of the tanh workspace read: a gated delta-rule linear-attention recurrence run
across the K positions of ONE omega window, read out at the newest position and gated by tanh(alpha).
STATELESS across policy calls (S is built inside one call from the window that was already shipped),
so this is a conditioner swap, NOT the RoboTTT fast-weight axis.

    k_i, q_i = l2(W_k w_i), l2(W_q w_i)        v_i = W_v w_i
    beta_i   = sigmoid(W_beta w_i)             gamma_i = exp(-softplus(W_decay w_i + c_i))
    S_i      = gamma_i S_{i-1} + beta_i (v_i - gamma_i S_{i-1} k_i) k_i^T
    out      = tanh(alpha) * R(S_K q_K)

PARITY CONTRACT with the JAX original (pinned by tests/test_groot_wsm_deltanet.py against a fixture
extracted from the JAX module): same params + same omega window => same output within fp tolerance.
The only representational difference is the weight LAYOUT — `nnx.Linear` holds `kernel [in, out]` and
computes `x @ kernel + bias`, while `torch.nn.Linear` holds `weight [out, in]`; `load_jax_params`
below transposes. Nothing else about the math is allowed to drift.

Inherited invariants (do not "simplify" these away):
* INIT DISCIPLINE: every projection is normally initialized; near-silence at step 0 comes ONLY from
  tanh(alpha) with alpha=1e-3. The readout is NOT zero-init, so every parameter has a live gradient
  from the first step (see the dead-gradient note in the jax tree's wsm_modulator.py).
* L2 on k AND q: k because the delta rule's k k^T must not blow up the state, q so the readout
  magnitude does not scale with head_dim.
* `pos_decay_bias` is [window_len, num_heads], zero-init, added to the decay LOGIT. Exact no-op at
  init, and its leading axis makes the trained window length STRUCTURALLY READABLE from the
  checkpoint -- that is how serve auto-detects the window without a new eval flag.
* Window-length mismatch is a HARD ERROR, never a silent reshape: a checkpoint trained at K=8 fed a
  K=1 window would read a different (and much weaker) conditioner than the one that was trained.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def current_workspace_token(omega_window: torch.Tensor) -> torch.Tensor:
    """Select causal omega_t from an oldest-to-newest window [..., K, D]."""
    if omega_window is None:
        raise ValueError("current_workspace_token requires a workspace-token window")
    if omega_window.ndim < 2 or omega_window.shape[-2] < 1:
        raise ValueError(f"expected [..., K, D] with K>=1, got shape={tuple(omega_window.shape)}")
    return omega_window[..., -1, :]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit-norm over the last axis. Keeps the delta-rule's k k^T outer products O(1).

    eps is added to the NORM (not inside the sqrt) to match the JAX original exactly.
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class WSMGatedDeltaNetConditioner(nn.Module):
    """Gated delta-rule read over the omega window -> one additive conditioning vector.

    Args mirror the JAX module. `cond_dim` is the width of the seam this is added into (for GR00T
    N1.7 that is the DiT/action-head conditioning width, not pi0.5's 1024).
    """

    def __init__(
        self,
        w_dim: int = 512,
        cond_dim: int = 1024,
        window_len: int = 1,
        num_heads: int = 2,
        head_dim: int = 256,
        gate_init: float = 1e-3,
        history_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(gate_init)):
            raise ValueError(f"gate_init must be finite, got {gate_init}")
        if window_len < 1:
            raise ValueError(f"window_len must be >= 1, got {window_len}")
        if num_heads < 1 or head_dim < 1:
            raise ValueError(f"num_heads/head_dim must be >= 1, got {num_heads}/{head_dim}")
        if not math.isfinite(float(history_dropout)) or not 0.0 <= float(history_dropout) < 1.0:
            raise ValueError(f"history_dropout must be in [0, 1), got {history_dropout}")
        self.w_dim = int(w_dim)
        self.cond_dim = int(cond_dim)
        self.window_len = int(window_len)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.history_dropout = float(history_dropout)
        inner = self.num_heads * self.head_dim
        self.proj_q = nn.Linear(w_dim, inner)
        self.proj_k = nn.Linear(w_dim, inner)
        self.proj_v = nn.Linear(w_dim, inner)
        self.proj_beta = nn.Linear(w_dim, self.num_heads)
        self.proj_decay = nn.Linear(w_dim, self.num_heads)
        self.pos_decay_bias = nn.Parameter(torch.zeros(self.window_len, self.num_heads))
        self.proj_readout = nn.Linear(inner, cond_dim)
        self.alpha = nn.Parameter(torch.full((cond_dim,), float(gate_init)))

    # -- parity plumbing ---------------------------------------------------------------------
    @torch.no_grad()
    def load_jax_params(self, arrays: dict) -> None:
        """Load a JAX/nnx parameter dump (kernels [in, out]) into this module (weights [out, in])."""

        def _set(lin: nn.Linear, name: str) -> None:
            kernel = torch.as_tensor(arrays[f"{name}_kernel"], dtype=torch.float32)
            bias = torch.as_tensor(arrays[f"{name}_bias"], dtype=torch.float32)
            if tuple(kernel.shape) != (lin.in_features, lin.out_features):
                raise ValueError(
                    f"{name}: jax kernel {tuple(kernel.shape)} != expected ({lin.in_features}, {lin.out_features})"
                )
            lin.weight.copy_(kernel.T)
            lin.bias.copy_(bias)

        for name in ("proj_q", "proj_k", "proj_v", "proj_beta", "proj_decay", "proj_readout"):
            _set(getattr(self, name), name)
        self.pos_decay_bias.copy_(torch.as_tensor(arrays["pos_decay_bias"], dtype=torch.float32))
        self.alpha.copy_(torch.as_tensor(arrays["alpha"], dtype=torch.float32))

    # -- forward ------------------------------------------------------------------------------
    def _split_heads(self, x: torch.Tensor, lead: tuple[int, ...], k_len: int) -> torch.Tensor:
        return x.reshape(*lead, k_len, self.num_heads, self.head_dim)

    def _history_keep_mask(self, lead: tuple[int, ...], k_len: int, device, generator) -> torch.Tensor:
        """[*lead, K] bool: True = this window element participates in the recurrence."""
        probs = torch.full((*lead, k_len), 1.0 - self.history_dropout, device=device)
        keep = torch.bernoulli(probs, generator=generator).bool()
        keep[..., -1] = True  # the newest slot (current timestep) is NEVER deleted
        return keep

    def forward(
        self,
        omega_window: torch.Tensor,
        *,
        train: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """[..., K, w_dim] (oldest..newest, causal) -> [..., cond_dim] additive conditioning term."""
        if omega_window is None:
            raise ValueError("WSMGatedDeltaNetConditioner requires the full omega window")
        if omega_window.ndim < 2:
            raise ValueError(f"expected [..., K, D], got shape={tuple(omega_window.shape)}")
        is_train = self.training if train is None else bool(train)
        if generator is not None and not is_train:
            # A serve/eval path can never be handed a dropout generator: the intervention is a
            # training regularizer and an inference-time deletion would be a silent distribution shift.
            raise ValueError(
                "WSMGatedDeltaNetConditioner: history dropout is train-only; got generator with train=False"
            )
        drop_history = is_train and self.history_dropout > 0.0

        lead = tuple(omega_window.shape[:-2])
        k_len = int(omega_window.shape[-2])
        if k_len != self.window_len:
            raise ValueError(
                f"omega window K={k_len} != trained window_len={self.window_len}; the serve-side "
                "window must match the trained recipe"
            )

        x = omega_window
        q = _l2_normalize(self._split_heads(self.proj_q(x), lead, k_len))
        k = _l2_normalize(self._split_heads(self.proj_k(x), lead, k_len))
        v = self._split_heads(self.proj_v(x), lead, k_len)
        beta = torch.sigmoid(self.proj_beta(x))  # [..., K, H]
        decay_logit = self.proj_decay(x) + self.pos_decay_bias.to(x.dtype)
        gamma = torch.exp(-F.softplus(decay_logit))  # [..., K, H] in (0,1]

        keep = self._history_keep_mask(lead, k_len, x.device, generator) if drop_history else None

        # S: [*lead, H, head_dim(v), head_dim(k)]
        state = torch.zeros(*lead, self.num_heads, self.head_dim, self.head_dim, dtype=v.dtype, device=v.device)
        for i in range(k_len):
            k_i = k[..., i, :, :]  # [*lead, H, Dk]
            v_i = v[..., i, :, :]  # [*lead, H, Dv]
            beta_i = beta[..., i, :]  # [*lead, H]
            gamma_i = gamma[..., i, :]  # [*lead, H]
            decayed = gamma_i[..., None, None] * state
            pred = torch.einsum("...hvk,...hk->...hv", decayed, k_i)
            err = beta_i[..., None] * (v_i - pred)
            updated = decayed + err[..., :, None] * k_i[..., None, :]
            if keep is None:
                state = updated
            else:
                # keep False => the element is ABSENT: no decay, no delta update, state untouched.
                state = torch.where(keep[..., i, None, None, None], updated, state)

        read = torch.einsum("...hvk,...hk->...hv", state, q[..., -1, :, :])
        projected = self.proj_readout(read.reshape(*lead, self.num_heads * self.head_dim))
        gate = torch.tanh(self.alpha).to(projected.dtype)
        return gate * projected
