"""RoboTTT fast weights (reduced form) for GR00T N1.7 — torch port of the pi0.5/JAX module
`openpi/models/robottt_fast_weights.py`.

RoboTTT (arXiv:2607.15275) makes the policy's recurrent state a set of *fast weights* `W` — the
parameters of a small fast model `f_W` updated by gradient descent during BOTH training and
inference (Eqs 1-2). The slow/meta params (learned init `W_0`, projections theta_Q/K/V, register
tokens, tanh gate alpha, learnable inner LR) are optimized by the outer task loss, so `W_0` is
meta-learned through gradients-of-gradients (MAML-style).

REDUCED FORM, not "vanilla". Injection is ONE gated vector added to the GR00T DiT's `temb` (the
global AdaLN bus), matching what the pi arm does at `adarms_cond`, rather than the paper's per-DiT-
layer TTT. This is deliberate: a matched reduction on both backbones isolates the backbone as the
only difference, which is what the sync validation tests. Omissions and rationale are tabulated in
`internal_planning_and_todos/aug_07/groot_sync_validation.md`; the equation->field map is
`internal_planning_and_todos/_archive/handover_to_opus/07a_robottt_equation_map.md`.

Hard separation: `W`/`W_t` here are RoboTTT fast weights ONLY. `omega_t` (the workspace token) is a
DIFFERENT thing handled by `wsm_gated_deltanet.py`; this module never touches it.

The fast-weight STATE `W` is PER-SAMPLE (a dict of tensors with a leading batch axis), threaded
through the loss/serve functions — never global mutable module state.

TWO TORCH-SPECIFIC GOTCHAS, both load-bearing:

1. **GeLU flavor.** `jax.nn.gelu` defaults to the TANH APPROXIMATION; `torch.nn.functional.gelu`
   defaults to the EXACT erf form. They differ by ~1e-3 absolute, which is far above fp tolerance and
   would silently make the torch arm a different model. This module uses `approximate="tanh"`
   everywhere to match the JAX original.
2. **`torch.no_grad` breaks the inner update.** The commit is a gradient step, so it needs autograd
   even at serve time, where the surrounding code runs under `no_grad`. `commit()` therefore forces
   `torch.enable_grad()` internally and detaches its output when not training. Wrapping this module's
   commit in `no_grad` and expecting it to work is the obvious mistake; it would raise rather than
   silently no-op, but the guard is here so it never gets that far.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

# Fast-weight state: the fast model's per-sample parameters, each with a leading batch axis.
FastWeights = dict[str, torch.Tensor]

_W_KEYS = ("w1", "b1", "w2", "b2")


def _gelu(x: torch.Tensor) -> torch.Tensor:
    """GeLU matching `jax.nn.gelu`'s DEFAULT (tanh approximation) — see gotcha 1 in the module doc."""
    return F.gelu(x, approximate="tanh")


@dataclasses.dataclass(frozen=True)
class RoboTTTConfig:
    """RoboTTT hyperparameters (GR00T N1.7 instantiation).

    `cond_dim` is the DiT inner width (1536 on the released N1.7 checkpoint, vs pi's 1024).

    THE THREE DATA-DETERMINED WIDTHS HAVE NO DEFAULTS, deliberately. `state_dim`, `action_dim` and
    `action_horizon` are properties of the tensors the *collator* hands the head, not tuning knobs,
    and they must be read from the objects that produce those tensors
    (`_groot_robottt_common.robottt_seam_dims`). The 2026-08-08 TTT canary died on
    `mat1 and mat2 shapes cannot be multiplied (2x132 and 64x256)` precisely because this dataclass
    used to carry pi-lineage defaults (64/32/16) that the GR00T install path silently inherited:
    every unit test synthesised its tensors AT those defaults, so nothing disagreed until the real
    processor fed a 132-wide state. A sentinel that raises is the only version of this field that
    cannot repeat that. `dims_source` records WHERE the three came from and is quoted verbatim in
    the shape-mismatch error, so a future failure names its own provenance.
    """

    fast_weights: bool = True  # online TTT update ON (off => exact reduction to the base policy)

    # --- fast model / TTT geometry ---
    token_dim: int = 256  # d: TTT token width (deviation D-3; a canary axis)
    fast_hidden: int = 128  # h: 2-layer MLP hidden width (App A.1: 2-layer MLP, GeLU)
    num_registers: int = 16  # N: learned register tokens (paper N=16)
    cond_dim: int = 1536  # C: GR00T DiT inner width; RoboTTT output rides temb
    # --- data-determined widths: NO DEFAULTS (see the class docstring) ---
    state_dim: int = -1  # proprio token input width == state_history_length * max_state_dim
    action_dim: int = -1  # A == the collator's max_action_dim
    action_horizon: int = -1  # H: action tokens per chunk-step == the processor's max_action_horizon

    # --- inner update (Eq 1) ---
    base_inner_lr: float = 0.1  # App A.1 constant base LR; eta = base * softplus(learned)
    learn_inner_lr: bool = True

    # --- gating (Eq 3) ---
    gate_init: float = 1e-3  # alpha init near zero: TTT contribution small at start
    alpha_scale: float = 1.0  # extra multiplier on O_t; alpha_scale=0 => exact parity with base

    # --- sequence training recipe (Eq 4, Fig 4) ---
    window_len: int = 8  # L: chunk-steps per contiguous-episode training window
    tbptt_segment: int = 8  # G: grad truncated every G steps; G>=L => full BPTT over the window

    #: Free-text provenance of state_dim/action_dim/action_horizon, quoted in every dim error.
    dims_source: str = "<unset>"

    def __post_init__(self) -> None:
        missing = [name for name in ("state_dim", "action_dim", "action_horizon") if int(getattr(self, name)) < 1]
        if missing:
            raise ValueError(
                f"[robottt] RoboTTTConfig is missing the data-determined width(s) {missing}. These "
                "have no defaults on purpose: they are the widths the GR00T collator feeds, so they "
                "must be derived from the live processor + model config via "
                "`_groot_robottt_common.robottt_seam_dims(processor, model_config)` and passed in. "
                "A default here is what let the 64-vs-132 canary failure through."
            )


class RoboTTTFastWeights(nn.Module):
    """Slow/meta params of one RoboTTT (TTT-KVB) layer + the fast-weight update/apply.

    Holds ONLY the meta-learned parameters (checkpoint subtree ``robottt_fast``). The fast-weight
    state `W` is created by ``init_state`` and threaded externally; it is never stored here.
    """

    def __init__(self, cfg: RoboTTTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h = cfg.token_dim, cfg.fast_hidden
        # KVB projections theta_Q/K/V (learned by the outer loss, Sec 2).
        self.proj_q = nn.Linear(d, d)
        self.proj_k = nn.Linear(d, d)
        self.proj_v = nn.Linear(d, d)
        # Learned fast-weight initialization W_0 (Sec 3.1). Zero-init the SECOND layer so
        # f_{W_0} == 0 exactly: the TTT contribution starts at zero, giving clean off/on parity.
        self.w0_w1 = nn.Parameter(torch.randn(d, h) / math.sqrt(d))
        self.w0_b1 = nn.Parameter(torch.zeros(h))
        self.w0_w2 = nn.Parameter(torch.zeros(h, d))
        self.w0_b2 = nn.Parameter(torch.zeros(d))
        # Learned register tokens R (N x d), prepended each timestep to carry info across time.
        self.registers = nn.Parameter(torch.randn(cfg.num_registers, d) / math.sqrt(d))
        # Input encoders for the proprio and action tokens fed to the TTT layer.
        self.state_in = nn.Linear(cfg.state_dim, d)
        self.action_in = nn.Linear(cfg.action_dim, d)
        # Readout O_TTT -> additive temb vector; zero-init => O_t == 0 at init regardless of the
        # (also zero) fast model, so RoboTTT rides on top of the base policy.
        self.readout = nn.Linear(d, cfg.cond_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        # tanh gate alpha (Eq 3), init near zero.
        self.alpha = nn.Parameter(torch.full((cfg.cond_dim,), cfg.gate_init))
        # Learnable inner LR: eta = base * softplus(log_inner_lr); init so eta == base at start.
        init_raw = math.log(math.e - 1.0)  # softplus(init_raw) == 1.0
        self.log_inner_lr = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    # ---- meta params ------------------------------------------------------------------------
    def inner_lr(self) -> torch.Tensor:
        raw = self.log_inner_lr
        scale = F.softplus(raw) if self.cfg.learn_inner_lr else torch.ones((), device=raw.device)
        return self.cfg.base_inner_lr * scale

    def init_state(self, batch: int) -> FastWeights:
        """Broadcast the learned init W_0 to a per-sample state [batch, ...] (episode boundary).

        `expand` keeps W_0 in the autograd graph, which is what makes it META-LEARNED: the outer loss
        reaches it through every commit. `.contiguous()` because the inner update writes new tensors
        and expanded views would otherwise alias.
        """
        return {
            "w1": self.w0_w1.unsqueeze(0).expand(batch, *self.w0_w1.shape).contiguous(),
            "b1": self.w0_b1.unsqueeze(0).expand(batch, *self.w0_b1.shape).contiguous(),
            "w2": self.w0_w2.unsqueeze(0).expand(batch, *self.w0_w2.shape).contiguous(),
            "b2": self.w0_b2.unsqueeze(0).expand(batch, *self.w0_b2.shape).contiguous(),
        }

    # ---- pure batched fast model + inner update ----------------------------------------------
    @staticmethod
    def _fast_forward(w: FastWeights, x: torch.Tensor) -> torch.Tensor:
        """f_W(x): 2-layer MLP, GeLU (App A.1). x: [B, m, d] -> [B, m, d], W per-sample."""
        z = _gelu(torch.baddbmm(w["b1"].unsqueeze(1), x, w["w1"]))
        return torch.baddbmm(w["b2"].unsqueeze(1), z, w["w2"])

    @classmethod
    def _inner_loss(cls, w: FastWeights, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """L_FW = mean ||f_W(K) - V||^2 (Eq 1), MEAN-normalized over tokens AND features. -> [B]

        The feature-axis normalization is load-bearing, not cosmetic: with a per-token SUM the
        inner-GD curvature scales with d, and at production geometry (d=256, h=128) eta=0.1 sits
        ~10x past the 2/lambda stability boundary — W diverges to NaN within 5 commits (verified on
        the pi side 2026-07-23; the toy test dims d=16/h=8 were stable, which is why unit tests alone
        missed it). A constant factor on L_FW is equivalent to rescaling eta, so this keeps Eq 1
        intact while making the (eta x curvature) product d-independent — required because d is an
        explicit canary axis (07a D-3).
        """
        pred = cls._fast_forward(w, k)
        return ((pred - v) ** 2).mean(dim=(-2, -1))  # [B]

    def commit(self, w: FastWeights, state: torch.Tensor, actions: torch.Tensor) -> FastWeights:
        """One conditional commit W_t from the FINALIZED/executed chunk (Eq 1). Batched, functional.

        K/V come from registers + proprio + executed action tokens (the just-observed transition).
        At serve time this runs exactly ONCE per executed chunk (deviation D-1).

        Because sample i's inner loss depends only on w[i], the gradient of the SUMMED loss w.r.t.
        the batched leaf IS the stack of per-sample gradients — this is the batched equivalent of the
        JAX original's `vmap(grad(...))`, not an approximation of it.
        """
        tokens = self._tokens_from(state, actions)  # [B, N+1+H, d]
        k = self.proj_k(tokens)
        v = self.proj_v(tokens)
        eta = self.inner_lr()

        # enable_grad: the inner update is a gradient step, so it must work even under the no_grad
        # that wraps the serve path (gotcha 2 in the module doc).
        with torch.enable_grad():
            leaves = [w[name] if w[name].requires_grad else w[name].requires_grad_(True) for name in _W_KEYS]
            wd = dict(zip(_W_KEYS, leaves))
            per_sample = self._inner_loss(wd, k, v)  # [B]
            grads = torch.autograd.grad(
                per_sample.sum(),
                leaves,
                create_graph=self.training,  # MAML meta-gradient only needed while training
            )
        out = {name: wd[name] - eta * g for name, g in zip(_W_KEYS, grads)}
        if not self.training:
            out = {name: t.detach() for name, t in out.items()}
        return out

    def assert_input_dims(self, state: torch.Tensor, actions: torch.Tensor | None, where: str = "") -> None:
        """Fail LOUD if the incoming tensors disagree with the cfg the parameters were built from.

        Without this the disagreement surfaces as `mat1 and mat2 shapes cannot be multiplied
        (2x132 and 64x256)` from inside `nn.Linear` — a message that names neither field, neither
        side's provenance, nor the fact that a *config* is wrong rather than a tensor. Every message
        below names BOTH numbers and where the cfg dims came from.
        """
        at = f" at {where}" if where else ""
        if state.ndim < 2 or int(state.shape[-1]) != int(self.cfg.state_dim):
            raise ValueError(
                f"[robottt] state width {tuple(state.shape)}[-1]={int(state.shape[-1])} != "
                f"cfg.state_dim={int(self.cfg.state_dim)}{at}. cfg dims came from: "
                f"{self.cfg.dims_source}. The fast-weight parameters were BUILT at the cfg width, "
                "so this cannot be fixed downstream — the derivation is wrong, not the batch."
            )
        if actions is None:
            return
        if actions.ndim < 3:
            raise ValueError(f"[robottt] actions must be [B, H, A], got {tuple(actions.shape)}{at}")
        if int(actions.shape[-1]) != int(self.cfg.action_dim):
            raise ValueError(
                f"[robottt] action width {tuple(actions.shape)}[-1]={int(actions.shape[-1])} != "
                f"cfg.action_dim={int(self.cfg.action_dim)}{at}. cfg dims came from: "
                f"{self.cfg.dims_source}."
            )
        if int(actions.shape[-2]) != int(self.cfg.action_horizon):
            raise ValueError(
                f"[robottt] action horizon {tuple(actions.shape)}[-2]={int(actions.shape[-2])} != "
                f"cfg.action_horizon={int(self.cfg.action_horizon)}{at}. cfg dims came from: "
                f"{self.cfg.dims_source}. H is the processor's max_action_horizon (the PADDED "
                "chunk length), not the modality config's number of real action steps."
            )

    def _tokens_from(self, state: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        """Per-chunk-step token bundle [B, m, d] = registers (+ proprio) (+ action tokens).

        Query bundle (actions=None): registers + proprio  -> m = N+1  (readout; no future action).
        Update bundle (actions given): registers + proprio + H action tokens -> m = N+1+H.
        """
        self.assert_input_dims(state, actions, where="RoboTTTFastWeights._tokens_from")
        b = state.shape[0]
        reg = self.registers.unsqueeze(0).expand(b, self.cfg.num_registers, self.cfg.token_dim)
        proprio = self.state_in(state).unsqueeze(1)  # [B, 1, d]
        toks = [reg, proprio]
        if actions is not None:
            toks.append(self.action_in(actions))  # [B, H, d]
        return torch.cat(toks, dim=1)

    def condition(self, w: FastWeights, state: torch.Tensor) -> torch.Tensor:
        """O_t = tanh(alpha) * readout(pool(f_W(Q_t))), the additive temb vector [B, C].

        Reads the ENTERING fast weights `w`, held fixed through every Euler pass at serve time
        (deviation D-1). Uses only registers + proprio — no unknown future action.
        """
        tokens = self._tokens_from(state, None)  # [B, N+1, d]
        q = self.proj_q(tokens)
        pooled = self._fast_forward(w, q).mean(dim=1)  # [B, d]  (Eq 2 pooling)
        projected = self.readout(pooled)  # [B, C]
        gate = torch.tanh(self.alpha) * self.cfg.alpha_scale
        return gate.to(projected.dtype) * projected

    # ---- scan-friendly sequence roll (tests / offline analysis) -------------------------------
    def run_sequence(
        self, state_seq: torch.Tensor, action_seq: torch.Tensor, *, tbptt_segment: int | None = None
    ) -> tuple[torch.Tensor, FastWeights]:
        """Roll the fast weights over L chunk-steps, TBPTT-truncated (Fig 4).

        Returns (O_seq [L, B, C], W_final). Apply-then-commit per step (deviation D-1). Gradients are
        cut at every `tbptt_segment` boundary by detaching the carried W; W_0 still receives gradient
        through the first segment. The forward VALUES are identical for any segment length — TBPTT
        affects only gradients, which is the basis of the equality test.
        """
        g = int(tbptt_segment if tbptt_segment is not None else self.cfg.tbptt_segment)
        if g < 1:
            raise ValueError(f"tbptt_segment must be >= 1, got {g}")
        b, length = state_seq.shape[0], state_seq.shape[1]
        w = self.init_state(b)
        outs = []
        for t in range(length):
            if t > 0 and (t % g) == 0:
                w = {name: leaf.detach() for name, leaf in w.items()}
            o_t = self.condition(w, state_seq[:, t])  # entering W conditions step t
            w = self.commit(w, state_seq[:, t], action_seq[:, t])  # finalized chunk -> W_{t+1}
            outs.append(o_t)
        return torch.stack(outs, dim=0), w
