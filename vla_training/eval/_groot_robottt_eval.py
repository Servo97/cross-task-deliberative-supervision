"""Serve-side re-attach + weight restore + episode driving for the GR00T RoboTTT fast weights.

WHY THIS FILE EXISTS — the same reason `_groot_wsm_deltanet_eval.py` does. `from_pretrained`
BYPASSES a train-path module patch: `install_robottt_action_head` monkeypatches
`Gr00tN1d7Pipeline.setup`, which only the TRAIN path runs. A serve that loads the checkpoint with
`Gr00tPolicy` never executes it, so the action head comes back as the STOCK class, the trained
`action_head.robottt_fast.*` tensors land nowhere, and the server silently serves the BASELINE
policy under the arm's name. ATTACH first, THEN load.

GEOMETRY IS RECOVERED FROM THE CHECKPOINT, not from flags — every width except `action_horizon`
is readable off a parameter shape (see `robottt_geometry_from_state_dict`). `action_horizon` is not
in any parameter, so it comes from `robottt_seam_dims(processor, model_config)` — the SAME function
the train path used, reading the SAME objects (the checkpoint ships its own processor config), and
the two derivations are cross-checked against each other on `state_dim` / `action_dim`. This is the
§23/§24 lesson made structural: the canary died because a config width (64) disagreed with the data
width (132), and no serve-side flag can reintroduce that gap.

EPISODE DRIVING. The websocket wire has no reset endpoint, so `wsm_t == 0` is the only boundary
signal (PHASE-1 §10). `robottt_serve_reset` rebuilds W to the learned init W_0 at each episode
start; `robottt_serve_commit` performs the apply-then-commit at chunk granularity (07a D-1) using
the state the head actually saw and the raw padded action chunk it actually produced — the same
[H=40, A=132] padded block the fast weights consumed in training (§24's recorded caveat).

FINITE GUARDS ARE FAIL-LOUD. A fast weight going non-finite is the D-9 signature; `robottt_commit`
raises rather than continuing an episode whose recurrence has diverged.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

__all__ = [
    "robottt_geometry_from_state_dict",
    "attach_and_restore_robottt",
    "robottt_serve_reset",
    "robottt_serve_commit",
]

PREFIX = "action_head.robottt_fast."


def robottt_geometry_from_state_dict(state_dict, prefix: str = PREFIX) -> dict:
    """Recover the fast-weight geometry from the trained tensors. Raises if not a RoboTTT ckpt.

    Shapes -> widths (`robottt_fast_weights.RoboTTTFastWeights.__init__`):
      ``state_in.weight``  [d, state_dim]      -> token_dim d, state_dim
      ``action_in.weight`` [d, action_dim]     -> action_dim
      ``readout.weight``   [cond_dim, d]       -> cond_dim
      ``w0_w1``            [d, h]              -> fast_hidden h
      ``registers``        [N, d]              -> num_registers N
    `action_horizon` is deliberately absent: it never enters a parameter shape, so it must come from
    the processor (see the module docstring).
    """
    sub = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not sub:
        available = sorted({k.rsplit(".", 1)[0] for k in state_dict if "robottt" in k or "ttt" in k})[:5]
        raise RuntimeError(
            f"[robottt-eval] no '{prefix}*' tensors in the checkpoint — this is not a RoboTTT "
            f"finetune. Serving it as one would silently be the baseline policy under the arm's "
            f"name. (ttt-ish prefixes present: {available or 'none'})"
        )

    for required in ("state_in.weight", "action_in.weight", "readout.weight", "w0_w1", "w0_w2", "registers", "alpha"):
        if required not in sub:
            raise RuntimeError(f"[robottt-eval] checkpoint is missing '{prefix}{required}'")

    token_dim, state_dim = (int(v) for v in sub["state_in.weight"].shape)
    action_in_d, action_dim = (int(v) for v in sub["action_in.weight"].shape)
    cond_dim, readout_d = (int(v) for v in sub["readout.weight"].shape)
    num_registers, reg_d = (int(v) for v in sub["registers"].shape)
    w1_d, fast_hidden = (int(v) for v in sub["w0_w1"].shape)

    for name, value in (("action_in", action_in_d), ("readout", readout_d), ("registers", reg_d), ("w0_w1", w1_d)):
        if value != token_dim:
            raise RuntimeError(
                f"[robottt-eval] '{name}' token axis {value} disagrees with state_in's "
                f"token_dim {token_dim}; the checkpoint is internally inconsistent"
            )
    if int(sub["alpha"].shape[0]) != cond_dim:
        raise RuntimeError(f"[robottt-eval] alpha width {int(sub['alpha'].shape[0])} != cond_dim {cond_dim}")
    if int(sub["w0_w2"].shape[0]) != fast_hidden:
        raise RuntimeError("[robottt-eval] w0_w2 hidden axis disagrees with w0_w1")

    return {
        "token_dim": token_dim,
        "fast_hidden": fast_hidden,
        "num_registers": num_registers,
        "cond_dim": cond_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_tensors": len(sub),
    }


def assert_finite_robottt(module, where: str = "after restore") -> None:
    """Fail loud on a non-finite fast-weight parameter (the GR00T Eval2 NaN-encoder contract)."""
    import torch

    bad = [name for name, tensor in module.state_dict().items() if not torch.isfinite(tensor).all()]
    if bad:
        raise RuntimeError(
            f"[robottt-eval] non-finite fast-weight tensors {where}: {bad}. Refusing to serve — a "
            "NaN W poisons temb and yields a scoreable but meaningless rollout."
        )


def attach_and_restore_robottt(policy, finetune_ckpt: str | Path) -> dict:
    """ATTACH the fast weights to a `from_pretrained` policy, THEN load the trained tensors.

    `policy` is the `Gr00tPolicy` (not the bare model): the processor is required, because
    `action_horizon` is a property of the tensors the processor pads and of nothing else.
    Returns the merged geometry dict. Also installs the serve-time stash the commit needs.
    """

    from vla_training.eval._groot_wsm_deltanet_eval import load_checkpoint_state_dict
    from vla_training.train.train_base._groot_robottt_common import (
        attach_robottt,
        robottt_config_from_env,
        robottt_seam_dims,
    )

    state_dict = load_checkpoint_state_dict(finetune_ckpt)
    geometry = robottt_geometry_from_state_dict(state_dict)

    model = policy.model
    processor = getattr(policy, "processor", None)
    if processor is None:
        raise RuntimeError(
            "[robottt-eval] the Gr00tPolicy has no .processor; action_horizon cannot be derived "
            "and there is deliberately no fallback (see §23/§24 — a silent width fallback is what "
            "failed the canary)."
        )
    dims = robottt_seam_dims(processor, model.action_head.config)

    # CROSS-CHECK the two independent derivations. The checkpoint tensors say what the mechanism was
    # TRAINED at; the processor says what this serve will FEED it. If they disagree the arm is
    # already inconsistent, and picking one would be a guess.
    for name in ("state_dim", "action_dim"):
        if int(dims[name]) != int(geometry[name]):
            raise RuntimeError(
                f"[robottt-eval] {name}: checkpoint tensors say {geometry[name]} but this "
                f"checkpoint's processor will feed {dims[name]}. {dims['dims_source']}. Refusing "
                "to serve — this is the 2026-08-08 canary failure mode with the numbers swapped."
            )

    cond_dim = int(model.action_head.model.proj_out_2.in_features)
    if cond_dim != int(geometry["cond_dim"]):
        raise RuntimeError(
            f"[robottt-eval] readout cond_dim {geometry['cond_dim']} != this model's DiT inner "
            f"width {cond_dim}; the conditioning vector would not be addable to temb"
        )

    cfg = robottt_config_from_env(
        cond_dim=cond_dim,
        state_dim=dims["state_dim"],
        action_dim=dims["action_dim"],
        action_horizon=dims["action_horizon"],
        dims_source=f"serve: {dims['dims_source']}",
    )
    # Widths that DO live in the checkpoint override the env defaults, so a stale WSM_TTT_* in the
    # box environment cannot make the attached module a different shape than the trained one.
    #
    # dataclasses.replace, NOT attribute assignment: RoboTTTConfig is a FROZEN dataclass, so the
    # three assignments this replaced raised FrozenInstanceError on the first real serve
    # (2026-08-08 ttt smoke, at `cfg.token_dim = ...`). Nothing caught it earlier because the CPU
    # checks built their configs through the constructor and never took this path -- which is why
    # the test added alongside this fix drives the REAL entry point on a staged checkpoint instead.
    cfg = dataclasses.replace(
        cfg,
        token_dim=int(geometry["token_dim"]),
        fast_hidden=int(geometry["fast_hidden"]),
        num_registers=int(geometry["num_registers"]),
    )
    for name in ("token_dim", "fast_hidden", "num_registers"):
        if int(getattr(cfg, name)) != int(geometry[name]):
            raise RuntimeError(
                f"[robottt-eval] {name} did not take on the rebuilt config "
                f"({getattr(cfg, name)} != {geometry[name]}); refusing to attach a module whose "
                f"shape does not match the trained one."
            )

    fast = attach_robottt(model, cfg=cfg, log_every=0)

    sub = {k[len(PREFIX) :]: v for k, v in state_dict.items() if k.startswith(PREFIX)}
    target = fast.state_dict()
    typed = {k: v.to(device=target[k].device, dtype=target[k].dtype) for k, v in sub.items() if k in target}
    # STRICT: a partial restore is exactly the failure this module exists to prevent.
    missing_keys, unexpected_keys = fast.load_state_dict(typed, strict=True)
    assert not missing_keys and not unexpected_keys
    assert_finite_robottt(fast)

    _install_serve_stash(model.action_head)

    geometry = dict(geometry)
    geometry.update(
        {"action_horizon": int(dims["action_horizon"]), "state_history_length": int(dims["state_history_length"])}
    )
    print(
        f"[robottt-eval] re-attached + restored {len(typed)} fast-weight tensors from "
        f"{finetune_ckpt} | {geometry} (from_pretrained bypasses the train-path patch, so this "
        f"step is what keeps the arm from serving as the baseline)",
        flush=True,
    )
    return geometry


def _install_serve_stash(action_head) -> None:
    """Record the (state, raw action chunk) the head actually used, for the post-chunk commit.

    The commit's inputs must be the TENSORS THE HEAD SAW, not a reconstruction from the wire reply:
    the reply is the unpadded 12-dim RoboCasa action, while the fast weights consume the padded
    [H, A] block (§24's recorded caveat). Reconstructing it would be a second, silently different
    definition of the mechanism's input.

    STASHED AS NUMPY FLOAT32, deliberately. `Gr00tPolicy.get_action` runs its forward inside
    `torch.inference_mode()` (gr00t_policy.py:407), so everything it produces is an INFERENCE
    TENSOR, and an inference tensor cannot participate in the autograd graph the commit builds —
    the commit *is* a gradient step (07a D-1/D-9), which is why `RoboTTTFastWeights.commit` forces
    `enable_grad()` internally. Materialising through numpy severs that provenance unconditionally,
    at the cost of one ~5 K-float device round-trip per replan. float32 loses nothing the mechanism
    uses: `robottt_fast` is fp32 and the TRAIN path also feeds it `action_seq.to(torch.float32)`.
    """
    import types

    inner = action_head.get_action_with_features

    def wrapped(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options=None,
        _inner=inner,
    ):
        out = _inner(backbone_features, state_features, embodiment_id, backbone_output, action_input, options)
        self._ttt_serve_state = action_input["state"].detach().float().cpu().numpy()
        self._ttt_serve_action = out["action_pred"].detach().float().cpu().numpy()
        return out

    action_head.get_action_with_features = types.MethodType(wrapped, action_head)
    action_head._ttt_serve_state = None
    action_head._ttt_serve_action = None


def robottt_serve_reset(action_head, batch: int = 1) -> None:
    """Episode boundary: W <- the learned init W_0. Call at `wsm_t == 0` and nowhere else."""
    from vla_training.train.train_base._groot_robottt_common import robottt_reset

    robottt_reset(action_head, batch)
    action_head._ttt_serve_state = None
    action_head._ttt_serve_action = None


def robottt_serve_commit(action_head) -> None:
    """Apply-then-commit (07a D-1): fold the chunk just produced into W. Once per inference."""
    import torch

    from vla_training.train.train_base._groot_robottt_common import robottt_commit

    state = getattr(action_head, "_ttt_serve_state", None)
    actions = getattr(action_head, "_ttt_serve_action", None)
    if state is None or actions is None:
        raise RuntimeError(
            "[robottt-eval] commit with no stashed chunk — the action head's serve path did not "
            "run. Either the mechanism was not attached (the arm is serving as the BASELINE) or "
            "_install_serve_stash did not take."
        )
    device = action_head.robottt_fast.readout.weight.device
    robottt_commit(action_head, torch.from_numpy(state).to(device), torch.from_numpy(actions).to(device))
    action_head._ttt_serve_state = None
    action_head._ttt_serve_action = None
