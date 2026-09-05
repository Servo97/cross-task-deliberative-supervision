"""RoboTTT fast-weights (reduced form) GR00T N1.7 wiring — dataset, collator, action head.

THREE least-invasive hooks, no edits to the vendored Isaac-GR00T repo:

  install_robottt_dataset(window_len, chunk_stride)
     `_groot_seq_common.install_sequence_dataset` — the stock `ShardedSingleStepDataset` is
     single-step and actively SHUFFLES step indices within a shard split, so contiguity has to be
     rebuilt. One item == one non-overlapping contiguous window of L chunk-steps of ONE episode.

  install_robottt_collator()
     `Gr00tN1d7Processor.collator` -> `build_sequence_collator(...)`. `vlm_content` is the only key
     the stock collator cannot stack; the wrapper flattens the per-item LIST of L contents
     WINDOW-MAJOR so the VLM tensors arrive at batch `B*L` like everything else.

  install_robottt_action_head(cfg)
     `Gr00tN1d7Pipeline.setup` -> reclass `model.action_head` to `RoboTTTActionHead`, patch the DiT
     timestep encoder so `temb` takes an additive stash (the SAME seam the deltanet arm uses), and
     attach a `RoboTTTFastWeights` built at the widths the COLLATOR actually feeds.

WHY THE WIDTHS ARE DERIVED AND NOT CONFIGURED (2026-08-08). The first TTT canary died at step 0 on
`mat1 and mat2 shapes cannot be multiplied (2x132 and 64x256)`: the fast weights had been built from
`RoboTTTConfig`'s pi-lineage defaults (state_dim 64) while the real GR00T processor pads proprio to
`max_state_dim = 132`. `state_dim`/`action_dim`/`action_horizon` are NOT hyperparameters — they are
the shapes of the collator's tensors — so `robottt_seam_dims` now reads them off the two objects
that produce those tensors (the live `Gr00tN1d7Processor` and the loaded model's config) and
cross-checks them against each other. Measured on the real offline processor + collator:

    per step        state [1, 132]            action [40, 132]      (both float32)
    at the seam     state_seq [B, L, 132]     action_seq [B, L, 40, 132]

i.e. state_dim = state_history_length(1) * max_state_dim(132); action_dim = max_action_dim(132);
H = max_action_horizon(40), which `setup.py` sets from `model_config.action_horizon` and which is
the PADDED chunk length — the RoboCasa modality config's 16 real steps x 12 real channels sit in the
top-left corner of that [40, 132] block, the rest zero (`action_mask` marks the real region).

THE ONE THING THAT MUST NOT BE GOT WRONG. The collator hands the head `[B, L, ...]` and the backbone
output at `B*L`. `run_sequence` returns `[L, B, C]`. Flattening `[B, L, ...]` with a row-major
`reshape(B*L, ...)` gives flat index `b*L + t` — WINDOW-major — which is exactly the order
`build_sequence_collator` used for the VLM keys, so `backbone_output[b*L + t]` is the observation of
window `b` at step `t`. The conditioning therefore has to be `O.permute(1, 0, 2).reshape(B*L, C)`,
NOT `O.reshape(B*L, C)`. The wrong one pairs every observation with another timestep's fast-weight
state: the shapes are identical, the loss still falls, and the policy is meaningless.
`tests/test_groot_robottt_wiring.py` pins the pairing element by element.

APPLY-THEN-COMMIT at chunk granularity (07a D-1): step t is conditioned by the ENTERING `W_t`, and
only afterwards does the finalized chunk produce `W_{t+1}`. `run_sequence` already enforces this;
the serve path re-implements the same order explicitly because it has no sequence to roll.

GR00T-venv-only: every gr00t/torch import is inside a function or a lazily-built class.
"""

from __future__ import annotations

__all__ = [
    "install_robottt_dataset",
    "install_robottt_collator",
    "install_robottt_action_head",
    "attach_robottt",
    "robottt_seam_dims",
    "robottt_config_from_env",
    "flatten_window_batch",
    "sequence_conditioning",
    "robottt_reset",
    "robottt_commit",
]

#: Keys the sequence dataset adds that the action head consumes itself and must NOT forward to the
#: vendored head (which would either stack-mismatch or be silently ignored).
_SEQ_ONLY = ("seq_len", "seq_window_len", "loss_mask", "reset")

#: Keys that carry a window axis and must be flattened [B, L, ...] -> [B*L, ...] before the
#: vendored head sees them. Listed EXPLICITLY rather than inferred from shape: `state` is
#: [B, L, history, dim] and `embodiment_id` is [B, L], so a shape heuristic would have to guess, and
#: guessing wrong here is the silent-transposition failure this module is most exposed to.
_WINDOWED = ("state", "action", "action_mask", "embodiment_id")


#: Names read off the live processor -> the widths it pads its own tensors to.
_PROC_FIELDS = ("max_state_dim", "max_action_dim", "max_action_horizon")
#: The model-config names that MUST agree with them (the same `model_config` builds both, but on the
#: `start_from_checkpoint` path the processor is rebuilt from the CHECKPOINT's processor_config.json
#: while the model config comes from the checkpoint's config.json — two files that can drift apart).
_MODEL_FIELDS = ("max_state_dim", "max_action_dim", "action_horizon")


def _int_attr(obj, name: str, what: str) -> int:
    if not hasattr(obj, name):
        raise AttributeError(
            f"[robottt] {what} has no '{name}'. The RoboTTT widths are READ from the objects that "
            "produce the collator's tensors; there is deliberately no fallback, because a silent "
            "fallback (state_dim=64 against a 132-wide state) is exactly what failed the "
            f"2026-08-08 canary. Got a {type(obj).__name__}: {sorted(vars(obj))[:12]}..."
        )
    return int(getattr(obj, name))


def robottt_seam_dims(processor, model_config) -> dict:
    """The widths the GR00T collator will actually feed, read from processor + model config.

    Returns ``{"state_dim", "action_dim", "action_horizon", "state_history_length",
    "dims_source"}``. Duck-typed on purpose (no gr00t import): the tests drive it with the metadata
    recorded from the real objects, so the test dims and the production dims come out of the SAME
    function and cannot diverge.

    Derivation, each number traced to the line that creates it:
      * state    `processing_gr00t_n1d7.py` pads proprio to `max_state_dim` and stacks the state
                 modality's `delta_indices`; `gr00t_n1d7.py:303` then asserts that leading axis
                 equals `state_history_length`. The head flattens both -> `T * max_state_dim`.
      * action   padded to `max_action_dim` on the feature axis and to `max_action_horizon` on the
                 time axis, so the seam sees [H, A] = [max_action_horizon, max_action_dim].
      * H        `setup.py:197` passes `model_config.action_horizon` in as `max_action_horizon`.
    """
    proc = {name: _int_attr(processor, name, "the GR00T processor") for name in _PROC_FIELDS}
    model = {name: _int_attr(model_config, name, "the GR00T model config") for name in _MODEL_FIELDS}
    hist = _int_attr(model_config, "state_history_length", "the GR00T model config")

    for proc_name, model_name in zip(_PROC_FIELDS, _MODEL_FIELDS):
        if proc[proc_name] != model[model_name]:
            raise ValueError(
                f"[robottt] processor.{proc_name}={proc[proc_name]} disagrees with "
                f"model_config.{model_name}={model[model_name]}. The processor decides the tensor "
                "the head receives and the model config decides the weights it was loaded into, so "
                "this run is already inconsistent independently of RoboTTT. Refusing to pick one."
            )

    # The processor's own state modality config must agree with state_history_length, or the
    # vendored head's own assert (gr00t_n1d7.py:303) will fire later on a batch we already sized for.
    for tag, cfgs in (getattr(processor, "modality_configs", None) or {}).items():
        state_cfg = cfgs.get("state") if hasattr(cfgs, "get") else None
        indices = getattr(state_cfg, "delta_indices", None)
        if indices is not None and len(indices) != hist:
            raise ValueError(
                f"[robottt] embodiment {tag!r} state modality has {len(indices)} delta_indices but "
                f"model_config.state_history_length={hist}; the state tensor's leading axis is "
                "ambiguous, so the proprio token width would be a guess."
            )

    if hist < 1:
        raise ValueError(f"[robottt] state_history_length must be >= 1, got {hist}")

    return {
        "state_dim": hist * proc["max_state_dim"],
        "action_dim": proc["max_action_dim"],
        "action_horizon": proc["max_action_horizon"],
        "state_history_length": hist,
        "dims_source": (
            f"robottt_seam_dims: state_dim = state_history_length({hist}) * "
            f"processor.max_state_dim({proc['max_state_dim']}); "
            f"action_dim = processor.max_action_dim({proc['max_action_dim']}); "
            f"action_horizon = processor.max_action_horizon({proc['max_action_horizon']}) "
            f"[= model_config.action_horizon]"
        ),
    }


def robottt_config_from_env(
    cond_dim: int, state_dim: int, action_dim: int, action_horizon: int, dims_source: str = "caller-supplied"
):
    """Build a RoboTTTConfig from WSM_TTT_* env, with the DATA-determined widths passed in.

    `WSM_TTT_STATE_DIM` / `WSM_TTT_ACTION_DIM` / `WSM_TTT_ACTION_HORIZON` exist ONLY as an escape
    hatch: unset (the normal case, and what all three ttt yamls do) the derived widths are used
    verbatim. An override that actually changes a width is recorded in `dims_source`, so the
    shape-mismatch error it eventually causes names the override rather than the derivation.
    """
    import os

    from workspace_models.networks.robottt_fast_weights import RoboTTTConfig

    def _i(name, default):
        return int(os.environ.get(name, default))

    def _f(name, default):
        return float(os.environ.get(name, default))

    derived = {"state_dim": int(state_dim), "action_dim": int(action_dim), "action_horizon": int(action_horizon)}
    dims = {name: _i(f"WSM_TTT_{name.upper()}", value) for name, value in derived.items()}
    overridden = {k: (derived[k], v) for k, v in dims.items() if v != derived[k]}
    if overridden:
        dims_source = f"{dims_source} + WSM_TTT_* OVERRIDES {overridden}"

    window_len = _i("WSM_SEQ_WINDOW_LEN", 8)
    return RoboTTTConfig(
        fast_weights=os.environ.get("WSM_TTT_FAST_WEIGHTS", "1") == "1",
        token_dim=_i("WSM_TTT_TOKEN_DIM", 256),
        fast_hidden=_i("WSM_TTT_FAST_HIDDEN", 128),
        num_registers=_i("WSM_TTT_NUM_REGISTERS", 16),
        cond_dim=cond_dim,
        **dims,
        base_inner_lr=_f("WSM_TTT_INNER_LR", 0.1),
        learn_inner_lr=os.environ.get("WSM_TTT_LEARN_INNER_LR", "1") == "1",
        gate_init=_f("WSM_TTT_GATE_INIT", 1e-3),
        alpha_scale=_f("WSM_TTT_ALPHA_SCALE", 1.0),
        window_len=window_len,
        # G >= L means full BPTT across the window, which is the default recipe; a smaller G is the
        # memory escape hatch and changes gradients only, never forward values.
        tbptt_segment=_i("WSM_TTT_TBPTT_SEGMENT", window_len),
        dims_source=dims_source,
    )


def install_robottt_dataset(window_len: int = 8, chunk_stride: int = 8) -> None:
    """Contiguous-window sequence dataset. Thin alias so the driver has one import surface."""
    from vla_training.train.train_base._groot_seq_common import install_sequence_dataset

    install_sequence_dataset(window_len=window_len, chunk_stride=chunk_stride)


def install_robottt_collator() -> None:
    """Patch `Gr00tN1d7Processor.collator` to the sequence-aware wrapper.

    That property is where `setup.py` picks up the collator for the HF Trainer, so it is the single
    chokepoint; the wrapper is INERT on a non-sequence batch (`vlm_content` a dict, not a list), so
    an eval/serve processor built from the same class behaves exactly as stock.
    """
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as proc_module

    from vla_training.train.train_base._groot_seq_common import build_sequence_collator

    cls = proc_module.Gr00tN1d7Processor
    if getattr(cls, "_robottt_patched", False):
        return

    def _collator(self):
        wrapped = getattr(self, "_robottt_collator", None)
        if wrapped is None or getattr(self, "_robottt_collator_src", None) is not self._collator:
            wrapped = build_sequence_collator(self._collator)
            self._robottt_collator = wrapped
            self._robottt_collator_src = self._collator
        return wrapped

    cls.collator = property(_collator)
    cls._robottt_patched = True
    print("[robottt] collator patched: window-major vlm_content flatten (B*L)", flush=True)


def flatten_window_batch(action_input, window_len: int):
    """[B, L, ...] -> [B*L, ...] row-major (flat index b*L + t) for the windowed keys.

    Returns (flat_dict, batch, length). Raises rather than guessing if a windowed key is missing its
    L axis — a batch that reached here without one is a collator regression, and letting it through
    would train on a silently misaligned pairing.
    """
    flat = {}
    batch = None
    for key, value in action_input.items():
        if key in _SEQ_ONLY:
            continue
        if key in _WINDOWED:
            if value.ndim < 2 or int(value.shape[1]) != window_len:
                raise ValueError(
                    f"[robottt] '{key}' has shape {tuple(value.shape)}; expected a window axis of "
                    f"length {window_len} at dim 1. The sequence collator did not run — refusing to "
                    "train on a misaligned batch."
                )
            batch = int(value.shape[0]) if batch is None else batch
            flat[key] = value.reshape(int(value.shape[0]) * window_len, *value.shape[2:])
        else:
            flat[key] = value
    if batch is None:
        raise ValueError(f"[robottt] no windowed key among {_WINDOWED} in the batch")
    return flat, batch, window_len


def sequence_conditioning(robottt, state_seq, action_seq):
    """Roll the fast weights over the window and return conditioning at flat index b*L + t.

    `run_sequence` yields [L, B, C]; `permute(1, 0, 2).reshape(B*L, C)` puts it in WINDOW-major
    order, matching the collator's flatten. See the module docstring for why the alternative is a
    silent catastrophe rather than a crash.
    """
    out_seq, w_final = robottt.run_sequence(state_seq, action_seq)  # [L, B, C]
    length, batch, cond = out_seq.shape
    return out_seq.permute(1, 0, 2).reshape(batch * length, cond), w_final


def attach_robottt(model, cfg, log_every: int = 50):
    """Reclass model.action_head, patch the DiT timestep encoder, attach RoboTTTFastWeights.

    `cfg` is REQUIRED and must already carry the derived widths — see `robottt_seam_dims`. It used
    to be optional, with `attach_robottt` guessing from `ah.state_encoder.max_state_dim` (an
    attribute `CategorySpecificMLP` does not have) and falling back to 64/32. Both guesses were
    silent, and the fallback is what shipped the failed canary.

    Returns the fast-weight module (the only net-new trainable subtree, `robottt_fast`; the
    projector + DiT train exactly as in the baseline finetune).
    """
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from transformers.feature_extraction_utils import BatchFeature

    from workspace_models.networks.robottt_fast_weights import RoboTTTFastWeights

    dit = model.action_head.model
    inner_dim = int(dit.proj_out_2.in_features)  # 1536 on the released N1.7 ckpt == temb width

    # Same additive `temb` seam the deltanet arm uses, so the two mechanisms differ in WHAT they
    # compute and not in WHERE it lands — which is what makes their comparison readable.
    te = dit.timestep_encoder
    if not getattr(te, "_wsm_patched", False):
        _orig_te_forward = te.forward

        def _patched_te_forward(timestep, _orig=_orig_te_forward, _mod=te):
            temb = _orig(timestep)
            cond = getattr(_mod, "_wsm_cond", None)
            return temb if cond is None else temb + cond.to(device=temb.device, dtype=temb.dtype)

        te.forward = _patched_te_forward
        te._wsm_cond = None
        te._wsm_patched = True

    ah = model.action_head
    if cfg is None:
        raise ValueError(
            "[robottt] attach_robottt requires an explicit cfg carrying the derived widths. Build "
            "it with robottt_config_from_env(**robottt_seam_dims(processor, model_config)); there "
            "is no derive-from-the-model-alone path, because the model config alone cannot see the "
            "processor's padding and the guess it used to make (64/32) failed the canary."
        )
    if int(cfg.cond_dim) != inner_dim:
        raise ValueError(
            f"[robottt] cond_dim {cfg.cond_dim} != DiT inner width {inner_dim}; the readout would "
            "not be addable to temb"
        )

    class RoboTTTActionHead(Gr00tN1d7ActionHead):
        def forward(self, backbone_output, action_input):
            assert getattr(self.config, "expand_batch", None) in (None, 1), (
                "[robottt] expand_batch unsupported (the conditioner batch would mismatch temb)"
            )
            length = action_input["seq_window_len"] if "seq_window_len" in action_input else None
            # FAIL-LOUD on the first train batch: without the sequence collator there is no window,
            # the recurrence cannot run, and this would train as plain baseline under the arm's name.
            if self.training and not self._ttt_checked:
                assert length is not None, (
                    "[robottt] NO 'seq_window_len' in the batch — the sequence collator did not "
                    "run, so there is no contiguous window to roll the fast weights over. This "
                    "would train as the BASELINE. Verify install_robottt_collator ran."
                )
                self._ttt_checked = True
            if length is None:
                return super().forward(backbone_output, action_input)

            length = int(length)
            flat, batch, length = flatten_window_batch(action_input, length)
            # The fast model reads proprio flat: state is [B, L, history, dim].
            state_seq = action_input["state"].reshape(batch, length, -1).to(torch.float32)
            action_seq = action_input["action"].to(torch.float32)
            # FIRST-FORWARD DIM GATE. The widths were derived from the processor before the
            # optimizer existed; this is the first moment a REAL collated batch can contradict that
            # derivation. Checking here (rather than letting nn.Linear say "2x132 and 64x256") is
            # what turns a 4-hour node-side mystery into a one-line verdict naming both numbers.
            self.robottt_fast.assert_input_dims(
                state_seq[:, 0], action_seq[:, 0], where="RoboTTTActionHead.forward (first collated train batch)"
            )
            cond, w_final = sequence_conditioning(self.robottt_fast, state_seq, action_seq)
            if not torch.isfinite(cond).all():
                raise RuntimeError(
                    "[robottt] non-finite conditioning from the fast-weight roll — refusing to "
                    "poison temb with NaN (D-9: a per-token SUM inner loss diverges at production "
                    "dims; see the module docstring)"
                )

            self.model.timestep_encoder._wsm_cond = cond
            try:
                out = super().forward(backbone_output, BatchFeature(data=flat))
            finally:
                self.model.timestep_encoder._wsm_cond = None

            if self.training and log_every and (self._ttt_step % log_every == 0):
                with torch.no_grad():
                    gate = torch.tanh(self.robottt_fast.alpha)
                    w2 = w_final["w2"]
                    print(
                        f"[robottt] step {self._ttt_step} flow {float(out['loss'].detach()):.4f} "
                        f"| cond |mu| {float(cond.abs().mean()):.5f} sd {float(cond.std()):.5f} "
                        f"| tanh(alpha) |mu| {float(gate.abs().mean()):.5f} "
                        f"| eta {float(self.robottt_fast.inner_lr()):.4f} "
                        f"| |W2| {float(w2.abs().mean()):.5f} "
                        f"| L={length} ACTIVE",
                        flush=True,
                    )
            self._ttt_step += 1
            return out

        # ---- SERVE ---------------------------------------------------------------------------
        @torch.no_grad()
        def get_action_with_features(
            self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None
        ):
            """Apply the ENTERING W (D-1); the commit happens separately, once per executed chunk.

            The websocket wire has no reset endpoint, so episode state is keyed externally and
            rebuilt by `robottt_reset`. Conditioning is held fixed across every Euler step — the
            readout is one additive vector with no per-step branching, so setting the stash around
            the vendored sampler is both sufficient and far less fragile than re-implementing it.
            """
            w = getattr(self, "_ttt_w", None)
            if w is None:
                raise RuntimeError(
                    "[robottt] serve path has no fast-weight state. Call robottt_reset(head, B) at "
                    "wsm_t == 0. Serving without it would silently be the baseline policy under "
                    "this checkpoint's name."
                )
            state = action_input["state"]
            cond = self.robottt_fast.condition(w, state.reshape(state.shape[0], -1).to(torch.float32))
            if not torch.isfinite(cond).all():
                raise RuntimeError("[robottt] non-finite serve conditioning — refusing to act")
            self.model.timestep_encoder._wsm_cond = cond
            try:
                return super().get_action_with_features(
                    backbone_features, state_features, embodiment_id, backbone_output, action_input, options
                )
            finally:
                self.model.timestep_encoder._wsm_cond = None

    ah.__class__ = RoboTTTActionHead
    ah._ttt_step = 0
    ah._ttt_checked = False
    ah._ttt_w = None

    ref = next((p for p in ah.parameters() if p.requires_grad), None)
    dev = ref.device if ref is not None else next(ah.parameters()).device
    dt = ref.dtype if ref is not None else next(ah.parameters()).dtype
    # fp32 for the fast weights: the inner update is a gradient step and bf16 loses the small
    # eta*grad increments that ARE the mechanism.
    ah.robottt_fast = RoboTTTFastWeights(cfg).to(device=dev, dtype=torch.float32)
    ah.robottt_fast.requires_grad_(True)
    print(
        f"[robottt] attached: cond_dim={cfg.cond_dim} d={cfg.token_dim} h={cfg.fast_hidden} "
        f"N={cfg.num_registers} state_dim={cfg.state_dim} action_dim={cfg.action_dim} "
        f"H={cfg.action_horizon} L={cfg.window_len} G={cfg.tbptt_segment} "
        f"eta={cfg.base_inner_lr} gate_init={cfg.gate_init} head_dtype={dt} "
        f"params={sum(p.numel() for p in ah.robottt_fast.parameters()) / 1e6:.2f}M",
        flush=True,
    )
    return ah.robottt_fast


def robottt_reset(action_head, batch: int) -> None:
    """Rebuild the serve-time fast weights to the learned init W_0 (episode boundary)."""
    action_head._ttt_w = action_head.robottt_fast.init_state(batch)


def robottt_commit(action_head, state, actions) -> None:
    """Commit the just-executed chunk: W_t -> W_{t+1}. Exactly once per executed chunk (D-1)."""
    import torch

    w = action_head._ttt_w
    if w is None:
        raise RuntimeError("[robottt] commit before reset — no fast-weight state")
    flat_state = state.reshape(state.shape[0], -1).to(torch.float32)
    action_head._ttt_w = action_head.robottt_fast.commit(w, flat_state, actions.to(torch.float32))
    for name, leaf in action_head._ttt_w.items():
        if not torch.isfinite(leaf).all():
            raise RuntimeError(
                f"[robottt] fast weight '{name}' went non-finite after a commit — the inner step "
                "is past the stability boundary (D-9). Refusing to continue the episode."
            )


def install_robottt_action_head(cfg=None) -> None:
    """TRAIN path: monkeypatch `Gr00tN1d7Pipeline.setup` to attach once the processor exists.

    WHY `setup` AND NOT `_create_model`. The fast weights' input widths are the widths the collator
    feeds, and those live on the PROCESSOR, which `_create_dataset` builds AFTER `_create_model`
    returns (`setup.py`: model -> dataset/processor -> collator). Attaching inside `_create_model`
    meant there was nothing to read the real widths from, so `RoboTTTConfig`'s pi-lineage defaults
    survived and the canary died on `(2x132 and 64x256)`. Patching `setup` keeps every property that
    mattered about the old seam — it still runs after the HF strict load, so `from_pretrained` has
    nothing extra to reconcile — and adds the one that was missing.

    NO LAZY INIT. `run()` calls `pipeline.setup()` to completion and only then constructs the
    Trainer, so `robottt_fast`'s parameters exist before the optimizer is built and are picked up by
    the normal parameter sweep. A width discovered on the first batch instead would produce
    parameters the optimizer never sees, which trains a frozen mechanism and reports nothing.

    `cfg` overrides the derivation entirely and is for tests/ablations only; the drivers pass None.
    """
    import gr00t.model.gr00t_n1d7.setup as setup_module

    _orig_setup = setup_module.Gr00tN1d7Pipeline.setup

    def _patched_setup(self):
        _orig_setup(self)  # model (HF strict-load) + dataset/processor + collator
        resolved = cfg
        if resolved is None:
            dims = robottt_seam_dims(self.processor, self.model.action_head.config)
            print(
                f"[robottt] derived seam dims: state_dim={dims['state_dim']} "
                f"(state_history_length={dims['state_history_length']}) "
                f"action_dim={dims['action_dim']} H={dims['action_horizon']} | "
                f"{dims['dims_source']}",
                flush=True,
            )
            resolved = robottt_config_from_env(
                cond_dim=int(self.model.action_head.model.proj_out_2.in_features),
                state_dim=dims["state_dim"],
                action_dim=dims["action_dim"],
                action_horizon=dims["action_horizon"],
                dims_source=dims["dims_source"],
            )
        attach_robottt(self.model, cfg=resolved)

    setup_module.Gr00tN1d7Pipeline.setup = _patched_setup
    print(
        "[robottt] action head patched (train path): fast-weight roll injected at temb, "
        "widths derived from the live processor",
        flush=True,
    )
