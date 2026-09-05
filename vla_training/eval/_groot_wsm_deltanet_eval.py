"""Serve-side re-attach + weight restore for the GR00T gated-DeltaNet workspace conditioner.

WHY THIS FILE EXISTS. `from_pretrained` BYPASSES a train-path module patch — the gotcha that has
bitten this codebase twice. `install_wsm_deltanet_action_head` monkeypatches
`Gr00tN1d7Pipeline._create_model`, which the TRAIN path goes through; a serve that loads the
checkpoint with `Gr00tPolicy`/`from_pretrained` never runs it, so the action head comes back as the
STOCK class with no conditioner. The trained `action_head.wsm_deltanet.*` tensors are present in the
checkpoint but land nowhere, and (because the head class is stock) the model then serves as the
BASELINE POLICY under the arm's name — a silent, plausible, completely wrong result.

The fix is the same shape `serve_groot_batched.py --mode wsm_cfg` uses: ATTACH the module first,
THEN load its weights over it. It is deliberately not the other way round; a strict load before
attach has nothing to load into.

AUTO-DETECTION, not flags. Everything the conditioner needs is recoverable from the checkpoint
tensors themselves — `window_len` from `pos_decay_bias [K, H]` (which is why that parameter has that
shape), `cond_dim`/`w_dim`/`num_heads`/`head_dim` from the projection shapes. So serve cannot be
pointed at a checkpoint with the wrong geometry via a stale CLI flag, which is exactly the class of
mistake that produced the invalid GR00T Eval2 numbers (a v1/100k NaN-weight encoder served against a
v2/50k-trained arm).

FINITE GUARDS ARE FAIL-LOUD, not clamped. After the NaN-encoder incident the contract everywhere in
this study is that a non-finite conditioning path stops the process rather than degrading to
something that still returns actions.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "load_checkpoint_state_dict",
    "deltanet_geometry_from_state_dict",
    "attach_and_restore_deltanet",
    "assert_finite_deltanet",
    "assert_fp32_deltanet",
]

PREFIX = "action_head.wsm_deltanet."


def load_checkpoint_state_dict(ckpt: str | Path) -> dict:
    """Read a GR00T checkpoint dir (safetensors shards or .bin) or a single file into one dict."""
    import torch

    path = Path(ckpt).expanduser()
    state: dict = {}
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if shards:
            from safetensors.torch import load_file

            for shard in shards:
                state.update(load_file(str(shard)))
        else:
            for shard in sorted(path.glob("pytorch_model*.bin")):
                state.update(torch.load(shard, map_location="cpu"))
    else:
        blob = torch.load(path, map_location="cpu")
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
    if not state:
        raise FileNotFoundError(f"[wsm-dn-eval] no model weights under {path}")
    return state


def deltanet_geometry_from_state_dict(state_dict, prefix: str = PREFIX) -> dict:
    """Recover the conditioner's geometry from the trained tensors. Raises if it is not a dn ckpt.

    This is what makes the serve path self-describing: the checkpoint IS the recipe, so an eval can
    neither disagree with the train recipe nor be run against a baseline checkpoint by accident.
    """
    from vla_training.train.train_base._groot_wsm_deltanet_common import window_len_from_state_dict

    sub = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not sub:
        available = sorted({k.rsplit(".", 1)[0] for k in state_dict if "wsm" in k})[:5]
        raise RuntimeError(
            f"[wsm-dn-eval] no '{prefix}*' tensors in the checkpoint — this is not a gated-DeltaNet "
            f"finetune. Serving it as one would silently be the baseline policy under the arm's "
            f"name. (wsm-ish prefixes present: {available or 'none'})"
        )

    for required in ("proj_q.weight", "proj_beta.weight", "proj_readout.weight", "alpha", "pos_decay_bias"):
        if required not in sub:
            raise RuntimeError(f"[wsm-dn-eval] checkpoint is missing '{prefix}{required}'")

    inner, w_dim = sub["proj_q.weight"].shape  # nn.Linear stores [out, in]
    num_heads = int(sub["proj_beta.weight"].shape[0])
    cond_dim = int(sub["proj_readout.weight"].shape[0])
    if int(inner) % num_heads:
        raise RuntimeError(f"[wsm-dn-eval] inner width {int(inner)} is not divisible by num_heads {num_heads}")
    geometry = {
        "w_dim": int(w_dim),
        "cond_dim": cond_dim,
        "window_len": window_len_from_state_dict(state_dict, prefix=prefix),
        "num_heads": num_heads,
        "head_dim": int(inner) // num_heads,
    }
    if int(sub["alpha"].shape[0]) != cond_dim:
        raise RuntimeError(f"[wsm-dn-eval] alpha width {int(sub['alpha'].shape[0])} != cond_dim {cond_dim}")
    if int(sub["pos_decay_bias"].shape[1]) != num_heads:
        raise RuntimeError("[wsm-dn-eval] pos_decay_bias head axis disagrees with proj_beta")
    return geometry


def assert_finite_deltanet(conditioner, where: str = "after restore") -> None:
    """Fail loud on a non-finite conditioner parameter.

    The GR00T Eval2 0% result was INVALID because a NaN-weight encoder was served and the pipeline
    happily produced actions from it. A conditioner that is NaN here would push NaN into `temb` and
    from there into every DiT block; the run must stop, not score zero.
    """
    import torch

    bad = [name for name, tensor in conditioner.state_dict().items() if not torch.isfinite(tensor).all()]
    if bad:
        raise RuntimeError(
            f"[wsm-dn-eval] non-finite conditioner tensors {where}: {bad}. Refusing to serve — a "
            "NaN conditioner poisons temb and yields a scoreable but meaningless rollout (see the "
            "GR00T Eval2 NaN-encoder incident)."
        )


def assert_fp32_deltanet(conditioner, where: str = "after restore") -> None:
    """Fail loud unless every restored conditioner parameter is float32.

    Serve must NOT inherit the served model's bf16 dtype. The checkpoints were produced from fp32
    master weights (all 14 `action_head.wsm_deltanet.*` safetensors headers are F32), and the
    jax->torch parity fixture that certified this conditioner against the pi0.5 implementation was
    an fp32 reference. Serving the same tensors in bf16 would be a numerics change nobody measured,
    on the one module whose whole job is to steer the DiT -- and it would be invisible, because the
    rollouts would still complete. Named for the 2026-08-08 smoke, where the bf16 module instead
    failed loudly against `_cond_from_window`'s fp32 input; that crash was the lucky outcome.
    """
    import torch

    bad = {
        name: str(tensor.dtype) for name, tensor in conditioner.state_dict().items() if tensor.dtype != torch.float32
    }
    if bad:
        raise RuntimeError(
            f"[wsm-dn-eval] conditioner parameters are not float32 {where}: {bad}. The trained "
            f"tensors are F32 in the checkpoint; serving them in another dtype is an unmeasured "
            f"numerics change. Refusing to serve."
        )


def attach_and_restore_deltanet(model, finetune_ckpt: str | Path, *, history_dropout: float = 0.0) -> dict:
    """ATTACH the conditioner to a `from_pretrained` model, THEN load its trained weights.

    Returns the recovered geometry. `history_dropout` is a TRAIN-time regularizer and is forced to
    0.0 at serve; it is exposed only so a deliberate ablation can set it.
    """
    from vla_training.train.train_base._groot_wsm_deltanet_common import attach_wsm_deltanet

    state_dict = load_checkpoint_state_dict(finetune_ckpt)
    geometry = deltanet_geometry_from_state_dict(state_dict)

    conditioner = attach_wsm_deltanet(
        model,
        w_dim=geometry["w_dim"],
        cond_dim=geometry["cond_dim"],
        window_len=geometry["window_len"],
        num_heads=geometry["num_heads"],
        head_dim=geometry["head_dim"],
        gate_init=1e-3,  # overwritten by the restore; only the shape matters here
        history_dropout=history_dropout,
        log_every=0,  # no train-step logging on the serve path
    )

    # FLOAT32 BEFORE THE RESTORE, and this is not a style choice.
    #
    # `attach_wsm_deltanet` derives its dtype from the action head's first trainable parameter
    # (_groot_wsm_deltanet_common.py:243-244). At TRAIN that is an fp32 master weight under the HF
    # bf16 Trainer, so the conditioner was built, trained and SAVED in float32 -- confirmed from the
    # checkpoint itself: every one of the 14 `action_head.wsm_deltanet.*` safetensors headers reads
    # F32 (as does the rest of the checkpoint). At SERVE the same line reads a model that
    # `from_pretrained` has already materialised in bfloat16, so the freshly attached module comes
    # back bf16 -- a dtype the trained weights were never produced under.
    #
    # That cost a smoke on 2026-08-08: `_cond_from_window` casts the omega window to float32 and
    # hands it to a bf16 `proj_q`, which dies with "mat1 and mat2 must have the same dtype, but got
    # Float and BFloat16" on the first inference. Restoring INTO a bf16 module would also have
    # silently downcast the fp32 checkpoint tensors, so the float() happens BEFORE the load rather
    # than after it -- no trained bit is rounded on the way in.
    #
    # Downstream is safe: the temb seam already casts the conditioner's OUTPUT to `temb.dtype` at
    # the add, so an fp32 conditioner feeds a bf16 DiT without a further change.
    conditioner.float()

    sub = {k[len(PREFIX) :]: v for k, v in state_dict.items() if k.startswith(PREFIX)}
    # STRICT: a partial restore is the failure this whole module exists to prevent, so it must be an
    # error and not a warning. Dtype/device are matched to the freshly attached module.
    target = conditioner.state_dict()
    typed = {k: v.to(device=target[k].device, dtype=target[k].dtype) for k, v in sub.items() if k in target}
    missing_keys, unexpected_keys = conditioner.load_state_dict(typed, strict=True)
    assert not missing_keys and not unexpected_keys  # strict=True would already have raised
    assert_fp32_deltanet(conditioner)
    assert_finite_deltanet(conditioner)

    print(
        f"[wsm-dn-eval] re-attached + restored {len(typed)} conditioner tensors from "
        f"{finetune_ckpt} | {geometry} | dtype=float32 (from_pretrained bypasses the train-path "
        f"patch, so this step is what keeps the arm from serving as the baseline; the float() is "
        f"what keeps it from serving in a dtype the weights were never trained in)",
        flush=True,
    )
    return geometry
