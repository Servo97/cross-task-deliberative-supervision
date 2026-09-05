"""Serve-side re-attach/restore for the GR00T gated-DeltaNet arm (module level).

`from_pretrained` BYPASSES the train-path patch that attaches the conditioner, so a serve that skips
the re-attach loads a STOCK action head, drops the trained `action_head.wsm_deltanet.*` tensors on
the floor, and runs the BASELINE policy under the arm's name. The checks here are the ones that make
that outcome impossible to reach silently:

  * the geometry is recovered FROM THE CHECKPOINT (no CLI flag can disagree with the trained recipe),
  * a checkpoint with no conditioner tensors is REFUSED rather than served as one,
  * a non-finite conditioner is REFUSED (the GR00T Eval2 0% was an invalid NaN-weight serve).

Pure torch + safetensors: the attach itself needs the gr00t venv and is exercised by the canary, but
everything that decides WHETHER to serve is testable here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_training.eval._groot_wsm_deltanet_eval import (  # noqa: E402
    PREFIX,
    assert_finite_deltanet,
    deltanet_geometry_from_state_dict,
    load_checkpoint_state_dict,
)
from workspace_models.networks.wsm_gated_deltanet import WSMGatedDeltaNetConditioner  # noqa: E402

GEOMETRY = {"w_dim": 512, "cond_dim": 1536, "window_len": 8, "num_heads": 2, "head_dim": 256}


def _conditioner(**overrides):
    return WSMGatedDeltaNetConditioner(**{**GEOMETRY, **overrides}, gate_init=1e-3)


def _checkpoint_state(conditioner, extra_noise: bool = True):
    """A checkpoint-shaped state dict: the conditioner under its real prefix, plus backbone noise."""
    state = {f"{PREFIX}{k}": v for k, v in conditioner.state_dict().items()}
    if extra_noise:  # the real ckpt is 12 GB of backbone; the filter must work
        state["action_head.model.proj_out_2.weight"] = torch.zeros(32, 1536)
        state["backbone.model.layers.0.mlp.up_proj.weight"] = torch.zeros(8, 8)
    return state


def test_geometry_is_recovered_from_the_checkpoint_not_from_flags():
    recovered = deltanet_geometry_from_state_dict(_checkpoint_state(_conditioner()))
    assert recovered == GEOMETRY


@pytest.mark.parametrize("window_len,num_heads,head_dim", [(2, 1, 64), (16, 4, 32), (8, 2, 256)])
def test_geometry_recovery_tracks_the_trained_recipe(window_len, num_heads, head_dim):
    """Every axis must come from the tensors — a stale serve flag cannot survive this."""
    cond = _conditioner(window_len=window_len, num_heads=num_heads, head_dim=head_dim)
    recovered = deltanet_geometry_from_state_dict(_checkpoint_state(cond))
    assert recovered["window_len"] == window_len
    assert recovered["num_heads"] == num_heads
    assert recovered["head_dim"] == head_dim
    assert recovered["cond_dim"] == GEOMETRY["cond_dim"] and recovered["w_dim"] == GEOMETRY["w_dim"]


def test_a_baseline_checkpoint_is_refused():
    """Serving a baseline ckpt through the deltanet path must fail, not quietly be the baseline."""
    baseline = {"action_head.model.proj_out_2.weight": torch.zeros(32, 1536)}
    with pytest.raises(RuntimeError, match="not a gated-DeltaNet"):
        deltanet_geometry_from_state_dict(baseline)


def test_a_truncated_conditioner_is_refused():
    state = _checkpoint_state(_conditioner())
    del state[f"{PREFIX}pos_decay_bias"]
    with pytest.raises(RuntimeError, match="missing"):
        deltanet_geometry_from_state_dict(state)


def test_non_finite_conditioner_is_refused():
    cond = _conditioner()
    assert_finite_deltanet(cond)  # healthy module passes
    with torch.no_grad():
        cond.alpha[0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite conditioner tensors"):
        assert_finite_deltanet(cond)


def test_load_checkpoint_state_dict_reads_sharded_safetensors(tmp_path):
    """The real ckpt is multi-shard safetensors; the reader must merge, not take the first shard."""
    from safetensors.torch import save_file

    state = _checkpoint_state(_conditioner())
    keys = sorted(state)
    half = len(keys) // 2
    save_file({k: state[k].contiguous() for k in keys[:half]}, str(tmp_path / "model-00001.safetensors"))
    save_file({k: state[k].contiguous() for k in keys[half:]}, str(tmp_path / "model-00002.safetensors"))

    loaded = load_checkpoint_state_dict(tmp_path)
    assert set(loaded) == set(state)
    assert deltanet_geometry_from_state_dict(loaded) == GEOMETRY


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint_state_dict(tmp_path / "empty")


# --------------------------------------------------------------------------------------------------
# dtype: the conditioner must end fp32 even when the served model is bf16 (2026-08-08 smoke)
# --------------------------------------------------------------------------------------------------
def test_fp32_assert_accepts_float32_and_names_any_other_dtype():
    from vla_training.eval._groot_wsm_deltanet_eval import assert_fp32_deltanet

    assert_fp32_deltanet(_conditioner())  # fp32: the trained dtype
    with pytest.raises(RuntimeError, match="not float32"):
        assert_fp32_deltanet(_conditioner().bfloat16())


def test_restoring_an_fp32_checkpoint_into_a_bf16_module_would_round_the_trained_weights():
    """WHY the float() happens BEFORE the load, not after it.

    `attach_and_restore_deltanet` casts each checkpoint tensor to the freshly attached module's
    dtype. If that module were still bf16 -- which is what `from_pretrained` hands the serve -- the
    restore itself would round every trained value to 8 mantissa bits, and a later `.float()` could
    not put them back. The checkpoint is F32 (verified from its safetensors headers), so this test
    pins the loss that ordering avoids.
    """
    trained = _conditioner()
    reference = trained.state_dict()["proj_q.weight"].clone()
    rounded = reference.to(torch.bfloat16).float()
    assert not torch.equal(reference, rounded), "expected bf16 to lose bits on these weights"
    # The shipped order restores into an fp32 module, so the values survive exactly.
    target = _conditioner().float()
    target.load_state_dict(
        {k: v.to(dtype=target.state_dict()[k].dtype) for k, v in trained.state_dict().items()}, strict=True
    )
    assert torch.equal(target.state_dict()["proj_q.weight"], reference)


def test_fp32_conditioner_feeds_a_bf16_temb_seam_finitely():
    """The end-to-end dtype contract: fp32 window -> fp32 conditioner -> bf16 temb, all finite.

    Mirrors `_cond_from_window` (fp32 input cast) and the temb add (`cond.to(dtype=temb.dtype)`)
    without needing the gr00t venv, so CI covers the seam the smoke crashed on.
    """
    conditioner = _conditioner().float()
    window = torch.randn(1, GEOMETRY["window_len"], GEOMETRY["w_dim"], dtype=torch.float32)
    cond = conditioner(window.to(dtype=torch.float32))  # the fp32 cast _cond_from_window does
    assert cond.dtype == torch.float32 and torch.isfinite(cond).all()

    temb = torch.randn(1, GEOMETRY["cond_dim"], dtype=torch.bfloat16)
    fused = temb + cond.to(device=temb.device, dtype=temb.dtype)
    assert fused.dtype == torch.bfloat16 and torch.isfinite(fused).all()


def test_a_bf16_conditioner_reproduces_the_smoke_crash_against_an_fp32_window():
    """The exact 2026-08-08 failure, pinned so the fix cannot be silently reverted."""
    bf16_conditioner = _conditioner().bfloat16()
    window = torch.randn(1, GEOMETRY["window_len"], GEOMETRY["w_dim"], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same dtype"):
        bf16_conditioner(window)
