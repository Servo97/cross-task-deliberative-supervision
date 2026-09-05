"""Drive the REAL `attach_and_restore_robottt` entry point, not a reconstruction of it.

WHY THIS FILE EXISTS. The ttt serve died on its first real startup with

    dataclasses.FrozenInstanceError: cannot assign to field 'token_dim'
    _groot_robottt_eval.py:159  cfg.token_dim = int(geometry["token_dim"])

`RoboTTTConfig` is frozen, and the serve path is the only caller that overrides the env-derived
widths with checkpoint-derived ones. Every existing CPU check built its config through the
constructor, so that branch had never executed. The lesson is not "add a frozen-dataclass test" --
it is that a serve entry point has to be exercised END TO END by something, so the tests below call
the shipped function on a real staged checkpoint and let it construct, attach and strict-restore.

CPU is enough: construction + restore never needs a GPU. Skips (never fails) when the checkpoint or
the gr00t venv is absent, so it stays useful in CI and decisive on the box.

Run on the box:  PYTHONPATH=. /data/work/groot_env/bin/python -m pytest tests/test_groot_robottt_serve_attach.py -q
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CKPT = Path("/data/work/ckpts_groot_rmb/rmb1t-heatpot-ttt")

gr00t = pytest.importorskip("gr00t", reason="needs the GR00T venv")
if not (CKPT / "config.json").exists():
    pytest.skip(f"no staged ttt checkpoint at {CKPT}", allow_module_level=True)


SHIM = Path("/data/work/groot_smoke/hf_offline_shim.py")


def _policy():
    """Build the policy exactly as `serve_groot_ws.main` does, gated-backbone handling included.

    GR00T constructs its processor from the GATED `nvidia/Cosmos-Reason2-2B` repo id, so without the
    offline shim + local_files_only this dies with a 401 long before the attach under test. Skipping
    that setup would make the test pass or fail for reasons unrelated to the serve path.
    """
    import importlib.util

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from vla_training.eval.serve_groot_ws import force_transformers_local_files_only
    from vla_training.train.train_base._groot_common import load_modality_config

    if not SHIM.exists():
        pytest.skip(f"no HF offline shim at {SHIM}")
    spec = importlib.util.spec_from_file_location("hf_offline_shim", SHIM)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    shim.install()
    force_transformers_local_files_only()
    load_modality_config()
    return Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve("new_embodiment"), model_path=str(CKPT), device="cpu", strict=True
    )


def test_the_real_serve_entry_point_attaches_and_restores():
    """This is the call `serve_groot_ws.py --mechanism ttt` makes. It must not raise."""
    import torch

    from vla_training.eval._groot_robottt_eval import (
        attach_and_restore_robottt,
        robottt_geometry_from_state_dict,
    )
    from vla_training.eval._groot_wsm_deltanet_eval import load_checkpoint_state_dict

    policy = _policy()
    geometry = attach_and_restore_robottt(policy, str(CKPT))

    # The checkpoint-derived widths must be what actually got attached -- the exact thing the
    # frozen-dataclass bug silently would NOT have done had the assignment been allowed to no-op.
    expected = robottt_geometry_from_state_dict(load_checkpoint_state_dict(CKPT))
    fast = policy.model.action_head.robottt_fast
    for name in ("token_dim", "fast_hidden", "num_registers"):
        assert int(getattr(fast.cfg, name)) == int(expected[name]), name
        assert int(geometry[name]) == int(expected[name]), name
    assert all(torch.isfinite(t).all() for t in fast.state_dict().values())


def test_the_config_the_serve_builds_is_frozen_and_still_takes_the_checkpoint_widths():
    """Pins BOTH halves of the bug: the dataclass is frozen, and the override still lands."""
    from vla_training.train.train_base._groot_robottt_common import robottt_config_from_env

    cfg = robottt_config_from_env(cond_dim=1536, state_dim=16, action_dim=12, action_horizon=16, dims_source="test")
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.token_dim = 999  # the line that broke the serve
    rebuilt = dataclasses.replace(cfg, token_dim=999)  # the shipped way
    assert rebuilt.token_dim == 999 and cfg.token_dim != 999


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
