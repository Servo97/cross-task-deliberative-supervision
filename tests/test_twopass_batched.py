"""H1 validation: batched CFG two-pass == sequential two-pass, exactly (perf-first mandate).
Fake DiT with attention + LN + additive temb conditioning — the batch-equivariance class the real DiT
belongs to. Run: <torch-env>/bin/python tests/test_twopass_batched.py"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_training.train.train_base._groot_wsm_cfg_common import cfg_velocities  # noqa: E402

torch.manual_seed(0)
B, L, D, H = 1, 24, 64, 16


class FakeTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, D)
        self._wsm_cond = None

    def forward(self, ts):
        temb = self.proj(ts.float().unsqueeze(-1))
        return temb if self._wsm_cond is None else temb + self._wsm_cond.to(temb.dtype)


class FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep_encoder = FakeTE()
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D)

    def forward(self, sa, enc, ts):
        temb = self.timestep_encoder(ts)[:, None, :]
        x = self.ln(sa + temb)
        a, _ = self.attn(x, enc, enc, need_weights=False)
        return self.out(x + a)


dit = FakeDiT().eval()
vl = torch.randn(B, 40, D)  # captured encoder state (tiled inside _vel)
sa = torch.randn(B, L, D)
ts = torch.randint(0, 100, (B,))
cond, uncond = torch.randn(B, D), torch.randn(B, D)


def _vel(sa_, ts_):
    n = sa_.shape[0] // B
    return dit(sa_, vl.repeat(n, 1, 1), ts_)[:, -H:]


with torch.no_grad():
    for s in (0.5, 1.0, 1.5, 4.0):
        v_b = cfg_velocities(_vel, dit.timestep_encoder, sa, ts, cond, uncond, s, mode="batched")
        v_s = cfg_velocities(_vel, dit.timestep_encoder, sa, ts, cond, uncond, s, mode="seq")
        d = float((v_b - v_s).abs().max())
        assert d < 1e-5, f"s={s}: batched != seq (max diff {d})"
        print(f"ok s={s}: max|batched-seq| = {d:.2e}")
    assert dit.timestep_encoder._wsm_cond is None, "helper must clear the temb stash"
print("BATCHED TWO-PASS EXACT — H1 validated")
