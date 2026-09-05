"""Extract a jax->torch parity fixture for WSMGatedDeltaNetConditioner.

Writes an npz holding: every projection weight/bias, pos_decay_bias, alpha, the input omega window,
and the jax reference output. The torch port must reproduce `out` from the same params + input.
"""

import pathlib

import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models.wsm_current_cond import WSMGatedDeltaNetConditioner

W_DIM, COND_DIM, WINDOW, HEADS, HEAD_DIM = 24, 16, 8, 2, 6
B = 3

m = WSMGatedDeltaNetConditioner(
    w_dim=W_DIM,
    cond_dim=COND_DIM,
    window_len=WINDOW,
    num_heads=HEADS,
    head_dim=HEAD_DIM,
    gate_init=1e-3,
    rngs=nnx.Rngs(0),
)

# Deterministic non-trivial params so no term is accidentally inert.
rng = np.random.default_rng(1234)


def fill(lin, fan_in, fan_out):
    lin.kernel.value = jnp.asarray(rng.normal(0, 0.05, (fan_in, fan_out)).astype(np.float32))
    lin.bias.value = jnp.asarray(rng.normal(0, 0.02, (fan_out,)).astype(np.float32))


inner = HEADS * HEAD_DIM
fill(m.proj_q, W_DIM, inner)
fill(m.proj_k, W_DIM, inner)
fill(m.proj_v, W_DIM, inner)
fill(m.proj_beta, W_DIM, HEADS)
fill(m.proj_decay, W_DIM, HEADS)
fill(m.proj_readout, inner, COND_DIM)
# non-zero so the positional decay prior is actually exercised
m.pos_decay_bias.value = jnp.asarray(rng.normal(0, 0.3, (WINDOW, HEADS)).astype(np.float32))
# alpha away from init so tanh(alpha) is not ~0 (a zero gate would hide readout bugs)
m.alpha.value = jnp.asarray(rng.normal(0, 0.5, (COND_DIM,)).astype(np.float32))

omega = rng.normal(0, 1.0, (B, WINDOW, W_DIM)).astype(np.float32)
out = np.asarray(m(jnp.asarray(omega)))

np.savez(
    str(pathlib.Path(__file__).resolve().parent / "deltanet_jax_parity.npz"),
    w_dim=W_DIM,
    cond_dim=COND_DIM,
    window_len=WINDOW,
    num_heads=HEADS,
    head_dim=HEAD_DIM,
    proj_q_kernel=np.asarray(m.proj_q.kernel.value),
    proj_q_bias=np.asarray(m.proj_q.bias.value),
    proj_k_kernel=np.asarray(m.proj_k.kernel.value),
    proj_k_bias=np.asarray(m.proj_k.bias.value),
    proj_v_kernel=np.asarray(m.proj_v.kernel.value),
    proj_v_bias=np.asarray(m.proj_v.bias.value),
    proj_beta_kernel=np.asarray(m.proj_beta.kernel.value),
    proj_beta_bias=np.asarray(m.proj_beta.bias.value),
    proj_decay_kernel=np.asarray(m.proj_decay.kernel.value),
    proj_decay_bias=np.asarray(m.proj_decay.bias.value),
    proj_readout_kernel=np.asarray(m.proj_readout.kernel.value),
    proj_readout_bias=np.asarray(m.proj_readout.bias.value),
    pos_decay_bias=np.asarray(m.pos_decay_bias.value),
    alpha=np.asarray(m.alpha.value),
    omega_window=omega,
    out=out,
)
print(
    "fixture written; out stats",
    out.shape,
    float(out.mean()),
    float(out.std()),
    "finite",
    bool(np.isfinite(out).all()),
)
