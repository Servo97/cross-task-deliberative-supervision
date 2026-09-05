#!/usr/bin/env python3
"""pi0.5 online-workspace policy server.

The historical legacy_cfg checkpoint path remains available for explicit reproduction. Every run must
select --interface legacy_cfg, cfg2 (strictly current-only P_t/null_t), tanh, or tanh_robottt (Q3: the
tanh workspace read AND the RoboTTT fast-weight loop combined). All interfaces add only to the action
expert's adarms_cond; cfg2/tanh/tanh_robottt never modify VLM or prefix tokens. The shared
WSMPiInferWrapper computes causal online omega_t and injects its oldest-to-newest window into the policy.
tanh_robottt additionally attaches the RoboTTTServeRunner: per env, W starts from the meta-learned init
at every t=0 reset, is HELD fixed through all Euler passes of one chunk, and advances by exactly one
inner-GD commit on the policy's own executed (model-space) chunk — identical to the Q2 robottt_fast
serve, on top of the identical tanh conditioning.

The tanh interface covers THREE conditioner variants sharing the wsm_tanh_cond subtree: the shipped
tanh MLP read of omega_t, the gated-DeltaNet steering variant that recurs across the omega window, and
the PTRM variant that additionally refines that read through a weight-tied recursive core with a Q
head. Which one a checkpoint holds — and, for the deltanet, how long a window it was trained on, and
for PTRM, how deep a recursion — is AUTO-DETECTED from the checkpoint's params metadata before the
model is built, so the sealed eval submit command needs no new flags. WSM_COND_TYPE is an explicit
override that must agree with the tree.

PTRM's INFERENCE knobs are the experiment (the checkpoint is fixed; K/sigma/selection are what the
evidence ladder sweeps), so they are read from the environment at serve construction and ride the
same Pi0Config path the deltanet geometry does — never a post-build patch, which the
from_pretrained-bypass lesson says would be silently discarded:

  WSM_PTRM_EVAL_K       int   >= 1     parallel rollouts                  (default 1)
  WSM_PTRM_EVAL_SIGMA   float >= 0     per-micro-step noise scale         (default 0.0)
  WSM_PTRM_EVAL_SELECT  q|random|mean  which rollout is decoded           (default q)

Unset means K=1, sigma=0.0, "q" — a deterministic full-depth read that is bit-identical to the
training graph's final conditioning vector, i.e. the honest PTRM-off control.

Guidance scale is a trace-time constant baked into Pi0 before policy construction. CFG uses one suffix call
at s=0 (learned-null) and s=1 (conditional), and two calls only at other scales; all branches share one
prefix KV cache. Tanh always uses one suffix call and requires s=1. Run one serve process per scale.

Example for new S2:

  python vla_training/eval/serve_pi_05_wsm_cfg.py \
      --interface cfg2 --finetune-ckpt <stage-s-cfg2/59999> \
      --tap-ckpt <H300+MG-feature-source/149999> \
      --encoder-ckpt <wsm_runs/pi_wsm_v1/wsm_step100000.pt> \
      --task-lang-table <wsm_policy_feats/pi_step100000/task_lang_table.npz> \
      --guidance-scale 1.0 --k-window 1 --stride 8 --tap-prompt terse --port 8000

Stage-E ω (the H14 deliberative encoders, ReMemBench arms P1'/P2'/P3'): pass
``--encoder-kind stage_e --pool-ckpt <wsm_step100000.pt> --stage-e-omega-root <.../omega/<cell>/remembench>``
with a Stage-E ``--encoder-ckpt``. The Stage-E front end (``vla_training/eval/stage_e_serve.py``) replaces
only the ``load_wsm`` encoder; the conditioner, the window rule, the wrapper and the policy build are the
sealed code path above, and the task-language table is derived from the ω store the policy trained on.
The default ``--encoder-kind wsm_v1`` is byte-for-byte the behaviour every sealed arm was served with.

Run in the openpi-jax-latest environment.
"""

from __future__ import annotations

import argparse
import functools
import os
import pathlib
from collections.abc import Mapping
from typing import NamedTuple

from vla_training.eval.serve_pi_05_wsm import _PI_PROPRIO_DIM, WSMPiInferWrapper, tap_min_batch

# The wsm_tanh_cond subtree is shared by two modules (see openpi.models.wsm_current_cond): the shipped
# tanh MLP read and the gated-DeltaNet steering variant. These leaf names are what tells them apart in
# a checkpoint; the tuples are load-bearing for auto-detection, not documentation.
_TANH_LEAF_PREFIXES = ("proj_t_in", "proj_t_out")
_DELTANET_LEAF_PREFIXES = (
    "proj_q",
    "proj_k",
    "proj_v",
    "proj_beta",
    "proj_decay",
    "proj_readout",
    "pos_decay_bias",
)
#: PTRM is a SUPERSET of the deltanet tree: it reuses every leaf above and adds the recursive core.
#: `step_bias` is [steps, cond_dim], so the trained recursion depth is structural exactly as
#: `pos_decay_bias` makes the trained window structural.
_PTRM_LEAF_PREFIXES = ("z0", "core_in", "core_out", "step_bias", "rms", "proj_out", "q_head")
WSM_COND_SUBTREE = "wsm_tanh_cond"
#: Serve-side conditioner names; also the accepted values of the WSM_COND_TYPE override.
WSM_COND_TYPES = ("tanh", "gated_deltanet", "gated_deltanet_ptrm")
#: Conditioners whose recurrence consumes the whole omega window (serve must ship exactly K rows).
WSM_WINDOWED_COND_TYPES = ("gated_deltanet", "gated_deltanet_ptrm")
PTRM_EVAL_SELECTS = ("q", "random", "mean")


class WSMCondSpec(NamedTuple):
    """The conditioner architecture recovered from a checkpoint (never from an eval flag)."""

    cond_type: str
    window: int
    num_heads: int | None = None
    head_dim: int | None = None
    ptrm_steps: int | None = None

    def describe(self) -> str:
        if self.cond_type in WSM_WINDOWED_COND_TYPES:
            return (
                f"{self.cond_type}(window={self.window}, heads={self.num_heads}, "
                f"head_dim={self.head_dim}"
                + (f", steps={self.ptrm_steps}" if self.ptrm_steps is not None else "")
                + ")"
            )
        return f"tanh(window={self.window})"


class PTRMEvalKnobs(NamedTuple):
    """The eval-only PTRM procedure, read from the environment at serve construction."""

    k: int
    sigma: float
    select: str
    zero_cond: bool = False

    @property
    def deterministic(self) -> bool:
        """True when this is the PTRM-off control (one rollout, no noise, nothing to select)."""
        return self.k == 1 and self.sigma == 0.0

    @property
    def is_default(self) -> bool:
        """True when NOTHING was asked for: the plain deterministic read, conditioning left intact.

        Distinct from `deterministic`, which is about the K/sigma procedure only. zero_cond is a
        deterministic cell that is nonetheless a different experiment, so it must not be waved
        through the 'no knobs were set' path.
        """
        return self.deterministic and not self.zero_cond

    def describe(self) -> str:
        return (
            f"K={self.k}, sigma={self.sigma:g}, select={self.select}"
            + (" [deterministic parity]" if self.deterministic else "")
            + (" ZERO_COND=ON (conditioning vector forced to 0 at the adaRMS seam)" if self.zero_cond else "")
        )


def ptrm_eval_knobs_from_env(environ: Mapping[str, str] | None = None) -> PTRMEvalKnobs:
    """Parse WSM_PTRM_EVAL_{K,SIGMA,SELECT}; defaults are the deterministic K=1/sigma=0/'q' read.

    Pure and unit-tested directly. Garbage must fail HERE, before a GPU is touched — a swept knob
    that silently fell back to its default would report a PTRM-off number under a PTRM-on label.
    """
    env = os.environ if environ is None else environ
    raw_k = env.get("WSM_PTRM_EVAL_K")
    raw_sigma = env.get("WSM_PTRM_EVAL_SIGMA")
    select = env.get("WSM_PTRM_EVAL_SELECT") or "q"
    raw_zero = env.get("WSM_PTRM_ZERO_COND")
    if raw_zero is None or raw_zero == "":
        zero_cond = False
    elif raw_zero.strip().lower() in ("1", "true", "yes", "on"):
        zero_cond = True
    elif raw_zero.strip().lower() in ("0", "false", "no", "off"):
        zero_cond = False
    else:
        # No truthiness guessing: a typo'd ablation flag that quietly evaluated to False would
        # report the CONDITIONED number under the zeroed-conditioning label.
        raise RuntimeError(f"[serve-pi-workspace] WSM_PTRM_ZERO_COND must be 1/0 (true/false), got {raw_zero!r}")
    try:
        k = int(raw_k) if raw_k is not None else 1
    except ValueError as error:
        raise RuntimeError(f"[serve-pi-workspace] WSM_PTRM_EVAL_K must be an integer, got {raw_k!r}") from error
    try:
        sigma = float(raw_sigma) if raw_sigma is not None else 0.0
    except ValueError as error:
        raise RuntimeError(f"[serve-pi-workspace] WSM_PTRM_EVAL_SIGMA must be a float, got {raw_sigma!r}") from error
    if k < 1:
        raise RuntimeError(f"[serve-pi-workspace] WSM_PTRM_EVAL_K must be >= 1, got {k}")
    if not (sigma >= 0.0) or sigma == float("inf"):
        raise RuntimeError(f"[serve-pi-workspace] WSM_PTRM_EVAL_SIGMA must be finite and >= 0, got {sigma}")
    if select not in PTRM_EVAL_SELECTS:
        raise RuntimeError(
            f"[serve-pi-workspace] WSM_PTRM_EVAL_SELECT must be one of {PTRM_EVAL_SELECTS}, got {select!r}"
        )
    return PTRMEvalKnobs(k=k, sigma=sigma, select=select, zero_cond=zero_cond)


def apply_zero_cond_patch() -> None:
    """WSM_PTRM_ZERO_COND: run the PTRM head in full, then force its output vector to zero.

    THE CAUSAL TEST this exists for: is the arm's gain carried by the serve-time conditioning READ,
    or is it trunk finetuning wearing a decorative head? Same checkpoint, same protocol, one term
    removed at the seam where it enters the policy.

    WHERE. pi0.sample_actions computes `tanh_vec = wsm_tanh_cond.eval_cond(...)` and adds exactly
    that vector to adarms_cond (`ac = adarms_cond + extra_cond`). eval_cond's return IS the added
    term — the tanh gate is already inside it — so zeroing the return zeroes the entire workspace
    contribution at the adaRMS seam and nothing else. The head still runs first: all T micro-steps,
    the Q head, and the selection all execute, so a head-side failure still surfaces.

    `vec * 0.0`, NOT `zeros_like`: multiplying propagates NaN/Inf, so a numerically broken head
    still poisons the action instead of being laundered into a clean zero by the ablation.

    WHY THE CLASS, NOT THE INSTANCE. openpi serves through nnx_utils.module_jit, which does
    nnx.split(module) once and nnx.merge(graphdef, state) INSIDE the jit, so the object whose
    eval_cond actually runs is a reconstruction, not the instance we patched. Measured on this flax
    version, an instance attribute does survive that round trip (nnx carries it as static graph
    metadata) — but that is an implementation detail of nnx's static capture, and the reconstruction
    is guaranteed only to share the CLASS. Patching the class does not depend on that detail. The
    sealed openpi tree on disk is never modified; this is a process-local override applied before
    the policy is built, so the tarball sha the train manifest pins still describes what is served.
    """
    from openpi.models.wsm_current_cond import WSMGatedDeltaNetPTRMConditioner as _Cond

    if getattr(_Cond.eval_cond, "_wsm_zero_cond", False):
        return
    original = _Cond.eval_cond

    @functools.wraps(original)
    def eval_cond_zeroed(self, *args, **kwargs):
        return original(self, *args, **kwargs) * 0.0

    eval_cond_zeroed._wsm_zero_cond = True
    _Cond.eval_cond = eval_cond_zeroed
    print(
        "[serve-pi-workspace] ZERO_COND: WSMGatedDeltaNetPTRMConditioner.eval_cond patched "
        "(head runs, output multiplied by 0 before the adaRMS add)",
        flush=True,
    )


def assert_zero_cond_active(policy) -> None:
    """Prove the ablation is real ON THE TRACED PATH, and that the head still produces a live vector.

    Two claims, because either one failing silently would invert the experiment's meaning:
      1. the UNPATCHED head returns a finite, NON-ZERO vector -> the thing being ablated exists, so a
         collapse cannot be blamed on an already-dead conditioner;
      2. the patched eval_cond returns EXACTLY zero when driven through nnx.split/merge -- the same
         graphdef round-trip module_jit performs -- so the zeroing survives into the served path.
    A direct out-of-jit call would pass even for an instance patch that the jit discards, which is
    precisely the failure this check is aimed at.
    """
    import flax.nnx as nnx
    import jax.numpy as jnp

    cond = policy._model.wsm_tanh_cond
    # Geometry off the CONDITIONER, not off Pi0: Pi0 copies only wsm_cond_type / wsm_ptrm_* out of
    # its config and never stores the window or w_dim, so reading them from the model raises
    # AttributeError at serve startup. The conditioner records its own w_dim/window_len.
    window = int(cond.window_len)
    w_dim = int(cond.w_dim)
    dummy = jnp.ones((1, window, w_dim), dtype=jnp.float32)

    original = getattr(type(cond).eval_cond, "__wrapped__", None)
    if original is None:
        raise RuntimeError("[serve-pi-workspace] ZERO_COND requested but eval_cond is not patched")
    ref = jnp.asarray(original(cond, dummy, rng=None, k=1, sigma=0.0, select="q"))
    if not bool(jnp.all(jnp.isfinite(ref))):
        raise RuntimeError("[serve-pi-workspace] ZERO_COND self-test: unpatched head returned non-finite")
    if float(jnp.max(jnp.abs(ref))) == 0.0:
        raise RuntimeError(
            "[serve-pi-workspace] ZERO_COND self-test: the UNPATCHED head already returns all zeros, "
            "so this ablation would remove nothing and its result would be uninterpretable"
        )

    graphdef, state = nnx.split(cond)

    def _through_merge(state):
        return nnx.merge(graphdef, state).eval_cond(dummy, rng=None, k=1, sigma=0.0, select="q")

    import jax

    got = jnp.asarray(jax.jit(_through_merge)(state))
    if float(jnp.max(jnp.abs(got))) != 0.0:
        raise RuntimeError(
            f"[serve-pi-workspace] ZERO_COND self-test FAILED: conditioning vector is not zero on the "
            f"jitted split/merge path (max|v|={float(jnp.max(jnp.abs(got))):g}) — the patch does not "
            f"reach the served policy. Refusing to serve a mislabeled ablation."
        )
    print(
        f"[serve-pi-workspace] ✓ ZERO_COND verified on the jitted path: unpatched head max|v|="
        f"{float(jnp.max(jnp.abs(ref))):.4f} -> served max|v|=0.0",
        flush=True,
    )


def classify_wsm_cond_subtree(
    leaf_shapes: Mapping[str, tuple[int, ...]], *, requested: str | None = None
) -> WSMCondSpec:
    """Recover the wsm_tanh_cond variant from its leaf names/shapes. Pure; unit-tested directly.

    `leaf_shapes` maps a '/'-joined path RELATIVE to wsm_tanh_cond (e.g. "proj_t_out/kernel") to that
    leaf's shape. The deltanet's `pos_decay_bias` is [window_len, num_heads], so the trained window
    length is structural — a checkpoint trained at K=8 can never be served at K=1. Anything ambiguous
    (both families present, neither present, a truncated deltanet tree, or an explicit request that
    contradicts the tree) raises instead of silently producing a garbage policy.
    """
    if not leaf_shapes:
        raise RuntimeError(f"[serve-pi-workspace] checkpoint has no {WSM_COND_SUBTREE} subtree — wrong ckpt/config")
    roots = {path.split("/", 1)[0] for path in leaf_shapes}
    is_tanh = bool(roots.intersection(_TANH_LEAF_PREFIXES))
    is_deltanet = bool(roots.intersection(_DELTANET_LEAF_PREFIXES))
    if is_tanh and is_deltanet:
        raise RuntimeError(
            f"[serve-pi-workspace] {WSM_COND_SUBTREE} mixes tanh and gated_deltanet leaves "
            f"({sorted(roots)}) — refusing to guess."
        )
    if not (is_tanh or is_deltanet):
        raise RuntimeError(
            f"[serve-pi-workspace] {WSM_COND_SUBTREE} matches no known conditioner; leaves={sorted(roots)}"
        )
    if is_tanh:
        spec = WSMCondSpec(cond_type="tanh", window=1)
    else:
        missing = [name for name in _DELTANET_LEAF_PREFIXES if name not in roots]
        if missing:
            raise RuntimeError(f"[serve-pi-workspace] gated_deltanet {WSM_COND_SUBTREE} is missing {missing}")
        marker = tuple(leaf_shapes["pos_decay_bias"])
        if len(marker) != 2 or marker[0] < 1 or marker[1] < 1:
            raise RuntimeError(f"[serve-pi-workspace] pos_decay_bias must be [window_len, num_heads], got {marker}")
        window, num_heads = int(marker[0]), int(marker[1])
        q_kernel = tuple(leaf_shapes["proj_q/kernel"])
        if len(q_kernel) != 2 or q_kernel[1] % num_heads:
            raise RuntimeError(
                f"[serve-pi-workspace] proj_q/kernel {q_kernel} is not divisible into {num_heads} heads"
            )
        # PTRM is the deltanet tree PLUS the recursive core. A PARTIAL core is never a deltanet
        # checkpoint with junk in it — it is a truncated/corrupt PTRM tree, and serving it as a
        # deltanet would silently drop the whole recursion the arm is about.
        ptrm_present = [name for name in _PTRM_LEAF_PREFIXES if name in roots]
        if ptrm_present and len(ptrm_present) != len(_PTRM_LEAF_PREFIXES):
            raise RuntimeError(
                f"[serve-pi-workspace] PTRM {WSM_COND_SUBTREE} is missing "
                f"{[name for name in _PTRM_LEAF_PREFIXES if name not in roots]}"
            )
        ptrm_steps = None
        if ptrm_present:
            depth = tuple(leaf_shapes["step_bias"])
            if len(depth) != 2 or depth[0] < 1:
                raise RuntimeError(f"[serve-pi-workspace] step_bias must be [steps, cond_dim], got {depth}")
            ptrm_steps = int(depth[0])
        spec = WSMCondSpec(
            cond_type="gated_deltanet_ptrm" if ptrm_present else "gated_deltanet",
            window=window,
            num_heads=num_heads,
            head_dim=q_kernel[1] // num_heads,
            ptrm_steps=ptrm_steps,
        )
    if requested is not None and requested != spec.cond_type:
        raise RuntimeError(
            f"[serve-pi-workspace] WSM_COND_TYPE={requested!r} but the checkpoint's "
            f"{WSM_COND_SUBTREE} is {spec.cond_type!r}. Refusing to serve a mismatched conditioner."
        )
    return spec


def read_wsm_cond_leaf_shapes(checkpoint_dir: str | pathlib.Path) -> dict[str, tuple[int, ...]]:
    """Peek the checkpoint's params tree WITHOUT a template (orbax metadata), before model build.

    `_model.restore_params` reads the same `metadata(...).item_metadata` mapping, so this uses the
    orbax API the codebase already depends on. It must run BEFORE construction: `BaseModelConfig.load`
    intersect-tree-loads against an ALREADY-BUILT model, so a mis-guessed conditioner would either
    hard-fail on a shape check or (worse) leave a freshly initialized subtree in place — the
    from_pretrained-bypass failure this file's assertions exist to catch.
    """
    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import download

    resolved = download.maybe_download(str(pathlib.Path(checkpoint_dir).expanduser()))
    params_path = pathlib.Path(resolved) / "params"
    with ocp.PyTreeCheckpointer() as ckptr:
        tree = ckptr.metadata(params_path).item_metadata["params"]
    shapes: dict[str, tuple[int, ...]] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = [str(getattr(k, "key", getattr(k, "name", k))) for k in path]
        if not keys or keys[0] != WSM_COND_SUBTREE:
            continue
        if keys[-1] == "value":  # nnx.State suffix, stripped exactly as restore_params does
            keys = keys[:-1]
        shapes["/".join(keys[1:])] = tuple(int(d) for d in leaf.shape)
    return shapes


def detect_wsm_cond_spec(checkpoint_dir: str | pathlib.Path) -> WSMCondSpec:
    """Checkpoint-driven conditioner selection; WSM_COND_TYPE is an explicit, checked override."""
    requested = os.environ.get("WSM_COND_TYPE") or None
    if requested is not None and requested not in WSM_COND_TYPES:
        raise RuntimeError(f"[serve-pi-workspace] WSM_COND_TYPE must be one of {WSM_COND_TYPES}, got {requested!r}")
    spec = classify_wsm_cond_subtree(read_wsm_cond_leaf_shapes(checkpoint_dir), requested=requested)
    print(
        f"[serve-pi-workspace] ✓ {WSM_COND_SUBTREE} auto-detected as {spec.describe()}"
        + (f" (WSM_COND_TYPE={requested} override agrees)" if requested else ""),
        flush=True,
    )
    return spec


def build_wsm_cfg_policy(
    finetune_ckpt: str, config_name: str, k_window: int, max_token_len: int, guidance_scale: float, p_drop: float
):
    """Build the pi0.5 WSM-CFG policy from the finetune ckpt with wsm_cfg=True + the trace-time guidance
    scale baked in. Mirrors serve_pi_05_wsm.build_wsm_policy: rebuild the matching TrainConfig directly
    (the dynamic train-time config_name is not in the static registry) with EMPTY data_dirs (norm stats
    come from the ckpt assets/). create_trained_policy intersect-tree-loads the wsm_cfg_cond subtree."""
    import os

    import openpi.models.pi0_config as pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as _config

    model = pi0_config.Pi0Config(
        pi05=True,
        max_token_len=int(max_token_len),
        wsm_cfg=True,
        wsm_cfg_p_drop=float(p_drop),
        wsm_cfg_with_future=False,
        wsm_cfg_guidance_scale=float(guidance_scale),
    )
    cfg = _config.TrainConfig(
        name=config_name,
        exp_name="pi05_rc365_wsm_cfg_ft",
        model=model,
        data=_config.LeRobotRobocasaDataConfig(data_dirs=[]),
    )
    print(
        f"[serve-pi-cfg] config={config_name} pi05={model.pi05} wsm_cfg={model.wsm_cfg} "
        f"K={int(k_window)} guidance_scale={model.wsm_cfg_guidance_scale} "
        f"max_token_len={model.max_token_len}",
        flush=True,
    )
    policy = policy_config.create_trained_policy(cfg, os.path.expanduser(finetune_ckpt))
    return policy


def build_stage_s_workspace_policy(
    finetune_ckpt: str,
    config_name: str,
    max_token_len: int,
    guidance_scale: float,
    p_drop: float,
    interface: str,
    cond_spec: WSMCondSpec | None = None,
    ptrm_knobs: PTRMEvalKnobs | None = None,
):
    """Rebuild a new Stage-S CFG2 or tanh policy without enabling legacy token injection.

    `cond_spec` is the checkpoint-detected wsm_tanh_cond architecture (tanh vs gated_deltanet vs
    gated_deltanet_ptrm plus its window/head/depth geometry). It is required for the tanh interface so
    the rebuilt module matches the trained one exactly; cfg2 never sees it.

    `ptrm_knobs` is the eval-only PTRM procedure. It rides the SAME Pi0Config kwargs as the geometry
    on purpose: a knob applied after `create_trained_policy` would be attached to an object the traced
    graph never consults, which is exactly the from_pretrained-bypass failure this file guards against.
    """
    import os

    import openpi.models.pi0_config as pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as _config

    if interface not in {"cfg2", "tanh"}:
        raise ValueError(f"expected cfg2 or tanh, got {interface!r}")
    if interface == "tanh" and float(guidance_scale) != 1.0:
        raise ValueError("tanh has no guidance sweep; use --guidance-scale 1.0")
    if interface == "tanh" and cond_spec is None:
        raise ValueError("the tanh interface requires a checkpoint-detected cond_spec")
    cond_kwargs = (
        {}
        if cond_spec is None or cond_spec.cond_type == "tanh"
        else {
            "wsm_cond_type": cond_spec.cond_type,
            "wsm_cond_window": cond_spec.window,
            "wsm_cond_num_heads": cond_spec.num_heads,
            "wsm_cond_head_dim": cond_spec.head_dim,
        }
    )
    if cond_spec is not None and cond_spec.cond_type == "gated_deltanet_ptrm":
        knobs = ptrm_knobs or PTRMEvalKnobs(k=1, sigma=0.0, select="q")
        cond_kwargs.update(
            {
                "wsm_ptrm_steps": cond_spec.ptrm_steps,
                "wsm_ptrm_eval_k": knobs.k,
                "wsm_ptrm_eval_sigma": knobs.sigma,
                "wsm_ptrm_eval_select": knobs.select,
            }
        )
    elif ptrm_knobs is not None and not ptrm_knobs.is_default:
        # A PTRM knob on a non-PTRM checkpoint is a mislabeled eval cell, not a no-op. `is_default`
        # rather than `deterministic` so ZERO_COND — a deterministic cell, but a DIFFERENT
        # experiment — cannot slip through on a checkpoint whose conditioner it does not patch.
        raise ValueError(
            f"PTRM eval knobs ({ptrm_knobs.describe()}) were set, but the checkpoint holds "
            f"{cond_spec.describe() if cond_spec else 'no workspace conditioner'}"
        )
    model = pi0_config.Pi0Config(
        pi05=True,
        max_token_len=int(max_token_len),
        wsm_cfg2=interface == "cfg2",
        wsm_tanh=interface == "tanh",
        wsm_cfg_p_drop=float(p_drop),
        wsm_cfg_guidance_scale=float(guidance_scale),
        **cond_kwargs,
    )
    cfg = _config.TrainConfig(
        name=config_name,
        exp_name=f"pi05_rc365_workspace_{interface}",
        model=model,
        data=_config.LeRobotRobocasaDataConfig(data_dirs=[]),
    )
    print(
        f"[serve-pi-workspace] config={config_name} interface={interface} "
        f"guidance_scale={model.wsm_cfg_guidance_scale} max_token_len={model.max_token_len}"
        + (f" cond={cond_spec.describe()}" if cond_spec is not None else "")
        + (
            f" ptrm_eval=({model.wsm_ptrm_eval_k}, {model.wsm_ptrm_eval_sigma:g}, {model.wsm_ptrm_eval_select})"
            if model.wsm_cond_type == "gated_deltanet_ptrm"
            else ""
        )
        # Appended, never folded into the ptrm_eval=(...) token: the sealed E0/E1/E2 cells gate on
        # that exact string, and a cell whose label format shifted under it could not be compared
        # against them. zero_cond announces itself as its own token or not at all.
        + (" zero_cond=ON" if ptrm_knobs is not None and ptrm_knobs.zero_cond else ""),
        flush=True,
    )
    return policy_config.create_trained_policy(cfg, os.path.expanduser(finetune_ckpt))


def build_stage_q_combined_policy(finetune_ckpt: str, config_name: str, max_token_len: int):
    """Rebuild the Q3 policy: pi05 + wsm_tanh (workspace read) + robottt (fast weights) — the two
    Stage-Q 2x2 booleans both on; every other RoboTTT hyperparameter is fixed across the 2x2 (07a),
    so serve needs no per-run knobs. Norm stats come from the checkpoint's assets/.
    expose_norm_actions=True is REQUIRED: the wrapper's fast-weight commit consumes the result's
    model-space norm_state/norm_actions rows (never the unnormalized client actions)."""
    import os

    import openpi.models.pi0_config as pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as _config

    model = pi0_config.Pi0Config(pi05=True, max_token_len=int(max_token_len), wsm_tanh=True, robottt=True)
    cfg = _config.TrainConfig(
        name=config_name,
        exp_name="pi05_rc365_stage_q_q3_serve",
        model=model,
        data=_config.LeRobotRobocasaDataConfig(data_dirs=[]),
    )
    print(
        f"[serve-pi-workspace] config={config_name} pi05={model.pi05} wsm_tanh={model.wsm_tanh} "
        f"robottt={model.robottt} max_token_len={model.max_token_len}",
        flush=True,
    )
    return policy_config.create_trained_policy(cfg, os.path.expanduser(finetune_ckpt), expose_norm_actions=True)


# Serve rebuilds the tanh conditioner with the Pi0Config default gate init (1e-3); training pins the
# same value (submit_pi_stage_s WSM_TANH_GATE_INIT default 0.001). assert_tanh_cond_trained compares
# the restored alpha against this constant to detect a re-initialized (unloaded) subtree.
TANH_GATE_INIT = 1e-3


def assert_tanh_cond_trained(policy, gate_init: float = TANH_GATE_INIT) -> None:
    """Refuse to serve unless the TRAINED wsm_tanh_cond subtree was actually restored.

    Mirrors serve_pi_05_robottt.assert_robottt_loaded: the tanh gate alpha is initialized to the
    constant gate_init everywhere, so an alpha that is still exactly that constant means the subtree
    was re-initialized instead of loaded (the from_pretrained-bypass failure mode) — the workspace
    read would then be a meaningless near-zero modulation and the eval would silently score ~S0."""
    import jax.numpy as jnp

    model = policy._model
    if not getattr(model, "wsm_tanh", False) or not hasattr(model, "wsm_tanh_cond"):
        raise RuntimeError("[serve-pi-workspace] model has no wsm_tanh_cond subtree; wrong config")
    alpha = jnp.asarray(model.wsm_tanh_cond.alpha.value)
    if bool(jnp.all(alpha == float(gate_init))):
        raise RuntimeError(
            "[serve-pi-workspace] wsm_tanh_cond.alpha is still exactly gate_init everywhere "
            "(untrained init) — the trained workspace read was NOT restored. Refusing to serve."
        )
    print(
        f"[serve-pi-workspace] ✓ wsm_tanh_cond restored (alpha departed from gate_init={gate_init:g})",
        flush=True,
    )


def assert_stage_s_conditioner_loaded(policy, interface: str, guidance_scale: float) -> None:
    """Validate the selected Stage-S subtree and reject an untrained zero-init CFG2 checkpoint."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    spec = {
        "cfg2": ("wsm_cfg2", "wsm_cfg2_cond"),
        "tanh": ("wsm_tanh", "wsm_tanh_cond"),
    }
    flag, attr = spec[interface]
    if not getattr(model, flag, False) or not hasattr(model, attr):
        raise RuntimeError(f"[serve-pi-workspace] model is missing required {attr} subtree")
    if float(getattr(model, "guidance_scale", -1.0)) != float(guidance_scale):
        raise RuntimeError("[serve-pi-workspace] trace-time guidance scale mismatch")
    cond = getattr(model, attr)
    arrays = [jnp.asarray(v.value) for _path, v in nnx.state(cond, nnx.Param).flat_state()]
    if not arrays or any(not bool(jnp.isfinite(a).all()) for a in arrays):
        raise RuntimeError(f"[serve-pi-workspace] {attr} parameters are missing or non-finite")
    # The tanh subtree holds either conditioner variant; report whichever output projection exists
    # (tanh: proj_t_out; gated_deltanet: proj_readout).
    readout_name = next((n for n in ("proj_t_out", "proj_readout") if hasattr(cond, n)), None)
    if readout_name is None:
        raise RuntimeError(f"[serve-pi-workspace] {attr} has no recognized output projection")
    out_l2 = float(jnp.linalg.norm(jnp.asarray(getattr(cond, readout_name).kernel.value)))
    if interface == "cfg2" and out_l2 == 0.0:
        raise RuntimeError("[serve-pi-workspace] wsm_cfg2_cond is still zero-init; refusing to serve a baseline")
    gate = float(jnp.max(jnp.abs(jnp.tanh(cond.alpha.value)))) if interface == "tanh" else None
    print(
        f"[serve-pi-workspace] ✓ {attr} present, finite, {readout_name} L2={out_l2:.4g}"
        + (f", max|tanh(alpha)|={gate:.4g}" if gate is not None else ""),
        flush=True,
    )


def assert_cfg_conditioner_loaded(policy, guidance_scale: float) -> None:
    """Fail loudly unless the served JAX model has wsm_cfg enabled, the conditioner's params are all finite,
    and its zero-init output projection (proj_t_out) departed from all-zeros — i.e. the TRAINED conditioner
    was actually restored. A zero proj_t_out kernel == cond is identically 0 == the baseline policy (the
    'serves as baseline' failure we must never repeat). Also sanity-check the trace-time guidance scale."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = policy._model
    if not getattr(model, "wsm_cfg", False) or not hasattr(model, "wsm_cfg_cond"):
        raise RuntimeError(
            "[serve-pi-cfg] served model has NO wsm_cfg_cond — refusing to serve a baseline "
            "policy (was the config built with wsm_cfg=True?)"
        )
    if float(getattr(model, "guidance_scale", -1.0)) != float(guidance_scale):
        raise RuntimeError(
            f"[serve-pi-cfg] model.guidance_scale={getattr(model, 'guidance_scale', None)} != "
            f"requested {guidance_scale} — the trace-time constant did not take; rebuild."
        )
    cond = model.wsm_cfg_cond
    arrays = [jnp.asarray(v.value) for _path, v in nnx.state(cond, nnx.Param).flat_state()]
    n_bad = sum(int(not bool(jnp.isfinite(a).all())) for a in arrays)
    if n_bad:
        raise RuntimeError(
            f"[serve-pi-cfg] wsm_cfg_cond has NON-FINITE params ({n_bad}/{len(arrays)} bad) — diverged ckpt; refusing."
        )
    out_l2 = float(jnp.linalg.norm(jnp.asarray(cond.proj_t_out.kernel.value)))
    if out_l2 == 0.0:
        raise RuntimeError(
            "[serve-pi-cfg] wsm_cfg_cond.proj_t_out kernel is ALL ZERO (untrained zero-init) "
            "-> cond is identically 0 == the BASELINE policy. Refusing to serve. (Did "
            "create_trained_policy load the finetune ckpt's conditioner?)"
        )
    print(
        f"[serve-pi-cfg] ✓ conditioner present: {len(arrays)} param tensors, all finite, "
        f"proj_t_out-kernel L2={out_l2:.4g} (>0 => trained), guidance_scale={guidance_scale}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-ckpt", required=True, help="pi workspace finetune ckpt (orbax step dir)")
    ap.add_argument(
        "--tap-ckpt",
        required=True,
        help="frozen π feature-source ckpt; must match the WSM encoder/cache encoder_id provenance",
    )
    ap.add_argument(
        "--encoder-ckpt",
        required=True,
        help="FROZEN WorkspaceModel ckpt (wsm_step*.pt), or a Stage-E encoder.pt with --encoder-kind stage_e",
    )
    # required=True for wsm_v1 (checked below with the same message argparse would give); a Stage-E
    # serve derives the table from the ω store and may omit it (or pass it as a cross-check).
    ap.add_argument("--task-lang-table", default=None, help="task_lang_table.npz (make_task_lang_table)")
    ap.add_argument(
        "--encoder-kind",
        default="wsm_v1",
        choices=("wsm_v1", "stage_e"),
        help="wsm_v1 (default; the sealed WorkspaceModel encoder) or stage_e (H14 Stage-E encoder: "
        "frozen PatchPool -> domain adapter -> shared trunk, see stage_e_serve.py)",
    )
    ap.add_argument(
        "--pool-ckpt",
        default=None,
        help="stage_e: the frozen WSMv1 pool checkpoint the wsm_pooled store was built with",
    )
    ap.add_argument("--stage-e-domain", default="remembench")
    ap.add_argument(
        "--stage-e-omega-root",
        default=None,
        help="stage_e: the ω store the policy trained on (.../omega/<cell>/remembench); "
        "source of the per-task language table and the encoder_id/step cross-check",
    )
    ap.add_argument(
        "--stage-e-pooled-root",
        default=None,
        help="stage_e: wsm_pooled store (p.npz) for the startup parity self-test",
    )
    ap.add_argument(
        "--stage-e-parity-demos",
        type=int,
        default=0,
        help="stage_e: replay N store demos through the serve stack before serving; "
        "refuse on a D7 FAIL (needs --stage-e-omega-root and --stage-e-pooled-root)",
    )
    ap.add_argument(
        "--stage-e-lang-table-mode",
        default="strict",
        choices=("strict", "task_mean_of_store"),
        help="stage_e: strict = the store must be serve-consistent (task_mean); "
        "task_mean_of_store = SMOKE ONLY on a per-episode store",
    )
    ap.add_argument(
        "--stage-e-table-out",
        default=None,
        help="stage_e: write the derived table (sealed task_lang_table.npz schema)",
    )
    ap.add_argument(
        "--expect-encoder-sha256", default=None, help="stage_e: refuse unless --encoder-ckpt has this sha256"
    )
    ap.add_argument(
        "--fork-dataset-py",
        default=None,
        help="stage_e: the fork's groot_openpi_dataset.py for the window-rule lock-step "
        "(default: the importable openpi tree)",
    )
    ap.add_argument(
        "--interface",
        choices=("legacy_cfg", "cfg2", "tanh", "tanh_robottt"),
        required=True,
        help="required: legacy_cfg reproduces old checkpoints; new Stage-S runs use cfg2 or tanh; "
        "Stage-Q Q3 uses tanh_robottt (tanh workspace read + online RoboTTT fast weights)",
    )
    ap.add_argument(
        "--guidance-scale",
        type=float,
        required=True,
        help="CFG scale s; s=0 and s=1 each use one suffix pass. Tanh requires 1.0.",
    )
    ap.add_argument("--config-name", default=None, help="TrainConfig provenance label")
    ap.add_argument("--configs-dir", default=None, help="dir holding wsm_robocasa_configs.py for the TAP")
    ap.add_argument("--max-token-len", type=int, default=200)
    ap.add_argument("--p-drop", type=float, default=0.2, help="CFG drop probability (eval architecture field)")
    ap.add_argument(
        "--k-window",
        type=int,
        default=1,
        help="new cfg2/tanh use newest omega_t only; pass 2 only for legacy_cfg reproduction",
    )
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--lang-dim", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=8, help="cache grid stride (advance grid by env step)")
    ap.add_argument(
        "--tap-prompt",
        default="expanded",
        choices=["expanded", "terse"],
        help=(
            "legacy_cfg may use historical per-task expanded prompts; new cfg2/tanh "
            "require 'terse', disable that lookup, and consume the canonical private "
            "wsm_prompt carried by every eval request"
        ),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.encoder_kind == "wsm_v1" and not args.task_lang_table:
        ap.error("the following arguments are required: --task-lang-table")
    if args.encoder_kind == "stage_e" and args.interface != "tanh":
        ap.error("--encoder-kind stage_e serves the tanh interface only (Stage-E ω arms are s1 recipes)")
    if args.interface != "legacy_cfg" and args.k_window != 1:
        raise ValueError(
            f"new Stage-S {args.interface} reads newest omega_t only; require --k-window 1, got {args.k_window}"
        )
    if args.interface != "legacy_cfg" and args.tap_prompt != "terse":
        raise ValueError(
            f"new Stage-S {args.interface} forbids historical demo-expanded tap prompts; "
            "require --tap-prompt terse plus the canonical private wsm_prompt manifest"
        )

    from openpi.serving import websocket_policy_server

    # Serve-only RoboTTT ablation, parsed ONCE before any model work (garbage tokens / missing
    # ROBOTTT_ABLATION_ACK=smoke must fail before a GPU is touched). Only tanh_robottt owns a
    # fast-weight lifecycle, so an ablation on any other interface is a mislabeled no-op.
    from vla_training.eval._robottt_ablation import ablation_from_env, probe_from_env

    ablation = ablation_from_env()
    probe = probe_from_env() if args.interface == "tanh_robottt" else None
    if ablation.active and args.interface != "tanh_robottt":
        raise SystemExit(
            f"[serve-pi-workspace] ROBOTTT_ABLATION={ablation.raw!r} set with "
            f"--interface {args.interface} (no fast weights); only tanh_robottt can be ablated."
        )

    from vla_training.eval._groot_wsm_eval import (
        WSMEvalConditioner,
        load_task_expanded_table,
        load_task_lang_table,
    )
    from workspace_models.features.generate_policy_features import load_wsm

    # 1. Build the historical CFG tree only when explicitly requested; new checkpoints select a separate
    #    current-only CFG2 or tanh tree and keep all VLM/prefix tokens untouched.
    config_name = (
        args.config_name
        or {
            "legacy_cfg": "pi05_robocasa_wsm_cfg_ft",
            "cfg2": "pi05_robocasa_workspace_stage_s",
            "tanh": "pi05_robocasa_workspace_stage_s",
            "tanh_robottt": "pi05_robocasa_stage_q_q3",
        }[args.interface]
    )
    print(
        f"[serve-pi-workspace] building interface={args.interface} from {args.finetune_ckpt} "
        f"(s={args.guidance_scale})",
        flush=True,
    )
    # Conditioner auto-detection, BEFORE any model is built: the wsm_tanh_cond subtree may hold either
    # the tanh MLP read or the gated-DeltaNet steering variant, and for the latter the trained window
    # length is structural (pos_decay_bias). The sealed eval submit gains no flags — the checkpoint is
    # the source of truth and WSM_COND_TYPE is a checked override.
    cond_spec = None
    ptrm_knobs = None
    serve_k_window = args.k_window
    if args.interface in ("tanh", "tanh_robottt"):
        cond_spec = detect_wsm_cond_spec(args.finetune_ckpt)
        if args.interface == "tanh_robottt" and cond_spec.cond_type != "tanh":
            raise SystemExit(
                f"[serve-pi-workspace] Q3 (tanh_robottt) is defined with the tanh conditioner; this "
                f"checkpoint holds {cond_spec.describe()}"
            )
        if cond_spec.cond_type in WSM_WINDOWED_COND_TYPES:
            # The online conditioner must ship exactly the window the recurrence was trained on. Its
            # padding is causal_window_indices — the SAME left-pad-by-repeating-the-oldest-grid-row
            # convention the train loader uses (groot_openpi_dataset._wsm_causal_window).
            serve_k_window = int(cond_spec.window)
            print(
                f"[serve-pi-workspace] {cond_spec.cond_type}: online omega window K={serve_k_window} "
                f"(checkpoint-derived; --k-window {args.k_window} applies to the tanh read only)",
                flush=True,
            )
        if cond_spec.cond_type == "gated_deltanet_ptrm":
            # Parsed BEFORE the model is built (garbage must fail before a GPU is touched) and stated
            # in the log, because K/sigma/selection are the experiment: an eval whose stdout does not
            # name them cannot be told apart from its own PTRM-off control after the fact.
            ptrm_knobs = ptrm_eval_knobs_from_env()
            print(
                f"[serve-pi-workspace] PTRM eval: {ptrm_knobs.describe()} over a depth-"
                f"{cond_spec.ptrm_steps} recursion",
                flush=True,
            )
            # BEFORE the model is built: module_jit freezes the module graph at construction, so a
            # conditioning ablation introduced afterwards would not be in the served graph.
            if ptrm_knobs.zero_cond:
                apply_zero_cond_patch()
        elif ptrm_eval_knobs_from_env().zero_cond:
            # ZERO_COND patches the PTRM conditioner specifically. On any other checkpoint it would
            # remove nothing while still labeling the cell as ablated.
            raise SystemExit(
                f"[serve-pi-workspace] WSM_PTRM_ZERO_COND=1 requires a PTRM checkpoint; this one "
                f"holds {cond_spec.describe()}"
            )

    if args.interface == "legacy_cfg":
        policy = build_wsm_cfg_policy(
            args.finetune_ckpt, config_name, args.k_window, args.max_token_len, args.guidance_scale, args.p_drop
        )
        assert_cfg_conditioner_loaded(policy, args.guidance_scale)
    elif args.interface == "tanh_robottt":
        if float(args.guidance_scale) != 1.0:
            raise ValueError("tanh_robottt has no guidance sweep; use --guidance-scale 1.0")
        policy = build_stage_q_combined_policy(args.finetune_ckpt, config_name, args.max_token_len)
        # Both trained subtrees must have actually been restored (from_pretrained-bypass trap):
        # the tanh workspace read AND the robottt_fast meta-learned parameters. Either one still at
        # its init means the eval would silently score a broken/half-conditioned policy.
        from vla_training.eval.serve_pi_05_robottt import assert_robottt_loaded

        assert_stage_s_conditioner_loaded(policy, "tanh", 1.0)
        assert_tanh_cond_trained(policy)
        assert_robottt_loaded(policy)
    else:
        policy = build_stage_s_workspace_policy(
            args.finetune_ckpt,
            config_name,
            args.max_token_len,
            args.guidance_scale,
            args.p_drop,
            args.interface,
            cond_spec,
            ptrm_knobs,
        )
        assert_stage_s_conditioner_loaded(policy, args.interface, args.guidance_scale)
        if args.interface == "tanh":
            assert_tanh_cond_trained(policy)
        # The trained-subtree assertions above still run FIRST and unmodified: the ablation must be
        # applied to a checkpoint that genuinely restored its conditioner, otherwise "zeroed" and
        # "never loaded" would be the same number.
        if ptrm_knobs is not None and ptrm_knobs.zero_cond:
            assert_zero_cond_active(policy)

    # 2. Frozen feature-source tap. S1/S2 full-finetune the action policy, so its backbone drifts and MUST
    #    NOT generate the WorkspaceModel inputs. The separately pinned tap checkpoint must be the exact
    #    H300+MG feature source recorded by the encoder_id/cache manifest.
    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

    tap_config = "pi05_rc_mg60_bal33"
    print(
        f"[serve-pi-workspace] loading frozen feature-source tap (config={tap_config}, ckpt={args.tap_ckpt})",
        flush=True,
    )
    tap = Pi05BackboneTap(
        args.tap_ckpt, config_name=tap_config, **({"configs_dir": args.configs_dir} if args.configs_dir else {})
    )

    # 3. Frozen WSM encoder + online-omega_t conditioner + task language table.
    if args.encoder_kind == "stage_e":
        # Stage-E front end on the SAME conditioner/wrapper/window-rule path; every check inside is
        # fail-closed (window-rule lock-step, encoder sha/id/step, dims, table consistency, optional
        # startup parity against the store the policy trained on).
        from vla_training.eval.stage_e_serve import build_stage_e_serve_stack

        stack = build_stage_e_serve_stack(
            encoder_ckpt=args.encoder_ckpt,
            pool_ckpt=args.pool_ckpt,
            domain=args.stage_e_domain,
            omega_root=args.stage_e_omega_root,
            task_lang_table=args.task_lang_table,
            lang_table_mode=args.stage_e_lang_table_mode,
            expect_sha256=args.expect_encoder_sha256,
            fork_dataset_py=args.fork_dataset_py,
            k_window=serve_k_window,
            stride=args.stride,
            device=args.device,
            interface=args.interface,
            parity_demos=args.stage_e_parity_demos,
            pooled_root=args.stage_e_pooled_root,
            table_out=args.stage_e_table_out,
        )
        encoder, conditioner, table = stack.front_end, stack.conditioner, stack.table
        if not tap_min_batch():
            print(
                "[stage-e-serve] NOTE: WSM_TAP_MIN_BATCH is unset. The wsm_pooled store was tapped at "
                "B=32; the kernel-matched pad (WSM_TAP_MIN_BATCH=8) reproduces it bit-exactly, an "
                "unpadded B=1 tap does not (omega_sidecar.py, measured 2026-08-08). Stage-E arms have "
                "no sealed unpadded number to protect; the cell runner sets 8.",
                flush=True,
            )
    else:
        encoder, _meta = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PI_PROPRIO_DIM)
        conditioner = WSMEvalConditioner(encoder, k_window=serve_k_window, stride=args.stride, device=args.device)
        table = load_task_lang_table(args.task_lang_table)

    expanded_table = None
    if args.tap_prompt == "expanded":
        expanded_table = load_task_expanded_table(args.task_lang_table)
        if not expanded_table or not any(expanded_table.values()):
            raise RuntimeError(
                "[serve-pi-cfg] --tap-prompt expanded but task_lang_table has no non-empty "
                "'expanded' strings — regenerate with `make_task_lang_table --cache-root`."
            )
        print(
            f"[serve-pi-cfg] tap-prompt=EXPANDED ({sum(bool(v) for v in expanded_table.values())}/"
            f"{len(expanded_table)} per-task strings present)",
            flush=True,
        )
    elif args.interface == "legacy_cfg":
        print(
            "[serve-pi-cfg] tap-prompt=TERSE (legacy environment-task fallback)",
            flush=True,
        )
    else:
        print(
            "[serve-pi-workspace] tap-prompt=CANONICAL PRIVATE wsm_prompt "
            "(demo-independent; required on every request)",
            flush=True,
        )

    # Q3 combined interface: attach the online fast-weight runner. The wrapper then runs BOTH halves
    # per chunk: workspace omega injection (identical to tanh) AND condition/commit on the per-env W
    # (identical to the Q2 robottt_fast serve). Fail-closed: the runner refuses a policy without the
    # robottt_fast subtree, and the wrapper refuses results without model-space norm rows.
    robottt_runner = None
    if args.interface == "tanh_robottt":
        from vla_training.eval._robottt_serve_runner import RoboTTTServeRunner

        robottt_runner = RoboTTTServeRunner(policy)
        if probe is not None:
            scalars = robottt_runner.trained_scalars()
            probe.log(
                "startup",
                ckpt=args.finetune_ckpt,
                config_name=config_name,
                interface=args.interface,
                stride=int(args.stride),
                ablation=ablation.spec,
                ablation_detail=ablation.as_metadata(),
                eta_effective=float(scalars["inner_lr_eta"]) * float(ablation.eta_scale),
                **scalars,
            )

    wrapped = WSMPiInferWrapper(
        policy,
        tap,
        conditioner,
        table,
        stride=args.stride,
        expanded_table=expanded_table,
        require_wsm_prompt=args.interface != "legacy_cfg",
        robottt_runner=robottt_runner,
        robottt_ablation=ablation if robottt_runner is not None else None,
        robottt_probe=probe,
    )
    print(
        f"[serve-pi-workspace] ✓ pi0.5 server ready on {args.host}:{args.port} "
        f"(interface={args.interface}, encoder_kind={args.encoder_kind}"
        f"{'/' + cond_spec.cond_type if cond_spec is not None else ''}"
        f"{' ' + ptrm_knobs.describe() if ptrm_knobs is not None else ''}, "
        f"s={args.guidance_scale}, K={serve_k_window}, "
        f"stride={args.stride}, {len(table)} tasks, "
        f"fast_weights={'ONLINE' if robottt_runner is not None else 'off'}, "
        f"robottt_ablation={ablation.describe()})",
        flush=True,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=wrapped.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
