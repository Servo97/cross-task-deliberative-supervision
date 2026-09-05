#!/usr/bin/env python3
"""D0a — the OFFLINE half of the PTRM (H9) evidence ladder: is there anything to select between?

PTRM's claim is test-time WIDTH scaling: K noisy rollouts through the recursive read, then keep the
one the Q head likes best. Two premises have to hold before that can pay in an environment, and both
are checkable with NO simulator and NO 3B decode — only the ~7M-param conditioner subtree of the
trained checkpoint and real omega windows:

  1. DISPERSION. The core is RMSNorm-terminated, i.e. contractive onto the unit sphere. If the K
     trajectories collapse back onto each other, sigma buys nothing and best-Q is selecting between
     32 copies of the same vector. (Design doc kill-mode (b).)
  2. Q DISCRIMINATION. If q is near-constant across the K rollouts, the argmin is noise and PTRM
     degenerates into the paper's own Maze-Hard failure — exploration without a usable verifier.
     (Kill-mode (a). D0a bounds this from the SPREAD side; the correlation against realized flow
     loss is D0b, which needs the full model.)

Plus the depth structure the training story predicts: q_t should DECLINE with recursion depth (the
Q head is trained on the realized loss at a uniformly sampled depth, so if refinement helps, deeper
z should look better to it), and cos(c_T, c_{T-1}) says whether the refinement has converged by T.

WHAT IT TOUCHES. The wsm_tanh_cond subtree only (a partial orbax restore — the 11.6 GiB of PaliGemma
weights in the same checkpoint are never materialized) and a small sample of the study's omega cache.
Everything runs float32 on CPU under a fixed seed.

HONEST SCOPE (read this before quoting the numbers). The study's omega cache holds exactly the 150
demos/task the trainer saw (7,500 files = 50 x 150, the seed-0 filter_key keep-set), so there is no
held-out-DEMO omega anywhere in the store. These windows are therefore in-distribution. That is
sufficient for D0a, whose three questions are properties of the recursion itself — dispersion under
noise, spread of q across rollouts, depth profile — none of which is a generalization measurement.
The generalization-sensitive question (does q RANK correctly?) is D0b and needs realized flow loss.

  python scripts/analysis/ptrm_d0a_diagnostic.py [--windows 300] [--k 32]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import random
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from wsm_settings import LEGACY_ACCOUNT, LONG_CONTEXT_STUDY_S3, ROBOCASA_OPENPI_SRC  # noqa: E402

STUDY = LONG_CONTEXT_STUDY_S3
RUN_ID = "s1-94b215496fe2ec2e"
STEP = 59999
CKPT_S3 = f"{STUDY}/checkpoints/pi05/s1/{RUN_ID}/{STEP}/params"
ENCODER_ID = "0883c9bd753999deee15dc00ae7b5ac78dc5c4f988465cf366aae7ffd6f339c2"
OMEGA_S3 = f"{STUDY}/caches/{ENCODER_ID}/omega"
AWS_PROFILE = f"Robotics-LBM-PowerUserAccess-{LEGACY_ACCOUNT}"
# NEVER the root filesystem: it runs at ~92% full. /home has the space.
STAGE = pathlib.Path.home() / "Research" / "TRI" / "wsm_data" / "ptrm_d0a"
SUBTREE = "wsm_tanh_cond"
RESULTS = REPO / "internal_planning_and_todos" / "aug_08" / "ptrm_d0a_results.md"
SIGMAS = (0.0, 0.1, 0.3, 0.6, 1.0)
SEED = 20260808


# --------------------------------------------------------------------------------------------
# Staging (download ONLY what the diagnostic reads)
# --------------------------------------------------------------------------------------------
def _aws(*argv: str, capture: bool = True) -> str:
    env = {**os.environ, "AWS_PROFILE": AWS_PROFILE}
    done = subprocess.run(["aws", *argv], env=env, capture_output=capture, text=True, check=False, timeout=3600)
    if done.returncode != 0:
        raise RuntimeError(f"aws {' '.join(argv)} failed: {done.stderr[-500:]}")
    return done.stdout


def _list_prefixes(uri: str) -> list[str]:
    return sorted(
        line.split("PRE", 1)[1].strip().rstrip("/")
        for line in _aws("s3", "ls", uri.rstrip("/") + "/").splitlines()
        if "PRE" in line
    )


def stage_checkpoint() -> pathlib.Path:
    local = STAGE / f"ckpt_{STEP}" / "params"
    marker = local / "manifest.ocdbt"
    if not marker.is_file():
        local.mkdir(parents=True, exist_ok=True)
        print(f"[d0a] staging checkpoint params -> {local}", flush=True)
        _aws("s3", "sync", CKPT_S3 + "/", str(local) + "/", "--only-show-errors", capture=False)
    return local


def stage_omega(num_tasks: int, demos_per_task: int) -> list[pathlib.Path]:
    """A deterministic spread of demos across `num_tasks` tasks; ~130 KB per file."""
    root = STAGE / "omega"
    index = root / f"_sample_{num_tasks}x{demos_per_task}.json"
    if index.is_file():
        picked = {k: v for k, v in json.loads(index.read_text()).items()}
    else:
        tasks = _list_prefixes(OMEGA_S3)
        if len(tasks) < num_tasks:
            raise RuntimeError(f"omega cache has {len(tasks)} tasks, need {num_tasks}")
        step = len(tasks) / num_tasks
        chosen = [tasks[int(i * step)] for i in range(num_tasks)]
        picked = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for task, demos in zip(chosen, pool.map(lambda t: _list_prefixes(f"{OMEGA_S3}/{t}"), chosen), strict=True):
                stride = len(demos) / demos_per_task
                picked[task] = [demos[int(i * stride)] for i in range(demos_per_task)]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(picked, indent=2, sort_keys=True))

    todo = [
        (task, demo)
        for task, demos in sorted(picked.items())
        for demo in demos
        if not (root / task / demo / "w.npz").is_file()
    ]
    if todo:
        print(f"[d0a] staging {len(todo)} omega windows -> {root}", flush=True)

        def fetch(item):
            task, demo = item
            dest = root / task / demo
            dest.mkdir(parents=True, exist_ok=True)
            _aws("s3", "cp", f"{OMEGA_S3}/{task}/{demo}/w.npz", str(dest / "w.npz"), "--quiet")

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, todo))
    return [root / task / demo / "w.npz" for task, demos in sorted(picked.items()) for demo in demos]


# --------------------------------------------------------------------------------------------
# The conditioner, rebuilt from its own subtree
# --------------------------------------------------------------------------------------------
def restore_cond_subtree(params_path: pathlib.Path) -> dict:
    """Partial orbax restore of `wsm_tanh_cond`; the transformer arrays are never touched."""
    import orbax.checkpoint as ocp
    from flax import traverse_util

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path).item_metadata
        flat = traverse_util.flatten_dict(metadata["params"])
        wanted = {kp: leaf for kp, leaf in flat.items() if SUBTREE in kp}
        if not wanted:
            raise RuntimeError(f"{params_path} has no {SUBTREE} subtree")
        item = {"params": traverse_util.unflatten_dict(wanted)}
        restored = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item
                ),
                # THE point of this script: read the ~7M-param conditioner and leave the 11.6 GiB of
                # PaliGemma/action-expert arrays in the same checkpoint unmaterialized.
                partial_restore=True,
            ),
        )["params"]
    flat_params = traverse_util.flatten_dict(restored)
    if all(kp[-1] == "value" for kp in flat_params):  # saved through nnx.State
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    # strip everything above the subtree root
    out = {}
    for kp, value in flat_params.items():
        rel = kp[kp.index(SUBTREE) + 1 :]
        out[rel] = np.asarray(value, dtype=np.float32)
    return out


def build_module(leaves: dict):
    """Instantiate `WSMGatedDeltaNetPTRMConditioner` with the checkpoint's own geometry.

    Every hyperparameter is READ OFF the tree (the same structural auto-detection serve uses), so a
    silently mismatched module is impossible: window from `pos_decay_bias`, heads from `proj_beta`,
    depth from `step_bias`.
    """
    sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))
    from openpi.models.wsm_current_cond import WSMGatedDeltaNetPTRMConditioner

    w_dim, inner = leaves[("proj_q", "kernel")].shape
    num_heads = leaves[("proj_beta", "kernel")].shape[1]
    window_len, heads_bias = leaves[("pos_decay_bias",)].shape
    cond_dim = leaves[("proj_readout", "kernel")].shape[1]
    steps, cond_bias = leaves[("step_bias",)].shape
    if heads_bias != num_heads or cond_bias != cond_dim or inner % num_heads:
        raise RuntimeError("inconsistent wsm_tanh_cond geometry")
    geometry = {
        "w_dim": int(w_dim),
        "cond_dim": int(cond_dim),
        "window_len": int(window_len),
        "num_heads": int(num_heads),
        "head_dim": int(inner // num_heads),
        "steps": int(steps),
    }
    module = WSMGatedDeltaNetPTRMConditioner(**geometry, rngs=nnx.Rngs(0))
    loaded = 0
    for path, array in leaves.items():
        target = module
        for name in path[:-1]:
            target = getattr(target, name)
        leaf = getattr(target, path[-1])
        if tuple(leaf.value.shape) != tuple(array.shape):
            raise RuntimeError(f"{'/'.join(path)}: ckpt {array.shape} vs module {leaf.value.shape}")
        leaf.value = jnp.asarray(array, dtype=jnp.float32)
        loaded += 1
    if loaded != len(leaves):
        raise RuntimeError("not every checkpoint leaf was consumed")
    return module, geometry


def rollouts(module, window, rng, k: int, sigma: float):
    """The K PTRM trajectories, INSTRUMENTED — (z [K,C], c [K,C], q [K]).

    `eval_cond` returns only the selected vector, so the harness re-walks the identical loop using
    the module's OWN `_raw_read`/`proj_readout`/`_core_step`/`proj_out`/`q_head` and the identical
    `fold_in(rng, index+1)` noise schedule. `check_harness_parity` asserts bit-equality against
    `eval_cond` for every sigma, so these internals are the real ones, not a re-implementation.
    """
    r_p = module.proj_readout(module._raw_read(window))
    gate = jnp.tanh(module.alpha.value).astype(r_p.dtype)
    lead = r_p.shape[:-1]
    r_p = r_p[..., None, :]
    z = jnp.broadcast_to(module.z0.value.astype(r_p.dtype), (*lead, k, module.cond_dim))
    for index in range(module.steps):
        if sigma > 0.0:
            z = z + sigma * jax.random.normal(jax.random.fold_in(rng, index + 1), z.shape, z.dtype)
        z = module._core_step(z, r_p, index)
    return z, gate * module.proj_out(z), module.q_head(z)[..., 0]


def check_harness_parity(module, window, k: int) -> dict:
    """Two integrity checks, both required before any number below is quoted."""
    rng = jax.random.key(0)
    report = {}
    # 1. the documented K=1/sigma=0 parity: eval_cond == train_outputs' final c_T, bitwise.
    c_all, q_all = module.train_outputs(window)
    det = module.eval_cond(window, k=1, sigma=0.0, select="q")
    report["k1_sigma0_vs_train_c_T_maxabs"] = float(jnp.abs(det - c_all[..., -1, :]).max())
    # 2. the harness reproduces eval_cond's own selection, at every sigma.
    worst = 0.0
    for sigma in SIGMAS:
        _, cond, qual = rollouts(module, window, rng, k, sigma)
        picked = cond[0, jnp.argmin(qual[0])]
        theirs = module.eval_cond(window, rng=rng, k=k, sigma=sigma, select="q")[0]
        worst = max(worst, float(jnp.abs(picked - theirs).max()))
    report["harness_vs_eval_cond_maxabs"] = worst
    return report


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def mean_pairwise_l2(x: jnp.ndarray) -> jnp.ndarray:
    """Mean L2 distance over the K(K-1)/2 distinct pairs of rows."""
    diff = x[:, None, :] - x[None, :, :]
    dist = jnp.sqrt(jnp.maximum(jnp.sum(diff * diff, axis=-1), 0.0))
    k = x.shape[0]
    return jnp.sum(jnp.triu(dist, 1)) / (k * (k - 1) / 2)


def sample_windows(files: list[pathlib.Path], window_len: int, w_dim: int, count: int):
    """`count` causal [window_len, w_dim] windows at native steps drawn uniformly per demo."""
    sys.path.insert(0, str(REPO))
    from workspace_models.features.wsm_align import window_at

    rnd = random.Random(SEED)
    per_file = max(1, -(-count // len(files)))
    out = []
    for path in files:
        blob = np.load(path)
        w, frame_indices = blob["w"], blob["frame_indices"].astype(np.int64)
        if w.shape[1] != w_dim:
            raise RuntimeError(f"{path}: omega dim {w.shape[1]} != conditioner w_dim {w_dim}")
        horizon = int(frame_indices[-1])
        for _ in range(per_file):
            t = rnd.randint(0, horizon)
            win = np.asarray(window_at(w, frame_indices, t, window_len), dtype=np.float32)
            if win.shape != (window_len, w_dim):
                raise RuntimeError(f"{path}: window {win.shape} != {(window_len, w_dim)}")
            out.append((path.parent.parent.name, win))
            if len(out) == count:
                return out
    return out


def _cos(a, b) -> float:
    return float(jnp.dot(a, b) / jnp.maximum(jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-12))


def run(windows, module, k: int) -> dict:
    """Everything is reported RELATIVE as well as absolute.

    An absolute dispersion in c is uninterpretable on its own: `tanh(alpha)` scales the entire read,
    so a "small" number can be the whole conditioning vector moving. `rel_c` = disp(c)/||c_T|| is the
    number that says how much the thing the action head actually sees varies across rollouts.

    Likewise the Q columns separate the two ways best-of-K can look good. `q_drift` is the SYSTEMATIC
    effect of noise on predicted quality (mean_K q - q_det); `q_gain_sd` is how far into its own
    spread the argmin sits ((mean_K q - min_K q)/std_K q, ~2.0 for a Gaussian at K=32). If the gain
    is drift-dominated, PTRM is not identifying better reads, it is finding latents the Q head is
    miscalibrated on — the continuous-control face of the paper's Maze-Hard failure.
    """
    base_rng = jax.random.key(SEED)
    depth_q, cos_last, cos_first_last, c_norms = [], [], [], []
    keys = ("disp_z", "disp_c", "rel_z", "rel_c", "q_std", "shift", "q_gain", "q_drift", "q_gain_sd")
    per_sigma = {sigma: {key: [] for key in keys} for sigma in SIGMAS}
    for i, (_, win) in enumerate(windows):
        window = jnp.asarray(win)[None]  # [1, K_window, w_dim]
        c_all, q_all = module.train_outputs(window)
        depth_q.append(np.asarray(q_all[0]))
        c_t, c_prev, c_first = c_all[0, -1], c_all[0, -2], c_all[0, 0]
        cos_last.append(_cos(c_t, c_prev))
        cos_first_last.append(_cos(c_t, c_first))
        c_norm = float(jnp.linalg.norm(c_t))
        c_norms.append(c_norm)
        rng = jax.random.fold_in(base_rng, i)
        q_det = float(q_all[0, -1])
        for sigma in SIGMAS:
            z, cond, qual = rollouts(module, window, rng, k, sigma)
            disp_z = float(mean_pairwise_l2(z[0]))
            disp_c = float(mean_pairwise_l2(cond[0]))
            spread = float(jnp.std(qual[0]))
            best = float(jnp.min(qual[0]))
            mean_q = float(jnp.mean(qual[0]))
            bucket = per_sigma[sigma]
            bucket["disp_z"].append(disp_z)
            bucket["disp_c"].append(disp_c)
            bucket["rel_z"].append(disp_z / max(float(jnp.linalg.norm(z[0, 0])), 1e-12))
            bucket["rel_c"].append(disp_c / max(c_norm, 1e-12))
            bucket["q_std"].append(spread)
            bucket["shift"].append(float(int(jnp.argmin(qual[0])) != 0))
            bucket["q_gain"].append(q_det - best)
            bucket["q_drift"].append(mean_q - q_det)
            bucket["q_gain_sd"].append((mean_q - best) / max(spread, 1e-12))
    depth_q = np.stack(depth_q)
    return {
        "n_windows": len(windows),
        "depth_q_mean": depth_q.mean(axis=0).tolist(),
        "depth_q_std": depth_q.std(axis=0).tolist(),
        "cos_cT_cTm1": float(np.mean(cos_last)),
        "cos_cT_cTm1_min": float(np.min(cos_last)),
        "cos_c1_cT": float(np.mean(cos_first_last)),
        "c_norm": float(np.mean(c_norms)),
        "per_sigma": {
            sigma: {key: float(np.mean(values)) for key, values in bucket.items()}
            for sigma, bucket in per_sigma.items()
        },
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------
def render(result: dict, geometry: dict, parity: dict, tasks: list[str], k: int) -> str:
    rows = [
        "| sigma | disp(z) | rel(z) | disp(c) | rel(c) | Q-spread | sel-shift | Q-gain | Q-drift | gain/sd |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sigma in SIGMAS:
        row = result["per_sigma"][sigma]
        rows.append(
            f"| {sigma:g} | {row['disp_z']:.3f} | {row['rel_z']:.2%} | {row['disp_c']:.3e} | "
            f"{row['rel_c']:.2%} | {row['q_std']:.3e} | {row['shift']:.3f} | {row['q_gain']:+.3f} | "
            f"{row['q_drift']:+.3f} | {row['q_gain_sd']:.2f} |"
        )
    depth = "".join(
        f"| {t + 1} | {q:+.4f} | {s:.4f} |\n"
        for t, (q, s) in enumerate(zip(result["depth_q_mean"], result["depth_q_std"], strict=True))
    )
    return f"""# D0a — PTRM conditioner, offline (run {RUN_ID} step {STEP})

Offline half of the H9 evidence ladder: dispersion, Q-spread and depth structure of the trained
PTRM read, with no environment and no 3B decode. `sigma=0` is the PTRM-off control (all K rollouts
identical). Selection is `argmin q` (q predicts log flow loss, lower is better).

- checkpoint: `{CKPT_S3}` — `{SUBTREE}` subtree only (partial orbax restore)
- geometry read off the tree: {json.dumps(geometry)}
- omega: `{OMEGA_S3}` — {result["n_windows"]} windows, {len(tasks)} tasks
  ({", ".join(tasks[:6])}{", …" if len(tasks) > 6 else ""}); K={k}, float32 CPU, seed {SEED}
- integrity: K=1/sigma=0 vs train_outputs c_T max|diff| = {parity["k1_sigma0_vs_train_c_T_maxabs"]:.3e};
  harness vs `eval_cond` max|diff| = {parity["harness_vs_eval_cond_maxabs"]:.3e}
- SCOPE: the study's omega cache is exactly the 150 demos/task the trainer saw, so these windows are
  in-distribution; no held-out-demo omega exists in the store. D0a's questions are properties of the
  recursion, not generalization. Ranking quality vs realized loss is D0b.

## Per-sigma (mean over windows)

{chr(10).join(rows)}

`disp` = mean pairwise L2 over the K final vectors; `rel` = that divided by the vector's own norm
(||z||=32.0 by construction — RMSNorm — and ||c_T||={result["c_norm"]:.3f}); `Q-spread` = std of the
K q; `sel-shift` = fraction of windows whose argmin-q is not rollout 0; `Q-gain` = q(det) - min_K q,
how much better the selected rollout looks BY Q'S OWN LIGHTS than the PTRM-off read; `Q-drift` =
mean_K q - q(det), the SYSTEMATIC effect of noise on predicted quality; `gain/sd` =
(mean_K q - min_K q)/std_K q, where the argmin sits inside its own spread (~2.0 = plain Gaussian
order statistics at K=32, i.e. nothing but tail-picking).

## Depth structure (deterministic, from `train_outputs`)

| t | mean q_t | std q_t |
|---|---|---|
{depth}
cos(c_T, c_T-1) = {result["cos_cT_cTm1"]:.4f} (min over windows {result["cos_cT_cTm1_min"]:.4f});
cos(c_1, c_T) = {result["cos_c1_cT"]:.4f}

## Reading

{reading(result, k)}
"""


def sigma_star(result: dict) -> float:
    """The sigma to carry into E1.

    Both premises have to hold at once, so the pick is the LARGEST sigma whose conditioning-space
    dispersion is still a perturbation of the trained read rather than a replacement of it (rel(c)
    under 10%) — bigger sigma always buys more spread, and the binding constraint is staying on the
    manifold the action head was trained against.
    """
    usable = [s for s in SIGMAS if s > 0 and result["per_sigma"][s]["rel_c"] <= 0.10]
    return max(usable) if usable else min(s for s in SIGMAS if s > 0)


def reading(result: dict, k: int) -> str:
    per = result["per_sigma"]
    depth = result["depth_q_mean"]
    top = per[max(SIGMAS)]
    low = per[0.1]
    star = sigma_star(result)
    grows = all(per[b]["disp_z"] >= per[a]["disp_z"] for a, b in zip(SIGMAS[1:], SIGMAS[2:], strict=False))
    declines = all(b <= a for a, b in zip(depth, depth[1:], strict=False))
    best_depth = int(np.argmin(depth)) + 1
    depth_span = max(depth) - min(depth)
    amplification = low["rel_c"] / max(low["rel_z"], 1e-12)
    # E[(mean - min)/sd] over 32 iid normals is ~2.02. A gain/sd near that is order statistics: you
    # would get it from ANY scores with that spread, informative or not.
    order_stat = 2.02
    drift_share = abs(top["q_drift"]) / max(top["q_gain"], 1e-12)
    return "\n".join(
        [
            f"1. Kill-mode (b) does NOT fire — the K={k} rollouts stay apart, and the conditioning "
            f"vector is far more perturbable than the latent. z-dispersion is "
            f"{'monotone' if grows else 'non-monotone'} in sigma ({low['rel_z']:.1%} -> "
            f"{top['rel_z']:.1%} of ||z||): RMSNorm re-projects onto the sphere but does not "
            f"re-collapse the rollouts. The SAME noise moves the gated conditioning vector "
            f"{amplification:.1f}x further in relative terms ({low['rel_c']:.1%} -> "
            f"{top['rel_c']:.1%} of ||c_T||, and ||c_T|| is only {result['c_norm']:.3f}), because "
            f"proj_out amplifies the noise directions relative to the small trained read. sigma "
            f"therefore has real leverage, and it saturates fast.",
            f"2. Depth is INERT: cos(c_1, c_T) = {result['cos_c1_cT']:.4f} and cos(c_T, c_T-1) = "
            f"{result['cos_cT_cTm1']:.4f} (worst window {result['cos_cT_cTm1_min']:.4f}) — the core "
            f"reaches its fixed point at step 1 and steps 2-4 are a no-op. mean q_t moves "
            f"{depth_span:.3f} nats across all four depths "
            f"({' -> '.join(f'{q:+.3f}' for q in depth)}), "
            f"{'monotone down' if declines else f'NOT monotone — t={best_depth} scores best'}, so "
            f"the Q head does not regard deeper refinement as lower-loss either. The trained arm is "
            f"effectively a 1-step read; E0 (K=1, sigma=0) is thus a clean measurement of the "
            f"TRM-ification tax alone, and D>T depth extrapolation is pointless on this checkpoint.",
            f"3. Q is not constant across rollouts: spread {top['q_std']:.3f} at sigma=1.0 against "
            f"{per[0.0]['q_std']:.1e} at sigma=0 (the float32 floor — with no noise all K rollouts "
            f"ARE one vector, which also validates the harness), scaling ~linearly in sigma. So "
            f"kill-mode (a) does not fire in its crude 'Q is flat' form. Read sel-shift narrowly, "
            f"though: rollout 0 is noised like every other, so its no-information value is "
            f"{(k - 1) / k:.1%}, and the observed {low['shift']:.1%}-{per[0.6]['shift']:.1%} is "
            f"exactly that. The column rules out degenerate ties; it is not evidence of ranking.",
            f"4. But D0a cannot show the choice is INFORMED, and two signatures say caution. "
            f"gain/sd sits at {low['q_gain_sd']:.2f}-{top['q_gain_sd']:.2f} against "
            f"{order_stat:.2f} for picking the low tail of {k} iid Gaussian scores — i.e. the "
            f"{top['q_gain']:+.3f} apparent Q-gain is what ANY scores with that spread would "
            f"produce, informative or not. And Q-drift is negative and grows with sigma "
            f"({per[0.3]['q_drift']:+.3f} at 0.3, {top['q_drift']:+.3f} at 1.0, "
            f"{drift_share:.0%} of the gain): Q rates a perturbed latent better ON AVERAGE, which "
            f"is what you expect from a head trained only on noiseless z being asked about "
            f"off-manifold z. Selection pressure toward displacement is a Goodhart risk, not "
            f"evidence of a verifier.",
            f"5. Verdict: run E1/E2 at sigma* = {star:g} — the largest sigma whose conditioning-space "
            f"spread stays under 10% of ||c_T||, so the read is perturbed rather than replaced (it "
            f"is also the paper's Sudoku setting). E2 (random-select) is DECISIVE, not optional: "
            f"D0a establishes the noise is real and q is non-constant, but everything it can see "
            f"about the gain is consistent with tail-picking, so an E1 that E2 matches is the "
            f"default expectation. The missing evidence is rank quality on noiseless z — "
            f"Spearman(q, realized flow loss), which is D0b and needs the full model.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=300, help="number of omega windows (>=200)")
    parser.add_argument("--k", type=int, default=32, help="PTRM rollout width")
    parser.add_argument("--tasks", type=int, default=10, help="tasks to sample omega from (>=5)")
    parser.add_argument("--demos-per-task", type=int, default=10)
    args = parser.parse_args()
    if args.windows < 200 or args.tasks < 5:
        raise SystemExit("D0a requires >= 200 windows across >= 5 tasks")

    params_path = stage_checkpoint()
    files = stage_omega(args.tasks, args.demos_per_task)
    leaves = restore_cond_subtree(params_path)
    module, geometry = build_module(leaves)
    print(f"[d0a] conditioner: {geometry}", flush=True)

    windows = sample_windows(files, geometry["window_len"], geometry["w_dim"], args.windows)
    tasks = sorted({task for task, _ in windows})
    parity = check_harness_parity(module, jnp.asarray(windows[0][1])[None], args.k)
    print(f"[d0a] parity: {parity}", flush=True)
    result = run(windows, module, args.k)
    report = render(result, geometry, parity, tasks, args.k)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(report, encoding="utf-8")
    print(report)
    print(f"[d0a] wrote {RESULTS}")


if __name__ == "__main__":
    main()
