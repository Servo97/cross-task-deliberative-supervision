#!/usr/bin/env python3
"""§5.7 prefix-ablation diagnostic — does the GDN read the demo slots at all?

THE RISK THIS EXISTS TO CATCH. The paper's Modul cross-attends to every memory token at every
action-expert layer. Our M3 arm instead runs a gated delta-rule recurrence across the 24-slot
window and reads out at the NEWEST slot, and the demo prefix occupies the 8 OLDEST slots. A decay
that washes the prefix out makes M3 mechanically identical to M1, and the pre-registered R1 reading
(`M3 - M1`) would then say "the demonstration does not help" when the truth is "the read cannot see
it". That is a mechanism failure, not a result, and it must be separated BEFORE the 800-episode
eval is spent.

WHAT IS MEASURED. For sampled (episode, decision) pairs, the conditioning vector the action expert
actually receives, with and without the demo prefix:

    intact   c  = GDN([demo_1..8 ; live_1..16])
    clamp    c' = GDN([w_0 x8      ; live_1..16])   the REALIZABLE counterfactual: exactly what a
                                                    no-demo task already gets (rmme_demo_prefix's
                                                    clamp-to-earliest fill)
    zero     c" = GDN([0 x8        ; live_1..16])   the raw-sensitivity upper bound

and reports, per pair, `||c - c'|| / ||c||` plus the absolute delta against an fp16 floor computed
the same way `stage_e_omega_parity.fp16_floor` does (half an ulp of the intact vector, i.e. the
largest difference attributable purely to fp16 storage).

VERDICT, pre-registered:

    PASS   median relative change on the CLAMP ablation >= `--bar` (default 0.05 = 5 %) AND the
           absolute delta above the fp16 floor on >= 95 % of pairs. The prefix is read.
    FAIL   absolute delta at or below the fp16 floor on >= 5 % of pairs, or median relative change
           below the bar. The prefix is NOT read: M3 is mechanically M1, R1 is uninterpretable,
           and the pre-registered fallback arm (a learned [2, w_dim] segment embedding added to
           omega before projection, +1,024 params in the same `wsm_tanh_cond` subtree) is promoted.

CLAMP is the gating ablation because it is the intervention the arm actually faces on the 7
no-demo tasks. ZERO is reported as a sanity upper bound only: a conditioner that moves under ZERO
but not under CLAMP is telling you the demo slots carry no more information than the episode's
first frame, which is itself the finding.

MEASURED AT INITIALISATION, 2026-09-02, 96 windows off real episode grids, `gate_init=1.0`.
This is why `POS_DECAY_BIAS_INIT` exists:

    pos_decay_bias init                              clamp rel-median   above fp16 floor
    (a) ZERO  — the sealed default                        6.85e-06            0.00
    (b) -8 on the 8 DEMO slots only                       1.79e-05            0.00
    (c) -8 on all 24 slots                                4.73e-01            1.00
    (d) -4 on the 16 LIVE slots only                      1.79e-01            1.00
    (e) -4 on all 24 slots            <- pre-registered   3.82e-01            1.00

At the sealed zero-init the demo prefix moves the conditioning vector by ~7e-6 relative — below
the fp16 floor on 100 % of windows — so M3 would be MECHANICALLY M1. (b) shows the decay that
erases the prefix is the one applied at the LIVE slots, not within the prefix. The fix is a
non-zero init of an EXISTING parameter (`gamma = exp(-softplus(logit + pos_decay_bias))`, so a
negative bias starts the recurrence near-lossless and lets training learn the decay instead of
starting past it). This is measured at random init with random omega, so it bounds what the
architecture can PROPAGATE at initialisation, not what a trained model does — but the gradient to
the prefix slots carries the same product of gammas, so a model starting at 1e-5 leakage cannot
easily train its way out. The same init is applied to M1/M2/M3/M3-ctrl so the ablation stays
one-factor.

Run BEFORE any 800-episode eval, on the M3 checkpoint and its own omega store:

    PYTHONPATH=<repo> python scripts/analysis/rmme_prefix_ablation.py \\
        --omega-root .../omega/E1b-4tap --pooled-root .../wsm_pooled/rmme_pi_100k_serve \\
        --conditioner .../wsm_tanh_cond.npz --out prefix_ablation.json

`--self-test` exercises every path on real episode grids with a synthetic omega store and a
freshly-initialised conditioner; it validates the SCRIPT, never the arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from workspace_models.features.rmme_demo_prefix import (
    DEFAULT_K_DEMO,
    DEFAULT_K_LIVE,
    DEFAULT_STRIDE,
    demo_prefix_slots,
    episode_record,
    load_episode,
    serve_aligned_grid,
    serve_frame_indices,
    window_for_step,
)
from workspace_models.networks.wsm_gated_deltanet import WSMGatedDeltaNetConditioner

ABLATIONS = ("clamp", "zero")
DEFAULT_BAR = 0.05
FP16_FRACTION_BAR = 0.95
#: Pre-registered `pos_decay_bias` initialisation for every M-arm — see the table above.
POS_DECAY_BIAS_INIT = -4.0


def fp16_floor(reference: np.ndarray) -> float:
    """Half an ulp of `reference` in fp16 — copied from stage_e_omega_parity so the two agree."""
    ref = np.abs(np.asarray(reference, dtype=np.float32))
    ulp = np.where(ref > 0, np.spacing(ref.astype(np.float16)).astype(np.float32), np.float32(6e-8))
    return float(ulp.max())


def ablate(window: np.ndarray, k_demo: int, mode: str) -> np.ndarray:
    """Return a copy of `window` with the demo slots replaced."""
    out = np.array(window, dtype=np.float32, copy=True)
    if mode == "zero":
        out[:k_demo] = 0.0
    elif mode == "clamp":
        # exactly `demo_prefix_slots`' no-demo fill: every prefix slot becomes the earliest omega
        out[:k_demo] = window[k_demo]  # the oldest LIVE slot is the earliest omega in the window
    else:
        raise ValueError(f"unknown ablation {mode!r}; expected one of {ABLATIONS}")
    return out


@torch.no_grad()
def measure(conditioner, windows: np.ndarray, k_demo: int) -> dict:
    """windows [N, K, D] -> per-ablation relative/absolute change statistics."""
    windows = np.asarray(windows, dtype=np.float32)
    intact = conditioner(torch.from_numpy(windows), train=False).float().numpy()
    intact_norm = np.linalg.norm(intact, axis=-1)
    result: dict[str, dict] = {}
    for mode in ABLATIONS:
        swapped = np.stack([ablate(w, k_demo, mode) for w in windows])
        other = conditioner(torch.from_numpy(swapped), train=False).float().numpy()
        delta = np.linalg.norm(intact - other, axis=-1)
        floor = np.asarray([fp16_floor(row) for row in intact], dtype=np.float32)
        # a K-dimensional L2 delta is compared against an L2 of per-element fp16 ulps
        floor_l2 = floor * np.sqrt(intact.shape[-1])
        relative = delta / np.maximum(intact_norm, 1e-12)
        above = delta > floor_l2
        result[mode] = {
            "n": int(delta.size),
            "relative_median": float(np.median(relative)),
            "relative_mean": float(relative.mean()),
            "relative_p05": float(np.quantile(relative, 0.05)),
            "relative_p95": float(np.quantile(relative, 0.95)),
            "absolute_median": float(np.median(delta)),
            "fp16_floor_l2_median": float(np.median(floor_l2)),
            "fraction_above_fp16_floor": float(above.mean()),
        }
    result["_intact"] = {
        "cond_norm_median": float(np.median(intact_norm)),
        "cond_dim": int(intact.shape[-1]),
        "n_windows": int(windows.shape[0]),
    }
    return result


def verdict(stats: dict, bar: float) -> dict:
    clamp = stats["clamp"]
    reads_prefix = clamp["relative_median"] >= bar and clamp["fraction_above_fp16_floor"] >= FP16_FRACTION_BAR
    return {
        "verdict": "PASS" if reads_prefix else "FAIL",
        "bar_relative_median": bar,
        "bar_fraction_above_fp16_floor": FP16_FRACTION_BAR,
        "gating_ablation": "clamp",
        "reading": (
            "the GDN reads the demo prefix; R1 (M3 - M1) is interpretable"
            if reads_prefix
            else "the GDN does NOT read the demo prefix — M3 is mechanically M1. R1 is "
            "uninterpretable; promote the [2, w_dim] segment-embedding fallback arm"
        ),
    }


def load_conditioner(
    path: str | None, *, w_dim: int, cond_dim: int, window_len: int, gate_init: float, decay_bias_init: float = 0.0
) -> WSMGatedDeltaNetConditioner:
    module = WSMGatedDeltaNetConditioner(
        w_dim=w_dim, cond_dim=cond_dim, window_len=window_len, gate_init=gate_init, history_dropout=0.0
    )
    if decay_bias_init:
        with torch.no_grad():
            module.pos_decay_bias.fill_(float(decay_bias_init))
    if path:
        # A trained checkpoint supplies its OWN pos_decay_bias; the init above is overwritten,
        # which is correct — the init only ever matters at step 0.
        blob = np.load(Path(path).expanduser())
        module.load_jax_params({k: np.asarray(blob[k]) for k in blob.files})
    module.eval()
    return module


# ------------------------------------------------------------------------------------ collection
def collect_windows(
    omega_root, pooled_root, *, tasks, per_task: int, per_episode: int, k_demo: int, k_live: int, stride_steps: int
) -> tuple[np.ndarray, list]:
    from workspace_models.labels.robomme_source import episodes_of

    windows, keys = [], []
    for task in tasks:
        for ep in list(episodes_of(task))[:per_task]:
            record = load_episode(omega_root, task, ep, pooled_root=pooled_root)
            grid, start = record["frame_indices"], record["exec_start_idx"]
            if start == 0:
                continue  # a no-demo task has no prefix to ablate
            live_steps = grid[grid >= start]
            picks = live_steps[np.unique(np.linspace(0, live_steps.size - 1, per_episode).astype(int))]
            for step in picks:
                w, _seg, valid = window_for_step(
                    "m3_demo_live", record, int(step), k_demo=k_demo, k_live=k_live, stride_steps=stride_steps
                )
                if not valid.all():
                    continue
                windows.append(w)
                keys.append({"task": task, "episode": int(ep), "step": int(step)})
    if not windows:
        raise SystemExit("no demo-bearing windows collected; the ablation has nothing to measure")
    return np.stack(windows), keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omega-root", default="")
    ap.add_argument("--pooled-root", default="")
    ap.add_argument("--conditioner", default="", help="npz of the checkpoint's wsm_tanh_cond subtree (JAX layout)")
    ap.add_argument(
        "--tasks",
        default="VideoPlaceOrder,VideoPlaceButton,VideoRepick,MoveCube,"
        "InsertPeg,RouteStick,PatternLock,VideoUnmask,VideoUnmaskSwap",
    )
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--per-episode", type=int, default=6)
    ap.add_argument("--k-demo", type=int, default=DEFAULT_K_DEMO)
    ap.add_argument("--k-live", type=int, default=DEFAULT_K_LIVE)
    ap.add_argument("--stride-steps", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--cond-dim", type=int, default=1024)
    ap.add_argument("--gate-init", type=float, default=1e-3)
    ap.add_argument(
        "--decay-bias-init",
        type=float,
        default=POS_DECAY_BIAS_INIT,
        help="pos_decay_bias init used ONLY when --conditioner is absent; a trained "
        "checkpoint always supplies its own",
    )
    ap.add_argument("--bar", type=float, default=DEFAULT_BAR)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="synthetic omega on REAL episode grids; validates the script, not an arm",
    )
    ap.add_argument(
        "--dataset-root",
        default="~/.cache/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/"
        "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20",
    )
    args = ap.parse_args()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    torch.manual_seed(0)

    if args.self_test:
        import pyarrow.parquet as pq

        from workspace_models.labels.robomme_source import episode_path, episodes_of

        root = Path(args.dataset_root).expanduser()
        rng = np.random.default_rng(0)
        windows, keys = [], []
        for task in tasks[:4]:
            for ep in list(episodes_of(task))[:2]:
                table = pq.read_table(episode_path(root, ep), columns=["exec_start_idx"])
                n = table.num_rows
                raw = table.column("exec_start_idx").to_pylist()
                start = int(raw[0][0] if isinstance(raw[0], (list, tuple)) else raw[0])
                if start == 0:
                    continue
                grid = serve_aligned_grid(n, start)
                w = rng.normal(size=(grid.size, 512)).astype(np.float32)
                record = episode_record(w, grid, start, task=task, episode=ep)
                served = serve_frame_indices(start, int(np.ceil((n - start) / 16)))
                for step in served[served < n][:6]:
                    win, _s, valid = window_for_step(
                        "m3_demo_live",
                        record,
                        int(step),
                        k_demo=args.k_demo,
                        k_live=args.k_live,
                        stride_steps=args.stride_steps,
                    )
                    assert valid.all()
                    windows.append(win)
                    keys.append({"task": task, "episode": int(ep), "step": int(step)})
        windows = np.stack(windows)
        # the ablations must actually change the window, and only in the demo slots
        for mode in ABLATIONS:
            other = ablate(windows[0], args.k_demo, mode)
            assert np.array_equal(other[args.k_demo :], windows[0][args.k_demo :]), mode
            assert not np.array_equal(other[: args.k_demo], windows[0][: args.k_demo]), mode
        # clamp must reproduce exactly what a no-demo episode already receives
        demo_rows, no_demo_valid = demo_prefix_slots(np.arange(0, 400, 16), 0, args.k_demo)
        assert not no_demo_valid.any() and (demo_rows == 0).all()
        print(
            f"[ablation:self-test] {windows.shape[0]} windows from real grids, "
            f"K={windows.shape[1]} ({args.k_demo} demo + {args.k_live} live)"
        )
        # gate_init 1e-3 makes the intact vector tiny by design; run BOTH so the script is
        # exercised at a realistic gate and at one where the signal is unambiguous.
        # REGRESSION: the sealed zero-init must FAIL and the pre-registered init must PASS. If
        # this ever flips, the pos_decay_bias finding has been silently undone.
        outcomes = {}
        for label, decay in (
            ("zero-init (sealed default)", 0.0),
            (f"pos_decay_bias={POS_DECAY_BIAS_INIT}", POS_DECAY_BIAS_INIT),
        ):
            cond = load_conditioner(
                None,
                w_dim=512,
                cond_dim=args.cond_dim,
                window_len=windows.shape[1],
                gate_init=1.0,
                decay_bias_init=decay,
            )
            stats = measure(cond, windows, args.k_demo)
            v = verdict(stats, args.bar)
            outcomes[decay] = v["verdict"]
            print(
                f"  {label:<32} clamp rel-median "
                f"{stats['clamp']['relative_median']:.3e} above-floor "
                f"{stats['clamp']['fraction_above_fp16_floor']:.3f} | zero rel-median "
                f"{stats['zero']['relative_median']:.3e} -> {v['verdict']}"
            )
            assert stats["clamp"]["n"] == windows.shape[0]
            assert 0.0 <= stats["clamp"]["fraction_above_fp16_floor"] <= 1.0
        assert outcomes[0.0] == "FAIL", "the sealed zero-init no longer washes out the prefix"
        assert outcomes[POS_DECAY_BIAS_INIT] == "PASS", "the pre-registered init stopped working"
        # a conditioner that is structurally blind to the prefix must FAIL the gate
        blind = load_conditioner(None, w_dim=512, cond_dim=args.cond_dim, window_len=args.k_live, gate_init=1.0)
        live_only = windows[:, args.k_demo :]
        blind_stats = measure(blind, np.ascontiguousarray(live_only), 0)
        # a no-op ablation must be a no-op up to kernel-level float noise, and must FAIL the gate
        assert blind_stats["clamp"]["relative_median"] < 1e-6, blind_stats["clamp"]
        assert blind_stats["clamp"]["fraction_above_fp16_floor"] == 0.0
        assert verdict(blind_stats, args.bar)["verdict"] == "FAIL"
        print(
            f"  negative control: a prefix-blind read scores rel-median "
            f"{blind_stats['clamp']['relative_median']:.2e} -> FAIL, as required"
        )
        print("[ablation:self-test] PASS — the script separates a read prefix from an unread one")
        return

    if not (args.omega_root and args.pooled_root and args.conditioner):
        raise SystemExit("--omega-root, --pooled-root and --conditioner are required (or pass --self-test)")
    windows, keys = collect_windows(
        args.omega_root,
        args.pooled_root,
        tasks=tasks,
        per_task=args.per_task,
        per_episode=args.per_episode,
        k_demo=args.k_demo,
        k_live=args.k_live,
        stride_steps=args.stride_steps,
    )
    cond = load_conditioner(
        args.conditioner,
        w_dim=windows.shape[-1],
        cond_dim=args.cond_dim,
        window_len=windows.shape[1],
        gate_init=args.gate_init,
        decay_bias_init=args.decay_bias_init,
    )
    stats = measure(cond, windows, args.k_demo)
    report = {
        "kind": "rmme_prefix_ablation",
        "k_demo": args.k_demo,
        "k_live": args.k_live,
        "stride_steps": args.stride_steps,
        "tasks": list(tasks),
        "n_windows": len(keys),
        "stats": stats,
        **verdict(stats, args.bar),
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(report, indent=1) + "\n")
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
